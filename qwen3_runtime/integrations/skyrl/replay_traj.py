"""Rebuild SkyRL GeneratorOutput from CodeScout traj JSON (TokenEvents).

Used so Task B can train from a completed generate without cloning 64 repos again.
Attempt 9 ours trajs cannot use this path: they had no TokenEvents.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any


def index_traj_dir(traj_dir: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (instance_id, repetition_id) to the saved JSON record."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    pattern = os.path.join(traj_dir, "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        rec = json.load(open(path))
        instance_id = str(rec.get("instance_id") or "")
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = instance_id + "_"
        rep = stem[len(prefix) :] if instance_id and stem.startswith(prefix) else stem.rsplit("_", 1)[-1]
        out[(instance_id, str(rep))] = rec
    return out


def _loss_mask(response_ids: list[int], tokenizer: Any, model_name: str) -> list[int]:
    buffer_succeed = 5
    if "Qwen3-4B-Instruct-2507" in (model_name or ""):
        buffer_succeed = 1
    buffer_precede = 1
    start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_token_id = tokenizer.convert_tokens_to_ids("assistant")
    mask: list[int] = []
    inside = False
    buffer = 0
    found_role_switch = False
    for token_id in response_ids:
        if token_id == start_token_id:
            inside = True
            for _ in range(buffer_precede):
                if mask:
                    mask.pop()
            mask.extend([0] * buffer_precede)
            mask.append(0)
        elif token_id == end_token_id and found_role_switch:
            inside = False
            mask.append(0)
            buffer = buffer_succeed
        else:
            if inside:
                mask.append(0)
            elif buffer:
                mask.append(0)
                buffer -= 1
            else:
                mask.append(1)
        found_role_switch = token_id == start_token_id
    return mask


def _exhausted(rec: dict[str, Any], max_turns: int = 6) -> bool:
    """Proxy for CodeScout: no localization_finish and TokenEvents >= max_turns."""
    msgs = rec.get("messages") or []
    n_te = sum(1 for m in msgs if m.get("kind") == "TokenEvent")
    finished = any(
        m.get("kind") == "ActionEvent" and m.get("tool_name") == "localization_finish"
        for m in msgs
    )
    return (not finished) and n_te >= max_turns


def rollout_from_record(
    rec: dict[str, Any],
    *,
    tokenizer: Any,
    model_name: str,
    step_wise: bool = False,
) -> list[tuple]:
    """Match CodeSearchGenerator.code_search_loop's TokenEvent packing (step_wise off)."""
    messages = rec.get("messages") or []
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    reward = float(rec.get("total_reward") or 0.0)
    metrics = rec.get("metrics_dict") or {}
    if not token_messages:
        return [([151643], reward, "error", [0], [151643], None, metrics)]
    if step_wise:
        out = []
        for message in token_messages:
            resp = list(message["response_token_ids"])
            prompt = list(message["prompt_token_ids"])
            out.append((resp, reward, "complete", [1] * len(resp), prompt, None, metrics))
        return out
    current_prompt_ids = list(token_messages[0]["prompt_token_ids"])
    ending_prompt_ids = list(token_messages[-1]["prompt_token_ids"])
    ending_response_ids = list(token_messages[-1]["response_token_ids"])
    current_response_ids = (ending_prompt_ids + ending_response_ids)[len(current_prompt_ids) :]
    mask = _loss_mask(current_response_ids, tokenizer, model_name)
    if _exhausted(rec):
        mask = [0] * len(mask)
    return [
        (
            current_response_ids,
            reward,
            "complete",
            mask,
            current_prompt_ids,
            None,
            metrics,
        )
    ]


def logprobs_for_rows(
    rec: dict[str, Any],
    rows: list[tuple],
    sidecar: dict[tuple[int, ...], list[list[float]]],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Lay sampling-time logprobs onto rows already packed by `rollout_from_record`.

    Kept out of the packing function because packing is about token layout and
    this is about a second tensor that only §5.4 needs.
    """
    from qwen3_runtime.rollout.logprobs import mask_coverage, place_turn_logprobs, turns_from_record

    turns, missing = turns_from_record(rec, sidecar)
    out: list[list[float]] = []
    agg = {"turns_placed": 0, "turns_unplaced": 0, "tokens_placed": 0, "tokens_unplaced": 0}
    cover = {"trained_positions": 0, "trained_with_rollout_logprob": 0, "trained_without_rollout_logprob": 0}

    # A real logprob is often exactly 0.0 -- more than half of them are, because
    # tool-call scaffolding is near-deterministic -- so which positions carry a
    # sampled value cannot be recovered from the values. Record it explicitly.
    covered_idx: list[list[int]] = []

    # step_wise emits one row per turn; otherwise the whole trajectory is one row.
    per_row_turns = [[t] for t in turns] if len(rows) == len(turns) and len(rows) > 1 else [turns]
    for row, row_turns in zip(rows, per_row_turns + [[]] * len(rows)):
        response_ids, mask = row[0], row[3]
        logprobs, stats = place_turn_logprobs(response_ids, row_turns)
        out.append(logprobs)
        covered_idx.append([i for i, c in enumerate(stats["covered"]) if c])
        for key in agg:
            agg[key] += stats[key]
        for key, val in mask_coverage(mask, stats["covered"]).items():
            cover[key] += val

    return out, {
        "instance_id": rec.get("instance_id"),
        "turns_without_sidecar": missing,
        **agg,
        **cover,
        "covered_idx": covered_idx,
    }


def generator_output_from_index(
    generator: Any,
    input_batch: dict[str, Any],
    indexed: dict[tuple[str, str], dict[str, Any]],
    sidecar: dict[tuple[int, ...], list[list[float]]] | None = None,
    coverage_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import copy

    from skyrl_train.generators.utils import get_rollout_metrics

    env_extras = input_batch["env_extras"]
    trajectory_ids = input_batch["trajectory_ids"]
    all_outputs = []
    all_logprobs: list[list[list[float]]] = []
    rewards_dict = []
    metrics_dict = []
    for extra, tid in zip(env_extras, trajectory_ids):
        instance_id = str(extra["instance_id"])
        rep = str(tid.repetition_id)
        rec = indexed.get((instance_id, rep))
        if rec is None:
            raise FileNotFoundError(f"no traj JSON for {instance_id}_{rep}")
        rows = rollout_from_record(
            rec,
            tokenizer=generator.tokenizer,
            model_name=str(getattr(generator, "model_name", "")),
            step_wise=bool(getattr(generator, "step_wise", False)),
        )
        all_outputs.append(rows)
        rewards_dict.append(rec.get("reward_dict") or {})
        metrics_dict.append(rec.get("metrics_dict") or {})
        if sidecar is not None:
            logprobs, coverage = logprobs_for_rows(rec, rows, sidecar)
            all_logprobs.append(logprobs)
            if coverage_out is not None:
                coverage_out.append(coverage)

    responses = sum([[output[0] for output in step_outputs] for step_outputs in all_outputs], [])
    rewards = sum([[output[1] for output in step_outputs] for step_outputs in all_outputs], [])
    stop_reasons = sum([[output[2] for output in step_outputs] for step_outputs in all_outputs], [])
    loss_masks = sum([[output[3] for output in step_outputs] for step_outputs in all_outputs], [])
    # SkyRL asserts rollout_logprobs.shape == loss_mask.shape, so this is either
    # fully populated or absent -- a partial list would fail that assert deeper in.
    rollout_logprobs = sum(all_logprobs, []) if sidecar is not None else None
    prompt_token_ids = sum([[output[4] for output in step_outputs] for step_outputs in all_outputs], [])
    out_trajectory_ids = []
    is_last_step = []
    for i, step_outputs in enumerate(all_outputs):
        for step_id in range(len(step_outputs)):
            out_trajectory_id = copy.deepcopy(trajectory_ids[i])
            out_trajectory_id.step = step_id
            out_trajectory_ids.append(out_trajectory_id.instance_id)
            is_last_step.append(step_id == len(step_outputs) - 1)
    tracked_metrics: dict[str, Any] = {}
    for tracker_name, tracker_dict in zip(["reward", "metrics"], [rewards_dict, metrics_dict]):
        for tracker_dict_item in tracker_dict:
            for key, val in tracker_dict_item.items():
                if not isinstance(val, (int, float)):
                    continue
                tracked_metrics.setdefault(f"{tracker_name}/{key}", []).append(val)
    for key, val in tracked_metrics.items():
        tracked_metrics[key] = sum(val) / len(val)
    return {
        "trajectory_ids": out_trajectory_ids,
        "prompt_token_ids": prompt_token_ids,
        "response_ids": responses,
        "rewards": rewards,
        "loss_masks": loss_masks,
        "stop_reasons": stop_reasons,
        "rollout_metrics": get_rollout_metrics(responses, rewards),
        "rollout_logprobs": rollout_logprobs,
        "is_last_step": is_last_step,
        **tracked_metrics,
    }


def patch_codescout_replay_traj(traj_dir: str | None = None) -> str:
    """Replace CodeSearchGenerator.generate with a disk replay. Empty dir = no-op."""
    if traj_dir is None:
        traj_dir = os.environ.get("QWEN3_REPLAY_TRAJ_DIR", "")
    traj_dir = (traj_dir or "").rstrip("/")
    if not traj_dir:
        return ""
    from src.generator.code_search_generator import CodeSearchGenerator

    indexed = index_traj_dir(traj_dir)
    orig = CodeSearchGenerator.generate
    if getattr(orig, "_qwen3_replay", False):
        return traj_dir

    # Replaying the run that produced the sidecar is what makes §5.4 cheap: the
    # sampled tokens are already fixed, so only the trainer's forward pass has
    # to run again.
    sidecar_path = os.environ.get("QWEN3_REPLAY_LOGPROB_SIDECAR", "")
    sidecar = None
    if sidecar_path:
        from qwen3_runtime.rollout.logprobs import load_sidecar

        sidecar = load_sidecar(sidecar_path)
        print(f"[qwen3] sidecar {len(sidecar)} distinct completions from {sidecar_path}", flush=True)
    coverage_path = os.environ.get("QWEN3_REPLAY_COVERAGE_OUT", "")

    async def generate(self, input_batch):
        print(f"[qwen3] replaying {len(indexed)} trajs from {traj_dir}", flush=True)
        coverage: list[dict[str, Any]] = [] if coverage_path else None  # type: ignore[assignment]
        out = generator_output_from_index(self, input_batch, indexed, sidecar, coverage)
        if coverage_path and coverage is not None:
            with open(coverage_path, "w") as fh:
                json.dump(
                    {
                        "schema_version": 1,
                        "sidecar": sidecar_path,
                        "traj_dir": traj_dir,
                        "per_trajectory": coverage,
                        "totals": {
                            key: sum(int(row.get(key, 0)) for row in coverage)
                            for key in (
                                "turns_placed",
                                "turns_unplaced",
                                "tokens_placed",
                                "tokens_unplaced",
                                "turns_without_sidecar",
                                "trained_positions",
                                "trained_with_rollout_logprob",
                                "trained_without_rollout_logprob",
                            )
                        },
                    },
                    fh,
                    indent=1,
                )
            print(f"[qwen3] wrote replay coverage to {coverage_path}", flush=True)
        return out

    generate._qwen3_replay = True  # type: ignore[attr-defined]
    generate._qwen3_replay_dir = traj_dir  # type: ignore[attr-defined]
    CodeSearchGenerator.generate = generate
    return traj_dir
