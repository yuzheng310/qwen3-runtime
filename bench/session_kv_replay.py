"""Teacher-forced persistent-KV replay for CodeScout frozen sessions.

Later turns prefill only the appended suffix. Recorded output_ids are forced so
the held prompt+completion stays a prefix of the next reconstructed prompt.
Does not replace sequential full-reprefill replay.
"""

from __future__ import annotations

import time
from pathlib import Path

from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.engine.serve import RequestTrace, _record_emitted, _step, run_sequential_requests


def group_task_turns(items: list[dict]) -> list[list[dict]]:
    """Preserve dump order; keep every turn of a task together."""
    groups: list[list[dict]] = []
    index: dict[str, int] = {}
    for item in items:
        tid = item["task_id"]
        if tid not in index:
            index[tid] = len(groups)
            groups.append([])
        groups[index[tid]].append(item)
    for turns in groups:
        turns.sort(key=lambda it: it["turn_id"])
    return groups


def _used_kv_blocks(engine: Engine) -> int:
    return engine.block_manager.num_blocks - engine.block_manager.num_free_blocks


def run_session_kv_on_engine(
    engine: Engine,
    items: list[dict],
    *,
    nvtx: bool = False,
) -> tuple[list[RequestTrace], dict]:
    for item in items:
        if not item.get("output_ids"):
            raise ValueError("session-kv replay needs output_ids on every request")

    extra = {
        "execution_model": "session_kv_teacher_force",
        "first_turn_prefill_tokens": 0,
        "later_turn_prefill_tokens": 0,
        "decode_tokens": 0,
        "mixed_step_tokens": 0,
        "first_turn_prefill_s": 0.0,
        "later_turn_prefill_s": 0.0,
        "decode_s": 0.0,
        "mixed_step_s": 0.0,
        "prompt_tokens": 0,
        "appended_prompt_tokens": 0,
        "generated_tokens": 0,
        "peak_kv_blocks": 0,
        "peak_kv_blocks_per_session": 0,
        "concurrent_sessions": 1,
        "n_resumed_turns": 0,
        "n_full_add_turns": 0,
        "forced_prefix_ok": True,
        "source_forced_length_ok": True,
        "source_forced_length_mismatches": [],
        "spec_proposed": 0,
        "spec_accepted": 0,
        "spec_verify_forwards": 0,
    }
    traces: list[RequestTrace] = []
    for turns in group_task_turns(items):
        rid = None
        held: list[int] = []
        for i, item in enumerate(turns):
            last = i + 1 == len(turns)
            prompt = list(item["input_ids"])
            forced = list(item["output_ids"])
            source_max_tokens = int(item["max_tokens"])
            if source_max_tokens != len(forced):
                extra["source_forced_length_ok"] = False
                extra["source_forced_length_mismatches"].append(
                    {
                        "task_id": item["task_id"],
                        "turn_id": item["turn_id"],
                        "max_tokens": source_max_tokens,
                        "output_len": len(forced),
                    }
                )
            max_tokens = len(forced)
            extra["prompt_tokens"] += len(prompt)
            extra["generated_tokens"] += max_tokens
            more = not last
            if rid is None:
                extra["n_full_add_turns"] += 1
                extra["appended_prompt_tokens"] += len(prompt)
                rid = engine.add_request(
                    prompt,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                    hold_kv=more,
                    forced_tokens=forced,
                )
                appended = len(prompt)
            else:
                if prompt[: len(held)] != held:
                    extra["forced_prefix_ok"] = False
                    raise RuntimeError(
                        f"session prefix mismatch task={item['task_id']} turn={item['turn_id']} "
                        f"held={len(held)} prompt={len(prompt)}"
                    )
                suffix = prompt[len(held) :]
                extra["n_resumed_turns"] += 1
                extra["appended_prompt_tokens"] += len(suffix)
                appended = len(suffix)
                engine.resume_request(
                    rid,
                    suffix,
                    max_tokens,
                    hold_kv=more,
                    forced_tokens=forced,
                    ignore_eos=True,
                )
            arrival = time.perf_counter()
            trace = RequestTrace(
                request_id=rid,
                arrival_s=arrival,
                prompt_len=appended,
                max_tokens=max_tokens,
                turn_id=item["turn_id"],
            )
            req = engine._requests[rid]
            while req.status not in (
                RequestStatus.PAUSED,
                RequestStatus.FINISHED,
            ):
                step_t0 = time.perf_counter()
                _step(engine, profile=False, nvtx=nvtx)
                step_dt = time.perf_counter() - step_t0
                stats = engine.last_step_stats or {}
                prefill = int(stats.get("prefill_tokens") or 0)
                decode = int(stats.get("decode_tokens") or 0)
                if i == 0:
                    extra["first_turn_prefill_tokens"] += prefill
                else:
                    extra["later_turn_prefill_tokens"] += prefill
                extra["decode_tokens"] += decode
                if prefill and decode:
                    extra["mixed_step_tokens"] += prefill + decode
                    extra["mixed_step_s"] += step_dt
                elif prefill:
                    if i == 0:
                        extra["first_turn_prefill_s"] += step_dt
                    else:
                        extra["later_turn_prefill_s"] += step_dt
                elif decode:
                    extra["decode_s"] += step_dt
                extra["spec_proposed"] += int(stats.get("spec_proposed") or 0)
                extra["spec_accepted"] += int(stats.get("spec_accepted") or 0)
                extra["spec_verify_forwards"] += int(stats.get("spec_verify_forwards") or 0)
                now = time.perf_counter()
                _record_emitted(engine, {rid: trace}, now)
                extra["peak_kv_blocks"] = max(extra["peak_kv_blocks"], _used_kv_blocks(engine))
                extra["peak_kv_blocks_per_session"] = max(
                    extra["peak_kv_blocks_per_session"],
                    len(req.block_table),
                )
            traces.append(trace)
            held = list(req.token_ids)
            if last:
                if req.status != RequestStatus.FINISHED:
                    engine.finish_request(rid)
                rid = None
                held = []
    extra["block_size"] = engine.config.block_size
    extra["num_kv_blocks"] = engine.config.num_kv_blocks
    return traces, extra


def replay_ours_session_kv(
    items: list[dict],
    model: Path,
    *,
    nvtx: bool,
    warmup: int = 1,
    pin_path: Path,
    max_num_seqs: int = 1,
    num_speculative_tokens: int = 0,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> tuple[list[RequestTrace], float, dict]:
    from qwen3_runtime.engine.factory import build_engine
    from qwen3_runtime.engine.memory import bytes_per_kv_slot

    engine = build_engine(
        model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2048,
        pin_path=pin_path,
        num_speculative_tokens=num_speculative_tokens,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    if warmup and items:
        run_sequential_requests(
            engine,
            [(items[0]["input_ids"], min(8, max(1, len(items[0]["output_ids"]))))],
            ignore_eos=True,
            nvtx=False,
        )
    t0 = time.perf_counter()
    traces, extra = run_session_kv_on_engine(engine, items, nvtx=nvtx)
    wall = time.perf_counter() - t0
    model_cfg = getattr(getattr(engine, "runner", None), "model", None)
    if model_cfg is not None:
        slot_bytes = bytes_per_kv_slot(model_cfg.cfg, dtype_bytes=2)
    else:
        slot_bytes = 0
    extra["kv_slot_bytes"] = slot_bytes
    extra["peak_kv_bytes"] = extra["peak_kv_blocks"] * extra["block_size"] * slot_bytes
    extra["peak_kv_bytes_per_session"] = (
        extra["peak_kv_blocks_per_session"] * extra["block_size"] * slot_bytes
    )
    extra["num_speculative_tokens"] = engine.config.num_speculative_tokens
    extra["ngram_min"] = engine.config.ngram_min
    extra["ngram_max"] = engine.config.ngram_max
    return traces, wall, extra
