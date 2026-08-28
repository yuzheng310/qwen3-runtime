#!/usr/bin/env python3
"""Offline analyzer for frozen CodeScout Trace-v1 artifacts.

No GPU. Reads public eval-rollout parquet (or compact JSONL) and writes
task_set_v1.json, summary_v1.json, and compact trace_v1.jsonl.

    uv run --with pyarrow python workloads/code_localization/analyze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


KV_BYTES_PER_TOKEN = 147_456  # Qwen3-4B GQA: 36L * 8 KV * 128 * 2 * 2
SCHEMA_VERSION = "code-localization-trace-v1"
SPLIT = "swe_bench_verified"
MODEL_CONFIG = "codescout_4b"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * p))))
    return ys[i]


def dist(xs: list[float]) -> dict[str, float | int | None]:
    n = len(xs)
    return {
        "n": n,
        "mean": (sum(xs) / n) if n else None,
        "p50": _quantile(xs, 0.50),
        "p90": _quantile(xs, 0.90),
        "p95": _quantile(xs, 0.95),
        "max": max(xs) if n else None,
    }


def _tool_names(messages: list[dict]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in _as_list(message.get("tool_calls")):
            if not call:
                continue
            fn = call.get("function") or {}
            names.append(fn.get("name") or "")
    return names


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def load_rollouts(parquet_path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()


def load_gold(parquet_path: Path) -> dict[str, dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    return {row["instance_id"]: row for row in rows}


def compact_task(row: dict, gold: dict[str, dict] | None) -> dict:
    messages = (row.get("chat_messages") or {}).get("messages") or []
    tools = (row.get("chat_messages") or {}).get("tools") or []
    metrics = row.get("metrics") or {}
    usages = _as_list(metrics.get("token_usages"))
    acc = metrics.get("accumulated_token_usage") or {}
    lats = _as_list(metrics.get("response_latencies"))
    names = _tool_names(messages)
    turns = sum(1 for m in messages if m.get("role") == "assistant")
    gold_row = (gold or {}).get(row["instance_id"], {})
    requests = []
    prev_prompt = None
    for i, usage in enumerate(usages):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        shared = min(prev_prompt, prompt) if prev_prompt is not None else 0
        requests.append(
            {
                "turn_id": i,
                "input_token_count": prompt,
                "output_token_count": completion,
                "context_length_before_generation": prompt,
                "context_growth_from_previous_turn": (
                    None if prev_prompt is None else prompt - prev_prompt
                ),
                "shared_prefix_tokens_with_previous_turn": (
                    None if prev_prompt is None else shared
                ),
                "shared_prefix_ratio_with_previous_turn": (
                    None if prev_prompt is None or prompt == 0 else shared / prompt
                ),
                "request_latency_s": (lats[i].get("latency") if i < len(lats) else None),
                "engine_model": (lats[i].get("model") if i < len(lats) else None),
                "cache_read_tokens": int(usage.get("cache_read_tokens") or 0),
            }
        )
        prev_prompt = prompt
    last_ctx = int(acc.get("per_turn_token") or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": row["instance_id"],
        "repo_id": gold_row.get("repo"),
        "base_commit": gold_row.get("base_commit"),
        "session_id": f"{MODEL_CONFIG}:{SPLIT}:{row['instance_id']}",
        "split": SPLIT,
        "model_config": MODEL_CONFIG,
        "turns": turns,
        "tool_calls": names,
        "n_terminal": names.count("terminal"),
        "n_localization_finish": names.count("localization_finish"),
        "n_messages": len(messages),
        "n_tool_schemas": len(tools),
        "prediction": row.get("prediction"),
        "messages_sha256": _stable_hash(messages),
        "tools_sha256": _stable_hash(tools),
        "input_tokens_sum": int(acc.get("prompt_tokens") or 0),
        "output_tokens_sum": int(acc.get("completion_tokens") or 0),
        "cache_read_tokens_sum": int(acc.get("cache_read_tokens") or 0),
        "peak_context_tokens": last_ctx,
        "peak_kv_bytes": last_ctx * KV_BYTES_PER_TOKEN,
        "requests": requests,
        "token_ids": "unavailable_in_public_rollouts",
    }


def summarize(tasks: list[dict]) -> dict:
    turns = [t["turns"] for t in tasks]
    tools = [len(t["tool_calls"]) for t in tasks]
    terminal = [t["n_terminal"] for t in tasks]
    finish = [t["n_localization_finish"] for t in tasks]
    req_in: list[float] = []
    req_out: list[float] = []
    ctx: list[float] = []
    growth: list[float] = []
    prefix_tok: list[float] = []
    prefix_ratio: list[float] = []
    lat: list[float] = []
    first_pref: list[float] = []
    incr: list[float] = []
    last_ctx = [t["peak_context_tokens"] for t in tasks]
    kv = [t["peak_kv_bytes"] for t in tasks]
    cache = [t["cache_read_tokens_sum"] for t in tasks]
    for task in tasks:
        for i, req in enumerate(task["requests"]):
            req_in.append(req["input_token_count"])
            req_out.append(req["output_token_count"])
            ctx.append(req["input_token_count"] + req["output_token_count"])
            if req["request_latency_s"] is not None:
                lat.append(float(req["request_latency_s"]))
            if i == 0:
                first_pref.append(req["input_token_count"])
            else:
                g = req["context_growth_from_previous_turn"] or 0
                incr.append(max(g, 0))
                growth.append(g)
                if req["shared_prefix_tokens_with_previous_turn"] is not None:
                    prefix_tok.append(req["shared_prefix_tokens_with_previous_turn"])
                if req["shared_prefix_ratio_with_previous_turn"] is not None:
                    prefix_ratio.append(req["shared_prefix_ratio_with_previous_turn"])
    sum_in = sum(req_in)
    sum_out = sum(req_out)
    sum_first = sum(first_pref)
    sum_incr = sum(incr)
    repeated = sum_in - (sum_first + sum_incr)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_tasks": len(tasks),
        "n_requests": len(req_in),
        "finish_called_once": sum(1 for x in finish if x == 1),
        "finish_missing": sum(1 for x in finish if x == 0),
        "distributions": {
            "turns_per_task": dist(turns),
            "tool_calls_per_task": dist(tools),
            "terminal_calls_per_task": dist(terminal),
            "localization_finish_per_task": dist(finish),
            "input_tokens_per_request": dist(req_in),
            "output_tokens_per_request": dist(req_out),
            "context_tokens_per_request": dist(ctx),
            "adjacent_context_growth": dist(growth),
            "adjacent_shared_prefix_tokens": dist(prefix_tok),
            "adjacent_shared_prefix_ratio": dist(prefix_ratio),
            "first_turn_prompt_tokens": dist(first_pref),
            "incremental_prompt_tokens": dist(incr),
            "peak_context_tokens_per_task": dist(last_ctx),
            "peak_kv_bytes_per_task": dist(kv),
            "recorded_request_latency_s": dist(lat),
            "cache_read_tokens_per_task": dist(cache),
        },
        "token_budget": {
            "sum_prompt_tokens_all_turns": sum_in,
            "sum_output_tokens": sum_out,
            "sum_first_turn_prompt_tokens": sum_first,
            "sum_incremental_prompt_tokens": sum_incr,
            "repeated_prefix_tokens_reprocessed": repeated,
            "repeated_prefix_fraction_of_all_prompts": repeated / sum_in if sum_in else None,
            "prompt_fraction_of_all_tokens": sum_in / (sum_in + sum_out) if (sum_in + sum_out) else None,
            "note": (
                "Public rollouts record cache_read_tokens=0. Adjacent shared-prefix "
                "uses min(prev_prompt, next_prompt)/next_prompt, i.e. a lower bound "
                "if the chat template is prefix-stable. GPU prefill/decode time is "
                "unmeasured here; token counts are the proxy."
            ),
        },
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "phase_timing": {
            "prefill_gpu_fraction": "unmeasured",
            "decode_gpu_fraction": "unmeasured",
            "runtime_host_fraction": "unmeasured",
            "inference_vs_tool_e2e": "unmeasured_in_public_rollouts",
            "recorded_eval_request_latency": "litellm_proxy/Qwen3-4B-gspo wall time, not qwen3-runtime",
        },
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=here / "raw" / "swe_bench_verified.parquet")
    parser.add_argument("--gold", type=Path, default=here / "raw" / "locagent_verified.parquet")
    parser.add_argument("--out-dir", type=Path, default=here)
    args = parser.parse_args()
    rollouts = load_rollouts(args.rollouts)
    gold = load_gold(args.gold) if args.gold.exists() else {}
    gold_ids = set(gold)
    roll_ids = {row["instance_id"] for row in rollouts}
    tasks = [compact_task(row, gold) for row in rollouts]
    tasks.sort(key=lambda t: t["task_id"])
    task_set = {
        "name": SCHEMA_VERSION,
        "description": (
            "All public CodeScout-4B SWE-Bench Verified eval rollouts. "
            "Not selected for favorable runtime. Six locagent tasks have no rollout."
        ),
        "selection": "full_public_codescout_4b_swe_bench_verified",
        "n_tasks": len(tasks),
        "split": SPLIT,
        "model_config": MODEL_CONFIG,
        "rollout_repo": "OpenHands/CodeScout_Eval_Rollouts",
        "rollout_revision": "c8150bdfcd7589cafb1717a5e9efd151a4b41b59",
        "gold_repo": "OpenHands/SWE-bench_Verified-locagent",
        "gold_revision": "ffa2f8bf98d03bc317695ad31594cd080d503a76",
        "missing_from_rollouts": sorted(gold_ids - roll_ids),
        "extra_in_rollouts": sorted(roll_ids - gold_ids),
        "tasks": [
            {
                "task_id": t["task_id"],
                "repo_id": t["repo_id"],
                "base_commit": t["base_commit"],
                "turns": t["turns"],
                "peak_context_tokens": t["peak_context_tokens"],
            }
            for t in tasks
        ],
    }
    summary = summarize(tasks)
    summary["missing_from_rollouts"] = task_set["missing_from_rollouts"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "task_set_v1.json").write_text(json.dumps(task_set, indent=2) + "\n")
    (args.out_dir / "summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_dir / "trace_v1.jsonl").open("w") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks to {args.out_dir}")
    print("repeated_prefix_fraction", summary["token_budget"]["repeated_prefix_fraction_of_all_prompts"])
    print("median turns", summary["distributions"]["turns_per_task"]["p50"])
    print("median in/out", summary["distributions"]["input_tokens_per_request"]["p50"],
          summary["distributions"]["output_tokens_per_request"]["p50"])


if __name__ == "__main__":
    main()
