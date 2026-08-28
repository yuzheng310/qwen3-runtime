#!/usr/bin/env python3
"""Reconstruct CodeScout-4B turn token IDs from public chat messages.

Recipe (frozen):
  tokenizer = OpenHands/CodeScout-4B @ eb233041350295c9ba74de5c6777e15fca0c8ddc
  apply_chat_template(prefix, tools=parquet OpenAI tools, add_generation_prompt=True)
  generated output = full[len(prompt):] minus a trailing newline after <|im_end|>

Public LiteLLM prompt_tokens are systematically shorter than this recipe (see
token_reconstruction_v1.json). Replay uses reconstructed input IDs so both
engines see the same sequence. Workload decode length is recorded
completion_tokens (validated against generated output length).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IM_END = 151645
NL = 198
TOKENIZER_REV = "OpenHands/CodeScout-4B@eb233041350295c9ba74de5c6777e15fca0c8ddc"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _content_text(message: dict) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in _as_list(content):
        if isinstance(item, dict) and item.get("text"):
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _normalize_tool_calls(message: dict) -> list[dict]:
    out: list[dict] = []
    for call in _as_list(message.get("tool_calls")):
        if not call:
            continue
        fn = call.get("function") or call
        args = fn.get("arguments")
        out.append(
            {
                "id": call.get("id"),
                "type": call.get("type") or "function",
                "function": {
                    "name": fn.get("name"),
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                },
            }
        )
    return out


def _normalize_message(message: dict) -> dict:
    role = message.get("role")
    norm: dict[str, Any] = {"role": role, "content": _content_text(message)}
    if role == "assistant":
        calls = _normalize_tool_calls(message)
        if calls:
            norm["tool_calls"] = calls
    if role == "tool" and message.get("tool_call_id"):
        norm["tool_call_id"] = message["tool_call_id"]
    return norm


def _normalize_tools(tools: list) -> list[dict]:
    return [tool for tool in _as_list(tools) if tool]


def _as_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "get") and not isinstance(value, (list, tuple, str)):
        extracted = value.get("input_ids")
        if extracted is not None:
            value = extracted
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value, list) and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def sha256_ids(ids: list[int]) -> str:
    h = hashlib.sha256()
    h.update(len(ids).to_bytes(8, "little"))
    for tok in ids:
        h.update(int(tok).to_bytes(4, "little", signed=True))
    return h.hexdigest()


def conversation_turns(messages: list[dict]) -> list[tuple[list[dict], dict]]:
    """Split the transcript into (prefix_before_assistant, assistant_message) turns."""
    turns: list[tuple[list[dict], dict]] = []
    prefix: list[dict] = []
    for message in messages:
        if message.get("role") == "assistant":
            turns.append((list(prefix), message))
        prefix.append(message)
    return turns


def strip_trailing_im_end_newline(ids: list[int]) -> list[int]:
    if len(ids) >= 2 and ids[-1] == NL and ids[-2] == IM_END:
        return ids[:-1]
    return ids


def generated_output_ids(prompt_ids: list[int], full_ids: list[int]) -> list[int] | None:
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None
    return strip_trailing_im_end_newline(full_ids[len(prompt_ids) :])


def encode_turn(tokenizer, prefix: list[dict], assistant: dict, tools: list[dict]) -> dict:
    prompt_ids = _as_ids(
        tokenizer.apply_chat_template(
            prefix,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    full_ids = _as_ids(
        tokenizer.apply_chat_template(
            prefix + [assistant],
            tools=tools or None,
            add_generation_prompt=False,
            tokenize=True,
        )
    )
    prompt_no_gen = _as_ids(
        tokenizer.apply_chat_template(
            prefix,
            tools=tools or None,
            add_generation_prompt=False,
            tokenize=True,
        )
    )
    output_ids = generated_output_ids(prompt_ids, full_ids)
    return {
        "prompt_ids": prompt_ids,
        "output_ids": output_ids,
        "prefix_stable": full_ids[: len(prompt_no_gen)] == prompt_no_gen,
        "generation_prompt_stable": full_ids[: len(prompt_ids)] == prompt_ids,
    }


def reconstruct_task(tokenizer, row: dict, *, keep_ids: bool = False) -> dict:
    chat = row.get("chat_messages") or {}
    messages = [_normalize_message(m) for m in _as_list(chat.get("messages"))]
    tools = _normalize_tools(chat.get("tools"))
    usages = _as_list((row.get("metrics") or {}).get("token_usages"))
    turns = conversation_turns(messages)
    requests = []
    for i, (prefix, assistant) in enumerate(turns):
        recorded = usages[i] if i < len(usages) else {}
        encoded = encode_turn(tokenizer, prefix, assistant, tools)
        prompt_ids = encoded["prompt_ids"]
        output_ids = encoded["output_ids"] or []
        rec_in = int(recorded.get("prompt_tokens") or 0)
        rec_out = int(recorded.get("completion_tokens") or 0)
        item = {
            "turn_id": i,
            "reconstructed_input_tokens": len(prompt_ids),
            "reconstructed_output_tokens": len(output_ids),
            "recorded_input_tokens": rec_in,
            "recorded_output_tokens": rec_out,
            "input_delta": len(prompt_ids) - rec_in,
            "input_count_match": len(prompt_ids) == rec_in,
            "output_count_match": len(output_ids) == rec_out,
            "input_sha256": sha256_ids(prompt_ids),
            "output_sha256": sha256_ids(output_ids),
            "prefix_stable": encoded["prefix_stable"],
            "generation_prompt_stable": encoded["generation_prompt_stable"],
        }
        if keep_ids:
            item["input_ids"] = prompt_ids
            item["output_ids"] = output_ids
        requests.append(item)
    return {
        "task_id": row["instance_id"],
        "n_turns": len(turns),
        "n_recorded": len(usages),
        "all_input_match": all(r["input_count_match"] for r in requests) and len(requests) == len(usages),
        "all_output_match": all(r["output_count_match"] for r in requests) and len(requests) == len(usages),
        "requests": requests,
    }


def select_stratified_task_ids(tasks: list[dict], n: int = 100) -> list[str]:
    """Deterministic context-stratified subset. Sort by peak context, take every k-th."""
    if n <= 0:
        return []
    ordered = sorted(tasks, key=lambda t: (int(t["peak_context_tokens"]), t["task_id"]))
    if n >= len(ordered):
        return [t["task_id"] for t in ordered]
    step = len(ordered) / n
    return [ordered[min(len(ordered) - 1, int(i * step))]["task_id"] for i in range(n)]


def summarize_reports(reports: list[dict]) -> dict:
    reqs = [q for r in reports for q in r["requests"]]
    in_abs = [abs(q["input_delta"]) for q in reqs]
    out_abs = [abs(q["reconstructed_output_tokens"] - q["recorded_output_tokens"]) for q in reqs]
    deltas = [q["input_delta"] for q in reqs]
    return {
        "n_tasks": len(reports),
        "tasks_all_input_match": sum(1 for r in reports if r["all_input_match"]),
        "tasks_all_output_match": sum(1 for r in reports if r["all_output_match"]),
        "n_requests": len(reqs),
        "requests_input_match": sum(1 for q in reqs if q["input_count_match"]),
        "requests_output_match": sum(1 for q in reqs if q["output_count_match"]),
        "prefix_stable_requests": sum(1 for q in reqs if q["prefix_stable"]),
        "generation_prompt_stable_requests": sum(1 for q in reqs if q["generation_prompt_stable"]),
        "input_delta": {
            "mean": (sum(deltas) / len(deltas)) if deltas else None,
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "unique": sorted(set(deltas)),
        },
        "input_abs_err": {
            "mean": (sum(in_abs) / len(in_abs)) if in_abs else None,
            "max": max(in_abs) if in_abs else None,
            "exact": sum(1 for x in in_abs if x == 0),
        },
        "output_abs_err": {
            "mean": (sum(out_abs) / len(out_abs)) if out_abs else None,
            "max": max(out_abs) if out_abs else None,
            "exact": sum(1 for x in out_abs if x == 0),
        },
        "tokenizer_revision": TOKENIZER_REV,
        "recipe": (
            "HF apply_chat_template(tools=parquet OpenAI schemas, add_generation_prompt=True); "
            "output = full[len(prompt):] minus trailing newline after <|im_end|>"
        ),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=here / "raw" / "swe_bench_verified.parquet")
    parser.add_argument("--tokenizer", type=Path, default=here / "tokenizer")
    parser.add_argument("--task-set", type=Path, default=here / "task_set_v1.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=here / "token_reconstruction_v1.json")
    parser.add_argument("--subset-out", type=Path, default=here / "replay_subset_v1.json")
    parser.add_argument("--subset-n", type=int, default=100)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    # The CLI accepts a local snapshot path and forbids network fallback.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        args.tokenizer, trust_remote_code=False, local_files_only=True
    )
    rows = pq.read_table(args.rollouts).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    reports = [reconstruct_task(tokenizer, row) for row in rows]
    summary = summarize_reports(reports)
    payload = {**summary, "tasks": reports}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    task_set = json.loads(args.task_set.read_text())
    subset_ids = select_stratified_task_ids(task_set["tasks"], args.subset_n)
    subset_set = set(subset_ids)
    subset_tasks = [t for t in task_set["tasks"] if t["task_id"] in subset_set]
    args.subset_out.write_text(
        json.dumps(
            {
                "name": "code-localization-replay-subset-v1",
                "parent": "code-localization-trace-v1",
                "selection": "stratified_by_peak_context_every_kth",
                "n_tasks": len(subset_ids),
                "task_ids": subset_ids,
                "tasks": subset_tasks,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote subset {len(subset_ids)} -> {args.subset_out}")


if __name__ == "__main__":
    main()
