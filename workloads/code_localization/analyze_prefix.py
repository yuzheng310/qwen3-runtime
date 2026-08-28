#!/usr/bin/env python3
"""Exact token-ID prefix analysis for code-localization-trace-v1.

Uses reconstructed prompt/output IDs. Does not use length-based
min(prev, next)/next. Writes prefix_analysis_v1.json (stats only).

This is Stage-2.5 measurement, not a serving optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from workloads.code_localization.analyze import dist
from workloads.code_localization.reconstruct_tokens import IM_END, NL, reconstruct_task


HERE = Path(__file__).resolve().parent


def lcp_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def classify_transition(prompt_t: list[int], completion_t: list[int], prompt_next: list[int]) -> dict[str, Any]:
    conv = prompt_t + completion_t
    lcp_prompt = lcp_len(prompt_t, prompt_next)
    lcp_session = lcp_len(conv, prompt_next)
    conv_nl = conv + [NL] if (conv and conv[-1] == IM_END) else conv
    lcp_session_nl = lcp_len(conv_nl, prompt_next)

    prompt_is_prefix = lcp_prompt == len(prompt_t) and len(prompt_t) > 0
    session_is_prefix = lcp_session == len(conv) and len(conv) > 0
    session_nl_is_prefix = lcp_session_nl == len(conv_nl) and len(conv_nl) > 0

    left_truncation = False
    if prompt_t and prompt_next and prompt_t[0] != prompt_next[0]:
        for nlen in (16, 8, 4):
            if len(prompt_next) < nlen:
                continue
            needle = prompt_next[:nlen]
            for start in range(1, max(1, len(prompt_t) - nlen + 1)):
                if prompt_t[start : start + nlen] == needle:
                    left_truncation = True
                    break
            if left_truncation:
                break

    if session_is_prefix:
        kind = "exact_append_session"
        reusable = len(conv)
        cause = "prompt_t+completion_t is an exact prefix of prompt_t+1"
    elif session_nl_is_prefix and not session_is_prefix:
        kind = "append_after_im_end_newline"
        reusable = len(conv_nl)
        cause = "assistant <|im_end|> trailing newline was stripped in reconstruction"
    elif prompt_is_prefix:
        kind = "prompt_prefix_completion_mismatch"
        reusable = len(prompt_t)
        cause = "prompt_t is an exact prefix, but reconstructed completion_t is not"
    elif left_truncation:
        kind = "left_truncation"
        reusable = lcp_prompt
        cause = "next prompt start occurs later in previous prompt (left drop)"
    elif 0 < lcp_prompt < len(prompt_t):
        kind = "internal_prefix_mutation"
        reusable = lcp_prompt
        cause = "shared header then diverge before end of prompt_t"
    elif lcp_prompt == 0:
        kind = "total_prefix_mismatch"
        reusable = 0
        cause = "no shared token prefix"
    else:
        kind = "unexplained"
        reusable = lcp_prompt
        cause = "unclassified"

    suffix = max(0, len(prompt_next) - reusable)
    return {
        "lcp_prompt_tokens": lcp_prompt,
        "lcp_session_tokens": lcp_session,
        "lcp_session_nl_tokens": lcp_session_nl,
        "lcp_prompt_ratio": _ratio(lcp_prompt, len(prompt_next)),
        "lcp_session_ratio": _ratio(lcp_session, len(prompt_next)),
        "prompt_is_exact_prefix": prompt_is_prefix,
        "session_is_exact_prefix": session_is_prefix,
        "session_nl_is_exact_prefix": session_nl_is_prefix,
        "kind": kind,
        "cause": cause,
        "reusable_prefix_tokens": reusable,
        "new_suffix_tokens": suffix,
        "reusable_ratio": _ratio(reusable, len(prompt_next)),
        "suffix_ratio": _ratio(suffix, len(prompt_next)),
        "prompt_t_tokens": len(prompt_t),
        "completion_t_tokens": len(completion_t),
        "prompt_next_tokens": len(prompt_next),
        "left_truncation": left_truncation,
    }


def analyze_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """sessions: [{task_id, requests: [{turn_id, input_ids, output_ids}]}]"""
    pairs: list[dict[str, Any]] = []
    for session in sessions:
        reqs = sorted(session["requests"], key=lambda r: r["turn_id"])
        for prev, cur in zip(reqs, reqs[1:]):
            row = classify_transition(prev["input_ids"], prev["output_ids"], cur["input_ids"])
            row["task_id"] = session["task_id"]
            row["turn_from"] = prev["turn_id"]
            row["turn_to"] = cur["turn_id"]
            pairs.append(row)

    n = len(pairs)
    kinds = Counter(p["kind"] for p in pairs)
    def pct_true(key: str) -> float | None:
        if not n:
            return None
        return sum(1 for p in pairs if p[key]) / n

    prompt_ratios = [p["lcp_prompt_ratio"] for p in pairs if p["lcp_prompt_ratio"] is not None]
    session_ratios = [p["lcp_session_ratio"] for p in pairs if p["lcp_session_ratio"] is not None]
    reusable_ratios = [p["reusable_ratio"] for p in pairs if p["reusable_ratio"] is not None]
    suffix_ratios = [p["suffix_ratio"] for p in pairs if p["suffix_ratio"] is not None]

    def ge(xs: list[float], t: float) -> float | None:
        if not xs:
            return None
        return sum(1 for x in xs if x >= t) / len(xs)

    total_prompt_next = sum(p["prompt_next_tokens"] for p in pairs)
    total_reusable = sum(p["reusable_prefix_tokens"] for p in pairs)
    total_suffix = sum(p["new_suffix_tokens"] for p in pairs)
    total_lcp_prompt = sum(p["lcp_prompt_tokens"] for p in pairs)
    total_lcp_session = sum(p["lcp_session_tokens"] for p in pairs)

    first_turn_prompt = 0
    all_prompt = 0
    for session in sessions:
        reqs = session["requests"]
        all_prompt += sum(len(r["input_ids"]) for r in reqs)
        first = next((r for r in reqs if r["turn_id"] == 0), None)
        if first is not None:
            first_turn_prompt += len(first["input_ids"])

    return {
        "n_sessions": len(sessions),
        "n_requests": sum(len(s["requests"]) for s in sessions),
        "n_adjacent_pairs": n,
        "exact_lcp_prompt_tokens": dist([p["lcp_prompt_tokens"] for p in pairs]),
        "exact_lcp_prompt_ratio": {
            **dist(prompt_ratios),
            "min": min(prompt_ratios) if prompt_ratios else None,
            "frac_ge_50": ge(prompt_ratios, 0.50),
            "frac_ge_75": ge(prompt_ratios, 0.75),
            "frac_ge_90": ge(prompt_ratios, 0.90),
            "frac_ge_95": ge(prompt_ratios, 0.95),
            "frac_ge_99": ge(prompt_ratios, 0.99),
        },
        "exact_lcp_session_tokens": dist([p["lcp_session_tokens"] for p in pairs]),
        "exact_lcp_session_ratio": {
            **dist(session_ratios),
            "min": min(session_ratios) if session_ratios else None,
        },
        "prompt_t_is_prefix_of_next": pct_true("prompt_is_exact_prefix"),
        "session_is_prefix_of_next": pct_true("session_is_exact_prefix"),
        "session_nl_is_prefix_of_next": pct_true("session_nl_is_exact_prefix"),
        "kind_counts": dict(kinds),
        "kind_fractions": {k: (v / n if n else None) for k, v in sorted(kinds.items())},
        "new_suffix_tokens": dist([p["new_suffix_tokens"] for p in pairs]),
        "reusable_prefix_tokens": dist([p["reusable_prefix_tokens"] for p in pairs]),
        "reusable_ratio": dist(reusable_ratios),
        "suffix_ratio": dist(suffix_ratios),
        "token_budget": {
            "sum_prompt_tokens_all_turns": all_prompt,
            "sum_first_turn_prompt_tokens": first_turn_prompt,
            "sum_later_prompt_tokens": total_prompt_next,
            "sum_exact_lcp_prompt_tokens": total_lcp_prompt,
            "sum_exact_lcp_session_tokens": total_lcp_session,
            "sum_reusable_prefix_tokens_strongest": total_reusable,
            "sum_new_suffix_tokens_strongest": total_suffix,
            "exact_lcp_prompt_fraction_of_later_prompts": _ratio(total_lcp_prompt, total_prompt_next),
            "exact_lcp_session_fraction_of_later_prompts": _ratio(total_lcp_session, total_prompt_next),
            "reusable_fraction_of_later_prompts": _ratio(total_reusable, total_prompt_next),
            "reusable_fraction_of_all_prompts": _ratio(total_reusable, all_prompt),
            "note": (
                "Length-based Stage-2 estimate used min(prev_len, next_len)/next_len. "
                "exact_lcp_prompt is token-identical prefix of prompt_t vs prompt_t+1. "
                "session continuation is LCP(prompt_t+completion_t, prompt_t+1). "
                "strongest reusable uses exact session prefix, else newline repair, "
                "else prompt prefix, else raw LCP."
            ),
        },
        "causes_if_session_not_exact": {
            k: kinds.get(k, 0)
            for k in (
                "append_after_im_end_newline",
                "prompt_prefix_completion_mismatch",
                "internal_prefix_mutation",
                "left_truncation",
                "total_prefix_mismatch",
                "unexplained",
            )
        },
        "example_non_session": [
            {
                "task_id": p["task_id"],
                "turn_from": p["turn_from"],
                "kind": p["kind"],
                "cause": p["cause"],
                "lcp_prompt": p["lcp_prompt_tokens"],
                "lcp_session": p["lcp_session_tokens"],
                "prompt_t": p["prompt_t_tokens"],
                "completion_t": p["completion_t_tokens"],
                "prompt_next": p["prompt_next_tokens"],
            }
            for p in pairs
            if not p["session_is_exact_prefix"]
        ][:12],
    }


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=HERE / "raw" / "swe_bench_verified.parquet")
    parser.add_argument("--tokenizer", type=Path, default=HERE / "tokenizer")
    parser.add_argument("--subset", type=Path, default=HERE / "replay_subset_v1.json")
    parser.add_argument("--out", type=Path, default=HERE / "prefix_analysis_v1.json")
    parser.add_argument("--limit", type=int, default=0)
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
    sessions = []
    for row in rows:
        report = reconstruct_task(tokenizer, row, keep_ids=True)
        sessions.append(
            {
                "task_id": report["task_id"],
                "requests": [
                    {
                        "turn_id": r["turn_id"],
                        "input_ids": r["input_ids"],
                        "output_ids": r["output_ids"],
                    }
                    for r in report["requests"]
                ],
            }
        )

    full = analyze_sessions(sessions)
    subset_ids = set(json.loads(args.subset.read_text())["task_ids"])
    subset = analyze_sessions([s for s in sessions if s["task_id"] in subset_ids])
    payload = {
        "schema_version": "code-localization-prefix-analysis-v1",
        "method": (
            "LCP on reconstructed token IDs from frozen CodeScout-4B tokenizer + "
            "HF apply_chat_template. Not min(len)/len."
        ),
        "tokenizer_files_sha256": {
            "tokenizer.json": sha256_file(args.tokenizer / "tokenizer.json"),
            "SHA256SUMS": sha256_file(HERE / "tokenizer" / "SHA256SUMS"),
        },
        "rollouts_sha256": sha256_file(args.rollouts),
        "subset_sha256": sha256_file(args.subset),
        "full_trace_v1": full,
        "replay_subset_v1": subset,
        "length_based_stage2_estimate": {
            "source": "workloads/code_localization/summary_v1.json",
            "adjacent_shared_prefix_ratio_p50": 0.8126860223507609,
            "repeated_prefix_fraction_of_all_prompts": 0.6984619883374915,
            "status": "approximate_length_stability_lower_bound",
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "full_pairs": full["n_adjacent_pairs"],
                "prompt_prefix_frac": full["prompt_t_is_prefix_of_next"],
                "session_prefix_frac": full["session_is_prefix_of_next"],
                "lcp_prompt_ratio_p50": full["exact_lcp_prompt_ratio"]["p50"],
                "kinds": full["kind_fractions"],
                "subset_pairs": subset["n_adjacent_pairs"],
                "subset_session_prefix_frac": subset["session_is_prefix_of_next"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
