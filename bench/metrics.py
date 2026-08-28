"""TTFT / TPOT / ITL / throughput aggregation. No invented numbers."""

from __future__ import annotations

from statistics import mean, median, pstdev

from qwen3_runtime.engine.serve import RequestTrace


def _mmm(xs: list[float]) -> dict:
    if not xs:
        return {
            "median": None,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "spread_flag": False,
        }
    med = float(median(xs))
    lo, hi = min(xs), max(xs)
    flag = bool(med > 0 and (hi - lo) / med > 0.03)
    return {
        "median": med,
        "mean": float(mean(xs)),
        "stdev": float(pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": lo,
        "max": hi,
        "spread_flag": flag,
    }


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    i = int(k)
    f = k - i
    if i + 1 < len(ys):
        return ys[i] * (1 - f) + ys[i + 1] * f
    return ys[i]


def compact_traces(traces: list[RequestTrace]) -> list[dict]:
    """Per-request TTFT dump. Capacity JSON used to drop this, which hid turn splits."""
    rows = []
    for t in traces:
        rows.append(
            {
                "request_id": t.request_id,
                "turn_id": t.turn_id,
                "prompt_len": t.prompt_len,
                "max_tokens": t.max_tokens,
                "admitted": t.admitted,
                "n_tokens": len(t.tokens),
                "ttft_s": t.ttft_s,
                "tpot_s": t.tpot_s,
                "queue_s": t.queue_s,
                "prefill_s": t.prefill_s,
                "cached_tokens": t.cached_tokens,
            }
        )
    return rows


def summarize_traces(traces: list[RequestTrace], wall_s: float) -> dict:
    admitted = [t for t in traces if t.admitted]
    forced = all(len(t.tokens) == t.max_tokens for t in admitted)
    ttfts = [t.ttft_s for t in admitted if t.ttft_s is not None]
    tpots = [t.tpot_s for t in admitted if t.tpot_s is not None]
    itls = [x for t in admitted for x in t.itl_s]
    n_out = sum(len(t.tokens) for t in admitted)
    return {
        "n_requests": len(traces),
        "n_admitted": len(admitted),
        "forced_length_ok": forced,
        "ttft_s": _mmm(ttfts),
        "tpot_s": _mmm(tpots),
        "itl_s_p50": percentile(itls, 0.50),
        "itl_s_p99": percentile(itls, 0.99),
        "output_tok_s": n_out / wall_s if wall_s > 0 else 0.0,
        "request_per_s": len(admitted) / wall_s if wall_s > 0 else 0.0,
        "wall_s": wall_s,
        "output_lens": [len(t.tokens) for t in admitted],
    }


def length_split_metrics(traces: list[RequestTrace]) -> dict:
    """Split TTFT/TPOT/queue by prompt length. Bimodal mixes use min vs max length."""
    admitted = [t for t in traces if t.admitted]
    by_len: dict[int, list[RequestTrace]] = {}
    for t in admitted:
        by_len.setdefault(t.prompt_len, []).append(t)

    def _group(prefix: str, group: list[RequestTrace]) -> dict:
        ttfts = [t.ttft_s for t in group if t.ttft_s is not None]
        tpots = [t.tpot_s for t in group if t.tpot_s is not None]
        queues = [t.queue_s for t in group if t.queue_s is not None]
        return {
            f"{prefix}_n": len(group),
            f"{prefix}_completed": sum(1 for t in group if len(t.tokens) == t.max_tokens),
            f"{prefix}_ttft_s_p50": percentile(ttfts, 0.50),
            f"{prefix}_ttft_s_p95": percentile(ttfts, 0.95),
            f"{prefix}_ttft_s_p99": percentile(ttfts, 0.99),
            f"{prefix}_tpot_s_p50": percentile(tpots, 0.50),
            f"{prefix}_tpot_s_p95": percentile(tpots, 0.95),
            f"{prefix}_tpot_s_p99": percentile(tpots, 0.99),
            f"{prefix}_queue_s_p50": percentile(queues, 0.50),
            f"{prefix}_queue_s_p95": percentile(queues, 0.95),
            f"{prefix}_queue_s_p99": percentile(queues, 0.99),
            f"{prefix}_max_queue_s": max(queues) if queues else None,
        }

    out: dict = {
        "prompt_len_histogram": {str(k): len(v) for k, v in sorted(by_len.items())},
        "max_queue_s": max((t.queue_s for t in admitted if t.queue_s is not None), default=None),
    }
    if len(by_len) >= 2:
        short_len = min(by_len)
        long_len = max(by_len)
        out["short_prompt_len"] = short_len
        out["long_prompt_len"] = long_len
        out.update(_group("short", by_len[short_len]))
        out.update(_group("long", by_len[long_len]))
        if len(by_len) > 2:
            other = [
                t
                for length, group in by_len.items()
                if length not in (short_len, long_len)
                for t in group
            ]
            out.update(_group("other", other))
    return out


def merge_trials(trial_metrics: list[dict]) -> dict:
    def col(key: str, inner: str | None = None) -> list[float]:
        out: list[float] = []
        for m in trial_metrics:
            val = m[key]
            if inner is None:
                if isinstance(val, (int, float)):
                    out.append(float(val))
            elif isinstance(val, dict) and val.get(inner) is not None:
                out.append(float(val[inner]))
        return out

    forced = all(m.get("forced_length_ok") is True for m in trial_metrics)
    ttft_meds = col("ttft_s", "median")
    tpot_meds = col("tpot_s", "median")
    return {
        "forced_length_ok": forced,
        "ttft_s": _mmm(ttft_meds),
        "tpot_s": _mmm(tpot_meds),
        "output_tok_s": _mmm(col("output_tok_s")),
        "request_per_s": _mmm(col("request_per_s")),
        "itl_s_p50": _mmm([m["itl_s_p50"] for m in trial_metrics if m.get("itl_s_p50") is not None]),
        "itl_s_p99": _mmm([m["itl_s_p99"] for m in trial_metrics if m.get("itl_s_p99") is not None]),
        "trials": trial_metrics,
    }
