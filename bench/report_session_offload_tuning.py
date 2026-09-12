"""Summarize completed trials without selecting fast repeats or mixing arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    args = data["args"]
    expected = args["arms"].split(",")
    repeats = args["repeats"]
    measured = [r for r in data["rows"] if r["repeat"] >= 0]
    complete = not data.get("failures") and repeats > 0
    arms = {}
    for arm in expected:
        rows = [r for r in measured if r["arm"] == arm]
        if sorted(r["repeat"] for r in rows) != list(range(repeats)):
            complete = False
        if not rows:
            continue
        times = [r["elapsed_s"] for r in rows]
        median = statistics.median(times)
        arms[arm] = {
            "n": len(rows),
            "median_s": median,
            "min_s": min(times),
            "max_s": max(times),
            "range_over_median_pct": (
                (max(times) - min(times)) / median * 100 if len(times) > 1 else None
            ),
            "prefill_tokens_median": statistics.median(
                r["work"]["prefill_tokens"] for r in rows
            ),
            "preemptions_median": statistics.median(
                r["work"]["preemptions"] for r in rows
            ),
            "saved_median": statistics.median(
                r["session"]["offload_saved"] for r in rows
            ),
            "restored_median": statistics.median(
                r["session"]["offload_restored"] for r in rows
            ),
            "cpu_peak_bytes_max": max(r["work"]["cpu_peak_bytes"] for r in rows),
            "transfer_s_median": statistics.median(
                r["transfer"]["save_s"] + r["transfer"]["restore_s"] for r in rows
            ),
            "all_forced_equal": all(r["forced_equal"] for r in rows),
            "output_tokens_per_s_median": statistics.median(
                sum(t["output_tokens"] for t in r["turns"]) / r["elapsed_s"]
                for r in rows
            ),
            "ttft_p50_s_median": statistics.median(
                statistics.median(t["ttft_s"] for t in r["turns"]) for r in rows
            ),
            "ttft_p95_s_median": statistics.median(
                statistics.quantiles([t["ttft_s"] for t in r["turns"]], n=20, method="inclusive")[18]
                if len(r["turns"]) > 1 else r["turns"][0]["ttft_s"] for r in rows
            ),
            "useful_restores_median": (
                statistics.median(r["session"]["offload_useful_restores"] for r in rows)
                if all("offload_useful_restores" in r["session"] for r in rows) else None
            ),
            "unused_d2h_bytes_after_cleanup_median": (
                statistics.median(r["offload_after_cleanup"]["unused_d2h_bytes"] for r in rows)
                if all("unused_d2h_bytes" in r.get("offload_after_cleanup", {}) for r in rows)
                else None
            ),
            "gpu_peak_allocated_bytes_max": (
                max(r["gpu_peak_allocated_bytes"] for r in rows)
                if all("gpu_peak_allocated_bytes" in r for r in rows) else None
            ),
            "warmup_s": [
                r["elapsed_s"]
                for r in data["rows"]
                if r["arm"] == arm and r["repeat"] == -1
            ],
        }
        if not arms[arm]["all_forced_equal"]:
            complete = False
    result = {
        "file": str(path),
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "complete": complete,
        "args": args,
        "resolved_num_kv_blocks": data.get("resolved_num_kv_blocks", args["blocks"]),
        "source_sha256": data.get("source_sha256"),
        "arms": arms,
    }
    controls = {a: r for a, r in arms.items() if a != "O-sync"}
    if complete and controls and "O-sync" in arms:
        best = min(controls, key=lambda a: controls[a]["median_s"])
        baseline = controls[best]["median_s"]
        result["fastest_control"] = best
        result["offload_time_reduction_pct"] = (
            (baseline - arms["O-sync"]["median_s"]) / baseline * 100
        )
        base_work = arms[best]["prefill_tokens_median"]
        result["offload_prefill_reduction_pct"] = (
            (base_work - arms["O-sync"]["prefill_tokens_median"]) / base_work * 100
            if base_work else None
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite summary")
    rows = [summarize(Path(p)) for p in args.paths]
    out.write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(
            json.dumps(
                {
                    "file": r["file"],
                    "complete": r["complete"],
                    "arms": {
                        a: {
                            k: v[k]
                            for k in [
                                "n",
                                "median_s",
                                "range_over_median_pct",
                                "prefill_tokens_median",
                            ]
                        }
                        for a, v in r["arms"].items()
                    },
                    "time_reduction_pct": r.get("offload_time_reduction_pct"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
