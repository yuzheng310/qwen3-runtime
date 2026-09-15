"""Validate raw runs before summarizing; never silently remove failed samples."""

import argparse
import json
import statistics
from pathlib import Path


def validate(rows):
    if not rows:
        raise ValueError("no samples")
    identities = [
        "source_sha256",
        "trace_sha256",
        "model_identity",
        "tasks",
        "expected_turns",
        "expected_outputs",
    ]
    for r in rows:
        if r.get("status") != "complete" or r.get("dirty"):
            raise ValueError("failed or dirty sample")
        for key in identities:
            if r.get(key) != rows[0].get(key) or r.get(key) is None:
                raise ValueError("incompatible " + key)
        c = r["effective_config"]
        if c.get("cpu_prefix_load_for_new_requests") is not False:
            raise ValueError("new request CPU load")
        if c.get("host_budget_scope") != "managed_host_buffers":
            raise ValueError("budget scope")
        for key in [
            "num_kv_blocks",
            "cpu_kv_max_bytes",
            "cpu_kv_pinned_max_bytes",
            "transfer_chunk_bytes",
            "max_num_seqs",
            "attention_backend",
            "num_speculative_tokens",
            "max_num_batched_tokens",
            "gpu_eviction_policy",
            "finish_notifications",
            "dwell",
        ]:
            if c.get(key) != rows[0]["effective_config"].get(key) or c.get(key) is None:
                raise ValueError("incompatible config " + key)
        if not r["correctness"]["completed"] or not r["correctness"]["forced_equal"]:
            raise ValueError("incomplete outputs")
        if (
            r["cleanup"]["requests"]
            or r["cleanup"]["free_blocks"] != c["num_kv_blocks"]
        ):
            raise ValueError("cleanup leak")
        metrics = r["metrics"]
        session = metrics["session"]
        if len(metrics["turns"]) != r["expected_turns"]:
            raise ValueError("turn mismatch")
        if sum(x["output_tokens"] for x in metrics["turns"]) != r["expected_outputs"]:
            raise ValueError("output mismatch")
        if (
            session.get("offload_cpu_managed_host_buffer_peak_bytes", 0)
            > c["cpu_kv_max_bytes"]
        ):
            raise ValueError("host budget exceeded")
        if (
            session.get("offload_cpu_managed_pinned_peak_bytes", 0)
            > c["cpu_kv_pinned_max_bytes"]
        ):
            raise ValueError("pinned budget exceeded")
        if session.get("offload_new_request_cpu_loads", 0):
            raise ValueError("out of scope CPU import")
        if not r["correctness"]["references_nonnegative"]:
            raise ValueError("negative references")
    for r in rows:
        if (
            type(r["repeat"]) is not int
            or r["repeat"] < -1
            or r["first_use"] != (r["repeat"] == -1)
        ):
            raise ValueError("inconsistent first-use/repeat identity")
        c = r["effective_config"]
        if not c.get("group_reservation_enabled") or not c.get("finish_notifications"):
            raise ValueError("reclamation or completion protocol changed")
        if c.get("num_speculative_tokens") != 0:
            raise ValueError("speculative execution outside registered scenario")
        peers = [x for x in rows if x["scenario_id"] == r["scenario_id"]]
        if any(
            x["effective_config"].get("concurrency") != c.get("concurrency")
            for x in peers
        ):
            raise ValueError("concurrency changed within scenario")
        session = r["metrics"]["session"]
        if r["arm"] != "A":
            for field in (
                "offload_cpu_managed_host_buffer_peak_bytes",
                "offload_cpu_managed_pinned_peak_bytes",
            ):
                if field not in session or session[field] < 0:
                    raise ValueError("missing/negative managed memory metric")
            if r["cleanup"]["offload"].get("cpu_managed_host_buffer_bytes") != 0:
                raise ValueError("managed storage remains after explicit release")
        if r["transfer_counters"].get("gpu_staging_peak_bytes", 0) > max(
            c["transfer_chunk_bytes"], 2 * 36 * 16 * 8 * 128 * 2
        ):
            raise ValueError("GPU staging exceeded registered bound")
    scenarios = {r["scenario_id"] for r in rows}
    if len(scenarios) != 2:
        raise ValueError("expected complete W0 and W1 matrix")
    for scenario in scenarios:
        for arm in ("A", "O1", "T"):
            repetitions = sorted(
                r["repeat"]
                for r in rows
                if r["scenario_id"] == scenario and r["arm"] == arm
            )
            if repetitions not in ([-1, 0, 1, 2], [-1, 0, 1, 2, 3, 4, 5, 6]):
                raise ValueError("missing arm/repeat or unregistered repetition count")
    seen = set()
    for r in rows:
        k = (r["scenario_id"], r["arm"], r["repeat"])
        if k in seen:
            raise ValueError("duplicate sample")
        seen.add(k)
    return True


def summarize(rows):
    validate(rows)
    out = {}
    for scenario in sorted({r["scenario_id"] for r in rows}):
        groups = {
            arm: sorted(
                [r for r in rows if r["scenario_id"] == scenario and r["arm"] == arm],
                key=lambda r: r["repeat"],
            )
            for arm in ["A", "O1", "T"]
        }
        record = {}
        for arm, rs in groups.items():
            warm = [r for r in rs if not r["first_use"]]
            if not warm:
                continue
            record[arm] = {
                "first_use_s": [r["replay_elapsed_s"] for r in rs if r["first_use"]],
                "warm_s": [r["replay_elapsed_s"] for r in warm],
                "median_s": statistics.median(r["replay_elapsed_s"] for r in warm),
                "median_d2h_bytes": statistics.median(
                    r["transfer_counters"]["d2h_completed_bytes"] for r in warm
                ),
                "median_h2d_bytes": statistics.median(
                    r["transfer_counters"]["h2d_completed_bytes"] for r in warm
                ),
                "median_rss_peak_bytes": statistics.median(
                    r["memory_observability"]["replay_sampled_rss_peak_bytes"]
                    for r in warm
                ),
            }
        if "T" in record and "O1" in record:
            record["time_reduction_T_vs_O1"] = (
                1 - record["T"]["median_s"] / record["O1"]["median_s"]
            )
            o = {
                r["repeat"]: r["replay_elapsed_s"]
                for r in groups["O1"]
                if not r["first_use"]
            }
            record["paired_T_minus_O1_s"] = [
                r["replay_elapsed_s"] - o[r["repeat"]]
                for r in groups["T"]
                if not r["first_use"] and r["repeat"] in o
            ]
        out[scenario] = record
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = [
        json.loads(p.read_text()) for p in sorted(Path(a.directory).glob("W*.json"))
    ]
    result = {"valid": validate(rows), "samples": len(rows), "summary": summarize(rows)}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
