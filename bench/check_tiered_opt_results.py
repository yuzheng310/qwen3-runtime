"""Validate raw runs before summarizing; never silently remove failed samples."""

import argparse
import json
import math
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
    if len(rows) != 24:
        raise ValueError("expected exactly 24 preregistered samples")
    for r in rows:
        c = r["effective_config"]
        arm = r["arm"]
        if (
            arm not in ["S0", "S1", "B0", "B1"]
            or r["git_commit"] != rows[0]["git_commit"]
        ):
            raise ValueError("arm/source mismatch")
        if c["cpu_kv_backend"] != ("block" if arm.startswith("B") else "snapshot"):
            raise ValueError("backend mismatch")
        if c["snapshot_mixed_restore"] != (arm == "S1") or c["cpu_kv_slab_bytes"] != (
            252 * 1024**2 if arm == "B1" else 0
        ):
            raise ValueError("treatment mismatch")
        for key, value in {
            "num_kv_blocks": 4096,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 2048,
            "cpu_kv_max_bytes": 8 * 1024**3,
            "cpu_kv_pinned_max_bytes": 4 * 1024**3,
            "transfer_chunk_bytes": 32 * 1024**2,
            "dwell": 1.0,
            "cuda_graph": True,
            "enable_prefix_cache": True,
            "session_cpu_offload": "sync",
        }.items():
            if c[key] != value:
                raise ValueError("registered config mismatch: " + key)
        if c["concurrency"] != (2 if r["scenario_id"].startswith("W0") else 8):
            raise ValueError("concurrency mismatch")
        if (
            not r.get("warmup_content_cleared")
            or r["managed_before_replay"]["cpu_committed_bytes"]
            or r["managed_before_replay"]["cpu_reserved_bytes"]
        ):
            raise ValueError("warmup content leakage")
        if r["expected_turns"] != 31 or r["expected_outputs"] != 3217:
            raise ValueError("wrong trace subset")
        if (
            not r["first_use"]
            and r["allocator_warmup_protocol"].get("prompt_tokens") != 14336
        ):
            raise ValueError("warmup mismatch")
        for v in [
            r["replay_elapsed_s"],
            r["host_warmup_s"],
            *r["transfer_counters"].values(),
            *r["metrics"]["transfer"].values(),
        ]:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError("invalid numeric metric")
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
        for arm in (
            ("S1", "B1") if scenario.startswith("W0") else ("S0", "S1", "B0", "B1")
        ):
            repetitions = sorted(
                r["repeat"]
                for r in rows
                if r["scenario_id"] == scenario and r["arm"] == arm
            )
            if repetitions != [-1, 0, 1, 2]:
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
            for arm in ["S0", "S1", "B0", "B1"]
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
        for target, baseline in [("B1", "S1"), ("S1", "S0"), ("B1", "B0")]:
            if target not in record or baseline not in record:
                continue
            key = target + "_vs_" + baseline
            record[key] = {
                "time_reduction": 1
                - record[target]["median_s"] / record[baseline]["median_s"],
                "paired_target_minus_baseline_s": [
                    a - b
                    for a, b in zip(
                        record[target]["warm_s"], record[baseline]["warm_s"]
                    )
                ],
            }
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
