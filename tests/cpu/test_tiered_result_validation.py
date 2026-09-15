"""Reject incomparable, incomplete, dirty, leaking or out-of-scope results."""

import pytest
from bench.check_tiered_results import validate


def fixture_rows():
    rows = []
    for scenario in ["W0", "W1"]:
        for arm in ["A", "O1", "T"]:
            for repeat in [-1, 0, 1, 2]:
                config = dict(
                    num_kv_blocks=4096,
                    cpu_kv_max_bytes=2**33,
                    cpu_kv_pinned_max_bytes=2**32,
                    transfer_chunk_bytes=2**25,
                    max_num_seqs=2,
                    attention_backend="flashinfer",
                    num_speculative_tokens=0,
                    max_num_batched_tokens=2048,
                    gpu_eviction_policy="frozen",
                    finish_notifications=True,
                    dwell=1,
                    cpu_prefix_load_for_new_requests=False,
                    host_budget_scope="managed_host_buffers",
                    group_reservation_enabled=True,
                    concurrency=2 if scenario == "W0" else 8,
                )
                rows.append(
                    dict(
                        status="complete",
                        dirty=False,
                        source_sha256="source",
                        trace_sha256="trace",
                        model_identity="weights",
                        tasks=["task"],
                        expected_turns=1,
                        expected_outputs=1,
                        effective_config=config,
                        correctness=dict(
                            completed=True,
                            forced_equal=True,
                            references_nonnegative=True,
                        ),
                        cleanup=dict(
                            requests=0,
                            free_blocks=4096,
                            offload=dict(cpu_managed_host_buffer_bytes=0),
                        ),
                        metrics=dict(
                            turns=[dict(output_tokens=1)],
                            session=dict(
                                offload_cpu_managed_host_buffer_peak_bytes=0,
                                offload_cpu_managed_pinned_peak_bytes=0,
                            ),
                        ),
                        transfer_counters=dict(gpu_staging_peak_bytes=0),
                        scenario_id=scenario,
                        arm=arm,
                        repeat=repeat,
                        first_use=repeat < 0,
                    )
                )
    return rows


def test_complete_registered_matrix_validates():
    assert validate(fixture_rows())


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty",
        "source",
        "trace",
        "model",
        "outputs",
        "budget",
        "pinned",
        "scope",
        "new_load",
        "gpu_policy",
        "concurrency",
        "finish",
        "groups",
        "spec",
        "leak",
        "negative_refs",
        "missing_memory",
        "repeat",
        "missing_sample",
        "staging",
        "failed",
    ],
)
def test_invalid_matrix_is_rejected(mutation):
    rows = fixture_rows()
    r = next(x for x in rows if x["arm"] == "O1")
    c = r["effective_config"]
    s = r["metrics"]["session"]
    if mutation == "dirty":
        r["dirty"] = True
    elif mutation == "source":
        r["source_sha256"] = "different"
    elif mutation == "trace":
        r["trace_sha256"] = "different"
    elif mutation == "model":
        r["model_identity"] = "different"
    elif mutation == "outputs":
        r["metrics"]["turns"][0]["output_tokens"] = 0
    elif mutation == "budget":
        s["offload_cpu_managed_host_buffer_peak_bytes"] = c["cpu_kv_max_bytes"] + 1
    elif mutation == "pinned":
        s["offload_cpu_managed_pinned_peak_bytes"] = c["cpu_kv_pinned_max_bytes"] + 1
    elif mutation == "scope":
        c["host_budget_scope"] = "payload"
    elif mutation == "new_load":
        c["cpu_prefix_load_for_new_requests"] = True
    elif mutation == "gpu_policy":
        c["gpu_eviction_policy"] = "different"
    elif mutation == "concurrency":
        c["concurrency"] += 1
    elif mutation == "finish":
        c["finish_notifications"] = False
    elif mutation == "groups":
        c["group_reservation_enabled"] = False
    elif mutation == "spec":
        c["num_speculative_tokens"] = 1
    elif mutation == "leak":
        r["cleanup"]["requests"] = 1
    elif mutation == "negative_refs":
        r["correctness"]["references_nonnegative"] = False
    elif mutation == "missing_memory":
        s.pop("offload_cpu_managed_host_buffer_peak_bytes")
    elif mutation == "repeat":
        r["first_use"] = False
    elif mutation == "missing_sample":
        rows.remove(r)
    elif mutation == "staging":
        r["transfer_counters"]["gpu_staging_peak_bytes"] = 2**30
    elif mutation == "failed":
        r["status"] = "failed"
    with pytest.raises(ValueError):
        validate(rows)
