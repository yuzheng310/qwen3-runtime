from copy import deepcopy
import pytest
from bench.check_tiered_opt_results import validate
from tests.cpu.test_tiered_result_validation import fixture_rows


def rows():
    base = fixture_rows()[0]
    result = []
    for scenario, arms in [("W0", ["S1", "B1"]), ("W1", ["S0", "S1", "B0", "B1"])]:
        for arm in arms:
            for repeat in [-1, 0, 1, 2]:
                r = deepcopy(base)
                r.update(
                    scenario_id=scenario,
                    arm=arm,
                    repeat=repeat,
                    first_use=repeat < 0,
                    git_commit="frozen",
                    warmup_content_cleared=True,
                    managed_before_replay={
                        "cpu_committed_bytes": 0,
                        "cpu_reserved_bytes": 0,
                    },
                    expected_turns=31,
                    expected_outputs=3217,
                    allocator_warmup_protocol={"prompt_tokens": 14336},
                    replay_elapsed_s=34.0,
                    host_warmup_s=2.0,
                )
                r["metrics"]["turns"] = [{"output_tokens": 100}] * 30 + [
                    {"output_tokens": 217}
                ]
                r["metrics"]["transfer"] = {"export_s": 0.1}
                r["effective_config"].update(
                    cpu_kv_backend="block" if arm.startswith("B") else "snapshot",
                    snapshot_mixed_restore=arm == "S1",
                    cpu_kv_slab_bytes=252 * 1024**2 if arm == "B1" else 0,
                    cuda_graph=True,
                    enable_prefix_cache=True,
                    session_cpu_offload="sync",
                    concurrency=2 if scenario == "W0" else 8,
                )
                result.append(r)
    return result


def test_registered_opt_matrix():
    assert validate(rows())


@pytest.mark.parametrize(
    "mutation",
    [
        "treatment",
        "dirty",
        "source",
        "warm_content",
        "warm_protocol",
        "extra",
        "missing",
        "nan",
        "wrong_pool",
        "leak",
        "pinned",
    ],
)
def test_reject_invalid_opt_comparison(mutation):
    rs = rows()
    r = rs[0]
    if mutation == "treatment":
        r["effective_config"]["snapshot_mixed_restore"] = False
    elif mutation == "dirty":
        r["dirty"] = True
    elif mutation == "source":
        r["git_commit"] = "other"
    elif mutation == "warm_content":
        r["managed_before_replay"]["cpu_committed_bytes"] = 1
    elif mutation == "warm_protocol":
        rs[1]["allocator_warmup_protocol"]["prompt_tokens"] = 1024
    elif mutation == "extra":
        rs.append(deepcopy(r))
    elif mutation == "missing":
        rs.pop()
    elif mutation == "nan":
        r["replay_elapsed_s"] = float("nan")
    elif mutation == "wrong_pool":
        r["effective_config"]["num_kv_blocks"] = 8192
    elif mutation == "leak":
        r["cleanup"]["requests"] = 1
    elif mutation == "pinned":
        r["metrics"]["session"]["offload_cpu_managed_pinned_peak_bytes"] = 2**33
    with pytest.raises(ValueError):
        validate(rs)
