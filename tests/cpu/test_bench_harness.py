from pathlib import Path
import json

from bench.drivers import vllm_first_token_s
from bench.env import git_dirty
from bench.metrics import _mmm, length_split_metrics
from bench.report import check_dir
from bench.run_bench import main as run_bench_main
from bench.workloads import FULL, PARITY_CASES, token_budget_for_case


def test_vllm_engine_kwargs_disable_prefix_cache():
    from bench.drivers import vllm_engine_kwargs

    kw = vllm_engine_kwargs("/tmp/model")
    assert kw["gpu_memory_utilization"] == 0.9
    assert kw["enable_prefix_caching"] is False
    assert kw["max_model_len"] == 16384
    assert "max_num_batched_tokens" not in kw
    longctx = vllm_engine_kwargs("/tmp/model", case="longctx")
    assert longctx["max_num_batched_tokens"] == 2048
    assert longctx["enable_prefix_caching"] is False
    assert vllm_first_token_s(None) is None

    class _Empty:
        pass

    assert vllm_first_token_s(_Empty()) is None

    class _Hit:
        first_token_time = 1.25

    assert vllm_first_token_s(_Hit()) == 1.25


def test_run_bench_tiny_cpu_json_has_required_fields(tmp_path):
    out = tmp_path / "qwen3-runtime_tiny_cpu.json"
    assert run_bench_main(["--case", "tiny_cpu", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["case"] == "tiny_cpu"
    assert data["workload"]["supplementary"] is True
    assert data["forced_length_ok"] is True
    assert data["metrics"]["output_tokens"]
    assert "git_commit" in data["environment"]
    assert "dirty" in data["environment"]
    check_dir(tmp_path)


def test_run_bench_full_scale_requires_model():
    try:
        run_bench_main(["--case", "latency", "--scale", "full", "--out", "unused.json"])
    except SystemExit as exc:
        assert "model" in str(exc).lower()
    else:
        raise AssertionError("expected refusal")


def test_run_bench_full_scale_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("bench.env.git_dirty", lambda repo: True)
    try:
        run_bench_main(
            [
                "--case",
                "latency",
                "--scale",
                "full",
                "--model",
                "unused",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
    except SystemExit as exc:
        assert "dirty" in str(exc).lower()
    else:
        raise AssertionError("expected refusal")


def test_run_bench_tiny_scale_latency_is_supplementary(tmp_path):
    out = tmp_path / "latency_tiny.json"
    assert run_bench_main(["--case", "latency", "--scale", "tiny", "--out", str(out), "--trials", "1"]) == 0
    data = json.loads(out.read_text())
    assert data["workload"]["supplementary"] is True
    assert data["workload"]["scale"] == "tiny"
    assert data["forced_length_ok"] is True
    assert data["metrics"]["ttft_s"]["median"] is not None
    assert data["metrics"]["ttft_s"]["mean"] is not None
    check_dir(tmp_path)


def test_mmm_reports_mean_and_stdev():
    stats = _mmm([10.0, 20.0, 30.0])
    assert stats["median"] == 20.0
    assert stats["mean"] == 20.0
    assert stats["stdev"] > 0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0


def test_parity_workloads_frozen_shapes():
    assert PARITY_CASES == ("decode", "latency", "batch8", "throughput", "prefill", "longctx")
    decode = FULL["decode"]
    assert (decode.concurrency, decode.prompt_tokens, decode.output_tokens) == (1, 256, 512)
    latency = FULL["latency"]
    assert (latency.concurrency, latency.prompt_tokens, latency.output_tokens) == (1, 512, 128)
    batch8 = FULL["batch8"]
    assert (batch8.concurrency, batch8.n_requests, batch8.prompt_tokens, batch8.output_tokens) == (8, 8, 256, 128)
    thru = FULL["throughput"]
    assert (thru.concurrency, thru.prompt_tokens, thru.output_tokens) == (16, 256, 128)
    assert batch8.prompt_tokens == thru.prompt_tokens
    assert batch8.output_tokens == thru.output_tokens
    prefill = FULL["prefill"]
    assert (prefill.prompt_tokens, prefill.output_tokens) == (2048, 16)
    longctx = FULL["longctx"]
    assert (longctx.prompt_tokens, longctx.output_tokens) == (8192, 64)
    assert token_budget_for_case("longctx", longctx) == 2048
    assert token_budget_for_case("throughput", FULL["throughput"]) == 4096
    assert token_budget_for_case("batch8", FULL["batch8"]) == 2048
    assert token_budget_for_case("prefill", FULL["prefill"]) == 2048
    assert token_budget_for_case("decode", FULL["decode"]) == 2048


def test_git_dirty_sees_this_repo():
    root = Path(__file__).resolve().parents[2]
    assert isinstance(git_dirty(root), bool)


def test_length_split_metrics_bimodal():
    from qwen3_runtime.serving.slo_harness import RequestTrace

    short = RequestTrace(request_id=1, arrival_s=0.0, prompt_len=256, max_tokens=4)
    short.admitted = True
    short.first_token_s = 0.04
    short.first_scheduled_s = 0.01
    short.tokens = [1, 2, 3, 4]
    long = RequestTrace(request_id=2, arrival_s=0.0, prompt_len=9981, max_tokens=4)
    long.admitted = True
    long.first_token_s = 3.0
    long.first_scheduled_s = 0.5
    long.tokens = [1, 2, 3, 4]
    split = length_split_metrics([short, long])
    assert split["short_prompt_len"] == 256
    assert split["long_prompt_len"] == 9981
    assert split["short_ttft_s_p99"] == 0.04
    assert split["long_ttft_s_p99"] == 3.0
    assert split["short_completed"] == 1
    assert split["long_completed"] == 1
