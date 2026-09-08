"""Task 1 SLO harness: fences, heterogeneous max_tokens, admission, goodput, backlog."""

import json
from types import SimpleNamespace

import pytest

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.serving.slo_harness import (
    AdmissionView,
    SessionTurn,
    always_admit,
    request_slo_admit,
    run_closed_batch,
    run_poisson,
    run_sequential_requests,
    run_sessions,
    session_slo_admit,
    slo_aware_admit,
    _record_emitted,
    RequestTrace,
)
from tests.cpu.test_engine import FakeModelRunner
from workloads.code_localization.output_lengths import recorded_output_tokens


def _engine(*, max_seqs=8, num_blocks=128, block_size=4, token_budget=64) -> Engine:
    return Engine(
        Config(
            max_num_batched_tokens=token_budget,
            max_num_seqs=max_seqs,
            num_kv_blocks=num_blocks,
            block_size=block_size,
        ),
        FakeModelRunner(),
    )


def test_poisson_serving_path_skips_cuda_sync(monkeypatch):
    calls = {"n": 0}

    def fake_sync(_engine):
        calls["n"] += 1

    monkeypatch.setattr("qwen3_runtime.serving.slo_harness._sync", fake_sync)
    prompts = [[1, 2], [3, 4]]
    arrivals = [0.0, 0.0]
    run_poisson(_engine(), prompts, [2, 2], arrivals, profile=False)
    assert calls["n"] == 0
    run_poisson(_engine(), prompts, [2, 2], arrivals, profile=True)
    assert calls["n"] > 0


def test_closed_batch_and_sequential_honor_profile_flag(monkeypatch):
    calls = {"n": 0}

    def fake_sync(_engine):
        calls["n"] += 1

    monkeypatch.setattr("qwen3_runtime.serving.slo_harness._sync", fake_sync)
    run_closed_batch(_engine(), [[1, 2]], 2, profile=False)
    run_sequential_requests(_engine(), [([1, 2], 2)], nvtx=False, profile=False)
    assert calls["n"] == 0
    run_closed_batch(_engine(), [[1, 2]], 2, profile=True)
    assert calls["n"] > 0


def test_poisson_honors_per_request_max_tokens():
    prompts = [[7, 8], [9, 10, 11], [1]]
    run = run_poisson(_engine(), prompts, [2, 5, 3], [0.0, 0.0, 0.0])
    assert [len(t.tokens) for t in run.traces] == [2, 5, 3]
    assert [t.max_tokens for t in run.traces] == [2, 5, 3]
    assert all(len(t.tokens) == t.max_tokens for t in run.traces)


def test_poisson_scalar_max_tokens_still_broadcasts():
    prompts = [[1, 2], [3, 4]]
    run = run_poisson(_engine(), prompts, 3, [0.0, 0.0])
    assert [len(t.tokens) for t in run.traces] == [3, 3]


def test_speculative_burst_records_observed_step_timestamp():
    trace = RequestTrace(request_id=7, arrival_s=0.0, prompt_len=2, max_tokens=3)
    trace.first_token_s = 1.0
    trace.last_token_s = 1.0
    trace.tokens = [10]
    engine = SimpleNamespace(last_emitted={7: [11, 12]})

    _record_emitted(engine, {7: trace}, 1.5)

    assert trace.tokens == [10, 11, 12]
    assert trace.itl_s == pytest.approx([0.5, 0.0])


def test_admission_seam_can_reject_without_add_request():
    prompts = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

    def first_two(view: AdmissionView) -> bool:
        return view.admitted < 2

    run = run_poisson(
        _engine(), prompts, [2] * 5, [0.0] * 5, admit=first_two, slo_ttft_s=10.0, slo_tpot_s=10.0
    )
    assert run.offered == 5
    assert run.admitted == 2
    assert run.rejected == 3
    assert run.completed == 2
    assert run.slo_met == 2
    assert sum(1 for t in run.traces if t.admitted) == 2
    assert sum(1 for t in run.traces if not t.admitted) == 3
    assert all(t.ttft_s is None for t in run.traces if not t.admitted)


def test_always_admit_is_the_default_policy():
    view = AdmissionView(
        prompt_len=4,
        max_tokens=8,
        arrival_s=0.0,
        waiting=99,
        running=99,
        offered=10,
        admitted=10,
    )
    assert always_admit(view) is True


def test_ttft_includes_queue_time_from_arrival():
    prompts = [[1, 2], [3, 4]]
    run = run_poisson(_engine(max_seqs=1), prompts, [4, 4], [0.0, 0.0])
    assert run.admitted == 2
    first, second = run.traces
    assert first.ttft_s is not None and second.ttft_s is not None
    assert second.ttft_s > first.ttft_s
    assert first.queue_s is not None and first.prefill_s is not None
    assert second.queue_s is not None
    assert abs((first.queue_s + first.prefill_s) - first.ttft_s) < 1e-6
    assert second.queue_s > first.queue_s
    assert run.step_mix is not None and run.step_mix["n_steps"] > 0
    assert run.step_mix["prefill_tokens_sum"] + run.step_mix["decode_tokens_sum"] > 0


def test_goodput_excludes_slo_violations():
    prompts = [[1, 2], [3, 4]]
    tight = run_poisson(
        _engine(), prompts, [3, 3], [0.0, 0.0], slo_ttft_s=-1.0, slo_tpot_s=-1.0
    )
    assert tight.completed == 2
    assert tight.slo_met == 0
    assert tight.goodput_per_s == 0.0
    loose = run_poisson(
        _engine(), prompts, [3, 3], [0.0, 0.0], slo_ttft_s=1e9, slo_tpot_s=1e9
    )
    assert loose.slo_met == 2
    assert loose.goodput_per_s is not None and loose.goodput_per_s > 0
    unset = run_poisson(_engine(), prompts, [3, 3], [0.0, 0.0])
    assert unset.slo_met is None
    assert unset.goodput_per_s is None


def test_backlog_is_sampled_and_has_a_trend():
    prompts = [[1, 2] for _ in range(6)]
    run = run_poisson(_engine(max_seqs=1), prompts, [3] * 6, [0.0] * 6)
    assert run.backlog
    assert any(s.depth > 0 for s in run.backlog)
    assert run.backlog_slope_per_s is not None


def test_run_slo_boundary_profile_tiny(tmp_path):
    from bench.run_slo_boundary_profile import main as profile_main

    out = tmp_path / "slo-boundary"
    assert profile_main(["--tiny", "--out-dir", str(out)]) == 0
    data = json.loads((out / "tiny.json").read_text())
    assert data["step_mix"]["n_steps"] > 0
    assert data["decomp"]["n_traces"] == 3
    assert data["decomp"]["queue_s_p99"] is not None
    from bench.run_slo_harness import main as slo_main

    out = tmp_path / "serving.json"
    assert slo_main(["--tiny", "--out", str(out), "--no-profile"]) == 0
    data = json.loads(out.read_text())
    assert data["workload"]["sync_policy"] == "serving"
    assert data["accounting"]["offered"] == data["accounting"]["admitted"]
    assert data["accounting"]["rejected"] == 0
    assert data["accounting"]["completed"] == data["accounting"]["offered"]
    assert "slope_per_s" in data["backlog"]
    assert data["forced_length_ok"] is True


def test_recorded_output_tokens_match_frozen_corpus_summary():
    xs = recorded_output_tokens()
    assert len(xs) == 2395
    assert min(xs) >= 1
    assert max(xs) == 375
    assert round(sum(xs) / len(xs), 2) == 93.93
    mid = sorted(xs)[len(xs) // 2]
    assert mid == 87


def test_run_sessions_holds_kv_across_think_time_and_counts_turns():
    engine = _engine(max_seqs=2, num_blocks=64, token_budget=32)
    sessions = [
        [
            SessionTurn(prompt=[1, 2, 3], max_tokens=2, think_s=0.01),
            SessionTurn(prompt=[9], max_tokens=2, think_s=0.0),
        ],
        [
            SessionTurn(prompt=[4, 5], max_tokens=2, think_s=0.01),
            SessionTurn(prompt=[8], max_tokens=1, think_s=0.0),
        ],
    ]
    run = run_sessions(engine, sessions, max_active=2)
    assert run.offered == 4
    assert run.admitted == 4
    assert run.rejected == 0
    assert run.completed == 4
    assert [len(t.tokens) for t in run.traces] == [2, 2, 2, 1]
    assert sorted(t.turn_id for t in run.traces) == [0, 0, 1, 1]
    assert engine.is_finished()
    assert engine.scheduler.paused == {}


def test_run_sessions_max_active_starts_third_after_one_finishes():
    engine = _engine(max_seqs=2, num_blocks=64, token_budget=32)
    sessions = [
        [SessionTurn(prompt=[1, 2], max_tokens=2, think_s=0.0)],
        [SessionTurn(prompt=[3, 4], max_tokens=2, think_s=0.0)],
        [SessionTurn(prompt=[5, 6], max_tokens=2, think_s=0.0)],
    ]
    run = run_sessions(engine, sessions, max_active=2)
    assert run.completed == 3
    assert run.offered == 3


def test_run_sessions_records_kv_exhaustion_instead_of_raising():
    engine = _engine(max_seqs=2, num_blocks=6, block_size=4, token_budget=16)
    long_think = [
        [
            SessionTurn(prompt=[1, 2, 3, 4], max_tokens=4, think_s=30.0),
            SessionTurn(prompt=[9, 9], max_tokens=2, think_s=0.0),
        ]
        for _ in range(4)
    ]
    run = run_sessions(engine, long_think, max_active=4)
    assert run.kv_exhausted or run.num_preemptions >= 1
    assert run.offered >= 1


def _view(**kw) -> AdmissionView:
    base = dict(
        prompt_len=256,
        max_tokens=8,
        arrival_s=0.0,
        waiting=0,
        running=0,
        offered=0,
        admitted=0,
    )
    base.update(kw)
    return AdmissionView(**base)


def test_slo_aware_admit_rejects_when_waiting_drain_exceeds_slo():
    # 40 queued jobs at 6.245 completed/s is ~6.4 s, above A=4.5 s.
    assert (
        slo_aware_admit(
            _view(waiting=40, running=8),
            slo_ttft_s=4.5,
            max_occupancy=None,
            drain_per_s=6.245,
        )
        is False
    )
    assert (
        slo_aware_admit(
            _view(waiting=0, running=8),
            slo_ttft_s=4.5,
            max_occupancy=None,
            drain_per_s=6.245,
        )
        is True
    )


def test_request_slo_admit_matches_task2_lambda10_drain():
    # λ=10 depth_max was 29 with running≈8, so waiting≈21 stays under A.
    assert request_slo_admit(_view(waiting=21, running=8)) is True
    # λ=12 depth_max 39: waiting≈31 predicts wait > A.
    assert request_slo_admit(_view(waiting=31, running=8)) is False


def test_session_slo_admit_caps_new_occupancy_at_six_and_keeps_held_slots():
    assert session_slo_admit(_view(waiting=5, running=0, paused=0)) is True
    assert session_slo_admit(_view(waiting=6, running=0, paused=0)) is False
    assert session_slo_admit(_view(waiting=0, running=3, paused=3)) is False
    # Later turn of an already-admitted session must not be shed.
    assert session_slo_admit(_view(waiting=0, running=3, paused=3, held_slot=True)) is True


def test_poisson_slo_admit_rejects_instead_of_enqueueing_past_predicted_wait():
    def tight(view: AdmissionView) -> bool:
        return slo_aware_admit(view, slo_ttft_s=0.5, max_occupancy=None, drain_per_s=1.0)

    prompts = [[1, 2] for _ in range(6)]
    run = run_poisson(_engine(max_seqs=1), prompts, [2] * 6, [0.0] * 6, admit=tight)
    assert run.offered == 6
    assert run.rejected >= 4
    assert run.admitted + run.rejected == 6
    assert all(t.ttft_s is None for t in run.traces if not t.admitted)


def test_sessions_slo_admit_rejects_overflow_first_turns_not_later_turns():
    engine = _engine(max_seqs=8, num_blocks=256, token_budget=64)
    sessions = [
        [
            SessionTurn(prompt=[1, 2], max_tokens=2, think_s=0.01),
            SessionTurn(prompt=[9], max_tokens=1, think_s=0.0),
        ]
        for _ in range(8)
    ]
    run = run_sessions(engine, sessions, max_active=8, admit=session_slo_admit)
    # Six sessions fully admitted (2 turns) + two rejected first-turns.
    assert run.offered == 14
    assert run.rejected == 2
    assert run.admitted == 12
    assert run.completed == 12


def test_summarize_traces_forced_length_ignores_rejects():
    from bench.metrics import summarize_traces

    prompts = [[1, 2], [3, 4], [5, 6]]

    def first_one(view: AdmissionView) -> bool:
        return view.admitted < 1

    run = run_poisson(_engine(), prompts, [2, 2, 2], [0.0, 0.0, 0.0], admit=first_one)
    assert run.rejected == 2
    metrics = summarize_traces(run.traces, run.wall_s)
    assert metrics["forced_length_ok"] is True
    assert metrics["n_admitted"] == 1


def test_note_new_tokens_records_ttft_and_itl_from_host_clock():
    from bench.vllm_slo import note_new_tokens
    from qwen3_runtime.serving.slo_harness import RequestTrace

    tr = RequestTrace(request_id=0, arrival_s=10.0, prompt_len=4, max_tokens=3)
    note_new_tokens(tr, [7], 10.5)
    note_new_tokens(tr, [7, 8, 9], 10.7)
    assert tr.tokens == [7, 8, 9]
    assert tr.ttft_s == pytest.approx(0.5)
    assert tr.itl_s[0] == pytest.approx(0.2)
    assert tr.itl_s[1] == pytest.approx(0.0)
    assert tr.tpot_s == pytest.approx(0.1)
