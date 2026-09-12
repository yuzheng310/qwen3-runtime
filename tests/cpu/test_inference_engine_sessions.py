"""Session rollout holds KV across a turn and batches; the adapter drops it on cue.

The fake engine here is a scheduler and a step loop rather than a stub that
returns a completion, because both properties under test are properties of how
requests interleave. A fake that answers a turn in one call can only ever show
a batch of one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.rollout.execution import SessionRollout
from qwen3_runtime.integrations.skyrl.inference_engine import (
    Qwen3InferenceEngine,
    _openai_logprob_content,
)


class FakeRequest:
    def __init__(self, request_id: int, hold_kv: bool):
        self.request_id = request_id
        self.hold_kv = hold_kv
        self.status = RequestStatus.WAITING
        self.emitted = 0
        self.last_logprob: float | None = None
        self.block_table = [0]
        self.cached_tokens = 0


class FakeEngine:
    """Emits ``tokens_per_step`` tokens per running request per step.

    One is the ordinary autoregressive shape. More than one is what speculative
    decoding produces, and it is the case the driver used to mis-handle, so it
    has to be reachable here.
    """

    tokens_per_step = 1

    def __init__(self, completion: list[int] | None = None):
        self.block_manager = SimpleNamespace(
            num_blocks=1000, block_size=16, num_free_blocks=900, epoch=0
        )
        self.calls: list[tuple] = []
        self.completion = completion if completion is not None else [900, 901]
        self.scheduler = SimpleNamespace(waiting=[], running=[])
        self._requests: dict[int, FakeRequest] = {}
        self.last_emitted: dict[int, list[int]] = {}
        self.last_emitted_logprobs: dict[int, list[float]] = {}
        self._next_id = 0
        self.fail_step_once = False
        self.batch_sizes: list[int] = []

    def add_request(self, ids, *, max_tokens, sampling, ignore_eos, stop_token_ids, hold_kv, forced_tokens=None):
        self._next_id += 1
        request = FakeRequest(self._next_id, hold_kv)
        self._requests[self._next_id] = request
        self.scheduler.waiting.append(request)
        self.calls.append(("add", list(ids), hold_kv))
        return self._next_id

    def resume_request(
        self, rid, suffix, max_tokens, *, hold_kv, sampling, ignore_eos, stop_token_ids, forced_tokens=None
    ):
        request = self._requests[rid]
        request.status = RequestStatus.WAITING
        request.emitted = 0
        self.scheduler.waiting.append(request)
        self.calls.append(("resume", rid, list(suffix), hold_kv))

    def step(self):
        if self.fail_step_once:
            self.fail_step_once = False
            raise RuntimeError("waiting request exceeds the KV pool even after shrinking")
        self.scheduler.running.extend(self.scheduler.waiting)
        self.scheduler.waiting.clear()
        self.batch_sizes.append(len(self.scheduler.running))
        self.last_emitted = {}
        self.last_emitted_logprobs = {}
        for request in list(self.scheduler.running):
            n = min(self.tokens_per_step, len(self.completion) - request.emitted)
            tokens = self.completion[request.emitted : request.emitted + n]
            request.emitted += n
            request.last_logprob = -0.5
            self.last_emitted[request.request_id] = list(tokens)
            # Published with the tokens, as the real engine does, so the driver
            # never has to reach back into a request that may already be gone.
            # Distinct values per position so a test can tell a real per-token
            # list from one scalar repeated.
            self.last_emitted_logprobs[request.request_id] = [
                -0.5 if i % 2 == 0 else -0.6 for i in range(n)
            ]
            if request.emitted >= len(self.completion):
                request.status = (
                    RequestStatus.PAUSED if request.hold_kv else RequestStatus.FINISHED
                )
                self.scheduler.running.remove(request)
                if not request.hold_kv:
                    self._requests.pop(request.request_id, None)

    def finish_request(self, rid):
        request = self._requests.pop(rid, None)
        if request in self.scheduler.running:
            self.scheduler.running.remove(request)
        if request in self.scheduler.waiting:
            self.scheduler.waiting.remove(request)
        n_blocks = len(getattr(request, "block_table", []) or []) if request is not None else 0
        self.block_manager.num_free_blocks = min(
            self.block_manager.num_blocks,
            self.block_manager.num_free_blocks + max(1, n_blocks),
        )
        self.calls.append(("finish", rid))

    def generate(self, ids, **kwargs):
        self.calls.append(("generate", list(ids), kwargs.get("hold_kv")))
        return list(self.completion)

    def apply_named_weights(self, items):
        self.calls.append(("apply", len(items)))
        return len(items)

    def sleep(self, level=1):
        self.calls.append(("sleep", level))
        return {}

    def abort_generation(self):
        self.calls.append(("abort",))


def _turn(wrapper: SessionRollout | Qwen3InferenceEngine, ids: list[int]) -> list[int]:
    return _turn_with_logprobs(wrapper, ids)[0]


def _turn_with_logprobs(
    wrapper: SessionRollout | Qwen3InferenceEngine, ids: list[int]
) -> tuple[list[int], list[float]]:
    async def run():
        rollout = wrapper.rollout if isinstance(wrapper, Qwen3InferenceEngine) else wrapper
        try:
            turn = await rollout.run_turn(ids, max_tokens=32, sampling=None)
            return turn.tokens, turn.logprobs
        finally:
            rollout.stop()

    return asyncio.run(run())


def _kinds(engine: FakeEngine) -> list[str]:
    return [call[0] for call in engine.calls]


def test_trajectory_end_releases_only_the_exact_paused_history(monkeypatch):
    monkeypatch.setenv("QWEN3_SESSION_KV", "1")
    engine = FakeEngine()
    wrapper = SessionRollout(engine)

    async def run():
        try:
            await wrapper.run_turn([1, 2], max_tokens=32, sampling=None)
            await wrapper.run_turn([3, 4], max_tokens=32, sampling=None)
            assert await wrapper.finish_session([1, 2]) == 0
            assert await wrapper.finish_session([1, 2, 900, 901]) == 1
            assert await wrapper.finish_session([1, 2, 900, 901]) == 0
            assert wrapper.session_report()["live_sessions"] == 1
            assert wrapper._sessions.session_tokens(2) == [3, 4, 900, 901]
        finally:
            wrapper.stop()

    asyncio.run(run())


def test_trajectory_end_does_not_guess_between_identical_siblings(monkeypatch):
    monkeypatch.setenv("QWEN3_SESSION_KV", "1")
    engine = FakeEngine()
    wrapper = SessionRollout(engine)

    async def run():
        try:
            await wrapper.run_turn([1, 2], max_tokens=32, sampling=None)
            await wrapper.run_turn([1, 2], max_tokens=32, sampling=None)
            assert await wrapper.finish_session([1, 2, 900, 901]) == 0
            assert wrapper.session_report()["live_sessions"] == 2
        finally:
            wrapper.stop()

    asyncio.run(run())


def test_the_first_turn_of_a_conversation_parks_its_kv_instead_of_freeing_it(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine()
    wrapper = SessionRollout(engine)

    assert _turn(wrapper, [1, 2, 3]) == [900, 901]
    assert engine.calls == [("add", [1, 2, 3], True)]


def test_the_next_turn_resumes_and_only_submits_what_the_tool_appended(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine(completion=[900, 901])
    wrapper = SessionRollout(engine)

    _turn(wrapper, [1, 2, 3])
    # Next prompt is the previous one, the model's answer, then the tool output.
    _turn(wrapper, [1, 2, 3, 900, 901, 50, 51])

    assert engine.calls[1] == ("resume", 1, [50, 51], True)
    assert wrapper.session_report()["tokens_reused"] == 5


def test_a_divergent_next_turn_starts_a_fresh_session_rather_than_rewinding(monkeypatch):
    """Rewinding is not token-for-token equal yet, so a divergence pays in full."""
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine(completion=[900, 901])
    wrapper = SessionRollout(engine)

    _turn(wrapper, list(range(100)))
    # The re-encoded history differs from what was emitted at the very end.
    _turn(wrapper, list(range(100)) + [900, 777, 778])

    assert _kinds(engine) == ["add", "add"]
    assert wrapper.session_report()["tokens_reused"] == 0


def test_a_weight_update_voids_every_session_so_none_resumes_onto_dropped_kv(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine()
    wrapper = Qwen3InferenceEngine(engine)
    _turn(wrapper, [1, 2, 3])

    asyncio.run(
        wrapper.update_named_weights(
            {"names": ["w"], "extras": [{"tensors": [object()]}], "dtypes": None, "shapes": None}
        )
    )
    _turn(wrapper, [1, 2, 3, 900, 901, 7])

    assert _kinds(engine) == ["add", "finish", "apply", "add"]


def test_sleeping_releases_the_sessions_it_was_holding(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine()
    wrapper = Qwen3InferenceEngine(engine)
    _turn(wrapper, [1, 2, 3])

    asyncio.run(wrapper.sleep(level=1))

    assert _kinds(engine) == ["add", "finish", "sleep"]


def test_turning_sessions_off_still_batches_but_keeps_no_kv(monkeypatch):
    """Batching is independent of reuse; turning reuse off must not serialize."""
    monkeypatch.setenv("QWEN3_SESSION_KV", "0")
    engine = FakeEngine()
    wrapper = SessionRollout(engine)

    _turn(wrapper, [1, 2, 3])
    _turn(wrapper, [1, 2, 3, 900, 901, 7])

    assert engine.calls == [
        ("add", [1, 2, 3], False),
        ("add", [1, 2, 3, 900, 901, 7], False),
    ]
    assert wrapper.session_report()["tokens_reused"] == 0


def test_concurrent_turns_reach_the_engine_together_instead_of_queueing(monkeypatch):
    """The point of the driver: four trajectories, one decode step, not four."""
    monkeypatch.delenv("QWEN3_BATCHING", raising=False)
    engine = FakeEngine(completion=[900, 901, 902])
    wrapper = SessionRollout(engine)

    async def run():
        # Queue arrivals before starting the worker: batching must not depend
        # on the OS letting the event loop outrun a zero-cost fake decode step.
        start_driver = wrapper._driver.start
        monkeypatch.setattr(wrapper._driver, "start", lambda: None)
        turns = [asyncio.create_task(wrapper.run_turn([i, i + 1], max_tokens=32, sampling=None)) for i in range(4)]
        await asyncio.sleep(0)
        start_driver()
        try:
            return await asyncio.wait_for(asyncio.gather(*turns), timeout=5)
        finally:
            wrapper.stop()

    results = asyncio.run(run())

    assert [turn.tokens for turn in results] == [[900, 901, 902]] * 4
    assert engine.batch_sizes == [4, 4, 4]
    assert wrapper.batching_report()["max_batch"] == 4


def test_the_control_arm_admits_one_turn_at_a_time(monkeypatch):
    """QWEN3_BATCHING=0 has to reproduce the old serial path exactly.

    It is what the batched span gets measured against, so if it quietly still
    batched, the speedup would be measured against itself.
    """
    monkeypatch.setenv("QWEN3_BATCHING", "0")
    engine = FakeEngine(completion=[900, 901, 902])
    wrapper = SessionRollout(engine)

    async def run():
        results = await asyncio.gather(
            *(wrapper.run_turn([i, i + 1], max_tokens=32, sampling=None) for i in range(4))
        )
        wrapper.stop()
        return results

    results = asyncio.run(run())

    assert [turn.tokens for turn in results] == [[900, 901, 902]] * 4
    assert set(engine.batch_sizes) == {1}
    assert len(engine.batch_sizes) == 12  # every token its own pass over the weights


def test_deferred_admission_waits_for_capacity_without_retrying_every_decode(monkeypatch):
    from qwen3_runtime.rollout.driver import AdmissionDeferred, EngineDriver

    engine = FakeEngine(completion=list(range(20)))
    engine.config = SimpleNamespace(max_num_seqs=2)
    driver = EngineDriver(engine)
    attempts = []

    def admit(second=False):
        if second:
            attempts.append(len(engine.batch_sizes))
            if engine.scheduler.running or engine.scheduler.waiting:
                raise AdmissionDeferred("active request owns the capacity")
        return engine.add_request(
            [1], max_tokens=20, sampling=None, ignore_eos=True,
            stop_token_ids=None, hold_kv=False,
        )

    async def run():
        start = driver.start
        monkeypatch.setattr(driver, "start", lambda: None)
        first = asyncio.create_task(driver.run_turn(admit))
        second = asyncio.create_task(driver.run_turn(lambda: admit(True)))
        await asyncio.sleep(0)
        start()
        try:
            return await asyncio.wait_for(asyncio.gather(first, second), 5)
        finally:
            driver.stop()

    assert [turn.tokens for turn in asyncio.run(run())] == [list(range(20))] * 2
    assert attempts == [0, 20]


def test_stopping_driver_resolves_deferred_admission():
    from qwen3_runtime.rollout.driver import AdmissionDeferred, EngineDriver

    driver = EngineDriver(FakeEngine())

    def defer():
        raise AdmissionDeferred("no capacity yet")

    async def run():
        turn = asyncio.create_task(driver.run_turn(defer))
        await asyncio.sleep(0)
        driver.run_on_engine(lambda: None)
        driver.stop()
        with pytest.raises(RuntimeError, match="stopped before"):
            await asyncio.wait_for(turn, 1)

    asyncio.run(run())


def test_each_turn_gets_its_own_logprobs_and_not_a_neighbours():
    engine = FakeEngine(completion=[900, 901])
    wrapper = SessionRollout(engine)

    async def run():
        results = await asyncio.gather(
            *(wrapper.run_turn([i], max_tokens=32, sampling=None) for i in range(3))
        )
        wrapper.stop()
        return results

    for turn in asyncio.run(run()):
        assert len(turn.logprobs) == len(turn.tokens) == 2


def test_a_multi_token_step_carries_a_logprob_for_every_token():
    """One engine step can emit several tokens; the turn must score all of them.

    This is the shape speculative decoding produces. The driver used to append
    one ``last_logprob`` per step however many tokens it emitted, so a spec
    burst arrived with a single value and the rest of the positions were filled
    with 0.0 downstream.
    """
    engine = FakeEngine(completion=[900, 901, 902, 903])
    engine.tokens_per_step = 2
    tokens, logprobs = _turn_with_logprobs(SessionRollout(engine), [1, 2, 3])
    assert tokens == [900, 901, 902, 903]
    assert logprobs == [-0.5, -0.6, -0.5, -0.6]


def test_a_short_logprob_list_is_refused_instead_of_padded_with_zero():
    """0.0 is a valid-looking logprob for a token nobody scored."""
    with pytest.raises(ValueError, match="logprobs"):
        _openai_logprob_content(None, [11, 12, 13], [-0.5, -0.6])


def test_an_unscored_token_is_refused_instead_of_served_as_a_policy_sample():
    with pytest.raises(ValueError, match="no sampling-time logprob"):
        _openai_logprob_content(None, [11, 12], [-0.5, float("nan")])


def test_a_full_pool_gives_the_parked_kv_back_rather_than_failing_the_step(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine()
    wrapper = SessionRollout(engine)
    _turn(wrapper, [1, 2, 3])
    engine.fail_step_once = True

    assert _turn(wrapper, [500, 501]) == [900, 901]
    # The stuck request is released, the resident session handed back, retry runs.
    assert _kinds(engine) == ["add", "add", "finish", "finish", "add"]


def test_an_unrelated_failure_is_not_swallowed_by_the_retry(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)

    class Broken(FakeEngine):
        def step(self):
            raise RuntimeError("model runner exploded")

    wrapper = SessionRollout(Broken())

    try:
        _turn(wrapper, [1, 2, 3])
    except RuntimeError as exc:
        assert "exploded" in str(exc)
    else:
        raise AssertionError("expected the failure to propagate")


def test_parked_kv_budget_is_counted_in_blocks_not_tokens():
    from qwen3_runtime.rollout.execution import _parked_kv_budget

    engine = FakeEngine()
    assert _parked_kv_budget(engine) == int(0.45 * 1000)
    wrapper = SessionRollout(engine)
    assert wrapper._sessions.max_blocks == int(0.45 * 1000)
    assert "held_blocks" in wrapper.session_report()


def test_a_tight_free_list_evicts_the_oldest_parked_session(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    engine = FakeEngine()
    wrapper = SessionRollout(engine)
    _turn(wrapper, [1, 2, 3])
    engine.block_manager.num_free_blocks = 5  # 10% of 1000 is 100
    _turn(wrapper, [1, 2, 3, 900, 901, 50])
    assert "finish" in _kinds(engine)
    assert wrapper.session_report()["evicted"] >= 1


def test_failed_resume_keeps_the_session_claimable_without_counting_a_turn(monkeypatch):
    engine = FakeEngine()
    rollout = SessionRollout(engine, enabled=True)

    async def run():
        try:
            first = await rollout.run_turn([1, 2], max_tokens=2, sampling=None)
            before = rollout.session_report()
            history = [1, 2, *first.tokens, 3]
            original = engine.resume_request

            def fail(*args, **kwargs):
                raise ValueError("injected resume failure")

            monkeypatch.setattr(engine, "resume_request", fail)
            with pytest.raises(ValueError, match="injected resume failure"):
                await rollout.run_turn(history, max_tokens=2, sampling=None)
            assert rollout.session_report() == before
            monkeypatch.setattr(engine, "resume_request", original)
            resumed = await rollout.run_turn(history, max_tokens=2, sampling=None)
            assert resumed.request_id == first.request_id
            assert resumed.tokens == first.tokens
            report = rollout.session_report()
            assert report["turns"] == 2
            assert report["started"] == report["resumed"] == 1
            assert report["tokens_reused"] == 4
        finally:
            rollout.clear()

    asyncio.run(run())
