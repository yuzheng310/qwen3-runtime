from __future__ import annotations

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.engine.session_offload import RestoreAdmissionError
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.integrations.skyrl.inference_engine import Qwen3InferenceEngine
from qwen3_runtime.rollout.execution import SessionRollout
from tests.cpu.test_tiny_qwen3 import tiny_config


def _engine(*, blocks: int = 32, cache: bool = False, active: int = 2, cpu_blocks: int = 16) -> Engine:
    model = Qwen3ForCausalLM(tiny_config()).eval()
    # Tiny CPU pool is float32, so this is an explicit finite host budget,
    # not an implicit allocation of all system RAM.
    probe_bytes = 2 * 2 * 2 * 4 * 2 * 4
    cfg = Config(
        block_size=4,
        num_kv_blocks=blocks,
        max_num_seqs=active,
        max_num_batched_tokens=32,
        enable_prefix_cache=cache,
        session_cpu_offload="sync",
        cpu_kv_max_bytes=probe_bytes * cpu_blocks,
        transfer_chunk_bytes=probe_bytes,
    )
    return Engine(cfg, PagedRunner(model))


def test_sync_pause_offload_restore_roundtrip_keeps_generation_equivalent():
    torch.manual_seed(71)
    prompt = [1, 2, 3, 4, 5, 6]
    suffix = [7, 8]
    held = _engine()
    rid = held.add_request(prompt, max_tokens=2, hold_kv=True)
    first = held.drain_request(rid)
    req = held._requests[rid]
    assert req.status == RequestStatus.PAUSED
    assert req.num_computed_tokens == len(req.token_ids) - 1
    blocks_before = len(req.block_table)

    held.offload_request(rid, session_key="trajectory-1")
    assert req.kv_residency == "cpu"
    assert req.block_table == []
    assert held.block_manager.num_free_blocks == held.block_manager.num_blocks
    assert held.has_cpu_snapshot(rid)
    assert held.session_offload.report()["saved"] == 1

    held.resume_request(rid, suffix, 2, hold_kv=False)
    assert req.kv_residency == "gpu"
    assert len(req.block_table) >= 1
    second = held.drain_request(rid)
    assert not held.has_cpu_snapshot(rid)
    assert held.session_offload.report()["restored"] == 1
    report = held.session_offload.report()
    assert report["useful_restores"] == 1
    assert report["imported_tokens"] == len(prompt) + len(first) - 1
    assert report["d2h_bytes"] == report["restored_d2h_bytes"]
    assert report["pending_d2h_bytes"] == report["unused_d2h_bytes"] == 0

    torch.manual_seed(71)
    fresh = _engine()
    expected = fresh.generate(prompt + first + suffix, max_tokens=2)
    assert second == expected
    assert blocks_before > 0


def test_offload_failure_and_cancel_do_not_leave_host_or_gpu_leases():
    engine = _engine()
    rid = engine.add_request([1, 2, 3, 4], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    req = engine._requests[rid]
    engine.offload_request(rid)
    assert engine.session_offload.report()["cpu_committed_bytes"] > 0
    engine.finish_request(rid)
    assert not engine.has_cpu_snapshot(rid)
    assert engine.session_offload.report()["cpu_committed_bytes"] == 0
    report = engine.session_offload.report()
    assert report["unused_d2h_bytes"] == report["d2h_bytes"] > 0
    assert req.block_table == []
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_oversized_save_preserves_useful_cpu_snapshots():
    import pytest
    from qwen3_runtime.kv.cpu_store import CpuKVCapacityError

    engine = _engine(blocks=32, cpu_blocks=2)
    small = engine.add_request([1] * 4, max_tokens=1, hold_kv=True)
    engine.drain_request(small)
    engine.offload_request(small)
    large = engine.add_request([1] * 12, max_tokens=1, hold_kv=True)
    engine.drain_request(large)
    with pytest.raises(CpuKVCapacityError):
        engine.offload_request(large)
    assert engine.has_cpu_snapshot(small)
    assert engine.session_offload.report()["cpu_evicted"] == 0
    assert engine._requests[large].block_table


def test_explicit_trajectory_end_releases_cpu_and_gpu_ownership():
    import asyncio

    engine = _engine()
    wrapper = SessionRollout(engine)
    rid = engine.add_request([1, 2, 3, 4], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    req = engine._requests[rid]
    history = list(req.token_ids)
    wrapper._sessions.park(rid, history, len(req.block_table))
    engine.offload_request(rid)
    wrapper._sessions.update_blocks(rid, 0)
    assert asyncio.run(wrapper.finish_session(history)) == 1
    assert not engine.has_cpu_snapshot(rid)
    assert rid not in engine._requests
    assert wrapper.session_report()["live_sessions"] == 0
    assert engine.session_offload.report()["cpu_committed_bytes"] == 0


def test_save_copy_failure_aborts_reservation_and_keeps_source(monkeypatch):
    engine = _engine()
    rid = engine.add_request([1, 2, 3, 4], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    req = engine._requests[rid]
    table = list(req.block_table)
    free = engine.block_manager.num_free_blocks

    def fail(*args, **kwargs):
        raise RuntimeError("injected D2H failure")

    monkeypatch.setattr(engine.runner.pool, "export_blocks", fail)
    import pytest

    with pytest.raises(RuntimeError, match="injected D2H failure"):
        engine.offload_request(rid)
    assert req.block_table == table
    assert engine.block_manager.num_free_blocks == free
    assert not engine.has_cpu_snapshot(rid)
    assert engine.session_offload.report()["cpu_reserved_bytes"] == 0


def test_restore_copy_failure_rolls_back_to_cpu_snapshot(monkeypatch):
    engine = _engine()
    rid = engine.add_request([1, 2, 3, 4], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    engine.offload_request(rid)
    original = engine.runner.pool.import_blocks

    def fail(*args, **kwargs):
        raise RuntimeError("injected H2D failure")

    monkeypatch.setattr(engine.runner.pool, "import_blocks", fail)
    import pytest

    with pytest.raises(RestoreAdmissionError, match="rolled back"):
        engine.resume_request(rid, [9], 1, hold_kv=False)
    req = engine._requests[rid]
    assert req.status == RequestStatus.PAUSED
    assert req.kv_residency == "cpu"
    assert req.block_table == []
    assert engine.has_cpu_snapshot(rid)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    monkeypatch.setattr(engine.runner.pool, "import_blocks", original)


def test_restore_reuses_live_apc_blocks_before_importing_missing_blocks():
    engine = _engine(cache=True)
    rid = engine.add_request(list(range(1, 11)), max_tokens=1, hold_kv=True)
    first = engine.drain_request(rid)
    engine.offload_request(rid)
    assert engine.block_manager.cache_blocks >= 1
    assert engine.has_cpu_snapshot(rid)
    engine.resume_request(rid, [20], 1, hold_kv=False, forced_tokens=[21])
    assert engine.session_offload.report()["restored"] == 1
    assert engine.drain_request(rid) == [21]
    assert first


def test_weight_update_invalidates_cpu_snapshot_before_new_generation():
    engine = _engine()
    rid = engine.add_request([1, 2, 3, 4], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    engine.offload_request(rid)
    assert engine.has_cpu_snapshot(rid)
    model = engine.runner.model
    items = [(name, param.detach().clone()) for name, param in model.named_parameters()]
    engine.apply_named_weights(items)
    assert not engine.has_cpu_snapshot(rid)
    assert engine._requests[rid].kv_residency == "none"
    assert engine._requests[rid].block_table == []


def test_skyrl_adapter_uses_the_same_sync_offload_path():
    """The multi-turn wrapper, not a benchmark-only cache, closes the loop."""
    model = Qwen3ForCausalLM(tiny_config()).eval()
    cfg = Config(
        block_size=4,
        num_kv_blocks=3,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        session_cpu_offload="sync",
        cpu_kv_max_bytes=4096,
        transfer_chunk_bytes=128,
    )
    engine = Engine(cfg, PagedRunner(model))
    wrapper = Qwen3InferenceEngine(engine)

    import asyncio

    async def run() -> tuple[list[int], list[int]]:
        first, _ = await wrapper._run_turn(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], max_tokens=1, sampling=None
        )
        wrapper.rollout.stop()
        second, _ = await wrapper._run_turn(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, *first, 10], max_tokens=1, sampling=None
        )
        wrapper.rollout.stop()
        return first, second

    first, second = asyncio.run(run())
    assert len(first) == len(second) == 1
    assert wrapper.session_report()["offload_saved"] >= 1
    assert wrapper.session_report()["offload_restored"] >= 1


def test_watermark_skips_cpu_sessions_and_reclaims_apc(monkeypatch):
    engine = _engine(blocks=3, cache=True)
    wrapper = SessionRollout(engine)
    first = engine.add_request(list(range(1, 9)), max_tokens=1, hold_kv=True)
    engine.drain_request(first)
    second = engine.add_request([20], max_tokens=1, hold_kv=True)
    engine.drain_request(second)
    req = engine._requests[first]
    wrapper._sessions.park(first, list(req.token_ids), len(req.block_table))
    original = wrapper._sessions.oldest_id
    calls = 0

    def bounded(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls <= 4, "watermark repeatedly selects a CPU session"
        return original(*args, **kwargs)

    monkeypatch.setattr(wrapper._sessions, "oldest_id", bounded)
    wrapper._offload_oldest_until_safe()
    assert engine.has_cpu_snapshot(first)
    assert engine.block_manager.num_free_blocks >= 1


def test_watermark_selects_gpu_session_after_older_cpu_session(monkeypatch):
    engine = _engine(blocks=3)
    wrapper = SessionRollout(engine)
    first = engine.add_request([1, 2], max_tokens=1, hold_kv=True)
    engine.drain_request(first)
    engine.offload_request(first)
    wrapper._sessions.park(first, list(engine._requests[first].token_ids), 0)
    second = engine.add_request(list(range(1, 10)), max_tokens=1, hold_kv=True)
    engine.drain_request(second)
    wrapper._sessions.park(second, list(engine._requests[second].token_ids), 3)
    original = wrapper._sessions.oldest_id
    calls = 0

    def bounded(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls <= 4, "watermark never reaches the GPU session"
        return original(*args, **kwargs)

    monkeypatch.setattr(wrapper._sessions, "oldest_id", bounded)
    wrapper._offload_oldest_until_safe()
    assert engine.has_cpu_snapshot(second)
    assert engine.block_manager.num_free_blocks == 3


def test_restore_fallback_forgets_retired_session_before_next_claim(monkeypatch):
    engine = _engine()
    wrapper = SessionRollout(engine)
    rid = engine.add_request([1, 2, 3, 4, 5, 6], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    engine.offload_request(rid)
    history = list(engine._requests[rid].token_ids)
    wrapper._sessions.park(rid, history, 0)

    def fail(_request):
        raise RestoreAdmissionError("injected admission failure")

    monkeypatch.setattr(engine.session_offload, "begin_restore", fail)
    cold = wrapper._admit(history + [9], max_tokens=1, sampling=None)
    assert rid not in engine._requests
    assert wrapper._sessions.session_tokens(rid) is None
    engine.finish_request(cold)
    again = wrapper._admit(history + [9], max_tokens=1, sampling=None)
    assert again in engine._requests


def test_apc_restore_counts_only_imported_physical_bytes(monkeypatch):
    engine = _engine(cache=True)
    rid = engine.add_request(list(range(1, 11)), max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    engine.offload_request(rid)
    imported = []
    original = engine.runner.pool.import_blocks

    def record(ids, source, **kwargs):
        imported.append(len(ids) * engine.runner.pool.block_bytes)
        return original(ids, source, **kwargs)

    monkeypatch.setattr(engine.runner.pool, "import_blocks", record)
    engine.resume_request(rid, [20], 1, hold_kv=False)
    assert engine.session_offload.report()["h2d_bytes"] == sum(imported)


def test_active_prefill_reclaims_idle_session_before_preempting_itself(monkeypatch):
    import asyncio

    engine = _engine(blocks=4)
    wrapper = SessionRollout(engine)
    original = engine.step
    steps = 0

    def bounded():
        nonlocal steps
        steps += 1
        assert steps < 8, (
            "active prefill repeatedly preempts itself while paused KV stays pinned"
        )
        return original()

    monkeypatch.setattr(engine, "step", bounded)

    async def run():
        try:
            await wrapper.run_turn([1, 2, 3, 4, 5, 6], max_tokens=1, sampling=None)
            await wrapper.run_turn(list(range(1, 10)), max_tokens=1, sampling=None)
        finally:
            wrapper.stop()

    asyncio.run(run())
    assert engine.scheduler.num_preemptions == 0
    assert engine.session_offload.report()["saved"] >= 1


def test_restore_under_active_pressure_preserves_snapshot_for_retry():
    import pytest

    engine = _engine(blocks=4)
    wrapper = SessionRollout(engine)
    rid = engine.add_request(list(range(1, 10)), max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    engine.offload_request(rid)
    history = list(engine._requests[rid].token_ids)
    wrapper._sessions.park(rid, history, 0)
    active = engine.add_request([1, 2, 3, 4, 5, 6], max_tokens=8)
    engine.step()
    before = wrapper.session_report()
    with pytest.raises(RuntimeError, match="defer"):
        wrapper._admit(history + [20], max_tokens=1, sampling=None)
    after = wrapper.session_report()
    assert {k: v for k, v in after.items() if not k.startswith("offload_")} == {
        k: v for k, v in before.items() if not k.startswith("offload_")
    }
    assert engine.has_cpu_snapshot(rid)
    assert wrapper._sessions.session_tokens(rid) == history
    engine.finish_request(active)
    resumed = wrapper._admit(history + [20], max_tokens=1, sampling=None)
    assert resumed == rid
    assert not engine.has_cpu_snapshot(rid)
    report = wrapper.session_report()
    assert report["turns"] == before["turns"] + 1
    assert report["resumed"] == before["resumed"] + 1
    assert report["tokens_reused"] == before["tokens_reused"] + len(history)


def test_serial_admission_cycles_multiple_offloaded_sessions_to_completion():
    import asyncio

    engine = _engine(blocks=4, cache=True, active=1)
    wrapper = SessionRollout(engine, session_max_blocks=4)

    async def task(index):
        history = [index + 1, 2, 3, 4, 5, 6]
        for suffix in [20, 21, 22]:
            turn = await wrapper.run_turn(history, max_tokens=1, sampling=None)
            history += turn.tokens + [suffix]
        return history

    async def run():
        try:
            return await asyncio.wait_for(asyncio.gather(*(task(i) for i in range(3))), 3)
        finally:
            wrapper.stop()

    assert len(asyncio.run(run())) == 3
    assert engine.session_offload.report()["restored"] > 0


def test_gpu_watermark_never_discards_a_cpu_only_session():
    engine = _engine(blocks=4)
    wrapper = SessionRollout(engine)
    rid = engine.add_request([1, 2, 3, 4, 5, 6], max_tokens=1, hold_kv=True)
    engine.drain_request(rid)
    history = list(engine._requests[rid].token_ids)
    engine.offload_request(rid)
    wrapper._sessions.park(rid, history, 0)
    active = engine.add_request(list(range(1, 15)), max_tokens=2)
    engine.step()
    assert engine.block_manager.num_free_blocks == 0
    wrapper._evict_for_free_watermark()
    assert engine.has_cpu_snapshot(rid), "discarding host KV cannot free GPU blocks"
    assert wrapper._sessions.session_tokens(rid) == history
    engine.finish_request(active)


def test_forced_replay_and_restore_share_turn_accounting_and_cleanup():
    import asyncio

    engine = _engine(blocks=3)
    rollout = SessionRollout(engine, enabled=True, stop_token_ids=(21,))

    async def run():
        try:
            prompt = list(range(1, 10))
            first = await rollout.run_turn(
                prompt, max_tokens=1, sampling=None,
                forced_tokens=[21], ignore_eos=True, stop_token_ids=(),
            )
            assert first.tokens == [21]
            assert engine.has_cpu_snapshot(first.request_id)
            history = prompt + first.tokens + [20]
            second = await rollout.run_turn(
                history, max_tokens=1, sampling=None,
                forced_tokens=[23], ignore_eos=True, stop_token_ids=(),
            )
            assert second.request_id == first.request_id
            assert second.tokens == [23]
            report = rollout.session_report()
            assert report["turns"] == 2
            assert report["started"] == report["resumed"] == 1
            assert report["offload_restored"] == 1
            assert await rollout.finish_session(history + second.tokens) == 1
            assert rollout.session_report()["live_sessions"] == 0
            assert engine.session_offload.snapshot_count == 0
            assert engine.block_manager.num_free_blocks == 3
        finally:
            rollout.clear()

    asyncio.run(run())


def test_restore_fallback_counts_one_cold_turn_through_the_rollout_interface(monkeypatch):
    import asyncio

    engine = _engine(blocks=3)
    rollout = SessionRollout(engine, enabled=True)

    async def run():
        try:
            prompt = list(range(1, 10))
            first = await rollout.run_turn(prompt, max_tokens=1, sampling=None)
            assert engine.has_cpu_snapshot(first.request_id)

            def refuse(_request):
                raise RestoreAdmissionError("injected permanent restore failure")

            monkeypatch.setattr(engine.session_offload, "begin_restore", refuse)
            second = await rollout.run_turn(
                prompt + first.tokens + [20], max_tokens=1, sampling=None,
            )
            assert second.request_id != first.request_id
            assert first.request_id not in engine._requests
            assert not engine.has_cpu_snapshot(first.request_id)
            report = rollout.session_report()
            assert report["turns"] == report["started"] == 2
            assert report["resumed"] == report["tokens_reused"] == 0
            assert report["tokens_prefilled"] == 20
            assert report["live_sessions"] == 1
        finally:
            rollout.clear()

    asyncio.run(run())
