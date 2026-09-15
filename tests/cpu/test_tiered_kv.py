import gc
import random
from dataclasses import replace

import pytest
import torch

from qwen3_runtime.engine.block_offload import BlockOffloadManager
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.engine.session_offload import (
    RestoreAdmissionError,
    SnapshotStaleError,
)
from qwen3_runtime.kv.block_store import BlockStore
from qwen3_runtime.kv.cpu_store import CpuKVCapacityError
from qwen3_runtime.kv.host_budget import HostBudget
from tests.cpu.test_session_offload import _engine


def engine(backend="block", **kwargs):
    e = _engine(**kwargs)
    e.config = replace(
        e.config,
        cpu_kv_backend=backend,
        transfer_chunk_bytes=4096,
        cpu_kv_max_bytes=e.runner.pool.block_bytes * kwargs.get("cpu_blocks", 16),
    )
    from qwen3_runtime.engine.session_offload import SessionOffloadManager

    e.session_offload = (
        BlockOffloadManager if backend == "block" else SessionOffloadManager
    )(e)
    return e


def paused(e, valid, tokens=None):
    r = Request(token_ids=list(tokens or range(1, valid + 2)), max_tokens=1)
    r.status = RequestStatus.PAUSED
    r.hold_kv = True
    e._requests[r.request_id] = r
    e.block_manager.allocate_for_tokens(r, valid)
    r.num_computed_tokens = valid
    e.scheduler.paused[r.request_id] = r
    return r


def invariant(e):
    b = e.block_manager
    assert len(b._free) == len(set(b._free))
    assert all(n >= 0 for n in b._ref_count)
    assert b.num_free_blocks + sum(n > 0 for n in b._ref_count) == b.num_blocks
    s = e.session_offload.store
    if s:
        st = s.stats()
        assert st["managed_host_buffer_bytes"] <= s.max_bytes
        assert st["managed_pinned_bytes"] <= s.pinned_max_bytes
        assert st["reserved_bytes"] == 0
        assert st.get("transfer_pins", 0) == 0


@pytest.mark.parametrize("valid", [0, 1, 3, 4, 5, 7, 8, 9])
def test_boundaries_roundtrip(valid):
    e = engine()
    r = paused(e, valid)
    p = e.runner.pool
    p.cache.normal_()
    expected = p.export_blocks(r.block_table, valid_tokens=valid, chunk_bytes=4096)
    e.session_offload.save(r, session_key="a")
    ticket = e.session_offload.begin_restore(r)
    e.session_offload.commit_restore(ticket)
    got = p.export_blocks(r.block_table, valid_tokens=valid, chunk_bytes=4096)
    assert torch.equal(got, expected)
    assert r.num_computed_tokens == valid
    invariant(e)


def test_incremental_and_private_tail_overlap():
    e = engine(cpu_blocks=16)
    r = paused(e, 5)
    m = e.session_offload
    m.save(r, session_key="a")
    first = m.stats["d2h_bytes"]
    t = m.begin_restore(r)
    m.commit_restore(t)
    e.block_manager.allocate_for_tokens(r, 4)
    r.token_ids.extend([7, 8, 9, 10])
    r.num_computed_tokens = 9
    m.save(r, session_key="a")
    assert m.stats["d2h_bytes"] - first == 2 * e.runner.pool.block_bytes
    assert m.stats["d2h_skipped_valid_replica_bytes"] == e.runner.pool.block_bytes
    assert sum(len(k) > 2 for k in m.store.index) == 1
    invariant(e)


def test_new_request_never_loads_cpu_but_save_deduplicates():
    e = engine(cache=True)
    a = paused(e, 8)
    m = e.session_offload
    m.save(a, session_key="a")
    keys = m._snapshots[a.request_id].keys
    e.finish_request(a.request_id)
    e.block_manager.reclaim_cached_blocks(e.block_manager.num_blocks)
    b = e.add_request(list(range(1, 12)), max_tokens=1, hold_kv=True)
    assert e._requests[b].num_computed_tokens == 0
    assert m.stats["h2d_bytes"] == 0 and m.stats["new_request_cpu_loads"] == 0
    e.drain_request(b)
    before = m.stats["d2h_bytes"]
    m.save(e._requests[b], session_key="b")
    assert m.stats["d2h_bytes"] - before == e.runner.pool.block_bytes
    assert all(m.store.lookup(k) is not None for k in keys)
    invariant(e)


def test_joint_reservation_and_partial_commit(monkeypatch):
    e = engine(cache=True, cpu_blocks=4)
    a = paused(e, 5)
    e.block_manager.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    e.block_manager.attach_cached_prefix_upto(b, b.token_ids, 4)
    e.block_manager.allocate_for_tokens(b, 1)
    b.num_computed_tokens = 5
    m = e.session_offload
    assert e.block_manager.reclaimable_blocks([a]) == 1
    assert e.block_manager.reclaimable_blocks([a, b]) == 3
    export = e.runner.pool.export_blocks
    calls = []

    def fail(ids, **kw):
        calls.append(ids)
        if len(calls) == 2:
            raise RuntimeError("injected B copy")
        return export(ids, **kw)

    monkeypatch.setattr(e.runner.pool, "export_blocks", fail)
    result = m.reclaim_group([a, b], e.block_manager.num_blocks)
    assert result["partial_commit"] and result["predicted_reclaimed_blocks"] is None
    assert not a.block_table and b.block_table
    assert result["actual_gpu_freed_blocks"] == 1
    assert m.store.lookup(m._snapshots[a.request_id].keys[0])
    invariant(e)


@pytest.mark.parametrize("backend", ["snapshot", "block"])
def test_group_capacity_rejected_before_copy(backend, monkeypatch):
    e = engine(backend, cpu_blocks=2)
    a = paused(e, 5)
    b = paused(e, 5, tokens=[10] * 6)

    def bad(*a, **kw):
        raise AssertionError("copy must not run")

    monkeypatch.setattr(e.runner.pool, "export_blocks", bad)
    r = e.session_offload.reclaim_group([a, b], e.block_manager.num_blocks)
    assert r["failure_reason"] and not r["committed_ids"]
    assert len(a.block_table) == len(b.block_table) == 2
    invariant(e)


def test_interleaved_gpu_cpu_prefix_and_common_hole():
    e = engine(cache=True)
    r = paused(e, 12)
    m = e.session_offload
    b = e.block_manager
    b.publish_full_blocks(r)
    m.save(r, session_key="a")
    manifest = m._snapshots[r.request_id]
    # CPU first, GPU second, CPU third: independently missing layer entries.
    key = manifest.hashes[0]
    bid = b._cache.lookup(key)
    b._cache.drop(key, bid)
    b._decref(bid)
    m.store.evict(manifest.keys[1])
    t = m.begin_restore(r)
    m.commit_restore(t)
    assert r.num_computed_tokens == 12
    m.save(r, session_key="a")
    manifest = m._snapshots[r.request_id]
    m.store.evict(manifest.keys[1])
    key = manifest.hashes[1]
    bid = b._cache.lookup(key)
    if bid is not None:
        b._cache.drop(key, bid)
        b._decref(bid)
    t = m.begin_restore(r)
    m.commit_restore(t)
    assert r.num_computed_tokens == 4
    invariant(e)


def test_stale_handles_pin_capacity_and_alias_ledger():
    b = HostBudget(128)
    t = b.allocate((32,), torch.float32)
    v = t[:1]
    del t
    gc.collect()
    assert b.owned == 128
    with pytest.raises(CpuKVCapacityError):
        b.allocate((1,), torch.float32)
    del v
    gc.collect()
    assert b.owned == 0
    s = BlockStore(128, 0, (2, 1, 4, 1, 1), torch.float32, 128)
    lease = s.reserve(["a", "b"])
    h = lease.handles["a"]
    s.commit(h, 4)
    s.commit(lease.handles["b"], 4)
    assert not s.evict("a")
    lease.close()
    assert s.evict("a")
    lease = s.reserve(["c"])
    s.commit(lease.handles["c"], 4)
    lease.close()
    assert s.resolve(h) is None
    s.invalidate_all()
    assert s.resolve(lease.handles["c"]) is None


@pytest.mark.parametrize("backend", ["snapshot", "block"])
def test_restore_requires_next_token_and_preserves_source(backend):
    e = engine(backend, blocks=2, cpu_blocks=4)
    r = paused(e, 8)
    m = e.session_offload
    m.save(r, session_key="a")
    with pytest.raises(RestoreAdmissionError):
        m.begin_restore(r)
    assert m.has_snapshot(r.request_id) and not r.block_table
    invariant(e)


def test_real_greedy_tiny_matches_held_gpu():
    torch.manual_seed(190)
    a = engine()
    torch.manual_seed(190)
    b = engine()
    prompt = [1, 2, 3, 4, 5, 6]
    ra = a.add_request(prompt, max_tokens=2, hold_kv=True)
    rb = b.add_request(prompt, max_tokens=2, hold_kv=True)
    assert a.drain_request(ra) == b.drain_request(rb)
    a.offload_request(ra)
    for e, r in [(a, ra), (b, rb)]:
        e.resume_request(r, [7, 8], 4, hold_kv=False)
    assert a.drain_request(ra) == b.drain_request(rb)
    assert a.session_offload.stats["useful_restores"] == 1
    invariant(a)


def test_random_reference_cache_sequences():
    # Independent bounded set model (no eviction when not explicitly requested).
    s = BlockStore(1280, 0, (2, 1, 4, 1, 1), torch.float32, 128)
    model = set()
    rng = random.Random(43)
    for _ in range(250):
        action = rng.choice(["save", "load", "evict", "invalidate"])
        k = rng.randrange(12)
        if action == "save":
            lease = s.reserve([k])
            h = lease.handles[k]
            if k not in model:
                s.commit(h, 4)
            lease.close()
            model.add(k)
        elif action == "evict":
            s.evict(k)
            model.discard(k)
        elif action == "invalidate":
            s.invalidate_all()
            model.clear()
        else:
            assert (s.lookup(k) is not None) == (k in model)
        assert {k for k in range(12) if s.lookup(k) is not None} == model
        assert s.stats()["managed_host_buffer_bytes"] <= s.max_bytes
        assert s.stats()["transfer_pins"] == 0


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_chunk_failures_and_metadata_failure_leave_source(failure_at, monkeypatch):
    e = engine(cpu_blocks=32)
    e.config = replace(e.config, transfer_chunk_bytes=e.runner.pool.block_bytes * 2)
    r = paused(e, 20)
    m = e.session_offload
    m.store.slab_blocks = 2
    table = list(r.block_table)
    f = e.runner.pool.export_blocks
    calls = 0

    def fail(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise RuntimeError("injected chunk failure")
        return f(*args, **kw)

    monkeypatch.setattr(e.runner.pool, "export_blocks", fail)
    with pytest.raises(RuntimeError):
        m.save(r, session_key="s")
    assert r.block_table == table and r.num_computed_tokens == 20
    assert not m.has_snapshot(r.request_id)
    invariant(e)


def test_generation_invalidate_and_failed_weight_update():
    e = engine()
    r = paused(e, 8)
    m = e.session_offload
    m.save(r, session_key="s")
    t = m.begin_restore(r)
    m.invalidate_all(weight_change=True)
    with pytest.raises(SnapshotStaleError):
        m.commit_restore(t)
    assert m.store.stats()["ready_blocks"] == 0
    e.invalidate_all_kv()
    invariant(e)


def test_cpu_eviction_does_not_erase_live_gpu_progress():
    e = engine()
    r = paused(e, 9)
    m = e.session_offload
    m.save(r, session_key="s")
    t = m.begin_restore(r)
    m.commit_restore(t)
    table = list(r.block_table)
    for key in tuple(m.store.index):
        m.store.evict(key)
    assert r.block_table == table and r.num_computed_tokens == 9
    e.resume_request(r.request_id, [10], 1, hold_kv=True)
    e.drain_request(r.request_id)
    invariant(e)


def test_snapshot_stale_open_reservation_cannot_commit_after_clear():
    from qwen3_runtime.kv.cpu_store import CpuKVStore, CpuKVTransactionError
    from tests.cpu.test_cpu_kv_store import _meta

    shape = (2, 1, 1, 4, 1, 2)
    s = CpuKVStore(64)
    r = s.reserve(_meta(1, shape, 64))
    s.invalidate_all()
    with pytest.raises(CpuKVTransactionError):
        r.commit()
    assert (
        s.stats()["reserved_bytes"] == 0
        and s.stats()["managed_host_buffer_bytes"] == 64
    )
    del r
    assert s.stats()["managed_host_buffer_bytes"] == 0


def test_rollout_abort_cleans_cpu_gpu_and_claimed_sessions():
    from qwen3_runtime.rollout.execution import SessionRollout

    e = engine()
    w = SessionRollout(e)
    a = paused(e, 5)
    b = paused(e, 5, tokens=[21] * 6)
    for r in [a, b]:
        w._sessions.park(r.request_id, r.token_ids, len(r.block_table))
    e.session_offload.save(a, session_key="s")
    w._sessions.update_blocks(a.request_id, 0)
    w._sessions.reserve_claim(b.token_ids + [22])
    w.abort()
    assert (
        not e._requests and not w._sessions._claims and not e.session_offload._snapshots
    )
    invariant(e)


def test_group_fatal_device_error_is_not_recovered(monkeypatch):
    e = engine()
    r = paused(e, 5)

    def fatal(*args, **kw):
        raise RuntimeError("CUDA error: illegal memory access")

    monkeypatch.setattr(e.runner.pool, "export_blocks", fatal)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        e.session_offload.reclaim_group([r], e.block_manager.num_blocks)
    assert r.block_table
    invariant(e)


def test_fatal_device_error_poisons_engine(monkeypatch):
    e = engine()
    r = paused(e, 5)

    def fatal(*args, **kw):
        raise RuntimeError("CUDA error: illegal memory access")

    monkeypatch.setattr(e.runner.pool, "export_blocks", fatal)
    with pytest.raises(RuntimeError):
        e.session_offload.save(r, session_key="s")
    with pytest.raises(RuntimeError, match="engine terminated"):
        e.add_request([1, 2], max_tokens=1)
    with pytest.raises(RuntimeError, match="engine terminated"):
        e.step()
    e.finish_request(r.request_id)
    invariant(e)
