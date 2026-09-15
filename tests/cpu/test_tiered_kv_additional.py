"""Post-matrix, CPU-only additional contract checks; not timing workloads."""

import random
import pytest
import torch
from tests.cpu.test_tiered_kv import engine, paused, invariant
from qwen3_runtime.engine.session_offload import RestoreAdmissionError


def test_random_session_manifest_reference():
    rng = random.Random(9142026)
    e = engine(cpu_blocks=128, blocks=64)
    m = e.session_offload
    live = {}
    reference_cpu = {}
    manifests = {}
    next_seed = 100
    for step in range(300):
        op = rng.choice(["create", "save", "restore", "evict", "finish", "invalidate"])
        if op == "create" and len(live) < 4:
            valid = rng.choice([1, 3, 4, 5, 7, 8, 9])
            r = paused(e, valid, tokens=[next_seed] * (valid + 1))
            next_seed += 1
            for bid in r.block_table:
                e.runner.pool.cache[:, :, bid].fill_(float(r.token_ids[0]))
            live[r.request_id] = r
        elif op == "save" and live:
            r = rng.choice(list(live.values()))
            if not r.block_table:
                continue
            before = r.num_computed_tokens
            token = float(r.token_ids[0])
            m.save(r, session_key=str(r.request_id))
            manifest = m._snapshots[r.request_id]
            # Reference data validity uses independently saved lengths/values;
            # no store state is used to derive expected restored prefix length.
            manifests[r.request_id] = (tuple(manifest.keys), before, token)
            for k in manifest.keys:
                reference_cpu[k] = token
            # A successful replacement retires only its previous private tail.
            current_tails = {
                k for keys, _, _ in manifests.values() for k in keys if len(k) > 2
            }
            reference_cpu = {
                k: v
                for k, v in reference_cpu.items()
                if len(k) == 2 or k in current_tails
            }
        elif op == "restore" and manifests:
            rid = rng.choice(list(manifests))
            r = live[rid]
            if r.block_table:
                continue
            keys, valid, value = manifests[rid]
            prefix = 0
            for k in keys:
                if k not in reference_cpu:
                    break
                prefix += 4
            prefix = min(prefix, valid)
            t = m.begin_restore(r)
            m.commit_restore(t)
            assert r.num_computed_tokens == prefix
            if prefix:
                data = e.runner.pool.export_blocks(
                    r.block_table, valid_tokens=prefix, chunk_bytes=4096
                )
                flat = data.permute(0, 1, 2, 3, 4, 5).flatten(2, 3)
                assert torch.all(flat[:, :, :prefix] == value)
        elif op == "evict" and reference_cpu:
            k = rng.choice(list(reference_cpu))
            assert m.store.evict(k)
            reference_cpu.pop(k)
        elif op == "finish" and live:
            rid = rng.choice(list(live))
            e.finish_request(rid)
            live.pop(rid)
            old = manifests.pop(rid, None)
            if old:
                for k in old[0]:
                    if len(k) > 2:
                        reference_cpu.pop(k, None)
        elif op == "invalidate":
            e.invalidate_all_kv()
            reference_cpu.clear()
            manifests.clear()
            # Existing histories remain valid; this reference restarts their
            # physical test contexts explicitly after global invalidation.
            for rid in list(live):
                e.finish_request(rid)
            live.clear()
        invariant(e)
        for k in reference_cpu:
            assert m.store.lookup(k) is not None
    for rid in list(live):
        e.finish_request(rid)
    e.invalidate_all_kv()
    invariant(e)


def test_metadata_commit_failure_keeps_gpu_source(monkeypatch):
    e = engine()
    r = paused(e, 9)
    m = e.session_offload
    table = list(r.block_table)

    def fail(*args, **kw):
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(m.store, "commit", fail)
    with pytest.raises(RuntimeError):
        m.save(r, session_key="s")
    assert table == r.block_table and r.num_computed_tokens == 9
    assert not m._snapshots
    invariant(e)


def test_restore_allocation_failure_keeps_cpu_source(monkeypatch):
    e = engine()
    r = paused(e, 9)
    m = e.session_offload
    m.save(r, session_key="s")
    original = e.block_manager._alloc_block
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return original()

    monkeypatch.setattr(e.block_manager, "_alloc_block", fail)
    with pytest.raises(RestoreAdmissionError):
        m.begin_restore(r)
    assert not r.block_table and m.has_snapshot(r.request_id)
    invariant(e)


def test_finish_does_not_delete_another_manifest_prefix():
    e = engine()
    a = paused(e, 8)
    b = paused(e, 8)
    m = e.session_offload
    m.reclaim_group([a, b], e.block_manager.num_blocks)
    keys = m._snapshots[b.request_id].keys
    e.finish_request(a.request_id)
    assert all(m.store.lookup(k) for k in keys)
    t = m.begin_restore(b)
    m.commit_restore(t)
    assert b.num_computed_tokens == 8
    invariant(e)


def test_private_tail_overlap_is_reserved_before_replacement():
    from qwen3_runtime.kv.cpu_store import CpuKVCapacityError

    e = engine(cpu_blocks=2)
    r = paused(e, 5)
    m = e.session_offload
    m.save(r, session_key="s")
    old = m._snapshots[r.request_id]
    t = m.begin_restore(r)
    m.commit_restore(t)
    e.block_manager.allocate_for_tokens(r, 1)
    r.num_computed_tokens = 6
    r.token_ids.append(7)
    table = list(r.block_table)
    with pytest.raises(CpuKVCapacityError):
        m.save(r, session_key="s")
    assert m._snapshots[r.request_id] is old and r.block_table == table
    assert all(m.store.lookup(k) for k in old.keys)
    invariant(e)


def test_two_zero_individual_reclaims_succeed_as_one_group():
    e = engine(cache=True)
    a = paused(e, 4)
    bm = e.block_manager
    bm.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    bm.attach_cached_prefix_upto(b, b.token_ids, 4)
    assert bm.reclaimable_blocks([a]) == bm.reclaimable_blocks([b]) == 0
    assert bm.reclaimable_blocks([a, b]) == 1
    result = e.session_offload.reclaim_group([a, b], bm.num_blocks)
    assert result["actual_gpu_freed_blocks"] == 1 and result["goal_satisfied"]
    assert e.session_offload.stats["d2h_bytes"] == e.runner.pool.block_bytes
    invariant(e)


@pytest.mark.parametrize("backend", ["snapshot", "block"])
@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_restore_first_middle_last_chunk_failure(backend, failure_at, monkeypatch):
    from dataclasses import replace

    e = engine(backend)
    r = paused(e, 9)
    m = e.session_offload
    e.config = replace(e.config, transfer_chunk_bytes=e.runner.pool.block_bytes)
    m.save(r, session_key="s")
    original = torch.Tensor.index_copy_
    calls = 0
    target = e.runner.pool.cache.untyped_storage().data_ptr()

    def fail(tensor, *args, **kwargs):
        nonlocal calls
        if tensor.untyped_storage().data_ptr() == target:
            calls += 1
            if calls == failure_at:
                raise RuntimeError("injected H2D chunk failure")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_copy_", fail)
    with pytest.raises(RestoreAdmissionError):
        m.begin_restore(r)
    assert calls == failure_at and not r.block_table and m.has_snapshot(r.request_id)
    invariant(e)


def test_one_pressure_episode_does_not_repeat_failed_group(monkeypatch):
    from qwen3_runtime.rollout.execution import SessionRollout

    e = engine(cache=True, blocks=4, cpu_blocks=16)
    w = SessionRollout(e)
    a = paused(e, 8)
    e.block_manager.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    e.block_manager.attach_cached_prefix_upto(b, b.token_ids, 8)
    for r in [a, b]:
        w._sessions.park(r.request_id, r.token_ids, len(r.block_table))
    exports = []

    def fail(ids, **kw):
        exports.append(list(ids))
        raise RuntimeError("recoverable pressure injection")

    monkeypatch.setattr(e.runner.pool, "export_blocks", fail)
    w._evict_for_free_watermark(min_free=4)
    assert len(exports) == 1
    assert a.token_ids and b.token_ids and len(w._sessions._sessions) == 2
    assert not a.block_table and not b.block_table
    w._evict_for_free_watermark(min_free=4)
    assert len(exports) == 1
    invariant(e)


def test_pinned_subbudget_falls_back_without_exceeding_managed_budget():
    from qwen3_runtime.kv.host_budget import HostBudget

    if not torch.cuda.is_available():
        pytest.skip("real pin-memory allocator requires CUDA")
    b = HostBudget(128, 64)
    a = b.allocate((16,), torch.float32)
    c = b.allocate((16,), torch.float32)
    assert a.is_pinned() and not c.is_pinned()
    assert b.owned == 128 and b.pinned == 64
    del a, c
    assert b.owned == b.pinned == 0


def test_snapshot_gpu_only_restore_is_not_cpu_write_use():
    e = engine("snapshot", cache=True)
    r = paused(e, 8)
    m = e.session_offload
    e.block_manager.publish_full_blocks(r)
    m.save(r, session_key="s")
    t = m.begin_restore(r)
    assert t.imported_bytes == 0
    m.commit_restore(t)
    m.mark_used([r])  # owner forward-completion hook, exercised as a unit boundary
    assert m.report()["restored_d2h_bytes"] == 0
    assert m.report()["unused_d2h_bytes"] == m.stats["d2h_bytes"]


def test_group_completed_copy_is_not_a_committed_cpu_copy(monkeypatch):
    e = engine()
    r = paused(e, 8)
    m = e.session_offload

    def fail(*args, **kw):
        raise RuntimeError("metadata commit rejected")

    monkeypatch.setattr(m.store, "commit", fail)
    result = m.reclaim_group([r], e.block_manager.num_blocks)
    assert result["cpu_committed_bytes"] == 0
    assert e.runner.pool.transfer_stats["d2h_completed_bytes"] > 0
    assert not result["committed_ids"] and r.block_table
    invariant(e)
