from dataclasses import replace
import torch
from tests.cpu.test_tiered_kv import engine, paused, invariant
from qwen3_runtime.engine.prefix_cache import ROOT_HASH, block_hash


def test_snapshot_recovers_gpu_hit_after_cpu_filled_hole():
    e = engine("snapshot", cache=True)
    e.config = replace(e.config, snapshot_mixed_restore=True)
    r = paused(e, 11)
    p = e.runner.pool
    p.cache.normal_()
    expected = p.export_blocks(r.block_table, valid_tokens=11, chunk_bytes=4096)
    bm = e.block_manager
    bm.publish_full_blocks(r)
    second = r.block_table[1]
    e.session_offload.save(r, session_key="probe")
    h = block_hash(ROOT_HASH, r.token_ids[:4])
    bid = bm._cache.lookup(h)
    bm._cache.drop(h, bid)
    bm._decref(bid)
    t = e.session_offload.begin_restore(r)
    assert r.block_table[1] == second
    assert t.imported_bytes == 2 * p.block_bytes
    got = p.export_blocks(r.block_table, valid_tokens=11, chunk_bytes=4096)
    assert torch.equal(got, expected)
    e.session_offload.commit_restore(t)
    assert not e.session_offload.has_snapshot(r.request_id)
    invariant(e)


import gc
import pytest
from qwen3_runtime.config import Config
from qwen3_runtime.kv.block_store import BlockStore
from qwen3_runtime.kv.cpu_store import CpuKVCapacityError
from qwen3_runtime.engine.session_offload import RestoreAdmissionError


def test_demand_slab_reuses_slots_and_charges_aliases():
    s = BlockStore(320, 0, (2, 1, 4, 1, 1), torch.float32, 256, demand_sized=True)
    lease = s.reserve(range(3))
    assert s.budget.owned == 96
    for h in lease.handles.values():
        s.commit(h, 4)
    lease.close()
    n = s.budget.allocation_count
    s.invalidate_all(release=False)
    lease = s.reserve(range(3))
    assert s.budget.allocation_count == n
    alias = s.view([lease.handles[0]])
    lease.close()
    s.invalidate_all()
    gc.collect()
    assert s.budget.owned == 96
    with pytest.raises(CpuKVCapacityError):
        s.reserve(range(8))
    del alias
    gc.collect()
    assert s.budget.owned == 0


def test_slab_size_does_not_enlarge_transfer_groups():
    s = BlockStore(320, 0, (2, 1, 4, 1, 1), torch.float32, 256, demand_sized=True)
    lease = s.reserve(range(8))
    assert s.budget.allocation_count == 1
    assert list(map(len, s.runs(range(8), max_blocks=2))) == [2, 2, 2, 2]
    lease.close()


def mixed_with_holes(valid=19, blocks=16):
    e = engine("snapshot", cache=True, blocks=blocks)
    e.config = replace(e.config, snapshot_mixed_restore=True)
    r = paused(e, valid)
    e.runner.pool.cache.normal_()
    e.block_manager.publish_full_blocks(r)
    e.session_offload.save(r, session_key="risk")
    parent = ROOT_HASH
    for i in range(valid // 4):
        parent = block_hash(parent, r.token_ids[i * 4 : (i + 1) * 4])
        if i % 2 == 0:
            bid = e.block_manager._cache.lookup(parent)
            e.block_manager._cache.drop(parent, bid)
            e.block_manager._decref(bid)
    return e, r


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_mixed_import_failure_releases_all_refs_and_preserves_snapshot(
    failure_call, monkeypatch
):
    e, r = mixed_with_holes()
    bm = e.block_manager
    refs = list(bm._ref_count)
    snapshot = e.session_offload.snapshot_for(r.request_id)
    expected = snapshot.buffer.clone()
    original = e.runner.pool.import_blocks
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("injected mixed import failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(e.runner.pool, "import_blocks", fail)
    with pytest.raises(RestoreAdmissionError):
        e.session_offload.begin_restore(r)
    assert bm._ref_count == refs
    assert not r.block_table and r.kv_residency == "cpu"
    assert e.session_offload.snapshot_for(r.request_id) is snapshot
    assert torch.equal(snapshot.buffer, expected)
    invariant(e)


def test_mixed_allocator_failure_after_gpu_pins(monkeypatch):
    e, r = mixed_with_holes()
    bm = e.block_manager
    refs = list(bm._ref_count)
    original = bm._alloc_block
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected allocation")
        return original()

    monkeypatch.setattr(bm, "_alloc_block", fail)
    with pytest.raises(RestoreAdmissionError):
        e.session_offload.begin_restore(r)
    assert bm._ref_count == refs and not r.block_table
    invariant(e)


@pytest.mark.parametrize("valid", [0, 1, 3, 4, 5, 8, 11])
def test_mixed_boundaries_and_next_token(valid):
    e = engine("snapshot", cache=True, blocks=16)
    e.config = replace(e.config, snapshot_mixed_restore=True)
    r = paused(e, valid)
    p = e.runner.pool
    p.cache.normal_()
    expected = p.export_blocks(r.block_table, valid_tokens=valid, chunk_bytes=4096)
    e.block_manager.publish_full_blocks(r)
    e.session_offload.save(r, session_key="bounds")
    t = e.session_offload.begin_restore(r)
    assert r.num_computed_tokens == valid
    assert torch.equal(
        p.export_blocks(r.block_table, valid_tokens=valid, chunk_bytes=4096), expected
    )
    e.session_offload.commit_restore(t)
    assert not e.session_offload.has_snapshot(r.request_id)
    invariant(e)


def test_mixed_next_token_capacity_failure_keeps_cpu_copy():
    e = engine("snapshot", cache=True, blocks=2, cpu_blocks=4)
    e.config = replace(e.config, snapshot_mixed_restore=True)
    r = paused(e, 8)
    e.block_manager.publish_full_blocks(r)
    e.session_offload.save(r, session_key="full")
    with pytest.raises(RestoreAdmissionError):
        e.session_offload.begin_restore(r)
    assert not r.block_table and e.session_offload.has_snapshot(r.request_id)
    invariant(e)


def test_configuration_defaults_and_invalid_slab():
    c = Config()
    assert c.cpu_kv_backend == "snapshot" and c.session_cpu_offload == "off"
    assert c.cpu_kv_slab_bytes == 0 and not c.snapshot_mixed_restore
    with pytest.raises(ValueError):
        Config(cpu_kv_slab_bytes=-1)


def test_mixed_gpu_hits_are_protected_during_reclaim():
    e, r = mixed_with_holes(blocks=6)
    bm = e.block_manager
    other = paused(e, 8, tokens=[77] * 9)
    bm.publish_full_blocks(other)
    bm.deallocate(other)
    t = e.session_offload.begin_restore(r)
    assert t.attached_tokens == 8
    assert len(r.block_table) == 5
    e.session_offload.commit_restore(t)
    invariant(e)


def test_mixed_fatal_import_poisoning(monkeypatch):
    e, r = mixed_with_holes()

    def fatal(*args, **kwargs):
        raise RuntimeError("CUDA illegal memory access")

    monkeypatch.setattr(e.runner.pool, "import_blocks", fatal)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        e.session_offload.begin_restore(r)
    assert e._fatal_error and not r.block_table
    with pytest.raises(RuntimeError):
        e.add_request([1], max_tokens=1)
    assert e.session_offload.has_snapshot(r.request_id)
