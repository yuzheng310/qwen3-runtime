from dataclasses import replace
import pytest
import torch
from qwen3_runtime.engine.async_offload import AsyncSnapshotOffloadManager
from qwen3_runtime.rollout.execution import SessionRollout
from tests.cpu.test_tiered_kv import engine, paused, invariant


class FakeCopy:
    def __init__(self, e, req, reservation):
        self.e, self.req, self.reservation = e, req, reservation
        self.queued_bytes = reservation.metadata.allocated_bytes
        self.done = False
        self.waits = 0
        self.fail_start = False
        self.fail_finish = False
    def start(self):
        self.e.runner.pool.export_blocks(self.req.block_table,
            valid_tokens=self.req.num_computed_tokens, chunk_bytes=4096,
            destination=self.reservation.buffer)
        if self.fail_start:
            raise RuntimeError("injected partial launch")
    def finish(self, *, wait):
        if self.fail_finish:
            raise RuntimeError("injected uncertain completion")
        if wait:
            self.waits += 1
            self.done = True
        return self.done
    def elapsed_ms(self):
        return 1.0


def setup(monkeypatch, **kwargs):
    e = engine("snapshot", **kwargs)
    e.config = replace(e.config, session_cpu_offload="async", session_offload_early_fraction=.2)
    e.session_offload = m = AsyncSnapshotOffloadManager(e)
    monkeypatch.setattr(m, "_can_async", lambda reservation: True)
    monkeypatch.setattr(m, "_make_transfer", lambda req, reservation: FakeCopy(e, req, reservation))
    return e, m


def test_unfinished_copy_keeps_source_and_uncommitted_host_owner(monkeypatch):
    e, m = setup(monkeypatch)
    r = paused(e, 5)
    table = list(r.block_table)
    free = e.block_manager.num_free_blocks
    assert m.enqueue_save(r, session_key="a")
    assert not m.enqueue_save(r, session_key="duplicate")
    assert m.poll() == []
    assert r.block_table == table and e.block_manager.num_free_blocks == free
    assert not m.has_snapshot(r.request_id)
    assert m.store.stats()["reserved_bytes"] > 0
    m.pending.transfer.done = True
    assert m.poll() == [r.request_id]
    assert r.kv_residency == "cpu" and not r.block_table
    assert m.store.stats()["reserved_bytes"] == 0
    assert m.stats["d2h_bytes"] == m.stats["cpu_committed_write_bytes"]
    invariant(e)


def test_shared_active_owner_remains_valid_after_async_commit(monkeypatch):
    e, m = setup(monkeypatch, cache=True)
    a = paused(e, 8)
    e.block_manager.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    e.block_manager.attach_cached_prefix_upto(b, b.token_ids, 8)
    table = list(b.block_table)
    m.enqueue_save(a, session_key="a")
    m.poll(wait=True)
    assert b.block_table == table
    assert all(e.block_manager._ref_count[i] > 0 for i in table)
    invariant(e)


@pytest.mark.parametrize("action", ["finish", "invalidate", "weights", "abort"])
def test_cancel_waits_before_releasing_source_or_host_storage(monkeypatch, action):
    e, m = setup(monkeypatch)
    r = paused(e, 5)
    m.enqueue_save(r, session_key="a")
    transfer = m.pending.transfer
    if action == "finish": e.finish_request(r.request_id)
    elif action == "invalidate": e.invalidate_all_kv()
    elif action == "abort": assert r.request_id in e.abort_generation()
    else: m.invalidate_all(weight_change=True)
    assert transfer.waits == 1 and m.pending is None
    assert m.store.stats()["reserved_bytes"] == 0
    assert m.stats["cpu_committed_write_bytes"] == 0
    if action == "weights":
        assert r.block_table and m.weight_epoch == 1
    else:
        assert e.block_manager.num_free_blocks == e.block_manager.num_blocks
    invariant(e)


def test_resume_joins_pending_save_before_mutating_request(monkeypatch):
    e, m = setup(monkeypatch)
    r = paused(e, 5)
    m.enqueue_save(r, session_key="a")
    transfer = m.pending.transfer
    e.resume_request(r.request_id, [20], 1, hold_kv=True)
    assert transfer.waits == 1 and m.pending is None
    assert r.block_table and m.stats["restored"] == 1
    invariant(e)


def test_partial_launch_failure_drains_before_aborting(monkeypatch):
    e, m = setup(monkeypatch)
    r = paused(e, 5)
    def make(req, reservation):
        t = FakeCopy(e, req, reservation)
        t.fail_start = True
        return t
    monkeypatch.setattr(m, "_make_transfer", make)
    with pytest.raises(RuntimeError, match="partial launch"):
        m.enqueue_save(r, session_key="a")
    assert m.pending is None and r.block_table
    assert m.store.stats()["reserved_bytes"] == 0
    assert m.stats["save_copy_failed"] == 1
    invariant(e)


def test_uncertain_completion_poison_retains_every_owner(monkeypatch):
    e, m = setup(monkeypatch)
    r = paused(e, 5)
    m.enqueue_save(r, session_key="a")
    m.pending.transfer.fail_finish = True
    with pytest.raises(RuntimeError, match="uncertain completion"):
        m.poll(wait=True)
    assert e._fatal_error and m.pending is not None and r.block_table
    assert m.store.stats()["reserved_bytes"] > 0
    # Model device recovery in this fake only, allowing test teardown.
    m.pending.transfer.fail_finish = False
    m.poll(wait=True, cancel=True)


def test_changed_history_or_epoch_aborts_without_releasing_current_pages(monkeypatch):
    e, m = setup(monkeypatch)
    r = paused(e, 5)
    m.enqueue_save(r, session_key="a")
    r.token_ids.append(90)
    m.poll(wait=True)
    assert r.block_table and m.stats["async_stale"] == 1
    assert not m.has_snapshot(r.request_id)
    invariant(e)


def test_cpu_or_pageable_fallback_is_explicitly_synchronous(monkeypatch):
    e, m = setup(monkeypatch)
    monkeypatch.setattr(m, "_can_async", lambda reservation: False)
    r = paused(e, 5)
    assert m.enqueue_save(r, session_key="a") is False
    assert m.pending is None and r.kv_residency == "cpu"
    assert m.stats["async_fallbacks"] == 1
    invariant(e)


def test_capacity_rejection_keeps_existing_snapshots(monkeypatch):
    e, m = setup(monkeypatch, cpu_blocks=2)
    a = paused(e, 5)
    m.save(a, session_key="a")
    b = paused(e, 5, tokens=[20,21,22,23,24,25])
    assert not m.enqueue_save(b, session_key="b")
    assert m.has_snapshot(a.request_id) and b.block_table
    invariant(e)


def test_early_save_then_hard_pressure_wait_is_bounded(monkeypatch):
    e, m = setup(monkeypatch, blocks=10)
    r = paused(e, 32)
    roll = SessionRollout(e)
    roll._sessions.park(r.request_id, r.token_ids, len(r.block_table))
    active = paused(e, 4, tokens=[60,61,62,63,64])
    roll._reclaim_for_active_step()
    assert m.pending is not None and r.block_table
    transfer = m.pending.transfer
    roll._evict_for_free_watermark(min_free=2)
    assert transfer.waits == 1 and m.pending is None
    assert active.block_table and not r.block_table
    invariant(e)
    roll.clear()
