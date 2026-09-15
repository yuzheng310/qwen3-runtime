"""Physical parked-page accounting through the real retirement call site."""

import pytest

from qwen3_runtime.rollout.execution import SessionRollout
from qwen3_runtime.rollout.session import SessionCache
from tests.cpu.test_tiered_kv import engine, invariant, paused


def park(roll, req):
    roll._pending_prompt[req.request_id] = list(req.token_ids)
    roll._park_or_release(req.request_id, [])


@pytest.mark.parametrize("backend", ["snapshot", "block"])
def test_shared_pages_do_not_destroy_session_without_freeing_memory(backend):
    e = engine(backend, cache=True, blocks=3, cpu_blocks=16)
    a = paused(e, 8)
    bm = e.block_manager
    bm.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    bm.attach_cached_prefix_upto(b, b.token_ids, 8)
    b.num_computed_tokens = 8
    assert a.block_table == b.block_table
    roll = SessionRollout(e)
    try:
        park(roll, a)
        park(roll, b)
        assert a.request_id in e._requests
        assert b.request_id in e._requests
        assert roll.session_report()["held_blocks"] == 2
        assert roll.session_report()["evicted"] == 0
        assert bm.num_free_blocks == 1
        assert e.session_offload.stats["saved"] == 0
        claim = roll._sessions.reserve_claim(a.token_ids + [99])
        assert claim is not None and claim.request_id == a.request_id
        roll._sessions.rollback_claim(claim)
        invariant(e)
    finally:
        roll.clear()


def test_park_does_not_evict_a_reserved_claim():
    cache = SessionCache(max_blocks=1, max_sessions=1)
    cache.park(1, [1, 2], 1)
    claim = cache.reserve_claim([1, 2, 3])
    assert claim is not None
    assert cache.park(2, [4, 5], 1) == [2]
    cache.commit_claim(claim)
    assert cache.stats["resumed"] == 1


@pytest.mark.parametrize("backend", ["snapshot", "block"])
@pytest.mark.parametrize("cpu_blocks", [1, 16])
def test_real_pressure_reclaims_shared_group_and_preserves_active_owner(backend, cpu_blocks):
    e = engine(backend, cache=True, blocks=3, cpu_blocks=cpu_blocks)
    a = paused(e, 8)
    bm = e.block_manager
    bm.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    bm.attach_cached_prefix_upto(b, b.token_ids, 8)
    b.num_computed_tokens = 8
    roll = SessionRollout(e)
    try:
        park(roll, a)
        park(roll, b)
        # A request outside the parked set owns the only remaining page.
        active = paused(e, 4, tokens=[20, 21, 22, 23, 24])
        active_table = list(active.block_table)
        assert bm.num_free_blocks == 0
        roll._reclaim_for_active_step()
        assert bm.num_free_blocks >= 1
        assert active.block_table == active_table
        assert a.request_id in e._requests and b.request_id in e._requests
        assert not a.block_table and not b.block_table
        assert roll.session_report()["held_blocks"] == 0
        assert roll.session_report()["evicted"] == 0
        if cpu_blocks == 1:
            assert a.kv_residency == b.kv_residency == "none"
            assert not e.has_cpu_snapshot(a.request_id)
            assert not e.has_cpu_snapshot(b.request_id)
        else:
            assert a.kv_residency == b.kv_residency == "cpu"
            assert e.has_cpu_snapshot(a.request_id)
            assert e.has_cpu_snapshot(b.request_id)
        invariant(e)
        e.finish_request(active.request_id)
    finally:
        roll.clear()


def test_unique_page_limit_still_retires_oldest_session():
    e = engine(blocks=8)
    a = paused(e, 4)
    b = paused(e, 4, tokens=[20, 21, 22, 23, 24])
    roll = SessionRollout(e, session_max_blocks=1, session_max_sessions=8)
    try:
        park(roll, a)
        park(roll, b)
        assert a.request_id not in e._requests
        assert b.request_id in e._requests
        assert roll.session_report()["held_blocks"] == 1
        assert roll.session_report()["evicted"] == 1
        invariant(e)
    finally:
        roll.clear()


def test_shared_sessions_still_obey_metadata_count_limit():
    e = engine(cache=True, blocks=3)
    a = paused(e, 8)
    e.block_manager.publish_full_blocks(a)
    b = paused(e, 0, tokens=a.token_ids)
    e.block_manager.attach_cached_prefix_upto(b, b.token_ids, 8)
    roll = SessionRollout(e, session_max_sessions=1)
    try:
        park(roll, a)
        park(roll, b)
        assert a.request_id not in e._requests
        assert b.request_id in e._requests
        assert roll.session_report()["live_sessions"] == 1
        assert roll.session_report()["held_blocks"] == 2
        invariant(e)
    finally:
        roll.clear()


def test_current_page_tables_override_stale_park_counts():
    tables = {1: [4, 5], 2: [4, 5]}
    cache = SessionCache(max_blocks=3, block_ids_for=tables.__getitem__)
    cache.park(1, [1], 2)
    cache.park(2, [2], 2)
    assert cache.report()["held_blocks"] == 2
    tables[1] = []
    tables[2] = [8]  # e.g. restore/rebinding; no copied page-id cache to invalidate.
    assert cache.report()["held_blocks"] == 1
