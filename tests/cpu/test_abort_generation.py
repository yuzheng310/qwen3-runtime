from qwen3_runtime.engine.request import RequestStatus
from tests.cpu.test_engine import _engine


def test_abort_mid_trajectory_frees_blocks_and_drops_the_request():
    engine = _engine(num_blocks=16, block_size=4)
    rid = engine.add_request([1, 2, 3, 4], max_tokens=8)
    engine.step()
    req = engine._requests[rid]
    assert req.status != RequestStatus.FINISHED
    aborted = engine.abort_generation()
    assert rid in aborted
    assert rid not in engine._requests
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    assert engine.scheduler.is_finished()


def test_abort_does_not_leave_paused_peer_on_freed_ids():
    engine = _engine(num_blocks=32, block_size=4)
    paused = engine.add_request([1, 2, 3], max_tokens=2, hold_kv=True)
    engine.drain_request(paused)
    held_blocks = list(engine._requests[paused].block_table)
    assert held_blocks
    running = engine.add_request([4, 5, 6, 7], max_tokens=8)
    engine.step()
    engine.abort_generation()
    assert running not in engine._requests
    still = engine._requests[paused]
    assert still.block_table == held_blocks
    for bid in still.block_table:
        assert engine.block_manager._ref_count[bid] > 0
    engine.finish_request(paused)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
