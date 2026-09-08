"""§5.1 sleep/wake: never resume onto a stale block table."""

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_engine import _engine
from tests.cpu.test_session_kv import tiny_config
import pytest
import torch


def test_sleep_empties_block_tables_and_returns_pages():
    engine = _engine(num_blocks=16, block_size=4)
    rid = engine.add_request([1, 2, 3, 4], max_tokens=3, hold_kv=True)
    engine.drain_request(rid)
    req = engine._requests[rid]
    assert req.block_table
    free_after_hold = engine.block_manager.num_free_blocks
    assert free_after_hold < engine.block_manager.num_blocks

    engine.sleep(level=1)
    assert req.block_table == []
    assert req.num_computed_tokens == 0
    assert req.status == RequestStatus.PAUSED
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    assert engine._asleep


def test_resume_after_sleep_without_wake_is_refused():
    engine = _engine()
    rid = engine.add_request([1, 2, 3], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    engine.sleep()
    with pytest.raises(RuntimeError, match="asleep"):
        engine.resume_request(rid, [9], 2, hold_kv=False)


def test_stale_block_table_after_sleep_fails_resume():
    """Gate: silently keeping physical ids after rebuild must not resume."""
    engine = _engine(num_blocks=16, block_size=4)
    rid = engine.add_request([1, 2, 3, 4], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    stale = [0, 1]
    engine.sleep()
    engine.wake_up()
    req = engine._requests[rid]
    req.block_table = list(stale)
    req.num_computed_tokens = 0
    with pytest.raises(RuntimeError, match="block table after KV invalidation"):
        engine.resume_request(rid, [8, 9], 2, hold_kv=False)


def test_sleep_wake_restart_matches_full_prefill_paged():
    torch.manual_seed(41)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4]
    suffix = [5, 6]
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)

    first = Engine(cfg, PagedRunner(model))
    rid = first.add_request(prompt, max_tokens=3, hold_kv=True)
    gen1 = first.drain_request(rid)
    first.sleep()
    first.wake_up()
    first.resume_request(rid, suffix, 2, hold_kv=False)
    gen2 = first.drain_request(rid)

    scratch = Engine(cfg, PagedRunner(model)).generate(prompt + gen1 + suffix, max_tokens=2)
    assert gen2 == scratch
    assert rid not in first._requests


def test_weights_only_wake_does_not_rebuild_kv_pool():
    from qwen3_runtime.rollout.lifecycle import wake_wants_kv

    assert wake_wants_kv(None) is True
    assert wake_wants_kv(["kv"]) is True
    assert wake_wants_kv(["kv_cache"]) is True
    assert wake_wants_kv(["weights", "kv_cache"]) is True
    assert wake_wants_kv(["weights"]) is False

    model = Qwen3ForCausalLM(tiny_config()).eval()
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    engine = Engine(cfg, PagedRunner(model))
    rid = engine.add_request([1, 2, 3, 4], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    engine.sleep()
    assert engine.runner.pool is None
    engine.wake_up(tags=["weights"])
    assert engine._asleep
    assert engine.runner.pool is None
    with pytest.raises(RuntimeError, match="asleep"):
        engine.resume_request(rid, [9], 2, hold_kv=False)
    engine.wake_up(tags=["kv_cache"])
    assert not engine._asleep
    assert engine.runner.pool is not None
