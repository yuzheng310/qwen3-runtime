"""Session KV × APC: parked blocks survive cache eviction; resume does not re-attach."""

from __future__ import annotations

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config


def _engine(model, *, cache: bool, num_blocks: int = 32) -> Engine:
    cfg = Config(
        block_size=4,
        num_kv_blocks=num_blocks,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_prefix_cache=cache,
    )
    return Engine(cfg, PagedRunner(model))


def test_parked_session_blocks_are_not_stolen_by_prefix_cache_eviction():
    """Parked blocks have request+cache refs; `_evict_unused` only takes refcount==1."""
    torch.manual_seed(51)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = _engine(model, cache=True, num_blocks=16)
    prompt = list(range(1, 13))
    rid = engine.add_request(prompt, max_tokens=4, hold_kv=True)
    engine.drain_request(rid)
    req = engine._requests[rid]
    parked = list(req.block_table)
    assert parked
    shared = [b for b in parked if engine.block_manager._ref_count[b] >= 2]
    assert shared, [engine.block_manager._ref_count[b] for b in parked]

    engine.block_manager._evict_unused(engine.block_manager.num_blocks)
    assert req.block_table == parked
    assert all(engine.block_manager._ref_count[b] >= 1 for b in parked)
    assert all(engine.block_manager._ref_count[b] >= 2 for b in shared)


def test_evicting_a_session_leaves_its_full_blocks_as_prefix_cache_entries():
    """Finish (the eviction path) drops the request ref; cache keeps the extra one."""
    torch.manual_seed(52)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = _engine(model, cache=True, num_blocks=16)
    prompt = list(range(1, 13))
    rid = engine.add_request(prompt, max_tokens=4, hold_kv=True)
    engine.drain_request(rid)
    table = list(engine._requests[rid].block_table)
    engine.finish_request(rid)
    assert engine.block_manager.cache_blocks >= 1
    # The pages are still allocated — they live as cache entries until pressure.
    assert engine.block_manager.num_free_blocks < engine.block_manager.num_blocks

    nxt = engine.add_request(prompt, max_tokens=3, hold_kv=False)
    req = engine._requests[nxt]
    assert req.cached_tokens >= 4
    assert req.block_table[: req.cached_tokens // engine.config.block_size] == table[
        : req.cached_tokens // engine.config.block_size
    ]
    engine.drain_request(nxt)


def test_resume_does_not_call_attach_cached_prefix():
    torch.manual_seed(53)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = _engine(model, cache=True)
    prompt = list(range(1, 9))
    rid = engine.add_request(prompt, max_tokens=4, hold_kv=True)
    engine.drain_request(rid)

    calls: list[int] = []
    orig = engine.block_manager.attach_cached_prefix

    def wrapped(req):
        calls.append(req.request_id)
        return orig(req)

    engine.block_manager.attach_cached_prefix = wrapped  # type: ignore[method-assign]
    engine.resume_request(rid, [20, 21], 3, hold_kv=False)
    engine.drain_request(rid)
    assert calls == []


def test_session_plus_apc_emits_the_same_tokens_as_session_only():
    torch.manual_seed(54)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = list(range(1, 13))
    suffix = [20, 21]

    def run(*, cache: bool) -> list[int]:
        torch.manual_seed(54)
        engine = _engine(model, cache=cache)
        rid = engine.add_request(prompt, max_tokens=4, hold_kv=True)
        first = engine.drain_request(rid)
        engine.resume_request(rid, suffix, 3, hold_kv=False)
        second = engine.drain_request(rid)
        return first + second

    assert run(cache=False) == run(cache=True)
