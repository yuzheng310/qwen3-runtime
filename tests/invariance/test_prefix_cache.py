"""Prefix-cache invariance: cold, warm, and shared-prefix paths match the uncached path."""

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.sampling import SamplingParams
from tests.cpu.test_tiny_qwen3 import tiny_config


def _engine(model, *, cache: bool) -> Engine:
    cfg = Config(
        block_size=4,
        num_kv_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_prefix_cache=cache,
    )
    return Engine(cfg, PagedRunner(model))


def test_same_prompt_cold_and_warm_match_uncached():
    torch.manual_seed(31)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = list(range(1, 13))
    uncached = _engine(model, cache=False).generate(prompt, max_tokens=4)
    eng = _engine(model, cache=True)
    cold = eng.generate(prompt, max_tokens=4)
    warm = eng.generate(prompt, max_tokens=4)
    assert cold == uncached
    assert warm == uncached
    assert eng.block_manager.cache_blocks >= 1


def test_shared_prefix_matches_uncached_path():
    torch.manual_seed(32)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    a = prefix + [9, 10]
    b = prefix + [11, 12]
    uncached_a = _engine(model, cache=False).generate(a, max_tokens=3)
    uncached_b = _engine(model, cache=False).generate(b, max_tokens=3)
    eng = _engine(model, cache=True)
    got_a = eng.generate(a, max_tokens=3)
    got_b = eng.generate(b, max_tokens=3)
    assert got_a == uncached_a
    assert got_b == uncached_b
    rid = eng.add_request(b, max_tokens=3)
    req = eng._requests[rid]
    assert req.cached_tokens == 8


def test_seeded_hold_kv_output_unchanged_with_prefix_cache():
    """APC is a memory optimization; a fixed seed must not move sampled tokens."""
    torch.manual_seed(41)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = list(range(1, 13))
    def run(*, cache: bool) -> list[int]:
        eng = _engine(model, cache=cache)
        params = SamplingParams(temperature=0.8, top_p=0.9, top_k=8, seed=7)
        rid = eng.add_request(prompt, max_tokens=4, hold_kv=True, sampling=params)
        return eng.drain_request(rid)

    assert run(cache=False) == run(cache=True)


def test_cold_batch_duplicate_remains_reusable_after_first_copy_eviction():
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4, 5]
    eng = _engine(model, cache=True)
    a, b = [eng.add_request(prompt.copy(), max_tokens=1, hold_kv=True) for _ in range(2)]
    eng.step()
    bm = eng.block_manager
    a_page = eng._requests[a].block_table[0]
    b_table = eng._requests[b].block_table.copy()
    assert a_page != b_table[0]
    eng.finish_request(a)
    bm.reclaim_cached_blocks(bm.num_free_blocks + 1)
    assert bm._ref_count[a_page] == 0
    c = eng.add_request(prompt.copy(), max_tokens=3)
    assert eng._requests[c].block_table == [b_table[0]]
    assert eng._requests[c].cached_tokens == 4
    assert eng._requests[b].block_table == b_table
    got = eng.drain_request(c)
    expected = _engine(model, cache=False).generate(prompt.copy(), max_tokens=3)
    assert got == expected
    eng.finish_request(b)
    bm.reclaim_cached_blocks(bm.num_blocks)
    assert bm.num_free_blocks == bm.num_blocks
    assert bm.cache_blocks == 0
