"""Prefix cache: share full blocks by content hash. Default off."""

from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.engine.prefix_cache import ROOT_HASH, PrefixCache, block_hash
from qwen3_runtime.engine.request import Request
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config
import torch
import pytest


def test_block_hash_is_stable_and_parent_sensitive():
    a = block_hash(ROOT_HASH, [1, 2, 3, 4])
    b = block_hash(ROOT_HASH, [1, 2, 3, 4])
    c = block_hash(ROOT_HASH, [1, 2, 3, 5])
    d = block_hash(a, [1, 2, 3, 4])
    assert a == b
    assert a != c
    assert a != d


def test_prefix_cache_off_does_not_share_blocks():
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=False)
    req = Request(token_ids=[1, 2, 3, 4, 5, 6, 7, 8], max_tokens=1)
    bm.allocate_for_tokens(req, 8)
    req.num_computed_tokens = 8
    bm.publish_full_blocks(req)
    assert bm.cache_blocks == 0
    bm.deallocate(req)
    assert bm.num_free_blocks == 8


def test_second_request_reuses_full_prefix_blocks():
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=True)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    first = Request(token_ids=list(prompt), max_tokens=1)
    bm.allocate_for_tokens(first, 8)
    first.num_computed_tokens = 8
    bm.publish_full_blocks(first)
    table = list(first.block_table)
    free_after_publish = bm.num_free_blocks
    bm.deallocate(first)
    # request dropped its refs; cache keeps the two full blocks
    assert bm.num_free_blocks == 6
    assert free_after_publish == 6

    second = Request(token_ids=list(prompt), max_tokens=1)
    hit = bm.attach_cached_prefix(second)
    assert hit == 8
    assert second.num_computed_tokens == 8
    assert second.cached_tokens == 8
    assert second.block_table == table


def test_eviction_frees_cache_only_blocks():
    bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
    req = Request(token_ids=list(range(1, 17)), max_tokens=1)
    bm.allocate_for_tokens(req, 16)
    req.num_computed_tokens = 16
    bm.publish_full_blocks(req)
    bm.deallocate(req)
    assert bm.num_free_blocks == 0
    assert bm.cache_blocks == 4
    hungry = Request(token_ids=list(range(20, 36)), max_tokens=1)
    assert bm.can_allocate_tokens(hungry, 16)
    bm.allocate_for_tokens(hungry, 16)
    assert len(hungry.block_table) == 4
    assert bm.cache_blocks == 0


def test_cached_generate_matches_uncached_tokens():
    torch.manual_seed(21)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 4, 7, 2, 9, 3, 6, 8, 1, 2]
    off = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    on = Config(
        block_size=4,
        num_kv_blocks=32,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        enable_prefix_cache=True,
    )
    uncached = Engine(off, PagedRunner(model)).generate(prompt, max_tokens=3)
    eng = Engine(on, PagedRunner(model))
    cold = eng.generate(prompt, max_tokens=3)
    warm = eng.generate(prompt, max_tokens=3)
    assert uncached == cold == warm


def test_lru_touch_is_oldest_first():
    cache = PrefixCache()
    a = block_hash(ROOT_HASH, [1])
    b = block_hash(ROOT_HASH, [2])
    c = block_hash(ROOT_HASH, [3])
    assert cache.insert(a, 0)
    assert cache.insert(b, 1)
    assert cache.insert(c, 2)
    cache.touch(a)
    h, bid = cache.pop_oldest()
    assert (h, bid) == (b, 1)


def test_publish_only_hashes_new_full_blocks():
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=True)
    req = Request(token_ids=list(range(1, 13)), max_tokens=1)
    bm.allocate_for_tokens(req, 8)
    req.num_computed_tokens = 8
    bm.publish_full_blocks(req)
    assert req.n_published_blocks == 2
    assert bm.cache_blocks == 2
    refs_after = list(bm._ref_count)
    bm.publish_full_blocks(req)
    assert bm._ref_count == refs_after
    bm.allocate_for_tokens(req, 4)
    req.num_computed_tokens = 12
    bm.publish_full_blocks(req)
    assert req.n_published_blocks == 3
    assert bm.cache_blocks == 3


def test_duplicate_cache_entries_have_independent_lru_and_removal():
    cache = PrefixCache()
    h = block_hash(ROOT_HASH, [1, 2, 3, 4])
    assert cache.insert(h, 0)
    assert cache.insert(h, 3)
    assert not cache.insert(h, 0)
    assert len(cache) == 2
    # Touching one copy must not keep its unused sibling hot.
    assert cache.pop_oldest() == (h, 3)
    assert cache.lookup(h) == 0
    assert cache.drop(h, 3) is None
    assert cache.drop(h, 0) == 0
    assert cache.lookup(h) is None
    assert len(cache) == 0
    assert cache.pop_oldest() is None
    cache.insert(h, 1)
    cache.insert(h, 2)
    cache.clear()
    assert cache.lookup(h) is None
    assert list(cache.lru_items()) == []


@pytest.mark.parametrize("restore", [False, True])
def test_duplicate_prefix_survives_first_copy_eviction(restore):
    bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
    prompt = [1, 2, 3, 4, 5]
    a = Request(token_ids=prompt.copy(), max_tokens=1)
    b = Request(token_ids=prompt.copy(), max_tokens=1)
    # Both allocate before either publishes, as in one cold batch.
    for req in (a, b):
        bm.allocate_for_tokens(req, 4)
        req.num_computed_tokens = 4
    a_page, b_page = a.block_table[0], b.block_table[0]
    for req in (a, b):
        bm.publish_full_blocks(req)
        bm.publish_full_blocks(req)
    assert bm.cache_blocks == 2
    assert bm._ref_count[a_page] == bm._ref_count[b_page] == 2
    bm.deallocate(a)
    # Pressure must evict A's copy, while B's live copy stays indexed.
    assert bm.reclaim_cached_blocks(min_free=3) == 1
    assert bm._ref_count[a_page] == 0
    assert bm._ref_count[b_page] == 2
    assert bm.cache_blocks == 1
    c = Request(token_ids=prompt.copy(), max_tokens=1)
    if restore:
        hit = bm.attach_cached_prefix_upto(c, prompt, 4)
    else:
        hit = bm.attach_cached_prefix(c)
    assert hit == 4
    assert c.block_table == [b_page]
    assert bm._ref_count[b_page] == 3
    bm.deallocate(b)
    assert bm.reclaim_cached_blocks(min_free=4) == 0  # C still protects it.
    bm.deallocate(c)
    assert bm.reclaim_cached_blocks(min_free=4) == 1
    assert bm.cache_blocks == 0
    assert bm.num_free_blocks == 4
    assert bm._ref_count == [0] * 4
    assert len(set(bm._free)) == 4


def test_all_idle_duplicate_copies_can_be_reclaimed_together():
    bm = BlockManager(num_blocks=3, block_size=4, enable_prefix_cache=True)
    reqs = [Request(token_ids=[1, 2, 3, 4, 5], max_tokens=1) for _ in range(3)]
    for req in reqs:
        bm.allocate_for_tokens(req, 4)
        req.num_computed_tokens = 4
        bm.publish_full_blocks(req)
        bm.deallocate(req)
    assert bm.cache_blocks == 3
    assert bm.reclaim_cached_blocks(min_free=3) == 3
    assert bm.cache_blocks == 0
    assert bm._ref_count == [0] * 3
    assert len(set(bm._free)) == 3
