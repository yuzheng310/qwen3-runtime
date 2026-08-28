"""BlockManager must admit KV incrementally — never reserve the full prompt up front."""

from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request


def test_first_chunk_does_not_reserve_full_prompt_blocks():
    # Spec example: 8192-token prompt, chunk=512, block_size=16.
    # Full prompt would need 512 blocks; the first chunk must need 32.
    bm = BlockManager(num_blocks=64, block_size=16)
    req = Request(token_ids=list(range(8192)), max_tokens=1)

    scheduled = 512
    assert bm.can_allocate_tokens(req, scheduled)
    bm.allocate_for_tokens(req, scheduled)

    assert len(req.block_table) == 32
    assert bm.num_free_blocks == 32


def test_second_chunk_appends_blocks_instead_of_reallocating():
    bm = BlockManager(num_blocks=64, block_size=16)
    req = Request(token_ids=list(range(8192)), max_tokens=1)
    bm.allocate_for_tokens(req, 512)
    first_table = list(req.block_table)
    req.num_computed_tokens = 512

    bm.allocate_for_tokens(req, 512)
    assert req.block_table[:32] == first_table
    assert len(req.block_table) == 64
    assert bm.num_free_blocks == 0


def test_full_prompt_does_not_fit_when_pool_only_covers_one_chunk():
    bm = BlockManager(num_blocks=32, block_size=16)
    req = Request(token_ids=list(range(8192)), max_tokens=1)
    assert bm.can_allocate_tokens(req, 512)
    assert not bm.can_allocate_tokens(req, 8192)


def test_deallocate_returns_blocks_to_the_pool():
    bm = BlockManager(num_blocks=8, block_size=16)
    req = Request(token_ids=list(range(32)), max_tokens=1)
    bm.allocate_for_tokens(req, 32)
    assert bm.num_free_blocks == 6
    bm.deallocate(req)
    assert req.block_table == []
    assert bm.num_free_blocks == 8


def test_slot_mapping_is_physical_block_times_block_size_plus_offset():
    bm = BlockManager(num_blocks=4, block_size=4)
    req = Request(token_ids=list(range(10)), max_tokens=1)
    bm.allocate_for_tokens(req, 6)
    # two blocks: ids 0 then 1 (free list is 0,1,2,3)
    assert req.block_table == [0, 1]
    assert bm.slot_mapping(req, start=0, num_tokens=6) == [0, 1, 2, 3, 4, 5]


def test_max_allocatable_tokens_is_free_blocks_times_block_size_minus_start():
    bm = BlockManager(num_blocks=2, block_size=16)
    req = Request(token_ids=list(range(40)), max_tokens=1)
    assert bm.max_allocatable_tokens(req, 512) == 32
    bm.allocate_for_tokens(req, 32)
    req.num_computed_tokens = 32
    assert bm.max_allocatable_tokens(req, 512) == 0


def test_decode_append_allocates_a_block_only_when_crossing_boundary():
    bm = BlockManager(num_blocks=4, block_size=4)
    req = Request(token_ids=list(range(4)), max_tokens=8)
    bm.allocate_for_tokens(req, 4)
    req.num_computed_tokens = 4
    assert len(req.block_table) == 1

    bm.allocate_for_tokens(req, 1)  # position 4 → new block
    assert len(req.block_table) == 2
    req.num_computed_tokens = 5
    bm.allocate_for_tokens(req, 1)  # still in the second block
    assert len(req.block_table) == 2
