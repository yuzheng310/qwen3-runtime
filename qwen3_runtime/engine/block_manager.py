"""Host-side physical KV block allocator and prefix-cache attach/publish."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from qwen3_runtime.engine.prefix_cache import ROOT_HASH, PrefixCache, block_hash
from qwen3_runtime.engine.request import Request

BlockCopier = Callable[[int, int], None]


class BlockManager:
    """Host-side physical block pool. Device tensors live in PagedKVPool.

    Allocation is always for a token *span*, never for the full prompt.
    Slot id = physical_block_id * block_size + offset.

    When ``enable_prefix_cache`` is set, full blocks are content-addressed:
    matching prefixes share physical blocks by refcount. The cache holds one
    extra ref so blocks survive after the last request finishes, until eviction
    under allocation pressure.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        *,
        enable_prefix_cache: bool = False,
    ):
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self._free: deque[int] = deque(range(num_blocks))
        self._ref_count = [0] * num_blocks
        self._cache = PrefixCache()
        self._copy_block: BlockCopier | None = None
        self.epoch = (
            0  # bumped on reset(); leftover tables from a prior epoch are stale
        )

    def set_block_copier(self, copy_block: BlockCopier | None) -> None:
        self._copy_block = copy_block

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def cache_blocks(self) -> int:
        return len(self._cache)

    def blocks_needed_for(self, end_token: int) -> int:
        if end_token <= 0:
            return 0
        return (end_token + self.block_size - 1) // self.block_size

    def can_allocate_tokens(self, req: Request, num_tokens: int) -> bool:
        if num_tokens <= 0:
            return True
        new_end = req.num_computed_tokens + num_tokens
        needed = self.blocks_needed_for(new_end) - len(req.block_table)
        if needed <= self.num_free_blocks:
            return True
        self._evict_unused(needed - self.num_free_blocks)
        return needed <= self.num_free_blocks

    def max_allocatable_tokens(self, req: Request, limit: int) -> int:
        """Largest n <= limit whose KV fits in already-held plus free blocks."""
        if limit <= 0:
            return 0
        start = req.num_computed_tokens
        extra = self.blocks_needed_for(start + limit) - len(req.block_table)
        if extra > self.num_free_blocks:
            self._evict_unused(extra - self.num_free_blocks)
        capacity = (len(req.block_table) + self.num_free_blocks) * self.block_size
        return max(0, min(limit, capacity - start))

    def allocate_for_tokens(self, req: Request, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        req.kv_epoch = self.epoch
        if not self.can_allocate_tokens(req, num_tokens):
            raise RuntimeError("KV pool exhausted")
        start = req.num_computed_tokens
        new_end = start + num_tokens
        needed = self.blocks_needed_for(new_end)
        while len(req.block_table) < needed:
            req.block_table.append(self._alloc_block())
        if self.enable_prefix_cache:
            first_block = start // self.block_size
            last_block = (new_end - 1) // self.block_size if new_end else -1
            for idx in range(first_block, last_block + 1):
                self._cow_if_shared(req, idx)

    def deallocate(self, req: Request) -> None:
        for block_id in req.block_table:
            self._decref(block_id)
        req.block_table.clear()

    def reclaimable_blocks(self, requests):
        from collections import Counter

        refs = Counter(b for r in requests for b in r.block_table)
        cache_refs = Counter(b for _, b in self._cache.lru_items())
        return sum(self._ref_count[b] == n + cache_refs[b] for b, n in refs.items())

    def resource_view(self) -> dict[str, int]:
        """A physical view; logical request lengths are intentionally absent."""
        unique = sum(refs > 0 for refs in self._ref_count)
        cached_ids = {b for _, b in self._cache.lru_items()}
        cache_only = sum(
            refs == 1 and block_id in cached_ids
            for block_id, refs in enumerate(self._ref_count)
        )
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "unique_allocated_blocks": unique,
            "cache_only_blocks": cache_only,
        }

    def reclaim_cached_blocks(self, min_free: int) -> int:
        """Reclaim only APC-owned blocks, preserving every live request ref."""
        before = self.num_free_blocks
        self._evict_unused(max(0, min_free - before))
        return self.num_free_blocks - before

    def reset(self) -> None:
        """Forget every allocation. Caller must have deallocated live requests.

        Sleep/wake and post-update invalidation rebuild the free list from scratch
        so a later resume cannot keep a stale physical id into a new pool.
        """
        self._cache.clear()
        self._free = deque(range(self.num_blocks))
        self._ref_count = [0] * self.num_blocks
        self.epoch += 1

    def truncate_kv(self, req: Request, num_tokens: int) -> None:
        """Drop trailing block-table entries so the table covers ``num_tokens`` slots.

        Speculative query KV past the committed prefix is discarded. Physical
        pages return to the free list; the next allocate overwrites them.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        needed = self.blocks_needed_for(num_tokens)
        while len(req.block_table) > needed:
            self._decref(req.block_table.pop())
        req.n_published_blocks = min(req.n_published_blocks, needed)

    def attach_cached_prefix(self, req: Request) -> int:
        """Share full cached prefix blocks. Sets num_computed_tokens and cached_tokens."""
        req.cached_tokens = 0
        req.n_published_blocks = 0
        req.prefix_parent = ROOT_HASH
        if not self.enable_prefix_cache:
            return 0
        parent = ROOT_HASH
        cached = 0
        bs = self.block_size
        n_full = len(req.token_ids) // bs
        # Leave at least one prompt token uncached so the last-token sample still runs.
        max_cached = max(0, len(req.token_ids) - 1)
        n_full = min(n_full, max_cached // bs)
        for i in range(n_full):
            tokens = req.token_ids[i * bs : (i + 1) * bs]
            h = block_hash(parent, tokens)
            bid = self._cache.lookup(h)
            if bid is None or self._ref_count[bid] <= 0:
                if bid is not None:
                    self._cache.drop(h, bid)
                break
            self._incref(bid)
            req.block_table.append(bid)
            self._cache.touch(h, bid)
            parent = h
            cached += bs
        req.num_computed_tokens = cached
        req.cached_tokens = cached
        req.n_published_blocks = cached // bs
        req.prefix_parent = parent
        return cached

    def attach_cached_prefix_upto(
        self, req: Request, tokens: list[int], num_tokens: int
    ) -> int:
        """Attach current APC full blocks to an empty restore target.

        Unlike ``attach_cached_prefix`` this does not infer validity from the
        request's full token list and never leaves a partial request table in
        place.  It is the only APC entry point used by CPU restore.
        """
        if req.block_table:
            raise RuntimeError("restore APC attach requires an empty block table")
        if num_tokens < 0 or num_tokens > len(tokens):
            raise ValueError("invalid restore token length")
        req.cached_tokens = 0
        req.n_published_blocks = 0
        req.prefix_parent = ROOT_HASH
        if not self.enable_prefix_cache:
            return 0
        parent = ROOT_HASH
        cached = 0
        full_blocks = num_tokens // self.block_size
        for index in range(full_blocks):
            block_tokens = tokens[
                index * self.block_size : (index + 1) * self.block_size
            ]
            h = block_hash(parent, block_tokens)
            bid = self._cache.lookup(h)
            if bid is None or self._ref_count[bid] <= 0:
                break
            self._incref(bid)
            req.block_table.append(bid)
            self._cache.touch(h, bid)
            parent = h
            cached += self.block_size
        req.num_computed_tokens = cached
        req.cached_tokens = cached
        req.n_published_blocks = cached // self.block_size
        req.prefix_parent = parent
        return cached

    def allocate_restore_blocks(self, req: Request, num_blocks: int) -> list[int]:
        """Reserve physical blocks for the missing part of a restore."""
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if num_blocks > self.num_free_blocks:
            self._evict_unused(num_blocks - self.num_free_blocks)
        if num_blocks > self.num_free_blocks:
            raise RuntimeError("KV pool exhausted during CPU session restore")
        req.kv_epoch = self.epoch
        for _ in range(num_blocks):
            req.block_table.append(self._alloc_block())
        return req.block_table[-num_blocks:] if num_blocks else []

    def publish_full_blocks(self, req: Request) -> None:
        """Insert newly completed full blocks into the prefix cache."""
        if not self.enable_prefix_cache:
            return
        bs = self.block_size
        n_full = min(len(req.block_table), req.num_computed_tokens // bs)
        parent = req.prefix_parent
        for i in range(req.n_published_blocks, n_full):
            tokens = req.token_ids[i * bs : (i + 1) * bs]
            if len(tokens) < bs:
                break
            h = block_hash(parent, tokens)
            bid = req.block_table[i]
            if self._cache.insert(h, bid):
                self._incref(bid)
            parent = h
            req.n_published_blocks = i + 1
            req.prefix_parent = parent

    def slot_mapping(self, req: Request, start: int, num_tokens: int) -> list[int]:
        slots: list[int] = []
        for pos in range(start, start + num_tokens):
            block_index = pos // self.block_size
            offset = pos % self.block_size
            slots.append(req.block_table[block_index] * self.block_size + offset)
        return slots

    def _cow_if_shared(self, req: Request, block_index: int) -> None:
        bid = req.block_table[block_index]
        if self._ref_count[bid] <= 1:
            return
        new_id = self._alloc_block()
        if self._copy_block is not None:
            self._copy_block(bid, new_id)
        self._decref(bid)
        req.block_table[block_index] = new_id

    def _evict_unused(self, n_blocks: int) -> None:
        if not self.enable_prefix_cache or n_blocks <= 0:
            return
        victims: list[tuple[bytes, int]] = []
        for h, bid in self._cache.lru_items():
            if self._ref_count[bid] == 1:
                victims.append((h, bid))
                if len(victims) >= n_blocks:
                    break
        for h, bid in victims:
            bid = self._cache.drop(h, bid)
            if bid is not None:
                self._decref(bid)

    def _alloc_block(self) -> int:
        if not self._free:
            self._evict_unused(1)
        block_id = self._free.popleft()
        self._ref_count[block_id] = 1
        return block_id

    def _incref(self, block_id: int) -> None:
        self._ref_count[block_id] += 1

    def _decref(self, block_id: int) -> None:
        self._ref_count[block_id] -= 1
        if self._ref_count[block_id] < 0:
            raise RuntimeError(f"block {block_id} refcount underflow")
        if self._ref_count[block_id] == 0:
            self._free.append(block_id)
