"""Content-addressed prefix cache at block_size granularity.

Not session-scoped hold_kv. A cold request whose prefix was seen before
reuses physical KV blocks by hash, with refcount sharing. Divergence
allocates unique blocks (copy-on-write if a shared block would be written).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import struct

ROOT_HASH = b"\x00" * 16


def block_hash(parent: bytes, token_ids: Sequence[int]) -> bytes:
    """Stable hash of (parent_hash, block tokens). Independent of PYTHONHASHSEED."""
    payload = hashlib.blake2b(digest_size=16)
    payload.update(parent)
    payload.update(struct.pack(f"<{len(token_ids)}I", *[int(t) & 0xFFFFFFFF for t in token_ids]))
    return payload.digest()


class PrefixCache:
    """Maps block hashes to physical ids. Holds one extra ref per cached block.

    A hash can index multiple physical copies produced by a cold batch.
    LRU tracks each (hash, block_id) separately, so evicting one copy preserves
    the others. BlockManager owns the corresponding cache references.
    """

    def __init__(self) -> None:
        self.hash_to_blocks: dict[bytes, dict[int, None]] = {}
        self._lru: OrderedDict[tuple[bytes, int], None] = OrderedDict()

    def __len__(self) -> int:
        # Count physical cached blocks, not distinct prefixes.
        return len(self._lru)

    def touch(self, h: bytes, block_id: int | None = None) -> None:
        if block_id is None:
            block_id = self.lookup(h)
        key = (h, block_id)
        if key in self._lru:
            self._lru.move_to_end(key)

    def lookup(self, h: bytes) -> int | None:
        blocks = self.hash_to_blocks.get(h)
        return next(iter(blocks)) if blocks else None

    def insert(self, h: bytes, block_id: int) -> bool:
        """Return True only for a newly registered physical copy."""
        blocks = self.hash_to_blocks.setdefault(h, {})
        if block_id in blocks:
            self.touch(h, block_id)
            return False
        blocks[block_id] = None
        self._lru[h, block_id] = None
        return True

    def lru_items(self):
        """Oldest cached (hash, block_id) first."""
        return iter(self._lru)

    def pop_oldest(self) -> tuple[bytes, int] | None:
        if not self._lru:
            return None
        h, bid = next(iter(self._lru))
        self.drop(h, bid)
        return h, bid

    def drop(self, h: bytes, block_id: int) -> int | None:
        """Remove only this copy; keep the prefix discoverable via other copies."""
        blocks = self.hash_to_blocks.get(h)
        if blocks is None or block_id not in blocks:
            return None
        del blocks[block_id]
        self._lru.pop((h, block_id))
        if not blocks:
            del self.hash_to_blocks[h]
        return block_id

    def clear(self) -> None:
        self.hash_to_blocks.clear()
        self._lru.clear()
