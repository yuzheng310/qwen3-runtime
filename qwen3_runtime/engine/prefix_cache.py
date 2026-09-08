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

    LRU is an OrderedDict: ``touch`` / ``insert`` move to the end, ``pop_oldest``
    is O(1). The previous deque.remove path was O(n) per decode step.
    """

    def __init__(self) -> None:
        self.hash_to_block: OrderedDict[bytes, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self.hash_to_block)

    def touch(self, h: bytes) -> None:
        if h in self.hash_to_block:
            self.hash_to_block.move_to_end(h)

    def lookup(self, h: bytes) -> int | None:
        return self.hash_to_block.get(h)

    def insert(self, h: bytes, block_id: int) -> bool:
        """Return True if this hash was newly inserted."""
        if h in self.hash_to_block:
            self.touch(h)
            return False
        self.hash_to_block[h] = block_id
        return True

    def lru_items(self):
        """Oldest cached (hash, block_id) first."""
        return iter(self.hash_to_block.items())

    def pop_oldest(self) -> tuple[bytes, int] | None:
        if not self.hash_to_block:
            return None
        return self.hash_to_block.popitem(last=False)

    def drop(self, h: bytes) -> int | None:
        return self.hash_to_block.pop(h, None)

    def clear(self) -> None:
        self.hash_to_block.clear()
