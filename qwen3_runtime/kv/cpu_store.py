"""A bounded, process-local CPU store for paused-session KV snapshots.

The store owns host buffers and their accounting.  It deliberately knows
nothing about requests, block tables, or eviction policy beyond an LRU stamp;
the engine owner thread decides when a snapshot is safe to create or remove.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch


class CpuKVCapacityError(RuntimeError):
    """The configured host budget cannot admit another complete snapshot."""


class CpuKVTransactionError(RuntimeError):
    """A reservation was committed, aborted, or otherwise used twice."""


@dataclass(frozen=True)
class KVSnapshotMetadata:
    session_key: str
    request_id: int
    snapshot_id: int
    weight_epoch: int
    kv_epoch: int
    token_ids: tuple[int, ...]
    num_computed_tokens: int
    logical_blocks: int
    tail_valid_tokens: int
    dtype: str
    num_layers: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    layout: str
    shape: tuple[int, ...]
    allocated_bytes: int


@dataclass
class KVReservation:
    metadata: KVSnapshotMetadata
    buffer: torch.Tensor
    _store: "CpuKVStore"
    _pinned: bool = False
    _state: str = "reserved"

    def commit(self) -> "KVSnapshot":
        return self._store.commit(self)

    def abort(self) -> None:
        self._store.abort(self)


@dataclass(frozen=True)
class KVSnapshot:
    metadata: KVSnapshotMetadata
    buffer: torch.Tensor
    _store: "CpuKVStore"
    pinned: bool = False

    def delete(self) -> None:
        self._store.delete(self.metadata.snapshot_id)


class CpuKVStore:
    """Capacity-accounted host storage with reserve/write/commit semantics."""

    def __init__(self, max_bytes: int, *, pinned_max_bytes: int = 0):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if pinned_max_bytes < 0 or pinned_max_bytes > max_bytes:
            raise ValueError("pinned_max_bytes must be within max_bytes")
        self.max_bytes = int(max_bytes)
        self.pinned_max_bytes = int(pinned_max_bytes)
        self._reserved_bytes = 0
        self._committed_bytes = 0
        self._pinned_bytes = 0
        self._reserved_pinned_bytes = 0
        self._clock = 0
        self._snapshots: "OrderedDict[int, KVSnapshot]" = OrderedDict()

    @staticmethod
    def required_bytes(shape: Iterable[int], dtype: torch.dtype) -> int:
        n = 1
        for dim in shape:
            n *= int(dim)
        return n * torch.empty((), dtype=dtype).element_size()

    def reserve(self, metadata: KVSnapshotMetadata) -> KVReservation:
        expected = self.required_bytes(metadata.shape, torch_dtype(metadata.dtype))
        if expected != metadata.allocated_bytes:
            raise ValueError("snapshot metadata byte count does not match its layout")
        if metadata.allocated_bytes <= 0 and metadata.logical_blocks:
            raise ValueError("non-empty snapshot must have positive storage")
        if (
            self._committed_bytes + self._reserved_bytes + metadata.allocated_bytes
            > self.max_bytes
        ):
            raise CpuKVCapacityError(
                f"CPU KV budget exhausted: need {metadata.allocated_bytes}, "
                f"available {self.max_bytes - self._committed_bytes - self._reserved_bytes}"
            )
        try:
            # Pinning is only meaningful when an accelerator exists.  CPU-only
            # validation remains runnable on hosts without a pin-memory pool.
            pin = bool(
                self.pinned_max_bytes
                and torch.cuda.is_available()
                and self._pinned_bytes
                + self._reserved_pinned_bytes
                + metadata.allocated_bytes
                <= self.pinned_max_bytes
            )
            # Keep each logical block's K/V and layers together in host memory.
            # The public six-axis shape stays unchanged; block-range slices
            # become contiguous DMA ranges after moving axis 2 to the front.
            k, layers, blocks, offsets, heads, dim = metadata.shape
            buffer = torch.empty(
                (blocks, k, layers, offsets, heads, dim),
                dtype=torch_dtype(metadata.dtype),
                pin_memory=pin,
            ).permute(1, 2, 0, 3, 4, 5)
        except Exception:
            raise
        self._reserved_bytes += metadata.allocated_bytes
        if pin:
            self._reserved_pinned_bytes += metadata.allocated_bytes
        return KVReservation(metadata=metadata, buffer=buffer, _store=self, _pinned=pin)

    def commit(self, reservation: KVReservation) -> KVSnapshot:
        if reservation._store is not self or reservation._state != "reserved":
            raise CpuKVTransactionError("reservation is not open")
        sid = reservation.metadata.snapshot_id
        if sid in self._snapshots:
            raise CpuKVTransactionError(f"snapshot {sid} already exists")
        self._reserved_bytes -= reservation.metadata.allocated_bytes
        self._committed_bytes += reservation.metadata.allocated_bytes
        if reservation._pinned:
            self._reserved_pinned_bytes -= reservation.metadata.allocated_bytes
            self._pinned_bytes += reservation.metadata.allocated_bytes
        snapshot = KVSnapshot(
            reservation.metadata, reservation.buffer, self, reservation._pinned
        )
        self._snapshots[sid] = snapshot
        reservation._state = "committed"
        self._touch(sid)
        return snapshot

    def abort(self, reservation: KVReservation) -> None:
        if reservation._store is not self or reservation._state != "reserved":
            raise CpuKVTransactionError("reservation is not open")
        self._reserved_bytes -= reservation.metadata.allocated_bytes
        if reservation._pinned:
            self._reserved_pinned_bytes -= reservation.metadata.allocated_bytes
        reservation._state = "aborted"

    def get(self, snapshot_id: int) -> KVSnapshot | None:
        snapshot = self._snapshots.get(int(snapshot_id))
        if snapshot is not None:
            self._touch(snapshot_id)
        return snapshot

    def delete(self, snapshot_id: int) -> KVSnapshot | None:
        snapshot = self._snapshots.pop(int(snapshot_id), None)
        if snapshot is not None:
            self._committed_bytes -= snapshot.metadata.allocated_bytes
            if snapshot.pinned:
                self._pinned_bytes -= snapshot.metadata.allocated_bytes
        return snapshot

    def evict_oldest(self, *, protected: set[int] | None = None) -> KVSnapshot | None:
        protected = protected or set()
        for sid in tuple(self._snapshots):
            if sid not in protected:
                return self.delete(sid)
        return None

    def invalidate_all(self) -> list[KVSnapshot]:
        removed = list(self._snapshots.values())
        self._snapshots.clear()
        self._committed_bytes = 0
        self._pinned_bytes = 0
        return removed

    def stats(self) -> dict[str, int]:
        return {
            "max_bytes": self.max_bytes,
            "committed_bytes": self._committed_bytes,
            "reserved_bytes": self._reserved_bytes,
            "inflight_bytes": self._reserved_bytes,
            "staging_bytes": 0,
            "pinned_bytes": self._pinned_bytes,
            "snapshots": len(self._snapshots),
        }

    def _touch(self, snapshot_id: int) -> None:
        self._clock += 1
        snapshot = self._snapshots.pop(int(snapshot_id))
        self._snapshots[int(snapshot_id)] = snapshot


def torch_dtype(name: str) -> torch.dtype:
    value = name.removeprefix("torch.")
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported snapshot dtype {name!r}")
    return dtype
