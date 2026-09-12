"""Owner-thread coordinator for synchronous paused-session KV offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.kv.cpu_store import (
    CpuKVCapacityError,
    CpuKVStore,
    KVSnapshot,
    KVSnapshotMetadata,
)

if TYPE_CHECKING:
    from qwen3_runtime.engine.engine import Engine


class SessionOffloadError(RuntimeError):
    """Base class for a locally recoverable offload failure."""


class SnapshotStaleError(SessionOffloadError):
    """The snapshot belongs to a prior model or KV generation."""


class RestoreAdmissionError(SessionOffloadError):
    """GPU capacity or request state cannot admit a restore right now."""


@dataclass(frozen=True)
class RestoreTicket:
    request_id: int
    snapshot_id: int
    attached_tokens: int
    imported_tokens: int
    imported_bytes: int


class SessionOffloadManager:
    """The sole owner of CPU snapshot handles and transfer generations.

    It is deliberately synchronous in this phase.  Every method is expected
    to run on the engine owner thread, so a successful save or restore has no
    late callback that can mutate scheduler state after invalidation.
    """

    def __init__(self, engine: "Engine"):
        self.engine = engine
        cfg = engine.config
        if cfg.session_cpu_offload == "async":
            raise ValueError(
                "session_cpu_offload=async is not implemented; use sync after P3"
            )
        self.enabled = cfg.session_cpu_offload == "sync"
        self.store = (
            CpuKVStore(
                cfg.cpu_kv_max_bytes,
                pinned_max_bytes=cfg.cpu_kv_pinned_max_bytes,
            )
            if self.enabled
            else None
        )
        self._snapshots: dict[int, KVSnapshot] = {}
        self._next_snapshot_id = 1
        self._operation = 0
        self.weight_epoch = 0
        self.stats: dict[str, int] = {
            "save_attempts": 0,
            "saved": 0,
            "restored": 0,
            "save_capacity_failed": 0,
            "save_copy_failed": 0,
            "restore_failed": 0,
            "stale": 0,
            "cancelled": 0,
            "invalidated": 0,
            "cpu_evicted": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "useful_restores": 0,
            "imported_tokens": 0,
            "restored_d2h_bytes": 0,
            "unused_d2h_bytes": 0,
        }

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def has_snapshot(self, request_id: int) -> bool:
        return int(request_id) in self._snapshots

    def snapshot_for(self, request_id: int) -> KVSnapshot | None:
        return self._snapshots.get(int(request_id))

    def save(self, req: Request, *, session_key: str) -> KVSnapshot:
        """Save a paused request, then release its GPU refs atomically."""
        if not self.enabled:
            raise SessionOffloadError("CPU session offload is disabled")
        if req.status != RequestStatus.PAUSED:
            raise SessionOffloadError("only a paused request can be offloaded")
        if req.kv_residency == "cpu" and req.offload_snapshot_id is not None:
            snapshot = self._snapshots.get(req.request_id)
            if snapshot is None:
                raise SnapshotStaleError(
                    "request claims CPU residency without a snapshot"
                )
            return snapshot
        pool = getattr(self.engine.runner, "pool", None)
        if pool is None or getattr(pool, "cache", None) is None:
            raise SessionOffloadError("runner has no live paged KV pool")
        block_size = self.engine.block_manager.block_size
        valid_tokens = int(req.num_computed_tokens)
        logical_blocks = (
            (valid_tokens + block_size - 1) // block_size if valid_tokens else 0
        )
        if logical_blocks > len(req.block_table):
            raise SessionOffloadError(
                "request KV table does not cover its valid prefix"
            )
        self.stats["save_attempts"] += 1
        self._operation += 1
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        shape = (
            2,
            pool.num_layers,
            logical_blocks,
            pool.block_size,
            pool.num_kv_heads,
            pool.head_dim,
        )
        allocated_bytes = CpuKVStore.required_bytes(shape, pool.cache.dtype)
        metadata = KVSnapshotMetadata(
            session_key=str(session_key),
            request_id=req.request_id,
            snapshot_id=snapshot_id,
            weight_epoch=self.weight_epoch,
            kv_epoch=self.engine.block_manager.epoch,
            token_ids=tuple(req.token_ids),
            num_computed_tokens=valid_tokens,
            logical_blocks=logical_blocks,
            tail_valid_tokens=(valid_tokens % block_size)
            or (block_size if valid_tokens else 0),
            dtype=str(pool.cache.dtype),
            num_layers=pool.num_layers,
            block_size=pool.block_size,
            num_kv_heads=pool.num_kv_heads,
            head_dim=pool.head_dim,
            layout="[K/V, layer, block, offset, kv_head, dim]",
            shape=tuple(shape),
            allocated_bytes=allocated_bytes,
        )
        assert self.store is not None
        if allocated_bytes > self.store.max_bytes:
            # No amount of eviction can admit this snapshot. Keep existing
            # useful CPU history instead of emptying the cache for a failure.
            self.stats["save_capacity_failed"] += 1
            raise CpuKVCapacityError(
                f"snapshot needs {allocated_bytes}, CPU budget is {self.store.max_bytes}"
            )
        try:
            reservation = self.store.reserve(metadata)
        except CpuKVCapacityError:
            # Make room only by evicting the oldest committed CPU snapshot.
            # Its request remains a valid paused session with no KV and will
            # cold-prefill on its next claim; it is never left half-restored.
            while True:
                evicted = self.evict_oldest_snapshot()
                if evicted is None:
                    self.stats["save_capacity_failed"] += 1
                    raise
                try:
                    reservation = self.store.reserve(metadata)
                    break
                except CpuKVCapacityError:
                    continue
        try:
            pool.export_blocks(
                req.block_table[:logical_blocks],
                valid_tokens=valid_tokens,
                chunk_bytes=self.engine.config.transfer_chunk_bytes,
                destination=reservation.buffer,
            )
            if self.engine.block_manager.epoch != metadata.kv_epoch:
                raise SnapshotStaleError("KV epoch changed during CPU save")
            snapshot = reservation.commit()
        except BaseException:
            if reservation._state == "reserved":
                reservation.abort()
            self.stats["save_copy_failed"] += 1
            raise

        # Commit the host copy before touching the source table.  This is the
        # critical no-data-loss ordering for synchronous offload.
        self._snapshots[req.request_id] = snapshot
        self.stats["saved"] += 1
        self.stats["d2h_bytes"] += metadata.allocated_bytes
        self.engine.block_manager.deallocate(req)
        req.num_computed_tokens = 0
        req.num_scheduled_tokens = 0
        req.cached_tokens = 0
        req.n_published_blocks = 0
        req.kv_epoch = metadata.kv_epoch
        req.kv_residency = "cpu"
        req.offload_snapshot_id = snapshot_id
        return snapshot

    def begin_restore(self, req: Request) -> RestoreTicket:
        """Reserve/import GPU blocks but retain the CPU snapshot until commit."""
        snapshot = self._snapshots.get(req.request_id)
        if snapshot is None:
            raise SnapshotStaleError("no CPU snapshot is registered for this request")
        pool = getattr(self.engine.runner, "pool", None)
        if pool is None or getattr(pool, "cache", None) is None:
            raise RestoreAdmissionError("runner has no live paged KV pool")
        meta = snapshot.metadata
        if (
            meta.weight_epoch != self.weight_epoch
            or meta.kv_epoch != self.engine.block_manager.epoch
            or tuple(req.token_ids) != meta.token_ids
            or meta.num_layers != pool.num_layers
            or meta.block_size != pool.block_size
            or meta.num_kv_heads != pool.num_kv_heads
            or meta.head_dim != pool.head_dim
            or meta.dtype != str(pool.cache.dtype)
        ):
            self.stats["stale"] += 1
            self._discard_stale(req)
            raise SnapshotStaleError(
                "CPU snapshot does not match the current model/KV layout"
            )
        block_manager = self.engine.block_manager
        block_manager.attach_cached_prefix_upto(
            req, list(meta.token_ids), meta.num_computed_tokens
        )
        attached = req.num_computed_tokens
        missing = meta.logical_blocks - len(req.block_table)
        try:
            targets = block_manager.allocate_restore_blocks(req, missing)
            if targets:
                pool.import_blocks(
                    targets,
                    snapshot.buffer,
                    valid_tokens=meta.num_computed_tokens,
                    chunk_bytes=self.engine.config.transfer_chunk_bytes,
                    source_block_offset=attached // meta.block_size,
                )
            req.num_computed_tokens = meta.num_computed_tokens
            req.num_scheduled_tokens = 0
            req.kv_epoch = meta.kv_epoch
            req.kv_residency = "gpu"
            return RestoreTicket(
                request_id=req.request_id,
                snapshot_id=meta.snapshot_id,
                attached_tokens=attached,
                imported_tokens=meta.num_computed_tokens - attached,
                imported_bytes=len(targets) * pool.block_bytes,
            )
        except BaseException:
            block_manager.deallocate(req)
            req.num_computed_tokens = 0
            req.num_scheduled_tokens = 0
            req.kv_residency = "cpu"
            self.stats["restore_failed"] += 1
            raise RestoreAdmissionError("CPU snapshot restore rolled back")

    def commit_restore(self, ticket: RestoreTicket) -> None:
        snapshot = self._snapshots.get(ticket.request_id)
        if snapshot is None or snapshot.metadata.snapshot_id != ticket.snapshot_id:
            raise SnapshotStaleError("restore ticket no longer owns its snapshot")
        assert self.store is not None
        self.store.delete(ticket.snapshot_id)
        self._snapshots.pop(ticket.request_id, None)
        self.stats["restored"] += 1
        self.stats["h2d_bytes"] += ticket.imported_bytes
        self.stats["useful_restores"] += int(ticket.imported_tokens > 0)
        self.stats["imported_tokens"] += ticket.imported_tokens
        self.stats["restored_d2h_bytes"] += snapshot.metadata.allocated_bytes

    def rollback_restore(self, req: Request, ticket: RestoreTicket | None) -> None:
        if ticket is None:
            return
        self.engine.block_manager.deallocate(req)
        req.num_computed_tokens = 0
        req.num_scheduled_tokens = 0
        req.kv_residency = "cpu"
        req.offload_snapshot_id = ticket.snapshot_id

    def remove(self, request_id: int) -> None:
        snapshot = self._snapshots.pop(int(request_id), None)
        if snapshot is not None and self.store is not None:
            self.store.delete(snapshot.metadata.snapshot_id)
            self.stats["unused_d2h_bytes"] += snapshot.metadata.allocated_bytes

    def evict_oldest_snapshot(self) -> int | None:
        if self.store is None:
            return None
        snapshot = self.store.evict_oldest()
        if snapshot is None:
            return None
        self.stats["unused_d2h_bytes"] += snapshot.metadata.allocated_bytes
        request_id = snapshot.metadata.request_id
        self._snapshots.pop(request_id, None)
        req = self.engine._requests.get(request_id)
        if req is not None:
            req.kv_residency = "none"
            req.offload_snapshot_id = None
            req.num_computed_tokens = 0
            req.num_scheduled_tokens = 0
        self.stats["cpu_evicted"] += 1
        return request_id

    def invalidate_all(self, *, weight_change: bool = False) -> None:
        if weight_change:
            self.weight_epoch += 1
        if self.store is not None:
            removed = self.store.invalidate_all()
            self.stats["invalidated"] += len(removed)
            self.stats["unused_d2h_bytes"] += sum(s.metadata.allocated_bytes for s in removed)
        for request_id in tuple(self._snapshots):
            req = self.engine._requests.get(request_id)
            if req is not None:
                req.kv_residency = "none"
                req.offload_snapshot_id = None
        self._snapshots.clear()

    def report(self) -> dict[str, int]:
        data = dict(self.stats)
        data["pending_d2h_bytes"] = sum(s.metadata.allocated_bytes for s in self._snapshots.values())
        if self.store is not None:
            data.update({f"cpu_{k}": v for k, v in self.store.stats().items()})
        else:
            data.update(
                {"cpu_committed_bytes": 0, "cpu_reserved_bytes": 0, "cpu_snapshots": 0}
            )
        return data

    def _discard_stale(self, req: Request) -> None:
        self.remove(req.request_id)
        req.kv_residency = "none"
        req.offload_snapshot_id = None
