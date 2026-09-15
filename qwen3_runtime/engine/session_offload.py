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

    def __init__(self, engine: Engine):
        self.engine = engine
        cfg = engine.config
        self.enabled = cfg.session_cpu_offload in {"sync", "async"}
        self.store = (
            CpuKVStore(
                cfg.cpu_kv_max_bytes,
                pinned_max_bytes=cfg.cpu_kv_pinned_max_bytes,
            )
            if self.enabled
            else None
        )
        self._group_reservations = {}
        self._pending_use = {}
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
            "cpu_committed_write_bytes": 0,
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
        metadata = self._metadata(req, session_key)
        valid_tokens = metadata.num_computed_tokens
        logical_blocks = metadata.logical_blocks
        allocated_bytes = metadata.allocated_bytes
        snapshot_id = metadata.snapshot_id
        self.stats["save_attempts"] += 1
        assert self.store is not None
        if allocated_bytes > self.store.max_bytes:
            # No amount of eviction can admit this snapshot. Keep existing
            # useful CPU history instead of emptying the cache for a failure.
            self.stats["save_capacity_failed"] += 1
            raise CpuKVCapacityError(
                f"snapshot needs {allocated_bytes}, CPU budget is {self.store.max_bytes}"
            )
        try:
            reservation = self._group_reservations.pop(req.request_id, None)
            if reservation is None:
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
        metadata = reservation.metadata
        snapshot_id = metadata.snapshot_id
        try:
            self._export_blocks(
                req.block_table[:logical_blocks],
                valid_tokens=valid_tokens,
                chunk_bytes=self.engine.config.transfer_chunk_bytes,
                destination=reservation.buffer,
            )
            if self.engine.block_manager.epoch != metadata.kv_epoch:
                raise SnapshotStaleError("KV epoch changed during CPU save")
            snapshot = reservation.commit()
            self.stats["cpu_committed_write_bytes"] += metadata.allocated_bytes
        except BaseException as exc:
            if fatal_device_error(exc):
                self.engine._fatal_error = str(exc)
            if reservation._state == "reserved":
                reservation.abort()
            self.stats["save_copy_failed"] += 1
            raise

        # Commit the host copy before touching the source table.  This is the
        # critical no-data-loss ordering for synchronous offload.
        self._snapshots[req.request_id] = snapshot
        self.stats["saved"] += 1
        self.engine.block_manager.deallocate(req)
        req.num_computed_tokens = 0
        req.num_scheduled_tokens = 0
        req.cached_tokens = 0
        req.n_published_blocks = 0
        req.kv_epoch = metadata.kv_epoch
        req.kv_residency = "cpu"
        req.offload_snapshot_id = snapshot_id
        return snapshot

    def early_save(self, req, *, session_key):
        """Best-effort early save without evicting another useful CPU snapshot."""
        metadata = self._metadata(req, session_key)
        try:
            reservation = self.store.reserve(metadata)
        except CpuKVCapacityError:
            return False
        self._group_reservations[req.request_id] = reservation
        try:
            self.save(req, session_key=session_key)
        finally:
            unused = self._group_reservations.pop(req.request_id, None)
            if unused is not None and unused._state == "reserved":
                unused.abort()
        return True

    def _export_blocks(self, *args, **kwargs):
        pool = self.engine.runner.pool
        before = pool.transfer_stats["d2h_completed_bytes"]
        try:
            return pool.export_blocks(*args, **kwargs)
        finally:
            # Completed transfers remain a cost even when commit later fails.
            self.stats["d2h_bytes"] += (
                pool.transfer_stats["d2h_completed_bytes"] - before
            )

    def _metadata(self, req, session_key):
        pool = self.engine.runner.pool
        valid_tokens = int(req.num_computed_tokens)
        block_size = self.engine.block_manager.block_size
        logical_blocks = (valid_tokens + block_size - 1) // block_size
        if (
            valid_tokens < 0
            or valid_tokens > len(req.token_ids)
            or logical_blocks > len(req.block_table)
        ):
            raise SessionOffloadError("invalid committed KV prefix")
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
        return metadata

    def prepare_group(self, requests):
        metadata = [self._metadata(r, str(r.request_id)) for r in requests]
        required = sum(m.allocated_bytes for m in metadata)
        if required > self.store.max_bytes:
            raise CpuKVCapacityError("complete snapshot group exceeds host budget")
        while self.store.budget.owned + required > self.store.max_bytes:
            if self.evict_oldest_snapshot() is None:
                raise CpuKVCapacityError(
                    "snapshot group blocked by outstanding host holders"
                )
        try:
            for meta in metadata:
                self._group_reservations[meta.request_id] = self.store.reserve(meta)
        except BaseException:
            self.close_group()
            raise

    def close_group(self):
        for reservation in self._group_reservations.values():
            if reservation._state == "reserved":
                reservation.abort()
        self._group_reservations.clear()

    def reclaim_group(self, requests, min_free):
        """Physical reservation precedes the first transfer; partial commits stand."""
        bm = self.engine.block_manager
        before = bm.num_free_blocks
        predicted = bm.reclaimable_blocks(requests)
        copied = self.stats["d2h_bytes"]
        committed = self.stats["cpu_committed_write_bytes"]
        result = dict(
            predicted_reclaimed_blocks=predicted,
            actual_gpu_freed_blocks=0,
            cpu_committed_bytes=0,
            partial_commit=False,
            goal_satisfied=False,
            failure_reason=None,
            no_progress=False,
            committed_ids=[],
        )
        self.stats["group_attempts"] = self.stats.get("group_attempts", 0) + 1
        try:
            self.prepare_group(requests)
            for req in requests:
                self.save(req, session_key=str(req.request_id))
                result["committed_ids"].append(req.request_id)
                bm.reclaim_cached_blocks(min_free)
                if bm.num_free_blocks >= min_free:
                    break
        except BaseException as exc:
            result["failure_reason"] = type(exc).__name__ + ": " + str(exc)
            result["predicted_reclaimed_blocks"] = None
            result["partial_commit"] = bool(result["committed_ids"])
            if fatal_device_error(exc):
                self.engine._fatal_error = str(exc)
            if fatal_device_error(exc) or not isinstance(
                exc, (RuntimeError, MemoryError)
            ):
                raise
        finally:
            self.close_group()
            result["actual_gpu_freed_blocks"] = bm.num_free_blocks - before
            result["cpu_committed_bytes"] = (
                self.stats["cpu_committed_write_bytes"] - committed
            )
            result["d2h_completed_bytes"] = self.stats["d2h_bytes"] - copied
            result["goal_satisfied"] = bm.num_free_blocks >= min_free
            result["no_progress"] = bm.num_free_blocks <= before
            self.last_group_result = result
            for name in ("partial_commit", "no_progress"):
                self.stats["group_" + name] = self.stats.get("group_" + name, 0) + int(
                    result[name]
                )
            if result["failure_reason"]:
                self.stats["group_rejected_or_failed"] = (
                    self.stats.get("group_rejected_or_failed", 0) + 1
                )
        return result

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
        if self.engine.config.snapshot_mixed_restore:
            return self._begin_mixed_restore(req, snapshot)
        block_manager = self.engine.block_manager
        block_manager.attach_cached_prefix_upto(
            req, list(meta.token_ids), meta.num_computed_tokens
        )
        attached = req.num_computed_tokens
        missing = meta.logical_blocks - len(req.block_table)
        try:
            needed = missing + int(meta.num_computed_tokens % meta.block_size == 0)
            block_manager.reclaim_cached_blocks(needed)
            if block_manager.num_free_blocks < needed:
                raise RestoreAdmissionError(
                    "history plus next execution token exceeds available KV pool"
                )
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
        except BaseException as exc:
            block_manager.deallocate(req)
            req.num_computed_tokens = 0
            req.num_scheduled_tokens = 0
            req.kv_residency = "cpu"
            self.stats["restore_failed"] += 1
            if fatal_device_error(exc):
                self.engine._fatal_error = str(exc)
                raise
            raise RestoreAdmissionError("CPU snapshot restore rolled back") from exc

    def _begin_mixed_restore(self, req, snapshot):
        """Keep whole-snapshot storage, but import only missing physical pages.

        Snapshot coverage is complete. A GPU hole is filled by its CPU page,
        permitting later GPU hits without ever attaching an uncovered suffix.
        Every GPU hit is pinned before any allocator reclaim. The CPU owner is
        retained until the normal commit; ordinary rollback releases all refs.
        """
        from qwen3_runtime.engine.prefix_cache import ROOT_HASH, block_hash

        if req.block_table:
            raise RestoreAdmissionError("restore requires released GPU references")
        bm, pool, meta = (
            self.engine.block_manager,
            self.engine.runner.pool,
            snapshot.metadata,
        )
        pins, new, plan = [], [], []
        parent = ROOT_HASH
        try:
            for i in range(meta.logical_blocks):
                bid = None
                if (i + 1) * meta.block_size <= meta.num_computed_tokens:
                    parent = block_hash(
                        parent,
                        meta.token_ids[i * meta.block_size : (i + 1) * meta.block_size],
                    )
                    if bm.enable_prefix_cache:
                        bid = bm._cache.lookup(parent)
                if bid is not None and bm._ref_count[bid] > 0:
                    bm._incref(bid)
                    pins.append(bid)
                    plan.append(bid)
                else:
                    plan.append(None)
            missing = sum(bid is None for bid in plan)
            needed = missing + int(meta.num_computed_tokens % meta.block_size == 0)
            bm.reclaim_cached_blocks(needed)
            if bm.num_free_blocks < needed:
                raise RestoreAdmissionError(
                    "history plus next execution token exceeds available KV pool"
                )
            for bid in plan:
                if bid is None:
                    bid = bm._alloc_block()
                    new.append(bid)
                req.block_table.append(bid)
            # Source-contiguous missing ranges; never copy a GPU hit over itself.
            start = 0
            imported = 0
            while start < len(plan):
                if plan[start] is not None:
                    start += 1
                    continue
                end = start + 1
                while end < len(plan) and plan[end] is None:
                    end += 1
                pool.import_blocks(
                    req.block_table[start:end],
                    snapshot.buffer,
                    valid_tokens=meta.num_computed_tokens,
                    chunk_bytes=self.engine.config.transfer_chunk_bytes,
                    source_block_offset=start,
                )
                imported += (
                    min(end * meta.block_size, meta.num_computed_tokens)
                    - start * meta.block_size
                )
                start = end
            req.num_computed_tokens = meta.num_computed_tokens
            req.num_scheduled_tokens = 0
            req.kv_epoch = meta.kv_epoch
            req.kv_residency = "gpu"
            req.cached_tokens = len(pins) * meta.block_size
            req.n_published_blocks = 0
            req.prefix_parent = ROOT_HASH
            return RestoreTicket(
                req.request_id,
                meta.snapshot_id,
                len(pins) * meta.block_size,
                imported,
                missing * pool.block_bytes,
            )
        except BaseException as exc:
            for bid in pins + new:
                bm._decref(bid)
            req.block_table.clear()
            req.num_computed_tokens = req.num_scheduled_tokens = req.cached_tokens = (
                req.n_published_blocks
            ) = 0
            req.prefix_parent = ROOT_HASH
            req.kv_residency = "cpu"
            self.stats["restore_failed"] += 1
            if fatal_device_error(exc):
                self.engine._fatal_error = str(exc)
                raise
            raise RestoreAdmissionError(
                "CPU snapshot mixed restore rolled back"
            ) from exc

    def commit_restore(self, ticket: RestoreTicket) -> None:
        snapshot = self._snapshots.get(ticket.request_id)
        if snapshot is None or snapshot.metadata.snapshot_id != ticket.snapshot_id:
            raise SnapshotStaleError("restore ticket no longer owns its snapshot")
        assert self.store is not None
        self.store.delete(ticket.snapshot_id)
        self._snapshots.pop(ticket.request_id, None)
        self.stats["restored"] += 1
        self.stats["h2d_bytes"] += ticket.imported_bytes
        req = self.engine._requests[ticket.request_id]
        self._pending_use[ticket.request_id] = (
            ticket,
            snapshot.metadata.allocated_bytes,
            req.num_computed_tokens,
            tuple(req.block_table),
        )

    def mark_used(self, requests):
        for req in requests:
            pending = self._pending_use.pop(req.request_id, None)
            if pending is None:
                continue
            ticket, written, valid, table = pending
            if (
                req.num_computed_tokens < valid
                or tuple(req.block_table[: len(table)]) != table
            ):
                self.stats["unused_d2h_bytes"] += written
                continue
            self.stats["useful_restores"] += int(ticket.imported_tokens > 0)
            self.stats["imported_tokens"] += ticket.imported_tokens
            used = min(written, ticket.imported_bytes)
            self.stats["restored_d2h_bytes"] += used
            self.stats["unused_d2h_bytes"] += written - used

    def rollback_restore(self, req: Request, ticket: RestoreTicket | None) -> None:
        if ticket is None:
            return
        self.engine.block_manager.deallocate(req)
        req.num_computed_tokens = 0
        req.num_scheduled_tokens = 0
        req.kv_residency = "cpu"
        req.offload_snapshot_id = ticket.snapshot_id

    def remove(self, request_id: int) -> None:
        pending = self._pending_use.pop(request_id, None)
        if pending is not None:
            self.stats["unused_d2h_bytes"] += pending[1]
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
            if not req.block_table:
                req.num_computed_tokens = 0
                req.num_scheduled_tokens = 0
        self.stats["cpu_evicted"] += 1
        return request_id

    def invalidate_all(self, *, weight_change: bool = False) -> None:
        self.close_group()
        self.stats["unused_d2h_bytes"] += sum(p[1] for p in self._pending_use.values())
        self._pending_use.clear()
        if weight_change:
            self.weight_epoch += 1
        if self.store is not None:
            removed = self.store.invalidate_all()
            self.stats["invalidated"] += len(removed)
            self.stats["unused_d2h_bytes"] += sum(
                s.metadata.allocated_bytes for s in removed
            )
        for request_id in tuple(self._snapshots):
            req = self.engine._requests.get(request_id)
            if req is not None:
                req.kv_residency = "none"
                req.offload_snapshot_id = None
        self._snapshots.clear()

    def report(self) -> dict[str, int]:
        data = dict(self.stats)
        data["uncommitted_d2h_bytes"] = (
            self.stats["d2h_bytes"] - self.stats["cpu_committed_write_bytes"]
        )
        data["pending_d2h_bytes"] = sum(
            s.metadata.allocated_bytes for s in self._snapshots.values()
        )
        data["pending_d2h_bytes"] += sum(p[1] for p in self._pending_use.values())
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


def fatal_device_error(exc):
    text = str(exc).lower()
    return any(
        x in text
        for x in (
            "device-side assert",
            "illegal memory access",
            "context is destroyed",
            "unspecified launch failure",
            "misaligned address",
        )
    )
