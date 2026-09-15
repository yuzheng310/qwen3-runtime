"""Owner-thread, event-backed snapshot D2H; restores remain synchronous.

Exactly one save may be in flight. The paused request retains all source refs
until the event completes. No worker thread mutates engine/cache state.
"""
from dataclasses import dataclass
import time
import torch

from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.engine.session_offload import SessionOffloadManager, SessionOffloadError
from qwen3_runtime.kv.cpu_store import CpuKVCapacityError


class SnapshotCopy:
    def __init__(self, pool, ids, destination, chunk_bytes, stream):
        self.pool = pool
        self.ids = list(ids)
        self.destination = destination
        self.chunk_bytes = chunk_bytes
        self.stream = stream
        self.event = None
        self.started = None
        self.staging = self.indices = None
        self.queued_bytes = 0
        self.accounted = False

    def start(self):
        pool = self.pool
        host = self.destination.permute(2, 0, 1, 3, 4, 5)
        if not pool.cache.is_cuda or not host.is_pinned() or not host.is_contiguous():
            raise ValueError("async copy requires contiguous pinned block-major storage")
        source = pool.cache.permute(2, 0, 1, 3, 4, 5)
        n = max(1, self.chunk_bytes // pool.block_bytes)
        producer = torch.cuda.current_stream(pool.cache.device)
        with torch.cuda.stream(self.stream):
            # Keep every allocation alive through completion; reuse staging only
            # on this ordered stream, after the preceding D2H finishes.
            self.indices = torch.tensor(self.ids, dtype=torch.long, device=pool.cache.device)
            start = time.perf_counter()
            self.staging = torch.empty((min(n, len(self.ids)), *source.shape[1:]),
                                       dtype=source.dtype, device=source.device)
            pool.transfer_stats["staging_allocation_calls"] += 1
            pool.transfer_stats["staging_allocation_s"] += time.perf_counter() - start
            pool.transfer_stats["gpu_staging_peak_bytes"] = max(
                pool.transfer_stats["gpu_staging_peak_bytes"], self.staging.numel() * self.staging.element_size())
            self.stream.wait_stream(producer)
            self.started = torch.cuda.Event(enable_timing=True)
            self.event = torch.cuda.Event(enable_timing=True)
            self.started.record(self.stream)
            for pos in range(0, len(self.ids), n):
                end = min(pos + n, len(self.ids))
                chunk = self.staging[:end-pos]
                torch.index_select(source, 0, self.indices[pos:end], out=chunk)
                size = (end-pos) * pool.block_bytes
                pool.transfer_stats["d2h_attempted_bytes"] += size
                host[pos:end].copy_(chunk, non_blocking=True)
                self.queued_bytes += size
                pool.transfer_stats["d2h_calls"] += 1
            self.event.record(self.stream)

    def ready(self):
        return self.event is not None and self.event.query()

    def finish(self, *, wait):
        # A launch error may leave a partially queued stream without an end event.
        if wait:
            start = time.perf_counter()
            self.stream.synchronize()
            self.pool.transfer_stats["synchronize_calls"] += 1
            self.pool.transfer_stats["synchronize_s"] += time.perf_counter() - start
        elif not self.ready():
            return False
        if not self.accounted:
            self.pool.transfer_stats["d2h_completed_bytes"] += self.queued_bytes
            self.accounted = True
        return True

    def elapsed_ms(self):
        return self.started.elapsed_time(self.event)


@dataclass
class PendingSave:
    req: object
    reservation: object
    table: tuple
    transfer: object


class AsyncSnapshotOffloadManager(SessionOffloadManager):
    def __init__(self, engine):
        super().__init__(engine)
        self.pending = None
        self._copy_stream = None
        self.stats.update(async_submitted=0, async_completed=0, async_cancelled=0,
                          async_fallbacks=0, async_wait_s=0.0, async_enqueue_s=0.0,
                          async_copy_gpu_ms=0.0, async_stale=0, async_steps_inflight=0)

    def _make_transfer(self, req, reservation):
        pool = self.engine.runner.pool
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=pool.cache.device)
        return SnapshotCopy(pool, req.block_table[:reservation.metadata.logical_blocks],
                            reservation.buffer, self.engine.config.transfer_chunk_bytes,
                            self._copy_stream)

    def _can_async(self, reservation):
        return self.engine.runner.pool.cache.is_cuda and reservation.buffer.is_pinned()

    def enqueue_save(self, req, *, session_key):
        if self.pending is not None:
            return False
        if req.status != RequestStatus.PAUSED or not req.block_table:
            raise SessionOffloadError("async save requires a paused resident request")
        meta = self._metadata(req, session_key)
        # Never evict useful snapshots merely to speculate on an early save.
        # The existing synchronous pressure fallback can reclaim host capacity.
        try:
            reservation = self.store.reserve(meta)
        except CpuKVCapacityError:
            self.stats["async_fallbacks"] += 1
            return False
        if not self._can_async(reservation):
            self._group_reservations[req.request_id] = reservation
            self.stats["async_fallbacks"] += 1
            super().save(req, session_key=session_key)
            return False
        try:
            transfer = self._make_transfer(req, reservation)
        except BaseException:
            reservation.abort()
            raise
        self.pending = PendingSave(req, reservation, tuple(req.block_table), transfer)
        self.stats["save_attempts"] += 1
        start = time.perf_counter()
        try:
            transfer.start()
        except BaseException:
            # Wait before releasing any host/staging owner, even after a partial launch.
            self.poll(wait=True, cancel=True)
            self.stats["save_copy_failed"] += 1
            raise
        finally:
            self.stats["async_enqueue_s"] += time.perf_counter() - start
        self.stats["async_submitted"] += 1
        return True

    def poll(self, *, wait=False, cancel=False):
        p = self.pending
        if p is None:
            return []
        start = time.perf_counter()
        try:
            if not p.transfer.finish(wait=wait):
                return []
        except BaseException as exc:
            # Completion is uncertain: retain owners and poison this engine.
            self.engine._fatal_error = "async transfer completion failed: " + str(exc)
            raise
        finally:
            if wait:
                self.stats["async_wait_s"] += time.perf_counter() - start
        meta = p.reservation.metadata
        self.stats["d2h_bytes"] += p.transfer.queued_bytes
        req = p.req
        valid = (self.engine._requests.get(req.request_id) is req
                 and req.status == RequestStatus.PAUSED
                 and tuple(req.block_table) == p.table
                 and tuple(req.token_ids) == meta.token_ids
                 and req.num_computed_tokens == meta.num_computed_tokens
                 and self.engine.block_manager.epoch == meta.kv_epoch
                 and self.weight_epoch == meta.weight_epoch)
        if cancel or not valid:
            p.reservation.abort()
            self.stats["async_cancelled"] += int(cancel)
            self.stats["async_stale"] += int(not valid)
        else:
            tail = meta.num_computed_tokens % meta.block_size
            if tail:
                p.reservation.buffer[:, :, -1, tail:].zero_()
            try:
                snapshot = p.reservation.commit()
            except BaseException:
                p.reservation.abort()
                self.pending = None
                self.stats["save_copy_failed"] += 1
                raise
            self._snapshots[req.request_id] = snapshot
            self.stats["cpu_committed_write_bytes"] += meta.allocated_bytes
            self.engine.block_manager.deallocate(req)
            req.num_computed_tokens = req.num_scheduled_tokens = req.cached_tokens = req.n_published_blocks = 0
            req.kv_residency = "cpu"
            req.kv_epoch = meta.kv_epoch
            req.offload_snapshot_id = meta.snapshot_id
            self.stats["saved"] += 1
            self.stats["async_completed"] += 1
            self.stats["async_copy_gpu_ms"] += p.transfer.elapsed_ms()
        self.pending = None
        return [req.request_id]

    def early_save(self, req, *, session_key):
        return self.enqueue_save(req, session_key=session_key)

    def wait_for(self, request_id):
        if self.pending is not None and self.pending.req.request_id == request_id:
            self.poll(wait=True)

    def save(self, req, *, session_key):
        self.poll(wait=True)
        return super().save(req, session_key=session_key)

    def reclaim_group(self, requests, min_free):
        self.poll(wait=True)
        return super().reclaim_group(requests, min_free)

    def remove(self, request_id):
        if self.pending is not None and self.pending.req.request_id == request_id:
            self.poll(wait=True, cancel=True)
        return super().remove(request_id)

    def invalidate_all(self, *, weight_change=False):
        self.poll(wait=True, cancel=True)
        return super().invalidate_all(weight_change=weight_change)

    def report(self):
        data = super().report()
        data["async_pending_saves"] = int(self.pending is not None)
        data["async_source_bytes"] = 0 if self.pending is None else self.pending.reservation.metadata.allocated_bytes
        return data
