"""Session-manifest-only tiered block cache. New requests use GPU APC only."""

from dataclasses import dataclass

from qwen3_runtime.engine.prefix_cache import ROOT_HASH, block_hash
from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.engine.session_offload import (
    RestoreAdmissionError,
    RestoreTicket,
    SessionOffloadError,
    SessionOffloadManager,
    SnapshotStaleError,
    fatal_device_error,
)
from qwen3_runtime.kv.block_store import BlockStore
from qwen3_runtime.kv.cpu_store import CpuKVCapacityError


@dataclass(frozen=True)
class Manifest:
    metadata: object
    namespace: tuple
    keys: tuple
    hashes: tuple


class BlockOffloadManager(SessionOffloadManager):
    def __init__(self, engine):
        super().__init__(engine)
        self.store = None
        self._block_group = None
        self._cohorts = {}
        self._pending_use = {}
        self.stats.update(
            d2h_skipped_valid_replica_bytes=0,
            new_request_cpu_loads=0,
            d2h_failed_bytes=0,
            restore_missing_blocks=0,
        )
        self._retired_used = self._retired_unused = 0
        self._ensure_store()

    def _ensure_store(self):
        p = self.engine.runner.pool
        if self.store is None:
            c = self.engine.config
            self.store = BlockStore(
                c.cpu_kv_max_bytes,
                c.cpu_kv_pinned_max_bytes,
                (2, p.num_layers, p.block_size, p.num_kv_heads, p.head_dim),
                p.cache.dtype,
                c.cpu_kv_slab_bytes or c.transfer_chunk_bytes,
                demand_sized=bool(c.cpu_kv_slab_bytes),
            )
        return self.store

    def _transfer_runs(self, keys):
        # Preserve bounded transfer/synchronization groups when host slabs grow.
        # This separates allocation effects from a synchronization policy change.
        return self.store.runs(
            keys,
            max_blocks=max(
                1, self.engine.config.transfer_chunk_bytes // self.store.block_bytes
            ),
        )

    def _namespace(self):
        p = self.engine.runner.pool
        return (
            id(self.engine.runner),
            repr(getattr(self.engine.runner.model, "cfg", None)),
            self.weight_epoch,
            self.engine.block_manager.epoch,
            str(p.cache.dtype),
            p.num_layers,
            p.block_size,
            p.num_kv_heads,
            p.head_dim,
        )

    def _manifest(self, req, session_key):
        meta = self._metadata(req, session_key)
        namespace = self._namespace()
        parent = ROOT_HASH
        keys = []
        hashes = []
        for i in range(meta.num_computed_tokens // meta.block_size):
            parent = block_hash(
                parent, req.token_ids[i * meta.block_size : (i + 1) * meta.block_size]
            )
            keys.append((namespace, parent))
            hashes.append(parent)
        if meta.num_computed_tokens % meta.block_size:
            keys.append((namespace, "tail", req.request_id, meta.snapshot_id))
            hashes.append(None)
        return Manifest(meta, namespace, tuple(keys), tuple(hashes))

    def prepare_group(self, requests):
        store = self._ensure_store()
        manifests = {
            r.request_id: self._manifest(r, str(r.request_id)) for r in requests
        }
        keys = [k for m in manifests.values() for k in m.keys]
        # An old private tail remains charged and protected until manifest commit.
        oldtails = [
            k
            for r in requests
            for k in getattr(self._snapshots.get(r.request_id), "keys", ())
            if len(k) > 2
        ]
        lease = store.reserve(keys, protect=oldtails)
        self._block_group = (manifests, lease)

    def close_group(self):
        if self._block_group:
            self._block_group[1].close()
            self._block_group = None
            live_tails = {
                k for m in self._snapshots.values() for k in m.keys if len(k) > 2
            }
            for k in tuple(self.store.index):
                if len(k) > 2 and k not in live_tails:
                    self.store.evict(k)

    def save(self, req, *, session_key):
        if not self.enabled or req.status != RequestStatus.PAUSED:
            raise SessionOffloadError("only paused requests may be saved")
        if not req.block_table and req.request_id in self._snapshots:
            return self._snapshots[req.request_id]
        store = self._ensure_store()
        self.stats["save_attempts"] += 1
        own = self._block_group is None
        try:
            if own:
                self.prepare_group([req])
        except CpuKVCapacityError:
            self.stats["save_capacity_failed"] += 1
            raise
        manifests, lease = self._block_group
        m = manifests[req.request_id]
        meta = m.metadata
        old = self._snapshots.get(req.request_id)
        try:
            missing = [k for k in m.keys if store.lookup(k) is None]
            self.stats["d2h_skipped_valid_replica_bytes"] += (
                len(m.keys) - len(missing)
            ) * store.block_bytes
            positions = {k: i for i, k in enumerate(m.keys)}
            for keys in self._transfer_runs(missing):
                handles = [lease.handles[k] for k in keys]
                ids = [req.block_table[positions[k]] for k in keys]
                valid = len(keys) * meta.block_size
                if (
                    keys[-1] == m.keys[-1]
                    and meta.num_computed_tokens % meta.block_size
                ):
                    valid -= (
                        meta.block_size - meta.num_computed_tokens % meta.block_size
                    )
                self._export_blocks(
                    ids,
                    valid_tokens=valid,
                    chunk_bytes=self.engine.config.transfer_chunk_bytes,
                    destination=store.view(handles),
                )
                if m.namespace != self._namespace():
                    raise SnapshotStaleError("namespace changed during block save")
                for j, (k, h) in enumerate(zip(keys, handles)):
                    n = (
                        meta.block_size
                        if j < len(keys) - 1
                        else valid - (len(keys) - 1) * meta.block_size
                    )
                    store.commit(h, n)
                    self.stats["cpu_committed_write_bytes"] += store.block_bytes
                    self._cohorts[h] = [store.block_bytes, False]
            if any(store.lookup(k) is None for k in m.keys):
                raise SnapshotStaleError("incomplete save")
            self._snapshots[req.request_id] = m
            self.engine.block_manager.deallocate(req)
            req.num_computed_tokens = req.num_scheduled_tokens = req.cached_tokens = (
                req.n_published_blocks
            ) = 0
            req.kv_epoch = meta.kv_epoch
            req.kv_residency = "cpu"
            req.offload_snapshot_id = meta.snapshot_id
            self.stats["saved"] += 1
            return m
        except BaseException as exc:
            if fatal_device_error(exc):
                self.engine._fatal_error = str(exc)
            self.stats["save_copy_failed"] += 1
            raise
        finally:
            if own:
                self.close_group()
            self._prune_cohorts()
            if old and self._snapshots.get(req.request_id) is not old:
                for k in old.keys:
                    if len(k) > 2:
                        store.evict(k)

    def begin_restore(self, req):
        m = self._snapshots.get(req.request_id)
        if (
            m is None
            or m.namespace != self._namespace()
            or tuple(req.token_ids) != m.metadata.token_ids
        ):
            raise SnapshotStaleError("session manifest identity/history mismatch")
        if req.block_table:
            raise RestoreAdmissionError(
                "restore requires released GPU session references"
            )
        store = self._ensure_store()
        bm = self.engine.block_manager
        meta = m.metadata
        plan = []
        pins = []
        new = []
        attached = 0
        imported = 0
        try:
            # Pin every GPU hit now, before any allocator eviction. CPU entries
            # are protected for the entire synchronous plan, including holes.
            for k, hsh in zip(m.keys, m.hashes):
                bid = bm._cache.lookup(hsh) if bm.enable_prefix_cache and hsh else None
                if bid is not None and bm._ref_count[bid] > 0:
                    bm._incref(bid)
                    pins.append(bid)
                    plan.append(("gpu", bid, k))
                    attached += meta.block_size
                else:
                    h = store.lookup(k)
                    if h is None:
                        break
                    store.resolve(h).pins += 1
                    plan.append(("cpu", h, k))
            valid = min(meta.num_computed_tokens, len(plan) * meta.block_size)
            needed = sum(kind == "cpu" for kind, _, _ in plan) + int(
                valid % meta.block_size == 0
            )
            bm.reclaim_cached_blocks(needed)
            if bm.num_free_blocks < needed:
                raise RestoreAdmissionError(
                    "history plus next execution token exceeds available KV pool"
                )
            for kind, loc, k in plan:
                if kind == "gpu":
                    req.block_table.append(loc)
                else:
                    bid = bm._alloc_block()
                    new.append(bid)
                    req.block_table.append(bid)
            cpu_items = [
                (i, loc, k) for i, (kind, loc, k) in enumerate(plan) if kind == "cpu"
            ]
            bykey = {k: (i, h) for i, h, k in cpu_items}
            for keys in self._transfer_runs([k for _, _, k in cpu_items]):
                hs = [bykey[k][1] for k in keys]
                ids = [req.block_table[bykey[k][0]] for k in keys]
                self.engine.runner.pool.import_blocks(
                    ids,
                    store.view(hs),
                    valid_tokens=len(keys) * meta.block_size,
                    chunk_bytes=self.engine.config.transfer_chunk_bytes,
                )
                self.stats["h2d_bytes"] += len(keys) * store.block_bytes
                for k in keys:
                    i, h = bykey[k]
                    imported += min(meta.block_size, valid - i * meta.block_size)
            req.num_computed_tokens = valid
            req.num_scheduled_tokens = 0
            req.kv_epoch = bm.epoch
            req.kv_residency = "gpu"
            req.cached_tokens = attached
            req.n_published_blocks = 0
            req.prefix_parent = ROOT_HASH
            self.stats["restore_missing_blocks"] += len(m.keys) - len(plan)
            self._pending_use[req.request_id] = (
                [h for _, h, _ in cpu_items],
                imported,
                valid,
                tuple(req.block_table),
            )
            return RestoreTicket(
                req.request_id,
                meta.snapshot_id,
                attached,
                imported,
                len(cpu_items) * store.block_bytes,
            )
        except BaseException as exc:
            for bid in pins + new:
                bm._decref(bid)
            req.block_table.clear()
            req.num_computed_tokens = 0
            req.num_scheduled_tokens = 0
            req.kv_residency = "cpu"
            self.stats["restore_failed"] += 1
            if fatal_device_error(exc):
                self.engine._fatal_error = str(exc)
                raise
            raise RestoreAdmissionError("block restore rolled back") from exc
        finally:
            for kind, loc, k in plan:
                if kind == "cpu":
                    slot = store.resolve(loc)
                    if slot:
                        slot.pins -= 1

    def commit_restore(self, ticket):
        m = self._snapshots.get(ticket.request_id)
        if m is None or m.metadata.snapshot_id != ticket.snapshot_id:
            raise SnapshotStaleError("stale restore ticket")
        self.stats["restored"] += 1

    def mark_used(self, requests):
        for req in requests:
            rid = req.request_id
            item = self._pending_use.pop(rid, None)
            if item is None:
                continue
            hs, tokens, valid, table = item
            if (
                req.num_computed_tokens < valid
                or tuple(req.block_table[: len(table)]) != table
            ):
                continue
            for h in hs:
                if h in self._cohorts:
                    self._cohorts[h][1] = True
            self.stats["useful_restores"] += int(tokens > 0)
            self.stats["imported_tokens"] += tokens

    def rollback_restore(self, req, ticket):
        super().rollback_restore(req, ticket)
        self._pending_use.pop(req.request_id, None)

    def remove(self, request_id):
        m = self._snapshots.pop(request_id, None)
        self._pending_use.pop(request_id, None)
        if m and self.store:
            for k in m.keys:
                if len(k) > 2:
                    self.store.evict(k)

    def invalidate_all(self, *, weight_change=False):
        self.close_group()
        if weight_change:
            self.weight_epoch += 1
        if self.store:
            self.store.invalidate_all()
        for rid in self._snapshots:
            req = self.engine._requests.get(rid)
            if req:
                req.offload_snapshot_id = None
                req.kv_residency = "none"
        self.stats["invalidated"] += len(self._snapshots)
        self._snapshots.clear()
        self._pending_use.clear()
        self._prune_cohorts()

    def _prune_cohorts(self):
        pending_handles = {h for item in self._pending_use.values() for h in item[0]}
        for h, (size, seen) in tuple(self._cohorts.items()):
            if (
                self.store is None or self.store.resolve(h) is None
            ) and h not in pending_handles:
                if seen:
                    self._retired_used += size
                else:
                    self._retired_unused += size
                del self._cohorts[h]

    def report(self):
        self._prune_cohorts()
        data = dict(self.stats)
        data["uncommitted_d2h_bytes"] = (
            self.stats["d2h_bytes"] - self.stats["cpu_committed_write_bytes"]
        )
        used = self._retired_used
        retired = self._retired_unused
        pending = 0
        for h, (size, seen) in self._cohorts.items():
            if seen:
                used += size
            elif self.store and self.store.resolve(h):
                pending += size
            else:
                retired += size
        data.update(
            restored_d2h_bytes=used, unused_d2h_bytes=retired, pending_d2h_bytes=pending
        )
        data.update(
            {
                "cpu_" + k: v
                for k, v in (
                    self.store.stats()
                    if self.store
                    else {"committed_bytes": 0, "reserved_bytes": 0, "snapshots": 0}
                ).items()
            }
        )
        data["manifests"] = len(self._snapshots)
        return data
