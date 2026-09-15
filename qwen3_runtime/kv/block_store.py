"""Bounded slab-backed immutable full blocks and private versioned tails.

Logical manifests never pin cache entries. A synchronous group lease protects
all its inputs and physically reserves every output before any transfer.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from qwen3_runtime.kv.cpu_store import CpuKVCapacityError, CpuKVTransactionError
from qwen3_runtime.kv.host_budget import HostBudget


@dataclass(frozen=True)
class Handle:
    epoch: int
    slot: int
    generation: int


@dataclass
class Slot:
    slab: int
    offset: int
    generation: int = 0
    key: object = None
    state: str = "FREE"
    pins: int = 0
    valid: int = 0


class Lease:
    def __init__(self, store, keys, handles):
        self.store, self.keys, self.handles = store, tuple(keys), handles
        self.epoch, self.closed = store.epoch, False

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.epoch != self.store.epoch:
            return
        for handle in self.handles.values():
            slot = self.store.resolve(handle, ready=False)
            if slot is None:
                continue
            slot.pins -= 1
            if slot.state == "WRITING":
                self.store._free(handle.slot)


class BlockStore:
    def __init__(
        self,
        max_bytes,
        pinned_max_bytes,
        block_shape,
        dtype,
        slab_bytes,
        *,
        demand_sized=False,
    ):
        self.demand_sized = demand_sized
        self.max_bytes, self.pinned_max_bytes = max_bytes, pinned_max_bytes
        self.budget = HostBudget(max_bytes, pinned_max_bytes)
        self.block_shape, self.dtype = tuple(block_shape), dtype
        import torch

        self.block_bytes = torch.empty((), dtype=dtype).element_size()
        for d in block_shape:
            self.block_bytes *= d
        self.slab_blocks = max(1, slab_bytes // self.block_bytes)
        self.slabs, self.slots, self.free = [], [], []
        self.index = OrderedDict()
        self.epoch = 0
        self.evicted = 0

    def handle(self, index):
        return Handle(self.epoch, index, self.slots[index].generation)

    def resolve(self, handle, ready=True):
        if handle.epoch != self.epoch or handle.slot >= len(self.slots):
            return None
        slot = self.slots[handle.slot]
        if slot.generation != handle.generation or slot.state == "FREE":
            return None
        return slot if not ready or slot.state == "READY" else None

    def lookup(self, key):
        i = self.index.get(key)
        if i is None or self.slots[i].state != "READY":
            return None
        self.index.move_to_end(key)
        return self.handle(i)

    def view(self, handles):
        slots = [self.resolve(h, ready=False) for h in handles]
        if not slots or any(s is None for s in slots):
            raise CpuKVTransactionError("stale block handle")
        first = slots[0]
        if any(
            s.slab != first.slab or s.offset != first.offset + j
            for j, s in enumerate(slots)
        ):
            raise ValueError("transfer run is not contiguous")
        return self.slabs[first.slab][first.offset : first.offset + len(slots)].permute(
            1, 2, 0, 3, 4, 5
        )

    def runs(self, keys, max_blocks=None):
        result = []
        for key in keys:
            i = self.index[key]
            s = self.slots[i]
            if result:
                prev = self.slots[self.index[result[-1][-1]]]
                if (
                    s.slab == prev.slab
                    and s.offset == prev.offset + 1
                    and (max_blocks is None or len(result[-1]) < max_blocks)
                ):
                    result[-1].append(key)
                    continue
            result.append([key])
        return result

    def _free(self, index):
        slot = self.slots[index]
        if slot.pins:
            raise CpuKVTransactionError("cannot evict transfer-protected block")
        self.index.pop(slot.key, None)
        slot.key, slot.state, slot.valid = None, "FREE", 0
        self.free.append(index)

    def evict(self, key):
        i = self.index.get(key)
        if i is None or self.slots[i].pins:
            return False
        self._free(i)
        self.evicted += 1
        return True

    def reserve(self, keys, *, protect=()):
        keys = tuple(dict.fromkeys(keys))
        protected = set(keys) | set(protect)
        if len(protected) * self.block_bytes > self.max_bytes:
            raise CpuKVCapacityError("complete protected block set exceeds budget")
        missing = [k for k in keys if k not in self.index]
        if any(
            self.slots[self.index[k]].state != "READY" for k in keys if k in self.index
        ):
            raise CpuKVTransactionError("overlapping open block reservation")
        # Fail before eviction if pinned/unreleasable allocations make it impossible.
        growth = max(0, (self.max_bytes - self.budget.owned) // self.block_bytes)
        reclaimable = sum(
            not s.pins and s.state == "READY" and s.key not in protected
            for s in self.slots
        )
        if len(missing) > len(self.free) + growth + reclaimable:
            raise CpuKVCapacityError(
                "protected or externally-held host storage prevents reservation"
            )
        held = []
        try:
            for key in protected:
                if key in self.index:
                    i = self.index[key]
                    self.slots[i].pins += 1
                    held.append(self.handle(i))
            for key in tuple(self.index):
                if len(self.free) + growth >= len(missing):
                    break
                if key not in protected and self.evict(key):
                    pass
            while len(self.free) < len(missing):
                count = min(
                    self.slab_blocks,
                    (self.max_bytes - self.budget.owned) // self.block_bytes,
                )
                if self.demand_sized:
                    count = min(count, len(missing) - len(self.free))
                if count <= 0:
                    raise CpuKVCapacityError("slab capacity exhausted")
                slab = self.budget.allocate((count, *self.block_shape), self.dtype)
                sid = len(self.slabs)
                self.slabs.append(slab)
                for offset in range(count):
                    self.free.append(len(self.slots))
                    self.slots.append(Slot(sid, offset))
            # Allocate in physical order so adjacent logical blocks share one DMA run.
            self.free.sort(reverse=True)
            for key in missing:
                i = self.free.pop()
                s = self.slots[i]
                s.generation += 1
                s.key = key
                s.state = "WRITING"
                s.pins = 1
                self.index[key] = i
                held.append(self.handle(i))
            handles = {self.slots[h.slot].key: h for h in held}
            return Lease(self, keys, handles)
        except BaseException:
            for h in held:
                s = self.resolve(h, ready=False)
                if s:
                    s.pins -= 1
                    if s.state == "WRITING":
                        self._free(h.slot)
            raise

    def commit(self, handle, valid):
        slot = self.resolve(handle, ready=False)
        if slot is None or slot.state != "WRITING" or not slot.pins:
            raise CpuKVTransactionError("block reservation is not writable")
        slot.state = "READY"
        slot.valid = valid

    def invalidate_all(self, release=True):
        self.epoch += 1
        self.index.clear()
        if release:
            self.slabs.clear()
            self.slots.clear()
            self.free.clear()
        else:
            self.free = list(range(len(self.slots)))
            for s in self.slots:
                s.state = "FREE"
                s.key = None
                s.pins = 0
                s.valid = 0
        return []

    def stats(self):
        ready = sum(s.state == "READY" for s in self.slots)
        reserved = sum(s.state == "WRITING" for s in self.slots)
        return {
            **self.budget.stats(),
            "max_bytes": self.max_bytes,
            "committed_bytes": ready * self.block_bytes,
            "reserved_bytes": reserved * self.block_bytes,
            "inflight_bytes": reserved * self.block_bytes,
            "free_slot_bytes": len(self.free) * self.block_bytes,
            "pinned_bytes": self.budget.pinned,
            "snapshots": ready,
            "ready_blocks": ready,
            "reserved_blocks": reserved,
            "transfer_pins": sum(s.pins for s in self.slots),
            "cpu_evictions": self.evicted,
            "resident_valid_kv_payload_bytes": sum(
                s.valid for s in self.slots if s.state == "READY"
            )
            * self.block_bytes
            // self.block_shape[2],
        }
