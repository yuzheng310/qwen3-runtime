"""Owner-thread allocation ledger; storage aliases and retired leases count once."""

from __future__ import annotations

import time
import weakref

import torch

from qwen3_runtime.kv.cpu_store import CpuKVCapacityError


class HostBudget:
    def __init__(self, max_bytes, pinned_max_bytes=0):
        self.max_bytes = max_bytes
        self.pinned_max_bytes = pinned_max_bytes
        self.allocations = {}
        self.peak = self.pinned_peak = self.allocation_count = 0
        self.allocation_s = 0.0

    @property
    def owned(self):
        return sum(n for ref, n, pin in self.allocations.values() if ref() is not None)

    @property
    def pinned(self):
        return sum(
            n for ref, n, pin in self.allocations.values() if pin and ref() is not None
        )

    def allocate(self, shape, dtype):
        size = torch.empty((), dtype=dtype).element_size()
        for d in shape:
            size *= d
        if self.owned + size > self.max_bytes:
            raise CpuKVCapacityError(
                f"managed host budget: {self.owned} + {size} > {self.max_bytes}"
            )
        pin = bool(
            torch.cuda.is_available()
            and self.pinned + size <= self.pinned_max_bytes
            and size
        )
        start = time.perf_counter()
        tensor = torch.empty(shape, dtype=dtype, pin_memory=pin)
        self.allocation_s += time.perf_counter() - start
        storage = tensor.untyped_storage()
        ident = storage._cdata

        def release(_ref, ident=ident):
            self.allocations.pop(ident, None)

        self.allocations[ident] = (weakref.ref(storage, release), storage.nbytes(), pin)
        self.allocation_count += 1
        self.peak = max(self.peak, self.owned)
        self.pinned_peak = max(self.pinned_peak, self.pinned)
        return tensor

    def stats(self):
        return dict(
            managed_host_buffer_bytes=self.owned,
            managed_host_buffer_peak_bytes=self.peak,
            managed_pinned_bytes=self.pinned,
            managed_pinned_peak_bytes=self.pinned_peak,
            host_allocation_count=self.allocation_count,
            host_allocation_s=self.allocation_s,
            staging_bytes=0,
        )
