"""Device paged KV pool and the five-field PagedBatch the model consumes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import time

import torch


@dataclass
class PagedBatch:
    """One packed forward. Five fields; backend scratch lives on ``AttentionState``."""

    pool: PagedKVPool
    slot_mapping: torch.Tensor  # [T] physical slots for the scheduled tokens
    block_tables: list  # per-sequence physical page ids
    kv_lens: list[int]  # per-sequence KV length after this forward
    cu_seqlens: list[int]  # packed query prefix sums, length = batch + 1


class PagedKVPool:
    """Physical KV blocks. Layout: [K/V, layer, block, offset, kv_head, dim].

    Not a per-request [max_seq_len] tensor. Slot = block_id * block_size + offset.

    4B default: [2, 36, num_blocks, 16, 8, 128].
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.transfer_stats = dict(
            d2h_completed_bytes=0,
            d2h_unconfirmed_bytes=0,
            h2d_unconfirmed_bytes=0,
            h2d_completed_bytes=0,
            d2h_attempted_bytes=0,
            h2d_attempted_bytes=0,
            d2h_calls=0,
            h2d_calls=0,
            synchronize_calls=0,
            synchronize_s=0.0,
            blocking_copy_calls=0,
            staging_allocation_s=0.0,
            staging_allocation_calls=0,
            gpu_staging_peak_bytes=0,
        )
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache = torch.zeros(
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

    def store(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        block_ids = slot_mapping // self.block_size
        offsets = slot_mapping % self.block_size
        self.cache[0, layer, block_ids, offsets] = key
        self.cache[1, layer, block_ids, offsets] = value

    def copy_block(self, src: int, dst: int) -> None:
        if src == dst:
            return
        self.cache[:, :, dst].copy_(self.cache[:, :, src])

    @property
    def block_bytes(self) -> int:
        """Bytes in one physical block, including K and V for every layer."""
        return (
            2
            * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.cache.element_size()
        )

    def export_blocks(
        self,
        block_ids: Sequence[int],
        *,
        valid_tokens: int,
        chunk_bytes: int,
        destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Copy logical blocks to CPU in bounded staging chunks.

        ``block_ids`` is the logical order of a request's block table.  The
        returned tensor is self-contained and keeps the pool dtype.  Padding
        after ``valid_tokens`` is zeroed so it can never be mistaken for valid
        KV after a later restore.
        """
        ids = [int(block_id) for block_id in block_ids]
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in ids):
            raise ValueError("block id outside the KV pool")
        if valid_tokens < 0 or valid_tokens > len(ids) * self.block_size:
            raise ValueError("valid_tokens does not fit the supplied block table")
        shape = (
            2,
            self.num_layers,
            len(ids),
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        if destination is None:
            destination = torch.empty(
                (
                    len(ids),
                    2,
                    self.num_layers,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_dim,
                ),
                dtype=self.cache.dtype,
                device="cpu",
            ).permute(1, 2, 0, 3, 4, 5)
        if tuple(destination.shape) != shape or destination.dtype != self.cache.dtype:
            raise ValueError("destination has the wrong KV snapshot layout")
        if destination.device.type != "cpu":
            raise ValueError("KV snapshots must be stored on CPU")
        if not ids:
            return destination
        per_block = max(1, self.block_bytes)
        blocks_per_chunk = max(1, chunk_bytes // per_block)
        device_blocks = self.cache.permute(2, 0, 1, 3, 4, 5)
        host_blocks = destination.permute(2, 0, 1, 3, 4, 5)
        indices = torch.tensor(ids, dtype=torch.long, device=self.cache.device)
        allocation_start = time.perf_counter()
        staging = torch.empty(
            (min(blocks_per_chunk, len(ids)), *device_blocks.shape[1:]),
            dtype=self.cache.dtype,
            device=self.cache.device,
        )
        self.transfer_stats["staging_allocation_s"] += (
            time.perf_counter() - allocation_start
        )
        self.transfer_stats["staging_allocation_calls"] += 1
        self.transfer_stats["gpu_staging_peak_bytes"] = max(
            self.transfer_stats["gpu_staging_peak_bytes"],
            staging.numel() * staging.element_size() if self.cache.is_cuda else 0,
        )
        async_copy = (
            self.cache.is_cuda
            and destination.is_pinned()
            and host_blocks.is_contiguous()
        )
        completed_before = self.transfer_stats["d2h_completed_bytes"]
        try:
            for start in range(0, len(ids), blocks_per_chunk):
                end = min(len(ids), start + blocks_per_chunk)
                chunk = staging[: end - start]
                torch.index_select(device_blocks, 0, indices[start:end], out=chunk)
                self.transfer_stats["d2h_attempted_bytes"] += (end - start) * per_block
                host_blocks[start:end].copy_(chunk, non_blocking=async_copy)
                self.transfer_stats["d2h_completed_bytes"] += (end - start) * per_block
                self.transfer_stats["d2h_calls"] += 1
                self.transfer_stats["blocking_copy_calls"] += int(
                    self.cache.is_cuda and not async_copy
                )
        finally:
            # This API remains synchronous: host ownership is committed only
            # after every copy completes, including when a later chunk fails.
            if async_copy:
                sync_start = time.perf_counter()
                try:
                    torch.cuda.current_stream(self.cache.device).synchronize()
                except BaseException:
                    self.transfer_stats["d2h_unconfirmed_bytes"] += (
                        self.transfer_stats["d2h_completed_bytes"] - completed_before
                    )
                    self.transfer_stats["d2h_completed_bytes"] = completed_before
                    raise
                finally:
                    self.transfer_stats["synchronize_s"] += (
                        time.perf_counter() - sync_start
                    )
                self.transfer_stats["synchronize_calls"] += 1
        if valid_tokens == 0:
            destination.zero_()
        else:
            tail = valid_tokens % self.block_size
            if tail:
                destination[:, :, -1, tail:].zero_()
        return destination

    def import_blocks(
        self,
        block_ids: Sequence[int],
        source: torch.Tensor,
        *,
        valid_tokens: int,
        chunk_bytes: int,
        source_block_offset: int = 0,
    ) -> None:
        """Write CPU snapshot blocks into physical pool blocks in chunks."""
        ids = [int(block_id) for block_id in block_ids]
        if (
            source.ndim != 6
            or source.dtype != self.cache.dtype
            or source.device.type != "cpu"
        ):
            raise ValueError("source has the wrong KV snapshot layout")
        if source.shape[0] != 2 or source.shape[1] != self.num_layers:
            raise ValueError("source has the wrong KV layer layout")
        if source.shape[3:] != (self.block_size, self.num_kv_heads, self.head_dim):
            raise ValueError("source has the wrong KV block layout")
        if source_block_offset < 0 or source_block_offset + len(ids) > source.shape[2]:
            raise ValueError("source block range is outside the snapshot")
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in ids):
            raise ValueError("block id outside the KV pool")
        if valid_tokens < 0 or valid_tokens > source.shape[2] * self.block_size:
            raise ValueError("valid_tokens does not fit the source snapshot")
        per_block = max(1, self.block_bytes)
        blocks_per_chunk = max(1, chunk_bytes // per_block)
        if not ids:
            return
        device_blocks = self.cache.permute(2, 0, 1, 3, 4, 5)
        host_blocks = source.permute(2, 0, 1, 3, 4, 5)
        indices = torch.tensor(ids, dtype=torch.long, device=self.cache.device)
        allocation_start = time.perf_counter()
        staging = torch.empty(
            (min(blocks_per_chunk, len(ids)), *device_blocks.shape[1:]),
            dtype=self.cache.dtype,
            device=self.cache.device,
        )
        self.transfer_stats["staging_allocation_s"] += (
            time.perf_counter() - allocation_start
        )
        self.transfer_stats["staging_allocation_calls"] += 1
        self.transfer_stats["gpu_staging_peak_bytes"] = max(
            self.transfer_stats["gpu_staging_peak_bytes"],
            staging.numel() * staging.element_size() if self.cache.is_cuda else 0,
        )
        async_copy = (
            self.cache.is_cuda and source.is_pinned() and host_blocks.is_contiguous()
        )
        completed_before = self.transfer_stats["h2d_completed_bytes"]
        try:
            for start in range(0, len(ids), blocks_per_chunk):
                end = min(len(ids), start + blocks_per_chunk)
                chunk = staging[: end - start]
                self.transfer_stats["h2d_attempted_bytes"] += (end - start) * per_block
                chunk.copy_(
                    host_blocks[
                        source_block_offset + start : source_block_offset + end
                    ],
                    non_blocking=async_copy,
                )
                device_blocks.index_copy_(0, indices[start:end], chunk)
                self.transfer_stats["h2d_completed_bytes"] += (end - start) * per_block
                self.transfer_stats["h2d_calls"] += 1
                self.transfer_stats["blocking_copy_calls"] += int(
                    self.cache.is_cuda and not async_copy
                )
        finally:
            if async_copy:
                sync_start = time.perf_counter()
                try:
                    torch.cuda.current_stream(self.cache.device).synchronize()
                except BaseException:
                    self.transfer_stats["h2d_unconfirmed_bytes"] += (
                        self.transfer_stats["h2d_completed_bytes"] - completed_before
                    )
                    self.transfer_stats["h2d_completed_bytes"] = completed_before
                    raise
                finally:
                    self.transfer_stats["synchronize_s"] += (
                        time.perf_counter() - sync_start
                    )
                self.transfer_stats["synchronize_calls"] += 1

    def gather(
        self,
        layer: int,
        block_table: Sequence[int] | torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.cache.device
        pos = torch.arange(seq_len, device=device)
        block_index = pos // self.block_size
        offsets = pos % self.block_size
        if isinstance(block_table, torch.Tensor):
            table = block_table
            if table.device != device or table.dtype != torch.long:
                table = table.to(device=device, dtype=torch.long)
        else:
            table = torch.tensor(block_table, device=device, dtype=torch.long)
        phys = table[block_index]
        key = self.cache[0, layer, phys, offsets]
        value = self.cache[1, layer, phys, offsets]
        return key, value
