from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass
class PagedBatch:
    pool: "PagedKVPool"
    slot_mapping: torch.Tensor
    block_tables: list
    kv_lens: list[int]
    cu_seqlens: list[int]
    # Filled on first FlashInfer layer of a forward so plan() is once, not per layer.
    flashinfer_mode: str | None = None
    # Optional decode wrapper (CUDA-graph persistent buffers). None = shared eager wrapper.
    flashinfer_wrapper: object | None = None
    # Graph decode: FlashInfer append_paged_kv_cache metadata (int32, device).
    append_batch_indices: torch.Tensor | None = None
    append_positions: torch.Tensor | None = None
    # Packed page table for the Triton decode backend; filled once per forward.
    triton_block_table: torch.Tensor | None = None
    triton_kv_lens: torch.Tensor | None = None
    # Frozen split-K count for CUDA Graph capture. None = pick from kv_len.
    triton_num_splits: int | None = None


class PagedKVPool:
    """Physical KV blocks. Layout: [K/V, layer, block, offset, kv_head, dim].

    Not a per-request [max_seq_len] tensor. Slot = block_id * block_size + offset.
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
