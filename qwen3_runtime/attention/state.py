"""Backend scratch for one forward (or one CUDA Graph)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionState:
    """Plan/page tensors filled on the first layer; later layers only read."""

    flashinfer_mode: str | None = None
    flashinfer_wrapper: object | None = None
    append_batch_indices: torch.Tensor | None = None
    append_positions: torch.Tensor | None = None
    kv_indptr_buf: torch.Tensor | None = None
    kv_indices_buf: torch.Tensor | None = None
    kv_last_page_len_buf: torch.Tensor | None = None
    triton_block_table: torch.Tensor | None = None
    triton_kv_lens: torch.Tensor | None = None
    triton_num_splits: int | None = None
