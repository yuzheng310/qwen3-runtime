"""Triton paged decode. Prefill is the pytorch reference, not a silent mix."""

from __future__ import annotations

import torch

from qwen3_runtime.attention.flashinfer_backend import store_flashinfer_kv
from qwen3_runtime.attention.pytorch import paged_pytorch
from qwen3_runtime.attention.state import AttentionState
from qwen3_runtime.kv.paged import PagedBatch


def ensure_triton_pages(
    paged: PagedBatch, device: torch.device, state: AttentionState
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.triton_block_table is not None and state.triton_kv_lens is not None:
        return state.triton_block_table, state.triton_kv_lens
    page_size = paged.pool.block_size
    n_pages = [(int(kv) + page_size - 1) // page_size for kv in paged.kv_lens]
    max_p = max(n_pages) if n_pages else 0
    table = torch.zeros(len(paged.kv_lens), max_p, dtype=torch.int32, device=device)
    for i, bt in enumerate(paged.block_tables):
        need = n_pages[i]
        if isinstance(bt, torch.Tensor):
            pages = bt[:need].to(device=device, dtype=torch.int32)
        else:
            pages = torch.tensor(bt[:need], dtype=torch.int32, device=device)
        if pages.numel() < need:
            raise ValueError("block table shorter than occupied pages")
        table[i, :need] = pages
    kv = torch.tensor(paged.kv_lens, dtype=torch.int32, device=device)
    state.triton_block_table = table
    state.triton_kv_lens = kv
    return table, kv


def paged_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    state: AttentionState,
) -> torch.Tensor:
    """Decode uses the Triton paged kernel. Prefill is the pytorch reference."""
    q_lens = [qe - qs for qs, qe in zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])]
    decode_only = bool(q_lens) and all(ql == 1 for ql in q_lens)
    if not decode_only:
        return paged_pytorch(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    store_flashinfer_kv(paged, layer_id, k, v, state)
    from qwen3_runtime.kernels.triton_decode import paged_decode_attention

    table, kv_lens = ensure_triton_pages(paged, q.device, state)
    out = paged_decode_attention(
        q,
        paged.pool.cache[0, layer_id],
        paged.pool.cache[1, layer_id],
        table,
        kv_lens,
        page_size=paged.pool.block_size,
        max_kv_len=max(paged.kv_lens),
        num_splits=state.triton_num_splits,
    )
    return out.contiguous().view(q.shape[0], num_attention_heads * head_dim)
