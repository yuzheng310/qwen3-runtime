"""Explicit paged attention: gather K/V from the pool, scores in torch."""

from __future__ import annotations

import math

import torch

from qwen3_runtime.kv.paged import PagedBatch


def causal_allowed(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)


def paged_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Gather K/V from the pool and compute explicit scores. q/k/v: [T, H, D] -> [T, H*D]."""
    paged.pool.store(layer_id, k, v, paged.slot_mapping)
    group = num_attention_heads // num_key_value_heads
    scale = 1.0 / math.sqrt(head_dim)
    chunks: list[torch.Tensor] = []
    for i, (qs, qe) in enumerate(zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])):
        q_len = qe - qs
        kv_len = paged.kv_lens[i]
        query = q[qs:qe]
        key, value = paged.pool.gather(layer_id, paged.block_tables[i], kv_len)
        key = key.repeat_interleave(group, dim=1)
        value = value.repeat_interleave(group, dim=1)
        allowed = causal_allowed(q_len, kv_len, q.device)
        query_h, key_h, value_h = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        scores = torch.matmul(query_h, key_h.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        ctx = torch.matmul(attn, value_h).transpose(0, 1).contiguous().view(q_len, -1)
        chunks.append(ctx)
    return torch.cat(chunks, dim=0)
