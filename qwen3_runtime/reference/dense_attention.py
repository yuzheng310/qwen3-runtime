"""Dense packed attention. Used when the model runs without a paged KV pool."""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch


def allowed_attention_mask(
    q_len: int, cu_seqlens: Sequence[int] | None, device: torch.device
) -> torch.Tensor:
    """True where attention is allowed. Single-sequence causal if cu_seqlens is None."""
    if cu_seqlens is None:
        return torch.tril(torch.ones(q_len, q_len, dtype=torch.bool, device=device))
    seq_id = torch.empty(q_len, dtype=torch.long, device=device)
    pos = torch.empty(q_len, dtype=torch.long, device=device)
    for i, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        seq_id[start:end] = i
        pos[start:end] = torch.arange(end - start, device=device)
    same = seq_id[:, None] == seq_id[None, :]
    causal = pos[:, None] >= pos[None, :]
    return same & causal


def dense_context(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: Sequence[int] | None,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Explicit scores over packed Q/K/V. Shapes: q/k/v ``[T, H, D]`` -> ``[T, H*D]``."""
    t = q.shape[0]
    group = num_attention_heads // num_key_value_heads
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    allowed = allowed_attention_mask(t, cu_seqlens, q.device)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    return torch.matmul(attn, vh).transpose(0, 1).contiguous().view(t, -1)
