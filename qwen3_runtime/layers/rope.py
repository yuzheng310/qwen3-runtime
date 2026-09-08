from __future__ import annotations

import torch
from torch import nn


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        self.theta = theta
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv)
        self.register_buffer("cos_base", freqs.cos(), persistent=False)
        self.register_buffer("sin_base", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_base[positions].unsqueeze(1)
        sin = self.sin_base[positions].unsqueeze(1)
        return cos, sin


def apply_qwen3_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    rope: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeoX RoPE torch reference. Production calls ``Ops.apply_rope`` instead."""
    cos, sin = rope(positions)
    return apply_rope_neox(q, cos, sin), apply_rope_neox(k, cos, sin)
