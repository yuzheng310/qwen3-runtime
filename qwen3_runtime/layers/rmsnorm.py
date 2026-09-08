from __future__ import annotations

import torch
from torch import nn

from qwen3_runtime.layers.ops import Ops


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6, ops: Ops | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))
        self.ops = ops or Ops.torch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops.rmsnorm(x, self.weight, self.eps)
