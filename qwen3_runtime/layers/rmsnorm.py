import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            try:
                from flashinfer import rmsnorm
            except ImportError:
                pass
            else:
                return rmsnorm(x, self.weight, self.eps)
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        y = x.float() * torch.rsqrt(var + self.eps)
        return (y * self.weight.float()).to(x.dtype)
