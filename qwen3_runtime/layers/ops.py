"""Fused operators. Chosen once at device placement, not inside ``forward``.

Each FlashInfer kernel has a torch reference of a few lines. CUDA Graph capture
sees a frozen callable, so a missing import cannot silently swap implementations
mid-step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)


def torch_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def torch_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """residual := residual + x; x := rmsnorm(residual). Out-of-place."""
    residual = residual + x
    x = torch_rmsnorm(residual, weight, eps)
    return x, residual


def torch_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    rope,
) -> tuple[torch.Tensor, torch.Tensor]:
    from qwen3_runtime.layers.rope import apply_rope_neox

    cos, sin = rope(positions)
    return apply_rope_neox(q, cos, sin), apply_rope_neox(k, cos, sin)


@dataclass(frozen=True)
class Ops:
    rmsnorm: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
    silu_and_mul: Callable[[torch.Tensor], torch.Tensor]
    fused_add_rmsnorm: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]
    ]
    apply_rope: Callable

    @classmethod
    def torch(cls) -> Ops:
        return cls(
            rmsnorm=torch_rmsnorm,
            silu_and_mul=torch_silu_and_mul,
            fused_add_rmsnorm=torch_fused_add_rmsnorm,
            apply_rope=torch_apply_rope,
        )

    @classmethod
    def flashinfer(cls) -> Ops:
        from flashinfer import apply_rope_pos_ids
        from flashinfer import fused_add_rmsnorm as fi_fused_add_rmsnorm
        from flashinfer import rmsnorm as fi_rmsnorm
        from flashinfer import silu_and_mul as fi_silu_and_mul

        def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
            return fi_rmsnorm(x, weight, eps)

        def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
            return fi_silu_and_mul(gate_up)

        def fused_add_rmsnorm(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            fi_fused_add_rmsnorm(x, residual, weight, eps)
            return x, residual

        def apply_rope(
            q: torch.Tensor,
            k: torch.Tensor,
            positions: torch.Tensor,
            rope,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return apply_rope_pos_ids(
                q, k, positions, interleave=False, rope_theta=rope.theta
            )

        return cls(
            rmsnorm=rmsnorm,
            silu_and_mul=silu_and_mul,
            fused_add_rmsnorm=fused_add_rmsnorm,
            apply_rope=apply_rope,
        )

    @classmethod
    def select(cls, device: torch.device | str) -> Ops:
        kind = device if isinstance(device, str) else device.type
        if kind == "cuda":
            try:
                return cls.flashinfer()
            except ImportError:
                return cls.torch()
        return cls.torch()
