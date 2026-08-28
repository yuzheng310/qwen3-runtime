import importlib.util

import pytest
import torch

from qwen3_runtime.layers.rmsnorm import RMSNorm

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="fused_add_rmsnorm needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed"),
]


def test_fused_add_rmsnorm_matches_add_then_rmsnorm():
    from flashinfer import fused_add_rmsnorm, rmsnorm

    torch.manual_seed(0)
    device = torch.device("cuda")
    eps = 1e-6
    x = torch.randn(4, 2560, device=device, dtype=torch.bfloat16)
    residual = torch.randn(4, 2560, device=device, dtype=torch.bfloat16)
    weight = RMSNorm(2560, eps).to(device=device, dtype=torch.bfloat16).weight
    want_res = x + residual
    want = rmsnorm(want_res, weight, eps)
    got_x = x.clone()
    got_res = residual.clone()
    fused_add_rmsnorm(got_x, got_res, weight, eps)
    torch.testing.assert_close(got_res, want_res, atol=0, rtol=0)
    torch.testing.assert_close(got_x, want, atol=2e-2, rtol=2e-2)
