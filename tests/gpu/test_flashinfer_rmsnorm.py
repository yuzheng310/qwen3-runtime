import importlib.util

import pytest
import torch

from qwen3_runtime.layers.ops import Ops, torch_rmsnorm
from qwen3_runtime.layers.rmsnorm import RMSNorm

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer rmsnorm needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed"),
]


def test_cuda_rmsnorm_matches_python_formula_hidden_and_per_head():
    torch.manual_seed(0)
    device = torch.device("cuda")
    ops = Ops.flashinfer()
    hidden = RMSNorm(2560, 1e-6, ops=ops).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(3, 2560, device=device, dtype=torch.bfloat16)
    got = hidden(x)
    want = torch_rmsnorm(x, hidden.weight, hidden.eps)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)

    qn = RMSNorm(128, 1e-6, ops=ops).to(device=device, dtype=torch.bfloat16)
    q = torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(qn(q), torch_rmsnorm(q, qn.weight, qn.eps), atol=1e-3, rtol=1e-3)
