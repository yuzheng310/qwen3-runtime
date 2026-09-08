import importlib.util

import pytest
import torch

from qwen3_runtime.layers.ops import Ops, torch_silu_and_mul
from qwen3_runtime.models.qwen3 import Qwen3MLP, Qwen3ModelConfig

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer silu_and_mul needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed"),
]


def test_cuda_mlp_silu_and_mul_matches_chunk_formula():
    torch.manual_seed(0)
    device = torch.device("cuda")
    cfg = Qwen3ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
    )
    mlp = Qwen3MLP(cfg, ops=Ops.flashinfer()).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(3, 16, device=device, dtype=torch.bfloat16)
    gu = mlp.gate_up_proj(x)
    want = mlp.down_proj(torch_silu_and_mul(gu))
    got = mlp(x)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=1e-2)
