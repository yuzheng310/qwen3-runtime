import importlib.util

import pytest
import torch

from qwen3_runtime.layers.rope import RotaryEmbedding, apply_qwen3_rope, apply_rope_neox

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer RoPE needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed"),
]


def test_cuda_qwen3_rope_matches_neox_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    rope = RotaryEmbedding(128, 64, 1_000_000.0).to(device)
    pos = torch.arange(4, device=device)
    q = torch.randn(4, 8, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, 2, 128, device=device, dtype=torch.bfloat16)
    cos, sin = rope(pos)
    q_ref = apply_rope_neox(q, cos, sin)
    k_ref = apply_rope_neox(k, cos, sin)
    q_got, k_got = apply_qwen3_rope(q, k, pos, rope)
    torch.testing.assert_close(q_got, q_ref, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(k_got, k_ref, atol=5e-3, rtol=1e-2)
