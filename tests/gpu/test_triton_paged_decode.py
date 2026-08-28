import importlib.util

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton paged decode needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton not installed"),
]


def test_triton_paged_decode_matches_pytorch_gather():
    from qwen3_runtime.kernels.triton_paged_decode import (
        paged_decode_attention,
        paged_decode_attention_ref,
    )

    torch.manual_seed(5)
    device = torch.device("cuda")
    n_q, n_kv, d, page, kv_len = 8, 2, 128, 16, 80
    n_pages = (kv_len + page - 1) // page
    q = torch.randn(n_q, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(n_pages + 1, page, n_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(n_pages, device=device, dtype=torch.int32)
    got = paged_decode_attention(q, k, v, table, kv_len, page_size=page)
    ref = paged_decode_attention_ref(q, k, v, table, kv_len, page_size=page)
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
