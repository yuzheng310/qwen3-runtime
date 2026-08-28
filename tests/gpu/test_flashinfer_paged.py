import importlib.util

import pytest
import torch

from qwen3_runtime.attention.paged import paged_context
from qwen3_runtime.kv.paged import PagedBatch, PagedKVPool

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer paged kernels need CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed"),
]


def test_flashinfer_paged_matches_sdpa_prefill_and_decode():
    """head_dim=128 matches Qwen3; tiny head_dim is not a FlashInfer kernel shape."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    q_len, n_heads, n_kv, head_dim = 4, 4, 2, 128
    q = torch.randn(q_len, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(q_len, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(q_len, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    pool_kw = dict(
        num_layers=1,
        num_blocks=8,
        block_size=16,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    def _run(backend: str) -> torch.Tensor:
        pool = PagedKVPool(**pool_kw)
        batch = PagedBatch(
            pool=pool,
            slot_mapping=torch.arange(q_len, device=device),
            block_tables=[torch.tensor([0], device=device, dtype=torch.long)],
            kv_lens=[q_len],
            cu_seqlens=[0, q_len],
        )
        return paged_context(backend, q.clone(), k.clone(), v.clone(), batch, 0, n_heads, n_kv, head_dim)

    sdpa = _run("sdpa")
    fi = _run("flashinfer")
    torch.testing.assert_close(fi.float(), sdpa.float(), atol=2e-2, rtol=2e-2)

    q1 = torch.randn(1, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k1 = torch.randn(1, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    v1 = torch.randn(1, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    kv_len = 20
    prefix = kv_len - 1
    # Same prefix for both backends: per-call torch.randn would compare unrelated caches.
    pk = torch.randn(prefix, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    pv = torch.randn(prefix, n_kv, head_dim, device=device, dtype=torch.bfloat16)

    def _decode(backend: str) -> torch.Tensor:
        pool = PagedKVPool(**pool_kw)
        pool.store(0, pk, pv, torch.arange(prefix, device=device))
        batch = PagedBatch(
            pool=pool,
            slot_mapping=torch.tensor([prefix], device=device),
            block_tables=[torch.tensor([0, 1], device=device, dtype=torch.long)],
            kv_lens=[kv_len],
            cu_seqlens=[0, 1],
        )
        return paged_context(backend, q1.clone(), k1.clone(), v1.clone(), batch, 0, n_heads, n_kv, head_dim)

    torch.testing.assert_close(_decode("flashinfer").float(), _decode("sdpa").float(), atol=2e-2, rtol=2e-2)
