"""Correctness for Triton paged decode vs PyTorch gather and FlashInfer."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from qwen3_runtime.attention.paged import paged_context
from qwen3_runtime.kv.paged import PagedBatch, PagedKVPool

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton paged decode needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton not installed"),
]


def _case(batch, n_q, n_kv, d, page, kv_lens, *, shuffled=False, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    max_len = max(kv_lens)
    n_pages = (max_len + page - 1) // page
    n_blocks = n_pages * batch + 4
    q = torch.randn(batch, n_q, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(n_blocks, page, n_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.zeros(batch, n_pages, device=device, dtype=torch.int32)
    cursor = 0
    for b, kv in enumerate(kv_lens):
        need = (kv + page - 1) // page
        ids = torch.arange(cursor, cursor + need, device=device, dtype=torch.int32)
        if shuffled:
            ids = ids[torch.randperm(need, device=device)]
        table[b, :need] = ids
        cursor += need
    lens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    return q, k, v, table, lens


def test_naive_and_splitk_match_pytorch_gather():
    from qwen3_runtime.kernels.triton_decode import (
        paged_decode_attention,
        paged_decode_attention_naive,
        paged_decode_attention_ref,
    )

    q, k, v, table, lens = _case(2, 8, 2, 128, 16, [17, 80], shuffled=True, seed=5)
    ref = paged_decode_attention_ref(q, k, v, table, lens, page_size=16)
    naive = paged_decode_attention_naive(q, k, v, table, lens, page_size=16)
    fast = paged_decode_attention(q, k, v, table, lens, page_size=16, num_splits=4, max_kv_len=80)
    torch.testing.assert_close(naive.float(), ref.float(), atol=1e-3, rtol=1e-3)
    # MMA decode is bf16; match the existing gather-oracle slack, not 1e-3 vs fp32.
    torch.testing.assert_close(fast.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_empty_splits_do_not_leak_stale_scratch():
    """Regression: splits past the end of a short sequence must not poison the merge.

    `mid_o` used to be allocated with `torch.empty`, and a split whose token
    range was empty never wrote its slot. `_combine_kernel` weights every slot
    by `exp(lse_s - M)`, which is 0 for an empty split -- but `0 * NaN` is NaN,
    so whatever the caching allocator left behind leaked into the output. The
    scratch is poisoned here so this fails loudly without the fix.
    """
    from qwen3_runtime.kernels.triton_decode import (
        _scratch,
        paged_decode_attention,
        paged_decode_attention_ref,
        reset_scratch,
    )

    batch, n_q, n_kv, d, page = 2, 8, 2, 128, 16
    # kv=3 over 8 splits leaves splits 3..7 empty; kv=40 fills all 8.
    kv_lens = [3, 40]
    num_splits = 8
    q, k, v, table, lens = _case(batch, n_q, n_kv, d, page, kv_lens, seed=13)

    reset_scratch()
    mid_o, mid_lse = _scratch(batch, n_q, num_splits, d, q.device)
    mid_o.fill_(float("nan"))
    mid_lse.fill_(float("nan"))
    try:
        got = paged_decode_attention(
            q, k, v, table, lens, page_size=page, num_splits=num_splits, max_kv_len=max(kv_lens)
        )
        assert torch.isfinite(got).all(), "stale scratch leaked into the output"
        ref = paged_decode_attention_ref(q, k, v, table, lens, page_size=page)
        torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
    finally:
        reset_scratch()


def test_zero_length_sequence_yields_zeros_not_nan():
    """kv_len==0 cannot be rejected host-side without a D2H sync, so the kernels absorb it."""
    from qwen3_runtime.kernels.triton_decode import (
        paged_decode_attention,
        paged_decode_attention_naive,
    )

    q, k, v, table, lens = _case(2, 8, 2, 128, 16, [1, 32], seed=17)
    lens[0] = 0
    for num_splits in (1, 4):
        got = paged_decode_attention(
            q, k, v, table, lens, page_size=16, num_splits=num_splits, max_kv_len=32
        )
        assert torch.isfinite(got).all()
        assert torch.all(got[0] == 0)
    naive = paged_decode_attention_naive(q, k, v, table, lens, page_size=16)
    assert torch.isfinite(naive).all()
    assert torch.all(naive[0] == 0)


def test_qwen3_shape_matches_pytorch_and_flashinfer():
    from qwen3_runtime.kernels.triton_decode import (
        paged_decode_attention,
        paged_decode_attention_ref,
    )

    q, k, v, table, lens = _case(4, 32, 8, 128, 16, [256, 257, 1024, 4096], seed=7)
    ref = paged_decode_attention_ref(q, k, v, table, lens, page_size=16)
    got = paged_decode_attention(q, k, v, table, lens, page_size=16, max_kv_len=4096)
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer not installed")
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    page = 16
    pieces, indptr, last = [], [0], []
    for b, kv in enumerate(lens.tolist()):
        n_pages = (int(kv) + page - 1) // page
        pieces.append(table[b, :n_pages].to(dtype=torch.int32))
        indptr.append(indptr[-1] + n_pages)
        rem = int(kv) % page
        last.append(page if rem == 0 else rem)
    wrapper.plan(
        torch.tensor(indptr, dtype=torch.int32, device="cuda"),
        torch.cat(pieces),
        torch.tensor(last, dtype=torch.int32, device="cuda"),
        32,
        8,
        128,
        page,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
    )
    fi = wrapper.run(q, (k, v))
    torch.testing.assert_close(got.float(), fi.float(), atol=2e-3, rtol=1e-3)


def test_qwen3_long_decode_allclose_flashinfer_1e3():
    """Success-criterion shape: GQA 32/8, head_dim=128, long KV."""
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer not installed")
    from qwen3_runtime.kernels.triton_decode import paged_decode_attention
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    q, k, v, table, lens = _case(1, 32, 8, 128, 16, [8192], seed=11)
    got = paged_decode_attention(q, k, v, table, lens, page_size=16, max_kv_len=8192)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    n_pages = 8192 // 16
    wrapper.plan(
        torch.tensor([0, n_pages], dtype=torch.int32, device="cuda"),
        table[0, :n_pages].to(dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        32,
        8,
        128,
        16,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
    )
    fi = wrapper.run(q, (k, v))
    torch.testing.assert_close(got.float(), fi.float(), atol=1e-3, rtol=1e-3)


def test_paged_context_triton_decode_matches_flashinfer():
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer not installed")
    torch.manual_seed(3)
    device = torch.device("cuda")
    n_heads, n_kv, head_dim, page, kv_len = 32, 8, 128, 16, 20
    q1 = torch.randn(1, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k1 = torch.randn(1, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    v1 = torch.randn_like(k1)
    prefix = kv_len - 1
    pk = torch.randn(prefix, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    pv = torch.randn_like(pk)
    pool_kw = dict(
        num_layers=1,
        num_blocks=8,
        block_size=page,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

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
        return paged_context(
            backend, q1.clone(), k1.clone(), v1.clone(), batch, 0, n_heads, n_kv, head_dim
        )

    torch.testing.assert_close(_decode("triton").float(), _decode("sdpa").float(), atol=1e-3, rtol=1e-3)
    fi = _decode("flashinfer")
    torch.testing.assert_close(_decode("triton").float(), fi.float(), atol=2e-2, rtol=2e-2)
