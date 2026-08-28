"""Triton paged *decode* attention (q_len=1) over the NHD block pool.

Not the serving default. Phase 3 showed isolated 10k prefill is library GEMM
(64%) with FlashInfer paged prefill at 30%. This kernel is the honest
comparison against FlashInfer on the long-context *decode* shape, where CUDA
graphs are already bypassed (KV > chunk cap). It is not a faster GEMV.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    block_table_ptr,
    kv_len,
    n_q_heads,
    n_kv_heads,
    page_size,
    n_pages,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_vd,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    h = tl.program_id(0)
    gqa = n_q_heads // n_kv_heads
    kv_h = h // gqa
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q = tl.load(q_ptr + h * stride_qh + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
    sm_scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))
    m_i = tl.full([], float("-inf"), tl.float32)
    l_i = tl.zeros([], tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for p in range(0, n_pages):
        bid = tl.load(block_table_ptr + p)
        tok_base = p * page_size
        for t in range(0, page_size):
            active = tok_base + t < kv_len
            k = tl.load(
                k_ptr + bid * stride_kb + t * stride_kp + kv_h * stride_kh + offs_d * stride_kd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            qk = tl.sum(q * k, axis=0) * sm_scale
            qk = tl.where(active, qk, float("-inf"))
            m_new = tl.maximum(m_i, qk)
            alpha = tl.exp(m_i - m_new)
            p_s = tl.exp(qk - m_new)
            p_s = tl.where(active, p_s, 0.0)
            l_i = l_i * alpha + p_s
            acc = acc * alpha
            v = tl.load(
                v_ptr + bid * stride_vb + t * stride_vp + kv_h * stride_vh + offs_d * stride_vd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            acc = acc + p_s * v
            m_i = m_new
    out = acc / l_i
    tl.store(out_ptr + h * stride_oh + offs_d * stride_od, out.to(tl.bfloat16), mask=mask_d)


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_len: int,
    *,
    page_size: int | None = None,
) -> torch.Tensor:
    """q: [Hq, D] bf16. caches: [blocks, page, Hkv, D] bf16. table: [n_pages] int32."""
    if q.dim() != 2:
        raise ValueError("q must be [n_q_heads, head_dim]")
    n_q_heads, head_dim = q.shape
    if k_cache.shape != v_cache.shape:
        raise ValueError("K/V cache shapes must match")
    if page_size is None:
        page_size = int(k_cache.shape[1])
    n_kv_heads = int(k_cache.shape[2])
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("Hq must be a multiple of Hkv")
    if kv_len < 1:
        raise ValueError("kv_len must be positive")
    n_pages = (int(kv_len) + page_size - 1) // page_size
    table = block_table[:n_pages].contiguous().to(dtype=torch.int32, device=q.device)
    if table.numel() < n_pages:
        raise ValueError("block_table shorter than occupied pages")
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    _paged_decode_kernel[(n_q_heads,)](
        q.contiguous(),
        k_cache,
        v_cache,
        out,
        table,
        int(kv_len),
        n_q_heads,
        n_kv_heads,
        page_size,
        n_pages,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0),
        out.stride(1),
        BLOCK_D=block_d,
        HEAD_DIM=head_dim,
    )
    return out


def paged_decode_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_len: int,
    *,
    page_size: int | None = None,
) -> torch.Tensor:
    """PyTorch gather + softmax. Numerical oracle for the Triton kernel."""
    if page_size is None:
        page_size = int(k_cache.shape[1])
    n_q_heads, head_dim = q.shape
    n_kv = k_cache.shape[2]
    group = n_q_heads // n_kv
    n_pages = (int(kv_len) + page_size - 1) // page_size
    keys = []
    vals = []
    for p in range(n_pages):
        bid = int(block_table[p])
        n = min(page_size, int(kv_len) - p * page_size)
        keys.append(k_cache[bid, :n])
        vals.append(v_cache[bid, :n])
    k = torch.cat(keys, dim=0)
    v = torch.cat(vals, dim=0)
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("hd,nhd->hn", q.float(), k.float()) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hn,nhd->hd", weights, v.float()).to(q.dtype)
