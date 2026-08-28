"""Triton paged *decode* attention (q_len=1) over PagedKVPool NHD pages.

This file is written to be read. It carries three kernels that are the same
algorithm at three levels of parallelism, so the cost of each step is visible:

===============================================================================
1. The problem
===============================================================================

Decode attention computes, for one query vector ``q`` per (sequence, head)::

    out = sum_j softmax(q . k_j / sqrt(d))_j * v_j     for j in [0, kv_len)

No causal mask is needed. The query *is* the last position, so it attends to
every key that exists. (Prefill needs the lower-triangular mask; decode does
not. Applying one here would be wrong.)

K and V do not live in a contiguous ``[kv_len, ...]`` tensor. They live in the
paged pool as ``[num_blocks, page_size, n_kv_heads, head_dim]``, and the
sequence's logical order is recovered through ``block_table[b]``: logical token
``t`` sits at page ``block_table[b, t // page_size]``, offset ``t % page_size``.

===============================================================================
2. Online softmax: why we never materialise the score vector
===============================================================================

The scores are ``kv_len`` wide, which for a long context does not fit in
registers. FlashAttention's streaming form keeps three running values instead:

    m   running max of the scores seen so far
    l   running sum of exp(score - m)
    acc running sum of exp(score - m) * v

When a new tile arrives with max ``m_new > m``, every previously accumulated
term was scaled by ``exp(-m)`` and must be rescaled to ``exp(-m_new)``. The
correction factor is ``alpha = exp(m - m_new)``, applied to both ``l`` and
``acc`` before the new tile is added. This is exact, not an approximation.

On the first tile ``m = -inf`` gives ``alpha = exp(-inf) = 0``, which correctly
zeroes the (empty) accumulators.

===============================================================================
3. GQA: one K/V stream feeds `gqa` query heads
===============================================================================

Qwen3-4B has 32 query heads over 8 KV heads, so ``gqa = 4`` query heads share
one KV head. The naive kernel assigns one program per (sequence, query head)
and therefore reads the same K/V four times. The production kernel assigns one
program per (sequence, *KV* head) and holds all ``gqa`` queries as rows of a
matrix, so the K/V tile is loaded once and reused. Decode is memory-bound, so
this is close to a 4x reduction in the quantity that matters.

Holding the group as a matrix also turns the dot product into a real MMA:
``[gqa, head_dim] @ [head_dim, BLOCK_N]``. The tensor core's minimum M is 16
while ``gqa`` is 4, so 12 of 16 rows are padding. That waste is real but it is
paid in compute, which decode has to spare.

===============================================================================
4. Split-K: parallelism when the batch is small
===============================================================================

At batch=1 the grid above is only ``n_kv_heads = 8`` programs. A 4090 has 128
SMs, so 120 of them idle while 8 stream the whole KV cache serially. Split-K
cuts the KV range into ``num_splits`` chunks and gives each its own program,
raising the grid to ``batch * n_kv_heads * num_splits``.

Each split produces a *partial* result over its own token range. Merging them
uses the log-sum-exp identity. Split ``s`` writes

    O_s   = (sum_{j in s} e^{x_j} v_j) / l_s          -- normalised
    lse_s = m_s + log(l_s)                            -- so e^{lse_s} = sum_{j in s} e^{x_j}

and because the exponentials simply partition,

    out = sum_j e^{x_j} v_j / sum_j e^{x_j}
        = sum_s e^{lse_s} O_s / sum_s e^{lse_s}
        = sum_s e^{lse_s - M} O_s / sum_s e^{lse_s - M}   with M = max_s lse_s

The last line is what ``_combine_kernel`` evaluates; subtracting ``M`` keeps it
in range. A split whose token range is empty writes ``lse_s = -inf``, which
contributes exactly zero weight.

Split-K is not free: it costs a second kernel launch and a pass over the
partials. ``pick_num_splits`` therefore returns 1 for short KV, where that
overhead dominates the occupancy it would buy.

===============================================================================
5. Numerical expectations
===============================================================================

``_naive_kernel`` accumulates in fp32 throughout and matches the PyTorch gather
oracle to ~1e-3. ``_gqa_split_kernel`` feeds bf16 operands to the tensor cores
(the probabilities are cast to bf16 before the P@V product, as FlashAttention
does), so it matches the fp32 oracle to ~2e-2 and matches FlashInfer -- which
makes the same trade -- to ~1e-3. Do not expect bitwise agreement with either;
greedy token ids can flip on a 1-ulp difference and are not a valid gate.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Tuned in docs/kernel-dev/STAGE_3_TUNING.md on an RTX 4090 (page_size=16,
# head_dim=128, GQA 32/8). BLOCK_N=16 fragments the K/V loads; 128 raises
# register pressure without a matching win.
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3

# Tensor-core minimum M. Caps the GQA group this kernel can pack into one MMA.
_BLOCK_H = 16

# Triton computes `phys * stride` in int32 before widening to a pointer, so a
# page index times the per-block stride must stay representable.
_INT32_MAX = 2**31 - 1


def pick_num_splits(batch: int, n_kv_heads: int, kv_len: int, sm_count: int) -> int:
    """Flash-decoding split count.

    Short KV stays single-pass (combine launch would dominate). Longer KV
    aims for ~128 tokens/split, capped so the grid is about 2 CTAs per SM.
    """
    groups = max(1, batch * n_kv_heads)
    if kv_len <= 256:
        return 1
    splits = max(1, (kv_len + 127) // 128)
    max_splits = min(32, max(1, (sm_count * 2) // groups))
    return int(min(splits, max_splits))


def paged_decode_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int | None = None,
) -> torch.Tensor:
    """PyTorch gather + softmax. Oracle for Triton kernels."""
    if page_size is None:
        page_size = int(k_cache.shape[1])
    batch, n_q_heads, head_dim = q.shape
    n_kv = k_cache.shape[2]
    group = n_q_heads // n_kv
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    for b in range(batch):
        kv_len = int(kv_lens[b])
        n_pages = (kv_len + page_size - 1) // page_size
        keys = []
        vals = []
        for p in range(n_pages):
            bid = int(block_table[b, p])
            n = min(page_size, kv_len - p * page_size)
            keys.append(k_cache[bid, :n])
            vals.append(v_cache[bid, :n])
        k = torch.cat(keys, dim=0).repeat_interleave(group, dim=1)
        v = torch.cat(vals, dim=0).repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b].float(), k.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hn,nhd->hd", weights, v.float()).to(q.dtype)
    return out


@triton.jit
def _naive_kernel(
    # Tensors.
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    block_table_ptr,
    kv_lens_ptr,
    # Strides: q[batch, q_head, dim].
    stride_qb,
    stride_qh,
    stride_qd,
    # Strides: k/v[block, page_offset, kv_head, dim].
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    # Strides: out[batch, q_head, dim], block_table[batch, page].
    stride_ob,
    stride_oh,
    stride_od,
    stride_btb,
    stride_btn,
    # Scalars.
    n_q_heads,
    n_kv_heads,
    sm_scale,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Stage-1 reference: one program per (sequence, query head), one page per step.

    Correctness-first. It ignores GQA sharing (each of the `gqa` query heads
    re-reads the same K/V) and offers no split-K, so batch=1 leaves most SMs
    idle. Kept as the readable baseline that `_gqa_split_kernel` is measured
    against; see docs/kernel-dev/STAGE_1_NAIVE.md.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    gqa = n_q_heads // n_kv_heads
    kv_h = h // gqa
    kv_len = tl.load(kv_lens_ptr + b)
    n_pages = tl.cdiv(kv_len, PAGE_SIZE)

    # head_dim need not be a power of two; BLOCK_D is, and mask_d hides the tail.
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q = tl.load(
        q_ptr + b * stride_qb + h * stride_qh + offs_d * stride_qd,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for p in range(n_pages):
        phys = tl.load(block_table_ptr + b * stride_btb + p * stride_btn)
        tok_base = p * PAGE_SIZE
        offs_t = tl.arange(0, PAGE_SIZE)
        # Only the final page is partially filled.
        mask_t = tok_base + offs_t < kv_len
        k = tl.load(
            k_ptr
            + phys * stride_kb
            + offs_t[:, None] * stride_ks
            + kv_h * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        qk = tl.sum(q[None, :] * k, 1) * sm_scale
        qk = tl.where(mask_t, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)
        p_s = tl.exp(qk - m_new)
        p_s = tl.where(mask_t, p_s, 0.0)
        l_i = l_i * alpha + tl.sum(p_s, 0)
        acc = acc * alpha

        v = tl.load(
            v_ptr
            + phys * stride_vb
            + offs_t[:, None] * stride_vs
            + kv_h * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc + tl.sum(p_s[:, None] * v, 0)
        m_i = m_new

    # l_i == 0 only when kv_len == 0. That cannot be screened host-side without a
    # D2H sync on kv_lens, so absorb it here: emit zeros rather than NaN.
    denom = tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(
        out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od,
        (acc / denom).to(out_ptr.dtype.element_ty),
        mask=mask_d,
    )


@triton.jit
def _gqa_split_kernel(
    # Tensors.
    q_ptr,
    k_ptr,
    v_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    block_table_ptr,
    kv_lens_ptr,
    # Strides: q[batch, q_head, dim].
    stride_qb,
    stride_qh,
    stride_qd,
    # Strides: k/v[block, page_offset, kv_head, dim].
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    # Strides: mid_o[batch, q_head, split, dim], mid_lse[batch, q_head, split].
    stride_mb,
    stride_mh,
    stride_ms,
    stride_md,
    stride_lb,
    stride_lh,
    stride_ls,
    # Strides: out[batch, q_head, dim], block_table[batch, page].
    stride_ob,
    stride_oh,
    stride_od,
    stride_btb,
    stride_btn,
    # Scalars.
    n_q_heads,
    n_kv_heads,
    num_splits,
    sm_scale,
    PAGE_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WRITE_DIRECT: tl.constexpr,
):
    """Production kernel: one program per (sequence, KV head, split).

    The whole GQA group is held as rows of one matrix so the K/V tile is loaded
    once for all of them, and the score computation becomes a tensor-core MMA.

    ``WRITE_DIRECT`` is set when ``num_splits == 1``: the program owns the whole
    KV range, so it normalises and writes ``out`` itself and no combine pass is
    launched. Otherwise it writes a normalised partial plus its log-sum-exp for
    ``_combine_kernel`` to merge.
    """
    b = tl.program_id(0)
    kv_h = tl.program_id(1)
    split_id = tl.program_id(2)
    gqa = n_q_heads // n_kv_heads
    kv_len = tl.load(kv_lens_ptr + b)

    # Split boundaries are per-sequence: a short sequence in a ragged batch
    # simply leaves its high splits empty rather than reading out of range.
    tok_per_split = tl.cdiv(kv_len, num_splits)
    split_start = split_id * tok_per_split
    split_end = tl.minimum(split_start + tok_per_split, kv_len)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    q_h = kv_h * gqa + offs_h
    mask_h = offs_h < gqa
    mask_d = offs_d < HEAD_DIM

    q = tl.load(
        q_ptr + b * stride_qb + q_h[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    )
    q_k = q.to(k_ptr.dtype.element_ty)

    # Rows [gqa, BLOCK_H) are MMA padding. Seeding their running max at 0
    # instead of -inf keeps `exp(e_max - n_e_max)` finite for them: with -inf on
    # both sides it evaluates exp(nan) and poisons the whole padded row. The
    # stores below are masked either way, but zeros are far easier to debug than
    # NaN when reading a register dump.
    neg_inf = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_max = tl.where(mask_h, neg_inf, 0.0)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    # Zero-trip when this split is past the end of a short sequence.
    for start_n in tl.range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_end
        # Split boundaries are not page-aligned, so each lane resolves its own
        # page. Masked lanes read block 0, which is in bounds and discarded.
        page_idx = offs_n // PAGE_SIZE
        page_off = offs_n % PAGE_SIZE
        phys = tl.load(
            block_table_ptr + b * stride_btb + page_idx * stride_btn,
            mask=mask_n,
            other=0,
        )
        k = tl.load(
            k_ptr
            + phys[None, :] * stride_kb
            + page_off[None, :] * stride_ks
            + kv_h * stride_kh
            + offs_d[:, None] * stride_kd,
            mask=mask_n[None, :] & mask_d[:, None],
            other=0.0,
        )
        qk = tl.dot(q_k, k) * sm_scale
        qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)
        acc = acc * re_scale[:, None]
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

        v = tl.load(
            v_ptr
            + phys[:, None] * stride_vb
            + page_off[:, None] * stride_vs
            + kv_h * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        # bf16 P matches FlashAttention/FlashInfer; see the numerics note above.
        acc += tl.dot(p.to(v.dtype), v)

    # e_sum is 0 for an empty split and for the padded rows. Dividing by 1 there
    # emits zeros; the accompanying lse of -inf is what actually discards them.
    denom = tl.where(e_sum > 0.0, e_sum, 1.0)
    if WRITE_DIRECT:
        tl.store(
            out_ptr + b * stride_ob + q_h[:, None] * stride_oh + offs_d[None, :] * stride_od,
            (acc / denom[:, None]).to(out_ptr.dtype.element_ty),
            mask=mask_h[:, None] & mask_d[None, :],
        )
    else:
        tl.store(
            mid_o_ptr
            + b * stride_mb
            + q_h[:, None] * stride_mh
            + split_id * stride_ms
            + offs_d[None, :] * stride_md,
            acc / denom[:, None],
            mask=mask_h[:, None] & mask_d[None, :],
        )
        lse = tl.where(e_sum > 0.0, e_max + tl.log(denom), float("-inf"))
        tl.store(
            mid_lse_ptr + b * stride_lb + q_h * stride_lh + split_id * stride_ls,
            lse,
            mask=mask_h,
        )


@triton.jit
def _combine_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    num_splits,
    stride_mb,
    stride_mh,
    stride_ms,
    stride_md,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Merge per-split partials: ``out = sum_s e^{lse_s - M} O_s / sum_s e^{lse_s - M}``.

    One program per (sequence, query head). Derivation is in the module
    docstring, section 4. Splits that saw no tokens carry ``lse_s = -inf`` and
    drop out of both sums.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(num_splits):
        lse = tl.load(mid_lse_ptr + b * stride_lb + h * stride_lh + s * stride_ls)
        partial = tl.load(
            mid_o_ptr + b * stride_mb + h * stride_mh + s * stride_ms + offs_d * stride_md,
            mask=mask_d,
            other=0.0,
        )
        n_e_max = tl.maximum(lse, e_max)
        # Both guards below cover `-inf - -inf`, which is nan rather than 0:
        # the first fires on the initial iteration, the second on empty splits.
        old_scale = tl.where(e_max == float("-inf"), 0.0, tl.exp(e_max - n_e_max))
        weight = tl.where(lse == float("-inf"), 0.0, tl.exp(lse - n_e_max))
        acc = acc * old_scale + weight * partial
        e_sum = e_sum * old_scale + weight
        e_max = n_e_max

    denom = tl.where(e_sum > 0.0, e_sum, 1.0)
    tl.store(
        out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od,
        (acc / denom).to(out_ptr.dtype.element_ty),
        mask=mask_d,
    )


# Split-K scratch, keyed by shape. Entries are deliberately never evicted: a
# captured CUDA Graph bakes in the device pointer it saw at capture time, so
# freeing or reallocating a buffer would leave the graph replaying against dead
# memory. The key space is bounded in practice by (batch sizes seen) x (values
# `pick_num_splits` returns), and num_splits==1 takes the WRITE_DIRECT path
# without allocating at all. Call `reset_scratch()` between unrelated test
# configurations, never while a graph that used it is still live.
_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def reset_scratch() -> None:
    """Drop cached split-K buffers. Unsafe while a capturing graph references them."""
    _SCRATCH.clear()


def _scratch(batch: int, n_heads: int, n_splits: int, head_dim: int, device: torch.device):
    key = (batch, n_heads, n_splits, head_dim, str(device))
    hit = _SCRATCH.get(key)
    if hit is None:
        # mid_o is zeroed, not `empty`. `_combine_kernel` multiplies every slot
        # by its weight, and an empty split's weight of 0 does not tame a NaN
        # left over in reused allocator memory: 0 * NaN is NaN. The split kernel
        # writes every slot including the empty ones, so this is defence in
        # depth against a future change re-introducing a skipped store. It costs
        # one memset per distinct shape, never per call. mid_lse needs no such
        # guard because its store is outside the WRITE_DIRECT branch and
        # therefore unconditional.
        mid_o = torch.zeros(batch, n_heads, n_splits, head_dim, device=device, dtype=torch.float32)
        mid_lse = torch.empty(batch, n_heads, n_splits, device=device, dtype=torch.float32)
        _SCRATCH[key] = (mid_o, mid_lse)
        hit = _SCRATCH[key]
    return hit


def _check_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    page_size: int,
) -> tuple[int, int, int, int]:
    """Validate shapes on the host only. Nothing here may read device values.

    A `.item()` or `.min()` on `kv_lens` would be a D2H sync on the per-layer
    decode path, which measured more expensive than the kernel itself
    (docs/kernel-dev/STAGE_3_TUNING.md, Bug 3.1).
    """
    if q.dim() != 3:
        raise ValueError("q must be [batch, n_q_heads, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K/V cache shapes must match")
    batch, n_q_heads, head_dim = q.shape
    n_kv_heads = int(k_cache.shape[2])
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be a multiple of n_kv_heads")
    if int(k_cache.shape[1]) != page_size:
        raise ValueError("page_size must match cache dim 1")
    if kv_lens.numel() != batch:
        raise ValueError("kv_lens must be [batch]")
    if block_table.shape[0] != batch:
        raise ValueError("block_table batch dim must match q")
    max_page_offset = (int(k_cache.shape[0]) - 1) * int(k_cache.stride(0))
    if max_page_offset > _INT32_MAX:
        raise ValueError(
            f"page pool too large for int32 addressing: {max_page_offset} > {_INT32_MAX}"
        )
    return batch, n_q_heads, n_kv_heads, head_dim


def paged_decode_attention_naive(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int | None = None,
) -> torch.Tensor:
    """Stage-1 kernel: grid (batch, q_head), page loop, no split-K."""
    if page_size is None:
        page_size = int(k_cache.shape[1])
    batch, n_q_heads, n_kv_heads, head_dim = _check_inputs(
        q, k_cache, v_cache, block_table, kv_lens, page_size
    )
    q = q.contiguous()
    table = block_table.contiguous().to(dtype=torch.int32)
    lens = kv_lens.contiguous().to(dtype=torch.int32)
    out = torch.empty_like(q)
    _naive_kernel[(batch, n_q_heads)](
        q,
        k_cache,
        v_cache,
        out,
        table,
        lens,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *out.stride(),
        *table.stride(),
        n_q_heads,
        n_kv_heads,
        1.0 / math.sqrt(head_dim),
        PAGE_SIZE=page_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        HEAD_DIM=head_dim,
        num_warps=_NUM_WARPS,
        num_stages=2,
    )
    return out


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int | None = None,
    num_splits: int | None = None,
    max_kv_len: int | None = None,
) -> torch.Tensor:
    """GQA + tiled MMA + optional split-K. Default serving kernel.

    Pass ``max_kv_len`` (a host-side int) whenever the caller already knows it.
    Leaving it ``None`` forces a ``kv_lens.max().item()``, i.e. a device sync on
    the decode path. Pass ``num_splits`` to freeze the grid for CUDA Graph
    capture, where it must not vary between replays.
    """
    if page_size is None:
        page_size = int(k_cache.shape[1])
    batch, n_q_heads, n_kv_heads, head_dim = _check_inputs(
        q, k_cache, v_cache, block_table, kv_lens, page_size
    )
    gqa = n_q_heads // n_kv_heads
    if gqa > _BLOCK_H:
        raise ValueError(f"GQA group larger than BLOCK_H={_BLOCK_H} is not implemented")

    q = q if q.is_contiguous() else q.contiguous()
    table = block_table if block_table.is_contiguous() else block_table.contiguous()
    if table.dtype != torch.int32:
        table = table.to(dtype=torch.int32)
    lens = kv_lens if kv_lens.is_contiguous() else kv_lens.contiguous()
    if lens.dtype != torch.int32:
        lens = lens.to(dtype=torch.int32)

    if num_splits is None:
        if max_kv_len is None:
            max_kv_len = int(lens.max().item())
        sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = pick_num_splits(batch, n_kv_heads, max_kv_len, sm)
    num_splits = max(1, int(num_splits))

    out = torch.empty_like(q)
    write_direct = num_splits == 1
    if write_direct:
        # The partial-output stores are compiled out under WRITE_DIRECT, so the
        # pointers are never dereferenced. `out` stands in for them rather than
        # an input tensor, so that a future regression corrupts the output
        # instead of silently rewriting q.
        mid_o = mid_lse = out
        mid_strides = (0, 0, 0, 0)
        lse_strides = (0, 0, 0)
    else:
        mid_o, mid_lse = _scratch(batch, n_q_heads, num_splits, head_dim, q.device)
        mid_strides = mid_o.stride()
        lse_strides = mid_lse.stride()

    block_d = triton.next_power_of_2(head_dim)
    _gqa_split_kernel[(batch, n_kv_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        mid_o,
        mid_lse,
        out,
        table,
        lens,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *mid_strides,
        *lse_strides,
        *out.stride(),
        *table.stride(),
        n_q_heads,
        n_kv_heads,
        num_splits,
        1.0 / math.sqrt(head_dim),
        PAGE_SIZE=page_size,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        HEAD_DIM=head_dim,
        WRITE_DIRECT=write_direct,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if not write_direct:
        _combine_kernel[(batch, n_q_heads)](
            mid_o,
            mid_lse,
            out,
            num_splits,
            *mid_strides,
            *lse_strides,
            *out.stride(),
            BLOCK_D=block_d,
            HEAD_DIM=head_dim,
            num_warps=_NUM_WARPS,
        )
    return out
