"""Paged attention over a physical block pool.

`pytorch` is the CPU/reference path (explicit scores). `sdpa` uses
`scaled_dot_product_attention` on gathered K/V — same pool layout, not a
different cache format. `flash_attn` is gather + FA (not a paged kernel).
`flashinfer` reads the NHD page tensors in place (no gather).
`triton` is decode-only (q_len=1); prefill falls back to flashinfer or sdpa.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from qwen3_runtime.kv.paged import PagedBatch

# FlashInfer documented default workspace (not a bench-tuned constant).
_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
_FI_STATE: dict[str, Any] = {}


def paged_context(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if backend == "pytorch":
        return _paged_explicit(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    if backend == "sdpa":
        return _paged_sdpa(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    if backend == "flash_attn":
        return _paged_flash(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    if backend == "flashinfer":
        return _paged_flashinfer(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    if backend == "triton":
        return _paged_triton(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    raise ValueError(f"unknown attention backend {backend!r}")


def flashinfer_page_tensors(
    block_tables: list,
    kv_lens: list[int],
    page_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build FlashInfer paged-KV indptr / indices / last_page_len on **CPU**.

    FlashInfer ``plan()`` copies these into its GPU buffers, then reads them
    back with ``tensor.to("cpu")``. Passing GPU tensors makes that a D2H sync
    every decode step (~0.2 ms bs=1, ~0.9 ms bs=16). ``device`` is accepted for
    call-site compatibility and ignored.
    """
    del device
    pieces: list[torch.Tensor] = []
    indptr = [0]
    last_page_len: list[int] = []
    for table, kv_len in zip(block_tables, kv_lens):
        n_pages = (int(kv_len) + page_size - 1) // page_size
        if n_pages <= 0:
            raise ValueError("FlashInfer page table requires kv_len > 0")
        if isinstance(table, torch.Tensor):
            pages = table[:n_pages].detach().to(dtype=torch.int32, device="cpu")
        else:
            pages = torch.tensor(table[:n_pages], dtype=torch.int32)
        if pages.numel() < n_pages:
            raise ValueError("block table shorter than occupied pages")
        pieces.append(pages)
        indptr.append(indptr[-1] + n_pages)
        rem = int(kv_len) % page_size
        last_page_len.append(page_size if rem == 0 else rem)
    indices = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, dtype=torch.int32)
    return (
        torch.tensor(indptr, dtype=torch.int32),
        indices,
        torch.tensor(last_page_len, dtype=torch.int32),
    )


def _causal_allowed(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)


def _paged_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    paged.pool.store(layer_id, k, v, paged.slot_mapping)
    group = num_attention_heads // num_key_value_heads
    scale = 1.0 / math.sqrt(head_dim)
    chunks: list[torch.Tensor] = []
    for i, (qs, qe) in enumerate(zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])):
        q_len = qe - qs
        kv_len = paged.kv_lens[i]
        qi = q[qs:qe]
        kk, vv = paged.pool.gather(layer_id, paged.block_tables[i], kv_len)
        kk = kk.repeat_interleave(group, dim=1)
        vv = vv.repeat_interleave(group, dim=1)
        allowed = _causal_allowed(q_len, kv_len, q.device)
        qh, kh, vh = qi.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        ctx = torch.matmul(attn, vh).transpose(0, 1).contiguous().view(q_len, -1)
        chunks.append(ctx)
    return torch.cat(chunks, dim=0)


def _paged_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    paged.pool.store(layer_id, k, v, paged.slot_mapping)
    group = num_attention_heads // num_key_value_heads
    scale = 1.0 / math.sqrt(head_dim)
    chunks: list[torch.Tensor] = []
    for i, (qs, qe) in enumerate(zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])):
        q_len = qe - qs
        kv_len = paged.kv_lens[i]
        qi = q[qs:qe]
        kk, vv = paged.pool.gather(layer_id, paged.block_tables[i], kv_len)
        kk = kk.repeat_interleave(group, dim=1)
        vv = vv.repeat_interleave(group, dim=1)
        allowed = _causal_allowed(q_len, kv_len, q.device)
        # SDPA bool mask: True means the position participates (inverse of MHA).
        out = F.scaled_dot_product_attention(
            qi.transpose(0, 1).unsqueeze(0),
            kk.transpose(0, 1).unsqueeze(0),
            vv.transpose(0, 1).unsqueeze(0),
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        chunks.append(out.squeeze(0).transpose(0, 1).contiguous().view(q_len, -1))
    return torch.cat(chunks, dim=0)


def _paged_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """FlashAttention on *gathered* K/V. Not a paged FA kernel; layout is unchanged."""
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise NotImplementedError(
            "flash_attn is not installed; GPU factory falls back to sdpa"
        ) from exc
    paged.pool.store(layer_id, k, v, paged.slot_mapping)
    scale = 1.0 / math.sqrt(head_dim)
    chunks: list[torch.Tensor] = []
    for i, (qs, qe) in enumerate(zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])):
        q_len = qe - qs
        kv_len = paged.kv_lens[i]
        qi = q[qs:qe]
        kk, vv = paged.pool.gather(layer_id, paged.block_tables[i], kv_len)
        # FA GQA: native kv heads. causal=True with S_q < S_k treats Q as the suffix.
        out = flash_attn_func(
            qi.unsqueeze(0),
            kk.unsqueeze(0),
            vv.unsqueeze(0),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
        )
        chunks.append(out.squeeze(0).contiguous().view(q_len, -1))
    return torch.cat(chunks, dim=0)


def make_flashinfer_graph_decode_wrapper(
    device: torch.device,
    batch_size: int,
    max_pages: int,
):
    """Decode wrapper with frozen batch size and persistent page-table buffers.

    `plan()` must run *outside* a CUDA Graph (FlashInfer contract). `run()` is what
    the graph captures.

    `disable_split_kv` is **not** a capture requirement, contrary to what this
    file used to claim. FlashInfer's decode kernel is persistent: its grid is
    fixed at compile time and `plan()` writes a work queue at fixed workspace
    offsets, so split-KV survives capture. Upstream documents the flag as a
    determinism switch defaulting to False, and separately documents
    `fixed_split_size` as the one that *is* graph-incompatible. See
    `use_decode_cuda_graph` for what this codebase currently passes and why that
    choice is still unmeasured.
    """
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    workspace = torch.empty(_FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    indices = torch.zeros(max_pages, dtype=torch.int32, device=device)
    last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    return BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr,
        paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last_page_len,
    )


def plan_flashinfer_decode(
    wrapper,
    paged: PagedBatch,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    *,
    disable_split_kv: bool = False,
) -> None:
    device = paged.pool.cache.device
    kv_indptr, kv_indices, last_page_len = flashinfer_page_tensors(
        paged.block_tables, paged.kv_lens, paged.pool.block_size, device
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        paged.pool.block_size,
        pos_encoding_mode="NONE",
        q_data_type=q_dtype,
        disable_split_kv=disable_split_kv,
    )


def _fi_wrappers(device: torch.device):
    key = str(device)
    cached = _FI_STATE.get(key)
    if cached is not None:
        return cached
    try:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
    except ImportError as exc:
        raise NotImplementedError(
            "flashinfer is not installed; factory falls back to sdpa"
        ) from exc
    workspace = torch.empty(_FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    # NHD matches PagedKVPool: [block, page_size, kv_head, dim].
    # use_tensor_cores=True is FlashInfer's GQA recommendation (Qwen3 is 32/8 GQA).
    cached = {
        "prefill": BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD"),
        "decode": BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=True
        ),
    }
    _FI_STATE[key] = cached
    return cached


def _store_flashinfer_kv(paged: PagedBatch, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
    """Write new K/V into the paged cache.

    CUDA-graph decode uses FlashInfer ``append_paged_kv_cache`` (one kernel for
    K+V, captured). Eager / SDPA keep ``PagedKVPool.store`` index_put so we do
    not rebuild page metadata on the hot eager path.
    """
    w = paged.flashinfer_wrapper
    if w is None or paged.append_batch_indices is None or paged.append_positions is None:
        paged.pool.store(layer_id, k, v, paged.slot_mapping)
        return
    from flashinfer import append_paged_kv_cache

    append_paged_kv_cache(
        k,
        v,
        paged.append_batch_indices,
        paged.append_positions,
        (paged.pool.cache[0, layer_id], paged.pool.cache[1, layer_id]),
        w._paged_kv_indices_buf,
        w._paged_kv_indptr_buf,
        w._paged_kv_last_page_len_buf,
        kv_layout="NHD",
    )


def _paged_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Paged FlashInfer. Reads pool pages in place; does not gather K/V."""
    wrappers = _fi_wrappers(q.device)
    decode_w = paged.flashinfer_wrapper or wrappers["decode"]
    _store_flashinfer_kv(paged, layer_id, k, v)
    page_size = paged.pool.block_size
    k_cache = paged.pool.cache[0, layer_id]
    v_cache = paged.pool.cache[1, layer_id]
    q_lens = [qe - qs for qs, qe in zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])]
    decode_only = bool(q_lens) and all(ql == 1 for ql in q_lens)

    if paged.flashinfer_mode is None:
        if decode_only:
            plan_flashinfer_decode(
                decode_w,
                paged,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                q.dtype,
                disable_split_kv=paged.flashinfer_wrapper is not None,
            )
            paged.flashinfer_mode = "decode"
        else:
            kv_indptr, kv_indices, last_page_len = flashinfer_page_tensors(
                paged.block_tables, paged.kv_lens, page_size, q.device
            )
            qo_indptr = torch.tensor(paged.cu_seqlens, dtype=torch.int32, device=q.device)
            wrappers["prefill"].plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                page_size,
                causal=True,
                pos_encoding_mode="NONE",
                q_data_type=q.dtype,
            )
            paged.flashinfer_mode = "prefill"

    if paged.flashinfer_mode == "decode":
        out = decode_w.run(q, (k_cache, v_cache))
    else:
        out = wrappers["prefill"].run(q, (k_cache, v_cache))
    return out.contiguous().view(q.shape[0], num_attention_heads * head_dim)


def _ensure_triton_pages(paged: PagedBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if paged.triton_block_table is not None and paged.triton_kv_lens is not None:
        return paged.triton_block_table, paged.triton_kv_lens
    page_size = paged.pool.block_size
    n_pages = [(int(kv) + page_size - 1) // page_size for kv in paged.kv_lens]
    max_p = max(n_pages) if n_pages else 0
    table = torch.zeros(len(paged.kv_lens), max_p, dtype=torch.int32, device=device)
    for i, bt in enumerate(paged.block_tables):
        need = n_pages[i]
        if isinstance(bt, torch.Tensor):
            pages = bt[:need].to(device=device, dtype=torch.int32)
        else:
            pages = torch.tensor(bt[:need], dtype=torch.int32, device=device)
        if pages.numel() < need:
            raise ValueError("block table shorter than occupied pages")
        table[i, :need] = pages
    kv = torch.tensor(paged.kv_lens, dtype=torch.int32, device=device)
    paged.triton_block_table = table
    paged.triton_kv_lens = kv
    return table, kv


def _paged_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Decode uses the Triton paged kernel. Prefill is not implemented here."""
    q_lens = [qe - qs for qs, qe in zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])]
    decode_only = bool(q_lens) and all(ql == 1 for ql in q_lens)
    if not decode_only:
        try:
            return _paged_flashinfer(
                q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
            )
        except NotImplementedError:
            return _paged_sdpa(
                q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
            )
    _store_flashinfer_kv(paged, layer_id, k, v)
    from qwen3_runtime.kernels.triton_decode import paged_decode_attention

    table, kv_lens = _ensure_triton_pages(paged, q.device)
    out = paged_decode_attention(
        q,
        paged.pool.cache[0, layer_id],
        paged.pool.cache[1, layer_id],
        table,
        kv_lens,
        page_size=paged.pool.block_size,
        max_kv_len=max(paged.kv_lens),
        num_splits=paged.triton_num_splits,
    )
    return out.contiguous().view(q.shape[0], num_attention_heads * head_dim)
