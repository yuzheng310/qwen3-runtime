"""FlashInfer paged attention: the shipping decode/prefill kernels."""

from __future__ import annotations

import torch

from qwen3_runtime.attention.state import AttentionState
from qwen3_runtime.kv.paged import PagedBatch

# FlashInfer documented default workspace (not a bench-tuned constant).
FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferRuntime:
    """One workspace + wrapper pair per device. Not a hidden module global."""

    def __init__(self) -> None:
        self._wrappers: dict[str, dict] = {}

    def reset(self) -> None:
        self._wrappers.clear()

    def wrappers(self, device: torch.device) -> dict:
        key = str(device)
        cached = self._wrappers.get(key)
        if cached is not None:
            return cached
        try:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
        except ImportError as exc:
            raise NotImplementedError(
                "flashinfer is not installed; factory falls back to pytorch"
            ) from exc
        workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        # NHD matches PagedKVPool: [block, page_size, kv_head, dim].
        cached = {
            "prefill": BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD"),
            "decode": BatchDecodeWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", use_tensor_cores=True
            ),
        }
        self._wrappers[key] = cached
        return cached


FLASHINFER = FlashInferRuntime()


def flashinfer_page_tensors(
    block_tables: list,
    kv_lens: list[int],
    page_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build FlashInfer paged-KV indptr / indices / last_page_len on **CPU**.

    FlashInfer ``plan()`` copies these into its GPU buffers, then reads them
    back with ``tensor.to("cpu")``. Passing GPU tensors makes that a D2H sync
    every decode step (~0.2 ms bs=1, ~0.9 ms bs=16). ``device`` is accepted for
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


def make_flashinfer_graph_decode_wrapper(
    device: torch.device,
    batch_size: int,
    max_pages: int,
):
    """Decode wrapper with frozen batch size and persistent page-table buffers.

    ``plan()`` must run *outside* a CUDA Graph (FlashInfer contract). ``run()``
    is what the graph captures.
    """
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    indices = torch.zeros(max_pages, dtype=torch.int32, device=device)
    last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr,
        paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last_page_len,
    )
    return wrapper, indptr, indices, last_page_len


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


def store_flashinfer_kv(
    paged: PagedBatch,
    layer_id: int,
    k: torch.Tensor,
    v: torch.Tensor,
    state: AttentionState,
) -> None:
    """Write new K/V into the paged cache.

    CUDA-graph decode uses FlashInfer ``append_paged_kv_cache`` (one kernel for
    K+V, captured). Eager keeps ``PagedKVPool.store`` so we do not rebuild page
    metadata on the hot eager path.
    """
    if (
        state.flashinfer_wrapper is None
        or state.append_batch_indices is None
        or state.append_positions is None
        or state.kv_indices_buf is None
        or state.kv_indptr_buf is None
        or state.kv_last_page_len_buf is None
    ):
        paged.pool.store(layer_id, k, v, paged.slot_mapping)
        return
    from flashinfer import append_paged_kv_cache

    append_paged_kv_cache(
        k,
        v,
        state.append_batch_indices,
        state.append_positions,
        (paged.pool.cache[0, layer_id], paged.pool.cache[1, layer_id]),
        state.kv_indices_buf,
        state.kv_indptr_buf,
        state.kv_last_page_len_buf,
        kv_layout="NHD",
    )


def paged_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    state: AttentionState,
) -> torch.Tensor:
    """Paged FlashInfer. Reads pool pages in place; does not gather K/V."""
    wrappers = FLASHINFER.wrappers(q.device)
    decode_w = state.flashinfer_wrapper or wrappers["decode"]
    store_flashinfer_kv(paged, layer_id, k, v, state)
    page_size = paged.pool.block_size
    k_cache = paged.pool.cache[0, layer_id]
    v_cache = paged.pool.cache[1, layer_id]
    q_lens = [qe - qs for qs, qe in zip(paged.cu_seqlens[:-1], paged.cu_seqlens[1:])]
    decode_only = bool(q_lens) and all(ql == 1 for ql in q_lens)
    if state.flashinfer_mode is None:
        _plan_eager(decode_w, wrappers, paged, q, page_size, decode_only, state,
                     num_attention_heads, num_key_value_heads, head_dim)
    if state.flashinfer_mode == "decode":
        out = decode_w.run(q, (k_cache, v_cache))
    else:
        out = wrappers["prefill"].run(q, (k_cache, v_cache))
    return out.contiguous().view(q.shape[0], num_attention_heads * head_dim)


def _plan_eager(
    decode_w,
    wrappers: dict,
    paged: PagedBatch,
    q: torch.Tensor,
    page_size: int,
    decode_only: bool,
    state: AttentionState,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    if decode_only:
        plan_flashinfer_decode(
            decode_w,
            paged,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            q.dtype,
            disable_split_kv=state.flashinfer_wrapper is not None,
        )
        state.flashinfer_mode = "decode"
        return
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
    state.flashinfer_mode = "prefill"
