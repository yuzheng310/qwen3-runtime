"""Decode-only CUDA Graph. Factory default is on for CUDA+FlashInfer.

Never replayed on prefill. FlashInfer: ``plan()`` outside capture, ``run()`` +
append inside. Triton: packed page table / kv_lens / frozen split-K updated
outside capture.

EXP-011 capped graph replay because frozen-CTA graphs fell behind eager
split-KV at long KV. EXP-011R left split-KV on and the graph led at every
length and batch, so ``build_engine`` passes ``max_kv=None`` unless
``disable_split_kv`` is set. The cap still guards the frozen-CTA path —
raising it without clearing that flag faults at batch 8 / 8K KV.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from qwen3_runtime.attention.flashinfer_backend import (
    make_flashinfer_graph_decode_wrapper,
    plan_flashinfer_decode,
)
from qwen3_runtime.attention.state import AttentionState
from qwen3_runtime.kv.paged import PagedBatch


def is_pure_decode(reqs: list) -> bool:
    return bool(reqs) and all(
        req.num_computed_tokens >= req.num_prompt_tokens
        and req.num_scheduled_tokens == 1
        and req.uncomputed_tokens == 1
        for req in reqs
    )


def decode_kv_page_signature(
    block_tables: list,
    kv_lens: Sequence[int],
    page_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Occupied page ids per sequence. Intra-page decode does not change this."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    sig: list[tuple[int, ...]] = []
    for table, kv in zip(block_tables, kv_lens, strict=True):
        n_pages = (int(kv) + page_size - 1) // page_size
        if n_pages <= 0:
            raise ValueError("decode plan signature requires kv_len > 0")
        if isinstance(table, torch.Tensor):
            pages = tuple(int(x) for x in table[:n_pages].tolist())
        else:
            pages = tuple(int(x) for x in table[:n_pages])
        if len(pages) < n_pages:
            raise ValueError("block table shorter than occupied pages")
        sig.append(pages)
    return tuple(sig)


def can_skip_flashinfer_decode_plan(
    previous: tuple[tuple[int, ...], ...] | None,
    block_tables: list,
    kv_lens: Sequence[int],
    page_size: int,
) -> tuple[bool, tuple[tuple[int, ...], ...]]:
    current = decode_kv_page_signature(block_tables, kv_lens, page_size)
    return previous is not None and previous == current, current


def use_decode_cuda_graph(reqs: list, *, max_kv: int | None) -> bool:
    """Replay decode graphs only while KV fits in ``max_kv`` (None = no cap)."""
    if not is_pure_decode(reqs):
        return False
    if max_kv is None:
        return True
    return all(req.num_computed_tokens <= max_kv for req in reqs)


class DecodeCudaGraph:
    """Decode graph: plan/metadata outside, run()+linears inside."""

    def __init__(self, runner, reqs: list):
        self.runner = runner
        self.batch_size = len(reqs)
        self.backend = runner.model.attention_backend
        pool = runner.pool
        device = pool.cache.device
        self._cfg = runner.model.cfg
        self._planned_pages: tuple[tuple[int, ...], ...] | None = None
        self._alloc_io(device, pool)
        self.attn_state = (
            self._setup_triton(runner, device, pool)
            if self.backend == "triton"
            else self._setup_flashinfer(device, pool)
        )
        self.batch = PagedBatch(
            pool,
            self.slots,
            [torch.zeros(0, dtype=torch.long, device=device) for _ in range(self.batch_size)],
            [0] * self.batch_size,
            list(range(self.batch_size + 1)),
        )
        self._fill(reqs, force_plan=True)
        self._capture(runner)

    def _alloc_io(self, device: torch.device, pool) -> None:
        n = self.batch_size
        cfg = self._cfg
        vocab = cfg.vocab_size
        dtype = next(self.runner.model.parameters()).dtype
        self.ids = torch.zeros(n, dtype=torch.long, device=device)
        self.pos = torch.zeros(n, dtype=torch.long, device=device)
        self.slots = torch.zeros(n, dtype=torch.long, device=device)
        self.logits = torch.empty(n, vocab, dtype=dtype, device=device)
        self.append_pos = torch.zeros(n, dtype=torch.int32, device=device)
        self.wrapper, self._indptr, self._indices, self._last_page_len = (
            # APC can reference the same physical page from every sequence.
            # FlashInfer stores logical references, not unique physical pages.
            make_flashinfer_graph_decode_wrapper(device, n, n * pool.num_blocks)
        )
        self._last_page_len_host = torch.zeros(n, dtype=torch.int32)

    def _setup_flashinfer(self, device: torch.device, pool) -> AttentionState:
        del pool
        return AttentionState(
            flashinfer_mode="decode",
            flashinfer_wrapper=self.wrapper,
            append_batch_indices=torch.arange(self.batch_size, dtype=torch.int32, device=device),
            append_positions=self.append_pos,
            kv_indptr_buf=self._indptr,
            kv_indices_buf=self._indices,
            kv_last_page_len_buf=self._last_page_len,
        )

    def _setup_triton(self, runner, device: torch.device, pool) -> AttentionState:
        from qwen3_runtime.kernels.triton_decode import pick_num_splits

        cap = runner.decode_graph_max_kv or 2048
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        splits = pick_num_splits(self.batch_size, self._cfg.num_key_value_heads, cap, sm)
        self.triton_table = torch.zeros(
            self.batch_size, pool.num_blocks, dtype=torch.int32, device=device
        )
        self.triton_kv_lens = torch.zeros(self.batch_size, dtype=torch.int32, device=device)
        state = self._setup_flashinfer(device, pool)
        state.flashinfer_mode = None
        state.triton_block_table = self.triton_table
        state.triton_kv_lens = self.triton_kv_lens
        state.triton_num_splits = splits
        return state

    def _capture(self, runner) -> None:
        with torch.inference_mode():
            for _ in range(3):
                self.logits.copy_(
                    runner.model(self.ids, self.pos, paged=self.batch, attn_state=self.attn_state)
                )
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.logits.copy_(
                runner.model(self.ids, self.pos, paged=self.batch, attn_state=self.attn_state)
            )
        with torch.inference_mode():
            self.graph.replay()

    def _fill(self, reqs: list, *, force_plan: bool = False) -> None:
        if len(reqs) != self.batch_size:
            raise ValueError("CUDA Graph batch size is frozen")
        tables, kv_lens = self._copy_decode_inputs(reqs)
        self.batch.block_tables = tables
        self.batch.kv_lens = kv_lens
        if self.backend == "triton":
            self._update_triton_pages(tables, kv_lens)
        self._sync_flashinfer_pages(tables, kv_lens, force_plan=force_plan)

    def _copy_decode_inputs(self, reqs: list) -> tuple[list, list[int]]:
        runner = self.runner
        tables = []
        kv_lens = []
        for i, req in enumerate(reqs):
            start = req.num_computed_tokens
            self.ids[i] = req.token_ids[start]
            self.pos[i] = start
            self.append_pos[i] = start
            self.slots[i] = runner.block_manager.slot_mapping(req, start, 1)[0]
            tables.append(req.block_table)
            kv_lens.append(start + 1)
        return tables, kv_lens

    def _update_triton_pages(self, tables: list, kv_lens: list[int]) -> None:
        page_size = self.runner.pool.block_size
        device = self.triton_table.device
        for i, (table, kv) in enumerate(zip(tables, kv_lens)):
            n_pages = (int(kv) + page_size - 1) // page_size
            if isinstance(table, torch.Tensor):
                pages = table[:n_pages].to(device=device, dtype=torch.int32)
            else:
                pages = torch.tensor(table[:n_pages], dtype=torch.int32, device=device)
            if pages.numel() < n_pages:
                raise ValueError("block table shorter than occupied pages")
            self.triton_table[i, :n_pages].copy_(pages)
            self.triton_kv_lens[i] = kv

    def _sync_flashinfer_pages(
        self, tables: list, kv_lens: list[int], *, force_plan: bool
    ) -> None:
        page_size = self.runner.pool.block_size
        skip, signature = can_skip_flashinfer_decode_plan(
            None if force_plan else self._planned_pages,
            tables,
            kv_lens,
            page_size,
        )
        if skip:
            for i, kv in enumerate(kv_lens):
                rem = kv % page_size
                self._last_page_len_host[i] = page_size if rem == 0 else rem
            self.attn_state.kv_last_page_len_buf[: self.batch_size].copy_(self._last_page_len_host)
            self.attn_state.flashinfer_mode = "decode"
            return
        cfg = self._cfg
        plan_flashinfer_decode(
            self.wrapper,
            self.batch,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            next(self.runner.model.parameters()).dtype,
            disable_split_kv=self.runner.decode_graph_disable_split_kv,
        )
        self._planned_pages = signature
        self.attn_state.flashinfer_mode = "decode"

    def replay(self, reqs: list, *, force_plan: bool = False) -> torch.Tensor:
        self._fill(reqs, force_plan=force_plan)
        with torch.inference_mode():
            self.graph.replay()
        return self.logits

    def release(self) -> None:
        self.graph = None
        self.wrapper = None
        self.batch = None
        self.runner = None
        self.logits = None
        self.ids = None
        self.pos = None
        self.slots = None
        self.attn_state = None
