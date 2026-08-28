"""Decode-only CUDA Graph: tiny probe (keep=False) plus FlashInfer-native DecodeCudaGraph.

Factory default is on for CUDA+FlashInfer. Never replayed on prefill.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GraphEval:
    logits_match: bool
    eager_ms: float
    graph_ms: float
    keep: bool
    reason: str


def evaluate_decode_graph(
    fn: nn.Module | None = None,
    *,
    hidden: int = 16,
    iters: int = 8,
) -> GraphEval:
    """Compare a tiny decode-shaped matmul under eager vs CUDA Graph.

    This is a *method* probe (capture/replay + numeric match), not a claim that
    graphs belong in Baseline v1. `keep` is false until Nsight shows launch
    overhead dominating a real 4B decode step.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for graph evaluation")
    device = torch.device("cuda")
    w = torch.randn(hidden, hidden, device=device, dtype=torch.float16)
    x = torch.randn(1, hidden, device=device, dtype=torch.float16)

    def step(inp: torch.Tensor) -> torch.Tensor:
        return inp @ w

    torch.cuda.synchronize()
    for _ in range(3):
        step(x)
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iters):
        eager = step(x)
    ender.record()
    torch.cuda.synchronize()
    eager_ms = starter.elapsed_time(ender) / iters

    static_in = torch.empty_like(x)
    static_out = torch.empty_like(eager)
    static_in.copy_(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out.copy_(step(static_in))
    starter.record()
    for _ in range(iters):
        graph.replay()
    ender.record()
    torch.cuda.synchronize()
    graph_ms = starter.elapsed_time(ender) / iters
    match = torch.allclose(static_out, step(x), atol=1e-3, rtol=1e-3)
    return GraphEval(
        logits_match=match,
        eager_ms=eager_ms,
        graph_ms=graph_ms,
        keep=False,
        reason="tiny-matmul probe only; 4B decode graphs are DecodeCudaGraph (kept with FlashInfer)",
    )


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
    """Occupied page ids per sequence. Intra-page decode does not change this.

    FlashInfer graph ``plan()`` rebuilds kernel metadata from indptr/indices.
    When the occupied pages are unchanged, only ``last_page_len`` moves.
    Signatures include the page ids so a reused graph cannot skip ``plan()``
    across requests that happen to share a page count.
    """
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
    """Replay decode graphs only while KV fits in one prefill chunk.

    The gate exists because the graph path plans with ``disable_split_kv=True``,
    which leaves SMs idle at long KV: at 8K, graph 17.8 ms vs eager split-KV
    11.9 ms TPOT (EXP-011).

    That flag is **not** required for capture, which is what EXP-011 assumed.
    FlashInfer's decode kernel is persistent with a fixed grid, and upstream
    documents ``disable_split_kv`` as a determinism switch defaulting to False.
    Passing True does buy bitwise-stable output, which the greedy-match
    acceptance relied on. Whether dropping it removes the need for this gate
    entirely is open and unmeasured; re-running EXP-011 with the flag off is the
    experiment.
    """
    if not is_pure_decode(reqs):
        return False
    if max_kv is None:
        return True
    return all(req.num_computed_tokens <= max_kv for req in reqs)


class DecodeCudaGraph:
    """Decode graph: plan/metadata outside, run()+linears inside.

    FlashInfer: ``plan()`` outside capture; ``run()`` + append inside.
    Triton: packed page table / kv_lens / frozen split-K updated outside capture.
    """

    def __init__(self, runner, reqs: list):
        from qwen3_runtime.attention.paged import (
            make_flashinfer_graph_decode_wrapper,
            plan_flashinfer_decode,
        )
        from qwen3_runtime.kv.paged import PagedBatch

        self.runner = runner
        self.batch_size = len(reqs)
        self.backend = getattr(runner.model, "attention_backend", "flashinfer")
        pool = runner.pool
        device = pool.cache.device
        cfg = runner.model.cfg
        self.ids = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        self.pos = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        self.slots = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        self._cfg = cfg
        vocab = cfg.vocab_size
        dtype = next(runner.model.parameters()).dtype
        self.logits = torch.empty(self.batch_size, vocab, dtype=dtype, device=device)
        self.append_pos = torch.zeros(self.batch_size, dtype=torch.int32, device=device)
        self.wrapper = make_flashinfer_graph_decode_wrapper(
            device, self.batch_size, pool.num_blocks
        )
        self._plan = plan_flashinfer_decode
        self._planned_pages: tuple[tuple[int, ...], ...] | None = None
        self._last_page_len_host = torch.zeros(self.batch_size, dtype=torch.int32)
        triton_table = triton_kv_lens = None
        triton_splits = None
        if self.backend == "triton":
            from qwen3_runtime.kernels.triton_decode import pick_num_splits

            cap = runner.decode_graph_max_kv or 2048
            sm = torch.cuda.get_device_properties(device).multi_processor_count
            triton_splits = pick_num_splits(self.batch_size, cfg.num_key_value_heads, cap, sm)
            self.triton_table = torch.zeros(
                self.batch_size, pool.num_blocks, dtype=torch.int32, device=device
            )
            self.triton_kv_lens = torch.zeros(self.batch_size, dtype=torch.int32, device=device)
            triton_table = self.triton_table
            triton_kv_lens = self.triton_kv_lens
        self.batch = PagedBatch(
            pool,
            self.slots,
            [torch.zeros(0, dtype=torch.long, device=device) for _ in range(self.batch_size)],
            [0] * self.batch_size,
            list(range(self.batch_size + 1)),
            flashinfer_mode="decode" if self.backend == "flashinfer" else None,
            flashinfer_wrapper=self.wrapper,
            triton_block_table=triton_table,
            triton_kv_lens=triton_kv_lens,
            triton_num_splits=triton_splits,
        )
        self.batch.append_batch_indices = torch.arange(
            self.batch_size, dtype=torch.int32, device=device
        )
        self.batch.append_positions = self.append_pos
        self._fill(reqs)
        with torch.no_grad():
            for _ in range(3):
                self.logits.copy_(runner.model(self.ids, self.pos, paged=self.batch))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.logits.copy_(runner.model(self.ids, self.pos, paged=self.batch))
        with torch.no_grad():
            self.graph.replay()

    def _fill(self, reqs: list, *, force_plan: bool = False) -> None:
        if len(reqs) != self.batch_size:
            raise ValueError("CUDA Graph batch size is frozen")
        runner = self.runner
        page_size = runner.pool.block_size
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
        self.batch.block_tables = tables
        self.batch.kv_lens = kv_lens
        if self.backend == "triton":
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
            self.wrapper._paged_kv_last_page_len_buf[: self.batch_size].copy_(
                self._last_page_len_host
            )
            self.batch.flashinfer_mode = "decode"
            return
        cfg = self._cfg
        self._plan(
            self.wrapper,
            self.batch,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            next(runner.model.parameters()).dtype,
            disable_split_kv=True,
        )
        self._planned_pages = signature
        self.batch.flashinfer_mode = "decode"

    def replay(self, reqs: list, *, force_plan: bool = False) -> torch.Tensor:
        self._fill(reqs, force_plan=force_plan)
        with torch.no_grad():
            self.graph.replay()
        return self.logits
