"""End-to-end GPU regression for the frozen production decode path.

Requires NVIDIA GPU + FlashInfer + QWEN3_RUNTIME_MODEL. Exercises:

- DecodeCudaGraph
- FlashInfer paged decode
- FlashInfer append_paged_kv_cache (graph KV write)
- plan reuse / last-page-length update across a KV page boundary

This is evidence hardening, not a new feature. Skipped on CPU.
Uses a small KV pool so the factory's 85%-of-free-memory allocator is not
required; the production classes and FlashInfer path are unchanged.
"""

from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from qwen3_runtime.attention import paged as paged_mod
from qwen3_runtime.config import Config
from qwen3_runtime.engine.cuda_graph import can_skip_flashinfer_decode_plan
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.factory import PIN
from qwen3_runtime.engine.model_runner import SplitPagedRunner
from qwen3_runtime.engine.request import Request
from qwen3_runtime.utils.loader import load_from_directory


def _model_dir() -> Path | None:
    raw = os.environ.get("QWEN3_RUNTIME_MODEL")
    return Path(raw) if raw else None


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="requires FlashInfer"),
    pytest.mark.skipif(_model_dir() is None, reason="set QWEN3_RUNTIME_MODEL to pinned Qwen3-4B"),
]


def _free_engine(engine) -> None:
    runner = getattr(engine, "runner", None)
    if runner is not None:
        runner._decode_graphs.clear()
        runner.pool = None
    del engine
    paged_mod.FLASHINFER.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _occupied_kv(pool, req: Request) -> torch.Tensor:
    n = req.num_computed_tokens + req.num_scheduled_tokens
    n_pages = (n + pool.block_size - 1) // pool.block_size
    ids = [int(b) for b in req.block_table[:n_pages]]
    index = torch.tensor(ids, device=pool.cache.device, dtype=torch.long)
    return pool.cache[:, :, index].detach().float().cpu().contiguous()


def _make_engine(model, *, cuda_graph: bool) -> Engine:
    block_size = 16
    runner = SplitPagedRunner(
        model, cuda_graph=cuda_graph, decode_graph_max_kv=2048
    )
    return Engine(
        Config(
            block_size=block_size,
            num_kv_blocks=64,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            attention_backend="flashinfer",
            cuda_graph=cuda_graph,
        ),
        runner,
    )


def _run_production_decode(model, *, cuda_graph: bool) -> dict:
    """Greedy decode that crosses page_size=16 and records plan-skip events."""
    block_size = 16
    prompt_len = 10
    max_tokens = 24
    prompt = [151643, 8948, 198] + [2610] * (prompt_len - 3)
    engine = _make_engine(model, cuda_graph=cuda_graph)
    skip_events: list[dict] = []
    tokens: list[int] = []
    kv_final = None
    graph_ids: list[int] = []
    rid = engine.add_request(prompt, max_tokens=max_tokens, ignore_eos=True)
    while not engine.is_finished():
        reqs = engine.scheduler.schedule()
        assert reqs, "scheduler produced an empty batch"
        req = reqs[0]
        using_graph = engine.runner._use_decode_graph(reqs)
        graph = engine.runner._decode_graphs.get(len(reqs))
        if using_graph and graph is not None:
            kv_lens = [r.num_computed_tokens + r.num_scheduled_tokens for r in reqs]
            skip, signature = can_skip_flashinfer_decode_plan(
                graph._planned_pages,
                [r.block_table for r in reqs],
                kv_lens,
                block_size,
            )
            skip_events.append(
                {
                    "skip": skip,
                    "kv": kv_lens[0],
                    "pages": len(signature[0]),
                    "last_page_len": (
                        block_size if kv_lens[0] % block_size == 0 else kv_lens[0] % block_size
                    ),
                }
            )
        token_ids = engine.runner.run(reqs)
        graph_ids = sorted(engine.runner._decode_graphs)
        will_finish = False
        if token_ids[0] is not None:
            generated = len(req.token_ids) - req.num_prompt_tokens + 1
            will_finish = generated >= max_tokens
        if will_finish:
            kv_final = _occupied_kv(engine.runner.pool, req)
        finished = {r.request_id for r in engine.scheduler.postprocess(reqs, token_ids)}
        for req_id, tok, _done in (
            (r.request_id, tok, r.request_id in finished) for r, tok in zip(reqs, token_ids)
        ):
            if req_id == rid and tok is not None:
                tokens.append(tok)
    assert len(tokens) == max_tokens
    result = {
        "tokens": tokens,
        "skip_events": skip_events,
        "graph_batch_sizes": graph_ids,
        "kv": kv_final,
        "cuda_graph": engine.config.cuda_graph,
        "backend": engine.runner.model.attention_backend,
    }
    _free_engine(engine)
    return result


def test_decode_cuda_graph_flashinfer_append_crosses_page_and_matches_eager():
    model_dir = _model_dir()
    assert model_dir is not None
    model = load_from_directory(
        model_dir,
        device="cuda",
        dtype=torch.bfloat16,
        pin=PIN,
        attention_backend="flashinfer",
    )
    graph = _run_production_decode(model, cuda_graph=True)
    eager = _run_production_decode(model, cuda_graph=False)

    assert graph["backend"] == "flashinfer"
    assert eager["backend"] == "flashinfer"
    assert graph["cuda_graph"] is True
    assert eager["cuda_graph"] is False
    assert graph["graph_batch_sizes"] == [1]
    assert eager["graph_batch_sizes"] == []

    events = graph["skip_events"]
    assert events, "DecodeCudaGraph never replayed"
    kv_values = [e["kv"] for e in events]
    assert any(kv < 16 for kv in kv_values), "graph replay never ran within the first page"
    assert any(kv == 17 for kv in kv_values) or any(
        e["pages"] >= 2 and not e["skip"] for e in events
    ), "graph replay never observed the page-size transition"
    assert any(e["skip"] for e in events), "plan reuse / last-page-len update never happened"
    assert any(not e["skip"] for e in events), "FlashInfer plan() never ran after capture"

    assert graph["tokens"] == eager["tokens"], (
        "graph+append tokens diverged from eager FlashInfer "
        f"graph={graph['tokens']} eager={eager['tokens']}"
    )
    assert graph["kv"] is not None and eager["kv"] is not None
    assert graph["kv"].shape == eager["kv"].shape
    torch.testing.assert_close(graph["kv"], eager["kv"], atol=2e-2, rtol=2e-2)
