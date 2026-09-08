"""Incremental paged KV: only scheduled tokens enter the model.

Prefill chunks write KV and attend to cached prefix. Decode attends to the
pool rather than recomputing the prompt. Packed queries use per-request
gather (unified scheduling, paged execution).

``split=True`` (CUDA factory default) issues mixed prefill/decode as two
forwards. Token ids match packed execution (invariance test).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from qwen3_runtime.engine.batch import pack_paged, sampled_logit_rows
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.cuda_graph import DecodeCudaGraph, is_pure_decode, use_decode_cuda_graph
from qwen3_runtime.engine.request import Request
from qwen3_runtime.engine.runner import ModelRunner
from qwen3_runtime.engine.weights import assert_weights_usable, offload_weights, reload_weights
from qwen3_runtime.kv.paged import PagedBatch, PagedKVPool
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.sampling import sample_scheduled


class PagedRunner(ModelRunner):
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        cuda_graph: bool = False,
        decode_graph_max_kv: int | None = None,
        decode_graph_disable_split_kv: bool = False,
        split: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.block_manager: BlockManager | None = None
        self.pool: PagedKVPool | None = None
        self.cuda_graph = cuda_graph
        self.decode_graph_max_kv = decode_graph_max_kv
        self.decode_graph_disable_split_kv = decode_graph_disable_split_kv
        self.split = split
        self._decode_graphs: dict[int, DecodeCudaGraph] = {}
        self._graph_eager: dict[int, int] = {}
        self._offloaded_from: torch.device | None = None
        self._released_params: dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]] = {}
        self._weights_unverified = False

    def bind(self, block_manager: BlockManager) -> None:
        self.block_manager = block_manager
        cfg = self.model.cfg
        param = next(self.model.parameters())
        self.pool = PagedKVPool(
            num_layers=cfg.num_hidden_layers,
            num_blocks=block_manager.num_blocks,
            block_size=block_manager.block_size,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=param.dtype,
            device=param.device,
        )
        block_manager.set_block_copier(self.pool.copy_block)

    def release_kv_pool(self) -> None:
        """Drop the device KV tensor and CUDA-graph captures that alias it."""
        graphs = list(self._decode_graphs.values())
        self._decode_graphs.clear()
        self._graph_eager.clear()
        for graph in graphs:
            graph.release()
        if self.block_manager is not None:
            self.block_manager.set_block_copier(None)
        if self.pool is not None:
            self.pool.cache = None
            self.pool = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rebuild_kv_pool(self) -> None:
        if self.block_manager is None:
            raise RuntimeError("PagedRunner.bind(block_manager) was not called")
        if self._offloaded_from is not None:
            raise RuntimeError("reload_weights() must run before rebuild_kv_pool()")
        if self.pool is not None:
            self.release_kv_pool()
        self.bind(self.block_manager)

    def offload_weights(self, *, keep_a_host_copy: bool = False) -> None:
        offload_weights(self, keep_a_host_copy=keep_a_host_copy)

    def reload_weights(self) -> None:
        reload_weights(self)

    def assert_weights_usable(self) -> None:
        assert_weights_usable(self)

    @property
    def weights_offloaded(self) -> bool:
        return self._offloaded_from is not None or bool(self._released_params)

    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        if self.pool is None or self.block_manager is None:
            raise RuntimeError("PagedRunner.bind(block_manager) was not called")
        self.assert_weights_usable()
        reqs = list(reqs)
        if self.split:
            out = self._run_split(reqs)
            if out is not None:
                return out
        return self._forward(reqs)

    def _run_split(self, reqs: list[Request]) -> list[int | None] | None:
        prefills = [req for req in reqs if not is_pure_decode([req])]
        decodes = [req for req in reqs if is_pure_decode([req])]
        if not prefills or not decodes:
            return None
        by_id: dict[int, int | None] = {}
        for group in (prefills, decodes):
            for req, tok in zip(group, self._forward(group)):
                by_id[req.request_id] = tok
        return [by_id[req.request_id] for req in reqs]

    def _forward(self, reqs: list[Request]) -> list[int | None]:
        if self._use_decode_graph(reqs):
            return self._run_decode_graph(reqs)
        return self._run_group(reqs)

    def run_logits(self, reqs: Sequence[Request]) -> torch.Tensor | None:
        """Logits for every scheduled token. None when every request is teacher-forced."""
        if self.pool is None or self.block_manager is None:
            raise RuntimeError("PagedRunner.bind(block_manager) was not called")
        reqs = list(reqs)
        if not reqs:
            return None
        packed = pack_paged(reqs, self.block_manager)
        device = self.pool.cache.device
        ids, pos, batch = self._device_batch(packed, device)
        need_logits = any(not req.spec_teacher_force for req in reqs)
        with torch.inference_mode():
            if not need_logits:
                self.model(ids, pos, paged=batch, compute_logits=False)
                return None
            return self.model(ids, pos, paged=batch)

    def _use_decode_graph(self, reqs: list[Request]) -> bool:
        if not self.cuda_graph:
            return False
        if self.model.attention_backend not in ("flashinfer", "triton"):
            return False
        return use_decode_cuda_graph(reqs, max_kv=self.decode_graph_max_kv)

    def _run_decode_graph(self, reqs: list[Request]) -> list[int | None]:
        n = len(reqs)
        eager_n = self._graph_eager.get(n, 0)
        if n not in self._decode_graphs:
            if eager_n < 2:
                self._graph_eager[n] = eager_n + 1
                return self._run_group(reqs)
            self._decode_graphs[n] = DecodeCudaGraph(self, reqs)
            logits = self._decode_graphs[n].logits
        else:
            logits = self._decode_graphs[n].replay(reqs)
        sample = [True] * len(reqs)
        return sample_scheduled(logits, reqs, sample)

    def _run_group(self, reqs: list[Request]) -> list[int | None]:
        packed = pack_paged(reqs, self.block_manager)
        device = self.pool.cache.device
        ids, pos, batch = self._device_batch(packed, device)
        rows = sampled_logit_rows(packed.cu_seqlens, packed.sample)
        with torch.inference_mode():
            if not rows:
                self.model(ids, pos, paged=batch, compute_logits=False)
                return [None] * len(reqs)
            select = torch.tensor(rows, dtype=torch.long, device=device)
            logits = self.model(ids, pos, paged=batch, select_rows=select)
        return sample_scheduled(logits, reqs, packed.sample)

    def _device_batch(
        self, packed, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, PagedBatch]:
        ids = torch.tensor(packed.ids, dtype=torch.long, device=device)
        pos = torch.tensor(packed.pos, dtype=torch.long, device=device)
        slot_t = torch.tensor(packed.slots, dtype=torch.long, device=device)
        batch = PagedBatch(self.pool, slot_t, packed.block_tables, packed.kv_lens, packed.cu_seqlens)
        return ids, pos, batch


class SplitPagedRunner(PagedRunner):
    """Historical name for ``PagedRunner(..., split=True)``."""

    def __init__(self, model: Qwen3ForCausalLM, **kwargs):
        kwargs.setdefault("split", True)
        super().__init__(model, **kwargs)
