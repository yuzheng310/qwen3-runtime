from collections.abc import Sequence

import torch

from qwen3_runtime.engine.batch import last_token_indices, sampled_logit_rows
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request
from qwen3_runtime.kv.paged import PagedBatch, PagedKVPool
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.sampling import sample_scheduled


class PytorchEagerRunner:
    """CPU/reference runner: recompute each prefix, packed varlen causal attention.

    Prefill chunks recompute from the start of the current sequence.
    This path is unified packed execution, not split prefill/decode forwards.
    """

    def __init__(self, model: Qwen3ForCausalLM):
        self.model = model
        self.model.eval()

    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        packed_ids: list[int] = []
        packed_pos: list[int] = []
        cu = [0]
        sample: list[bool] = []
        for req in reqs:
            end = req.num_computed_tokens + req.num_scheduled_tokens
            packed_ids.extend(req.token_ids[:end])
            packed_pos.extend(range(end))
            cu.append(len(packed_ids))
            sample.append(end >= len(req.token_ids))
        ids = torch.tensor(packed_ids, dtype=torch.long)
        pos = torch.tensor(packed_pos, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids, pos, cu_seqlens=cu)
        last = logits.index_select(0, torch.tensor(last_token_indices(cu), dtype=torch.long))
        tokens = sample_scheduled(last, reqs, sample)
        return tokens

    def run_logits(self, reqs: Sequence[Request]) -> torch.Tensor | None:
        """Logits for every scheduled token, packed in request order. None if KV-only."""
        reqs = list(reqs)
        if not reqs:
            return None
        if all(req.spec_teacher_force for req in reqs):
            return None
        packed_ids: list[int] = []
        packed_pos: list[int] = []
        cu = [0]
        select: list[int] = []
        for req in reqs:
            end = req.num_computed_tokens + req.num_scheduled_tokens
            start_row = len(packed_ids)
            packed_ids.extend(req.token_ids[:end])
            packed_pos.extend(range(end))
            cu.append(len(packed_ids))
            n = req.num_scheduled_tokens
            select.extend(range(start_row + end - n, start_row + end))
        ids = torch.tensor(packed_ids, dtype=torch.long)
        pos = torch.tensor(packed_pos, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids, pos, cu_seqlens=cu)
        return logits.index_select(0, torch.tensor(select, dtype=torch.long))


class PagedRunner:
    """Incremental paged KV: only scheduled tokens enter the model.

    Prefill chunks write KV and attend to cached prefix. Decode attends to the
    pool rather than recomputing the prompt. Packed queries use per-request
    gather (unified scheduling, paged execution).
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        cuda_graph: bool = False,
        decode_graph_max_kv: int | None = None,
    ):
        self.model = model
        self.model.eval()
        self.block_manager: BlockManager | None = None
        self.pool: PagedKVPool | None = None
        self.cuda_graph = cuda_graph
        self.decode_graph_max_kv = decode_graph_max_kv
        self._decode_graphs: dict[int, object] = {}
        self._graph_eager: dict[int, int] = {}

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

    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        if self.pool is None or self.block_manager is None:
            raise RuntimeError("PagedRunner.bind(block_manager) was not called")
        reqs = list(reqs)
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
        packed_ids: list[int] = []
        packed_pos: list[int] = []
        slots: list[int] = []
        cu = [0]
        tables: list = []
        kv_lens: list[int] = []
        for req in reqs:
            start = req.num_computed_tokens
            n = req.num_scheduled_tokens
            end = start + n
            packed_ids.extend(req.token_ids[start:end])
            packed_pos.extend(range(start, end))
            slots.extend(self.block_manager.slot_mapping(req, start, n))
            cu.append(len(packed_ids))
            tables.append(req.block_table)
            kv_lens.append(end)
        device = self.pool.cache.device
        ids = torch.tensor(packed_ids, dtype=torch.long, device=device)
        pos = torch.tensor(packed_pos, dtype=torch.long, device=device)
        slot_t = torch.tensor(slots, dtype=torch.long, device=device)
        batch = PagedBatch(self.pool, slot_t, tables, kv_lens, cu)
        need_logits = any(not req.spec_teacher_force for req in reqs)
        with torch.no_grad():
            if not need_logits:
                self.model(ids, pos, paged=batch, compute_logits=False)
                return None
            return self.model(ids, pos, paged=batch)

    def _use_decode_graph(self, reqs: list[Request]) -> bool:
        if not self.cuda_graph:
            return False
        if getattr(self.model, "attention_backend", None) not in ("flashinfer", "triton"):
            return False
        from qwen3_runtime.engine.cuda_graph import use_decode_cuda_graph

        return use_decode_cuda_graph(reqs, max_kv=self.decode_graph_max_kv)

    def _run_decode_graph(self, reqs: list[Request]) -> list[int | None]:
        from qwen3_runtime.engine.cuda_graph import DecodeCudaGraph

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
        sample = [
            req.num_computed_tokens + req.num_scheduled_tokens >= len(req.token_ids)
            for req in reqs
        ]
        return sample_scheduled(logits, reqs, sample)

    def _run_group(self, reqs: list[Request]) -> list[int | None]:
        packed_ids: list[int] = []
        packed_pos: list[int] = []
        slots: list[int] = []
        cu = [0]
        tables: list = []
        kv_lens: list[int] = []
        sample: list[bool] = []
        for req in reqs:
            start = req.num_computed_tokens
            n = req.num_scheduled_tokens
            end = start + n
            packed_ids.extend(req.token_ids[start:end])
            packed_pos.extend(range(start, end))
            slots.extend(self.block_manager.slot_mapping(req, start, n))
            cu.append(len(packed_ids))
            tables.append(req.block_table)
            kv_lens.append(end)
            sample.append(end >= len(req.token_ids))
        device = self.pool.cache.device
        ids = torch.tensor(packed_ids, dtype=torch.long, device=device)
        pos = torch.tensor(packed_pos, dtype=torch.long, device=device)
        slot_t = torch.tensor(slots, dtype=torch.long, device=device)
        batch = PagedBatch(self.pool, slot_t, tables, kv_lens, cu)
        rows = sampled_logit_rows(cu, sample)
        with torch.no_grad():
            if not rows:
                self.model(ids, pos, paged=batch, compute_logits=False)
                return [None] * len(reqs)
            # Decode-shaped batches already have one row per request. Skip
            # index_select so CUDA Graph capture stays model(ids, pos, paged).
            select = None
            if rows != list(range(ids.shape[0])):
                select = torch.tensor(rows, dtype=torch.long, device=device)
            logits = self.model(ids, pos, paged=batch, select_rows=select)
        if select is None:
            return sample_scheduled(logits, reqs, sample)
        sampled_reqs = [req for req, flag in zip(reqs, sample) if flag]
        sampled_flags = [True] * len(sampled_reqs)
        sampled_toks = sample_scheduled(logits, sampled_reqs, sampled_flags)
        out: list[int | None] = []
        j = 0
        for flag in sample:
            if flag:
                out.append(sampled_toks[j])
                j += 1
            else:
                out.append(None)
        return out


class SplitPagedRunner(PagedRunner):
    """Baseline v1 GPU default: split mixed prefill/decode into two forwards.

    Unified scheduling still produces one batch. Execution may issue:
    1. packed prefill/recompute chunk forward,
    2. decode forward.
    Token ids must match packed `PagedRunner` (invariance test).
    """

    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        if self.pool is None or self.block_manager is None:
            raise RuntimeError("PagedRunner.bind(block_manager) was not called")
        reqs = list(reqs)
        prefills: list[Request] = []
        decodes: list[Request] = []
        for req in reqs:
            if (
                req.num_computed_tokens >= req.num_prompt_tokens
                and req.num_scheduled_tokens == 1
                and req.uncomputed_tokens == 1
            ):
                decodes.append(req)
            else:
                prefills.append(req)
        if not prefills or not decodes:
            return PagedRunner.run(self, reqs)
        by_id: dict[int, int | None] = {}
        for group in (prefills, decodes):
            for req, tok in zip(group, PagedRunner.run(self, group)):
                by_id[req.request_id] = tok
        return [by_id[req.request_id] for req in reqs]
