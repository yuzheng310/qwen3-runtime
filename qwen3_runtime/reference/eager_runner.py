"""CPU/reference runner: recompute each prefix, packed varlen causal attention."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from qwen3_runtime.engine.batch import last_token_indices, pack_eager
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request
from qwen3_runtime.engine.runner import ModelRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.sampling import sample_scheduled


class PytorchEagerRunner(ModelRunner):
    """Prefill chunks recompute from the start of the current sequence.

    This path is unified packed execution, not split prefill/decode forwards.
    """

    def __init__(self, model: Qwen3ForCausalLM):
        self.model = model
        self.model.eval()
        self.pool = None

    def bind(self, block_manager: BlockManager) -> None:
        del block_manager

    def release_kv_pool(self) -> None:
        return

    def offload_weights(self, *, keep_a_host_copy: bool = False) -> None:
        del keep_a_host_copy

    def reload_weights(self) -> None:
        return

    def rebuild_kv_pool(self) -> None:
        return

    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        packed = pack_eager(reqs)
        ids = torch.tensor(packed.ids, dtype=torch.long)
        pos = torch.tensor(packed.pos, dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(ids, pos, cu_seqlens=packed.cu_seqlens)
        last = logits.index_select(0, torch.tensor(last_token_indices(packed.cu_seqlens), dtype=torch.long))
        idx = [i for i, flag in enumerate(packed.sample) if flag]
        if not idx:
            return [None] * len(reqs)
        sampled = last.index_select(0, torch.tensor(idx, dtype=torch.long))
        return sample_scheduled(sampled, reqs, packed.sample)

    def run_logits(self, reqs: Sequence[Request]) -> torch.Tensor | None:
        """Logits for every scheduled token, packed in request order. None if KV-only."""
        reqs = list(reqs)
        if not reqs:
            return None
        if all(req.spec_teacher_force for req in reqs):
            return None
        packed = pack_eager(reqs)
        ids = torch.tensor(packed.ids, dtype=torch.long)
        pos = torch.tensor(packed.pos, dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(ids, pos, cu_seqlens=packed.cu_seqlens)
        return logits.index_select(0, torch.tensor(packed.scheduled_rows, dtype=torch.long))
