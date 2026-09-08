from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from qwen3_runtime.engine.block_manager import BlockManager
    from qwen3_runtime.engine.request import Request
    from qwen3_runtime.kv.paged import PagedKVPool
    from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM


class ModelRunner(ABC):
    """Forward + KV/weight lifecycle. Engine calls these; it does not probe with getattr."""

    model: Qwen3ForCausalLM
    pool: PagedKVPool | None

    @abstractmethod
    def bind(self, block_manager: BlockManager) -> None:
        """Attach the host block table to a device KV pool (no-op if the runner has none)."""

    @abstractmethod
    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        """Return one sampled token per request, or None if this step should not sample.

        A returned token must be accompanied by exactly one
        ``sampling.record_sampled`` call on that request. The engine refuses to
        emit a token it cannot score: the distribution a token was drawn from
        is the one thing a trainer cannot reconstruct afterwards, and a missing
        logprob has historically been filled downstream with a zero, which
        reads as certainty. ``sample_scheduled`` does this for both real
        runners.
        """

    @abstractmethod
    def run_logits(self, reqs: Sequence[Request]) -> torch.Tensor | None:
        """Packed logits for every scheduled token. None if every request is teacher-forced."""

    @abstractmethod
    def release_kv_pool(self) -> None:
        """Drop the device KV tensor and anything that aliases it."""

    @abstractmethod
    def offload_weights(self, *, keep_a_host_copy: bool = False) -> None:
        """Park or discard weights so a colocated trainer can use the VRAM."""

    @abstractmethod
    def reload_weights(self) -> None:
        """Put weights back on device, or allocate NaN-filled tensors for an incoming update."""

    @abstractmethod
    def rebuild_kv_pool(self) -> None:
        """Allocate a fresh KV pool on the weights' device. Requires bind() first."""
