"""One request's token ids, KV bookkeeping, sampling, and stop conditions."""

from __future__ import annotations

from enum import Enum, auto
from itertools import count
from typing import TYPE_CHECKING

from qwen3_runtime.engine.prefix_cache import ROOT_HASH
from qwen3_runtime.sampling import make_generator
from qwen3_runtime.sampling_params import SamplingParams

if TYPE_CHECKING:
    from qwen3_runtime.spec_decode import NgramIndex


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()


class Request:
    """One inference request. The engine speaks token ids, not text."""

    _ids = count()

    def __init__(
        self,
        token_ids: list[int],
        max_tokens: int = 16,
        ignore_eos: bool = True,
        *,
        sampling: SamplingParams | None = None,
        stop_token_ids: tuple[int, ...] = (),
        stop_strings: tuple[str, ...] = (),
        forced_tokens: list[int] | None = None,
        min_tokens: int | None = None,
        detokenizer=None,
    ):
        if not token_ids:
            raise ValueError("prompt token_ids must be non-empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.request_id = next(Request._ids)
        self.status = RequestStatus.WAITING
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_computed_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.hold_kv = False
        self.cached_tokens = 0
        self.n_published_blocks = 0
        self.prefix_parent = ROOT_HASH
        self.sampling = sampling or SamplingParams()
        self.stop_token_ids = tuple(stop_token_ids)
        self.stop_strings = tuple(stop_strings or self.sampling.stop_strings)
        self.min_tokens = int(self.sampling.min_tokens if min_tokens is None else min_tokens)
        self.forced_tokens = list(forced_tokens) if forced_tokens is not None else None
        self.rng = make_generator(self.sampling.seed)
        self.spec_draft_len = 0
        self.spec_teacher_force = False
        self.kv_epoch = 0
        # GPU/CPU residency is separate from RequestStatus.PAUSED.  A CPU
        # resident request remains matchable but is never schedulable until a
        # restore transaction rebuilds its GPU table.
        self.kv_residency = "none"
        self.offload_snapshot_id: int | None = None
        self.last_logprob: float | None = None
        self.logprobs: list[float] = []
        self.top_logprobs: list[list[tuple[int, float]]] = []
        self.finish_reason: str | None = None
        self.detokenizer = detokenizer
        self.ngram_index: NgramIndex | None = None

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    @property
    def uncomputed_tokens(self) -> int:
        """Tokens in `token_ids` not yet written to KV (prompt or preempt recompute)."""
        return max(0, len(self.token_ids) - self.num_computed_tokens)

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        if self.detokenizer is not None:
            self.detokenizer.feed(token_id)

    def next_forced_token(self) -> int | None:
        """Next teacher-forced id, or None. Cursor is output tokens already committed."""
        if self.forced_tokens is None:
            return None
        produced = len(self.token_ids) - self.num_prompt_tokens
        if 0 <= produced < len(self.forced_tokens):
            return self.forced_tokens[produced]
        return None

    def remaining_forced_tokens(self) -> list[int]:
        if self.forced_tokens is None:
            return []
        produced = max(0, len(self.token_ids) - self.num_prompt_tokens)
        return list(self.forced_tokens[produced:])

    def generated_text(self) -> str:
        if self.detokenizer is None:
            return ""
        return self.detokenizer.generated_text()

    def reset_kv_state(self) -> None:
        """Clear computed KV bookkeeping after deallocate / sleep / weight update."""
        self.num_computed_tokens = 0
        self.num_scheduled_tokens = 0
        self.spec_draft_len = 0
        self.spec_teacher_force = False
        self.cached_tokens = 0
        self.n_published_blocks = 0
        self.ngram_index = None
        self.kv_residency = "none"
        self.offload_snapshot_id = None
