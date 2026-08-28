from enum import Enum, auto
from itertools import count

from qwen3_runtime.engine.prefix_cache import ROOT_HASH
from qwen3_runtime.sampling import SamplingParams, make_generator


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
        forced_tokens: list[int] | None = None,
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
        self.forced_tokens = list(forced_tokens) if forced_tokens is not None else None
        self.rng = make_generator(self.sampling.seed)
        self.spec_draft_len = 0
        self.spec_teacher_force = False

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
