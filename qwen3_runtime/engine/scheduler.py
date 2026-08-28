from collections import deque

from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.sampling import SamplingParams, make_generator
from qwen3_runtime.spec_decode import propose_drafts


class Scheduler:
    """Unified scheduling, memory-aware admission.

    One step may mix prefill chunks and decode tokens. Running requests are
    packed first; leftover ``max_num_batched_tokens`` admits waiting prefills.
    KV is reserved only for tokens scheduled this step. Chunk size also
    shrinks to remaining free blocks. If a running request still cannot
    obtain a slot, the youngest running request is preempted (free KV,
    recompute from token 0). Execution may still split prefills and
    decodes (see DESIGN.md).
    """

    def __init__(self, config: Config, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.paused: dict[int, Request] = {}
        self.num_preemptions = 0

    def add(self, req: Request) -> None:
        max_len = len(req.token_ids) + req.max_tokens
        needed = self.block_manager.blocks_needed_for(max_len)
        if needed > self.block_manager.num_blocks:
            raise RuntimeError(
                f"request length {max_len} needs {needed} KV blocks; pool has {self.block_manager.num_blocks}"
            )
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running and not self.paused

    def schedule(self) -> list[Request]:
        bound = self.config.max_num_seqs + len(self.running) + len(self.waiting) + 1
        for _ in range(max(bound, 1)):
            scheduled: list[Request] = []
            token_budget = self.config.max_num_batched_tokens
            token_budget = self._schedule_running(scheduled, token_budget)
            self._schedule_waiting(scheduled, token_budget)
            if scheduled:
                return scheduled
            if self.running:
                self._preempt(self.running.pop())
                continue
            if self.waiting:
                raise RuntimeError(
                    "waiting request exceeds the KV pool even after shrinking the chunk"
                )
            return []
        raise RuntimeError("scheduler could not admit a batch after preemption")

    def postprocess(self, reqs: list[Request], token_ids: list[int | None]) -> list[Request]:
        """Advance computed tokens; append samples for rows that caught up to seq len.

        `token_ids[i]` is None when request i did not produce a new token
        (incomplete prefill / recompute chunk).
        """
        if len(token_ids) != len(reqs):
            raise ValueError("token_ids must align with scheduled requests")
        finished: list[Request] = []
        for req, token_id in zip(reqs, token_ids):
            req.num_computed_tokens += req.num_scheduled_tokens
            req.num_scheduled_tokens = 0
            if self.config.enable_prefix_cache:
                self.block_manager.publish_full_blocks(req)
            if token_id is None:
                continue
            if req.forced_tokens is not None:
                produced = len(req.token_ids) - req.num_prompt_tokens
                if produced < len(req.forced_tokens):
                    token_id = req.forced_tokens[produced]
            req.append_token(token_id)
            if self._complete_if_done(req):
                finished.append(req)
        return finished

    def complete_after_spec(self, req: Request) -> bool:
        """Pause or finish a request whose spec tokens are already on token_ids."""
        if self.config.enable_prefix_cache:
            self.block_manager.publish_full_blocks(req)
        return self._complete_if_done(req)

    def _complete_if_done(self, req: Request) -> bool:
        n_out = len(req.token_ids) - req.num_prompt_tokens
        token_id = req.token_ids[-1]
        hit_stop = (not req.ignore_eos) and (
            token_id in req.stop_token_ids
            or (
                self.config.eos_token_id is not None
                and token_id == self.config.eos_token_id
            )
        )
        done = hit_stop or n_out >= req.max_tokens
        if not done:
            return False
        if req.hold_kv:
            self.pause(req)
        else:
            self._finish(req)
        return True

    def _maybe_attach_spec_drafts(self, req: Request, token_budget: int) -> None:
        k = self.config.num_speculative_tokens
        if k <= 0 or req.spec_draft_len or not req.is_prefill_complete:
            return
        if req.uncomputed_tokens != 1:
            return
        drafts, teacher = propose_drafts(
            req, k=k, ngram_min=self.config.ngram_min, ngram_max=self.config.ngram_max
        )
        if not drafts:
            return
        need = 1 + len(drafts)
        if need > token_budget:
            return
        if self.block_manager.max_allocatable_tokens(req, need) < need:
            return
        req.token_ids.extend(drafts)
        req.spec_draft_len = len(drafts)
        req.spec_teacher_force = teacher

    def _strip_spec_drafts(self, req: Request) -> None:
        n = req.spec_draft_len
        if n <= 0:
            return
        del req.token_ids[-n:]
        req.spec_draft_len = 0
        req.spec_teacher_force = False

    def pause(self, req: Request) -> None:
        """Keep KV resident; the request is not schedulable until resume()."""
        try:
            self.running.remove(req)
        except ValueError:
            pass
        req.num_scheduled_tokens = 0
        req.status = RequestStatus.PAUSED
        self.paused[req.request_id] = req

    def resume(
        self,
        req: Request,
        suffix: list[int],
        max_tokens: int,
        *,
        hold_kv: bool,
        sampling: SamplingParams | None = None,
        ignore_eos: bool | None = None,
        stop_token_ids: tuple[int, ...] | None = None,
        forced_tokens: list[int] | None = None,
    ) -> None:
        """Append a new prompt suffix and decode budget. Prefix KV stays."""
        if req.request_id not in self.paused:
            raise RuntimeError("resume of a request that is not paused")
        if not suffix:
            raise ValueError("resume suffix must be non-empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        max_len = len(req.token_ids) + len(suffix) + max_tokens
        needed = self.block_manager.blocks_needed_for(max_len)
        if needed > self.block_manager.num_blocks:
            raise RuntimeError(
                f"resume length {max_len} needs {needed} KV blocks; pool has {self.block_manager.num_blocks}"
            )
        del self.paused[req.request_id]
        req.token_ids.extend(suffix)
        req.num_prompt_tokens = len(req.token_ids)
        req.max_tokens = max_tokens
        req.hold_kv = hold_kv
        if sampling is not None:
            req.sampling = sampling
            req.rng = make_generator(sampling.seed)
        if ignore_eos is not None:
            req.ignore_eos = ignore_eos
        if stop_token_ids is not None:
            req.stop_token_ids = tuple(stop_token_ids)
        req.forced_tokens = list(forced_tokens) if forced_tokens is not None else None
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def release(self, req: Request) -> None:
        """Drop KV and remove the request from every queue (cancel / session end)."""
        self.paused.pop(req.request_id, None)
        try:
            self.waiting.remove(req)
        except ValueError:
            pass
        if req.status != RequestStatus.FINISHED:
            self._finish(req)

    def _schedule_running(self, scheduled: list[Request], token_budget: int) -> int:
        pending = list(self.running)
        token_budget = self._consider_running(pending, scheduled, token_budget)
        self.running = deque(pending)
        return token_budget

    def _consider_running(
        self,
        reqs: list[Request],
        scheduled: list[Request],
        token_budget: int,
    ) -> int:
        for req in reqs:
            if len(scheduled) >= self.config.max_num_seqs or token_budget <= 0:
                continue
            self._maybe_attach_spec_drafts(req, token_budget)
            n = self._tokens_to_schedule(req, token_budget)
            if n <= 0:
                self._strip_spec_drafts(req)
                continue
            if req.spec_draft_len and n != 1 + req.spec_draft_len:
                self._strip_spec_drafts(req)
                n = self._tokens_to_schedule(req, token_budget)
                if n <= 0:
                    continue
            self.block_manager.allocate_for_tokens(req, n)
            req.num_scheduled_tokens = n
            scheduled.append(req)
            token_budget -= n
        return token_budget

    def _schedule_waiting(self, scheduled: list[Request], token_budget: int) -> None:
        deferred: deque[Request] = deque()
        while self.waiting:
            if len(scheduled) >= self.config.max_num_seqs or token_budget <= 0:
                break
            req = self.waiting.popleft()
            n = self._tokens_to_schedule(req, token_budget)
            if n <= 0:
                deferred.append(req)
                continue
            self.block_manager.allocate_for_tokens(req, n)
            req.num_scheduled_tokens = n
            req.status = RequestStatus.RUNNING
            scheduled.append(req)
            self.running.append(req)
            token_budget -= n
        self.waiting.extendleft(reversed(deferred))

    def _tokens_to_schedule(self, req: Request, token_budget: int) -> int:
        remaining = req.uncomputed_tokens
        if remaining <= 0 or token_budget <= 0:
            return 0
        want = min(remaining, token_budget)
        return self.block_manager.max_allocatable_tokens(req, want)

    def _preempt(self, req: Request) -> None:
        self._strip_spec_drafts(req)
        self.block_manager.deallocate(req)
        req.num_computed_tokens = 0
        req.num_scheduled_tokens = 0
        req.cached_tokens = 0
        if self.config.enable_prefix_cache:
            self.block_manager.attach_cached_prefix(req)
        req.status = RequestStatus.WAITING
        self.waiting.appendleft(req)
        self.num_preemptions += 1

    def _finish(self, req: Request) -> None:
        req.status = RequestStatus.FINISHED
        self.block_manager.deallocate(req)
        try:
            self.running.remove(req)
        except ValueError:
            pass
