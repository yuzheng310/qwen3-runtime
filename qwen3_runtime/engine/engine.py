from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.engine.runner import ModelRunner
from qwen3_runtime.engine.scheduler import Scheduler
from qwen3_runtime.sampling import SamplingParams
from qwen3_runtime.spec_decode import trim_emitted, verify_greedy, verify_sampled


class Engine:
    def __init__(self, config: Config, runner: ModelRunner):
        self.config = config
        self.block_manager = BlockManager(
            num_blocks=config.num_kv_blocks,
            block_size=config.block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.scheduler = Scheduler(config, self.block_manager)
        self.runner = runner
        bind = getattr(runner, "bind", None)
        if bind is not None:
            bind(self.block_manager)
        if config.num_speculative_tokens > 0 and not hasattr(runner, "run_logits"):
            raise RuntimeError("speculative decoding needs a runner with run_logits")
        self._requests: dict[int, Request] = {}
        self.last_step_stats: dict | None = None
        self.last_emitted: dict[int, list[int]] = {}

    def add_request(
        self,
        token_ids: list[int],
        max_tokens: int = 16,
        *,
        ignore_eos: bool = True,
        hold_kv: bool = False,
        sampling: SamplingParams | None = None,
        stop_token_ids: tuple[int, ...] = (),
        forced_tokens: list[int] | None = None,
    ) -> int:
        req = Request(
            token_ids=token_ids,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            sampling=sampling,
            stop_token_ids=stop_token_ids,
            forced_tokens=forced_tokens,
        )
        req.hold_kv = hold_kv
        self.scheduler.add(req)
        self._requests[req.request_id] = req
        if self.config.enable_prefix_cache:
            self.block_manager.attach_cached_prefix(req)
        return req.request_id

    def resume_request(
        self,
        request_id: int,
        suffix: list[int],
        max_tokens: int,
        *,
        hold_kv: bool,
        sampling: SamplingParams | None = None,
        ignore_eos: bool | None = None,
        stop_token_ids: tuple[int, ...] | None = None,
        forced_tokens: list[int] | None = None,
    ) -> None:
        req = self._requests[request_id]
        self.scheduler.resume(
            req,
            suffix,
            max_tokens,
            hold_kv=hold_kv,
            sampling=sampling,
            ignore_eos=ignore_eos,
            stop_token_ids=stop_token_ids,
            forced_tokens=forced_tokens,
        )

    def finish_request(self, request_id: int) -> None:
        req = self._requests.get(request_id)
        if req is None:
            return
        self.scheduler.release(req)
        self._requests.pop(request_id, None)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def drain_request(self, request_id: int) -> list[int]:
        """Run until this request pauses or finishes. Other paused sessions stay resident."""
        req = self._requests[request_id]
        completion: list[int] = []
        while req.status not in (RequestStatus.PAUSED, RequestStatus.FINISHED):
            progressed = False
            for rid, _tok, _done in self.step():
                progressed = True
                if rid == request_id:
                    completion.extend(self.last_emitted.get(rid, []))
            if not progressed:
                raise RuntimeError(
                    f"request {request_id} status={req.status.name} made no progress"
                )
        return completion

    def step(self) -> list[tuple[int, int | None, bool]]:
        preempt_before = self.scheduler.num_preemptions
        reqs = self.scheduler.schedule()
        self.last_emitted = {}
        if not reqs:
            self.last_step_stats = None
            if self.scheduler.waiting or self.scheduler.running:
                raise RuntimeError("scheduler produced an empty step while requests remain")
            return []
        prefill_tokens = 0
        decode_tokens = 0
        prefill_reqs = 0
        decode_reqs = 0
        spec_proposed = 0
        spec_accepted = 0
        spec_verify_forwards = 0
        spec_accept_lens: list[int] = []
        for req in reqs:
            n = req.num_scheduled_tokens
            if req.is_prefill_complete:
                decode_tokens += n
                decode_reqs += 1
            else:
                prefill_tokens += n
                prefill_reqs += 1
        spec_reqs = [req for req in reqs if req.spec_draft_len]
        rest = [req for req in reqs if not req.spec_draft_len]
        token_by_id: dict[int, int | None] = {}
        finished_ids: set[int] = set()
        if spec_reqs:
            spec_verify_forwards = len(spec_reqs)
            logits = self.runner.run_logits(spec_reqs)
            offset = 0
            for req in spec_reqs:
                n = req.num_scheduled_tokens
                row = None if logits is None else logits[offset : offset + n]
                offset += n
                n_draft = req.spec_draft_len
                drafts = req.token_ids[-n_draft:] if n_draft else []
                emitted = self._verify_spec(req, row)
                n_acc = 0
                for draft, tok in zip(drafts, emitted):
                    if tok != draft:
                        break
                    n_acc += 1
                spec_proposed += n_draft
                spec_accepted += n_acc
                spec_accept_lens.append(n_acc)
                self._commit_spec(req, emitted)
                self.last_emitted[req.request_id] = emitted
                token_by_id[req.request_id] = emitted[-1] if emitted else None
                req.spec_draft_len = 0
                req.spec_teacher_force = False
                req.num_scheduled_tokens = 0
                if self.scheduler.complete_after_spec(req):
                    finished_ids.add(req.request_id)
        if rest:
            rest_toks = self.runner.run(rest)
            for req in self.scheduler.postprocess(rest, rest_toks):
                finished_ids.add(req.request_id)
            for req, tok in zip(rest, rest_toks):
                if tok is None:
                    self.last_emitted[req.request_id] = []
                    token_by_id[req.request_id] = None
                else:
                    new_tok = req.token_ids[-1]
                    self.last_emitted[req.request_id] = [new_tok]
                    token_by_id[req.request_id] = new_tok
        reported: list[int | None] = [token_by_id[req.request_id] for req in reqs]
        self.last_step_stats = {
            "req_ids": [req.request_id for req in reqs],
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_reqs": prefill_reqs,
            "decode_reqs": decode_reqs,
            "preempts": self.scheduler.num_preemptions - preempt_before,
            "spec_proposed": spec_proposed,
            "spec_accepted": spec_accepted,
            "spec_verify_forwards": spec_verify_forwards,
            "spec_accept_lens": spec_accept_lens,
        }
        result = [
            (req.request_id, tok, req.request_id in finished_ids)
            for req, tok in zip(reqs, reported)
        ]
        for req in reqs:
            if req.status == RequestStatus.FINISHED:
                self._requests.pop(req.request_id, None)
        return result

    def _verify_spec(self, req: Request, logits: object) -> list[int]:
        drafts = req.token_ids[-req.spec_draft_len :]
        if req.spec_teacher_force:
            return trim_emitted(req, drafts, eos_token_id=self.config.eos_token_id)
        if logits is None:
            raise RuntimeError("speculative verify needs target logits")
        if req.sampling.is_greedy():
            emitted = verify_greedy(drafts, logits)
        else:
            emitted = verify_sampled(drafts, logits, req.sampling, req.rng)
        return trim_emitted(req, emitted, eos_token_id=self.config.eos_token_id)

    def _commit_spec(self, req: Request, emitted: list[int]) -> None:
        n_draft = req.spec_draft_len
        prefix_len = len(req.token_ids) - n_draft
        if not emitted:
            req.token_ids = req.token_ids[:prefix_len]
            keep = max(0, len(req.token_ids) - 1)
            req.num_computed_tokens = keep
            self.block_manager.truncate_kv(req, keep)
            return
        req.token_ids = req.token_ids[:prefix_len] + emitted
        keep = len(req.token_ids) - 1
        req.num_computed_tokens = keep
        self.block_manager.truncate_kv(req, keep)

    def generate(
        self,
        token_ids: list[int],
        max_tokens: int = 16,
        *,
        ignore_eos: bool = True,
        sampling: SamplingParams | None = None,
        stop_token_ids: tuple[int, ...] = (),
        hold_kv: bool = False,
    ) -> list[int]:
        rid = self.add_request(
            token_ids,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            hold_kv=hold_kv,
            sampling=sampling,
            stop_token_ids=stop_token_ids,
        )
        return self.drain_request(rid)
