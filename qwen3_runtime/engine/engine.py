"""Engine: schedule → forward → sample. Session KV and speculative decode live here."""

from __future__ import annotations

from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.engine.runner import ModelRunner
from qwen3_runtime.engine.scheduler import Scheduler
from qwen3_runtime.engine.session_offload import SessionOffloadManager
from qwen3_runtime.sampling import SamplingParams
from qwen3_runtime.spec_decode import commit_spec, verify_request


class Engine:
    def __init__(self, config: Config, runner: ModelRunner):
        self.config = config
        if config.num_kv_blocks is None:
            raise ValueError(
                "num_kv_blocks must be set (factory computes it from free VRAM)"
            )
        self.block_manager = BlockManager(
            num_blocks=config.num_kv_blocks,
            block_size=config.block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.scheduler = Scheduler(config, self.block_manager)
        self.runner = runner
        self.runner.bind(self.block_manager)
        self._requests: dict[int, Request] = {}
        self.last_step_stats: dict | None = None
        self.last_emitted: dict[int, list[int]] = {}
        # Paired with ``last_emitted`` and published by the same step. Readers
        # used to reach back into ``_requests`` for ``last_logprob``, which is
        # one value however many tokens the step emitted, and which is gone
        # entirely once a finished request is popped below.
        self.last_emitted_logprobs: dict[int, list[float]] = {}
        self._fatal_error = None
        self._asleep = False
        self.last_sleep_memory: dict | None = None
        self.last_aborted_ids: list[int] = []
        if config.session_cpu_offload == "async":
            from qwen3_runtime.engine.async_offload import AsyncSnapshotOffloadManager

            self.session_offload = AsyncSnapshotOffloadManager(self)
        elif config.cpu_kv_backend == "block" and config.session_cpu_offload == "sync":
            from qwen3_runtime.engine.block_offload import BlockOffloadManager

            self.session_offload = BlockOffloadManager(self)
        else:
            self.session_offload = SessionOffloadManager(self)

    def add_request(
        self,
        token_ids: list[int],
        max_tokens: int = 16,
        *,
        ignore_eos: bool = True,
        hold_kv: bool = False,
        sampling: SamplingParams | None = None,
        stop_token_ids: tuple[int, ...] = (),
        stop_strings: tuple[str, ...] = (),
        forced_tokens: list[int] | None = None,
        tokenizer=None,
    ) -> int:
        self._check_health()
        from qwen3_runtime.serving.detokenizer import attach_detokenizer

        detok, strings = attach_detokenizer(
            tokenizer,
            token_ids,
            stop_strings,
            sampling.stop_strings if sampling else (),
        )
        req = Request(
            token_ids=token_ids,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            sampling=sampling,
            stop_token_ids=stop_token_ids,
            stop_strings=strings,
            forced_tokens=forced_tokens,
            detokenizer=detok,
        )
        req.hold_kv = hold_kv
        if self._asleep:
            raise RuntimeError("engine is asleep; wake_up() before add_request")
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
        self._check_health()
        req = self._requests[request_id]
        if self._asleep:
            raise RuntimeError("engine is asleep; wake_up() before resume_request")
        wait_for = getattr(self.session_offload, "wait_for", None)
        if wait_for is not None:
            wait_for(request_id)
        restore_ticket = None
        if req.kv_residency == "cpu":
            restore_ticket = self.session_offload.begin_restore(req)
        if req.block_table:
            stale_epoch = req.kv_epoch != self.block_manager.epoch
            if stale_epoch or req.num_computed_tokens == 0:
                raise RuntimeError(
                    f"resume of request {request_id} has a block table after KV invalidation"
                )
        try:
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
        except BaseException:
            self.session_offload.rollback_restore(req, restore_ticket)
            raise
        if restore_ticket is not None:
            self.session_offload.commit_restore(restore_ticket)

    def finish_request(self, request_id: int) -> None:
        req = self._requests.get(request_id)
        if req is None:
            return
        self.session_offload.remove(request_id)
        self.scheduler.release(req)
        self._requests.pop(request_id, None)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def config_report(self) -> dict[str, object]:
        """The knobs this engine actually got, read off the engine.

        Speculation was hardcoded off in one adapter and left on in the other
        for the whole rollout stage, and every published per-token number was
        attributed to a configuration nobody had written down. An argument at a
        call site is not evidence of what an engine is running; this is. Both
        front doors print it and the stats file carries it, so a measurement
        and its configuration travel together.

        ``num_kv_blocks`` comes from the block manager because the config value
        is a request and the pool is what was granted.
        """
        cfg = self.config
        return {
            "num_speculative_tokens": cfg.num_speculative_tokens,
            "ngram_min": cfg.ngram_min,
            "ngram_max": cfg.ngram_max,
            "enable_prefix_cache": cfg.enable_prefix_cache,
            "block_size": cfg.block_size,
            "num_kv_blocks": self.block_manager.num_blocks,
            "max_num_seqs": cfg.max_num_seqs,
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "attention_backend": cfg.attention_backend,
            "cuda_graph": cfg.cuda_graph,
            "eos_token_id": cfg.eos_token_id,
            "session_cpu_offload": cfg.session_cpu_offload,
            "cpu_kv_max_bytes": cfg.cpu_kv_max_bytes,
            "cpu_kv_pinned_max_bytes": cfg.cpu_kv_pinned_max_bytes,
            "transfer_chunk_bytes": cfg.transfer_chunk_bytes,
            "session_offload": self.session_offload.report(),
        }

    def stream_request(self, request_id: int):
        """Yield new token ids until this request pauses or finishes."""
        req = self._requests[request_id]
        logprobs: list[float] = []
        while req.status not in (RequestStatus.PAUSED, RequestStatus.FINISHED):
            progressed = False
            for rid, _tok, _done in self.step():
                progressed = True
                if rid == request_id:
                    emitted = self.last_emitted.get(rid, [])
                    step_logprobs = self.last_emitted_logprobs.get(rid, [])
                    if len(step_logprobs) != len(emitted):
                        raise RuntimeError(
                            f"request {rid} emitted {len(emitted)} tokens with "
                            f"{len(step_logprobs)} logprobs"
                        )
                    logprobs.extend(step_logprobs)
                    yield from emitted
            if not progressed:
                raise RuntimeError(
                    f"request {request_id} status={req.status.name} made no progress"
                )
        self.last_completion_logprobs = {request_id: logprobs}

    def drain_request(self, request_id: int) -> list[int]:
        return list(self.stream_request(request_id))

    def _check_health(self):
        if self._fatal_error is not None:
            raise RuntimeError(
                "engine terminated after fatal device error: " + self._fatal_error
            )

    def step(self) -> list[tuple[int, int | None, bool]]:
        self._check_health()
        try:
            return self._step_impl()
        except BaseException as exc:
            from qwen3_runtime.engine.session_offload import fatal_device_error

            if fatal_device_error(exc):
                self._fatal_error = str(exc)
            raise

    def _step_impl(self) -> list[tuple[int, int | None, bool]]:
        preempt_before = self.scheduler.num_preemptions
        reqs = self.scheduler.schedule()
        self.last_emitted = {}
        self.last_emitted_logprobs = {}
        if not reqs:
            self.last_step_stats = None
            if self.scheduler.waiting or self.scheduler.running:
                raise RuntimeError(
                    "scheduler produced an empty step while requests remain"
                )
            return []
        counts = self._collect_step_stats(reqs, preempt_before)
        spec_reqs = [req for req in reqs if req.spec_draft_len]
        rest = [req for req in reqs if not req.spec_draft_len]
        token_by_id: dict[int, int | None] = {}
        finished_ids: set[int] = set()
        spec_stats = self._run_spec_step(spec_reqs, token_by_id, finished_ids)
        if rest:
            self._run_normal_step(rest, token_by_id, finished_ids)
        self.last_step_stats = {**counts, **spec_stats}
        result = [
            (
                req.request_id,
                token_by_id[req.request_id],
                req.request_id in finished_ids,
            )
            for req in reqs
        ]
        for req in reqs:
            if req.status == RequestStatus.FINISHED:
                self.session_offload.remove(req.request_id)
                self._requests.pop(req.request_id, None)
        return result

    def _run_spec_step(
        self,
        spec_reqs: list[Request],
        token_by_id: dict[int, int | None],
        finished_ids: set[int],
    ) -> dict:
        stats = {
            "spec_proposed": 0,
            "spec_accepted": 0,
            "spec_verify_forwards": 0,
            "spec_accept_lens": [],
        }
        if not spec_reqs:
            return stats
        stats["spec_verify_forwards"] = len(spec_reqs)
        logits = self.runner.run_logits(spec_reqs)
        offset = 0
        for req in spec_reqs:
            n = req.num_scheduled_tokens
            row = None if logits is None else logits[offset : offset + n]
            offset += n
            n_draft = req.spec_draft_len
            drafts = req.token_ids[-n_draft:] if n_draft else []
            emitted, emitted_logprobs = verify_request(
                req, row, eos_token_id=self.config.eos_token_id
            )
            n_acc = 0
            for draft, tok in zip(drafts, emitted):
                if tok != draft:
                    break
                n_acc += 1
            stats["spec_proposed"] += n_draft
            stats["spec_accepted"] += n_acc
            stats["spec_accept_lens"].append(n_acc)
            commit_spec(req, emitted, emitted_logprobs, self.block_manager)
            self.last_emitted[req.request_id] = emitted
            self.last_emitted_logprobs[req.request_id] = emitted_logprobs
            token_by_id[req.request_id] = emitted[-1] if emitted else None
            req.spec_draft_len = 0
            req.spec_teacher_force = False
            req.num_scheduled_tokens = 0
            if self.scheduler.complete_after_spec(req):
                finished_ids.add(req.request_id)
        return stats

    def _run_normal_step(
        self,
        rest: list[Request],
        token_by_id: dict[int, int | None],
        finished_ids: set[int],
    ) -> None:
        # How long each record was before the runner touched it. Reading
        # ``logprobs[-1]`` afterwards is only correct if the runner appended
        # this step's value, and a runner that returns a token without scoring
        # it would otherwise hand back the previous token's logprob -- the same
        # staleness the driver used to have with ``last_logprob``.
        scored_before = [len(req.logprobs) for req in rest]
        rest_toks = self.runner.run(rest)
        if hasattr(self.session_offload, "mark_used"):
            self.session_offload.mark_used(rest)
        for req in self.scheduler.postprocess(rest, rest_toks):
            finished_ids.add(req.request_id)
        for req, tok, n_before in zip(rest, rest_toks, scored_before):
            if tok is None:
                self.last_emitted[req.request_id] = []
                self.last_emitted_logprobs[req.request_id] = []
                token_by_id[req.request_id] = None
            else:
                new_tok = req.token_ids[-1]
                self.last_emitted[req.request_id] = [new_tok]
                if len(req.logprobs) - n_before != 1:
                    raise RuntimeError(
                        f"request {req.request_id} emitted one token but recorded "
                        f"{len(req.logprobs) - n_before} logprobs"
                    )
                self.last_emitted_logprobs[req.request_id] = req.logprobs[-1:]
                token_by_id[req.request_id] = new_tok

    def _collect_step_stats(self, reqs: list[Request], preempt_before: int) -> dict:
        prefill_tokens = decode_tokens = prefill_reqs = decode_reqs = 0
        for req in reqs:
            n = req.num_scheduled_tokens
            if req.is_prefill_complete:
                decode_tokens += n
                decode_reqs += 1
            else:
                prefill_tokens += n
                prefill_reqs += 1
        return {
            "req_ids": [req.request_id for req in reqs],
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_reqs": prefill_reqs,
            "decode_reqs": decode_reqs,
            "preempts": self.scheduler.num_preemptions - preempt_before,
        }

    def generate(
        self, token_ids: list[int], max_tokens: int = 16, **kwargs
    ) -> list[int]:
        return list(
            self.stream_request(
                self.add_request(token_ids, max_tokens=max_tokens, **kwargs)
            )
        )

    def invalidate_all_kv(self) -> None:
        """Drop every block. Held sessions become empty-KV restarts, never stale tables."""
        self.session_offload.invalidate_all()
        for req in list(self._requests.values()):
            self.block_manager.deallocate(req)
            req.reset_kv_state()
        self.block_manager.reset()

    def offload_request(
        self, request_id: int, *, session_key: str | None = None
    ) -> None:
        """Synchronously save one paused request and release its physical blocks."""
        self._check_health()
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(request_id)
        self.session_offload.save(req, session_key=session_key or str(request_id))

    def has_cpu_snapshot(self, request_id: int) -> bool:
        return self.session_offload.has_snapshot(request_id)

    def snapshot_cuda_memory(self) -> dict[str, int | None]:
        from qwen3_runtime.rollout.lifecycle import snapshot_cuda_memory

        return snapshot_cuda_memory()

    def sleep(self, level: int = 1) -> dict[str, int | None]:
        from qwen3_runtime.rollout.lifecycle import sleep_engine

        return sleep_engine(self, level=level)

    def wake_up(self, tags: object | None = None) -> None:
        from qwen3_runtime.rollout.lifecycle import wake_engine

        wake_engine(self, tags=tags)

    def apply_named_weights(
        self,
        items: list[tuple[str, object]],
        *,
        invalidate_kv: bool = True,
    ) -> list[str]:
        from qwen3_runtime.rollout.lifecycle import (
            apply_named_weights,
            park_live_sessions,
        )

        if invalidate_kv:
            # Invalidate before the first parameter write.  If the update
            # fails halfway through, no old snapshot can be admitted against
            # a mixed model; the engine stays a cold-KV engine.
            self.session_offload.invalidate_all(weight_change=True)
            for req in list(self._requests.values()):
                self.block_manager.deallocate(req)
                req.reset_kv_state()
            self.block_manager.reset()
        applied = apply_named_weights(self.runner.model, items)
        if invalidate_kv:
            park_live_sessions(self)
        return applied

    def abort_generation(self) -> list[int]:
        from qwen3_runtime.rollout.lifecycle import abort_generation

        return abort_generation(self)
