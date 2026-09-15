"""Execute multi-turn sessions on one engine owner thread.

The caller supplies tokens and consumes a Turn. Matching, restore admission,
rollback, accounting, parking and pressure reclamation belong here; the
SessionCache and EngineDriver are internal seams, not caller protocols.
"""

from __future__ import annotations

import asyncio
import os

from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.session_offload import (
    RestoreAdmissionError,
    SessionOffloadError,
    fatal_device_error,
)
from qwen3_runtime.rollout.driver import AdmissionDeferred, EngineDriver, Turn
from qwen3_runtime.rollout.session import SessionCache, session_kv_enabled
from qwen3_runtime.sampling import SamplingParams

PARKED_KV_SHARE = 0.45
FREE_BLOCK_WATERMARK = 0.10


def _parked_kv_budget(engine: Engine, share: float = PARKED_KV_SHARE) -> int:
    """How many physical blocks resident sessions may sit on.

    The scheduler only ever preempts a *running* request, so a paused session
    keeps its blocks until something finishes it. Park too much and an incoming
    prefill has nowhere to go and the scheduler raises rather than reclaiming.
    The budget leaves the majority of the pool for requests that are working.
    Count is blocks, not tokens: prefix-cache sharing makes a token sum overstate
    the pages a parked session actually pins.
    """
    return max(1, int(share * engine.block_manager.num_blocks))


class SessionRollout:
    """Own session admission and retirement; the driver serializes both with steps."""

    def __init__(
        self,
        engine: Engine,
        *,
        stop_token_ids: tuple[int, ...] = (),
        enabled: bool | None = None,
        session_max_blocks: int | None = None,
        session_max_sessions: int | None = None,
    ):
        self.engine = engine
        self._stop_token_ids = stop_token_ids
        self._capacity_retries = 0
        self._enabled = session_kv_enabled() if enabled is None else enabled
        self._session_offload = getattr(engine, "session_offload", None)
        self._cpu_offload = bool(
            self.enabled and self._session_offload and self._session_offload.enabled
        )
        # Physical pages bound resident KV. Keep a separate count bound for
        # shared-page and CPU-only sessions, which may pin no additional pages.
        parked_blocks = (
            engine.block_manager.num_blocks
            if self._cpu_offload
            else _parked_kv_budget(engine)
        )
        if session_max_blocks is not None:
            if session_max_blocks < 1:
                raise ValueError("session_max_blocks must be positive")
            parked_blocks = session_max_blocks
        session_count = (
            parked_blocks if session_max_sessions is None else session_max_sessions
        )
        if session_count < 1:
            raise ValueError("session_max_sessions must be positive")
        self._sessions = SessionCache(
            max_blocks=parked_blocks, max_sessions=session_count,
            block_ids_for=self._session_block_ids,
        )
        self._driver = EngineDriver(engine, before_step=self._reclaim_for_active_step)
        # Prompt per in-flight request, so the turn can be parked under the
        # sequence its KV actually covers once the engine thread finishes it.
        self._pending_prompt: dict[int, list[int]] = {}
        # QWEN3_BATCHING=0 admits one turn at a time. Not a tuning knob: it is
        # the control arm, and it has to run through the same driver and the
        # same instrument as the batched one or the two spans are not comparable.
        self._one_at_a_time = (
            asyncio.Lock() if os.environ.get("QWEN3_BATCHING", "1") == "0" else None
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def run_turn(
        self,
        ids: list[int],
        *,
        max_tokens: int,
        sampling: SamplingParams | None,
        forced_tokens: list[int] | None = None,
        ignore_eos: bool = False,
        stop_token_ids: tuple[int, ...] | None = None,
    ) -> Turn:
        """Complete and park a turn before admitting its continuation.

        Explicit forced tokens use the same admission and recovery path as
        generation. They are for recorded replay, not policy sampling.
        """
        # Own the submitted history even if a caller reuses its list later.
        ids = list(ids)
        options = dict(
            max_tokens=max_tokens,
            sampling=sampling,
            forced_tokens=None if forced_tokens is None else list(forced_tokens),
            ignore_eos=ignore_eos,
            stop_token_ids=stop_token_ids,
        )
        if self._one_at_a_time is not None:
            async with self._one_at_a_time:
                return await self._run(ids, options)
        return await self._run(ids, options)

    async def _run(self, ids: list[int], options: dict) -> Turn:
        try:
            return await self._driver.run_turn(
                lambda: self._admit(ids, **options),
                on_done=self._park_or_release,
            )
        except RuntimeError as exc:
            if "KV pool" not in str(exc) or not self.enabled:
                raise
            self._capacity_retries += 1
            self._driver.run_on_engine(self._drop_sessions)
            return await self._driver.run_turn(
                lambda: self._admit(ids, **options),
                on_done=self._park_or_release,
            )

    def _admit(
        self,
        ids: list[int],
        *,
        max_tokens: int,
        sampling: SamplingParams | None,
        forced_tokens: list[int] | None = None,
        ignore_eos: bool = False,
        stop_token_ids: tuple[int, ...] | None = None,
    ) -> int:
        """Reserve and resume atomically; only admitted work enters the ledger."""
        stops = self._stop_token_ids if stop_token_ids is None else stop_token_ids
        options = dict(
            sampling=sampling,
            ignore_eos=ignore_eos,
            stop_token_ids=stops,
            forced_tokens=forced_tokens,
        )
        claim = self._sessions.reserve_claim(ids) if self.enabled else None
        if claim is not None:
            try:
                self.engine.resume_request(
                    claim.request_id,
                    claim.suffix,
                    max_tokens,
                    hold_kv=True,
                    **options,
                )
            except SessionOffloadError as exc:
                self._sessions.rollback_claim(claim)
                if not self._cpu_offload:
                    raise
                if isinstance(exc, RestoreAdmissionError):
                    scheduler = self.engine.scheduler
                    if scheduler.running or scheduler.waiting:
                        raise AdmissionDeferred(
                            "defer CPU restore until active work yields"
                        ) from exc
                    snapshot = self._session_offload.snapshot_for(claim.request_id)
                    if snapshot is not None:
                        manager = self.engine.block_manager
                        before = manager.num_free_blocks
                        self._offload_oldest_until_safe(
                            min_free=snapshot.metadata.logical_blocks
                        )
                        if manager.num_free_blocks > before:
                            raise AdmissionDeferred(
                                "defer CPU restore after reclaiming idle KV"
                            ) from exc
                self._sessions.forget(claim.request_id)
                self.engine.finish_request(claim.request_id)
            except BaseException:
                self._sessions.rollback_claim(claim)
                raise
            else:
                self._sessions.commit_claim(claim)
                self._pending_prompt[claim.request_id] = ids
                return claim.request_id
        request_id = self.engine.add_request(
            ids,
            max_tokens=max_tokens,
            hold_kv=self.enabled,
            **options,
        )
        if self.enabled:
            self._sessions.record_start(ids)
        self._pending_prompt[request_id] = ids
        return request_id

    def _session_block_ids(self, request_id: int) -> list[int]:
        req = self.engine._requests.get(request_id)
        return [] if req is None else req.block_table

    def _park_or_release(self, request_id: int, tokens: list[int]) -> None:
        """Retire one turn. Runs on the engine thread, before the next is admitted."""
        ids = self._pending_prompt.pop(request_id, None)
        if not self.enabled or ids is None:
            return
        req = self.engine._requests.get(request_id)
        n_blocks = len(getattr(req, "block_table", None) or [])
        self._retire(self._sessions.park(request_id, ids + tokens, n_blocks))
        self._evict_for_free_watermark()

    def _poll_offload(self, *, wait=False):
        poll = getattr(self._session_offload, "poll", None)
        if poll is None:
            return
        for rid in poll(wait=wait):
            self._sessions.update_blocks(rid, len(self._session_block_ids(rid)))

    def _early_offload(self):
        if not self._cpu_offload:
            return
        fraction = getattr(self.engine.config, "session_offload_early_fraction", 0.0)
        manager = self.engine.block_manager
        if not fraction or getattr(self._session_offload, "pending", None) is not None:
            return
        target = max(1, int(fraction * manager.num_blocks))
        if manager.num_free_blocks >= target:
            return
        manager.reclaim_cached_blocks(target)
        if manager.num_free_blocks >= target:
            return
        for session in sorted(self._sessions._sessions.values(), key=lambda s: s.stamp):
            if session.request_id in self._sessions._claims:
                continue
            req = self.engine._requests.get(session.request_id)
            if req is None or not req.block_table:
                continue
            # One bounded attempt per step. A zero-marginal shared group stays
            # with the existing grouped hard-pressure fallback.
            if manager.reclaimable_blocks([req]) <= 0:
                continue
            try:
                self._session_offload.early_save(req, session_key=str(req.request_id))
            except (RuntimeError, MemoryError) as exc:
                if fatal_device_error(exc):
                    self.engine._fatal_error = str(exc)
                if self.engine._fatal_error:
                    raise
            self._sessions.update_blocks(req.request_id, len(req.block_table))
            manager.reclaim_cached_blocks(target)
            break

    def _reclaim_for_active_step(self) -> None:
        """Idle KV must yield before a runnable request preempts itself."""
        self._poll_offload()
        self._early_offload()
        if getattr(self._session_offload, "pending", None) is not None:
            self._session_offload.stats["async_steps_inflight"] += 1
        manager = getattr(self.engine, "block_manager", None)
        if not self.enabled or manager is None or manager.num_free_blocks > 0:
            return
        self._evict_for_free_watermark(min_free=1)

    def _offload_oldest_until_safe(self, *, min_free: int | None = None) -> None:
        if not self._cpu_offload:
            return
        manager = getattr(self.engine, "block_manager", None)
        if manager is None:
            return
        if min_free is None:
            min_free = max(1, int(FREE_BLOCK_WATERMARK * manager.num_blocks))
        # One bounded pressure episode: consider oldest prefixes of at most
        # eight parked members. Never retry a failed signature in this call.
        parked = sorted(self._sessions._sessions.values(), key=lambda x: x.stamp)
        candidates = [
            self.engine._requests[s.request_id]
            for s in parked
            if s.request_id not in self._sessions._claims
            and s.request_id in self.engine._requests
            and self.engine._requests[s.request_id].block_table
        ]
        attempts = 0
        while candidates and manager.num_free_blocks < min_free and attempts < 8:
            group = []
            for req in candidates[:8]:
                group.append(req)
                if (
                    manager.reclaimable_blocks(group)
                    >= min_free - manager.num_free_blocks
                ):
                    break
            attempts += 1
            result = self._session_offload.reclaim_group(group, min_free)
            for req in group:
                self._sessions.update_blocks(req.request_id, len(req.block_table))
            candidates = candidates[len(group) :]
            if result["goal_satisfied"]:
                break

    def _evict_for_free_watermark(self, *, min_free: int | None = None) -> None:
        """Keep a slice of the pool free so the next prefill can allocate.

        The 45% parked-block cap is a static bound. This is the live one: if
        free pages have already dropped under 10% of the pool, drop the oldest
        parked session until they have not, or until nothing is parked.
        """
        manager = getattr(self.engine, "block_manager", None)
        if manager is None or not self.enabled:
            return
        num_blocks = getattr(manager, "num_blocks", None)
        if not num_blocks:
            return
        if min_free is None:
            min_free = max(1, int(FREE_BLOCK_WATERMARK * num_blocks))
        # Do not copy a session just because disposable APC entries fill the
        # free list. Exhaust that unreferenced tier before touching live KV.
        reclaim = getattr(manager, "reclaim_cached_blocks", None)
        if reclaim is not None:
            reclaim(min_free)
        if manager.num_free_blocks < min_free:
            self._poll_offload(wait=True)
            if reclaim is not None:
                reclaim(min_free)
        self._offload_oldest_until_safe(min_free=min_free)
        while True:
            num_free = getattr(manager, "num_free_blocks", None)
            if num_free is None or num_free >= min_free:
                break
            extra = self._sessions.oldest_id(require_blocks=True)
            if extra is None:
                break
            req = self.engine._requests.get(extra)
            if req is None:
                self._sessions.forget(extra)
                continue
            if self._session_offload is None:
                self._sessions.evict_oldest(require_blocks=True)
                self.engine.finish_request(extra)
                continue
            self._session_offload.remove(extra)
            manager.deallocate(req)
            req.num_computed_tokens = req.num_scheduled_tokens = req.cached_tokens = (
                req.n_published_blocks
            ) = 0
            req.kv_residency = "none"
            req.offload_snapshot_id = None
            self._sessions.update_blocks(extra, 0)
            manager.reclaim_cached_blocks(min_free)

    def _retire(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self.engine.finish_request(request_id)

    async def finish_session(self, token_ids: list[int]) -> int:
        """Release an explicitly finished trajectory, including a CPU snapshot."""
        history = list(token_ids)

        def finish() -> int:
            request_id = self._sessions.finish_exact(history)
            if request_id is None:
                return 0
            self.engine.finish_request(request_id)
            return 1

        # run_on_engine waits synchronously for the GPU owner. Keep that wait
        # off the caller/Ray event loop so other turns and timeouts can run.
        return await asyncio.to_thread(self._driver.run_on_engine, finish)

    def _drop_sessions(self) -> None:
        """Forget every resident session, e.g. because the KV pool was dropped.

        A session still on the books after an invalidation would be resumed onto
        a block table that no longer describes anything.
        """
        self._retire(self._sessions.drop_all())

    def session_report(self) -> dict[str, float | int]:
        report = self._sessions.report()
        if self._session_offload is not None:
            report.update(
                {f"offload_{k}": v for k, v in self._session_offload.report().items()}
            )
        return report

    def stop(self) -> None:
        """Join the owner thread before an external weight/pool operation."""
        self._driver.stop()

    def clear(self, *, reset_stats: bool = False) -> None:
        """Stop execution and release registered sessions before invalidation."""
        self.stop()
        self._drop_sessions()
        self._pending_prompt.clear()
        if reset_stats:
            self._sessions.reset_stats()
            for key in self._driver.stats:
                self._driver.stats[key] = 0
            self._capacity_retries = 0

    def abort(self) -> None:
        self.stop()
        owned = set(self._pending_prompt) | set(self._sessions._sessions)
        self._pending_prompt.clear()
        self._sessions.drop_all()
        self.engine.abort_generation()
        for request_id in owned:
            self.engine.finish_request(request_id)

    def batching_report(self) -> dict[str, float | int]:
        return {**self._driver.report(), "capacity_retries": self._capacity_retries}
