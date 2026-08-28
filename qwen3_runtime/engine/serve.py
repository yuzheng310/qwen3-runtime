"""Timed drain / Poisson serving. Tokenizer stays outside; this speaks token ids."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import time

import torch

from qwen3_runtime.engine.engine import Engine


def _sync(engine: Engine) -> None:
    runner = getattr(engine, "runner", None)
    model = getattr(runner, "model", None)
    if model is None:
        return
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type == "cuda":
        torch.cuda.synchronize()


def _maybe_sync(engine: Engine, profile: bool) -> None:
    if profile:
        _sync(engine)


def _nvtx_range(name: str):
    class _Nvtx:
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
            return self

        def __exit__(self, *exc):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            return False

    return _Nvtx()


@dataclass
class RequestTrace:
    request_id: int
    arrival_s: float
    prompt_len: int
    max_tokens: int
    tokens: list[int] = field(default_factory=list)
    first_token_s: float | None = None
    first_scheduled_s: float | None = None
    itl_s: list[float] = field(default_factory=list)
    last_token_s: float | None = None
    admitted: bool = True
    turn_id: int | None = None
    cached_tokens: int = 0

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.arrival_s

    @property
    def queue_s(self) -> float | None:
        """Time from arrival until the request is first scheduled onto a batch."""
        if self.first_scheduled_s is None:
            return None
        return self.first_scheduled_s - self.arrival_s

    @property
    def prefill_s(self) -> float | None:
        """Time from first schedule until first output token."""
        if self.first_token_s is None or self.first_scheduled_s is None:
            return None
        return self.first_token_s - self.first_scheduled_s

    @property
    def tpot_s(self) -> float | None:
        if not self.itl_s:
            return None
        return sum(self.itl_s) / len(self.itl_s)


@dataclass(frozen=True)
class AdmissionView:
    """What an admit policy may see. Policy is a later candidate; this is the seam."""

    prompt_len: int
    max_tokens: int
    arrival_s: float
    waiting: int
    running: int
    offered: int
    admitted: int
    paused: int = 0
    held_slot: bool = False

    @property
    def occupancy(self) -> int:
        return self.waiting + self.running + self.paused


def always_admit(_view: AdmissionView) -> bool:
    return True


# Frozen A from ADR 0004. Drain and occupancy from Task-2 knees (westb, 070cdc5).
FROZEN_SLO_TTFT_S = 4.5
FROZEN_SLO_TPOT_S = 0.100
REQUEST_DRAIN_PER_S = 6.245
SESSION_MAX_OCCUPANCY = 6
_PREFILL_S_AT_9981 = 1.08
_PREFILL_REF_TOKENS = 9981
_MIN_PREFILL_S = 0.02


def _estimated_prefill_s(prompt_len: int) -> float:
    return max(_MIN_PREFILL_S, prompt_len * (_PREFILL_S_AT_9981 / _PREFILL_REF_TOKENS))


def slo_aware_admit(
    view: AdmissionView,
    *,
    slo_ttft_s: float,
    max_occupancy: int | None,
    drain_per_s: float | None,
) -> bool:
    """Hold p99 TTFT ≤ A by refusing work whose predicted wait already exceeds A.

    Resume turns (`held_slot`) keep the slot they already occupy. New arrivals
    are refused when occupancy is at the session knee, or when FCFS drain of
    `waiting` plus this request's prefill would miss A.
    """
    if view.held_slot:
        return True
    if max_occupancy is not None and view.occupancy >= max_occupancy:
        return False
    if drain_per_s is not None and drain_per_s > 0:
        predicted = view.waiting / drain_per_s + _estimated_prefill_s(view.prompt_len)
        if predicted > slo_ttft_s:
            return False
    return True


def request_slo_admit(view: AdmissionView) -> bool:
    return slo_aware_admit(
        view,
        slo_ttft_s=FROZEN_SLO_TTFT_S,
        max_occupancy=None,
        drain_per_s=REQUEST_DRAIN_PER_S,
    )


def session_slo_admit(view: AdmissionView) -> bool:
    return slo_aware_admit(
        view,
        slo_ttft_s=FROZEN_SLO_TTFT_S,
        max_occupancy=SESSION_MAX_OCCUPANCY,
        drain_per_s=None,
    )


AdmitFn = Callable[[AdmissionView], bool]


@dataclass(frozen=True)
class BacklogSample:
    t_s: float
    waiting: int
    running: int
    paused: int = 0
    kv_free_blocks: int | None = None

    @property
    def depth(self) -> int:
        return self.waiting + self.running


@dataclass
class ServeRun:
    traces: list[RequestTrace]
    offered: int
    admitted: int
    rejected: int
    completed: int
    slo_met: int | None
    backlog: list[BacklogSample]
    wall_s: float
    origin_s: float
    sync_policy: str
    num_preemptions: int = 0
    kv_exhausted: bool = False
    step_mix: dict | None = None

    @property
    def goodput_per_s(self) -> float | None:
        if self.slo_met is None or self.wall_s <= 0:
            return None
        return self.slo_met / self.wall_s

    @property
    def backlog_slope_per_s(self) -> float | None:
        return backlog_slope(self.backlog)


def backlog_slope(samples: Sequence[BacklogSample]) -> float | None:
    """Least-squares slope of queue+running depth versus time. None if underdetermined."""
    if len(samples) < 2:
        return None
    xs = [s.t_s for s in samples]
    ys = [s.depth for s in samples]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / var_x


def _record_token(trace: RequestTrace, tok: int, now: float) -> None:
    if trace.first_token_s is None:
        trace.first_token_s = now
    else:
        assert trace.last_token_s is not None
        trace.itl_s.append(now - trace.last_token_s)
    trace.last_token_s = now
    trace.tokens.append(tok)


def _record_emitted(engine: Engine, traces: dict[int, RequestTrace], now: float) -> None:
    """Record every token at its observed step-completion timestamp."""
    for rid, toks in engine.last_emitted.items():
        tr = traces.get(rid)
        if tr is None or not toks:
            continue
        for tok in toks:
            _record_token(tr, tok, now)


def _max_tokens_list(max_tokens: int | Sequence[int], n: int) -> list[int]:
    if isinstance(max_tokens, int):
        return [max_tokens] * n
    if len(max_tokens) != n:
        raise ValueError("max_tokens must be a scalar or align with prompts")
    return [int(x) for x in max_tokens]


def _meets_slo(
    trace: RequestTrace,
    *,
    slo_ttft_s: float | None,
    slo_tpot_s: float | None,
) -> bool:
    if not trace.admitted or len(trace.tokens) != trace.max_tokens:
        return False
    if slo_ttft_s is not None and (trace.ttft_s is None or trace.ttft_s > slo_ttft_s):
        return False
    if slo_tpot_s is not None:
        if trace.max_tokens <= 1:
            pass
        elif trace.tpot_s is None or trace.tpot_s > slo_tpot_s:
            return False
    return True


def _step(engine: Engine, *, profile: bool, nvtx: bool) -> list[tuple[int, int | None, bool]]:
    if nvtx:
        with _nvtx_range("engine.step"):
            _maybe_sync(engine, profile)
            rows = engine.step()
            _maybe_sync(engine, profile)
            return rows
    _maybe_sync(engine, profile)
    rows = engine.step()
    _maybe_sync(engine, profile)
    return rows


def _is_kv_exhaustion(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "exceeds the KV pool" in msg
        or "could not admit a batch after preemption" in msg
        or "KV blocks" in msg
    )


def _empty_step_mix() -> dict:
    return {
        "n_steps": 0,
        "n_decode_only": 0,
        "n_prefill_only": 0,
        "n_mixed": 0,
        "prefill_tokens_sum": 0,
        "decode_tokens_sum": 0,
        "prefill_reqs_sum": 0,
        "decode_reqs_sum": 0,
        "scheduled_reqs_sum": 0,
        "tokens_scheduled_sum": 0,
        "n_preempt_steps": 0,
        "preempts_in_schedule": 0,
    }


def _note_step(engine: Engine, traces_by_rid: dict[int, RequestTrace], step_t0: float, mix: dict) -> None:
    stats = getattr(engine, "last_step_stats", None)
    if not stats:
        return
    mix["n_steps"] += 1
    mix["prefill_tokens_sum"] += int(stats["prefill_tokens"])
    mix["decode_tokens_sum"] += int(stats["decode_tokens"])
    pf, dec = int(stats["prefill_reqs"]), int(stats["decode_reqs"])
    mix["prefill_reqs_sum"] += pf
    mix["decode_reqs_sum"] += dec
    mix["scheduled_reqs_sum"] += pf + dec
    mix["tokens_scheduled_sum"] += int(stats["prefill_tokens"]) + int(stats["decode_tokens"])
    if pf > 0 and dec > 0:
        mix["n_mixed"] += 1
    elif dec > 0:
        mix["n_decode_only"] += 1
    elif pf > 0:
        mix["n_prefill_only"] += 1
    if int(stats["preempts"]) > 0:
        mix["n_preempt_steps"] += 1
        mix["preempts_in_schedule"] += int(stats["preempts"])
    for rid in stats["req_ids"]:
        tr = traces_by_rid.get(rid)
        if tr is not None and tr.first_scheduled_s is None:
            tr.first_scheduled_s = step_t0


def _sample_backlog(engine: Engine, origin: float, backlog: list[BacklogSample]) -> None:
    bm = engine.scheduler.block_manager
    backlog.append(
        BacklogSample(
            t_s=time.perf_counter() - origin,
            waiting=len(engine.scheduler.waiting),
            running=len(engine.scheduler.running),
            paused=len(engine.scheduler.paused),
            kv_free_blocks=bm.num_free_blocks,
        )
    )


def _cached_tokens(engine: Engine, rid: int) -> int:
    req = engine._requests.get(rid)
    return 0 if req is None else int(req.cached_tokens)


def run_sequential_requests(
    engine: Engine,
    items: list[tuple[list[int], int]],
    *,
    ignore_eos: bool = True,
    nvtx: bool = True,
    profile: bool = False,
) -> list[RequestTrace]:
    """One request at a time with per-request max_tokens. KV is freed on finish.

    Used by CodeScout Trace Replay. Not a production scheduler change.
    Default serving path has no CUDA fences; pass profile=True for Nsight.
    """
    traces: list[RequestTrace] = []
    for prompt, max_tokens in items:
        _maybe_sync(engine, profile)
        t0 = time.perf_counter()
        rid = engine.add_request(prompt, max_tokens=max_tokens, ignore_eos=ignore_eos)
        trace = RequestTrace(
            request_id=rid,
            arrival_s=t0,
            prompt_len=len(prompt),
            max_tokens=max_tokens,
            cached_tokens=_cached_tokens(engine, rid),
        )
        while not engine.is_finished():
            _step(engine, profile=profile, nvtx=nvtx)
            now = time.perf_counter()
            _record_emitted(engine, {trace.request_id: trace}, now)
        traces.append(trace)
    return traces


def run_closed_batch(
    engine: Engine,
    prompts: list[list[int]],
    max_tokens: int | Sequence[int],
    *,
    ignore_eos: bool = True,
    profile: bool = False,
) -> list[RequestTrace]:
    """Submit all prompts at t0 (throughput / latency / longctx / prefill / decode)."""
    lengths = _max_tokens_list(max_tokens, len(prompts))
    _maybe_sync(engine, profile)
    t0 = time.perf_counter()
    traces: dict[int, RequestTrace] = {}
    for prompt, n_out in zip(prompts, lengths):
        rid = engine.add_request(prompt, max_tokens=n_out, ignore_eos=ignore_eos)
        traces[rid] = RequestTrace(
            request_id=rid,
            arrival_s=t0,
            prompt_len=len(prompt),
            max_tokens=n_out,
            cached_tokens=_cached_tokens(engine, rid),
        )
    while not engine.is_finished():
        _step(engine, profile=profile, nvtx=True)
        now = time.perf_counter()
        _record_emitted(engine, traces, now)
    return [traces[rid] for rid in traces]


def run_poisson(
    engine: Engine,
    prompts: list[list[int]],
    max_tokens: int | Sequence[int],
    arrivals_s: list[float],
    *,
    ignore_eos: bool = True,
    profile: bool = False,
    admit: AdmitFn | None = None,
    slo_ttft_s: float | None = None,
    slo_tpot_s: float | None = None,
) -> ServeRun:
    """Online serving: add_request at Poisson arrival times, step when work exists.

    TTFT is from arrival, not admission. Default path does not CUDA-synchronize
    around step(); profile=True restores the Nsight fences.
    """
    if len(prompts) != len(arrivals_s):
        raise ValueError("arrivals_s must align with prompts")
    lengths = _max_tokens_list(max_tokens, len(prompts))
    admit_fn = admit or always_admit
    _maybe_sync(engine, profile)
    origin = time.perf_counter()
    traces: list[RequestTrace] = []
    rid_map: dict[int, RequestTrace] = {}
    backlog: list[BacklogSample] = []
    next_i = 0
    n = len(prompts)
    offered = 0
    admitted = 0
    rejected = 0
    reject_id = -1
    mix = _empty_step_mix()
    while next_i < n or not engine.is_finished():
        now = time.perf_counter()
        elapsed = now - origin
        while next_i < n and arrivals_s[next_i] <= elapsed:
            arrival = origin + arrivals_s[next_i]
            view = AdmissionView(
                prompt_len=len(prompts[next_i]),
                max_tokens=lengths[next_i],
                arrival_s=arrival,
                waiting=len(engine.scheduler.waiting),
                running=len(engine.scheduler.running),
                offered=offered,
                admitted=admitted,
                paused=len(engine.scheduler.paused),
                held_slot=False,
            )
            offered += 1
            if admit_fn(view):
                rid = engine.add_request(
                    prompts[next_i], max_tokens=lengths[next_i], ignore_eos=ignore_eos
                )
                trace = RequestTrace(
                    request_id=rid,
                    arrival_s=arrival,
                    prompt_len=len(prompts[next_i]),
                    max_tokens=lengths[next_i],
                    admitted=True,
                    cached_tokens=_cached_tokens(engine, rid),
                )
                rid_map[rid] = trace
                traces.append(trace)
                admitted += 1
            else:
                traces.append(
                    RequestTrace(
                        request_id=reject_id,
                        arrival_s=arrival,
                        prompt_len=len(prompts[next_i]),
                        max_tokens=lengths[next_i],
                        admitted=False,
                    )
                )
                reject_id -= 1
                rejected += 1
            next_i += 1
        _sample_backlog(engine, origin, backlog)
        if engine.scheduler.waiting or engine.scheduler.running:
            t0 = time.perf_counter()
            _step(engine, profile=profile, nvtx=True)
            _note_step(engine, rid_map, t0, mix)
            step_t = time.perf_counter()
            _record_emitted(engine, rid_map, step_t)
            continue
        if next_i >= n:
            break
        sleep_for = (origin + arrivals_s[next_i]) - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
    wall_s = time.perf_counter() - origin
    completed = sum(1 for t in traces if t.admitted and len(t.tokens) == t.max_tokens)
    slo_applied = slo_ttft_s is not None or slo_tpot_s is not None
    slo_met = (
        sum(
            1
            for t in traces
            if _meets_slo(t, slo_ttft_s=slo_ttft_s, slo_tpot_s=slo_tpot_s)
        )
        if slo_applied
        else None
    )
    return ServeRun(
        traces=traces,
        offered=offered,
        admitted=admitted,
        rejected=rejected,
        completed=completed,
        slo_met=slo_met,
        backlog=backlog,
        wall_s=wall_s,
        origin_s=origin,
        sync_policy="profile" if profile else "serving",
        num_preemptions=engine.scheduler.num_preemptions,
        step_mix=mix,
    )


@dataclass
class SessionTurn:
    prompt: list[int]
    max_tokens: int
    think_s: float = 0.0


def run_sessions(
    engine: Engine,
    sessions: list[list[SessionTurn]],
    *,
    profile: bool = False,
    admit: AdmitFn | None = None,
    slo_ttft_s: float | None = None,
    slo_tpot_s: float | None = None,
    max_active: int | None = None,
) -> ServeRun:
    """N concurrent multi-turn sessions. KV stays allocated during think time.

    Later turns pass only the new suffix in `SessionTurn.prompt`. Prefix KV is
    not recomputed. This is the session-capacity load model, not a prefix-cache
    feature flag.
    """
    if not sessions:
        raise ValueError("sessions must be non-empty")
    for sess in sessions:
        if not sess:
            raise ValueError("each session must have at least one turn")
    admit_fn = admit or always_admit
    n_active_cap = len(sessions) if max_active is None else max_active
    _maybe_sync(engine, profile)
    origin = time.perf_counter()
    traces: list[RequestTrace] = []
    rid_trace: dict[int, RequestTrace] = {}
    backlog: list[BacklogSample] = []
    offered = 0
    admitted = 0
    rejected = 0
    reject_id = -1
    mix = _empty_step_mix()

    class _State:
        def __init__(self, turns: list[SessionTurn]):
            self.turns = turns
            self.idx = 0
            self.rid: int | None = None
            self.wake_s: float | None = None
            self.done = False

    states = [_State(s) for s in sessions]
    next_start = 0

    def _in_flight() -> int:
        return sum(1 for s in states if (s.rid is not None or s.wake_s is not None) and not s.done)

    def _start_turn(st: _State, turn: SessionTurn, now: float, *, resume: bool) -> None:
        nonlocal offered, admitted, rejected, reject_id
        view = AdmissionView(
            prompt_len=len(turn.prompt),
            max_tokens=turn.max_tokens,
            arrival_s=now,
            waiting=len(engine.scheduler.waiting),
            running=len(engine.scheduler.running),
            offered=offered,
            admitted=admitted,
            paused=len(engine.scheduler.paused),
            held_slot=resume,
        )
        offered += 1
        more = st.idx + 1 < len(st.turns)
        if not admit_fn(view):
            traces.append(
                RequestTrace(
                    request_id=reject_id,
                    arrival_s=now,
                    prompt_len=len(turn.prompt),
                    max_tokens=turn.max_tokens,
                    admitted=False,
                    turn_id=st.idx,
                )
            )
            reject_id -= 1
            rejected += 1
            st.done = True
            st.rid = None
            st.wake_s = None
            return
        if resume:
            assert st.rid is not None
            engine.resume_request(st.rid, turn.prompt, turn.max_tokens, hold_kv=more)
            rid = st.rid
        else:
            rid = engine.add_request(
                turn.prompt, max_tokens=turn.max_tokens, ignore_eos=True, hold_kv=more
            )
            st.rid = rid
        trace = RequestTrace(
            request_id=rid,
            arrival_s=now,
            prompt_len=len(turn.prompt),
            max_tokens=turn.max_tokens,
            admitted=True,
            turn_id=st.idx,
            cached_tokens=_cached_tokens(engine, rid),
        )
        rid_trace[rid] = trace
        traces.append(trace)
        admitted += 1
        st.wake_s = None

    kv_exhausted = False
    try:
        while any(not s.done for s in states) or not engine.scheduler.is_finished() or engine.scheduler.paused:
            now = time.perf_counter()
            while next_start < len(states) and _in_flight() < n_active_cap:
                st = states[next_start]
                next_start += 1
                _start_turn(st, st.turns[0], now, resume=False)
            for st in states:
                if st.done or st.wake_s is None or now < st.wake_s:
                    continue
                st.idx += 1
                _start_turn(st, st.turns[st.idx], now, resume=True)
            _sample_backlog(engine, origin, backlog)
            if engine.scheduler.waiting or engine.scheduler.running:
                t0 = time.perf_counter()
                rows = _step(engine, profile=profile, nvtx=True)
                _note_step(engine, rid_trace, t0, mix)
                step_t = time.perf_counter()
                finished_rids = set()
                _record_emitted(engine, rid_trace, step_t)
                for rid, tok, done in rows:
                    if done:
                        finished_rids.add(rid)
                for st in states:
                    if st.rid in finished_rids and not st.done:
                        last = st.idx == len(st.turns) - 1
                        if last:
                            st.done = True
                            st.rid = None
                        else:
                            think = st.turns[st.idx].think_s
                            st.wake_s = step_t + think
                continue
            wakes = [st.wake_s for st in states if st.wake_s is not None and not st.done]
            if not wakes and next_start >= len(states) and all(s.done or s.rid is None for s in states):
                break
            if wakes:
                sleep_for = min(wakes) - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue
            break
    except RuntimeError as exc:
        if not _is_kv_exhaustion(exc):
            raise
        kv_exhausted = True
        for st in states:
            st.done = True
    wall_s = time.perf_counter() - origin
    completed = sum(1 for t in traces if t.admitted and len(t.tokens) == t.max_tokens)
    slo_applied = slo_ttft_s is not None or slo_tpot_s is not None
    slo_met = (
        sum(1 for t in traces if _meets_slo(t, slo_ttft_s=slo_ttft_s, slo_tpot_s=slo_tpot_s))
        if slo_applied
        else None
    )
    return ServeRun(
        traces=traces,
        offered=offered,
        admitted=admitted,
        rejected=rejected,
        completed=completed,
        slo_met=slo_met,
        backlog=backlog,
        wall_s=wall_s,
        origin_s=origin,
        sync_policy="profile" if profile else "serving",
        num_preemptions=engine.scheduler.num_preemptions,
        kv_exhausted=kv_exhausted,
        step_mix=mix,
    )
