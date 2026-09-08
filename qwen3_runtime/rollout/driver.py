"""Drive one Engine from one thread so concurrent turns share a batch.

``Engine.drain_request`` steps until *one* request stops, which is the right
shape for a caller that has a single request and the wrong shape for an agent
loop. Under SkyRL several trajectories are in flight at once, but the adapter
declares ``async def`` and then calls into a synchronous engine, so the event
loop is held for the whole turn and the others simply queue. The engine can
batch -- ``step()`` schedules everything runnable -- but nothing ever gives it
more than one request to schedule.

That matters more than it sounds. Decode is bound by memory bandwidth, so a
batch of one reads every byte of the weights to produce a single token, and the
same read would have served eight. On a measured CodeScout step decode is 84%
of engine time at 11.19 ms/token.

So: one thread owns the engine and loops on ``step()``, callers hand it work and
await a future that the thread resolves. Scheduler mutations are queued onto the
same thread rather than taken under a lock, because ``step()`` walks the
scheduler queues and a mutation halfway through one is not a race anybody wants
to debug.

This is the design the HTTP adapter in ``scripts/codescout_openai_server.py``
has been running behind the benchmarks; that copy is left alone rather than
refactored underneath results that were measured on it.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from qwen3_runtime.engine.request import RequestStatus


@dataclass
class Turn:
    """What one request produced once it paused or finished.

    ``len(tokens) == len(logprobs)`` always. Both lists are extended from the
    same engine step, so a turn either carries the logprob every token was
    sampled under or it does not exist. Under speculative decoding a step
    emits several tokens at once, and the previous shape of this -- one
    ``last_logprob`` appended per step -- left most positions uncovered, which
    the OpenAI adapter then filled with zeros.
    """

    request_id: int
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    on_done: Callable[[int, list[int]], None] | None
    turn: Turn | None = None

    def resolve(self, value: Any) -> None:
        """Hand the result back to the caller's loop, not to this thread."""
        if self.future.done():
            return
        if isinstance(value, BaseException):
            self.loop.call_soon_threadsafe(self._set_exception, value)
        else:
            self.loop.call_soon_threadsafe(self._set_result, value)

    def _set_result(self, value: Any) -> None:
        if not self.future.done():
            self.future.set_result(value)

    def _set_exception(self, exc: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(exc)


class EngineDriver:
    """Owns the engine thread. Start it for a generation phase, stop it after.

    The thread only exists while there is generating to do. Weight sync, sleep
    and wake stay on the caller's thread and simply require the driver to be
    stopped, which removes any question of a NCCL collective or an IPC copy
    landing while ``step()`` is halfway through a batch.
    """

    def __init__(self, engine: Any, *, idle_poll_s: float = 0.02) -> None:
        self.engine = engine
        self.idle_poll_s = idle_poll_s
        self._cond = threading.Condition()
        self._pending: list[Callable[[], None]] = []
        self._waiters: dict[int, _Waiter] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = {"steps": 0, "batched_requests": 0, "max_batch": 0}

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        with self._cond:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="qwen3-engine-driver", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        with self._cond:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
            self._cond.notify_all()
        thread.join(timeout=timeout)
        with self._cond:
            self._thread = None
            stranded = list(self._waiters.values())
            self._waiters.clear()
        for waiter in stranded:
            waiter.resolve(RuntimeError("engine driver stopped before the turn finished"))

    # -- submitting work ---------------------------------------------------

    async def run_turn(
        self,
        start: Callable[[], int],
        *,
        on_done: Callable[[int, list[int]], None] | None = None,
    ) -> Turn:
        """Await one turn.

        ``start`` registers the request and returns its id; it runs on the
        engine thread, so whatever it consults -- a session registry, the block
        manager -- it sees a scheduler that nothing else is touching.
        ``on_done`` runs there too, once the request stops, for the bookkeeping
        that has to happen before another turn is admitted.
        """
        self.start()
        loop = asyncio.get_running_loop()
        waiter = _Waiter(loop=loop, future=loop.create_future(), on_done=on_done)
        with self._cond:
            self._pending.append(lambda: self._begin(waiter, start))
            self._cond.notify()
        return await waiter.future

    def run_on_engine(self, fn: Callable[[], Any]) -> Any:
        """Run something on the engine thread and wait for it.

        For callers that must touch the scheduler while the driver is up. If it
        is down, this is just a call.
        """
        if self._thread is None or threading.current_thread() is self._thread:
            return fn()
        box: dict[str, Any] = {}
        done = threading.Event()

        def wrapped() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # handed to the caller, not swallowed
                box["error"] = exc
            finally:
                done.set()

        with self._cond:
            self._pending.append(wrapped)
            self._cond.notify()
        if not done.wait(timeout=600):
            raise TimeoutError("engine thread did not pick up the work")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # -- the thread --------------------------------------------------------

    def _begin(self, waiter: _Waiter, start: Callable[[], int]) -> None:
        try:
            request_id = start()
        except BaseException as exc:
            waiter.resolve(exc)
            return
        waiter.turn = Turn(request_id=request_id)
        self._waiters[request_id] = waiter

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while self._pending:
                    self._pending.pop(0)()
                scheduler = self.engine.scheduler
                if not (scheduler.waiting or scheduler.running):
                    # Paused sessions are not work; they are just resident.
                    self._cond.wait(timeout=self.idle_poll_s)
                    continue
                in_flight = len(scheduler.running) + len(scheduler.waiting)
            try:
                self.engine.step()
                self._harvest(in_flight)
            except BaseException as exc:
                # Harvesting is inside the guard because it enforces the
                # token/logprob pairing. A turn that cannot honour that has to
                # fail its caller, not silently hand back a short list.
                self._fail_everyone(exc)
                continue

    def _fail_everyone(self, exc: BaseException) -> None:
        with self._cond:
            waiters = list(self._waiters.values())
            self._waiters.clear()
        for waiter in waiters:
            # Still registered, so its blocks are still spoken for. Leaving them
            # would shrink the pool with every failure, and the caller may well
            # retry into the space this frees.
            if waiter.turn is not None:
                try:
                    self.engine.finish_request(waiter.turn.request_id)
                except BaseException:
                    pass
            waiter.resolve(exc)

    def _harvest(self, in_flight: int) -> None:
        emitted = dict(self.engine.last_emitted)
        emitted_logprobs = dict(self.engine.last_emitted_logprobs)
        for request_id, tokens in emitted.items():
            n_lp = len(emitted_logprobs.get(request_id, ()))
            if n_lp != len(tokens):
                raise RuntimeError(
                    f"request {request_id} emitted {len(tokens)} tokens with {n_lp} logprobs"
                )
        self.stats["steps"] += 1
        self.stats["batched_requests"] += in_flight
        self.stats["max_batch"] = max(self.stats["max_batch"], in_flight)

        finished: list[_Waiter] = []
        with self._cond:
            for request_id, waiter in list(self._waiters.items()):
                turn = waiter.turn
                new_tokens = emitted.get(request_id, [])
                if turn is None:
                    continue
                turn.tokens.extend(new_tokens)
                turn.logprobs.extend(emitted_logprobs.get(request_id, ()))
                request = self.engine._requests.get(request_id)
                if request is None or request.status in (
                    RequestStatus.PAUSED,
                    RequestStatus.FINISHED,
                ):
                    self._waiters.pop(request_id, None)
                    finished.append(waiter)

        for waiter in finished:
            turn = waiter.turn
            assert turn is not None
            if waiter.on_done is not None:
                try:
                    waiter.on_done(turn.request_id, turn.tokens)
                except BaseException as exc:
                    waiter.resolve(exc)
                    continue
            waiter.resolve(turn)

    def report(self) -> dict[str, float | int]:
        steps = self.stats["steps"]
        return {
            **self.stats,
            "mean_batch": round(self.stats["batched_requests"] / max(1, steps), 2),
        }
