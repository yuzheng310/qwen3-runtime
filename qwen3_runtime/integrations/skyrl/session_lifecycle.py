"""Explicit trajectory completion for the pinned CodeScout/SkyRL integration."""

import asyncio
from functools import wraps


def wrap_trajectory_finish(fn):
    """Notify after the agent really stopped, including max-turn termination.

    No finish-tool heuristic: a generated tool call can still be rejected by
    the agent. The completed rollout's final token history is authoritative.
    A release miss only loses a cache optimization; it never changes samples.
    """
    @wraps(fn)
    async def wrapped(self, *args, **kwargs):
        result = await fn(self, *args, **kwargs)
        rows, _rewards, metrics = result
        if not rows or rows[-1][2] == "error":
            return result
        last = rows[-1]
        history = list(last[4]) + list(last[0])
        client = getattr(self, "inference_engine_client", None)
        engines = getattr(client, "engines", ())
        owners = [e for e in engines if callable(getattr(e, "finish_session", None))]
        if not owners:
            metrics["qwen3/session_release_unsupported"] = 1
            return result

        async def notify(owner):
            return await asyncio.wait_for(owner.finish_session(history), timeout=5.0)

        outcomes = await asyncio.gather(*(notify(e) for e in owners), return_exceptions=True)
        metrics["qwen3/session_released"] = sum(
            int(n) for n in outcomes if not isinstance(n, BaseException)
        )
        metrics["qwen3/session_release_errors"] = sum(
            isinstance(n, BaseException) for n in outcomes
        )
        metrics["qwen3/session_release_missed"] = sum(n == 0 for n in outcomes)
        return result

    wrapped._qwen3_finishes_sessions = True
    return wrapped
