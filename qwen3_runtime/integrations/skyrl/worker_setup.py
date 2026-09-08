"""Ray worker hook: CUDA peaks, FSDP wrap-name coerce, cgroup freeze guards."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import is_dataclass, replace
from typing import Any, Callable

_WATCHDOG_STARTED = False
_ABORTING = False

# Patterns that are the GRPO job, not jupyter/sshd/tensorboard.
_JOB_PKILL_PATTERNS = (
    "step52_one_grpo_step",
    "FSDPPolicyWorkerBase",
    "init_and_run",
    "AsyncVLLM",
)
_RAY_PKILL_PATTERNS = (
    "gcs_server",
    "raylet",
)


def parse_cgroup_events(text: str) -> dict[str, int]:
    """Parse ``memory.events`` (``key value`` lines) into a dict."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return out


def read_cgroup(root: str = "/sys/fs/cgroup") -> dict[str, Any] | None:
    """Return current/high/max bytes and event counters, or None if missing."""
    try:
        current = int(open(os.path.join(root, "memory.current")).read())
        high_raw = open(os.path.join(root, "memory.high")).read().strip()
        max_raw = open(os.path.join(root, "memory.max")).read().strip()
        events = parse_cgroup_events(open(os.path.join(root, "memory.events")).read())
    except OSError:
        return None
    high = None if high_raw == "max" else int(high_raw)
    maximum = None if max_raw == "max" else int(max_raw)
    return {
        "current": current,
        "high": high,
        "max": maximum,
        "events": events,
    }


def cgroup_should_abort(
    *,
    current: int,
    high: int | None,
    high_events: int,
    prev_high_events: int | None,
    over_streak: int,
    high_event_delta_limit: int,
    over_high_polls: int,
) -> tuple[bool, int, str]:
    """Decide whether the job must die to protect sshd.

    AutoDL ``memory.high`` is read-only and ``swap.max=0``. Crossing high
    does not OOM-kill; the kernel throttles the whole cgroup (sshd included).
    Abort on a reclaim storm, or if we sit over high for several polls.
    """
    if high is not None and high > 0 and current > high:
        over_streak += 1
    else:
        over_streak = 0
    # memory.events.high is a cgroup-lifetime counter. A new job must baseline
    # it (prev=None) or a leftover 20k from the previous abort kills the next
    # launch instantly while current is a few GiB.
    if prev_high_events is None:
        delta = 0
    else:
        delta = high_events - prev_high_events
        if delta < 0:
            delta = 0
    if delta >= high_event_delta_limit:
        return True, over_streak, f"memory.events.high delta {delta} >= {high_event_delta_limit}"
    if over_streak >= over_high_polls:
        return True, over_streak, f"memory.current {current} > memory.high {high} for {over_streak} polls"
    return False, over_streak, ""


def unpin_offload_policy(policy: Any) -> Any:
    """FSDP2 ``CPUOffloadPolicy(pin_memory=True)`` mlocks host pages.

    PyTorch: pinned offload 'cannot be used by other processes'. On AutoDL
    that is sshd. Unpinning is the documented escape for insufficient CPU RAM.
    """
    if policy is None or not getattr(policy, "pin_memory", False):
        return policy
    if is_dataclass(policy):
        return replace(policy, pin_memory=False)
    cls = type(policy)
    try:
        return cls(pin_memory=False)
    except TypeError:
        return policy


def unpin_fsdp_kwargs(fsdp_kwargs: dict | None) -> dict | None:
    if not fsdp_kwargs:
        return fsdp_kwargs
    policy = fsdp_kwargs.get("offload_policy")
    unpinned = unpin_offload_policy(policy)
    if unpinned is policy:
        return fsdp_kwargs
    patched = dict(fsdp_kwargs)
    patched["offload_policy"] = unpinned
    return patched


def _coerce_fsdp2_wrap_names_to_list() -> None:
    """transformers 5 may store ``_no_split_modules`` as a set; SkyRL indexes [0]."""
    try:
        import skyrl_train.distributed.fsdp_utils as fsdp_utils
    except ImportError:
        return
    orig = fsdp_utils.apply_fsdp2
    if getattr(orig, "_qwen3_list_wrap", False):
        return

    def apply_fsdp2(model, fsdp_kwargs, config):
        names = getattr(model, "_no_split_modules", None)
        if isinstance(names, set):
            model._no_split_modules = list(names)
        try:
            wrap = config.get("wrap_policy") if config is not None else None
            if wrap is not None:
                cls = wrap.get("transformer_layer_cls_to_wrap")
                if isinstance(cls, set):
                    wrap["transformer_layer_cls_to_wrap"] = list(cls)
        except Exception:
            pass
        if os.environ.get("QWEN3_FSDP_PIN_MEMORY", "0") != "1":
            fsdp_kwargs = unpin_fsdp_kwargs(fsdp_kwargs)
        return orig(model, fsdp_kwargs, config)

    apply_fsdp2._qwen3_list_wrap = True  # type: ignore[attr-defined]
    fsdp_utils.apply_fsdp2 = apply_fsdp2
    try:
        import skyrl_train.distributed.fsdp_strategy as fsdp_strategy

        fsdp_strategy.apply_fsdp2 = apply_fsdp2
    except ImportError:
        pass


def apply_skyrl_runtime_patches() -> None:
    _coerce_fsdp2_wrap_names_to_list()


def _write_heartbeat(path: str, rec: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as handle:
            json.dump(rec, handle)
        os.replace(tmp, path)
    except OSError:
        pass


def abort_training_job(reason: str, *, pkill: Callable[..., Any] | None = None) -> None:
    """Kill the GRPO job tree. Do not touch sshd/jupyter."""
    global _ABORTING
    if _ABORTING:
        return
    _ABORTING = True
    print(f"[qwen3] cgroup watchdog abort: {reason}", flush=True)
    runner = pkill if pkill is not None else subprocess.call
    for pattern in _JOB_PKILL_PATTERNS + _RAY_PKILL_PATTERNS:
        try:
            runner(["pkill", "-9", "-f", pattern])
        except Exception:
            continue
    for pid_path in ("/tmp/step52-a5.pid", "/tmp/step52-b.pid"):
        try:
            pid = int(open(pid_path).read().strip())
        except (OSError, ValueError):
            continue
        if pid <= 1 or pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.kill(os.getpid(), signal.SIGKILL)
    except OSError:
        os._exit(137)


def start_cgroup_watchdog(
    *,
    cgroup_root: str = "/sys/fs/cgroup",
    heartbeat_path: str | None = None,
    interval_s: float | None = None,
    high_event_delta_limit: int | None = None,
    over_high_polls: int | None = None,
    abort: Callable[[str], None] | None = None,
) -> bool:
    """Driver-side thread: heartbeat + abort if memory.high starts throttling."""
    global _WATCHDOG_STARTED
    if os.environ.get("QWEN3_CGROUP_WATCHDOG", "0") != "1":
        return False
    if _WATCHDOG_STARTED:
        return True
    _WATCHDOG_STARTED = True
    path = heartbeat_path or os.environ.get("QWEN3_CGROUP_HEARTBEAT", "/tmp/qwen3-cgroup.json")
    interval = float(interval_s if interval_s is not None else os.environ.get("QWEN3_CGROUP_POLL_S", "2"))
    delta_limit = int(
        high_event_delta_limit
        if high_event_delta_limit is not None
        else os.environ.get("QWEN3_CGROUP_HIGH_EVENT_KILL", "20000")
    )
    polls = int(
        over_high_polls
        if over_high_polls is not None
        else os.environ.get("QWEN3_CGROUP_OVER_HIGH_POLLS", "3")
    )
    kill = abort or abort_training_job

    def _loop() -> None:
        prev_high: int | None = None
        over_streak = 0
        while True:
            snap = read_cgroup(cgroup_root)
            rec: dict[str, Any] = {"pid": os.getpid(), "ts": time.time()}
            if snap is None:
                rec["error"] = "cgroup unreadable"
                _write_heartbeat(path, rec)
                time.sleep(interval)
                continue
            events = snap["events"]
            high_events = int(events.get("high", 0))
            abort_now, over_streak, reason = cgroup_should_abort(
                current=int(snap["current"]),
                high=snap["high"],
                high_events=high_events,
                prev_high_events=prev_high,
                over_streak=over_streak,
                high_event_delta_limit=delta_limit,
                over_high_polls=polls,
            )
            rec.update(
                {
                    "current": snap["current"],
                    "high": snap["high"],
                    "max": snap["max"],
                    "high_events": high_events,
                    "over_streak": over_streak,
                }
            )
            _write_heartbeat(path, rec)
            if abort_now:
                rec["abort"] = reason
                _write_heartbeat(path, rec)
                kill(reason)
                return
            prev_high = high_events
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True, name="qwen3-cgroup-watchdog").start()
    return True


def _nice_worker() -> None:
    """Leave CPU for sshd. Does not help a memory.high D-state stall."""
    try:
        os.nice(10)
    except OSError:
        pass


def init() -> None:
    apply_skyrl_runtime_patches()
    _nice_worker()
    path = os.environ.get("QWEN3_TRAIN_PEAK_FILE")
    if not path:
        return

    def _loop() -> None:
        best = -1
        while True:
            time.sleep(2.0)
            try:
                import torch

                if not torch.cuda.is_available():
                    continue
                allocated = int(torch.cuda.memory_allocated())
                peak = int(torch.cuda.max_memory_allocated())
                if peak <= best:
                    continue
                best = peak
                rec = {
                    "pid": os.getpid(),
                    "allocated": allocated,
                    "max_allocated": peak,
                    "reserved": int(torch.cuda.memory_reserved()),
                }
                with open(f"{path}.{os.getpid()}", "w") as handle:
                    json.dump(rec, handle)
            except Exception:
                continue

    threading.Thread(target=_loop, daemon=True, name="qwen3-cuda-peak").start()
