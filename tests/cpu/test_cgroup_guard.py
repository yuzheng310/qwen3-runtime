"""Anti-freeze guards: FSDP2 unpin + cgroup memory.high abort."""

from dataclasses import dataclass

from qwen3_runtime.integrations.skyrl.worker_setup import (
    cgroup_should_abort,
    parse_cgroup_events,
    unpin_fsdp_kwargs,
    unpin_offload_policy,
)


def test_parse_cgroup_events():
    text = "low 0\nhigh 27700000\nmax 0\noom 0\noom_kill 0\n"
    assert parse_cgroup_events(text)["high"] == 27700000
    assert parse_cgroup_events(text)["oom_kill"] == 0


def test_abort_on_high_event_storm():
    abort, streak, reason = cgroup_should_abort(
        current=80 * 1024**3,
        high=88 * 1024**3,
        high_events=25000,
        prev_high_events=1000,
        over_streak=0,
        high_event_delta_limit=20000,
        over_high_polls=3,
    )
    assert abort is True
    assert "delta" in reason
    assert streak == 0


def test_abort_after_sitting_over_high():
    abort, streak, reason = cgroup_should_abort(
        current=89 * 1024**3,
        high=88 * 1024**3,
        high_events=10,
        prev_high_events=8,
        over_streak=2,
        high_event_delta_limit=20000,
        over_high_polls=3,
    )
    assert abort is True
    assert streak == 3
    assert "memory.current" in reason


def test_first_sample_baselines_lifetime_high_events():
    abort, streak, _ = cgroup_should_abort(
        current=4 * 1024**3,
        high=68 * 1024**3,
        high_events=29289,
        prev_high_events=None,
        over_streak=0,
        high_event_delta_limit=20000,
        over_high_polls=3,
    )
    assert abort is False
    assert streak == 0


def test_no_abort_when_under_high_and_quiet():
    abort, streak, _ = cgroup_should_abort(
        current=70 * 1024**3,
        high=88 * 1024**3,
        high_events=12,
        prev_high_events=10,
        over_streak=2,
        high_event_delta_limit=20000,
        over_high_polls=3,
    )
    assert abort is False
    assert streak == 0


@dataclass
class _Policy:
    pin_memory: bool = True


def test_unpin_offload_policy_clears_pin():
    assert unpin_offload_policy(_Policy(pin_memory=True)).pin_memory is False
    keep = _Policy(pin_memory=False)
    assert unpin_offload_policy(keep) is keep
    assert unpin_offload_policy(None) is None


def test_unpin_fsdp_kwargs_rewrites_policy():
    kwargs = {"mesh": "x", "offload_policy": _Policy(pin_memory=True)}
    out = unpin_fsdp_kwargs(kwargs)
    assert out is not kwargs
    assert out["offload_policy"].pin_memory is False
    assert out["mesh"] == "x"
    assert kwargs["offload_policy"].pin_memory is True
