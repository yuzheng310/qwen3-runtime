"""What reaches a Ray worker's environment.

This selection had no test, and it silently dropped four variables over the
course of the rollout work. The failure is invisible: nothing raises, the actor
simply runs with the feature off, and the run looks legitimate until someone
compares counters that should not be zero. These tests exist so the next
QWEN3_ variable is covered by construction rather than by remembering.
"""

from __future__ import annotations

from qwen3_runtime.integrations.skyrl.inject import collect_worker_env


def test_any_qwen3_variable_travels_without_being_listed():
    picked = collect_worker_env({"QWEN3_A_SETTING_INVENTED_JUST_NOW": "1"})
    assert picked["QWEN3_A_SETTING_INVENTED_JUST_NOW"] == "1"


def test_the_variables_that_were_dropped_in_practice_are_covered():
    # Each of these reached the actor only after a run had already been spent
    # measuring a feature that was not actually on.
    env = {
        "QWEN3_SESSION_KV": "1",
        "QWEN3_STATS_FILE": "/tmp/stats.json",
        "QWEN3_BATCHING": "1",
        "QWEN3_SPEC_TOKENS": "16",
    }
    assert collect_worker_env(env) == env


def test_foreign_names_travel_only_when_listed():
    picked = collect_worker_env({"PATH": "/usr/bin", "SOME_OTHER_TOOL_HOME": "/opt/x"})
    assert picked == {"PATH": "/usr/bin"}


def test_empty_values_are_not_propagated():
    # An empty string is how a shell spells "unset" here, and forwarding it
    # would override a default the worker would otherwise pick up.
    assert collect_worker_env({"QWEN3_SESSION_KV": "", "PATH": ""}) == {}


def test_selection_does_not_mutate_the_environment_it_reads():
    env = {"QWEN3_SESSION_KV": "1", "PATH": "/usr/bin"}
    before = dict(env)
    collect_worker_env(env)
    assert env == before
