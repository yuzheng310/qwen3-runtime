"""The actual SkyRL actor entry point must forward opt-in offload settings."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_runtime.engine import factory
from qwen3_runtime.integrations.skyrl import inference_engine


def actor_class(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(remote=lambda cls: cls, get_gpu_ids=lambda: []),
    )
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        ),
    )
    monkeypatch.setattr(
        inference_engine, "Qwen3InferenceEngine", lambda engine, **kw: engine
    )
    path = Path(inference_engine.__file__).with_name("ray_actor.py")
    spec = importlib.util.spec_from_file_location("_offload_actor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3RayActor


@pytest.mark.parametrize("explicit", [False, True])
def test_actor_forwards_offload_environment_and_explicit_overrides(
    monkeypatch, explicit
):
    values = {
        "QWEN3_SESSION_CPU_OFFLOAD": "sync",
        "QWEN3_CPU_KV_MAX_BYTES": "1073741824",
        "QWEN3_CPU_KV_PINNED_MAX_BYTES": "536870912",
        "QWEN3_KV_TRANSFER_CHUNK_BYTES": "33554432",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            config_report=lambda: {}, block_manager=SimpleNamespace()
        )

    monkeypatch.setattr(factory, "build_engine", build)
    cls = actor_class(monkeypatch)
    overrides = (
        {
            "session_cpu_offload": "off",
            "cpu_kv_max_bytes": 0,
            "cpu_kv_pinned_max_bytes": 0,
            "transfer_chunk_bytes": 1024,
        }
        if explicit
        else {}
    )
    cls("unused-model", **overrides)
    expected = overrides or {
        "session_cpu_offload": "sync",
        "cpu_kv_max_bytes": 1073741824,
        "cpu_kv_pinned_max_bytes": 536870912,
        "transfer_chunk_bytes": 33554432,
    }
    for key, value in expected.items():
        assert seen.get(key) == value


def test_actor_keeps_offload_disabled_by_default(monkeypatch):
    for key in (
        "QWEN3_SESSION_CPU_OFFLOAD",
        "QWEN3_CPU_KV_MAX_BYTES",
        "QWEN3_CPU_KV_PINNED_MAX_BYTES",
        "QWEN3_KV_TRANSFER_CHUNK_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            config_report=lambda: {}, block_manager=SimpleNamespace()
        )

    monkeypatch.setattr(factory, "build_engine", build)
    actor_class(monkeypatch)("unused-model")
    assert seen.get("session_cpu_offload") == "off"
    assert seen.get("cpu_kv_max_bytes") == 0
    assert seen.get("cpu_kv_pinned_max_bytes") == 0
