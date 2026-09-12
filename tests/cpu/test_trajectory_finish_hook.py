import asyncio
import threading
from types import SimpleNamespace

from qwen3_runtime.integrations.skyrl.inference_engine import Qwen3InferenceEngine
from tests.cpu.test_inference_engine_sessions import FakeEngine


def test_finish_does_not_block_the_async_loop_while_engine_is_busy(monkeypatch):
    adapter = Qwen3InferenceEngine(FakeEngine())
    gate = threading.Event()

    def busy_owner(fn):
        # A finite wait makes the old synchronous implementation fail cleanly.
        gate.wait(timeout=0.5)
        return fn()

    monkeypatch.setattr(adapter._driver, "run_on_engine", busy_owner)

    async def run():
        task = asyncio.create_task(adapter.finish_session([1, 2, 3]))
        try:
            await asyncio.sleep(0)
            assert not task.done(), "finish blocked the event loop until the owner returned"
        finally:
            gate.set()
            await task

    asyncio.run(run())


def test_codescout_result_notifies_the_real_session_owner(monkeypatch):
    from qwen3_runtime.integrations.skyrl.session_lifecycle import wrap_trajectory_finish

    monkeypatch.setenv("QWEN3_SESSION_KV", "1")
    engine = FakeEngine()
    adapter = Qwen3InferenceEngine(engine)
    generator = SimpleNamespace(inference_engine_client=SimpleNamespace(engines=[adapter]))
    result = [[([900, 901], 1.25, "complete", [1, 1], [1, 2], None, {})], {}, {}]

    async def rollout(self):
        await adapter._run_turn([1, 2], max_tokens=32, sampling=None)
        return result

    async def run():
        try:
            actual = await wrap_trajectory_finish(rollout)(generator)
            assert actual is result
            assert actual[0][0][1] == 1.25
            assert adapter.session_report()["live_sessions"] == 0
            assert actual[2]["qwen3/session_released"] == 1
        finally:
            adapter._driver.stop()

    asyncio.run(run())


def test_finish_notification_failure_does_not_change_rollout():
    from qwen3_runtime.integrations.skyrl.session_lifecycle import wrap_trajectory_finish

    class Unavailable:
        async def finish_session(self, tokens):
            raise RuntimeError("actor unavailable")

    generator = SimpleNamespace(inference_engine_client=SimpleNamespace(engines=[Unavailable()]))
    result = [[([3], 0.5, "complete", [1], [1, 2], None, {})], {}, {}]

    async def rollout(self):
        return result

    assert asyncio.run(wrap_trajectory_finish(rollout)(generator)) is result
    assert result[2]["qwen3/session_release_errors"] == 1


def test_existing_codescout_patch_wires_finish_once(monkeypatch):
    import sys
    from qwen3_runtime.integrations.skyrl.inject import patch_codescout_rollout_concurrency

    calls = []

    class Owner:
        async def finish_session(self, ids):
            calls.append(ids)
            return 1

    class Generator:
        inference_engine_client = SimpleNamespace(engines=[Owner()])

        async def code_search_loop(self):
            return [[([3], 1.0, "complete", [1], [1, 2], None, {})], {}, {}]

    monkeypatch.setitem(sys.modules, "src.generator.code_search_generator",
                        SimpleNamespace(CodeSearchGenerator=Generator))
    monkeypatch.setenv("QWEN3_FINISH_SESSIONS", "1")
    assert patch_codescout_rollout_concurrency(2) == 2
    assert patch_codescout_rollout_concurrency(2) == 2
    asyncio.run(Generator().code_search_loop())
    assert calls == [[1, 2, 3]]


def test_ray_wrapper_forwards_trajectory_finish_to_actor(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    class SkyWrapper:
        def __init__(self, actor):
            self.inference_engine_actor = actor

    calls = []

    async def remote(ids):
        calls.append(ids)
        return 1

    monkeypatch.setitem(sys.modules, "skyrl_train.inference_engines.ray_wrapped_inference_engine",
                        SimpleNamespace(RayWrappedInferenceEngine=SkyWrapper))
    path = Path(__file__).parents[2] / "qwen3_runtime/integrations/skyrl/ray_wrapped.py"
    spec = importlib.util.spec_from_file_location("_qwen3_wrapper_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper = module.Qwen3RayWrappedInferenceEngine(SimpleNamespace(finish_session=SimpleNamespace(remote=remote)))
    assert asyncio.run(wrapper.finish_session([1, 2, 3])) == 1
    assert calls == [[1, 2, 3]]
