import asyncio
import inspect

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.integrations.skyrl.inference_engine import Qwen3InferenceEngine
from qwen3_runtime.integrations.skyrl.protocol import InferenceEngineInterface as LocalABC
from tests.cpu.test_session_kv import tiny_config
import torch


def test_wrapper_is_a_concrete_abc():
    try:
        from skyrl_train.inference_engines.base import InferenceEngineInterface as SkyABC
    except ImportError:
        SkyABC = LocalABC
    assert not inspect.isabstract(Qwen3InferenceEngine)
    assert issubclass(Qwen3InferenceEngine, SkyABC)


def test_wrapper_generate_and_abort_and_sleep():
    torch.manual_seed(3)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    wrapped = Qwen3InferenceEngine(Engine(cfg, PagedRunner(model)))

    async def _run():
        assert wrapped.tp_size() == wrapped.pp_size() == wrapped.dp_size() == 1
        out = await wrapped.generate(
            {
                "prompts": None,
                "prompt_token_ids": [[1, 2, 3, 4]],
                "sampling_params": {"max_tokens": 3, "temperature": 0.0},
                "session_ids": None,
            }
        )
        assert len(out["response_ids"][0]) == 3
        assert "response_logprobs" in out
        assert len(out["response_logprobs"][0]) == 3
        await wrapped.init_weight_update_communicator("127.0.0.1", 0, 0, 1, "qwen3", "cuda_ipc")
        tensor = model.embed_tokens.weight.detach().clone()
        tensor.add_(0.5)
        await wrapped.update_named_weights(
            {
                "names": ["embed_tokens.weight"],
                "dtypes": ["bf16"],
                "shapes": [list(tensor.shape)],
                "extras": [{"tensors": [tensor]}],
            }
        )
        await wrapped.abort_generation()
        mem = await wrapped.sleep(level=1)
        await wrapped.wake_up()
        await wrapped.reset_prefix_cache()
        await wrapped.teardown()
        return mem

    asyncio.run(_run())


def test_weight_group_join_is_gated_to_real_rendezvous():
    from qwen3_runtime.integrations.skyrl.inference_engine import _should_join_weight_update_group

    assert _should_join_weight_update_group("cuda_ipc", 1, 0) is False
    assert _should_join_weight_update_group("nccl", 1, 12345) is False
    assert _should_join_weight_update_group("nccl", 2, 0) is False
    assert _should_join_weight_update_group("nccl", 2, 29500) is True
    assert _should_join_weight_update_group("gloo", 2, 29500) is True
