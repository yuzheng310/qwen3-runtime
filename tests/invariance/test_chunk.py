import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.reference.eager_runner import PytorchEagerRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig


def tiny_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


def test_chunked_prefill_tokens_match_unchunked():
    torch.manual_seed(5)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = list(range(1, 21))
    runner = PytorchEagerRunner(model)

    def gen(budget: int) -> list[int]:
        cfg = Config(block_size=4, num_kv_blocks=64, max_num_seqs=2, max_num_batched_tokens=budget)
        return Engine(cfg, runner).generate(prompt, max_tokens=4)

    unchunked = gen(64)
    assert gen(7) == unchunked
    assert gen(3) == unchunked
    assert gen(1) == unchunked
