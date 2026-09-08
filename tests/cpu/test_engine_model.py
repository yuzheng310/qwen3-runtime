import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.reference.eager_runner import PytorchEagerRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig
from qwen3_runtime.sampling import greedy_logits


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


def _sequential_greedy(model: Qwen3ForCausalLM, prompt: list[int], max_tokens: int) -> list[int]:
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor(ids, dtype=torch.long)
            pos = torch.arange(len(ids), dtype=torch.long)
            logits = model(x, pos)
            ids.append(greedy_logits(logits[-1:].contiguous())[0])
    return ids[len(prompt) :]


def test_engine_greedy_matches_sequential_recompute():
    torch.manual_seed(3)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 5, 9]
    expected = _sequential_greedy(model, prompt, max_tokens=4)

    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=4, max_num_batched_tokens=16)
    engine = Engine(cfg, PytorchEagerRunner(model))
    got = engine.generate(prompt, max_tokens=4)
    assert got == expected


def test_engine_real_model_alone_equals_batched():
    torch.manual_seed(4)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt_a = [2, 3, 7]
    prompt_b = [8, 1, 4, 6, 9]

    cfg = Config(block_size=4, num_kv_blocks=64, max_num_seqs=4, max_num_batched_tokens=8)
    alone = Engine(cfg, PytorchEagerRunner(model)).generate(prompt_a, max_tokens=3)

    cfg = Config(block_size=4, num_kv_blocks=64, max_num_seqs=4, max_num_batched_tokens=8)
    batched = Engine(cfg, PytorchEagerRunner(model))
    id_a = batched.add_request(prompt_a, max_tokens=3)
    batched.add_request(prompt_b, max_tokens=2)
    got_a: list[int] = []
    while not batched.is_finished():
        for rid, tok, _done in batched.step():
            if rid == id_a and tok is not None:
                got_a.append(tok)
    assert got_a == alone
