import torch

from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig
from tests.cpu.reference_qwen3 import ReferenceQwen3


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


def test_tiny_qwen3_matches_independent_reference_logits():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    ref = ReferenceQwen3(cfg)
    ref.load_from_fused(model)
    ref.eval()

    ids = torch.tensor([1, 4, 9, 2, 7], dtype=torch.long)
    pos = torch.arange(ids.numel(), dtype=torch.long)
    with torch.no_grad():
        got = model(ids, pos)
        exp = ref(ids, pos)
    torch.testing.assert_close(got, exp, atol=1e-5, rtol=1e-5)


def test_tiny_qwen3_greedy_tokens_are_stable():
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(tiny_config())
    model.eval()
    ids = torch.tensor([3, 1, 8], dtype=torch.long)
    pos = torch.arange(ids.numel(), dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, pos)
    token = int(logits[-1].argmax())
    with torch.no_grad():
        again = int(model(ids, pos)[-1].argmax())
    assert token == again
    assert 0 <= token < 32
