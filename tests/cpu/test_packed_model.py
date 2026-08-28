import torch

from qwen3_runtime.engine.batch import last_token_indices
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


def test_packed_unequal_prompts_match_separate_forwards():
    torch.manual_seed(2)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    a = torch.tensor([1, 2, 3], dtype=torch.long)
    b = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
    packed = torch.cat([a, b])
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], dtype=torch.long)
    cu = [0, 3, 8]
    with torch.no_grad():
        packed_logits = model(packed, positions, cu_seqlens=cu)
        last = packed_logits[last_token_indices(cu)]
        la = model(a, torch.arange(3))[-1]
        lb = model(b, torch.arange(5))[-1]
    torch.testing.assert_close(last[0], la, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(last[1], lb, atol=1e-5, rtol=1e-5)


def test_select_rows_lm_head_matches_full_then_index():
    torch.manual_seed(2)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    packed = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], dtype=torch.long)
    cu = [0, 3, 8]
    idx = torch.tensor(last_token_indices(cu), dtype=torch.long)
    with torch.no_grad():
        full = model(packed, positions, cu_seqlens=cu)
        selected = model(packed, positions, cu_seqlens=cu, select_rows=idx)
        skipped = model(packed, positions, cu_seqlens=cu, compute_logits=False)
    torch.testing.assert_close(selected, full[idx], atol=1e-5, rtol=1e-5)
    assert skipped.shape[0] == packed.shape[0]
    assert skipped.shape[-1] == model.cfg.hidden_size
