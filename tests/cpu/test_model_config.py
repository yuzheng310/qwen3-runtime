import json
from pathlib import Path

from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

PIN = Path("docs/pins/Qwen3-4B/config.json")


def test_from_hf_reads_pinned_qwen3_4b_config():
    cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    assert cfg.hidden_size == 2560
    assert cfg.num_hidden_layers == 36
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 9728
    assert cfg.vocab_size == 151936
    assert cfg.rope_theta == 1_000_000
    assert cfg.max_position_embeddings == 40960
    assert cfg.tie_word_embeddings is True
    assert cfg.rms_norm_eps == 1e-6


def test_from_hf_rejects_sliding_window():
    raw = json.loads(PIN.read_text())
    raw["use_sliding_window"] = True
    try:
        Qwen3ModelConfig.from_hf_config(raw)
    except ValueError as exc:
        assert "sliding" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
