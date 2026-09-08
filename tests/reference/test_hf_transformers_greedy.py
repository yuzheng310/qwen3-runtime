"""Correctness Baseline v0 methodology on a tiny net: HuggingFace Qwen3ForCausalLM is the oracle."""

import pytest
import torch

from qwen3_runtime.config import Config
from qwen3_runtime.reference.hf_greedy import hf_greedy_tokens, runtime_greedy_tokens
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config
from tests.hf_state_dict import dump_hf_state_dict


def test_paged_runtime_greedy_matches_transformers_qwen3():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(13)
    ours = Qwen3ForCausalLM(tiny_config()).eval()
    cfg = tiny_config()
    hf_cfg = transformers.Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=True,
        attention_bias=False,
        hidden_act="silu",
    )
    hf = transformers.Qwen3ForCausalLM(hf_cfg).eval()
    missing, unexpected = hf.load_state_dict(dump_hf_state_dict(ours), strict=False)
    assert not missing, missing
    prompt = [1, 5, 9, 2]
    hf_tokens = hf_greedy_tokens(hf, prompt, max_tokens=5)
    engine = Engine(
        Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=32),
        PagedRunner(ours),
    )
    rt_tokens = runtime_greedy_tokens(engine, prompt, max_tokens=5)
    assert rt_tokens == hf_tokens
