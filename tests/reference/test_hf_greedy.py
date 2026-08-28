"""HF greedy token-id gate. Requires local weights; skipped on Mac by default."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

PIN_DIR = Path(__file__).resolve().parents[2] / "docs" / "pins" / "Qwen3-4B"


def _model_dir() -> Path | None:
    raw = os.environ.get("QWEN3_RUNTIME_MODEL")
    return Path(raw) if raw else None


@pytest.mark.skipif(_model_dir() is None, reason="set QWEN3_RUNTIME_MODEL to the pinned Qwen3-4B directory")
def test_checkpoint_config_matches_pin():
    cfg = json.loads((_model_dir() / "config.json").read_text())
    pin = json.loads((PIN_DIR / "config.json").read_text())
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "rope_theta",
        "tie_word_embeddings",
    ):
        assert cfg[key] == pin[key], key


@pytest.mark.skipif(_model_dir() is None, reason="set QWEN3_RUNTIME_MODEL to the pinned Qwen3-4B directory")
@pytest.mark.skipif(
    not torch.cuda.is_available() and not os.environ.get("QWEN3_RUNTIME_ALLOW_CPU_4B"),
    reason="4B greedy on CPU is opt-in via QWEN3_RUNTIME_ALLOW_CPU_4B",
)
def test_qwen3_4b_paged_greedy_matches_hf_token_ids():
    transformers = pytest.importorskip("transformers")

    from qwen3_runtime.config import Config
    from qwen3_runtime.correctness.hf_greedy import hf_greedy_tokens, runtime_greedy_tokens
    from qwen3_runtime.engine.engine import Engine
    from qwen3_runtime.engine.factory import select_attention_backend
    from qwen3_runtime.engine.model_runner import PagedRunner
    from qwen3_runtime.utils.loader import load_from_directory

    model_dir = _model_dir()
    pin = PIN_DIR / "config.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    backend = select_attention_backend()
    ours = load_from_directory(
        model_dir, device=device, dtype=dtype, pin=pin, attention_backend=backend
    )
    hf = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    prompt = [151643, 8948, 198]
    hf_tokens = hf_greedy_tokens(hf, prompt, max_tokens=4)
    engine = Engine(
        Config(block_size=16, num_kv_blocks=256, max_num_seqs=1, max_num_batched_tokens=2048),
        PagedRunner(ours),
    )
    rt = runtime_greedy_tokens(engine, prompt, max_tokens=4)
    assert rt == hf_tokens
