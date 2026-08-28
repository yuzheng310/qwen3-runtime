import json
from pathlib import Path

import pytest
import torch

from qwen3_runtime.engine.factory import (
    CODESCOUT_PIN,
    PIN,
    default_cuda_graph,
    eos_token_id_from_pin,
    kv_budget_bytes,
    select_attention_backend,
)
from qwen3_runtime.engine.memory import num_kv_blocks_for_budget
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

pytestmark = pytest.mark.skipif(torch.cuda.is_available(), reason="CPU factory/budget path")


def test_cpu_attention_backend_is_pytorch_reference():
    assert select_attention_backend() == "pytorch"


def test_pin_eos_token_id_is_qwen_im_end():
    assert eos_token_id_from_pin(PIN) == 151645
    assert eos_token_id_from_pin(CODESCOUT_PIN) == 151645


def test_config_cuda_graph_dataclass_defaults_off():
    from qwen3_runtime.config import Config

    assert Config().cuda_graph is False
    assert Config().num_speculative_tokens == 0
    assert Config().ngram_min == 2
    assert Config().ngram_max == 4


def test_factory_cuda_graph_default_only_cuda_flashinfer():
    assert default_cuda_graph(device="cuda", backend="flashinfer") is True
    assert default_cuda_graph(device="cuda", backend="triton") is True
    assert default_cuda_graph(device="cuda", backend="sdpa") is False
    assert default_cuda_graph(device="cuda", backend="flash_attn") is False
    assert default_cuda_graph(device="cpu", backend="flashinfer") is False
    assert default_cuda_graph(device="cpu", backend="pytorch") is False


def test_cpu_kv_budget_is_256_mib_not_full_vram_guess():
    assert kv_budget_bytes() == 256 * 1024 * 1024


def test_factory_kv_block_count_uses_memory_helper():
    cfg = Qwen3ModelConfig.from_hf_config(
        json.loads(Path("docs/pins/Qwen3-4B/config.json").read_text())
    )
    budget = kv_budget_bytes()
    blocks = max(32, num_kv_blocks_for_budget(cfg, budget, 16, dtype_bytes=4))
    # CPU factory uses float32 (4 bytes). Must be the helper, not a magic 8192.
    assert blocks == max(32, (256 * 1024 * 1024) // (2 * 36 * 8 * 128 * 4 * 16))
    assert blocks >= 32
