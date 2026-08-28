import json
from pathlib import Path

from qwen3_runtime.engine.memory import (
    bytes_per_block,
    bytes_per_kv_slot,
    estimate_weight_bytes,
    kv_budget_for_simulated_card,
    num_kv_blocks_for_budget,
)
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

PIN = Path("docs/pins/Qwen3-4B/config.json")


def test_qwen3_4b_kv_block_math_is_traceable():
    cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    # 2 (K/V) * 36 layers * 8 kv heads * 128 dim * 2 bytes * 16 tokens
    assert bytes_per_block(cfg, 16, dtype_bytes=2) == 2 * 36 * 8 * 128 * 2 * 16
    blocks = num_kv_blocks_for_budget(cfg, 8 * 1024**3, 16, dtype_bytes=2)
    assert blocks > 1000
    weights = estimate_weight_bytes(cfg, dtype_bytes=2)
    # 4B-class BF16 weights are ~8 GiB, not 24 GiB and not 100 MiB.
    assert 6 * 1024**3 < weights < 12 * 1024**3


def test_simulated_24gib_card_is_the_primary_session_kv_cap():
    cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    budget_24 = kv_budget_for_simulated_card(cfg, 24.0)
    budget_48 = kv_budget_for_simulated_card(cfg, 48.0)
    assert budget_24 < budget_48
    assert 12 * 1024**3 < budget_24 < 16 * 1024**3
    per_token = bytes_per_kv_slot(cfg, dtype_bytes=2)
    p50_peak = 15_898
    n_p50 = budget_24 // (per_token * p50_peak)
    assert 5 <= n_p50 <= 8
    try:
        kv_budget_for_simulated_card(cfg, 1.0)
    except ValueError as exc:
        assert "no KV" in str(exc)
    else:
        raise AssertionError("expected ValueError for a 1 GiB card")
