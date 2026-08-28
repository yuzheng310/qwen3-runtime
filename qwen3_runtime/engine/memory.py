"""KV capacity from remaining VRAM. Weights are not part of the KV pool."""

from __future__ import annotations

from qwen3_runtime.models.qwen3 import Qwen3ModelConfig


def bytes_per_kv_slot(cfg: Qwen3ModelConfig, *, dtype_bytes: int = 2) -> int:
    """One token of K+V across all layers."""
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * dtype_bytes


def bytes_per_block(cfg: Qwen3ModelConfig, block_size: int, *, dtype_bytes: int = 2) -> int:
    return bytes_per_kv_slot(cfg, dtype_bytes=dtype_bytes) * block_size


def num_kv_blocks_for_budget(
    cfg: Qwen3ModelConfig,
    kv_budget_bytes: int,
    block_size: int,
    *,
    dtype_bytes: int = 2,
) -> int:
    per = bytes_per_block(cfg, block_size, dtype_bytes=dtype_bytes)
    if per <= 0:
        raise ValueError("invalid block size")
    return max(0, kv_budget_bytes // per)


def kv_budget_for_simulated_card(
    cfg: Qwen3ModelConfig,
    card_gib: float,
    *,
    overhead_gib: float = 2.0,
    dtype_bytes: int = 2,
) -> int:
    """KV pool as if the process ran on a smaller card.

    Subtracts BF16 weight bytes and a fixed activation/fragmentation overhead
    from ``card_gib``. Used so a 48 GiB AutoDL framebuffer can report a
    retail-4090-capped session-capacity figure.
    """
    if card_gib <= 0:
        raise ValueError("card_gib must be positive")
    card = int(card_gib * (1024**3))
    weights = estimate_weight_bytes(cfg, dtype_bytes=dtype_bytes)
    overhead = int(overhead_gib * (1024**3))
    budget = card - weights - overhead
    if budget <= 0:
        raise ValueError(
            f"card_gib={card_gib} leaves no KV after weights={weights} overhead={overhead}"
        )
    return budget


def estimate_weight_bytes(cfg: Qwen3ModelConfig, *, dtype_bytes: int = 2) -> int:
    """Parameter footprint excluding optimizer state. Tied lm_head is not double-counted."""
    h = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    per_layer = (
        h * (q_dim + 2 * kv_dim)  # qkv
        + q_dim * h  # o
        + 2 * cfg.head_dim  # q/k rms
        + 2 * h  # input/post rms
        + h * (2 * cfg.intermediate_size)  # gate_up
        + cfg.intermediate_size * h  # down
    )
    embed = cfg.vocab_size * h
    final_rms = h
    lm = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h
    params = embed + cfg.num_hidden_layers * per_layer + final_rms + lm
    return params * dtype_bytes
