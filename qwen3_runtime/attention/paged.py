"""Paged attention over a physical block pool.

Three backends, each answering a different question:

- ``pytorch`` — what paged attention is (explicit scores; CPU tests).
- ``flashinfer`` — what we ship.
- ``triton`` — a decode kernel you can read. Prefill uses pytorch, not a
  silent mix of flashinfer-then-sdpa.
"""

from __future__ import annotations

import torch

from qwen3_runtime.attention.flashinfer_backend import (
    FLASHINFER,
    FlashInferRuntime,
    flashinfer_page_tensors,
    make_flashinfer_graph_decode_wrapper,
    paged_flashinfer,
    plan_flashinfer_decode,
    store_flashinfer_kv,
)
from qwen3_runtime.attention.pytorch import paged_pytorch
from qwen3_runtime.attention.state import AttentionState
from qwen3_runtime.attention.triton_attn import paged_triton
from qwen3_runtime.kv.paged import PagedBatch

__all__ = [
    "AttentionState",
    "FLASHINFER",
    "FlashInferRuntime",
    "flashinfer_page_tensors",
    "make_flashinfer_graph_decode_wrapper",
    "paged_context",
    "plan_flashinfer_decode",
    "store_flashinfer_kv",
]


def paged_context(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    paged: PagedBatch,
    layer_id: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attn_state: AttentionState | None = None,
) -> torch.Tensor:
    state = attn_state if attn_state is not None else AttentionState()
    if backend == "pytorch":
        return paged_pytorch(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim
        )
    if backend == "flashinfer":
        return paged_flashinfer(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim, state
        )
    if backend == "triton":
        return paged_triton(
            q, k, v, paged, layer_id, num_attention_heads, num_key_value_heads, head_dim, state
        )
    raise ValueError(f"unknown attention backend {backend!r}")
