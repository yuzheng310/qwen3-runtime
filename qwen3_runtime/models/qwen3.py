from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from qwen3_runtime.attention.paged import AttentionState, paged_context
from qwen3_runtime.kv.paged import PagedBatch
from qwen3_runtime.layers.ops import Ops
from qwen3_runtime.layers.rmsnorm import RMSNorm
from qwen3_runtime.layers.rope import RotaryEmbedding
from qwen3_runtime.reference.dense_attention import dense_context


@dataclass
class Qwen3ModelConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf_config(cls, raw: Mapping[str, Any]) -> Qwen3ModelConfig:
        arch = raw.get("architectures") or []
        if "Qwen3ForCausalLM" not in arch:
            raise ValueError(f"unsupported architectures: {arch}")
        if raw.get("hidden_act") != "silu":
            raise ValueError(f"unsupported hidden_act: {raw.get('hidden_act')}")
        if raw.get("use_sliding_window"):
            raise ValueError("sliding window attention is out of scope")
        if raw.get("rope_scaling") not in (None, {}):
            raise ValueError(f"unsupported rope_scaling: {raw.get('rope_scaling')}")
        if raw.get("attention_bias"):
            raise ValueError("Qwen3 qkv bias is not implemented")
        head_dim = int(raw.get("head_dim") or int(raw["hidden_size"]) // int(raw["num_attention_heads"]))
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=head_dim,
            intermediate_size=int(raw["intermediate_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            tie_word_embeddings=bool(raw["tie_word_embeddings"]),
        )


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen3ModelConfig,
        rope: RotaryEmbedding,
        backend: str = "pytorch",
        ops: Ops | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.ops = ops or Ops.torch()
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, ops=self.ops)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, ops=self.ops)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None = None,
        paged: PagedBatch | None = None,
        layer_id: int = 0,
        attn_state: AttentionState | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        num_tokens = x.shape[0]
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        # x: [T, H] -> q: [T, n_heads, D], k/v: [T, n_kv, D]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(num_tokens, cfg.num_attention_heads, cfg.head_dim)
        k = k.view(num_tokens, cfg.num_key_value_heads, cfg.head_dim)
        v = v.view(num_tokens, cfg.num_key_value_heads, cfg.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.ops.apply_rope(q, k, positions, self.rope)
        if paged is not None:
            ctx = paged_context(
                self.backend,
                q,
                k,
                v,
                paged,
                layer_id,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                attn_state,
            )
            return self.o_proj(ctx)
        ctx = dense_context(
            q,
            k,
            v,
            cu_seqlens,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        return self.o_proj(ctx)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3ModelConfig, ops: Ops | None = None):
        super().__init__()
        self.ops = ops or Ops.torch()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.ops.silu_and_mul(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3ModelConfig,
        rope: RotaryEmbedding,
        layer_idx: int,
        attention_backend: str = "pytorch",
        ops: Ops | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ops = ops or Ops.torch()
        self.eps = cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, ops=self.ops)
        self.attn = Qwen3Attention(cfg, rope, backend=attention_backend, ops=self.ops)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, ops=self.ops)
        self.mlp = Qwen3MLP(cfg, ops=self.ops)

    def forward(
        self,
        h: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None,
        paged: PagedBatch | None,
        attn_state: AttentionState | None,
        next_norm_weight: torch.Tensor,
        *,
        last: bool,
        compute_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.attn(
            h,
            positions,
            cu_seqlens=cu_seqlens,
            paged=paged,
            layer_id=self.layer_idx,
            attn_state=attn_state,
        )
        h, residual = self.ops.fused_add_rmsnorm(
            h, residual, self.post_attention_layernorm.weight, self.eps
        )
        h = self.mlp(h)
        if last and not compute_logits:
            return h, residual
        h, residual = self.ops.fused_add_rmsnorm(h, residual, next_norm_weight, self.eps)
        return h, residual


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3ModelConfig, attention_backend: str = "pytorch", ops: Ops | None = None):
        super().__init__()
        self.cfg = cfg
        self.attention_backend = attention_backend
        self.ops = ops or Ops.torch()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    cfg, self.rope, i, attention_backend=attention_backend, ops=self.ops
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, ops=self.ops)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def set_ops(self, ops: Ops) -> None:
        self.ops = ops
        for module in self.modules():
            if hasattr(module, "ops") and module is not self:
                module.ops = ops

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        device = next(out.parameters()).device
        out.set_ops(Ops.select(device))
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None = None,
        paged: PagedBatch | None = None,
        select_rows: torch.Tensor | None = None,
        compute_logits: bool = True,
        attn_state: AttentionState | None = None,
    ) -> torch.Tensor:
        """Transformer + optional LM head.

        Both CPU and CUDA walk this residual loop. ``Ops`` (bound at ``.to()``)
        chooses torch vs FlashInfer fused_add_rmsnorm; the graph of calls does
        not change.

        Shapes: ``input_ids`` ``[T]`` → hidden ``[T, H]`` → logits ``[T, V]``
        (or ``[R, V]`` when ``select_rows`` picks R packed positions).
        """
        if paged is not None and attn_state is None:
            attn_state = AttentionState()
        h = self.embed_tokens(input_ids)
        residual = h
        eps = self.cfg.rms_norm_eps
        h = self.ops.rmsnorm(h, self.layers[0].input_layernorm.weight, eps)
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1].input_layernorm.weight if i + 1 < n else self.norm.weight
            h, residual = layer(
                h,
                residual,
                positions,
                cu_seqlens,
                paged,
                attn_state,
                nxt,
                last=i + 1 == n,
                compute_logits=compute_logits,
            )
        if not compute_logits:
            return h
        if select_rows is not None:
            h = h.index_select(0, select_rows)
        weight = self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight
        return F.linear(h, weight)
