from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from qwen3_runtime.attention.paged import paged_context
from qwen3_runtime.kv.paged import PagedBatch
from qwen3_runtime.layers.rmsnorm import RMSNorm
from qwen3_runtime.layers.rope import RotaryEmbedding, apply_qwen3_rope


def _allowed_attention_mask(q_len: int, cu_seqlens: Sequence[int] | None, device: torch.device) -> torch.Tensor:
    """True where attention is allowed. Single-sequence causal if cu_seqlens is None."""
    if cu_seqlens is None:
        return torch.tril(torch.ones(q_len, q_len, dtype=torch.bool, device=device))
    seq_id = torch.empty(q_len, dtype=torch.long, device=device)
    pos = torch.empty(q_len, dtype=torch.long, device=device)
    for i, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        seq_id[start:end] = i
        pos[start:end] = torch.arange(end - start, device=device)
    same = seq_id[:, None] == seq_id[None, :]
    causal = pos[:, None] >= pos[None, :]
    return same & causal


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
    def from_hf_config(cls, raw: Mapping[str, Any]) -> "Qwen3ModelConfig":
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
    def __init__(self, cfg: Qwen3ModelConfig, rope: RotaryEmbedding, backend: str = "pytorch"):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, q_dim + 2 * kv_dim, bias=False)
        self.o = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None = None,
        paged: PagedBatch | None = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        cfg = self.cfg
        t = x.shape[0]
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        qkv = self.qkv(x)
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(t, cfg.num_attention_heads, cfg.head_dim)
        k = k.view(t, cfg.num_key_value_heads, cfg.head_dim)
        v = v.view(t, cfg.num_key_value_heads, cfg.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_qwen3_rope(q, k, positions, self.rope)
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
            )
            return self.o(ctx)
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scale = 1.0 / math.sqrt(cfg.head_dim)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        allowed = _allowed_attention_mask(t, cu_seqlens, x.device)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        ctx = torch.matmul(attn, vh).transpose(0, 1).contiguous().view(t, -1)
        return self.o(ctx)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        if x.is_cuda:
            try:
                from flashinfer import silu_and_mul
            except ImportError:
                pass
            else:
                return self.down(silu_and_mul(gate_up))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3ModelConfig,
        rope: RotaryEmbedding,
        layer_idx: int,
        attention_backend: str = "pytorch",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Qwen3Attention(cfg, rope, backend=attention_backend)
        self.post_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None = None,
        paged: PagedBatch | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.input_norm(x),
            positions,
            cu_seqlens=cu_seqlens,
            paged=paged,
            layer_id=self.layer_idx,
        )
        x = x + self.mlp(self.post_norm(x))
        return x


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3ModelConfig, attention_backend: str = "pytorch"):
        super().__init__()
        self.cfg = cfg
        self.attention_backend = attention_backend
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(cfg, self.rope, i, attention_backend=attention_backend)
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None = None,
        paged: PagedBatch | None = None,
        select_rows: torch.Tensor | None = None,
        compute_logits: bool = True,
    ) -> torch.Tensor:
        """Transformer + optional LM head.

        KV is written inside attention regardless of ``compute_logits``. Prefill
        chunks that do not sample skip the vocab GEMM. ``select_rows`` applies
        the LM head only to those packed positions (last token per sampled
        sequence). Decode CUDA Graph capture leaves both at defaults.
        """
        h = self.embed(input_ids)
        if h.is_cuda:
            fused = self._forward_fused_residual(
                h, positions, cu_seqlens, paged, select_rows, compute_logits
            )
            if fused is not None:
                return fused
        for layer in self.layers:
            h = layer(h, positions, cu_seqlens=cu_seqlens, paged=paged)
        if not compute_logits:
            return h
        h = self.final_norm(h)
        if select_rows is not None:
            h = h.index_select(0, select_rows)
        weight = self.embed.weight if self.cfg.tie_word_embeddings else self.lm_head.weight
        return F.linear(h, weight)

    def _forward_fused_residual(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: Sequence[int] | None,
        paged: PagedBatch | None,
        select_rows: torch.Tensor | None,
        compute_logits: bool,
    ) -> torch.Tensor | None:
        """CUDA residual = fused_add_rmsnorm (FlashInfer). CPU keeps add then RMSNorm."""
        try:
            from flashinfer import fused_add_rmsnorm
        except ImportError:
            return None
        residual = h
        h = self.layers[0].input_norm(h)
        eps = self.cfg.rms_norm_eps
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer.attn(
                h, positions, cu_seqlens=cu_seqlens, paged=paged, layer_id=layer.layer_idx
            )
            fused_add_rmsnorm(h, residual, layer.post_norm.weight, eps)
            h = layer.mlp(h)
            nxt = self.layers[i + 1].input_norm.weight if i + 1 < n else self.final_norm.weight
            if i + 1 == n and not compute_logits:
                return h
            fused_add_rmsnorm(h, residual, nxt, eps)
        if select_rows is not None:
            h = h.index_select(0, select_rows)
        weight = self.embed.weight if self.cfg.tie_word_embeddings else self.lm_head.weight
        return F.linear(h, weight)
