"""Independent tiny-Qwen3 oracle: explicit GQA + QK-Norm-before-RoPE. Not used at runtime."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ReferenceRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        y = x.float() * torch.rsqrt(var + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class ReferenceQwen3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h, d, nh, nkv = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        self.embed = nn.Embedding(cfg.vocab_size, h)
        self.q = nn.ModuleList([nn.Linear(h, nh * d, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.k = nn.ModuleList([nn.Linear(h, nkv * d, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.v = nn.ModuleList([nn.Linear(h, nkv * d, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.o = nn.ModuleList([nn.Linear(nh * d, h, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.q_norm = nn.ModuleList([ReferenceRMSNorm(d, cfg.rms_norm_eps) for _ in range(cfg.num_hidden_layers)])
        self.k_norm = nn.ModuleList([ReferenceRMSNorm(d, cfg.rms_norm_eps) for _ in range(cfg.num_hidden_layers)])
        self.attn_norm = nn.ModuleList([ReferenceRMSNorm(h, cfg.rms_norm_eps) for _ in range(cfg.num_hidden_layers)])
        self.mlp_norm = nn.ModuleList([ReferenceRMSNorm(h, cfg.rms_norm_eps) for _ in range(cfg.num_hidden_layers)])
        self.gate = nn.ModuleList([nn.Linear(h, cfg.intermediate_size, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.up = nn.ModuleList([nn.Linear(h, cfg.intermediate_size, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.down = nn.ModuleList([nn.Linear(cfg.intermediate_size, h, bias=False) for _ in range(cfg.num_hidden_layers)])
        self.final_norm = ReferenceRMSNorm(h, cfg.rms_norm_eps)
        inv = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float) / d)
        )
        t = torch.arange(cfg.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def load_from_fused(self, model: nn.Module) -> None:
        cfg = self.cfg
        q = cfg.num_attention_heads * cfg.head_dim
        kv = cfg.num_key_value_heads * cfg.head_dim
        self.embed.weight.data.copy_(model.embed_tokens.weight)
        self.final_norm.weight.data.copy_(model.norm.weight)
        for i, layer in enumerate(model.layers):
            qkv = layer.attn.qkv_proj.weight
            self.q[i].weight.data.copy_(qkv[:q])
            self.k[i].weight.data.copy_(qkv[q : q + kv])
            self.v[i].weight.data.copy_(qkv[q + kv :])
            self.o[i].weight.data.copy_(layer.attn.o_proj.weight)
            self.q_norm[i].weight.data.copy_(layer.attn.q_norm.weight)
            self.k_norm[i].weight.data.copy_(layer.attn.k_norm.weight)
            self.attn_norm[i].weight.data.copy_(layer.input_layernorm.weight)
            self.mlp_norm[i].weight.data.copy_(layer.post_attention_layernorm.weight)
            gu = layer.mlp.gate_up_proj.weight
            mid = cfg.intermediate_size
            self.gate[i].weight.data.copy_(gu[:mid])
            self.up[i].weight.data.copy_(gu[mid:])
            self.down[i].weight.data.copy_(layer.mlp.down_proj.weight)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.embed(input_ids)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        for i in range(cfg.num_hidden_layers):
            n = self.attn_norm[i](h)
            t = n.shape[0]
            q = self.q[i](n).view(t, cfg.num_attention_heads, cfg.head_dim)
            k = self.k[i](n).view(t, cfg.num_key_value_heads, cfg.head_dim)
            v = self.v[i](n).view(t, cfg.num_key_value_heads, cfg.head_dim)
            q = self.q_norm[i](q)
            k = self.k_norm[i](k)
            q = apply_rope_neox(q, cos, sin)
            k = apply_rope_neox(k, cos, sin)
            group = cfg.num_attention_heads // cfg.num_key_value_heads
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
            qh = q.transpose(0, 1)
            kh = k.transpose(0, 1)
            vh = v.transpose(0, 1)
            scale = 1.0 / math.sqrt(cfg.head_dim)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), diagonal=1)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            ctx = torch.matmul(attn, vh).transpose(0, 1).contiguous().view(t, -1)
            h = h + self.o[i](ctx)
            n = self.mlp_norm[i](h)
            h = h + self.down[i](F.silu(self.gate[i](n)) * self.up[i](n))
        h = self.final_norm(h)
        return F.linear(h, self.embed.weight)
