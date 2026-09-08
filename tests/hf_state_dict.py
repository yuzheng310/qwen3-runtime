"""Inverse of ``load_hf_state_dict`` for tests that round-trip a tiny net."""

from __future__ import annotations

import torch

from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM


def dump_hf_state_dict(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    hf: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": model.embed_tokens.weight.detach().contiguous().cpu().clone(),
        "model.norm.weight": model.norm.weight.detach().contiguous().cpu().clone(),
    }
    if cfg.tie_word_embeddings:
        hf["lm_head.weight"] = hf["model.embed_tokens.weight"].clone()
    else:
        hf["lm_head.weight"] = model.lm_head.weight.detach().contiguous().cpu().clone()
    for i, layer in enumerate(model.layers):
        q, k, v = layer.attn.qkv_proj.weight.detach().cpu().split([q_dim, kv_dim, kv_dim], dim=0)
        gate, up = layer.mlp.gate_up_proj.weight.detach().cpu().split(cfg.intermediate_size, dim=0)
        p = f"model.layers.{i}"
        hf[f"{p}.self_attn.q_proj.weight"] = q.contiguous().clone()
        hf[f"{p}.self_attn.k_proj.weight"] = k.contiguous().clone()
        hf[f"{p}.self_attn.v_proj.weight"] = v.contiguous().clone()
        hf[f"{p}.self_attn.o_proj.weight"] = layer.attn.o_proj.weight.detach().contiguous().cpu().clone()
        hf[f"{p}.self_attn.q_norm.weight"] = layer.attn.q_norm.weight.detach().contiguous().cpu().clone()
        hf[f"{p}.self_attn.k_norm.weight"] = layer.attn.k_norm.weight.detach().contiguous().cpu().clone()
        hf[f"{p}.input_layernorm.weight"] = layer.input_layernorm.weight.detach().contiguous().cpu().clone()
        hf[f"{p}.post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight.detach().contiguous().cpu().clone()
        )
        hf[f"{p}.mlp.gate_proj.weight"] = gate.contiguous().clone()
        hf[f"{p}.mlp.up_proj.weight"] = up.contiguous().clone()
        hf[f"{p}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight.detach().contiguous().cpu().clone()
    return hf
