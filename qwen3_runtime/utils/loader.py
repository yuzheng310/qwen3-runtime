"""Load HuggingFace Qwen3 weights into fused modules.

Native parameter names match HF where they are 1:1
(``embed_tokens``, ``norm``, ``input_layernorm``, ``post_attention_layernorm``,
``o_proj``, ``down_proj``, ``q_norm``, ``k_norm``). The fused leftovers are
``qkv_proj`` = cat(q, k, v) and ``gate_up_proj`` = cat(gate, up).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file

from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig

PIN_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "rope_theta",
    "tie_word_embeddings",
)


def load_hf_state_dict(model: Qwen3ForCausalLM, hf: dict[str, torch.Tensor]) -> None:
    """Copy HuggingFace Qwen3ForCausalLM keys onto the fused native modules."""
    cfg = model.cfg
    model.embed_tokens.weight.data.copy_(hf["model.embed_tokens.weight"])
    model.norm.weight.data.copy_(hf["model.norm.weight"])
    if not cfg.tie_word_embeddings:
        model.lm_head.weight.data.copy_(hf["lm_head.weight"])
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        q = hf[f"{p}.self_attn.q_proj.weight"]
        k = hf[f"{p}.self_attn.k_proj.weight"]
        v = hf[f"{p}.self_attn.v_proj.weight"]
        layer.attn.qkv_proj.weight.data.copy_(torch.cat([q, k, v], dim=0))
        layer.attn.o_proj.weight.data.copy_(hf[f"{p}.self_attn.o_proj.weight"])
        layer.attn.q_norm.weight.data.copy_(hf[f"{p}.self_attn.q_norm.weight"])
        layer.attn.k_norm.weight.data.copy_(hf[f"{p}.self_attn.k_norm.weight"])
        layer.input_layernorm.weight.data.copy_(hf[f"{p}.input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(hf[f"{p}.post_attention_layernorm.weight"])
        gate = hf[f"{p}.mlp.gate_proj.weight"]
        up = hf[f"{p}.mlp.up_proj.weight"]
        layer.mlp.gate_up_proj.weight.data.copy_(torch.cat([gate, up], dim=0))
        layer.mlp.down_proj.weight.data.copy_(hf[f"{p}.mlp.down_proj.weight"])


def assert_matches_pin(raw: Mapping, pin: Mapping) -> None:
    for key in PIN_KEYS:
        if raw.get(key) != pin.get(key):
            raise ValueError(f"config {key}={raw.get(key)!r} != pin {pin.get(key)!r}")


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        spec = json.loads(index.read_text())
        weight_map = spec.get("weight_map") or {}
        files = sorted(set(weight_map.values()))
        if not files:
            raise FileNotFoundError(f"empty weight_map in {index}")
        tensors: dict[str, torch.Tensor] = {}
        for name in files:
            tensors.update(load_file(str(path / name)))
        return tensors
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    tensors = {}
    for file in files:
        tensors.update(load_file(str(file)))
    return tensors


def load_from_directory(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    pin: str | Path | None = None,
    attention_backend: str = "pytorch",
) -> Qwen3ForCausalLM:
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    if pin is not None:
        assert_matches_pin(raw, json.loads(Path(pin).read_text()))
    cfg = Qwen3ModelConfig.from_hf_config(raw)
    model = Qwen3ForCausalLM(cfg, attention_backend=attention_backend)
    load_hf_state_dict(model, _load_tensors(path))
    # Convert while moving, so BF16 inference never stages a full FP32 model
    # on the accelerator before the factory measures its remaining capacity.
    model = model.to(device=device, dtype=dtype)
    return model.eval()
