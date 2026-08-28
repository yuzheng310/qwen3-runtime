import torch

from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig
from qwen3_runtime.utils.loader import load_hf_state_dict
from tests.cpu.reference_qwen3 import ReferenceQwen3
from tests.cpu.test_tiny_qwen3 import tiny_config


def _hf_style_from_reference(ref: ReferenceQwen3, cfg: Qwen3ModelConfig) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": ref.embed.weight.detach().clone(),
        "model.norm.weight": ref.final_norm.weight.detach().clone(),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = ref.attn_norm[i].weight.detach().clone()
        sd[f"{p}.post_attention_layernorm.weight"] = ref.mlp_norm[i].weight.detach().clone()
        sd[f"{p}.self_attn.q_proj.weight"] = ref.q[i].weight.detach().clone()
        sd[f"{p}.self_attn.k_proj.weight"] = ref.k[i].weight.detach().clone()
        sd[f"{p}.self_attn.v_proj.weight"] = ref.v[i].weight.detach().clone()
        sd[f"{p}.self_attn.o_proj.weight"] = ref.o[i].weight.detach().clone()
        sd[f"{p}.self_attn.q_norm.weight"] = ref.q_norm[i].weight.detach().clone()
        sd[f"{p}.self_attn.k_norm.weight"] = ref.k_norm[i].weight.detach().clone()
        sd[f"{p}.mlp.gate_proj.weight"] = ref.gate[i].weight.detach().clone()
        sd[f"{p}.mlp.up_proj.weight"] = ref.up[i].weight.detach().clone()
        sd[f"{p}.mlp.down_proj.weight"] = ref.down[i].weight.detach().clone()
    return sd


def test_hf_style_qkv_and_gate_up_mapping_matches_reference():
    torch.manual_seed(6)
    cfg = tiny_config()
    fused = Qwen3ForCausalLM(cfg).eval()
    ref = ReferenceQwen3(cfg)
    ref.load_from_fused(fused)
    # scramble fused weights then reload from HF-style dict
    fused = Qwen3ForCausalLM(cfg).eval()
    load_hf_state_dict(fused, _hf_style_from_reference(ref, cfg))

    ids = torch.tensor([2, 8, 1, 4], dtype=torch.long)
    pos = torch.arange(ids.numel(), dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(fused(ids, pos), ref(ids, pos), atol=1e-5, rtol=1e-5)
