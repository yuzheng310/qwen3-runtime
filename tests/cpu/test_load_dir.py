import json

import torch
from safetensors.torch import save_file

from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.utils.loader import load_from_directory
from tests.cpu.test_tiny_qwen3 import tiny_config
from tests.hf_state_dict import dump_hf_state_dict


def _tiny_hf_config() -> dict:
    cfg = tiny_config()
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "attention_bias": False,
        "use_sliding_window": False,
        "rope_scaling": None,
        "torch_dtype": "float32",
    }


def test_load_from_directory_roundtrip(tmp_path):
    torch.manual_seed(12)
    original = Qwen3ForCausalLM(tiny_config()).eval()
    (tmp_path / "config.json").write_text(json.dumps(_tiny_hf_config()))
    save_file(dump_hf_state_dict(original), str(tmp_path / "model.safetensors"))

    loaded = load_from_directory(tmp_path).eval()
    ids = torch.tensor([3, 1, 8, 2], dtype=torch.long)
    pos = torch.arange(ids.numel())
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, pos), original(ids, pos), atol=1e-5, rtol=1e-5)

def test_load_from_directory_rejects_pin_mismatch(tmp_path):
    torch.manual_seed(12)
    original = Qwen3ForCausalLM(tiny_config()).eval()
    (tmp_path / "config.json").write_text(json.dumps(_tiny_hf_config()))
    save_file(dump_hf_state_dict(original), str(tmp_path / "model.safetensors"))
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({**_tiny_hf_config(), "hidden_size": 999}))
    try:
        load_from_directory(tmp_path, pin=pin)
    except ValueError as exc:
        assert "hidden_size" in str(exc)
    else:
        raise AssertionError("expected pin mismatch")


def test_load_from_directory_reads_sharded_index(tmp_path):
    torch.manual_seed(14)
    original = Qwen3ForCausalLM(tiny_config()).eval()
    (tmp_path / "config.json").write_text(json.dumps(_tiny_hf_config()))
    hf = dump_hf_state_dict(original)
    keys = list(hf)
    mid = max(1, len(keys) // 2)
    a = {k: hf[k] for k in keys[:mid]}
    b = {k: hf[k] for k in keys[mid:]}
    save_file(a, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(b, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **{k: "model-00001-of-00002.safetensors" for k in a},
                    **{k: "model-00002-of-00002.safetensors" for k in b},
                }
            }
        )
    )
    loaded = load_from_directory(tmp_path).eval()
    ids = torch.tensor([3, 1, 8, 2], dtype=torch.long)
    pos = torch.arange(ids.numel())
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, pos), original(ids, pos), atol=1e-5, rtol=1e-5)


def test_target_dtype_avoids_full_precision_device_transfer(tmp_path, monkeypatch):
    """Measure bytes crossing the device boundary, emulated on CPU."""
    original = Qwen3ForCausalLM(tiny_config()).eval()
    (tmp_path / "config.json").write_text(json.dumps(_tiny_hf_config()))
    save_file(dump_hf_state_dict(original), str(tmp_path / "model.safetensors"))
    real_to = Qwen3ForCausalLM.to
    transferred = []

    def inspect_to(self, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            dtype = kwargs.get("dtype")
            transferred.append(sum(
                p.numel() * (torch.empty((), dtype=dtype).element_size()
                             if dtype is not None else p.element_size())
                for p in self.parameters()
            ))
            kwargs = {**kwargs, "device": "cpu"}
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(Qwen3ForCausalLM, "to", inspect_to)
    loaded = load_from_directory(tmp_path, device="cuda", dtype=torch.bfloat16)
    expected_bytes = sum(p.numel() * 2 for p in original.parameters())
    assert transferred == [expected_bytes]
    for actual, expected in zip(loaded.parameters(), original.parameters()):
        assert torch.equal(actual, expected.to(torch.bfloat16))
