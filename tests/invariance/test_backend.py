import importlib.util

import pytest
import torch

from qwen3_runtime.attention.paged import paged_context
from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.kv.paged import PagedBatch, PagedKVPool
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config


def _paged_context(backend: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_len: int) -> torch.Tensor:
    """One sequence: store scheduled K/V at the suffix of `kv_len` and attend."""
    q_len, n_heads, head_dim = q.shape
    n_kv = k.shape[1]
    assert q_len <= kv_len
    start = kv_len - q_len
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=16,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    if start:
        torch.manual_seed(0)
        pk = torch.randn(start, n_kv, head_dim, dtype=q.dtype, device=q.device)
        pv = torch.randn(start, n_kv, head_dim, dtype=q.dtype, device=q.device)
        pool.store(0, pk, pv, torch.arange(start, device=q.device))
    batch = PagedBatch(
        pool=pool,
        slot_mapping=torch.arange(start, kv_len, device=q.device),
        block_tables=[[0]],
        kv_lens=[kv_len],
        cu_seqlens=[0, q_len],
    )
    return paged_context(backend, q, k, v, batch, 0, n_heads, n_kv, head_dim)


@pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="flashinfer paged kernels need CUDA")
def test_paged_flashinfer_matches_pytorch_on_prefill_and_decode():
    torch.manual_seed(3)
    device = torch.device("cuda")
    q_len, kv_len, n_heads, n_kv, head_dim = 4, 4, 4, 2, 128
    q = torch.randn(q_len, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(q_len, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(q_len, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    pt = _paged_context("pytorch", q, k, v, kv_len)
    fi = _paged_context("flashinfer", q.clone(), k.clone(), v.clone(), kv_len)
    torch.testing.assert_close(fi.float(), pt.float(), atol=2e-2, rtol=2e-2)

    q1 = torch.randn(1, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k1 = torch.randn(1, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    v1 = torch.randn(1, n_kv, head_dim, device=device, dtype=torch.bfloat16)
    pt_d = _paged_context("pytorch", q1, k1, v1, kv_len=6)
    fi_d = _paged_context("flashinfer", q1.clone(), k1.clone(), v1.clone(), kv_len=6)
    torch.testing.assert_close(fi_d.float(), pt_d.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="flashinfer not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="flashinfer generate needs CUDA")
def test_flashinfer_paged_generate_matches_pytorch_explicit():
    torch.manual_seed(22)
    cfg = tiny_config()
    pytorch = Qwen3ForCausalLM(cfg, attention_backend="pytorch").eval()
    fi_model = Qwen3ForCausalLM(cfg, attention_backend="flashinfer").eval()
    fi_model.load_state_dict(pytorch.state_dict())
    device = torch.device("cuda")
    pytorch = pytorch.to(device)
    fi_model = fi_model.to(device)
    prompt = [1, 4, 7, 2, 9, 3]
    engine_cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    a = Engine(engine_cfg, PagedRunner(pytorch)).generate(prompt, max_tokens=4)
    b = Engine(
        Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16),
        PagedRunner(fi_model),
    ).generate(prompt, max_tokens=4)
    assert a == b


def test_flashinfer_backend_without_package_raises():
    if importlib.util.find_spec("flashinfer"):
        pytest.skip("flashinfer installed; paged path is live")
    with pytest.raises(NotImplementedError):
        paged_context(
            "flashinfer",
            torch.zeros(1, 4, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
            None,  # type: ignore[arg-type]
            0,
            4,
            2,
            4,
        )
