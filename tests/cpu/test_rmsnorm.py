import torch

from qwen3_runtime.layers.rmsnorm import RMSNorm


def test_cpu_rmsnorm_matches_float_formula():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    m = RMSNorm(8, 1e-6)
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + 1e-6)
    want = (y * m.weight.float()).to(x.dtype)
    torch.testing.assert_close(m(x), want)
