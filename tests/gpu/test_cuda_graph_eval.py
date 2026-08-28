import pytest
import torch

from qwen3_runtime.engine.cuda_graph import evaluate_decode_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph evaluation needs NVIDIA")
def test_cuda_graph_probe_matches_eager_and_does_not_auto_keep():
    result = evaluate_decode_graph()
    assert result.logits_match
    assert result.keep is False
    assert "Nsight" in result.reason or "evaluation" in result.reason
