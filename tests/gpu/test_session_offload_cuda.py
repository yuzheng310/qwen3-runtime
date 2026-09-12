from __future__ import annotations

import pytest
import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig
from qwen3_runtime.kv.paged import PagedKVPool


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _cuda_tiny_config() -> Qwen3ModelConfig:
    # FlashInfer's CUDA RoPE implementation requires a supported head_dim.
    return Qwen3ModelConfig(
        vocab_size=64,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


def test_cuda_sync_pause_offload_restore_roundtrip():
    model = (
        Qwen3ForCausalLM(_cuda_tiny_config())
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    engine = Engine(
        Config(
            block_size=4,
            num_kv_blocks=16,
            max_num_seqs=2,
            max_num_batched_tokens=32,
            enable_prefix_cache=False,
            session_cpu_offload="sync",
            cpu_kv_max_bytes=1024 * 1024,
            transfer_chunk_bytes=128,
        ),
        PagedRunner(model),
    )
    request_id = engine.add_request([1, 2, 3, 4, 5, 6], max_tokens=1, hold_kv=True)
    first = engine.drain_request(request_id)
    request = engine._requests[request_id]

    engine.offload_request(request_id, session_key="cuda-test")
    assert request.kv_residency == "cpu"
    assert request.block_table == []
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks

    engine.resume_request(request_id, [7, 8], 1, hold_kv=False, forced_tokens=[21])
    second = engine.drain_request(request_id)
    torch.cuda.synchronize()

    assert first
    assert second == [21]
    assert engine.session_offload.report()["saved"] == 1
    assert engine.session_offload.report()["restored"] == 1


@pytest.mark.parametrize("pinned", [False, True])
def test_cuda_transfer_fragmented_blocks_and_partial_apc_restore(pinned):
    pool = PagedKVPool(3, 19, 16, 2, 128, torch.bfloat16, torch.device("cuda"))
    pool.cache.normal_()
    ids = [13, 2, 15, 1, 18]
    host = torch.empty(
        (5, 2, 3, 16, 2, 128), dtype=torch.bfloat16, pin_memory=pinned
    ).permute(1, 2, 0, 3, 4, 5)
    expected = pool.cache[:, :, ids].cpu()
    expected[:, :, -1, 7:].zero_()
    pool.export_blocks(
        ids, valid_tokens=71, chunk_bytes=2 * pool.block_bytes, destination=host
    )
    assert torch.equal(host, expected)
    target = [3, 9, 0]
    pool.import_blocks(
        target,
        host,
        valid_tokens=71,
        source_block_offset=2,
        chunk_bytes=2 * pool.block_bytes,
    )
    assert torch.equal(pool.cache[:, :, target].cpu(), expected[:, :, 2:])


@pytest.mark.parametrize("spec", [0, 16])
def test_cuda_offload_matches_held_greedy_tokens_and_logprobs(spec):
    torch.manual_seed(193)
    model = (
        Qwen3ForCausalLM(_cuda_tiny_config(), attention_backend="flashinfer")
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    outputs = []
    for offload in [False, True]:
        engine = Engine(
            Config(
                block_size=16,
                num_kv_blocks=16,
                max_num_seqs=1,
                max_num_batched_tokens=128,
                num_speculative_tokens=spec,
                session_cpu_offload="sync",
                cpu_kv_max_bytes=4 * 1024**2,
                cpu_kv_pinned_max_bytes=4 * 1024**2,
                transfer_chunk_bytes=128 * 1024,
            ),
            PagedRunner(model),
        )
        rid = engine.add_request(list(range(1, 48)), max_tokens=4, hold_kv=True)
        first = engine.drain_request(rid)
        if offload:
            engine.offload_request(rid)
        engine.resume_request(rid, [8, 9, 10], 8, hold_kv=False)
        second = engine.drain_request(rid)
        outputs.append((first, second, engine.last_completion_logprobs[rid]))
    assert outputs[0] == outputs[1]


def test_decode_graph_page_buffer_covers_shared_logical_references():
    from dataclasses import replace

    cfg = replace(_cuda_tiny_config(), max_position_embeddings=256)
    model = (
        Qwen3ForCausalLM(cfg, attention_backend="flashinfer")
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    engine = Engine(
        Config(
            block_size=16,
            num_kv_blocks=12,
            max_num_seqs=2,
            max_num_batched_tokens=256,
            enable_prefix_cache=True,
        ),
        PagedRunner(model, cuda_graph=True),
    )
    prompt = [i % 60 + 1 for i in range(128)]
    engine.generate(prompt, max_tokens=1)
    ids = [engine.add_request(prompt, max_tokens=4) for _ in range(2)]
    got = {rid: [] for rid in ids}
    while engine.scheduler.waiting or engine.scheduler.running:
        for rid, tok, _ in engine.step():
            if rid in got and tok is not None:
                got[rid].append(tok)
    assert got[ids[0]] == got[ids[1]]
    assert len(got[ids[0]]) == 4
    assert 2 in engine.runner._decode_graphs
