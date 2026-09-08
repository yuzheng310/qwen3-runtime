"""§5.2: after a weight update every KV block is invalid; the gate is not silent."""

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_session_kv import tiny_config
import torch


def _cfg() -> Config:
    return Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)


def _perturb(model: Qwen3ForCausalLM, delta: float = 4.0) -> list[tuple[str, torch.Tensor]]:
    items: list[tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            tensor = param.detach().clone()
            tensor.add_(delta)
            items.append((name, tensor))
    return items


def test_fresh_greedy_after_update_matches_from_scratch_load():
    torch.manual_seed(7)
    w0 = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4]
    engine = Engine(_cfg(), PagedRunner(w0))
    items = _perturb(w0)
    engine.apply_named_weights(items)

    got = engine.generate(prompt, max_tokens=4)

    w1 = Qwen3ForCausalLM(tiny_config()).eval()
    w1.load_state_dict(w0.state_dict())
    scratch = Engine(_cfg(), PagedRunner(w1)).generate(prompt, max_tokens=4)
    assert got == scratch


def test_stale_kv_surviving_an_update_fails_the_token_gate():
    """A leftover block table is the silent training bug. Catch it directly.

    A *fresh* generate after an update can still match from-scratch even when a
    paused session's pages remain (the 5.2 gate is a new request). The test that
    fails when invalidation is skipped is: those pages are still allocated.
    """
    torch.manual_seed(8)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4]
    cfg = _cfg()
    held = Engine(cfg, PagedRunner(model))
    rid = held.add_request(prompt, max_tokens=3, hold_kv=True)
    held.drain_request(rid)
    assert held._requests[rid].block_table
    items = _perturb(model, delta=6.0)
    held.apply_named_weights(items, invalidate_kv=False)
    assert held._requests[rid].block_table, "token-for-token on a fresh request would miss these leftover pages"
    held.apply_named_weights(_perturb(model, delta=0.0), invalidate_kv=True)
    assert held._requests[rid].block_table == []
    assert held._requests[rid].num_computed_tokens == 0



def test_stale_block_table_after_update_fails_resume():
    torch.manual_seed(10)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = Engine(_cfg(), PagedRunner(model))
    rid = engine.add_request([1, 2, 3, 4], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    engine.apply_named_weights(_perturb(model, delta=1.0))
    req = engine._requests[rid]
    req.block_table = [0]
    req.num_computed_tokens = 0
    import pytest

    with pytest.raises(RuntimeError, match="block table after KV invalidation"):
        engine.resume_request(rid, [8, 9], 2, hold_kv=False)


def test_prefix_cache_survives_only_when_invalidate_is_skipped():
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    cfg = Config(
        block_size=4,
        num_kv_blocks=32,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        enable_prefix_cache=True,
    )
    engine = Engine(cfg, PagedRunner(model))
    engine.generate([1, 2, 3, 4], max_tokens=2)
    assert engine.block_manager.cache_blocks > 0
    engine.apply_named_weights(_perturb(model, delta=1.0), invalidate_kv=False)
    assert engine.block_manager.cache_blocks > 0
    engine.apply_named_weights(_perturb(model, delta=0.0), invalidate_kv=True)
    assert engine.block_manager.cache_blocks == 0


def test_resume_across_weight_update_reprefills_not_stale_table():
    """§5.2 amended gate: resume is the only path that can read pre-update KV.

    A fresh generate cannot catch leftover pages (U2). This resume must either
    re-prefill or fail loudly. Silently keeping the pre-update table is a fail.
    """
    torch.manual_seed(12)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4]
    suffix = [5, 6]
    cfg = _cfg()
    engine = Engine(cfg, PagedRunner(model))
    rid = engine.add_request(prompt, max_tokens=3, hold_kv=True)
    gen1 = engine.drain_request(rid)
    req = engine._requests[rid]
    pre_table = list(req.block_table)
    pre_computed = req.num_computed_tokens
    assert pre_table and pre_computed > 0

    engine.apply_named_weights(_perturb(model, delta=1.5))

    req = engine._requests[rid]
    assert req.block_table == [], f"paused session kept pre-update table {req.block_table}"
    assert req.num_computed_tokens == 0

    engine.resume_request(rid, suffix, 4, hold_kv=False)
    gen2 = engine.drain_request(rid)

    w1 = Qwen3ForCausalLM(tiny_config()).eval()
    w1.load_state_dict(model.state_dict())
    scratch = Engine(cfg, PagedRunner(w1)).generate(prompt + gen1 + suffix, max_tokens=4)
    assert gen2 == scratch


def test_resume_with_preupdate_table_is_refused_not_silent():
    """If drop-all misses the paused table, resume must not skip re-prefill."""
    torch.manual_seed(13)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = Engine(_cfg(), PagedRunner(model))
    rid = engine.add_request([1, 2, 3, 4], max_tokens=3, hold_kv=True)
    engine.drain_request(rid)
    leftover = list(engine._requests[rid].block_table)
    assert leftover
    engine.apply_named_weights(_perturb(model, delta=2.0), invalidate_kv=False)
    # Simulate a missed drop-all: leftover table, generation advanced.
    # block_manager.epoch is the only authority; do not poke a second counter.
    engine.block_manager.epoch += 1
    req = engine._requests[rid]
    assert req.block_table == leftover
    assert req.num_computed_tokens > 0
    import pytest

    with pytest.raises(RuntimeError, match="block table after KV invalidation"):
        engine.resume_request(rid, [8, 9], 2, hold_kv=False)


def test_weight_update_drops_every_block():
    torch.manual_seed(9)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = Engine(_cfg(), PagedRunner(model))
    rid = engine.add_request([1, 2, 3, 4], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    engine.apply_named_weights(_perturb(model, delta=1.0))
    req = engine._requests[rid]
    assert req.block_table == []
    assert req.num_computed_tokens == 0
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
