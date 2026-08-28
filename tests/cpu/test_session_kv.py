from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.engine.model_runner import PagedRunner, PytorchEagerRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig
from qwen3_runtime.sampling import SamplingParams
from tests.cpu.test_engine import _engine
import torch


def tiny_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


class ConstTokenRunner:
    def __init__(self, token: int):
        self.token = token

    def run(self, reqs):
        out = []
        for req in reqs:
            done = req.num_computed_tokens + req.num_scheduled_tokens >= len(req.token_ids)
            out.append(self.token if done else None)
        return out


def test_eos_stops_and_keeps_the_stop_token():
    cfg = Config(
        max_num_batched_tokens=32,
        max_num_seqs=2,
        num_kv_blocks=32,
        block_size=4,
        eos_token_id=5,
    )
    engine = Engine(cfg, ConstTokenRunner(5))
    got = engine.generate([1, 2], max_tokens=8, ignore_eos=False)
    assert got == [5]


def test_stop_token_ids_stop_without_config_eos():
    cfg = Config(max_num_batched_tokens=32, max_num_seqs=2, num_kv_blocks=32, block_size=4)
    engine = Engine(cfg, ConstTokenRunner(9))
    got = engine.generate([1, 2], max_tokens=8, ignore_eos=False, stop_token_ids=(9,))
    assert got == [9]


def test_ignore_eos_does_not_stop_on_eos_id():
    cfg = Config(
        max_num_batched_tokens=32,
        max_num_seqs=2,
        num_kv_blocks=32,
        block_size=4,
        eos_token_id=5,
    )
    engine = Engine(cfg, ConstTokenRunner(5))
    got = engine.generate([1, 2], max_tokens=4, ignore_eos=True)
    assert got == [5, 5, 5, 5]


def test_forced_tokens_are_what_step_reports():
    engine = _engine()
    rid = engine.add_request([1, 2], max_tokens=3, forced_tokens=[11, 12, 13])
    got = engine.drain_request(rid)
    assert got == [11, 12, 13]


def test_hold_kv_resume_matches_full_prefill_greedy_paged():
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4]
    suffix = [5, 6]
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)

    gen1 = Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=3)
    full = prompt + gen1 + suffix
    gen2_full = Engine(cfg, PagedRunner(model)).generate(full, max_tokens=2)

    held = Engine(cfg, PagedRunner(model))
    rid = held.add_request(prompt, max_tokens=3, hold_kv=True)
    gen1_b = held.drain_request(rid)
    req = held._requests[rid]
    assert req.status == RequestStatus.PAUSED
    assert gen1_b == gen1
    assert req.token_ids == prompt + gen1
    assert req.num_computed_tokens == len(req.token_ids) - 1
    positions_before = req.num_computed_tokens
    held.resume_request(rid, suffix, 2, hold_kv=False)
    assert req.num_computed_tokens == positions_before
    gen2_b = held.drain_request(rid)
    assert gen2_b == gen2_full
    assert req.status == RequestStatus.FINISHED
    assert held.block_manager.num_free_blocks == held.block_manager.num_blocks


def test_hold_kv_resume_matches_eager_full_prefill():
    torch.manual_seed(12)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [2, 4, 6]
    suffix = [7, 8, 9]
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    gen1 = Engine(cfg, PytorchEagerRunner(model)).generate(prompt, max_tokens=2)
    gen2_full = Engine(cfg, PytorchEagerRunner(model)).generate(prompt + gen1 + suffix, max_tokens=2)
    held = Engine(cfg, PytorchEagerRunner(model))
    rid = held.add_request(prompt, max_tokens=2, hold_kv=True)
    assert held.drain_request(rid) == gen1
    held.resume_request(rid, suffix, 2, hold_kv=False)
    assert held.drain_request(rid) == gen2_full


def test_finish_request_releases_paused_kv():
    engine = _engine(num_blocks=16, block_size=4)
    free0 = engine.block_manager.num_free_blocks
    rid = engine.add_request([1, 2, 3, 4], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    assert engine.block_manager.num_free_blocks < free0
    engine.finish_request(rid)
    assert engine.block_manager.num_free_blocks == free0
    assert rid not in engine.scheduler.paused


def test_two_sessions_do_not_leak_kv_or_tokens():
    engine = _engine()
    a = engine.add_request([1, 2], max_tokens=2, hold_kv=True, forced_tokens=[11, 12])
    b = engine.add_request([30, 31, 32], max_tokens=2, hold_kv=True, forced_tokens=[41, 42])
    while engine._requests[a].status != RequestStatus.PAUSED or engine._requests[b].status != RequestStatus.PAUSED:
        engine.step()
    blocks_a = list(engine._requests[a].block_table)
    blocks_b = list(engine._requests[b].block_table)
    assert set(blocks_a).isdisjoint(set(blocks_b))
    engine.resume_request(a, [13], 1, hold_kv=False, forced_tokens=[14])
    assert engine.drain_request(a) == [14]
    assert engine._requests[b].status == RequestStatus.PAUSED
    assert engine._requests[b].token_ids[-2:] == [41, 42]
    engine.finish_request(b)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_paused_session_consumes_no_decode_slot_while_peer_decodes():
    engine = _engine(max_seqs=1, token_budget=32)
    paused = engine.add_request([1, 2, 3], max_tokens=1, hold_kv=True, forced_tokens=[9])
    engine.drain_request(paused)
    assert engine._requests[paused].status == RequestStatus.PAUSED
    peer = engine.add_request([4, 5], max_tokens=2, hold_kv=False, forced_tokens=[7, 8])
    assert engine.drain_request(peer) == [7, 8]
    assert engine._requests[paused].status == RequestStatus.PAUSED
    engine.finish_request(paused)


def test_seeded_engine_sampling_is_deterministic_on_tiny_model():
    torch.manual_seed(13)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 5, 9]
    params = SamplingParams(temperature=0.8, top_p=0.9, top_k=8, seed=42)
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    a = Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=5, sampling=params)
    b = Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=5, sampling=params)
    assert a == b
    greedy = Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=5)
    # Temperature sampling is allowed to match greedy, but the seed path must be stable.
    assert len(a) == 5 and len(greedy) == 5
