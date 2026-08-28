import pytest
import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner, PytorchEagerRunner
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.sampling import SamplingParams, make_generator
from qwen3_runtime.spec_decode import propose_ngram, trim_emitted, verify_greedy, verify_sampled
from tests.cpu.test_engine_model import _sequential_greedy, tiny_config


def test_ngram_proposes_tokens_after_earliest_longest_suffix_match():
    # Suffix [1, 2] of length 2 matches at the start; draft the following ids.
    assert propose_ngram([1, 2, 3, 1, 2], min_ngram=2, max_ngram=4, k=2) == [3, 1]


def test_ngram_returns_empty_when_context_is_shorter_than_min():
    assert propose_ngram([1, 2], min_ngram=3, max_ngram=4, k=4) == []


def test_ngram_prefers_longer_match_inside_the_window():
    tokens = [9, 8, 7, 1, 2, 3, 1, 2, 3]
    # Longest match of the suffix [1,2,3] is n=3; draft after the first occurrence.
    assert propose_ngram(tokens, min_ngram=2, max_ngram=4, k=2) == [1, 2]


def test_verify_greedy_all_accepted_appends_bonus():
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 4.0],
            [9.0, 0.0, 0.0],
        ]
    )
    assert verify_greedy([1, 2], logits) == [1, 2, 0]


def test_verify_greedy_rejects_at_first_mismatch_and_emits_argmax():
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],
            [7.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
        ]
    )
    assert verify_greedy([1, 2], logits) == [1, 0]


def test_verify_greedy_zero_accept_emits_recovered_argmax():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert verify_greedy([1], logits) == [0]


def test_sampled_distribution_matches_target_at_live_temperature():
    q = torch.tensor([0.7, 0.3])
    temperature = 0.6
    logits = (temperature * torch.log(q)).repeat(2, 1)
    params = SamplingParams(temperature=temperature)
    n = 4000
    counts = [0, 0]
    for i in range(n):
        emitted = verify_sampled([0], logits, params, make_generator(20_000 + i))
        counts[emitted[0]] += 1
    freq = [c / n for c in counts]
    err = max(abs(freq[0] - 0.7), abs(freq[1] - 0.3))
    tv = 0.5 * (abs(freq[0] - 0.7) + abs(freq[1] - 0.3))
    assert err < 0.04, {"freq": freq, "max_abs_err": err, "tv": tv, "n": n}


def test_trim_emitted_stops_on_eos_and_stop_ids():
    req = Request([1, 2], max_tokens=8, ignore_eos=False, stop_token_ids=(9,))
    req.spec_draft_len = 4
    req.token_ids.extend([0, 0, 0, 0])
    assert trim_emitted(req, [3, 5, 7, 8], eos_token_id=5) == [3, 5]
    req2 = Request([1, 2], max_tokens=8, ignore_eos=False, stop_token_ids=(9,))
    req2.spec_draft_len = 3
    req2.token_ids.extend([0, 0, 0])
    assert trim_emitted(req2, [4, 9, 1], eos_token_id=5) == [4, 9]


def test_engine_requires_run_logits_when_speculation_is_enabled():
    with pytest.raises(RuntimeError, match="run_logits"):
        Engine(Config(num_speculative_tokens=4), object())


def test_sampled_top_k_one_matches_argmax():
    """top_k=1 is supported; Leviathan must still emit the unique remaining token."""
    logits = torch.tensor([[0.1, 4.0, 0.2], [1.0, 0.0, 0.0]])
    params = SamplingParams(temperature=1.0, top_k=1)
    for i in range(32):
        emitted = verify_sampled([0], logits, params, make_generator(3000 + i))
        assert emitted[0] == 1


def test_sampled_distribution_matches_target_on_a_tiny_vocab():
    """Leviathan with p=1_{draft}: P(emit draft)=q(draft); else residual.

    Target q = [0.7, 0.3]. Draft always proposes 0. Empirical frequencies must
    stay close to q, not to a point mass on the draft.
    """
    q = torch.tensor([0.7, 0.3])
    logits = torch.log(q).repeat(2, 1)  # draft row + bonus row
    params = SamplingParams(temperature=1.0)
    n = 4000
    counts = [0, 0]
    for i in range(n):
        gen = make_generator(10_000 + i)
        emitted = verify_sampled([0], logits, params, gen)
        counts[emitted[0]] += 1
    freq = [c / n for c in counts]
    err = max(abs(freq[0] - 0.7), abs(freq[1] - 0.3))
    tv = 0.5 * (abs(freq[0] - 0.7) + abs(freq[1] - 0.3))
    assert err < 0.04, {"freq": freq, "max_abs_err": err, "tv": tv, "n": n}


def test_truncate_kv_returns_trailing_blocks_to_the_pool():
    bm = BlockManager(num_blocks=8, block_size=4)
    req = Request(list(range(10)), max_tokens=8)
    bm.allocate_for_tokens(req, 10)
    assert len(req.block_table) == 3
    free_after_alloc = bm.num_free_blocks
    bm.truncate_kv(req, 5)
    assert len(req.block_table) == 2
    assert bm.num_free_blocks == free_after_alloc + 1


def test_config_rejects_inverted_ngram_window():
    with pytest.raises(ValueError, match="ngram"):
        Config(ngram_min=4, ngram_max=2)


def _engine_with_spec(model, *, spec: int, paged: bool, block_size: int = 4) -> Engine:
    cfg = Config(
        block_size=block_size,
        num_kv_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        num_speculative_tokens=spec,
        ngram_min=2,
        ngram_max=4,
        eos_token_id=31,
    )
    runner = PagedRunner(model) if paged else PytorchEagerRunner(model)
    return Engine(cfg, runner)


def test_greedy_spec_matches_autoregressive_on_short_and_repeated_prompts():
    torch.manual_seed(21)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompts = [
        [1, 5, 9],
        [1, 2, 1, 2, 1, 2, 1, 2],
        list(range(1, 20)),
    ]
    for prompt in prompts:
        expected = _sequential_greedy(model, prompt, max_tokens=8)
        for paged in (False, True):
            got = _engine_with_spec(model, spec=4, paged=paged).generate(prompt, max_tokens=8)
            assert got == expected, {"prompt": prompt, "paged": paged, "got": got, "expected": expected}


def test_greedy_spec_hold_kv_resume_matches_full_prefill():
    torch.manual_seed(22)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 2, 3, 4, 1, 2, 3]
    suffix = [5, 6]
    baseline = Engine(
        Config(block_size=4, num_kv_blocks=64, max_num_seqs=2, max_num_batched_tokens=32),
        PagedRunner(model),
    )
    gen1 = baseline.generate(prompt, max_tokens=5)
    gen2 = Engine(
        Config(block_size=4, num_kv_blocks=64, max_num_seqs=2, max_num_batched_tokens=32),
        PagedRunner(model),
    ).generate(prompt + gen1 + suffix, max_tokens=4)

    held = _engine_with_spec(model, spec=4, paged=True)
    rid = held.add_request(prompt, max_tokens=5, hold_kv=True)
    assert held.drain_request(rid) == gen1
    req = held._requests[rid]
    assert req.status == RequestStatus.PAUSED
    assert req.num_computed_tokens == len(req.token_ids) - 1
    held.resume_request(rid, suffix, 4, hold_kv=False)
    assert held.drain_request(rid) == gen2
    assert req.status == RequestStatus.FINISHED


def test_teacher_force_spec_emits_recorded_tokens_and_keeps_session_prefix():
    torch.manual_seed(23)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = _engine_with_spec(model, spec=4, paged=True)
    forced = [7, 8, 9, 10]
    rid = engine.add_request([1, 2, 3], max_tokens=4, hold_kv=True, forced_tokens=forced)
    assert engine.drain_request(rid) == forced
    req = engine._requests[rid]
    assert req.status == RequestStatus.PAUSED
    assert req.token_ids == [1, 2, 3] + forced
    engine.resume_request(rid, [11], 2, hold_kv=False, forced_tokens=[12, 13])
    assert engine.drain_request(rid) == [12, 13]


def test_eos_inside_spec_window_stops_deterministically():
    class EosSpecRunner:
        def run(self, reqs):
            return [
                1
                if req.num_computed_tokens + req.num_scheduled_tokens >= len(req.token_ids)
                else None
                for req in reqs
            ]

        def run_logits(self, reqs):
            rows = sum(req.num_scheduled_tokens for req in reqs)
            logits = torch.zeros((rows, 8))
            logits[:, 5] = 10.0
            return logits

    cfg = Config(
        block_size=4,
        num_kv_blocks=32,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        num_speculative_tokens=4,
        eos_token_id=5,
    )
    engine = Engine(cfg, EosSpecRunner())

    got = engine.generate([1, 2, 1, 2], max_tokens=8, ignore_eos=False)

    assert got == [1, 5]
