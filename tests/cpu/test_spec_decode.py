import json
import math

import pytest
import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.engine.scheduler import Scheduler
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from qwen3_runtime.reference.eager_runner import PytorchEagerRunner
from qwen3_runtime.reference.propose_ngram import propose_ngram
from qwen3_runtime.sampling import SamplingParams, make_generator, record_sampled
from qwen3_runtime.spec_decode import (
    NgramIndex,
    _batch_target_probs,
    trim_emitted,
    verify_greedy,
    verify_sampled,
)
from tests.cpu.test_engine import NullLifecycleRunner
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


def test_ngram_index_matches_propose_ngram_on_random_and_unit_cases():
    from qwen3_runtime.spec_decode import NgramIndex

    cases = [
        [1, 2, 3, 1, 2],
        [9, 8, 7, 1, 2, 3, 1, 2, 3],
        [1, 2],
        [7, 7, 7, 7, 7],
        list(range(20)),
    ]
    rng = torch.Generator()
    rng.manual_seed(20260901)
    for _ in range(80):
        n = int(torch.randint(4, 64, (1,), generator=rng).item())
        cases.append(torch.randint(0, 11, (n,), generator=rng).tolist())
    for tokens in cases:
        for min_n, max_n, k in ((2, 4, 16), (2, 4, 2), (1, 3, 8)):
            ref = propose_ngram(tokens, min_n, max_n, k)
            idx = NgramIndex(min_n, max_n)
            idx.extend(tokens)
            got = idx.propose(k)
            assert got == ref, {"tokens": tokens, "ref": ref, "got": got, "minmaxk": (min_n, max_n, k)}
            # Incremental catch-up equals a full rebuild.
            idx2 = NgramIndex(min_n, max_n)
            mid = max(1, len(tokens) // 2)
            idx2.extend(tokens[:mid])
            idx2.extend(tokens[mid:])
            assert idx2.propose(k) == ref


def test_ngram_index_matches_propose_ngram_on_frozen_replay_prefixes():
    """Plan §4: drafts identical to propose_ngram on frozen Replay prefixes.

    Public traces omit generated output ids. Replay input_ids for the Phase 1
    eight instance_ids are the frozen contexts that exist in-tree.
    """
    from pathlib import Path

    from qwen3_runtime.spec_decode import NgramIndex

    phase1_ids = {
        "django__django-14534",
        "django__django-11532",
        "django__django-12308",
        "pytest-dev__pytest-10051",
        "django__django-16595",
        "sympy__sympy-21379",
        "sympy__sympy-20590",
        "django__django-15277",
    }
    path = Path(__file__).resolve().parents[2] / "workloads/code_localization/token_ids/replay_subset_v1.jsonl"
    if not path.is_file():
        pytest.skip("optional frozen replay token corpus is not distributed")
    n_checked = 0
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("task_id") not in phase1_ids:
                continue
            tokens = row["input_ids"]
            n = len(tokens)
            cuts = sorted({max(4, n * p // 100) for p in (25, 50, 75, 90, 100)})
            for cut in cuts:
                prefix = tokens[:cut]
                ref = propose_ngram(prefix, 2, 4, 16)
                idx = NgramIndex(2, 4)
                idx.extend(prefix)
                assert idx.propose(16) == ref
                n_checked += 1
    assert n_checked >= 8 * 4


def test_verify_greedy_all_accepted_appends_bonus():
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 4.0],
            [9.0, 0.0, 0.0],
        ]
    )
    tokens, logprobs = verify_greedy([1, 2], logits)
    assert tokens == [1, 2, 0]
    assert len(logprobs) == 3


def test_verify_greedy_rejects_at_first_mismatch_and_emits_argmax():
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],
            [7.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
        ]
    )
    tokens, logprobs = verify_greedy([1, 2], logits)
    assert tokens == [1, 0]
    assert len(logprobs) == 2


def test_verify_greedy_zero_accept_emits_recovered_argmax():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tokens, logprobs = verify_greedy([1], logits)
    assert tokens == [0]
    assert len(logprobs) == 1


def test_sampled_distribution_matches_target_at_live_temperature():
    q = torch.tensor([0.7, 0.3])
    temperature = 0.6
    logits = (temperature * torch.log(q)).repeat(2, 1)
    params = SamplingParams(temperature=temperature)
    n = 4000
    counts = [0, 0]
    for i in range(n):
        emitted, _ = verify_sampled([0], logits, params, make_generator(20_000 + i))
        counts[emitted[0]] += 1
    freq = [c / n for c in counts]
    err = max(abs(freq[0] - 0.7), abs(freq[1] - 0.3))
    tv = 0.5 * (abs(freq[0] - 0.7) + abs(freq[1] - 0.3))
    assert err < 0.04, {"freq": freq, "max_abs_err": err, "tv": tv, "n": n}


def test_trim_emitted_stops_on_eos_and_stop_ids():
    req = Request([1, 2], max_tokens=8, ignore_eos=False, stop_token_ids=(9,))
    req.spec_draft_len = 4
    req.token_ids.extend([0, 0, 0, 0])
    assert trim_emitted(req, [3, 5, 7, 8], [-1.0, -2.0, -3.0, -4.0], eos_token_id=5) == (
        [3, 5],
        [-1.0, -2.0],
    )
    req2 = Request([1, 2], max_tokens=8, ignore_eos=False, stop_token_ids=(9,))
    req2.spec_draft_len = 3
    req2.token_ids.extend([0, 0, 0])
    assert trim_emitted(req2, [4, 9, 1], [-1.0, -2.0, -3.0], eos_token_id=5) == (
        [4, 9],
        [-1.0, -2.0],
    )


def test_trim_emitted_refuses_tokens_it_cannot_score():
    req = Request([1, 2], max_tokens=8)
    req.spec_draft_len = 2
    req.token_ids.extend([0, 0])
    with pytest.raises(ValueError, match="logprobs"):
        trim_emitted(req, [3, 4], [-1.0], eos_token_id=None)


def test_model_runner_requires_run_logits():
    from qwen3_runtime.engine.runner import ModelRunner

    assert "run_logits" in ModelRunner.__abstractmethods__
    with pytest.raises(TypeError):
        ModelRunner()


def test_sampled_top_k_one_matches_argmax():
    """top_k=1 is supported; Leviathan must still emit the unique remaining token."""
    logits = torch.tensor([[0.1, 4.0, 0.2], [1.0, 0.0, 0.0]])
    params = SamplingParams(temperature=1.0, top_k=1)
    for i in range(32):
        emitted, _ = verify_sampled([0], logits, params, make_generator(3000 + i))
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
        emitted, _ = verify_sampled([0], logits, params, gen)
        counts[emitted[0]] += 1
    freq = [c / n for c in counts]
    err = max(abs(freq[0] - 0.7), abs(freq[1] - 0.3))
    tv = 0.5 * (abs(freq[0] - 0.7) + abs(freq[1] - 0.3))
    assert err < 0.04, {"freq": freq, "max_abs_err": err, "tv": tv, "n": n}


def test_sampled_reports_the_target_density_not_the_residuals():
    """A rejected draft is redrawn from the residual, but the policy is the target.

    Speculative sampling is distribution-preserving, so the emitted token is
    distributed as ``q`` whichever branch produced it, and ``log q(token)`` is
    the behaviour logprob a trainer needs. Reporting the residual's own density
    would describe a distribution the policy never sampled from -- and it is the
    branch that is easy to get wrong, because the residual is what the code
    literally draws from.
    """
    q = torch.tensor([0.7, 0.3])
    logits = torch.log(q).repeat(2, 1)
    params = SamplingParams(temperature=1.0)
    seen = set()
    for i in range(400):
        emitted, logprobs = verify_sampled([0], logits, params, make_generator(50_000 + i))
        assert len(emitted) == len(logprobs)
        for tok, lp in zip(emitted, logprobs):
            seen.add(tok)
            assert lp == pytest.approx(math.log(float(q[tok])), abs=1e-5)
    assert seen == {0, 1}, "both the accept and the reject branch must be exercised"


def test_verifier_honours_the_knobs_the_row_sampler_honours():
    """min_p / bias / penalties reach the verifier, or the two paths disagree.

    The verifier used to run its own short chain -- temperature, top-k, top-p --
    so any other knob silently applied on one decode path and not the other.
    """
    logits = torch.tensor([[0.0, 3.0, 2.9, -5.0], [0.0, 0.0, 0.0, 0.0]])

    masked = SamplingParams(temperature=1.0, min_p=0.9)
    probs = _batch_target_probs(logits, masked)
    assert float(probs[0, 3]) == 0.0, "min_p must exclude the tail on the verify path too"

    biased = SamplingParams(temperature=1.0, logit_bias={3: 50.0})
    assert int(_batch_target_probs(logits, biased)[0].argmax()) == 3

    penalised = SamplingParams(temperature=1.0, repetition_penalty=100.0)
    prefixes = [[1], [1, 1]]
    penalised_probs = _batch_target_probs(logits, penalised, prefixes)
    assert float(penalised_probs[0, 1]) < float(_batch_target_probs(logits, SamplingParams(temperature=1.0))[0, 1])


def test_top_logprobs_turns_speculation_off_rather_than_shipping_less():
    """The verifier has no top-k list, so a request that wants one is not drafted."""
    cfg = Config(
        block_size=4,
        num_kv_blocks=32,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        num_speculative_tokens=4,
        ngram_min=2,
        ngram_max=4,
    )
    sched = Scheduler(cfg, BlockManager(num_blocks=32, block_size=4))
    req = Request([1, 2, 1, 2], max_tokens=8, sampling=SamplingParams(top_logprobs=3))
    req.num_computed_tokens = len(req.token_ids) - 1
    sched._maybe_attach_spec_drafts(req, token_budget=32)
    assert req.spec_draft_len == 0


def _generate_with_logprobs(engine, prompt, max_tokens):
    rid = engine.add_request(list(prompt), max_tokens=max_tokens)
    tokens = list(engine.stream_request(rid))
    logprobs = next(iter(engine.last_completion_logprobs.values()))
    return tokens, logprobs


def test_greedy_spec_logprobs_match_the_autoregressive_path_token_for_token():
    """Speculation is a speed change, so the reported logprobs must not move.

    Same tokens was already gated; same *logprobs* was not, and the verifier
    returned no logprobs at all -- a burst of N tokens reached the adapter with
    one value, and the missing positions were filled with 0.0.
    """
    torch.manual_seed(24)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    for prompt in ([1, 5, 9], [1, 2, 1, 2, 1, 2, 1, 2], list(range(1, 20))):
        base_tokens, base_lps = _generate_with_logprobs(
            _engine_with_spec(model, spec=0, paged=True), prompt, 8
        )
        spec_tokens, spec_lps = _generate_with_logprobs(
            _engine_with_spec(model, spec=4, paged=True), prompt, 8
        )
        assert spec_tokens == base_tokens
        assert len(spec_lps) == len(spec_tokens)
        assert spec_lps == pytest.approx(base_lps, abs=1e-4), {"prompt": prompt}


def test_every_spec_burst_reports_one_logprob_per_token():
    """The invariant, at the seam the adapter reads: one logprob per token, per step."""
    torch.manual_seed(25)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    for params in (SamplingParams(), SamplingParams(temperature=0.8, top_p=0.9, seed=7)):
        engine = _engine_with_spec(model, spec=4, paged=True)
        rid = engine.add_request([1, 2, 1, 2, 1, 2], max_tokens=12, sampling=params)
        bursts = 0
        while engine._requests.get(rid) is not None and not engine.is_finished():
            engine.step()
            emitted = engine.last_emitted.get(rid, [])
            assert len(engine.last_emitted_logprobs.get(rid, [])) == len(emitted)
            bursts += len(emitted) > 1
        assert bursts, "no multi-token step happened; this would pass vacuously"


def test_a_requests_logprob_record_stays_as_long_as_its_output():
    """``req.logprobs`` must cover every generated token, spec bursts included.

    A step that finds no n-gram match runs the ordinary sampler, which reads
    the tail of this list. If a speculative burst appends tokens without
    appending their logprobs, the next ordinary step reports the logprob of
    whatever preceded the burst -- stale, plausible, and invisible.
    """
    torch.manual_seed(27)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    for params in (SamplingParams(), SamplingParams(temperature=0.8, seed=3)):
        engine = _engine_with_spec(model, spec=4, paged=True)
        rid = engine.add_request([1, 2, 1, 2, 1, 2], max_tokens=10, hold_kv=True, sampling=params)
        engine.drain_request(rid)
        req = engine._requests[rid]
        assert len(req.logprobs) == len(req.token_ids) - req.num_prompt_tokens


def test_teacher_forced_tokens_report_nan_rather_than_a_plausible_zero():
    """A replayed token has no sampling-time logprob. Say so; do not invent one.

    ``run_logits`` skips the forward outright when every scheduled request is
    teacher-forced, so those positions have no distribution to be scored
    against. NaN is what they get: it keeps one logprob per token, and unlike
    the 0.0 that used to be padded in downstream it cannot be mistaken for a
    token the policy was certain about.
    """
    torch.manual_seed(26)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = _engine_with_spec(model, spec=4, paged=True)
    forced = [7, 8, 9, 10]
    rid = engine.add_request([1, 2, 3], max_tokens=4, forced_tokens=forced)
    tokens = list(engine.stream_request(rid))
    logprobs = next(iter(engine.last_completion_logprobs.values()))
    assert tokens == forced
    assert len(logprobs) == len(tokens)
    assert any(math.isnan(lp) for lp in logprobs), "an unscored token must say so"
    assert all(math.isnan(lp) or lp <= 0.0 for lp in logprobs)


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
    class EosSpecRunner(NullLifecycleRunner):
        def run(self, reqs):
            out = []
            for req in reqs:
                done = req.num_computed_tokens + req.num_scheduled_tokens >= len(req.token_ids)
                out.append(1 if done else None)
                if done:
                    record_sampled(req, -0.5)
            return out

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
