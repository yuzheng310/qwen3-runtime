import math

import pytest
import torch

from qwen3_runtime.sampling import (
    SamplingParams,
    apply_top_k_top_p,
    chosen_logprobs,
    greedy_logits,
    make_generator,
    process_logits,
    process_logits_batch,
    sample_from_logits,
    sample_scheduled,
)


# One knob off default per case, plus one case with several at once. Speculative
# verification runs the batched chain, ordinary sampling runs the row chain, and
# nothing else in the tree compares them -- for most of this stage the batched
# side silently skipped penalties, min_p and logit bias entirely.
_CHAIN_CASES = [
    SamplingParams(),
    SamplingParams(temperature=1.0),
    SamplingParams(temperature=0.6),
    SamplingParams(temperature=1.0, top_p=0.8),
    SamplingParams(temperature=1.0, top_k=3),
    SamplingParams(temperature=1.0, min_p=0.2),
    SamplingParams(temperature=1.0, logit_bias={2: 4.0, 5: -3.0}),
    SamplingParams(temperature=1.0, repetition_penalty=1.4),
    SamplingParams(temperature=1.0, presence_penalty=0.7),
    SamplingParams(temperature=1.0, frequency_penalty=0.3),
    SamplingParams(temperature=0.0, repetition_penalty=1.4, logit_bias={1: 2.0}),
    SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=6,
        min_p=0.05,
        repetition_penalty=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.2,
        logit_bias={0: 1.5},
    ),
]


@pytest.mark.parametrize("params", _CHAIN_CASES, ids=lambda p: f"T{p.temperature}")
def test_batched_processor_chain_equals_the_row_chain(params):
    torch.manual_seed(11)
    rows = torch.randn(4, 12)
    # Each row scores a different position, so each gets its own penalty context.
    prefixes = [[1, 1, 2], [1, 1, 2, 5], [1, 1, 2, 5, 5], [1, 1, 2, 5, 5, 0]]
    batched = process_logits_batch(rows, params, prefixes)
    looped = torch.stack(
        [process_logits(row, params, prefix) for row, prefix in zip(rows, prefixes)]
    )
    assert torch.equal(batched.isneginf(), looped.isneginf())
    finite = ~batched.isneginf()
    assert torch.allclose(batched[finite], looped[finite], atol=1e-5)


def test_batched_processor_refuses_penalties_without_per_row_context():
    rows = torch.randn(3, 8)
    params = SamplingParams(temperature=1.0, repetition_penalty=1.3)
    with pytest.raises(ValueError, match="prefix per row"):
        process_logits_batch(rows, params)


def test_chosen_logprobs_matches_log_softmax_of_the_processed_row():
    torch.manual_seed(12)
    rows = torch.randn(3, 9)
    params = SamplingParams(temperature=0.8, top_p=0.9)
    processed = process_logits_batch(rows, params)
    tokens = processed.argmax(dim=-1).tolist()
    got = chosen_logprobs(processed, tokens)
    want = [
        float(torch.log_softmax(processed[i].float(), dim=-1)[tok].item())
        for i, tok in enumerate(tokens)
    ]
    assert got == pytest.approx(want, abs=1e-6)


def test_apply_top_k_top_p_batched_matches_rows():
    torch.manual_seed(0)
    rows = torch.randn(5, 17)
    batched = apply_top_k_top_p(rows, top_k=4, top_p=0.8)
    looped = torch.stack([apply_top_k_top_p(row, top_k=4, top_p=0.8) for row in rows])
    assert torch.equal(batched.isneginf(), looped.isneginf())
    finite = ~batched.isneginf()
    assert torch.allclose(batched[finite], looped[finite], atol=1e-5)



def test_greedy_picks_argmax_per_row():
    logits = torch.tensor(
        [
            [0.1, 0.7, 0.2],
            [3.0, -1.0, 2.5],
        ]
    )
    assert greedy_logits(logits) == [1, 0]


def test_greedy_ties_resolve_to_lowest_token_id():
    """§5.0 policy: equal maximal logits resolve to the lowest token id.

    torch.argmax returns the first maximal index. A different convention
    (last index, random) would invalidate a greedy token-id comparison.
    """
    tied = torch.tensor([1.0, 3.0, 3.0, 2.0])
    assert greedy_logits(tied.unsqueeze(0)) == [1]
    assert sample_from_logits(tied, SamplingParams(temperature=0.0)) == 1
    zeros = torch.zeros(8)
    assert greedy_logits(zeros.unsqueeze(0)) == [0]
    assert sample_from_logits(zeros, SamplingParams(temperature=0.0)) == 0
    python_rows = torch.tensor([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]])
    assert greedy_logits(python_rows) == [0, 0]


def test_greedy_logits_stays_on_tensor():
    rows = [
        [0.1, 0.7, 0.2],
        [3.0, -1.0, 2.5],
        [0.0, 0.0, 0.0],
    ]
    tensor = torch.tensor(rows, dtype=torch.float32)
    assert greedy_logits(tensor) == [1, 0, 0]


def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 0.7, 0.2])
    assert sample_from_logits(logits, SamplingParams(temperature=0.0)) == 1
    assert sample_from_logits(logits, SamplingParams(temperature=-1.0)) == 1


def test_top_k_one_matches_argmax_even_with_temperature():
    logits = torch.tensor([0.0, 4.0, 1.0, 3.0])
    tok = sample_from_logits(logits, SamplingParams(temperature=1.5, top_k=1, seed=0), make_generator(0))
    assert tok == 1


def test_greedy_logprob_is_log_softmax_of_chosen_id():
    from qwen3_runtime.sampling import sample_from_logits_with_logprob

    logits = torch.tensor([0.0, 4.0, 1.0, 3.0])
    tok, lp = sample_from_logits_with_logprob(logits, SamplingParams(temperature=0.0))
    assert tok == 1
    expect = float(torch.log_softmax(logits.float(), dim=-1)[1])
    assert abs(lp - expect) < 1e-5


def test_seeded_multinomial_is_deterministic():
    logits = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.8])
    params = SamplingParams(temperature=0.9, top_p=0.95, top_k=4, seed=123)
    a = sample_from_logits(logits, params, make_generator(123))
    b = sample_from_logits(logits, params, make_generator(123))
    assert a == b


def test_make_generator_stays_on_cpu():
    gen = make_generator(0)
    assert gen is not None
    assert gen.device.type == "cpu"


def test_cpu_generator_is_device_stable():
    logits = torch.randn(64, dtype=torch.float32)
    params = SamplingParams(temperature=0.8, seed=11)
    cpu_tok = sample_from_logits(logits, params, make_generator(11))
    assert cpu_tok == sample_from_logits(logits.clone(), params, make_generator(11))
    if torch.cuda.is_available():
        gpu = logits.cuda()
        a = sample_from_logits(gpu, params, make_generator(11))
        b = sample_from_logits(gpu, params, make_generator(11))
        assert a == b


class _Req:
    def __init__(self, temperature, seed):
        self.sampling = SamplingParams(temperature=temperature, seed=seed)
        self.rng = make_generator(seed)
        self.token_ids = [1]
        self.num_prompt_tokens = 1
        self.logprobs = []
        self.top_logprobs = []
        self.last_logprob = None

    def next_forced_token(self):
        return None


def test_sample_scheduled_batched_matches_per_row():
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    reqs = [_Req(0.6, seed) for seed in (1, 2, 3, 4)]
    flags = [True, False, True, True]
    sampled_rows = logits[[i for i, flag in enumerate(flags) if flag]]
    batched = sample_scheduled(sampled_rows, reqs, flags)
    per_row = []
    for req, flag, row in zip(reqs, flags, logits):
        if not flag:
            per_row.append(None)
            continue
        per_row.append(sample_from_logits(row, req.sampling, make_generator(req.sampling.seed)))
    assert batched == per_row


def test_top_p_keeps_the_heaviest_token():
    logits = torch.tensor([0.0, 10.0, -5.0, -5.0])
    filtered = apply_top_k_top_p(logits.clone(), top_k=0, top_p=0.5)
    assert int(filtered.argmax().item()) == 1
    assert torch.isneginf(filtered[0]) or filtered[0] < filtered[1]


@pytest.mark.parametrize("temperature", [math.nan, math.inf, -math.inf])
def test_sampling_rejects_non_finite_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=temperature)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, math.nan, math.inf])
def test_sampling_rejects_invalid_top_p(top_p):
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=top_p)


def test_logit_bias_selects_the_biased_id():
    from qwen3_runtime.sampling import sample_from_logits

    logits = torch.zeros(4)
    tok = sample_from_logits(logits, SamplingParams(temperature=0.0, logit_bias={3: 5.0}))
    assert tok == 3


def test_repetition_penalty_suppresses_seen_ids():
    from qwen3_runtime.sampling import sample_from_logits

    logits = torch.tensor([4.0, 1.5, 0.0])
    tok = sample_from_logits(
        logits, SamplingParams(temperature=0.0, repetition_penalty=10.0), token_ids=[0]
    )
    assert tok == 1


def test_min_p_masks_tokens_below_the_peak_fraction():
    from qwen3_runtime.sampling import apply_min_p

    logits = torch.tensor([8.0, 0.0, -4.0])
    filtered = apply_min_p(logits, 0.9)
    assert torch.isneginf(filtered[2])


def test_forced_token_is_applied_at_the_chain_tail():
    req = _Req(0.0, 0)
    req.forced_tokens = [2]
    req.next_forced_token = lambda: 2
    logits = torch.tensor([[9.0, 8.0, 0.0]])
    assert sample_scheduled(logits, [req], [True]) == [2]
    assert req.logprobs
