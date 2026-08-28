import math

import pytest
import torch

from qwen3_runtime.sampling import (
    SamplingParams,
    apply_top_k_top_p,
    greedy,
    greedy_logits,
    make_generator,
    sample_from_logits,
)


def test_greedy_picks_argmax_per_row():
    logits = [
        [0.1, 0.7, 0.2],
        [3.0, -1.0, 2.5],
    ]
    assert greedy(logits) == [1, 0]


def test_greedy_logits_matches_greedy_and_stays_on_tensor():
    rows = [
        [0.1, 0.7, 0.2],
        [3.0, -1.0, 2.5],
        [0.0, 0.0, 0.0],
    ]
    tensor = torch.tensor(rows, dtype=torch.float32)
    assert greedy_logits(tensor) == greedy(rows)
    assert greedy_logits(tensor) == [1, 0, 0]


def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 0.7, 0.2])
    assert sample_from_logits(logits, SamplingParams(temperature=0.0)) == 1
    assert sample_from_logits(logits, SamplingParams(temperature=-1.0)) == 1


def test_top_k_one_matches_argmax_even_with_temperature():
    logits = torch.tensor([0.0, 4.0, 1.0, 3.0])
    tok = sample_from_logits(logits, SamplingParams(temperature=1.5, top_k=1, seed=0), make_generator(0))
    assert tok == 1


def test_seeded_multinomial_is_deterministic():
    logits = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.8])
    params = SamplingParams(temperature=0.9, top_p=0.95, top_k=4, seed=123)
    a = sample_from_logits(logits, params, make_generator(123))
    b = sample_from_logits(logits, params, make_generator(123))
    assert a == b


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
