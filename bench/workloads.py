"""Frozen workload regimes. Tiny scale is supplementary and must never be headline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED = 20260825
PIN_REPO = "Qwen/Qwen3-4B"
PIN_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
# README table still requires these six (historical + retained online).
HEADLINE_CASES = ("latency", "throughput", "online", "longctx", "prefill", "decode")
# Production Parity vs vLLM closed-batch. Frozen in docs/PARITY_WORKLOADS.md.
PARITY_CASES = ("decode", "latency", "batch8", "throughput", "prefill", "longctx")
ALL_CASES = tuple(dict.fromkeys((*HEADLINE_CASES, *PARITY_CASES)))


@dataclass(frozen=True)
class Workload:
    name: str
    scale: str
    concurrency: int
    n_requests: int
    prompt_tokens: int
    output_tokens: int
    poisson_rate: float | None = None

    @property
    def supplementary(self) -> bool:
        return self.scale != "full"


FULL: dict[str, Workload] = {
    "latency": Workload("latency", "full", 1, 1, 512, 128),
    "throughput": Workload("throughput", "full", 16, 16, 256, 128),
    "online": Workload("online", "full", 8, 32, 256, 128, poisson_rate=8.0),
    "longctx": Workload("longctx", "full", 1, 1, 8192, 64),
    "prefill": Workload("prefill", "full", 1, 1, 2048, 16),
    "decode": Workload("decode", "full", 1, 1, 256, 512),
    # Same tokens as throughput; concurrency 8 is the moderate-batch point (spec §9 B).
    "batch8": Workload("batch8", "full", 8, 8, 256, 128),
}

# Frozen longctx chunk cap (PARITY_WORKLOADS). Not a throughput/batch8 knob.
LONGCTX_CHUNK_BUDGET = 2048
# Phase 1 length axis. 9981 is the session first-turn prompt, not a round 10k.
TTFT_LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 9981, 16384)


def token_budget_for_case(case: str, workload: Workload) -> int:
    """Scheduler token budget for a frozen parity case.

    longctx stays at 2048 (chunked 8K). Other closed-batch cases must fit
    ``concurrency * prompt_tokens`` in one prefill wave. A global 2048 cap
    splits throughput (16×256=4096) into two waves of 8; vLLM LLM.generate on
    a 4090 defaults ``max_num_batched_tokens=8192``.
    """
    if case == "longctx":
        return LONGCTX_CHUNK_BUDGET
    return max(LONGCTX_CHUNK_BUDGET, workload.concurrency * workload.prompt_tokens)


TINY: dict[str, Workload] = {
    "latency": Workload("latency", "tiny", 1, 1, 8, 4),
    "throughput": Workload("throughput", "tiny", 2, 2, 6, 3),
    "online": Workload("online", "tiny", 2, 3, 6, 3, poisson_rate=50.0),
    "longctx": Workload("longctx", "tiny", 1, 1, 16, 2),
    "prefill": Workload("prefill", "tiny", 1, 1, 12, 2),
    "decode": Workload("decode", "tiny", 1, 1, 6, 8),
    "batch8": Workload("batch8", "tiny", 2, 2, 6, 3),
}


def get_workload(case: str, scale: str) -> Workload:
    table = TINY if scale == "tiny" else FULL
    if case not in table:
        raise KeyError(case)
    return table[case]


def make_prompts(workload: Workload, vocab_size: int, *, seed: int = SEED) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    hi = max(2, min(int(vocab_size) - 1, 1000))
    ids = rng.integers(1, hi, size=(workload.n_requests, workload.prompt_tokens))
    return ids.tolist()


def mix_length_prompts(
    n: int,
    *,
    short: int,
    long: int,
    long_frac: float,
    vocab_size: int,
    seed: int = SEED,
) -> list[list[int]]:
    """Shuffle a mix of two prompt lengths. Used by the high-concurrency matrix."""
    from types import SimpleNamespace

    n_long = int(round(n * long_frac))
    n_short = n - n_long
    short_ps = make_prompts(SimpleNamespace(n_requests=n_short, prompt_tokens=short), vocab_size, seed=seed)
    long_ps = make_prompts(SimpleNamespace(n_requests=n_long, prompt_tokens=long), vocab_size, seed=seed + 11)
    out = short_ps + long_ps
    rng = np.random.default_rng(seed + 5)
    return [out[i] for i in rng.permutation(n)]


def poisson_arrivals(n: int, rate: float, *, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed + 1)
    gaps = rng.exponential(1.0 / rate, size=n)
    arrivals = np.cumsum(gaps)
    arrivals[0] = 0.0
    return arrivals.tolist()
