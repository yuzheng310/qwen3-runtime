from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingParams:
    """CodeScout rollout sampling. Defaults keep the historical greedy path.

    temperature <= 0 is greedy (argmax). top_k <= 0 disables top-k.
    top_p == 1 disables nucleus. Matches the usual vLLM/HF order:
    temperature → top-k → top-p → softmax → multinomial.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")

    def is_greedy(self) -> bool:
        return self.temperature <= 0.0


def greedy(logits: Sequence[Sequence[float]]) -> list[int]:
    """Argmax per row. Used for Correctness Baseline v0; no temperature."""
    tokens: list[int] = []
    for row in logits:
        if not row:
            raise ValueError("empty logits row")
        best = 0
        for i, v in enumerate(row):
            if v > row[best]:
                best = i
        tokens.append(best)
    return tokens


def greedy_logits(logits: torch.Tensor) -> list[int]:
    """Argmax on-device. Copies only token ids to host, not the vocab row."""
    if logits.ndim != 2:
        raise ValueError(f"greedy_logits expects [rows, vocab], got {tuple(logits.shape)}")
    return logits.argmax(dim=-1).tolist()


def apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """In-place-safe filter on a 1-D vocab row. HuggingFace TopK/TopP warper order."""
    if logits.ndim != 1:
        raise ValueError(f"apply_top_k_top_p expects [vocab], got {tuple(logits.shape)}")
    x = logits
    vocab = x.numel()
    if top_k > 0:
        k = min(int(top_k), vocab)
        thresh = torch.topk(x, k).values[-1]
        x = torch.where(x < thresh, torch.full_like(x, torch.finfo(x.dtype).min), x)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cum > top_p
        sorted_remove[1:] = sorted_remove[:-1].clone()
        sorted_remove[0] = False
        remove = torch.empty_like(sorted_remove)
        remove.scatter_(0, sorted_idx, sorted_remove)
        x = torch.where(remove, torch.full_like(x, torch.finfo(x.dtype).min), x)
    return x


def sample_from_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token from a 1-D logit row. CPU multinomial so seed is device-stable."""
    if logits.ndim != 1:
        raise ValueError(f"sample_from_logits expects [vocab], got {tuple(logits.shape)}")
    if params.is_greedy():
        return int(logits.argmax().item())
    x = logits.float()
    if params.temperature != 1.0:
        x = x / params.temperature
    x = apply_top_k_top_p(x, params.top_k, params.top_p)
    probs = torch.softmax(x, dim=-1)
    cpu = probs.detach().float().cpu()
    idx = torch.multinomial(cpu, 1, generator=generator)
    return int(idx.item())


def make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def sample_scheduled(
    logits: torch.Tensor,
    reqs: Sequence,
    sample: Sequence[bool],
) -> list[int | None]:
    """Sample only rows that complete this step. Does not advance RNG for prefills."""
    if logits.ndim != 2:
        raise ValueError(f"sample_scheduled expects [rows, vocab], got {tuple(logits.shape)}")
    n_sample = sum(1 for flag in sample if flag)
    if logits.shape[0] == len(reqs):
        rows = logits
        row_for_req = list(range(len(reqs)))
    elif logits.shape[0] == n_sample:
        rows = logits
        row_for_req = []
        j = 0
        for flag in sample:
            if flag:
                row_for_req.append(j)
                j += 1
            else:
                row_for_req.append(-1)
    else:
        raise ValueError(
            f"logits rows {logits.shape[0]} match neither {len(reqs)} requests nor {n_sample} samples"
        )
    out: list[int | None] = []
    for req, flag, row_i in zip(reqs, sample, row_for_req):
        if not flag:
            out.append(None)
            continue
        out.append(sample_from_logits(rows[row_i], req.sampling, req.rng))
    return out
