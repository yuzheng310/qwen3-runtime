"""One logits-processor chain. Greedy is temperature=0 on that chain."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from qwen3_runtime.sampling_params import SamplingParams

__all__ = [
    "SamplingParams",
    "apply_top_k_top_p",
    "chosen_logprobs",
    "greedy_logits",
    "make_generator",
    "penalties_are_default",
    "process_logits",
    "process_logits_batch",
    "record_sampled",
    "sample_from_logits",
    "sample_from_logits_with_logprob",
    "sample_from_probs",
    "sample_scheduled",
]


def greedy_logits(logits: torch.Tensor) -> list[int]:
    """Argmax on-device. Copies only token ids to host, not the vocab row."""
    if logits.ndim != 2:
        raise ValueError(f"greedy_logits expects [rows, vocab], got {tuple(logits.shape)}")
    return logits.argmax(dim=-1).tolist()


def apply_logit_bias(logits: torch.Tensor, bias: dict[int, float] | None) -> torch.Tensor:
    if not bias:
        return logits
    out = logits.clone()
    for tok, value in bias.items():
        out[..., int(tok)] = out[..., int(tok)] + float(value)
    return out


def apply_penalties(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    *,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    """HF/vLLM penalties on a 1-D vocab row. No-op when all knobs are default."""
    if repetition_penalty == 1.0 and presence_penalty == 0.0 and frequency_penalty == 0.0:
        return logits
    if not token_ids:
        return logits
    vocab = logits.shape[-1]
    counts = torch.zeros(vocab, dtype=torch.long, device=logits.device)
    idx = torch.tensor([t for t in token_ids if 0 <= t < vocab], dtype=torch.long, device=logits.device)
    if idx.numel():
        counts.scatter_add_(0, idx, torch.ones_like(idx))
    out = logits.clone()
    seen = counts > 0
    if repetition_penalty != 1.0:
        pos = out > 0
        out = torch.where(seen & pos, out / repetition_penalty, out)
        out = torch.where(seen & ~pos, out * repetition_penalty, out)
    if presence_penalty != 0.0:
        out = out - presence_penalty * seen.to(out.dtype)
    if frequency_penalty != 0.0:
        out = out - frequency_penalty * counts.to(out.dtype)
    return out


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Drop tokens below ``min_p`` of the peak. Per row on a ``[rows, vocab]`` batch."""
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits.float(), dim=-1)
    thresh = min_p * probs.amax(dim=-1, keepdim=True)
    return torch.where(probs < thresh, torch.full_like(logits, float("-inf")), logits)


def apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Filter on a 1-D vocab row or a [rows, vocab] batch. HuggingFace TopK/TopP order."""
    if logits.ndim == 1:
        return _apply_top_k_top_p_rows(logits.unsqueeze(0), top_k, top_p).squeeze(0)
    if logits.ndim == 2:
        return _apply_top_k_top_p_rows(logits, top_k, top_p)
    raise ValueError(f"apply_top_k_top_p expects [vocab] or [rows, vocab], got {tuple(logits.shape)}")


def _apply_top_k_top_p_rows(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    x = logits
    vocab = x.shape[-1]
    if top_k > 0:
        k = min(int(top_k), vocab)
        thresh = torch.topk(x, k, dim=-1).values[..., -1:]
        x = torch.where(x < thresh, torch.full_like(x, torch.finfo(x.dtype).min), x)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, dim=-1, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cum > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.empty_like(sorted_remove)
        remove.scatter_(-1, sorted_idx, sorted_remove)
        x = torch.where(remove, torch.full_like(x, torch.finfo(x.dtype).min), x)
    return x


def penalties_are_default(params: SamplingParams) -> bool:
    return (
        params.repetition_penalty == 1.0
        and params.presence_penalty == 0.0
        and params.frequency_penalty == 0.0
    )


def process_logits(logits: torch.Tensor, params: SamplingParams, token_ids: Sequence[int]) -> torch.Tensor:
    """Apply the processor chain to a 1-D vocab row. Does not sample."""
    x = apply_logit_bias(logits, params.logit_bias)
    x = apply_penalties(
        x,
        token_ids,
        repetition_penalty=params.repetition_penalty,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
    )
    if params.is_greedy():
        return x
    if params.temperature != 1.0:
        x = x.float() / params.temperature
    x = apply_min_p(x, params.min_p)
    return apply_top_k_top_p(x, params.top_k, params.top_p)


def process_logits_batch(
    logits: torch.Tensor,
    params: SamplingParams,
    prefixes: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """``process_logits`` over ``[rows, vocab]``. Same stages, same order, per row.

    Speculative verification scores several positions from one forward, and it
    has to score them on the chain the ordinary sampler would have used, or the
    two paths stop drawing from the same distribution the moment anything but
    temperature and top-k/top-p is set.

    Bias, temperature, ``min_p`` and top-k/top-p depend only on the row, so they
    run on the whole block. The penalties read the tokens generated *before* the
    position a row scores, which differ row to row, so they need one prefix per
    row. ``apply_penalties`` returns its input untouched when its three knobs
    are default, which is why that loop costs nothing on ordinary sampling.
    """
    if logits.ndim != 2:
        raise ValueError(f"process_logits_batch expects [rows, vocab], got {tuple(logits.shape)}")
    x = apply_logit_bias(logits, params.logit_bias)
    if not penalties_are_default(params):
        if prefixes is None or len(prefixes) != x.shape[0]:
            n = "None" if prefixes is None else len(prefixes)
            raise ValueError(
                f"penalised sampling needs one prefix per row; got {n} for {x.shape[0]} rows"
            )
        x = torch.stack(
            [
                apply_penalties(
                    x[i],
                    prefixes[i],
                    repetition_penalty=params.repetition_penalty,
                    presence_penalty=params.presence_penalty,
                    frequency_penalty=params.frequency_penalty,
                )
                for i in range(x.shape[0])
            ]
        )
    if params.is_greedy():
        return x
    if params.temperature != 1.0:
        x = x.float() / params.temperature
    x = apply_min_p(x, params.min_p)
    return apply_top_k_top_p(x, params.top_k, params.top_p)


def chosen_logprobs(processed: torch.Tensor, tokens: Sequence[int]) -> list[float]:
    """log P(token) under an already-processed ``[rows, vocab]`` block.

    A ``logsumexp`` reduction rather than a full ``log_softmax``: the caller
    wants one entry per row, and materialising ``[rows, vocab]`` log-probs to
    read ``rows`` of them is the expensive way. The gather and the subtraction
    stay on device so this costs one host transfer for the whole block, not one
    per token -- this sits on the decode path, which is where every host
    round-trip in this engine has previously been found.
    """
    if processed.ndim != 2:
        raise ValueError(f"chosen_logprobs expects [rows, vocab], got {tuple(processed.shape)}")
    if len(tokens) > processed.shape[0]:
        raise ValueError(f"{len(tokens)} tokens against {processed.shape[0]} rows")
    if not tokens:
        return []
    n = len(tokens)
    x = processed[:n].float()
    idx = torch.tensor([int(t) for t in tokens], dtype=torch.long, device=x.device)
    picked = x.gather(1, idx.unsqueeze(1)).squeeze(1) - torch.logsumexp(x, dim=-1)
    return [float(v) for v in picked.detach().cpu().tolist()]


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    """Sample one index from a 1-D CPU probability row with a CPU generator.

    ``torch.multinomial`` on a 151,936-vocab row is ~18 ms (Phase 5.0 microbench).
    Inverse CDF is ~0.09 ms. The generator stays on CPU, so the seed is still
    device-stable. Not bit-identical to ``torch.multinomial``.
    """
    if probs.ndim != 1:
        raise ValueError(f"inverse-CDF expects [vocab], got {tuple(probs.shape)}")
    if probs.device.type != "cpu":
        raise ValueError("inverse-CDF sampler expects a CPU row")
    u = torch.rand((), dtype=torch.float32, generator=generator)
    cdf = probs.float().cumsum(0)
    idx = int(torch.searchsorted(cdf, u, right=True).item())
    n = probs.numel()
    if idx >= n:
        idx = n - 1
    elif idx < 0:
        idx = 0
    return idx


def sample_from_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
    token_ids: Sequence[int] = (),
) -> int:
    tok, _lp = sample_from_logits_with_logprob(logits, params, generator, token_ids)
    return tok


def sample_from_logits_with_logprob(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
    token_ids: Sequence[int] = (),
) -> tuple[int, float]:
    tok, lp, _top = _sample_row(logits, params, generator, token_ids)
    return tok, lp


def _sample_row(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
    token_ids: Sequence[int],
) -> tuple[int, float, list[tuple[int, float]]]:
    """Draw one token from a 1-D vocab row. Returns id, logprob, top-k logprobs."""
    if logits.ndim != 1:
        raise ValueError(f"sample_from_logits expects [vocab], got {tuple(logits.shape)}")
    processed = process_logits(logits, params, token_ids)
    if params.is_greedy():
        tok = int(processed.argmax().item())
        logp = torch.log_softmax(processed.float(), dim=-1)
        lp = float(logp[tok].item())
        return tok, lp, _topk_logprobs(logp, params.top_logprobs)
    probs = torch.softmax(processed.float(), dim=-1)
    cpu = probs.detach().cpu()
    tok = sample_from_probs(cpu, generator)
    lp = float(torch.log(cpu[tok].clamp_min(1e-30)).item())
    logp = torch.log(cpu.clamp_min(1e-30))
    return tok, lp, _topk_logprobs(logp, params.top_logprobs)


def _topk_logprobs(logp: torch.Tensor, k: int) -> list[tuple[int, float]]:
    if k <= 0:
        return []
    k = min(int(k), logp.numel())
    values, indices = torch.topk(logp, k)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())]


def record_sampled(req, logprob: float) -> None:
    """Note what a token was drawn under. One call per token a runner returns.

    ``last_logprob`` stays for callers that want the most recent value; the list
    is what the engine reads, because one step can emit several tokens and a
    single scalar cannot describe a speculative burst.
    """
    lp = float(logprob)
    req.last_logprob = lp
    req.logprobs.append(lp)


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
    """Sample only rows that complete this step. Logits are ``[n_sample, vocab]``.

    Forced tokens are applied here, at the tail of the chain, not in the scheduler.
    """
    if logits.ndim != 2:
        raise ValueError(f"sample_scheduled expects [rows, vocab], got {tuple(logits.shape)}")
    sampled = [(i, req) for i, (req, flag) in enumerate(zip(reqs, sample)) if flag]
    if logits.shape[0] != len(sampled):
        raise ValueError(
            f"logits rows {logits.shape[0]} must equal {len(sampled)} sampled requests"
        )
    out: list[int | None] = [None] * len(reqs)
    for row, (i, req) in zip(logits, sampled):
        forced = req.next_forced_token()
        params = req.sampling
        if forced is not None:
            tok = int(forced)
            logp = torch.log_softmax(process_logits(row, params, req.token_ids).float(), dim=-1)
            lp = float(logp[tok].item())
            top = _topk_logprobs(logp, params.top_logprobs)
        else:
            tok, lp, top = _sample_row(row, params, req.rng, req.token_ids)
        out[i] = tok
        record_sampled(req, lp)
        if params.top_logprobs > 0:
            req.top_logprobs.append(top)
    return out
