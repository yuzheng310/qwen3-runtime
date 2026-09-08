"""N-gram speculative decoding + Leviathan target verification.

Proposer: vLLM 0.27.1 ``_find_longest_matched_ngram_and_propose_tokens``
(Prompt Lookup Decoding), ported to Python without Numba.
Verifier: vLLM V1 ``rejection_sampler`` greedy / ``NO_DRAFT_PROBS`` paths
(Leviathan et al., arXiv:2211.17192). Synthetic acceptance is not ported.

License: Apache-2.0 (upstream vLLM). This file is an adaptation, not a copy
of the Triton kernels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from qwen3_runtime.engine.request import Request
from qwen3_runtime.sampling import (
    SamplingParams,
    chosen_logprobs,
    process_logits_batch,
    sample_from_probs,
)


class NgramIndex:
    """Incremental first-occurrence index. Longest L first, then earliest start.

    Must match ``qwen3_runtime.reference.propose_ngram.propose_ngram`` token-for-token.
    Do not cap the search window.
    """

    def __init__(self, min_ngram: int, max_ngram: int):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.tokens: list[int] = []
        self.first: list[dict[tuple[int, ...], int]] = [{} for _ in range(max_ngram + 1)]

    def extend(self, tokens: list[int]) -> None:
        for tok in tokens:
            self.tokens.append(int(tok))
            n = len(self.tokens)
            for length in range(self.min_ngram, self.max_ngram + 1):
                if n < length:
                    continue
                key = tuple(self.tokens[n - length : n])
                self.first[length].setdefault(key, n - length)

    def propose(self, k: int, max_model_len: int = 65536) -> list[int]:
        return _propose_from_first(
            self.tokens, self.first, self.min_ngram, self.max_ngram, k, max_model_len
        )


def _propose_from_first(
    origin_tokens: list[int],
    first: list[dict[tuple[int, ...], int]],
    min_ngram: int,
    max_ngram: int,
    k: int,
    max_model_len: int,
) -> list[int]:
    total_token = len(origin_tokens)
    if total_token < min_ngram or k <= 0 or min_ngram < 1 or max_ngram < min_ngram:
        return []
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return []
    for length in range(max_ngram, min_ngram - 1, -1):
        if total_token < length:
            continue
        key = tuple(origin_tokens[total_token - length :])
        start = first[length].get(key)
        if start is None:
            continue
        nxt = start + length
        if nxt >= total_token:
            continue
        take = min(k, total_token - nxt)
        if take <= 0:
            continue
        return list(origin_tokens[nxt : nxt + take])
    return []


def remaining_output_budget(req: Request) -> int:
    n_draft = req.spec_draft_len
    produced = len(req.token_ids) - n_draft - req.num_prompt_tokens
    return max(0, req.max_tokens - produced)


def propose_drafts(req: Request, *, k: int, ngram_min: int, ngram_max: int) -> tuple[list[int], bool]:
    """Return (drafts, teacher_force). Teacher-force uses remaining forced ids."""
    room = remaining_output_budget(req)
    if room <= 0 or k <= 0:
        return [], False
    k = min(k, room)
    if req.forced_tokens is not None:
        drafts = list(req.remaining_forced_tokens()[:k])
        return drafts, True
    index = req.ngram_index
    if index is None or index.min_ngram != ngram_min or index.max_ngram != ngram_max:
        index = NgramIndex(ngram_min, ngram_max)
        index.extend(list(req.token_ids))
        req.ngram_index = index
    elif len(index.tokens) < len(req.token_ids):
        index.extend(list(req.token_ids[len(index.tokens) :]))
    elif len(index.tokens) > len(req.token_ids):
        index = NgramIndex(ngram_min, ngram_max)
        index.extend(list(req.token_ids))
        req.ngram_index = index
    return index.propose(k), False


_MIN_PROB = 1e-30


def _logprob(p: float) -> float:
    """Same floor as ``sampling._sample_row``, so both paths report on one scale."""
    return float(math.log(max(p, _MIN_PROB)))


def _batch_target_probs(
    logits: torch.Tensor,
    params: SamplingParams,
    prefixes: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Target distribution per verified position, on the ordinary sampling chain."""
    return torch.softmax(process_logits_batch(logits, params, prefixes).float(), dim=-1)


def spec_prefixes(req: Request) -> list[list[int]]:
    """The context each verified row is conditioned on, one list per row.

    Row *i* predicts the token at ``prefix_len + i``, so its penalties have to
    see everything generated before that position -- the same context
    ``sample_scheduled`` hands an ordinary token.
    """
    n_draft = req.spec_draft_len
    prefix_len = len(req.token_ids) - n_draft
    return [req.token_ids[: prefix_len + i] for i in range(n_draft + 1)]


def verify_greedy(
    drafts: list[int],
    logits: torch.Tensor,
    params: SamplingParams | None = None,
    prefixes: Sequence[Sequence[int]] | None = None,
) -> tuple[list[int], list[float]]:
    """Accept while draft == argmax; emit argmax on the first mismatch; else bonus.

    Returns the emitted tokens and the logprob each was chosen under, read off
    the same processed rows the argmax was taken on.
    """
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError(
            f"greedy verify expects [k+1, vocab], got {tuple(logits.shape)} for k={len(drafts)}"
        )
    processed = process_logits_batch(logits, params or SamplingParams(), prefixes)
    argmax = processed.argmax(dim=-1).tolist()
    out: list[int] = []
    for i, draft in enumerate(drafts):
        out.append(int(argmax[i]))
        if out[-1] != draft:
            return out, chosen_logprobs(processed, out)
    out.append(int(argmax[len(drafts)]))
    return out, chosen_logprobs(processed, out)


def verify_sampled(
    drafts: list[int],
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
    prefixes: Sequence[Sequence[int]] | None = None,
) -> tuple[list[int], list[float]]:
    """Leviathan with draft_prob=1 (vLLM n-gram / NO_DRAFT_PROBS).

    The reported logprob is the **target** density of the emitted token at every
    position, including at a rejection, where the draw itself came from the
    residual. That is the behaviour logprob a trainer needs: speculative
    sampling is distribution-preserving, so marginalising over accept and reject
    leaves the emitted token distributed exactly as ``probs``. Reporting the
    residual's own density would describe a policy that never ran.
    """
    if params.is_greedy():
        return verify_greedy(drafts, logits, params, prefixes)
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError(
            f"sampled verify expects [k+1, vocab], got {tuple(logits.shape)} for k={len(drafts)}"
        )
    probs = _batch_target_probs(logits, params, prefixes)
    device = probs.device
    n_draft = len(drafts)
    draft_t = torch.tensor(drafts, device=device, dtype=torch.long)
    q_cpu = probs[:n_draft].gather(1, draft_t.unsqueeze(1)).detach().flatten().cpu()
    out: list[int] = []
    lps: list[float] = []
    for i, draft in enumerate(drafts):
        q = float(q_cpu[i].item())
        uniform = float(torch.rand((), generator=generator).item())
        if q >= uniform:
            out.append(int(draft))
            lps.append(_logprob(q))
            continue
        row = probs[i].detach().float().cpu()
        residual = row.clone()
        residual[int(draft)] = 0.0
        total = float(residual.sum().item())
        if total <= 0.0:
            recovered = int(row.argmax().item())
        else:
            recovered = sample_from_probs(residual / residual.sum(), generator)
        out.append(recovered)
        lps.append(_logprob(float(row[recovered].item())))
        return out, lps
    bonus = probs[n_draft].detach().float().cpu()
    tok = sample_from_probs(bonus, generator)
    out.append(tok)
    lps.append(_logprob(float(bonus[tok].item())))
    return out, lps


def forced_logprobs(
    tokens: list[int],
    logits,
    params: SamplingParams,
    prefixes: Sequence[Sequence[int]],
) -> list[float]:
    """Score teacher-forced tokens the way ``sample_scheduled`` scores forced ones.

    ``run_logits`` returns None when every scheduled request is teacher-forced,
    and a replay with no target distribution has no logprob to report. NaN is
    the honest value there: it keeps one logprob per token while making any
    downstream arithmetic fail loudly, instead of a plausible-looking zero that
    a trainer would happily consume.
    """
    if logits is None or not tokens:
        return [float("nan")] * len(tokens)
    processed = process_logits_batch(logits[: len(tokens)], params, list(prefixes)[: len(tokens)])
    return chosen_logprobs(processed, tokens)


def trim_emitted(
    req: Request,
    tokens: list[int],
    logprobs: list[float],
    *,
    eos_token_id: int | None,
) -> tuple[list[int], list[float]]:
    """Clamp to max_tokens; stop on EOS/stop unless ignore_eos (teacher-force skips stop).

    Token and logprob are trimmed as one unit. A token that reaches a trainer
    without the logprob it was sampled under is worse than one that never
    arrives, because nothing downstream can see that it is missing.
    """
    if len(tokens) != len(logprobs):
        raise ValueError(f"{len(tokens)} tokens against {len(logprobs)} logprobs")
    room = remaining_output_budget(req)
    out: list[int] = []
    lps: list[float] = []
    teacher = req.spec_teacher_force
    for tok, lp in zip(tokens[:room], logprobs[:room]):
        out.append(int(tok))
        lps.append(float(lp))
        if teacher:
            continue
        if req.ignore_eos:
            continue
        if tok in req.stop_token_ids:
            break
        if eos_token_id is not None and tok == eos_token_id:
            break
    return out, lps


def verify_request(
    req: Request, logits, *, eos_token_id: int | None
) -> tuple[list[int], list[float]]:
    drafts = req.token_ids[-req.spec_draft_len :]
    prefixes = spec_prefixes(req)
    if req.spec_teacher_force:
        forced = forced_logprobs(drafts, logits, req.sampling, prefixes)
        return trim_emitted(req, drafts, forced, eos_token_id=eos_token_id)
    if logits is None:
        raise RuntimeError("speculative verify needs target logits")
    if req.sampling.is_greedy():
        emitted, lps = verify_greedy(drafts, logits, req.sampling, prefixes)
    else:
        emitted, lps = verify_sampled(drafts, logits, req.sampling, req.rng, prefixes)
    return trim_emitted(req, emitted, lps, eos_token_id=eos_token_id)


def commit_spec(
    req: Request, emitted: list[int], logprobs: list[float], block_manager
) -> None:
    if len(emitted) != len(logprobs):
        raise ValueError(f"{len(emitted)} spec tokens against {len(logprobs)} logprobs")
    n_draft = req.spec_draft_len
    prefix_len = len(req.token_ids) - n_draft
    if not emitted:
        req.token_ids = req.token_ids[:prefix_len]
        keep = max(0, len(req.token_ids) - 1)
        req.num_computed_tokens = keep
        block_manager.truncate_kv(req, keep)
        return
    req.token_ids = req.token_ids[:prefix_len] + emitted
    # Drafts were proposed, never sampled, so they carry no logprob. These do:
    # one per token the verifier emitted, appended in the same order and at the
    # same time as the tokens, so the two lists cannot drift apart later.
    req.logprobs.extend(logprobs)
    req.last_logprob = logprobs[-1]
    keep = len(req.token_ids) - 1
    req.num_computed_tokens = keep
    block_manager.truncate_kv(req, keep)
