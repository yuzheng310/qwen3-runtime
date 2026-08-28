"""N-gram speculative decoding + Leviathan target verification.

Proposer: vLLM 0.27.1 ``_find_longest_matched_ngram_and_propose_tokens``
(Prompt Lookup Decoding), ported to Python without Numba.
Verifier: vLLM V1 ``rejection_sampler`` greedy / ``NO_DRAFT_PROBS`` paths
(Leviathan et al., arXiv:2211.17192). Synthetic acceptance is not ported.

License: Apache-2.0 (upstream vLLM). This file is an adaptation, not a copy
of the Triton kernels.
"""

from __future__ import annotations

import torch

from qwen3_runtime.sampling import SamplingParams, apply_top_k_top_p, sample_from_logits


def propose_ngram(
    origin_tokens: list[int],
    min_ngram: int,
    max_ngram: int,
    k: int,
    max_model_len: int = 65536,
) -> list[int]:
    """Longest suffix n-gram in [min_ngram, max_ngram], then up to k tokens after it.

    Earliest match in the original sequence wins (vLLM LPS on the reversed ids).
    """
    total_token = len(origin_tokens)
    if total_token < min_ngram or k <= 0 or min_ngram < 1 or max_ngram < min_ngram:
        return []
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return []
    tokens = origin_tokens[::-1]
    lps = [0] * max_ngram
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1
    if longest_ngram < min_ngram:
        return []
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return list(origin_tokens[start_position : start_position + k])


def remaining_output_budget(req) -> int:
    n_draft = int(getattr(req, "spec_draft_len", 0) or 0)
    produced = len(req.token_ids) - n_draft - req.num_prompt_tokens
    return max(0, req.max_tokens - produced)


def propose_drafts(req, *, k: int, ngram_min: int, ngram_max: int) -> tuple[list[int], bool]:
    """Return (drafts, teacher_force). Teacher-force uses remaining forced ids."""
    room = remaining_output_budget(req)
    if room <= 0 or k <= 0:
        return [], False
    k = min(k, room)
    if req.forced_tokens is not None:
        produced = len(req.token_ids) - req.num_prompt_tokens
        left = req.forced_tokens[produced:]
        drafts = list(left[:k])
        return drafts, True
    return propose_ngram(req.token_ids, ngram_min, ngram_max, k), False


def _target_probs(logits_row: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    x = logits_row.detach().float().cpu()
    if params.is_greedy():
        return torch.softmax(x, dim=-1)
    if params.temperature != 1.0:
        x = x / params.temperature
    x = apply_top_k_top_p(x, params.top_k, params.top_p)
    return torch.softmax(x, dim=-1)


def verify_greedy(drafts: list[int], logits: torch.Tensor) -> list[int]:
    """Accept while draft == argmax; emit argmax on the first mismatch; else bonus."""
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError(
            f"greedy verify expects [k+1, vocab], got {tuple(logits.shape)} for k={len(drafts)}"
        )
    argmax = logits.argmax(dim=-1).tolist()
    out: list[int] = []
    for i, draft in enumerate(drafts):
        tok = int(argmax[i])
        out.append(tok)
        if tok != draft:
            return out
    out.append(int(argmax[len(drafts)]))
    return out


def verify_sampled(
    drafts: list[int],
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
) -> list[int]:
    """Leviathan with draft_prob=1 (vLLM n-gram / NO_DRAFT_PROBS)."""
    if params.is_greedy():
        return verify_greedy(drafts, logits)
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError(
            f"sampled verify expects [k+1, vocab], got {tuple(logits.shape)} for k={len(drafts)}"
        )
    out: list[int] = []
    for i, draft in enumerate(drafts):
        probs = _target_probs(logits[i], params)
        q = float(probs[int(draft)].item())
        uniform = float(torch.rand((), generator=generator).item())
        # vLLM: accepted = (draft_prob > 0) and (target_prob / draft_prob >= uniform)
        if q >= uniform:
            out.append(int(draft))
            continue
        residual = probs.clone()
        residual[int(draft)] = 0.0
        total = float(residual.sum().item())
        if total <= 0.0:
            recovered = int(probs.argmax().item())
        else:
            recovered = int(torch.multinomial(residual, 1, generator=generator).item())
        out.append(recovered)
        return out
    out.append(sample_from_logits(logits[len(drafts)], params, generator))
    return out


def trim_emitted(req, tokens: list[int], *, eos_token_id: int | None) -> list[int]:
    """Clamp to max_tokens; stop on EOS/stop unless ignore_eos (teacher-force skips stop)."""
    room = remaining_output_budget(req)
    out: list[int] = []
    teacher = bool(getattr(req, "spec_teacher_force", False))
    for tok in tokens[:room]:
        out.append(int(tok))
        if teacher:
            continue
        if req.ignore_eos:
            continue
        if tok in req.stop_token_ids:
            break
        if eos_token_id is not None and tok == eos_token_id:
            break
    return out
