"""Attach sampling-time logprobs to a replayed SkyRL rollout.

SkyRL carries two logprob tensors through a GRPO step: `rollout_logprobs`, which
the generator reports from sampling time, and `action_log_probs`, which the
trainer recomputes in its forward pass. Comparing them is the only instrument in
this plan that can catch rollout-train mismatch, the failure class that corrupts
gradients without raising anything.

CodeScout's generator hardcodes `rollout_logprobs: None`, so the field arrives
empty. The engine does record every sampled token and its logprob to a sidecar,
and each trajectory's TokenEvents carry the same `response_token_ids`, so the two
can be matched on exact token sequences rather than on timing or ordering
heuristics.

One thing that does not always match: a turn's emitted tokens are not guaranteed
to survive the round trip into the next turn's prompt. OpenHands re-renders the
conversation as text and re-encodes it, and the tokenizer may merge differently
than the model sampled -- one measured case emitted `[7245, 4]` where the
re-encoded prompt holds the single token `32328`. Those positions carry a
trainer logprob for a token the policy never emitted. They are reported rather
than papered over; see `scripts/audit_retokenization.py`.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


def load_sidecar(path: str) -> dict[tuple[int, ...], list[list[float]]]:
    """Index sidecar records by their exact token sequence.

    A value is a list because the same completion can legitimately recur; entries
    are consumed in file order so repeats stay distinguishable.
    """
    by_tokens: dict[tuple[int, ...], list[list[float]]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = rec.get("token_ids") or []
            lps = rec.get("logprobs") or []
            if not ids or len(ids) != len(lps):
                continue
            by_tokens.setdefault(tuple(ids), []).append([float(x) for x in lps])
    return by_tokens


def _find_from(seq: list[int], sub: list[int], start: int) -> int:
    n = len(sub)
    if n == 0:
        return -1
    for i in range(start, len(seq) - n + 1):
        if seq[i : i + n] == sub:
            return i
    return -1


def place_turn_logprobs(
    response_ids: list[int],
    turns: Iterable[tuple[list[int], list[float]]],
) -> tuple[list[float], dict[str, Any]]:
    """Lay each turn's logprobs onto its span inside the replayed response.

    Turns are located left to right so a repeated tool call cannot bind to an
    earlier copy. A turn that cannot be located does not abort the walk: the
    cursor stays put and the next turn is tried, because a single re-tokenized
    turn should cost only its own positions.
    """
    out = [0.0] * len(response_ids)
    covered = [False] * len(response_ids)
    cursor = 0
    placed = unplaced = placed_tokens = unplaced_tokens = 0

    for ids, lps in turns:
        at = _find_from(response_ids, ids, cursor)
        if at < 0:
            unplaced += 1
            unplaced_tokens += len(ids)
            continue
        for k, lp in enumerate(lps):
            out[at + k] = lp
            covered[at + k] = True
        cursor = at + len(ids)
        placed += 1
        placed_tokens += len(ids)

    return out, {
        "turns_placed": placed,
        "turns_unplaced": unplaced,
        "tokens_placed": placed_tokens,
        "tokens_unplaced": unplaced_tokens,
        "covered": covered,
    }


def turns_from_record(
    rec: dict[str, Any],
    sidecar: dict[tuple[int, ...], list[list[float]]],
) -> tuple[list[tuple[list[int], list[float]]], int]:
    """Pair each TokenEvent with its sidecar logprobs. Returns (turns, n_missing)."""
    turns: list[tuple[list[int], list[float]]] = []
    missing = 0
    for msg in rec.get("messages") or []:
        if msg.get("kind") != "TokenEvent":
            continue
        ids = list(msg.get("response_token_ids") or [])
        if not ids:
            continue
        bucket = sidecar.get(tuple(ids))
        if not bucket:
            missing += 1
            continue
        turns.append((ids, bucket.pop(0)))
    return turns, missing


def mask_coverage(loss_mask: list[int], covered: list[bool]) -> dict[str, int]:
    """How much of the trained span actually carries a sampling-time logprob."""
    trained = sum(1 for m in loss_mask if m)
    both = sum(1 for m, c in zip(loss_mask, covered) if m and c)
    return {
        "trained_positions": trained,
        "trained_with_rollout_logprob": both,
        "trained_without_rollout_logprob": trained - both,
    }
