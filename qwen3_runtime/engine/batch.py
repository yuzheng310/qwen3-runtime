"""Packed-batch helpers.

Prefill is flattened tokens, not one row per request. Last-token selection
must be explicit (see project spec §11).
"""

from collections.abc import Sequence


def last_token_indices(cu_seqlens_q: Sequence[int]) -> list[int]:
    if len(cu_seqlens_q) < 2:
        raise ValueError("cu_seqlens_q must contain at least one sequence")
    return [cu_seqlens_q[i] - 1 for i in range(1, len(cu_seqlens_q))]


def sampled_logit_rows(cu_seqlens_q: Sequence[int], sample: Sequence[bool]) -> list[int]:
    """Packed rows that need an LM-head logit (last token of each sampling sequence)."""
    last = last_token_indices(cu_seqlens_q)
    if len(sample) != len(last):
        raise ValueError("sample flags must align with packed sequences")
    return [last[i] for i, flag in enumerate(sample) if flag]


def select_last_rows(packed: Sequence, cu_seqlens_q: Sequence[int]) -> list:
    return [packed[i] for i in last_token_indices(cu_seqlens_q)]
