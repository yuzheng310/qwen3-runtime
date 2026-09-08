"""KMP n-gram proposer. Oracle for ``NgramIndex``; not on the hot path."""
from __future__ import annotations


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
