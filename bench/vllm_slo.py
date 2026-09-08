"""Helpers for vLLM SLO traces. Importable without the vLLM package."""

from __future__ import annotations

from qwen3_runtime.serving.slo_harness import RequestTrace, _record_token


def note_new_tokens(trace: RequestTrace, token_ids: list[int], now: float) -> None:
    """Append newly streamed output ids at the host clock instant `now`."""
    prev = len(trace.tokens)
    if len(token_ids) < prev:
        raise ValueError("streamed token ids shrank")
    for tok in token_ids[prev:]:
        _record_token(trace, tok, now)
