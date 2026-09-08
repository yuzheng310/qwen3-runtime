"""Sampling hyperparameters. Greedy is ``temperature <= 0`` on the same chain."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class SamplingParams:
    """temperature → penalties / bias → min_p → top-k → top-p → softmax → sample.

    ``temperature <= 0`` is greedy (argmax after the same processors).
    ``top_k <= 0`` disables top-k. ``top_p == 1`` disables nucleus.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    min_tokens: int = 0
    stop_strings: tuple[str, ...] = ()
    logprobs: bool = False
    top_logprobs: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if not math.isfinite(self.min_p) or not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must satisfy 0 <= min_p <= 1")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if self.top_logprobs < 0:
            raise ValueError("top_logprobs must be non-negative")
        if self.repetition_penalty <= 0 or not math.isfinite(self.repetition_penalty):
            raise ValueError("repetition_penalty must be positive and finite")

    def is_greedy(self) -> bool:
        return self.temperature <= 0.0
