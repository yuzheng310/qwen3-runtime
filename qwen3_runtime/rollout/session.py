"""Match an incoming chat completion to the paused session that already holds it.

The SkyRL path is handed a whole conversation on every turn and is given no
session identity, so a resident session has to be recognised from the tokens
themselves. That is what this does: it keeps the sequence each paused session is
sitting on and, for a new request, finds the session whose sequence the request
continues.

Matching on content rather than on a caller-supplied id is what makes this a
drop-in: OpenHands does not have to thread a session through, and reuse is only
ever granted for tokens that are bit-identical to what the KV was built from.

The rule has to survive GRPO, where one instance is sampled several times and
the siblings open with byte-identical prompts. A sibling's first turn is a
prefix of a paused session, not a continuation of it, so the request must extend
past what the session holds before it can claim it. Without that, sibling B
would take A's session and A would pay a full re-prefill for nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def session_kv_enabled() -> bool:
    """Off only for measuring what it is worth; the product ships with it on."""
    return os.environ.get("QWEN3_SESSION_KV", "1").strip().lower() not in {"0", "false", "no"}


@dataclass
class _Session:
    request_id: int
    tokens: list[int]  # prompt plus what the model generated, i.e. what the KV covers
    blocks: int  # physical blocks the parked request still holds
    stamp: int


@dataclass
class Claim:
    request_id: int
    shared: int  # tokens the session already has computed and this request can keep
    held: int  # tokens the session was sitting on
    suffix: list[int]  # what still has to be prefilled


@dataclass
class SessionCache:
    """Paused sessions, keyed by the sequence their KV covers.

    ``max_sessions`` and ``max_blocks`` are eviction bounds, not a tuning knob:
    parked KV is real memory, and a rollout that never closes a conversation
    would otherwise hold the pool until prefill starts failing. Eviction is
    least-recently-used and hands the caller the ids it must release. The
    bound is physical blocks: with prefix-cache sharing, summing logical
    tokens overstates the marginal pages a parked session actually pins.

    A claim is granted only when the new prompt is a strict extension of a
    parked sequence. A re-tokenized history that diverges starts a new
    request; truncating held KV and prefilling the rest did not match a
    from-scratch prefill.

    ``max_blocks`` has no default: a block count means nothing without the pool
    it is a share of, and a default carrying one machine's pool size would read
    as a measured cap on every other machine.
    """

    max_blocks: int
    max_sessions: int = 16

    _sessions: dict[int, _Session] = field(default_factory=dict)
    _clock: int = 0
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "turns": 0,
            "resumed": 0,
            "started": 0,
            "tokens_prefilled": 0,
            "tokens_reused": 0,
            "evicted": 0,
        }
    )

    def claim(self, ids: list[int]) -> Claim | None:
        """Find the paused session this request continues, and take it."""
        self.stats["turns"] += 1
        best: Claim | None = None
        for session in self._sessions.values():
            shared = _common_prefix_len(session.tokens, ids)
            if shared >= len(ids):
                # Not longer than what the session holds: a sibling sample
                # replaying the shared opening, not this session's next turn.
                continue
            if shared != len(session.tokens):
                continue
            if best is None or shared > best.shared:
                best = Claim(
                    request_id=session.request_id,
                    shared=shared,
                    held=len(session.tokens),
                    suffix=ids[shared:],
                )
        if best is None:
            self.stats["started"] += 1
            self.stats["tokens_prefilled"] += len(ids)
            return None

        del self._sessions[best.request_id]
        self.stats["resumed"] += 1
        self.stats["tokens_reused"] += best.shared
        self.stats["tokens_prefilled"] += len(best.suffix)
        return best

    def park(self, request_id: int, tokens: list[int], blocks: int) -> list[int]:
        """Register a paused session. Returns ids the caller must finish."""
        self._clock += 1
        self._sessions[request_id] = _Session(
            request_id, list(tokens), max(0, int(blocks)), self._clock
        )
        return self._evict_to_fit()

    def evict_oldest(self) -> int | None:
        """Drop the least-recently parked session. Caller must finish the id."""
        if not self._sessions:
            return None
        oldest = min(self._sessions.values(), key=lambda s: s.stamp)
        del self._sessions[oldest.request_id]
        self.stats["evicted"] += 1
        return oldest.request_id

    def forget(self, request_id: int) -> None:
        self._sessions.pop(request_id, None)

    def reset_stats(self) -> None:
        """Zero the counters, keeping the parked sessions. For A/B arms."""
        for key in self.stats:
            self.stats[key] = 0

    def drop_all(self) -> list[int]:
        """Every session is gone, e.g. the KV pool was invalidated. Returns their ids."""
        ids = list(self._sessions)
        self._sessions.clear()
        return ids

    def _evict_to_fit(self) -> list[int]:
        evicted: list[int] = []
        while self._sessions and (
            len(self._sessions) > self.max_sessions or self._held_blocks() > self.max_blocks
        ):
            oldest = min(self._sessions.values(), key=lambda s: s.stamp)
            del self._sessions[oldest.request_id]
            evicted.append(oldest.request_id)
            self.stats["evicted"] += 1
        return evicted

    def _held_blocks(self) -> int:
        return sum(s.blocks for s in self._sessions.values())

    def _held_tokens(self) -> int:
        return sum(len(s.tokens) for s in self._sessions.values())

    def report(self) -> dict[str, float | int]:
        total = self.stats["tokens_prefilled"] + self.stats["tokens_reused"]
        return {
            **self.stats,
            "live_sessions": len(self._sessions),
            "held_blocks": self._held_blocks(),
            "held_tokens": self._held_tokens(),
            "reuse_pct": round(100.0 * self.stats["tokens_reused"] / max(1, total), 1),
        }
