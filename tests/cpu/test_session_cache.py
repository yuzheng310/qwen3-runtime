"""Which paused session a turn is allowed to continue."""

from __future__ import annotations

import pytest

from qwen3_runtime.rollout.session import SessionCache, session_kv_enabled


def _cache(**kwargs) -> SessionCache:
    # A bound is required at construction; tests that are not exercising
    # eviction say so by asking for more than they park.
    kwargs.setdefault("max_blocks", 10_000)
    return SessionCache(**kwargs)


def _park(cache: SessionCache, request_id: int, tokens: list[int], blocks: int = 1) -> list[int]:
    return cache.park(request_id, tokens, blocks)


def test_the_next_turn_of_a_conversation_keeps_the_kv_it_already_built():
    cache = _cache()
    _park(cache, 7, [1, 2, 3, 4, 5])

    claim = cache.claim([1, 2, 3, 4, 5, 6, 7])

    assert claim is not None
    assert claim.request_id == 7
    assert claim.shared == 5
    assert claim.suffix == [6, 7]


def test_a_sibling_sample_replaying_the_same_opening_does_not_steal_the_session():
    """GRPO samples one instance several times; their first turns are identical.

    The sibling's prompt is a prefix of what the session holds, not an
    extension of it. Handing over the session would break the trajectory that
    is actually mid-flight and buy the sibling nothing.
    """
    cache = _cache()
    prompt = list(range(100))
    _park(cache, 1, prompt + [900, 901])  # trajectory A: prompt plus its answer

    assert cache.claim(prompt) is None
    assert cache.stats["started"] == 1
    assert cache.stats["resumed"] == 0


def test_a_turn_that_diverged_starts_over_because_rewinding_is_not_yet_equal():
    """The rewind path changes sampled tokens, so a divergence pays full price.

    It is the cheap trade: a measured step diverges on 6 turns out of 295.
    """
    cache = _cache()
    _park(cache, 3, [1, 2, 3, 4, 5, 6])

    assert cache.claim([1, 2, 3, 99, 100, 101, 102]) is None
    assert cache.stats["started"] == 1


def test_an_unrelated_conversation_starts_its_own_session():
    cache = _cache()
    _park(cache, 1, list(range(100)))

    assert cache.claim([500, 501, 502]) is None


def test_the_longest_match_wins_when_several_sessions_share_an_opening():
    cache = _cache()
    shared = list(range(50))
    _park(cache, 1, shared + [60])
    _park(cache, 2, shared + [60, 61, 62])

    claim = cache.claim(shared + [60, 61, 62, 63])

    assert claim is not None
    assert claim.request_id == 2
    assert claim.shared == 53


def test_a_claimed_session_is_gone_so_two_turns_cannot_share_one_request():
    cache = _cache()
    _park(cache, 4, [1, 2, 3])

    assert cache.claim([1, 2, 3, 4]) is not None
    assert cache.claim([1, 2, 3, 5]) is None


def test_parked_kv_is_capped_and_the_evicted_ids_come_back_to_be_released():
    """Paused sessions are never preempted, so nothing else reclaims them."""
    cache = _cache(max_blocks=2, max_sessions=99)
    assert _park(cache, 1, list(range(100)), blocks=1) == []
    assert _park(cache, 2, list(range(100)), blocks=1) == []

    evicted = _park(cache, 3, list(range(100)), blocks=1)

    assert evicted == [1]  # least recently parked
    assert cache.stats["evicted"] == 1
    assert cache.report()["held_blocks"] == 2


def test_the_session_count_is_capped_even_when_the_conversations_are_short():
    cache = _cache(max_sessions=2, max_blocks=10_000)
    _park(cache, 1, [1])
    _park(cache, 2, [2])

    assert _park(cache, 3, [3]) == [1]


def test_dropping_everything_hands_back_every_id_so_none_leak():
    cache = _cache()
    _park(cache, 1, [1, 2])
    _park(cache, 2, [3, 4])

    assert sorted(cache.drop_all()) == [1, 2]
    assert cache.drop_all() == []
    assert cache.claim([1, 2, 3]) is None


def test_the_report_says_how_much_prefill_the_sessions_actually_saved():
    cache = _cache()
    cache.claim(list(range(100)))  # miss: prefills 100
    _park(cache, 1, list(range(120)))
    cache.claim(list(range(130)))  # hit: reuses 120, prefills 10

    report = cache.report()

    assert report["tokens_prefilled"] == 110
    assert report["tokens_reused"] == 120
    assert report["reuse_pct"] == pytest.approx(52.2, abs=0.1)
    assert report["turns"] == 2
    assert report["held_blocks"] == 0  # claimed out of the cache


def test_session_kv_is_on_unless_a_measurement_turns_it_off(monkeypatch):
    monkeypatch.delenv("QWEN3_SESSION_KV", raising=False)
    assert session_kv_enabled() is True

    monkeypatch.setenv("QWEN3_SESSION_KV", "0")
    assert session_kv_enabled() is False
