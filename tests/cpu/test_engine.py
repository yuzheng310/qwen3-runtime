from __future__ import annotations

from collections.abc import Sequence

import pytest

from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.request import Request
from qwen3_runtime.engine.runner import ModelRunner
from qwen3_runtime.sampling import record_sampled


class NullLifecycleRunner(ModelRunner):
    """Test runner with the ABC lifecycle methods as no-ops."""

    def __init__(self) -> None:
        self.model = None
        self.pool = None

    def bind(self, block_manager: BlockManager) -> None:
        del block_manager

    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        raise NotImplementedError

    def run_logits(self, reqs: Sequence[Request]):
        raise RuntimeError("speculative decoding needs a runner with run_logits")

    def release_kv_pool(self) -> None:
        return

    def offload_weights(self, *, keep_a_host_copy: bool = False) -> None:
        del keep_a_host_copy

    def reload_weights(self) -> None:
        return

    def rebuild_kv_pool(self) -> None:
        return


class FakeModelRunner(NullLifecycleRunner):
    """Token ids depend only on the request, not on batch mates.

    Every returned token is scored, because that is the ``ModelRunner.run``
    contract: the engine will not emit a token this runner cannot account for.
    The value is a stand-in -- there is no distribution behind these ids -- but
    it has to be there, and there has to be exactly one per token.
    """

    FAKE_LOGPROB = -0.5

    def run(self, reqs: list[Request]) -> list[int | None]:
        out: list[int | None] = []
        for req in reqs:
            will_complete_prefill = (
                req.num_computed_tokens + req.num_scheduled_tokens >= req.num_prompt_tokens
            )
            if not will_complete_prefill:
                out.append(None)
                continue
            forced = req.next_forced_token()
            if forced is not None:
                out.append(forced)
                record_sampled(req, self.FAKE_LOGPROB)
                continue
            out.append((req.token_ids[0] + req.num_computed_tokens) % 97 + 1)
            record_sampled(req, self.FAKE_LOGPROB)
        return out


def _engine(
    token_budget=2048,
    max_seqs=8,
    num_blocks=128,
    block_size=16,
    enable_prefix_cache=False,
) -> Engine:
    cfg = Config(
        max_num_batched_tokens=token_budget,
        max_num_seqs=max_seqs,
        num_kv_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_cache=enable_prefix_cache,
    )
    return Engine(cfg, FakeModelRunner())


def test_generate_emits_max_tokens_for_a_short_prompt():
    engine = _engine()
    prompt = [10, 11, 12]
    completion = engine.generate(prompt, max_tokens=4)
    assert len(completion) == 4


def test_finished_generation_does_not_remain_in_request_registry():
    engine = _engine()

    engine.generate([1, 2, 3], max_tokens=2)

    assert engine._requests == {}


def test_add_rejects_non_positive_max_tokens_without_registering_request():
    engine = _engine()

    with pytest.raises(ValueError, match="max_tokens"):
        engine.add_request([1, 2, 3], max_tokens=0)

    assert engine._requests == {}
    assert engine.scheduler.is_finished()


def test_add_capacity_failure_does_not_register_request():
    engine = _engine(num_blocks=1, block_size=4)

    with pytest.raises(RuntimeError, match="needs"):
        engine.add_request([1, 2, 3, 4], max_tokens=1)

    assert engine._requests == {}
    assert engine.scheduler.is_finished()


def test_request_alone_matches_request_inside_batch():
    prompt_a = [3, 4, 5, 6]
    prompt_b = list(range(20, 40))

    alone = _engine(token_budget=64, max_seqs=4)
    tokens_alone = alone.generate(prompt_a, max_tokens=3)

    batched = _engine(token_budget=64, max_seqs=4)
    id_a = batched.add_request(prompt_a, max_tokens=3)
    batched.add_request(prompt_b, max_tokens=2)
    got = {id_a: []}
    while not batched.is_finished():
        for rid, tok, _done in batched.step():
            if rid == id_a and tok is not None:
                got[id_a].append(tok)

    assert got[id_a] == tokens_alone
