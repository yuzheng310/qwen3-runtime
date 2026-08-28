from qwen3_runtime.config import Config
from qwen3_runtime.engine.block_manager import BlockManager
from qwen3_runtime.engine.request import Request, RequestStatus
from qwen3_runtime.engine.scheduler import Scheduler


def _scheduler(
    num_blocks=64,
    block_size=16,
    max_seqs=4,
    token_budget=512,
) -> Scheduler:
    cfg = Config(
        block_size=block_size,
        num_kv_blocks=num_blocks,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=token_budget,
    )
    return Scheduler(cfg, BlockManager(num_blocks=num_blocks, block_size=block_size))


def test_chunked_prefill_admits_only_chunk_kv_not_full_prompt():
    # Spec example: 8192-token prompt, chunk=512, block_size=16.
    # Full prompt needs 512 blocks; the first chunk must occupy 32.
    # Pool is large enough to *finish* the request (add() rejects otherwise).
    sched = _scheduler(num_blocks=513, block_size=16, token_budget=512)
    req = Request(token_ids=list(range(8192)), max_tokens=1)
    sched.add(req)

    batch = sched.schedule()
    assert [r.request_id for r in batch] == [req.request_id]
    assert req.num_scheduled_tokens == 512
    assert len(req.block_table) == 32
    assert sched.block_manager.num_free_blocks == 481
    assert req.status == RequestStatus.RUNNING


def test_token_budget_chunks_a_long_prompt_across_steps():
    sched = _scheduler(num_blocks=256, block_size=16, token_budget=128)
    req = Request(token_ids=list(range(300)), max_tokens=1)
    sched.add(req)

    batch = sched.schedule()
    assert req.num_scheduled_tokens == 128
    sched.postprocess(batch, token_ids=[None])
    assert req.num_computed_tokens == 128
    assert not req.is_prefill_complete

    batch = sched.schedule()
    assert req.num_scheduled_tokens == 128
    sched.postprocess(batch, token_ids=[None])
    batch = sched.schedule()
    assert req.num_scheduled_tokens == 44  # 300 - 256


def test_two_waiting_prefills_share_token_budget():
    sched = _scheduler(num_blocks=64, block_size=8, max_seqs=4, token_budget=10)
    a = Request(token_ids=list(range(8)), max_tokens=1)
    b = Request(token_ids=list(range(8, 16)), max_tokens=1)
    sched.add(a)
    sched.add(b)

    batch = sched.schedule()
    assert len(batch) == 2
    assert a.num_scheduled_tokens == 8
    assert b.num_scheduled_tokens == 2  # remainder of budget
    assert len(a.block_table) == 1
    assert len(b.block_table) == 1


def test_full_budget_prefill_starves_peer_waiting_decode():
    """A 9981-token waiting prefill takes the entire 2048 budget; a decode-ready
    peer in waiting is not scheduled in the same step. This is the mixed-load HOL.
    """
    sched = _scheduler(num_blocks=2048, block_size=16, max_seqs=8, token_budget=2048)
    long = Request(token_ids=list(range(9981)), max_tokens=8)
    short = Request(token_ids=list(range(16)), max_tokens=8)
    sched.add(long)
    sched.add(short)
    batch = sched.schedule()
    assert long in batch and short not in batch
    assert long.num_scheduled_tokens == 2048
    sched.postprocess(batch, token_ids=[None])
    batch = sched.schedule()
    assert long in batch and short not in batch
    assert long.num_scheduled_tokens == 2048


def test_unified_schedule_mixes_decode_and_new_prefill():
    sched = _scheduler(num_blocks=32, block_size=4, max_seqs=4, token_budget=16)
    decode_req = Request(token_ids=list(range(4)), max_tokens=4)
    sched.add(decode_req)
    batch = sched.schedule()
    sched.postprocess(batch, token_ids=[99])
    assert decode_req.is_prefill_complete
    assert decode_req.status == RequestStatus.RUNNING

    prefill_req = Request(token_ids=list(range(40)), max_tokens=1)
    sched.add(prefill_req)
    batch = sched.schedule()
    assert decode_req in batch
    assert prefill_req in batch
    assert decode_req.num_scheduled_tokens == 1
    assert prefill_req.num_scheduled_tokens == 15
    assert not prefill_req.is_prefill_complete


def test_waiting_prefill_shrinks_to_remaining_free_blocks():
    sched = _scheduler(num_blocks=4, block_size=16, token_budget=512)
    a = Request(token_ids=list(range(31)), max_tokens=1)
    b = Request(token_ids=list(range(40)), max_tokens=1)
    sched.add(a)
    sched.add(b)
    batch = sched.schedule()
    assert a in batch and b in batch
    assert a.num_scheduled_tokens == 31
    assert b.num_scheduled_tokens == 32
    assert len(b.block_table) == 2
    assert not b.is_prefill_complete


def test_preempt_youngest_running_when_decode_needs_a_new_block():
    sched = _scheduler(num_blocks=4, block_size=4, max_seqs=4, token_budget=32)
    a = Request(token_ids=list(range(8)), max_tokens=4)
    b = Request(token_ids=list(range(8, 16)), max_tokens=4)
    sched.add(a)
    sched.add(b)
    batch = sched.schedule()
    assert {r.request_id for r in batch} == {a.request_id, b.request_id}
    sched.postprocess(batch, token_ids=[11, 22])
    assert a.is_prefill_complete and b.is_prefill_complete
    assert sched.block_manager.num_free_blocks == 0

    batch = sched.schedule()
    assert sched.num_preemptions >= 1
    assert batch
    assert any(r.num_computed_tokens == 0 and r.num_scheduled_tokens > 1 for r in batch)


def test_add_rejects_sequence_that_cannot_fit_in_the_pool():
    sched = _scheduler(num_blocks=2, block_size=16)
    req = Request(token_ids=list(range(48)), max_tokens=1)
    try:
        sched.add(req)
    except RuntimeError as exc:
        assert "needs" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_throughput_shaped_2048_budget_two_prefill_waves():
    # 16×256 prompts at budget 2048 schedule 8 then 8 (the C gap vs vLLM 8192).
    sched = _scheduler(num_blocks=512, block_size=16, max_seqs=16, token_budget=2048)
    reqs = [Request(token_ids=list(range(256)), max_tokens=128) for _ in range(16)]
    for req in reqs:
        sched.add(req)
    first = sched.schedule()
    assert len(first) == 8
    assert sum(r.num_scheduled_tokens for r in first) == 2048
    sched.postprocess(first, token_ids=[1] * 8)
    second = sched.schedule()
    # First 8 start decode while the rest prefill (mixed step).
    assert sum(1 for r in second if r.num_scheduled_tokens == 1) == 8
    assert sum(r.num_scheduled_tokens for r in second) == 2048


def test_throughput_shaped_4096_budget_one_prefill_wave():
    sched = _scheduler(num_blocks=512, block_size=16, max_seqs=16, token_budget=4096)
    reqs = [Request(token_ids=list(range(256)), max_tokens=128) for _ in range(16)]
    for req in reqs:
        sched.add(req)
    batch = sched.schedule()
    assert len(batch) == 16
    assert sum(r.num_scheduled_tokens for r in batch) == 4096


def test_hold_kv_pause_keeps_blocks_and_resume_only_uncomputed_suffix():
    sched = _scheduler(num_blocks=16, block_size=4, max_seqs=2, token_budget=32)
    req = Request(token_ids=[1, 2, 3, 4], max_tokens=2)
    req.hold_kv = True
    sched.add(req)
    while req.status != RequestStatus.PAUSED:
        batch = sched.schedule()
        assert batch
        toks = [10] * len(batch)
        sched.postprocess(batch, toks)
    assert req.status == RequestStatus.PAUSED
    assert req.request_id in sched.paused
    free_paused = sched.block_manager.num_free_blocks
    assert free_paused < sched.block_manager.num_blocks
    computed = req.num_computed_tokens
    assert req.uncomputed_tokens == 1
    sched.resume(req, [7, 8], 1, hold_kv=False)
    assert req.status == RequestStatus.WAITING
    assert req.uncomputed_tokens == 3
    assert req.num_computed_tokens == computed
    assert sched.block_manager.num_free_blocks == free_paused
    batch = sched.schedule()
    assert req.num_scheduled_tokens == 3
    sched.postprocess(batch, [None])
    batch = sched.schedule()
    sched.postprocess(batch, [11])
    assert req.status == RequestStatus.FINISHED
    assert sched.paused == {}


def test_failed_resume_is_atomic_and_keeps_paused_session():
    sched = _scheduler(num_blocks=2, block_size=4, max_seqs=2, token_budget=16)
    req = Request(token_ids=[1, 2, 3], max_tokens=1)
    req.hold_kv = True
    sched.add(req)
    while req.status != RequestStatus.PAUSED:
        batch = sched.schedule()
        sched.postprocess(batch, [9])

    token_ids = list(req.token_ids)
    block_table = list(req.block_table)
    computed = req.num_computed_tokens
    free_blocks = sched.block_manager.num_free_blocks

    try:
        sched.resume(req, [10, 11, 12, 13], 4, hold_kv=False)
    except RuntimeError as exc:
        assert "needs" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert req.status == RequestStatus.PAUSED
    assert sched.paused[req.request_id] is req
    assert req not in sched.waiting
    assert req.token_ids == token_ids
    assert req.block_table == block_table
    assert req.num_computed_tokens == computed
    assert sched.block_manager.num_free_blocks == free_blocks
