from tests.cpu.test_engine import _engine
from bench.session_kv_replay import group_task_turns, run_session_kv_on_engine


def test_group_task_turns_keeps_dump_order():
    items = [
        {"task_id": "b", "turn_id": 0},
        {"task_id": "a", "turn_id": 1},
        {"task_id": "a", "turn_id": 0},
        {"task_id": "b", "turn_id": 1},
    ]
    groups = group_task_turns(items)
    assert [g[0]["task_id"] for g in groups] == ["b", "a"]
    assert [t["turn_id"] for t in groups[1]] == [0, 1]


def test_session_kv_teacher_force_appends_only_suffix():
    engine = _engine(token_budget=64, num_blocks=32, block_size=4)
    items = [
        {
            "task_id": "t0",
            "turn_id": 0,
            "input_ids": [1, 2, 3],
            "output_ids": [8, 9],
            "max_tokens": 2,
        },
        {
            "task_id": "t0",
            "turn_id": 1,
            "input_ids": [1, 2, 3, 8, 9, 20, 21],
            "output_ids": [22],
            "max_tokens": 1,
        },
    ]
    traces, extra = run_session_kv_on_engine(engine, items, nvtx=False)
    assert extra["forced_prefix_ok"] is True
    assert extra["n_full_add_turns"] == 1
    assert extra["n_resumed_turns"] == 1
    assert extra["appended_prompt_tokens"] == 3 + 2
    assert extra["prompt_tokens"] == 3 + 7
    assert [list(t.tokens) for t in traces] == [[8, 9], [22]]
    assert extra["later_turn_prefill_tokens"] > 0
    assert extra["first_turn_prefill_tokens"] >= 3
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_session_kv_reports_source_forced_length_mismatch():
    engine = _engine(token_budget=64, num_blocks=32, block_size=4)
    items = [
        {
            "task_id": "t0",
            "turn_id": 0,
            "input_ids": [1, 2, 3],
            "output_ids": [8],
            "max_tokens": 2,
        }
    ]

    _traces, extra = run_session_kv_on_engine(engine, items, nvtx=False)

    assert extra["source_forced_length_ok"] is False
    assert extra["source_forced_length_mismatches"] == [
        {"task_id": "t0", "turn_id": 0, "max_tokens": 2, "output_len": 1}
    ]
