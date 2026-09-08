import json

from qwen3_runtime.rollout.logprobs import (
    load_sidecar,
    mask_coverage,
    place_turn_logprobs,
    turns_from_record,
)


def test_a_turn_lands_on_its_own_span_and_nowhere_else():
    seq = [9, 9, 1, 2, 3, 9, 9]
    out, stats = place_turn_logprobs(seq, [([1, 2, 3], [-0.5, -0.25, -0.125])])
    assert out == [0.0, 0.0, -0.5, -0.25, -0.125, 0.0, 0.0]
    assert stats["turns_placed"] == 1
    assert stats["tokens_placed"] == 3
    assert stats["covered"] == [False, False, True, True, True, False, False]


def test_repeated_tool_call_binds_left_to_right_not_twice_to_the_first():
    # An agent that runs the same command twice emits the same token chunk twice.
    seq = [7, 7, 0, 7, 7]
    out, stats = place_turn_logprobs(seq, [([7, 7], [-1.0, -2.0]), ([7, 7], [-3.0, -4.0])])
    assert out == [-1.0, -2.0, 0.0, -3.0, -4.0]
    assert stats["turns_placed"] == 2


def test_a_turn_that_did_not_survive_retokenization_costs_only_itself():
    # Middle turn is absent from the sequence; the turn after it must still land.
    seq = [1, 1, 5, 5]
    out, stats = place_turn_logprobs(
        seq,
        [([1, 1], [-0.5, -0.5]), ([4, 4], [-9.0, -9.0]), ([5, 5], [-0.25, -0.25])],
    )
    assert out == [-0.5, -0.5, -0.25, -0.25]
    assert stats["turns_placed"] == 2
    assert stats["turns_unplaced"] == 1
    assert stats["tokens_unplaced"] == 2


def test_output_is_always_the_length_skyrl_asserts_on():
    seq = list(range(50))
    out, _ = place_turn_logprobs(seq, [([3, 4], [-1.0, -1.0])])
    assert len(out) == len(seq)


def test_sidecar_drops_records_whose_logprobs_do_not_pair_with_tokens(tmp_path):
    p = tmp_path / "sidecar.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"token_ids": [1, 2], "logprobs": [-0.1, -0.2]}),
                json.dumps({"token_ids": [3, 4], "logprobs": [-0.3]}),  # truncated
                json.dumps({"token_ids": [], "logprobs": []}),
                "",
            ]
        )
    )
    idx = load_sidecar(str(p))
    assert list(idx) == [(1, 2)]


def test_identical_completions_stay_distinguishable_in_file_order(tmp_path):
    p = tmp_path / "sidecar.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"token_ids": [8], "logprobs": [-1.0]}),
                json.dumps({"token_ids": [8], "logprobs": [-2.0]}),
            ]
        )
    )
    idx = load_sidecar(str(p))
    rec = {"messages": [{"kind": "TokenEvent", "response_token_ids": [8]}] * 2}
    turns, missing = turns_from_record(rec, idx)
    assert missing == 0
    assert [lp for _, lp in turns] == [[-1.0], [-2.0]]


def test_turns_with_no_sidecar_entry_are_counted_not_silently_dropped():
    rec = {"messages": [{"kind": "TokenEvent", "response_token_ids": [42]}]}
    turns, missing = turns_from_record(rec, {})
    assert turns == []
    assert missing == 1


def test_coverage_reports_the_trained_span_the_trainer_will_actually_use():
    # Position 3 is trained but carries no sampling-time logprob.
    cov = mask_coverage([0, 1, 1, 1], [False, True, True, False])
    assert cov == {
        "trained_positions": 3,
        "trained_with_rollout_logprob": 2,
        "trained_without_rollout_logprob": 1,
    }
