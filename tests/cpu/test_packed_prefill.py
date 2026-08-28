from qwen3_runtime.engine.batch import last_token_indices, sampled_logit_rows, select_last_rows


def test_last_token_indices_for_unequal_packed_prefill():
    # Two requests packed: lengths 3 and 5 → last computed rows 2 and 7.
    cu_seqlens_q = [0, 3, 8]
    assert last_token_indices(cu_seqlens_q) == [2, 7]


def test_select_last_rows_does_not_assume_one_row_per_request():
    packed = ["a0", "a1", "a2", "b0", "b1"]
    cu_seqlens_q = [0, 3, 5]
    assert select_last_rows(packed, cu_seqlens_q) == ["a2", "b1"]


def test_chunked_prefill_selects_last_computed_token_not_prompt_end():
    # Prompt length 8, this chunk computes tokens [4, 7) → one request, last index 3 in the packed chunk.
    cu_seqlens_q = [0, 4]
    assert last_token_indices(cu_seqlens_q) == [3]


def test_sampled_logit_rows_skips_incomplete_prefill():
    cu = [0, 4, 5]
    assert sampled_logit_rows(cu, [False, True]) == [4]
    assert sampled_logit_rows(cu, [False, False]) == []
    assert sampled_logit_rows(cu, [True, True]) == [3, 4]
