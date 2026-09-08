import json
from pathlib import Path

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.factory import PIN, CODESCOUT_PIN, resolve_pin
from qwen3_runtime.serving.slo_harness import run_sequential_requests
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig
from qwen3_runtime.utils.loader import PIN_KEYS
from tests.cpu.test_engine import FakeModelRunner
from workloads.code_localization.analyze_prefix import classify_transition, lcp_len
from workloads.code_localization.reconstruct_tokens import (
    generated_output_ids,
    select_stratified_task_ids,
    strip_trailing_im_end_newline,
    IM_END,
    NL,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_SET = ROOT / "workloads" / "code_localization" / "task_set_v1.json"
RECON = ROOT / "workloads" / "code_localization" / "token_reconstruction_v1.json"
SUBSET = ROOT / "workloads" / "code_localization" / "replay_subset_v1.json"


def test_codescout_pin_is_instruct2507_rope_not_stage1():
    stage1 = json.loads(PIN.read_text())
    scout = json.loads(CODESCOUT_PIN.read_text())
    assert resolve_pin() == PIN
    assert resolve_pin(pin_path=CODESCOUT_PIN) == CODESCOUT_PIN
    cfg = Qwen3ModelConfig.from_hf_config(scout)
    assert cfg.rope_theta == 5_000_000
    assert cfg.max_position_embeddings == 262144
    for key in PIN_KEYS:
        if key in ("rope_theta",):
            assert scout[key] != stage1[key]
            continue
        assert scout[key] == stage1[key]


def test_generated_output_strips_only_im_end_newline():
    prompt = [1, 2, 3]
    body = [10, 11, IM_END]
    assert generated_output_ids(prompt, prompt + body + [NL]) == body
    assert strip_trailing_im_end_newline(body) == body


def test_sequential_replay_honors_per_request_max_tokens():
    engine = Engine(
        Config(max_num_batched_tokens=32, max_num_seqs=1, num_kv_blocks=32, block_size=4),
        FakeModelRunner(),
    )
    traces = run_sequential_requests(
        engine, [([7, 8], 2), ([9, 10, 11], 5)], nvtx=False
    )
    assert [len(t.tokens) for t in traces] == [2, 5]
    assert all(t.forced_length_ok if False else len(t.tokens) == t.max_tokens for t in traces)


def test_replay_subset_is_100_stratified_and_in_task_set():
    tasks = json.loads(TASK_SET.read_text())["tasks"]
    ids = select_stratified_task_ids(tasks, 100)
    assert len(ids) == 100
    assert len(set(ids)) == 100
    all_ids = {t["task_id"] for t in tasks}
    assert set(ids) <= all_ids
    if SUBSET.exists():
        frozen = json.loads(SUBSET.read_text())
        assert frozen["task_ids"] == ids


def test_exact_lcp_and_session_continuation_classifier():
    prompt = [1, 2, 3, 4]
    completion = [9, IM_END]
    nxt = prompt + completion + [20, 21]
    row = classify_transition(prompt, completion, nxt)
    assert lcp_len(prompt, nxt) == 4
    assert row["prompt_is_exact_prefix"] is True
    assert row["session_is_exact_prefix"] is True
    assert row["kind"] == "exact_append_session"
    assert row["new_suffix_tokens"] == 2

    nxt_nl = prompt + completion + [NL, 20]
    row_nl = classify_transition(prompt, completion, nxt_nl)
    assert row_nl["session_is_exact_prefix"] is True

    mutated = [1, 2, 99, 4, 5]
    row_m = classify_transition(prompt, completion, mutated)
    assert row_m["kind"] == "internal_prefix_mutation"
    assert row_m["lcp_prompt_tokens"] == 2

    truncated_src = list(range(40))
    truncated = list(range(10, 50))
    row_t = classify_transition(truncated_src, completion, truncated)
    assert row_t["left_truncation"] is True
    assert row_t["kind"] == "left_truncation"


def test_prefix_analysis_v1_session_continuation_is_exact():
    path = ROOT / "workloads" / "code_localization" / "prefix_analysis_v1.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    full = data["full_trace_v1"]
    assert full["n_adjacent_pairs"] == 1901
    assert full["prompt_t_is_prefix_of_next"] == 1.0
    assert full["session_is_prefix_of_next"] == 1.0
    assert full["kind_counts"]["exact_append_session"] == 1901
    assert full["kind_counts"].get("left_truncation", 0) == 0
    assert full["causes_if_session_not_exact"]["left_truncation"] == 0
    assert full["causes_if_session_not_exact"]["internal_prefix_mutation"] == 0
    assert full["new_suffix_tokens"]["p50"] == 1633
    assert round(full["token_budget"]["reusable_fraction_of_all_prompts"], 3) == 0.706
    sub = data["replay_subset_v1"]
    assert sub["n_adjacent_pairs"] == 371
    assert sub["session_is_prefix_of_next"] == 1.0
    assert sub["kind_counts"]["exact_append_session"] == 371
    assert sub["new_suffix_tokens"]["p50"] == 1882
    assert data["length_based_stage2_estimate"]["status"] == (
        "approximate_length_stability_lower_bound"
    )
    if not RECON.exists():
        return
    data = json.loads(RECON.read_text())
    assert data["n_requests"] == 2395
    assert data["prefix_stable_requests"] == data["n_requests"]
    assert data["generation_prompt_stable_requests"] == data["n_requests"]
    assert data["input_delta"]["unique"] == [28]
    assert data["requests_output_match"] == 2387
    assert data["output_abs_err"]["max"] == 1


def test_apply_item_limits_keeps_all_turns_of_first_n_tasks():
    from bench.replay_code_localization import apply_item_limits, parse_speculative_config

    items = [
        {"task_id": "a", "turn_id": 0},
        {"task_id": "a", "turn_id": 1},
        {"task_id": "b", "turn_id": 0},
        {"task_id": "c", "turn_id": 0},
        {"task_id": "c", "turn_id": 1},
        {"task_id": "c", "turn_id": 2},
    ]
    sliced = apply_item_limits(items, limit_tasks=2)
    assert [it["task_id"] for it in sliced] == ["a", "a", "b"]
    reqs = apply_item_limits(items, limit_requests=3)
    assert len(reqs) == 3
    both = apply_item_limits(items, limit_tasks=2, limit_requests=2)
    assert [it["task_id"] for it in both] == ["a", "a"]
    cfg = parse_speculative_config(
        '{"method": "ngram", "prompt_lookup_min": 5, "prompt_lookup_max": 5, "num_speculative_tokens": 5}'
    )
    assert cfg["method"] == "ngram"
    assert cfg["num_speculative_tokens"] == 5
    assert parse_speculative_config(None) is None
