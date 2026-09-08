"""Rebuild SkyRL rollouts from CodeScout traj JSON (Task B train-only)."""

import json
from pathlib import Path

from qwen3_runtime.integrations.skyrl.replay_traj import (
    index_traj_dir,
    patch_codescout_replay_traj,
    rollout_from_record,
)


class _Tok:
    def convert_tokens_to_ids(self, tok: str) -> int:
        return {"<|im_start|>": 151644, "assistant": 77091}[tok]


def test_index_and_concat_token_events(tmp_path: Path):
    rec = {
        "instance_id": "repo__task",
        "total_reward": 2.0,
        "reward_dict": {"multilevel_localization_f1_reward": 2.0},
        "metrics_dict": {"tokens": 10, "steps": 2},
        "messages": [
            {
                "kind": "TokenEvent",
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5],
            },
            {
                "kind": "ActionEvent",
                "tool_name": "localization_finish",
            },
            {
                "kind": "TokenEvent",
                "prompt_token_ids": [1, 2, 3, 4, 5, 8, 9],
                "response_token_ids": [10, 11],
            },
        ],
    }
    path = tmp_path / "step_1" / "train" / "repo__task_3.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec))
    indexed = index_traj_dir(str(tmp_path))
    assert ("repo__task", "3") in indexed
    rows = rollout_from_record(rec, tokenizer=_Tok(), model_name="CodeScout-4B")
    resp, reward, stop, mask, prompt, _, metrics = rows[0]
    assert reward == 2.0
    assert stop == "complete"
    assert prompt == [1, 2, 3]
    assert resp == [4, 5, 8, 9, 10, 11]
    assert len(mask) == len(resp)
    assert all(x in (0, 1) for x in mask)
    assert metrics["tokens"] == 10


def test_missing_token_events_are_error_rollouts():
    rec = {
        "instance_id": "x",
        "total_reward": 0.0,
        "messages": [{"kind": "ConversationErrorEvent"}],
    }
    rows = rollout_from_record(rec, tokenizer=_Tok(), model_name="CodeScout-4B")
    assert rows[0][2] == "error"
    assert rows[0][0] == [151643]
    assert rows[0][3] == [0]


def test_exhausted_without_finish_zeros_mask():
    rec = {
        "instance_id": "x",
        "total_reward": 0.0,
        "messages": [
            {
                "kind": "TokenEvent",
                "prompt_token_ids": [1],
                "response_token_ids": [2],
            }
            for _ in range(6)
        ],
    }
    rec["messages"][-1]["prompt_token_ids"] = [1, 9]
    rec["messages"][-1]["response_token_ids"] = [3, 4]
    rows = rollout_from_record(rec, tokenizer=_Tok(), model_name="CodeScout-4B")
    assert rows[0][3] == [0] * len(rows[0][0])


def test_replay_patch_noop_without_env(monkeypatch):
    monkeypatch.delenv("QWEN3_REPLAY_TRAJ_DIR", raising=False)
    assert patch_codescout_replay_traj() == ""
