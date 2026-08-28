import importlib.util
import json
from pathlib import Path

import pytest


def _load_adapter():
    path = Path(__file__).resolve().parents[2] / "scripts" / "codescout_openai_server.py"
    spec = importlib.util.spec_from_file_location("codescout_openai_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decode_content_strips_stop_strings():
    mod = _load_adapter()

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "call()<|im_end|>\n"

    assert mod.decode_content(Tok(), [1], include_stop_str=False) == "call()"
    assert "<|im_end|>" in mod.decode_content(Tok(), [1], include_stop_str=True)


def test_parse_qwen_tool_calls_hermes_xml():
    mod = _load_adapter()
    text = (
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>'
    )
    calls = mod.parse_qwen_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "terminal"
    assert json.loads(calls[0]["function"]["arguments"])["command"] == "ls"
    assert mod.parse_qwen_tool_calls("no tools") == []


def test_normalize_messages_flattens_content_parts():
    mod = _load_adapter()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}
    ]
    assert mod.normalize_messages(messages)[0]["content"] == "hello world"


def test_tokenize_chat_accepts_hf_batch_encoding_dict():
    mod = _load_adapter()

    class Tok:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [10, 11, 12]}

    assert mod.tokenize_chat(Tok(), [{"role": "user", "content": "x"}], None, {}) == [10, 11, 12]


def test_reset_sessions_finishes_paused_and_queued():
    mod = _load_adapter()

    class Req:
        def __init__(self, request_id):
            self.request_id = request_id

    class FakeSched:
        def __init__(self):
            self.paused = {7: Req(7)}
            self.waiting = [Req(8)]
            self.running = []

    class FakeBM:
        num_blocks = 10
        num_free_blocks = 3

    class FakeEngine:
        def __init__(self):
            self.scheduler = FakeSched()
            self.block_manager = FakeBM()
            self.finished = []

        def finish_request(self, rid):
            self.finished.append(rid)
            self.scheduler.paused.pop(rid, None)

    engine = FakeEngine()
    state = mod.RolloutState(engine, tokenizer=None, model_name="x")
    assert state.reset_sessions() == 2
    assert engine.finished == [7, 8]
    snap = state.snapshot_stats()
    assert snap["paused_sessions"] == 0
    assert snap["kv_used_blocks"] == 7


def test_adapter_defaults_to_loopback_and_does_not_trust_remote_code():
    mod = _load_adapter()
    args = mod.build_parser().parse_args(["--model", "/tmp/model"])

    assert args.host == "127.0.0.1"
    assert args.trust_remote_code is False


def test_adapter_refuses_remote_bind_without_explicit_override():
    mod = _load_adapter()
    args = mod.build_parser().parse_args(["--model", "/tmp/model", "--host", "0.0.0.0"])

    with pytest.raises(SystemExit, match="allow-remote"):
        mod.validate_bind(args)


def test_continuation_failure_does_not_evict_unrelated_paused_session():
    mod = _load_adapter()

    class Req:
        request_id = 7

    class FakeSched:
        paused = {7: Req()}
        waiting = []
        running = []

    class FakeBM:
        num_blocks = 10
        num_free_blocks = 5

    class FakeEngine:
        scheduler = FakeSched()
        block_manager = FakeBM()

        def __init__(self):
            self.finished = []

        def add_request(self, *_args, **_kwargs):
            raise RuntimeError("KV exhausted")

        def finish_request(self, rid):
            self.finished.append(rid)
            self.scheduler.paused.pop(rid, None)

    engine = FakeEngine()
    state = mod.RolloutState(engine, tokenizer=None, model_name="x")

    with pytest.raises(RuntimeError, match="KV exhausted"):
        state._continue_or_add([1, 2], max_tokens=1)

    assert engine.finished == []
    assert 7 in engine.scheduler.paused


def test_adapter_resumes_only_its_explicit_active_request():
    mod = _load_adapter()

    class Req:
        def __init__(self, request_id, token_ids):
            self.request_id = request_id
            self.token_ids = token_ids

    class FakeSched:
        def __init__(self):
            self.paused = {
                7: Req(7, [1, 2, 3]),
                8: Req(8, [1, 2, 3, 4]),
            }
            self.waiting = []
            self.running = []

    class FakeBM:
        num_blocks = 10
        num_free_blocks = 5

    class FakeEngine:
        def __init__(self):
            self.scheduler = FakeSched()
            self.block_manager = FakeBM()
            self.resumed = []

        def resume_request(self, rid, suffix, max_tokens, **kwargs):
            self.resumed.append((rid, suffix, max_tokens, kwargs))

        def add_request(self, *_args, **_kwargs):
            raise AssertionError("should resume the active request")

    state = mod.RolloutState(FakeEngine(), tokenizer=None, model_name="x")
    state.active_request_id = 7

    rid = state._continue_or_add([1, 2, 3, 9], max_tokens=2, hold_kv=True)

    assert rid == 7
    assert state.engine.resumed[0][:3] == (7, [9], 2)


def test_load_task_ids_prefer_small():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_level2_live_subset.py"
    spec = importlib.util.spec_from_file_location("run_level2_live_subset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    subset = Path(__file__).resolve().parents[2] / "workloads" / "code_localization" / "replay_subset_v1.json"
    ids = mod.load_task_ids(subset, 5, prefer_small=True)
    assert len(ids) == 5
    assert all(tid.startswith(("pytest-dev__", "psf__")) for tid in ids)
    assert ids[0] == "pytest-dev__pytest-6202"
