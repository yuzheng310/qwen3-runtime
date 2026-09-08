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


def test_assistant_chat_message_drops_xml_content_when_tool_calls_present():
    """Qwen template would otherwise emit content XML and tool_calls XML."""
    mod = _load_adapter()
    xml = '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>'
    calls = mod.parse_qwen_tool_calls(xml)
    msg = mod.assistant_chat_message(xml, calls)
    assert msg["content"] == ""
    assert msg["tool_calls"] == calls
    mixed = "note\n" + xml
    keep = mod.assistant_chat_message(mixed, calls)
    assert keep["content"] == mixed
    assert keep["tool_calls"] == calls
    plain = mod.assistant_chat_message("done", [])
    assert plain == {"role": "assistant", "content": "done"}


def test_plan_session_continuation_exact_and_replace():
    mod = _load_adapter()
    held = [1, 2, 3, 9]
    assert mod.plan_session_continuation(held, [1, 2, 3, 9, 10, 11]) == ("exact", [10, 11])
    assert mod.plan_session_continuation(held, [1, 2, 3, 8, 10]) == ("replace", [])
    assert mod.plan_session_continuation(held, [9, 8, 7]) == ("replace", [])
    assert mod.plan_session_continuation(held, [1, 2]) == ("replace", [])


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
    assert args.max_num_seqs == 16
    assert args.enable_prefix_cache is False


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
    assert state.stats["n_resume_exact"] == 1
    assert state.stats["appended_prompt_tokens"] == 1
    assert "rewind_to" not in state.engine.resumed[0][3]


def test_parse_adapter_path_routes_session_and_default():
    mod = _load_adapter()
    assert mod.parse_adapter_path("/v1/chat/completions") == ("chat", "default")
    assert mod.parse_adapter_path("/v1/s/g0s3/chat/completions") == ("chat", "g0s3")
    assert mod.parse_adapter_path("/v1/s/g0s3/reset") == ("reset_one", "g0s3")
    assert mod.parse_adapter_path("/v1/reset") == ("reset_all", None)
    assert mod.parse_adapter_path("/v1/models") == ("other", None)


def test_identical_turn0_prompts_stay_on_disjoint_sessions():
    """GRPO group members send byte-identical turn-0 prompts. Prefix matching
    across sessions would silently cross-wire KV. Path keys must keep them disjoint.
    """
    mod = _load_adapter()

    class Req:
        def __init__(self, request_id, token_ids):
            self.request_id = request_id
            self.token_ids = token_ids

    class FakeSched:
        def __init__(self):
            self.paused = {}
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
            self.finished = []
            self._n = 0

        def add_request(self, token_ids, **_kwargs):
            self._n += 1
            rid = self._n
            self.scheduler.paused[rid] = Req(rid, list(token_ids))
            return rid

        def resume_request(self, rid, suffix, max_tokens, **kwargs):
            self.resumed.append((rid, list(suffix), max_tokens))
            self.scheduler.paused[rid].token_ids.extend(suffix)

        def finish_request(self, rid):
            self.finished.append(rid)
            self.scheduler.paused.pop(rid, None)

    engine = FakeEngine()
    state = mod.RolloutState(engine, tokenizer=None, model_name="x")
    prompt = [1, 2, 3]
    rid_a = state._continue_or_add(prompt, max_tokens=1, session_key="sA", hold_kv=True)
    rid_b = state._continue_or_add(prompt, max_tokens=1, session_key="sB", hold_kv=True)
    assert rid_a != rid_b
    assert state.sessions["sA"] == rid_a
    assert state.sessions["sB"] == rid_b

    rid_a2 = state._continue_or_add(prompt + [9], max_tokens=2, session_key="sA", hold_kv=True)
    rid_b2 = state._continue_or_add(prompt + [8], max_tokens=2, session_key="sB", hold_kv=True)
    assert rid_a2 == rid_a
    assert rid_b2 == rid_b
    assert engine.resumed == [(rid_a, [9], 2), (rid_b, [8], 2)]
    assert engine.finished == []
    assert engine.scheduler.paused[rid_a].token_ids == [1, 2, 3, 9]
    assert engine.scheduler.paused[rid_b].token_ids == [1, 2, 3, 8]


def test_diverged_turn_starts_a_new_request_on_the_same_key_and_leaves_the_peer():
    """A's re-tokenized turn-1 is not an exact extension of A's held sequence.

    Searching paused sessions by token prefix would resume B. Divergence must
    finish A, add a new request on A's URL key, and leave B untouched.
    """
    mod = _load_adapter()

    class Req:
        def __init__(self, request_id, token_ids):
            self.request_id = request_id
            self.token_ids = list(token_ids)

    class FakeSched:
        def __init__(self):
            self.paused = {}
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
            self.finished = []
            self._n = 0

        def add_request(self, token_ids, **_kwargs):
            self._n += 1
            rid = self._n
            self.scheduler.paused[rid] = Req(rid, list(token_ids))
            return rid

        def resume_request(self, rid, suffix, max_tokens, **kwargs):
            req = self.scheduler.paused[rid]
            req.token_ids.extend(suffix)
            self.resumed.append((rid, list(suffix), max_tokens, dict(kwargs)))

        def finish_request(self, rid):
            self.finished.append(rid)
            self.scheduler.paused.pop(rid, None)

    engine = FakeEngine()
    state = mod.RolloutState(engine, tokenizer=None, model_name="x")
    prompt = [1, 2, 3]
    rid_a = state._continue_or_add(prompt + [9], max_tokens=1, session_key="sA", hold_kv=True)
    rid_b = state._continue_or_add(prompt + [8], max_tokens=1, session_key="sB", hold_kv=True)
    assert rid_a != rid_b
    assert engine.scheduler.paused[rid_a].token_ids == [1, 2, 3, 9]
    assert engine.scheduler.paused[rid_b].token_ids == [1, 2, 3, 8]

    rid_a2 = state._continue_or_add(prompt + [8, 10], max_tokens=2, session_key="sA", hold_kv=True)
    assert rid_a2 != rid_a
    assert rid_a in engine.finished
    assert rid_b in engine.scheduler.paused
    assert engine.resumed == []
    assert engine.scheduler.paused[rid_b].token_ids == [1, 2, 3, 8]
    assert state.sessions["sA"] == rid_a2
    assert state.sessions["sB"] == rid_b


def test_adapter_diverged_turn_full_adds_instead_of_resume():
    """Re-tokenized turn-1 that is not an exact extension finishes and adds."""
    mod = _load_adapter()

    class Req:
        def __init__(self, request_id, token_ids):
            self.request_id = request_id
            self.token_ids = token_ids

    class FakeSched:
        def __init__(self):
            self.paused = {7: Req(7, [1, 2, 3, 9])}
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
            self.finished = []
            self.added = []

        def resume_request(self, rid, suffix, max_tokens, **kwargs):
            raise AssertionError("diverged turn must not resume")

        def add_request(self, token_ids, **_kwargs):
            self.added.append(list(token_ids))
            self.scheduler.paused[8] = Req(8, list(token_ids))
            return 8

        def finish_request(self, rid):
            self.finished.append(rid)
            self.scheduler.paused.pop(rid, None)

    engine = FakeEngine()
    state = mod.RolloutState(engine, tokenizer=None, model_name="x")
    state.sessions["default"] = 7
    rid = state._continue_or_add([1, 2, 3, 8, 10], max_tokens=2, hold_kv=True)
    assert rid == 8
    assert engine.finished == [7]
    assert engine.added == [[1, 2, 3, 8, 10]]
    assert engine.resumed == []
    assert state.stats["n_full_add"] == 1
    assert state.stats["n_resume_exact"] == 0


def test_engine_loop_two_sessions_identical_prompt_disjoint_ids():
    """HTTP-shaped chat_completions on the engine thread: identical turn-0, two keys."""
    import threading

    from qwen3_runtime.engine.request import RequestStatus

    mod = _load_adapter()

    class Req:
        def __init__(self, request_id, token_ids):
            self.request_id = request_id
            self.token_ids = list(token_ids)
            self.status = RequestStatus.WAITING
            self.block_table = [request_id * 10]

    class FakeSched:
        def __init__(self):
            self.paused = {}
            self.waiting = []
            self.running = []

    class FakeBM:
        num_blocks = 32
        num_free_blocks = 16

    class FakeEngine:
        def __init__(self):
            self.scheduler = FakeSched()
            self.block_manager = FakeBM()
            self._requests = {}
            self.last_emitted = {}
            self.last_step_stats = None
            self._n = 0

        def add_request(self, token_ids, **_kwargs):
            self._n += 1
            req = Req(self._n, token_ids)
            self.scheduler.waiting.append(req)
            self._requests[req.request_id] = req
            return req.request_id

        def resume_request(self, *_a, **_k):
            raise AssertionError("turn-0 should add, not resume")

        def finish_request(self, rid):
            self._requests.pop(rid, None)
            self.scheduler.paused.pop(rid, None)
            self.scheduler.waiting = [r for r in self.scheduler.waiting if r.request_id != rid]

        def step(self):
            self.last_emitted = {}
            moved = list(self.scheduler.waiting)
            self.scheduler.waiting.clear()
            for req in moved:
                req.status = RequestStatus.PAUSED
                self.scheduler.paused[req.request_id] = req
                self.last_emitted[req.request_id] = [90 + req.request_id]
            self.last_step_stats = {"decode_reqs": len(moved), "preempts": 0, "req_ids": [r.request_id for r in moved]}
            return []

    class Tok:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2, 3]

        def decode(self, ids, skip_special_tokens=False):
            return "ok"

    state = mod.RolloutState(FakeEngine(), Tok(), "x")
    body = {"messages": [{"role": "user", "content": "same"}]}
    got = {}

    def run(key):
        got[key] = state.chat_completions(body, session_key=key)

    threads = [threading.Thread(target=run, args=(k,)) for k in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()
    assert got["a"]["id"] != got["b"]["id"]
    assert got["a"]["choices"][0]["token_ids"] != got["b"]["choices"][0]["token_ids"]
    assert set(state.sessions) == {"a", "b"}
    assert state.sessions["a"] != state.sessions["b"]
    assert state.engine._requests[state.sessions["a"]].block_table != state.engine._requests[state.sessions["b"]].block_table


def test_per_session_reset_does_not_evict_peer():
    mod = _load_adapter()

    class Req:
        def __init__(self, request_id):
            self.request_id = request_id

    class FakeSched:
        def __init__(self):
            self.paused = {7: Req(7), 8: Req(8)}
            self.waiting = []
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
    state.sessions["a"] = 7
    state.sessions["b"] = 8
    assert state.reset_session("a") == 1
    assert engine.finished == [7]
    assert 8 in engine.scheduler.paused
    assert "b" in state.sessions
    assert "a" not in state.sessions


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
    assert mod.session_base_url("http://127.0.0.1:8080/v1/", "w3") == "http://127.0.0.1:8080/v1/s/w3/"
    assert mod.session_reset_url("http://127.0.0.1:8080/v1/", "w3") == "http://127.0.0.1:8080/v1/s/w3/reset"
    assert mod.session_reset_url("http://127.0.0.1:8080/v1/", None).endswith("/v1/reset")


def test_dump_thread_stacks_includes_caller():
    mod = _load_adapter()
    snap = mod.dump_thread_stacks()
    assert snap["n"] >= 1
    names = {t["name"] for t in snap["threads"]}
    assert "MainThread" in names
    joined = "".join(t["top"] for t in snap["threads"])
    assert "dump_thread_stacks" in joined or "test_dump_thread_stacks" in joined


def test_xml_plus_tool_calls_breaks_held_prefix_empty_content_does_not():
    """Live OpenHands echoes assistant content and tool_calls. CodeScout's jinja
    then emits <tool_call> twice, so held prompt+generation is not a prefix of
    the next apply_chat_template. Empty content + tool_calls restores the prefix.
    """
    jinja2 = pytest.importorskip("jinja2")
    tokenizers = pytest.importorskip("tokenizers")
    tok_dir = Path(__file__).resolve().parents[2] / "workloads" / "code_localization" / "tokenizer"
    tmpl_path = tok_dir / "chat_template.jinja"
    tok_path = tok_dir / "tokenizer.json"
    if not tmpl_path.is_file() or not tok_path.is_file():
        pytest.skip("CodeScout tokenizer blobs are not in the tree")

    env = jinja2.Environment(loader=jinja2.BaseLoader())
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    tmpl = env.from_string(tmpl_path.read_text())
    tokenizer = tokenizers.Tokenizer.from_file(str(tok_path))
    im_end = 151645

    def render(messages, tools, add_generation_prompt):
        return tmpl.render(
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt
        )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def lcp(left, right):
        n = min(len(left), len(right))
        i = 0
        while i < n and left[i] == right[i]:
            i += 1
        return i

    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "run",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    msgs0 = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Find the bug in django."},
    ]
    xml = '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>'
    prompt0 = encode(render(msgs0, tools, True))
    held = prompt0 + encode(xml) + [im_end]
    tool_msg = {"role": "tool", "content": "ok\nfile.py", "tool_call_id": "call_0"}
    call = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": "ls"})},
    }
    both = encode(
        render(
            msgs0 + [{"role": "assistant", "content": xml, "tool_calls": [call]}, tool_msg],
            tools,
            True,
        )
    )
    empty = encode(
        render(
            msgs0 + [{"role": "assistant", "content": "", "tool_calls": [call]}, tool_msg],
            tools,
            True,
        )
    )
    assert both[: len(held)] != held
    assert lcp(held, both) == len(held) - 1
    assert empty[: len(held)] == held
    suffix = empty[len(held) :]
    assert suffix
    assert "<tool_response>" in tokenizer.decode(suffix)
