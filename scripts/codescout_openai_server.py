#!/usr/bin/env python3
"""OpenAI-shaped /v1/chat/completions adapter for CodeScout rollouts.

One process, one Engine, one engine thread. HTTP is ThreadingHTTPServer.
Sessions are keyed by URL path: ``POST /v1/s/{key}/chat/completions``.
``/v1/chat/completions`` is session ``default`` so existing clients stay valid.
Prefix matching is within a session only — GRPO group members send byte-identical
turn-0 prompts and must not share a request id. If the re-tokenized prompt is not
an exact extension of the paused ``token_ids``, finish and ``add_request``.

Tokenizer and HTTP stay outside qwen3-runtime core.
"""

from __future__ import annotations

import argparse
import cProfile
import ipaddress
import json
import os
import pstats
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from qwen3_runtime.engine.factory import CODESCOUT_PIN, build_engine
from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.sampling import SamplingParams

SESSION_CHAT_RE = re.compile(r"^/v1/s/([^/]+)/chat/completions$")
SESSION_RESET_RE = re.compile(r"^/v1/s/([^/]+)/reset$")


def dump_thread_stacks() -> dict:
    """In-process substitute for py-spy when the host has no CAP_SYS_PTRACE."""
    frames = sys._current_frames()
    by_ident = {t.ident: t for t in threading.enumerate()}
    threads = []
    for ident, fr in frames.items():
        t = by_ident.get(ident)
        name = t.name if t is not None else str(ident)
        stack = traceback.format_stack(fr)
        threads.append(
            {
                "name": name,
                "ident": ident,
                "stack": stack,
                "top": stack[-1].strip() if stack else "",
            }
        )
    return {"n": len(threads), "threads": threads}


def parse_adapter_path(path: str) -> tuple[str, str | None]:
    """Map URL path to (action, session_key).

    action is ``chat``, ``reset_one``, ``reset_all``, or ``other``.
    ``/v1/chat/completions`` uses session key ``default``.
    """
    path = urlparse(path).path.rstrip("/") or "/"
    match = SESSION_CHAT_RE.fullmatch(path)
    if match:
        return "chat", match.group(1)
    match = SESSION_RESET_RE.fullmatch(path)
    if match:
        return "reset_one", match.group(1)
    if path in ("/v1/chat/completions", "/chat/completions"):
        return "chat", "default"
    if path in ("/v1/reset", "/reset"):
        return "reset_all", None
    return "other", None


class _Waiter:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.completion: list[int] = []
        self.error: BaseException | None = None
        self.request_id: int | None = None

IM_END = 151645
END_OF_TEXT = 151643
STOP_TOKEN_IDS = (IM_END, END_OF_TEXT)
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def plan_session_continuation(
    held: list[int], new_ids: list[int]
) -> tuple[str, list[int]]:
    """Map a re-tokenized chat onto a paused session.

    Returns ``(kind, suffix)``:
    - ``exact``: ``held`` is a strict prefix of ``new_ids``.
    - ``replace``: finish and ``add_request`` (not an exact extension).
    """
    if held and new_ids and len(new_ids) > len(held) and new_ids[: len(held)] == held:
        return "exact", new_ids[len(held) :]
    return "replace", []


def assistant_chat_message(content: str, tool_calls: list[dict]) -> dict:
    """OpenAI assistant message for OpenHands.

    CodeScout's chat template renders ``content`` *and* ``tool_calls``. Echoing
    the Hermes XML in ``content`` while also populating ``tool_calls`` duplicates
    ``<tool_call>`` on the next turn, so held ``token_ids`` stop being a prefix
    of the re-tokenized prompt. If content is only those XML blocks, drop it.
    """
    message = {"role": "assistant", "content": content}
    if not tool_calls:
        return message
    message["tool_calls"] = tool_calls
    leftover = TOOL_CALL_RE.sub("", content or "").strip()
    if not leftover:
        message["content"] = ""
    return message


def parse_qwen_tool_calls(text: str) -> list[dict]:
    """Map Qwen XML tool calls to OpenAI chat.completion tool_calls.

    CodeScout-4B emits Hermes/Qwen ``<tool_call>`` in the decoded text. OpenHands
    ``native_tool_calling=True`` (the SDK default for ``openai/`` models) will
    not execute Terminal unless this field is populated, same as vLLM's
    tool-call parser.
    """
    calls = []
    for i, raw in enumerate(TOOL_CALL_RE.findall(text or "")):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not obj.get("name"):
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": str(obj["name"]), "arguments": args},
            }
        )
    return calls


def _json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def _write_json(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def normalize_messages(messages: list) -> list[dict]:
    """HF Qwen templates concatenate content as a string; OpenHands may send parts."""
    out = []
    for message in messages:
        item = dict(message)
        item["content"] = _content_to_str(item.get("content"))
        if "tool_calls" in item and item["tool_calls"] is None:
            item.pop("tool_calls")
        out.append(item)
    return out


def normalize_tools(tools):
    if not tools:
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
    return tools


def tokenize_chat(tokenizer, messages: list, tools, extra: dict) -> list[int]:
    kwargs = dict(extra.get("chat_template_kwargs") or {})
    add_generation_prompt = bool(kwargs.pop("add_generation_prompt", True))
    kwargs.setdefault("enable_thinking", False)
    messages = normalize_messages(messages)
    tools = normalize_tools(tools)
    ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **kwargs,
    )
    if hasattr(ids, "get") and not isinstance(ids, (list, tuple, str)):
        extracted = ids.get("input_ids")
        if extracted is not None:
            ids = extracted
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids, list) and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def decode_content(tokenizer, token_ids: list[int], *, include_stop_str: bool) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if include_stop_str:
        return text
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop)[0]
    return text


class RolloutState:
    def __init__(
        self,
        engine,
        tokenizer,
        model_name: str,
        *,
        hold_kv: bool = True,
        completion_log: Path | None = None,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.hold_kv = hold_kv
        self.completion_log = Path(completion_log) if completion_log else None
        self._log_lock = threading.Lock()
        self.lock = threading.Lock()
        self._cond = threading.Condition(self.lock)
        self.sessions: dict[str, int] = {}
        self._waiters: dict[int, _Waiter] = {}
        self._pending: list = []
        self._loop_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = {
            "n_chat": 0,
            "tokenize_s": 0.0,
            "drain_s": 0.0,
            "engine_step_s": 0.0,
            "n_steps": 0,
            "decode_reqs_sum": 0,
            "n_decode_steps": 0,
            "preempts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "n_resume_exact": 0,
            "n_full_add": 0,
            "appended_prompt_tokens": 0,
        }

    @property
    def active_request_id(self) -> int | None:
        return self.sessions.get("default")

    @active_request_id.setter
    def active_request_id(self, rid: int | None) -> None:
        if rid is None:
            self.sessions.pop("default", None)
        else:
            self.sessions["default"] = rid

    def snapshot_stats(self) -> dict:
        paused = len(self.engine.scheduler.paused)
        used = self.engine.block_manager.num_blocks - self.engine.block_manager.num_free_blocks
        n_decode = int(self.stats["n_decode_steps"] or 0)
        return {
            **self.stats,
            "paused_sessions": paused,
            "n_waiting": len(self.engine.scheduler.waiting),
            "n_running": len(self.engine.scheduler.running),
            "kv_used_blocks": used,
            "kv_num_blocks": self.engine.block_manager.num_blocks,
            "mean_decode_batch": (self.stats["decode_reqs_sum"] / n_decode) if n_decode else 0.0,
            "last_step_stats": getattr(self.engine, "last_step_stats", None),
        }

    def _on_engine_thread(self) -> bool:
        return self._loop_thread is None or threading.current_thread() is self._loop_thread

    def _run_on_engine(self, fn):
        """Scheduler mutations run on the engine thread, never concurrent with step()."""
        if self._on_engine_thread():
            return fn()
        result: dict = {}
        done = threading.Event()

        def wrapped() -> None:
            try:
                result["v"] = fn()
            except Exception as exc:
                result["e"] = exc
            finally:
                done.set()

        with self._cond:
            self._pending.append(wrapped)
            self._cond.notify()
        if not done.wait(timeout=600):
            raise TimeoutError("engine thread stalled on submitted op")
        if "e" in result:
            raise result["e"]
        return result.get("v")

    def _reset_sessions_unlocked(self) -> int:
        ids = list(self.engine.scheduler.paused.keys())
        ids += [req.request_id for req in list(self.engine.scheduler.waiting)]
        ids += [req.request_id for req in list(self.engine.scheduler.running)]
        n = 0
        for rid in dict.fromkeys(ids):
            self.engine.finish_request(rid)
            waiter = self._waiters.pop(rid, None)
            if waiter is not None:
                waiter.event.set()
            n += 1
        self.sessions.clear()
        return n

    def _reset_session_unlocked(self, session_key: str) -> int:
        rid = self.sessions.pop(session_key, None)
        if rid is None:
            return 0
        self.engine.finish_request(rid)
        waiter = self._waiters.pop(rid, None)
        if waiter is not None:
            waiter.event.set()
        return 1

    def reset_sessions(self) -> int:
        return self._run_on_engine(self._reset_sessions_unlocked)

    def reset_session(self, session_key: str) -> int:
        return self._run_on_engine(lambda: self._reset_session_unlocked(session_key))

    def append_completion_log(self, record: dict) -> None:
        if self.completion_log is None:
            return
        self.completion_log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with self._log_lock:
            with open(self.completion_log, "a", encoding="utf-8") as handle:
                handle.write(line)

    def _continue_or_add(self, token_ids: list[int], *, session_key: str = "default", **kwargs) -> int:
        rid = self.sessions.get(session_key)
        req = self.engine.scheduler.paused.get(rid) if rid is not None else None
        if req is not None:
            held = req.token_ids
            kind, suffix = plan_session_continuation(held, token_ids)
            if kind == "exact":
                resume_kwargs = dict(kwargs)
                max_tokens = int(resume_kwargs.pop("max_tokens"))
                self.engine.resume_request(rid, suffix, max_tokens, **resume_kwargs)
                self.stats["n_resume_exact"] += 1
                self.stats["appended_prompt_tokens"] += len(suffix)
                return rid
            self.engine.finish_request(rid)
            self.sessions.pop(session_key, None)
        rid = self.engine.add_request(token_ids, **kwargs)
        self.sessions[session_key] = rid
        self.stats["n_full_add"] += 1
        self.stats["appended_prompt_tokens"] += len(token_ids)
        return rid

    def _ensure_engine_loop(self) -> None:
        start = None
        with self.lock:
            if self._loop_thread is not None:
                return
            start = threading.Thread(
                target=self._engine_loop,
                name="codescout-engine-loop",
                daemon=True,
            )
            self._loop_thread = start
        start.start()

    def _write_cprofile(self, prof: cProfile.Profile, dump: str) -> None:
        prof.dump_stats(dump)
        txt = dump + ".txt"
        with open(txt, "w") as f:
            stats = pstats.Stats(prof, stream=f)
            stats.sort_stats("cumtime")
            stats.print_stats(80)
        print(f"cprofile_dumped {dump}", flush=True)

    def _engine_loop(self) -> None:
        dump = os.environ.get("CODESCOUT_CPROFILE")
        prof = cProfile.Profile() if dump else None
        t_end = time.time() + float(os.environ.get("CODESCOUT_CPROFILE_S", "45")) if dump else None
        dumped = False
        if prof is not None:
            prof.enable()
        while not self._stop.is_set():
            if prof is not None and dump and not dumped and t_end is not None and time.time() >= t_end:
                prof.disable()
                self._write_cprofile(prof, dump)
                dumped = True
                prof = None
            with self._cond:
                while self._pending:
                    self._pending.pop(0)()
                sched = self.engine.scheduler
                has_work = bool(sched.waiting or sched.running)
                if not has_work:
                    self._cond.wait(timeout=0.05)
                    continue
            t0 = time.perf_counter()
            try:
                self.engine.step()
            except Exception as exc:
                with self.lock:
                    waiters = list(self._waiters.values())
                    self._waiters.clear()
                for waiter in waiters:
                    waiter.error = exc
                    waiter.event.set()
                continue
            step_s = time.perf_counter() - t0
            emitted = dict(self.engine.last_emitted)
            stats = self.engine.last_step_stats or {}
            with self.lock:
                self.stats["engine_step_s"] += step_s
                self.stats["n_steps"] += 1
                self.stats["preempts"] += int(stats.get("preempts") or 0)
                decode_reqs = int(stats.get("decode_reqs") or 0)
                if decode_reqs:
                    self.stats["decode_reqs_sum"] += decode_reqs
                    self.stats["n_decode_steps"] += 1
                done: list[int] = []
                for rid, waiter in self._waiters.items():
                    waiter.completion.extend(emitted.get(rid, []))
                    req = self.engine._requests.get(rid)
                    if req is None or req.status in (RequestStatus.PAUSED, RequestStatus.FINISHED):
                        waiter.event.set()
                        done.append(rid)
                for rid in done:
                    self._waiters.pop(rid, None)

    def chat_completions(self, body: dict, session_key: str = "default") -> dict:
        extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
        # LiteLLM may flatten extra_body keys onto the request.
        extra = {**extra, **{k: body[k] for k in body if k in (
            "return_token_ids",
            "include_stop_str_in_output",
            "chat_template_kwargs",
        )}}
        messages = body["messages"]
        tools = body.get("tools")
        t_tok = time.perf_counter()
        token_ids = tokenize_chat(self.tokenizer, messages, tools, extra)
        tokenize_s = time.perf_counter() - t_tok
        temperature = body.get("temperature")
        if temperature is None:
            temperature = 0.6
        top_p = body.get("top_p")
        if top_p is None:
            top_p = 1.0
        top_k = int(body.get("top_k") or 0)
        seed = body.get("seed")
        raw_max_tokens = body.get("max_completion_tokens")
        if raw_max_tokens is None:
            raw_max_tokens = body.get("max_tokens")
        max_tokens = 2048 if raw_max_tokens is None else int(raw_max_tokens)
        include_stop = bool(extra.get("include_stop_str_in_output", False))
        sampling = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
            seed=int(seed) if seed is not None else None,
        )
        self._ensure_engine_loop()
        waiter = _Waiter()
        kwargs = dict(
            max_tokens=max_tokens,
            hold_kv=self.hold_kv,
            ignore_eos=False,
            sampling=sampling,
            stop_token_ids=STOP_TOKEN_IDS,
        )

        def submit() -> None:
            try:
                rid = self._continue_or_add(token_ids, session_key=session_key, **kwargs)
                waiter.request_id = rid
                self._waiters[rid] = waiter
            except Exception as exc:
                waiter.error = exc
                waiter.event.set()

        t_drain = time.perf_counter()
        with self._cond:
            self._pending.append(submit)
            self._cond.notify()
        if not waiter.event.wait(timeout=600):
            raise TimeoutError(f"session {session_key} timed out")
        if waiter.error is not None:
            raise waiter.error
        drain_s = time.perf_counter() - t_drain
        completion = waiter.completion
        rid = waiter.request_id
        with self.lock:
            self.stats["n_chat"] += 1
            self.stats["tokenize_s"] += tokenize_s
            self.stats["drain_s"] += drain_s
            self.stats["prompt_tokens"] += len(token_ids)
            self.stats["completion_tokens"] += len(completion)
        content = decode_content(self.tokenizer, completion, include_stop_str=include_stop)
        created = int(time.time())
        tool_calls = parse_qwen_tool_calls(content)
        message = assistant_chat_message(content, tool_calls)
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "token_ids": completion,
        }
        payload = {
            "id": f"chatcmpl-{rid}-{created}",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or self.model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(token_ids),
                "completion_tokens": len(completion),
                "total_tokens": len(token_ids) + len(completion),
            },
            "prompt_token_ids": token_ids,
        }
        self.append_completion_log(
            {
                "kind": "chat",
                "session_key": session_key,
                "temperature": float(temperature),
                "hold_kv": self.hold_kv,
                "prompt_n": len(token_ids),
                "prompt_token_ids": token_ids,
                "completion_token_ids": list(completion),
                "token_source": "engine",
                "n_messages": len(messages) if isinstance(messages, list) else None,
                "content_head": (content or "")[:200],
                "usage": payload["usage"],
            }
        )
        return payload


def make_handler(state: RolloutState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/")
            if path in ("/v1/models", "/models"):
                _write_json(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": [{"id": state.model_name, "object": "model", "owned_by": "qwen3-runtime"}],
                    },
                )
                return
            if path in ("/stats", "/v1/stats"):
                _write_json(self, 200, state.snapshot_stats())
                return
            if path in ("/debug/stacks", "/v1/debug/stacks"):
                _write_json(self, 200, dump_thread_stacks())
                return
            _write_json(self, 404, {"error": {"message": "not found"}})

        def do_POST(self):
            action, session_key = parse_adapter_path(self.path)
            if action == "other":
                _write_json(self, 404, {"error": {"message": "not found"}})
                return
            if action == "reset_all":
                n = state.reset_sessions()
                state.append_completion_log({"kind": "reset", "scope": "all", "released": n})
                _write_json(self, 200, {"released": n})
                return
            if action == "reset_one":
                n = state.reset_session(session_key)
                state.append_completion_log(
                    {"kind": "reset", "scope": "one", "session": session_key, "released": n}
                )
                _write_json(self, 200, {"released": n, "session": session_key})
                return
            try:
                body = _json_body(self)
                payload = state.chat_completions(body, session_key=session_key or "default")
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                _write_json(self, 500, {"error": {"message": str(exc), "type": type(exc).__name__}})
                return
            _write_json(self, 200, payload)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow a non-loopback bind. This adapter has no authentication.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repository Python code. Not required by CodeScout-4B.",
    )
    parser.add_argument("--served-model-name", default="CodeScout-4B")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Active decode batch cap. Sweep N must be <= this. Binding limit is KV blocks.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--pin-path", type=Path, default=CODESCOUT_PIN)
    parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help="APC for GRPO group prefixes (Phase 1 C3 arm). Default off.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=16,
        help="N-gram speculative window for Agent rollout. 0 disables.",
    )
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument(
        "--hold-kv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Park session KV across turns. --no-hold-kv forces a full add every turn (arm A3).",
    )
    parser.add_argument(
        "--completion-log",
        type=Path,
        default=None,
        help="JSONL of prompt/completion token ids per chat (and resets). Required for §5.0.",
    )
    return parser


def validate_bind(args: argparse.Namespace) -> None:
    is_loopback = args.host == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback and not args.allow_remote:
        raise SystemExit("non-loopback bind requires --allow-remote; adapter has no authentication")


def main() -> None:
    args = build_parser().parse_args()
    validate_bind(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    engine = build_engine(
        args.model,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        pin_path=args.pin_path,
        enable_prefix_cache=args.enable_prefix_cache,
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )
    state = RolloutState(
        engine,
        tokenizer,
        args.served_model_name,
        hold_kv=bool(args.hold_kv),
        completion_log=args.completion_log,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"codescout openai adapter http://{args.host}:{args.port}/v1/ "
        f"model={args.served_model_name} hold_kv={args.hold_kv} spec={args.num_speculative_tokens}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
