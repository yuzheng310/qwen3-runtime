#!/usr/bin/env python3
"""Minimal OpenAI /v1/chat/completions adapter for CodeScout rollouts.

Not a generic gateway. One process, one Engine, one mutex, one sequential
client session. OpenHands resends the full chat each turn.

Tokenizer and HTTP stay outside qwen3-runtime core.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from qwen3_runtime.engine.factory import CODESCOUT_PIN, build_engine
from qwen3_runtime.sampling import SamplingParams

IM_END = 151645
END_OF_TEXT = 151643
STOP_TOKEN_IDS = (IM_END, END_OF_TEXT)
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


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
    def __init__(self, engine, tokenizer, model_name: str):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.lock = threading.Lock()
        self.active_request_id: int | None = None
        self.stats = {
            "n_chat": 0,
            "tokenize_s": 0.0,
            "drain_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def snapshot_stats(self) -> dict:
        paused = len(self.engine.scheduler.paused)
        used = self.engine.block_manager.num_blocks - self.engine.block_manager.num_free_blocks
        return {
            **self.stats,
            "paused_sessions": paused,
            "kv_used_blocks": used,
            "kv_num_blocks": self.engine.block_manager.num_blocks,
        }

    def reset_sessions(self) -> int:
        ids = list(self.engine.scheduler.paused.keys())
        ids += [req.request_id for req in list(self.engine.scheduler.waiting)]
        ids += [req.request_id for req in list(self.engine.scheduler.running)]
        n = 0
        for rid in dict.fromkeys(ids):
            self.engine.finish_request(rid)
            n += 1
        self.active_request_id = None
        return n

    def _continue_or_add(self, token_ids: list[int], **kwargs) -> int:
        rid = self.active_request_id
        req = self.engine.scheduler.paused.get(rid) if rid is not None else None
        if req is not None:
            held = req.token_ids
            if 0 < len(held) < len(token_ids) and token_ids[: len(held)] == held:
                resume_kwargs = dict(kwargs)
                max_tokens = int(resume_kwargs.pop("max_tokens"))
                self.engine.resume_request(
                    rid,
                    token_ids[len(held) :],
                    max_tokens,
                    **resume_kwargs,
                )
                return rid
            self.engine.finish_request(rid)
            self.active_request_id = None
        rid = self.engine.add_request(token_ids, **kwargs)
        self.active_request_id = rid
        return rid

    def chat_completions(self, body: dict) -> dict:
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
        with self.lock:
            t_drain = time.perf_counter()
            rid = self._continue_or_add(
                token_ids,
                max_tokens=max_tokens,
                hold_kv=True,
                ignore_eos=False,
                sampling=sampling,
                stop_token_ids=STOP_TOKEN_IDS,
            )
            completion = self.engine.drain_request(rid)
            drain_s = time.perf_counter() - t_drain
            self.stats["n_chat"] += 1
            self.stats["tokenize_s"] += tokenize_s
            self.stats["drain_s"] += drain_s
            self.stats["prompt_tokens"] += len(token_ids)
            self.stats["completion_tokens"] += len(completion)
        content = decode_content(self.tokenizer, completion, include_stop_str=include_stop)
        created = int(time.time())
        tool_calls = parse_qwen_tool_calls(content)
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "token_ids": completion,
        }
        return {
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
            _write_json(self, 404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/")
            if path not in ("/v1/chat/completions", "/chat/completions", "/v1/reset", "/reset"):
                _write_json(self, 404, {"error": {"message": "not found"}})
                return
            if path in ("/v1/reset", "/reset"):
                with state.lock:
                    n = state.reset_sessions()
                _write_json(self, 200, {"released": n})
                return
            try:
                body = _json_body(self)
                payload = state.chat_completions(body)
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
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--pin-path", type=Path, default=CODESCOUT_PIN)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=16,
        help="N-gram speculative window for Agent rollout. 0 disables.",
    )
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
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
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )
    state = RolloutState(engine, tokenizer, args.served_model_name)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"codescout openai adapter http://{args.host}:{args.port}/v1/ model={args.served_model_name}")
    server.serve_forever()


if __name__ == "__main__":
    main()
