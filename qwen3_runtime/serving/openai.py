"""OpenAI-shaped HTTP API. Session KV is ``/v1/s/{key}/chat/completions``."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from qwen3_runtime.engine.request import RequestStatus
from qwen3_runtime.sampling_params import SamplingParams

SESSION_CHAT_RE = re.compile(r"^/v1/s/([^/]+)/chat/completions$")
SESSION_RESET_RE = re.compile(r"^/v1/s/([^/]+)/reset$")


def parse_path(path: str) -> tuple[str, str | None]:
    path = urlparse(path).path.rstrip("/") or "/"
    match = SESSION_CHAT_RE.fullmatch(path)
    if match:
        return "chat", match.group(1)
    match = SESSION_RESET_RE.fullmatch(path)
    if match:
        return "reset_one", match.group(1)
    if path in ("/v1/chat/completions", "/chat/completions"):
        return "chat", "default"
    if path in ("/v1/completions", "/completions"):
        return "complete", None
    if path in ("/v1/models", "/models"):
        return "models", None
    if path in ("/v1/reset", "/reset"):
        return "reset_all", None
    return "other", None


def plan_session_continuation(held: list[int], new_ids: list[int]) -> tuple[str, list[int]]:
    if held and new_ids and len(new_ids) > len(held) and new_ids[: len(held)] == held:
        return "exact", new_ids[len(held) :]
    return "replace", []


def sampling_from_body(body: dict) -> SamplingParams:
    return SamplingParams(
        temperature=float(body.get("temperature") or 0.0),
        top_p=float(body.get("top_p") or 1.0),
        top_k=int(body.get("top_k") or 0),
        min_p=float(body.get("min_p") or 0.0),
        seed=body.get("seed"),
        stop_strings=tuple(body.get("stop") or ()),
        logprobs=bool(body.get("logprobs")),
        top_logprobs=int(body.get("top_logprobs") or 0),
    )


class OpenAIHandler(BaseHTTPRequestHandler):
    server: "OpenAIServer"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        action, _key = parse_path(self.path)
        if action == "models":
            self._json(200, {"object": "list", "data": [{"id": self.server.model_id, "object": "model"}]})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        action, key = parse_path(self.path)
        body = self._read_json()
        if action == "reset_one" and key is not None:
            self.server.finish_session(key)
            self._json(200, {"ok": True})
            return
        if action == "reset_all":
            self.server.finish_all()
            self._json(200, {"ok": True})
            return
        if action == "chat":
            self._chat(body, key or "default")
            return
        if action == "complete":
            self._complete(body)
            return
        self._json(404, {"error": {"message": "not found"}})

    def _chat(self, body: dict, key: str) -> None:
        messages = body.get("messages") or []
        ids = self.server.tokenize_chat(messages, body.get("tools"))
        max_tokens = int(body.get("max_tokens") or 16)
        sampling = sampling_from_body(body)
        stream = bool(body.get("stream"))
        tokens = self.server.run_session(key, ids, max_tokens, sampling)
        text = self.server.decode(tokens)
        payload = _completion_payload(self.server.model_id, text, stream=False)
        if stream:
            self._sse(payload)
            return
        self._json(200, payload)

    def _complete(self, body: dict) -> None:
        prompt = body.get("prompt") or ""
        ids = self.server.tokenize_text(prompt)
        max_tokens = int(body.get("max_tokens") or 16)
        sampling = sampling_from_body(body)
        tokens = self.server.run_prompt(ids, max_tokens, sampling)
        text = self.server.decode(tokens)
        self._json(200, _completion_payload(self.server.model_id, text, chat=False))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, payload: dict) -> None:
        chunk = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": payload["choices"][0]["message"]["content"]}, "finish_reason": None}],
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _completion_payload(model: str, text: str, *, chat: bool = True, stream: bool = False) -> dict:
    choice = {"index": 0, "finish_reason": "stop"}
    if chat:
        choice["message"] = {"role": "assistant", "content": text}
        obj = "chat.completion"
    else:
        choice["text"] = text
        obj = "text_completion"
    return {"id": "cmpl-qwen3", "object": obj, "model": model, "choices": [choice]}


class OpenAIServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], engine: Any, tokenizer: Any, *, model_id: str = "qwen3"):
        super().__init__(addr, OpenAIHandler)
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.sessions: dict[str, int] = {}

    def tokenize_chat(self, messages: list, tools) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True
        )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids, list) and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def tokenize_text(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(x) for x in ids]

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def run_prompt(self, ids: list[int], max_tokens: int, sampling: SamplingParams) -> list[int]:
        return self.engine.generate(ids, max_tokens=max_tokens, ignore_eos=False, sampling=sampling)

    def run_session(self, key: str, ids: list[int], max_tokens: int, sampling: SamplingParams) -> list[int]:
        rid = self.sessions.get(key)
        req = self.engine._requests.get(rid) if rid is not None else None
        if req is None or req.status != RequestStatus.PAUSED:
            rid = self.engine.add_request(
                ids, max_tokens=max_tokens, hold_kv=True, sampling=sampling, ignore_eos=False,
                stop_strings=sampling.stop_strings, tokenizer=self.tokenizer,
            )
            self.sessions[key] = rid
            return self.engine.drain_request(rid)
        kind, suffix = plan_session_continuation(list(req.token_ids), ids)
        if kind != "exact":
            self.engine.finish_request(rid)
            rid = self.engine.add_request(
                ids, max_tokens=max_tokens, hold_kv=True, sampling=sampling, ignore_eos=False,
                stop_strings=sampling.stop_strings, tokenizer=self.tokenizer,
            )
            self.sessions[key] = rid
            return self.engine.drain_request(rid)
        self.engine.resume_request(rid, suffix, max_tokens, hold_kv=True, sampling=sampling)
        return self.engine.drain_request(rid)

    def finish_session(self, key: str) -> None:
        rid = self.sessions.pop(key, None)
        if rid is not None:
            self.engine.finish_request(rid)

    def finish_all(self) -> None:
        for key in list(self.sessions):
            self.finish_session(key)


def serve(engine: Any, tokenizer: Any, *, host: str = "127.0.0.1", port: int = 8000, model_id: str = "qwen3"):
    server = OpenAIServer((host, port), engine, tokenizer, model_id=model_id)
    server.serve_forever()


def main() -> None:
    import argparse

    from qwen3_runtime.engine.factory import build_engine

    parser = argparse.ArgumentParser(description="OpenAI-shaped server with session routes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    engine = build_engine(args.model, max_num_seqs=8, max_num_batched_tokens=2048)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    serve(engine, tokenizer, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
