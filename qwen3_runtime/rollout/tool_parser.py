"""Hermes/Qwen XML tool-call mapping for OpenAI chat.completion responses."""

from __future__ import annotations

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# The engine decodes with skip_special_tokens=False so the logprob sidecar's
# token ids line up with the text it reports. That leaves chat frame markers in
# the completion, and they are the template's to emit, never the assistant's.
CHAT_FRAME_MARKERS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def _strip_frame_markers(text: str) -> str:
    for marker in CHAT_FRAME_MARKERS:
        text = text.replace(marker, "")
    return text


def assistant_chat_message(content: str, tool_calls: list[dict]) -> dict:
    """OpenAI assistant message for OpenHands.

    Hermes XML has to come out of ``content`` once the calls are parsed, or the
    next chat-template pass renders each call twice: once from the leftover
    ``content`` and once from ``tool_calls``. Measured on a real GRPO step, the
    duplicate rendering put 1.91x as many ``<tool_call>`` markers into the
    training sequence as the policy emitted, so roughly half of the positions
    the loss mask selected were tokens the policy never produced.

    Keeping only the non-XML remainder matters as much as the check that used to
    guard it. A completion that ends ``</tool_call><|im_end|>`` leaves a
    non-empty remainder, and prose before a call leaves a real one, and in both
    cases the old guard fell through and kept the XML.
    """
    message = {"role": "assistant", "content": content}
    if not tool_calls:
        return message
    message["tool_calls"] = tool_calls
    message["content"] = _strip_frame_markers(TOOL_CALL_RE.sub("", content or "")).strip()
    return message


def parse_qwen_tool_calls(text: str) -> list[dict]:
    """Map Qwen XML tool calls to OpenAI chat.completion tool_calls."""
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
