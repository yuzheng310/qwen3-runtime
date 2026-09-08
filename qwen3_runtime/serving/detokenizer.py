"""Incremental detokenize for stop-string matching and SSE streaming."""

from __future__ import annotations

from typing import Any


class IncrementalDetokenizer:
    """Decode as tokens arrive. Stop strings match generated text only.

    Incomplete UTF-8 / tokenizer prefix spaces: if a new decode does not
    extend the previous string, replace rather than append a stale prefix.
    """

    def __init__(self, tokenizer: Any, prompt_ids: list[int] | None = None):
        self.tokenizer = tokenizer
        self.ids = list(prompt_ids or [])
        self.text = self._decode(self.ids)
        self.prompt_text_len = len(self.text)

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def feed(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        new = self._decode(self.ids)
        if new.startswith(self.text):
            delta = new[len(self.text) :]
            self.text = new
            return delta
        delta = new[self.prompt_text_len :] if len(new) >= self.prompt_text_len else new
        self.text = new
        return delta

    def generated_text(self) -> str:
        return self.text[self.prompt_text_len :]


def attach_detokenizer(tokenizer, token_ids: list[int], stop_strings: tuple[str, ...], sampling_stop: tuple[str, ...] = ()) -> tuple:
    strings = tuple(stop_strings or sampling_stop)
    if strings and tokenizer is not None:
        return IncrementalDetokenizer(tokenizer, prompt_ids=token_ids), strings
    return None, strings
