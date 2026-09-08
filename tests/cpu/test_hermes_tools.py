"""Hermes XML → OpenAI tool_calls, used by the SkyRL chat_completion path."""

from qwen3_runtime.rollout.tool_parser import assistant_chat_message, parse_qwen_tool_calls


def test_parse_qwen_tool_calls_hermes_xml():
    text = (
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>'
    )
    calls = parse_qwen_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"


def test_assistant_message_drops_xml_when_only_tool_calls():
    xml = '<tool_call>{"name": "terminal", "arguments": {"command": "pwd"}}</tool_call>'
    calls = parse_qwen_tool_calls(xml)
    msg = assistant_chat_message(xml, calls)
    assert msg["content"] == ""
    assert msg["tool_calls"] == calls


def test_trailing_stop_marker_does_not_keep_the_xml():
    """Decoding keeps `<|im_end|>`, which used to make the leftover look non-empty.

    On a real GRPO step that put 1.91x as many `<tool_call>` markers into the
    training sequence as the policy emitted.
    """
    xml = (
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>'
        "<|im_end|>"
    )
    msg = assistant_chat_message(xml, parse_qwen_tool_calls(xml))
    assert msg["content"] == ""
    assert "<tool_call>" not in msg["content"]


def test_prose_before_a_call_is_kept_but_the_call_is_not_rendered_twice():
    mixed = (
        "Let me look at the parser.\n"
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>'
        "<|im_end|>"
    )
    msg = assistant_chat_message(mixed, parse_qwen_tool_calls(mixed))
    assert msg["content"] == "Let me look at the parser."
    assert "<tool_call>" not in msg["content"]
    assert len(msg["tool_calls"]) == 1


def test_several_calls_in_one_completion_all_leave_content():
    text = "".join(
        f'<tool_call>\n{{"name": "terminal", "arguments": {{"command": "c{i}"}}}}\n</tool_call>'
        for i in range(4)
    ) + "<|im_end|>"
    msg = assistant_chat_message(text, parse_qwen_tool_calls(text))
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 4


def test_content_is_untouched_when_there_are_no_tool_calls():
    msg = assistant_chat_message("plain answer", [])
    assert msg["content"] == "plain answer"
    assert "tool_calls" not in msg


def test_flatten_openai_content_parts_for_qwen_template():
    """CodeScout chat_template.jinja line 4 does `messages[0].content + '\\n\\n'`."""
    from qwen3_runtime.integrations.skyrl.inference_engine import (
        _flatten_message_content,
        _normalize_chat_messages,
        _normalize_tools,
    )

    assert _flatten_message_content("hello") == "hello"
    assert _flatten_message_content([{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]) == "hello world"
    msgs = _normalize_chat_messages(
        [{"role": "system", "content": [{"type": "text", "text": "You are a helper."}]}]
    )
    assert msgs[0]["content"] == "You are a helper."
    assert msgs[0]["content"] + "\n\n" == "You are a helper.\n\n"
    assert _normalize_tools(None) is None
    assert _normalize_tools([{"type": "function", "function": {"name": "terminal"}}])


def test_coerce_token_ids_from_hf_batch_encoding():
    from qwen3_runtime.integrations.skyrl.inference_engine import (
        _apply_chat_template_ids,
        _coerce_token_ids,
    )

    class Tok:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(c) for c in text]

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return {"input_ids": [10, 11, 12]}

    assert _coerce_token_ids({"input_ids": [1, 2, 3]}, Tok()) == [1, 2, 3]
    assert _apply_chat_template_ids(Tok(), [{"role": "user", "content": "x"}]) == [10, 11, 12]


def test_openai_logprob_content_token_is_a_string():
    """LiteLLM rejects token=None; attempt 9 64/64 trajs died on that."""
    from qwen3_runtime.integrations.skyrl.inference_engine import _openai_logprob_content

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            del skip_special_tokens
            return {151657: "<tool_call>", 198: "\n"}.get(ids[0], "?")

    rows = _openai_logprob_content(Tok(), [151657, 198], [-0.1, 0.0])
    assert [r["token"] for r in rows] == ["<tool_call>", "\n"]
    assert all(isinstance(r["token"], str) for r in rows)
    assert rows[0]["token_id"] == 151657
    assert rows[0]["bytes"] == list("<tool_call>".encode("utf-8"))


def test_chat_completion_shape_has_openhands_token_id_fields():
    """CodeScout sets litellm_extra_body.return_token_ids; OpenHands reads these keys."""
    prompt_ids = [1, 2, 3]
    completion = [4, 5]
    payload = {
        "prompt_token_ids": prompt_ids,
        "choices": [
            {
                "token_ids": completion,
                "provider_specific_fields": {"token_ids": completion},
            }
        ],
    }
    assert payload["prompt_token_ids"] == prompt_ids
    assert payload["choices"][0]["provider_specific_fields"]["token_ids"] == completion


def test_chat_completion_max_tokens_prefers_max_completion_tokens():
    """OpenAI/LiteLLM send max_completion_tokens; _sampling_from_dict defaults to 16."""
    from qwen3_runtime.integrations.skyrl.inference_engine import _chat_max_tokens, _sampling_from_dict

    _, default_max = _sampling_from_dict({})
    assert default_max == 16
    assert _chat_max_tokens({}) == 2048
    assert _chat_max_tokens({"max_completion_tokens": 8192}) == 8192
    assert _chat_max_tokens({"max_tokens": 128}) == 128
    assert _chat_max_tokens({"max_completion_tokens": 8192, "max_tokens": 16}) == 8192


def test_coerce_token_ids_rejects_dict_keys_as_ids():
    """list(BatchEncoding) is ['input_ids', ...]; that must not reach torch.tensor."""
    from qwen3_runtime.integrations.skyrl.inference_engine import _coerce_token_ids

    class Tok:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [1]

    try:
        _coerce_token_ids(["input_ids", "attention_mask"], Tok())
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
