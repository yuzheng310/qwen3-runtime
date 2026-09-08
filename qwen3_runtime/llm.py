"""Public facade: ``LLM.generate`` and ``LLM.session`` over the production engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.factory import build_engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig
from qwen3_runtime.sampling_params import SamplingParams


class Session:
    """One held KV sequence. ``turn`` prefills only the new suffix after the first call."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.request_id: int | None = None
        self.prompt_ids: list[int] = []
        self.last_tokens: list[int] = []
        self.prompt_tokens = 0  # prompt tokens handed to this session, summed over turns
        self.reused_tokens = 0  # of those, the ones already in KV and never prefilled again

    def turn(self, messages, max_tokens: int = 16, sampling: SamplingParams | None = None) -> str:
        ids = self.llm.encode(messages)
        sampling = sampling or SamplingParams()
        if self.request_id is None:
            rid = self.llm.engine.add_request(
                ids, max_tokens=max_tokens, hold_kv=True, sampling=sampling, ignore_eos=True
            )
            self.request_id = rid
            self.last_tokens = self.llm.engine.drain_request(rid)
            self.prompt_tokens += len(ids)
            # A first turn holds no KV of its own, so a shared prefix block is
            # the only reuse available to it.
            req = self.llm.engine._requests.get(rid)
            self.reused_tokens += req.cached_tokens if req is not None else 0
        else:
            req = self.llm.engine._requests[self.request_id]
            held = list(req.token_ids)
            if len(ids) > len(held) and ids[: len(held)] == held:
                self.llm.engine.resume_request(
                    self.request_id, ids[len(held) :], max_tokens, hold_kv=True, sampling=sampling
                )
                self.last_tokens = self.llm.engine.drain_request(self.request_id)
                self.prompt_tokens += len(ids)
                self.reused_tokens += len(held)
            else:
                # Divergent history: the held KV is worthless, so this turn
                # restarts and counts as a full prefill.
                self.close()
                return self.turn(messages, max_tokens=max_tokens, sampling=sampling)
        return self.llm.decode(self.last_tokens)

    def report(self) -> dict[str, float | int]:
        """Prompt tokens this session was handed, and how many it did not prefill.

        Counts the KV the session holds across turns. It is not
        ``Request.cached_tokens``, which only the prefix cache writes and which
        stays zero on the resume path however much that path reuses.
        """
        total = self.prompt_tokens
        return {
            "prompt_tokens": total,
            "reused_tokens": self.reused_tokens,
            "reuse_pct": 100.0 * self.reused_tokens / total if total else 0.0,
        }

    def close(self) -> None:
        if self.request_id is not None:
            self.llm.engine.finish_request(self.request_id)
        self.request_id = None
        self.prompt_ids = []

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class LLM:
    """nano-vllm-shaped entry. Tiny in-memory models need no download."""

    def __init__(
        self,
        model: str | Path | Qwen3ForCausalLM | None = None,
        *,
        tokenizer: Any = None,
        config: Qwen3ModelConfig | None = None,
        max_num_seqs: int = 4,
        max_num_batched_tokens: int = 256,
        enable_prefix_cache: bool = False,
        **factory_kwargs,
    ):
        self.tokenizer = tokenizer
        if isinstance(model, Qwen3ForCausalLM):
            cfg = Config(
                block_size=4,
                num_kv_blocks=64,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_cache=enable_prefix_cache,
            )
            self.engine = Engine(cfg, PagedRunner(model))
        elif model is None:
            torch.manual_seed(0)
            model = Qwen3ForCausalLM(config or _tiny_config()).eval()
            cfg = Config(
                block_size=4,
                num_kv_blocks=64,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_cache=enable_prefix_cache,
            )
            self.engine = Engine(cfg, PagedRunner(model))
        else:
            self.engine = build_engine(
                model,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_cache=enable_prefix_cache,
                **factory_kwargs,
            )

    def encode(self, prompt) -> list[int]:
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return [int(x) for x in prompt]
        if self.tokenizer is None:
            if isinstance(prompt, str):
                return [min(31, (ord(ch) % 30) + 1) for ch in prompt] or [1]
            if isinstance(prompt, list):
                text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in prompt)
                return self.encode(text)
            raise TypeError("prompt must be token ids, a string, or chat messages")
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt)
        else:
            ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(x) for x in ids]

    def decode(self, tokens: Sequence[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(t) for t in tokens)
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    def generate(self, prompts, sampling: SamplingParams | None = None, max_tokens: int = 16):
        sampling = sampling or SamplingParams()
        if isinstance(prompts, list) and prompts and isinstance(prompts[0], list):
            return [self.generate(p, sampling=sampling, max_tokens=max_tokens) for p in prompts]
        if isinstance(prompts, list) and prompts and isinstance(prompts[0], int):
            return self.engine.generate(prompts, max_tokens=max_tokens, sampling=sampling)
        ids = self.encode(prompts)
        text = self.decode(self.engine.generate(ids, max_tokens=max_tokens, sampling=sampling))
        return text

    def session(self) -> Session:
        return Session(self)


def _tiny_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
