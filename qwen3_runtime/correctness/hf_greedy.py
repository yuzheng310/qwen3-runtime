"""Greedy token-id comparison. HuggingFace generate() defaults are NOT used."""

from collections.abc import Sequence

import torch

from qwen3_runtime.engine.engine import Engine


@torch.no_grad()
def hf_greedy_tokens(model, prompt: Sequence[int], max_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_tokens):
        logits = model(ids, use_cache=False).logits[0, -1]
        tok = int(logits.argmax(dim=-1).item())
        out.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
    return out


def runtime_greedy_tokens(engine: Engine, prompt: Sequence[int], max_tokens: int) -> list[int]:
    return engine.generate(list(prompt), max_tokens=max_tokens)
