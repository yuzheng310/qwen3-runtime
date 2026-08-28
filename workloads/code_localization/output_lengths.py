"""Empirical completion lengths from the frozen 494-task CodeScout corpus."""

from __future__ import annotations

import json
from pathlib import Path

RECONSTRUCTION = Path(__file__).resolve().parent / "token_reconstruction_v1.json"


def recorded_output_tokens(path: Path | None = None) -> list[int]:
    """Per-request `recorded_output_tokens` in dump order (2,395 requests)."""
    data = json.loads((path or RECONSTRUCTION).read_text())
    out: list[int] = []
    for task in data["tasks"]:
        for req in task["requests"]:
            out.append(int(req["recorded_output_tokens"]))
    if len(out) != 2395:
        raise ValueError(f"expected 2395 recorded output lengths, got {len(out)}")
    return out
