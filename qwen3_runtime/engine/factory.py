"""Build a 4B (or local checkpoint) engine. Does not download weights."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.memory import num_kv_blocks_for_budget
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.utils.loader import load_from_directory

PIN = Path(__file__).resolve().parents[2] / "docs" / "pins" / "Qwen3-4B" / "config.json"
PIN_REPO = "Qwen/Qwen3-4B"
PIN_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
CODESCOUT_PIN = Path(__file__).resolve().parents[2] / "docs" / "pins" / "CodeScout-4B" / "config.json"


def resolve_pin(*, pin: bool = True, pin_path: str | Path | None = None) -> Path | None:
    """Stage-1 default remains Qwen3-4B. CodeScout Replay passes pin_path."""
    if pin_path is not None:
        return Path(pin_path)
    return PIN if pin else None


def default_cuda_graph(*, device: str, backend: str) -> bool:
    """Decode CUDA Graph factory default: CUDA + FlashInfer or Triton decode."""
    return device == "cuda" and backend in ("flashinfer", "triton")


def decode_graph_max_kv_for(
    *, disable_split_kv: bool, max_num_batched_tokens: int
) -> int | None:
    """Chunk cap for decode graph replay, or None for no cap.

    EXP-011 capped replay at the chunk size because frozen-CTA graphs fall behind
    eager split-KV as KV grows. EXP-011R re-measured with split-KV left on and the
    graph led at every length and batch, so the cap belongs to the frozen-CTA path
    alone.
    """
    return max_num_batched_tokens if disable_split_kv else None


def select_attention_backend() -> str:
    if not torch.cuda.is_available():
        return "pytorch"
    try:
        import flashinfer  # noqa: F401

        return "flashinfer"
    except ImportError:
        return "pytorch"


def eos_token_id_from_pin(pin_path: Path | None) -> int | None:
    """HF config eos_token_id. Lists (Qwen generation_config style) take the first id."""
    if pin_path is None or not pin_path.exists():
        return None
    raw = json.loads(pin_path.read_text())
    eos = raw.get("eos_token_id")
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    return int(eos) if eos is not None else None


def _num_kv_blocks(model, *, device: str, block_size: int, kv_budget: int | None) -> int:
    dtype_bytes = 2 if device == "cuda" else 4
    requested = kv_budget_bytes(kv_budget)
    budget = requested
    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
        physical = int(free * 0.90)
        if physical < requested:
            budget = physical
    return max(32, num_kv_blocks_for_budget(model.cfg, budget, block_size, dtype_bytes=dtype_bytes))


def kv_budget_bytes(budget: int | None = None) -> int:
    """Bytes to hand the paged KV pool.

    Pass ``budget`` to pin the pool (two-arm comparisons). Otherwise CUDA takes
    85% of free VRAM at build time; CPU uses 256 MiB.
    """
    if budget is not None:
        if budget <= 0:
            raise ValueError(f"kv_budget must be positive, got {budget!r}")
        return budget
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        return int(free * 0.85)
    return 256 * 1024 * 1024


def build_engine(
    model_dir: str | Path,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    block_size: int = 16,
    split: bool | None = None,
    pin: bool = True,
    pin_path: str | Path | None = None,
    attention_backend: str | None = None,
    cuda_graph: bool | None = None,
    decode_graph_disable_split_kv: bool = False,
    kv_budget: int | None = None,
    enable_prefix_cache: bool = False,
    num_speculative_tokens: int = 0,
    ngram_min: int = 2,
    ngram_max: int = 4,
    session_cpu_offload: str = "off",
    cpu_kv_max_bytes: int = 0,
    cpu_kv_pinned_max_bytes: int = 0,
    transfer_chunk_bytes: int = 8 * 1024 * 1024,
) -> Engine:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    backend = attention_backend or select_attention_backend()
    if cuda_graph is None:
        cuda_graph = default_cuda_graph(device=device, backend=backend)
    model = load_from_directory(
        model_dir,
        device=device,
        dtype=dtype,
        pin=resolve_pin(pin=pin, pin_path=pin_path),
        attention_backend=backend,
    )
    num_blocks = _num_kv_blocks(model, device=device, block_size=block_size, kv_budget=kv_budget)
    if split is None:
        split = device == "cuda"
    graph_max_kv = decode_graph_max_kv_for(
        disable_split_kv=decode_graph_disable_split_kv,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    runner = PagedRunner(
        model,
        cuda_graph=cuda_graph,
        decode_graph_max_kv=graph_max_kv,
        decode_graph_disable_split_kv=decode_graph_disable_split_kv,
        split=split,
    )
    pin_file = resolve_pin(pin=pin, pin_path=pin_path)
    config = Config(
        block_size=block_size,
        num_kv_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        attention_backend=backend,
        cuda_graph=cuda_graph,
        enable_prefix_cache=enable_prefix_cache,
        eos_token_id=eos_token_id_from_pin(pin_file),
        num_speculative_tokens=num_speculative_tokens,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        session_cpu_offload=session_cpu_offload,
        cpu_kv_max_bytes=cpu_kv_max_bytes,
        cpu_kv_pinned_max_bytes=cpu_kv_pinned_max_bytes,
        transfer_chunk_bytes=transfer_chunk_bytes,
    )
    return Engine(config, runner)
