"""Build a 4B (or local checkpoint) engine. Does not download weights."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.memory import num_kv_blocks_for_budget
from qwen3_runtime.engine.model_runner import PagedRunner, SplitPagedRunner
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


def select_attention_backend() -> str:
    if not torch.cuda.is_available():
        return "pytorch"
    # Kept after 1a84c2d E2E: paged FlashInfer, not gather+SDPA. Fallback if missing.
    try:
        import flashinfer  # noqa: F401

        return "flashinfer"
    except ImportError:
        pass
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attn"


def eos_token_id_from_pin(pin_path: Path | None) -> int | None:
    """HF config eos_token_id. Lists (Qwen generation_config style) take the first id."""
    if pin_path is None or not pin_path.exists():
        return None
    raw = json.loads(pin_path.read_text())
    eos = raw.get("eos_token_id")
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    return int(eos) if eos is not None else None


def kv_budget_bytes() -> int:
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
    kv_budget: int | None = None,
    enable_prefix_cache: bool = False,
    num_speculative_tokens: int = 0,
    ngram_min: int = 2,
    ngram_max: int = 4,
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
    dtype_bytes = 2 if dtype == torch.bfloat16 else 4
    requested = kv_budget_bytes() if kv_budget is None else kv_budget
    budget = requested
    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
        physical = int(free * 0.90)
        if physical < requested:
            budget = physical
    num_blocks = max(
        32,
        num_kv_blocks_for_budget(model.cfg, budget, block_size, dtype_bytes=dtype_bytes),
    )
    if split is None:
        split = device == "cuda"
    runner = (
        SplitPagedRunner(
            model, cuda_graph=cuda_graph, decode_graph_max_kv=max_num_batched_tokens
        )
        if split
        else PagedRunner(
            model, cuda_graph=cuda_graph, decode_graph_max_kv=max_num_batched_tokens
        )
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
    )
    return Engine(config, runner)
