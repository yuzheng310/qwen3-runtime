from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Runtime knobs. Model weights/revision live in MODEL_SPEC, not here.

    These defaults are CPU-safe so ``Config()`` can be constructed in unit tests.
    ``build_engine`` is the production path: it selects flashinfer (or pytorch),
    turns on CUDA Graph for flashinfer/triton, and fills ``num_kv_blocks`` from
    free VRAM.
    """

    block_size: int = 16
    # None until factory (or a test) sets it. Engine refuses to start without it.
    num_kv_blocks: int | None = None
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 2048
    eos_token_id: int | None = None
    attention_backend: str = "pytorch"
    # Decode-only CUDA Graph. Dataclass stays off; factory turns it on for CUDA+FlashInfer/Triton.
    cuda_graph: bool = False
    enable_prefix_cache: bool = False
    # Speculative decoding. 0 disables.
    num_speculative_tokens: int = 0
    ngram_min: int = 2
    ngram_max: int = 4

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("ngram window must satisfy 1 <= ngram_min <= ngram_max")
