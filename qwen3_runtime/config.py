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
    # Session CPU KV offload is deliberately opt-in.  ``async`` is reserved
    # for a future event-backed implementation and is rejected by Engine.
    session_cpu_offload: str = "off"
    cpu_kv_max_bytes: int = 0
    cpu_kv_pinned_max_bytes: int = 0
    transfer_chunk_bytes: int = 8 * 1024 * 1024
    max_inflight_transfers: int = 1
    restore_wait_budget: int = 1

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("ngram window must satisfy 1 <= ngram_min <= ngram_max")
        if self.session_cpu_offload not in {"off", "sync", "async"}:
            raise ValueError("session_cpu_offload must be one of: off, sync, async")
        if self.session_cpu_offload == "sync" and self.cpu_kv_max_bytes <= 0:
            raise ValueError("cpu_kv_max_bytes must be positive when CPU offload is enabled")
        if self.cpu_kv_max_bytes < 0 or self.cpu_kv_pinned_max_bytes < 0:
            raise ValueError("CPU KV budgets must be non-negative")
        if self.cpu_kv_pinned_max_bytes > self.cpu_kv_max_bytes and self.cpu_kv_max_bytes:
            raise ValueError("pinned CPU KV budget is part of, not extra to, the CPU budget")
        if self.transfer_chunk_bytes <= 0:
            raise ValueError("transfer_chunk_bytes must be positive")
        if self.max_inflight_transfers != 1:
            raise ValueError("the first implementation supports exactly one transfer")
        if self.restore_wait_budget < 0:
            raise ValueError("restore_wait_budget must be non-negative")
