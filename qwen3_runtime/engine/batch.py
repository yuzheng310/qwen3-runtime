"""Packed-batch helpers.

Prefill is flattened tokens, not one row per request. Last-token selection
must be explicit (see project spec §11).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen3_runtime.engine.block_manager import BlockManager
    from qwen3_runtime.engine.request import Request


@dataclass
class PackedBatch:
    """Host-side packed sequences. Device tensors are built at the call site."""

    ids: list[int] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)
    cu_seqlens: list[int] = field(default_factory=lambda: [0])
    block_tables: list = field(default_factory=list)
    kv_lens: list[int] = field(default_factory=list)
    sample: list[bool] = field(default_factory=list)
    scheduled_rows: list[int] = field(default_factory=list)


def last_token_indices(cu_seqlens_q: Sequence[int]) -> list[int]:
    if len(cu_seqlens_q) < 2:
        raise ValueError("cu_seqlens_q must contain at least one sequence")
    return [cu_seqlens_q[i] - 1 for i in range(1, len(cu_seqlens_q))]


def sampled_logit_rows(cu_seqlens_q: Sequence[int], sample: Sequence[bool]) -> list[int]:
    """Packed rows that need an LM-head logit (last token of each sampling sequence)."""
    last = last_token_indices(cu_seqlens_q)
    if len(sample) != len(last):
        raise ValueError("sample flags must align with packed sequences")
    return [last[i] for i, flag in enumerate(sample) if flag]


def pack_paged(reqs: Sequence[Request], block_manager: BlockManager) -> PackedBatch:
    """Pack only the newly scheduled span. slots/block_tables come from the pool."""
    packed = PackedBatch()
    for req in reqs:
        start = req.num_computed_tokens
        n = req.num_scheduled_tokens
        end = start + n
        packed.ids.extend(req.token_ids[start:end])
        packed.pos.extend(range(start, end))
        packed.slots.extend(block_manager.slot_mapping(req, start, n))
        packed.cu_seqlens.append(len(packed.ids))
        packed.block_tables.append(req.block_table)
        packed.kv_lens.append(end)
        packed.sample.append(end >= len(req.token_ids))
        packed.scheduled_rows.extend(range(packed.cu_seqlens[-2], packed.cu_seqlens[-1]))
    return packed


def pack_eager(reqs: Sequence[Request]) -> PackedBatch:
    """Recompute each prefix from token 0. No paged slots."""
    packed = PackedBatch()
    for req in reqs:
        end = req.num_computed_tokens + req.num_scheduled_tokens
        start_row = len(packed.ids)
        packed.ids.extend(req.token_ids[:end])
        packed.pos.extend(range(end))
        packed.cu_seqlens.append(len(packed.ids))
        packed.sample.append(end >= len(req.token_ids))
        n = req.num_scheduled_tokens
        packed.scheduled_rows.extend(range(start_row + end - n, start_row + end))
    return packed
