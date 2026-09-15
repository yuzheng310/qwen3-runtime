from __future__ import annotations

import pytest
import torch

from qwen3_runtime.kv.cpu_store import (
    CpuKVCapacityError,
    CpuKVStore,
    KVSnapshotMetadata,
)
from qwen3_runtime.kv.paged import PagedKVPool


def _meta(
    snapshot_id: int, shape: tuple[int, ...], allocated: int
) -> KVSnapshotMetadata:
    return KVSnapshotMetadata(
        session_key="s",
        request_id=1,
        snapshot_id=snapshot_id,
        weight_epoch=0,
        kv_epoch=0,
        token_ids=(1, 2, 3, 4, 5),
        num_computed_tokens=5,
        logical_blocks=shape[2],
        tail_valid_tokens=1,
        dtype="torch.float32",
        num_layers=shape[1],
        block_size=shape[3],
        num_kv_heads=shape[4],
        head_dim=shape[5],
        layout="[K/V, layer, block, offset, kv_head, dim]",
        shape=shape,
        allocated_bytes=allocated,
    )


def test_cpu_store_reserve_commit_abort_and_budget_accounting():
    shape = (2, 1, 1, 4, 1, 2)
    nbytes = CpuKVStore.required_bytes(shape, torch.float32)
    store = CpuKVStore(nbytes)
    reservation = store.reserve(_meta(1, shape, nbytes))
    reservation.buffer.fill_(3)
    snapshot = reservation.commit()
    assert store.stats()["committed_bytes"] == nbytes
    assert torch.all(snapshot.buffer == 3)
    with pytest.raises(CpuKVCapacityError):
        store.reserve(_meta(2, shape, nbytes))
    removed = store.delete(1)
    assert removed is snapshot
    assert store.stats()["committed_bytes"] == 0

    # Deleting an index entry does not release storage still held by a caller.
    assert store.stats()["managed_host_buffer_bytes"] == nbytes
    with pytest.raises(CpuKVCapacityError):
        store.reserve(_meta(3, shape, nbytes))
    del removed, snapshot, reservation
    assert store.stats()["managed_host_buffer_bytes"] == 0
    reservation = store.reserve(_meta(3, shape, nbytes))
    reservation.abort()
    assert store.stats()["reserved_bytes"] == 0


def test_paged_pool_roundtrip_is_block_ordered_and_zeroes_tail_padding():
    pool = PagedKVPool(
        num_layers=2,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for block in range(4):
        pool.cache[:, :, block].fill_(block + 1)
    snapshot = pool.export_blocks([2, 0], valid_tokens=5, chunk_bytes=1)
    assert snapshot.shape == (2, 2, 2, 4, 1, 2)
    assert torch.all(snapshot[:, :, 0, :4] == 3)
    assert torch.all(snapshot[:, :, 1, :1] == 1)
    assert torch.count_nonzero(snapshot[:, :, 1, 1:]) == 0

    pool.cache.zero_()
    pool.import_blocks([1, 3], snapshot, valid_tokens=5, chunk_bytes=1)
    key, value = pool.gather(0, [1, 3], seq_len=5)
    assert torch.all(key[:4] == 3)
    assert torch.all(key[4:] == 1)
    torch.testing.assert_close(value, key)


def test_paged_pool_rejects_invalid_snapshot_layout():
    pool = PagedKVPool(1, 2, 4, 1, 2, torch.float32, torch.device("cpu"))
    with pytest.raises(ValueError, match="valid_tokens"):
        pool.export_blocks([0], valid_tokens=5, chunk_bytes=64)
