import torch

from qwen3_runtime.kv.paged import PagedKVPool


def test_store_and_gather_roundtrip_across_block_boundary():
    pool = PagedKVPool(
        num_layers=2,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    k = torch.randn(6, 2, 8)
    v = torch.randn(6, 2, 8)
    slots = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    pool.store(layer=1, key=k, value=v, slot_mapping=slots)
    gk, gv = pool.gather(layer=1, block_table=[0, 1], seq_len=6)
    torch.testing.assert_close(gk, k)
    torch.testing.assert_close(gv, v)
    # other layer untouched
    z, _ = pool.gather(layer=0, block_table=[0, 1], seq_len=6)
    assert torch.count_nonzero(z) == 0


def test_gather_tensor_block_table_matches_python_list():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    k = torch.randn(6, 2, 8)
    v = torch.randn(6, 2, 8)
    pool.store(layer=0, key=k, value=v, slot_mapping=torch.arange(6))
    table_t = torch.tensor([0, 1], dtype=torch.long)
    gk_list, gv_list = pool.gather(layer=0, block_table=[0, 1], seq_len=6)
    gk_t, gv_t = pool.gather(layer=0, block_table=table_t, seq_len=6)
    torch.testing.assert_close(gk_t, gk_list)
    torch.testing.assert_close(gv_t, gv_list)
