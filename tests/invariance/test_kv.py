import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.reference.eager_runner import PytorchEagerRunner
from qwen3_runtime.kv.paged import PagedBatch, PagedKVPool
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config


def test_paged_chunked_prefill_matches_full_sequence_logits():
    torch.manual_seed(7)
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    with torch.no_grad():
        full = model(ids, torch.arange(ids.numel()))

    pool = PagedKVPool(
        num_layers=cfg.num_hidden_layers,
        num_blocks=4,
        block_size=4,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    block_table = [0, 1]

    def chunk(start: int, n: int) -> torch.Tensor:
        pos = torch.arange(start, start + n)
        slots = pos.clone()
        batch = PagedBatch(
            pool=pool,
            slot_mapping=slots,
            block_tables=[block_table],
            kv_lens=[start + n],
            cu_seqlens=[0, n],
        )
        return model(ids[start : start + n], pos, paged=batch)

    with torch.no_grad():
        chunk(0, 3)
        second = chunk(3, 5)
    torch.testing.assert_close(second[-1], full[-1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second[0], full[3], atol=1e-5, rtol=1e-5)


def test_paged_engine_tokens_match_eager_recompute():
    torch.manual_seed(8)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = [1, 4, 7, 2, 9]
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    eager = Engine(cfg, PytorchEagerRunner(model)).generate(prompt, max_tokens=4)
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16)
    paged = Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=4)
    assert paged == eager


def test_paged_block_size_does_not_change_tokens():
    torch.manual_seed(9)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = list(range(1, 13))
    got = {}
    for block_size in (4, 8, 16):
        cfg = Config(block_size=block_size, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=32)
        got[block_size] = Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=3)
    assert got[4] == got[8] == got[16]


def test_paged_chunked_generate_matches_unchunked():
    torch.manual_seed(10)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prompt = list(range(1, 17))
    def gen(budget: int) -> list[int]:
        cfg = Config(block_size=4, num_kv_blocks=64, max_num_seqs=2, max_num_batched_tokens=budget)
        return Engine(cfg, PagedRunner(model)).generate(prompt, max_tokens=3)

    assert gen(3) == gen(64)


def test_paged_packed_batch_matches_separate_requests():
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    a = [1, 2, 3]
    b = [4, 5, 6, 7, 8]
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=4, max_num_batched_tokens=32)
    alone_a = Engine(cfg, PagedRunner(model)).generate(a, max_tokens=2)
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=4, max_num_batched_tokens=32)
    alone_b = Engine(cfg, PagedRunner(model)).generate(b, max_tokens=2)
    cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=4, max_num_batched_tokens=32)
    eng = Engine(cfg, PagedRunner(model))
    id_a = eng.add_request(a, max_tokens=2)
    id_b = eng.add_request(b, max_tokens=2)
    got = {id_a: [], id_b: []}
    while not eng.is_finished():
        for rid, tok, _done in eng.step():
            if tok is not None:
                got[rid].append(tok)
    assert got[id_a] == alone_a
    assert got[id_b] == alone_b
