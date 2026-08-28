import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config


def test_preempted_paged_generate_matches_requests_run_alone():
    torch.manual_seed(21)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    a = list(range(1, 9))
    b = list(range(2, 10))
    roomy = Config(block_size=4, num_kv_blocks=64, max_num_seqs=4, max_num_batched_tokens=32)
    tight = Config(block_size=4, num_kv_blocks=4, max_num_seqs=4, max_num_batched_tokens=32)
    alone_a = Engine(roomy, PagedRunner(model)).generate(a, max_tokens=3)
    alone_b = Engine(
        Config(block_size=4, num_kv_blocks=64, max_num_seqs=4, max_num_batched_tokens=32),
        PagedRunner(model),
    ).generate(b, max_tokens=3)

    eng = Engine(tight, PagedRunner(model))
    id_a = eng.add_request(a, max_tokens=3)
    id_b = eng.add_request(b, max_tokens=3)
    got = {id_a: [], id_b: []}
    while not eng.is_finished():
        for rid, tok, _done in eng.step():
            if tok is not None:
                got[rid].append(tok)
    assert got[id_a] == alone_a
    assert got[id_b] == alone_b
    assert eng.scheduler.num_preemptions >= 1
