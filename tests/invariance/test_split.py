import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner, SplitPagedRunner
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
from tests.cpu.test_tiny_qwen3 import tiny_config


def test_split_paged_generate_matches_packed_paged():
    torch.manual_seed(23)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    a = list(range(1, 8))
    b = list(range(2, 11))
    packed_cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=4, max_num_batched_tokens=16)
    split_cfg = Config(block_size=4, num_kv_blocks=32, max_num_seqs=4, max_num_batched_tokens=16)

    def run(cfg, runner_cls):
        eng = Engine(cfg, runner_cls(model))
        id_a = eng.add_request(a, max_tokens=3)
        id_b = eng.add_request(b, max_tokens=3)
        got = {id_a: [], id_b: []}
        while not eng.is_finished():
            for rid, tok, _done in eng.step():
                if tok is not None:
                    got[rid].append(tok)
        return got[id_a], got[id_b]

    pa, pb = run(packed_cfg, PagedRunner)
    sa, sb = run(split_cfg, SplitPagedRunner)
    assert (sa, sb) == (pa, pb)
