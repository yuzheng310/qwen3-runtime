from qwen3_runtime.sampling_params import SamplingParams
from tests.cpu.test_engine import _engine


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def test_max_tokens_sets_finish_reason_length():
    engine = _engine()
    rid = engine.add_request([1, 2, 3], max_tokens=2, hold_kv=True)
    engine.drain_request(rid)
    assert engine._requests[rid].finish_reason == "length"


def test_stop_token_sets_finish_reason_stop():
    engine = _engine()
    rid = engine.add_request(
        [1, 2],
        max_tokens=8,
        ignore_eos=False,
        stop_token_ids=(9,),
        forced_tokens=[7, 9, 1],
        hold_kv=True,
    )
    out = engine.drain_request(rid)
    assert out == [7, 9]
    assert engine._requests[rid].finish_reason == "stop"


def test_min_tokens_delays_eos_stop():
    engine = _engine()
    rid = engine.add_request(
        [1, 2],
        max_tokens=2,
        ignore_eos=False,
        stop_token_ids=(9,),
        sampling=SamplingParams(min_tokens=2),
        forced_tokens=[9, 8],
        hold_kv=True,
    )
    out = engine.drain_request(rid)
    assert out == [9, 8]
    assert engine._requests[rid].finish_reason == "length"


def test_stop_string_sets_finish_reason():
    engine = _engine()
    rid = engine.add_request(
        [0, 1],
        max_tokens=8,
        stop_strings=("C",),
        tokenizer=_Tok(),
        forced_tokens=[2],
        hold_kv=True,
    )
    engine.drain_request(rid)
    assert engine._requests[rid].finish_reason == "stop_string"


def test_abort_sets_finish_reason():
    engine = _engine()
    rid = engine.add_request([1, 2, 3, 4], max_tokens=8)
    engine.step()
    req = engine._requests[rid]
    engine.abort_generation()
    assert req.finish_reason == "abort"


def test_stream_request_matches_generate():
    engine = _engine()
    prompt = [1, 5, 9]
    a = engine.generate(prompt, max_tokens=3)
    engine2 = _engine()
    rid = engine2.add_request(prompt, max_tokens=3)
    b = list(engine2.stream_request(rid))
    assert a == b


def test_per_token_logprobs_are_recorded():
    import torch

    from qwen3_runtime.config import Config
    from qwen3_runtime.engine.engine import Engine
    from qwen3_runtime.engine.model_runner import PagedRunner
    from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM
    from tests.cpu.test_engine_model import tiny_config

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    engine = Engine(
        Config(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16),
        PagedRunner(model),
    )
    rid = engine.add_request(
        [1, 2, 3], max_tokens=3, hold_kv=True, sampling=SamplingParams(temperature=0.0, top_logprobs=3)
    )
    out = engine.drain_request(rid)
    req = engine._requests[rid]
    assert len(req.logprobs) == len(out) == 3
    assert len(req.top_logprobs) == 3
    assert all(len(row) == 3 for row in req.top_logprobs)
