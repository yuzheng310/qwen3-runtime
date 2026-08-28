from qwen3_runtime.config import Config
from qwen3_runtime.engine.cuda_graph import (
    can_skip_flashinfer_decode_plan,
    decode_kv_page_signature,
    is_pure_decode,
    use_decode_cuda_graph,
)
from qwen3_runtime.engine.request import Request


def _decode_req(*, prompt: int, computed: int) -> Request:
    req = Request(list(range(prompt)), max_tokens=8)
    req.num_computed_tokens = computed
    req.num_scheduled_tokens = 1
    req.append_token(0)
    return req


def test_decode_kv_page_signature_is_occupied_page_ids():
    assert decode_kv_page_signature([[3, 7, 9]], [17], 16) == ((3, 7),)
    assert decode_kv_page_signature([[3, 7, 9]], [16], 16) == ((3,),)
    assert decode_kv_page_signature([[3, 7], [1]], [32, 5], 16) == ((3, 7), (1,))


def test_skip_flashinfer_decode_plan_only_when_pages_are_unchanged():
    tables = [[3, 7, 9]]
    _, sig17 = can_skip_flashinfer_decode_plan(None, tables, [17], 16)
    skip18, _ = can_skip_flashinfer_decode_plan(sig17, tables, [18], 16)
    skip32, _ = can_skip_flashinfer_decode_plan(sig17, tables, [32], 16)
    skip33, _ = can_skip_flashinfer_decode_plan(sig17, tables, [33], 16)
    assert skip18 is True
    assert skip32 is True
    assert skip33 is False
    other_blocks, _ = can_skip_flashinfer_decode_plan(sig17, [[4, 8]], [18], 16)
    assert other_blocks is False


def test_use_decode_cuda_graph_skips_beyond_chunk_cap():
    short = _decode_req(prompt=256, computed=256)
    long = _decode_req(prompt=8192, computed=8192)
    assert use_decode_cuda_graph([short], max_kv=2048)
    assert not use_decode_cuda_graph([long], max_kv=2048)
    assert use_decode_cuda_graph([long], max_kv=None)
    prefill = Request(list(range(256)), max_tokens=8)
    prefill.num_computed_tokens = 0
    prefill.num_scheduled_tokens = 256
    assert not use_decode_cuda_graph([prefill], max_kv=2048)


def test_cuda_graph_config_defaults_off():
    assert Config().cuda_graph is False


def test_is_pure_decode_matches_split_runner_rule():
    req = Request([1, 2, 3], max_tokens=4)
    req.num_computed_tokens = 3
    req.num_scheduled_tokens = 1
    req.append_token(9)
    assert is_pure_decode([req])
    req.num_computed_tokens = 2
    assert not is_pure_decode([req])
