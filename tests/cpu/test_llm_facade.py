from qwen3_runtime import LLM, SamplingParams
from qwen3_runtime.serving.openai import parse_path, plan_session_continuation


def test_llm_generate_tiny_random_ids():
    llm = LLM()
    out = llm.generate([1, 2, 3], SamplingParams(temperature=0.0), max_tokens=3)
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(t, int) for t in out)


def test_llm_session_second_turn_extends_held_kv():
    llm = LLM()
    greedy = SamplingParams(temperature=0.0)
    with llm.session() as session:
        session.turn([1, 2, 3], max_tokens=2, sampling=greedy)
        held = [1, 2, 3] + session.last_tokens + [7]
        session.turn(held, max_tokens=2, sampling=greedy)
        assert len(session.last_tokens) == 2
        # Turn 1 prefills 3 and holds 5 (prompt plus what it generated). Turn 2
        # is handed those 5 plus one new token, so it prefills exactly one.
        report = session.report()
        assert report["prompt_tokens"] == 3 + len(held)
        assert report["reused_tokens"] == len(held) - 1
        assert report["reuse_pct"] > 50.0


def test_openai_session_paths():
    assert parse_path("/v1/s/g0s3/chat/completions") == ("chat", "g0s3")
    assert parse_path("/v1/s/g0s3/reset") == ("reset_one", "g0s3")
    assert parse_path("/v1/chat/completions") == ("chat", "default")
    assert parse_path("/v1/completions") == ("complete", None)
    assert parse_path("/v1/models") == ("models", None)
    assert plan_session_continuation([1, 2, 3], [1, 2, 3, 4]) == ("exact", [4])
    assert plan_session_continuation([1, 2, 3], [9, 8]) == ("replace", [])
