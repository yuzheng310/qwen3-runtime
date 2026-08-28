import sys

from bench.run_high_concurrency_matrix import _recorded_command
from bench.workloads import mix_length_prompts


def test_mixed_80_20_prompt_lengths():
    ps = mix_length_prompts(20, short=256, long=9981, long_frac=0.20, vocab_size=151936, seed=20260825)
    assert len(ps) == 20
    n_long = sum(1 for p in ps if len(p) == 9981)
    n_short = sum(1 for p in ps if len(p) == 256)
    assert n_long == 4
    assert n_short == 16


def test_recorded_command_preserves_every_cli_argument():
    argv = [
        "--out-dir",
        "/tmp/results",
        "--model",
        "/models/qwen",
        "--workloads",
        "A",
        "--rates",
        "12",
        "--num-speculative-tokens",
        "16",
        "--ngram-min",
        "2",
        "--ngram-max",
        "4",
        "--allow-dirty",
    ]

    assert _recorded_command(argv) == [
        sys.executable,
        "-m",
        "bench.run_high_concurrency_matrix",
        *argv,
    ]
