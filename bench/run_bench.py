"""Write measured JSON. Tiny/supplementary runs cannot populate 4B README tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bench.env import capture
from bench.metrics import merge_trials, summarize_traces
from bench.workloads import (
    ALL_CASES,
    PIN_REPO,
    PIN_REV,
    SEED,
    get_workload,
    make_prompts,
    token_budget_for_case,
)

SCHEMA_VERSION = 1


def _tiny_model():
    from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig

    cfg = Qwen3ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        rope_theta=10000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval()


def _run_once(engine_name: str, engine, prompts, workload):
    from bench.drivers import runtime_traces

    traces, wall = runtime_traces(engine, prompts, workload)
    return summarize_traces(traces, wall)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="tiny_cpu")
    parser.add_argument("--engine", default="qwen3-runtime", choices=("qwen3-runtime", "hf", "vllm"))
    parser.add_argument("--scale", default=None, choices=("tiny", "full"))
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--split", action="store_true", default=False)
    parser.add_argument("--no-split", action="store_true", default=False)
    parser.add_argument(
        "--attention-backend",
        default=None,
        choices=("pytorch", "sdpa", "flash_attn", "flashinfer", "triton"),
        help="Explicit attention backend. Default: factory (flashinfer on CUDA if installed).",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode-only CUDA Graph. Default: on for CUDA+FlashInfer (kept after 0d29dd2 E2E).",
    )
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]

    if args.case == "tiny_cpu":
        args.scale = "tiny"
        args.engine = "qwen3-runtime"

    if args.case in ALL_CASES and args.scale is None:
        args.scale = "full"
    if args.scale is None:
        args.scale = "tiny"

    if args.case in ALL_CASES and args.scale == "full" and not args.model:
        raise SystemExit("full-scale headline cases need --model or QWEN3_RUNTIME_MODEL")

    if args.scale == "full":
        from bench.env import git_dirty

        if git_dirty(repo):
            raise SystemExit("full-scale JSON requires a clean git tree (dirty=false)")

    if args.engine == "hf" and args.case not in ("latency", "prefill", "decode"):
        raise SystemExit("HF driver is latency-oriented; use qwen3-runtime or vLLM for serving cases")
    if args.engine == "vllm" and args.case == "online":
        raise SystemExit("vLLM online is vllm bench serve, not LLM.generate; not silently substituted")
    if args.engine in {"hf", "vllm"} and args.scale != "full":
        raise SystemExit(f"{args.engine} driver is full-scale only")

    warmup = args.warmup if args.warmup is not None else (0 if args.scale == "tiny" else 1)
    n_trials = args.trials if args.trials is not None else (1 if args.scale == "tiny" else 3)

    if args.case == "tiny_cpu":
        from qwen3_runtime.config import Config
        from qwen3_runtime.engine.engine import Engine
        from qwen3_runtime.engine.model_runner import PagedRunner
        from qwen3_runtime.engine.serve import run_closed_batch

        prompt = list(range(1, 9))
        engine = Engine(
            Config(block_size=4, num_kv_blocks=32, max_num_seqs=1, max_num_batched_tokens=32),
            PagedRunner(_tiny_model()),
        )
        traces = run_closed_batch(engine, [prompt], 8)
        metrics = summarize_traces(traces, traces[0].last_token_s - traces[0].arrival_s if traces[0].last_token_s else 0)
        metrics["output_tokens"] = traces[0].tokens
        record = {
            "schema_version": SCHEMA_VERSION,
            "engine": "qwen3-runtime",
            "case": "tiny_cpu",
            "seed": SEED,
            "workload": {"name": "tiny_cpu", "supplementary": True, "scale": "tiny"},
            "engine_config": {"block_size": 4},
            "environment": capture(repo, ["python", "-m", "bench.run_bench", "--case", "tiny_cpu"]),
            "forced_length_ok": bool(metrics["forced_length_ok"]),
            "metrics": metrics,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        return 0

    if args.case not in ALL_CASES:
        raise SystemExit(f"unknown case {args.case!r}")

    workload = get_workload(args.case, args.scale)
    split = True if args.split else False if args.no_split else None

    from bench.drivers import hf_latency_traces, runtime_traces, tiny_engine, vllm_closed_batch
    from qwen3_runtime.engine.factory import build_engine

    vocab = 32
    engine = None
    if args.engine == "qwen3-runtime":
        if args.scale == "tiny":
            model = _tiny_model()
            vocab = model.cfg.vocab_size
            engine = tiny_engine(model, workload, split=bool(split))
        else:
            engine = build_engine(
                args.model,
                max_num_seqs=max(workload.concurrency, min(workload.n_requests, 32)),
                max_num_batched_tokens=token_budget_for_case(args.case, workload),
                split=(True if split is None else split),
                attention_backend=args.attention_backend,
                cuda_graph=args.cuda_graph,
            )
            vocab = engine.runner.model.cfg.vocab_size
    elif args.engine == "hf":
        vocab = 151936
    else:
        vocab = 151936

    prompts = make_prompts(workload, vocab, seed=SEED)

    def one_trial():
        if args.engine == "qwen3-runtime":
            assert engine is not None
            traces, wall = runtime_traces(engine, prompts, workload)
            return summarize_traces(traces, wall)
        if args.engine == "hf":
            traces, wall = hf_latency_traces(args.model, prompts[0], workload.output_tokens)
            return summarize_traces(traces, wall)
        traces, wall = vllm_closed_batch(
            args.model, prompts, workload.output_tokens, case=args.case
        )
        return summarize_traces(traces, wall)

    for _ in range(warmup):
        one_trial()
    trial_metrics = [one_trial() for _ in range(n_trials)]
    metrics = merge_trials(trial_metrics)
    supplementary = workload.supplementary or args.scale != "full"
    record = {
        "schema_version": SCHEMA_VERSION,
        "engine": args.engine,
        "case": args.case,
        "seed": SEED,
        "workload": {
            "name": workload.name,
            "scale": args.scale,
            "supplementary": supplementary,
            "model": PIN_REPO if args.scale == "full" else "tiny-synthetic",
            "revision": PIN_REV if args.scale == "full" else None,
            "concurrency": workload.concurrency,
            "n_requests": workload.n_requests,
            "prompt_tokens": workload.prompt_tokens,
            "output_tokens": workload.output_tokens,
            "poisson_rate": workload.poisson_rate,
        },
        "engine_config": {
            "split": bool(getattr(engine, "runner", None).__class__.__name__ == "SplitPagedRunner")
            if engine is not None
            else None,
            "block_size": getattr(getattr(engine, "config", None), "block_size", None),
            "num_kv_blocks": getattr(getattr(engine, "config", None), "num_kv_blocks", None),
            "attention_backend": getattr(getattr(engine, "config", None), "attention_backend", None),
            "cuda_graph": getattr(getattr(engine, "config", None), "cuda_graph", False),
            "max_num_batched_tokens": getattr(
                getattr(engine, "config", None), "max_num_batched_tokens", None
            ),
        },
        "environment": capture(
            repo,
            ["python", "-m", "bench.run_bench", "--case", args.case, "--scale", args.scale, "--engine", args.engine],
        ),
        "forced_length_ok": bool(metrics["forced_length_ok"]),
        "warmup": warmup,
        "metrics": metrics,
    }
    record["environment"]["dtype"] = "bf16" if args.scale == "full" else "fp32"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    if not record["forced_length_ok"]:
        raise SystemExit("forced_length_ok is false; trial void")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
