"""Task 1: measure the CUDA-fence artefact on the Poisson serving path.

Same workload twice: profile=True (old fences) vs profile=False (serving path).
Per-request max_tokens is sampled from the frozen 494-task corpus.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from bench.env import capture, git_dirty
from bench.metrics import merge_trials, percentile, summarize_traces
from bench.workloads import SEED, make_prompts, poisson_arrivals
from workloads.code_localization.output_lengths import recorded_output_tokens

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[1]


def _sample_output_tokens(n: int, *, seed: int) -> list[int]:
    import numpy as np

    corpus = recorded_output_tokens()
    rng = np.random.default_rng(seed + 2)
    idx = rng.integers(0, len(corpus), size=n)
    return [int(corpus[i]) for i in idx]


def _backlog_payload(run) -> dict:
    depths = [s.depth for s in run.backlog]
    return {
        "n_samples": len(run.backlog),
        "depth_mean": (sum(depths) / len(depths)) if depths else None,
        "depth_max": max(depths) if depths else None,
        "slope_per_s": run.backlog_slope_per_s,
        "samples": [
            {
                "t_s": s.t_s,
                "waiting": s.waiting,
                "running": s.running,
                "paused": s.paused,
                "depth": s.depth,
                "kv_free_blocks": getattr(s, "kv_free_blocks", None),
            }
            for s in run.backlog
        ],
        "paused_max": max((s.paused for s in run.backlog), default=None),
    }


def _accounting(run) -> dict:
    return {
        "offered": run.offered,
        "admitted": run.admitted,
        "rejected": run.rejected,
        "completed": run.completed,
        "slo_met": run.slo_met,
        "goodput_per_s": run.goodput_per_s,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, CUDA-synchronize before and after every engine.step (old Nsight path).",
    )
    parser.add_argument("--n-requests", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--poisson-rate", type=float, default=8.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--slo-ttft-s", type=float, default=None)
    parser.add_argument("--slo-tpot-s", type=float, default=None)
    parser.add_argument("--tiny", action="store_true", help="Synthetic CPU model; supplementary.")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    if not args.tiny and not args.model:
        raise SystemExit("full-scale needs --model or QWEN3_RUNTIME_MODEL")
    if not args.tiny and not args.allow_dirty and git_dirty(REPO):
        raise SystemExit("full-scale JSON requires a clean git tree (dirty=false)")

    from qwen3_runtime.config import Config
    from qwen3_runtime.engine.engine import Engine
    from qwen3_runtime.engine.factory import build_engine
    from qwen3_runtime.engine.model_runner import PagedRunner
    from qwen3_runtime.engine.serve import run_poisson
    from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig

    if args.tiny:
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
        model = Qwen3ForCausalLM(cfg).eval()
        vocab = cfg.vocab_size
        prompt_tokens = min(args.prompt_tokens, 8)
        n_requests = min(args.n_requests, 3)
        max_tokens = [2] * n_requests
        engine = Engine(
            Config(
                block_size=4,
                num_kv_blocks=64,
                max_num_seqs=min(args.max_num_seqs, 2),
                max_num_batched_tokens=32,
            ),
            PagedRunner(model),
        )
    else:
        engine = build_engine(
            args.model,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        vocab = engine.runner.model.cfg.vocab_size
        prompt_tokens = args.prompt_tokens
        n_requests = args.n_requests
        max_tokens = _sample_output_tokens(n_requests, seed=args.seed)

    prompts = make_prompts(
        SimpleNamespace(n_requests=n_requests, prompt_tokens=prompt_tokens),
        vocab,
        seed=args.seed,
    )
    arrivals = poisson_arrivals(n_requests, args.poisson_rate, seed=args.seed)

    def one_trial():
        run = run_poisson(
            engine,
            prompts,
            max_tokens,
            arrivals,
            profile=args.profile,
            slo_ttft_s=args.slo_ttft_s,
            slo_tpot_s=args.slo_tpot_s,
        )
        metrics = summarize_traces(run.traces, run.wall_s)
        metrics["accounting"] = _accounting(run)
        metrics["backlog"] = {
            "n_samples": len(run.backlog),
            "depth_mean": _backlog_payload(run)["depth_mean"],
            "depth_max": _backlog_payload(run)["depth_max"],
            "slope_per_s": run.backlog_slope_per_s,
        }
        metrics["ttft_s"]["p99"] = percentile(
            [t.ttft_s for t in run.traces if t.ttft_s is not None], 0.99
        )
        metrics["tpot_s"]["p99"] = percentile(
            [t.tpot_s for t in run.traces if t.tpot_s is not None], 0.99
        )
        metrics["sync_policy"] = run.sync_policy
        return metrics, run

    warmup = 0 if args.tiny else args.warmup
    n_trials = 1 if args.tiny else args.trials
    for _ in range(warmup):
        one_trial()
    pairs = [one_trial() for _ in range(n_trials)]
    trial_metrics = [m for m, _ in pairs]
    last_run = pairs[-1][1]
    metrics = merge_trials(trial_metrics)
    metrics["accounting"] = trial_metrics[-1]["accounting"]
    metrics["backlog"] = _backlog_payload(last_run)
    metrics["sync_policy"] = last_run.sync_policy
    metrics["ttft_s"]["p99"] = trial_metrics[-1]["ttft_s"].get("p99")
    metrics["tpot_s"]["p99"] = trial_metrics[-1]["tpot_s"].get("p99")

    cmd = ["python", "-m", "bench.run_slo_harness", "--out", str(args.out)]
    if args.profile:
        cmd.append("--profile")
    else:
        cmd.append("--no-profile")
    record = {
        "schema_version": SCHEMA_VERSION,
        "engine": "qwen3-runtime",
        "case": "slo_harness_fence" if args.profile else "slo_harness_serving",
        "seed": args.seed,
        "workload": {
            "name": "poisson_corpus_output",
            "scale": "tiny" if args.tiny else "full",
            "supplementary": True,
            "n_requests": n_requests,
            "prompt_tokens": prompt_tokens,
            "output_tokens": "recorded_output_tokens@token_reconstruction_v1",
            "output_tokens_used": max_tokens,
            "poisson_rate": args.poisson_rate,
            "arrival_process": "poisson",
            "sync_policy": last_run.sync_policy,
            "warmup": warmup,
            "trials": n_trials,
        },
        "engine_config": {
            "max_num_seqs": args.max_num_seqs if not args.tiny else min(args.max_num_seqs, 2),
            "max_num_batched_tokens": args.max_num_batched_tokens if not args.tiny else 32,
            "block_size": getattr(engine.config, "block_size", None),
            "num_kv_blocks": getattr(engine.config, "num_kv_blocks", None),
            "attention_backend": getattr(engine.config, "attention_backend", None),
            "cuda_graph": getattr(engine.config, "cuda_graph", False),
        },
        "environment": capture(REPO, cmd),
        "forced_length_ok": bool(metrics["forced_length_ok"]),
        "warmup": warmup,
        "metrics": metrics,
        "accounting": _accounting(last_run),
        "backlog": _backlog_payload(last_run),
    }
    record["environment"]["dtype"] = "fp32" if args.tiny else "bf16"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    if not record["forced_length_ok"]:
        raise SystemExit("forced_length_ok is false; trial void")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
