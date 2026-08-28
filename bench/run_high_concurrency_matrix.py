"""High-concurrency capacity matrix (research only). Always-admit, frozen A/B.

Workloads A–E from docs/HIGH_CONCURRENCY_RESEARCH.md. Writes one JSON per
(workload, λ) plus summary.json. GPU JSON should land outside the worktree
then be copied to bench/results/high-concurrency-knees/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from bench.env import capture, git_dirty
from bench.metrics import compact_traces, length_split_metrics, percentile, summarize_traces
from bench.run_slo_harness import _accounting, _backlog_payload, _sample_output_tokens
from bench.workloads import SEED, make_prompts, mix_length_prompts, poisson_arrivals
from qwen3_runtime.engine.factory import PIN, build_engine
from qwen3_runtime.engine.memory import kv_budget_for_simulated_card
from qwen3_runtime.engine.serve import run_poisson
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

REPO = Path(__file__).resolve().parents[1]
A_SLO = 4.5
B_SLO = 0.100
VOCAB = 151936


def _recorded_command(argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "bench.run_high_concurrency_matrix", *argv]


def _decomp(traces) -> dict:
    admitted = [t for t in traces if t.admitted]
    queues = [t.queue_s for t in admitted if t.queue_s is not None]
    prefills = [t.prefill_s for t in admitted if t.prefill_s is not None]
    ttfts = [t.ttft_s for t in admitted if t.ttft_s is not None]
    tpots = [t.tpot_s for t in admitted if t.tpot_s is not None]
    p99_ttft = percentile(ttfts, 0.99)
    p99_queue = percentile(queues, 0.99)
    return {
        "queue_s_p50": percentile(queues, 0.50),
        "queue_s_p95": percentile(queues, 0.95),
        "queue_s_p99": p99_queue,
        "prefill_s_p50": percentile(prefills, 0.50),
        "prefill_s_p95": percentile(prefills, 0.95),
        "prefill_s_p99": percentile(prefills, 0.99),
        "ttft_s_p95": percentile(ttfts, 0.95),
        "tpot_s_p95": percentile(tpots, 0.95),
        "queue_share_of_ttft_p99": (
            None
            if p99_ttft in (None, 0)
            else (p99_queue or 0) / p99_ttft
        ),
    }


def _kv_peak(run) -> dict:
    frees = [s.kv_free_blocks for s in run.backlog if s.kv_free_blocks is not None]
    return {
        "kv_free_blocks_min": min(frees) if frees else None,
        "kv_free_blocks_end": frees[-1] if frees else None,
    }


def _mix_prompts(n: int, *, short: int, long: int, long_frac: float, seed: int) -> list[list[int]]:
    return mix_length_prompts(
        n, short=short, long=long, long_frac=long_frac, vocab_size=VOCAB, seed=seed
    )


def _scheduler_stats(run) -> dict:
    mix = run.step_mix or {}
    n = int(mix.get("n_steps") or 0)
    tokens = int(mix.get("tokens_scheduled_sum") or 0)
    prefills = int(mix.get("prefill_reqs_sum") or 0)
    scheduled = int(mix.get("scheduled_reqs_sum") or 0)
    depths = [s.depth for s in run.backlog] if run.backlog else []
    return {
        "tokens_scheduled_per_step_mean": (tokens / n) if n else None,
        "prefill_reqs_per_step_mean": (prefills / n) if n else None,
        "scheduled_reqs_per_step_mean": (scheduled / n) if n else None,
        "active_requests_mean": (sum(depths) / len(depths)) if depths else None,
        "active_requests_max": max(depths) if depths else None,
    }


def _payload(run, *, case: str, rate: float, workload: dict, command: list[str]) -> dict:
    rec = {
        "schema_version": 1,
        "engine": "qwen3-runtime",
        "case": case,
        "seed": SEED,
        "workload": {**workload, "poisson_rate": rate, "warmup": 1, "admit": "always"},
        "environment": capture(REPO, command),
        "metrics": summarize_traces(run.traces, run.wall_s),
        "accounting": _accounting(run),
        "backlog": _backlog_payload(run),
        "decomp": _decomp(run.traces),
        "step_mix": run.step_mix,
        "scheduler_stats": _scheduler_stats(run),
        "length_split": length_split_metrics(run.traces),
        "num_preemptions": run.num_preemptions,
        "kv": _kv_peak(run),
        "per_request": compact_traces(run.traces),
        "slo_ttft_s": A_SLO,
        "slo_tpot_s": B_SLO,
    }
    rec["metrics"]["ttft_s_p50"] = percentile(
        [t.ttft_s for t in run.traces if t.admitted and t.ttft_s is not None], 0.50
    )
    rec["metrics"]["ttft_s_p99"] = percentile(
        [t.ttft_s for t in run.traces if t.admitted and t.ttft_s is not None], 0.99
    )
    rec["metrics"]["tpot_s_p50"] = percentile(
        [t.tpot_s for t in run.traces if t.admitted and t.tpot_s is not None], 0.50
    )
    rec["metrics"]["tpot_s_p99"] = percentile(
        [t.tpot_s for t in run.traces if t.admitted and t.tpot_s is not None], 0.99
    )
    rec["environment"]["dtype"] = "bf16"
    return rec


def _one(engine, prompts, max_tokens, rate, n, seed) -> object:
    arrivals = poisson_arrivals(n, rate, seed=seed)
    run_poisson(
        engine, prompts, max_tokens, arrivals, slo_ttft_s=A_SLO, slo_tpot_s=B_SLO
    )
    return run_poisson(
        engine, prompts, max_tokens, arrivals, slo_ttft_s=A_SLO, slo_tpot_s=B_SLO
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument("--workloads", default="A,B,C,D,E")
    parser.add_argument(
        "--rates",
        default=None,
        help="Comma-separated Poisson rates; overrides the per-workload default sweep.",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    args = parser.parse_args(raw_argv)
    if not args.model:
        raise SystemExit("need --model")
    if not args.allow_dirty and git_dirty(REPO):
        raise SystemExit("dirty worktree")

    pin_cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    kv = kv_budget_for_simulated_card(pin_cfg, 24.0)
    engine = build_engine(
        args.model,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_budget=kv,
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {x.strip().upper() for x in args.workloads.split(",") if x.strip()}
    summary = []

    def rates_for(defaults: tuple[float, ...]) -> tuple[float, ...]:
        if not args.rates:
            return defaults
        return tuple(float(x) for x in args.rates.split(",") if x.strip())

    def emit(name: str, rate: float, run, workload: dict) -> None:
        workload = {
            **workload,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
        rec = _payload(
            run,
            case=f"hc_{name}_l{rate:g}",
            rate=rate,
            workload=workload,
            command=_recorded_command(raw_argv),
        )
        rec["engine_config"] = {
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "num_speculative_tokens": args.num_speculative_tokens,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "enable_prefix_cache": False,
            "kv_budget_bytes": kv,
        }
        path = args.out_dir / f"{name}_lambda_{rate:g}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        row = {
            "workload": name,
            "lambda": rate,
            "n": workload.get("n_requests"),
            "goodput_per_s": run.goodput_per_s,
            "ttft_p50": rec["metrics"]["ttft_s_p50"],
            "ttft_p99": rec["metrics"]["ttft_s_p99"],
            "tpot_p50": rec["metrics"]["tpot_s_p50"],
            "tpot_p99": rec["metrics"]["tpot_s_p99"],
            "queue_p99": rec["decomp"]["queue_s_p99"],
            "prefill_p99": rec["decomp"]["prefill_s_p99"],
            "queue_share_p99": rec["decomp"]["queue_share_of_ttft_p99"],
            "slo_met": run.slo_met,
            "completed": run.completed,
            "preempts": run.num_preemptions,
            "mixed_steps": (run.step_mix or {}).get("n_mixed"),
            "decode_only_steps": (run.step_mix or {}).get("n_decode_only"),
            "short_ttft_p99": rec["length_split"].get("short_ttft_s_p99"),
            "long_ttft_p99": rec["length_split"].get("long_ttft_s_p99"),
            "tokens_per_step": rec["scheduler_stats"]["tokens_scheduled_per_step_mean"],
            "prefills_per_step": rec["scheduler_stats"]["prefill_reqs_per_step_mean"],
            "kv_free_min": rec["kv"]["kv_free_blocks_min"],
            "holds_ab": (
                rec["metrics"]["ttft_s_p99"] is not None
                and rec["metrics"]["tpot_s_p99"] is not None
                and rec["metrics"]["ttft_s_p99"] <= A_SLO
                and rec["metrics"]["tpot_s_p99"] <= B_SLO
            ),
        }
        summary.append(row)
        print(json.dumps(row), flush=True)

    if "A" in wanted:
        n, pt, out = 48, 256, 128
        prompts = make_prompts(SimpleNamespace(n_requests=n, prompt_tokens=pt), VOCAB, seed=SEED)
        wl = {"name": "A_short_homog", "n_requests": n, "prompt_tokens": pt, "max_tokens": out}
        for rate in rates_for((4.0, 6.0, 8.0, 10.0, 12.0)):
            emit("A", rate, _one(engine, prompts, out, rate, n, SEED), wl)

    if "B" in wanted:
        n, pt, out = 16, 9981, 128
        prompts = make_prompts(SimpleNamespace(n_requests=n, prompt_tokens=pt), VOCAB, seed=SEED)
        wl = {"name": "B_long_homog", "n_requests": n, "prompt_tokens": pt, "max_tokens": out}
        for rate in rates_for((1.0, 2.0, 3.0, 4.0)):
            emit("B", rate, _one(engine, prompts, out, rate, n, SEED), wl)

    if "C" in wanted:
        n, out = 40, 128
        prompts = _mix_prompts(n, short=256, long=9981, long_frac=0.20, seed=SEED)
        wl = {
            "name": "C_mixed_80_20",
            "n_requests": n,
            "prompt_tokens": "80pct_256_20pct_9981",
            "max_tokens": out,
        }
        for rate in rates_for((2.0, 4.0, 6.0, 8.0)):
            emit("C", rate, _one(engine, prompts, out, rate, n, SEED), wl)

    if "C90" in wanted:
        n, out = 40, 128
        prompts = _mix_prompts(n, short=256, long=9981, long_frac=0.10, seed=SEED)
        wl = {
            "name": "C90_mixed_90_10",
            "n_requests": n,
            "prompt_tokens": "90pct_256_10pct_9981",
            "max_tokens": out,
        }
        for rate in rates_for((2.0, 4.0, 6.0)):
            emit("C90", rate, _one(engine, prompts, out, rate, n, SEED), wl)

    if "C50" in wanted:
        n, out = 40, 128
        prompts = _mix_prompts(n, short=256, long=9981, long_frac=0.50, seed=SEED)
        wl = {
            "name": "C50_mixed_50_50",
            "n_requests": n,
            "prompt_tokens": "50pct_256_50pct_9981",
            "max_tokens": out,
        }
        for rate in rates_for((2.0, 4.0, 6.0)):
            emit("C50", rate, _one(engine, prompts, out, rate, n, SEED), wl)

    if "D" in wanted:
        n, pt, out = 32, 256, 256
        prompts = make_prompts(SimpleNamespace(n_requests=n, prompt_tokens=pt), VOCAB, seed=SEED)
        wl = {"name": "D_decode_heavy", "n_requests": n, "prompt_tokens": pt, "max_tokens": out}
        for rate in rates_for((4.0, 6.0, 8.0, 10.0)):
            emit("D", rate, _one(engine, prompts, out, rate, n, SEED), wl)

    if "E" in wanted:
        n = 32
        prompts = _mix_prompts(n, short=2048, long=9981, long_frac=0.5, seed=SEED + 7)
        outs = _sample_output_tokens(n, seed=SEED)
        wl = {
            "name": "E_agent_like",
            "n_requests": n,
            "prompt_tokens": "50pct_2048_50pct_9981",
            "max_tokens": "codescout_corpus",
        }
        for rate in rates_for((1.0, 2.0, 3.0, 4.0)):
            emit("E", rate, _one(engine, prompts, outs, rate, n, SEED), wl)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
