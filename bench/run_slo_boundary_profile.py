"""Task 3: serving-level profile at the frozen SLO boundary.

Request: Poisson λ=12 (p99 TTFT just above A).
Session: N=8 on a simulated 24 GiB card (first preempt storm; p99 above A and B).

Does not implement an intervention. Records queue vs prefill split and
prefill/decode mix per step. JSON is written outside the git worktree on GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from bench.env import capture, git_dirty
from bench.metrics import percentile
from bench.run_slo_harness import _accounting, _backlog_payload, _sample_output_tokens
from bench.run_session_capacity import _make_sessions
from bench.workloads import SEED, make_prompts, poisson_arrivals
from qwen3_runtime.engine.factory import PIN, build_engine
from qwen3_runtime.engine.memory import kv_budget_for_simulated_card
from qwen3_runtime.serving.slo_harness import run_poisson, run_sessions
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

REPO = Path(__file__).resolve().parents[1]
A = 4.5
B = 0.100


def _decomp(traces) -> dict:
    admitted = [t for t in traces if t.admitted]
    queues = [t.queue_s for t in admitted if t.queue_s is not None]
    prefills = [t.prefill_s for t in admitted if t.prefill_s is not None]
    ttfts = [t.ttft_s for t in admitted if t.ttft_s is not None]
    tpots = [t.tpot_s for t in admitted if t.tpot_s is not None]
    return {
        "n_traces": len(admitted),
        "queue_s_p50": percentile(queues, 0.50),
        "queue_s_p99": percentile(queues, 0.99),
        "prefill_s_p50": percentile(prefills, 0.50),
        "prefill_s_p99": percentile(prefills, 0.99),
        "ttft_s_p50": percentile(ttfts, 0.50),
        "ttft_s_p99": percentile(ttfts, 0.99),
        "tpot_s_p50": percentile(tpots, 0.50),
        "tpot_s_p99": percentile(tpots, 0.99),
        "queue_share_of_ttft_p99": (
            None
            if not ttfts or percentile(ttfts, 0.99) in (None, 0)
            else (percentile(queues, 0.99) or 0) / percentile(ttfts, 0.99)
        ),
        "per_request": [
            {
                "prompt_len": t.prompt_len,
                "max_tokens": t.max_tokens,
                "ttft_s": t.ttft_s,
                "queue_s": t.queue_s,
                "prefill_s": t.prefill_s,
                "tpot_s": t.tpot_s,
            }
            for t in admitted
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-request", action="store_true")
    parser.add_argument("--skip-session", action="store_true")
    args = parser.parse_args(argv)
    if not args.tiny and not args.model:
        raise SystemExit("need --model")
    if not args.tiny and not args.allow_dirty and git_dirty(REPO):
        raise SystemExit("dirty worktree")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.tiny:
        from tests.cpu.test_engine import FakeModelRunner
        from qwen3_runtime.config import Config
        from qwen3_runtime.engine.engine import Engine

        engine = Engine(
            Config(max_num_batched_tokens=32, max_num_seqs=2, num_kv_blocks=64, block_size=4),
            FakeModelRunner(),
        )
        prompts = [[1, 2], [3, 4, 5], [6, 7]]
        arrivals = poisson_arrivals(3, 8.0, seed=SEED)
        run = run_poisson(engine, prompts, [2, 3, 2], arrivals, slo_ttft_s=A, slo_tpot_s=B)
        rec = {
            "schema_version": 1,
            "case": "slo_boundary_tiny",
            "step_mix": run.step_mix,
            "decomp": _decomp(run.traces),
            "accounting": _accounting(run),
        }
        (args.out_dir / "tiny.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps({"tiny": True, "steps": run.step_mix["n_steps"]}))
        return 0

    pin_cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    kv24 = kv_budget_for_simulated_card(pin_cfg, 24.0)
    corpus_out = _sample_output_tokens(96, seed=SEED)

    def _release(engine) -> None:
        import gc
        import torch

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_request:
        engine = build_engine(
            args.model, max_num_seqs=8, max_num_batched_tokens=2048
        )
        n = 96
        prompts = make_prompts(
            SimpleNamespace(n_requests=n, prompt_tokens=256),
            engine.runner.model.cfg.vocab_size,
            seed=SEED,
        )
        arrivals = poisson_arrivals(n, 12.0, seed=SEED)
        max_tokens = corpus_out[:n]
        run_poisson(engine, prompts, max_tokens, arrivals, profile=False)
        run = run_poisson(
            engine,
            prompts,
            max_tokens,
            arrivals,
            profile=False,
            slo_ttft_s=A,
            slo_tpot_s=B,
        )
        rec = {
            "schema_version": 1,
            "engine": "qwen3-runtime",
            "case": "slo_boundary_request_lambda12",
            "seed": SEED,
            "workload": {
                "name": "request_capacity",
                "poisson_rate": 12.0,
                "n_requests": n,
                "prompt_tokens": 256,
                "sync_policy": "serving",
                "slo_ttft_s": A,
                "slo_tpot_s": B,
                "warmup": 1,
            },
            "environment": capture(REPO, ["python", "-m", "bench.run_slo_boundary_profile"]),
            "accounting": _accounting(run),
            "backlog": {
                "depth_max": _backlog_payload(run)["depth_max"],
                "slope_per_s": run.backlog_slope_per_s,
                "paused_max": _backlog_payload(run)["paused_max"],
            },
            "num_preemptions": run.num_preemptions,
            "step_mix": run.step_mix,
            "decomp": _decomp(run.traces),
        }
        rec["environment"]["dtype"] = "bf16"
        (args.out_dir / "request_lambda12.json").write_text(json.dumps(rec, indent=2) + "\n")
        d = rec["decomp"]
        print(
            json.dumps(
                {
                    "case": rec["case"],
                    "slo_met": run.slo_met,
                    "ttft_p99": d["ttft_s_p99"],
                    "queue_p99": d["queue_s_p99"],
                    "prefill_p99": d["prefill_s_p99"],
                    "mixed_steps": run.step_mix["n_mixed"] if run.step_mix else None,
                    "dirty": rec["environment"]["dirty"],
                }
            ),
            flush=True,
        )
        _release(engine)

    if not args.skip_session:
        engine = build_engine(
            args.model,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            kv_budget=kv24,
        )
        n_sess = 8
        sessions = _make_sessions(
            n_sess,
            turns=5,
            prompt_tokens=9981,
            suffix_tokens=1633,
            think_s=1.0,
            max_tokens=corpus_out,
            vocab=engine.runner.model.cfg.vocab_size,
            seed=SEED,
        )
        run_sessions(
            engine, sessions, profile=False, max_active=n_sess, slo_ttft_s=A, slo_tpot_s=B
        )
        run = run_sessions(
            engine, sessions, profile=False, max_active=n_sess, slo_ttft_s=A, slo_tpot_s=B
        )
        rec = {
            "schema_version": 1,
            "engine": "qwen3-runtime",
            "case": "slo_boundary_session_n8_24gib",
            "seed": SEED,
            "workload": {
                "name": "session_capacity",
                "n_sessions": n_sess,
                "turns": 5,
                "prompt_tokens": 9981,
                "suffix_tokens": 1633,
                "think_s": 1.0,
                "sync_policy": "serving",
                "simulate_card_gib": 24.0,
                "slo_ttft_s": A,
                "slo_tpot_s": B,
                "warmup": 1,
            },
            "environment": capture(REPO, ["python", "-m", "bench.run_slo_boundary_profile"]),
            "accounting": _accounting(run),
            "backlog": {
                "depth_max": _backlog_payload(run)["depth_max"],
                "slope_per_s": run.backlog_slope_per_s,
                "paused_max": _backlog_payload(run)["paused_max"],
            },
            "num_preemptions": run.num_preemptions,
            "kv_exhausted": run.kv_exhausted,
            "step_mix": run.step_mix,
            "decomp": _decomp(run.traces),
        }
        rec["environment"]["dtype"] = "bf16"
        (args.out_dir / "session_n8_24gib.json").write_text(json.dumps(rec, indent=2) + "\n")
        d = rec["decomp"]
        print(
            json.dumps(
                {
                    "case": rec["case"],
                    "slo_met": run.slo_met,
                    "ttft_p99": d["ttft_s_p99"],
                    "queue_p99": d["queue_s_p99"],
                    "prefill_p99": d["prefill_s_p99"],
                    "preempts": run.num_preemptions,
                    "mixed_steps": run.step_mix["n_mixed"] if run.step_mix else None,
                    "dirty": rec["environment"]["dirty"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
