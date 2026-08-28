"""Session-capacity sweep: N concurrent multi-turn sessions holding KV during think time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from bench.env import capture, git_dirty
from bench.metrics import compact_traces, percentile, summarize_traces
from bench.run_slo_harness import _accounting, _backlog_payload, _sample_output_tokens
from bench.workloads import SEED, make_prompts
from qwen3_runtime.engine.factory import PIN, build_engine
from qwen3_runtime.engine.memory import (
    bytes_per_block,
    estimate_weight_bytes,
    kv_budget_for_simulated_card,
)
from qwen3_runtime.engine.serve import (
    SessionTurn,
    always_admit,
    request_slo_admit,
    run_sessions,
    session_slo_admit,
)
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

REPO = Path(__file__).resolve().parents[1]

_ADMIT = {
    "always": always_admit,
    "request-slo": request_slo_admit,
    "session-slo": session_slo_admit,
}


def _make_sessions(
    n_sessions: int,
    *,
    turns: int,
    prompt_tokens: int,
    suffix_tokens: int,
    think_s: float,
    max_tokens: list[int],
    vocab: int,
    seed: int,
) -> list[list[SessionTurn]]:
    first = make_prompts(
        SimpleNamespace(n_requests=n_sessions, prompt_tokens=prompt_tokens),
        vocab,
        seed=seed,
    )
    suffixes = make_prompts(
        SimpleNamespace(n_requests=n_sessions * max(turns - 1, 1), prompt_tokens=suffix_tokens),
        vocab,
        seed=seed + 3,
    )
    sessions: list[list[SessionTurn]] = []
    k = 0
    for i in range(n_sessions):
        seq = [
            SessionTurn(
                prompt=first[i],
                max_tokens=max_tokens[i % len(max_tokens)],
                think_s=think_s if turns > 1 else 0.0,
            )
        ]
        for t in range(1, turns):
            seq.append(
                SessionTurn(
                    prompt=suffixes[k],
                    max_tokens=max_tokens[(i + t) % len(max_tokens)],
                    think_s=think_s if t + 1 < turns else 0.0,
                )
            )
            k += 1
        sessions.append(seq)
    return sessions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument("--n-values", default="1,2,4,6,8,12")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=9981)
    parser.add_argument("--suffix-tokens", type=int, default=1633)
    parser.add_argument("--think-s", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--slo-ttft-s", type=float, default=None)
    parser.add_argument("--slo-tpot-s", type=float, default=None)
    parser.add_argument(
        "--admit",
        choices=sorted(_ADMIT),
        default="always",
        help="always = Task-2 baseline; session-slo = Task-4 occupancy cap at N=6.",
    )
    parser.add_argument("--kv-budget-bytes", type=int, default=None)
    parser.add_argument(
        "--simulate-card-gib",
        type=float,
        default=24.0,
        help="Cap KV as weights+overhead subtracted from this card size. 24 is the Task-2 primary.",
    )
    parser.add_argument(
        "--no-simulate-card",
        action="store_true",
        help="Use factory 85%%-of-free KV (invalid as a session-capacity headline).",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if not args.model:
        raise SystemExit("need --model")
    if not args.allow_dirty and git_dirty(REPO):
        raise SystemExit("dirty worktree")
    ns = [int(x) for x in args.n_values.split(",") if x.strip()]
    pin_cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    simulate_card_gib: float | None
    if args.no_simulate_card:
        kv_budget = args.kv_budget_bytes
        simulate_card_gib = None
    elif args.kv_budget_bytes is not None:
        kv_budget = args.kv_budget_bytes
        simulate_card_gib = None
    else:
        simulate_card_gib = args.simulate_card_gib
        kv_budget = kv_budget_for_simulated_card(pin_cfg, simulate_card_gib)
    corpus_out = _sample_output_tokens(64, seed=args.seed)
    admit_fn = _ADMIT[args.admit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    def _build():
        return build_engine(
            args.model,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_budget=kv_budget,
        )

    engine = _build()
    vocab = engine.runner.model.cfg.vocab_size
    weight_bytes = estimate_weight_bytes(engine.runner.model.cfg, dtype_bytes=2)
    for n in ns:
        sessions = _make_sessions(
            n,
            turns=args.turns,
            prompt_tokens=args.prompt_tokens,
            suffix_tokens=args.suffix_tokens,
            think_s=args.think_s,
            max_tokens=corpus_out,
            vocab=vocab,
            seed=args.seed,
        )
        for _ in range(args.warmup):
            warm = run_sessions(
                engine,
                sessions,
                profile=False,
                max_active=n,
                admit=admit_fn,
                slo_ttft_s=args.slo_ttft_s,
                slo_tpot_s=args.slo_tpot_s,
            )
            if warm.kv_exhausted:
                engine = _build()
        run = run_sessions(
            engine,
            sessions,
            profile=False,
            max_active=n,
            admit=admit_fn,
            slo_ttft_s=args.slo_ttft_s,
            slo_tpot_s=args.slo_tpot_s,
        )
        metrics = summarize_traces(run.traces, run.wall_s)
        ttfts = [t.ttft_s for t in run.traces if t.ttft_s is not None]
        tpots = [t.tpot_s for t in run.traces if t.tpot_s is not None]
        rec = {
            "schema_version": 1,
            "engine": "qwen3-runtime",
            "case": f"session_capacity_n{n}",
            "seed": args.seed,
            "workload": {
                "name": "session_capacity",
                "n_sessions": n,
                "turns": args.turns,
                "prompt_tokens": args.prompt_tokens,
                "suffix_tokens": args.suffix_tokens,
                "think_s": args.think_s,
                "sync_policy": "serving",
                "kv_held_during_think": True,
                "warmup": args.warmup,
                "max_num_seqs": args.max_num_seqs,
                "admit": args.admit,
                "slo_ttft_s": args.slo_ttft_s,
                "slo_tpot_s": args.slo_tpot_s,
            },
            "engine_config": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "num_kv_blocks": engine.config.num_kv_blocks,
                "block_size": engine.config.block_size,
                "kv_budget_bytes_requested": kv_budget,
                "kv_budget_bytes": engine.config.num_kv_blocks
                * bytes_per_block(engine.runner.model.cfg, engine.config.block_size, dtype_bytes=2),
                "simulate_card_gib": simulate_card_gib,
                "weight_bytes": weight_bytes,
                "overhead_gib": None if simulate_card_gib is None else 2.0,
            },
            "environment": capture(REPO, ["python", "-m", "bench.run_session_capacity", "--n-values", str(n)]),
            "forced_length_ok": bool(metrics["forced_length_ok"]),
            "metrics": metrics,
            "accounting": _accounting(run),
            "backlog": _backlog_payload(run),
            "per_request": compact_traces(run.traces),
            "ttft_s_p50": percentile(ttfts, 0.50),
            "ttft_s_p99": percentile(ttfts, 0.99),
            "tpot_s_p50": percentile(tpots, 0.50),
            "tpot_s_p99": percentile(tpots, 0.99),
            "num_preemptions": run.num_preemptions,
            "kv_exhausted": run.kv_exhausted,
        }
        rec["environment"]["dtype"] = "bf16"
        (args.out_dir / f"n_{n}.json").write_text(json.dumps(rec, indent=2) + "\n")
        row = {
            "n_sessions": n,
            "wall_s": run.wall_s,
            "offered": run.offered,
            "admitted": run.admitted,
            "rejected": run.rejected,
            "completed": run.completed,
            "slo_met": run.slo_met,
            "goodput_per_s": run.goodput_per_s,
            "ttft_p50": rec["ttft_s_p50"],
            "ttft_p99": rec["ttft_s_p99"],
            "tpot_p50": rec["tpot_s_p50"],
            "tpot_p99": rec["tpot_s_p99"],
            "backlog_slope_per_s": run.backlog_slope_per_s,
            "backlog_depth_max": rec["backlog"]["depth_max"],
            "forced_length_ok": rec["forced_length_ok"],
            "num_preemptions": run.num_preemptions,
            "kv_exhausted": run.kv_exhausted,
            "dirty": rec["environment"]["dirty"],
        }
        summary.append(row)
        print(json.dumps(row), flush=True)
        if run.kv_exhausted:
            engine = _build()
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
