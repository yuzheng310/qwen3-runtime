"""vLLM request/session capacity at the frozen SLO. Isolated venv only.

Fairness (also written into each JSON):

- dtype BF16, max_num_seqs=8, max_num_batched_tokens=2048, same as ours.
- 24 GiB primary: kv_cache_memory_bytes = kv_budget_for_simulated_card(24).
- Request: prefix cache OFF (independent Poisson prompts; matches ours).
- Session: prefix cache ON, later turns send the full growing prompt so APC
  can hit. Ours instead holds KV and sends only the suffix. Reported, not hidden.
- Admission: vLLM always-admit. Ours Task-4 `--admit request-slo/session-slo`
  is a separate curve. Compare always-admit to always-admit unless stated.
- TTFT from our arrival timestamp to the first streamed output token (host
  clock), not vLLM metrics.arrival_time.
- Sampling: temperature 0, ignore_eos, per-request max_tokens from the 494-trace
  corpus, same seed as ours.

Run with the Python interpreter from the pinned vLLM environment:
`python -m bench.run_vllm_capacity`.
Do not import this module into the conda 2.8 + FlashInfer 0.6.17 path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from bench.env import capture, git_dirty
from bench.metrics import compact_traces, length_split_metrics, percentile, summarize_traces
from bench.run_slo_harness import _accounting, _backlog_payload, _sample_output_tokens
from bench.run_session_capacity import _make_sessions
from bench.vllm_slo import note_new_tokens
from bench.workloads import SEED, TTFT_LENGTHS, make_prompts, mix_length_prompts, poisson_arrivals
from qwen3_runtime.engine.factory import PIN
from qwen3_runtime.engine.memory import kv_budget_for_simulated_card
from qwen3_runtime.serving.slo_harness import (
    BacklogSample,
    RequestTrace,
    ServeRun,
    _meets_slo,
    backlog_slope,
)
from qwen3_runtime.models.qwen3 import Qwen3ModelConfig

REPO = Path(__file__).resolve().parents[1]


def _sampling(max_tokens: int):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    kwargs = dict(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("detokenize", None)
        kwargs.pop("output_kind", None)
        return SamplingParams(**kwargs)


def _engine_args(
    model: str,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enable_prefix_caching: bool,
    kv_cache_memory_bytes: int | None,
    max_model_len: int,
):
    from vllm.engine.arg_utils import AsyncEngineArgs

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.90,
    )
    if kv_cache_memory_bytes is not None:
        kwargs["kv_cache_memory_bytes"] = int(kv_cache_memory_bytes)
    return AsyncEngineArgs(**kwargs)


async def _consume_generate(engine, prompt: list[int], max_tokens: int, request_id: str, trace: RequestTrace) -> None:
    from vllm.inputs import TokensPrompt

    stream = engine.generate(
        TokensPrompt(prompt_token_ids=prompt),
        _sampling(max_tokens),
        request_id=request_id,
    )
    async for out in stream:
        ids = list(out.outputs[0].token_ids)
        note_new_tokens(trace, ids, time.perf_counter())


def _serve_run(traces: list[RequestTrace], backlog: list[BacklogSample], origin: float, *, slo_ttft_s, slo_tpot_s) -> ServeRun:
    wall_s = time.perf_counter() - origin
    offered = len(traces)
    admitted = sum(1 for t in traces if t.admitted)
    rejected = offered - admitted
    completed = sum(1 for t in traces if t.admitted and len(t.tokens) == t.max_tokens)
    slo_applied = slo_ttft_s is not None or slo_tpot_s is not None
    slo_met = (
        sum(1 for t in traces if _meets_slo(t, slo_ttft_s=slo_ttft_s, slo_tpot_s=slo_tpot_s))
        if slo_applied
        else None
    )
    return ServeRun(
        traces=traces,
        offered=offered,
        admitted=admitted,
        rejected=rejected,
        completed=completed,
        slo_met=slo_met,
        backlog=backlog,
        wall_s=wall_s,
        origin_s=origin,
        sync_policy="serving",
    )


async def _poisson(
    engine,
    prompts: list[list[int]],
    max_tokens: list[int],
    arrivals_s: list[float],
    *,
    slo_ttft_s: float | None,
    slo_tpot_s: float | None,
) -> ServeRun:
    origin = time.perf_counter()
    traces = [
        RequestTrace(
            request_id=i,
            arrival_s=origin + arrivals_s[i],
            prompt_len=len(prompts[i]),
            max_tokens=max_tokens[i],
            admitted=True,
        )
        for i in range(len(prompts))
    ]
    inflight = {"n": 0}
    backlog: list[BacklogSample] = []
    stop = asyncio.Event()

    async def _sampler() -> None:
        while not stop.is_set():
            backlog.append(
                BacklogSample(
                    t_s=time.perf_counter() - origin,
                    waiting=0,
                    running=inflight["n"],
                )
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass

    async def _one(i: int) -> None:
        delay = traces[i].arrival_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        inflight["n"] += 1
        try:
            await _consume_generate(engine, prompts[i], max_tokens[i], f"r{i}", traces[i])
        finally:
            inflight["n"] -= 1

    sampler = asyncio.create_task(_sampler())
    try:
        await asyncio.gather(*[_one(i) for i in range(len(prompts))])
    finally:
        stop.set()
        await sampler
    return _serve_run(traces, backlog, origin, slo_ttft_s=slo_ttft_s, slo_tpot_s=slo_tpot_s)


async def _sequential_length(engine, prompts: list[list[int]], max_tokens: int) -> list[RequestTrace]:
    traces: list[RequestTrace] = []
    for i, prompt in enumerate(prompts):
        arrival = time.perf_counter()
        trace = RequestTrace(
            request_id=i,
            arrival_s=arrival,
            prompt_len=len(prompt),
            max_tokens=max_tokens,
            admitted=True,
            turn_id=0,
        )
        await _consume_generate(engine, prompt, max_tokens, f"len{i}", trace)
        traces.append(trace)
    return traces


async def _sessions(
    engine,
    sessions,
    *,
    enable_prefix_caching: bool,
    slo_ttft_s: float | None,
    slo_tpot_s: float | None,
) -> ServeRun:
    """N concurrent sessions. Later turns append suffix to the running prefix.

    APC (when enabled) sees the full prompt; ours holds KV and sends suffix only.
    ``max_model_len`` must cover first prompt + later suffixes + generated tokens
    (p50 peak in this load is ~16–18k; the Qwen3-4B pin is 40960). 16384 is too
    small and rejects later turns.
    """
    origin = time.perf_counter()
    traces: list[RequestTrace] = []
    inflight = {"n": 0}
    backlog: list[BacklogSample] = []
    stop = asyncio.Event()

    async def _sampler() -> None:
        while not stop.is_set():
            backlog.append(
                BacklogSample(
                    t_s=time.perf_counter() - origin,
                    waiting=0,
                    running=inflight["n"],
                )
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass

    async def _one_session(sess, sid: int) -> None:
        prefix: list[int] = []
        for t_i, turn in enumerate(sess):
            prompt = list(turn.prompt) if t_i == 0 else prefix + list(turn.prompt)
            arrival = time.perf_counter()
            trace = RequestTrace(
                request_id=sid * 100 + t_i,
                arrival_s=arrival,
                prompt_len=len(prompt),
                max_tokens=turn.max_tokens,
                admitted=True,
                turn_id=t_i,
            )
            traces.append(trace)
            inflight["n"] += 1
            try:
                await _consume_generate(engine, prompt, turn.max_tokens, f"s{sid}t{t_i}", trace)
            finally:
                inflight["n"] -= 1
            prefix = prompt + list(trace.tokens)
            if turn.think_s > 0:
                await asyncio.sleep(turn.think_s)

    sampler = asyncio.create_task(_sampler())
    try:
        await asyncio.gather(*[_one_session(s, i) for i, s in enumerate(sessions)])
    finally:
        stop.set()
        await sampler
    return _serve_run(traces, backlog, origin, slo_ttft_s=slo_ttft_s, slo_tpot_s=slo_tpot_s)


def _record(out: Path, rec: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")


def _payload(run: ServeRun, *, case: str, seed: int, workload: dict, fairness: dict, command: list[str]) -> dict:
    ttfts = [t.ttft_s for t in run.traces if t.ttft_s is not None]
    tpots = [t.tpot_s for t in run.traces if t.tpot_s is not None]
    metrics = summarize_traces(run.traces, run.wall_s)
    rec = {
        "schema_version": 1,
        "engine": "vllm",
        "case": case,
        "seed": seed,
        "workload": workload,
        "fairness": fairness,
        "environment": capture(REPO, command),
        "forced_length_ok": bool(metrics["forced_length_ok"]),
        "metrics": metrics,
        "accounting": _accounting(run),
        "backlog": _backlog_payload(run),
        "per_request": compact_traces(run.traces),
        "length_split": length_split_metrics(run.traces),
        "ttft_s_p50": percentile(ttfts, 0.50),
        "ttft_s_p99": percentile(ttfts, 0.99),
        "tpot_s_p50": percentile(tpots, 0.50),
        "tpot_s_p99": percentile(tpots, 0.99),
        "backlog_slope_per_s": backlog_slope(run.backlog),
    }
    rec["environment"]["dtype"] = "bf16"
    rec["environment"]["vllm"] = rec["environment"].get("vllm")
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("request", "session", "length"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument("--rates", default="2,4,6,8,10,12,16")
    parser.add_argument("--n-values", default="1,2,4,6,8,12")
    parser.add_argument("--n-requests", type=int, default=96)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=None, help="Constant decode length. Default: CodeScout corpus sample.")
    parser.add_argument("--mix-short", type=int, default=None)
    parser.add_argument("--mix-long", type=int, default=None)
    parser.add_argument("--mix-long-frac", type=float, default=None)
    parser.add_argument("--lengths", default=",".join(str(x) for x in TTFT_LENGTHS))
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--session-prompt-tokens", type=int, default=9981)
    parser.add_argument("--suffix-tokens", type=int, default=1633)
    parser.add_argument("--think-s", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help=(
            "Replay count before the measured run. Session mode default APC is ON and "
            "warmup reuses the same `sessions` object, so warmup>=1 makes measured "
            "first turns prefix-cache hits. Use --warmup 0 for a cold first-turn TTFT."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--slo-ttft-s", type=float, default=4.5)
    parser.add_argument("--slo-tpot-s", type=float, default=0.1)
    parser.add_argument("--simulate-card-gib", type=float, default=24.0)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if not args.model:
        raise SystemExit("need --model")
    if not args.allow_dirty and git_dirty(REPO):
        raise SystemExit("dirty worktree")
    pin_cfg = Qwen3ModelConfig.from_hf_config(json.loads(PIN.read_text()))
    kv_bytes = kv_budget_for_simulated_card(pin_cfg, args.simulate_card_gib)
    if args.prefix_cache is None:
        enable_pc = args.mode == "session"
    else:
        enable_pc = args.prefix_cache

    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(
        _engine_args(
            args.model,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=enable_pc,
            kv_cache_memory_bytes=kv_bytes,
            max_model_len=args.max_model_len,
        )
    )
    fairness = {
        "dtype": "bf16",
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_bytes": kv_bytes,
        "simulate_card_gib": args.simulate_card_gib,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": enable_pc,
        "admission": "vllm-always-admit",
        "ttft_clock": "host_arrival_to_first_streamed_token",
        "session_later_turn": (
            "full_prompt_plus_suffix_for_apc" if args.mode == "session" else None
        ),
        "ours_hold_kv_suffix_only": args.mode == "session",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    async def _run() -> None:
        if args.mode == "request":
            vocab = 151936
            if args.mix_long_frac is not None:
                if args.mix_short is None or args.mix_long is None:
                    raise SystemExit("--mix-long-frac requires --mix-short and --mix-long")
                prompts = mix_length_prompts(
                    args.n_requests,
                    short=args.mix_short,
                    long=args.mix_long,
                    long_frac=args.mix_long_frac,
                    vocab_size=vocab,
                    seed=args.seed,
                )
                prompt_tokens = f"{int(round((1 - args.mix_long_frac) * 100))}pct_{args.mix_short}_{int(round(args.mix_long_frac * 100))}pct_{args.mix_long}"
            else:
                prompts = make_prompts(
                    SimpleNamespace(n_requests=args.n_requests, prompt_tokens=args.prompt_tokens),
                    vocab,
                    seed=args.seed,
                )
                prompt_tokens = args.prompt_tokens
            if args.output_tokens is not None:
                max_tokens = [args.output_tokens] * args.n_requests
                output_desc: int | str = args.output_tokens
            else:
                max_tokens = _sample_output_tokens(args.n_requests, seed=args.seed)
                output_desc = "recorded_output_tokens@token_reconstruction_v1"
            for rate in [float(x) for x in args.rates.split(",") if x.strip()]:
                arrivals = poisson_arrivals(args.n_requests, rate, seed=args.seed)
                for _ in range(args.warmup):
                    await _poisson(
                        engine,
                        prompts,
                        max_tokens,
                        arrivals,
                        slo_ttft_s=args.slo_ttft_s,
                        slo_tpot_s=args.slo_tpot_s,
                    )
                run = await _poisson(
                    engine,
                    prompts,
                    max_tokens,
                    arrivals,
                    slo_ttft_s=args.slo_ttft_s,
                    slo_tpot_s=args.slo_tpot_s,
                )
                rec = _payload(
                    run,
                    case=f"vllm_request_capacity_l{rate:g}",
                    seed=args.seed,
                    workload={
                        "name": "request_capacity",
                        "arrival_process": "poisson",
                        "poisson_rate": rate,
                        "n_requests": args.n_requests,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_desc,
                        "sync_policy": "serving",
                        "warmup": args.warmup,
                        "max_num_seqs": args.max_num_seqs,
                        "admit": "vllm-always-admit",
                        "slo_ttft_s": args.slo_ttft_s,
                        "slo_tpot_s": args.slo_tpot_s,
                    },
                    fairness=fairness,
                    command=["python", "-m", "bench.run_vllm_capacity", "--mode", "request", "--rates", str(rate)],
                )
                _record(args.out_dir / f"lambda_{rate:g}.json", rec)
                row = {
                    "lambda": rate,
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
                    "forced_length_ok": rec["forced_length_ok"],
                    "dirty": rec["environment"]["dirty"],
                }
                summary.append(row)
                print(json.dumps(row), flush=True)
        elif args.mode == "length":
            vocab = 151936
            lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
            for i, n_tok in enumerate(lengths):
                n_total = args.warmup + args.reps
                prompts = make_prompts(
                    SimpleNamespace(n_requests=n_total, prompt_tokens=n_tok),
                    vocab,
                    seed=args.seed + i * 17,
                )
                if args.warmup:
                    await _sequential_length(engine, prompts[: args.warmup], args.decode_tokens)
                traces = await _sequential_length(engine, prompts[args.warmup :], args.decode_tokens)
                ttfts = [t.ttft_s for t in traces if t.ttft_s is not None]
                rec = {
                    "schema_version": 1,
                    "engine": "vllm",
                    "case": f"vllm_ttft_vs_length_{n_tok}",
                    "seed": args.seed,
                    "workload": {
                        "name": "ttft_vs_length",
                        "prompt_tokens": n_tok,
                        "max_tokens": args.decode_tokens,
                        "reps": args.reps,
                        "warmup": args.warmup,
                        "n_sessions": 1,
                        "prefix": False,
                        "queue": False,
                        "max_num_batched_tokens": args.max_num_batched_tokens,
                    },
                    "fairness": fairness,
                    "environment": capture(
                        REPO,
                        ["python", "-m", "bench.run_vllm_capacity", "--mode", "length", "--lengths", str(n_tok)],
                    ),
                    "ttft_s_p50": percentile(ttfts, 0.50),
                    "ttft_s_p99": percentile(ttfts, 0.99),
                    "ttft_s_min": min(ttfts) if ttfts else None,
                    "ttft_s_max": max(ttfts) if ttfts else None,
                    "per_request": compact_traces(traces),
                }
                rec["environment"]["dtype"] = "bf16"
                _record(args.out_dir / f"len_{n_tok}.json", rec)
                row = {
                    "prompt_tokens": n_tok,
                    "n": len(ttfts),
                    "ttft_p50": rec["ttft_s_p50"],
                    "ttft_p99": rec["ttft_s_p99"],
                    "ttft_min": rec["ttft_s_min"],
                    "ttft_max": rec["ttft_s_max"],
                    "dirty": rec["environment"]["dirty"],
                }
                summary.append(row)
                print(json.dumps(row), flush=True)
        else:
            vocab = 151936
            corpus_out = _sample_output_tokens(64, seed=args.seed)
            for n in [int(x) for x in args.n_values.split(",") if x.strip()]:
                sessions = _make_sessions(
                    n,
                    turns=args.turns,
                    prompt_tokens=args.session_prompt_tokens,
                    suffix_tokens=args.suffix_tokens,
                    think_s=args.think_s,
                    max_tokens=corpus_out,
                    vocab=vocab,
                    seed=args.seed,
                )
                for _ in range(args.warmup):
                    await _sessions(
                        engine,
                        sessions,
                        enable_prefix_caching=enable_pc,
                        slo_ttft_s=args.slo_ttft_s,
                        slo_tpot_s=args.slo_tpot_s,
                    )
                run = await _sessions(
                    engine,
                    sessions,
                    enable_prefix_caching=enable_pc,
                    slo_ttft_s=args.slo_ttft_s,
                    slo_tpot_s=args.slo_tpot_s,
                )
                rec = _payload(
                    run,
                    case=f"vllm_session_capacity_n{n}",
                    seed=args.seed,
                    workload={
                        "name": "session_capacity",
                        "n_sessions": n,
                        "turns": args.turns,
                        "prompt_tokens": args.session_prompt_tokens,
                        "suffix_tokens": args.suffix_tokens,
                        "think_s": args.think_s,
                        "sync_policy": "serving",
                        "kv_held_during_think": False,
                        "prefix_cache": enable_pc,
                        "warmup": args.warmup,
                        "max_num_seqs": args.max_num_seqs,
                        "admit": "vllm-always-admit",
                        "slo_ttft_s": args.slo_ttft_s,
                        "slo_tpot_s": args.slo_tpot_s,
                    },
                    fairness=fairness,
                    command=["python", "-m", "bench.run_vllm_capacity", "--mode", "session", "--n-values", str(n)],
                )
                rec["num_preemptions"] = None
                _record(args.out_dir / f"n_{n}.json", rec)
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
                    "forced_length_ok": rec["forced_length_ok"],
                    "dirty": rec["environment"]["dirty"],
                }
                summary.append(row)
                print(json.dumps(row), flush=True)

    asyncio.run(_run())
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
