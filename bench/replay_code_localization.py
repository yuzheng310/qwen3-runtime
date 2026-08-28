"""Trace Replay for code-localization-trace-v1 / replay-subset-v1.

Does not implement an optimization. Loads CodeScout-4B through the Instruct-2507
pin (rope_theta=5e6). Both engines receive reconstructed token IDs; decode
length is the recorded completion_tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from workloads.code_localization.reconstruct_tokens import reconstruct_task


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "workloads" / "code_localization" / "trace_v1.jsonl"
SUBSET = ROOT / "workloads" / "code_localization" / "replay_subset_v1.json"
ROLLOUTS = ROOT / "workloads" / "code_localization" / "raw" / "swe_bench_verified.parquet"
TOKENIZER = ROOT / "workloads" / "code_localization" / "tokenizer"
CODESCOUT_PIN = ROOT / "docs" / "pins" / "CodeScout-4B" / "config.json"
MAX_CONTEXT = 65536
HISTORICAL_IDS = ROOT / "workloads" / "code_localization" / "token_ids" / "replay_subset_v1.jsonl"
SESSION_IDS = ROOT / "workloads" / "code_localization" / "token_ids" / "replay_subset_v1_session.jsonl"
CODESCOUT_REPO = ROOT.parent / "codescout"


def load_subset_ids(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return list(data["task_ids"])


def dump_items(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def load_dumped_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def apply_item_limits(
    items: list[dict], *, limit_requests: int = 0, limit_tasks: int = 0
) -> list[dict]:
    """Slice dumped IDs. Tasks keep every turn of the selected sessions."""
    if limit_tasks:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            tid = item["task_id"]
            if tid in seen:
                continue
            seen.add(tid)
            ordered.append(tid)
            if len(ordered) >= limit_tasks:
                break
        keep = set(ordered)
        items = [item for item in items if item["task_id"] in keep]
    if limit_requests:
        items = items[:limit_requests]
    return items


def parse_speculative_config(raw: str | None) -> dict | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("--speculative-config must be a JSON object")
    return data


def _vllm_spec_metrics(out) -> dict:
    """Any numeric/container attr whose name looks like spec-decode accounting."""
    found: dict = {}
    objs = [out, getattr(out, "metrics", None)]
    outputs = getattr(out, "outputs", None) or []
    if outputs:
        objs.append(outputs[0])
    needles = ("spec", "accept", "draft", "ngram", "suffix")
    for obj in objs:
        if obj is None:
            continue
        prefix = type(obj).__name__
        for name in dir(obj):
            if name.startswith("_"):
                continue
            if not any(n in name.lower() for n in needles):
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            if callable(val):
                continue
            if isinstance(val, (int, float, bool, str, list, dict, type(None))):
                found[f"{prefix}.{name}"] = val
    return found


def reconstruct_items(task_ids: list[str] | None) -> list[dict]:
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        TOKENIZER, trust_remote_code=False, local_files_only=True
    )
    wanted = set(task_ids) if task_ids is not None else None
    items: list[dict] = []
    for row in pq.read_table(ROLLOUTS).to_pylist():
        tid = row["instance_id"]
        if wanted is not None and tid not in wanted:
            continue
        report = reconstruct_task(tokenizer, row, keep_ids=True)
        for req in report["requests"]:
            items.append(
                {
                    "task_id": tid,
                    "turn_id": req["turn_id"],
                    "input_ids": req["input_ids"],
                    "output_ids": req["output_ids"],
                    "max_tokens": req["recorded_output_tokens"],
                    "reconstructed_input_tokens": req["reconstructed_input_tokens"],
                    "recorded_input_tokens": req["recorded_input_tokens"],
                    "output_count_match": req["output_count_match"],
                    "generation_prompt_stable": req["generation_prompt_stable"],
                }
            )
    if wanted is not None:
        have = {i["task_id"] for i in items}
        missing = [t for t in task_ids or [] if t not in have]
        if missing:
            raise SystemExit(f"subset tasks missing from rollouts: {missing[:8]}")
        order = {tid: i for i, tid in enumerate(task_ids or [])}
        items.sort(key=lambda it: (order[it["task_id"]], it["turn_id"]))
    return items


def _phase_metrics(traces) -> dict:
    from bench.metrics import percentile

    prefills = [t.ttft_s for t in traces if t.ttft_s is not None]
    decodes = []
    for t in traces:
        if t.first_token_s is not None and t.last_token_s is not None:
            decodes.append(t.last_token_s - t.first_token_s)
    walls = []
    for t in traces:
        if t.last_token_s is not None:
            walls.append(t.last_token_s - t.arrival_s)
    prefill_sum = sum(prefills)
    decode_sum = sum(decodes)
    wall_sum = sum(walls)
    return {
        "prefill_s": {
            "sum": prefill_sum,
            "p50": percentile(prefills, 0.50),
            "p95": percentile(prefills, 0.95),
        },
        "decode_s": {
            "sum": decode_sum,
            "p50": percentile(decodes, 0.50),
            "p95": percentile(decodes, 0.95),
        },
        "request_latency_s": {
            "p50": percentile(walls, 0.50),
            "p95": percentile(walls, 0.95),
        },
        "prefill_fraction_of_measured_request": (
            prefill_sum / wall_sum if wall_sum else None
        ),
        "decode_fraction_of_measured_request": (
            decode_sum / wall_sum if wall_sum else None
        ),
    }


def replay_ours(items: list[dict], model: Path, *, nvtx: bool, warmup: int = 1) -> tuple[list, float]:
    from qwen3_runtime.engine.factory import build_engine
    from qwen3_runtime.engine.serve import run_sequential_requests

    engine = build_engine(
        model,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        pin_path=CODESCOUT_PIN,
    )
    if warmup and items:
        run_sequential_requests(
            engine,
            [(items[0]["input_ids"], min(8, max(1, items[0]["max_tokens"])))],
            ignore_eos=True,
            nvtx=nvtx,
        )
    pairs = [(it["input_ids"], it["max_tokens"]) for it in items]
    t0 = time.perf_counter()
    traces = run_sequential_requests(engine, pairs, ignore_eos=True, nvtx=nvtx)
    return traces, time.perf_counter() - t0


def _vllm_cached_tokens(out) -> int | None:
    """APC hit size if vLLM names it. Does not use num_computed_tokens (ambiguous)."""
    for obj in (out, getattr(out, "outputs", [None])[0], getattr(out, "metrics", None)):
        if obj is None:
            continue
        for name in ("num_cached_tokens", "cached_tokens"):
            val = getattr(obj, name, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
    return None


def _vllm_metric_keys(out) -> list[str]:
    keys: list[str] = []
    for obj in (out, getattr(out, "outputs", [None])[0], getattr(out, "metrics", None)):
        if obj is None:
            continue
        keys.append(type(obj).__name__ + ":" + ",".join(sorted(n for n in dir(obj) if not n.startswith("_"))))
    return keys


def replay_vllm(
    items: list[dict],
    model: Path,
    *,
    warmup: int = 1,
    enable_prefix_caching: bool = False,
    speculative_config: dict | None = None,
    kv_cache_dtype: str | None = None,
) -> tuple[list, float, dict]:
    from bench.drivers import RequestTrace, vllm_first_token_s
    from vllm import LLM, SamplingParams

    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        TokensPrompt = None  # type: ignore[misc, assignment]

    kwargs = {
        "model": str(model),
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": MAX_CONTEXT,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": enable_prefix_caching,
        "max_num_batched_tokens": 2048,
    }
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    extra: dict = {
        "enable_prefix_caching": enable_prefix_caching,
        "speculative_config": speculative_config,
        "kv_cache_dtype": kv_cache_dtype,
        "vllm_kwargs": dict(kwargs),
        "cached_tokens_per_request": [],
        "request_wall_s": [],
        "spec_metrics_per_request": [],
        "peak_memory_allocated_bytes": None,
    }
    try:
        llm = LLM(skip_tokenizer_init=True, **kwargs)
    except TypeError as exc:
        if speculative_config is not None or kv_cache_dtype:
            raise TypeError(
                f"LLM() rejected speculative_config/kv_cache_dtype kwargs: {exc}; "
                f"keys={sorted(kwargs)}"
            ) from exc
        kwargs.pop("enable_prefix_caching", None)
        extra["enable_prefix_caching_kwarg_accepted"] = False
        llm = LLM(skip_tokenizer_init=True, **kwargs)
    else:
        extra["enable_prefix_caching_kwarg_accepted"] = True

    if warmup and items:
        warm = SamplingParams(temperature=0.0, max_tokens=min(8, max(1, items[0]["max_tokens"])), ignore_eos=True)
        if TokensPrompt is not None:
            llm.generate([TokensPrompt(prompt_token_ids=items[0]["input_ids"])], warm)
        else:
            llm.generate(prompt_token_ids=[items[0]["input_ids"]], sampling_params=warm)

    traces = []
    t0 = time.perf_counter()
    for i, item in enumerate(items):
        params = SamplingParams(
            temperature=0.0, max_tokens=item["max_tokens"], ignore_eos=True
        )
        req_t0 = time.perf_counter()
        if TokensPrompt is not None:
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=item["input_ids"])], params
            )
        else:
            outputs = llm.generate(
                prompt_token_ids=[item["input_ids"]], sampling_params=params
            )
        out = outputs[0]
        ids = list(out.outputs[0].token_ids)
        metrics = getattr(out, "metrics", None)
        first = vllm_first_token_s(metrics)
        arrival = getattr(metrics, "arrival_time", req_t0) if metrics else req_t0
        extra["cached_tokens_per_request"].append(_vllm_cached_tokens(out))
        extra["request_wall_s"].append(time.perf_counter() - req_t0)
        extra["spec_metrics_per_request"].append(_vllm_spec_metrics(out))
        if i == 0:
            extra["vllm_output_metric_keys"] = _vllm_metric_keys(out)
        traces.append(
            RequestTrace(
                request_id=i,
                arrival_s=float(arrival) if arrival is not None else req_t0,
                prompt_len=len(item["input_ids"]),
                max_tokens=item["max_tokens"],
                tokens=ids,
                first_token_s=first,
                itl_s=[],
                last_token_s=time.perf_counter(),
            )
        )
    try:
        import torch

        if torch.cuda.is_available():
            extra["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception:
        extra["peak_memory_allocated_bytes"] = None
    return traces, time.perf_counter() - t0, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("qwen3-runtime", "vllm"))
    parser.add_argument("--model", type=Path, help="local CodeScout-4B directory")
    parser.add_argument("--subset", type=Path, default=SUBSET)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N reconstructed requests")
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=0,
        help="first N unique task_ids in dump/subset order (keeps all turns of those tasks)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--nvtx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ids",
        type=Path,
        default=HISTORICAL_IDS,
        help="pre-dumped reconstructed token IDs (written if missing)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument(
        "--session-kv",
        action="store_true",
        help="qwen3-runtime: persistent session KV + teacher-forced output_ids. "
        "Does not overwrite historical full-reprefill JSON. Uses SESSION_IDS dump.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="vLLM Automatic Prefix Caching. Default off (fair Stage-2). Stage-2.5 APC ON uses --enable-prefix-caching.",
    )
    parser.add_argument(
        "--speculative-config",
        default=None,
        help="JSON object passed verbatim as LLM(speculative_config=...). vLLM 0.27 EngineArgs dict.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="Optional vLLM kv_cache_dtype (e.g. fp8). Not token-identical.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=0,
        help="qwen3-runtime n-gram speculative window. 0 disables.",
    )
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    args = parser.parse_args()

    if not CODESCOUT_PIN.exists():
        raise SystemExit(f"missing CodeScout pin {CODESCOUT_PIN}")
    if not TOKENIZER.exists():
        raise SystemExit(f"missing tokenizer dir {TOKENIZER}")
    if not ROLLOUTS.exists():
        raise SystemExit(f"missing rollouts parquet {ROLLOUTS}")

    task_ids = None if args.all_tasks else load_subset_ids(args.subset)
    ids_path = args.ids
    if args.session_kv and args.ids == HISTORICAL_IDS:
        ids_path = SESSION_IDS
    if ids_path.exists() and not args.all_tasks:
        items = load_dumped_items(ids_path)
        if task_ids is not None:
            wanted = set(task_ids)
            items = [it for it in items if it["task_id"] in wanted]
        if args.session_kv and any(not it.get("output_ids") for it in items):
            items = reconstruct_items(task_ids)
            dump_items(ids_path, items)
    else:
        items = reconstruct_items(task_ids)
        if not args.all_tasks:
            dump_items(ids_path, items)
    items = apply_item_limits(
        items, limit_requests=int(args.limit or 0), limit_tasks=int(args.limit_tasks or 0)
    )
    if args.dump_only:
        print(f"dumped {len(items)} requests -> {ids_path}")
        return
    if args.engine is None or args.model is None or args.out is None:
        raise SystemExit("--engine, --model, and --out are required unless --dump-only")
    print(f"replay {len(items)} requests engine={args.engine} model={args.model}")

    apc_extra: dict = {}
    session_extra: dict = {}
    if args.engine == "qwen3-runtime" and args.session_kv:
        from bench.session_kv_replay import replay_ours_session_kv

        traces, wall, session_extra = replay_ours_session_kv(
            items,
            args.model,
            nvtx=args.nvtx,
            warmup=args.warmup,
            pin_path=CODESCOUT_PIN,
            num_speculative_tokens=args.num_speculative_tokens,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
        )
        prefix_cache = False
    elif args.engine == "qwen3-runtime":
        traces, wall = replay_ours(items, args.model, nvtx=args.nvtx, warmup=args.warmup)
        prefix_cache = False
    else:
        traces, wall, apc_extra = replay_vllm(
            items,
            args.model,
            warmup=args.warmup,
            enable_prefix_caching=bool(args.enable_prefix_caching),
            speculative_config=parse_speculative_config(args.speculative_config),
            kv_cache_dtype=args.kv_cache_dtype,
        )
        prefix_cache = bool(args.enable_prefix_caching)
        if args.session_kv:
            session_extra = {
                "execution_model": "vllm_full_prompt",
                "note": "vLLM has no hold_kv; closest reuse is enable_prefix_caching",
            }

    from bench.env import capture, git_commit, git_dirty
    from bench.metrics import percentile, summarize_traces
    import hashlib

    def _sha(path: Path) -> str | None:
        if not path.exists():
            return None
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    summary = summarize_traces(traces, wall)
    if args.session_kv and session_extra.get("source_forced_length_ok") is not True:
        summary["forced_length_ok"] = False
    req_walls = apc_extra.get("request_wall_s") or []
    if not req_walls:
        req_walls = [
            (tr.last_token_s - tr.arrival_s)
            if tr.last_token_s is not None
            else None
            for tr in traces
        ]
    cached = apc_extra.get("cached_tokens_per_request") or []
    spec_rows = apc_extra.get("spec_metrics_per_request") or []
    n_tasks = len({it["task_id"] for it in items})
    seconds_per_task = wall / n_tasks if n_tasks else None
    tasks_per_gpu_hour = (3600.0 / seconds_per_task) if seconds_per_task else None
    ttfts = [t.ttft_s for t in traces if t.ttft_s is not None]
    tpots = [t.tpot_s for t in traces if t.tpot_s is not None]
    codescout_env = None
    if CODESCOUT_REPO.is_dir():
        codescout_env = {
            "git_commit": git_commit(CODESCOUT_REPO),
            "dirty": git_dirty(CODESCOUT_REPO),
        }
    payload = {
        "schema_version": 1,
        "mode": "trace_replay_session_kv" if args.session_kv else "trace_replay",
        "case": "code-localization-replay",
        "seed": None,
        "engine": args.engine,
        "model_pin": "OpenHands/CodeScout-4B@eb233041350295c9ba74de5c6777e15fca0c8ddc",
        "codescout_pin": str(CODESCOUT_PIN.relative_to(ROOT)),
        "codescout_source": codescout_env,
        "session_kv": bool(args.session_kv),
        "n_requests": len(items),
        "n_tasks": n_tasks,
        "subset": None if args.all_tasks else str(args.subset),
        "limit_requests": int(args.limit or 0) or None,
        "limit_tasks": int(args.limit_tasks or 0) or None,
        "nvtx": args.nvtx if args.engine == "qwen3-runtime" else None,
        "prefix_cache": prefix_cache,
        "chunk_tokens": 2048,
        "warmup": args.warmup,
        "trials": [{"wall_s": wall, "n_requests": len(items)}],
        "workload": {
            "name": "replay_subset_v1" if not args.all_tasks else "code-localization-trace-v1",
            "n_tasks": n_tasks,
            "n_requests": len(items),
            "sequential": True,
            "batch": 1,
        },
        "engine_config": apc_extra.get("vllm_kwargs")
        if apc_extra
        else {
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
            "pin_path": str(CODESCOUT_PIN.relative_to(ROOT)),
            "session_kv": bool(args.session_kv),
            "num_speculative_tokens": args.num_speculative_tokens,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
        },
        "metrics": summary,
        "seconds_per_task": seconds_per_task,
        "tasks_per_gpu_hour": tasks_per_gpu_hour,
        "ttft_percentiles": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "tpot_percentiles": {
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
            "p99": percentile(tpots, 0.99),
        },
        "forced_length_ok": summary.get("forced_length_ok"),
        "phases": _phase_metrics(traces),
        "session_kv_stats": session_extra or None,
        "environment": capture(
            ROOT,
            [sys.executable, "-m", "bench.replay_code_localization", *sys.argv[1:]],
        ),
        "artifact_hashes": {
            "subset_sha256": _sha(args.subset),
            "ids_sha256": _sha(ids_path),
            "historical_ids_sha256": _sha(HISTORICAL_IDS),
        },
        "vllm_apc": {
            k: apc_extra[k]
            for k in (
                "enable_prefix_caching",
                "enable_prefix_caching_kwarg_accepted",
                "speculative_config",
                "kv_cache_dtype",
                "vllm_kwargs",
                "peak_memory_allocated_bytes",
                "vllm_output_metric_keys",
            )
            if k in apc_extra
        }
        if apc_extra
        else None,
        "requests": [
            {
                "task_id": it["task_id"],
                "turn_id": it["turn_id"],
                "prompt_len": it["reconstructed_input_tokens"],
                "appended_len": tr.prompt_len,
                "max_tokens": it["max_tokens"],
                "output_len": len(tr.tokens),
                "output_token_ids": list(tr.tokens),
                "ttft_s": tr.ttft_s,
                "tpot_s": tr.tpot_s,
                "request_wall_s": req_walls[i] if i < len(req_walls) else None,
                "cached_tokens": cached[i] if i < len(cached) else None,
                "spec_metrics": spec_rows[i] if i < len(spec_rows) else None,
            }
            for i, (it, tr) in enumerate(zip(items, traces))
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "metrics": summary,
                "phases": payload["phases"],
                "seconds_per_task": seconds_per_task,
                "tasks_per_gpu_hour": tasks_per_gpu_hour,
                "session_kv_stats": session_extra or None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
