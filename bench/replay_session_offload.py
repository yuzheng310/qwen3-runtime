"""Identical real token trajectories through the EngineDriver and SkyRL adapter.

Forced replay isolates serving work, and is NOT a numerical correctness gate.
Tool delay and concurrent task assignment are explicit synthetic parameters.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.factory import _num_kv_blocks
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.integrations.skyrl.inference_engine import Qwen3InferenceEngine
from qwen3_runtime.utils.loader import load_from_directory


def select_tasks(path, ranks):
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        groups[r["task_id"]].append(r)
    groups = sorted(
        groups.values(),
        key=lambda rs: max(len(r["input_ids"]) + len(r["output_ids"]) for r in rs),
    )
    chosen = [sorted(groups[i], key=lambda r: r["turn_id"]) for i in ranks]
    for turns in chosen:
        for prev, nxt in zip(turns, turns[1:]):
            held = prev["input_ids"] + prev["output_ids"]
            if nxt["input_ids"][: len(held)] != held:
                raise ValueError("trace is not append-only")
    return chosen


def source_digest():
    digest = hashlib.sha256()
    paths = sorted(Path("qwen3_runtime").rglob("*.py")) + [Path(__file__)]
    for path in paths:
        digest.update(
            str(path.relative_to(Path.cwd()) if path.is_absolute() else path).encode()
        )
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_artifact(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


async def replay(
    engine, tasks, *, arm, concurrency, dwell, policy, keep_completed=False
):
    os.environ["QWEN3_SESSION_KV"] = "0" if arm == "apc-only" else "1"
    policy_args = (
        {}
        if policy == "legacy"
        else {
            "session_max_blocks": engine.block_manager.num_blocks,
            "session_max_sessions": 128,
        }
    )
    wrapper = Qwen3InferenceEngine(engine, **policy_args)
    totals = {
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "preemptions": 0,
        "min_free_blocks": engine.block_manager.num_blocks,
        "max_batch": 0,
        "cpu_peak_bytes": 0,
        "capacity_retries": 0,
    }
    original_step = engine.step
    first_at = {}

    def step():
        result = original_step()
        s = engine.last_step_stats or {}
        totals["prefill_tokens"] += s.get("prefill_tokens", 0)
        totals["decode_tokens"] += s.get("decode_tokens", 0)
        totals["preemptions"] += s.get("preempts", 0)
        totals["min_free_blocks"] = min(
            totals["min_free_blocks"], engine.block_manager.num_free_blocks
        )
        now = time.perf_counter()
        for rid, tokens in engine.last_emitted.items():
            if tokens:
                first_at.setdefault(rid, now)
        return result

    engine.step = step
    transfer = {
        "save_s": 0.0,
        "restore_s": 0.0,
        "reserve_s": 0.0,
        "export_s": 0.0,
        "import_s": 0.0,
    }
    manager = engine.session_offload
    operations = [
        (engine.runner.pool, "export_blocks", "export_s"),
        (engine.runner.pool, "import_blocks", "import_s"),
    ]
    if manager.store is not None:
        operations.append((manager.store, "reserve", "reserve_s"))
    for owner, method, key in operations:
        original = getattr(owner, method)

        def measure(*args, _fn=original, _key=key, **kw):
            start = time.perf_counter()
            try:
                return _fn(*args, **kw)
            finally:
                transfer[_key] += time.perf_counter() - start

        setattr(owner, method, measure)
    for method, key in [("save", "save_s"), ("begin_restore", "restore_s")]:
        original = getattr(manager, method)

        def record(*args, _fn=original, _key=key, **kw):
            start = time.perf_counter()
            try:
                return _fn(*args, **kw)
            finally:
                transfer[_key] += time.perf_counter() - start
                totals["cpu_peak_bytes"] = max(
                    totals["cpu_peak_bytes"], manager.report()["cpu_committed_bytes"]
                )

        setattr(manager, method, record)
    semaphore = asyncio.Semaphore(concurrency)
    turns_out = []

    async def task(turns):
        async with semaphore:
            for i, item in enumerate(turns):
                ids = list(item["input_ids"])
                forced = list(item["output_ids"])
                if not forced:
                    raise ValueError("missing recorded output")
                start = time.perf_counter()

                turn = await wrapper.rollout.run_turn(
                    ids, max_tokens=len(forced), sampling=None,
                    forced_tokens=forced, ignore_eos=True, stop_token_ids=(),
                )
                elapsed = time.perf_counter() - start
                if turn.tokens != forced:
                    raise AssertionError("forced replay length/content mismatch")
                turns_out.append(
                    {
                        "task_id": item["task_id"],
                        "turn_id": item["turn_id"],
                        "elapsed_s": elapsed,
                        "ttft_s": first_at.pop(turn.request_id) - start,
                        "prompt_tokens": len(ids),
                        "output_tokens": len(forced),
                    }
                )
                if i + 1 < len(turns):
                    await asyncio.sleep(dwell)
                elif not keep_completed:
                    await wrapper.finish_session(ids + forced)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    async def watchdog():
        previous = -1
        while True:
            await asyncio.sleep(15)
            steps = wrapper.rollout.batching_report()["steps"]
            if steps == previous:
                scheduler = engine.scheduler
                print(
                    json.dumps(
                        {
                            "stalled_arm": arm,
                            "steps": steps,
                            "completed_turns": len(turns_out),
                            "running": [r.request_id for r in scheduler.running],
                            "waiting": [r.request_id for r in scheduler.waiting],
                            "free": engine.block_manager.num_free_blocks,
                            "requests": [
                                {
                                    "id": r.request_id,
                                    "status": str(r.status),
                                    "residency": r.kv_residency,
                                    "blocks": len(r.block_table),
                                }
                                for r in engine._requests.values()
                            ],
                            "session": wrapper.session_report(),
                        }
                    ),
                    flush=True,
                )
                raise RuntimeError("replay made no progress for 15 seconds")
            previous = steps

    watch = asyncio.create_task(watchdog())
    work = asyncio.gather(*(task(t) for t in tasks))
    try:
        done, _ = await asyncio.wait(
            [work, watch], timeout=1800, return_when=asyncio.FIRST_COMPLETED
        )
        if watch in done:
            await watch
        if work not in done:
            raise TimeoutError("replay exceeded 1800 seconds")
        await work
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report = wrapper.session_report()
        totals["capacity_retries"] = wrapper.rollout.batching_report()["capacity_retries"]
        totals["max_batch"] = wrapper.rollout.batching_report()["max_batch"]
        result = {
            "arm": arm,
            "elapsed_s": elapsed,
            "turns": turns_out,
            "session": report,
            "work": totals,
            "transfer": transfer,
            "forced_equal": True,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        work.cancel()
        watch.cancel()
        await asyncio.gather(work, watch, return_exceptions=True)
        wrapper.rollout.clear()
        engine.invalidate_all_kv()
    result["offload_after_cleanup"] = engine.session_offload.report()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--trace",
        default="workloads/code_localization/token_ids/replay_subset_v1_session.jsonl",
    )
    p.add_argument("--ranks", default="10,20,30,40,50,60,70,80")
    p.add_argument("--arms", default="B11,O-sync,apc-only")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-active", type=int, default=8)
    p.add_argument("--dwell", type=float, default=0.2)
    p.add_argument(
        "--blocks",
        type=int,
        default=2048,
        help="0 uses the factory's available-VRAM budget once for every arm",
    )
    p.add_argument("--cpu-gib", type=float, default=16)
    p.add_argument("--chunk-mib", type=int, default=32)
    p.add_argument("--pinned", action="store_true")
    p.add_argument("--apc", type=int, choices=[0, 1], default=1)
    p.add_argument("--policy", choices=["equal", "legacy"], default="equal")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--spec", type=int, default=0)
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument(
        "--keep-completed",
        action="store_true",
        help="match adapter retention when no explicit trajectory-end notification exists",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite result")
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(a.trace, [int(x) for x in a.ranks.split(",")])
    model = load_from_directory(
        a.model, device="cuda", dtype=torch.bfloat16, attention_backend="flashinfer"
    )
    blocks = a.blocks or _num_kv_blocks(
        model, device="cuda", block_size=16, kv_budget=None
    )
    artifact = {
        "args": vars(a),
        "source": "recorded_tokens_synthetic_arrivals_and_tool_delay",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "source_sha256": source_digest(),
        "model_config_sha256": hashlib.sha256(
            (Path(a.model) / "config.json").read_bytes()
        ).hexdigest(),
        "trace_sha256": hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),
        "tasks": [t[0]["task_id"] for t in tasks],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "resolved_num_kv_blocks": blocks,
        "rows": [],
    }
    write_artifact(out, artifact)
    for repeat in range(-1, a.repeats):
        arms = a.arms.split(",")
        random.Random(7100 + repeat).shuffle(arms)
        for arm in arms:
            active = min(a.concurrency, a.max_active)
            cfg = Config(
                num_kv_blocks=blocks,
                max_num_seqs=active,
                max_num_batched_tokens=a.batch_tokens,
                attention_backend="flashinfer",
                cuda_graph=True,
                enable_prefix_cache=bool(a.apc),
                num_speculative_tokens=a.spec,
                session_cpu_offload="sync" if arm == "O-sync" else "off",
                cpu_kv_max_bytes=int(a.cpu_gib * 1024**3),
                cpu_kv_pinned_max_bytes=int(a.cpu_gib * 1024**3) if a.pinned else 0,
                transfer_chunk_bytes=a.chunk_mib * 1024**2,
            )
            e = Engine(cfg, PagedRunner(model, cuda_graph=True, split=True))
            # Warm up representative decode graphs on this exact Engine.
            for n in range(1, active + 1):
                for _ in range(n):
                    e.add_request(list(range(1, 33)), max_tokens=5)
                while e.scheduler.waiting or e.scheduler.running:
                    e.step()
            e.invalidate_all_kv()
            torch.cuda.synchronize()
            try:
                row = asyncio.run(
                    replay(
                        e,
                        tasks,
                        arm=arm,
                        concurrency=a.concurrency,
                        dwell=a.dwell,
                        policy=a.policy,
                        keep_completed=a.keep_completed,
                    )
                )
            except BaseException as exc:
                artifact.setdefault("failures", []).append(
                    {
                        "repeat": repeat,
                        "arm": arm,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                write_artifact(out, artifact)
                raise
            row.update({"repeat": repeat, "engine_config": asdict(cfg)})
            artifact["rows"].append(row)
            write_artifact(out, artifact)
            print(
                json.dumps(
                    {
                        "repeat": repeat,
                        "arm": arm,
                        "elapsed_s": row["elapsed_s"],
                        "work": row["work"],
                        "transfer": row["transfer"],
                        "saved": row["session"].get("offload_saved"),
                        "restored": row["session"].get("offload_restored"),
                    }
                ),
                flush=True,
            )
            e.runner.release_kv_pool()
            del e
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
