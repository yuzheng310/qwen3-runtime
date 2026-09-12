"""Completed transfer and recomputation costs on real trace prefixes.

This isolates the economic boundary; it is not an end-to-end serving result.
All generation checks use sampled greedy tokens, never forced outputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.utils.loader import load_from_directory


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--trace",
        default="workloads/code_localization/token_ids/replay_subset_v1_session.jsonl",
    )
    p.add_argument("--lengths", default="128,512,2048,8192,16384")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--chunk-mib", type=int, default=8)
    p.add_argument("--pinned", action="store_true")
    p.add_argument("--spec", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in Path(a.trace).read_text().splitlines()]
    lengths = [int(n) for n in a.lengths.split(",")]
    source = next(r for r in rows if len(r["input_ids"]) >= max(lengths))
    model = load_from_directory(
        a.model, device="cuda", dtype=torch.bfloat16, attention_backend="flashinfer"
    )
    manifest = {
        "args": vars(a),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "source_sha256": hashlib.sha256(
            b"".join(
                str(path).encode() + path.read_bytes()
                for path in sorted(Path("qwen3_runtime").rglob("*.py"))
                + [Path(__file__)]
            )
        ).hexdigest(),
        "model_config_sha256": hashlib.sha256(
            (Path(a.model) / "config.json").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "source_task": source["task_id"],
        "trace_sha256": hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),
        "source": "real_trace_prefix_length_sweep",
        "measurement_notes": (
            "The untimed KV oracle export warms staging/export kernels before saving. "
            "first_sample is the first sample at this length, not a fully cold process; "
            "replay repeat=-1 records end-to-end first-use costs."
        ),
        "rows": [],
    }
    for length in lengths:
        config = Config(
            num_kv_blocks=(length + 128 + 15) // 16,
            attention_backend="flashinfer",
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            cuda_graph=True,
            num_speculative_tokens=a.spec,
            session_cpu_offload="sync",
            cpu_kv_max_bytes=8 * 1024**3,
            cpu_kv_pinned_max_bytes=8 * 1024**3 if a.pinned else 0,
            transfer_chunk_bytes=a.chunk_mib * 1024**2,
        )
        e = Engine(config, PagedRunner(model, cuda_graph=True, split=True))
        prompt = source["input_ids"][:length]
        samples = []
        first_sample = None
        for repeat in range(-1, a.repeats):
            rid = e.add_request(prompt, max_tokens=1, hold_kv=True)
            first, cold_s = timed(partial(e.drain_request, rid))
            req = e._requests[rid]
            original = e.runner.pool.export_blocks(
                req.block_table,
                valid_tokens=req.num_computed_tokens,
                chunk_bytes=config.transfer_chunk_bytes,
            )
            _, save_s = timed(partial(e.offload_request, rid))
            _, restore_s = timed(
                partial(e.resume_request, rid, [30, 31], 8, hold_kv=False)
            )
            restored = e.runner.pool.export_blocks(
                req.block_table,
                valid_tokens=req.num_computed_tokens,
                chunk_bytes=config.transfer_chunk_bytes,
            )
            equal = torch.equal(original, restored)
            tokens, resume_s = timed(partial(e.drain_request, rid))
            lps = list(e.last_completion_logprobs[rid])
            cold_id = e.add_request(prompt + first + [30, 31], max_tokens=8)
            expected, recompute_s = timed(partial(e.drain_request, cold_id))
            expected_lps = e.last_completion_logprobs[cold_id]
            held_id = e.add_request(prompt, max_tokens=1, hold_kv=True)
            held_first = e.drain_request(held_id)
            e.resume_request(held_id, [30, 31], 8, hold_kv=False)
            held_tokens = e.drain_request(held_id)
            held_lps = e.last_completion_logprobs[held_id]
            row = {
                "length": length,
                "repeat": repeat,
                "cold_prefill_s": cold_s,
                "save_s": save_s,
                "restore_s": restore_s,
                "resumed_generation_s": resume_s,
                "cold_generation_s": recompute_s,
                "saved_recompute_s": recompute_s - resume_s,
                "kv_exact": equal,
                "greedy_equal": tokens == expected,
                "logprob_max_abs": max(abs(x - y) for x, y in zip(lps, expected_lps)),
                "held_greedy_equal": first == held_first and tokens == held_tokens,
                "held_logprob_max_abs": max(abs(x - y) for x, y in zip(lps, held_lps)),
                "bytes": original.numel() * original.element_size(),
            }
            if (
                not equal
                or not row["held_greedy_equal"]
                or row["held_logprob_max_abs"] != 0
            ):
                raise AssertionError(row)
            if repeat >= 0:
                samples.append(row)
            else:
                first_sample = row
            del original, restored
        med = {
            k: statistics.median(r[k] for r in samples)
            for k in samples[0]
            if isinstance(samples[0][k], float)
        }
        result = {"length": length, "medians": med, "samples": samples,
                  "first_sample": first_sample}
        manifest["rows"].append(result)
        out.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"length": length, **med}), flush=True)
        e.runner.release_kv_pool()
        del e
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
