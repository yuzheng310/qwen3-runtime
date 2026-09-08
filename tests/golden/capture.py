"""Capture greedy token ids and decode tok/s for the teaching-refactor gate.

Requires CUDA and QWEN3_RUNTIME_MODEL pointing at the pinned Qwen3-4B snapshot.
Illegal combinations (pytorch + CUDA Graph) are skipped, not failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PROMPT = [151643, 8948, 198]
MAX_TOKENS = 32
THROUGHPUT_PROMPT = list(range(151643, 151643 + 256))
THROUGHPUT_DECODE = 64

ARMS = (
    {"backend": "pytorch", "cuda_graph": False, "num_speculative_tokens": 0},
    {"backend": "pytorch", "cuda_graph": False, "num_speculative_tokens": 16},
    {"backend": "flashinfer", "cuda_graph": False, "num_speculative_tokens": 0},
    {"backend": "flashinfer", "cuda_graph": True, "num_speculative_tokens": 0},
    {"backend": "flashinfer", "cuda_graph": True, "num_speculative_tokens": 16},
    {"backend": "triton", "cuda_graph": False, "num_speculative_tokens": 0},
    {"backend": "triton", "cuda_graph": True, "num_speculative_tokens": 0},
    {"backend": "triton", "cuda_graph": True, "num_speculative_tokens": 16},
)


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
    return out.strip()


def _build(model_dir: Path, arm: dict):
    from qwen3_runtime.engine.factory import build_engine

    return build_engine(
        model_dir,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        attention_backend=arm["backend"],
        cuda_graph=arm["cuda_graph"],
        num_speculative_tokens=arm["num_speculative_tokens"],
        split=True,
    )


def _run_arm(model_dir: Path, arm: dict) -> dict:
    import torch

    from qwen3_runtime.sampling import SamplingParams

    engine = _build(model_dir, arm)
    greedy = SamplingParams(temperature=0.0)
    tokens = engine.generate(PROMPT, max_tokens=MAX_TOKENS, ignore_eos=True, sampling=greedy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    engine.generate(THROUGHPUT_PROMPT, max_tokens=8, ignore_eos=True, sampling=greedy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.generate(
        THROUGHPUT_PROMPT, max_tokens=THROUGHPUT_DECODE, ignore_eos=True, sampling=greedy
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s = THROUGHPUT_DECODE / elapsed if elapsed > 0 else 0.0
    return {
        **arm,
        "token_ids": tokens,
        "decode_tok_s": round(tok_s, 3),
        "elapsed_s": round(elapsed, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    model = args.model or Path(os.environ["QWEN3_RUNTIME_MODEL"])
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("golden capture requires CUDA")

    arms = []
    errors = []
    for arm in ARMS:
        label = f"{arm['backend']} graph={arm['cuda_graph']} spec={arm['num_speculative_tokens']}"
        try:
            print(f"capturing {label}", flush=True)
            arms.append(_run_arm(model, arm))
        except Exception as exc:  # noqa: BLE001 — record, keep going
            errors.append({"arm": arm, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {label}: {exc}", flush=True)

    payload = {
        "commit": _git_sha(),
        "prompt_token_ids": PROMPT,
        "max_tokens": MAX_TOKENS,
        "throughput_prompt_len": len(THROUGHPUT_PROMPT),
        "throughput_decode": THROUGHPUT_DECODE,
        "model": str(model),
        "arms": arms,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} ({len(arms)} arms, {len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
