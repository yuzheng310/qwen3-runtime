#!/usr/bin/env python3
"""Print GPU-session readiness. Exit 1 if a required artifact is missing.

Does not power on, create, or modify any AutoDL instance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    missing: list[str] = []
    for p in (
        ROOT / "docs/pins/Qwen3-4B/config.json",
        ROOT / "bench/CONTRACT.md",
        ROOT / "docs/PARITY_WORKLOADS.md",
        ROOT / "docs/PERFORMANCE_GAP.md",
        ROOT / "tests/reference/test_hf_transformers_greedy.py",
        ROOT / "qwen3_runtime/correctness/cli.py",
    ):
        if not p.exists():
            missing.append(str(p))
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        commit = "uncommitted"
        missing.append("git commit (repo has no HEAD yet — required before a GPU session)")
    dirty = True
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(porcelain.strip())
    except subprocess.CalledProcessError:
        missing.append("git status")
    print("hypothesis: clean-commit 4090 baseline; decompose vs-vLLM gap before engine changes")
    print("correctness: QWEN3_RUNTIME_MODEL=... python -m qwen3_runtime.correctness.cli --max-tokens 16")
    print("benchmark: python -m bench.run_bench --case decode --scale full --engine qwen3-runtime")
    print("parity cases: decode latency batch8 throughput prefill longctx")
    print("profile: QWEN3_RUNTIME_MODEL=/path/to/model bash scripts/nsys_profile.sh decode")
    print("git:", commit, "dirty:", dirty)
    if dirty:
        missing.append("clean working tree (dirty=false required for full-scale JSON)")
    if missing:
        print("NOT READY:", *missing, sep="\n  ")
        return 1
    print("docs/pin/contract/parity freeze present. 4B weights and GPU still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
