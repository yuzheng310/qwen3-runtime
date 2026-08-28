#!/usr/bin/env python3
"""Download the pinned Qwen3-4B snapshot. Does not start AutoDL or create GPU instances."""

from __future__ import annotations

import argparse
from pathlib import Path

PIN_REPO = "Qwen/Qwen3-4B"
PIN_REV = "1cfa9a7208912126459214e8b04321603b3df60c"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/Qwen3-4B"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=PIN_REPO,
        revision=PIN_REV,
        local_dir=str(args.out),
    )
    print(f"downloaded {PIN_REPO}@{PIN_REV} -> {args.out}")
    print("HF greedy: QWEN3_RUNTIME_MODEL=" + str(args.out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
