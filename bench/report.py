"""Validate benchmark JSON against the minimum artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = (
    "schema_version",
    "engine",
    "case",
    "environment",
    "metrics",
    "forced_length_ok",
)


def load_raw(path: Path) -> dict:
    data = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    if data.get("forced_length_ok") is not True:
        raise ValueError(f"{path} forced_length_ok is not true")
    env = data["environment"]
    if "git_commit" not in env:
        raise ValueError(f"{path} environment missing git_commit")
    if "command" not in env:
        raise ValueError(f"{path} environment missing command")
    if "timestamp_utc" not in env and "timestamp" not in env:
        raise ValueError(f"{path} environment missing timestamp")
    return data


def check_dir(raw_dir: Path) -> list[dict]:
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON in {raw_dir}")
    return [load_raw(p) for p in files]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, required=True)
    args = parser.parse_args(argv)
    check_dir(args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
