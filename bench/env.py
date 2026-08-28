"""Environment capture for raw JSON. Missing fields stay null, never guessed."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def git_commit(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def capture(repo: Path, command: list[str]) -> dict:
    env: dict = {
        "git_commit": git_commit(repo),
        "dirty": git_dirty(repo),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "hostname": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": None,
        "gpu_memory_total_bytes": None,
        "nvidia_driver": None,
        "triton": None,
        "vllm": None,
        "transformers": None,
        "flash_attn": None,
        "flashinfer": None,
        "dtype": None,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_memory_total_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
        try:
            env["nvidia_driver"] = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
                .splitlines()[0]
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            env["nvidia_driver"] = None
    try:
        import triton  # noqa: F401

        env["triton"] = getattr(triton, "__version__", "present")
    except ImportError:
        env["triton"] = None
    for name in ("vllm", "transformers", "flash_attn", "flashinfer"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "present")
        except ImportError:
            env[name] = None
    return env
