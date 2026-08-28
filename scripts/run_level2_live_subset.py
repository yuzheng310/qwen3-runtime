#!/usr/bin/env python3
"""Level-2 live CodeScout subset through qwen3-runtime.

Runs CodeScout's CustomAgent + Terminal + localization_finish against a
running OpenAI adapter. Does not import SkyRL/Ray. Not Level-1 Replay.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = ROOT / "workloads" / "code_localization" / "replay_subset_v1.json"
SMALL_PREFIXES = ("pytest-dev__", "psf__")
PROMPTS = None
DEFAULT_GITHUB_MIRROR = "https://ghproxy.net/https://github.com/"


def configure_github_mirror() -> None:
    """AutoDL cannot clone github.com directly (GnuTLS/TLS reset). Optional rewrite."""
    mirror = os.environ.get("GITHUB_CLONE_MIRROR", DEFAULT_GITHUB_MIRROR)
    if not mirror or mirror in ("0", "off", "none"):
        return
    os.environ.setdefault("GIT_SSL_NO_VERIFY", "1")
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = f"url.{mirror.rstrip('/')}/.insteadOf"
    os.environ["GIT_CONFIG_VALUE_0"] = "https://github.com/"


def clone_instance_cached(repo: str, commit: str, tid: str, cache_root: Path) -> tuple[bool, Path | None]:
    """Materialize the task commit. Prefer GitHub archive (curl -k); git fetch is fallback.

    AutoDL ghproxy TLS verify fails; ``GIT_SSL_NO_VERIFY`` / curl ``-k`` matches
    the prior live harness. Same github.com URLs as CodeScout ``clone_instance``.
    """
    import subprocess
    import tarfile

    cache_root.mkdir(parents=True, exist_ok=True)
    instance_path = cache_root / f"{repo.replace('/', '_')}_{tid}"
    if instance_path.exists() and (instance_path / ".git").exists() and (
        (instance_path / "pyproject.toml").exists() or (instance_path / "setup.py").exists()
    ):
        return True, instance_path
    if instance_path.exists():
        shutil.rmtree(instance_path, ignore_errors=True)
    tmp = cache_root / f"{tid}.tgz"
    archives = [
        f"https://ghproxy.net/https://github.com/{repo}/archive/{commit}.tar.gz",
        f"https://ghfast.top/https://github.com/{repo}/archive/{commit}.tar.gz",
    ]
    for archive in archives:
        try:
            print(f"  archive {archive}", flush=True)
            subprocess.run(
                ["curl", "-kL", "--max-time", "120", "-o", str(tmp), archive],
                check=True,
                capture_output=True,
                text=True,
            )
            if tmp.stat().st_size < 1024:
                raise RuntimeError(f"tiny archive {tmp.stat().st_size}B")
            instance_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tmp, "r:gz") as tf:
                tf.extractall(instance_path, filter="data")
            members = [p for p in instance_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if len(members) == 1:
                inner = members[0]
                for child in inner.iterdir():
                    dest = instance_path / child.name
                    if dest.exists():
                        continue
                    child.rename(dest)
                shutil.rmtree(inner, ignore_errors=True)
            subprocess.run(["git", "-C", str(instance_path), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(instance_path), "add", "-A"], check=True, capture_output=True, text=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(instance_path),
                    "-c",
                    "user.email=live@local",
                    "-c",
                    "user.name=live",
                    "commit",
                    "-m",
                    f"snapshot {commit[:8]}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            tmp.unlink(missing_ok=True)
            return True, instance_path
        except Exception as exc:
            print(f"  archive failed: {exc}", flush=True)
            shutil.rmtree(instance_path, ignore_errors=True)
            tmp.unlink(missing_ok=True)
    url = f"https://github.com/{repo}.git"
    try:
        instance_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(instance_path), "init"], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(instance_path), "remote", "add", "origin", url],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(instance_path), "fetch", "--depth", "1", "origin", commit],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(instance_path), "checkout", "--force", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, instance_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        err = getattr(exc, "stderr", None) or str(exc)
        print(f"  shallow fetch failed {tid}: {str(err)[-300:]}", flush=True)
        shutil.rmtree(instance_path, ignore_errors=True)
        return False, None
    """Fetch one commit. Prefer shallow git; fall back to GitHub archive tarball.

    Uses the same github.com URL as CodeScout ``clone_instance``; GIT_CONFIG
    rewrite from ``configure_github_mirror`` still applies.
    """
    import subprocess
    import tarfile
    import urllib.request

    cache_root.mkdir(parents=True, exist_ok=True)
    instance_path = cache_root / f"{repo.replace('/', '_')}_{tid}"
    if instance_path.exists() and (instance_path / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(instance_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if commit and head.stdout.strip().startswith(str(commit)[:8]):
            return True, instance_path
    if instance_path.exists():
        shutil.rmtree(instance_path, ignore_errors=True)
    url = f"https://github.com/{repo}.git"
    try:
        instance_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(instance_path), "init"], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(instance_path), "remote", "add", "origin", url],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(instance_path), "fetch", "--depth", "1", "origin", commit],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(instance_path), "checkout", "--force", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, instance_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        err = getattr(exc, "stderr", None) or str(exc)
        print(f"  shallow fetch failed {tid}: {str(err)[-300:]}", flush=True)
        shutil.rmtree(instance_path, ignore_errors=True)
    archives = [
        f"https://ghproxy.net/https://github.com/{repo}/archive/{commit}.tar.gz",
        f"https://codeload.github.com/{repo}/tar.gz/{commit}",
    ]
    tmp = cache_root / f"{tid}.tgz"
    for archive in archives:
        try:
            print(f"  archive {archive}", flush=True)
            urllib.request.urlretrieve(archive, tmp)
            instance_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tmp, "r:gz") as tf:
                tf.extractall(instance_path, filter="data")
            members = [p for p in instance_path.iterdir() if p.is_dir()]
            if len(members) == 1:
                inner = members[0]
                for child in inner.iterdir():
                    child.rename(instance_path / child.name)
                inner.rmdir()
            subprocess.run(["git", "-C", str(instance_path), "init"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(instance_path), "add", "-A"], check=True, capture_output=True, text=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(instance_path),
                    "-c",
                    "user.email=live@local",
                    "-c",
                    "user.name=live",
                    "commit",
                    "-m",
                    f"snapshot {commit[:8]}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            tmp.unlink(missing_ok=True)
            return True, instance_path
        except Exception as exc:
            print(f"  archive failed: {exc}", flush=True)
            shutil.rmtree(instance_path, ignore_errors=True)
            tmp.unlink(missing_ok=True)
    return False, None


def _http_json(url: str, payload: dict | None = None, method: str | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_task_ids(subset: Path, n: int, prefer_small: bool) -> list[str]:
    data = json.loads(subset.read_text())
    ids = list(data["task_ids"])
    if prefer_small:
        ids = [tid for tid in ids if tid.startswith(SMALL_PREFIXES)]
    return ids[:n]


def load_locagent_rows(task_ids: list[str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("OpenHands/SWE-bench_Verified-locagent", split="test")
    wanted = set(task_ids)
    by_id = {}
    for row in ds:
        tid = row.get("instance_id")
        if tid in wanted:
            by_id[tid] = dict(row)
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise SystemExit(f"locagent missing instance_ids: {missing}")
    return [by_id[t] for t in task_ids]


def get_structured_locations(events) -> list[dict] | None:
    from openhands.sdk.event import ActionEvent
    from src.tools.localization_finish import LocalizationFinishAction

    hits = [
        event
        for event in events
        if isinstance(event, ActionEvent)
        and event.source == "agent"
        and isinstance(event.action, LocalizationFinishAction)
    ]
    if len(hits) != 1:
        return None
    locations = []
    for loc in hits[0].action.locations:
        locations.append(
            {
                "file": loc.file,
                "class_name": loc.class_name,
                "function_name": loc.function_name,
            }
        )
    return locations


def run_one(instance: dict, *, base_url: str, model_name: str, max_turns: int, work_root: Path) -> dict:
    from openhands.sdk import LLM, Conversation
    from openhands.sdk.conversation.response_utils import get_agent_final_response
    from openhands.sdk.tool import Tool, register_tool
    from openhands.tools.terminal import TerminalTool
    from src.agent.agent import CustomAgent
    from src.prompts.prompt_builder import get_instruction
    from src.rewards import get_reward_function
    from src.tools.localization_finish import LocalizationFinishTool

    prompts_dir = Path(PROMPTS)
    system_prompt = str(prompts_dir / "templates" / "system_prompt_custom_finish.j2")
    user_prompt = str(prompts_dir / "templates" / "file_module_custom_finish.j2")
    repo = instance.get("repo") or instance.get("repo_id")
    commit = instance.get("base_commit")
    tid = instance["instance_id"]
    ok, working_dir = False, None
    last_err = None
    for attempt in range(3):
        ok, working_dir = clone_instance_cached(str(repo), str(commit), tid, work_root / "git-cache")
        if ok and working_dir is not None:
            break
        last_err = f"clone_failed_attempt_{attempt+1}"
        time.sleep(2 * (attempt + 1))
    if not ok or working_dir is None:
        return {
            "task_id": tid,
            "ok": False,
            "error": last_err or "clone_failed",
            "wall_s": 0.0,
        }
    register_tool(LocalizationFinishTool.name, LocalizationFinishTool)
    llm_kwargs = dict(
        model="openai/" + model_name,
        base_url=base_url,
        api_key="sk-xxx",
        temperature=0.6,
        litellm_extra_body={
            "return_token_ids": True,
            "include_stop_str_in_output": False,
            "chat_template_kwargs": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        },
    )
    try:
        llm = LLM(usage_id="agent", **llm_kwargs)
    except TypeError:
        llm = LLM(service_id="agent", **llm_kwargs)
    agent = CustomAgent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name="localization_finish"),
        ],
        system_prompt_filename=system_prompt,
    )
    conv_kwargs = dict(
        agent=agent,
        max_iteration_per_run=max_turns,
        workspace=str(working_dir),
    )
    try:
        conversation = Conversation(visualizer=None, **conv_kwargs)
    except TypeError:
        conversation = Conversation(visualize=False, **conv_kwargs)
    instruction = get_instruction(instance, user_prompt, str(working_dir))
    t0 = time.perf_counter()
    start_ts = datetime.now(timezone.utc).isoformat()
    error = None
    messages = []
    final_message = ""
    structured = None
    try:
        conversation.send_message(instruction)
        conversation.run()
        messages = [event.model_dump() for event in conversation.state.events]
        final_message = get_agent_final_response(conversation.state.events) or ""
        structured = get_structured_locations(conversation.state.events)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            messages = [event.model_dump() for event in conversation.state.events]
            final_message = get_agent_final_response(conversation.state.events) or ""
            structured = get_structured_locations(conversation.state.events)
        except Exception:
            pass
    finally:
        wall_s = time.perf_counter() - t0
        try:
            conversation.close()
        except Exception:
            pass
        try:
            import subprocess

            subprocess.run(
                ["git", "-C", str(working_dir), "reset", "--hard", "HEAD"],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(working_dir), "clean", "-fd"],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass
    reward_val = None
    reward_detail = None
    try:
        fn = get_reward_function("multilevel_localization_f1_reward")
        reward_val, reward_detail = fn(
            final_message, instance, structured_locations=structured
        )
    except Exception as exc:
        reward_detail = {"error": f"{type(exc).__name__}: {exc}"}
    n_turns = sum(1 for m in messages if m.get("kind") == "TokenEvent")
    if n_turns == 0:
        n_turns = sum(1 for m in messages if m.get("source") == "agent" and m.get("kind") == "MessageEvent")
    n_terminal = sum(1 for m in messages if m.get("tool_name") == "terminal")
    n_finish = sum(1 for m in messages if m.get("tool_name") == "localization_finish")
    return {
        "task_id": tid,
        "ok": error is None,
        "error": error,
        "repo": repo,
        "base_commit": commit,
        "wall_s": wall_s,
        "start_timestamp": start_ts,
        "n_events": len(messages),
        "n_turns_est": n_turns,
        "n_terminal_actions": n_terminal,
        "n_localization_finish": n_finish,
        "structured_locations": structured,
        "reward": reward_val,
        "reward_detail": reward_detail,
        "final_message_head": (final_message or "")[:500],
        "working_dir": str(working_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codescout", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--prefer-small", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--backend-label", default="qwen3-runtime")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1/")
    parser.add_argument("--model-name", default="CodeScout-4B")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--work-root", type=Path, default=Path("/root/autodl-tmp/codescout-live"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    global PROMPTS
    codescout = args.codescout.resolve()
    sys.path.insert(0, str(codescout))
    PROMPTS = codescout / "src" / "prompts"
    os.chdir(codescout)
    configure_github_mirror()

    task_ids = args.task_ids or load_task_ids(args.subset, args.n_tasks, args.prefer_small)
    instances = load_locagent_rows(task_ids)
    args.work_root.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    rows = []
    for instance in instances:
        tid = instance["instance_id"]
        print(f"=== live {tid} ===", flush=True)
        try:
            row = run_one(
                instance,
                base_url=args.base_url,
                model_name=args.model_name,
                max_turns=args.max_turns,
                work_root=args.work_root,
            )
        except Exception as exc:
            row = {
                "task_id": tid,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}",
                "wall_s": 0.0,
            }
        print(json.dumps({k: row[k] for k in ("task_id", "ok", "error", "wall_s", "reward") if k in row}), flush=True)
        rows.append(row)
        partial = {
            "schema_version": 1,
            "mode": "live_codescout_subset",
            "partial": True,
            "completed_ok": sum(1 for r in rows if r.get("ok")),
            "n_tasks_so_far": len(rows),
            "tasks": rows,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(partial, indent=2, default=str) + "\n")
        try:
            _http_json(args.base_url.rstrip("/") + "/reset", {}, method="POST")
        except Exception:
            try:
                _http_json(args.base_url.rstrip("/") + "/reset")
            except Exception:
                pass
    wall = time.perf_counter() - t_all
    stats = {}
    try:
        stats = _http_json(args.base_url.rstrip("/") + "/stats")
    except Exception as exc:
        stats = {"error": str(exc)}
    inference_s = float(stats.get("drain_s") or 0.0)
    n_ok = sum(1 for r in rows if r.get("ok"))
    payload = {
        "schema_version": 1,
        "mode": "live_codescout_subset",
        "backend": args.backend_label,
        "n_tasks": len(rows),
        "completed_ok": n_ok,
        "task_ids": task_ids,
        "max_turns": args.max_turns,
        "temperature": 0.6,
        "thinking": False,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "codescout_path": str(codescout),
        "wall_s": wall,
        "inference_drain_s": inference_s,
        "tool_or_env_s": max(0.0, wall - inference_s) if inference_s else None,
        "trajectories_per_hour": (3600.0 * n_ok / wall) if wall and n_ok else 0.0,
        "n_terminal_actions_total": sum(int(r.get("n_terminal_actions") or 0) for r in rows),
        "n_localization_finish_total": sum(int(r.get("n_localization_finish") or 0) for r in rows),
        "mean_reward": (sum(float(r["reward"]) for r in rows if r.get("reward") is not None) / n_ok) if n_ok else None,
        "inference_gpu_s_per_ok_task": (inference_s / n_ok) if n_ok else None,
        "server_stats": stats,
        "tasks": rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Do not report these as Level-1 Replay numbers.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "tasks"}, indent=2, default=str))


if __name__ == "__main__":
    main()
