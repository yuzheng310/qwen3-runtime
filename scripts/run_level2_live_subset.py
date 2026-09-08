#!/usr/bin/env python3
"""Level-2 live CodeScout subset through qwen3-runtime.

Runs CodeScout's CustomAgent + Terminal + localization_finish against a
running OpenAI adapter. Does not import SkyRL/Ray. Not Level-1 Replay.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
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


def session_base_url(base_url: str, session_key: str | None) -> str:
    if not session_key:
        return base_url
    root = base_url.rstrip("/")
    return f"{root}/s/{session_key}/"


def session_reset_url(base_url: str, session_key: str | None) -> str:
    if session_key:
        return session_base_url(base_url, session_key).rstrip("/") + "/reset"
    return base_url.rstrip("/") + "/reset"


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
                ["curl", "-kL", "--max-time", "300", "-o", str(tmp), archive],
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
            timeout=300,
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


def extract_tool_turns(messages: list[dict]) -> list[dict]:
    """Per-turn tool command and observation size. Does not change the agent loop.

    Used by the A0 control: if tokens diverge, this log shows whether the tool
    command/output moved first. Parsing is read-only over Conversation events.
    """
    turns: list[dict] = []
    pending: dict | None = None
    for event in messages:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or event.get("type") or "")
        tool_name = event.get("tool_name")
        action = event.get("action") if isinstance(event.get("action"), dict) else {}
        if tool_name is None:
            tool_name = action.get("name") or action.get("tool_name")
        command = action.get("command")
        if command is None and isinstance(action.get("arguments"), dict):
            command = action["arguments"].get("command")
        if command is None and isinstance(event.get("args"), dict):
            command = event["args"].get("command")
        is_action = ("Action" in kind) or (command is not None and "Observation" not in kind)
        is_obs = "Observation" in kind or kind.endswith("ObservationEvent")
        if is_action and (command is not None or tool_name):
            pending = {
                "turn": len(turns),
                "kind": kind,
                "tool_name": tool_name,
                "command": command if isinstance(command, str) else (json.dumps(command) if command is not None else None),
                "output": None,
                "output_bytes": 0,
            }
            continue
        if is_obs or (pending is not None and event.get("content") is not None and "Action" not in kind):
            output = event.get("content")
            if output is None:
                obs = event.get("observation")
                if isinstance(obs, dict):
                    output = obs.get("content") or obs.get("text") or obs.get("output")
                elif isinstance(obs, str):
                    output = obs
            if isinstance(output, dict):
                output = output.get("content") or output.get("text") or json.dumps(output)
            if output is None:
                extras = event.get("extras") if isinstance(event.get("extras"), dict) else {}
                output = extras.get("output") or extras.get("content")
            text = output if isinstance(output, str) else ("" if output is None else str(output))
            n_bytes = len(text.encode("utf-8"))
            if pending is None:
                pending = {
                    "turn": len(turns),
                    "kind": kind,
                    "tool_name": tool_name,
                    "command": None,
                    "output": None,
                    "output_bytes": n_bytes,
                }
            pending["output"] = text[:2000]
            pending["output_bytes"] = n_bytes
            pending["observation_kind"] = kind
            turns.append(pending)
            pending = None
    if pending is not None:
        turns.append(pending)
    return turns


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


def run_one(
    instance: dict,
    *,
    base_url: str,
    model_name: str,
    max_turns: int,
    work_root: Path,
    session_key: str | None = None,
    temperature: float = 0.6,
    request_logprobs: bool = False,
) -> dict:
    from openhands.sdk import LLM, Conversation
    from openhands.sdk.conversation.response_utils import get_agent_final_response
    from openhands.sdk.tool import Tool, register_tool
    from openhands.tools.terminal import TerminalTool
    if os.environ.get("STEP51_PIN_RECORD") or os.environ.get("STEP51_PIN_REPLAY"):
        import importlib.util

        helper = Path(__file__).resolve().parent / "step51_pin_terminal.py"
        spec = importlib.util.spec_from_file_location("step51_pin_terminal", helper)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {helper}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.install_pin()
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
    cache_tid = str(instance.get("_cache_tid") or tid)
    ok, working_dir = False, None
    last_err = None
    for attempt in range(3):
        ok, working_dir = clone_instance_cached(str(repo), str(commit), cache_tid, work_root / "git-cache")
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
    extra_body = {
        "return_token_ids": True,
        "include_stop_str_in_output": False,
        "chat_template_kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    }
    if request_logprobs:
        extra_body["logprobs"] = True
        extra_body["top_logprobs"] = 20
        extra_body["return_tokens_as_token_ids"] = True
    llm_kwargs = dict(
        model="openai/" + model_name,
        base_url=session_base_url(base_url, session_key),
        api_key="sk-xxx",
        temperature=float(temperature),
        litellm_extra_body=extra_body,
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
        if session_key:
            try:
                _http_json(session_reset_url(base_url, session_key), {}, method="POST")
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
    tool_turns = extract_tool_turns(messages)
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
        "tool_turns": tool_turns,
        "n_tool_turns": len(tool_turns),
        "structured_locations": structured,
        "reward": reward_val,
        "reward_detail": reward_detail,
        "final_message_head": (final_message or "")[:500],
        "working_dir": str(working_dir),
        "session_key": session_key,
        "cache_tid": cache_tid,
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="LLM sampling temperature. §5.0 uses 0.0.",
    )
    parser.add_argument(
        "--request-logprobs",
        action="store_true",
        help="Ask the OpenAI server for logprobs + token_id placeholders (vLLM §5.0 capture).",
    )
    parser.add_argument("--work-root", type=Path, default=Path("codescout-live"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1, help="N workers, each with /v1/s/{key}/")
    parser.add_argument(
        "--session-urls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="N>1 uses /v1/s/{worker}/chat/completions. Disable for vLLM, which has no session paths.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Replicate each instance G times (GRPO shape). Each replica gets its own checkout.",
    )
    parser.add_argument("--prewarm-only", action="store_true", help="Clone every repo then exit. Untimed.")
    args = parser.parse_args()

    global PROMPTS
    codescout = args.codescout.resolve()
    sys.path.insert(0, str(codescout))
    PROMPTS = codescout / "src" / "prompts"
    os.chdir(codescout)
    configure_github_mirror()

    task_ids = args.task_ids or load_task_ids(args.subset, args.n_tasks, args.prefer_small)
    instances = load_locagent_rows(task_ids)
    if args.group_size < 1 or args.concurrency < 1:
        raise SystemExit("concurrency and group-size must be >= 1")
    if args.group_size > 1:
        expanded = []
        for inst in instances:
            for g in range(args.group_size):
                row = dict(inst)
                row["_replica"] = g
                row["_cache_tid"] = f"{inst['instance_id']}__g{g}"
                expanded.append(row)
        instances = expanded
    args.work_root.mkdir(parents=True, exist_ok=True)

    if args.prewarm_only:
        ok = 0
        for inst in instances:
            tid = inst.get("_cache_tid") or inst["instance_id"]
            repo = inst.get("repo") or inst.get("repo_id")
            commit = inst.get("base_commit")
            print(f"prewarm {tid}", flush=True)
            success, path = clone_instance_cached(str(repo), str(commit), str(tid), args.work_root / "git-cache")
            print(" ", success, path, flush=True)
            ok += int(bool(success))
        print(f"prewarm_ok {ok}/{len(instances)}", flush=True)
        if ok < len(instances):
            raise SystemExit(f"prewarm incomplete {ok}/{len(instances)}")
        return

    def _write_partial(rows: list) -> None:
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

    t_all = time.perf_counter()
    rows: list = []
    if args.concurrency == 1:
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
                    temperature=args.temperature,
                    request_logprobs=args.request_logprobs,
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
            _write_partial(rows)
            try:
                _http_json(args.base_url.rstrip("/") + "/reset", {}, method="POST")
            except Exception:
                try:
                    _http_json(args.base_url.rstrip("/") + "/reset")
                except Exception:
                    pass
    else:
        work_q: queue.Queue = queue.Queue()
        for inst in instances:
            work_q.put(inst)
        rows_lock = threading.Lock()

        def worker(wid: int) -> None:
            key = f"w{wid}"
            while True:
                try:
                    instance = work_q.get_nowait()
                except queue.Empty:
                    return
                tid = instance["instance_id"]
                print(f"=== live {tid} session={key} ===", flush=True)
                try:
                    row = run_one(
                        instance,
                        base_url=args.base_url,
                        model_name=args.model_name,
                        max_turns=args.max_turns,
                        work_root=args.work_root,
                        session_key=key if args.session_urls else None,
                        temperature=args.temperature,
                        request_logprobs=args.request_logprobs,
                    )
                except Exception as exc:
                    row = {
                        "task_id": tid,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}",
                        "wall_s": 0.0,
                        "session_key": key,
                    }
                print(json.dumps({k: row[k] for k in ("task_id", "ok", "error", "wall_s", "reward", "session_key") if k in row}), flush=True)
                with rows_lock:
                    rows.append(row)
                    _write_partial(rows)
                work_q.task_done()

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    wall = time.perf_counter() - t_all
    stats = {}
    try:
        stats = _http_json(args.base_url.rstrip("/") + "/stats")
    except Exception as exc:
        stats = {"error": str(exc)}
    source_commit = os.environ.get("SOURCE_COMMIT", "uncommitted")
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    inference_s = float(stats.get("drain_s") or 0.0)
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_finish = sum(1 for r in rows if (r.get("n_localization_finish") or 0) > 0)
    conv = sum(float(r.get("wall_s") or 0) for r in rows)
    rewards = [float(r["reward"]) for r in rows if r.get("reward") is not None]
    payload = {
        "schema_version": 1,
        "mode": "live_codescout_subset",
        "backend": args.backend_label,
        "n_tasks": len(rows),
        "completed_ok": n_ok,
        "task_ids": task_ids,
        "concurrency": args.concurrency,
        "group_size": args.group_size,
        "session_urls": args.session_urls,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "thinking": False,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "codescout_path": str(codescout),
        "source_commit": source_commit,
        "wall_s": wall,
        "conversation_wall_sum_s": conv,
        "conversation_s_per_task": (conv / len(rows)) if rows else None,
        "n_localization_finish_trajectories": n_finish,
        "finish_rate": (n_finish / len(rows)) if rows else None,
        "inference_drain_s": inference_s,
        "tool_or_env_s": max(0.0, wall - inference_s) if inference_s else None,
        "trajectories_per_hour": (3600.0 * n_ok / wall) if wall and n_ok else 0.0,
        "n_terminal_actions_total": sum(int(r.get("n_terminal_actions") or 0) for r in rows),
        "n_localization_finish_total": sum(int(r.get("n_localization_finish") or 0) for r in rows),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "inference_gpu_s_per_ok_task": (inference_s / n_ok) if n_ok else None,
        "server_stats": stats,
        "tasks": rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Do not report these as Level-1 Replay numbers. Parity uses conversation_s_per_task, never outer wall_s.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "tasks"}, indent=2, default=str))


if __name__ == "__main__":
    main()
