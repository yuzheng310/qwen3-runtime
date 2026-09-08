"""Local SkyRL InferenceEngineInterface for colocated single-GPU GRPO."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any, Dict, Optional

from qwen3_runtime.rollout.driver import EngineDriver
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.sampling import SamplingParams
from qwen3_runtime.rollout.tool_parser import assistant_chat_message, parse_qwen_tool_calls
from qwen3_runtime.rollout.session import SessionCache, session_kv_enabled

IM_END = 151645
END_OF_TEXT = 151643
STOP_TOKEN_IDS = (IM_END, END_OF_TEXT)

try:
    from skyrl_train.inference_engines.base import (
        InferenceEngineInput,
        InferenceEngineInterface,
        InferenceEngineOutput,
        NamedWeightsUpdateRequest,
    )
except ImportError:  # tests / boxes without the SkyRL venv
    from qwen3_runtime.integrations.skyrl.protocol import (
        InferenceEngineInput,
        InferenceEngineInterface,
        InferenceEngineOutput,
        NamedWeightsUpdateRequest,
    )


def _flatten_message_content(content: Any) -> str:
    """OpenHands sends OpenAI content-part lists; the Qwen/Hermes template does `content + '\\n'`."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _normalize_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        msg = dict(raw)
        msg["content"] = _flatten_message_content(msg.get("content"))
        out.append(msg)
    return out


def _normalize_tools(tools: Any) -> list[Any] | None:
    if tools is None or tools is False:
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
    if isinstance(tools, list) and tools:
        return tools
    return None


def _coerce_token_ids(raw: Any, tokenizer: Any) -> list[int]:
    """transformers 5 `apply_chat_template(tokenize=True)` may return a BatchEncoding.

    `list(batch_encoding)` is dict keys (`'input_ids'`, …). Feeding that to
    `torch.tensor(..., dtype=long)` raises `ValueError: too many dimensions 'str'`.
    Same extraction as `scripts/codescout_openai_server.py:tokenize_chat`.
    """
    if hasattr(raw, "get") and not isinstance(raw, (list, tuple, str)):
        extracted = raw.get("input_ids")
        if extracted is not None:
            raw = extracted
    elif hasattr(raw, "input_ids") and not isinstance(raw, (list, tuple, str)):
        raw = raw.input_ids
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, str):
        raw = tokenizer.encode(raw, add_special_tokens=False)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raw = list(raw)
    if raw and isinstance(raw[0], str):
        raise TypeError(f"chat template produced non-token ids: {raw[:8]!r}")
    return [int(x) for x in raw]


def _apply_chat_template_ids(tokenizer: Any, messages: list[Any], **kwargs: Any) -> list[int]:
    kwargs = dict(kwargs)
    kwargs.setdefault("tokenize", True)
    kwargs.setdefault("return_dict", False)
    try:
        raw = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            raw = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("return_dict", None)
            raw = tokenizer.apply_chat_template(messages, **kwargs)
    return _coerce_token_ids(raw, tokenizer)


def _chat_max_tokens(body: Dict[str, Any]) -> int:
    """LiteLLM/OpenAI send max_completion_tokens; do not inherit generate()'s 16-token default."""
    raw_max = body.get("max_completion_tokens")
    if raw_max is None:
        raw_max = body.get("max_tokens")
    if raw_max is None:
        raw_max = body.get("max_generate_length")
    return 2048 if raw_max is None else int(raw_max)


def _sampling_from_dict(raw: Optional[Dict[str, Any]]) -> tuple[SamplingParams, int]:
    raw = raw or {}
    max_tokens = int(raw.get("max_tokens") or raw.get("max_generate_length") or 16)
    top_k = raw.get("top_k", 0)
    if top_k is None or int(top_k) < 0:
        top_k = 0
    params = SamplingParams(
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        top_k=int(top_k),
        seed=raw.get("seed"),
    )
    return params, max_tokens


def _openai_logprob_content(
    tokenizer: Any, token_ids: list[int], logprobs: list[float]
) -> list[dict[str, Any]]:
    """LiteLLM ``ChoiceLogprobs`` requires ``token`` to be a string.

    Attempt 9: engine wrote 320 sidecar completions, but every OpenHands traj
    is ``ConversationErrorEvent`` (``content.N.token Input should be a valid
    string``). Those JSON files have no ``TokenEvent``s and cannot be replayed
    into ``ppo_train``. Keep ``token_id`` for §5.4; add the decoded piece so
    the agent loop can proceed.

    A short ``logprobs`` used to be padded with ``0.0``. That is a valid-looking
    logprob -- probability 1 -- for a token nobody scored, and under speculative
    decoding it was most of the completion. It raises now: a caller that cannot
    say what a token was sampled under must not be able to claim certainty
    about it.
    """
    if len(logprobs) != len(token_ids):
        raise ValueError(
            f"{len(token_ids)} tokens against {len(logprobs)} logprobs; "
            "the engine owes one logprob per sampled token"
        )
    content: list[dict[str, Any]] = []
    for i, tid in enumerate(token_ids):
        tok = int(tid)
        lp = float(logprobs[i])
        if math.isnan(lp):
            raise ValueError(
                f"token {tok} at position {i} carries no sampling-time logprob; "
                "teacher-forced tokens must not be served as policy samples"
            )
        if tokenizer is not None:
            piece = tokenizer.decode([tok], skip_special_tokens=False)
        else:
            piece = str(tok)
        if piece is None:
            piece = ""
        content.append(
            {
                "token": piece,
                "bytes": list(piece.encode("utf-8")),
                "logprob": lp,
                "top_logprobs": None,
                "token_id": tok,
            }
        )
    return content


def _append_logprob_sidecar(kind: str, token_ids: list[int], logprobs: list[float]) -> None:
    """Record what was sampled. Never record a completion it cannot account for.

    ``rollout.logprobs.load_sidecar`` drops any record whose two lists disagree,
    so a mismatch here does not corrupt the train/rollout comparison -- it
    deletes it, silently, and the instrument reports a missing turn rather than
    a broken engine. Refusing to write the record puts the failure at the cause.
    """
    if len(logprobs) != len(token_ids):
        raise ValueError(
            f"{kind}: {len(token_ids)} tokens against {len(logprobs)} logprobs"
        )
    path = os.environ.get("QWEN3_LOGPROB_SIDECAR")
    if not path:
        return
    rec = {
        "kind": kind,
        "token_ids": list(token_ids),
        "logprobs": [float(x) for x in logprobs],
        "n": len(token_ids),
        "source_commit": os.environ.get("QWEN3_SOURCE_COMMIT", ""),
        "t": time.time(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(rec) + "\n")


def _should_join_weight_update_group(backend: Any, world_size: Any, master_port: Any) -> bool:
    """Join SkyRL's custom process group only for a real NCCL/gloo rendezvous.

    CPU tests pass backend='cuda_ipc' and port 0; those must not touch NCCL.
    """
    if str(backend) not in ("nccl", "gloo"):
        return False
    try:
        if int(world_size) <= 1:
            return False
        if int(master_port) <= 0:
            return False
    except (TypeError, ValueError):
        return False
    return True


PARKED_KV_SHARE = 0.45
FREE_BLOCK_WATERMARK = 0.10


def _parked_kv_budget(engine: Engine, share: float = PARKED_KV_SHARE) -> int:
    """How many physical blocks resident sessions may sit on.

    The scheduler only ever preempts a *running* request, so a paused session
    keeps its blocks until something finishes it. Park too much and an incoming
    prefill has nowhere to go and the scheduler raises rather than reclaiming.
    The budget leaves the majority of the pool for requests that are working.
    Count is blocks, not tokens: prefix-cache sharing makes a token sum overstate
    the pages a parked session actually pins.
    """
    return max(1, int(share * engine.block_manager.num_blocks))


class Qwen3InferenceEngine(InferenceEngineInterface):
    """In-process engine. CUDA IPC weight updates; KV dropped after every update."""

    def __init__(self, engine: Engine, *, tokenizer: Any | None = None):
        self.engine = engine
        self.tokenizer = tokenizer
        self._weight_comm: dict[str, Any] | None = None
        self._model_update_group = None
        self._ipc_keepalives: list[Any] = []
        self._session_kv = session_kv_enabled()
        # Blocks are the binding bound: a parked session pins at least one, so
        # the block budget caps the count as well. The count cap is kept only so
        # that an engine reporting zero-block sessions cannot grow the dict.
        parked_blocks = _parked_kv_budget(engine)
        self._sessions = SessionCache(max_blocks=parked_blocks, max_sessions=parked_blocks)
        self._driver = EngineDriver(engine)
        # Prompt per in-flight request, so the turn can be parked under the
        # sequence its KV actually covers once the engine thread finishes it.
        self._pending_prompt: dict[int, list[int]] = {}
        self._first_turn_at: float | None = None
        self._last_turn_at: float = 0.0
        # QWEN3_BATCHING=0 admits one turn at a time. Not a tuning knob: it is
        # the control arm, and it has to run through the same driver and the
        # same instrument as the batched one or the two spans are not comparable.
        self._one_at_a_time = (
            asyncio.Lock() if os.environ.get("QWEN3_BATCHING", "1") == "0" else None
        )

    def tp_size(self) -> int:
        return 1

    def pp_size(self) -> int:
        return 1

    def dp_size(self) -> int:
        return 1

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        ids = input_batch.get("prompt_token_ids")
        if not ids:
            prompts = input_batch.get("prompts")
            if not prompts or self.tokenizer is None:
                raise ValueError("generate requires prompt_token_ids (or prompts plus a tokenizer)")
            ids = [
                _apply_chat_template_ids(self.tokenizer, p, add_generation_prompt=True)
                for p in prompts
            ]
        params, max_tokens = _sampling_from_dict(input_batch.get("sampling_params"))
        response_ids: list[list[int]] = []
        responses: list[str] = []
        stops: list[str] = []
        response_logprobs: list[list[float]] = []
        for prompt in ids:
            out = self.engine.generate(list(prompt), max_tokens=max_tokens, sampling=params, ignore_eos=True)
            response_ids.append(out)
            stored = getattr(self.engine, "last_completion_logprobs", None) or {}
            lps = next(iter(stored.values()), [])
            response_logprobs.append(lps)
            _append_logprob_sidecar("generate", out, lps)
            if self.tokenizer is not None:
                responses.append(self.tokenizer.decode(out, skip_special_tokens=True))
            else:
                responses.append("")
            stops.append("length")
        return {
            "responses": responses,
            "response_ids": response_ids,
            "stop_reasons": stops,
            "response_logprobs": response_logprobs,
        }

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json") if "json" in request_payload else request_payload
        messages = body.get("messages") or []
        if self.tokenizer is None:
            return {
                "error": {
                    "message": "tokenizer required for chat_completion",
                    "type": "invalid_request",
                    "code": 400,
                }
            }
        extra = body.get("chat_template_kwargs") or {}
        if isinstance(body.get("extra_body"), dict):
            extra = {**extra, **(body["extra_body"].get("chat_template_kwargs") or {})}
        messages = _normalize_chat_messages(list(messages))
        tools = _normalize_tools(body.get("tools"))
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": bool(extra.get("add_generation_prompt", True)),
            "enable_thinking": bool(extra.get("enable_thinking", False)),
        }
        if tools is not None:
            template_kwargs["tools"] = tools
        ids = _apply_chat_template_ids(self.tokenizer, messages, **template_kwargs)
        params, _ignored_max = _sampling_from_dict(body)
        max_tokens = _chat_max_tokens(body)
        completion, lps = await self._run_turn(ids, max_tokens=max_tokens, sampling=params)
        _append_logprob_sidecar("chat_completion", completion, lps)
        text = self.tokenizer.decode(completion, skip_special_tokens=False)
        tool_calls = parse_qwen_tool_calls(text)
        message = assistant_chat_message(text, tool_calls)
        created = int(time.time())
        # OpenHands TokenEvent (return_token_ids=True) reads
        # raw_response["prompt_token_ids"] and
        # choices[0]["provider_specific_fields"]["token_ids"]. LiteLLM copies
        # unknown top-level keys onto ModelResponse and unknown choice keys
        # into provider_specific_fields. Missing prompt_token_ids was attempt
        # 10: Action/Observation then AttributeError on every traj.
        return {
            "id": f"chatcmpl-qwen3-{created}",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or "qwen3_runtime",
            "prompt_token_ids": ids,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "token_ids": completion,
                    "provider_specific_fields": {"token_ids": completion},
                    "logprobs": {"content": _openai_logprob_content(self.tokenizer, completion, lps)},
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(completion),
                "total_tokens": len(ids) + len(completion),
            },
        }

    async def _run_turn(
        self, ids: list[int], *, max_tokens: int, sampling: SamplingParams
    ) -> tuple[list[int], list[float]]:
        if self._one_at_a_time is not None:
            async with self._one_at_a_time:
                return await self._turn(ids, max_tokens=max_tokens, sampling=sampling)
        return await self._turn(ids, max_tokens=max_tokens, sampling=sampling)

    async def _turn(
        self, ids: list[int], *, max_tokens: int, sampling: SamplingParams
    ) -> tuple[list[int], list[float]]:
        """Generate one agent turn, batched with whatever else is in flight.

        Awaiting here rather than blocking is the whole point: several
        trajectories call this at once, and only if the event loop is free can
        they reach the engine together and share a decode step. Continuing the
        session that already holds the conversation is the other half -- without
        it the engine re-prefills the whole history every turn.
        """
        try:
            turn = await self._driver.run_turn(
                lambda: self._admit(ids, max_tokens=max_tokens, sampling=sampling),
                on_done=self._park_or_release,
            )
        except RuntimeError as exc:
            if "KV pool" not in str(exc) or not self._session_kv:
                raise
            # Resident sessions have squeezed out the request that is actually
            # trying to run. Hand the pool back and recompute: a cache is not
            # worth failing a training step over.
            self._driver.run_on_engine(self._drop_sessions)
            turn = await self._driver.run_turn(
                lambda: self._admit(ids, max_tokens=max_tokens, sampling=sampling),
                on_done=self._park_or_release,
            )
        self._note_turn()
        return turn.tokens, turn.logprobs

    def _note_turn(self) -> None:
        """Keep an on-disk snapshot current so generation can be read as it ends.

        These numbers used to surface only through the print in ``sleep``, which
        SkyRL calls after policy training -- half an hour of work that says
        nothing about generation, and which a measurement run has no reason to
        wait for. A small JSON write per turn costs nothing next to a forward
        pass and makes the generate phase readable the moment it finishes.
        """
        now = time.perf_counter()
        if self._first_turn_at is None:
            self._first_turn_at = now
        self._last_turn_at = now
        path = os.environ.get("QWEN3_STATS_FILE")
        if not path:
            return
        payload = {
            "session": self.session_report(),
            "batching": self._driver.report(),
            "generation_span_s": round(self._last_turn_at - self._first_turn_at, 2),
            # Read off the engine, not off this adapter's arguments. Speculation
            # was hardcoded off on this path and on for every published
            # rollout-track number, and no artifact recorded which one produced
            # it. A measurement now carries its configuration.
            "engine": self.engine.config_report(),
        }
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w") as handle:
                json.dump(payload, handle)
            os.replace(tmp, path)
        except OSError:
            pass  # a measurement aid must never take a training step down

    def _admit(self, ids: list[int], *, max_tokens: int, sampling: SamplingParams) -> int:
        """Put one turn on the scheduler. Runs on the engine thread."""
        claim = self._sessions.claim(ids) if self._session_kv else None
        if claim is not None:
            self.engine.resume_request(
                claim.request_id,
                claim.suffix,
                max_tokens,
                hold_kv=True,
                sampling=sampling,
                ignore_eos=False,
                stop_token_ids=STOP_TOKEN_IDS,
            )
            self._pending_prompt[claim.request_id] = ids
            return claim.request_id
        request_id = self.engine.add_request(
            ids,
            max_tokens=max_tokens,
            sampling=sampling,
            ignore_eos=False,
            stop_token_ids=STOP_TOKEN_IDS,
            hold_kv=self._session_kv,
        )
        self._pending_prompt[request_id] = ids
        return request_id

    def _park_or_release(self, request_id: int, tokens: list[int]) -> None:
        """Retire one turn. Runs on the engine thread, before the next is admitted."""
        ids = self._pending_prompt.pop(request_id, None)
        if not self._session_kv or ids is None:
            return
        req = self.engine._requests.get(request_id)
        n_blocks = len(getattr(req, "block_table", None) or [])
        self._retire(self._sessions.park(request_id, ids + tokens, n_blocks))
        self._evict_for_free_watermark()

    def _evict_for_free_watermark(self) -> None:
        """Keep a slice of the pool free so the next prefill can allocate.

        The 45% parked-block cap is a static bound. This is the live one: if
        free pages have already dropped under 10% of the pool, drop the oldest
        parked session until they have not, or until nothing is parked.
        """
        manager = getattr(self.engine, "block_manager", None)
        if manager is None or not self._session_kv:
            return
        num_blocks = getattr(manager, "num_blocks", None)
        if not num_blocks:
            return
        min_free = max(1, int(FREE_BLOCK_WATERMARK * num_blocks))
        while True:
            num_free = getattr(manager, "num_free_blocks", None)
            if num_free is None or num_free >= min_free:
                break
            extra = self._sessions.evict_oldest()
            if extra is None:
                break
            before = num_free
            self._retire([extra])
            after = getattr(manager, "num_free_blocks", None)
            if after is None or after <= before:
                break

    def _retire(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self.engine.finish_request(request_id)

    def _drop_sessions(self) -> None:
        """Forget every resident session, e.g. because the KV pool was dropped.

        A session still on the books after an invalidation would be resumed onto
        a block table that no longer describes anything.
        """
        self._retire(self._sessions.drop_all())

    def session_report(self) -> dict[str, float | int]:
        return self._sessions.report()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json") if "json" in request_payload else request_payload
        prompt = body.get("prompt") or ""
        if self.tokenizer is None:
            return {"error": {"message": "tokenizer required for completion", "type": "invalid_request"}}
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        params, max_tokens = _sampling_from_dict(body)
        max_tokens = int(body.get("max_tokens") or max_tokens)
        completion = self.engine.generate(list(ids), max_tokens=max_tokens, sampling=params)
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return {
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                    "token_ids": completion,
                }
            ]
        }

    async def sleep(self, *args: Any, **kwargs: Any):
        level = kwargs.get("level", args[0] if args else 1)
        # Generation is over. Take the engine thread down first: everything
        # after this point -- dropping KV, parking weights, the IPC copy that
        # follows -- has to be the only thing touching the engine.
        self._driver.stop()
        # This is where the reuse and the batching are worth reporting: the log
        # is the only place a training run shows them.
        if self._session_kv:
            print(f"[qwen3] session KV: {json.dumps(self.session_report())}", flush=True)
        print(f"[qwen3] batching: {json.dumps(self._driver.report())}", flush=True)
        self._drop_sessions()
        return self.engine.sleep(level=int(level))

    async def wake_up(self, *args: Any, **kwargs: Any):
        tags = kwargs.get("tags", args[0] if args else None)
        self.engine.wake_up(tags=tags)

    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        # Weight *tensors* still move over CUDA IPC (the remote NCCL one-tensor
        # path in §3 is not implemented). SkyRL's colocated FSDP worker still
        # gathers this call with trainer-side init_custom_process_group
        # (rank 0, world_size = engines+1). Returning without joining leaves
        # that NCCL rendezvous hung forever — that was A5 attempt 3.
        self._weight_comm = {
            "master_addr": master_addr,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
            "path": "cuda_ipc",
        }
        if _should_join_weight_update_group(backend, world_size, master_port):
            # Join on this actor thread: NCCL needs the CUDA context that
            # loaded the model. asyncio.to_thread would put NCCL on a worker
            # thread without that context.
            self._join_weight_update_group(
                master_addr,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
                override_existing,
            )
        return self._weight_comm

    def _join_weight_update_group(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool,
    ) -> None:
        import torch
        import torch.distributed as dist
        from torch.distributed import destroy_process_group

        from skyrl_train.distributed.utils import init_custom_process_group
        from skyrl_train.utils.utils import get_tcp_url

        if getattr(self, "_model_update_group", None) is not None:
            if override_existing:
                destroy_process_group(self._model_update_group)
                self._model_update_group = None
            else:
                return

        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())

        rank = int(rank_offset)
        if dist.is_initialized():
            rank = int(dist.get_rank()) + int(rank_offset)
        print(
            f"[qwen3] join weight-update group backend={backend} rank={rank} "
            f"world_size={world_size} group={group_name} "
            f"{master_addr}:{master_port}",
            flush=True,
        )
        self._model_update_group = init_custom_process_group(
            backend=backend,
            init_method=get_tcp_url(str(master_addr), int(master_port)),
            world_size=int(world_size),
            rank=rank,
            group_name=str(group_name),
        )
        print("[qwen3] weight-update group joined", flush=True)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        from qwen3_runtime.rollout.lifecycle import tensor_from_ipc_handle

        names = request["names"]
        if not names:
            raise ValueError("Update weight request should have at least one entry in 'names'")
        # The IPC copy lands in the model's own tensors, so it cannot run while
        # sleep has them parked on host. SkyRL does wake weights first, but a
        # caller that skips that would otherwise write into host tensors and
        # silently leave the device stale.
        reload_weights = getattr(getattr(self.engine, "runner", None), "reload_weights", None)
        if reload_weights is not None:
            reload_weights()
        extras = request.get("extras") or []
        items: list[tuple[str, object]] = []
        if extras and "ipc_handles" in extras[0]:
            handles = [extra["ipc_handles"] for extra in extras]
            packed = bool(request.get("packed"))
            if packed:
                raise NotImplementedError("packed CUDA IPC is not required for the single-GPU 4B path")
            for name, handle in zip(names, handles):
                reconstructed = tensor_from_ipc_handle(handle)
                self._ipc_keepalives.append(reconstructed)
                items.append((name, reconstructed))
        elif extras and "tensors" in extras[0]:
            # Test/local path: already-on-device tensors, still goes through IPC-shaped apply + KV drop.
            for name, tensor in zip(names, extras[0]["tensors"]):
                items.append((name, tensor))
        else:
            raise ValueError(
                "colocated engine requires extras[].ipc_handles (CUDA IPC). "
                "NCCL one-tensor-per-call is the remote path and is not implemented here."
            )
        # The IPC copy writes straight into the model's tensors, so nothing may
        # be mid-forward. apply_named_weights then drops all KV, which voids
        # every resident session.
        self._driver.stop()
        self._drop_sessions()
        applied = self.engine.apply_named_weights(items)
        # Copy is done. Drop reconstructed IPC tensors so a full-model sync
        # does not keep a second copy of every parameter on this GPU.
        self._ipc_keepalives.clear()
        return applied

    async def teardown(self):
        self._driver.stop()
        self.engine.sleep(level=1)
        self._ipc_keepalives.clear()
        if self._model_update_group is not None:
            try:
                from torch.distributed import destroy_process_group

                destroy_process_group(self._model_update_group)
            except Exception:
                pass
            self._model_update_group = None

    async def reset_prefix_cache(self):
        n = self.engine.block_manager.cache_blocks
        if n:
            self.engine.block_manager._evict_unused(n)

    async def abort_generation(self) -> None:
        self._driver.stop()  # strands in-flight turns on purpose; that is the abort
        self._pending_prompt.clear()
        self._sessions.drop_all()  # abort frees their blocks; do not finish them twice
        self.engine.abort_generation()
