"""Ray actor wrapping Qwen3InferenceEngine so SkyRL can colocate it on 0.2 GPU."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import ray


def _pin_cuda_visible_devices() -> None:
    ids = ray.get_gpu_ids()
    if ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(ids[0]))


@ray.remote
class Qwen3RayActor:
    def __init__(
        self,
        model_dir: str,
        *,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 131072,
        enable_prefix_cache: bool = False,
        num_speculative_tokens: int | None = None,
        session_cpu_offload: str | None = None,
        cpu_kv_max_bytes: int | None = None,
        cpu_kv_pinned_max_bytes: int | None = None,
        transfer_chunk_bytes: int | None = None,
        logprob_path: str = "",
        source_commit: str = "",
    ):
        _pin_cuda_visible_devices()
        from transformers import AutoTokenizer

        from qwen3_runtime.engine.factory import CODESCOUT_PIN, build_engine
        from qwen3_runtime.integrations.skyrl.inference_engine import (
            Qwen3InferenceEngine,
        )

        if logprob_path:
            os.environ["QWEN3_LOGPROB_SIDECAR"] = logprob_path
        if source_commit:
            os.environ["QWEN3_SOURCE_COMMIT"] = source_commit
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # The training path shipped with speculation hardcoded off. That was how
        # the integration happened to be written, not a decision -- no
        # measurement behind it and no mention in the plan -- while the rollout
        # track measured its per-token numbers with it on. Reachable now so the
        # two paths can be compared on the same setting instead of differing by
        # an accident. QWEN3_SPEC_TOKENS overrides for a measurement arm.
        spec = num_speculative_tokens
        if spec is None:
            spec = int(os.environ.get("QWEN3_SPEC_TOKENS", "0"))
        # QWEN3_* settings are propagated to Ray workers by inject.py.
        # Explicit actor arguments take precedence, including zero budgets.
        offload = session_cpu_offload
        if offload is None:
            offload = os.environ.get("QWEN3_SESSION_CPU_OFFLOAD", "off")
        cpu_bytes = cpu_kv_max_bytes
        if cpu_bytes is None:
            cpu_bytes = int(os.environ.get("QWEN3_CPU_KV_MAX_BYTES", "0"))
        pinned_bytes = cpu_kv_pinned_max_bytes
        if pinned_bytes is None:
            pinned_bytes = int(os.environ.get("QWEN3_CPU_KV_PINNED_MAX_BYTES", "0"))
        chunk_bytes = transfer_chunk_bytes
        if chunk_bytes is None:
            chunk_bytes = int(
                os.environ.get("QWEN3_KV_TRANSFER_CHUNK_BYTES", str(8 * 1024**2))
            )
        # CodeScout-4B pin (rope_theta=5e6). Not the §5.0 tool pin; factory default is Qwen3-4B.
        engine = build_engine(
            model_dir,
            max_num_seqs=int(max_num_seqs),
            max_num_batched_tokens=int(max_num_batched_tokens),
            enable_prefix_cache=bool(enable_prefix_cache),
            num_speculative_tokens=int(spec),
            session_cpu_offload=offload,
            cpu_kv_max_bytes=int(cpu_bytes),
            cpu_kv_pinned_max_bytes=int(pinned_bytes),
            transfer_chunk_bytes=int(chunk_bytes),
            pin_path=CODESCOUT_PIN,
        )
        # Off the engine, not off the arguments above. The accident this line
        # exists for was an argument and an engine disagreeing.
        print(
            f"[qwen3] engine config: {json.dumps(engine.config_report())}", flush=True
        )
        # Print the realized capacity, not the requested one. Comparing this
        # engine against vLLM is only meaningful if both got the same number of
        # KV tokens, and neither side's config knob states that directly.
        blocks = getattr(engine.block_manager, "num_blocks", None)
        block_size = getattr(engine.block_manager, "block_size", None)
        if blocks is not None and block_size is not None:
            print(
                f"[qwen3] KV pool: {blocks} blocks x {block_size} tokens = "
                f"{blocks * block_size} tokens",
                flush=True,
            )
        self._impl = Qwen3InferenceEngine(engine, tokenizer=tokenizer)

    def tp_size(self) -> int:
        return self._impl.tp_size()

    def pp_size(self) -> int:
        return self._impl.pp_size()

    def dp_size(self) -> int:
        return self._impl.dp_size()

    async def generate(self, input_batch):
        return await self._impl.generate(input_batch)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._impl.chat_completion(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._impl.completion(request_payload)

    async def finish_session(self, token_ids: list[int]) -> int:
        return await self._impl.finish_session(token_ids)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._impl.sleep(*args, **kwargs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._impl.wake_up(*args, **kwargs)

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
        return await self._impl.init_weight_update_communicator(
            master_addr,
            master_port,
            rank_offset,
            world_size,
            group_name,
            backend,
            override_existing,
        )

    async def update_named_weights(self, request):
        return await self._impl.update_named_weights(request)

    async def teardown(self):
        return await self._impl.teardown()

    async def reset_prefix_cache(self):
        return await self._impl.reset_prefix_cache()

    async def abort_generation(self) -> None:
        return await self._impl.abort_generation()
