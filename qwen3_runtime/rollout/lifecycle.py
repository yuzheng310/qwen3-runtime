"""Copy named tensors into the fused Qwen3 module. CUDA IPC reconstruct helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import nn

from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM


def _copy_(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.shape != src.shape:
        raise ValueError(f"shape mismatch: dst {tuple(dst.shape)} src {tuple(src.shape)}")
    dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _copy_into(dst: torch.Tensor, src: torch.Tensor, sl: slice | None = None) -> None:
    src = src.to(device=dst.device, dtype=dst.dtype)
    if sl is None:
        _copy_(dst, src)
    else:
        dst.data[sl].copy_(src)


def _copy_hf_layer(layer, tail: str, tensor: torch.Tensor, cfg) -> None:
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    mapping = {
        "self_attn.o_proj.weight": layer.attn.o_proj.weight,
        "self_attn.q_norm.weight": layer.attn.q_norm.weight,
        "self_attn.k_norm.weight": layer.attn.k_norm.weight,
        "input_layernorm.weight": layer.input_layernorm.weight,
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        "mlp.down_proj.weight": layer.mlp.down_proj.weight,
    }
    if tail in mapping:
        _copy_(mapping[tail], tensor)
        return
    slices = {
        "self_attn.q_proj.weight": (layer.attn.qkv_proj.weight, slice(0, q_dim)),
        "self_attn.k_proj.weight": (layer.attn.qkv_proj.weight, slice(q_dim, q_dim + kv_dim)),
        "self_attn.v_proj.weight": (layer.attn.qkv_proj.weight, slice(q_dim + kv_dim, None)),
        "mlp.gate_proj.weight": (layer.mlp.gate_up_proj.weight, slice(0, cfg.intermediate_size)),
        "mlp.up_proj.weight": (layer.mlp.gate_up_proj.weight, slice(cfg.intermediate_size, None)),
    }
    if tail not in slices:
        raise KeyError(f"unknown weight name tail {tail!r}")
    dst, sl = slices[tail]
    _copy_into(dst, tensor, sl)


def copy_named_weight(model: Qwen3ForCausalLM, name: str, tensor: torch.Tensor) -> None:
    """Write one tensor. Accepts native fused names or HuggingFace Qwen3 names."""
    native = dict(model.named_parameters())
    if name in native:
        _copy_(native[name], tensor)
        return
    if name == "model.embed_tokens.weight":
        _copy_(model.embed_tokens.weight, tensor)
        return
    if name == "model.norm.weight":
        _copy_(model.norm.weight, tensor)
        return
    if name == "lm_head.weight":
        dst = model.embed_tokens.weight if model.cfg.tie_word_embeddings else model.lm_head.weight
        _copy_(dst, tensor)
        return
    prefix = "model.layers."
    if not name.startswith(prefix):
        raise KeyError(f"unknown weight name {name!r}")
    rest = name[len(prefix) :]
    layer_s, _, tail = rest.partition(".")
    try:
        _copy_hf_layer(model.layers[int(layer_s)], tail, tensor, model.cfg)
    except KeyError as exc:
        raise KeyError(f"unknown weight name {name!r}") from exc


def apply_named_weights(model: nn.Module, items: Iterable[tuple[str, torch.Tensor]]) -> list[str]:
    applied: list[str] = []
    if not isinstance(model, Qwen3ForCausalLM):
        params = dict(model.named_parameters())
        for name, tensor in items:
            if name not in params:
                raise KeyError(name)
            _copy_(params[name], tensor)
            applied.append(name)
        return applied
    for name, tensor in items:
        copy_named_weight(model, name, tensor)
        applied.append(name)
    return applied


def gpu_uuid(device: torch.device | int | None = None) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA IPC requires a CUDA device")
    index = torch.cuda.current_device() if device is None else int(torch.device(device).index or 0)
    return str(torch.cuda.get_device_properties(index).uuid)


def ipc_handles_for_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Producer-side handle, same shape as SkyRL vLLM extras[].ipc_handles."""
    if tensor.device.type != "cuda":
        raise ValueError("CUDA IPC handle requires a CUDA tensor")
    tensor = tensor.contiguous()
    from torch.multiprocessing.reductions import reduce_tensor

    func, args = reduce_tensor(tensor)
    return {gpu_uuid(tensor.device): (func, args)}


def tensor_from_ipc_handle(handle: Mapping[str, Any], *, device_id: int | None = None) -> torch.Tensor:
    uid = gpu_uuid()
    packed = handle.get(uid)
    if packed is None:
        if len(handle) == 1:
            packed = next(iter(handle.values()))
        else:
            raise KeyError(f"no IPC handle for GPU uuid {uid}; have {list(handle)}")
    func, args = packed
    list_args = list(args)
    if device_id is None:
        device_id = torch.cuda.current_device()
    if len(list_args) > 6:
        list_args[6] = device_id
    try:
        return func(*list_args)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\n{_ipc_failure_hint(exc)}") from exc


def _ipc_failure_hint(exc: BaseException) -> str:
    """Translate the two IPC failures that are environment, not code.

    ``pidfd_getfd`` only shows up once the producer allocated through CUDA VMM
    (expandable_segments), which exports memory as a file descriptor instead of
    a cudaIpcMemHandle. Grabbing that descriptor needs ptrace permission on a
    sibling Ray actor, which a container without CAP_SYS_PTRACE will refuse.
    """
    import os

    text = str(exc)
    if "pidfd_getfd" not in text:
        return "The IPC producer must be alive and on the same GPU as this engine."
    hints = [
        "CUDA IPC could not adopt the producer's memory.",
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') or '<unset>'}",
    ]
    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        hints.append(
            "expandable_segments puts the allocator on CUDA VMM, whose handles "
            "travel as file descriptors. Unset it to go back to cudaIpcMemHandle."
        )
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as fh:
            scope = fh.read().strip()
        if scope != "0":
            hints.append(
                f"yama/ptrace_scope={scope}: a process may only take descriptors "
                "from its own descendants, and Ray actors are siblings."
            )
    except OSError:
        pass
    return " ".join(hints)


def snapshot_cuda_memory() -> dict[str, int | None]:
    """Bytes currently allocated and the process peak. None on CPU."""
    if not torch.cuda.is_available():
        return {"allocated": None, "max_allocated": None, "reserved": None}
    device = torch.cuda.current_device()
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
    }


def wake_wants_kv(tags: object | None) -> bool:
    """True unless SkyRL asked for a weights-only wake (KV stays released)."""
    if tags is None:
        return True
    if isinstance(tags, str):
        names = {tags.lower()}
    else:
        try:
            names = {str(t).lower() for t in tags}
        except TypeError:
            return True
    if not names:
        return True
    return bool(names & {"kv", "kv_cache", "kvcache"})


def park_live_sessions(engine: Any) -> None:
    """Move waiting/running/paused onto paused with no KV, still resumable."""
    from qwen3_runtime.engine.request import RequestStatus

    live = list(engine._requests.values())
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine.scheduler.paused.clear()
    for req in live:
        engine.block_manager.deallocate(req)
        req.reset_kv_state()
        req.status = RequestStatus.PAUSED
        engine.scheduler.paused[req.request_id] = req


def sleep_engine(engine: Any, level: int = 1) -> dict[str, int | None]:
    """Release the KV pool. Weights stay on device.

    Mid-trajectory sessions stay registered as restarts. ``level`` matches
    SkyRL/vLLM's kwargs, and every level drops KV — that part is not
    optional, because KV from the previous policy is not valid under the
    next one.

    Sample packing left enough spare VRAM beside a sleeping 4B engine that
    moving the weights is not worth the host copy. ``PagedRunner.offload_weights``
    still exists for a caller that is genuinely out of room.
    """
    del level
    park_live_sessions(engine)
    engine.block_manager.reset()
    engine.runner.release_kv_pool()
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    engine._asleep = True
    snap = snapshot_cuda_memory()
    engine.last_sleep_memory = snap
    return snap


def wake_engine(engine: Any, tags: object | None = None) -> None:
    """Restore weights and/or the KV pool.

    SkyRL colocated training calls ``wake_up(tags=["weights"])`` *before*
    CUDA-IPC sync, while the policy model is still on GPU. Rebuilding the
    KV pool there is an OOM. ``tags=["kv_cache"]`` rebuilds the pool after
    the policy has been offloaded. ``tags=None`` still means both.
    """
    engine.runner.reload_weights()
    if wake_wants_kv(tags):
        engine.runner.rebuild_kv_pool()
        engine._asleep = False
        return
    print(f"[qwen3] wake_up tags={tags!r}: weights back on device, KV pool not rebuilt", flush=True)


def abort_generation(engine: Any) -> list[int]:
    aborted: list[int] = []
    targets = list(engine.scheduler.waiting) + list(engine.scheduler.running)
    for req in targets:
        rid = req.request_id
        req.finish_reason = "abort"
        engine.scheduler.release(req)
        engine._requests.pop(rid, None)
        aborted.append(rid)
        if req.block_table:
            raise RuntimeError(f"abort left request {rid} with a live block table")
    engine.last_aborted_ids = aborted
    return aborted
