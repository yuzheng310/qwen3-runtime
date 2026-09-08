"""Park / restore BF16 weights so a colocated trainer can take the VRAM."""

from __future__ import annotations

from typing import Any

import torch


def offload_weights(runner: Any, *, keep_a_host_copy: bool = False) -> None:
    """Give the colocated trainer the ~8 GiB of weights nothing is reading.

    Default discards bytes: SkyRL overwrites every parameter over CUDA IPC
    before the next generate, so a host copy only buys a restore of data that
    is about to be thrown away. ``keep_a_host_copy`` is for a wake without
    a weight update; host pages stay unpinned on purpose.
    """
    if runner._offloaded_from is not None or runner._released_params:
        return
    param = next(runner.model.parameters(), None)
    if param is None or param.device.type != "cuda":
        return
    device = param.device
    if keep_a_host_copy:
        runner._offloaded_from = device
        runner.model.to("cpu")
    else:
        for name, tensor in runner.model.named_parameters():
            runner._released_params[name] = (tuple(tensor.shape), tensor.dtype, device)
            tensor.data = torch.empty(0, dtype=tensor.dtype, device=device)
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reload_weights(runner: Any) -> None:
    """Put the weights back on device, or make room for the ones arriving."""
    if runner._offloaded_from is not None:
        device, runner._offloaded_from = runner._offloaded_from, None
        runner.model.to(device)
        return
    if not runner._released_params:
        return
    params = dict(runner.model.named_parameters())
    for name, (shape, dtype, device) in runner._released_params.items():
        # NaN rather than empty: fused qkv is one tensor filled by three HF
        # names, so "every destination got written" is a byte check.
        params[name].data = torch.full(shape, float("nan"), dtype=dtype, device=device)
    runner._released_params.clear()
    runner._weights_unverified = True


def assert_weights_usable(runner: Any) -> None:
    if runner._offloaded_from is not None or runner._released_params:
        raise RuntimeError("weights are not on device; wake_up must run before generate")
    if not runner._weights_unverified:
        return
    blank = [name for name, p in runner.model.named_parameters() if torch.isnan(p).any()]
    if blank:
        raise RuntimeError(
            f"{len(blank)} parameter(s) released on sleep still hold unwritten "
            f"bytes after the weight update, e.g. {sorted(blank)[:3]}; generating "
            "now would sample from uninitialised memory"
        )
    runner._weights_unverified = False
