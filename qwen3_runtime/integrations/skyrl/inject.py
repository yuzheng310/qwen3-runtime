"""Inject Qwen3InferenceEngine into SkyRL's local-engine factory.

SkyRL's create_ray_wrapped_inference_engines imports vllm only when
backend=="vllm". Patching the factory lets the GRPO trainer keep using
get_sampling_params_for_backend("vllm", ...) while the actual engine is ours.

The factory returns a Ray-wrapped actor (0.2 GPU when colocated), not an
in-process engine on the CPU entrypoint actor.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

# Names we do not own, and so cannot recognize by prefix.
_FOREIGN_ENV = (
    "PATH",
    "PYTHONPATH",
    "TMPDIR",
    "PIP_CACHE_DIR",
    "GITHUB_CLONE_MIRROR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_0",
    "GIT_CONFIG_VALUE_0",
    "RAY_health_check_failure_threshold",
    "RAY_health_check_timeout_ms",
    "RAY_health_check_period_ms",
    "RAY_object_store_memory",
    "RAY_total_memory_bytes",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MALLOC_ARENA_MAX",
)


def collect_worker_env(environ: dict[str, str]) -> dict[str, str]:
    """Pick the variables a Ray worker needs out of an environment.

    Everything this project configures is QWEN3_-prefixed, so the prefix is the
    rule and the list is only for names we do not own. Enumerating ours by hand
    cost four silent failures: a variable missing from the list does not raise,
    it just never reaches the actor, so the feature it controls is quietly off
    in the one process where it matters.
    """
    picked = {k: v for k, v in environ.items() if v and k.startswith("QWEN3_")}
    for key in _FOREIGN_ENV:
        val = environ.get(key)
        if val:
            picked[key] = val
    return picked


def make_local_engines(cfg: Any, colocate_pg: Any, tokenizer: Any) -> list:
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    del tokenizer
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine

    from qwen3_runtime.integrations.skyrl.ray_actor import Qwen3RayActor

    use_hybrid = colocate_pg is not None
    num_gpus = 0.2 if use_hybrid else 1.0
    sched = None
    if colocate_pg is not None:
        sched = PlacementGroupSchedulingStrategy(
            placement_group=colocate_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
    actor = Qwen3RayActor.options(
        num_cpus=num_gpus,
        num_gpus=num_gpus,
        scheduling_strategy=sched,
    ).remote(
        str(cfg.trainer.policy.model.path),
        max_num_seqs=int(cfg.generator.get("max_num_seqs") or 8),
        max_num_batched_tokens=int(cfg.generator.get("max_num_batched_tokens") or 131072),
        enable_prefix_cache=bool(cfg.generator.get("enable_prefix_caching", False)),
        logprob_path=os.environ.get("QWEN3_LOGPROB_SIDECAR", ""),
        source_commit=os.environ.get("QWEN3_SOURCE_COMMIT", ""),
    )
    engines = [RayWrappedInferenceEngine(actor)]
    if bool(cfg.trainer.placement.colocate_all):
        ray.get(actor.sleep.remote(level=1))
    return engines


def wrap_async_with_semaphore(fn, limit: int):
    """Cap in-flight awaits. Used so 64 CodeScout coroutines do not spawn 64 OpenHands clones."""
    limit = max(1, int(limit))
    state: dict[str, Any] = {"sem": None}

    async def wrapped(*args, **kwargs):
        if state["sem"] is None:
            state["sem"] = asyncio.Semaphore(limit)
        async with state["sem"]:
            return await fn(*args, **kwargs)

    wrapped._qwen3_oh_limit = limit  # type: ignore[attr-defined]
    return wrapped


def patch_codescout_rollout_concurrency(limit: int | None = None) -> int:
    """Do not edit CodeScout. Wrap ``code_search_loop`` so ``asyncio.gather`` of 64
    trajectories only runs ``limit`` ``init_and_run`` Ray tasks at once.

    AutoDL cgroup is ~92 GiB; 64 concurrent OpenHands clones killed Ray (attempt 8).
    Trajectory count stays 8×8=64.
    """
    if limit is None:
        limit = int(os.environ.get("QWEN3_OH_CONCURRENCY", "4"))
    from src.generator.code_search_generator import CodeSearchGenerator

    orig = CodeSearchGenerator.code_search_loop
    if getattr(orig, "_qwen3_oh_limited", False):
        return int(getattr(orig, "_qwen3_oh_limit", limit))
    wrapped = wrap_async_with_semaphore(orig, limit)
    wrapped._qwen3_oh_limited = True  # type: ignore[attr-defined]
    CodeSearchGenerator.code_search_loop = wrapped
    return limit


def patch_codescout_replay_traj(traj_dir: str | None = None) -> str:
    """Train from on-disk CodeScout JSON. Empty ``QWEN3_REPLAY_TRAJ_DIR`` is a no-op."""
    from qwen3_runtime.integrations.skyrl.replay_traj import patch_codescout_replay_traj as _patch

    return _patch(traj_dir)


def patch_skyrl_factory() -> None:
    import skyrl_train.entrypoints.main_base as main_base

    from qwen3_runtime.integrations.skyrl.worker_setup import apply_skyrl_runtime_patches

    apply_skyrl_runtime_patches()
    main_base.create_ray_wrapped_inference_engines_from_config = make_local_engines


def patch_initialize_ray() -> None:
    """Keep PYTHONPATH / peak-file env on Ray workers; optional setup hook."""
    import os

    import ray as ray_lib
    from skyrl_train.utils.ppo_utils import sync_registries
    from skyrl_train.utils.utils import prepare_runtime_environment
    import skyrl_train.utils as skyrl_utils
    import skyrl_train.utils.utils as skyrl_utils_mod

    def initialize_ray(cfg):
        env_vars = prepare_runtime_environment(cfg)
        env_vars.update(collect_worker_env(dict(os.environ)))
        runtime_env: dict = {"env_vars": env_vars}
        init_kwargs: dict[str, Any] = {}
        store = os.environ.get("RAY_object_store_memory")
        if store:
            init_kwargs["object_store_memory"] = int(store)
        mem = os.environ.get("RAY_total_memory_bytes")
        if mem:
            init_kwargs["_memory"] = int(mem)
        try:
            runtime_env["worker_process_setup_hook"] = "qwen3_runtime.integrations.skyrl.worker_setup.init"
            ray_lib.init(runtime_env=runtime_env, **init_kwargs)
        except Exception:
            ray_lib.init(runtime_env={"env_vars": env_vars}, **init_kwargs)
        sync_registries()

    skyrl_utils.initialize_ray = initialize_ray
    skyrl_utils_mod.initialize_ray = initialize_ray
