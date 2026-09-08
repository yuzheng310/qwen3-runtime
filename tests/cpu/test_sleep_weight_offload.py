"""Sleep parks the weights too, because a colocated trainer needs that memory.

The clean re-run died allocating its last training microbatch with 1.26 GiB
free while the sleeping engine still held ~10 GiB, most of it BF16 weights that
nothing reads until the trainer hands new ones back over CUDA IPC.
"""

import torch


class _Runner:
    """Enough of PagedRunner to exercise the release/restore contract on CPU."""

    def __init__(self, device="cpu"):
        self.model = torch.nn.Linear(4, 4).to(device)
        self._offloaded_from = None
        self._released_params = {}
        self._weights_unverified = False
        self.pool = object()
        self.block_manager = object()


def _real_runner():
    from qwen3_runtime.engine.model_runner import PagedRunner

    runner = _Runner()
    for name in (
        "offload_weights",
        "reload_weights",
        "rebuild_kv_pool",
        "assert_weights_usable",
    ):
        setattr(runner, name, getattr(PagedRunner, name).__get__(runner))
    return runner


def _released_runner():
    """A runner whose weights were dropped on sleep, as on a real CUDA device."""
    runner = _real_runner()
    device = torch.device("cpu")
    for name, tensor in runner.model.named_parameters():
        runner._released_params[name] = (tuple(tensor.shape), tensor.dtype, device)
        tensor.data = torch.empty(0, dtype=tensor.dtype, device=device)
    return runner


def test_offload_is_a_noop_when_the_weights_are_not_on_a_gpu():
    runner = _real_runner()
    runner.offload_weights()
    assert runner._offloaded_from is None and not runner._released_params
    runner.reload_weights()
    runner.assert_weights_usable()


def test_reload_without_an_offload_is_a_noop():
    runner = _real_runner()
    runner.reload_weights()
    assert runner._offloaded_from is None and not runner._released_params
    runner.assert_weights_usable()


def test_offload_is_idempotent_and_keeps_the_first_device():
    runner = _real_runner()
    runner._offloaded_from = torch.device("cuda:0")
    runner.offload_weights()
    assert runner._offloaded_from == torch.device("cuda:0")


def test_released_weights_come_back_as_real_shapes_that_announce_they_are_unwritten():
    runner = _released_runner()
    assert all(p.numel() == 0 for p in runner.model.parameters())

    runner.reload_weights()

    weight = dict(runner.model.named_parameters())["weight"]
    assert weight.shape == (4, 4)
    assert torch.isnan(weight).all()


def test_generating_before_the_trainer_refills_the_weights_is_refused():
    """Discarding on sleep is only safe because this path exists."""
    runner = _released_runner()
    runner.reload_weights()

    try:
        runner.assert_weights_usable()
        raise AssertionError("expected a refusal")
    except RuntimeError as exc:
        assert "unwritten bytes" in str(exc)


def test_a_partly_written_fused_tensor_is_still_refused():
    """The real bug: q/k/v arrive under three names and share one qkv tensor,
    so 'every incoming name was applied' does not mean 'every byte was written'."""
    runner = _released_runner()
    runner.reload_weights()

    params = dict(runner.model.named_parameters())
    params["weight"].data[:2] = 0.5  # only the first slice arrived
    params["bias"].data.fill_(0.0)

    try:
        runner.assert_weights_usable()
        raise AssertionError("expected a refusal while half the tensor is unwritten")
    except RuntimeError as exc:
        assert "weight" in str(exc)


def test_a_complete_weight_update_clears_the_refusal():
    runner = _released_runner()
    runner.reload_weights()

    for tensor in runner.model.parameters():
        tensor.data.fill_(0.5)

    runner.assert_weights_usable()


def test_the_scan_runs_once_per_wake_not_once_per_forward():
    runner = _released_runner()
    runner.reload_weights()
    for tensor in runner.model.parameters():
        tensor.data.fill_(0.5)

    runner.assert_weights_usable()
    assert runner._weights_unverified is False


def test_the_host_copy_path_needs_no_refill_because_nothing_was_thrown_away():
    runner = _real_runner()
    runner.offload_weights(keep_a_host_copy=True)
    runner.reload_weights()

    runner.assert_weights_usable()


def test_rebuilding_the_pool_while_the_weights_are_parked_is_refused():
    """bind() sizes the pool off a parameter, so a parked model would put it in host RAM."""
    runner = _real_runner()
    runner._offloaded_from = torch.device("cuda:0")
    try:
        runner.rebuild_kv_pool()
        raise AssertionError("expected a refusal")
    except RuntimeError as exc:
        assert "reload_weights" in str(exc)


def test_the_pidfd_failure_names_expandable_segments_and_ptrace_scope(monkeypatch):
    """A cryptic pidfd error cost a whole run; it should explain itself next time."""
    from qwen3_runtime.rollout.lifecycle import _ipc_failure_hint

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    hint = _ipc_failure_hint(RuntimeError("pidfd_getfd: Operation not permitted"))
    assert "expandable_segments" in hint
    assert "cudaIpcMemHandle" in hint


def test_unrelated_ipc_failures_are_not_blamed_on_the_allocator():
    from qwen3_runtime.rollout.lifecycle import _ipc_failure_hint

    hint = _ipc_failure_hint(RuntimeError("invalid device ordinal"))
    assert "expandable_segments" not in hint


def test_kv_budget_can_be_pinned_so_two_engines_can_be_compared():
    """vLLM's fraction covers weights+KV+activations; ours took whatever was free."""
    from qwen3_runtime.engine.factory import kv_budget_bytes

    assert kv_budget_bytes(20 * 1024**3) == 20 * 1024**3


def test_a_nonsense_kv_budget_is_refused_rather_than_silently_ignored():
    from qwen3_runtime.engine.factory import kv_budget_bytes

    try:
        kv_budget_bytes(0)
        raise AssertionError("expected a refusal")
    except ValueError as exc:
        assert "positive" in str(exc)
