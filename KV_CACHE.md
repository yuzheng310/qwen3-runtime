# Session-aware tiered KV — September 15 update

The runtime supports optional synchronous snapshot or block CPU KV storage and
experimental early asynchronous snapshot D2H. All remain disabled by default.
This update includes the implementation measured at archived source revision
`711635486a5ab1ca155f05c56134261bc99a1da8`; the public repository has its own history.
The results below are archived GPU measurements, not new GPU tests of this release.

## Design

| Responsibility | Implementation | Contract |
|---|---|---|
| Block CPU cache | [block_offload.py](qwen3_runtime/engine/block_offload.py), [block_store.py](qwen3_runtime/kv/block_store.py) | Namespace includes model/weight/KV identity and layout; chained content hashes identify complete blocks, while tail pages have private versions. Copy only missing blocks; retain evictable CPU replicas after restore. |
| Managed host capacity | [host_budget.py](qwen3_runtime/kv/host_budget.py), [session_offload.py](qwen3_runtime/engine/session_offload.py) | Charge storage lifetimes, reservations and slab capacity; pinned memory is a subset of the total budget. Reserve an entire reclamation group before copying. |
| Mixed restore | [block_offload.py](qwen3_runtime/engine/block_offload.py) | Compose a contiguous prefix from GPU and CPU hits, pin references before allocation, reserve next-step capacity, and roll back unsuccessful restoration. A hole stops prefix recovery. |
| Session accounting | [session.py](qwen3_runtime/rollout/session.py), [execution.py](qwen3_runtime/rollout/execution.py) | Count the union of shared physical pages rather than summing logical history lengths; protect reserved continuation claims and reclaim actual free pages. |
| Early async save | [async_offload.py](qwen3_runtime/engine/async_offload.py) | One in-flight snapshot D2H on a separate CUDA stream; events establish completion before ownership release. Resume joins pending work; finish, abort and invalidation drain it. |

The CPU cache is used through retained session manifests. It does not add global
CPU prefix loading for arbitrary new requests, distributed caching or disk storage.
The block backend is synchronous. Async mode only covers early snapshot D2H;
H2D, emergency reclamation and explicit `Engine.offload_request` remain synchronous.
Pageable reservations fall back to synchronous save. Uncertain device completion
retains ownership and terminates the engine instead of reusing possibly live memory.

## Configuration

Pass these arguments through the existing factory and execute turns with
`SessionRollout`. The example selects the block backend; it is not a complete
benchmark reproduction command.

```python
from qwen3_runtime.engine.factory import build_engine
from qwen3_runtime.rollout.execution import SessionRollout

engine = build_engine(
    model_dir,
    max_num_seqs=2,
    max_num_batched_tokens=2048,
    enable_prefix_cache=True,
    session_cpu_offload="sync",
    cpu_kv_backend="block",
    cpu_kv_max_bytes=4 * 1024**3,
    cpu_kv_pinned_max_bytes=4 * 1024**3,
    transfer_chunk_bytes=32 * 1024**2,
)
rollout = SessionRollout(engine)
```

Use `cpu_kv_backend="snapshot"` to return to synchronous snapshots, or
`session_cpu_offload="off"` to disable CPU storage. Defaults are unchanged:
`off`, `snapshot`, `session_offload_early_fraction=0.0`,
`snapshot_mixed_restore=False` and `cpu_kv_slab_bytes=0` (legacy slab sizing).
Existing integrations do not automatically switch to block mode.

For the experimental async path, use `session_cpu_offload="async"`,
`cpu_kv_backend="snapshot"` and a nonzero early fraction (the experiment used
`0.20`). `SessionRollout` drives its step-before-execution trigger; constructing
an Engine alone does not schedule background saves. Async block mode is rejected.
The experiment also enabled mixed snapshot restore and used 8 GiB CPU / 4 GiB
pinned. These are separate experimental settings, not deployment defaults.

## Same-budget comparison with vLLM

Qwen3-4B BF16 on one RTX 4090 D (about 24 GiB), 24 trajectories / 102 turns /
9,871 forced output tokens, synthetic 1 s tool waits, active slots 2 and batch
budget 2,048 tokens. GPU KV was 5,922 blocks × 16 tokens = 13,971,750,912 bytes.
CPU-offload arms shared a 4 GiB managed CPU budget including a 4 GiB pinned limit.
Each arm ran three times. End-to-end time is batch replay completion including
queueing, synthetic tool waits and restore work; model loading and warmup are separate.

| Arm | All end-to-end samples (s) | Median (s) | D2H (GiB) | H2D (GiB) | Prefill tokens |
|---|---|---:|---:|---:|---:|
| vLLM APC (V0), no CPU KV | 155.803, 155.428, 155.875 | 155.803 | 0 | 0 | 794,941 |
| vLLM APC + native offload (V4) | 155.914, 157.479, 157.358 | 157.358 | 113.199 | 0 | 794,941 |
| Runtime snapshot (S4) | 158.217, 158.431, 156.370 | 158.217 | 86.605 | 3.274 | 743,102 |
| Runtime block (T4) | 154.188, 154.103, 153.178 | 154.103 | 64.553 | 3.968 | 738,059 |

Relative to V4, T4 reduced median elapsed time by **2.07%**, D2H by **42.97%**,
and prefill tokens by **7.16%**. Relative to S4, time fell **2.60%** and D2H fell
**25.46%**, while H2D increased. Lower traffic does not guarantee lower latency.
The median per-run p95 TTFT was **34.666 s** for T4 versus **33.018 s** for V4;
this is not an across-the-board latency win.

[Per-run analyzed metrics](bench/results/kv-tiered/comparison.json) retain all 12
rows, full-precision timing and resource values, plus the archived analysis and
original-record hashes. Summaries can be recomputed without GPU access:

```bash
python scripts/summarize_kv_comparison.py
```

### Interpretation and limits

- These are descriptive n=3 observations of one constrained workload, without a
  statistical significance claim or a stable throughput ranking.
- We retain sessions across turns; vLLM completes each turn and submits full history
  through APC. Prefill differences include session retention and cache policy, so
  they cannot all be attributed to CPU deduplication. vLLM already has native
  block caching and asynchronous D2H/H2D.
- V4 performed no H2D in these runs. A separate forced GPU-clear check verified
  restoration works. A single 16 GiB CPU/pinned diagnostic restored 1.307 GiB and
  completed in 152.300 s; it has extra memory and n=1 and is not in this ranking.
- Managed KV budgets do not cap allocator retention, driver staging or RSS. Median
  sampled RSS peaks were S4 **14.058 GiB**, T4 **6.124 GiB**, V4 **6.949 GiB**.
- vLLM's algorithm source was tag **v0.29.0**; the environment version string was
  `0.29.1.dev2+g5f7b949fa`. Native pinned tensor storage was used, rather than the
  default shared-memory mmap layout. The comparison used its compiled execution path.
- Formal vLLM runs did not request logprobs, while this runtime computes selected
  token logprobs. Forced-token adapter costs also differ. Two short greedy probes
  matched tokens, but maximum logprob differences were 0.306686 and 0.001474.
  This is not evidence of RL numerical equivalence or equal sampling costs.
- The public export contains analyzed metrics and hashes, not private token traces,
  checkpoints or operating logs. It supports result inspection and summary
  recomputation, not a self-contained reproduction of the original GPU workload.

## Async snapshot: retained negative result

A separate fixed24 experiment used 8 GiB managed CPU / 4 GiB pinned, with the same
GPU block count and workload dimensions. It is not merged with the 4/4 GiB matrix.

| Snapshot policy | All end-to-end samples (s) | Median (s) |
|---|---|---:|
| S: synchronous pressure saves | 172.070, 173.706, 172.728 | 172.728 |
| PS: early synchronous saves | 174.177, 175.029, 178.251 | 175.029 |
| AS: early asynchronous saves | 174.774, 174.254, 177.811 | 174.774 |

AS was **1.18% slower** than S and **0.15% faster** than PS by sample medians.
No end-to-end benefit sufficient to enable async by default was demonstrated.
PS and AS matched recorded prefill and transfer work. Early saving reduced
prefill but increased transfers: D2H 86.605 → 92.740 GiB and H2D 8.051 → 14.133 GiB.
Only 13 of 56 AS saves used async D2H; emergency saves and H2D remained synchronous.
This does not establish that asynchronous block storage would help or hurt.

[Async per-run metrics](bench/results/kv-tiered/async-comparison.json) also retain
the single synchronous block reference T (153.891 s), explicitly n=1 and not a
stable ranking. Earlier small-load tiered experiments likewise found transfer
reductions without a clear end-to-end win; their different budgets and workload
sizes should not be mixed into the latest matrix.

## Verification

The published tests exercise block/tail sharing, reservation failure, copy failure,
rollback, reference ownership, unique physical-page accounting, finish/invalidate,
async cancellation and invalid result rejection. See [the release test record](TEST_RESULTS.md)
for the actual public-source run, source digest and resource-dependent skips.
GPU measurements and model checks were not rerun during publication. The prior
snapshot measurements remain documented in [the historical report](SESSION_CPU_OFFLOAD.md).
