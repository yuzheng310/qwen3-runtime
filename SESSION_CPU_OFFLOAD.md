# Optional session KV offload

The runtime can synchronously save paused session KV in a bounded CPU store
and restore it when the trajectory continues. It remains **disabled by default**.
This is a capacity fallback: useful when a reused history would otherwise be
evicted and recomputed, with a cost for host allocation and both transfer directions.

## Implementation and configuration

- CPU snapshots track model/KV versions. Weight updates and invalidation discard
  stale snapshots; completion, abort, and sleep reclaim their state.
- Restore admission accounts for GPU capacity and can defer or fall back to
  recomputation. A snapshot larger than the entire CPU budget is rejected before
  evicting other useful snapshots.
- The CPU budget includes pinned memory; it is not an additional allocation.
  Save/restore are synchronous. `async` offload is not implemented.
- Explicit `finish_session(final_token_ids)` releases only a unique, unclaimed,
  exact completed history. Waiting for the engine owner does not block the caller's
  event loop. A missing/error trajectory is not guessed to be complete.
- Model loading now transfers directly to the target device and dtype, avoiding
  a transient full-precision GPU copy.

For the pinned SkyRL integration, set these before constructing the engine:

```bash
export QWEN3_SESSION_CPU_OFFLOAD=sync
export QWEN3_CPU_KV_MAX_BYTES=34359738368
export QWEN3_CPU_KV_PINNED_MAX_BYTES=34359738368
export QWEN3_KV_TRANSFER_CHUNK_BYTES=33554432
export QWEN3_FINISH_SESSIONS=1
```

These example budgets require sufficient host RAM in addition to trainer memory.
Set `QWEN3_SESSION_CPU_OFFLOAD=off` to disable CPU KV. Completion notification is
independently opt-in and also benefits GPU-only sessions. The hook targets SkyRL
fork revision `81e5a97c7430503c0c4e6508497cc5aa01a0c624` and CodeScout's completed
`code_search_loop` result. Other generators can call `finish_session` explicitly.
Direct engine users can pass the corresponding arguments to `build_engine`.

## Measured boundary, 2026-09-12

These are **source-hashed exploratory measurements**, separate from the frozen
clean-commit serving benchmarks. The same CodeScout-4B BF16 checkpoint ran on one
RTX 4090 reporting 49,140 MiB VRAM, with PyTorch 2.8.0+cu128 and FlashInfer 0.6.17.
This is not a retail 24 GiB card. Runs were serialized.

Each main comparison uses 24 tasks, 102 turns, 9,871 fixed output tokens, and
three measured repetitions plus a separately recorded first-use pass. Arrivals
are closed-loop and tool delay is a synthetic 1 s; output tokens are forced.
The table reports medians, not full GRPO steps or numerical-correctness evidence.

| Scenario | Faster GPU control | CPU offload | Interpretation |
|---|---:|---:|---|
| 4 clients, completed sessions released, 26.40 GiB pool | 90.171 s | 90.091 s | 0.09% difference, below observed variation; no CPU saves or restores |
| 24 clients / 8 active, fixed 26.40 GiB pool | 66.798 s (APC-only) | 58.950 s | 11.75% less elapsed time; 31.44% fewer prefill tokens |
| 24 clients / 8 active, automatic 33.47 GiB pool | 54.533 s (Session KV + APC) | 54.436 s | 0.18% difference; no CPU saves or restores |

The CPU arm in both 24-client cases is **Session KV + APC + CPU offload**.
It was not tested without session continuation. The recorded arm mapping is:

| Arm | Session KV | APC | CPU offload | Fixed-pool median | Automatic-pool median |
|---|---|---|---|---:|---:|
| `B11` | On | On | Off | 78.437 s | 54.533 s |
| `apc-only` | Off | On | Off | 66.798 s | 55.116 s |
| `O-sync` | On | On | Synchronous | 58.950 s | 54.436 s |

Thus adding offload to Session KV + APC reduced fixed-pool time by **24.84%**.
The headline **11.75%** uses the faster GPU control, APC-only. These results
support a conditional benefit of the combined design, not universal superiority.
The published per-case `arm_configuration` includes APC/offload flags from the
archived engine configuration; session enablement is derived from the replay
helper's arm mapping and cross-checked against recorded session counters.

The 24-client comparisons use APC enabled, equal GPU parking budgets, spec=0,
2,048 batch-token budget, 16-token blocks, 32 MiB transfer chunks, and a 32 GiB
pinned CPU budget. In the fixed-pool case, offload saved and usefully restored
28 snapshots per run; prefill fell from 447,245 to 306,619 tokens. Its median
completed transfer time was 4.334 s and CPU snapshot peak was 7.552 GiB. Warm
elapsed range/median was 1.09% for APC-only and 0.31% for offload. First-use
offload took 64.707 s versus APC-only's 67.377 s; host allocation cost is not
hidden in the warm result. These controls do not establish a global optimum.

In the 4-client comparison, APC is off and legacy GPU parking quotas differ
(45% for GPU-only, 100% for offload). With completion cleanup both arms do the
same prefill work and make no CPU copies. Before cleanup, offload saved about
29.34 GiB per run without any restores; a small timing difference against the
legacy GPU control cannot be attributed to CPU hits. Finishing trajectories
removes that waste. The 0.09% warm difference is smaller than both arms' roughly
0.3% observed range/median; it is not evidence of acceleration.

### Loading and available capacity

Three fresh processes per loading path measured peak allocated GPU memory
**15.870 → 7.608 GiB** and automatic KV capacity **12,013 → 15,232 blocks**
(+26.80%). Loading time medians were 39.180 and 41.516 s: no loading-speed claim.
Complete checkpoint weights (8,044,936,192 bytes) and 64 MiB of nonpersistent
RoPE buffers were exactly equal between conversion paths.

The added capacity belongs to the GPU baseline, not to CPU offload. The larger
GPU-only pool completed this replay faster than fixed-pool offload. A default
offload recommendation cannot be based only on the smaller baseline pool.

## Evidence and reproduction

[Verification records](bench/results/session-cpu-offload/verification.json)
publish check-level evidence for **58 archived replay executions**, **6 CUDA
tests**, and **16 unforced real-model samples**. The 58 executions include
repeated arms and first-use passes, not 58 independent workloads. The 16 samples
cover four prefix lengths from one source trajectory, with three repeats and a
first sample at each length. These counts are not additive test-suite totals.
The CUDA test log reported one deprecation warning and no failures.

[Numeric observations](bench/results/session-cpu-offload/observations.json)
contain all first-use and measured runs for the four main cases, fresh-process
loader records, source/trace hashes, and the independent full-weight and Ray
checks. Only selected fields are published; original artifact hashes refer to
unprojected archived records. This export is not a complete raw experiment archive.
It contains 40 individual main-case replay records; the verification audit covers
18 additional ancillary executions through their check results and artifact hashes.

### Regenerate the README figures without a GPU

```bash
uv run --no-project --with matplotlib==3.11.2 python scripts/plot_session_offload.py
```

The plotting script reads the published observations, checks its warm-run medians
against the archived summary, and emits PNG, SVG and
[numeric figure provenance](docs/assets/performance/offload-figure-data.json).
The boundary chart shows all available controls, all three warm samples and each
first-use pass. All axes start at zero; the points are observations, not confidence
intervals. The engineering chart separates loading/capacity improvements from
completion cleanup, so those gains are not attributed to CPU-cache hits.

### Replay requirements

The retained-session group uses v1; other groups use v2. Their sole runtime
difference is the nonblocking wait in `finish_session`, which retained-session
replay does not call. Source hashes identify measured snapshots, not clean Git
commits. Benchmark helpers are included, but model weights and raw task tokens
are not redistributed. Exact replay requires the trace whose SHA-256 is
`27e314db2365682ad7418d1c464230ece2bf3288d766a5ac693fd1c1485d4bb0`.
The JSONL schema is one row per turn with `task_id`, `turn_id`, `input_ids`, and
`output_ids`; successive turns must extend the preceding completed history.

From the repository root, with compatible CUDA/FlashInfer dependencies:

```bash
python -m bench.replay_session_offload \
  --model "$MODEL" --trace "$TRACE" \
  --ranks 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92 \
  --blocks 12013 --concurrency 24 --max-active 8 --dwell 1 \
  --apc 1 --policy equal --cpu-gib 32 --chunk-mib 32 --pinned \
  --spec 0 --batch-tokens 2048 --repeats 3 --out fixed24.json
```

Use `--blocks 0` for the automatic-capacity comparison. Automatic capacity is
hardware/process dependent, so inspect `resolved_num_kv_blocks`. For 4 clients,
use `--concurrency 4 --apc 0 --policy legacy --arms B11,O-sync`; add
`--keep-completed` only for the retention ablation. The default releases completed
trajectories. Every invocation needs a new output path.

`bench.profile_model_loading` measures one fresh loader process;
`bench.profile_session_offload` checks real greedy continuations and transfer
costs with a supplied trace. Forced replay alone cannot validate model output.
Separate GPU checks found bit-exact KV and held-GPU continuation equality across
16 real-model cases. Full recomputation can differ numerically from held GPU KV;
this is not claimed to be bit-exact across all execution paths. Actual Ray
generation/finish/sleep/wake was checked with a synthetic trajectory envelope.

**No new complete GRPO training step, reward improvement, or convergence result
was measured.** End-to-end validation remains pending. Keep CPU offload optional
and evaluate useful restores, actual avoided prefill, transfer time, and complete
step latency on the intended workload before enabling it in training.

## Scope relative to general KV cache systems

This implementation owns synchronous snapshots of paused sessions on one GPU.
It already includes pinned-memory support, block-oriented transfers, version
invalidation and restore rollback. Its CPU store is not a shared, content-addressed
prefix cache across sessions or instances. Restoring a snapshot can reuse an
already-cached GPU prefix and import only the missing blocks.

[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design)
organizes reusable prefixes across GPU, host and optional storage tiers and
supports layerwise compute/load overlap.
[vLLM's OffloadingConnector](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
provides chunk-based caching, configurable admission/eviction and tiered backends.
[LMCache multiprocess mode](https://docs.lmcache.ai/mp/architecture.html)
separates the cache service from inference engines and provides storage,
prefetch and management components. These are architecture comparisons, not
same-hardware performance measurements against this runtime.

The current project stops at a validated, optional session-capacity mechanism.
Shared CPU prefix blocks, asynchronous DMA and additional storage backends are
future work only if a measured workload justifies their complexity. There is
no claim of feature parity or production readiness comparable to those systems.
