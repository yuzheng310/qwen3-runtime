# Published benchmark evidence

The frozen serving baseline uses clean-commit runs with `forced_length_ok=true`.
The separate rollout-feature section contains historical integration observations
with a different, explicitly limited provenance record. Numeric
measurements and reproducibility fields are preserved from the development
evidence store; ephemeral benchmark-host names are removed from the public JSON.

## Optional CPU session offload

[Implementation, measurements, and reproduction](SESSION_CPU_OFFLOAD.md) document
source-hashed exploratory diagnostics from 2026-09-12. Fixed 26.40 GiB capacity
and 24 clients yielded 11.75% less replay time than the faster GPU control;
4-client and larger-pool comparisons showed no credible CPU-tier benefit.
These records are separate from the frozen benchmarks below and do not measure
a new complete GRPO training step. [Selected per-run evidence](bench/results/session-cpu-offload/observations.json)
includes first-use costs and the corrected GPU-capacity baseline.

![Offload timing with all controls and individual repeated observations](docs/assets/performance/offload-boundary.png)

[Engineering improvements](docs/assets/performance/offload-engineering.png)
separately show loading peak, available KV blocks and avoided unused CPU writes.
[The plotting script](scripts/plot_session_offload.py) regenerates both figures
from the public JSON; [figure values](docs/assets/performance/offload-figure-data.json)
record their source SHA-256. Bars are medians; points are individual observations,
not confidence intervals. [Verification records](bench/results/session-cpu-offload/verification.json)
provide the archived replay audit and per-case unforced model checks. No GPU
benchmark was rerun to produce these visualizations.

## Closed-batch runtime comparison

Median output tokens/s, including prompt and output wall time:

| Workload | qwen3-runtime | vLLM 0.27.1 | parity |
|---|---:|---:|---:|
| Decode: 256 prompt / 512 output / concurrency 1 | 100.40 | 104.19 | 96.36% |
| Latency: 512 / 128 / concurrency 1 | 98.09 | 102.35 | 95.83% |
| Batch 8: 256 / 128 | 693.26 | 709.58 | 97.70% |
| Throughput: 256 / 128 / concurrency 16 | 1,212.83 | 1,271.13 | 95.41% |
| Prefill-shaped: 2,048 / 16 | 57.33 | 60.31 | 95.06% |
| Long context: 8,192 / 64 | 48.70 | 51.43 | 94.70% |

Raw evidence: [`bench/results/runtime-vs-vllm/`](bench/results/runtime-vs-vllm/).
The comparison uses Qwen3-4B BF16, seed `20260825`, vLLM prefix caching disabled,
and a matched long-context chunking configuration. These are closed-batch
measurements on the recorded environment, not a universal serving claim.

## Six-session SLO comparison

The recorded SLO bounds are p99 TTFT ≤ 4.5 s from host arrival and p99 TPOT ≤
100 ms. Both engines completed 30 turns and 28 met both per-request bounds.

| Engine | goodput | p99 TTFT | p99 TPOT |
|---|---:|---:|---:|
| qwen3-runtime | 1.355 requests/s | 4.418 s | 90.3 ms |
| vLLM 0.27.1 | 1.398 requests/s | 4.420 s | 88.0 ms |

Raw evidence:

- [qwen3-runtime N=6](bench/results/session-capacity/qwen3-runtime_n6.json)
- [vLLM N=6, fresh process](bench/results/session-capacity/vllm_n6.json)

The workload uses five turns per session, a 9,981-token first prompt, 1,633-token
suffixes, one second of think time, and completion lengths sampled from the
frozen CodeScout length distribution. The runtime holds session KV; vLLM
resubmits growing prompts with prefix caching enabled.

## Recorded environment

The raw JSON records exact commands and software versions. The benchmark host
reported an RTX 4090-class GPU with 50,866,487,296 bytes of framebuffer. The
session experiment simulated a 24 GiB card budget after model weights and fixed
overhead. Model weights are not distributed in this repository.


<a id="performance-figures"></a>

## Performance figures: metrics, sources, and reproduction

The figures use established serving metrics rather than a composite project
score. [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html)
reports output throughput, request latency, and time to first token.
[vLLM's metric definitions](https://docs.vllm.ai/projects/spyre/en/latest/user_guide/performance.html)
distinguish per-request TPOT from individual inter-token latency (ITL), and
[vLLM bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve/) supports goodput
with explicit latency thresholds. [NVIDIA's benchmarking guidance](https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html)
also recommends concurrency sweeps to expose throughput/latency tradeoffs.

| Plotted metric | Exact source field / interpretation |
|---|---|
| Output throughput | Session: `metrics.output_tok_s`; baseline: `metrics.output_tok_s.median`. Output tokens divided by the measured workload wall time. Session wall time includes simulated tool think time and draining; it is not kernel throughput. |
| Goodput | `accounting.goodput_per_s` = `accounting.slo_met / metrics.wall_s`. Completed turns satisfying both TTFT ≤ 4.5 s and TPOT ≤ 0.100 s per turn. |
| p99 TTFT | `ttft_s_p99`, in seconds, measured from host arrival to first output token. It includes queueing; it is not prefill kernel time. |
| p99 TPOT | `tpot_s_p99 × 1000`, in milliseconds. The p99 across each request's mean inter-token interval; **not** p99 across all individual token intervals. |

No request E2E or trajectory-latency curve is inferred from total wall time.
No speedup is inferred from the percentage of reusable prompt tokens.

### Session concurrency sweep

![Session concurrency sweep](docs/assets/performance/session-scaling.png)

- Runtime revision: `070cdc50951aa36302357bcd5a6c08b0fbd60b05`, recorded on
  2026-08-26; all 12 selected records have `dirty=false` and `forced_length_ok=true`.
- Same recorded RTX 4090-class GPU with 50,866,487,296 bytes of framebuffer;
  Qwen3-4B BF16, PyTorch 2.8.0+cu128. This is not a retail 24 GiB GPU comparison.
- Actual allocated KV: 15,576,072,192 bytes (**14.51 GiB**) and 30,111,694,848
  bytes (**28.04 GiB**). The latter requested 38.51 GiB but allocated 28.04 GiB;
  the figure labels actual allocations, not simulated card sizes.
- N = 1, 2, 4, 6, 8, 12 sessions; five sequential turns each; 9,981-token first
  input, 1,633-token suffixes, 1 s simulated tool think time; seed 20260825;
  one warmup run; maximum scheduled sequences 8 and token budget 2,048.
  Output lengths follow the frozen CodeScout distribution, not a constant length.
- One measured run per point (5N requests), with no repeated-run confidence
  intervals. p99 values are retained as recorded using linear interpolation of
  ordered request observations. With 5–60 requests, these are small-sample tail
  estimates. Passing the two marginal p99 thresholds does not imply every
  request meets both SLOs: at N=6 with 14.51 GiB KV, 28 of 30 turns meet both.
- Lines only connect measured points; no smoothing, extrapolation, or invented
  intermediate measurements. The TTFT axis is logarithmic and explicitly labeled.
- Both allocations pass the two recorded p99 bounds at N=6 and fail at N=8.
  Among sampled values, additional KV does not increase that boundary. At N=12,
  TTFT p99 is 41.05 s versus 9.63 s, and recorded preemptions are 84 versus 0.
  These observations describe this historical workload, not universal capacity.

Sources: [session-scaling JSON](bench/results/session-scaling/) plus the already
published [14.51 GiB N=6 record](bench/results/session-capacity/qwen3-runtime_n6.json).
The two series are runtime memory-budget experiments; no vLLM concurrency curve
is synthesized from the sparse or cache-contaminated comparison runs.

### Six-workload throughput comparison

![Serving baseline](docs/assets/performance/serving-baseline.png)

Each panel uses a matched runtime/vLLM workload from the closed-batch evidence
above. Bars are median output tokens/s over three measured trials; whiskers are
minimum to maximum, **not confidence intervals**. The prefill-shaped runtime
has a low trial which remains visible. Axes start at zero and vary by panel;
compare the engines within a panel, not bar lengths across panels. The plotting
script checks prompt length, output length, and concurrency for each pair.

### Rebuild and audit

From the public repository root (no model or GPU needed):

```bash
uv run --no-project --with matplotlib==3.11.1 python scripts/plot_performance.py
```

The script reads only published JSON, validates clean/forced-length provenance
and sweep configuration, and checks goodput arithmetic. It writes PNG and SVG
figures plus [figure-data.json](docs/assets/performance/figure-data.json), containing
the plotted values, source paths, source revisions, and SHA-256 hashes of the
sanitized input files. Host names are omitted from published benchmark JSON.
SVGs: [session curve](docs/assets/performance/session-scaling.svg) ·
[serving comparison](docs/assets/performance/serving-baseline.svg).

These figures visualize retained historical results. They do not represent a
new GPU benchmark of the current runtime or an end-to-end RL training result.


<a id="rollout-feature-observations"></a>

## Rollout feature observations (historical, single-run)

These figures are **not part of the clean/forced-length serving benchmark set**.
They expose useful archived measurements from the CodeScout/SkyRL integration
work, with their limitations intact. [observations.json](bench/results/rollout-features/observations.json)
is a field-preserving extract of the archived step report and adapter counters;
its `source_report_sha256` / `source_counter_sha256` fields identify the original
inputs. Raw conversations, tool output, and private environment paths are not
published. Source hashes identify the archive; they do not substitute for the
unavailable full experiment environment or independent reproduction.

### Session KV integration

![GRPO step timings](docs/assets/performance/grpo-session-kv.png)

The baseline revision is `b1a8c47`; the session-enabled report references
`f48ea3c`. Held settings: 8 instances, 8 samples per prompt, maximum 6 turns,
temperature 1.0, seed 42, a 192,000-token KV pool, sample packing, warm Git cache.
Both records represent one complete GRPO step. The session-enabled revision
also offloads weights during sleep, so the full-step change cannot be causally
assigned to session KV alone. The retained report lacks the full environment
and clean-worktree metadata used by the frozen serving schema.

| SkyRL timer | Baseline (s) | Session-enabled (s) |
|---|---:|---:|
| Generate | 513.35 | 408.68 |
| Forward logprobs / values / reward | 101.77 | 99.49 |
| Policy train | 1225.60 | 1220.41 |
| Weight synchronization | 2.23 | 2.21 |
| Complete step | 1844.14 | 1732.44 |

Component timers are not assumed to sum to the full-step timer; no synthetic
stacked decomposition is drawn. The figure plots directly recorded generation
and complete-step times on separate axes. Enabled counters: 292 turns, 223
resumed, 69 started, 859,350 prompt tokens prefilled, 1,626,541 reused (65.4%).
A token-reuse percentage is not a time-saving percentage. One seed and one step
cannot establish reward improvement or multi-step training convergence.

### Concurrent rollout driver

![GRPO batching comparison](docs/assets/performance/grpo-batching.png)

The archived experiment report states that both arms use the same build and
instrument, with sessions on, 8 instances × 8 samples, and OpenHands concurrency
4. The counter files do not record an exact build revision or full hardware
snapshot. The serial control uses the same driver but admits one turn at a
time; the other arm permits batching. Plotted values are checked against both
retained counter files before extracting the public record.

| Counter / timer | Serial | Batched |
|---|---:|---:|
| Generation span (s) | 380.67 | 289.34 |
| Engine steps | 28,753 | 12,840 |
| Request-steps (sum of requests served per engine step) | 28,753 | 28,991 |
| Mean batch, recorded | 1.00 | 2.26 |
| Maximum batch | 1 | 4 |
| Turns | 299 | 297 |
| Session reuse (%) | 67.5 | 66.2 |
| Prefilled tokens | 815,677 | 883,992 |
| Evicted sessions | 60 | 63 |

Batching saves engine steps, but the batched arm also recomputes more prefill
and sees different sampled trajectories. A batch step is not a constant-time
unit, so step reduction cannot be equated to wall-time speedup. The 1.32× span
ratio is directly measured, with one run per arm and no error bars.

The adapter span runs from first turn admission to last turn retirement and
includes tool execution between turns. It differs from SkyRL's generation
phase (which also brackets wake-up and other boundary work). **Do not multiply
1.26× and 1.32× into a combined acceleration claim.** No end-to-end combined
training-step measurement is provided.

### Evidence not promoted to performance claims

The fitted prefill/decode time shares are model-based estimates, so they are
not presented as measured profiler time. Historical speculative-decoding timings
predate the current logprob-accounting fix; they are not promoted as validated
training-backend performance for this release. GRPO sibling-prefix reuse is not
presented as a measured acceleration without a suitable ablation. CPU correctness
tests for logprobs and weight invalidation establish behavior, not GPU speed.

### Reproduce the figures

```bash
uv run --no-project --with matplotlib==3.11.1 python scripts/plot_rollout_features.py
```

The script uses the published observation extract and writes PNG/SVG figures
plus [rollout-figure-data.json](docs/assets/performance/rollout-figure-data.json)
with the input hash and plotted values. This reproduces the visualization,
not the original training experiment. SVGs: [session KV](docs/assets/performance/grpo-session-kv.svg)
· [batching](docs/assets/performance/grpo-batching.svg).


<a id="kv-four-arm-diagnostics"></a>

## Four-arm Session KV / APC diagnostics

[Observation extract](bench/results/kv-ablation/observations.json) retains the
plotted numeric fields, original input hashes, and available provenance flags.
These diagnostic records are separate from the validated serving baseline.

- **None:** no retained session and no automatic prefix cache.
- **Session KV:** retain and resume one conversation's KV; APC off.
- **APC:** submit the full history and recover matching cached prefix blocks;
  no retained session.
- **Session KV + APC:** session continuation plus content-addressed prefix reuse.

### Sequential replay

100 tasks / 471 requests, CodeScout-4B, spec=0, batch 1. All four archives have
the same subset/ID hashes and recorded output-length sequence, but all record
`environment.dirty=true`, `git_commit=uncommitted`, and `forced_length_ok=false`.
Matching lengths do not override these flags or establish token equivalence.
Warmup is 1 for none/session-only and 0 for APC/combined. There is one trial per
arm. These limitations preclude a validated speedup claim; the chart labels them.
The card differs from the earlier GRPO integration experiments: do not chain
ratios or compare absolute timings between them.

| Arm | Recorded s/task | Later-turn prefill tokens | Peak occupied KV blocks |
|---|---:|---:|---:|
| None | 9.686 | 5,284,019 | 2,783 |
| Session KV | 6.333 | 1,304,392 | 2,783 |
| APC | 6.196 | 1,307,139 | 12,013 |
| Session KV + APC | 6.175 | 1,304,392 | 12,013 |

Peak occupied blocks include cache retention and are not total GPU allocation.
APC's higher occupancy is not automatically wasted memory. Session-only and
APC are nearly substitutes for repeated prefill on this sequential workload;
the small timing difference between APC and the combination has no repeat-run
uncertainty estimate. The observation does not support superiority over APC.

### Concurrent replay

Four arms × concurrency 1/4/8/16, 16 recorded conversations including GRPO
siblings, 67 completed turns per cell, a 12,013-block pool, spec=0, one run/cell.
All cells report zero scheduler preemptions. The next turn arrives immediately,
so this is not a live tool-waiting load. The retained per-cell files lack full
environment and clean-worktree metadata. No confidence intervals are invented.

At concurrency 16, Session KV alone records 221,445 later-turn prefill tokens
and 10 session evictions; combined records 161,482 and 23. APC-only records
161,811 tokens. More evictions can coexist with less recomputation when cached
prefix blocks remain reusable. Evictions do not measure lost tokens, nor are
they scheduler preemptions. Arms without sessions trivially have zero session
evictions; that is not a direct efficiency ranking.

The archived investigation reports at most 3 parked sessions / 2,155 parked
blocks versus a 5,405-block park cap. It did not demonstrate parked-session
pool saturation. The absence of dwell time prevents transferring this curve to
live GRPO memory pressure. Separate numerical probes also marked their gate
false for session-only and combined; a time/counter plot cannot certify
numerical correctness. No recommendation to enable APC universally follows.

### Interpretation and reproduction

The supported architectural advantage is trajectory-owned state and lifecycle:
exact session continuation, explicit release, and invalidation across weight
updates. The counter evidence supports removing repeated prefill versus no
reuse. APC is a complementary reuse mechanism, not a baseline the project
consistently defeats. Choosing a policy must account for dwell time, eviction,
shared prefixes, and the actual speculative-decoding configuration.

```bash
uv run --no-project --with matplotlib==3.11.1 python scripts/plot_kv_ablation.py
```

This rebuilds the diagnostic images from the published extract, not the original
GPU experiment. [Plotted data and hash](docs/assets/performance/kv-ablation-figure-data.json).
SVG: [sequential](docs/assets/performance/kv-four-arm-replay.svg) ·
[concurrent](docs/assets/performance/kv-four-arm-concurrency.svg).
