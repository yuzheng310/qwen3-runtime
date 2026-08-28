# Published benchmark evidence

Only clean-commit runs with `forced_length_ok=true` are included here. Numeric
measurements and reproducibility fields are preserved from the development
evidence store; ephemeral benchmark-host names are removed from the public JSON.

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
