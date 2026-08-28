# Benchmark contract

Published performance numbers must name a retained artifact, source revision,
workload, and unit. Full-scale results are publishable only when the source
worktree was clean and every request produced exactly its configured output
length.

## Frozen identity

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-4B` |
| Model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Dtype | BF16 |
| Decoding | greedy |
| Forced length | `ignore_eos=True`, exact `max_tokens` |
| Seed | `20260825` |
| Production baseline | vLLM, version recorded in each JSON |

The benchmark host and its actual framebuffer size are recorded in every JSON;
the session experiment separately records its simulated 24 GiB capacity budget.
The pinned model architecture is in
[`docs/pins/Qwen3-4B/config.json`](../docs/pins/Qwen3-4B/config.json).

## Closed-batch workloads

| Case | concurrency / requests | prompt tokens | output tokens |
|---|---:|---:|---:|
| `decode` | 1 / 1 | 256 | 512 |
| `latency` | 1 / 1 | 512 | 128 |
| `batch8` | 8 / 8 | 256 | 128 |
| `throughput` | 16 / 16 | 256 | 128 |
| `prefill` | 1 / 1 | 2,048 | 16 |
| `longctx` | 1 / 1 | 8,192 | 64 |

Prompts are deterministic token IDs sampled uniformly from
`[1, min(vocab_size - 1, 1000))`. The protocol uses one warmup and three measured
trials and records median, mean, standard deviation, minimum, and maximum.

For the vLLM comparison, prefix caching is disabled. The long-context case uses
`max_num_batched_tokens=2048` for both engines. Other qwen3-runtime cases use
`max(2048, concurrency * prompt_tokens)`.

## Session workload

The six-session result uses five turns per session, a 9,981-token first prompt,
1,633-token suffixes, one second of think time, and output lengths sampled from
the frozen CodeScout length distribution. qwen3-runtime holds session KV and
submits suffixes; vLLM receives growing full prompts with prefix caching enabled.
Both use `max_num_seqs=8`, `max_num_batched_tokens=2048`, and the same simulated
24 GiB KV budget calculation.

The serving bounds are p99 TTFT ≤ 4.5 seconds from host arrival and p99 TPOT ≤
100 milliseconds. Goodput counts only requests meeting both bounds.

## Required artifact fields

Every published JSON must contain:

- `schema_version`, `engine`, `case`, `seed`, and `workload`;
- `engine_config` and `environment`;
- the command, timestamp, software versions, source revision, and `dirty=false`;
- `metrics` and `forced_length_ok=true`.

Validate a result directory with:

```bash
python -m bench.report --check PATH
```

The HuggingFace greedy token-ID tests under `tests/reference/` are the
correctness gate for a benchmark GPU. Tiny CPU workloads are supplementary and
must not be presented as full-scale performance evidence.
