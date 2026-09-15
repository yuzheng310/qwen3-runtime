# Test results

- Status: **passed**
- Recorded at: `2026-09-15T16:48:50.597176+08:00`
- Command: `.venv/bin/python -m pytest tests/ -o addopts='' -q -ra --junitxml=<local-output>`
- Python: `3.13.12` (macOS ARM64, public repository environment)
- Duration: `4.161 s`
- Total: **513**
- Passed: **488**
- Skipped: **25**
- Failed / errors: **0 / 0**
- Package build: offline wheel and source distribution passed.

[Machine-readable summary](bench/results/verification/cpu-tests.json) identifies
the verified runtime and test contents by SHA-256. This validates the published
source, not the identity of historical GPU benchmark commits.

CPU coverage includes block/tail deduplication, reservation and copy failures,
mixed restore, reference ownership, physical-page session accounting, claim
protection, async save lifecycle/cancellation, and invalid benchmark rejection.
The skips require CUDA (including real pinned allocation), model weights,
CodeScout tokenizer blobs, or the optional frozen replay-token corpus.
Internal tools and their dependent tests remain outside the public snapshot.

The [new KV report](KV_CACHE.md) and per-run JSON describe archived GPU replays.
They were not rerun during publication. Prior snapshot checks remain in
[the historical offload report](SESSION_CPU_OFFLOAD.md) and
[archived verification records](bench/results/session-cpu-offload/verification.json).
Those evidence categories are separate from this CPU test count. A new complete
GRPO training comparison remains pending.
