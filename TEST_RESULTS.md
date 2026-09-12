# Test results

- Status: **passed**
- Recorded at: `2026-09-12T14:24:27.571419+08:00`
- Command: `uv run --frozen python -m pytest tests/ -q --junitxml=<local-output>`
- Python: `3.13.12` (macOS ARM64)
- Duration: `4.576 s`
- Total: **386**
- Passed: **362**
- Skipped: **24**
- Failed / errors: **0 / 0**
- Package build: `uv build` passed (wheel and source distribution).

[Machine-readable summary](bench/results/verification/cpu-tests.json) identifies
the verified runtime and test contents by SHA-256. This is validation of the
published source snapshot, not a claim that benchmark runs used its Git commit.

Tests ran in the public repository's isolated environment. Skips require
optional GPU, model, backend, or frozen replay-token resources. CPU coverage
includes the new store, transactional restore, stale-weight rejection, scheduler
capacity handling, and exact trajectory completion.

Independent prior GPU checks covered six CUDA offload tests, 16 real-model KV
round-trips, full checkpoint/buffer equivalence, and actual Ray finish/sleep/wake.
Their scope and measured source hashes are recorded in
[the offload report](SESSION_CPU_OFFLOAD.md). They were not rerun during this
publication step. A new complete GRPO training comparison remains pending.

Internal analysis and server-local Git-wrapper tests stay outside the public
snapshot together with their unpublished tools and data.
