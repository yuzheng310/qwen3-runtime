# Test results

- Status: **passed**
- Recorded at: `2026-09-08T19:02:41.953397+08:00`
- Command: `uv run --frozen python -m pytest tests/ -q`
- Python: `3.13.12` (macOS ARM64)
- Duration: `3.173 s`
- Total: **345**
- Passed: **327**
- Skipped: **18**
- Failed: **0**
- Errors: **0**
- Package build: `uv build` passed (wheel and source distribution).
- Concurrent batch regression: **30/30** repeated runs passed.

Tests ran in an isolated environment in this public repository. Skips require
optional GPU, model, backend, or frozen replay-token resources. GPU performance
and full SkyRL training were not rerun on this CPU-only verification machine.
Published benchmark measurements retain their original provenance.

Internal experiment-analysis and server-local Git-wrapper tests are excluded
from the public snapshot together with their unpublished tools and data.
