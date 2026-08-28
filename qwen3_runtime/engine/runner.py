from collections.abc import Sequence
from typing import Protocol

from qwen3_runtime.engine.request import Request


class ModelRunner(Protocol):
    def run(self, reqs: Sequence[Request]) -> list[int | None]:
        """Return one sampled token per request, or None if this step should not sample."""
        ...
