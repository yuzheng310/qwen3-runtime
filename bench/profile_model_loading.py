"""Run in a fresh process for each source version; no model download."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from qwen3_runtime.engine.factory import _num_kv_blocks
from qwen3_runtime.utils import loader


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "free": free,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite measurement")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; CPU byte-count tests are not GPU measurements")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    before = memory()
    start = time.perf_counter()
    model = loader.load_from_directory(args.model, device="cuda", dtype=torch.bfloat16,
                                       attention_backend="flashinfer")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    after = memory()
    blocks = _num_kv_blocks(model, device="cuda", block_size=16, kv_budget=None)
    torch.cuda.empty_cache()
    after_empty_cache = memory()
    reclaimed_blocks = _num_kv_blocks(model, device="cuda", block_size=16, kv_budget=None)
    source = Path(loader.__file__)
    result = {
        "args": vars(args), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "loader_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        "load_s": load_s, "before": before, "after_load": after,
        "after_empty_cache_diagnostic": after_empty_cache,
        "factory_blocks": blocks, "factory_blocks_after_empty_cache": reclaimed_blocks,
        "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
