#!/usr/bin/env python3
"""HF greedy token-id gate. HuggingFace generate() defaults are not used.

  QWEN3_RUNTIME_MODEL=/path/to/Qwen3-4B python -m qwen3_runtime.reference.cli
"""

from __future__ import annotations

import argparse
import os

from qwen3_runtime.reference.hf_greedy import hf_greedy_tokens, runtime_greedy_tokens
from qwen3_runtime.engine.factory import build_engine, select_attention_backend
from qwen3_runtime.engine.memory import bytes_per_block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("QWEN3_RUNTIME_MODEL"))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[151643, 8948, 198, 2610])
    parser.add_argument(
        "--backend",
        default=None,
        help="Attention backend. Default: factory (flashinfer on CUDA if installed).",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode-only CUDA Graph. Default: on for CUDA+FlashInfer.",
    )
    parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help="Content-addressed prefix cache.",
    )
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("set --model or QWEN3_RUNTIME_MODEL")
    backend = args.backend or select_attention_backend()
    engine = build_engine(
        args.model,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        split=False,
        attention_backend=backend,
        cuda_graph=args.cuda_graph,
        enable_prefix_cache=args.enable_prefix_cache,
    )
    rt = runtime_greedy_tokens(engine, args.prompt_tokens, args.max_tokens)
    print("runtime", rt)
    print("bytes_per_block", bytes_per_block(engine.runner.model.cfg, 16))
    print("device", next(engine.runner.model.parameters()).device)
    print("backend", engine.runner.model.attention_backend)
    print("cuda_graph", engine.config.cuda_graph)
    device = str(next(engine.runner.model.parameters()).device)
    dtype = next(engine.runner.model.parameters()).dtype
    if os.environ.get("QWEN3_RUNTIME_SKIP_HF"):
        return 0
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers required for HF side") from exc
    hf = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    hf_tok = hf_greedy_tokens(hf, args.prompt_tokens, args.max_tokens)
    print("hf", hf_tok)
    if rt != hf_tok:
        raise SystemExit(f"mismatch runtime={rt} hf={hf_tok}")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
