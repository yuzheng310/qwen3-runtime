"""Engine drivers. HF is latency-only. vLLM is the production baseline."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from qwen3_runtime.config import Config
from qwen3_runtime.engine.engine import Engine
from qwen3_runtime.engine.model_runner import PagedRunner
from qwen3_runtime.serving.slo_harness import RequestTrace, run_closed_batch, run_poisson
from qwen3_runtime.models.qwen3 import Qwen3ForCausalLM

from bench.workloads import Workload, poisson_arrivals


def tiny_engine(model: Qwen3ForCausalLM, workload: Workload, *, split: bool) -> Engine:
    n_slots = workload.n_requests * (workload.prompt_tokens + workload.output_tokens)
    block_size = 4
    num_blocks = max(32, (n_slots + block_size - 1) // block_size + 8)
    runner = PagedRunner(model, split=bool(split))
    return Engine(
        Config(
            block_size=block_size,
            num_kv_blocks=num_blocks,
            max_num_seqs=max(workload.concurrency, workload.n_requests),
            max_num_batched_tokens=max(32, workload.prompt_tokens),
        ),
        runner,
    )


def runtime_traces(
    engine: Engine,
    prompts: list[list[int]],
    workload: Workload,
) -> tuple[list[RequestTrace], float]:
    t0 = time.perf_counter()
    if workload.name == "online":
        assert workload.poisson_rate is not None
        arrivals = poisson_arrivals(len(prompts), workload.poisson_rate)
        traces = run_poisson(engine, prompts, workload.output_tokens, arrivals).traces
    else:
        traces = run_closed_batch(engine, prompts, workload.output_tokens)
    return traces, time.perf_counter() - t0


def hf_latency_traces(model_dir: str | Path, prompt: list[int], max_tokens: int) -> tuple[list[RequestTrace], float]:
    """Single-request greedy with KV cache. Not a serving baseline."""
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    t_origin = time.perf_counter()
    tokens: list[int] = []
    itls: list[float] = []
    first: float | None = None
    last = t_origin
    with torch.no_grad():
        out = model(ids, use_cache=True)
        tok = int(out.logits[0, -1].argmax().item())
        if device == "cuda":
            torch.cuda.synchronize()
        first = time.perf_counter()
        tokens.append(tok)
        past = out.past_key_values
        last = first
        for _ in range(max_tokens - 1):
            nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
            out = model(nxt, use_cache=True, past_key_values=past)
            tok = int(out.logits[0, -1].argmax().item())
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            itls.append(now - last)
            last = now
            tokens.append(tok)
            past = out.past_key_values
    wall = time.perf_counter() - t_origin
    trace = RequestTrace(
        request_id=0,
        arrival_s=t_origin,
        prompt_len=len(prompt),
        max_tokens=max_tokens,
        tokens=tokens,
        first_token_s=first,
        itl_s=itls,
        last_token_s=last,
    )
    return [trace], wall


def vllm_first_token_s(metrics: object | None) -> float | None:
    """vLLM RequestOutput.metrics.first_token_time, or None.

    Must not substitute generate() wall time. Closed-batch LLM.generate often
    leaves metrics unset; that makes TTFT == wall and is not a real TTFT.
    """
    first = getattr(metrics, "first_token_time", None) if metrics is not None else None
    if first is None:
        return None
    return float(first)


_vllm_llm = None
_vllm_cache_key: tuple | None = None


def vllm_engine_kwargs(model_key: str, *, case: str | None = None) -> dict:
    """vLLM closed-batch knobs for Production Parity.

    Prefix caching is **off**. vLLM 0.27 resolves ``enable_prefix_caching=None``
    to True. Warmup and measured trials use the same token-id prompts, so a
    cache hit would skip prefill on D/E (and shrink TTFT on A–C). This phase
    does not implement prefix cache; turning it off is matching scope, not
    weakening ``gpu_memory_utilization``.

    longctx pins ``max_num_batched_tokens=2048`` (frozen chunk cap). vLLM
    ``LLM.generate`` on a 4090 otherwise defaults that to 8192, so 8K prefill
    is one shot while we chunk. Other cases leave the upstream default.
    """
    kwargs = {
        "model": model_key,
        "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
        "trust_remote_code": True,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": False,
    }
    if case == "longctx":
        kwargs["max_num_batched_tokens"] = 2048
    return kwargs


def vllm_closed_batch(
    model_dir: str | Path,
    prompts: list[list[int]],
    max_tokens: int,
    *,
    case: str | None = None,
) -> tuple[list[RequestTrace], float]:
    """Closed-batch vLLM generate. Engine load is cached; wall is generate() only."""
    from vllm import LLM, SamplingParams

    global _vllm_llm, _vllm_cache_key
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        TokensPrompt = None  # type: ignore[misc, assignment]
    model_key = str(model_dir)
    kwargs = vllm_engine_kwargs(model_key, case=case)
    cache_key = (model_key, tuple(sorted(kwargs.items())))
    if _vllm_llm is None or _vllm_cache_key != cache_key:
        try:
            _vllm_llm = LLM(skip_tokenizer_init=True, **kwargs)
        except TypeError:
            kwargs.pop("enable_prefix_caching", None)
            try:
                _vllm_llm = LLM(skip_tokenizer_init=True, **kwargs)
            except TypeError:
                _vllm_llm = LLM(**kwargs)
        _vllm_cache_key = cache_key
    llm = _vllm_llm
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    if TokensPrompt is not None:
        reqs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
        try:
            outputs = llm.generate(reqs, params)
        except TypeError:
            outputs = llm.generate(reqs, sampling_params=params)
    else:
        outputs = llm.generate(prompt_token_ids=prompts, sampling_params=params)
    wall = time.perf_counter() - t0
    traces: list[RequestTrace] = []
    for i, out in enumerate(outputs):
        ids = list(out.outputs[0].token_ids)
        metrics = getattr(out, "metrics", None)
        first = vllm_first_token_s(metrics)
        arrival = getattr(metrics, "arrival_time", t0) if metrics else t0
        traces.append(
            RequestTrace(
                request_id=i,
                arrival_s=float(arrival) if arrival is not None else t0,
                prompt_len=len(prompts[i]),
                max_tokens=max_tokens,
                tokens=ids,
                first_token_s=first,
                itl_s=[],
                last_token_s=t0 + wall,
            )
        )
    return traces, wall
