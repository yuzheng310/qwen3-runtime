<div align="center">

<img src="docs/assets/logo.png" alt="qwen3-runtime logo" width="168">

<h1>qwen3-runtime</h1>

<h3>面向 Qwen3 的紧凑型单 GPU rollout 推理引擎</h3>

<p>跨工具调用保留 KV · 并发轨迹生成 · SkyRL 训练集成</p>

<p>
  <a href="LICENSE"><img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-4C8BF5?style=flat-square"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="docs/pins/Qwen3-4B/config.json"><img alt="模型" src="https://img.shields.io/badge/model-Qwen3-7C3AED?style=flat-square"></a>
  <a href="TEST_RESULTS.md"><img alt="测试" src="https://img.shields.io/badge/tests-327%20passed-2EA44F?style=flat-square"></a>
</p>

<p>
  <a href="#quick-start">快速开始</a> · <a href="#architecture">架构与集成</a> · <a href="#rollout-results">Rollout 收益</a> · <a href="#performance">Serving 曲线</a> · <a href="README.md">English</a>
</p>

</div>


---

`qwen3-runtime` 实现多轮 Agent rollout 的推理部分：生成 action，在工具执行期间
保留 session KV，再从新增 observation 继续生成。统一的 driver 将并发 trajectory
送入共享 batch，SkyRL 集成层连接推理与训练生命周期。

项目面向单 GPU 上的 Qwen3，以 CodeScout-4B 代码定位轨迹作为参考 workload，
用于 rollout 系统研究。模型执行、调度、状态管理与集成代码各自保持明确边界。

## 核心能力

| 能力 | 实现方式 |
|---|---|
| 跨轮次保留 KV | 生成后暂停 session，复用精确的 token 前缀，仅 prefill 新增后缀；历史不一致时重新计算。 |
| 并发 rollout | 统一 engine driver 接收异步请求，通过 continuous batching、chunked prefill、Paged KV 与内存感知调度执行。 |
| 采样与 logprob | 支持 temperature、top-k/top-p/min-p、penalty、带 seed 的采样与逐 token logprob；rollout 输出保持 token 和 logprob 数量一致。 |
| 推测解码 | 由目标模型验证 n-gram 候选，回滚未接受部分的 KV。 |
| 训练生命周期 | SkyRL 适配器提供生成、sleep/wake、权重更新、中止与 session 清理接口；更新策略权重前释放旧 session。 |
| 推理入口 | `LLM.generate()` / `LLM.session()`，以及兼容 OpenAI 协议的 CodeScout 适配器。 |

<a id="rollout-results"></a>

## Rollout 机制带来的实际变化

### 在工具调用之间保留 session KV

![Session KV 集成前后的 GRPO 生成阶段与完整 step 耗时](docs/assets/performance/grpo-session-kv.png)

一组归档 CodeScout GRPO 实验使用 8 个 prompt、每个采样 8 条轨迹。启用 session
的集成版本把生成阶段从 **513.35 s 降到 408.68 s（1.26×）**，完整 step 从
**1844.14 s 降到 1732.44 s（1.064×）**。生成更快并不会等比例地缩短整个训练
step；启用组在 223 个 resumed turn 中复用了约 163 万 prompt token。

这是**历史集成实验**：启用组还包含 sleep 时卸载权重的改动，不能把完整 step
的收益全部归因于 KV。每组只有一个 step、一个 seed，采样轨迹也有差异，不作为
训练质量或当前版本性能的证明。

### 让 GRPO 的并发 turn 真正进入同一个 batch

![开启 session KV 时串行与批处理 rollout driver 的性能对照](docs/assets/performance/grpo-batching.png)

两边都开启 session KV；同一 build 的串行对照每次只接纳一个 turn，批处理组允许
多个 turn 共享 engine step。实测生成区间 **380.67 → 289.34 s（1.32×）**，
引擎步数 **28,753 → 12,840**，平均 batch **1.00 → 2.26**。Request-step
总量接近（28,753 与 28,991），体现的是把相近的 decode 工作合并到更少的引擎步骤中。

这个计时区间从首次接纳 turn 到最后一个 turn 结束，包含中间的工具执行，
与上面的 SkyRL generation timer 边界不同，**两项加速比不能相乘**。
它也是单次观测，分别产生 299 和 297 个 turn，不是逐 token 相同的 replay。
[证据来源、限制与绘图脚本](RESULTS.md#rollout-feature-observations)。

<a id="performance"></a>

## 性能一览

![并发 session 数与输出吞吐、goodput、p99 TTFT、p99 TPOT 的关系](docs/assets/performance/session-scaling.png)

**增加 KV 能缓解过载时的吞吐下降，但没有消除延迟边界。** 在这组历史扫描的
已测点中，两档内存预算到 N=6 时仍满足记录的 p99 阈值；到 N=8 时，两者的
TTFT 和 TPOT 都越界，更多 KV 则保住了更多 goodput。两条线是**同一 GPU 的
两档 KV 分配量**，不是两款 GPU 的性能对比。

每个点来自一轮实测，每个 session 运行五个 turn。Goodput 是同时满足
**TTFT ≤ 4.5 s 且 TPOT ≤ 100 ms** 的 turn 数除以测量总耗时。TTFT 图使用
对数轴；每点只有 5–60 个 turn，尾延迟估计受样本量限制。这是历史测量，
不是当前版本的容量保证。[指标定义、原始数据与复现方式](RESULTS.md#performance-figures)。

<a id="quick-start"></a>

## 快速开始

### 1. 安装并运行 CPU 示例

在仓库根目录执行：

```bash
uv sync --frozen --all-extras
```

通过 `uv run python` 运行以下代码。示例使用**随机初始化的微型模型**，演示接口和
KV 生命周期；输出是 token ID，不具备实际语言生成能力，也不需要下载模型。

```python
from qwen3_runtime import LLM, SamplingParams

llm = LLM()  # Tiny random model on CPU; no weights to download.
greedy = SamplingParams(temperature=0.0)
print(llm.generate([1, 2, 3], sampling=greedy, max_tokens=3))

with llm.session() as session:
    prompt_ids = [1, 2, 3]
    session.turn(prompt_ids, sampling=greedy, max_tokens=2)
    # Keep the exact generated IDs, then append a toy observation token.
    next_prompt_ids = prompt_ids + session.last_tokens + [7]
    session.turn(next_prompt_ids, sampling=greedy, max_tokens=2)
    print(session.report())
```

每次 `session.turn()` 都传入完整 token 历史。要复用已有 KV，第二轮必须精确保留
上一轮的 prompt 和生成 token，再追加新输入。`session.report()` 展示复用统计，
退出上下文会释放 session。把渲染后的聊天文本重新编码可能改变前缀并触发完整
prefill，因此示例直接使用 token ID 明确展示这一约束。

`LLM.generate()` 是同步易用入口。并发 trajectory 的批处理由 SkyRL 适配器使用的
rollout driver 提供；向该同步入口传入 prompt 列表并不等于并发批处理。

### 2. 使用模型权重

仓库不包含权重。查看下载参数，并指向符合 pin 配置的 Qwen3-4B 本地目录：

```bash
uv run python scripts/fetch_model.py --help
export QWEN3_RUNTIME_MODEL=/path/to/Qwen3-4B
uv run python scripts/gpu_ready.py
```

```python
import os
from transformers import AutoTokenizer
from qwen3_runtime import LLM, SamplingParams

model_dir = os.environ["QWEN3_RUNTIME_MODEL"]
llm = LLM(model_dir, tokenizer=AutoTokenizer.from_pretrained(model_dir))
print(llm.generate("Explain what a KV cache stores.",
                   sampling=SamplingParams(temperature=0.0), max_tokens=64))
```

文本输入需要显式提供 tokenizer。完整模型的 GPU 推理需要匹配的 CUDA/PyTorch
环境，以及足够容纳权重和 KV 的显存。Factory 在 CUDA 上优先选择已安装的
FlashInfer，否则使用 PyTorch；也可以显式选择 Triton 后端。`--all-extras` 安装
项目声明的开发与正确性依赖，**不包含 FlashInfer、Ray 或 SkyRL**；加速和训练
环境需要另外准备兼容依赖。

### 3. 验证或运行 benchmark

```bash
uv run --frozen python -m pytest tests -q
uv run python -m bench.run_bench \
  --case decode --scale full --engine qwen3-runtime \
  --model "$QWEN3_RUNTIME_MODEL" --out /tmp/qwen3-runtime_decode.json
```

测试可在 CPU 上执行，依赖额外资源的检查会跳过。完整 benchmark 需要对应硬件和
权重，默认拒绝 dirty Git worktree。测量约定见 [bench/CONTRACT.md](bench/CONTRACT.md)。

<a id="architecture"></a>

## 架构与训练集成

```mermaid
flowchart LR
    A["Agent / tool / environment"] -->|"Prompt + observation"| D["Rollout driver"]
    D --> E["Engine: batch + generate"]
    E -->|"Tokens + logprobs"| A
    E <--> K["Session KV: pause / resume"]
    A -->|"Completed trajectories"| T["External trainer + reward"]
    T -->|"Updated policy weights"| I["SkyRL integration"]
    I -->|"Sleep / update / wake"| E
    I -->|"Invalidate old sessions"| K
    classDef runtime fill:#fff4d6,stroke:#f59e0b,color:#111827;
    class D,E,K,I runtime;
```

高亮组件位于本仓库。外部 trainer 决定何时更新策略，集成层提供权重更新和
生命周期接口，runtime 管理受影响的推理状态。旧策略生成的 KV 不能沿用到
新权重下的生成过程。

[SkyRL 适配器](qwen3_runtime/integrations/skyrl/inference_engine.py) 提供
`generate`、`chat_completion`、`sleep`、`wake_up`、`update_named_weights` 和
`abort_generation`。[Factory 注入模块](qwen3_runtime/integrations/skyrl/inject.py)
提供 `patch_skyrl_factory()`，可在兼容的 SkyRL 入口创建 inference engine 前调用。
Ray 和 SkyRL 是外部依赖；这里提供集成入口，不是一条独立可运行的训练启动命令。

Engine 不实现 RL 算法、reward、Agent planner、工具 sandbox 或多租户网关。
集成行为有 CPU 回归测试覆盖；最近一次公开验证没有执行完整 SkyRL 训练任务。

<a id="results"></a>

## 测量证据与适用范围

### 冻结的 serving 基线

以下数据来自**历史 clean revision**，使用记录中的 RTX 4090-class 48 GiB 环境
和 Qwen3-4B BF16；它们不代表当前源码快照已经重新完成 GPU 性能验证。

![qwen3-runtime 与 vLLM 在六种固定负载下的实测输出吞吐量](docs/assets/performance/serving-baseline.png)

柱长是三次 trial 的中位数，误差线保留实测最小值到最大值，包括 prefill-shaped
负载中的波动。每个子图单独取刻度并从零开始；不同负载不连成一条缩放曲线。

| 测量 | qwen3-runtime | vLLM 0.27.1 | 相对表现 |
|---|---:|---:|---:|
| 六个 closed-batch workload 的 output throughput | — | — | 94.70–97.70% |
| 六 session SLO goodput | 1.355 req/s | 1.398 req/s | 96.9% |

[RESULTS.md](RESULTS.md) 提供测量口径、具体环境和原始 JSON 链接。这些 serving
测量不能用来证明端到端训练加速或 reward 提升。

### 参考 workload

保留的统计摘要描述了 494 个 CodeScout task、2,395 次模型调用：

| 特征 | 观测值 |
|---|---:|
| 每个 task 的调用次数中位数 | 5 |
| 输入 / 输出 token 数中位数 | 9,981 / 87 |
| 输入历史精确满足 append-only 的相邻 turn | 1,901 / 1,901 |
| 跨轮次具有复用潜力的 prompt token | 70.6% |

相邻 turn 数量描述的是**输入历史结构**，不是模型输出正确性。复用比例描述的是
workload 的优化空间，不是实测加速比或保证的缓存命中率。精简统计位于
[workloads/code_localization/](workloads/code_localization/)；原始 replay token
语料和模型权重不随仓库发布。

## 验证与限制

最近一次[测试记录](TEST_RESULTS.md)：CPU 上 **327 通过、18 跳过**，wheel 与源码
包构建成功，并发 batch 回归连续 30 次通过。测试覆盖 session 隔离、权重更新时
释放状态、token/logprob 对齐、采样与 speculative commit/rollback。
跳过项依赖可选 GPU、模型、后端或 replay token 资源。本次快照未重跑 GPU 性能
测试和完整 SkyRL 训练。

## 仓库导航

| 路径 | 职责 |
|---|---|
| [llm.py](qwen3_runtime/llm.py) | 同步生成与 session 入口 |
| [engine/](qwen3_runtime/engine/) | 调度、请求生命周期、模型执行与权重处理 |
| [rollout/](qwen3_runtime/rollout/) | 并发 driver、session、生命周期与 logprob 处理 |
| [integrations/skyrl/](qwen3_runtime/integrations/skyrl/) | 训练适配器、Ray actor 与 factory 集成 |
| [attention/](qwen3_runtime/attention/) | PyTorch、FlashInfer 与 Triton attention 后端 |
| [serving/](qwen3_runtime/serving/) | 协议辅助、detokenization 与 SLO harness |
| [CodeScout server](scripts/codescout_openai_server.py) | 兼容 OpenAI 协议的 HTTP 适配器 |
| [bench/](bench/) · [tests/](tests/) | 测量工具与正确性检查 |

---

Rollout 系统研究 · [Apache License 2.0](LICENSE) · [English](README.md)
