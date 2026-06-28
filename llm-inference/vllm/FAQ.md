# vLLM 常见问题（FAQ）

> 把 vLLM 部署、性能、显存、并行、量化、长文本与版本依赖中最容易踩的坑，按「现象 → 根因 → 机制 → 解法」讲清楚。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] [[llm-inference/README]] [[llm-optimizer/kv-cache]] [[llm-inference/vllm/服务启动参数]]

## 阅读地图

| 小节 | 解决什么困惑 | 关键词 |
|------|--------------|--------|
| 0 | 一句话锚点 | FAQ 是什么 |
| 1 | 排障前置：vLLM 是怎么跑起来的 | Engine / Scheduler / KV |
| 2 | 安装 & 版本依赖坑 | cublas / CUDA / torch ABI |
| 3 | OOM 与显存占用 | gpu-memory-utilization / KV |
| 4 | 性能不达预期 | batching / 长度 / 预热 |
| 5 | 多卡 / 并行报错 | TP / PP / NCCL |
| 6 | 量化 & 精度问题 | FP8 / AWQ / GPTQ |
| 7 | 长上下文与截断 | max-model-len / RoPE |
| 8 | 输出异常 / 乱码 / 停不下来 | stop / template / sampling |
| 表 | 速查坑表 | — |

## 0. 一句话锚点

> **FAQ 不是「随机问题集合」，而是 vLLM 几个核心机制（PagedAttention 显存分页、Continuous Batching 连续批处理、单 Engine 调度）在真实环境里「被边界条件触发」时的故障投影。**

记住一条主线：**几乎所有 vLLM 的「报错/慢/OOM」都能回溯到三个量之间的张力——「模型权重显存」「KV Cache 显存」「并发请求数 × 序列长度」。** 把这三者的关系搞清楚，80% 的问题不查文档也能定位。

> 本文给的是**稳定原理 + 排查方法**。涉及具体参数名、默认值、版本号时，请以你所用版本的官方文档（`vllm --help`、docs.vllm.ai）与源码为准——vLLM 迭代极快，**不要把本文当成版本手册背**。

## 1. 地基：vLLM 是怎么跑起来的（排障前你必须有的心智模型）

要排障，先要知道一个请求从进来到出 token，经过哪些环节。任何一环卡住，都会变成某类 FAQ。

```
HTTP 请求 (OpenAI 兼容)
   │  prompt + sampling_params
   ▼
┌──────────────────────────────────────────────┐
│  API Server (FastAPI / OpenAI 兼容层)          │
└───────────────┬──────────────────────────────┘
                ▼
┌──────────────────────────────────────────────┐
│  LLMEngine                                     │
│  ┌──────────────┐   ┌──────────────────────┐  │
│  │  Scheduler   │──▶│  Block Manager(KV)    │  │ ← 显存够不够在这里决定
│  │ waiting/run  │   │  分页：逻辑块→物理块  │  │
│  └──────┬───────┘   └──────────────────────┘  │
│         ▼                                      │
│  ┌──────────────┐                              │
│  │  Worker(s)   │  每张 GPU 一个 worker         │
│  │  Model + KV  │  TP/PP 在这里切分             │
│  └──────┬───────┘                              │
└─────────┼────────────────────────────────────┘
          ▼
   一步 forward → 一批 token（continuous batching）
          ▼
   流式返回 / 完成
```

三个关键结论，后面所有 FAQ 都靠它推：

1. **显存 = 模型权重 + KV Cache + 激活/临时缓冲。** vLLM 启动时会「预留」一大块显存给 KV Cache（由 `gpu-memory-utilization` 控制比例），剩下不够放权重就 OOM。
2. **KV Cache 被切成固定大小的「块（block）」分页管理**（PagedAttention），所以并发能力 ≈ 总 KV 块数 ÷ 每个请求占的块数，而每请求块数 ∝ 序列长度。
3. **调度是连续批处理**：每个 forward step 都会重新组批，新请求随时插入、完成的请求随时退出，不像静态批处理要等齐。

> 想深入这三点：[[llm-optimizer/kv-cache]]、[[llm-inference/vllm/请求处理流程]]、[[llm-inference/vllm/源码]]。

## 2. 安装与版本依赖：最隐蔽的一类坑

这正是本文件原始那条记录指向的问题：

```
pip3 install vllm
pip3 install nvidia-cublas-cu12==12.3.4.1   # 修复某些 cublas 版本不匹配导致的报错
```
> 对应社区 issue：https://github.com/vllm-project/vllm/issues/5001

**根因（机制层面）**：vLLM 依赖一条很长的 GPU 软件栈——`CUDA driver → CUDA runtime → cuBLAS/cuDNN → PyTorch（带特定 CUDA ABI）→ vLLM 编译的 CUDA kernel`。pip 安装时这些组件的版本是**各自解析**的，很容易出现：vLLM 期望的 cuBLAS 版本，与 torch 带的、或系统里的 CUDA 不匹配，于是运行时报符号找不到 / `undefined symbol` / cuBLAS 初始化失败。手动 `pip install nvidia-cublas-cu12==<某版本>` 就是把这一格对齐。

```
驱动 (nvidia-smi 看 CUDA Version) ── 必须 ≥ 运行时要求
  └─ CUDA runtime ──┐
        cuBLAS/cuDNN ├─ 三者要和 torch 编译用的 CUDA「同一大版本」
  PyTorch (cu12x) ──┘
        └─ vLLM 预编译 wheel（针对某个 torch+CUDA 组合编译）
```

排查与规避（讲方法，不背具体版本号）：

- **先看驱动**：`nvidia-smi` 右上角 CUDA Version 是驱动支持的上限，太老要先升驱动。
- **用官方推荐组合**：优先按 vLLM 官方安装页给的 `pip install vllm`（它会带匹配的 torch），不要在已有环境里随意混装别的 torch/CUDA。
- **隔离环境**：用全新 conda/venv，避免老环境残留的 `nvidia-*` 包污染。
- **容器优先**：生产强烈建议用官方镜像，整条栈已对齐，省掉 90% 依赖地狱。
- **遇到 `undefined symbol` / cuBLAS 报错**：基本都是 ABI/版本错配，按报错里提到的库去固定版本（如上例），或换回官方推荐的 torch。

> 具体哪个 cuBLAS/torch 版本配哪个 vLLM，**以官方 release notes 为准**，本文不给死数字。

## 3. OOM 与显存占用：FAQ 的「重灾区」

**现象**：启动就 OOM；或者跑着跑着并发一高就 OOM。

**机制**：回到第 1 节的显存等式。vLLM 启动时做了一件容易让人误解的事——它会**主动占用接近上限的显存**（由「显存利用率」类参数控制，常见默认约 0.9），把剩余显存几乎全划给 KV Cache 池，以最大化并发。所以：

```
单卡显存 (如 80GB)
├── 模型权重               (固定，取决于参数量×精度)
├── 框架/激活/临时缓冲     (相对小但非零)
└── KV Cache 池            (= 利用率上限 − 上面两项)   ← 决定能塞多少并发
```

数值例子（直观感受，非精确）：一个 13B 模型 FP16 权重约 26GB；若卡是 24GB，**光权重就放不下**——这类 OOM 不是 KV 的问题，是卡太小，得换更大显存 / 量化 / 多卡 TP。

**常见解法（讲权衡）**：

| 手段 | 做什么 | 代价/权衡 |
|------|--------|-----------|
| 调低显存利用率参数 | 给系统/其它进程留余量，减少「抢显存」类 OOM | 留太多 → KV 池变小 → 并发降 |
| 降低最大序列长度 `max-model-len` | 每请求 KV 上限变小，块预算更宽松 | 太短会截断长 prompt |
| 限制最大并发序列数 | 直接限住同时在跑的请求数 | 吞吐下降 |
| 张量并行 TP 多卡 | 把权重切到多卡，单卡只放一部分 | 需要卡间高速互联，见第 5 节 |
| 量化（FP8/AWQ/GPTQ） | 权重显存减半甚至更多 | 精度/兼容性，见第 6 节 |

> 关键认知：**「启动后显存几乎占满」是正常现象，不是泄漏**——那是 KV 池被预留了。判断真 OOM 要看是不是权重都放不下，或并发把 KV 块用爆。

## 4. 性能不达预期：吞吐/延迟为什么不如宣传

**现象**：QPS 上不去、首 token 慢（TTFT 高）、或单请求 decode 慢。

逐项拆（每个都对应一个可调旋钮）：

1. **没吃满 batch**：vLLM 的吞吐优势来自连续批处理。如果你**串行**发请求（发一个等一个），等于退化成 batch=1，GPU 闲着。→ 用并发客户端 / 压测工具同时压。
2. **prefill 与 decode 的不对称**：长 prompt 的 prefill 是计算密集（一次算完整个 prompt 的 KV），decode 是访存密集（一次一个 token）。TTFT 高通常是 prompt 太长 + prefill 排队。→ 关注是否开启了分块预填充类机制、PD 分离思路（见 [[PD分离]]）。
3. **首次/冷启动慢**：第一次 forward 要编译/预热 CUDA kernel、构图。→ 预热几条请求再上线，别拿首请求测延迟。
4. **采样开销**：复杂 sampling（高 `top_k`、`logprobs`、guided/约束解码）会增加每步开销。→ 评测时控制采样参数一致。
5. **被显存逼到频繁抢占（preemption）**：KV 池不够时，vLLM 会把部分请求**换出（swap/recompute）**再换回，看起来就是「时快时慢」。→ 监控是否有大量 preemption，有就降并发或加显存。

```
吞吐瓶颈定位流程
   慢？
    ├─ 并发上去了吗 ──否→ 客户端并发不足（最常见）
    ├─ 有大量 preemption 吗 ──是→ KV 不够，降并发/max-len
    ├─ prompt 很长 / TTFT 高 ──是→ prefill 瓶颈，看分块预填充/PD分离
    └─ 是不是冷启动首请求 ──是→ 预热后再测
```

> 指标口径（TTFT/TPOT/吞吐/goodput）见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]，别用错指标下结论。

## 5. 多卡与并行：NCCL / TP / PP 报错

**现象**：开张量并行（TP>1）启动卡住、`NCCL error`、`timeout`、进程 hang。

**机制**：TP（张量并行）把每一层的权重矩阵按维度切到多张卡，每步 forward 都要做 **all-reduce/all-gather** 类集合通信把分片结果合起来。这条通信路径依赖 NCCL + 卡间互联（NVLink / PCIe / IB）。任一环不通就 hang 或报错。

```
TP=4 的一层（简化）
  GPU0  GPU1  GPU2  GPU3
   │     │     │     │   各算一片
   └──── all-reduce ────┘   ← NCCL 在这里，卡间带宽决定快慢
              ▼
         合并后的输出
```

排查清单（讲类别）：

- **互联拓扑**：`nvidia-smi topo -m` 看卡间是 NVLink 还是 PCIe；跨 NUMA / 走 PCIe 会慢很多，TP 收益打折。
- **NCCL 不通 / hang**：常见是网络接口选错、防火墙、`NCCL_*` 环境变量没配；可开 `NCCL_DEBUG=INFO` 看它卡在哪一步。
- **TP 数要能整除注意力头数**：切分维度对不上会直接报错。
- **多机**：TP 一般同机内，跨机走 PP（流水并行）或 TP+PP 混合，跨机务必走 IB / 高速网，否则通信吃满。
- **进程数/可见卡**：`CUDA_VISIBLE_DEVICES` 与 TP 数不一致会导致只用到部分卡或报错。

> 通信原理与 NCCL：[[ai-infra/网络/NCCL]]、[[ai-infra/网络/集合通信原语]]、[[ai-infra/网络/InfiniBand]]；TP 原理：[[大模型推理张量并行]]。

## 6. 量化与精度：FP8 / AWQ / GPTQ 的坑

**现象**：加载量化模型报错、精度变差、或某些量化在你的卡上不支持。

**机制**：量化用更少 bit 存权重（甚至激活/KV），换显存与带宽。不同方案要求不同：

| 方案 | 大致定位 | 关键前提 |
|------|----------|----------|
| FP8 | 权重/激活 8bit 浮点，吞吐+显存双收益 | 需较新硬件（支持 FP8 的 GPU）与对应 kernel；见 [[llm-inference/vllm/FP8]] |
| AWQ | 权重 4bit，激活感知，质量较好 | 需用 AWQ 量化好的权重，加载时指定量化类型 |
| GPTQ | 权重 4bit，经典方案 | 同上，需 GPTQ 格式权重 |

常见坑：

- **量化类型没声明对**：vLLM 需要知道权重是哪种量化，类型对不上会加载失败。**具体参数名以版本文档为准**。
- **硬件不支持**：FP8 等在老卡上无对应硬件指令，要么不支持要么回退到慢路径。
- **精度回退**：4bit 量化对小模型 / 数学/代码任务更敏感，**上线前一定做下游评测**，别只看能跑通。
- **KV Cache 量化**是另一回事（把 KV 也压到 FP8/INT8 省显存），与权重量化独立，开了能显著扩并发但可能轻微掉点。

> 量化原理：[[llm-compression/quantization/量化基础]]、[[llm-compression/README]]。

## 7. 长上下文与截断

**现象**：长 prompt 被截断、报「超过 max-model-len」、或开了超长上下文后显存暴涨。

**机制**：

- `max-model-len` 是该实例允许的**最大序列长度（prompt+生成）**。它同时决定 KV 块预算上限——开越长，单请求最坏占的 KV 越多，并发越低，启动时预留也越多。这是「长上下文 ↔ 并发」的直接 trade-off。
- 超过模型**训练长度**还要更长，需要 RoPE 外推/缩放类机制（YaRN 等），vLLM 通过相应配置支持，但**外推会影响质量**，不是免费午餐。

```
max-model-len ↑  →  每请求 KV 上限 ↑  →  KV 池能装的并发 ↓
   （所以「我要 128K 上下文」会显著吃掉并发能力）
```

实践注意点：

- 只在**确实需要**时把 `max-model-len` 设大；大多数业务用不到模型上限。
- 超过模型原生长度，先确认模型本身/配置是否支持长上下文外推，再谈精度。
- 长 prompt 的 TTFT 天然高（prefill 重），结合第 4 节。

> 详见 [[llm-inference/vllm/长文本推理]]。

## 8. 输出异常：乱码 / 停不下来 / 重复

| 现象 | 常见根因 | 方向 |
|------|----------|------|
| 输出乱码/不连贯 | 量化掉点；或 chat template / tokenizer 不匹配 | 核对模型对应的对话模板与分词器 |
| 停不下来、说个没完 | 没设/设错 `stop` 序列或 `eos`；`max_tokens` 太大 | 配好停止条件 |
| 重复 / 复读 | 采样参数（temperature 过低、缺 repetition/presence penalty） | 调采样 |
| 结构化输出不对 | 没用约束/引导解码 | 见 [[GuidedGeneration]] |

> 核心：vLLM 只是**忠实执行**你给的 sampling 与模板。输出「不对」绝大多数是**请求参数/模板**问题，不是引擎 bug——先排查 prompt 模板与采样参数。

## 常见问题速查表

| 现象 | 最可能根因 | 第一步动作 |
|------|------------|------------|
| 启动后显存几乎占满 | 正常，KV 池被预留 | 不用管，除非真权重放不下 |
| 启动即 OOM | 权重就放不下 / 利用率太高 | 量化 / TP 多卡 / 调利用率 |
| 跑着才 OOM | 并发×长度把 KV 用爆 | 降并发 / 降 max-len |
| `undefined symbol` / cuBLAS 报错 | torch/CUDA/cuBLAS 版本错配 | 固定匹配版本或用官方镜像 |
| 吞吐上不去 | 客户端并发不足 | 并发压测，别串行 |
| 时快时慢 | KV 不够触发抢占换出 | 看 preemption 指标，降并发 |
| TP 启动 hang | NCCL/互联不通 | `NCCL_DEBUG=INFO` + 看 topo |
| 量化模型加载失败 | 量化类型没指定对/硬件不支持 | 核对量化格式与卡能力 |
| 输出停不下来 | stop/eos/max_tokens 没配好 | 修采样与停止条件 |
| 长 prompt 被截断 | 超 `max-model-len` | 调大（但牺牲并发）或缩短输入 |

> 通用排障顺序：**先看日志报错类别 → 再回到「权重/KV/并发×长度」三角定位 → 最后才怀疑引擎本身**。版本相关的精确参数与默认值，一律以 `vllm --help` 和官方文档为准。

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- vLLM 总览：[[llm-inference/vllm/README]]
- 推理总览：[[llm-inference/README]]
- 启动参数：[[llm-inference/vllm/服务启动参数]]
- 请求处理流程：[[llm-inference/vllm/请求处理流程]] ｜ 源码：[[llm-inference/vllm/源码]] ｜ 长文本：[[llm-inference/vllm/长文本推理]] ｜ FP8：[[llm-inference/vllm/FP8]]
- KV Cache：[[llm-optimizer/kv-cache]] ｜ 量化基础：[[llm-compression/quantization/量化基础]]
- 张量并行：[[大模型推理张量并行]] ｜ NCCL：[[ai-infra/网络/NCCL]] ｜ 集合通信：[[ai-infra/网络/集合通信原语]]
- 性能指标口径：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
