# TGI Benchmark（text-generation-benchmark 推理性能压测工具）

> TGI 自带的命令行基准测试工具，对【已经启动的单个 TGI 服务】做"预填充/解码"两阶段的低层性能剖析，给出时延与吞吐量曲线。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-eval/README]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] [[llm-inference/README]] [[llm-optimizer/kv-cache]]

## 阅读地图

| 小节 | 你会学到 | 关键词 |
|------|---------|--------|
| 0 锚点 | 一句话记住它是什么 | 单服务、底层、两阶段 |
| 1 地基 | 为什么需要它、和 HTTP 压测的区别 | gRPC 直连、绕过 router |
| 2 整体架构 | 工具在 TGI 体系里的位置 | router / shard / benchmark |
| 3 三步流程 | 预热→预填充→解码 | warmup / prefill / decode |
| 4 两阶段计算原理 | 为什么 prefill 和 decode 要分开测 | 计算密集 vs 访存密集 |
| 5 指标含义 | 每个数字代表什么、怎么读 | 时延/吞吐/p50/p90 |
| 6 参数说明 | 各参数做什么用、怎么权衡 | batch-size / sequence-length |
| 7 示例 | 一次典型压测怎么跑 | 命令骨架 |
| 坑 | 容易踩的雷 | 与线上不一致 |

## 0. 一句话锚点

**TGI Benchmark = "对一个已经跑起来的 TGI 模型分片（shard）直接发推理请求，分别测量 prefill 阶段和 decode 阶段的时延与吞吐量"的底层压测工具。** 它不测 HTTP 接口、不测端到端排队，而是贴着模型推理本身去量"这块 GPU 上这个模型，预填充一次要多久、每生成一个 token 要多久"。

> 注意区分：它是 TGI 项目里 `benchmark/` 子目录提供的二进制工具（常以 `text-generation-benchmark` 形式调用），不同于用 `wrk`/`llmperf`/`vllm benchmark` 这类从客户端打 HTTP 的端到端压测。具体命令名、参数名与默认值以官方仓库为准：
> - https://github.com/huggingface/text-generation-inference/tree/main/benchmark

## 1. 地基：它解决什么问题

做大模型推理服务，性能问题通常分两层：

1. **服务层 / 端到端**：用户从发请求到拿到完整回复要多久？这包含网络、HTTP 解析、排队（continuous batching 调度）、prefill、decode、序列化等所有环节。`wrk`、`llmperf`、压 OpenAI 兼容接口都属于这一层。
2. **推理核心层**：抛开排队和网络，**模型本身**在这块 GPU 上算得快不快？prefill 一段 prompt 要多少毫秒？生成每个 token 要多少毫秒？

TGI Benchmark 解决的是**第 2 层**。它的价值在于：

- **隔离变量**：绕开 router 的连续批处理调度与 HTTP 栈，直接量"裸推理"性能，便于定位"是模型/算子慢，还是调度/网络慢"。
- **分阶段**：把推理拆成 **prefill（预填充）** 与 **decode（解码）** 两个本质不同的阶段分别测量——这两个阶段的瓶颈完全不一样（见第 4 节）。
- **扫描参数**：可以批量扫 batch size、sequence length，画出"吞吐/时延随并发或序列长度变化"的曲线，用来做容量规划和调参。

```
        端到端压测(HTTP)                TGI Benchmark(直连)
   ┌──────────────────────┐        ┌──────────────────────┐
   │ 网络 + HTTP 解析       │        │   (跳过)              │
   │ router 排队/批处理     │        │   (跳过)              │
   │ ┌──────────────────┐ │        │ ┌──────────────────┐ │
   │ │ prefill + decode │ │  vs    │ │ prefill | decode │ │  ← 只测这里，且分开测
   │ └──────────────────┘ │        │ └──────────────────┘ │
   │ 反序列化 + 网络回传    │        │   (跳过)              │
   └──────────────────────┘        └──────────────────────┘
   测"用户体感总时延"               测"模型在这卡上的核心速度"
```

## 2. 整体架构：工具在 TGI 体系里的位置

TGI（Text Generation Inference）的典型运行结构：

```
                 HTTP / OpenAI 兼容请求
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  router (Rust)                   │  ← 排队、连续批处理(continuous batching)、
        │   - 校验/限流/调度                │     token 流式返回
        └─────────────────────────────────┘
                          │ gRPC
                          ▼
        ┌─────────────────────────────────┐
        │  model server / shard (Python)   │  ← 真正跑模型前向，含 prefill / decode
        │   - shard 0 (GPU0) ... shard N    │     张量并行多卡时多个 shard
        └─────────────────────────────────┘
                          ▲
                          │ gRPC 直连(绕过 router)
        ┌─────────────────────────────────┐
        │  text-generation-benchmark       │  ← 本工具：直接对 shard 发 prefill/decode 请求
        └─────────────────────────────────┘
```

**关键点**：Benchmark 工具通过 **gRPC 直接连接模型 server（shard）**，而不是走 router 的 HTTP 入口。所以它测到的是"模型分片的纯推理性能"，看不到 router 层的排队与批处理收益/开销。因此它需要先有一个**已经启动并加载好模型**的 TGI server 作为被测对象，工具本身不负责加载模型。

## 3. 核心流程：三步——预热、预填充、解码

这是本主题最核心的内容，对应原始笔记里的"三步"。源码逻辑可参考（版本会变，以仓库为准）：
- https://github.com/huggingface/text-generation-inference/blob/v1.4.3/benchmark/src/generation.rs#L63

整体每一轮（针对某个 batch_size × sequence_length 组合）大致是：

```
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  ① 预热      │ →  │ ② 预填充     │ →  │ ③ 解码       │
  │  Warmup     │    │  Prefill     │    │  Decode     │
  └─────────────┘    └─────────────┘    └─────────────┘
   跑几轮丢弃          只测"吃 prompt"    测"逐 token 生成"
   不计入结果          的一次前向          循环若干步
```

### ③-1 预热（Warmup）
先空跑若干次完整的"预填充 + 解码"，目的是：
- 触发 CUDA kernel 的 JIT 编译 / autotune、cuBLAS/cuDNN 算法选择落地；
- 让显存分配器把 KV Cache、激活的显存块都分配好（避免首次分配的抖动）；
- 让 GPU 频率从低功耗态拉满（避免冷启动测出偏慢的数据）。

**预热结果不计入统计**，只是为了让后续测量稳定。这是所有严肃 GPU 基准测试的通用做法——不预热，第一次的数会被一次性开销污染。

### ③-2 预填充（Prefill，预填充时延）
对长度为 `sequence_length` 的输入 prompt 做**一次**前向，把整段 prompt 的 KV 一次性算出来并填进 KV Cache。它对应：
- **预填充时延**：处理整段 prompt 这一次前向花的时间。
- **预填充吞吐量（token/s）**：约等于 `batch_size × sequence_length / 预填充时延`，即"每秒能消化多少输入 token"。

### ③-3 解码（Decode，逐 token 生成）
在 prefill 得到的 KV Cache 基础上，**自回归地一次生成一个 token**，循环若干步。它对应：
- **解码 token 时延**：生成**单个** token 的时间，约等于线上感受到的"吐字间隔"（ITL / TPOT，inter-token latency）。
- **解码端到端时延**：把若干步解码加起来的总时间。
- **解码吞吐量（token/s）**：约等于 `batch_size / 解码 token 时延`，即"每秒能吐出多少输出 token"。

## 4. 为什么必须把 Prefill 和 Decode 分开测（底层原理）

这是理解整个工具的关键。两个阶段的计算/访存特性截然不同：

| 维度 | Prefill（预填充） | Decode（解码） |
|------|------------------|---------------|
| 一次处理的 token 数 | 整段 prompt（几十~几千） | 每步只 1 个 token |
| 矩阵形状 | 大矩阵×矩阵（GEMM） | 矩阵×向量（GEMV 为主） |
| 瓶颈 | **计算密集**（compute-bound），吃算力 FLOPs | **访存密集**（memory-bound），吃显存带宽 |
| 随序列变长 | 时延随 prompt 长度近似线性增长 | 每步要读越来越大的 KV Cache，单步略增 |
| 优化方向 | FlashAttention、张量并行、算子融合 | KV Cache 量化/PagedAttention、更大 batch 摊薄带宽 |

直观理解访存密集：decode 每一步只算 1 个新 token 的注意力，但**必须把前面所有 token 的 KV Cache 从显存读一遍**。算力没用满，时间几乎全花在"搬数据"上。所以 decode 阶段加大 batch_size 往往能显著提升吞吐量（带宽被多条序列摊薄复用），而 prefill 阶段算力本就接近打满，加 batch 提升有限。

把这两段混在一起测，得到的只是一个被平均掉的、无法指导优化的数字；分开测才能知道"该优化算力还是该优化带宽"。

> 关于这些名词更系统的定义，见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

## 5. 指标含义：每个数字怎么读

工具通常会对每个 (batch_size, sequence_length) 组合，重复多次测量并给出统计分布：

```
   一次组合的输出(示意，数值仅为示范，非真实测试结果)
   ┌──────────────────────────────────────────────┐
   │ batch_size = 8   seq_len = 512                │
   ├──────────────────────────────────────────────┤
   │ Prefill:                                      │
   │   latency  p50=42ms  p90=48ms  p99=55ms       │
   │   throughput ≈ 8*512 / 0.042s ≈ 97500 tok/s   │
   ├──────────────────────────────────────────────┤
   │ Decode:                                       │
   │   per-token  p50=9ms  p90=11ms                │
   │   e2e(若干步) p50=...                          │
   │   throughput ≈ 8 / 0.009s ≈ 888 tok/s         │
   └──────────────────────────────────────────────┘
```

读法要点：
- **看 p50/p90/p99 分布而非只看均值**：尾时延（p99）才决定用户最坏体验和 SLA 是否达标。
- **Prefill 吞吐 ≫ Decode 吞吐 是正常的**：prefill 一次并行处理整段 prompt，单位时间内"过"的 token 自然多；decode 一步只产 1 个 token/序列。
- **Decode per-token 时延 ≈ 用户的吐字快慢**：这是聊天体验最敏感的指标。
- **吞吐 vs 时延要一起看**：加大 batch 通常提升吞吐但拉高单请求时延，工具的多组合扫描正是为了画出这条权衡曲线，帮你选"在可接受时延下的最大吞吐"工作点。

## 6. 参数说明（讲含义与权衡，不背默认值）

> 以下是这类工具**普遍具备的参数类别**及其作用。**精确的参数拼写、是否必填、默认值请以 `text-generation-benchmark --help` 与官方仓库为准**，不同版本会有差异。

| 参数类别 | 作用 | 怎么权衡 |
|---------|------|---------|
| 目标模型 / tokenizer | 指定被测模型，需与已启动的 server 一致 | 必须和线上同一模型/同一精度，否则数据无意义 |
| batch size（可多值扫描） | 一次前向并发处理的序列数 | 越大吞吐越高但单请求时延越高、显存占用越大；扫一组值画曲线 |
| sequence length（输入长度） | prompt 的 token 数 | 越长 prefill 越久、KV Cache 越大；按线上真实分布设 |
| decode 步数 / 生成长度 | decode 阶段循环多少步 | 太少噪声大，取贴近线上输出长度的值 |
| 运行/重复次数（runs） | 每组合重复测多少次取统计 | 越多越稳但越慢；保证够算出 p90/p99 |
| 预热次数（warmup） | 计入统计前空跑几次 | 至少 1~数次，避免冷启动污染 |
| server 连接地址 | gRPC 连到哪个 shard | 指向已启动的 TGI model server |

**核心权衡一句话**：用 batch size 和 sequence length 这两个旋钮，去描出"吞吐—时延"权衡曲面，找到满足你时延预算（如 ITL < 50ms）前提下吞吐最大的配置。

## 7. 典型使用流程

```
  Step 1  启动被测 TGI 服务(加载好模型，记下其 gRPC/shard 地址)
            └─ text-generation-launcher --model-id <模型> ...
                       │
  Step 2  运行 benchmark，指定模型与要扫描的 batch/seq 组合
            └─ text-generation-benchmark \
                 --tokenizer-name <与服务相同的模型> \
                 --batch-size 1 --batch-size 8 --batch-size 32 \
                 --sequence-length 512 ...
                       │
  Step 3  工具自动: 预热 → 逐组合做 prefill/decode 测量
                       │
  Step 4  读 TUI/输出表: 对比各组合的 prefill/decode 时延与吞吐
                       │
  Step 5  画"吞吐-时延"曲线，选定生产配置
```

> 上面的命令仅为**结构示意**，真实子命令名、参数名与是否需要 `--tokenizer-name` 等以 `--help` 输出为准；不要把示意当作可直接复制的精确命令。

## 常见问题 / 坑

| 坑 | 说明 | 应对 |
|----|------|------|
| 把它当端到端压测 | 它绕过 router，**看不到排队/连续批处理**带来的真实线上吞吐 | 端到端要另用 HTTP 压测（llmperf / wrk / vllm-benchmark）|
| 不预热就读数 | 首次 kernel 编译、显存分配、GPU 升频会让前几次偏慢 | 必须配置 warmup，丢弃预热结果 |
| 只看均值 | 均值掩盖尾时延 | 关注 p90/p99，重复足够次数 |
| 模型/精度与线上不一致 | 量化、并行度、KV Cache 设置不同，结果不可迁移 | 用与生产**完全相同**的模型、精度、并行配置 |
| seq_len/输出长度脱离实际 | 测试分布与线上不符，结论偏差大 | 按真实业务的输入/输出长度分布设参数 |
| 混淆 prefill 与 decode 吞吐 | 二者本质不同，不可直接相加或比较优劣 | 分开理解：prefill 吃算力、decode 吃带宽（见第 4 节）|
| GPU 被其他进程占用 | 共享卡导致频率/带宽被抢，数据抖动 | 独占 GPU、固定功耗/频率后再测 |
| 版本差异 | 参数名、输出格式、源码行号随版本变化 | 一切以当前版本 `--help` 和官方仓库为准 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-eval/README]]
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- [[llm-inference/README]]
- [[llm-optimizer/kv-cache]]
- 官方 benchmark 目录：https://github.com/huggingface/text-generation-inference/tree/main/benchmark
- 生成逻辑源码（版本会变）：https://github.com/huggingface/text-generation-inference/blob/v1.4.3/benchmark/src/generation.rs#L63
