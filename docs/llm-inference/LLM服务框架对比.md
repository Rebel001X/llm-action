# LLM 服务框架对比（vLLM / TGI / FasterTransformer / FlexFlow / TensorRT-LLM）

> 一句话定位：把"训练好的大模型"变成"能扛高并发、低延迟、高吞吐的在线服务"的中间件层，本文从最底层的请求生命周期、显存账本、调度算法讲清楚各框架的取舍。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/大模型推理张量并行]] · [[llm-inference/解码策略]] · [[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]]

## 阅读地图

| 节 | 你将搞懂 | 关键产物 |
|---|---|---|
| 0 | 一句话锚点：服务框架到底在优化什么 | 三角约束 |
| 1 | 地基：一次推理请求的生命周期 | Prefill/Decode 两阶段 ASCII 图 |
| 2 | 评测指标的精确定义（TTFT/TPOT/吞吐/QPS） | 公式 + 手算 |
| 3 | 显存账本：权重 + KV Cache + 激活 | 逐数手算 7B/13B/70B |
| 4 | 核心瓶颈：为什么 Decode 是访存受限 | 算术强度手算 |
| 5 | 框架杀手锏一：Continuous Batching | 时序 ASCII 图 |
| 6 | 框架杀手锏二：PagedAttention / KV 分页 | 块表 ASCII 图 + 碎片手算 |
| 7 | 框架杀手锏三：张量并行 / 流水并行 | 通信量手算 |
| 8 | 逐框架剖析：vLLM/TGI/FT/FlexFlow/TRT-LLM | 优劣表 |
| 9 | 数值手算综合示例：A100 上 13B 服务 | 端到端估算 |
| 10 | 选型决策树 | ASCII 决策图 |

---

## 0. 一句话锚点

LLM 服务框架在求解一个**三角约束优化问题**：在固定的 GPU 显存与算力下，同时最大化**吞吐量（Throughput）**、最小化**延迟（Latency）**、控制**成本（$/百万 token）**。这三者互相拉扯：

```
            延迟低 (好交互)
               ▲
               │   增大 batch → 吞吐↑ 但单请求延迟↑
               │   减小 batch → 延迟↓ 但 GPU 利用率↓
   吞吐高 ◄────┼────► 成本低
  (省GPU)      │     (省钱)
               ▼
          所有框架都在这个三角内做权衡
```

所有现代框架（vLLM、TGI、TensorRT-LLM）的创新，本质都是**把同一台 GPU 上能同时服务的请求数（有效 batch）做大，而不增加显存浪费与尾延迟**。

---

## 1. 地基：一次推理请求的生命周期

自回归 LLM 生成分两个性质完全不同的阶段，这是理解一切服务框架的起点。

### 1.1 Prefill（预填充）阶段

把整段 prompt（长度 $L_{in}$）一次性喂进模型，**并行**计算所有位置的注意力，输出第一个 token，同时把每层每个位置的 Key/Value 写进 KV Cache。

```
Prompt: "中国的首都是"  (L_in = 6 tokens)
   t0  t1  t2  t3  t4  t5
   │   │   │   │   │   │
   ▼   ▼   ▼   ▼   ▼   ▼
┌───────────────────────────┐
│   一次前向，6 个位置并行    │  ← 矩阵×矩阵 (GEMM)，计算受限
│   写入 KV[layer][0..5]     │
└───────────────────────────┘
            │
            ▼  产出第 1 个输出 token "北"
```

### 1.2 Decode（解码）阶段

每次只输入**上一步生成的 1 个 token**，读取整段历史 KV Cache，算出下一个 token。串行、逐 token。

```
step1: 输入"北" + 读 KV[0..5]  → 产 "京"   写 KV[6]
step2: 输入"京" + 读 KV[0..6]  → 产 "，"   写 KV[7]
step3: 输入"，" + 读 KV[0..7]  → 产 "是"   写 KV[8]
 ...
每步: 矩阵×向量 (GEMV)，batch=1 时访存受限
```

| 对比维度 | Prefill | Decode |
|---|---|---|
| 输入长度 | 整段 prompt（多 token） | 单 token |
| 并行度 | 高（位置并行） | 低（逐步） |
| 计算形态 | GEMM（矩阵乘矩阵） | GEMV（矩阵乘向量） |
| 瓶颈 | 算力（compute-bound） | 显存带宽（memory-bound） |
| 决定的指标 | TTFT（首 token 时延） | TPOT（每 token 时延） |

> 服务框架的核心难题：Prefill 想要大算力、Decode 想要把 batch 拼大以摊薄访存。两阶段诉求矛盾，催生了 chunked-prefill、PD 分离等技术。详见 [[llm-inference/KV-Cache优化]]。

---

## 2. 评测指标的精确定义

各框架 README 里的"吞吐""延迟"含义并不统一，必须先钉死定义，否则对比无意义。

| 指标 | 英文 | 定义 | 公式 |
|---|---|---|---|
| 首 token 时延 | TTFT | 从收到请求到吐出第 1 个 token | $T_{prefill}+排队$ |
| 每 token 时延 | TPOT/ITL | Decode 阶段相邻 token 间隔 | $T_{decode\_total}/(L_{out}-1)$ |
| 端到端时延 | E2E Latency | 整条请求总耗时 | $TTFT+(L_{out}-1)\cdot TPOT$ |
| 吞吐量 | Throughput | 全系统每秒产出 token 数 | $\sum_{req} L_{out} / T_{wall}$ |
| 请求吞吐 | QPS/RPS | 每秒完成的请求数 | $N_{req}/T_{wall}$ |

### 用户体感锚点

```
TTFT ─── 决定"按下回车后多久开始动" (聊天 < 1s 才不焦虑)
TPOT ─── 决定"打字速度" (人阅读 ~5-10 token/s, < 50ms/token 即流畅)
吞吐  ─── 决定"这张卡每天能服务多少人 → 成本"
```

**手算示例**：某请求 $L_{in}=512$，$L_{out}=256$，实测 TTFT = 180 ms，Decode 总耗时 = 6375 ms。
- $TPOT = 6375/(256-1) = 25\ \text{ms/token}$
- $E2E = 180 + 255\times25 = 6555\ \text{ms} \approx 6.56\ \text{s}$
- 单请求 Decode 吞吐 $= 255/6.375 = 40\ \text{token/s}$。若 batch=32 并发，系统吞吐可达 $\sim 32\times40 = 1280\ \text{token/s}$（理想，受带宽约束打折）。

---

## 3. 显存账本：服务框架真正在抠的东西

GPU 显存 = 模型权重 + KV Cache + 激活/临时 buffer + 框架开销。服务框架几乎不动权重，主战场是 **KV Cache**。

### 3.1 权重显存

$$M_{weight} = N_{params}\times \text{bytes\_per\_param}$$

| 模型 | 参数量 | FP16 (2B) | INT8 (1B) | INT4 (0.5B) |
|---|---|---|---|---|
| LLaMA-7B | 7e9 | 14 GB | 7 GB | 3.5 GB |
| LLaMA-13B | 13e9 | 26 GB | 13 GB | 6.5 GB |
| LLaMA-70B | 70e9 | 140 GB | 70 GB | 35 GB |

> 量化把权重压扁，是服务框架"塞进单卡"的关键，见 [[llm-compression/quantization/量化基础]] 与 [[llm-compression/quantization/fp8]]。

### 3.2 KV Cache 显存（服务框架的主战场）

每个 token、每层、要存 K 和 V 各一份：

$$M_{kv} = 2 \times N_{layer}\times N_{kv\_head}\times d_{head}\times L_{seq}\times B \times \text{bytes}$$

其中 $2$ 是 K+V，$N_{kv\_head}\times d_{head}$ 在 MHA 下等于 $d_{model}$，GQA/MQA 下更小。

**手算（LLaMA-13B，FP16，MHA）**：$N_{layer}=40$，$d_{model}=5120$，单 token 单序列：
$$2\times40\times5120\times2\text{B} = 819{,}200\ \text{B} \approx 0.78\ \text{MB/token}$$

- 序列长 2048：$0.78\times2048 \approx 1.6\ \text{GB/请求}$
- 同时服务 32 个这样的请求：$1.6\times32 = 51.2\ \text{GB}$ 仅 KV！

```
A100 80GB 显存怎么花 (13B FP16):
┌────────────────────────────────────────────┐
│ 权重 26GB │ 临时 ~4GB │  KV Cache 可用 ~50GB │
└────────────────────────────────────────────┘
                              └─ 决定能并发多少请求
   50GB / 1.6GB ≈ 31 个 2K 长度的请求 (满载理想)
```

这就是为什么 **KV Cache 的利用效率 = 框架的吞吐天花板**。GQA（如 LLaMA-2-70B 用 8 个 KV head）能把上式 KV 砍到 1/8，详见 [[llm-algo/transformer/模型架构]]。

---

## 4. 核心瓶颈：为什么 Decode 受访存约束

判断一个 kernel 是算力受限还是访存受限，看**算术强度** $I = \dfrac{\text{FLOPs}}{\text{Bytes accessed}}$，与硬件的 ridge point $\dfrac{\text{峰值算力}}{\text{峰值带宽}}$ 比较。

**A100 ridge point**：FP16 峰值算力 312 TFLOPS，HBM 带宽 2.0 TB/s：
$$I_{ridge} = \frac{312\times10^{12}}{2.0\times10^{12}} = 156\ \text{FLOP/Byte}$$

**Decode 阶段（batch=1）一次矩阵-向量乘** $W\in\mathbb{R}^{n\times n}$ 乘向量：
- FLOPs $= 2n^2$，读取权重 Bytes $= 2n^2$（FP16）
- $I = \dfrac{2n^2}{2n^2} = 1\ \text{FLOP/Byte} \ll 156$ → **极度访存受限**

```
Roofline:
性能 ▲
峰值 ┤            ┌──────────  计算受限区 (Prefill 在这)
312T │           /
     │          /
     │         /  ← I=156 拐点
     │        /
     │       /
     │  ●---┘   I=1, Decode 卡在带宽斜坡最左端
     └──┴──────────────────► 算术强度 I
        1     156
```

**结论**：Decode 时 GPU 算力几乎闲置（利用率可能 < 5%），瓶颈是"把权重从 HBM 搬进来"。**唯一解药：增大 batch**——一次搬进的权重被 $B$ 个请求复用，算术强度提升到 $\approx B$，$B\geq156$ 时才接近算力饱和。这正是 **Continuous Batching** 的物理动机。FLOPs 估算细节见 [[llm-algo/FLOPs]]。

---

## 5. 杀手锏一：Continuous Batching（连续/动态批处理）

### 5.1 传统静态批的浪费

```
Static Batching (旧方式，请求等长才高效):
请求A (需10步) ███████████
请求B (需3步)  ███------- ← 早完成但被迫干等
请求C (需7步)  ███████---
请求D (需4步)  ████------
              └ 整批必须等最长的 A 结束才能释放/换新批
GPU 利用: 大量 '-' 是空转气泡
```

### 5.2 Continuous Batching（iteration-level scheduling）

vLLM/TGI/FT 都采用：**以"一步生成"为调度粒度**，谁完成就立刻移出、空位立刻补新请求进来。

```
Continuous Batching:
slot0: A A A A A A A A A A  (完成→换 E E E ...)
slot1: B B B[完成]E E E E E E E
slot2: C C C C C C C[完]F F F
slot3: D D D D[完]G G G G G G
       └ 任意时刻 4 个槽几乎全满，无气泡
GPU 利用率 ↑↑，吞吐可提升 2~4×
```

**收益手算**：静态批 4 请求平均长度利用率假设 60%，连续批可达 ~95%，吞吐提升 $0.95/0.60 \approx 1.6\times$；叠加可动态扩大有效 batch（空位即时补），实测常见 **2–4×**（vLLM 论文相对 FT/朴素 HF 报告更高）。

> 该思想源自 Orca（OSDI'22）的 iteration-level scheduling；TGI 称之为 continuous batching，vLLM 内置调度器实现。

---

## 6. 杀手锏二：PagedAttention / KV 分页

### 6.1 问题：KV Cache 的内存碎片

朴素实现给每个请求**预留 max_seq_len 的连续显存**，但实际输出长度未知：

```
预留 max=2048, 实际只用了 312:
请求A: [████░░░░░░░░░░░░░░░░░░░░]  内部碎片 = 浪费 84%
请求B: [██░░░░░░░░░░░░░░░░░░░░░░]
请求C: [███████░░░░░░░░░░░░░░░░░]
   └ 还有放不下整块的外部碎片
研究测得朴素方案 KV 有效利用率仅 ~20–40%
```

### 6.2 PagedAttention：把 KV Cache 当虚拟内存分页

借鉴 OS 分页：KV 切成固定大小 **block**（如 16 token/块），逻辑连续、物理可离散，用**块表（block table）**映射。

```
逻辑视图 (请求A 的 KV 序列)        物理显存池 (block 0..N)
┌────┬────┬────┬────┐            ┌──┬──┬──┬──┬──┬──┬──┐
│blk0│blk1│blk2│blk3│            │7 │A0│B0│2 │A2│A1│  │
└─┬──┴─┬──┴─┬──┴─┬──┘            └──┴──┴──┴──┴──┴──┴──┘
  │    │    │    └─► 物理块 (空闲池按需分配)
  │    │    └──────► A2
  │    └───────────► A1   (块表: A 的 logical→physical)
  └────────────────► A0
碎片只剩"最后一块的零头" → 利用率 >90%
```

**碎片手算**：block=16，请求实际长 312 token → 用 $\lceil312/16\rceil=20$ 块 = 320 槽，仅浪费 8 槽 = **2.5%**，远胜预留 2048 的 84% 浪费。

**额外红利——前缀共享（Prefix Caching）**：相同 system prompt 的多个请求，其前缀 KV block 可**只算一次、多请求共享同一物理块**（copy-on-write）。

```
请求A: [共享前缀块]→[A私有...]
请求B: [共享前缀块]→[B私有...]   ← 同一物理块, 引用计数=2
省下重复 Prefill 的算力 + 显存
```

> PagedAttention 是 vLLM 的招牌（SOSP'23）；TGI 后续也引入了 paged 风格的 KV 管理。深入见 [[llm-inference/KV-Cache优化]] 与 [[llm-optimizer/kv-cache]]。注意力核常配合 [[llm-optimizer/FlashAttention]] 的分块 softmax。

---

## 7. 杀手锏三：张量并行 / 流水并行（多卡服务）

单卡放不下（如 70B FP16 需 140GB > 80GB），必须拆到多卡。

### 7.1 张量并行（Tensor Parallel, TP）

把每层的权重矩阵**按列/行切到 N 卡**，每步前向做一次 **All-Reduce** 汇总。

```
TP=2 的一层 MLP:
   x ──┬──► [W1 左半]@GPU0 ──► [W2 上半]@GPU0 ──┐
       │                                        ├─AllReduce─► y
       └──► [W1 右半]@GPU1 ──► [W2 下半]@GPU1 ──┘
每个 Transformer 层 = 2 次 AllReduce (Attn 后 + MLP 后)
```

**通信量手算（TP=N，每层每 token，FP16）**：一次 All-Reduce 传输量 $\approx 2\times\frac{N-1}{N}\times d_{model}\times2\text{B}$。13B（$d=5120$）、TP=2：
$$2\times\tfrac{1}{2}\times5120\times2 = 10{,}240\ \text{B/层/token} \approx 10\ \text{KB}$$
40 层 ×2 次/层 $\approx 0.8\ \text{MB/token}$。这要求 GPU 间用 **NVLink（~600GB/s）**，走 PCIe（~32GB/s）会让 TP 失血。集合通信原语见 [[ai-infra/网络/集合通信原语]]，TP 推理细节见 [[llm-inference/大模型推理张量并行]]。

### 7.2 流水并行（Pipeline Parallel, PP）

按**层**切到不同卡，卡间只传一次激活（点对点），通信省但有流水气泡。推理服务里 TP 优先（低延迟），PP 用于跨节点扩展。GPU 通信硬件基础见 [[ai-infra/算力/GPU工作原理]]。

```
TP vs PP:
 TP: 切每一层(权重) → 通信频繁但量小, 要 NVLink, 延迟低 → 服务首选
 PP: 切层组       → 通信少但有气泡, 可跨机, 吞吐导向
```

---

## 8. 逐框架剖析

### 8.1 vLLM
- **底牌**：PagedAttention + Continuous Batching + Prefix Caching；OpenAI 兼容 API。
- **强项**：高吞吐、KV 利用率高、社区活跃、支持广泛量化（AWQ/GPTQ/FP8）与 TP/PP。
- **取向**：吞吐优先，通用易用，已成开源服务事实标准。
- **代价**：极致单请求低延迟场景未必胜过厂商专用编译方案。
- 仓库参考：本目录 [[llm-inference/vllm]]。

### 8.2 Huggingface TGI（Text Generation Inference）
- **底牌**：Rust 写的高性能 server + Python 模型层；continuous batching、张量并行、量化、流式输出、Prometheus 指标。
- **强项**：与 HF 生态/Hub 无缝；生产级（鉴权、限流、可观测）；多硬件后端。
- **基准维度**（其 benchmark 工具，对应原始文件）：预填充延迟、预填充吞吐（token/s）、解码总延迟、解码单 token 延迟、解码吞吐（token/s）。
- **取向**：工程化、企业部署友好。

### 8.3 FasterTransformer (FT, NVIDIA)
- **底牌**：高度手写 CUDA kernel + 算子融合 + TP/PP，早期延迟标杆。
- **强项**：极致 kernel 优化、对 Megatron-530B/GPT-175B 等超大模型的多卡基准。
- **基准维度**：固定 input=60/output=20 的延迟；每 GPU 吞吐（句子/秒）；标注 TP/PP 配置。
- **现状**：NVIDIA 已将其能力演进/并入 **TensorRT-LLM**，FT 进入维护态（以官方为准）。

### 8.4 FlexFlow Serve
- **底牌**：基于自动并行搜索的服务系统，强调**投机推理（speculative inference）**与并行策略自动化。
- **基准**：以"每秒生成 token 延迟"为指标，模型 LLaMA-30B/65B、OPT-30B，对比 vLLM/TGI/FT。
- **取向**：研究/前沿并行策略探索。

### 8.5 TensorRT-LLM（NVIDIA，现代延迟标杆）
- **底牌**：把模型编译成 TensorRT engine，in-flight batching（≈continuous batching）、paged KV、FP8/INT4 量化、定制 attention kernel。
- **强项**：在 NVIDIA GPU 上单请求**最低延迟**、最高单卡吞吐之一。
- **代价**：需编译、对模型/硬件耦合紧，灵活性低于 vLLM。

### 对比速查表

| 框架 | 核心创新 | 批处理 | KV 管理 | 量化 | 主打 | 灵活性 |
|---|---|---|---|---|---|---|
| vLLM | PagedAttention | Continuous | 分页+前缀共享 | AWQ/GPTQ/FP8 | 吞吐/通用 | 高 |
| TGI | Rust server+生产化 | Continuous | Paged 风格 | bitsandbytes/GPTQ/EETQ | 企业部署 | 高 |
| FT | 手写 CUDA 融合 | Continuous(后期) | 连续 | INT8/FP16 | 大模型多卡基准 | 中 |
| FlexFlow | 自动并行+投机 | 动态 | — | — | 研究前沿 | 中 |
| TensorRT-LLM | 编译+in-flight batch | In-flight | Paged | FP8/INT4/INT8 | 极致延迟 | 低 |

> 框架/版本演进快，上表为稳定机制层面的定性，具体支持矩阵**以各官方文档为准**。

---

## 9. 数值手算综合示例：A100-80G 上服务 LLaMA-13B

**配置**：FP16，单卡 A100 80GB，2048 上下文，benchmark prompt $L_{in}=512$、$L_{out}=256$。

**第 1 步 · 显存预算**
- 权重 $13e9\times2\text{B}=26\ \text{GB}$
- 临时/框架开销 $\approx4\ \text{GB}$
- 可用 KV $= 80-26-4 = 50\ \text{GB}$

**第 2 步 · 单请求 KV 占用**（§3.2 已算 0.78 MB/token）
- 满序列 2048：$0.78\times2048 = 1.6\ \text{GB/请求}$
- 实际平均长度 768（512+256）：$0.78\times768 = 0.6\ \text{GB/请求}$

**第 3 步 · 最大并发**
- 朴素预留 2048：$50/1.6 \approx 31$ 请求
- PagedAttention 按实际 768 分页：$50/0.6 \approx 83$ 请求 → **并发提升 ~2.7×**

**第 4 步 · Prefill 算力**（FLOPs $\approx 2\times N_{params}\times L_{in}$，见 [[llm-algo/FLOPs]]）
$$2\times13e9\times512 = 1.33\times10^{13}\ \text{FLOP}$$
A100 实效算力按峰值 50% 计 $156\ \text{TFLOPS}$：$TTFT_{compute}\approx 1.33e13/1.56e14 = 85\ \text{ms}$。

**第 5 步 · Decode 带宽**（每 token 至少搬一遍权重 26GB，HBM 2TB/s）
$$TPOT_{floor}=26e9/2e12 = 13\ \text{ms/token (batch=1)}$$
但 batch=64 时权重被复用，单 token 带宽成本摊薄，TPOT 可降到个位 ms 级（受 KV 读取与 kernel 开销约束）。

**结论**：PagedAttention 把并发从 31 拉到 83、Continuous Batching 把 GPU 从访存空转拉满 → 系统吞吐数量级提升，这正是 vLLM/TGI 相对朴素 HF pipeline 报告 数倍 throughput 的来源。

---

## 10. 选型决策树

```
                ┌─ 单卡放得下模型? ──No──► 上 TP(NVLink) / 多卡, 看 TGI/vLLM/TRT-LLM
                │
   你的目标? ───┤
                │
   ┌────────────┼─────────────┬───────────────────┐
   ▼            ▼             ▼                   ▼
最高吞吐/省钱  极致低延迟    HF生态/企业部署    研究并行策略
   │            │             │                   │
 vLLM        TensorRT-LLM    TGI               FlexFlow
(分页+连续批) (编译+FP8)    (Rust+可观测)      (自动并行+投机)
```

**经验法则**
1. 默认选 **vLLM**：通用、吞吐高、易上手，开源服务事实标准。
2. 纯 NVIDIA 卡 + 追求最低延迟/最高单卡吞吐、能接受编译流程 → **TensorRT-LLM**。
3. 深度绑定 HuggingFace、要企业级可观测与限流 → **TGI**。
4. 大模型多卡基准/历史对照 → **FT**（注意已并入 TRT-LLM 演进线）。
5. 任何框架都先把 **量化 + 长上下文的 KV 预算**算清楚（§3、§9），否则并发数会被显存卡死。

---

## 常见问题

| 问题 | 简答 |
|---|---|
| 吞吐和延迟为什么不能同时拉满? | 增大 batch 提吞吐但拉长单请求排队/计算 → §0 三角约束 |
| 为什么 Decode 慢且 GPU 闲? | batch=1 时算术强度≈1，访存受限，算力空转 → §4 |
| PagedAttention 到底省了什么? | 消灭 KV 内部/外部碎片，利用率 20–40%→>90%，并发翻倍 → §6 |
| Continuous Batching 和静态批区别? | 调度粒度从"整批"细化到"每步"，完成即换新，消除气泡 → §5 |
| 同一 system prompt 多请求能省吗? | 能，前缀 KV 共享物理块，省重复 Prefill → §6.2 |
| TP 为什么要 NVLink? | 每 token 每层多次 All-Reduce，PCIe 带宽不够会拖垮延迟 → §7.1 |
| 选 vLLM 还是 TRT-LLM? | 通用/易用选 vLLM，纯 N 卡极致延迟选 TRT-LLM → §10 |
| KV Cache 怎么估? | $2\,N_{layer}d_{model}L B \times$bytes，13B 约 0.78MB/token → §3.2 |

---

## 🔗 跳转链接

- 总览/导航：[[00-知识地图]] · [[llm-inference/README]]
- KV Cache：[[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]]
- 注意力核：[[llm-optimizer/FlashAttention]]
- 并行：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]]
- 解码：[[llm-inference/解码策略]]
- 模型结构与算量：[[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 量化：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 硬件：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 框架仓库：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
