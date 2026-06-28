# LLM 推理框架（Inference / Serving Framework）

> 一句话定位：把"一个训练好的大模型权重"变成"能同时为成千上万用户高吞吐、低延迟、稳定吐 token 的在线服务"的那一层系统软件。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/解码策略]] · [[llm-inference/大模型推理张量并行]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点：框架到底解决什么 | 调度 + 显存 + 算子 |
| 1 | 地基：自回归推理的两阶段本质 | Prefill / Decode、memory-bound |
| 2 | 框架要解决的 5 个核心问题 | 显存/批处理/算子/并行/调度 |
| 3 | PagedAttention 与 KV-Cache 分页 | 显存碎片、近零浪费 |
| 4 | Continuous Batching（连续批处理） | iteration-level 调度 |
| 5 | 算子层：FlashAttention / 融合 / 量化 | kernel、CUDA Graph |
| 6 | 分布式推理：TP / PP / EP 怎么切 | 通信量手算 |
| 7 | 投机解码 / 多 token 预测 | draft-verify |
| 8 | 主流框架横评：vLLM / TGI / TRT-LLM / SGLang / MII | 选型 |
| 9 | 性能指标体系：TTFT / TPOT / 吞吐 | SLO |
| 数值 | 显存账本 + 吞吐手算 | Roofline |
| FAQ | 常见坑与选型 | — |

## 0. 一句话锚点

推理框架 = **调度器（决定谁先算、批多大）** + **显存管理器（KV-Cache 怎么放）** + **高性能算子（attention/GEMM 怎么快）** + **分布式层（模型怎么切到多卡）**，四件套围绕一个目标：在固定的 GPU 上，把 **吞吐量（tokens/s）** 拉满，同时把 **延迟（TTFT、TPOT）** 压到 SLO 以内。

训练框架（[[ai-framework/megatron-lm/README]]、[[ai-framework/deepspeed/README]]）关心"如何最快算完一次反向传播"；推理框架关心"如何让一堆**长度不一、到达时刻不一**的请求共享一块 GPU 还都跑得快"。这是两个完全不同的调度难题。

```
        ┌────────────────────── LLM 推理框架 ──────────────────────┐
请求 →  │  [调度器]  [显存/KV管理]  [高性能算子]  [分布式TP/PP]   │ → 流式 token
        │  谁先算    KV放哪不浪费    attn/gemm快    多卡切分通信  │
        └──────────────────────────────────────────────────────────┘
                         ↑ 目标：吞吐↑  +  延迟↓
```

## 1. 地基：自回归推理的两阶段本质

LLM 解码是**自回归**的：第 $t$ 个 token 依赖前面所有 token。一次请求被切成两个性质完全不同的阶段。

### 1.1 Prefill（预填充）

把整段 prompt（设长度 $S$）一次性喂进去，**并行**算出所有位置的隐藏态，并生成第一个输出 token。这里有大矩阵乘（$S$ 个 token 同时过网络），**算力密集（compute-bound）**，GPU 利用率高。

### 1.2 Decode（解码 / 增量生成）

之后每步只输入**上一个** token（长度 1），算出下一个 token。每步的矩阵乘都是"瘦长形"（$1 \times d$ 乘权重），**访存密集（memory-bound）**：算得少，却要把整个模型权重 + 整段 KV-Cache 从显存搬一遍。

```
Prefill: [t1 t2 t3 ... tS]  ── 一次大并行 ──> 出 token t_{S+1}   (compute-bound, GPU忙)
Decode : [t_{S+1}] -> t_{S+2} -> t_{S+3} -> ...                 (memory-bound, 搬权重)
            每步只算 1 个 token，瓶颈在显存带宽
```

> 关键洞察：**Decode 阶段是访存瓶颈**。单条请求 decode 时 GPU 算力闲置 90%+，所以框架的核心命题变成"**把多条请求的 decode 攒成一个大 batch**，用足够的并行掩盖访存延迟"。这就是 continuous batching 的动机。详见 [[llm-inference/解码策略]]。

### 1.3 为什么 Decode 是 memory-bound（算术强度视角）

算术强度 $I = \dfrac{\text{FLOPs}}{\text{Bytes}}$。设模型参数量 $P$，FP16 权重 2 字节。

- batch=1 时，decode 一步约做 $2P$ FLOPs（每个参数一次乘加），需搬运 $2P$ 字节权重。$I \approx \dfrac{2P}{2P}=1$ FLOP/Byte。
- 现代 GPU 的 ridge point（算力/带宽）远大于 1，例如 A100 ≈ $\dfrac{312\text{ TFLOPS}}{2.0\text{ TB/s}} \approx 156$。$I=1 \ll 156$ → 严重 memory-bound。
- batch=$B$ 时权重只搬一次却服务 $B$ 条，$I \approx B$。要逼近 ridge point 需 $B \approx 156$。**这就是为什么要把 batch 做大**。

## 2. 框架要解决的 5 个核心问题

```
┌───────────────────────────────────────────────────────────┐
│ ① 显存管理   KV-Cache 怎么放才不浪费？        → PagedAttention│
│ ② 批处理     长度不一的请求怎么拼 batch？      → Continuous Batch│
│ ③ 高性能算子 attention/GEMM 怎么算最快？       → FlashAttn/融合│
│ ④ 分布式     放不下一张卡怎么切？             → TP / PP / EP   │
│ ⑤ 调度       谁先算、抢占、优先级、公平性？     → 调度策略       │
└───────────────────────────────────────────────────────────┘
```

后面逐个拆。

## 3. 显存管理：PagedAttention 与 KV-Cache 分页

### 3.1 KV-Cache 是什么、为什么是显存大头

Decode 时每生成一个 token，都要和**之前所有** token 做注意力。为避免重算，框架把每层每个历史 token 的 Key、Value 缓存下来 → KV-Cache。它随序列长度**线性增长**，且每条请求独占一份。详见 [[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]。

单 token 单层 KV 字节数：
$$\text{bytes} = 2 \times n_{kv} \times d_{head} \times \text{dtype}$$
（2 = K 和 V）。整条请求：
$$\text{KV} = 2 \times L \times n_{kv} \times d_{head} \times S \times B \times \text{dtype}$$
其中 $L$ 层数，$n_{kv}$ KV 头数（GQA 下小于 query 头数），$S$ 序列长，$B$ batch。

### 3.2 朴素方案的痛：碎片化

早期框架给每条请求**预留最大长度**的连续显存（如 2048）。但真实输出可能只有 50 token → 浪费 97%。还有外部碎片（连续块凑不出）。vLLM 论文测得朴素方案有效利用率仅 **20%~40%**。

### 3.3 PagedAttention：借鉴操作系统虚拟内存

把 KV-Cache 切成固定大小的 **block（页，如 16 token/页）**，物理上**不要求连续**，用一张 **block table** 把"逻辑序列位置 → 物理块号"映射起来。

```
逻辑序列(请求A):  [tok0..15][tok16..31][tok32..47]
                     │         │          │   block table
                     ▼         ▼          ▼
物理显存池:   ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
              │blk7││blk3││blk9││空 ││空 ││blk2│  (任意散布)
              └────┘└────┘└────┘└────┘└────┘└────┘
                A的页可散落各处；新请求来直接拿空块，无需连续大段
```

收益：
1. **几乎零浪费**：只在最后一页有不到 1 页的内部碎片（平均半页）。利用率 ≈ 96%+。
2. **共享（Copy-on-Write）**：beam search、并行采样、共享系统 prompt 时，多个序列可**共享同一物理块**，写时才复制 → 省大量显存。
3. 显存利用率↑ 直接换成 **batch 更大 → 吞吐更高**。

### 3.4 数值：分页省了多少显存

设 Llama-2-7B：$L=32$，$n_{kv}=32$（无 GQA），$d_{head}=128$，FP16(2B)。
单 token KV：$2 \times 32 \times 32 \times 128 \times 2 = 524288$ B = **512 KB/token**。

- 朴素预留 2048：每请求 $512\text{KB} \times 2048 = 1\text{ GB}$。实际只用 100 token → 真实需 50 MB，**浪费 95%**。
- 分页（页=16）：100 token 占 $\lceil 100/16 \rceil = 7$ 页 = 112 token 容量 = 56 MB，浪费仅最后一页约 6 MB（11%）。
- 同样 40 GB 可用显存：朴素约能放 $40/1 = 40$ 条；分页约 $40\text{GB}/56\text{MB} \approx 700$ 条。**batch 提升约 17×**（理想上限），实测吞吐提升 2~4×。

## 4. Continuous Batching（连续批处理）

### 4.1 静态批 vs 连续批

朴素**静态批**：凑齐 $B$ 条一起跑，**等最慢那条结束**才能换下一批。短请求被长请求拖死，GPU 大量空转。

**连续批（也叫 in-flight / iteration-level batching）**：调度粒度细到**每一步 decode（一个 iteration）**。某条请求生成完 EOS 立刻退出、释放它的块；空出的 slot 立刻塞进排队中的新请求。

```
静态批 (耗时被最长请求决定)：
  req1 ████████░░░░░░░░  ← 早结束却占着位
  req2 ████████████████
  req3 ████░░░░░░░░░░░░
       └ 整批等到这里才换 ┘   GPU 大量空泡

连续批 (每步动态进出)：
  step→ 1 2 3 4 5 6 7 8 ...
  slot1 A A A B B B C C   (A完→B进→...)
  slot2 D D E E E E F F
        ↑ 任意 slot 一旦空出立即被新请求填满，GPU 始终满载
```

### 4.2 Prefill 与 Decode 的混批冲突

Prefill 是大块计算、Decode 是小块计算，混在一起会互相干扰：
- **Prefill 优先**：吞吐高但新请求一来就插队，老请求的 TPOT 抖动（卡顿）。
- **Chunked Prefill（分块预填充）**：把长 prompt 的 prefill 切成小块，和 decode 一起调度，平滑延迟。vLLM、TRT-LLM 都支持。
- **PD 分离（Prefill-Decode Disaggregation）**：把 prefill 和 decode 放到**不同 GPU 池**，各自批处理，避免互扰，是当前大规模服务的趋势（如 DistServe、Mooncake）。

## 5. 算子层：让每一步算得更快

### 5.1 FlashAttention：避免实体化 $S\times S$ 矩阵

标准 attention 要写出 $S \times S$ 的分数矩阵到 HBM，$O(S^2)$ 访存。FlashAttention 用**分块 + online softmax**在 SRAM 内增量算，**不落地大矩阵**，访存降一个量级，长上下文尤其关键。深入见 [[llm-optimizer/FlashAttention]]、[[flash-attention/FlashAttention]]。Decode 阶段对应 FlashDecoding（按 KV 长度切分并行）。

### 5.2 算子融合与 CUDA Graph

- **融合（fusion）**：把 LayerNorm+残差、QKV 投影、SwiGLU 等多个小 kernel 合成一个，减少 kernel 启动开销和中间结果往返 HBM。
- **CUDA Graph**：decode 每步 kernel 序列固定，录制成图一次性重放，省掉每步 CPU 启动 kernel 的开销（小 batch 时 launch 开销占比可观）。
- **PagedAttention kernel**：专门处理"非连续块"的 attention kernel，是分页能落地的前提。

### 5.3 量化推理

把权重/激活/KV 从 FP16 降到 INT8/FP8/INT4，**减少访存字节数**——正好打 decode 的 memory-bound 痛点。详见 [[llm-compression/quantization/量化基础]]、[[llm-compression/quantization/fp8]]。

```
权重量化 INT4：搬运字节 ÷4 → decode 吞吐近似 ×?(受带宽支配)
KV-Cache 量化 FP8：KV 显存减半 → batch 更大
注意：低比特需反量化 kernel，compute-bound 阶段(prefill)可能不划算
```

## 6. 分布式推理：放不下一张卡怎么办

当模型 + KV 超过单卡显存，必须切分。原理同训练，但推理更看重**通信延迟**（在 decode 关键路径上）。

### 6.1 张量并行 TP（层内切）

把每层的权重矩阵按列/行切到 $N$ 张卡，每张算一部分，用 **AllReduce** 合并。延迟低但每层都要通信，要求卡间高带宽（NVLink）。详见 [[llm-inference/大模型推理张量并行]]、[[ai-infra/网络/集合通信原语]]。

### 6.2 流水线并行 PP（层间切）

把不同**层**放到不同卡，激活 P2P 传递。通信量小（只传激活），但推理 batch 不像训练能填满流水线，易有气泡，更适合跨节点。

### 6.3 专家并行 EP（MoE 专用）

MoE 模型把不同 expert 放到不同卡，token 经 **All-to-All** 路由到对应 expert。详见 [[llm-algo/moe/README]]。

### 6.4 数值：TP=2 每 token 的通信量

Transformer 每层有 2 个 AllReduce（attention 后、MLP 后）。Ring-AllReduce 在 $N$ 卡上每卡收发约 $2\cdot\frac{N-1}{N}\cdot M$ 字节（$M$=数据量）。

decode 单 token、隐藏维 $d=4096$、FP16(2B)、$L=32$ 层：
- 每次 AllReduce 数据 $M = d \times 2 = 8192$ B。
- 每层 2 次，每 token 通信回合 $= 2L = 64$ 次。
- $N=2$：每卡每 token 收发 $\approx 64 \times 2\times\frac{1}{2}\times 8192 = 64 \times 8192 \approx 512$ KB。
- NVLink 300 GB/s 下纯传输 $\approx \frac{512\text{KB}}{300\text{GB/s}} \approx 1.7\,\mu s$/token，但**延迟（latency）+ 同步开销**在小消息上才是主导，这正是 TP 要 NVLink、跨 PCIe 会明显掉速的原因。计算/通信重叠见 [[llm-optimizer/计算通信重叠]]。

```
TP=2 单层 decode：
  GPU0 ──算半个MLP──┐                  ┌── 继续下一层
                    ├─ AllReduce(同步)─┤
  GPU1 ──算半个MLP──┘                  └──
  每层 2 次同步，全在 decode 关键路径上 → 卡间带宽即吞吐天花板
```

## 7. 投机解码（Speculative Decoding）

用一个**小的 draft 模型**（或多预测头 Medusa、EAGLE）一次猜 $k$ 个 token，再用**大模型一次并行 verify**。验证通过的就白赚，验证失败从分歧点重来。把"逐 token 串行"变"一次验多个"，把 memory-bound 的 decode 变得更 compute-bound，吞吐可提升 2~3×而**不改变输出分布**（用 rejection sampling 保证）。详见 [[llm-inference/解码策略]]、`docs/llm-inference/flexflow/投机采样.md`。

```
draft 小模型: 猜 [a b c d]   (便宜，串行 4 步但每步极快)
大模型 verify: 一次并行验 [a b c d] → 接受 [a b] 拒 c
净收益: 一次大模型前向产出 ≥1 个 token (期望 1+ 接受数)
```

## 8. 主流框架横评

> 护栏：以下为稳定的设计取向，**功能/默认值/版本以各官方文档为准**。

| 框架 | 出身 | 杀手锏 | 适用场景 |
|------|------|--------|----------|
| **vLLM** | UC Berkeley | PagedAttention + 连续批，吞吐标杆 | 大批量、高吞吐在线服务；社区生态最广 |
| **HF TGI** | HuggingFace | 与 HF 生态无缝、生产级特性全 | 依赖 HF 模型、要开箱即用的服务 |
| **TensorRT-LLM** | NVIDIA | 编译式极致 kernel + FP8，NV 卡上极快 | 追求单卡极限延迟/吞吐、绑定 N 卡 |
| **SGLang** | — | RadixAttention（前缀树共享 KV）、结构化输出快 | 多轮/共享前缀、Agent、约束解码 |
| **DeepSpeed-MII** | Microsoft | 基于 DeepSpeed-Inference 的张量并行 + 内核 | 已用 DeepSpeed 栈、大模型多卡 |
| **LMDeploy** | OpenMMLab | TurboMind 引擎、量化强 | 国产化、量化部署 |

详细对比见 [[llm-inference/LLM服务框架对比]]。

### 8.1 一句话取向

- 要**最大吞吐 + 活跃社区** → vLLM。
- 要 **NV 卡上的极限性能**、能接受编译流程 → TensorRT-LLM。
- 要**共享前缀 / Agent / 约束解码** → SGLang。
- 已在 **HF / DeepSpeed 栈** → TGI / DeepSpeed-MII 顺手。

```
吞吐 ▲           TRT-LLM ● (NV卡, 编译重)
     │   vLLM ● ────● SGLang (前缀共享)
     │     TGI ●
     │ DeepSpeed-MII ●
     └───────────────────────────► 易用性/通用性
```

## 9. 性能指标体系

| 指标 | 含义 | 受谁支配 |
|------|------|----------|
| **TTFT** (Time To First Token) | 首 token 延迟 | prefill 时长 + 排队 |
| **TPOT** (Time Per Output Token) | 每后续 token 延迟 | decode 访存带宽 + batch |
| **吞吐 Throughput** | 全系统 tokens/s | batch 大小 + 显存利用率 |
| **并发 / Goodput** | 满足 SLO 的有效吞吐 | 调度策略 |

吞吐与延迟是**权衡**：batch 越大吞吐越高，但单请求 TPOT 越长。框架的工作就是在 SLO 约束下把 batch 推到最大。

## 数值手算：一张 A100-80G 能服务多少并发？

以 Llama-2-13B、FP16、序列均 1024、A100-80GB 为例。

1. **权重**：13B × 2B = **26 GB**。
2. **激活/框架开销**：留约 **4 GB**。
3. **KV 可用**：$80 - 26 - 4 = 50$ GB。
4. **单请求 KV**（$L=40, n_{kv}=40, d_{head}=128$，无 GQA）：
   单 token = $2\times40\times40\times128\times2 = 1.6$ MB/token。
   1024 token = **1.6 GB/请求**。
5. **理论并发** $= 50\text{ GB} / 1.6\text{ GB} \approx 31$ 条。
6. 若 KV 用 **FP8**（1B）：单请求 0.8 GB → **62 条**，并发翻倍。
7. 若换 **GQA**（如 $n_{kv}=8$）：单 token = $2\times40\times8\times128\times2=0.32$ MB → 1024 token = 0.32 GB → 并发约 **156 条**，再 5×。

> 结论：决定并发上限的不是算力而是 **KV 显存**。所以现代框架的优化几乎都围绕"压 KV"：分页、量化、GQA、前缀共享。这也是 [[llm-algo/transformer/模型架构]] 里 GQA 设计的工程动机。

## 常见问题

| 问题 | 答案 |
|------|------|
| vLLM 为什么快？ | PagedAttention（显存近零浪费 → 大 batch）+ continuous batching（GPU 不空转）+ 高效 kernel 三者叠加。 |
| Prefill 和 Decode 哪个是瓶颈？ | 短输出场景 prefill 占比高；长输出场景 decode 主导且是 memory-bound，框架重点优化 decode。 |
| 为什么单请求推理 GPU 利用率很低？ | decode batch=1 时算术强度 ≈1，远低于 GPU ridge point，纯访存等待。需攒 batch。 |
| TP 还是 PP？ | 单节点内 NVLink 充足用 TP（低延迟）；跨节点带宽差用 PP（少通信），常 TP+PP 混合。 |
| 量化一定更快吗？ | KV/权重量化减访存对 memory-bound 的 decode 有效；但反量化有开销，compute-bound 的 prefill 未必划算，需实测。 |
| 投机解码会降低质量吗？ | 不会。verify 阶段用 rejection sampling 保证输出分布与原模型一致，只省时间。 |
| chunked prefill 解决什么？ | 把长 prompt 的 prefill 切块与 decode 混批，避免新请求插队造成老请求 TPOT 抖动。 |
| 为什么要 PD 分离？ | prefill（compute-bound）与 decode（memory-bound）特性冲突，分到不同 GPU 池各自优化，提升整体 goodput。 |

## 🔗 跳转链接

- 总览/导航：[[00-知识地图]] · [[llm-inference/README]]
- KV-Cache 原理与优化：[[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]]
- 解码与投机采样：[[llm-inference/解码策略]] · `docs/llm-inference/flexflow/投机采样.md`
- 张量并行与通信：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]]
- 注意力算子：[[llm-optimizer/FlashAttention]] · [[flash-attention/FlashAttention]]
- 量化部署：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 模型结构与算量：[[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 框架横评：[[llm-inference/LLM服务框架对比]]
- 训练栈对照：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 硬件基础：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
