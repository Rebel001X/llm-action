# FlashInfer (注意力kernel库)

> FlashInfer 是一个专为 **LLM 推理服务（serving）** 设计的注意力（Attention）算子库，把 Prefill / Decode / 多种 KV 布局（含 PagedKV）/ 融合 RoPE / 量化 KV-Cache / MLA 等全部做成高度优化的 GPU kernel，被 vLLM、SGLang、TGI 等主流推理引擎采纳为底层注意力后端。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-optimizer/FlashAttention]] [[llm-inference/Flash-Decoding]] [[llm-inference/vllm/README]]

---

## 阅读地图

| 节 | 你会搞懂 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点 | serving 注意力 kernel 库 |
| 1 | 地基：为什么 serving 需要专用 kernel | 训练 vs 推理、变长 batch、PagedKV |
| 2 | 注意力本身回顾（最底层） | $QK^\top$、softmax、online softmax |
| 3 | FlashAttention 是什么、FlashInfer 与它的关系 | tiling、IO 感知、谁负责训练谁负责推理 |
| 4 | Prefill vs Decode 两种形态 | compute-bound vs memory-bound |
| 5 | KV 布局：连续 / Ragged / PagedKV | 内存碎片、分页、prefetch |
| 6 | 核心机制①：用 Tensor Core 跑 GQA Decode | CUDA Core 算力墙、低算术强度 |
| 7 | 核心机制②：融合 RoPE | 为什么 RoPE 不能用 Tensor Core |
| 8 | 核心机制③：量化 KV-Cache（4-bit） | 显存换带宽 |
| 9 | 核心机制④：负载均衡调度与 Plan/Run | 变长序列、CTA 切分 |
| 10 | MLA（DeepSeek）的特殊内核 | 吸收矩阵、KV 压缩 |
| — | 数值例子 / 典型场景 | 显存 + 带宽手算 |
| — | 对照表 / 常见问题 / 跳转 | 选型决策 |

---

## 0. 一句话锚点

> **FlashAttention 是「单个注意力算子怎么算得快」；FlashInfer 是「在一个真实推理服务里，面对一堆长短不一、还在分页（paging）的请求，注意力这一步整体怎么算得又快又省」。**

把它记成一句话：**FlashInfer = 面向 serving 的注意力 kernel 全家桶 + 调度器**。它不是一个新的注意力算法，而是把注意力在「推理服务」这个特定场景下的所有变体（Prefill/Decode、GQA/MHA/MLA、连续/分页 KV、是否融合 RoPE、是否量化）全部实现成可调用的高性能 CUDA kernel，并提供一套 Python/PyTorch API。

---

## 1. 地基：它解决什么问题

### 1.1 先分清「训练」和「推理服务」是两个世界

很多人以为「注意力 kernel 不就是 FlashAttention 吗，训练能用推理也能用」。**不对。** 训练和推理服务的负载特征差别巨大：

```
                训练 (training)            推理服务 (serving)
            ┌─────────────────────┐   ┌───────────────────────────┐
形态        │ 一个大 batch        │   │ 很多请求, 长短不一        │
            │ 序列长度对齐/padding │   │ 序列长度天差地别(8~32k)   │
每步算什么  │ 完整前向+反向        │   │ 只前向; 且分两阶段        │
            │                     │   │  Prefill(整段)+Decode(逐token)│
KV 怎么存   │ 临时, 算完就丢       │   │ KV-Cache 要长期保留+复用  │
            │                     │   │ 还要分页(PagedAttention)  │
瓶颈        │ 算力(compute-bound) │   │ Decode 是带宽(memory-bound)│
            └─────────────────────┘   └───────────────────────────┘
```

**核心矛盾：** FlashAttention 这类算子是为「训练 / 长序列前向」优化的，假设输入是规整的、连续的张量。但推理服务里：

1. **请求长度高度不均**：一个 batch 里可能有人问 8 个 token，有人贴了 20k token 的文档。统一 padding 到最长会浪费巨量算力。
2. **Decode 阶段每次只产生 1 个新 token**：query 长度 = 1，这让注意力从「矩阵×矩阵」退化成「向量×矩阵」，**算术强度（arithmetic intensity）极低**，瓶颈从算力变成显存带宽。
3. **KV-Cache 被分页（Paging）管理**：[[llm-inference/vllm/README]] 的 PagedAttention 把 KV-Cache 切成固定大小的「块（block/page）」，物理上不连续。普通 kernel 假设 KV 是一整块连续内存，根本没法直接读分页 KV。

### 1.2 所以「serving 专用 kernel」要补的三件事

```
FlashAttention 给的:  规整连续输入的快速注意力
                              │
serving 还缺:                 ▼
   ① 处理变长 batch (Ragged)，不 padding 也能算
   ② Decode (q_len=1) 的极致带宽优化（GQA/Flash-Decoding 思路）
   ③ 读「分页、不连续」的 PagedKV，而且要快
   ④ 把 RoPE、量化等顺手融进 kernel，少一次显存往返
   ⑤ 面对一堆不等长请求，做负载均衡调度
                              │
                              ▼
                这些正是 FlashInfer 干的事
```

> 一句话：**FlashInfer 把「推理服务里注意力会遇到的所有不规整 + 所有变体」一次性 kernel 化了。**

---

## 2. 把注意力拆到最原子（不假设你记得公式）

注意力的输入是三组向量：Query（$Q$）、Key（$K$）、Value（$V$）。对第 $i$ 个 query：

$$
\text{Attn}(q_i) = \sum_{j} \underbrace{\frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d})}}_{\text{权重 } a_{ij}} \; v_j
$$

逐步说「为什么」：

1. $q_i \cdot k_j$：query 和每个 key 做点积，衡量「第 $i$ 个位置该多关注第 $j$ 个位置」。
2. $/\sqrt{d}$（$d$ 是 head 维度）：防止点积随维度变大而过大，使 softmax 不至于饱和。
3. $\exp(\cdot)$ 再归一化：把分数变成和为 1 的概率权重 $a_{ij}$。
4. $\sum_j a_{ij} v_j$：用权重对 value 做加权平均，得到输出。

**朴素实现的致命点**：要先算出完整的 $S = QK^\top$（大小 $n\times n$），写回显存，再读回来做 softmax，再乘 $V$。当 $n$ 很大时，这个 $n\times n$ 中间矩阵的**显存读写**就是瓶颈。

**Online softmax（在线 softmax）** 是关键技巧：不一次性算完整行，而是分块（tile）流式处理，边读 $K/V$ 块边更新「当前最大值 $m$、当前分母 $\ell$、当前累加输出 $o$」，从而**永远不把 $n\times n$ 中间矩阵落到显存**。这就是 FlashAttention / FlashInfer 共同的数学地基。

```
朴素:  Q,K → S(n×n) →写显存→读回→softmax→×V    (中间矩阵爆显存带宽)
Flash: 分块流式, 在 SRAM 里 online-softmax, S 永不落地
```

---

## 3. FlashAttention 是什么 + FlashInfer 与它的关系

### 3.1 FlashAttention 一句话

[[llm-optimizer/FlashAttention]] 是一种 **IO 感知（IO-aware）** 的精确注意力算法：通过 tiling + online softmax，把注意力的中间矩阵留在片上 SRAM，**大幅减少对 HBM（显存）的读写**，从而把注意力从「带宽受限」拉回「算力受限」，速度和显存都优化。它主要面向**训练**和**长序列 Prefill**。

### 3.2 二者关系（最容易混淆，重点讲）

```
        ┌──────────────────────────────────────────────┐
        │  共同地基: tiling + online-softmax (Flash 思想) │
        └──────────────────────────────────────────────┘
                 │                              │
                 ▼                              ▼
   ┌───────────────────────┐    ┌──────────────────────────────────┐
   │   FlashAttention      │    │           FlashInfer             │
   │ ─ 偏训练/长序列前向    │    │ ─ 偏推理 serving                 │
   │ ─ 规整连续输入         │    │ ─ 变长 batch / Ragged / PagedKV  │
   │ ─ 单一最优 kernel      │    │ ─ Prefill+Decode+GQA+MLA+量化全套│
   │ ─ 有反向传播           │    │ ─ 通常只前向, 但带调度器         │
   └───────────────────────┘    └──────────────────────────────────┘
         "怎么算得快"                 "在服务里整体怎么又快又省"
```

要点对比：

- **不是替代关系，是分工**：FlashAttention 是「算法 + 极致单 kernel」；FlashInfer 是「面向 serving 场景的 kernel 库 + 调度」，它**继承** Flash 的数学思想，但解决的是「不规整 + 多变体 + 分页」这些 FlashAttention 不直接管的工程问题。
- **Flash-Decoding 的位置**：[[llm-inference/Flash-Decoding]] 是专为 Decode（$q\_len=1$）阶段把 KV 维度**切块并行 + 跨块归约**的技巧，解决 Decode 时 SM（流多处理器）利用率低的问题。FlashInfer 的 Decode kernel **吸收并推广了 Flash-Decoding 的思想**，并扩展到 PagedKV / GQA / 量化。
- **谁被谁用**：推理引擎（vLLM/SGLang）在 serving 时，注意力这一步可以**选择**用 FlashInfer 作为后端；而 FlashInfer 内部 kernel 又用到 FlashAttention 式的 online softmax。

---

## 4. Prefill vs Decode：两种完全不同的形态

LLM 推理分两阶段，注意力在两阶段的特征**截然相反**，这是理解 FlashInfer 为何要分别优化的关键。

```
请求: "请总结这篇文章 <20k token 文档>"  然后逐字生成回答

阶段1  Prefill (预填充)          阶段2  Decode (解码/自回归生成)
┌─────────────────────────┐    ┌──────────────────────────────┐
│ 一次性处理整段 prompt    │    │ 每步只生成 1 个新 token      │
│ q_len = 20000            │    │ q_len = 1                    │
│ 大矩阵 × 大矩阵          │    │ 1个向量 × 全部历史 KV        │
│ 算术强度高 → 算力受限    │    │ 算术强度极低 → 带宽受限      │
│ ✦ 适合 Tensor Core 满载  │    │ ✦ 难点: 读海量 KV, SM 闲置   │
└─────────────────────────┘    └──────────────────────────────┘
   FlashInfer: Prefill kernel       FlashInfer: Decode kernel
   (类 FlashAttention)               (类 Flash-Decoding + PagedKV)
```

**为什么 Decode 是带宽受限？** 设 batch=1、序列已有 $L$ 个 token、隐层 $d$、层数 $\ell$。生成 1 个新 token，注意力要**读取全部 KV-Cache**（$\propto L$），但只做 $\propto L$ 次乘加；每读 1 个数只算几次浮点 → 算术强度 $\approx O(1)$，远低于 GPU 的「计算/带宽」比（动辄上百），所以瓶颈是「把 KV 从显存搬进来」的带宽，而非算力。**优化方向必然是：减少 KV 读取量（量化）、提高读取并行度（Flash-Decoding 切块）、减少冗余搬运（GQA 共享 KV）。**

---

## 5. KV 布局：连续 / Ragged / PagedKV

FlashInfer 的一大卖点是**原生支持多种 KV 布局**，尤其是 PagedKV。先看三种布局长什么样：

```
① 连续 (Contiguous, padding 到等长)  —— 简单但浪费
  req0 |■■■■□□□□|   (□ = padding 浪费)
  req1 |■■■■■■■■|
  req2 |■□□□□□□□|

② Ragged (变长拼接, 用 indptr 指明每条边界) —— 不浪费 padding
  data:  |req0 req0 req0 req0|req1...req1|req2|
  indptr: 0 ───────────────► 4 ────────► 12 ► 13   (前缀和指针)
  好处: Prefill 时不 padding, 一个 kernel 处理整个不等长 batch

③ PagedKV (分页, vLLM 风格) —— 解决显存碎片 + 动态增长
  逻辑序列(连续) ─映射─► 物理块表(page table, 可不连续)
  req0: [blk7][blk2][blk9]...      物理显存被切成固定大小 block
  req1: [blk3][blk1]...            一条序列的 KV 散落在不同 block
        每个 block 存 (block_size) 个 token 的 K 和 V
```

**为什么需要 PagedKV？** 见 [[llm-inference/vllm/README]]：传统做法给每个请求预留「最大长度」的连续 KV 显存，导致**内部碎片**（预留多用得少）和**外部碎片**（连续大块难凑）。PagedAttention 借鉴操作系统虚拟内存的「分页」思想，把 KV 切成固定大小的物理块按需分配，显存利用率从 ~20-40% 提到 ~90%+，还能让多请求**共享前缀**（prefix sharing）。

**代价**：物理块不连续。普通 kernel 顺序读连续 KV，而 PagedKV 要先查**块表（page table / block table）** 才知道下一段 KV 在哪。这会引入**间接寻址延迟**。FlashInfer 的关键优化（来自原文）：

> **PageAttention 实现「预取（prefetch）」了页表结构的索引，最小化 page 大小对算子性能的影响。** 也就是说，在计算当前块时就提前把下一个块的索引/数据预取进来，把间接寻址的延迟「藏」在计算后面，于是即便 page_size 很小（碎片更少）也不掉性能。

```
不预取:  查表→等→读blk→算  查表→等→读blk→算   (等待暴露, 慢)
预取:    查表→读blk→算
              ↘ 同时预取下一个 blk 的索引/数据
         查表(已就绪)→读→算                      (延迟被藏起来)
```

---

## 6. 核心机制① — 用 Tensor Core 跑 GQA Decode（重点）

这是 FlashInfer 论文最核心的贡献之一，也是原文强调的点。

### 6.1 先懂 GQA

```
MHA(多头注意力):   每个 Q head 配 1 个独立 KV head   (KV 多, 显存大)
        Q: ●●●●●●●●  (8 个 query head)
        KV:■■■■■■■■  (8 套 KV, 各自独立)

GQA(分组查询注意力): 多个 Q head 共享 1 套 KV head    (KV 少, 省显存/带宽)
        Q: ●●●● ●●●● (8 个 query head, 分 2 组)
        KV:  ■    ■   (只 2 套 KV, 组内共享)   ← Llama/大模型常用
```

GQA 让多个 query head 共享同一份 KV，**KV-Cache 显存和读取带宽都成倍下降**，对带宽受限的 Decode 是大利好。

### 6.2 问题：Decode 时 GQA 的「算力墙」

朴素 GQA Decode 用 **CUDA Core**（普通浮点单元）实现。但 Decode 时 $q\_len=1$，每个 query head 只有 1 行，矩阵退化成「向量 × 矩阵」，**没法喂饱 Tensor Core**（Tensor Core 要的是矩阵×矩阵的块），只能用 CUDA Core，而 CUDA Core 的峰值算力远低于 Tensor Core → **被算力所限制**。

### 6.3 FlashInfer 的解法：把 GQA Decode「伪装成 Prefill 形状」喂 Tensor Core

原文：

> 使用 CUDA Cores 的传统 GQA 实现会被算力所限制。FlashInfer 提出使用**预填充阶段的自注意力内核（用 Tensor Core 实现）** 来做 GQA 的**解码**自注意力操作。

直觉：GQA 里**一组内的多个 query head 共享同一套 KV**。虽然每个 head 的 $q\_len=1$，但「同组的 $g$ 个 head」可以**在 head 维度上拼成一个 $g\times d$ 的小矩阵**，于是「向量×矩阵」就变回了「小矩阵×矩阵」，**重新具备喂 Tensor Core 的形状**，从而用上算力更高的 Tensor Core 内核。

```
朴素:  每个 head 单独 [1×d] × [d×L]  → 向量乘, 只能 CUDA Core
                                       ↓ 算力受限

FlashInfer: 同组 g 个共享 KV 的 head 堆叠
       [g×d] × [d×L]  → 变成矩阵乘   → 喂给 Tensor Core (Prefill kernel)
                                       ↑ 吃满算力, 更快
```

**为什么这样可行又划算**：因为这 $g$ 个 head 读的是**同一份 KV**（GQA 的本质），把它们打包成矩阵不会增加 KV 读取量，却把计算从低效的 CUDA Core 迁到了高效的 Tensor Core。group_size 越大（如 GQA 8:1），收益越明显。

---

## 7. 核心机制② — 融合 RoPE（fused RoPE）

### 7.1 RoPE 是什么

旋转位置编码（Rotary Position Embedding）：通过对 $Q/K$ 向量按位置**旋转一个角度**来注入位置信息。每个二维子空间 $(x_1,x_2)$ 旋转角 $\theta$：

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} =
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

### 7.2 为什么 RoPE「不能用 Tensor Core」

原文一句关键话：

> **RoPE 需要 sin/cos 等计算，不能使用 Tensor Cores 加速。**

Tensor Core 只会干一件事：**矩阵乘加（MMA）**。而 RoPE 要算 $\sin\theta,\cos\theta$（超越函数）、做逐元素的旋转——这些是**逐元素 / 特殊函数**运算，属于 CUDA Core / SFU（特殊函数单元）的活儿，Tensor Core 无能为力。

### 7.3 所以要「融合（fuse）」

如果 RoPE 单独跑一个 kernel：要把 $Q/K$ 从显存读出 → 旋转 → 写回显存 → 注意力 kernel 再读回来。多一次**显存往返**，在带宽受限的 Decode 里很亏。

FlashInfer 把 RoPE **融合进注意力 kernel**：在注意力 kernel 已经把 $Q/K$ 块加载进片上 SRAM 之后，**就地（on-the-fly）** 旋转，再立刻参与 $QK^\top$。省掉中间的显存往返。

```
不融合:  [读Q,K]→RoPE kernel→[写回]→[再读]→Attn kernel→out   (两次往返)
融合:    [读Q,K 进 SRAM]→就地 RoPE→QK^T→softmax→×V→out      (一次往返)
```

---

## 8. 核心机制③ — 量化 KV-Cache（4-bit）

原文：**量化自注意力，KV-Cache 4bit。**

### 8.1 动机：Decode 是带宽受限，KV-Cache 越小越快

KV-Cache 既占**显存**（决定能放多长上下文、多大 batch），又决定每步 Decode 要**读多少字节**（决定速度）。把 KV 从 FP16（2 字节）量化到 INT4（0.5 字节），**显存和读取带宽直接降到 1/4**。

```
KV-Cache 精度对单 token KV 字节数 (per layer, MHA, d_head=128, n_kv_head)
FP16:  2 bytes/elem   ████████  100%
INT8:  1 byte /elem   ████       50%
INT4:  0.5 byte/elem  ██         25%   ← 4× 更省, Decode 更快
```

### 8.2 代价与权衡

- **省**：显存 4×、带宽 4×、能跑更长上下文 / 更大 batch。
- **付**：① 精度损失（需 per-channel / per-token scale 等手段控制误差）；② kernel 里要做**反量化（dequant）**，FlashInfer 把反量化**融合**进注意力 kernel（读进来即解，不额外往返）。
- **何时用**：长上下文、显存吃紧、对极致精度不那么敏感的 serving 场景。

---

## 9. 核心机制④ — 负载均衡调度（Plan / Run 两段式）

变长 batch 带来一个工程难题：**怎么把不等长的请求均匀地分给 GPU 的众多 SM？** 若简单「一个请求一个 block（CTA）」，长请求拖慢、短请求闲置，SM 利用率差。

FlashInfer 的典型用法是**两段式**（理念，具体 API 以官方文档为准）：

```
Plan 阶段 (一次, 拿到各请求长度/indptr/block table 后):
   ┌──────────────────────────────────────────────┐
   │ 根据每条序列长度, 把"总工作量"切成均匀的块,    │
   │ 规划好每个 CTA(线程块)负责哪段 KV → 负载均衡   │
   │ 长序列被拆给多个 CTA(Flash-Decoding 式切分)    │
   └──────────────────────────────────────────────┘
                       │ 产出调度计划(可复用)
                       ▼
Run 阶段 (每步 Decode 反复调用, 复用 Plan 结果):
   ┌──────────────────────────────────────────────┐
   │ 按计划并行执行 attention, 跨块 online-softmax  │
   │ 归约得到每个 query 的最终输出                  │
   └──────────────────────────────────────────────┘
```

好处：**把「怎么切分/调度」的开销摊到 Plan 一次**，Decode 每步只跑高效的 Run，且做到 SM 负载均衡。这正是「serving 专用」相比「单一 FlashAttention kernel」多出来的一层。

---

## 10. MLA（DeepSeek）的特殊内核

DeepSeek 的 **MLA（Multi-head Latent Attention）** 把 KV 压缩成一个低维**潜在向量（latent）**，再在注意力里通过**吸收矩阵**展开，从而把 KV-Cache 压得极小。它的注意力形状和 MHA/GQA 不同，需要**专门的 kernel**。FlashInfer 为 MLA 设计了专用内核（详见原文链接的「FlashInfer 中 DeepSeek MLA 的内核设计」），处理「读压缩 KV + 吸收矩阵 + 高效 Decode」这套独特数据流。要点：

- MLA 的 KV-Cache 比 GQA 还小，**带宽收益更大**，但计算图更复杂（多了吸收/上投影），需要定制 kernel 才能既省又快。
- 这也再次说明 FlashInfer 的定位：**注意力一旦出现新变体（GQA→MLA），serving 就需要对应的专用 kernel，而 FlashInfer 就是「收纳这些变体」的库。**

---

## 数值例子 / 典型场景（手算）

**场景：Llama-3 70B 风格模型，GQA，Decode 阶段的 KV-Cache 显存与带宽。**

设定（用于演示，非精确指定）：层数 $\ell=80$，注意力隐藏维 $d=8192$，GQA 把 KV head 压到 query head 的 $1/8$，故 **KV 维度** $d_{kv} = 8192/8 = 1024$，KV 精度 FP16（2 字节），$K$ 和 $V$ 各一份。

**① 每 token、每层的 KV 字节数：**
$$
2(\text{K,V}) \times d_{kv} \times 2\,\text{B} = 2 \times 1024 \times 2 = 4096\ \text{B} = 4\ \text{KB}
$$

**② 整模型每 token KV：**
$$
4\ \text{KB} \times 80\ \text{层} = 320\ \text{KB / token}
$$

**③ 上下文 $L=8192$ token 的单条序列 KV-Cache：**
$$
320\ \text{KB} \times 8192 \approx 2.62\ \text{GB}
$$

**④ Decode 每步要读多少 KV（带宽视角）：** 生成 1 个 token 需读全部历史 KV ≈ **2.62 GB**。若 GPU 显存带宽 $\approx 3.35\ \text{TB/s}$（HBM3 量级），仅「读 KV」的时间下界：
$$
\frac{2.62\ \text{GB}}{3350\ \text{GB/s}} \approx 0.78\ \text{ms / token}
$$
对应单序列上限 $\approx 1280\ \text{token/s}$ —— **这就是 Decode 被带宽锁死的铁证**（计算时间远小于此）。

**⑤ 量化到 INT4 的收益：** KV 字节 ×1/4 → 单序列 KV 降到 $\approx 0.66\ \text{GB}$，读 KV 时间降到 $\approx 0.20\ \text{ms}$，理论 token/s **提升约 4×**。这正是 FlashInfer 量化 KV-Cache 的价值。

**⑥ GQA 的对比：** 若退回 MHA（KV 不压缩，$d_{kv}=8192$），上面每 token KV 变 8×→ $2.56\ \text{MB/token}$，$L=8192$ 单序列 KV ≈ **21 GB**，Decode 读 KV 时间 ≈ 6.3 ms/token。**GQA 直接把 Decode 加速约 8×、显存省约 8×** —— 这解释了为何大模型普遍用 GQA，也解释了 FlashInfer 为何要专门优化 GQA Decode。

---

## 对照表（与同类对比）

| 维度 | FlashAttention | Flash-Decoding | FlashInfer |
|------|----------------|----------------|------------|
| 定位 | 单注意力算子极致优化 | Decode 阶段并行技巧 | serving 注意力 kernel 库+调度 |
| 主战场 | 训练 / 长序列 Prefill | 推理 Decode | 推理服务全流程 |
| q_len | 大（Prefill） | =1（Decode） | 两者都覆盖 |
| 变长 batch | 需 padding/特殊处理 | 不直接管 | 原生 Ragged 支持 |
| PagedKV | 不原生 | 不原生 | 原生 + 索引 prefetch |
| GQA Decode | — | 部分 | Tensor Core 化（重点） |
| 融合 RoPE | 部分实现有 | 否 | 是（就地融合） |
| 量化 KV | 否 | 否 | 是（含 4-bit） |
| MLA | 否 | 否 | 专用 kernel |
| 谁用它 | 训练框架/各引擎 | 部分引擎 | vLLM / SGLang / TGI 等 |

> 一句话选型：**写训练 kernel 或长 Prefill → FlashAttention 思路；做 LLM serving、要 PagedKV/变长/GQA/量化 → 用 FlashInfer 当注意力后端。** 它们不是竞争，而是层层包含（FlashInfer 内部就用 Flash 思想）。

---

## 常见问题

| 问题 | 回答 |
|------|------|
| FlashInfer 是新注意力算法吗？ | 不是。它是 serving 场景的注意力**kernel 库 + 调度器**，沿用 FlashAttention 的 online-softmax 数学地基。 |
| 它和 vLLM 是竞争关系吗？ | 不是。vLLM 是**推理引擎**（含调度/批处理/PagedKV 管理）；FlashInfer 是它可选的**注意力 kernel 后端**。vLLM/SGLang 采用 FlashInfer 加速注意力。 |
| 为什么 Decode 要专用 kernel？ | Decode 的 $q\_len=1$ 使注意力退化为向量×矩阵，算术强度极低 → **带宽受限**；要用 Flash-Decoding 切块、GQA、量化、PagedKV 等专门手段。 |
| 为什么 GQA Decode 要用 Tensor Core？ | 朴素 GQA 用 CUDA Core 受算力限制；把同组共享 KV 的多个 head 堆成小矩阵，恢复矩阵乘形状即可喂 Tensor Core 提速。 |
| 为什么 RoPE 不能 Tensor Core？ | RoPE 含 sin/cos 与逐元素旋转，属特殊函数/逐元素运算，Tensor Core 只做矩阵乘加，故只能 CUDA Core；因此**融合进注意力 kernel**省显存往返。 |
| PagedKV 为什么要 prefetch 索引？ | 分页后物理块不连续，需查块表间接寻址；**预取**下一块索引/数据把延迟藏在计算后，使小 page_size 也不掉性能。 |
| 4-bit KV 会掉精度吗？ | 会有损失，靠 per-channel/per-token scale 等控制；换来显存与带宽 4× 收益，适合长上下文/显存紧张场景。 |
| MLA 为什么要单独 kernel？ | MLA 把 KV 压成低维 latent + 吸收矩阵，数据流与 MHA/GQA 不同，需定制 kernel 才能又省又快。 |
| 版本号/具体 API 怎么查？ | 本文讲稳定机制；精确 API、安装 whl、版本以**官方文档/仓库为准**（见下方链接）。 |

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 算法地基（IO 感知精确注意力）：[[llm-optimizer/FlashAttention]]
- Decode 阶段并行技巧：[[llm-inference/Flash-Decoding]]
- 推理引擎 / PagedAttention：[[llm-inference/vllm/README]]

**官方与参考资料：**
- 仓库：https://github.com/flashinfer-ai/flashinfer
- 安装 whl 示例：https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/
- 用 FlashInfer 加速自注意力（中文）：https://zhuanlan.zhihu.com/p/681506469
- FlashInfer 中 DeepSeek MLA 内核设计（中文）：https://zhuanlan.zhihu.com/p/25920092499
