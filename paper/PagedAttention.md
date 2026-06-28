# PagedAttention 论文精读（vLLM 的内核）

> 一句话定位：把操作系统「虚拟内存 + 分页」的思想搬到 KV Cache 上，消灭显存碎片、按需分配、支持共享，从而把 LLM 在线服务的吞吐拉高 2–4 倍。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-optimizer/FlashAttention]]

> 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon et al., SOSP 2023）；系统实现即 **vLLM**。本笔记数字均标「约/见原文」。

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | 分页、KV Cache |
| 1 | 为什么 KV Cache 是瓶颈 | 自回归、显存账、碎片 |
| 2 | 旧系统错在哪（连续分配） | 内部/外部碎片、预留浪费 |
| 3 | 核心隐喻：OS 虚拟内存 → KV | 逻辑块/物理块/块表 |
| 4 | PagedAttention 算法 | 非连续、分块 attention |
| 5 | 块管理：分配/回收/写时复制 | CoW、引用计数 |
| 6 | 调度与抢占（preemption） | swap、recompute |
| 7 | 分布式：张量并行下的共享管理器 | SPMD、单一管理器 |
| 8 | 应用：共享前缀、Beam、并行采样 | prefix sharing |
| * | 关键公式/算法/数值手算 | 块数、碎片率、吞吐 |
| 末 | 评价 / 局限 / 工程启示 | 对照表 |

---

## 0. 一句话锚点

LLM 推理时，每个请求都要保存「已经算过的 token 的 Key/Value 向量」= **KV Cache**。它**很大、长度事先不知道、还在不停增长**。旧系统给每个请求预留一整段**连续显存**，结果大量显存被「预留但没用」或「用完留下小空洞」浪费掉。

**PagedAttention** 把 KV Cache 切成固定大小的**块（block）**，像操作系统管内存页那样**按需、非连续**地分配；attention 计算被改写成**能在不连续的块上工作**。配套的 vLLM 用一张**块表（block table）**把「逻辑顺序」映射到「物理块」，于是：碎片几乎为零、显存利用率逼近 100%、还能让多个请求**共享**相同的 KV 块。

---

## 1. 地基：为什么 KV Cache 是 LLM 服务的命门

### 1.1 自回归解码的两个阶段

```
Prompt:  "中国 的 首都 是"          ← Prefill（一次并行算完所有 prompt token）
            │  生成 K,V 存入 cache
            ▼
Decode:  "北"  → "京"  → "。" → <eos>  ← 每步只算 1 个新 token，但要"看"前面所有 K,V
```

- **Prefill（预填充）**：把整段 prompt 一次性喂进去，并行计算，算力（compute）密集。
- **Decode（解码）**：一次只生成一个 token，但每生成一步都要读取**之前所有 token 的 K、V**。这一步是**访存（memory）密集**——算得少、读得多。

为了不在每步都把前面 token 重算一遍，就把它们的 Key/Value **缓存**下来 → 这就是 KV Cache。它把 decode 从 $O(n^2)$ 重算降到每步 $O(n)$ 读取，但代价是**吃显存**。

### 1.2 KV Cache 到底多大？（显存账，必算）

每个 token 在每一层都要存一份 K 和一份 V，大小：

$$\text{每 token 字节} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times \text{dtype}$$

以 **13B 类模型**为例（约值，见原文设定）：$n_{layers}=40,\ n_{heads}=40,\ d_{head}=128$，FP16（2 字节）：

$$2 \times 40 \times 40 \times 128 \times 2 = 1,\!638,\!400 \text{ 字节} \approx 1.6\ \text{KB/token}$$

- 一条 **2048 token** 的序列：$1.6\text{KB} \times 2048 \approx 3.2\ \text{MB}$。
- 一张 **40GB** 显卡装完权重（约 26GB）后剩约 12GB 给 KV，约能放 $12\text{GB}/3.2\text{MB} \approx \textbf{3700}$ 这种满长序列的**槽位预算**——但实际请求长度参差不齐，**怎么分配**直接决定能并发多少请求。

> 核心张力：**显存是吞吐天花板**。能同时塞进显存的请求越多（batch 越大），GPU 利用率越高、吞吐越大。所以「省 KV 显存」≈「提吞吐」。

---

## 2. 旧系统错在哪：连续分配的三宗罪

在 PagedAttention 之前，主流框架（如早期 FasterTransformer / Orca 思路）给每个请求**预先分配一整段连续显存**，按「该请求可能的最大长度」预留。问题：

```
请求 A (max_len=2048)  ┌──────────────────────────────────────┐
实际只生成了 350 token │██████ 已用350 │░░░░░░░░ 预留但永远用不到 ░░░░░░░│
                       └──────────────────────────────────────┘
                         内部碎片(预留浪费)         ↑ 这部分白占
```

| 浪费类型 | 含义 | 旧系统表现 |
|----------|------|-----------|
| **预留浪费（reserved）** | 按 max_len 占着，但当前还没生成到 | 整段提前锁死 |
| **内部碎片（internal）** | 请求结束时实际长度 < 预留长度 | 末尾大段空着 |
| **外部碎片（external）** | 不同请求段之间留下大小不一的小空洞，凑不出一段连续大块 | 明明有空显存却分不出来 |

论文测得：旧系统里**真正存有效 KV 数据的显存常常只有 20%–40%**（约，见原文 Fig.2），其余被三种碎片吃掉。直接后果：**batch 上不去 → GPU 闲着 → 吞吐低**。

> 类比：连续分配像「给每位客人按他可能吃的最大份提前端一整桌菜」，多数桌子只动了几口，餐厅却已经满座。

---

## 3. 核心隐喻：操作系统的虚拟内存与分页

PagedAttention 的灵魂是一句话：**KV Cache 不必连续。**

操作系统早就解决过「进程要连续地址，物理内存却碎片化」这个矛盾——用**分页**：把内存切成固定大小的**页（page）**，进程看到的是连续的**虚拟地址**，底层是一张**页表**映射到任意散落的**物理页**。

vLLM 把这套原封不动搬过来：

```
          逻辑视图(请求看到的)              物理视图(显存真实布局)
          Logical KV Blocks                Physical KV Blocks (GPU显存)
        ┌──────┬──────┬──────┐            ┌──────┐ ┌──────┐ ┌──────┐
请求 →  │ blk0 │ blk1 │ blk2 │            │ Pblk7│ │ Pblk2│ │ Pblk9│ ...散落各处
        └──┬───┴──┬───┴──┬───┘            └──────┘ └──────┘ └──────┘
           │      │      │       Block Table(块表):
           └──────┼──────┼──────►  logical 0 → physical 7
                  └──────┼──────►  logical 1 → physical 2
                         └──────►  logical 2 → physical 9
```

| OS 概念 | vLLM 对应 | 说明 |
|---------|-----------|------|
| 进程 | 一个请求/序列 | |
| 页（page） | **KV block** | 固定容纳 $B$ 个 token 的 K/V（典型 $B=16$，见原文） |
| 虚拟地址 | 逻辑块号 | 请求眼里连续：blk0,blk1,... |
| 物理页 | **物理块** | 显存里任意位置，可不连续 |
| 页表 | **块表（block table）** | 逻辑块 → 物理块 + 「该块填了几个 token」 |

**关键好处**：分配变成「按块按需领」，只在**最后一个块**可能有不满的空位 → 碎片**最多就是一个块（≤ B-1 个 token）**，浪费率从「几十%」降到「<4%」（约，见原文）。

---

## 4. PagedAttention 算法：在非连续块上算 attention

普通 attention（一次 decode 步，query 是当前新 token $q_i$）：

$$
a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i}\exp(q_i^\top k_t/\sqrt{d})},\qquad
o_i = \sum_{j=1}^{i} a_{ij}\, v_j
$$

它要求 $k_1..k_i,\ v_1..v_i$ 在显存里连续好取。PagedAttention 的改造：**把求和按块切开**，每次从一个物理块里取出 $B$ 个 K/V 来算局部分数，再跨块汇总：

```
PagedAttention(单个 query q_i):
  for 每个逻辑块 b in 该请求块表:
      P_b = block_table[b]                 # 查物理块号
      K_b, V_b = 物理块 P_b 里的 B 个 key/value
      S_b = q_i · K_b^T / sqrt(d)          # 块内分数 (1×B)
      # 累积 softmax 分母 与 加权 V (online-softmax 风格, 与 FlashAttention 同源)
  o_i = 归一化后 跨块求和(S_b 加权 V_b)
```

要点：
1. **物理块不连续没关系**——kernel 通过块表逐块寻址。
2. **块是 attention 的最小粒度**，块内仍然连续（GPU 访存友好），块间靠块表跳转。
3. 与 **FlashAttention** 的在线 softmax / 分块累积思想**互补**：FlashAttention 解决「单个 attention 内部不落地大矩阵」，PagedAttention 解决「KV 在显存里怎么摆」。二者可叠加（见 [[llm-optimizer/FlashAttention]]）。

> 直觉：query 不再要求「面前摆一长条连续 KV」，而是「我有一张地图（块表），需要哪块就去哪块取」。

---

## 5. 块的生命周期：分配、回收、写时复制（CoW）

### 5.1 按需增长

请求每生成满 $B$ 个 token，就向**块管理器（Block Manager）** 申请一个新物理块，写进块表。只有**当前正在写的最后一个块**可能不满。

```
生成中:  blk0[满16] blk1[满16] blk2[用了3/16]   ← 只有 blk2 有空位
请求结束:  整条序列的物理块一次性归还到 free 池(空闲块链表)
```

### 5.2 引用计数 + 写时复制（Copy-on-Write）

多个序列可以**指向同一个物理块**（典型场景：同一 prompt 的并行采样 / beam search 共享前缀）。物理块带**引用计数 ref_count**：

```
共享只读阶段:        写入触发 CoW:
seq1 ─┐              seq1 想写 blk(ref=2)
      ├─► Pblk5(ref=2)  → 复制一份新块 Pblk8 给 seq1 独占(ref=1)
seq2 ─┘                  原 Pblk5 ref 减为 1 归 seq2
```

- **读共享**：多个序列共用同一物理块，ref_count > 1。
- **写分裂**：谁要修改一个共享块，就先**复制出新块**再改（Copy-on-Write），避免互相污染。
- **回收**：ref_count 归 0，块回到 free 池。

这正是「共享前缀只存一份」省显存的机制（见第 8 节）。

---

## 6. 调度与抢占：显存不够时怎么办

vLLM 用**连续批处理（continuous batching）**：每个 decode step 都能让已完成的请求离场、新请求入场，而不是等整批做完（见 [[llm-inference/连续批处理?]]）。当**新请求要块、但 free 池空了**时，需要**抢占（preemption）** 已有请求，腾出块：

| 策略 | 做法 | 代价 / 何时用 |
|------|------|--------------|
| **Swapping（换出）** | 把被抢占请求的 KV 块拷到 **CPU 内存**，需要时再拷回 | 占用 PCIe 带宽；块大、序列长时合算 |
| **Recomputation（重算）** | 直接丢弃其 KV，恢复时把该序列**重新 prefill** 一遍 | 不占 CPU 内存；prefill 是并行的，常常更快 |

抢占以**整条序列的全部块**为单位（all-or-nothing），因为一条序列的 KV 必须同进同出才能继续解码。论文给出两策略在不同长度下各有胜负（约，见原文）。

```
显存满 → 选一个"牺牲"序列 → swap/recompute 释放它的块 → 新请求获得块继续跑
                                   ↑ 释放的块进 free 池
```

---

## 7. 分布式：张量并行下的「单一 KV 管理器」

> 本节对应原文件已有内容：多 GPU（Megatron-LM 风格张量并行）时如何共享 KV 管理。

许多模型一张卡放不下，需**张量模型并行**（见 [[B07:llm-inference/大模型推理张量并行]]）。vLLM 支持 Megatron-LM 风格、**SPMD（单程序多数据）** 执行：

- 线性层切块做分块矩阵乘，GPU 间用 **all-reduce** 同步中间结果（见 [[ai-infra/网络/集合通信原语]]）。
- **注意力按 head 维切**：每个 SPMD 进程只负责多头注意力里的**一部分 head**。

关键观察 → **块表只需一份**：

```
            ┌────────────── Scheduler ──────────────┐
            │   单一 KV Cache Manager (逻辑→物理块表) │
            └───────┬───────────────┬────────────────┘
        广播控制消息 │  (token id + block table) │ 广播
            ┌───────▼──────┐  ┌──────▼───────┐
            │  GPU worker0 │  │  GPU worker1 │  ... 各存自己那部分 head 的 KV
            │ 同一物理块ID  │  │ 同一物理块ID  │
            └──────────────┘  └──────────────┘
   解码中 worker 间只用 all-reduce 同步, 不为内存管理而同步; 末尾把采样 token 回传调度器
```

为什么能共享一份块表？**每个分片处理的是同一组输入 token，因此需要相同位置的 KV**；各 worker 用相同的物理块 ID，但**每个 worker 只存自己那几个 head 的 KV 片段**。于是：worker **无需为内存管理互相同步**，只需在每步开头收到调度器广播的（token id + 块表）即可。这把内存管理从分布式难题降回成**集中式、单点**问题。

---

## 8. 应用：共享让显存「一份当多份用」

| 场景 | 共享什么 | 收益 |
|------|----------|------|
| **并行采样**（一个 prompt 生成多个候选） | 整段 prompt 的 KV 块只存一份 | prompt 越长省得越多 |
| **Beam Search** | 多束之间公共前缀的块共享，分叉处 CoW | 大幅省显存、少拷贝 |
| **共享系统 prompt / few-shot 前缀** | 所有请求复用同一前缀块（prefix caching） | 高并发同模板时省巨量显存（见 [[llm-inference/KV-Cache优化]]） |

```
并行采样: 1 个 prompt → 4 个输出
旧法: prompt KV ×4 份         PagedAttention: prompt KV ×1 份(ref=4) + 各自分叉块
                              省下约 (4-1)/4 = 75% 的 prompt KV
```

---

## 关键公式 / 算法 / 数值手算

### A. 一条序列需要几个块

$$\text{块数} = \left\lceil \frac{\text{序列长度 } L}{\text{块大小 } B} \right\rceil$$

例：$L=350,\ B=16 \Rightarrow \lceil 350/16\rceil = 22$ 块。最后一块只用了 $350 - 21\times16 = 14$ 个槽，浪费 2 个。

### B. 碎片率（PagedAttention 上界）

每条序列**最多浪费一个块**（末块不满）：

$$\text{浪费率} \le \frac{B-1}{L}\quad\xrightarrow{B=16,\,L=350}\quad \frac{15}{350}\approx 4.3\%$$

对比旧连续分配：若按 max_len=2048 预留、实际只用 350，浪费率 $= 1 - 350/2048 \approx \textbf{83\%}$。**4% vs 83%** 就是吞吐差距的来源。

### C. 能并发多少请求（粗算）

设可用 KV 显存 $M=12$GB，每 token 约 $1.6$KB（第 1.2 节），平均序列 $L=512$：

$$\text{每请求约占} = 512 \times 1.6\text{KB} \approx 0.82\text{MB}$$
$$\text{并发上限} \approx \frac{12\text{GB}}{0.82\text{MB}} \approx \textbf{14600 token 等价槽} / \dots$$

实际同时活跃请求数 ≈ $M / (\bar{L}\times \text{每token})$。**碎片越少 → 等式右边分母越接近真实占用 → 并发越大 → 吞吐越高。** 论文报告端到端吞吐相对旧系统提升 **约 2–4×**（见原文 Fig.12 等）。

---

## 评价 / 对照 / 局限

| 维度 | PagedAttention / vLLM | 旧连续分配 |
|------|----------------------|-----------|
| 显存利用率 | 接近 100%（碎片 < 4%，约） | 常仅 20–40%（约） |
| 长度未知 | 按块按需增长，天然支持 | 必须按 max_len 预留 |
| 共享/前缀复用 | 块级共享 + CoW，原生支持 | 难，需整段复制 |
| attention kernel | 需定制（分块寻址） | 标准连续 kernel |
| 寻址开销 | 多一次块表查找/间接寻址 | 无 |

**局限 / 注意**：
- **块表与寻址带来少量开销**，块太小（B 过小）会放大；B 太大又回到碎片问题——需权衡（典型 B=16，见原文/以官方为准）。
- 需要**定制 CUDA kernel**，移植到新硬件/新 attention 变体有工程成本。
- 抢占（swap/recompute）在极端过载时仍有抖动；调度策略是后续工作的活跃方向。
- 思想已成行业事实标准：后续 **prefix caching、PD 分离**（见 [[llm-inference/PD分离]]）、分布式 KV 池等都建立在「KV 可分块、可寻址、可共享」之上。

**对工程的启示**：
1. **「连续」往往是个伪需求**——加一层间接映射（块表），就能把碎片问题降维。
2. **把 OS 经典方案迁移到 ML 系统**（分页、CoW、引用计数、swap）极有威力。
3. 在线服务优化的真正杠杆常在**内存/调度**而非纯算子；省显存 = 增 batch = 提吞吐。

---

## 🔗 跳转链接

- 知识总览：[[00-知识地图]]
- 系统实现：[[llm-inference/vllm/README]]
- KV 相关：[[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 算子互补：[[llm-optimizer/FlashAttention]]
- 批处理与分离：[[llm-inference/连续批处理?]] · [[llm-inference/PD分离]]
- 并行与通信：[[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]]
- 指标对齐：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
