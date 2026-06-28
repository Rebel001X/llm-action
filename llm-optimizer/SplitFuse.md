# SplitFuse / Chunked Prefill
> 把"很长的 prefill"切成小块，与正在进行的 decode 混合进同一批前向，既不让长 prompt 阻塞短请求的吐字，又把算力填满。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/PD分离]] [[llm-inference/KV-Cache优化]]

## 阅读地图

| 你想知道 | 看哪节 |
|---|---|
| prefill / decode 到底是什么 | §1 |
| 为什么长 prefill 会"卡住"别人吐字 | §2 |
| 一次前向是算力受限还是带宽受限 | §3 |
| chunked prefill 怎么切、怎么拼 | §4 |
| SplitFuse / DeepSpeed-FastGen 的做法 | §5 |
| chunk 大小怎么选（权衡） | §6 |
| 配合什么调度策略 | §7 |
| 一步步手算 TTFT / TPOT / 算力利用率 | §8 |
| 和 PD 分离、连续批处理的关系 | §9 |
| 对照表 / 常见问题 | 末尾 |

## 0. 一句话锚点

> **长 prompt 的 prefill 是"一大坨重计算"，会霸占一整次前向，让本该每几十毫秒吐一个字的 decode 请求被迫排队等待。Chunked Prefill（SplitFuse 是其代表实现）把这坨重计算切成固定 token 预算的小块，每次前向只塞一块 prefill + 一堆 decode，于是吐字延迟（TPOT）稳定、算力（GPU）又被填满，吞吐与延迟同时改善。**

记住三个缩写，后面反复用：

- **TTFT**（Time To First Token）：从请求到来，到吐出第 1 个 token 的时间。≈ prefill 时间。
- **TPOT**（Time Per Output Token）：decode 阶段每吐一个 token 的平均时间。也叫 ITL（Inter-Token Latency）。
- **吞吐**（Throughput）：单位时间整个系统处理的 token 数（prefill + decode 都算）。

## 1. 地基：prefill 与 decode 是两种完全不同的负载

自回归 LLM 推理分两个阶段。设 prompt 长 $p$ 个 token，要生成 $g$ 个 token。

**Prefill（预填充 / prompt 处理）**：一次性把 $p$ 个 token 全喂进去，并行计算它们的注意力，写满 KV-Cache，最后产出第 1 个输出 token。这是一次"$p$ 个 token 同时进 GPU"的大前向。

**Decode（解码 / 自回归生成）**：每步只喂 **1 个** 新 token（上一步刚生成的），读取之前缓存的全部 KV，算出下一个 token。一个请求生成 $g$ 个 token 就要做 $g$ 次这样的"1-token 前向"。

```
请求生命周期（单条）：
  到达
   │
   ▼
 ┌──────────────────── Prefill ────────────────────┐
 │ 一次性处理 p 个 token, 写 KV-Cache, 出第1个token │  ← 决定 TTFT
 └──────────────────────────────────────────────────┘
   │
   ▼
 ┌──── Decode step1 ──┐┌── step2 ──┐┌── step3 ──┐ ...  共 g 步
 │ 喂1 token,出1 token││ 喂1,出1   ││ 喂1,出1   │      ← 每步决定 TPOT
 └────────────────────┘└───────────┘└───────────┘
```

**关键差别**：prefill 一次喂 $p$ 个 token（$p$ 可能上千），decode 一次喂 1 个 token。它们的"token 量级"差几个数量级，这正是后面所有问题的根源。

## 2. 痛点：为什么长 prefill 会阻塞 decode（队头阻塞）

在一个共享 GPU 的服务里，调度器每一轮（iteration）组一个 batch 做前向。如果某轮里塞进了一个长 prompt 的整段 prefill，这一轮前向会很久。**这一轮里所有其他请求的 decode 都只能等它算完**，因为 GPU 一次只跑一个前向 kernel 序列。

```
朴素方案（prefill 不切块）—— decode 被一次长 prefill 卡住：

时间轴 ──────────────────────────────────────────────►
        ┌───────────────────────┐
batch A │  长 prefill (2048 tok) │              一整次大前向 ≈ 80 ms
        └───────────────────────┘
        ▲ 这 80ms 内, 已经在生成的请求 B/C/D 一个字都吐不出来
        │   它们本来每 ~20ms 该吐一个 token, 现在被迫停 80ms
        ▼
        => B/C/D 的 TPOT 出现 80ms 的"毛刺"(jitter), 用户看到卡顿
```

这叫 **generation stall / head-of-line blocking（队头阻塞）**：长任务的一次重前向，让所有轻量 decode 任务的吐字节奏被打断。表现为 TPOT 的 p99 尾延迟暴涨——平均看着还行，但总有用户卡住。

反过来，如果为了让 decode 流畅，**只做 decode、把 prefill 推后**，又会让长 prompt 的 TTFT 变长，而且 decode batch 通常很"空"（见 §3），算力浪费、吞吐低。**这就是 TTFT、TPOT、吞吐三者的不可能三角的来源。**

## 3. 为什么能"混着算"还更划算：算力受限 vs 带宽受限

一次前向是慢在"算"还是慢在"搬数据"，由 **算术强度（arithmetic intensity）= 计算量 / 访存量** 决定，对比 GPU 的 ops:byte 比值。

- **Prefill 是 compute-bound（算力受限）**：一次进 $p$ 个 token，权重只需从显存读一遍，却能服务 $p$ 个 token 的矩阵乘——计算量大、访存被摊薄，GPU 的 Tensor Core 被喂饱。
- **Decode 是 memory-bound（带宽受限）**：一次只进 1 个 token，但仍要把**整个模型权重**和**全部 KV-Cache**从显存读一遍，才算 1 个 token。算得少、搬得多，Tensor Core 大量空转。

```
单次前向, GPU 利用率示意（▓=在算, ░=在等显存/空转）：

纯 decode (batch 很小):  ▓░░░░░░░░░  算力利用率低, 带宽打满
纯 prefill (长 prompt):  ▓▓▓▓▓▓▓▓▓▓  算力打满, 但独占一轮

混合 (1块prefill+一堆decode):
                        ▓▓▓▓▓▓▓▓░░  prefill 把算力填满,
                                    decode 顺便"搭便车"完成
```

**洞察**：decode 那一轮 GPU 算力本来就闲着（在等显存）。如果这一轮顺手塞一小块 prefill 的计算进去，**几乎不增加这一轮的耗时**（耗时由带宽决定），却白白完成了一部分 prefill 工作。这就是"融合（Fuse）"的收益来源——**用 prefill 的算填满 decode 的算力空窗**。

## 4. 解法：Chunked Prefill —— 把长 prefill 切块再融合

核心两步，名字 **SplitFuse = Split（切）+ Fuse（融）**：

**Split（切块）**：把长度 $p$ 的 prefill 不再一次喂完，而是按一个固定的 **token 预算** $C$（chunk size，例如 512）切成 $\lceil p/C \rceil$ 块，每块只处理 $C$ 个 prompt token。

**Fuse（融合）**：每一轮前向，凑够一个目标 token 预算 $T$：先把所有等待 decode 的请求各贡献 1 个 token 放进去，**剩余预算用一块 prefill 填满**。于是每轮 batch 是"少量 prefill token + 多个 decode token"的混合。

```
chunked + fused 调度（一个长 prefill=2048, 拆成 4 块, C=512）：

轮次:    iter1        iter2        iter3        iter4
       ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
prefill│chunk512│   │chunk512│   │chunk512│   │chunk512│
decode │D D D D │   │D D D D │   │D D D D │   │D D D D │  ← decode 每轮都在前进!
       └────────┘   └────────┘   └────────┘   └────────┘
每轮≈   ~24ms        ~24ms        ~24ms        ~24ms      时延平滑, 无 80ms 毛刺
         ▲ 4 轮后整个 prefill 完成, 期间 decode 从未被饿死
```

切块后，注意力怎么保证正确？**靠 KV-Cache 累积**：处理 chunk $k$ 时，它的 query 要 attend 到"前面所有 chunk 已写入 KV-Cache 的 key/value" + "本 chunk 内部"。因为前几块的 KV 已经留在 cache 里，所以分块计算和一次性计算在数学上等价（causal mask 下结果完全一致）。这也说明 chunked prefill 强依赖 KV-Cache 管理 → 见 [[llm-inference/KV-Cache优化]]。

```
KV-Cache 视角（处理第3块时）：

KV-Cache:  [chunk1 已写][chunk2 已写][chunk3 正在写]  [给后续 decode 留]
            └───── 第3块的 query attend 这一整段 ─────┘
                    (前两块的 KV 直接复用, 不重算)
```

## 5. SplitFuse 与 DeepSpeed-FastGen

**DeepSpeed-FastGen** 是微软 DeepSpeed 团队提出的高吞吐推理系统，其核心调度技术就叫 **Dynamic SplitFuse**。它正是上面 §4 的工程化：

1. **维持每轮 token 预算恒定**（fixed forward token budget $T$）：不管来的请求长短，每轮前向喂进去的总 token 数尽量贴着同一个目标 $T$。这样**每轮前向耗时稳定**，TPOT 抖动小。
2. **长 prompt 被切**：超长 prefill 切成多块，分摊到多轮，避免单轮过载。
3. **短 prompt 被拼**：太短、单独成批会浪费算力的 prefill，被合并、并和 decode 一起凑满预算。
4. **decode 永不被饿死**：因为每轮都先保证在途 decode 各前进一步，再用 prefill 补预算。

> 注意：vLLM、TensorRT-LLM、SGLang 等也都实现了 chunked prefill（开关名/默认值/预算单位各家不同）。本文讲稳定机制，**具体开关名、默认 chunk 大小、API 以各官方文档为准**，不要照搬版本号。

```
Dynamic SplitFuse 一轮的组装（目标预算 T=768 token/轮）：

  待处理:  [decode×600 在途] + [prompt_A=2048] + [prompt_B=300]
                │
                ▼  组装这一轮 batch, 凑满 ~768
  本轮 batch = 600 个 decode token (每个在途请求各1) 
             + 168 个 prefill token (从 prompt_A 切一块)
             ───────────────────────────────────────
             = 768 token  ← 贴住预算, 耗时可预测
  prompt_A 剩 1880 token 留到后续几轮继续切
```

## 6. chunk 大小 / token 预算怎么选（核心权衡）

设每轮预算 $T$（≈ chunk 上限）。这是一个**旋钮**：

| $T$ 偏小 | $T$ 偏大 |
|---|---|
| 每轮前向短 → TPOT 低、抖动小 ✅ | 每轮前向长 → TPOT 高、可能毛刺 ❌ |
| 长 prefill 被切成很多轮 → TTFT 变长 ❌ | prefill 块大 → TTFT 短 ✅ |
| 块太小 → 重回 memory-bound, 算力没喂饱, 吞吐降 ❌ | 块够大 → compute-bound, 吞吐高 ✅ |
| 调度/kernel 启动开销占比上升 ❌ | 开销摊薄 ✅ |

存在一个**最优区间**：$T$ 要**大到足以让前向进入 compute-bound 区**（把 Tensor Core 喂饱），又**小到单轮耗时不破坏 TPOT SLA**。经验上常取数百到一两千 token 量级，**务必按自己模型/硬件实测**。

```
吞吐 vs chunk大小（示意曲线）：

吞吐 ▲           ___________  饱和区(已 compute-bound)
     │         /
     │       /  ← 拐点: 此处刚把算力喂饱
     │     /
     │   /  memory-bound 区(块太小, 浪费算力)
     └──┼──────────────────────► chunk size T
       拐点              选 T 在拐点右侧一点, 兼顾 TPOT
```

## 7. 配合的调度策略（保留：MindIE 等系统的实践）

chunked prefill 解决"单轮怎么组"，**请求级别先服务谁**则由调度策略决定。一些推理引擎（如华为 MindIE）暴露了可选策略：

| 取值 | 策略 | 含义 | 适合 |
|---|---|---|---|
| 0 | **FCFS** 先来先服务 | 按到达顺序 | 通用、最公平，**默认推荐** |
| 4 | **SJF** 短任务优先 | 短输入先做 | 追求极限吞吐；长输入可能等很久 |
| 5 | **LJF** 长任务优先 | 长输入先做 | 特定场景 |
| 6 | **Skip-Join MLFQ** 多级反馈队列 | 分级队列动态调整 | 吞吐与长输入等待取平衡 |
| 7 | **SJF-MLFQ** | SJF + 多级反馈 | 同上，偏吞吐 |

**建议**：一般用 **FCFS**；追求极限吞吐用 **SJF**（但长输入等待时间会大幅增长）；想在吞吐与长输入等待间取平衡，用 **Skip-Join MLFQ** 或 **SJF-MLFQ**。

> 关系：调度策略选"先处理哪个请求"，chunked prefill 决定"被选中的长请求怎样切进每一轮"。二者正交、可叠加。
> 参考：华为 MindIE LLM 文档（以官方为准）。

## 8. 数值示例 / 逐数手算

用一组**假设的**小数字走通全链路（实际数值请按硬件实测）。

**设定**：长 prompt $p=2048$，要生成 $g=200$。同时系统里有 $B=30$ 个在途 decode 请求。
- 单条 decode（1 token）前向耗时 $t_d = 0.8\text{ ms}$（带宽受限，batch 内多个 decode 近似并行，整轮 decode 仍≈这个量级）。
- prefill 处理 1 个 token 的算力成本，折算成"每 512 token 一块前向耗时" $t_{c}=20\text{ ms/块}$。

**(a) 朴素方案：一次性 prefill 整段**

prefill 整段 2048 token 一次前向。设其耗时与 token 数近似线性（compute-bound 区）：

$$t_{\text{prefill}} = 2048/512 \times 20 = 4 \times 20 = 80\text{ ms}$$

这 80 ms 内，30 个在途 decode 请求**全部停摆**。它们本应每 ~0.8 ms 推进一步，现在卡 80 ms，相当于丢了约 $80/0.8 = 100$ 步的吐字节奏 → **TPOT 出现 80 ms 毛刺**。

- 长 prompt 的 **TTFT = 80 ms**（出第 1 个 token）。
- 在途请求 **TPOT 尾延迟 = 80 ms**（被卡那一下）。糟糕的 p99。

**(b) Chunked Prefill：切 4 块，每块 512，与 decode 融合**

每轮 = 1 块 prefill(512) + 30 个 decode。每轮耗时取两者的"主导项"。因为 prefill 块是 compute-bound、decode 是 memory-bound，**两者在一轮里部分重叠**，单轮耗时近似由较大项决定，这里取：

$$t_{\text{iter}} \approx t_{c} + \text{(decode 增量很小)} \approx 20 + 4 = 24\text{ ms/轮}$$

（24 ms 里包含了 30 个 decode 各推进 1 步——它们"搭便车"几乎免费完成。）

- 长 prompt 走完 4 轮：**TTFT = 4 × 24 = 96 ms**（比 (a) 的 80 ms 略长 ↑）。
- 在途请求每轮都吐 1 个 token：**TPOT ≈ 24 ms，无 80 ms 毛刺**（尾延迟从 80 → 24，**↓ 70%**）。

**(c) 吞吐对比（每轮完成多少 token 的有效工作）**

- 方案 (a) 那 80 ms：只完成 prefill 2048 token，decode 0 个。有效 token = 2048，速率 $2048/80 = 25.6$ tok/ms。
- 方案 (b) 那 96 ms（4 轮）：完成 prefill 2048 + decode $4×30 = 120$ token = **2168 token**，速率 $2168/96 ≈ 22.6$ tok/ms。

> 看起来 (b) 的瞬时速率略低，但 (b) **同时**让 30 个用户持续吐字、长 prompt 也几乎同时完成；(a) 则是这 80 ms 里 30 个用户全卡死。真实系统里 chunked 的"算力填满"效应会让长时间窗口的总吞吐反超——**关键收益是把 decode 的算力空窗用 prefill 填满**。

**(d) 验证"块太小会掉吞吐"**：若把块切成 64（$t$ 退化到 memory-bound，设每块 8 ms 但只处理 64 token），处理 2048 需 32 轮 × 8 = 256 ms，远慢于 80 ms——**块太小，算力没喂饱，吞吐崩**。印证了 §6 的拐点结论。

```
三方案 TPOT 时间线对比（每个█ = 在途请求成功吐一个token, ✗=被卡）：

(a) 整段prefill : █ ✗✗✗✗✗✗✗✗✗✗(卡80ms) █ █ █ ...   ← 长毛刺
(b) chunked     : █ █ █ █ █ █ █ █ █ █ █ █ ...        ← 平滑
(d) 块太小      : █····█····█····(整体变慢, 吞吐崩)   ← 算力浪费
```

## 9. 与 PD 分离 / 连续批处理的关系

| 技术 | 解决什么 | 一句话 |
|---|---|---|
| **连续批处理** (continuous/in-flight batching) | batch 内请求长度/进度不齐 | 每轮动态增删请求，不等整批做完 |
| **Chunked Prefill / SplitFuse** | 长 prefill 阻塞 decode | 把 prefill 切块，与 decode 融进同一轮 |
| **PD 分离** (Prefill-Decode 分离) | prefill 与 decode 资源争抢 | 用**不同 GPU/实例**分别跑 prefill 和 decode，物理隔离 |

- **chunked prefill 与 PD 分离是两条不同路线**，且常被对比：
  - **Chunked（融合派）**：prefill、decode 在**同一 GPU 同一轮**混合，靠切块平滑。省机器、实现相对简单，适合中小规模。
  - **PD 分离（隔离派）**：prefill 集群、decode 集群分开，各自调优（prefill 拼大 batch 吃算力，decode 凑大 batch 吃带宽），需要在两者间**传输 KV-Cache**。适合超大规模、对 SLA 极致要求。详见 [[llm-inference/PD分离]]。
- 三者并非互斥：连续批处理是底座，其上可叠加 chunked prefill；超大规模再上 PD 分离。
- 三者都重度依赖 KV-Cache 的高效管理（PagedAttention 等）→ [[llm-inference/KV-Cache优化]]。

```
路线选择（粗略）：
  单/少卡, 通用服务      → 连续批处理 + Chunked Prefill (SplitFuse)
  大规模, 严格 TTFT/TPOT → PD 分离 (prefill池 + decode池, 传 KV)
```

## 对照 / 复杂度表

| 维度 | 整段 prefill（朴素） | Chunked Prefill / SplitFuse |
|---|---|---|
| 单轮前向耗时 | 随 prompt 长度暴涨（不可控） | 贴住固定预算 $T$（可控、稳定） |
| decode TPOT 抖动 | 大（长 prompt 来时毛刺） | 小（每轮都推进） |
| TTFT（长 prompt） | 较短 | 略增（切多轮） |
| GPU 算力利用率 | decode 轮极低、prefill 轮极高 | 持续接近饱和（融合填空窗） |
| 吞吐（长窗口） | 受队头阻塞拖累 | 高（算力被填满） |
| 实现复杂度 | 低 | 中（需切块+预算调度+KV累积） |
| KV-Cache 依赖 | 一般 | 强（分块 attend 历史 KV） |

## 常见问题

| 疑问 | 真相 |
|---|---|
| 切块会改变模型输出吗？ | 不会。causal mask 下，分块计算与一次性计算**数学等价**，靠 KV-Cache 累积保证 query 能 attend 到全部历史。 |
| chunked 一定提升 TTFT 吗？ | 不一定。长 prompt 的 TTFT 可能**略增**（被切成多轮）；它真正改善的是 **TPOT 抖动**和**整体吞吐**。 |
| chunk 越小越好？ | 否。太小会退回 **memory-bound**，算力喂不饱、调度开销占比升，吞吐反降（§6 拐点、§8(d)）。 |
| 它和连续批处理是一回事吗？ | 不是。连续批处理管"请求动态增删"，chunked prefill 管"长 prefill 怎么切进每一轮"，二者叠加。 |
| 有了 PD 分离还需要 chunked 吗？ | 看规模。小规模用 chunked 更省；超大规模用 PD 分离物理隔离。它们是**互斥的两条主路线**，也可在各自集群内部再用别的技巧。 |
| 为什么 decode "搭便车"几乎免费？ | 因为 decode 那一轮 GPU 在等显存（算力空窗），顺手塞 prefill 的计算几乎不增加该轮耗时（耗时由带宽主导）。 |
| chunk 大小要不要动态调？ | 最好动态。Dynamic SplitFuse 就是按"恒定 token 预算"动态决定每轮切多少 prefill，使每轮耗时稳定。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引
- [[llm-inference/PD分离]] — 与 chunked 并列的另一条主路线（物理隔离 prefill/decode）
- [[llm-inference/KV-Cache优化]] — chunked prefill 正确性与效率的底层依赖

> 参考：DeepSpeed-FastGen（Dynamic SplitFuse）、华为 MindIE LLM 调度文档。具体开关名/默认值/参数以各官方文档为准。
