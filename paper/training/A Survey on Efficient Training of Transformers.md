# A Survey on Efficient Training of Transformers（高效 Transformer 训练综述精读）

> 一句话定位：这是一篇"训练侧效率"的全景地图，把"如何更快、更省显存地训练 Transformer"拆成**计算优化 / 内存优化 / 硬件-算法协同设计**三大主线，逐一盘点代表性技术。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-optimizer/FlashAttention]] · [[llm-train/README]] · [[Reducing Activation Recomputation in Large Transformer Models]] · [[GaLore]]

## 阅读地图（先看这张表，再读正文）

| 维度 | 这篇综述讲了什么 | 一句话本质 | 工程对应物 |
|---|---|---|---|
| 计算优化 | 优化器/初始化、稀疏训练、过参数化、大批量训练、token dropping | 在**数学/算法层**减少需要做的乘加 | LAMB、Lottery Ticket、MoE |
| 数据效率 | Token masking、重要性采样、数据选择 | 减少"喂进去的样本数/token 数" | 课程学习、去重、coreset |
| 内存优化 | 激活重计算、混合精度、ZeRO/卸载、参数高效微调 | 在**系统层**省显存换吞吐 | Activation Checkpoint、ZeRO、LoRA |
| 硬件-算法协同 | 低精度算子、高效注意力专用硬件、FlashAttention | 让**算法形态贴合硬件**的内存层级 | Sanger、ELSA、FlashAttention |
| 评价口径 | 训练吞吐、峰值显存、收敛步数、能效 | "三角权衡"：算力 / 显存 / 精度 | TFLOPs、MFU、tokens/s |

> 提示：这是一篇 **survey（综述）**，它的价值不在某个单一公式，而在**给出一套分类法（taxonomy）**，让你看到任何一个"加速训练"的工作，能立刻定位它在棋盘上的哪一格。

---

## 0. 一句话锚点

**训练一个大 Transformer，瓶颈从来不是"算不出来"，而是"算得太慢、显存装不下、卡间通信太贵"。** 这篇综述把所有缓解手段归到一个统一框架里：**要么减少计算量（compute），要么减少内存占用（memory），要么让算法适配硬件（co-design）**——本质都是在 **算力 × 显存 × 精度** 这个三角里做权衡。

---

## 1. 地基 / 前置：为什么 Transformer 训练这么贵？

### 1.1 三笔账（必须心里有数）

训练一步（forward + backward）的开销由三部分构成：

```
            ┌────────────── 一次训练 step 的成本 ──────────────┐
            │                                                  │
   ┌────────▼────────┐   ┌────────▼────────┐   ┌──────────────▼──────────────┐
   │   计算 FLOPs     │   │   显存 Memory    │   │   通信 Communication          │
   │ 矩阵乘 + Attn    │   │ 参数+梯度+优化器  │   │ 跨卡 AllReduce / AllGather   │
   │  ~6·N·D (训练)   │   │ +激活值(activations)│   │  与并行策略强相关             │
   └─────────────────┘   └─────────────────┘   └─────────────────────────────┘
```

- **计算量**：自回归 Transformer 训练一个 token 的 FLOPs 约为 $6N$（$N$=参数量），其中 forward 约 $2N$、backward 约 $4N$。注意力部分还有一个随序列长度 $L$ **平方增长**的项 $O(L^2 d)$。
- **显存**：由 4 块构成——**参数 + 梯度 + 优化器状态 + 激活值**。其中 Adam 的优化器状态（一阶动量 $m$、二阶动量 $v$）就占 $2\times$ 参数大小；激活值随 batch、序列长度、层数线性增长，常常是显存第一杀手。
- **通信**：当模型/数据被切到多卡，每步要同步梯度（数据并行）或交换激活（张量/流水并行），通信量与并行策略耦合。

### 1.2 注意力的"平方诅咒"

标准自注意力：

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

- $Q,K,V \in \mathbb{R}^{L\times d}$，$QK^\top$ 是 $L\times L$ 矩阵。
- **计算**是 $O(L^2 d)$，**显存**是 $O(L^2)$（要把整张注意力矩阵物化到 HBM）。
- 序列从 2K 拉到 32K，注意力开销涨 **256 倍**。这就是为什么"高效注意力"是综述里浓墨重彩的一格。

### 1.3 综述的总框架（一张图记住全篇）

```
                 高效训练 Transformer
                         │
     ┌───────────────────┼───────────────────────┐
     ▼                   ▼                        ▼
  ① 计算优化           ② 内存优化              ③ 硬件-算法协同
  (Computation)       (Memory)               (Co-Design)
  ├ 优化器/初始化       ├ 激活重计算            ├ 硬件感知低精度
  ├ 稀疏训练           ├ 混合精度训练           ├ 高效注意力专用硬件
  ├ 过参数化           ├ 显存高效优化器/卸载     └ FlashAttention(IO-aware)
  ├ 大批量训练         ├ 参数高效微调(PEFT)
  └ 数据选择/Token掩码  └ ZeRO/激活切分
```

---

## 2. 计算优化（Computation Efficiency）

目标：**让"该做的乘加"变少，或让"每一步走得更快地收敛"**。

### 2.1 优化器与初始化：更快收敛 = 更少 step

- **更好的优化器**：相比 SGD，Adam/AdamW 收敛更快；大批量场景用 **LARS / LAMB**（逐层自适应学习率），让 batch 放大到上万时仍稳定。
- **更好的初始化**：好的初始化能减少 warmup、避免梯度爆炸/消失。代表如 **Fixup / T-Fixup / Admin**，让深层 Transformer 不依赖 LayerNorm 也能稳定起步。
- 本质：**收敛步数 ↓ ⇒ 总 FLOPs ↓**，这是"算法层省算力"，免费午餐。

```
   收敛曲线对比(示意)
 loss│ \
     │  \____ SGD: 需要很多 step
     │   \
     │    \__ Adam/LAMB: 同样精度用更少 step
     └──────────────────────► step
```

### 2.2 稀疏训练（Sparse Training）

核心思想：**"彩票假说"（Lottery Ticket Hypothesis）——一个稠密大网络里藏着一个稀疏子网络，单独训练它就能达到原网络精度。**

- 训练全程只更新/保留一部分权重（动态稀疏，如 RigL：周期性地剪掉小权重、长回大梯度位置的连接）。
- 收益：稀疏矩阵乘理论上能省算力。**坑**：非结构化稀疏在 GPU 上难加速（访存不规整），实际加速常打折，需结构化稀疏（block-sparse）或专用硬件配合。

### 2.3 过参数化（Over-parameterization）与 MoE

- **反直觉的事实**：训练时**更宽更大**的模型，反而往往"用更少的样本/步数"达到目标精度（优化地形更光滑）。
- 工程化身：**Mixture-of-Experts（MoE）**——参数总量巨大，但每个 token 只激活少数专家（top-k），**参数多但单步 FLOPs 不变**。这正是"过参数化省算力"的现实落地。详见 [[llm-algo/moe/README]]。

```
    Dense FFN              MoE FFN (top-2)
   ┌────────┐            ┌──┬──┬──┬──┐  ← N 个专家
   │  全量   │            │E1│E2│E3│E4│
   │  计算   │   token →  └▲─┴▲─┴──┴──┘  路由只选 2 个
   └────────┘             激活 E1,E2     算力≈2/N
```

### 2.4 大批量训练（Large-batch Training）

- 把 batch 放大 → 单 epoch 内 step 数变少 → 更易用满多卡并行。
- 风险：大 batch 泛化变差（"sharp minima"）、需配套 **学习率线性放大 + warmup + LARS/LAMB**。
- 本质：用**并行度换 wall-clock 时间**，FLOPs 总量不变但墙钟时间显著下降。

### 2.5 数据选择 / Token 掩码：少喂数据也能学好

这是"**数据效率**"维度，和算力维度正交但同样省成本。

- **重要性采样（Importance Sampling）**：优先采"信息量大/损失大"的样本，跳过已学会的简单样本 ⇒ 同精度用更少样本。
- **Token masking / dropping**：训练中**丢掉一部分 token 不参与计算**（如对 MLM 只算被 mask 的位置；或对长序列动态丢弃低信息 token）⇒ 直接减少每步 FLOPs。
- **数据去重 / coreset / 课程学习（curriculum）**：从易到难安排数据，提升收敛效率。

> 一句话：**计算优化 = "每步更快收敛" + "每步少算" + "少喂数据"** 三条腿。

---

## 3. 内存优化（Memory Efficiency）

目标：**在不改变数学结果的前提下，把显存峰值压下来**，从而能训更大模型 / 更大 batch。

### 3.1 激活重计算（Activation Recomputation / Checkpointing）

- **问题**：反向传播需要前向的激活值；存全部激活 → 显存 $O(\text{层数} \times \text{batch} \times L \times d)$ 爆炸。
- **做法**：前向只**保存少量 checkpoint**（如每层入口），反向时**重新前向计算**中间激活。
- **账**：用 $\sqrt{n}$ 策略，$n$ 层显存从 $O(n)$ 降到 $O(\sqrt{n})$，**代价是多约 1/3 的前向计算**（约 +33% 算力换大幅省显存）。
- 深入见姊妹论文 [[Reducing Activation Recomputation in Large Transformer Models]]（选择性重计算：只重算最便宜/最占显存的部分，把代价从 ~33% 压到个位数百分比）。

```
   普通:  存所有激活            重计算: 只存 checkpoint
   [A0][A1][A2]...[An]  显存↑↑   [A0]......[Ak]......  显存↓
                                 反向时区间内 re-forward → 算力↑
```

### 3.2 混合精度训练（Mixed Precision）

- 参数主副本用 FP32（master weights），前向/反向用 **FP16/BF16** 算 ⇒ 算子吞吐翻倍、激活显存减半。
- **FP16 的坑**：动态范围小，小梯度下溢为 0 ⇒ 需 **loss scaling**（把 loss 乘一个大系数，反向后再除回）。**BF16** 指数位与 FP32 相同、不易溢出，已成训练默认。
- 更激进的 **FP8 训练**（H100/Transformer Engine）进一步省一半，需逐 tensor 缩放。详见 [[llm-compression/quantization/fp8]]。

### 3.3 显存高效优化器与卸载（ZeRO / Offload）

- **痛点**：Adam 的优化器状态 = $2\times$ 参数（FP32 的 $m,v$）+ FP32 master weights，单是优化器相关就 $\sim 12$ 字节/参数。
- **ZeRO**：把**优化器状态 / 梯度 / 参数**沿数据并行维**切片**到各卡，每卡只存 $1/N$，用时再 AllGather。三个 stage 逐级省显存。
- **Offload**：把暂时不用的状态/参数下放到 **CPU 内存甚至 NVMe**，用 PCIe 带宽换显存容量（ZeRO-Offload / ZeRO-Infinity）。
- 详见 [[llm-train/pytorch/distribution/README]]。

```
   普通DP: 每卡都存全套 [P][G][Opt]   显存=单卡满
   ZeRO:   卡0 [P/3][G/3][Opt/3]
           卡1 [P/3][G/3][Opt/3]      显存≈1/N，按需 AllGather 补齐
           卡2 [P/3][G/3][Opt/3]
```

### 3.4 参数高效微调（PEFT）

- 微调时**冻结绝大部分参数，只训练极小一部分**：**LoRA**（注入低秩矩阵 $\Delta W = BA$）、Adapter、Prefix/Prompt Tuning、BitFit。
- 收益：可训练参数从 100% 降到 <1%，优化器状态、梯度显存随之骤降，单卡即可微调大模型。
- 注意：PEFT 主要服务"微调"，不解决从零预训练的算力问题。

---

## 4. 硬件 / 算法协同设计（Hardware-Algorithm Co-Design）

目标：**让算法的"计算形态"贴合硬件的"内存层级与算力特征"**——这是综述最有特色的一节（也是原文末尾详写的部分）。

### 4.1 硬件感知的低精度（Hardware-aware Low Precision）

- 降精度同时减**内存**和**计算**；用硬件友好的**定点/整数**（而非浮点）表示，可用更小的乘法器/加法器/存储块 ⇒ **功耗和速度双赢**。
- 可与剪枝、低秩近似叠加：
  - **Sanger**：用 **4-bit 的 Q、K** 算出稀疏注意力掩码的量化预测，再把稀疏掩码重排成**结构化块**交给可重构硬件处理。
  - **DOTA**：用**低秩变换 + 低精度计算**识别注意力中不重要的连接；结合 **token 级并行 + 乱序执行**，相对 GPU 报告 **约 152.6× 加速**（数字见原文）。

### 4.2 高效注意力的专用硬件（Efficient Attention Accelerators）

只为"相似度高"的 K 计算注意力，过滤无关连接来省算力：

- **A³**：只挑那些**可能与给定 Q 相似度高**的 K，减少注意力计算。
- **ELSA**：用**哈希相似度**过滤掉与某个 Q 不相关的 K；配专用加速器，相对 16GB Nvidia V100 报告 **约 58.1× 加速、能效提升约三个数量级**（数字见原文）。

### 4.3 FlashAttention：把"IO-aware"做到极致（重点）

**这是协同设计里最成功、已成事实标准的一个。** 核心洞察：注意力在 GPU 上**不是 compute-bound，而是 memory(IO)-bound**——瓶颈在 HBM ↔ SRAM 的来回搬运，而不是算力。

```
        GPU 内存层级(越上越快越小)
   ┌─────────────────────────────┐
   │  SRAM  ~20MB, ~19 TB/s       │ ← 片上，极快极小
   ├─────────────────────────────┤
   │  HBM   ~40-80GB, ~1.5-3 TB/s │ ← 显存，慢一个量级
   └─────────────────────────────┘
   标准Attn: 把 L×L 大矩阵写回 HBM 再读回 → IO 爆炸
   Flash :   分块(tiling)在 SRAM 里算完再出来 → 不物化 L×L
```

- **关键技术**：① **Tiling 分块** + **online softmax**（边读边更新 softmax 的最大值与归一化因子，无需先看到整行）；② **不把 $L\times L$ 注意力矩阵写回 HBM**；③ 反向用**重计算**省激活显存。
- **效果**：显存从 $O(L^2)$ 降到 $O(L)$，端到端训练显著加速；**数学结果与标准注意力完全等价**（不是近似！）。
- 详见 [[llm-optimizer/FlashAttention]]。

---

## 关键公式 / 算法 / 数值示例（手算一遍才算懂）

### A. 训练显存的"四块账"

以 7B 参数、Adam、FP16 前向 + FP32 master 为例（不含激活）：

| 项 | 字节/参数 | 7B 合计 |
|---|---|---|
| FP16 参数 | 2 | 14 GB |
| FP16 梯度 | 2 | 14 GB |
| FP32 master weights | 4 | 28 GB |
| Adam $m$ | 4 | 28 GB |
| Adam $v$ | 4 | 28 GB |
| **合计** | **16** | **≈112 GB** |

> 结论：**单是"参数+梯度+优化器"就要 ~16 字节/参数**，7B 模型 ~112GB，单张 80GB 卡都放不下——这正是 ZeRO 切片要解决的。激活值还要另算，是大 batch / 长序列下的额外杀手。

### B. 激活重计算的"算力换显存"

- 设 $n$ 层，每层前向算力 $C$，激活显存 $M$。
- 不重计算：显存 $\approx nM$，算力 $= nC$（仅前向）。
- $\sqrt{n}$ 重计算：显存 $\approx \sqrt{n}\,M$，前向需多算约 $1\times$（反向时区间内 re-forward）⇒ **总前向算力约 $+33\%$**，显存从 $O(n)$ 降到 $O(\sqrt{n})$。

> 数值：$n=64$ 层，显存因子从 64 → 8（**省 8 倍**），代价 ~+1/3 前向算力。选择性重计算可把代价压到个位数百分比。

### C. 注意力的平方账（为什么长序列要 FlashAttention）

标准注意力物化 $QK^\top$（FP16）的显存：$2 \cdot B \cdot h \cdot L^2$ 字节（$B$=batch，$h$=头数）。

- $B=1,\ h=32,\ L=8192$：$2\times32\times8192^2 \approx 4.3\times10^9$ 字节 $\approx$ **4 GB**（单这一张中间矩阵！）。
- FlashAttention 不物化它，显存随 $L$ **线性**，这 4GB 直接消失。

### D. MoE 的"参数多但算力不变"

- Dense FFN：每 token 过全量 FFN，FLOPs $\propto$ 全部参数。
- MoE top-2 of 64 专家：每 token 只过 2 个专家，**激活算力 $\approx 2/64 = 1/32$**，但模型容量（总参数）大 32 倍 ⇒ 这就是 §2.3 "过参数化省算力"的工程实现。

---

## 评价 / 对照 / 局限（一张表收口）

| 技术 | 省什么 | 代价 | 是否近似 | 成熟度 |
|---|---|---|---|---|
| 混合精度(BF16) | 显存½ + 算力↑ | 极小 | 否 | 默认标配 |
| 激活重计算 | 激活显存 | +算力 ~5–33% | 否 | 默认标配 |
| ZeRO / Offload | 参数/梯度/优化器显存 | 通信↑ / PCIe 带宽 | 否 | 主流 |
| LoRA/PEFT | 优化器+梯度显存 | 仅适用微调 | 否(近似全量微调) | 主流 |
| FlashAttention | 注意力显存 $L^2{\to}L$ | 实现复杂 | **否(精确)** | 事实标准 |
| 稀疏训练 | 算力(理论) | GPU 难加速、调参难 | 是(子网络) | 研究为主 |
| 大批量+LAMB | 墙钟时间 | 泛化风险、调参 | 否 | 主流 |
| 低精度专用硬件 | 算力+功耗 | 需专用硬件、量化误差 | 是 | 学术/专用芯片 |

**这篇综述的局限（作为读者要清醒）**：
1. **综述天然滞后**：FP8 训练、3D 并行（TP×PP×DP）、序列并行、长上下文训练等"系统侧"进展在原文未必充分展开——以最新工程实践为准。
2. **数字依赖原文上下文**：152.6×、58.1× 等是**特定硬件/任务下的报告值**，并非通用加速比，**见原文**。
3. **三角权衡无银弹**：几乎每个技术都在 **算力 / 显存 / 精度（或泛化）** 之间换取，没有"全赢"，要按场景选组合。
4. **理论加速 ≠ 实测加速**：尤其非结构化稀疏、低精度，受访存模式与硬件支持限制，**以实测为准**。

---

## 🔗 跳转链接

- 总枢纽：[[00-知识地图]]
- 注意力与系统：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-algo/transformer/模型架构]]
- 训练系统：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[Reducing Activation Recomputation in Large Transformer Models]] · [[GaLore]]
- 压缩/精度：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-compression/sparsity/README]]
- 模型扩展：[[llm-algo/moe/README]]
- 指标口径：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 底层硬件：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]]
