# Megatron-DeepSpeed (框架)

> 把 NVIDIA Megatron 的**张量并行(TP)+流水并行(PP)** 和微软 DeepSpeed 的 **ZeRO/Offload** 缝在一起的训练框架；让"切模型"和"切优化器状态"两套互补的省显存手段同时生效，组成 **3D 并行(TP×PP×DP-ZeRO)**，BigScience 正是用它在 384 张 A100 上训出了 176B 的 BLOOM。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/megatron-lm/README]] [[ai-framework/deepspeed/README]] [[llm-train/megatron-deepspeed/README]]

## 阅读地图

| 节 | 主题 | 你将能回答 |
|----|------|-----------|
| 0 | 一句话锚点 | 它到底是"谁缝谁"、缝出来干嘛 |
| 1 | 地基/前置 | TP、PP、DP、ZeRO 各自只解决哪一种"爆" |
| 2 | 为什么要缝 | Megatron 缺什么、DeepSpeed 缺什么、互补在哪 |
| 3 | 整体架构 | 缝合点在哪、谁管模型谁管优化器 |
| 4 | 3D 并行的笛卡尔积 | GPU 怎么被三个维度切成网格 |
| 5 | rank 与通信组 | 一张卡同时属于哪几个组、各跑什么原语 |
| 6 | ZeRO 与 TP/PP 的边界 | 为什么 ZeRO 一般只配到 ZeRO-1、不能乱叠 |
| 7 | BLOOM 实战 | BigScience 的真实并行配置长什么样 |
| 8 | 配置怎么写 | 命令行参数 + ds_config.json 各管哪半边 |
| — | 数值例子 | 176B 模型在 8×6×8=384 卡上的网格手算 |
| — | 对照表/常见问题/跳转 | 选型与排错 |

---

## 0. 一句话锚点

**Megatron-DeepSpeed = Megatron-LM 的"切模型"能力 ⊕ DeepSpeed 的"切优化器状态 + 卸载"能力，二者拼成一个能训千亿参数的训练栈。**

- **Megatron-LM** 擅长把**一个 Transformer 层内部的大矩阵**切到多卡(张量并行 TP)，再把**不同层**切到多卡(流水并行 PP)。它解决的是"**单层/单模型参数装不下**"。
- **DeepSpeed** 擅长把**优化器状态、梯度、参数**这三样在数据并行维度上**去冗余**(ZeRO)，还能把它们**卸载(offload)到 CPU/NVMe**。它解决的是"**数据并行时每卡重复存一份太浪费**"。
- 二者**切的维度不同、互不冲突**，所以可以叠加。叠加后就是 **3D 并行**：TP × PP × DP(DP 这一维由 DeepSpeed ZeRO 接管)。

```
   一个 Transformer 太大，三个方向同时爆：

   ┌────────────────┬─────────────────┬──────────────────────┐
   │  层内矩阵太大   │   层数太多       │  DP 时每卡重复存优化器 │
   ├────────────────┼─────────────────┼──────────────────────┤
   │  Megatron TP   │   Megatron PP   │   DeepSpeed ZeRO     │
   │ (切单层的 W)    │  (把层分段)      │ (切 opt/grad/param)  │
   └────────────────┴─────────────────┴──────────────────────┘
            └──────────── 三者正交，可同时开 ───────────┘
                          = 3D 并行
```

> 命名提示：仓库名常见两个——微软的 `microsoft/Megatron-DeepSpeed` 与 BigScience 的 `bigscience-workshop/Megatron-DeepSpeed`(训 BLOOM 用的就是后者，是前者的 fork+改造)。两者思想一致，具体分支/参数**以官方仓库 README 为准**。

---

## 1. 地基：四种并行各解决哪种"爆"(原子拆解)

不先把四个名词分清楚，后面"3D 并行"就是一团浆糊。先看训练时显存花在哪：

```
┌───────────────────────────────────────────────────────────┐
│ 单卡显存 = 模型状态(参数+梯度+优化器状态) + 激活值 + 缓冲/碎片 │
└───────────────────────────────────────────────────────────┘
```

对一个参数量为 $\Psi$ 的模型，用 Adam + 混合精度(fp16)训练，**模型状态**的字节数约为：

$$
\text{模型状态} \approx \underbrace{2\Psi}_{\text{fp16 参数}} + \underbrace{2\Psi}_{\text{fp16 梯度}} + \underbrace{12\Psi}_{\text{fp32 参数+动量+方差(Adam)}} = 16\Psi \ \text{字节}
$$

即**每个参数约 16 字节**。175B 模型光模型状态就 $\approx 175\times10^9 \times 16 \approx 2.8\ \text{TB}$——一张 80GB 卡的 35 倍。必须切。四种切法：

| 并行 | 切什么 | 谁提供 | 解决的"爆" | 主要通信 |
|------|--------|--------|-----------|----------|
| **DP** 数据并行 | 切数据(batch)，模型每卡一整份 | PyTorch/DeepSpeed | 算力不够、想吃更大 batch | 梯度 All-Reduce |
| **TP** 张量并行 | 切**单层内的权重矩阵** | Megatron | 单层参数/激活装不下 | 层内 All-Reduce(频繁) |
| **PP** 流水并行 | 切**层(把模型纵向分段)** | Megatron/DeepSpeed | 总层数太多装不下 | 段间 P2P(点对点) |
| **ZeRO** | 在 DP 维**去冗余**地切模型状态 | DeepSpeed | DP 时每卡重复存一份太浪费 | 参数 All-Gather / 梯度 Reduce-Scatter |

> 关键直觉：**TP/PP 在"切模型本身"，ZeRO 在"切数据并行里的冗余"。前者改变模型在卡上的物理布局，后者只是把同一份逻辑模型的状态分散存。维度正交，故可叠加。** 细节见 [[ai-framework/megatron-lm/README]] 与 [[ai-framework/deepspeed/README]]。

---

## 2. 为什么要缝？各自缺什么

```
   Megatron-LM 单独用：
   ┌──────────────────────────────────────────────┐
   │ ✓ TP/PP 很强，kernel 高效                      │
   │ ✗ DP 维仍是朴素 DDP：每个 DP rank 存整份优化器  │
   │   状态 → 优化器显存随 DP 度数线性浪费           │
   └──────────────────────────────────────────────┘

   DeepSpeed 单独用：
   ┌──────────────────────────────────────────────┐
   │ ✓ ZeRO 把优化器/梯度/参数切碎，DP 维零冗余      │
   │ ✓ Offload 到 CPU/NVMe                          │
   │ ✗ 没有 Megatron 那套层内 TP 的高效实现         │
   │   (单层矩阵太大时仍然装不下)                    │
   └──────────────────────────────────────────────┘

   缝起来：
   ┌──────────────────────────────────────────────┐
   │ Megatron 管 TP+PP(切模型) ⊕ DeepSpeed 管       │
   │ DP 维 ZeRO+Offload(切冗余) → 两短互补成一长     │
   └──────────────────────────────────────────────┘
```

一句话：**Megatron 把"一份模型"切小，DeepSpeed 把"多份重复的状态"去重。** 缺了 Megatron，单层矩阵 OOM；缺了 DeepSpeed，DP 维优化器状态成倍浪费。所以训千亿模型时常常两个都要。

---

## 3. 整体架构：缝合点在哪

代码层面，Megatron-DeepSpeed 的做法是**保留 Megatron 的模型定义与 TP/PP 调度，把数据并行那一层的优化器/梯度管理换成 DeepSpeed 引擎**。

```
        ┌─────────────────────────────────────────────────┐
        │             训练脚本 (pretrain_gpt.py 等)         │
        └─────────────────────────────────────────────────┘
                 │ 模型构建/前反向            │ 优化器/梯度/状态
                 ▼                           ▼
        ┌──────────────────────┐   ┌──────────────────────┐
        │   Megatron 部分        │   │   DeepSpeed 部分       │
        │ ─ Transformer 层定义   │   │ ─ deepspeed.initialize │
        │ ─ TP 切矩阵(ColumnPara │   │ ─ ZeRO 优化器(切 opt/  │
        │   llelLinear/RowParal) │   │   grad/param)          │
        │ ─ PP 流水调度(1F1B)    │   │ ─ Offload(CPU/NVMe)    │
        │ ─ 各并行通信组的建立    │   │ ─ 混合精度/梯度累积     │
        └──────────────────────┘   └──────────────────────┘
                 │                           │
                 └───────── 同一组 GPU ───────┘
              (Megatron 建好 TP/PP/DP 通信组，
               DeepSpeed 在"DP 组"上做 ZeRO)
```

**缝合的关键：通信组(process group)的划分由 Megatron 负责建立**(它把全部 GPU 切成 TP 组、PP 组、DP 组)，而 **DeepSpeed 的 ZeRO 只在"DP 组"内部生效**——因为 ZeRO 的去冗余本就是针对数据并行副本的。这正是二者能正交叠加的工程根因。

---

## 4. 3D 并行：GPU 被切成一个三维网格

设总卡数为 $N$，三个并行度满足：

$$
N = \text{TP} \times \text{PP} \times \text{DP}
$$

每张 GPU 由一个三元组 $(t, p, d)$ 唯一标识：$t$ 是它在 TP 组里的位置，$p$ 是它在哪个流水段，$d$ 是它属于第几个数据并行副本。

```
   例：TP=2, PP=2, DP=2 → 共 8 卡，排成 2×2×2 立方体

         DP=0 副本                 DP=1 副本
   ┌──────────────────┐     ┌──────────────────┐
   │  PP 段0  PP 段1   │     │  PP 段0  PP 段1   │
   │ ┌────┐  ┌────┐   │     │ ┌────┐  ┌────┐   │
   │ │G0 G1│  │G2 G3│  │     │ │G4 G5│  │G6 G7│  │
   │ └────┘  └────┘   │     │ └────┘  └────┘   │
   │  ↑TP=2   ↑TP=2   │     │  ↑TP=2   ↑TP=2   │
   └──────────────────┘     └──────────────────┘

   ─ 横向(TP)：G0↔G1 切同一层的矩阵，层内 All-Reduce
   ─ 纵向(PP)：段0→段1 传激活，P2P send/recv
   ─ 跨副本(DP)：G0↔G4 同位置卡做梯度同步 + ZeRO 切分
```

**配置原则(经验，非死规则)：**
1. **TP 优先填满单机内的卡**(NVLink 高带宽)，因为 TP 通信最频繁、最吃带宽。典型 $\text{TP}\le 8$(单节点 8 卡)。
2. **PP 跨节点**，因为它只在段间传一次激活，对带宽不敏感，适合走较慢的 InfiniBand。
3. **DP 放最外层**，用 ZeRO 去冗余 + 吃更大全局 batch。

---

## 5. 一张卡同时属于哪几个通信组

这是初学者最容易绕晕的地方。**同一张 GPU 会同时是三个通信组的成员**，每个组跑不同的集合通信原语。

```
   GPU (t=0, p=1, d=0) 同时在：

   ┌─ TP 组：{所有 (·, 1, 0)} ── 层内 All-Reduce(前向/反向各一次)
   │
   ├─ PP 组：{所有 (0, ·, 0)} ── 与前后段 P2P 传激活/梯度
   │
   └─ DP 组：{所有 (0, 1, ·)} ── 梯度 Reduce-Scatter + 参数 All-Gather (ZeRO)
```

| 维度 | 组内成员 | 何时通信 | 原语 | 数据量级 |
|------|----------|----------|------|----------|
| TP | 同层不同切片 | 每个 TP 层前向 1 次、反向 1 次 | All-Reduce | 激活，**最大、最频繁** |
| PP | 相邻流水段 | 每个 micro-batch 段边界 | P2P send/recv | 一份激活，中等 |
| DP | 同位置不同副本 | 每步反向后 | Reduce-Scatter + All-Gather | 梯度/参数，可与计算重叠 |

> 因为 TP 的 All-Reduce 既大又密，所以**务必让 TP 组落在同一物理节点的 NVLink 上**，否则训练会被通信拖死。集合通信原语细节见 [[ai-framework/megatron-lm/README]]。

---

## 6. ZeRO 与 TP/PP 的边界：为什么常只配 ZeRO-1

ZeRO 分三级(回顾 [[ai-framework/deepspeed/README]])：

```
   ZeRO-1：切【优化器状态】     省 ~4×(Adam 那 12Ψ 的大头)
   ZeRO-2：切【优化器状态+梯度】 再省一点
   ZeRO-3：切【优化器状态+梯度+参数】 参数也分散，前向时临时 All-Gather
```

在 Megatron-DeepSpeed 里，**TP/PP 已经把"参数和梯度"切到不同卡上了**，所以:

- **ZeRO-1**(只切优化器状态)与 TP/PP **配合最干净**：优化器状态那 $12\Psi$ 的大头在 DP 维去冗余，而参数/梯度的切分交给 Megatron 的 TP/PP。这是 BLOOM 等大规模训练的**常见组合：TP+PP+ZeRO-1**。
- **ZeRO-2/3** 也要切梯度甚至参数，会与 PP 的梯度管理、TP 的参数布局**产生职责重叠和冲突**，配置复杂、通信叠加。所以**开了 PP 时，DeepSpeed 侧通常只到 ZeRO-1**(具体兼容性以官方文档为准)。
- 若**不开 PP**(只 TP + DP)，则可以放心上 ZeRO-2/3，让 DeepSpeed 把梯度/参数也切了。

```
   组合速查(经验值，以官方为准)：
   ┌──────────────────────┬──────────────────────────────┐
   │ TP + PP + DP          │ DeepSpeed 用 ZeRO-1 (最稳)     │
   │ TP + DP (无 PP)        │ ZeRO-2/3 皆可，进一步省显存    │
   │ DP only (模型本身放得下)│ 纯 ZeRO-1/2/3，不需 Megatron  │
   └──────────────────────┴──────────────────────────────┘
```

---

## 7. BLOOM 实战：BigScience 怎么用它训 176B

BLOOM(BigScience Large Open-science Open-access Multilingual)是 2022 年由 BigScience 协作训练的 176B 参数多语言大模型，**训练框架正是 `bigscience-workshop/Megatron-DeepSpeed`**。它是"Megatron-DeepSpeed 能跑通千亿级"的标志性公开案例。

公开资料中的典型规模(约数，**以 BigScience 官方记录为准**)：

```
   模型：BLOOM-176B
   硬件：约 384 张 NVIDIA A100 80GB(法国 Jean Zay 超算)
   并行：3D 并行
        ┌─ TP = 4    (单层矩阵切 4 份，落在节点内 NVLink)
        ├─ PP = 12   (模型纵向切 12 段)
        └─ DP = 8    (8 个数据并行副本，ZeRO-1 去冗余)
   校验：TP × PP × DP = 4 × 12 × 8 = 384  ✓ = 总卡数
   精度：bf16 混合精度
   其它：激活重计算(activation checkpointing)进一步省激活显存
```

```
   它解决的三件事，正好对应三种并行：

   单层 70k×... 的大矩阵装不下  → TP=4   切矩阵
   70 层 Transformer 一卡放不下 → PP=12  分段
   想吃更大 batch、别浪费优化器  → DP=8 + ZeRO-1
```

BLOOM 的意义：它证明了**开源协作 + Megatron-DeepSpeed 框架**可以复刻 GPT-3 量级的训练，并把训练细节(配置、日志、踩坑)完全公开，成为后来很多团队的参照模板。训练侧落地细节见 [[llm-train/megatron-deepspeed/README]]。

---

## 8. 配置怎么写：命令行 + ds_config.json 分工

Megatron-DeepSpeed 的配置**天然分两半**，正对应它"两个框架缝合"的本质：

```
   ┌─────────────────────────────┐   ┌────────────────────────────┐
   │ Megatron 侧：命令行参数        │   │ DeepSpeed 侧：ds_config.json │
   │ ─ 模型/并行的"物理切法"        │   │ ─ ZeRO/优化器/精度的"省显存" │
   └─────────────────────────────┘   └────────────────────────────┘
```

**(a) Megatron 命令行(控制 3D 并行的切法)** —— 以下为示意，**确切参数名以官方脚本为准**：

```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 4 \     # TP 度
  --pipeline-model-parallel-size 12 \  # PP 度
  --num-layers 70 --hidden-size 14336 \
  --micro-batch-size 1 \               # 每次前向的最小 batch
  --global-batch-size 2048 \           # 全局 batch(决定 DP/梯度累积)
  --seq-length 2048 \
  --recompute-activations \            # 激活重计算
  --deepspeed --deepspeed_config ds_config.json   # 把 DP 维交给 DeepSpeed
```

> DP 度通常**不直接写**，而是由 `总卡数 ÷ (TP×PP)` 推出来；全局 batch ÷ (micro-batch × DP) = 梯度累积步数。

**(b) DeepSpeed 配置 `ds_config.json`(控制 ZeRO/精度/卸载)** —— 示意：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "zero_optimization": {
    "stage": 1,                    // 配 PP 时常用 ZeRO-1
    "offload_optimizer": { "device": "cpu" }   // 可选：卸载到 CPU
  },
  "bf16": { "enabled": true },     // 混合精度
  "gradient_clipping": 1.0
}
```

**心智模型：左边(命令行)决定"模型怎么被物理切到卡上"，右边(json)决定"剩下的 DP 维怎么省显存"。** 字段名与版本强相关，**务必以你所用仓库分支的官方文档为准**(参 [[ai-framework/deepspeed/README]] 的配置章节)。

---

## 数值例子：384 卡网格的手算与显存账

**例 1：网格是否自洽。** 给定 TP=4、PP=12、DP=8，

$$
N = \text{TP}\times\text{PP}\times\text{DP} = 4\times12\times8 = 384 \ \text{卡} \quad\checkmark
$$

若想把 DP 提到 16(吃更大 batch)，则需 $4\times12\times16=768$ 卡。

**例 2：优化器状态被 ZeRO-1 省了多少。** 176B 模型 Adam 优化器状态(fp32 参数+动量+方差)约 $12\Psi = 12\times176\times10^9 \approx 2.1\ \text{TB}$。

- 没 ZeRO：每个 DP 副本(每个 $(t,p)$ 位置)都存一整份对应分片，DP=8 → **重复 8 份**。
- 开 ZeRO-1：这 8 份在 DP 组内**切成 8 片各存 1/8**，单卡该项显存 $\div 8$。

$$
\text{单卡优化器状态} \xrightarrow{\text{ZeRO-1}} \frac{1}{\text{DP}} = \frac{1}{8} \ \text{原来的量}
$$

**例 3：单层矩阵被 TP 切了多少。** hidden=14336 的 FFN 升维矩阵约 $14336\times(4\times14336)\approx 8.2\times10^8$ 参数，TP=4 后每卡只持 $1/4 \approx 2.05\times10^8$。

**综合直觉：** TP 把"单层矩阵"$\div4$，PP 把"层数"$\div12$，ZeRO-1 把"优化器冗余"$\div8$——三刀各砍一个维度，才把 2.8TB 的庞然大物塞进 80GB 的卡里。

---

## 常见问题

| 问题 | 答 |
|------|-----|
| Megatron-DeepSpeed 是新框架吗？ | 不是，是把 Megatron-LM 和 DeepSpeed 两个已有框架**缝合**的集成代码库 |
| 微软版和 BigScience 版有何区别？ | BigScience 版是微软版的 fork，为训 BLOOM 做了大量改造；思想一致，分支不同，**以官方为准** |
| ZeRO 和 TP 冲突吗？ | 不冲突，切的维度正交：TP 切层内矩阵，ZeRO 切 DP 维冗余 |
| 为什么开 PP 时只用 ZeRO-1？ | ZeRO-2/3 也要管梯度/参数，会与 PP 职责重叠；ZeRO-1 只切优化器状态，最干净(以官方兼容性为准) |
| TP 该开多大？ | 一般 $\le 8$，且**限制在单节点 NVLink 内**，因其 All-Reduce 又大又频繁 |
| DP 度怎么定？ | 通常 $\text{DP}=N/(\text{TP}\times\text{PP})$，不直接写死 |
| 和纯 DeepSpeed(只 ZeRO)比何时选它？ | 单层矩阵都装不下、需要 TP/PP 时选它；模型本身放得下、只想省 DP 冗余则纯 DeepSpeed 足够 |
| 和 Megatron-LM 原生比？ | 原生 Megatron 的 DP 是朴素 DDP(优化器状态成倍冗余)，缝 DeepSpeed 后 DP 维可 ZeRO 去冗余 |
| 现在还推荐用吗？ | BLOOM 时代的经典栈；新项目也常直接用 Megatron-LM 原生(已内置类 ZeRO 的分布式优化器)或 Megatron-Core，**选型以最新生态为准** |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，从这里找其它主题
- [[ai-framework/megatron-lm/README]] — TP/PP/SP 的切法与高效 kernel(本框架的"切模型"一半)
- [[ai-framework/deepspeed/README]] — ZeRO 三级、Offload、配置 JSON(本框架的"省冗余"一半)
- [[llm-train/megatron-deepspeed/README]] — 用该框架实际跑训练任务的落地步骤
