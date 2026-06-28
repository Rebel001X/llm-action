# 分布式训练（Distributed Training）总论

> 当单卡装不下、算不快、等不起时，把一个训练任务"切开"丢到多张 GPU / 多台机器上协同完成，就是分布式训练。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-inference/大模型推理张量并行]] · [[llm-algo/FLOPs]]

> ⚠️ 一句话警钟（保留原始要义）：**用纯 FP16 训练巨型 LLM 是一个禁忌**——它会带来更多的数值稳定性挑战（梯度下溢、Loss 尖刺、Adam 状态溢出）。现代大模型训练默认使用 **BF16 / FP16+loss-scaling / 混合精度 + FP32 master weights**，本文最后专门讲清这件事。

---

## 阅读地图

| 节 | 你会学到 | 关键产出 |
|---|---|---|
| 0 | 一句话锚点：为什么要分布式 | 三堵墙：显存墙、算力墙、时间墙 |
| 1 | 地基：一次训练迭代发生了什么 | 前向/反向/优化器 三阶段 + 显存四大件 |
| 2 | 单卡显存如何被吃光（手算） | 参数/梯度/优化器/激活 的字节账本 |
| 3 | 数据并行 DP / DDP | 复制模型、切分数据、AllReduce 梯度 |
| 4 | ZeRO（DeepSpeed）切优化器状态 | Stage 1/2/3 显存逐级下降 |
| 5 | 张量并行 TP（层内切） | 一层矩阵乘横切/竖切 + AllReduce |
| 6 | 流水线并行 PP（层间切） | 微批 + 气泡（bubble）手算 |
| 7 | 序列并行 / 上下文并行 | 切 LayerNorm/Dropout、切序列维 |
| 8 | 3D 并行：怎么组合 | DP×TP×PP 网格映射 |
| 9 | 通信原语与带宽账 | AllReduce/AllGather/ReduceScatter 通信量 |
| 10 | 混合精度与"FP16 禁忌" | BF16 vs FP16、loss scaling、master weights |
| ★ | 数值手算大全 | 7B/13B 模型多种并行下的显存与通信账本 |

---

## 0. 一句话锚点

分布式训练要打破**三堵墙**：

```
        显存墙                算力墙                  时间墙
   ┌──────────────┐    ┌──────────────┐       ┌──────────────┐
   │ 模型 + 优化器 │    │ 一次迭代的     │       │ 训完要数月     │
   │ 装不进一张卡  │    │ FLOPs 太大     │       │ 单卡等不起     │
   └──────┬───────┘    └──────┬───────┘       └──────┬───────┘
          │                   │                      │
     切「模型」            切「数据」               「叠卡」
   TP / PP / ZeRO        DP / DDP              更多 GPU 并行
```

- **切数据**（Data Parallel）：每张卡放一份完整模型，喂不同的数据 → 解决"算力/时间墙"。
- **切模型**（Tensor / Pipeline / ZeRO）：把模型的参数/层/状态拆到多卡 → 解决"显存墙"。
- 实战中三者组合，称为 **3D 并行**。

---

## 1. 地基：一次训练迭代里发生了什么

任何并行策略都建立在"一次 iteration"的三阶段之上，先把它焊死：

```
 一个 mini-batch 的生命周期
 ────────────────────────────────────────────────
  ① Forward   输入 x → 逐层算 → 得到 Loss
              （沿途缓存「激活值 activation」供反向用）
  ② Backward  从 Loss 反向 → 逐层算梯度 ∂L/∂W
              （消费激活值，产出「梯度 gradient」）
  ③ Optimizer 用梯度 + 优化器状态（Adam 的 m,v）更新参数 W
 ────────────────────────────────────────────────
  显存里同时存在的四大件：
   [参数 W] [梯度 G] [优化器状态 OS] [激活 A]
```

**为什么要先记住"四大件"？** 因为每一种并行策略，本质都是在回答：*这四件东西，哪几件复制、哪几件切开、切到几张卡？*

| 并行策略 | 参数 W | 梯度 G | 优化器状态 OS | 激活 A |
|---|---|---|---|---|
| 数据并行 DP | 复制 | 复制 | 复制 | 切（按 batch） |
| ZeRO-1 | 复制 | 复制 | **切** | 切 |
| ZeRO-2 | 复制 | **切** | **切** | 切 |
| ZeRO-3 | **切** | **切** | **切** | 切 |
| 张量并行 TP | **切**（层内） | **切** | **切** | 切（部分） |
| 流水线并行 PP | **切**（层间） | **切** | **切** | 切（按层） |

---

## 2. 单卡显存如何被吃光（必须会手算）

设模型参数量为 $\Psi$（单位：个）。用 **Adam + 混合精度**训练时，逐件算字节：

| 件 | 精度 | 每参数字节 | 公式 |
|---|---|---|---|
| FP16 参数（计算用） | 2B | 2 | $2\Psi$ |
| FP16 梯度 | 2B | 2 | $2\Psi$ |
| FP32 master 参数 | 4B | 4 | $4\Psi$ |
| FP32 Adam 动量 $m$ | 4B | 4 | $4\Psi$ |
| FP32 Adam 方差 $v$ | 4B | 4 | $4\Psi$ |

把"参数+梯度+优化器状态"加总（**不含激活**）：

$$
M_{\text{model}} = (2 + 2 + 4 + 4 + 4)\,\Psi = 16\,\Psi \text{ 字节}
$$

这就是著名的 **"每参数 16 字节"** 经验法则。

### 手算 1：7B 模型纯模型态显存

$$
\Psi = 7\times10^9,\quad M = 16 \times 7\times10^9 = 1.12\times10^{11}\text{ B} \approx 104\ \text{GB}
$$

一张 80GB 的 A100 / H100 **装不下** → 必须切。这就是显存墙的数字证据。

### 手算 2：激活值（Activation）有多大？

激活随 **batch size B、序列长度 S、层数 L、隐藏维 H** 增长。一个常用的 Transformer 单层激活估算（开启重计算前）：

$$
A_{\text{layer}} \approx s \cdot b \cdot h \cdot L \cdot (\text{34} + 5\cdot\frac{a\cdot s}{h})\ \text{字节(粗略量级)}
$$

直觉版手算：GPT-3 13B，$H=5120, L=40, S=2048, B=1$，激活可达**几十 GB**，常和模型态同量级。所以**激活重计算（gradient checkpointing）**几乎是大模型标配——用算力换显存。

> 详细的逐项字节拆解见 [[llm-algo/FLOPs]] 与 docs 内的"Transformer 内存估算"。

```
单卡显存账本（80GB 卡，7B 模型，B=4,S=2048 示意）
 ┌─────────────────────────────┐ 80 GB
 │ 激活 A        ~?? GB（可被重计算压缩）│
 ├─────────────────────────────┤
 │ 优化器状态 OS  48 GB (12Ψ)   │  ← 大头！ZeRO 专治
 ├─────────────────────────────┤
 │ 梯度 G        14 GB (2Ψ)     │
 ├─────────────────────────────┤
 │ 参数 W        14 GB (2Ψ+...) │
 └─────────────────────────────┘
 结论：优化器状态(m,v,master) 占了 3/4，这是 ZeRO 的攻击点。
```

---

## 3. 数据并行 DP / DDP（最常用的起点）

**思想**：每张卡放一份**完整模型副本**，把一个大 batch 切成 N 份小 batch，各算各的，最后把梯度**求平均**让所有副本同步。

```
       Global Batch = 256，4 张 GPU
  ┌────────┬────────┬────────┬────────┐
  │ GPU0   │ GPU1   │ GPU2   │ GPU3   │
  │ 模型副本│ 模型副本│ 模型副本│ 模型副本│  ← 完全相同
  │ data[0:64]│data[64:128]│data[128:192]│data[192:256]│
  └───┬────┴───┬────┴───┬────┴───┬────┘
      │ 各自 Forward + Backward → 局部梯度 g_i
      └────────────┬───────────────────┘
              AllReduce(求和/平均)
        g = (g0+g1+g2+g3)/4  →  广播回每张卡
      └────────────┬───────────────────┘
        各卡用相同 g 更新 → 模型副本仍然一致
```

**关键通信**：每步一次梯度 **AllReduce**，通信量 $\approx 2\Psi$ 字节（Ring-AllReduce 与卡数 N 几乎无关，见第 9 节）。

- **DataParallel（DP，单进程多线程）**：已过时，GIL+主卡瓶颈。
- **DistributedDataParallel（DDP，每卡一进程）**：现代标准。反向时**梯度边算边通信**（bucket 分桶 + 通信/计算重叠，见 [[llm-optimizer/计算通信重叠]]）。

> 局限：DP 解决不了显存墙——每张卡都要装完整的 $16\Psi$。7B 装不下，DP 再多卡也救不了。于是需要"切模型"。

---

## 4. ZeRO：把"复制"的优化器状态切开（DeepSpeed 核心）

ZeRO（Zero Redundancy Optimizer）是对 DP 的"零冗余"改造：DP 里 OS/G/W 在每张卡上是**完全冗余的副本**，ZeRO 把它们沿 DP 维**切片**，每卡只存 $1/N$。

```
  DP（冗余）            ZeRO-1        ZeRO-2          ZeRO-3
 ┌──────────┐      ┌──────────┐  ┌──────────┐   ┌──────────┐
 │ W (复制) │      │ W (复制) │  │ W (复制) │   │ W  1/N   │
 │ G (复制) │      │ G (复制) │  │ G  1/N   │   │ G  1/N   │
 │ OS(复制) │      │ OS 1/N   │  │ OS 1/N   │   │ OS 1/N   │
 └──────────┘      └──────────┘  └──────────┘   └──────────┘
  16Ψ/卡          4Ψ+12Ψ/N      2Ψ+14Ψ/N      16Ψ/N
```

每卡显存（混合精度，N 张卡）：

$$
\text{ZeRO-1}: 4\Psi + \frac{12\Psi}{N},\quad
\text{ZeRO-2}: 2\Psi + \frac{14\Psi}{N},\quad
\text{ZeRO-3}: \frac{16\Psi}{N}
$$

### 手算 3：7B + ZeRO-3 + 64 卡

$$
\frac{16 \times 7\times10^9}{64} = \frac{112\ \text{GB}}{64} \approx 1.75\ \text{GB/卡}
$$

模型态从 104GB/卡 暴降到 1.75GB/卡。**代价**：ZeRO-2 通信量与 DP 相当；**ZeRO-3** 在前向/反向时需要 **AllGather 参数**（用完即丢），通信量约升到 $3\Psi$，多了一次参数收集。**ZeRO-Offload / Infinity** 进一步把状态卸到 CPU/NVMe 内存，用带宽换显存。

> 详见 [[ai-framework/deepspeed/README]]。

---

## 5. 张量并行 TP（层内切，Megatron 招牌）

当**单层都太大**（如 H=12288 的 FFN 矩阵），就把**一个矩阵乘法**切到多卡。以 MLP $Y = \text{GeLU}(XA)\,B$ 为例：

```
  TP=2 切 MLP：A 按「列」切，B 按「行」切
  ───────────────────────────────────────────
   X ──┬──► [A1] ─► GeLU ─► [B1] ─┐
       │                          ├─(+)─► AllReduce ─► Y
       └──► [A2] ─► GeLU ─► [B2] ─┘
       GPU0 持 A1,B1        GPU1 持 A2,B2
  ───────────────────────────────────────────
  关键：A 列切后 GeLU 可独立做（非线性不跨卡）；
        B 行切后两边部分和相加 → 一次 AllReduce。
```

Attention 同理：按**注意力头**切（每卡算一部分 head），最后 AllReduce 合并。

- 每个 Transformer 层：**前向 2 次、反向 2 次 AllReduce**（一次在 Attention，一次在 MLP）。
- 通信极频繁、对带宽极敏感 → **TP 通常只在单机 8 卡内、走 NVLink**，不跨机。

### 手算 4：TP 单层通信量

设隐藏维 $H$，序列 $S$，batch $B$，BF16。一次 AllReduce 传输的张量是 $[B,S,H]$：

$$
\text{字节} = 2 \cdot B \cdot S \cdot H,\quad \text{AllReduce 实际流量} \approx 2\times(2BSH)
$$

H=12288、S=2048、B=1：单次约 $2\times(2\cdot1\cdot2048\cdot12288)\approx 0.1$ GB，每层 4 次、几十层 → 每步 GB 级流量，**必须 NVLink**。

> 推理侧的 TP 切法与此一脉相承，见 [[llm-inference/大模型推理张量并行]]。

---

## 6. 流水线并行 PP（层间切）

把模型**按层**纵向切成若干 **stage**，每个 stage 放一组卡上，像工厂流水线一样传递激活。

```
  PP=4：48 层切成 4 段，每段 12 层
  GPU0[L0-11] → GPU1[L12-23] → GPU2[L24-35] → GPU3[L36-47]
        激活 →        激活 →         激活 →   (Loss)
        梯度 ←        梯度 ←         梯度 ←
```

**朴素 PP 的问题——气泡（bubble）**：同一时刻只有一个 stage 在干活，其余空转：

```
  时间 →
 G0: F0 .. .. .. B0 ..        '.'=空闲(bubble)
 G1: .. F0 .. .. .. B0
 G2: .. .. F0 .. .. ..
 G3: .. .. .. F0B0 ..
       气泡占比高 → 利用率低
```

**解法：micro-batch 切分（GPipe/1F1B）**——把一个 mini-batch 切成 $m$ 个 micro-batch 填满流水线：

$$
\text{气泡占比} = \frac{p-1}{m+p-1}
$$

### 手算 5：PP 气泡

$p=4$ 个 stage，$m=8$ 个 micro-batch：

$$
\frac{4-1}{8+4-1} = \frac{3}{11} \approx 27\%\ \text{气泡}
$$

把 micro-batch 加到 $m=32$：$\frac{3}{35}\approx 8.6\%$。**结论：micro-batch 越多气泡越小**，但激活显存随之上升，需权衡。**1F1B** 调度（交替 forward/backward）能在不增显存的前提下进一步压气泡。

- PP 跨机友好：stage 之间只传**激活**（点对点 Send/Recv），通信量远小于 TP。

---

## 7. 序列并行 SP / 上下文并行

TP 把 Attention/MLP 切了，但 **LayerNorm、Dropout、残差** 这些"逐元素"操作仍在每卡冗余计算、冗余存激活。**序列并行（Megatron-SP）**沿**序列维 S** 把这些算子也切开：

```
  TP 切 hidden 维  +  SP 切 sequence 维
  ┌─────────────────────────────────────┐
  │ Attention/MLP : 切 H（TP）           │
  │ LayerNorm/Dropout : 切 S（SP）       │
  │ 衔接处用 AllGather / ReduceScatter   │
  └─────────────────────────────────────┘
  收益：激活显存进一步下降，长序列尤其明显。
```

**上下文并行 / Ring-Attention**：面向超长上下文（数十万 token），把序列切到多卡，用环形通信交换 K/V 块来算全局 Attention。与 [[llm-optimizer/FlashAttention]] 的分块思想同源——都是"分块算 Attention，不实例化完整 $S\times S$ 矩阵"。

> Attention 的分块/在线 softmax 见 [[llm-optimizer/FlashAttention]]；KV 的显存账见 [[llm-optimizer/kv-cache]] / [[llm-inference/KV-Cache优化]]。

---

## 8. 3D 并行：把三种切法拼成网格

真实的千亿模型训练 = **DP × TP × PP** 三维笛卡尔积。映射原则：**通信越频繁，放得越近**。

```
  例：64 卡 = TP2 × PP4 × DP8 ？  这里举 TP2×PP2×DP2=8 卡
  ───────────────────────────────────────────────
   维度    通信频率   放置                带宽需求
   TP      每层多次   同机 NVLink         最高
   PP      每 stage   机内/机间 P2P       中
   DP      每步一次   跨机 InfiniBand     最低（可重叠）
  ───────────────────────────────────────────────
        机器A                机器B
   ┌──────────────┐    ┌──────────────┐
   │ GPU0  GPU1   │    │ GPU4  GPU5   │   TP 在机内相邻卡
   │ (TP对)       │    │ (TP对)       │
   │ GPU2  GPU3   │    │ GPU6  GPU7   │   PP 跨这些 TP 组
   └──────────────┘    └──────────────┘
        └──────── DP 跨机 AllReduce ────────┘
```

口诀：**TP 不出机，PP 跨机省带宽，DP 包在最外层做梯度同步。** 框架层面由 [[ai-framework/megatron-lm/README]]（TP/PP/SP）+ [[ai-framework/deepspeed/README]]（ZeRO/DP）组合实现。

---

## 9. 通信原语与带宽账（必懂）

分布式训练的所有同步都归结为几个**集合通信原语**（NCCL 实现，详见 [[ai-infra/网络/集合通信原语]]）：

```
  AllReduce      = ReduceScatter + AllGather（最常用，DP 梯度同步）
  AllGather      = 把各卡分片拼成全量（ZeRO-3 取参数、TP 收集）
  ReduceScatter  = 求和后每卡只留一片（ZeRO 切梯度）
  Broadcast / Reduce / All2All（MoE 路由用 All2All，见 [[llm-algo/moe/README]]）
```

**Ring-AllReduce 通信量手算**：N 张卡、每卡数据量 $D$ 字节，Ring 算法每卡收发量与 N 几乎无关：

$$
\text{每卡通信量} \approx 2D\cdot\frac{N-1}{N} \xrightarrow{N\text{大}} 2D
$$

这就是 DP 可扩展性好的根因：**加卡几乎不增加单卡通信量**（带宽守恒），只要网络拓扑是环。

### 手算 6：DP 梯度同步耗时

7B 模型，梯度 $D=2\Psi=14$ GB，单卡有效带宽 100 GB/s（NVLink/IB 视情况）：

$$
t \approx \frac{2D}{BW} = \frac{2\times14}{100}\approx 0.28\ \text{s/步}
$$

若计算每步耗 1s，则通信占比 22%——**用计算/通信重叠（梯度边算边 AllReduce）可把这部分藏掉**（[[llm-optimizer/计算通信重叠]]）。

---

## 10. 混合精度训练与"FP16 禁忌"（呼应开篇）

> **用纯 FP16 训练巨型 LLM 是一个禁忌。** 现在把原因讲到底。

### 为什么纯 FP16 危险

FP16 只有 **5 位指数**，动态范围约 $6\times10^{-8} \sim 6.5\times10^4$。

```
  FP16 表示范围（窄！）
  下溢 ↓ 0.00006             上溢 ↑ 65504
  ──────┼───────────────────────┼──────
        小梯度直接变 0（下溢）   大激活/loss 变 Inf（上溢）
```

- **梯度下溢**：训练后期梯度很小，$<6\times10^{-8}$ 直接归零 → 模型停止学习。
- **数值不稳**：累加误差、Loss 尖刺（spike）、NaN。

### 两个救命补丁

**① FP32 Master Weights（主权重）**：参数副本用 FP32 保存（这就是第 2 节里那 $4\Psi$）。前向/反向用 FP16 算（省显存省算力），但**更新权重在 FP32 上做**，避免"大权重 + 小更新"被 FP16 舍入吃掉。

```
  FP16 计算  ──梯度──►  转 FP32  ──► 在 FP32 master 上更新
       ▲                                    │
       └──────── 拷回 FP16 供下次前向 ◄──────┘
```

**② Loss Scaling（损失缩放）**：反向前把 Loss 乘以一个大因子 $s$（如 $2^{16}$），让小梯度"放大"到 FP16 可表示区间，更新前再除回 $s$。动态 loss scaling 会在出现 Inf/NaN 时自动减小 $s$。

### 更好的答案：BF16

**BF16** 与 FP32 **同样 8 位指数**（动态范围一致，约 $10^{38}$），只是尾数少（精度低）。

```
  FP32 : [1符号][8指数][23尾数]   范围大、精度高
  BF16 : [1符号][8指数][ 7尾数]   范围=FP32，精度低 ← 几乎不会上下溢
  FP16 : [1符号][5指数][10尾数]   精度尚可，但范围窄 ← 危险
```

- **BF16 通常不需要 loss scaling**（范围够大），稳定性远好于 FP16，是当下大模型训练首选。
- 代价：尾数少 → 数值噪声大，但对深度网络训练影响可控。
- 更激进的 **FP8 训练**（H100+）用于加速，需更精细的缩放，见 [[llm-compression/quantization/fp8]]。

**一句话**：能用 **BF16** 就别用裸 FP16；非用 FP16 不可，就**必须**配 **FP32 master + 动态 loss scaling**。

---

## ★ 数值手算大全（速查）

| 场景 | 公式 | 代入 | 结果 |
|---|---|---|---|
| 模型态显存（Adam 混精） | $16\Psi$ | 7B | 104 GB（单卡爆） |
| 13B 模型态 | $16\Psi$ | 13B | 208 GB |
| ZeRO-3 / 64 卡 | $16\Psi/N$ | 7B,64 | 1.75 GB/卡 |
| ZeRO-2 / 8 卡 | $2\Psi+14\Psi/N$ | 7B,8 | 14+15.3≈29 GB/卡 |
| DP 梯度通信 | $\approx 2\Psi$ | 7B | ~28 GB 收发/步 |
| Ring-AllReduce/卡 | $2D\frac{N-1}{N}$ | D=14G,N=64 | ~27.6 GB |
| PP 气泡 | $\frac{p-1}{m+p-1}$ | p=4,m=8 | 27% |
| TP 单次 AllReduce | $2BSH$ | B1,S2048,H12288 | ~50 MB |

> 想把单层/单算子的字节与 FLOPs 拆得更细，去 [[llm-algo/FLOPs]] 和 [[llm-algo/transformer/模型架构]]。

---

## 常见问题（FAQ）

| 问题 | 答案 |
|---|---|
| DP 和 ZeRO 啥关系？ | ZeRO 是"零冗余版的 DP"，逻辑上仍是数据并行，只是把 OS/G/W 切片不再冗余复制。 |
| 显存不够，先上哪个？ | 顺序一般是：激活重计算 → ZeRO-2/3 / Offload → TP（单机8卡内）→ PP（跨机）。先省显存再切模型。 |
| TP 为什么不跨机？ | 每层多次 AllReduce，对带宽极敏感；跨机 IB 带宽 < 机内 NVLink，会被通信拖死。 |
| PP 为什么能跨机？ | stage 间只传激活的点对点 Send/Recv，通信量小、频率低。 |
| 加卡梯度同步会变慢吗？ | Ring-AllReduce 下单卡通信量 $\approx 2D$ 与卡数几乎无关，扩展性好。 |
| 为什么优化器状态这么占？ | Adam 的 master(4)+m(4)+v(4)=12 字节/参数，是参数本身(2)的 6 倍，故 ZeRO 优先切它。 |
| BF16 一定比 FP16 好？ | 训练稳定性上通常是；但 BF16 精度（尾数）更低，对个别数值敏感任务需评估。 |
| 激活重计算代价？ | 反向时重新前向一遍被丢弃的激活，约多 1/3 的计算量，换来激活显存大幅下降。 |
| MoE 用什么并行？ | 专家并行 + All2All 路由 token；见 [[llm-algo/moe/README]]。 |

---

## 🔗 跳转链接

- 训练总览与流程：[[llm-train/README]] · [[00-知识地图]]
- 框架实现：[[ai-framework/megatron-lm/README]]（TP/PP/SP）· [[ai-framework/deepspeed/README]]（ZeRO/Offload）
- 通信与硬件：[[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 算量与内存：[[llm-algo/FLOPs]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]]
- 注意力/KV：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 推理侧并行：[[llm-inference/大模型推理张量并行]] · [[llm-inference/README]]
- 重叠与精度：[[llm-optimizer/计算通信重叠]] · [[llm-compression/quantization/fp8]] · [[llm-compression/quantization/量化基础]]
- 混合专家：[[llm-algo/moe/README]]
- 微调/对齐（下游）：[[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
