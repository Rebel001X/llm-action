# 流水线并行（Pipeline Parallelism, PP）

> 把模型按层切成若干"阶段"，像工厂流水线一样让多张 GPU 接力计算，用"微批次"填满流水线以摊薄空泡。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-train/README]]

## 阅读地图

| 节 | 你将搞懂 | 关键公式/结论 |
|---|---|---|
| 0 | 一句话锚点 | PP = 层间切分 + 微批次接力 |
| 1 | 地基：为什么需要 PP | 显存墙 / 与 DP、TP 的分工 |
| 2 | 朴素 PP 与空泡 | $\text{Bubble fraction}=\frac{p-1}{m+p-1}$ |
| 3 | GPipe（按微批次填充） | 先全前向再全反向 |
| 4 | PipeDream / 1F1B | 稳态期一前一后，省激活显存 |
| 5 | 交错式 1F1B（虚拟阶段） | 空泡再缩 $v$ 倍 |
| 6 | 通信量与显存手算 | 点对点 send/recv，激活显存是瓶颈 |
| 7 | 切分策略与负载均衡 | embedding/loss 层不均的坑 |
| 8 | 与 TP/DP/ZeRO 组合 | 3D 并行 |
| 数值 | 端到端手算 | 空泡率 / 显存 / 通信 |
| FAQ | 常见坑 | — |

---

## 0. 一句话锚点

**数据并行（DP）复制模型、切分数据；张量并行（TP）把单层矩阵切开；流水线并行（PP）把不同的层放到不同设备上，前一段算完把激活值"传"给后一段。**

- DP：每张卡都有完整模型 → 显存不省，省的是吞吐。
- TP：一层内的大矩阵横/纵切 → 通信频繁（每层 2 次 all-reduce），通常限在单机内。
- **PP：把第 1\~8 层放 GPU0，第 9\~16 层放 GPU1……** → 只在阶段边界传一次激活，通信少，可跨机，但天然有"空泡"（bubble）。

```
单卡放不下的大模型（48 层）：
GPU0: [Layer 1..12 ] --激活--> GPU1: [Layer 13..24] --激活--> GPU2: [Layer 25..36] --激活--> GPU3: [Layer 37..48] -> Loss
        前向 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
        ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← 反向 梯度
```

---

## 1. 地基 / 前置

### 1.1 为什么需要 PP —— 显存墙

训练一个 Transformer，单卡显存被四样东西吃掉（详见 [[transformer内存估算]]）：

$$
M_{\text{total}} = \underbrace{M_{\text{params}}}_{\text{权重}} + \underbrace{M_{\text{grads}}}_{\text{梯度}} + \underbrace{M_{\text{optim}}}_{\text{优化器状态}} + \underbrace{M_{\text{act}}}_{\text{激活}}
$$

以 FP16 训练 + Adam 为例，**每个参数**占：权重 2B + 梯度 2B + Adam（FP32 动量+方差+FP32 主权重）= 2+2+(4+4+4)=**16 字节/参数**（混合精度的经典 16B 估算）。

> 175B 参数 × 16B = **2.8 TB**，单张 80GB 的 A100/H100 根本放不下。必须把"层"分到多卡上 → 这就是 PP 的出发点。

### 1.2 三种并行的分工（一张表记住）

| 维度 | 切什么 | 通信原语 | 通信频率 | 典型放置 |
|---|---|---|---|---|
| 数据并行 DP | 切 batch | all-reduce 梯度 | 每 step 1 次 | 跨机 |
| 张量并行 TP | 切单层矩阵 | all-reduce 激活 | 每层 2 次 | 单机内（NVLink） |
| 流水线并行 PP | 切层（阶段） | **point-to-point** send/recv | 每阶段边界 1 次 | 可跨机 |

PP 的通信量最小（只传阶段边界的激活张量），所以适合跨机、跨节点扩展。代价是**空泡**。

### 1.3 术语锚点

- **stage（阶段）**：一段连续的层，放在一个 PP rank 上。阶段数 = 流水线深度 $p$。
- **micro-batch（微批次, MBS）**：把一个 mini-batch 再切成 $m$ 份小块（chunks），逐个喂进流水线。
- **F / B**：Forward（前向）/ Backward（反向）。一个 B 的计算量约是 F 的 **2 倍**。

> DP + PP 的全局批量公式：`global_batch = mbs × chunks × dp_degree`。
> 例：DP=4，微批次大小 mbs=8，chunks=32 → 全局批量 = $8\times32\times4=1024$。

---

## 2. 朴素流水线并行与"空泡"

**朴素做法**：把一整个 mini-batch 当成一块，GPU0 算完整层 1\~12 的前向，传给 GPU1……反向同理。问题：**任一时刻只有 1 张卡在干活，其余全在等。**

```
朴素 PP（p=4 个阶段，1 个 batch）：横轴=时间，每格=一个阶段算一次
时间 → t1   t2   t3   t4   t5   t6   t7   t8
GPU0 [F ]                          [B ]
GPU1      [F ]                [B ]
GPU2           [F ]      [B ]
GPU3                [F ][B ]
        ↑___________利用率极低___________↑
利用率 ≈ 1/p = 1/4 = 25%
```

### 2.1 用微批次填充流水线

把 mini-batch 切成 $m$ 个微批次，让它们"错峰"进入流水线，就能让多张卡同时忙起来。

```
GPipe 思路（p=4 阶段，m=4 微批次），F=前向，数字=微批次编号
时间 →   1    2    3    4    5    6    7
GPU0  [F1][F2][F3][F4]
GPU1       [F1][F2][F3][F4]
GPU2            [F1][F2][F3][F4]
GPU3                 [F1][F2][F3][F4]
        ←填充→  ←——稳态(全忙)——→  ←排空→
         warmup        steady       drain
```

### 2.2 空泡率公式（核心手算）

设阶段数 $p$、微批次数 $m$。流水线需要 $p-1$ 步"填充"才装满，末尾 $p-1$ 步"排空"。这两段就是**空泡（bubble）**。

$$
\boxed{\text{Bubble fraction} = \frac{p-1}{m+p-1}}
$$

- 总时间单位（按前向计）：填充+稳态+排空 = $(p-1) + m + 0 = m + p - 1$ 个"时隙"。
- 理想（无空泡）需要 $m$ 个时隙，所以浪费比例 = $\dfrac{(m+p-1)-m}{m+p-1}=\dfrac{p-1}{m+p-1}$。

**直觉结论**：$m$ 越大（微批次越多），空泡越小。经验规则 $m \ge 4p$ 时空泡基本可忽略。

> 手算：$p=4$。
> - $m=4$：bubble = $3/7 = 42.9\%$（差）
> - $m=16$：bubble = $3/19 = 15.8\%$
> - $m=32$：bubble = $3/35 = 8.6\%$（好）

---

## 3. GPipe：先全前向，再全反向

GPipe（Google）的调度：**所有微批次的前向都做完，再统一做反向**，反向时用重计算（activation recomputation）省显存。

```
GPipe 完整时序（p=4, m=4）。F=前向 B=反向（B 约 2× 时长，这里画作 2 格）
时间 →
GPU0 F1 F2 F3 F4 ·· ·· ·· ·· ·· ·· B4 B4 B3 B3 B2 B2 B1 B1
GPU1 ·· F1 F2 F3 F4 ·· ·· ·· B4 B4 B3 B3 B2 B2 B1 B1 ··
GPU2 ·· ·· F1 F2 F3 F4 ·· B4 B4 B3 B3 B2 B2 B1 B1 ·· ··
GPU3 ·· ·· ·· F1 F2 F3 F4 B4 B3 B2 B1 ·· ·· ·· ·· ·· ··
                 ↑前向全做完↑  ↑反向倒着来↑
```

**GPipe 的痛点：显存峰值高。** 因为要先把 $m$ 个微批次的前向**全部**激活值缓存住，等反向才用。显存随 $m$ 线性增长：

$$
M_{\text{act}}^{\text{GPipe}} \propto m \times (\text{单微批次单阶段激活})
$$

→ 为了减空泡要增大 $m$，但 $m$ 一大显存又爆。**这个矛盾正是 1F1B 要解决的。**

---

## 4. PipeDream / 1F1B：一前一后，省激活显存

**1F1B（One-Forward-One-Backward）**：进入稳态后，每个阶段**做一次前向、紧接着做一次反向**，反向一做完就**立刻释放**那一份激活。这样同时驻留的激活数 = 流水线深度 $p$，而不是微批次数 $m$。

```
1F1B 时序（p=4, m=4）。注意稳态期 F、B 交替出现
时间 →   1   2   3   4   5   6   7   8   9  10  11
GPU0  F1  F2  F3  F4  B1  F5* B2  ...        ←warmup 灌 p 个 F
GPU1      F1  F2  F3  B1  F4  B2  F5* ...
GPU2          F1  F2  B1  F3  B2  F4  B3 ...
GPU3              F1  B1  F2  B2  F3  B3  F4  B4   ←最深阶段最先 1F1B
       ←warmup(p个F)→ ←——steady: F,B,F,B,F,B——→ ←drain(剩余B)→
```

### 4.1 1F1B 显存优势（手算）

| 调度 | 同时驻留激活份数 | 空泡率 |
|---|---|---|
| GPipe | $m$（全部微批次） | $\frac{p-1}{m+p-1}$ |
| **1F1B** | $\le p$（仅流水线深度） | $\frac{p-1}{m+p-1}$（相同） |

> 空泡率两者一样，但 **1F1B 的激活显存与 $m$ 解耦**。于是你可以把 $m$ 开得很大来压空泡，而显存只跟 $p$ 走。这就是 Megatron-LM 默认用 1F1B 的原因。

```
激活显存随时间（示意，越高越占显存）
GPipe:  ▁▂▃▄▅▆▇█  ████  ▇▆▅▄▃▂▁   ← 峰值正比于 m
1F1B :  ▁▂▃▄ ▄▄▄▄ ▄▄▄▄ ▄▃▂▁       ← 峰值被压在 ~p，平台型
```

### 4.2 权重版本问题（PipeDream 的"异步"坑）

PipeDream 为了零空泡曾用**异步**更新：不同微批次可能用不同版本的权重做前向/反向 → 引入"权重过时（staleness）"。解决方案：

- **weight stashing**：每个阶段缓存前向时用的权重版本，反向时取回 → 保证同一微批次前后向权重一致，但多存权重。
- **PipeDream-2BW / 1F1B（同步版）**：Megatron 采用同步 1F1B，所有微批次反向后统一更新，**数值上与朴素 PP 完全等价**（无 staleness），只是少了 GPipe 的显存峰值。✅ 训练首选。

---

## 5. 交错式 1F1B（Interleaved / 虚拟流水线阶段）

进一步压空泡：让**每张 GPU 持有多段不连续的层**（虚拟阶段, virtual stages）。设每卡 $v$ 个虚拟阶段，则每个"块"更小，填充/排空更快。

```
非交错: GPU0=[L1-12]            交错(v=2): GPU0=[L1-6]  [L25-30]
        GPU1=[L13-24]                     GPU1=[L7-12] [L31-36]
        GPU2=[L25-36]                     GPU2=[L13-18][L37-42]
        GPU3=[L37-48]                     GPU3=[L19-24][L43-48]
   一卡一大段，填充慢          一卡两小段交错，填充时隙变小 v 倍
```

空泡率变为：

$$
\text{Bubble}_{\text{interleaved}} = \frac{1}{v}\cdot\frac{p-1}{m+p-1}
$$

> **代价**：通信次数变成 $v$ 倍（点对点 send/recv 更频繁），所以 $v$ 不能无限大，通常 $v=2$ 或 $4$，需 NVLink/高带宽互联才划算。

> 手算：$p=4, m=16, v=2$ → bubble $= \frac{1}{2}\cdot\frac{3}{19}=7.9\%$（比非交错的 15.8% 砍半）。

---

## 6. 通信量与显存估算（PP 的两本账）

### 6.1 通信：只在阶段边界传激活

PP 用**点对点（P2P）send/recv**，不是 all-reduce。每个微批次在每条阶段边界，前向传一次激活、反向传一次激活的梯度。

设单个微批次在边界处的激活张量形状为 $[b, s, h]$（micro-batch × seq_len × hidden），FP16 每元素 2 字节。**一条边界、一个微批次、单方向**的通信量：

$$
V_{\text{boundary}} = b \cdot s \cdot h \cdot 2 \text{ 字节}
$$

```
阶段边界通信（点对点，非集合通信）：
GPU_i  --[send 激活 b×s×h]-->  GPU_{i+1}     (前向)
GPU_i  <--[recv 梯度 b×s×h]--  GPU_{i+1}     (反向)
   ↑ 一条链，只与相邻阶段通信，不广播
```

> 手算：$b=1, s=2048, h=12288$（GPT-3 175B），FP16。
> $V = 1\times2048\times12288\times2 = 50{,}331{,}648 \text{ B} \approx 48\text{ MB}$（每微批次每边界每方向）。
> 对比 TP 的 all-reduce：每层都要传 $\sim b\cdot s\cdot h$ 量级、且每层 2 次、跨整个 TP 组 → PP 通信量远小于 TP，所以 PP 适合跨机。

### 6.2 显存：每张卡只放 $1/p$ 的模型

权重/梯度/优化器状态被切成 $p$ 份：

$$
M_{\text{model per GPU}} \approx \frac{M_{\text{params}}+M_{\text{grads}}+M_{\text{optim}}}{p}
$$

但**激活显存不是简单 $/p$**，它由调度方式决定（见 §4.1）。1F1B 下：

$$
M_{\text{act per GPU}} \approx p_{\text{depth}} \times (b\cdot s\cdot h \times \text{层数/阶段} \times c)
$$

其中 $c$ 是与是否开重计算有关的系数。**开激活重计算（recompute）后激活显存可再降一个数量级，代价是多约 1/3 的前向 FLOPs。**

---

## 7. 切分策略与负载均衡

PP 的隐藏坑：**阶段算力不均 → 最慢的阶段拖累整条流水线**（木桶效应）。

```
不均衡（第一阶段含 embedding，最后含 LM head + loss）：
GPU0 [emb + 11层]  ███████░     <- 偏重
GPU1 [12层      ]  ██████
GPU2 [12层      ]  ██████
GPU3 [11层 + head+loss] ████████ <- 偏重 → 整体被它卡住

均衡做法：给重的阶段少分几层，或单独处理 embedding/loss
GPU0 [emb + 10层]  ██████
GPU1 [13层      ]  ██████
GPU2 [13层      ]  ██████
GPU3 [10层 + head] ██████        <- 各阶段时长拉齐
```

均衡要点：
1. **embedding / LM-head / loss** 不算"标准 Transformer 层"，单独计入首尾阶段的负载。
2. 词表巨大时 LM-head（$h\times V$）+ 交叉熵很重，常和 TP 联合切分。
3. 追求**各阶段前向耗时相等**，而非"层数相等"。
4. 阶段数 $p$ 不宜过大：$p$ 越大空泡 $\frac{p-1}{m+p-1}$ 越大、链路越长。

---

## 8. 与 TP / DP / ZeRO 组合 —— 3D 并行

实际训练超大模型用 **3D 并行**：TP（机内）× PP（跨机）× DP（再复制）。

```
3D 并行布局（举例：TP=2, PP=4, DP=2，共 2×4×2=16 GPU）
           DP 副本 0                       DP 副本 1
   ┌─ Stage0 ─┐ ... ┌─ Stage3 ─┐   ┌─ Stage0 ─┐ ... ┌─ Stage3 ─┐
   │ G0  G1   │     │ G6  G7   │   │ G8  G9   │     │ G14 G15  │
   │ └TP对┘   │     │ └TP对┘   │   │ └TP对┘   │     │ └TP对┘   │
   └──────────┘     └──────────┘   └──────────┘     └──────────┘
    └────────── PP 链 (4 阶段) ──┘   └────────── PP 链 (4 阶段) ──┘
              └──────────── DP all-reduce 跨副本 ────────────┘

通信带宽需求：TP(最高,机内NVLink) > PP(中,机间) > DP(每step一次all-reduce)
```

分工口诀：
- **TP 放最内层**（机内 NVLink，通信最密）。
- **PP 跨节点**（通信稀疏，容忍较低带宽）。
- **DP 在最外层**（每 step 一次梯度 all-reduce），可叠加 ZeRO 切优化器状态进一步省显存（见 [[ai-framework/deepspeed/README]]）。

---

## 数值手算：端到端一个例子

**设定**：48 层 Transformer，$h=12288$，$s=2048$，FP16+Adam，PP=8 阶段，DP=4，micro-batch $b=1$，chunks $m=32$。

**(1) 全局批量**
$$
\text{global batch} = b \times m \times \text{dp} = 1\times32\times4 = 128
$$

**(2) 空泡率（1F1B，非交错）**
$$
\text{bubble} = \frac{p-1}{m+p-1} = \frac{8-1}{32+8-1} = \frac{7}{39} \approx 17.9\%
$$
→ 想降到 ~9%，把 $m$ 翻倍到 64：$\frac{7}{71}\approx 9.9\%$；或开交错 $v=2$：$\frac{1}{2}\cdot17.9\%\approx 9.0\%$。

**(3) 每阶段边界通信量（单微批次单方向）**
$$
V = b\cdot s\cdot h\cdot 2 = 1\times2048\times12288\times2 \approx 48\text{ MB}
$$
一个 step 共 $m$ 个微批次、前向+反向两方向：$48\text{MB}\times32\times2 \approx 3\text{ GB} / \text{边界} / \text{step}$，但分散在整个 step 时间里且与相邻卡 P2P，远小于一次 175B 梯度 all-reduce 的量级。

**(4) 每卡模型显存（权重+梯度+优化器）**
假设总参数 $N$，16B/参数：每卡 $\approx \frac{16N}{p}=\frac{16N}{8}=2N$ 字节。若 $N=20\text{B}$ → 每卡模型态 $\approx 40\text{GB}$，激活另算（1F1B 峰值 $\sim p$ 份，配合重计算压到可接受）。

---

## 常见问题

| 问题 | 答案 |
|---|---|
| PP 和 TP 区别一句话？ | TP 切"一层内的矩阵"，通信密（每层 all-reduce）；PP 切"层"，通信稀（边界 P2P）。 |
| 为什么有空泡？ | 流水线填充/排空时部分卡空闲，比例 $\frac{p-1}{m+p-1}$。 |
| 怎么减空泡？ | 增大微批次数 $m$（经验 $m\ge4p$）；用交错式（$v$ 倍缩减）。 |
| GPipe vs 1F1B 选谁？ | 训练选 **1F1B**：空泡相同但激活显存与 $m$ 解耦、显存峰值低。 |
| 1F1B 会有 staleness（权重过时）吗？ | 同步 1F1B（Megatron）**没有**，数值等价朴素 PP；异步 PipeDream 才有，需 weight stashing。 |
| 反向比前向慢多少？ | 反向 FLOPs 约为前向 **2 倍**，排程时需考虑。 |
| 阶段怎么切才均衡？ | 拉齐"各阶段前向耗时"，而非层数；注意 embedding/LM-head/loss 偏重。 |
| PP 通信用什么原语？ | **点对点 send/recv**（不是 all-reduce），见 [[ai-infra/网络/集合通信原语]]。 |
| $p$ 越大越好吗？ | 否。$p$ 大 → 空泡大、链路长、负载更难均衡。 |
| 和 ZeRO 冲突吗？ | 不冲突，常 PP+TP+ZeRO-DP 三层叠加（3D 并行）。 |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-framework/megatron-lm/README]] — Megatron 1F1B / 交错式实现
- [[ai-framework/deepspeed/README]] — DeepSpeed PP + ZeRO 组合
- [[ai-infra/网络/集合通信原语]] — send/recv 与 all-reduce 对比
- [[llm-inference/大模型推理张量并行]] — 张量并行（与 PP 互补）
- [[llm-algo/transformer/模型架构]] — 被切分的对象：Transformer 层
- [[llm-algo/FLOPs]] — 前向/反向计算量估算
- [[transformer内存估算]] — 显存四大项与 16B/参数
- [[llm-optimizer/计算通信重叠]] — 用计算掩盖通信的思想
- [[llm-train/README]] — 训练全流程总览
