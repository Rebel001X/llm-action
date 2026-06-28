# 数据并行（Data Parallelism, DP）

> 把同一份模型复制到多卡，每卡吃一片数据，靠 All-Reduce 把梯度对齐——这是分布式训练最基础、最常用的一招。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-algo/FLOPs]] · [[llm-train/README]]

## 阅读地图

| 节 | 内容 | 你将学到 |
|---|---|---|
| 0 | 一句话锚点 | DP 在干什么 |
| 1 | 地基/前置 | SGD、mini-batch、梯度、集合通信 |
| 2 | 朴素 DP（Parameter Server） | 中心化聚合 + 它的瓶颈 |
| 3 | 去中心化 DDP + Ring-All-Reduce | 主流方案与通信量推导 |
| 4 | 通信量逐数手算 | $2(N-1)/N \cdot M$ 怎么来的 |
| 5 | 计算/通信重叠 | bucket + 反向边算边传 |
| 6 | DP 的内存账本 | 为什么显存爆，引出 ZeRO |
| 7 | ZeRO 三阶段 | 把冗余切掉 |
| 8 | DP vs TP vs PP | 何时用哪种 |
| 9 | 数值手算示例 | 7B 模型实算一遍 |
| 10 | 常见问题 | 踩坑速查 |

---

## 0. 一句话锚点

**数据并行 = 模型复制 N 份 + 数据切 N 份 + 每步把 N 份梯度做平均（All-Reduce）后同步更新。**

核心不变量：**每一步结束后，所有卡上的模型参数完全一致**。只要保证这一点，DP 在数学上等价于"用 N 倍大的 batch 在单卡上训练"。

```
        全局 batch = 256
   ┌──────────┬──────────┬──────────┬──────────┐
   │ 64 样本  │ 64 样本  │ 64 样本  │ 64 样本  │   ← 数据切片
   ▼          ▼          ▼          ▼
 ┌────┐     ┌────┐     ┌────┐     ┌────┐
 │GPU0│     │GPU1│     │GPU2│     │GPU3│         ← 每卡一份完整模型副本
 │模型│     │模型│     │模型│     │模型│
 └─┬──┘     └─┬──┘     └─┬──┘     └─┬──┘
   │ g0       │ g1       │ g2       │ g3        ← 各自算出局部梯度
   └─────┬────┴─────┬────┴─────┬────┘
         ▼  All-Reduce(求和/求平均) ▼
   g = (g0+g1+g2+g3)/4   ← 每卡都拿到同一个平均梯度
   每卡各自 optimizer.step() → 参数仍然一致
```

---

## 1. 地基 / 前置

### 1.1 mini-batch SGD 回顾
单卡训练一步：
$$\theta \leftarrow \theta - \eta \cdot \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell(x_i;\theta)$$
其中 $B$ 是 batch 大小，$\eta$ 是学习率，$\nabla_\theta \ell$ 是单样本梯度。关键观察：**梯度是样本梯度的"平均"**，而平均可以拆开分块算再合并：

$$\frac{1}{B}\sum_{i=1}^{B} g_i = \frac{1}{N}\sum_{k=1}^{N}\Big(\underbrace{\frac{1}{B/N}\sum_{i \in \text{片}k} g_i}_{\text{第 }k\text{ 卡的局部平均梯度}}\Big)$$

这个"可分块求平均"的代数性质，就是数据并行成立的全部数学依据。

### 1.2 为什么能并行
- 每个样本的前向、反向**互不依赖**（同一组参数下）。
- 唯一需要"碰头"的点是：把各卡梯度合并成全局梯度。
- 这个"碰头"由**集合通信原语 All-Reduce** 完成 → 见 [[ai-infra/网络/集合通信原语]]。

### 1.3 训练一步的四个阶段
```
 ① Forward  : 各卡独立前向，得到 loss
 ② Backward : 各卡独立反向，得到局部梯度 g_k
 ③ Sync     : All-Reduce(g_0..g_{N-1}) → 平均梯度 g   ★唯一的通信点
 ④ Update   : 各卡用同一个 g 各自 optimizer.step()
```
DP 的全部工程难点，几乎都集中在第 ③ 步：**怎么把通信做得又快又省**。

---

## 2. 朴素数据并行：Parameter Server（中心化）

最早的做法：设一台/几台"参数服务器(PS)"，worker 算完梯度发给 PS，PS 求和后把新参数广播回去。

```
   worker0 ──push g0──┐                  ┌──pull θ──► worker0
   worker1 ──push g1──┤    ┌─────────┐   ├──pull θ──► worker1
   worker2 ──push g2──┼───►│   PS    │───┼──pull θ──► worker2
   worker3 ──push g3──┘    │ Σg, 更新 │   └──pull θ──► worker3
                          └─────────┘
```

**致命瓶颈：PS 的网络是热点。** 设模型有 $M$ 字节梯度，$N$ 个 worker：
- PS **入向**要收 $N\cdot M$ 字节（每个 worker 推一份）。
- PS **出向**要发 $N\cdot M$ 字节（广播给每个 worker）。
- → PS 单点带宽需求随 $N$ **线性增长**，$N$ 一大就堵死。

所以现代大模型训练几乎不用纯 PS，而用**去中心化的 All-Reduce**。

---

## 3. 去中心化 DDP + Ring-All-Reduce（主流）

PyTorch `DistributedDataParallel`(DDP)、Horovod 都基于 **Ring-All-Reduce**：把 N 张卡连成一个环，**没有中心节点**，每张卡只跟左右邻居通信，带宽利用率接近满。

把待规约的梯度向量切成 $N$ 块。算法分两个阶段，每阶段 $N-1$ 步。

### 3.1 阶段一：Reduce-Scatter（边传边加）
```
初始(N=4)，每卡持有完整梯度被切成 a/b/c/d 四块：
   GPU0: a0 b0 c0 d0
   GPU1: a1 b1 c1 d1
   GPU2: a2 b2 c2 d2
   GPU3: a3 b3 c3 d3

每步：第 k 卡把"某一块"发给右邻 k+1，右邻累加。
3 步后，每张卡上有"某一块"的全局求和结果：
   GPU0: ··  ··  ··  Σd   ← 持有 d 块的全和
   GPU1: Σa  ··  ··  ··    ← 持有 a 块的全和
   GPU2: ··  Σb  ··  ··
   GPU3: ··  ··  Σc  ··
```

### 3.2 阶段二：All-Gather（边传边覆盖）
```
再走 N-1 步，把每块的全和沿环传一圈，最终每卡都集齐 Σa Σb Σc Σd：
   GPU0: Σa Σb Σc Σd
   GPU1: Σa Σb Σc Σd
   GPU2: Σa Σb Σc Σd
   GPU3: Σa Σb Σc Σd        ← All-Reduce 完成，所有卡梯度一致
```

> Ring-All-Reduce 的精妙：**任何时刻每条链路上都在传输，不存在中心拥塞点**，带宽与 $N$ 无关（理论上）。详见 [[ai-infra/网络/集合通信原语]]。

---

## 4. 通信量逐数手算：$\frac{2(N-1)}{N}M$ 是怎么来的

设单卡完整梯度大小 $M$ 字节，$N$ 张卡。每块大小 $=M/N$。

| 阶段 | 步数 | 每步每卡发送 | 每卡总发送 |
|---|---|---|---|
| Reduce-Scatter | $N-1$ | $M/N$ | $(N-1)\cdot M/N$ |
| All-Gather | $N-1$ | $M/N$ | $(N-1)\cdot M/N$ |
| **合计** | $2(N-1)$ | — | $\boxed{\dfrac{2(N-1)}{N}M}$ |

**手算 N=4，M=1 GB：**
$$\text{每卡发送} = \frac{2\times(4-1)}{4}\times 1\text{GB} = \frac{6}{4} = 1.5\ \text{GB}$$

**关键结论：当 $N$ 很大时，$\frac{2(N-1)}{N}\to 2$，每卡通信量趋近 $2M$，与卡数无关！** 这正是 Ring-All-Reduce 能扩展到几千卡的根本原因——对比 PS 方案 PS 单点要收发 $2NM$（随 $N$ 线性爆炸）。

```
每卡通信量 vs 卡数 N：
 2M ┤                 ●━━━●━━━●━━━●  Ring(收敛到 2M)
    │           ●
1.5M┤      ●
    │   ●
    └──┬───┬───┬───┬───┬──► N
       2   4   8  16  32
   (PS 方案这里是一条斜率为 2M 的直线，早就冲出图外)
```

---

## 5. 计算 / 通信重叠（DDP 的提速核心）

如果"等反向全算完 → 再 All-Reduce"，通信时 GPU 算力闲置，效率低。DDP 用两个技巧把通信藏到计算后面：

### 5.1 反向传播边算边通信
反向是从最后一层往前算的。**某一层梯度一算完，立刻就可以发起这层的 All-Reduce**，不必等前面的层。于是：
```
时间 ──────────────────────────────►
反向: [L4][L3][L2][L1][L0]
通信:      [c4][c3][c2][c1][c0]   ← 错开一格，通信叠在后续层的计算上
                                  最后只剩 c0 一小段"露在外面"
```

### 5.2 Bucket（梯度分桶）
逐层 All-Reduce 会发起太多次小通信（每次都有固定启动开销/延迟）。DDP 把多层梯度攒进一个**桶（bucket，如 25MB）**，桶满了再一次性 All-Reduce，**用更少的大消息摊薄延迟**。

```
 grad层: g7 g6 g5 | g4 g3 g2 | g1 g0
 bucket: └─ Bucket2 ─┘└─ Bucket1 ─┘└Bucket0┘
         满即触发一次 All-Reduce
```

> 更系统的重叠原理 → [[llm-optimizer/计算通信重叠]]。

---

## 6. 数据并行的内存账本：为什么显存会爆

DP 的"模型复制"意味着**每张卡都存一整套训练状态**，毫无节省。以混合精度(fp16 计算 + fp32 master + Adam) 为例，设参数量 $\Psi$：

| 状态 | 精度 | 每参数字节 | 说明 |
|---|---|---|---|
| fp16 参数 | 2B | $2\Psi$ | 前向/反向用 |
| fp16 梯度 | 2B | $2\Psi$ | 反向产物 |
| fp32 参数副本(master) | 4B | $4\Psi$ | 优化器主权重 |
| fp32 动量 momentum | 4B | $4\Psi$ | Adam 一阶 |
| fp32 方差 variance | 4B | $4\Psi$ | Adam 二阶 |
| **合计** | — | $\boxed{16\Psi}$ | 著名的"16 倍" |

**手算：7B 模型（$\Psi=7\times10^9$）的优化器+参数+梯度状态：**
$$16 \times 7\times10^9 = 1.12\times10^{11}\ \text{字节} \approx 112\ \text{GB}$$

> 注意这 112GB **还不含激活值**。一张 80GB 的 A100/H100 根本放不下 → 单纯 DP 训不动 7B 全参。**而且这 112GB 在每张卡上都重复一份，纯属冗余。** 这正是 ZeRO 要解决的问题。激活内存细算见 [[llm-algo/FLOPs]]。

---

## 7. ZeRO：把 DP 的冗余切掉（DeepSpeed 招牌）

ZeRO(Zero Redundancy Optimizer) 洞察：上面 $16\Psi$ 在 N 卡上**各存一份是浪费**。既然最终要 All-Reduce，不如**把状态切片分摊到各卡**，需要时再通信取回。三阶段递进：

```
         参数P  梯度G  优化器状态O
 ZeRO-0 (纯DP):  全 │ 全 │ 全        每卡 16Ψ
 ZeRO-1      :  全 │ 全 │ 1/N        切优化器状态
 ZeRO-2      :  全 │ 1/N│ 1/N        再切梯度
 ZeRO-3      : 1/N │ 1/N│ 1/N        连参数也切(用时 all-gather)
```

每卡显存（混合精度，$K=12$ 为优化器系数）：
$$\text{ZeRO-1}: 2\Psi+2\Psi+\frac{12\Psi}{N}\quad \text{ZeRO-2}: 2\Psi+\frac{(2+12)\Psi}{N}\quad \text{ZeRO-3}: \frac{16\Psi}{N}$$

**手算 7B，N=64，ZeRO-3：** $\dfrac{16\times7\times10^9}{64} \approx 1.75\ \text{GB/卡}$ —— 从 112GB 降到 1.75GB！代价是 ZeRO-3 要额外 all-gather 参数，通信量约为 DP 的 1.5 倍。详见 [[ai-framework/deepspeed/README]]。

```
内存(GB/卡, 7B 模型, N=64):
 112 ┤●  ZeRO-0
     │
  ~9 ┤   ●  ZeRO-1
  ~4 ┤      ●  ZeRO-2
 1.75┤         ●  ZeRO-3
     └──┴──┴──┴──┴──►
```

---

## 8. DP vs TP vs PP：何时用哪种

| 维度 | 数据并行 DP | 张量并行 TP | 流水线并行 PP |
|---|---|---|---|
| 切什么 | 切**数据**(batch) | 切**单层权重**(矩阵) | 切**层**(stage) |
| 模型副本 | 每卡完整一份 | 一层被拆到多卡 | 一段层在一卡 |
| 通信原语 | All-Reduce(梯度) | All-Reduce(每层激活) | Send/Recv(激活) |
| 通信频率 | 每步一次(可重叠) | 每层多次(高频) | 每 micro-batch |
| 显存节省 | 无(ZeRO 才有) | 显著(权重被切) | 显著(层被切) |
| 适合 | 模型能放进单卡 | 单层太大放不下 | 模型层数多、太深 |
| 跨节点友好 | 友好(低频通信) | 不友好(需 NVLink) | 较友好 |

经验法则：**DP 优先用满（配 ZeRO）→ 单卡放不下再叠 TP(节点内 NVLink)→ 还放不下再叠 PP(跨节点)**，三者正交可组合成 3D 并行。张量并行细节见 [[ai-framework/megatron-lm/README]] 与 [[llm-inference/大模型推理张量并行]]。

```
3D 并行布局示意 (DP×TP×PP):
   PP stage0      PP stage1
   ┌────┬────┐   ┌────┬────┐
   │G0  │G1  │   │G4  │G5  │  ← TP 组(组内切权重)
   ├────┼────┤   ├────┼────┤
   │G2  │G3  │   │G6  │G7  │
   └────┴────┘   └────┴────┘
   └─ DP 副本0 ─┘ └─ DP 副本... All-Reduce 跨副本同梯度
```

---

## 9. 数值手算示例：一步完整的 DP

**设定**：4 卡，全局 batch=256（每卡 64），模型梯度 $M=1$GB，卡间互联带宽 $B=100$ GB/s（NVLink 级）。

**① 通信量**：每卡发送 $\frac{2(N-1)}{N}M = \frac{2\times3}{4}\times1 = 1.5$ GB。

**② 通信耗时**（理想，带宽打满）：
$$t_{\text{comm}} = \frac{1.5\ \text{GB}}{100\ \text{GB/s}} = 15\ \text{ms}$$

**③ 假设单卡前向+反向计算耗时** $t_{\text{compute}} = 50$ ms。

**④ 无重叠**：每步 $= 50 + 15 = 65$ ms。
**⑤ 有重叠（§5）**：通信藏在反向里，若反向占 30ms 足以覆盖 15ms 通信，则每步 $\approx 50 + (15-30)_{+} = 50$ ms。**重叠把通信开销几乎抹平。**

**⑥ 扩展效率**：单卡跑 batch=64 也是 50ms。理想 4 卡线性加速应是"用 50ms 处理了 4×64=256 样本"。
$$\text{加速比} = \frac{4\times 50}{50}=4\times（理想）,\quad \text{实测} \frac{4\times50}{65}\approx 3.08\times（无重叠）$$
有重叠时加速比回到接近 $4\times$ —— 这就是为什么计算/通信重叠如此关键。

**⑦ 学习率缩放**：batch 从 64→256（4 倍），按线性缩放规则学习率也乘 4（或用 $\sqrt{}$ 规则乘 2），否则大 batch 收敛会变差。

---

## 10. 常见问题

| 问题 | 解答 |
|---|---|
| DP 和 DDP 有啥区别？ | PyTorch `DataParallel`(单进程多线程，有 GIL+主卡瓶颈，已不推荐)；`DistributedDataParallel`(每卡一进程 + Ring-All-Reduce，主流)。 |
| DP 能省显存吗？ | **纯 DP 不能**，每卡存全套状态。要省显存得叠 ZeRO / TP / PP。 |
| BatchNorm 在 DP 下统计量对吗？ | 默认每卡只统计自己那片，需用 `SyncBatchNorm` 跨卡同步均值方差。LayerNorm 无此问题（不跨样本统计）。 |
| All-Reduce 求和还是求平均？ | 通常 All-Reduce **求和**，再各卡除以 $N$（或把 $1/N$ 折进 loss）得平均梯度，与单卡大 batch 等价。 |
| 大 batch 掉点怎么办？ | warmup + 线性/平方根学习率缩放；必要时用 LARS/LAMB 优化器。 |
| 通信成瓶颈怎么办？ | ①梯度 bucket 调大；②计算通信重叠；③梯度压缩/fp16 all-reduce；④升级互联(NVLink/IB)。 |
| 卡数翻倍通信会翻倍吗？ | Ring-All-Reduce 每卡通信量 $\frac{2(N-1)}{N}M$ 收敛到 $2M$，**几乎不随卡数增长**，这是它能扩展的关键。 |
| 梯度累积 (grad accumulation) 算 DP 吗？ | 不算并行，但思想相通：单卡多步累积梯度=模拟大 batch；可与 DP 叠加放大有效 batch。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站总入口
- [[ai-framework/deepspeed/README]] — ZeRO / DeepSpeed 完整实现
- [[ai-framework/megatron-lm/README]] — 与 TP/PP 组合的 3D 并行
- [[ai-infra/网络/集合通信原语]] — All-Reduce / Ring / Reduce-Scatter 原理
- [[llm-optimizer/计算通信重叠]] — bucket 与重叠的通用机制
- [[llm-algo/FLOPs]] — 计算量与激活内存估算
- [[llm-train/README]] — 训练全景与并行选型
- [[llm-inference/大模型推理张量并行]] — 张量并行对照（B07）
