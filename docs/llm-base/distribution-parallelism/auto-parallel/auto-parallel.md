# 自动并行（Auto-Parallel）

> 让编译器/搜索器替你决定"模型怎么切、放哪张卡、怎么通信"，把分布式训练从"手工调参"变成"自动求解"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/大模型推理张量并行]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-algo/FLOPs]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | 切分=排布+通信 |
| 1 | 地基：手工并行为什么不够 | DP/TP/PP 的组合爆炸 |
| 2 | 把"并行"抽象成数学对象 | SBP / sharding spec |
| 3 | 算子内并行（Intra-op） | TP、激活切分、通信代价 |
| 4 | 算子间并行（Inter-op） | PP、stage 划分、气泡 |
| 5 | 代价模型：怎么给一个方案打分 | 计算+通信+内存 |
| 6 | 搜索：在指数空间里找最优 | ILP / DP / 贪心 |
| 7 | 代表系统巡礼 | Alpa/FlexFlow/Tofu/Colossal-Auto/MindSpore |
| 数值节 | 逐数手算一次切分代价 | AllReduce 字节数 |
| FAQ | 易错点 | 表格 |

## 0. 一句话锚点

**自动并行 = 给定（模型计算图 + 集群拓扑），自动求解一个"切分与放置方案"，使端到端时间最短，且每张卡显存不溢出。**

它要同时回答三个问题：

```
┌────────────────────────────────────────────────────────┐
│  Q1 每个张量/算子怎么切？     → 算子内并行 (intra-op)   │
│  Q2 计算图怎么分段放到哪些卡？→ 算子间并行 (inter-op)   │
│  Q3 切分边界怎么补通信？      → 自动插入 AllReduce/...  │
└────────────────────────────────────────────────────────┘
```

手工并行（Megatron 的 TP、DeepSpeed 的 ZeRO、GPipe 的 PP）是人来回答这三问；自动并行是**搜索器/编译器**来回答。

## 1. 地基：手工并行为什么"不够用"

回顾三种基础并行（细节见 [[ai-framework/megatron-lm/README]]）：

- **数据并行 DP**：每卡一份完整模型，切 batch，靠 AllReduce 同步梯度。
- **张量并行 TP**：把单个矩阵乘按行/列切到多卡（见 [[llm-inference/大模型推理张量并行]]）。
- **流水线并行 PP**：把"层"切成 stage，像流水线一样错峰执行。

真实大模型要**混合**三者（3D 并行），自由度爆炸：

```
设 N 张卡，要分配 (dp, tp, pp) 使 dp·tp·pp = N
N=512 时合法因子组合就有几十种；
每种组合下，每一层还能选「切哪一维 / 是否重计算 / 是否 offload」
   ⇒ 方案数随层数 L 指数增长：O(K^L)
```

人类只能凭经验试几个点；**自动并行把它变成一个最优化问题**，在指数空间里求解。

> 关键洞察：DP / TP / PP 不是三种"模式"，而是同一个切分空间里的三个**子维度**。自动并行统一表达它们，而非分别枚举。

## 2. 把"并行"抽象成数学对象：分布式张量

要让机器搜索，先得让"切分"可被描述、可被组合。核心抽象：**给每个张量打一个"分布标签"**。OneFlow 叫 **SBP**，PyTorch DTensor / Alpa 叫 **sharding spec**。

一个张量在 N 卡上的排布只有三类原语：

```
张量 X，沿某网格维放置：
  S(i)  Split   : 沿 X 的第 i 轴切，每卡持有一片  (例: 按列切权重)
  B     Broadcast/Replicate: 每卡持有完整副本     (例: 复制的 bias)
  P     Partial : 每卡持有"部分和"，逻辑值=各卡相加 (AllReduce 前的中间态)
```

ASCII：一个 `[4,4]` 矩阵在 2 卡上的几种排布

```
  原矩阵 X[4,4]            S(0) 按行切            S(1) 按列切
  ┌────────────┐        卡0 ┌──────┐ 行0-1     卡0 ┌──┐ 列0-1
  │ x00 ... x03│             └──────┘            卡1 └──┘ 列2-3
  │  ...       │        卡1 ┌──────┐ 行2-3
  │ x30 ... x33│             └──────┘
  └────────────┘

  B 复制: 卡0 = 卡1 = 完整 X      P 部分和: X = 卡0_片 + 卡1_片 (需 AllReduce 才"实"）
```

**为什么需要 P（Partial）这一类？** 因为矩阵乘"切内积维"会天然产生部分和：
$Y = A_{[m,k]} B_{[k,n]}$，若把 $k$ 维切两半，则 $Y = A_1 B_1 + A_2 B_2$，每卡先算一个加项，得到 `P` 态，再 AllReduce 才得到完整 `Y`。**有了 SBP，"何时该插 AllReduce" 就从一个布尔标签的转换里自动推出来**——这正是自动并行的地基。

## 3. 算子内并行（Intra-operator Parallelism）

**Intra-op = 把单个算子（矩阵乘/注意力）的内部切到多卡**，本质就是给每个算子的输入/输出张量选 SBP。

以 Transformer MLP（[[llm-algo/mlp]]）的两次矩阵乘为例，$Y = \text{GeLU}(XA)B$：

```
列切 A，行切 B（Megatron 经典方案，全程只 1 次 AllReduce）：
        X (B复制)
          │
   ┌──────┴──────┐
 卡0: X·A_col0  卡1: X·A_col1     A 按列切 S(1)，输出按列切 → GeLU 逐元素不跨卡 ✔
   │             │
 卡0: ·B_row0  卡1: ·B_row1       B 按行切 S(0)，每卡得"部分和" P
   └──────┬──────┘
       AllReduce                  把 P → 实张量
          │
          Y (B复制)
```

**自动并行在这里做什么？** 它不预设"列切再行切"，而是为 $A,B$ 各自枚举 `{S(0),S(1),B}`，组合出所有合法切法，再用代价模型（第 5 节）比较：

```
候选切法              产生的通信
A:S(1) B:S(0)     →  1×AllReduce(Y)        ← Megatron 选的，通信最省
A:S(0) B:S(1)     →  AllGather + ...        ← 更贵
A:B   B:B         →  0 通信，但每卡算全量    ← 不省算力（退化为复制）
...
```

注意力（[[llm-algo/transformer/模型架构]]）同理：多头注意力天然可按"头"切（S 在 head 维），自动并行会发现这一点。激活、LayerNorm 的切分也一并求解（这正是 Megatron 序列并行手工做、自动并行自动做的事）。

## 4. 算子间并行（Inter-operator Parallelism）

**Inter-op = 把计算图切成若干"段（stage）"，不同段放不同卡组**，本质是流水线并行 PP 的泛化。

```
计算图（按层）           切成 3 个 stage
L1─L2─L3─L4─L5─L6   ⇒   [L1 L2] [L3 L4] [L5 L6]
                          卡组0    卡组1    卡组2
                            └─send→ └─send→        段间只传边界激活
```

流水线把一个 batch 拆成 micro-batch 错峰流动，但首尾有"气泡（bubble）"：

```
时间 →
卡组0: F1 F2 F3 F4 ........ B1 B2 B3 B4
卡组1: .. F1 F2 F3 F4 .. B1 B2 B3 B4
卡组2: .... F1 F2 F3 F4 B1 B2 B3 B4
       ↑ 填充气泡        ↑ 排空气泡   （斜三角=浪费）
```

气泡占比近似：

$$\text{bubble ratio} \approx \frac{p-1}{m+p-1}$$

其中 $p$=stage 数，$m$=micro-batch 数。**自动并行要同时决定"切几个 stage、每个 stage 多少层、配几张卡、micro-batch 多大"**，让气泡 + 通信最小。Alpa 的关键贡献正是：**inter-op 用动态规划求 stage 划分，intra-op 用整数规划求每段内切分，分层求解**，把指数空间拆成两个可解子问题。

## 5. 代价模型：怎么给一个方案打分

搜索需要一个 `cost(plan) → 标量` 函数。它至少含三项：

```
cost(plan) = Σ_op  T_compute(op)              ← 算力（FLOPs / 峰值算力）
           + Σ_edge T_comm(edge, sbp_in, sbp_out)  ← 通信（重切分代价）
           + penalty( peak_memory > capacity )      ← 显存硬约束
```

- **T_compute**：算子 FLOPs ÷ 单卡有效算力（见 [[llm-algo/FLOPs]]）。
- **T_comm**：当一条边两端 SBP 不一致，要做"重切分（resharding）"，插入集合通信（见 [[ai-infra/网络/集合通信原语]]）。代价由通信量 ÷ 带宽决定。
- **显存**：参数 + 梯度 + 优化器状态 + 激活，切分后每卡只占一份；超容量直接判非法（或触发重计算/offload）。

**SBP 转换 → 通信原语对照表**（自动插入的核心规则）：

| 源 SBP | 目标 SBP | 触发的通信 | 量级 |
|---|---|---|---|
| `P` | `B` | AllReduce | 全量 |
| `P` | `S` | ReduceScatter | 全量/N |
| `S` | `B` | AllGather | 全量 |
| `S(i)`| `S(j)` | All-to-All | 全量 |
| `B` | `S` | 本地切片（Split） | 0 |
| `B` | `B` | 无 | 0 |

> 记忆法：要"消除部分和"必走 Reduce 类；要"补齐分片"必走 Gather 类；换切分维走 All-to-All；从复制变切分免费。

## 6. 搜索：在指数空间里找方案

有了代价模型，剩下就是"在所有合法 plan 里找 argmin"。常用三类方法：

```
① 整数线性规划 ILP   —— 给每个算子的每种切法一个 0/1 变量，
                        约束"相邻算子 SBP 兼容"，目标=总代价最小。
                        优点：最优；缺点：变量随图规模膨胀。  (Alpa intra-op)

② 动态规划 DP        —— 沿计算图链式结构逐段递推最优 stage 划分。
                        适合 inter-op（层是近似链状）。       (Alpa inter-op / PP)

③ 贪心 / 随机搜索    —— MCMC、模拟退火、启发式。
                        快但非最优。                          (FlexFlow)
```

**分层求解（Alpa 的范式）** 是工业界主流，因为它把 $O(K^L)$ 砍成两个小问题：

```
                 计算图
                   │
        ┌──────────┴───────────┐
   inter-op 层：DP 切 stage     （决定"分几段、放哪些卡"）
        │
   每个 stage 内
        │
   intra-op 层：ILP 选每个算子 SBP（决定"每个算子怎么切"）
        │
   再叠加 内存优化：重计算 / ZeRO 切分 / offload
```

> 共性痛点（呼应文件原注）：**模型量级很大时，搜索空间巨大，求解时间可能很长。** 工程上靠"对称性剪枝、子图复用、profiling 校准代价、缓存搜索结果"来加速。

## 数值手算：一次 AllReduce 到底传多少字节

设 Megatron MLP 的 AllReduce，隐藏维 $h$，序列长 $s$，micro-batch $b$，张量并行度 $t=4$（4 卡一组），fp16（2 字节/元素）。

```
被规约的激活张量 Y 形状 = [b, s, h]
元素数         = b·s·h
字节数 V       = b·s·h · 2 (bytes)
```

取 $b=1, s=2048, h=8192$：

```
元素数 = 1 × 2048 × 8192 = 16,777,216 ≈ 1.68e7
字节数 V = 1.68e7 × 2     = 33,554,432 B ≈ 32 MiB
```

Ring-AllReduce 的通信量公式（见 [[ai-infra/网络/集合通信原语]]）：每卡收发约 $2\cdot\frac{t-1}{t}\cdot V$。

```
每卡传输 = 2 × (4-1)/4 × 32 MiB = 2 × 0.75 × 32 = 48 MiB
```

若卡间带宽 $BW = 200\,\text{GB/s}$（NVLink 级别）：

```
通信时间 ≈ 48 MiB / 200 GB/s
        = 48 × 1.049e6 B / 200e9 B/s
        ≈ 5.03e7 / 2.0e11
        ≈ 2.5e-4 s = 0.25 ms   （单次 AllReduce，忽略延迟项）
```

**每个 Transformer 层前/反各 1 次 AllReduce**，一层约 $4$ 次（fwd MLP+Attn，bwd 同），共 $\approx 1.0$ ms 通信。这个数字就是代价模型里 `T_comm` 的一项；自动并行会拿它和"换一种切法"的代价相比，挑更小者。

**对照另一方案**：若改成 $t=8$，单次每卡传输 $=2\times\frac{7}{8}\times32=56$ MiB，通信变多但每卡算力负担减半——代价模型据此权衡，这正是"自动"的意义。

## 7. 代表系统一览

| 系统 | intra-op | inter-op | 搜索方法 | 一句话 |
|---|---|---|---|---|
| **Alpa** | SPMD 切分 | stage 划分 | ILP + DP（分层） | 自动并行的范式之作，编译到 XLA |
| **FlexFlow** | 维度切分 | 设备放置 | MCMC 随机搜索 + 模拟器 | 用"模拟执行"估代价 |
| **Tofu** | 算子内划分 | — | 递归+动态规划 | 早期 intra-op 自动切 |
| **Colossal-Auto** | 基于 PyTorch FX | + activation ckpt | ILP | 接入 Colossal-AI 生态 |
| **MindSpore** | sharding 传播 | 半自动/全自动 | 策略传播+搜索 | 框架原生算子级自动并行 |

> 论文锚点（原文件已列）：Tofu (1807.08887)、FlexFlow (1807.05358)、Alpa (2201.12023)。版本/接口细节以各官方文档为准。

设计取舍一句话总结：

```
表达力↑  →  搜索空间↑  →  求解时间↑  →  需更强剪枝/近似
（全自动好用，但"搜得动"是工程命门）
```

## 常见问题

| 问题 | 答案 |
|---|---|
| 自动并行能完全取代手工吗？ | 大体能找到接近手工最优的方案，但超大模型搜索慢，常需人给"模板/约束"缩小空间。 |
| SBP 里的 `P`（部分和）能直接用吗？ | 不能当"实张量"用，下游若需完整值必须先 AllReduce/ReduceScatter 转成 `B`/`S`。 |
| intra-op 和 inter-op 谁更省通信？ | intra-op 通信量大但延迟低（同机 NVLink）；inter-op 通信量小（只传边界激活）但有气泡。通常机内 intra、机间 inter。 |
| 为什么搜索这么慢？ | 方案数随层数指数增长；缓解靠分层求解、对称剪枝、子图复用、缓存。 |
| 和 ZeRO/重计算什么关系？ | 它们是"内存优化算子"，被纳入代价模型作为可选项一起搜（[[ai-framework/deepspeed/README]]）。 |
| 代价模型不准会怎样？ | 选出次优方案。工业系统用真机 profiling 校准带宽/算力常数。 |

## 🔗 跳转链接

- 张量并行原理（intra-op 的手工版）：[[llm-inference/大模型推理张量并行]]
- 集合通信原语（AllReduce/AllGather/All-to-All 代价）：[[ai-infra/网络/集合通信原语]]
- Megatron 3D 并行实践：[[ai-framework/megatron-lm/README]]
- DeepSpeed / ZeRO 内存优化（被纳入搜索）：[[ai-framework/deepspeed/README]]
- FLOPs 与算力估算（代价模型的计算项）：[[llm-algo/FLOPs]]
- Transformer 架构（被切的对象）：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]]
- 计算通信重叠（隐藏自动并行插入的通信）：[[llm-optimizer/计算通信重叠]]
- 训练总览：[[llm-train/README]]
- 知识地图：[[00-知识地图]]
