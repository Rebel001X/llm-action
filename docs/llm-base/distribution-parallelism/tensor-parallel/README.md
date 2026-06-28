# 张量并行（Tensor Parallelism, TP）

> 把单个算子（GEMM / Attention）的权重矩阵**沿某一维切开**分到多张卡上，让一层的计算被多卡协同完成；代价是**层内**要插入集合通信。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/大模型推理张量并行]] · [[ai-framework/megatron-lm/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-optimizer/计算通信重叠]]

## 阅读地图

| 节 | 你将学到 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | 切权重 vs 切数据 |
| 1 | 地基：为什么要切、切谁、GEMM 回顾 | 显存墙、$Y=XA$ |
| 2 | 列并行（Column Parallel） | 切列、前向 all-reduce 在哪一侧 |
| 3 | 行并行（Row Parallel） | 切行、前向 all-reduce |
| 4 | Megatron MLP：列+行 = 1 次 all-reduce | $f$/$g$ 共轭算子 |
| 5 | Megatron Self-Attention：按 head 切 | QKV 列并行、O 行并行 |
| 6 | 前向/反向各 2 次 all-reduce 的来历 | $f,g$ 与梯度 |
| 7 | Embedding / LM Head / LayerNorm / Dropout | 词表切分、不切的部分 |
| 8 | 通信量与带宽：为什么 TP 不跨节点 | NVLink vs IB |
| 9 | Sequence Parallel：把剩下的也切了 | all-reduce → reduce-scatter+all-gather |
| 数值 | 7B/70B 逐数手算显存与通信 | bytes / s |
| FAQ | 常见坑 | TP×PP×DP |

## 0. 一句话锚点

- **数据并行（DP）**：每张卡都有**完整模型**，切的是 batch。卡多 → 吞吐高，但**装不下大模型**。
- **张量并行（TP）**：每张卡只有**模型的一部分**（半个权重矩阵），切的是**算子内部的张量维度**。多卡**合起来**才算完一层。
- 一句话：**DP 切样本，TP 切权重的“宽/高”，PP（流水线并行）切“层”。** 三者正交，可叠加成 3D 并行。

```
       一层 Linear: Y = X · A   (A 是 [k, n] 权重)
  ┌────────────── DP ──────────────┐   每卡整块 A，喂不同 batch
  ┌────────────── TP ──────────────┐   每卡半块 A，喂同一份 X
  ┌────────────── PP ──────────────┐   不同卡放不同的“层”
```

## 1. 地基：显存墙 + GEMM 回顾

### 1.1 为什么必须切权重

一个 $P$ 参数的模型，**仅权重**（fp16，2 byte/参数）就要 $2P$ 字节。训练时还要算上**梯度 + 优化器状态**（Adam：fp32 权重副本 + 一阶动量 + 二阶动量），经典系数 **16 byte/参数**：

$$\text{训练静态显存} \approx 16 \times P \ \text{bytes}$$

- 7B：$16\times 7\times10^9 = 112\ \text{GB}$ —— 单张 80GB A100 **装不下**。
- 70B：$1120\ \text{GB}$ —— 必须切。

切的方式：DP 不能减少**单卡权重**（每卡都有整份），所以**单卡显存墙**只能靠 TP / PP / ZeRO 来破。TP 的卖点是：**把一层的权重和激活一起摊薄**，且对计算图侵入小。

### 1.2 GEMM 是一切的原子

Transformer 里 99% 的算力花在矩阵乘法（GEMM）上。一个全连接层就是：

$$Y = X A,\qquad X\in\mathbb{R}^{b\times k},\ A\in\mathbb{R}^{k\times n},\ Y\in\mathbb{R}^{b\times n}$$

矩阵乘有两个天然可切的维度：**A 的列 $n$**、**A 的行 $k$**。TP 的全部魔法就是“切哪一维、切完怎么拼”。详见 [[llm-algo/mlp]]。

```
            A  [k × n]
        ┌───────┬───────┐
   k 行 │  A1   │  A2    │  ← 沿“列 n”切 = 列并行
        └───────┴───────┘
        ─────────────────
   切“行 k”：A1 在上、A2 在下 = 行并行
```

## 2. 列并行（Column Parallel Linear）

把 $A$ 沿**列**切成 $[A_1, A_2]$（每张卡拿一半列）。输入 $X$ **完整复制**到每张卡：

$$Y_1 = X A_1,\quad Y_2 = X A_2,\quad Y = [\,Y_1,\ Y_2\,]$$

- **前向**：两卡各算各的，**不需要通信**就能各自拿到 $Y_i$（输出在“列方向”被天然切开）。若下游需要完整 $Y$ 才 all-gather。
- **反向**：要算 $\dot X = \dot Y A^\top = \dot Y_1 A_1^\top + \dot Y_2 A_2^\top$，两卡的贡献必须相加 → **反向需要 all-reduce**。

```
         X (完整, 复制)                  X (完整, 复制)
            │                               │
        ┌───┴───┐                       ┌───┴───┐
   GPU0 │ X·A1  │ = Y1            GPU1   │ X·A2  │ = Y2
        └───────┘                       └───────┘
   前向: 无通信, 输出沿列切开 [Y1 | Y2]
   反向: dX = dY1·A1ᵀ + dY2·A2ᵀ  → all-reduce
```

记住口诀：**列并行，前向不通信、反向 all-reduce。** 通信算子记为 $f$（前向恒等、反向 all-reduce）。

## 3. 行并行（Row Parallel Linear）

把 $A$ 沿**行**切成 $\begin{bmatrix}A_1\\A_2\end{bmatrix}$。这时输入 $X$ 必须也沿**列**切成 $[X_1, X_2]$（否则维度对不上）：

$$Y = X A = [X_1, X_2]\begin{bmatrix}A_1\\A_2\end{bmatrix} = X_1 A_1 + X_2 A_2$$

- **前向**：每卡算 $X_i A_i$ 得到一个**部分和**（partial sum），形状已是完整 $[b\times n]$，但**数值不完整**。必须 all-reduce 把两卡相加 → **前向需要 all-reduce**。
- **反向**：$\dot X_i = \dot Y A_i^\top$，$\dot Y$ 本就完整（每卡都有），各算各的，**反向不需要通信**。

```
   X1 (切片)        X2 (切片)
      │                │
  ┌───┴───┐        ┌───┴───┐
  │ X1·A1 │        │ X2·A2 │   ← 各自是 [b×n] 的“部分和”
  └───┬───┘        └───┬───┘
      └──── all-reduce(相加) ────→  Y = X1A1 + X2A2
   前向: all-reduce ; 反向: 无通信
```

口诀：**行并行，前向 all-reduce、反向不通信。** 通信算子记为 $g$（前向 all-reduce、反向恒等）。$f$ 与 $g$ 是一对**共轭**算子。

## 4. Megatron MLP：列并行接行并行 = 每层 1 次 all-reduce

Transformer 的 FFN/MLP 是两层 GEMM 夹一个非线性（GELU）：

$$Z = \text{Dropout}\big(\text{GELU}(X A)\,B\big)$$

Megatron 的**精妙之处**：第一层用**列并行**，第二层用**行并行**，这样**中间结果天然对齐、整段不需要中间通信**，只在**最后**做一次 all-reduce。

```
   X(完整)──[f]──┬─ GPU0: GELU(X·A1) ─┐
                 │                     ├─ 各算 (·)·Bi → 部分和
                 └─ GPU1: GELU(X·A2) ─┘
                          │
   A 列并行 (切 4h 维)     B 行并行 (切 4h 维)
                          │
                       [g] all-reduce → Z(完整)
```

为什么对齐？列并行让 $\text{GELU}(XA_i)$ 沿“隐藏维 $4h$”被切成两半；行并行的 $B$ 恰好也沿同一维 $4h$ 切成两半，于是 $\text{GELU}(XA_i)\cdot B_i$ 直接是部分和。**关键**：GELU 是逐元素非线性，$\text{GELU}([Y_1,Y_2]) = [\text{GELU}(Y_1),\text{GELU}(Y_2)]$，切开算不影响结果——若中间放了 all-gather 再 GELU 就白白多一次通信。

- **前向**：$f$（恒等）+ $g$（all-reduce）→ **MLP 前向 1 次 all-reduce**。
- **反向**：$g$（恒等）+ $f$（all-reduce）→ **MLP 反向 1 次 all-reduce**。

## 5. Megatron Self-Attention：天然按 head 切

多头注意力本来就是 $h$ 个**独立的头**并排算，是 TP 的“天选”结构。设 $a$ 个注意力头、TP 度 $t$，则**每卡管 $a/t$ 个头**。

- $W_Q, W_K, W_V$：**列并行**（按 head 把 QKV 投影矩阵切开），每卡独立算自己负责的那几个头的 $Q,K,V$ 与 $\text{softmax}(QK^\top/\sqrt{d})V$。**头之间无依赖，前向不通信。**
- 输出投影 $W_O$：**行并行**，把各卡的 head 输出拼接的等价计算变成部分和 → 最后 **1 次 all-reduce**。

```
        X(完整)
   ┌──── [f] ────┐
 GPU0 head 0..1   GPU1 head 2..3   (各自做 QKVᵀ → softmax → ·V)
   │                  │
   └── Wo 行并行 ──[g] all-reduce ──→ Attn 输出(完整)
```

- **前向**：1 次 all-reduce；**反向**：1 次 all-reduce。
- 与 MLP 合起来：**每个 Transformer Block 前向 2 次 all-reduce（Attn 1 + MLP 1），反向 2 次。** 这就是开篇“每层有两个 all reduce”的来历。结构细节见 [[llm-algo/transformer/模型架构]]，RoPE 等位置编码不改变切分逻辑 [[llm-algo/旋转编码RoPE]]。

## 6. 前向/反向各 2 次 all-reduce：把 $f$、$g$ 数清楚

| 子层 | 前向 $f$ | 中间 | 后向 $g$ | 前向通信 | 反向通信 |
|------|----------|------|----------|----------|----------|
| Attention | 列并行(QKV) | 各 head 独立 | 行并行($W_O$) | 1×AR | 1×AR |
| MLP | 列并行($A$) | GELU 逐元素 | 行并行($B$) | 1×AR | 1×AR |
| **每 Block** | | | | **2×AR** | **2×AR** |

$L$ 层模型，**一次前向 $2L$ 次 all-reduce，一次反向 $2L$ 次**。这是 TP 最重的开销，也是它**只能跑在高带宽域**（单机 NVLink）的根因。延伸阅读 [[ai-infra/网络/集合通信原语]]。

## 7. 容易被忽略的部件

- **Embedding / LM Head**：词表 $V$ 极大（如 15 万），按**词表维度**切（vocab parallel）。前向用 all-reduce/all-gather 拼回 logits；输入/输出 embedding 常**共享权重**，需保证两端切法一致。
- **LayerNorm / RMSNorm**：参数极少（每特征 1~2 个标量），**不切**，每卡保留完整副本，对完整隐藏向量做归一化（在基础 TP 中隐藏维是完整的）。
- **Dropout**：需保证各卡随机种子策略一致，否则切分后 mask 不匹配。
- **残差连接**：在完整隐藏维上做，TP 不改变它。

```
  [完整 h] → LN(不切) → Attn(TP) → +残差 → LN(不切) → MLP(TP) → +残差 → [完整 h]
```

## 8. 通信量与带宽：为什么 TP 不跨节点

all-reduce 在 $t$ 张卡、消息大小 $M$（字节）时，环形算法的**单卡收发量** $\approx 2M\cdot\frac{t-1}{t}\approx 2M$（$t$ 较大时）。

每个 Block 前向 2 次、反向 2 次，每次 all-reduce 的 $M$ 正比于一份激活：$M = b\cdot s\cdot h\cdot 2\ \text{bytes}$（fp16，$b$ batch，$s$ 序列长，$h$ 隐藏维）。

- 这些通信**串在关键路径上**（必须等通信完才能算下一步），且 **TP 计算与通信难以重叠**（all-reduce 是同步点）。
- NVLink（A100：单卡双向 ~600 GB/s）≫ 跨节点 InfiniBand（~25–50 GB/s/卡量级）。把 $2L$ 次同步 all-reduce 放到 IB 上，通信会吃掉绝大部分时间。

> 结论：**TP 度数一般 ≤ 单机卡数（8）**，跨机用 PP/DP。重叠技巧见 [[llm-optimizer/计算通信重叠]]。

```
  单机内: GPU0═NVLink═GPU1 ... 600GB/s  → TP 放这里
  机间:   Node0 ──IB── Node1   50GB/s   → PP / DP 放这里
```

## 9. 序列并行（Sequence Parallelism, SP）——把“没切的”也切了

基础 TP 里，LayerNorm / Dropout / 残差作用在**完整隐藏维**上，每卡都存了完整激活，**激活显存没省到这部分**。SP 进一步在**序列维 $s$** 上切这些区域：

- TP 区（GEMM）需要完整序列 → 用 **all-gather**；
- SP 区（LN/Dropout）按序列切 → 用 **reduce-scatter**。
- 数学等价：**1 次 all-reduce = 1 次 reduce-scatter + 1 次 all-gather**，所以**总通信量不变**，但**激活显存按 $1/t$ 下降**。这是 Megatron-LM 训练长序列的标配，常与 [[llm-optimizer/FlashAttention]] 协同省显存。

```
  基础TP:  LN(完整,每卡都存) ─AR─ GEMM(完整)        激活全份
  SP:      LN(按s切, 1/t)  ─AG─ GEMM ─RS─ LN(1/t)   激活 1/t
           (all-reduce 拆成 reduce-scatter + all-gather)
```

## 数值手算

**例 1：列并行权重切分（FFN 第一层）**
GPT 风格，$h=4096$，FFN 放大 4 倍 → $A\in\mathbb{R}^{4096\times16384}$。TP=4：
- 每卡列数 $16384/4 = 4096$，本卡权重 $4096\times4096 = 16.8\text{M}$ 参数 ≈ $33.6$ MB（fp16）。整块是 $134$ MB → **单卡省 4×**。

**例 2：每 Block 一次 all-reduce 的消息大小**
$b=8,\ s=2048,\ h=4096$，fp16：
$$M = 8\times2048\times4096\times2 = 134{,}217{,}728\ \text{bytes} \approx 128\ \text{MB}$$
单卡 all-reduce 收发 $\approx 2M = 256$ MB。每 Block 前向 2 次 → 512 MB；80 层前向 ≈ **40 GB** 跨卡流量。在 NVLink(600GB/s) 上 ≈ $40/600 \approx 67$ ms；在 IB(50GB/s) 上 ≈ **0.8 s** —— 直观看出为何 TP 不上 IB。

**例 3：70B 模型用 TP 破显存墙（仅推理权重）**
70B，fp16 权重 $= 2\times70\times10^9 = 140$ GB。单张 80GB 卡装不下。
- TP=2：每卡 $70$ GB ✅ 勉强（还要 KV cache）。
- TP=4：每卡 $35$ GB ✅ 舒适。推理侧 KV cache 与 TP 的交互见 [[llm-inference/KV-Cache优化]] / [[llm-optimizer/kv-cache]]。

**例 4：训练侧组合（3D 并行）**
70B 训练静态显存 $\approx 16\times70\text{e}9 = 1120$ GB。单机 8×80GB = 640 GB 不够。
- TP=8（机内）每卡承担 $1120/8 = 140$ GB ❌ 仍超 80GB；
- 再叠 ZeRO/PP 跨机分摊：如 TP=8 × PP=4 × DP=… 把每卡降到 < 80 GB。说明 **TP 单打独斗有上限，必须与 PP/DP/ZeRO 组合**。框架实现见 [[ai-framework/megatron-lm/README]] 与 [[ai-framework/deepspeed/README]]。

## 常见问题

| 问题 | 解答 |
|------|------|
| TP 和 DP 区别？ | DP 每卡整模型、切 batch、梯度 all-reduce 在**反向末尾**1 次；TP 每卡半个权重、切张量、**层内**多次 all-reduce。 |
| 为什么列并行接行并行？ | 让中间激活天然对齐（GELU 逐元素 + 同维切分），全段只需 1 次 all-reduce，省掉中间 all-gather。 |
| TP 度能任意大吗？ | 不能。前/反向各 $2L$ 次同步 all-reduce 在关键路径上，受带宽限制，实践上 TP ≤ 单机卡数（多为 8）。 |
| 头数不能整除 TP？ | 要求 `num_heads % tp == 0`；否则无法按 head 均分 QKV，需 padding 或换 TP 度。 |
| LayerNorm 也切吗？ | 基础 TP 不切（参数极少、需完整隐藏维）；SP 才按序列维切其激活。 |
| SP 省了什么？ | 省**激活显存**（LN/Dropout/残差区降到 $1/t$）；总通信量不变（AR=RS+AG）。 |
| TP 能跨节点吗？ | 技术上可以，但 IB 带宽远低于 NVLink，$2L$ 次同步通信会成瓶颈，强烈不建议。 |
| 推理也用 TP 吗？ | 用。大模型单卡装不下 / 想降延迟时按 TP 切，KV cache 也随之按 head 切。见 [[llm-inference/大模型推理张量并行]]。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航
- [[llm-inference/大模型推理张量并行]] — 推理侧 TP（B07 重点）
- [[llm-inference/README]] / [[llm-inference/KV-Cache优化]] — 推理与 KV cache
- [[ai-framework/megatron-lm/README]] — Megatron-LM 实现
- [[ai-framework/deepspeed/README]] — ZeRO 与 TP 组合
- [[ai-infra/网络/集合通信原语]] — all-reduce / reduce-scatter / all-gather
- [[llm-optimizer/计算通信重叠]] — 通信与计算重叠
- [[llm-optimizer/FlashAttention]] / [[llm-optimizer/kv-cache]] — 注意力与缓存优化
- [[llm-algo/transformer/模型架构]] / [[llm-algo/mlp]] / [[llm-algo/FLOPs]] — 结构、MLP 与算力估算
- [[llm-algo/moe/README]] — MoE 与专家并行（TP 的近亲）
- [[ai-infra/算力/GPU工作原理]] / [[ai-infra/ai-hardware/CUDA]] — 硬件底座
