# Flash-Decoding

> Flash-Decoding 是把 FlashAttention 在 **decode 阶段** 沿 **KV 序列维度** 切块并行、再做一次"二次归约"的注意力计算方法，专门救活长上下文逐 token 解码时 GPU 被严重闲置的问题。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-optimizer/FlashAttention]] [[llm-inference/KV-Cache优化]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | KV 维并行 + 二次归约 |
| 1 | 地基：decode 阶段长什么样 | prefill vs decode、q 只有 1 行 |
| 2 | FlashAttention 在 decode 的并行不足 | 并行维耗尽、SM 闲置 |
| 3 | online-softmax 复习（二次归约的前提） | running max / running sum |
| 4 | Flash-Decoding 的切分 | split-KV、partial 结果 |
| 5 | 第二阶段：跨 split 的归约 | rescale 合并 m/l/O |
| 6 | 为什么对长上下文 decode 提速 | 并行度 ∝ 序列长度 |
| 7 | 与 FlashAttention 的区别 | 并行维不同、多一次 reduce |
| 8 | 数值手算 | 把 split 合并算到底 |
| 9 | 复杂度/对照表 | FLOPs/访存/并行度 |
| 10 | 常见问题 | 疑问→真相 |

---

## 0. 一句话锚点

decode 时每步只生成 **1 个** token，所以 query 只有 **1 行**，但要和 **整段 KV Cache**（可能上万 token）做注意力。FlashAttention 原本把并行度放在"query 行数 × head 数 × batch"上——可这里 query 行数=1，并行度直接塌掉，GPU 几百个 SM 大半在睡觉。**Flash-Decoding 把那条长长的 KV 序列切成若干块，让多个 SM 各算一块的"局部注意力"，最后用 online-softmax 的归约规则把局部结果正确合并成全局结果。** 它不改变数学结果，只改变"谁来算、并行切在哪一维"。

---

## 1. 地基：自回归推理的两个阶段

大模型推理分两段，注意力的形状完全不同：

```
┌─────────────── Prefill（预填充）───────────────┐
 输入 prompt: "今天 天气 很"  (3 个 token 一起进)
 Q 形状: [3, d]      K,V 形状: [3, d]
 注意力矩阵 S = Q·Kᵀ : [3, 3]   ← 一次算很多行
 → 并行度天然很高
└────────────────────────────────────────────────┘

┌─────────────── Decode（解码，逐 token）─────────┐
 已生成: "今天 天气 很"，现在求下一个词
 新 Q 形状: [1, d]   ← 只有 1 行！
 K,V 形状: [N, d]    ← N = 已有全部 token（KV Cache）
 注意力矩阵 S = Q·Kᵀ : [1, N]   ← 一行 × 超长列
 → 每生成一个 token 重复一次，N 不断变长
└────────────────────────────────────────────────┘
```

**为什么 decode 的 Q 只有 1 行？** 因为之前 token 的 K、V 都已经算过并缓存在 [[llm-inference/KV-Cache优化|KV Cache]] 里，本步只需把"当前这一个新 token"的 query 拿去和历史所有 K、V 算注意力，得到一个输出向量，再过 FFN 预测下一个 token。这就是为什么 decode 是 **memory-bound（访存受限）**：算的量很小（1×N），但要把整段 KV（N×d）从显存搬进来。

记号约定（后面通篇用）：
- $N$：KV 序列长度（上下文长度，如 4096、32768）
- $d$：每个 head 的维度（如 128）
- $q\in\mathbb{R}^{1\times d}$：当前 step 的 query（单行）
- $K,V\in\mathbb{R}^{N\times d}$：缓存的键、值

注意力定义（标准 softmax 注意力）：

$$
\text{out}=\text{softmax}\!\left(\frac{qK^\top}{\sqrt d}\right)V,\qquad qK^\top\in\mathbb{R}^{1\times N}
$$

---

## 2. FlashAttention 在 decode 的"并行不足"

先回顾：[[llm-optimizer/FlashAttention|FlashAttention]] 的核心贡献是 **不把 N×N 的注意力矩阵写回显存**，而是把 K、V 分块、在 SRAM 里用 online-softmax 流式累加。它把工作 **网格化** 成很多并行块——经典的并行维是：

$$
\text{并行块数}=\underbrace{B}_{batch}\times\underbrace{H}_{heads}\times\underbrace{\lceil M/B_r\rceil}_{query\ 行块}
$$

其中 $M$ 是 query 行数。在 **prefill**，$M$ 等于 prompt 长度（几百上千），三者相乘轻松占满 GPU 所有 SM（A100 有 108 个 SM）。

**但在 decode：$M=1$。** query 行块数 = 1。于是：

```
并行块数 = B × H × 1

举例:  batch=1, heads=32  →  只有 32 个并行块
GPU A100 有 108 个 SM
            ┌───────────────────────────────┐
SM 占用:    │■■■■■■■■■■■■■■■░░░░░░░░░░░░░░░░░░░│  32/108 在干活
            └───────────────────────────────┘
            其余 ~70% 的 SM 完全闲置！
```

更糟的是：每一个 block 内部要顺着 **整条 N 长的 KV** 串行地跑 online-softmax（一块 K/V 接一块），N=32768 时这是个又细又长的"独苗"任务。**核心矛盾**：

> FlashAttention 把并行度押在 query 维上，可 decode 的 query 维只有 1 行——并行度被它自己的设计"锁死"了，序列再长也没法靠它把空闲 SM 喂饱。

而 decode 本来就是 memory-bound，能不能跑满显存带宽，几乎完全取决于"有没有足够多的并行任务同时去搬 KV"。并行块太少 → 带宽吃不满 → 长上下文 decode 巨慢。

---

## 3. 复习 online-softmax（二次归约的数学前提）

Flash-Decoding 的"合并局部结果"靠的就是 softmax 可以 **分块增量** 计算。先把它讲透，否则第 5 节看不懂。

softmax 要先减去最大值防溢出。对一行分数 $s_1,\dots,s_N$：

$$
o=\sum_{j}\frac{e^{s_j-m}}{\ell}\,v_j,\quad m=\max_j s_j,\quad \ell=\sum_j e^{s_j-m}
$$

**关键洞察**：如果把列分成两块 A、B，各自算出局部三元组 $(m,\ell,o)$，可以无损合并。设：

- 块 A：局部最大 $m_A$，局部和 $\ell_A=\sum_{j\in A}e^{s_j-m_A}$，局部输出 $o_A=\sum_{j\in A}e^{s_j-m_A}v_j$
- 块 B：同理 $m_B,\ell_B,o_B$

合并到全局 $m=\max(m_A,m_B)$：

$$
\ell = e^{m_A-m}\ell_A + e^{m_B-m}\ell_B
$$
$$
o_{\text{合并}} = e^{m_A-m}o_A + e^{m_B-m}o_B,\qquad \text{out}=\frac{o_{\text{合并}}}{\ell}
$$

直觉：每块当初是 **按自己的局部最大值** 减的指数，现在统一到全局最大值 $m$，就给每块乘一个 **修正因子** $e^{m_{\text{块}}-m}$（一定 $\le 1$）。这一步叫 **rescale（重缩放）**。这就是把若干"局部注意力"拼回"全局注意力"的全部秘密。

```
块A: (m_A, ℓ_A, o_A) ┐
                      ├──► 取全局 m=max ──► 各乘 e^{m_块−m} ──► 相加 ──► 除以 ℓ
块B: (m_B, ℓ_B, o_B) ┘                       (修正因子≤1)
```

---

## 4. Flash-Decoding 第一阶段：沿 KV 序列切分并行

既然 query 维只有 1 行没法切，那就 **切 KV 序列维**——这正是 decode 时唯一"又长又好切"的维度。

把长度为 $N$ 的 KV 切成 $S$ 个 **split**（块），每块长 $\approx N/S$。**让每个 split 由一个独立的并行任务（线程块/SM）处理**：

```
KV Cache (序列长 N=8 演示, 切成 S=4 个 split):

K,V:  [k0 k1 | k2 k3 | k4 k5 | k6 k7]
       split0   split1  split2  split3
         │        │       │       │
单行 q ──┼────────┼───────┼───────┤   q 广播给每个 split
         ▼        ▼       ▼       ▼
       局部       局部    局部    局部     ← 4 个 SM 同时算！
     (m0,ℓ0,o0)(m1,ℓ1,o1)(m2,ℓ2,o2)(m3,ℓ3,o3)
```

每个 split **内部** 就是一次标准的 FlashAttention（online-softmax 流式），只不过它只看自己负责的那一小段 K、V，产出一个 **局部三元组** $(m_i,\ell_i,o_i)$：

$$
m_i=\max_{j\in \text{split}_i}\frac{q k_j^\top}{\sqrt d},\quad
\ell_i=\!\!\sum_{j\in\text{split}_i}\!\! e^{s_j-m_i},\quad
o_i=\!\!\sum_{j\in\text{split}_i}\!\! e^{s_j-m_i}v_j
$$

**为什么这样就能喂饱 GPU？** 原来并行块数 $=B\times H\times 1$，现在变成：

$$
\text{并行块数}=B\times H\times S
$$

多出来的 $S$ 倍直接把空闲 SM 占满。$S$ 通常按"让总块数 ≈ SM 数量的整数倍"来选（运行时根据 N 和 GPU 自适应）。

---

## 5. Flash-Decoding 第二阶段：跨 split 的二次归约

第一阶段产出 $S$ 个局部三元组，它们各自按 **自己 split 的局部最大值** 归一化过，彼此不能直接相加。需要 **第二个 kernel（归约 kernel）** 把它们合并——这就是"二次归约"（reduction），数学上就是第 3 节那条合并公式推广到 $S$ 块：

$$
m=\max_{i} m_i,\qquad
\ell=\sum_{i=1}^{S} e^{m_i-m}\,\ell_i,\qquad
o=\sum_{i=1}^{S} e^{m_i-m}\,o_i
$$
$$
\boxed{\;\text{out}=\dfrac{o}{\ell}=\dfrac{\sum_i e^{m_i-m}o_i}{\sum_i e^{m_i-m}\ell_i}\;}
$$

```
阶段1 (并行, S 个 SM)        阶段2 (归约, 轻量)
┌─────────────────────┐     ┌──────────────────────────┐
│ split0 →(m0,ℓ0,o0)  │──┐  │ m = max(m0..m_{S-1})      │
│ split1 →(m1,ℓ1,o1)  │──┤  │ 每块 ×e^{m_i−m}           │
│  ...                │  ├─►│ ℓ=Σ..  o=Σ..             │
│ split_{S-1}→(...)   │──┘  │ out = o/ℓ   (单个向量)    │
└─────────────────────┘     └──────────────────────────┘
   读整段 KV，重活           只读 S 个小三元组，几乎免费
```

第二阶段读取的数据量只有 $S\times(2+d)$ 个数（每块一个 $m_i$、一个 $\ell_i$、一个长度 $d$ 的 $o_i$），$S$ 通常几十到上百，**代价相对第一阶段可忽略**。整个过程的最终结果与"不切分直接算"在数值上 **完全等价**（除浮点舍入外）。

**为什么必须分两个 kernel？** 因为各 split 在不同 SM 上并行跑，互相不知道对方的 $m_i$；要等所有 split 都算完才能取全局 $\max$。GPU 上跨线程块的全局同步最干净的做法就是"结束一个 kernel、再起一个 kernel"。

---

## 6. 为什么 Flash-Decoding 提升长上下文 decode

把三件事串起来看就明白了：

1. **并行度从常数变成与序列长度挂钩。** FlashAttention 在 decode 的并行度是 $B\cdot H\cdot 1$（与 $N$ 无关，常数）；Flash-Decoding 是 $B\cdot H\cdot S$，而 $S\propto N$（N 越长切越多块）。**上下文越长，Flash-Decoding 能动用的并行任务越多**，正好抵消 N 变长带来的工作量。

```
延迟 vs 上下文长度 N (示意):

延迟│                    ╱ FlashAttention(decode)
    │                 ╱      ← 并行度=常数, 几乎线性涨
    │              ╱
    │           ╱
    │      ___________________ Flash-Decoding
    │  ___╱                     ← 并行度∝N, 长 N 时近乎持平
    └────────────────────────► N
       1k   8k   32k   64k
```

2. **decode 是 memory-bound，带宽=性能。** 性能瓶颈是"多快把整段 KV 从 HBM 搬进来"。只有足够多的并行块同时发起访存，才能把显存带宽吃满。Flash-Decoding 提供的就是这些并行块。

3. **沿用 FlashAttention 的省显存优势。** 第一阶段每个 split 内部仍是 online-softmax，**不物化** $1\times N$ 的注意力矩阵；只额外存 $S$ 个小三元组。所以它既要并行度又不牺牲显存。

实测意义：在 N 达到几万 token 的长上下文场景，Flash-Decoding 对单步 decode 注意力可带来数倍量级的加速（具体倍数随 N、GPU、head 数而变，**以官方实现/基准为准**），而短上下文（N 小、并行度本就够）收益不明显，甚至因多一次归约略有开销——所以实现里常按 N 是否够长来决定要不要启用。

---

## 7. 与 FlashAttention 的区别（核心对照）

它俩 **不是替代关系**：Flash-Decoding 是 FlashAttention 思想在 decode 场景的"再切一刀 + 二次归约"扩展。

```
FlashAttention(标准):
  并行: [ batch × head × Q行块 ]   ← 切 query 维
  一个 kernel, 块内沿 KV 流式 softmax, 直接出 out

Flash-Decoding:
  并行: [ batch × head × Q行块(=1) × KV-split ]  ← 多切 KV 维
  两个 kernel: ①各 split 出局部(m,ℓ,o)  ②归约合并出 out
```

| 维度 | FlashAttention | Flash-Decoding |
|------|----------------|----------------|
| 主战场 | prefill / 训练（Q 多行） | decode（Q 单行） |
| 并行切在哪 | query 行 + batch + head | **额外切 KV 序列维** |
| query 行数 | 多（M 大） | 1 |
| kernel 数 | 1 个 | 2 个（计算 + 归约） |
| 是否物化 S 矩阵 | 否（SRAM 流式） | 否（每 split 内仍流式） |
| 并行度随 N | 不增长（N 不影响 Q 行数） | **随 N 增长（split 变多）** |
| 额外存储 | 无 | $S$ 个局部三元组 $(m_i,\ell_i,o_i)$ |
| 数值结果 | 精确 softmax | 与之 **完全等价**（二次归约无损） |

一句话区别：**FlashAttention 把"不存大矩阵"做到极致；Flash-Decoding 在此之上把"并行度"从 query 维借到 KV 维**，专治 decode 单行 query 导致的并行荒。

---

## 8. 数值示例（把切分+二次归约算到底）

设一个 head，$d=2$，KV 长度 $N=4$，缩放 $\sqrt d$ 省略（设为 1）。

**输入：**
- $q=[1,\,0]$
- $K$ 四行：$k_0=[2,0],k_1=[0,2],k_2=[1,0],k_3=[0,0]$
- $V$ 四行：$v_0=[1,0],v_1=[0,1],v_2=[2,0],v_3=[0,2]$

**第一步：分数 $s_j=q\cdot k_j$**（$q=[1,0]$ 只取每行第一维）

$$
s_0=2,\quad s_1=0,\quad s_2=1,\quad s_3=0
$$

**切成 S=2 个 split：** split0 = {0,1}，split1 = {2,3}。

**split0 局部计算**（$s_0=2,s_1=0$）：
- $m_0=\max(2,0)=2$
- $e^{2-2}=1,\ e^{0-2}=e^{-2}\approx0.1353$
- $\ell_0=1+0.1353=1.1353$
- $o_0=1\cdot v_0+0.1353\cdot v_1=[1,0]+[0,0.1353]=[1,\,0.1353]$

**split1 局部计算**（$s_2=1,s_3=0$）：
- $m_1=\max(1,0)=1$
- $e^{1-1}=1,\ e^{0-1}=e^{-1}\approx0.3679$
- $\ell_1=1+0.3679=1.3679$
- $o_1=1\cdot v_2+0.3679\cdot v_3=[2,0]+[0,0.7358]=[2,\,0.7358]$

**第二步：二次归约。** 全局最大 $m=\max(m_0,m_1)=\max(2,1)=2$。修正因子：

$$
e^{m_0-m}=e^{2-2}=1,\qquad e^{m_1-m}=e^{1-2}=e^{-1}\approx0.3679
$$

合并分母与分子：

$$
\ell = 1\cdot1.1353 + 0.3679\cdot1.3679 = 1.1353+0.5032=1.6385
$$
$$
o = 1\cdot[1,0.1353] + 0.3679\cdot[2,0.7358]=[1,0.1353]+[0.7358,0.2707]=[1.7358,\,0.4060]
$$
$$
\text{out}=\frac{o}{\ell}=\frac{[1.7358,\,0.4060]}{1.6385}=[1.0594,\,0.2478]
$$

**验证（不切分直接算）：** 全局 $m=2$，
$e^{s-2}=[1,\,0.1353,\,0.3679,\,0.1353]$，和 $=1.6385$（✓与上面一致），
$o=1[1,0]+0.1353[0,1]+0.3679[2,0]+0.1353[0,2]=[1.7358,\,0.4060]$（✓一致），
$\text{out}=[1.0594,\,0.2478]$。

**结论：切分并行 + 二次归约的结果与原始 softmax 注意力逐位相等。** 修正因子 $e^{m_i-m}$ 正是让"各按局部最大值减过的指数"统一到全局基准的桥梁。

---

## 9. 复杂度 / 对照表

设 $B$ batch，$H$ heads，$N$ KV 长，$d$ head 维，$S$ split 数。

| 指标 | FlashAttention(decode) | Flash-Decoding |
|------|------------------------|----------------|
| 注意力 FLOPs | $O(B H N d)$ | $O(B H N d)$（不变，活没变多） |
| KV 访存量 | $O(B H N d)$ | $O(B H N d)$（仍要读全段 KV） |
| 并行块数 | $B H\cdot 1$ | $B H S$ |
| 额外中间存储 | 0 | $B H S(2+d)$ 个数 |
| 第二阶段开销 | — | $O(B H S d)$，$S\ll N$ 故可忽略 |
| 物化 $N$-长 score | 否 | 否 |
| 长 N 时 SM 利用率 | 低（块太少） | 高（块 $\propto N$） |

要点：Flash-Decoding **不减少总计算量和总访存量**（这俩由问题本身决定），它减少的是 **延迟**——通过把同样多的访存分摊到更多并行 SM 上、同时发起，从而吃满显存带宽。这正是 memory-bound 场景该用的杠杆。

---

## 10. 常见问题

| 疑问 | 真相 |
|------|------|
| Flash-Decoding 是新的注意力公式吗？ | 不是。结果与标准 softmax 注意力 **数值等价**，只是计算的并行划分不同。 |
| 它取代 FlashAttention 吗？ | 不。它是 FlashAttention 在 decode 的扩展；prefill/训练仍用普通 FlashAttention。 |
| 为什么 prefill 不需要它？ | prefill 的 Q 有很多行，$B H\cdot\lceil M/B_r\rceil$ 已能占满 SM，不缺并行度。 |
| split 越多越快吗？ | 不一定。$S$ 太大→每块太小、归约开销和启动开销上升；通常取到"总块数≈SM 数倍数"即可。 |
| 它能省显存吗？ | 不显著省 KV 显存；省的是"不物化 $1\times N$ score"，并额外占用极小的 $(m,\ell,o)$ 中间量。 |
| 为什么要两个 kernel？ | 各 split 在不同 SM 并行，取全局 $\max$ 需所有 split 算完；用第二个 kernel 做全局同步+归约最干净。 |
| 短上下文还划算吗？ | 收益小甚至略负（多一次归约）；实现常按 N 是否够长决定启用。 |
| 它和 PagedAttention 冲突吗？ | 正交。PagedAttention 管 KV 显存怎么分块存（见 [[llm-inference/KV-Cache优化]]），Flash-Decoding 管注意力怎么并行算，可同时使用。 |
| 二次归约会丢精度吗？ | 不会（除常规浮点舍入）；rescale 是无损的代数恒等变换，见第 8 节验证。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引与本主题定位
- [[llm-optimizer/FlashAttention]] — 前置：online-softmax 与分块流式注意力，理解本文 §3、§7 的基础
- [[llm-inference/KV-Cache优化]] — decode 为何 memory-bound、KV Cache 怎么存（PagedAttention 等），与本文正交互补
