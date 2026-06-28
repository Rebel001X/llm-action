# MoE 并行（Expert Parallelism / 专家并行）

> 一句话定位：把"一群专家 FFN"按卡切开，用 **All-to-All** 把每个 token 送到它该去的专家上算完再送回来——这是 MoE 大模型独有的第 5 种并行维度。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/moe/README]] · [[llm-algo/mlp]] · [[llm-algo/transformer/模型架构]] · [[ai-infra/网络/集合通信原语]] · [[llm-inference/大模型推理张量并行]] · [[llm-optimizer/计算通信重叠]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | EP=切专家，A2A=搬 token |
| 1 | 地基：稠密 FFN → MoE 层 | Gating / Top-k / Expert |
| 2 | 为什么要专家并行 | 参数量爆炸、显存墙 |
| 3 | EP 的数据流（All-to-All 两次） | dispatch / combine |
| 4 | 容量因子 capacity & 丢弃 token | $C$、drop、padding |
| 5 | EP 与 DP/TP/PP 怎么叠 | 5D 并行、通信组 |
| 6 | All-to-All 通信量手算 | 字节数、带宽、耗时 |
| 7 | 显存与 FLOPs 估算 | 激活参数、稀疏度 |
| 8 | 负载均衡 loss 与抖动 | aux loss、热专家 |
| 9 | 通信优化与重叠 | 分组 A2A、overlap |
| 末 | 数值手算 + 常见问题 + 跳转 | — |

## 0. 一句话锚点

- **稠密模型**：每个 token 走**全部**参数。参数量 = 计算量，强耦合。
- **MoE 模型**：每个 token 只走 **Top-k 个专家**（典型 k=1 或 2），其余专家"不激活"。于是**总参数量**可以做到万亿级，而**单 token 的计算量（激活参数）**仍很小。
- **专家并行（Expert Parallelism, EP）**：把 $E$ 个专家分散到 $N_{ep}$ 张卡上，每张卡放 $E/N_{ep}$ 个专家。token 在哪张卡上产生，但它要去的专家可能在别的卡上——所以需要一次**全员对全员的搬运**：**All-to-All**。

```
稠密 FFN：           MoE 层（E=4 专家, Top-1）：
  token ──► FFN        token ──► Gating ──► 选 1 个专家
            (全参数)              │
                                  ├─► Expert0 (卡0)
                                  ├─► Expert1 (卡1)
                                  ├─► Expert2 (卡2)
                                  └─► Expert3 (卡3)
```

> 记忆口诀：**TP 切一个矩阵，EP 切一堆专家；TP 用 All-Reduce，EP 用 All-to-All。**

## 1. 地基：从稠密 FFN 到 MoE 层

### 1.1 稠密 Transformer 的 FFN

标准 Transformer block 里的前馈层（见 [[llm-algo/mlp]]）：

$$\text{FFN}(x) = W_2 \,\sigma(W_1 x), \quad W_1 \in \mathbb{R}^{d_{ff}\times d},\ W_2 \in \mathbb{R}^{d\times d_{ff}}$$

通常 $d_{ff}=4d$。这一层的参数占整个 block 的约 2/3，是显存和算力的大头。

### 1.2 把一个 FFN 换成"一群 FFN + 一个门"

MoE 层（Mixture of Experts，见 [[llm-algo/moe/README]]）把单个 FFN 复制成 $E$ 份**专家** $\{E_1,\dots,E_E\}$，再加一个**门控网络（Gating/Router）**：

$$g(x) = \text{softmax}(W_g x) \in \mathbb{R}^{E}, \quad W_g\in\mathbb{R}^{E\times d}$$

取 $g(x)$ 中**最大的 k 个**专家（Top-k），输出为加权和：

$$y = \sum_{i \in \text{Top-k}(g(x))} g_i(x)\cdot E_i(x)$$

```
                      ┌─────────── Gating(softmax) ───────────┐
   x ─►─┬─────────────┤  g = [0.6, 0.1, 0.25, 0.05]           │
        │             └───────────────────────────────────────┘
        │                       │ Top-2 → {专家0(0.6), 专家2(0.25)}
        │       ┌──────────┐    │
        ├──────►│ Expert 0 │────┼─► ×0.6 ─┐
        │       └──────────┘    │          ├─►(+)─► y
        └──────►│ Expert 2 │────┴─► ×0.25 ─┘
                └──────────┘
   未被选中的 Expert 1 / Expert 3 这一步不计算（稀疏）
```

> 关键直觉：**总参数 ↑↑（$E$ 个专家），但每个 token 实际算的只有 k 个专家** → "大容量、低激活算力"。这正是 GShard / Switch-Transformer / Mixtral / DeepSeek-MoE 的共同思想。

## 2. 为什么需要专家并行（动机）

把 $E$ 设大（如 64、256、甚至上千），单卡放不下所有专家的权重——**显存墙**。

举例：一个专家 FFN 参数量（$d=4096,\ d_{ff}=4d$，两矩阵）：

$$P_{expert}=2\cdot d\cdot d_{ff}=2\cdot 4096\cdot 16384 \approx 1.34\times10^8 \approx 0.134\text{B}$$

- $E=8$：$\approx 1.07$B，单卡尚可。
- $E=64$：$\approx 8.6$B 参数，**仅这一层**就要 17.2 GB（fp16），叠上多层根本放不下。
- $E=256$：单层 $\approx 34$B，必须把专家**摊到多卡**。

于是引入 **EP（Expert Parallelism）**：$N_{ep}$ 张卡，每卡持有 $E/N_{ep}$ 个专家。这样每卡只存自己那份专家权重，显存随卡数线性下降。代价是：token 必须被路由到"专家所在的卡"，引入 **All-to-All 通信**。

```
EP=4 (E=8, 每卡 2 专家):
 卡0: [E0 E1]   卡1: [E2 E3]   卡2: [E4 E5]   卡3: [E6 E7]
   ▲ 本卡产生的 token 可能要去 E5(卡2) → 必须跨卡搬运
```

## 3. EP 的数据流：两次 All-to-All

EP 的核心是**两次 All-to-All**：一次把 token 发到目标专家（dispatch），算完再发回原 rank（combine）。详见 [[ai-infra/网络/集合通信原语]] 中 All-to-All 的定义。

### 3.1 单层 MoE 的 7 个步骤

```
 ① Router 打分      每个 token 在本卡算 gating，得到目标专家 id + 权重
 ② Permute/排序     按目标专家把本卡 token 分桶（让同一专家的 token 连续）
 ③ All-to-All #1    dispatch：把每个桶发给"持有该专家"的卡
 ④ Expert FFN       每卡对收到的 token 做本地专家计算（GEMM）
 ⑤ All-to-All #2    combine：把算完的结果发回 token 原来的卡
 ⑥ Un-permute       还原 token 原始顺序
 ⑦ 加权合并         乘以 gating 权重，Top-k 求和 → 输出
```

### 3.2 ASCII：dispatch 的 All-to-All

设 EP=4，每卡 6 个 token，每个 token 的目标专家 id 写在格子里：

```
 dispatch 前（每卡按目标专家分桶）：
 卡0: [E0 E0 | E1 | E2 E2 | E3 ]
 卡1: [E0    | E1 E1| E2   | E3 E3]
 卡2: [E0 E0 | E1 | E2     | E3 E3]
 卡3: [E0    | E1 E1| E2 E2| E3 ]

           All-to-All  ──►  (每卡把第 j 块发给卡 j)

 dispatch 后（每卡收齐"发给自己专家"的全部 token）：
 卡0(E0): 来自[卡0,卡1,卡2,卡3 的 E0 桶]
 卡1(E1): 来自[卡0,卡1,卡2,卡3 的 E1 桶]
 卡2(E2): 来自[...]      卡3(E3): 来自[...]
```

All-to-All 的本质：**矩阵转置式交换**——卡 $i$ 的第 $j$ 个分块，发给卡 $j$；卡 $j$ 收下后放在第 $i$ 个位置。combine（#2）就是它的逆操作。

## 4. 容量因子（Capacity）与 token 丢弃

All-to-All 在多数框架里要求**每卡发往每个专家的 token 数固定**（静态 shape，便于通信和 GEMM）。于是定义**专家容量** $C$：

$$C = \left\lceil \text{capacity\_factor} \cdot \frac{k\cdot T}{E} \right\rceil$$

其中 $T$=单卡（或全局，看实现）token 数，$k$=Top-k，$E$=专家数。$\frac{kT}{E}$ 是"理想均匀分配下每个专家应得的 token 数"。

- **容量因子 > 1**（如 1.25、2.0）：留缓冲，减少丢弃，但要 padding 浪费算力。
- **超过 $C$ 的 token 被丢弃（drop）**：该专家这一步对它不计算，靠残差连接"跳过"。
- **不足 $C$ 的用 0 padding 补齐**。

```
 capacity C=4：
 专家0 收到 6 个 token：[t1 t2 t3 t4 | t5 t6]
                        └─ 算这 4 个 ─┘ └ drop ┘（溢出丢弃）
 专家1 收到 2 个 token：[t7 t8 | 0 0]
                               └ padding ┘（补零，浪费）
```

> 权衡：容量因子越大 → 丢得少（质量好）但 padding 浪费多（算力浪费）；越小 → 省算力但丢 token 多（精度掉）。DeepSeek-MoE 等用 **dropless / 不定长 GEMM（grouped GEMM）** 来彻底避免丢弃。

## 5. EP 与 DP/TP/PP 的叠加（5D 并行）

EP 不是单独用的，而是和 [[llm-inference/大模型推理张量并行]]（TP）、数据并行（DP）、流水线并行（PP）正交组合。关键是搞清**每种并行用哪个通信组**。

```
 一个典型 64 卡布局：DP=2 × PP=2 × (TP=2 × EP=8 在同一 16 卡内)
 ┌─ DP 组0 ────────────────────────┐  ┌─ DP 组1 ──────────────┐
 │  PP stage0   PP stage1          │  │  ...复制一份...        │
 │ ┌TP×EP=16卡┐ ┌TP×EP=16卡┐       │  │                        │
 │ │ EP 切专家 │ │           │       │  │                        │
 │ │ TP 切单专 │ │           │       │  │                        │
 │ │  家的矩阵 │ │           │       │  │                        │
 │ └──────────┘ └──────────┘       │  │                        │
 └─────────────────────────────────┘  └────────────────────────┘
```

| 并行 | 切什么 | 通信原语 | 频率 |
|---|---|---|---|
| DP | batch（数据） | All-Reduce（梯度） | 每 step 一次 |
| TP | 单个权重矩阵（行/列） | All-Reduce（前/反向） | 每层多次 |
| PP | 层（stage） | P2P send/recv | 层间 |
| **EP** | **专家（整组 FFN）** | **All-to-All** | **每 MoE 层 2 次** |

- **EP 与 TP 正交**：EP 决定"哪些专家在哪张卡"，TP 再把"单个专家的 $W_1,W_2$ 矩阵"切到组内多卡。两者通信组不同（EP 用 A2A，TP 用 All-Reduce）。
- **EP 常被嵌在 DP 组内**：非 MoE 部分（attention、router）走普通 DP；MoE 部分把 DP 组重新解释成 EP 组做 All-to-All。这就是 DeepSpeed-MoE / Megatron-MoE 的做法。

## 6. All-to-All 通信量手算（核心）

这是 EP 与 TP/DP 性能差异的关键。设：

- 隐藏维 $d$，单卡 token 数 $T$，Top-k $=k$，EP 卡数 $N$，数据 2 字节（fp16/bf16）。
- 每个 token 被发往 $k$ 个专家。

### 6.1 单次 All-to-All 每卡发送/接收字节数

每卡发出去的 token 总数 $\approx k\cdot T$（每 token 复制 k 份去 k 个专家），每个 token 向量 $d$ 维：

$$V_{send} = k\cdot T\cdot d\cdot 2\ \text{bytes}$$

dispatch + combine 共两次 A2A，故**单 MoE 层每卡总通信量**：

$$V_{layer} \approx 2\cdot(k\cdot T\cdot d\cdot 2) = 4kTd\ \text{bytes}$$

### 6.2 代入数值

取 $T=2048$（单卡 token），$d=4096$，$k=2$：

$$V_{send}=2\cdot 2048\cdot 4096\cdot 2 = 3.36\times10^7\ \text{B}\approx 32\ \text{MiB}$$

单层两次 A2A：$V_{layer}\approx 64$ MiB / 卡 / 层。

### 6.3 耗时估算

设 NVLink/IB 有效带宽 $B=150$ GB/s（$=1.5\times10^{11}$ B/s）：

$$t_{A2A}=\frac{V_{layer}}{B}=\frac{6.7\times10^7}{1.5\times10^{11}}\approx 4.5\times10^{-4}\,\text{s}=0.45\ \text{ms / 层}$$

若模型有 32 个 MoE 层：$\approx 14$ ms 纯通信。和 TP 的 All-Reduce 比，A2A 是**全交换**，在跨节点（带宽更低、如 25–50 GB/s）时会成为瓶颈——这也是"**EP 尽量放节点内、TP 跨节点要谨慎**"的原因。

> 对比 TP All-Reduce 量级：Ring All-Reduce 每卡约传 $2\frac{N-1}{N}\cdot M$（$M$=张量字节），与 EP 的 A2A 量级相近，但 **A2A 是稠密的全连接流量**，对网络拓扑（fat-tree/rail）更敏感。参见 [[ai-infra/网络/集合通信原语]]。

## 7. 显存与 FLOPs 估算

### 7.1 显存：EP 如何降权重显存

总专家参数 $P_E = E\cdot P_{expert}$。EP=N 时单卡专家权重：

$$P_{per\,card}=\frac{E}{N}\cdot P_{expert}$$

例：$E=64$，$P_{expert}=0.134$B，fp16（2B）：

- 不分（EP=1）：$64\cdot0.134\cdot2\ \text{B/param}=17.2$ GB（单层！放不下）
- EP=8：$8\cdot0.134\cdot2=2.15$ GB / 卡 / 层 ✅
- EP=64：$1\cdot0.134\cdot2=0.27$ GB / 卡 / 层 ✅✅

显存随 EP 卡数**线性下降**——这就是 EP 的核心收益。

### 7.2 FLOPs：激活参数 vs 总参数

MoE 的精妙在于**前向 FLOPs 只与"激活参数"有关，与总参数无关**：

$$\text{激活参数} = P_{\text{非MoE}} + k\cdot P_{expert}$$

单 token 单 MoE 层前向 FLOPs（仅专家部分，$2\times$ 因乘加）：

$$\text{FLOPs}_{expert}= 2\cdot k\cdot (2\,d\,d_{ff}) = 2\cdot 2\cdot 2\cdot 4096\cdot16384\approx 1.07\times10^9$$

注意：**和 $E$ 无关**——专家从 8 个加到 256 个，单 token 算力不变（仍只算 k 个）！这就是 MoE"**参数涨、算力不涨**"的本质。详见 [[llm-algo/FLOPs]]。

```
 总参数 ↑↑↑ (E=256)       激活算力 ── 恒定 (只算 k=2)
   ████████████████          ██
 模型"知道"得更多            但每 token 花的算力不变
```

## 8. 负载均衡：热专家与辅助 loss

理想是每个专家收到 $\frac{kT}{E}$ 个 token，现实中 Router 会"偏心"，造成**热专家（hot expert）**——某些专家被挤爆（大量 drop），某些专家闲置（padding 浪费 + 没学好）。

### 8.1 负载均衡辅助损失（aux loss）

GShard/Switch 引入辅助损失（让分配更均匀）：

$$\mathcal{L}_{aux}=\alpha\cdot E\cdot \sum_{i=1}^{E} f_i\cdot P_i$$

- $f_i$：分到专家 $i$ 的 token 比例（实际 dispatch 占比）。
- $P_i$：专家 $i$ 的平均门控概率（softmax 平均）。
- $f_i\cdot P_i$ 越不均匀越大 → 惩罚倾斜，$\alpha$ 是权重系数。

### 8.2 系统级影响

```
 不均衡时：             均衡时：
 E0 ████████ drop!      E0 ████
 E1 █                   E1 ████
 E2 ██                  E2 ████
 E3 (空) padding 浪费    E3 ████
 → A2A 木桶效应：最慢的卡决定整体耗时
```

由于 All-to-All 是**同步**的，**最慢的专家卡拖慢所有卡**（木桶效应）。所以负载均衡不只是精度问题，更是**吞吐问题**。工程上还有 **专家容量动态调整、expert-choice routing（专家选 token 而非 token 选专家）、DeepSeek 的无 aux-loss 偏置均衡** 等方案。

## 9. 通信优化与计算/通信重叠

A2A 是 EP 的瓶颈，优化方向（细节见 [[llm-optimizer/计算通信重叠]]）：

1. **层次化 A2A**：先节点内 A2A（NVLink 快），再节点间 A2A（IB 慢），减少跨节点流量。
2. **计算-通信重叠**：把 dispatch A2A 与上一专家的 GEMM 重叠；DeepSeek 用双 micro-batch 流水把 A2A 藏到计算后面。
3. **Grouped GEMM / dropless**：用变长分组矩阵乘避免 padding 与丢弃。
4. **量化通信**：A2A 传 fp8/int8 token（见 [[llm-compression/quantization/fp8]]），把字节数减半。
5. **Token 去重**：Top-k 时同一 token 去多个专家，可只发一份再本地复制。

```
 朴素（串行）：  [A2A dispatch]→[Expert GEMM]→[A2A combine]   慢
 重叠（流水）：  micro-batch A：  [GEMM]──────
                 micro-batch B：       [A2A]──[GEMM]
                 ↑ A 的计算盖住 B 的通信，时间线压缩
```

---

## 数值手算：一个完整 MoE 层的资源账

**配置**：$d=4096$，$d_{ff}=16384$，$E=64$ 专家，Top-k $k=2$，EP=8（每卡 8 专家），单卡 $T=2048$ token，bf16（2 字节），带宽 $B=150$ GB/s。

**① 权重显存（单卡单层）**

$$P_{expert}=2\cdot4096\cdot16384\approx1.34\times10^8,\quad \frac{E}{N}=8$$
$$\text{显存}=8\cdot1.34\times10^8\cdot2\ \text{B}\approx 2.15\ \text{GB}$$

**② All-to-All 通信（单卡单层，两次）**

单次 A2A 每卡发送 $k\,T\,d\cdot2\,\text{B}=2\cdot2048\cdot4096\cdot2\approx3.36\times10^7$ B；dispatch+combine 两次：
$$V_{layer}=2\times3.36\times10^7= 6.7\times10^7\ \text{B}\approx 64\ \text{MiB}$$
$$t_{A2A}=\frac{6.7\times10^7}{1.5\times10^{11}}\approx 0.45\ \text{ms / 层}$$

**③ 专家计算 FLOPs（单卡单层）**

单卡收到约 $kT=4096$ 个 token（均衡假设），每 token 过 1 个专家 FFN：
$$\text{FLOPs}=kT\cdot 2\cdot(2\,d\,d_{ff})=4096\cdot2\cdot2\cdot4096\cdot16384\approx 1.1\times10^{12}=1.1\ \text{TFLOP}$$
A100 bf16 峰值 $\approx312$ TFLOPS，理想 $\approx3.5$ ms（实际算 MFU 打折）。

**④ 通信占比**：$0.45 / (0.45+3.5)\approx 11\%$ 单层时间花在 A2A——**节点内尚可，跨节点（带宽掉到 1/3）会逼近 30%+**，这就是 EP 拓扑要 rail-optimized 的根因。

> 结论：EP 用"线性下降的显存 + 恒定的激活算力"换来"两次全交换 A2A 的通信代价"。是否划算取决于**专家是否放得进节点内**和**负载是否均衡**。

## 常见问题

| 问题 | 答案 |
|---|---|
| EP 和 TP 的根本区别？ | TP 切**一个**矩阵、用 All-Reduce；EP 切**一群专家**、用 All-to-All。可正交叠加。 |
| 为什么用 All-to-All 不用 All-Reduce？ | token 要去**不同**专家（不同卡），是"点对点的全交换"，不是"求和"。 |
| 容量因子设多大？ | 训练常 1.0–1.25，推理可更小；越大丢得少但 padding 浪费多。dropless 方案可省掉它。具体默认值以官方框架为准。 |
| 总参数 1T 但只激活 37B 是什么意思？ | 总参数=全部专家权重；激活参数=单 token 实际过的 k 个专家+共享部分。FLOPs 只跟激活参数走。 |
| 热专家为什么拖慢全卡？ | A2A 同步，最慢专家卡决定 barrier 时间（木桶效应）。需负载均衡。 |
| EP 跨节点行不行？ | 行但贵：跨节点 A2A 带宽低，通信占比飙升。优先把 EP 放节点内 NVLink。 |
| 推理时 MoE 怎么并行？ | 同样 EP+A2A；batch 小时容量利用率低，常配合 expert 放置/合并优化。参考 [[llm-inference/README]]。 |
| Top-1 vs Top-2？ | Top-1（Switch）最省算力、A2A 量减半；Top-2（GShard/Mixtral）质量更稳。 |

## 🔗 跳转链接

- 上层导航：[[00-知识地图]]
- MoE 算法本体：[[llm-algo/moe/README]] · [[llm-algo/mlp]] · [[llm-algo/transformer/模型架构]]
- 并行邻居：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]]
- 通信优化：[[llm-optimizer/计算通信重叠]] · [[llm-optimizer/kv-cache]]
- 量化通信：[[llm-compression/quantization/fp8]] · [[llm-compression/quantization/量化基础]]
- 框架实现：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 算力基础：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]] · [[llm-algo/FLOPs]]
- 训练全局：[[llm-train/README]] · [[llm-inference/README]]
