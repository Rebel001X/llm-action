# 序列并行（Sequence Parallelism, SP）

> 把长序列“按时间维度”切给多张卡，专治激活值显存爆炸与超长上下文训练。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[B07:llm-inference/大模型推理张量并行]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]

## 阅读地图

| 你想知道的 | 去第几节 | 一句话答案 |
| --- | --- | --- |
| SP 到底切什么维度 | §0 §1 | 切 **序列长度 $S$**，不是切隐藏维 $H$（那是 TP） |
| 为什么需要它 | §1 | 激活显存正比于 $S$，长序列下激活才是显存第一杀手 |
| Megatron-SP 怎么和 TP 配 | §2 | 在 LayerNorm/Dropout 处切 $S$，用 all-gather/reduce-scatter 衔接 TP |
| Ring-Attention 怎么切注意力 | §3 | 把 KV 块在环上轮转，**O(1) 额外显存**算全局注意力 |
| DeepSpeed-Ulysses 怎么做 | §4 | 用 all-to-all 在“切序列”和“切头”之间来回转 |
| 通信量/显存怎么算 | §5 §6 | 给出逐项手算 |
| 和 TP/PP/CP 啥关系 | §7 | 对照表 + 选型 |
| 最小实现思路 | §8 | 伪代码骨架 |

---

## 0. 一句话锚点

> **序列并行 = 沿 $S$（token 维）把一条样本切成 $P$ 段，每张卡只持有 $S/P$ 个 token 的激活；在“需要看到全序列”的算子（主要是 Self-Attention）处，用集合通信把信息补齐。**

一句话区分三兄弟（设隐藏维 $H$、序列 $S$、batch $B$、head 数 $a$）：

```
DP（数据并行）：切 B  —— 每卡一份完整模型，跑不同样本，梯度 all-reduce
TP（张量并行）：切 H  —— 每卡持有权重矩阵的一片（列/行切分）
SP（序列并行）：切 S  —— 每卡持有序列的一段，激活显存按 S/P 下降
```

为什么要单独发明 SP？因为 **DP 救不了单条样本太长，TP 切 $H$ 也救不了与 $S$ 成正比的那部分激活（尤其是注意力分数矩阵 $S\times S$）**。SP 是唯一直接砍 $S$ 维显存的并行方式。

---

## 1. 地基：为什么长序列会“激活爆显存”

### 1.1 显存账：训练显存 = 权重 + 优化器 + 梯度 + **激活**

前三项只和参数量 $N$、数据类型有关，**与序列长度 $S$ 无关**。真正随 $S$ 暴涨的是**激活（activation）**——前向算出、反向要用、必须缓存的中间张量。

一层 Transformer 的激活，量级上（省略常数）：

$$
\text{Act}_{\text{layer}} \;\approx\; \underbrace{c_1\, B S H}_{\text{线性层/LayerNorm 等}} \;+\; \underbrace{c_2\, B\, a\, S^2}_{\text{注意力分数 } QK^\top}
$$

- 第一项 $\propto S$（线性）：MLP、QKV 投影、残差等中间结果。
- 第二项 $\propto S^2$（平方）：注意力打分矩阵 $\text{scores}\in\mathbb{R}^{a\times S\times S}$。**这是长序列的真正炸点。**

> ⚠️ FlashAttention 通过分块（tiling）把这块 $S^2$ 的**显存**降到了线性（不再实体化整个 $S\times S$），但**计算量仍是 $O(S^2)$**，且第一项的 $\propto BSH$ 激活依然存在。所以即便用了 Flash，长序列下激活仍是显存大头——SP 仍然必要。详见 [[llm-optimizer/FlashAttention]]。

### 1.2 数值直觉：8K vs 128K 上下文

设 $H=8192$、层数 $L=80$、$B=1$、bf16（2 字节），只估**线性那一项**（每层约 $\sim 34\,BSH$ 字节是常见经验系数，这里取保守 $\approx 20\,BSH$ 仅作量级演示）：

```
单层线性激活 ≈ 20 · B · S · H 字节
S = 8K   : 20 · 1 · 8192 · 8192  ≈ 1.34 GB / 层
S = 128K : 20 · 1 · 131072 · 8192 ≈ 21.5 GB / 层  ← 单层就吃满一张卡
全模型 ×80 层：8K→约 107GB，128K→约 1.7TB（不切根本放不下）
```

> 数字仅作量级演示，精确系数随实现（是否重计算/Flash/融合）变化很大，**以实测为准**。结论是稳的：**$S$ 一拉长，激活线性甚至平方放大，必须沿 $S$ 切。**

### 1.3 SP 的核心洞察

> Transformer 里**绝大多数算子是“逐 token 独立”的**：LayerNorm、Dropout、残差加、MLP（两个线性层 + 激活函数），它们对每个 token 的计算互不依赖。
> **唯一需要“跨 token 交互”的是 Self-Attention**（每个 query 要看所有 key/value）。

所以策略很清晰：

```
逐 token 独立的算子  → 直接按 S 切，各算各的，零通信
Self-Attention       → 需要全序列，靠集合通信补齐
```

不同 SP 方案，本质就是**“在注意力处如何补齐全序列信息”的不同选择**。

---

## 2. 流派一：Megatron 序列并行（与张量并行协同）

这是论文 *Reducing Activation Recomputation in Large Transformer Models*（Korthikanti et al., 2022，约 NVIDIA）提出的工程方案，**SP 不单独存在，而是 TP 的“补丁”**。

### 2.1 问题：纯 TP 没切掉的那块激活

Megatron-TP 切的是权重的 $H$ 维：注意力按 head 切、MLP 按列/行切。但 **LayerNorm、Dropout、残差** 这些算子 TP 没切——它们在每张 TP 卡上**冗余地持有完整的 $(B,S,H)$ 激活**。长序列下，这块冗余激活很可观。

### 2.2 解法：在“TP 不切的区域”改成按 $S$ 切

把 Transformer 一层切成两类区域，**复用同一组 TP 卡，沿不同维度切**：

```
        ┌──────────────── 一个 Transformer Layer（TP 度 = 2 为例）─────────────────┐
        │                                                                          │
 输入   │  [LayerNorm]      g →   [Self-Attn / MLP]    → g̅    [Dropout+残差]        │
 (B,S,H)│   SP 区:切 S            TP 区:切 H                   SP 区:切 S            │
        │  每卡持 (B,S/2,H)      每卡持权重的一半            每卡持 (B,S/2,H)        │
        └──────────────────────────────────────────────────────────────────────────┘

 衔接算子：
   g  (进入 TP 区)  = all-gather    沿 S 把 (B,S/2,H) 拼回 (B,S,H)
   g̅ (离开 TP 区)  = reduce-scatter 把 TP 的部分和求和并沿 S 切回 (B,S/2,H)
```

### 2.3 关键技巧：all-reduce 拆成 all-gather + reduce-scatter

纯 TP 在进/出 TP 区各需一次 **all-reduce**。Megatron-SP 把每次 all-reduce **等价拆**成：

$$
\text{all-reduce} \;\equiv\; \text{reduce-scatter} \;+\; \text{all-gather}
$$

- 进 TP 区：`all-gather`（S 切 → S 全）
- 出 TP 区：`reduce-scatter`（部分和求和 + 切回 S）

**通信量不变**（环形算法下 all-reduce ≈ 2× 单向数据量 = all-gather + reduce-scatter 之和），但**SP 区的激活从冗余的 $(B,S,H)$ 降到了 $(B,S/P,H)$**。这就是“几乎免费”地省下了 LayerNorm/Dropout 区域的激活显存。

> 直觉口诀：**TP 区切 $H$、SP 区切 $S$，用 all-gather/reduce-scatter 在两区之间“换装”。通信总量约等于原来的 all-reduce，激活显存却下降。**

---

## 3. 流派二：Ring Attention（环形注意力，超长上下文的主力）

当 TP 度受限（通常 ≤ 单机 8 卡，受 NVLink 约束），而你想做 128K、1M token 时，需要能**跨更多卡、甚至跨机**扩展的注意力切分。Ring Attention（Liu et al., 2023，约 UC Berkeley）是答案。它常被称为 **Context Parallel（CP）**。

### 3.1 核心思想：KV 在环上轮转，Q 不动

把序列沿 $S$ 切成 $P$ 段，卡 $i$ 持有第 $i$ 段的 $Q_i, K_i, V_i$。注意力要让每个 $Q_i$ 看到**全部** $K,V$。Ring Attention 不一次性收齐所有 KV（那会爆显存），而是：

```
环形拓扑（P=4）：卡0 → 卡1 → 卡2 → 卡3 → 卡0

每一步：每张卡用“本地 Q”和“当前手里的 KV 块”算一次局部注意力，
        然后把 KV 块发给下一张卡、从上一张卡收一个新 KV 块。
共 P 步，每张卡都见过全部 P 个 KV 块 → 等价于全局注意力。
```

ASCII 时间线（卡0 的视角，P=4）：

```
 step0: Q0 与 K0V0 算 → 累加          [同时把 K0V0 发给卡1，从卡3 收 K3V3]
 step1: Q0 与 K3V3 算 → 累加          [转手 K3V3 给卡1，收 K2V2]
 step2: Q0 与 K2V2 算 → 累加          [转手...，收 K1V1]
 step3: Q0 与 K1V1 算 → 累加          [环闭合]
 ───────────────────────────────────
 结果：Q0 看遍 K0..K3，即全序列注意力
```

### 3.2 关键：在线 Softmax（online softmax）让“累加”正确

注意力做的是 $\text{softmax}(QK^\top)V$。Softmax 跨整行归一化，不能简单把各块结果相加。Ring Attention 借用 FlashAttention 同款的**在线 softmax**，维护三个滚动量：

- $m$：到目前为止见过的最大打分（数值稳定用）
- $\ell$：到目前为止的指数和（分母）
- $O$：到目前为止的加权输出（分子方向）

来一个新 KV 块就**重缩放并合并**，全程不需要实体化整行 $S$ 长的分数。详细推导见 [[llm-optimizer/FlashAttention]]，这里给合并规则：

$$
m_{\text{new}} = \max(m, \tilde m),\quad
\ell_{\text{new}} = e^{m-m_{\text{new}}}\ell + e^{\tilde m - m_{\text{new}}}\tilde\ell,\quad
O_{\text{new}} = \frac{e^{m-m_{\text{new}}}\ell\,O + e^{\tilde m - m_{\text{new}}}\,\tilde O}{\ell_{\text{new}}}
$$

其中带 $\tilde\cdot$ 的是新块的局部统计量。

### 3.3 杀手锏：计算与通信重叠 + O(1) 额外显存

- **显存**：任意时刻每卡只多持有**一个** KV 块（$O(S/P)$），而不是全部 KV。这就是 Ring Attention 能上 1M token 的根本原因。
- **重叠**：第 $t$ 步在算注意力时，第 $t{+}1$ 步要用的 KV 块正在网上飞。**计算时间 $\gtrsim$ 通信时间时，通信被完全隐藏**，几乎免费拿到全局注意力。

### 3.4 因果掩码的负载均衡坑

因果（causal）注意力下，靠后的 query 要看更多的 key，**直接顺序切会导致卡之间计算量不均**（持有靠后段的卡累死、靠前段的卡闲死）。实践用 **zig-zag / 条带切分**（如 Llama3、Megatron-CP 的做法）：把序列切成 $2P$ 段，卡 $i$ 同时拿“第 $i$ 段 + 第 $2P{-}1{-}i$ 段”，让每张卡的因果三角负载大致相等。

```
顺序切（P=4，✓=要算的因果块，越后越满）：
  卡0: ▓               卡3: ▓▓▓▓   ← 严重不均
zig-zag 切（每卡拿一前一后两段）：
  卡0: 段0 + 段7   卡1: 段1 + 段6  ...  ← 负载拉平
```

---

## 4. 流派三：DeepSpeed-Ulysses（all-to-all 切头/切序列互转）

DeepSpeed-Ulysses（Jacobs et al., 2023，约 Microsoft）走另一条路：**用 all-to-all 在“切 $S$”和“切 head”之间瞬间切换**。

### 4.1 机制

- 进入注意力**前**：激活按 $S$ 切（每卡 $S/P$ 个 token、但持有**全部** head 维 $H$）。
- 一次 **all-to-all**：转成“每卡持有**全部** $S$ 个 token、但只有 $a/P$ 个 head”。
- 这时每卡可以**对自己负责的 head 独立做完整的全序列注意力**（因为它有这些 head 的全部 $S$）。
- 注意力**后**：再来一次 **all-to-all** 转回“切 $S$”布局。

```
 注意力前 (切 S):   卡i 持 [S/P tokens, 全部 a 个 head]
         │  all-to-all（沿 S 散开、沿 head 收拢）
         ▼
 注意力中 (切 head): 卡i 持 [全部 S tokens, a/P 个 head] → 本地算全序列注意力
         │  all-to-all（逆操作）
         ▼
 注意力后 (切 S):   回到 [S/P tokens, 全部 a 个 head]
```

### 4.2 取舍

| 维度 | Ring Attention | DeepSpeed-Ulysses |
| --- | --- | --- |
| 通信原语 | P2P 环形轮转（可与计算重叠） | all-to-all（两次/层） |
| 通信量随 $P$ | 与 $P$ 弱相关，易跨机扩展 | all-to-all 量 ∝ 激活大小，受 head 数限制 |
| 并行度上限 | 几乎不受 head 数限制（适合超长） | 受 head 数约束（$P \le a$） |
| 实现复杂度 | 高（在线 softmax + 负载均衡） | 较低（复用标准注意力 kernel） |
| 典型场景 | 1M+ 超长、跨机 | 中长序列、单机/小集群 |

> 现代框架（Megatron-LM、TransformerEngine）常把二者**混合**（hierarchical CP）：机内用 Ulysses/all-to-all，机间用 Ring，吃 NVLink 也吃带宽。**以官方实现为准。**

---

## 5. 关键公式 / 算法 / 数值示例

### 5.1 激活显存：切前 vs 切后

设单层激活 $\text{Act}_{\text{layer}}(S) \approx c_1 BSH + c_2 B a S^2$（见 §1.1）。沿 $S$ 切成 $P$ 份后，每卡承担：

$$
\text{Act}^{\text{SP}}_{\text{per-card}} \approx \frac{c_1 BSH}{P} + \frac{c_2 B a S^2}{P}
$$

> 线性项严格 $/P$。平方项在 Ring/Flash 下不实体化整张 $S\times S$，每卡注意力的瞬时 KV 缓冲是 $O(S/P)$，所以也大致 $/P$。**结论：SP 把激活显存近似降到 $1/P$。**

### 5.2 Ring Attention 通信量手算

每步每卡发送一个 KV 块。一个 KV 块大小（bf16）：

$$
\text{size}_{KV} = 2 \times B \times \frac{S}{P} \times H \times 2\,\text{bytes}\quad(\text{K 和 V 各一份})
$$

共 $P{-}1$ 次轮转（最后一步不用再发）。**单卡前向总发送量**：

$$
\text{Comm}_{\text{fwd}} \approx (P-1)\cdot \text{size}_{KV} \;\approx\; 2BSH \times 2\,\text{bytes}\quad(\text{近似与 }P\text{ 无关})
$$

**手算例子**：$B=1$、$S=128\text{K}$、$H=8192$、$P=8$、bf16：

```
size_KV = 2 · 1 · (131072/8) · 8192 · 2 B
        = 2 · 16384 · 8192 · 2 B ≈ 0.5 GB / 块
单卡前向发送 ≈ (8-1) · 0.5 GB ≈ 3.5 GB ≈ 2BSH·2B（≈4.3GB 量级）
```

只要这 3.5GB 的传输时间 < 8 步局部注意力的计算时间，通信就被算力**完全隐藏**。NVLink（数百 GB/s）下，这通常成立 → **Ring Attention 接近“免费扩展上下文”**。数字为量级演示，**以实测为准**。

### 5.3 Megatron-SP 通信量

每层前向：1 次 all-gather + 1 次 reduce-scatter（替代原 TP 的 2 次 all-reduce）。环形算法下，单卡单次 all-gather/reduce-scatter 的数据量约 $\frac{P-1}{P}\cdot(B\,S\,H)\cdot 2\,\text{bytes}$，两者之和 $\approx$ 一次 all-reduce 的量。**所以 Megatron-SP 相对纯 TP 的通信量基本持平，纯赚激活显存。**

### 5.4 Ring Attention 在线注意力（前向）伪代码

```
# 每张卡 i：本地 Q_i, K_i, V_i，形状 [B, S/P, H]
m = -inf;  l = 0;  O = 0                  # 在线 softmax 三件套
kv = (K_i, V_i)                            # 手里的 KV 块
for step in range(P):
    # 与下一张卡的通信和本步计算重叠
    send(kv, to=(i+1) % P);  recv(kv_next, from=(i-1) % P)
    s   = Q_i @ kv.K.T / sqrt(d)           # [B, S/P, S/P] 局部分数（causal 需掩码）
    m_b = rowmax(s)
    p   = exp(s - m_b)
    l_b = rowsum(p)
    O_b = p @ kv.V
    # 合并（数值稳定，见 §3.2 公式）
    m_new = max(m, m_b)
    scale_old = exp(m - m_new);  scale_new = exp(m_b - m_new)
    l = scale_old*l + scale_new*l_b
    O = scale_old*l_old*O ... 按 §3.2 归一化
    m = m_new
    kv = kv_next                           # 接力，进入下一步
return O / l                               # 最终输出 [B, S/P, H]
```

---

## 6. 显存收益总览（数值表）

设 $H=8192$、$L=80$、$B=1$、bf16，只看线性激活项（§1.2 同口径，量级演示）：

| 配置 | 每卡序列 | 每卡激活（全模型，约） | 能否单卡放下 |
| --- | --- | --- | --- |
| $S$=128K，无 SP | 131072 | ~1.7 TB | 否（远超 80GB） |
| $S$=128K，SP=8 | 16384 | ~215 GB | 仍需配 TP/PP/重计算 |
| $S$=128K，SP=8 + 重计算 | 16384 | 大幅下降 | 可行 |
| $S$=128K，SP=32（跨机 Ring） | 4096 | ~54 GB | 接近可行 |

> 实战里 SP **从不单飞**，总是和 TP（切 $H$）、PP（切层）、ZeRO/重计算叠加。SP 负责砍掉那条**没人能替它砍的 $\propto S$（与 $S^2$）的激活**。

---

## 7. 评价 / 对照 / 局限

| 并行方式 | 切哪个维 | 主要省什么 | 主要通信 | 适用 |
| --- | --- | --- | --- | --- |
| 数据并行 DP | $B$（样本） | 吞吐扩展 | 梯度 all-reduce | 样本多、显存够 |
| 张量并行 TP | $H$（隐藏维） | 权重+部分激活 | all-reduce（层内、高频） | 单层放不下、单机 NVLink |
| 流水并行 PP | $L$（层） | 权重（按层分段） | P2P 激活（边界） | 模型层数多、跨机 |
| **序列并行 SP/CP** | **$S$（token）** | **$\propto S$、$\propto S^2$ 的激活** | all-gather/reduce-scatter 或 ring P2P / all-to-all | **长/超长上下文** |

**局限与坑：**

| 局限 | 说明 |
| --- | --- |
| 注意力是唯一“贵”的同步点 | 非注意力算子零通信，但注意力的跨卡通信省不掉，长序列下是主要开销 |
| 因果掩码负载不均 | 顺序切会让靠后段的卡偏忙，需 zig-zag/条带切分 |
| 与 TP 的“切维”要对齐 | Megatron 中 SP 复用 TP 通信组，切 $S$/切 $H$ 的切换点必须严格匹配，否则数值错 |
| 通信难完全隐藏时会拖慢 | 序列不够长或带宽不足，ring 通信无法被计算掩盖，加速比下滑 |
| Ulysses 受 head 数限制 | 并行度 $P\le a$，超长场景需配 Ring 才能继续扩 |
| 位置编码要小心 | RoPE 等需按**全局**位置索引计算，切段后每卡要用正确的全局偏移 |

**一句话评价**：SP/CP 是把 Transformer 推向 **百万 token 上下文** 的关键支柱——它精准地切掉了 DP/TP/PP 都救不了的那条随序列长度增长的激活曲线，代价是把“看全序列”的责任压在注意力的集合通信上，而 Ring Attention 用计算-通信重叠把这个代价几乎抹平。

---

## 8. 最小实现思路（落地清单）

1. **切数据**：在 dataloader/采样器处，把每条样本沿 $S$ 切成 $P$ 段，建立 `sp_group`（通信组）。注意保留**全局位置索引**给 RoPE。
2. **非注意力算子**：LayerNorm/MLP/Dropout/残差 **原样跑**，每卡只处理 $S/P$，零改动、零通信。
3. **注意力替换**：把标准 `scaled_dot_product_attention` 换成 ring/ulysses 实现（在线 softmax + 环形 P2P 或 all-to-all）。
4. **反向**：在线 softmax 的反向同样需要环形传 KV 的梯度，框架（Megatron-CP / `ring-flash-attn`）已封装，优先复用。
5. **与 TP/PP 组合**：先定 TP 度（受单机卡数/NVLink 约束），再用 SP/CP 把 $S$ 维继续切到跨机，最后叠 PP 与重计算。
6. **验证**：先用短序列对齐**单卡基线**的 loss/logits（数值一致性测试），再放长。

> 工程上**不要从零造轮子**：直接用 Megatron-LM 的 `--context-parallel-size`、DeepSpeed-Ulysses 或 `ring-flash-attn`。**具体参数与默认值以官方文档为准。**

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 上一层：[[llm-train/pytorch/distribution/README]] · [[llm-train/README]]
- 强相关：[[B07:llm-inference/大模型推理张量并行]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]
- 通信底座：[[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]]
- 推理侧延伸：[[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/vllm/README]]
- 指标口径：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
