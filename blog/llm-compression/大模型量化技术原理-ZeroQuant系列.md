# 大模型量化技术原理 - ZeroQuant 系列

> 微软 DeepSpeed 团队的训练后量化（PTQ）系列：从 ZeroQuant（W8A8 + 逐层知识蒸馏）一路演进到 V2（低秩补偿 LoRC）、FP（FP8/FP4 浮点格式）、HERO（硬件感知 W8A8）。一句话定位：**不重训、低成本、对 Transformer 友好**的工业级量化方案。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-inference/KV-Cache优化]] · [[llm-algo/transformer/模型架构]]

## 阅读地图

| 节 | 内容 | 你将带走什么 |
|----|------|--------------|
| 0 | 一句话锚点 | 一句话记住整个系列在干嘛 |
| 1 | 地基：PTQ vs QAT、对称/非对称、per-tensor/per-token | 量化的基本词汇与困难来源 |
| 2 | ZeroQuant：细粒度量化 + 逐层知识蒸馏（LKD）+ 高效内核 | 为什么能"零成本"量化 |
| 3 | ZeroQuant-V2：系统化对照 + 低秩补偿 LoRC | W4A8/W4A16 怎么救回精度 |
| 4 | ZeroQuant-FP：FP8/FP4 浮点格式登场 | 为什么浮点比整型更稳 |
| 5 | ZeroQuant-HERO：硬件增强的鲁棒 W8A8 | 怎么把算子融合算进量化设计 |
| 公式/示例 | 量化-反量化公式、显存账、INT8 加速比手算 | 能自己算一遍 |
| 评价 | 四代横向对照表 + 局限 | 选型时怎么挑 |

## 0. 一句话锚点

> **ZeroQuant 系列 = "训练后量化的工程化套路"**：用**细粒度量化**（权重 per-row / 激活 per-token）压住离群值带来的误差，用**逐层知识蒸馏**（不需要原始训练数据、不需要反向传播全网络）把精度找回来，再配**专门的融合内核**把"省下来的比特"真正变成"更快的推理"。后续三代分别补上：**低秩补偿**（救 INT4）、**浮点格式**（FP8/FP4 比 INT 更抗离群值）、**硬件协同**（把量化点设计在算子融合边界上）。

## 1. 地基：量化到底难在哪

### 1.1 量化是什么（30 秒直觉）

把一个 FP16 张量（每个数 16 bit）映射到 INT8（8 bit）甚至 INT4（4 bit）。核心就一个线性映射：

$$
x_{\text{int}} = \text{round}\!\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot x_{\text{int}}
$$

其中 $s$ 是缩放因子（scale）。对**对称量化**，$s = \dfrac{\max(|x|)}{2^{b-1}-1}$（$b$ 为比特数）。反量化 $\hat{x}$ 与原值 $x$ 的差就是**量化误差**。

```
 FP16 连续值          INT8 离散格点(256 档)
   |                       |  |  |  |  |  |
 -A .... 0 .... +A   →    -127 ......... +127
   每个真实值落到最近格点, 误差 ≤ s/2
```

### 1.2 PTQ vs QAT —— ZeroQuant 走的是 PTQ

| 维度 | PTQ（训练后量化） | QAT（量化感知训练） |
|------|------------------|---------------------|
| 是否重训 | 不需要 / 极少校准 | 需要完整训练 |
| 成本 | 分钟～小时级 | 天～周级 + 全量数据 |
| 精度 | 低比特易掉点 | 通常更高 |
| 适合 | 大模型（重训不起） | 中小模型 |

大模型（数十亿～万亿参数）**重训成本天文数字**，所以 ZeroQuant 全系列都站在 PTQ 这一侧——这正是"Affordable（负担得起）"的含义。

### 1.3 困难的根源：激活离群值（outlier）

Transformer 的激活里，**少数几个通道（channel）的数值能比其他通道大几十上百倍**。一旦用 per-tensor（整张量一个 scale）量化激活，scale 被离群值撑大，绝大多数正常值就被压到只剩几个格点 → 精度雪崩。

```
激活某层的逐通道幅度 (per-tensor 量化的灾难)
通道:  c0  c1  c2 ... c47 ...  c310 ...
幅度:  ▁   ▁   ▂      ▁     ███████ (离群通道, ×100)
         一个 c310 把全局 scale 撑大 → c0~c47 全被压扁
```

这就是后面 SmoothQuant、AWQ、以及 ZeroQuant 各代反复要解决的核心矛盾。ZeroQuant 的第一招就是**别用 per-tensor**。

### 1.4 量化粒度（granularity）—— ZeroQuant 的关键词

| 粒度 | 一个 scale 覆盖谁 | 优点 | 代价 |
|------|------------------|------|------|
| per-tensor | 整个张量 | 最省、最快 | 被离群值毁掉 |
| per-channel / per-row（权重） | 每行/每输出通道一个 | 抗权重异质 | scale 数变多 |
| per-token（激活） | 每个 token 一个 | 抗激活离群、动态 | 需运行时算 scale |
| per-group | 每 g 个元素一组 | INT4 救命 | 内核更复杂 |

ZeroQuant 的招牌组合：**权重 per-row（group-wise）+ 激活 per-token（动态）**。

```
      权重 W [out, in]                激活 X [token, hidden]
  ┌──────────────────┐           ┌──────────────────┐
  │ row0  → s_w0      │           │ tok0 → s_x0       │  ← 每个 token
  │ row1  → s_w1      │           │ tok1 → s_x1       │     一个 scale,
  │ ...   每行一scale │           │ ...   动态计算    │     运行时即算
  └──────────────────┘           └──────────────────┘
   权重量化是离线静态的            激活量化是在线动态的
```

## 2. ZeroQuant（2022）：细粒度 + 逐层蒸馏 + 高效内核

> 论文：*ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers*（NeurIPS 2022）。目标是 **W8A8**（权重 8bit、激活 8bit），主打"不要数据、不要长训练、还能给真加速"。

它由三个支柱组成，缺一不可：

### 2.1 支柱一：细粒度硬件友好量化方案

- **权重**：group-wise（逐行/逐组）量化。比 per-tensor 更细，又能被 GPU 高效执行（按行加载本就符合 GEMM 的 tiling）。
- **激活**：token-wise（per-token）**动态**量化。每个 token 在推理时即时统计自己的 $\max|x|$ 算 scale，天然避开"一个离群 token 撑大全局 scale"。

为什么这能扛住离群值？因为离群值往往集中在**特定 token 或特定通道**，把 scale 局部化后，离群只污染它自己那一组，不再连累全局。

```
ZeroQuant 数据流 (一层 Linear)
  X(FP16) ──per-token 动态量化──► X_int8
  W(FP16) ──per-row  离线量化───► W_int8        (静态, 一次到位)
  X_int8 × W_int8 ──INT8 GEMM──► Y_int32
  Y_int32 ──反量化(s_x ⊗ s_w)──► Y(FP16)
```

### 2.2 支柱二：逐层知识蒸馏（LKD, Layer-by-layer Knowledge Distillation）

这是 ZeroQuant 最聪明的地方。直接量化到 INT8/INT4 会掉点，正常做法是用数据重训（QAT），但那既要数据又要算力。LKD 的思路：

**把"量化后的第 k 层"去对齐"原始（FP16）第 k 层"的输出。** 逐层、独立地做，损失就是：

$$
\mathcal{L}_{\text{LKD}}^{(k)} = \big\| \, L_k(\hat{W}_k;\, X) \; - \; L_k(W_k;\, X) \, \big\|^2
$$

其中 $L_k$ 是第 $k$ 层，$\hat{W}_k$ 是量化权重，$X$ 是上一层（用**原始模型**）的输出。

为什么这是"零成本"级别的：

1. **不需要标注/原始训练数据**——可以用任意数据甚至随机文本喂进去，因为我们对齐的是"老师层的输出"而不是真实标签。
2. **不需要端到端反向传播**——只在**单层**内做优化，显存和算力只占一层，能在单卡上跑超大模型。
3. **逐层串行**——一层调完再调下一层，老师始终是原始全精度模型。

```
LKD 逐层对齐 (老师=FP16 原模型, 学生=量化层)
                老师层 L_k(W_k)
   X ───┬──────────────────────► Y_teacher
        │                              │  对齐 (MSE)
        └────► 学生层 L_k(Ŵ_k) ──► Y_student
              只优化 Ŵ_k 的量化参数, 显存只占一层
```

直觉：与其让"考砸的整张卷子"重做，不如**一道题一道题**对答案订正——便宜、可控、可并行调度。

### 2.3 支柱三：高度优化的推理后端（融合内核）

量化省了比特，但如果"量化/反量化"本身开销很大，就白省了。ZeroQuant 把量化、反量化、GeLU、LayerNorm 等**算子融合**进 GEMM 前后，减少 HBM 读写。Transformer 推理本就**访存瓶颈**（memory-bound），把权重从 FP16 变 INT8 直接把权重搬运量减半，这才是加速的来源。

```
未融合:  GEMM → 写回 → 读 → 量化 → 写 → 读 → GeLU ...(多次往返 HBM)
已融合:  [量化 ⊕ GEMM ⊕ 反量化 ⊕ GeLU] 一个 kernel, 中间量留寄存器/共享内存
```

### 2.4 ZeroQuant 的成绩（定性）

- W8A8 几乎无损，可在 BERT / GPT-3 风格模型上落地。
- 对更大的模型（如百亿级），论文还展示了 **W4A8** 的可行性（配合 LKD）。
- 端到端推理可获**约数倍**加速（具体倍数见原文，依模型/硬件而定）。

> 数字一律以原文为准；这里只给量级直觉。

## 3. ZeroQuant-V2（2023）：系统对照 + 低秩补偿 LoRC

> 论文：*ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation*。两件事：(1) 把 INT4/INT8、权重/激活的各种组合**系统地横评**一遍；(2) 提出 **LoRC（Low Rank Compensation，低秩补偿）** 把 INT4 的精度找回来。

### 3.1 系统性结论（很有工程指导意义）

- **激活比权重难量化得多**：W4A16 往往比 W8A8 还稳，因为激活离群值是主要杀手。所以低比特优先压**权重**（W4），激活保持 16/8 bit。
- **模型越大，越能扛量化**——大模型有冗余，但同时**离群值现象也更严重**，需要更细粒度。
- **INT4 单靠 round-to-nearest（RTN）会明显掉点**，需要补偿手段（LoRC，或与 GPTQ 类方法结合）。

```
量化难度阶梯 (经验):
  W8A8  ≈ 无损
  W4A16 → 轻微掉点 (LoRC 可救)
  W4A8  → 中等掉点 (LoRC + 细粒度)
  W4A4  → 困难 (激活 INT4 离群难压)
```

### 3.2 LoRC：用一个低秩项补回量化误差

核心观察：量化误差 $E = W - \hat{W}$ 不是随机噪声，它有**结构**，可以用一个**低秩矩阵**来近似并加回去。

对误差做 SVD，取前 $r$ 个奇异值，构造低秩补偿 $U_r \Sigma_r V_r^\top$：

$$
W \approx \hat{W} + \underbrace{U_r \Sigma_r V_r^{\top}}_{\text{低秩补偿(可再量化)}}, \qquad
E = W - \hat{W} \overset{\text{SVD}}{=} U \Sigma V^{\top}
$$

```
原权重 W (FP16)
   │ 量化
   ▼
  Ŵ (INT4)  +  误差 E = W - Ŵ
                 │ SVD, 只保留前 r 个奇异值
                 ▼
            Û = U_r  (d×r),  V̂ = V_r (r×k)   ← r 很小(如 8~32)
   推理: Y = X·Ŵ(INT4)  +  (X·Û)·V̂            ← 主路 INT4 + 旁路低秩
```

**显存代价极小**：主权重 $d\times k$ 是 INT4，补偿项只多两个瘦长矩阵 $d\times r$ 和 $r\times k$，$r \ll \min(d,k)$。比如 $d=k=4096$、$r=16$，补偿项参数量 $= 2\times4096\times16 \approx 13万$，仅占原权重 $4096^2\approx1678万$ 的 **约 0.8%**。用不到 1% 的额外参数，把 INT4 的精度大幅拉回——这就是 LoRC 的性价比。

> 直觉：INT4 把权重"压扁"丢的信息，大多集中在少数几个主方向上（SVD 的大奇异值），用一个低秩"补丁"专门补这几个方向就够了。

## 4. ZeroQuant-FP（2023）：浮点格式（FP8 / FP4）登场

> 论文：*ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats*。核心主张：**做低比特时，浮点（FP8/FP4）往往比整型（INT8/INT4）更稳**，尤其在有离群值的激活上。

### 4.1 为什么浮点比整型抗离群值

整型量化是**均匀格点**（等间距）；浮点是**非均匀**——靠近 0 的地方格点密、远离 0 的地方格点稀。LLM 的权重/激活分布是**钟形 + 长尾**：绝大多数值集中在 0 附近，少数离群值在远端。

```
INT4 (均匀, 16 档):    | | | | | | | | | | | | | | | |   ← 0 附近不够密
FP4  (非均匀):     ||||| | |   |    |     |        |       ← 0 附近密, 长尾也能覆盖
分布:             ▁▂▅█▅▂▁ .................. ▁(离群)
```

浮点把"格点预算"花在数据真正密集的地方，对**有长尾的 LLM 分布天然契合**——这就是 FP 在低比特更稳的根因。FP8 还可选 **E4M3 / E5M2** 两种指数/尾数划分：E4M3 精度高、动态范围小（适合权重/激活），E5M2 范围大、精度低（适合梯度）。

### 4.2 W4A8 + FP 的组合拳

ZeroQuant-FP 主打 **W4A8**：权重 FP4（或 INT4）、激活 FP8。再叠加 V2 的 LoRC 低秩补偿，进一步缩小掉点。关键工程点：

- FP8 在新硬件（如 Hopper 架构 Tensor Core）上**有原生算力支持**，不是软件模拟，所以"更稳"的同时"还很快"。
- 权重用 FP4 时，需要 group-wise scale（延续 ZeroQuant 的细粒度传统）。

```
ZeroQuant-FP 一层数据流 (W4A8)
  X ──per-token──► X_FP8 ─┐
  W ──group-wise─► W_FP4 ─┤── FP GEMM(硬件原生) ──► Y
                          └── + LoRC 低秩旁路(补偿 FP4 误差)
```

> 结论（定性）：在 W4A8 设定下，FP 格式 + LoRC 能逼近 FP16 基线；具体掉点幅度见原文。

## 5. ZeroQuant-HERO（2023）：硬件增强的鲁棒 W8A8

> 论文：*ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers*。前几代关注"精度怎么救"，HERO 把视角转向**硬件执行效率**：让量化的"边界"恰好落在**算子融合的边界**上，避免反复在 INT 与 FP 之间来回转换。

### 5.1 问题：量化点放错了，反量化开销吃掉收益

一个典型 Transformer block 里有 LayerNorm、QKV 投影、Attention、残差、MLP。如果每个 Linear 各自量化/反量化，**INT↔FP 的转换会在 HBM 反复读写**，把 INT8 GEMM 省下来的时间又赔进去。

```
朴素 W8A8 (转换过多):
  LN(FP) → 量化 → QKV(INT8) → 反量化(FP) → Attn(FP) → 量化 → ...
            ↑ 每个箭头都可能是一次 HBM 往返

HERO (按融合边界放量化点):
  [LN ⊕ 量化] → QKV INT8 → [反量化 ⊕ 残差 ⊕ 下一步量化] 融合
            ↑ 量化算子被"吸"进相邻算子, 中间量不落 HBM
```

### 5.2 HERO 的三类设计

1. **内存敏感算子（LayerNorm/残差）的量化-反量化与之融合**：把量化点放在 LN 之后、Linear 之前，让 LN 直接吐 INT8。
2. **计算敏感算子（GEMM/Attention）走 INT8**：享受 Tensor Core 的 INT8 峰值算力。
3. **鲁棒性处理**：对实在难量化的少数模块（如某些 Attention 分支）保留更高精度，做**混合精度**，保证整体不掉点——这就是 "Robust" 的来源。

> 一句话：HERO = "把量化当成编译/调度问题来做"，让**量化方案与算子融合策略协同设计**，把 W8A8 在真实硬件上的端到端收益最大化。

## 关键公式 / 算法 / 数值示例

### A. 量化-反量化（对称、per-row/per-token）

$$
s = \frac{\max(|x|)}{2^{b-1}-1},\quad
x_{\text{int}} = \text{clip}\!\Big(\text{round}\big(\tfrac{x}{s}\big),\,-(2^{b-1}{-}1),\,2^{b-1}{-}1\Big),\quad
\hat{x}=s\cdot x_{\text{int}}
$$

### B. INT8 量化的 GEMM 反量化（双 scale）

激活 per-token scale $s_x$、权重 per-row scale $s_w$，则
$$
Y_{ij} \approx s_{x,i}\, s_{w,j} \sum_k (X_{\text{int}})_{ik}\,(W_{\text{int}})_{kj}
$$
即 **INT32 累加结果**乘上"行 scale ⊗ 列 scale"得到 FP 输出。

### C. 显存账（手算）—— 以 7B 模型为例

设参数量 $N = 7\times10^9$。仅算权重存储：

| 精度 | 每参数字节 | 权重显存 | 相对 FP16 |
|------|-----------|----------|-----------|
| FP16 | 2 B | $14$ GB | 1.0× |
| INT8 (W8) | 1 B | $7$ GB | 0.5× |
| INT4 (W4) | 0.5 B | $3.5$ GB | 0.25× |

$14\text{ GB} = 7\times10^9 \times 2\text{ B} / 10^9$。W4 把权重压到 **约 3.5 GB**，加上 LoRC 补偿项（< 1%，约 +0.03 GB）几乎可忽略——这就是为什么 24GB 显卡能塞下原本放不下的模型。

### D. INT8 加速比的来源（手算直觉）

Transformer 解码阶段是 **memory-bound**：每生成一个 token 要把全部权重从 HBM 读一遍。
- FP16 权重 14 GB，HBM 带宽设 $B$（GB/s），读一遍 $\approx 14/B$ 秒。
- INT8 权重 7 GB，读一遍 $\approx 7/B$ 秒。

理想情况下**权重搬运时间减半 → 解码吞吐约 2×**（实际受反量化、kernel 效率折损，见原文）。这解释了为什么 ZeroQuant 强调"高效内核"——**没有融合内核，省下的带宽换不成真加速**。

### E. LoRC 的额外开销（手算）

$d=k=4096$，$r=16$：补偿参数 $=2dr = 2\times4096\times16 = 131072$；原权重 $dk=16{,}777{,}216$。比例 $\approx 0.78\%$。推理多两次瘦矩阵乘 $X\!\cdot\!\hat U$（$\text{token}\times d \times r$）和 $(\cdot)\!\cdot\!\hat V$，FLOPs 增加同量级（< 1%）。

## 评价 / 对照 / 局限

### 四代横向对照

| 版本 | 目标精度 | 核心武器 | 解决的问题 | 适用场景 |
|------|----------|----------|------------|----------|
| ZeroQuant | W8A8(可 W4A8) | 细粒度量化 + LKD逐层蒸馏 + 融合内核 | 零数据/低成本量化 + 真加速 | 工业落地首选基线 |
| ZeroQuant-V2 | W4A16/W4A8 | 系统横评 + LoRC低秩补偿 | INT4 掉点 | 显存极度受限 |
| ZeroQuant-FP | W4A8(FP) | FP8/FP4 浮点格式 + LoRC | 离群值下整型不稳 | 新硬件(原生FP8) |
| ZeroQuant-HERO | W8A8 | 硬件感知 + 算子融合协同 | 反量化开销吃掉收益 | 极致端到端吞吐 |

### 与同类方法的关系

| 方法 | 主战场 | 与 ZeroQuant 的关系 |
|------|--------|---------------------|
| LLM.int8() | 离群值分离(混合精度分解) | 思路互补：HERO 也用混合精度兜底 |
| SmoothQuant | 把激活难度"迁移"到权重 | 同样在治激活离群，可叠加 |
| GPTQ | 二阶信息逐列量化权重 | 可替代/配合 LoRC 做 W4 补偿 |
| AWQ | 按激活重要性保护权重通道 | 同为权重低比特路线 |

### 局限与注意事项

1. **激活 INT4（A4）仍是硬骨头**：离群值在激活上最难压，全系列对 A4 都偏保守。
2. **动态 per-token 量化有运行时开销**：每 token 算 scale，需内核高度优化才不拖后腿。
3. **LoRC 的秩 $r$ 是超参**：太小补不回、太大失去意义，需按层调。
4. **FP8/FP4 依赖硬件**：老卡（无原生 FP8 Tensor Core）上 FP 路线优势打折扣。
5. **数字以原文为准**：本文加速比/掉点幅度均为量级直觉，精确值见各篇论文实验表。

## 🔗 跳转链接

- 📍 [[00-知识地图]]
- [[llm-compression/quantization/量化基础]] —— 对称/非对称、per-tensor/channel、PTQ/QAT 的根基
- [[llm-compression/quantization/fp8]] —— E4M3/E5M2 与 ZeroQuant-FP 的浮点格式细节
- [[llm-compression/sparsity/README]] —— 量化的姊妹压缩路线：稀疏/剪枝
- [[llm-inference/KV-Cache优化]] —— 推理访存瓶颈与 KV-Cache 量化的延伸
- [[llm-inference/vllm/README]] —— 把量化模型部署进高吞吐推理引擎
- [[llm-algo/transformer/模型架构]] —— 离群值现象的来源：Transformer 结构
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] —— 加速比/吞吐怎么量化评估
- [[docs/transformer内存估算]] —— 配合显存账理解量化省了多少

---

参考文档：

- DeepSpeed（实现）: https://github.com/microsoft/DeepSpeed
- ZeroQuant: https://arxiv.org/pdf/2206.01861.pdf
- ZeroQuant-V2: https://arxiv.org/abs/2303.08302
- ZeroQuant-FP: https://arxiv.org/pdf/2307.09782.pdf
- ZeroQuant-HERO: https://arxiv.org/pdf/2310.17723.pdf
- 大模型量化概述: https://www.zhihu.com/question/627484732/answer/3261671478
