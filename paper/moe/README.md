# MoE 混合专家：从稀疏激活到万亿参数大模型(论文精读合集)

> 一句话定位：MoE(Mixture-of-Experts，混合专家)用「一个路由器 + 一堆专家 FFN + Top-k 稀疏激活」把模型**总参数量**做大、却让**每个 token 实际计算量**几乎不变，是当下万亿级大模型(GShard / Switch / GLaM / Mixtral / DeepSeek-MoE / DBRX)的核心骨架。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/moe/README]] · [[llm-algo/transformer/模型架构]] · [[llm-train/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-inference/PD分离]]

## 阅读地图

| 节 | 读什么 | 一句话收获 |
|----|--------|-----------|
| 0 | 一句话锚点 | MoE = 稀疏激活的条件计算 |
| 1 | 地基 | 为什么"参数越多越好"撞上算力墙 |
| 2 | 原始 MoE(1991/2017) | 路由器 + 专家 + 加权求和 |
| 3 | GShard | 把 MoE 塞进 Transformer + 容量因子 + 辅助负载均衡损失 |
| 4 | Switch Transformer | Top-1 路由，简单到极致 |
| 5 | GLaM / 路由演进 | 节能与质量的甜点 |
| 6 | Mixtral 8x7B | 开源 MoE 的标杆 |
| 7 | DeepSeek-MoE | 细粒度专家 + 共享专家 |
| 8 | LoRA × MoE(MoLA) | 微调期的 MoE：把专家做成 LoRA |
| 9 | 系统工程 | 专家并行 / All-to-All / 显存与通信账 |
| 公式节 | 路由/负载均衡/FLOPs/显存手算 | 能上手估算 |
| 评价节 | 横向对照 + 局限 | 知道何时别用 MoE |

---

## 0. 一句话锚点

> **稠密模型**：每个 token 都过完整的网络，参数量 = 计算量，二者绑死。
> **MoE 模型**：把一层巨大的 FFN 拆成 N 个小 FFN(专家)，每个 token 只被路由到其中 k 个(通常 k=1 或 2)。于是：
>
> $$\text{总参数} \uparrow N\text{倍}, \qquad \text{单 token FLOPs} \approx \text{不变}$$
>
> 这就是 **条件计算(conditional computation)** / **稀疏激活(sparse activation)**。一句话：**用参数量换知识容量，用稀疏换算力。**

---

## 1. 地基：为什么需要 MoE(问题背景)

### 1.1 缩放定律的代价

Scaling Law 告诉我们：模型越大、数据越多，loss 越低。但稠密 Transformer 的算力随参数**线性增长**：

```
稠密模型扩容:  参数 ×10  ⇒  训练算力 ×10  ⇒  推理延迟 ×10  ⇒  💸💸💸
```

一个 FFN 层占 Transformer 约 2/3 的参数和算力(`d_model → 4·d_model → d_model`)。如果想把"知识容量"做大，最贵的就是 FFN。MoE 的洞察：

> **不是每个 token 都需要整个 FFN 的全部容量。** 处理"代码"的 token 和处理"诗歌"的 token，可能需要不同的子网络。让 token **按需选择**子网络，就能在不增加单步算力的前提下扩容。

### 1.2 直觉类比

```
稠密 FFN:  一个全科医生，什么病都看(参数全开)
MoE   :   一家医院，挂号台(路由器)把病人分给专科医生(专家)
          ┌─────────────────────────────────────────┐
          │  挂号台 Router:  看症状(token表示)分诊     │
          │     ↓        ↓         ↓        ↓          │
          │  心内科   骨科     皮肤科    神经科  ...    │  ← N 个专家
          │  (只去 1~2 个科室，不用全院会诊)            │
          └─────────────────────────────────────────┘
```

---

## 2. 原始 MoE：路由器 + 专家(1991 Jacobs；2017 Shazeer "Outrageously Large")

### 2.1 结构(逐原子拆解)

一个 MoE 层 = **门控网络 Gating/Router** + **N 个专家网络 Expert**。

```
            x (token 表示, 维度 d_model)
              │
       ┌──────┴───────┐
       ▼              ▼
   Router g(x)     [复制给被选中的专家]
   = softmax(x·Wg)
   选出 Top-k 专家
       │
       ▼ 权重 g_i
  ┌────┬────┬────┬─────┐
  │ E1 │ E2 │ E3 │ ... │  每个 Ei 是一个独立 FFN
  └────┴────┴────┴─────┘
       │(仅 Top-k 个真正计算)
       ▼
  y = Σ_{i∈Top-k} g_i(x) · E_i(x)   ← 加权求和
```

- **Router**：一个最简单的线性层 $W_g \in \mathbb{R}^{d \times N}$，把 token 投影成 N 个分数，softmax 后取 Top-k。
- **Expert** $E_i$：通常就是一个标准 FFN(`Linear → 激活 → Linear`)。
- **输出**：被选中专家输出的**门控加权和**。没被选中的专家这一步**完全不算**(稀疏)。

### 2.2 关键创新点(2017 Shazeer 等)

1. **Noisy Top-k Gating**：路由分数里加可学习高斯噪声，鼓励探索、避免早期就锁死到少数专家。
2. **辅助负载均衡损失**(见公式节)：防止"赢者通吃"——少数专家被路由所有 token，其余饿死。
3. 在 LSTM 之间插入 MoE 层，做到 **1370 亿参数** 的语言模型(2017 年的"outrageously large")。

### 2.3 核心痛点(后续所有工作都在解决)

| 痛点 | 现象 | 后续解法 |
|------|------|---------|
| 负载不均 | 专家被路由的 token 数极不平衡 | 辅助损失、容量因子、丢弃溢出 token |
| 路由不可导 | Top-k 选择是离散的 | 用门控权重做软加权 + STE 近似 |
| 通信开销 | 专家分布在不同设备，要 All-to-All | 专家并行 + 通信优化 |

---

## 3. GShard：把 MoE 工业化塞进 Transformer(Google, 2020)

> 论文做了什么：把 MoE **每隔一层** 替换 Transformer 的 FFN，配合 **自动分片(SPMD)** 把专家铺到几千张 TPU 上，训出 **6000 亿参数** 的多语言翻译模型。它定义了后续 MoE 的工程范式。

### 3.1 核心方法

```
标准 Transformer Block:        GShard MoE Block (隔层替换):
  ┌─ Self-Attn ─┐               ┌─ Self-Attn ─┐
  │   ↓ +残差   │               │   ↓ +残差   │
  ├─    FFN    ─┤      ⇒        ├─  MoE-FFN  ─┤  ← 这层 FFN 换成 MoE
  │   ↓ +残差   │               │  (Top-2)    │
  └─────────────┘               └─────────────┘
```

### 3.2 三个关键工程创新

1. **专家容量(Expert Capacity)**：给每个专家设一个**固定容量** C(能处理的 token 上限)。

   $$C = \text{capacity\_factor} \times \frac{\text{tokens\_per\_batch}}{N_{\text{experts}}}$$

   - 路由到某专家的 token 超过 C → **溢出 token 被丢弃**(skip，直接走残差)。
   - capacity_factor 通常取 1.0~2.0：越大越不丢 token 但越浪费显存/计算。这是 MoE 最重要的旋钮之一。

2. **辅助负载均衡损失**：鼓励 token 在专家间均匀分布(公式节详解)。
3. **Top-2 路由**：每个 token 选 2 个专家，兼顾质量与稀疏。

### 3.3 对工程的启示

- MoE 不是免费的：固定容量意味着**显存按最坏情况预留**，且丢 token 会损质量。
- 隔层放 MoE(而非每层)是质量/成本的折中——注意力层仍稠密共享。

---

## 4. Switch Transformer：把路由简化到 Top-1(Google, 2021)

> 论文做了什么：证明 **Top-1 路由(每个 token 只去 1 个专家)** 不仅够用，还更稳更省。训出 **1.6 万亿参数** 模型，并给出大量稳定化技巧。

### 4.1 核心简化

```
GShard Top-2:  token → 选 2 专家 → 2 次 FFN + 加权        (路由复杂、通信×2)
Switch Top-1:  token → 选 1 专家 → 1 次 FFN               (最简、通信减半)
   y = g_i(x) · E_i(x)   (i = argmax 路由分数)
```

GShard 当年认为"至少要 Top-2 才有梯度对比"，Switch 反驳：**Top-1 也能学好**，且：
- 路由计算减半、All-to-All 通信减半；
- 每个专家 batch 更大，硬件利用率更高。

### 4.2 关键稳定化技巧(很实用)

| 技巧 | 解决什么 | 做法 |
|------|---------|------|
| **selective precision** | bf16/fp16 下路由 softmax 数值不稳 | 路由器内部用 **fp32** 计算 |
| **更小初始化** | 大 MoE 训练发散 | router 权重初始化 scale 调小(约 ×0.1) |
| **expert dropout** | 微调期过拟合 | 专家层用更高 dropout |
| **capacity factor** | 丢 token | 训练 1.0~1.25，推理可调大 |

### 4.3 实验结论(定性，数字见原文)

- 相同算力预算下，Switch 比稠密 T5 **预训练加速约数倍**(见原文 Figure)。
- 稀疏模型蒸馏回稠密小模型，可保留**约 30%** 的质量增益(具体见原文)。

---

## 5. GLaM 与路由演进：质量/能耗甜点(Google, 2021)

> GLaM(Generalist Language Model)：**1.2 万亿参数**的 decoder-only MoE，Top-2 路由。卖点是**能效**：

- 推理时只激活**约 1/16 的参数**(单 token 实际算的参数远小于总量)。
- 论文称：达到/超过 GPT-3 质量，训练能耗**约为其 1/3**、推理 FLOPs 更低(数字见原文)。

启示：MoE 的真正价值在 **"等质量下更省"** 或 **"等成本下更强"**，二选一看怎么用容量旋钮。

---

## 6. Mixtral 8x7B：开源 MoE 的标杆(Mistral AI, 2024)

> 论文/技术报告做了什么：开源一个 **8 专家、Top-2** 的 decoder-only MoE。命名"8x7B"是营销简写——**总参数约 47B(并非 8×7=56B，因为注意力等是共享的)**，但每个 token 只激活**约 13B** 参数。

### 6.1 关键事实(常被误解，务必记牢)

```
   8x7B 不等于 8 个独立的 7B 模型拼起来！
   ┌──────────────────────────────────────────────┐
   │  共享部分: Embedding / Attention / LayerNorm    │  ← 所有 token 共用
   │  MoE 部分: 每层 8 个 FFN 专家, Top-2 激活        │  ← 只有 FFN 被复制成 8 份
   └──────────────────────────────────────────────┘
   总参数 ≈ 47B   ·   每 token 激活 ≈ 13B
   ⇒ 推理速度/显存接近 13B 稠密，质量接近/超过 70B 稠密
```

### 6.2 工程启示

- **显存** 按 47B 算(权重全部要驻留)，但 **计算/带宽** 按 13B 算 → "胖但快"。
- 这正是 MoE 对推理部署的双刃剑：**省算力但费显存**。
- 后续 DBRX、Qwen-MoE、DeepSeek-V2/V3 都沿用并放大此范式。

---

## 7. DeepSeek-MoE：细粒度专家 + 共享专家(DeepSeek, 2024)

> 论文做了什么：针对"专家专精度不够、知识冗余"两个痛点，提出两条改进，让相同激活参数下质量更高。

### 7.1 两大创新

1. **细粒度专家切分(Fine-Grained Expert Segmentation)**
   把每个专家**做小**、数量**做多**，同时**等比例提高 Top-k**。

   ```
   传统:  16 个大专家, Top-2  →  组合数 C(16,2)=120 种
   细粒度: 64 个小专家, Top-8  →  组合数 C(64,8)≈4.4×10⁹ 种
   总激活参数不变，但专家"组合表达力"暴涨 ⇒ 更灵活的专精
   ```

2. **共享专家隔离(Shared Expert Isolation)**
   留出 1~2 个**永远被激活**的"共享专家"承载**通用知识**，让路由专家专注**差异化知识**，减少冗余。

   ```
   token x
     ├──→ [共享专家 Es] ──────────────┐  (恒激活, 学公共知识)
     └──→ Router → Top-k 路由专家 ─────┤  (按需激活, 学专精知识)
                                       ▼
                       y = Es(x) + Σ g_i·E_i(x)
   ```

### 7.2 对工程的启示

细粒度 + 共享专家几乎成为 2024 年后大型 MoE 的标配(DeepSeek-V2/V3 把它推到 256 路由专家 + Top-8 量级，数字见原文)。

---

## 8. LoRA × MoE：微调期的混合专家(MoLA / LoRA-meets-MoE，本目录原始方向)

> 原始链接方向：[MoLA](https://github.com/GCYZSL/MoLA) 与 [大模型微调新范式：当 LoRA 遇见 MoE](https://zhuanlan.zhihu.com/p/683637455)。
> 思路：**预训练已是稠密大模型**，但在 **PEFT 微调** 阶段，把"专家"做成轻量的 **LoRA 适配器**，用路由器在多个 LoRA 间稀疏选择，得到"任务/领域自适应"的低成本专家化。

### 8.1 核心思想

```
冻结的稠密 FFN 权重 W0 (不动)
        │
        x ──→ W0·x  (主干, 共享)
        │
        └─→ Router → Top-k 个 LoRA 专家:
              每个专家 = 低秩 ΔW_i = B_i A_i (秩 r 很小)
        y = W0·x + Σ_{i∈Top-k} g_i · (B_i A_i) x
```

- **专家 = LoRA**：每个专家只有 $2 \cdot d \cdot r$ 个参数(r 通常 4~64)，极轻。
- **MoLA 的额外发现**(见仓库)：不同 Transformer **层**应分配**不同数量**的 LoRA 专家——**高层(靠近输出)需要更多专家**，低层可以更少("layer-wise expert allocation")。

### 8.2 为什么有用

| 维度 | 全参 MoE 预训练 | LoRA-MoE 微调 |
|------|----------------|---------------|
| 训练成本 | 极高(从零训) | 极低(冻主干, 只训 LoRA) |
| 适用场景 | 造基座大模型 | 多任务/多领域定制 |
| 缓解灾难性遗忘 | — | 路由把任务隔到不同 LoRA, 减少互相干扰 |
| 部署 | 权重巨大 | 主干共享 + 几 MB 适配器 |

### 8.3 局限

- 路由器在微调小数据上易过拟合 / 退化为只用一个专家 → 仍需负载均衡正则。
- 专家数、秩 r、放哪几层都是超参，需要搜。

---

## 9. 系统工程：专家并行、All-to-All 与三本账

### 9.1 专家并行(Expert Parallelism, EP)

把 N 个专家**分到不同 GPU**，每卡只放一部分专家。token 需要"飞"到目标专家所在的卡，算完再飞回来——这就是 **两次 All-to-All** 通信。

```
   设备0      设备1      设备2      设备3
   E0,E1     E2,E3     E4,E5     E6,E7
     │         │         │         │
  本卡 token 按路由结果分发到各卡  ← All-to-All #1 (dispatch)
     ▼         ▼         ▼         ▼
   各卡专家本地计算 FFN
     │         │         │         │
  结果按来源收回原卡        ← All-to-All #2 (combine)
```

### 9.2 与其他并行的关系

```
DP 数据并行: 复制整模型,切 batch | TP 张量并行: 切单层权重矩阵
EP 专家并行: 切专家(每卡不同专家) | PP 流水并行: 切层(不同卡放不同层)
↑ 大型 MoE 通常 DP × TP × EP × PP 四者叠加 (3D/4D 并行)
```

---

## 关键公式 / 算法 / 数值示例

### A. 路由与门控

路由分数与 Top-k 选择：
$$h(x) = W_g \cdot x, \qquad g_i(x) = \frac{\exp(h_i(x))}{\sum_{j} \exp(h_j(x))}, \qquad \text{选 } \text{Top-}k\{g_i\}$$

层输出(以 Top-k 为例)：
$$y = \sum_{i \in \text{Top-}k(x)} g_i(x)\, E_i(x)$$

### B. 辅助负载均衡损失(Switch 形式，最常用)

设这一 batch 共 T 个 token、N 个专家：
- $f_i$ = 被**路由**(argmax)到专家 i 的 **token 占比**；
- $P_i$ = 专家 i 的**平均路由概率**(softmax 概率均值)。

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- 当负载完全均匀时 $f_i=P_i=1/N$，$\sum f_i P_i = 1/N$，损失最小。
- $\alpha$ 是小系数(典型量级 $10^{-2}$，以原文为准)。
- 乘 $N$ 是为了让损失尺度与专家数解耦。
- 总损失 = 语言模型损失 + $\mathcal{L}_{\text{aux}}$。

### C. 容量因子与丢弃

$$C = \text{cf} \times \frac{T}{N}, \qquad \text{drop\_rate} = \frac{\#\{\text{溢出 token}\}}{T}$$

**手算示例**：T = 4096 token，N = 8 专家，cf = 1.25：
$$C = 1.25 \times \frac{4096}{8} = 1.25 \times 512 = 640 \text{ token/专家}$$
若某专家被路由到 700 个 token，则 $700 - 640 = 60$ 个 token 溢出被丢(走残差)。

### D. FLOPs：MoE 省了多少计算(单 token，单 MoE 层)

一个 FFN(`d → 4d → d`)的前向 FLOPs ≈ $2 \times (d \cdot 4d + 4d \cdot d) = 16 d^2$。

| 方案 | 单 token FFN-FLOPs | 说明 |
|------|--------------------|------|
| 稠密(等价单 FFN) | $16 d^2$ | 基线 |
| MoE Top-1 | $\approx 16 d^2$ | 只算 1 个专家，与稠密相当 |
| MoE Top-2 | $\approx 32 d^2$ | 算 2 个专家 |
| 稠密"扩 8 倍宽" | $\approx 128 d^2$ | 同样 8× 参数但全激活 → 贵 8× |

**结论**：MoE 用 Top-2 拿到 8× 参数容量，单 token 只多花约 2× 算力(相对单专家)，而非 8×。

### E. 显存账(Mixtral 风格，d 抽象化)

```
权重显存:  按"总参数"算 (47B 全部要常驻)
           bf16 ⇒ 47B × 2 B ≈ 94 GB  (需多卡/量化)
计算/带宽: 按"激活参数"算 (~13B) ⇒ 解码速度像 13B
⇒ MoE = "占显存像大模型，跑起来像小模型"
```

**手算**：bf16 下每参数 2 字节。47B 参数 ⇒ $47\times10^9 \times 2 \approx 94$ GB 仅权重，单张 80GB A100 放不下 → 必须 TP/EP 切分或量化到 fp8/int4(见 [[llm-compression/quantization/fp8]])。

---

## 评价 / 对照 / 局限

### 横向对照

| 模型 | 年份 | 路由 | 专家数 | 总/激活参数 | 标志创新 |
|------|------|------|--------|------------|---------|
| Shazeer MoE | 2017 | Noisy Top-k | 数千 | ~137B / 稀疏 | 首个超大 MoE + 辅助损失 |
| GShard | 2020 | Top-2 | 2048 | ~600B | 容量因子 + SPMD 分片 |
| Switch | 2021 | **Top-1** | ~2048+ | ~1.6T | 极简路由 + 稳定化技巧 |
| GLaM | 2021 | Top-2 | 64/层 | ~1.2T / ~1/16 | 能效甜点 |
| Mixtral 8x7B | 2024 | Top-2 | 8 | ~47B / ~13B | 开源标杆 |
| DeepSeek-MoE | 2024 | Top-k(大) | 多(细粒度) | 见原文 | 细粒度 + 共享专家 |
| MoLA(LoRA-MoE) | 2024 | Top-k | 层级自适应 | 主干冻+轻量 | 微调期 MoE |

### 局限(知道何时别用)

| 局限 | 说明 | 缓解 |
|------|------|------|
| 显存爆炸 | 总参数全要驻留 | 量化、EP/TP 切分、专家 offload |
| 通信瓶颈 | All-to-All 受网络带宽限制 | 高速互联(NVLink/IB)、通信计算重叠 |
| 训练不稳 | 路由 softmax 数值/负载塌缩 | fp32 路由、辅助损失、小初始化 |
| token 丢弃 | 溢出 token 掉质量 | 调大 cf、无丢弃路由(BASE/Sinkhorn) |
| 微调易过拟合 | 路由器在小数据退化 | 正则、冻路由、LoRA-MoE |
| 部署复杂 | 调度/批处理比稠密难 | 专用推理框架(vLLM 等支持 MoE) |

> **一句话评价**：MoE 是"**用显存和工程复杂度，换算力效率和知识容量**"的交易。当你**算力受限但显存/带宽充裕**、且追求大容量时，MoE 极强；当你**显存吃紧、部署要简单**时，稠密模型更省心。

---

## 🔗 跳转链接

- 知识地图枢纽：[[00-知识地图]]
- 模型架构基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
- 训练与并行：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[ai-infra/网络/集合通信原语]]
- 推理与部署：[[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]] · [[B07:llm-inference/大模型推理张量并行]]
- 压缩(MoE 必备)：[[llm-compression/quantization/fp8]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/sparsity/README]]
- 微调对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 性能指标：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]

> 参考原始资料：MoLA 仓库 <https://github.com/GCYZSL/MoLA> ；《当 LoRA 遇见 MoE》<https://zhuanlan.zhihu.com/p/683637455> 。论文精确数字与超参以各自原文/官方为准。
