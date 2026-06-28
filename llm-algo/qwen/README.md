# Qwen 系列架构

> 阿里通义千问开源大模型家族：以 LLaMA 式 Decoder-only Transformer 为底座，在 RoPE/RMSNorm/SwiGLU/GQA 之上做工程化打磨，强多语言、强长上下文、有 MoE 变体。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/llama/模型架构]] [[llm-algo/qwen2]]

---

## 阅读地图

| 节 | 内容 | 你会得到什么 |
|----|------|--------------|
| 0 | 一句话锚点 | Qwen 到底是什么，与 LLaMA 一句话差异 |
| 1 | 地基/前置 | Decoder-only、自回归、Attention 最底层回顾 |
| 2 | RoPE 旋转位置编码 | 为什么用、怎么算，逐数手算一个 2 维例子 |
| 3 | RMSNorm | 为什么去掉均值、公式、与 LayerNorm 对比 |
| 4 | SwiGLU FFN | 门控为什么有效、三矩阵、参数量手算 |
| 5 | GQA 分组注意力 | KV cache 怎么省，省多少，手算显存 |
| 6 | 整体 Block 数据流 | ASCII 全景图，残差走向 |
| 7 | Qwen1.5→2→2.5 演进 | 每代改了什么，规模配置表 |
| 8 | 长上下文 | YaRN/Dual Chunk Attention 原理 |
| 9 | Qwen-MoE | 稀疏专家、细粒度+共享专家，激活参数手算 |
| 10 | 多语言 & 词表 | 为什么 15 万词表、tokenizer 设计 |
| — | 数值示例/手算 | 参数量、KV cache、FLOPs 全套手算 |
| — | 与 LLaMA 差异 | 逐点对照表 |
| — | 高频面试问答 | 踩点答案 + 追问 |

---

## 0. 一句话锚点

**Qwen（通义千问）是阿里巴巴开源的 Decoder-only 大语言模型系列**。它的骨架和 LLaMA 几乎同构（RoPE + RMSNorm（Pre-Norm）+ SwiGLU + GQA），核心差异是：

- **QKV 投影带 bias**（LLaMA 全程无 bias），其余 Linear 无 bias；
- **超大词表（约 15.2 万）**，强化中文/多语言与代码；
- **长上下文工程**（YaRN、Dual Chunk Attention，Qwen2.5 可达 128K，部分模型 1M）；
- **有 MoE 稀疏变体**（A14B/A3B 等，"A"=Activated 激活参数）。

> 记忆钩子：**"LLaMA 的身子 + QKV 加偏置 + 大词表 + 长窗口 + 可选 MoE"**。

---

## 1. 地基/前置（不假设你记得）

### 1.1 Decoder-only 自回归是什么

语言模型做一件事：给定前文 token 序列 $x_1,\dots,x_{t-1}$，预测下一个 token 的概率分布

$$P(x_t \mid x_1,\dots,x_{t-1})$$

"自回归"=一次预测一个、把预测结果接到输入末尾继续预测。"Decoder-only"=只用 Transformer 的解码器堆叠，**因果掩码**（Causal Mask）保证第 $t$ 个位置只能看见 $\le t$ 的位置。

### 1.2 一层 Transformer Block 由什么组成

```
输入 x  ──► [归一化] ──► [自注意力 Attn] ──► (+残差) ──► [归一化] ──► [前馈 FFN] ──► (+残差) ──► 输出
```

Qwen 用 **Pre-Norm**（归一化放在子层"前面"），这让深层网络梯度更稳，能堆几十层。下面逐个拆解 Qwen 用到的四块原子部件。

### 1.3 注意力最底层回顾

对每个 token 算出 Query/Key/Value 三个向量。注意力权重是 Q 和所有 K 的相似度（点积）经 softmax：

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$\sqrt{d_k}$ 是缩放，防止点积过大让 softmax 饱和（梯度消失）。**位置信息**不在这个公式里——必须额外注入，这就是 RoPE 的工作。

---

## 2. RoPE：旋转位置编码

### 2.1 为什么需要它

自注意力本身对 token 顺序**无感**（打乱输入注意力值不变）。必须告诉模型"谁在前谁在后"。绝对位置编码（加一个 position embedding）外推差；Qwen 全系用 **RoPE（Rotary Position Embedding）**——把位置编码成**旋转**，作用在 Q、K 上。

### 2.2 核心思想：用旋转编码相对位置

把 $d$ 维向量两两配对成 $d/2$ 个二维平面。位置 $m$ 的向量，在第 $i$ 个平面上旋转角度 $m\theta_i$，其中

$$\theta_i = 10000^{-2i/d}, \quad i=0,1,\dots,d/2-1$$

低维 $i$ 小 → $\theta$ 大 → 转得快（管短距离）；高维 $i$ 大 → $\theta$ 小 → 转得慢（管长距离）。

**关键性质**：Q（位置 $m$）和 K（位置 $n$）点积后，只依赖**相对位置 $m-n$**：

$$\langle R_m q,\ R_n k\rangle = \langle q,\ R_{n-m} k\rangle$$

这就是 RoPE 能良好外推长上下文的根。

### 2.3 ASCII：一个二维平面上的旋转

```
        位置 m=2, 角度 = 2·θ
         y
         │       q'(旋转后)
         │     ╱
         │    ╱  ↺ 转 2θ
         │   ╱
         │  ╱___ q(原始)
         │ ╱  ╲
         │╱    ╲ θ
─────────┼──────────► x
         │
   旋转矩阵 R(α) = [cos α  -sin α]
                   [sin α   cos α]
```

### 2.4 逐数手算（2 维，看懂就懂全部）

设某平面分量 $q=(1,\ 0)$，位置 $m=2$，$\theta=0.5$ rad，旋转角 $\alpha=m\theta=1.0$ rad。
$\cos 1.0\approx 0.5403,\ \sin 1.0\approx 0.8415$。

$$q' = R(1.0)\,q = \begin{bmatrix}0.5403 & -0.8415\\ 0.8415 & 0.5403\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}0.5403\\0.8415\end{bmatrix}$$

模长 $\sqrt{0.5403^2+0.8415^2}=\sqrt{0.292+0.708}=1.0$ —— **旋转保模长**，只改方向（=位置信息），不改向量"内容大小"。这就是 RoPE 不破坏特征尺度的原因。

### 2.5 长上下文外推与 base

把 base（默认 10000）调大（如 100 万、Qwen 长文模型用 **1,000,000**）→ 所有 $\theta_i$ 变小 → 旋转变慢 → 同样训练长度内角度跨度变小 → 外推更平滑。这叫 **NTK-aware / base 缩放**，配合 YaRN 是 Qwen 长上下文的基础（见 §8）。

---

## 3. RMSNorm：均方根归一化

### 3.1 LayerNorm 回顾与"为什么去掉均值"

LayerNorm：$\;y=\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot\gamma+\beta$，要算均值 $\mu$、方差 $\sigma^2$、有缩放 $\gamma$ 和偏置 $\beta$。

RMSNorm 发现：**重新中心化（减均值）对效果贡献小，重新缩放才是关键**。于是只保留缩放：

$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}\cdot \gamma$$

少算一个均值、少一个 $\beta$ 参数、少一次减法广播 → **更快、更省、效果几乎不掉**。Qwen 全系采用 RMSNorm。

### 3.2 逐数手算

设 $x=(3,\ 4,\ 0,\ 0)$，$d=4$，$\epsilon$ 忽略，$\gamma=1$。

$$\text{RMS}=\sqrt{\tfrac{3^2+4^2+0+0}{4}}=\sqrt{\tfrac{25}{4}}=\sqrt{6.25}=2.5$$

$$y=\frac{x}{2.5}=(1.2,\ 1.6,\ 0,\ 0)$$

验证：新向量的均方根 $\sqrt{(1.44+2.56)/4}=\sqrt{1}=1$ → 归一化把"尺度"拉到 1，模型只需用 $\gamma$ 学习每维该放大多少。

### 3.3 对比表

| | LayerNorm | RMSNorm（Qwen） |
|--|-----------|------------------|
| 减均值 | 是 | **否** |
| 可学习参数 | $\gamma,\beta$ | 仅 $\gamma$ |
| 统计量 | 均值+方差 | 仅均方根 |
| 速度 | 基准 | 略快（~7-10%） |

---

## 4. SwiGLU 前馈网络

### 4.1 普通 FFN vs 门控

传统 FFN：$\;\text{FFN}(x)=W_2\,\text{ReLU}(W_1 x)$，两个矩阵。SwiGLU 用 **门控线性单元（GLU）+ SiLU/Swish 激活**，三个矩阵：

$$\text{SwiGLU}(x)=W_{\text{down}}\Big(\underbrace{\text{SiLU}(W_{\text{gate}}\,x)}_{\text{门}}\ \odot\ \underbrace{(W_{\text{up}}\,x)}_{\text{值}}\Big)$$

其中 $\text{SiLU}(z)=z\cdot\sigma(z)=\dfrac{z}{1+e^{-z}}$，$\odot$ 是逐元素相乘。

**为什么有效**：门控让网络对每个隐藏维度"自适应开关"，表达力比固定 ReLU 强；SiLU 平滑可导，优化更稳。代价是多一个矩阵（参数+50%），所以中间维 $d_{ff}$ 通常按 $\approx \tfrac{2}{3}\times 4d$ 缩小以保持总参数量。

### 4.2 ASCII 数据流

```
        x (d)
         │
    ┌────┴─────┐
    ▼          ▼
 W_gate     W_up      (各 d → d_ff)
    │          │
  SiLU         │
    │          │
    └──► ⊙ ◄───┘      逐元素相乘 (门 × 值)
         │
       W_down         (d_ff → d)
         │
         ▼  输出 (d)
```

### 4.3 参数量手算（以 Qwen2-7B 近似配置）

取 $d=3584$，$d_{ff}=18944$（以官方为准）。每层 FFN 三矩阵：

$$3\times d\times d_{ff}=3\times 3584\times 18944\approx 2.04\times 10^{8}\ \text{(≈2.04 亿)}$$

注意力部分（QKVO，GQA 下 KV 头少）远小于此，故 **FFN 是参数大头**。28 层时仅 FFN 就约 57 亿参数，与"7B"量级吻合。

---

## 5. GQA：分组查询注意力

### 5.1 痛点：KV cache 爆显存

自回归推理要缓存每个 token 的 K、V（KV cache）。多头注意力（MHA）每个 Q 头配一个独立 KV 头，KV cache 随头数线性增长，长序列下吃光显存。

### 5.2 三种方案

```
MHA：  Q1 Q2 Q3 Q4        每个 Q 头独享 KV  → KV 多，效果好，最费
       │  │  │  │
       K1 K2 K3 K4

GQA：  Q1 Q2 | Q3 Q4      每组共享一个 KV   → 折中（Qwen 用）
        \ /     \ /
        K1       K2

MQA：  Q1 Q2 Q3 Q4        所有 Q 共享 1 个 KV → KV 最少，质量略降
        \ │ │ /
          K1
```

Qwen2 起全系采用 **GQA**：Query 头数多，KV 头数少（每若干 Q 头共享一组 KV）。

### 5.3 KV cache 手算（GQA 省多少）

设 $L=32$ 层，隐藏维 $d=4096$，head_dim=128，MHA 有 32 个 KV 头，GQA 用 8 个 KV 头（4:1 分组）。序列长 $S=4096$，fp16（2 字节），存 K 和 V（×2）：

KV cache 字节数 $= 2\times L\times S\times (n_{kv}\times d_{head})\times 2$

- **MHA**（32 KV 头）：$2\times32\times4096\times(32\times128)\times2 = 4.29\times10^{10}\approx 40\ \text{GB}$
- **GQA**（8 KV 头）：$2\times32\times4096\times(8\times128)\times2 \approx 10\ \text{GB}$

**省到 1/4**。这就是 Qwen 能在单卡跑长上下文的关键工程。

---

## 6. 整体 Block 数据流（ASCII 全景）

```
                      ┌─────────────────── Qwen Decoder Block ×N ───────────────────┐
 token ids            │                                                              │
   │                  │   x ──► RMSNorm ──► [Attention(GQA)] ──┐                      │
   ▼                  │   │                  ↑RoPE 注入 Q,K     │                      │
 Embedding (15.2万词)─┼─► x ─────────────────────────────────► (+) 残差 ──┐           │
   │                  │                                          │         │           │
   │                  │   r ──► RMSNorm ──► [SwiGLU FFN] ────────┼──► (+) 残差         │
   │                  │   └──────────────────────────────────────┘   │               │
   └──────────────────┼─────────────────────────────────────────────► 输出 → 下一层  │
                      └──────────────────────────────────────────────────────────────┘
                                          │ (最后一层后)
                                          ▼
                              RMSNorm(final) ──► LM Head (tie/untie) ──► logits (词表大小)
```

要点：**Pre-Norm**（Norm 在子层前），两条残差直通梯度高速公路；RoPE 只作用在 Attention 的 Q、K 上；QKV 投影**带 bias**（Qwen 标志性细节）。

---

## 7. Qwen1.5 → Qwen2 → Qwen2.5 演进

### 7.1 各代关键变化

| 代际 | 关键点 |
|------|--------|
| **Qwen（初代）** | Decoder-only，部分模型 MHA；QKV 带 bias 的设计确立；约 15 万词表 |
| **Qwen1.5** | 代码并入 HF Transformers（`transformers>=4.37.0`，免 `trust_remote_code`）；统一支持 **32K 上下文**；6 个尺寸 0.5B/1.8B/4B/7B/14B/72B + Int4/Int8 GPTQ/AWQ/GGUF 量化；改善多语言与对齐 |
| **Qwen2** | **全系 GQA**；引入 **MoE（Qwen2-57B-A14B）**；长上下文用 **YaRN + Dual Chunk Attention**（部分至 128K）；多语言扩到约 30 种 |
| **Qwen2.5** | 预训练数据扩到约 **18T tokens**；指令遵循、长文本生成、结构化输出（JSON）、数学/代码大幅提升；旗舰长上下文 **128K**、生成 8K；尺寸覆盖 0.5B→72B，含 Coder / Math 专用线 |

### 7.2 规模配置参考（数字以官方技术报告为准）

| 模型 | 层数 | 隐藏维 d | Q 头 / KV 头 | 上下文 |
|------|------|---------|--------------|--------|
| Qwen2.5-0.5B | 约 24 | 约 896 | 14 / 2 | 32K |
| Qwen2.5-7B | 约 28 | 约 3584 | 28 / 4 | 128K* |
| Qwen2.5-72B | 约 80 | 约 8192 | 64 / 8 | 128K* |

\* 长上下文需启用 YaRN/DCA，具体以官方配置为准；上表数值为公开近似，请以 HF config 为准。

---

## 8. 长上下文：YaRN + Dual Chunk Attention

### 8.1 问题

训练时序列长度有限（如 32K），但推理想喂 128K。直接外推 RoPE 会"角度越界"导致注意力崩坏。

### 8.2 YaRN（Yet another RoPE extensioN）

按频率分段缩放 RoPE：高频（管局部）几乎不动，低频（管全局）按比例插值压缩角度，并配一个注意力温度修正。效果：用少量长文微调即可把窗口扩 4×/8×，困惑度平稳。本质就是 §2.5 的 base/频率缩放的精细版。

### 8.3 Dual Chunk Attention（DCA）

把超长序列切成若干 chunk，**块内**用真实相对位置、**块间**用重映射后的位置索引，使任意两 token 的相对距离落在训练见过的范围内。

```
长序列:  [chunk0][chunk1][chunk2]...
块内位置:  0..k   0..k    0..k     ← 真实近距离
块间索引:  用相对块号映射，保持"距离不超训练上限"
```

YaRN 改频率、DCA 改位置索引，二者叠加让 Qwen 在远超训练长度时仍稳定。

---

## 9. Qwen-MoE：稀疏专家

### 9.1 思想

MoE（Mixture of Experts）把 FFN 换成 N 个并行专家，每个 token 经 **路由器（Router）** 只激活 Top-k 个专家 → **总参数大、单次计算小**。

### 9.2 Qwen MoE 的两个细节

- **细粒度专家（fine-grained）**：把专家切得更小、数量更多，组合更灵活；
- **共享专家（shared expert）**：始终激活的专家，承载通用知识，其余路由专家学专门知识。

```
        token x
          │
     ┌────┴─────┐
     ▼          ▼
 [Shared Expert]   [Router] ──► 选 Top-k 个路由专家
     │              ┌──┬──┬──┬── ... ──┐
     │              E1 E2 E3 E4 ...    EN   (只算被选中的)
     └──────► (+) 加权求和 ◄────────────┘
                   │
                   ▼  输出
```

### 9.3 激活参数手算（"A14B"的含义）

以 **Qwen2-57B-A14B** 为例：**总参数约 57B，每 token 只激活约 14B**。算力近似只随"激活参数"走，所以推理成本约等于一个 14B 稠密模型，但容量（知识上限）接近 57B。这就是 MoE 的核心红利：**用 14B 的钱办 57B 的事（在被路由命中的能力上）**。

---

## 10. 多语言 & 词表设计

### 10.1 为什么用约 15.2 万的大词表

- **中文/多语言**：中文不是空格分词，子词颗粒大词表能更短地编码中文/日文/韩文/阿拉伯文等，**降低同一句话的 token 数** → 省算力、长文更友好；
- **代码**：保留缩进、符号的高效编码；
- Qwen2/2.5 覆盖约 **29-30 种语言**，词表用 byte-level BPE，保证任何字节都能编码（无 UNK）。

### 10.2 大词表的代价与权衡

Embedding 和 LM Head 各 $V\times d$。$V=151{,}936,\ d=3584$ 时单个就 $\approx 5.4\times10^8$（5.4 亿）参数。小模型里词表占比很高，所以 Qwen 小模型常 **tie embedding**（输入输出共享权重）以省参数。

---

## 数值示例 / 全套手算

### A. 7B 量级单层参数（GQA, 取 d=3584, d_ff=18944, Q头28/KV头4, head_dim=128）

- Attn 投影：Q $d\times(28\cdot128)=3584\times3584$；K,V 各 $3584\times(4\cdot128)=3584\times512$；O $3584\times3584$。
  合计 $\approx 3584\times(3584+512+512+3584)=3584\times8192\approx 2.94\times10^7$（约 0.29 亿）。
- FFN：$3\times3584\times18944\approx 2.04\times10^8$（约 2.04 亿，见 §4.3）。
- **FFN 约为 Attn 的 7 倍** → 大模型参数主要在 FFN。

### B. 单 token 前向 FLOPs 经验公式

稠密模型一次前向约 $2\times P$ FLOPs（$P$=参数量）。7B 模型生成 1 个 token ≈ $2\times7\times10^9=1.4\times10^{10}$ FLOPs。生成 1000 token ≈ $1.4\times10^{13}$ FLOPs（不含 KV cache 复用节省）。

### C. KV cache（见 §5.3）

GQA(8 KV 头) 相对 MHA(32 KV 头) 省到 **1/4**，4096 长度 32 层 4096 维下从 ~40GB 降到 ~10GB。

### D. RMSNorm 手算见 §3.2；RoPE 手算见 §2.4。

---

## 与 LLaMA 的逐点对照

| 维度 | LLaMA | Qwen | 说明 |
|------|-------|------|------|
| 架构范式 | Decoder-only | Decoder-only | 同构 |
| 归一化 | RMSNorm Pre-Norm | RMSNorm Pre-Norm | 相同 |
| 位置编码 | RoPE | RoPE | 相同；Qwen 长文调大 base |
| FFN | SwiGLU | SwiGLU | 相同 |
| 注意力 | MHA→GQA(LLaMA2/3) | GQA(Qwen2+) | 相同思路 |
| **QKV bias** | **无 bias** | **QKV 有 bias** | ⭐ 标志性差异 |
| 词表 | 32K(L2)/128K(L3) | **约 151.6K** | Qwen 更大，强中文 |
| 多语言 | 偏英文 | 约 30 语种、强中文 | ⭐ |
| MoE | 部分 (L4) | Qwen2/3 有 MoE | 都走稀疏 |
| 长上下文 | RoPE 缩放 | YaRN + DCA | Qwen 工程更细 |
| 框架接入 | HF 原生 | HF 原生(1.5+) | 相同 |

> 一句话：**Qwen = LLaMA 同款积木，加 QKV bias、大词表、强多语言、更细的长上下文工程，并提供 MoE 变体。**

---

## 常见问题 / 高频面试追问

| 问题 | 踩点答案 | 追问 / 陷阱 |
|------|----------|-------------|
| Qwen 和 LLaMA 架构差别？ | 同构(RoPE/RMSNorm/SwiGLU/GQA)；差异：**QKV 带 bias**、约 15.2 万大词表、强多语言、YaRN+DCA 长文、有 MoE | 追问"bias 为什么有用"：经验上略增表达力/稳定性，代价极小 |
| 为什么用 RMSNorm 不用 LayerNorm？ | 去掉减均值，只重缩放，几乎不掉点且更快更省 | 追问公式：$x/\sqrt{\text{mean}(x^2)+\epsilon}\cdot\gamma$ |
| RoPE 为什么能外推长上下文？ | 旋转编码的是**相对位置**，且保模长不破坏特征；调大 base/YaRN 可平滑外推 | 追问 base：默认 1e4，长文模型用 1e6 |
| GQA 省了什么，省多少？ | 省 **KV cache**；KV 头数/Q 头数 倍数即压缩比，常见 4× 或 8× | 陷阱：GQA 省的是 KV 不是 Q 计算；MQA 更极端但质量略降 |
| SwiGLU 比 ReLU-FFN 好在哪？ | 门控自适应通路 + SiLU 平滑；代价多一个矩阵，故 $d_{ff}$ 取约 $\tfrac23\cdot4d$ | 追问参数：三矩阵 $3\,d\,d_{ff}$ |
| "A14B" 是什么意思？ | MoE 的**激活参数 14B**，总参数更大（如 57B），算力随激活参数走 | 追问：细粒度专家 + 共享专家 |
| Qwen2.5 比 2 提升点？ | ~18T 数据、指令遵循/结构化输出(JSON)、数学代码、128K 长文、8K 生成 | 陷阱：128K 需启用 YaRN/DCA，非默认全开 |
| 为什么词表这么大？ | 高效编码中文/多语言/代码，降 token 数；代价是 embedding/head 参数大，小模型 tie | 追问：byte-level BPE 无 UNK |
| 长上下文两板斧？ | **YaRN**(按频率缩放 RoPE) + **Dual Chunk Attention**(块内真实/块间重映射位置) | 二者正交叠加 |

---

## 模型下载（ModelScope / HuggingFace）

```bash
# 初代 / 1.5 / 2.5
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
git clone https://www.modelscope.cn/qwen/Qwen1.5-0.5B.git
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```

参考实现（HF Transformers）：
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

> Qwen1.5 起代码并入 HF Transformers，`transformers>=4.37.0` 即可直接加载，无需 `trust_remote_code`；已支持 vLLM、SGLang、AutoGPTQ 等推理/量化框架。

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站索引导航
- [[llm-algo/llama/模型架构]] — LLaMA 架构对照，理解 Qwen 同款积木
- [[llm-algo/qwen2]] — Qwen2 细节深入
