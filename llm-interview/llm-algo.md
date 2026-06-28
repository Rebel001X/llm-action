# 面试·算法与模型

> 大模型「结构层」面试一站式：从 Attention 家族（MHA/MQA/GQA/MLA）、位置编码（RoPE）、稀疏化（MoE）到各家架构与归一化设计，全部拆到原子并配手算。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/moe/README]] [[llm-algo/旋转编码RoPE]] [[llm-algo/deepseek/DeepSeek-V2]]

## 阅读地图

| 节 | 主题 | 你会得到 | 高频追问 |
|---|---|---|---|
| 1 | 归一化地基 | BN/LN/RMSNorm/Pre-Post-Norm | 为什么 LLM 用 RMSNorm+Pre-Norm |
| 2 | 注意力总账 | Q/K/V、缩放、KV Cache | 为什么除以 $\sqrt{d_k}$ |
| 3 | MHA→MQA→GQA→MLA | KV 压缩四级演进 | 各省多少显存、掉点吗 |
| 4 | RoPE | 旋转位置编码 + 外推 | 为什么能做相对位置、NTK/YaRN |
| 5 | MoE | 稀疏专家、路由、负载均衡 | 激活参数 vs 总参数 |
| 6 | 各家架构 | LLaMA/GPT/DeepSeek/Qwen/Mixtral | 它们各自的关键创新 |
| 7 | 手算专场 | 参数量/显存/FLOPs/KV Cache | 现场口算 |
| 8 | 高频陷阱 | 易错点清单 | — |

## 0. 一句话锚点

- **Attention 演进的唯一主线**：在「不掉点」前提下，把推理时最贵的 **KV Cache** 越压越小 —— MHA → MQA → GQA → MLA。
- **RoPE 的本质**：把「绝对位置」编码成「旋转角度」，让两个 token 的注意力只依赖**相对位置差** $m-n$。
- **MoE 的本质**：用「总参数大、激活参数小」换「容量大、算力省」。
- **归一化在 LLM 的共识**：Pre-Norm + RMSNorm（好训、省算）。

---

## 1. 地基：归一化（Normalization）

### 1.1 为什么要归一化
深层网络中，每层输出分布会随训练漂移（Internal Covariate Shift 的直觉），导致梯度不稳。归一化把激活拉回到「均值 0、方差 1 附近」，再用可学习的 $\gamma,\beta$ 缩放平移，既稳又不丢表达力。

### 1.2 BatchNorm vs LayerNorm —— 沿哪个轴统计是核心区别

设一批激活张量形状 `[B, T, H]`（批、序列、隐藏维）：

```
            H (特征/channel)
            ────────────►
  B,T  ┌───────────────────┐
 (样本)│  x x x x x x x x   │
   │   │  x x x x x x x x   │
   ▼   │  x x x x x x x x   │
       └───────────────────┘

LayerNorm: 对【每个 token 自己的 H 维】求 μ,σ  → 横着切一行
BatchNorm: 对【同一 channel 跨 batch 所有样本】求 μ,σ → 竖着切一列
```

- **BN** 沿 batch 维统计 → 依赖 batch 大小、且训练/推理统计量不一致（要存 running mean/var）。序列长度可变 + 推理常 batch=1 的 LLM 场景非常不友好。
- **LN** 沿特征维统计 → **与 batch 无关**，每个 token 独立归一化，天然适配变长序列与自回归推理。**所以 Transformer 用 LN 不用 BN。**

> 选择口诀：**某个维度的差异性需要被模型拟合，就别在那个维度归一化。** NLP 里 batch 内不同句子语义差异很大，沿 batch 归一化会抹掉这种差异，故选 LN。

LayerNorm 公式（对一个 token 的隐藏向量 $x\in\mathbb{R}^H$）：
$$\mu=\frac1H\sum_i x_i,\quad \sigma^2=\frac1H\sum_i (x_i-\mu)^2,\quad \text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

### 1.3 RMSNorm —— LLaMA/Baichuan2 都用它

动机（论文 *Root Mean Square Layer Normalization*）：LayerNorm 既算均值又算方差，开销大。作者发现**re-centering（减均值）贡献小，真正有用的是 re-scaling（除以尺度）**。于是丢掉均值，只用均方根：
$$\text{RMS}(x)=\sqrt{\frac1H\sum_i x_i^2+\epsilon},\qquad \text{RMSNorm}(x)=\gamma\odot\frac{x}{\text{RMS}(x)}$$

- 省去求均值与减均值，**节省约 7%~64% 归一化运算**，且参数少了 $\beta$。
- 效果与 LN 相当甚至略好 → 成为现代 LLM 默认。

**手算**：$x=[3,-1,2,0]$，$H=4$，$\epsilon$ 忽略。
$\sum x_i^2 = 9+1+4+0=14$；$\text{RMS}=\sqrt{14/4}=\sqrt{3.5}\approx1.871$；
$\text{RMSNorm}(x)\approx[1.604,-0.535,1.069,0]\odot\gamma$。
对比 LN：均值 $\mu=(3-1+2+0)/4=1$，需先减均值再算方差 —— 多两步。

### 1.4 Post-Norm vs Pre-Norm —— 残差里 Norm 放哪

```
Post-Norm:  x ─► Sublayer ─► (+x) ─► Norm ─► out      Norm(x + F(x))
                    残差在 Norm 之前

Pre-Norm:   x ─►─────────────►(+)──► out              x + F(Norm(x))
            └► Norm ─► Sublayer ─┘
                    残差是一条干净的恒等通路
```

- **Pre-Norm 更好训**：恒等路径 $x$ 不经过 Norm 直接相加，梯度可无损回传，深层不易梯度消失/爆炸 → **几乎所有大模型用 Pre-Norm**。
- **Post-Norm 上限更高（同深度下）**：归一化作用在残差之后，正则更强、鲁棒性更好；但深层难训。
- 结论：**层数少 → Post-Norm 可能更好；层数多（LLM）→ Pre-Norm 必选**。
- 折中方案：DeepNorm、Sandwich-Norm（Pre+Post 各放一个）等。

---

## 2. 注意力总账：先把 Self-Attention 算清楚

### 2.1 Scaled Dot-Product Attention
输入序列嵌入 $X\in\mathbb{R}^{T\times d}$，通过三个投影得到 Query/Key/Value：
$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V$$
$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```
 Q (T×d_k)        K (T×d_k)
    │  ┌── Kᵀ ──┐    │
    └─►│ QKᵀ    │ T×T 分数矩阵
       └────────┘
          │ ÷√d_k  → softmax(逐行) → 权重 A (T×T)
          ▼
        A · V  →  输出 (T×d_v)
```

**为什么除以 $\sqrt{d_k}$？** 设 $q_i,k_i$ 各分量独立、均值 0 方差 1，则点积 $q\cdot k=\sum_{i=1}^{d_k} q_ik_i$ 的方差为 $d_k$。$d_k$ 大时点积数值大 → softmax 进入饱和区 → 梯度趋近 0。除以 $\sqrt{d_k}$ 把方差拉回 1，避免饱和。

**手算（为什么不缩放会爆）**：$d_k=64$，点积期望量级 $\sqrt{64}=8$。两个 logit 相差 8，softmax 后概率比 $e^8\approx2981$ —— 几乎 one-hot，梯度近 0。除以 $\sqrt{64}=8$ 后量级回到 1，分布平滑。

### 2.2 因果掩码与 KV Cache（推理性能的命根子）
自回归生成时，第 $t$ 步只需用当前 token 的 $q_t$ 去注意**历史所有** $k_{\le t},v_{\le t}$。把历史 K/V 缓存下来，每步只算一个新 token，避免重复计算 —— 这就是 **KV Cache**。

- 推理瓶颈从「算力」转为「**显存带宽**」：每生成一个 token 都要把整个 KV Cache 从 HBM 读一遍。
- **KV Cache 显存** ≈ $2 \times L \times T \times H \times \text{dtype}$（2 是 K 和 V，$L$ 层，$T$ 序列长，$H$ 隐藏维）。
- **这就是 MQA/GQA/MLA 全部要解决的问题**：把 $H$（确切说是 KV 头数）压小。

---

## 3. 注意力家族：MHA → MQA → GQA → MLA（面试核心）

### 3.1 直观全景图
设隐藏维 $d=H$，头数 $n_h$，每头维度 $d_h=H/n_h$。区别只在 **K/V 有几组头**：

```
        Q heads        K/V heads         KV Cache 大小
MHA   ████████ (8)   ████████ (8)        100%   每个 Q 头一套独立 KV
MQA   ████████ (8)   █        (1)        1/8    所有 Q 头共享一套 KV
GQA   ████████ (8)   ██       (2组)      1/4    每组 Q 头共享一套 KV
MLA   ████████ (8)   ▣ 低秩潜向量c        ~1/14  存压缩潜向量，用时再升维
```

### 3.2 MHA（Multi-Head Attention，原版）
把 $Q,K,V$ 拆成 $n_h$ 个头，各自做 attention 再拼接。**每个头都有自己独立的 K、V**。
- KV Cache 头数 = $n_h$（最大）。表达力最强，显存最贵。

### 3.3 MQA（Multi-Query Attention）
**所有 Query 头共享同一组 K、V**（KV 头数 = 1）。
- KV Cache 直接缩小 $n_h$ 倍。
- 代价：表达力受限，**容易掉点**、训练可能不稳。PaLM、Falcon 用过。

### 3.4 GQA（Grouped-Query Attention，工业界主流）
折中：把 $n_h$ 个 Q 头分成 $g$ 组，**每组共享一套 K、V**（KV 头数 = $g$，$1<g<n_h$）。
- $g=1$ 退化为 MQA，$g=n_h$ 退化为 MHA。
- LLaMA-2/3 70B、Mixtral 等广泛采用（常 $g=8$）。**几乎不掉点，KV Cache 降到 $g/n_h$**。

**手算（GQA 省多少）**：LLaMA-2 70B，$n_h=64$，$g=8$。KV Cache 缩小 $64/8=8$ 倍。若原本 KV Cache 占 40GB，GQA 后约 5GB。

### 3.5 MLA（Multi-head Latent Attention，DeepSeek-V2 创新）
GQA 减头数会损表达力。MLA 换思路：**不减头，而是把 K/V 联合压成一个低秩潜向量 $c^{KV}$ 缓存**，用时再升维还原各头的 K/V。→ 详见 [[llm-algo/deepseek/DeepSeek-V2]]。

```
        token h_t (d 维)
           │  ↓ 下投影 W_DKV (d → d_c，d_c 很小)
        c_t^{KV} (压缩潜向量)   ←─── 只缓存这个！
           │  ↑ 上投影 W_UK / W_UV
        K_t, V_t (各头还原)  ── 用矩阵吸收技巧，推理时无需显式还原
```

- 只缓存 $c^{KV}$（维度 $d_c \ll n_h\cdot d_h$），KV Cache 比 MHA 小一个数量级。
- 关键技巧 **「矩阵吸收」**：上投影矩阵可与 $W_Q$、输出投影合并，推理时不必显式重建 K/V，省算又省存。
- RoPE 兼容问题：旋转作用在还原后的 K 上会破坏低秩结构，DeepSeek 用 **解耦 RoPE（decoupled RoPE）** —— 额外保留一小段专门承载位置的维度。
- 效果：**KV Cache ≈ MHA 的 1/14 左右**（DeepSeek-V2 报告口径，以官方为准），且性能不降反升。

### 3.6 四者对比表

| 方案 | KV 头数 | KV Cache 相对量 | 表达力 | 典型模型 |
|---|---|---|---|---|
| MHA | $n_h$ | 1（基准） | 最强 | GPT-2/3、原始 Transformer |
| MQA | 1 | $1/n_h$ | 易掉点 | PaLM、Falcon、StarCoder |
| GQA | $g$ | $g/n_h$ | 几乎无损 | LLaMA-2/3 70B、Mixtral、Qwen2 |
| MLA | 低秩 $c$ | ≈1/14（V2 口径） | ≥MHA | DeepSeek-V2/V3 |

---

## 4. RoPE：旋转位置编码（→ [[llm-algo/旋转编码RoPE]]）

### 4.1 要解决什么
Transformer 本身对位置无感（attention 是集合运算）。需要注入位置信息，且希望注意力 $q_m\cdot k_n$ **只依赖相对位置 $m-n$**，并能外推到训练没见过的更长序列。

### 4.2 核心思想：把位置编成「旋转角」
不在输入上加位置向量，而是**对 Q、K 向量按其位置旋转一个角度**。二维情形，位置 $m$ 的旋转矩阵：
$$R_m=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}$$
对 $q$（位置 $m$）和 $k$（位置 $n$）分别旋转后做内积：
$$(R_m q)^\top (R_n k)=q^\top R_m^\top R_n k=q^\top R_{n-m}\,k$$

```
       k 在位置 n            q 在位置 m
        ╲ 旋转 nθ            ╱ 旋转 mθ
         ╲                  ╱
   内积只看夹角差 (m-n)θ  ← 这就是「相对位置」！
```

> **关键性质**：旋转矩阵满足 $R_m^\top R_n = R_{n-m}$，所以内积自动变成相对位置 $m-n$ 的函数 —— 这是 RoPE 优雅之处：**用绝对位置编码，得到相对位置效果**。

### 4.3 高维：分组旋转 + 频率衰减
把 $d_h$ 维向量两两配对成 $d_h/2$ 个二维子空间，第 $i$ 组用频率
$$\theta_i=\text{base}^{-2i/d_h},\quad \text{base}=10000$$
- 低维（$i$ 小）→ 高频，刻画近距离；高维（$i$ 大）→ 低频，刻画远距离。
- 实现上不显式构造旋转矩阵，用逐元素 $\cos/\sin$ + 「交错取负」即可。

**手算**：$d_h=4$，base=10000。$\theta_0=10000^{0}=1$，$\theta_1=10000^{-2/4}=10000^{-0.5}=0.01$。位置 $m=2$：第 0 组转 $2$ rad，第 1 组转 $0.02$ rad —— 高频组转得快、低频组转得慢。

### 4.4 长度外推：NTK / YaRN（高频追问点）
直接外推到更长序列会因高频「转太多圈」而失效。
- **位置插值 PI**：把位置 $m$ 缩放为 $m\cdot\frac{L_{train}}{L_{target}}$，等于把所有频率压缩 —— 简单但损高频分辨率。
- **NTK-aware**：只调 base（增大 base），让低频拉伸多、高频几乎不动，兼顾远近。
- **YaRN**：分频段处理 + 注意力温度修正，外推效果更好，是当前长上下文常用方案。

---

## 5. MoE：混合专家（→ [[llm-algo/moe/README]]）

### 5.1 动机：把 FFN 换成「一堆专家 + 路由」
稠密模型每个 token 都过全部参数，算力随参数线性增长。MoE 想：**参数总量很大（容量大），但每个 token 只激活一小部分（算力省）**。

```
            token x
               │
          ┌── Router(Gate) ── softmax → 选 Top-k 专家
          ▼
   ┌────┬────┬────┬────┬────┐   N 个专家(各是一个 FFN)
   │ E1 │ E2 │ E3 │ E4 │... │
   └─▲──┴────┴─▲──┴────┴────┘
     │ 只激活被选中的 k 个    │
     └──── 加权求和 ─────────┘ → 输出
```

### 5.2 路由与 Top-k
门控网络给每个专家打分 $g_i=\text{softmax}(x\cdot W_g)_i$，选分数最高的 $k$ 个专家（常 $k=1$ 或 $2$），输出为这 $k$ 个专家结果按门控权重加权和：
$$y=\sum_{i\in\text{TopK}} g_i\,E_i(x)$$

### 5.3 总参数 vs 激活参数（必背概念）
**手算（Mixtral 8x7B）**：8 个专家、每 token Top-2。
- 「8x7B」**不等于** 56B。专家只是 FFN 部分，注意力等是共享的，实际总参约 **47B**。
- 每 token 只激活 2 个专家 → **激活参数约 13B**。
- 即：**用 13B 的推理算力，跑出接近 47B 容量的效果**。这就是 MoE 的卖点。

DeepSeek-V3：**671B 总参 / 约 37B 激活**（细粒度专家 + 共享专家 + Top-8 路由，数字以官方技术报告为准）。

### 5.4 负载均衡（最常考的坑）
Router 容易「偷懒」：总把 token 发给少数几个专家 → 其他专家训不动、算力浪费、还可能爆显存。
- **辅助损失 Auxiliary Loss**：惩罚专家负载分布不均（鼓励每个专家被选概率均匀）。
- **容量因子 Capacity Factor**：给每个专家设处理上限，超出的 token 被 drop（或走残差）。
- **DeepSeek-V3 创新**：**Auxiliary-Loss-Free** 负载均衡（用可学习偏置动态调整路由），避免辅助损失干扰主任务。
- **细粒度专家 + 共享专家**：把专家切更细提升组合数，同时留 1~2 个「共享专家」处理通用知识，减少冗余。

---

## 6. 各家模型架构速记（面试常被点名）

| 模型 | Attention | 位置编码 | Norm | 激活/FFN | 关键创新 |
|---|---|---|---|---|---|
| GPT-2/3 | MHA | 学习式绝对位置 | LN, Pre-Norm | GELU | Decoder-only 范式确立 |
| LLaMA-1 | MHA | RoPE | RMSNorm, Pre | **SwiGLU** | 开源高质量基座、纯 Decoder |
| LLaMA-2/3 | **GQA**(大号) | RoPE | RMSNorm, Pre | SwiGLU | GQA 省 KV、更长上下文 |
| Mistral 7B | **GQA + 滑窗注意力** | RoPE | RMSNorm | SwiGLU | SWA 控长上下文算力 |
| Mixtral 8x7B | GQA | RoPE | RMSNorm | **MoE(8选2)** | 稀疏 MoE 实用化 |
| Qwen2 | GQA | RoPE | RMSNorm | SwiGLU | 多语言、长上下文、含 MoE 版 |
| DeepSeek-V2/V3 | **MLA** | 解耦 RoPE | RMSNorm | **细粒度 MoE+共享专家** | MLA 压 KV、无辅助损失均衡、MTP |
| GLM | MHA | RoPE(2D) | Post/Deep | GeGLU | 自回归填空预训练目标 |

> **SwiGLU** = $\text{SwiGLU}(x)=(\text{Swish}(xW_1))\odot(xW_3)$，门控激活，比 ReLU/GELU 更强；代价是 FFN 多一个矩阵，故隐藏维常取 $\frac{8}{3}d$ 而非 $4d$ 以对齐参数量。
> **Mistral 滑窗注意力(SWA)**：每个 token 只看前 $w$ 个 token（如 4096），通过层数堆叠间接获得更大感受野，把长序列注意力从 $O(T^2)$ 降到 $O(T\cdot w)$。
> **MTP（Multi-Token Prediction）**：DeepSeek-V3 训练时一次预测多个未来 token，增强信号、利于推测解码。

---

## 7. 手算专场（现场口算模板）

### 7.1 Transformer 参数量估算
单层主要来自 **Attention（4 个 $d\times d$ 投影）** 和 **FFN（两个 $d\times 4d$）**：
$$P_{\text{layer}}\approx \underbrace{4d^2}_{\text{Q,K,V,O}}+\underbrace{2\cdot 4d^2}_{\text{FFN}}=12d^2$$
全模型 $\approx 12\,L\,d^2$（忽略 embedding/norm）。

**手算**：GPT-3 13B 量级，$d=5120$，$L=40$。
$12\times40\times5120^2=12\times40\times2.62\times10^7\approx1.26\times10^{10}\approx12.6\text{B}$ —— 与 13B 量级吻合。

### 7.2 KV Cache 显存
$$\text{Mem}_{KV}=2\times L\times T\times d_{kv}\times \text{bytes}$$
其中 $d_{kv}=n_{kv}\cdot d_h$（MHA 时 $n_{kv}=n_h$）。

**手算（MHA，FP16）**：$L=40$，$T=2048$，$d=5120$，FP16=2 字节。
$2\times40\times2048\times5120\times2 = 3.35\times10^9\approx 3.4\text{GB（每条序列！）}$。
batch=32 → 约 **107GB**，远超单卡 → 故必须 GQA/MLA 压缩，或用 PagedAttention 管理。
若换 GQA（$g/n_h=1/8$）→ 约 13GB；若 MLA（≈1/14）→ 约 7.6GB。

### 7.3 训练显存四大块
- 参数 $P$、梯度 $P$、Adam 优化器状态 $2P$（一阶+二阶矩）。
- FP16 混合精度还要 FP32 主权重副本。
- 粗算 Adam+FP16：约 $16P$ 字节（2+2+4+4+4 主权重）。7B 模型 → 约 **112GB**，故需 ZeRO/张量并行。

### 7.4 前向 FLOPs
经验法则：**每个 token 前向约 $2P$ FLOPs**，训练（含反向）约 **$6P$**（前向 1 倍、反向 2 倍）。
**手算**：训 7B 模型、1T token → $6\times7\times10^9\times10^{12}=4.2\times10^{22}$ FLOPs。

---

## 8. 高频问题 / 追问清单

| 问题 | 踩点答案 | 陷阱 |
|---|---|---|
| 为什么除以 $\sqrt{d_k}$ | 点积方差为 $d_k$，缩放回 1 防 softmax 饱和、梯度消失 | 别只背公式，要会推方差 |
| LLM 为何不用 BatchNorm | BN 依赖 batch、变长序列+推理 batch=1 不稳；LN 逐 token 独立 | 不是「BN 不能用」而是不适配 |
| RMSNorm 比 LN 好在哪 | 去掉减均值(re-centering)，只 re-scaling，省 7%~64% 算力，效果相当 | 别说「精度更高」，是「相当且更快」 |
| Pre-Norm vs Post-Norm | Pre 好训（恒等通路）；Post 上限高但深层难训；LLM 选 Pre | 别绝对说 Pre 一定好，同深度 Post 效果更优 |
| MQA/GQA/MLA 区别 | MQA 共享 1 套 KV，GQA 分组共享，MLA 低秩压缩潜向量 | 三者都是为压 **KV Cache**，不是省算力 |
| GQA 为什么几乎不掉点 | 保留多组 KV，比 MQA 表达力强很多，又比 MHA 省 | 别说「完全无损」 |
| MLA 怎么和 RoPE 共存 | 解耦 RoPE：留一段专门维度承载位置，避免破坏低秩 | 直接旋转还原的 K 会破坏矩阵吸收 |
| RoPE 为什么是相对位置 | $R_m^\top R_n=R_{n-m}$，内积只依赖 $m-n$ | 容易答成「加了绝对位置」 |
| RoPE 怎么外推 | PI 插值 / NTK 调 base / YaRN 分频段 | 别只说「调 base」 |
| MoE 8x7B 是 56B 吗 | 不是，约 47B 总参、约 13B 激活（注意力共享） | 最常见的算错 |
| MoE 负载均衡 | 辅助损失/容量因子；DeepSeek-V3 用无辅助损失偏置法 | 不均衡会浪费专家+爆容量 |
| SwiGLU 为何隐藏维取 8/3 d | 门控多一个矩阵，缩维以对齐 $4d$ FFN 参数量 | — |
| Mistral 滑窗注意力 | 局部窗口 $w$，堆层扩感受野，$O(Tw)$ | 不是稀疏 MoE，别混 |

---

## 🔗 跳转链接

- 📍 [[00-知识地图]] —— 全局导航
- [[llm-algo/moe/README]] —— MoE 详解（路由/负载均衡/各家实现）
- [[llm-algo/旋转编码RoPE]] —— RoPE 推导与外推方案
- [[llm-algo/deepseek/DeepSeek-V2]] —— MLA 与无辅助损失负载均衡

参考阅读：
- RMSNorm 解析 https://zhuanlan.zhihu.com/p/694909672
- BN/LN 归一化对比 https://zhuanlan.zhihu.com/p/428620330 ；https://zhuanlan.zhihu.com/p/643560888
- Norm 代码实现 https://zhuanlan.zhihu.com/p/656647661
