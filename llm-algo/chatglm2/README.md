# ChatGLM2

> ChatGLM2 是清华 THUDM（智谱）2023 年发布的第二代开源中英双语对话大模型，在第一代基础上换装 **Multi-Query Attention（MQA）+ FlashAttention + RoPE + SwiGLU**，把上下文从 2K 拉到 32K，并显著提升推理效率与中文能力。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/chatglm/README]] [[llm-algo/chatglm3/README]]

> 参考实现：https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py
> 说明：本文规格数字标"约/以官方为准"，版本/CLI/API 以官方文档为准，不编造未知精确数字。

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点：ChatGLM2 到底改了什么 | 二代/换骨 |
| 1 | 地基：Transformer Decoder、Attention、KV Cache | 前置概念 |
| 2 | 相对 ChatGLM1 的关键改动总览 | 对照表 |
| 3 | 注意力：MQA（不是严格 GQA）怎么省显存 | MQA/KV Cache |
| 4 | FlashAttention：为什么能算更长上下文 | IO 感知/分块 |
| 5 | 位置编码：RoPE 旋转位置编码 | 长度外推 |
| 6 | 前馈网络：SwiGLU 与 SiLU 激活 | 门控 FFN |
| 7 | 归一化：RMSNorm 与 Post→Pre 结构 | 训练稳定 |
| 8 | 长上下文：2K→32K 是怎么做到的 | 训练+外推 |
| 9 | 中文优化：词表/数据/对齐 | 双语 |
| 10 | 训练流程：预训练 + 对齐 | 1.4T tokens |
| ★ | 数值例子：MQA 省了多少 KV Cache | 手算 |
| Q | 常见问题 | FAQ |

---

## 0. 一句话锚点

**一句话**：ChatGLM2 = ChatGLM1 的"换骨升级"——保留 GLM 的训练范式（自回归填空），但把网络内部的注意力、激活、位置编码、归一化全部换成当时主流大模型（LLaMA 系）的高效组件，从而做到 **更强（1.4T 训练）、更长（32K 上下文）、更快（MQA 推理）**。

记住三个数字（约/以官方为准）：

```
        ChatGLM-6B            ChatGLM2-6B
        ┌──────────┐         ┌──────────┐
上下文  │   2K     │  ─────► │  32K     │  (对话训练 8K)
推理    │  MHA     │  ─────► │  MQA     │  KV Cache 大幅下降
训练    │ ~1T tok  │  ─────► │ 1.4T tok │  + 人类偏好对齐
        └──────────┘         └──────────┘
```

> 注意一个常见误解：很多资料（包括本仓库标题）写"GQA"，但 ChatGLM2-6B 官方用的是 **MQA（Multi-Query Attention）**，可看作 GQA 在 `分组数=1` 时的极端特例。第 3 节会把 MHA / MQA / GQA 的关系讲透。

---

## 1. 地基/前置（不假设你记得）

把每个原子概念先讲清，后面才看得懂改动"为什么"。

### 1.1 Decoder-only Transformer 一层在做什么

大模型主体是 $L$ 层堆叠，每层两个子模块：

```
  输入 x (seq_len × d_model)
        │
   ┌────▼─────┐
   │ Attention │  ← token 之间"看彼此"
   └────┬─────┘
        │ + 残差
   ┌────▼─────┐
   │   FFN    │  ← 每个 token 各自做非线性变换
   └────┬─────┘
        │ + 残差
        ▼  输出
```

### 1.2 注意力（Attention）的本质

每个 token 生成三个向量：Query $Q$（我想找什么）、Key $K$（我能被什么找到）、Value $V$（我携带的信息）。注意力公式：

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $QK^\top$：每个 query 与所有 key 的相似度（点积）。
- $\sqrt{d_k}$：缩放，防止点积过大导致 softmax 梯度消失。
- softmax：把相似度变成"权重和为 1"的注意力分布。
- 乘 $V$：按权重把信息加权汇总。

### 1.3 多头（Multi-Head）

把 $d_{model}$ 切成 $h$ 个"头"，每头独立做注意力再拼接，让模型从多个子空间看关系。**标准多头注意力（MHA）里，Q/K/V 的头数相等，都是 $h$。**

### 1.4 KV Cache（推理为什么慢/吃显存）

生成式推理是逐 token 自回归。生成第 $t$ 个 token 时，需要前面所有 token 的 $K,V$。为避免重复计算，把它们缓存起来——这就是 **KV Cache**。它的显存占用为：

$$\text{KVCache} = 2 \times L \times h_{kv} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{bytes}$$

其中 `2` 是 K 和 V 各一份，$h_{kv}$ 是 **KV 头数**。**减少 $h_{kv}$ 就能成倍减少 KV Cache** —— 这正是 MQA 的核心动机（见第 3 节）。

---

## 2. 相对 ChatGLM1 的关键改动总览

这是本文的"骨架对照表"。后面每一节展开一行。

| 维度 | ChatGLM-6B（一代） | ChatGLM2-6B（二代） | 为什么改 |
|------|--------------------|---------------------|----------|
| 注意力 | MHA（多头，K/V 头数 = Q 头数） | **MQA**（多个 Q 头共享 1 组 K/V） | KV Cache 小、推理快 |
| 注意力实现 | 普通实现 | **FlashAttention** | 省显存、能算长序列 |
| 位置编码 | 2D RoPE（GLM 特殊的二维位置） | **常规 RoPE**（旋转位置编码，位置/写法调整） | 简化、利于长度外推 |
| 激活函数 | GELU | **SwiGLU（SiLU 门控）** | 表达力更强、收敛更好 |
| 归一化 | LayerNorm（含 Post-LN 成分） | **RMSNorm**（Pre-Norm 结构） | 更稳更快 |
| 上下文 | 2K | **32K**（对话训练 8K） | 长文档/长对话 |
| 训练量 | 约 1T tokens | **约 1.4T tokens** + 人类偏好对齐 | 性能更强 |
| 拼接格式 | `[gMASK]` + 特定 prompt | 多轮对话格式优化 | 对话体验 |

> 仓库末尾原话"激活函数不同、RotaryEmbedding 位置不同"对应表里的第 3、4 行；本文把它们补全为完整改动清单。

```
        一代 (MHA, GELU, LN, 2K)
                 │  换骨
                 ▼
        二代 (MQA, SwiGLU, RMSNorm, RoPE, FlashAttn, 32K)
                 │  渐进
                 ▼
        三代 ChatGLM3 (更强对齐/工具调用)  → [[llm-algo/chatglm3/README]]
```

---

## 3. 注意力升级：MQA（不是严格 GQA）

### 3.1 MHA → MQA → GQA 一张图看懂

核心区别只在一处：**K/V 的头数**。

```
MHA (多头注意力)            MQA (多查询注意力)        GQA (分组查询注意力)
8 个 Q 头                   8 个 Q 头                  8 个 Q 头
8 个 K/V 头                 1 组 K/V 头(共享)          2 组 K/V 头(分组共享)

Q1 Q2 ... Q8               Q1 Q2 ... Q8               Q1..Q4   Q5..Q8
│  │      │                 \  \    /  /               \  |  /   \ | /
K1 K2 ... K8                 \  \  /  /                  KV组1     KV组2
V1 V2 ... V8                  (KV共享1组)
KV Cache 最大                KV Cache 最小              KV Cache 折中
质量基准                     质量略降                   质量≈MHA(常用)
```

- **MHA**：$h_q = h_{kv}$，质量最好但 KV Cache 最大。
- **MQA**：$h_{kv}=1$，所有 query 头共享同一组 K/V，KV Cache 缩到 $1/h$，推理最快。**ChatGLM2-6B 用的就是 MQA。**
- **GQA**：$1 < h_{kv} < h_q$，把 query 头分成 $g$ 组，每组共享一组 K/V，是 MHA 和 MQA 的折中（LLaMA-2 70B 等用的是这个）。MQA = GQA 在 $g=1$ 的特例。

### 3.2 为什么 MQA 几乎不掉质量却大省显存

直觉：K/V 主要承担"被检索"的角色，多个 query 头共享一组 K/V，只是少了 K/V 的多样性，而 Q 仍是多头（保留多视角检索能力）。实践中质量损失很小，但 KV Cache 直接除以头数。

```
推理时单层 KV Cache 对比 (示意, 设 h=32):
MHA : ████████████████████████████████  (32 份 K/V)
MQA : █                                  (1  份 K/V)  → 约 1/32
```

> 工程收益：KV Cache 小 → 同样显存能放更长序列 / 更大 batch → 吞吐更高、显存峰值更低。这就是"更高效的推理"的来源。

---

## 4. FlashAttention：为什么能算更长上下文

### 4.1 普通注意力的瓶颈是"显存"，不是"算力"

注意力要算并存下 $S=QK^\top$ 这个 $\text{seq}\times\text{seq}$ 的大矩阵。序列长 $n$ 时它是 $O(n^2)$，显存随 $n^2$ 爆炸。32K 上下文意味着 $32000^2 \approx 10^9$ 个分数 —— 直接存不下。

### 4.2 FlashAttention 的核心思想：分块 + 不落地

FlashAttention 把 Q/K/V 切成小块（tile），在 GPU 的高速 SRAM 里**分块计算并用 online-softmax 增量累加**，从不把完整的 $n\times n$ 矩阵写回慢速显存（HBM）。

```
普通: Q×K^T → [n×n 大矩阵写回 HBM] → softmax → ×V   (IO 爆炸)

Flash: 
  for K/V 块 j:
    for Q 块 i:
       局部 = Qi × Kj^T            ← 在 SRAM 里
       online-softmax 增量更新 Oi  ← 不存完整 S
  ───────────────────────────────
  从不物化 n×n 矩阵，HBM 读写 O(n) 而非 O(n²)
```

- **数学结果完全一致**（精确注意力，非近似），只是计算顺序/内存布局变了。
- 收益：显存从 $O(n^2)$ 降到约 $O(n)$，速度也更快（瓶颈本来就是 IO）。
- 正因如此，基座才能在 32K 长度上训练/推理。

---

## 5. 位置编码：RoPE（旋转位置编码）

### 5.1 问题：Attention 本身"看不到顺序"

点积注意力对 token 是"置换不变"的，必须显式注入位置信息。

### 5.2 RoPE 的做法：把"位置"变成"旋转角度"

RoPE 不给 Q/K 加位置向量，而是按位置 $m$ 把 Q/K 的二维子向量**旋转一个角度** $m\theta$：

$$\langle R_m q,\; R_n k\rangle = f(q,k,\,m-n)$$

```
位置0:  →        位置1:  ↗        位置2:  ↑     (同一向量被旋转不同角度)
两个 token 的注意力只依赖它们的"相对角度差" (m−n)
```

- 关键性质：内积只依赖**相对位置** $m-n$，天然支持相对位置语义。
- 这对**长度外推**友好——配合插值等技巧，更容易从短上下文泛化到长上下文（32K 的基础之一）。
- 相对一代：ChatGLM1 用的是 GLM 特有的 2D 位置编码；二代调整为更常规的 RoPE 写法与作用位置（这正是仓库原话"RotaryEmbedding 位置不同"）。

---

## 6. 前馈网络：SwiGLU 与 SiLU 激活

### 6.1 SiLU（即 Swish）

$$\text{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}$$

形状像"平滑版 ReLU"，负区不硬截断为 0，梯度更平滑。代码里就是 `F.silu`（对应仓库"说明"里点到的 `F.silu`）。

### 6.2 SwiGLU：带门控的 FFN

普通 FFN：$\text{FFN}(x)=W_2\,\phi(W_1 x)$。SwiGLU 把第一层拆成两路，一路当"门"：

$$\text{SwiGLU}(x)=W_2\big(\;\text{SiLU}(W_a x)\;\odot\;(W_b x)\;\big)$$

```
        x
       ╱ ╲
   W_a       W_b
    │         │
  SiLU        │      ← 一路过激活当"开关"
    │         │
    └──── ⊙ ──┘      ← 逐元素相乘(门控)
         │
        W_2
         │
        out
```

- $\odot$ 是逐元素乘。门控让网络"动态决定让哪些信息通过"，表达力更强、收敛更好。
- 相对一代的 GELU，SwiGLU 是当时大模型（PaLM/LLaMA）验证有效的升级。

---

## 7. 归一化：RMSNorm 与 Pre-Norm

### 7.1 LayerNorm vs RMSNorm

LayerNorm 要减均值再除标准差；RMSNorm **省掉减均值**，只用均方根缩放：

$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}\cdot g$$

- $g$ 是可学习的缩放参数；省掉去均值，计算更省、经验上同样稳。
- 对应仓库"说明"里点到的 `RMSNorm`。

### 7.2 Post-Norm → Pre-Norm

把 Norm 放在残差**之前**（Pre-Norm），让残差通路是"干净的恒等映射"，深层网络训练更稳、更易收敛。

```
Post-Norm:  x ─►[子层]─►(+x)─►[Norm]      深了易不稳
Pre-Norm :  x ─►[Norm]─►[子层]─►(+x)      残差更干净，更稳  ← 二代
```

---

## 8. 长上下文：2K → 32K 是怎么做到的

不是单点魔法，而是几件事叠加：

```
┌─ RoPE          → 相对位置 + 利于长度外推
├─ FlashAttention→ 显存 O(n²)→O(n)，长序列才算得动
├─ 长序列训练    → 基座在长上下文上预训练
└─ 32K 专版      → 另发布 ChatGLM2-6B-32K（位置插值等增强长程）
```

- 基座上下文：2K → **32K**（约/以官方为准）。
- 对话阶段：用 **8K** 上下文训练（覆盖绝大多数对话场景，兼顾效率）。
- 超长需求：官方另发布 **ChatGLM2-6B-32K** 变体，针对 32K 做了进一步增强。

---

## 9. 中文优化

ChatGLM 系列定位是"中英双语、对中文友好"，二代延续并强化：

- **训练语料**：约 1.4T 的**中英标识符**预训练，中文占比高，覆盖中文知识与语感（约/以官方为准）。
- **词表/分词**：面向中文友好的词表设计，减少中文被切碎，提高编码效率与中文表现。
- **对齐**：在中文对话/指令上做监督微调 + 人类偏好对齐，使中文问答、写作、推理更贴合人类预期。
- **混合目标**：沿用 GLM 的混合目标函数（自回归空白填充），兼顾理解与生成。

> 与纯英文为主的同期模型相比，ChatGLM2 在中文榜单与实际中文对话上有明显优势，是国内"开箱即用"的代表性中文基座之一。

---

## 10. 训练流程

```
┌─────────────────────────────────────────────┐
│ ① 预训练 (Pretrain)                          │
│   GLM 混合目标(自回归填空) on ~1.4T 中英 tok  │
│   基座上下文 2K→32K                          │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ ② 监督微调 (SFT)                             │
│   多轮对话/指令数据，对话上下文 8K            │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ ③ 人类偏好对齐 (RLHF/偏好优化)               │
│   让回答更有用、更安全、更符合人类偏好        │
└─────────────────────────────────────────────┘
```

- ① **预训练**：GLM 混合目标 + 约 1.4T tokens（约/以官方为准）。
- ② **SFT**：对话/指令数据，8K 上下文。
- ③ **对齐**：人类偏好对齐，提升有用性与安全性。

---

## ★ 数值例子：MQA 到底省了多少 KV Cache（手算）

用第 1.4 节公式估算（数字为教学示意，结构以官方为准）。设：层数 $L=28$、隐藏维 $d=4096$、头维 $d_{head}=128$、查询头数 $h_q=32$，序列 $n=8192$、batch=1、FP16（2 字节）。

KV Cache 公式：$2 \times L \times h_{kv} \times d_{head} \times n \times \text{bytes}$。

**MHA（$h_{kv}=h_q=32$）：**

$$2\times28\times32\times128\times8192\times2 \approx 3.0\times10^{9}\ \text{字节}\approx 2.7\ \text{GB}$$

**MQA（$h_{kv}=1$）：**

$$2\times28\times1\times128\times8192\times2 \approx 9.4\times10^{7}\ \text{字节}\approx 0.09\ \text{GB}$$

```
KV Cache (8K上下文, 示意):
MHA : ████████████████████████████████  ~2.7 GB
MQA : █                                  ~0.09 GB   → 约 1/32
```

**结论**：MQA 把 KV Cache 缩到约 $1/h_q$（这里约 1/32）。同样显存下，要么放下 32 倍长的序列，要么开 32 倍大的 batch —— 这就是"更高效推理 + 更低显存"的来源。把上下文拉到 32K 时，这种节省更关键。

> 提醒：上面是示意手算，真实 ChatGLM2-6B 的层数/头数/维度请以官方 `modeling_chatglm.py` 与 config 为准。

---

## 常见问题

| 问题 | 解答 |
|------|------|
| ChatGLM2 用的是 GQA 还是 MQA？ | 官方 ChatGLM2-6B 用 **MQA**（$h_{kv}=1$）。MQA 是 GQA 在分组数=1 的特例，所以叫"GQA"不算大错，但精确说法是 MQA。 |
| MQA 会不会掉质量？ | 会有极小损失，但因 Q 仍多头、K/V 主要负责被检索，实测质量几乎不降，换来巨大显存/速度收益。 |
| FlashAttention 是近似算法吗？ | 不是。结果与标准注意力**数学等价**，只是分块计算、不物化 $n\times n$ 矩阵，省显存提速度。 |
| 2K 到 32K 是单靠 FlashAttention 吗？ | 不是。是 RoPE（相对位置/外推）+ FlashAttention（算得动）+ 长序列训练 + 32K 专版插值等共同作用。 |
| 对话为什么只训 8K？ | 8K 覆盖绝大多数对话场景，兼顾效果与效率；超长需求用 ChatGLM2-6B-32K 变体。 |
| 二代和一代最核心的差别？ | 内部组件全面换成高效现代组件（MQA/FlashAttn/RoPE/SwiGLU/RMSNorm），训练范式仍是 GLM 混合目标。 |
| `F.silu` 和 `RMSNorm` 在哪用？ | SiLU 用于 SwiGLU 前馈门控；RMSNorm 用作各子层的 Pre-Norm 归一化。 |
| 数字可信吗？ | 上下文 2K/8K/32K、约 1.4T tokens 等为官方口径概数，精确结构以官方仓库为准。 |

---

## 🔗 跳转链接

- 知识地图（总览）：[[00-知识地图]]
- 上一代（理解演进起点）：[[llm-algo/chatglm/README]]
- 下一代（继续演进/工具调用）：[[llm-algo/chatglm3/README]]
- 官方参考实现：https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py
