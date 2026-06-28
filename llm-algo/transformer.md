# Transformer 架构详解

> 一句话定位：Transformer 用「自注意力 + 残差 + LayerNorm + 位置编码」彻底替换 RNN 的循环结构，让序列建模可以完全并行，是当今所有大模型的地基。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]

## 阅读地图

| 小节 | 你将学到 | 适合人群 |
| --- | --- | --- |
| 0. 一句话锚点 | Transformer 到底解决了什么问题 | 所有人 |
| 1. 地基/前置 | 张量形状、注意力直觉、为什么放弃 RNN | 入门 |
| 2. 整体架构 | Encoder-Decoder 的数据流（ASCII 图） | 入门 |
| 3. 输入表示 | Token Embedding + 位置编码（公式+手算） | 入门 |
| 4. 缩放点积注意力 | $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ 逐步推导 | 核心 |
| 5. 多头注意力 | 为什么要多个头，参数怎么切 | 核心 |
| 6. Add & Norm | 残差防退化 + LayerNorm 稳定训练 | 核心 |
| 7. 前馈网络 FFN | 逐位置 MLP，参数量大头 | 核心 |
| 8. 掩码与自回归 | Decoder 如何防止「偷看未来」 | 核心 |
| 9. 参数量与显存 | 数值估算，对接面试 | 进阶 |
| 实操/参考 | 原文权威实现与教程链接 | 动手党 |
| 常见问题/坑 | 易错点速查表 | 复习 |

## 0. 一句话锚点

> **Transformer 是「编码器－解码器架构」的一个实践，尽管在实际情况中编码器或解码器可以单独使用。**

- 只用 Encoder → BERT 类（双向理解）。
- 只用 Decoder → GPT / LLaMA / DeepSeek 类（自回归生成，当今主流大模型）。
- Encoder + Decoder → 原始论文《Attention Is All You Need》、T5、机器翻译。

核心三句话（原文要点，逐条展开）：
1. **多头自注意力**用于表示输入序列和输出序列，不过解码器必须通过**掩蔽机制**来保留自回归属性。
2. **残差连接和层规范化**是训练非常深度模型的重要工具。
3. 基于位置的**前馈网络**使用同一个多层感知机，对所有序列位置的表示进行转换。

## 1. 地基/前置

### 1.1 为什么放弃 RNN？

| 维度 | RNN/LSTM | Transformer |
| --- | --- | --- |
| 计算方式 | 时间步串行，$t$ 必须等 $t-1$ | 全序列并行，一次矩阵乘 |
| 长距离依赖 | 信息要走 $O(n)$ 步，易丢失 | 任意两位置直接 $O(1)$ 连接 |
| GPU 利用率 | 低（串行卡住） | 高（大矩阵乘满载） |
| 顺序信息 | 天生有（按时间走） | **没有**，需额外注入位置编码 |

> 关键代价：Transformer「不采用 RNN 结构，而是使用全局信息，不能利用单词的顺序信息」——所以必须显式加位置编码（见第 3 节）。

### 1.2 张量形状约定（贯穿全文）

```
B  = batch size（批大小）
L  = sequence length（序列长度）
d_model = 模型隐藏维度（如 512 / 768 / 4096）
h  = 注意力头数（如 8 / 12 / 32）
d_k = d_v = d_model / h（每个头的维度）
d_ff = FFN 中间维度（通常 = 4 * d_model）
```

输入张量 `X` 形状：`[B, L, d_model]`。整个 Transformer 内部张量形状**始终保持** `[B, L, d_model]`，这是「残差连接能加得起来」的前提。

## 2. 整体架构

```
                          ┌─────────── Decoder ───────────┐
   ┌──── Encoder ────┐    │                               │
   │                 │    │   Output Embedding + PosEnc    │
 Input Emb + PosEnc  │    │            │                  │
        │            │    │   ┌────────▼─────────┐         │
   ┌────▼──────┐     │    │   │ Masked Multi-Head │ ←掩码  │
   │ Multi-Head │    │    │   │   Self-Attention  │        │
   │ Self-Attn  │    │    │   └────────┬─────────┘         │
   └────┬───────┘    │    │       Add & Norm               │
    Add & Norm       │    │   ┌────────▼─────────┐         │
   ┌────▼──────┐     │    │   │  Cross-Attention  │◄───────┼── Encoder 输出 (K,V)
   │   FFN     │     │ ───┼──►│ (Q来自Decoder)    │         │
   └────┬──────┘     │    │   └────────┬─────────┘         │
    Add & Norm       │    │       Add & Norm               │
        │  ×N 层      │    │   ┌────────▼─────────┐         │
        └─────────────┘    │   │       FFN         │  ×N层  │
                           │   └────────┬─────────┘         │
                           │       Add & Norm               │
                           │            │                  │
                           │     Linear + Softmax → 词表概率 │
                           └───────────────────────────────┘
```

每个 Encoder 层 = `Self-Attn → Add&Norm → FFN → Add&Norm`，堆 N 层（原论文 N=6）。
每个 Decoder 层比 Encoder 多一个 **Cross-Attention**（Q 来自解码器，K/V 来自编码器输出），实现「翻译时对齐源句」。GPT 类纯 Decoder 模型**去掉了 Cross-Attention**，只保留 Masked Self-Attn + FFN。

## 3. 输入表示

> **Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。** 因为 Transformer 使用全局信息、不能利用单词顺序，而顺序对 NLP 非常重要，所以用位置 Embedding 保存单词在序列中的相对或绝对位置。
>
> **位置 Embedding 用 PE 表示，PE 的维度与单词 Embedding 相同。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者。**

### 3.1 正余弦位置编码公式

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- `pos`：单词在序列中的位置（0,1,2,…）。
- `i`：维度索引（0 ≤ i < d_model/2）；偶数维用 sin，奇数维用 cos。
- 分母随 `i` 增大 → 波长从 $2\pi$ 增长到 $10000\cdot2\pi$，不同维度编码不同「频率」的位置信息。

最终输入：`X = TokenEmbedding(token) + PE(pos)`，二者直接相加（形状都是 `[L, d_model]`）。

### 3.2 数值手算（d_model = 4，pos = 1）

| 维度 i | 频率 $10000^{2i/4}$ | 角度 $1/\text{freq}$ | 函数 | PE 值 |
| --- | --- | --- | --- | --- |
| 0 (2i=0) | $10000^0=1$ | 1.0 | sin | 0.841 |
| 1 (2i=0) | 1 | 1.0 | cos | 0.540 |
| 2 (2i=2) | $10000^{0.5}=100$ | 0.01 | sin | 0.010 |
| 3 (2i=2) | 100 | 0.01 | cos | 1.000 |

→ `PE(1) = [0.841, 0.540, 0.010, 1.000]`。低维变化快（区分近邻），高维变化慢（区分远距离）。

> 演进：原始正余弦是绝对位置编码；现代大模型多用 **RoPE 旋转位置编码**（把位置信息编进 Q/K 的旋转角，天然支持相对位置 + 外推）→ 见 [[llm-algo/旋转编码RoPE]]。

## 4. 缩放点积注意力（Scaled Dot-Product Attention）

这是整个 Transformer 的心脏。

### 4.1 公式

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

逐步拆解（原子化）：
1. **生成 Q/K/V**：输入 `X[L,d]` 分别乘三个权重矩阵 → `Q = XW_Q`, `K = XW_K`, `V = XW_V`，形状都是 `[L, d_k]`。
2. **打分**：`scores = Q·Kᵀ`，形状 `[L, L]`，第 `(i,j)` 个元素 = 位置 i 的 query 与位置 j 的 key 的相似度。
3. **缩放**：除以 $\sqrt{d_k}$。**为什么？** 当 $d_k$ 大时，点积方差 ≈ $d_k$，数值变大会把 softmax 推到饱和区（梯度趋 0）。除以 $\sqrt{d_k}$ 把方差拉回 1，梯度健康。
4. **softmax**：对每一行归一化 → 得到「位置 i 该关注每个位置多少」的权重，和为 1。
5. **加权求和**：`权重 · V` → 每个位置得到一个上下文向量 `[L, d_k]`。

### 4.2 ASCII 数据流

```
X[L,d] ──┬──×W_Q──► Q[L,dk] ─┐
         ├──×W_K──► K[L,dk] ─┴─► QKᵀ[L,L] ──÷√dk──► softmax(行) ──┐
         └──×W_V──► V[L,dk] ───────────────────────────────────×─┴─► out[L,dk]
```

### 4.3 缩放的数值对比（为什么不能省）

假设 $d_k=64$，Q、K 各元素 ~ N(0,1)：
- 点积期望幅度 ≈ $\sqrt{64}=8$。两个分数 8 与 2，`softmax` → $[0.9975, 0.0025]$，几乎 one-hot，反向传播梯度极小。
- 除以 $\sqrt{64}=8$ 后变为 1 与 0.25，`softmax` → $[0.68, 0.32]$，梯度正常。

## 5. 多头注意力（Multi-Head Attention）

### 5.1 为什么要「多头」

单个注意力只能学一种「关注模式」。多头 = 并行跑 `h` 套独立的 Q/K/V 投影，每个头在**不同子空间**捕捉不同关系（有的关注语法、有的关注指代、有的关注长距离）。

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O,\quad \text{head}_i=\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 5.2 维度切分（关键：参数量不变）

```
d_model = 512, h = 8  →  每个头 d_k = 512/8 = 64
┌──────────────── d_model = 512 ────────────────┐
│ head0 │ head1 │ head2 │ ... │ head7 │  (每段64)
└───64──┴───64──┴───64──┴─────┴───64──┘
   各自独立做 Attention，再 Concat 回 512，过 W^O 混合
```

> 设计精髓：`d_k = d_model / h`，所以 8 个头的总计算量 ≈ 1 个 512 维头，**用同样的算力换来多视角**。

## 6. Add & Norm 层

> **Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。**

### 6.1 残差连接：为什么防退化

$$\text{output} = \mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$$

- 子层只需学「残差」$\Delta = \text{SubLayer}(x)$，即使 $\Delta\approx0$，信息也能原样穿过 → 深层网络不会因为某层失效而崩。
- 梯度有「高速公路」$x$ 直达底层，缓解梯度消失，才能堆 24/48/100+ 层。

### 6.2 LayerNorm vs BatchNorm（对照表）

| 维度 | BatchNorm | LayerNorm（Transformer 用） |
| --- | --- | --- |
| 归一化方向 | 跨样本（batch 维） | 跨特征（每个 token 自己的 d_model 维） |
| 依赖 batch 大小 | 是，小 batch 不稳 | 否 |
| 变长序列/NLP | 不友好 | 天然适配 |
| 推理时 | 需存全局统计量 | 即算即用 |

$$\mathrm{LayerNorm}(x) = \gamma\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\quad \mu,\sigma\text{ 在最后一维 }d_{model}\text{ 上算}$$

> 演进：现代大模型常用 **Pre-Norm**（`x + SubLayer(LayerNorm(x))`，训练更稳）和 **RMSNorm**（只除以均方根、省去减均值，LLaMA 用），但思想同源。

## 7. 基于位置的前馈网络（Position-wise FFN）

> **Transformer 模型中基于位置的前馈网络使用同一个多层感知机，作用是对所有序列位置的表示进行转换。**

$$\mathrm{FFN}(x) = \max(0,\ xW_1+b_1)\,W_2+b_2$$

- 两层线性 + 中间激活（原论文 ReLU，现代多用 GELU/SwiGLU）。
- **「逐位置」**：同一个 MLP 独立作用于每个 token，token 之间不交互（交互在注意力里完成）。
- 维度变化：`d_model → d_ff(通常 4×) → d_model`，先升维「展开特征」再降维「压缩」。

```
[L, 512] ──W1──► [L, 2048] ──ReLU──► [L, 2048] ──W2──► [L, 512]
```

> 参数量大头：FFN 两个矩阵 ≈ $2\times d_{model}\times d_{ff} = 2\times512\times2048\approx2.1\text{M}$/层，约占每层参数的 2/3。这也是 **MoE** 替换 FFN 为多专家、稀疏激活省算力的下手点 → 见 [[llm-algo/moe/README]]。

## 8. 掩码与自回归属性

Decoder 在训练时能看到整句答案，但生成时只能看到「已生成的部分」。为保证训练与推理一致，必须用 **因果掩码（Causal Mask）** 屏蔽未来位置：

```
scores 矩阵 (L=4)，✓=可见  ✗=屏蔽(置 -∞ → softmax后≈0)
        k0  k1  k2  k3
  q0  [  ✓   ✗   ✗   ✗ ]
  q1  [  ✓   ✓   ✗   ✗ ]
  q2  [  ✓   ✓   ✓   ✗ ]
  q3  [  ✓   ✓   ✓   ✓ ]   ← 位置 i 只能看 ≤ i
```

实现：在 softmax 前把上三角部分加上 $-\infty$（或 -1e9），softmax 后这些权重变 0。这就是「解码器必须通过掩蔽机制来保留自回归属性」的含义。

> 推理优化：自回归逐 token 生成时，前面 token 的 K/V 可缓存复用，避免重复计算 → 见 [[llm-optimizer/kv-cache]]；注意力本身的 IO 优化见 [[llm-optimizer/FlashAttention]]。

## 9. 参数量与显存估算（对接面试）

单个 Transformer 层（d=d_model, d_ff=4d）参数量速算：
- 注意力 Q/K/V/O 四个矩阵：$4d^2$
- FFN 两个矩阵：$2\cdot d\cdot4d = 8d^2$
- 合计 ≈ $12d^2$/层（忽略 bias 与 LayerNorm）。

例：d=4096，则单层 ≈ $12\times4096^2 \approx 2\times10^8$ = 2 亿参数；32 层 ≈ 64 亿，加 Embedding/词表即 7B 量级。详细显存（参数+激活+KV Cache+优化器状态）拆解见 [[docs/transformer内存估算]] 与 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

## 实操 / 权威实现与参考

> 以下为原文保留的高价值学习资源，按「先读原理 → 再读代码」顺序排列。

**原理图解（中文）**
- Transformer 模型详解（图解最完整版）：https://zhuanlan.zhihu.com/p/338817680
- OpenAI ChatGPT（一）：十分钟读懂 Transformer：https://zhuanlan.zhihu.com/p/600773858

**源码精读（中文）**
- OpenAI ChatGPT（一）：Tensorflow 实现 Transformer：https://zhuanlan.zhihu.com/p/603243890
- GPT（一）transformer 原理和代码详解：https://zhuanlan.zhihu.com/p/632880248
- Transformer 源码详解（Pytorch 版本）：https://zhuanlan.zhihu.com/p/398039366
- 搞懂 Transformer 结构，看这篇 PyTorch 实现就够了：https://zhuanlan.zhihu.com/p/339207092

**权威英文实现（强烈推荐逐行跑一遍）**
- 哈佛 annotated-transformer（带注释的完整 Notebook）：https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb
- Overview: The Implemented Transformer：https://medium.com/@hunter-j-phillips/overview-the-implemented-transformer-eafd87fe9589
- Multi-Head Attention：https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a
- Layer Normalization：https://medium.com/@hunter-j-phillips/layer-normalization-e9ae93eb3c9c
- Positional Encoding：https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6

## 常见问题 / 坑

| 现象 / 疑问 | 原因 | 对策 |
| --- | --- | --- |
| 训练 loss 不降、梯度近 0 | 忘了 $\sqrt{d_k}$ 缩放，softmax 饱和 | 必须除以 $\sqrt{d_k}$ |
| 生成时模型「偷看未来」、loss 异常低但推理差 | 没加因果掩码 | Decoder 自注意力上三角置 $-\infty$ |
| 位置打乱后结果不变 | 没加位置编码（自注意力对顺序不敏感） | 输入加 PE / 用 RoPE |
| 残差加不起来、形状报错 | 子层改变了 d_model | 保持全程 `[B,L,d_model]` 不变 |
| 深层网络发散/NaN | Post-Norm 深层不稳 | 改 Pre-Norm、warmup、梯度裁剪 |
| 多头但效果没提升 | head 数过多导致 $d_k$ 太小、表达力不足 | 平衡 h 与 d_k（保证 d_k≥32） |
| 长文本显存爆 | 注意力 $O(L^2)$ + KV Cache 线性增长 | FlashAttention + KV Cache + 量化 |
| BatchNorm 用在 NLP 效果差 | 变长序列、小 batch 统计不稳 | 用 LayerNorm/RMSNorm |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 架构同源：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 推理优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测/估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
