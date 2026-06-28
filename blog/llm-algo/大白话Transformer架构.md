# 大白话 Transformer 架构

> 用最朴素的比喻把 Transformer 拆到原子级：注意力是"查字典 + 加权投票"，FFN 是"逐位置的小专家"，残差+LayerNorm 是"高速公路+稳压器"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[docs/transformer内存估算]]

## 阅读地图

| 你想知道 | 看哪节 | 一句话结论 |
|---|---|---|
| Transformer 为啥取代 RNN | §1 地基 | 并行 + 长依赖，把"串行递归"换成"全连接注意力" |
| 注意力到底在算什么 | §2 §3 | Q 问、K 答、V 取值；相似度→softmax→加权和 |
| 为啥要多头 | §4 | 多个子空间并行"看不同的关系" |
| 为啥要除以 $\sqrt{d_k}$ | §3 / 公式节 | 防止点积过大导致 softmax 饱和、梯度消失 |
| FFN/残差/LayerNorm 各干啥 | §5 §6 §7 | 非线性升维 / 高速公路 / 稳压器 |
| 位置信息怎么来的 | §8 | 注意力本身无序，靠位置编码注入 |
| 一层堆起来长啥样 | §9 | 数据流全景 ASCII 图 |
| 参数量/显存/FLOPs 怎么算 | 数值示例节 | 给出可手算公式 |
| 编码器/解码器/Decoder-only 区别 | 评价节 | 现代 LLM 多是 Decoder-only |

## 0. 一句话锚点

**Transformer = 注意力(让每个词去"看"全文相关的词) + 前馈网络(逐位置做非线性变换) + 残差与归一化(让深层网络稳定可训) + 位置编码(补回顺序信息)，整体抛弃递归与卷积、纯靠注意力做序列建模。** 这一句话出自论文标题《Attention Is All You Need》(2017)。

## 1. 地基：为什么需要 Transformer（问题背景）

在 Transformer 之前，序列建模主要靠 **RNN/LSTM/GRU**。它们有两个致命痛点：

1. **串行依赖、无法并行**：第 $t$ 步的隐状态 $h_t$ 必须等 $h_{t-1}$ 算完才能算。一句 1000 词的句子要顺序走 1000 步，GPU 的并行算力被浪费。
2. **长距离依赖衰减**：信息要"逐跳传递"，相隔很远的两个词之间隔着几百次矩阵乘和非线性，梯度容易消失/爆炸，"记不住"远处的词。

CNN（如 ConvSeq2Seq）能并行，但单层卷积感受野有限，要堆很多层才能让远处的词"互相看见"。

**Transformer 的解法**：用 **自注意力(self-attention)** 让序列里**任意两个位置一步直连**——第 5 个词想看第 800 个词，距离永远是"1 跳"。同时所有位置的计算可以打包成大矩阵乘法，**天然并行**。

```
RNN：信息像接力跑，远处要传很多棒
  词1 → 词2 → 词3 → ... → 词800   (800 步串行, 长路衰减)

Self-Attention：信息像开全员视频会议，谁都能直接对话
  词1 ─┐
  词2 ─┼─►  每个词同时"看见"所有词  (1 跳直连, 全并行)
  ...  │
  词800─┘
```

代价：注意力是 $O(n^2)$ 复杂度（$n$ 个词两两都要算关系），序列很长时开销大——这正是后续 FlashAttention、KV-Cache、稀疏注意力要优化的地方（见 [[llm-optimizer/FlashAttention]]、[[llm-optimizer/kv-cache]]）。

## 2. 注意力的直觉：查字典 + 加权投票

把注意力想成"**带模糊匹配的字典查询**"：

- **Query(Q，查询)**：我现在这个词，想问什么？（"我后面该接什么？"）
- **Key(K，键)**：每个候选词的"标签/索引"，用来跟 Query 匹配。
- **Value(V，值)**：每个候选词真正携带的"内容信息"。

流程：拿 Query 去和**每一个** Key 比相似度 → 相似度越高，说明那个词越相关 → 把相似度归一化成权重（加起来=1）→ 用这些权重对所有 Value 做加权求和，得到输出。

```
            "the cat sat on the ___"  当前词 Query: "on"
   ┌──────────────────────────────────────────────┐
   │  和每个词的 Key 比相似度(打分):                 │
   │     the:0.1  cat:0.6  sat:0.9  on:0.2  the:0.1 │
   │  softmax 归一化成权重(求和=1):                  │
   │     0.05    0.22     0.55    0.10    0.05  ...  │
   │  按权重把各词的 Value 加权混合 → 输出向量        │
   └──────────────────────────────────────────────┘
   含义: "on" 主要参考了 "sat" 和 "cat"，更可能接一个地点名词
```

**自注意力(self-attention)** 就是 Q、K、V 都来自**同一句话自己**——句子内部互相"打分投票"。

## 3. 注意力的数学：一行公式说清

标准的**缩放点积注意力(Scaled Dot-Product Attention)**：

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

逐项拆解（设序列长 $n$，每个向量维度 $d_k$）：

1. **$QK^\top$**：形状 $[n,d_k]\times[d_k,n]=[n,n]$。第 $(i,j)$ 个元素是"第 $i$ 个词的 Query 和第 $j$ 个词的 Key 的点积"，即**相似度打分**。
2. **$/\sqrt{d_k}$（缩放）**：为什么要除？若 Q、K 各维独立、均值 0 方差 1，则点积 $\sum_{k=1}^{d_k} q_k k_k$ 的方差约为 $d_k$，维度越大数值越大。数值过大会让 softmax **进入饱和区**——一个值独大、其余接近 0，梯度几乎为 0，训练学不动。除以 $\sqrt{d_k}$ 把方差拉回约 1，softmax 保持"温和"。
3. **softmax**：对每一行做归一化，把打分变成"加起来=1"的权重（注意力分布）。
4. **$\times V$**：$[n,n]\times[n,d_v]=[n,d_v]$，用权重对 Value 加权求和，得到每个位置的新表示。

```
   Q[n×d]      K[n×d]
     │           │
     └──► QKᵀ ◄──┘     得到 [n×n] 打分矩阵 (谁关注谁)
           │
        ÷ √d_k          缩放, 稳住数值
           │
        softmax(按行)    → 权重矩阵 A[n×n], 每行和=1
           │
        A · V            → 输出 [n×d], 每个位置=相关 Value 的混合
```

**Q、K、V 哪来的？** 由输入 $X$（每个词的嵌入）分别乘三个可学习权重矩阵得到：
$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$。这三个 $W$ 是模型要学的参数。

## 4. 多头注意力：开多个"分会场"

单个注意力只能学一种"关系视角"。**多头注意力(Multi-Head Attention, MHA)** 把 Q/K/V 切成 $h$ 份（每份维度 $d_k=d_{model}/h$），每份独立做一次注意力，再把结果拼接、过一个输出投影：

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\quad
\text{head}_i=\text{Attention}(XW_Q^i, XW_K^i, XW_V^i)
$$

直觉：不同的头学不同的"关系"——有的头专门看"主谓一致"，有的看"代词指代"，有的看"邻近词"。就像一群专家，每人盯一类线索，最后汇总。

```
            输入 X [n × d_model]
   ┌──────────┬──────────┬─────── ... ──────┐
   │ head 1   │ head 2   │       head h      │   每头独立做注意力
   │ 看语法   │ 看指代   │      看局部       │   (在各自子空间)
   └────┬─────┴────┬─────┴────────┬──────────┘
        └──────────┴── Concat ────┘  拼回 [n × d_model]
                       │
                     × W_O          输出投影, 融合各头
```

> 工程演进：MHA 显存/带宽开销大，推理时催生了 **MQA(多查询，多头共享一份 K/V)** 与 **GQA(分组查询，几组共享 K/V)**，大幅压缩 KV-Cache，是现代 LLM 推理优化的关键（见 [[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]）。

## 5. 前馈网络 FFN：逐位置的"小专家"

注意力负责"**词之间**混合信息"，FFN 负责"**每个位置自己**做深加工"。它对**每个位置独立**地做两次线性变换、中间夹一个非线性激活：

$$
\text{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2
$$

- 典型维度：先升维到 $d_{ff}=4\,d_{model}$（如 512→2048），激活后再降回 $d_{model}$。**先放大再压缩**，给非线性变换更大的"工作空间"。
- 激活 $\sigma$：原论文用 **ReLU**；现代 LLM 多用 **GELU / SwiGLU**（SwiGLU 用门控，效果更好，是 LLaMA 等的标配）。
- 关键点：FFN **不跨位置**，第 5 个词和第 800 个词各自过同一套 FFN、互不影响——所以可以完全并行。

```
   每个位置的向量 x [d_model]
        │
     × W1 + b1   → 升维 [d_ff = 4·d_model]   "展开思考空间"
        │
      σ(·)        → 非线性 (ReLU/GELU/SwiGLU) "挑出关键模式"
        │
     × W2 + b2   → 降维回 [d_model]           "压缩成结论"
```

> FFN 通常占了 Transformer **大头参数量**（约 2/3）。把 FFN 换成"很多个专家、每次只激活几个"就得到 **MoE(混合专家)**，在不显著增加计算的前提下放大参数容量（见 [[llm-algo/moe/README]]）。

## 6. 残差连接：给深层网络修一条"高速公路"

每个子层（注意力、FFN）的输出都不是直接往上传，而是**加回输入**：

$$
\text{output} = x + \text{Sublayer}(x)
$$

为什么？深层网络反向传播时，梯度要连乘很多层，容易消失。残差让梯度有一条"直通车道"——求导时 $\frac{\partial(x+f(x))}{\partial x}=1+f'(x)$，那个 **+1** 保证梯度不会被压成 0。同时子层只需学"**增量/修正量** $f(x)$"，比从零学整个映射更容易，也更容易学到恒等映射（什么都不改也行）。

```
    x ───────────────┐ (恒等捷径, 梯度直通)
    │                │
  Sublayer(x)        +  ──► 输出
    └────────────────┘
   "主路做加工, 旁路保底; 即使主路学坏了, 信息也能原样穿过去"
```

## 7. LayerNorm：每个词向量自己的"稳压器"

**层归一化(Layer Normalization)** 对**每个位置的特征向量自身**做归一化（沿特征维 $d_{model}$，而非沿 batch）：

$$
\text{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta,\quad
\mu=\frac1d\sum_i x_i,\ \ \sigma^2=\frac1d\sum_i (x_i-\mu)^2
$$

把向量各维拉到均值≈0、方差≈1，再用可学习的 $\gamma,\beta$ 缩放平移。作用：稳定每层输入分布，缓解梯度消失/爆炸，让训练更快更稳。**为什么用 LayerNorm 不用 BatchNorm？** 序列长度可变、batch 内样本长短不一，且推理常 batch=1，BatchNorm 的"跨样本统计"不稳定；LayerNorm 只看单个样本自己，与 batch 无关。

**Post-LN vs Pre-LN（重要演进）**：

```
  Post-LN (原论文):  x ─►[ Sublayer ]─►(+x)─►[ LayerNorm ]─► 
        归一化在残差相加之后  → 深层时训练不稳, 需小心 warmup

  Pre-LN (现代主流):  x ─►[ LayerNorm ]─►[ Sublayer ]─►(+x)─► 
        归一化在子层之前      → 梯度更稳, 容易训很深 (GPT/LLaMA 用此)
```

> 现代 LLM 还常把 LayerNorm 换成更省的 **RMSNorm**（只做均方根缩放、不减均值），如 LLaMA 系列。

## 8. 位置编码：把"顺序"补回来

注意力是**置换不变(permutation-invariant)** 的——打乱词序，注意力算出的集合是一样的，它本身**不知道谁在前谁在后**。所以必须额外注入位置信息。

- **正弦位置编码(原论文)**：用不同频率的 sin/cos 给每个位置生成一个固定向量，加到词嵌入上。好处是能外推到训练时没见过的更长序列。
  $$ PE_{(pos,2i)}=\sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\quad PE_{(pos,2i+1)}=\cos\!\Big(\frac{pos}{10000^{2i/d}}\Big) $$
- **可学习位置编码**：直接学一张"位置→向量"表（BERT/GPT-2 用）。
- **旋转位置编码 RoPE（现代主流）**：把位置信息以"旋转角度"的形式注入 Q/K，天然编码**相对位置**，长度外推友好，是 LLaMA、Qwen 等的标配。

```
   词嵌入  Embedding(token)   [d_model]
                +              ← 加上 / 旋转融入 位置信息
   位置编码  PE(position)     [d_model]
                =
         带位置信息的输入向量  → 送入第一层
```

## 9. 全景图：一个 Transformer 层的数据流

把前面所有零件拼起来（以 **Pre-LN Decoder 层** 为例，现代 LLM 主流）：

```
        输入 X  [n × d_model]
          │
    ┌─────┴──────────────────────────┐  ← 残差捷径(保留 X)
    │   LayerNorm                     │
    │   Masked Multi-Head Attention   │  词之间混合信息(带因果掩码)
    └─────┬──────────────────────────┘
          + ◄── 加回 X
          │  X1 = X + Attn(LN(X))
    ┌─────┴──────────────────────────┐  ← 残差捷径(保留 X1)
    │   LayerNorm                     │
    │   FFN (升维→激活→降维)          │  逐位置非线性加工
    └─────┬──────────────────────────┘
          + ◄── 加回 X1
          │  X2 = X1 + FFN(LN(X1))
          ▼
        输出 X2  [n × d_model]   → 堆叠 N 层 (GPT-3 有 96 层)
```

**因果掩码(causal mask)**：解码器做"预测下一个词"，第 $i$ 个位置**不能偷看**未来的 $i+1,\dots,n$。实现上在 $QK^\top$ 的上三角位置加 $-\infty$，softmax 后这些权重变 0。

**整体堆叠**：`Embedding+位置编码 → N×(注意力层+FFN层) → 最后 LayerNorm → 线性投影到词表 → softmax 出下一个词概率`。

## 关键公式 / 算法 / 数值示例

**(A) 一个 Transformer 层的核心计算（伪代码）**
```
def transformer_layer(X):          # X: [n, d_model]
    h = X + MHA(LayerNorm(X))      # 注意力子层 + 残差 (Pre-LN)
    out = h + FFN(LayerNorm(h))    # 前馈子层 + 残差
    return out
```

**(B) 参数量手算**（单层，忽略偏置/LN 的少量参数）
- 注意力：$W_Q,W_K,W_V,W_O$ 各 $d_{model}^2$ → $4\,d_{model}^2$
- FFN：$W_1$ 是 $d_{model}\times d_{ff}$、$W_2$ 是 $d_{ff}\times d_{model}$，取 $d_{ff}=4d_{model}$ → $8\,d_{model}^2$
- **单层合计 ≈ $12\,d_{model}^2$**

例：$d_{model}=4096$，$N=32$ 层 →
单层 $\approx 12\times4096^2 \approx 2.01\times10^8$ ≈ 2 亿参数；
32 层 $\approx 6.4\times10^9$ ≈ **64 亿**（再加 Embedding $V\times d_{model}$，词表 $V\approx 32000$ 时约 1.3 亿）。量级与一个 ~7B 模型相符。

**(C) 注意力计算量 FLOPs**（单头，序列 $n$）
- $QK^\top$：约 $2n^2 d_k$；$\text{softmax}\cdot V$：约 $2n^2 d_v$。
- 因此注意力随 $n$ 呈 **$O(n^2)$** 增长——这是长上下文的成本来源，催生 FlashAttention（省显存、不省 FLOPs）与稀疏/线性注意力。

**(D) KV-Cache 显存手算**（推理自回归，缓存每层每个历史 token 的 K、V）
$$ \text{KV显存} = 2\times N_{layer}\times n\times d_{model}\times \text{bytes} $$
例：$N=32$ 层、$d_{model}=4096$、序列 $n=2048$、FP16(2 字节)：
$2\times32\times2048\times4096\times2 \approx 2.1\times10^9$ 字节 ≈ **2 GB**（单条序列、单 batch）。这解释了为什么长上下文推理显存吃紧，以及 GQA/MQA、PagedAttention 为何重要（见 [[llm-inference/KV-Cache优化]]、[[llm-inference/PD分离]]、[[docs/transformer内存估算]]）。

## 评价 / 对照 / 局限

| 维度 | 说明 |
|---|---|
| **核心优势** | 全并行训练、任意距离 1 跳直连、可扩展到千亿参数 |
| **主要局限** | 注意力 $O(n^2)$，长序列昂贵；位置外推需特殊设计(RoPE/ALiBi) |
| **三种架构** | Encoder-only(BERT，理解类) / Decoder-only(GPT/LLaMA，生成类，**当今 LLM 主流**) / Encoder-Decoder(T5/原始 Transformer，翻译类) |
| **演进脉络** | MHA→GQA/MQA(省 KV)；ReLU-FFN→SwiGLU；LayerNorm→RMSNorm；Post-LN→Pre-LN；绝对位置→RoPE；稠密 FFN→MoE |
| **训练相关** | 深层稳定靠残差+Pre-LN；显存/并行见张量并行、流水并行 [[llm-train/README]] |

**对照一句话记忆**：
- **注意力** = 词与词之间"开会投票"（混合上下文信息）
- **FFN** = 每个词自己"闭关深加工"（非线性升维）
- **残差** = 高速公路（梯度直通、保底信息）
- **LayerNorm** = 稳压器（稳定每层分布）
- **位置编码** = 给无序的注意力补回"先来后到"

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 架构细节进阶：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
- 注意力与缓存优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 推理系统：[[llm-inference/PD分离]] · [[llm-inference/vllm/README]]
- 内存与并行：[[docs/transformer内存估算]] · [[llm-train/README]] · [[B07:llm-inference/大模型推理张量并行]]
- 对齐训练：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 性能指标：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
