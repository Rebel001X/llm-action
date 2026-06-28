# ChatGLM / GLM 架构

> GLM 用「自回归填空(Autoregressive Blank Infilling)」把 BERT 的双向理解与 GPT 的自回归生成统一进一个目标函数；ChatGLM 是基于 GLM 训练范式、面向中英双语对话深度优化的系列模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/transformer/模型架构]] [[llm-algo/旋转编码RoPE]]

---

## 阅读地图

| 节 | 主题 | 你将搞懂 |
|----|------|----------|
| 0 | 一句话锚点 | GLM 到底解决了什么"目标函数割裂"问题 |
| 1 | 地基 | 三种预训练范式(AE/AR/Seq2Seq)各自的命门 |
| 2 | 自回归填空 | 怎么 mask span、怎么打乱、损失怎么算(逐 token 手算) |
| 3 | 2D 位置编码 | 为什么要两个位置 id、怎么编码、ASCII 对照表 |
| 4 | Prefix-LM 注意力 | Part A 双向 + Part B 单向的 mask 矩阵手画 |
| 5 | 与纯 Decoder 差异 | GPT/BERT/T5/GLM 四方对照 |
| 6 | ChatGLM 演进 | 6B→2→3→4 关键改动与原因 |
| 7 | 中文优化 | 词表、tokenizer、双语语料 |
| 8 | 规模配置 | ChatGLM-6B 参数量逐层手算 |
| 9 | 数值示例 | 一个完整样本从 mask 到 loss 走一遍 |
| 10 | 面试问答 | 高频问题 + 踩点答案 + 追问 |

---

## 0. 一句话锚点

- **BERT(AE,自编码)** 擅长"理解"(双向看上下文)，但**不会生成**(预测的 token 之间互相独立，无法连贯续写)。
- **GPT(AR,自回归)** 擅长"生成"(从左到右逐词预测)，但**看不到右边**(单向，理解类任务吃亏)。
- **GLM** 的核心 trick：把句子里挖掉若干连续片段(span)，**剩下的部分双向编码**(像 BERT)，**被挖掉的片段自回归地、逐 token 地生成出来**(像 GPT)。
- 一个目标函数同时获得"双向理解 + 自回归生成"两种能力 → 这就是 **自回归填空 (Autoregressive Blank Infilling)**。
- **ChatGLM** = GLM 训练范式 + 中英双语大规模语料 + 对话指令微调(SFT/RLHF) + 工程优化(量化、长上下文)。

---

## 1. 地基：三种预训练范式的命门

在理解 GLM 之前，先把三种范式拆到最原子。设原句 token 序列为 $x=(x_1,\dots,x_n)$。

```
范式            可见范围          预测目标               典型模型     命门
───────────────────────────────────────────────────────────────────────
自编码 AE        全句双向          被 [MASK] 的位置        BERT         预测的 mask 彼此独立,不能生成
                ←─────────→       P(x_i | x_\{mask})                  (假设各 mask 条件独立)
自回归 AR        只能看左边         下一个 token            GPT          看不到右侧上下文
                ─────────→        P(x_t | x_{<t})                     单向,理解类任务弱
序列到序列 S2S   编码器双向         解码器逐 token          T5/BART      两套参数(enc+dec),参数利用率低
                + 解码器单向       P(y_t | y_{<t}, x)
```

**AE 为什么不能生成？** BERT 的目标是 $\prod_i P(x_i \mid x_{\setminus \text{mask}})$，每个被 mask 的位置**独立预测**。如果连续 mask 掉两个词，模型不知道"第二个词要在第一个词已确定的前提下"生成，于是两个 mask 之间没有依赖，产物不连贯。

**GLM 的破局点**：让被挖空的片段**内部也是自回归的**(逐 token，前一个 token 是后一个的条件)，同时整段空白对原文其余部分是**双向可见**的。下面逐步展开。

---

## 2. 自回归填空 (Autoregressive Blank Infilling)

### 2.1 直觉与流程

```
原句:  [x1  x2  x3  x4  x5  x6]
        ↓ 随机采样若干连续 span 挖空 (span 长度服从泊松分布 λ≈3, 总遮盖≈15%)
挖空:  [x1  x2  [M] x4  x5  [M]]        ← span1={x3}, span2={x6}
                 ↑span1       ↑span2

拆成两部分:
  Part A (Corrupted text, 双向):  x1 x2 [M] x4 x5 [M]
  Part B (被挖的 span,自回归):    要把 [M] 处的真实内容生成出来

关键: span 之间打乱顺序 (shuffle), 训练模型不依赖填空顺序
拼接送入同一个 Transformer:
  ┌──────────── Part A ────────────┐ ┌──── Part B (span 们, 各自以 [S] 起,以 [E] 终) ────┐
  [x1] [x2] [M] [x4] [x5] [M]        [S] x6 [E]   [S] x3 [E]
   └── 双向注意力,互相都能看 ──┘        └─ 自回归: 生成时只看左侧 + 整个 Part A ─┘
```

- `[M]` (`[MASK]`)：占位 token，标记 Part A 中哪里被挖空。
- `[S]` (start)：Part B 中每个 span 的起始符，自回归生成从它开始。
- `[E]` (end)：span 结束符，模型学会"何时停"。
- **span 长度 ~ 泊松分布**(λ≈3)：短到单词、长到短语都覆盖；总遮盖比例约 15%(类 BERT)。
- **span 顺序打乱**：填 span1 时能看到 span2 的"槽位"但看不到其内容，迫使模型在任意顺序下都能填空。

### 2.2 两类预训练目标(GLM 同时做)

| 目标 | span 数量/长度 | 像谁 | 训练什么能力 |
|------|----------------|------|--------------|
| **文档级 (Document-level)** | 1 个长 span(覆盖 50%~100% 句尾) | GPT | 长文本生成 |
| **句子级 (Sentence-level)** | 多个短 span(总 15%) | BERT/T5 | 双向理解、短填空 |

→ 用统一框架，靠**采样不同的 span 配置**就能在"理解"和"生成"之间连续滑动，这是 GLM 的精髓。

### 2.3 损失函数(逐 token)

GLM 最大化被挖空内容的对数似然：

$$\mathcal{L}_{\text{GLM}} = -\sum_{\text{span } s}\ \sum_{j=1}^{|s|}\ \log P\big(z_{s,j}\mid \underbrace{x_{\text{PartA}}}_{\text{双向}},\ \underbrace{z_{s,<j}}_{\text{本 span 已生成}},\ \underbrace{z_{\text{prev spans}}}_{\text{已填完的 span}}\big)$$

读法："第 $s$ 个 span 的第 $j$ 个 token，给定**整个 Part A**(双向可见) + **本 span 前 $j-1$ 个 token** + **之前已填完的 span**，的条件概率"。span 内部是自回归的(逐 token)，对 Part A 是双向的。

---

## 3. 2D 位置编码 (2D Positional Encoding)

### 3.1 为什么需要两个位置 id？

普通 Transformer 一个 token 一个位置 id 就够了。但 GLM 把 span 的内容(Part B)**拼到序列末尾**，模型必须同时知道两件事：

1. 这个被生成的 token，**原本应该插回原句的哪个位置**？(否则填回去就乱了)
2. 这个 token，**在它所属 span 内部排第几**？(span 内部自回归需要顺序)

单一位置 id 无法同时表达"在原句的全局槽位"和"在 span 内的局部偏移"，于是 GLM 给每个 token **两个**位置 id：

- **Position 1 (intra-position / 句内位置)**：token 在**原始未损坏句子**中的全局位置。被同一个 `[M]` 替换的整个 span，Part B 里它们都**共享**这个 `[M]` 的位置。
- **Position 2 (inter-position / span 内位置)**：token 在**它所属 span 内部**的偏移；Part A 的 token 这一维全为 0。

### 3.2 ASCII 对照表(关键！亲手对一遍)

接 2.1 的例子：原句 `x1 x2 x3 x4 x5 x6`，挖 span1={x3}(原位 3)、span2={x6}(原位 6)，打乱后 Part B 先填 span2 再 span1。

```
拼接序列:   x1  x2  [M] x4  x5  [M] | [S] x6 [E] | [S] x3 [E]
            ───────Part A───────── | ──span2──  | ──span1──

Position 1  1   2   3   4   5   6  |  6   6   6 |  3   3   3   ← "应插回原句的位置"
(句内全局)                            └─[S]/x6/[E] 都=6(span2 对应 [M] 在原位6)
                                                   └─都=3(span1 对应原位3)

Position 2  0   0   0   0   0   0  |  0   1   2 |  0   1   2   ← "span 内部第几个"
(span内局部)                          [S]=0,x6=1,[E]=2          [S]=0,x3=1,[E]=2
```

要点拆解：

- Part A 全部 token：**Position 2 = 0**(它们不属于任何待生成 span)。
- 每个 span 的 `[S]` 起始符：**Position 2 = 0**，随后 token 在 span 内 +1 递增。
- 同一 span 内所有 token 的 **Position 1 相同**(都等于该 span 被挖处的原句位置)——这正是把它们"填回原位"的依据。
- 两个位置 id 各自查 embedding 表，**相加**后融入 token 表示。

> ✦ 一句话记住：**Position 1 管"填回哪"，Position 2 管"span 内排第几"。** (ChatGLM-6B 仍用这套 2D 编码 + 相对位置；从 ChatGLM2 起改用 **RoPE 旋转位置编码**，见 [[llm-algo/旋转编码RoPE]] 与第 6 节。)

---

## 4. Prefix-LM 注意力掩码

### 4.1 GLM 的注意力 = Part A 双向 + Part B 单向

这正是 **Prefix-LM**(前缀语言模型)：序列前半段(prefix)互相双向可见，后半段自回归。GLM 把"Part A=prefix(双向)"、"Part B=要生成的(单向)"。

```
注意力可见性规则:
  ① Part A 内部:          互相都能看 (双向)
  ② Part A ← Part B:      看不见 (生成内容不能泄露给上下文)
  ③ Part B → Part A:      能看见整个 Part A (生成时利用全部上下文)
  ④ Part B 内部:          只能看自己左边 (自回归, 含已填完的前序 span)
```

### 4.2 手画 Attention Mask 矩阵

行=query(谁在看)，列=key(被看)，✓=可见，✗=mask 掉。简化样本：Part A=`x1 x2 [M]`，Part B=`[S] x3`(填 [M])。

```
            key→  x1   x2   [M]  | [S]  x3
   query↓
   x1            ✓    ✓    ✓   | ✗    ✗     ┐
   x2            ✓    ✓    ✓   | ✗    ✗     │ Part A: 行内全 ✓ (双向)
   [M]           ✓    ✓    ✓   | ✗    ✗     ┘ 且看不见 Part B (✗✗)
   ──────────────────────────────────────
   [S]           ✓    ✓    ✓   | ✓    ✗     ┐ Part B: 能看全部 Part A
   x3            ✓    ✓    ✓   | ✓    ✓     ┘ + 只看自己左边 (下三角)
```

- 左上 3×3 块全 ✓ → Part A 双向(对比 GPT 的因果 mask 是下三角)。
- 右上 3×2 块全 ✗ → Part A 看不见待生成内容(防泄露)。
- 左下 2×3 块全 ✓ → Part B 能利用整个上下文。
- 右下 2×2 块是**下三角** → Part B 内部自回归。

> 对比记忆：**BERT** 全 ✓(纯双向)；**GPT** 纯下三角(纯单向)；**GLM** 左上双向块 + 右下因果块 = **块状混合**。

---

## 5. 与纯 Decoder / 其他范式的差异

| 维度 | BERT (AE) | GPT (纯 Decoder, AR) | T5 (Encoder-Decoder) | **GLM** |
|------|-----------|----------------------|----------------------|---------|
| 注意力 | 全双向 | 因果下三角(单向) | enc 双向 / dec 单向 | **Part A 双向 + Part B 单向(Prefix-LM)** |
| 参数 | 单栈 | 单栈 | **两套**(enc+dec) | **单栈**(参数利用率高) |
| 能否生成 | ✗(mask 独立) | ✓ | ✓ | ✓(span 自回归) |
| 双向理解 | ✓ | ✗ | ✓(enc 侧) | ✓(Part A 侧) |
| 位置编码 | 绝对/可学习 | 绝对→RoPE | 相对(bucket) | **2D 位置编码**(6B)→RoPE(2+) |
| 预训练目标 | MLM | next-token | span corruption | **自回归填空** |
| 一句话 | 只懂不写 | 只写半懂 | 双能但双参数 | 单栈兼双能 |

**与"纯 Decoder(GPT/LLaMA)"最本质的差异：** 纯 Decoder 全程因果 mask，任何 token 都看不到右侧；GLM 把输入的"已知部分"放进 Part A 并对其开放**双向**注意力，只对"待生成部分"用单向。所以同样是单栈架构，GLM 在"已知上下文"上比 GPT 多了双向感知能力。

> 注：ChatGLM2/3/4 的**架构骨架已回归到更接近主流纯 Decoder** 的实现(RoPE + 因果注意力 + Multi-Query/GQA)，但其**预训练目标仍带 GLM 的填空基因**，对话场景下输入可视作 prefix。"GLM 范式"与"具体某代实现"要分开看。

---

## 6. ChatGLM 演进史(为什么这么改)

```
GLM-130B ──► ChatGLM-6B ──► ChatGLM2-6B ──► ChatGLM3-6B ──► ChatGLM4 / GLM-4
(基座,中英)  (对话首版)      (效率+长上下文)   (工具/Agent)      (能力对齐 GPT-4 级)
```

| 版本 | 关键改动 | 解决的痛点 |
|------|----------|-----------|
| **GLM-130B** | 1300 亿参数中英双语基座，自回归填空预训练 | 提供高质量中英基座，验证 GLM 范式可 scale |
| **ChatGLM-6B** | 62 亿参数，SFT+反馈自助+RLHF 对话微调；**2D 位置编码 + Prefix-LM**；GeLU；Pre-LN；INT4 量化可消费级显卡跑 | 让对话模型在 6GB 显存级别本地可用 |
| **ChatGLM2-6B** | 改用 **RoPE** 位置编码；引入 **Multi-Query Attention (MQA)** 提升推理吞吐、降 KV cache；上下文 2K→**32K**；更优训练目标 | 长上下文 + 推理提速 + 显存下降 |
| **ChatGLM3-6B** | 全新 prompt 格式与 **function call / 代码解释器 / Agent** 能力；保留 base/chat 版本 | 面向工具调用与智能体应用 |
| **GLM-4 / ChatGLM4** | 综合能力对标 GPT-4 级；**128K~1M 长上下文**；多模态(GLM-4V)；All Tools | 通用强能力 + 超长上下文 + 多模态 |

**改动的统一逻辑**：早期(6B)用 2D 位置编码 + Prefix-LM 体现 GLM 思想；随着对话/长文成为主战场，**RoPE(长度外推友好) + MQA/GQA(省 KV cache) + 因果实现** 这套更利于推理效率的组合逐步取代了原始 2D 编码——架构向主流靠拢，但训练目标的 GLM 填空基因保留。

---

## 7. 中文 / 双语优化

- **双语语料平衡**：GLM-130B 中英双语训练，中文语料占比高，避免"英文为主、中文凑数"导致的中文短板。
- **词表 (vocabulary)**：ChatGLM 采用 **SentencePiece (BPE/Unigram)** 风格分词，词表约 **13 万**(ChatGLM-6B `icetk` ≈ 130528)，对中文常用字/词覆盖好，**单字平均 token 数低**(中文一个汉字常≈1 token)，相比纯英文 GPT 词表在中文上更省 token、更连贯。
- **指令/对话格式**：为中文对话设计专门的角色标记(如 `[Round]`、`问:`/`答:`，3 代起用 `<|user|>`/`<|assistant|>` 等)，贴合中文交互习惯。
- **量化与端侧**：官方提供 INT4/INT8 量化，6B INT4 约 6GB 显存即可推理，降低中文用户本地部署门槛(见第 8 节显存手算)。

---

## 8. 规模配置与参数量手算 (ChatGLM-6B)

公开近似配置(以官方为准)：

```
hidden_size  d = 4096      layers L = 28      heads h = 32 (head_dim=128)
ffn_inner    = 4d = 16384  vocab V ≈ 130528   max_seq ≈ 2048
激活 GeLU    LN: Pre-LN    位置: 2D 位置编码
```

### 8.1 单层 Transformer 参数(约)

注意力 QKVO 四个 $d\times d$ 矩阵：

$$4 d^2 = 4 \times 4096^2 = 4 \times 16{,}777{,}216 = 67{,}108{,}864 \approx 67.1\text{M}$$

FFN 两个矩阵 $d\to 4d \to d$：

$$2 \times d \times 4d = 8 d^2 = 8 \times 16{,}777{,}216 = 134{,}217{,}728 \approx 134.2\text{M}$$

单层合计(忽略 LN/bias 的小量)：

$$67.1 + 134.2 \approx 201.3\text{M}/\text{层}$$

### 8.2 全模型参数(约)

$$28 \text{ 层} \times 201.3\text{M} \approx 5{,}636\text{M} \approx 5.64\text{B (Transformer 主体)}$$

加 **词嵌入** $V\times d = 130528\times 4096 \approx 534.6\text{M}$(输入/输出若共享则计一次)：

$$5.64\text{B} + 0.53\text{B} \approx 6.17\text{B} \approx 62\text{ 亿参数}$$  ✓ 与"6B"吻合。

### 8.3 显存手算(推理，权重部分)

| 精度 | 每参数字节 | 6.2B 权重显存 |
|------|-----------|----------------|
| FP16/BF16 | 2 B | $6.2\text{e9}\times 2 = 12.4\text{ GB}$ |
| INT8 | 1 B | $\approx 6.2\text{ GB}$ |
| INT4 | 0.5 B | $\approx 3.1\text{ GB}$ + 少量 scale/激活 ≈ **约 6 GB 可跑** |

> 这正是 ChatGLM-6B "消费级显卡本地可跑"的来源：INT4 权重约 3.1GB，加 KV cache、激活、框架开销，6GB 显存级别即可推理。

### 8.4 量化 scale 手算(对称 INT4 举例)

设某权重张量 $\max(|W|)=0.84$。INT4 对称量化范围 $[-7,7]$(4bit 取 $-2^3+1 \sim 2^3-1$)：

$$\text{scale}=\frac{\max(|W|)}{7}=\frac{0.84}{7}=0.12$$

某权重 $w=0.30 \Rightarrow q=\text{round}(0.30/0.12)=\text{round}(2.5)=3$(就近偶数/四舍五入按实现)，反量化 $\hat w = 3\times 0.12 = 0.36$，量化误差 $|0.36-0.30|=0.06$。

---

## 9. 数值示例：一个样本从 mask 到 loss

原句(token 化后)：`今天 天气 很 好 ， 我们 去 公园`(8 token，记 $x_1..x_8$)。

```
Step1 采样 span:  挖 span1={x3,x4}="很好"(原位3-4), span2={x7}="去"(原位7)
                  总遮盖 3/8 ≈ 37.5% (示例放大,实际≈15%)

Step2 Part A:     今天 天气 [M] ， 我们 [M] 公园
                  位置id1:  1   2   3   5   6   7   8     ([M]占被挖处起始位)
                  位置id2:  0   0   0   0   0   0   0

Step3 打乱 span:  先 span2 后 span1
   Part B:        [S] 去  [E]  |  [S] 很  好  [E]
   位置id1:        7   7   7   |   3   3   3   3   (span2→7, span1→3)
   位置id2:        0   1   2   |   0   1   2   3

Step4 自回归预测(只算 Part B 的真实 token, [S]不算 loss):
   预测"去":  P(去 | PartA, [S]_span2)                 → -log p1
   预测[E]:   P([E]| PartA, [S]去)                      → -log p2
   预测"很":  P(很 | PartA, span2已填, [S]_span1)       → -log p3
   预测"好":  P(好 | PartA, span2已填, [S]很)            → -log p4
   预测[E]:   P([E]| PartA, span2已填, [S]很好)          → -log p5
```

设各步预测概率 $p_1..p_5 = 0.5,\ 0.9,\ 0.4,\ 0.8,\ 0.95$，则该样本平均交叉熵：

$$\mathcal{L}=-\frac{1}{5}\sum \ln p_i = -\frac{1}{5}(\ln0.5+\ln0.9+\ln0.4+\ln0.8+\ln0.95)$$

$$= -\frac{1}{5}(-0.693-0.105-0.916-0.223-0.051)= -\frac{1}{5}(-1.988)=0.398$$

对应困惑度 $\text{PPL}=e^{0.398}\approx 1.49$。每个 token 的预测都**同时利用了双向的 Part A 和 span 内已生成的左侧**——这就是自回归填空在数值上的样子。

---

## 10. 面试问答清单(高频问题→踩点答案→追问)

**Q1. GLM 的核心创新一句话？**
答：自回归填空——挖掉连续 span，剩余部分双向编码，被挖 span 自回归逐 token 生成，用单一目标函数统一了 BERT 的理解和 GPT 的生成。
- 追问：为什么不直接用 T5？→ T5 是 enc-dec 两套参数；GLM **单栈**，参数利用率更高，且通过 span 配置可在理解/生成间平滑切换。

**Q2. 为什么要 2D 位置编码，单个位置 id 不行吗？**
答：Part B 被拼到末尾，需要同时知道"填回原句哪个槽位"(Position 1)和"在 span 内排第几"(Position 2)，单一 id 无法同时表达全局槽位与局部偏移。
- 追问：Part A 的 Position 2 是多少？→ 全 0(不属于任何待生成 span)。同 span 的 token Position 1 相同。

**Q3. GLM 的注意力和 GPT 有何不同？**
答：GPT 全程因果下三角(单向)；GLM 是 Prefix-LM：Part A 内部双向、Part B 对 Part A 可见但内部因果、Part A 看不见 Part B。即左上双向块 + 右下因果块。
- 追问：为什么 Part A 不能看 Part B？→ 防止待生成答案泄露给上下文，否则训练作弊。

**Q4. span 长度和遮盖比例怎么定？**
答：span 长度服从泊松分布(λ≈3)，总遮盖≈15%(句子级目标)；文档级目标则用 1 个长 span(覆盖句尾 50%~100%)，模拟长文生成。span 顺序打乱以学到顺序无关的填空。

**Q5. ChatGLM 从 6B 到 2/3/4 架构上变了什么？为什么？**
答：6B 用 2D 位置编码 + Prefix-LM；2 代起改 **RoPE**(长度外推好)、引入 **MQA/GQA**(降 KV cache、提速)、上下文从 2K 扩到 32K/128K+，逐步靠拢主流纯 Decoder 实现，但保留 GLM 填空预训练基因。
- 追问：MQA 省在哪？→ 多个 Q 头共享一组 K/V，KV cache 显存按头数倍数下降，长上下文推理更省。

**Q6. ChatGLM 为什么中文好？**
答：中英双语平衡语料 + 约 13 万的 SentencePiece 词表(中文单字常≈1 token，覆盖好、省 token)+ 中文对话格式设计。

**Q7. 6B 模型 INT4 为什么能在消费级显卡跑？(让你算)**
答：6.2B 参数 INT4 = 0.5B/参数字节 ≈ 3.1GB 权重，加 KV cache/激活/框架开销，约 6GB 显存级别即可推理(对比 FP16 需 12.4GB)。

**Q8. GLM 和 Pre-LN/GeLU 有什么关系？(实现细节)**
答：ChatGLM 在实现上把 Post-LN 改为 **Pre-LN**(残差路径更稳、利于深层训练)，激活用 **GeLU**(平滑、比 ReLU 表达力强)，输出层用线性层预测词。这些是工程稳态选择，非 GLM 范式本身。

---

## 对照/复杂度表

| 项 | 公式/数值(ChatGLM-6B 约) | 备注 |
|----|--------------------------|------|
| 单层注意力参数 | $4d^2=67.1\text{M}$ | QKVO |
| 单层 FFN 参数 | $8d^2=134.2\text{M}$ | $d\to4d\to d$ |
| 28 层主体 | $\approx 5.64\text{B}$ | |
| +词嵌入 | $V d\approx0.53\text{B}$ | $V\approx130528$ |
| 总参数 | $\approx 6.2\text{B}$ | "6B" |
| FP16 权重显存 | $12.4\text{GB}$ | 2 B/参数 |
| INT4 权重显存 | $\approx 3.1\text{GB}$ | 0.5 B/参数 |
| 自注意力计算复杂度 | $O(n^2 d)$ | $n$=序列长，长上下文瓶颈 |

---

## 常见问题 / 高频追问

| 问题 | 要点 / 陷阱 |
|------|------------|
| GLM 是 encoder-decoder 吗？ | **不是两套参数**。是单栈 Transformer + Prefix-LM 注意力 mask，"借鉴了 enc-dec 思想"≠"是 enc-dec 架构"。 |
| `[M]` 和 `[S]`/`[E]` 区别？ | `[M]` 在 Part A 占空位；`[S]`/`[E]` 在 Part B 标 span 的起止，`[E]` 让模型学会停。 |
| 同 span 的 token 位置 id1 为何相同？ | 它们都要填回原句**同一处被挖区域**，靠 Position 1 标记"插回哪"。 |
| ChatGLM2 还用 2D 位置编码吗？ | 不。2 代起改 **RoPE**(见 [[llm-algo/旋转编码RoPE]])；2D 编码是 6B 时代的特征。 |
| 为什么说 GLM 比 GPT 强在"理解"？ | 已知上下文(Part A)享双向注意力，GPT 全单向，理解类任务 GLM 占优。 |
| 自回归填空 vs T5 span corruption？ | 都挖 span，但 GLM **单栈** + span 内部 2D 位置 + 双向 Part A；T5 是 enc-dec 双栈。 |

---

## 🔗 跳转链接

- 全局导航：[[00-知识地图]]
- Transformer 架构地基(注意力/FFN/LN)：[[llm-algo/transformer/模型架构]]
- RoPE 旋转位置编码(ChatGLM2+ 采用)：[[llm-algo/旋转编码RoPE]]

> 一句话收尾：**GLM = 单栈 Transformer + 自回归填空 + Prefix-LM 注意力 + 2D 位置编码**，用一个目标函数把"双向理解"和"自回归生成"焊在一起；ChatGLM 在此之上做中英双语 + 对话对齐 + 推理工程优化，并在后续版本逐步向 RoPE/MQA 的主流高效实现演进。
