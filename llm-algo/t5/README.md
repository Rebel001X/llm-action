# T5 (Encoder-Decoder)

> T5 = Text-to-Text Transfer Transformer：把**所有** NLP 任务统一成"输入一段文本 → 输出一段文本"，用一个标准 Encoder-Decoder Transformer 通吃。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/transformer/模型架构]] [[llm-algo/bert]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点 | text-to-text |
| 1 | 地基/前置：Transformer、BERT、GPT 各管什么 | Encoder/Decoder |
| 2 | text-to-text 统一范式：为什么"万物皆文本" | task prefix |
| 3 | Encoder-Decoder 架构逐层拆解 | self/cross-attn |
| 4 | 相对位置偏置（T5 的招牌创新） | relative bias |
| 5 | span corruption 预训练目标 | sentinel token |
| 6 | T5 vs Decoder-only（GPT/LLaMA） | 双向 vs 单向 |
| 7 | 规模配置与训练要点 | C4、Adafactor |
| 8 | 数值示例 / 逐数手算 | 参数量、注意力、掩码率 |
| 9 | 面试问答清单 | 高频考点+陷阱 |
| — | 对照/复杂度表、常见问题、跳转 | — |

---

## 0. 一句话锚点

**T5 把翻译、摘要、分类、问答、相似度打分……全部改写成"喂一串文本、吐一串文本"，然后用一个原汁原味的 Encoder-Decoder Transformer 来学。** 例：输入 `"translate English to German: That is good."` → 输出 `"Das ist gut."`；连"判断两句相似度 0~5"这种回归，输出也是文本 `"3.8"`。一个模型、一个损失（交叉熵）、一套流程，统一所有任务——这就是 text-to-text 的威力。

---

## 1. 地基 / 前置：三种 Transformer 用法

原始 Transformer（[[llm-algo/transformer/模型架构]]）有两半：Encoder 和 Decoder。后来的模型各取所需，分裂成三大流派：

```
              ┌────────────────────────────────────────────┐
              │            原始 Transformer (2017)           │
              │     Encoder (双向)  +  Decoder (单向)         │
              └───────────────┬───────────┬──────────────────┘
            只留 Encoder       │           │       只留 Decoder
         ┌──────────────┐     │           │     ┌──────────────┐
         │   BERT 系     │     │           │     │   GPT 系      │
         │  双向编码      │     │           │     │  自回归生成    │
         │  擅长"理解"    │     │           │     │  擅长"生成"    │
         │ [[llm-algo/bert]]│  │           │     │ (GPT/LLaMA)  │
         └──────────────┘     │           │     └──────────────┘
                              ▼           ▼
                       ┌──────────────────────┐
                       │    T5 系 (本文)        │
                       │  完整 Encoder-Decoder  │
                       │  既能理解、又能生成     │
                       └──────────────────────┘
```

| 流派 | 注意力 | 代表 | 强项 | 弱项 |
|------|--------|------|------|------|
| Encoder-only | 双向 | BERT、RoBERTa | 分类、NER、抽取 | 不能自由生成 |
| Decoder-only | 单向(因果) | GPT、LLaMA | 开放式生成 | 输入端无双向上下文 |
| **Encoder-Decoder** | Enc 双向 + Dec 单向 | **T5**、BART | 输入理解+输出生成 | 参数/算力翻倍 |

**记忆锚点**：BERT 只会"读"，GPT 只会"写"，**T5 既会读又会写**——读用双向 Encoder，写用单向 Decoder，中间用 cross-attention 把两者缝起来。

---

## 2. text-to-text 统一范式

### 2.1 痛点：传统多任务模型千头万绪

在 T5 之前，不同任务有不同的"输出头"：
- 分类 → 接一个 softmax 分类头，输出 logits 向量
- NER → 接一个序列标注头（每个 token 一个标签）
- 回归（相似度）→ 接一个线性层输出一个实数
- 生成 → 接一个 vocab-size 的语言模型头

每换一个任务就要换一套头、换一套损失、换一套数据格式。**乱。**

### 2.2 T5 的解法：所有输出都是"文本字符串"

把任务类型写进**输入前缀（task prefix）**，输出统一是 token 序列：

```
任务          输入（文本）                                输出（文本）
────────────────────────────────────────────────────────────────────
翻译     "translate English to German: That is good."  "Das ist gut."
摘要     "summarize: <一长段新闻……>"                    "<短摘要>"
分类     "cola sentence: The course is jumping well."   "acceptable"
相似度   "stsb sentence1: ... sentence2: ..."           "3.8"
问答     "question: ... context: ..."                   "<答案文本>"
```

**关键洞察**：连"分类标签"和"分数"都当成文本 token 来预测。模型不需要知道"acceptable 是第 3 类"，它只要学会生成字符串 `acceptable` 即可。

```
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │  传统：N 个任务 N 套头     │         │  T5：N 个任务 1 套接口     │
   │                          │         │                          │
   │  输入 → 编码器 →┬→分类头   │         │  "prefix: 输入文本"        │
   │                ├→标注头   │   ⟹     │        ↓                  │
   │                ├→回归头   │         │  统一 Encoder-Decoder     │
   │                └→生成头   │         │        ↓                  │
   │  （每个头独立训练）        │         │  "输出文本"（交叉熵 LM 损失）│
   └──────────────────────────┘         └──────────────────────────┘
```

### 2.3 为什么这样行得通

- **预训练与微调同构**：预训练目标（见第 5 节）也是 text→text，迁移时无缝衔接。
- **正负迁移可控**：多任务混在一起训练，前缀帮模型区分"现在该干哪个任务"。
- **损失唯一**：永远是 token 级交叉熵 $\mathcal{L}=-\sum_t \log P(y_t \mid y_{<t}, x)$，工程极简。

---

## 3. Encoder-Decoder 架构逐层拆解

### 3.1 整体数据流

```
 输入 x: "translate English to German: That is good."
            │
            ▼  (tokenize + embedding)
 ┌──────────────────────────────────────────┐
 │              ENCODER (N 层)                │   每层:
 │  ┌────────────────────────────────────┐  │   1) 双向 Self-Attention
 │  │  双向 Self-Attn → FFN  (重复 N 次)   │  │      (每个 token 看全句)
 │  └────────────────────────────────────┘  │   2) FeedForward
 │            输出: 记忆 memory  H_enc        │   + 残差 + LayerNorm(RMS式)
 └──────────────────┬───────────────────────┘
                    │ memory 传给每一个 decoder 层
                    ▼
 ┌──────────────────────────────────────────┐
 │              DECODER (N 层)                │   每层:
 │  ┌────────────────────────────────────┐  │   1) 因果 Self-Attn
 │  │ 因果Self-Attn → Cross-Attn → FFN    │  │      (只看已生成的)
 │  │            (重复 N 次)               │  │   2) Cross-Attn
 │  └────────────────────────────────────┘  │      (Q来自decoder,
 │            ↓                              │       K,V来自 H_enc)
 │       线性投影 → softmax over vocab        │   3) FeedForward
 └──────────────────┬───────────────────────┘
                    ▼
 输出 y: "Das ist gut." （自回归，一次一个 token）
```

### 3.2 三种注意力的分工（这是 Encoder-Decoder 的灵魂）

| 注意力 | 位置 | Q 来自 | K,V 来自 | 掩码 | 作用 |
|--------|------|--------|----------|------|------|
| Encoder Self-Attn | Encoder | 输入 | 输入 | 无（双向） | 充分理解输入 |
| Decoder Self-Attn | Decoder | 已生成 | 已生成 | 因果（下三角） | 保证自回归不偷看未来 |
| **Cross-Attn** | Decoder | 已生成(decoder) | **Encoder 输出** | 无 | 让生成时"回看输入" |

**Cross-Attention 是桥**：解码第 $t$ 步时，decoder 拿当前状态当 Query，去 encoder 的全部输出里"检索"该关注输入的哪部分。翻译时输出 "Das" 自然会重点对齐输入的 "That"。

$$\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\!\left(\frac{Q_{dec} K_{enc}^\top}{\sqrt{d_k}}\right) V_{enc}$$

### 3.3 T5 的"瘦身"细节（与原始 Transformer 的差异）

T5 在标准 Transformer 上做了几处简化，使训练更稳：

1. **LayerNorm 去掉 bias、去掉减均值**：用类 RMSNorm，$\;\text{RMSNorm}(x)=\dfrac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}}\cdot g$。
2. **LayerNorm 放在子层输入处**（pre-norm 风格的变体），并在每个 block 外加残差。
3. **去掉了正弦/可学习的绝对位置编码**，改用**相对位置偏置**（见第 4 节）。
4. **Encoder 与 Decoder 的 embedding、以及输出投影 三者共享同一张词表矩阵**（weight tying），省参数。
5. FFN 默认 ReLU（后续 T5.1.1 改用 GEGLU/GeGLU 提升效果）。

---

## 4. 相对位置偏置（T5 招牌创新）

### 4.1 为什么不用绝对位置编码

原始 Transformer 把"第几个位置"编码成一个向量加到 embedding 上。问题：
- 模型学到的是"第 5 个位置长这样"，**外推**到训练时没见过的更长序列就崩。
- 语言的本质是**相对关系**（"形容词在名词前面" vs "在句子第几个字"），相对更自然。

### 4.2 T5 怎么做：往注意力 logits 里加一个"偏置标量"

标准注意力打分：$e_{ij} = \dfrac{q_i \cdot k_j}{\sqrt{d_k}}$。

T5 改成：

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{\,r(i,j)}$$

其中 $b$ 是一个**可学习的标量偏置**，只依赖相对距离 $i-j$，**与 token 内容、与具体位置无关**。

```
 注意力分数矩阵 (query i 看 key j)，每格再 +b[bucket(i-j)]

         j=0   j=1   j=2   j=3
 i=0    +b0   +b-1  +b-2  +b-3      ← 相对距离 = i - j
 i=1    +b+1  +b0   +b-1  +b-2
 i=2    +b+2  +b+1  +b0   +b-1
 i=3    +b+3  +b+2  +b+1  +b0

 b0,b+1,... 是少量可学习标量，按"距离分桶"共享
```

### 4.3 关键技巧：分桶（bucketing）+ 每个 head 一套

- 不为每个相对距离单独学一个标量（那会很多且远距离样本稀疏）。而是把相对距离**对数分桶**：近距离精细（0,1,2,3 各一桶），远距离粗放（4-6 一桶、7-11 一桶……）。默认 32 个桶，超过 128 距离截断到同一桶。
- **每个注意力 head 学自己的一套桶偏置**（H 个 head × 32 桶）。
- **偏置在所有层之间共享**（只第一层的相对位置模块计算，后续层复用），参数极省。

### 4.4 为什么这招好

- **可外推**：因为只看相对距离且远距离分桶，长序列也有合理偏置。
- **参数极少**：H×32 个标量 ≈ 几百到几千个参数，相比绝对位置 embedding（seq_len × d_model）小到忽略不计。
- **平移不变**：把整句往后挪，相对关系不变，注意力模式不变——符合语言直觉。

---

## 5. span corruption 预训练目标

### 5.1 BERT 式 MLM 的局限

[[llm-algo/bert]] 的 Masked LM：随机遮 15% 的**单个 token**，让模型预测被遮的词。问题：
- 输出和输入一样长，**没有"生成短输出"的训练信号**，与 T5 的 text-to-text 生成范式不契合。
- 一次只预测一个孤立 token，学不到"连续片段"的生成能力。

### 5.2 T5 的做法：遮掉连续 span，用哨兵 token 占位

随机挑选一些**连续片段（span）**整段遮掉，每段用一个独一无二的**哨兵 token（sentinel）** `<X>` `<Y>` `<Z>`… 代替；**目标输出 = 这些哨兵 + 它们对应的原文片段**。

```
 原句:   Thank you for inviting me to your party last week .
 遮的span: [for inviting]                和    [last]

 Encoder 输入 (corrupted):  Thank you <X> me to your party <Y> week .
 Decoder 目标 (targets)  :  <X> for inviting <Y> last <Z>
        ↑ 只需吐出"被遮的内容"，输出比输入短很多
```

### 5.3 设计参数（T5 论文经大量消融得出的默认值）

| 超参 | 默认值 | 含义 |
|------|--------|------|
| corruption rate | **15%** | 被遮 token 占比 |
| mean span length | **3** | 每个被遮 span 平均长度 |
| sentinel | 100 个保留 token | `<extra_id_0>` … `<extra_id_99>` |

### 5.4 为什么这样设计好

- **输出短**：只生成被遮内容（≈输入的 15%），训练比"重建整句"高效得多。
- **天然对齐 text-to-text**：输入文本→输出文本，预训练与微调**同一接口**。
- **学到 span 级生成**：连续片段的填空逼模型建模短语/子句结构，比单 token MLM 更接近真实生成。
- 论文消融显示 span corruption 综合表现优于 BERT-MLM、纯 LM、deshuffling 等替代目标。

---

## 6. T5 vs Decoder-only（GPT/LLaMA）

```
  Encoder-Decoder (T5)                  Decoder-only (GPT/LLaMA)
  ──────────────────────                ────────────────────────
  输入 ──► [双向Encoder] ──memory──┐     输入+输出 拼成一条序列
                                   │      ──► [单向Decoder]
  输出 ◄── [单向Decoder]◄cross-attn┘            因果掩码贯穿始终
  输入端：每个 token 看全句(双向)        输入端：token 只能看左边(单向)
```

| 维度 | Encoder-Decoder (T5) | Decoder-only (GPT/LLaMA) |
|------|----------------------|--------------------------|
| 输入端注意力 | **双向**（理解充分） | 单向（看不到右侧上下文） |
| 参数效率 | 同等深度参数≈2×（两套栈） | 一套栈，更省 |
| KV-Cache 推理 | 输入只编码一次，输出端缓存 | 全序列缓存，简单统一 |
| 长输入+短输出 | 优（摘要、翻译、抽取式QA） | 也能做但输入无双向 |
| 开放式长生成/few-shot | 较弱，需微调 | **强**（GPT-3 范式、in-context） |
| 工程/扩展生态 | 相对小众 | 主流（绝大多数大模型走这条路） |
| 典型代表 | T5、BART、mT5、UL2、FLAN-T5 | GPT 系、LLaMA、Qwen、DeepSeek |

**为什么如今大模型多走 Decoder-only？** 单栈结构简单、扩到千亿工程更顺；few-shot/in-context 在 decoder-only 上涌现更充分；一条序列统一"输入+输出"使 KV-Cache 实现统一。

**T5 路线仍有价值的场景**：输入很长、输出很短、需对输入做**双向充分理解**的任务（机器翻译、长文摘要、抽取式问答、表格转文本）。FLAN-T5（指令微调版）至今是小参数高性价比基线。

---

## 7. 规模配置与训练要点

| 版本 | 参数量(约) | d_model | 层数(Enc=Dec) | heads | 说明 |
|------|-----------|---------|---------------|-------|------|
| T5-Small | 60M | 512 | 6 | 8 | — |
| T5-Base | 220M | 768 | 12 | 12 | 对标 BERT-Base 量级 |
| T5-Large | 770M | 1024 | 24 | 16 | — |
| T5-3B | 3B | 1024 | 24 | 32 | d_ff 很宽 |
| T5-11B | 11B | 1024 | 24 | 128 | 当年最大版本 |

> 具体超参以官方技术报告/HuggingFace config 为准；上表为常见公开数值的近似。

训练要点（数字以官方为准）：
- **数据**：自建 **C4**（Colossal Clean Crawled Corpus），从 Common Crawl 清洗出约 750GB 干净英文文本。
- **优化器**：**Adafactor**（省显存的 Adam 变体，不存全量二阶动量），适配 11B 大模型。
- **学习率**：inverse-square-root 调度（预热后按 $1/\sqrt{\text{step}}$ 衰减）。
- **预训练目标**：span corruption（第 5 节）。
- **多任务微调**：把多个下游任务按比例混合，统一 text-to-text。
- 衍生：**mT5**（多语言）、**T5.1.1**（GEGLU+无 dropout 预训练）、**FLAN-T5**（指令微调）、**ByT5**（字节级）。

---

## 8. 数值示例 / 逐数手算

### 8.1 参数量手算（Encoder-Decoder 为何"贵"）

单层 Transformer 主要参数（忽略 LayerNorm 等小项），设 $d=d_{model}$、$d_{ff}$：

- Self-Attn 的 $W_Q,W_K,W_V,W_O$：$4 d^2$
- FFN 两个矩阵：$2\, d \cdot d_{ff}$

以 **T5-Base** 估算（$d=768$，$d_{ff}=3072=4d$，层数 Enc=Dec=12）：

- 每个 Encoder 层：$4d^2 + 2 d d_{ff} = 4(768)^2 + 2(768)(3072)$
  $= 4\times589824 + 2\times2359296 = 2{,}359{,}296 + 4{,}718{,}592 = 7.08\text{M}$
- Encoder 12 层 ≈ $12 \times 7.08 = 85$M
- Decoder 每层多一个 Cross-Attn（再加 $4d^2 \approx 2.36$M）→ 每层 ≈ $9.44$M
- Decoder 12 层 ≈ $12 \times 9.44 = 113$M
- 词嵌入（共享）：$|V|\times d \approx 32000 \times 768 \approx 24.6$M
- 合计 ≈ $85 + 113 + 25 \approx 223\text{M}$ ✅ 与官方 T5-Base ≈220M 吻合。

**结论**：Decoder 比 Encoder 每层多一套 cross-attn（≈+33% 参数），这是 Encoder-Decoder 比同深度 Decoder-only "贵"的根因。

### 8.2 一个微型注意力 + 相对偏置手算

设 1 个 head，$d_k=2$，3 个 token。某 query $q_2=(1,0)$，三个 key：
$k_0=(1,0),\;k_1=(0,1),\;k_2=(1,1)$。

点积分数 $q_2\cdot k_j$：
- $j=0:\,1\cdot1+0\cdot0=1$
- $j=1:\,1\cdot0+0\cdot1=0$
- $j=2:\,1\cdot1+0\cdot1=1$

除以 $\sqrt{d_k}=\sqrt2\approx1.414$：$\;[0.707,\;0,\;0.707]$。

加相对位置偏置（设已学到：相对距离 +2→$b=-1.0$，+1→$b=-0.3$，0→$b=0$）：
- $i{=}2,j{=}0$，距离+2：$0.707-1.0=-0.293$
- $i{=}2,j{=}1$，距离+1：$0-0.3=-0.3$
- $i{=}2,j{=}2$，距离 0：$0.707+0=0.707$

softmax（$e^{-0.293}=0.746,\;e^{-0.3}=0.741,\;e^{0.707}=2.028$，和 $=3.515$）：
$$[0.212,\;0.211,\;0.577]$$

**解读**：相对偏置把"看自己（距离0）"的权重抬高、把"看更远的过去"压低——这正是 T5 让模型自己学出来的"近处更重要"的归纳偏置。

### 8.3 span corruption 掩码率手算

句子 100 个 token，corruption rate 15%、mean span length 3：
- 被遮 token 总数 ≈ $100 \times 0.15 = 15$ 个
- span 个数 ≈ $15 / 3 = 5$ 个 → 需要 5 个哨兵 `<extra_id_0..4>`
- Encoder 输入长度 ≈ $100 - 15 + 5 = 90$（被遮 15 个换成 5 个哨兵）
- Decoder 目标长度 ≈ $5(\text{哨兵}) + 15(\text{原文}) + 1(\text{结尾哨兵}) = 21$

**对比 BERT-MLM**：目标长度会是 100（要重建整句长度的输出位）。T5 目标仅 21，**训练算力省约 5×**，这就是 span corruption 高效的量化证据。

---

## 9. 面试问答清单

**Q1：T5 的核心思想一句话？**
踩点：把所有 NLP 任务统一成 text-to-text（输入文本→输出文本），用任务前缀区分任务，一套 Encoder-Decoder + 一个交叉熵损失通吃。
追问：连分类/回归也是文本？→ 是，标签和分数都当字符串 token 预测，模型不需要分类头。

**Q2：T5 为什么用 Encoder-Decoder 而不是 Decoder-only？**
踩点：很多目标任务是"长输入→短输出"（翻译/摘要/抽取QA），需要对输入做**双向**充分理解，双向 Encoder 比单向 Decoder 更合适；Decoder 负责生成，cross-attn 把两者连起来。
追问：那为什么现在大模型多 decoder-only？→ 单栈结构简单、易扩到千亿、few-shot/in-context 涌现更强、KV-Cache 统一。

**Q3：T5 的位置编码有什么特别？**
踩点：用**相对位置偏置**——往注意力 logits 加一个只依赖相对距离的可学习标量 $b_{r(i,j)}$，按对数分桶（默认32桶），每个 head 一套，层间共享。
追问：好处？→ 参数极少、可外推长序列、平移不变。陷阱：它是加在 attention score 上的标量，不是加在 embedding 上的向量。

**Q4：span corruption 和 BERT 的 MLM 区别？**
踩点：MLM 遮单个 token、输出和输入等长；span corruption 遮**连续片段**、用哨兵占位、**只输出被遮内容**（输出短）。前者 encoder-only 范式，后者天然 text-to-text。
追问：默认参数？→ 15% 掩码、平均 span 长 3、100 个 sentinel token。

**Q5：T5 与 BART 都是 Encoder-Decoder，区别？**
踩点：预训练目标不同。T5=span corruption（哨兵填空，目标只含被遮内容）；BART=去噪自编码（遮挡/打乱/删除等多种噪声，目标=重建**完整**原句）。T5 还首创相对位置偏置 + 严格 text-to-text 接口。

**Q6（陷阱）：T5 输入端是因果掩码吗？**
踩点：**不是**。Encoder 是**双向无掩码**自注意力（每个 token 看全句）；只有 Decoder 的 self-attn 才用因果（下三角）掩码。把 encoder 也当因果是常见错误。

**Q7：FLAN-T5 比 T5 强在哪？**
踩点：FLAN-T5 在 T5 基础上做大规模**指令微调**（上千个任务、含 CoT），zero-shot/few-shot 指令遵循能力大幅提升，是小参数高性价比的强基线。

---

## 对照 / 复杂度表

| 维度 | BERT | GPT/LLaMA | **T5** |
|------|------|-----------|--------|
| 架构 | Encoder-only | Decoder-only | Encoder-Decoder |
| 注意力 | 双向 | 单向因果 | Enc双向+Dec因果+Cross |
| 位置编码 | 可学习绝对 | 绝对/RoPE/ALiBi | **相对位置偏置** |
| 预训练目标 | MLM(+NSP) | 自回归 LM | **span corruption** |
| 输出形式 | 任务专属头 | 文本生成 | **统一 text-to-text** |
| 擅长 | 理解/分类 | 开放生成 | 长输入→短输出 |

**复杂度**（序列长 $n$、维度 $d$）：自注意力 $O(n^2 d)$。推理时 encoder 编码 $O(n_{src}^2 d)$ 只一次；decoder 每步 self-attn $O(n_{tgt}d)$ + cross-attn $O(n_{src}d)$。相对位置偏置只加 $O(1)$ 标量查表，不增渐进复杂度。

---

## 常见问题 / 高频追问

| 问题 | 一句话答案 |
|------|-----------|
| T5 全称？ | Text-to-Text Transfer Transformer |
| 训练数据？ | 自建 C4（清洗版 Common Crawl，约 750GB） |
| 优化器为什么用 Adafactor？ | 不存全量二阶动量，省显存，能撑 11B |
| sentinel token 是什么？ | 占位被遮 span 的特殊 token `<extra_id_0..99>` |
| 相对偏置加在哪？ | 注意力 logits（score）上，不是 embedding |
| Encoder 用因果掩码吗？ | 不用，双向无掩码 |
| cross-attn 的 K,V 来自哪？ | Encoder 最终输出（memory） |
| 共享词表了吗？ | Enc/Dec embedding 与输出投影三者权重共享 |
| 多语言版？ | mT5（101 种语言） |
| 指令微调版？ | FLAN-T5 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-algo/transformer/模型架构]] — Encoder-Decoder 与注意力机制的底层原理
- [[llm-algo/bert]] — Encoder-only 与 MLM 预训练，与 T5 的 span corruption 对照阅读

---

## 附：相关中文数据集与开源资源

### T5 PEGASUS（中文生成式预训练）
- https://github.com/ZhuiyiTechnology/t5-pegasus
- https://github.com/renmada/t5-pegasus-pytorch
- T5 PEGASUS：开源一个中文生成式预训练模型: https://zhuanlan.zhihu.com/p/359509608

### LCSTS_new（中文短摘要数据集）
- https://www.luge.ai/#/luge/dataDetail?id=10

生成式短摘要数据集，以微博原文为输入，1~2 句话的短摘要为输出。LCSTS_new 是中文短摘要最常用的 LCSTS 数据集的升级版，数据量与质量均显著提升；信息提炼时与原文的**事实一致性**需重点关注。

```json
{
  "id": 6,
  "summary": "中国游客大增多国放宽签证",
  "content": "①北京和上海户籍的游客可获得韩国多次签证；②“整容客”可以不经由韩国使领馆、直接在网上申请签证；③中泰免签的实施日期尚未敲定；④越南已向中国持通行证旅游的公民全面开放。"
}
```

### AdvertiseGen（广告文案生成数据集）
- https://www.luge.ai/#/luge/dataDetail?id=9

```json
{
  "content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞",
  "summary": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"
}
```

> 用法提示：T5/T5-PEGASUS 做中文摘要时输入前缀写 `"summarize: <原文>"`、目标为短摘要；做广告文案时输入结构化属性串、输出文案。统一走 text-to-text 接口即可微调。
