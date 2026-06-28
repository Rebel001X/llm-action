# BERT (双向编码器)
> 用"完形填空"自监督预训练出的 Encoder-only 双向语言理解模型：每个词都能同时看到左右上下文，专攻分类/标注/抽取类任务，天生不会逐词生成。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/transformer/模型架构]] [[llm-algo/gpt2/模型架构]]

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | Encoder-only / 双向 |
| 1 | 地基：Transformer Encoder 复习 | Self-Attention / 残差 / LayerNorm |
| 2 | 为什么要"双向"，GPT 为什么不能直接双向 | 信息泄漏 / 因果掩码 |
| 3 | 输入表示：三种 Embedding + [CLS]/[SEP] | Token/Segment/Position |
| 4 | 预训练任务一：MLM 完形填空 | 15% 掩码 / 80-10-10 |
| 5 | 预训练任务二：NSP 句对关系 | IsNext / NotNext |
| 6 | 整体数据流（ASCII 全景图） | Embedding→Encoder→Head |
| 7 | 微调范式：一个模型四种任务头 | 句分类/标注/QA/句对 |
| 8 | 为什么 BERT 不能"生成" | 无因果掩码 / 无自回归 |
| 9 | 规模配置与手算参数量 | Base 110M / Large 340M |
| 10 | HuggingFace 类层次对照源码 | BertModel / BertForXxx |
| — | 数值手算（参数/显存/MLM损失/注意力） | 逐数演算 |
| — | 对照表 + 高频面试问答 | BERT vs GPT |

## 0. 一句话锚点

- **BERT = Bidirectional Encoder Representations from Transformers**（2018, Google）。
- 它只取了 Transformer 的**编码器（Encoder）那一半**，堆叠 L 层。
- 训练方式是**自监督**：把句子里随机一些词盖住（[MASK]），让模型根据**左右两边**的词把它猜回来（MLM）；外加判断两句话是否相邻（NSP）。
- 产出是**每个 token 的上下文向量**和**整句的 [CLS] 向量**，用来做下游"理解"任务：情感分类、命名实体识别、问答抽取、句子对匹配。
- **它不生成文本**——因为它训练时每个位置都偷看了未来，没有"只根据前文预测下一个词"的能力。这正是它与 [[llm-algo/gpt2/模型架构]] 的根本分水岭。

记住一个对偶：**GPT 是"会写作的预言家"（单向、自回归、生成）；BERT 是"会读懂的考官"（双向、并行、理解）**。

## 1. 地基：先把 Transformer Encoder 复习清楚

BERT 没有发明新的网络结构，它的骨架就是 [[llm-algo/transformer/模型架构]] 里的 Encoder。一层 Encoder 做两件事：

**(a) 多头自注意力（Multi-Head Self-Attention）**——让每个词去"查"句子里所有词，按相关度加权汇聚信息。核心公式：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

- $Q=XW_Q,\ K=XW_K,\ V=XW_V$，$X$ 是输入向量序列。
- $\frac{1}{\sqrt{d_k}}$ 是缩放：点积随维度 $d_k$ 增大而方差变大，会把 softmax 推到饱和区（梯度消失），除以 $\sqrt{d_k}$ 把方差拉回 1 量级。
- 多头 = 把 $d$ 维切成 $h$ 份各自做注意力再拼接，让不同头关注不同语法/语义关系。

**(b) 前馈网络（FFN）**——逐位置的两层 MLP，中间维度放大 4 倍，激活用 **GELU**：

$$\text{FFN}(x)=\text{GELU}(xW_1+b_1)W_2+b_2,\quad W_1\in\mathbb{R}^{d\times 4d}$$

两个子层都包了**残差 + LayerNorm**：$\text{LN}(x+\text{Sublayer}(x))$（BERT 用 Post-LN）。

```
一层 Encoder（BertLayer）：
   x ──►┌───────────────┐
        │ Self-Attention│──► +x ──► LayerNorm ──┐
        └───────────────┘                       │
   ┌────────────────────────────────────────────┘
   └──►┌───────────────┐
       │  FFN (4d GELU)│──► + ──► LayerNorm ──► 输出
       └───────────────┘
```

**关键差异（先埋钩子）**：Transformer 原论文的 Encoder 自注意力是**全可见**的（没有掩码），Decoder 才有"因果掩码"。BERT 直接用了**全可见的 Encoder**——这就是"双向"的来源。

## 2. 为什么非要"双向"？GPT 为什么做不到？

理解一个词的含义，常常需要**右边**的信息。例：

```
句子：  我  把  钱  存  进  了  ___  里
候选：  银行(bank) ?  河岸(bank) ?
```

要判断 ___ 是"银行"还是别的，光看左边"我把钱存进了"不够确定，看到右边"里"会更稳。**理想的语言理解应当同时利用左右上下文**。

- **GPT（单向）**：用**因果掩码（causal mask）**强制每个位置只能看 $\le t$ 的词，目标是 $P(x_t\mid x_{<t})$。这天然适合"逐词生成"，但每个词丢掉了右侧信息。
- **能不能让 GPT 直接双向？不能。** 如果去掉因果掩码做"预测下一个词"，位置 $t$ 的输入里就**直接含有答案 $x_t$**（它能看到自己和未来），任务退化成"把输入抄到输出"，学不到东西。这叫**信息泄漏（label leakage）**。

```
单向(GPT)注意力可见性          双向(BERT)注意力可见性
       t1 t2 t3 t4                  t1 t2 t3 t4
  t1 [ ■  .  .  . ]            t1 [ ■  ■  ■  ■ ]
  t2 [ ■  ■  .  . ]            t2 [ ■  ■  ■  ■ ]
  t3 [ ■  ■  ■  . ]            t3 [ ■  ■  ■  ■ ]
  t4 [ ■  ■  ■  ■ ]            t4 [ ■  ■  ■  ■ ]
  ■=可见 .=被掩码               全部可见，无因果掩码
```

**BERT 的破局思路**：既然"预测下一个词"会泄漏，那就换个不会泄漏的目标——**把要预测的词从输入里盖掉（[MASK]）**，让模型用其余左右词去猜。这样既保留了双向可见性，又没有泄漏。这就是下面第 4 节的 MLM。

## 3. 输入表示：三种 Embedding 相加 + 两个特殊符号

BERT 的每个输入 token 的向量 = **三种 Embedding 逐元素相加**（对应源码 `BertEmbeddings`）：

$$E_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$$

```
原始:    [CLS]   我    爱    NLP  [SEP]   它    很    难   [SEP]
          │      │     │     │     │      │     │     │     │
Token  E: e_CLS  e_我  e_爱  e_NLP e_SEP  e_它  e_很  e_难  e_SEP
Segment: E_A    E_A   E_A   E_A   E_A    E_B   E_B   E_B   E_B   ← 区分句A/句B
Position:P_0    P_1   P_2   P_3   P_4    P_5   P_6   P_7   P_8   ← 可学习位置(非正弦)
          └────────────── 三者相加 ──────────────┘
                          ↓ LayerNorm + Dropout
                  送入第一层 Encoder
```

**[CLS]（classification）**：永远放在序列**最前面**。它经过所有层注意力后，会聚合全句信息，最终的 [CLS] 向量被当作"整句表示"，接一个分类头即可做句子级任务。

**[SEP]（separator）**：句子**分隔/结束符**。单句任务结尾放一个 [SEP]；句对任务用 [SEP] 隔开句 A 和句 B。

**Segment Embedding**：只有两个向量 $E_A,E_B$，告诉模型"这个 token 属于第几句"，让 NSP 和句对任务能区分两句。

**Position Embedding**：BERT 用**可学习的**位置向量（不是 Transformer 原版的正弦函数），最大长度 512，所以 BERT 输入**最长 512 个 token**。

**分词器**：WordPiece（子词），词表约 30522。未登录词拆成子词，如 `playing → play ##ing`，`##` 表示词内续接。

## 4. 预训练任务一：MLM（Masked Language Model，完形填空）

**做法**：随机选语料中 **15%** 的 token 作为预测目标。对被选中的 token，按 **80/10/10** 处理：

| 比例 | 处理 | 例（原词 play） | 为什么 |
|------|------|------|--------|
| 80% | 换成 [MASK] | `[MASK]` | 主任务：靠上下文还原 |
| 10% | 换成随机词 | `apple` | 防模型只在见到[MASK]时才认真；逼它对每个词都建表示 |
| 10% | 保持原词 | `play` | 缓解"预训练有[MASK]、微调没[MASK]"的分布失配 |

**为什么留 10% 随机 / 10% 不变？** 因为下游微调时**输入里根本没有 [MASK]**。如果训练时被选中的 100% 都变成 [MASK]，模型会学成"只有看到 [MASK] 才输出有意义的预测"，造成预训练-微调的输入分布不一致。掺入随机词和原词，迫使模型对**每一个**位置都维持准确的上下文表示。

**损失**：只在**被掩码的位置**算交叉熵（其余位置不回传 MLM 损失）：

$$\mathcal{L}_{\text{MLM}}=-\frac{1}{|M|}\sum_{i\in M}\log P(x_i\mid x_{\setminus M})$$

$M$ 是被掩码位置集合，$x_{\setminus M}$ 是未掩码的上下文。预测头（`BertOnlyMLMHead`）把每个掩码位置的 $d$ 维向量经一层变换后投到词表大小 $V$ 上 softmax。

```
输入:  我  爱  [MASK]  这个  框架
                  │
        ┌─────────┴──────────┐ 双向 Encoder（左右都能看）
        ▼                    ▼
   "爱""这个""框架" 等上下文都参与
        │
   预测头 → softmax over 词表(30522)
        │
   目标:  NLP   ← 算交叉熵损失
```

**与 GPT 对比**：GPT 是 $P(x_t\mid x_{<t})$，每个位置都贡献损失、自回归；MLM 是 $P(x_i\mid \text{左右})$，只有 15% 位置贡献损失（所以 MLM **样本效率较低**，需要更多步数才收敛）。

## 5. 预训练任务二：NSP（Next Sentence Prediction，句对关系）

很多下游任务是**两句话之间的关系**（问答、自然语言推理、句对匹配）。NSP 让模型学会句间关系：

- 50% 正例：B 是 A 在原文中的**真实下一句** → 标签 `IsNext`。
- 50% 负例：B 从语料**随机抽**一句 → 标签 `NotNext`。
- 用 **[CLS]** 位置的最终向量过一个二分类头（`BertOnlyNSPHead`）判断。

```
[CLS] 句A的token... [SEP] 句B的token... [SEP]
  │
  └─► 取 [CLS] 向量 ─► 线性二分类 ─► {IsNext, NotNext}
```

**总损失** = 两个任务相加：$\mathcal{L}=\mathcal{L}_{\text{MLM}}+\mathcal{L}_{\text{NSP}}$（对应 `BertForPreTraining` 同时输出两个 head）。

**后续争议**：RoBERTa 实验发现 NSP **几乎没用甚至有害**，去掉 NSP、只做 MLM 并用更大数据/更长训练反而更好。ALBERT 把 NSP 换成更难的 SOP（句序预测）。所以 NSP 是"BERT 原版设计，但非必需"——面试常考点。

## 6. 整体数据流：一张 ASCII 全景图

```
                       文本: "[CLS] 我 爱 NLP [SEP]"
                                  │  WordPiece 分词 + 加特殊符
                                  ▼
   ┌──────────────── BertEmbeddings ────────────────┐
   │  Token-E + Segment-E + Position-E → LN → Dropout │
   └──────────────────────┬──────────────────────────┘
                          ▼   (seq_len × hidden)
   ┌─────────────── BertEncoder (L 层) ───────────────┐
   │  ┌── BertLayer ──────────────────────────────┐   │
   │  │ BertAttention: SelfAttn → SelfOutput(+LN)  │   │  ×12(Base)
   │  │ BertIntermediate(4d,GELU) → BertOutput(+LN)│   │  或×24(Large)
   │  └────────────────────────────────────────────┘   │
   └──────────────────────┬──────────────────────────┘
                          ▼
        ┌─────────────────┴──────────────────┐
        ▼                                     ▼
  序列输出(每个token向量)              [CLS]向量 → BertPooler(Tanh)
        │                                     │
   ┌────┴─────┐                          ┌────┴─────┐
   MLM头/标注/QA                      NSP/句子分类
```

**两个出口要记牢**：
1. **sequence_output**：形状 `(batch, seq_len, hidden)`，每个 token 一个向量 → 用于**token 级**任务（NER、QA 起止位置、MLM）。
2. **pooled_output**：`BertPooler` 取 [CLS] 向量再过 `Linear+Tanh`，形状 `(batch, hidden)` → 用于**句子级**任务（分类、NSP）。

## 7. 微调范式：一个预训练模型，四类任务头

BERT 的范式是 **"Pre-train once, fine-tune many"**：预训练好的主干（`BertModel`）参数复用，下游只需**加一个很薄的任务头**，再用任务数据**端到端微调**（主干也一起更新，学习率很小，如 2e-5～5e-5，训练 2～4 个 epoch）。

| 任务类型 | 用哪个输出 | 加的头 | HF 类 | 例子 |
|----------|-----------|--------|-------|------|
| 单句/句对分类 | pooled [CLS] | Linear→softmax | `BertForSequenceClassification` | 情感、NLI、句对相似 |
| 序列标注 | 每个token | Linear（逐位置）| `BertForTokenClassification` | NER、词性标注 |
| 抽取式问答 | 每个token | 两个Linear（起/止）| `BertForQuestionAnswering` | SQuAD：预测答案span起止 |
| 多选 | pooled [CLS] | Linear打分 | `BertForMultipleChoice` | SWAG常识推理 |

```
       ┌── 预训练大模型 BertModel (冻结结构, 微调权重) ──┐
       │                                              │
   情感分类          NER             SQuAD问答      句对匹配
   [CLS]→2类      每token→标签    每token→起/止   [CLS]→匹配
```

**为什么有效**：MLM+NSP 让主干学到了通用的语法、语义、世界知识；下游任务只是在这套表示上"接一个分类器"，所需标注数据大幅减少，效果还显著超过从零训练。这就是 BERT 引爆 NLP "预训练-微调"范式的根本原因。

## 8. 为什么 BERT 不能"生成"文本？

这是最高频的考点，把三条原因讲透：

**(1) 没有因果掩码，无法定义自回归概率。** 生成需要逐词建模 $P(x_t\mid x_{<t})$，要求位置 $t$ **看不到** $x_{\ge t}$。BERT 的注意力是全可见的，每个位置都偷看了未来，无法给出"只依赖前文"的条件分布。

**(2) 训练目标不是"预测下一个"。** MLM 是"还原被盖住的词"，预测分布是 $P(x_i\mid \text{左右上下文})$。即使你想让它一个词一个词地吐，它从没学过"给定前缀预测续写"，输出不连贯。

**(3) 解码无法并行展开成序列。** 自回归生成靠"上一步输出喂给下一步"，BERT 一次性并行编码整段、没有这种逐步状态。硬要用 BERT 生成（如对每个 [MASK] 取 argmax 再迭代）效率低且质量差，属于研究性玩法（Mask-Predict），不是它的设计目的。

```
GPT 生成:  前缀 → 预测t → 拼回 → 预测t+1 → ...  (自回归, 逐步)
BERT 理解: 整句一次性双向编码 → 输出每个token的"看懂了"向量 (并行, 一次)
```

一句话：**双向是"理解"的福音，却是"生成"的死刑**——能看到未来就无法诚实地预测未来。要生成请用 Decoder-only（GPT）或 Encoder-Decoder（T5/BART）。

## 9. 规模配置与参数量手算

| 配置 | 层数 L | 隐藏维 H | 头数 A | 中间维 | 参数量 |
|------|--------|---------|--------|--------|--------|
| BERT-Base | 12 | 768 | 12 | 3072 | 约 110M |
| BERT-Large | 24 | 1024 | 16 | 4096 | 约 340M |

> 数字以原论文为准。词表 V≈30522，最大序列长 512。

**手算 BERT-Base 参数量（量级核对，约值）**：

设 $H=768,\ L=12,\ V=30522$，中间维 $4H=3072$。

- **Embedding 层**：
  - Token：$V\times H = 30522\times768 \approx 23.4\text{M}$
  - Position：$512\times768 \approx 0.39\text{M}$
  - Segment：$2\times768 \approx 0.0015\text{M}$
  - 小计 ≈ **23.8M**
- **每层 Encoder**：
  - 自注意力 $Q,K,V,O$ 四个矩阵：$4\times H\times H = 4\times768^2 \approx 2.36\text{M}$
  - FFN 两层：$H\times4H + 4H\times H = 2\times768\times3072 \approx 4.72\text{M}$
  - 偏置+LN 忽略小项，单层 ≈ **7.08M**
- **12 层**：$12\times7.08 \approx 85\text{M}$
- **总计**：$23.8 + 85 \approx 108.8\text{M} \approx 110\text{M}$ ✓ 与官方一致。

**结论**：参数主要堆在 **Encoder 各层（约 78%）** 和 **Token Embedding（约 22%）**。

## 10. HuggingFace 类层次对照（把源码骨架讲活）

文件顶部那串类，正是 HF `transformers` 里 BERT 的真实结构。逐个对上：

```
BertModel  ←── 主干，输出 sequence_output + pooled_output
 ├─ BertEmbeddings          ← 第3节：Token+Segment+Position+LN
 ├─ BertEncoder             ← 第6节：L 层堆叠
 │   └─ BertLayer  (×L)     ← 一层 Encoder
 │       ├─ BertAttention
 │       │   ├─ BertSelfAttention   ← 算 Q/K/V、softmax 注意力
 │       │   └─ BertSelfOutput      ← 投影 + 残差 + LayerNorm
 │       ├─ BertIntermediate        ← FFN 第1层(放大4倍, GELU)
 │       └─ BertOutput              ← FFN 第2层 + 残差 + LayerNorm
 └─ BertPooler              ← 取[CLS]→Linear+Tanh→pooled_output

预训练头：
 BertPredictionHeadTransform   ← MLM头里的 Dense+GELU+LN
 BertLMPredictionHead          ← 投到词表V(权重常与Token-Embedding绑定/tied)
 BertOnlyMLMHead               ← 仅 MLM
 BertOnlyNSPHead               ← 仅 NSP(二分类)
 BertPreTrainingHeads          ← MLM + NSP 一起

下游封装(BertPreTrainedModel 子类)：
 BertForPreTraining            ← MLM+NSP, 复现预训练
 BertForMaskedLM               ← 只做完形填空/MLM 继续训练
 BertForNextSentencePrediction ← 只判句对
 BertForSequenceClassification ← 句/句对分类(用pooled)
 BertForTokenClassification    ← 序列标注NER(用sequence_output)
 BertForQuestionAnswering      ← 抽取式QA(起止两个logits)
 BertForMultipleChoice         ← 多选打分
 BertLMHeadModel               ← 给"BERT当decoder"用(EncoderDecoder架构里，需配因果掩码)
```

**读源码要点**：`BertForXxx` = `BertModel`（共享预训练权重）+ 一个轻量 head。换任务 = 换 head + 喂任务数据微调，主干几乎不动。这就是第 7 节范式在代码里的样子。

## 数值手算合集

**手算 1：一条样本掩码多少个词？**
序列含 128 个有效 token，掩码比例 15% → $128\times0.15\approx 19$ 个位置被选中。其中约 $19\times0.8\approx 15$ 个变 [MASK]，$2$ 个变随机词，$2$ 个保持原样。**MLM 损失只在这 19 个位置上计算**，其余 109 个位置不回传 MLM 梯度。

**手算 2：MLM 单点损失。**
若某 [MASK] 处模型对正确词 "NLP" 给出概率 $p=0.2$，该位置交叉熵 $=-\ln 0.2\approx 1.609$。若 19 个掩码位平均概率 0.2，则 $\mathcal{L}_{\text{MLM}}\approx 1.609$。训练良好后正确词概率升到 0.9，损失降到 $-\ln 0.9\approx 0.105$。

**手算 3：注意力打分缩放。**
$H=768,\ A=12$ → 每头 $d_k=768/12=64$。缩放因子 $1/\sqrt{64}=1/8=0.125$。若某 $q\cdot k$ 原始点积 $=40$，缩放后 $=40\times0.125=5$，再进 softmax，避免数值过大导致梯度饱和。

**手算 4：自注意力计算量随长度平方增长。**
注意力矩阵 $QK^\top$ 形状 $(\text{seq},\text{seq})$。序列 128 → $128^2=16384$ 对打分；序列 512 → $512^2=262144$，是前者的 **16 倍**。这就是 BERT 最长 512、长文本吃力的根因（复杂度 $O(n^2)$）。

**手算 5：推理显存粗估（Base, FP16, 单条）。**
权重 $110\text{M}\times2\text{B}\approx 220\text{MB}$。激活（一层约 $\text{seq}\times H\times2$B，$128\times768\times2\approx0.2$MB，×12层×若干中间量）量级几十 MB。**故单条推理约 0.3GB 量级**；微调还需存梯度+优化器状态（Adam 约权重的 2 倍），训练显存 ≈ $220\text{MB}\times(1+1+2)\approx 0.9$GB 加激活，故 Base 微调在消费级显卡可行。

## 对照表：BERT vs GPT vs Encoder-Decoder

| 维度 | BERT (Encoder-only) | GPT (Decoder-only) | T5/BART (Enc-Dec) |
|------|--------------------|--------------------|--------------------|
| 注意力 | 双向全可见 | 单向因果掩码 | 编码双向+解码因果 |
| 预训练目标 | MLM(+NSP) | 自回归 LM | Span 去噪/重排 |
| 损失覆盖 | 仅15%掩码位 | 每个位置 | 视方案 |
| 能否生成 | **不能** | 能 | 能 |
| 擅长 | 理解/分类/抽取 | 续写/对话/生成 | 翻译/摘要(seq2seq) |
| 输入长度 | ≤512 | 较长 | 较长 |
| 经典规模 | 110M/340M | 117M→175B+ | 60M→11B |
| 特殊符 | [CLS][SEP] | 无(或BOS/EOS) | 任务前缀 |

## 常见问题 / 高频面试追问

| 问题 | 踩点答案 | 追问/陷阱 |
|------|---------|----------|
| BERT 为什么叫"双向"？ | 自注意力无因果掩码，每个位置可见左右全部 token | 追问：BiLSTM 也双向，差别？答：BiLSTM 是两个单向 LSTM**拼接**(浅层双向)，BERT 是每层注意力**深度双向**联合 |
| 为什么用 MLM 而不是普通 LM？ | 双向下做"预测下一词"会信息泄漏；掩码把答案从输入移除即可双向无泄漏 | 代价：样本效率低(仅15%位贡献损失)、有[MASK]失配 |
| 80/10/10 各为什么？ | 80%[MASK]主任务；10%随机逼模型对每词建表示；10%原词缓解预训练-微调失配(下游无[MASK]) | 陷阱：别答成"为了数据增强" |
| [CLS]/[SEP] 作用？ | [CLS]聚合全句→句级分类；[SEP]分隔句对/标识结束；配 Segment-E 区分两句 | 追问：[CLS]为何能代表全句？因经过所有层注意力聚合了全序列信息 |
| NSP 有用吗？ | 原版用，但 RoBERTa 证明去掉更好，ALBERT 换成 SOP | 说明 NSP 太简单(主题区分即可猜对) |
| BERT 能做生成吗？为什么？ | 不能：无因果掩码→泄漏、目标非"预测下一词"、无自回归解码状态 | 想生成→用 GPT/T5/BART |
| BERT 与 GPT 根本区别？ | 双向 Encoder/理解 vs 单向 Decoder/生成；目标 MLM vs LM | 别只说"一个编码一个解码"，要点出可见性与目标 |
| 微调时改了什么？ | 加薄任务头，主干小学习率端到端微调(2e-5量级,2-4 epoch) | 追问：能否冻结主干只训头？可(feature-based)，但全微调通常更好 |
| 输入最长 512 的原因？ | Position-E 表只学到512；注意力 $O(n^2)$ 开销 | 长文档→Longformer/分块 |
| Position 用可学习还是正弦？ | BERT 用**可学习**位置嵌入 | 与原版 Transformer 正弦编码不同 |
| 参数量怎么估到 110M？ | Embedding≈24M + 12层×7M≈85M ≈109M | 见第9节手算 |

## 🔗 跳转链接

- 知识总图：[[00-知识地图]]
- 它的骨架来源（Encoder 细节）：[[llm-algo/transformer/模型架构]]
- 它的对偶（单向生成式）：[[llm-algo/gpt2/模型架构]]
