# Embedding 模型

> 把"语言/文本"压成一串定长向量，让"语义相近"等价于"向量靠近"——这是 RAG、语义搜索、推荐、聚类的共同底座。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-application/vector-db/README]] [[llm-application/rag/embedding]]

## 阅读地图

| 节 | 主题 | 你将学到 | 难度 |
|----|------|---------|------|
| 0 | 一句话锚点 | embedding 到底是什么 | ★ |
| 1 | 地基/前置 | 向量、维度、为什么要定长 | ★ |
| 2 | 从 token 到句向量 | 池化（mean/CLS/last） | ★★ |
| 3 | 相似度度量 | cos / dot / L2 的关系 | ★★ |
| 4 | 对比学习训练 | InfoNCE、正负样本、温度 | ★★★ |
| 5 | 模型谱系 | Sentence-BERT / BGE / M3E / GTE / E5 | ★★ |
| 6 | BGE-M3 三合一 | Dense+Sparse+ColBERT | ★★★ |
| 7 | 为什么是 RAG 核心 | 召回质量决定上限 | ★★ |
| 8 | 评测 MTEB | 8 类任务、C-MTEB | ★★ |
| 9 | 选型决策 | 维度/语种/长度/成本 | ★★ |
| 10 | 数值例子/对照 | 手算 cos、显存、库规模 | ★★ |
| - | 常见问题 + 跳转 | 易错点速查 | ★ |

---

## 0. 一句话锚点

**Embedding 模型 = 一个函数 $f:\text{文本}\to \mathbb{R}^d$**，它把任意长度的文本映射到一个固定 $d$ 维实数向量，使得"人类觉得意思相近"的两段文本，其向量在几何上也"靠得近"。

```
  "猫坐在垫子上"  ──┐
                    ├─► f(·) ──► [0.12, -0.83, 0.05, ... ]  (d=1024 维)
  "一只猫在毯子上" ─┘                    ▲
                                  这两句的向量 cos 相似度 ≈ 0.91（近）
  "今天股市暴跌"   ──► f(·) ──► [-0.44, 0.30, 0.77, ...]
                                  与上面 cos ≈ 0.08（远）
```

关键词只有三个：**定长**（不管输入多长，输出永远 $d$ 维）、**稠密**（每一维都是有意义的实数，不是 one-hot）、**语义**（距离 ≈ 意思差异）。

---

## 1. 地基/前置

### 1.1 什么是"向量"，为什么用它表示语义

一个 $d$ 维向量就是 $d$ 个实数排成一列：$\mathbf{v}=(v_1,v_2,\dots,v_d)$。把每段文本变成空间里的一个**点**后，"语义关系"就变成了"几何关系"：

- 意思相近 → 两点距离小 / 夹角小
- 类比关系 → 向量平移（经典 `king - man + woman ≈ queen`）
- 主题聚类 → 同主题点扎堆成簇

```
  语义空间（示意，真实是 768/1024 维，这里压成 2 维）

        ^
        │   ● 狗            ● 猫
        │      ● 宠物
        │
        │                       ● 财报
        │              ● 股票  ● 利率
        ┼──────────────────────────────►
       动物簇                  金融簇
```

### 1.2 为什么必须"定长"

下游要做的事是**比较**和**检索**：算两段文本相不相似、在百万文档里找最近邻。这些操作（点积、建索引）都要求所有向量维度一致。可文本长度天差地别（3 个字 vs 3000 字），所以 embedding 模型的核心难点之一，就是把"变长 token 序列"压成"定长向量"——这一步叫**池化（pooling）**，见第 2 节。

### 1.3 词向量 vs 句向量（两代技术）

| | 词向量（Word2Vec/GloVe，2013） | 句向量（Sentence-BERT 起，2019） |
|---|---|---|
| 单位 | 单个词 → 一个静态向量 | 整段文本 → 一个向量 |
| 上下文 | 无（"苹果"水果/公司同一向量） | 有（Transformer 编码全句） |
| 一词多义 | 不能区分 | 能区分 |
| 用途 | 早期 NLP 特征 | RAG / 语义搜索 / 聚类 |

现代 embedding 模型几乎都是**句向量**，底座是 BERT 类双向 Encoder（或 LLM Decoder 改造），这正是用户笔记里提到的 "sentence-bert 类"。

---

## 2. 从 token 到句向量：池化

### 2.1 数据流全景

```
 输入文本 "猫坐在垫子上"
     │  ① 分词 (tokenizer)
     ▼
 [CLS] 猫 坐 在 垫 子 上 [SEP]        ← token 序列，长度 L=8
     │  ② Transformer Encoder (N 层)
     ▼
 token 向量矩阵 H ∈ R^{L×d}          ← 每个 token 一个 d 维向量
   h_CLS  h_猫  h_坐 ... h_SEP
     │  ③ 池化 (pooling)：L 个向量 → 1 个向量
     ▼
 句向量 e ∈ R^d                      ← 定长！
     │  ④ L2 归一化 (可选但常用)
     ▼
 单位向量 ê (‖ê‖=1)
```

第 ③ 步是把"变长"变"定长"的关键。

### 2.2 三种主流池化方式

**① Mean Pooling（均值池化）——最常用、最稳**

把所有有效 token 向量按位平均（要按 attention mask 排除 padding）：

$$ \mathbf{e}=\frac{\sum_{i=1}^{L} m_i\,\mathbf{h}_i}{\sum_{i=1}^{L} m_i},\qquad m_i\in\{0,1\}\ \text{为掩码} $$

直觉：句义是所有词义的"重心"。Sentence-BERT、BGE、GTE、E5 大多用它。

**② CLS Pooling（取首 token）**

直接拿 `[CLS]` 位置的输出 $\mathbf{h}_{\text{CLS}}$ 当句向量。BERT 预训练时 `[CLS]` 就被设计来代表全句，但**未经微调时它并不天然适合相似度**，需配对比学习训练才好用。

**③ Last-token Pooling（取末 token）**

LLM 类（Decoder-only，如 E5-Mistral、GTE-Qwen）因果注意力下只有最后一个 token "看过"全文，故取最后一个 token 的输出。

```
 mean:  [▓][▓][▓][▓][▓]  → 全部平均   ●
 cls:   [▓][ ][ ][ ][ ]  → 只取第一个 ●
 last:  [ ][ ][ ][ ][▓]  → 只取最后一个●
        CLS 词 词 词 SEP
```

经验：**Encoder 模型 → mean 最稳；Decoder/LLM 模型 → last-token**。池化方式必须和训练时一致，推理时换池化会严重掉点。

---

## 3. 相似度度量

得到两个句向量 $\mathbf{a},\mathbf{b}$ 后，怎么量化"近不近"？

### 3.1 三种度量

**余弦相似度 cosine（最常用）**——只看方向、不看长度：

$$ \cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}=\frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\sqrt{\sum_i b_i^2}}\in[-1,1] $$

**点积 dot product**——方向 + 长度都算：$\ \mathbf{a}\cdot\mathbf{b}=\sum_i a_i b_i$

**欧氏距离 L2**——空间直线距离：$\ \|\mathbf{a}-\mathbf{b}\|_2=\sqrt{\sum_i (a_i-b_i)^2}$

### 3.2 一个关键结论：归一化后三者等价

若先做 L2 归一化（$\|\mathbf{a}\|=\|\mathbf{b}\|=1$），则：

$$ \cos(\mathbf{a},\mathbf{b})=\mathbf{a}\cdot\mathbf{b},\qquad \|\mathbf{a}-\mathbf{b}\|_2^2=2-2\cos(\mathbf{a},\mathbf{b}) $$

```
 归一化把所有向量放到单位球面上：
        ┌─── 单位圆 (‖v‖=1) ───┐
       a●╲ θ                    │
        │  ╲                    │   • cos 大 ⇔ 夹角 θ 小 ⇔ L2 距离小
        │   ●b                  │   • 三者排序完全一致！
        └──────────────────────┘
```

**实践要点**：绝大多数 embedding 模型输出**先归一化**，此时检索用点积（最快）就等价于 cos，向量库（如 Faiss IP 索引）也按此假设优化。所以"该用 cos 还是 dot"在归一化前提下不是问题——它们一回事。

---

## 4. 对比学习训练（embedding 模型的灵魂）

### 4.1 为什么不能直接用 BERT 输出

裸 BERT 的句向量做相似度很差（各向异性、向量挤在锥形小区域）。要让"语义近=向量近"成立，必须用**对比学习（Contrastive Learning）显式训练**：拉近正样本对、推远负样本对。

### 4.2 正负样本怎么来

```
 一条训练样本 = (query, 正例 d+, 一堆负例 d1-,d2-,...)

  query:  "如何重置路由器"
  d+   :  "路由器恢复出厂设置的步骤"   ← 语义相关（正）
  d1-  :  "如何更换手机壳"             ← 不相关（负）
  d2-  :  "路由器的历史发展"           ← 难负例 hard negative（话题沾边但不答问）
```

- **正例**来源：人工标注问答对、点击日志、标题-正文、翻译对、同义改写。
- **负例**来源：① **In-batch negatives**——同一 batch 里别人的正例就当我的负例，免费且高效；② **Hard negatives**——用一个弱检索器挖"看着像但其实不对"的难负例，这是 BGE 等模型涨点的关键。

### 4.3 InfoNCE 损失（核心公式）

对一个 query $q$，正例 $d^+$，负例集合 $\{d^-_j\}$，相似度记 $s(\cdot,\cdot)$（通常是 cos），温度 $\tau$：

$$ \mathcal{L}=-\log\frac{\exp\big(s(q,d^+)/\tau\big)}{\exp\big(s(q,d^+)/\tau\big)+\sum_j \exp\big(s(q,d^-_j)/\tau\big)} $$

读法：这就是一个**以"正例是否被选中"为目标的 softmax 分类**。分子是正例打分，分母是正例+所有负例。loss 小 ⇔ 正例得分远高于所有负例。

- **温度 $\tau$**：缩放 logits。$\tau$ 小 → 分布尖锐 → 更狠地惩罚难负例（但易不稳）；典型 $0.01\!\sim\!0.05$。
- **batch 越大** → in-batch 负例越多 → 对比信号越强（故 embedding 训练偏好大 batch / 梯度缓存技巧）。

```
 训练目标几何图：

   训练前              训练后（对比学习拉/推）
   q  d+  d-           q ●──● d+   ← 拉近
   ●  ●   ●     ──►        ╲
    挤成一团               ●  d-   ← 推远
```

### 4.4 两阶段训练范式（BGE/E5 典型）

```
 阶段一：弱监督预训练
   海量"弱配对"(标题-正文、网页对、翻译对) + in-batch negatives
   → 学到粗粒度语义对齐
        │
        ▼
 阶段二：有监督微调
   高质量标注对 + 难负例挖掘 + (BGE) 指令前缀
   → 精修，刷高检索/重排质量
```

E5 还提出给 query/passage 加前缀（`query:` / `passage:`）；BGE 中文检索时建议给查询加指令前缀 `为这个句子生成表示以用于检索相关文章：`——这些"指令"让同一模型适配不同任务。

---

## 5. 模型谱系：Sentence-BERT / BGE / M3E / GTE / E5

```
 时间轴
 2019 ── Sentence-BERT ── 双塔 + 池化，开创句向量对比训练
 2022 ── E5 (微软) ────── 弱监督预训练 + query/passage 前缀
 2023 ── M3E (摩搜) ────── 中文社区早期高质量开源
      ── GTE (阿里) ────── 多阶段对比，small/base/large 全家桶
      ── BGE (智源BAAI) ── 中文检索标杆，C-MTEB 登顶
 2024 ── BGE-M3 ───────── 多语言/多功能/多粒度三合一（见第6节）
      ── GTE-Qwen / E5-Mistral ── LLM 当 backbone 的新一代
```

| 模型 | 出品方 | 特点 | 语种 | 典型维度 |
|------|--------|------|------|----------|
| **Sentence-BERT** | UKP | 句向量鼻祖，双塔孪生网络 | 英/多语 | 768 |
| **M3E** | MokaAI（摩搜） | 早期中文友好，社区流行 | 中/英 | 768 |
| **BGE**（bge-large-zh） | 智源 BAAI | 中文检索强，难负例+指令微调 | 中/英分版 | 1024 |
| **GTE**（gte-large） | 阿里达摩 | 多阶段对比，工程成熟 | 中/英/多 | 1024 |
| **E5**（e5-large） | 微软 | query/passage 前缀范式 | 多语 | 1024 |
| **BGE-M3** | 智源 BAAI | Dense+Sparse+ColBERT 三合一 | 100+ 语言 | 1024 |

> 用户笔记提到的 **llm-embedder**（智源）是 BGE 系列的"为 LLM 检索而生"的统一 embedding，覆盖知识/记忆/示例/工具四类检索；**sentence-bert 类**正是这一整支谱系的源头。

**Sentence-BERT 的关键创新**：BERT 算句对相似度原本要把两句拼起来过一次模型（交叉编码，$N^2$ 次太慢）；SBERT 改成**双塔/孪生（bi-encoder）**——两句各自独立编码成向量，离线存好，在线只算向量相似度，把检索从"过模型"降为"查向量库"，这才让大规模语义检索可行。

```
 交叉编码 cross-encoder      双塔 bi-encoder (SBERT/BGE...)
  [句A;句B] → 模型 → 分数      句A→模型→向量a ┐
   精准但每对都要跑模型         句B→模型→向量b ┘→ cos(a,b)
   →适合"重排 rerank"          可预存向量，海量检索 →"召回 recall"
```

记住这条分工：**bi-encoder 负责快速召回，cross-encoder（reranker）负责精排**，RAG 里两者常配合。

---

## 6. BGE-M3：一个模型三种检索

BGE-M3 把三种检索能力塞进同一个模型，一次前向同时产出三种表示：

```
                ┌──► Dense  : 整句一个稠密向量 (语义召回)
  文本 → BGE-M3 ─┼──► Sparse : 词级权重(类似 BM25 的可学习版，关键词召回)
                └──► ColBERT: 每个 token 一个向量 (细粒度后期交互)
```

- **Dense**：传统稠密向量，抓"语义"。
- **Sparse（Lexical）**：输出每个 token 的重要性权重，擅长"必须命中某关键词"的场景（型号、专有名词），弥补稠密向量"对精确词不敏感"的弱点。
- **Multi-Vector / ColBERT**：保留每个 token 的向量，检索时做 token 级 late-interaction（MaxSim），更精细但更占空间。

实践常把三路得分加权融合（hybrid），互补提升召回，同时支持最长约 8192 token、100+ 语言。这就是用户笔记里"智源的工作 BGE"的最新形态。

---

## 7. 为什么 Embedding 是 RAG 的核心

### 7.1 RAG 数据流里 embedding 出现两次

```
 离线建库                          在线问答
 ───────                          ───────
 文档 → 切块 chunk                 用户问题 query
   │ 【embedding 模型】              │ 【同一 embedding 模型】
   ▼                                ▼
 块向量 → 存入向量库 ◄──检索(ANN)── query 向量
                          │ top-k 命中块
                          ▼
                   拼进 Prompt → LLM 生成答案
```

### 7.2 为什么它是上限

RAG 的链路是 **检索 → 生成**。LLM 再强，**没召回到正确文档就一定答不对**（garbage in → garbage out）。embedding 决定了"语义匹配"这一步的质量，是整条链路的**召回天花板**：

- embedding 差 → 相关文档排不进 top-k → LLM 无米下炊 → 幻觉
- embedding 好 → 正确证据进入上下文 → LLM 只需"读着回答"

所以一句话：**RAG 的好坏，七分在召回，召回的好坏，七分在 embedding（+切块）**。这也是为什么生产里常再叠一个 **reranker（cross-encoder）** 精排 top-k，详见 [[llm-application/rag/embedding]]。

---

## 8. 评测：MTEB / C-MTEB

**MTEB（Massive Text Embedding Benchmark）** 是 embedding 模型的"高考"，用一个统一榜单衡量通用能力，避免"只在某个任务好"。

### 8.1 八大类任务

```
 MTEB
 ├─ Retrieval     检索（与 RAG 最相关，权重最高的参考）
 ├─ Reranking     重排
 ├─ Classification 分类（向量+逻辑回归）
 ├─ Clustering    聚类
 ├─ Pair Class.   句对二分类（是否同义/蕴含）
 ├─ STS           语义文本相似度（与人工打分相关性）
 ├─ Summarization 摘要质量评估
 └─ Bitext Mining 跨语言句对挖掘
```

- **C-MTEB** 是其中文版，选中文模型主要看它（BGE、GTE-zh 长期居前）。
- 主指标：Retrieval 看 **nDCG@10**，STS 看 **Spearman 相关系数**，分类看 **accuracy**。

### 8.2 看榜三条纪律

1. **看子任务而非总分**：你做 RAG 就重点看 Retrieval，别被总分误导。
2. **同语种比**：英文榜强不代表中文强，中文务必查 C-MTEB。
3. **防过拟合榜单**：榜分接近时，用**你自己的业务数据**做小样本评测才算数。

---

## 9. 选型决策

```
 选 embedding 模型，问自己 5 个问题：

 ① 语种?        中文为主 → BGE-zh / GTE-zh / M3E ; 多语 → BGE-M3 / E5-multi
 ② 文本多长?    短文本/句 → 任意 ; 长文档 → 看 max_length (BGE-M3 ~8k)
 ③ 维度预算?    省存储/快 → 512~768 ; 求精度 → 1024+ (或 Matryoshka 可裁维)
 ④ 部署方式?    自建 GPU → 开源(BGE/GTE) ; 图省事 → API(OpenAI/Cohere等)
 ⑤ 要不要 rerank? 召回够 → 只 bi-encoder ; 求精 → 加 cross-encoder 精排
```

| 维度 | 小（512-768） | 大（1024+） |
|------|--------------|------------|
| 存储/内存 | 省（约一半） | 费 |
| 检索速度 | 快 | 慢 |
| 精度上限 | 略低 | 略高 |
| 适合 | 海量库、低延迟 | 高精度问答 |

**Matryoshka 表示学习**：训练时让前 $k$ 维就能单独用，于是同一模型可"按需截断"到 768/512/256 维换取速度，无需重训——选型时优先这类弹性模型。

---

## 10. 数值例子 / 对照

### 10.1 手算余弦相似度

设两个 3 维向量（真实是 1024 维，这里缩小演示）：

$$ \mathbf{a}=(1,\;2,\;2),\qquad \mathbf{b}=(2,\;0,\;1) $$

点积：$\mathbf{a}\cdot\mathbf{b}=1\!\times\!2+2\!\times\!0+2\!\times\!1=4$

模长：$\|\mathbf{a}\|=\sqrt{1+4+4}=3,\quad \|\mathbf{b}\|=\sqrt{4+0+1}=\sqrt5\approx2.236$

$$ \cos(\mathbf{a},\mathbf{b})=\frac{4}{3\times2.236}\approx\frac{4}{6.708}\approx0.596 $$

结论：约 0.6，中等相关。归一化后 $\hat{\mathbf a}=(0.333,0.667,0.667)$、$\hat{\mathbf b}=(0.894,0,0.447)$，其点积 $=0.333\!\times\!0.894+0+0.667\!\times\!0.447\approx0.596$——**与上面 cos 完全一致**，验证了第 3.2 节"归一化后点积=cos"。

### 10.2 向量库规模 / 显存估算

设语料 **100 万** 个文档块，embedding 维度 $d=1024$，用 float32（4 字节/数）存储：

$$ \text{存储} = 10^6 \times 1024 \times 4\ \text{B} = 4.096\times10^9\ \text{B}\approx 3.8\ \text{GiB} $$

- 改用 **float16** 减半 ≈ 1.9 GiB；用 **int8 量化** 再减 ≈ 0.95 GiB。
- 若 $d$ 从 1024 降到 512（Matryoshka 截断），存储与检索计算量**直接砍半**。
- 加上 ANN 索引（HNSW）的图结构开销，实际内存约为原始向量的 1.2~1.5 倍，落地按 **约 5~6 GiB** 预留（以实测为准）。

### 10.3 编码吞吐粗算（仅数量级，具体以实测为准）

单张主流推理卡（约 312 TFLOPS 级 FP16，**以官方规格为准**）跑一个 base 级（约 1 亿参数）encoder，短文本批量编码常见在**约每秒数千到上万条**量级；建百万级库通常**分钟级**完成。瓶颈多在 batch、序列长度与 IO，而非纯算力。

---

## 常见问题

| 问题 | 答案 |
|------|------|
| cos 还是 dot？ | 向量已归一化时**两者等价**；多数模型默认归一化，用点积最快 |
| 池化能随便换吗？ | 不能，必须与训练一致，换了严重掉点（mean↔cls↔last 不通用） |
| 为什么裸 BERT 句向量差？ | 各向异性，需对比学习显式拉近正例、推远负例 |
| query 和 doc 用同一模型吗？ | 通常同一双塔；E5/BGE 给 query 加前缀/指令以区分角色 |
| 维度越高越好？ | 不一定；高维更准但更费存储/算力，看 Matryoshka 弹性裁剪 |
| embedding 和 reranker 区别？ | embedding=bi-encoder 快召回；reranker=cross-encoder 精排，二者互补 |
| 中文选哪个？ | 看 **C-MTEB**：BGE-zh / BGE-M3 / GTE-zh 优先，再用自有数据复测 |
| 长文档怎么办？ | 切块（chunk）+ 重叠；或用长上下文模型（BGE-M3 约 8k） |
| 难负例为什么重要？ | 提供"看着像但错"的强对比信号，是 BGE 等涨点关键 |
| 关键词召回不到怎么补？ | 上 hybrid：稠密向量 + Sparse/BM25（BGE-M3 一模型即可） |

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 向量数据库（存什么、ANN/HNSW 怎么建索引）：[[llm-application/vector-db/README]]
- RAG 中的 embedding 实践（切块、前缀、rerank 配合）：[[llm-application/rag/embedding]]
