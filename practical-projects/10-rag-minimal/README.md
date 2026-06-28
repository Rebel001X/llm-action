# 从零实现最小 RAG(检索增强生成)

一个**零依赖、CPU 几秒跑通**的 RAG(Retrieval-Augmented Generation,检索增强生成)最小实现。
只用 `numpy`,不连任何 API、不下载任何模型/数据集,把 RAG 的完整闭环讲清楚:
**检索 (Retrieve) → 增强 (Augment) → 生成 (Generate)**。

---

## 一、演示什么原理

大模型的知识被"冻结"在训练好的参数里,对于训练时没见过的**私域 / 时效性事实**,
它只能"凭参数硬答",于是产生**幻觉(hallucination)**——一本正经地编造。

RAG 的思路:**先去外部知识库检索相关资料,把资料塞进 prompt 再让模型作答**,
从而把答案"锚定(grounding)"在真实证据上。本脚本把这五步全部手写出来:

1. **知识库**:内置 8 条虚构公司 "Acme" 的规章/产品事实(模型预训练时绝不可能知道)。
2. **向量化(Embedding)**:用 **TF-IDF** 把每条文档和用户问题编码成向量。
   - `TF`(词频)× `IDF`(逆文档频率):一个词在某文档里出现多、在全语料里又稀有,
     就最能代表该文档主题。真实 RAG 把这步换成 BERT/BGE/OpenAI 稠密向量,原理一致。
3. **检索(Retrieve)**:对"问题向量"与每条"文档向量"算**余弦相似度**,取 Top-K。
   这正是向量数据库(Faiss/Milvus)做的事。
4. **增强(Augment)**:把检索到的文档拼进 prompt 的 `Context:` 区,再附上问题。
5. **生成(Generate)**:为保证离线可跑,用一个极简**抽取式生成器**(在上下文里挑与问题
   词重叠最多的句子)替代真实 LLM 的 decoder——它承担的角色与 LLM 完全一致:
   "**基于给定 context 产出答案**"。

**对照实验**(脚本里的可量化信号):
- `No-RAG`(闭卷):没有上下文 → 套用一个虚构的"自信猜测",在事实题上命中率 **0%**。
- `RAG`(开卷):先检索再回答 → 命中率 **100%**。
- 两者差值就是 RAG 价值的直接度量。

---

## 二、怎么跑

```bash
cd practical-projects/10-rag-minimal
python rag_minimal.py
```

环境:Python 3 + numpy(CPU,几秒内跑完)。无需 GPU / 网络 / API key。

> 注:脚本所有 `print` 仅用英文/ASCII,避免 Windows GBK 控制台的 `UnicodeEncodeError`;
> 中文说明都放在注释和本 README 里。

---

## 三、预期输出(真实跑通摘录)

建索引阶段:

```
Knowledge base : 8 documents
Vocabulary size: 82 unique words
Doc matrix     : shape (8, 82) (L2-normalized TF-IDF)
```

单题示例(可看到余弦相似度分数 + 命中的文档):

```
Q1: Who founded Acme and in what year?
  [No-RAG ] answer: Acme was founded in 2008 and offers 15 vacation days (model guess, ungrounded).
  [No-RAG ] correct? NO
  [RAG    ] retrieved top-2:
      #1 (cos=0.371) Acme Corp was founded in 2011 in the city of Greenfield by Dana Lee.
      #2 (cos=0.184) Acme employees get 25 paid vacation days per year plus public holidays.
  [RAG    ] answer: Acme Corp was founded in 2011 in the city of Greenfield by Dana Lee.
  [RAG    ] correct? YES
```

最终对照汇总(成功信号):

```
RESULTS over 5 factual questions
  No-RAG (closed book) accuracy: 0/5 = 0%
  RAG    (retrieve+gen) accuracy: 5/5 = 100%
  Accuracy gain from RAG        : +100 percentage points
======================================================================
SUCCESS: retrieval grounding reduced hallucination as expected.
```

**结论**:闭卷 0% → 检索增强 100%,RAG 把幻觉直接消除,命中率提升 **+100 个百分点**。

---

## 四、对应 llm-action 文档

- RAG 总览与方案:[`../../llm-application/rag`](../../llm-application/rag)
  - [`README.md`](../../llm-application/rag/README.md)
  - [`方案.md`](../../llm-application/rag/方案.md)
  - [`embedding.md`](../../llm-application/rag/embedding.md)
  - [`存在的一些问题.md`](../../llm-application/rag/存在的一些问题.md)
- Embedding 模型:[`../../llm-application/embbedding-model.md`](../../llm-application/embbedding-model.md)
- 向量数据库:[`../../llm-application/vector-db`](../../llm-application/vector-db)
- 应用全景:[`../../llm-application`](../../llm-application)

---

## 五、社区参考

- RAG 原始论文:[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- 本仓库 RAG 应用文档:[llm-application/RAG](../../llm-application/rag)
- 向量检索库 [Faiss](https://github.com/facebookresearch/faiss) /
  开源中文 Embedding [BGE](https://github.com/FlagOpen/FlagEmbedding)

---

## 六、局限 / 与真实工程的差异

本实现刻意做成"玩具"以求清晰可跑,与生产级 RAG 的差距:

| 环节 | 本 demo | 真实工程 |
| --- | --- | --- |
| 向量化 | TF-IDF(稀疏、词面匹配) | 稠密语义向量(BERT/BGE/OpenAI),能匹配同义/改写 |
| 检索 | numpy 暴力点积(O(N)) | 向量数据库 + ANN 索引(HNSW/IVF),百万级毫秒召回 |
| 切分 | 整句即一个 chunk | 滑动窗口 + 重叠 + 语义切分,处理长文档 |
| 生成 | 抽取式(挑最相关句子) | 真实 LLM 生成,会综合/改写多条证据 |
| 知识库 | 内置 8 条 | 海量文档,需增量入库、去重、更新 |

**已知失败模式(教学要点)**:TF-IDF 是**词面匹配**,存在**词表不匹配(vocabulary mismatch)**问题。
例如问 "return **policy window**" 而文档写的是 "**returns** ... within 30 days ... **receipt**",
因关键词不重合(且 `return` vs `returns` 不做词干归并)会**检索失败**——这正是真实场景里要升级到
**稠密语义检索**的原因。本 demo 的问句已对齐语料用词以保证 100% 命中;把 Q5 改回
"What is the return policy window?" 即可复现这一失败,直观体会稀疏检索的边界。

其他生产关注点:检索召回率/精确率评测、重排序(reranker)、引用溯源、超长上下文压缩、
多路召回(向量 + 关键词 BM25 混合)、防注入与权限隔离等。
