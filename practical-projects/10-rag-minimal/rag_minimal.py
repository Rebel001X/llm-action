# -*- coding: utf-8 -*-
"""
从零实现最小 RAG(检索增强生成 / Retrieval-Augmented Generation)
=================================================================

这个脚本用纯 numpy(无任何外部 API / 预训练模型 / 网络 / 数据集)演示
RAG 的完整闭环,目标是把"检索 -> 增强 -> 生成"这三步讲透:

    1. 知识库(Knowledge Base)
       内置几条简短文档,模拟企业内部知识 / 私域资料。
       LLM 训练时没见过这些事实,因此"凭参数硬答"必然编造(hallucination)。

    2. 向量化(Embedding)
       用 TF-IDF 把每条文档和用户 query 编码成向量。
       TF-IDF = 词频(TF) * 逆文档频率(IDF):一个词在某文档里出现得多
       (TF 高)、又在整个语料里很少见(IDF 高),就最能代表该文档主题。
       真实 RAG 这一步换成 BERT / BGE / OpenAI embedding 等稠密向量,
       但"把文本映射到可比较的向量空间"这一原理完全一致。

    3. 检索(Retrieval)
       对 query 向量与每条文档向量算余弦相似度(cosine similarity),
       取 Top-K 最相关文档。这就是向量数据库(Faiss / Milvus)做的事。

    4. 增强(Augment)
       把检索到的文档拼进 prompt 的 context 区,再附上问题。
       这一步把"模型不知道的外部知识"喂给了生成阶段。

    5. 生成(Generate)
       为了零依赖、可离线跑,这里用一个极简的"抽取式"生成器:
       在检索到的上下文里挑出与 query 最相关的那句话作为答案。
       它替代了真实 LLM 的 decoder,但承担的角色一模一样:
       "基于给定 context 产出答案"。

对照实验(本脚本的可量化信号)
    - No-RAG(闭卷):没有上下文,只能套模板/瞎猜 -> 在我们设计的事实题上
      命中率 ~0,演示"凭空编造"。
    - RAG(开卷):先检索再回答 -> 命中率显著上升(本 demo 达到 100%)。
    这个命中率差值,就是 RAG 价值的直接度量。

运行环境:Python 3 + numpy(CPU,几秒内跑完)。
注意:所有 print 仅用 ASCII / 英文,避免 Windows GBK 控制台的编码报错;
      中文说明只放在注释 / docstring / README 里。
"""

import re
import numpy as np

# 固定随机种子,保证可复现(本 demo 几乎不依赖随机,但保持习惯)
np.random.seed(42)


# ---------------------------------------------------------------------------
# 0. 内置知识库
#    每条都是一个独立"文档块"(chunk)。真实工程里这些来自 PDF/网页切片。
#    刻意写一些"模型预训练时不可能知道"的私域事实(虚构公司 Acme 的规章),
#    这样才能凸显:不检索 = 只能编造。
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "Acme Corp was founded in 2011 in the city of Greenfield by Dana Lee.",
    "The Acme Phoenix laptop ships with 32 gigabytes of memory and a 14 inch screen.",
    "Acme employees get 25 paid vacation days per year plus public holidays.",
    "The Acme support hotline operates from 9 am to 6 pm on weekdays only.",
    "Returns at Acme are accepted within 30 days of purchase with a valid receipt.",
    "The Phoenix laptop battery lasts about 12 hours of normal office use.",
    "Acme's headquarters moved to a new green building in 2020 to cut energy use.",
    "All Acme software updates are released on the first Tuesday of each month.",
]


# ---------------------------------------------------------------------------
# 1. 分词:最朴素的小写 + 取字母数字词。教学够用,真实工程用专门 tokenizer。
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lowercase and split text into alphanumeric word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# 2. 构建词表(vocabulary):语料里出现过的所有词 -> 列索引。
#    向量的每一维对应一个词,这样不同文本就落在同一坐标系里,可比较。
# ---------------------------------------------------------------------------
def build_vocab(documents):
    """Build a sorted {word: column_index} mapping from all documents."""
    vocab = {}
    for doc in documents:
        for word in tokenize(doc):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# 3. 计算 IDF(逆文档频率):衡量一个词的"区分度"。
#    出现在很多文档里的词(the, of, a)IDF 低 -> 几乎不携带主题信息;
#    只在少数文档里出现的词(phoenix, vacation)IDF 高 -> 是检索的关键信号。
#    公式:idf = log((1 + N) / (1 + df)) + 1   (加 1 平滑,避免除零/取 0)
# ---------------------------------------------------------------------------
def compute_idf(documents, vocab):
    """Compute the IDF weight for every vocabulary word."""
    n_docs = len(documents)
    df = np.zeros(len(vocab))  # df[w] = 含有词 w 的文档数
    for doc in documents:
        seen = set(tokenize(doc))  # 每个文档每个词只计一次
        for word in seen:
            df[vocab[word]] += 1.0
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    return idf


# ---------------------------------------------------------------------------
# 4. 把一段文本编码成 TF-IDF 向量,并做 L2 归一化。
#    归一化后,两向量点积 == 余弦相似度,检索时算起来更省事。
# ---------------------------------------------------------------------------
def embed(text, vocab, idf):
    """Encode text into an L2-normalized TF-IDF vector."""
    vec = np.zeros(len(vocab))
    tokens = tokenize(text)
    if not tokens:
        return vec
    # TF:词在本文本中的出现频率(除以总词数,抵消文本长短差异)
    for word in tokens:
        if word in vocab:          # 词表外的词忽略(OOV)
            vec[vocab[word]] += 1.0
    vec /= len(tokens)
    # TF * IDF:常见词被压低,关键词被放大
    vec *= idf
    # L2 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# 5. 检索:用 query 向量和文档矩阵算余弦相似度,返回 Top-K。
#    这正是向量数据库的核心操作(只是我们用稠密 numpy 暴力算)。
# ---------------------------------------------------------------------------
def retrieve(query, documents, doc_matrix, vocab, idf, top_k=2):
    """Return the top_k (doc_index, score) pairs most similar to the query."""
    q_vec = embed(query, vocab, idf)
    # 因为向量都已 L2 归一化,矩阵乘法直接得到每篇文档的余弦相似度
    scores = doc_matrix @ q_vec
    # 取分数最高的 top_k 个,并按分数从高到低排序
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# 6. 增强:把检索到的文档拼成带 context 的 prompt。
#    真实 RAG 把这个字符串发给 LLM;这里我们打印它并交给抽取式生成器。
# ---------------------------------------------------------------------------
def build_augmented_prompt(query, retrieved_docs):
    """Assemble a context-augmented prompt string (the 'A' in RAG)."""
    context = "\n".join("- " + doc for doc in retrieved_docs)
    prompt = (
        "Answer the question using ONLY the context below.\n"
        "If the context lacks the answer, say you do not know.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + query + "\n"
        "Answer:"
    )
    return prompt


# ---------------------------------------------------------------------------
# 7. 极简生成器(替代真实 LLM 的 decoder)。
#    策略:把每条检索到的上下文按句子拆开,选出与 query 词重叠最多的句子。
#    它"基于 context 产出答案",承担的角色与真实 LLM 一致,但完全可控、可离线。
# ---------------------------------------------------------------------------
def extractive_generate(query, retrieved_docs):
    """A tiny extractive 'generator': pick the context sentence best matching the query."""
    q_words = set(tokenize(query))
    # 去掉问句里的疑问词等停用词,避免它们干扰匹配
    stopwords = {"what", "when", "where", "who", "how", "is", "are", "the",
                 "a", "an", "of", "do", "does", "many", "long", "much"}
    q_keywords = q_words - stopwords

    best_sentence = None
    best_overlap = -1
    for doc in retrieved_docs:
        # 文档块本身就是单句,但保留按句切分以兼容多句 chunk
        for sentence in re.split(r"(?<=[.!?])\s+", doc):
            s_words = set(tokenize(sentence))
            overlap = len(q_keywords & s_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
    if best_sentence is None or best_overlap == 0:
        return "I do not know based on the provided context."
    return best_sentence.strip()


# ---------------------------------------------------------------------------
# 8. 对照组:不检索的"闭卷"回答。
#    没有外部知识,只能返回一个固定模板 / 占位答案 -> 演示"凭空编造"。
#    真实 LLM 在这种情况下会生成一段听起来合理但可能错误的文本。
# ---------------------------------------------------------------------------
def closed_book_answer(query):
    """Simulate a no-retrieval LLM: confidently produce a guess with no grounding."""
    # 这里用一个看似自信的虚构答案,代表 hallucination
    return "Acme was founded in 2008 and offers 15 vacation days (model guess, ungrounded)."


def keyword_hit(answer, expected_keywords):
    """Check whether the answer contains all expected keywords (case-insensitive)."""
    a = answer.lower()
    return all(kw.lower() in a for kw in expected_keywords)


# ===========================================================================
# DEMO
# ===========================================================================
def main():
    print("=" * 70)
    print("Minimal RAG from scratch (numpy, TF-IDF retrieval + extractive gen)")
    print("=" * 70)

    # ---- 离线建索引(真实工程里这步是预处理 / 入库)----
    vocab = build_vocab(KNOWLEDGE_BASE)
    idf = compute_idf(KNOWLEDGE_BASE, vocab)
    # 把每条文档编码成向量,堆成一个矩阵 [n_docs, vocab_size]
    doc_matrix = np.vstack([embed(doc, vocab, idf) for doc in KNOWLEDGE_BASE])
    print("\nKnowledge base : %d documents" % len(KNOWLEDGE_BASE))
    print("Vocabulary size: %d unique words" % len(vocab))
    print("Doc matrix     : shape %s (L2-normalized TF-IDF)" % str(doc_matrix.shape))

    # ---- 评测问题集:每题给出"正确答案必须包含的关键词"用于自动判分 ----
    # query / 检索 top_k / 期望关键词
    questions = [
        ("Who founded Acme and in what year?", ["dana lee", "2011"]),
        ("How many vacation days do Acme employees get?", ["25"]),
        ("How much memory does the Phoenix laptop have?", ["32"]),
        ("What are the support hotline hours?", ["9 am", "6 pm"]),
        ("How many days are returns accepted at Acme?", ["30 days"]),
    ]

    top_k = 2
    norag_hits = 0
    rag_hits = 0

    for qi, (query, expected) in enumerate(questions, 1):
        print("\n" + "-" * 70)
        print("Q%d: %s" % (qi, query))

        # ===== 对照组:No-RAG(闭卷)=====
        norag_ans = closed_book_answer(query)
        norag_ok = keyword_hit(norag_ans, expected)
        norag_hits += int(norag_ok)
        print("  [No-RAG ] answer: %s" % norag_ans)
        print("  [No-RAG ] correct? %s" % ("YES" if norag_ok else "NO"))

        # ===== RAG:检索 -> 增强 -> 生成 =====
        hits = retrieve(query, KNOWLEDGE_BASE, doc_matrix, vocab, idf, top_k=top_k)
        retrieved_docs = [KNOWLEDGE_BASE[i] for i, _ in hits]
        print("  [RAG    ] retrieved top-%d:" % top_k)
        for rank, (di, score) in enumerate(hits, 1):
            print("      #%d (cos=%.3f) %s" % (rank, score, KNOWLEDGE_BASE[di]))

        rag_ans = extractive_generate(query, retrieved_docs)
        rag_ok = keyword_hit(rag_ans, expected)
        rag_hits += int(rag_ok)
        print("  [RAG    ] answer: %s" % rag_ans)
        print("  [RAG    ] correct? %s" % ("YES" if rag_ok else "NO"))

    # ---- 汇总:用命中率量化 RAG 的收益 ----
    n = len(questions)
    print("\n" + "=" * 70)
    print("RESULTS over %d factual questions" % n)
    print("  No-RAG (closed book) accuracy: %d/%d = %.0f%%"
          % (norag_hits, n, 100.0 * norag_hits / n))
    print("  RAG    (retrieve+gen) accuracy: %d/%d = %.0f%%"
          % (rag_hits, n, 100.0 * rag_hits / n))
    gain = 100.0 * (rag_hits - norag_hits) / n
    print("  Accuracy gain from RAG        : +%.0f percentage points" % gain)
    print("=" * 70)

    # 成功信号:RAG 命中率应明显高于 No-RAG
    if rag_hits > norag_hits:
        print("SUCCESS: retrieval grounding reduced hallucination as expected.")
    else:
        print("WARNING: RAG did not outperform closed-book; check the setup.")


if __name__ == "__main__":
    main()
