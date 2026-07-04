# -*- coding: utf-8 -*-
"""检索排序 & recall@k 正确性测试。"""

import os
import sys

# 让 tests/ 能 import 上一级的项目模块（无需安装包）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import load_corpus, load_queries, gold_map          # noqa: E402
from chunking import build_chunks                                # noqa: E402
from vectorizer import TfidfVectorizer, HashingVectorizer, cosine_scores  # noqa: E402
from retriever import Retriever                                  # noqa: E402
from metrics import recall_at_k, precision_at_k, mrr, retrieved_doc_ids, evaluate_queries  # noqa: E402
import numpy as np                                               # noqa: E402


def _build_retriever(vec=None, chunk_size=40, overlap=8):
    docs = load_corpus()
    chunks = build_chunks(docs, chunk_size, overlap)
    return Retriever(vec or TfidfVectorizer()).index(chunks)


def test_search_returns_sorted_by_score_desc():
    """检索结果必须按分数严格非升序排列。"""
    r = _build_retriever()
    hits = r.search("recall at k trade off latency", top_k=6)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "结果未按分数降序排列"
    assert len(hits) <= 6


def test_top_k_bounds_result_count():
    """返回条数不超过 top_k，也不超过总 chunk 数。"""
    r = _build_retriever()
    assert len(r.search("caching", top_k=3)) == 3
    assert len(r.search("caching", top_k=999)) == len(r.chunks)


def test_relevant_doc_ranks_high():
    """对一个关键词明确的查询，正确文档应出现在最前面。"""
    r = _build_retriever()
    hits = r.search("semantic cache cuts latency and cost", top_k=3)
    top_docs = retrieved_doc_ids(hits)
    assert "d05" in top_docs, f"缓存文档 d05 未被召回，top={top_docs}"
    assert top_docs[0] == "d05", f"最相关文档未排第一，top={top_docs}"


def test_recall_monotonic_in_k():
    """recall@k 关于 k 单调不减（k 越大，捞回的相关文档只多不少）。"""
    r = _build_retriever()
    q = load_queries()[5]  # q6: recall / top-k 相关，多相关文档
    prev = -1.0
    for k in [1, 2, 3, 5, 8]:
        rec = recall_at_k(r.search(q.question, top_k=k), q.relevant_doc_ids, k)
        assert rec >= prev - 1e-9, f"recall@{k}={rec} 比 recall@小k 还低，非单调"
        prev = rec


def test_recall_at_k_math_is_correct():
    """用手工构造的命中列表，逐值核对 recall/precision/mrr 的定义。"""
    from retriever import RetrievedChunk
    from chunking import Chunk

    def rc(doc):
        return RetrievedChunk(chunk=Chunk(f"{doc}#0", doc, doc, "x", 0), score=1.0)

    hits = [rc("d1"), rc("d2"), rc("d3")]   # 检索到 d1,d2,d3
    relevant = ["d2", "d4"]                 # 相关的是 d2,d4
    # top-3 命中相关集合中的 d2（1 个），共 2 个相关 → recall=1/2
    assert abs(recall_at_k(hits, relevant, 3) - 0.5) < 1e-9
    # top-3 里相关 1 个 / 返回 3 个 → precision=1/3
    assert abs(precision_at_k(hits, relevant, 3) - 1 / 3) < 1e-9
    # 第一个相关(d2)排在第 2 位 → MRR=1/2
    assert abs(mrr(hits, relevant) - 0.5) < 1e-9
    # top-1 里没有相关 → recall@1=0
    assert recall_at_k(hits, relevant, 1) == 0.0


def test_empty_relevant_recall_is_one():
    """没有相关文档时约定 recall=1（无可漏），避免除零。"""
    r = _build_retriever()
    assert recall_at_k(r.search("anything", top_k=3), [], 3) == 1.0


def test_hashing_vectorizer_also_ranks_reasonably():
    """哈希向量化器（无词表）也应能把相关文档排进前列，证明接口可替换。"""
    r = _build_retriever(vec=HashingVectorizer(n_features=512))
    hits = r.search("reranking improves precision", top_k=3)
    assert "d06" in retrieved_doc_ids(hits)


def test_cosine_of_identical_vectors_is_one():
    """L2 归一化后，向量与自身的余弦为 1（数值正确性底线）。"""
    vec = TfidfVectorizer()
    mat = vec.fit_transform(["retrieval augmented generation", "caching latency"])
    s = cosine_scores(mat[0], mat[:1])
    assert abs(s[0] - 1.0) < 1e-9


def test_average_recall_improves_with_k_across_queryset():
    """整测集上，recall@5 应显著高于 recall@1（数据集级别的趋势）。"""
    r = _build_retriever()
    gold = gold_map()
    per1 = {q.qid: r.search(q.question, top_k=1) for q in load_queries()}
    per5 = {q.qid: r.search(q.question, top_k=5) for q in load_queries()}
    m1 = evaluate_queries(per1, gold, 1)
    m5 = evaluate_queries(per5, gold, 5)
    assert m5["recall@5"] > m1["recall@1"]
