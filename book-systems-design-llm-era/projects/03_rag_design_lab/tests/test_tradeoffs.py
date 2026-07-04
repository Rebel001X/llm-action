# -*- coding: utf-8 -*-
"""chunk 大小 / top-k 权衡 & 重排效果测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import load_corpus, load_queries, gold_map    # noqa: E402
from chunking import build_chunks, chunk_document          # noqa: E402
from vectorizer import TfidfVectorizer                      # noqa: E402
from retriever import Retriever, rerank                     # noqa: E402
from metrics import evaluate_queries, retrieved_doc_ids     # noqa: E402
from pipeline import RAGPipeline, CostModel                 # noqa: E402


def _eval_at(chunk_size, overlap, k):
    docs = load_corpus()
    chunks = build_chunks(docs, chunk_size, overlap)
    r = Retriever(TfidfVectorizer()).index(chunks)
    gold = gold_map()
    per = {q.qid: r.search(q.question, top_k=k) for q in load_queries()}
    return evaluate_queries(per, gold, k), chunks


def test_larger_top_k_raises_recall_lowers_precision():
    """经典权衡：k↑ → 数据集平均 recall 不降、precision 不升。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    gold = gold_map()

    def avg(k):
        per = {q.qid: r.search(q.question, top_k=k) for q in load_queries()}
        m = evaluate_queries(per, gold, k)
        return m[f"recall@{k}"], m[f"precision@{k}"]

    r1, p1 = avg(1)
    r5, p5 = avg(5)
    assert r5 >= r1, "top-k 增大后 recall 反而下降"
    assert p5 <= p1, "top-k 增大后 precision 反而上升"
    assert r5 > r1, "top-k 从 1→5 应带来 recall 的真实提升"


def test_chunk_size_changes_chunk_count():
    """chunk 越小 → chunk 数越多（更细粒度）。"""
    docs = load_corpus()
    small = build_chunks(docs, 10, 0)
    large = build_chunks(docs, 100, 0)
    assert len(small) > len(large)


def test_no_overlap_covers_all_words():
    """无重叠切块：所有 chunk 词数之和应等于原文词数（不丢词不重复）。"""
    from vectorizer import tokenize
    docs = load_corpus()
    d = docs[2]
    chunks = chunk_document(d, chunk_size=7, overlap=0)
    total_words = sum(len(tokenize(c.text)) for c in chunks)
    assert total_words == len(tokenize(d.text))


def test_overlap_does_not_infinite_loop():
    """overlap 接近/等于 chunk_size 不会死循环（step 兜底为 >=1）。"""
    docs = load_corpus()
    chunks = chunk_document(docs[0], chunk_size=5, overlap=5)
    assert len(chunks) > 0
    assert len(chunks) < 10_000              # 若死循环这里会爆炸


def test_rerank_improves_or_keeps_top1_precision():
    """重排后，top-1 的相关命中率不应低于纯一阶段（重排目的就是把最贴题的顶上来）。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    gold = gold_map()

    def top1_hit_rate(use_rerank):
        hits_at1 = 0
        for q in load_queries():
            hits = r.search(q.question, top_k=8)
            if use_rerank:
                hits = rerank(q.question, hits, top_n=3)
            top1 = retrieved_doc_ids(hits)[:1]
            if top1 and top1[0] in gold[q.qid]:
                hits_at1 += 1
        return hits_at1 / len(load_queries())

    base = top1_hit_rate(False)
    reranked = top1_hit_rate(True)
    assert reranked >= base, f"重排后 top-1 命中率下降：{reranked} < {base}"


def test_cost_grows_with_context_size():
    """成本模型：送生成的上下文越多（rerank_top_n 越大），prompt token 与成本越高。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    q = load_queries()[0].question

    small = RAGPipeline(r, top_k=8, rerank_top_n=1).run(q)
    large = RAGPipeline(r, top_k=8, rerank_top_n=5).run(q)
    assert large.prompt_tokens >= small.prompt_tokens
    assert large.cost_usd >= small.cost_usd
    assert large.latency_ms >= small.latency_ms


def test_rerank_adds_latency():
    """开启重排会增加延迟（多了一个打分阶段）——权衡的另一面。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    q = load_queries()[0].question
    no_rr = RAGPipeline(r, top_k=8, use_rerank=False, rerank_top_n=3).run(q)
    with_rr = RAGPipeline(r, top_k=8, use_rerank=True, rerank_top_n=3).run(q)
    assert with_rr.latency_ms > no_rr.latency_ms
