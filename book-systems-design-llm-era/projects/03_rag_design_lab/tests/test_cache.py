# -*- coding: utf-8 -*-
"""缓存命中降延迟 & 命中统计正确性测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                       # noqa: E402
from corpus import load_corpus, load_queries             # noqa: E402
from chunking import build_chunks                         # noqa: E402
from vectorizer import TfidfVectorizer                    # noqa: E402
from retriever import Retriever                           # noqa: E402
from cache import SemanticCache                           # noqa: E402
from pipeline import RAGPipeline                          # noqa: E402


def _pipe(use_cache=True, threshold=0.9):
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    cache = SemanticCache(threshold=threshold)
    return RAGPipeline(r, cache=cache, top_k=5, use_cache=use_cache, rerank_top_n=3), cache


def test_exact_cache_hit_lowers_latency():
    """同一 query 第二次必须命中缓存，且延迟远低于首次。"""
    pipe, cache = _pipe()
    q = load_queries()[0].question
    first = pipe.run(q)
    second = pipe.run(q)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.latency_ms < first.latency_ms
    assert second.cost_usd == 0.0            # 命中缓存不再产生生成成本
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_cache_returns_same_answer():
    """命中缓存返回的答案文本应与首次完全一致（确定性）。"""
    pipe, _ = _pipe()
    q = load_queries()[1].question
    a1 = pipe.run(q).answer.text
    a2 = pipe.run(q).answer.text
    assert a1 == a2


def test_no_cache_never_hits():
    """关闭缓存时，重复查询永不命中，两次延迟都是完整管线延迟。"""
    pipe, cache = _pipe(use_cache=False)
    q = load_queries()[0].question
    t1 = pipe.run(q)
    t2 = pipe.run(q)
    assert t1.cache_hit is False and t2.cache_hit is False
    assert abs(t1.latency_ms - t2.latency_ms) < 1e-6
    assert cache.stats.total == 0            # 缓存根本没被调用


def test_semantic_hit_on_paraphrase():
    """语义缓存：对'几乎同义'的改写查询应命中（阈值放宽时）。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    cache = SemanticCache(threshold=0.5)     # 放宽阈值以触发语义命中
    pipe = RAGPipeline(r, cache=cache, top_k=5, use_cache=True)
    pipe.run("how does caching cut latency and cost")
    # 换个措辞但语义高度重叠
    t = pipe.run("how does caching reduce latency and cost")
    assert t.cache_hit is True


def test_high_threshold_avoids_false_hit():
    """阈值设 1.0 时，不同措辞不应误命中（严格模式保正确性）。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    cache = SemanticCache(threshold=1.0)
    pipe = RAGPipeline(r, cache=cache, top_k=5, use_cache=True)
    pipe.run("how does caching cut latency")
    t = pipe.run("what is a reranker")       # 完全不同的问题
    assert t.cache_hit is False


def test_cache_hit_rate_accounting():
    """命中率统计：1 次未命中 + 2 次命中 → hit_rate=2/3。"""
    pipe, cache = _pipe()
    q = load_queries()[0].question
    pipe.run(q)     # miss
    pipe.run(q)     # hit
    pipe.run(q)     # hit
    assert cache.stats.hits == 2 and cache.stats.misses == 1
    assert abs(cache.stats.hit_rate - 2 / 3) < 1e-9


def test_cache_fifo_eviction():
    """超容量按 FIFO 淘汰：容量 2，写入 3 条后最早那条被淘汰。"""
    cache = SemanticCache(threshold=1.0, max_size=2)
    v = np.array([1.0, 0.0])
    cache.put("a", "A", v)
    cache.put("b", "B", v)
    cache.put("c", "C", v)                    # 触发淘汰 "a"
    assert len(cache) == 2
    assert cache.get("a") is None             # a 已被淘汰
    assert cache.get("c") == "C"
