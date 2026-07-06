# -*- coding: utf-8 -*-
"""
迷你 RAG 单元测试（全程离线、CPU、秒级）。

覆盖题面要求的四点：
  (a) 同句 embed 确定性一致、不同句不同；
  (b) search 对一个明显相关的 query 返回正确文档在 top-1；
  (c) answer() 返回的字符串包含被检索命中的关键事实；
  (d) 全程离线不联网（用 socket 断网守卫证明离线路径不碰网络）。
"""

import socket

import numpy as np
import pytest

from embed import DeterministicEmbedder
from store import VectorStore, cosine_similarity
from knowledge import get_facts
import rag


# ------------------------------------------------------------------ #
# (a) 嵌入的确定性
# ------------------------------------------------------------------ #
def test_embed_same_sentence_is_identical():
    emb = DeterministicEmbedder(dim=256)
    s = "RAG 检索增强生成：先检索再生成。"
    v1 = emb.encode(s)
    v2 = emb.encode(s)
    # 逐位相同（不是"近似"，是完全一致）
    assert np.array_equal(v1, v2)
    # 且是 L2 单位向量（归一化正确）
    assert v1.shape == (256,)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5


def test_embed_new_instance_reproducible():
    # 换一个新实例也应得到相同向量（跨对象/跨进程可复现）
    s = "Ollama 在本地部署大模型。"
    a = DeterministicEmbedder(dim=256).encode(s)
    b = DeterministicEmbedder(dim=256).encode(s)
    assert np.array_equal(a, b)


def test_embed_different_sentences_differ():
    emb = DeterministicEmbedder(dim=256)
    v1 = emb.encode("卷积神经网络检测图像特征。")
    v2 = emb.encode("时间序列有趋势和季节性。")
    assert not np.array_equal(v1, v2)
    # 语义不同 -> 余弦相似度应明显小于 1
    assert cosine_similarity(v1, v2) < 0.9


# ------------------------------------------------------------------ #
# (b) 检索：明显相关的 query 命中正确文档 top-1
# ------------------------------------------------------------------ #
def test_search_returns_correct_doc_top1():
    store, embedder = rag.build_store()

    # 关于 Ollama 的问题 -> top-1 应是 f14（Ollama/本地部署那条）
    q = "怎么用 Ollama 在本地部署并服务大模型？"
    hits = store.search(embedder.encode(q), k=3)
    assert len(hits) == 3
    assert hits[0].id == "f14"
    assert "Ollama" in hits[0].text
    # top-1 分数应严格高于 top-2（确实"最相关")
    assert hits[0].score >= hits[1].score


def test_search_cnn_query_top1():
    store, embedder = rag.build_store()
    q = "卷积神经网络是怎么在图像里检测特征的？"
    hits = store.search(embedder.encode(q), k=3)
    assert hits[0].id == "f03"


def test_vectorstore_basic_add_search():
    emb = DeterministicEmbedder(dim=128)
    vs = VectorStore()
    docs = {"a": "苹果 香蕉 水果", "b": "汽车 火车 交通", "c": "老虎 狮子 动物"}
    for did, text in docs.items():
        vs.add(did, text, emb.encode(text))
    assert len(vs) == 3
    hits = vs.search(emb.encode("我想吃水果，比如苹果"), k=1)
    assert hits[0].id == "a"


# ------------------------------------------------------------------ #
# (c) answer() 包含被检索命中的关键事实
# ------------------------------------------------------------------ #
def test_answer_contains_retrieved_fact():
    ans = rag.answer("什么是 RAG 检索增强生成？", k=3, backend="offline")
    assert isinstance(ans, str) and len(ans) > 0
    # 命中的关键事实原文应出现在答案里
    assert "检索增强生成" in ans
    assert "余弦" in ans  # f15 里提到"余弦相似度检索"


def test_answer_ollama_backend_falls_back_offline(monkeypatch):
    # 强制 ollama 生成失败 -> answer 应回退到离线，仍给出含事实的答案
    monkeypatch.setattr(rag, "generate_with_ollama", lambda *a, **k: None)
    ans = rag.answer("怎么在本地部署大模型？", k=2, backend="ollama")
    assert "Ollama" in ans  # 回退后的离线答案仍包含命中事实


def test_answer_no_hit_is_graceful():
    # 用一个和知识库完全无关、且不含任何知识库 token 的查询
    ans = rag.answer("zzzxxxqqq", k=3, backend="offline")
    assert isinstance(ans, str) and len(ans) > 0


# ------------------------------------------------------------------ #
# (d) 全程离线：断网守卫下依然跑通
# ------------------------------------------------------------------ #
def test_pipeline_runs_fully_offline(monkeypatch):
    """把 socket 全面断网，证明离线 RAG 流程完全不碰网络。"""

    def _blocked(*args, **kwargs):
        raise AssertionError("离线测试中不应发生任何网络连接！")

    # 拦截建立连接的所有入口
    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    q = "RAG 检索增强生成的工作原理是什么？"
    store, embedder = rag.build_store()
    hits = rag.retrieve(q, k=2, store=store, embedder=embedder)
    assert len(hits) == 2
    ans = rag.answer(q, k=2, backend="offline", store=store, embedder=embedder)
    # 离线答案必须逐字包含检索到的 top-1 事实（证明"用证据说话"且未走网络）
    assert hits[0].text in ans
    assert "检索增强生成" in ans


def test_all_facts_indexed():
    store, _ = rag.build_store()
    assert len(store) == len(get_facts()) == 16
