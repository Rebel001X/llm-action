# -*- coding: utf-8 -*-
"""
retriever.py —— 检索器：索引 chunk、按 query 打分排序、返回 top-k，含可选重排。

管线：query → 向量化 → 与所有 chunk 向量算余弦 → 排序 → 取 top-k →（可选）重排。

设计要点：
  · 索引一次（fit + transform），查询多次 —— 摊薄向量化成本。
  · 检索结果保留 chunk_id / doc_id / score，供 recall 评测与"引用映射"使用。
  · 重排（rerank）是"两阶段检索"的第二阶段：先用便宜的一阶段召回一批候选，
    再用更贵但更准的打分器（这里演示一个轻量词重叠打分 + 可选 torch 实现）精排 top-n。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from chunking import Chunk
from vectorizer import TfidfVectorizer, cosine_scores, tokenize


@dataclass
class RetrievedChunk:
    """一条检索结果：命中的 chunk + 一阶段分数 +（若重排过）重排分数。"""
    chunk: Chunk
    score: float                       # 一阶段（向量余弦）分数
    rerank_score: Optional[float] = None  # 二阶段重排分数（未重排则为 None）


class Retriever:
    """
    一个最小可用的稠密/稀疏检索器。

    用法：
        r = Retriever(vectorizer=TfidfVectorizer())
        r.index(chunks)
        hits = r.search("my query", top_k=5)
    """

    def __init__(self, vectorizer=None) -> None:
        # 默认用 TF-IDF；可传入 HashingVectorizer 或未来的 dense embedding，接口一致。
        self.vectorizer = vectorizer if vectorizer is not None else TfidfVectorizer()
        self.chunks: List[Chunk] = []
        self.doc_matrix: np.ndarray = np.zeros((0, 0))
        self._indexed = False

    def index(self, chunks: Sequence[Chunk]) -> "Retriever":
        """建立索引：对所有 chunk 文本 fit + transform，得到 (N, V) 向量矩阵。"""
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        # fit_transform：TF-IDF 会在这里学词表；Hashing 是无状态的，直接 transform。
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self._indexed = True
        return self

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """一阶段检索：把 query 向量化，与所有 chunk 算余弦，返回分数最高的 top_k。"""
        assert self._indexed, "必须先 index 再 search"
        if not self.chunks:
            return []
        # 关键：query 必须用"已 fit 的同一个向量化器" transform，坐标系才对齐。
        q_vec = self.vectorizer.transform([query])[0]
        scores = cosine_scores(q_vec, self.doc_matrix)   # (N,)
        k = min(top_k, len(self.chunks))
        # argpartition 取 top-k 更快（O(N)），再对这 k 个精确排序（O(k log k)）。
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]  # 组内按分数降序
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
                for i in top_idx]


# ---------------------------------------------------------------------------
# 可选重排器（第二阶段）
# ---------------------------------------------------------------------------
def _lexical_overlap_score(query: str, passage: str) -> float:
    """
    轻量重排打分：query 与 passage 的词集合 Jaccard-ish 重叠度。
    直觉：一阶段 TF-IDF 已经召回了"大致相关"的候选，
    重排再看一眼"关键词是否真的都覆盖到"，把最贴题的顶上去。
    这是一个便宜但有效的重排信号（真实系统会换成 cross-encoder）。
    """
    q = set(tokenize(query))
    p = set(tokenize(passage))
    if not q or not p:
        return 0.0
    inter = len(q & p)
    return inter / (len(q) ** 0.5 * len(p) ** 0.5)   # 用几何平均归一，避免长段占便宜


def _try_torch_score(query: str, passage: str) -> Optional[float]:
    """
    可选：若本机装了 CPU 版 torch，用它算同一个重叠分数（演示"重排器可换实现"）。
    结果与 numpy 版数值一致——重点是展示"打分器是可插拔组件"，而非依赖 torch。
    缺 torch 时返回 None，调用方自动回退到 numpy 版。
    """
    try:
        import torch  # 局部导入：没装也不影响其它功能
    except Exception:
        return None
    q = set(tokenize(query))
    p = set(tokenize(passage))
    if not q or not p:
        return 0.0
    inter = float(len(q & p))
    denom = torch.sqrt(torch.tensor(float(len(q)))) * torch.sqrt(torch.tensor(float(len(p))))
    return float(torch.tensor(inter) / denom)


def rerank(query: str, hits: List[RetrievedChunk], top_n: int,
           use_torch: bool = False) -> List[RetrievedChunk]:
    """
    对一阶段召回的 hits 做重排，返回重排后的前 top_n。

    参数：
      top_n     : 重排后保留几条（通常 <= 一阶段 top_k）
      use_torch : True 且本机有 torch 时用 torch 打分，否则 numpy。功能等价。

    ⚠️ 坑：重排必须在"一阶段召回的候选集"内做。若一阶段没召回到正确 chunk，
        再强的重排也救不回来——所以一阶段 top_k 不能设太小（recall 上限由它决定）。
    """
    scored: List[RetrievedChunk] = []
    for h in hits:
        s = _try_torch_score(query, h.chunk.text) if use_torch else None
        if s is None:
            s = _lexical_overlap_score(query, h.chunk.text)
        scored.append(RetrievedChunk(chunk=h.chunk, score=h.score, rerank_score=s))
    # 按重排分数降序；重排分相同则用一阶段分数兜底（稳定、可复现）
    scored.sort(key=lambda x: (x.rerank_score, x.score), reverse=True)
    return scored[:top_n]
