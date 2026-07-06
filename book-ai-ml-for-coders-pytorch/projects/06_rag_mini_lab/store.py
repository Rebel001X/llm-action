# -*- coding: utf-8 -*-
"""
store.py —— 极简向量库（VectorStore）

对应第 18 章"搭建向量库 + 相似度检索"，只是把书里用的 ChromaDB 换成一个
几十行、零依赖、易读的 numpy 实现，方便你看清 RAG 检索的内核：
    余弦相似度 top-k。

书里强调的两条铁律，这里都遵守：
  1) 入库文本和查询 prompt 必须用**同一套嵌入**，向量空间才可比。
     （本库不关心用哪个嵌入器，只负责存/搜；调用方负责保证一致。）
  2) 相似度默认用**余弦**：只看方向、不看长度，适合比"意思像不像"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SearchHit:
    """一条检索命中结果。"""
    id: str
    text: str
    score: float  # 余弦相似度，∈ [-1, 1]，越大越相似


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """两个向量的余弦相似度 = a·b / (|a||b|)。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


class VectorStore:
    """把 (id, text, vec) 存起来，并支持余弦 top-k 检索。"""

    def __init__(self):
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._vecs: List[np.ndarray] = []
        self._matrix: Optional[np.ndarray] = None  # 缓存的 (N, dim) 矩阵

    # ------------------------------------------------------------------ #
    def add(self, id: str, text: str, vec: np.ndarray) -> None:
        """新增一条记录。vec 应为一维 np.ndarray。"""
        v = np.asarray(vec, dtype=np.float32).ravel()
        self._ids.append(str(id))
        self._texts.append(str(text))
        self._vecs.append(v)
        self._matrix = None  # 失效缓存，下次 search 重建

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------ #
    def _ensure_matrix(self) -> np.ndarray:
        """把所有向量堆成 (N, dim) 矩阵并缓存，便于一次性向量化比对。"""
        if self._matrix is None:
            if self._vecs:
                self._matrix = np.stack(self._vecs, axis=0)
            else:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
        return self._matrix

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[SearchHit]:
        """返回与 query_vec 余弦相似度最高的 top-k 条命中（降序）。"""
        if len(self) == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).ravel()
        mat = self._ensure_matrix()  # (N, dim)

        # 向量化余弦：先各自 L2 归一化，再做矩阵乘 -> 每行一个余弦分数
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-12:
            return []
        q_unit = q / q_norm

        row_norms = np.linalg.norm(mat, axis=1)          # (N,)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        mat_unit = mat / row_norms[:, None]              # (N, dim)

        scores = mat_unit @ q_unit                       # (N,)  == 余弦相似度

        k = max(1, min(int(k), len(self)))
        # argpartition 取 top-k，再对这 k 个精确排序（降序）
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchHit(id=self._ids[i], text=self._texts[i], score=float(scores[i]))
            for i in top_idx
        ]
