# -*- coding: utf-8 -*-
"""
vectorizer.py —— 把文本变成向量（纯 numpy，零外部模型、零下载）。

我们提供两种"稀疏词向量化器"，都不需要任何预训练模型，因此**完全离线**：

  1) TfidfVectorizer   —— 经典 TF-IDF：词频 × 逆文档频率，词表随语料动态构建。
  2) HashingVectorizer —— 特征哈希（feature hashing / hashing trick）：把词直接哈希到
                          固定维度桶里，**无需存词表**，天然支持流式/超大词表，代价是哈希冲突。

🔬 第一性原理：检索的本质是"给 query 和每篇 doc 各算一个向量，再比相似度排序"。
    向量化器决定了"语义用什么坐标表示"。TF-IDF 用"加权词频"当坐标，
    虽不懂同义词，但**可解释、可离线、够快**，是理解 RAG 检索管线的最佳起点。
    真实系统把这里换成 dense embedding（如 bge / e5）即可，**下游接口完全不变**。
"""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

# 简单英文分词：小写化 + 抽取字母数字词。语料是英文，故不引入中文分词依赖。
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """把一段文本切成小写词列表。"""
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """
    TF-IDF 向量化器（fit 学词表与 idf，transform 出向量）。

    公式（本项目采用的定义，和 sklearn 的平滑版一致的思路）：
        tf(t, d)  = 词 t 在文档 d 中出现次数
        idf(t)    = ln( (1 + N) / (1 + df(t)) ) + 1     # +1 平滑，避免除零/负无穷
        weight    = tf * idf
    最后对每个文档向量做 L2 归一化，这样"余弦相似度"就退化成"点积"，更快。
    """

    def __init__(self) -> None:
        self.vocabulary_: Dict[str, int] = {}   # 词 -> 列索引
        self.idf_: np.ndarray = np.zeros(0)     # 每个词的 idf 权重，形状 (V,)
        self._fitted = False

    def fit(self, docs: Sequence[str]) -> "TfidfVectorizer":
        """从语料学习词表与 idf。"""
        n_docs = len(docs)
        df: Counter = Counter()          # document frequency：词出现在多少篇文档里
        tokenized = [tokenize(d) for d in docs]
        for toks in tokenized:
            for t in set(toks):          # 用 set：同一文档里一个词只计一次 df
                df[t] += 1
        # 词表按词典序固定顺序，保证可复现（dict 插入序 + 排序）
        vocab = sorted(df.keys())
        self.vocabulary_ = {t: i for i, t in enumerate(vocab)}
        # 计算 idf：稀有词权重高，"the/is"这种到处都有的词权重低
        idf = np.zeros(len(vocab), dtype=np.float64)
        for t, i in self.vocabulary_.items():
            idf[i] = math.log((1 + n_docs) / (1 + df[t])) + 1.0
        self.idf_ = idf
        self._fitted = True
        return self

    def transform(self, docs: Sequence[str]) -> np.ndarray:
        """把文本转成 L2 归一化后的 TF-IDF 矩阵，形状 (n_docs, V)。"""
        assert self._fitted, "必须先 fit 再 transform"
        V = len(self.vocabulary_)
        mat = np.zeros((len(docs), V), dtype=np.float64)
        for row, d in enumerate(docs):
            counts = Counter(tokenize(d))
            for t, c in counts.items():
                j = self.vocabulary_.get(t)
                if j is not None:                 # 训练时没见过的词直接忽略（OOV）
                    mat[row, j] = c * self.idf_[j]
        return _l2_normalize(mat)

    def fit_transform(self, docs: Sequence[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


class HashingVectorizer:
    """
    特征哈希向量化器：把词哈希到 n_features 个桶，不存词表。

    优点：内存恒定、天然增量（新词不改变维度）、无需先见全语料。
    代价：哈希冲突（两个不同词落到同一桶）会引入噪声——这是"无状态"的代价。
    带符号哈希（signed hashing）：用第二个哈希决定 +1/-1，减少冲突带来的系统性偏差。
    """

    def __init__(self, n_features: int = 256, use_idf_like: bool = False) -> None:
        self.n_features = int(n_features)
        # 这里不学 idf，纯 tf 版即可演示"哈希向量化"的思想；保留参数位以便扩展。
        self.use_idf_like = use_idf_like

    @staticmethod
    def _hash(token: str) -> int:
        """确定性哈希（不用内置 hash，因为它带随机盐、跨进程不稳定）。"""
        h = 2166136261
        for ch in token:                          # FNV-1a 变体，纯 Python 稳定可复现
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def transform(self, docs: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(docs), self.n_features), dtype=np.float64)
        for row, d in enumerate(docs):
            for t in tokenize(d):
                h = self._hash(t)
                col = h % self.n_features             # 主哈希决定落哪个桶
                sign = 1.0 if (h >> 16) & 1 else -1.0  # 副哈希决定符号，抵消部分冲突
                mat[row, col] += sign
        return _l2_normalize(mat)

    def fit(self, docs: Sequence[str]) -> "HashingVectorizer":
        return self  # 无状态，fit 什么都不做，接口对齐 TF-IDF 便于替换

    def fit_transform(self, docs: Sequence[str]) -> np.ndarray:
        return self.transform(docs)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化：||v|| = 1。零向量保持为零（避免除零）。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_scores(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    余弦相似度打分：因为向量已 L2 归一化，余弦 == 点积。
    query_vec : (V,) 或 (1, V)
    doc_matrix: (N, V)
    返回      : (N,) 每篇文档对该 query 的相似度分数。
    """
    q = query_vec.reshape(-1)
    return doc_matrix @ q
