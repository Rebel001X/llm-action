# -*- coding: utf-8 -*-
"""
embed.py —— 句向量编码器

对应书里第 5、6、18 章的核心思想：
  - 第 5 章：把语言编码成数字（tokenize）。
  - 第 6 章：用 embedding 把 token 映射成稠密向量，让"语义"变成可计算的东西。
  - 第 18 章：RAG 的检索靠"语义相似"，而语义相似必须先有向量。

本文件提供两条路径：
  1) DeterministicEmbedder —— **完全离线、确定性**的句向量。
     手法：hashing bag-of-words（把 token 用 hashlib 稳定哈希到固定维度的桶里，
     再 L2 归一化）。同一句话永远得到同一个向量，不联网、不下载任何模型。
     之所以用 hashlib 而不是 Python 内置 hash()：内置 hash() 对字符串默认带随机盐
     （PYTHONHASHSEED），跨进程结果会变；hashlib.md5 保证逐字节可复现。
  2) SentenceTransformerEmbedder —— 可选的"真实"句向量（try import，缺失/离线自动放弃）。
     如果你装了 sentence-transformers，就能一行切换到工业级嵌入。

对外统一接口：.dim（维度）、.encode(text) -> np.ndarray(shape=(dim,))、
             .encode_batch(list[str]) -> np.ndarray(shape=(n, dim))
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np


# ------------------------------------------------------------------ #
# 分词：混合中英文的极简 tokenizer
# ------------------------------------------------------------------ #
def _tokenize(text: str) -> List[str]:
    """把一段文本切成 token 列表（无外部依赖）。

    策略：
      - 英文/数字：按连续的字母数字串切成"词"（word 级特征）。
      - 中文：没有分词器，所以退化为"字 + 相邻字二元组（bigram）"，
        这样"检索增强"这类词组也能通过 bigram 命中，overlap 越多相似度越高。
    """
    text = text.lower()
    tokens: List[str] = []

    # 英文 / 数字词
    tokens.extend(re.findall(r"[a-z0-9]+", text))

    # 中文：按连续汉字串取 unigram + bigram
    for run in re.findall(r"[一-鿿]+", text):
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i:i + 2])

    return tokens


def _stable_bucket(token: str, dim: int) -> int:
    """用 md5 把 token 稳定映射到 [0, dim) 的桶号（跨进程可复现）。"""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


class DeterministicEmbedder:
    """离线、确定性的 hashing bag-of-words 句向量编码器。

    参数
    ----
    dim : int
        向量维度（哈希桶数量）。越大冲突越少，默认 256 对小知识库足够。

    性质
    ----
    - 确定性：encode("同一句话") 每次、每进程都返回**逐位相同**的向量。
    - 归一化：输出是 L2 单位向量，于是"点积 == 余弦相似度"，方便检索。
    """

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            idx = _stable_bucket(tok, self.dim)
            vec[idx] += 1.0                      # 词袋计数
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm                          # L2 归一化 -> 单位向量
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts], axis=0)


class SentenceTransformerEmbedder:
    """可选：用 sentence-transformers 得到工业级句向量（离线不可用时请勿实例化）。

    仅在你**主动**安装了 sentence-transformers、且本地已有模型缓存时使用。
    默认的 RAG 流程不会用到它，以保证 pytest 全程离线、秒级完成。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # 延迟导入
        except Exception as e:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "未安装 sentence-transformers，请改用 DeterministicEmbedder，"
                "或先 `pip install sentence-transformers`。"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - 依赖外部模型
        v = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        vs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vs, dtype=np.float32)


def get_embedder(prefer_real: bool = False, dim: int = 256):
    """工厂：默认返回离线确定性编码器；prefer_real=True 时尝试真实嵌入，失败自动回退。"""
    if prefer_real:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass  # 缺库/离线 -> 回退到离线确定性编码器
    return DeterministicEmbedder(dim=dim)
