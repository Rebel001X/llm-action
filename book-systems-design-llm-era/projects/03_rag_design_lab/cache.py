# -*- coding: utf-8 -*-
"""
cache.py —— 语义缓存（semantic cache）：命中则跳过检索+生成，直接返回旧答案。

《Systems Design in the LLM Era》第 2 章"低延迟"维度的核心武器之一。
    LLM 生成慢（5–10s）又贵（按 token 计费）。如果两个用户问的问题"语义几乎一样"，
    第二次就没必要再跑一遍完整管线——把第一次的答案缓存起来直接返回。

两种命中判定：
  1) 精确缓存（exact）：归一化文本完全相同才命中。简单、零误命中，但复用率低。
  2) 语义缓存（semantic）：query 向量与缓存 query 向量的余弦 >= 阈值即命中。
     复用率高，但阈值太低会"答非所问"（误命中）——这是复用率与正确性的权衡。

本模块同时记录命中/未命中统计，供 demo 量化"缓存把平均延迟/成本降了多少"。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _normalize_text(q: str) -> str:
    """精确缓存的 key：小写 + 压缩空白。避免仅因大小写/空格差异而 miss。"""
    return " ".join(q.lower().split())


@dataclass
class CacheStats:
    """缓存统计：命中率是评估缓存价值的核心。"""
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class _Entry:
    """一条缓存记录：query 文本、可选 query 向量、以及缓存的答案对象。"""
    query: str
    vector: Optional[np.ndarray]
    value: Any


class SemanticCache:
    """
    支持"精确 + 语义"两种命中的缓存。

    参数：
      threshold : 语义命中的余弦阈值（[0,1]，越高越严格、越不易误命中）。
      max_size  : 最多缓存多少条（超出按 FIFO 淘汰，演示"缓存有容量上限"这一现实约束）。
    """

    def __init__(self, threshold: float = 0.92, max_size: int = 1024) -> None:
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold
        self.max_size = max_size
        self._entries: List[_Entry] = []
        self.stats = CacheStats()
        self._exact_index: Dict[str, int] = {}   # 归一化文本 -> _entries 下标，O(1) 精确查

    def get(self, query: str, query_vec: Optional[np.ndarray] = None) -> Optional[Any]:
        """
        查缓存。先试精确命中（快、零误判），再试语义命中（需提供 query_vec）。
        命中返回缓存值并计一次 hit；未命中返回 None 并计一次 miss。
        """
        norm = _normalize_text(query)
        # 1) 精确命中
        idx = self._exact_index.get(norm)
        if idx is not None:
            self.stats.hits += 1
            return self._entries[idx].value
        # 2) 语义命中（需要向量）
        if query_vec is not None and self._entries:
            q = query_vec.reshape(-1)
            best_sim, best_i = -1.0, -1
            for i, e in enumerate(self._entries):
                if e.vector is None:
                    continue
                sim = float(np.dot(q, e.vector))  # 双方已 L2 归一化，点积即余弦
                if sim > best_sim:
                    best_sim, best_i = sim, i
            if best_i >= 0 and best_sim >= self.threshold:
                self.stats.hits += 1
                return self._entries[best_i].value
        # 3) 未命中
        self.stats.misses += 1
        return None

    def put(self, query: str, value: Any, query_vec: Optional[np.ndarray] = None) -> None:
        """写入缓存。超容量按 FIFO 淘汰最老的一条。"""
        norm = _normalize_text(query)
        if norm in self._exact_index:            # 已存在则更新值即可
            self._entries[self._exact_index[norm]].value = value
            return
        if len(self._entries) >= self.max_size:
            old = self._entries.pop(0)           # FIFO：踢掉最早进来的
            self._exact_index.pop(_normalize_text(old.query), None)
            # 下标整体前移，重建精确索引（小缓存代价可忽略；大缓存应换环形队列）
            self._exact_index = {_normalize_text(e.query): i
                                 for i, e in enumerate(self._entries)}
        vec = None if query_vec is None else query_vec.reshape(-1).copy()
        self._entries.append(_Entry(query=query, vector=vec, value=value))
        self._exact_index[norm] = len(self._entries) - 1

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)
