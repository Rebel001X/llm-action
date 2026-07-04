# -*- coding: utf-8 -*-
"""
metrics.py —— 检索质量指标：recall@k / precision@k / MRR。

为什么这些指标？
    RAG 的第一性问题是"生成之前，检索有没有把正确证据捞上来"。
    · recall@k    ：top-k 里覆盖了多少比例的相关文档 —— RAG 最关键指标（漏了就没法答对）。
    · precision@k ：top-k 里有多少比例是相关的 —— 反映"塞进 prompt 的噪声"程度（影响成本/干扰）。
    · MRR         ：第一个相关结果的排名倒数 —— 反映"最相关的有没有排在前面"。

注意粒度：chunk 检索结果要**先按 doc_id 归并**再和 gold（文档级标注）比，
    因为一篇文档可能被切成多个 chunk，命中任一 chunk 即视为"命中该文档"。
"""

from typing import Dict, List, Sequence

from retriever import RetrievedChunk


def retrieved_doc_ids(hits: Sequence[RetrievedChunk]) -> List[str]:
    """把 chunk 级检索结果按出现顺序去重成 doc_id 列表（保持排名）。"""
    seen = set()
    out: List[str] = []
    for h in hits:
        d = h.chunk.doc_id
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(hits: Sequence[RetrievedChunk], relevant: Sequence[str], k: int) -> float:
    """
    recall@k = |{相关文档} ∩ {top-k 命中的文档}| / |{相关文档}|
    k 作用在"去重后的 doc 排名"上。relevant 为空时约定 recall=1.0（无可漏）。
    """
    rel = set(relevant)
    if not rel:
        return 1.0
    docs = retrieved_doc_ids(hits)[:k]
    hit = len(rel & set(docs))
    return hit / len(rel)


def precision_at_k(hits: Sequence[RetrievedChunk], relevant: Sequence[str], k: int) -> float:
    """precision@k = top-k 里相关文档数 / k（k 以实际返回数为上限）。"""
    rel = set(relevant)
    docs = retrieved_doc_ids(hits)[:k]
    if not docs:
        return 0.0
    hit = len(rel & set(docs))
    return hit / len(docs)


def mrr(hits: Sequence[RetrievedChunk], relevant: Sequence[str]) -> float:
    """Mean Reciprocal Rank（单条查询版）：第一个相关文档排名的倒数，没命中则 0。"""
    rel = set(relevant)
    for rank, d in enumerate(retrieved_doc_ids(hits), start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def evaluate_queries(per_query_hits: Dict[str, List[RetrievedChunk]],
                     gold: Dict[str, List[str]], k: int) -> Dict[str, float]:
    """
    在一批查询上求平均指标。
    per_query_hits: qid -> 该查询的检索结果
    gold          : qid -> 相关文档 id 列表
    返回          : {"recall@k":..., "precision@k":..., "mrr":...}（对所有 query 取均值）
    """
    if not per_query_hits:
        return {f"recall@{k}": 0.0, f"precision@{k}": 0.0, "mrr": 0.0}
    rs, ps, ms = [], [], []
    for qid, hits in per_query_hits.items():
        rel = gold.get(qid, [])
        rs.append(recall_at_k(hits, rel, k))
        ps.append(precision_at_k(hits, rel, k))
        ms.append(mrr(hits, rel))
    n = len(per_query_hits)
    return {
        f"recall@{k}": sum(rs) / n,
        f"precision@{k}": sum(ps) / n,
        "mrr": sum(ms) / n,
    }
