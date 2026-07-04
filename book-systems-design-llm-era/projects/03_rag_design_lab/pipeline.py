# -*- coding: utf-8 -*-
"""
pipeline.py —— 端到端 RAG 管线 + 延迟/成本模型（把所有组件串起来）。

它把 chunking → retriever →（可选 rerank）→ cache → MockLLM generator 串成一条链，
并附一个**解析式的延迟/成本模型**，让我们能在不真正调 LLM 的情况下，
量化"chunk 大小 / top-k / 缓存命中"如何影响 端到端延迟 和 每次查询成本。

🔬 为什么用"模型"而不是真计时？
    真实 LLM 延迟受网络/负载抖动，且本机离线无法调用。工程实践中做容量规划时，
    普遍用**参数化解析模型**先估量级、画权衡曲线，再用真实压测校准。
    这里的系数（每 token 毫秒、每 token 美元）是可配置的、量级贴近真实的占位值。
"""

from dataclasses import dataclass, field
from typing import List, Optional

from cache import SemanticCache
from chunking import Chunk
from generator import Answer, MockLLM
from retriever import Retriever, RetrievedChunk, rerank
from vectorizer import tokenize


# ---------------------------------------------------------------------------
# 延迟/成本系数（可调）。数量级对齐"CPU 检索便宜、LLM 生成贵"的现实。
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    retrieval_ms_per_chunk: float = 0.02     # 每比对一个 chunk 的检索耗时（ms）
    rerank_ms_per_candidate: float = 0.5     # 每重排一个候选的耗时（ms，比检索贵）
    gen_ms_base: float = 200.0               # 生成固定开销（ms，模型加载/首 token 前）
    gen_ms_per_token: float = 4.0            # 每个输入 token 的生成边际耗时（ms）
    usd_per_1k_tokens: float = 0.002         # 输入 token 计费（美元/千 token）
    cache_hit_ms: float = 1.0                # 命中缓存的固定极低延迟（ms）

    def prompt_tokens(self, context_chunks: List[Chunk]) -> int:
        """把上下文 chunk 的词数近似当作 prompt token 数（1 词 ≈ 1 token 的粗略近似）。"""
        return sum(len(tokenize(c.text)) for c in context_chunks)


@dataclass
class QueryTrace:
    """一次查询的可观测记录：结果 + 各阶段延迟/成本 + 是否命中缓存。"""
    query: str
    answer: Answer
    hits: List[RetrievedChunk]
    cache_hit: bool
    latency_ms: float
    cost_usd: float
    prompt_tokens: int


class RAGPipeline:
    """
    端到端可配置 RAG 管线。

    可调旋钮（对应实验维度）：
      top_k        : 一阶段召回多少 chunk（↑ 提 recall，但 ↑ 延迟/成本）
      use_rerank   : 是否开启二阶段重排（↑ 精度，↑ 一点延迟）
      rerank_top_n : 重排后保留几条送生成（= 生成实际用的上下文条数上限）
      use_cache    : 是否启用语义缓存（命中则几乎零延迟零成本）
    """

    def __init__(self, retriever: Retriever, generator: Optional[MockLLM] = None,
                 cache: Optional[SemanticCache] = None, cost_model: Optional[CostModel] = None,
                 top_k: int = 5, use_rerank: bool = False, rerank_top_n: int = 3,
                 use_cache: bool = False, use_torch_rerank: bool = False) -> None:
        self.retriever = retriever
        self.generator = generator if generator is not None else MockLLM(max_context=rerank_top_n)
        self.cache = cache
        self.cost = cost_model if cost_model is not None else CostModel()
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.rerank_top_n = rerank_top_n
        self.use_cache = use_cache
        self.use_torch_rerank = use_torch_rerank

    def run(self, query: str) -> QueryTrace:
        """跑一条查询，返回带延迟/成本的完整 trace。"""
        # ---- 0) 缓存查询（命中则短路整条管线）----
        if self.use_cache and self.cache is not None:
            # 用检索器的向量化器给 query 编码，作为语义缓存的 key 向量
            q_vec = self.retriever.vectorizer.transform([query])[0]
            cached = self.cache.get(query, q_vec)
            if cached is not None:
                ans, hits, ptoks = cached
                return QueryTrace(query=query, answer=ans, hits=hits, cache_hit=True,
                                  latency_ms=self.cost.cache_hit_ms, cost_usd=0.0,
                                  prompt_tokens=ptoks)

        latency = 0.0

        # ---- 1) 一阶段检索 ----
        hits = self.retriever.search(query, top_k=self.top_k)
        latency += self.cost.retrieval_ms_per_chunk * len(self.retriever.chunks)

        # ---- 2) 可选重排 ----
        if self.use_rerank and hits:
            latency += self.cost.rerank_ms_per_candidate * len(hits)
            hits = rerank(query, hits, top_n=self.rerank_top_n,
                          use_torch=self.use_torch_rerank)
        else:
            hits = hits[:self.rerank_top_n]   # 不重排也只把前 rerank_top_n 条送生成

        # ---- 3) 生成（MockLLM）----
        context_chunks = [h.chunk for h in hits]
        ptoks = self.cost.prompt_tokens(context_chunks)
        latency += self.cost.gen_ms_base + self.cost.gen_ms_per_token * ptoks
        cost_usd = ptoks / 1000.0 * self.cost.usd_per_1k_tokens
        answer = self.generator.generate(query, hits)

        # ---- 4) 写缓存（供后续相同/相似查询命中）----
        if self.use_cache and self.cache is not None:
            q_vec = self.retriever.vectorizer.transform([query])[0]
            self.cache.put(query, (answer, hits, ptoks), q_vec)

        return QueryTrace(query=query, answer=answer, hits=hits, cache_hit=False,
                          latency_ms=latency, cost_usd=cost_usd, prompt_tokens=ptoks)
