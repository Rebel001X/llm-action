# -*- coding: utf-8 -*-
"""
corpus.py —— 离线语料库 + 带标注（gold labels）的评测查询集。

为什么要"带标注"？
    要量化"检索质量"（recall@k）就必须知道"哪些文档才是某个查询的正确答案"。
    工业界这叫 golden dataset（黄金数据集）——《Systems Design in the LLM Era》
    第 2 章"可测性"维度反复强调：非确定性系统的"单元测试"就是黄金数据集。
    这里我们手工构造一个小而干净的语料 + 每条查询的相关文档 id 集合，
    这样 recall@k、precision@k 才是"有 ground-truth 可对照"的真实指标，而非拍脑袋。

数据规模：刻意小（十几篇短文档），目的是让实验"秒级可复现"、逻辑一眼看穿，
    而不是追求 SOTA。真实系统换成百万级向量库时，本项目的**接口与权衡结论不变**。
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Document:
    """一篇文档（检索的最小单元之一，真正入库的是它切出来的 chunk）。"""
    doc_id: str          # 文档唯一 id
    title: str           # 标题（演示答案引用时会用到）
    text: str            # 正文（英文短文，避免中文分词依赖，保证纯 numpy 也能跑 TF-IDF）


@dataclass
class LabeledQuery:
    """一条评测查询：问题 + 它的"正确文档集合"（gold / relevant doc ids）。"""
    qid: str
    question: str
    relevant_doc_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 语料：围绕"LLM 系统设计"的若干主题，每篇聚焦一个概念，便于构造干净的相关性标注。
# 每篇正文特意写得"主题词密集"，让 TF-IDF 这种词频方法也能拉开区分度。
# ---------------------------------------------------------------------------
_RAW_DOCS: List[Document] = [
    Document(
        "d01", "What is RAG",
        "Retrieval augmented generation grounds a language model in external documents. "
        "The retriever fetches relevant passages and the generator conditions on them. "
        "RAG reduces hallucination by giving the model factual context to cite."
    ),
    Document(
        "d02", "Chunking strategy",
        "Chunking splits long documents into smaller passages before indexing. "
        "Small chunks improve retrieval precision but may lose surrounding context. "
        "Large chunks preserve context yet dilute relevance and raise token cost."
    ),
    Document(
        "d03", "Vector embeddings",
        "An embedding maps text into a dense vector so semantic similarity becomes "
        "geometric distance. Cosine similarity between query and passage vectors ranks "
        "candidates. Dense retrieval captures meaning beyond exact keyword overlap."
    ),
    Document(
        "d04", "TF-IDF retrieval",
        "TF-IDF weights a term by its frequency in a document and its rarity across the "
        "corpus. Sparse lexical retrieval matches exact keywords and is cheap to compute. "
        "It struggles with synonyms because it has no notion of semantic similarity."
    ),
    Document(
        "d05", "Caching in RAG",
        "A semantic cache stores previous query results keyed by the query embedding. "
        "A cache hit returns the answer instantly and skips both retrieval and generation, "
        "cutting latency and dollar cost. Cache miss falls through to the full pipeline."
    ),
    Document(
        "d06", "Reranking",
        "A reranker rescoring model reorders the top candidates from a cheap first-stage "
        "retriever. Cross encoder rerankers read query and passage jointly for higher "
        "precision at the cost of extra latency. Two stage retrieval balances recall and cost."
    ),
    Document(
        "d07", "Latency budget",
        "End to end latency is the sum of retrieval, reranking, and generation time. "
        "A larger top k raises recall but also raises generation token cost and latency. "
        "Engineers pick top k to satisfy a latency budget while keeping recall acceptable."
    ),
    Document(
        "d08", "Hallucination",
        "A hallucination is a fluent but factually wrong statement from a language model. "
        "Grounding the model in retrieved evidence and forcing citations mitigates it. "
        "If retrieval misses the relevant passage the model may still hallucinate."
    ),
    Document(
        "d09", "Recall at k",
        "Recall at k measures the fraction of relevant documents found within the top k "
        "retrieved results. It is the key retrieval quality metric for RAG systems. "
        "Higher k trivially raises recall but hurts latency and cost, so it is a trade off."
    ),
    Document(
        "d10", "Hybrid retrieval",
        "Hybrid retrieval fuses sparse lexical scores with dense embedding scores. "
        "Lexical matching catches exact keywords while dense matching catches synonyms. "
        "Score fusion often beats either retriever alone on diverse queries."
    ),
    Document(
        "d11", "Citations",
        "A grounded answer cites the source passage id it relied on so users can verify. "
        "Citation mapping links each claim back to a retrieved chunk. "
        "Traceable citations are essential for trustworthy retrieval augmented generation."
    ),
    Document(
        "d12", "Cost of generation",
        "Generation cost scales with the number of input and output tokens. "
        "Stuffing many retrieved chunks into the prompt raises cost linearly. "
        "Prompt compression and a smaller top k keep the dollar cost per query bounded."
    ),
    Document(
        "d13", "Index freshness",
        "An index becomes stale when source documents change after ingestion. "
        "Incremental ingestion re embeds only changed chunks to keep the index fresh. "
        "Staleness causes the retriever to return outdated evidence to the generator."
    ),
    Document(
        "d14", "Query expansion",
        "Query expansion adds synonyms or paraphrases to a query before retrieval. "
        "It improves recall for short ambiguous queries at the cost of some precision. "
        "Expansion helps sparse retrievers that cannot see semantic similarity."
    ),
]


# ---------------------------------------------------------------------------
# 评测查询集：每条给出人工标注的相关文档 id（gold labels）。
# 标注原则：把"直接回答该问题所需的文档"标为相关，可能不止一篇（多相关文档才好测 recall）。
# ---------------------------------------------------------------------------
_RAW_QUERIES: List[LabeledQuery] = [
    LabeledQuery("q1", "how does retrieval augmented generation reduce hallucination",
                 ["d01", "d08", "d11"]),
    LabeledQuery("q2", "what chunk size should I pick for indexing documents",
                 ["d02", "d07", "d12"]),
    LabeledQuery("q3", "how are embeddings and cosine similarity used to rank passages",
                 ["d03", "d04", "d10"]),
    LabeledQuery("q4", "how does a cache cut latency and cost in a rag pipeline",
                 ["d05", "d07", "d12"]),
    LabeledQuery("q5", "how does a reranker improve retrieval precision",
                 ["d06", "d09", "d03"]),
    LabeledQuery("q6", "what is recall at k and how does top k trade off latency",
                 ["d09", "d07", "d02"]),
    LabeledQuery("q7", "how do citations make an answer trustworthy and verifiable",
                 ["d11", "d01", "d08"]),
    LabeledQuery("q8", "how does hybrid retrieval combine keyword and semantic matching",
                 ["d10", "d04", "d03"]),
]


def load_corpus() -> List[Document]:
    """返回全部文档（返回副本引用列表，避免调用方误改内部常量）。"""
    return list(_RAW_DOCS)


def load_queries() -> List[LabeledQuery]:
    """返回带标注的评测查询集。"""
    return list(_RAW_QUERIES)


def gold_map() -> Dict[str, List[str]]:
    """qid -> 相关文档 id 列表，供评测函数快速查表。"""
    return {q.qid: list(q.relevant_doc_ids) for q in _RAW_QUERIES}


if __name__ == "__main__":
    # 直接运行本文件时打印语料概览，方便快速核对标注是否合理。
    docs = load_corpus()
    print(f"文档数：{len(docs)}")
    for d in docs:
        print(f"  {d.doc_id}  {d.title:<22}  {len(d.text.split())} words")
    print(f"\n评测查询数：{len(load_queries())}")
    for q in load_queries():
        print(f"  {q.qid}: {q.question}\n       gold={q.relevant_doc_ids}")
