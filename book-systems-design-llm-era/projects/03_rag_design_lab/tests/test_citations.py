# -*- coding: utf-8 -*-
"""引用映射真实片段 & 生成契约测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import load_corpus, load_queries              # noqa: E402
from chunking import build_chunks                          # noqa: E402
from vectorizer import TfidfVectorizer                     # noqa: E402
from retriever import Retriever, RetrievedChunk            # noqa: E402
from chunking import Chunk                                 # noqa: E402
from generator import (MockLLM, Answer, Citation,          # noqa: E402
                       verify_citations, verify_answer_against_hits)


def _hits_for(question, top_k=5):
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    return r.search(question, top_k=top_k)


def test_answer_has_citation_per_context():
    """答案里的引用条数应等于实际用到的上下文条数（max_context 上限内）。"""
    q = load_queries()[0].question
    hits = _hits_for(q)
    ans = MockLLM(max_context=3).generate(q, hits)
    assert len(ans.citations) == min(3, len(hits))
    assert ans.cited_markers() == list(range(1, len(ans.citations) + 1))


def test_citation_snippet_is_real_substring():
    """每条引用的 snippet 必须是被引 chunk 的真实原文子串（零幻觉）。"""
    q = load_queries()[3].question
    hits = _hits_for(q)
    ans = MockLLM(max_context=3).generate(q, hits)
    assert verify_answer_against_hits(ans, hits) is True
    # 逐条硬核对：snippet 出现在对应 chunk 文本里
    by_id = {h.chunk.chunk_id: h.chunk for h in hits}
    for c in ans.citations:
        chunk = by_id[c.chunk_id]
        assert " ".join(c.snippet.lower().split()) in " ".join(chunk.text.lower().split())
        assert c.doc_id == chunk.doc_id


def test_citation_marker_maps_to_correct_chunk():
    """答案第 i 个引用 [S{i}] 必须映射到检索结果的第 i 个 chunk。"""
    q = load_queries()[2].question
    hits = _hits_for(q)
    ans = MockLLM(max_context=3).generate(q, hits)
    for i, c in enumerate(ans.citations, start=1):
        assert c.marker == i
        assert c.chunk_id == hits[i - 1].chunk.chunk_id


def test_no_context_yields_idk_no_citation():
    """没有检索结果时必须拒答（返回 I don't know），且不产生任何引用。"""
    ans = MockLLM().generate("anything", [])
    assert "don't know" in ans.text.lower()
    assert ans.citations == []
    assert verify_citations(ans) is True     # 无引用也算契约成立


def test_forged_citation_is_rejected():
    """把 snippet 篡改成 chunk 里不存在的文本，校验必须失败（防造假守卫生效）。"""
    chunk = Chunk("d99#0", "d99", "Fake", "the cat sat on the mat", 0)
    hits = [RetrievedChunk(chunk=chunk, score=1.0)]
    forged = Answer(
        text="dogs are loyal companions [S1]",
        citations=[Citation(marker=1, chunk_id="d99#0", doc_id="d99",
                            title="Fake", snippet="dogs are loyal companions")],
        used_chunk_ids=["d99#0"],
    )
    # snippet 不是 chunk 原文子串 → 必须被拒
    assert verify_answer_against_hits(forged, hits) is False


def test_citation_to_unretrieved_chunk_is_rejected():
    """引用了一个根本没被检索到的 chunk_id → 必须被拒（防止凭空引用）。"""
    chunk = Chunk("d1#0", "d1", "Real", "grounding reduces hallucination", 0)
    hits = [RetrievedChunk(chunk=chunk, score=1.0)]
    bad = Answer(
        text="grounding reduces hallucination [S1]",
        citations=[Citation(marker=1, chunk_id="dX#9", doc_id="d1",
                            title="Real", snippet="grounding reduces hallucination")],
        used_chunk_ids=["dX#9"],
    )
    assert verify_answer_against_hits(bad, hits) is False


def test_marker_gap_is_rejected():
    """引用编号跳号（缺 [S1] 只有 [S2]）→ 契约校验必须失败。"""
    bad = Answer(
        text="some claim [S2]",
        citations=[Citation(marker=2, chunk_id="d1#0", doc_id="d1",
                            title="t", snippet="some claim")],
    )
    assert verify_citations(bad) is False


def test_end_to_end_answer_is_verifiable_for_all_queries():
    """全查询集端到端：每条生成答案都应通过引用真实性校验。"""
    docs = load_corpus()
    chunks = build_chunks(docs, 40, 8)
    r = Retriever(TfidfVectorizer()).index(chunks)
    llm = MockLLM(max_context=3)
    for q in load_queries():
        hits = r.search(q.question, top_k=5)
        ans = llm.generate(q.question, hits)
        assert verify_answer_against_hits(ans, hits), f"{q.qid} 引用校验失败"
