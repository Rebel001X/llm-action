# -*- coding: utf-8 -*-
"""
generator.py —— 离线 MockLLM：把检索到的 chunk 组织成"带引用的答案"。

为什么要 MockLLM？
    本项目要求**完全离线、不需 key**。真实生成器（Claude / GPT）在这里被替换成一个
    确定性的"抽取式"MockLLM：它不发明事实，只从检索到的 chunk 里挑句子拼答案，
    并给每句挂上引用标记 [S1] [S2]…，再把标记映射回真实 chunk_id / 文档标题。

    这样做的好处：
      1) 确定性 → 可写单元测试（同输入同输出）。
      2) "引用映射"可被严格校验：每个 [S{i}] 必须对应一个真实存在的检索片段，
         且该片段确实来自被引用的文档 —— 这正是"可信 RAG"的核心可测点。
    真实系统把 generate() 内部换成一次 LLM 调用即可，**引用契约（citation contract）不变**。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from retriever import RetrievedChunk
from vectorizer import tokenize


@dataclass
class Citation:
    """一条引用：答案里的 [S{marker}] 对应哪个 chunk / 文档。"""
    marker: int          # 引用序号，从 1 起（对应答案文本里的 [S1]）
    chunk_id: str        # 真实被引用的 chunk id
    doc_id: str
    title: str
    snippet: str         # 被引用的具体文本片段（可回溯核验）


@dataclass
class Answer:
    """MockLLM 的输出：答案文本 + 引用列表 + 用到的上下文。"""
    text: str
    citations: List[Citation] = field(default_factory=list)
    used_chunk_ids: List[str] = field(default_factory=list)

    def cited_markers(self) -> List[int]:
        """从答案文本里解析出所有 [S{n}] 的编号，用于校验引用完整性。"""
        return [int(m) for m in re.findall(r"\[S(\d+)\]", self.text)]


def _best_sentence(question: str, passage: str) -> str:
    """
    从一个 chunk 里挑"和问题最相关"的一句话（词重叠最多的句子）。
    抽取式生成：只搬运原文句子，绝不编造 —— 这是 MockLLM"零幻觉"的关键。
    """
    # 简单按句号切句；chunk 本身较短，够用。
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
    if not sentences:
        return passage.strip()
    q = set(tokenize(question))
    best, best_score = sentences[0], -1
    for s in sentences:
        score = len(q & set(tokenize(s)))
        if score > best_score:
            best, best_score = s, score
    return best


class MockLLM:
    """
    确定性抽取式生成器：读 top 检索片段 → 每片挑一句 → 拼成带引用的答案。

    参数：
      max_context : 最多用前几个 chunk 来组织答案（对应"塞进 prompt 的上下文条数"，影响成本）。
    """

    def __init__(self, max_context: int = 3) -> None:
        self.max_context = max_context

    def generate(self, question: str, hits: List[RetrievedChunk]) -> Answer:
        """
        生成带引用答案。若没有任何检索结果，明确返回"我不知道"——
        这也是可信 RAG 的要求：宁可拒答，不可幻觉。
        """
        if not hits:
            return Answer(text="I don't know — no relevant context was retrieved.",
                          citations=[], used_chunk_ids=[])
        used = hits[:self.max_context]
        parts: List[str] = []
        citations: List[Citation] = []
        for i, h in enumerate(used, start=1):
            sent = _best_sentence(question, h.chunk.text)
            # 关键：句子后紧跟引用标记 [S{i}]，标记编号 i 与 citations[i-1] 严格对应。
            parts.append(f"{sent} [S{i}]")
            citations.append(Citation(
                marker=i,
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                snippet=sent,
            ))
        text = " ".join(parts)
        return Answer(text=text, citations=citations,
                      used_chunk_ids=[h.chunk.chunk_id for h in used])


def verify_citations(answer: Answer) -> bool:
    """
    校验引用契约（供测试与生产守卫使用）：
      1) 答案文本里每个 [S{n}] 都能在 citations 里找到对应 marker；
      2) 每条 citation 的 snippet 确实是它所引 chunk 的子串（引用真实、可回溯）；
      3) marker 连续从 1 起（1..len(citations)），不缺号不跳号。
    任一不满足即返回 False —— 这就是"引用映射真实片段"的自动化检查。
    """
    markers_in_text = set(answer.cited_markers())
    markers_declared = {c.marker for c in answer.citations}
    if markers_in_text != markers_declared:
        return False
    expected = set(range(1, len(answer.citations) + 1))
    if markers_declared != expected:
        return False
    for c in answer.citations:
        # snippet 必须真的出现在被引 chunk 的文本里（大小写/空白不敏感的宽松子串判断）
        norm_snip = " ".join(c.snippet.lower().split())
        # 这里我们无法直接拿到 chunk 全文，故约定 snippet 已由 generate 从 chunk 抽取；
        # 交叉校验放在 verify_answer_against_hits 里做（能同时拿到 hits）。
        if not norm_snip:
            return False
    return True


def verify_answer_against_hits(answer: Answer, hits: List[RetrievedChunk]) -> bool:
    """
    更强的校验：把答案的每条引用 snippet 与"真实检索到的 chunk 文本"对齐核验，
    确认 snippet 确实是对应 chunk 的原文子串，且 chunk_id/doc_id 与检索结果一致。
    """
    if not verify_citations(answer):
        return False
    by_id = {h.chunk.chunk_id: h.chunk for h in hits}
    for c in answer.citations:
        chunk = by_id.get(c.chunk_id)
        if chunk is None:
            return False                      # 引用了一个根本没被检索到的 chunk → 造假
        if c.doc_id != chunk.doc_id:
            return False                      # doc_id 对不上 → 引用映射错乱
        norm_snip = " ".join(c.snippet.lower().split())
        norm_full = " ".join(chunk.text.lower().split())
        if norm_snip not in norm_full:
            return False                      # snippet 不是原文子串 → 幻觉/篡改
    return True
