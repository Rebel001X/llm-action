# -*- coding: utf-8 -*-
"""
chunking.py —— 把文档切成 chunk（检索真正入库的最小单元）。

为什么要 chunk？
    LLM 上下文有限、检索也希望"命中的片段尽量聚焦"。把长文切成小段：
      · chunk 小 → 命中更精准（precision↑），但一段可能丢掉上下文（recall 可能↓）
      · chunk 大 → 上下文更全（单段信息多），但相关性被稀释、塞进 prompt 的 token 更多（成本↑）
    这就是《Systems Design in the LLM Era》第 2 章反复强调的 **"模式即权衡"**。
    本实验的一个核心目的就是**用数据画出 chunk 大小的权衡曲线**。

本模块提供"按词数滑动窗口 + 可选重叠"的切法——工业界最常用的基线切法。
"""

from dataclasses import dataclass
from typing import List

from corpus import Document
from vectorizer import tokenize


@dataclass
class Chunk:
    """一个 chunk：既记它来自哪篇文档（用于引用/评测），也记它自己的 id 与文本。"""
    chunk_id: str      # 形如 "d02#1"，全局唯一
    doc_id: str        # 来源文档 id（recall 评测按 doc_id 归并）
    title: str         # 来源文档标题（生成答案引用时展示）
    text: str          # chunk 正文
    position: int      # 它是该文档的第几个 chunk（从 0 起）


def chunk_document(doc: Document, chunk_size: int, overlap: int = 0) -> List[Chunk]:
    """
    把一篇文档按"词数窗口"切块。

    参数：
      chunk_size : 每个 chunk 最多多少个词（token 近似）
      overlap    : 相邻 chunk 重叠多少词（>0 可缓解"切在句子中间丢上下文"的问题）

    ⚠️ 常见坑：overlap >= chunk_size 会导致窗口不前进 → 死循环 / chunk 爆炸。
        这里强制 step = max(1, chunk_size - overlap) 兜底。
    """
    assert chunk_size > 0, "chunk_size 必须为正"
    assert overlap >= 0, "overlap 不能为负"
    words = tokenize(doc.text)
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[Chunk] = []
    pos = 0
    start = 0
    while start < len(words):
        window = words[start:start + chunk_size]
        text = " ".join(window)
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{pos}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=text,
            position=pos,
        ))
        pos += 1
        start += step
        # 若这一窗已经吃到文档末尾，就停（避免最后再产出一个纯重叠的重复块）
        if start + chunk_size >= len(words) and start < len(words):
            tail = words[start:]
            if tail and tail != window[-len(tail):]:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}#{pos}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=" ".join(tail),
                    position=pos,
                ))
            break
    return chunks


def build_chunks(docs: List[Document], chunk_size: int, overlap: int = 0) -> List[Chunk]:
    """对整个语料切块，返回扁平 chunk 列表（这就是"待索引的最小单元集合"）。"""
    out: List[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size, overlap))
    return out
