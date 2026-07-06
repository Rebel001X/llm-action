# -*- coding: utf-8 -*-
"""
rag.py —— 迷你 RAG 主流程（对应第 18 章，串起 14–17 章）

一条完整链路：
    问题 --embed--> 向量 --检索(余弦 top-k)--> 命中片段 --拼 prompt--> 生成答案

生成环节有三条分支，默认走**离线**，保证不联网、秒级、可复现：
  - "offline"（默认）：把命中的事实抽取组织成模板答案（不需要任何 LLM）。
  - "ollama"  ：调本地 Ollama（第 17 章）。连不上就自动回退离线。
  - "hf"      ：调 transformers 本地/微调模型（第 15/16 章）。缺失就自动回退离线。

README 里详细讲了如何把离线生成"升级"成 Ollama 或微调后的 HF 模型。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from embed import DeterministicEmbedder, get_embedder
from knowledge import get_facts
from store import SearchHit, VectorStore


# ------------------------------------------------------------------ #
# 建库：把知识库嵌入并写入向量库（离线索引阶段）
# ------------------------------------------------------------------ #
def build_store(embedder=None) -> Tuple[VectorStore, object]:
    """构建向量库。返回 (store, embedder)。

    注意：建库与查询必须用**同一个 embedder**（第 18 章反复强调的铁律），
    所以这里把 embedder 一并返回，供后续检索复用。
    """
    if embedder is None:
        embedder = DeterministicEmbedder(dim=256)
    store = VectorStore()
    for fact in get_facts():
        vec = embedder.encode(fact["text"])
        store.add(fact["id"], fact["text"], vec)
    return store, embedder


# ------------------------------------------------------------------ #
# 检索：问题 -> 向量 -> 余弦 top-k（在线查询阶段的前半段）
# ------------------------------------------------------------------ #
def retrieve(question: str, k: int = 3, store=None, embedder=None) -> List[SearchHit]:
    if store is None or embedder is None:
        store, embedder = build_store(embedder)
    q_vec = embedder.encode(question)
    return store.search(q_vec, k=k)


def _build_context(hits: List[SearchHit]) -> str:
    """把命中片段拼成一段可注入 prompt 的 context（第 18 章的『增强』动作）。"""
    return "\n\n".join(h.text for h in hits)


def _build_prompt(question: str, context: str) -> str:
    """组织成一份"只准依据 context 回答"的 prompt（书里 system prompt 的精神）。"""
    return (
        "你是一个严谨的助手，只能根据下面提供的【资料】回答问题；"
        "如果资料里没有答案，就说明资料中没有相关信息。\n\n"
        f"【资料】\n{context}\n\n"
        f"【问题】{question}\n\n【答案】"
    )


# ------------------------------------------------------------------ #
# 生成分支
# ------------------------------------------------------------------ #
def generate_offline(question: str, hits: List[SearchHit]) -> str:
    """离线模板生成：把命中的事实抽取、组织成答案（不依赖任何 LLM）。

    这保证了默认路径完全离线、确定性——答案里逐字包含检索命中的原文事实，
    既是"用证据说话"的 RAG 精神，也让结果可被单元测试精确验证。
    """
    if not hits:
        return f"抱歉，知识库中没有找到与「{question}」相关的信息。"

    lines = [f"根据知识库，关于「{question}」，检索到以下相关事实："]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h.text}（相关度 {h.score:.3f}）")
    lines.append("")
    # 把最相关的一条作为直接答案给出（去掉末尾句号更像"回答"）
    lines.append("综上，最相关的答案是：" + hits[0].text)
    return "\n".join(lines)


def generate_with_ollama(
    question: str,
    context: str,
    model: str = "llama3.1:latest",
    url: str = "http://localhost:11434/api/chat",
    temperature: float = 0.2,
    timeout: float = 30.0,
) -> Optional[str]:
    """调本地 Ollama（第 17 章）。任何异常（未装 requests / 连不上 / 报错）返回 None。"""
    try:
        import requests  # 延迟导入，缺失也不影响离线路径
    except Exception:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, say you don't know."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature}}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception:
        return None  # 连不上就回退


def generate_with_hf(
    question: str,
    context: str,
    model_name: str = "distilgpt2",
    max_new_tokens: int = 80,
) -> Optional[str]:
    """调 transformers 本地模型（第 15/16 章：可换成你微调后的模型目录）。

    默认**不会**被调用（离线索引/测试都走 offline），因为首次使用需要下载权重。
    只有你显式选择 backend='hf' 时才会尝试；失败（缺库/缺权重/离线）返回 None。
    """
    try:
        from transformers import pipeline  # 延迟导入
    except Exception:
        return None
    try:
        gen = pipeline("text-generation", model=model_name)
        prompt = _build_prompt(question, context)
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = out[0]["generated_text"]
        # 只返回 prompt 之后新生成的部分
        return text[len(prompt):].strip() or text
    except Exception:
        return None


# ------------------------------------------------------------------ #
# 顶层入口：answer()
# ------------------------------------------------------------------ #
def answer(
    question: str,
    k: int = 3,
    backend: str = "offline",
    store=None,
    embedder=None,
) -> str:
    """回答一个问题。

    参数
    ----
    backend : {"offline", "ollama", "hf", "auto"}
        - "offline"（默认）：纯离线模板，保证不联网、确定性。
        - "ollama" / "hf"  ：尝试真实 LLM，失败自动回退离线。
        - "auto"           ：先试 ollama，再试 hf，最后离线。
    """
    if store is None or embedder is None:
        store, embedder = build_store(embedder)

    hits = retrieve(question, k=k, store=store, embedder=embedder)
    context = _build_context(hits)

    if backend in ("ollama", "auto"):
        out = generate_with_ollama(question, context)
        if out:
            return out
    if backend in ("hf", "auto"):
        out = generate_with_hf(question, context)
        if out:
            return out

    # 默认 / 兜底：离线模板生成
    return generate_offline(question, hits)


if __name__ == "__main__":  # pragma: no cover
    # 小自测：直接 `python rag.py`
    for q in ["什么是 RAG？", "怎么在本地部署大模型？"]:
        print("Q:", q)
        print(answer(q, k=2))
        print("-" * 60)
