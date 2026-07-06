# -*- coding: utf-8 -*-
"""
run_demo.py —— 迷你 RAG 演示

问几个问题，打印：检索到的片段（带相似度）+ 生成的答案。
默认走离线分支，不联网、几秒跑完。

用法：
    python run_demo.py                # 离线模板生成（默认）
    python run_demo.py ollama         # 尝试本地 Ollama，连不上自动回退离线
    python run_demo.py hf             # 尝试本地 transformers 模型，缺失自动回退离线
"""

from __future__ import annotations

import sys

# Windows 控制台可能默认 GBK，强制 UTF-8 输出，避免中文乱码。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag import answer, build_store, retrieve


QUESTIONS = [
    "什么是 RAG 检索增强生成？",
    "怎么在本地部署和服务大模型？",
    "卷积神经网络是用来干什么的？",
    "Dataset 和 DataLoader 有什么用？",
    "余弦相似度是什么，取值范围多少？",
]


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "offline"
    store, embedder = build_store()  # 建一次库，多次复用

    print(f"=== 迷你 RAG 演示（backend={backend}）===\n")
    for q in QUESTIONS:
        print(f"❓ 问题：{q}")

        hits = retrieve(q, k=3, store=store, embedder=embedder)
        print("🔎 检索到的片段（top-3，按相似度降序）：")
        for i, h in enumerate(hits, 1):
            print(f"   [{i}] ({h.score:.3f}) {h.text}")

        ans = answer(q, k=3, backend=backend, store=store, embedder=embedder)
        print("💬 生成的答案：")
        for line in ans.splitlines():
            print(f"   {line}")
        print("-" * 70)


if __name__ == "__main__":
    main()
