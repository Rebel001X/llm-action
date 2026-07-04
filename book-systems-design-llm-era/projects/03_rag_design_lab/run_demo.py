# -*- coding: utf-8 -*-
"""
run_demo.py —— RAG 设计权衡实验一键演示（离线出图）。

跑什么：
  1) top-k vs recall / precision / 端到端延迟 / 成本 的权衡曲线；
  2) chunk 大小 vs recall / chunk 数 的权衡；
  3) 缓存开/关 对平均延迟的影响（命中率驱动）；
  4) 打印一条端到端带引用答案的示例（证明"引用映射真实片段"）。

产物：在本目录生成 3 张 PNG 图 + 终端打印一份权衡小结表。
所有计算离线、确定性，不联网、不需 key。
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")                    # 无界面后端：服务器/CI 也能出图，不弹窗
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import rcParams          # noqa: E402

# 中文字体：优先微软雅黑，退回黑体；并修正负号显示为方块的问题。
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import load_corpus, load_queries, gold_map          # noqa: E402
from chunking import build_chunks                                # noqa: E402
from vectorizer import TfidfVectorizer                           # noqa: E402
from retriever import Retriever                                  # noqa: E402
from metrics import evaluate_queries                             # noqa: E402
from cache import SemanticCache                                  # noqa: E402
from generator import MockLLM, verify_answer_against_hits        # noqa: E402
from pipeline import RAGPipeline                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _build(chunk_size=40, overlap=8):
    docs = load_corpus()
    chunks = build_chunks(docs, chunk_size, overlap)
    r = Retriever(TfidfVectorizer()).index(chunks)
    return r, chunks


# ---------------------------------------------------------------------------
# 实验 1：top-k vs recall / precision / 延迟 / 成本
# ---------------------------------------------------------------------------
def experiment_topk():
    r, _ = _build()
    gold = gold_map()
    queries = load_queries()
    ks = [1, 2, 3, 4, 5, 6, 8, 10]
    recalls, precisions, lat, cost = [], [], [], []
    for k in ks:
        per = {q.qid: r.search(q.question, top_k=k) for q in queries}
        m = evaluate_queries(per, gold, k)
        recalls.append(m[f"recall@{k}"])
        precisions.append(m[f"precision@{k}"])
        # 延迟/成本：用管线跑一遍取平均（rerank_top_n 跟随 k，模拟"上下文越多越贵"）
        lats, costs = [], []
        for q in queries:
            pipe = RAGPipeline(r, top_k=k, rerank_top_n=min(k, 5))
            t = pipe.run(q.question)
            lats.append(t.latency_ms)
            costs.append(t.cost_usd)
        lat.append(sum(lats) / len(lats))
        cost.append(sum(costs) / len(costs) * 1000)   # 换算成 $/千次 便于观感

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(ks, recalls, "o-", color="#2563eb", label="recall@k（检索质量↑）", linewidth=2)
    ax1.plot(ks, precisions, "s--", color="#16a34a", label="precision@k（噪声↓越干净）", linewidth=2)
    ax1.set_xlabel("top-k（一阶段召回条数）")
    ax1.set_ylabel("质量指标（0~1）")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, lat, "^-", color="#dc2626", label="平均端到端延迟 (ms)", linewidth=2)
    ax2.set_ylabel("端到端延迟 (ms)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    # 合并两个 y 轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    plt.title("实验1：top-k 的权衡 —— recall 涨、precision 跌、延迟涨（模式即权衡）")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_topk_tradeoff.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, list(zip(ks, recalls, precisions, lat, cost))


# ---------------------------------------------------------------------------
# 实验 2：chunk 大小 vs recall / chunk 数
# ---------------------------------------------------------------------------
def experiment_chunk_size():
    gold = gold_map()
    queries = load_queries()
    sizes = [8, 12, 16, 24, 40, 64, 100]
    recalls, nchunks = [], []
    K = 5
    for cs in sizes:
        r, chunks = _build(chunk_size=cs, overlap=max(0, cs // 5))
        per = {q.qid: r.search(q.question, top_k=K) for q in queries}
        m = evaluate_queries(per, gold, K)
        recalls.append(m[f"recall@{K}"])
        nchunks.append(len(chunks))

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(sizes, recalls, "o-", color="#7c3aed", linewidth=2, label=f"recall@{K}")
    ax1.set_xlabel("chunk 大小（词数）")
    ax1.set_ylabel(f"recall@{K}", color="#7c3aed")
    ax1.tick_params(axis="y", labelcolor="#7c3aed")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sizes, nchunks, "s--", color="#ea580c", linewidth=2, label="chunk 总数")
    ax2.set_ylabel("chunk 总数（索引规模/成本）", color="#ea580c")
    ax2.tick_params(axis="y", labelcolor="#ea580c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    plt.title("实验2：chunk 大小的权衡 —— 太小碎片化、太大稀释相关性")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_chunk_tradeoff.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, list(zip(sizes, recalls, nchunks))


# ---------------------------------------------------------------------------
# 实验 3：缓存开/关 对平均延迟的影响（模拟带重复的查询流）
# ---------------------------------------------------------------------------
def experiment_cache():
    r, _ = _build()
    queries = load_queries()
    # 构造一个"有重复"的查询流：40% 概率复读上一条（近似真实热点分布）
    stream = []
    import random
    random.seed(7)
    for _ in range(60):
        if stream and random.random() < 0.4:
            stream.append(stream[-1])
        else:
            stream.append(random.choice(queries).question)

    def avg_latency(use_cache):
        cache = SemanticCache(threshold=0.95) if use_cache else None
        pipe = RAGPipeline(r, cache=cache, top_k=5, use_cache=use_cache, rerank_top_n=3)
        lats = [pipe.run(q).latency_ms for q in stream]
        hr = cache.stats.hit_rate if cache else 0.0
        return sum(lats) / len(lats), hr

    lat_off, _ = avg_latency(False)
    lat_on, hit_rate = avg_latency(True)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(["无缓存", f"有缓存\n(命中率 {hit_rate:.0%})"], [lat_off, lat_on],
                  color=["#94a3b8", "#0ea5e9"], width=0.55)
    ax.set_ylabel("平均端到端延迟 (ms)")
    ax.set_title("实验3：语义缓存把平均延迟按'命中率'比例压下来")
    for b, v in zip(bars, [lat_off, lat_on]):
        ax.text(b.get_x() + b.get_width() / 2, v + max(lat_off, lat_on) * 0.01,
                f"{v:.1f} ms", ha="center", va="bottom", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_cache_effect.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, (lat_off, lat_on, hit_rate)


# ---------------------------------------------------------------------------
# 示例：一条端到端带引用的答案
# ---------------------------------------------------------------------------
def demo_cited_answer():
    r, _ = _build()
    q = load_queries()[0].question
    hits = r.search(q, top_k=5)
    ans = MockLLM(max_context=3).generate(q, hits)
    ok = verify_answer_against_hits(ans, hits)
    return q, ans, ok


def main():
    print("=" * 72)
    print("RAG 设计权衡实验（离线 / 确定性 / 纯 numpy 检索 + MockLLM 生成）")
    print("=" * 72)

    out1, rows1 = experiment_topk()
    print("\n[实验1] top-k 权衡：")
    print(f"{'k':>3} | {'recall':>7} | {'precision':>9} | {'延迟ms':>8} | {'成本$/千次':>10}")
    for k, rec, pre, lat, cost in rows1:
        print(f"{k:>3} | {rec:>7.3f} | {pre:>9.3f} | {lat:>8.1f} | {cost:>10.4f}")
    print(f"  → 图已保存：{out1}")

    out2, rows2 = experiment_chunk_size()
    print("\n[实验2] chunk 大小权衡：")
    print(f"{'size':>5} | {'recall@5':>8} | {'chunk数':>7}")
    for cs, rec, n in rows2:
        print(f"{cs:>5} | {rec:>8.3f} | {n:>7}")
    print(f"  → 图已保存：{out2}")

    out3, (lat_off, lat_on, hr) = experiment_cache()
    print("\n[实验3] 缓存效果：")
    print(f"  无缓存平均延迟 = {lat_off:.1f} ms")
    print(f"  有缓存平均延迟 = {lat_on:.1f} ms  (命中率 {hr:.0%})")
    print(f"  延迟下降 = {(1 - lat_on / lat_off) * 100:.1f}%")
    print(f"  → 图已保存：{out3}")

    q, ans, ok = demo_cited_answer()
    print("\n[示例] 一条带引用的离线答案：")
    print(f"  问题：{q}")
    print(f"  答案：{ans.text}")
    print("  引用映射（每个 [S{i}] → 真实片段）：")
    for c in ans.citations:
        print(f"    [S{c.marker}] doc={c.doc_id} 《{c.title}》 chunk={c.chunk_id}")
        print(f"          片段: {c.snippet}")
    print(f"  引用真实性校验通过：{ok}")

    print("\n" + "=" * 72)
    print("完成。共生成 3 张 PNG。核心结论：'模式即权衡'——")
    print("  top-k↑ → recall↑ 但 延迟/成本↑ 且 precision↓；")
    print("  chunk 太小碎片化、太大稀释相关性，存在最优区间；")
    print("  缓存按命中率比例砍平均延迟与成本；重排以少量延迟换精度。")
    print("=" * 72)


if __name__ == "__main__":
    main()
