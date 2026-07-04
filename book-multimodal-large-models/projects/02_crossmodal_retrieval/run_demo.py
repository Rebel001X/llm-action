# -*- coding: utf-8 -*-
"""
跨模态检索 · 可视化演示
=====================
运行:  python run_demo.py

产出 (保存在本目录 figures/ 下):
    1. recall_curve.png   —— 文->图 与 图->文 的 recall@k 曲线,叠加不同噪声
    2. norm_vs_temp.png   —— 归一化 (改结果) vs 温度 (不改排序) 的对比
    3. retrieval_examples.png —— 一条文本 query 的 top-k 检索样例热力条

同时在终端打印一张指标汇总表。全程 Agg 后端 + 中文字体,离线可跑。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端: 只出图不弹窗,服务器/CI 友好
import matplotlib.pyplot as plt
import numpy as np

# 中文字体 (Windows 常见); 负号正常显示。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from retrieval import (
    add_noise,
    ranks_of_ground_truth,
    recall_at_k,
    retrieval_report,
    similarity_matrix,
    softmax_retrieval_probs,
    topk_indices,
)
from toydata import make_paired_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# --------------------------------------------------------------------------- #
# 演示 1: 双向 recall@k 曲线 + 噪声影响                                          #
# --------------------------------------------------------------------------- #
def demo_recall_curves():
    banner("演示 1: recall@k 曲线 (文->图 / 图->文) 与噪声影响")
    # dim=32 + modality_sigma=0.7: 基础对齐不完美,留出让噪声「拉开差距」的空间。
    img, txt = make_paired_embeddings(n=500, dim=32, modality_sigma=0.7, seed=0)
    ks = [1, 2, 3, 5, 10, 20, 50, 100]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sigmas = [0.0, 0.5, 1.0]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for ax, direction in zip(axes, ["t2i", "i2t"]):
        for sigma, c in zip(sigmas, colors):
            # 关键: 每个噪声档位都用「同一颗种子」的新 rng, 保证不同 σ 之间
            # 是独立可比的实验 (而非在同一条噪声流上累加), 曲线才会干净单调。
            rng = np.random.default_rng(100)
            # 只给「query 侧」加噪,模拟检索输入变脏。
            if direction == "t2i":  # 文 -> 图: query=文, gallery=图
                q = add_noise(txt, sigma, rng=rng)
                g = img
            else:  # 图 -> 文: query=图, gallery=文
                q = add_noise(img, sigma, rng=rng)
                g = txt
            sim = similarity_matrix(q, g)
            ranks = ranks_of_ground_truth(sim)
            recalls = [recall_at_k(ranks, k) for k in ks]
            medr = float(np.median(ranks))
            ax.plot(
                ks, recalls, "-o", color=c, markersize=4,
                label=f"σ={sigma}  (MedR={medr:.0f})",
            )
        title = "文 → 图 (text-to-image)" if direction == "t2i" else "图 → 文 (image-to-text)"
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel("recall@k")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend(title="query 噪声")

    fig.suptitle("跨模态检索: recall@k 随 k 单调上升;噪声越大曲线越低", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "recall_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")

    # 终端汇总表
    print("\n  方向    σ    R@1    R@5    R@10   MedR")
    print("  " + "-" * 40)
    for direction in ["t2i", "i2t"]:
        for sigma in sigmas:
            rng2 = np.random.default_rng(100)  # 与画图同种子, 数值一致
            if direction == "t2i":
                q, g = add_noise(txt, sigma, rng=rng2), img
            else:
                q, g = add_noise(img, sigma, rng=rng2), txt
            rep = retrieval_report(q, g, ks=(1, 5, 10))
            print(
                f"  {direction:>4}  {sigma:>3}  {rep['recall@1']:.3f}  "
                f"{rep['recall@5']:.3f}  {rep['recall@10']:.3f}  "
                f"{rep['median_rank']:.0f}"
            )


# --------------------------------------------------------------------------- #
# 演示 2: 归一化改结果, 温度不改排序                                             #
# --------------------------------------------------------------------------- #
def demo_norm_and_temperature():
    banner("演示 2: 归一化 (改结果) vs 温度 (只改置信度, 不改排序)")
    img, txt = make_paired_embeddings(n=400, dim=32, modality_sigma=0.7, seed=1)

    # 制造「长度偏置」: 把每个文嵌入按一个随机幅度 (0.05~20 倍) 缩放,
    # 破坏公平性 —— 长向量天然点积更大, 会挤到检索前排。
    rng = np.random.default_rng(2)
    scales = rng.uniform(0.05, 20.0, size=(txt.shape[0], 1))
    txt_scaled = txt * scales

    r_no = retrieval_report(txt_scaled, img, ks=(1, 5, 10), normalize=False)
    r_yes = retrieval_report(txt_scaled, img, ks=(1, 5, 10), normalize=True)

    print("  带长度偏置的文->图检索:")
    print(f"    不归一化: R@1={r_no['recall@1']:.3f}  MedR={r_no['median_rank']:.0f}")
    print(f"    归一化  : R@1={r_yes['recall@1']:.3f}  MedR={r_yes['median_rank']:.0f}")
    print("    => 归一化消除长度偏置, 通常显著提升 recall。")

    # 图: 左=归一化前后 recall 对比; 右=同一行不同温度的概率分布。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ks = ["R@1", "R@5", "R@10"]
    no_vals = [r_no["recall@1"], r_no["recall@5"], r_no["recall@10"]]
    yes_vals = [r_yes["recall@1"], r_yes["recall@5"], r_yes["recall@10"]]
    x = np.arange(len(ks))
    w = 0.35
    axes[0].bar(x - w / 2, no_vals, w, label="不归一化", color="#d62728")
    axes[0].bar(x + w / 2, yes_vals, w, label="L2 归一化", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ks)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("recall")
    axes[0].set_title("归一化 = 消除长度偏置, 改变检索结果")
    axes[0].legend(loc="lower right")
    for i, (a, b) in enumerate(zip(no_vals, yes_vals)):
        axes[0].text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)

    # 右: 取第 0 条 query 与全库的相似度, 不同温度看 softmax 概率。
    sim = similarity_matrix(txt, img)  # 用干净嵌入让 top-1 明显
    row = sim[0:1]  # (1, N)
    top = topk_indices(row, k=8)[0]  # 展示 top-8 的概率分布
    for T, c in zip([0.05, 0.2, 1.0], ["#1f77b4", "#ff7f0e", "#9467bd"]):
        p = softmax_retrieval_probs(row, temperature=T)[0]
        axes[1].plot(range(8), p[top], "-o", color=c, label=f"T={T}")
    axes[1].set_title("温度 T: 只改概率『软硬』, 不改 top-k 顺序")
    axes[1].set_xlabel("top-k 位次 (顺序不随 T 变)")
    axes[1].set_ylabel("softmax 检索概率")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="温度")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "norm_vs_temp.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# 演示 3: 检索样例 (一条文本 query 的 top-k 命中可视化)                          #
# --------------------------------------------------------------------------- #
def demo_examples():
    banner("演示 3: 检索样例 —— 文本 query 的 top-5 命中情况")
    # dim=24 + sigma=1.0: 故意让检索有难度, 这样既有 top-1 命中也有未命中,
    # 展示「rank>1」这种真实检索里常见的情形。
    img, txt = make_paired_embeddings(n=200, dim=24, modality_sigma=1.0, seed=4)
    sim = similarity_matrix(txt, img)  # 文 -> 图
    top = topk_indices(sim, k=5)

    # 挑几条 query 展示: 有的 top-1 命中, 有的排在后面。
    ranks = ranks_of_ground_truth(sim)
    # 取 3 条命中 top1 的, 3 条没命中 top1 的, 合成一张示意图。
    hit = np.where(ranks == 1)[0][:3]
    miss = np.where(ranks > 1)[0][:3]
    show = list(hit) + list(miss)

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(show) + 1.5))
    for r, qi in enumerate(show):
        for c in range(5):
            g = top[qi, c]
            correct = g == qi  # 正确图的下标就是 query 自己的下标
            color = "#2ca02c" if correct else "#cccccc"
            ax.add_patch(
                plt.Rectangle((c, r), 0.92, 0.92, color=color, ec="white")
            )
            ax.text(
                c + 0.46, r + 0.46, f"#{g}", ha="center", va="center",
                fontsize=8, color="black",
            )
        ax.text(-0.3, r + 0.46, f"文{qi}\n(rank={ranks[qi]})",
                ha="right", va="center", fontsize=8)

    ax.set_xlim(-2, 5)
    ax.set_ylim(0, len(show))
    ax.set_xticks(np.arange(5) + 0.46)
    ax.set_xticklabels([f"top{c+1}" for c in range(5)])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_title("文本→图像检索样例 (绿=正确配对被检出, 灰=其它)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "retrieval_examples.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [saved] {out}")
    print(f"  示例中 rank=1 表示 top-1 直接命中; rank>1 表示正确图排在更后面。")


def main():
    demo_recall_curves()
    demo_norm_and_temperature()
    demo_examples()
    banner("完成! 打开 figures/ 目录查看 3 张图")


if __name__ == "__main__":
    main()
