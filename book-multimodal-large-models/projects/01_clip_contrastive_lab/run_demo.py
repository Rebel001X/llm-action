# -*- coding: utf-8 -*-
"""
run_demo.py — 一键跑通并出图(离线 / CPU / 中文标注)
====================================================

生成三张图,保存到本目录:
  1) sim_before_after.png : 训练前 vs 训练后的图文相似度热图(对比对角线是否变亮)
  2) loss_curve.png       : InfoNCE 损失曲线 + recall@1 曲线 + 温度曲线
  3) retrieval_ranks.png  : 训练后"图->文"检索排名分布(对角命中越多越好)

matplotlib 使用 Agg 后端(无需显示器),中文字体 Microsoft YaHei。
"""

import os
import sys

# ⚠️坑:Windows 控制台默认 GBK 编码,print 含 emoji/特殊符号会 UnicodeEncodeError。
# 把标准输出重配为 UTF-8(Python 3.7+ 支持 reconfigure)。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")  # 非交互后端:适合脚本 / 无 GUI 环境
import matplotlib.pyplot as plt
import numpy as np
import torch

# 中文显示配置(⚠️坑:不设这两行,中文会变成方框、负号会缺失)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from clip_lab import make_toy_pairs, recall_at_1, train_clip

HERE = os.path.dirname(os.path.abspath(__file__))


def plot_sim_before_after(hist):
    """并排画训练前 / 训练后的相似度热图。"""
    sim_b = hist["sim_before"].numpy()
    sim_a = hist["sim_after"].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, title in [
        (axes[0], sim_b, "训练前:相似度矩阵(对角线=配对)"),
        (axes[1], sim_a, "训练后:相似度矩阵(对角线应更亮)"),
    ]:
        im = ax.imshow(mat, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("文本索引 j")
        ax.set_ylabel("图像索引 i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="余弦相似度")
    fig.suptitle("CLIP 对比学习:训练让配对(对角线)相似度上升、非配对下降", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(HERE, "sim_before_after.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_curves(hist):
    """画损失 / recall@1 / 温度 三条曲线。"""
    steps = np.arange(len(hist["loss"]))
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 左轴:损失
    l1 = ax1.plot(steps, hist["loss"], color="#d62728", label="InfoNCE 损失")
    ax1.set_xlabel("训练步 step")
    ax1.set_ylabel("InfoNCE 损失", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    # 右轴:recall@1
    ax2 = ax1.twinx()
    l2 = ax2.plot(steps, hist["recall"], color="#1f77b4", label="recall@1")
    ax2.set_ylabel("recall@1", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.02, 1.02)

    # 温度曲线画在损失轴上(数值范围接近),用虚线区分
    l3 = ax1.plot(steps, hist["temperature"], color="#2ca02c", linestyle="--", label="温度 temperature")

    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    ax1.set_title("训练动态:损失下降、recall@1 上升、温度自适应")
    fig.tight_layout()
    out = os.path.join(HERE, "loss_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_retrieval_ranks(hist):
    """训练后:每张图在'图->文'检索中,配对文本被排到第几名(0=命中)。"""
    sim = hist["sim_after"]
    n = sim.shape[0]
    # 对每一行(每张图),按相似度从高到低排序,找配对(对角)文本的名次。
    order = sim.argsort(dim=1, descending=True)  # (N,N) 每行是文本索引的降序排列
    ranks = []
    for i in range(n):
        rank = (order[i] == i).nonzero(as_tuple=True)[0].item()  # 配对文本的名次(0-based)
        ranks.append(rank)
    ranks = np.array(ranks)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(ranks, bins=np.arange(-0.5, n + 0.5, 1.0), color="#9467bd", edgecolor="white")
    ax.set_xlabel("配对文本的检索名次(0 = recall@1 命中)")
    ax.set_ylabel("图像数量")
    hit = float((ranks == 0).mean())
    ax.set_title(f"训练后 图->文 检索名次分布(recall@1 = {hit:.2%})")
    ax.set_xlim(-0.5, min(n, 10) - 0.5)  # 只看前 10 名,命中应堆在 0
    fig.tight_layout()
    out = os.path.join(HERE, "retrieval_ranks.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    torch.manual_seed(0)
    print("=" * 64)
    print("CLIP 对比学习 Lab · 离线 CPU 演示")
    print("=" * 64)

    imgs, txts, labels = make_toy_pairs(n_pairs=64, seed=0)
    print(f"数据:{imgs.shape[0]} 对样本 | 图特征 {imgs.shape[1]}维 | 文特征 {txts.shape[1]}维")

    model, hist = train_clip(imgs, txts, steps=300, seed=0, verbose=True)

    r_before = recall_at_1(hist["sim_before"])
    r_after = recall_at_1(hist["sim_after"])
    print("-" * 64)
    print(f"训练前 recall@1 = {r_before:.3f}")
    print(f"训练后 recall@1 = {r_after:.3f}")
    print(f"训练后 等效温度 = {float(model.temperature):.4f}")
    print(f"对角均值 {hist['sim_after'].diag().mean():.3f} vs "
          f"非对角均值 {hist['sim_after'][~torch.eye(imgs.shape[0], dtype=torch.bool)].mean():.3f}")
    print("-" * 64)

    p1 = plot_sim_before_after(hist)
    p2 = plot_curves(hist)
    p3 = plot_retrieval_ranks(hist)
    print("已保存图像:")
    for p in (p1, p2, p3):
        print("  ", p)
    print("完成 ✅")


if __name__ == "__main__":
    main()
