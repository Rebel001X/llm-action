# -*- coding: utf-8 -*-
"""
run_demo.py
===========
目标检测指标演示脚本（离线、纯 numpy + matplotlib）。

产出三张图（保存到 outputs/ 目录，无需联网、无需 GPU）：
    1. nms_before_after.png  —— NMS 前后框对比（去重可视化）
    2. pr_curve.png          —— 一批预测框上的 Precision-Recall 曲线 + AP
    3. map_bars.png          —— 多类 AP / mAP 柱状图

运行： python run_demo.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器/CI 上也能出图
import matplotlib.pyplot as plt
import numpy as np

# 中文字体设置（Windows 常见字体，避免中文乱码/负号方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from detection_metrics import (
    nms,
    match_detections,
    precision_recall_curve,
    average_precision,
    compute_ap,
    mean_average_precision,
    iou_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 小工具：在坐标轴上画一批矩形框
# ---------------------------------------------------------------------
def draw_boxes(ax, boxes, scores=None, color="tab:blue", lw=2, alpha=1.0):
    """把 (N,4) 的框 [x1,y1,x2,y2] 画成矩形，可选在角上标注分数。"""
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor=color, linewidth=lw, alpha=alpha)
        ax.add_patch(rect)
        if scores is not None:
            ax.text(x1, y1 - 2, f"{scores[i]:.2f}", color=color,
                    fontsize=9, weight="bold")


# ---------------------------------------------------------------------
# 演示 1：NMS 前后对比
# ---------------------------------------------------------------------
def demo_nms():
    """构造两个物体、每个物体一堆重叠的候选框，展示 NMS 去重效果。"""
    rng = np.random.default_rng(0)

    # 物体 A 中心 (30,30)，物体 B 中心 (75,60)，各生成一簇抖动框
    def cluster(cx, cy, n, jitter=6):
        boxes, scores = [], []
        for _ in range(n):
            dx, dy = rng.uniform(-jitter, jitter, size=2)
            w, h = rng.uniform(18, 24, size=2)
            x1, y1 = cx + dx - w / 2, cy + dy - h / 2
            boxes.append([x1, y1, x1 + w, y1 + h])
            scores.append(rng.uniform(0.4, 0.99))
        return np.array(boxes), np.array(scores)

    ba, sa = cluster(30, 30, 6)
    bb, sb = cluster(75, 60, 7)
    boxes = np.vstack([ba, bb])
    scores = np.concatenate([sa, sb])

    keep = nms(boxes, scores, iou_threshold=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        ax.set_xlim(0, 110)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # 图像坐标：y 向下
        ax.grid(alpha=0.3)

    axes[0].set_title(f"NMS 前：{len(boxes)} 个候选框（重复严重）")
    draw_boxes(axes[0], boxes, scores, color="tab:red", lw=1.2, alpha=0.7)

    axes[1].set_title(f"NMS 后：保留 {len(keep)} 个框（每物体一个）")
    draw_boxes(axes[1], boxes[keep], scores[keep], color="tab:green", lw=2.2)

    fig.suptitle("非极大抑制 (Non-Maximum Suppression) 去重演示  IoU阈值=0.5",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "nms_before_after.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[demo_nms]  候选框 {len(boxes)} 个 -> NMS 后保留 {len(keep)} 个")
    print(f"[demo_nms]  图已保存: {path}")


# ---------------------------------------------------------------------
# 演示 2：PR 曲线 + AP
# ---------------------------------------------------------------------
def demo_pr_curve():
    """在一批玩具预测框上，画单类的 PR 曲线并标注两种口径的 AP。"""
    rng = np.random.default_rng(1)

    # 真值：一排 10 个不重叠的框
    gt = np.array([[i * 15, 0, i * 15 + 10, 10] for i in range(10)], dtype=float)

    # 预测：命中一部分真值（略微抖动），再加一些误检（空地上的框）
    preds, scores = [], []
    # 命中 7 个真值（带小抖动 + 高分）
    for i in [0, 1, 2, 3, 5, 6, 8]:
        j = rng.uniform(-1.5, 1.5, size=2)
        b = gt[i].copy()
        b[[0, 2]] += j[0]
        b[[1, 3]] += j[1]
        preds.append(b)
        scores.append(rng.uniform(0.6, 0.98))
    # 4 个误检（远离所有真值的空地）
    for _ in range(4):
        x = rng.uniform(0, 140)
        preds.append([x, 40, x + 10, 50])
        scores.append(rng.uniform(0.3, 0.8))
    preds = np.array(preds)
    scores = np.array(scores)

    tp, fp, sc, n_gt = match_detections(preds, scores, gt, iou_threshold=0.5)
    precision, recall = precision_recall_curve(tp, fp, n_gt)
    ap_all = average_precision(precision, recall, "all_points")
    ap_11 = average_precision(precision, recall, "11_points")

    fig, ax = plt.subplots(figsize=(7, 6))
    # 原始锯齿 PR 曲线
    ax.plot(recall, precision, "o-", color="tab:blue", label="原始 PR 曲线", ms=5)

    # 画 all_points 的单调包络（阶梯）
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ax.step(mrec, mpre, where="post", color="tab:orange",
            linestyle="--", label="单调包络 (all-points)")
    ax.fill_between(mrec, mpre, step="post", alpha=0.15, color="tab:orange")

    ax.set_xlabel("Recall 查全率")
    ax.set_ylabel("Precision 查准率")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(f"Precision-Recall 曲线\nAP(all-points)={ap_all:.3f}   "
                 f"AP(11-point)={ap_11:.3f}", fontsize=12, weight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pr_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[demo_pr]   n_gt={n_gt}  预测={len(preds)}  "
          f"AP(all)={ap_all:.3f}  AP(11)={ap_11:.3f}")
    print(f"[demo_pr]   图已保存: {path}")


# ---------------------------------------------------------------------
# 演示 3：多类 mAP 柱状图
# ---------------------------------------------------------------------
def demo_map():
    """构造 3 个类别、不同检测质量，算 per-class AP 与 mAP，画柱状图。"""
    rng = np.random.default_rng(2)

    detections, gts = {}, {}
    class_names = {0: "person", 1: "car", 2: "dog"}

    for c in range(3):
        n_gt = 8
        gt = np.array([[i * 20, c * 30, i * 20 + 12, c * 30 + 12]
                       for i in range(n_gt)], dtype=float)
        gts[c] = gt

        # 不同类别设定不同的“命中率”，让 AP 有区分度
        hit_ratio = [0.9, 0.65, 0.4][c]
        n_hit = int(n_gt * hit_ratio)
        preds, scores = [], []
        for i in range(n_hit):                       # 命中的框（带抖动、高分）
            b = gt[i].copy() + rng.uniform(-1, 1, size=4)
            preds.append(b)
            scores.append(rng.uniform(0.6, 0.98))
        for _ in range(4):                            # 误检（空地、分数偏低）
            x = rng.uniform(0, 160)
            preds.append([x, c * 30 + 60, x + 12, c * 30 + 72])
            scores.append(rng.uniform(0.3, 0.7))
        detections[c] = (np.array(preds), np.array(scores))

    mAP, per = mean_average_precision(detections, gts, 0.5, "all_points")

    fig, ax = plt.subplots(figsize=(7, 5))
    names = [class_names[c] for c in sorted(per.keys())]
    vals = [per[c] for c in sorted(per.keys())]
    bars = ax.bar(names, vals, color=["tab:blue", "tab:green", "tab:red"], alpha=0.8)
    ax.axhline(mAP, color="black", linestyle="--", label=f"mAP = {mAP:.3f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AP (IoU=0.5, all-points)")
    ax.set_title("多类别 AP 与 mAP", fontsize=13, weight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "map_bars.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[demo_map]  per-class AP = "
          + ", ".join(f"{class_names[c]}={per[c]:.3f}" for c in sorted(per)))
    print(f"[demo_map]  mAP = {mAP:.3f}")
    print(f"[demo_map]  图已保存: {path}")


def main():
    print("=" * 60)
    print("目标检测指标实验室 —— 演示开始 (纯 numpy, 离线出图)")
    print("=" * 60)
    demo_nms()
    print("-" * 60)
    demo_pr_curve()
    print("-" * 60)
    demo_map()
    print("=" * 60)
    print(f"全部完成。三张图在: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
