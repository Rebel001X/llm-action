# -*- coding: utf-8 -*-
"""
run_demo.py —— 离线评估指标库的玩具演示
=========================================

本脚本用一份"玩具级"的分类 + 推荐结果，跑一遍指标库，输出：
  1. 终端里的**指标报告**（分类 / 排序 / 校准 / 切片）。
  2. 三张图（保存到 ./figures/）：
       - roc_pr_curves.png       ROC 曲线 + PR 曲线（左右并排）
       - reliability_diagram.png 可靠性曲线（校准图）
       - slice_bars.png          分组切片柱状图

离线可跑：不联网、不下模型、无 key。matplotlib 用 Agg 后端直接存文件。

跑法： python run_demo.py
"""

import os
import sys

# ⚠️ Windows 控制台默认编码是 GBK，直接 print emoji / 生僻字符会抛
#    UnicodeEncodeError。这里把 stdout/stderr 强制切到 UTF-8，一劳永逸。
#    （reconfigure 在 Python 3.7+ 可用；用 try 保护旧环境或已被重定向的流。）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np

# ------- matplotlib 中文与无界面配置（务必在 pyplot 之前设好后端）-------
import matplotlib

matplotlib.use("Agg")  # 用 Agg 后端：不弹窗、纯画到文件，适合服务器 / 无显示环境
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

import metrics as M

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 造玩具数据：一个"点击率预测"模型（预测用户会不会点广告）
# =============================================================================
def make_toy_classification(n=400, seed=7):
    """造一份带真实规律但有噪声的二分类数据。

    设定：正样本（会点击）分数普遍偏高，但两类有重叠（模拟真实不完美）。
    额外造一个 group 维度（设备类型），故意让 mobile 组更难分，
    好让切片分析暴露出组间差距。
    """
    rng = np.random.default_rng(seed)

    # 60% 桌面 desktop，40% 移动 mobile
    groups = rng.choice(["desktop", "mobile"], size=n, p=[0.6, 0.4])

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    for i in range(n):
        # 真实点击率：桌面用户基础点击率高一些
        base = 0.35 if groups[i] == "desktop" else 0.25
        y_true[i] = int(rng.random() < base)

        # 模型输出的概率：正样本抽偏高的 Beta，负样本抽偏低的 Beta
        # mobile 组我们让分布更"糊"（重叠更多）→ 模型更难分
        if groups[i] == "desktop":
            if y_true[i] == 1:
                y_prob[i] = rng.beta(5, 2)   # 偏高
            else:
                y_prob[i] = rng.beta(2, 5)   # 偏低
        else:  # mobile：正负分布更接近，区分度差
            if y_true[i] == 1:
                y_prob[i] = rng.beta(3, 3)
            else:
                y_prob[i] = rng.beta(2.5, 3)

    # 故意制造"过度自信"：把概率整体往两端拉，破坏校准，好让 ECE 看得见
    y_prob_miscal = np.clip(y_prob ** 0.6, 1e-4, 1 - 1e-4)
    return y_true, y_prob, y_prob_miscal, groups


def make_toy_ranking(seed=11):
    """造一份推荐排序数据：3 个 query，每个 query 一批候选 item。

    relevance ∈ {0,1,2}（不相关 / 相关 / 非常相关），score 为模型打分。
    """
    rng = np.random.default_rng(seed)
    relevances, scores = [], []
    for _ in range(3):
        m = rng.integers(6, 10)  # 每个 query 6~9 个候选
        rel = rng.integers(0, 3, size=m)
        # 让 score 与 rel 正相关但带噪声（模型大致对，但不完美）
        score = rel + rng.normal(0, 1.0, size=m)
        relevances.append(rel.tolist())
        scores.append(score.tolist())
    return relevances, scores


# =============================================================================
# 打印指标报告
# =============================================================================
def print_classification_report(y_true, y_prob):
    print("=" * 62)
    print("【1. 分类指标 Classification】")
    print("-" * 62)

    # 用 0.5 阈值二值化，算 P/R/F1
    y_pred = (y_prob >= 0.5).astype(int)
    cm = M.confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:  TP={cm.tp}  FP={cm.fp}  FN={cm.fn}  TN={cm.tn}")
    print(f"Precision(精确率)     = {cm.precision:.4f}")
    print(f"Recall   (召回率)     = {cm.recall:.4f}")
    print(f"F1                     = {cm.f1:.4f}")
    print(f"Accuracy (准确率)     = {cm.accuracy:.4f}")
    print(f"ROC-AUC                = {M.roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC (梯形)          = {M.pr_auc_score(y_true, y_prob):.4f}")
    print(f"AP (Average Precision) = {M.average_precision(y_true, y_prob):.4f}")


def print_ranking_report(relevances, scores):
    print("=" * 62)
    print("【2. 排序指标 Ranking】(k=5)")
    print("-" * 62)
    k = 5
    ndcgs = [
        M.ndcg_at_k(r, s, k) for r, s in zip(relevances, scores)
    ]
    for i, v in enumerate(ndcgs):
        print(f"  query#{i}  NDCG@{k} = {v:.4f}")
    print(f"平均 NDCG@{k}            = {np.mean(ndcgs):.4f}")
    print(f"MAP@{k}                  = {M.mean_average_precision(relevances, scores, k):.4f}")
    print(f"MRR                     = {M.mean_reciprocal_rank(relevances, scores):.4f}")


def print_calibration_report(y_true, y_prob, y_prob_miscal):
    print("=" * 62)
    print("【3. 校准指标 Calibration】")
    print("-" * 62)
    print(f"原始概率     Brier = {M.brier_score(y_true, y_prob):.4f}   "
          f"ECE = {M.expected_calibration_error(y_true, y_prob):.4f}")
    print(f"过度自信概率 Brier = {M.brier_score(y_true, y_prob_miscal):.4f}   "
          f"ECE = {M.expected_calibration_error(y_true, y_prob_miscal):.4f}")
    print("  ↑ 破坏校准后 ECE 明显变大：概率不再可信，但 AUC 可能没变。")


def print_slice_report(y_true, y_prob, groups):
    print("=" * 62)
    print("【4. 切片分析 Slice Analysis】(按设备分组)")
    print("-" * 62)

    def auc_fn(yt, ys):
        return M.roc_auc_score(yt, ys)

    def recall_fn(yt, ys):
        # 用 0.5 阈值算 recall
        yp = (np.asarray(ys) >= 0.5).astype(int)
        return M.confusion_matrix(yt, yp).recall

    report = M.slice_report(
        y_true, y_prob, groups, {"ROC-AUC": auc_fn, "Recall@0.5": recall_fn}
    )
    for metric_name, per_group in report.items():
        print(f"  {metric_name}:")
        for g, v in per_group.items():
            tag = " (整体)" if g == "__overall__" else ""
            print(f"      {g:<12s}{tag}: {v:.4f}")
    print("  ↑ 注意 mobile 组通常明显低于 desktop：平均值掩盖了长尾短板。")
    return report


# =============================================================================
# 画图
# =============================================================================
def plot_roc_pr(y_true, y_prob):
    fpr, tpr, _ = M.roc_curve(y_true, y_prob)
    precision, recall, _ = M.precision_recall_curve(y_true, y_prob)
    auc = M.roc_auc_score(y_true, y_prob)
    ap = M.average_precision(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左：ROC ----
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="随机基线 (AUC=0.5)")
    ax.set_xlabel("假正率 FPR = FP/(FP+TN)")
    ax.set_ylabel("真正率 TPR = TP/(TP+FN)")
    ax.set_title("ROC 曲线（越贴左上越好）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ---- 右：PR ----
    ax = axes[1]
    ax.plot(recall, precision, color="#d62728", lw=2, label=f"PR (AP={ap:.3f})")
    baseline = np.mean(y_true)  # 随机分类器的 PR 基线 = 正样本占比
    ax.axhline(baseline, ls="--", color="gray", lw=1,
               label=f"随机基线 (正样本率={baseline:.3f})")
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("PR 曲线（不平衡数据更该看它）")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle("ROC 与 PR 曲线对比", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "roc_pr_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_reliability(y_true, y_prob, y_prob_miscal):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="完美校准 (y=x)")

    for prob, name, color in [
        (y_prob, "原始概率", "#2ca02c"),
        (y_prob_miscal, "过度自信概率", "#ff7f0e"),
    ]:
        conf, acc, count = M.reliability_curve(y_true, prob, n_bins=10)
        ece = M.expected_calibration_error(y_true, prob, n_bins=10)
        mask = ~np.isnan(conf)  # 跳过空桶
        # 点的大小 ∝ 桶内样本数，让稀疏桶不误导
        sizes = 30 + 300 * (count[mask] / max(count[mask].max(), 1))
        ax.plot(conf[mask], acc[mask], "-o", color=color, lw=1.5,
                label=f"{name} (ECE={ece:.3f})")
        ax.scatter(conf[mask], acc[mask], s=sizes, color=color, alpha=0.4,
                   edgecolors="none")

    ax.set_xlabel("平均预测置信度 Confidence")
    ax.set_ylabel("实际准确率 Accuracy")
    ax.set_title("可靠性曲线 Reliability Diagram\n（点越偏离对角线，校准越差）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "reliability_diagram.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_slice_bars(report):
    metric_names = [m for m in report.keys()]
    # 收集所有组名（去掉 __overall__，单独放最后）
    groups = [g for g in report[metric_names[0]].keys() if g != "__overall__"]
    groups = sorted(groups) + ["__overall__"]
    labels = [g if g != "__overall__" else "整体" for g in groups]

    x = np.arange(len(groups))
    width = 0.8 / len(metric_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mname in enumerate(metric_names):
        vals = [report[mname][g] for g in groups]
        bars = ax.bar(x + i * width, vals, width, label=mname)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.1)
    ax.set_title("分组切片指标对比\n（发现 mobile 组的性能短板）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "slice_bars.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("\n玩具离线评估报告  (Offline Metrics Lab Demo)\n")

    y_true, y_prob, y_prob_miscal, groups = make_toy_classification()
    relevances, scores = make_toy_ranking()

    print_classification_report(y_true, y_prob)
    print_ranking_report(relevances, scores)
    print_calibration_report(y_true, y_prob, y_prob_miscal)
    report = print_slice_report(y_true, y_prob, groups)

    print("=" * 62)
    print("【出图 Figures】")
    p1 = plot_roc_pr(y_true, y_prob)
    p2 = plot_reliability(y_true, y_prob, y_prob_miscal)
    p3 = plot_slice_bars(report)
    print(f"  已保存: {p1}")
    print(f"  已保存: {p2}")
    print(f"  已保存: {p3}")
    print("=" * 62)
    print("完成 ✅  打开 figures/ 目录查看三张图。")


if __name__ == "__main__":
    main()
