# -*- coding: utf-8 -*-
"""
run_demo.py —— LLM-as-a-Judge 评估器一站式演示。

跑一遍就能得到:
    1) pointwise:裁判打分 vs 人类金标准的「一致性报告」
       (Cohen's kappa / Pearson / Spearman / raw agreement / 分歧明细)—— book 7.8
    2) pairwise:交换顺序检测「位置偏置」,对比干净裁判 vs 带偏置裁判 —— book 7.7.1
    3) 出两张图:
       - fig_consistency.png  裁判 vs 金标准 散点 + 分布(校准一致性可视化)
       - fig_position_bias.png 干净 vs 带偏置裁判的位置一致率对比(位置偏置可视化)

离线纯本地运行:python run_demo.py
"""

from __future__ import annotations

import os
import sys

# Windows 控制台默认 GBK,中文输出会乱码;强制 stdout 用 UTF-8(踩坑点)。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ---- matplotlib 必须在 import pyplot 之前设 Agg 后端(无显示环境/CI 友好)----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体 + 负号正常显示(Windows 常用雅黑/黑体)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from dataset import build_pairwise_dataset, build_pointwise_dataset  # noqa: E402
from judge import GoldOracleJudge, MockLLMJudge  # noqa: E402
from harness import (  # noqa: E402
    run_pairwise_validation,
    run_pairwise_with_swap,
    run_pointwise_validation,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# Part 1: pointwise —— 裁判 vs 金标准的一致性报告(book 7.8)
# ---------------------------------------------------------------------------

def demo_pointwise():
    _hr("Part 1 | Pointwise 打分:裁判 vs 人类金标准 的一致性报告 (book 7.8)")
    samples, gold_points = build_pointwise_dataset()
    judge = MockLLMJudge(position_bias=0.0, honor_reference=True)
    gold = GoldOracleJudge(gold_points, {})
    rep = run_pointwise_validation(judge, gold, samples, dimension="helpfulness")

    print("\n每条样本的裁判分 vs 金标准分:")
    print(f"  {'sample':<8}{'judge':>7}{'gold':>7}   note")
    for s, j, g in zip(samples, rep.judge_scores, rep.gold_scores):
        flag = "  <-- 分歧" if j != g else ""
        print(f"  {s.sample_id:<8}{j:>7}{g:>7}{flag}")

    print("\n一致性指标(book 7.8.2):")
    print(f"  raw agreement(原始一致率)  = {rep.raw_agree:.3f}   "
          f"<- 别只看这个,会高估")
    print(f"  Cohen's kappa(扣运气净一致) = {rep.kappa.kappa:.3f}  "
          f"[{rep.kappa.interpretation}]")
    print(f"    (p_o={rep.kappa.p_observed:.3f}, p_e={rep.kappa.p_expected:.3f})")
    print(f"  Pearson  相关(连续趋势对齐) = {rep.pearson:.3f}")
    print(f"  Spearman 秩相关            = {rep.spearman:.3f}")

    print("\n聚合(book 7.5 Scoring 看趋势):")
    print(f"  裁判  mean={rep.judge_agg.mean:.2f} std={rep.judge_agg.std:.2f} "
          f"dist={rep.judge_agg.distribution}")
    print(f"  金标准 mean={rep.gold_agg.mean:.2f} std={rep.gold_agg.std:.2f} "
          f"dist={rep.gold_agg.distribution}")

    print(f"\n分歧分析(book 7.8.1 第 5 步,最有价值):共 {len(rep.disagreements)} 处")
    for d in rep.disagreements:
        print(f"  [{d['sample_id']}] judge={d['judge_score']} gold={d['gold_score']}"
              f"  | {d['judge_justification']}")

    return samples, rep


# ---------------------------------------------------------------------------
# Part 2: pairwise —— 位置偏置检测(book 7.7.1)
# ---------------------------------------------------------------------------

def demo_position_bias():
    _hr("Part 2 | Pairwise 位置偏置:交换顺序检测 (book 7.7.1)")
    samples, gold_pairs = build_pairwise_dataset()

    clean = MockLLMJudge(position_bias=0.0)
    biased = MockLLMJudge(position_bias=0.6)

    clean_rep = run_pairwise_with_swap(clean, samples, "helpfulness")
    biased_rep = run_pairwise_with_swap(biased, samples, "helpfulness")

    print("\n干净裁判(position_bias=0.0):")
    print(f"  原始顺序胜者: {clean_rep.original_winners}")
    print(f"  交换顺序胜者: {clean_rep.swapped_winners}")
    print(f"  位置一致率 = {clean_rep.position_bias.consistency_rate:.3f}  "
          f"(翻转率 {clean_rep.position_bias.flip_rate:.3f}, "
          f"偏爱首位率 {clean_rep.position_bias.prefers_first_rate:.3f})")

    print("\n带位置偏置裁判(position_bias=0.6):")
    print(f"  原始顺序胜者: {biased_rep.original_winners}")
    print(f"  交换顺序胜者: {biased_rep.swapped_winners}")
    print(f"  位置一致率 = {biased_rep.position_bias.consistency_rate:.3f}  "
          f"(翻转率 {biased_rep.position_bias.flip_rate:.3f}, "
          f"偏爱首位率 {biased_rep.position_bias.prefers_first_rate:.3f})")

    print("\n解读:干净裁判换序后胜者干净翻转 -> 一致率高;")
    print("      带偏置裁判总偏爱『第一个位置』-> 换序后自相矛盾 -> 一致率骤降。")

    # 稳健裁决(洗掉位置偏置)后再与金标准验证 —— book 7.7.1 + 7.8
    _hr("Part 3 | 洗掉位置偏置后再验证:稳健胜者 vs 金标准 (book 7.7.1 + 7.8)")
    gold = GoldOracleJudge({}, gold_pairs)
    val = run_pairwise_validation(clean, gold, samples, "helpfulness")
    print(f"\n  稳健胜者(正反一致才算赢): {val.robust_winners}")
    print(f"  人类金标准胜者          : {val.gold_winners}")
    print(f"  win-rate agreement = {val.win_rate_agree:.3f}")
    print(f"  Cohen's kappa      = {val.kappa.kappa:.3f} [{val.kappa.interpretation}]")

    return samples, clean_rep, biased_rep


# ---------------------------------------------------------------------------
# 出图 1:一致性(裁判分 vs 金标准分)
# ---------------------------------------------------------------------------

def plot_consistency(samples, rep, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左:散点 —— 裁判分 vs 金标准分(点越贴近对角线越一致)
    ax = axes[0]
    jx = rep.gold_scores
    jy = rep.judge_scores
    ax.scatter(jx, jy, s=120, c="#2b8cbe", edgecolors="white", zorder=3)
    for s, x, y in zip(samples, jx, jy):
        ax.annotate(s.sample_id, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.plot([1, 5], [1, 5], "--", color="gray", label="完全一致(对角线)")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("人类金标准分")
    ax.set_ylabel("LLM 裁判分")
    ax.set_title(f"裁判 vs 金标准 打分散点\n"
                 f"kappa={rep.kappa.kappa:.2f}  pearson={rep.pearson:.2f}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 右:打分分布对比(裁判 vs 金标准),直观看校准偏移
    ax = axes[1]
    buckets = [1, 2, 3, 4, 5]
    jd = [rep.judge_agg.distribution.get(b, 0) for b in buckets]
    gd = [rep.gold_agg.distribution.get(b, 0) for b in buckets]
    width = 0.38
    x = range(len(buckets))
    ax.bar([i - width / 2 for i in x], jd, width, label="LLM 裁判",
           color="#2b8cbe")
    ax.bar([i + width / 2 for i in x], gd, width, label="人类金标准",
           color="#e6550d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets)
    ax.set_xlabel("分值 (1~5)")
    ax.set_ylabel("样本数")
    ax.set_title("打分分布对比(校准偏移可视化)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("LLM-as-a-Judge 校准一致性报告 (book 7.8)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n[图已保存] {out_path}")


# ---------------------------------------------------------------------------
# 出图 2:位置偏置(干净 vs 带偏置裁判)
# ---------------------------------------------------------------------------

def plot_position_bias(clean_rep, biased_rep, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左:位置一致率对比条形图
    ax = axes[0]
    labels = ["干净裁判\n(bias=0.0)", "带偏置裁判\n(bias=0.6)"]
    cons = [clean_rep.position_bias.consistency_rate,
            biased_rep.position_bias.consistency_rate]
    bars = ax.bar(labels, cons, color=["#31a354", "#de2d26"], width=0.55)
    ax.axhline(1.0, ls="--", color="gray", label="理想一致率=1.0")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("位置一致率 (C / N)")
    ax.set_title("交换顺序后的位置一致率\n越低=位置偏置越严重 (book 7.7.1)")
    for b, v in zip(bars, cons):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 右:偏爱首位率对比(0.5 为无偏基线)
    ax = axes[1]
    pf = [clean_rep.position_bias.prefers_first_rate,
          biased_rep.position_bias.prefers_first_rate]
    bars = ax.bar(labels, pf, color=["#31a354", "#de2d26"], width=0.55)
    ax.axhline(0.5, ls="--", color="gray", label="无偏基线=0.5")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("偏爱『第一个位置』的比率")
    ax.set_title("偏爱首位率\n>0.5 说明系统性偏向第一个回答")
    for b, v in zip(bars, pf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("位置偏置检测:交换顺序一致性 (book 7.7.1)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[图已保存] {out_path}")


def main():
    samples_p, rep_p = demo_pointwise()
    samples_q, clean_rep, biased_rep = demo_position_bias()

    fig1 = os.path.join(HERE, "fig_consistency.png")
    fig2 = os.path.join(HERE, "fig_position_bias.png")
    plot_consistency(samples_p, rep_p, fig1)
    plot_position_bias(clean_rep, biased_rep, fig2)

    _hr("完成")
    print("两张图已生成,可用于报告/PR:")
    print(f"  - {fig1}")
    print(f"  - {fig2}")
    print("\n一句话记住(book 题眼):LLM 裁判是评估仪器,不是真理机器;")
    print("先校准、再验证、显式测位置偏置,它才值得信任。")


if __name__ == "__main__":
    main()
