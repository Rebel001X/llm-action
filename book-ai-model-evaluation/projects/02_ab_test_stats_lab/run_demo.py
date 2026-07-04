# -*- coding: utf-8 -*-
"""
run_demo.py —— A/B 测试统计实验室 · 可视化演示

跑一遍就会：
    1. 打印一个完整的 A/B 决策例子（两比例检验：p 值、CI、显著性）
    2. 打印样本量估算与功效表
    3. 画三张图（保存到 figures/ 目录）：
        - power_curve.png       功效曲线（power 随每组样本量上升）
        - peeking_fpr.png       偷看假阳性率膨胀图
        - ci_coverage.png       CI 覆盖率蒙特卡洛验证
本机无 GUI/无网络，用 Agg 后端 + 微软雅黑字体确保中文正常、离线可跑。

运行：python run_demo.py
"""
import os
import sys

# Windows 控制台默认 GBK，打印 emoji/部分中文会报 UnicodeEncodeError。
# 把标准输出重设为 UTF-8（Python 3.7+ 支持 reconfigure），确保离线打印不崩。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")  # 无界面后端，只出文件不弹窗（服务器/CI 友好）
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 常见），负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import ab_stats as ab

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def section(title: str):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# -----------------------------------------------------------------------------
# 演示 1：一个完整的 A/B 决策
# -----------------------------------------------------------------------------
def demo_decision():
    section("演示 1 · 一个完整的 A/B 决策（两比例检验）")
    # 场景：推荐模型 A（对照）CTR 基线 10%，模型 B（实验）看起来更高
    x_a, n_a = 1023, 10000   # A 组：1 万曝光，1023 次点击 → 10.23%
    x_b, n_b = 1140, 10000   # B 组：1 万曝光，1140 次点击 → 11.40%
    r = ab.two_proportion_ztest(x_a, n_a, x_b, n_b, alpha=0.05,
                                alternative="two-sided")
    print(f"A 组 CTR = {x_a/n_a:.4%}  ({x_a}/{n_a})")
    print(f"B 组 CTR = {x_b/n_b:.4%}  ({x_b}/{n_b})")
    print(f"绝对提升 effect = {r.effect:+.4%}")
    print(f"z 统计量        = {r.statistic:.4f}")
    print(f"p 值            = {r.p_value:.5f}")
    print(f"95% 置信区间    = [{r.ci_low:+.4%}, {r.ci_high:+.4%}]")
    print(f"是否显著(α=.05) = {'是 ✅ 倾向上线 B' if r.significant else '否 ❌ 证据不足'}")
    print("解读：CI 不含 0 → 显著；但最差情况仅 "
          f"{r.ci_low:+.2%}，是否上线还要看这个提升值不值回工程成本。")


# -----------------------------------------------------------------------------
# 演示 2：样本量与功效表
# -----------------------------------------------------------------------------
def demo_sample_size():
    section("演示 2 · 开测前算样本量（power analysis）")
    p0 = 0.10
    print(f"基线转化率 p0 = {p0:.0%}，目标功效 power=0.80，α=0.05（双尾）")
    print(f"{'MDE(绝对)':>12} | {'相对提升':>8} | {'每组样本量':>12} | {'双组合计':>12}")
    print("-" * 54)
    for mde in [0.005, 0.01, 0.02, 0.03, 0.05]:
        n = ab.sample_size_two_proportions(p0, mde, alpha=0.05, power=0.80)
        print(f"{mde:>12.1%} | {mde/p0:>7.0%} | {n:>12,} | {2*n:>12,}")
    print("\n⚠️ 注意：MDE 减半，样本量约 ×4。想检出越小的效应，代价越贵。")


# -----------------------------------------------------------------------------
# 演示 3：功效曲线
# -----------------------------------------------------------------------------
def demo_power_curve():
    section("演示 3 · 画功效曲线 power_curve.png")
    ns = np.linspace(200, 12000, 60).astype(int)
    p0 = 0.10
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mde, color in [(0.01, "#d62728"), (0.02, "#1f77b4"), (0.03, "#2ca02c")]:
        powers = [ab.power_two_proportions(p0, mde, n, alpha=0.05) for n in ns]
        ax.plot(ns, powers, color=color, lw=2,
                label=f"MDE = {mde:.0%}（{p0:.0%}→{p0+mde:.0%}）")
        # 标出达到 0.80 功效所需样本量
        n_need = ab.sample_size_two_proportions(p0, mde, power=0.80)
        if n_need <= ns.max():
            ax.plot([n_need], [0.80], "o", color=color, ms=7)
            ax.annotate(f"n≈{n_need:,}", (n_need, 0.80),
                        textcoords="offset points", xytext=(6, -14),
                        fontsize=9, color=color)
    ax.axhline(0.80, ls="--", color="gray", lw=1)
    ax.text(ns.max(), 0.805, "行业惯例 power=0.80", ha="right",
            va="bottom", color="gray", fontsize=9)
    ax.set_xlabel("每组样本量 n")
    ax.set_ylabel("统计功效 power (1−β)")
    ax.set_title("功效曲线：样本量越大 / 要检的效应越大 → 功效越高", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "power_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("已保存:", out)


# -----------------------------------------------------------------------------
# 演示 4：偷看假阳性率膨胀图（本项目主菜）
# -----------------------------------------------------------------------------
def demo_peeking():
    section("演示 4 · 画偷看假阳性率膨胀图 peeking_fpr.png")
    peek_counts = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    print("跑蒙特卡洛（A/A 测试，H0 为真，任何显著都是假阳性）...")
    counts, fprs = ab.peeking_fpr_curve(
        peek_counts=peek_counts, n_max=3000, alpha=0.05,
        n_trials=2000, p=0.10, seed=0)
    for k, f in zip(counts, fprs):
        bar = "█" * int(f * 100)
        print(f"  偷看 {k:>2} 次 → 实际假阳性率 {f:.1%}  {bar}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(counts, fprs, "o-", color="#d62728", lw=2, ms=6,
            label="偷看策略实际假阳性率")
    ax.axhline(0.05, ls="--", color="#1f77b4", lw=1.5,
               label="名义显著性水平 α = 5%")
    ax.fill_between(counts, 0.05, fprs, where=[f > 0.05 for f in fprs],
                    color="#d62728", alpha=0.12, label="被偷看“偷走”的额外假阳性")
    ax.set_xlabel("偷看次数（实验期间查看 p 值并可提前停止的次数）")
    ax.set_ylabel("实际假阳性率 (Type I error)")
    ax.set_title("偷看陷阱：反复查看 p 值把 5% 的假阳性率抬到 20%+", fontsize=12)
    ax.set_ylim(0, max(fprs) * 1.15)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "peeking_fpr.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("已保存:", out)


# -----------------------------------------------------------------------------
# 演示 5：CI 覆盖率蒙特卡洛
# -----------------------------------------------------------------------------
def demo_ci_coverage():
    section("演示 5 · 画 CI 覆盖率验证图 ci_coverage.png")
    confidences = [0.80, 0.90, 0.95, 0.99]
    covs = []
    for c in confidences:
        cov = ab.ci_coverage_two_proportions(
            0.10, 0.12, n=3000, alpha=1 - c, n_trials=3000, seed=0)
        covs.append(cov)
        print(f"  名义 {c:.0%} CI → 实测覆盖率 {cov:.1%}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(confidences))
    ax.bar(x - 0.18, confidences, width=0.36, color="#1f77b4",
           label="名义置信水平（理论）")
    ax.bar(x + 0.18, covs, width=0.36, color="#2ca02c",
           label="蒙特卡洛实测覆盖率")
    for xi, (c, cov) in enumerate(zip(confidences, covs)):
        ax.text(xi + 0.18, cov + 0.005, f"{cov:.1%}", ha="center",
                fontsize=9, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0%}" for c in confidences])
    ax.set_xlabel("置信水平 (1−α)")
    ax.set_ylabel("覆盖率")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("置信区间覆盖率验证：实测 ≈ 名义，说明 CI 实现正确", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, "ci_coverage.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("已保存:", out)


def main():
    print("A/B 测试统计实验室 · 演示开始（纯 numpy，离线可跑）")
    demo_decision()
    demo_sample_size()
    demo_power_curve()
    demo_peeking()
    demo_ci_coverage()
    section("全部完成")
    print("三张图已保存到:", FIG_DIR)
    print("  - power_curve.png   功效曲线")
    print("  - peeking_fpr.png   偷看假阳性率膨胀（本项目主菜）")
    print("  - ci_coverage.png   CI 覆盖率验证")


if __name__ == "__main__":
    main()
