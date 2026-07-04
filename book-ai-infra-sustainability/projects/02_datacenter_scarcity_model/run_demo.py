# -*- coding: utf-8 -*-
"""
稀缺性模型可视化 Demo
=====================

跑法:
    python run_demo.py

产出(全部写在本项目文件夹,不联网、不需 GPU、不需 key):
    - out/cumulative_vs_budget.png   多情景"累计碳排 vs 碳预算"曲线
    - out/factor_decomposition.png   Kaya 三因子分解(算力↑ / 强度↓ / 碳强度↓)
    - out/remaining_budget.png       剩余碳预算随年份(穿过 0 即赤字)
    - 控制台打印每个情景的稀缺性结论

matplotlib 使用 Agg 后端(无窗口、纯出图),中文用微软雅黑。
"""

import os
import sys

import numpy as np
import matplotlib

# ⚠️ 必须在 import pyplot 之前设定后端;Agg = 纯文件渲染,无需显示环境
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体 + 正常显示负号(Windows 常见坑)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import scarcity_model as sm

# 控制台 UTF-8(Windows GBK 坑)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def plot_cumulative_vs_budget(scenarios):
    """图 1:多情景累计碳排曲线 + 碳预算水平线(核心图)。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    budget = scenarios[0].carbon_budget_gt  # 假设各情景共享同一预算

    for scn in scenarios:
        res = sm.run_scenario(scn)
        years = np.arange(scn.years)
        line, = ax.plot(years, res.cumulative, linewidth=2.2,
                        label=f"{scn.name}(压力 {sm.scarcity_pressure(scn):.2f})")
        # 在超预算的那一年打一个醒目标记
        if res.exceeded_year is not None:
            yx = res.exceeded_year
            ax.scatter([yx], [res.cumulative[yx]], s=80, zorder=5,
                       color=line.get_color(), edgecolors="black")
            ax.annotate(f"第{yx}年超预算", (yx, res.cumulative[yx]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=9, color=line.get_color())

    # 碳预算 = 那面"墙"
    ax.axhline(budget, color="red", linestyle="--", linewidth=2,
               label=f"剩余碳预算 = {budget:.1f} GtCO2")
    ax.fill_between(range(scenarios[0].years), budget,
                    ax.get_ylim()[1], color="red", alpha=0.05)

    ax.set_title("数据中心 AI 累计碳排 vs 碳预算(稀缺性视角)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("年份(相对第 0 年)")
    ax.set_ylabel("累计碳排 (GtCO2)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "cumulative_vs_budget.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_factor_decomposition(scn):
    """图 2:Kaya 三因子分解,直观看到"增长 vs 抵消"的拔河。"""
    years = np.arange(scn.years)
    comp = sm.compute_curve(scn.years, scn.compute_growth)
    eff = sm.efficiency_curve(scn.years, scn.efficiency_gain)
    carb = sm.carbon_intensity_curve(scn.years, scn.renewable_start,
                                     scn.renewable_end, scn.grid_carbon_fossil)
    ann = sm.annual_emissions(scn)
    ann_rel = ann / ann[0] if ann[0] != 0 else ann

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, comp, linewidth=2, label="算力需求(↑ 推高碳排)")
    ax.plot(years, eff, linewidth=2, label="能耗强度(↓ 能效改进抵消)")
    # 碳强度归一到第 0 年方便同图比较
    carb_rel = carb / carb[0] if carb[0] != 0 else carb
    ax.plot(years, carb_rel, linewidth=2, label="碳强度(↓ 可再生降碳)")
    ax.plot(years, ann_rel, linewidth=2.8, linestyle="--", color="black",
            label="年碳排(三者相乘,相对第0年)")

    ax.set_title(f"Kaya 三因子分解 —— {scn.name}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("年份(相对第 0 年)")
    ax.set_ylabel("相对值(第 0 年 = 1.0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "factor_decomposition.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_remaining_budget(scenarios):
    """图 3:剩余碳预算随年份;穿过 0 那一刻进入"碳赤字"。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for scn in scenarios:
        rem = sm.remaining_budget_curve(scn)
        ax.plot(np.arange(scn.years), rem, linewidth=2.2, label=scn.name)

    ax.axhline(0, color="red", linestyle="--", linewidth=2,
               label="预算耗尽线(下方 = 碳赤字)")
    ax.set_title("剩余碳预算随年份(稀缺资源的消耗)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("年份(相对第 0 年)")
    ax.set_ylabel("剩余碳预算 (GtCO2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "remaining_budget.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    scenarios = sm.default_scenarios()

    print("=" * 70)
    print("数据中心算力增长 vs 碳预算 —— 稀缺性模型 Demo")
    print("=" * 70)
    for scn in scenarios:
        res = sm.run_scenario(scn)
        verdict = ("✅ 预算内" if res.within_budget
                   else f"❌ 第 {res.exceeded_year} 年超预算")
        print(f"[{scn.name:<26}] 累计={res.total_emission_gt:6.2f} Gt / "
              f"预算={res.budget_gt:.1f} Gt | 压力={sm.scarcity_pressure(scn):5.2f} | {verdict}")

    print("-" * 70)
    p1 = plot_cumulative_vs_budget(scenarios)
    p2 = plot_factor_decomposition(scenarios[0])  # 拿基准情景做因子分解
    p3 = plot_remaining_budget(scenarios)
    print("已生成图片:")
    for p in (p1, p2, p3):
        print("  -", p)
    print("=" * 70)
    print("完成。用图片查看器打开 out/ 下的 PNG 即可。")


if __name__ == "__main__":
    main()
