# -*- coding: utf-8 -*-
"""
run_demo.py —— AI 能耗-碳排-水耗计算器 可视化演示
=================================================

离线运行,产出 3 张图(保存到 ./figures/):
    1. 不同模型规模的训练碳排柱状图
    2. 同一训练在不同电网(可再生占比不同)下的碳排对比
    3. PUE 敏感性曲线(PUE 从 1.0 → 2.0,碳排如何上升)

以及一张推理摊薄的表(打印到终端)。

⚠️ 用 matplotlib 的 Agg 后端(无窗口、纯写文件),中文字体用微软雅黑/黑体,
   保证在无显示器、无 GUI 的机器上也能出图。
"""

import os
import sys

# ⚠️ Windows 坑:默认控制台是 GBK 编码,打印 emoji / 生僻字会 UnicodeEncodeError。
# 把标准输出重配为 UTF-8(Python 3.7+ 支持),保证跨平台打印中文与 emoji 不崩。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")  # 关键:无界面后端,只写文件不弹窗
import matplotlib.pyplot as plt

# 中文显示:优先微软雅黑,退化到黑体;负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import carbon_calculator as cc


FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 图 1:不同模型规模的训练碳排
# ---------------------------------------------------------------------------

def plot_model_scale_carbon():
    """不同规模模型(用 GPU 数 × 时长近似算力)训练碳排柱状图。"""
    # (名称, 参数量B, GPU数, 时长h, 单卡功率W)—— 教学近似,量级贴近真实
    models = [
        ("1.5B\n(小)", 1.5, 64, 200, 400),
        ("7B\n(中)", 7, 512, 400, 400),
        ("13B\n(大)", 13, 1024, 500, 700),
        ("70B\n(超大)", 70, 2048, 720, 700),
        ("175B\n(GPT-3 级)", 175, 4096, 800, 700),
    ]
    names, carbons = [], []
    for name, params, gpus, hours, watt in models:
        cfg = cc.TrainingConfig(name=name, num_params_billion=params,
                                num_gpus=gpus, train_hours=hours,
                                gpu_power_watts=watt, pue=1.4,
                                grid_gco2_per_kwh=475.0)
        r = cc.compute_training_footprint(cfg)
        names.append(name)
        carbons.append(r.carbon_tco2)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(names, carbons, color="#c44e52", edgecolor="black", alpha=0.85)
    ax.set_ylabel("训练碳排 (tCO2e)")
    ax.set_title("不同规模模型的训练碳足迹(全球平均电网,PUE=1.4)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, c in zip(bars, carbons):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "01_model_scale_carbon.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 2:不同电网(可再生占比)下的碳排
# ---------------------------------------------------------------------------

def plot_grid_comparison():
    """固定一个 13B 训练任务,对比不同电网碳排 + 可再生占比曲线。"""
    cfg = cc.TrainingConfig(name="13B 训练", num_params_billion=13,
                            num_gpus=1024, train_hours=500,
                            gpu_power_watts=700, pue=1.4)
    results = cc.compare_grids(cfg)
    names = [r.label.split("@")[-1].strip() for r in results]
    carbons = [r.carbon_tco2 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左:各电网碳排(已按碳排升序,清洁在左)
    colors = plt.cm.RdYlGn_r([i / max(1, len(carbons) - 1)
                              for i in range(len(carbons))])
    ax1.barh(names, carbons, color=colors, edgecolor="black")
    ax1.set_xlabel("训练碳排 (tCO2e)")
    ax1.set_title("同一 13B 训练在不同电网下的碳足迹\n(能耗相同,脏电网碳排高数十倍)")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for i, c in enumerate(carbons):
        ax1.text(c, i, f" {c:,.0f}", va="center", fontsize=8)

    # 右:可再生占比 → 碳排(单调下降)
    shares = [i / 20 for i in range(21)]  # 0% ~ 100%
    share_carbons = []
    for s in shares:
        intensity = cc.renewable_share_to_intensity(s)
        v = cc.TrainingConfig(num_gpus=1024, train_hours=500,
                              gpu_power_watts=700, pue=1.4,
                              grid_gco2_per_kwh=intensity)
        share_carbons.append(cc.compute_training_footprint(v).carbon_tco2)
    ax2.plot([s * 100 for s in shares], share_carbons,
             marker="o", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("电网可再生能源占比 (%)")
    ax2.set_ylabel("训练碳排 (tCO2e)")
    ax2.set_title("可再生占比越高 → 碳排越低(线性下降)")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "02_grid_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 3:PUE 敏感性
# ---------------------------------------------------------------------------

def plot_pue_sensitivity():
    """PUE 从 1.0 扫到 2.0,看设施能耗 / 碳排如何随之上升。"""
    pues = [1.0 + 0.05 * i for i in range(21)]  # 1.00 ~ 2.00
    energies, carbons = [], []
    for p in pues:
        cfg = cc.TrainingConfig(num_gpus=1024, train_hours=500,
                                gpu_power_watts=700, pue=p,
                                grid_gco2_per_kwh=475.0)
        r = cc.compute_training_footprint(cfg)
        energies.append(r.facility_energy_kwh / 1000.0)  # MWh
        carbons.append(r.carbon_tco2)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(pues, energies, marker="s", color="#1f77b4",
             linewidth=2, label="设施能耗 (MWh)")
    ax1.set_xlabel("PUE (设施总电 / IT 电)")
    ax1.set_ylabel("设施能耗 (MWh)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(pues, carbons, marker="^", color="#c44e52",
             linewidth=2, label="碳排 (tCO2e)")
    ax2.set_ylabel("碳排 (tCO2e)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")

    # 标注 PUE=1.0(理想)与 PUE=1.5(典型)的对比
    ax1.axvline(1.5, color="gray", linestyle=":", alpha=0.7)
    ax1.text(1.5, min(energies), " 典型 1.5", color="gray", fontsize=9)

    ax1.set_title("PUE 敏感性:每 0.1 的 PUE 都是白烧的电与碳")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "03_pue_sensitivity.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 推理摊薄示例(终端打印)
# ---------------------------------------------------------------------------

def print_inference_table():
    print("\n" + "=" * 66)
    print("推理摊薄示例:不同电网下,处理 10 亿 token 的账单与每千 token 强度")
    print("=" * 66)
    grids = {
        "煤电为主": cc.GRID_CARBON_INTENSITY["煤电为主 (Coal-heavy)"],
        "全球平均": cc.GRID_CARBON_INTENSITY["全球平均 (Global avg)"],
        "北欧水电": cc.GRID_CARBON_INTENSITY["北欧 (Nordic, 水电为主)"],
    }
    total = 1_000_000_000  # 10 亿 token
    header = f"{'电网':<10}{'总碳排(tCO2e)':>16}{'gCO2/千token':>16}{'mL水/千token':>16}"
    print(header)
    print("-" * len(header))
    for name, gco2 in grids.items():
        cfg = cc.InferenceConfig(name=name, num_gpus=8, gpu_power_watts=700,
                                 pue=1.4, throughput_tokens_per_sec=5000,
                                 grid_gco2_per_kwh=gco2)
        r = cc.compute_inference_footprint(cfg, total)
        print(f"{name:<10}{r.carbon_tco2:>16.3f}"
              f"{r.extra['gco2_per_1k_tokens']:>16.4f}"
              f"{r.extra['ml_water_per_1k_tokens']:>16.4f}")
    print("=" * 66)


def main():
    print("生成图表中(Agg 后端,输出到 figures/)...")
    p1 = plot_model_scale_carbon()
    p2 = plot_grid_comparison()
    p3 = plot_pue_sensitivity()
    print("已保存:")
    for p in (p1, p2, p3):
        print("   -", p)
    print_inference_table()
    print("\n完成 ✅ 三张图已生成,推理摊薄表已打印。")


if __name__ == "__main__":
    main()
