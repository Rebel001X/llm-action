# -*- coding: utf-8 -*-
"""
run_demo.py
===========
优化技术收益计算器 —— 可视化 Demo。

跑一遍：
    python run_demo.py

会做三件事：
  1) 在终端打印「baseline → +量化 → +连续批 → +KV量化」的叠加收益表；
  2) 出一张**吞吐瀑布图**（waterfall）：直观看到每加一层优化，吞吐涨多少；
  3) 出一张**显存分解 + batch 天花板图**：看量化/KV量化如何抬高显存 batch 上限。

图片保存为 PNG（用 Agg 后端，无需显示器 / 无需 GUI，纯离线）。
"""

import os

import matplotlib

# ⚠️ 必须在 import pyplot 之前设置 Agg 后端：无显示器 / 无 GUI 也能出图，纯离线渲染到文件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体（Windows 常见的微软雅黑 / 黑体），并修复负号显示为方块的问题。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from optimization_calculator import (  # noqa: E402
    OptimizationCalculator,
    OptimizationConfig,
    default_setup,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def build_stages():
    """定义瀑布的四个阶段：从裸 baseline 逐级叠加三大优化。"""
    return [
        ("Baseline\nFP16, batch=1", OptimizationConfig(weight_dtype="fp16", batch_size=1, kv_dtype="fp16")),
        ("+ 量化\nW4A16 (int4)", OptimizationConfig(weight_dtype="int4", batch_size=1, kv_dtype="fp16")),
        ("+ 连续批\nbatch=32", OptimizationConfig(weight_dtype="int4", batch_size=32, kv_dtype="fp16")),
        ("+ KV量化\nKV int8, batch=64", OptimizationConfig(weight_dtype="int4", batch_size=64, kv_dtype="int8")),
    ]


def print_table(rows):
    """终端打印叠加收益表。"""
    print("\n" + "=" * 92)
    print("优化叠加收益表（Llama-7B on A100-80G，prompt=1024 / gen=512）")
    print("=" * 92)
    header = f"{'阶段':<22}{'权重GB':>9}{'KV GB':>9}{'总GB':>9}{'吞吐tok/s':>12}{'有效batch':>10}{'相对上步':>10}"
    print(header)
    print("-" * 92)
    for r in rows:
        m = r["metrics"]
        # 去掉阶段名里的换行，便于表格对齐
        name = r["stage"].replace("\n", " ")
        speedup = r["speedup_vs_prev"]
        speedup_s = f"×{speedup:.2f}" if speedup != float("inf") else "n/a"
        print(f"{name:<22}{m.weight_mem_gb:>9.2f}{m.kv_mem_gb:>9.2f}{m.total_mem_gb:>9.2f}"
              f"{m.throughput_tok_s:>12.1f}{m.effective_batch:>10}{speedup_s:>10}")
    print("-" * 92)
    base = rows[0]["metrics"]
    full = rows[-1]["metrics"]
    print(f"总收益：吞吐 ×{full.throughput_tok_s / base.throughput_tok_s:.1f}，"
          f"权重显存 ÷{base.weight_mem_gb / full.weight_mem_gb:.1f}")
    print("=" * 92 + "\n")


def plot_throughput_waterfall(rows, out_path):
    """
    吞吐瀑布图（waterfall）：
    每根柱子从上一阶段的吞吐「接力」往上叠，直观展示每层优化贡献了多少吞吐增量。
    """
    labels = [r["stage"] for r in rows]
    values = [r["metrics"].throughput_tok_s for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.2))

    # 瀑布：第 0 根从 0 起；后面每根从「上一根顶端」起，画出增量段
    bottoms = [0.0]
    for i in range(1, len(values)):
        bottoms.append(values[i - 1])

    colors = ["#8894a6", "#f0a35e", "#5eb36a", "#5e93d1"]
    for i, (lab, val, bot) in enumerate(zip(labels, values, bottoms)):
        height = val if i == 0 else val - bot
        ax.bar(i, height, bottom=bot, color=colors[i % len(colors)], edgecolor="black", width=0.62)
        # 顶端标注绝对吞吐
        ax.text(i, val + max(values) * 0.015, f"{val:,.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        # 增量段中间标注 delta（第 0 根不标）
        if i > 0:
            delta = val - bot
            ax.annotate(f"+{delta:,.0f}", xy=(i, bot + height / 2), ha="center", va="center",
                        fontsize=9.5, color="black")
        # 连接虚线，强调「接力叠加」
        if i > 0:
            ax.plot([i - 1 + 0.31, i - 0.31], [values[i - 1], values[i - 1]],
                    linestyle="--", color="gray", linewidth=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("decode 吞吐（token/s）", fontsize=12)
    ax.set_title("优化叠加瀑布图：量化 → 连续批 → KV量化，逐级看吞吐收益",
                 fontsize=14, fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[已保存] 吞吐瀑布图 -> {out_path}")


def plot_memory_and_batch_ceiling(calc: OptimizationCalculator, out_path):
    """
    左图：不同优化组合下的显存分解（权重 vs KV）；
    右图：不同量化组合能装下的 batch 天花板（显存墙）。
    """
    combos = [
        ("FP16权重\nKV FP16", "fp16", "fp16"),
        ("W4A16\nKV FP16", "int4", "fp16"),
        ("W4A16\nKV int8", "int4", "int8"),
        ("FP8权重\nKV fp8", "fp8", "fp8"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    # ---- 左：显存分解（固定 batch=16 做对比）----
    names = [c[0] for c in combos]
    w_mem = [calc.weight_memory_gb(c[1]) for c in combos]
    kv_mem = [calc.kv_memory_gb(16, c[2]) for c in combos]
    x = range(len(combos))
    ax1.bar(x, w_mem, color="#5e93d1", edgecolor="black", label="权重显存")
    ax1.bar(x, kv_mem, bottom=w_mem, color="#f0a35e", edgecolor="black", label="KV缓存(batch=16)")
    ax1.axhline(calc.hw.vram_gb, color="red", linestyle="--", linewidth=1.5,
                label=f"显存容量 {calc.hw.vram_gb:.0f}GB")
    for i, (w, k) in enumerate(zip(w_mem, kv_mem)):
        ax1.text(i, w + k + 1.2, f"{w + k:.1f}", ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, fontsize=9.5)
    ax1.set_ylabel("显存占用（GB）", fontsize=12)
    ax1.set_title("显存分解：量化压权重，KV量化压KV", fontsize=12.5, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)

    # ---- 右：batch 天花板（显存能装下多少并发）----
    max_batches = [calc.max_batch_by_memory(c[1], c[2]) for c in combos]
    bars = ax2.bar(x, max_batches, color="#5eb36a", edgecolor="black")
    for i, b in enumerate(max_batches):
        ax2.text(i, b + max(max_batches) * 0.02, str(b), ha="center",
                 fontsize=11, fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, fontsize=9.5)
    ax2.set_ylabel("显存允许的最大 batch", fontsize=12)
    ax2.set_title("batch 天花板：量化+KV量化如何撑高并发", fontsize=12.5, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[已保存] 显存/batch天花板图 -> {out_path}")


def main():
    calc, model, hw, wl = default_setup()
    stages = build_stages()
    rows = calc.waterfall(stages)

    # 1) 终端表
    print_table(rows)

    # 2) 瀑布图
    plot_throughput_waterfall(rows, os.path.join(HERE, "waterfall_throughput.png"))

    # 3) 显存 + batch 天花板图
    plot_memory_and_batch_ceiling(calc, os.path.join(HERE, "memory_batch_ceiling.png"))

    print("Demo 完成。生成了 2 张 PNG，可用图片查看器打开。")


if __name__ == "__main__":
    main()
