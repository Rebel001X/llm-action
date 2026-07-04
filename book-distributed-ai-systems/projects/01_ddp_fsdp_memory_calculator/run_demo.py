# -*- coding: utf-8 -*-
"""
run_demo.py —— DDP vs ZeRO/FSDP 显存计算器 可视化 Demo
=====================================================
产出 3 张图(保存到 ./figures/):
  1. fig1_strategy_breakdown.png : 固定 7B 模型, 各策略 单卡显存 分块堆叠柱状图 + 40G/80G 显存线
  2. fig2_dp_scaling.png         : ZeRO-3/FSDP 下, 单卡显存 随 DP 度 变化曲线(几种模型规模)
  3. fig3_model_size_grid.png    : 不同模型规模 × 不同策略 的单卡显存 分组柱状图

离线纯本地运行:python run_demo.py
matplotlib 用 Agg 后端(无需显示器),中文用 微软雅黑。
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # ⚠️ 无 GUI/服务器环境必须先设 Agg,再 import pyplot
import matplotlib.pyplot as plt
import numpy as np

# 中文字体 & 负号正常显示(Windows 常见坑)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 让脚本能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_calculator import (  # noqa: E402
    ModelConfig,
    ParallelStrategy,
    compare_all_strategies,
    estimate_model_state,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 统一配色:参数/梯度/优化器
COLOR_P = "#4C72B0"   # 蓝 - 参数
COLOR_G = "#55A868"   # 绿 - 梯度
COLOR_O = "#C44E52"   # 红 - 优化器


def fig1_strategy_breakdown(num_params=7e9, dp_degree=8, gpu_lines=(40, 80)):
    """图1:7B 模型固定 DP=8,各策略单卡显存(P/G/O 堆叠)柱状图。"""
    cfg = ModelConfig(num_params=num_params, optimizer="adam",
                      param_dtype="fp16", dp_degree=dp_degree)
    results = compare_all_strategies(cfg)
    names = list(results.keys())
    p = np.array([results[n].param_bytes for n in names]) / (1024 ** 3)
    g = np.array([results[n].grad_bytes for n in names]) / (1024 ** 3)
    o = np.array([results[n].optimizer_bytes for n in names]) / (1024 ** 3)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, p, label="参数 P (fp16)", color=COLOR_P)
    ax.bar(x, g, bottom=p, label="梯度 G (fp16)", color=COLOR_G)
    ax.bar(x, o, bottom=p + g, label="优化器状态 O (Adam fp32)", color=COLOR_O)

    # 总量标注
    total = p + g + o
    for xi, t in zip(x, total):
        ax.text(xi, t + max(total) * 0.01, f"{t:.1f}G",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # GPU 显存参考线
    line_styles = ["--", ":"]
    for gpu, ls in zip(gpu_lines, line_styles):
        ax.axhline(gpu, color="gray", linestyle=ls, linewidth=1.4)
        ax.text(len(names) - 0.4, gpu + 1, f"{gpu}GB 卡",
                color="gray", fontsize=9, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("单卡模型状态显存 (GB)", fontsize=12)
    ax.set_title(f"{num_params/1e9:.0f}B 模型 · Adam · fp16 · DP={dp_degree}\n"
                 f"各并行策略 单卡显存对比(越往右分片越激进)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_strategy_breakdown.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig2_dp_scaling(model_sizes=(7e9, 13e9, 70e9), gpu_line=80):
    """图2:ZeRO-3/FSDP 下,单卡显存随 DP 度增大而下降的曲线。"""
    dp_range = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    for size, c in zip(model_sizes, colors):
        ys = []
        for D in dp_range:
            cfg = ModelConfig(num_params=size, optimizer="adam",
                              param_dtype="fp16", dp_degree=D)
            bd = estimate_model_state(cfg, ParallelStrategy.ZERO3)
            ys.append(bd.model_state_gb)
        ax.plot(dp_range, ys, marker="o", color=c,
                label=f"{size/1e9:.0f}B 模型")

    ax.axhline(gpu_line, color="gray", linestyle="--", linewidth=1.4)
    ax.text(dp_range[-1], gpu_line + 2, f"{gpu_line}GB 卡",
            color="gray", ha="right", fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dp_range)
    ax.set_xticklabels([str(d) for d in dp_range])
    ax.set_xlabel("数据并行度 DP(参与分片的卡数)", fontsize=12)
    ax.set_ylabel("单卡模型状态显存 (GB, log)", fontsize=12)
    ax.set_title("ZeRO-3 / FSDP:单卡显存 ∝ 1/DP\n"
                 "卡越多,单卡越省 —— 这就是 FSDP 能训超大模型的关键", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_dp_scaling.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig3_model_size_grid(dp_degree=8):
    """图3:不同模型规模 × 各策略 单卡显存 分组柱状图。"""
    sizes = [1e9, 7e9, 13e9, 70e9]
    strategies = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    data = {s: [] for s in strategies}
    for size in sizes:
        cfg = ModelConfig(num_params=size, optimizer="adam",
                          param_dtype="fp16", dp_degree=dp_degree)
        r = compare_all_strategies(cfg)
        for s in strategies:
            data[s].append(r[s].model_state_gb)

    x = np.arange(len(sizes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    palette = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
    for i, (s, c) in enumerate(zip(strategies, palette)):
        ax.bar(x + (i - 1.5) * width, data[s], width, label=s, color=c)

    ax.axhline(80, color="gray", linestyle="--", linewidth=1.2)
    ax.text(len(sizes) - 0.5, 82, "80GB 卡", color="gray", ha="right", fontsize=9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s/1e9)}B" for s in sizes])
    ax.set_xlabel("模型规模", fontsize=12)
    ax.set_ylabel("单卡模型状态显存 (GB, log)", fontsize=12)
    ax.set_title(f"不同模型规模 × 并行策略 单卡显存(DP={dp_degree})\n"
                 f"注意 70B 模型只有 ZeRO-3 才勉强进 80G 卡", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_model_size_grid.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    # ⚠️ Windows 控制台默认 GBK,重设 UTF-8 以打印中文/emoji
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("DDP vs ZeRO/FSDP 显存计算器 —— Demo")
    print("=" * 60)

    # 先打印一份 7B 的文字对比表
    from memory_calculator import summarize
    print(summarize(ModelConfig(7e9, "adam", "fp16", 8), gpu_mem_gb=40.0))
    print()

    print("正在生成图表...")
    f1 = fig1_strategy_breakdown()
    print(f"  [1/3] 已保存: {f1}")
    f2 = fig2_dp_scaling()
    print(f"  [2/3] 已保存: {f2}")
    f3 = fig3_model_size_grid()
    print(f"  [3/3] 已保存: {f3}")
    print()
    print("✅ 全部完成!图在 figures/ 目录。")


if __name__ == "__main__":
    main()
