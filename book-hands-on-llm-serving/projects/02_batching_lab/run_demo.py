# -*- coding: utf-8 -*-
"""
run_demo.py —— 批处理策略对比可视化 Demo
====================================================================

一键跑完四种批处理策略，打印指标表格，并出两张图：
  1. batching_timeline.png —— 四策略「GPU 并行度随时间变化」的时间线，
     直观看出：静态/动态有大量「批尾独占」和「攒批空转」的低占用区，
     连续批处理常年填满槽位。
  2. batching_metrics.png  —— 吞吐 / token吞吐 / 并行利用率 / p50/p95/p99 尾延迟
     的分组柱状图，一眼看清连续批处理的全面领先。

运行：  python run_demo.py
输出：  控制台指标表 + 当前目录下两张 png（无需 GPU / 联网）。
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

# ⚠️ 关键：无显示环境（服务器/CI/后台）必须用 Agg 后端，否则 plt.savefig 可能报错。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体 + 负号正常显示（Windows 用微软雅黑；缺失则退化到黑体）。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from batching_sim import make_workload, run_all  # noqa: E402


# 四策略统一配色与中文名，图例/柱状图共用。
STRATEGY_ORDER = ["no_batching", "static", "dynamic", "continuous"]
STRATEGY_CN = {
    "no_batching": "不批处理",
    "static": "静态批",
    "dynamic": "动态批",
    "continuous": "连续批(赢)",
}
STRATEGY_COLOR = {
    "no_batching": "#9e9e9e",   # 灰
    "static": "#ef6c57",        # 橙红
    "dynamic": "#f5b642",       # 黄
    "continuous": "#3a9e5c",    # 绿（赢家）
}


def print_table(results) -> None:
    """在控制台打印一张对齐的指标表。"""
    header = (f"{'策略':<10}{'吞吐(req/步)':>12}{'token吞吐':>10}"
              f"{'并行利用率':>11}{'空闲率':>9}{'p50':>7}{'p95':>7}{'p99':>7}{'完成':>6}")
    print("\n" + "=" * len(header))
    print("批处理策略对比（同一份工作负载，60 请求，重尾输出）")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in STRATEGY_ORDER:
        res = results[name]
        m = res.metrics_dict()
        print(f"{STRATEGY_CN[name]:<10}{m['throughput']:>12.3f}{m['token_throughput']:>10.2f}"
              f"{m['slot_utilization']:>11.2%}{m['idle_slot_ratio']:>9.2%}"
              f"{m['p50']:>7.1f}{m['p95']:>7.1f}{m['p99']:>7.1f}{int(m['completed']):>6}")
    print("=" * len(header))
    # 关键结论
    c = results["continuous"]
    s = results["static"]
    print(f"\n连续批相对静态批：吞吐 ×{c.throughput / s.throughput:.2f}，"
          f"并行利用率 {s.slot_utilization:.1%} → {c.slot_utilization:.1%}，"
          f"p99 尾延迟 {s.latency_percentile(99):.0f} → {c.latency_percentile(99):.0f} 步"
          f"（降 {1 - c.latency_percentile(99) / s.latency_percentile(99):.0%}）。\n")


def plot_timeline(results, out_path: str) -> None:
    """四个子图，各画一条「每步活跃请求数」的时间线（面积图）。"""
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    # 统一横轴到最长的时间线，方便直接对比「谁先跑完」
    max_len = max(len(results[n].timeline) for n in STRATEGY_ORDER)
    # 统一纵轴到最大容量，方便对比「槽位填得满不满」
    cap = max(results[n].capacity for n in STRATEGY_ORDER)

    for ax, name in zip(axes, STRATEGY_ORDER):
        res = results[name]
        tl = np.asarray(res.timeline, dtype=float)
        x = np.arange(len(tl))
        ax.fill_between(x, tl, step="post", color=STRATEGY_COLOR[name], alpha=0.85)
        ax.axhline(res.capacity, ls="--", lw=1, color="#555",
                   label=f"容量上限={res.capacity}")
        util = res.slot_utilization
        ax.set_ylabel(f"{STRATEGY_CN[name]}\n活跃请求数", fontsize=10)
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, cap + 0.6)
        ax.grid(alpha=0.25)
        # 右上角标注关键指标
        ax.text(0.995, 0.92,
                f"总步={res.total_steps}  并行利用率={util:.0%}  p99={res.latency_percentile(99):.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("仿真时间步（一次前向 = 1 步）", fontsize=11)
    fig.suptitle("四种批处理策略的 GPU 并行度时间线\n"
                 "（面积越接近容量上限=槽位越满=GPU 越不浪费；越早触底=越早全部跑完）",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[已保存] 时间线图 → {out_path}")


def plot_metrics(results, out_path: str) -> None:
    """两块柱状图：左=吞吐/利用率类，右=尾延迟类。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = STRATEGY_ORDER
    labels = [STRATEGY_CN[n] for n in names]
    colors = [STRATEGY_COLOR[n] for n in names]

    # ---------- 左图：吞吐 + 并行利用率（双指标，归一化到各自最大值以便同图对比）----------
    thr = np.array([results[n].throughput for n in names])
    util = np.array([results[n].slot_utilization for n in names])
    x = np.arange(len(names))
    w = 0.38
    b1 = ax1.bar(x - w / 2, thr / thr.max(), w, label="吞吐（相对最大值）",
                 color=colors, edgecolor="black", alpha=0.9)
    b2 = ax1.bar(x + w / 2, util, w, label="并行利用率",
                 color=colors, edgecolor="black", hatch="//", alpha=0.55)
    # 在柱顶标真实数值
    for rect, v in zip(b1, thr):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for rect, v in zip(b2, util):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                 f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.18)
    ax1.set_title("吞吐 & GPU 并行利用率（越高越好）", fontsize=12)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)

    # ---------- 右图：p50 / p95 / p99 尾延迟分组柱 ----------
    p50 = np.array([results[n].latency_percentile(50) for n in names])
    p95 = np.array([results[n].latency_percentile(95) for n in names])
    p99 = np.array([results[n].latency_percentile(99) for n in names])
    w2 = 0.26
    ax2.bar(x - w2, p50, w2, label="p50", color="#7fb3d5", edgecolor="black")
    ax2.bar(x, p95, w2, label="p95", color="#f5b042", edgecolor="black")
    ax2.bar(x + w2, p99, w2, label="p99", color="#e15241", edgecolor="black")
    for i in range(len(names)):
        ax2.text(x[i] + w2, p99[i] + p99.max() * 0.01, f"{p99[i]:.0f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("尾延迟 p50/p95/p99（越低越好，单位=仿真步）", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("批处理策略指标总览：连续批处理吞吐更高、GPU 更满、尾延迟更低",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[已保存] 指标柱状图 → {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    # 生成一份「有重尾输出、中等到达率」的工作负载，最能体现四策略差异。
    wl = make_workload(n_requests=60, seed=0, arrival_rate=0.5)
    results = run_all(wl, batch_size=8, max_delay=3)

    print_table(results)
    plot_timeline(results, os.path.join(here, "batching_timeline.png"))
    plot_metrics(results, os.path.join(here, "batching_metrics.png"))
    print("Demo 完成：可用图片查看器打开上面两张 png。")


if __name__ == "__main__":
    main()
