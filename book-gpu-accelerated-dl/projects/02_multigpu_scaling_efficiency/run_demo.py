# -*- coding: utf-8 -*-
"""
run_demo.py — 多 GPU 扩展效率模型可视化演示

跑一次会:
  1) 在控制台打印一张"卡数 vs 加速比/效率"的对照表(不同通信占比);
  2) 出两张图(保存到 figures/):
     - fig1_speedup.png    :卡数 vs 加速比,含理想线性参考线 + Amdahl 天花板;
     - fig2_efficiency.png :卡数 vs 扩展效率,展示通信占比如何压垮强扩展;
  3) 出一张组合大图 fig3_dashboard.png(加速比 + 效率并排)。

运行:  python run_demo.py

要点:matplotlib 用 Agg 后端(无需显示器/纯离线出图),中文字体设 Microsoft YaHei。
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # 关键:无界面后端,离线/服务器/CI 都能出图
import matplotlib.pyplot as plt
import numpy as np

# 中文与负号显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scaling_model import (  # noqa: E402
    CommModel,
    amdahl_speedup,
    half_efficiency_n,
    max_speedup_amdahl,
    scaling_curve,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 卡数扫描范围(2 的幂,覆盖单机 8 卡到多机 256 卡)
N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# 固定串行占比;调不同通信占比看强扩展如何被压垮
SERIAL = 0.02  # 2% 串行(已经相当好的实现)

# 四种通信情形:从"几乎无通信"到"通信很重"
SCENARIOS = [
    ("无通信(理想 Amdahl)", None,
     dict(color="#2ca02c", ls="-",  marker="o")),
    ("轻通信 comm_base=0.01", CommModel(message_bytes=1e8, bandwidth=1e10, scaling="ring"),
     dict(color="#1f77b4", ls="-",  marker="s")),
    ("中通信 comm_base=0.05", CommModel(message_bytes=5e8, bandwidth=1e10, scaling="ring"),
     dict(color="#ff7f0e", ls="--", marker="^")),
    ("重通信 comm_base=0.20", CommModel(message_bytes=2e9, bandwidth=1e10, scaling="ring"),
     dict(color="#d62728", ls="-.", marker="v")),
]


def print_table():
    """控制台打印对照表(用 ASCII,避免 Windows 控制台中文乱码影响可读性)。"""
    print("=" * 78)
    print("Multi-GPU Strong-Scaling  (serial fraction s = %.2f)" % SERIAL)
    print("Amdahl ceiling 1/s = %.1f" % max_speedup_amdahl(SERIAL))
    print("=" * 78)
    header = "%6s" % "N"
    for label, _, _ in SCENARIOS:
        # 表头用英文短标签
        pass
    labels = ["no-comm", "light(0.01)", "mid(0.05)", "heavy(0.20)"]
    header = "%6s | " % "N" + " | ".join("%12s" % l for l in labels)
    print(header)
    print("-" * len(header))
    curves = [scaling_curve(N_LIST, SERIAL, comm) for _, comm, _ in SCENARIOS]
    for i, n in enumerate(N_LIST):
        row = "%6d | " % n
        cells = []
        for c in curves:
            cells.append("%5.2fx/%3.0f%%" % (c["speedup"][i], 100 * c["efficiency"][i]))
        row += " | ".join("%12s" % x for x in cells)
        print(row)
    print("-" * len(header))
    print("cell = speedup / efficiency%")
    print()
    for label, comm, _ in SCENARIOS:
        hn = half_efficiency_n(SERIAL, comm)
        print("  half-efficiency N (E<=50%%) for [%-24s] = %s"
              % (label, hn if hn is not None else ">searchmax"))
    print("=" * 78)


def plot_speedup():
    """图 1:卡数 vs 加速比。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ns = np.asarray(N_LIST, dtype=float)

    # 理想线性参考线
    ax.plot(ns, ns, color="gray", ls=":", lw=1.5, label="理想线性 S=N")

    # Amdahl 天花板水平线
    ceil = max_speedup_amdahl(SERIAL)
    ax.axhline(ceil, color="black", ls="--", lw=1.0, alpha=0.6,
               label="Amdahl 天花板 1/s = %.0f" % ceil)

    for label, comm, style in SCENARIOS:
        curve = scaling_curve(N_LIST, SERIAL, comm)
        ax.plot(curve["n"], curve["speedup"], label=label, lw=2, **style)

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST])
    ax.set_xlabel("GPU 卡数 N(对数轴)")
    ax.set_ylabel("加速比 Speedup  S(N)")
    ax.set_title("多 GPU 强扩展:加速比 vs 卡数\n(串行占比 s=%.0f%%,通信越重越早偏离理想线)"
                 % (SERIAL * 100))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_speedup.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_efficiency():
    """图 2:卡数 vs 扩展效率。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="理想效率 E=1")
    ax.axhline(0.5, color="red", ls="--", lw=1.0, alpha=0.6, label="半效率线 E=0.5")

    for label, comm, style in SCENARIOS:
        curve = scaling_curve(N_LIST, SERIAL, comm)
        ax.plot(curve["n"], curve["efficiency"], label=label, lw=2, **style)

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("GPU 卡数 N(对数轴)")
    ax.set_ylabel("扩展效率 Efficiency  E(N)=S(N)/N")
    ax.set_title("多 GPU 强扩展:扩展效率 vs 卡数\n(效率跌破 0.5 = 一半算力被串行/通信浪费)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_efficiency.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_dashboard():
    """图 3:组合大图(加速比 + 效率并排),便于一眼对比。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ns = np.asarray(N_LIST, dtype=float)

    # 左:加速比
    ax1.plot(ns, ns, color="gray", ls=":", lw=1.5, label="理想线性 S=N")
    ceil = max_speedup_amdahl(SERIAL)
    ax1.axhline(ceil, color="black", ls="--", lw=1.0, alpha=0.6,
                label="Amdahl 天花板=%.0f" % ceil)
    for label, comm, style in SCENARIOS:
        c = scaling_curve(N_LIST, SERIAL, comm)
        ax1.plot(c["n"], c["speedup"], label=label, lw=2, **style)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(N_LIST)
    ax1.set_xticklabels([str(n) for n in N_LIST])
    ax1.set_xlabel("GPU 卡数 N")
    ax1.set_ylabel("加速比 S(N)")
    ax1.set_title("(a) 加速比 vs 卡数")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")

    # 右:效率
    ax2.axhline(1.0, color="gray", ls=":", lw=1.5, label="理想效率 E=1")
    ax2.axhline(0.5, color="red", ls="--", lw=1.0, alpha=0.6, label="半效率 E=0.5")
    for label, comm, style in SCENARIOS:
        c = scaling_curve(N_LIST, SERIAL, comm)
        ax2.plot(c["n"], c["efficiency"], label=label, lw=2, **style)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(N_LIST)
    ax2.set_xticklabels([str(n) for n in N_LIST])
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("GPU 卡数 N")
    ax2.set_ylabel("扩展效率 E(N)")
    ax2.set_title("(b) 扩展效率 vs 卡数")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle("多 GPU 扩展效率仪表盘(Amdahl + 通信开销模型,s=%.0f%%)"
                 % (SERIAL * 100), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "fig3_dashboard.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print_table()
    f1 = plot_speedup()
    f2 = plot_efficiency()
    f3 = plot_dashboard()
    print("\n已保存图像:")
    for f in (f1, f2, f3):
        print("  -", f)
    print("\n演示完成。用图片查看器打开 figures/ 目录即可。")


if __name__ == "__main__":
    main()
