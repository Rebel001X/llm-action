# -*- coding: utf-8 -*-
"""
run_demo.py — 分块矩阵乘访存模型 · 可视化演示
==============================================

跑一遍就能得到「tile 大小 vs HBM 访问量 / 算术强度 / 复用率」的曲线,
以及一张屋顶线(roofline)示意图,把「分块为什么减少访存」一图看懂。

运行:
    python run_demo.py

产出(保存在本文件同目录):
    fig1_hbm_vs_tile.png          HBM 访问量随 tile 下降(对数轴)
    fig2_intensity_reuse.png      算术强度 & 复用率随 tile 上升
    fig3_roofline.png             屋顶线:分块把 kernel 推离访存墙

同时在终端打印一张对照表。全程 CPU、离线、无需 GPU。
"""

import matplotlib
matplotlib.use("Agg")  # 无界面后端,写文件不弹窗
import matplotlib.pyplot as plt
import numpy as np

# 中文字体 + 负号正常显示(Windows 常见坑)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiled_matmul import (  # noqa: E402
    analyze_square,
    sweep_tiles,
    max_tile_for_smem,
)

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1024                 # 用一个较大的方阵让对比更醒目
DTYPE_BYTES = 4          # fp32
TILES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SMEM_KB = 48             # 典型 SM 的 shared memory(KB)


def _fmt_bytes(x: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:6.2f} {unit}"
        x /= 1024
    return f"{x:6.2f} TB"


def print_table():
    print("=" * 78)
    print(f"方阵 matmul  n = {N}   dtype = fp32({DTYPE_BYTES}B)   "
          f"FLOPs = {2*N**3/1e9:.2f} GFLOP")
    print("-" * 78)
    print(f"{'tile T':>7} | {'HBM读':>12} | {'HBM总':>12} | "
          f"{'算术强度':>10} | {'复用率':>8}")
    print("-" * 78)
    naive = analyze_square(N, 1, scheme="naive", dtype_bytes=DTYPE_BYTES)
    print(f"{'naive':>7} | {_fmt_bytes(naive.hbm_read_bytes):>12} | "
          f"{_fmt_bytes(naive.hbm_bytes_total):>12} | "
          f"{naive.arithmetic_intensity:8.2f} FB | {naive.reuse_factor:6.1f}x")
    for t in TILES:
        m = analyze_square(N, t, scheme="tiled", dtype_bytes=DTYPE_BYTES)
        print(f"{t:>7} | {_fmt_bytes(m.hbm_read_bytes):>12} | "
              f"{_fmt_bytes(m.hbm_bytes_total):>12} | "
              f"{m.arithmetic_intensity:8.2f} FB | {m.reuse_factor:6.1f}x")
    tmax = max_tile_for_smem(SMEM_KB * 1024, DTYPE_BYTES, num_tiles=2)
    print("-" * 78)
    print(f"48KB shared memory 下,能放下 A/B 两个子块的最大方形 tile ≈ {tmax}")
    print("=" * 78)


def fig1_hbm_vs_tile():
    sw = sweep_tiles(N, TILES, dtype_bytes=DTYPE_BYTES)
    tiles = sw["tiles"]
    read_mb = sw["hbm_read_bytes"] / 1e6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tiles, read_mb, "o-", color="#4dabf7", lw=2, ms=7,
            label="分块 kernel:HBM 读入")
    ax.axhline(sw["naive_read_bytes"] / 1e6, ls="--", color="#ff6b6b",
               lw=2, label="朴素 kernel(基线,= 2·n³)")

    # 标出 shared memory 上限对应的 tile
    tmax = max_tile_for_smem(SMEM_KB * 1024, DTYPE_BYTES, num_tiles=2)
    ax.axvline(tmax, ls=":", color="#51cf66", lw=2,
               label=f"48KB SMEM 上限 tile≈{tmax}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("tile 边长 T(对数轴)")
    ax.set_ylabel("从 HBM 读入的数据量 / MB(对数轴)")
    ax.set_title(f"分块越大,HBM 访问越少(n={N},每翻倍 T 读量减半)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    out = os.path.join(HERE, "fig1_hbm_vs_tile.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def fig2_intensity_reuse():
    sw = sweep_tiles(N, TILES, dtype_bytes=DTYPE_BYTES)
    tiles = sw["tiles"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#c084fc"
    ax1.plot(tiles, sw["arithmetic_intensity"], "s-", color=color1,
             lw=2, ms=7, label="算术强度 AI")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("tile 边长 T(对数轴)")
    ax1.set_ylabel("算术强度 / (FLOP·Byte⁻¹)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    ax2 = ax1.twinx()
    color2 = "#f59f00"
    ax2.plot(tiles, sw["reuse_factor"], "^--", color=color2,
             lw=2, ms=7, label="复用率 (= T)")
    ax2.set_ylabel("复用率(每个搬入元素被算几次)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"分块把 kernel 从「访存受限」推向「计算受限」(n={N})")
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    out = os.path.join(HERE, "fig2_intensity_reuse.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def fig3_roofline():
    """
    屋顶线(roofline)示意:横轴算术强度,纵轴可达性能。
    斜坡 = 带宽墙(perf = BW * AI),平台 = 算力峰值。
    把 naive 与各 tile 的 AI 点标上去,直观看到分块如何「爬坡」。
    用一组具代表性的假想硬件参数(仅示意,不代表真实设备)。
    """
    peak_flops = 20e12      # 20 TFLOP/s(假想 fp32 峰值)
    peak_bw = 900e9         # 900 GB/s(假想 HBM 带宽)
    ridge_ai = peak_flops / peak_bw  # 脊点:AI 超过这里才计算受限

    ai_axis = np.logspace(-1, 3, 400)
    roof = np.minimum(peak_flops, peak_bw * ai_axis) / 1e12  # TFLOP/s

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ai_axis, roof, "-", color="#495057", lw=2.5, label="屋顶线")
    ax.axvline(ridge_ai, ls=":", color="#868e96", lw=1.5,
               label=f"脊点 AI={ridge_ai:.1f}")

    # 标注 naive + 若干 tile 的算术强度
    naive = analyze_square(N, 1, scheme="naive", dtype_bytes=DTYPE_BYTES)
    ax.scatter([naive.arithmetic_intensity],
               [min(peak_flops, peak_bw * naive.arithmetic_intensity) / 1e12],
               color="#ff6b6b", s=90, zorder=5,
               label=f"naive AI={naive.arithmetic_intensity:.2f}")
    for t, c in zip((16, 32, 64, 128), ("#4dabf7", "#51cf66",
                                        "#c084fc", "#f59f00")):
        m = analyze_square(N, t, scheme="tiled", dtype_bytes=DTYPE_BYTES)
        perf = min(peak_flops, peak_bw * m.arithmetic_intensity) / 1e12
        ax.scatter([m.arithmetic_intensity], [perf], color=c, s=80, zorder=5,
                   label=f"tile={t} AI={m.arithmetic_intensity:.1f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("算术强度 / (FLOP·Byte⁻¹)")
    ax.set_ylabel("可达性能 / (TFLOP·s⁻¹)")
    ax.set_title("屋顶线:分块把 matmul 从带宽墙抬向算力峰值(参数为示意)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    out = os.path.join(HERE, "fig3_roofline.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    print_table()
    fig1_hbm_vs_tile()
    fig2_intensity_reuse()
    fig3_roofline()
    print("\n完成:三张 PNG 已写入项目目录。用图片查看器打开对比即可。")


if __name__ == "__main__":
    main()
