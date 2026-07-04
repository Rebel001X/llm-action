# -*- coding: utf-8 -*-
"""
run_demo.py
===========
可视化演示：并行 step 复杂度随规模 n 的增长。

输出三张图（保存到 figures/ 目录）：
  1) fig_steps_vs_n.png    —— 各模式 step 数 vs n，对比 O(log n) 与 O(n)
  2) fig_work_vs_n.png     —— 各 scan 的 work vs n，对比 O(n) 与 O(n log n)
  3) fig_work_per_step.png —— reduction 每一步的工作量如何逐步减半

离线运行：不联网、不需 GPU。matplotlib 用 Agg 后端直接存文件。
运行：  python run_demo.py
"""
import os

import matplotlib

# ⚠️ 必须在 import pyplot 之前设置 Agg 后端：无显示环境（服务器/CI）也能出图
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体与负号显示（Windows 常用 Microsoft YaHei / SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from parallel_patterns import (  # noqa: E402
    ceil_log2,
    reduce_tree,
    scan_blelloch,
    scan_hillis_steele,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RNG = np.random.default_rng(20260704)


def demo_steps_vs_n():
    """图1：step 数 vs n —— 直观看到 O(log n) 曲线几乎是「平的」，O(n) 直冲天花板。"""
    ns = [2 ** k for k in range(1, 17)]  # 2 .. 65536
    red_steps, hs_steps, bl_steps = [], [], []
    for n in ns:
        x = RNG.standard_normal(n)
        _, sr = reduce_tree(x)
        _, sh = scan_hillis_steele(x)
        _, sb = scan_blelloch(x)
        red_steps.append(sr.steps)
        hs_steps.append(sh.steps)
        bl_steps.append(sb.steps)

    serial_steps = [n - 1 for n in ns]  # 串行深度 = n-1（O(n)）

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ns, serial_steps, "o-", color="#d62728", label="串行 Serial：step = n-1（O(n)）")
    ax.plot(ns, red_steps, "s-", color="#1f77b4", label="Reduction 树形：ceil(log2 n)（O(log n)）")
    ax.plot(ns, hs_steps, "^-", color="#2ca02c", label="Scan Hillis-Steele：log2 n（O(log n)）")
    ax.plot(ns, bl_steps, "d-", color="#9467bd", label="Scan Blelloch：2·log2 n（O(log n)）")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("输入规模 n（对数轴，2 的幂）")
    ax.set_ylabel("并行步数 step / span（对数轴）")
    ax.set_title("并行 step 复杂度：O(log n) vs O(n)\n（GPU 之所以快：深度是对数级，n 翻千倍 step 只 +10）")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_steps_vs_n.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def demo_work_vs_n():
    """图2：work vs n —— Hillis-Steele 的 O(n log n) 明显高于 Blelloch / 串行的 O(n)。"""
    ns = [2 ** k for k in range(2, 17)]
    hs_work, bl_work, serial_work = [], [], []
    for n in ns:
        x = RNG.standard_normal(n)
        _, sh = scan_hillis_steele(x)
        _, sb = scan_blelloch(x)
        hs_work.append(sh.work)
        bl_work.append(sb.work)
        serial_work.append(n - 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ns, serial_work, "o-", color="#d62728", label="串行 Serial：work = n-1（O(n)）")
    ax.plot(ns, bl_work, "d-", color="#9467bd", label="Blelloch：work ≈ 2n（O(n)，work-efficient）")
    ax.plot(ns, hs_work, "^-", color="#2ca02c", label="Hillis-Steele：work ≈ n·log n（O(n log n)）")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("输入规模 n（对数轴）")
    ax.set_ylabel("总工作量 work / 有效加法次数（对数轴）")
    ax.set_title("Scan 的 work 对比：步高效 ≠ 工作高效\nHillis-Steele 步少但总加法更多；Blelloch 才是 work-efficient")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_work_vs_n.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def demo_work_per_step():
    """图3：树形归约每一步的工作量 —— 每步减半，是「工作量塔」逐层收窄的直观图。"""
    n = 4096
    x = RNG.standard_normal(n)
    _, stats = reduce_tree(x)
    steps = list(range(1, stats.steps + 1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(steps, stats.work_per_step, color="#1f77b4", alpha=0.85)
    for b, w in zip(bars, stats.work_per_step):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(w),
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(f"归约的第几步（n={n}，共 {stats.steps} 步 = ceil(log2 {n})）")
    ax.set_ylabel("本步的有效加法次数（活跃线程数）")
    ax.set_title(f"树形归约：每步工作量减半（Σ = {stats.work} = n-1）\n"
                 "GPU 上后期活跃线程锐减 → 出现『线程闲置』，也是归约优化的重点")
    ax.set_xticks(steps)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_work_per_step.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def print_summary_table():
    """在终端打印一张各模式复杂度对照表（供快速自检）。"""
    print("\n" + "=" * 68)
    print(" 并行模式复杂度对照（示例 n=4096）")
    print("=" * 68)
    n = 4096
    x = RNG.standard_normal(n)
    _, sr = reduce_tree(x)
    _, sh = scan_hillis_steele(x)
    _, sb = scan_blelloch(x)
    rows = [
        ("Reduction (树形)", sr.steps, sr.work, "O(log n)", "O(n)"),
        ("Scan Hillis-Steele", sh.steps, sh.work, "O(log n)", "O(n log n)"),
        ("Scan Blelloch", sb.steps, sb.work, "O(log n)", "O(n)"),
    ]
    print(f"{'模式':<22}{'step':>8}{'work':>12}{'step阶':>12}{'work阶':>14}")
    print("-" * 68)
    for name, st, wk, so, wo in rows:
        print(f"{name:<22}{st:>8}{wk:>12}{so:>12}{wo:>14}")
    print("=" * 68)


def main():
    print(">> 生成并行 step / work 复杂度图 ...")
    p1 = demo_steps_vs_n()
    p2 = demo_work_vs_n()
    p3 = demo_work_per_step()
    print_summary_table()
    print("\n已生成 3 张图：")
    for p in (p1, p2, p3):
        print("  -", p)
    print("\n完成。用图片查看器打开 figures/ 下的 PNG 即可。")


if __name__ == "__main__":
    main()
