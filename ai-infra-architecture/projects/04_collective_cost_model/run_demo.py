"""
run_demo.py —— 用 α-β 代价模型对比 ring / tree / double-binary-tree 三种 AllReduce,
打印结论表并生成三张对比图。

运行:python run_demo.py

生成:
  1) cost_vs_msgsize.png —— 耗时 vs 消息大小(log-log),看"小消息 tree 赢、大消息 ring 赢"的交叉
  2) cost_vs_P.png       —— 耗时 vs 卡数 P,分别看小消息(延迟主导)与大消息(带宽主导)
  3) best_algo_map.png   —— (P, 消息大小) 平面上"谁最快"的分区图 + 总线带宽 busbw 曲线
"""
import sys
try:                                                    # Windows 终端默认 GBK,统一切 UTF-8 免乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")                                   # 无显示环境后端
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from collective_cost import (
    allreduce_time, bus_bandwidth, crossover_size, best_algo,
    ALGOS, DEFAULT_NET,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

NET = DEFAULT_NET                                        # α=5μs, 100 GB/s
COLORS = {"ring": "#C44E52", "tree": "#4C72B0", "double_binary_tree": "#55A868"}
LABELS = {"ring": "ring(环)", "tree": "tree(朴素树)",
          "double_binary_tree": "double-binary-tree(NCCL)"}


def _fmt_bytes(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024 or u == "GB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def print_summary():
    print("=" * 68)
    print(f"AllReduce α-β 代价模型   (α={NET.alpha*1e6:.0f}μs, 带宽={NET.bandwidth_gbps:.0f}GB/s)")
    print("=" * 68)
    print(f"{'算法':<26}{'步数公式':<14}{'通信量公式':<16}")
    for k, a in ALGOS.items():
        print(f"{LABELS[k]:<24}{a.steps_formula:<14}{a.bytes_formula:<16}")
    print("-" * 68)
    for P in [8, 64]:
        nx = crossover_size(P, NET.alpha, NET.beta, "ring", "tree")
        print(f"P={P:<4} ring↔tree 交叉点 ≈ {_fmt_bytes(nx):<8}"
              f"(小于它 tree 赢、大于它 ring 赢)")
    print("-" * 68)
    print(f"{'消息':<10}{'P':<6}{'ring(ms)':<12}{'tree(ms)':<12}{'DBT(ms)':<12}{'最优':<12}")
    for N in [1e3, 1e5, 1e7, 1e9]:
        for P in [8, 64]:
            tr = allreduce_time("ring", N, P, NET.alpha, NET.beta) * 1e3
            tt = allreduce_time("tree", N, P, NET.alpha, NET.beta) * 1e3
            td = allreduce_time("double_binary_tree", N, P, NET.alpha, NET.beta) * 1e3
            win = LABELS[best_algo(N, P, NET.alpha, NET.beta)].split("(")[0]
            print(f"{_fmt_bytes(N):<10}{P:<6}{tr:<12.4f}{tt:<12.4f}{td:<12.4f}{win:<12}")


# --------------------------------------------------------------------------
# 图 1:耗时 vs 消息大小(两块子图:P=8 与 P=64),展示 ring↔tree 交叉
# --------------------------------------------------------------------------
def fig_cost_vs_msgsize():
    sizes = np.logspace(3, 9, 300)                      # 1 KB ~ 1 GB
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, P in zip(axes, [8, 64]):
        for k in ALGOS:
            t = [allreduce_time(k, N, P, NET.alpha, NET.beta) * 1e3 for N in sizes]
            ls = "--" if k == "double_binary_tree" else "-"
            ax.loglog(sizes, t, ls, color=COLORS[k], lw=2.4, label=LABELS[k])
        nx = crossover_size(P, NET.alpha, NET.beta, "ring", "tree")
        ax.axvline(nx, color="gray", ls=":", lw=1.3)
        ax.text(nx * 1.15, ax.get_ylim()[0] * 3,
                f"交叉≈{_fmt_bytes(nx)}\n(左 tree赢 | 右 ring赢)",
                color="gray", fontsize=9)
        ax.set_title(f"P = {P} 卡", fontsize=12, weight="bold")
        ax.set_xlabel("消息大小 N(字节,log)")
        ax.set_ylabel("AllReduce 耗时(ms,log)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("AllReduce 耗时 vs 消息大小:小消息 tree 赢、大消息 ring 赢、DBT 两全其美",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("cost_vs_msgsize.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图 2:耗时 vs 卡数 P(左:小消息延迟主导;右:大消息带宽主导)
# --------------------------------------------------------------------------
def fig_cost_vs_P():
    Ps = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scenarios = [(1e3, "小消息 1KB(延迟主导 → 步数越少越好)"),
                 (1e9, "大消息 1GB(带宽主导 → 通信量越小越好)")]
    for ax, (N, title) in zip(axes, scenarios):
        for k in ALGOS:
            t = [allreduce_time(k, N, P, NET.alpha, NET.beta) * 1e3 for P in Ps]
            ls = "--" if k == "double_binary_tree" else "-"
            ax.semilogx(Ps, t, ls, marker="o", ms=4, color=COLORS[k], lw=2.2,
                        base=2, label=LABELS[k])
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("卡数 P(log2)")
        ax.set_ylabel("AllReduce 耗时(ms)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("AllReduce 耗时 vs 卡数 P:ring 延迟 O(P) 爆炸,tree/DBT 只 O(log P)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("cost_vs_P.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图 3:左=(P, N) 平面"谁最快"分区图;右=busbw vs 消息大小
# --------------------------------------------------------------------------
def fig_best_algo_map():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左:排除 DBT,比较 ring vs tree 的胜负分区(DBT 恒赢,画它就全绿没信息) ---
    Ps = np.unique(np.round(np.logspace(np.log2(2), np.log2(1024), 40, base=2))).astype(int)
    sizes = np.logspace(3, 9, 40)
    idx = {"ring": 0, "tree": 1}
    Z = np.zeros((len(sizes), len(Ps)))
    for j, P in enumerate(Ps):
        for i, N in enumerate(sizes):
            two = min(["ring", "tree"],
                      key=lambda k: allreduce_time(k, N, int(P), NET.alpha, NET.beta))
            Z[i, j] = idx[two]
    ax = axes[0]
    cmap = ListedColormap([COLORS["ring"], COLORS["tree"]])
    ax.pcolormesh(Ps, sizes, Z, cmap=cmap, shading="auto", alpha=0.85)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    # 叠加 ring↔tree 交叉曲线
    nx = [crossover_size(int(P), NET.alpha, NET.beta, "ring", "tree") for P in Ps]
    ax.plot(Ps, nx, "k-", lw=2, label="ring / tree 交叉线")
    ax.set_title("ring vs tree 胜负分区(上=大消息 ring 赢,下=小消息 tree 赢)",
                 fontsize=10.5, weight="bold")
    ax.set_xlabel("卡数 P(log2)"); ax.set_ylabel("消息大小 N(字节,log)")
    ax.legend(loc="lower right", fontsize=9)
    ax.text(3, 3e8, "ring 赢", color="white", fontsize=12, weight="bold")
    ax.text(300, 3e3, "tree 赢", color="white", fontsize=12, weight="bold")

    # --- 右:总线带宽 busbw vs 消息大小(P=64),看谁吃满硬件 ---
    ax = axes[1]
    sizes2 = np.logspace(3, 9, 200)
    P = 64
    for k in ALGOS:
        bw = [bus_bandwidth(k, N, P, NET.alpha, NET.beta) for N in sizes2]
        ls = "--" if k == "double_binary_tree" else "-"
        ax.semilogx(sizes2, bw, ls, color=COLORS[k], lw=2.4, label=LABELS[k])
    ax.axhline(NET.bandwidth_gbps, color="gray", ls=":", lw=1.2)
    ax.text(1.2e3, NET.bandwidth_gbps * 0.94, "链路峰值", color="gray", fontsize=9)
    ax.set_title(f"总线带宽 busbw vs 消息大小(P={P}):大消息才吃满硬件",
                 fontsize=10.5, weight="bold")
    ax.set_xlabel("消息大小 N(字节,log)"); ax.set_ylabel("busbw(GB/s)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9)

    fig.suptitle("谁最快 & 谁吃满硬件", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("best_algo_map.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    print_summary()
    fig_cost_vs_msgsize()
    fig_cost_vs_P()
    fig_best_algo_map()
    print("\n已生成 3 张图:cost_vs_msgsize.png / cost_vs_P.png / best_algo_map.png")
    print("解读:")
    print("  · 小消息(1KB):α·steps 主导 → tree/DBT 的 2·log2(P) 步碾压 ring 的 2(P-1) 步。")
    print("  · 大消息(1GB):β·bytes 主导 → ring/DBT 的 2(P-1)/P·N 碾压 tree 的 2·log2(P)·N。")
    print("  · double-binary-tree = 树的对数延迟 + 环的最优带宽 → 全区间 Pareto 最优(NCCL 大集群默认)。")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
