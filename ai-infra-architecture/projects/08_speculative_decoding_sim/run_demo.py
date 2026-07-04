"""
run_demo.py —— 投机解码期望加速的可视化 demo

跑法:python run_demo.py
产出(当前目录):
  1. speedup_vs_k.png     不同接受率 α 下,加速比随草稿长度 k 的曲线 + 最优点
  2. speedup_heatmap.png  加速比在 (α, k) 网格上的热图,含盈亏平衡线(=1)
  3. sim_vs_closed.png    蒙特卡洛平均敲定 token 数 与 闭式期望 的对拍

全程离线、CPU、无需模型/联网。matplotlib 用 Agg 后端 + YaHei 中文字体。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spec_decode import (
    expected_tokens, simulate_mean_tokens, expected_speedup,
    optimal_k, optimal_speedup, breakeven_alpha, speedup_grid, DEFAULT_C,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C = DEFAULT_C            # 草稿/目标成本比 = 0.2


# ---------------------------------------------------------------------------
def fig_speedup_vs_k(path="speedup_vs_k.png"):
    ks = np.arange(1, 21)
    alphas = [0.4, 0.6, 0.75, 0.9]
    colors = ["#8C8C8C", "#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for a, col in zip(alphas, colors):
        sp = [expected_speedup(a, int(k), C) for k in ks]
        ax.plot(ks, sp, "-o", ms=4, color=col, lw=2, label=f"α={a}")
        kstar = optimal_k(a, C, k_max=int(ks[-1]))
        ax.scatter([kstar], [expected_speedup(a, kstar, C)],
                   s=130, facecolors="none", edgecolors=col, lw=2.2, zorder=5)
    ax.axhline(1.0, color="red", ls="--", lw=1.2)
    ax.text(14.4, 1.03, "盈亏平衡线 speedup=1(下方=反而变慢)", color="red", fontsize=9)
    ax.set_xlabel("草稿长度 k(一轮猜几个 token)")
    ax.set_ylabel("期望加速比 speedup(相对普通自回归)")
    ax.set_title(f"投机解码加速比 vs 草稿长度 k(成本比 c={C};○=最优 k*)",
                 fontsize=12, weight="bold")
    ax.grid(True, alpha=0.25); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_heatmap(path="speedup_heatmap.png"):
    alphas = np.linspace(0.1, 0.98, 45)
    ks = np.arange(1, 17)
    grid = speedup_grid(alphas, ks, C)          # (α, k)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[ks[0] - 0.5, ks[-1] + 0.5, alphas[0], alphas[-1]],
                   vmin=0.5, vmax=grid.max())
    # 盈亏平衡等高线 speedup=1
    cs = ax.contour(ks, alphas, grid, levels=[1.0], colors="black", linewidths=2)
    ax.clabel(cs, fmt={1.0: "speedup=1"}, fontsize=9)
    # 每个 α 的最优 k* 轨迹
    kstar = [optimal_k(a, C, k_max=int(ks[-1])) for a in alphas]
    ax.plot(kstar, alphas, "b--", lw=1.8, label="最优 k*(α)")
    ax.set_xlabel("草稿长度 k")
    ax.set_ylabel("接受率 α")
    ax.set_title(f"期望加速比热图(成本比 c={C})", fontsize=12, weight="bold")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax, label="加速比 speedup")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_sim_vs_closed(path="sim_vs_closed.png"):
    ks = np.arange(1, 11)
    alphas = [0.5, 0.7, 0.9]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for a, col in zip(alphas, colors):
        exact = [expected_tokens(a, int(k)) for k in ks]
        mc = [simulate_mean_tokens(a, int(k), n_steps=120_000, seed=2025) for k in ks]
        ax.plot(ks, exact, "-", color=col, lw=2, label=f"闭式 α={a}")
        ax.plot(ks, mc, "x", color=col, ms=9, mew=2, label=f"模拟 α={a}")
    ax.set_xlabel("草稿长度 k")
    ax.set_ylabel("一轮期望敲定 token 数 E")
    ax.set_title("蒙特卡洛模拟 vs 闭式期望(两者重合=公式验证通过)",
                 fontsize=12, weight="bold")
    ax.grid(True, alpha=0.25); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    print("=" * 66)
    print("投机解码期望加速模拟(草稿-验证;成本比 c = %.2f)" % C)
    print("=" * 66)

    # 1) 闭式 vs 模拟 对拍(打印一张小表)
    print(f"{'α':>5}{'k':>4}{'闭式E':>10}{'模拟E':>10}{'|误差|':>10}")
    for a in [0.3, 0.6, 0.9]:
        for k in [2, 5, 8]:
            e = expected_tokens(a, k)
            m = simulate_mean_tokens(a, k, n_steps=200_000, seed=1)
            print(f"{a:>5.2f}{k:>4d}{e:>10.4f}{m:>10.4f}{abs(e-m):>10.4f}")

    # 2) 每个 α 的最优草稿长度 k* 与最优加速比
    print("\n不同接受率下的最优草稿长度 k* 与加速比:")
    print(f"{'α':>6}{'k*':>5}{'最优speedup':>14}{'盈亏平衡α(k=k*)':>18}")
    for a in [0.4, 0.6, 0.75, 0.9, 0.95]:
        ks = optimal_k(a, C, k_max=20)
        sp = optimal_speedup(a, C, k_max=20)
        be = breakeven_alpha(ks, C)
        print(f"{a:>6.2f}{ks:>5d}{sp:>14.3f}{be:>18.3f}")

    # 3) 出图
    figs = [fig_speedup_vs_k(), fig_heatmap(), fig_sim_vs_closed()]
    print("\n已生成配图:")
    for f in figs:
        print("  -", f)

    print("\n解读:")
    print("  · α 越高,曲线越高、最优草稿越长(k* 右移)—— 草稿越像大模型越赚。")
    print("  · 红/黑'=1'线以下是亏损区:α 太低、k 太大时白付草稿成本反而更慢。")
    print("  · 模拟均值与闭式期望几乎重合,验证 E=(1-α^{k+1})/(1-α) 正确。")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
