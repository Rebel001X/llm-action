# -*- coding: utf-8 -*-
"""
run_demo.py —— 分布式推理服务仿真:可视化 Demo
================================================

跑三组实验并出图(全部离线,不联网、不加载模型):

  图 1  scaling_throughput.png —— 副本数 vs 饱和吞吐(near-linear + 边际收益)
  图 2  latency_vs_load.png    —— 到达率 vs p50/p95/p99 尾延迟(负载升尾延迟爆炸)
  图 3  batching_compare.png   —— 连续 vs 静态批处理(吞吐/尾延迟/槽位利用率)
  图 4  utilization_curve.png  —— 固定需求下,副本数 vs 槽位利用率(过度扩容浪费)

运行:  python run_demo.py
输出:  当前目录下 4 个 .png,并在终端打印指标表。

matplotlib 强制 Agg 后端(无窗口、可在无显示器/CI 环境出图),
中文字体用微软雅黑,负号正常显示。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端,必须在 pyplot 之前设置
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体与负号:Windows 上微软雅黑;缺失则退回黑体。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import serving_sim as ss  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def fig1_scaling_throughput() -> str:
    """图 1:副本数 vs 饱和吞吐。制造饱和(极高到达率)以测容量上限。"""
    replicas = [1, 2, 3, 4, 6, 8]
    thr_tok = []
    for nr in replicas:
        m = ss.simulate(
            n_replicas=nr, rate=1000.0, duration=8.0, max_batch=16, seed=1
        )
        thr_tok.append(m["throughput_tok"])

    # 理想线性参考线:以单副本吞吐为斜率。
    ideal = [thr_tok[0] * nr for nr in replicas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(replicas, thr_tok, "o-", lw=2, ms=8, color="#2E86DE",
            label="实测饱和吞吐 (tok/s)")
    ax.plot(replicas, ideal, "--", lw=1.5, color="#95A5A6",
            label="理想线性扩展(参考)")
    ax.set_xlabel("副本数 (replicas)")
    ax.set_ylabel("饱和吞吐 (tokens/s)")
    ax.set_title("副本横向扩展:饱和吞吐近线性增长\n(无 batch_slowdown,负载均匀)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(HERE, "scaling_throughput.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig2_latency_vs_load() -> str:
    """图 2:到达率 vs 尾延迟。固定 2 副本,扫到达率,看 p50/p95/p99。"""
    rates = [10, 20, 30, 40, 50, 60, 70, 80]
    p50s, p95s, p99s = [], [], []
    for rate in rates:
        m = ss.simulate(
            n_replicas=2, rate=float(rate), duration=18.0,
            max_batch=16, seed=2,
        )
        p50s.append(m["p50"])
        p95s.append(m["p95"])
        p99s.append(m["p99"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rates, p50s, "o-", lw=2, label="p50 (中位延迟)", color="#27AE60")
    ax.plot(rates, p95s, "s-", lw=2, label="p95", color="#E67E22")
    ax.plot(rates, p99s, "^-", lw=2, label="p99 (尾延迟)", color="#C0392B")
    ax.set_xlabel("到达率 λ (req/s)  —— 2 副本")
    ax.set_ylabel("端到端延迟 (s)")
    ax.set_title("负载升高 → 尾延迟先缓后爆(接近饱和时排队激增)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(HERE, "latency_vs_load.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig3_batching_compare() -> str:
    """图 3:连续 vs 静态批处理,三项指标并列柱状。"""
    common = dict(
        n_replicas=1, rate=25.0, duration=20.0,
        max_batch=16, token_mean=64.0, seed=3,
    )
    cont = ss.simulate(continuous=True, **common)
    stat = ss.simulate(continuous=False, **common)

    labels = ["吞吐\n(req/s)", "p99 尾延迟\n(s)", "槽位利用率\n(%)"]
    cont_vals = [
        cont["throughput_req"],
        cont["p99"],
        cont["mean_slot_utilization"] * 100,
    ]
    stat_vals = [
        stat["throughput_req"],
        stat["p99"],
        stat["mean_slot_utilization"] * 100,
    ]

    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], cont_vals, w,
                label="连续批处理 (continuous)", color="#2E86DE")
    b2 = ax.bar([i + w / 2 for i in x], stat_vals, w,
                label="静态批处理 (static)", color="#E74C3C")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("指标值")
    ax.set_title("连续批处理 vs 静态批处理\n(同负载:连续吞吐更高、尾延迟更低、槽位更满)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    # 柱顶标数值。
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)
    out = os.path.join(HERE, "batching_compare.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig4_utilization_curve() -> str:
    """图 4:固定需求,副本数 vs 槽位利用率 + 吞吐(边际收益/过度扩容)。"""
    replicas = [1, 2, 3, 4, 6, 8]
    thr, util = [], []
    for nr in replicas:
        m = ss.simulate(
            n_replicas=nr, rate=60.0, duration=20.0, max_batch=16, seed=5
        )
        thr.append(m["throughput_req"])
        util.append(m["mean_slot_utilization"] * 100)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2E86DE"
    ax1.plot(replicas, thr, "o-", lw=2, color=color1, label="吞吐 (req/s)")
    ax1.set_xlabel("副本数 (replicas) —— 固定到达率 λ=60 req/s")
    ax1.set_ylabel("吞吐 (req/s)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#E67E22"
    ax2.plot(replicas, util, "s--", lw=2, color=color2,
             label="槽位利用率 (%)")
    ax2.set_ylabel("槽位利用率 (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("边际收益递减:副本够用后吞吐封顶,\n继续扩容只让槽位利用率下滑(浪费成本)")
    # 合并两个轴的图例。
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    out = os.path.join(HERE, "utilization_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def print_summary_table() -> None:
    """终端打印一张对比表,便于无图环境快速看结论。"""
    print("\n" + "=" * 64)
    print("分布式推理服务仿真 —— 指标速览")
    print("=" * 64)

    print("\n[1] 副本横向扩展(饱和,λ=1000):副本数 → 饱和吞吐(tok/s)")
    base = None
    for nr in (1, 2, 4, 8):
        m = ss.simulate(n_replicas=nr, rate=1000.0, duration=8.0, seed=1)
        t = m["throughput_tok"]
        if base is None:
            base = t
        print(f"   {nr} 副本: {t:8.0f} tok/s   (相对单副本 {t / base:4.2f}x)")

    print("\n[2] 负载 vs 尾延迟(2 副本):λ → p50 / p95 / p99 (s)")
    for rate in (10, 40, 70):
        m = ss.simulate(n_replicas=2, rate=float(rate), duration=18.0, seed=2)
        print(f"   λ={rate:3d}: p50={m['p50']:.3f}  "
              f"p95={m['p95']:.3f}  p99={m['p99']:.3f}")

    print("\n[3] 连续 vs 静态批处理(1 副本, λ=25):")
    common = dict(n_replicas=1, rate=25.0, duration=20.0,
                  max_batch=16, token_mean=64.0, seed=3)
    c = ss.simulate(continuous=True, **common)
    s = ss.simulate(continuous=False, **common)
    print(f"   连续: 吞吐 {c['throughput_req']:5.2f} req/s  "
          f"p99 {c['p99']:6.3f}s  槽位利用率 {c['mean_slot_utilization']:5.1%}")
    print(f"   静态: 吞吐 {s['throughput_req']:5.2f} req/s  "
          f"p99 {s['p99']:6.3f}s  槽位利用率 {s['mean_slot_utilization']:5.1%}")

    print("\n[4] 边际收益(固定 λ=60):副本数 → 吞吐 / 槽位利用率")
    for nr in (1, 2, 4, 8):
        m = ss.simulate(n_replicas=nr, rate=60.0, duration=20.0, seed=5)
        print(f"   {nr} 副本: 吞吐 {m['throughput_req']:5.2f} req/s   "
              f"槽位利用率 {m['mean_slot_utilization']:5.1%}")
    print("=" * 64)


def main() -> None:
    print("开始仿真并出图(Agg 后端,微软雅黑)...")
    outs = [
        fig1_scaling_throughput(),
        fig2_latency_vs_load(),
        fig3_batching_compare(),
        fig4_utilization_curve(),
    ]
    for p in outs:
        print(f"  已保存: {p}")
    print_summary_table()
    print("\n完成。共生成 4 张图。")


if __name__ == "__main__":
    main()
