"""
run_demo.py —— 跑一个到达流,对比"静态批处理 vs 连续批处理",打印指标并出两张图。
运行:python run_demo.py

产出:
  · 终端:两种策略的 吞吐 / GPU 利用率 / 空闲 slot·步 / 平均&尾延迟 / 平均排队 对比表。
  · timeline.png:两种策略的 GPU 占用(活跃 slot 数)随时间变化的时间线 + 请求甘特图。
  · metrics.png :吞吐 / 利用率 / 平均&p99 延迟 / 平均排队 的柱状对比。
"""
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 无显示环境,用 Agg 后端
import matplotlib.pyplot as plt

from continuous_batching import (
    Request, simulate_static, simulate_continuous, summarize,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BATCH = 4
C_STATIC, C_CONT = "#C44E52", "#55A868"     # 红=静态,绿=连续


def make_workload(n=60, seed=0):
    """混合到达流:大多数短输出(交互),少数长输出(长文)→ 批内长度方差大,最能体现差异。"""
    rnd = random.Random(seed)
    reqs = []
    for i in range(n):
        if rnd.random() < 0.25:
            out = rnd.randint(60, 110)               # 长生成
        else:
            out = rnd.randint(2, 12)                 # 短交互
        reqs.append(Request(i, arrival=rnd.uniform(0, 30),
                            prompt_len=rnd.randint(16, 256), output_len=out))
    return reqs


def print_table(s, c):
    print("=" * 78)
    print(f"静态批处理 vs 连续批处理(batch_size={BATCH},60 条混合到达流)")
    print("=" * 78)
    cols = [
        ("吞吐(tok/s)", "throughput_tok_s", "{:.1f}"),
        ("GPU利用率", "gpu_util", "{:.1%}"),
        ("空闲slot·步", "idle_slot_steps", "{:.0f}"),
        ("makespan(ms)", "makespan_ms", "{:.0f}"),
        ("平均延迟(ms)", "mean_latency", "{:.1f}"),
        ("p99延迟(ms)", "p99_latency", "{:.1f}"),
        ("平均排队(ms)", "mean_queue", "{:.1f}"),
    ]
    print(f"{'指标':<16}{'静态':>14}{'连续':>14}{'连续/静态':>14}")
    for label, key, fmt in cols:
        vs, vc = s[key], c[key]
        ratio = (vc / vs) if vs else float("nan")
        print(f"{label:<16}{fmt.format(vs):>14}{fmt.format(vc):>14}{ratio:>13.2f}x")
    print("-" * 78)
    print(f"结论:连续批处理吞吐 ×{c['throughput_tok_s']/s['throughput_tok_s']:.2f}、"
          f"利用率 {s['gpu_util']:.0%}→{c['gpu_util']:.0%}、"
          f"p99 延迟 {s['p99_latency']:.0f}→{c['p99_latency']:.0f}ms。")


def occupancy_series(res):
    """把 res.occupancy(逐运行步)映射到墙钟时间轴,返回 (t_ms, active)。"""
    t0 = res.first_start
    xs = [t0 + i * res.step_ms for i in range(len(res.occupancy))]
    return xs, list(res.occupancy)


def plot_timeline(rs, rc, path="timeline.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    # ---- 上排:GPU 占用时间线(活跃 slot 数)----
    for ax, res, color in ((axes[0, 0], rs, C_STATIC), (axes[0, 1], rc, C_CONT)):
        xs, ys = occupancy_series(res)
        ax.fill_between(xs, ys, step="post", color=color, alpha=0.65)
        ax.axhline(BATCH, color="gray", ls="--", lw=1)
        ax.text(xs[0] if xs else 0, BATCH + 0.05, f"batch 上限={BATCH}",
                fontsize=9, color="gray")
        ax.set_ylim(0, BATCH + 1)
        ax.set_ylabel("活跃 slot 数")
        ax.set_xlabel("时间 (ms)")
        util = res.gpu_util()
        ax.set_title(f"{res.name}:GPU 占用时间线(利用率 {util:.0%})",
                     fontsize=12, weight="bold")

    # ---- 下排:请求甘特图(每条请求占 slot 的区间)----
    for ax, res, color in ((axes[1, 0], rs, C_STATIC), (axes[1, 1], rc, C_CONT)):
        reqs = sorted(res.requests, key=lambda r: r.start)
        for row, r in enumerate(reqs):
            # 排队段(灰):arrival→start;服务段(彩):start→done
            ax.barh(row, r.start - r.arrival, left=r.arrival,
                    color="#BBBBBB", height=0.8)
            ax.barh(row, r.done - r.start, left=r.start,
                    color=color, height=0.8)
        ax.set_ylabel("请求(按开始时刻排序)")
        ax.set_xlabel("时间 (ms)")
        ax.set_title(f"{res.name}:请求甘特图(灰=排队,彩=服务)", fontsize=11)
        ax.set_xlim(0, max(rs.makespan, rc.makespan) * 1.02)

    fig.suptitle("静态批处理 vs 连续批处理:GPU 占用与请求生命周期",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics(s, c, path="metrics.png"):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    groups = [
        ("吞吐 (tok/s)", "throughput_tok_s", 1.0),
        ("GPU 利用率", "gpu_util", 100.0),        # 显示为百分比
        ("延迟 (ms)", None, 1.0),                  # 特殊:平均+p99 两根
        ("平均排队 (ms)", "mean_queue", 1.0),
    ]
    for ax, (title, key, scale) in zip(axes, groups):
        if title.startswith("延迟"):
            x = np.arange(2)
            ax.bar(x - 0.18, [s["mean_latency"], s["p99_latency"]], width=0.36,
                   label="静态", color=C_STATIC)
            ax.bar(x + 0.18, [c["mean_latency"], c["p99_latency"]], width=0.36,
                   label="连续", color=C_CONT)
            ax.set_xticks(x)
            ax.set_xticklabels(["平均", "p99"])
            ax.legend(fontsize=9)
        else:
            vals = [s[key] * scale, c[key] * scale]
            bars = ax.bar(["静态", "连续"], vals, color=[C_STATIC, C_CONT])
            for b, v in zip(bars, vals):
                suf = "%" if scale == 100.0 else ""
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f"{v:.0f}{suf}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=12, weight="bold")
    fig.suptitle("连续批处理:更高吞吐 / 更高利用率 / 更低延迟 / 更短排队",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    wl = make_workload()
    rs = simulate_static(wl, batch_size=BATCH)
    rc = simulate_continuous(wl, batch_size=BATCH)
    s, c = summarize(rs), summarize(rc)
    print_table(s, c)
    p1 = plot_timeline(rs, rc)
    p2 = plot_metrics(s, c)
    print(f"\n已生成图:{p1} 、 {p2}")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
