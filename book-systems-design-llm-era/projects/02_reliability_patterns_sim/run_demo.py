# -*- coding: utf-8 -*-
"""
run_demo.py —— 可靠性模式模拟器 · 可视化对比
=============================================

跑法：  python run_demo.py

产出（写在本目录 figures/ 下）：
    fig1_patterns_compare.png  —— 五种模式的 可用性 / P99尾延迟 / 平均延迟 对比柱状图
    fig2_availability_vs_failrate.png —— 随下游故障率上升，各模式可用性曲线
    fig3_retry_tradeoff.png    —— 重试次数 vs (成功率 / P99延迟) 的权衡曲线

并在终端打印一张对比表。全程离线、无随机漂移（固定 seed）。
"""

import matplotlib
matplotlib.use("Agg")  # 无界面后端：只出文件不弹窗，服务器/CI 友好
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体 + 负号正常显示（Windows 上 Microsoft YaHei，退化到 SimHei）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

import os
import reliability as R

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 全局仿真参数（可自行调大 N 看更平滑的曲线）
N = 800
FAIL_RATE = 0.35    # 下游 5xx 概率
SLOW_RATE = 0.25    # 长尾慢请求概率
SLOW_MS = 500       # 慢请求耗时
SEED = 2025


# ---------------------------------------------------------------------------
# 定义五种"模式"：从裸奔到全副武装
# ---------------------------------------------------------------------------
def build_modes():
    """返回 {模式名: 客户端工厂}，五种由弱到强的可靠性配置。

    这是一条**单调变好**的阶梯，每一级只加一样东西，方便观察"这样东西带来了什么"：

        ① 裸奔        —— 什么都没有，作为基线。
        ② 仅超时      —— 只加超时。注意：会把"慢"变成"失败"，**不配兜底时可用性反降**！
                         这是刻意保留的"坑演示"：超时必须搭配兜底/重试才有意义。
        ③ 超时+重试   —— 加重试。成功率回升，但看尾延迟怎么被推高。
        ④ +降级兜底   —— 加 fallback。可用性被"焊死"在 100%（下游全挂也不怕）。
        ⑤ 全副武装    —— 再加断路器。可用性同样 100%，但**尾延迟/负载显著下降**：
                         下游不健康时断路器 OPEN，直接 fast-fail 到兜底，
                         不再傻傻重试/等超时。这是断路器的真正价值。

    ⚠️ 教学要点：断路器**不提升**可用性（那是 fallback 的活），它降低的是
       "坏时段"的尾延迟和对下游的无效冲击。对比④⑤的 P99 就能看出来。
    """
    common = dict(fail_rate=FAIL_RATE, slow_rate=SLOW_RATE, slow_ms=SLOW_MS, seed=SEED)
    retry = R.RetryConfig(max_attempts=3, base_delay_ms=80, mode="capped")
    breaker = R.BreakerConfig(fail_threshold=5, cooldown_ms=2000)
    return {
        "① 裸奔\n(无保护)": R.make_factory(**common),
        "② 仅超时\n(反成坑)": R.make_factory(
            **common, enable_timeout=True, timeout_ms=200),
        "③ 超时+重试": R.make_factory(
            **common, enable_timeout=True, timeout_ms=200,
            enable_retry=True, retry_cfg=retry),
        "④ +降级兜底": R.make_factory(
            **common, enable_timeout=True, timeout_ms=200,
            enable_retry=True, enable_fallback=True, retry_cfg=retry),
        "⑤ 全副武装\n(+断路器)": R.make_factory(
            **common, enable_timeout=True, timeout_ms=200,
            enable_retry=True, enable_breaker=True, enable_fallback=True,
            retry_cfg=retry, breaker_cfg=breaker),
    }


def print_table(results):
    """在终端打印一张对比表。"""
    print("\n" + "=" * 78)
    print(f"{'模式':<20}{'可用性':>10}{'成功率':>10}{'P50(ms)':>10}"
          f"{'P99(ms)':>10}{'均延(ms)':>10}{'均尝试':>8}")
    print("-" * 78)
    for name, m in results.items():
        flat = name.replace("\n", " ")
        print(f"{flat:<20}{m.availability*100:>9.1f}%"
              f"{m.availability*100:>9.1f}%"
              f"{m.p50_latency:>10.1f}{m.p99_latency:>10.1f}"
              f"{m.avg_latency:>10.1f}{m.avg_attempts:>8.2f}")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# 图 1：五模式 三指标 对比
# ---------------------------------------------------------------------------
def plot_patterns_compare(results):
    names = [n.replace("\n", "\n") for n in results.keys()]
    avail = [m.availability * 100 for m in results.values()]
    p99 = [m.p99_latency for m in results.values()]
    avg = [m.avg_latency for m in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    bars0 = axes[0].bar(names, avail, color="#2e8b57")
    axes[0].set_title("可用性 Availability（越高越好）", fontsize=13)
    axes[0].set_ylabel("可用性 %")
    axes[0].set_ylim(0, 105)
    for b, v in zip(bars0, avail):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                     ha="center", fontsize=10)

    bars1 = axes[1].bar(names, p99, color="#c0504d")
    axes[1].set_title("P99 尾延迟（越低越好）", fontsize=13)
    axes[1].set_ylabel("延迟 ms")
    for b, v in zip(bars1, p99):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=10)

    bars2 = axes[2].bar(names, avg, color="#4472c4")
    axes[2].set_title("平均延迟（越低越好）", fontsize=13)
    axes[2].set_ylabel("延迟 ms")
    for b, v in zip(bars2, avg):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=10)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"可靠性模式对比（下游 fail_rate={FAIL_RATE}, slow_rate={SLOW_RATE}, N={N}）",
        fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(FIG_DIR, "fig1_patterns_compare.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 2：故障率扫描 —— 各模式可用性曲线
# ---------------------------------------------------------------------------
def plot_availability_vs_failrate():
    fail_rates = [i / 20 for i in range(0, 21)]  # 0.0 ~ 1.0 步进 0.05

    def curve(**kw):
        ys = []
        for fr in fail_rates:
            factory = R.make_factory(fail_rate=fr, slow_rate=0.0, seed=SEED, **kw)
            m = R.simulate(factory, 400)
            ys.append(m.availability * 100)
        return ys

    y_bare = curve()
    y_retry = curve(enable_retry=True,
                    retry_cfg=R.RetryConfig(max_attempts=4, mode="capped"))
    y_fb = curve(enable_fallback=True)
    y_all = curve(enable_retry=True, enable_breaker=True, enable_fallback=True,
                  retry_cfg=R.RetryConfig(max_attempts=4, mode="capped"),
                  breaker_cfg=R.BreakerConfig(fail_threshold=5, cooldown_ms=2000))

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [f * 100 for f in fail_rates]
    ax.plot(xs, y_bare, "o-", label="裸奔（无保护）", color="#888888")
    ax.plot(xs, y_retry, "s-", label="仅重试", color="#4472c4")
    ax.plot(xs, y_fb, "^-", label="仅降级兜底", color="#2e8b57")
    ax.plot(xs, y_all, "D-", label="全副武装", color="#c0504d", linewidth=2.5)

    ax.set_xlabel("下游故障率 fail_rate（%）", fontsize=12)
    ax.set_ylabel("用户视角可用性（%）", fontsize=12)
    ax.set_title("故障率越高，可靠性模式的价值越大\n"
                 "（注意：只有 fallback 能在下游全挂时保住可用性）", fontsize=13)
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_availability_vs_failrate.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 3：重试次数的权衡 —— 成功率↑ 但 P99↑
# ---------------------------------------------------------------------------
def plot_retry_tradeoff():
    attempts_list = [1, 2, 3, 4, 5, 6]
    avails, p99s = [], []
    for a in attempts_list:
        factory = R.make_factory(
            fail_rate=0.5, slow_rate=0.0, seed=SEED,
            enable_retry=(a > 1),
            retry_cfg=R.RetryConfig(max_attempts=a, base_delay_ms=100,
                                    mode="capped", jitter=False),
        )
        m = R.simulate(factory, 500)
        avails.append(m.availability * 100)
        p99s.append(m.p99_latency)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "#2e8b57"
    ax1.set_xlabel("最大尝试次数 max_attempts", fontsize=12)
    ax1.set_ylabel("可用性（%）", color=color1, fontsize=12)
    ax1.plot(attempts_list, avails, "o-", color=color1, linewidth=2, label="可用性")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#c0504d"
    ax2.set_ylabel("P99 尾延迟（ms）", color=color2, fontsize=12)
    ax2.plot(attempts_list, p99s, "s--", color=color2, linewidth=2, label="P99延迟")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("重试的权衡：成功率↑（收益递减），但尾延迟↑（线性增长）\n"
                  "fail_rate=0.5，这就是'用户在等时 60 秒重试=宕机'的量化", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_retry_tradeoff.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    print("正在运行可靠性模式仿真……（离线、确定性）")
    modes = build_modes()
    results = {name: R.simulate(factory, N) for name, factory in modes.items()}
    print_table(results)

    p1 = plot_patterns_compare(results)
    p2 = plot_availability_vs_failrate()
    p3 = plot_retry_tradeoff()

    print("已生成图表：")
    for p in (p1, p2, p3):
        print("  -", p)
    print("\n结论速览：")
    print("  · 仅超时会把'慢'变成'失败'——若不配兜底/重试，可用性反而下降（②，坑！）。")
    print("  · 重试提升成功率，但推高尾延迟、放大下游负载（③）。")
    print("  · 只有降级兜底(fallback)能在下游全挂时把可用性焊在 100%（④）。")
    print("  · 断路器不提升可用性，但在坏时段快速失败，压低尾延迟/保护下游（对比④⑤的P99）。")


if __name__ == "__main__":
    main()
