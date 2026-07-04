# -*- coding: utf-8 -*-
"""
run_demo.py —— 一键跑「扫 QPS」实验并出图。

输出：
  1) 命令行：每个 QPS 点的指标摘要（吞吐 / goodput / 利用率 / E2E 尾延迟）。
  2) 图 1  latency_throughput.png ：延迟-吞吐曲线（双子图）
        左：E2E p50/p95/p99 随 QPS 变化（尾延迟起飞的「拐点」一目了然）。
        右：吞吐 vs goodput vs 利用率随 QPS 变化（goodput 见顶后掉头）。
  3) 图 2  slo_attainment.png     ：SLO 达标比例随 QPS 单调下降。

跑法：  python run_demo.py
无需 GPU / 网络 / 模型。matplotlib 用 Agg 后端，直接落盘 PNG。
"""

import os

import matplotlib

# 必须在 import pyplot 之前设定：Agg 是无界面后端，服务器/本机无显示器也能出图
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体：优先微软雅黑，回退黑体；并修正负号显示为方块的问题
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from slo_harness import SLO, WorkloadConfig, sweep_qps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # ---- 实验配置 ----
    # 单请求服务时间 ≈ 1.7s（decode 主导）。用 32 个并发槽位（≈ 连续批处理里能同时
    # 在飞的序列数），理论容量 ≈ 32/1.7 ≈ 18.6 req/s，knee 落在 QPS≈16~18。
    num_workers = 32
    qps_list = [2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 22]
    base_cfg = WorkloadConfig(
        num_requests=5000,      # 请求数越多，p99 越可信
        mean_prompt_len=512,
        mean_output_len=128,
        prefill_ms_per_tok=0.30,
        decode_ms_per_tok=12.0,
        prefill_fixed_ms=8.0,
        seed=2024,
    )
    slo = SLO(ttft_ms=800.0, tpot_ms=20.0, e2e_ms=5000.0)

    print("=" * 78)
    print(f"扫 QPS 实验  |  workers={num_workers}  |  "
          f"SLO: TTFT<={slo.ttft_ms}ms TPOT<={slo.tpot_ms}ms E2E<={slo.e2e_ms}ms")
    print("=" * 78)

    metrics = sweep_qps(qps_list, base_cfg, slo, num_workers=num_workers)
    for m in metrics:
        print(m.summary_line())

    # ---- 抽取绘图用序列 ----
    qps = [m.qps_offered for m in metrics]
    p50 = [m.e2e_ms["p50"] for m in metrics]
    p95 = [m.e2e_ms["p95"] for m in metrics]
    p99 = [m.e2e_ms["p99"] for m in metrics]
    thr = [m.throughput for m in metrics]
    good = [m.goodput for m in metrics]
    util = [m.utilization * 100 for m in metrics]
    ratio = [m.goodput_ratio * 100 for m in metrics]

    # =========================================================
    # 图 1：延迟-吞吐曲线（左右双子图）
    # =========================================================
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # 左：E2E 延迟百分位随 QPS
    axL.plot(qps, p50, "o-", label="E2E p50", color="#2e7d32")
    axL.plot(qps, p95, "s-", label="E2E p95", color="#f9a825")
    axL.plot(qps, p99, "^-", label="E2E p99", color="#c62828")
    axL.axhline(slo.e2e_ms, ls="--", color="gray", label=f"E2E SLO={slo.e2e_ms:.0f}ms")
    axL.set_xlabel("QPS（到达率 λ，请求/秒）")
    axL.set_ylabel("E2E 延迟（毫秒）")
    axL.set_title("延迟随负载上升：尾延迟先起飞")
    axL.legend()
    axL.grid(True, alpha=0.3)

    # 右：吞吐 / goodput / 利用率随 QPS（双 y 轴）
    axR.plot(qps, thr, "o-", label="吞吐 throughput", color="#1565c0")
    axR.plot(qps, good, "s-", label="有效吞吐 goodput", color="#6a1b9a")
    axR.set_xlabel("QPS（到达率 λ，请求/秒）")
    axR.set_ylabel("吞吐（请求/秒）")
    axR.set_title("goodput 见顶后掉头：违反 SLO 的票作废")
    axR.grid(True, alpha=0.3)

    axR2 = axR.twinx()
    axR2.plot(qps, util, "^--", label="利用率 utilization(%)", color="#00838f")
    axR2.set_ylabel("利用率（%）")
    axR2.set_ylim(0, 105)

    # 合并两个 y 轴的图例
    lines1, labels1 = axR.get_legend_handles_labels()
    lines2, labels2 = axR2.get_legend_handles_labels()
    axR.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    out1 = os.path.join(HERE, "latency_throughput.png")
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"\n[图已保存] {out1}")

    # =========================================================
    # 图 2：SLO 达标比例随 QPS 单调下降
    # =========================================================
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qps, ratio, "o-", color="#ad1457", linewidth=2)
    ax.axhline(95, ls="--", color="gray", label="常见目标线 95%")
    ax.fill_between(qps, ratio, 0, alpha=0.12, color="#ad1457")
    ax.set_xlabel("QPS（到达率 λ，请求/秒）")
    ax.set_ylabel("SLO 达标比例 goodput_ratio（%）")
    ax.set_title("负载越高，越多请求违反 SLO（达标比例单调下降）")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "slo_attainment.png")
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    print(f"[图已保存] {out2}")

    # ---- 找「拐点」：goodput 达到最大的那个 QPS ----
    best = max(metrics, key=lambda m: m.goodput)
    print("\n" + "-" * 78)
    print(f"容量规划建议：goodput 最大点在 QPS≈{best.qps_offered:.0f}，"
          f"此时 goodput={best.goodput:.2f} req/s，"
          f"达标比例={best.goodput_ratio*100:.1f}%，"
          f"利用率={best.utilization*100:.1f}%。")
    print("超过该点后继续加压，吞吐几乎不涨、尾延迟起飞、goodput 反降 —— 得扩容或降级。")
    print("-" * 78)


if __name__ == "__main__":
    main()
