# -*- coding: utf-8 -*-
"""
run_demo.py —— 一键跑仿真并出图(离线、无 GPU、无网络)
========================================================

产出 3 张图到 ./figures/ :
    1) fig_timeline.png   : 负载 / 副本 / 延迟 三联时间线(核心叙事图)
    2) fig_tradeoff.png   : 成本-SLO 权衡曲线(扫不同 target)
    3) fig_metric_compare.png : queue / latency / gpu 三种触发指标对比

运行:  python run_demo.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # ⚠️ 必须在 import pyplot 之前:无显示环境用 Agg 后端出文件
import matplotlib.pyplot as plt
import numpy as np

# ⚠️ 中文字体 + 负号正常显示(Windows 常见坑)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from autoscale_sim import (
    SimConfig, run_sim, compute_metrics, simulate,
    trapezoid_load, spiky_load, diurnal_load,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _print_metrics(title, m):
    print(f"\n=== {title} ===")
    order = ["slo_ok_ratio", "p95_latency", "p99_latency", "max_latency",
             "pod_hours", "cost", "avg_overshoot", "flaps", "peak_pods",
             "mean_ready_pods", "throughput"]
    for k in order:
        v = m[k]
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")


# =============================================================================
# 图 1:三联时间线(负载 / 副本 / 延迟)—— 最核心的叙事图
# =============================================================================
def plot_timeline():
    cfg = SimConfig(arrival_fn=trapezoid_load, horizon=600.0)
    out = simulate(cfg)
    r, m = out["result"], out["metrics"]
    _print_metrics("图1 · 梯形负载时间线", m)

    t = r["t"]
    # 理论所需就绪副本(把负载正好压住的下界),用于对照过冲
    need = np.clip(np.ceil(r["lambda"] / cfg.per_pod_capacity),
                   cfg.min_replicas, cfg.max_replicas)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # --- (a) 负载:到达率 λ(t) ---
    ax = axes[0]
    ax.plot(t, r["lambda"], color="#1f77b4", lw=2, label="到达率 λ(t) (rps)")
    ax.fill_between(t, 0, r["lambda"], color="#1f77b4", alpha=0.12)
    ax.set_ylabel("到达率 (rps)")
    ax.set_title("① 负载:请求到达率随时间变化(梯形冲击波)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # --- (b) 副本:就绪 vs 期望 vs 理论所需 ---
    ax = axes[1]
    ax.plot(t, r["total_pods"], color="#ff7f0e", lw=2, label="总副本(含启动中)")
    ax.plot(t, r["ready_pods"], color="#2ca02c", lw=1.6, label="已就绪副本")
    ax.plot(t, need, color="#d62728", lw=1.4, ls="--", label="理论所需(下界)")
    ax.fill_between(t, r["ready_pods"], need,
                    where=(r["ready_pods"] > need), color="#ff7f0e", alpha=0.15,
                    label="过冲区(overshoot)")
    ax.set_ylabel("副本数 (pods)")
    ax.set_title(f"② 副本:自动扩缩容响应(峰值 {int(m['peak_pods'])} pods,"
                 f"平均过冲 {m['avg_overshoot']:.1f})")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # --- (c) 延迟:平均延迟 vs SLO 阈值 ---
    ax = axes[2]
    ax.plot(t, r["latency"], color="#9467bd", lw=1.8, label="平均延迟 (s)")
    ax.axhline(cfg.slo_latency, color="#d62728", ls="--", lw=1.5,
               label=f"SLO 阈值 = {cfg.slo_latency}s")
    # 标出违反 SLO 的时段
    viol = r["latency"] > cfg.slo_latency
    ax.fill_between(t, 0, r["latency"], where=viol, color="#d62728", alpha=0.25,
                    label="违反 SLO 时段")
    ax.set_ylabel("延迟 (s)")
    ax.set_xlabel("时间 (s)")
    ax.set_title(f"③ 延迟:SLO 满足率 {m['slo_ok_ratio']*100:.1f}% · "
                 f"p95={m['p95_latency']:.2f}s · 成本={m['cost']:.2f}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("K8s 自动扩缩容仿真:负载 → 副本 → 延迟 闭环时间线",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(FIG_DIR, "fig_timeline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[已保存] {path}")


# =============================================================================
# 图 2:成本-SLO 权衡曲线(扫不同目标利用率)
# =============================================================================
def plot_tradeoff():
    targets = [1, 2, 3, 5, 8, 12, 18, 25]  # 每 Pod 目标队列长度(越大越激进越省)
    costs, slos, overs, flaps = [], [], [], []
    for tm in targets:
        cfg = SimConfig(target_metric=float(tm), arrival_fn=trapezoid_load,
                        horizon=600.0, rng_seed=7, metric="queue")
        m = compute_metrics(run_sim(cfg), cfg)
        costs.append(m["cost"])
        slos.append(m["slo_ok_ratio"] * 100)
        overs.append(m["avg_overshoot"])
        flaps.append(m["flaps"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "#1f77b4"
    ax1.set_xlabel("目标每 Pod 队列长度(越大越激进 → 副本越少)")
    ax1.set_ylabel("SLO 满足率 (%)", color=color1)
    ln1 = ax1.plot(targets, slos, "o-", color=color1, lw=2, label="SLO 满足率 (%)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("成本 (pod·小时·单价)", color=color2)
    ln2 = ax2.plot(targets, costs, "s--", color=color2, lw=2, label="总成本")
    ax2.tick_params(axis="y", labelcolor=color2)

    lns = ln1 + ln2
    ax1.legend(lns, [l.get_label() for l in lns], loc="center right")
    ax1.set_title("成本-SLO 权衡:目标利用率越激进越省钱,但 SLO 会下降\n"
                  "(左上=贵而稳,右下=省而险,选型即在此曲线上找甜点)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_tradeoff.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[已保存] {path}")
    print("  权衡表: " + " | ".join(
        f"tgt={tt} SLO={ss:.1f}% cost={cc:.1f}"
        for tt, ss, cc in zip(targets, slos, costs)))


# =============================================================================
# 图 3:三种触发指标(queue / latency / gpu)对比
# =============================================================================
def plot_metric_compare():
    settings = [("queue", 5.0, "队列长度触发(KEDA 式)"),
                ("latency", 0.5, "延迟触发"),
                ("gpu", 0.7, "GPU 利用率触发")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for metric, target, label in settings:
        cfg = SimConfig(metric=metric, target_metric=target,
                        arrival_fn=spiky_load, horizon=600.0)
        r = run_sim(cfg)
        m = compute_metrics(r, cfg)
        axes[0].plot(r["t"], r["total_pods"], lw=1.8,
                     label=f"{label} (SLO {m['slo_ok_ratio']*100:.0f}%, "
                           f"成本 {m['cost']:.1f})")
        axes[1].plot(r["t"], r["latency"], lw=1.6, label=label)

    # 负载参考线(第二个 y 轴叠一条 λ)
    r0 = run_sim(SimConfig(arrival_fn=spiky_load, horizon=600.0))
    axL = axes[0].twinx()
    axL.plot(r0["t"], r0["lambda"], color="gray", ls=":", lw=1.2, alpha=0.7)
    axL.set_ylabel("到达率 λ (rps, 灰点线)", color="gray")

    axes[0].set_ylabel("总副本数 (pods)")
    axes[0].set_title("不同触发指标在'尖峰负载'下的扩容行为对比")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(1.0, color="#d62728", ls="--", lw=1.2, label="SLO=1.0s")
    axes[1].set_ylabel("平均延迟 (s)")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_title("对应的延迟表现(尖峰 + 启动延迟下谁更容易破 SLO)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("KEDA 式多指标触发对比:queue vs latency vs gpu",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(FIG_DIR, "fig_metric_compare.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[已保存] {path}")


def main():
    print("=" * 60)
    print("K8s 自动扩缩容仿真 · run_demo(离线出图)")
    print("=" * 60)
    plot_timeline()
    plot_tradeoff()
    plot_metric_compare()
    print("\n全部完成!图片在:", FIG_DIR)


if __name__ == "__main__":
    main()
