# -*- coding: utf-8 -*-
"""
run_demo.py —— 跑一遍估算器，打印典型场景报表 + 出两张图。

图 1：QPS 扫描 —— 成本($/月) 与 GPU 数 随 QPS 变化（验证线性 + ceil 阶梯）。
图 2：单场景成本拆解 + 模型/精度对比条形图。

离线可跑：matplotlib 用 Agg 后端（不弹窗，存 PNG）；中文用 Microsoft YaHei。
运行：  python run_demo.py
产物：  fig_qps_scan.png / fig_cost_breakdown.png（存在本目录）
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # ⚠️ 必须在 import pyplot 之前设，否则无头环境会试图找显示器而报错
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体 + 负号正常显示（Windows 常见坑：不设就是一堆方框 □□□）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimator import (  # noqa: E402
    DType,
    estimate_by_name,
    format_report,
    GPU_CATALOG,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def demo_reports():
    """打印三个典型场景的报表，直观感受不同规模的成本量级。"""
    print("\n########## 场景 A：中等聊天服务（8B / A100-80G）##########")
    a = estimate_by_name(qps=50, avg_input_tokens=1024, avg_output_tokens=256,
                         model_name="Llama3-8B", gpu_name="A100-80G")
    print(format_report(a))

    print("\n########## 场景 B：大模型 RAG（70B / H100-80G，长输入）##########")
    b = estimate_by_name(qps=20, avg_input_tokens=4096, avg_output_tokens=512,
                         model_name="Llama3-70B", gpu_name="H100-80G")
    print(format_report(b))

    print("\n########## 场景 C：70B + INT8 量化（省显存/省钱）##########")
    c = estimate_by_name(qps=20, avg_input_tokens=4096, avg_output_tokens=512,
                         model_name="Llama3-70B", gpu_name="H100-80G",
                         weight_dtype=DType.INT8, kv_dtype=DType.INT8)
    print(format_report(c))
    print(f"\n>>> 量化后月成本从 ${b.cost_per_month_usd:,.0f} 降到 "
          f"${c.cost_per_month_usd:,.0f}，省 "
          f"{(1 - c.cost_per_month_usd / b.cost_per_month_usd) * 100:.0f}%")


def fig_qps_scan():
    """图 1：QPS vs (月成本, GPU 数)，双 y 轴。"""
    qps_list = [5, 10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000]
    cost, gpus, unit = [], [], []
    for q in qps_list:
        e = estimate_by_name(qps=q, avg_input_tokens=1024, avg_output_tokens=256,
                             model_name="Llama3-8B", gpu_name="A100-80G")
        cost.append(e.cost_per_month_usd / 1000.0)  # 换成 千美元/月
        gpus.append(e.num_gpus)
        unit.append(e.cost_per_million_tokens_usd)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color1 = "#c0392b"
    ax1.set_xlabel("QPS（每秒请求数）")
    ax1.set_ylabel("月成本（千美元 / 月）", color=color1)
    ax1.plot(qps_list, cost, "o-", color=color1, label="月成本")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#2471a3"
    ax2.set_ylabel("所需 GPU 数（A100-80G）", color=color2)
    ax2.plot(qps_list, gpus, "s--", color=color2, label="GPU 数")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("图 1｜QPS 扫描：月成本 & GPU 数随 QPS 近似线性增长\n"
              "（Llama3-8B / A100-80G / 1024-in / 256-out）")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_qps_scan.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[已保存] {out}")
    print(f"          $/1M token 基本恒定（单位成本与 QPS 无关）: "
          f"{unit[2]:.4f} ~ {unit[-1]:.4f}")


def fig_cost_breakdown():
    """图 2：左=显存拆解（权重 vs KV），右=不同模型/精度的 $/1M token 对比。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # —— 左图：单卡显存拆解（堆叠条）——
    scenarios = [
        ("8B FP16", "Llama3-8B", DType.FP16, "A100-80G"),
        ("70B FP16", "Llama3-70B", DType.FP16, "H100-80G"),
        ("70B INT8", "Llama3-70B", DType.INT8, "H100-80G"),
    ]
    labels, weights, kvs = [], [], []
    for name, m, dt, g in scenarios:
        e = estimate_by_name(qps=20, avg_input_tokens=4096, avg_output_tokens=512,
                             model_name=m, gpu_name=g,
                             weight_dtype=dt, kv_dtype=dt)
        labels.append(name)
        weights.append(e.weight_gb)
        # 把并发 KV 摊到卡数上，得到单卡 KV
        kvs.append(e.kv_concurrent_gb / e.num_gpus)

    x = range(len(labels))
    axL.bar(x, weights, label="权重显存", color="#8e44ad")
    axL.bar(x, kvs, bottom=weights, label="KV Cache（单卡摊薄）", color="#e67e22")
    axL.set_xticks(list(x))
    axL.set_xticklabels(labels)
    axL.set_ylabel("单卡显存占用（GB）")
    axL.set_title("图 2a｜显存拆解：权重 vs KV Cache\n量化把两者一起砍")
    axL.legend()
    axL.grid(True, axis="y", alpha=0.3)

    # —— 右图：$/1M token 对比（不同模型 × GPU）——
    combos = [
        ("8B\nA100", "Llama3-8B", "A100-80G", DType.FP16),
        ("8B\nH100", "Llama3-8B", "H100-80G", DType.FP16),
        ("8B\nL40S", "Llama3-8B", "L40S-48G", DType.FP16),
        ("70B\nH100", "Llama3-70B", "H100-80G", DType.FP16),
        ("70B-INT8\nH100", "Llama3-70B", "H100-80G", DType.INT8),
    ]
    names, units = [], []
    for label, m, g, dt in combos:
        e = estimate_by_name(qps=50, avg_input_tokens=1024, avg_output_tokens=256,
                             model_name=m, gpu_name=g,
                             weight_dtype=dt, kv_dtype=dt)
        names.append(label)
        units.append(e.cost_per_million_tokens_usd)

    bars = axR.bar(names, units, color="#16a085")
    axR.set_ylabel("$ / 百万 token")
    axR.set_title("图 2b｜单位成本对比\n（选对模型+卡+精度，$/1M token 可差数十倍）")
    axR.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, units):
        axR.text(b.get_x() + b.get_width() / 2, v, f"${v:.2f}",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = os.path.join(HERE, "fig_cost_breakdown.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[已保存] {out}")


def main():
    demo_reports()
    print("\n" + "=" * 60)
    print("  正在生成图表……")
    print("=" * 60)
    fig_qps_scan()
    fig_cost_breakdown()
    print("\n✅ Demo 结束。两张 PNG 已生成在项目目录。")


if __name__ == "__main__":
    main()
