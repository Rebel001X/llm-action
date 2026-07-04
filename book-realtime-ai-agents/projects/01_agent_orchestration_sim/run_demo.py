# -*- coding: utf-8 -*-
"""
run_demo.py —— 端到端演示
==========================

做两件事:
  1) 打印一条「反思重试」任务（flaky）的完整执行轨迹（规划→执行→反思→重试→成功）
  2) 批量仿真整套任务，画两张图存到 outputs/:
       - metrics.png    : 完成率 & 平均步数 & 轨迹准确率
       - reflection.png : 「无反思 vs 有反思」的完成率对比 + 分类型完成率

离线可跑：不联网、不调模型、不需要 key。
matplotlib 用 Agg 后端 + 微软雅黑，保证中文不乱码、无需显示器。

运行:
    python run_demo.py
"""

from __future__ import annotations

import os
import sys

# Windows 控制台默认 GBK，直接 print emoji/中文会 UnicodeEncodeError。
# 强制把标准输出改成 UTF-8，保证轨迹里的 🧭🛠️🤔 和中文都能打出来。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001  某些重定向环境可能没有 reconfigure，忽略即可
    pass

import matplotlib
matplotlib.use("Agg")  # 无界面后端：只出图片，不弹窗（服务器/CI 友好）
import matplotlib.pyplot as plt

# 中文字体 & 负号修复（Windows 上装了微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from agent import OrchestrationAgent
from tools import build_default_registry
from tasks import build_task_suite, Task
from simulation import run_simulation, compare_with_without_reflection


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ #
# A. 打印一条任务的完整轨迹
# ------------------------------------------------------------------ #
def print_trace(task: Task) -> None:
    reg = build_default_registry()
    agent = OrchestrationAgent(reg, max_steps=6, max_reflections=2)
    ep = agent.run(task)

    print("=" * 70)
    print(f"任务 {task.tid}: {task.prompt}")
    print(f"任务类型: {task.kind} | 标准答案: {task.expected!r}")
    print("-" * 70)
    for st in ep.trace:
        icon = {"plan": "🧭", "execute": "🛠️", "reflect": "🤔"}.get(st.phase, "·")
        line = f"[step {st.step}] {icon} {st.phase:<7}"
        if st.tool_name:
            line += f" tool={st.tool_name}"
        if st.thought:
            line += f" | 想法: {st.thought}"
        if st.result is not None:
            line += f" | 结果: {st.result}"
        if st.note:
            line += f" | {st.note}"
        print(line)
    print("-" * 70)
    print(f"最终: success={ep.success} | answer={ep.answer!r} | "
          f"steps={ep.steps} | reflections={ep.reflections} | "
          f"terminated={ep.terminated_reason}")
    print("=" * 70)
    print()


# ------------------------------------------------------------------ #
# B. 批量仿真 + 出图
# ------------------------------------------------------------------ #
def plot_metrics() -> str:
    sim = run_simulation()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：三大核心指标
    ax = axes[0]
    names = ["完成率\n(completion)", "轨迹准确率\n(trajectory acc)"]
    vals = [sim.completion_rate, sim.tool_trajectory_acc]
    bars = ax.bar(names, vals, color=["#4C72B0", "#55A868"])
    ax.set_ylim(0, 1.05)
    ax.set_title("批量仿真核心指标（越高越好）")
    ax.set_ylabel("比例")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", va="bottom", fontweight="bold")

    # 右图：分任务类型的完成率
    ax2 = axes[1]
    bk = sim.by_kind()
    kinds = list(bk.keys())
    rates = [bk[k] for k in kinds]
    bars2 = ax2.bar(kinds, rates, color="#C44E52")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"分任务类型完成率（平均步数 {sim.avg_steps:.2f}）")
    ax2.set_ylabel("完成率")
    for b, v in zip(bars2, rates):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                 ha="center", va="bottom")

    fig.suptitle("Agent 编排仿真 · 指标概览", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "metrics.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_reflection() -> str:
    cmp = compare_with_without_reflection()
    without = cmp["without_reflection"]
    with_ref = cmp["with_reflection"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["无反思\n(max_reflections=0)", "有反思\n(max_reflections=2)"]
    vals = [without.completion_rate, with_ref.completion_rate]
    bars = ax.bar(labels, vals, color=["#8C8C8C", "#4C72B0"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("完成率")
    ax.set_title("反思循环把完成率抬了上去（flaky 任务反思后重试成功）")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", va="bottom", fontweight="bold")
    # 画一条提升箭头
    ax.annotate("", xy=(1, vals[1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2))
    ax.text(0.5, (vals[0] + vals[1]) / 2 + 0.05,
            f"+{(vals[1]-vals[0]):.0%}", color="#C44E52",
            ha="center", fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT, "reflection.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
# C. main
# ------------------------------------------------------------------ #
def main() -> None:
    suite = build_task_suite()
    by_tid = {t.tid: t for t in suite}

    print("\n########## 演示 1：一条『失败→反思→重试→成功』任务的轨迹 ##########\n")
    print_trace(by_tid["t6"])  # flaky 任务

    print("########## 演示 2：一条普通计算任务的轨迹（对照）##########\n")
    print_trace(by_tid["t1"])  # 计算任务

    print("########## 演示 3：一条『无解→安全终止』任务的轨迹 ##########\n")
    print_trace(by_tid["t7"])  # unknown 任务

    print("########## 批量仿真统计 ##########\n")
    sim = run_simulation()
    print(f"任务总数        : {sim.n}")
    print(f"完成率          : {sim.completion_rate:.1%}")
    print(f"平均执行步数    : {sim.avg_steps:.2f}")
    print(f"工具轨迹准确率  : {sim.tool_trajectory_acc:.1%}")
    print(f"总工具调用次数  : {sim.total_tool_calls}")
    print(f"总反思次数      : {sim.total_reflections}")
    print(f"分类型完成率    : {sim.by_kind()}")

    cmp = compare_with_without_reflection()
    print("\n反思对比:")
    print(f"  无反思完成率 : {cmp['without_reflection'].completion_rate:.1%}")
    print(f"  有反思完成率 : {cmp['with_reflection'].completion_rate:.1%}")

    p1 = plot_metrics()
    p2 = plot_reflection()
    print(f"\n已保存图表:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
