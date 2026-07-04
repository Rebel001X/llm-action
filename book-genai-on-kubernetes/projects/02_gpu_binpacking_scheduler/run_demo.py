# -*- coding: utf-8 -*-
"""
GPU 装箱调度仿真 —— 可视化 Demo
================================

跑一批作业，对比 first-fit / best-fit / worst-fit 三种策略在
「利用率 / 碎片率 / 排队时延 / 被拒作业」上的差异，并出两张图：

  1. node_occupancy_heatmap.png —— 三策略下各节点的 GPU 占用热图
  2. strategy_compare.png       —— 四项指标柱状对比

离线运行：matplotlib 用 Agg 后端（不弹窗、不需 GUI），中文字体
Microsoft YaHei / SimHei。直接 `python run_demo.py` 即可。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # ⚠️ 必须在 import pyplot 之前设，否则无头环境会报错
import matplotlib.pyplot as plt
import numpy as np

# 中文字体 + 负号正常显示（Windows 常见坑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from scheduler import (
    Job,
    Scheduler,
    make_homogeneous_cluster,
    cluster_gpu_utilization,
)


# =============================================================================
# 1) 构造一个会「拉开策略差距」的作业负载
# =============================================================================

def build_tight_batch(seed: int = 27, n_jobs: int = 18) -> list[Job]:
    """静态装箱用：一批「总需求逼近集群容量」的作业（大小混合）。

    刻意让集群'差不多刚好装满'，这样谁碎片多、谁就会先放不下 → 拉开
    first-fit / best-fit / worst-fit 的差距（best-fit 靠紧凑装箱多塞一个）。

    这批作业不带到达时间/时长（静态快照，不释放）。
    """
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_jobs):
        gpus = int(rng.choice([1, 2, 3, 5], p=[0.25, 0.3, 0.25, 0.2]))
        jobs.append(Job(job_id=i, gpus=gpus, cpu=1.0, mem_gb=1.0))
    return jobs


def build_stream(seed: int = 7, n_jobs: int = 50) -> list[Job]:
    """离散事件仿真用：到达时间错开、含运行时长的作业流（会占用/释放）。"""
    rng = np.random.default_rng(seed)
    jobs = []
    t = 0.0
    for i in range(n_jobs):
        gpus = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.3, 0.2, 0.15]))
        jobs.append(Job(
            job_id=i,
            gpus=gpus,
            cpu=float(rng.choice([2, 4, 8])),
            mem_gb=float(rng.choice([8, 16, 32])),
            duration=float(rng.integers(20, 40)),
            arrival=t,
        ))
        t += float(rng.uniform(0.1, 0.6))  # 到达紧凑 → 制造资源争抢
    return jobs


def clone_jobs(jobs: list[Job]) -> list[Job]:
    """深拷贝一批作业（清掉运行期回填字段），保证每种策略从同一起点开始。"""
    return [
        Job(job_id=j.job_id, gpus=j.gpus, gpu_mem_gb=j.gpu_mem_gb,
            cpu=j.cpu, mem_gb=j.mem_gb, duration=j.duration, arrival=j.arrival)
        for j in jobs
    ]


# =============================================================================
# 2) 图一：节点占用热图（静态装箱快照，最直观看碎片）
# =============================================================================

def plot_occupancy_heatmap(strategies, jobs, n_nodes, gpus_per_node, out_path):
    """三种策略各跑一次静态装箱，画出各节点的 GPU 占用条。

    横轴 = 节点，纵轴 = 策略；颜色深浅 = 该节点已用 GPU 数。
    一眼看出：first-fit / worst-fit 会留下更多'半满'节点（浅色斑块 = 碎片），
    best-fit 更倾向把节点'压满或压空'。
    """
    fig, axes = plt.subplots(len(strategies), 1,
                             figsize=(11, 2.4 * len(strategies)))
    if len(strategies) == 1:
        axes = [axes]
    # 拉开子图间距，避免上一格的横轴标签压到下一格标题
    fig.subplots_adjust(hspace=0.75)

    for k, (ax, strat) in enumerate(zip(axes, strategies)):
        nodes = make_homogeneous_cluster(
            n_nodes, gpus_per_node=gpus_per_node, cpu_per_node=64, mem_per_node=512)
        res = Scheduler(nodes, strat).schedule_static(clone_jobs(jobs))

        used = np.array([[n.used_gpus for n in nodes]])
        im = ax.imshow(used, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=gpus_per_node)
        ax.set_yticks([0])
        ax.set_yticklabels([f"{strat}"])
        ax.set_xticks(range(n_nodes))
        # 只在最下面一格显示节点标签，其余隐藏，避免与标题重叠
        if k == len(strategies) - 1:
            ax.set_xticklabels([f"节点{i}" for i in range(n_nodes)], fontsize=9)
        else:
            ax.set_xticklabels([""] * n_nodes)

        # 在每个格子里标注 "已用/总"
        for j, n in enumerate(nodes):
            txt_color = "black" if n.used_gpus < gpus_per_node * 0.6 else "white"
            ax.text(j, 0, f"{n.used_gpus:.0f}/{gpus_per_node}",
                    ha="center", va="center", color=txt_color, fontsize=9)

        util = cluster_gpu_utilization(nodes)
        # 数半满节点数（碎片来源）
        half = sum(1 for n in nodes if 0 < n.used_gpus < n.gpus)
        ax.set_title(
            f"{strat}  —  放下 {res.placed} 个 / 拒 {res.rejected} 个，"
            f"利用率 {util:.0%}，半满(碎片)节点 {half} 个，碎片率 {res.fragmentation:.0%}",
            fontsize=11, loc="left")

    fig.suptitle("GPU 节点占用热图：不同装箱策略的碎片对比\n"
                 "（颜色越深=越满；出现'半满'节点即产生碎片）",
                 fontsize=13, y=1.0)
    fig.colorbar(im, ax=axes, label="已用 GPU 数", shrink=0.6)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1] 已保存节点占用热图 -> {out_path}")


# =============================================================================
# 3) 图二：四项指标柱状对比（离散事件仿真，更真实）
# =============================================================================

def plot_strategy_compare(strategies, jobs, n_nodes, gpus_per_node, out_path):
    """跑离散事件仿真，对比四项关键指标。"""
    results = {}
    for strat in strategies:
        nodes = make_homogeneous_cluster(
            n_nodes, gpus_per_node=gpus_per_node, cpu_per_node=64, mem_per_node=512)
        results[strat] = Scheduler(nodes, strat).simulate(clone_jobs(jobs))

    labels = strategies
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # (a) 峰值利用率
    peak_util = [results[s].peak_util * 100 for s in labels]
    axs[0, 0].bar(x, peak_util, color=colors)
    axs[0, 0].set_title("① 峰值 GPU 利用率（越高越省钱）")
    axs[0, 0].set_ylabel("利用率 %")
    _annotate(axs[0, 0], x, peak_util, fmt="{:.0f}%")

    # (b) 峰值碎片率
    peak_frag = [results[s].peak_frag * 100 for s in labels]
    axs[0, 1].bar(x, peak_frag, color=colors)
    axs[0, 1].set_title("② 峰值碎片率（越低越好，best-fit 应最低）")
    axs[0, 1].set_ylabel("碎片率 %")
    _annotate(axs[0, 1], x, peak_frag, fmt="{:.1f}%")

    # (c) 平均排队时延
    avg_wait = [results[s].avg_wait for s in labels]
    axs[1, 0].bar(x, avg_wait, color=colors)
    axs[1, 0].set_title("③ 平均排队时延（时间单位，越低越好）")
    axs[1, 0].set_ylabel("等待时长")
    _annotate(axs[1, 0], x, avg_wait, fmt="{:.1f}")

    # (d) 被拒作业数
    rejected = [results[s].rejected for s in labels]
    axs[1, 1].bar(x, rejected, color=colors)
    axs[1, 1].set_title("④ 被拒作业数（越低越好）")
    axs[1, 1].set_ylabel("作业数")
    _annotate(axs[1, 1], x, rejected, fmt="{:.0f}")

    for ax in axs.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("装箱策略四维对比（离散事件仿真）", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2] 已保存策略对比图 -> {out_path}")
    return results


def _annotate(ax, x, values, fmt="{:.0f}"):
    """在柱子顶端标数值。"""
    for xi, v in zip(x, values):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=10)


# =============================================================================
# 4) 主流程
# =============================================================================

def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))

    strategies = ["first_fit", "best_fit", "worst_fit"]

    # --- 图一用「紧凑静态批」：小集群 + 逼近满载，best-fit 优势最明显 ---
    tight_nodes, tight_gpus = 4, 8
    batch = build_tight_batch(seed=27, n_jobs=18)
    total_demand = sum(j.gpus for j in batch)
    print(f"[静态装箱] {len(batch)} 个作业，总需求 {total_demand} 张 GPU，"
          f"集群 {tight_nodes}×{tight_gpus}={tight_nodes * tight_gpus} 张（逼近满载）\n")
    plot_occupancy_heatmap(
        strategies, batch, tight_nodes, tight_gpus,
        os.path.join(here, "node_occupancy_heatmap.png"))

    # --- 图二用「作业流」：大集群 + 离散事件仿真（会释放资源） ---
    n_nodes, gpus_per_node = 5, 8
    jobs = build_stream(seed=7, n_jobs=50)
    print(f"\n[离散事件仿真] {len(jobs)} 个作业流，集群 {n_nodes}×{gpus_per_node}"
          f"={n_nodes * gpus_per_node} 张卡\n")
    results = plot_strategy_compare(
        strategies, jobs, n_nodes, gpus_per_node,
        os.path.join(here, "strategy_compare.png"))

    # 文本汇总表
    print("\n===== 仿真指标汇总 =====")
    print(f"{'策略':<12}{'放置':>6}{'被拒':>6}{'峰值利用率':>12}"
          f"{'峰值碎片率':>12}{'均等待':>10}{'最大等待':>10}")
    for s in strategies:
        r = results[s]
        print(f"{s:<12}{r.placed:>6}{r.rejected:>6}"
              f"{r.peak_util:>11.0%}{r.peak_frag:>11.1%}"
              f"{r.avg_wait:>10.2f}{r.max_wait:>10.2f}")

    # MIG 附加演示：同一批小作业，开/不开 MIG 能塞下多少
    print("\n===== MIG 切分收益演示（8 卡节点，48 个 10GB 小作业）=====")
    for mig in (False, True):
        nodes = make_homogeneous_cluster(1, gpus_per_node=8, cpu_per_node=256,
                                         mem_per_node=2048, mig_enabled=mig)
        small = [Job(job_id=i, gpus=1, gpu_mem_gb=10, cpu=1, mem_gb=1)
                 for i in range(48)]
        res = Scheduler(nodes, "best_fit").schedule_static(small)
        tag = "开启 MIG" if mig else "关闭 MIG"
        print(f"  {tag}: 8 张卡放下 {res.placed:>2}/48 个 10GB 作业，"
              f"GPU 利用率 {res.gpu_util:.0%}")
    print("  → MIG 把 8 张整卡切成 56 个 1/7 实例，小作业密度提升 7 倍。")

    print("\n完成。两张 PNG 已生成在项目目录下。")


if __name__ == "__main__":
    main()
