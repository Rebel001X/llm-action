# -*- coding: utf-8 -*-
"""
_gen_sched.py —— 生成《集群调度与编排》讲义配图(真实 PNG)。
运行:python _gen_sched.py  →  在本目录生成 sched_*.png
主题:gang scheduling / 拓扑感知放置 / 大规模训练故障率与 MTBF。
中文用 Microsoft YaHei。数字为主流千卡~万卡集群的量级参考。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C = dict(blue="#4C72B0", orange="#DD8452", green="#55A868", red="#C44E52",
         purple="#8172B3", gray="#8C8C8C", teal="#64B5CD", yellow="#E6C200",
         light="#E9EEF5", dark="#33383D")


def box(ax, x, y, w, h, text, fc, fs=11, tc="white", ec="none", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                                 color=color, lw=lw, linestyle=ls))


# ================================================================ 1) Gang scheduling 示意
def fig_gang():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.suptitle("Gang Scheduling(成组调度 · 全或无 All-or-Nothing)", fontsize=15, weight="bold")

    # 左:逐个调度 → 资源碎片 + 死锁
    a1.set_xlim(0, 10); a1.set_ylim(0, 10); a1.axis("off")
    a1.set_title("【差】逐 Pod 调度:碎片占用 → 互相死锁", fontsize=12, color=C["red"])
    # 4 台机器,每台 2 卡
    hosts = ["node1", "node2", "node3", "node4"]
    for i, hn in enumerate(hosts):
        x = 0.6 + i * 2.35
        a1.add_patch(Rectangle((x, 1.3), 1.9, 7.0, fc=C["light"], ec=C["gray"], lw=1.3))
        a1.text(x + 0.95, 8.55, hn, ha="center", fontsize=9.5, color=C["dark"])
        # 两个卡槽
        for k in range(2):
            a1.add_patch(Rectangle((x + 0.2, 1.7 + k * 3.1), 1.5, 2.7, fc="white", ec=C["gray"], lw=1))
    # Job A 抢到 4 张分散卡(蓝),Job B 抢到 4 张(橙),都缺 4 张 → 谁都不能启动
    jobA = [(0, 0), (0, 1), (1, 0), (2, 0)]
    jobB = [(1, 1), (2, 1), (3, 0), (3, 1)]
    for (i, k) in jobA:
        x = 0.6 + i * 2.35
        a1.add_patch(Rectangle((x + 0.2, 1.7 + k * 3.1), 1.5, 2.7, fc=C["blue"], ec="white", lw=1.5))
        a1.text(x + 0.95, 1.7 + k * 3.1 + 1.35, "A", ha="center", va="center", color="white", fontsize=12, weight="bold")
    for (i, k) in jobB:
        x = 0.6 + i * 2.35
        a1.add_patch(Rectangle((x + 0.2, 1.7 + k * 3.1), 1.5, 2.7, fc=C["orange"], ec="white", lw=1.5))
        a1.text(x + 0.95, 1.7 + k * 3.1 + 1.35, "B", ha="center", va="center", color="white", fontsize=12, weight="bold")
    a1.text(5.0, 0.55, "Job A 需 8 卡只抢到 4,Job B 同理 → 资源被瓜分,\n谁都攒不齐、谁都不释放 → 死锁(deadlock),利用率≈0",
            ha="center", fontsize=9.5, color=C["red"])

    # 右:gang scheduling → 全组一起上
    a2.set_xlim(0, 10); a2.set_ylim(0, 10); a2.axis("off")
    a2.set_title("【优】Gang:凑齐 8 卡才整组入场,否则一张不占", fontsize=12, color=C["green"])
    for i, hn in enumerate(hosts):
        x = 0.6 + i * 2.35
        a2.add_patch(Rectangle((x, 1.3), 1.9, 7.0, fc=C["light"], ec=C["gray"], lw=1.3))
        a2.text(x + 0.95, 8.55, hn, ha="center", fontsize=9.5, color=C["dark"])
    # Job A 整组占满 node1/node2(8卡),Job B 在队列里等 node3/node4 就绪
    for i in [0, 1]:
        x = 0.6 + i * 2.35
        for k in range(2):
            a2.add_patch(Rectangle((x + 0.2, 1.7 + k * 3.1), 1.5, 2.7, fc=C["blue"], ec="white", lw=1.5))
            a2.text(x + 0.95, 1.7 + k * 3.1 + 1.35, "A", ha="center", va="center", color="white", fontsize=12, weight="bold")
    for i in [2, 3]:
        x = 0.6 + i * 2.35
        for k in range(2):
            a2.add_patch(Rectangle((x + 0.2, 1.7 + k * 3.1), 1.5, 2.7, fc="white", ec=C["orange"], lw=1.6, ls="--"))
            a2.text(x + 0.95, 1.7 + k * 3.1 + 1.35, "B?", ha="center", va="center", color=C["orange"], fontsize=11)
    a2.text(5.0, 0.55, "Job A 整组(gang)一次到位 → 立即开跑;\nJob B 未凑齐 → 在队列等待,不占用资源(避免死锁)",
            ha="center", fontsize=9.5, color=C["green"])

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("sched_gang.png", bbox_inches="tight"); plt.close(fig)


# ================================================================ 2) 拓扑感知放置
def fig_topology():
    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    ax.set_xlim(0, 12.7); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("拓扑感知调度(Topology-Aware Placement):同作业的卡尽量同交换机 → 降通信跳数",
                 fontsize=13.5, weight="bold")

    # Spine
    box(ax, 5.0, 8.6, 2.0, 0.9, "Spine 交换机\n(核心)", C["purple"], 10)
    # 两个 Leaf(机架顶 ToR)
    box(ax, 1.6, 6.5, 2.0, 0.85, "Leaf/ToR-1\n机架 A", C["teal"], 9.5)
    box(ax, 8.4, 6.5, 2.0, 0.85, "Leaf/ToR-2\n机架 B", C["teal"], 9.5)
    arrow(ax, 5.6, 8.6, 3.0, 7.35, C["gray"], lw=1.6)
    arrow(ax, 6.4, 8.6, 9.0, 7.35, C["gray"], lw=1.6)

    # 每个机架 2 台机,每台机内 4 卡(NVLink 全互联)
    def rack(x0, leaf_cx, good=True):
        for m in range(2):
            hx = x0 + m * 2.6
            box(ax, hx, 3.6, 2.2, 1.9, "", C["light"], tc=C["dark"], ec=C["gray"])
            ax.text(hx + 1.1, 5.25, f"Host\nNVLink 域", ha="center", va="center", fontsize=8, color=C["dark"])
            for g in range(4):
                gx = hx + 0.18 + (g % 2) * 1.0
                gy = 3.75 + (g // 2) * 0.72
                fc = C["green"] if good else C["orange"]
                ax.add_patch(Rectangle((gx, gy), 0.85, 0.6, fc=fc, ec="white", lw=1))
                ax.text(gx + 0.42, gy + 0.3, "G", ha="center", va="center", color="white", fontsize=8)
            arrow(ax, hx + 1.1, 5.5, leaf_cx, 6.5, C["gray"], lw=1.2, ls="--")
    rack(0.6, 2.6, good=True)
    rack(7.4, 9.4, good=True)

    # 带宽/延迟标注
    ax.text(6.0, 7.9, "跨 Spine:~几十 μs / 带宽最紧张", ha="center", fontsize=9, color=C["purple"])
    ax.text(2.6, 6.05, "同 Leaf 跨机:RDMA ~µs", ha="center", fontsize=8.5, color=C["teal"])
    ax.text(9.4, 6.05, "同 Leaf 跨机:RDMA ~µs", ha="center", fontsize=8.5, color=C["teal"])
    ax.text(1.9, 3.35, "机内 8/4 卡:NVLink ~900GB/s / 纳秒级", ha="left", fontsize=8.5, color=C["green"])

    # 右下角:通信量-带宽映射规则
    ax.add_patch(Rectangle((0.5, 0.3), 11.0, 2.4, fc="#FBFBFB", ec=C["gray"], lw=1))
    ax.text(0.8, 2.35, "放置原则:通信越重,越要放近", fontsize=10.5, color=C["dark"], weight="bold")
    ax.text(0.8, 1.75, "• TP(张量并行,每层 all-reduce,通信最重)→ 塞进同一 NVLink 域(机内 8 卡)", fontsize=9.2, color=C["dark"])
    ax.text(0.8, 1.20, "• DP/PP(梯度同步 / 层间点对点,通信较轻)→ 同 Leaf 机架内,或走 rail-optimized 网络", fontsize=9.2, color=C["dark"])
    ax.text(0.8, 0.65, "• 最差:同一作业的 8 卡被撒到 8 台不同机架 → all-reduce 全走 Spine,通信成瓶颈,MFU 暴跌", fontsize=9.2, color=C["red"])

    fig.tight_layout()
    fig.savefig("sched_topology.png", bbox_inches="tight"); plt.close(fig)


# ================================================================ 3) 故障率 / MTBF / 弹性恢复
def fig_reliability():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # 左:整机 MTBF 与作业 MTBF 随规模衰减
    a1.set_title("规模越大,作业越频繁被单点故障打断", fontsize=12, weight="bold")
    N = np.array([8, 64, 256, 1024, 4096, 16384])
    # 假设单卡/单节点 MTBF ~ 5万小时(含 GPU/HBM/网卡/光模块/电源)。8卡节点。
    node_mtbf_h = 50000.0          # 单节点平均无故障时间(小时)
    gpus_per_node = 8
    n_nodes = N / gpus_per_node
    job_mtbf_h = node_mtbf_h / n_nodes           # 串联系统:任一节点挂 = 作业挂
    a1.plot(N, job_mtbf_h, "o-", color=C["red"], lw=2.2, ms=7, label="整作业 MTBF(小时)")
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xlabel("作业规模(GPU 数)"); a1.set_ylabel("平均无故障时间 MTBF(小时,对数)")
    for x, y in zip(N, job_mtbf_h):
        a1.annotate(f"{y:.0f}h" if y >= 1 else f"{y*60:.0f}min", (x, y),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=C["red"])
    a1.axhline(24, color=C["gray"], ls="--", lw=1)
    a1.text(9, 27, "1 天", fontsize=8.5, color=C["gray"])
    a1.grid(True, which="both", ls=":", alpha=0.5)
    a1.legend(fontsize=9, loc="upper right")
    a1.text(150, 9000, "串联可靠性:\n作业MTBF = 节点MTBF / 节点数\n(假设单节点MTBF≈5万小时)",
            fontsize=8.5, color=C["dark"], ha="left", va="top")

    # 右:一次故障的时间轴(检测→重调度→从 checkpoint 恢复)与 checkpoint 频率权衡
    a2.set_title("弹性容错:检测→恢复的时间账 & checkpoint 权衡", fontsize=12, weight="bold")
    a2.set_xlim(0, 10); a2.set_ylim(0, 10); a2.axis("off")
    # 时间轴
    a2.add_patch(FancyArrowPatch((0.5, 8.6), (9.7, 8.6), arrowstyle="-|>", mutation_scale=16, color=C["dark"], lw=1.6))
    segs = [(0.5, 2.2, "正常训练", C["green"]),
            (2.2, 3.2, "故障发生\n静默/崩溃", C["red"]),
            (3.2, 4.6, "检测/心跳超时", C["orange"]),
            (4.6, 6.2, "重调度\n换备机", C["purple"]),
            (6.2, 7.7, "载入 ckpt", C["blue"]),
            (7.7, 9.6, "追回丢失步", C["teal"])]
    for (x0, x1, t, c) in segs:
        a2.add_patch(Rectangle((x0, 7.6), x1 - x0, 0.9, fc=c, ec="white", lw=1.2))
        a2.text((x0 + x1) / 2, 8.05, t, ha="center", va="center", color="white", fontsize=8)
    a2.text(0.5, 6.9, "浪费时间 = 检测 + 重调度 + 恢复 + 回滚(自上次 ckpt 以来的进度)", fontsize=9, color=C["dark"])

    # checkpoint 频率权衡曲线
    a2.set_position(a2.get_position())
    ax3 = fig.add_axes([0.60, 0.13, 0.35, 0.42])
    T = np.linspace(5, 240, 200)  # checkpoint 间隔(分钟)
    ckpt_cost_per = 3.0           # 每次存 ckpt 耗时(分钟)
    fail_per_day = 4.0            # 每天故障次数(万卡级)
    # 每天总开销 = 存ckpt开销 + 期望回滚(平均回滚半个间隔)* 故障次数
    overhead = (1440.0 / T) * ckpt_cost_per + (T / 2.0) * fail_per_day
    ax3.plot(T, overhead, color=C["blue"], lw=2.2)
    t_opt = T[np.argmin(overhead)]
    ax3.axvline(t_opt, color=C["red"], ls="--", lw=1.4)
    ax3.annotate(f"最优≈{t_opt:.0f}min", (t_opt, overhead.min()),
                 textcoords="offset points", xytext=(8, 18), fontsize=8.5, color=C["red"])
    ax3.set_xlabel("checkpoint 间隔(分钟)", fontsize=8.5)
    ax3.set_ylabel("每天浪费(分钟)", fontsize=8.5)
    ax3.tick_params(labelsize=8)
    ax3.set_title("存太勤→存盘费时;存太疏→回滚多", fontsize=8.8)
    ax3.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig("sched_reliability.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_gang()
    fig_topology()
    fig_reliability()
    print("done: sched_gang.png / sched_topology.png / sched_reliability.png")
