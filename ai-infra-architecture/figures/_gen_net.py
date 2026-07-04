# -*- coding: utf-8 -*-
"""
_gen_net.py —— 生成「网络与通信」讲义配图(真实 PNG)。
主题:RDMA/RoCE/IB · NCCL · 集合通信算法 · 集群拓扑。
运行:python _gen_net.py  →  在本目录生成 net_*.png
数字均为主流硬件量级(H100 集群 / 400G IB / NVLink4)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C = dict(blue="#4C72B0", orange="#DD8452", green="#55A868", red="#C44E52",
         purple="#8172B3", gray="#8C8C8C", teal="#64B5CD", yellow="#E6C200",
         dark="#333333", light="#EAEAEA")


def box(ax, x, y, w, h, text, fc, fs=11, tc="white", ec="none", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=15, color=color, lw=lw, linestyle=ls))


# ============================================================ 1) 集群拓扑
def fig_topology():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.set_title("集群拓扑:机内 NVLink 全互联(TP域) + 跨机 rail-optimized fat-tree(DP/PP)",
                 fontsize=14, weight="bold")

    # ---- Spine 层(fat-tree 上层)
    for i in range(4):
        box(ax, 1.6 + i * 2.5, 7.0, 1.6, 0.7, f"Spine {i}", C["purple"], 10)
    ax.text(0.1, 7.35, "Spine\n脊交换机", fontsize=10, color=C["purple"], weight="bold", va="center")

    # ---- Leaf / Rail 层
    rail_x = [1.6, 4.1, 6.6, 9.1]
    for i, x in enumerate(rail_x):
        box(ax, x, 5.4, 1.6, 0.7, f"Leaf/Rail {i}", C["blue"], 10)
    ax.text(0.1, 5.75, "Leaf\n(每条 rail\n一台)", fontsize=9.5, color=C["blue"], weight="bold", va="center")

    # spine<->leaf 全连接(fat-tree:任意 leaf 到任意 spine 都有路)
    for lx in rail_x:
        for i in range(4):
            arrow(ax, lx + 0.8, 6.1, 2.4 + i * 2.5, 7.0, color=C["light"], style="-", lw=1)

    # ---- 两个 GPU 节点(每节点 8 卡 + NVSwitch)
    def node(x0, name, col):
        box(ax, x0, 0.4, 4.4, 1.05, f"{name}:NVSwitch 全互联域(NVLink4 ~900 GB/s 双向)",
            C["green"], 10.5)
        for g in range(8):
            gx = x0 + 0.15 + g * 0.53
            box(ax, gx, 1.7, 0.45, 0.75, f"G{g}", col, 9, tc="white")
            # GPU -> NVSwitch
            arrow(ax, gx + 0.22, 1.7, gx + 0.22, 1.45, color=C["gray"], style="-", lw=1.3)
            # GPU g 的网卡 -> 对应 rail(rail 0..3 各接两卡示意)
            rail = rail_x[g % 4]
            arrow(ax, gx + 0.22, 2.45, rail + 0.8, 5.4, color=C["orange"], style="-", lw=0.9, ls="--")
        return

    node(0.6, "节点 A", C["teal"])
    node(6.4, "节点 B", C["red"])

    # 标注
    ax.text(6.0, 3.35, "跨机:每卡一张 400G IB/RoCE 网卡\n同号 GPU 走同一条 rail(rail-optimized)",
            fontsize=9.5, color=C["orange"], ha="center",
            bbox=dict(boxstyle="round", fc="#FFF4E8", ec=C["orange"]))
    ax.text(6.0, 0.05, "机内 = TP(通信最重,必须最高带宽)   |   机间 = DP / PP(通信相对少,可跨 rail)",
            fontsize=10, color=C["dark"], ha="center", weight="bold")

    plt.tight_layout()
    fig.savefig("net_topology.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================ 2) 三种 AllReduce 对比
def fig_allreduce():
    p = 64                      # GPU 数
    alpha = 2e-6                # 每步固定延迟 2 us(一跳)
    beta = 50e9                 # 有效带宽 50 GB/s(≈400 Gbps)
    n = np.logspace(3, 9, 200)  # 消息大小 1KB..1GB(字节)

    # 延迟模型(秒),换算成毫秒
    t_ring = (2 * (p - 1) * alpha + 2 * (p - 1) / p * n / beta) * 1e3
    t_recdbl = (np.log2(p) * alpha + np.log2(p) * n / beta) * 1e3     # 递归倍增/tree:步数少但搬 log2(p) 份
    t_dbt = (2 * np.log2(p) * alpha + 2 * n / beta) * 1e3            # double-binary-tree:低延迟+带宽近最优

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # -- 左:延迟 vs 消息大小(log-log)
    ax1.loglog(n / 1e6, t_ring, color=C["blue"], lw=2.6, label="Ring(带宽最优,步数 2(p-1))")
    ax1.loglog(n / 1e6, t_recdbl, color=C["red"], lw=2.6, label="Tree/递归倍增(延迟低,搬 log2(p) 份)")
    ax1.loglog(n / 1e6, t_dbt, color=C["green"], lw=2.6, ls="--",
               label="Double-Binary-Tree(NCCL:两者兼得)")
    # 注:log2 用普通字符,避免字体缺字
    ax1.set_xlabel("消息大小 / MB (log)")
    ax1.set_ylabel("AllReduce 耗时 / ms (log)")
    ax1.set_title(f"p={p} GPU:延迟 vs 消息大小(α=2µs/步,β=50GB/s)", fontsize=12, weight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=9.5, loc="upper left")
    ax1.axvspan(1e-3, 0.1, color=C["red"], alpha=0.06)
    ax1.axvspan(10, 1e3, color=C["blue"], alpha=0.06)
    ax1.text(4e-3, ax1.get_ylim()[1] * 0.25, "小消息\ntree/DBT 降延迟",
             fontsize=9, color=C["red"], ha="center")
    ax1.text(120, ax1.get_ylim()[1] * 0.25, "大消息\nring/DBT 提带宽",
             fontsize=9, color=C["blue"], ha="center")

    # -- 右:通信量(带宽项:总搬运数据 / 单份消息)与步数
    labels = ["Ring", "Tree\n(递归倍增)", "Double-\nBinary-Tree"]
    volume = [2 * (p - 1) / p, np.log2(p), 2.0]          # 每卡搬运的数据(以单份 n 为单位)
    steps = [2 * (p - 1), np.log2(p), 2 * np.log2(p)]    # 通信步数(∝延迟)
    x = np.arange(3)
    w = 0.38
    b1 = ax2.bar(x - w / 2, volume, w, color=C["orange"], label="带宽项:搬运数据量(×n)")
    ax2b = ax2.twinx()
    b2 = ax2b.bar(x + w / 2, steps, w, color=C["purple"], label="延迟项:通信步数")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("搬运数据量(以 n 为单位)", color=C["orange"])
    ax2b.set_ylabel("通信步数(∝ 固定延迟)", color=C["purple"])
    ax2.set_title(f"p={p}:通信量 vs 步数(越低越好)", fontsize=12, weight="bold")
    for xi, v in zip(x, volume):
        ax2.text(xi - w / 2, v + 0.15, f"{v:.1f}n", ha="center", fontsize=9, color=C["orange"])
    for xi, s in zip(x, steps):
        ax2b.text(xi + w / 2, s + 1.5, f"{int(s)}", ha="center", fontsize=9, color=C["purple"])
    lines = [b1, b2]
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper center")

    plt.tight_layout()
    fig.savefig("net_allreduce.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================ 3) RDMA vs TCP 数据路径
def fig_rdma():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("传统 TCP/IP(多次拷贝+内核) vs RDMA(内核旁路+零拷贝) vs GPUDirect RDMA",
                 fontsize=13.5, weight="bold")

    # ---- 左:传统 TCP 路径
    ax.text(2.0, 7.5, "① 传统 TCP/IP", fontsize=12, weight="bold", color=C["red"], ha="center")
    box(ax, 0.4, 6.2, 3.2, 0.7, "应用 App 缓冲区", C["gray"], 10)
    box(ax, 0.4, 4.9, 3.2, 0.7, "内核 Socket 缓冲区", C["red"], 10)
    box(ax, 0.4, 3.6, 3.2, 0.7, "TCP/IP 协议栈(CPU)", C["red"], 10)
    box(ax, 0.4, 2.3, 3.2, 0.7, "网卡 NIC", C["blue"], 10)
    for y0, y1 in [(6.2, 5.6), (4.9, 4.3), (3.6, 3.0)]:
        arrow(ax, 2.0, y0, 2.0, y1, color=C["red"], lw=2)
    ax.text(3.75, 4.9, "多次内存拷贝\n+ 中断 + 上下文切换\n→ 高 CPU、高延迟",
            fontsize=9, color=C["red"], va="center")

    # ---- 中:RDMA
    ax.text(6.0, 7.5, "② RDMA(IB/RoCE)", fontsize=12, weight="bold", color=C["green"], ha="center")
    box(ax, 4.6, 6.2, 3.0, 0.7, "应用 App 缓冲区(注册内存)", C["green"], 9.5)
    box(ax, 4.6, 2.3, 3.0, 0.7, "RNIC(RDMA 网卡)", C["blue"], 10)
    arrow(ax, 6.1, 6.2, 6.1, 3.0, color=C["green"], lw=2.6)
    ax.text(6.1, 4.6, "内核旁路\nkernel bypass\n零拷贝\nzero-copy", fontsize=9,
            color=C["green"], ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#EAF6EE", ec=C["green"]))
    box(ax, 4.6, 4.9, 3.0, 0.001, "", "none")  # 占位
    ax.text(6.0, 1.9, "CPU 不参与数据搬运", fontsize=9, color=C["dark"], ha="center")

    # ---- 右:GPUDirect RDMA
    ax.text(10.2, 7.5, "③ GPUDirect RDMA", fontsize=12, weight="bold", color=C["purple"], ha="center")
    box(ax, 8.7, 6.2, 3.0, 0.7, "GPU 显存 HBM", C["purple"], 10)
    box(ax, 8.7, 2.3, 3.0, 0.7, "RNIC(RDMA 网卡)", C["blue"], 10)
    arrow(ax, 10.2, 6.2, 10.2, 3.0, color=C["purple"], lw=2.8)
    ax.text(10.2, 4.6, "网卡直读/直写\nGPU 显存\n不经 CPU 内存\n中转", fontsize=9,
            color=C["purple"], ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#F0ECF7", ec=C["purple"]))
    ax.text(10.2, 1.9, "GPU 与网卡 最短路径", fontsize=9, color=C["dark"], ha="center")

    # ---- 底部对比条
    ax.text(6.0, 0.9, "延迟:TCP ~数十µs   =>   RDMA ~1-2µs   =>   GPUDirect 再省一次 PCIe 往返 / CPU 拷贝",
            fontsize=10.5, color=C["dark"], ha="center", weight="bold",
            bbox=dict(boxstyle="round", fc=C["light"], ec=C["gray"]))

    plt.tight_layout()
    fig.savefig("net_rdma.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_topology()
    fig_allreduce()
    fig_rdma()
    print("done: net_topology.png, net_allreduce.png, net_rdma.png")
