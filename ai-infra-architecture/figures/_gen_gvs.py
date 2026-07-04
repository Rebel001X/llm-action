# -*- coding: utf-8 -*-
"""生成 11_GPU虚拟化与共享 篇专属配图(前缀 gvs = GPU Virtualization & Sharing)。
只产出 gvs_*.png,不碰其它人的图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else "."

C_GPU = "#e8f0fe"
C_A = "#ffd8b1"   # 任务A
C_B = "#bce6c0"   # 任务B
C_C = "#c9c0ff"   # 任务C
C_HW = "#f6d0d0"
C_DRV = "#fde3a7"
C_RT = "#cfe8ff"
C_CT = "#d5f5d5"


def box(ax, x, y, w, h, text, fc, ec="#333", fs=10, lw=1.4, bold=False, radius=0.02):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.006,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#111")


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=lw, color=color))


# ============================================================
# 图1:MIG vs MPS vs time-slicing —— 三种共享方式的"空间/时间"本质对比
# ============================================================
def fig_schemes():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle("一块 GPU 三种共享方式:时间片轮转 / MPS 空间并发 / MIG 硬件分区",
                 fontsize=15, fontweight="bold", y=0.99)

    # ---- 面板1:time-slicing(时间维度轮流) ----
    ax = axes[0]
    ax.set_title("① time-slicing 时间片轮转\n(整块 GPU 轮流给不同上下文)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.5, 8.4, 9, 1.0, "整块 GPU(全部 SM+全部显存)", C_GPU, fs=10, bold=True)
    # 时间轴上一段段轮流
    ax.annotate("", xy=(9.4, 6.6), xytext=(0.5, 6.6),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.8))
    ax.text(9.5, 6.2, "时间 t", fontsize=10, color="#555")
    seg = [("A", C_A), ("B", C_B), ("C", C_C), ("A", C_A), ("B", C_B), ("C", C_C)]
    w = 9.0 / len(seg)
    for i, (lab, c) in enumerate(seg):
        box(ax, 0.5 + i * w, 4.3, w - 0.05, 2.0, lab, c, fs=12, bold=True)
    ax.text(5, 3.4, "任一时刻只有 1 个上下文在跑,靠调度器快速切换", ha="center",
            fontsize=9.5, color="#444")
    ax.text(5, 1.8, "隔离:【无】内存/故障隔离\n利用率:中(切换有开销,空泡多)\n无需硬件支持,最简单",
            ha="center", fontsize=10, color="#b22", va="center",
            bbox=dict(boxstyle="round", fc="#fff5f5", ec="#e0a0a0"))

    # ---- 面板2:MPS(空间并发,共享同一上下文) ----
    ax = axes[1]
    ax.set_title("② MPS 多进程服务\n(多进程同时占用不同 SM,单上下文)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.5, 8.4, 9, 1.0, "整块 GPU · 共享显存池(无硬边界)", C_GPU, fs=10, bold=True)
    # 同一时刻 SM 被三色瓜分,但边界虚线(软划分)
    xs = 0.5
    parts = [("A", C_A, 3.6), ("B", C_B, 2.9), ("C", C_C, 2.0)]
    for lab, c, ww in parts:
        p = Rectangle((xs, 4.0), ww, 3.4, fc=c, ec="#666", lw=1.2, ls="--")
        ax.add_patch(p)
        ax.text(xs + ww / 2, 5.7, lab, ha="center", va="center", fontsize=13, fontweight="bold")
        xs += ww
    ax.text(5, 3.3, "同一时刻三进程并发跑在不同 SM 上(填满空泡)", ha="center",
            fontsize=9.5, color="#444")
    ax.text(5, 1.8, "隔离:【弱】共享显存,一进程越界/崩溃殃及全部\n利用率:高(并发填满 SM)\n可选 SM/显存百分比软限额",
            ha="center", fontsize=10, color="#a60", va="center",
            bbox=dict(boxstyle="round", fc="#fffbf0", ec="#e0c080"))

    # ---- 面板3:MIG(硬件分区,各自独立) ----
    ax = axes[2]
    ax.set_title("③ MIG 多实例 GPU\n(硬件把 GPU 切成隔离实例)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.5, 8.4, 9, 1.0, "物理 GPU(A100/H100)", C_GPU, fs=10, bold=True)
    # 三块实心硬边界 + 各自带 SM+显存+L2
    labels = [("GI 1\nSM+显存+L2\n独立", C_A, 3.05),
              ("GI 2\n独立", C_B, 3.05),
              ("GI 3\n独立", C_C, 2.9)]
    xs = 0.5
    for lab, c, ww in labels:
        box(ax, xs, 4.0, ww - 0.1, 3.4, lab, c, fs=9.5, bold=True, lw=2.4)
        xs += ww
    ax.text(5, 3.3, "每个实例有专属 SM 切片 + 专属显存通路 + 专属 L2", ha="center",
            fontsize=9.2, color="#444")
    ax.text(5, 1.8, "隔离:【强】内存/带宽/故障硬隔离,QoS 有保证\n利用率:中高(粒度固定,可能有碎片)\n仅 A100/H100/B200 等支持",
            ha="center", fontsize=10, color="#161", va="center",
            bbox=dict(boxstyle="round", fc="#f2fff2", ec="#90c090"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE + "\\gvs_schemes.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ============================================================
# 图2:容器 GPU 软件栈(从容器里的 CUDA app 到 GPU 硬件)+ k8s device plugin
# ============================================================
def fig_container_stack():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis("off")
    ax.set_title("容器里怎么用上 GPU:nvidia-container-runtime 注入 + k8s device plugin 分配",
                 fontsize=14.5, fontweight="bold", y=1.0)

    # 左列:单机容器栈(自底向上)
    ax.text(3.2, 11.3, "单机:一个容器如何拿到 GPU", ha="center", fontsize=12, fontweight="bold", color="#245")
    box(ax, 0.6, 0.5, 5.2, 1.1, "GPU 硬件(SM / HBM / MIG 实例)", C_HW, bold=True, fs=11)
    box(ax, 0.6, 2.0, 5.2, 1.1, "宿主机 NVIDIA 内核驱动 + nvidia-uvm\n(容器不打包驱动,复用宿主)", C_DRV, fs=10)
    box(ax, 0.6, 3.5, 5.2, 1.1, "runc(OCI 运行时,起容器进程)", "#e7e7e7", fs=10.5)
    box(ax, 0.6, 5.0, 5.2, 1.3,
        "nvidia-container-runtime / -toolkit\nhook: 把 /dev/nvidiaX+用户态库+ldconfig\n注入进容器 rootfs", C_RT, fs=9.6, bold=True)
    box(ax, 0.6, 6.8, 5.2, 1.1, "containerd / dockerd", "#e7e7e7", fs=10.5)
    box(ax, 0.6, 8.3, 5.2, 1.5,
        "容器:你的 CUDA 应用\n(vLLM / PyTorch,自带 CUDA runtime\n但 NOT 驱动)", C_CT, fs=10, bold=True)
    # 上行箭头
    for y in [1.6, 3.1, 4.6, 6.3, 7.9]:
        arrow(ax, 3.2, y, 3.2, y + 0.4, color="#555", lw=1.8)
    ax.text(6.9, 7.2, "关键:driver 在宿主,\nlib 由 runtime 注入", fontsize=9.0,
            color="#a60", va="center", ha="center")

    # 右列:k8s 分配路径
    ax.text(10.4, 11.3, "集群:GPU 怎么被调度分配", ha="center", fontsize=12, fontweight="bold", color="#245")
    box(ax, 8.0, 9.4, 5.4, 1.1, "Pod 请求 resources:\nlimits: nvidia.com/gpu: 1", C_CT, fs=10, bold=True)
    box(ax, 8.0, 7.6, 5.4, 1.1, "kube-scheduler\n按可分配 GPU 数放置 Pod", "#e7e7e7", fs=10)
    box(ax, 8.0, 5.8, 5.4, 1.3,
        "kubelet + NVIDIA device plugin\n发现 GPU→上报 capacity→分配时\n把设备/环境变量传给容器", C_RT, fs=9.6, bold=True)
    box(ax, 8.0, 4.0, 5.4, 1.1, "NVIDIA_VISIBLE_DEVICES=<uuid>\n→ container-runtime 注入该卡", C_DRV, fs=9.6)
    box(ax, 8.0, 2.2, 5.4, 1.1, "容器只看得到被分到的 GPU\n(或 MIG 实例 nvidia.com/mig-1g.10gb)", C_HW, fs=9.4, bold=True)
    for y in [9.4, 7.6, 5.8, 4.0]:
        arrow(ax, 10.7, y, 10.7, y - 0.4, color="#555", lw=1.8)

    # 连接两列:device plugin 也调用同一 runtime
    arrow(ax, 8.0, 4.55, 5.8, 5.65, color="#c05", lw=1.8, style="-|>")
    ax.text(6.9, 5.2, "复用同一\ninjection 机制", fontsize=8.6, color="#c05", ha="center")

    out = HERE + "\\gvs_container_stack.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ============================================================
# 图3:拓扑/NUMA 亲和 —— 把容器绑到就近的 GPU/网卡/内存
# ============================================================
def fig_topology():
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("拓扑/NUMA 亲和:CPU-GPU-NIC-内存的远近决定带宽与延迟",
                 fontsize=14.5, fontweight="bold", y=1.0)

    # 两个 NUMA 节点(socket)
    ax.add_patch(Rectangle((0.4, 1.2), 6.2, 8.4, fc="#eef4ff", ec="#5577aa", lw=2))
    ax.add_patch(Rectangle((7.4, 1.2), 6.2, 8.4, fc="#fff0ee", ec="#aa6655", lw=2))
    ax.text(3.5, 9.2, "NUMA 节点 0(Socket 0)", ha="center", fontsize=12, fontweight="bold", color="#245")
    ax.text(10.5, 9.2, "NUMA 节点 1(Socket 1)", ha="center", fontsize=12, fontweight="bold", color="#524")

    # 节点0 组件
    box(ax, 1.0, 7.2, 2.2, 1.2, "CPU 0", "#dfe8ff", bold=True, fs=11)
    box(ax, 3.8, 7.2, 2.2, 1.2, "本地内存\nDRAM 0", "#d5f5d5", fs=10, bold=True)
    box(ax, 1.0, 5.0, 2.2, 1.1, "PCIe 交换", "#eee", fs=9.5)
    box(ax, 3.8, 5.0, 2.2, 1.1, "PCIe 交换", "#eee", fs=9.5)
    box(ax, 0.9, 2.6, 2.0, 1.6, "GPU 0", C_A, bold=True, fs=11)
    box(ax, 3.1, 2.6, 1.7, 1.6, "GPU 1", C_A, bold=True, fs=11)
    box(ax, 5.0, 2.6, 1.4, 1.6, "NIC 0\nRDMA", C_C, bold=True, fs=9.5)

    # 节点1 组件
    box(ax, 8.0, 7.2, 2.2, 1.2, "CPU 1", "#ffe1dc", bold=True, fs=11)
    box(ax, 10.8, 7.2, 2.2, 1.2, "本地内存\nDRAM 1", "#d5f5d5", fs=10, bold=True)
    box(ax, 8.0, 5.0, 2.2, 1.1, "PCIe 交换", "#eee", fs=9.5)
    box(ax, 10.8, 5.0, 2.2, 1.1, "PCIe 交换", "#eee", fs=9.5)
    box(ax, 7.9, 2.6, 2.0, 1.6, "GPU 2", C_B, bold=True, fs=11)
    box(ax, 10.1, 2.6, 1.7, 1.6, "GPU 3", C_B, bold=True, fs=11)
    box(ax, 12.0, 2.6, 1.4, 1.6, "NIC 1\nRDMA", C_C, bold=True, fs=9.5)

    # NVLink among GPUs in a node
    ax.plot([2.9, 3.1], [3.4, 3.4], color="#0a0", lw=3)
    ax.text(2.0, 4.35, "NVLink", fontsize=8.5, color="#0a0")
    ax.plot([9.9, 10.1], [3.4, 3.4], color="#0a0", lw=3)

    # 本地连线(近)
    for (x1, y1, x2, y2) in [(2.1, 7.2, 2.1, 6.1), (2.1, 5.0, 1.9, 4.2),
                              (4.9, 5.0, 5.7, 4.2), (2.1, 8.0, 3.8, 7.8)]:
        ax.plot([x1, x2], [y1, y2], color="#3a7", lw=2.2)

    # 跨 socket UPI(远)
    ax.annotate("", xy=(7.9, 6.6), xytext=(3.3, 6.6),
                arrowprops=dict(arrowstyle="<|-|>", color="#c33", lw=2.6, ls="--"))
    ax.text(5.6, 6.9, "UPI/QPI 跨 socket(远,窄且高延迟)", ha="center",
            fontsize=9.5, color="#c33", fontweight="bold")

    # 近 vs 远 一句话对比表
    ax.text(7.0, 0.55,
            "【就近·同NUMA】GPU0-DRAM0-NIC0 走本地 PCIe,带宽足、延迟低      "
            "【跨NUMA·远】GPU0 读 DRAM1 要过 UPI,带宽腰斩、延迟翻倍",
            ha="center", fontsize=10.2, color="#333",
            bbox=dict(boxstyle="round", fc="#fffef0", ec="#ccc"))

    out = HERE + "\\gvs_topology.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_schemes()
    fig_container_stack()
    fig_topology()
    print("ALL DONE")
