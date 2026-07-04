# -*- coding: utf-8 -*-
"""
_gen_opc.py —— 生成《算子与编译》讲义专属配图(真实 PNG)。
前缀 opc = OPerator & Compile。
运行:python _gen_opc.py  →  在本目录生成 opc_*.png
数字为量级示意(H100 级:HBM 带宽 ~3.35 TB/s,kernel launch ~5 µs)。
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
         dark="#33373D")


def box(ax, x, y, w, h, text, fc, fs=11, tc="white", ec="none", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=16, color=color, lw=lw, linestyle=ls))


# =============================================================== 图1:融合前后 HBM 往返对比
def fig_hbm_roundtrip():
    # 场景:对 4096x4096 的 fp16 张量做 4 个逐元素算子链  y = gelu(x*w + b)*scale
    S_MB = 4096 * 4096 * 2 / 1e6          # 单份张量字节(MB) ≈ 33.6 MB
    k = 4                                  # 链上算子个数
    BW = 3.35e6                            # HBM 带宽 MB/s = 3.35 TB/s
    launch_us = 5.0                        # 单次 kernel launch 开销

    # 未融合:每个算子各读一份写一份 -> 2*k 份;融合:全程只读1写1 -> 2 份
    traf_unfused = 2 * k * S_MB
    traf_fused = 2 * S_MB
    t_mem_unfused = traf_unfused / BW * 1e6   # µs
    t_mem_fused = traf_fused / BW * 1e6
    t_launch_unfused = k * launch_us
    t_launch_fused = 1 * launch_us

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle("逐元素算子链融合前后对比  y = gelu(x·w + b)·scale  (4096×4096 fp16,H100 量级)",
                 fontsize=14, weight="bold")

    # --- 面板 A:HBM 往返流量 ---
    ax = axes[0]
    bars = ax.bar(["未融合\n(4 个 kernel)", "融合\n(1 个 kernel)"],
                  [traf_unfused, traf_fused], color=[C["red"], C["green"]], width=0.55)
    ax.set_ylabel("HBM 读写流量 (MB)")
    ax.set_title("① HBM 往返流量  ↓4×", fontsize=12)
    for b, v in zip(bars, [traf_unfused, traf_fused]):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f} MB",
                ha="center", fontsize=11, weight="bold")
    ax.text(0, traf_unfused * 0.5, "每算子\n读1写1\n= 8 份", ha="center",
            va="center", color="white", fontsize=9)
    ax.text(1, traf_fused * 0.5, "只读1\n写1\n= 2 份", ha="center",
            va="center", color="white", fontsize=9)
    ax.set_ylim(0, traf_unfused * 1.18)

    # --- 面板 B:kernel launch 次数 ---
    ax = axes[1]
    bars = ax.bar(["未融合", "融合"], [k, 1], color=[C["red"], C["green"]], width=0.55)
    ax.set_ylabel("kernel launch 次数")
    ax.set_title("② kernel 启动次数  ↓4×", fontsize=12)
    for b, v in zip(bars, [k, 1]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v}",
                ha="center", fontsize=12, weight="bold")
    ax.set_ylim(0, k * 1.25)

    # --- 面板 C:端到端耗时(访存 + 启动 堆叠)---
    ax = axes[2]
    labels = ["未融合", "融合"]
    mem = [t_mem_unfused, t_mem_fused]
    lau = [t_launch_unfused, t_launch_fused]
    ax.bar(labels, mem, color=C["blue"], width=0.55, label="HBM 传输耗时")
    ax.bar(labels, lau, bottom=mem, color=C["orange"], width=0.55, label="launch 开销")
    tot_un = mem[0] + lau[0]
    tot_f = mem[1] + lau[1]
    ax.text(0, tot_un + 3, f"≈{tot_un:.0f} µs", ha="center", fontsize=11, weight="bold")
    ax.text(1, tot_f + 3, f"≈{tot_f:.0f} µs", ha="center", fontsize=11, weight="bold")
    ax.set_ylabel("耗时 (µs)")
    ax.set_title(f"③ 端到端耗时  {tot_un / tot_f:.1f}× 加速", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, tot_un * 1.25)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("opc_hbm_roundtrip.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================== 图2:torch.compile 编译栈
def fig_compile_stack():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("torch.compile 编译栈:从 Python 到 Triton/C++",
                 fontsize=15, weight="bold")

    # 主竖直流水线(左侧)
    box(ax, 0.4, 7.7, 4.6, 1.0, "① Python 模型 (eager)\n@torch.compile 包一层", C["gray"], 11)
    box(ax, 0.4, 6.2, 4.6, 1.1,
        "② TorchDynamo\nCPython 帧钩子·抓字节码\n→ FX Graph + Guards", C["blue"], 10.5)
    box(ax, 0.4, 4.7, 4.6, 1.1,
        "③ AOTAutograd\n生成前向+反向联合图\n算子下沉到 ATen/prims", C["purple"], 10.5)
    box(ax, 0.4, 3.2, 4.6, 1.1,
        "④ TorchInductor(后端)\n图优化:融合/布局/内存复用\n代码生成 codegen", C["teal"], 10.5)
    box(ax, 0.2, 1.35, 2.3, 1.1, "⑤a GPU 后端\n生成 Triton kernel\n+ CUDA Graph", C["green"], 10)
    box(ax, 2.9, 1.35, 2.3, 1.1, "⑤b CPU 后端\n生成 C++/OpenMP\n+ 向量化", C["orange"], 10)

    for (y1, y2) in [(7.7, 7.3), (6.2, 7.3 - 1.1 + 0.0)]:
        pass
    arrow(ax, 2.7, 7.7, 2.7, 7.32)
    arrow(ax, 2.7, 6.2, 2.7, 5.82)
    arrow(ax, 2.7, 4.7, 2.7, 4.32)
    arrow(ax, 2.5, 3.2, 1.6, 2.46)
    arrow(ax, 2.9, 3.2, 3.8, 2.46)

    # 右侧:guard 失败 / graph break 说明
    box(ax, 6.4, 6.2, 5.7, 1.1,
        "Guards(护栏)\n记录假设:shape/dtype/常量…\n运行时不满足 → 重编译 recompile", C["yellow"], 10, tc=C["dark"])
    arrow(ax, 5.0, 6.75, 6.4, 6.75, color=C["yellow"], lw=2)

    box(ax, 6.4, 4.5, 5.7, 1.3,
        "Graph Break(图断裂,危险!)\n遇到无法追踪的操作(print / .item() /\n数据依赖控制流 / 未支持算子)\n→ 图被切成多段,回落 eager 执行", C["red"], 10)
    arrow(ax, 5.0, 5.25, 6.4, 5.15, color=C["red"], lw=2, style="-|>")

    # 底部:优化收益条
    box(ax, 6.4, 2.6, 5.7, 1.0,
        "TorchInductor 关键优化\n• 逐元素/归约融合  • 布局选择\n• 常量折叠  • 内存复用/inplace  • 循环平铺", C["teal"], 9.5)
    arrow(ax, 4.9, 3.75, 6.4, 3.15, color=C["teal"], lw=1.8, ls="--")

    # 底部图例:mode
    ax.text(0.4, 0.55, "常用:torch.compile(model, mode=\"max-autotune\")   "
            "backend=\"inductor\"(默认) / \"cudagraphs\" / \"onnxrt\"",
            fontsize=10, color=C["dark"])
    ax.text(6.4, 1.7, "缩短:一次编译多次复用;\n动态 shape 用 dynamic=True 避免频繁重编译",
            fontsize=9.5, color=C["dark"])

    fig.savefig("opc_compile_stack.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================== 图3:Triton 平铺 + 自动调优
def fig_triton_autotune():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Triton:block 平铺编程 + 自动调优 autotune", fontsize=14, weight="bold")

    # --- 左:tiling 网格示意 ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("① 把大张量切成 program 各管一个 block(tile)", fontsize=12)
    # 画一个大矩阵被切成 tile
    n = 4
    x0, y0, W, H = 1.0, 1.2, 7.6, 7.0
    tw, th = W / n, H / n
    idx = 0
    for i in range(n):
        for j in range(n):
            fc = C["green"] if (i, j) == (1, 2) else "#DCE6F0"
            ax.add_patch(Rectangle((x0 + j * tw, y0 + (n - 1 - i) * th), tw * 0.94, th * 0.94,
                                   fc=fc, ec=C["blue"], lw=1.2))
            pid = i * n + j
            ax.text(x0 + j * tw + tw / 2, y0 + (n - 1 - i) * th + th / 2,
                    f"pid\n{pid}", ha="center", va="center", fontsize=8.5,
                    color=(C["dark"] if (i, j) != (1, 2) else "white"))
            idx += 1
    ax.text(x0 + W / 2, y0 + H + 0.5, "grid = (N/BLOCK,)  每个 program(pid)负责一个 tile",
            ha="center", fontsize=10, color=C["dark"])
    ax.text(x0 + W / 2, y0 - 0.55,
            "tl.load(x + offs, mask=offs<N) → 片上算 → tl.store 写回",
            ha="center", fontsize=9.5, color=C["blue"])

    # --- 右:autotune 不同 BLOCK 的实测延迟(量级示意) ---
    ax = axes[1]
    ax.set_title("② autotune 扫 BLOCK_SIZE 选最快配置", fontsize=12)
    blocks = [128, 256, 512, 1024, 2048]
    # 量级示意:太小 launch/occupancy 差、太大寄存器溢出;1024 附近最优
    lat = np.array([48, 33, 26, 23, 31], dtype=float)  # µs
    colors = [C["gray"]] * len(blocks)
    best = int(np.argmin(lat))
    colors[best] = C["green"]
    bars = ax.bar([str(b) for b in blocks], lat, color=colors, width=0.6)
    for b, v in zip(bars, lat):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}",
                ha="center", fontsize=10)
    ax.annotate("最优\n(过小: 占用率低/launch 多;\n过大: 寄存器溢出)",
                xy=(best, lat[best]), xytext=(best - 0.3, lat[best] + 14),
                fontsize=9, ha="center", color=C["green"],
                arrowprops=dict(arrowstyle="-|>", color=C["green"]))
    ax.set_xlabel("BLOCK_SIZE")
    ax.set_ylabel("kernel 延迟 (µs,量级示意)")
    ax.set_ylim(0, max(lat) * 1.35)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("opc_triton_autotune.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_hbm_roundtrip()
    fig_compile_stack()
    fig_triton_autotune()
    print("done: opc_hbm_roundtrip.png, opc_compile_stack.png, opc_triton_autotune.png")
