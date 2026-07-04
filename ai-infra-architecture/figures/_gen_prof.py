# -*- coding: utf-8 -*-
"""
_gen_prof.py —— 生成「可观测性与性能剖析」讲义配图(真实 PNG)。
运行:python _gen_prof.py  →  在本目录生成 prof_*.png
前缀 prof_ 专属本篇,数字为主流训练/推理场景的量级(H100 集群)。
中文用 Microsoft YaHei。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C = dict(blue="#4C72B0", orange="#DD8452", green="#55A868", red="#C44E52",
         purple="#8172B3", gray="#8C8C8C", teal="#64B5CD", yellow="#E6C200",
         idle="#D9D9D9")


def rbox(ax, x, y, w, h, text, fc, fs=10, tc="white", ec="none", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=1.4, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=15, color=color, lw=lw, linestyle=ls))


# ============================================================ 图1:训练一步时间线分解
def fig_timeline():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    fig.suptitle("训练一步(one step)的时间线分解:重叠好 vs 有瓶颈", fontsize=15, weight="bold")

    # 段:(起点, 长度, 颜色, 标签)
    # ---- 上:重叠良好(理想)总 ~90ms
    good = [
        (0,   30, C["blue"],   "Forward 前向"),
        (30,  52, C["green"],  "Backward 反向"),
        (82,   8, C["purple"], "Optimizer 优化器"),
    ]
    # 通信与 DataLoader 被隐藏(与计算重叠),画在下方细条示意
    good_hidden = [
        (32, 48, C["orange"], "AllReduce 与反向重叠(隐藏)"),
        (-6, 30, C["teal"],  "DataLoader 预取(隐藏)"),
    ]
    # ---- 下:有瓶颈 总 ~150ms
    bad = [
        (0,   16, C["idle"],  "GPU 空转\n(DataLoader 卡)"),
        (16,  32, C["blue"],  "Forward"),
        (48,  55, C["green"], "Backward"),
        (103, 30, C["orange"],"AllReduce\n(未重叠·串行等待)"),
        (133, 10, C["idle"],  "PP 气泡\nbubble"),
        (143,  8, C["purple"],"Optimizer"),
    ]

    def draw(ax, segs, y, title, total, mfu):
        ax.set_ylim(0, 3)
        for (x, w, c, lab) in segs:
            ax.broken_barh([(x, w)], (y, 0.9), facecolors=c, edgecolor="white", lw=1.2)
            if w >= 8:
                ax.text(x + w / 2, y + 0.45, lab, ha="center", va="center",
                        fontsize=8.5, color="white", weight="bold")
        ax.text(-14, y + 0.45, title, ha="right", va="center", fontsize=10, weight="bold")
        ax.annotate(f"一步 {total} ms · MFU≈{mfu}", (total, y + 1.05),
                    fontsize=9.5, color="#333", ha="right", weight="bold")

    # 上图
    a1.set_title("① 健康:通信/取数与计算重叠,GPU 无空转", fontsize=11, loc="left", color=C["green"])
    for (x, w, c, lab) in good_hidden:
        a1.broken_barh([(x, w)], (0.35, 0.5), facecolors=c, alpha=0.55, edgecolor="white")
        a1.text(x + w / 2, 0.6, lab, ha="center", va="center", fontsize=8, color="#333")
    draw(a1, good, 1.4, "GPU 计算流", 90, "48%")
    a1.set_yticks([]); a1.set_xlim(-30, 165)
    a1.axvline(90, ls="--", lw=1, color=C["green"])

    # 下图
    a2.set_title("② 有瓶颈:取数卡→前反向→通信串行等待→PP 气泡(GPU 反复空转)",
                 fontsize=11, loc="left", color=C["red"])
    draw(a2, bad, 1.4, "GPU 计算流", 151, "29%")
    a2.set_yticks([]); a2.set_xlim(-30, 165)
    a2.axvline(90, ls="--", lw=1, color=C["green"])
    a2.axvline(151, ls="--", lw=1, color=C["red"])
    a2.set_xlabel("时间(毫秒 ms) →", fontsize=10)
    a2.annotate("同样的计算,却因空转/串行多花 61ms(+68%)", (151, 0.2),
                fontsize=9, color=C["red"], ha="right")

    legend = [Patch(fc=C["blue"], label="Forward 前向"),
              Patch(fc=C["green"], label="Backward 反向"),
              Patch(fc=C["orange"], label="通信 AllReduce"),
              Patch(fc=C["purple"], label="Optimizer"),
              Patch(fc=C["teal"], label="DataLoader 取数"),
              Patch(fc=C["idle"], label="空转/气泡 idle")]
    a1.legend(handles=legend, loc="upper right", ncol=3, fontsize=8.5, framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("prof_step_timeline.png", bbox_inches="tight"); plt.close(fig)


# ============================================================ 图2:瓶颈定位决策树
def fig_tree():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("瓶颈定位决策树:一步慢,到底卡在哪一层?", fontsize=15, weight="bold")

    rbox(ax, 4.3, 8.9, 3.4, 0.9, "一步 step 变慢 / MFU 低\n先 torch.profiler + Nsight 看时间线", C["gray"], 10)

    # 第一分叉:GPU 是否忙?
    rbox(ax, 4.3, 7.4, 3.4, 0.85, "GPU 时间线上有大片空隙吗?\n(SM 占用长时间≈0)", "#5B5B5B", 10)
    arrow(ax, 6.0, 8.9, 6.0, 8.25)

    # 左:有空隙 → CPU/数据/通信
    rbox(ax, 1.1, 6.0, 3.0, 0.8, "有:GPU 在等谁?", C["orange"], 10, weight="bold")
    arrow(ax, 4.3, 7.75, 2.6, 6.8); ax.text(3.1, 7.15, "有空隙", fontsize=8.5, color=C["orange"])

    rbox(ax, 0.1, 4.3, 2.15, 1.15, "DataLoader 卡\n(数据-bound)\n征兆:step 头部\nGPU 空,CPU 满", C["teal"], 8.5)
    rbox(ax, 2.45, 4.3, 2.15, 1.15, "通信不重叠\n(通信-bound)\n征兆:NCCL kernel\n占大块·计算流停", C["red"], 8.5)
    arrow(ax, 1.9, 6.0, 1.2, 5.45); arrow(ax, 2.9, 6.0, 3.5, 5.45)

    # 右:GPU 满 → 看是算力还是访存
    rbox(ax, 7.9, 6.0, 3.0, 0.8, "满:GPU 一直在算", C["green"], 10, weight="bold")
    arrow(ax, 7.7, 7.75, 9.4, 6.8); ax.text(8.6, 7.15, "无空隙", fontsize=8.5, color=C["green"])

    rbox(ax, 7.9, 4.6, 3.0, 0.75, "Nsight Compute 看热点 kernel:\n算术强度 vs 脊点(roofline)", "#3F7A4F", 9)
    arrow(ax, 9.4, 6.0, 9.4, 5.35)

    rbox(ax, 6.7, 2.7, 2.15, 1.25, "访存-bound\nmemory-bound\n征兆:DRAM读带宽\n高·Tensor Core空\n(LN/逐元素/decode)", C["purple"], 8.5)
    rbox(ax, 9.05, 2.7, 2.15, 1.25, "算力-bound\ncompute-bound\n征兆:Tensor Core\n吃满·带宽没打满\n(大 GEMM/prefill)", C["blue"], 8.5)
    arrow(ax, 8.9, 4.6, 7.8, 3.95); arrow(ax, 9.9, 4.6, 10.1, 3.95)

    # 对策条
    fixes = [
        (0.1, "对策:num_workers↑·\npin_memory·预取·\n数据格式/存储提速"),
        (2.45, "对策:通信与反向\n重叠·融合梯度桶·\n换拓扑/减 TP 跨机"),
        (6.7, "对策:算子融合·\nFlashAttn·提高 batch·\n减少 HBM 往返"),
        (9.05, "对策:混合精度/fp8·\n提高 occupancy·\n喂满张量核·换更大 GEMM"),
    ]
    for (x, t) in fixes:
        rbox(ax, x, 0.9, 2.15, 1.4, t, "#444", 8.2)
    arrow(ax, 1.17, 4.3, 1.17, 2.3); arrow(ax, 3.52, 4.3, 3.52, 2.3)
    arrow(ax, 7.77, 2.7, 7.77, 2.3); arrow(ax, 10.12, 2.7, 10.12, 2.3)

    ax.text(6, 0.15, "逐层排除:先分「GPU 空转 or 满」,再在满里分「访存 or 算力」——四类瓶颈对应四类对策",
            ha="center", fontsize=9.5, color="#555")
    fig.tight_layout()
    fig.savefig("prof_bottleneck_tree.png", bbox_inches="tight"); plt.close(fig)


# ============================================================ 图3:MFU 损耗瀑布
def fig_mfu_waterfall():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    labels = ["硬件峰值\n(BF16 990T)", "访存-bound\nkernel", "通信\n未重叠", "DataLoader\n空转",
              "PP/DP\n气泡", "重计算\nrecompute", "实测 MFU"]
    # 从 100% 峰值出发,逐项扣减,最后落到实际 MFU
    starts = 100
    losses = [-14, -12, -8, -9, -7]      # 各项损耗(百分点)
    final = starts + sum(losses)         # ≈50%? 调:100-50=50
    vals = [starts] + losses + [None]

    cum = starts
    colors = [C["gray"], C["purple"], C["red"], C["teal"], C["orange"], C["yellow"], C["green"]]
    for i, v in enumerate(vals):
        if i == 0:
            ax.bar(i, starts, color=colors[i], edgecolor="white")
            ax.text(i, starts + 1.5, "100%", ha="center", fontsize=9, weight="bold")
        elif i == len(vals) - 1:
            ax.bar(i, cum, color=colors[i], edgecolor="white")
            ax.text(i, cum + 1.5, f"{cum:.0f}%", ha="center", fontsize=10, weight="bold", color=C["green"])
        else:
            ax.bar(i, -v, bottom=cum + v, color=colors[i], edgecolor="white", alpha=0.9)
            ax.text(i, cum + v / 2, f"{v}", ha="center", va="center", fontsize=9, color="white", weight="bold")
            ax.plot([i - 0.4, i + 0.4], [cum, cum], color="#888", lw=0.8, ls="--")
            cum += v

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("相对硬件峰值算力的利用率(%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title("MFU 损耗瀑布:峰值算力是怎样一层层「漏掉」的(典型大模型训练)", fontsize=13.5, weight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.text(3, 92, "MFU = 实测有效算力 / 硬件峰值算力\n业界大模型训练常见 35%~55%",
            fontsize=9.5, color="#444",
            bbox=dict(boxstyle="round", fc="#FFF6E6", ec=C["orange"]))
    fig.tight_layout()
    fig.savefig("prof_mfu_waterfall.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_timeline()
    fig_tree()
    fig_mfu_waterfall()
    print("done: prof_step_timeline.png / prof_bottleneck_tree.png / prof_mfu_waterfall.png")
