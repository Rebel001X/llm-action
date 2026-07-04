# -*- coding: utf-8 -*-
"""
_gen_quant.py —— 生成《量化与低精度》讲义专属配图(真实 PNG)。
前缀 quant_,避免与既有图重名。运行:python _gen_quant.py
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
         purple="#8172B3", gray="#8C8C8C", teal="#64B5CD", yellow="#E6C200")


def box(ax, x, y, w, h, text, fc, fs=11, tc="white", ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                                 color=color, lw=lw, linestyle=ls))


# =============================================================== 1) 量化映射示意
def fig_mapping():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ---- 左:对称量化 symmetric (INT8, per-tensor)
    r = 6.0                      # 浮点绝对最大值 |x|max
    qmax = 127
    scale = r / qmax
    xf = np.linspace(-r, r, 400)
    q = np.clip(np.round(xf / scale), -127, 127)
    a1.plot(xf, q * scale, color=C["blue"], lw=2, label="反量化后 (阶梯)")
    a1.plot(xf, xf, color=C["gray"], ls="--", lw=1.2, label="理想 y=x")
    a1.axvline(0, color="#bbb", lw=0.8); a1.axhline(0, color="#bbb", lw=0.8)
    a1.axvline(r, color=C["red"], ls=":", lw=1.4)
    a1.axvline(-r, color=C["red"], ls=":", lw=1.4)
    a1.set_title("对称量化 Symmetric (INT8 权重典型)", fontsize=12.5, weight="bold")
    a1.set_xlabel("浮点值 x (fp16)"); a1.set_ylabel("反量化 dequant(Q(x))")
    a1.text(-r, -r + 0.4, f"零点 z=0\nscale s=|x|max/127\n={r:.0f}/127≈{scale:.3f}",
            fontsize=9.5, color=C["blue"], va="bottom",
            bbox=dict(fc="white", ec=C["blue"], alpha=0.9))
    a1.text(r, 5.6, "clip 到 ±|x|max", fontsize=9, color=C["red"], ha="right")
    a1.set_xlim(-7, 7); a1.set_ylim(-7, 7); a1.legend(loc="lower right", fontsize=9)
    a1.grid(alpha=0.25)

    # ---- 右:非对称量化 asymmetric (UINT8, 激活典型)
    xmin, xmax = -2.0, 8.0       # 激活常有偏(如 GELU/ReLU 后)
    qn, qp = 0, 255
    s2 = (xmax - xmin) / (qp - qn)
    z = round(qn - xmin / s2)
    xf2 = np.linspace(xmin, xmax, 400)
    q2 = np.clip(np.round(xf2 / s2 + z), qn, qp)
    a2.plot(xf2, (q2 - z) * s2, color=C["green"], lw=2, label="反量化后 (阶梯)")
    a2.plot(xf2, xf2, color=C["gray"], ls="--", lw=1.2, label="理想 y=x")
    a2.axvline(0, color="#bbb", lw=0.8); a2.axhline(0, color="#bbb", lw=0.8)
    a2.axvline(xmin, color=C["red"], ls=":", lw=1.4)
    a2.axvline(xmax, color=C["red"], ls=":", lw=1.4)
    a2.set_title("非对称量化 Asymmetric (UINT8 激活典型)", fontsize=12.5, weight="bold")
    a2.set_xlabel("浮点值 x (激活,分布有偏)"); a2.set_ylabel("反量化 dequant(Q(x))")
    a2.text(xmin + 0.2, 6.0, f"scale s=(max-min)/255\n=({xmax:.0f}-({xmin:.0f}))/255≈{s2:.3f}\n零点 z=round(-min/s)={z}",
            fontsize=9.5, color=C["green"], va="top",
            bbox=dict(fc="white", ec=C["green"], alpha=0.9))
    a2.set_xlim(xmin - 0.5, xmax + 0.5); a2.set_ylim(xmin - 0.5, xmax + 0.5)
    a2.legend(loc="lower right", fontsize=9); a2.grid(alpha=0.25)

    fig.suptitle("量化映射:把连续浮点区间 → 有限整数格,scale 定步长、zero-point 定偏移",
                 fontsize=13.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("quant_mapping.png", bbox_inches="tight"); plt.close(fig)


# =============================================================== 2) 精度-显存-速度权衡
def fig_tradeoff():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0))

    # ---- 左:7B 模型权重显存(GB) & 相对推理吞吐
    fmts = ["FP32", "FP16/BF16", "FP8", "INT8", "INT4"]
    bits = [32, 16, 8, 8, 4]
    mem = [7e9 * b / 8 / 1e9 for b in bits]        # 7B 参数 * bits/8 → GB
    x = np.arange(len(fmts))
    cols = [C["gray"], C["blue"], C["teal"], C["green"], C["orange"]]
    bars = a1.bar(x, mem, color=cols, width=0.62)
    for b, m in zip(bars, mem):
        a1.text(b.get_x() + b.get_width() / 2, m + 0.4, f"{m:.1f} GB",
                ha="center", fontsize=10, weight="bold")
    a1.set_xticks(x); a1.set_xticklabels(fmts, fontsize=10)
    a1.set_ylabel("7B 模型权重显存 (GB)")
    a1.set_title("位宽 → 显存:每砍一半位宽,显存/带宽减半", fontsize=12.5, weight="bold")
    a1.set_ylim(0, 32); a1.grid(axis="y", alpha=0.25)
    a1.text(0.5, 30, "24GB 卡红线", fontsize=9, color=C["red"])
    a1.axhline(24, color=C["red"], ls="--", lw=1.2)

    # ---- 右:位宽 vs 精度损失(困惑度上升,示意量级)
    bitw = np.array([16, 8, 6, 4, 3, 2])
    ppl_naive = np.array([0.0, 0.3, 1.2, 6.0, 20.0, 90.0])   # 朴素 RTN
    ppl_smart = np.array([0.0, 0.1, 0.4, 1.0, 3.5, 25.0])    # GPTQ/AWQ 等
    a2.plot(bitw, ppl_naive, "-o", color=C["red"], lw=2, label="朴素 RTN(逐值四舍五入)")
    a2.plot(bitw, ppl_smart, "-s", color=C["green"], lw=2, label="GPTQ / AWQ(误差感知)")
    a2.invert_xaxis()
    a2.set_xticks(bitw); a2.set_xticklabels([f"{b}bit" for b in bitw])
    a2.set_xlabel("权重位宽(越右越激进)")
    a2.set_ylabel("困惑度 PPL 相对上升(示意)")
    a2.set_title("位宽 → 精度:好算法把 4bit 的损失压到可用", fontsize=12.5, weight="bold")
    a2.axvspan(4.5, 3.5, color=C["yellow"], alpha=0.18)
    a2.text(4, 60, "4bit 甜点区\n显存/2 精度可控", fontsize=9.5, color="#7a6a00", ha="center")
    a2.legend(fontsize=9.5); a2.grid(alpha=0.25); a2.set_ylim(-3, 95)

    fig.suptitle("量化的三角:显存/带宽 ↓ · 吞吐 ↑ · 精度 ↓ —— 好方法把精度损失压到最小",
                 fontsize=13.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("quant_tradeoff.png", bbox_inches="tight"); plt.close(fig)


# =============================================================== 3) outlier + SmoothQuant 迁移
def fig_smooth():
    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])

    # ---- 左:激活 outlier 分布(少数通道幅值极大)
    a1 = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    nch = 48
    amp = np.abs(rng.normal(0, 1, nch))
    amp[7] = 22; amp[23] = 30; amp[38] = 18            # 少数 outlier 通道
    cols = [C["red"] if v > 10 else C["blue"] for v in amp]
    a1.bar(np.arange(nch), amp, color=cols, width=0.85)
    a1.set_title("激活的 outlier 问题:少数通道幅值×20~100", fontsize=12.5, weight="bold")
    a1.set_xlabel("激活通道 channel"); a1.set_ylabel("|激活| 幅值(示意)")
    a1.set_ylim(0, 36)
    a1.annotate("outlier 通道\n撑爆 per-tensor scale", xy=(23, 30), xytext=(33, 27),
                fontsize=9.5, color=C["red"], ha="center",
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    a1.axhline(3, color=C["gray"], ls="--", lw=1)
    a1.text(1, 4.2, "普通通道", fontsize=9, color=C["blue"])
    a1.grid(axis="y", alpha=0.2)

    # ---- 右:SmoothQuant 难度迁移示意
    a2 = fig.add_subplot(gs[0, 1]); a2.axis("off")
    a2.set_xlim(0, 10); a2.set_ylim(0, 10)
    a2.set_title("SmoothQuant:把激活的量化难度迁移给权重", fontsize=12.5, weight="bold")
    # before
    box(a2, 0.4, 7.4, 3.6, 1.6, "激活 X\n难量化(有 outlier)\n★★★★★", C["red"], 10)
    box(a2, 5.9, 7.4, 3.6, 1.6, "权重 W\n易量化(平滑)\n★☆☆☆☆", C["green"], 10)
    a2.text(5, 8.2, "×", fontsize=16, ha="center", color="#555")
    # transform
    box(a2, 2.6, 4.5, 4.8, 1.2, "按通道除/乘 缩放因子 s\nX'=X·diag(1/s),  W'=diag(s)·W", C["purple"], 10)
    arrow(a2, 2.2, 7.4, 3.2, 5.7, C["gray"])
    arrow(a2, 7.7, 7.4, 6.8, 5.7, C["gray"])
    # after
    box(a2, 0.4, 1.6, 3.6, 1.6, "激活 X'\n变好量化\n★★☆☆☆", C["green"], 10)
    box(a2, 5.9, 1.6, 3.6, 1.6, "权重 W'\n略变难(可控)\n★★☆☆☆", C["orange"], 10)
    a2.text(5, 2.4, "×", fontsize=16, ha="center", color="#555")
    arrow(a2, 3.4, 4.5, 2.2, 3.2, C["green"])
    arrow(a2, 6.6, 4.5, 7.7, 3.2, C["orange"])
    a2.text(5, 0.7, "乘积 X'·W' = X·W 不变(数学等价),但两边都好量化了",
            ha="center", fontsize=9.5, color="#333",
            bbox=dict(fc=C["yellow"], alpha=0.25, ec="none"))

    fig.tight_layout()
    fig.savefig("quant_smooth.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_mapping()
    fig_tradeoff()
    fig_smooth()
    print("done: quant_mapping.png / quant_tradeoff.png / quant_smooth.png")
