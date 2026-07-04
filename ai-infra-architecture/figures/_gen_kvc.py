# -*- coding: utf-8 -*-
"""
_gen_kvc.py —— 生成《KV Cache 深入》讲义配图(真实 PNG)。
运行:python _gen_kvc.py  →  在本目录生成 kvc_*.png
前缀 kvc = KV Cache。数字均按主流配置量级(70B-ish: L=80, d_head=128, fp16)。
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
         light="#EAEAF2")


def rbox(ax, x, y, w, h, text, fc, fs=10, tc="white", ec="none", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def cell(ax, x, y, w, h, text, fc, fs=9, tc="black", ec="white"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))


# ================================================================ 1) PagedAttention 块表 + 前缀共享
def fig_paged():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("PagedAttention:像操作系统分页一样管理 KV Cache(块表 + 前缀共享)",
                 fontsize=14, weight="bold")

    # 两个逻辑序列(共享前缀 "系统提示")
    ax.text(1.4, 6.15, "序列 A 的逻辑 KV 块", ha="center", fontsize=10, color=C["blue"], weight="bold")
    ax.text(1.4, 4.05, "序列 B 的逻辑 KV 块", ha="center", fontsize=10, color=C["green"], weight="bold")
    la = ["前缀0", "前缀1", "A的tok", "A的tok"]
    lb = ["前缀0", "前缀1", "B的tok"]
    for i, t in enumerate(la):
        fc = C["orange"] if i < 2 else C["blue"]
        cell(ax, 0.3 + i * 0.85, 5.4, 0.8, 0.55, t, fc, 8.5, "white")
    for i, t in enumerate(lb):
        fc = C["orange"] if i < 2 else C["green"]
        cell(ax, 0.3 + i * 0.85, 3.3, 0.8, 0.55, t, fc, 8.5, "white")

    # 块表(logical block -> physical block number)
    ax.text(5.0, 6.15, "块表 Block Table\n(逻辑块号 → 物理块号)", ha="center", fontsize=9.5,
            color="#333", weight="bold")
    bt_a = [("L0", "#7"), ("L1", "#3"), ("L2", "#5"), ("L3", "#1")]
    bt_b = [("L0", "#7"), ("L1", "#3"), ("L2", "#9")]
    for i, (l, p) in enumerate(bt_a):
        cell(ax, 4.1, 5.35 - i * 0.0, 0.6, 0.0, "", C["light"])  # noop for spacing
    y0 = 5.55
    ax.text(4.35, y0 + 0.25, "A", fontsize=9, color=C["blue"], weight="bold")
    for i, (l, p) in enumerate(bt_a):
        cell(ax, 4.2, y0 - i * 0.42, 0.55, 0.4, l, "white", 8, "black", ec=C["blue"])
        cell(ax, 4.75, y0 - i * 0.42, 0.55, 0.4, p, C["blue"], 8, "white", ec="white")
    y1 = 3.6
    ax.text(4.35, y1 + 0.25, "B", fontsize=9, color=C["green"], weight="bold")
    for i, (l, p) in enumerate(bt_b):
        cell(ax, 4.2, y1 - i * 0.42, 0.55, 0.4, l, "white", 8, "black", ec=C["green"])
        cell(ax, 4.75, y1 - i * 0.42, 0.55, 0.4, p, C["green"], 8, "white", ec="white")

    # 物理 KV 块池(HBM,非连续)
    ax.text(8.7, 6.15, "物理 KV 块池(HBM 显存,非连续)", ha="center", fontsize=9.5,
            color="#333", weight="bold")
    # 8 个物理块,#7/#3 为共享前缀,#5/#1 属 A,#9 属 B,其余空闲
    phys = {"#7": C["orange"], "#3": C["orange"], "#5": C["blue"], "#1": C["blue"],
            "#9": C["green"], "#2": C["gray"], "#4": C["gray"], "#8": C["gray"]}
    order = ["#1", "#2", "#3", "#4", "#5", "#7", "#8", "#9"]
    for i, k in enumerate(order):
        col = i % 4
        row = i // 4
        fc = phys[k]
        lab = k + ("\n共享前缀" if fc == C["orange"] else ("\n空闲" if fc == C["gray"] else ""))
        cell(ax, 6.7 + col * 1.15, 5.15 - row * 1.15, 1.05, 1.0, lab, fc, 8.5, "white")

    # 箭头:块表 -> 物理块(共享前缀汇聚)
    arrow(ax, 5.3, 5.35, 6.7, 4.65, C["orange"], lw=1.6)   # A.L0/L1 -> #7/#3
    arrow(ax, 5.3, 3.4, 6.7, 4.15, C["orange"], lw=1.6)    # B.L0/L1 -> #7/#3 (共享)
    arrow(ax, 5.3, 4.5, 6.7, 5.15, C["blue"], lw=1.4)      # A -> #5
    arrow(ax, 5.3, 3.0, 8.0, 4.0, C["green"], lw=1.4)      # B -> #9
    # 逻辑 -> 块表
    arrow(ax, 3.7, 5.5, 4.15, 5.5, "#666", lw=1.3)
    arrow(ax, 2.9, 3.55, 4.15, 3.7, "#666", lw=1.3)

    ax.text(5.75, 0.35,
            "关键点:① KV 按固定大小「块 block」(如 16 token/块)分配 → 碎片≈0;"
            "② 相同前缀(系统提示/few-shot)只存一份物理块,多序列共享(前缀缓存);"
            "③ 逻辑连续、物理离散,靠块表寻址 —— 正是虚拟内存分页的思想。",
            ha="center", fontsize=9, color="#444",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF6E6", ec=C["orange"]))
    fig.tight_layout()
    fig.savefig("kvc_paged_attention.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================ 2) 各注意力变体 KV 大小对比
def fig_variants():
    # 参考配置:L=80 层,d_head=128,fp16(2B)。每 token KV 字节 = 2 * L * n_kv * d_head * 2
    L, dh, dt = 80, 128, 2
    def kv_kib(n_kv):
        return 2 * L * n_kv * dh * dt / 1024.0
    names = ["MHA\n(n_kv=64)", "GQA-8\n(n_kv=8)", "MQA\n(n_kv=1)", "MLA\n(潜在压缩)"]
    # MHA/GQA/MQA 用公式;MLA 存压缩潜在向量 c_KV(dim 512)+ rope(64),单份 = L*(512+64)*2B
    vals = [kv_kib(64), kv_kib(8), kv_kib(1), L * (512 + 64) * 2 / 1024.0]
    colors = [C["red"], C["blue"], C["green"], C["purple"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    # 左:每 token KV 大小(log)
    bars = a1.bar(range(4), vals, color=colors, width=0.62)
    a1.set_yscale("log")
    a1.set_ylabel("每 token 的 KV 大小(KiB,log)", fontsize=11)
    a1.set_title("各注意力变体:单 token KV 占用\n(L=80, d_head=128, fp16)", fontsize=12, weight="bold")
    a1.set_xticks(range(4)); a1.set_xticklabels(names, fontsize=10)
    base = vals[0]
    for b, v in zip(bars, vals):
        red = base / v
        a1.text(b.get_x() + b.get_width() / 2, v * 1.08,
                f"{v:.0f} KiB\n↓{red:.0f}×", ha="center", va="bottom", fontsize=9.5, weight="bold")
    a1.set_ylim(20, vals[0] * 3)
    a1.grid(axis="y", ls="--", alpha=0.4)

    # 右:整批 KV 显存(GB) vs 序列长度(GQA-8),叠不同 batch,画 HBM 预算线
    seq = np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 1024
    per_tok_gb = kv_kib(8) / 1024.0 / 1024.0  # KiB -> GiB
    for bsz, col in [(1, C["teal"]), (16, C["orange"]), (64, C["red"])]:
        a2.plot(seq / 1024, per_tok_gb * seq * bsz, "-o", color=col, lw=2, ms=4,
                label=f"batch={bsz}")
    a2.axhline(60, color=C["gray"], ls="--", lw=2)
    a2.text(2, 63, "≈可用 HBM 预算 60 GB(80GB 卡去掉权重)", fontsize=9, color=C["gray"])
    a2.set_xlabel("序列长度(K tokens)", fontsize=11)
    a2.set_ylabel("整批 KV 显存(GB)", fontsize=11)
    a2.set_title("KV 显存随「序列长度 × batch」爆炸(GQA-8)\n→ 为何必须分页/量化/卸载",
                 fontsize=12, weight="bold")
    a2.set_xscale("log", base=2); a2.set_yscale("log")
    a2.set_xticks([1, 4, 16, 64, 128]); a2.set_xticklabels(["1", "4", "16", "64", "128"])
    a2.legend(fontsize=10); a2.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("kvc_attention_variants.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================ 3) KV 卸载分层 HBM->DRAM->SSD
def fig_tiering():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.8); ax.axis("off")
    ax.set_title("KV Cache 分层存储与卸载:HBM → DRAM → SSD(热的留显存,冷的往下沉)",
                 fontsize=13.5, weight="bold")

    tiers = [
        ("GPU HBM(显存)", "容量 ~80 GB · 带宽 ~3.35 TB/s · 延迟 ~0.1 µs",
         "存:活跃 batch 的 KV(热)", C["red"], 4.0, 6.4),
        ("CPU DRAM(主存)", "容量 ~1–2 TB · 带宽 ~200–400 GB/s · 延迟 ~0.1 µs+PCIe",
         "存:可复用前缀 KV / 换出的会话(温)", C["orange"], 2.55, 8.2),
        ("本地 NVMe SSD", "容量 ~10–100 TB · 带宽 ~5–14 GB/s · 延迟 ~100 µs",
         "存:海量历史会话前缀 KV(冷)", C["blue"], 1.1, 9.6),
    ]
    x0 = 0.7
    for name, spec, use, col, y, w in tiers:
        cx = 0.7 + (10 - w) / 2 * 0.0 + (10 - w) / 2
        rbox(ax, cx, y, w, 1.15, f"{name}\n{spec}", col, 10)
        ax.text(cx + w + 0.15, y + 0.57, use, ha="left", va="center", fontsize=9.5, color="#333")

    # 上下箭头:换出/预取
    arrow(ax, 2.6, 4.0, 2.6, 3.7, C["gray"], lw=2.2)
    arrow(ax, 1.9, 2.55, 1.9, 2.25, C["gray"], lw=2.2)
    ax.text(2.75, 3.72, "换出 offload(显存吃紧)", ha="left", fontsize=8.5, color=C["gray"])
    ax.text(2.05, 2.27, "换出", ha="left", fontsize=8.5, color=C["gray"])
    arrow(ax, 8.3, 3.7, 8.3, 4.0, C["green"], lw=2.2)
    arrow(ax, 9.6, 2.25, 9.6, 2.55, C["green"], lw=2.2)
    ax.text(8.45, 3.72, "命中即预取 prefetch", ha="left", fontsize=8.5, color=C["green"])

    ax.text(5.5, 0.45,
            "权衡:越往下容量越大、越便宜,但带宽↓延迟↑。命中远端前缀 KV 仍需搬回 HBM;"
            "只有「搬运时间 < 重新 prefill 时间」才划算 → 常配 int8/fp8 量化压小搬运量。",
            ha="center", fontsize=9.2, color="#444",
            bbox=dict(boxstyle="round,pad=0.5", fc="#EEF6EE", ec=C["green"]))
    fig.tight_layout()
    fig.savefig("kvc_kv_tiering.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_paged()
    fig_variants()
    fig_tiering()
    print("done: kvc_paged_attention.png, kvc_attention_variants.png, kvc_kv_tiering.png")
