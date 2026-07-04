"""生成 14_功耗散热与成本 讲义专属配图(前缀 tco_)。
运行:python _gen_tco.py  → 输出 3 张 PNG 到当前 figures 目录。
数据为公开量级(datasheet / 行业估算),仅示意本质,非精确报价。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p, os.path.getsize(p), "bytes")


# ============================================================
# 图1:各代 GPU 单卡功耗(TDP) + 能效(perf/W,相对值)
# ============================================================
def fig_power_efficiency():
    gens = ["V100\n(2017)", "A100\n(2020)", "H100\n(2022)",
            "H200\n(2023)", "B200\n(2024)", "GB200*\n(2024)"]
    tdp = [300, 400, 700, 700, 1000, 1200]           # 单 GPU 典型 TDP (W)
    # 低精度稠密算力(TFLOPS,近似:V100 fp16 125, A100 312, H100 990 bf16,
    # H200 同 H100, B200 ~2250 fp16 稠密, GB200 单 GPU ~2500)
    flops = [125, 312, 990, 990, 2250, 2500]
    perf_w = [f / t for f, t in zip(flops, tdp)]      # TFLOPS/W
    rel = [p / perf_w[0] for p in perf_w]             # 相对 V100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(gens))

    # 左:TDP 柱状 + 机柜密度含义
    colors = ["#8ecae6", "#8ecae6", "#219ebc", "#219ebc", "#fb8500", "#e63946"]
    bars = ax1.bar(x, tdp, color=colors, edgecolor="black", width=0.62)
    for b, v in zip(bars, tdp):
        ax1.text(b.get_x() + b.get_width()/2, v + 12, f"{v}W",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.axhline(700, ls="--", color="gray", lw=1)
    ax1.text(0.05, 720, "H100 风冷极限 ~700W", fontsize=9, color="gray")
    ax1.set_xticks(x); ax1.set_xticklabels(gens, fontsize=9.5)
    ax1.set_ylabel("单 GPU 功耗 TDP (瓦)", fontsize=11)
    ax1.set_title("① 单卡功耗一路飙升 → 撞上「功耗墙」", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1400)
    ax1.grid(axis="y", ls=":", alpha=0.5)
    ax1.annotate("", xy=(5, 1200), xytext=(0, 300),
                 arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.6, ls="--"))
    ax1.text(2.3, 1150, "7 年 ×4 功耗\n必须上液冷", fontsize=9.5,
             color="#e63946", ha="center")

    # 右:能效 perf/W(相对 V100)—— 能效改善更快
    bars2 = ax2.bar(x, rel, color="#2a9d8f", edgecolor="black", width=0.62)
    for b, v, pw in zip(bars2, rel, perf_w):
        ax2.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.1f}×\n({pw:.2f} T/W)",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(gens, fontsize=9.5)
    ax2.set_ylabel("能效 perf/W(相对 V100 = 1×)", fontsize=11)
    ax2.set_title("② 能效(每瓦算力)比功耗涨得更快\n→ 每 token 电费在降", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(rel)*1.25)
    ax2.grid(axis="y", ls=":", alpha=0.5)

    fig.suptitle("GPU 各代:功耗 vs 能效(公开量级,示意)", fontsize=13, fontweight="bold", y=1.02)
    save(fig, "tco_power_efficiency.png")


# ============================================================
# 图2:3 年 TCO 拆解(自建) + 自建 vs 云租盈亏平衡
# ============================================================
def fig_tco_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(wspace=0.45)

    # --- 左:一台 8×H100 服务器 3 年 TCO 饼图(量级示意,美元)---
    labels = ["硬件采购 CapEx\n(服务器+GPU)", "电费 OpEx\n(GPU+整机)",
              "散热/制冷\n(PUE 溢出)", "网络 InfiniBand",
              "数据中心\n(机位/机柜)", "运维人力"]
    # 8×H100 整机约 $250k;3 年电费(整机 ~10kW,$0.1/kWh):~$26k;
    # 冷却按 PUE1.3 额外 ~30% 电:~$8k;网络分摊 ~$20k;机位 ~$15k;运维 ~$12k
    vals = [250, 26, 8, 20, 15, 12]
    colors = ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#8d99ae"]
    total = sum(vals)
    wedges, _ = ax1.pie(vals, colors=colors, startangle=90, radius=1.1,
                        wedgeprops=dict(edgecolor="white", lw=1.5))
    ax1.legend(wedges, [f"{l}  ${v}k ({v/total*100:.0f}%)" for l, v in zip(labels, vals)],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8.5,
               frameon=False, ncol=2)
    ax1.set_title(f"① 8×H100 服务器 3 年 TCO ≈ ${total}k\n(硬件采购占大头,但电+冷是持续 OpEx)",
                  fontsize=11.5, fontweight="bold")

    # --- 右:自建 vs 云租 累计成本 盈亏平衡 ---
    months = np.arange(0, 37)
    capex = 250_000            # 自建一次性
    self_monthly = (26_000 + 8_000 + 20_000 + 15_000 + 12_000) / 36  # 月 OpEx
    self_cum = capex + self_monthly * months
    # 云租:8×H100 约 $2.5/GPU/h → 8*2.5*24*30 ≈ $43.2k/月(按需),预留折扣按 ~$1.8/h
    cloud_ondemand = 8 * 2.5 * 24 * 30 * months
    cloud_reserved = 8 * 1.8 * 24 * 30 * months

    ax2.plot(months, self_cum/1000, color="#e63946", lw=2.4, marker="", label="自建(CapEx+OpEx)")
    ax2.plot(months, cloud_ondemand/1000, color="#264653", lw=2.2, ls="--", label="云租·按需 $2.5/GPU·h")
    ax2.plot(months, cloud_reserved/1000, color="#2a9d8f", lw=2.2, ls="-.", label="云租·预留 $1.8/GPU·h")

    # 盈亏平衡点(自建 vs 按需)
    diff = self_cum - cloud_ondemand
    idx = np.argmin(np.abs(diff))
    ax2.axvline(months[idx], color="gray", ls=":", lw=1)
    ax2.scatter([months[idx]], [self_cum[idx]/1000], color="black", zorder=5)
    ax2.text(months[idx]+0.5, self_cum[idx]/1000-120,
             f"≈第 {months[idx]} 月\n跨过盈亏平衡\n(高利用率下自建更省)",
             fontsize=9, color="black")

    ax2.set_xlabel("月", fontsize=11)
    ax2.set_ylabel("累计成本(千美元)", fontsize=11)
    ax2.set_title("② 自建 vs 云租:累计成本与盈亏平衡\n(前提:利用率高、长期占用)",
                  fontsize=11.5, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(ls=":", alpha=0.5)
    ax2.set_xlim(0, 36)

    save(fig, "tco_breakdown.png")


# ============================================================
# 图3:散热方案对比(风冷/冷板/浸没)+ PUE + 机柜密度
# ============================================================
def fig_cooling_pue():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    methods = ["风冷\nAir", "冷板液冷\nD2C Cold-Plate", "单相浸没\nImmersion 1P", "两相浸没\nImmersion 2P"]
    rack_kw = [20, 80, 100, 150]       # 单机柜可支持功率密度(kW)
    pue = [1.5, 1.2, 1.08, 1.04]       # 典型 PUE
    xc = np.arange(len(methods))

    # 左:机柜功率密度(柱) + PUE(折线,右轴)
    colors = ["#8ecae6", "#219ebc", "#2a9d8f", "#0f766e"]
    bars = ax1.bar(xc, rack_kw, color=colors, edgecolor="black", width=0.6)
    for b, v in zip(bars, rack_kw):
        ax1.text(b.get_x()+b.get_width()/2, v+2, f"{v}kW",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("单机柜可支持功率密度 (kW)", fontsize=11)
    ax1.set_xticks(xc); ax1.set_xticklabels(methods, fontsize=9.5)
    ax1.set_ylim(0, 180)
    ax1.set_title("① 散热能力决定机柜密度\n(液冷才能塞下 B200 机柜)", fontsize=12, fontweight="bold")

    axr = ax1.twinx()
    axr.plot(xc, pue, color="#e63946", marker="o", lw=2.2, ms=8)
    for xx, p in zip(xc, pue):
        axr.text(xx, p+0.01, f"PUE {p}", color="#e63946", fontsize=9.5,
                 ha="center", fontweight="bold")
    axr.set_ylabel("PUE(越低越好)", color="#e63946", fontsize=11)
    axr.set_ylim(1.0, 1.7)
    axr.tick_params(axis="y", colors="#e63946")

    # 右:PUE 拆解——1 度 IT 电,配套要多花多少
    ax2.axis("off")
    ax2.set_title("② PUE = 数据中心总电 / IT 设备电", fontsize=12, fontweight="bold")

    def bar_block(y, it_frac, over_frac, label, pue_v, col):
        # IT 部分
        ax2.add_patch(Rectangle((0.05, y), it_frac*0.7, 0.09,
                                facecolor="#2a9d8f", edgecolor="black"))
        ax2.text(0.05+it_frac*0.7/2, y+0.045, "IT 算力\n1.0", ha="center",
                 va="center", fontsize=8.5, color="white", fontweight="bold")
        # 溢出(制冷+供配电损耗)
        ax2.add_patch(Rectangle((0.05+it_frac*0.7, y), over_frac*0.7, 0.09,
                                facecolor=col, edgecolor="black"))
        ax2.text(0.05+it_frac*0.7+over_frac*0.7/2, y+0.045,
                 f"制冷/损耗\n{over_frac:.2f}", ha="center", va="center",
                 fontsize=8.5, color="black")
        ax2.text(0.05, y+0.13, f"{label}(PUE={pue_v}):1 度算力电 → 总耗 {pue_v} 度",
                 fontsize=9.5, fontweight="bold")

    bar_block(0.80, 1.0, 0.5, "传统风冷机房", 1.5, "#f4a261")
    bar_block(0.52, 1.0, 0.2, "冷板液冷", 1.2, "#e9c46a")
    bar_block(0.24, 1.0, 0.08, "浸没液冷", 1.08, "#a8dadc")
    ax2.text(0.05, 0.10,
             "同样跑满 1MW IT 算力:\n"
             "PUE1.5 → 总电 1.5MW;PUE1.08 → 总电 1.08MW\n"
             "→ 电费直接省 ~28%,还能上更高机柜密度",
             fontsize=9.5, color="#264653",
             bbox=dict(boxstyle="round", fc="#f1faee", ec="#264653"))
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    save(fig, "tco_cooling_pue.png")


if __name__ == "__main__":
    fig_power_efficiency()
    fig_tco_breakdown()
    fig_cooling_pue()
    print("ALL DONE")
