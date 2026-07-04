# -*- coding: utf-8 -*-
"""
_gen_ie.py —— 生成《推理引擎架构:连续批处理/调度/投机解码/chunked prefill》讲义配图。
运行:python _gen_ie.py  →  在本目录生成 ie_*.png(前缀 ie = inference engine)。
中文用 Microsoft YaHei;数值为主流引擎/硬件的量级示意。
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
         light="#E8E8E8")


def box(ax, x, y, w, h, text, fc, fs=10, tc="white", ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=15, color=color, lw=lw, linestyle=ls))


# ================================================================ 图1:静态 vs 连续批处理时间线
def fig_batching():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2))
    reqs = ["R1", "R2", "R3", "R4"]
    ycol = [C["blue"], C["green"], C["orange"], C["purple"]]

    # ---- 静态批处理:同批一起进,等最慢的 R1 全部算完才收尾、才放下一批 ----
    a1.set_title("① 静态批处理 static batching:整批同进同出,被最长请求拖死",
                 fontsize=13, weight="bold")
    # 每个请求:prefill 1 格 + decode 若干格;R1 最长(生成 8),其余早早结束却不能退出
    gen = [8, 3, 4, 2]            # 各请求要生成的 token 数
    for i, r in enumerate(reqs):
        y = 3 - i
        # prefill 块
        a1.add_patch(Rectangle((0, y), 1, 0.7, fc=C["red"], ec="white"))
        a1.text(0.5, y + 0.35, "P", ha="center", va="center", color="white", fontsize=9)
        # decode 块(整批一起走,step 数 = max(gen) = 8)
        for t in range(8):
            if t < gen[i]:
                a1.add_patch(Rectangle((1 + t, y), 1, 0.7, fc=ycol[i], ec="white"))
                a1.text(1.5 + t, y + 0.35, "d", ha="center", va="center",
                        color="white", fontsize=8)
            else:
                # 已完成但被迫占坑(padding / 空转,浪费算力)
                a1.add_patch(Rectangle((1 + t, y), 1, 0.7, fc=C["light"], ec="white",
                                       hatch="//"))
        a1.text(-0.3, y + 0.35, r, ha="right", va="center", fontsize=10, weight="bold")
    a1.axvline(9, color="k", ls="--", lw=1.2)
    a1.text(9.1, 3.9, "整批收尾 → 才能放下一批", fontsize=9.5, color="k")
    a1.text(5, -0.7, "灰色斜纹 = 请求已结束却被迫占坑空转(padding);GPU 有效利用率低",
            ha="center", fontsize=9.5, color=C["red"])
    a1.set_xlim(-1.2, 12.5); a1.set_ylim(-1.2, 4.3); a1.axis("off")

    # ---- 连续批处理:请求随时进出,空位立刻被新请求填满 ----
    a2.set_title("② 连续批处理 continuous / in-flight batching:随时进出,空位即刻补新请求",
                 fontsize=13, weight="bold")
    # R1 长;R2/R3/R4 结束后,立即换入 R5/R6/R7 继续算,槽位不空转
    layout = {
        0: [("P", C["red"], "R1")] + [("d", C["blue"], "")]*10,      # R1 长
        1: [("P", C["red"], "R2")] + [("d", C["green"], "")]*3
            + [("P", C["red"], "R5")] + [("d", C["teal"], "")]*6,     # R2 完 → R5 进
        2: [("P", C["red"], "R3")] + [("d", C["orange"], "")]*4
            + [("P", C["red"], "R6")] + [("d", C["yellow"], "")]*5,   # R3 完 → R6 进
        3: [("P", C["red"], "R4")] + [("d", C["purple"], "")]*2
            + [("P", C["red"], "R7")] + [("d", C["gray"], "")]*7,     # R4 完 → R7 进
    }
    for i in range(4):
        y = 3 - i
        for t, (lab, col, name) in enumerate(layout[i]):
            a2.add_patch(Rectangle((t, y), 1, 0.7, fc=col, ec="white"))
            a2.text(t + 0.5, y + 0.35, lab, ha="center", va="center",
                    color="white", fontsize=8)
            if name:
                a2.text(t + 0.5, y + 0.95, name, ha="center", fontsize=8,
                        color="k", weight="bold")
    a2.text(5, -0.7, "无空转:某请求一结束,调度器立刻把新请求填进该槽 → GPU 利用率拉满",
            ha="center", fontsize=9.5, color=C["green"])
    a2.set_xlim(-1.2, 12.5); a2.set_ylim(-1.2, 4.6); a2.axis("off")

    # 图例
    for ax in (a1, a2):
        ax.add_patch(Rectangle((10.4, 3.6), 0.4, 0.35, fc=C["red"])); ax.text(10.9, 3.77, "P=prefill", fontsize=8, va="center")
        ax.add_patch(Rectangle((10.4, 3.05), 0.4, 0.35, fc=C["blue"])); ax.text(10.9, 3.22, "d=decode", fontsize=8, va="center")

    fig.tight_layout()
    fig.savefig("ie_batching_timeline.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图2:chunked prefill 交织
def fig_chunked_prefill():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4))

    # 上:不切块 —— 一个长 prefill 独占一个 iteration,阻塞正在 decode 的请求
    a1.set_title("① 不切块:长 prompt 的 prefill 独占一个迭代 → 正在 decode 的请求被卡(TPOT 抖动)",
                 fontsize=12, weight="bold")
    # 迭代格子:decode 稳定跑,某一步插入一个巨大的 prefill
    seq = [("d", 1), ("d", 1), ("PREFILL 2048 tok", 6), ("d", 1), ("d", 1), ("d", 1)]
    x = 0
    for lab, w in seq:
        col = C["red"] if "PREFILL" in lab else C["blue"]
        a1.add_patch(Rectangle((x, 0), w, 1, fc=col, ec="white"))
        a1.text(x + w / 2, 0.5, lab, ha="center", va="center", color="white",
                fontsize=9 if "PREFILL" in lab else 8)
        x += w
    a1.annotate("这一大步里,所有 decode 请求都得等 → 一次长长的卡顿",
                xy=(5, 1.05), xytext=(5, 1.8), ha="center", fontsize=9.5, color=C["red"],
                arrowprops=dict(arrowstyle="-|>", color=C["red"]))
    a1.set_xlim(-0.5, 12); a1.set_ylim(-0.5, 2.4); a1.axis("off")

    # 下:chunked prefill —— 把长 prefill 切成小块,和 decode 交织进同一批
    a2.set_title("② chunked prefill:长 prefill 切成小块,与 decode 交织同批 → TTFT/TPOT 都平滑",
                 fontsize=12, weight="bold")
    # 每个迭代都是 [几个 decode + 一小块 prefill] 拼成的混合批
    chunks = ["P1/4", "P2/4", "P3/4", "P4/4"]
    x = 0
    ci = 0
    for step in range(6):
        # 每步:先放 2 个 decode 小块,再放一个 prefill chunk(若还有)
        a2.add_patch(Rectangle((x, 0), 0.5, 1, fc=C["blue"], ec="white"))
        a2.text(x + 0.25, 0.5, "d", ha="center", va="center", color="white", fontsize=7)
        a2.add_patch(Rectangle((x + 0.5, 0), 0.5, 1, fc=C["green"], ec="white"))
        a2.text(x + 0.75, 0.5, "d", ha="center", va="center", color="white", fontsize=7)
        if ci < len(chunks):
            a2.add_patch(Rectangle((x + 1.0, 0), 1.0, 1, fc=C["orange"], ec="white"))
            a2.text(x + 1.5, 0.5, chunks[ci], ha="center", va="center", color="white", fontsize=8)
            ci += 1
            w = 2.0
        else:
            w = 1.0
        # 迭代分隔线
        a2.axvline(x, color="w", lw=0)
        a2.text(x + w / 2, -0.35, f"iter {step+1}", ha="center", fontsize=7.5, color="#555")
        x += w + 0.15
    a2.text(x + 0.3, 0.5, "…", fontsize=14)
    a2.annotate("每个迭代都混着 decode(蓝/绿)+ 一小块 prefill(橙)\n→ decode 不再被长时间独占",
                xy=(3, 1.05), xytext=(3, 1.9), ha="center", fontsize=9.5, color=C["green"],
                arrowprops=dict(arrowstyle="-|>", color=C["green"]))
    a2.set_xlim(-0.5, 12); a2.set_ylim(-0.9, 2.6); a2.axis("off")

    fig.tight_layout()
    fig.savefig("ie_chunked_prefill.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图3:投机解码流程 + 期望加速
def fig_speculative():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # 左:流程框图
    a1.set_title("投机解码 speculative decoding:草稿 + 并行验证", fontsize=13, weight="bold")
    a1.set_xlim(0, 10); a1.set_ylim(0, 8); a1.axis("off")
    box(a1, 0.4, 6.4, 3.2, 1.1, "① 小草稿模型 draft\n自回归猜 K 个 token\n(便宜·快)", C["teal"], 9.5)
    box(a1, 0.4, 4.3, 3.2, 1.1, "候选序列\nx1' x2' … xK'(K 个草稿)", C["orange"], 9.5)
    box(a1, 5.0, 5.3, 4.4, 1.2, "② 大模型 target\n一次前向并行验证 K+1 个位置\n(一次 forward 拿到 K+1 组分布)", C["blue"], 9.5)
    box(a1, 5.0, 2.7, 4.4, 1.5, "③ 逐位接受/拒绝\n接受最长匹配前缀 → n 个\n拒绝处从大模型分布重采 1 个\n(保证输出分布与大模型一致)", C["green"], 9)
    box(a1, 1.6, 0.6, 6.4, 1.1, "一轮拿到 n+1 个 token(1 ≤ n+1 ≤ K+1)\n验证是 1 次大模型前向 → 摊薄成本", C["purple"], 9.5)
    arrow(a1, 2.0, 6.4, 2.0, 5.4, C["teal"])
    arrow(a1, 3.6, 4.85, 5.0, 5.6, C["orange"])
    arrow(a1, 7.2, 5.3, 7.2, 4.25, C["blue"])
    arrow(a1, 5.0, 3.0, 3.6, 1.7, C["green"])
    arrow(a1, 2.0, 4.3, 2.0, 1.7, C["gray"], ls="--", lw=1.5)
    a1.text(2.15, 3.0, "重采点", fontsize=8, color=C["gray"], rotation=90, va="center")

    # 右:期望加速随"接受率 α"变化(简化模型)
    a2.set_title("期望加速 vs 接受率 α(草稿长度 K)", fontsize=13, weight="bold")
    alpha = np.linspace(0.1, 0.95, 100)
    # 每轮期望被接受 token 数(几何):E = (1-α^(K+1))/(1-α)
    # 加速比 ≈ E / (1 + K*c),c=草稿相对大模型的成本比(取 0.15)
    c = 0.15
    for K, col in [(3, C["blue"]), (5, C["green"]), (7, C["orange"])]:
        E = (1 - alpha ** (K + 1)) / (1 - alpha)
        speed = E / (1 + K * c)
        a2.plot(alpha, speed, color=col, lw=2.2, label=f"K={K}")
    a2.axhline(1.0, color=C["red"], ls="--", lw=1.2)
    a2.text(0.12, 1.05, "=1(无收益线)", color=C["red"], fontsize=8.5)
    a2.set_xlabel("接受率 α(草稿 token 被大模型接受的概率)")
    a2.set_ylabel("期望加速比 (×)")
    a2.set_xlim(0.1, 0.95); a2.set_ylim(0, 4.2)
    a2.grid(alpha=0.3); a2.legend(title="草稿长度", fontsize=9)
    a2.text(0.5, 3.7, "α 高才划算;α 低时草稿成本反成负担", fontsize=9, color="#444", ha="center")

    fig.tight_layout()
    fig.savefig("ie_speculative_decoding.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_batching()
    fig_chunked_prefill()
    fig_speculative()
    print("done: ie_batching_timeline.png, ie_chunked_prefill.png, ie_speculative_decoding.png")
