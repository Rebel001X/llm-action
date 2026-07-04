# -*- coding: utf-8 -*-
"""
_gen_sd.py —— 生成「存储与数据」讲义(06 篇)专属配图(真实 PNG)。
前缀 sd = Storage & Data,避免与仓库既有图重名。
运行:python _gen_sd.py  →  在本目录生成 sd_*.png
中文用 Microsoft YaHei。所有数字为主流硬件/大模型的量级估算。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C = dict(blue="#4C72B0", orange="#DD8452", green="#55A868", red="#C44E52",
         purple="#8172B3", gray="#8C8C8C", teal="#64B5CD", yellow="#E6C200")


def box(ax, x, y, w, h, text, fc, fs=11, tc="white", ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.2, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, linestyle=ls))


# ============================================================ 1) 数据流水线各级 + 喂不饱 GPU
def fig_pipeline():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 8.2),
                                 gridspec_kw={"height_ratios": [1.05, 1]})

    # ---- 上:流水线各级流程图 ----
    a1.set_xlim(0, 12); a1.set_ylim(0, 4.6); a1.axis("off")
    a1.set_title("训练数据流水线:从存储到 GPU 的六级流水(每级都可能成瓶颈)",
                 fontsize=13, weight="bold")
    stages = [
        ("① 读盘 / 拉取\nread / fetch", C["blue"],   "磁盘/网络 I/O", "S3/Lustre → 本地\n随机小文件 = 元数据风暴"),
        ("② 解码\ndecode",              C["teal"],   "CPU 单核",      "jpeg/音视频解码\n往往最重的一级"),
        ("③ 增强\naugment",             C["green"],  "CPU 多核",      "resize/裁剪/mask\n数据增强"),
        ("④ tokenize\n分词",            C["orange"], "CPU + GIL",     "BPE/SentencePiece\nPython 常卡 GIL"),
        ("⑤ 组 batch\ncollate",         C["purple"], "CPU + pin",     "pad/堆叠\npin_memory 锁页"),
        ("⑥ H2D 拷贝\nto GPU",          C["red"],    "PCIe",          "异步拷贝\n与计算重叠"),
    ]
    w = 1.72; gap = 0.22; x0 = 0.25
    for i, (name, col, who, note) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(a1, x, 2.35, w, 1.35, name, col, 9.5)
        a1.text(x + w / 2, 2.05, who, ha="center", fontsize=8.5, color="#222", weight="bold")
        a1.text(x + w / 2, 1.35, note, ha="center", fontsize=7.6, color="#555")
        if i < len(stages) - 1:
            arrow(a1, x + w, 3.02, x + w + gap, 3.02, "#444")
    box(a1, x0, 0.15, 11.5, 0.75,
        "瓶颈三连:磁盘/网络 I/O  +  CPU 解码算力  +  Python GIL —— 任一环慢,GPU 就空转饿肚子",
        C["gray"], 10)

    # ---- 下:GPU 利用率(喂不饱 vs 喂饱)----
    labels = ["朴素\n单进程 DataLoader", "多进程\nnum_workers", "多进程+预取\nprefetch/pin", "+ NVMe本地缓存\n+ WebDataset"]
    gpu_util = [32, 68, 88, 96]
    colors = [C["red"], C["orange"], C["green"], C["blue"]]
    bars = a2.bar(labels, gpu_util, color=colors, width=0.62)
    a2.axhline(95, color="#888", ls="--", lw=1)
    a2.text(3.35, 96.2, "目标:GPU 常忙 >95%", fontsize=8.5, color="#666", ha="right")
    a2.set_ylabel("GPU 计算利用率 (%)")
    a2.set_ylim(0, 105)
    a2.set_title("同一模型/同一批 GPU:数据流水线越强,GPU 越不挨饿(示意量级)",
                 fontsize=12, weight="bold")
    for b, v in zip(bars, gpu_util):
        a2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=10, weight="bold")
    a2.text(0.01, -0.22, "本质:GPU 算得快(µs 级),数据要跨 磁盘(ms)→CPU→PCIe 才到;不提前把下一 batch 备好,算力就浪费在等 I/O。",
            transform=a2.transAxes, fontsize=8.6, color="#444")
    fig.tight_layout()
    fig.savefig("sd_pipeline.png", bbox_inches="tight"); plt.close(fig)


# ============================================================ 2) checkpoint 分片 + 体积
def fig_checkpoint():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4),
                                 gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- 左:ZeRO-3 / FSDP 分片 → 分片 checkpoint ----
    a1.set_xlim(0, 10); a1.set_ylim(0, 10); a1.axis("off")
    a1.set_title("ZeRO-3 / FSDP:训练态分片 → checkpoint 按 rank 分片保存",
                 fontsize=12, weight="bold")
    a1.text(5, 9.5, "完整训练状态(每参数 16 B)= 权重2 + 梯度2 + fp32主副本4 + 动量m 4 + 方差v 4",
            ha="center", fontsize=8.6, color="#333")
    parts = [("权重 fp16\n2B", C["blue"]), ("梯度 fp16\n2B", C["teal"]),
             ("master fp32\n4B", C["orange"]), ("m 动量\n4B", C["green"]),
             ("v 方差\n4B", C["purple"])]
    ranks = 4
    colw = 8.6 / ranks
    for r in range(ranks):
        x = 0.7 + r * colw
        box(a1, x, 1.6, colw - 0.18, 6.9, "", "#EEF1F6", ec="#ccc", tc="#333")
        a1.text(x + (colw - 0.18) / 2, 8.15, f"GPU {r}\n(rank {r})", ha="center",
                fontsize=9, color="#123", weight="bold")
        ph = 5.9 / len(parts)
        for k, (t, col) in enumerate(parts):
            box(a1, x + 0.08, 1.75 + k * ph, colw - 0.34, ph - 0.1,
                t.split("\n")[0], col, 7.2)
        a1.text(x + (colw - 0.18) / 2, 1.2, f"分片 {r}\nshard", ha="center", fontsize=7.5, color=C["red"])
    a1.text(5, 0.45, "每卡只存自己那 1/N 分片 → N 张卡并行写 N 个文件,写盘带宽叠加;续训时按 rank 精确回填",
            ha="center", fontsize=8.4, color="#444")

    # ---- 右:不同规模模型 checkpoint 体积(堆叠)----
    models = ["7B", "13B", "70B", "175B", "405B"]
    P = np.array([7, 13, 70, 175, 405])          # 十亿参数
    w_fp16 = P * 2                                # GB 权重
    master = P * 4                                # fp32 主副本
    mom = P * 4                                   # m
    var = P * 4                                   # v
    x = np.arange(len(models))
    a2.bar(x, w_fp16, 0.6, label="权重 fp16 (2B)", color=C["blue"])
    a2.bar(x, master, 0.6, bottom=w_fp16, label="fp32 主副本 (4B)", color=C["orange"])
    a2.bar(x, mom, 0.6, bottom=w_fp16 + master, label="Adam m (4B)", color=C["green"])
    a2.bar(x, var, 0.6, bottom=w_fp16 + master + mom, label="Adam v (4B)", color=C["purple"])
    total = w_fp16 + master + mom + var
    for i, t in enumerate(total):
        a2.text(i, t + total.max() * 0.015, f"{t:.0f} GB", ha="center", fontsize=8.6, weight="bold")
    a2.set_xticks(x); a2.set_xticklabels(models)
    a2.set_ylabel("完整训练 checkpoint 体积 (GB)")
    a2.set_xlabel("模型规模(参数量)")
    a2.set_title("大模型 checkpoint 有多大(权重 + Adam 优化器状态,≈14 B/参数)",
                 fontsize=11, weight="bold")
    a2.legend(fontsize=8, loc="upper left")
    a2.set_ylim(0, total.max() * 1.18)
    a2.text(0.30, 0.60, "推理只存权重 ≈ 2B/参数;\n训练要连优化器一起存 ≈ 7×!\n70B 全量 ckpt ≈ 1 TB,\n405B ≈ 5.6 TB。",
            transform=a2.transAxes, fontsize=8.4, color="#444",
            bbox=dict(fc="#FFF7E6", ec="#E6C200", alpha=0.95))
    fig.tight_layout()
    fig.savefig("sd_checkpoint_shard.png", bbox_inches="tight"); plt.close(fig)


# ============================================================ 3) 存储栈:吞吐 vs 元数据
def fig_storage_stack():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---- 左:存储层级金字塔式带宽 ----
    tiers = ["本地内存/page cache\nDRAM", "本地 NVMe SSD\n单盘", "并行文件系统\nLustre / GPFS(集群聚合)",
             "对象存储\nS3 / OSS / GCS"]
    bw = [200000, 6000, 1000000, 100000]   # MB/s 量级(聚合)
    colors = [C["red"], C["orange"], C["green"], C["blue"]]
    y = np.arange(len(tiers))[::-1]
    a1.barh(y, bw, color=colors, height=0.62)
    a1.set_yticks(y); a1.set_yticklabels(tiers, fontsize=9)
    a1.set_xscale("log")
    a1.set_xlabel("聚合读带宽 MB/s(对数轴,集群量级)")
    a1.set_title("存储层级:带宽差几个数量级", fontsize=11.5, weight="bold")
    notes = ["~200 GB/s·纳秒级延迟", "~6 GB/s·微秒级·随机 IOPS 高",
             "数百节点聚合 >1 TB/s·大文件顺序读", "近乎无限容量·高延迟·元数据/QPS 受限"]
    for yi, v, n in zip(y, bw, notes):
        a1.text(v, yi, "  " + n, va="center", fontsize=7.8, color="#333")
    a1.set_xlim(1e3, 5e6)

    # ---- 右:吞吐 vs 元数据 —— 大文件 vs 海量小文件 ----
    a2.axis("off")
    a2.set_title("同样 100 GB 数据:文件粒度决定成败", fontsize=11.5, weight="bold")
    rows = [
        ["方案", "文件数", "元数据/open", "有效吞吐"],
        ["海量小文件\n(1 张图 = 1 文件)", "1 亿+", "每次都 stat/open\n→ 元数据风暴", "低(卡在 IOPS)"],
        ["打包 shard\n(WebDataset .tar)", "~1 万", "顺序读 tar\n几乎无 open", "高(打满带宽)"],
        ["列存 Parquet\n(结构化/文本)", "~千", "按列/行组读\n可谓词下推", "高 + 省 I/O"],
        ["mmap\n(单大文件内存映射)", "1", "按页惰性加载\n零拷贝", "极高(随机也快)"],
    ]
    tb = a2.table(cellText=rows, loc="center", cellLoc="center", bbox=[0, 0.05, 1, 0.92])
    tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 2.0)
    for j in range(4):
        tb[0, j].set_facecolor(C["blue"]); tb[0, j].set_text_props(color="white", weight="bold")
    tint = ["#FDECEA", "#E8F5E9", "#E8F5E9", "#E8F5E9"]
    for i in range(1, 5):
        for j in range(4):
            tb[i, j].set_facecolor(tint[i - 1])
    fig.tight_layout()
    fig.savefig("sd_storage_stack.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fns = [fig_pipeline, fig_checkpoint, fig_storage_stack]
    for f in fns:
        f(); print("  ok:", f.__name__)
    print("done, generated", len(fns), "figures")
