# -*- coding: utf-8 -*-
"""
make_figures.py —— 生成 AI-Infra 底层架构讲义配图(真实 PNG)。
运行:python make_figures.py  →  在本目录生成一批 .png
中文用 Microsoft YaHei。所有数字标注均为主流硬件的量级(H100 / 典型服务器 CPU)。
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


# ---------------------------------------------------------------- 1) PD 分离架构
def fig_pd_arch():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("PD 分离架构(Prefill / Decode Disaggregation)", fontsize=15, weight="bold")
    box(ax, 0.2, 2.6, 1.4, 0.9, "用户请求\nrequests", C["gray"], 10)
    box(ax, 2.0, 4.2, 3.0, 1.3, "Prefill 集群\n(算力受限 compute-bound)\n大 batch·高算力·TP", C["blue"], 10)
    box(ax, 2.0, 0.5, 3.0, 1.3, "Decode 集群\n(访存受限 memory-bound)\n多请求·高显存带宽·大KV", C["green"], 10)
    box(ax, 5.8, 2.55, 2.2, 1.0, "KV Cache\n传输 / 池化\n(NVLink/RDMA)", C["orange"], 10)
    box(ax, 8.3, 2.6, 1.5, 0.9, "流式输出\ntokens", C["gray"], 10)
    arrow(ax, 1.6, 3.05, 2.0, 4.6, C["blue"])
    arrow(ax, 5.0, 4.6, 6.9, 3.55, C["orange"])
    arrow(ax, 6.9, 2.55, 5.0, 1.4, C["orange"])
    arrow(ax, 5.0, 1.0, 8.3, 2.9, C["green"])
    ax.text(5, 5.75, "把一次推理的两个阶段拆到不同 GPU 池:各自用最优并行/批策略,互不干扰",
            ha="center", fontsize=10, color="#444")
    ax.text(3.5, 4.05, "TTFT 敏感·计算密集", ha="center", fontsize=8.5, color=C["blue"])
    ax.text(3.5, 0.35, "TPOT 敏感·带宽密集", ha="center", fontsize=8.5, color=C["green"])
    fig.tight_layout(); fig.savefig("pd_disaggregation.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 2) PD 前后对比
def fig_pd_compare():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # 左:prefill vs decode 特性
    metrics = ["算术强度", "batch 大小", "对延迟指标", "瓶颈"]
    a1.axis("off"); a1.set_title("Prefill vs Decode 特性对比", fontsize=13, weight="bold")
    rows = [["阶段", "Prefill(预填充)", "Decode(解码)"],
            ["处理", "整段 prompt 并行", "逐 token 自回归"],
            ["瓶颈", "算力 compute-bound", "访存 memory-bound"],
            ["关键指标", "TTFT 首 token 延迟", "TPOT 每 token 延迟"],
            ["batch", "小(单请求即饱和算力)", "大(需大 batch 提带宽利用)"],
            ["KV cache", "生成", "读取+追加"]]
    tb = a1.table(cellText=rows, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.7)
    for j in range(3):
        tb[0, j].set_facecolor(C["blue"]); tb[0, j].set_text_props(color="white", weight="bold")
    # 右:利用率对比
    labels = ["合置(colocated)\nprefill+decode 抢卡", "PD 分离\n各自优化"]
    gpu_util = [58, 88]; goodput = [1.0, 1.7]
    x = np.arange(2)
    a2.bar(x - 0.18, gpu_util, 0.36, label="GPU 利用率(%)", color=C["blue"])
    a2b = a2.twinx()
    a2b.bar(x + 0.18, goodput, 0.36, label="有效吞吐(相对)", color=C["green"])
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylabel("GPU 利用率 (%)"); a2b.set_ylabel("有效吞吐 goodput(相对)")
    a2.set_title("分离前后(示意)", fontsize=13, weight="bold")
    a2.legend(loc="upper left", fontsize=8); a2b.legend(loc="upper right", fontsize=8)
    a2.set_ylim(0, 100); a2b.set_ylim(0, 2)
    fig.tight_layout(); fig.savefig("pd_compare.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 3) GPU 显存层级
def fig_gpu_mem():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    levels = ["寄存器\nRegister", "共享内存/L1\nShared/L1", "L2 Cache", "HBM3\n显存"]
    bw = [100000, 33000, 12000, 3350]     # GB/s(量级,H100)
    cap = [0.000256, 0.228, 50, 80000]    # MB
    lat = [1, 25, 200, 500]               # cycles(约)
    colors = [C["red"], C["orange"], C["green"], C["blue"]]
    a1.barh(levels, bw, color=colors)
    a1.set_xscale("log"); a1.set_xlabel("带宽 GB/s(对数轴)")
    a1.set_title("GPU 显存层级:带宽(H100 量级)", fontsize=12, weight="bold")
    for i, v in enumerate(bw):
        a1.text(v, i, f" {v:,}", va="center", fontsize=9)
    a2.barh(levels, lat, color=colors)
    a2.set_xscale("log"); a2.set_xlabel("延迟(周期 cycles,约)")
    a2.set_title("越快越小越贵:延迟", fontsize=12, weight="bold")
    for i, v in enumerate(lat):
        a2.text(v, i, f" ~{v}cyc / 容量{['256B/线程','228KB/SM','50MB','80GB'][i]}", va="center", fontsize=8.5)
    fig.tight_layout(); fig.savefig("gpu_memory_hierarchy.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 4) Roofline
def fig_roofline():
    fig, ax = plt.subplots(figsize=(8.2, 5))
    peak = 990.0                 # TFLOP/s (H100 bf16 tensor, 量级)
    bw = 3.35                    # TB/s
    ai = np.logspace(-1, 3.2, 400)
    perf = np.minimum(peak, bw * ai * 1000 / 1000)   # bw(TB/s)*AI(FLOP/B)=TFLOP/s
    perf = np.minimum(peak, bw * ai)                 # TB/s * FLOP/B = TFLOP/s
    ax.loglog(ai, perf, color=C["blue"], lw=2.5)
    ridge = peak / bw
    ax.axvline(ridge, color=C["gray"], ls="--", lw=1)
    ax.text(ridge * 1.1, 5, f"脊点 ridge\n≈{ridge:.0f} FLOP/B", fontsize=9, color=C["gray"])
    pts = {"逐元素/LayerNorm\n(memory-bound)": (2, bw * 2),
           "注意力 Attention": (20, bw * 20),
           "GEMM 矩阵乘\n(compute-bound)": (400, peak)}
    for name, (x, y) in pts.items():
        ax.scatter([x], [min(y, peak)], s=60, zorder=5, color=C["red"])
        ax.annotate(name, (x, min(y, peak)), textcoords="offset points", xytext=(6, -18), fontsize=8.5)
    ax.fill_between(ai, 0.1, perf, where=ai < ridge, alpha=0.08, color=C["green"])
    ax.fill_between(ai, 0.1, perf, where=ai >= ridge, alpha=0.08, color=C["orange"])
    ax.text(0.3, 3, "访存受限\nmemory-bound", color=C["green"], fontsize=10)
    ax.text(500, 30, "算力受限\ncompute-bound", color=C["orange"], fontsize=10)
    ax.set_xlabel("算术强度 Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("可达性能 (TFLOP/s)")
    ax.set_title("Roofline 模型(H100 量级:峰值 ~990 TFLOP/s bf16,HBM ~3.35 TB/s)", fontsize=11, weight="bold")
    ax.grid(True, which="both", alpha=0.25); ax.set_ylim(1, 2000)
    fig.tight_layout(); fig.savefig("roofline.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 5) GPU 层级结构
def fig_gpu_struct():
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("GPU 层级结构:Chip → GPC → SM → Warp / Core", fontsize=14, weight="bold")
    box(ax, 0.2, 0.3, 9.6, 5.2, "", C["blue"], ec="none");
    ax.text(5, 5.15, "GPU 芯片(如 H100:~132 个 SM,80GB HBM3)", ha="center", color="white", fontsize=11)
    for i in range(3):
        box(ax, 0.6 + i * 3.1, 3.0, 2.9, 1.9, "", C["teal"])
        ax.text(2.05 + i * 3.1, 4.7, f"GPC {i}", ha="center", color="#123", fontsize=9)
        for j in range(2):
            box(ax, 0.8 + i * 3.1 + j * 1.45, 3.15, 1.3, 1.35,
                "SM\nwarp调度×4\nCUDA核+张量核\nShared/L1 228KB", C["orange"], 6.5)
    box(ax, 0.6, 1.5, 8.8, 1.0, "L2 Cache(全片共享,~50 MB)", C["green"], 11)
    box(ax, 0.6, 0.5, 8.8, 0.8, "HBM3 显存(~80 GB,~3.35 TB/s)", C["purple"], 11)
    ax.text(9.55, 4.0, "SIMT:一条指令\n驱动 32 线程(warp)\n超额线程隐藏延迟",
            ha="right", fontsize=8, color="white")
    fig.tight_layout(); fig.savefig("gpu_sm_structure.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 6) CPU 缓存层级
def fig_cpu_cache():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    levels = ["L1D\n32KB", "L2\n1MB", "L3\n~32MB", "DRAM\n~百GB"]
    lat_ns = [1.0, 4.0, 15.0, 90.0]
    colors = [C["red"], C["orange"], C["green"], C["blue"]]
    bars = ax.bar(levels, lat_ns, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("访问延迟 (ns,对数轴)")
    ax.set_title("CPU 缓存层级:典型服务器(延迟随容量指数上升)", fontsize=12, weight="bold")
    cyc = ["~4 cyc", "~12 cyc", "~40 cyc", "~200-300 cyc"]
    for b, ns, cc in zip(bars, lat_ns, cyc):
        ax.text(b.get_x() + b.get_width() / 2, ns, f"{ns:g} ns\n{cc}", ha="center", va="bottom", fontsize=9)
    ax.text(0.02, 0.9, "关键洞察:一次 DRAM 缺失 ≈ 上百条指令的时间\n→ 局部性(locality)与缓存友好访问是 CPU 性能的命脉",
            transform=ax.transAxes, fontsize=9, color="#444", va="top")
    fig.tight_layout(); fig.savefig("cpu_cache_hierarchy.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 7) CPU 流水线+乱序
def fig_cpu_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("CPU 超标量乱序流水线(Superscalar Out-of-Order)", fontsize=14, weight="bold")
    stages = ["取指\nFetch", "译码\nDecode", "重命名\nRename", "发射\nDispatch\n(乱序窗口)",
              "执行\nExecute\n(多端口)", "提交\nRetire\n(顺序)"]
    cols = [C["gray"], C["gray"], C["purple"], C["orange"], C["blue"], C["green"]]
    for i, (s, c) in enumerate(zip(stages, cols)):
        box(ax, 0.3 + i * 1.78, 1.5, 1.55, 1.4, s, c, 9)
        if i < 5:
            arrow(ax, 0.3 + i * 1.78 + 1.55, 2.2, 0.3 + (i + 1) * 1.78, 2.2)
    # 执行端口
    for j, p in enumerate(["ALU", "ALU", "Load", "Store", "FP/SIMD"]):
        box(ax, 7.4, 3.2 - j * 0.62, 1.0, 0.5, p, C["teal"], 8, tc="#123")
    ax.text(5.5, 0.7, "乱序:在数据就绪时抢先执行、隐藏内存延迟;分支预测 + 投机执行;最后按程序序顺序提交(保证正确性)",
            ha="center", fontsize=9, color="#444")
    fig.tight_layout(); fig.savefig("cpu_pipeline_ooo.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 8) NUMA
def fig_numa():
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("NUMA:非一致内存访问(多路服务器)", fontsize=14, weight="bold")
    box(ax, 0.5, 1.5, 3.4, 2.6, "", C["blue"]); ax.text(2.2, 3.9, "Socket 0", ha="center", color="white", fontsize=11)
    box(ax, 6.1, 1.5, 3.4, 2.6, "", C["green"]); ax.text(7.8, 3.9, "Socket 1", ha="center", color="white", fontsize=11)
    box(ax, 0.8, 2.6, 2.8, 1.0, "多核 + L1/L2/L3", C["orange"], 9)
    box(ax, 6.4, 2.6, 2.8, 1.0, "多核 + L1/L2/L3", C["orange"], 9)
    box(ax, 0.8, 1.6, 2.8, 0.8, "本地内存 DRAM 0", C["purple"], 9)
    box(ax, 6.4, 1.6, 2.8, 0.8, "本地内存 DRAM 1", C["purple"], 9)
    arrow(ax, 3.9, 2.8, 6.1, 2.8, C["red"], lw=2.5)
    ax.text(5, 3.05, "UPI/跨槽互联\n远程访问更慢", ha="center", fontsize=8.5, color=C["red"])
    ax.text(2.2, 1.2, "本地 ~90ns", ha="center", fontsize=8.5, color=C["purple"])
    ax.text(5, 0.6, "内存绑定 numactl / first-touch:让线程访问本地内存,避免跨槽 → AI 训练数据加载、KV cache 放置都要考虑",
            ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(); fig.savefig("numa.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 9) 内存序 / store buffer
def fig_mem_order():
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("内存重排:store buffer 导致 StoreLoad 乱序(经典 Dekker)", fontsize=13, weight="bold")
    box(ax, 0.5, 3.2, 4.0, 1.2, "线程 1\n  x = 1   (store)\n  r1 = y  (load)", C["blue"], 10)
    box(ax, 5.5, 3.2, 4.0, 1.2, "线程 2\n  y = 1   (store)\n  r2 = x  (load)", C["green"], 10)
    box(ax, 0.5, 1.8, 4.0, 0.9, "Store Buffer 1\n(x=1 还没写回缓存)", C["orange"], 9)
    box(ax, 5.5, 1.8, 4.0, 0.9, "Store Buffer 2\n(y=1 还没写回缓存)", C["orange"], 9)
    box(ax, 2.0, 0.4, 6.0, 0.9, "共享内存 / 缓存一致性域", C["purple"], 10)
    arrow(ax, 2.5, 3.2, 2.5, 2.7, C["orange"]); arrow(ax, 7.5, 3.2, 7.5, 2.7, C["orange"])
    arrow(ax, 2.5, 1.8, 3.5, 1.3, C["gray"]); arrow(ax, 7.5, 1.8, 6.5, 1.3, C["gray"])
    ax.text(5, 2.35, "两个 load 都可能读到旧值 0!\n→ r1==r2==0(顺序一致性 SC 下不可能)",
            ha="center", fontsize=9.5, color=C["red"])
    ax.text(5, 0.05, "修复:内存屏障 memory fence / 原子 acquire-release / std::atomic —— 这就是「内存模型」要解决的",
            ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(); fig.savefig("memory_ordering.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 10) MESI
def fig_mesi():
    fig, ax = plt.subplots(figsize=(7.6, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("缓存一致性 MESI 状态机", fontsize=14, weight="bold")
    pos = {"M": (2, 8, C["red"], "Modified\n已改·独占·脏"),
           "E": (8, 8, C["green"], "Exclusive\n干净·独占"),
           "S": (8, 2, C["blue"], "Shared\n干净·共享"),
           "I": (2, 2, C["gray"], "Invalid\n无效")}
    for k, (x, y, c, t) in pos.items():
        ax.add_patch(plt.Circle((x, y), 1.25, color=c)); ax.text(x, y, k, ha="center", va="center",
                     color="white", fontsize=16, weight="bold")
        ax.text(x, y - 1.7, t, ha="center", fontsize=8.5)
    def edge(a, b, txt, off=0.0):
        (x1, y1, *_), (x2, y2, *_) = pos[a], pos[b]
        arrow(ax, x1, y1, x2, y2, "#555", lw=1.5)
        ax.text((x1 + x2) / 2 + off, (y1 + y2) / 2, txt, fontsize=7.5, color="#333",
                ha="center", bbox=dict(fc="white", ec="none", alpha=0.7))
    edge("I", "E", "读缺失·无其他持有"); edge("I", "S", "读缺失·有其他持有")
    edge("E", "M", "本地写"); edge("S", "M", "本地写(使其他失效)")
    edge("M", "I", "其他核写"); edge("S", "I", "其他核写")
    ax.text(5, 0.3, "一致性:任一时刻一个缓存行的写权限唯一;读到的都是最新值。GPU 也有类似(带 scope 的)一致性域。",
            ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(); fig.savefig("cache_coherence_mesi.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 11) CPU vs GPU
def fig_cpu_vs_gpu():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, title, n, sz, col, sub in [
        (a1, "CPU:延迟导向(few big cores)", 4, 0.9, C["blue"], "大缓存+乱序+分支预测\n把单线程做快"),
        (a2, "GPU:吞吐导向(many small cores)", 12, 0.42, C["green"], "海量轻量核+超额线程\n用并行隐藏延迟")]:
        ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off"); ax.set_title(title, fontsize=12, weight="bold")
        cols = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, cols)
            box(ax, 0.6 + c * (sz + 0.25), 3.4 - r * (sz + 0.25), sz, sz, "核", col, 8)
        box(ax, 0.6, 0.6, 4.6, 1.0, sub, C["gray"], 9)
    fig.suptitle("两种设计哲学", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig("cpu_vs_gpu.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fns = [fig_pd_arch, fig_pd_compare, fig_gpu_mem, fig_roofline, fig_gpu_struct,
           fig_cpu_cache, fig_cpu_pipeline, fig_numa, fig_mem_order, fig_mesi, fig_cpu_vs_gpu]
    for f in fns:
        f(); print("  ok:", f.__name__)
    print("done, generated", len(fns), "figures")
