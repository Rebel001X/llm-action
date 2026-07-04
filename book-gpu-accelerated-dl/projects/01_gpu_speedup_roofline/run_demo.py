# -*- coding: utf-8 -*-
"""
run_demo.py —— 一键演示：本机实测 Roofline + GPU vs CPU 加速比估算
================================================================================
运行：python run_demo.py

会做三件事并全部打印/出图：
  1. 【实测】用大 GEMM 测本机峰值算力、用大向量加法测本机峰值带宽（真 GPU 同款方法）。
  2. 【画图】把本机两条屋顶画成 Roofline，并把典型 DL 算子（ReLU/GEMM/Attn…）打点上去，
            一眼看清谁 memory-bound、谁 compute-bound。
  3. 【估算】对每个算子做「若搬到 A100 / H100 / RTX-4090」的解析加速比，画柱状图。

输出图片（保存在本目录）：
  - roofline_cpu.png        本机实测 Roofline + 算子打点
  - speedup_bars.png        典型算子在不同 GPU 上的估算加速比
  - roofline_gpu_compare.png CPU vs 三块 GPU 的屋顶线对比

⚠️ 全程离线、无 GPU、无网络。matplotlib 用 Agg 后端，不弹窗，直接存文件。
"""

import os
import sys

# Windows 控制台默认 GBK，遇到 emoji/特殊字符会 UnicodeEncodeError。
# 把标准输出重配为 UTF-8（errors="replace" 兜底），保证任何终端都能跑完。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib

# 【硬性要求】非交互后端 + 中文字体，务必在 pyplot 之前设置
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import roofline as rl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 图 1：本机实测 Roofline + 典型算子打点
# =============================================================================

def plot_cpu_roofline(prof: rl.MachineProfile, out_path: str) -> None:
    peak_flops = prof.peak_flops   # FLOP/s
    peak_bw = prof.peak_bw         # Byte/s
    ridge = prof.ridge             # 屋脊点 AI*

    # 横轴 AI 范围：从很低（memory-bound）到很高（compute-bound）
    ai = np.logspace(-2, 4, 400)  # 0.01 ~ 10000 FLOP/Byte
    # 屋顶线：逐点 min(算力, 带宽*AI)，转成 GFLOP/s 方便读
    roof = np.minimum(peak_flops, peak_bw * ai) / 1e9

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(ai, roof, color="#c62828", lw=2.5, label="本机 Roofline 屋顶线")

    # 标注两段
    ax.axhline(peak_flops / 1e9, color="#1565c0", ls="--", lw=1.2,
              label=f"算力屋顶 = {peak_flops/1e9:.0f} GFLOP/s (实测)")
    ax.axvline(ridge, color="#2e7d32", ls=":", lw=1.5,
              label=f"屋脊点 AI* = {ridge:.1f} FLOP/Byte")

    # 带宽斜坡文字（放在斜线中段）
    ax.text(0.03, peak_bw * 0.03 / 1e9 * 1.3,
            f"带宽斜坡\n{peak_bw/1e9:.0f} GB/s (实测)",
            color="#6a1b9a", fontsize=9, rotation=32)

    # 打点：典型 DL 算子
    pts = rl.typical_dl_ops()
    for p in pts:
        attain = rl.roofline_perf(p.ai, peak_flops, peak_bw) / 1e9
        bound = "compute" if rl.is_compute_bound(p.ai, peak_flops, peak_bw) else "memory"
        color = "#ef6c00" if bound == "compute" else "#00838f"
        ax.scatter(p.ai, attain, s=90, marker=p.marker, color=color,
                  edgecolors="black", zorder=5)
        ax.annotate(p.name, (p.ai, attain),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("算术强度 Arithmetic Intensity (FLOP/Byte)  →  越右越算力密集")
    ax.set_ylabel("可达性能 (GFLOP/s)")
    ax.set_title("本机实测 Roofline 屋顶线模型\n"
                "橙点=compute-bound(卡算力)   青点=memory-bound(卡带宽)")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# 图 2：典型算子在不同 GPU 上的估算加速比（柱状图）
# =============================================================================

def plot_speedup_bars(prof: rl.MachineProfile, out_path: str,
                     gpus=("A100-40GB", "H100-SXM", "RTX-4090")) -> None:
    cpu_pf, cpu_bw = prof.peak_flops, prof.peak_bw
    pts = rl.typical_dl_ops()
    op_names = [p.name for p in pts]

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(gpus)
    x = np.arange(len(pts))

    for gi, gpu in enumerate(gpus):
        spec = rl.GPU_SPECS[gpu]
        # 用 TF32（若有）代表 GPU 的实际训练算力，否则退回 fp32
        gpu_flops = (spec["tf32_tflops"] or spec["fp32_tflops"]) * 1e12
        gpu_bw = spec["bandwidth_gbps"] * 1e9
        speedups = []
        for p in pts:
            est = rl.estimate_speedup(p.ai, cpu_pf, cpu_bw, gpu_flops, gpu_bw, p.name)
            speedups.append(est.speedup)
        bars = ax.bar(x + gi * width, speedups, width, label=gpu)
        # 顶部标数值
        for b, s in zip(bars, speedups):
            ax.annotate(f"{s:.0f}x", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)

    ax.set_xticks(x + width * (len(gpus) - 1) / 2)
    ax.set_xticklabels(op_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("估算加速比 (GPU / 本机 CPU，越高越好)")
    ax.set_title("若把算子搬到 GPU：解析加速比估算\n"
                "注意——大 GEMM(compute-bound) 加速远大于 ReLU/Add(memory-bound)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.legend(title="目标 GPU", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# 图 3：CPU vs 三块 GPU 的屋顶线对比
# =============================================================================

def plot_gpu_compare(prof: rl.MachineProfile, out_path: str,
                    gpus=("A100-40GB", "H100-SXM", "RTX-4090")) -> None:
    ai = np.logspace(-2, 4, 400)
    fig, ax = plt.subplots(figsize=(9, 6))

    # 本机 CPU
    roof_cpu = np.minimum(prof.peak_flops, prof.peak_bw * ai) / 1e12  # TFLOP/s
    ax.loglog(ai, roof_cpu, lw=2.5, color="black",
              label=f"本机 CPU (实测 {prof.peak_flops/1e12:.2f} TF, {prof.peak_bw/1e9:.0f} GB/s)")

    for gpu in gpus:
        spec = rl.GPU_SPECS[gpu]
        gpu_flops = (spec["tf32_tflops"] or spec["fp32_tflops"]) * 1e12
        gpu_bw = spec["bandwidth_gbps"] * 1e9
        roof = np.minimum(gpu_flops, gpu_bw * ai) / 1e12
        ax.loglog(ai, roof, lw=2.0, ls="--",
                  label=f"{gpu} ({gpu_flops/1e12:.0f} TF, {spec['bandwidth_gbps']:.0f} GB/s)")

    ax.set_xlabel("算术强度 (FLOP/Byte)")
    ax.set_ylabel("可达性能 (TFLOP/s)")
    ax.set_title("屋顶线对比：本机 CPU vs 主流 GPU\n"
                "GPU 的水平段(算力)高出几十倍，斜坡段(带宽)只高几倍")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("  DL 算子 GPU vs CPU 加速比 + Roofline —— 本机实测演示")
    print("=" * 70)

    print("\n[1/3] 正在实测本机屋顶（大 GEMM 测算力 + 大向量加法测带宽）...")
    prof = rl.profile_machine(gemm_n=2048, mem_n=1 << 24, repeats=5)
    print(f"      实测峰值算力 : {prof.gemm.gflops:8.1f} GFLOP/s  "
          f"(来自 {prof.gemm.name}, AI={prof.gemm.arithmetic_intensity:.0f}, "
          f"{'compute-bound' if prof.gemm.arithmetic_intensity>prof.ridge else 'memory-bound'})")
    print(f"      实测峰值带宽 : {prof.mem.gbps:8.1f} GB/s      "
          f"(来自 {prof.mem.name}, AI={prof.mem.arithmetic_intensity:.3f}, "
          f"{'memory-bound' if prof.mem.arithmetic_intensity<prof.ridge else 'compute-bound'})")
    print(f"      本机屋脊点   : {prof.ridge:8.1f} FLOP/Byte  "
          f"(AI 高于它=卡算力，低于它=卡带宽)")

    print("\n[2/3] 典型 DL 算子落点分析（本机）:")
    print(f"      {'算子':<16}{'AI(FLOP/Byte)':>16}{'瓶颈':>12}{'可达(GFLOP/s)':>16}")
    for p in rl.typical_dl_ops():
        bound = "compute" if rl.is_compute_bound(p.ai, prof.peak_flops, prof.peak_bw) else "memory"
        attain = rl.roofline_perf(p.ai, prof.peak_flops, prof.peak_bw) / 1e9
        print(f"      {p.name:<16}{p.ai:>16.3f}{bound:>12}{attain:>16.1f}")

    print("\n[3/3] 若搬到 GPU 的解析加速比（TF32 算力 / 实测 CPU）:")
    gpus = ("A100-40GB", "H100-SXM", "RTX-4090")
    header = "      {:<16}".format("算子") + "".join(f"{g:>14}" for g in gpus)
    print(header)
    for p in rl.typical_dl_ops():
        row = f"      {p.name:<16}"
        for gpu in gpus:
            spec = rl.GPU_SPECS[gpu]
            gpu_flops = (spec["tf32_tflops"] or spec["fp32_tflops"]) * 1e12
            gpu_bw = spec["bandwidth_gbps"] * 1e9
            est = rl.estimate_speedup(p.ai, prof.peak_flops, prof.peak_bw,
                                     gpu_flops, gpu_bw, p.name)
            row += f"{est.speedup:>12.0f}x"
        print(row)

    # ---- 出图 ----
    p1 = os.path.join(HERE, "roofline_cpu.png")
    p2 = os.path.join(HERE, "speedup_bars.png")
    p3 = os.path.join(HERE, "roofline_gpu_compare.png")
    plot_cpu_roofline(prof, p1)
    plot_speedup_bars(prof, p2, gpus)
    plot_gpu_compare(prof, p3, gpus)

    print("\n已生成图片：")
    for p in (p1, p2, p3):
        print(f"  - {p}")
    print("\n完成 ✅  （提示：大 GEMM 加速比远大于 ReLU/Add，正是 Roofline 的核心洞察）")


if __name__ == "__main__":
    main()
