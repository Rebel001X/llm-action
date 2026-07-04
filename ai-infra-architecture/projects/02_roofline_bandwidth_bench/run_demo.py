"""
run_demo.py —— 实测本机内存带宽 + 算力,打印结果并生成 measured_roofline.png。
运行:python run_demo.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench import (measure_bandwidth, measure_gflops, arithmetic_intensity_triad,
                   arithmetic_intensity_matmul, roofline_perf)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 60)
    print("本机实测:内存带宽 & 算力(CPU/DRAM;方法与 GPU 一致)")
    print("=" * 60)
    bw = measure_bandwidth()
    print(f"{'算子':<10}{'带宽 GB/s':>12}")
    for name, (gbps, _) in bw.items():
        print(f"{name:<10}{gbps:>12.1f}")
    peak_bw = max(g for g, _ in bw.values())          # GB/s
    gflops, _ = measure_gflops()
    print(f"\n矩阵乘算力(peak,实测): {gflops:.1f} GFLOP/s")
    print(f"内存带宽(peak,实测): {peak_bw:.1f} GB/s")
    ridge = gflops / peak_bw
    print(f"脊点 ridge point: {ridge:.1f} FLOP/Byte")

    # ---- 画实测 roofline ----
    ai = np.logspace(-2, 3, 300)
    perf = [roofline_perf(x, gflops, peak_bw) for x in ai]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ai, perf, color="#4C72B0", lw=2.5, label="Roofline(本机实测)")
    ax.axvline(ridge, color="gray", ls="--", lw=1)
    ax.text(ridge * 1.1, min(perf) * 3, f"脊点≈{ridge:.0f}", color="gray", fontsize=9)
    # 标注实测算子点
    pts = [("triad(访存受限)", arithmetic_intensity_triad(),
            bw["triad"][0] * arithmetic_intensity_triad()),
           ("矩阵乘(算力受限)", arithmetic_intensity_matmul(1200), gflops)]
    for name, x, y in pts:
        ax.scatter([x], [min(y, gflops)], s=70, color="#C44E52", zorder=5)
        ax.annotate(name, (x, min(y, gflops)), textcoords="offset points", xytext=(6, -14), fontsize=9)
    ax.set_xlabel("算术强度 FLOP/Byte"); ax.set_ylabel("可达性能 GFLOP/s")
    ax.set_title("本机实测 Roofline(numpy on CPU/DRAM)", fontsize=12, weight="bold")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig("measured_roofline.png", bbox_inches="tight"); plt.close(fig)
    print("\n已生成 measured_roofline.png")
    print("解读:triad 算术强度 ~1/12,死死卡在带宽屋檐;矩阵乘算术强度高,逼近算力屋顶。")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
