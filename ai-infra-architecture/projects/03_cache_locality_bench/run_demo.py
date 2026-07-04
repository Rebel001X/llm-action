"""
run_demo.py —— 实测缓存/局部性效应,打印结果并生成 cache_effects.png。
运行:python run_demo.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from locality_bench import contiguous_vs_transpose, stride_effect, working_set_bandwidth

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 58)
    print("本机实测:CPU 缓存 / 局部性效应")
    print("=" * 58)
    tc, tt = contiguous_vs_transpose(3000)
    print(f"连续拷贝 A.copy():   {tc*1000:7.2f} ms")
    print(f"转置拷贝 A.T.copy(): {tt*1000:7.2f} ms   → 慢 {tt/tc:.1f}×(跨步读,缓存不友好)")
    se = stride_effect(m=1_000_000)
    print("\n步长 stride → 求和耗时(ms):")
    for k, v in se.items():
        print(f"  stride={k:>2}: {v*1000:6.2f} ms")
    ws = working_set_bandwidth()
    print("\n工作集大小 → 流式带宽(GB/s):")
    for k, v in ws.items():
        tag = " (DRAM)" if k >= 100000 else " (cache)"
        print(f"  {k:>7} KB: {v:6.1f} GB/s{tag}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar([str(k) for k in se], [v * 1000 for v in se.values()], color="#DD8452")
    a1.axvline(2.5, color="red", ls="--"); a1.text(2.6, max(se.values()) * 1000 * 0.8,
              "缓存行边界\n(stride≈8)", color="red", fontsize=9)
    a1.set_xlabel("步长 stride"); a1.set_ylabel("求和耗时 (ms)")
    a1.set_title("步长效应:越过缓存行后变慢", fontsize=12, weight="bold")
    ks = list(ws.keys()); vs = list(ws.values())
    a2.bar([f"{k}KB" for k in ks], vs, color=["#55A868"] * (len(ks) - 1) + ["#C44E52"])
    a2.set_ylabel("流式带宽 GB/s"); a2.set_title("工作集:缓存 vs DRAM 悬崖", fontsize=12, weight="bold")
    a2.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig("cache_effects.png", bbox_inches="tight"); plt.close(fig)
    print("\n已生成 cache_effects.png")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
