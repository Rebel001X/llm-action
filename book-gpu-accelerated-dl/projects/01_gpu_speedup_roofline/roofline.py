# -*- coding: utf-8 -*-
"""
roofline.py —— DL 算子 GPU vs CPU 加速比 + Roofline 屋顶线模型（核心库）
================================================================================

本模块做三件事：
  1. 用「真实 GPU 一样的方法」在本机 CPU 上实测两个硬件上限：
       - 峰值算力 FLOP/s（用大矩阵乘 GEMM 逼近，计算密集）
       - 峰值内存带宽 Byte/s（用大向量逐元素读写逼近，访存密集）
  2. 用解析（analytical）模型，把这两个上限套进 Roofline 屋顶线，
     算出任意算子的「性能上限 = min(算力屋顶, 带宽屋顶 × 算术强度)」。
  3. 给出「若把同一算子搬到 GPU」的加速比估算（纯解析，不需要真 GPU）。

为什么这样设计？
  - 你本机没有 GPU、不能联网、不能下模型。但 Roofline 的**方法论**在 CPU/GPU
    上是完全一致的：都是「先测两条屋顶，再看算子落在哪条屋顶下」。
  - 所以我们在 CPU 上把整套方法跑通、把图画出来，再用 GPU 的规格参数（写死在
    表里的公开 spec）做解析对比——这正是工业界做「上卡前评估」的标准手法。

术语对照（中英并列，后文不再重复解释）：
  - FLOP        = Floating-point OPeration          浮点运算（次数）
  - FLOP/s      = 每秒浮点运算次数                    算力 / 吞吐
  - GEMM        = GEneral Matrix Multiply           通用矩阵乘 C = A·B
  - AI          = Arithmetic Intensity              算术强度 = FLOP / Byte
  - Roofline    = 屋顶线模型                          性能上限的可视化
  - Ridge Point = 屋脊点 / 平衡点                      算力屋顶与带宽屋顶的交点
  - Bound       = 受限于……                            compute-bound / memory-bound

作者约定：全部函数纯 numpy + 标准库，零网络、零 GPU、零外部权重。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# 第 0 部分：GPU 规格表（公开 spec，写死，用于解析对比；不联网）
# =============================================================================
# 说明：这些是各卡「厂商标称峰值」，来源为公开数据手册。我们只用它们做
#       「若在 GPU 上会怎样」的解析估算，不代表你一定能打满（实际常打 30~70%）。
#       - fp32_tflops：单精度峰值算力（TFLOP/s，Tensor Core 关掉的 CUDA Core 值）
#       - tf32_tflops：TF32 Tensor Core 峰值（训练常用；仅供参考）
#       - bandwidth_gbps：显存带宽（GB/s，1 GB = 1e9 Byte）
# -----------------------------------------------------------------------------
GPU_SPECS: Dict[str, Dict[str, float]] = {
    # 名称              fp32峰值    tf32峰值    显存带宽(GB/s)
    "A100-40GB":  {"fp32_tflops": 19.5,  "tf32_tflops": 156.0, "bandwidth_gbps": 1555.0},
    "A100-80GB":  {"fp32_tflops": 19.5,  "tf32_tflops": 156.0, "bandwidth_gbps": 2039.0},
    "H100-SXM":   {"fp32_tflops": 67.0,  "tf32_tflops": 494.0, "bandwidth_gbps": 3350.0},
    "V100":       {"fp32_tflops": 15.7,  "tf32_tflops": 0.0,   "bandwidth_gbps": 900.0},
    "RTX-4090":   {"fp32_tflops": 82.6,  "tf32_tflops": 82.6,  "bandwidth_gbps": 1008.0},
    "RTX-3090":   {"fp32_tflops": 35.6,  "tf32_tflops": 35.6,  "bandwidth_gbps": 936.0},
    "T4":         {"fp32_tflops": 8.1,   "tf32_tflops": 0.0,   "bandwidth_gbps": 320.0},
}


# =============================================================================
# 第 1 部分：算术强度（Arithmetic Intensity, AI）—— Roofline 的横轴
# =============================================================================

def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    r"""
    算术强度 AI = 浮点运算次数 / 搬运的字节数。

    数学定义：
        AI = FLOP / Byte    (单位：FLOP/Byte)

    直觉：每从内存搬 1 个字节，能顺带做多少次浮点运算。
      - AI 大  → 计算密集（compute-bound），瓶颈在算力，GPU 的 Tensor Core 才吃得饱。
      - AI 小  → 访存密集（memory-bound），瓶颈在带宽，算力再强也在等数据。

    这是 Roofline 图的**横轴**。它把「算子的本质属性」抽象成一个标量。

    参数
    ----
    flops       : 该算子的浮点运算总次数（次）
    bytes_moved : 该算子需要从/向内存搬运的字节数（Byte）

    返回
    ----
    AI（FLOP/Byte）；bytes_moved<=0 时返回 0.0（无搬运不谈强度）。
    """
    if bytes_moved <= 0:
        return 0.0
    return float(flops) / float(bytes_moved)


def gemm_flops(m: int, n: int, k: int) -> float:
    r"""
    GEMM C[m,n] = A[m,k] · B[k,n] 的浮点运算次数。

    第一性原理推导：
      - 输出 C 有 m*n 个元素。
      - 每个元素是长度 k 的点积：k 次乘法 + (k-1) 次加法 ≈ 2k 次浮点运算。
      - 惯例上把「一次乘加 (MAC)」记成 2 FLOP，故总量 ≈ 2 * m * n * k。

    公式：
        FLOP_gemm = 2 * m * n * k
    """
    return 2.0 * m * n * k


def gemm_bytes(m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    r"""
    GEMM 的最小内存搬运量（读 A、读 B、写 C，各一次；理想无重复读）。

    公式：
        Byte_gemm = (m*k + k*n + m*n) * dtype_bytes
      - 读 A：m*k 个元素
      - 读 B：k*n 个元素
      - 写 C：m*n 个元素
    dtype_bytes：fp32=4, fp16/bf16=2, fp64=8。

    ⚠️ 这是「理想下界」。真实 GEMM 因分块（tiling）会重复读，实际 Byte 更大，
       但 AI 的量级结论不变——大方阵 GEMM 的 AI 随 N 线性增长，是典型 compute-bound。
    """
    return (m * k + k * n + m * n) * dtype_bytes


def elementwise_flops(n: int, ops_per_elem: float = 1.0) -> float:
    r"""
    逐元素（element-wise）算子的浮点运算次数，如 ReLU/加法/缩放。

    公式：
        FLOP_ew = n * ops_per_elem
      - ReLU：约 1 op/elem（一次比较取 max）
      - y = a*x + b（AXPY）：约 2 op/elem（一乘一加）
    """
    return float(n) * ops_per_elem


def elementwise_bytes(n: int, n_read: int = 1, n_write: int = 1,
                      dtype_bytes: int = 4) -> float:
    r"""
    逐元素算子的内存搬运量。

    公式：
        Byte_ew = (n_read + n_write) * n * dtype_bytes
      - ReLU(x)：读 x、写 y → n_read=1, n_write=1 → 2*n*dtype_bytes
      - z = x + y：读 x、读 y、写 z → n_read=2, n_write=1 → 3*n*dtype_bytes

    🔬 关键洞察：逐元素算子 FLOP≈n、Byte≈几倍 n，所以 AI ≈ 常数（不随 n 变），
       且这个常数很小（ReLU 的 AI ≈ 1/(2*4) = 0.125 FLOP/Byte）。
       → 逐元素算子永远是 memory-bound，这就是为什么「算子融合(kernel fusion)」
         如此重要：把多个逐元素算子合并，省掉中间结果的反复读写。
    """
    return (n_read + n_write) * n * dtype_bytes


# =============================================================================
# 第 2 部分：Roofline 屋顶线核心公式 —— 给定硬件，任意 AI 的性能上限
# =============================================================================

def roofline_perf(ai: float, peak_flops: float, peak_bw: float) -> float:
    r"""
    Roofline 屋顶线：给定算术强度 AI，可达到的**性能上限**（FLOP/s）。

    核心公式（Williams et al., 2009, "Roofline: An Insightful Visual
    Performance Model for Multicore Architectures"）：

        Attainable_FLOPs = min( peak_flops,  peak_bw * AI )
                                 └─算力屋顶─┘  └──带宽斜坡──┘

    两段理解：
      - 当 AI 小：peak_bw*AI < peak_flops，取带宽项 → 斜线（memory-bound）。
        性能随 AI 线性上升，因为你搬得多就能算得多，但一直在等内存。
      - 当 AI 大：peak_bw*AI > peak_flops，取算力项 → 水平线（compute-bound）。
        性能被算力封顶，再增大 AI 也没用，算力单元已打满。

    两段的交点就是「屋脊点 Ridge Point」，见 ridge_point()。

    参数
    ----
    ai         : 算术强度（FLOP/Byte）
    peak_flops : 峰值算力（FLOP/s）
    peak_bw    : 峰值带宽（Byte/s）

    返回
    ----
    可达性能上限（FLOP/s）
    """
    return min(peak_flops, peak_bw * ai)


def ridge_point(peak_flops: float, peak_bw: float) -> float:
    r"""
    屋脊点（Ridge Point）：算力屋顶与带宽斜坡的交点处的 AI。

    在交点上：peak_bw * AI = peak_flops  →  AI* = peak_flops / peak_bw

    意义（面试高频）：
      - AI < AI*  → 该硬件上此算子是 memory-bound（在斜坡左侧）。
      - AI > AI*  → compute-bound（在水平段下方）。
      - AI* 越大，说明这块硬件「越难喂饱」——算力很强但带宽跟不上，
        需要 AI 很高的算子（大 GEMM）才能打满算力。
        H100 的 AI*（TF32）非常高，正是「带宽墙(memory wall)」的体现。
    """
    if peak_bw <= 0:
        return float("inf")
    return peak_flops / peak_bw


def is_compute_bound(ai: float, peak_flops: float, peak_bw: float) -> bool:
    """该算子在此硬件上是否 compute-bound（算力受限）。AI≥屋脊点即为真。"""
    return ai >= ridge_point(peak_flops, peak_bw)


# =============================================================================
# 第 3 部分：本机实测 —— 用「真 GPU 一样的方法」测 CPU 的算力与带宽
# =============================================================================
# 方法论（和 NVIDIA nsight / cutlass profiler 一致）：
#   - 测算力：跑一个足够大的 GEMM（compute-bound），FLOP/耗时 = 实测 FLOP/s。
#   - 测带宽：跑一个足够大的逐元素读写（memory-bound），Byte/耗时 = 实测 Byte/s。
#   - 都要「预热(warm-up)」消除首次分配/JIT 抖动，再取多次中位数抗噪。
# -----------------------------------------------------------------------------

@dataclass
class BenchResult:
    """一次基准测试的结果容器。"""
    name: str
    seconds: float           # 单次最优/中位耗时（秒）
    flops: float             # 该操作的浮点运算次数
    bytes_moved: float       # 该操作搬运的字节数
    achieved_flops: float    # 实测算力 = flops / seconds（FLOP/s）
    achieved_bw: float       # 实测带宽 = bytes_moved / seconds（Byte/s）

    @property
    def gflops(self) -> float:
        """实测算力，单位 GFLOP/s（1e9 FLOP/s）。"""
        return self.achieved_flops / 1e9

    @property
    def gbps(self) -> float:
        """实测带宽，单位 GB/s（1e9 Byte/s）。"""
        return self.achieved_bw / 1e9

    @property
    def arithmetic_intensity(self) -> float:
        return arithmetic_intensity(self.flops, self.bytes_moved)


def _time_op(op, repeats: int = 5) -> float:
    """
    计时小工具：先预热一次（丢弃），再跑 repeats 次取**中位数**（抗噪）。

    为什么取中位数不取最小值？
      - 最小值容易受「恰好没被系统调度打断」的幸运样本影响，偏乐观。
      - 中位数对偶发的 OS 抖动更稳健，是 profiler 常用的稳健统计量。
    这里用 perf_counter（高精度、不受系统时钟调整影响）。
    """
    op()  # warm-up：触发内存分配、cache 预热、numpy 内部线程池启动
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        op()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    samples.sort()
    return samples[len(samples) // 2]


def measure_gemm_flops(n: int = 2048, repeats: int = 5,
                      dtype=np.float32) -> BenchResult:
    r"""
    实测本机峰值算力：跑一个 n×n × n×n 的方阵 GEMM。

    为什么用 GEMM 测算力？
      - 方阵 GEMM 的 AI = 2n³ / (3n²·4) = n/6 FLOP/Byte，随 n 线性增大，
        当 n 足够大（这里默认 2048，AI≈341）时远超 CPU 屋脊点 → compute-bound，
        计时到的就是「算力上限」而非「带宽上限」。这正是测峰值算力的正确姿势。
      - numpy 的 @ 底层调用高度优化的 BLAS（OpenBLAS/MKL），多线程 + SIMD +
        cache 分块，是 CPU 上能打满算力的少数算子之一——和 GPU 用 cuBLAS 同理。

    返回 BenchResult，其中 achieved_flops 即本机实测算力（FLOP/s）。
    """
    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)

    def op():
        # np.matmul 会分配新数组返回；这就是我们要计时的对象
        return a @ b

    sec = _time_op(op, repeats)
    flops = gemm_flops(n, n, n)
    bytes_moved = gemm_bytes(n, n, n, dtype_bytes=np.dtype(dtype).itemsize)
    return BenchResult(
        name=f"GEMM {n}x{n}",
        seconds=sec,
        flops=flops,
        bytes_moved=bytes_moved,
        achieved_flops=flops / sec,
        achieved_bw=bytes_moved / sec,
    )


def measure_memory_bandwidth(n: int = 1 << 24, repeats: int = 5,
                            dtype=np.float32) -> BenchResult:
    r"""
    实测本机峰值内存带宽：跑一个超大向量的逐元素读写（STREAM-Add: z = x + y）。

    为什么用逐元素测带宽？
      - 逐元素加法的 AI = 1 FLOP / (3 数组·4 Byte) ≈ 0.083 FLOP/Byte，
        远低于 CPU 屋脊点 → memory-bound，计时到的就是「带宽上限」。
      - n 默认 2^24 ≈ 1678 万元素 ≈ 64 MB/数组，远大于 CPU 的 L3 cache（几十 MB），
        保证真的在打主存 DRAM 而不是缓存命中——否则测出的是 cache 带宽（虚高）。

    字节统计：读 x、读 y、写 z ⇒ 恰好 3 次数组访问（STREAM Add 的经典口径，
    每次访问 n 个元素，无临时数组污染，字节数与代码一一对应，最干净）。

    返回 BenchResult，其中 achieved_bw 即本机实测带宽（Byte/s）。
    """
    x = np.random.rand(n).astype(dtype)
    y = np.random.rand(n).astype(dtype)
    z = np.empty_like(x)  # 预分配输出，避免把内存分配开销掺进带宽计时

    def op_add():
        np.add(x, y, out=z)  # z = x + y：读 x、读 y、写 z —— 三股 DRAM 流量
        return z

    sec = _time_op(op_add, repeats)
    n_arrays = 3  # 读 x、读 y、写 z
    bytes_moved = n_arrays * n * np.dtype(dtype).itemsize
    flops = elementwise_flops(n, ops_per_elem=1.0)  # 一次加法
    return BenchResult(
        name=f"Add n={n}",
        seconds=sec,
        flops=flops,
        bytes_moved=bytes_moved,
        achieved_flops=flops / sec,
        achieved_bw=bytes_moved / sec,
    )


# =============================================================================
# 第 4 部分：GPU vs CPU 加速比估算（解析，不需要真 GPU）
# =============================================================================

@dataclass
class SpeedupEstimate:
    """一个算子在 CPU 实测 vs GPU 解析下的加速比结果。"""
    op_name: str
    ai: float
    cpu_attainable_flops: float   # CPU 上此 AI 的 Roofline 上限（用实测屋顶）
    gpu_attainable_flops: float   # GPU 上此 AI 的 Roofline 上限（用 spec 屋顶）
    speedup: float                # gpu / cpu
    cpu_bound: str                # "compute" or "memory"
    gpu_bound: str


def estimate_speedup(ai: float,
                    cpu_peak_flops: float, cpu_peak_bw: float,
                    gpu_peak_flops: float, gpu_peak_bw: float,
                    op_name: str = "op") -> SpeedupEstimate:
    r"""
    估算某算子「从 CPU 搬到 GPU」的理论加速比。

    做法：把同一个 AI 分别代入 CPU 和 GPU 的 Roofline，取两者上限之比。

        speedup = roofline_perf(AI, GPU屋顶) / roofline_perf(AI, CPU屋顶)

    这抓住了 Roofline 的精髓——**加速比取决于算子落在哪条屋顶下**：
      - 若算子 compute-bound：加速比 ≈ GPU算力 / CPU算力（算力比）。
      - 若算子 memory-bound ：加速比 ≈ GPU带宽 / CPU带宽（带宽比）。
      - 两者往往差很多！GPU 算力比 CPU 高几十倍，但带宽比只高几倍。
        → 这解释了「为什么逐元素算子上 GPU 加速有限，而大 GEMM 加速惊人」。
    """
    cpu_perf = roofline_perf(ai, cpu_peak_flops, cpu_peak_bw)
    gpu_perf = roofline_perf(ai, gpu_peak_flops, gpu_peak_bw)
    speedup = gpu_perf / cpu_perf if cpu_perf > 0 else float("inf")
    return SpeedupEstimate(
        op_name=op_name,
        ai=ai,
        cpu_attainable_flops=cpu_perf,
        gpu_attainable_flops=gpu_perf,
        speedup=speedup,
        cpu_bound="compute" if is_compute_bound(ai, cpu_peak_flops, cpu_peak_bw) else "memory",
        gpu_bound="compute" if is_compute_bound(ai, gpu_peak_flops, gpu_peak_bw) else "memory",
    )


# =============================================================================
# 第 5 部分：典型 DL 算子清单（用于打点到 Roofline 图上）
# =============================================================================

@dataclass
class OpPoint:
    """一个待打点的算子：名字 + FLOP + Byte（→ 自动算 AI）。"""
    name: str
    flops: float
    bytes_moved: float
    marker: str = "o"

    @property
    def ai(self) -> float:
        return arithmetic_intensity(self.flops, self.bytes_moved)


def typical_dl_ops(dtype_bytes: int = 4) -> List[OpPoint]:
    r"""
    返回一组典型 DL 算子的打点，覆盖从 memory-bound 到 compute-bound 的谱系。

    覆盖：
      - ReLU / 逐元素加法      ：极低 AI，铁定 memory-bound
      - LayerNorm（近似）      ：低 AI，memory-bound
      - 小 GEMM（512）         ：中 AI
      - 大 GEMM（4096）        ：高 AI，compute-bound
      - 注意力 QK^T（近似）    ：中高 AI
    这些点撒在 Roofline 上，一眼看清「谁卡带宽、谁卡算力」。
    """
    pts: List[OpPoint] = []

    # ReLU: 1024^2 元素，1 op/elem，读1写1
    n = 1024 * 1024
    pts.append(OpPoint("ReLU (1M)",
                      elementwise_flops(n, 1.0),
                      elementwise_bytes(n, 1, 1, dtype_bytes), "s"))

    # 逐元素加法 z=x+y: 读2写1
    pts.append(OpPoint("Add (1M)",
                      elementwise_flops(n, 1.0),
                      elementwise_bytes(n, 2, 1, dtype_bytes), "v"))

    # LayerNorm 近似：每元素 ~5 op，读1写1（忽略统计量二次遍历）
    pts.append(OpPoint("LayerNorm (1M)",
                      elementwise_flops(n, 5.0),
                      elementwise_bytes(n, 1, 1, dtype_bytes), "D"))

    # 小 GEMM 512x512x512
    m = k = nn = 512
    pts.append(OpPoint("GEMM 512",
                      gemm_flops(m, nn, k),
                      gemm_bytes(m, nn, k, dtype_bytes), "^"))

    # 大 GEMM 4096
    m = k = nn = 4096
    pts.append(OpPoint("GEMM 4096",
                      gemm_flops(m, nn, k),
                      gemm_bytes(m, nn, k, dtype_bytes), "*"))

    # 注意力 QK^T：seq=1024, d=64 → [1024,64]·[64,1024]
    s, d = 1024, 64
    pts.append(OpPoint("Attn QK^T",
                      gemm_flops(s, s, d),
                      gemm_bytes(s, s, d, dtype_bytes), "P"))

    return pts


# =============================================================================
# 便捷入口：一次性跑完实测 + 汇总（供 run_demo 与测试复用）
# =============================================================================

@dataclass
class MachineProfile:
    """本机实测的完整画像。"""
    gemm: BenchResult
    mem: BenchResult

    @property
    def peak_flops(self) -> float:
        """本机实测峰值算力（FLOP/s）——来自大 GEMM。"""
        return self.gemm.achieved_flops

    @property
    def peak_bw(self) -> float:
        """本机实测峰值带宽（Byte/s）——来自 AXPY。"""
        return self.mem.achieved_bw

    @property
    def ridge(self) -> float:
        """本机屋脊点 AI*。"""
        return ridge_point(self.peak_flops, self.peak_bw)


def profile_machine(gemm_n: int = 2048, mem_n: int = 1 << 24,
                   repeats: int = 5) -> MachineProfile:
    """跑完两项实测，返回本机画像（算力屋顶 + 带宽屋顶）。"""
    gemm = measure_gemm_flops(n=gemm_n, repeats=repeats)
    mem = measure_memory_bandwidth(n=mem_n, repeats=repeats)
    return MachineProfile(gemm=gemm, mem=mem)


if __name__ == "__main__":
    # 简易自检：直接 python roofline.py 就能看到本机屋顶
    prof = profile_machine()
    print(f"[实测] 峰值算力  : {prof.gemm.gflops:8.1f} GFLOP/s  (来自 {prof.gemm.name})")
    print(f"[实测] 峰值带宽  : {prof.mem.gbps:8.1f} GB/s      (来自 {prof.mem.name})")
    print(f"[实测] 屋脊点 AI*: {prof.ridge:8.1f} FLOP/Byte")
