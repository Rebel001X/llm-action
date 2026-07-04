"""
bench.py —— 在本机**实测**内存带宽与算力,并据此画 Roofline(对应 ../../02_GPU结构.md)。

用 numpy 在大数组上跑访存受限算子(STREAM 风格)测带宽 GB/s,用大矩阵乘测算力 GFLOP/s,
再算每个算子的算术强度 FLOP/Byte,标到 roofline 上——直观看到谁 memory-bound、谁 compute-bound。
这测的是 **CPU/DRAM**(本机无 GPU),但方法与 GPU 完全一致。
"""
from __future__ import annotations
import time
import numpy as np

DTYPE = np.float64
BYTES = np.dtype(DTYPE).itemsize      # 8


def _best_time(fn, reps=7):
    fn()                               # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)                     # 取最快一次,减少噪声


def measure_bandwidth(n=8_000_000):
    """返回 {算子: (GB/s, 移动字节数)}。n 取大以越过 cache 落到 DRAM。"""
    a = np.ones(n, DTYPE); b = np.ones(n, DTYPE); c = np.empty(n, DTYPE)
    out = {}
    # copy: 读 N + 写 N = 2N
    t = _best_time(lambda: np.copyto(c, a)); out["copy"] = (2 * n * BYTES / t / 1e9, 2 * n * BYTES)
    # scale: c = 2*a, 读 N + 写 N = 2N
    t = _best_time(lambda: np.multiply(a, 2.0, out=c)); out["scale"] = (2 * n * BYTES / t / 1e9, 2 * n * BYTES)
    # add: c = a + b, 读 2N + 写 N = 3N
    t = _best_time(lambda: np.add(a, b, out=c)); out["add"] = (3 * n * BYTES / t / 1e9, 3 * n * BYTES)
    # triad(STREAM): a = b + 3*c, 读 2N + 写 N = 3N
    tmp = np.empty(n, DTYPE)
    def triad():
        np.multiply(c, 3.0, out=tmp); np.add(b, tmp, out=a)
    t = _best_time(triad); out["triad"] = (3 * n * BYTES / t / 1e9, 3 * n * BYTES)
    return out


def measure_gflops(n=1200):
    """大矩阵乘 C=A@B 测算力。FLOP = 2*n^3。"""
    a = np.random.rand(n, n).astype(DTYPE); b = np.random.rand(n, n).astype(DTYPE)
    t = _best_time(lambda: a @ b, reps=3)
    flop = 2 * n ** 3
    return flop / t / 1e9, flop


def arithmetic_intensity_triad():
    """triad: 2 FLOP(一乘一加) / 24 字节(3 个 float64) = 1/12 FLOP/Byte。"""
    return 2.0 / (3 * BYTES)


def arithmetic_intensity_matmul(n):
    """矩阵乘: 2n^3 FLOP / (3n^2 * 8) 字节 = n/12 FLOP/Byte。"""
    return (2 * n ** 3) / (3 * n ** 2 * BYTES)


def roofline_perf(ai, peak_gflops, peak_gbps):
    """给定算术强度,roofline 可达性能 = min(峰值算力, 峰值带宽×算术强度)。"""
    return min(peak_gflops, peak_gbps * ai)
