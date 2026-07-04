"""
locality_bench.py —— 在本机**实测** CPU 缓存/局部性效应(对应 ../../03_CPU结构.md)。

三个能在 numpy 里干净复现的效应:
  1) 连续 vs 转置(跨步)访问:转置拷贝要跨步读内存,缓存不友好 → 明显更慢(空间局部性 + 缓存行)
  2) 步长效应:固定读 m 个元素,步长越过缓存行(64B=8 个 float64)后,每元素多占一条缓存行 → 变慢
  3) 工作集大小:数据能装进 cache 时的流式带宽 > 落到 DRAM 时(缓存悬崖)
方法与内存带宽实测(项目 02)一致;这里聚焦"访问模式"而非纯带宽。
"""
from __future__ import annotations
import time
import numpy as np

DTYPE = np.float64
LINE = 64                       # 缓存行 64 字节
ELEMS_PER_LINE = LINE // np.dtype(DTYPE).itemsize   # 8 个 float64/行


def _best(fn, reps=7):
    fn()
    return min((_t(fn) for _ in range(reps)))


def _t(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


# ---------------- 1) 连续 vs 转置 ----------------
def contiguous_vs_transpose(n=4000):
    """A.copy()(连续读)vs A.T.copy()(跨步读)。返回 (t_连续, t_转置)。"""
    A = np.random.rand(n, n).astype(DTYPE)
    t_contig = _best(lambda: A.copy(), reps=5)
    t_trans = _best(lambda: np.ascontiguousarray(A.T), reps=5)
    return t_contig, t_trans


# ---------------- 2) 步长效应 ----------------
def stride_effect(strides=(1, 2, 4, 8, 16, 32), m=2_000_000):
    """从一个大数组里按 stride 取 m 个元素求和;返回 {stride: 耗时秒}。
    stride 越过缓存行(>=8)后,每个有用元素多拖一条缓存行 → 变慢。"""
    big = np.ones(m * max(strides) + 8, DTYPE)
    out = {}
    for s in strides:
        view = big[:m * s:s]
        out[s] = _best(lambda v=view: v.sum(), reps=5)
    return out


# ---------------- 3) 工作集大小(缓存悬崖) ----------------
def working_set_bandwidth(sizes_kb=(16, 256, 4096, 262144), total_bytes=4_000_000_000):
    """对不同大小的数组做等量的流式读写(a+=1),measure GB/s。
    小到能进 cache → 带宽高;大到落 DRAM → 带宽低。返回 {KB: GB/s}。"""
    out = {}
    for kb in sizes_kb:
        n = max(1024, kb * 1024 // np.dtype(DTYPE).itemsize)
        a = np.ones(n, DTYPE)
        reps = max(3, int(total_bytes // (n * a.itemsize)))
        def work(a=a, reps=reps):
            for _ in range(reps):
                a += 1.0
        t = _best(work, reps=3)
        moved = reps * n * a.itemsize * 2          # 读 + 写
        out[kb] = moved / t / 1e9
    return out
