# -*- coding: utf-8 -*-
"""
parallel_patterns.py
====================
用 numpy「模拟」GPU 的并行原语（parallel primitives）。

本模块的目标不是追求单机最快，而是把 GPU 上三大经典并行模式的**思路**
用可读的 numpy 代码显式写出来，并同时给出：

    1) 串行参考实现（serial / sequential）—— 用来对拍（cross-check）正确性；
    2) 并行版本（parallel-style）—— 按 GPU kernel 的「按步（step）推进」思路组织；
    3) 复杂度探针（complexity probe）—— 统计每种算法的 step 数与 work 数。

三大模式：
    - reduction （归约，树形）
    - scan      （前缀和/扫描，Hillis-Steele 与 Blelloch 两种）
    - histogram （直方图，私有化 privatization + 归约）

【核心概念：work 与 step（work-span 模型 / PRAM 模型）】
    - work（工作量 T1）：所有处理器做的**总操作数**，衡量「总电费」。
    - step（步数 / span / depth T∞）：在**无限多处理器**下的关键路径长度，
      衡量「最快能多少步跑完」，也就是并行深度（parallel depth）。
    GPU 之所以快，是因为它能把 work 摊到成千上万个线程上并行执行，
    于是真正决定墙钟时间的往往是 step（O(log n)）而不是 work（O(n)）。

参考：Practical GPU Programming —— 并行模式（parallel patterns）章节。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np


# =============================================================================
# 复杂度探针：记录一次算法执行的 step / work
# =============================================================================
@dataclass
class ComplexityStats:
    """记录一次并行算法执行过程中的复杂度统计。

    属性
    ----
    steps : int
        并行步数（span / depth）。在真正的 GPU 上，同一步内的所有线程并行，
        步与步之间需要一次同步（__syncthreads 或 kernel 边界）。
    work : int
        总工作量（所有「虚拟线程」做的有效算术操作次数之和）。
    n : int
        输入规模。
    """

    n: int = 0
    steps: int = 0
    work: int = 0
    # 每一步实际参与运算的元素个数，便于画「工作量随步衰减」的图
    work_per_step: List[int] = field(default_factory=list)

    def add_step(self, ops: int) -> None:
        """记录一个并行步，本步共做了 ops 次有效运算。"""
        self.steps += 1
        self.work += ops
        self.work_per_step.append(ops)


# =============================================================================
# 模式一：Reduction（归约）—— 树形并行
# =============================================================================
def reduce_serial(x: np.ndarray) -> float:
    """串行归约：一个循环累加，作为正确性基准（ground truth）。

    - 是什么：把数组所有元素用 + 合并成一个标量。
    - work：n-1 次加法；step：n-1（完全串行，深度等于长度）。
    """
    acc = np.array(0.0, dtype=np.float64)
    for v in x:
        acc = acc + v
    return float(acc)


def reduce_tree(x: np.ndarray) -> Tuple[float, ComplexityStats]:
    """树形归约（tree reduction）—— GPU 上归约的标准思路。

    思路（以求和为例，n=8）::

        step0:  a0 a1 a2 a3 a4 a5 a6 a7
                 \\ /   \\ /   \\ /   \\ /       每对相邻元素相加，配对间彼此独立→可并行
        step1:  (a0+a1) (a2+a3) (a4+a5) (a6+a7)
                    \\   /          \\   /
        step2:  (…4项…)            (…4项…)
                        \\        /
        step3:            总和

    - step 数 = ceil(log2 n)  → O(log n)   （关键：深度是对数级）
    - work    = n-1 次加法    → O(n)        （总加法次数不变，和串行一样）

    这就是「work 不变、step 从 O(n) 砍到 O(log n)」的典型例子：
    加法总数省不掉，但把它们排成一棵树，深度就从 n 变成 log n。

    返回 (归约结果, 复杂度统计)。
    """
    # 用 float64 累加，避免 float32 求和的数值误差，方便与串行严格对拍
    buf = x.astype(np.float64).copy()
    stats = ComplexityStats(n=len(x))
    size = len(buf)
    if size == 0:
        return 0.0, stats

    # 每一步把「相邻两半」相加，规模减半（向上取整处理奇数长度）
    while size > 1:
        half = size // 2          # 能配成对的对数
        # 向量化：一次性把前 half 个与「中段 half 个」相加——模拟同一步内多线程并行
        left = buf[:half]
        right = buf[half:2 * half]
        buf[:half] = left + right
        # 奇数长度时，最后一个「落单」元素直接搬到前面，不参与本步加法
        if size % 2 == 1:
            buf[half] = buf[size - 1]
            size = half + 1
        else:
            size = half
        stats.add_step(half)      # 本步做了 half 次加法
    return float(buf[0]), stats


# =============================================================================
# 模式二：Scan（扫描 / 前缀和）
# =============================================================================
def scan_serial_inclusive(x: np.ndarray) -> np.ndarray:
    """串行「包含式」前缀和（inclusive prefix sum）。

    y[i] = x[0] + x[1] + ... + x[i]   （包含自己）
    work = n-1；step = n-1。用作对拍基准。
    """
    y = x.astype(np.float64).copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + y[i]
    return y


def scan_serial_exclusive(x: np.ndarray) -> np.ndarray:
    """串行「排他式」前缀和（exclusive prefix sum）。

    y[0] = 0；y[i] = x[0] + ... + x[i-1]   （不含自己）
    Blelloch scan 天然产出的是 exclusive 形式。
    """
    y = np.zeros(len(x), dtype=np.float64)
    acc = 0.0
    for i in range(len(x)):
        y[i] = acc
        acc += float(x[i])
    return y


def scan_hillis_steele(x: np.ndarray) -> Tuple[np.ndarray, ComplexityStats]:
    """Hillis-Steele scan（inclusive）—— 「步高效」但「工作不省」。

    算法（对每一步 d = 1, 2, 4, 8, ...）::

        for d in 1,2,4,...< n:
            并行 for 所有 i:
                if i >= d:  y[i] = y[i] + y[i-d]

    直观理解：第一步每个元素加上左边 1 个，第二步加上左边 2 个（其实已含 4 个和），
    每步「有效覆盖范围」翻倍，log n 步后每个位置都覆盖了它左边所有元素。

    - step = ceil(log2 n)   → O(log n)   ✅ 步很浅
    - work = Σ (n-d) ≈ n·log n → O(n log n) ⚠️ 比串行的 O(n) 还多！

    这就是经典权衡：Hillis-Steele **step 少但 work 多**（work-inefficient）。
    元素少、追求延迟时用它；元素多、在意吞吐/能耗时用下面的 Blelloch。

    返回 (inclusive 前缀和, 复杂度统计)。
    """
    y = x.astype(np.float64).copy()
    stats = ComplexityStats(n=len(x))
    n = len(y)
    if n <= 1:
        return y, stats

    d = 1
    while d < n:
        # 关键：必须读「旧值」再写，否则同一步内会互相污染。
        # GPU 上通常用双缓冲（double buffering / ping-pong）解决读写冲突；
        # 这里用一份快照 prev 来模拟「本步开始时的旧状态」。
        prev = y.copy()
        # 并行更新所有 i>=d 的位置：y[i] = prev[i] + prev[i-d]
        y[d:] = prev[d:] + prev[:n - d]
        stats.add_step(n - d)     # 本步有 n-d 个位置做了加法
        d *= 2
    return y, stats


def scan_blelloch(x: np.ndarray) -> Tuple[np.ndarray, ComplexityStats]:
    """Blelloch scan（work-efficient，exclusive）—— 「工作省」的两趟扫描。

    分两个阶段，都在「就地（in-place）」的数组上操作，要求 n 是 2 的幂
    （非 2 的幂时先补零到最近的 2 的幂，最后截断）：

    阶段 1：Up-sweep（reduce，自底向上建部分和树）::

        for d in 1,2,4,...,n/2:
            并行 for i in 0, 2d, 4d, ...:
                a[i+2d-1] += a[i+d-1]

    阶段 2：Down-sweep（自顶向下派发前缀）::

        a[n-1] = 0                       # 把根置零（这一步造就 exclusive 语义）
        for d in n/2,...,4,2,1:
            并行 for i in 0, 2d, 4d, ...:
                t = a[i+d-1]
                a[i+d-1] = a[i+2d-1]
                a[i+2d-1] += t

    - step = 2·log2 n         → O(log n)   ✅ 步仍是对数级（up + down 两趟）
    - work = O(n)             ✅✅ 总加法约 2n，和串行同阶，故称 work-efficient

    代价：常数更大、访存/控制更复杂；小数组上反而可能比 Hillis-Steele 慢。

    返回 (exclusive 前缀和, 复杂度统计)。
    """
    stats = ComplexityStats(n=len(x))
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64), stats

    # 补零到最近的 2 的幂：GPU 上常见做法（padding），保证完美二叉树
    m = 1
    while m < n:
        m *= 2
    a = np.zeros(m, dtype=np.float64)
    a[:n] = x.astype(np.float64)

    # ---- 阶段 1：Up-sweep（归约建树）----
    d = 1
    while d < m:
        idx = np.arange(0, m, 2 * d)          # i = 0, 2d, 4d, ...
        a[idx + 2 * d - 1] += a[idx + d - 1]  # 右孩子累加左孩子
        stats.add_step(len(idx))              # 本步做了 len(idx) 次加法
        d *= 2

    # ---- 置根为 0：exclusive 语义的关键一步 ----
    a[m - 1] = 0.0

    # ---- 阶段 2：Down-sweep（派发前缀）----
    d = m // 2
    while d >= 1:
        idx = np.arange(0, m, 2 * d)
        left = idx + d - 1
        right = idx + 2 * d - 1
        t = a[left].copy()                    # 暂存左孩子旧值
        a[left] = a[right]                    # 左孩子 <- 父节点（=左侧前缀）
        a[right] = a[right] + t               # 右孩子 <- 父 + 原左孩子
        stats.add_step(len(idx))              # 每个 idx 一次加法（swap 不计 work）
        d //= 2

    return a[:n], stats


# =============================================================================
# 模式三：Histogram（直方图）—— 私有化（privatization）
# =============================================================================
def histogram_serial(x: np.ndarray, num_bins: int) -> np.ndarray:
    """串行直方图：逐元素给对应 bin 计数 +1，作为对拍基准。

    - work = n（每个元素一次自增）；step = n（串行）。
    - x 中的值必须落在 [0, num_bins) 的整数区间内。
    """
    hist = np.zeros(num_bins, dtype=np.int64)
    for v in x:
        hist[int(v)] += 1
    return hist


def histogram_privatized(
    x: np.ndarray, num_bins: int, num_workers: int = 8
) -> Tuple[np.ndarray, ComplexityStats]:
    """私有化直方图（privatized histogram）—— GPU 上避免原子冲突的经典手法。

    ⚠️ 朴素并行直方图的问题：多个线程同时对同一个 bin 做 `hist[b] += 1`，
    这是一个 read-modify-write，必须用**原子操作（atomicAdd）**，
    热点 bin 上会出现严重的**争用（contention）**，把并行退化成串行。

    私有化思路：给每个线程 / 每个 block 一份**私有副本**（private copy），
    各自无冲突地累加，最后把所有副本用**归约（reduction）**合并。
    这样把「一个全局热点」拆成「多个互不干扰的局部计数」。

        thread0 -> hist0  ┐
        thread1 -> hist1  ├─(树形归约求和)─> 最终直方图
        thread2 -> hist2  ┘
        ...

    - 私有累加阶段：work = n（每个元素只进自己那份副本）；step ≈ n/num_workers（各副本可并行）
    - 合并阶段：对 num_workers 份、每份 num_bins 长的副本做树形归约，
      step ≈ log2(num_workers)。
    本函数用统计口径记录「合并阶段」的 step（体现私有化后合并是 O(log workers)）。

    返回 (直方图, 复杂度统计)。
    """
    stats = ComplexityStats(n=len(x))
    # 1) 分片：把输入切成 num_workers 段，模拟 num_workers 个线程/block
    workers = max(1, int(num_workers))
    # 每个 worker 一份私有直方图（privatization）
    private = np.zeros((workers, num_bins), dtype=np.int64)
    chunks = np.array_split(np.arange(len(x)), workers)
    for w, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        # 各 worker 在自己的私有副本上无冲突累加（用 bincount 向量化模拟）
        vals = x[idx].astype(np.int64)
        private[w] += np.bincount(vals, minlength=num_bins)[:num_bins]

    # 2) 树形归约合并所有私有副本（沿 worker 维度做 O(log workers) 步合并）
    size = workers
    buf = private
    while size > 1:
        half = size // 2
        buf[:half] = buf[:half] + buf[half:2 * half]
        if size % 2 == 1:
            buf[half] = buf[size - 1]
            size = half + 1
        else:
            size = half
        stats.add_step(half * num_bins)   # 本步合并了 half 份、每份 num_bins 个 bin
    hist = buf[0].copy()
    return hist, stats


# =============================================================================
# 复杂度理论值：给测试与画图用的解析公式
# =============================================================================
def ceil_log2(n: int) -> int:
    """返回 ceil(log2 n)（n>=1）。n==1 时为 0。"""
    if n <= 1:
        return 0
    return int(np.ceil(np.log2(n)))


def theoretical_steps(pattern: str, n: int) -> int:
    """给定模式与规模，返回并行 step 数的理论值（用于测试断言与画图）。

    - "reduction"     : ceil(log2 n)
    - "hillis_steele" : ceil(log2 n)
    - "blelloch"      : 2 * log2(m)，m 为 >=n 的最近 2 的幂
    """
    if pattern in ("reduction", "hillis_steele"):
        return ceil_log2(n)
    if pattern == "blelloch":
        m = 1
        while m < n:
            m *= 2
        # up-sweep log2(m) 步 + down-sweep log2(m) 步
        return 2 * (ceil_log2(m) if m > 1 else 0)
    raise ValueError(f"未知模式: {pattern}")
