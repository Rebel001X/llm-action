# -*- coding: utf-8 -*-
"""
test_parallel_patterns.py
=========================
对三大并行模式做全面测试：
  1) 并行结果 == 串行结果（正确性对拍）
  2) scan 前缀语义正确（inclusive / exclusive）
  3) work / step 复杂度断言（O(log n) step、work-efficient 与否）
  4) 边界情形（空、单元素、奇数长度、非 2 的幂、全零、负数、大规模）

运行：  python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest

# 让测试无论从哪运行都能 import 到上级目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel_patterns import (  # noqa: E402
    ComplexityStats,
    ceil_log2,
    histogram_privatized,
    histogram_serial,
    reduce_serial,
    reduce_tree,
    scan_blelloch,
    scan_hillis_steele,
    scan_serial_exclusive,
    scan_serial_inclusive,
    theoretical_steps,
)

RNG = np.random.default_rng(20260704)


# -----------------------------------------------------------------------------
# 1. Reduction
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 100, 1000, 4096])
def test_reduce_tree_matches_serial(n):
    """树形归约结果 == 串行归约（数值严格一致，用 float64 累加）。"""
    x = RNG.standard_normal(n)
    got, stats = reduce_tree(x)
    expected = reduce_serial(x)
    assert np.isclose(got, expected, rtol=1e-9, atol=1e-9)
    # numpy 求和也应一致（作为第三方交叉验证）
    assert np.isclose(got, float(x.astype(np.float64).sum()), rtol=1e-9, atol=1e-9)


def test_reduce_empty_and_single():
    """边界：空数组归约为 0；单元素归约为其本身，且 0 步。"""
    got0, s0 = reduce_tree(np.array([], dtype=np.float64))
    assert got0 == 0.0 and s0.steps == 0
    got1, s1 = reduce_tree(np.array([42.0]))
    assert got1 == 42.0 and s1.steps == 0


@pytest.mark.parametrize("n", [2, 3, 4, 7, 8, 16, 17, 1000, 4096])
def test_reduce_step_is_log_n(n):
    """step 复杂度断言：树形归约步数 == ceil(log2 n) == O(log n)。"""
    x = RNG.standard_normal(n)
    _, stats = reduce_tree(x)
    assert stats.steps == theoretical_steps("reduction", n) == ceil_log2(n)


@pytest.mark.parametrize("n", [2, 4, 8, 16, 1024, 4096])
def test_reduce_work_is_n_minus_1(n):
    """work 复杂度断言：树形归约总加法数 == n-1（与串行同阶 O(n)）。"""
    x = RNG.standard_normal(n)
    _, stats = reduce_tree(x)
    assert stats.work == n - 1


# -----------------------------------------------------------------------------
# 2. Scan —— Hillis-Steele（inclusive）
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 15, 16, 17, 64, 1000])
def test_hillis_steele_matches_serial_inclusive(n):
    """Hillis-Steele 结果 == 串行 inclusive 前缀和（逐元素相等）。"""
    x = RNG.standard_normal(n)
    got, _ = scan_hillis_steele(x)
    expected = scan_serial_inclusive(x)
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_hillis_steele_prefix_semantics():
    """显式核对前缀语义：[1,2,3,4] -> [1,3,6,10]。"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    got, _ = scan_hillis_steele(x)
    assert np.allclose(got, [1, 3, 6, 10])
    # 每个位置都应等于「到自己为止」的和
    assert np.allclose(got, np.cumsum(x))


@pytest.mark.parametrize("n", [2, 4, 8, 16, 17, 64, 1000])
def test_hillis_steele_step_is_log_n(n):
    """step 断言：Hillis-Steele 步数 == ceil(log2 n)（步高效）。"""
    x = RNG.standard_normal(n)
    _, stats = scan_hillis_steele(x)
    assert stats.steps == theoretical_steps("hillis_steele", n)


@pytest.mark.parametrize("n", [8, 16, 64, 256, 1024])
def test_hillis_steele_work_is_superlinear(n):
    """work 断言：Hillis-Steele 总加法 == Σ(n-d) 且 > n（work-inefficient）。

    d = 1,2,4,...；work = Σ(n-d)。应严格大于串行的 n-1。
    """
    x = RNG.standard_normal(n)
    _, stats = scan_hillis_steele(x)
    # 解析期望值：Σ over d=1,2,4,... < n of (n-d)
    expected_work = 0
    d = 1
    while d < n:
        expected_work += (n - d)
        d *= 2
    assert stats.work == expected_work
    assert stats.work > (n - 1)  # 比串行做了更多加法


# -----------------------------------------------------------------------------
# 2. Scan —— Blelloch（exclusive, work-efficient）
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 15, 16, 17, 64, 1000])
def test_blelloch_matches_serial_exclusive(n):
    """Blelloch 结果 == 串行 exclusive 前缀和（含非 2 的幂，内部补零后截断）。"""
    x = RNG.standard_normal(n)
    got, _ = scan_blelloch(x)
    expected = scan_serial_exclusive(x)
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_blelloch_prefix_semantics():
    """显式核对 exclusive 语义：[1,2,3,4] -> [0,1,3,6]。"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    got, _ = scan_blelloch(x)
    assert np.allclose(got, [0, 1, 3, 6])


def test_blelloch_vs_hillis_relationship():
    """两种 scan 的关系：inclusive[i] == exclusive[i] + x[i]。"""
    x = RNG.standard_normal(37)
    inc, _ = scan_hillis_steele(x)
    exc, _ = scan_blelloch(x)
    assert np.allclose(inc, exc + x.astype(np.float64))


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 1024])
def test_blelloch_work_is_linear(n):
    """work 断言：Blelloch 是 work-efficient —— 总加法 O(n)，约 2·(n-1)，
    且严格小于同规模 Hillis-Steele 的 work（对 n>=8）。"""
    x = RNG.standard_normal(n)
    _, stats_b = scan_blelloch(x)
    # up-sweep + down-sweep 对 2 的幂 m=n：各 (m-1) 次加法 => 2(m-1)
    assert stats_b.work == 2 * (n - 1)
    if n >= 8:
        _, stats_h = scan_hillis_steele(x)
        assert stats_b.work < stats_h.work  # work-efficient 优于 Hillis-Steele


@pytest.mark.parametrize("n", [2, 4, 8, 16, 1024])
def test_blelloch_step_is_2log_n(n):
    """step 断言：Blelloch 步数 == 2·log2 n（两趟扫描，仍 O(log n)）。"""
    x = RNG.standard_normal(n)
    _, stats = scan_blelloch(x)
    assert stats.steps == theoretical_steps("blelloch", n)


def test_scan_empty_and_single():
    """边界：空 / 单元素 scan。"""
    hs0, _ = scan_hillis_steele(np.array([], dtype=np.float64))
    bl0, _ = scan_blelloch(np.array([], dtype=np.float64))
    assert hs0.size == 0 and bl0.size == 0
    hs1, _ = scan_hillis_steele(np.array([9.0]))
    bl1, _ = scan_blelloch(np.array([9.0]))
    assert np.allclose(hs1, [9.0])   # inclusive
    assert np.allclose(bl1, [0.0])   # exclusive


# -----------------------------------------------------------------------------
# 3. Histogram
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("num_bins", [4, 8, 16])
@pytest.mark.parametrize("num_workers", [1, 4, 8, 16])
def test_histogram_matches_serial(num_bins, num_workers):
    """私有化直方图 == 串行直方图（逐 bin 相等），且与 numpy 计数一致。"""
    x = RNG.integers(0, num_bins, size=5000)
    got, _ = histogram_privatized(x, num_bins, num_workers=num_workers)
    expected = histogram_serial(x, num_bins)
    assert np.array_equal(got, expected)
    assert np.array_equal(got, np.bincount(x, minlength=num_bins)[:num_bins])
    assert got.sum() == len(x)  # 计数守恒：所有元素都被计入


def test_histogram_hot_bin_contention():
    """热点 bin：所有元素都落进同一个 bin，私有化仍正确（无原子丢失）。"""
    x = np.zeros(1234, dtype=np.int64)  # 全部进 bin 0
    got, _ = histogram_privatized(x, num_bins=4, num_workers=8)
    assert got[0] == 1234 and got[1:].sum() == 0


def test_histogram_merge_step_is_log_workers():
    """step 断言：合并阶段步数 == ceil(log2 num_workers)（私有化后 O(log workers)）。"""
    x = RNG.integers(0, 8, size=1000)
    for w in [2, 4, 8, 16]:
        _, stats = histogram_privatized(x, num_bins=8, num_workers=w)
        assert stats.steps == ceil_log2(w)


def test_histogram_empty():
    """边界：空输入 -> 全零直方图。"""
    got, _ = histogram_privatized(np.array([], dtype=np.int64), num_bins=8, num_workers=4)
    assert got.sum() == 0 and got.shape == (8,)


# -----------------------------------------------------------------------------
# 4. 通用不变量 / 复杂度探针自身
# -----------------------------------------------------------------------------
def test_ceil_log2_values():
    """ceil_log2 边界与典型值。"""
    assert ceil_log2(1) == 0
    assert ceil_log2(2) == 1
    assert ceil_log2(3) == 2
    assert ceil_log2(4) == 2
    assert ceil_log2(1024) == 10
    assert ceil_log2(1025) == 11


def test_complexity_stats_accumulates():
    """ComplexityStats.add_step 正确累加 steps/work，并记录逐步工作量。"""
    s = ComplexityStats(n=8)
    s.add_step(4)
    s.add_step(2)
    s.add_step(1)
    assert s.steps == 3 and s.work == 7
    assert s.work_per_step == [4, 2, 1]


def test_step_grows_logarithmically_not_linearly():
    """核心命题：n 翻 1024 倍，step 只增加约 log2(1024)=10，而非线性增长。

    这是整个实验的「题眼」：并行深度是对数级的。
    """
    _, s_small = reduce_tree(RNG.standard_normal(16))
    _, s_big = reduce_tree(RNG.standard_normal(16 * 1024))
    # 规模 ×1024，step 只 +10（log2 1024），远小于线性会带来的 ×1024
    assert s_big.steps - s_small.steps == 10
    assert s_big.steps < 30  # 绝对值仍然很小
