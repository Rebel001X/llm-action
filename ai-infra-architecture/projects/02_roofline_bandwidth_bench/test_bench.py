"""
test_bench.py —— 验证实测带宽/算力在合理范围,且算术强度/roofline 公式正确。
运行:python -m pytest -q
"""
import math
import numpy as np
from bench import (
    measure_bandwidth, measure_gflops, arithmetic_intensity_triad,
    arithmetic_intensity_matmul, roofline_perf, BYTES,
)


def test_bandwidth_positive_and_sane():
    bw = measure_bandwidth(n=2_000_000)
    for name, (gbps, nbytes) in bw.items():
        assert gbps > 0.5, f"{name} 带宽异常低 {gbps}"
        assert gbps < 5000, f"{name} 带宽异常高 {gbps}"     # DRAM/缓存量级上限
        assert nbytes > 0


def test_gflops_positive_and_sane():
    g, flop = measure_gflops(n=400)
    assert g > 0.1 and g < 100000
    assert flop == 2 * 400 ** 3


def test_arithmetic_intensity_values():
    assert math.isclose(arithmetic_intensity_triad(), 2.0 / 24)          # 1/12
    # 矩阵乘算术强度随 n 线性增长
    assert arithmetic_intensity_matmul(1200) > arithmetic_intensity_matmul(120)
    assert math.isclose(arithmetic_intensity_matmul(120), 120 / 12)


def test_roofline_is_min_of_two_ceilings():
    # 低算术强度 → 受带宽限制;高算术强度 → 受算力限制
    assert math.isclose(roofline_perf(0.1, 1000, 50), 50 * 0.1)          # memory-bound
    assert math.isclose(roofline_perf(1000, 1000, 50), 1000)            # compute-bound
    ridge = 1000 / 50                                                    # 脊点
    assert math.isclose(roofline_perf(ridge, 1000, 50), 1000)


def test_matmul_is_compute_bound_relative_to_triad():
    # 矩阵乘的算术强度应远高于 triad → 更靠近算力屋顶
    assert arithmetic_intensity_matmul(1200) > arithmetic_intensity_triad() * 100
