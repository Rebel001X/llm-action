# -*- coding: utf-8 -*-
"""
test_roofline.py —— Roofline 项目的单元测试
================================================================================
覆盖四类：
  1. 公式正确性：算术强度 AI、GEMM/逐元素 FLOP 与 Byte、屋顶线 min 语义。
  2. 屋脊点 / bound 判定 的边界行为。
  3. 加速比估算 的方向性（compute-bound 用算力比、memory-bound 用带宽比）。
  4. 本机实测 的健全性（算力/带宽为正、且落在合理物理量级内）。

全部离线、无 GPU、无网络。运行：python -m pytest -q
"""

import os
import sys

import numpy as np
import pytest

# 让 tests/ 能 import 上一级目录的 roofline.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roofline as rl  # noqa: E402


# =============================================================================
# 1. 算术强度 AI
# =============================================================================

def test_arithmetic_intensity_basic():
    # 10 FLOP / 5 Byte = 2 FLOP/Byte
    assert rl.arithmetic_intensity(10.0, 5.0) == pytest.approx(2.0)


def test_arithmetic_intensity_zero_bytes_returns_zero():
    # 没有字节搬运则强度定义为 0（避免除零）
    assert rl.arithmetic_intensity(10.0, 0.0) == 0.0
    assert rl.arithmetic_intensity(10.0, -1.0) == 0.0


def test_arithmetic_intensity_is_flop_over_byte():
    # 定义式验证：AI == flops / bytes
    flops, byts = 12345.0, 678.0
    assert rl.arithmetic_intensity(flops, byts) == pytest.approx(flops / byts)


# =============================================================================
# 2. GEMM 与 逐元素 的 FLOP / Byte 公式
# =============================================================================

def test_gemm_flops_formula():
    # 2*m*n*k
    assert rl.gemm_flops(2, 3, 4) == pytest.approx(2 * 2 * 3 * 4)
    assert rl.gemm_flops(1000, 1000, 1000) == pytest.approx(2e9)


def test_gemm_bytes_formula_fp32():
    # (m*k + k*n + m*n) * 4
    m, n, k = 2, 3, 4
    expected = (m * k + k * n + m * n) * 4
    assert rl.gemm_bytes(m, n, k, dtype_bytes=4) == pytest.approx(expected)


def test_gemm_bytes_fp16_is_half_of_fp32():
    m, n, k = 8, 8, 8
    b32 = rl.gemm_bytes(m, n, k, dtype_bytes=4)
    b16 = rl.gemm_bytes(m, n, k, dtype_bytes=2)
    assert b16 == pytest.approx(b32 / 2)


def test_gemm_ai_grows_linearly_with_n():
    # 方阵 GEMM 的 AI = 2n^3 / (3 n^2 * 4) = n/6，应随 n 线性增长
    for n in (256, 512, 1024, 2048):
        ai = rl.arithmetic_intensity(rl.gemm_flops(n, n, n),
                                    rl.gemm_bytes(n, n, n, 4))
        assert ai == pytest.approx(n / 6.0, rel=1e-6)


def test_elementwise_flops_and_bytes():
    n = 1000
    assert rl.elementwise_flops(n, 1.0) == pytest.approx(1000)
    assert rl.elementwise_flops(n, 2.0) == pytest.approx(2000)
    # ReLU: 读1写1 → 2*n*4
    assert rl.elementwise_bytes(n, 1, 1, 4) == pytest.approx(2 * n * 4)
    # z=x+y: 读2写1 → 3*n*4
    assert rl.elementwise_bytes(n, 2, 1, 4) == pytest.approx(3 * n * 4)


def test_elementwise_is_memory_bound_low_ai():
    # 逐元素算子 AI 应该很小（远小于 1 通常）
    n = 1 << 20
    ai = rl.arithmetic_intensity(rl.elementwise_flops(n, 1.0),
                                rl.elementwise_bytes(n, 1, 1, 4))
    assert ai < 1.0  # ReLU 的 AI = 1/8 = 0.125


# =============================================================================
# 3. Roofline 屋顶线 min 语义
# =============================================================================

def test_roofline_takes_min():
    # 屋顶线 = min(算力, 带宽*AI)
    peak_flops, peak_bw = 100.0, 10.0
    # AI 小：带宽项占优（10*1=10 < 100）
    assert rl.roofline_perf(1.0, peak_flops, peak_bw) == pytest.approx(10.0)
    # AI 大：算力项占优（10*1000=10000 > 100 → 封顶 100）
    assert rl.roofline_perf(1000.0, peak_flops, peak_bw) == pytest.approx(100.0)


def test_roofline_equals_explicit_min():
    # 显式对照 min 公式
    for ai in (0.01, 0.1, 1, 5, 10, 50, 500):
        pf, bw = 1e12, 1e11
        assert rl.roofline_perf(ai, pf, bw) == pytest.approx(min(pf, bw * ai))


def test_roofline_memory_bound_is_linear_in_ai():
    # 在带宽斜坡段，性能应与 AI 成正比
    pf, bw = 1e12, 1e11
    ai1, ai2 = 0.5, 1.5  # 都远低于屋脊点 pf/bw=10
    p1 = rl.roofline_perf(ai1, pf, bw)
    p2 = rl.roofline_perf(ai2, pf, bw)
    assert p2 / p1 == pytest.approx(ai2 / ai1)


def test_roofline_compute_bound_is_flat():
    # 在算力段，AI 再增大性能不变
    pf, bw = 1e12, 1e11
    p1 = rl.roofline_perf(100.0, pf, bw)
    p2 = rl.roofline_perf(1000.0, pf, bw)
    assert p1 == pytest.approx(pf)
    assert p2 == pytest.approx(pf)


# =============================================================================
# 4. 屋脊点 与 bound 判定
# =============================================================================

def test_ridge_point_formula():
    # AI* = peak_flops / peak_bw
    assert rl.ridge_point(1000.0, 10.0) == pytest.approx(100.0)


def test_ridge_point_zero_bw_is_inf():
    assert rl.ridge_point(1000.0, 0.0) == float("inf")


def test_ridge_point_is_crossover():
    # 在屋脊点处，两条屋顶应相等
    pf, bw = 500.0, 25.0
    ai_star = rl.ridge_point(pf, bw)
    assert bw * ai_star == pytest.approx(pf)


def test_is_compute_bound_boundary():
    pf, bw = 1000.0, 10.0  # 屋脊点 = 100
    assert rl.is_compute_bound(150.0, pf, bw) is True   # 右侧 → compute
    assert rl.is_compute_bound(50.0, pf, bw) is False   # 左侧 → memory
    assert rl.is_compute_bound(100.0, pf, bw) is True   # 恰在屋脊（>=）


# =============================================================================
# 5. 加速比估算：方向性
# =============================================================================

def test_speedup_compute_bound_uses_flops_ratio():
    # 高 AI（compute-bound）→ 加速比 ≈ 算力比
    cpu_pf, cpu_bw = 1e12, 1e11    # ridge=10
    gpu_pf, gpu_bw = 2e13, 1e12    # ridge=20；算力是 CPU 的 20 倍
    est = rl.estimate_speedup(1000.0, cpu_pf, cpu_bw, gpu_pf, gpu_bw, "bigGEMM")
    assert est.cpu_bound == "compute"
    assert est.gpu_bound == "compute"
    # 两边都 compute-bound → 加速比 = 算力比 = 20
    assert est.speedup == pytest.approx(gpu_pf / cpu_pf)


def test_speedup_memory_bound_uses_bw_ratio():
    # 低 AI（memory-bound）→ 加速比 ≈ 带宽比
    cpu_pf, cpu_bw = 1e12, 1e11
    gpu_pf, gpu_bw = 2e13, 5e11    # 带宽是 CPU 的 5 倍
    est = rl.estimate_speedup(0.1, cpu_pf, cpu_bw, gpu_pf, gpu_bw, "relu")
    assert est.cpu_bound == "memory"
    assert est.gpu_bound == "memory"
    # 两边都 memory-bound → 加速比 = 带宽比 = 5
    assert est.speedup == pytest.approx(gpu_bw / cpu_bw)


def test_speedup_positive_and_finite():
    est = rl.estimate_speedup(1.0, 1e12, 1e11, 2e13, 1e12, "x")
    assert est.speedup > 0
    assert np.isfinite(est.speedup)


def test_compute_bound_speedup_larger_than_memory_bound():
    # 同一对硬件：算力比 > 带宽比时，compute-bound 算子的加速应更大
    cpu_pf, cpu_bw = 1e12, 1e11
    gpu_pf, gpu_bw = 4e13, 2e11   # 算力比40 vs 带宽比2
    hi = rl.estimate_speedup(1e4, cpu_pf, cpu_bw, gpu_pf, gpu_bw)  # compute
    lo = rl.estimate_speedup(0.05, cpu_pf, cpu_bw, gpu_pf, gpu_bw)  # memory
    assert hi.speedup > lo.speedup


# =============================================================================
# 6. GPU 规格表 健全性
# =============================================================================

def test_gpu_specs_all_positive():
    assert len(rl.GPU_SPECS) >= 3
    for name, spec in rl.GPU_SPECS.items():
        assert spec["fp32_tflops"] > 0, name
        assert spec["bandwidth_gbps"] > 0, name


def test_gpu_ridge_point_high_for_h100():
    # H100 TF32 屋脊点应显著高于其 fp32 屋脊点（Tensor Core 拉高算力→更难喂饱）
    h = rl.GPU_SPECS["H100-SXM"]
    ridge_fp32 = rl.ridge_point(h["fp32_tflops"] * 1e12, h["bandwidth_gbps"] * 1e9)
    ridge_tf32 = rl.ridge_point(h["tf32_tflops"] * 1e12, h["bandwidth_gbps"] * 1e9)
    assert ridge_tf32 > ridge_fp32


# =============================================================================
# 7. 典型算子清单
# =============================================================================

def test_typical_ops_span_memory_to_compute():
    pts = rl.typical_dl_ops()
    ais = [p.ai for p in pts]
    # 应该同时存在很小的 AI（逐元素）和很大的 AI（大 GEMM）
    assert min(ais) < 1.0
    assert max(ais) > 100.0


def test_typical_ops_have_positive_flops_bytes():
    for p in rl.typical_dl_ops():
        assert p.flops > 0, p.name
        assert p.bytes_moved > 0, p.name
        assert p.ai > 0, p.name


# =============================================================================
# 8. 本机实测：健全性（为正、物理量级合理）
# =============================================================================

def test_measure_gemm_flops_positive_and_sane():
    # 用较小尺寸让测试快；仍要 compute-bound
    res = rl.measure_gemm_flops(n=512, repeats=3)
    assert res.seconds > 0
    assert res.achieved_flops > 0
    # 现代 CPU 上 BLAS 单精度算力量级：>1 GFLOP/s，且 < 100 TFLOP/s（上限极宽松）
    assert 1e9 < res.achieved_flops < 1e14
    # FLOP 公式应与 gemm_flops 一致
    assert res.flops == pytest.approx(rl.gemm_flops(512, 512, 512))


def test_measure_bandwidth_positive_and_sane():
    # 用较小 n 让测试快，但仍需 > L2 以体现主存
    res = rl.measure_memory_bandwidth(n=1 << 22, repeats=3)
    assert res.seconds > 0
    assert res.achieved_bw > 0
    # 主存带宽量级：>0.1 GB/s，< 10 TB/s（极宽松上限）
    assert 1e8 < res.achieved_bw < 1e13
    assert res.bytes_moved == pytest.approx(3 * (1 << 22) * 4)


def test_profile_machine_gives_positive_roofs():
    prof = rl.profile_machine(gemm_n=512, mem_n=1 << 22, repeats=2)
    assert prof.peak_flops > 0
    assert prof.peak_bw > 0
    assert prof.ridge > 0
    # 屋脊点应等于 峰值算力/峰值带宽
    assert prof.ridge == pytest.approx(prof.peak_flops / prof.peak_bw)


def test_gemm_is_compute_bound_bandwidth_is_memory_bound():
    # 方法论自洽性：测算力用的大 GEMM 应 compute-bound；测带宽用的 Add 应 memory-bound
    prof = rl.profile_machine(gemm_n=1024, mem_n=1 << 22, repeats=2)
    gemm_ai = prof.gemm.arithmetic_intensity
    mem_ai = prof.mem.arithmetic_intensity
    # 大 GEMM 的 AI 应高于屋脊点（compute-bound）
    assert gemm_ai > prof.ridge
    # 逐元素 Add 的 AI 应低于屋脊点（memory-bound）
    assert mem_ai < prof.ridge


def test_bench_result_unit_properties():
    res = rl.measure_gemm_flops(n=256, repeats=2)
    # gflops == achieved_flops/1e9
    assert res.gflops == pytest.approx(res.achieved_flops / 1e9)
    assert res.gbps == pytest.approx(res.achieved_bw / 1e9)
