# -*- coding: utf-8 -*-
"""
test_tiled_matmul.py — 分块 matmul 与访存模型的单元测试
=======================================================

四类断言(与 README 里承诺的一一对应):
  A. 数值正确性:分块结果 == numpy 直接 matmul(各种形状/tile,含边界)
  B. 单调性:tile 越大 → HBM 访问越少(直到 shared memory 上限)
  C. 公式精确性:复用率、算术强度、FLOPs 与解析式吻合
  D. 健壮性:非法参数抛错、边界(tile=1、tile>=n、非整除)行为正确

运行:  python -m pytest -q
"""

import numpy as np
import pytest

import sys
import os

# 让 tests/ 能 import 上一级目录的 tiled_matmul.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiled_matmul import (  # noqa: E402
    matmul_naive,
    matmul_tiled,
    analyze,
    analyze_square,
    max_tile_for_smem,
    sweep_tiles,
    MemoryModel,
)


# ---------------------------------------------------------------------
# A. 数值正确性:分块结果必须逐元素等于 numpy 的 A @ B
# ---------------------------------------------------------------------

@pytest.mark.parametrize("M,K,N", [
    (8, 8, 8),      # 方阵、整除
    (16, 16, 16),
    (10, 7, 13),    # 全部互质 → 强制走边界截断
    (1, 5, 1),      # 退化成向量点积
    (5, 1, 5),      # K=1
    (32, 24, 16),
])
@pytest.mark.parametrize("tile", [1, 2, 3, 4, 8, 100])
def test_tiled_equals_numpy(M, K, N, tile):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K))
    B = rng.standard_normal((K, N))
    ref = A @ B
    out = matmul_tiled(A, B, tile)
    assert out.shape == ref.shape
    # 浮点累加顺序不同,用 allclose 而非逐位相等
    assert np.allclose(out, ref, rtol=1e-10, atol=1e-10)


def test_tiled_equals_naive_loop():
    """分块实现也要等于「可读的三重循环基线」。"""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((12, 9))
    B = rng.standard_normal((9, 7))
    ref = matmul_naive(A, B)
    for tile in (1, 2, 5, 9, 20):
        out = matmul_tiled(A, B, tile)
        assert np.allclose(out, ref, rtol=1e-10, atol=1e-10)


def test_naive_matches_numpy():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 11))
    B = rng.standard_normal((11, 4))
    assert np.allclose(matmul_naive(A, B), A @ B, rtol=1e-10, atol=1e-10)


def test_tile_larger_than_matrix_is_single_block():
    """tile 远大于矩阵 → 整个矩阵就是一个块,结果仍正确。"""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 4))
    B = rng.standard_normal((4, 4))
    assert np.allclose(matmul_tiled(A, B, 999), A @ B)


# ---------------------------------------------------------------------
# B. 单调性:tile 越大,HBM 访问越少(严格单调下降,直到无法再分)
# ---------------------------------------------------------------------

def test_hbm_read_strictly_decreases_with_tile():
    """在 tile 整除 n 的一串取值上,HBM 读字节严格随 tile 增大而减少。"""
    n = 256
    tiles = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    reads = [analyze_square(n, t).hbm_read_bytes for t in tiles]
    for a, b in zip(reads, reads[1:]):
        assert b < a, f"HBM 读应随 tile 增大而减少,但 {b} 不小于 {a}"


def test_hbm_total_decreases_with_tile():
    """读+写总字节也应随 tile 增大而下降(写量固定,读量下降)。"""
    n = 128
    tiles = [1, 2, 4, 8, 16, 32, 64, 128]
    totals = [analyze_square(n, t).hbm_bytes_total for t in tiles]
    for a, b in zip(totals, totals[1:]):
        assert b < a


def test_arithmetic_intensity_increases_with_tile():
    """算术强度随 tile 增大而单调上升 —— 从访存受限走向计算受限。"""
    n = 256
    tiles = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ai = [analyze_square(n, t).arithmetic_intensity for t in tiles]
    for a, b in zip(ai, ai[1:]):
        assert b > a


def test_tiling_beats_naive():
    """任何 tile>=2 的分块,HBM 读都严格少于朴素方案。"""
    n = 512
    naive = analyze_square(n, 1, scheme="naive")
    for t in (2, 16, 32, 64):
        tiled = analyze_square(n, t, scheme="tiled")
        assert tiled.hbm_read_bytes < naive.hbm_read_bytes
        assert tiled.arithmetic_intensity > naive.arithmetic_intensity


def test_smem_upper_bound_caps_useful_tile():
    """
    分块的收益到 shared memory 上限为止:
    超过 max_tile_for_smem 的 tile 在真实 kernel 里放不进片上。
    这里验证 max_tile_for_smem 随容量单调、随字节数反向单调。
    """
    t48 = max_tile_for_smem(48 * 1024, dtype_bytes=4)
    t96 = max_tile_for_smem(96 * 1024, dtype_bytes=4)
    assert t96 > t48                       # 容量越大,可放 tile 越大
    t_fp32 = max_tile_for_smem(48 * 1024, dtype_bytes=4)
    t_fp64 = max_tile_for_smem(48 * 1024, dtype_bytes=8)
    assert t_fp64 < t_fp32                 # 元素越大,能放的 tile 越小
    # 验证放进去后确实不超容量(2 个 T×T 子块)
    smem = 48 * 1024
    t = max_tile_for_smem(smem, dtype_bytes=4, num_tiles=2)
    assert 2 * t * t * 4 <= smem
    assert 2 * (t + 1) * (t + 1) * 4 > smem  # 再大一格就超了


# ---------------------------------------------------------------------
# C. 公式精确性:复用率 / 算术强度 / FLOPs / HBM 读量的闭式解
# ---------------------------------------------------------------------

def test_reuse_factor_equals_tile_when_divisible():
    """方阵、tile 整除 n 时,复用率精确等于 tile 边长 T。"""
    n = 240  # 能被 1,2,3,4,5,6,8,10,12,16,... 整除
    for t in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24):
        assert n % t == 0
        m = analyze_square(n, t, scheme="tiled")
        assert m.reuse_factor == pytest.approx(float(t), rel=1e-12)


def test_naive_read_elems_is_2nnn():
    """朴素方案读入元素数 == 2*M*N*K(方阵即 2 n^3)。"""
    M, N, K = 7, 5, 11
    m = analyze(M, N, K, tile=1, scheme="naive")
    assert m.hbm_read_elems == pytest.approx(2.0 * M * N * K)


def test_tiled_read_elems_closed_form_divisible():
    """整除情形:方阵分块读入元素数 == 2 n^3 / T。"""
    n = 128
    for t in (1, 2, 4, 8, 16, 32, 64, 128):
        m = analyze_square(n, t, scheme="tiled")
        expected = 2.0 * n * n * n / t
        assert m.hbm_read_elems == pytest.approx(expected, rel=1e-12)


def test_flops_formula():
    """FLOPs == 2*M*N*K,且与方案(naive/tiled)无关。"""
    M, N, K = 9, 13, 6
    a = analyze(M, N, K, tile=1, scheme="naive")
    b = analyze(M, N, K, tile=4, scheme="tiled")
    assert a.flops == pytest.approx(2.0 * M * N * K)
    assert b.flops == pytest.approx(2.0 * M * N * K)
    assert a.flops == b.flops


def test_arithmetic_intensity_definition():
    """AI 必须 == FLOPs / HBM总字节,逐字段对齐。"""
    m = analyze_square(256, 32, scheme="tiled", dtype_bytes=4)
    assert m.arithmetic_intensity == pytest.approx(
        m.flops / m.hbm_bytes_total, rel=1e-12
    )


def test_dtype_bytes_scales_bytes_not_ai_much():
    """字节数翻倍 → HBM 字节翻倍、AI 减半(FLOPs 不变)。"""
    a = analyze_square(256, 16, dtype_bytes=4)
    b = analyze_square(256, 16, dtype_bytes=8)
    assert b.hbm_bytes_total == pytest.approx(2.0 * a.hbm_bytes_total)
    assert b.arithmetic_intensity == pytest.approx(
        0.5 * a.arithmetic_intensity, rel=1e-9
    )
    # 复用率是纯几何量,与 dtype 无关
    assert a.reuse_factor == pytest.approx(b.reuse_factor)


def test_write_elems_is_mn():
    """C 每个元素只写回一次 → 写量恒为 M*N。"""
    m = analyze(7, 5, 11, tile=3, scheme="tiled")
    assert m.hbm_write_elems == pytest.approx(7 * 5)


# ---------------------------------------------------------------------
# D. 边界与非整除:ceil 逻辑正确、非法参数抛错
# ---------------------------------------------------------------------

def test_non_divisible_tile_uses_ceil():
    """
    非整除时读量由 ceil 决定:
    n=10, T=3 → ceil(10/3)=4 个块列/块行。
    read_A = n^2 * ceil(n/T) = 100*4 = 400;read_B 同;read_elems=800。
    """
    m = analyze_square(10, 3, scheme="tiled")
    assert m.hbm_read_elems == pytest.approx(800.0)
    # 与「整除近似式 2 n^3 / T = 2000/3 ≈ 666.7」不同,ceil 会偏大 —— 合理
    assert m.hbm_read_elems > 2.0 * 10 ** 3 / 3


def test_tile_one_tiled_equals_naive_read():
    """tile=1 的分块方案,读量应退化到与朴素完全一致(2 n^3)。"""
    n = 32
    tiled1 = analyze_square(n, 1, scheme="tiled")
    naive = analyze_square(n, 1, scheme="naive")
    assert tiled1.hbm_read_elems == pytest.approx(naive.hbm_read_elems)
    assert tiled1.reuse_factor == pytest.approx(1.0)


def test_tile_equals_n_single_block():
    """tile==n:整个矩阵一个块,复用率 == n。"""
    n = 64
    m = analyze_square(n, n, scheme="tiled")
    assert m.reuse_factor == pytest.approx(float(n))


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_invalid_tile_raises(bad):
    A = np.ones((4, 4))
    with pytest.raises(ValueError):
        matmul_tiled(A, A, bad)
    with pytest.raises(ValueError):
        analyze_square(16, bad, scheme="tiled")


def test_invalid_scheme_raises():
    with pytest.raises(ValueError):
        analyze_square(16, 4, scheme="magic")


def test_inner_dim_mismatch_raises():
    A = np.ones((3, 4))
    B = np.ones((5, 2))  # 4 != 5
    with pytest.raises(ValueError):
        matmul_tiled(A, B, 2)
    with pytest.raises(ValueError):
        matmul_naive(A, B)


def test_non_2d_raises():
    with pytest.raises(ValueError):
        matmul_tiled(np.ones((2, 2, 2)), np.ones((2, 2)), 1)


def test_invalid_dims_in_analyze():
    with pytest.raises(ValueError):
        analyze(0, 4, 4, tile=2)
    with pytest.raises(ValueError):
        analyze(4, 4, -3, tile=2)


# ---------------------------------------------------------------------
# E. sweep 与数据类的一致性
# ---------------------------------------------------------------------

def test_sweep_tiles_consistent():
    n = 256
    tiles = [8, 16, 32, 64]
    sw = sweep_tiles(n, tiles)
    assert len(sw["tiles"]) == len(tiles)
    # sweep 里的读量应与逐个 analyze 一致
    for i, t in enumerate(tiles):
        m = analyze_square(n, t)
        assert sw["hbm_read_bytes"][i] == pytest.approx(m.hbm_read_bytes)
        assert sw["arithmetic_intensity"][i] == pytest.approx(
            m.arithmetic_intensity)
    # naive 基线读量应大于任何 tiled 读量
    assert sw["naive_read_bytes"] > sw["hbm_read_bytes"].max()


def test_memory_model_is_frozen_dataclass():
    m = analyze_square(64, 8)
    assert isinstance(m, MemoryModel)
    with pytest.raises(Exception):
        m.tile = 999  # frozen → 不可变


def test_max_tile_for_smem_typical_values():
    """48KB / fp32 / 2 子块 → T≈78;粗略校验数量级正确。"""
    t = max_tile_for_smem(48 * 1024, dtype_bytes=4, num_tiles=2)
    assert 70 <= t <= 85
    # 单子块(num_tiles=1)时能放更大的 tile
    assert max_tile_for_smem(48 * 1024, num_tiles=1) > t
