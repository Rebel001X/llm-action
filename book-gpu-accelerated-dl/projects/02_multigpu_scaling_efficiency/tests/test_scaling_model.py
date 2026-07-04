# -*- coding: utf-8 -*-
"""
test_scaling_model.py — 多 GPU 扩展效率模型的单元测试

覆盖需求里的四类硬性性质:
  1) 通信为 0 且 s=0 时 → 线性加速(S(N)==N, E(N)==1)
  2) 通信增大 → 效率下降(单调性)
  3) 效率 = 加速比 / N ∈ (0, 1](恒成立)
  4) 边界与异常(N<1、s 越界、带宽<=0、非法拓扑、单卡特判)

运行:  python -m pytest -q
"""

import math
import os
import sys

import numpy as np
import pytest

# 让测试无论从哪个目录运行都能 import 到上一层的 scaling_model.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaling_model import (  # noqa: E402
    CommModel,
    amdahl_efficiency,
    amdahl_speedup,
    communication_time,
    efficiency_with_comm,
    half_efficiency_n,
    max_speedup_amdahl,
    scaling_curve,
    speedup_with_comm,
)


# =============================================================================
# 1) 无通信 + 完全并行 → 线性加速
# =============================================================================

@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 128])
def test_no_comm_zero_serial_is_linear(n):
    """s=0 且无通信:S(N)==N,E(N)==1(完美线性加速)。"""
    assert speedup_with_comm(n, serial_fraction=0.0, comm=None) == pytest.approx(n)
    assert efficiency_with_comm(n, serial_fraction=0.0, comm=None) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64])
def test_pure_amdahl_zero_serial_is_linear(n):
    """纯 Amdahl 接口在 s=0 时也应给出线性加速。"""
    assert amdahl_speedup(n, 0.0) == pytest.approx(n)
    assert amdahl_efficiency(n, 0.0) == pytest.approx(1.0)


def test_comm_model_zero_message_is_linear():
    """通信数据量为 0(message_bytes=0)等价于无通信 → 线性。"""
    comm = CommModel(message_bytes=0.0, bandwidth=1.0e9, scaling="ring")
    for n in [1, 2, 8, 32]:
        assert speedup_with_comm(n, 0.0, comm) == pytest.approx(n)
        assert efficiency_with_comm(n, 0.0, comm) == pytest.approx(1.0)


# =============================================================================
# 2) 单卡特判:N=1 恒等
# =============================================================================

@pytest.mark.parametrize("s", [0.0, 0.05, 0.3, 1.0])
@pytest.mark.parametrize("scaling", ["linear", "log", "ring"])
def test_single_gpu_speedup_is_one(s, scaling):
    """无论 s、拓扑如何,单卡加速比恒为 1、效率恒为 1(通信在 N=1 时为 0)。"""
    comm = CommModel(message_bytes=1e9, bandwidth=1e9, scaling=scaling)
    assert speedup_with_comm(1, s, comm) == pytest.approx(1.0)
    assert efficiency_with_comm(1, s, comm) == pytest.approx(1.0)


@pytest.mark.parametrize("scaling", ["linear", "log", "ring"])
def test_comm_time_zero_at_single_gpu(scaling):
    """单卡不需要跨卡同步 → 通信时间为 0。"""
    comm = CommModel(message_bytes=1e9, bandwidth=1e9, scaling=scaling)
    assert comm.comm_time(1) == 0.0
    assert comm.growth(1) == 0.0


# =============================================================================
# 3) 通信增大 → 效率下降(核心论点:通信占比限制强扩展)
# =============================================================================

@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_more_comm_lowers_efficiency(n):
    """固定 s、固定拓扑,增大 message_bytes(通信占比更大)→ 效率单调下降。"""
    s = 0.05
    low = CommModel(message_bytes=1e7, bandwidth=1e10, scaling="ring")
    mid = CommModel(message_bytes=1e8, bandwidth=1e10, scaling="ring")
    high = CommModel(message_bytes=1e9, bandwidth=1e10, scaling="ring")
    e_low = efficiency_with_comm(n, s, low)
    e_mid = efficiency_with_comm(n, s, mid)
    e_high = efficiency_with_comm(n, s, high)
    assert e_low > e_mid > e_high


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_comm_lowers_speedup_vs_pure_amdahl(n):
    """有通信的加速比一定 <= 纯 Amdahl(通信只会拖慢,不会加速)。"""
    s = 0.05
    comm = CommModel(message_bytes=2e8, bandwidth=1e10, scaling="ring")
    assert speedup_with_comm(n, s, comm) < amdahl_speedup(n, s)


def test_lower_bandwidth_lowers_efficiency():
    """带宽越低 → 通信越慢 → 效率越低。"""
    s = 0.05
    n = 16
    fast = CommModel(message_bytes=1e8, bandwidth=1e11, scaling="ring")
    slow = CommModel(message_bytes=1e8, bandwidth=1e9, scaling="ring")
    assert efficiency_with_comm(n, s, fast) > efficiency_with_comm(n, s, slow)


def test_larger_serial_lowers_efficiency():
    """串行占比越大 → 效率越低(Amdahl 本身的性质)。"""
    n = 16
    assert amdahl_efficiency(n, 0.01) > amdahl_efficiency(n, 0.1) > amdahl_efficiency(n, 0.5)


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
def test_topology_ordering_ring_best(n):
    """相同数据量下,通信增长:linear(最差) > log > ring(最好)
    → 加速比:ring >= log >= linear。"""
    s = 0.02
    args = dict(message_bytes=5e8, bandwidth=1e10)
    s_lin = speedup_with_comm(n, s, CommModel(scaling="linear", **args))
    s_log = speedup_with_comm(n, s, CommModel(scaling="log", **args))
    s_ring = speedup_with_comm(n, s, CommModel(scaling="ring", **args))
    assert s_ring >= s_log >= s_lin


# =============================================================================
# 4) 效率恒等式 E = S / N,且 E ∈ (0, 1]
# =============================================================================

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 32, 64, 128])
@pytest.mark.parametrize("s", [0.0, 0.02, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("scaling", ["linear", "log", "ring"])
def test_efficiency_equals_speedup_over_n(n, s, scaling):
    """恒等式:E(N) == S(N) / N。"""
    comm = CommModel(message_bytes=3e8, bandwidth=1e10, scaling=scaling)
    s_val = speedup_with_comm(n, s, comm)
    e_val = efficiency_with_comm(n, s, comm)
    assert e_val == pytest.approx(s_val / n)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("s", [0.0, 0.01, 0.05, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("scaling", ["linear", "log", "ring"])
def test_efficiency_in_zero_one(n, s, scaling):
    """效率恒落在 (0, 1]:上界 1 是因为通信/串行只会拖慢;下界 >0 因分母有限。"""
    comm = CommModel(message_bytes=1e9, bandwidth=1e9, scaling=scaling)
    e = efficiency_with_comm(n, s, comm)
    assert 0.0 < e <= 1.0 + 1e-12  # 容许极小浮点误差


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_efficiency_upper_bound_pure_amdahl(n):
    """纯 Amdahl 效率也必须落在 (0, 1]。"""
    for s in [0.0, 0.05, 0.3, 1.0]:
        e = amdahl_efficiency(n, s)
        assert 0.0 < e <= 1.0 + 1e-12


def test_speedup_positive_even_when_comm_dominates():
    """即使通信极大导致加卡变慢,加速比仍应为正数(不会崩成 0/负)。"""
    huge = CommModel(message_bytes=1e15, bandwidth=1.0, scaling="linear")
    for n in [2, 8, 64]:
        s_val = speedup_with_comm(n, 0.1, huge)
        assert s_val > 0.0


# =============================================================================
# 5) Amdahl 天花板与单调性
# =============================================================================

def test_amdahl_ceiling():
    """N→∞ 时 S→1/s。取很大的 N,应逼近 1/s 但不超过。"""
    s = 0.05
    ceil = max_speedup_amdahl(s)
    assert ceil == pytest.approx(20.0)
    big = amdahl_speedup(10_000_000, s)
    assert big < ceil
    assert big == pytest.approx(ceil, rel=1e-3)


def test_amdahl_ceiling_zero_serial_is_inf():
    """s=0 时天花板为 +inf(理论可无限加速)。"""
    assert math.isinf(max_speedup_amdahl(0.0))


def test_amdahl_speedup_monotonic_in_n():
    """纯 Amdahl:卡数越多,加速比单调不减。"""
    s = 0.1
    prev = 0.0
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        cur = amdahl_speedup(n, s)
        assert cur >= prev
        prev = cur


def test_fully_serial_no_speedup():
    """s=1(完全串行):无论多少卡,加速比恒为 1,效率 = 1/N。"""
    for n in [1, 2, 8, 64]:
        assert amdahl_speedup(n, 1.0) == pytest.approx(1.0)
        assert amdahl_efficiency(n, 1.0) == pytest.approx(1.0 / n)


# =============================================================================
# 6) 通信时间公式:comm ∝ 数据量,反比于带宽
# =============================================================================

def test_comm_time_proportional_to_bytes():
    """通信时间 ∝ message_bytes(带宽/拓扑/卡数固定,翻倍数据量→翻倍通信)。"""
    t1 = communication_time(8, message_bytes=1e8, bandwidth=1e10, scaling="ring")
    t2 = communication_time(8, message_bytes=2e8, bandwidth=1e10, scaling="ring")
    assert t2 == pytest.approx(2.0 * t1)


def test_comm_time_inverse_bandwidth():
    """通信时间 ∝ 1/带宽(带宽翻倍→通信减半)。"""
    t1 = communication_time(8, message_bytes=1e8, bandwidth=1e10, scaling="ring")
    t2 = communication_time(8, message_bytes=1e8, bandwidth=2e10, scaling="ring")
    assert t2 == pytest.approx(t1 / 2.0)


def test_comm_base_property():
    """comm_base 属性 = 数据量 / 带宽。"""
    comm = CommModel(message_bytes=5e8, bandwidth=1e10, scaling="ring")
    assert comm.comm_base == pytest.approx(0.05)


def test_growth_values():
    """各拓扑增长因子 f(N) 的具体取值。"""
    comm_lin = CommModel(scaling="linear")
    comm_log = CommModel(scaling="log")
    comm_ring = CommModel(scaling="ring")
    assert comm_lin.growth(8) == pytest.approx(8.0)
    assert comm_log.growth(8) == pytest.approx(3.0)          # log2(8)=3
    assert comm_ring.growth(8) == pytest.approx(7.0 / 8.0)   # (8-1)/8


# =============================================================================
# 7) scaling_curve 结构与一致性
# =============================================================================

def test_scaling_curve_shapes_and_keys():
    comm = CommModel(message_bytes=2e8, bandwidth=1e10, scaling="ring")
    ns = [1, 2, 4, 8, 16, 32]
    curve = scaling_curve(ns, 0.05, comm)
    for k in ("n", "speedup", "efficiency", "ideal"):
        assert k in curve
        assert len(curve[k]) == len(ns)
    # ideal 就是 n 本身
    assert np.allclose(curve["ideal"], np.asarray(ns, dtype=float))
    # efficiency == speedup / n 逐点一致
    assert np.allclose(curve["efficiency"], curve["speedup"] / np.asarray(ns))


def test_scaling_curve_speedup_below_ideal():
    """真实加速比(有 s、有通信)处处 <= 理想线性(N=1 相等,其余更低)。"""
    comm = CommModel(message_bytes=2e8, bandwidth=1e10, scaling="ring")
    curve = scaling_curve([1, 2, 4, 8, 16, 32, 64], 0.05, comm)
    assert np.all(curve["speedup"] <= curve["ideal"] + 1e-9)


def test_scaling_curve_empty_raises():
    with pytest.raises(ValueError):
        scaling_curve([], 0.05, None)


# =============================================================================
# 8) half_efficiency_n �forforfor拐点
# =============================================================================

def test_half_efficiency_n_found():
    """自测参数下(s=0.05, ring, comm_base=0.02)半效率卡数应为 16。"""
    comm = CommModel(message_bytes=2e8, bandwidth=1e10, scaling="ring")
    assert half_efficiency_n(0.05, comm) == 16


def test_half_efficiency_n_none_for_perfect():
    """完全并行、无通信 → 效率恒为 1,永不跌破 0.5 → 返回 None。"""
    assert half_efficiency_n(0.0, None, n_max=1000) is None


def test_half_efficiency_more_comm_smaller_n():
    """通信越大,越早跌破半效率(拐点 N 越小)。"""
    s = 0.02
    low = CommModel(message_bytes=1e7, bandwidth=1e10, scaling="ring")
    high = CommModel(message_bytes=1e9, bandwidth=1e10, scaling="ring")
    n_low = half_efficiency_n(s, low)
    n_high = half_efficiency_n(s, high)
    assert n_high is not None and n_low is not None
    assert n_high < n_low


# =============================================================================
# 9) 异常与边界输入
# =============================================================================

@pytest.mark.parametrize("bad_n", [0, -1, -8])
def test_invalid_n_raises(bad_n):
    with pytest.raises(ValueError):
        amdahl_speedup(bad_n, 0.1)
    with pytest.raises(ValueError):
        speedup_with_comm(bad_n, 0.1, None)


def test_non_integer_n_raises():
    with pytest.raises(TypeError):
        amdahl_speedup(2.5, 0.1)


@pytest.mark.parametrize("bad_s", [-0.01, 1.01, 2.0, -1.0])
def test_invalid_serial_fraction_raises(bad_s):
    with pytest.raises(ValueError):
        amdahl_speedup(4, bad_s)


@pytest.mark.parametrize("bad_bw", [0.0, -1.0, -1e9])
def test_invalid_bandwidth_raises(bad_bw):
    with pytest.raises(ValueError):
        CommModel(message_bytes=1e8, bandwidth=bad_bw, scaling="ring")


def test_negative_message_bytes_raises():
    with pytest.raises(ValueError):
        CommModel(message_bytes=-1.0, bandwidth=1e10, scaling="ring")


def test_invalid_scaling_raises():
    with pytest.raises(ValueError):
        CommModel(message_bytes=1e8, bandwidth=1e10, scaling="banana")


def test_numpy_integer_n_accepted():
    """numpy 整数应被接受(方便从数组循环传入)。"""
    n = np.int64(8)
    assert speedup_with_comm(n, 0.05, None) == pytest.approx(amdahl_speedup(8, 0.05))
