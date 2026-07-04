"""
test_cost_model.py —— 验证 AllReduce α-β 代价模型的公式与定性结论。
运行:python -m pytest -q

覆盖点:
  · ring 通信量 = 2(P-1)/P·N、步数 = 2(P-1)
  · tree/DBT 步数 = 2·log2(P);DBT 带宽系数 = ring(最优)
  · α-β 线性、带宽换算
  · 定性:小消息 tree 赢、大消息 ring 赢、DBT 全程 Pareto 最优
  · crossover 交叉点的自洽性、ring 带宽渐近 →2N
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collective_cost import (  # noqa: E402
    alpha_beta_time, bandwidth_to_beta, _log2_ceil,
    ring_steps, tree_steps, dbt_steps,
    ring_bytes, tree_bytes, dbt_bytes,
    ring_coeff, tree_coeff, dbt_coeff,
    allreduce_time, bus_bandwidth, crossover_size, best_algo,
    Network, DEFAULT_NET, ALGOS,
)

POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


# --------------------------------------------------------------------------
# 1) ring 的黄金公式:通信量 = 2(P-1)/P·N,步数 = 2(P-1)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("P", [2, 3, 4, 8, 16, 64, 100])
@pytest.mark.parametrize("N", [1024.0, 1e6, 1e9])
def test_ring_communication_volume(P, N):
    assert math.isclose(ring_bytes(N, P), 2.0 * (P - 1) / P * N, rel_tol=1e-12)


@pytest.mark.parametrize("P", [2, 4, 8, 16, 64, 128])
def test_ring_steps_formula(P):
    assert ring_steps(P) == 2 * (P - 1)


def test_ring_bandwidth_asymptote_to_2N():
    """P→∞ 时 ring 带宽系数 → 2(即每节点搬 ~2N,达带宽下界)。"""
    assert ring_coeff(2) == pytest.approx(1.0)          # 2·1/2 = 1
    assert ring_coeff(1_000_000) == pytest.approx(2.0, abs=1e-4)
    # 单调递增趋近 2,永不超过 2
    assert ring_coeff(8) < ring_coeff(64) < 2.0


# --------------------------------------------------------------------------
# 2) tree / double-binary-tree 步数 = 2·log2(P)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("P", [2, 4, 8, 16, 32, 64, 128, 256, 1024])
def test_tree_steps_is_2_log2P(P):
    assert tree_steps(P) == 2 * int(math.log2(P))
    assert dbt_steps(P) == 2 * int(math.log2(P))


@pytest.mark.parametrize("P", POW2)
def test_dbt_same_latency_as_tree_but_bandwidth_as_ring(P):
    """DBT = 树的对数延迟 + 环的最优带宽(两全其美)。"""
    assert dbt_steps(P) == tree_steps(P)                 # 延迟同树
    assert math.isclose(dbt_coeff(P), ring_coeff(P))     # 带宽同环(最优)


def test_log2_ceil_handles_non_power_of_two():
    assert _log2_ceil(1) == 0
    assert _log2_ceil(2) == 1
    assert _log2_ceil(5) == 3        # 需要深度 3 的树
    assert _log2_ceil(8) == 3
    assert _log2_ceil(9) == 4
    assert _log2_ceil(1024) == 10


def test_single_node_is_free():
    assert ring_steps(1) == 0 and tree_steps(1) == 0 and dbt_steps(1) == 0
    assert allreduce_time("ring", N=1e9, P=1, alpha=1e-6, beta=1e-11) == 0.0


# --------------------------------------------------------------------------
# 3) α-β 基础:线性、带宽换算
# --------------------------------------------------------------------------
def test_alpha_beta_is_linear():
    assert alpha_beta_time(steps=10, nbytes=1e6, alpha=2e-6, beta=1e-11) == \
        pytest.approx(10 * 2e-6 + 1e6 * 1e-11)
    # 叠加性:steps 与 bytes 各自线性
    t1 = alpha_beta_time(4, 0, 3e-6, 1e-11)
    t2 = alpha_beta_time(0, 5e6, 3e-6, 1e-11)
    assert alpha_beta_time(4, 5e6, 3e-6, 1e-11) == pytest.approx(t1 + t2)


def test_bandwidth_to_beta_roundtrip():
    # 100 GB/s → β,再由 β 反推带宽
    beta = bandwidth_to_beta(100.0)
    assert beta == pytest.approx(1e-11)                  # 1/(100e9)
    assert (1.0 / beta) / 1e9 == pytest.approx(100.0)
    with pytest.raises(ValueError):
        bandwidth_to_beta(0)


def test_allreduce_time_matches_manual():
    net = Network(alpha=5e-6, bandwidth_gbps=100.0)
    P, N = 8, 4e6
    t = allreduce_time("ring", N, P, net.alpha, net.beta)
    manual = net.alpha * (2 * (P - 1)) + net.beta * (2 * (P - 1) / P) * N
    assert t == pytest.approx(manual)


# --------------------------------------------------------------------------
# 4) 定性结论:小消息 tree 赢、大消息 ring 赢
# --------------------------------------------------------------------------
@pytest.mark.parametrize("P", [8, 16, 64])
def test_small_message_tree_beats_ring(P):
    net = DEFAULT_NET
    tiny = 1024.0                                        # 1 KB:延迟主导
    t_ring = allreduce_time("ring", tiny, P, net.alpha, net.beta)
    t_tree = allreduce_time("tree", tiny, P, net.alpha, net.beta)
    assert t_tree < t_ring


@pytest.mark.parametrize("P", [8, 16, 64])
def test_large_message_ring_beats_tree(P):
    net = DEFAULT_NET
    huge = 1e9                                           # 1 GB:带宽主导
    t_ring = allreduce_time("ring", huge, P, net.alpha, net.beta)
    t_tree = allreduce_time("tree", huge, P, net.alpha, net.beta)
    assert t_ring < t_tree


@pytest.mark.parametrize("P", [4, 8, 16, 32, 64])
def test_dbt_is_pareto_best_everywhere(P):
    """DBT 在小/中/大各种消息下都不劣于 ring 和 tree(取全局最优)。"""
    net = DEFAULT_NET
    for N in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]:
        t_dbt = allreduce_time("double_binary_tree", N, P, net.alpha, net.beta)
        t_ring = allreduce_time("ring", N, P, net.alpha, net.beta)
        t_tree = allreduce_time("tree", N, P, net.alpha, net.beta)
        assert t_dbt <= t_ring + 1e-18
        assert t_dbt <= t_tree + 1e-18


# --------------------------------------------------------------------------
# 5) 交叉点 crossover 的自洽性
# --------------------------------------------------------------------------
@pytest.mark.parametrize("P", [8, 16, 64])
def test_crossover_between_ring_and_tree(P):
    net = DEFAULT_NET
    Nx = crossover_size(P, net.alpha, net.beta, "ring", "tree")
    assert Nx > 0
    # 交叉点处两算法耗时应相等
    t_ring = allreduce_time("ring", Nx, P, net.alpha, net.beta)
    t_tree = allreduce_time("tree", Nx, P, net.alpha, net.beta)
    assert t_ring == pytest.approx(t_tree, rel=1e-9)
    # 交叉点左侧 tree 赢、右侧 ring 赢
    assert allreduce_time("tree", Nx * 0.5, P, net.alpha, net.beta) < \
        allreduce_time("ring", Nx * 0.5, P, net.alpha, net.beta)
    assert allreduce_time("ring", Nx * 2, P, net.alpha, net.beta) < \
        allreduce_time("tree", Nx * 2, P, net.alpha, net.beta)


def test_crossover_moves_up_with_P():
    """P 越大,ring 的延迟惩罚越重,交叉点向更大消息移动。"""
    net = DEFAULT_NET
    n8 = crossover_size(8, net.alpha, net.beta, "ring", "tree")
    n64 = crossover_size(64, net.alpha, net.beta, "ring", "tree")
    assert n64 > n8 > 0


def test_ring_dbt_no_positive_crossover():
    """ring 与 DBT 带宽系数相同,只差步数 → 无正交叉点(DBT 恒不劣)。"""
    net = DEFAULT_NET
    assert crossover_size(16, net.alpha, net.beta, "ring", "double_binary_tree") <= 0


# --------------------------------------------------------------------------
# 6) busbw 与 best_algo 语义
# --------------------------------------------------------------------------
def test_busbw_positive_and_below_peak():
    net = DEFAULT_NET
    peak = net.bandwidth_gbps
    for algo in ALGOS:
        bw = bus_bandwidth(algo, 1e8, 16, net.alpha, net.beta)
        assert 0 < bw < peak * 1.001                     # 不可能超过链路峰值


def test_busbw_grows_with_message_size():
    """消息越大,固定 α 开销被摊薄,busbw 越接近峰值(大消息效率高)。"""
    net = DEFAULT_NET
    small = bus_bandwidth("ring", 1e4, 16, net.alpha, net.beta)
    large = bus_bandwidth("ring", 1e9, 16, net.alpha, net.beta)
    assert large > small


def test_best_algo_picks_expected():
    net = DEFAULT_NET
    # DBT 应在两端都最优
    assert best_algo(1e3, 64, net.alpha, net.beta) == "double_binary_tree"
    assert best_algo(1e9, 64, net.alpha, net.beta) == "double_binary_tree"
    # 若把 DBT 排除,小消息应是 tree、大消息应是 ring
    def best_of(names, N, P):
        return min(names, key=lambda k: allreduce_time(k, N, P, net.alpha, net.beta))
    assert best_of(["ring", "tree"], 1e3, 64) == "tree"
    assert best_of(["ring", "tree"], 1e9, 64) == "ring"
