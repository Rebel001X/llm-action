"""
test_spec_decode.py —— 投机解码建模的金标准测试

覆盖:
  · 闭式期望公式在手算小例子 / 边界上正确;单调性
  · 蒙特卡洛模拟均值 ≈ 闭式期望(容差内对拍)
  · 接受率越高 → 加速比越大
  · 存在最优草稿长度 k*(内部极大值,非边界)
  · 最优 k* 随接受率 α 增大而不减
  · 实测加速比 ≈ 闭式加速比
  · 盈亏平衡接受率:低于它 speedup < 1

运行:python -m pytest -q
"""
import math
import sys
import os

import numpy as np
import pytest

# 允许 tests/ 直接 import 上级目录的 spec_decode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_decode import (
    expected_tokens, simulate_mean_tokens, simulate_step,
    expected_speedup, simulate_speedup, optimal_k, optimal_speedup,
    breakeven_alpha, speedup_grid, DEFAULT_C,
)


# ---------------------------------------------------------------- 闭式期望
def test_expected_tokens_hand_computed():
    # α=0.5, k=2:E = (1 - 0.5^3)/(1 - 0.5) = (1 - 0.125)/0.5 = 1.75
    assert math.isclose(expected_tokens(0.5, 2), 1.75, rel_tol=1e-12)
    # α=0.8, k=3:E = (1 - 0.8^4)/(0.2) = (1 - 0.4096)/0.2 = 2.952
    assert math.isclose(expected_tokens(0.8, 3), 2.952, rel_tol=1e-12)


def test_expected_tokens_boundaries():
    # α=0:每轮只敲定 1 个(修正 token)
    assert math.isclose(expected_tokens(0.0, 5), 1.0)
    # α=1:全接受 + 1 个 bonus = k+1
    assert math.isclose(expected_tokens(1.0, 5), 6.0)
    # 恒有 1 ≤ E ≤ k+1
    for a in [0.1, 0.3, 0.55, 0.9]:
        for k in [1, 3, 7]:
            e = expected_tokens(a, k)
            assert 1.0 - 1e-9 <= e <= k + 1 + 1e-9


def test_expected_tokens_monotonic():
    # 固定 k,E 随 α 递增;固定 α,E 随 k 递增
    ks = 4
    es_alpha = [expected_tokens(a, ks) for a in np.linspace(0.05, 0.95, 10)]
    assert all(b > a for a, b in zip(es_alpha, es_alpha[1:]))
    a = 0.7
    es_k = [expected_tokens(a, k) for k in range(1, 8)]
    assert all(b > a_ for a_, b in zip(es_k, es_k[1:]))


# ---------------------------------------------------------------- 模拟对拍
@pytest.mark.parametrize("alpha,k", [(0.3, 3), (0.5, 4), (0.7, 5), (0.9, 6)])
def test_simulation_matches_closed_form(alpha, k):
    mc = simulate_mean_tokens(alpha, k, n_steps=300_000, seed=123)
    exact = expected_tokens(alpha, k)
    assert abs(mc - exact) < 0.02, f"MC={mc} vs 闭式={exact}"


def test_simulate_step_in_range():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        t = simulate_step(0.6, 4, rng)
        assert 1 <= t <= 5           # 1 ≤ n+1 ≤ k+1


def test_simulation_converges_more_samples_tighter():
    # 样本越多,与闭式的偏差应总体更小(用两个量级对比)
    exact = expected_tokens(0.7, 5)
    err_small = abs(simulate_mean_tokens(0.7, 5, n_steps=2_000, seed=7) - exact)
    err_big = abs(simulate_mean_tokens(0.7, 5, n_steps=500_000, seed=7) - exact)
    assert err_big < err_small


# ---------------------------------------------------------------- 加速比
def test_higher_acceptance_more_speedup():
    # 固定 k、c,加速比随 α 单调递增
    speeds = [expected_speedup(a, k=5, c=DEFAULT_C) for a in [0.2, 0.4, 0.6, 0.8, 0.95]]
    assert all(b > a for a, b in zip(speeds, speeds[1:]))


def test_speedup_sim_matches_closed_form():
    mc = simulate_speedup(0.8, 4, c=0.2, n_steps=300_000, seed=11)
    exact = expected_speedup(0.8, 4, c=0.2)
    assert abs(mc - exact) < 0.02


def test_low_alpha_high_cost_can_be_slowdown():
    # 接受率低 + 草稿不便宜 ⇒ 加速比 < 1(反而变慢)
    assert expected_speedup(0.15, k=6, c=0.5) < 1.0
    # 接受率高 + 草稿便宜 ⇒ 明显 > 1
    assert expected_speedup(0.9, k=5, c=0.1) > 1.5


# ---------------------------------------------------------------- 最优 k
def test_optimal_k_is_interior_peak():
    # α=0.8, c=0.2:最优 k 应落在 (1, k_max) 内部,且峰值严格高于两端
    k_max = 30
    kstar = optimal_k(0.8, c=0.2, k_max=k_max)
    assert 1 <= kstar < k_max
    s_star = expected_speedup(0.8, kstar, 0.2)
    assert s_star > expected_speedup(0.8, k_max, 0.2)   # 右侧确实回落 ⇒ 内部极大
    assert s_star >= expected_speedup(0.8, 1, 0.2)


def test_optimal_k_grows_with_alpha():
    # α 越高,越值得提更长的草稿 ⇒ k* 不减
    ks = [optimal_k(a, c=0.2, k_max=40) for a in [0.5, 0.7, 0.85, 0.95]]
    assert all(b >= a for a, b in zip(ks, ks[1:]))
    assert ks[-1] > ks[0]            # 端到端确实增大


def test_optimal_speedup_matches_grid_max():
    # optimal_speedup 应等于在同一网格上暴力搜到的最大值
    alpha, c, k_max = 0.85, 0.15, 25
    brute = max(expected_speedup(alpha, k, c) for k in range(1, k_max + 1))
    assert math.isclose(optimal_speedup(alpha, c, k_max), brute, rel_tol=1e-12)


# ---------------------------------------------------------------- 盈亏平衡
def test_breakeven_alpha_is_root():
    k, c = 6, 0.2
    a_star = breakeven_alpha(k, c)
    assert 0.0 < a_star < 1.0
    assert math.isclose(expected_speedup(a_star, k, c), 1.0, abs_tol=1e-6)
    # 临界点两侧:低于变慢,高于变快
    assert expected_speedup(a_star - 0.05, k, c) < 1.0
    assert expected_speedup(a_star + 0.05, k, c) > 1.0


# ---------------------------------------------------------------- 网格形状
def test_speedup_grid_shape_and_values():
    alphas = [0.3, 0.6, 0.9]
    ks = [1, 2, 4, 8]
    g = speedup_grid(alphas, ks, c=0.2)
    assert g.shape == (3, 4)
    # 抽查一个格点与逐点函数一致
    assert math.isclose(g[2, 3], expected_speedup(0.9, 8, 0.2), rel_tol=1e-12)
