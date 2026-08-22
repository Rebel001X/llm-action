"""test_prehistory.py —— 第 08 篇（史前史）的断言验证。

三组断言，对应史前史三条分支：
  A. Blockwise 的**精确匹配判据**：T=0 下是 L2（复用 test_caliber.py 已验证的结论），
     T>0 下输出分布塌成 one-hot，TV = 1 - max(p)，且**与草稿好坏无关**。
  B. NAT 的**条件独立假设**：最优独立近似 = 边缘乘积，代价 = total correlation。
  C. Jacobi 的**前缀一错全废**：luck=0 时每轮净前进恰好 1 token，迭代次数 = 序列长度。

统计类命题一律多次重复（trials）并固定种子。
"""
from __future__ import annotations

import numpy as np
import pytest

from prehistory import (bimodal_example, exact_match_bias, exact_match_output_dist,
                        jacobi_rounds, jacobi_scan, kl, marginals, product_dist,
                        total_correlation)
from spec import ToyMarkov, perturb

# ---------------------------------------------------------------------------
# A. Blockwise / 精确匹配判据
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V,seed", [(5, 0), (5, 3), (8, 1), (8, 4)])
def test_exact_match_rule_collapses_sampling_to_greedy(V, seed):
    """精确匹配判据下，输出恒为 argmax(p) 的 one-hot —— 采样被整个关掉。"""
    tgt = ToyMarkov(V, seed=seed, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=seed + 11)
    for last in range(V):
        p, q = tgt.dist(last), drf.dist(last)
        d = exact_match_output_dist(p, q)
        assert d.sum() == pytest.approx(1.0)
        assert d[int(np.argmax(p))] == pytest.approx(1.0)
        assert int(np.count_nonzero(d)) == 1


@pytest.mark.parametrize("V,seed", [(5, 0), (8, 1), (12, 2)])
def test_exact_match_bias_equals_one_minus_max_prob(V, seed):
    """偏差有闭式：TV(out, p) = 1 - max(p)，最大逐点误差同样是 1 - max(p)。"""
    tgt = ToyMarkov(V, seed=seed, temp=1.0)
    drf = perturb(tgt, eps=1.0, seed=seed + 5)
    for last in range(V):
        p, q = tgt.dist(last), drf.dist(last)
        err, tv = exact_match_bias(p, q)
        assert tv == pytest.approx(1.0 - float(p.max()), abs=1e-12)
        assert err == pytest.approx(1.0 - float(p.max()), abs=1e-12)
        assert tv > 0.3          # 玩具词表上这已经是巨大的分布偏移，不是数值噪声


def test_exact_match_bias_is_independent_of_draft_quality():
    """**与 L1 正好相反的性质**：换草稿不改变偏差一个小数位。

    L1（min(1,p/q) + 残差重采）下草稿只影响速度不影响分布；
    exact-match 判据下草稿连分布都碰不到 —— 因为输出压根不来自 q。
    这条是"必须换判据"而不是"必须换更好的草稿"的直接证据。
    """
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    p = tgt.dist(0)
    tvs, agrees = [], []
    for eps in (0.0, 0.5, 1.2, 3.0, 8.0):
        drf = perturb(tgt, eps=eps, seed=1)
        tvs.append(exact_match_bias(p, drf.dist(0))[1])
        agrees.append(float(np.mean(tgt.T.argmax(1) == drf.T.argmax(1))))
    assert max(tvs) - min(tvs) < 1e-12          # 偏差完全不动
    assert max(agrees) - min(agrees) > 0.5      # 而草稿质量确实拉开了差距


def test_exact_match_is_lossless_only_when_target_is_deterministic():
    """精确匹配判据"无损"的充要条件：目标分布本身已经是 one-hot（即 T=0）。

    这就是本库口径里 L2 与 L1 的分界线的可判定形式。
    """
    p_sharp = np.array([0.0, 0.0, 1.0, 0.0])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert exact_match_bias(p_sharp, q)[1] == pytest.approx(0.0, abs=1e-12)
    p_soft = np.array([0.3, 0.3, 0.25, 0.15])
    assert exact_match_bias(p_soft, q)[1] > 0.5


# ---------------------------------------------------------------------------
# B. NAT / 条件独立假设
# ---------------------------------------------------------------------------


def test_best_independent_approximation_is_product_of_marginals():
    """在独立分布族里，边缘乘积是 KL(P||·) 的最小值点。随机试探 4000 次，无一更优。"""
    rng = np.random.default_rng(7)
    for _ in range(200):
        P = rng.dirichlet(np.ones(27)).reshape(3, 3, 3)
        base = kl(P, product_dist(marginals(P)))
        for _ in range(20):
            alt = [rng.dirichlet(np.ones(3)) for _ in range(3)]
            assert kl(P, product_dist(alt)) >= base - 1e-9


@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (2, 3, 4), (3, 3, 3)])
def test_independent_factorization_cost_equals_total_correlation(shape):
    """恒等式：min_Q∈独立族 KL(P||Q) = TC(P) = ΣH(Y_t) - H(Y)。

    这条把「条件独立假设的代价」从形容词变成一个可以打印出来的数。
    """
    rng = np.random.default_rng(int(np.prod(shape)))
    for _ in range(30):
        P = rng.dirichlet(np.ones(int(np.prod(shape)))).reshape(shape)
        assert kl(P, product_dist(marginals(P))) == pytest.approx(
            total_correlation(P), abs=1e-9)


def test_total_correlation_is_zero_iff_independent():
    rng = np.random.default_rng(3)
    ms = [rng.dirichlet(np.ones(4)) for _ in range(3)]
    assert total_correlation(product_dist(ms)) == pytest.approx(0.0, abs=1e-12)
    P, _, _ = bimodal_example()
    assert total_correlation(P) > 0.6


def test_bimodal_example_leaks_half_the_mass():
    """Gu 等 2017 的 'Danke schön / Vielen Dank' 例子：

    最优独立近似把 **一半** 的概率质量放到目标分布里概率为 0 的串上，
    代价恰好 1 bit = ln2 nats。这是 multimodality problem 的最小数值形态。
    """
    P, _, _ = bimodal_example()
    Q = product_dist(marginals(P))
    assert float(Q[P == 0].sum()) == pytest.approx(0.5, abs=1e-12)
    assert kl(P, Q) == pytest.approx(np.log(2.0), abs=1e-12)
    assert total_correlation(P) == pytest.approx(np.log(2.0), abs=1e-12)


def test_cost_grows_with_sequence_length_for_correlated_targets():
    """代价随序列长度增长：把双峰例子拼接 T 次，TC = T·ln2，线性增长。

    这解释了"加算力不解决"：代价是**目标分布**的性质，且随生成长度累积。
    """
    P, _, _ = bimodal_example()
    prev = 0.0
    for reps in (1, 2, 3):
        J = P
        for _ in range(reps - 1):
            J = np.tensordot(J, P, axes=0)
        tc = total_correlation(J)
        assert tc == pytest.approx(reps * np.log(2.0), abs=1e-9)
        assert tc > prev
        prev = tc


# ---------------------------------------------------------------------------
# C. Jacobi / 不动点迭代
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [8, 16, 32, 64])
def test_jacobi_advances_exactly_one_token_per_iteration(T):
    """luck=0（前缀一错全废）时收敛轮数 = 序列长度，净前进恰好 1 token/轮。

    含义：迭代次数与自回归**完全相同**，而每轮要算 T 个位置 —— 净亏损。
    """
    for seed in range(5):
        rounds, adv = jacobi_rounds(T, luck=0.0, seed=seed)
        assert rounds == T
        assert adv == pytest.approx(1.0)


def test_jacobi_ideal_speedup_is_exactly_one_without_luck():
    """理想加速比（假设并行验证免费）在 luck=0 时恰好 1.000 —— 一个 token 都没省。"""
    scan = jacobi_scan(T=32, lucks=(0.0,), trials=20)
    assert scan[0][2] == pytest.approx(1.0, abs=1e-9)


def test_jacobi_needs_large_luck_rate_to_reach_useful_speedup():
    """要越过 1.5×，本模型需要 luck 显著大于 0.2。

    对照 CLLM 实测：原模型 fast-forward 1.1 token/iteration（含白拿的 1.0），
    净收益约 0.1 —— 离这个门槛差一个数量级。
    """
    scan = dict((lk, sp) for lk, _, sp in
                jacobi_scan(T=32, lucks=(0.1, 0.2, 0.3, 0.5), trials=100))
    assert scan[0.1] < 1.2
    assert scan[0.2] < 1.5
    assert scan[0.5] > 1.5
    assert scan[0.1] < scan[0.2] < scan[0.3] < scan[0.5]   # 单调
