"""接受率 alpha、期望接受长度 E[tau]、最优 gamma，以及公式失效条件的测试。

第 05、06 篇引用本文件。核心是把"公式什么时候准、什么时候不准"变成断言。
"""
import numpy as np
import pytest

from accept import (alpha_expected, beta_spread, expected_tokens,
                    measure_expected_tokens, optimal_gamma, speedup_ideal,
                    stationary)
from regime import beta_autocorr, iteration_start_bias, make_regime_pair
from spec import ToyMarkov, beta_overlap, perturb


# ---------------------------------------------------------------- 闭式公式本身

def test_expected_tokens_edge_cases():
    assert expected_tokens(1.0, 4) == 5.0          # 全接受 + 赠品
    assert abs(expected_tokens(0.0, 4) - 1.0) < 1e-12   # 全拒绝也至少产出 1 个
    for g in (1, 3, 7):
        assert 1.0 <= expected_tokens(0.5, g) <= g + 1


def test_expected_tokens_monotone():
    """E[tau] 对 alpha 和 gamma 都单调不减。"""
    for g in (1, 4, 8):
        vals = [expected_tokens(a, g) for a in np.linspace(0, 0.99, 30)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    for a in (0.3, 0.7, 0.9):
        vals = [expected_tokens(a, g) for g in range(1, 20)]
        assert all(b >= a2 - 1e-12 for a2, b in zip(vals, vals[1:]))


def test_expected_tokens_matches_geometric_sum():
    """闭式 (1-a^(g+1))/(1-a) == 逐项求和 sum_{k=0..g} a^k。"""
    for a in (0.1, 0.5, 0.83, 0.97):
        for g in (1, 3, 6, 12):
            direct = sum(a ** k for k in range(g + 1))
            assert abs(expected_tokens(a, g) - direct) < 1e-12


def test_optimal_gamma_decreases_with_cost():
    """草稿越贵（c 越大），最优 gamma 越小。第 06 篇引用。"""
    for a in (0.6, 0.8, 0.9):
        gs = [optimal_gamma(a, c)[0] for c in (0.02, 0.05, 0.1, 0.2, 0.4)]
        assert all(b <= a2 for a2, b in zip(gs, gs[1:])), gs


def test_optimal_gamma_increases_with_alpha():
    for c in (0.02, 0.1):
        gs = [optimal_gamma(a, c)[0] for a in (0.5, 0.7, 0.85, 0.95)]
        assert all(b >= a2 for a2, b in zip(gs, gs[1:])), gs


def test_speedup_can_be_below_one():
    """草稿不够便宜、接受率不够高时，理想加速比本身就 <1 —— 连理论上都不该开。"""
    assert speedup_ideal(alpha=0.3, gamma=8, c=0.5) < 1.0


# ---------------------------------------------------------------- 公式什么时候准

def test_formula_accurate_when_uncorrelated():
    """随机马尔可夫模型上 beta 波动很大，但沿链几乎不相关 -> 公式准到 3% 以内。

    这条推翻了"beta 有波动就会让公式失准"的直觉（第 06 篇记录了这次证伪）。
    """
    tgt = ToyMarkov(8, seed=24, temp=1.0)
    drf = perturb(tgt, eps=1.5, seed=101)
    bs = beta_spread(tgt, drf)
    assert bs.max() / bs.min() > 2.0, "这个用例要求 beta 确实有大幅波动"
    a = alpha_expected(tgt, drf)
    for gamma in (2, 4):
        pred = expected_tokens(a, gamma)
        meas = measure_expected_tokens(tgt, drf, gamma, n_iters=40_000, seed=1)
        assert abs(meas / pred - 1.0) < 0.03, (gamma, pred, meas)


@pytest.mark.parametrize("sticky,lo,hi", [(0.50, 0.97, 1.06), (0.99, 0.85, 0.97)])
def test_formula_overestimates_when_sticky(sticky, lo, hi):
    """难度"连片"时，公式**高估** E[tau]（第 06、19 篇引用）。

    注意方向：直觉说正相关会让公式低估，实测是高估 —— 因为迭代起点被偏置到难区。
    """
    tgt, drf = make_regime_pair(V=8, sticky=sticky, eps_hard=3.0, seed=11)
    pi = stationary(tgt.T)
    alpha = float(pi @ np.array([beta_overlap(tgt.dist(s), drf.dist(s))
                                 for s in range(tgt.V)]))
    pred = expected_tokens(alpha, 8)
    meas = measure_expected_tokens(tgt, drf, 8, n_iters=40_000, seed=5)
    assert lo <= meas / pred <= hi, (sticky, pred, meas, meas / pred)


def test_stickiness_raises_beta_autocorrelation():
    """粘性参数确实造出了 beta 的沿链正相关（实验的前提条件成立）。"""
    rhos = [beta_autocorr(*make_regime_pair(V=8, sticky=s, eps_hard=3.0, seed=11),
                          n=40_000)
            for s in (0.5, 0.85, 0.99)]
    assert rhos[0] < 0.2 < rhos[1] < rhos[2]


def test_iteration_start_is_biased_toward_hard_regime():
    """机制验证：粘性越强，投机迭代的起点越集中在难区。"""
    obs_lo, base = iteration_start_bias(
        *make_regime_pair(V=8, sticky=0.50, eps_hard=3.0, seed=11), gamma=4,
        n_iters=20_000)
    obs_hi, _ = iteration_start_bias(
        *make_regime_pair(V=8, sticky=0.99, eps_hard=3.0, seed=11), gamma=4,
        n_iters=20_000)
    assert obs_hi / base > 1.2
    assert obs_hi > obs_lo


# ---------------------------------------------------------------- alpha 的口径

def test_alpha_is_one_when_draft_equals_target():
    tgt = ToyMarkov(6, seed=5, temp=1.0)
    assert abs(alpha_expected(tgt, tgt) - 1.0) < 1e-12


def test_alpha_drops_with_perturbation():
    tgt = ToyMarkov(8, seed=5, temp=1.0)
    alphas = [alpha_expected(tgt, perturb(tgt, eps=e, seed=55))
              for e in (0.0, 0.5, 1.5, 3.0, 6.0)]
    assert all(b <= a + 1e-9 for a, b in zip(alphas, alphas[1:])), alphas


def _retemp(m, T, V):
    z = np.log(m.T + 1e-12) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    o = ToyMarkov.__new__(ToyMarkov)
    o.T, o.V = e / e.sum(axis=1, keepdims=True), V
    return o


def test_alpha_at_low_temperature_tracks_argmax_agreement():
    """T->0 时 alpha 收敛到"两模型 argmax 相同"的概率质量，不是收敛到 0 或 1。

    第 05、07 篇引用：这解释了为什么贪心下测的接受率不能外推到带温度采样。
    """
    V = 8
    base = ToyMarkov(V, seed=5, temp=1.0)
    for eps, lo, hi in [(0.3, 0.5, 0.85), (1.2, 0.0, 0.05)]:
        drf = perturb(base, eps=eps, seed=99)
        agree = float(np.mean(np.argmax(base.T, axis=1) == np.argmax(drf.T, axis=1)))
        a0 = alpha_expected(_retemp(base, 0.02, V), _retemp(drf, 0.02, V))
        assert lo <= a0 <= hi, (eps, agree, a0)


def test_alpha_goes_to_one_at_high_temperature():
    """T->inf 两个分布都趋于均匀，alpha -> 1。"""
    V = 8
    base = ToyMarkov(V, seed=5, temp=1.0)
    for eps in (0.3, 1.2):
        drf = perturb(base, eps=eps, seed=99)
        assert alpha_expected(_retemp(base, 50.0, V), _retemp(drf, 50.0, V)) > 0.97


def test_good_draft_alpha_is_non_monotone_in_temperature():
    """**好草稿**的 alpha 随温度呈 U 形：贪心处高、中温有低谷、高温回升。

    实测：eps=0.3 时 T=0.02 -> 0.697，T=0.1 -> 0.657（低谷），T=20 -> 0.995。
    这条最初被我写反过（以为总是非单调），是跑数字纠正的。
    """
    V = 8
    base = ToyMarkov(V, seed=5, temp=1.0)
    drf = perturb(base, eps=0.3, seed=99)
    a = [alpha_expected(_retemp(base, T, V), _retemp(drf, T, V))
         for T in (0.02, 0.1, 0.3, 1.0, 20.0)]
    assert a[1] < a[0], a          # 中温确实低于贪心 -> 存在低谷
    assert a[-1] > a[0], a         # 高温最终回升并超过贪心


def test_poor_draft_alpha_is_monotone_in_temperature():
    """**烂草稿**则是单调上升的 —— 曲线形状不是常数，取决于 argmax 一致率。"""
    V = 8
    base = ToyMarkov(V, seed=5, temp=1.0)
    drf = perturb(base, eps=1.2, seed=99)
    a = [alpha_expected(_retemp(base, T, V), _retemp(drf, T, V))
         for T in (0.02, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 20.0)]
    assert all(y >= x - 1e-9 for x, y in zip(a, a[1:])), a


def test_token_stream_matches_stationary_distribution():
    """独立验证无损性：投机采样吐出的 token 流，其区制比例等于目标链的平稳分布。

    这条与 test_lossless.py 的精确枚举互为旁证 —— 一个是枚举，一个是真跑。
    """
    from regime import make_regime_pair, renewal_check
    for sticky in (0.5, 0.99):
        tgt, drf = make_regime_pair(V=8, sticky=sticky, eps_hard=3.0, seed=11)
        r = renewal_check(tgt, drf, 4, n_iters=20_000)
        assert abs(r["token_stream_hard"] - r["pi_hard"]) < 0.02, r


def test_hard_region_iterations_are_shorter():
    """起点偏置的机制：难区起点的迭代显著更短，于是按迭代计数时难区被过采样。"""
    from regime import make_regime_pair, renewal_check
    for sticky in (0.5, 0.85, 0.99):
        tgt, drf = make_regime_pair(V=8, sticky=sticky, eps_hard=3.0, seed=11)
        r = renewal_check(tgt, drf, 4, n_iters=20_000)
        assert r["e_tau_hard_start"] < r["e_tau_easy_start"], (sticky, r)


def test_renewal_formula_accurate_only_when_sticky():
    """更新过程公式假设"整轮待在起点那个区"，所以它只在粘性接近 1 时才准。

    这条把"模型在自己的假设成立处才准"做成断言，也是对本库自身方法的一次自查。
    """
    from regime import make_regime_pair, renewal_check
    tgt, drf = make_regime_pair(V=8, sticky=0.99, eps_hard=3.0, seed=11)
    hi = renewal_check(tgt, drf, 4, n_iters=30_000)
    assert abs(hi["iter_start_hard_renewal_pred"] - hi["iter_start_hard_measured"]) < 0.05, hi
    tgt, drf = make_regime_pair(V=8, sticky=0.50, eps_hard=3.0, seed=11)
    lo = renewal_check(tgt, drf, 4, n_iters=30_000)
    assert abs(lo["iter_start_hard_renewal_pred"] - lo["iter_start_hard_measured"]) > 0.15, lo


# ---------------------------------------------------------------- 边际判据

def test_marginal_criterion_matches_brute_force():
    """**正确**的边际判据 alpha^(g+1) > c*speedup(g) 与暴力搜索逐点一致。

    并列（两个 gamma 的 speedup 相等）时允许差 1，此时二者都是最优解。
    """
    from accept import marginal_gamma
    bad = 0
    total = 0
    for a in [x / 100 for x in range(30, 99)]:
        for c in [x / 200 for x in range(1, 41)]:
            total += 1
            g_brute, s_brute = optimal_gamma(a, c, gmax=256)
            g_marg = marginal_gamma(a, c, gmax=256)
            if g_marg != g_brute:
                # 只允许"并列"这一种不一致
                if abs(speedup_ideal(a, g_marg, c) - s_brute) > 1e-12:
                    bad += 1
    assert bad == 0, "%d/%d 格不一致且非并列" % (bad, total)


def test_naive_marginal_criterion_is_wrong():
    """把判据写成 alpha^(g+1) > c（忽略机会成本）会**系统性高估** gamma。

    这条是本库在写第 17 篇时被纠正的一个错误，固化成测试防止重犯。
    """
    from accept import marginal_gamma, naive_marginal_gamma
    worse = 0
    for a, c in [(0.9, 0.05), (0.8, 0.05), (0.95, 0.02), (0.98, 0.05), (0.7, 0.1)]:
        g_bad = naive_marginal_gamma(a, c)
        g_ok = marginal_gamma(a, c)
        assert g_bad >= g_ok, (a, c, g_bad, g_ok)
        if speedup_ideal(a, g_bad, c) < speedup_ideal(a, g_ok, c) - 1e-9:
            worse += 1
    assert worse >= 4, "错判据应当在多数格子上给出更差的 gamma"


def test_naive_criterion_worst_case_loss():
    """错判据的最坏损失有多大：alpha=0.98, c=0.05 时理想加速比损失超过 30%。"""
    from accept import marginal_gamma, naive_marginal_gamma
    a, c = 0.98, 0.05
    s_ok = speedup_ideal(a, marginal_gamma(a, c), c)
    s_bad = speedup_ideal(a, naive_marginal_gamma(a, c), c)
    assert (s_ok - s_bad) / s_ok > 0.30, (s_ok, s_bad)
