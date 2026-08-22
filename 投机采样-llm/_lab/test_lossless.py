"""无损性（铁律一的 L1 口径）的判定性测试。

这些测试不是"跑一遍看起来对"，而是**精确枚举**投机采样的全部分支，
把输出分布与目标模型的链式概率逐点相减。误差是机器精度 => 命题为真。
"""
import numpy as np
import pytest

from spec import (ToyMarkov, beta_overlap, exact_stream_dist, exact_target_dist,
                  first_token_dist, iteration_dist, max_pointwise_error, perturb,
                  residual_dist, speculative_generate, speculative_step,
                  total_variation)

EPS = 1e-12


# ---------------------------------------------------------------- 基础恒等式

def test_residual_is_valid_distribution():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.dirichlet(np.ones(7))
        q = rng.dirichlet(np.ones(7))
        r = residual_dist(p, q)
        assert r.min() >= -EPS
        assert abs(r.sum() - 1.0) < EPS


def test_residual_mass_equals_rejection_prob():
    """sum_x max(0, p-q) == 1 - sum_x min(p,q) == TV(p,q)。三个量是同一个数。"""
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.dirichlet(np.ones(9))
        q = rng.dirichlet(np.ones(9))
        assert abs(np.maximum(0, p - q).sum() - (1 - beta_overlap(p, q))) < EPS
        assert abs(np.maximum(0, p - q).sum() - total_variation(p, q)) < EPS


def test_beta_equals_one_minus_tv():
    """接受率恒等式 beta = 1 - TV(p,q)。第 05 篇引用这一条。"""
    rng = np.random.default_rng(2)
    for _ in range(100):
        p = rng.dirichlet(np.ones(11))
        q = rng.dirichlet(np.ones(11))
        assert abs(beta_overlap(p, q) - (1.0 - total_variation(p, q))) < EPS


# ---------------------------------------------------------------- 旗舰：L1 无损

@pytest.mark.parametrize("V,gamma,eps", [
    (3, 1, 0.0), (3, 2, 0.5), (3, 3, 1.5),
    (4, 1, 4.0), (4, 2, 1.5), (4, 3, 0.5), (5, 2, 2.0),
])
def test_exact_distribution_lossless(V, gamma, eps):
    """精确枚举：投机采样输出流的前 3 个 token 的联合分布 == 目标模型链式概率。

    第 04 篇的核心断言就是这一条。注意它是**逐点相等**，不是"统计上无差异"。
    """
    tgt = ToyMarkov(V, seed=V, temp=1.0)
    drf = perturb(tgt, eps=eps, seed=V + 100)
    assert max_pointwise_error(tgt, drf, 0, 3, gamma) < EPS


def test_lossless_holds_for_terrible_draft():
    """草稿烂到接受率只有 0.3 时，分布依然精确无损 —— 草稿质量只影响速度。"""
    tgt = ToyMarkov(4, seed=4, temp=1.0)
    drf = perturb(tgt, eps=4.0, seed=104)
    alpha = beta_overlap(tgt.dist(0), drf.dist(0))
    assert alpha < 0.4, "这个用例要求草稿确实很烂"
    assert max_pointwise_error(tgt, drf, 0, 3, 3) < EPS


def test_lossless_holds_for_adversarial_draft():
    """最坏情况：草稿把几乎全部质量押在 target 最不可能的 token 上，接受率 <0.1。

    结论仍然是精确无损 —— 这条排除了"草稿得足够好，无损才成立"的误解。
    """
    tgt = ToyMarkov(4, seed=7, temp=1.0)
    drf = ToyMarkov.__new__(ToyMarkov)
    Q = np.full_like(tgt.T, 0.01)
    for s_ in range(tgt.V):
        Q[s_, int(np.argmin(tgt.T[s_]))] = 1.0
    drf.T, drf.V = Q / Q.sum(axis=1, keepdims=True), tgt.V
    assert beta_overlap(tgt.dist(0), drf.dist(0)) < 0.15
    assert max_pointwise_error(tgt, drf, 0, 3, 2) < EPS


def test_probabilities_sum_to_one():
    tgt = ToyMarkov(4, seed=3, temp=1.0)
    drf = perturb(tgt, eps=1.0, seed=33)
    for gamma in (1, 2, 3):
        d = iteration_dist(tgt, drf, 0, gamma)
        assert abs(sum(d.values()) - 1.0) < EPS
        st = exact_stream_dist(tgt, drf, 0, 3, gamma)
        assert abs(sum(st.values()) - 1.0) < 1e-9


def test_target_dist_reference_is_consistent():
    """对照组自身的自洽性：链式概率之和为 1。"""
    tgt = ToyMarkov(4, seed=9, temp=1.0)
    d = exact_target_dist(tgt, 0, 3)
    assert abs(sum(d.values()) - 1.0) < EPS
    assert len(d) == 4 ** 3


# ---------------------------------------------------------------- 常见错法确实有偏

def test_naive_resample_is_biased():
    """社区高频错法一：拒绝后直接从 p 重采。精确枚举证明它有偏（误差 >0.1）。

    第 04 篇与第 27 篇引用这一条。
    """
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p_true = tgt.dist(0)
    err = float(np.abs(first_token_dist(tgt, drf, 0, 3, "naive_p") - p_true).max())
    assert err > 1e-2, "naive_p 必须表现出可观测的偏差"


def test_threshold_rule_is_biased():
    """社区高频错法二：把 min(1,p/q) 的随机判据写成 p>=q 的确定判据。有偏。"""
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p_true = tgt.dist(0)
    err = float(np.abs(first_token_dist(tgt, drf, 0, 3, "threshold") - p_true).max())
    assert err > 1e-2


def test_no_bonus_token_is_still_lossless():
    """不发"赠品 token"**不会**破坏无损性，只是变慢。区分"错"与"慢"。"""
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p_true = tgt.dist(0)
    err = float(np.abs(first_token_dist(tgt, drf, 0, 3, "no_bonus") - p_true).max())
    assert err < EPS


def test_correct_variant_is_unbiased_where_others_fail():
    """同一组模型上：correct 无偏，另两种有偏 —— 对照才有说服力。"""
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p_true = tgt.dist(0)
    errs = {v: float(np.abs(first_token_dist(tgt, drf, 0, 3, v) - p_true).max())
            for v in ("correct", "naive_p", "threshold")}
    assert errs["correct"] < EPS < errs["naive_p"]
    assert errs["correct"] < errs["threshold"]


# ---------------------------------------------------------------- 采样实现与枚举一致

def test_sampling_matches_enumeration():
    """掷骰子的实现与精确枚举一致（大样本，卡方式的逐点比较）。

    这条保证 demo 里跑的那份代码就是被证明过的那个算法。
    """
    tgt = ToyMarkov(4, seed=2, temp=1.0)
    drf = perturb(tgt, eps=1.0, seed=22)
    gamma, n = 2, 200_000
    exact = iteration_dist(tgt, drf, 0, gamma)
    rng = np.random.default_rng(0)
    cnt = {}
    for _ in range(n):
        c = tuple(speculative_step(tgt, drf, 0, gamma, rng))
        cnt[c] = cnt.get(c, 0) + 1
    for k, pk in exact.items():
        if pk < 1e-3:
            continue
        emp = cnt.get(k, 0) / n
        # 3 倍标准误容差
        se = (pk * (1 - pk) / n) ** 0.5
        assert abs(emp - pk) < 4 * se + 1e-3, (k, pk, emp)


def test_generate_length_and_range():
    tgt = ToyMarkov(5, seed=1, temp=1.0)
    drf = perturb(tgt, eps=1.0, seed=11)
    rng = np.random.default_rng(0)
    toks, iters = speculative_generate(tgt, drf, 0, 50, 3, rng)
    assert len(toks) == 50
    assert 50 / 4 <= iters <= 50          # 每轮 1..gamma+1 个 token
    assert all(0 <= t < 5 for t in toks)


def test_chunk_length_bounds():
    """每轮产出必须落在 [1, gamma+1]。少于 1 表示实现漏掉了拒绝处的重采。"""
    tgt = ToyMarkov(5, seed=1, temp=1.0)
    drf = perturb(tgt, eps=2.0, seed=11)
    rng = np.random.default_rng(3)
    for gamma in (1, 2, 4):
        for _ in range(2000):
            c = speculative_step(tgt, drf, 0, gamma, rng)
            assert 1 <= len(c) <= gamma + 1


def test_residual_fallback_is_live_and_correct():
    """`residual_dist` 的兜底分支：初稿 docstring 的三句话全错，这条测真相。

    初稿说"p==q 时残差永远用不到，并在测试里断言这条路不会被走到"——
      ① 引的 `test_spec.py` 不存在（在本文件）；
      ② "永远用不到"不对；
      ③ 从来没有那样的测试。
    但对抗审稿给的反面说法（"一定会被走到"）**也不完全对**。实测 1000 组
    (seed, state) 上 `1-beta(p,p)` 的符号：**正 236 / 负 109 / 恰好 0 655**。
    也就是说：它是**浮点噪声，符号两边都可能**，恰好为 0 的只占三分之二。
    正 -> 拒绝分支被走到、用上兜底；负或 0 -> 跳过。**两种情况结果都对。**
    """
    import numpy as np
    signs = {"pos": 0, "neg": 0, "zero": 0}
    for seed in range(200):
        m = ToyMarkov(5, seed=seed)
        for st in range(5):
            r = 1.0 - beta_overlap(m.dist(st), m.dist(st))
            signs["pos" if r > 0 else ("neg" if r < 0 else "zero")] += 1
    assert signs["pos"] > 100, signs        # 确实有"会被走到"的情形
    assert signs["zero"] > 100, signs       # 也确实有"用不到"的情形
    # 兜底返回的是合法分布
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    r = residual_dist(tgt.dist(0), tgt.dist(0))
    assert abs(r.sum() - 1.0) < 1e-12 and r.min() >= 0
    # 无论走哪条分支，端到端分布仍精确等于 p
    assert max_pointwise_error(tgt, tgt, 0, 3, 3) < EPS
