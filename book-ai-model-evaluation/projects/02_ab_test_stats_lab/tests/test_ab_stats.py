# -*- coding: utf-8 -*-
"""
test_ab_stats.py —— A/B 统计核心库的单元测试

测试策略（对应 README 的验收标准）：
    1. 分布函数对拍已知表值（norm/t 的 CDF/PPF）
    2. 检验统计量/p 值对拍手算已知数据
    3. 样本量公式在已知输入下给出预期数量级，并与 power 互为反函数
    4. CI 覆盖率的蒙特卡洛验证（正确的 95% CI 覆盖率应≈0.95）
    5. peeking 抬高假阳性率（偷看 FPR 显著高于名义 α）
运行：python -m pytest -q
"""
import math
import os
import sys

import numpy as np
import pytest

# 让测试无论从哪个目录运行都能 import 到上一层的 ab_stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ab_stats as ab  # noqa: E402


# =============================================================================
# 1. 分布函数：对拍标准表值
# =============================================================================
class TestDistributions:
    def test_norm_cdf_known(self):
        # Φ(0)=0.5，Φ(1.96)≈0.975，Φ(-1.96)≈0.025
        assert ab.norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
        assert ab.norm_cdf(1.96) == pytest.approx(0.9750021, abs=1e-6)
        assert ab.norm_cdf(-1.96) == pytest.approx(0.0249979, abs=1e-6)

    def test_norm_ppf_known(self):
        # 常用临界值
        assert ab.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
        assert ab.norm_ppf(0.95) == pytest.approx(1.644854, abs=1e-5)
        assert ab.norm_ppf(0.80) == pytest.approx(0.841621, abs=1e-5)

    def test_norm_ppf_cdf_inverse(self):
        # PPF 与 CDF 互为反函数
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            assert ab.norm_cdf(ab.norm_ppf(p)) == pytest.approx(p, abs=1e-8)

    def test_norm_ppf_domain(self):
        with pytest.raises(ValueError):
            ab.norm_ppf(0.0)
        with pytest.raises(ValueError):
            ab.norm_ppf(1.0)

    def test_t_cdf_known(self):
        # t_cdf(0,df)=0.5；Cauchy: t_cdf(1,1)=0.75（df=1 时 t 即柯西分布）
        assert ab.t_cdf(0.0, 20) == pytest.approx(0.5, abs=1e-10)
        assert ab.t_cdf(1.0, 1) == pytest.approx(0.75, abs=1e-8)
        assert ab.t_cdf(-1.0, 1) == pytest.approx(0.25, abs=1e-8)

    def test_t_ppf_known(self):
        # 标准 t 表：df=20 双尾 0.05 临界值=2.085963；df=1 → 12.706205
        assert ab.t_ppf(0.975, 20) == pytest.approx(2.085963, abs=1e-4)
        assert ab.t_ppf(0.975, 1) == pytest.approx(12.706205, abs=1e-3)
        assert ab.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-4)

    def test_t_ppf_cdf_inverse(self):
        # PPF 与 CDF 互为反函数（大自由度趋近正态）
        for df in [5, 20, 100]:
            for p in [0.1, 0.5, 0.9, 0.975]:
                assert ab.t_cdf(ab.t_ppf(p, df), df) == pytest.approx(p, abs=1e-6)

    def test_t_approaches_normal(self):
        # df 很大时 t 分布应逼近正态
        assert ab.t_ppf(0.975, 100000) == pytest.approx(ab.norm_ppf(0.975), abs=1e-3)


# =============================================================================
# 2. 两比例 z 检验：对拍手算
# =============================================================================
class TestTwoProportion:
    def test_known_statistic(self):
        # A: 200/1000=0.20, B: 250/1000=0.25。手算：
        # p_pool=(450)/(2000)=0.225, SE=sqrt(0.225*0.775*(2/1000))=0.018672
        # z=(0.25-0.20)/0.018672=2.6778
        r = ab.two_proportion_ztest(200, 1000, 250, 1000)
        assert r.effect == pytest.approx(0.05, abs=1e-12)
        assert r.statistic == pytest.approx(2.6778, abs=1e-3)
        # p 值双尾 = 2*(1-Φ(2.6778)) ≈ 0.00741
        assert r.p_value == pytest.approx(0.00741, abs=1e-4)
        assert r.significant

    def test_no_difference_gives_z_zero(self):
        # 两组完全一样 → z=0, p=1, 不显著
        r = ab.two_proportion_ztest(100, 1000, 100, 1000)
        assert r.statistic == pytest.approx(0.0, abs=1e-12)
        assert r.p_value == pytest.approx(1.0, abs=1e-9)
        assert not r.significant

    def test_ci_contains_effect(self):
        r = ab.two_proportion_ztest(200, 1000, 250, 1000)
        # 效应点估计必须落在自己的 CI 中间
        assert r.ci_low < r.effect < r.ci_high
        # 显著 → CI 不含 0
        assert not (r.ci_low <= 0 <= r.ci_high)

    def test_one_sided_smaller_p_than_two_sided(self):
        # 单尾（方向正确时）p 值应为双尾的一半
        r2 = ab.two_proportion_ztest(200, 1000, 250, 1000, alternative="two-sided")
        r1 = ab.two_proportion_ztest(200, 1000, 250, 1000, alternative="larger")
        assert r1.p_value == pytest.approx(r2.p_value / 2, rel=1e-6)

    def test_one_sided_ci_is_half_open(self):
        r = ab.two_proportion_ztest(200, 1000, 250, 1000, alternative="larger")
        assert r.ci_high == math.inf
        assert r.ci_low > -math.inf


# =============================================================================
# 3. 两均值 Welch t 检验
# =============================================================================
class TestTwoMean:
    def test_shifted_normal_detected(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 500)
        b = rng.normal(0.5, 1.0, 500)  # 均值差 0.5，样本量足够，应显著
        r = ab.two_mean_ttest(a, b)
        assert r.effect == pytest.approx(0.5, abs=0.15)
        assert r.significant
        assert r.p_value < 0.001

    def test_same_distribution_not_significant_on_average(self):
        # 同分布：单次可能偶然显著，但统计量应接近 0（这里固定种子取一个稳的）
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 1.0, 2000)
        b = rng.normal(0.0, 1.0, 2000)
        r = ab.two_mean_ttest(a, b)
        assert abs(r.statistic) < 3.0  # 极大概率不越界
        assert r.df > 3000  # Welch 自由度对大等方差样本应接近 n_a+n_b-2

    def test_welch_df_reasonable(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 2, 200)  # 方差不等
        r = ab.two_mean_ttest(a, b)
        # Welch 自由度应在 (min(n)-1, n_a+n_b-2) 之间
        assert 99 <= r.df <= 298

    def test_ci_contains_effect(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 300)
        b = rng.normal(0.3, 1, 300)
        r = ab.two_mean_ttest(a, b)
        assert r.ci_low < r.effect < r.ci_high


# =============================================================================
# 4. 样本量与功效：公式正确 + 互为反函数
# =============================================================================
class TestSampleSizeAndPower:
    def test_sample_size_proportions_magnitude(self):
        # 10%→12%，80% power，双尾 0.05：教科书量级约 3800/组
        n = ab.sample_size_two_proportions(0.10, 0.02, alpha=0.05, power=0.80)
        assert 3500 <= n <= 4200

    def test_power_roundtrip_proportions(self):
        # 用样本量公式算出 n，再算 power，应≈目标 power
        for power in [0.80, 0.90]:
            n = ab.sample_size_two_proportions(0.10, 0.02, power=power)
            back = ab.power_two_proportions(0.10, 0.02, n)
            assert back == pytest.approx(power, abs=0.01)

    def test_power_roundtrip_means(self):
        for power in [0.80, 0.90]:
            n = ab.sample_size_two_means(sigma=1.0, mde_abs=0.2, power=power)
            back = ab.power_two_means(1.0, 0.2, n)
            assert back == pytest.approx(power, abs=0.01)

    def test_mde_halving_quadruples_n(self):
        # MDE 减半 → 样本量约 ×4（统计学经典结论）
        n1 = ab.sample_size_two_means(1.0, 0.2, power=0.80)
        n2 = ab.sample_size_two_means(1.0, 0.1, power=0.80)
        assert n2 / n1 == pytest.approx(4.0, rel=0.02)

    def test_power_increases_with_n(self):
        # 样本量越大，功效越高（单调）
        p_small = ab.power_two_proportions(0.10, 0.02, 1000)
        p_large = ab.power_two_proportions(0.10, 0.02, 8000)
        assert p_large > p_small
        assert p_large > 0.95


# =============================================================================
# 5. CI 覆盖率：蒙特卡洛验证
# =============================================================================
class TestCICoverage:
    def test_95_ci_covers_about_95(self):
        cov = ab.ci_coverage_two_proportions(0.10, 0.12, n=3000,
                                             n_trials=3000, seed=0)
        # 3000 次试验，95% CI 覆盖率的采样误差约 ±0.8%，给宽松区间
        assert 0.93 <= cov <= 0.97

    def test_90_ci_covers_about_90(self):
        cov = ab.ci_coverage_two_proportions(0.20, 0.20, n=2000, alpha=0.10,
                                             n_trials=3000, seed=5)
        assert 0.87 <= cov <= 0.93


# =============================================================================
# 6. 偷看 peeking：抬高假阳性率
# =============================================================================
class TestPeeking:
    def test_fixed_fpr_near_alpha(self):
        # 只看终点一次（固定样本量）→ 假阳性率应≈名义 α=0.05
        r = ab.simulate_peeking(n_max=2000, n_peeks=10, alpha=0.05,
                                n_trials=3000, p=0.10, seed=0)
        assert 0.03 <= r.fpr_fixed <= 0.07

    def test_peeking_inflates_fpr(self):
        # 偷看 10 次的假阳性率应显著高于固定策略，且远超 0.05
        r = ab.simulate_peeking(n_max=2000, n_peeks=10, alpha=0.05,
                                n_trials=3000, p=0.10, seed=0)
        assert r.fpr_peeking > r.fpr_fixed
        assert r.fpr_peeking > 0.12  # 经验上 10 次偷看约 0.19~0.22

    def test_fpr_monotone_in_peeks(self):
        # 偷看次数越多，假阳性率越高（大体单调）
        counts, fprs = ab.peeking_fpr_curve(
            peek_counts=(1, 5, 20), n_max=2000, n_trials=2000, seed=0)
        assert fprs[0] < fprs[1] < fprs[2]
        assert fprs[0] == pytest.approx(0.05, abs=0.02)  # 看 1 次≈α
