# -*- coding: utf-8 -*-
"""
稀缺性模型的单元测试 (pytest)。

覆盖点(对应任务要求):
  1. 能效改进抵消部分增长        —— efficiency_gain 越大,累计碳排越小
  2. 可再生降碳                  —— 可再生渗透越高,累计碳排越小
  3. 超预算年份计算正确          —— budget_exceeded_year 命中/不命中
  4. 边界情况                    —— years=1、零增长、100% 可再生、除零、参数校验
外加:曲线机制、Kaya 乘积、单调性、归一化、复现性等本质性质。
"""

import numpy as np
import pytest

import scarcity_model as sm


# ---------------------------------------------------------------------------
# 基础曲线机制
# ---------------------------------------------------------------------------
def test_compute_curve_starts_at_one_and_compounds():
    c = sm.compute_curve(4, 0.30)
    assert c[0] == pytest.approx(1.0)
    assert c[1] == pytest.approx(1.30)
    assert c[3] == pytest.approx(1.30 ** 3)
    # 严格单调递增(增长率为正)
    assert np.all(np.diff(c) > 0)


def test_efficiency_curve_decays():
    e = sm.efficiency_curve(4, 0.15)
    assert e[0] == pytest.approx(1.0)
    assert e[1] == pytest.approx(0.85)
    assert e[3] == pytest.approx(0.85 ** 3)
    # 严格单调递减(有改进)
    assert np.all(np.diff(e) < 0)


def test_efficiency_zero_gain_is_flat():
    # 零能效改进 => 强度恒为 1
    e = sm.efficiency_curve(5, 0.0)
    assert np.allclose(e, 1.0)


def test_renewable_curve_linear_interpolation():
    r = sm.renewable_curve(5, 0.2, 0.6)
    assert r[0] == pytest.approx(0.2)
    assert r[-1] == pytest.approx(0.6)
    # 中间点为线性:0.2, 0.3, 0.4, 0.5, 0.6
    assert r[2] == pytest.approx(0.4)


def test_renewable_curve_single_year():
    # 边界:只有第 0 年,直接返回 start
    r = sm.renewable_curve(1, 0.35, 0.9)
    assert r.shape == (1,)
    assert r[0] == pytest.approx(0.35)


def test_carbon_intensity_from_renewable():
    # 可再生 100% => 碳强度 0;可再生 0% => 碳强度 = fossil
    c_full = sm.carbon_intensity_curve(1, 1.0, 1.0, fossil=1.0)
    c_none = sm.carbon_intensity_curve(1, 0.0, 0.0, fossil=1.0)
    assert c_full[0] == pytest.approx(0.0)
    assert c_none[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 年碳排 / 累计碳排:归一化与单调性
# ---------------------------------------------------------------------------
def test_annual_year0_equals_base_emission():
    # 归一化保证:第 0 年年碳排恰好 == base_emission_gt
    scn = sm.Scenario(name="t", base_emission_gt=0.123)
    ann = sm.annual_emissions(scn)
    assert ann[0] == pytest.approx(0.123)


def test_cumulative_is_monotonic_nondecreasing():
    # 累计碳排来自 cumsum,且年碳排非负 => 单调不减(稀缺性水位只涨不落)
    scn = sm.Scenario(name="t")
    cum = sm.cumulative_emissions(scn)
    assert np.all(np.diff(cum) >= -1e-12)


def test_cumulative_equals_prefix_sum_of_annual():
    scn = sm.Scenario(name="t")
    ann = sm.annual_emissions(scn)
    cum = sm.cumulative_emissions(scn)
    assert np.allclose(cum, np.cumsum(ann))


def test_annual_is_kaya_product():
    # 验证年碳排确实是三因子乘积(归一化后)的形状
    scn = sm.Scenario(name="t", efficiency_gain=0.1, compute_growth=0.2)
    comp = sm.compute_curve(scn.years, scn.compute_growth)
    eff = sm.efficiency_curve(scn.years, scn.efficiency_gain)
    carb = sm.carbon_intensity_curve(scn.years, scn.renewable_start,
                                     scn.renewable_end, scn.grid_carbon_fossil)
    raw = comp * eff * carb
    expected = scn.base_emission_gt * raw / raw[0]
    assert np.allclose(sm.annual_emissions(scn), expected)


# ---------------------------------------------------------------------------
# 要求 1:能效改进抵消部分增长
# ---------------------------------------------------------------------------
def test_efficiency_gain_reduces_cumulative():
    base = sm.Scenario(name="low", efficiency_gain=0.05)
    better = sm.Scenario(name="high", efficiency_gain=0.25)
    # 其它条件相同,能效改进更强 => 累计碳排更小
    assert (sm.cumulative_emissions(better)[-1]
            < sm.cumulative_emissions(base)[-1])


def test_efficiency_can_flip_verdict():
    # 强能效改进能把"超预算"翻转成"预算内"(抵消增长的力度足够)
    weak = sm.Scenario(name="weak", efficiency_gain=0.05, carbon_budget_gt=5.0)
    strong = sm.Scenario(name="strong", efficiency_gain=0.30, carbon_budget_gt=5.0)
    assert sm.budget_exceeded_year(weak) is not None      # 超了
    assert sm.budget_exceeded_year(strong) is None        # 没超


def test_efficiency_offsets_growth_year_over_year():
    # 当能效改进率 == 算力增长率的对应比例时,年碳排持平(抵消)。
    # 若 (1+g)*(1-r) == 1 且碳强度恒定,则年碳排应当逐年不变。
    g = 0.25
    r = 1.0 - 1.0 / (1.0 + g)   # 解 (1+g)(1-r)=1 => r = g/(1+g)
    scn = sm.Scenario(name="offset", compute_growth=g, efficiency_gain=r,
                      renewable_start=0.4, renewable_end=0.4)  # 碳强度恒定
    ann = sm.annual_emissions(scn)
    assert np.allclose(ann, ann[0])   # 完美抵消 => 年碳排持平


# ---------------------------------------------------------------------------
# 要求 2:可再生降碳
# ---------------------------------------------------------------------------
def test_renewable_reduces_cumulative():
    low = sm.Scenario(name="low", renewable_start=0.2, renewable_end=0.3)
    high = sm.Scenario(name="high", renewable_start=0.2, renewable_end=0.9)
    # 起点相同,末端可再生更高 => 累计碳排更小
    assert (sm.cumulative_emissions(high)[-1]
            < sm.cumulative_emissions(low)[-1])


def test_full_renewable_from_start_is_zero_carbon():
    # 边界+除零:第 0 年就 100% 可再生 => 碳强度恒 0 => 全程零碳(不会崩)
    scn = sm.Scenario(name="green", renewable_start=1.0, renewable_end=1.0)
    ann = sm.annual_emissions(scn)
    assert np.allclose(ann, 0.0)
    assert sm.budget_exceeded_year(scn) is None


def test_full_renewable_end_pushes_intensity_to_zero():
    scn = sm.Scenario(name="g", renewable_start=0.3, renewable_end=1.0)
    ann = sm.annual_emissions(scn)
    # 最后一年可再生 100% => 该年碳排应为 0
    assert ann[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 要求 3:超预算年份计算正确
# ---------------------------------------------------------------------------
def test_budget_exceeded_year_none_when_within():
    # 预算极大 => 永不超预算 => None
    scn = sm.Scenario(name="rich", carbon_budget_gt=1e9)
    assert sm.budget_exceeded_year(scn) is None


def test_budget_exceeded_year_zero_when_first_year_over():
    # 预算比第 0 年碳排还小 => 第 0 年(索引 0)就超
    scn = sm.Scenario(name="poor", base_emission_gt=1.0, carbon_budget_gt=0.5)
    assert sm.budget_exceeded_year(scn) == 0


def test_budget_exceeded_year_matches_manual_search():
    # 用累计曲线手工找第一个 > budget 的下标,和函数结果对拍
    scn = sm.Scenario(name="t", carbon_budget_gt=3.0)
    cum = sm.cumulative_emissions(scn)
    manual = None
    for i, v in enumerate(cum):
        if v > 3.0:
            manual = i
            break
    assert sm.budget_exceeded_year(scn) == manual


def test_budget_exceeded_uses_strict_greater():
    # 精确等于预算不算超(side='right' 语义);略微超一点点才算
    scn = sm.Scenario(name="t", years=3, base_emission_gt=1.0,
                      compute_growth=0.0, efficiency_gain=0.0,
                      renewable_start=0.0, renewable_end=0.0,
                      grid_carbon_fossil=1.0)
    # 年碳排恒为 1.0 => 累计 = [1,2,3]
    cum = sm.cumulative_emissions(scn)
    assert np.allclose(cum, [1.0, 2.0, 3.0])
    scn_eq = sm.Scenario(name="eq", years=3, base_emission_gt=1.0,
                         compute_growth=0.0, efficiency_gain=0.0,
                         renewable_start=0.0, renewable_end=0.0,
                         carbon_budget_gt=3.0)
    # 累计末值 == 预算 3.0,恰好相等 => 不算超 => None
    assert sm.budget_exceeded_year(scn_eq) is None
    scn_over = sm.Scenario(name="over", years=3, base_emission_gt=1.0,
                           compute_growth=0.0, efficiency_gain=0.0,
                           renewable_start=0.0, renewable_end=0.0,
                           carbon_budget_gt=2.9)
    # 预算 2.9,累计第 2 年(=3.0)> 2.9 => 返回 2
    assert sm.budget_exceeded_year(scn_over) == 2


# ---------------------------------------------------------------------------
# 剩余预算曲线 & 压力指数
# ---------------------------------------------------------------------------
def test_remaining_budget_curve():
    scn = sm.Scenario(name="t", carbon_budget_gt=4.0)
    rem = sm.remaining_budget_curve(scn)
    cum = sm.cumulative_emissions(scn)
    assert np.allclose(rem, 4.0 - cum)
    # 剩余预算应单调不增(累计不减)
    assert np.all(np.diff(rem) <= 1e-12)


def test_scarcity_pressure_ratio():
    scn = sm.Scenario(name="t", carbon_budget_gt=2.0)
    total = float(sm.cumulative_emissions(scn)[-1])
    assert sm.scarcity_pressure(scn) == pytest.approx(total / 2.0)


def test_scarcity_pressure_gt_one_means_over():
    over = sm.Scenario(name="over", carbon_budget_gt=0.01)
    within = sm.Scenario(name="within", carbon_budget_gt=1e9)
    assert sm.scarcity_pressure(over) > 1.0
    assert sm.scarcity_pressure(within) < 1.0


# ---------------------------------------------------------------------------
# run_scenario 打包结果
# ---------------------------------------------------------------------------
def test_run_scenario_result_consistency():
    scn = sm.Scenario(name="t", carbon_budget_gt=3.0)
    res = sm.run_scenario(scn)
    assert res.total_emission_gt == pytest.approx(res.cumulative[-1])
    assert res.within_budget == (res.exceeded_year is None)
    assert res.budget_gt == 3.0
    assert res.annual.shape == (scn.years,)
    assert res.cumulative.shape == (scn.years,)


def test_default_scenarios_run():
    # 预置情景库全部能跑通,且能效/可再生更强的情景压力更小
    scns = sm.default_scenarios()
    assert len(scns) == 4
    pressures = [sm.scarcity_pressure(s) for s in scns]
    # 双管齐下(最后一个)应当是压力最小的
    assert pressures[-1] == min(pressures)


# ---------------------------------------------------------------------------
# 边界与参数校验
# ---------------------------------------------------------------------------
def test_single_year_scenario():
    scn = sm.Scenario(name="one", years=1, base_emission_gt=0.2)
    ann = sm.annual_emissions(scn)
    cum = sm.cumulative_emissions(scn)
    assert ann.shape == (1,)
    assert cum[0] == pytest.approx(0.2)


def test_zero_growth_zero_efficiency_flat_emissions():
    # 无增长、无能效改进、碳强度恒定 => 年碳排持平
    scn = sm.Scenario(name="flat", compute_growth=0.0, efficiency_gain=0.0,
                      renewable_start=0.4, renewable_end=0.4)
    ann = sm.annual_emissions(scn)
    assert np.allclose(ann, ann[0])


def test_reproducible():
    # 无随机性:同参数两次运行结果完全一致
    scn = sm.Scenario(name="t")
    a = sm.cumulative_emissions(scn)
    b = sm.cumulative_emissions(scn)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kwargs", [
    dict(years=0),
    dict(compute_growth=-1.0),
    dict(compute_growth=-2.0),
    dict(efficiency_gain=1.0),
    dict(efficiency_gain=-0.1),
    dict(renewable_start=1.5),
    dict(renewable_end=-0.1),
    dict(base_emission_gt=-1.0),
    dict(carbon_budget_gt=0.0),
    dict(carbon_budget_gt=-3.0),
])
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        sm.Scenario(name="bad", **kwargs)


def test_negative_growth_shrinks_compute():
    # 边界:算力可以负增长(需求萎缩),曲线单调递减但为正
    c = sm.compute_curve(4, -0.2)
    assert np.all(c > 0)
    assert np.all(np.diff(c) < 0)
