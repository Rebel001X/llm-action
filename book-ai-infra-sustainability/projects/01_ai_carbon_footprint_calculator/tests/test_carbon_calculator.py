# -*- coding: utf-8 -*-
"""
carbon_calculator 的单元测试
============================

测试策略:每条测试对应一个「物理直觉」或「书中论点」,失败即说明公式或直觉被破坏。
用 pytest.approx 处理浮点比较;用参数化覆盖多电网。
"""

import math
import os
import sys

import pytest

# 让测试无论从哪个目录运行都能 import 到被测模块(项目根加入 sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import carbon_calculator as cc  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 能耗 = 功率 × 时长 (× PUE)
# ---------------------------------------------------------------------------

def test_it_energy_basic():
    # 1000 W = 1 kW,跑 1 卡 1 小时 = 1 kWh
    assert cc.it_energy_kwh(num_gpus=1, gpu_power_watts=1000,
                            hours=1) == pytest.approx(1.0)


def test_it_energy_scales_linearly_with_all_factors():
    base = cc.it_energy_kwh(2, 400, 10)
    # 卡数翻倍 → 能耗翻倍
    assert cc.it_energy_kwh(4, 400, 10) == pytest.approx(2 * base)
    # 时长翻倍 → 能耗翻倍
    assert cc.it_energy_kwh(2, 400, 20) == pytest.approx(2 * base)
    # 功率翻倍 → 能耗翻倍
    assert cc.it_energy_kwh(2, 800, 10) == pytest.approx(2 * base)


def test_utilization_reduces_energy():
    full = cc.it_energy_kwh(10, 400, 5, utilization=1.0)
    half = cc.it_energy_kwh(10, 400, 5, utilization=0.5)
    assert half == pytest.approx(full / 2)


def test_apply_pue_multiplies():
    # 能耗 = IT 能耗 × PUE
    assert cc.apply_pue(100.0, 1.5) == pytest.approx(150.0)
    # PUE=1.0(完美)时设施能耗 = IT 能耗
    assert cc.apply_pue(100.0, 1.0) == pytest.approx(100.0)


def test_training_energy_equals_power_times_time_times_pue():
    cfg = cc.TrainingConfig(num_gpus=100, gpu_power_watts=500,
                            train_hours=10, pue=1.5)
    r = cc.compute_training_footprint(cfg)
    # 手算:0.5kW × 100 × 10h = 500 kWh(IT);× 1.5 = 750 kWh(设施)
    assert r.it_energy_kwh == pytest.approx(500.0)
    assert r.facility_energy_kwh == pytest.approx(750.0)


# ---------------------------------------------------------------------------
# 2. 碳 = 能耗 × 碳强度
# ---------------------------------------------------------------------------

def test_carbon_equals_energy_times_intensity():
    # 1000 kWh × 500 gCO2/kWh = 500000 g = 0.5 t
    assert cc.energy_to_carbon_tco2(1000.0, 500.0) == pytest.approx(0.5)


def test_zero_intensity_grid_zero_carbon():
    # 纯清洁电网(碳强度 0)→ 碳排为 0,但能耗仍存在
    cfg = cc.TrainingConfig(grid_gco2_per_kwh=0.0)
    r = cc.compute_training_footprint(cfg)
    assert r.carbon_tco2 == pytest.approx(0.0)
    assert r.facility_energy_kwh > 0.0


def test_carbon_scales_with_intensity():
    cfg_dirty = cc.TrainingConfig(grid_gco2_per_kwh=900.0)
    cfg_clean = cc.TrainingConfig(grid_gco2_per_kwh=90.0)
    dirty = cc.compute_training_footprint(cfg_dirty)
    clean = cc.compute_training_footprint(cfg_clean)
    # 碳强度 10 倍 → 碳排 10 倍(能耗不变)
    assert dirty.carbon_tco2 == pytest.approx(10 * clean.carbon_tco2)
    assert dirty.facility_energy_kwh == pytest.approx(clean.facility_energy_kwh)


# ---------------------------------------------------------------------------
# 3. 水 = 能耗 × 水耗系数
# ---------------------------------------------------------------------------

def test_water_equals_energy_times_coeff():
    assert cc.energy_to_water_liters(1000.0, 1.8) == pytest.approx(1800.0)


def test_water_scales_with_energy():
    small = cc.compute_training_footprint(cc.TrainingConfig(train_hours=10))
    big = cc.compute_training_footprint(cc.TrainingConfig(train_hours=20))
    assert big.water_liters == pytest.approx(2 * small.water_liters)


# ---------------------------------------------------------------------------
# 4. 可再生占比越高 → 碳越低(单调性)
# ---------------------------------------------------------------------------

def test_renewable_share_monotonic():
    intensities = [cc.renewable_share_to_intensity(s)
                   for s in [0.0, 0.25, 0.5, 0.75, 1.0]]
    # 严格单调递减
    for a, b in zip(intensities, intensities[1:]):
        assert b < a


def test_full_renewable_lowest_carbon():
    dirty_intensity = cc.renewable_share_to_intensity(0.0)   # 全化石
    clean_intensity = cc.renewable_share_to_intensity(1.0)   # 全清洁
    dirty = cc.compute_training_footprint(
        cc.TrainingConfig(grid_gco2_per_kwh=dirty_intensity))
    clean = cc.compute_training_footprint(
        cc.TrainingConfig(grid_gco2_per_kwh=clean_intensity))
    assert clean.carbon_tco2 < dirty.carbon_tco2


def test_compare_grids_sorted_and_clean_first():
    cfg = cc.TrainingConfig(name="test")
    results = cc.compare_grids(cfg)
    # 返回按碳排升序:第一个碳最低
    carbons = [r.carbon_tco2 for r in results]
    assert carbons == sorted(carbons)
    # 北欧/法国这类低碳电网应排在煤电之前
    assert results[0].carbon_tco2 <= results[-1].carbon_tco2
    # 能耗对所有电网相同(只有碳/水随电网变的是碳)
    energies = [r.facility_energy_kwh for r in results]
    assert all(math.isclose(e, energies[0]) for e in energies)


# ---------------------------------------------------------------------------
# 5. 推理按 token 摊薄
# ---------------------------------------------------------------------------

def test_inference_per_token_amortization():
    cfg = cc.InferenceConfig(num_gpus=8, gpu_power_watts=400, pue=1.4,
                             throughput_tokens_per_sec=5000,
                             grid_gco2_per_kwh=475.0)
    r = cc.compute_inference_footprint(cfg, total_tokens=1_000_000)
    # 每千 token 指标应为正
    assert r.extra["gco2_per_1k_tokens"] > 0
    assert r.extra["wh_per_1k_tokens"] > 0
    assert r.extra["ml_water_per_1k_tokens"] > 0


def test_inference_double_tokens_double_absolute_but_same_per_token():
    cfg = cc.InferenceConfig()
    r1 = cc.compute_inference_footprint(cfg, total_tokens=1_000_000)
    r2 = cc.compute_inference_footprint(cfg, total_tokens=2_000_000)
    # 绝对碳排翻倍
    assert r2.carbon_tco2 == pytest.approx(2 * r1.carbon_tco2)
    # 但每千 token 的强度不变(摊薄的本质:单位成本恒定)
    assert r2.extra["gco2_per_1k_tokens"] == pytest.approx(
        r1.extra["gco2_per_1k_tokens"])


def test_inference_zero_tokens_zero_footprint():
    cfg = cc.InferenceConfig()
    r = cc.compute_inference_footprint(cfg, total_tokens=0)
    assert r.carbon_tco2 == pytest.approx(0.0)
    assert r.extra["gco2_per_1k_tokens"] == pytest.approx(0.0)


def test_higher_throughput_lower_per_token_carbon():
    slow = cc.InferenceConfig(throughput_tokens_per_sec=2000)
    fast = cc.InferenceConfig(throughput_tokens_per_sec=8000)
    rs = cc.compute_inference_footprint(slow, 1_000_000)
    rf = cc.compute_inference_footprint(fast, 1_000_000)
    # 吞吐越高,处理同样 token 用时越短 → 每千 token 碳排越低(效率的价值)
    assert rf.extra["gco2_per_1k_tokens"] < rs.extra["gco2_per_1k_tokens"]


# ---------------------------------------------------------------------------
# 6. PUE 敏感性 & 参数校验(坑点)
# ---------------------------------------------------------------------------

def test_higher_pue_more_carbon():
    low = cc.compute_training_footprint(cc.TrainingConfig(pue=1.1))
    high = cc.compute_training_footprint(cc.TrainingConfig(pue=1.8))
    assert high.carbon_tco2 > low.carbon_tco2
    assert high.facility_energy_kwh > low.facility_energy_kwh


def test_pue_below_one_raises():
    with pytest.raises(ValueError):
        cc.TrainingConfig(pue=0.9)


def test_bad_utilization_raises():
    with pytest.raises(ValueError):
        cc.TrainingConfig(utilization=1.5)
    with pytest.raises(ValueError):
        cc.TrainingConfig(utilization=0.0)


def test_zero_throughput_raises():
    with pytest.raises(ValueError):
        cc.InferenceConfig(throughput_tokens_per_sec=0)


def test_negative_intensity_raises():
    with pytest.raises(ValueError):
        cc.TrainingConfig(grid_gco2_per_kwh=-1)


# ---------------------------------------------------------------------------
# 7. 端到端量级 sanity check(数值不能离谱)
# ---------------------------------------------------------------------------

def test_end_to_end_magnitude_reasonable():
    # 一个大型训练:1024 卡 × 700W × 720h,PUE 1.4,全球平均电网
    cfg = cc.TrainingConfig(name="big", num_gpus=1024, gpu_power_watts=700,
                            train_hours=720, pue=1.4,
                            grid_gco2_per_kwh=475.0)
    r = cc.compute_training_footprint(cfg)
    # IT 能耗手算:0.7 × 1024 × 720 = 516096 kWh
    assert r.it_energy_kwh == pytest.approx(0.7 * 1024 * 720)
    # 设施能耗为百万 kWh 量级,碳排为百吨量级 —— 与公开大模型训练估算同数量级
    assert 1e5 < r.facility_energy_kwh < 1e7
    assert 100 < r.carbon_tco2 < 1000
