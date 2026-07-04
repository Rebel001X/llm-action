# -*- coding: utf-8 -*-
"""
test_slo_harness.py —— SLO harness 的单元测试与性质测试（property tests）。

覆盖四类断言：
  A) 百分位计算正确（与手算 + numpy 对拍）。
  B) 负载升高，尾延迟（p95/p99）单调不降 —— 排队论的基本规律。
  C) goodput 关于 QPS 先增后（因尾延迟违反 SLO）掉头，达标比例随负载升高单调不增。
  D) 各种恒等式 / 边界：goodput<=throughput、利用率∈[0,1]、TTFT<=E2E 等。

跑法：  python -m pytest -q
"""

import math
import os
import sys

import pytest

# 让测试无论从哪个目录运行都能 import 到被测模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slo_harness import (  # noqa: E402
    SLO,
    Request,
    WorkloadConfig,
    compute_metrics,
    generate_workload,
    percentile,
    percentiles,
    run_once,
    simulate,
    sweep_qps,
)

# numpy 用于「对拍」百分位；本机已装
np = pytest.importorskip("numpy")


# ----------------------------------------------------------------------
# A. 百分位正确性
# ----------------------------------------------------------------------
def test_percentile_hand_computed():
    """在小数据上和手算结果比对（线性插值法）。"""
    data = [1, 2, 3, 4]  # n=4，下标 0..3
    # p50: pos = 3*0.5 = 1.5 → 在 x[1]=2 与 x[2]=3 间插值 0.5 → 2.5
    assert percentile(data, 50) == pytest.approx(2.5)
    # p0 / p100 = 最小 / 最大
    assert percentile(data, 0) == pytest.approx(1.0)
    assert percentile(data, 100) == pytest.approx(4.0)
    # p25: pos = 3*0.25 = 0.75 → x[0]=1, x[1]=2, frac=0.75 → 1.75
    assert percentile(data, 25) == pytest.approx(1.75)


def test_percentile_matches_numpy():
    """随机数据上和 numpy.percentile(method='linear') 完全一致。"""
    rng = np.random.default_rng(42)
    for _ in range(20):
        arr = rng.normal(size=rng.integers(2, 500)).tolist()
        for q in [0, 1, 25, 50, 90, 95, 99, 99.9, 100]:
            mine = percentile(arr, q)
            ref = float(np.percentile(arr, q, method="linear"))
            assert mine == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_percentile_single_element():
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 99) == 7.0


def test_percentile_unsorted_input():
    """输入不必预排序，结果与排序后一致。"""
    assert percentile([3, 1, 2], 50) == pytest.approx(2.0)
    assert percentile([9, 1, 5, 3, 7], 75) == pytest.approx(7.0)


def test_percentiles_batch_equals_singles():
    """批量 percentiles 与逐个 percentile 结果一致。"""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=200).tolist()
    qs = [50, 95, 99]
    batch = percentiles(arr, qs)
    for q in qs:
        assert batch[q] == pytest.approx(percentile(arr, q))


def test_percentile_rejects_bad_q():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)
    with pytest.raises(ValueError):
        percentile([], 50)


# ----------------------------------------------------------------------
# B. 仿真基本性质
# ----------------------------------------------------------------------
# 单请求服务时间 ≈ 1.7s（decode 主导）。为让系统在中等 QPS 下才饱和，
# 用较多的并发槽位 W（对应连续批处理里能同时在飞的序列数）。
# W=32 时理论容量 ≈ 32/1.7 ≈ 18.6 req/s，knee 落在 QPS≈16~18。
BASE_WORKERS = 32


def _base_cfg(**kw):
    d = dict(num_requests=3000, mean_prompt_len=512, mean_output_len=128, seed=7)
    d.update(kw)
    return WorkloadConfig(**d)


def test_simulate_causality():
    """时间戳因果链必须成立：arrival <= start <= first_token <= finish。"""
    reqs = generate_workload(_base_cfg(qps=8.0))
    served = simulate(reqs, num_workers=BASE_WORKERS)
    for r in served:
        assert r.arrival <= r.start + 1e-9
        assert r.start <= r.first_token + 1e-9
        assert r.first_token <= r.finish + 1e-9
        # first_token - start 应等于 prefill_time
        assert (r.first_token - r.start) == pytest.approx(r.prefill_time)
        assert (r.finish - r.first_token) == pytest.approx(r.decode_time)


def test_ttft_le_e2e():
    """TTFT 一定不超过 E2E（首 token 早于完成）。"""
    m = run_once(_base_cfg(qps=8.0), SLO(), num_workers=BASE_WORKERS)
    assert m.ttft_ms["p50"] <= m.e2e_ms["p50"] + 1e-6
    assert m.ttft_ms["p99"] <= m.e2e_ms["p99"] + 1e-6


def test_more_workers_reduce_latency():
    """同样负载下，worker 越多，排队越少，p99 E2E 应下降；单槽利用率降。"""
    cfg = _base_cfg(qps=16.0)  # 该负载下 8 个槽位会拥塞，32 个较宽松
    m8 = run_once(cfg, SLO(), num_workers=8)
    m32 = run_once(cfg, SLO(), num_workers=32)
    assert m32.e2e_ms["p99"] < m8.e2e_ms["p99"]
    assert m32.utilization <= m8.utilization + 1e-9  # 摊薄到更多槽位，单槽利用率降


# ----------------------------------------------------------------------
# C. 负载升高 → 尾延迟单调上升；goodput 规律
# ----------------------------------------------------------------------
def test_tail_latency_monotonic_in_qps():
    """
    核心性质：固定 worker 数，QPS 单调上升时，p99 E2E 尾延迟应单调不降。
    这是排队论「利用率越高、等待越长」的直接体现。
    """
    qps_list = [2, 6, 10, 14, 18, 22]
    slo = SLO()
    cfg = _base_cfg(num_requests=5000)
    metrics = sweep_qps(qps_list, cfg, slo, num_workers=BASE_WORKERS)
    p99s = [m.e2e_ms["p99"] for m in metrics]
    p95s = [m.e2e_ms["p95"] for m in metrics]
    # 允许极小的数值抖动容差
    for a, b in zip(p99s, p99s[1:]):
        assert b >= a - 1e-6, f"p99 尾延迟未单调上升: {p99s}"
    for a, b in zip(p95s, p95s[1:]):
        assert b >= a - 1e-6, f"p95 尾延迟未单调上升: {p95s}"


def test_goodput_ratio_monotonic_decreasing():
    """
    达标比例（goodput_ratio）随 QPS 升高应单调不增：
    负载越高越拥塞，越多请求违反 SLO。
    """
    qps_list = [2, 6, 10, 14, 18, 22]
    slo = SLO(ttft_ms=800, tpot_ms=20, e2e_ms=5000)
    cfg = _base_cfg(num_requests=5000)
    metrics = sweep_qps(qps_list, cfg, slo, num_workers=BASE_WORKERS)
    ratios = [m.goodput_ratio for m in metrics]
    for a, b in zip(ratios, ratios[1:]):
        assert b <= a + 1e-6, f"达标比例未单调下降: {ratios}"
    # 低负载几乎全达标，高负载明显掉下来（否则 SLO 设得没意义）
    assert ratios[0] > 0.9
    assert ratios[-1] < ratios[0]


def test_goodput_le_throughput():
    """恒等式：goodput <= throughput（达标票是全部票的子集）。"""
    for q in [2, 8, 16, 22]:
        m = run_once(_base_cfg(qps=q), SLO(), num_workers=BASE_WORKERS)
        assert m.goodput <= m.throughput + 1e-9
        assert 0.0 <= m.goodput_ratio <= 1.0


def test_utilization_in_range_and_increases_with_qps():
    """利用率必须落在 [0,1]，且随 QPS 升高单调不减（更多活干、槽位更忙）。"""
    qps_list = [2, 4, 8, 12, 16]
    metrics = sweep_qps(qps_list, _base_cfg(), SLO(), num_workers=BASE_WORKERS)
    utils = [m.utilization for m in metrics]
    for u in utils:
        assert 0.0 <= u <= 1.0
    for a, b in zip(utils, utils[1:]):
        assert b >= a - 1e-6, f"利用率未随 QPS 上升: {utils}"


# ----------------------------------------------------------------------
# D. SLO 判定与边界
# ----------------------------------------------------------------------
def test_slo_is_good_logic():
    """逐维度验证 SLO.is_good：任一维度超标即判负。"""
    # 构造一个已完成的请求：ttft=0.1s, tpot=0.02s, e2e=1.0s
    r = Request(
        req_id=0, arrival=0.0, prompt_len=100, output_len=11,
        prefill_time=0.1, decode_time=0.2,
    )
    r.start = 0.0
    r.first_token = 0.1     # ttft = 0.1s = 100ms
    r.finish = 0.3          # e2e = 0.3s = 300ms; decode=0.2s over 10 tokens → tpot=20ms
    assert r.ttft == pytest.approx(0.1)
    assert r.tpot == pytest.approx(0.02)
    assert r.e2e == pytest.approx(0.3)

    assert SLO(ttft_ms=150, tpot_ms=30, e2e_ms=400).is_good(r) is True
    assert SLO(ttft_ms=50).is_good(r) is False    # ttft 超
    assert SLO(tpot_ms=10).is_good(r) is False    # tpot 超
    assert SLO(e2e_ms=200).is_good(r) is False    # e2e 超
    # None 表示不设限，应放行
    assert SLO(ttft_ms=None, tpot_ms=None, e2e_ms=None).is_good(r) is True


def test_tpot_single_token_no_zero_division():
    """output_len<=1 时 tpot 不应除零。"""
    r = Request(0, 0.0, 100, 1, prefill_time=0.1, decode_time=0.0)
    r.start = 0.0
    r.first_token = 0.1
    r.finish = 0.1
    assert r.tpot == pytest.approx(0.0)  # 约定：返回 decode_time


def test_reproducible_with_seed():
    """同一 seed 结果完全可复现。"""
    m1 = run_once(_base_cfg(qps=5.0, seed=123), SLO(), num_workers=2)
    m2 = run_once(_base_cfg(qps=5.0, seed=123), SLO(), num_workers=2)
    assert m1.e2e_ms["p99"] == pytest.approx(m2.e2e_ms["p99"])
    assert m1.goodput == pytest.approx(m2.goodput)


def test_simulate_rejects_zero_workers():
    with pytest.raises(ValueError):
        simulate(generate_workload(_base_cfg(qps=1.0)), num_workers=0)


def test_workload_lengths_are_positive():
    """合成的长度必须为正（否则服务时间无意义）。"""
    reqs = generate_workload(_base_cfg(qps=2.0, num_requests=500))
    for r in reqs:
        assert r.prompt_len >= 1
        assert r.output_len >= 1
        assert r.prefill_time > 0
        assert r.decode_time >= 0


def test_arrivals_are_sorted():
    """到达流按时间升序。"""
    reqs = generate_workload(_base_cfg(qps=3.0, num_requests=500))
    arrivals = [r.arrival for r in reqs]
    assert arrivals == sorted(arrivals)
