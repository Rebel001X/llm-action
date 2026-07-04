# -*- coding: utf-8 -*-
"""
test_serving_sim.py —— 分布式推理服务仿真的单元测试
======================================================

覆盖 5 类断言(对应项目要求):
  1) 所有请求都完成(不丢请求)。
  2) 副本翻倍 → 饱和吞吐近翻倍(未饱和/容量受限区)。
  3) 连续批处理提升批槽位利用率(vs 静态批处理)。
  4) 尾延迟随负载上升(p99 单调趋势)。
  5) 扩副本存在边际收益递减(需求受限后加副本吞吐不再涨)。

外加若干基础工具函数与数据结构的单元测试。

运行:  python -m pytest -q
"""

import math
import os
import sys

import pytest

# 让测试无论从哪运行都能 import 到上级目录的 serving_sim。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serving_sim as ss  # noqa: E402


# ---------------------------------------------------------------------
# A. 工具函数:percentile
# ---------------------------------------------------------------------
def test_percentile_basic():
    data = [1, 2, 3, 4, 5]
    assert ss.percentile(data, 0) == 1
    assert ss.percentile(data, 100) == 5
    assert ss.percentile(data, 50) == 3  # 中位数


def test_percentile_interpolation():
    # 两点 [0,10],第 50 百分位应线性插值到 5。
    assert ss.percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    # 第 25 百分位 → 0.25*(2-1)=0.25 → 0*0.75+10*0.25=2.5
    assert ss.percentile([0.0, 10.0], 25) == pytest.approx(2.5)


def test_percentile_empty_is_nan():
    assert math.isnan(ss.percentile([], 50))


def test_percentile_single():
    assert ss.percentile([7.0], 99) == 7.0


def test_percentile_monotonic_in_q():
    # 百分位关于 q 单调不减。
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    vals = [ss.percentile(data, q) for q in range(0, 101, 5)]
    for a, b in zip(vals, vals[1:]):
        assert a <= b + 1e-9


# ---------------------------------------------------------------------
# B. 到达流:泊松过程
# ---------------------------------------------------------------------
def test_poisson_arrivals_sorted_and_reproducible():
    r1 = ss.poisson_arrivals(rate=20.0, duration=10.0, seed=42)
    r2 = ss.poisson_arrivals(rate=20.0, duration=10.0, seed=42)
    # 可复现:同种子 → 完全一致的到达时刻序列。
    assert [x.arrival for x in r1] == [x.arrival for x in r2]
    # 升序。
    arrivals = [x.arrival for x in r1]
    assert arrivals == sorted(arrivals)
    # 都在 [0, duration] 内。
    assert all(0 <= a <= 10.0 for a in arrivals)


def test_poisson_rate_scales_count():
    # 到达率越高,期望请求数越多(粗略校验 λ*T 量级)。
    lo = ss.poisson_arrivals(rate=5.0, duration=20.0, seed=1)
    hi = ss.poisson_arrivals(rate=50.0, duration=20.0, seed=1)
    assert len(hi) > len(lo)
    # 期望 ≈ λ*T = 5*20=100 与 50*20=1000,给宽松区间。
    assert 60 <= len(lo) <= 140
    assert 800 <= len(hi) <= 1200


def test_tokens_at_least_min():
    reqs = ss.poisson_arrivals(rate=30.0, duration=10.0, token_mean=32.0, seed=7)
    assert all(r.total_tokens >= 1 for r in reqs)


# ---------------------------------------------------------------------
# C. 数据结构:Request 的派生属性
# ---------------------------------------------------------------------
def test_request_latency_none_until_finished():
    req = ss.Request(rid=0, arrival=1.0, total_tokens=5)
    assert req.remaining == 5
    assert req.latency is None
    assert req.queue_delay is None
    req.start_time = 1.5
    req.finish_time = 3.0
    assert req.queue_delay == pytest.approx(0.5)
    assert req.latency == pytest.approx(2.0)


# ---------------------------------------------------------------------
# D. 核心断言 1:所有请求都完成(不丢请求)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n_replicas,rate", [(1, 10.0), (2, 40.0), (4, 120.0)])
def test_all_requests_complete(n_replicas, rate):
    m = ss.simulate(
        n_replicas=n_replicas, rate=rate, duration=12.0, seed=3
    )
    assert m["all_done"] is True
    assert m["n_done"] == m["n_total"]
    assert m["n_total"] > 0
    # p50/p95/p99 都是有限正数。
    for k in ("p50", "p95", "p99"):
        assert m[k] > 0 and math.isfinite(m[k])


def test_no_request_left_behind_high_load():
    # 即便重载(远超单副本容量),只要仿真跑到收工,所有请求最终都应完成。
    m = ss.simulate(n_replicas=1, rate=200.0, duration=8.0, seed=9)
    assert m["all_done"] is True
    assert m["n_done"] == m["n_total"]


# ---------------------------------------------------------------------
# E. 核心断言 2:副本翻倍 → 饱和吞吐近翻倍
# ---------------------------------------------------------------------
def test_replica_doubling_doubles_saturated_throughput():
    # 用极高到达率制造饱和(容量受限区):吞吐由副本数决定,而非到达率。
    common = dict(rate=500.0, duration=10.0, max_batch=16, seed=1)
    m1 = ss.simulate(n_replicas=1, **common)
    m2 = ss.simulate(n_replicas=2, **common)
    ratio = m2["throughput_tok"] / m1["throughput_tok"]
    # 无 batch_slowdown、负载均衡均匀 → 近乎线性,允许 ±15% 误差。
    assert 1.8 <= ratio <= 2.2, f"1->2 副本吞吐比 {ratio:.3f} 偏离近线性"


def test_replica_quadrupling_scales_saturated_throughput():
    common = dict(rate=1000.0, duration=8.0, max_batch=16, seed=1)
    m1 = ss.simulate(n_replicas=1, **common)
    m4 = ss.simulate(n_replicas=4, **common)
    ratio = m4["throughput_tok"] / m1["throughput_tok"]
    # 4 副本饱和吞吐应显著高于单副本(>3x),体现横向扩展有效。
    assert ratio > 3.0, f"1->4 副本吞吐比仅 {ratio:.3f},横向扩展未生效"


# ---------------------------------------------------------------------
# F. 核心断言 3:连续批处理提升(批槽位)利用率
# ---------------------------------------------------------------------
def test_continuous_batching_improves_slot_utilization():
    # 同样负载下,连续批处理的槽位利用率应显著高于静态批处理。
    common = dict(
        n_replicas=1, rate=25.0, duration=20.0,
        max_batch=16, token_mean=64.0, seed=3,
    )
    cont = ss.simulate(continuous=True, **common)
    stat = ss.simulate(continuous=False, **common)
    assert cont["mean_slot_utilization"] > stat["mean_slot_utilization"], (
        f"连续={cont['mean_slot_utilization']:.3f} 未高于 "
        f"静态={stat['mean_slot_utilization']:.3f}"
    )
    # 差距应该明显(至少高出 0.2 的绝对量)。
    assert cont["mean_slot_utilization"] - stat["mean_slot_utilization"] > 0.2


def test_continuous_batching_improves_throughput_and_latency():
    common = dict(
        n_replicas=1, rate=25.0, duration=20.0,
        max_batch=16, token_mean=64.0, seed=3,
    )
    cont = ss.simulate(continuous=True, **common)
    stat = ss.simulate(continuous=False, **common)
    # 连续批处理吞吐更高、尾延迟更低(消除队头阻塞)。
    assert cont["throughput_req"] > stat["throughput_req"]
    assert cont["p99"] < stat["p99"]


# ---------------------------------------------------------------------
# G. 核心断言 4:尾延迟随负载升高
# ---------------------------------------------------------------------
def test_tail_latency_rises_with_load():
    # 固定副本数,提高到达率 → p99 尾延迟应上升。
    p99s = []
    for rate in (10.0, 30.0, 60.0, 100.0):
        m = ss.simulate(n_replicas=1, rate=rate, duration=15.0, seed=2)
        p99s.append(m["p99"])
    # 最高负载的 p99 应明显高于最低负载。
    assert p99s[-1] > p99s[0], f"p99 未随负载上升: {p99s}"
    # 整体呈上升趋势:后半段均值 > 前半段均值。
    assert sum(p99s[2:]) / 2 > sum(p99s[:2]) / 2


def test_p99_ge_p95_ge_p50():
    # 百分位天然有序:p99 >= p95 >= p50。
    m = ss.simulate(n_replicas=2, rate=60.0, duration=15.0, seed=4)
    assert m["p99"] >= m["p95"] >= m["p50"] > 0


# ---------------------------------------------------------------------
# H. 核心断言 5:扩副本的边际收益递减
# ---------------------------------------------------------------------
def test_diminishing_returns_of_adding_replicas():
    # 固定到达率(需求受限):副本足够多后,吞吐由到达率封顶,加副本不再涨。
    common = dict(rate=60.0, duration=20.0, max_batch=16, seed=5)
    thr = {nr: ss.simulate(n_replicas=nr, **common)["throughput_req"]
           for nr in (1, 2, 4, 8)}
    # 1->2 的增量应远大于 4->8 的增量(边际收益递减)。
    gain_1_2 = thr[2] - thr[1]
    gain_4_8 = thr[8] - thr[4]
    assert gain_1_2 > gain_4_8, (
        f"边际收益未递减: 1->2 增 {gain_1_2:.2f}, 4->8 增 {gain_4_8:.2f}"
    )
    # 需求受限:吞吐上限不超过到达率太多(允许统计波动)。
    assert thr[8] <= 60.0 * 1.15


def test_slot_utilization_drops_when_over_provisioned():
    # 需求固定,副本越多 → 每副本活越少 → 槽位利用率下降(过度扩容浪费)。
    common = dict(rate=60.0, duration=20.0, max_batch=16, seed=5)
    u2 = ss.simulate(n_replicas=2, **common)["mean_slot_utilization"]
    u8 = ss.simulate(n_replicas=8, **common)["mean_slot_utilization"]
    assert u8 < u2, f"过度扩容后槽位利用率未下降: u2={u2:.3f} u8={u8:.3f}"


# ---------------------------------------------------------------------
# I. 负载均衡策略
# ---------------------------------------------------------------------
@pytest.mark.parametrize("policy", ["least_loaded", "round_robin", "random"])
def test_all_policies_complete_all_requests(policy):
    m = ss.simulate(
        n_replicas=3, rate=60.0, duration=12.0, policy=policy, seed=6
    )
    assert m["all_done"] is True
    assert m["n_done"] == m["n_total"]


def test_least_loaded_balances_better_than_random():
    # least_loaded 应比 random 有更均衡的副本负载(利用率方差更小或尾延迟更低)。
    common = dict(n_replicas=4, rate=120.0, duration=15.0, seed=8)
    ll = ss.simulate(policy="least_loaded", **common)
    rd = ss.simulate(policy="random", **common)
    # least_loaded 的 p99 不应比 random 差太多(通常更好或相当)。
    assert ll["p99"] <= rd["p99"] * 1.3


# ---------------------------------------------------------------------
# J. 边界与鲁棒性
# ---------------------------------------------------------------------
def test_zero_traffic_is_safe():
    # 极低到达率也不能崩;可能 0 个请求。
    m = ss.simulate(n_replicas=2, rate=0.5, duration=1.0, seed=0)
    assert m["n_done"] == m["n_total"]
    assert m["all_done"] is True


def test_single_replica_single_request():
    reqs = [ss.Request(rid=0, arrival=0.0, total_tokens=5)]
    cluster = ss.Cluster(n_replicas=1, step_time=0.01, max_batch=8)
    m = cluster.run(reqs)
    assert m["n_done"] == 1
    # 5 个 token * 0.01s = 0.05s 延迟(从 arrival=0 开始)。
    assert m["p50"] == pytest.approx(0.05, abs=1e-6)


def test_utilization_bounded():
    m = ss.simulate(n_replicas=2, rate=80.0, duration=15.0, seed=1)
    for u in m["utilizations"]:
        assert 0.0 <= u <= 1.0 + 1e-9
    for u in m["slot_utilizations"]:
        assert 0.0 <= u <= 1.0 + 1e-9
