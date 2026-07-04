# -*- coding: utf-8 -*-
"""
tests/test_autoscale.py —— 自动扩缩容仿真的行为测试
====================================================

测试策略:不测"某个数字恰好等于几",而是测**控制器的定性行为契约**
(behavioral contract),这更贴近工程实践,也更抗随机噪声:

    1) 高负载 → 必须扩容(scale-up):供不应求时副本数要涨上去
    2) 低负载 → 必须缩容(scale-down):供大于求时副本数要降下来
    3) 冷却防抖(cooldown / anti-flapping):加了缩容冷却后,抖动次数不增反降
    4) 成本-SLO 权衡(cost-SLO trade-off):目标利用率越低 → SLO 越好但越贵
    5) 若干不变量(invariants):副本永远在 [min, max]、启动延迟真的生效等

运行:  python -m pytest -q
"""

import os
import sys

import numpy as np
import pytest

# 让测试能 import 上级目录的 autoscale_sim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoscale_sim import (  # noqa: E402
    SimConfig, Fleet, Queue, Autoscaler,
    run_sim, compute_metrics, simulate,
    trapezoid_load, spiky_load, diurnal_load,
)


# =============================================================================
# 1. 高负载 → 扩容
# =============================================================================
def test_high_load_scales_up():
    """持续高负载下,副本数必须从初始值显著上涨。"""
    cfg = SimConfig(
        horizon=200.0,
        init_replicas=2,
        arrival_fn=lambda t: 200.0,     # 恒定高负载 200 rps,每 Pod 只有 10 rps
        poisson_arrivals=False,
    )
    res = run_sim(cfg)
    peak_ready = res["ready_pods"].max()
    # 200 rps / 10 rps每Pod = 至少需要 20 个 Pod
    assert peak_ready >= 18, f"高负载下峰值副本 {peak_ready} 太小,没扩起来"
    # 结束时的副本数应远大于初始
    assert res["ready_pods"][-1] > cfg.init_replicas * 3


def test_high_load_meets_capacity():
    """扩容后,稳态就绪容量应能覆盖到达率(积压最终被压下去)。"""
    cfg = SimConfig(
        horizon=300.0,
        init_replicas=2,
        arrival_fn=lambda t: 150.0,
        poisson_arrivals=False,
    )
    res = run_sim(cfg)
    # 看仿真后半段的平均积压:应该接近 0(容量跟上了)
    tail_backlog = res["backlog"][-50:].mean()
    assert tail_backlog < 50.0, f"高负载稳态积压 {tail_backlog} 仍很大,扩容没跟上"


# =============================================================================
# 2. 低负载 → 缩容
# =============================================================================
def test_low_load_scales_down():
    """先高后低:高负载把副本顶上去,随后低负载应把副本缩回来。"""
    def load(t):
        return 200.0 if t < 100.0 else 10.0   # 前100s高,后面低

    cfg = SimConfig(
        horizon=600.0,
        init_replicas=2,
        arrival_fn=load,
        poisson_arrivals=False,
        scale_down_cooldown=30.0,   # 缩容别太慢,好在仿真时长内看到效果
        max_scale_down_step=8,
    )
    res = run_sim(cfg)
    peak_ready = res["ready_pods"].max()
    final_ready = res["ready_pods"][-1]
    assert peak_ready >= 15, "高负载段没扩起来,无法验证缩容"
    assert final_ready < peak_ready * 0.5, \
        f"低负载段副本 {final_ready} 未明显低于峰值 {peak_ready},没缩下来"


def test_never_below_min_replicas():
    """无论负载多低,副本数永远不低于 min_replicas。"""
    cfg = SimConfig(
        horizon=400.0,
        min_replicas=3,
        init_replicas=10,
        arrival_fn=lambda t: 1.0,      # 几乎没负载
        poisson_arrivals=False,
        scale_down_cooldown=10.0,
    )
    res = run_sim(cfg)
    assert res["total_pods"].min() >= 3, "副本数跌破了 min_replicas"
    assert res["ready_pods"][-1] <= 6, "极低负载下最终副本应缩到接近 min"


# =============================================================================
# 3. 冷却防抖(anti-flapping)
# =============================================================================
def test_cooldown_reduces_flapping():
    """加长缩容冷却窗口后,抖动次数应不增(通常显著下降)。

    构造一条"锯齿"负载(高低快速交替),没有冷却时控制器会疯狂扩缩;
    加上冷却后,缩容被"按住",flapping 明显减少。
    """
    def sawtooth(t):
        # 每 20 秒在高低之间切换
        return 200.0 if int(t // 20) % 2 == 0 else 20.0

    common = dict(horizon=400.0, init_replicas=5, arrival_fn=sawtooth,
                  poisson_arrivals=False, tolerance=0.05)

    cfg_no_cd = SimConfig(scale_down_cooldown=0.0, scale_up_cooldown=0.0, **common)
    cfg_cd = SimConfig(scale_down_cooldown=120.0, scale_up_cooldown=0.0, **common)

    m_no_cd = compute_metrics(run_sim(cfg_no_cd), cfg_no_cd)
    m_cd = compute_metrics(run_sim(cfg_cd), cfg_cd)

    assert m_cd["flaps"] <= m_no_cd["flaps"], \
        f"加冷却后抖动 {m_cd['flaps']} 反而多于无冷却 {m_no_cd['flaps']}"


def test_tolerance_band_suppresses_micro_scaling():
    """容差带内的小偏差不应触发扩缩(指标恰在目标附近时副本稳定)。"""
    cfg = SimConfig(
        horizon=200.0,
        metric="queue",
        target_metric=5.0,
        tolerance=0.30,           # 宽容差带
        init_replicas=4,
        # 让每 Pod 队列长度稳定在目标附近:4 Pod、40 rps、cap10 → 恰好打平
        arrival_fn=lambda t: 40.0,
        poisson_arrivals=False,
    )
    res = run_sim(cfg)
    # 稳态后副本数波动应很小
    tail = res["ready_pods"][-80:]
    assert tail.std() <= 3.0, f"宽容差带下副本仍抖动明显(std={tail.std():.2f})"


# =============================================================================
# 4. 成本-SLO 权衡
# =============================================================================
def test_cost_slo_tradeoff():
    """目标利用率越激进(target 越大 → 每 Pod 扛得越满)→ 越省钱,但 SLO 越差。

    这是自动扩缩容最核心的工程权衡,必须能在仿真里复现出来。
    """
    base = dict(horizon=500.0, init_replicas=2, arrival_fn=trapezoid_load,
                rng_seed=7, metric="queue")

    # 保守:目标队列很短(每 Pod 只排 2 个)→ 副本多、贵、SLO 好
    cfg_conservative = SimConfig(target_metric=2.0, **base)
    # 激进:目标队列长(每 Pod 排 15 个)→ 副本少、便宜、SLO 差
    cfg_aggressive = SimConfig(target_metric=15.0, **base)

    m_cons = compute_metrics(run_sim(cfg_conservative), cfg_conservative)
    m_aggr = compute_metrics(run_sim(cfg_aggressive), cfg_aggressive)

    # 保守方案更贵
    assert m_cons["cost"] > m_aggr["cost"], \
        f"保守方案成本 {m_cons['cost']:.2f} 应高于激进 {m_aggr['cost']:.2f}"
    # 保守方案 SLO 不差于激进(通常更好)
    assert m_cons["slo_ok_ratio"] >= m_aggr["slo_ok_ratio"] - 1e-9, \
        f"保守方案 SLO {m_cons['slo_ok_ratio']:.3f} 反不如激进 {m_aggr['slo_ok_ratio']:.3f}"


def test_more_pods_lower_latency():
    """同样负载下,目标越保守(副本越多)→ 平均延迟越低。"""
    base = dict(horizon=300.0, init_replicas=2,
                arrival_fn=lambda t: 120.0, poisson_arrivals=False, metric="queue")
    m_lo = compute_metrics(run_sim(SimConfig(target_metric=2.0, **base)),
                           SimConfig(target_metric=2.0, **base))
    m_hi = compute_metrics(run_sim(SimConfig(target_metric=20.0, **base)),
                           SimConfig(target_metric=20.0, **base))
    assert m_lo["p95_latency"] <= m_hi["p95_latency"] + 1e-6


# =============================================================================
# 5. 启动延迟 / 机群 / 队列 的单元级测试
# =============================================================================
def test_pod_startup_delay_is_real():
    """新扩容的 Pod 不是立刻就绪:在 startup_delay 内 ready 不该增加。"""
    fleet = Fleet(ready=2, startup_delay=10.0)
    fleet.scale_to(10)                 # 想要 10 个
    assert fleet.request_ready() == 2  # 但立刻仍只有 2 个就绪
    assert fleet.total() == 10         # 总数(含启动中)是 10
    # 推进 5 秒,还没到 10 秒,仍不就绪
    fleet.tick(5.0)
    assert fleet.request_ready() == 2
    # 再推进 5 秒(累计 10s),启动完成
    fleet.tick(5.0)
    assert fleet.request_ready() == 10


def test_scale_down_prefers_warming_pods():
    """缩容时优先砍掉'启动中'的 Pod(还没干活,砍了不心疼)。"""
    fleet = Fleet(ready=5, startup_delay=10.0)
    fleet.scale_to(12)                 # +7 进入 warming
    assert fleet.request_ready() == 5
    fleet.scale_to(6)                  # 缩到 6:应先砍 warming(7个),ready 保 5→只砍到6
    assert fleet.request_ready() == 5  # ready 未被动(优先砍 warming)
    assert fleet.total() == 6


def test_queue_drains_with_enough_capacity():
    """容量充足时,队列应被清空。"""
    q = Queue()
    obs = q.step(arrivals=5.0, ready_pods=10, per_pod_capacity=10.0, dt=1.0)
    # 到达 5,容量 100,应全部处理完
    assert obs["backlog"] == 0.0
    assert obs["served"] == 5.0
    assert obs["latency"] == 0.0


def test_queue_builds_when_underprovisioned():
    """容量不足时,队列应积压且延迟上升。"""
    q = Queue()
    obs = q.step(arrivals=100.0, ready_pods=1, per_pod_capacity=10.0, dt=1.0)
    assert obs["backlog"] == 90.0      # 到达100,处理10,剩90
    assert obs["latency"] > 0.0
    assert obs["q_per_pod"] == 90.0    # 每 Pod 队列 = 90/1


# =============================================================================
# 6. 全局不变量(invariants)
# =============================================================================
@pytest.mark.parametrize("arrival_fn", [trapezoid_load, spiky_load, diurnal_load])
def test_replicas_within_bounds(arrival_fn):
    """任何负载下,总副本数恒在 [min_replicas, max_replicas] 内。"""
    cfg = SimConfig(horizon=600.0, arrival_fn=arrival_fn,
                    min_replicas=2, max_replicas=50)
    res = run_sim(cfg)
    assert res["total_pods"].min() >= 2
    assert res["total_pods"].max() <= 50


def test_metric_gpu_and_latency_modes_run():
    """三种触发指标(queue/latency/gpu)都能跑通并给出合理结果。"""
    for metric, target in [("queue", 5.0), ("latency", 0.5), ("gpu", 0.7)]:
        cfg = SimConfig(horizon=300.0, metric=metric, target_metric=target,
                        arrival_fn=trapezoid_load)
        out = simulate(cfg)
        m = out["metrics"]
        assert 0.0 <= m["slo_ok_ratio"] <= 1.0
        assert m["cost"] > 0.0
        assert m["throughput"] > 0.0


def test_determinism_with_seed():
    """同一随机种子 → 结果完全可复现。"""
    cfg1 = SimConfig(horizon=200.0, rng_seed=123)
    cfg2 = SimConfig(horizon=200.0, rng_seed=123)
    r1, r2 = run_sim(cfg1), run_sim(cfg2)
    assert np.array_equal(r1["backlog"], r2["backlog"])
    assert np.array_equal(r1["ready_pods"], r2["ready_pods"])


def test_slo_ratio_is_fraction():
    """SLO 满足率必须是 [0,1] 的比例。"""
    out = simulate(SimConfig(horizon=400.0))
    assert 0.0 <= out["metrics"]["slo_ok_ratio"] <= 1.0


def test_zero_load_costs_min():
    """零负载时,成本应对应 min_replicas 附近(不会莫名扩容烧钱)。"""
    cfg = SimConfig(horizon=300.0, min_replicas=2, init_replicas=2,
                    arrival_fn=lambda t: 0.0, poisson_arrivals=False)
    res = run_sim(cfg)
    assert res["total_pods"].max() <= 3, "零负载竟然扩容了,说明控制逻辑有 bug"
