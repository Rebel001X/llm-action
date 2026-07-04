# -*- coding: utf-8 -*-
"""
test_reliability.py —— 可靠性模式模拟器的单元 + 集成测试
=========================================================

本文件用 pytest 把"直觉"钉成"可验证的命题"，覆盖题目要求的四条核心：

    1. 断路器在故障率高时会打开（OPEN）。
    2. 重试提高成功率，但增加延迟。
    3. fallback 保底：底层全挂时用户视角仍然成功。
    4. 断路器状态机在各转移下行为正确（CLOSED→OPEN→HALF_OPEN→CLOSED）。

外加一批更细的单元测试（退避计算、超时、百分位、确定性复现）。

运行：  python -m pytest -q
"""

import random
import sys
import os

# 让测试无论从哪跑都能 import 到上一级的 reliability.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reliability as R  # noqa: E402


# ============================================================================
# A. FlakyDependency 基本行为
# ============================================================================

def test_dependency_never_fails_when_zero_fail_rate():
    """fail_rate=0 时，1000 次调用都不该抛错。"""
    dep = R.FlakyDependency(fail_rate=0.0, rng=random.Random(1))
    for _ in range(1000):
        latency = dep.call()
        assert latency > 0


def test_dependency_always_fails_when_fail_rate_one():
    """fail_rate=1 时，每次调用都必然抛 DependencyError。"""
    dep = R.FlakyDependency(fail_rate=1.0, rng=random.Random(1))
    for _ in range(50):
        try:
            dep.call()
            assert False, "本应抛 DependencyError"
        except R.DependencyError:
            pass


def test_dependency_is_deterministic():
    """同 seed → 同一串结果（确定性/可复现），这是全项目可测的地基。"""
    dep1 = R.FlakyDependency(fail_rate=0.3, rng=random.Random(123))
    dep2 = R.FlakyDependency(fail_rate=0.3, rng=random.Random(123))
    seq1, seq2 = [], []
    for _ in range(100):
        try:
            seq1.append(("ok", round(dep1.call(), 4)))
        except R.DependencyError:
            seq1.append(("fail", None))
        try:
            seq2.append(("ok", round(dep2.call(), 4)))
        except R.DependencyError:
            seq2.append(("fail", None))
    assert seq1 == seq2


# ============================================================================
# B. 退避计算（compute_backoff_ms）
# ============================================================================

def test_backoff_exp_grows():
    """指数退避（无抖动）应严格翻倍：100, 200, 400。"""
    cfg = R.RetryConfig(base_delay_ms=100, mode="exp", jitter=False)
    rng = random.Random(0)
    assert R.compute_backoff_ms(1, cfg, rng) == 100
    assert R.compute_backoff_ms(2, cfg, rng) == 200
    assert R.compute_backoff_ms(3, cfg, rng) == 400


def test_backoff_capped_respects_cap():
    """封顶退避不得超过 cap_ms。"""
    cfg = R.RetryConfig(base_delay_ms=500, mode="capped", cap_ms=1000, jitter=False)
    rng = random.Random(0)
    assert R.compute_backoff_ms(1, cfg, rng) == 500
    assert R.compute_backoff_ms(2, cfg, rng) == 1000   # 1000 而非 1000*2
    assert R.compute_backoff_ms(3, cfg, rng) == 1000   # 仍封顶
    assert R.compute_backoff_ms(9, cfg, rng) == 1000


def test_backoff_fixed_is_constant():
    """固定退避每次都一样。"""
    cfg = R.RetryConfig(base_delay_ms=250, mode="fixed", jitter=False)
    rng = random.Random(0)
    for a in range(1, 6):
        assert R.compute_backoff_ms(a, cfg, rng) == 250


def test_backoff_jitter_in_range():
    """带抖动时，结果落在 [0.5*base, 1.0*base] 内。"""
    cfg = R.RetryConfig(base_delay_ms=100, mode="fixed", jitter=True)
    rng = random.Random(7)
    for _ in range(200):
        v = R.compute_backoff_ms(1, cfg, rng)
        assert 50.0 <= v <= 100.0


# ============================================================================
# C. 断路器状态机（核心命题 4：状态机正确）
# ============================================================================

def test_breaker_starts_closed():
    """断路器初始态必须是 CLOSED（正常放行）。"""
    b = R.CircuitBreaker(R.BreakerConfig())
    assert b.state == R.CircuitState.CLOSED
    assert b.allow(now_ms=0) is True


def test_breaker_opens_after_threshold():
    """连续失败达阈值 → CLOSED 转 OPEN；OPEN 下 allow 返回 False（快速失败）。"""
    b = R.CircuitBreaker(R.BreakerConfig(fail_threshold=3, cooldown_ms=1000))
    b.on_failure(now_ms=0)
    b.on_failure(now_ms=0)
    assert b.state == R.CircuitState.CLOSED  # 还差一次
    b.on_failure(now_ms=0)
    assert b.state == R.CircuitState.OPEN     # 第 3 次跳闸
    assert b.allow(now_ms=0) is False         # 冷却期内快速失败


def test_breaker_success_resets_failure_count():
    """CLOSED 态里，一次成功就把连续失败计数清零（连续失败才跳闸）。"""
    b = R.CircuitBreaker(R.BreakerConfig(fail_threshold=3))
    b.on_failure(now_ms=0)
    b.on_failure(now_ms=0)
    b.on_success()               # 清零
    b.on_failure(now_ms=0)
    b.on_failure(now_ms=0)
    assert b.state == R.CircuitState.CLOSED   # 因为被打断，没连续 3 次


def test_breaker_half_open_after_cooldown():
    """OPEN 冷却到期后，allow 会切到 HALF_OPEN 并放行探针。"""
    b = R.CircuitBreaker(R.BreakerConfig(fail_threshold=1, cooldown_ms=1000))
    b.on_failure(now_ms=0)                     # 跳闸于 t=0
    assert b.state == R.CircuitState.OPEN
    assert b.allow(now_ms=500) is False        # 冷却未到
    assert b.allow(now_ms=1000) is True        # 到期 → 放行探针
    assert b.state == R.CircuitState.HALF_OPEN


def test_breaker_half_open_success_recovers():
    """半开态里连续成功够 probes → 完全恢复 CLOSED。"""
    b = R.CircuitBreaker(R.BreakerConfig(fail_threshold=1, cooldown_ms=1000,
                                         half_open_probes=2))
    b.on_failure(now_ms=0)
    b.allow(now_ms=1000)                       # → HALF_OPEN
    b.on_success()                             # 1/2
    assert b.state == R.CircuitState.HALF_OPEN
    b.on_success()                             # 2/2 → 恢复
    assert b.state == R.CircuitState.CLOSED


def test_breaker_half_open_failure_reopens():
    """半开态里探针一失败 → 立刻重新 OPEN，冷却重启。"""
    b = R.CircuitBreaker(R.BreakerConfig(fail_threshold=1, cooldown_ms=1000))
    b.on_failure(now_ms=0)
    b.allow(now_ms=1000)                       # → HALF_OPEN
    b.on_failure(now_ms=1000)                  # 探针失败
    assert b.state == R.CircuitState.OPEN
    assert b.allow(now_ms=1500) is False       # 冷却又从 t=1000 重启，1500 还没到


# ============================================================================
# D. 集成命题 1：故障率高 → 断路器一定会打开
# ============================================================================

def test_circuit_opens_under_high_failure_rate():
    """故障率 90% 时，跑一批请求后断路器必然进入过 OPEN（有请求被快速失败）。"""
    factory = R.make_factory(
        fail_rate=0.9, seed=42,
        enable_breaker=True, enable_fallback=True,
        breaker_cfg=R.BreakerConfig(fail_threshold=3, cooldown_ms=5000),
    )
    m = R.simulate(factory, n_requests=200)
    # n_opened > 0 表示确实有请求在 OPEN 态被快速失败/直接兜底
    assert m.n_opened > 0, "高故障率下断路器竟然从未打开，逻辑有误"


def test_circuit_stays_closed_when_healthy():
    """依赖健康（故障率 0）时，断路器不该打开，所有请求都由 primary 服务。"""
    factory = R.make_factory(
        fail_rate=0.0, seed=42,
        enable_breaker=True, enable_fallback=True,
        breaker_cfg=R.BreakerConfig(fail_threshold=3),
    )
    m = R.simulate(factory, n_requests=200)
    assert m.n_opened == 0
    assert m.served_by_primary == 200


# ============================================================================
# E. 集成命题 2：重试提高成功率，但增加延迟
# ============================================================================

def test_retry_improves_success_rate():
    """中等故障率下，开重试的成功率应显著高于不开重试。"""
    common = dict(fail_rate=0.4, seed=7)

    no_retry = R.make_factory(**common, enable_retry=False)
    with_retry = R.make_factory(
        **common, enable_retry=True,
        retry_cfg=R.RetryConfig(max_attempts=4, mode="capped", jitter=False),
    )
    m0 = R.simulate(no_retry, 500)
    m1 = R.simulate(with_retry, 500)
    assert m1.availability > m0.availability, (
        f"重试没提高成功率：{m0.availability:.3f} -> {m1.availability:.3f}"
    )


def test_retry_increases_latency_and_attempts():
    """重试提高成功率的代价：平均延迟上升、平均尝试次数上升。"""
    common = dict(fail_rate=0.4, seed=7)
    no_retry = R.make_factory(**common, enable_retry=False)
    with_retry = R.make_factory(
        **common, enable_retry=True,
        retry_cfg=R.RetryConfig(max_attempts=4, base_delay_ms=100,
                                mode="capped", jitter=False),
    )
    m0 = R.simulate(no_retry, 500)
    m1 = R.simulate(with_retry, 500)
    assert m1.avg_attempts > m0.avg_attempts   # 更多尝试
    assert m1.avg_latency > m0.avg_latency     # 更高延迟（含退避等待）


# ============================================================================
# F. 集成命题 3：fallback 保底
# ============================================================================

def test_fallback_guarantees_success_even_when_primary_dead():
    """依赖 100% 挂（fail_rate=1）时，只要开了 fallback，用户视角可用性=100%。"""
    factory = R.make_factory(
        fail_rate=1.0, seed=42,
        enable_fallback=True,
    )
    m = R.simulate(factory, 300)
    assert m.availability == 1.0, "fallback 未能保底"
    assert m.served_by_fallback == 300
    assert m.served_by_primary == 0


def test_no_fallback_means_total_failure():
    """依赖 100% 挂且不开 fallback、不开重试 → 可用性=0（对照组）。"""
    factory = R.make_factory(fail_rate=1.0, seed=42)
    m = R.simulate(factory, 300)
    assert m.availability == 0.0


def test_fallback_is_fast():
    """fallback 兜底应该很快（缓存响应），P99 远低于'死等超时'的坏情况。"""
    factory = R.make_factory(
        fail_rate=1.0, seed=42,
        enable_fallback=True,
        fallback=lambda: 5.0,   # 5ms 的缓存响应
    )
    m = R.simulate(factory, 300)
    # 全部 fallback，延迟应该在个位数 ms 量级（primary 失败成本 + 5ms 兜底）
    assert m.p99_latency < 100.0


# ============================================================================
# G. 超时
# ============================================================================

def test_timeout_turns_slow_into_failure():
    """开超时后，慢请求(500ms) 超过 200ms 阈值会被判失败；不开超时则算成功。"""
    # slow_rate=1 表示每次都慢；fail_rate=0 表示下游本身不返 5xx
    off = R.make_factory(fail_rate=0.0, slow_rate=1.0, seed=3,
                         enable_timeout=False, slow_ms=500)
    on = R.make_factory(fail_rate=0.0, slow_rate=1.0, seed=3,
                        enable_timeout=True, timeout_ms=200, slow_ms=500,
                        enable_fallback=False)
    m_off = R.simulate(off, 200)
    m_on = R.simulate(on, 200)
    # 不开超时：慢但成功 → 可用性高
    assert m_off.availability == 1.0
    # 开超时且不兜底：慢=失败 → 可用性显著下降
    assert m_on.availability < 0.5


def test_timeout_with_fallback_recovers():
    """超时 + fallback：慢请求被超时判失败，但 fallback 兜底 → 可用性恢复到 100%。"""
    on = R.make_factory(fail_rate=0.0, slow_rate=1.0, seed=3,
                        enable_timeout=True, timeout_ms=200, slow_ms=500,
                        enable_fallback=True)
    m = R.simulate(on, 200)
    assert m.availability == 1.0


# ============================================================================
# H. 指标工具
# ============================================================================

def test_percentile_basic():
    """P100=最大值，P1≈最小值附近，单调不减。"""
    data = list(range(1, 101))   # 1..100
    assert R.percentile(data, 100) == 100
    assert R.percentile(data, 50) == 50
    assert R.percentile(data, 99) == 99
    assert R.percentile([], 99) == 0.0


def test_metrics_availability_math():
    """可用性 = 成功/总数 的算术正确。"""
    m = R.Metrics()
    m.n = 10
    m.n_success = 7
    assert abs(m.availability - 0.7) < 1e-9


# ============================================================================
# I. 组合拳：四模式全开 应当是"最稳"的
# ============================================================================

def test_all_patterns_together_maximize_availability():
    """四模式全开时，在高故障 + 高慢速下，用户视角可用性仍应=100%（靠 fallback 兜底）。"""
    factory = R.make_factory(
        fail_rate=0.7, slow_rate=0.3, seed=99,
        enable_retry=True, enable_timeout=True,
        enable_breaker=True, enable_fallback=True,
        retry_cfg=R.RetryConfig(max_attempts=3, mode="capped"),
        breaker_cfg=R.BreakerConfig(fail_threshold=5, cooldown_ms=3000),
        timeout_ms=200,
    )
    m = R.simulate(factory, 500)
    assert m.availability == 1.0
