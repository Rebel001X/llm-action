# -*- coding: utf-8 -*-
"""
reliability.py —— 可靠性模式模拟器（核心库）
================================================

配套《Systems Design in the LLM Era》第 2 章"断路器 + 分层降级"一节。

本模块把书里散落的四种可靠性模式做成**可运行、可测量、可复现**的代码：

    1. 重试 + 退避     Retry with Backoff
    2. 超时           Timeout
    3. 断路器         Circuit Breaker（三态状态机：CLOSED / OPEN / HALF_OPEN）
    4. 降级           Fallback（分层降级 Tiered Fallback）

设计目标（第一性原理）：
    - **确定性**：所有随机性都走一个显式的 random.Random(seed)，同 seed 必得同结果，
      这样 pytest 才能断言"断路器在故障率高时一定会打开"这类命题。
    - **零依赖**：只用 Python 标准库（random / time / enum / dataclasses），
      本机离线可跑，不联网、不下模型、不需要 key。
    - **可测量**：每次调用都返回一个 CallResult，run_demo.py 据此统计
      可用性 / 成功率 / 平均延迟 / P99 尾延迟。

术语中英并列，注释全中文，力求零基础也能逐行读懂。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional


# ============================================================================
# 0. 异常与数据结构
# ============================================================================

class DependencyError(Exception):
    """依赖返回的"业务/服务器错误"（对应真实世界的 HTTP 5xx）。

    为什么要单独一个异常类？——因为断路器/重试**只应该对"可重试的故障"生效**。
    如果什么异常都拦，就会把"用户输入非法(4xx)"这种不该重试的错误也重试，属于坑。
    这里为了教学，把所有依赖故障统一成 DependencyError，真实项目里你会区分
    可重试(5xx / 超时 / 限流 429) vs 不可重试(4xx 参数错误)。
    """


class TimeoutError_(Exception):
    """超时错误。

    注意不要直接用内置 TimeoutError 覆盖语义会混乱，这里用带下划线的名字，
    表示"我们模拟器自己定义的超时"。它也是一种"可重试故障"。
    """


class CircuitOpenError(Exception):
    """断路器处于 OPEN（打开/跳闸）状态时，直接快速失败抛出的异常。

    这是断路器的**核心价值**：当下游已经明显不健康时，请求**根本不发出去**，
    立刻失败（fail-fast），既不拖死自己（不占连接、不等超时），
    也不雪上加霜地打爆对方（stop hammering a dying service）。
    """


@dataclass
class CallResult:
    """一次"端到端调用"的结果记录。run_demo 用它来算各种指标。

    字段含义：
        success   : 最终是否成功（True/False）。注意——即使底层依赖失败，
                    只要 fallback 兜底成功，从"用户视角"看也是 success=True。
        latency_ms: 端到端耗时（毫秒），把重试等待、退避、超时墙都算进去。
        served_by : 最终由谁服务的（"primary" / "fallback" / "cache" / None）。
                    用来观察"有多少请求是靠降级保住的"。
        attempts  : 一共发起了几次底层尝试（含重试）。用来观察"重试放大"效应。
        opened    : 本次调用时断路器是否处于 OPEN（被快速失败）。
    """
    success: bool
    latency_ms: float
    served_by: Optional[str] = None
    attempts: int = 0
    opened: bool = False


# ============================================================================
# 1. FlakyDependency —— 会"抽风"的下游依赖（被测对象）
# ============================================================================

class FlakyDependency:
    """一个有故障率 + 有延迟的"外部依赖"（模拟 OpenAI / 数据库 / 微服务）。

    真实世界里，下游会出现三种"坏"：
        1. 返回错误（5xx）——概率 = fail_rate
        2. 变慢（长尾延迟）——概率 = slow_rate，慢到 slow_ms
        3. 正常但也有基础延迟——base_ms 上下浮动

    我们用一个可注入的 random.Random 保证**可复现**。
    """

    def __init__(
        self,
        fail_rate: float = 0.0,
        base_ms: float = 20.0,
        slow_rate: float = 0.0,
        slow_ms: float = 500.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        # fail_rate：每次调用有多大概率直接抛 DependencyError（0~1）
        self.fail_rate = fail_rate
        # base_ms：正常情况下的基础延迟（毫秒）
        self.base_ms = base_ms
        # slow_rate：多大概率触发"长尾慢请求"
        self.slow_rate = slow_rate
        # slow_ms：慢的时候有多慢
        self.slow_ms = slow_ms
        # rng：随机源。不传就自己 new 一个（不可复现）；测试时一定要传固定 seed 的。
        self.rng = rng or random.Random()

    def call(self) -> float:
        """发起一次调用。

        返回：本次调用的"真实耗时"（毫秒）。
        抛出：DependencyError —— 表示下游返回了 5xx。

        注意这里**不真正 sleep**！我们返回一个"虚拟耗时"数字，
        由上层的 Timeout 逻辑去判断"这个耗时是否超过超时阈值"。
        这样 1 万次仿真在几毫秒内就跑完，而不是真的等几秒——
        这是模拟器的常见做法：用"逻辑时间"代替"物理时间"。
        """
        # 第一步：先决定这次是否"变慢"
        if self.rng.random() < self.slow_rate:
            # 慢请求：耗时在 slow_ms 上下 ±20% 浮动
            latency = self.slow_ms * self.rng.uniform(0.8, 1.2)
        else:
            # 正常请求：基础延迟上下 ±30% 浮动
            latency = self.base_ms * self.rng.uniform(0.7, 1.3)

        # 第二步：再决定这次是否"失败"（返回 5xx）
        # 注意：失败也是有耗时的——错误响应也要等下游处理完才返回，
        # 所以我们让它已经"花了"latency 才失败（这点很多人会忽略）。
        if self.rng.random() < self.fail_rate:
            raise DependencyError(f"依赖返回 5xx（耗时 {latency:.1f}ms）")

        return latency


# ============================================================================
# 2. Retry —— 重试 + 退避（Retry with Backoff）
# ============================================================================

@dataclass
class RetryConfig:
    """重试配置。

    max_attempts : 最多尝试几次（含首次）。=1 表示不重试。
    base_delay_ms: 退避基准延迟。
    mode         : "capped" 封顶退避 / "exp" 指数退避 / "fixed" 固定间隔。
    cap_ms       : 封顶退避的上限（书里交互式场景推荐"起步 500ms、最多 1s"）。
    jitter       : 是否加抖动（jitter），避免"惊群/重试风暴"同时打下游。
    """
    max_attempts: int = 3
    base_delay_ms: float = 100.0
    mode: str = "capped"   # capped | exp | fixed
    cap_ms: float = 1000.0
    jitter: bool = True


def compute_backoff_ms(attempt: int, cfg: RetryConfig, rng: random.Random) -> float:
    """计算第 attempt 次重试前应该等待多少毫秒。

    参数 attempt 从 1 开始（第 1 次重试 = 第一次失败之后）。

    三种退避策略（这是面试高频，务必理解差异）：

        fixed（固定）  : 每次都等 base_delay_ms。简单，但可能"同步重试风暴"。
        exp（指数）    : base * 2^(attempt-1)。等待翻倍增长——适合**异步**场景
                        （"没人在等，可以耗着，扛过 5 分钟供应商故障"）。
        capped（封顶） : 指数增长但**封顶**在 cap_ms。适合**交互式/同步**场景
                        （"用户在等！一个 60 秒的重试等于宕机"）。

    jitter（抖动）：在算出的等待值上乘以 [0.5, 1.0] 的随机因子。
        为什么要抖动？——如果 1000 个客户端同时在 t=0 失败，都精确等 200ms，
        它们会在 t=200ms **再次同时**打下游，形成"重试风暴/惊群(thundering herd)"。
        加抖动把它们打散到 [100ms, 200ms] 区间，削平尖峰。这是生产级退避的必备项。
    """
    if cfg.mode == "fixed":
        delay = cfg.base_delay_ms
    elif cfg.mode == "exp":
        # 指数：100, 200, 400, 800, ...
        delay = cfg.base_delay_ms * (2 ** (attempt - 1))
    elif cfg.mode == "capped":
        # 先指数增长，再用 min 封顶
        delay = min(cfg.base_delay_ms * (2 ** (attempt - 1)), cfg.cap_ms)
    else:
        raise ValueError(f"未知退避模式：{cfg.mode}")

    if cfg.jitter:
        # 乘性抖动（"equal jitter"的简化版）：结果落在 [0.5*delay, 1.0*delay]
        delay = delay * rng.uniform(0.5, 1.0)

    return delay


# ============================================================================
# 3. CircuitBreaker —— 断路器（三态状态机）
# ============================================================================

class CircuitState(Enum):
    """断路器的三种状态。

    CLOSED（闭合）   : 正常放行流量。像电路闭合——电流（请求）能通过。
                      （注意这个命名反直觉：CLOSED = 正常工作，OPEN = 断开/故障）
    OPEN（打开/跳闸）: 熔断中。所有请求**直接快速失败**，不发给下游。
    HALF_OPEN（半开）: 冷却期结束后的"试探态"。放行少量 canary（金丝雀）请求，
                      成功就恢复(→CLOSED)，失败就重新跳闸(→OPEN)。
    """
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class BreakerConfig:
    """断路器配置。

    fail_threshold  : 连续失败多少次就跳闸（CLOSED → OPEN）。
    cooldown_ms     : 跳闸后冷却多久，才允许进入 HALF_OPEN 试探。
    half_open_probes: 半开态需要连续成功多少个 canary 才恢复到 CLOSED。
    """
    fail_threshold: int = 5
    cooldown_ms: float = 2000.0
    half_open_probes: int = 2


class CircuitBreaker:
    """断路器实现。

    它是一个**状态机 + 计数器**。核心不变量（invariant）：
        - CLOSED 下累计连续失败达 fail_threshold → 跳到 OPEN，并记录跳闸时刻。
        - OPEN 下，若"当前逻辑时间 - 跳闸时刻 >= cooldown_ms" → 允许进入 HALF_OPEN。
        - HALF_OPEN 下，连续成功 half_open_probes 次 → 回到 CLOSED（完全恢复）；
                       只要有一次失败 → 立刻回到 OPEN（重启冷却）。

    我们用**逻辑时钟** now_ms（由调用方注入当前时间），而不是真实 time.time()，
    这样测试可以精确控制"过了多久"，无需真的 sleep。
    """

    def __init__(self, cfg: BreakerConfig) -> None:
        self.cfg = cfg
        self.state = CircuitState.CLOSED
        # 连续失败计数（CLOSED 态用）
        self._consecutive_failures = 0
        # 半开态里连续成功的 canary 数
        self._consecutive_probe_successes = 0
        # 跳闸发生的逻辑时刻（毫秒）；None 表示从未跳闸
        self._opened_at_ms: Optional[float] = None

    # ---- 查询：现在允许放行请求吗？----
    def allow(self, now_ms: float) -> bool:
        """在 now_ms 这个时刻，断路器是否允许放行一个请求。

        返回 True  → 上层可以真正调用下游。
        返回 False → 上层应当直接快速失败（抛 CircuitOpenError）。

        关键逻辑：OPEN 态下会检查冷却是否到期，到期就**顺手切到 HALF_OPEN** 并放行
        （这次放行的就是 canary 探针）。
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # 冷却到期了吗？
            assert self._opened_at_ms is not None
            if now_ms - self._opened_at_ms >= self.cfg.cooldown_ms:
                # 到期 → 进入半开，放行一个探针
                self.state = CircuitState.HALF_OPEN
                self._consecutive_probe_successes = 0
                return True
            # 还在冷却中 → 快速失败
            return False

        # HALF_OPEN：放行探针（真实系统会限流只放少量，这里教学简化为放行）
        return True

    # ---- 反馈：这次调用成功了 ----
    def on_success(self) -> None:
        """下游调用成功后，喂给断路器的反馈。"""
        if self.state == CircuitState.HALF_OPEN:
            # 半开态：累计成功探针
            self._consecutive_probe_successes += 1
            if self._consecutive_probe_successes >= self.cfg.half_open_probes:
                # 探针够了 → 完全恢复
                self._close()
        else:
            # CLOSED 态：一次成功就把连续失败计数清零
            self._consecutive_failures = 0

    # ---- 反馈：这次调用失败了 ----
    def on_failure(self, now_ms: float) -> None:
        """下游调用失败后，喂给断路器的反馈。"""
        if self.state == CircuitState.HALF_OPEN:
            # 半开态里探针失败 → 立刻重新跳闸，重启冷却
            self._open(now_ms)
            return

        # CLOSED 态：累加连续失败
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.cfg.fail_threshold:
            self._open(now_ms)

    # ---- 内部：跳闸 ----
    def _open(self, now_ms: float) -> None:
        self.state = CircuitState.OPEN
        self._opened_at_ms = now_ms
        self._consecutive_probe_successes = 0

    # ---- 内部：恢复闭合 ----
    def _close(self) -> None:
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_probe_successes = 0
        self._opened_at_ms = None


# ============================================================================
# 4. ReliableClient —— 把四种模式组装成一个"可靠客户端"
# ============================================================================

class ReliableClient:
    """把 超时 + 重试 + 断路器 + 降级 组合到一起的客户端。

    这就是书里"网关(Gateway)"里那段可靠性逻辑的**最小可运行版**。
    一次 client.call() 的完整决策链（也是本项目最值得记的一张流程）：

        1. 问断路器 allow(now)？—— 不允许 → 直接走 fallback（或彻底失败）。
        2. 允许 → 进入"带重试的调用循环"：
            a. 调 dependency.call() 拿到耗时 latency。
            b. 若 latency > timeout_ms → 判超时（TimeoutError_），算失败。
            c. 若抛 DependencyError → 算失败。
            d. 失败 → 反馈断路器 on_failure；若还有重试次数，退避后再来一次。
            e. 成功 → 反馈断路器 on_success，返回成功。
        3. 重试全部用尽仍失败 → 走 fallback 兜底（若配置了）。

    通过开关（enable_*）可以单独打开/关闭每种模式，
    run_demo.py 正是靠这些开关做"各模式对比图"。
    """

    def __init__(
        self,
        dependency: FlakyDependency,
        rng: random.Random,
        retry_cfg: Optional[RetryConfig] = None,
        breaker_cfg: Optional[BreakerConfig] = None,
        timeout_ms: float = 200.0,
        fallback: Optional[Callable[[], float]] = None,
        enable_retry: bool = False,
        enable_timeout: bool = False,
        enable_breaker: bool = False,
        enable_fallback: bool = False,
    ) -> None:
        self.dep = dependency
        self.rng = rng
        self.retry_cfg = retry_cfg or RetryConfig()
        self.timeout_ms = timeout_ms
        # fallback 是一个"备用服务"：无参、返回耗时(ms)、不会失败（Tier 4 兜底）。
        # 默认给一个极快的"缓存响应"兜底。
        self.fallback = fallback or (lambda: 5.0)

        self.enable_retry = enable_retry
        self.enable_timeout = enable_timeout
        self.enable_breaker = enable_breaker
        self.enable_fallback = enable_fallback

        # 断路器（即便没启用也建一个，逻辑里用 enable_breaker 门控）
        self.breaker = CircuitBreaker(breaker_cfg or BreakerConfig())

        # 逻辑时钟：每次 call 结束后，把这次的耗时累加进去，模拟"时间流逝"。
        # 这样断路器的冷却计时才有意义（否则 now_ms 永远是 0，冷却永远不到期）。
        self._clock_ms = 0.0

    # ------------------------------------------------------------------
    def _try_once(self) -> float:
        """真正发一次请求，处理超时。成功返回耗时；失败抛异常。"""
        latency = self.dep.call()  # 可能抛 DependencyError
        if self.enable_timeout and latency > self.timeout_ms:
            # 超时：即使下游最终会返回，我们也已经不等了。
            # 注意超时也是"花了 timeout_ms 才放弃"，不是 0 成本。
            raise TimeoutError_(f"超时：{latency:.1f}ms > {self.timeout_ms}ms")
        return latency

    # ------------------------------------------------------------------
    def call(self) -> CallResult:
        """发起一次端到端调用，返回 CallResult。"""
        total_latency = 0.0
        attempts = 0

        # ---- 阶段 1：断路器闸门 ----
        if self.enable_breaker and not self.breaker.allow(self._clock_ms):
            # 断路器 OPEN 且冷却未到 → 快速失败，直接兜底
            if self.enable_fallback:
                fb_latency = self.fallback()
                total_latency += fb_latency
                self._advance_clock(total_latency)
                return CallResult(True, total_latency, served_by="fallback",
                                  attempts=0, opened=True)
            # 没兜底 → 彻底失败（但快速，几乎 0 延迟）
            self._advance_clock(total_latency)
            return CallResult(False, total_latency, served_by=None,
                              attempts=0, opened=True)

        # ---- 阶段 2：带重试的调用循环 ----
        # 若关闭重试，max_attempts 强制为 1（只试一次）。
        max_attempts = self.retry_cfg.max_attempts if self.enable_retry else 1

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                latency = self._try_once()
                total_latency += latency
                # 成功：喂断路器
                if self.enable_breaker:
                    self.breaker.on_success()
                self._advance_clock(total_latency)
                return CallResult(True, total_latency, served_by="primary",
                                  attempts=attempts, opened=False)

            except (DependencyError, TimeoutError_):
                # 失败：无论 5xx 还是超时，都要
                #   (1) 把"这次失败也花了的时间"累加进延迟；
                #   (2) 喂断路器 on_failure；
                #   (3) 若还有重试机会，退避等待后再来。
                # 失败耗时估算：超时按 timeout_ms 算，5xx 按 base 附近估算。
                # 这里为简化，用 timeout_ms 与 base 的较小者作为"失败成本"。
                fail_cost = self.timeout_ms if self.enable_timeout else self.dep.base_ms
                total_latency += fail_cost

                if self.enable_breaker:
                    self.breaker.on_failure(self._clock_ms + total_latency)

                # 还有重试次数吗？
                if attempt < max_attempts:
                    backoff = compute_backoff_ms(attempt, self.retry_cfg, self.rng)
                    total_latency += backoff  # 退避等待也算进端到端延迟
                    continue
                # 没有重试次数了 → 跳出循环去兜底
                break

        # ---- 阶段 3：兜底（fallback）----
        if self.enable_fallback:
            fb_latency = self.fallback()
            total_latency += fb_latency
            self._advance_clock(total_latency)
            return CallResult(True, total_latency, served_by="fallback",
                              attempts=attempts, opened=False)

        # ---- 彻底失败 ----
        self._advance_clock(total_latency)
        return CallResult(False, total_latency, served_by=None,
                          attempts=attempts, opened=False)

    # ------------------------------------------------------------------
    def _advance_clock(self, elapsed_ms: float) -> None:
        """推进逻辑时钟。断路器的冷却计时依赖它。"""
        self._clock_ms += elapsed_ms


# ============================================================================
# 5. 仿真与指标统计
# ============================================================================

@dataclass
class Metrics:
    """一批调用的聚合指标。这就是 SRE 每天盯的那些数字。"""
    n: int = 0
    n_success: int = 0
    latencies: List[float] = field(default_factory=list)
    served_by_primary: int = 0
    served_by_fallback: int = 0
    total_attempts: int = 0
    n_opened: int = 0

    @property
    def availability(self) -> float:
        """可用性 = 成功数 / 总数（用户视角，含 fallback 兜底成功）。"""
        return self.n_success / self.n if self.n else 0.0

    @property
    def avg_latency(self) -> float:
        """平均延迟（毫秒）。"""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def p99_latency(self) -> float:
        """P99 尾延迟（毫秒）——99% 的请求都比它快。尾延迟才是用户体验杀手。"""
        return percentile(self.latencies, 99)

    @property
    def p50_latency(self) -> float:
        """P50 中位延迟（毫秒）。"""
        return percentile(self.latencies, 50)

    @property
    def avg_attempts(self) -> float:
        """平均每请求发起了几次底层尝试（>1 说明重试在放大负载）。"""
        return self.total_attempts / self.n if self.n else 0.0


def percentile(data: List[float], p: float) -> float:
    """计算百分位数（最近秩法 nearest-rank，简单稳健，无需 numpy）。

    p=99 表示 P99：把数据升序排列，取第 ceil(p/100 * n) 个。
    为什么自己写而不用 numpy？——核心库保持零依赖，能被任何环境直接 import；
    numpy 只在 run_demo 画图时才需要。
    """
    if not data:
        return 0.0
    ordered = sorted(data)
    import math
    k = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[k - 1]


def simulate(client_factory: Callable[[], ReliableClient], n_requests: int) -> Metrics:
    """跑一轮仿真：用 client_factory 造一个客户端，连发 n_requests 个请求，聚合指标。

    为什么传"工厂函数"而不是直接传 client？——因为断路器/时钟是**有状态**的，
    每种配置应当用一个全新的、干净的 client 来跑，避免上一轮的状态污染。
    工厂模式让 run_demo 能一行切换配置。
    """
    client = client_factory()
    m = Metrics()
    for _ in range(n_requests):
        r = client.call()
        m.n += 1
        m.total_attempts += r.attempts
        m.latencies.append(r.latency_ms)
        if r.success:
            m.n_success += 1
        if r.served_by == "primary":
            m.served_by_primary += 1
        elif r.served_by == "fallback":
            m.served_by_fallback += 1
        if r.opened:
            m.n_opened += 1
    return m


# ============================================================================
# 6. 便捷构造器：一键造出"某种模式"的客户端工厂
# ============================================================================

def make_factory(
    fail_rate: float,
    slow_rate: float = 0.0,
    seed: int = 42,
    enable_retry: bool = False,
    enable_timeout: bool = False,
    enable_breaker: bool = False,
    enable_fallback: bool = False,
    **kwargs,
) -> Callable[[], ReliableClient]:
    """返回一个"客户端工厂"。run_demo 用不同开关组合造出不同模式做对比。

    注意 seed：dependency 和 client 共用一个独立 seed，保证不同"模式"面对的是
    **同一串故障序列**，对比才公平（controlled experiment）。
    """
    def factory() -> ReliableClient:
        # 每次造 client 都用同 seed 的全新 rng，保证故障序列一致
        rng = random.Random(seed)
        dep = FlakyDependency(
            fail_rate=fail_rate,
            base_ms=kwargs.get("base_ms", 20.0),
            slow_rate=slow_rate,
            slow_ms=kwargs.get("slow_ms", 500.0),
            rng=rng,
        )
        return ReliableClient(
            dependency=dep,
            rng=rng,
            retry_cfg=kwargs.get("retry_cfg"),
            breaker_cfg=kwargs.get("breaker_cfg"),
            timeout_ms=kwargs.get("timeout_ms", 200.0),
            fallback=kwargs.get("fallback"),
            enable_retry=enable_retry,
            enable_timeout=enable_timeout,
            enable_breaker=enable_breaker,
            enable_fallback=enable_fallback,
        )
    return factory
