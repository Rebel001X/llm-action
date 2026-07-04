# -*- coding: utf-8 -*-
"""
serving_sim.py —— 分布式推理服务吞吐/延迟仿真核心引擎
================================================================

主题:多副本(replica)+ 连续批处理(continuous batching)的离散事件仿真。

我们不加载任何真实模型、不联网。整个世界被抽象成:
  - 请求(Request):有到达时刻、需要生成的 token 数。
  - 副本(Replica):一块"GPU 卡"的抽象,内部跑一个连续批处理调度器。
  - 集群(Cluster):N 个副本 + 一个负载均衡器(load balancer)。
  - 时钟:离散时间步(step),每步代表一次前向(forward)迭代。

为什么用"离散事件 / 逐步迭代"而不是排队论闭式解?
  连续批处理的核心行为——请求在生成过程中动态加入/离开同一个 batch——
  排队论(M/M/c 之类)无法解析刻画。用仿真逐步推进,才能真实反映
  "batch 里同时有多少条 token 在跑"这种时变利用率。

术语中英并列:
  - throughput 吞吐 = 单位时间完成的请求数(req/s)或 token 数(tok/s)
  - latency 延迟 = 单条请求从到达到完成的时间
  - p50/p95/p99 = 延迟分布的第 50/95/99 百分位(尾延迟 tail latency)
  - utilization 利用率 = 副本忙碌(batch 非空)的时间占比
  - continuous batching 连续批处理 = 迭代级(iteration-level)动态拼批
  - static batching 静态批处理 = 攒满一批一起跑、跑完一起走

作者:配套《Distributed AI Systems》实战项目 02
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# =====================================================================
# 1. 数据结构:请求
# =====================================================================
@dataclass
class Request:
    """一条推理请求。

    字段:
      rid          : 请求 id(全局唯一)
      arrival      : 到达时刻(单位:秒)
      total_tokens : 该请求要生成的 token 总数(decode 步数)
      remaining    : 还剩多少 token 没生成(仿真中递减)
      start_time   : 第一次被调度进 batch 的时刻(用于算排队时间)
      finish_time  : 生成完最后一个 token 的时刻
    """
    rid: int
    arrival: float
    total_tokens: int
    remaining: int = field(init=False)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    def __post_init__(self) -> None:
        # remaining 初始化为 total_tokens。用 __post_init__ 是因为
        # dataclass 的 field(init=False) 不能直接引用另一个字段做默认值。
        self.remaining = self.total_tokens

    @property
    def latency(self) -> Optional[float]:
        """端到端延迟 = 完成时刻 - 到达时刻。未完成则为 None。"""
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival

    @property
    def queue_delay(self) -> Optional[float]:
        """排队时延 = 首次被调度时刻 - 到达时刻(等了多久才轮到我)。"""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival


# =====================================================================
# 2. 到达流生成:泊松过程
# =====================================================================
def poisson_arrivals(
    rate: float,
    duration: float,
    token_mean: float = 128.0,
    token_min: int = 1,
    seed: int = 0,
) -> List[Request]:
    """生成一段泊松到达流(Poisson arrival process)。

    参数:
      rate       : 平均到达率 λ(req/s)。
      duration   : 仿真时长(秒)。
      token_mean : 每条请求生成 token 数的均值(几何分布近似真实长尾)。
      token_min  : token 数下限(至少 1)。
      seed       : 随机种子,保证可复现。

    🔬 第一性原理:为什么用泊松过程?
      真实线上流量,在"用户彼此独立、瞬时到达率恒定"的近似下,
      相邻两次到达的时间间隔服从指数分布(exponential),
      到达次数服从泊松分布——这就是泊松过程的定义。
      指数分布采样:inter = -ln(U)/λ,U~Uniform(0,1)。

    返回:按到达时间升序排列的 Request 列表。
    """
    rng = random.Random(seed)
    reqs: List[Request] = []
    t = 0.0
    rid = 0
    while True:
        # 指数分布间隔:-ln(U)/λ。U 取 (0,1] 避免 log(0)。
        u = 1.0 - rng.random()  # (0,1]
        inter = -math.log(u) / rate
        t += inter
        if t > duration:
            break
        # token 数用几何分布(离散、长尾),均值≈token_mean。
        # 几何分布参数 p = 1/token_mean;采样 ceil(ln(U)/ln(1-p))。
        p = 1.0 / max(token_mean, 1.0)
        u2 = 1.0 - rng.random()
        n_tok = int(math.ceil(math.log(u2) / math.log(1.0 - p))) if p < 1.0 else 1
        n_tok = max(token_min, n_tok)
        reqs.append(Request(rid=rid, arrival=t, total_tokens=n_tok))
        rid += 1
    return reqs


# =====================================================================
# 3. 副本:连续批处理调度器
# =====================================================================
class Replica:
    """单个副本(≈ 一块 GPU)。内部实现连续批处理。

    模型:时间被切成等长的 step(迭代),每个 step 时长 = step_time 秒。
    每个 step 里,副本从"运行集合(running set)"中取最多 max_batch 条请求,
    每条各生成 1 个 token(remaining -= 1);生成完(remaining==0)的请求离场,
    腾出的 slot 立刻让排队中的新请求补进来——这就是"连续/迭代级批处理"。

    对比静态批处理:静态批要等整批都生成完才放人、才收新人,
    短请求被长请求"拖住"(head-of-line blocking),利用率低。

    ⚠️ 简化假设(便于教学,面试要能说清代价):
      1) step_time 与 batch 内 token 数无关(理想化;真实中 batch 越大,
         单 step 略慢,但每 token 摊薄更快 → 我们用 batch_slowdown 建模这点)。
      2) 忽略 prefill/decode 区分,把每条请求视为纯 decode(逐 token)。
      3) 忽略 KV cache 显存上限(用 max_batch 间接体现容量)。
    """

    def __init__(
        self,
        step_time: float = 0.01,
        max_batch: int = 16,
        batch_slowdown: float = 0.0,
        continuous: bool = True,
    ) -> None:
        """
        step_time      : 一个迭代的基础耗时(秒)。0.01s = 每卡 100 iter/s。
        max_batch      : 一个 batch 最多并发多少条请求(容量)。
        batch_slowdown : 批变大导致的单 step 变慢系数。
                         实际 step 时长 = step_time * (1 + batch_slowdown * (b-1))。
                         =0 表示理想线性扩展;>0 表示大 batch 有边际成本。
        continuous     : True=连续批处理;False=静态批处理(用于对照实验)。
        """
        self.step_time = step_time
        self.max_batch = max_batch
        self.batch_slowdown = batch_slowdown
        self.continuous = continuous

        self.running: List[Request] = []   # 当前在 batch 里跑的请求
        self.queue: List[Request] = []     # 已分配到本副本、等待进 batch 的请求
        self.clock: float = 0.0            # 本副本的本地时钟

        # 统计量
        self.busy_time: float = 0.0        # 累计"batch 非空"的时长(算时间利用率)
        self.total_time: float = 0.0       # 累计推进的总时长
        self.tokens_done: int = 0          # 累计生成 token 数
        self.reqs_done: int = 0            # 累计完成请求数
        # slot 占用:每个 step 记录 (batch 大小 * dt),用于算"批槽位利用率"。
        # 时间利用率只看 batch 是否非空(1 条也算满);槽位利用率看 batch 有多满,
        # 这才是真正区分"连续 vs 静态批处理"的指标(见 slot_utilization)。
        self.slot_busy: float = 0.0        # ∑ (b * dt),b=该 step batch 大小

    def admit(self, req: Request) -> None:
        """负载均衡器把一条请求派发到本副本的等待队列。"""
        self.queue.append(req)

    def _refill_batch(self) -> None:
        """把等待队列里的请求补进 running,直到达到 max_batch。

        连续批处理:任何时候只要有空 slot 就补(下面 step 里每步都调)。
        静态批处理:只有 running 全空时才一次性攒一批(见 step)。
        """
        while self.queue and len(self.running) < self.max_batch:
            req = self.queue.pop(0)
            if req.start_time is None:
                req.start_time = self.clock  # 记录首次被调度时刻
            self.running.append(req)

    def has_work(self) -> bool:
        """本副本是否还有活儿(running 或 queue 非空)。"""
        return bool(self.running) or bool(self.queue)

    def step(self) -> float:
        """推进一个迭代,返回本 step 实际耗时。

        流程:
          1) 补批(连续:每步补;静态:仅当 running 空时补)。
          2) 若 running 空(且 queue 也空)→ 空转,时钟按 step_time 前进但不算 busy。
          3) 否则:batch 内每条 remaining-=1,tokens_done 增加;
             算本 step 时长(含 batch_slowdown);推进时钟;
             把 remaining==0 的请求标记完成、移出 running。
        """
        if self.continuous:
            self._refill_batch()
        else:
            # 静态批处理:上一批必须全部跑完(running 空)才收新批。
            if not self.running:
                self._refill_batch()

        b = len(self.running)
        if b == 0:
            # 没活干:时钟前进一个基础 step,但不计入 busy_time。
            self.clock += self.step_time
            self.total_time += self.step_time
            return self.step_time

        # 本 step 实际时长:batch 越大略慢(batch_slowdown 建模)。
        dt = self.step_time * (1.0 + self.batch_slowdown * (b - 1))

        # batch 内每条请求生成 1 个 token。
        finished: List[Request] = []
        for req in self.running:
            req.remaining -= 1
            self.tokens_done += 1
            if req.remaining <= 0:
                req.finish_time = self.clock + dt  # step 末尾完成
                finished.append(req)

        # 推进时钟与统计。
        self.clock += dt
        self.total_time += dt
        self.busy_time += dt          # 有 batch 在跑 → 这段时间算(时间)忙
        self.slot_busy += b * dt      # 槽位占用:b 个 slot 忙了 dt 秒

        # 完成的请求离场。
        for req in finished:
            self.running.remove(req)
            self.reqs_done += 1

        return dt

    @property
    def utilization(self) -> float:
        """时间利用率 = 忙碌时长 / 总时长。batch 只要非空就算忙。

        ⚠️ 注意:高负载下几乎永远有 backlog,这个值会顶到 100%,
        看不出批处理好坏。要看批处理效率,请用下面的 slot_utilization。
        """
        if self.total_time <= 0:
            return 0.0
        return self.busy_time / self.total_time

    @property
    def slot_utilization(self) -> float:
        """槽位利用率 = ∑(b*dt) / (max_batch * total_time)。

        含义:把"max_batch 个槽位 × 全部时间"当作满负荷分母,
        实际用了多少 batch 槽位·秒当分子。batch 越满、空转越少 → 越高。

        🔬 这才是区分连续/静态批处理的关键:
          静态批处理里,一批请求长短不一,短的先跑完却要空等长的,
          batch 实际占用的槽位随时间衰减(b 从满逐步掉到 1),槽位利用率低;
          连续批处理里,短请求一走空 slot 立刻被新请求填满,槽位常年接近满,
          槽位利用率高。
        """
        denom = self.max_batch * self.total_time
        if denom <= 0:
            return 0.0
        return self.slot_busy / denom


# =====================================================================
# 4. 集群:N 副本 + 负载均衡
# =====================================================================
class Cluster:
    """N 个副本组成的推理集群 + 负载均衡器。

    负载均衡策略(load balancing policy):
      - "least_loaded" : 派给"当前队列+running 最少"的副本(默认,近似 JSQ)。
      - "round_robin"  : 轮询。
      - "random"       : 随机。

    仿真采用"全局同步时钟"的简化模型:所有副本共用一个全局时间轴,
    每一轮(global step)所有有活的副本各推进一个迭代。这样便于统计,
    也足以刻画吞吐/利用率/尾延迟的相对关系(教学目的)。
    """

    def __init__(
        self,
        n_replicas: int,
        step_time: float = 0.01,
        max_batch: int = 16,
        batch_slowdown: float = 0.0,
        continuous: bool = True,
        policy: str = "least_loaded",
    ) -> None:
        assert n_replicas >= 1
        self.replicas: List[Replica] = [
            Replica(step_time, max_batch, batch_slowdown, continuous)
            for _ in range(n_replicas)
        ]
        self.policy = policy
        self._rr = 0            # round_robin 指针
        self._rng = random.Random(12345)

    def _pick_replica(self) -> Replica:
        """按策略选一个副本接收新请求。"""
        if self.policy == "round_robin":
            r = self.replicas[self._rr % len(self.replicas)]
            self._rr += 1
            return r
        if self.policy == "random":
            return self._rng.choice(self.replicas)
        # least_loaded(默认):队列+running 总负载最小者。
        return min(self.replicas, key=lambda r: len(r.queue) + len(r.running))

    def run(self, requests: List[Request]) -> Dict[str, object]:
        """跑完整段仿真。

        输入:按 arrival 升序的请求列表。
        逻辑:全局时钟从 0 推进;每一轮 global step:
          (a) 把"到达时刻 <= 当前全局时钟"且尚未派发的请求派给副本;
          (b) 所有有活的副本各推进一个迭代;
          (c) 全局时钟 = 所有推进副本里最小的新本地时钟(近似同步)。
        直到所有请求都派发且所有副本无活。

        返回:统计字典(见 collect_metrics 使用)。
        """
        reqs = sorted(requests, key=lambda r: r.arrival)
        n = len(reqs)
        idx = 0                 # 下一个待派发请求的下标
        global_clock = 0.0

        # 让每个副本的本地时钟对齐全局(简化:直接共享推进节奏)。
        # 安全阀:step 上限,防止死循环(理论上不会触发)。
        max_steps = 10_000_000
        steps = 0

        while True:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("超出最大步数,可能存在逻辑错误")

            # (a) 派发所有已到达的请求。
            while idx < n and reqs[idx].arrival <= global_clock + 1e-12:
                self._pick_replica().admit(reqs[idx])
                idx += 1

            # 判断是否还有活:有未派发请求 或 任一副本有活。
            any_work = any(r.has_work() for r in self.replicas)
            if idx >= n and not any_work:
                break  # 全部派发完 且 所有副本空 → 收工

            if not any_work:
                # 副本都空,但还有请求没到 → 时钟快进到下一个到达时刻。
                if idx < n:
                    global_clock = reqs[idx].arrival
                    # 同步副本本地时钟(空转不计 busy)。
                    for r in self.replicas:
                        r.clock = global_clock
                continue

            # (b) 所有有活的副本各推进一个迭代;记录最小推进后时钟。
            for r in self.replicas:
                if r.has_work():
                    r.step()  # step 内部推进 r.clock

            # (c) 全局时钟 = 有活副本中最小的本地时钟(近似同步节拍)。
            busy_clocks = [r.clock for r in self.replicas if r.has_work()]
            if busy_clocks:
                global_clock = min(busy_clocks)
            else:
                # 本轮把活干完了,时钟取所有副本最大值以便下轮派发。
                global_clock = max(r.clock for r in self.replicas)

        return self.collect_metrics(reqs)

    def collect_metrics(self, reqs: List[Request]) -> Dict[str, object]:
        """汇总统计:吞吐、延迟分布、利用率等。"""
        latencies = [r.latency for r in reqs if r.latency is not None]
        n_done = len(latencies)
        n_total = len(reqs)

        # makespan:从第一条到达到最后一条完成的总时长。
        first_arrival = min((r.arrival for r in reqs), default=0.0)
        last_finish = max((r.finish_time for r in reqs if r.finish_time is not None),
                          default=0.0)
        makespan = max(last_finish - first_arrival, 1e-9)

        total_tokens = sum(r.tokens_done for r in self.replicas)

        metrics: Dict[str, object] = {
            "n_total": n_total,
            "n_done": n_done,
            "all_done": n_done == n_total,
            "makespan": makespan,
            "throughput_req": n_done / makespan,          # req/s
            "throughput_tok": total_tokens / makespan,    # tok/s
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean_latency": (sum(latencies) / n_done) if n_done else float("nan"),
            "utilizations": [r.utilization for r in self.replicas],
            "mean_utilization": (
                sum(r.utilization for r in self.replicas) / len(self.replicas)
            ),
            "slot_utilizations": [r.slot_utilization for r in self.replicas],
            "mean_slot_utilization": (
                sum(r.slot_utilization for r in self.replicas) / len(self.replicas)
            ),
            "n_replicas": len(self.replicas),
        }
        return metrics


# =====================================================================
# 5. 工具函数:百分位
# =====================================================================
def percentile(data: List[float], q: float) -> float:
    """计算第 q 百分位(线性插值),纯 Python 实现,避免依赖 numpy。

    q ∈ [0,100]。空列表返回 nan。
    步骤:排序 → 定位 rank = (q/100)*(n-1) → 线性插值相邻两点。
    """
    if not data:
        return float("nan")
    s = sorted(data)
    n = len(s)
    if n == 1:
        return s[0]
    rank = (q / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# =====================================================================
# 6. 顶层便捷函数:一键跑一个场景
# =====================================================================
def simulate(
    n_replicas: int,
    rate: float,
    duration: float = 20.0,
    step_time: float = 0.01,
    max_batch: int = 16,
    token_mean: float = 64.0,
    batch_slowdown: float = 0.0,
    continuous: bool = True,
    policy: str = "least_loaded",
    seed: int = 0,
) -> Dict[str, object]:
    """一键仿真:给定副本数与到达率,返回指标字典。

    这是 tests 和 run_demo 都会调用的统一入口。
    """
    reqs = poisson_arrivals(
        rate=rate, duration=duration, token_mean=token_mean, seed=seed
    )
    cluster = Cluster(
        n_replicas=n_replicas,
        step_time=step_time,
        max_batch=max_batch,
        batch_slowdown=batch_slowdown,
        continuous=continuous,
        policy=policy,
    )
    return cluster.run(reqs)


if __name__ == "__main__":
    # 快速自检:跑一个中等负载场景,打印关键指标。
    m = simulate(n_replicas=2, rate=30.0, duration=15.0, seed=1)
    print("=== 自检:2 副本, λ=30 req/s ===")
    print(f"完成 {m['n_done']}/{m['n_total']} 请求, all_done={m['all_done']}")
    print(f"吞吐: {m['throughput_req']:.2f} req/s, {m['throughput_tok']:.1f} tok/s")
    print(f"延迟 p50/p95/p99 = "
          f"{m['p50']:.3f} / {m['p95']:.3f} / {m['p99']:.3f} s")
    print(f"平均利用率: {m['mean_utilization']:.1%}")
