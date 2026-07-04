# -*- coding: utf-8 -*-
"""
autoscale_sim.py —— K8s 自动扩缩容(HPA / KEDA 式)离散事件仿真核心
================================================================

本模块用纯 numpy + 标准库,从零实现一个"到达流 → 队列 → 一组 Pod 副本处理
→ 自动扩缩容控制器"的闭环仿真,用来定量回答:

    - 给定一条随时间变化的请求到达率,自动扩缩容策略能否守住 SLO(延迟)?
    - 为此花了多少钱(pod·小时成本)?
    - 扩缩过程有多"抖"(抖动 flapping)、有多"过"(过冲 overshoot)?

设计哲学(第一性原理):
    Kubernetes 的 HPA(Horizontal Pod Autoscaler)本质上是一个**离散时间的
    反馈控制器**:每隔一个"同步周期(sync period)"采样一次指标(metric),
    与目标值(target)相比,按比例算出期望副本数,再受"冷却(cooldown)"和
    "启动延迟(pod warmup)"约束后作用回系统。KEDA 在此之上把"指标源"从
    CPU/内存扩展到了队列长度、Kafka lag、GPU 利用率等外部信号。

    所以只要把下面四块讲清楚,就抓住了自动扩缩容的本质:
        1) 负载模型:到达流 arrivals(t)  —— 需求
        2) 服务模型:N 个 Pod、每 Pod 的处理能力 + 启动延迟 —— 供给
        3) 排队模型:请求排队、被处理、延迟 = 排队 + 服务 —— 供需撮合
        4) 控制模型:采样指标 → 期望副本 → 冷却/延迟约束 —— 反馈调节

    我们用**固定步长 Δt 的离散时间推进(fixed-step discrete time)**而不是
    纯事件驱动,因为 HPA 本身就是按固定周期采样的,固定步长更贴近真实语义,
    也更好读、好测。

术语中英并列:
    - Pod 副本(replica)      : 一个无状态服务实例,可水平复制
    - 到达率(arrival rate)   : 单位时间进入系统的请求数 λ(t)
    - 每 Pod 服务率(μ, service rate): 一个 Pod 每秒能处理的请求数
    - 目标利用率(target utilization): HPA 想把每 Pod 维持在的忙碌程度
    - 冷却期(cooldown / stabilization window): 两次同向扩缩的最小间隔
    - 启动延迟(pod warmup / startup delay): 新 Pod 从创建到能干活的时间
    - SLO(Service Level Objective): 这里指"延迟 ≤ 阈值"的比例目标
    - 过冲(overshoot): 副本数超过"其实需要的量"的程度
    - 抖动(flapping / thrashing): 短时间内反复扩了又缩、缩了又扩

作者:AI 系统工程师 · 配套《Generative AI on Kubernetes》
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


# =============================================================================
# 1. 配置对象:把所有可调旋钮集中在一处(便于做参数扫描 / 权衡实验)
# =============================================================================
@dataclass
class SimConfig:
    """仿真配置(所有时间单位统一为"秒")。

    这些字段就是你在真实 HPA / KEDA YAML 里会写的旋钮,只是换成了 Python:
        - dt / sync_period 对应 HPA 的 --horizontal-pod-autoscaler-sync-period
        - scale_up_cooldown / scale_down_cooldown 对应 behavior.scaleUp/scaleDown 的
          stabilizationWindowSeconds
        - min_replicas / max_replicas 对应 HPA 的 minReplicas / maxReplicas
        - target_metric 对应 KEDA 的 trigger target(如"每 Pod 队列长度=30")
        - pod_startup_delay 对应新 Pod 冷启动 + 拉镜像 + readinessProbe 通过的时间
    """

    # ---- 时间轴 ----
    dt: float = 1.0                 # 仿真步长 Δt(秒)。也当作 HPA 采样周期
    horizon: float = 600.0          # 总仿真时长(秒)

    # ---- 服务能力(供给侧)----
    per_pod_capacity: float = 10.0  # 每个 Pod 每秒能处理的请求数 μ(rps/pod)
    pod_startup_delay: float = 15.0 # 新 Pod 从"被创建"到"能处理请求"的启动延迟(秒)

    # ---- 副本边界 ----
    min_replicas: int = 2
    max_replicas: int = 60
    init_replicas: int = 2          # 仿真开始时已就绪的副本数

    # ---- 控制器(HPA/KEDA 式)----
    # 指标类型:'queue'(队列长度/Pod)| 'latency'(平均延迟)| 'gpu'(GPU 利用率)
    metric: str = "queue"
    target_metric: float = 5.0      # 目标指标值(每 Pod)。队列模式=每 Pod 排队请求数
    tolerance: float = 0.10         # 容差带:|当前/目标 - 1| < tolerance 时不动作(防抖核心)
    scale_up_cooldown: float = 0.0  # 扩容冷却(秒)。真实 HPA 默认 0(扩容要快)
    scale_down_cooldown: float = 60.0  # 缩容冷却(秒)。真实 HPA 默认 300(缩容要稳)
    max_scale_up_rate: float = 2.0  # 单次扩容最多变为原来的几倍(限制暴涨,HPA 默认可翻倍)
    max_scale_down_step: int = 4    # 单次缩容最多减少多少副本(平滑缩容,防抖动)

    # ---- SLO 与成本 ----
    slo_latency: float = 1.0        # SLO 延迟阈值(秒):延迟 ≤ 此值算"达标"
    pod_cost_per_hour: float = 2.5  # 每 Pod·小时的价格(例如一张 GPU 的云价)

    # ---- 负载模型 ----
    # arrival_fn(t) -> 该时刻的"到达率 λ(t)"(rps)。默认给一个梯形冲击波
    arrival_fn: Optional[Callable[[float], float]] = None
    rng_seed: int = 42              # 泊松到达随机数种子(可复现)
    poisson_arrivals: bool = True   # True: 每步到达数服从 Poisson(λ·dt);False: 取确定的 λ·dt


# =============================================================================
# 2. 负载发生器:几种典型到达流(需求侧)
# =============================================================================
def trapezoid_load(t: float,
                   base: float = 20.0,
                   peak: float = 260.0,
                   t_up: float = 120.0,
                   t_full: float = 200.0,
                   t_down: float = 380.0,
                   t_end: float = 460.0) -> float:
    """梯形负载:低 → 线性爬升 → 高原 → 线性回落 → 低。

    模拟一次"营销活动 / 早高峰"式的流量冲击,是压测自动扩缩容最经典的形状:
    爬坡段考验扩容速度,回落段考验缩容的稳(不能一掉就狂缩)。
    """
    if t < t_up:
        return base
    if t < t_full:
        # 线性爬升
        return base + (peak - base) * (t - t_up) / (t_full - t_up)
    if t < t_down:
        return peak
    if t < t_end:
        # 线性回落
        return peak - (peak - base) * (t - t_down) / (t_end - t_down)
    return base


def spiky_load(t: float, base: float = 30.0, spikes: Optional[List] = None) -> float:
    """尖峰负载:平时低,若干瞬时尖峰。用来暴露"启动延迟 + 冷却"下的 SLO 破口。"""
    if spikes is None:
        spikes = [(150, 200, 300.0), (350, 380, 350.0)]  # (起, 止, 峰值 rps)
    for s, e, p in spikes:
        if s <= t < e:
            return p
    return base


def diurnal_load(t: float, base: float = 40.0, amp: float = 120.0, period: float = 600.0) -> float:
    """昼夜正弦负载:模拟一天的潮汐流量。考验缩容不过度、成本-SLO 的长期权衡。"""
    return base + amp * (0.5 * (1.0 - np.cos(2.0 * np.pi * t / period)))


# =============================================================================
# 3. Pod 生命周期:处理"启动延迟"—— 新 Pod 不是立刻能干活的
# =============================================================================
@dataclass
class Fleet:
    """副本机群(fleet):区分"已就绪 ready"与"启动中 warming"的 Pod。

    ⚠️ 这是仿真里最容易被忽略、但对结论影响最大的一块:
       真实世界扩容有滞后 —— 你在 t 时刻决定加 5 个 Pod,它们要到
       t + startup_delay 才真正开始分担负载。这段"空窗期"正是 SLO 被击穿的
       高发区,也是"过冲(为了赶紧压住延迟而超量扩容)"的根源。
    """
    ready: int                       # 当前已就绪(能处理请求)的 Pod 数
    startup_delay: float             # 启动延迟(秒)
    # 启动中的 Pod:列表元素为 [剩余启动时间, 数量]
    warming: List[List[float]] = field(default_factory=list)

    def total(self) -> int:
        """已就绪 + 启动中 的总副本数(成本按"占用即计费"算,启动中也要花钱)。"""
        return self.ready + sum(int(n) for _, n in self.warming)

    def request_ready(self) -> int:
        return self.ready

    def scale_to(self, target: int) -> None:
        """把"目标副本数"作用到机群上。

        - 扩容:新增的 Pod 进入 warming 队列,带 startup_delay 的倒计时。
        - 缩容:优先砍掉"启动中"的 Pod(它们还没干活,砍了不心疼、也更快生效),
                不够再砍"已就绪"的。这贴近 K8s 优先驱逐未 Ready Pod 的直觉。
        """
        cur_total = self.total()
        if target > cur_total:                       # ---- 扩容 ----
            add = target - cur_total
            if self.startup_delay <= 0:
                self.ready += add                    # 无启动延迟:立即就绪
            else:
                self.warming.append([self.startup_delay, float(add)])
        elif target < cur_total:                     # ---- 缩容 ----
            remove = cur_total - target
            # 先从 warming(从最晚创建的开始砍)
            while remove > 0 and self.warming:
                rem_t, n = self.warming[-1]
                if n <= remove:
                    remove -= int(n)
                    self.warming.pop()
                else:
                    self.warming[-1][1] = n - remove
                    remove = 0
            # 再砍 ready
            if remove > 0:
                self.ready = max(0, self.ready - remove)

    def tick(self, dt: float) -> None:
        """推进 dt:启动中的 Pod 倒计时,归零者转为 ready。"""
        still_warming: List[List[float]] = []
        for rem_t, n in self.warming:
            rem_t -= dt
            if rem_t <= 1e-9:
                self.ready += int(n)                 # 启动完成,加入就绪
            else:
                still_warming.append([rem_t, n])
        self.warming = still_warming


# =============================================================================
# 4. 排队模型:一个"确定性流体近似 + 泊松到达"的 M/M/c 风格队列
# =============================================================================
class Queue:
    """请求队列 + 服务台(server=已就绪 Pod)。

    我们用**流体近似(fluid approximation)**:每一步
        - 进来 a 个请求(泊松或确定)
        - 有效服务能力 = ready_pods · per_pod_capacity · dt(单位:请求/步)
        - 出去 min(队列现存, 服务能力) 个请求
    延迟用**利特尔法则(Little's Law)**的即时版本估计:
        平均延迟 ≈ 队列长度 / 有效服务率
    这不是精确的 M/M/c 排队论,但对"自动扩缩容宏观行为"足够真实,而且透明、可测、
    不引入随机噪声掩盖控制逻辑。🔬 排队论精确公式见 README 面试点。
    """

    def __init__(self):
        self.backlog: float = 0.0    # 当前积压(队列中等待 + 正在处理的请求数)

    def step(self, arrivals: float, ready_pods: int, per_pod_capacity: float,
             dt: float) -> Dict[str, float]:
        """推进一步,返回本步的观测量。"""
        # 1) 新到达先入队
        self.backlog += arrivals
        # 2) 本步服务能力(能处理多少请求)
        capacity = ready_pods * per_pod_capacity * dt
        served = min(self.backlog, capacity)
        self.backlog -= served
        # 3) 估计当前平均延迟(利特尔法则的即时近似):
        #    延迟 ≈ 队列长度 / 服务速率(每秒能处理多少)
        service_rate = max(ready_pods * per_pod_capacity, 1e-9)  # 请求/秒
        latency = self.backlog / service_rate
        # 4) 每 Pod 队列长度(KEDA 队列触发器用的核心指标)
        q_per_pod = self.backlog / max(ready_pods, 1)
        return {
            "arrivals": arrivals,
            "served": served,
            "backlog": self.backlog,
            "latency": latency,
            "q_per_pod": q_per_pod,
        }


# =============================================================================
# 5. 自动扩缩容控制器:HPA / KEDA 的算法内核
# =============================================================================
class Autoscaler:
    """HPA 式比例控制器 + 冷却 + 容差带 + 变化率限制。

    核心公式(直接照抄 Kubernetes HPA 官方算法):

        desiredReplicas = ceil( currentReplicas * ( currentMetric / targetMetric ) )

    再叠加:
        - 容差带 tolerance:比值落在 [1-tol, 1+tol] 内 → 不动(避免地毯式抖动)
        - 冷却窗口 cooldown:同向动作要间隔足够久(缩容尤其要稳)
        - 变化率限制:一次别扩太猛(max_scale_up_rate)、别缩太狠(max_scale_down_step)
        - 边界钳制:min_replicas ≤ desired ≤ max_replicas
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.current: int = cfg.init_replicas
        self.last_scale_up_t: float = -1e9
        self.last_scale_down_t: float = -1e9

    def _current_metric_per_pod(self, obs: Dict[str, float], ready: int) -> float:
        """从观测里取出"控制器关心的指标(每 Pod)"。

        三种触发源(KEDA 的精髓:指标可插拔):
            queue  : 每 Pod 队列长度  q_per_pod
            latency: 平均延迟(注意:延迟不是"每 Pod"量,直接用整体延迟对比目标)
            gpu    : GPU 利用率 —— 这里用"负载/容量"近似:每 Pod 忙碌程度
        """
        m = self.cfg.metric
        if m == "queue":
            return obs["q_per_pod"]
        if m == "latency":
            return obs["latency"]
        if m == "gpu":
            # GPU 利用率 ≈ 到达带来的瞬时负载 / 当前就绪容量,钳到 [0, 1+]
            demand = obs["arrivals"] / max(self.cfg.dt, 1e-9)   # rps
            cap = max(ready * self.cfg.per_pod_capacity, 1e-9)  # rps
            return demand / cap
        raise ValueError(f"未知 metric: {m}")

    def decide(self, t: float, obs: Dict[str, float], ready: int) -> int:
        """给出本步的"期望总副本数"。这是整个控制器的入口。"""
        cfg = self.cfg
        cur = self.current
        target = cfg.target_metric
        metric = self._current_metric_per_pod(obs, ready)

        # ---- 比值(ratio):当前指标 / 目标指标 ----
        ratio = metric / max(target, 1e-9)

        # ---- 容差带:接近目标就不动(HPA 的 tolerance,默认 0.1)----
        if abs(ratio - 1.0) < cfg.tolerance:
            return cur

        # ---- HPA 核心公式:期望副本 = ceil(当前副本 · 比值)----
        desired = int(np.ceil(cur * ratio))

        # ---- 变化率限制 ----
        if desired > cur:  # 扩容:限制单次最大倍率
            desired = min(desired, int(np.ceil(cur * cfg.max_scale_up_rate)))
            # 至少 +1(否则 ceil 后可能因取整卡住)
            desired = max(desired, cur + 1)
        else:              # 缩容:限制单次最大步长
            desired = max(desired, cur - cfg.max_scale_down_step)
            desired = min(desired, cur - 1)  # 至少 -1

        # ---- 边界钳制 ----
        desired = int(np.clip(desired, cfg.min_replicas, cfg.max_replicas))
        if desired == cur:
            return cur

        # ---- 冷却窗口:同向动作必须间隔足够久 ----
        if desired > cur:  # 想扩容
            if t - self.last_scale_up_t < cfg.scale_up_cooldown:
                return cur  # 还在扩容冷却里,忍住
            self.last_scale_up_t = t
        else:              # 想缩容
            if t - self.last_scale_down_t < cfg.scale_down_cooldown:
                return cur  # 还在缩容冷却里,忍住
            self.last_scale_down_t = t

        self.current = desired
        return desired


# =============================================================================
# 6. 主仿真循环:把上面四块串成闭环,逐步推进并记录时间线
# =============================================================================
def run_sim(cfg: SimConfig) -> Dict[str, np.ndarray]:
    """跑一整条仿真,返回逐步时间线(numpy 数组,便于画图 / 统计)。

    闭环顺序(每步 Δt):
        采样指标(上一步的观测) → 控制器决策期望副本 → 作用到机群(带启动延迟)
        → 机群 tick(启动中 Pod 倒计时)→ 队列 step(到达/服务/延迟)→ 记录
    """
    if cfg.arrival_fn is None:
        cfg.arrival_fn = trapezoid_load
    rng = np.random.default_rng(cfg.rng_seed)

    fleet = Fleet(ready=cfg.init_replicas, startup_delay=cfg.pod_startup_delay)
    queue = Queue()
    scaler = Autoscaler(cfg)

    n_steps = int(round(cfg.horizon / cfg.dt))
    # 时间线容器
    ts = np.zeros(n_steps)
    arr = np.zeros(n_steps)         # 实际到达数(每步)
    lam = np.zeros(n_steps)         # 名义到达率 λ(t)(rps)
    ready_pods = np.zeros(n_steps)  # 已就绪副本
    total_pods = np.zeros(n_steps)  # 总副本(含启动中,用于成本)
    backlog = np.zeros(n_steps)     # 队列积压
    latency = np.zeros(n_steps)     # 平均延迟
    served = np.zeros(n_steps)      # 每步处理量
    desired_arr = np.zeros(n_steps) # 控制器期望副本(决策值)

    # 上一步观测:第 0 步没有历史,用一个"零负载"观测冷启动
    last_obs = {"arrivals": 0.0, "served": 0.0, "backlog": 0.0,
                "latency": 0.0, "q_per_pod": 0.0}

    for i in range(n_steps):
        t = i * cfg.dt

        # (a) 控制器基于"上一步观测 + 当前就绪副本"决策
        desired = scaler.decide(t, last_obs, fleet.request_ready())
        fleet.scale_to(desired)

        # (b) 机群推进:启动中的 Pod 倒计时(可能有 Pod 本步转就绪)
        fleet.tick(cfg.dt)

        # (c) 生成本步到达数
        lam_t = float(cfg.arrival_fn(t))
        if cfg.poisson_arrivals:
            a = float(rng.poisson(max(lam_t * cfg.dt, 0.0)))
        else:
            a = lam_t * cfg.dt

        # (d) 队列推进:用"就绪副本"处理请求
        obs = queue.step(a, fleet.request_ready(), cfg.per_pod_capacity, cfg.dt)
        last_obs = obs  # 供下一步控制器采样

        # (e) 记录
        ts[i] = t
        arr[i] = a
        lam[i] = lam_t
        ready_pods[i] = fleet.request_ready()
        total_pods[i] = fleet.total()
        backlog[i] = obs["backlog"]
        latency[i] = obs["latency"]
        served[i] = obs["served"]
        desired_arr[i] = desired

    return {
        "t": ts, "arrivals": arr, "lambda": lam,
        "ready_pods": ready_pods, "total_pods": total_pods,
        "backlog": backlog, "latency": latency, "served": served,
        "desired": desired_arr,
    }


# =============================================================================
# 7. 指标计算:把时间线压成几个"能上汇报 PPT"的数字
# =============================================================================
def compute_metrics(result: Dict[str, np.ndarray], cfg: SimConfig) -> Dict[str, float]:
    """从时间线算出 SLO 满足率 / 成本 / 过冲 / 抖动等汇总指标。

    - slo_ok_ratio  : 延迟 ≤ slo_latency 的时间步占比(越高越好)
    - p95_latency   : 延迟的 95 分位(越低越好)
    - pod_hours     : 总副本·小时(Σ total_pods·dt / 3600),即成本的物理量
    - cost          : pod_hours · 单价(越低越好)
    - overshoot     : 过冲 = 平均(总副本 - 理论所需副本)的正部分(越低越省)
    - flaps         : 抖动次数 = 期望副本"方向反转"的次数(越低越稳)
    - throughput    : 总处理请求数
    - drop_ratio    : 若队列爆掉的丢弃占比(此模型不主动丢,恒为 0,占位对齐)
    """
    dt = cfg.dt
    lat = result["latency"]
    total_pods = result["total_pods"]
    lam = result["lambda"]

    # SLO
    slo_ok = float(np.mean(lat <= cfg.slo_latency))
    p95 = float(np.percentile(lat, 95))
    p99 = float(np.percentile(lat, 99))
    max_lat = float(np.max(lat))

    # 成本
    pod_hours = float(np.sum(total_pods) * dt / 3600.0)
    cost = pod_hours * cfg.pod_cost_per_hour

    # 过冲:理论所需就绪副本 ≈ ceil(λ / per_pod_capacity)(把负载正好压住)
    need = np.ceil(lam / max(cfg.per_pod_capacity, 1e-9))
    need = np.clip(need, cfg.min_replicas, cfg.max_replicas)
    over = total_pods - need
    overshoot = float(np.mean(np.clip(over, 0, None)))  # 平均超配副本数

    # 抖动:期望副本序列的方向反转次数
    desired = result["desired"]
    diff = np.diff(desired)
    sign = np.sign(diff)
    sign = sign[sign != 0]          # 只看真正变化的步
    flaps = int(np.sum(sign[1:] != sign[:-1])) if sign.size > 1 else 0

    throughput = float(np.sum(result["served"]))

    return {
        "slo_ok_ratio": slo_ok,
        "p95_latency": p95,
        "p99_latency": p99,
        "max_latency": max_lat,
        "pod_hours": pod_hours,
        "cost": cost,
        "avg_overshoot": overshoot,
        "flaps": flaps,
        "throughput": throughput,
        "mean_ready_pods": float(np.mean(result["ready_pods"])),
        "peak_pods": float(np.max(total_pods)),
    }


# =============================================================================
# 8. 便捷入口:一行跑完 + 打印摘要(供 run_demo / 手动实验调用)
# =============================================================================
def simulate(cfg: Optional[SimConfig] = None) -> Dict[str, object]:
    """跑一条仿真并返回 {'result': 时间线, 'metrics': 汇总}。"""
    if cfg is None:
        cfg = SimConfig()
    result = run_sim(cfg)
    metrics = compute_metrics(result, cfg)
    return {"result": result, "metrics": metrics, "cfg": cfg}


if __name__ == "__main__":
    # 直接 `python autoscale_sim.py` 时跑一条默认仿真并打印摘要
    out = simulate()
    m = out["metrics"]
    print("=== K8s 自动扩缩容仿真摘要(默认梯形负载)===")
    for k, v in m.items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
