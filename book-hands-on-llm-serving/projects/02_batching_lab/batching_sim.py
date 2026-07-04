# -*- coding: utf-8 -*-
"""
batching_sim.py —— LLM 推理批处理策略离散事件仿真内核
====================================================================

本文件用**纯 Python（只依赖 numpy）** 仿真 LLM 推理服务端的四种批处理策略，
对比它们的**吞吐（throughput）**、**尾延迟（tail latency, p50/p95/p99）**、
**GPU 利用率（利用/空闲）**。不需要 GPU、不需要模型、不需要联网。

为什么能用仿真代替真机？
------------------------------------------------------------
LLM 自回归解码（autoregressive decode）有一个**极其规整**的时间结构：
  1. prefill 阶段：一次性并行处理整个输入 prompt（prompt_len 个 token）；
  2. decode 阶段：**逐 token** 生成，一次前向（forward）出 1 个 token，
     直到生成 out_len 个 token 或遇到 EOS。
每一次前向（无论 prefill 还是 decode）在硬件上是一个**离散的时间步（step / iteration）**。
这正是「iteration-level scheduling（迭代级调度）」这一术语的来源——连续批处理正是
在**每个迭代**结束后做调度决策的。因此我们把「一次前向」当作仿真的**最小时间单元**，
用离散事件推进，就能忠实还原四种策略的行为差异，而无需真的跑一个 7B 模型。

四种策略（对应原书第 6 章 6.1~6.3）：
------------------------------------------------------------
  1. no_batching   —— 不批处理：请求严格串行，一个跑完才跑下一个。基线。
  2. static        —— 静态批处理：**攒够** batch_size 个请求才开跑；一批必须
                      **整批一起完成**（等最慢的那条）才能开下一批。离线友好、在线要命。
  3. dynamic       —— 动态批处理：满 batch_size **或** 到 max_delay 上限就开跑；
                      同样是**整批完成**才开下一批（这是它相比连续批的根本缺陷）。
  4. continuous    —— 连续批处理（又名 in-flight / iterative batching）：
                      **每个迭代**结束后，凡是跑完的请求立刻离场、队列里的新请求
                      立刻补位，GPU 几乎不空转。现代 vLLM/SGLang 的默认策略。

作者：AI-Infra 实战教程 · 配套《Hands-On LLM Serving and Optimization》第 6 章
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


# ====================================================================
# 一、数据结构：请求（Request）与仿真结果（SimResult）
# ====================================================================

@dataclass
class Request:
    """一个推理请求。

    字段含义：
      req_id      : 请求编号（生成顺序）。
      arrival     : 到达时刻（单位：仿真步 step；下同）。请求在此刻才出现在队列里。
      prompt_len  : 输入 prompt 的 token 数（决定 prefill 的工作量）。
      out_len     : 需要生成的输出 token 数（决定 decode 需要多少个迭代）。
      —— 下面几个是仿真过程中被填充的「运行时状态」——
      start       : 请求**首次**被调度进 batch（开始 prefill）的时刻。
      finish      : 请求**完成最后一个 token** 的时刻。
      remaining   : 还剩多少个 decode 迭代没做（初始 = out_len）。
                    prefill 单独用一个标志处理，见 prefilled。
      prefilled   : 是否已完成 prefill。
    """
    req_id: int
    arrival: int
    prompt_len: int
    out_len: int
    # 运行时状态
    start: Optional[int] = None
    finish: Optional[int] = None
    remaining: int = 0
    prefilled: bool = False

    def __post_init__(self) -> None:
        # decode 需要的迭代数 = 输出 token 数（一次前向出一个 token）
        self.remaining = self.out_len

    @property
    def latency(self) -> Optional[int]:
        """端到端延迟 = 完成时刻 - 到达时刻（含排队等待）。未完成则为 None。"""
        if self.finish is None:
            return None
        return self.finish - self.arrival

    @property
    def is_done(self) -> bool:
        return self.finish is not None


@dataclass
class SimResult:
    """一次仿真跑完后的所有指标与时间线，供测试与画图使用。"""
    strategy: str                    # 策略名
    requests: List[Request]          # 所有请求（含运行时状态），用于算延迟
    total_steps: int                 # 总耗时（仿真步）
    busy_steps: int                  # GPU 在「干活」的步数（batch 非空）
    idle_steps: int                  # GPU 空闲步数 = total - busy
    timeline: List[int]              # 每一步 batch 里有多少个活跃请求（画时间线用）
    capacity: int = 8                # GPU 批容量（多少个槽位），用于算「并行利用率」
    # —— 便捷派生指标 ——
    n_requests: int = 0
    completed: int = 0

    def __post_init__(self) -> None:
        self.n_requests = len(self.requests)
        self.completed = sum(1 for r in self.requests if r.is_done)

    # ---------- 派生指标 ----------
    @property
    def throughput(self) -> float:
        """吞吐 = 完成的请求数 / 总耗时（请求/步）。越大越好。"""
        if self.total_steps == 0:
            return 0.0
        return self.completed / self.total_steps

    @property
    def token_throughput(self) -> float:
        """token 吞吐 = 完成的输出 token 总数 / 总耗时。更贴近真实 tokens/s 指标。"""
        if self.total_steps == 0:
            return 0.0
        done_tokens = sum(r.out_len for r in self.requests if r.is_done)
        return done_tokens / self.total_steps

    @property
    def gpu_utilization(self) -> float:
        """时间维利用率 = busy / total（GPU 有没有在「开机干活」）。

        ⚠️ 这个口径**偏乐观**：只要 batch 非空就算 busy，哪怕批里只剩 1 个慢请求
        占着 8 个槽位。它衡量的是「有没有空转」，不衡量「并行度浪费」。
        真正能区分四种策略优劣的是下面的 slot_utilization。
        """
        if self.total_steps == 0:
            return 0.0
        return self.busy_steps / self.total_steps

    @property
    def slot_utilization(self) -> float:
        """并行利用率（slot / capacity 利用率）——本项目衡量 GPU 空闲的**核心口径**。

        物理直觉：GPU 一次前向能并行服务 capacity 个请求（capacity 个「槽位」）。
        每一步真正占用了多少槽位（= timeline 里那一步的活跃请求数），
        对 capacity 取比值再对**整段时间**平均，就是「并行能力被用掉了多少」。

        - 静态/动态批处理：批尾只剩 1~2 个长请求时，其余槽位空着 → 并行利用率被拉低；
          攒批/请求稀疏时干脆整步 0 占用 → 进一步拉低。
        - 连续批处理：跑完一个立刻补一个，槽位常年填满 → 并行利用率最高。
        这正是原书「continuous batching 让 GPU 几乎不空转」的量化体现。
        """
        if self.total_steps == 0 or self.capacity == 0:
            return 0.0
        used = float(np.sum(self.timeline))          # 所有步累计占用的槽位数
        return used / (self.total_steps * self.capacity)

    @property
    def idle_slot_ratio(self) -> float:
        """空闲槽位比例 = 1 - slot_utilization。越小越好（GPU 越不浪费）。"""
        return 1.0 - self.slot_utilization

    def latencies(self) -> np.ndarray:
        """所有已完成请求的端到端延迟数组（用于算 p50/p95/p99）。"""
        lat = [r.latency for r in self.requests if r.is_done]
        return np.asarray(lat, dtype=float)

    def latency_percentile(self, p: float) -> float:
        """第 p 百分位延迟（尾延迟）。p=99 即 p99。无完成请求返回 nan。"""
        lat = self.latencies()
        if lat.size == 0:
            return float("nan")
        # 用线性插值法，和 numpy 默认一致；p 传 50/95/99
        return float(np.percentile(lat, p))

    def metrics_dict(self) -> Dict[str, float]:
        """打包所有关键指标成 dict，方便打印和画柱状图。"""
        return {
            "throughput": self.throughput,
            "token_throughput": self.token_throughput,
            "gpu_utilization": self.gpu_utilization,
            "slot_utilization": self.slot_utilization,
            "idle_slot_ratio": self.idle_slot_ratio,
            "idle_ratio": 1.0 - self.gpu_utilization,
            "p50": self.latency_percentile(50),
            "p95": self.latency_percentile(95),
            "p99": self.latency_percentile(99),
            "mean_latency": float(np.mean(self.latencies())) if self.completed else float("nan"),
            "total_steps": float(self.total_steps),
            "completed": float(self.completed),
        }


# ====================================================================
# 二、工作负载生成（Workload Generation）
# ====================================================================

def make_workload(
    n_requests: int = 60,
    seed: int = 0,
    arrival_rate: float = 0.5,
    prompt_lo: int = 4,
    prompt_hi: int = 40,
    out_mean: float = 20.0,
    out_heavy_ratio: float = 0.15,
    out_heavy_mult: float = 6.0,
) -> List[Request]:
    """生成一批有真实感的推理请求。

    关键点是**输出长度高度不均**：大部分请求短（几十 token），少数「重请求」
    极长（几百 token）。这正是原书反复强调的 LLM 痛点——
    「一个超长请求拖着一整批空等」，也是连续批处理相对静态/动态批处理拉开差距的根源。

    参数：
      n_requests     : 请求总数。
      seed           : 随机种子（保证可复现、测试稳定）。
      arrival_rate   : 到达率（请求/步）。用泊松过程（指数分布间隔）模拟到达。
                       越小 → 请求越稀疏 → 静态批处理「攒批」等待越久（坑越明显）。
      prompt_lo/hi   : prompt 长度均匀分布区间。
      out_mean       : 普通请求输出长度均值（几何分布近似）。
      out_heavy_ratio: 「重请求」占比。
      out_heavy_mult : 重请求输出长度是普通的多少倍。

    返回：按到达时刻升序排列的 Request 列表。
    """
    rng = np.random.default_rng(seed)
    reqs: List[Request] = []
    t = 0.0
    for i in range(n_requests):
        # ---- 到达时刻：泊松过程 = 相邻到达间隔服从指数分布 ----
        # 间隔均值 = 1 / arrival_rate。取整到最近的仿真步。
        gap = rng.exponential(1.0 / arrival_rate)
        t += gap
        arrival = int(round(t))

        # ---- prompt 长度：均匀分布 ----
        prompt_len = int(rng.integers(prompt_lo, prompt_hi + 1))

        # ---- 输出长度：多数短、少数极长（重尾）----
        if rng.random() < out_heavy_ratio:
            # 重请求：均值放大 out_heavy_mult 倍
            base = rng.geometric(1.0 / (out_mean * out_heavy_mult))
        else:
            base = rng.geometric(1.0 / out_mean)
        out_len = int(max(1, base))

        reqs.append(Request(req_id=i, arrival=arrival,
                            prompt_len=prompt_len, out_len=out_len))
    # 保证按到达时刻排序（泊松累加本就单调，稳妥起见再排一次）
    reqs.sort(key=lambda r: (r.arrival, r.req_id))
    return reqs


def _clone(reqs: List[Request]) -> List[Request]:
    """深拷贝请求列表，重置运行时状态。

    ⚠️ 坑：四种策略必须跑在**同一份工作负载**上才公平可比，
    但每次仿真都会往 Request 里写 start/finish 等状态。所以每个策略开跑前
    都要 clone 一份干净的请求，互不污染。
    """
    out = []
    for r in reqs:
        nr = Request(req_id=r.req_id, arrival=r.arrival,
                    prompt_len=r.prompt_len, out_len=r.out_len)
        out.append(nr)
    return out


# ====================================================================
# 三、成本模型（Cost Model）：一次前向要多久？
# ====================================================================

def step_cost(batch: List[Request]) -> int:
    """一次前向（迭代）耗多少「步」。

    在真实 GPU 上，一个批次的一次前向耗时≈常数（受显存带宽/权重加载主导），
    **和 batch 内请求数弱相关**——这正是 batching 提高吞吐的物理根源：
    多个请求「搭同一趟前向的便车」，把固定的权重加载成本摊薄。

    为了让仿真简单且忠实，我们令**每次前向固定耗 1 步**，无论 batch 里有几个请求。
    于是：
      - batching 的收益 = 「同一步内并行完成多个请求的 1 个 token」；
      - 空转的代价 = 「某一步 batch 为空 / 只有 1 个慢请求，浪费了并行能力」。
    这个抽象足以复现四种策略的**相对**优劣（本项目的目的），
    真实工程里前向耗时还受 token 总数、KV cache、算子实现影响（见 README 的坑）。
    """
    return 1 if batch else 0


# ====================================================================
# 四、四种批处理策略的仿真实现
# ====================================================================

def simulate_no_batching(reqs: List[Request], capacity: int = 8) -> SimResult:
    """策略①：不批处理（串行基线）。

    capacity 只用于**统一并行利用率的分母**：硬件本可并行 capacity 个请求，
    但串行策略每步至多占 1 个槽位，因此 slot_utilization 会非常低——
    这正是「完全不 batch 有多浪费」的量化。

    每个请求**独占** GPU：先 prefill（1 步），再 decode out_len 步，
    完全跑完才处理下一个。请求必须等到前一个彻底结束 + 自己已到达，才开始。

    这是最差的吞吐、但每个请求一旦开始就没有「和别人共享」的干扰。
    它是对照组——用来量化「批处理到底带来多少提升」。
    """
    work = _clone(reqs)
    t = 0                         # 当前仿真时刻
    busy = 0                      # 累计干活步数
    timeline: List[int] = []
    for r in sorted(work, key=lambda x: (x.arrival, x.req_id)):
        # GPU 可能提前空闲，但请求还没到 → 快进到到达时刻（这段是 idle）
        if t < r.arrival:
            # 记录空闲步
            for _ in range(r.arrival - t):
                timeline.append(0)
            t = r.arrival
        # prefill：1 步
        r.start = t
        timeline.append(1)
        t += 1
        busy += 1
        # decode：out_len 步，每步 1 个 token
        for _ in range(r.out_len):
            timeline.append(1)
            t += 1
            busy += 1
        r.finish = t                # 完成时刻 = 最后一个 token 出来之后
    total = t
    return SimResult(strategy="no_batching", requests=work,
                    total_steps=total, busy_steps=busy,
                    idle_steps=total - busy, timeline=timeline,
                    capacity=capacity)


def simulate_static(reqs: List[Request], batch_size: int = 8) -> SimResult:
    """策略②：静态批处理（static batching）。

    规则（对应原书 6.2 的「傻等」）：
      - 服务端**死等**，直到队列里**攒够** batch_size 个请求，才开跑一批；
      - 队列耗尽（最后不足一批）时，把剩下的凑一批开跑（否则永远发不出去）；
      - 一批一旦开跑，**整批一起前向**，直到批里**最慢**（out_len 最大）的请求
        也完成，才算这批结束，才能开下一批。

    两个致命缺陷都会在指标里现形：
      (a) 攒批等待 → 先到的请求干等后面的人来，排队延迟大；
      (b) 整批等最慢 → 短请求早就跑完了还得陪着最长的那条，GPU 在这段
          只服务 1 个请求却占着整批的位置 → **利用率低、尾延迟高**。
    """
    work = _clone(reqs)
    arrivals = sorted(work, key=lambda x: (x.arrival, x.req_id))
    idx = 0                       # 下一个待入队的请求下标
    t = 0
    busy = 0
    timeline: List[int] = []
    pending: List[Request] = []   # 已到达、还没被组批的请求

    def advance_to(t_target: int) -> None:
        """快进到 t_target，期间 GPU 空闲，记录 idle 步。"""
        nonlocal t
        while t < t_target:
            timeline.append(0)
            t += 1

    n = len(arrivals)
    while idx < n or pending:
        # 1) 把「此刻及之前已到达」的请求全部入队
        while idx < n and arrivals[idx].arrival <= t:
            pending.append(arrivals[idx])
            idx += 1

        # 2) 攒批：不够一批且后面还有请求没到 → 快进到下一个到达时刻继续攒
        if len(pending) < batch_size and idx < n:
            advance_to(arrivals[idx].arrival)
            continue

        # 3) 若此刻队列为空（所有已到达都处理完，但还有未来请求）→ 快进
        if not pending:
            if idx < n:
                advance_to(arrivals[idx].arrival)
                continue
            else:
                break

        # 4) 取一批（最多 batch_size 个）开跑
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        for r in batch:
            r.start = t
            r.prefilled = False
        # prefill：整批一起 1 步
        timeline.append(len(batch))
        t += 1
        busy += 1
        for r in batch:
            r.prefilled = True
        # decode：跑到批里最长的请求也完成。整批同步推进。
        max_out = max(r.out_len for r in batch)
        for k in range(max_out):
            # 这一步里还「活着」（尚未生成够 out_len）的请求数
            active = sum(1 for r in batch if r.out_len > k)
            timeline.append(active)
            t += 1
            busy += 1
            # 谁在这一步做完了最后一个 token，就在这一步结束时记 finish
            for r in batch:
                if r.out_len == k + 1:
                    r.finish = t
    total = t
    return SimResult(strategy="static", requests=work,
                    total_steps=total, busy_steps=busy,
                    idle_steps=total - busy, timeline=timeline,
                    capacity=batch_size)


def simulate_dynamic(reqs: List[Request], batch_size: int = 8,
                    max_delay: int = 3) -> SimResult:
    """策略③：动态批处理（dynamic batching）。

    相比静态，多了一个 **max_delay（最大等待时间）** 参数（原书 6.2 的两大关键参数
    之一，另一个是 batch size）。规则：
      - 只要满足**任一**条件就开跑一批：
          (a) 队列已攒够 batch_size 个请求；**或**
          (b) 队列里**最早**到达的请求已经等了 >= max_delay 步（超时兜底）。
      - 一批依旧是**整批同步、等最慢**才结束（这一点和静态相同！所以 LLM 场景
        仍有「等最慢」的空转缺陷，需要连续批处理才根治——见原书 6.3）。

    效果：靠 max_delay 缓解了「攒批干等」的排队延迟，吞吐/尾延迟通常介于
    静态与连续之间。
    """
    work = _clone(reqs)
    arrivals = sorted(work, key=lambda x: (x.arrival, x.req_id))
    idx = 0
    t = 0
    busy = 0
    timeline: List[int] = []
    pending: List[Request] = []

    def advance_to(t_target: int) -> None:
        nonlocal t
        while t < t_target:
            timeline.append(0)
            t += 1

    n = len(arrivals)
    while idx < n or pending:
        while idx < n and arrivals[idx].arrival <= t:
            pending.append(arrivals[idx])
            idx += 1

        if not pending:
            if idx < n:
                advance_to(arrivals[idx].arrival)
                continue
            else:
                break

        # 判断是否满足开跑条件
        full = len(pending) >= batch_size
        oldest_wait = t - pending[0].arrival
        timed_out = oldest_wait >= max_delay

        if not (full or timed_out):
            # 还没到开跑条件：要么快进到下一个到达（可能凑满），
            # 要么快进到「最早请求超时」的时刻，取更早者。
            next_arrival = arrivals[idx].arrival if idx < n else None
            deadline = pending[0].arrival + max_delay
            targets = [x for x in (next_arrival, deadline) if x is not None and x > t]
            if targets:
                advance_to(min(targets))
                continue
            # 兜底：直接开跑
        # 开跑一批
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        for r in batch:
            r.start = t
        timeline.append(len(batch))
        t += 1
        busy += 1
        max_out = max(r.out_len for r in batch)
        for k in range(max_out):
            active = sum(1 for r in batch if r.out_len > k)
            timeline.append(active)
            t += 1
            busy += 1
            for r in batch:
                if r.out_len == k + 1:
                    r.finish = t
    total = t
    return SimResult(strategy="dynamic", requests=work,
                    total_steps=total, busy_steps=busy,
                    idle_steps=total - busy, timeline=timeline,
                    capacity=batch_size)


def simulate_continuous(reqs: List[Request], max_batch_size: int = 8) -> SimResult:
    """策略④：连续批处理（continuous / in-flight / iterative batching）。⭐

    核心与前三者的根本不同（原书 6.3 的精髓）：
      - **调度发生在每一个迭代（step）结束后**，而不是每一批结束后；
      - batch 是一个「运行中的槽位集合」，最多 max_batch_size 个槽位；
      - **每一步**：给槽位里所有请求各推进 1 个 token；
        谁完成了（生成够 out_len）就**立刻离场**，腾出的槽位在**下一步开始前**
        立刻从队列里补新请求进来。→ 无「等最慢」空等、无「攒批」干等。

    这里我们用一个统一的时间步循环，混合处理 prefill 与 decode：
    为忠实又不过度复杂，新请求进入槽位后**第一步做 prefill**，之后每步做 1 个 decode。
    prefill 也占该请求在这一步的位置（真实引擎里 prefill 与 decode 可同批，
    即 chunked prefill / 混批，见原书 6.4，这里做了合理简化）。

    这就是 vLLM/SGLang 的默认行为，也是它们相对传统服务框架吞吐能到数倍~数十倍
    （Anyscale 2023：最高 23×）的关键。
    """
    work = _clone(reqs)
    arrivals = sorted(work, key=lambda x: (x.arrival, x.req_id))
    idx = 0
    t = 0
    busy = 0
    timeline: List[int] = []
    running: List[Request] = []   # 当前在槽位里的请求
    n = len(arrivals)

    def fill_slots() -> None:
        """把已到达、还在等待的请求补进空槽位（直到满或没有可补的）。"""
        nonlocal idx
        while len(running) < max_batch_size and idx < n and arrivals[idx].arrival <= t:
            r = arrivals[idx]
            r.start = t
            r.prefilled = False       # 进来后第一步做 prefill
            running.append(r)
            idx += 1

    while idx < n or running:
        # 1) 补位：把此刻能进的请求填进空槽
        fill_slots()

        # 2) 若槽位空且还有未来请求没到 → 快进到下一个到达（这段 GPU idle）
        if not running:
            if idx < n:
                # 快进
                target = arrivals[idx].arrival
                while t < target:
                    timeline.append(0)
                    t += 1
                continue
            else:
                break

        # 3) 执行一步前向：槽位里每个请求推进一次
        timeline.append(len(running))
        t += 1
        busy += 1
        still: List[Request] = []
        for r in running:
            if not r.prefilled:
                # 这一步是 prefill：完成后本步不出 decode token（简化）
                r.prefilled = True
                still.append(r)
            else:
                # decode：出 1 个 token
                r.remaining -= 1
                if r.remaining <= 0:
                    r.finish = t          # 完成，立刻离场
                else:
                    still.append(r)
        running = still
        # 循环顶端会立刻 fill_slots()，实现「跑完一个补一个」

    total = t
    return SimResult(strategy="continuous", requests=work,
                    total_steps=total, busy_steps=busy,
                    idle_steps=total - busy, timeline=timeline,
                    capacity=max_batch_size)


# ====================================================================
# 五、一键跑全部策略
# ====================================================================

def run_all(reqs: List[Request], batch_size: int = 8,
            max_delay: int = 3) -> Dict[str, SimResult]:
    """在同一份工作负载上跑四种策略，返回 {策略名: SimResult}。"""
    return {
        "no_batching": simulate_no_batching(reqs, capacity=batch_size),
        "static": simulate_static(reqs, batch_size=batch_size),
        "dynamic": simulate_dynamic(reqs, batch_size=batch_size, max_delay=max_delay),
        "continuous": simulate_continuous(reqs, max_batch_size=batch_size),
    }


if __name__ == "__main__":
    # 快速自检：跑一遍并打印关键指标
    wl = make_workload(n_requests=60, seed=0)
    results = run_all(wl)
    header = (f"{'策略':<12}{'吞吐':>8}{'token吞吐':>10}{'并行利用率':>11}"
              f"{'p50':>7}{'p95':>7}{'p99':>7}{'完成':>6}")
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        m = res.metrics_dict()
        print(f"{name:<12}{m['throughput']:>8.3f}{m['token_throughput']:>10.2f}"
              f"{m['slot_utilization']:>11.2%}"
              f"{m['p50']:>7.1f}{m['p95']:>7.1f}{m['p99']:>7.1f}{int(m['completed']):>6}")
