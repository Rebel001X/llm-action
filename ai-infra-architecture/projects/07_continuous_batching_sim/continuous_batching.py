"""
continuous_batching.py —— 连续批处理 vs 静态批处理 的推理服务调度模拟器
==================================================================

对应 ../../09_推理引擎架构_连续批处理_调度_投机解码_chunked_prefill.md。

用一个**离散步(iteration/step)仿真**,把两种批处理策略的本质差异量化出来:

  · 静态批处理 static batching:凑一批 → 整批一起跑 → **必须等批内最慢的那条请求
    (output_len 最大)跑完,整批才能退休**;跑的过程中先完成的短请求,其 slot 只能
    "空转/padding" 等着;**新到达的请求也进不来**,得等整批 drain 完才能组下一批。
    → 两大浪费:① 批内长度方差导致的空 slot;② 队头阻塞(HOL),新请求排大队。

  · 连续批处理 continuous batching(又名 iteration-level scheduling / in-flight batching):
    **每一步(每个 decode iteration)结束后,谁完成谁就立刻让出 slot,等待队列里的新请求
    马上补进来**。batch 像"流水线"一样被持续填满 → GPU 空转少、吞吐高、排队短。
    这正是 vLLM / TGI / TensorRT-LLM 的核心调度机制。

两个模拟器共用同一套「按步推进」的引擎,唯一区别是**准入策略**:
  · 连续:每一步都把 batch 回填到 batch_size(work-conserving,不浪费空 slot)。
  · 静态:只有当前批**彻底跑空**(running 为空)时,才一次性准入下一批。

金标准(pytest,见 tests/):
  1) 不变量:所有请求都产出全部 token、都有合法的 ttft/done。
  2) 连续吞吐 ≥ 静态吞吐(连续是 work-conserving,makespan 更短)。
  3) 连续的 GPU 空闲 slot-steps ≤ 静态(连续把空 slot 立刻回填)。
  4) 连续的尾延迟 / 平均排队延迟 优于静态。

时间单位:一个 step = STEP_MS 毫秒(默认 1ms)。一步 = 引擎跑一次 forward。
本模块纯 Python、零第三方依赖、秒级可跑。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import math

# ------------------------------ 可调参数 ------------------------------
STEP_MS = 1.0                    # 一个 iteration(step)的时长(ms)
PREFILL_TOKENS_PER_STEP = 128    # 一步能 prefill 多少个 prompt token(算力受限的近似)


# ------------------------------ 请求对象 ------------------------------
@dataclass
class Request:
    """一条推理请求。arrival 为到达时刻(ms);prompt_len / output_len 为 token 数。"""
    rid: int
    arrival: float
    prompt_len: int
    output_len: int
    # ---- 仿真产出(reset 后填充) ----
    start: float = -1.0                              # 被准入(开始占 slot)的时刻
    ttft: float = -1.0                               # 首 token 延迟 = 首 token 时刻 - arrival
    token_times: list = field(default_factory=list)  # 每个输出 token 的产出时刻(ms)
    done: float = -1.0                               # 最后一个 token 的时刻(完成时刻)

    # prefill 需要几步:∝ prompt_len,算力受限;至少 1 步
    def prefill_steps(self) -> int:
        return max(1, math.ceil(self.prompt_len / PREFILL_TOKENS_PER_STEP))

    # 一条请求占用 slot 的总步数 = prefill 步 + decode 步(每步出 1 个 token)
    def service_steps(self) -> int:
        return self.prefill_steps() + self.output_len

    # 端到端延迟 = 完成时刻 - 到达时刻
    def latency(self) -> float:
        return self.done - self.arrival

    # 排队延迟 = 被准入时刻 - 到达时刻(纯粹"等着上车"的时间)
    def queue_delay(self) -> float:
        return self.start - self.arrival

    def clone(self) -> "Request":
        return Request(self.rid, self.arrival, self.prompt_len, self.output_len)


class _Slot:
    """batch 里的一个活跃槽位:包裹一条请求 + 剩余步数。"""
    __slots__ = ("req", "remaining")

    def __init__(self, req: Request):
        self.req = req
        self.remaining = req.service_steps()


# ------------------------------ 仿真结果 ------------------------------
@dataclass
class SimResult:
    name: str
    requests: list
    occupancy: list          # 每个"真实运行步"的活跃 slot 数(用于时间线/利用率)
    first_start: float       # 第一条请求被准入的时刻
    makespan: float          # 最后一条请求完成的时刻
    batch_size: int
    step_ms: float

    # ---- 派生指标 ----
    def window_steps(self) -> int:
        """服务窗口 [first_start, makespan] 覆盖的总步数(含中途 GPU 空闲步)。"""
        return int(round((self.makespan - self.first_start) / self.step_ms))

    def busy_slot_steps(self) -> int:
        """被真实占用的 slot·步 数(分子)。"""
        return int(sum(self.occupancy))

    def total_slot_steps(self) -> int:
        """理论可用的 slot·步 数(分母)= 窗口步数 × batch_size。"""
        return self.window_steps() * self.batch_size

    def idle_slot_steps(self) -> int:
        """空闲(浪费)的 slot·步 数 = 可用 - 占用。越小越好。"""
        return self.total_slot_steps() - self.busy_slot_steps()

    def gpu_util(self) -> float:
        """GPU 利用率 = 占用 slot·步 / 可用 slot·步(0~1)。"""
        d = self.total_slot_steps()
        return self.busy_slot_steps() / d if d > 0 else 0.0

    def total_output_tokens(self) -> int:
        return sum(len(r.token_times) for r in self.requests)

    def throughput_tok_s(self) -> float:
        """吞吐(tokens/s):服务窗口内产出的总 token / 窗口时长。"""
        w = self.makespan - self.first_start
        return self.total_output_tokens() / w * 1000.0 if w > 0 else 0.0

    def throughput_req_s(self) -> float:
        w = self.makespan - self.first_start
        return len(self.requests) / w * 1000.0 if w > 0 else 0.0

    def latencies(self) -> list:
        return [r.latency() for r in self.requests]

    def queue_delays(self) -> list:
        return [r.queue_delay() for r in self.requests]

    def ttfts(self) -> list:
        return [r.ttft for r in self.requests]


def _percentile(xs: list, q: float) -> float:
    """最近秩(nearest-rank)百分位。q∈[0,1]。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = int(math.ceil(q * len(s)) - 1)
    return s[min(max(idx, 0), len(s) - 1)]


def _schedule_tokens(r: Request, start: float, step_ms: float) -> None:
    """给一条请求排定 ttft / token_times / done(start = 被准入时刻)。

    时间线:占 slot 后先做 prefill_steps 步(不出 token),随后每 decode 步出 1 个 token。
      · 首个 decode token 在 start + (prefill_steps+1)*step 产出 → 定义 ttft。
      · 第 k 个 token(k 从 0 计)在 start + (prefill_steps+1+k)*step。
      · 最后一个 token 时刻 = start + service_steps*step = done。
    这套记账在两个模拟器里**完全一致**,唯一变量是 start(被准入时刻)——
    这正是"静态 vs 连续"的差异所在:连续能更早准入 → start 更早 → 延迟更低。
    """
    pf = r.prefill_steps()
    r.start = start
    r.token_times = [start + (pf + 1 + k) * step_ms for k in range(r.output_len)]
    r.ttft = (r.token_times[0] - r.arrival) if r.output_len > 0 else 0.0
    r.done = r.token_times[-1] if r.output_len > 0 else start + pf * step_ms


def _simulate(requests, batch_size, mode, step_ms=STEP_MS, max_steps=10_000_000):
    """按步推进的统一引擎。mode ∈ {"continuous", "static"}。

    每一步:① 收入到达的请求进等待队列 → ② 按策略准入 → ③ 判断终止/空转跳步
             → ④ 跑一步(所有活跃 slot 各推进 1 步,完成的让出 slot)。
    """
    assert batch_size >= 1, "batch_size 至少为 1"
    assert mode in ("continuous", "static")
    reqs = sorted((r.clone() for r in requests), key=lambda r: r.arrival)

    p = 0                       # 指向下一个尚未到达的请求
    waiting = deque()           # 已到达、尚未准入(FIFO)
    running = []                # 活跃 slot 列表
    occupancy = []              # 每个真实运行步的活跃 slot 数
    first_start = None
    t = 0                       # 当前步号

    while t < max_steps:
        now = t * step_ms
        # ① 收入到达的请求(arrival <= now)
        while p < len(reqs) and reqs[p].arrival <= now + 1e-9:
            waiting.append(reqs[p])
            p += 1
        # ② 准入策略:连续=每步回填到满;静态=仅当整批跑空才准入下一批
        can_admit = (mode == "continuous") or (len(running) == 0)
        if can_admit:
            while len(running) < batch_size and waiting:
                r = waiting.popleft()
                _schedule_tokens(r, now, step_ms)
                if first_start is None:
                    first_start = now
                running.append(_Slot(r))
        # ③ 终止 / 空转跳步
        if not running:
            if p >= len(reqs) and not waiting:
                break                                   # 全部完成
            if not waiting:                             # GPU 空闲,直接跳到下一个到达
                t = int(math.ceil(reqs[p].arrival / step_ms))
                continue
        # ④ 跑一步:先统计占用,再推进、退休完成者
        occupancy.append(len(running))
        finished = []
        for slot in running:
            slot.remaining -= 1
            if slot.remaining <= 0:
                finished.append(slot)
        if finished:
            fin = set(id(s) for s in finished)
            running = [s for s in running if id(s) not in fin]
        t += 1

    makespan = max((r.done for r in reqs), default=0.0)
    if first_start is None:
        first_start = 0.0
    return SimResult(
        name="continuous" if mode == "continuous" else "static",
        requests=reqs, occupancy=occupancy, first_start=first_start,
        makespan=makespan, batch_size=batch_size, step_ms=step_ms,
    )


# ------------------------------ 对外 API ------------------------------
def simulate_static(requests, batch_size, step_ms=STEP_MS) -> SimResult:
    """静态批处理:凑一批一起跑,等最慢的跑完整批才退休,期间不准入新请求。"""
    res = _simulate(requests, batch_size, "static", step_ms)
    res.name = "静态批处理"
    return res


def simulate_continuous(requests, batch_size, step_ms=STEP_MS) -> SimResult:
    """连续批处理:每步谁完成谁让出 slot,等待队列立即补入(iteration-level)。"""
    res = _simulate(requests, batch_size, "continuous", step_ms)
    res.name = "连续批处理"
    return res


def summarize(res: SimResult) -> dict:
    """把一次仿真压成一行关键指标,便于打印/对比/画图。"""
    lat = res.latencies()
    q = res.queue_delays()
    return {
        "name": res.name,
        "makespan_ms": res.makespan,
        "throughput_tok_s": res.throughput_tok_s(),
        "throughput_req_s": res.throughput_req_s(),
        "gpu_util": res.gpu_util(),
        "idle_slot_steps": res.idle_slot_steps(),
        "busy_slot_steps": res.busy_slot_steps(),
        "mean_latency": sum(lat) / len(lat) if lat else 0.0,
        "p50_latency": _percentile(lat, 0.50),
        "p95_latency": _percentile(lat, 0.95),
        "p99_latency": _percentile(lat, 0.99),
        "mean_queue": sum(q) / len(q) if q else 0.0,
        "p99_queue": _percentile(q, 0.99),
        "mean_ttft": sum(res.ttfts()) / len(res.ttfts()) if res.requests else 0.0,
    }
