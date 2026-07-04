"""
pd_sim.py —— PD 分离 vs 合置 的推理服务调度模拟器(离散时间 tick 仿真)

对应 ../../01_PD分离架构.md。用一个简化但能抓住本质的模型,量化:
  · 合置 colocated:prefill 和 decode 抢同一批 GPU slot;prefill 突发时 decode 会"卡顿"(TPOT 尖峰)
  · 分离 disaggregated:prefill 池与 decode 池独立;decode 永不被 prefill 阻塞(TPOT 平稳)

模型(一个 tick = 一个 decode 步,DECODE_STEP 毫秒):
  · prefill 是算力受限的:耗时 ∝ prompt_len(占用一个 slot 若干 tick)
  · decode 是访存受限的:每 tick 让所有活跃 decode 请求各 +1 token(连续批处理),
    但**只有当本 tick 有空闲 slot 时**才能推进;slot 全被 prefill 占满 → decode 停一拍
  · 分离时 prefill→decode 之间有 KV cache 传输开销 ∝ prompt_len
金标准(pytest):分离下 decode 的最大步间隔 ≈ 一个 tick(不被阻塞);合置下会出现 > 一个 tick 的停顿。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

DECODE_STEP = 1.0            # 一个 decode tick 的时长(ms)
PREFILL_PER_TOK = 0.5        # 每个 prompt token 的 prefill 耗时(ms)——算力受限
KV_TRANSFER_PER_TOK = 0.02   # 分离时每 token 的 KV 传输耗时(ms)


@dataclass
class Request:
    rid: int
    arrival: float
    prompt_len: int
    output_len: int
    # 结果
    ttft: float = -1.0                       # 首 token 延迟
    token_times: list = field(default_factory=list)  # 每个输出 token 的产出时刻
    done_time: float = -1.0

    def prefill_ticks(self):
        ms = self.prompt_len * PREFILL_PER_TOK
        return max(1, math.ceil(ms / DECODE_STEP))

    def tpot(self):
        """每 token 延迟 = 相邻 token 产出间隔的平均。"""
        if len(self.token_times) < 2:
            return DECODE_STEP
        d = [b - a for a, b in zip(self.token_times, self.token_times[1:])]
        return sum(d) / len(d)

    def max_decode_gap(self):
        if len(self.token_times) < 2:
            return DECODE_STEP
        return max(b - a for a, b in zip(self.token_times, self.token_times[1:]))


class _Prefill:
    __slots__ = ("req", "remaining")
    def __init__(self, req):
        self.req = req; self.remaining = req.prefill_ticks()


def _run(requests, n_prefill_slots, n_decode_slots, colocated, max_ticks=100000):
    """
    colocated=True:  n_prefill_slots 忽略,所有 slot = n_decode_slots,prefill/decode 共用。
    colocated=False: prefill 用 n_prefill_slots(独立),decode 用 n_decode_slots(独立,不被阻塞)。
    """
    reqs = sorted(requests, key=lambda r: r.arrival)
    for r in reqs:
        r.ttft = -1.0; r.token_times = []; r.done_time = -1.0
    pending = list(reqs)                    # 尚未开始 prefill
    prefilling = []                         # 进行中的 prefill
    decoding = []                           # 活跃 decode 请求
    ready_after_transfer = []               # (ready_time, req) 分离时等 KV 传输
    total_slots = n_decode_slots if colocated else n_prefill_slots
    t = 0
    while (pending or prefilling or decoding or ready_after_transfer) and t < max_ticks:
        now = t * DECODE_STEP
        # 1) 到达且有 prefill 容量 → 开始 prefill
        cap = (total_slots if colocated else n_prefill_slots) - len(prefilling)
        i = 0
        while i < len(pending) and cap > 0:
            if pending[i].arrival <= now:
                prefilling.append(_Prefill(pending.pop(i))); cap -= 1
            else:
                i += 1
        # 2) 推进 prefill;完成的进入 decode(分离时先经 KV 传输延迟)
        still = []
        for pf in prefilling:
            pf.remaining -= 1
            if pf.remaining <= 0:
                pf.req.ttft = now + DECODE_STEP - pf.req.arrival    # 首 token 约在 prefill 结束
                if colocated:
                    decoding.append(pf.req)
                else:
                    delay = pf.req.prompt_len * KV_TRANSFER_PER_TOK
                    ready_after_transfer.append((now + delay, pf.req))
            else:
                still.append(pf)
        prefilling = still
        # 分离:KV 传输完成的进入 decode
        if not colocated:
            still_transfer = []
            for ready, req in ready_after_transfer:
                if ready <= now:
                    decoding.append(req)
                else:
                    still_transfer.append((ready, req))
            ready_after_transfer = still_transfer
        # 3) decode:本 tick 是否有空闲 slot 推进 decode?
        if colocated:
            busy = len(prefilling)                     # prefill 占用的 slot
            free = total_slots - busy
            can_decode = free >= 1                     # 有空闲 slot 才能 decode
        else:
            can_decode = n_decode_slots >= 1           # 独立 decode 池,永远能
        if can_decode:
            finished = []
            for req in decoding:
                req.token_times.append(now + DECODE_STEP)
                if len(req.token_times) >= req.output_len:
                    req.done_time = now + DECODE_STEP
                    finished.append(req)
            for req in finished:
                decoding.remove(req)
        t += 1
    makespan = max((r.done_time for r in reqs if r.done_time > 0), default=0.0)
    total_tokens = sum(len(r.token_times) for r in reqs)
    return {
        "requests": reqs,
        "makespan_ms": makespan,
        "throughput_tok_per_s": (total_tokens / makespan * 1000) if makespan > 0 else 0.0,
        "p99_tpot": _p99([r.tpot() for r in reqs]),
        "max_decode_gap": max((r.max_decode_gap() for r in reqs), default=DECODE_STEP),
        "mean_ttft": sum(r.ttft for r in reqs) / len(reqs),
    }


def simulate_colocated(requests, n_slots):
    return _run(requests, 0, n_slots, colocated=True)


def simulate_disaggregated(requests, n_prefill, n_decode):
    return _run(requests, n_prefill, n_decode, colocated=False)


def _p99(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.99 * len(s)) - 1))]
