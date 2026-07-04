# -*- coding: utf-8 -*-
"""
slo_harness.py —— LLM 服务 SLO / 指标 harness 的核心模块。

本模块干三件事：
  1) 用一个「离散事件仿真」（discrete-event simulation）模拟一个 LLM 推理服务：
     请求按泊松过程到达，服务器有 `num_workers` 个并发槽位（对应「连续批处理 continuous
     batching」里能同时在飞的序列数），每个请求的耗时 = prefill 时间 + decode 时间。
  2) 记录每个请求的关键时间戳，算出工业界最常用的服务指标：
     TTFT / TPOT / E2E 延迟 的 p50/p95/p99、吞吐（throughput）、
     goodput（满足 SLO 的「有效吞吐」比例）、worker 利用率（≈ GPU 利用率的代理指标）。
  3) 把上述指标封装成不依赖第三方库的纯 Python 数据结构，方便测试与画图。

设计目标：**零依赖**（仅用标准库 + 可选 numpy 做对照），本机 CPU 秒级跑完，结果可复现。

术语中英对照（面试高频）：
  - TTFT  = Time To First Token       首 token 时间 —— 由 prefill 决定，影响「响应快不快」。
  - TPOT  = Time Per Output Token     每 token 时间 —— 由 decode 决定，影响「吐字流不流畅」。
  - E2E   = End-to-End latency        端到端延迟 = 排队 + prefill + decode。
  - QPS   = Queries Per Second        每秒请求数（到达率 λ）。
  - goodput= 满足 SLO 的有效吞吐比例  —— 「吞吐再高，违反 SLO 也是废票」。

作者：AI-Infra 实战项目（配套《Hands-On LLM Serving and Optimization》第 4~5 章）。
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ======================================================================
# 1. 百分位（percentile）—— 一切延迟指标的基石
# ======================================================================
def percentile(data: Sequence[float], q: float) -> float:
    """
    计算一组数据的第 q 百分位（q ∈ [0, 100]），采用「线性插值」法
    （与 numpy.percentile 默认的 'linear' / method='linear' 一致）。

    为什么要自己实现？—— 面试里「手写 p99」是高频题；而且我们要在测试里
    拿它和 numpy 对拍，证明实现正确。

    算法（第一性原理）：
      1) 把数据升序排序，得到 x[0] <= x[1] <= ... <= x[n-1]。
      2) 第 q 百分位落在「秩（rank）」pos = (n-1) * q/100 处。
         注意用 n-1：因为下标从 0 到 n-1，两端分别对应 0% 和 100%。
      3) pos 一般不是整数，设 lo = floor(pos), hi = ceil(pos)，
         在 x[lo] 与 x[hi] 之间按小数部分 frac 做线性插值。

    :param data: 一维数值序列（不必预先排序）。
    :param q:    百分位，0~100。q=50 即中位数 p50。
    :return:     第 q 百分位的值。
    """
    if not data:
        raise ValueError("percentile() 需要非空数据")
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"q 必须在 [0,100]，收到 {q}")

    xs = sorted(data)
    n = len(xs)
    if n == 1:
        return float(xs[0])

    # pos：第 q 百分位在「下标空间」里的位置（可能是小数）
    pos = (n - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        # 正好落在某个整数下标上，无需插值
        return float(xs[lo])
    frac = pos - lo  # 小数部分，∈ (0,1)
    # 线性插值：越靠近 hi，hi 的权重越大
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def percentiles(data: Sequence[float], qs: Sequence[float]) -> Dict[float, float]:
    """一次性算多个百分位，返回 {q: value}。内部对数据只排序一次（省时）。"""
    if not data:
        raise ValueError("percentiles() 需要非空数据")
    xs = sorted(data)
    n = len(xs)
    out: Dict[float, float] = {}
    for q in qs:
        if not (0.0 <= q <= 100.0):
            raise ValueError(f"q 必须在 [0,100]，收到 {q}")
        if n == 1:
            out[q] = float(xs[0])
            continue
        pos = (n - 1) * (q / 100.0)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            out[q] = float(xs[lo])
        else:
            frac = pos - lo
            out[q] = float(xs[lo] * (1.0 - frac) + xs[hi] * frac)
    return out


# ======================================================================
# 2. 请求模型：一个请求要跑多久？
# ======================================================================
@dataclass
class Request:
    """
    一个推理请求的「画像」。真实系统里这些量来自 tokenizer 与硬件；
    这里我们用简单模型合成，但保留了「prefill 随输入长度、decode 随输出长度」的本质。

    字段：
      req_id       : 请求编号（到达顺序）。
      arrival      : 到达时刻（秒）。
      prompt_len   : 输入 token 数（prefill 规模）。
      output_len   : 输出 token 数（decode 步数）。
      prefill_time : 该请求的 prefill 耗时（秒）—— 决定 TTFT 的主体。
      decode_time  : 该请求的 decode 总耗时（秒）—— = output_len * 每 token 时间。

    运行时才填的字段（仿真过程中回填）：
      start        : 真正开始被服务的时刻（= 拿到 worker 槽位的时刻）。
      first_token  : 首 token 产出时刻（= start + prefill_time）。
      finish       : 请求完成时刻（= first_token + decode_time）。
    """
    req_id: int
    arrival: float
    prompt_len: int
    output_len: int
    prefill_time: float
    decode_time: float

    start: float = -1.0
    first_token: float = -1.0
    finish: float = -1.0

    # ---- 便捷的延迟视图（服务完成后才有意义）----
    @property
    def wait(self) -> float:
        """排队等待时间 = 开始服务 - 到达。反映拥塞程度。"""
        return self.start - self.arrival

    @property
    def ttft(self) -> float:
        """TTFT = 首 token 时刻 - 到达 = 排队 + prefill。用户「等了多久看到第一个字」。"""
        return self.first_token - self.arrival

    @property
    def tpot(self) -> float:
        """
        TPOT = decode 阶段平均每个输出 token 的时间。
        output_len<=1 时 decode 无「后续 token」，约定返回 decode_time 本身（不为 0，避免除零）。
        """
        if self.output_len <= 1:
            return self.decode_time
        # 首 token 由 prefill 产出，后续 (output_len-1) 个 token 由 decode 产出
        return self.decode_time / (self.output_len - 1)

    @property
    def e2e(self) -> float:
        """端到端延迟 = 完成时刻 - 到达时刻 = 排队 + prefill + decode。"""
        return self.finish - self.arrival


# ======================================================================
# 3. 负载生成器：合成一条到达流
# ======================================================================
@dataclass
class WorkloadConfig:
    """
    合成负载的配置。默认值大致对标一个「7B 模型 / 单卡」量级的服务，
    但数值本身不重要，重要的是**趋势正确**（QPS↑ → 尾延迟↑，见 README）。

    qps                : 平均到达率 λ（请求/秒）。到达间隔服从指数分布 Exp(λ)。
    num_requests       : 仿真总请求数。越大统计越稳，尾部（p99）越可信。
    mean_prompt_len    : 平均输入长度（token）。
    mean_output_len    : 平均输出长度（token）。
    prefill_ms_per_tok : 每个输入 token 的 prefill 耗时（毫秒）。prefill 是 compute-bound，
                         这里近似为「线性于 prompt_len」。
    decode_ms_per_tok  : 每个输出 token 的 decode 耗时（毫秒）。decode 是 memory-bound，
                         这里近似为「线性于 output_len」。
    prefill_fixed_ms   : prefill 的固定开销（毫秒，如 kernel launch / 采样）。
    seed               : 随机种子，保证可复现。
    """
    qps: float = 4.0
    num_requests: int = 3000
    mean_prompt_len: int = 512
    mean_output_len: int = 128
    prefill_ms_per_tok: float = 0.30
    decode_ms_per_tok: float = 12.0
    prefill_fixed_ms: float = 8.0
    seed: int = 0


def _sample_len(rng: random.Random, mean: int, low: int = 8) -> int:
    """
    采样一个「长度」。真实 LLM 流量是**右偏长尾**（多数请求中等长度、少数很长），
    长尾请求正是尾延迟的元凶。但纯指数分布尾巴过肥（会出现 10 倍于均值的长度，
    把低负载下的 TTFT 也拖到几秒），不利于「低负载几乎全达标」这一容量规划直觉。

    这里用**对数正态分布（log-normal）**：它右偏、有长尾，但尾巴比指数温和，
    且均值可控。取 sigma=0.5（中等离散），令 E[X]=mean：
        若 X~LogNormal(mu, sigma)，则 E[X]=exp(mu + sigma^2/2)，
        故 mu = ln(mean) - sigma^2/2。
    """
    sigma = 0.5
    mu = math.log(max(mean, 2)) - 0.5 * sigma * sigma
    val = math.exp(rng.gauss(mu, sigma))
    return max(low, int(val))


def generate_workload(cfg: WorkloadConfig) -> List[Request]:
    """
    生成一条到达流（arrival stream）。

    到达过程：**泊松过程（Poisson process）** —— 相邻到达间隔 ~ 指数分布 Exp(λ)，
    这是排队论里「无记忆」到达的标准模型，也是真实在线流量的常用近似。

    每个请求的服务时间：
      prefill_time = prefill_fixed + prompt_len * prefill_ms_per_tok
      decode_time  = output_len   * decode_ms_per_tok
    （单位统一换算成「秒」。）

    :return: 按到达时间升序排列的 Request 列表。
    """
    rng = random.Random(cfg.seed)
    lam = cfg.qps
    reqs: List[Request] = []
    t = 0.0
    for i in range(cfg.num_requests):
        # 1) 到达时间：累加一个 Exp(λ) 间隔
        gap = rng.expovariate(lam) if lam > 0 else 0.0
        t += gap

        # 2) 采样输入/输出长度（长尾）
        plen = _sample_len(rng, cfg.mean_prompt_len)
        olen = _sample_len(rng, cfg.mean_output_len)

        # 3) 由长度算服务时间（毫秒 → 秒）
        prefill_s = (cfg.prefill_fixed_ms + plen * cfg.prefill_ms_per_tok) / 1000.0
        decode_s = (olen * cfg.decode_ms_per_tok) / 1000.0

        reqs.append(
            Request(
                req_id=i,
                arrival=t,
                prompt_len=plen,
                output_len=olen,
                prefill_time=prefill_s,
                decode_time=decode_s,
            )
        )
    return reqs


# ======================================================================
# 4. 离散事件仿真：把请求「跑」过服务器
# ======================================================================
def simulate(
    requests: Sequence[Request],
    num_workers: int = 1,
) -> List[Request]:
    """
    最小可用的「c 服务台」离散事件仿真（相当于 M/G/c 排队系统）。

    模型假设：
      - 有 `num_workers` 个并发服务槽位。在 LLM 服务里，这近似「连续批处理」允许的
        最大 in-flight 序列数：槽位空出来，队首请求就立刻补进去。
      - 每个请求独占一个槽位，耗时 = prefill_time + decode_time，期间该槽位忙。
        （这是**简化**：真实连续批处理里 prefill/decode 会交错、显存共享；这里抓「排队 +
         服务时长」的一阶效应，足以复现尾延迟随负载上升的规律。）
      - FIFO（先到先服务）。

    实现（事件驱动，O(N log c)）：
      用一个大小为 num_workers 的最小堆存「每个槽位的空闲时刻」。
      对按到达序处理的每个请求：
        start = max(它的到达时刻, 最早空闲的槽位时刻)     # 要么没排队，要么等到槽位空
        该槽位下次空闲 = start + 服务时间
      于是天然实现了「有空位就上、没空位就排队」。

    :param requests: 已按 arrival 升序的请求（本函数不修改到达序，只回填时间戳）。
    :param num_workers: 并发槽位数 c。
    :return: 同一批 Request 对象（已回填 start/first_token/finish）。
    """
    if num_workers < 1:
        raise ValueError("num_workers 必须 >= 1")

    # 最小堆：num_workers 个槽位，初始都在时刻 0 空闲
    free_at: List[float] = [0.0] * num_workers
    heapq.heapify(free_at)

    for r in requests:
        earliest_free = free_at[0]  # 最早空闲的槽位
        start = max(r.arrival, earliest_free)  # 没空位就等到 earliest_free
        r.start = start
        r.first_token = start + r.prefill_time
        r.finish = r.first_token + r.decode_time
        # 该槽位下次空闲时刻更新回堆
        heapq.heapreplace(free_at, r.finish)

    return list(requests)


# ======================================================================
# 5. SLO 定义与指标聚合
# ======================================================================
@dataclass
class SLO:
    """
    服务等级目标（Service Level Objective）。一个请求「达标（good）」当且仅当
    它同时满足所有被设定（非 None）的门槛。

    ttft_ms : TTFT 上限（毫秒）。超了 = 首字太慢。
    tpot_ms : TPOT 上限（毫秒）。超了 = 吐字太卡（流式体验差）。
    e2e_ms  : E2E 上限（毫秒）。超了 = 整体太慢。

    约定：None 表示「该维度不设 SLO」。
    """
    ttft_ms: Optional[float] = 500.0
    tpot_ms: Optional[float] = 50.0
    e2e_ms: Optional[float] = 4000.0

    def is_good(self, r: Request) -> bool:
        """判断单个请求是否满足全部已设定的 SLO 门槛。"""
        if self.ttft_ms is not None and r.ttft * 1000.0 > self.ttft_ms:
            return False
        if self.tpot_ms is not None and r.tpot * 1000.0 > self.tpot_ms:
            return False
        if self.e2e_ms is not None and r.e2e * 1000.0 > self.e2e_ms:
            return False
        return True


@dataclass
class Metrics:
    """一次仿真的完整指标快照。所有延迟以**毫秒**存储，方便阅读与画图。"""
    num_requests: int
    qps_offered: float          # 到达率（offered load）
    throughput: float           # 实测吞吐（完成数 / 墙钟时间）
    goodput: float              # 有效吞吐（满足 SLO 的完成数 / 墙钟时间）
    goodput_ratio: float        # 达标比例 = 达标数 / 总数 ∈ [0,1]
    utilization: float          # worker 忙碌占比 ≈ GPU 利用率代理

    ttft_ms: Dict[str, float] = field(default_factory=dict)  # {'p50','p95','p99','mean'}
    tpot_ms: Dict[str, float] = field(default_factory=dict)
    e2e_ms: Dict[str, float] = field(default_factory=dict)

    def summary_line(self) -> str:
        """一行文本摘要，便于命令行观察。"""
        return (
            f"QPS={self.qps_offered:5.2f} | "
            f"吞吐={self.throughput:5.2f} req/s | "
            f"goodput={self.goodput:5.2f} req/s ({self.goodput_ratio*100:4.1f}%) | "
            f"util={self.utilization*100:4.1f}% | "
            f"E2E p50/p95/p99={self.e2e_ms['p50']:.0f}/"
            f"{self.e2e_ms['p95']:.0f}/{self.e2e_ms['p99']:.0f}ms"
        )


def compute_metrics(
    served: Sequence[Request],
    slo: SLO,
    num_workers: int,
    qps_offered: float,
) -> Metrics:
    """
    从「已服务完的请求列表」聚合出所有指标。

    关键量的定义（第一性原理，务必分清）：
      - 墙钟时间（makespan）：从第一个请求到达，到最后一个请求完成，跨度多长。
        我们用它当分母算吞吐——这是「系统实际运转了多久」。
      - 吞吐 throughput = 完成请求数 / 墙钟时间。
      - goodput = 达标请求数 / 墙钟时间。**goodput ≤ throughput 恒成立**。
      - utilization = Σ每个请求的服务时间 / (num_workers * 墙钟时间)。
        分子是「所有槽位累计的忙碌时长」，分母是「所有槽位在墙钟时间里可提供的总时长」。
        这正是「多路服务台的平均利用率」，用作 GPU 利用率的代理指标。

    :param served: simulate() 回填过时间戳的请求。
    :param slo:    SLO 门槛。
    :param num_workers: 槽位数 c（算利用率要用）。
    :param qps_offered: 名义到达率（记录用，便于画曲线）。
    """
    if not served:
        raise ValueError("compute_metrics() 需要非空的已服务请求")

    n = len(served)
    t0 = min(r.arrival for r in served)
    t_end = max(r.finish for r in served)
    makespan = max(t_end - t0, 1e-9)  # 防 0

    # --- 达标判定 ---
    good_flags = [slo.is_good(r) for r in served]
    num_good = sum(good_flags)

    throughput = n / makespan
    goodput = num_good / makespan
    goodput_ratio = num_good / n

    # --- 利用率：所有槽位的忙碌总时长 / 可用总时长 ---
    busy = sum(r.prefill_time + r.decode_time for r in served)
    utilization = busy / (num_workers * makespan)
    # 数值上可能因边界略超 1，夹到 [0,1] 更符合「利用率」语义
    utilization = min(max(utilization, 0.0), 1.0)

    # --- 延迟百分位（毫秒）---
    def dist_ms(values: List[float]) -> Dict[str, float]:
        pcs = percentiles(values, [50, 95, 99])
        return {
            "p50": pcs[50] * 1000.0,
            "p95": pcs[95] * 1000.0,
            "p99": pcs[99] * 1000.0,
            "mean": (sum(values) / len(values)) * 1000.0,
        }

    ttft_ms = dist_ms([r.ttft for r in served])
    tpot_ms = dist_ms([r.tpot for r in served])
    e2e_ms = dist_ms([r.e2e for r in served])

    return Metrics(
        num_requests=n,
        qps_offered=qps_offered,
        throughput=throughput,
        goodput=goodput,
        goodput_ratio=goodput_ratio,
        utilization=utilization,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
    )


# ======================================================================
# 6. 顶层便捷入口：一条龙「配置 → 仿真 → 指标」
# ======================================================================
def run_once(
    cfg: WorkloadConfig,
    slo: SLO,
    num_workers: int = 1,
) -> Metrics:
    """
    跑一次完整实验：生成负载 → 仿真 → 聚合指标。这是外部（测试 / demo）最常调用的入口。
    """
    reqs = generate_workload(cfg)
    served = simulate(reqs, num_workers=num_workers)
    return compute_metrics(served, slo, num_workers=num_workers, qps_offered=cfg.qps)


def sweep_qps(
    qps_list: Sequence[float],
    base_cfg: WorkloadConfig,
    slo: SLO,
    num_workers: int = 1,
) -> List[Metrics]:
    """
    「扫 QPS」实验：固定其它配置，只改到达率，观察延迟-吞吐曲线如何变化。
    这是容量规划（capacity planning）最核心的一张图：找到 goodput 见顶、
    尾延迟起飞的「拐点（knee）」。

    :return: 与 qps_list 一一对应的 Metrics 列表。
    """
    results: List[Metrics] = []
    for q in qps_list:
        # dataclasses.replace 的手写版：拷贝配置只改 qps，保证各点可比
        cfg = WorkloadConfig(
            qps=q,
            num_requests=base_cfg.num_requests,
            mean_prompt_len=base_cfg.mean_prompt_len,
            mean_output_len=base_cfg.mean_output_len,
            prefill_ms_per_tok=base_cfg.prefill_ms_per_tok,
            decode_ms_per_tok=base_cfg.decode_ms_per_tok,
            prefill_fixed_ms=base_cfg.prefill_fixed_ms,
            seed=base_cfg.seed,
        )
        results.append(run_once(cfg, slo, num_workers=num_workers))
    return results
