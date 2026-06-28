#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续批处理(Continuous Batching) + 分页 KV 缓存(PagedAttention)调度模拟器
==========================================================================

这是什么
--------
一个 **纯 Python 离散事件模拟器**(不依赖外部数据集 / 网络 / GPU),用来在 CPU 上
"几十秒内跑通"地直观对比两种 LLM 推理服务的批处理 + 显存管理策略:

  (a) 静态批处理 (Static Batching, 也叫 request-level batching)
      - 把一批请求凑齐后一起送进模型;
      - 右 padding 到 batch 内最长序列;
      - 整批必须等到 **最慢(最长输出)的那条请求** 解码完才能返回;
      - KV 缓存按 "max_seq_len" 预留连续显存 -> 大量内部碎片 (internal fragmentation)。

  (b) 连续批处理 (Continuous Batching, 迭代级 iteration-level 调度)
      + 分页 KV 缓存 (PagedAttention 思想, block_size 个 token 一块, 按需分配/释放)
      - 调度粒度是 "一次 forward = 一个 token step",而不是 "一整条请求";
      - 某条请求一旦生成完 EOS,**当步立即退出**并把它占的 KV 块还给显存池;
      - 等待队列里的新请求 **可以在任意 step 即时补位** 进入运行批次;
      - KV 按 16-token 的 block 动态分配,只有最后一块可能半空 -> 碎片极小。

模拟器输出两种策略的:吞吐 (tokens/s)、平均/尾部延迟 (latency)、
KV 显存峰值与碎片率 (fragmentation),从而 **定量** 说明 vLLM 那篇论文
(PagedAttention, arXiv:2309.06180) 为什么能大幅提升吞吐、降低显存浪费。

注意
----
* 这是 **教学用的离散事件模拟**,不跑真实矩阵乘 / 真实注意力。
  "一个 token step 的耗时" 用一个与 batch 内活跃请求数相关的简单成本模型来近似
  (近似 roofline: step 时间 = 固定开销 + 每条请求的边际开销)。
* 真实工程(vLLM/SGLang)里还有 prefix caching、chunked prefill、抢占换出、
  copy-on-write 共享块等机制,这里都做了简化,见 README "局限" 一节。

打印一律用 ASCII(Windows GBK 控制台打印中文会 UnicodeEncodeError);
中文只出现在注释 / docstring / README 里。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# torch 仅用于演示 "环境里确实有它" + 做一次极小的张量自检,核心模拟是纯 numpy/Python。
try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch 必装,这里只是稳妥
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# 0. 全局可复现 + 成本模型超参数                                              #
# --------------------------------------------------------------------------- #

SEED = 42

# --- KV 分页参数 ---
BLOCK_SIZE = 16          # 每个 KV block 容纳的 token 数(vLLM 默认 16),对应原理:分页粒度
TOTAL_KV_BLOCKS = 64     # 显存池里一共有多少个 KV block(toy 规模,可制造 "显存不够" 压力)

# --- step 时间成本模型(单位:毫秒)---
# 真实里:一次 decode forward 的延迟 ~= 常数开销(kernel launch / 读权重) + 与并发请求数
# 近似线性的边际开销。我们用这个简化模型,让模拟既快又能体现 "批越大单位 token 越省"。
STEP_FIXED_MS = 8.0      # 每个 forward step 的固定开销(摊给整批,batch 越大越划算)
STEP_PER_REQ_MS = 1.2    # 批内每多一条活跃请求,这一 step 额外增加的耗时

# 一条 token 在 KV 里占的 "格子" 就是 1(我们用 token 数当显存单位,直观)。


# --------------------------------------------------------------------------- #
# 1. 请求定义 + 负载生成                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Request:
    """一条推理请求(toy)。"""
    req_id: int
    arrival_ms: float          # 到达时间(ms),模拟在线流量错峰到达
    prompt_len: int            # prompt(prefill)token 数
    output_len: int            # 需要解码生成的 token 数(decode 步数)

    # ---- 运行期状态(模拟过程中更新)----
    generated: int = 0         # 已生成的 token 数
    start_ms: float = -1.0     # 首次被调度执行的时间(用于排队延迟)
    finish_ms: float = -1.0    # 完成时间
    kv_blocks: List[int] = field(default_factory=list)  # 当前持有的 KV block 物理号

    @property
    def total_len(self) -> int:
        """该请求最终占用的逻辑 KV 长度 = prompt + 已生成。"""
        return self.prompt_len + self.generated

    @property
    def final_len(self) -> int:
        """该请求生命周期内的最大 KV 长度 = prompt + 全部输出。"""
        return self.prompt_len + self.output_len

    @property
    def is_done(self) -> bool:
        return self.generated >= self.output_len


def make_workload(n_requests: int, rng: random.Random) -> List[Request]:
    """
    生成一批 "长度差异很大、错峰到达" 的请求。
    长度方差大 + 错峰到达,正是静态批处理最吃亏、连续批处理收益最大的场景:
      - 长度差异大 -> 静态批的右 padding 浪费 + 等最慢请求的 "队头阻塞" 严重;
      - 错峰到达   -> 静态批必须等凑批 / 等整批结束,新请求迟迟进不来。
    """
    reqs: List[Request] = []
    t = 0.0
    for i in range(n_requests):
        # 到达间隔:指数分布(泊松到达),平均 ~20ms 来一个
        t += rng.expovariate(1.0 / 20.0)
        prompt_len = rng.randint(8, 40)        # prompt 长度 8~40
        # 输出长度故意做成 "长尾":多数短,少数很长 -> 放大队头阻塞效应
        if rng.random() < 0.25:
            output_len = rng.randint(60, 120)  # 25% 是 "长输出"
        else:
            output_len = rng.randint(8, 40)    # 75% 是 "短输出"
        reqs.append(Request(req_id=i, arrival_ms=t,
                            prompt_len=prompt_len, output_len=output_len))
    return reqs


# --------------------------------------------------------------------------- #
# 2. 分页 KV 缓存显存池(PagedAttention 的核心数据结构)                        #
# --------------------------------------------------------------------------- #

class PagedKVCache:
    """
    模拟 vLLM 的 block 管理器:把 KV 缓存切成固定大小 (BLOCK_SIZE) 的物理块,
    用一个 "空闲块列表" 按需分配 / 释放。
    原理对应:操作系统的分页虚拟内存 —— 逻辑上连续的 KV 序列,物理上散落在不同块里,
    从而 **消除外部碎片**,内部碎片最多只有 "每条请求最后一块的半空部分"。
    """

    def __init__(self, total_blocks: int, block_size: int):
        self.block_size = block_size
        self.total_blocks = total_blocks
        # 空闲物理块号池;分配=弹出,释放=放回。这就是 vLLM block allocator 的极简版。
        self.free_blocks: List[int] = list(range(total_blocks))

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self.free_blocks)

    def blocks_needed(self, n_tokens: int) -> int:
        """存 n_tokens 个 token 需要几块(向上取整)——分页的本质。"""
        return math.ceil(n_tokens / self.block_size)

    def can_allocate(self, n_blocks: int) -> bool:
        return len(self.free_blocks) >= n_blocks

    def allocate(self, req: Request, n_tokens_total: int) -> bool:
        """
        确保 req 持有足够块来容纳 n_tokens_total 个 token。
        返回 True=分配成功;False=显存不够(调用方需让该请求继续排队/等待)。
        这正是 "decode 时序列变长 -> 按需追加一块" 的按需分配行为。
        """
        need = self.blocks_needed(max(1, n_tokens_total))
        have = len(req.kv_blocks)
        if need <= have:
            return True
        extra = need - have
        if not self.can_allocate(extra):
            return False
        for _ in range(extra):
            req.kv_blocks.append(self.free_blocks.pop())
        return True

    def free(self, req: Request) -> None:
        """请求完成 -> 立即归还它的所有块(连续批处理才能即时复用显存)。"""
        for b in req.kv_blocks:
            self.free_blocks.append(b)
        req.kv_blocks.clear()

    def fragmentation(self, running: List[Request]) -> float:
        """
        计算当前 **内部碎片率** = (已分配块容量 - 实际占用 token) / 已分配块容量。
        分页方案下只有 "每条请求最后一块" 可能半空,所以这个值很小。
        """
        if self.used_blocks == 0:
            return 0.0
        allocated_slots = self.used_blocks * self.block_size
        real_tokens = sum(r.total_len for r in running)
        return max(0.0, (allocated_slots - real_tokens) / allocated_slots)


# --------------------------------------------------------------------------- #
# 3. step 时间成本模型(两种策略共用,保证对比公平)                            #
# --------------------------------------------------------------------------- #

def step_time_ms(active_reqs: int) -> float:
    """
    一次 forward step(给批内每条请求各推进 1 个 token)的耗时近似。
    batch 越大 -> 固定开销被摊薄 -> 单 token 成本下降,这是批处理省时的根因。
    """
    if active_reqs <= 0:
        return 0.0
    return STEP_FIXED_MS + STEP_PER_REQ_MS * active_reqs


# --------------------------------------------------------------------------- #
# 4. 指标汇总                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class SimResult:
    name: str
    makespan_ms: float            # 处理完所有请求的总墙钟时间
    total_output_tokens: int      # 总共生成的 token 数
    avg_latency_ms: float         # 平均端到端延迟(完成时间 - 到达时间)
    p99_latency_ms: float         # 尾延迟
    throughput_tok_s: float       # 吞吐 tokens/s
    peak_kv_slots: int            # KV 峰值占用(以 token "格子" 计)
    avg_frag: float               # 平均内部碎片率
    n_steps: int                  # 总共跑了多少个 forward step

    def print_report(self) -> None:
        # 一律英文,避免 Windows GBK 控制台报错
        print(f"  [{self.name}]")
        print(f"    forward steps      : {self.n_steps}")
        print(f"    makespan (ms)      : {self.makespan_ms:10.1f}")
        print(f"    output tokens      : {self.total_output_tokens}")
        print(f"    throughput (tok/s) : {self.throughput_tok_s:10.1f}")
        print(f"    avg latency (ms)   : {self.avg_latency_ms:10.1f}")
        print(f"    p99 latency (ms)   : {self.p99_latency_ms:10.1f}")
        print(f"    peak KV slots      : {self.peak_kv_slots}")
        print(f"    avg KV frag (%)    : {100.0 * self.avg_frag:10.1f}")


def _latency_stats(reqs: List[Request]) -> tuple[float, float]:
    lat = sorted(r.finish_ms - r.arrival_ms for r in reqs)
    avg = float(np.mean(lat))
    # p99(toy 规模直接取最接近 99 分位的样本)
    idx = min(len(lat) - 1, int(math.ceil(0.99 * len(lat)) - 1))
    p99 = lat[idx]
    return avg, p99


# --------------------------------------------------------------------------- #
# 5. 策略 (a):静态批处理 (Static Batching)                                     #
# --------------------------------------------------------------------------- #

def run_static_batching(requests: List[Request], batch_size: int,
                        max_seq_len: int) -> SimResult:
    """
    静态批处理 + 连续显存预留:
      - 凑够 batch_size 条(或等到流量到齐)就发车;
      - **右 padding 到批内最长输出**:整批的 step 数 = 批内 max(output_len);
        短请求在长请求没跑完前 **不能提前退出**(队头阻塞 + padding 浪费算力)。
      - KV 显存按 max_seq_len **静态预留**(经典实现:为最坏情况预留连续显存),
        于是碎片 = 预留 - 实际,通常非常大。
    """
    reqs = sorted([Request(**vars(r)) for r in requests], key=lambda x: x.arrival_ms)
    for r in reqs:
        r.generated = 0
        r.kv_blocks = []
        r.start_ms = -1.0
        r.finish_ms = -1.0

    clock = 0.0
    n_steps = 0
    peak_kv_slots = 0
    frag_samples: List[float] = []

    pending = list(reqs)            # 按到达顺序排队
    i = 0
    while i < len(pending):
        batch = pending[i:i + batch_size]
        i += len(batch)

        # 发车时刻 = max(当前时钟, 批内最后一条的到达时间)。必须等齐才能发车。
        clock = max(clock, max(r.arrival_ms for r in batch))
        for r in batch:
            r.start_ms = clock

        # ---- 静态显存:整批按 max_seq_len 预留(每条都占 max_seq_len 个格子)----
        reserved_slots = len(batch) * max_seq_len
        peak_kv_slots = max(peak_kv_slots, reserved_slots)

        # ---- 整批一起 prefill(简化:prefill 也按一个 step 计成本,体现固定开销)----
        clock += step_time_ms(len(batch))
        n_steps += 1

        # ---- decode:整批同步推进,步数 = 批内最长 output_len(右 padding 等最慢)----
        max_out = max(r.output_len for r in batch)
        for step in range(max_out):
            active = [r for r in batch if r.output_len > step]  # 仍需产出的(其余在 padding)
            # 注意:静态批里即使 active 变少,整批仍占着 batch_size 个 slot,算力照付固定批宽
            clock += step_time_ms(len(batch))   # 关键:成本按 **整批宽度**,不是 active 数
            n_steps += 1
            for r in active:
                r.generated += 1
                if r.is_done and r.finish_ms < 0:
                    r.finish_ms = clock         # 但完成时刻仍是 "整批同步" 的这一步

            # 实际占用 token vs 预留 -> 碎片
            real = sum(r.total_len for r in batch)
            frag_samples.append(max(0.0, (reserved_slots - real) / reserved_slots))

        # 整批结束(右 padding 语义:即便短请求早就生成完,返回时间也被拖到整批最大步)
        for r in batch:
            if r.finish_ms < 0:
                r.finish_ms = clock

    total_out = sum(r.output_len for r in reqs)
    avg_lat, p99_lat = _latency_stats(reqs)
    thr = total_out / (clock / 1000.0) if clock > 0 else 0.0
    return SimResult(
        name="static_batching",
        makespan_ms=clock,
        total_output_tokens=total_out,
        avg_latency_ms=avg_lat,
        p99_latency_ms=p99_lat,
        throughput_tok_s=thr,
        peak_kv_slots=peak_kv_slots,
        avg_frag=float(np.mean(frag_samples)) if frag_samples else 0.0,
        n_steps=n_steps,
    )


# --------------------------------------------------------------------------- #
# 6. 策略 (b):连续批处理 + 分页 KV (Continuous Batching + PagedAttention)      #
# --------------------------------------------------------------------------- #

def run_continuous_batching(requests: List[Request], max_running: int) -> SimResult:
    """
    迭代级 (iteration-level) 调度 + 分页 KV:
      - 每个 step 给 "当前运行批次" 里每条请求各产出 1 个 token;
      - 完成 EOS 的请求 **当步立即退出 + 释放 KV 块**(没有队头阻塞、没有 padding);
      - 等待队列里的请求 **当步即时补位** 进入运行批次(只要还有空位 + 够 KV 块);
      - KV 按 BLOCK_SIZE 的块 **按需分配**:序列变长就追加一块,最后一块可能半空。
    这就是 vLLM 的核心调度循环(简化版)。
    """
    reqs = sorted([Request(**vars(r)) for r in requests], key=lambda x: x.arrival_ms)
    for r in reqs:
        r.generated = 0
        r.kv_blocks = []
        r.start_ms = -1.0
        r.finish_ms = -1.0

    kv = PagedKVCache(TOTAL_KV_BLOCKS, BLOCK_SIZE)

    clock = 0.0
    n_steps = 0
    peak_kv_slots = 0
    frag_samples: List[float] = []

    waiting: List[Request] = []        # 已到达、等待被调度的请求
    running: List[Request] = []        # 正在运行批次中的请求
    not_arrived = list(reqs)           # 还没到达的(按时间排序)
    finished = 0
    total_req = len(reqs)

    def admit_new() -> None:
        """调度器:把等待队列里的请求尽量塞进运行批次(连续批处理的 '即时补位')。"""
        # 先把 "已到达" 的请求从 not_arrived 移到 waiting
        while not_arrived and not_arrived[0].arrival_ms <= clock + 1e-9:
            waiting.append(not_arrived.pop(0))
        # 在不超过 max_running 且 KV 块够的前提下,尽量入队
        j = 0
        while j < len(waiting) and len(running) < max_running:
            r = waiting[j]
            # 入队需要为它的 prompt + 第一个待生成 token 预留块(prefill 的初始块)
            need_tokens = r.prompt_len + 1
            need_blocks = kv.blocks_needed(need_tokens) - len(r.kv_blocks)
            if kv.can_allocate(need_blocks):
                kv.allocate(r, need_tokens)
                r.start_ms = clock
                running.append(r)
                waiting.pop(j)
            else:
                j += 1   # 这条暂时塞不下,看下一条(显存压力下的 best-effort)

    # 主循环:只要还有请求没完成,就一直推进 step
    while finished < total_req:
        # 若运行批次为空,说明在等下一个请求到达 -> 把时钟快进到它的到达时刻
        if not running:
            if not_arrived:
                clock = max(clock, not_arrived[0].arrival_ms)
            admit_new()
            if not running:
                # 仍然没人(理论上不会),保险跳出
                break

        admit_new()   # 每个 step 开始都尝试补位

        # ---- 执行一个 forward step:成本只按 **当前活跃请求数**(不是固定批宽)----
        clock += step_time_ms(len(running))
        n_steps += 1

        done_this_step: List[Request] = []
        for r in running:
            r.generated += 1
            # decode 后序列 +1,可能需要追加一个 KV block(按需分配)
            ok = kv.allocate(r, r.total_len)
            if not ok:
                # 显存真的不够:回退这一步的生成(真实 vLLM 会抢占/换出,这里简化为不推进)
                r.generated -= 1
                continue
            if r.is_done:
                r.finish_ms = clock     # 完成就出,延迟即时结算(无队头阻塞)
                done_this_step.append(r)

        # ---- 完成的请求立即释放 KV 块,空间当步可被新请求复用 ----
        for r in done_this_step:
            kv.free(r)
            running.remove(r)
            finished += 1

        # ---- 采样显存指标 ----
        peak_kv_slots = max(peak_kv_slots, kv.used_blocks * BLOCK_SIZE)
        frag_samples.append(kv.fragmentation(running))

    total_out = sum(r.output_len for r in reqs)
    avg_lat, p99_lat = _latency_stats(reqs)
    thr = total_out / (clock / 1000.0) if clock > 0 else 0.0
    return SimResult(
        name="continuous_paged",
        makespan_ms=clock,
        total_output_tokens=total_out,
        avg_latency_ms=avg_lat,
        p99_latency_ms=p99_lat,
        throughput_tok_s=thr,
        peak_kv_slots=peak_kv_slots,
        avg_frag=float(np.mean(frag_samples)) if frag_samples else 0.0,
        n_steps=n_steps,
    )


# --------------------------------------------------------------------------- #
# 7. 可视化(可选;matplotlib 缺失则纯文本,不崩)                              #
# --------------------------------------------------------------------------- #

def try_plot(static: SimResult, cont: SimResult, out_png: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")          # 无显示环境也能存图
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = ["static_batching", "continuous_paged"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(labels, [static.throughput_tok_s, cont.throughput_tok_s],
                color=["#bbbbbb", "#3b82f6"])
    axes[0].set_title("Throughput (tok/s)  higher=better")

    axes[1].bar(labels, [static.avg_latency_ms, cont.avg_latency_ms],
                color=["#bbbbbb", "#3b82f6"])
    axes[1].set_title("Avg latency (ms)  lower=better")

    axes[2].bar(labels, [100 * static.avg_frag, 100 * cont.avg_frag],
                color=["#bbbbbb", "#3b82f6"])
    axes[2].set_title("KV fragmentation (%)  lower=better")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 8. demo 主程序                                                               #
# --------------------------------------------------------------------------- #

def main() -> None:
    # ---- 可复现 ----
    random.seed(SEED)
    np.random.seed(SEED)
    if _HAS_TORCH:
        torch.manual_seed(SEED)
        # 极小自检:确认 torch CPU 张量可用(教学项目里证明环境 OK)
        _ = (torch.arange(4, dtype=torch.float32) ** 2).sum().item()

    rng = random.Random(SEED)

    print("=" * 68)
    print(" Continuous Batching + Paged KV-Cache  scheduling simulator")
    print(" (toy discrete-event sim; reproduces the vLLM PagedAttention idea)")
    print("=" * 68)

    # ---- 生成负载 ----
    N = 48
    workload = make_workload(N, rng)
    avg_out = float(np.mean([r.output_len for r in workload]))
    max_out = max(r.output_len for r in workload)
    print(f" workload          : {N} requests, "
          f"avg output={avg_out:.1f} tok, max output={max_out} tok")
    print(f" KV pool           : {TOTAL_KV_BLOCKS} blocks x {BLOCK_SIZE} tok/block "
          f"= {TOTAL_KV_BLOCKS * BLOCK_SIZE} slots")
    print(f" max running batch : 16 (both strategies use comparable width)")
    print("-" * 68)

    # ---- 静态批:batch_size=16,KV 静态预留到 "实际遇到的最大序列长度" ----
    max_seq_len = max(r.final_len for r in workload)
    static = run_static_batching(workload, batch_size=16, max_seq_len=max_seq_len)

    # ---- 连续批 + 分页 ----
    cont = run_continuous_batching(workload, max_running=16)

    # ---- 报告 ----
    print(" RESULTS")
    static.print_report()
    cont.print_report()
    print("-" * 68)

    # ---- 量化收益(这就是 "成功" 的可量化信号)----
    speedup = static.makespan_ms / cont.makespan_ms if cont.makespan_ms else float("nan")
    thr_gain = cont.throughput_tok_s / static.throughput_tok_s if static.throughput_tok_s else float("nan")
    lat_drop = static.avg_latency_ms / cont.avg_latency_ms if cont.avg_latency_ms else float("nan")
    frag_static = 100 * static.avg_frag
    frag_cont = 100 * cont.avg_frag

    print(" SUMMARY (continuous+paged vs static)")
    print(f"   makespan speedup        : {speedup:6.2f}x  faster")
    print(f"   throughput gain         : {thr_gain:6.2f}x  more tok/s")
    print(f"   avg-latency improvement : {lat_drop:6.2f}x  lower")
    print(f"   KV fragmentation        : {frag_static:5.1f}%  ->  {frag_cont:4.1f}%")
    print(f"   peak KV slots           : {static.peak_kv_slots}  ->  {cont.peak_kv_slots}")

    # ---- 自动断言:跑通即应观察到连续批处理全面占优(教学结论必须成立)----
    assert thr_gain > 1.0, "continuous batching should out-throughput static"
    assert frag_cont < frag_static, "paged KV should fragment far less than static reservation"
    print("-" * 68)
    print(" CHECK PASSED: continuous+paged wins on throughput AND fragmentation.")

    # ---- 可选画图 ----
    png = try_plot(static, cont,
                   "C:/Users/jianm/Desktop/llm-action/practical-projects/"
                   "11-paged-attention-batching-sim/comparison.png")
    if png:
        print(f" figure saved      : {png}")
    else:
        print(" matplotlib not available -> skipped figure (text report above is enough)")

    print("=" * 68)
    print(" DONE.")


if __name__ == "__main__":
    main()
