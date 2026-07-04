# -*- coding: utf-8 -*-
"""
GPU 装箱调度仿真核心引擎（GPU Bin-Packing Scheduler Simulation）
================================================================

对应《Generative AI on Kubernetes》第 7 章「作业调度优化」的
§7.2 Bin Packing（装箱）与 NVIDIA KAI 的 MIG 切分。

本文件用纯 Python（零 GPU、零网络、零 K8s）复刻真实 GPU 集群调度器的
**核心决策逻辑**：

    作业（Job）带着 GPU/显存需求进队 → 调度器从集群里挑一个节点（Node）
    落地（first-fit / best-fit + MIG 切分）→ 记录利用率、碎片、排队时延、被拒。

设计目标：**可读 > 花哨**。所有数据结构都是 @dataclass，所有算法都
是几十行能读懂的纯函数，方便你在 README 里逐行对照。

术语对照（book ↔ 本仿真）：
    - MostAllocated 打分策略   ↔  best-fit（谁最满往谁塞）
    - LeastAllocated 打分策略  ↔  worst-fit / spread（谁最空往谁塞）
    - first-fit                ↔  简化基线：从头扫到第一个能放下的节点
    - MIG（Multi-Instance GPU）↔  把 1 张物理 GPU 切成多个逻辑实例
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import heapq


# =============================================================================
# 1) 数据模型：Job（作业）/ Node（节点）/ 集群
# =============================================================================

@dataclass
class Job:
    """一个待调度的作业（K8s 里对应一个 Pod / PodGroup）。

    字段说明：
        job_id     : 唯一标识
        gpus       : 需要多少张 GPU（整数张卡；MIG 切分见下）
        gpu_mem_gb : 每张 GPU 上需要的显存（GB）——用来做 MIG 切分判断
        cpu        : 需要多少 CPU 核（vCPU）
        mem_gb     : 需要多少主存（GB）
        duration   : 作业运行时长（仿真时间单位；到点自动释放资源）
        arrival    : 到达时刻（进入队列的时间）

    💡 为什么显存要单列一项？因为 MIG（Multi-Instance GPU）切分的本质
       就是"一张大卡按显存切成几份小卡"。只看'几张卡'看不出能不能切。
    """
    job_id: int
    gpus: int
    gpu_mem_gb: float = 0.0
    cpu: float = 1.0
    mem_gb: float = 4.0
    duration: float = 10.0
    arrival: float = 0.0

    # ---- 运行期由调度器回填（不参与相等比较）----
    node_id: Optional[int] = field(default=None, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)

    @property
    def wait_time(self) -> Optional[float]:
        """排队时延 = 开始时刻 - 到达时刻。未调度则为 None。"""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival


@dataclass
class Node:
    """集群里的一台 GPU 节点（K8s 里对应一个 Node）。

    容量（capacity）是恒定上限，used_* 是当前已占用量。
    可用量 = 容量 - 已用。

    MIG 相关：
        mig_gpu_mem_gb : 单张物理 GPU 的总显存（如 A100=80GB）
        mig_enabled    : 该节点是否开启 MIG 切分能力
    """
    node_id: int
    gpus: int                       # 物理 GPU 张数（容量）
    cpu: float                      # CPU 核数（容量）
    mem_gb: float                   # 主存 GB（容量）
    gpu_mem_gb: float = 80.0        # 单卡显存（默认 A100 80GB）
    mig_enabled: bool = False       # 是否支持 MIG 切分

    # ---- 已用量 ----
    used_gpus: float = 0.0          # 已用 GPU（MIG 下可能是小数，如 0.5 张）
    used_cpu: float = 0.0
    used_mem_gb: float = 0.0

    # ---- 记账：落在该节点上的作业 id ----
    job_ids: list[int] = field(default_factory=list)

    # ----- 可用量 -----
    @property
    def free_gpus(self) -> float:
        return self.gpus - self.used_gpus

    @property
    def free_cpu(self) -> float:
        return self.cpu - self.used_cpu

    @property
    def free_mem_gb(self) -> float:
        return self.mem_gb - self.used_mem_gb

    # ----- 利用率（0~1）-----
    @property
    def gpu_util(self) -> float:
        return self.used_gpus / self.gpus if self.gpus else 0.0

    def can_fit(self, job: Job) -> bool:
        """判断该节点当前能否放下这个作业（容量检查）。

        ⚠️ 这里就是"装箱不超容量"这条铁律的落点：任何一维超了都不能放。
        MIG 场景下，一个作业可能只占 0.5 张卡（见 gpu_demand_on）。
        """
        gpu_need = _gpu_demand_on(job, self)
        return (
            gpu_need <= self.free_gpus + 1e-9
            and job.cpu <= self.free_cpu + 1e-9
            and job.mem_gb <= self.free_mem_gb + 1e-9
        )

    def place(self, job: Job, now: float) -> None:
        """把作业放到本节点上：扣减资源 + 记账。调用前须 can_fit()==True。"""
        gpu_need = _gpu_demand_on(job, self)
        assert self.can_fit(job), f"Node{self.node_id} 放不下 Job{job.job_id}（违反容量约束）"
        self.used_gpus += gpu_need
        self.used_cpu += job.cpu
        self.used_mem_gb += job.mem_gb
        self.job_ids.append(job.job_id)
        job.node_id = self.node_id
        job.start_time = now

    def release(self, job: Job) -> None:
        """作业结束，归还资源。"""
        gpu_need = _gpu_demand_on(job, self)
        self.used_gpus -= gpu_need
        self.used_cpu -= job.cpu
        self.used_mem_gb -= job.mem_gb
        # 数值兜底，避免浮点漂移导致负数
        self.used_gpus = max(0.0, self.used_gpus)
        self.used_cpu = max(0.0, self.used_cpu)
        self.used_mem_gb = max(0.0, self.used_mem_gb)
        if job.job_id in self.job_ids:
            self.job_ids.remove(job.job_id)


# =============================================================================
# 2) MIG 切分：把「几张卡 + 每卡显存需求」翻译成「等效占多少张物理卡」
# =============================================================================

# MIG 切分档位（以 A100 80GB 为例，NVIDIA 官方 profile 简化版）：
# 一张 80GB 的 A100 可以切成 1/2/3/7 份，对应每份显存 80/40/26.6/10 GB。
# 这里我们用"最接近且不小于需求的档位"来算占用比例。
# key = 每份显存(GB)，value = 该档位下 1 份占整卡的比例。
MIG_PROFILES: dict[float, float] = {
    10.0: 1.0 / 7,   # 1g.10gb  → 占 1/7 张卡
    20.0: 2.0 / 7,   # 2g.20gb  → 占 2/7 张卡
    40.0: 1.0 / 2,   # 3g.40gb  → 占 1/2 张卡（近似）
    80.0: 1.0,       # 7g.80gb  → 占整张卡
}


def mig_fraction(gpu_mem_gb: float, card_mem_gb: float = 80.0) -> float:
    """给定显存需求，返回它在一张 `card_mem_gb` 的卡上占多少比例（0~1）。

    算法：从小到大扫 MIG 档位，挑第一个显存 >= 需求的档位；
         其占卡比例按"档位份额 × (card_mem / 80)"缩放到实际卡型。

    🔬 第一性原理：MIG 不是"随便切"，而是硬件把 SM/显存/L2 按固定
       profile 物理隔离。所以我们必须"向上取档"——需要 15GB 也得占
       20GB 那一档（2g.20gb），剩下的 5GB 是切分粒度带来的内部碎片。

    >>> round(mig_fraction(10, 80), 4)
    0.1429
    >>> round(mig_fraction(15, 80), 4)   # 15GB 向上取到 20GB 档 = 2/7
    0.2857
    >>> mig_fraction(0, 80)              # 不需要显存 → 不占卡
    0.0
    """
    if gpu_mem_gb <= 0:
        return 0.0
    scale = card_mem_gb / 80.0
    for tier_mem in sorted(MIG_PROFILES):       # 10, 20, 40, 80
        if gpu_mem_gb <= tier_mem * scale + 1e-9:
            return MIG_PROFILES[tier_mem]
    # 超过单卡显存：占满整卡（多卡需求由 job.gpus 表达）
    return 1.0


def _gpu_demand_on(job: Job, node: Node) -> float:
    """作业在某节点上等效需要多少张「物理 GPU」。

    两种模式：
      1) 节点不支持 MIG，或作业没写显存需求，或作业要 >=1 整张卡：
         直接返回 job.gpus（整数张卡，最常见）。
      2) 节点支持 MIG 且作业只要「1 张卡的一部分显存」（gpus==1 且写了显存）：
         按 MIG 档位切分，返回小数占比（如 0.5 张卡）。

    ⚠️ 常见坑：MIG 只对"要不到一整张卡"的小作业有意义。一个要 4 张卡
       的分布式训练，谈 MIG 切分没意义（它本来就吃满整卡）。所以我们
       只在 gpus==1 且显存需求 < 整卡时才切。
    """
    if (
        node.mig_enabled
        and job.gpus == 1
        and 0 < job.gpu_mem_gb < node.gpu_mem_gb
    ):
        return mig_fraction(job.gpu_mem_gb, node.gpu_mem_gb)
    return float(job.gpus)


# =============================================================================
# 3) 放置策略：first-fit / best-fit / worst-fit
# =============================================================================

def pick_first_fit(nodes: list[Node], job: Job) -> Optional[Node]:
    """First-Fit：从头扫到**第一个**能放下的节点就放。

    - 优点：O(N) 最快，实现最简单。
    - 缺点：不挑肥拣瘦，容易把小作业撒到很多节点，碎片偏多。
    - 对应：最朴素的基线，不是 K8s 默认（K8s 默认是打分制）。
    """
    for node in nodes:
        if node.can_fit(job):
            return node
    return None


def pick_best_fit(nodes: list[Node], job: Job) -> Optional[Node]:
    """Best-Fit（≈ K8s `MostAllocated` 装箱）：在能放下的节点里，
    挑**放下后剩余 GPU 最少**的那个——也就是"谁最满往谁塞"。

    🔬 为什么这样最省钱？把作业塞进"已经很满"的节点，就能让另一些
       节点保持**完全空闲**，autoscaler 才能安全地把空节点缩容删除。
       （原书：GPU 节点每小时 $10~30，空节点=纯烧钱）

    关键：打分键 = 放置后的剩余 GPU（free_gpus - gpu_need），越小越好。
    """
    best: Optional[Node] = None
    best_leftover = float("inf")
    for node in nodes:
        if not node.can_fit(job):
            continue
        leftover = node.free_gpus - _gpu_demand_on(job, node)  # 放后剩余 GPU
        if leftover < best_leftover - 1e-9:
            best_leftover = leftover
            best = node
    return best


def pick_worst_fit(nodes: list[Node], job: Job) -> Optional[Node]:
    """Worst-Fit（≈ K8s 默认 `LeastAllocated` 分散）：挑**放下后剩余
    GPU 最多**的节点——"谁最空往谁塞"。追求负载均衡/高可用，代价是
    没有空节点、无法缩容、成本高。这里作为**对照组**存在。
    """
    best: Optional[Node] = None
    best_leftover = -1.0
    for node in nodes:
        if not node.can_fit(job):
            continue
        leftover = node.free_gpus - _gpu_demand_on(job, node)
        if leftover > best_leftover + 1e-9:
            best_leftover = leftover
            best = node
    return best


# 策略名 → 函数，方便配置驱动
STRATEGIES = {
    "first_fit": pick_first_fit,
    "best_fit": pick_best_fit,
    "worst_fit": pick_worst_fit,
}


# =============================================================================
# 4) 指标：利用率 / 碎片率
# =============================================================================

def cluster_gpu_utilization(nodes: list[Node]) -> float:
    """集群 GPU 利用率 = Σ已用GPU / Σ总GPU（0~1）。"""
    total = sum(n.gpus for n in nodes)
    used = sum(n.used_gpus for n in nodes)
    return used / total if total else 0.0


def fragmentation(nodes: list[Node]) -> float:
    """碎片率（fragmentation）：**非空但未满**节点上的空闲 GPU 占总 GPU 的比例。

    直觉：一张卡如果散落在很多"半满"节点上，就算总空闲很多，也放不下
    一个需要整节点的大作业——这些"够不着的空闲"就是碎片。

    定义（本仿真）：
        碎片 = Σ over 节点[ 若 0 < used < 容量，则该节点空闲GPU ] / 总GPU

    - 完全空的节点不算碎片（它随时能被大作业整块使用，或被缩容）。
    - 完全满的节点没有空闲，也不算碎片。
    - 只有"用了一半、卡在中间"的节点贡献碎片。

    💡 best-fit 的核心卖点就是：在典型负载下，碎片率 ≤ first-fit，
       因为它总把作业往最满的节点塞，尽量不去"开新的半满节点"。
    """
    total = sum(n.gpus for n in nodes)
    if not total:
        return 0.0
    frag = 0.0
    for n in nodes:
        if n.used_gpus > 1e-9 and n.free_gpus > 1e-9:   # 非空且非满
            frag += n.free_gpus
    return frag / total


# =============================================================================
# 5) 调度器 + 离散事件仿真（会释放资源，跑一条完整时间线）
# =============================================================================

@dataclass
class ScheduleResult:
    """一次调度/仿真跑完后的汇总指标。"""
    strategy: str
    placed: int                 # 成功调度的作业数
    rejected: int               # 被拒（放不下）的作业数
    avg_wait: float             # 平均排队时延
    max_wait: float             # 最大排队时延
    gpu_util: float             # 结束时刻的集群 GPU 利用率
    peak_util: float            # 整条时间线上的峰值利用率
    fragmentation: float        # 结束时刻的碎片率
    peak_frag: float            # 峰值碎片率
    rejected_ids: list[int] = field(default_factory=list)


class Scheduler:
    """把「策略 + 一组节点」封装成一个可复用的调度器。

    两种用法：
      - schedule_static(jobs): 一次性把所有作业塞进去，不考虑释放（静态装箱）。
      - simulate(jobs):        离散事件仿真——作业到达/结束会占用/释放资源，
                               放不下的作业进等待队列，等有资源了再上（更真实）。
    """

    def __init__(self, nodes: list[Node], strategy: str = "best_fit"):
        assert strategy in STRATEGIES, f"未知策略 {strategy}，可选 {list(STRATEGIES)}"
        self.nodes = nodes
        self.strategy = strategy
        self.pick = STRATEGIES[strategy]

    # ---------- 静态装箱：只放不放释放 ----------
    def schedule_static(self, jobs: list[Job]) -> ScheduleResult:
        """把 jobs 按顺序逐个放置，不模拟时间流逝（作业永不结束）。

        用途：教学演示"同一批作业，不同策略产生的碎片差异"，最直观。
        """
        placed, rejected_ids = 0, []
        waits = []
        peak_util = peak_frag = 0.0
        for job in jobs:
            node = self.pick(self.nodes, job)
            if node is None:
                rejected_ids.append(job.job_id)
                continue
            node.place(job, now=job.arrival)     # 静态场景下 start=arrival
            placed += 1
            waits.append(0.0)
            peak_util = max(peak_util, cluster_gpu_utilization(self.nodes))
            peak_frag = max(peak_frag, fragmentation(self.nodes))
        return ScheduleResult(
            strategy=self.strategy,
            placed=placed,
            rejected=len(rejected_ids),
            avg_wait=(sum(waits) / len(waits) if waits else 0.0),
            max_wait=(max(waits) if waits else 0.0),
            gpu_util=cluster_gpu_utilization(self.nodes),
            peak_util=peak_util,
            fragmentation=fragmentation(self.nodes),
            peak_frag=peak_frag,
            rejected_ids=rejected_ids,
        )

    # ---------- 离散事件仿真：作业会到达、运行、结束、释放 ----------
    def simulate(self, jobs: list[Job], max_requeue: int = 10_000) -> ScheduleResult:
        """离散事件仿真（discrete-event simulation）。

        事件驱动：用一个最小堆按时间推进，两类事件：
            ("arrival", t, job)  作业到达 → 尝试放置，放不下则进等待队列
            ("finish",  t, job)  作业结束 → 释放资源 → 唤醒等待队列重试

        排队时延 = 作业真正开始运行的时刻 - 到达时刻。
        被拒 = 到时间线结束仍无法放下（这里指单个作业永远大于任何节点容量）。

        🔬 这就是真实调度器的骨架：Pending 队列 + 资源事件 + 回填重试。
        """
        # 事件堆：(time, seq, kind, job)。seq 做稳定排序，避免比较 Job。
        heap: list[tuple[float, int, str, Job]] = []
        seq = 0
        for job in jobs:
            heapq.heappush(heap, (job.arrival, seq, "arrival", job))
            seq += 1

        waiting: list[Job] = []          # Pending 队列
        placed, rejected_ids = 0, []
        waits: list[float] = []
        peak_util = peak_frag = 0.0
        requeue_guard = 0

        def try_place(job: Job, now: float) -> bool:
            """尝试放置一个作业；成功则登记 finish 事件并回填等待时间。"""
            nonlocal seq, placed
            node = self.pick(self.nodes, job)
            if node is None:
                return False
            node.place(job, now)
            placed += 1
            waits.append(job.wait_time or 0.0)
            heapq.heappush(heap, (now + job.duration, _next_seq(), "finish", job))
            return True

        def _next_seq() -> int:
            nonlocal seq
            seq += 1
            return seq

        def drain_waiting(now: float) -> None:
            """有资源释放后，尽量把等待队列里的作业放上去（FIFO 回填）。"""
            nonlocal requeue_guard
            progressed = True
            while progressed:
                progressed = False
                for job in list(waiting):
                    requeue_guard += 1
                    if requeue_guard > max_requeue:
                        return
                    if try_place(job, now):
                        waiting.remove(job)
                        progressed = True

        while heap:
            t, _, kind, job = heapq.heappop(heap)
            if kind == "arrival":
                if not try_place(job, t):
                    # 放不下：先判断是否"永远放不下"（比任何单节点都大）→ 直接拒
                    if _impossible_anywhere(self.nodes, job):
                        rejected_ids.append(job.job_id)
                    else:
                        waiting.append(job)
            else:  # finish
                node = self.nodes[job.node_id]
                node.release(job)
                drain_waiting(t)
            peak_util = max(peak_util, cluster_gpu_utilization(self.nodes))
            peak_frag = max(peak_frag, fragmentation(self.nodes))

        # 时间线结束仍在等待的：算作被拒（资源始终不够）
        for job in waiting:
            rejected_ids.append(job.job_id)

        return ScheduleResult(
            strategy=self.strategy,
            placed=placed,
            rejected=len(rejected_ids),
            avg_wait=(sum(waits) / len(waits) if waits else 0.0),
            max_wait=(max(waits) if waits else 0.0),
            gpu_util=cluster_gpu_utilization(self.nodes),
            peak_util=peak_util,
            fragmentation=fragmentation(self.nodes),
            peak_frag=peak_frag,
            rejected_ids=sorted(set(rejected_ids)),
        )


def _impossible_anywhere(nodes: list[Node], job: Job) -> bool:
    """作业是否**在任何节点全空时都放不下**（需求超过最大单节点容量）。

    这种作业永远进不去，直接判拒，避免死等。
    """
    for n in nodes:
        gpu_need = _gpu_demand_on(job, n)
        if (
            gpu_need <= n.gpus + 1e-9
            and job.cpu <= n.cpu + 1e-9
            and job.mem_gb <= n.mem_gb + 1e-9
        ):
            return False
    return True


# =============================================================================
# 6) 便捷构造器：造一个同构集群
# =============================================================================

def make_homogeneous_cluster(
    n_nodes: int,
    gpus_per_node: int = 8,
    cpu_per_node: float = 64.0,
    mem_per_node: float = 512.0,
    gpu_mem_gb: float = 80.0,
    mig_enabled: bool = False,
) -> list[Node]:
    """造 n 个规格相同的 GPU 节点（典型：8×A100 节点）。"""
    return [
        Node(
            node_id=i,
            gpus=gpus_per_node,
            cpu=cpu_per_node,
            mem_gb=mem_per_node,
            gpu_mem_gb=gpu_mem_gb,
            mig_enabled=mig_enabled,
        )
        for i in range(n_nodes)
    ]


if __name__ == "__main__":
    # 冒烟自测：跑一个小例子，打印两种策略的碎片对比
    import random
    random.seed(0)
    demo_jobs = [
        Job(job_id=i, gpus=random.choice([1, 2, 4]),
            cpu=random.choice([4, 8]), mem_gb=random.choice([16, 32]),
            duration=random.randint(5, 20), arrival=i * 2)
        for i in range(20)
    ]
    for strat in ("first_fit", "best_fit", "worst_fit"):
        nodes = make_homogeneous_cluster(4)
        # 每次都用全新的 Job 副本，避免上一轮回填的 node_id/start_time 串味
        batch = [Job(job_id=j.job_id, gpus=j.gpus, cpu=j.cpu, mem_gb=j.mem_gb)
                 for j in demo_jobs]
        res = Scheduler(nodes, strat).schedule_static(batch)
        print(f"{strat:10s} placed={res.placed:2d} rej={res.rejected:2d} "
              f"util={res.gpu_util:.2%} frag={res.fragmentation:.2%}")
