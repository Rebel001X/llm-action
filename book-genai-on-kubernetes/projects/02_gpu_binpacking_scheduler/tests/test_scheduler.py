# -*- coding: utf-8 -*-
"""
GPU 装箱调度器单元测试
======================

覆盖四条主线（对应任务要求）：
  1. 装箱不超容量        —— test_capacity_*
  2. best-fit 碎片 ≤ first-fit（典型负载）—— test_bestfit_frag_le_firstfit
  3. 利用率计算正确      —— test_utilization_*
  4. MIG 切分正确        —— test_mig_*

外加：排队时延 / 被拒 / 仿真闭环 的回归测试。
"""

import os
import sys
import random

import pytest

# 让测试无论从哪跑都能 import 到上级目录的 scheduler.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import (  # noqa: E402
    Job,
    Node,
    Scheduler,
    make_homogeneous_cluster,
    mig_fraction,
    _gpu_demand_on,
    cluster_gpu_utilization,
    fragmentation,
    pick_first_fit,
    pick_best_fit,
    pick_worst_fit,
    _impossible_anywhere,
)


# =========================================================================
# 1) 装箱不超容量：任何策略、任何时刻，used_* 都不得超过容量
# =========================================================================

def _assert_no_overcommit(nodes):
    """断言集群里没有任何一台节点超卖（任一维度 used <= 容量）。"""
    for n in nodes:
        assert n.used_gpus <= n.gpus + 1e-9, f"Node{n.node_id} GPU 超卖"
        assert n.used_cpu <= n.cpu + 1e-9, f"Node{n.node_id} CPU 超卖"
        assert n.used_mem_gb <= n.mem_gb + 1e-9, f"Node{n.node_id} 内存超卖"
        assert n.used_gpus >= -1e-9, "GPU 负占用（释放 bug）"


@pytest.mark.parametrize("strategy", ["first_fit", "best_fit", "worst_fit"])
def test_capacity_never_exceeded_static(strategy):
    """静态装箱：无论哪种策略，都不能超容量。"""
    nodes = make_homogeneous_cluster(3, gpus_per_node=8, cpu_per_node=32, mem_per_node=256)
    jobs = [Job(job_id=i, gpus=(i % 4) + 1, cpu=8, mem_gb=32) for i in range(30)]
    Scheduler(nodes, strategy).schedule_static(jobs)
    _assert_no_overcommit(nodes)


@pytest.mark.parametrize("strategy", ["first_fit", "best_fit", "worst_fit"])
def test_capacity_never_exceeded_simulation(strategy):
    """离散事件仿真：有到达/结束/释放，全程都不能超容量。"""
    random.seed(42)
    nodes = make_homogeneous_cluster(4)
    jobs = [
        Job(job_id=i, gpus=random.choice([1, 2, 4]), cpu=random.choice([4, 8]),
            mem_gb=random.choice([16, 32]), duration=random.randint(3, 15),
            arrival=i * 1.0)
        for i in range(60)
    ]
    Scheduler(nodes, strategy).simulate(jobs)
    _assert_no_overcommit(nodes)


def test_place_rejects_overcommit_assertion():
    """直接对着 place() 塞一个放不下的作业，必须 assert 报错。"""
    node = Node(node_id=0, gpus=2, cpu=8, mem_gb=32)
    node.place(Job(job_id=0, gpus=2, cpu=4, mem_gb=8), now=0)  # 刚好占满 GPU
    assert node.free_gpus == 0
    with pytest.raises(AssertionError):
        node.place(Job(job_id=1, gpus=1, cpu=1, mem_gb=1), now=0)  # 再来一张 → 超


def test_can_fit_each_dimension():
    """can_fit 必须逐维检查：GPU/CPU/内存任一维不足都要判 False。"""
    node = Node(node_id=0, gpus=4, cpu=8, mem_gb=32)
    assert node.can_fit(Job(job_id=0, gpus=4, cpu=8, mem_gb=32))      # 恰好放下
    assert not node.can_fit(Job(job_id=1, gpus=5, cpu=1, mem_gb=1))   # GPU 超
    assert not node.can_fit(Job(job_id=2, gpus=1, cpu=9, mem_gb=1))   # CPU 超
    assert not node.can_fit(Job(job_id=3, gpus=1, cpu=1, mem_gb=64))  # 内存超


# =========================================================================
# 2) best-fit 碎片 ≤ first-fit（典型负载下）
# =========================================================================

def test_bestfit_frag_le_firstfit():
    """在一个精心设计的典型负载下，best-fit 的碎片率应 ≤ first-fit。

    场景：3 个节点每台 4 GPU。先来一串 2-GPU 作业。
      - first-fit：从头塞，2+2 填满 node0，2+2 填满 node1... 看似也满，
        但我们混入不同大小的作业制造差异。
    这里用一个已知会拉开差距的作业序列。
    """
    def build_nodes():
        return make_homogeneous_cluster(4, gpus_per_node=4, cpu_per_node=64, mem_per_node=512)

    # 作业序列：一堆 3-GPU 和 1-GPU 交替。
    # first-fit 会把 3 塞进 node0（剩1）、下一个 3 放不进 node0 只能开 node1（剩1）...
    # 留下一堆"剩 1 GPU"的半满节点 = 高碎片。
    # best-fit 会把后续 1-GPU 精准填进那些"剩 1"的洞 = 低碎片。
    jobs = [
        Job(job_id=0, gpus=3, cpu=1, mem_gb=1),
        Job(job_id=1, gpus=3, cpu=1, mem_gb=1),
        Job(job_id=2, gpus=3, cpu=1, mem_gb=1),
        Job(job_id=3, gpus=1, cpu=1, mem_gb=1),
        Job(job_id=4, gpus=1, cpu=1, mem_gb=1),
        Job(job_id=5, gpus=1, cpu=1, mem_gb=1),
    ]

    ff_nodes = build_nodes()
    ff = Scheduler(ff_nodes, "first_fit").schedule_static(
        [Job(job_id=j.job_id, gpus=j.gpus, cpu=j.cpu, mem_gb=j.mem_gb) for j in jobs])

    bf_nodes = build_nodes()
    bf = Scheduler(bf_nodes, "best_fit").schedule_static(
        [Job(job_id=j.job_id, gpus=j.gpus, cpu=j.cpu, mem_gb=j.mem_gb) for j in jobs])

    # 两者都应放下全部 6 个作业（总需求 3*3+3*1=12 = 3 节点*4）
    assert bf.placed == ff.placed == 6
    # 核心断言：best-fit 碎片 ≤ first-fit
    assert bf.fragmentation <= ff.fragmentation + 1e-9, (
        f"best-fit 碎片({bf.fragmentation}) 应 ≤ first-fit({ff.fragmentation})")


def test_bestfit_frag_le_firstfit_randomized():
    """随机负载的统计验证：多数随机种子下 best-fit 碎片不劣于 first-fit。

    我们不要求 100% 严格（bin packing 是 NP-hard，个别序列 first-fit 也可能
    碰巧更好），只要求「平均碎片 best-fit ≤ first-fit」这一典型结论成立。
    """
    ff_total = bf_total = 0.0
    trials = 40
    for seed in range(trials):
        random.seed(seed)
        job_specs = [
            (i, random.choice([1, 2, 3]))
            for i in range(15)
        ]

        def mk():
            return make_homogeneous_cluster(5, gpus_per_node=4, cpu_per_node=64, mem_per_node=512)

        ff = Scheduler(mk(), "first_fit").schedule_static(
            [Job(job_id=i, gpus=g, cpu=1, mem_gb=1) for i, g in job_specs])
        bf = Scheduler(mk(), "best_fit").schedule_static(
            [Job(job_id=i, gpus=g, cpu=1, mem_gb=1) for i, g in job_specs])
        ff_total += ff.fragmentation
        bf_total += bf.fragmentation

    avg_ff = ff_total / trials
    avg_bf = bf_total / trials
    assert avg_bf <= avg_ff + 1e-9, (
        f"平均碎片 best-fit({avg_bf:.4f}) 应 ≤ first-fit({avg_ff:.4f})")


# =========================================================================
# 3) 利用率计算正确
# =========================================================================

def test_utilization_empty_cluster():
    nodes = make_homogeneous_cluster(2, gpus_per_node=4)
    assert cluster_gpu_utilization(nodes) == 0.0
    assert fragmentation(nodes) == 0.0


def test_utilization_full_cluster():
    """填满所有 GPU → 利用率 100%，碎片 0（没有半满节点）。"""
    nodes = make_homogeneous_cluster(2, gpus_per_node=4, cpu_per_node=64, mem_per_node=512)
    Scheduler(nodes, "first_fit").schedule_static(
        [Job(job_id=i, gpus=4, cpu=1, mem_gb=1) for i in range(2)])
    assert cluster_gpu_utilization(nodes) == pytest.approx(1.0)
    assert fragmentation(nodes) == pytest.approx(0.0)


def test_utilization_half():
    """2 节点各 4 GPU，占用 4 张 → 利用率 = 4/8 = 50%。"""
    nodes = make_homogeneous_cluster(2, gpus_per_node=4, cpu_per_node=64, mem_per_node=512)
    Scheduler(nodes, "first_fit").schedule_static(
        [Job(job_id=0, gpus=4, cpu=1, mem_gb=1)])
    assert cluster_gpu_utilization(nodes) == pytest.approx(0.5)
    # node0 满、node1 空 → 无半满节点 → 碎片 0
    assert fragmentation(nodes) == pytest.approx(0.0)


def test_fragmentation_definition():
    """碎片率定义验证：只有'非空且非满'节点贡献碎片。

    构造：node0 用了 2/4（半满，贡献 2 碎片），node1 满（0），node2 空（0）。
    总 GPU = 12，碎片 = 2/12。
    """
    nodes = make_homogeneous_cluster(3, gpus_per_node=4, cpu_per_node=64, mem_per_node=512)
    nodes[0].place(Job(job_id=0, gpus=2, cpu=1, mem_gb=1), now=0)   # 半满
    nodes[1].place(Job(job_id=1, gpus=4, cpu=1, mem_gb=1), now=0)   # 满
    # node2 保持空
    assert fragmentation(nodes) == pytest.approx(2 / 12)
    assert cluster_gpu_utilization(nodes) == pytest.approx(6 / 12)


# =========================================================================
# 4) MIG 切分正确
# =========================================================================

def test_mig_fraction_tiers():
    """MIG 档位向上取整：需求落在哪个 profile。"""
    # 10GB → 1/7 张卡
    assert mig_fraction(10, 80) == pytest.approx(1 / 7)
    # 5GB → 仍向上取到最小档 10GB = 1/7
    assert mig_fraction(5, 80) == pytest.approx(1 / 7)
    # 15GB → 向上取到 20GB 档 = 2/7
    assert mig_fraction(15, 80) == pytest.approx(2 / 7)
    # 40GB → 1/2 张卡
    assert mig_fraction(40, 80) == pytest.approx(1 / 2)
    # 80GB → 整张卡
    assert mig_fraction(80, 80) == pytest.approx(1.0)
    # 0 → 不占卡
    assert mig_fraction(0, 80) == 0.0
    # 超过整卡显存 → 占满整卡
    assert mig_fraction(120, 80) == pytest.approx(1.0)


def test_mig_demand_on_node():
    """_gpu_demand_on：MIG 节点上小作业按比例占卡，非 MIG 节点占整卡。"""
    mig_node = Node(node_id=0, gpus=8, cpu=64, mem_gb=512, gpu_mem_gb=80, mig_enabled=True)
    plain_node = Node(node_id=1, gpus=8, cpu=64, mem_gb=512, gpu_mem_gb=80, mig_enabled=False)

    small = Job(job_id=0, gpus=1, gpu_mem_gb=10)   # 只要 10GB 显存
    # MIG 节点：占 1/7 张卡
    assert _gpu_demand_on(small, mig_node) == pytest.approx(1 / 7)
    # 非 MIG 节点：占整张卡（无法切）
    assert _gpu_demand_on(small, plain_node) == 1.0

    # 多卡作业即使在 MIG 节点也不切（谈切分无意义）
    big = Job(job_id=1, gpus=4, gpu_mem_gb=80)
    assert _gpu_demand_on(big, mig_node) == 4.0


def test_mig_packing_more_jobs():
    """MIG 让一张卡塞下更多小作业：7 个 10GB 作业刚好占满 1 张 A100。"""
    node = Node(node_id=0, gpus=1, cpu=64, mem_gb=512, gpu_mem_gb=80, mig_enabled=True)
    placed = 0
    for i in range(10):
        job = Job(job_id=i, gpus=1, gpu_mem_gb=10, cpu=1, mem_gb=1)
        if node.can_fit(job):
            node.place(job, now=0)
            placed += 1
    # 1 张卡 = 7 个 1/7 实例
    assert placed == 7
    assert node.used_gpus == pytest.approx(1.0)
    # 第 8 个放不下
    assert not node.can_fit(Job(job_id=99, gpus=1, gpu_mem_gb=10, cpu=1, mem_gb=1))


def test_mig_no_overcommit_on_node():
    """MIG 切分也不能超容量：塞到超过整卡就得拒。"""
    node = Node(node_id=0, gpus=1, cpu=64, mem_gb=512, gpu_mem_gb=80, mig_enabled=True)
    node.place(Job(job_id=0, gpus=1, gpu_mem_gb=40), now=0)  # 占 1/2
    node.place(Job(job_id=1, gpus=1, gpu_mem_gb=40), now=0)  # 再占 1/2 → 满
    assert node.used_gpus == pytest.approx(1.0)
    assert not node.can_fit(Job(job_id=2, gpus=1, gpu_mem_gb=10))  # 满了


# =========================================================================
# 5) 排队时延 / 被拒 / 仿真闭环
# =========================================================================

def test_queueing_delay_in_simulation():
    """资源不足时作业排队，有释放后回填 → 排队时延 > 0。"""
    # 1 个节点 1 张卡，两个各要 1 卡、各跑 10 的作业同时到达。
    # 第 1 个立刻上，第 2 个要等第 1 个跑完（t=10）才上 → wait≈10。
    node = Node(node_id=0, gpus=1, cpu=64, mem_gb=512)
    jobs = [
        Job(job_id=0, gpus=1, cpu=1, mem_gb=1, duration=10, arrival=0),
        Job(job_id=1, gpus=1, cpu=1, mem_gb=1, duration=10, arrival=0),
    ]
    res = Scheduler([node], "first_fit").simulate(jobs)
    assert res.placed == 2
    assert res.rejected == 0
    assert res.max_wait == pytest.approx(10.0)      # 第二个等了 10
    assert res.avg_wait == pytest.approx(5.0)       # (0 + 10) / 2


def test_rejected_job_too_big():
    """需求超过最大单节点容量的作业 → 直接被拒，不死等。"""
    nodes = make_homogeneous_cluster(2, gpus_per_node=4)
    jobs = [
        Job(job_id=0, gpus=1, cpu=1, mem_gb=1, duration=5, arrival=0),
        Job(job_id=1, gpus=8, cpu=1, mem_gb=1, duration=5, arrival=0),  # 要 8 卡，单节点只有 4
    ]
    res = Scheduler(nodes, "best_fit").simulate(jobs)
    assert 1 in res.rejected_ids
    assert res.placed == 1


def test_simulation_releases_resources():
    """仿真结束后所有作业都跑完 → 集群应回到空（利用率 0）。"""
    nodes = make_homogeneous_cluster(2, gpus_per_node=4, cpu_per_node=64, mem_per_node=512)
    jobs = [
        Job(job_id=i, gpus=2, cpu=1, mem_gb=1, duration=5, arrival=i * 3.0)
        for i in range(6)
    ]
    res = Scheduler(nodes, "best_fit").simulate(jobs)
    assert res.placed == 6
    assert res.rejected == 0
    # 所有作业结束后集群清空
    assert cluster_gpu_utilization(nodes) == pytest.approx(0.0)


def test_impossible_anywhere_helper():
    """_impossible_anywhere：只要有一台节点全空能放下就返回 False。"""
    nodes = make_homogeneous_cluster(2, gpus_per_node=4, cpu_per_node=8, mem_per_node=32)
    assert _impossible_anywhere(nodes, Job(job_id=0, gpus=8, cpu=1, mem_gb=1))   # 8>4 → 永远放不下
    assert not _impossible_anywhere(nodes, Job(job_id=1, gpus=4, cpu=8, mem_gb=32))  # 恰好放下


# =========================================================================
# 6) 策略函数直接测（挑节点逻辑）
# =========================================================================

def test_pick_best_fit_chooses_fullest():
    """best-fit 应挑'放下后剩余最少'的节点。"""
    nodes = make_homogeneous_cluster(3, gpus_per_node=8, cpu_per_node=64, mem_per_node=512)
    nodes[0].used_gpus = 2     # 剩 6
    nodes[1].used_gpus = 5     # 剩 3  ← 放 2 卡作业后剩 1，最少 → best-fit 选它
    nodes[2].used_gpus = 0     # 剩 8
    chosen = pick_best_fit(nodes, Job(job_id=0, gpus=2, cpu=1, mem_gb=1))
    assert chosen.node_id == 1


def test_pick_worst_fit_chooses_emptiest():
    """worst-fit 应挑'放下后剩余最多'的节点（最空的）。"""
    nodes = make_homogeneous_cluster(3, gpus_per_node=8, cpu_per_node=64, mem_per_node=512)
    nodes[0].used_gpus = 2
    nodes[1].used_gpus = 5
    nodes[2].used_gpus = 0     # 最空 → worst-fit 选它
    chosen = pick_worst_fit(nodes, Job(job_id=0, gpus=2, cpu=1, mem_gb=1))
    assert chosen.node_id == 2


def test_pick_first_fit_chooses_first():
    """first-fit 应挑第一个能放下的（node0）。"""
    nodes = make_homogeneous_cluster(3, gpus_per_node=8, cpu_per_node=64, mem_per_node=512)
    chosen = pick_first_fit(nodes, Job(job_id=0, gpus=2, cpu=1, mem_gb=1))
    assert chosen.node_id == 0


def test_pick_returns_none_when_full():
    """集群全满时所有策略都返回 None（触发被拒/排队）。"""
    nodes = make_homogeneous_cluster(2, gpus_per_node=2, cpu_per_node=8, mem_per_node=32)
    for n in nodes:
        n.used_gpus = n.gpus   # 占满 GPU
    job = Job(job_id=0, gpus=1, cpu=1, mem_gb=1)
    assert pick_first_fit(nodes, job) is None
    assert pick_best_fit(nodes, job) is None
    assert pick_worst_fit(nodes, job) is None
