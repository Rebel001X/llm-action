"""
collectives.py —— 从零实现集合通信原语(重点:ring AllReduce = reduce-scatter + all-gather)

对应《Ultra-Scale Playbook》附录"Parallel Programming Crash Course" + 第 3 章 ZeRO 的通信基础。

ring AllReduce(Baidu 环算法)的精髓:
    每个 rank 把自己的向量切成 P 段。
    · reduce-scatter(P-1 步):数据沿环流动,每步把收到的一段加到本地对应段;
      结束时每个 rank 拥有"某一段"的完整求和结果。
    · all-gather(P-1 步):把这些已求和的段再沿环转一圈,让每个 rank 都拿到全部段。
    每个 rank 收发的数据量 = 2(P-1)/P · N,**与 P 几乎无关** → 带宽最优。

本文件用单进程"模拟环上逐步消息传递"来验证算法正确性(结果 == 手工求和);
真·多进程(用 torch.distributed 的 isend/irecv 从零搭 ring,并和官方 all_reduce 对拍)见 run_demo.py。
"""
from __future__ import annotations
import torch


def reference_sum(tensors):
    """所有 rank 张量逐元素求和(AllReduce(SUM) 的正确答案)。"""
    return torch.stack(tensors, dim=0).sum(dim=0)


def _split_chunks(vec, P):
    """把一维向量切成 P 段(要求可整除,便于清晰演示)。"""
    assert vec.numel() % P == 0, "向量长度需能被 P 整除"
    return list(torch.chunk(vec, P))


def ring_allreduce_sim(tensors):
    """
    单进程模拟 ring AllReduce。tensors: P 个等长一维张量(每个 rank 一个)。
    返回 P 个结果张量(应全部等于逐元素求和)。
    """
    P = len(tensors)
    if P == 1:
        return [tensors[0].clone()]
    data = [_split_chunks(t.clone(), P) for t in tensors]      # data[r][c] = rank r 的第 c 段

    # ---- 阶段一:reduce-scatter(P-1 步)----
    # 第 i 步:rank r 把第 (r-i)%P 段发给 (r+1)%P;收到的加到本地第 (r-1-i)%P 段
    for i in range(P - 1):
        sent = {}
        for r in range(P):
            c = (r - i) % P
            sent[(r + 1) % P] = (c, data[r][c].clone())        # 发给下一个 rank
        for r in range(P):
            c, payload = sent[r]                                # 从上一个 rank 收到
            data[r][c] = data[r][c] + payload

    # ---- 阶段二:all-gather(P-1 步)----
    # 第 i 步:rank r 把已求和的第 (r+1-i)%P 段发给 (r+1)%P;收到覆盖本地对应段
    for i in range(P - 1):
        sent = {}
        for r in range(P):
            c = (r + 1 - i) % P
            sent[(r + 1) % P] = (c, data[r][c].clone())
        for r in range(P):
            c, payload = sent[r]
            data[r][c] = payload

    return [torch.cat(chunks) for chunks in data]


def ring_broadcast_sim(tensors, src=0):
    """把 src 的张量广播给所有 rank(此处直接返回 src 的副本,语义演示用)。"""
    return [tensors[src].clone() for _ in tensors]


def all_gather_sim(tensors):
    """AllGather:每个 rank 得到所有 rank 张量的拼接。"""
    full = torch.cat([t for t in tensors])
    return [full.clone() for _ in tensors]


def reduce_scatter_sim(tensors):
    """ReduceScatter:先逐元素求和,再把结果按 rank 切段分给各 rank。"""
    P = len(tensors)
    s = reference_sum(tensors)
    chunks = _split_chunks(s, P)
    return [chunks[r].clone() for r in range(P)]


def allreduce_bytes_per_rank(N, P, bytes_per_elem=4):
    """ring AllReduce 每个 rank 收发的数据量 = 2(P-1)/P · N。"""
    return 2 * (P - 1) / P * N * bytes_per_elem
