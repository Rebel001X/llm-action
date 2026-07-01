"""
zero_dp.py —— 从零实现数据并行 DDP 与 ZeRO-1/2/3(核心逻辑,单进程可验证)

对应《Ultra-Scale Playbook》第 3 章(数据并行 + ZeRO)。

关键洞察(也是 pytest 的金标准):
    DDP 与 ZeRO **不改变参数更新的数学**,只改变"谁存什么、谁算什么"。
    所以只要实现正确,把分片全部聚合回来,结果必须和"单卡在全 batch 上做一步 Adam"**逐元素相等**。
    我们在单进程里模拟 world_size 个分片来验证这一点(不依赖多进程,稳过);
    真·多进程(gloo)版见 run_demo.py。

中英并列:数据并行 data parallelism、全规约 all-reduce、规约散射 reduce-scatter、
全收集 all-gather、零冗余优化器 ZeRO、优化器状态 optimizer states。
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 玩具模型 + 数据(固定随机种子,保证可复现)
# ----------------------------------------------------------------------------
class ToyMLP(nn.Module):
    def __init__(self, d_in=8, d_hidden=16, d_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


def make_model(seed=0):
    torch.manual_seed(seed)
    return ToyMLP()


def make_batch(n, d_in=8, d_out=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g)
    y = torch.randn(n, d_out, generator=g)
    return x, y


def compute_grads(model, x, y):
    """在一批数据上前向 + MSE(mean)+ 反向,返回每个参数的梯度(list[Tensor])。"""
    model.zero_grad(set_to_none=True)
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()], loss.item()


# ----------------------------------------------------------------------------
# 扁平化工具:把所有参数/梯度拉成一个大向量,方便按元素分片(ZeRO 的做法)
# ----------------------------------------------------------------------------
def flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(flat, like):
    out, i = [], 0
    for t in like:
        n = t.numel()
        out.append(flat[i:i + n].reshape(t.shape))
        i += n
    return out


# ----------------------------------------------------------------------------
# 手写 Adam(逐元素;单卡参考实现就用它,保证和分片版数学一致)
# ----------------------------------------------------------------------------
def adam_step(p, g, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    p = p - lr * mhat / (vhat.sqrt() + eps)
    return p, m, v


# ----------------------------------------------------------------------------
# 分片索引:把长度 n 的向量尽量均匀切成 world_size 段(ZeRO 沿 dp 维分片)
# ----------------------------------------------------------------------------
def shard_ranges(n, world_size):
    base, rem = divmod(n, world_size)
    ranges, start = [], 0
    for r in range(world_size):
        size = base + (1 if r < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


# ----------------------------------------------------------------------------
# DDP:各 rank 在自己的数据分片上算梯度 → all-reduce 求平均 → 更新
# 单进程模拟:把 batch 切成 world_size 份,分别算梯度再平均。
# ----------------------------------------------------------------------------
def ddp_average_grads(per_rank_grads):
    """all-reduce(mean):把各 rank 的梯度逐参数求平均。"""
    world = len(per_rank_grads)
    return [sum(g[i] for g in per_rank_grads) / world for i in range(len(per_rank_grads[0]))]


# ----------------------------------------------------------------------------
# ZeRO-3 一步(单进程模拟 world_size 个分片)
#   - 梯度已 all-reduce(平均)得到全量梯度 flat_g
#   - 参数/优化器状态按元素分片:每个 rank 只持有并更新自己那段
#   - 更新后 all-gather 参数分片,拼回完整参数
# 因为 Adam 是逐元素的,分片更新 + 拼回 == 整体更新(这正是我们要验证的)。
# ----------------------------------------------------------------------------
def zero3_step(flat_p, flat_g, states, t, world_size, **adam_kw):
    """states: list,每个 rank 一份 {'m','v'}(只覆盖自己那段)。返回新的 flat_p。"""
    n = flat_p.numel()
    ranges = shard_ranges(n, world_size)
    new_p = flat_p.clone()
    for r, (a, b) in enumerate(ranges):
        if b <= a:
            continue
        p_shard = flat_p[a:b]
        g_shard = flat_g[a:b]
        m, v = states[r]["m"], states[r]["v"]
        p_new, m_new, v_new = adam_step(p_shard, g_shard, m, v, t, **adam_kw)
        new_p[a:b] = p_new
        states[r]["m"], states[r]["v"] = m_new, v_new
    return new_p


def init_zero_states(n, world_size):
    ranges = shard_ranges(n, world_size)
    return [{"m": torch.zeros(b - a), "v": torch.zeros(b - a)} for (a, b) in ranges]
