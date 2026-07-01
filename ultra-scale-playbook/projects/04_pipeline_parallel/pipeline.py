"""
pipeline.py —— 从零实现流水线并行 PP(AFAB / 1F1B 调度 + 气泡分析)

对应《Ultra-Scale Playbook》第 6 章(流水线并行)。

流水线并行把**层**切到不同 stage(卡),激活在 stage 间逐段传递。
把一个 batch 拆成 m 个 micro-batch 送入流水线,梯度按 micro-batch **累加**;
所有 micro-batch 走完再做**一次**优化器更新。

关键洞察(pytest 金标准):
    不管用哪种调度(AFAB 全前全后 / 1F1B 一前一后),流水线累加出来的梯度
    都 **等于** 单进程在整个 batch 上一次前向+反向的梯度(用 sum 归约时逐元素相等)。
    调度只影响**气泡 bubble** 和**激活显存**,不影响数学结果。

气泡:流水线填充/排空阶段有 stage 空闲。理想气泡比例 = (p-1)/(m+p-1)。
    → micro-batch 数 m 越多,气泡越小;这就是为什么要切很多 micro-batch。
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 把一个"深"模型切成 num_stages 段(每段若干层)
# ----------------------------------------------------------------------------
def build_stages(num_stages=4, layers_per_stage=2, dim=8, seed=0):
    torch.manual_seed(seed)
    stages = []
    for _ in range(num_stages):
        blocks = []
        for _ in range(layers_per_stage):
            blocks += [nn.Linear(dim, dim), nn.Tanh()]
        stages.append(nn.Sequential(*blocks))
    return stages


def all_params(stages):
    ps = []
    for s in stages:
        ps += list(s.parameters())
    return ps


def zero_grads(stages):
    for p in all_params(stages):
        p.grad = None


def forward_through_stages(stages, x):
    """激活在 stage 间逐段传递(单进程里就是把上一段输出喂给下一段)。"""
    a = x
    for s in stages:
        a = s(a)
    return a


def grads_snapshot(stages):
    return [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for p in all_params(stages)]


# ----------------------------------------------------------------------------
# 参考:单进程整个 batch 一次前向+反向(sum 归约)
# ----------------------------------------------------------------------------
def reference_grads(stages, x, y):
    zero_grads(stages)
    out = forward_through_stages(stages, x)
    loss = ((out - y) ** 2).sum()
    loss.backward()
    return grads_snapshot(stages)


# ----------------------------------------------------------------------------
# 流水线:把 batch 切成 m 个 micro-batch,逐个前向+反向,梯度累加
# (数值上与调度顺序无关;AFAB 与 1F1B 结果相同)
# ----------------------------------------------------------------------------
def pipeline_grads(stages, x, y, num_micro):
    zero_grads(stages)
    xs = torch.chunk(x, num_micro, dim=0)
    ys = torch.chunk(y, num_micro, dim=0)
    for xi, yi in zip(xs, ys):
        out = forward_through_stages(stages, xi)
        loss = ((out - yi) ** 2).sum()
        loss.backward()          # .grad 累加
    return grads_snapshot(stages)


# ----------------------------------------------------------------------------
# 调度事件序列(用于气泡分析 / demo 可视化)
# 返回按"时间步"排列的 (rank/stage, 'F'|'B', micro_id) 事件
# ----------------------------------------------------------------------------
def schedule_afab(num_stages, num_micro):
    """AFAB:所有 micro 的前向全做完,再做所有反向。气泡大、激活显存大(要存 m 份)。"""
    events = []
    for m in range(num_micro):              # 全部前向
        for s in range(num_stages):
            events.append((s, "F", m))
    for m in range(num_micro):              # 全部反向
        for s in reversed(range(num_stages)):
            events.append((s, "B", m))
    return events


def schedule_1f1b(num_stages, num_micro):
    """1F1B:稳态里每个 stage 交替一前一后,激活显存降到 ~p 份(而非 m 份)。"""
    events = []
    # 简化的 1F1B:warmup 前向,稳态 1F1B,drain 反向
    warmup = num_stages - 1
    for m in range(min(warmup, num_micro)):
        events.append(("warmup-F", m))
    f, b = min(warmup, num_micro), 0
    while b < num_micro:
        if f < num_micro:
            events.append(("steady-F", f)); f += 1
        events.append(("steady-B", b)); b += 1
    return events


def bubble_ratio(num_stages, num_micro):
    """理想气泡比例 = (p-1)/(m+p-1)。"""
    p, m = num_stages, num_micro
    return (p - 1) / (m + p - 1)
