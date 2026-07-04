"""
ring_attention.py —— 从零实现上下文并行 CP / Ring Attention(在线 softmax + 序列分块)

对应《Ultra-Scale Playbook》第 5 章(上下文并行)。

超长序列(128k+)时,连激活/注意力都放不下 → 沿**序列维**把 Q/K/V 切到各 rank。
每个 rank 持有一段 Q_i,并在**环 ring**上逐跳接收其它 rank 的 K/V 块,用**在线 softmax**
边收边算、边累加输出,**永不物化完整的 S×S 注意力矩阵**,显存 O(S) 而非 O(S²)。

金标准(pytest):Ring Attention 的输出必须 == 标准全注意力(逐元素,含因果掩码)。
本文件是单进程可验证核心(在单进程里模拟环上每一步);真·多进程见 run_demo.py。

在线 softmax 递推(和 FlashAttention 一致):
    对每个新的 K/V 块,维护 running max m、running sum l、running output O:
        m_new = max(m, rowmax(S_block))
        α = exp(m - m_new)             # 旧状态重缩放因子
        l = α·l + rowsum(exp(S_block - m_new))
        O = α·O + exp(S_block - m_new) @ V_block
    最后 O /= l。初始 m=-inf → α=0,自动忽略空状态。
"""
from __future__ import annotations
import math
import torch


# ----------------------------------------------------------------------------
# 参考:标准全注意力
# ----------------------------------------------------------------------------
def full_attention(Q, K, V, causal=False):
    d = Q.shape[-1]
    S = Q @ K.transpose(-1, -2) / math.sqrt(d)          # (s, s)
    if causal:
        s = Q.shape[-2]
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    return A @ V


# ----------------------------------------------------------------------------
# 在线 softmax:把一个 K/V 块并入 running (m, l, O)
# ----------------------------------------------------------------------------
def online_update(m, l, O, S_block, V_block):
    """S_block: (bq, bk) 已经是 QK^T/sqrt(d) 且已加好掩码(-inf 表示屏蔽)。"""
    block_max = S_block.max(dim=-1).values                       # (bq,)
    m_new = torch.maximum(m, block_max)
    # 处理整行全 -inf 的块(该 query 在此块无可见 key):其贡献为 0
    m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
    alpha = torch.exp(m - m_safe)                                # 旧状态重缩放
    p = torch.exp(S_block - m_safe.unsqueeze(-1))                # (bq, bk)
    p = torch.nan_to_num(p, nan=0.0)                             # -inf-(-inf) 保护
    l = alpha * l + p.sum(dim=-1)
    O = alpha.unsqueeze(-1) * O + p @ V_block
    return m_new, l, O


# ----------------------------------------------------------------------------
# Ring Attention(单进程模拟:每个 query 块遍历所有 key 块)
# ----------------------------------------------------------------------------
def ring_attention(Q, K, V, num_ranks, causal=False):
    s, d = Q.shape[-2], Q.shape[-1]
    assert s % num_ranks == 0, "序列长度需能被 rank 数整除"
    b = s // num_ranks
    Qs = Q.split(b, dim=-2)
    Ks = K.split(b, dim=-2)
    Vs = V.split(b, dim=-2)
    outs = []
    for i, Qi in enumerate(Qs):                                  # 每个 query 块(每个 rank)
        m = torch.full((b,), float("-inf"))
        l = torch.zeros(b)
        O = torch.zeros(b, d)
        # 环上逐块接收 K/V(这里按顺序遍历所有块,等价于走完一整圈 ring)
        for j in range(num_ranks):
            if causal and j > i:
                continue                                         # 因果:未来 key 块整块跳过
            S_block = Qi @ Ks[j].transpose(-1, -2) / math.sqrt(d)
            if causal and j == i:                                # 对角块:块内因果掩码
                mask = torch.triu(torch.ones(b, b, dtype=torch.bool), diagonal=1)
                S_block = S_block.masked_fill(mask, float("-inf"))
            m, l, O = online_update(m, l, O, S_block, Vs[j])
        outs.append(O / l.unsqueeze(-1))
    return torch.cat(outs, dim=-2)


# ----------------------------------------------------------------------------
# Zig-Zag 负载均衡:因果注意力下,靠后的 query 块工作量大。
# zigzag 把序列位置重排,让每个 rank 拿到"一前一后"两段,计算量摊平。
# 这里给出重排索引 + 逆重排;验证:重排后做 ring attention 再逆重排 == 全因果注意力。
# ----------------------------------------------------------------------------
def zigzag_indices(s, num_ranks):
    """把 2*num_ranks 个 chunk 交错分配:rank r 拿 chunk r 和 chunk (2P-1-r)。"""
    assert s % (2 * num_ranks) == 0
    c = s // (2 * num_ranks)                                     # 每个小 chunk 长度
    perm = []
    for r in range(num_ranks):
        perm.append(torch.arange(r * c, (r + 1) * c))
        far = 2 * num_ranks - 1 - r
        perm.append(torch.arange(far * c, (far + 1) * c))
    return torch.cat(perm)
