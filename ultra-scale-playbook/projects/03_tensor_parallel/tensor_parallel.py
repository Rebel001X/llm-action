"""
tensor_parallel.py —— 从零实现张量并行 TP(列并行 / 行并行 Linear、并行 MLP、按头切注意力)

对应《Ultra-Scale Playbook》第 4 章(张量并行 + 序列并行)。

线性层 y = x @ W^T + b,W 形状 (out, in)。
  · 列并行 ColumnParallel:按 out 维把 W 切成 p 段 W_i(out/p, in)。
      各 rank 算 Y_i = x @ W_i^T,拼接得完整 Y。前向无需通信,反向对 x 的梯度需 all-reduce(求和)。
  · 行并行 RowParallel:按 in 维把 W 切成 p 段 W_i(out, in/p),x 也按列切 x_i。
      各 rank 算部分和 x_i @ W_i^T,前向 all-reduce(求和)得 Y。
  · Megatron MLP:第一层列并行(h→4h)、GELU 逐元素(切了也不用通信)、第二层行并行(4h→h)。
      → 每个 MLP 前向 1 次 all-reduce、反向 1 次 all-reduce。
  · 注意力:按注意力头 heads 切分(每 rank 一部分头)。

金标准:并行前向/反向的结果(输出 + 梯度)必须 == 单卡整体计算(逐元素相等)。
本文件是单进程可验证的核心;真·多进程见 run_demo.py。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 参考实现(单卡)
# ----------------------------------------------------------------------------
def linear(x, W, b=None):
    return F.linear(x, W, b)


def split_cols_of_output(W, p):
    """按输出维(行方向,W 的第 0 维)切成 p 段 —— 列并行。"""
    return list(torch.chunk(W, p, dim=0))


def split_rows_of_input(W, p):
    """按输入维(列方向,W 的第 1 维)切成 p 段 —— 行并行。"""
    return list(torch.chunk(W, p, dim=1))


# ----------------------------------------------------------------------------
# 列并行 / 行并行 前向(单进程模拟 p 个分片)
# ----------------------------------------------------------------------------
def column_parallel_forward(x, W_shards, b_shards=None):
    """各 rank 算 Y_i=x@W_i^T,拼接(all-gather)得完整输出。"""
    outs = []
    for i, Wi in enumerate(W_shards):
        bi = b_shards[i] if b_shards is not None else None
        outs.append(linear(x, Wi, bi))
    return torch.cat(outs, dim=-1)


def row_parallel_forward(x, W_shards):
    """x 按输入维切分,各 rank 算部分和,all-reduce(求和)得完整输出。"""
    p = len(W_shards)
    x_shards = torch.chunk(x, p, dim=-1)
    partial = [linear(xi, Wi) for xi, Wi in zip(x_shards, W_shards)]
    return sum(partial)


# ----------------------------------------------------------------------------
# 并行 MLP:列并行 → GELU → 行并行
# ----------------------------------------------------------------------------
def reference_mlp(x, W1, b1, W2, b2):
    h = F.gelu(linear(x, W1, b1))
    return linear(h, W2, b2)


def parallel_mlp(x, W1, b1, W2, b2, p):
    # 第一层列并行:W1 (4h, h) 按 out 切;b1 (4h,) 也按 out 维(dim=0)切成 p 段
    W1_shards = split_cols_of_output(W1, p)
    b1_shards = split_cols_of_output(b1, p)                          # b1 是 1-D,dim=0 即 out 维
    h_parallel = column_parallel_forward(x, W1_shards, b1_shards)   # (batch, 4h) 拼回
    h_act = F.gelu(h_parallel)                                      # 逐元素,切不切都一样
    # 第二层行并行:W2 (h, 4h) 按 in 切;偏置只加一次(在 all-reduce 之后)
    W2_shards = split_rows_of_input(W2, p)
    y = row_parallel_forward(h_act, W2_shards)
    return y + b2


# ----------------------------------------------------------------------------
# 按头切分的多头自注意力
# ----------------------------------------------------------------------------
def reference_attention(x, Wq, Wk, Wv, Wo, num_heads):
    B, S, H = x.shape
    d = H // num_heads
    q = linear(x, Wq).view(B, S, num_heads, d).transpose(1, 2)     # (B, nh, S, d)
    k = linear(x, Wk).view(B, S, num_heads, d).transpose(1, 2)
    v = linear(x, Wv).view(B, S, num_heads, d).transpose(1, 2)
    att = torch.softmax(q @ k.transpose(-1, -2) / d ** 0.5, dim=-1)
    o = (att @ v).transpose(1, 2).reshape(B, S, H)                 # 合并头
    return linear(o, Wo)


def parallel_attention(x, Wq, Wk, Wv, Wo, num_heads, p):
    """把 num_heads 平均分到 p 个 rank;各 rank 独立算自己的头,输出投影用行并行。"""
    assert num_heads % p == 0
    B, S, H = x.shape
    d = H // num_heads
    heads_per = num_heads // p
    # QKV 列并行:按头切 = 按 out 维切成 p 段
    Wq_s, Wk_s, Wv_s = (split_cols_of_output(W, p) for W in (Wq, Wk, Wv))
    Wo_s = split_rows_of_input(Wo, p)     # 输出投影行并行(按 in=H 维切)
    partial_outs = []
    for r in range(p):
        q = linear(x, Wq_s[r]).view(B, S, heads_per, d).transpose(1, 2)
        k = linear(x, Wk_s[r]).view(B, S, heads_per, d).transpose(1, 2)
        v = linear(x, Wv_s[r]).view(B, S, heads_per, d).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / d ** 0.5, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, S, heads_per * d)   # 本 rank 的头拼起来
        partial_outs.append(linear(o, Wo_s[r]))                      # 行并行部分和
    return sum(partial_outs)                                          # all-reduce
