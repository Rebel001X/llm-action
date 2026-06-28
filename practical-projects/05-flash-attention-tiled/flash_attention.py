# -*- coding: utf-8 -*-
"""
从零实现 FlashAttention（分块 tiling + 在线 softmax / online-softmax）
================================================================

这是什么
--------
本文件用纯 PyTorch（CPU 即可跑）从零实现 FlashAttention 的核心算法，并与
朴素注意力（naive attention）做逐位数值对比，验证「FlashAttention 不是近似、
而是精确的内存重排」这一结论。

核心思想（对应 ../../llm-optimizer/FlashAttention.md）：
  - 朴素注意力先把 S = Q @ K^T 这张 n×n 的分数矩阵整张物化（materialize）到
    显存，再 softmax，再乘 V。瓶颈是这张 n×n 矩阵的反复读写，显存是 O(n^2)。
  - FlashAttention 把 Q、K、V 沿序列维切成小块（tile），逐块在「片上」计算，
    用 online softmax 维护每个 query 行的 running max(m) / running sum(l) /
    running 未归一化输出(O)，边算边累加，永不物化完整的 n×n 矩阵。
    每个 query 行只需 O(d) 的状态，总额外显存从 O(n^2) 降到 O(n)。

online softmax 的灵魂（递推）：
  m_new = max(m_old, m_blk)                     # running max 只增不减
  rescale = exp(m_old - m_new)  (<= 1)          # 把旧累加器翻译到新基准
  l_new = rescale * l_old + rowsum(exp(S_blk - m_new))
  O_new = rescale * O_old + exp(S_blk - m_new) @ V_blk
  ...（所有块处理完）...
  O = O / l                                     # 收尾只除一次

关键认知：FlashAttention 省的是「显存占用 + HBM 访存次数」，不是 FLOPs。
本 demo 在 CPU 上不会复现 GPU 的加速（CPU 无 SRAM/HBM 层级），但能完整、
精确地复现「分块 + online softmax = 标准注意力」的数值等价性与 O(n) 显存账。

怎么跑：
    python flash_attention.py
"""

import math
import torch

# ---- 可复现：固定随机种子，CPU、单线程，避免不同机器结果漂移 ----
torch.manual_seed(0)
DTYPE = torch.float64  # 用 float64 做对拍，把数值误差从「算法」里彻底剥离出来


# ======================================================================
# 1. 朴素注意力（baseline）：会显式物化 n×n 的分数矩阵 S 和权重矩阵 P
# ======================================================================
def naive_attention(Q, K, V):
    """
    标准注意力：O = softmax(Q K^T / sqrt(d)) V
    形状：Q,K,V 均为 (n, d)，返回 O 为 (n, d)。
    注意这里 S、P 都是 (n, n)，显存 O(n^2)——这正是 FlashAttention 要消灭的东西。
    """
    n, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.transpose(-1, -2)) * scale          # (n, n) 整张物化  <-- O(n^2)
    # 标准的数值稳定 softmax：先减每行最大值再取 exp，防止 e^x 上溢
    m = S.max(dim=-1, keepdim=True).values          # 每行最大值 (n, 1)
    P = torch.exp(S - m)                            # (n, n) 又一张 O(n^2) 矩阵
    P = P / P.sum(dim=-1, keepdim=True)             # 按行归一化
    O = P @ V                                       # (n, d)
    return O


# ======================================================================
# 2. FlashAttention（分块 + online softmax），永不物化 n×n 矩阵
#    实现采用 v2 视角：外层遍历 Q 块，内层遍历 K/V 块。
# ======================================================================
def flash_attention(Q, K, V, block_q=16, block_k=16):
    """
    分块 + online softmax 版注意力，与 naive_attention 数值等价。

    block_q / block_k：Q 块行数 Br、K/V 块行数 Bc。真实 kernel 里它们由 SRAM
    容量决定（典型 64/128）；这里取小值便于在 toy 规模上清楚展示分块逻辑。

    关键：全程只用到大小为 O(Br*Bc) 的临时分数块 S_blk 和每行 O(d) 的累加器，
    从不分配 (n, n) 的中间矩阵——这就是 O(n) 额外显存的来源。
    """
    n, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    device, dtype = Q.device, Q.dtype

    # 输出缓冲（最终结果）。其余都是每行的小状态，分块循环里就地更新。
    O = torch.zeros(n, d, device=device, dtype=dtype)

    # 外层：遍历 Q 行块。每个 Q 块独立维护自己的 online-softmax 状态。
    for i in range(0, n, block_q):
        qi = Q[i : i + block_q]                     # (Br, d)，放进「片上」的那块
        br = qi.shape[0]

        # ---- 每个 query 行的 running 状态（online softmax 的全部记忆）----
        m = torch.full((br, 1), float("-inf"), device=device, dtype=dtype)  # running max
        l = torch.zeros((br, 1), device=device, dtype=dtype)               # running 分母 sum
        acc = torch.zeros((br, d), device=device, dtype=dtype)            # running 未归一化输出

        # 内层：依次把 K/V 块搬进来，online 地把它「融合」进当前状态
        for j in range(0, n, block_k):
            kj = K[j : j + block_k]                 # (Bc, d)
            vj = V[j : j + block_k]                 # (Bc, d)

            # (1) 算当前小块的分数：只有 Br×Bc 大，绝不会是 n×n
            s_blk = (qi @ kj.transpose(-1, -2)) * scale   # (Br, Bc)

            # (2) 块内每行最大值，并更新全局 running max（只增不减）
            m_blk = s_blk.max(dim=-1, keepdim=True).values    # (Br, 1)
            m_new = torch.maximum(m, m_blk)                   # (Br, 1)

            # (3) rescale 修正因子 = exp(m_old - m_new) <= 1
            #     它把「按旧 max 算的旧累加器」翻译到「按新 max」的基准上。
            #     这是 online softmax 数值稳定 + 正确性的灵魂。
            rescale = torch.exp(m - m_new)                    # (Br, 1)

            # (4) 当前块的未归一化权重 p = exp(S_blk - m_new)
            p = torch.exp(s_blk - m_new)                      # (Br, Bc)

            # (5) 在线更新分母 l 和 未归一化输出累加器 acc：
            #     旧的部分统一乘 rescale，再加上新块的贡献。
            l = rescale * l + p.sum(dim=-1, keepdim=True)     # (Br, 1)
            acc = rescale * acc + p @ vj                      # (Br, d)

            # (6) 滚动 running max
            m = m_new

        # ---- 收尾：所有 K/V 块处理完，对该 Q 块做唯一一次归一化（除以 l）----
        O[i : i + block_q] = acc / l

    return O


# ======================================================================
# 3. 显存账（理论估算）：朴素 O(n^2) vs FlashAttention O(n)
#    在 CPU demo 上无法真测 HBM，这里按元素个数给出「中间矩阵」占用对比。
# ======================================================================
def memory_footprint_elements(n, d, block_q, block_k):
    """
    返回两种实现「额外中间状态」需要的元素个数（不含 Q/K/V/O 本身）。
    朴素：要物化 S(n*n) + P(n*n)。
    Flash：每个 Q 块只需 S_blk(Br*Bc) + 每行状态 m,l(Br) + acc(Br*d)。
    """
    naive_extra = 2 * n * n                                  # S + P，两张 n×n
    flash_extra = block_q * block_k + 2 * block_q + block_q * d
    return naive_extra, flash_extra


def main():
    # -------- toy 规模：CPU 几秒内跑完，又足够暴露分块/边界逻辑 --------
    n, d = 128, 32                  # 序列长 128、头维 32
    block_q, block_k = 16, 24       # 故意取「除不尽 n」的块，测试尾块/边界正确性
    print("=" * 64)
    print("FlashAttention from scratch: tiling + online softmax")
    print("=" * 64)
    print(f"seq_len n = {n}, head_dim d = {d}")
    print(f"block_q (Br) = {block_q}, block_k (Bc) = {block_k}  "
          f"(n not divisible -> exercises ragged tail blocks)")
    print("-" * 64)

    Q = torch.randn(n, d, dtype=DTYPE)
    K = torch.randn(n, d, dtype=DTYPE)
    V = torch.randn(n, d, dtype=DTYPE)

    # ---- 正确性对拍：FlashAttention 应与朴素注意力逐位相等 ----
    O_naive = naive_attention(Q, K, V)
    O_flash = flash_attention(Q, K, V, block_q=block_q, block_k=block_k)

    max_abs_err = (O_naive - O_flash).abs().max().item()
    mean_abs_err = (O_naive - O_flash).abs().mean().item()
    is_close = torch.allclose(O_naive, O_flash, atol=1e-10, rtol=1e-8)

    print("[correctness] flash vs naive attention output")
    print(f"  max  abs error = {max_abs_err:.3e}")
    print(f"  mean abs error = {mean_abs_err:.3e}")
    print(f"  torch.allclose(atol=1e-10) = {is_close}")
    print(f"  -> RESULT: {'PASS (numerically identical)' if is_close else 'FAIL'}")
    print("-" * 64)

    # ---- 边角案例：单块即覆盖全序列时，online softmax 应退化为标准 softmax ----
    O_oneblock = flash_attention(Q, K, V, block_q=n, block_k=n)
    one_err = (O_naive - O_oneblock).abs().max().item()
    print("[edge case] single block (Br=Bc=n) degenerates to plain softmax")
    print(f"  max abs error = {one_err:.3e}  "
          f"({'PASS' if one_err < 1e-10 else 'FAIL'})")
    print("-" * 64)

    # ---- 显存账：n 增大时朴素 O(n^2) 爆炸，Flash 额外状态保持小且不随 n^2 长 ----
    print("[memory] extra intermediate-state elements (NOT counting Q/K/V/O)")
    print(f"  {'n':>6} | {'naive O(n^2)':>14} | {'flash (per Qblk)':>16} | {'ratio':>10}")
    for nn in [128, 512, 2048, 8192]:
        nv, fl = memory_footprint_elements(nn, d, block_q, block_k)
        print(f"  {nn:>6} | {nv:>14,} | {fl:>16,} | {nv / fl:>9.1f}x")
    print("  note: naive grows ~n^2; flash per-block state is constant in n.")
    print("        on real GPUs this O(n^2)->O(n) is what enables long context.")
    print("-" * 64)

    # ---- 简单吞吐/正确性总览（多组随机输入重复验证，确保不是偶然）----
    worst = 0.0
    trials = 20
    for t in range(trials):
        q = torch.randn(n, d, dtype=DTYPE)
        k = torch.randn(n, d, dtype=DTYPE)
        v = torch.randn(n, d, dtype=DTYPE)
        e = (naive_attention(q, k, v)
             - flash_attention(q, k, v, block_q=block_q, block_k=block_k)).abs().max().item()
        worst = max(worst, e)
    print(f"[stress] {trials} random trials, worst-case max abs error = {worst:.3e}")
    print(f"  -> {'ALL PASS' if worst < 1e-9 else 'SOME FAIL'}")
    print("=" * 64)
    print("DONE. FlashAttention reproduces naive attention exactly; "
          "extra memory is O(n), not O(n^2).")


if __name__ == "__main__":
    main()
