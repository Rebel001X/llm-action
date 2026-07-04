"""
spec_decode.py —— 投机解码(speculative decoding)的期望加速建模与蒙特卡洛模拟

对应架构篇 ../../09_推理引擎架构...md 的「5. 投机解码」小节。

核心思想(草稿-验证 draft-then-verify):
  1. 草稿模型(draft model,便宜快)一次自回归猜出 k 个候选 token;
  2. 目标大模型(target model)**一次前向**并行验证这 k+1 个位置;
  3. 从头逐位比对,接受最长匹配前缀(接受 n 个),在第一个被拒处从大模型
     修正分布重采 1 个 token —— 于是一轮拿到 n+1 个 token(1 ≤ n+1 ≤ k+1),
     且**数学上无损**(拒绝采样修正保证输出分布 = 大模型直接采样)。

本模块提供:
  · 闭式期望         expected_tokens(alpha, k)  = (1 - α^{k+1}) / (1 - α)
  · 蒙特卡洛模拟     simulate_mean_tokens(...)  与闭式对拍
  · 期望加速比       expected_speedup(alpha, k, c) = E / (1 + c·k)
  · 最优草稿长度     optimal_k(alpha, c, k_max) —— 存在内部极大值
  · 盈亏平衡接受率   breakeven_alpha(k, c) —— 加速比 = 1 的临界 α

术语约定:
  α (alpha)  : 单个草稿 token 被接受的平均概率(acceptance rate),i.i.d. 近似。
  k          : 一轮草稿提出的 token 数(draft length,doc 里记作 K)。
  c          : 草稿模型 / 目标模型 的单步成本比(draft-to-target cost ratio,常见 0.1~0.2)。
  E          : 一轮期望"敲定"的 token 数(含拒绝点/末尾那 1 个 bonus token)。
"""
from __future__ import annotations
import numpy as np

# 草稿模型相对目标模型的单步成本比(默认草稿约便宜 5 倍)
DEFAULT_C = 0.2


# ----------------------------------------------------------------------------
# 1) 闭式期望:一轮敲定的 token 数
# ----------------------------------------------------------------------------
def expected_tokens(alpha: float, k: int) -> float:
    r"""一轮投机解码期望敲定的 token 数(闭式)。

    推导(i.i.d. 接受近似):设接受的草稿前缀长度为 X(0 ≤ X ≤ k)。
    因逐位比对、遇到第一个拒绝即停,故 P(X ≥ i) = α^i(前 i 个都被接受)。
    一轮敲定的 token 数 = X + 1(那 +1 是拒绝点重采 / 全接受时的 bonus token)。

        E = 1 + Σ_{i=1..k} P(X ≥ i) = 1 + Σ_{i=1..k} α^i
          = 1 + α(1 - α^k)/(1 - α)
          = (1 - α^{k+1}) / (1 - α)

    边界:
      α → 0  ⇒ E = 1     (草稿几乎全被拒,仅拿到那 1 个修正 token)
      α → 1  ⇒ E = k + 1  (草稿全被接受 + 1 个 bonus)
    """
    if k < 0:
        raise ValueError("k 必须 ≥ 0")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必须落在 [0, 1]")
    if alpha >= 1.0:
        return float(k + 1)          # 极限:全接受 + bonus
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


# ----------------------------------------------------------------------------
# 2) 蒙特卡洛模拟:直接掷骰子统计一轮敲定的 token 数
# ----------------------------------------------------------------------------
def simulate_mean_tokens(alpha: float, k: int, n_steps: int = 200_000,
                         seed: int = 0) -> float:
    """模拟大量投机解码轮次,返回实际平均敲定 token 数(应 ≈ expected_tokens)。

    实现(向量化):对每一轮独立掷 k 次 Bernoulli(α)。接受的前缀长度 =
    "开头连续 True 的个数"。用累积乘积 cumprod 巧妙求前缀长度:一旦出现
    第一个 False,cumprod 就变 0,后续保持 0 —— 于是每行 cumprod 的和恰好
    等于开头连续 True 的长度(accepted)。敲定 token 数 = accepted + 1。
    """
    if k == 0:
        return 1.0                   # 不提草稿,退化为普通自回归:每轮 1 个 token
    rng = np.random.default_rng(seed)
    accepts = rng.random((n_steps, k)) < alpha          # (n_steps, k) 的 bool
    leading = np.cumprod(accepts, axis=1).sum(axis=1)    # 每行开头连续 True 的长度
    tokens = leading + 1                                  # +1: 拒绝点重采 / bonus
    return float(tokens.mean())


def simulate_step(alpha: float, k: int, rng: np.random.Generator) -> int:
    """模拟单轮,返回本轮敲定的 token 数(1 ≤ 返回值 ≤ k+1)。教学用、逐轮可读。"""
    accepted = 0
    for _ in range(k):
        if rng.random() < alpha:
            accepted += 1
        else:
            break                    # 遇到第一个拒绝即停,后面全丢弃
    return accepted + 1


# ----------------------------------------------------------------------------
# 3) 期望加速比:把"草稿成本"算进去
# ----------------------------------------------------------------------------
def expected_speedup(alpha: float, k: int, c: float = DEFAULT_C) -> float:
    r"""相对普通自回归的期望加速比(Leviathan 2023 式)。

    成本模型:一轮里草稿模型自回归跑 k 步(每步耗 c 个"目标单位"),目标模型
    再做 1 次并行验证(耗 1 个目标单位)。故一轮成本 ≈ (c·k + 1) 目标单位,
    产出 E 个 token。普通自回归是 1 个目标单位产 1 个 token,故:

        speedup = E / (1 + c·k) = (1 - α^{k+1}) / [ (1 - α)(1 + c·k) ]

    speedup > 1 才真正划算;α 低或 c 高时可能 < 1(白付草稿成本反而更慢)。
    """
    if c < 0:
        raise ValueError("成本比 c 必须 ≥ 0")
    return expected_tokens(alpha, k) / (1.0 + c * k)


def simulate_speedup(alpha: float, k: int, c: float = DEFAULT_C,
                     n_steps: int = 200_000, seed: int = 0) -> float:
    """用蒙特卡洛平均 token 数换算实测加速比,应 ≈ expected_speedup。"""
    mean_tok = simulate_mean_tokens(alpha, k, n_steps=n_steps, seed=seed)
    return mean_tok / (1.0 + c * k)


# ----------------------------------------------------------------------------
# 4) 最优草稿长度:随 k 增大,分子饱和、分母线性增长 ⇒ 存在内部极大值
# ----------------------------------------------------------------------------
def optimal_k(alpha: float, c: float = DEFAULT_C, k_max: int = 20) -> int:
    r"""在 k ∈ [1, k_max] 上使期望加速比最大的草稿长度 k*。

    直觉:E(k) 随 k 增大而饱和到上界 1/(1-α)(边际接受收益递减),而成本
    (1 + c·k) 线性增长 —— 两者之比必在某个 k* 处达到峰值后回落。α 越高、
    c 越小,饱和越慢、k* 越大。
    """
    ks = np.arange(1, k_max + 1)
    speeds = np.array([expected_speedup(alpha, int(k), c) for k in ks])
    return int(ks[int(np.argmax(speeds))])


def optimal_speedup(alpha: float, c: float = DEFAULT_C, k_max: int = 20) -> float:
    """最优 k* 处能拿到的加速比。"""
    return expected_speedup(alpha, optimal_k(alpha, c, k_max), c)


# ----------------------------------------------------------------------------
# 5) 盈亏平衡接受率:speedup(α*, k, c) = 1 时的临界 α(二分求根)
# ----------------------------------------------------------------------------
def breakeven_alpha(k: int, c: float = DEFAULT_C) -> float:
    """给定 k、c,求使期望加速比恰好 = 1 的接受率 α*(低于它就"白忙一场变慢")。

    speedup(α) 在 α∈(0,1) 上单调递增(k、c 固定),故可二分。若 α=1 时仍
    speedup ≤ 1(即草稿太贵,c·k ≥ k,几乎不可能),返回 1.0。
    """
    lo, hi = 0.0, 1.0 - 1e-12
    if expected_speedup(hi, k, c) <= 1.0:
        return 1.0                   # 再高的接受率也回不了本
    if expected_speedup(lo, k, c) >= 1.0:
        return 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if expected_speedup(mid, k, c) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# 便捷:批量算加速比网格(供 run_demo 画热图)
# ----------------------------------------------------------------------------
def speedup_grid(alphas, ks, c: float = DEFAULT_C) -> np.ndarray:
    """返回形状 (len(alphas), len(ks)) 的加速比矩阵。"""
    grid = np.empty((len(alphas), len(ks)))
    for i, a in enumerate(alphas):
        for j, k in enumerate(ks):
            grid[i, j] = expected_speedup(float(a), int(k), c)
    return grid
