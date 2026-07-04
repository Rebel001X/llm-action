# -*- coding: utf-8 -*-
"""
ab_stats.py —— A/B 测试统计核心库（纯 numpy，零 scipy 依赖）

本模块把《AI Model Evaluation》第 6 章「在线评估与 A/B 测试」里
6.4（统计地基）与 6.5（偷看陷阱）用**可运行的代码**落地：

    1. 分布函数：正态 CDF/PPF、t 分布 CDF/PPF（自实现，避免依赖 scipy）
    2. 两比例检验（two-proportion z-test）—— 转化率/CTR 这类 0/1 指标
    3. 两均值检验（two-sample t-test，Welch）—— 停留时长/客单价这类连续指标
    4. 置信区间（CI）—— 效应量的区间估计
    5. 样本量估算（sample size / power analysis）—— 开测前算需要多少人
    6. 统计功效（power）—— 给定样本量能检出多大效应的概率
    7. 偷看模拟（peeking simulation）—— 演示反复看 p 值如何抬高假阳性率

设计原则（第一性原理）：
    - 所有"查表"的分布分位数都用**数值方法**从零算出来，方便逐行讲解，
      也让整个库离线、无外部依赖。
    - 每个函数都返回**具名结果对象**（dataclass），字段自解释，便于教学与测试。

作者注：本文件注释用中文，术语中英并列，配套 README.md 逐行讲解。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# =============================================================================
# 第 0 部分：分布函数（自实现，替代 scipy.stats）
# =============================================================================
# 为什么要自己写？—— 教学库要"透明可讲"，且本机无网络、不想引 scipy。
# 正态分布的 CDF 可以用误差函数 erf 精确表达，Python 标准库 math.erf 就够用。
# t 分布稍复杂，用正则不完全 Beta 函数（自实现连分数展开）算 CDF，
# 再用二分法反解出分位数 PPF。精度对 A/B 测试完全够用（误差 < 1e-8）。


def norm_cdf(x: float) -> float:
    """标准正态分布 N(0,1) 的累积分布函数 CDF：P(Z <= x)。

    数学：Φ(x) = 1/2 * [1 + erf(x / sqrt(2))]。
    erf 是误差函数（error function），math 库直接提供，数值精度很高。
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_sf(x: float) -> float:
    """生存函数 survival function：P(Z > x) = 1 - Φ(x)。右尾概率常用它。"""
    return 1.0 - norm_cdf(x)


def norm_ppf(p: float) -> float:
    """标准正态分位数函数 PPF（percent point function），即 CDF 的反函数。

    给定累积概率 p，返回 z 使得 Φ(z) = p。
    实现：Acklam 有理逼近算法（业界经典），最大绝对误差约 1.15e-9。
    用途：算临界值 z_{α/2}（如 p=0.975 → z≈1.96）。
    """
    if not (0.0 < p < 1.0):
        raise ValueError("norm_ppf 需要 0 < p < 1，收到 %r" % (p,))
    # Acklam 逼近的系数（a,b 为中心区，c,d 为尾部区）
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:  # 左尾：用尾部逼近
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:  # 右尾：对称使用尾部逼近
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # 中心区：有理逼近
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _betacf(a: float, b: float, x: float) -> float:
    """正则不完全 Beta 函数 I_x(a,b) 的连分数（continued fraction）展开。

    这是 Numerical Recipes 里的经典实现，供 t 分布 CDF 使用。
    """
    MAXIT, EPS, FPMIN = 200, 3.0e-14, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则不完全 Beta 函数 I_x(a,b)，取值 [0,1]，t 分布 CDF 的核心。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # 前置系数 bt = x^a (1-x)^b / B(a,b)，用 lgamma 保数值稳定
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a  # 直接展开收敛快
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b  # 用对称性换到收敛快的一侧


def t_cdf(t: float, df: float) -> float:
    """学生 t 分布的 CDF：P(T <= t)，自由度 df。

    数学：用正则不完全 Beta 表达。x = df/(df+t^2)，
    则 P(T<=t) = 1 - 0.5*I_x(df/2, 1/2)（t>0），t<0 由对称性得。
    """
    x = df / (df + t * t)
    ib = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - ib if t > 0 else ib


def t_sf(t: float, df: float) -> float:
    """t 分布右尾概率 P(T > t) = 1 - t_cdf。"""
    return 1.0 - t_cdf(t, df)


def t_ppf(p: float, df: float) -> float:
    """t 分布分位数（CDF 的反函数），用二分法求解。

    给定累积概率 p 与自由度 df，返回 t 使 t_cdf(t,df)=p。
    大自由度时 t 分布趋近正态，因此用 norm_ppf 作为搜索区间参考。
    """
    if not (0.0 < p < 1.0):
        raise ValueError("t_ppf 需要 0 < p < 1")
    # 以正态分位数为中心，取一个足够宽的对称搜索区间
    lo, hi = -100.0, 100.0
    for _ in range(200):  # 二分 200 次，区间宽度缩到 2e-58，远超所需精度
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# 第 1 部分：两比例检验（two-proportion z-test）—— 0/1 型指标
# =============================================================================
# 场景：A 组（对照/control）转化 x_a / n_a，B 组（实验/treatment）转化 x_b / n_b。
# 问题：B 的转化率是否真的和 A 不同（或更高）？
# H0：p_a = p_b（两组转化率相等）。


@dataclass
class TestResult:
    """一次假设检验的完整结果（比例或均值检验通用）。"""
    statistic: float      # 检验统计量（z 或 t）
    p_value: float        # p 值
    effect: float         # 效应量（点估计：p_b - p_a 或 mean_b - mean_a）
    ci_low: float         # 效应量置信区间下界
    ci_high: float        # 效应量置信区间上界
    alpha: float          # 显著性水平
    alternative: str      # 备择假设方向：'two-sided' | 'larger' | 'smaller'
    df: Optional[float] = None  # 自由度（仅 t 检验有）

    @property
    def significant(self) -> bool:
        """是否统计显著：p < alpha。"""
        return self.p_value < self.alpha


def _p_from_z(z: float, alternative: str) -> float:
    """根据备择假设方向，把 z 统计量换算成 p 值。"""
    if alternative == "two-sided":
        return 2.0 * norm_sf(abs(z))       # 双尾：两侧尾概率之和
    if alternative == "larger":
        return norm_sf(z)                  # 右尾：P(Z > z)
    if alternative == "smaller":
        return norm_cdf(z)                 # 左尾：P(Z < z)
    raise ValueError("alternative 须为 two-sided/larger/smaller")


def two_proportion_ztest(
    x_a: int, n_a: int, x_b: int, n_b: int,
    alpha: float = 0.05, alternative: str = "two-sided",
) -> TestResult:
    """两比例 z 检验：比较 B 组与 A 组的转化率。

    参数：
        x_a, n_a：A 组转化数、总人数（如 A 组 5000 人点了 500 次）
        x_b, n_b：B 组转化数、总人数
        alpha：显著性水平（默认 0.05）
        alternative：'two-sided'(默认) / 'larger'(检验 B>A) / 'smaller'(B<A)

    统计量（合并方差 pooled 版，用于 p 值）：
        p_pool = (x_a + x_b) / (n_a + n_b)                 # H0 下两组同率，合并估计
        SE_pool = sqrt(p_pool*(1-p_pool)*(1/n_a + 1/n_b))  # H0 下差值的标准误
        z = (p_b - p_a) / SE_pool

    置信区间用**非合并（unpooled）** 标准误——因为 CI 描述真实差异，
    不应假设 H0 成立：
        SE_unpool = sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
        CI = (p_b - p_a) ± z_{crit} * SE_unpool
    """
    p_a = x_a / n_a
    p_b = x_b / n_b
    effect = p_b - p_a

    # --- 检验统计量：合并标准误（H0 下的方差估计） ---
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z = effect / se_pool if se_pool > 0 else 0.0
    p_value = _p_from_z(z, alternative)

    # --- 置信区间：非合并标准误（不假设 H0） ---
    se_unpool = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    if alternative == "two-sided":
        z_crit = norm_ppf(1 - alpha / 2)
        ci_low, ci_high = effect - z_crit * se_unpool, effect + z_crit * se_unpool
    elif alternative == "larger":
        z_crit = norm_ppf(1 - alpha)
        ci_low, ci_high = effect - z_crit * se_unpool, math.inf
    else:  # smaller
        z_crit = norm_ppf(1 - alpha)
        ci_low, ci_high = -math.inf, effect + z_crit * se_unpool

    return TestResult(statistic=z, p_value=p_value, effect=effect,
                      ci_low=ci_low, ci_high=ci_high, alpha=alpha,
                      alternative=alternative)


# =============================================================================
# 第 2 部分：两均值检验（Welch's two-sample t-test）—— 连续型指标
# =============================================================================
# 场景：A/B 两组的停留时长、客单价这类连续变量。Welch t 检验不假设两组方差相等，
# 是更稳健的默认选择（Student t 假设等方差，现实很少成立）。


def two_mean_ttest(
    a: np.ndarray, b: np.ndarray,
    alpha: float = 0.05, alternative: str = "two-sided",
) -> TestResult:
    """Welch 两样本 t 检验：比较两组连续指标的均值。

    参数：
        a, b：两组的观测数组（一维 numpy 数组）
        alpha, alternative：同上

    统计量：
        mean_diff = mean_b - mean_a
        SE = sqrt(var_a/n_a + var_b/n_b)          # 用样本方差(ddof=1)
        t = mean_diff / SE
        df 用 Welch–Satterthwaite 近似（非整数自由度）
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.size, b.size
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)  # ddof=1：无偏样本方差
    effect = mean_b - mean_a

    se = math.sqrt(var_a / n_a + var_b / n_b)
    t = effect / se if se > 0 else 0.0

    # Welch–Satterthwaite 自由度近似（两组方差不等时的有效自由度）
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den if den > 0 else (n_a + n_b - 2)

    # p 值
    if alternative == "two-sided":
        p_value = 2.0 * t_sf(abs(t), df)
    elif alternative == "larger":
        p_value = t_sf(t, df)
    elif alternative == "smaller":
        p_value = t_cdf(t, df)
    else:
        raise ValueError("alternative 非法")

    # 置信区间（非合并，t 临界值）
    if alternative == "two-sided":
        t_crit = t_ppf(1 - alpha / 2, df)
        ci_low, ci_high = effect - t_crit * se, effect + t_crit * se
    elif alternative == "larger":
        t_crit = t_ppf(1 - alpha, df)
        ci_low, ci_high = effect - t_crit * se, math.inf
    else:
        t_crit = t_ppf(1 - alpha, df)
        ci_low, ci_high = -math.inf, effect + t_crit * se

    return TestResult(statistic=t, p_value=p_value, effect=effect,
                      ci_low=ci_low, ci_high=ci_high, alpha=alpha,
                      alternative=alternative, df=df)


# =============================================================================
# 第 3 部分：样本量估算（power analysis）—— 开测前算需要多少人
# =============================================================================


def sample_size_two_proportions(
    p_baseline: float, mde_abs: float,
    alpha: float = 0.05, power: float = 0.80, alternative: str = "two-sided",
) -> int:
    """两比例检验的每组样本量估算（近似公式）。

    参数：
        p_baseline：基线转化率（A 组预期率，如 0.10）
        mde_abs：最小可检测效应（绝对值，如 +0.02 表示要检出 10%→12%）
        alpha：显著性水平
        power：目标功效（1-β，行业惯例 0.80）
        alternative：'two-sided' 用 z_{α/2}，单尾用 z_α

    公式（等样本、正态近似）：
        n = (z_{α'} + z_β)^2 * [p1(1-p1) + p2(1-p2)] / (p2 - p1)^2
      其中 p1=p_baseline, p2=p_baseline+mde, z_β=norm_ppf(power)。
      向上取整（ceil），因为样本量必须是整数且宁多勿少。
    """
    p1 = p_baseline
    p2 = p_baseline + mde_abs
    z_alpha = norm_ppf(1 - alpha / 2) if alternative == "two-sided" else norm_ppf(1 - alpha)
    z_beta = norm_ppf(power)
    var_sum = p1 * (1 - p1) + p2 * (1 - p2)
    n = (z_alpha + z_beta) ** 2 * var_sum / (mde_abs ** 2)
    return int(math.ceil(n))


def sample_size_two_means(
    sigma: float, mde_abs: float,
    alpha: float = 0.05, power: float = 0.80, alternative: str = "two-sided",
) -> int:
    """两均值检验的每组样本量估算（连续指标，等方差近似）。

    公式：n = 2 * (z_{α'} + z_β)^2 * σ^2 / Δ^2
      σ 为组内标准差，Δ 为要检出的均值差（MDE）。
      因子 2 来自"两组各 n，差值方差 = 2σ²/n"。
    """
    z_alpha = norm_ppf(1 - alpha / 2) if alternative == "two-sided" else norm_ppf(1 - alpha)
    z_beta = norm_ppf(power)
    n = 2.0 * (z_alpha + z_beta) ** 2 * (sigma ** 2) / (mde_abs ** 2)
    return int(math.ceil(n))


# =============================================================================
# 第 4 部分：统计功效（power）—— 给定样本量能检出效应的概率
# =============================================================================


def power_two_proportions(
    p_baseline: float, mde_abs: float, n_per_group: int,
    alpha: float = 0.05, alternative: str = "two-sided",
) -> float:
    """给定每组样本量，计算两比例检验的统计功效（1-β）。

    思路（正态近似）：真实效应 δ=mde，真实差值分布近似
        N(δ, SE_alt^2)，SE_alt = sqrt(p1(1-p1)/n + p2(1-p2)/n)。
    在 H0 下临界值 = z_crit * SE_null。功效 = P(观测差 > 临界值 | 真实 δ)。
    这里用常见近似：power = Φ( δ/SE_alt − z_crit )（双尾再加极小的反向尾，通常忽略）。
    """
    p1 = p_baseline
    p2 = p_baseline + mde_abs
    se_alt = math.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z_crit = norm_ppf(1 - alpha / 2) if alternative == "two-sided" else norm_ppf(1 - alpha)
    # 主尾功效
    z = abs(mde_abs) / se_alt - z_crit
    power = norm_cdf(z)
    if alternative == "two-sided":
        # 反向尾（效应落在另一侧被判显著的概率），量级极小，加上更精确
        power += norm_cdf(-abs(mde_abs) / se_alt - z_crit)
    return float(min(max(power, 0.0), 1.0))


def power_two_means(
    sigma: float, mde_abs: float, n_per_group: int,
    alpha: float = 0.05, alternative: str = "two-sided",
) -> float:
    """给定每组样本量，计算两均值检验的统计功效（连续指标）。

    SE = sqrt(2)*σ/sqrt(n)；power = Φ(Δ/SE − z_crit)。
    """
    se = math.sqrt(2.0) * sigma / math.sqrt(n_per_group)
    z_crit = norm_ppf(1 - alpha / 2) if alternative == "two-sided" else norm_ppf(1 - alpha)
    z = abs(mde_abs) / se - z_crit
    power = norm_cdf(z)
    if alternative == "two-sided":
        power += norm_cdf(-abs(mde_abs) / se - z_crit)
    return float(min(max(power, 0.0), 1.0))


# =============================================================================
# 第 5 部分：偷看模拟（peeking / optional stopping）
# =============================================================================
# 演示第 6.5.1 节的核心结论：即便 H0 为真（A/A 测试，两组同分布），
# 只要你"反复看 p 值、一旦 <α 就停"，实际假阳性率会远超名义 α。


@dataclass
class PeekingResult:
    """偷看模拟的结果汇总。"""
    n_peeks: int              # 每次实验偷看几次
    fpr_peeking: float        # 偷看策略下的实际假阳性率
    fpr_fixed: float          # 固定样本量（只在终点看一次）的假阳性率
    alpha: float              # 名义显著性水平


def simulate_peeking(
    n_max: int = 2000, n_peeks: int = 10, alpha: float = 0.05,
    n_trials: int = 2000, p: float = 0.10, seed: int = 0,
) -> PeekingResult:
    """蒙特卡洛模拟"偷看"如何抬高假阳性率（two-proportion，A/A 测试）。

    参数：
        n_max：每组最终样本量上限
        n_peeks：把 [1, n_max] 均匀切成 n_peeks 个检查点（偷看时刻）
        alpha：判显著的阈值
        n_trials：模拟多少次独立实验
        p：两组共同的真实转化率（A/A：两组一样，故 H0 为真）
        seed：随机种子

    逻辑：
        对每次实验，逐步累加两组的伯努利样本；在每个检查点算一次 z 检验。
        - 偷看策略：任一检查点 p<α 即"宣布显著"（提前停止）。
        - 固定策略：只在最后一个检查点看一次。
        因为 H0 为真，任何"显著"都是假阳性。统计两种策略的假阳性比例。
    """
    rng = np.random.default_rng(seed)
    # 检查点（每组累计到的样本数），至少 30 起步避免小样本近似崩坏
    checkpoints = np.linspace(n_max / n_peeks, n_max, n_peeks).astype(int)
    checkpoints = np.maximum(checkpoints, 30)

    z_crit = norm_ppf(1 - alpha / 2)  # 双尾临界值

    false_pos_peek = 0
    false_pos_fixed = 0

    for _ in range(n_trials):
        # 一次性生成两组 0/1 序列（伯努利），A/A 同率 p
        a = rng.random(n_max) < p
        b = rng.random(n_max) < p
        ca = np.cumsum(a)  # A 组累计转化数
        cb = np.cumsum(b)  # B 组累计转化数

        peeked_significant = False
        for n in checkpoints:
            xa, xb = ca[n - 1], cb[n - 1]
            pa, pb = xa / n, xb / n
            p_pool = (xa + xb) / (2 * n)
            se = math.sqrt(p_pool * (1 - p_pool) * (2.0 / n)) if 0 < p_pool < 1 else 0.0
            if se == 0:
                continue
            z = (pb - pa) / se
            if abs(z) > z_crit:            # 该检查点显著
                peeked_significant = True
                break                       # 偷看策略：一旦显著立刻停止

        if peeked_significant:
            false_pos_peek += 1

        # 固定策略：只看终点这一次
        n = checkpoints[-1]
        xa, xb = ca[n - 1], cb[n - 1]
        p_pool = (xa + xb) / (2 * n)
        se = math.sqrt(p_pool * (1 - p_pool) * (2.0 / n)) if 0 < p_pool < 1 else 0.0
        if se > 0 and abs((xb / n - xa / n) / se) > z_crit:
            false_pos_fixed += 1

    return PeekingResult(
        n_peeks=n_peeks,
        fpr_peeking=false_pos_peek / n_trials,
        fpr_fixed=false_pos_fixed / n_trials,
        alpha=alpha,
    )


def peeking_fpr_curve(
    peek_counts=(1, 2, 3, 5, 10, 20),
    n_max: int = 2000, alpha: float = 0.05,
    n_trials: int = 1500, p: float = 0.10, seed: int = 0,
):
    """对不同"偷看次数"分别跑模拟，返回 (peek_counts, fpr_list)。

    用于画"假阳性率随偷看次数膨胀"曲线。
    """
    fprs = []
    for k in peek_counts:
        res = simulate_peeking(n_max=n_max, n_peeks=k, alpha=alpha,
                               n_trials=n_trials, p=p, seed=seed)
        fprs.append(res.fpr_peeking)
    return list(peek_counts), fprs


# =============================================================================
# 第 6 部分：CI 覆盖率的蒙特卡洛验证（供测试与演示）
# =============================================================================


def ci_coverage_two_proportions(
    p_a: float, p_b: float, n: int,
    alpha: float = 0.05, n_trials: int = 3000, seed: int = 0,
) -> float:
    """蒙特卡洛验证：two-proportion 的 (1-alpha) CI 是否真的覆盖真实效应 (p_b-p_a)。

    做 n_trials 次实验，每次算一个 CI，统计"CI 包含真实 p_b-p_a"的比例。
    正确实现的 CI，覆盖率应≈ 1-alpha（如 0.95）。
    """
    rng = np.random.default_rng(seed)
    true_effect = p_b - p_a
    covered = 0
    for _ in range(n_trials):
        xa = int(rng.binomial(n, p_a))
        xb = int(rng.binomial(n, p_b))
        # 避免退化（全 0 或全 n 导致 SE=0）
        xa = min(max(xa, 1), n - 1)
        xb = min(max(xb, 1), n - 1)
        res = two_proportion_ztest(xa, n, xb, n, alpha=alpha, alternative="two-sided")
        if res.ci_low <= true_effect <= res.ci_high:
            covered += 1
    return covered / n_trials
