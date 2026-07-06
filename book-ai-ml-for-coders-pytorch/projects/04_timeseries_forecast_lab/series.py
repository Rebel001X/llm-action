"""series.py —— 合成时间序列（对应第 9 章「理解序列与时间序列数据」）。

本模块负责“造数据”：把一条真实业务里常见的时间序列拆成三个可解释的成分
    序列 = 趋势(trend) + 季节性(seasonality) + 噪声(noise)
并用固定随机种子生成，保证每次运行完全一致（离线、可复现）。

- trend        : 缓慢单调的长期漂移（比如用户量随时间增长）。
- seasonality  : 周期性波动（比如每周/每年的规律，用正弦叠加模拟）。
- noise        : 不可预测的随机扰动（决定了“理论最优”的误差下限）。

第 10、11 章的所有模型，本质上都是在“看一段历史窗口、预测下一个值”，
而它们能赢过朴素基线的原因，就是能学出上面 trend+seasonality 这两个
**确定性**成分，把误差压到接近 noise 的水平。
"""

from __future__ import annotations

import numpy as np


def _trend(time: np.ndarray, slope: float) -> np.ndarray:
    """线性趋势：slope 决定长期漂移的斜率。"""
    return slope * time


def _seasonality(time: np.ndarray, period: float, amplitude: float) -> np.ndarray:
    """季节性：用一个基波 + 一个二次谐波叠加，形状比纯正弦更“像真实数据”。

    period    : 主周期（多少个时间步重复一次）。
    amplitude : 主波振幅。
    """
    phase = 2.0 * np.pi * time / period
    season = np.sin(phase) + 0.5 * np.sin(2.0 * phase)  # 基波 + 二次谐波
    return amplitude * season


def make_series(
    seed: int = 0,
    n: int = 1000,
    slope: float = 0.001,
    period: float = 50.0,
    amplitude: float = 0.5,
    noise_std: float = 0.02,
    baseline: float = 0.5,
) -> np.ndarray:
    """生成一条 1D 的 trend+seasonality+noise 序列。

    参数默认值经过挑选，使序列“可学”：季节性明显、噪声很小，
    这样第 10 章的 DNN 能明显赢过“朴素预测（用上一个值）”。

    返回:
        shape (n,) 的 float64 numpy 数组。
    """
    rng = np.random.default_rng(seed)  # 固定种子 -> 完全可复现
    time = np.arange(n, dtype=np.float64)

    series = (
        baseline
        + _trend(time, slope)
        + _seasonality(time, period, amplitude)
        + rng.normal(0.0, noise_std, size=n)
    )
    return series.astype(np.float64)


def train_valid_split(
    series: np.ndarray, split: float = 0.8, context: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """把序列按时间顺序切成 训练段 / 验证段（时间序列不能随机打乱切分！）。

    参数:
        split   : 训练段占比（0~1）。
        context : 让验证段向前多带 `context` 个点，用于给验证窗口补足历史，
                  这样验证段第一个窗口也能有完整上下文（默认 0 = 不重叠）。

    返回:
        (train, valid) 两段 1D numpy 数组。
    """
    if not 0.0 < split < 1.0:
        raise ValueError(f"split 必须在 (0,1) 之间, 收到 {split}")
    k = int(len(series) * split)
    start = max(0, k - context)
    return series[:k].copy(), series[start:].copy()
