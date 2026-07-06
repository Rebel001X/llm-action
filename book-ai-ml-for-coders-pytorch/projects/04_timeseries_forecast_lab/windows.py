"""windows.py —— 窗口化（对应第 9 章「Windowed Dataset」的核心手法）。

时间序列监督学习的第一步：把一整条序列切成一堆
    (过去 window 个值)  ->  (未来 horizon 个值)
的样本对。这一步是第 10、11 章所有模型的共同输入格式。

例如 window=3, horizon=1, 序列 [a,b,c,d,e]:
    X=[a,b,c] -> y=[d]
    X=[b,c,d] -> y=[e]
"""

from __future__ import annotations

import numpy as np
import torch


def make_windows(
    series, window: int, horizon: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 1D 序列切成 (X, y) 监督样本，返回 torch.float32 张量。

    参数:
        series  : 1D 序列（numpy 数组 / list / torch 张量都行）。
        window  : 输入窗口长度（用过去多少个点预测）。
        horizon : 预测步数（未来多少个点），默认 1。

    返回:
        X : shape (N, window)   —— 每行是一段历史窗口。
        y : shape (N, horizon)  —— 每行是紧接其后的未来目标。
        其中 N = len(series) - window - horizon + 1。
    """
    if isinstance(series, torch.Tensor):
        arr = series.detach().cpu().numpy()
    else:
        arr = np.asarray(series)
    arr = arr.astype(np.float32).reshape(-1)  # 拉平成 1D

    if window < 1 or horizon < 1:
        raise ValueError("window 和 horizon 必须 >= 1")
    n = len(arr) - window - horizon + 1
    if n <= 0:
        raise ValueError(
            f"序列太短: len={len(arr)}, 需要 > window+horizon-1={window + horizon - 1}"
        )

    xs = np.empty((n, window), dtype=np.float32)
    ys = np.empty((n, horizon), dtype=np.float32)
    for i in range(n):
        xs[i] = arr[i : i + window]
        ys[i] = arr[i + window : i + window + horizon]

    return torch.from_numpy(xs), torch.from_numpy(ys)
