"""metrics.py —— 预测误差指标（对应第 9 章的评估思路）。

时间序列回归最常用的两个指标:
    MAE (Mean Absolute Error)  : 平均绝对误差，和目标同量纲，直观。
    MSE (Mean Squared Error)   : 平均平方误差，对大误差惩罚更重，是训练用的损失。

两个函数都同时接受 numpy 数组和 torch 张量，返回 python float，方便打印/断言。
"""

from __future__ import annotations

import numpy as np
import torch


def _to_flat_tensor(x) -> torch.Tensor:
    """把任意输入（tensor / ndarray / list）统一成 1D float32 张量。"""
    if isinstance(x, torch.Tensor):
        t = x.detach().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return t.reshape(-1)


def mae(pred, target) -> float:
    """平均绝对误差 mean(|pred - target|)。"""
    p = _to_flat_tensor(pred)
    t = _to_flat_tensor(target)
    return float(torch.mean(torch.abs(p - t)))


def mse(pred, target) -> float:
    """平均平方误差 mean((pred - target)^2)。"""
    p = _to_flat_tensor(pred)
    t = _to_flat_tensor(target)
    return float(torch.mean((p - t) ** 2))
