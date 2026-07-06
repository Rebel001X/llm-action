"""engine.py —— 通用训练循环（对应第 10 章的模型训练流程）。

把“造好的窗口样本 (X,y)” + “任意 nn.Module 模型”喂进来，
用 MSE 损失 + Adam 优化器做小批量训练，返回每个 epoch 的平均损失。
这样第 10、11 章的 DNN / Conv1D / LSTM 可以复用同一套训练代码。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def train_forecaster(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> list[float]:
    """训练一个预测模型（回归 + MSE 损失）。

    参数:
        model      : 任意输入 (N, window)、输出 (N, horizon) 的 nn.Module。
        X, y       : make_windows 产出的训练张量。
        epochs     : 训练轮数（小数据几十轮即可，秒级完成）。
        lr         : Adam 学习率。
        batch_size : 小批量大小。
        weight_decay: L2 正则（默认 0）。
        verbose    : 是否打印每轮损失。

    返回:
        losses : 长度为 epochs 的列表，每个 epoch 的平均训练 MSE。
                 正常情况下应单调下降（可用于测试“确实在学习”）。
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n = X.shape[0]
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        # 每个 epoch 打乱样本顺序（时间序列的“样本”是窗口，窗口之间可以打乱）
        perm = torch.randperm(n)
        running, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.shape[0]
            seen += xb.shape[0]

        epoch_loss = running / max(seen, 1)
        losses.append(epoch_loss)
        if verbose:
            print(f"epoch {epoch + 1:3d}/{epochs}  MSE={epoch_loss:.6f}")

    return losses


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """在评估模式下对 X 做一次前向推理，返回预测张量 (N, horizon)。"""
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    return model(X)
