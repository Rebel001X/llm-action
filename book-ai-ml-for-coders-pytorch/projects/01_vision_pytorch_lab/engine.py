# -*- coding: utf-8 -*-
"""
engine.py —— 通用训练/评估引擎（对应第 2、3 章的训练循环）

- train():    标准 PyTorch 训练循环（前向 -> 求损失 -> 反向 -> 更新），
              返回每个 epoch 的平均损失历史；内置简单的 early stopping。
- evaluate(): 在给定 loader 上计算分类准确率。

一切都在 CPU 上、用小网络小数据跑，几秒内完成。
"""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """返回分类准确率（0~1）。"""
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def _avg_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """在一个 loader 上计算平均损失（供 early stopping 监控用）。"""
    model.eval()
    running = 0.0
    total = 0
    for xb, yb in loader:
        loss = criterion(model(xb), yb)
        running += loss.item() * xb.size(0)
        total += xb.size(0)
    return running / max(total, 1)


def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    val_loader: Optional[DataLoader] = None,
    patience: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> List[float]:
    """训练模型，返回逐 epoch 的平均训练损失列表 history。

    参数：
        model:      待训练的网络（nn.Module）。
        loader:     训练 DataLoader。
        epochs:     最大训练轮数。
        lr:         Adam 学习率。
        val_loader: 若提供，则用验证损失做 early stopping；否则用训练损失。
        patience:   连续多少个 epoch 没有改善就提前停止。
        seed:       固定优化随机性，保证可复现。
        verbose:    是否打印每个 epoch 的进度。

    early stopping：
        监控 loss（有验证集用验证 loss，否则用训练 loss），
        记录最优权重；连续 patience 轮没有变得更好就停止并恢复最优权重。
    """
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: List[float] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for ep in range(epochs):
        model.train()
        running = 0.0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)                 # 前向
            loss = criterion(logits, yb)       # 交叉熵损失
            loss.backward()                    # 反向传播
            optimizer.step()                   # 参数更新
            running += loss.item() * xb.size(0)
            total += xb.size(0)

        epoch_loss = running / max(total, 1)
        history.append(epoch_loss)

        # early stopping 监控值：优先用验证损失
        monitor = _avg_loss(model, val_loader, criterion) if val_loader is not None else epoch_loss

        if verbose:
            tag = "val_loss" if val_loader is not None else "train_loss"
            print(f"  epoch {ep + 1}/{epochs}  train_loss={epoch_loss:.4f}  {tag}={monitor:.4f}")

        # 有改善（更小）就记录最优权重并清零耐心计数
        if monitor < best_loss - 1e-4:
            best_loss = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stopping at epoch {ep + 1}")
                break

    # 恢复到监控指标最优的那组权重
    model.load_state_dict(best_state)
    return history
