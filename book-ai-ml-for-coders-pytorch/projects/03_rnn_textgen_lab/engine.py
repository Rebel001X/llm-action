# -*- coding: utf-8 -*-
"""训练循环（对应原书第 8 章的模型训练）。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(model: nn.Module, dataset, epochs: int = 30, batch_size: int = 32, lr: float = 0.01, seed: int = 0):
    """在 (X, y) 数据集上训练下一个词预测。返回每个 epoch 的平均损失列表。"""
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # 内含 log-softmax，logits 直接进来即可

    history = []
    model.train()
    for _ in range(epochs):
        total, n = 0.0, 0
        for X, y in loader:
            opt.zero_grad()
            logits = model(X)            # (batch, vocab)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        history.append(total / max(n, 1))
    return history
