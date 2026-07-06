# -*- coding: utf-8 -*-
"""
models.py —— 两种图像分类网络（对应第 2、3 章）

- build_dnn(): 第 2 章的全连接网络（DNN / MLP）
    Flatten -> Linear -> ReLU -> Linear(10)
  把 28x28 的图片"拉直"成 784 维向量再分类。简单但丢掉了空间结构。

- build_cnn(): 第 3 章的卷积网络（CNN）
    [Conv2d -> ReLU -> MaxPool2d] x2 -> Flatten -> Linear(10)
  卷积核在图像上滑动，直接从像素里"检测特征"（边缘、方块、条纹…），
  因此在图像任务上通常比同规模的 DNN 更准、更省参数。
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def build_dnn(in_shape=(1, 28, 28), hidden: int = 128, n_classes: int = 10) -> nn.Module:
    """第 2 章：全连接 DNN。

    Flatten -> Linear(784, hidden) -> ReLU -> Linear(hidden, 10)
    """
    in_features = int(np.prod(in_shape))  # 1*28*28 = 784
    return nn.Sequential(
        nn.Flatten(),                       # (B,1,28,28) -> (B,784)
        nn.Linear(in_features, hidden),     # 全连接隐藏层
        nn.ReLU(),                          # 非线性激活
        nn.Linear(hidden, n_classes),       # 输出 10 类 logits
    )


def build_cnn(n_classes: int = 10) -> nn.Module:
    """第 3 章：卷积 CNN。

    Conv(1->16) -> ReLU -> MaxPool(28->14)
    Conv(16->32) -> ReLU -> MaxPool(14->7)
    Flatten -> Linear(32*7*7, 10)
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,1,28,28)->(B,16,28,28)
        nn.ReLU(),
        nn.MaxPool2d(2),                              # ->(B,16,14,14)
        nn.Conv2d(16, 32, kernel_size=3, padding=1),  # ->(B,32,14,14)
        nn.ReLU(),
        nn.MaxPool2d(2),                              # ->(B,32,7,7)
        nn.Flatten(),                                 # ->(B,32*7*7=1568)
        nn.Linear(32 * 7 * 7, n_classes),             # ->(B,10)
    )
