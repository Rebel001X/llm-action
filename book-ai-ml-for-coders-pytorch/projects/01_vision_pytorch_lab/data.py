# -*- coding: utf-8 -*-
"""
data.py —— 确定性合成的 "FashionMNIST 形状" 数据集（对应第 2 章）

设计目标：
1) 数据形状与真实 FashionMNIST 完全一致：图像 (N, 1, 28, 28)，标签 0..9 共 10 类。
2) 完全离线、可复现：所有随机性都由固定种子控制（np.random.default_rng(seed)）。
3) 类别在像素上"可分"：给每一类一个独特的空间图案（条纹 / 方块 / 边框 / 十字…），
   再叠加噪声与小幅平移抖动，使得小模型（DNN/CNN）能学到 >70% 的准确率，
   同时又不是"一眼就能背下来"的平凡任务。

另外提供一个 *可选* 的真实数据加载函数 load_real_fashionmnist()，
它用 try import torchvision 门控：装了 torchvision 就能加载真实 FashionMNIST，
没装（本环境即如此）就抛出友好的提示，绝不影响离线合成分支与 pytest。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 10 个类别的中文名（与 FashionMNIST 顺序对应，仅用于打印/可视化）
CLASS_NAMES = [
    "T恤", "裤子", "套衫", "连衣裙", "外套",
    "凉鞋", "衬衫", "运动鞋", "包", "短靴",
]


def _prototypes() -> np.ndarray:
    """构造 10 个类别的"原型图案"，形状 (10, 28, 28)，取值 0/1。

    每一类都是一个视觉上明显不同的空间模式——这是"类别可分"的核心来源。
    CNN 靠局部卷积核就能捕捉这些边缘/方块/条纹特征，因此通常比 DNN 更强。
    """
    p = np.zeros((10, 28, 28), dtype=np.float32)

    # 0: 竖条纹（每隔 4 列亮一条）
    p[0][:, ::4] = 1.0
    # 1: 横条纹（每隔 4 行亮一条）
    p[1][::4, :] = 1.0
    # 2: 主对角线（两像素宽，便于卷积核捕捉）
    for i in range(28):
        p[2][i, i] = 1.0
        if i + 1 < 28:
            p[2][i, i + 1] = 1.0
    # 3: 左上方块
    p[3][4:12, 4:12] = 1.0
    # 4: 右上方块
    p[4][4:12, 16:24] = 1.0
    # 5: 左下方块
    p[5][16:24, 4:12] = 1.0
    # 6: 右下方块
    p[6][16:24, 16:24] = 1.0
    # 7: 居中实心块
    p[7][10:18, 10:18] = 1.0
    # 8: 空心边框
    p[8][2:26, 2:26] = 1.0
    p[8][6:22, 6:22] = 0.0
    # 9: 十字（加号）
    p[9][13:15, :] = 1.0
    p[9][:, 13:15] = 1.0
    return p


def synthetic_fashion(
    n_per_class: int = 200,
    seed: int = 0,
    noise: float = 0.35,
    jitter: int = 2,
):
    """生成合成数据集。

    参数：
        n_per_class: 每个类别的样本数（总数 = n_per_class * 10）。
        seed:        随机种子，保证完全可复现。
        noise:       叠加的高斯噪声标准差（越大越难）。
        jitter:      每张图随机整数平移的最大幅度（模拟位置变化）。

    返回：
        X: torch.FloatTensor，形状 (N, 1, 28, 28)
        y: torch.LongTensor， 形状 (N,)，取值 0..9
    """
    rng = np.random.default_rng(seed)
    protos = _prototypes()
    n = n_per_class * 10

    X = np.zeros((n, 1, 28, 28), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    k = 0
    for c in range(10):
        for _ in range(n_per_class):
            img = protos[c].copy()
            # 随机平移（roll）模拟物体位置抖动，让模型学"平移不变"的特征
            dy = int(rng.integers(-jitter, jitter + 1))
            dx = int(rng.integers(-jitter, jitter + 1))
            img = np.roll(img, dy, axis=0)
            img = np.roll(img, dx, axis=1)
            # 叠加高斯噪声
            img = img + rng.normal(0.0, noise, size=(28, 28)).astype(np.float32)
            X[k, 0] = img
            y[k] = c
            k += 1

    # 打乱顺序，避免同类样本聚在一起
    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]
    return torch.from_numpy(X), torch.from_numpy(y)


def make_loaders(
    n_per_class_train: int = 200,
    n_per_class_test: int = 60,
    batch_size: int = 64,
    seed: int = 0,
):
    """一站式构造训练/测试 DataLoader（做了标准化）。

    标准化统计量只在训练集上计算，再应用到测试集——这是防止"数据泄漏"的规范做法。
    返回：train_loader, test_loader
    """
    # 训练集与测试集使用不同的种子偏移，保证两者不重叠但都可复现
    Xtr, ytr = synthetic_fashion(n_per_class_train, seed=seed)
    Xte, yte = synthetic_fashion(n_per_class_test, seed=seed + 999)

    # 用训练集统计量做标准化（zero-mean / unit-std）
    mean = Xtr.mean()
    std = Xtr.std().clamp_min(1e-6)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    g = torch.Generator().manual_seed(seed)  # DataLoader 洗牌也固定种子
    train_loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, generator=g
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_real_fashionmnist(root: str = "./_data", train: bool = True):
    """（可选）加载真实的 FashionMNIST —— 用 torchvision 门控。

    本环境没有安装 torchvision，也没联网，所以这个函数默认走不通；
    它的存在是为了告诉你"如何一行切换到真实数据"。

    在装了 torchvision 且能联网的机器上，你可以这样用：
        from torchvision import datasets, transforms
        ds = load_real_fashionmnist(train=True)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

    返回的 Dataset 每个样本是 (image[1,28,28], label)，与合成数据完全同构，
    因此 models.py / engine.py 的代码无需任何改动即可复用。
    """
    try:
        from torchvision import datasets, transforms  # 门控：缺失就走 except
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "未安装 torchvision，无法加载真实 FashionMNIST。\n"
            "请先 `pip install torchvision` 并联网；\n"
            "或直接使用离线合成数据 synthetic_fashion()/make_loaders()。"
        ) from e

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),  # -> [1,28,28]，取值 [0,1]
            transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST 官方均值/方差
        ]
    )
    return datasets.FashionMNIST(root=root, train=train, download=True, transform=tfm)
