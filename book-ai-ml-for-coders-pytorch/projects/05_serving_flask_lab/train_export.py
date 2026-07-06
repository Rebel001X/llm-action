# -*- coding: utf-8 -*-
"""
train_export.py —— 训练一个小分类器并导出权重

对应第 12 章「训练完就要想着怎么上线」：训练 -> 评估 -> 保存 state_dict。

运行：
    python train_export.py

会在 ./artifacts/model.pt 生成权重文件，供 app.py / handler.py 加载。
注意：pytest **不依赖**这个磁盘产物——测试内部自建模型（见 tests/）。
"""

import os

import torch
import torch.nn as nn

from inference import TinyClassifier, NUM_FEATURES, NUM_CLASSES, DEFAULT_MODEL_PATH


def make_synthetic_data(n_per_class=80, seed=0):
    """生成确定性的三类高斯簇合成数据（离线、可复现，不联网、不下载）。

    三个类别的中心在特征空间里分得很开，故很容易被小网络学会，
    保证测试里的「损失下降 / 高准确率」断言稳定成立。
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],
            [-2.0, 2.0, 0.0, 0.0],
            [0.0, -2.0, 2.0, 0.0],
        ]
    )
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        pts = centers[c] + 0.6 * torch.randn(n_per_class, NUM_FEATURES, generator=g)
        xs.append(pts)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    # 打乱顺序（同样用固定 generator，保证可复现）
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


def train(epochs=40, lr=0.05, seed=0):
    """全批量训练一个 TinyClassifier，返回 (model, losses)。"""
    torch.manual_seed(seed)
    X, y = make_synthetic_data(seed=seed)

    model = TinyClassifier()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    return model, losses


def accuracy(model, X, y):
    """在给定数据上算准确率（0~1）。"""
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
    return (preds == y).float().mean().item()


def export(model, path=DEFAULT_MODEL_PATH):
    """把 state_dict 保存到磁盘（真实上线时 TorchServe/Flask 会加载它）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def main():
    model, losses = train()
    X, y = make_synthetic_data()
    acc = accuracy(model, X, y)
    path = export(model)
    print(f"初始损失: {losses[0]:.4f}  ->  最终损失: {losses[-1]:.4f}")
    print(f"训练集准确率: {acc * 100:.1f}%")
    print(f"已导出权重到: {path}")


if __name__ == "__main__":
    main()
