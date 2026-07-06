# -*- coding: utf-8 -*-
"""
test_vision.py —— 视觉实验室的验收测试（全部离线、CPU、秒级）。

覆盖规格要求：
(a) DNN / CNN 前向输出形状为 (batch, 10)；
(b) 合成数据上训练几个 epoch 后 loss 明显下降；
(c) 训练后测试准确率显著高于随机（>0.3，随机基线仅 0.1）。
另外附带数据形状、可复现性、torchvision 门控等健壮性测试。
"""

from __future__ import annotations

import numpy as np
import torch

from data import CLASS_NAMES, make_loaders, synthetic_fashion
from engine import evaluate, train
from models import build_cnn, build_dnn


# ---------- (a) 前向输出形状 ----------

def test_dnn_forward_shape():
    model = build_dnn()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)


def test_cnn_forward_shape():
    model = build_cnn()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)


# ---------- 数据健壮性 ----------

def test_synthetic_data_shape_and_labels():
    X, y = synthetic_fashion(n_per_class=5, seed=0)
    assert X.shape == (50, 1, 28, 28)
    assert X.dtype == torch.float32
    assert y.shape == (50,)
    # 标签覆盖 0..9 全部 10 类
    assert set(y.tolist()) == set(range(10))
    assert len(CLASS_NAMES) == 10


def test_synthetic_data_reproducible():
    X1, y1 = synthetic_fashion(n_per_class=5, seed=0)
    X2, y2 = synthetic_fashion(n_per_class=5, seed=0)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


# ---------- (b) 训练后 loss 明显下降 ----------

def _small_loaders():
    return make_loaders(
        n_per_class_train=40, n_per_class_test=20, batch_size=64, seed=0
    )


def test_dnn_loss_decreases():
    torch.manual_seed(0)
    train_loader, _ = _small_loaders()
    model = build_dnn()
    hist = train(model, train_loader, epochs=5, lr=1e-3, patience=5)
    assert len(hist) >= 2
    # 末轮损失明显低于首轮（至少下降 20%）
    assert hist[-1] < hist[0] * 0.8


def test_cnn_loss_decreases():
    torch.manual_seed(0)
    train_loader, _ = _small_loaders()
    model = build_cnn()
    hist = train(model, train_loader, epochs=5, lr=1e-3, patience=5)
    assert len(hist) >= 2
    assert hist[-1] < hist[0] * 0.8


# ---------- (c) 训练后准确率高于随机 ----------

def test_dnn_accuracy_above_random():
    torch.manual_seed(0)
    train_loader, test_loader = _small_loaders()
    model = build_dnn()
    train(model, train_loader, epochs=6, lr=1e-3, val_loader=test_loader, patience=5)
    acc = evaluate(model, test_loader)
    assert acc > 0.3  # 随机基线 0.1


def test_cnn_accuracy_above_random():
    torch.manual_seed(0)
    train_loader, test_loader = _small_loaders()
    model = build_cnn()
    train(model, train_loader, epochs=6, lr=1e-3, val_loader=test_loader, patience=5)
    acc = evaluate(model, test_loader)
    assert acc > 0.3


# ---------- early stopping 行为 ----------

def test_early_stopping_stops_early():
    torch.manual_seed(0)
    train_loader, test_loader = _small_loaders()
    model = build_cnn()
    # patience=1 且给一个很大的 epochs：几乎必然提前停止，history 长度 < epochs
    hist = train(
        model, train_loader, epochs=50, lr=1e-3, val_loader=test_loader, patience=1
    )
    assert len(hist) < 50


# ---------- torchvision 门控（未安装时应给出友好报错，且不崩溃） ----------

def test_real_loader_gated():
    import importlib.util

    from data import load_real_fashionmnist

    if importlib.util.find_spec("torchvision") is None:
        # 本环境没有 torchvision：应抛 RuntimeError 而非 ImportError 崩溃
        import pytest

        with pytest.raises(RuntimeError):
            load_real_fashionmnist()
    else:
        # 若恰好装了 torchvision，也不真的联网下载，仅确认函数可调用即可
        assert callable(load_real_fashionmnist)
