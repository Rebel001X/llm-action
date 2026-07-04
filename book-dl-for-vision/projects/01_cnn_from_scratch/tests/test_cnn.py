# -*- coding: utf-8 -*-
"""
test_cnn.py —— 对 cnn_numpy 的核心算子做单元测试。

测试思路（三条腿）：
    1) 手算对拍：用一个小到能用笔算的输入，验证 conv/pool/relu 的具体数值；
    2) torch 对拍：把 numpy 结果与 torch.nn.functional 的官方实现比，误差 < 1e-10；
    3) 公式对拍：验证输出尺寸公式、感受野公式在各种 stride/padding 组合下都对。

运行：  python -m pytest -q
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# 让 tests/ 能找到上一级目录里的 cnn_numpy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_numpy import (  # noqa: E402
    conv2d,
    max_pool2d,
    relu,
    flatten,
    linear,
    conv_output_size,
    receptive_field,
    TinyCNN,
)


# =============================================================================
# 1. 输出尺寸公式
# =============================================================================
@pytest.mark.parametrize(
    "in_size,kernel,stride,padding,expected",
    [
        (5, 3, 1, 0, 3),    # 经典无填充：(5-3)/1+1 = 3
        (5, 3, 1, 1, 5),    # same 卷积：padding=1 保持尺寸
        (7, 3, 2, 0, 3),    # stride=2：(7-3)/2+1 = 3
        (28, 5, 1, 2, 28),  # LeNet 常见：5x5 核 + pad2 保持 28
        (32, 2, 2, 0, 16),  # 2x2 stride2 池化：尺寸减半
        (6, 6, 1, 0, 1),    # 核等于输入：输出 1x1
    ],
)
def test_conv_output_size(in_size, kernel, stride, padding, expected):
    assert conv_output_size(in_size, kernel, stride, padding) == expected


def test_conv_output_size_matches_torch():
    """随机若干组参数，与 torch 真实卷积的输出 H/W 对比。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        H = int(rng.integers(5, 33))
        k = int(rng.integers(1, 6))
        s = int(rng.integers(1, 4))
        p = int(rng.integers(0, 3))
        if H + 2 * p - k < 0:
            continue
        x = torch.randn(1, 1, H, H)
        w = torch.randn(1, 1, k, k)
        y = F.conv2d(x, w, stride=s, padding=p)
        assert y.shape[-1] == conv_output_size(H, k, s, p)


def test_conv_output_size_raises_when_kernel_too_big():
    with pytest.raises(ValueError):
        conv_output_size(3, 5, 1, 0)  # 5x5 核 > 3x3 输入，无填充 → 报错


# =============================================================================
# 2. conv2d 数值对拍 torch.nn.functional.conv2d
# =============================================================================
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("C_in,C_out", [(1, 1), (3, 4), (2, 5)])
def test_conv2d_matches_torch(stride, padding, C_in, C_out):
    rng = np.random.default_rng(42)
    N, H, W, k = 2, 9, 9, 3
    x = rng.standard_normal((N, C_in, H, W))
    w = rng.standard_normal((C_out, C_in, k, k))
    b = rng.standard_normal(C_out)

    mine = conv2d(x, w, b, stride=stride, padding=padding)
    ref = F.conv2d(
        torch.from_numpy(x),
        torch.from_numpy(w),
        torch.from_numpy(b),
        stride=stride,
        padding=padding,
    ).numpy()

    assert mine.shape == ref.shape
    np.testing.assert_allclose(mine, ref, atol=1e-10, rtol=1e-10)


def test_conv2d_no_bias_matches_torch():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 2, 6, 6))
    w = rng.standard_normal((3, 2, 3, 3))
    mine = conv2d(x, w, bias=None, stride=1, padding=0)
    ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(mine, ref, atol=1e-10)


def test_conv2d_hand_computed():
    """一个能用笔算的例子：3x3 输入 + 3x3 全 1 核 = 窗口内元素之和。

    输入:            核(全1):
      1 2 3           1 1 1
      4 5 6           1 1 1
      7 8 9           1 1 1

    无 padding、stride 1 → 输出 1x1，值 = 1+2+...+9 = 45。
    """
    x = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
    w = np.ones((1, 1, 3, 3))
    out = conv2d(x, w, stride=1, padding=0)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == pytest.approx(45.0)


def test_conv2d_hand_computed_with_stride_padding():
    """2x2 平均核（值 0.25）、pad1、stride2，对 3x3 输入，手算一个角落。

    验证左上角输出 = 只有一个真实像素 x[0,0]=1 落在窗口，其余是 padding 的 0，
    结果应为 0.25 * (0+0+0+1) = 0.25。
    """
    x = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
    w = np.full((1, 1, 2, 2), 0.25)
    out = conv2d(x, w, stride=2, padding=1)
    # 与 torch 完整对拍
    ref = F.conv2d(
        torch.from_numpy(x), torch.from_numpy(w), stride=2, padding=1
    ).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-12)
    assert out[0, 0, 0, 0] == pytest.approx(0.25)


# =============================================================================
# 3. max_pool2d 对拍 + relu 正确
# =============================================================================
@pytest.mark.parametrize("kernel,stride", [(2, 2), (2, 1), (3, 2)])
def test_max_pool2d_matches_torch(kernel, stride):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 8, 8))
    mine = max_pool2d(x, kernel=kernel, stride=stride)
    ref = F.max_pool2d(torch.from_numpy(x), kernel_size=kernel, stride=stride).numpy()
    assert mine.shape == ref.shape
    np.testing.assert_allclose(mine, ref, atol=1e-12)


def test_max_pool2d_hand_computed():
    """2x2 池化在 4x4 上，检查每个窗口的最大值。"""
    x = np.array(
        [[1, 3, 2, 4],
         [5, 6, 7, 8],
         [9, 2, 1, 0],
         [3, 4, 5, 6]],
        dtype=np.float64,
    ).reshape(1, 1, 4, 4)
    out = max_pool2d(x, kernel=2, stride=2)
    # 四个 2x2 窗口的最大值：左上6 右上8 左下9 右下6
    expected = np.array([[6, 8], [9, 6]], dtype=np.float64).reshape(1, 1, 2, 2)
    np.testing.assert_allclose(out, expected)


def test_max_pool2d_negative_padding_uses_neg_inf():
    """全负输入 + padding：验证不会错误地选到填充 0（-inf 填充的正确性）。"""
    x = -np.ones((1, 1, 2, 2), dtype=np.float64) * 5.0  # 全 -5
    mine = max_pool2d(x, kernel=2, stride=2, padding=1)  # pad 后 4x4
    ref = F.max_pool2d(torch.from_numpy(x), kernel_size=2, stride=2, padding=1).numpy()
    np.testing.assert_allclose(mine, ref, atol=1e-12)
    # 至少中心那些包含真实像素的窗口应为 -5，而不是 0
    assert mine.max() == pytest.approx(-5.0)


def test_relu():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 3.0])
    out = relu(x)
    np.testing.assert_allclose(out, [0, 0, 0, 0.5, 3.0])
    # 对拍 torch
    ref = F.relu(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(out, ref)


def test_relu_shape_preserved():
    x = np.random.default_rng(0).standard_normal((2, 3, 4, 4))
    assert relu(x).shape == x.shape


# =============================================================================
# 4. flatten / linear
# =============================================================================
def test_flatten_shape_and_order():
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2)
    f = flatten(x)
    assert f.shape == (2, 12)
    # 与 torch.flatten(x, 1) 展平顺序一致
    ref = torch.flatten(torch.from_numpy(x), 1).numpy()
    np.testing.assert_allclose(f, ref)


def test_linear_matches_torch():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 6))
    w = rng.standard_normal((5, 6))
    b = rng.standard_normal(5)
    mine = linear(x, w, b)
    ref = F.linear(
        torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b)
    ).numpy()
    assert mine.shape == (4, 5)
    np.testing.assert_allclose(mine, ref, atol=1e-12)


# =============================================================================
# 5. 感受野公式
# =============================================================================
def test_receptive_field_single_layer():
    # 单个 3x3 卷积，感受野就是 3
    assert receptive_field([3], [1]) == 3


def test_receptive_field_two_3x3_equals_5x5():
    """两个 3x3 stride1 堆叠 → 感受野 5，等价一个 5x5。VGG 的经典论点。"""
    assert receptive_field([3, 3], [1, 1]) == 5


def test_receptive_field_three_3x3_equals_7x7():
    assert receptive_field([3, 3, 3], [1, 1, 1]) == 7


def test_receptive_field_with_stride():
    """conv(3,s1) → pool(2,s2) → conv(3,s1)。
    逐层：RF=1,jump=1
      conv3 s1: RF=1+2*1=3,  jump=1
      pool2 s2: RF=3+1*1=4,  jump=2
      conv3 s1: RF=4+2*2=8,  jump=2
    """
    assert receptive_field([3, 2, 3], [1, 2, 1]) == 8


def test_receptive_field_length_mismatch_raises():
    with pytest.raises(ValueError):
        receptive_field([3, 3], [1])


# =============================================================================
# 6. 端到端：TinyCNN 前向 shape 正确 & 可复现
# =============================================================================
def test_tinycnn_forward_shape():
    net = TinyCNN(in_channels=1, num_filters=4, kernel=3, num_classes=3, seed=0)
    x = np.random.default_rng(0).standard_normal((2, 1, 16, 16))
    logits = net.forward(x)
    assert logits.shape == (2, 3)
    # 中间缓存都在
    assert set(net.cache) == {"conv", "relu", "pool", "logits"}
    # same 卷积后空间尺寸不变 16x16，池化后 8x8
    assert net.cache["conv"].shape == (2, 4, 16, 16)
    assert net.cache["pool"].shape == (2, 4, 8, 8)


def test_tinycnn_reproducible():
    x = np.random.default_rng(5).standard_normal((1, 1, 12, 12))
    a = TinyCNN(seed=123).forward(x)
    b = TinyCNN(seed=123).forward(x)
    np.testing.assert_allclose(a, b)


def test_full_pipeline_matches_torch():
    """把 conv→relu→pool→flatten→linear 整条链路与 torch 逐层对拍。"""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((2, 3, 10, 10))
    cw = rng.standard_normal((6, 3, 3, 3))
    cb = rng.standard_normal(6)

    # numpy 路线
    z = conv2d(x, cw, cb, stride=1, padding=1)
    a = relu(z)
    p = max_pool2d(a, kernel=2, stride=2)
    f = flatten(p)
    fw = rng.standard_normal((4, f.shape[1]))
    fb = rng.standard_normal(4)
    y = linear(f, fw, fb)

    # torch 路线（同样的权重）
    tz = F.conv2d(torch.from_numpy(x), torch.from_numpy(cw), torch.from_numpy(cb),
                  stride=1, padding=1)
    ta = F.relu(tz)
    tp = F.max_pool2d(ta, 2, 2)
    tf = torch.flatten(tp, 1)
    ty = F.linear(tf, torch.from_numpy(fw), torch.from_numpy(fb))

    np.testing.assert_allclose(y, ty.numpy(), atol=1e-9)
