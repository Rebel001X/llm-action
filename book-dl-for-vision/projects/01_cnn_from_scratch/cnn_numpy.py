# -*- coding: utf-8 -*-
"""
cnn_numpy.py —— 纯 NumPy 从零实现 CNN 前向传播的核心算子。

本模块只依赖 numpy，实现一个卷积神经网络「前向推理」需要的全部积木：
    - conv2d       二维卷积（支持 stride / padding，多输入通道、多输出通道）
    - max_pool2d   最大池化（支持 stride / padding）
    - relu         逐元素整流线性单元
    - flatten      把 (N, C, H, W) 展平成 (N, C*H*W)
    - linear       全连接层（矩阵乘 + 偏置）
    - 一些工具函数：conv_output_size / receptive_field

数据布局约定（与 PyTorch 一致，方便对拍）：
    - 输入 / 输出特征图：NCHW  = (batch, channels, height, width)
    - 卷积核权重：       OIHW  = (out_channels, in_channels, kH, kW)

设计目标：
    1) 数值上与 torch.nn.functional.conv2d / max_pool2d 完全对齐（float64 下误差 < 1e-10）；
    2) 代码短小、可读，用于教学而非追求极致性能；
    3) 只做「前向」，不含反向传播（反向留给下一个项目）。
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# 0. 工具函数：输出尺寸公式 与 感受野计算
# =============================================================================
def conv_output_size(in_size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """计算卷积 / 池化在某一个空间维度上的输出尺寸。

    经典公式（向下取整）：

        out = floor( (in + 2*padding - kernel) / stride ) + 1

    参数
    ----
    in_size : 输入这一维的长度（H 或 W）
    kernel  : 核在这一维的大小
    stride  : 步幅
    padding : 单侧填充数（两侧各 padding，所以是 +2*padding）

    返回
    ----
    int : 输出这一维的长度

    ⚠️ 坑：当 (in + 2*padding - kernel) 为负，说明核比（填充后的）输入还大，
        没有任何一个合法窗口，输出维度会 <= 0，此处直接抛错，避免下游诡异 shape。
    """
    numerator = in_size + 2 * padding - kernel
    if numerator < 0:
        raise ValueError(
            f"核比输入大：in={in_size}, kernel={kernel}, padding={padding} "
            f"→ 分子 {numerator} < 0，无合法卷积窗口。"
        )
    return numerator // stride + 1


def receptive_field(kernels, strides):
    """计算一串卷积/池化层「堆叠」之后，最后一层单个像素的『感受野』。

    感受野 = 输出特征图上一个点，能『看到』原始输入上多大的区域。

    递推公式（从后往前逐层累加）：

        RF_0 = 1
        RF_l = RF_{l-1} + (kernel_l - 1) * jump_{l-1}
        jump_l = jump_{l-1} * stride_l ,  jump_0 = 1

    其中 jump（有时叫 effective stride）是「相邻两个输出点，对应到输入上间隔多少像素」。

    参数
    ----
    kernels : 每层核大小的列表，如 [3, 3, 2]
    strides : 每层步幅的列表，长度需与 kernels 相同

    返回
    ----
    int : 顶层一个点在原图上的感受野边长

    💡 面试高频：为什么小卷积核堆叠（如两个 3x3）能替代一个大核（5x5）？
        两个 3x3、stride=1 → RF = 1 + 2 + 2 = 5，等价 5x5 感受野，
        但参数更少（2*3*3=18 < 5*5=25）、非线性更多。VGG 的核心洞见。
    """
    if len(kernels) != len(strides):
        raise ValueError("kernels 与 strides 长度必须相同")
    rf = 1        # 感受野，初始 1（输出点自己）
    jump = 1      # 有效步幅，初始 1
    for k, s in zip(kernels, strides):
        rf = rf + (k - 1) * jump
        jump = jump * s
    return rf


# =============================================================================
# 1. padding 辅助
# =============================================================================
def _pad_nchw(x: np.ndarray, padding: int) -> np.ndarray:
    """对 NCHW 张量的『空间维度』(H, W) 做零填充；N、C 维不动。"""
    if padding == 0:
        return x
    # np.pad 的 pad_width 与轴一一对应：(N不填, C不填, H两侧填, W两侧填)
    return np.pad(
        x,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0.0,
    )


# =============================================================================
# 2. conv2d —— 二维卷积（前向）
# =============================================================================
def conv2d(x, weight, bias=None, stride=1, padding=0):
    """二维互相关（deep learning 里俗称『卷积』，其实不翻核）。

    形状约定
    --------
    x      : (N, C_in, H, W)          输入特征图
    weight : (C_out, C_in, kH, kW)    卷积核
    bias   : (C_out,) 或 None         每个输出通道一个偏置
    返回   : (N, C_out, H_out, W_out)

    实现策略：最朴素的『滑窗 + 广播乘加』。
    为了可读性没有用 im2col，但用向量化把 batch/通道/核内乘加交给 numpy，
    只在输出的两个空间维度上写 Python for 循环，速度足够跑 demo/测试。

    🔬 第一性原理：卷积的本质是『局部加权求和 + 权重共享』。
        - 局部：每个输出点只由输入上一个 kH×kW 的小窗口决定（局部连接）；
        - 共享：同一个核在整张图上滑动复用（平移不变，参数量与图大小无关）。
        这两点正是 CNN 相比全连接网络的杀手锏。
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    N, C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = weight.shape
    # ⚠️ 坑：权重的输入通道必须与输入特征图通道一致，否则维度对不上还不报错很难查
    assert C_in == C_in_w, f"输入通道不匹配: x 有 {C_in}，weight 期望 {C_in_w}"

    # 先算输出尺寸（用上面统一的公式，保证与测试一致）
    H_out = conv_output_size(H, kH, stride, padding)
    W_out = conv_output_size(W, kW, stride, padding)

    xp = _pad_nchw(x, padding)  # 填充后的输入
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)

    # 在输出空间上滑窗：对每个输出位置 (i, j)，取对应输入窗口做加权和
    for i in range(H_out):
        h_start = i * stride
        h_end = h_start + kH
        for j in range(W_out):
            w_start = j * stride
            w_end = w_start + kW
            # patch: (N, C_in, kH, kW) —— 当前所有 batch 在这个窗口的切片
            patch = xp[:, :, h_start:h_end, w_start:w_end]
            # 用爱因斯坦求和：对 (C_in, kH, kW) 三个维度乘加，得到 (N, C_out)
            #   patch  下标 n c y x
            #   weight 下标 o c y x
            #   结果    下标 n o        （c,y,x 被求和掉）
            out[:, :, i, j] = np.einsum("ncyx,ocyx->no", patch, weight)

    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        # bias 形状 (C_out,) → reshape 成 (1, C_out, 1, 1) 广播到每个空间位置
        out = out + bias.reshape(1, C_out, 1, 1)

    return out


# =============================================================================
# 3. max_pool2d —— 最大池化（前向）
# =============================================================================
def max_pool2d(x, kernel=2, stride=None, padding=0):
    """二维最大池化。

    x      : (N, C, H, W)
    kernel : 池化窗口边长（正方形窗口）
    stride : 步幅；默认 None 时取 = kernel（不重叠池化，最常见）
    返回   : (N, C, H_out, W_out)

    ⚠️ 坑：这里 padding 用 -inf 而不是 0！
        如果用 0 填充，当某个窗口恰好全落在填充区、而真实值都是负数时，
        max 会错误地选到填充的 0。用 -inf 保证填充值永远不会被选中，
        与 torch.nn.functional.max_pool2d 行为一致。
    """
    x = np.asarray(x, dtype=np.float64)
    if stride is None:
        stride = kernel

    N, C, H, W = x.shape
    H_out = conv_output_size(H, kernel, stride, padding)
    W_out = conv_output_size(W, kernel, stride, padding)

    if padding > 0:
        # 用 -inf 填充，保证 max 不会选到填充值
        xp = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=-np.inf,
        )
    else:
        xp = x

    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        h_start = i * stride
        h_end = h_start + kernel
        for j in range(W_out):
            w_start = j * stride
            w_end = w_start + kernel
            window = xp[:, :, h_start:h_end, w_start:w_end]  # (N, C, k, k)
            # 在最后两个维度（窗口内）取最大
            out[:, :, i, j] = window.max(axis=(2, 3))
    return out


# =============================================================================
# 4. relu / flatten / linear
# =============================================================================
def relu(x):
    """整流线性单元：ReLU(x) = max(0, x)。逐元素，形状不变。

    🔬 为什么用 ReLU 不用 sigmoid？
        - 计算便宜（一个 max）；
        - 正区间导数恒为 1，缓解梯度消失（虽然前向用不到导数，但这是它流行的根因）；
        - 稀疏激活（负的全变 0）。
    ⚠️ 坑：np.maximum 是逐元素取大（两个数组比较），别写成 np.max（那是求整体最大值，会把张量塌成标量）。
    """
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, x)


def flatten(x):
    """把 (N, C, H, W) 展平成 (N, C*H*W)，保留 batch 维。

    展平顺序默认按 C-order（行优先），与 PyTorch 的 torch.flatten(x, 1) 一致：
    先 W 变化最快，然后 H，最后 C。这样接全连接层时权重排布才对得上。
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    return x.reshape(N, -1)


def linear(x, weight, bias=None):
    """全连接层：y = x @ Wᵀ + b。

    形状约定（同 torch.nn.Linear）：
        x      : (N, in_features)
        weight : (out_features, in_features)
        bias   : (out_features,) 或 None
        返回   : (N, out_features)

    ⚠️ 坑：权重是 (out, in) 而不是 (in, out)，所以要转置 weight.T 再右乘。
        这是 PyTorch 的约定，记反了矩阵乘会直接维度报错（幸好会报错）。
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    y = x @ weight.T  # (N, in) @ (in, out) = (N, out)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        y = y + bias.reshape(1, -1)
    return y


# =============================================================================
# 5. 一个把上面积木串起来的迷你 CNN（纯前向），供 demo 使用
# =============================================================================
class TinyCNN:
    """一个玩具级 CNN：Conv → ReLU → MaxPool → Flatten → Linear。

    权重全部随机初始化（固定种子，可复现）。它不会『学习』，
    只用来演示前向数据流 & 特征图可视化。
    """

    def __init__(self, in_channels=1, num_filters=4, kernel=3, num_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        # 卷积核 (C_out, C_in, k, k)，用小尺度初始化避免数值爆炸
        self.conv_w = rng.standard_normal((num_filters, in_channels, kernel, kernel)) * 0.1
        self.conv_b = np.zeros(num_filters)
        self.kernel = kernel
        self.num_filters = num_filters
        # 全连接层权重在第一次前向时按实际展平长度懒初始化
        self.fc_w = None
        self.fc_b = None
        self.num_classes = num_classes
        self.rng = rng
        # 缓存中间激活，方便可视化
        self.cache = {}

    def forward(self, x):
        """x: (N, C_in, H, W) → logits: (N, num_classes)。"""
        z1 = conv2d(x, self.conv_w, self.conv_b, stride=1, padding=1)  # same 卷积
        a1 = relu(z1)
        p1 = max_pool2d(a1, kernel=2, stride=2)
        f = flatten(p1)
        if self.fc_w is None:
            # 懒初始化全连接层，用 Xavier 风格缩放
            in_features = f.shape[1]
            scale = np.sqrt(1.0 / in_features)
            self.fc_w = self.rng.standard_normal((self.num_classes, in_features)) * scale
            self.fc_b = np.zeros(self.num_classes)
        logits = linear(f, self.fc_w, self.fc_b)
        # 缓存各层输出用于画图
        self.cache = {"conv": z1, "relu": a1, "pool": p1, "logits": logits}
        return logits


__all__ = [
    "conv_output_size",
    "receptive_field",
    "conv2d",
    "max_pool2d",
    "relu",
    "flatten",
    "linear",
    "TinyCNN",
]
