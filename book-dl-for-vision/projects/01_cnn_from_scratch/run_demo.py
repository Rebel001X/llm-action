# -*- coding: utf-8 -*-
"""
run_demo.py —— 用纯 NumPy 的 CNN 前向，对一张『合成图』做卷积并可视化特征图。

它做三件事：
    1) 合成一张 32x32 灰度图（十字 + 圆环 + 斜纹），不需要任何外部图片/数据集；
    2) 用几个『手工设计的经典卷积核』（Sobel 边缘、拉普拉斯、模糊）做卷积，
       让你直观看到不同核提取的不同特征；
    3) 跑一遍 TinyCNN，展示随机核 → ReLU → MaxPool 的特征图，并画出感受野随层增长。

输出：在本目录生成两张 PNG：
    - feature_maps.png   经典核的特征图
    - tinycnn_maps.png   TinyCNN 各层特征图

全程离线、纯 CPU、几秒钟跑完。
"""

import os
import sys

# ⚠️ Windows 控制台默认 GBK 编码，直接 print emoji/部分中文会抛 UnicodeEncodeError。
#    在最开头把 stdout/stderr 重设为 UTF-8，保证 print 不崩（本机踩坑点）。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib

# ⚠️ 必须在 import pyplot 之前设置后端为 Agg：
#    Agg 是无窗口的『离屏』后端，服务器/无显示器环境也能出图，只写文件不弹窗。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体设置：优先微软雅黑，退回黑体；关掉负号变方块的毛病
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 让脚本能 import 同目录的 cnn_numpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_numpy import conv2d, relu, max_pool2d, TinyCNN, receptive_field  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_synthetic_image(size=32):
    """合成一张 (size, size) 的灰度图：十字 + 圆环 + 斜纹，特征丰富便于观察卷积效果。"""
    img = np.zeros((size, size), dtype=np.float64)
    c = size // 2

    # 1) 中间一个十字（水平 + 垂直亮条）
    img[c - 1:c + 1, :] = 1.0
    img[:, c - 1:c + 1] = 1.0

    # 2) 一个圆环
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    ring = (r > size * 0.30) & (r < size * 0.38)
    img[ring] = 0.8

    # 3) 左上角一片斜纹（每隔几行画一条）
    for k in range(0, size // 2, 3):
        for t in range(size // 2):
            yidx = t
            xidx = t + k
            if xidx < size // 2:
                img[yidx, xidx] = 0.6

    return img


# 手工设计的经典 3x3 卷积核（教科书常见）
CLASSIC_KERNELS = {
    "Sobel-X 竖直边缘": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float64),
    "Sobel-Y 水平边缘": np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]], dtype=np.float64),
    "Laplacian 全向边缘": np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=np.float64),
    "Box Blur 均值模糊": np.ones((3, 3), dtype=np.float64) / 9.0,
}


def demo_classic_kernels(img):
    """用经典核逐个卷积，画成一排对比图。"""
    x = img.reshape(1, 1, *img.shape)  # NCHW

    n = len(CLASSIC_KERNELS)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("原始合成图\n(十字+圆环+斜纹)")
    axes[0].axis("off")

    for ax, (name, k) in zip(axes[1:], CLASSIC_KERNELS.items()):
        w = k.reshape(1, 1, 3, 3)
        # same 卷积（pad1）保持尺寸，方便与原图对齐观察
        out = conv2d(x, w, stride=1, padding=1)[0, 0]
        ax.imshow(out, cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle("经典卷积核提取的不同特征（纯 NumPy conv2d）", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(HERE, "feature_maps.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def demo_tinycnn(img):
    """跑 TinyCNN，展示 conv→relu→pool 三个阶段的特征图（取前 4 个通道）。"""
    x = img.reshape(1, 1, *img.shape)
    net = TinyCNN(in_channels=1, num_filters=4, kernel=3, num_classes=3, seed=1)
    logits = net.forward(x)

    conv = net.cache["conv"][0]   # (4, H, W)
    act = net.cache["relu"][0]    # (4, H, W)
    pool = net.cache["pool"][0]   # (4, H/2, W/2)

    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    stages = [("Conv 卷积输出", conv), ("ReLU 激活后", act), ("MaxPool 池化后", pool)]
    for row, (stage_name, feat) in enumerate(stages):
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(feat[col], cmap="viridis")
            ax.set_title(f"{stage_name}\n通道 {col}", fontsize=9)
            ax.axis("off")

    # 计算这条网络到 pool 层的感受野，写进标题
    rf = receptive_field([3, 2], [1, 2])  # conv3(s1) + pool2(s2)
    fig.suptitle(
        f"TinyCNN 各层特征图（4 个随机核）  |  池化层单点感受野 = {rf}×{rf} 像素\n"
        f"logits = {np.round(logits[0], 3)}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = os.path.join(HERE, "tinycnn_maps.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main():
    print("=" * 60)
    print("纯 NumPy CNN 前向 —— 特征图可视化 Demo")
    print("=" * 60)

    img = make_synthetic_image(32)
    print(f"[1/3] 合成图已生成，shape={img.shape}, 值域=[{img.min()}, {img.max()}]")

    p1 = demo_classic_kernels(img)
    print(f"[2/3] 经典核特征图已保存 -> {p1}")

    p2 = demo_tinycnn(img)
    print(f"[3/3] TinyCNN 特征图已保存 -> {p2}")

    # 顺手打印几层的感受野，呼应 README 里的面试点
    print("\n感受野速查（说明小核堆叠可媲美大核）：")
    print(f"  单个 3x3            -> {receptive_field([3], [1])}")
    print(f"  两个 3x3 堆叠        -> {receptive_field([3, 3], [1, 1])}  (等价 5x5)")
    print(f"  三个 3x3 堆叠        -> {receptive_field([3, 3, 3], [1, 1, 1])}  (等价 7x7)")
    print("\n完成 ✅  打开上面两张 PNG 查看特征图。")


if __name__ == "__main__":
    main()
