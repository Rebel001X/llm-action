# -*- coding: utf-8 -*-
"""
run_demo.py —— 端到端演示：在合成 FashionMNIST 上分别训练 DNN 与 CNN 并对比。

运行：
    python run_demo.py

它会：
1) 生成确定性合成数据（离线、CPU、秒级）。
2) 训练第 2 章的 DNN 与第 3 章的 CNN。
3) 打印两者的测试准确率对比。
4) 把两条训练损失曲线画在一起，保存到 figures/loss_curve.png。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图（保存到文件）
import matplotlib.pyplot as plt

from data import make_loaders
from engine import evaluate, train
from models import build_cnn, build_dnn


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) 数据
    train_loader, test_loader = make_loaders(
        n_per_class_train=200, n_per_class_test=60, batch_size=64, seed=0
    )

    results = {}
    histories = {}

    # 2) 分别训练 DNN 与 CNN
    for name, builder in [("DNN", build_dnn), ("CNN", build_cnn)]:
        print(f"\n=== 训练 {name} ===")
        model = builder()
        hist = train(
            model,
            train_loader,
            epochs=8,
            lr=1e-3,
            val_loader=test_loader,
            patience=3,
            verbose=True,
        )
        acc = evaluate(model, test_loader)
        results[name] = acc
        histories[name] = hist
        print(f"{name} 测试准确率 = {acc:.3f}")

    # 3) 打印对比
    print("\n=== 准确率对比 ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.3f}")
    print("（随机猜测的基线约为 0.10）")

    # 4) 画损失曲线
    plt.figure(figsize=(7, 4.5))
    for name, hist in histories.items():
        plt.plot(range(1, len(hist) + 1), hist, marker="o", label=f"{name} train loss")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("DNN vs CNN training loss (synthetic FashionMNIST)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(fig_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\n损失曲线已保存到: {out}")


if __name__ == "__main__":
    main()
