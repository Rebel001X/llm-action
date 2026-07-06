"""run_demo.py —— 一键对比五种预测方法（对应第 9~11 章的综合实验）。

流程:
    1. 造一条 trend+seasonality+noise 的序列（series.py）。
    2. 按时间切 train/valid，并各自窗口化（windows.py）。
    3. 统计基线: naive / moving-average（models.py，无需训练）。
    4. 训练 DNN / Conv1D / LSTM（engine.py）。
    5. 在验证集上比较 MAE，并把“预测 vs 真实”画到 figures/。

运行:
    python run_demo.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图（保存到文件）
import matplotlib.pyplot as plt
import numpy as np
import torch

from engine import predict, train_forecaster
from metrics import mae
from models import (
    Conv1DForecaster,
    DNNForecaster,
    LSTMForecaster,
    moving_average_forecast,
    naive_forecast,
)
from series import make_series, train_valid_split
from windows import make_windows

WINDOW = 20
HORIZON = 1
SEED = 0


def main() -> dict[str, float]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) 造序列 + 时间顺序切分（验证段带 WINDOW 个点上下文）
    series = make_series(seed=SEED, n=1000)
    train, valid = train_valid_split(series, split=0.8, context=WINDOW)

    # 2) 窗口化
    Xtr, ytr = make_windows(train, WINDOW, HORIZON)
    Xva, yva = make_windows(valid, WINDOW, HORIZON)

    results: dict[str, float] = {}

    # 3) 统计基线（不训练）
    results["naive"] = mae(naive_forecast(Xva), yva)
    results["moving_avg"] = mae(moving_average_forecast(Xva, avg_window=5), yva)

    # 4) 会学习的模型
    preds_for_plot: dict[str, np.ndarray] = {}

    dnn = DNNForecaster(WINDOW, HORIZON)
    train_forecaster(dnn, Xtr, ytr, epochs=40, lr=1e-2, verbose=False)
    p = predict(dnn, Xva)
    results["dnn"] = mae(p, yva)
    preds_for_plot["DNN"] = p.numpy().reshape(-1)

    conv = Conv1DForecaster(WINDOW, HORIZON)
    train_forecaster(conv, Xtr, ytr, epochs=40, lr=1e-2, verbose=False)
    p = predict(conv, Xva)
    results["conv1d"] = mae(p, yva)
    preds_for_plot["Conv1D"] = p.numpy().reshape(-1)

    lstm = LSTMForecaster(hidden=32, horizon=HORIZON)
    train_forecaster(lstm, Xtr, ytr, epochs=40, lr=1e-2, verbose=False)
    p = predict(lstm, Xva)
    results["lstm"] = mae(p, yva)
    preds_for_plot["LSTM"] = p.numpy().reshape(-1)

    # 5) 打印对比表
    print("\n验证集 MAE 对比（越小越好）:")
    print("-" * 34)
    for name, score in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<12s} MAE = {score:.5f}")
    print("-" * 34)
    best = min(results, key=results.get)
    print(f"最优方法: {best}  (naive 基线 = {results['naive']:.5f})\n")

    # 6) 画图: 预测 vs 真实
    os.makedirs("figures", exist_ok=True)
    truth = yva.numpy().reshape(-1)

    # 注: 图内文字用英文, 避免默认字体缺中文字形导致方块/警告
    plt.figure(figsize=(11, 5))
    plt.plot(truth, label="truth", color="black", linewidth=1.6)
    for name, pr in preds_for_plot.items():
        plt.plot(pr, label=f"{name} pred", alpha=0.75, linewidth=1.0)
    plt.title("Time-series forecasting: prediction vs truth (validation)")
    plt.xlabel("validation time step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join("figures", "forecast_vs_truth.png")
    plt.savefig(out1, dpi=110)
    plt.close()

    # 7) 画图: 各方法 MAE 柱状图
    plt.figure(figsize=(8, 4.5))
    names = list(results.keys())
    scores = [results[n] for n in names]
    colors = ["#888" if n in ("naive", "moving_avg") else "#3b7" for n in names]
    plt.bar(names, scores, color=colors)
    plt.axhline(results["naive"], color="red", linestyle="--", linewidth=1, label="naive baseline")
    plt.title("Validation MAE by method (gray=baseline, green=neural net)")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join("figures", "mae_comparison.png")
    plt.savefig(out2, dpi=110)
    plt.close()

    print(f"图已保存: {out1}  和  {out2}")
    return results


if __name__ == "__main__":
    main()
