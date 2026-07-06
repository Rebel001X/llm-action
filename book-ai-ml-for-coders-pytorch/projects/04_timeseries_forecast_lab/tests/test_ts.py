"""tests/test_ts.py —— 时间序列预测实验的自动化验证。

覆盖需求:
    (a) 窗口形状正确;
    (b) naive / moving-average 的 MAE 是有限正数;
    (c) DNN 训练后 MAE 低于 naive 基线（合成可学序列）;
    (d) Conv1D / LSTM 前向形状正确。

全部离线、CPU、秒级完成，使用固定随机种子保证可复现。
"""

from __future__ import annotations

import math
import os
import sys

import torch

# 让测试无论从哪个目录运行，都能 import 到项目根的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine import predict, train_forecaster  # noqa: E402
from metrics import mae, mse  # noqa: E402
from models import (  # noqa: E402
    Conv1DForecaster,
    DNNForecaster,
    LSTMForecaster,
    moving_average_forecast,
    naive_forecast,
)
from series import make_series, train_valid_split  # noqa: E402
from windows import make_windows  # noqa: E402

WINDOW = 20
HORIZON = 1
SEED = 0


def _prepare():
    """公共夹具: 造序列、切分、窗口化，返回训练/验证张量。"""
    torch.manual_seed(SEED)
    series = make_series(seed=SEED, n=1000)
    train, valid = train_valid_split(series, split=0.8, context=WINDOW)
    Xtr, ytr = make_windows(train, WINDOW, HORIZON)
    Xva, yva = make_windows(valid, WINDOW, HORIZON)
    return Xtr, ytr, Xva, yva


# ---------------------------------------------------------------------------
# (a) 窗口形状正确
# ---------------------------------------------------------------------------
def test_window_shapes():
    series = make_series(seed=SEED, n=200)
    X, y = make_windows(series, window=10, horizon=1)
    n_expected = 200 - 10 - 1 + 1
    assert X.shape == (n_expected, 10)
    assert y.shape == (n_expected, 1)
    assert X.dtype == torch.float32 and y.dtype == torch.float32

    # 多步预测 horizon>1 也要形状正确
    X3, y3 = make_windows(series, window=10, horizon=3)
    n3 = 200 - 10 - 3 + 1
    assert X3.shape == (n3, 10)
    assert y3.shape == (n3, 3)

    # 窗口内容确实是原序列对应片段（第 0 个样本）
    assert torch.allclose(X[0], torch.tensor(series[0:10], dtype=torch.float32))
    assert torch.allclose(y[0], torch.tensor(series[10:11], dtype=torch.float32))


# ---------------------------------------------------------------------------
# (b) 基线 MAE 是有限正数
# ---------------------------------------------------------------------------
def test_baseline_mae_finite_positive():
    _, _, Xva, yva = _prepare()

    naive_mae = mae(naive_forecast(Xva), yva)
    ma_mae = mae(moving_average_forecast(Xva, avg_window=5), yva)

    for score in (naive_mae, ma_mae):
        assert math.isfinite(score), "MAE 必须是有限数"
        assert score > 0.0, "含噪序列上基线 MAE 应为正数"

    # MSE 也应是有限正数
    naive_mse = mse(naive_forecast(Xva), yva)
    assert math.isfinite(naive_mse)
    assert naive_mse > 0.0


# ---------------------------------------------------------------------------
# (c) DNN 训练后 MAE 低于 naive 基线
# ---------------------------------------------------------------------------
def test_dnn_beats_naive():
    Xtr, ytr, Xva, yva = _prepare()

    naive_mae = mae(naive_forecast(Xva), yva)

    torch.manual_seed(SEED)
    dnn = DNNForecaster(WINDOW, HORIZON)
    losses = train_forecaster(dnn, Xtr, ytr, epochs=40, lr=1e-2)

    # 训练损失应明显下降（确实在学习）
    assert losses[-1] < losses[0], "训练损失应下降"

    dnn_mae = mae(predict(dnn, Xva), yva)
    assert dnn_mae < naive_mae, (
        f"DNN 应赢过 naive 基线: dnn={dnn_mae:.5f} vs naive={naive_mae:.5f}"
    )


# ---------------------------------------------------------------------------
# (d) Conv1D / LSTM 前向形状正确
# ---------------------------------------------------------------------------
def test_conv_lstm_forward_shapes():
    _, _, Xva, yva = _prepare()
    n = Xva.shape[0]

    conv = Conv1DForecaster(WINDOW, HORIZON)
    out_conv = conv(Xva)
    assert out_conv.shape == (n, HORIZON)

    lstm = LSTMForecaster(hidden=16, horizon=HORIZON)
    out_lstm = lstm(Xva)
    assert out_lstm.shape == (n, HORIZON)

    # 多步 horizon 的前向形状也要正确
    conv3 = Conv1DForecaster(WINDOW, horizon=3)
    assert conv3(Xva).shape == (n, 3)
    lstm3 = LSTMForecaster(hidden=16, horizon=3)
    assert lstm3(Xva).shape == (n, 3)
