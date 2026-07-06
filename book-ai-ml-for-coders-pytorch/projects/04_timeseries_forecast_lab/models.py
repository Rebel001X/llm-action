"""models.py —— 五种预测方法（对应第 10、11 章）。

从“不学习的统计基线”到“会学习的神经网络”，层层递进:

    第 9 章  基线:
        naive_forecast            —— 用上一个值当预测（随机游走假设）。
        moving_average_forecast   —— 用窗口内均值当预测（滑动平均去噪）。

    第 10 章 全连接网络:
        DNNForecaster             —— 把窗口拉平送进 MLP。

    第 11 章 卷积 / 循环:
        Conv1DForecaster          —— nn.Conv1d 在时间轴上滑动提取局部模式。
        LSTMForecaster            —— nn.LSTM 沿时间步循环，捕捉长程依赖。

约定: 所有方法的输入窗口 X 形状为 (N, window)，输出预测形状为 (N, horizon)，
      这样能直接和 windows.make_windows 产出的 y 对齐、送进 metrics 计算。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 统计基线（无参数、不需要训练）
# ---------------------------------------------------------------------------
def naive_forecast(X: torch.Tensor) -> torch.Tensor:
    """朴素预测: 直接用窗口里“最后一个值”作为对下一步的预测。

    这是时间序列里最强、最难打败的基线之一（随机游走假设）。
    输入 X: (N, window) -> 输出: (N, 1)。
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    return X[:, -1:].clone()  # 取每行最后一列，保持 (N,1)


def moving_average_forecast(X: torch.Tensor, avg_window: int | None = None) -> torch.Tensor:
    """滑动平均预测: 用窗口内最后 avg_window 个值的均值作为预测。

    avg_window=None 时对整段输入窗口取均值。均值能压掉噪声，
    但也会“抹平”趋势和季节性的拐点，所以对有明显季节性的序列不如会学习的模型。
    输入 X: (N, window) -> 输出: (N, 1)。
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    if avg_window is None or avg_window >= X.shape[1]:
        sub = X
    else:
        sub = X[:, -avg_window:]
    return sub.mean(dim=1, keepdim=True)  # (N,1)


# ---------------------------------------------------------------------------
# 第 10 章: 全连接 DNN
# ---------------------------------------------------------------------------
class DNNForecaster(nn.Module):
    """把窗口拉平 (N, window) 送进多层感知机, 输出 (N, horizon)。"""

    def __init__(self, window: int, horizon: int = 1, hidden=(32, 16)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = window
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 第 11 章: 一维卷积
# ---------------------------------------------------------------------------
class Conv1DForecaster(nn.Module):
    """用 nn.Conv1d 在时间轴上滑动提取局部形状, 再接全连接输出。

    输入 (N, window) 会被 reshape 成 (N, 1, window):
        1 个输入通道(单变量序列), window 是时间长度。
    卷积用 padding 保持时间长度不变, 便于展平后接线性层。
    """

    def __init__(self, window: int, horizon: int = 1, channels: int = 16, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=channels, kernel_size=kernel, padding=kernel // 2
        )
        self.act = nn.ReLU()
        self.head = nn.Linear(channels * window, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)          # (N, window) -> (N, 1, window)
        z = self.act(self.conv(z))  # (N, channels, window)
        z = z.flatten(1)            # (N, channels*window)
        return self.head(z)         # (N, horizon)


# ---------------------------------------------------------------------------
# 第 11 章: LSTM 循环网络
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """用 nn.LSTM 沿时间步循环, 取最后时刻的隐藏态做预测。

    输入 (N, window) 会被 reshape 成 (N, window, 1):
        每个时间步的特征维是 1(单变量)。
    """

    def __init__(self, hidden: int = 32, horizon: int = 1, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1)     # (N, window) -> (N, window, 1)
        out, _ = self.lstm(z)   # out: (N, window, hidden)
        last = out[:, -1, :]    # 取最后一个时间步 (N, hidden)
        return self.head(last)  # (N, horizon)
