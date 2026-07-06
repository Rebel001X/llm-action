# 项目 04 · 时间序列预测实验（基线 / DNN / Conv1D / LSTM）

> 对应原书 **第 9~11 章**：
> - 第 9 章 · 理解序列与时间序列数据（趋势 / 季节性 / 噪声、窗口化、朴素基线）
> - 第 10 章 · 用 DNN 预测序列（把窗口拉平送进全连接网络）
> - 第 11 章 · 用卷积（Conv1D）与循环（LSTM）方法做序列建模

一句话：**造一条“趋势 + 季节性 + 噪声”的合成序列，用同一套窗口数据，把 5 种预测方法从弱到强跑一遍，用 MAE 客观比较，并画图。**

全程 **CPU、离线、秒级**，不下载任何数据/模型，`pytest` 全绿。

---

## 3 分钟跑通

```bash
# 1) 安装依赖（若已装 torch/numpy/matplotlib 可跳过）
pip install -r requirements.txt

# 2) 跑测试（应全绿，约 2 秒）
python -m pytest -q

# 3) 跑完整对比实验 + 出图
python run_demo.py
#   -> 终端打印各方法验证集 MAE 排名
#   -> figures/forecast_vs_truth.png  预测 vs 真实曲线
#   -> figures/mae_comparison.png     各方法 MAE 柱状图
```

参考输出（种子固定，结果可复现）：

```
验证集 MAE 对比（越小越好）:
  lstm         MAE = 0.02208
  dnn          MAE = 0.02449
  conv1d       MAE = 0.02579
  naive        MAE = 0.05454
  moving_avg   MAE = 0.15323
最优方法: lstm  (naive 基线 = 0.05454)
```

**读法**：三个神经网络（DNN / Conv1D / LSTM）的 MAE 都只有 naive 基线的一半左右——因为它们学会了序列里**确定性的趋势 + 季节性**，把误差压到接近噪声下限；而 `moving_avg`（滑动平均）反而更差，因为它把季节性的拐点“抹平”了。这正是第 9~11 章想让你体会的核心结论。

---

## 文件说明

| 文件 | 作用 | 对应章 |
|------|------|--------|
| `series.py` | `make_series(seed)` 生成 **trend + seasonality + noise** 的 1D 序列；`train_valid_split` 按时间顺序切分 | 第 9 章 |
| `windows.py` | `make_windows(series, window, horizon=1)` 把序列切成 `(X, y)` 监督样本，返回 `torch.float32` 张量 | 第 9 章 |
| `metrics.py` | `mae` / `mse` 两个误差指标，同时接受 numpy 与 torch 输入 | 第 9 章 |
| `models.py` | 5 种方法：`naive_forecast`、`moving_average_forecast`、`DNNForecaster`、`Conv1DForecaster`（`nn.Conv1d`）、`LSTMForecaster`（`nn.LSTM`） | 第 10、11 章 |
| `engine.py` | `train_forecaster(model, X, y, ...)` 通用训练循环（MSE + Adam），返回逐 epoch 损失 | 第 10 章 |
| `run_demo.py` | 串起全流程：造数据 → 窗口化 → 5 法对比 → 出图到 `figures/` | 第 9~11 章 |
| `tests/test_ts.py` | 4 个自动化测试（见下） | — |

### 5 种方法一览

```
不学习的统计基线:
  naive_forecast          预测“下一个 = 上一个值”         —— 随机游走假设，最难打败的基线之一
  moving_average_forecast 预测“下一个 = 窗口内均值”       —— 去噪，但会抹平季节性拐点

会学习的神经网络（都吃 (N, window) 窗口，吐 (N, horizon) 预测）:
  DNNForecaster    窗口拉平 -> MLP                        —— 第 10 章
  Conv1DForecaster (N,1,window) -> nn.Conv1d 提局部形状   —— 第 11 章
  LSTMForecaster   (N,window,1) -> nn.LSTM 沿时间步循环   —— 第 11 章
```

---

## 测试覆盖（`tests/test_ts.py`）

| 测试 | 验证内容 |
|------|----------|
| `test_window_shapes` | (a) 窗口 `X`/`y` 形状正确（含 `horizon>1`），且窗口内容确实取自原序列对应片段 |
| `test_baseline_mae_finite_positive` | (b) `naive` / `moving-average` 的 MAE、MSE 都是**有限正数** |
| `test_dnn_beats_naive` | (c) 训练损失下降，且 **DNN 训练后 MAE < naive 基线**（合成可学序列） |
| `test_conv_lstm_forward_shapes` | (d) `Conv1D` / `LSTM` 前向输出形状正确（含 `horizon>1`） |

```bash
python -m pytest -q
# 4 passed in ~2s
```

---

## 关键设计点（为什么这样造数据）

- **序列 = 趋势 + 季节性 + 噪声**：季节性用“基波 + 二次谐波”正弦叠加，形状更像真实数据；噪声很小、季节性明显，所以序列“可学”，神经网络能稳定赢过 naive 基线（这是测试 (c) 成立的前提）。
- **时间顺序切分，绝不随机打乱**：验证段永远在训练段“之后”，避免用未来信息预测过去（数据泄漏）。验证段额外向前带 `window` 个点作上下文，让验证窗口也有完整历史。
- **固定随机种子**：`np.random.default_rng(seed)` + `torch.manual_seed(seed)`，保证每次运行、每台机器结果一致，测试稳定。
- **小而快**：序列 1000 点、窗口 20、训练 40 epoch、网络很小，整套实验 CPU 上秒级完成。

---

## 如何换成真实数据 / 真实场景

本项目的数据接口和模型接口是**解耦**的，换真实数据只需替换“造序列”这一步：

1. **换成真实时间序列**（股价、电力负荷、气温、销量……）：
   把 `make_series(...)` 换成你自己的加载函数，只要最终得到一条 **1D numpy 数组**即可，后面的 `make_windows` / 模型 / 训练全部不用改。
   ```python
   import pandas as pd
   series = pd.read_csv("your_data.csv")["value"].to_numpy(dtype="float64")
   train, valid = train_valid_split(series, split=0.8, context=WINDOW)
   ```
   真实数据通常还需要**标准化**（如 `(x-mean)/std`，用**训练段**的统计量），因为真实序列量纲差别大、趋势更强。

2. **多变量 / 多步预测**：
   - 多步预测：`make_windows(series, window, horizon=k)`，模型 `horizon=k` 即可直接输出未来 k 步。
   - 多变量：把 `Conv1d(in_channels=1, ...)` / `LSTM(input_size=1, ...)` 的输入维改成特征数，并让 `make_windows` 产出 `(N, window, features)`。

3. **换成更强的模型 / 真实“时序大模型”**：
   本项目刻意只用 CPU 小网络。若要接真实预训练时序模型或 LLM 路线，建议做成**可选路径**（`try import`，缺失就回落到这里的离线合成分支，保证 `pytest` 依旧绿）：
   ```python
   try:
       from your_ts_foundation_model import load_pretrained   # 例如某时序基础模型
       model = load_pretrained(...)
   except Exception:
       model = DNNForecaster(WINDOW, HORIZON)  # 离线回落，测试不依赖外部下载
   ```
   同理，若想用 LLM 做“文本化时间序列预测 / 解释”，也应把联网/大模型调用包在 `try/except` 里，默认走本地合成分支。

---

## 依赖

见 `requirements.txt`：`torch`（CPU 即可）、`numpy`、`matplotlib`、`pytest`。**不需要** `datasets` / `diffusers` / 联网。
