# 01 · PyTorch 视觉实验室（DNN + CNN）

> 对应书本 **第 2 章《计算机视觉入门：FashionMNIST 与神经元》** 与 **第 3 章《卷积神经网络：在图像中检测特征》**。

用一套**确定性合成的 "FashionMNIST 形状" 数据**，先用全连接 **DNN**、再用 **CNN** 做 10 类图像分类，
亲手对比"把图片拉直喂给全连接层" vs "让卷积核在图像上滑动检测特征"的差别。

**全程离线、纯 CPU、几秒跑完**，不下载任何数据或模型，`pytest` 全绿。

---

## 1. 三分钟跑通

```bash
cd projects/01_vision_pytorch_lab

# （可选）装依赖；本机若已装好 torch/numpy/matplotlib/pytest 可跳过
pip install -r requirements.txt

# 跑测试（应当全绿）
python -m pytest -q

# 跑端到端演示：训练 DNN 与 CNN，打印准确率对比，并把损失曲线存到 figures/
python run_demo.py
```

`run_demo.py` 会在终端打印类似：

```
=== 准确率对比 ===
DNN: 0.9xx
CNN: 0.9xx
（随机猜测的基线约为 0.10）
```

并在 `figures/loss_curve.png` 生成一张 DNN vs CNN 的训练损失曲线。

---

## 2. 数据从哪来？（为什么能离线还能学得会）

真实 FashionMNIST 需要联网下载。本项目改用 **`data.py` 里确定性合成的数据**：

- 形状与真实数据**完全一致**：图像 `(N, 1, 28, 28)`，标签 `0..9` 共 10 类。
- 每一类都有一个**独特的空间图案**（竖条纹 / 横条纹 / 对角线 / 四角方块 / 居中块 / 边框 / 十字…），
  再叠加**高斯噪声**和**小幅随机平移**。
- 因此类别在像素上"可分"，小模型能学到 **>90%** 的准确率；同时噪声与平移让它不至于平凡。
- 所有随机性由固定种子（`np.random.default_rng(seed)` / `torch.manual_seed`）控制，**完全可复现**。

这套图案正好也解释了 **CNN 为什么擅长图像**：这些边缘、方块、条纹本质上是"局部特征"，
卷积核滑过去就能检测到，无需像 DNN 那样把空间结构拉直丢弃。

---

## 3. 文件说明

| 文件 | 作用 |
|------|------|
| `data.py` | 合成数据 `synthetic_fashion()`；一站式 `make_loaders()`（含训练集标准化、防泄漏）；以及**可选的**真实数据加载 `load_real_fashionmnist()`（用 `try import torchvision` 门控）。 |
| `models.py` | `build_dnn()`：`Flatten→Linear→ReLU→Linear(10)`（第 2 章）；`build_cnn()`：`[Conv2d→ReLU→MaxPool2d]×2→Flatten→Linear(10)`（第 3 章）。 |
| `engine.py` | `train()` 标准训练循环 + 简单 early stopping（监控验证/训练损失，恢复最优权重）；`evaluate()` 返回准确率。 |
| `run_demo.py` | 端到端：合成数据上分别训练 DNN 与 CNN，打印准确率对比，保存损失曲线到 `figures/`。 |
| `tests/test_vision.py` | 断言 (a) 前向输出形状 `(batch,10)`；(b) 训练几轮后 loss 明显下降；(c) 准确率 >0.3（远高于随机 0.1）。另测数据形状、可复现性、early stopping、torchvision 门控。 |
| `conftest.py` | 让 pytest 能从项目根导入 `data/models/engine`。 |
| `requirements.txt` | 依赖清单（CPU 版 torch + numpy + matplotlib + pytest）。 |

---

## 4. 如何换成"真实数据 / 真实模型"

**换成真实 FashionMNIST（只改数据源，模型和训练代码不用动）：**

1. 安装 torchvision 并联网：`pip install torchvision`。
2. 把 `run_demo.py` 里的 `make_loaders(...)` 换成：

   ```python
   from torch.utils.data import DataLoader
   from data import load_real_fashionmnist

   train_ds = load_real_fashionmnist(train=True)   # 首次会自动 download 到 ./_data
   test_ds  = load_real_fashionmnist(train=False)
   train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
   test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)
   ```

   因为真实数据每个样本同样是 `(image[1,28,28], label)`，`models.py` / `engine.py` **一行都不用改**。

**换成更强/真实的骨干模型：**

- 把 `build_cnn()` 换成更深的卷积网络，或用 `torchvision.models`（如 ResNet，把首层 conv 改成单通道输入即可）。
- 想接"真实 LLM / 扩散模型"的多模态实验？本项目刻意保持第 2–3 章的最小闭环；
  真实大模型请走后续章节对应的项目，那里会用 `try import transformers/diffusers` 做**可选路径**，
  缺失就自动回落到离线合成分支，保证测试始终能过。

---

## 5. 你应该观察到的现象

- DNN 和 CNN 都能把准确率从随机的 ~0.10 拉到 **>0.9**（合成任务比真实简单）。
- 损失曲线单调下降；early stopping 会在验证损失不再改善时提前停车并回滚到最优权重。
- 在更"贴近真实、更难"的数据上（调大 `data.py` 的 `noise`），CNN 相对 DNN 的优势会更明显——
  这正是第 3 章想让你体会的核心：**卷积利用了图像的空间结构**。
