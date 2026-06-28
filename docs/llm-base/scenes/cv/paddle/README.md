# PaddlePaddle 计算机视觉实战：人脸关键点检测（Landmark Detection）

> 用飞桨（PaddlePaddle）从零搭一条最小可跑的 CV 训练流水线，主线任务是「人脸/物体关键点回归」，把数据→网络→损失→训练→推理每一步拆到原子级。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[../pytorch/README|scenes/cv/pytorch 同主题对照]] · [[../README|scenes/cv 总览]] · [[../../FLOPs|llm-algo/FLOPs]] · [[../../transformer/模型架构|llm-algo/transformer/模型架构]] · [[../../../../ai-infra/算力/GPU工作原理|ai-infra/算力/GPU工作原理]]

参考：
- 飞桨官方实践教程（关键点检测）：https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/cv/landmark_detection.html （命令/接口默认值以官方为准）

---

## 阅读地图

| 节 | 你会得到什么 | 关键产物 |
|----|--------------|----------|
| 0  | 一句话锚点：关键点检测 = 坐标回归 | 任务定义 |
| 1  | 地基：飞桨张量、动态图、CV 数据排布 NCHW | 心智模型 |
| 2  | 数据集 BSDS500 / 人脸点位的下载与解析 | Dataset |
| 3  | 数据增强为什么必须「同步变换标签」 | Transform |
| 4  | 网络：从 Backbone 到回归头 | `paddle.nn` |
| 5  | 损失：L2 / Wing Loss / NME 指标 | Loss/Metric |
| 6  | 训练循环：前向→反向→优化器→学习率 | train loop |
| 7  | 显存 & FLOPs 手算：一张图要花多少 | 数值估算 |
| 8  | 推理与可视化：把点画回原图 | predict |
| 9  | 多卡（数据并行）一图看懂 | `fleet` |
| —  | 数值手算 / 常见问题 / 跳转链接 | 速查 |

---

## 0. 一句话锚点

**人脸关键点检测（facial landmark / keypoint detection）= 给定一张 $H\times W$ 的人脸图，输出 $K$ 个点的像素坐标 $(x_i, y_i)$。**

它本质上是一个**回归问题**，不是分类：

```
输入图像                    模型                      输出向量
┌──────────┐          ┌───────────┐          (x1,y1, x2,y2, ... xK,yK)
│   👤      │  ──────► │  CNN +    │  ──────►  长度 = 2K 的实数向量
│  (脸)     │          │ 回归头     │          例如 K=68 → 136 个数
└──────────┘          └───────────┘
```

对比一下三种 CV 任务的「输出形状」，理解为什么关键点是回归：

```
分类 classification :  图 → 1 个类别 id        (argmax over C)
检测 detection      :  图 → N 个 [类别,框]      (框=4个数+1类别)
分割 segmentation   :  图 → H×W 的逐像素标签
关键点 landmark     :  图 → 2K 个连续坐标值     ← 回归（本文）
```

一句话：**分类输出离散标签用交叉熵；关键点输出连续坐标用 L2 类损失。** 这决定了后面网络最后一层、损失函数、评价指标全都不同。

---

## 1. 地基：飞桨的张量、动态图与 NCHW 排布

### 1.1 飞桨 = 动态图为主的深度学习框架

飞桨（PaddlePaddle）和 PyTorch 同属「动态图（命令式）」框架：代码写到哪、算到哪，可以直接 `print` 中间张量。核心对象是 **Tensor**（多维数组 + 自动求导信息）。

```
paddle.Tensor
 ├─ 数值数据      （像 numpy.ndarray）
 ├─ dtype         float32 / int64 ...
 ├─ place         CPUPlace / CUDAPlace(0)   ← 数据在哪块设备
 └─ stop_gradient False → 参与反向传播（要算梯度）
                  True  → 冻结，不算梯度
```

> 类比：`stop_gradient=False` ≈ PyTorch 的 `requires_grad=True`，但**布尔含义相反**，初学最容易踩坑。

### 1.2 图像在内存里长什么样：NCHW

飞桨卷积默认数据排布是 **NCHW**：

```
N = batch（一批几张图）
C = channel（通道，RGB=3）
H = height（高，像素行数）
W = width （宽，像素列数）

一个 batch 的张量形状：[N, C, H, W]
例：32 张 256×256 的彩图 → [32, 3, 256, 256]
```

为什么是 NCHW 而不是 NHWC？因为 cuDNN/卷积核在 GPU 上对「同一通道的连续像素」做卷积最高效，把 C 放在 H、W 前面，使同通道平面在内存里更连续。ASCII 看内存展开顺序（最右维 W 变化最快）：

```
[N=0]
  [C=0(R通道)]  行0: w0 w1 w2 ...   ← 内存里先铺满 R 整张平面
                行1: ...
  [C=1(G通道)]  ...                 ← 再铺 G 整张平面
  [C=2(B通道)]  ...                 ← 再铺 B
[N=1] ...
```

> 注意：用 `paddle.vision` / OpenCV 读进来的图常是 **HWC + uint8 + BGR/RGB**，喂给网络前必须转成 **CHW + float32 + 归一化**，这一步在第 3 节的 `Transform` 里做。

---

## 2. 数据集：下载与解析

原始 README 里给的是 **BSDS500**（伯克利分割数据集），它本身是做边缘/分割的；做关键点更典型的是人脸点位数据集（如 300W、WFLW）。无论哪个，**数据流水线骨架一样**，下面用通用骨架讲清楚。

### 2.1 下载（命令以官方为准）

```bash
# 示例：下载并解压（具体 URL/数据集以官方教程为准）
wget --no-check-certificate \
  http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -zxvf BSR_bsds500.tgz
```

> 工具/URL 默认值可能随站点变化，**以飞桨官方教程页为准**；这里只讲流程：拿到「图片 + 标注文件」两样东西。

### 2.2 标注长什么样

关键点标注通常是一行一张图：图片路径 + $2K$ 个坐标。

```
img_0001.jpg  x1 y1 x2 y2 ... x68 y68
img_0002.jpg  x1 y1 ...
```

### 2.3 自定义 Dataset（飞桨范式）

飞桨用 `paddle.io.Dataset` + `DataLoader`，和 PyTorch 几乎一一对应：

```python
import paddle, cv2, numpy as np
from paddle.io import Dataset, DataLoader

class LandmarkDataset(Dataset):
    def __init__(self, list_file, transform=None):
        self.samples = [l.split() for l in open(list_file)]
        self.transform = transform

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img = cv2.imread(rec[0])                       # HWC, BGR, uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # → RGB
        pts = np.array(rec[1:], dtype='float32').reshape(-1, 2)  # [K,2]
        if self.transform:
            img, pts = self.transform(img, pts)        # 图和点一起变换！
        return img, pts.flatten()                      # 图[C,H,W], 标签[2K]

    def __len__(self):
        return len(self.samples)
```

`DataLoader` 负责**批处理 + 多进程预取 + 打乱**：

```
原始样本流 ─┐
            ├─►[worker0]┐
shuffle ───┤  [worker1]├─► collate 拼 batch ─► [N,C,H,W],[N,2K] ─► GPU
            └─►[worker2]┘
```

---

## 3. 数据增强：为什么必须「同步变换标签」

这是关键点任务**最容易出 bug** 的地方。分类任务里水平翻转图片，标签不变；但关键点任务里，**翻转/旋转/缩放图片时，坐标标签必须做同样的几何变换**，否则点和脸对不上。

```
原图（点在左眼）          水平翻转后
┌─────────┐              ┌─────────┐
│ • 左眼   │   翻转图      │   右眼 • │   ← 像素镜像了
│      右  │   但若标签    │ ✗(点没动)│   ← 标签还指原位置 = 错!
└─────────┘   不变 ✗      └─────────┘
```

正确做法：图变换的**同一个几何矩阵**作用到点上。

### 3.1 缩放（resize 到固定输入尺寸 $H_0\times W_0$）

设原图 $W\times H$，目标 $W_0\times H_0$，缩放比例：

$$s_x = \frac{W_0}{W}, \quad s_y = \frac{H_0}{H}$$

每个点坐标同步乘比例：

$$x' = x \cdot s_x, \quad y' = y \cdot s_y$$

```
原图 640×480, 点(320,240)  ──resize→ 256×256
sx = 256/640 = 0.4 ,  sy = 256/480 ≈ 0.533
x' = 320 × 0.4   = 128
y' = 240 × 0.533 ≈ 128   → 点跟着缩到 (128,128) ✓
```

### 3.2 水平翻转（概率 0.5）

$$x' = W_0 - 1 - x, \quad y' = y$$

> 额外坑：翻转后**左右语义点要交换索引**（左眼角↔右眼角），否则点位对但语义错。

### 3.3 归一化（数值稳定的前提）

像素 $0\sim255$ 太大、量纲不一，要标准化：

$$\hat{x}_{\text{pix}} = \frac{x_{\text{pix}}/255 - \mu}{\sigma}$$

坐标标签也常**归一化到 $[0,1]$**（除以宽高），让回归目标尺度统一、损失更好收敛：

$$\tilde{x} = x'/W_0, \quad \tilde{y} = y'/H_0$$

```
增强流水线（图与点并行）：
图: HWC uint8 ─► resize ─► flip? ─► /255,标准化 ─► HWC→CHW float32
点:  [K,2]    ─► ×s    ─► W-1-x ─► /W,/H 归一化  ─► flatten [2K]
              └────────── 同一套几何参数 ──────────┘
```

---

## 4. 网络：Backbone + 回归头

关键点网络 = **特征提取 Backbone**（把图压成语义特征）+ **回归头**（把特征映成 $2K$ 个数）。

```
[N,3,256,256]
   │  Backbone: 一堆 Conv+BN+ReLU+下采样
   ▼
[N,C',8,8]        ← 空间被压小、通道变多（语义浓缩）
   │  GlobalAvgPool / Flatten
   ▼
[N, C'*8*8] 或 [N,C']
   │  Linear 回归头
   ▼
[N, 2K]           ← 直接就是 K 个点的 (x,y)
```

### 4.1 一个最小可跑的飞桨网络

```python
import paddle.nn as nn

class LandmarkNet(nn.Layer):
    def __init__(self, K=68):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2D(ci, co, 3, padding=1),
                nn.BatchNorm2D(co),
                nn.ReLU(),
                nn.MaxPool2D(2))          # H,W 各减半
        self.feat = nn.Sequential(
            block(3, 32),    # 256→128
            block(32, 64),   # 128→64
            block(64, 128),  # 64→32
            block(128, 256), # 32→16
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),       # [N,256,16,16]→[N,256,1,1]
            nn.Flatten(),                  # → [N,256]
            nn.Linear(256, 2 * K))         # → [N,2K]

    def forward(self, x):
        return self.head(self.feat(x))     # [N,2K]
```

### 4.2 卷积输出尺寸公式（必背）

一层卷积/池化后的空间尺寸：

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

其中 $k$=核大小，$p$=padding，$s$=stride。

```
Conv 3×3, p=1, s=1 : H_out = (H+2-3)/1 +1 = H   （尺寸不变）
MaxPool 2×2, s=2   : H_out = (H-2)/2  +1 = H/2  （减半）
256 →128 →64 →32 →16   （4 次 block 后）
```

---

## 5. 损失与指标

### 5.1 L2（MSE）损失：最朴素的回归损失

预测 $\hat{p}$ 与真值 $p$（都是 $2K$ 维），均方误差：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{2K}\sum_{i=1}^{2K}(\hat{p}_i - p_i)^2$$

```python
loss = paddle.nn.functional.mse_loss(pred, target)   # 标量
```

### 5.2 Wing Loss：关键点专用，照顾小误差

L2 对小误差「不够敏感」（误差平方后更小），关键点要求精细，常用 **Wing Loss** 在小误差区放大梯度：

$$\text{wing}(e)=\begin{cases} w\ln(1+|e|/\epsilon), & |e|<w \\ |e|-C, & \text{otherwise}\end{cases}$$

```
loss
 │   小误差区: 对数曲线，梯度大（逼着模型把点对准）
 │  ╱‾‾‾‾  大误差区: 线性，对离群点不过度惩罚
 │ ╱
 └────────────► |e|
   -w   0   w
```

### 5.3 评价指标 NME（归一化平均误差）

关键点不看准确率，看 **NME（Normalized Mean Error）**：所有点的平均像素距离，再除以一个「归一化尺度 $d$」（常用两眼间距或框对角线），消除人脸大小影响：

$$\text{NME} = \frac{1}{K}\sum_{i=1}^{K}\frac{\lVert \hat{p}_i - p_i \rVert_2}{d}$$

```
单点欧氏距离: ‖(x̂-x, ŷ-y)‖ = √((x̂-x)² + (ŷ-y)²)
NME 越小越好；常乘 100 用百分比报告（如 NME=3.5%）
```

---

## 6. 训练循环：前向 → 反向 → 优化器

飞桨动态图训练四步（和 PyTorch 同构，只是 API 名字不同）：

```
        ┌─────────────────────────────────────────┐
        │  for batch (img, target) in loader:      │
        │     pred = model(img)        ① 前向       │
        │     loss = loss_fn(pred,tgt) ② 算损失     │
        │     loss.backward()          ③ 反向求梯度  │
        │     opt.step()               ④ 更新参数    │
        │     opt.clear_grad()         ⑤ 清梯度!     │
        └─────────────────────────────────────────┘
```

```python
model = LandmarkNet(K=68)
opt = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())

for epoch in range(EPOCHS):
    model.train()
    for img, target in train_loader:
        pred = model(img)                                  # ①
        loss = paddle.nn.functional.mse_loss(pred, target) # ②
        loss.backward()                                    # ③
        opt.step()                                         # ④
        opt.clear_grad()                                   # ⑤ 不清梯度会累加!
```

> ⑤ `clear_grad()` 等价 PyTorch 的 `zero_grad()`。飞桨默认**梯度累加**，忘了清就等于无意中放大了 batch，是经典 bug。

### 6.1 为什么要学习率衰减

梯度下降一步：$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$。$\eta$（学习率）太大震荡、太小慢。常用「先大后小」：

```
lr
 │‾‾‾\___          warmup 升上去 → 平台 → 衰减下来
 │   /    \___
 └──────────────► step
飞桨: paddle.optimizer.lr.PiecewiseDecay / CosineAnnealingDecay
```

---

## 数值手算：一张图的显存与 FLOPs

以第 4 节的 `LandmarkNet`，输入 `[N=32, 3, 256, 256]`，逐层算两个量：**激活值显存** 与 **卷积 FLOPs**。

### 7.1 激活值显存（前向中间张量）

每个 float32 = 4 字节。一层激活张量元素数 = $N\times C\times H\times W$。

```
层             形状[N,C,H,W]           元素数            显存(float32,字节)
输入           32×3×256×256          6,291,456         × 4 = 24.0 MB
block1 后      32×32×128×128         16,777,216        × 4 = 64.0 MB
block2 后      32×64×64×64           8,388,608         × 4 = 32.0 MB
block3 后      32×128×32×32          4,194,304         × 4 = 16.0 MB
block4 后      32×256×16×16          2,097,152         × 4 =  8.0 MB
────────────────────────────────────────────────────────────────
前向激活合计 ≈ 24+64+32+16+8 = 144 MB（仅前几张主特征图）
```

> 反向传播要**保存前向激活**用于求梯度，所以训练显存 ≈ 激活 + 参数 + 优化器状态 + 梯度。Adam 优化器每个参数额外存 2 份状态（一阶、二阶动量），即「参数显存 ×3」量级。详见 [[../../FLOPs|llm-algo/FLOPs]] 与 [[../../../../ai-infra/算力/GPU工作原理|GPU工作原理]]。

### 7.2 卷积 FLOPs

一层卷积的乘加次数（FLOPs，按乘加各算 1 计 2）：

$$\text{FLOPs} = 2 \cdot N \cdot H_{out} W_{out} \cdot C_{out} \cdot (C_{in} \cdot k^2)$$

算 `block1`（Conv 3×3，$C_{in}=3, C_{out}=32$，输出 $256\times256$，先卷积再池化，卷积时空间还是 256）：

```
2 × 32(N) × 256 × 256 × 32(Cout) × (3 × 3×3(Cin·k²))
= 2 × 32 × 65,536 × 32 × 27
≈ 2 × 32 × 65,536 × 864
≈ 3.62 × 10^9  FLOPs ≈ 3.62 GFLOPs   （仅这一层、这一批）
```

```
直觉：FLOPs ∝ 输出像素数(H·W) × 输出通道 × 输入通道 × 核面积
下采样让 H·W 快速变小，所以浅层(大分辨率)最吃算力。
```

### 7.3 NME 端到端手算

假设 $K=4$ 个点，预测与真值（像素）、两眼间距 $d=80$：

```
点  真值(x,y)    预测(x̂,ŷ)   距离 ‖·‖
1   (30,40)      (33,44)     √(3²+4²)=5
2   (90,40)      (90,43)     √(0²+3²)=3
3   (50,90)      (54,93)     √(4²+3²)=5
4   (70,90)      (67,86)     √(3²+4²)=5
平均距离 = (5+3+5+5)/4 = 4.5 px
NME = 4.5 / 80 = 0.05625 = 5.6%
```

---

## 8. 推理与可视化

训练完，关掉梯度、切到 `eval`，把归一化坐标还原回像素再画点。

```python
model.eval()
with paddle.no_grad():                      # 推理不建反向图，省显存
    pred = model(img)                       # [1, 2K]，归一化坐标
pts = pred.numpy().reshape(-1, 2)
pts[:, 0] *= W0; pts[:, 1] *= H0            # 反归一化回像素
# cv2.circle 把每个点画回原图
```

```
模型输出(归一化)        反归一化              画到图上
(0.5,0.5) ───×W0,×H0──► (128,128) ──cv2.circle──► 👁 标在脸上
```

> 推理省显存的关键：`paddle.no_grad()` + `model.eval()`（BatchNorm 用滑动均值而非当前 batch 统计）。原理同 KV 推理省内存的思路，见 [[../../../llm-inference/README|llm-inference/README]]。

---

## 9. 多卡数据并行（一图看懂）

单卡训不动/想加速时，飞桨用 `paddle.distributed.fleet` 做**数据并行**：每张卡放完整模型副本，各吃 batch 的一部分，反向后**用 AllReduce 同步梯度**。

```
        全局 batch=128，4 卡，每卡 micro-batch=32
GPU0[模型副本]─grad0─┐
GPU1[模型副本]─grad1─┤   AllReduce      每卡拿到
GPU2[模型副本]─grad2─┤ ───求和/平均──►  相同的平均梯度 → 各自 step
GPU3[模型副本]─grad3─┘
```

通信量手算（环形 AllReduce，参数量 $P$，float32）：每卡收发约 $2P$ 字节量级。设模型 $P=10^7$ 参数：

```
单卡同步流量 ≈ 2 × P × 4 字节 = 2 × 10^7 × 4 = 80 MB / step
（环形算法把它摊到 N-1 步，单链路压力 ≈ 80MB/(N-1)）
```

集合通信原语（AllReduce/AllGather/Broadcast）的原子级讲解见 [[../../../../ai-infra/网络/集合通信原语|集合通信原语]]；计算与通信如何重叠见 [[../../../llm-optimizer/计算通信重叠|计算通信重叠]]。

---

## 常见问题

| 问题 | 原因 | 解法 |
|------|------|------|
| 点全挤在图中心 | 标签没归一化 / 学习率太大 | 坐标 /W,/H；调小 lr |
| 翻转后点和脸镜像反了 | 只翻图没翻标签，或没交换左右索引 | 几何变换同步作用到点 + 交换语义索引 |
| loss 不降 | 忘了 `clear_grad()`，梯度累加 | 每步 `opt.clear_grad()` |
| 训练正常推理崩 | 推理没 `model.eval()`，BN 用了 batch 统计 | `eval()` + `paddle.no_grad()` |
| 显存爆 (OOM) | batch 太大 / 激活太多 | 减 batch、降分辨率、用 AMP 混合精度 |
| `stop_gradient` 把全网冻住 | 误把可训练张量设 `stop_gradient=True` | 仅冻结需要冻的层 |
| BGR/RGB 颜色错乱 | OpenCV 默认 BGR | `cv2.cvtColor(..., BGR2RGB)` |
| NME 看着小但点不准 | 归一化尺度 d 选太大 | 用两眼间距/框对角线统一口径 |

---

## 🔗 跳转链接

- 同主题 PyTorch 对照实现：[[../pytorch/README|scenes/cv/pytorch/README]]
- CV 场景总览：[[../README|scenes/cv/README]] · [[../../../README|llm-base/scenes/README]]
- 计算量/FLOPs 估算：[[../../FLOPs|llm-algo/FLOPs]]
- 模型架构（卷积/注意力对照）：[[../../transformer/模型架构|llm-algo/transformer/模型架构]] · [[../../mlp|llm-algo/mlp]]
- 推理省显存机制：[[../../../llm-inference/README|llm-inference/README]] · [[../../../llm-inference/KV-Cache优化|llm-inference/KV-Cache优化]]
- 混合精度 / FP8 数值：[[../../../llm-compression/quantization/量化基础|量化基础]] · [[../../../llm-compression/quantization/fp8|fp8]]
- 多卡并行与通信：[[../../../../ai-infra/网络/集合通信原语|集合通信原语]] · [[../../../llm-optimizer/计算通信重叠|计算通信重叠]] · [[../../../llm-inference/大模型推理张量并行|大模型推理张量并行]]
- 硬件基础：[[../../../../ai-infra/算力/GPU工作原理|GPU工作原理]] · [[../../../../ai-infra/ai-hardware/CUDA|CUDA]]
- 训练框架：[[../../../ai-framework/deepspeed/README|deepspeed]] · [[../../../ai-framework/megatron-lm/README|megatron-lm]]
- 知识总入口：[[00-知识地图]]
