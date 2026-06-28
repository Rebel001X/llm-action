# 计算机视觉（CV）任务全景与底层机制

> 一句话定位：把"图像/视频 → 像素张量 → 卷积/注意力特征 → 任务头"这条链路从最底层讲清，覆盖分类/检测/分割/生成等主流任务，并把卷积尺寸、FLOPs、显存逐数手算出来。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-optimizer/FlashAttention]] · [[llm-base/scenes/multi-modal/README]]

## 阅读地图

| 节 | 内容 | 你将能手算/画出 |
|----|------|----------------|
| 0 | 一句话锚点 | CV 的统一抽象 |
| 1 | 地基：像素张量 NCHW | 一张图占多少字节 |
| 2 | 卷积原子操作 | 输出尺寸 + FLOPs + 参数量 |
| 3 | 池化 / 步长 / 感受野 | 感受野逐层累加 |
| 4 | 经典骨干网络谱系 | ResNet 残差为什么 work |
| 5 | 图像分类任务头 | softmax + 交叉熵 |
| 6 | 目标检测 | anchor / IoU / NMS 手算 |
| 7 | 语义/实例分割 | 上采样与 mIoU |
| 8 | 图像生成（GAN/Diffusion） | 去噪一步公式 |
| 9 | 超分/风格迁移/姿态/跟踪 | 各自的输入输出形状 |
| 10 | Vision Transformer | patch 数与序列长度 |
| 11 | 数值手算总表 | 一个 ResNet-50 前向账本 |

---

## 0. 一句话锚点

> **一切 CV 任务 = 设计一个函数 $f_\theta:\ \text{图像张量} \to \text{结构化输出}$，区别只在"输出长什么样"和"用什么损失教它"。**

- 输入永远是张量 $X\in\mathbb{R}^{N\times C\times H\times W}$（批 × 通道 × 高 × 宽）。
- 输出形状决定任务：
  - 一个向量 → **分类**；
  - 一组框+类别 → **检测**；
  - 一张逐像素标签图 → **分割**；
  - 一张同尺寸新图 → **生成/超分/风格迁移**；
  - 一组坐标点 → **姿态估计/关键点**。

```
            ┌─────────────────── 通用 CV 流水线 ───────────────────┐
图像 → [预处理] → [骨干 Backbone] → [颈 Neck] → [任务头 Head] → 输出
 HxWx3   归一化     CNN/ViT 提特征    FPN/上采样   分类/框/掩码
         resize     (共享, 可预训练)  (多尺度融合)  (任务特定)
```

**关键直觉**：Backbone 在不同任务间高度复用（ImageNet 预训练权重），换任务往往只换 Head + 损失。这正是"迁移学习"在 CV 落地的根本原因。

---

## 1. 地基：图像就是一个数字张量

一张图片在内存里就是 $H\times W$ 个像素，每个像素 $C$ 个通道（RGB 即 $C=3$），每个数通常是 `uint8`（0~255）或归一化后的 `float32`。

```
单个像素 (i,j):  [R, G, B]  例如 [128, 64, 200]
整张图   H×W×C:
        W 方向 →
   ┌───────────────┐
 H │ ███████████   │  每格 = 3 个字节(uint8) 或 3×4 字节(float32)
 ↓ │ ███████████   │
   └───────────────┘
```

**布局 NCHW vs NHWC**：
- `NCHW`（PyTorch 默认）：同一通道的像素在内存连续，利于卷积；
- `NHWC`（TensorFlow / TensorRT 常用）：同一像素的通道连续，利于某些硬件向量化。

### 手算：一张图占多少显存？

一张 $224\times224$ RGB 图，`float32`：

$$224\times224\times3\times4\ \text{字节} = 602{,}112\ \text{B} \approx 0.57\ \text{MB}$$

一个 batch=256：$0.57\times256\approx147\ \text{MB}$（**只是输入，激活会大得多，见第 11 节**）。

---

## 2. 卷积：CV 的核心原子操作

卷积 = 一个小窗口（卷积核 kernel）在图上滑动，每个位置做"逐元素相乘再求和"。

```
输入 5×5(单通道)        核 3×3            输出 3×3
┌─────────────┐       ┌───────┐         ┌─────────┐
│ a b c d e   │       │w1 w2 w3│  滑动→  │ o11 ... │
│ f g h i j   │   *   │w4 w5 w6│  ====>  │ ...     │
│ k l m n o   │       │w7 w8 w9│         │         │
│ ...         │       └───────┘         └─────────┘
└─────────────┘
o11 = a·w1+b·w2+c·w3+f·w4+g·w5+h·w6+k·w7+l·w8+m·w9
```

为什么有效：**局部连接**（只看邻域→平移不变性）+ **权重共享**（同一个核扫全图→参数量小）。

### 2.1 输出尺寸公式（必背）

$$H_{out}=\left\lfloor\frac{H_{in}+2P-K}{S}\right\rfloor+1$$

其中 $K$=核大小，$P$=padding，$S$=stride。宽同理。

**手算**：$H_{in}=224,\ K=7,\ P=3,\ S=2$（ResNet 第一层）：
$$\left\lfloor\frac{224+6-7}{2}\right\rfloor+1=\left\lfloor\frac{223}{2}\right\rfloor+1=111+1=112$$
→ 输出 $112\times112$。

### 2.2 参数量

一个卷积层：$\text{params}=K_h\cdot K_w\cdot C_{in}\cdot C_{out}+C_{out}(\text{bias})$

**手算**：$3\times3$ 卷积，$C_{in}=64,\ C_{out}=128$：
$$3\times3\times64\times128+128=73{,}728+128=73{,}856$$

### 2.3 FLOPs（每层乘加量）

$$\text{FLOPs}\approx 2\cdot H_{out}\cdot W_{out}\cdot C_{out}\cdot(K_h\cdot K_w\cdot C_{in})$$
（因子 2 = 一次乘 + 一次加；详见 [[llm-algo/FLOPs]]）

**手算**：上层在 $112\times112$ 输出上：
$$2\times112\times112\times128\times(3\times3\times64)\approx 1.85\times10^{9}\ \text{FLOPs}\ (\approx1.85\ \text{GFLOPs})$$

> 直觉：**卷积的计算量集中在"特征图大 × 通道多"的层**，这也是为什么早期层算力贵、深层参数贵。

---

## 3. 池化、步长与感受野

**池化（Pooling）**：在窗口内取最大（MaxPool）或平均（AvgPool），降分辨率、增鲁棒性，无参数。

```
MaxPool 2×2, stride 2:
 1 3 | 2 1        3 5
 4 2 | 5 0   →
 -----+----       8 6
 7 1 | 6 4
 0 8 | 3 2
```

**感受野（Receptive Field）**：输出一个像素"看到"原图多大区域。逐层递推：
$$RF_{l}=RF_{l-1}+(K_l-1)\cdot\prod_{i<l}S_i$$

**手算**：三层 $3\times3$、stride=1 堆叠：
- 第1层 RF=3；第2层 RF=3+(3-1)·1=5；第3层 RF=5+(3-1)·1=7。

> 这解释了"两个 $3\times3$ 等效一个 $5\times5$ 感受野，但参数更少（$2\times9$ vs $25$）且非线性更多"——VGG 的设计哲学。

---

## 4. 骨干网络谱系（Backbone）

```
LeNet(1998) → AlexNet(2012) → VGG(2014) → GoogLeNet/Inception
                                              │
                                              ▼
ResNet(2015,残差) → DenseNet → ResNeXt → EfficientNet(2019)
                                              │
                                              ▼
                          Vision Transformer(2020) → Swin → ConvNeXt
```

**ResNet 的残差为什么 work**：学 $H(x)=F(x)+x$，让网络只需学"残差 $F(x)$"。

```
   x ──────────────┐(恒等捷径 identity)
   │               │
   ▼               ▼
[conv→BN→ReLU→conv→BN] ──(+)── ReLU ── 输出
```

梯度回传时 $\frac{\partial}{\partial x}(F(x)+x)=F'(x)+1$，那个 **"+1" 让梯度不会消失**，于是能堆到 100+ 层。这与 [[llm-algo/transformer/模型架构]] 里 Transformer 的残差连接是同一思想。

---

## 5. 图像分类：最基础的任务头

Backbone 出特征图 → 全局平均池化（GAP）→ 全连接 → softmax。

```
特征图 7×7×2048 ──GAP──> 向量 2048 ──FC──> logits 1000 ──softmax──> 概率
```

**softmax + 交叉熵**：
$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\quad \mathcal{L}=-\sum_i y_i\log p_i$$

**手算**：3 类 logits $z=[2,1,0.1]$：
- $e^z=[7.389,\ 2.718,\ 1.105]$，和 $=11.212$
- $p=[0.659,\ 0.242,\ 0.099]$
- 真值为第 0 类：$\mathcal{L}=-\log 0.659=0.417$

> 与 LLM 输出 token 概率完全同构（见 [[llm-inference/解码策略]]），只是这里"词表"换成"类别表"。

---

## 6. 目标检测：框 + 类别

输出每个目标的 **边界框 (x,y,w,h) + 类别 + 置信度**。两大流派：

```
两阶段(Two-Stage)         一阶段(One-Stage)
Faster R-CNN              YOLO / SSD / RetinaNet
图→区域提议(RPN)→精修      图→直接密集预测框
准但慢                    快, 端到端
```

### 6.1 IoU（交并比）——评估框是否准

$$IoU=\frac{\text{交集面积}}{\text{并集面积}}$$

**手算**：预测框 $[0,0,2,2]$（面积 4），真值框 $[1,1,3,3]$（面积 4）：
- 交集 $[1,1,2,2]$ 面积 $=1$；并集 $=4+4-1=7$
- $IoU=1/7\approx0.143$ → 重叠很差。

### 6.2 NMS（非极大值抑制）——去重叠框

```
按置信度排序: B1(0.9) B2(0.85) B3(0.7)
取 B1 → 凡与 B1 的 IoU>阈值(0.5) 的框删掉
→ B2 与 B1 IoU=0.6 删除; B3 IoU=0.1 保留
结果: B1, B3
```

### 6.3 Anchor（锚框）

预设不同尺度/长宽比的参考框，网络只学"相对偏移"，降低回归难度。$\Delta=(t_x,t_y,t_w,t_h)$ 编码框的修正量。

---

## 7. 语义/实例分割：逐像素标签

| 任务 | 输出 |
|------|------|
| 语义分割 | 每像素一个类别（不区分同类个体） |
| 实例分割 | 每像素类别 + 个体 ID（Mask R-CNN） |
| 全景分割 | 语义 + 实例统一 |

**编码器-解码器（U-Net）结构**——下采样提语义、上采样恢分辨率，跳连补细节：

```
输入 ──[下采样×4]──> 瓶颈 ──[上采样×4]──> 同尺寸掩码
   │   skip ─────────────────────►  │
   │   skip ───────────────►        │
   (低层细节通过跳连直接喂给解码器)
```

**上采样（反卷积/双线性）** 把小特征图放大。**评估指标 mIoU**：各类 IoU 取平均。

**手算 mIoU**：2 类，类0 IoU=0.8，类1 IoU=0.6 → $mIoU=(0.8+0.6)/2=0.7$。

---

## 8. 图像生成：GAN 与扩散模型

### 8.1 GAN（对抗）

```
噪声 z ──[生成器 G]──> 假图 ──┐
                              ├─►[判别器 D]─► 真/假
真实图 ───────────────────────┘
G 想骗过 D, D 想识破 G —— 博弈到纳什均衡
```

### 8.2 Diffusion（扩散，当前主流）

前向：逐步加高斯噪声把图变纯噪声；反向：训一个网络逐步去噪。

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

去噪网络 $\epsilon_\theta(x_t,t)$ 预测噪声，损失 $\mathcal{L}=\mathbb{E}\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2$。

```
x0 ──加噪──> x1 ──> ... ──> xT (纯噪声)
x0 <─去噪─ x1 <── ... <── xT   (反向采样, U-Net/DiT 预测噪声)
```

Stable Diffusion 在**潜空间**做扩散（VAE 压缩），文生图见 [[llm-base/scenes/multi-modal/README]]。文本条件通过交叉注意力注入（机制与 [[llm-optimizer/FlashAttention]] 同源）。

---

## 9. 其它任务的输入输出形状

| 任务 | 输入 | 输出 | 关键点 |
|------|------|------|--------|
| 超分辨率 SR | 低清 $H\times W$ | 高清 $rH\times rW$ | 上采样 + 重建损失，PSNR/SSIM 评估 |
| 风格迁移 | 内容图 + 风格图 | 融合图 | 内容损失 + Gram 矩阵风格损失 |
| 姿态估计 | 人像 | $K$ 个关键点热力图 | argmax 热力图取坐标 |
| 目标跟踪 | 视频帧序列 | 每帧目标框轨迹 | 检测 + 关联（ReID/卡尔曼） |
| 视频分类 | $T$ 帧 clip | 动作类别 | 3D 卷积或时序注意力 |
| 图像去噪/去模糊 | 退化图 | 清晰图 | 残差学习退化-清晰映射 |
| 抠图 Matting | 图 | alpha 透明度图 | 逐像素回归 [0,1] |

**手算 PSNR**：MSE=0.0025（归一化到 [0,1]），峰值=1：
$$PSNR=10\log_{10}\frac{1^2}{0.0025}=10\log_{10}400\approx26\ \text{dB}$$

---

## 10. Vision Transformer：把图像切成 token

ViT 把卷积换成"图像分块 + Transformer"。一张图切成 $P\times P$ 的 patch，每个 patch 拉平+线性投影=一个 token。

```
224×224 图, patch=16:
  patch 数 = (224/16)×(224/16) = 14×14 = 196 个 token
  + 1 个 [CLS] token → 序列长度 197
  每个 token 维度 = 16×16×3 = 768 → 投影到 d_model
```

**手算序列长度对显存的影响**：注意力是 $O(L^2)$。$L=197$ 尚小，但高分辨率（$L=1024+$）时 $L^2$ 爆炸——这正是 [[llm-optimizer/FlashAttention]] 在视觉里同样重要的原因。位置编码可用可学习或 [[llm-algo/旋转编码RoPE]] 类方案。

```
图 → [切 patch] → [线性投影+位置编码] → [Transformer Encoder×N] → [CLS]→分类头
```

Swin Transformer 用"滑窗 + 层级下采样"把复杂度降回近线性，更适合检测/分割等密集任务。

---

## 11. 数值手算总表：ResNet-50 前向账本

以 batch=1、输入 $224\times224\times3$ 为例（近似量级）：

| 阶段 | 输出特征图 | 通道 | 该层激活内存(float32) |
|------|-----------|------|----------------------|
| conv1 (7×7,s2) | 112×112 | 64 | $112^2\times64\times4\approx3.2$ MB |
| maxpool (s2) | 56×56 | 64 | $\approx0.8$ MB |
| stage2 | 56×56 | 256 | $56^2\times256\times4\approx3.2$ MB |
| stage3 | 28×28 | 512 | $\approx1.6$ MB |
| stage4 | 14×14 | 1024 | $\approx0.8$ MB |
| stage5 | 7×7 | 2048 | $\approx0.4$ MB |
| GAP+FC | 1000 | — | 极小 |

**全模型规模（公认值）**：参数约 **25.6M**，前向计算约 **4.1 GFLOPs**。

**训练显存粗算**：激活内存随 batch 线性放大，且训练要存所有中间激活做反传，外加优化器状态。参数 25.6M：
- 权重 fp32：$25.6\text{M}\times4\approx102$ MB
- Adam 优化器（动量+方差）：$\times2\approx205$ MB
- 梯度：$\approx102$ MB
- → 仅参数相关已 **约 410 MB**，激活随 batch 另算。混合精度可减半，参见 [[llm-algo/FLOPs]] 与 [[llm-train/README]] 的内存估算思路。

```
显存四大块:  [参数] + [梯度] + [优化器状态] + [激活]
            固定        固定        固定          ∝ batch×分辨率
```

---

## 常见问题

| 问题 | 答案 |
|------|------|
| 为什么 CNN 比全连接省参数？ | 权重共享：一个核扫全图，参数量与图大小无关，只与核大小×通道有关 |
| $3\times3$ 卷积为何流行？ | 两个 $3\times3$ = 一个 $5\times5$ 感受野，参数更少($18$ vs $25$)、非线性更多 |
| 检测一阶段 vs 两阶段怎么选？ | 要速度/端到端选 YOLO 系；要极致精度/小目标可选两阶段 |
| 分割上采样有哪些方式？ | 双线性插值、转置卷积、像素重排(PixelShuffle) |
| GAN vs Diffusion？ | GAN 快但难训(模式崩溃)；Diffusion 稳、质量高但采样慢(多步) |
| ViT 一定比 CNN 强？ | 大数据下 ViT 占优；小数据 CNN 的归纳偏置(平移不变)更省样本。ConvNeXt 证明 CNN 仍可逼平 |
| CV 与 LLM 有何相通？ | 残差/注意力/softmax-CE/迁移学习/FLOPs-内存估算方法论完全互通 |
| 高分辨率 OOM 怎么办？ | 降 batch、梯度检查点、混合精度、FlashAttention、滑窗注意力(Swin) |

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 架构同源（残差/注意力）：[[llm-algo/transformer/模型架构]]
- 计算量与内存方法论：[[llm-algo/FLOPs]] · [[llm-train/README]]
- 注意力加速（ViT/扩散同样适用）：[[llm-optimizer/FlashAttention]]
- 位置编码：[[llm-algo/旋转编码RoPE]]
- 多模态/文生图（CLIP/BLIP/Stable Diffusion）：[[llm-base/scenes/multi-modal/README]]
- 概率输出与采样（与分类 softmax 同构）：[[llm-inference/解码策略]]
- 框架实现：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 集合通信（多卡训练 CV 大模型）：[[ai-infra/网络/集合通信原语]]
- GPU 计算原理（卷积/矩阵乘落到硬件）：[[ai-infra/算力/GPU工作原理]]
