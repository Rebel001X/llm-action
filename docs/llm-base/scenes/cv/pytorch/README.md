# PyTorch 计算机视觉（CV）从底层到手算

> 用 PyTorch 把"图像 → 张量 → 卷积/注意力 → 输出"这条链路从最底层讲清，并对卷积 FLOPs、感受野、显存逐数手算。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]]、[[llm-algo/FLOPs]]、[[ai-infra/算力/GPU工作原理]]、[[ai-infra/ai-hardware/CUDA]]、[[llm-train/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键产物 |
|---|---|---|
| 0 | 一句话锚点 | CV 在 PyTorch 里的本质 |
| 1 | 地基：张量、NCHW、dtype、device | 内存布局直觉 |
| 2 | 图像如何进入网络（DataLoader/transform） | 预处理流水线 |
| 3 | 卷积层原理 + 输出尺寸/参数量手算 | $H_{out}$ 公式 |
| 4 | 卷积 FLOPs 与显存逐数手算 | 数值表 |
| 5 | 感受野、池化、归一化、激活 | 感受野递推 |
| 6 | 经典骨干网络（ResNet 残差块） | 梯度通路 |
| 7 | 任务头：分类/检测/分割 | 输出张量形状 |
| 8 | 训练循环五步与 autograd | 反向传播 |
| 9 | 混合精度 AMP 与显存优化 | fp16/bf16 |
| 10 | 多卡数据并行 DDP + 通信量手算 | AllReduce |
| 11 | ViT：CV 与 Transformer 的桥 | patch embedding |
| 数值 | 端到端 ResNet18 一张图手算 | FLOPs/显存 |
| FAQ | 常见坑表 | 排错 |

---

## 0. 一句话锚点

**计算机视觉在 PyTorch 里 = 把图像编码成形状为 `[N, C, H, W]` 的浮点张量，让它流过一串"权重张量 + 可微算子"，最后在某个张量上算损失、反向求梯度、更新权重。**

所有"高级"概念——卷积、ResNet、检测框、分割掩码——都只是这条主干上不同形状的张量与不同的算子组合。把张量形状和数据流搞清楚，CV 代码就不再神秘。

```
 一张图片                  网络（算子链）                  任务输出
 ┌─────────┐   transform   ┌──────────────────────┐   head   ┌──────────┐
 │ JPG 像素 │ ───────────▶ │ Conv→BN→ReLU→…→Pool   │ ──────▶ │ logits / │
 │ 0..255   │   归一化      │ 全是 张量×权重 的运算  │          │ box/mask │
 └─────────┘              └──────────────────────┘          └──────────┘
       uint8 [H,W,3]            float32 [N,C,H,W]              [N, num_cls]
```

---

## 1. 地基：张量、NCHW、dtype、device

PyTorch 的唯一核心数据结构是 `torch.Tensor`：一块连续内存 + 描述如何解释它的元数据（`shape`、`stride`、`dtype`、`device`）。

### 1.1 为什么 CV 用 NCHW

CV 张量约定为 `[N, C, H, W]`（批、通道、高、宽）。内存里是**行优先（C-contiguous）**展平：W 变化最快，N 最慢。

```
 逻辑视图 [N=2, C=3, H=2, W=2]        内存里的线性排布（stride 决定）
   N0                                  N0C0: a b c d
    C0  [a b]   C1 [e f]   C2 [i j]    N0C1: e f g h
        [c d]      [g h]      [k l]    N0C2: i j k l
   N1                                  N1C0: m n o p
    C0  [m n]   ...                    ...
        [o p]
   stride = (C*H*W, H*W, W, 1) = (12, 4, 2, 1)
```

`stride` 是"沿某维走 1 步，线性地址跳几个元素"。`x[n,c,h,w]` 的地址 $= n\cdot12 + c\cdot4 + h\cdot2 + w\cdot1$。理解 stride 就理解了 `view/reshape/permute` 为何有时不复制内存、有时报错。

### 1.2 dtype 与每元素字节数

| dtype | 字节 | 用途 |
|---|---|---|
| float32 | 4 | 默认训练精度 |
| float16 (fp16) | 2 | AMP，动态范围窄 |
| bfloat16 | 2 | AMP，指数位同 fp32，更稳 |
| uint8 | 1 | 原始像素 0..255 |
| int64 | 8 | 分类标签 / 索引 |

> 一个 `[32,3,224,224]` 的 fp32 输入张量占用 $32\times3\times224\times224\times4 = 19{,}267{,}584$ B $\approx 18.4$ MB。换 fp16 减半。这就是显存估算的最小单元。详见 [[llm-compression/quantization/fp8]]。

### 1.3 device

`x.to('cuda')` 把张量搬到 GPU 显存。CV 训练 99% 时间在 GPU；CPU↔GPU 拷贝（PCIe）是常见瓶颈，DataLoader 的 `pin_memory=True` 可加速这一步。GPU 为何快见 [[ai-infra/算力/GPU工作原理]]。

---

## 2. 图像如何进入网络

```
 磁盘 JPG ─decode→ uint8[H,W,3] ─ToTensor→ float[3,H,W]/255 ─Normalize→ float[3,H,W]
   │              │ (HWC, 0..255)        │ (CHW, 0..1)            │ (均值方差白化)
 Dataset       PIL/cv2               permute+scale            (x-mean)/std
   └──────────────── DataLoader 批量+多进程 ──────────────▶ [N,3,H,W]
```

- **`Dataset`**：定义 `__len__` 和 `__getitem__(i) → (image, label)`。
- **`transforms`**：`ToTensor()` 做了两件事——`HWC→CHW` 的 `permute`，以及 `/255` 把 uint8 缩到 $[0,1]$。`Normalize(mean,std)` 做 $(x-\mu)/\sigma$ 白化（ImageNet 常用 `mean=[0.485,0.456,0.406]`）。
- **`DataLoader`**：`batch_size` 拼批、`num_workers` 多进程预取、`shuffle` 打乱。它把 CPU 预处理与 GPU 计算**重叠**起来，是吞吐关键；重叠思想见 [[llm-optimizer/计算通信重叠]]。

**为什么要 Normalize**：把输入分布拉到均值 0、方差 1 附近，使第一层卷积的激活落在激活函数敏感区，梯度幅度均衡，收敛更快、更稳。

---

## 3. 卷积层：原理 + 尺寸/参数量手算

卷积是 CV 的心脏。一个卷积核就是一小块权重张量 `[C_out, C_in, k, k]`，它在输入特征图上滑窗做点积。

```
 输入 [C_in=1, 5x5]      核 3x3        输出 [C_out=1, 3x3]（stride=1,pad=0）
  1 2 3 0 1            w0 w1 w2       核覆盖左上 3x3，逐元素乘加得 1 个数
  0 1 2 3 0            w3 w4 w5   →   再向右/向下滑窗，得 3x3 个数
  1 0 1 2 3            w6 w7 w8
  2 1 0 1 2            滑窗点积：out[0,0]=Σ input·w
  0 2 1 0 1
```

### 3.1 输出尺寸公式（必背）

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d\,(k-1) - 1}{s} \right\rfloor + 1$$

其中 $p$=padding，$s$=stride，$k$=kernel，$d$=dilation（普通卷积 $d=1$）。

**手算**：$H_{in}=224, k=3, p=1, s=1, d=1$ →
$$H_{out}=\lfloor (224+2-2-1)/1 \rfloor +1 = \lfloor 223 \rfloor +1 = 224.$$
"3×3, pad=1, stride=1"是 same 卷积（尺寸不变），ResNet 大量用它。

**手算 2（下采样）**：$H_{in}=224,k=7,p=3,s=2$ → $H_{out}=\lfloor(224+6-6-1)/2\rfloor+1=\lfloor111.5\rfloor+1=112$。这是 ResNet 第一层 `conv7x7,s2` 把 224→112。

### 3.2 参数量

一个卷积层参数 $= C_{out}\times C_{in}\times k\times k\ (+\,C_{out}\ \text{偏置})$。
3×3、64→128 通道：$128\times64\times3\times3 = 73{,}728$ 个权重。注意：**参数量与特征图空间尺寸 H,W 无关**，但 FLOPs 与之相关（见下节）——这是 CV 显存/算力分析的关键区别。

---

## 4. 卷积 FLOPs 与显存逐数手算

### 4.1 单卷积层 FLOPs

每个输出元素需要 $C_{in}\cdot k\cdot k$ 次乘加（MAC）。按 1 MAC = 2 FLOPs：

$$\text{FLOPs} = 2\cdot \underbrace{(C_{out}\cdot H_{out}\cdot W_{out})}_{\text{输出元素数}}\cdot \underbrace{(C_{in}\cdot k\cdot k)}_{\text{每点运算}}$$

**手算**：输入 `[64,56,56]`，卷积 3×3，64→64，pad=1,s=1，输出 `[64,56,56]`：
$$2\times(64\cdot56\cdot56)\times(64\cdot3\cdot3) = 2\times200{,}704\times576 \approx 2.31\times10^8 = 0.231\ \text{GFLOPs}.$$
ResNet18 全网约 1.8 GFLOPs/图（224 分辨率），就是把所有层这样累加。FLOPs 总览见 [[llm-algo/FLOPs]]。

### 4.2 激活值显存（前向）

每层输出特征图都要在前向时**留住**以备反向用，这是 CV 显存大头。

$$\text{激活显存} = N\cdot C_{out}\cdot H_{out}\cdot W_{out}\cdot \text{bytes}$$

**手算**：batch=32，某层输出 `[256,28,28]`，fp32：
$$32\times256\times28\times28\times4 = 32\times256\times784\times4 \approx 25.7\ \text{MB （单层）}.$$
整张 ResNet 几十层叠加，激活轻松占数 GB——这正是 batch 不能太大的原因，也是梯度检查点（重算换显存）和 AMP（半精度）要解决的问题。

```
 显存账本（训练一张/批）
 ┌──────────────────────────────────────────────┐
 │ 权重 W          : 参数量 × 4B                  │  ← 与 H,W 无关
 │ 梯度 dW         : 参数量 × 4B（与 W 同形）      │
 │ 优化器状态(Adam): 参数量 × 8B（m,v 两份）       │
 │ 激活 activation : Σ 每层输出 × 4B × batch       │  ← 随 batch、分辨率暴涨
 └──────────────────────────────────────────────┘
   Adam 训练显存 ≈ 权重×(4+4+8) + 激活；推理只剩 权重×4 + 单层激活
```

> 经验：CV 训练显存以**激活**为主导（高分辨率），LLM 训练显存以**权重+优化器状态**为主导（参数量巨大）。两套直觉都要有。

---

## 5. 感受野、池化、归一化、激活

### 5.1 感受野递推

感受野 RF = 输出某点"看到"的输入区域大小。逐层递推：

$$RF_{l} = RF_{l-1} + (k_l - 1)\cdot \prod_{i<l} s_i$$

**手算**：三层全是 3×3、stride=1，初始 $RF_0=1$：
- L1：$1+(3-1)\cdot1=3$
- L2：$3+(3-1)\cdot1=5$
- L3：$5+(3-1)\cdot1=7$

```
 输入 ──L1(3x3)──▶ ──L2(3x3)──▶ ──L3(3x3)──▶ 输出一个点
 RF:   1            3            5            7  ← 看到原图 7x7 区域
```
两个 3×3 堆叠（RF=5）等效一个 5×5，但参数更少（$2\cdot9$ vs $25$）、非线性更多——这是 VGG/ResNet 偏爱小核堆叠的理由。

### 5.2 池化、BN、激活

- **MaxPool 2×2,s2**：空间减半、通道不变，无参数，扩大感受野。
- **BatchNorm**：对每个通道在 `(N,H,W)` 维上做 $(x-\mu)/\sqrt{\sigma^2+\epsilon}\cdot\gamma+\beta$，稳定分布、允许更大学习率。训练用 batch 统计，推理用滑动平均统计——**`model.eval()` 必须调**，否则 BN/Dropout 行为错。
- **ReLU** $\max(0,x)$：引入非线性、缓解梯度消失、稀疏激活。

---

## 6. 经典骨干：ResNet 残差块

深网难训的根因是梯度在长链上连乘衰减/爆炸。残差连接给梯度开一条"高速公路"。

```
        x
        │────────────────┐ (identity 捷径)
        ▼                │
   Conv3x3→BN→ReLU       │
        ▼                │
   Conv3x3→BN            │
        ▼                ▼
        └────► (+) ◄─────┘
              ▼
            ReLU
        输出 = ReLU(F(x) + x)
```

数学上 $y=F(x)+x$，则 $\dfrac{\partial y}{\partial x}=F'(x)+1$。那个 **+1** 保证即使 $F'(x)\approx0$，梯度也能原样传回，解决退化问题。这与 Transformer 的残差思想同源，见 [[llm-algo/transformer/模型架构]]。

ResNet18 结构：`conv7x7,s2 → maxpool → [64×2, 128×2, 256×2, 512×2] 残差块 → GAP → fc1000`。每个 stage 第一个块用 stride=2 下采样、通道翻倍。

---

## 7. 任务头：分类 / 检测 / 分割

骨干（backbone）输出特征图，不同"头"把它变成不同任务的输出张量。

```
 backbone 特征 [N, 512, 7, 7]
   │
   ├─ 分类头  : GlobalAvgPool→[N,512]→Linear→[N,1000]          → 每类分数
   │
   ├─ 检测头  : 在特征图每个格点预测 框(4)+类别(C)+置信度(1)    → [N, A, 4+1+C]
   │            （Anchor/Grid，如 YOLO、Faster R-CNN）
   │
   └─ 分割头  : 上采样回原分辨率，逐像素分类 → [N, num_cls, H, W]
                （FCN/U-Net：编码器下采样 + 解码器上采样 + skip）
```

| 任务 | 输出张量形状 | 典型损失 |
|---|---|---|
| 图像分类 | `[N, num_cls]` | CrossEntropy |
| 语义分割 | `[N, num_cls, H, W]` | 逐像素 CrossEntropy / Dice |
| 目标检测 | `[N, A, 4+1+C]` | 回归(框)+分类 |

**U-Net 的 skip 连接**：把编码器同分辨率特征拼接到解码器，找回下采样丢失的细节边界——和残差捷径同样是"给信息开旁路"的思想。

---

## 8. 训练循环五步与 autograd

PyTorch 训练永远是这五步，背下来：

```
 for x, y in loader:
   optimizer.zero_grad()        # 1. 清空上一轮梯度（梯度默认累加！）
   out = model(x)               # 2. 前向：建动态计算图，存中间激活
   loss = criterion(out, y)     # 3. 算标量损失
   loss.backward()              # 4. 反向：autograd 沿图链式法则求 dW
   optimizer.step()             # 5. 用 dW 更新 W：W -= lr * dW (SGD)
```

```
 前向（建图）              反向（autograd 回传）
 x→[Conv]→a1→[ReLU]→a2→…→loss     loss.backward()
        │     │     │              dL/dW = dL/da · da/dW  链式相乘
        存激活 存激活 存激活    ◀──── 沿存下的激活逐层回传
```

- **`zero_grad()` 为何必须**：`.grad` 是累加的（为支持梯度累积），不清零会把多个 batch 的梯度叠在一起。
- **梯度累积技巧**：每 K 个小 batch 才 `step()` 一次，等效放大 batch、省显存。
- **`with torch.no_grad()`**：推理时关闭建图，省下激活显存与开销。

---

## 9. 混合精度 AMP 与显存优化

fp16/bf16 让激活与计算减半显存、用上 Tensor Core 提速。

```
 AMP 流程（torch.cuda.amp）
   with autocast():            # 算子自动选 fp16/fp32（matmul/conv 走 fp16）
       out = model(x); loss = criterion(out,y)
   scaler.scale(loss).backward()   # loss 放大，防 fp16 下小梯度下溢成 0
   scaler.step(optimizer)          # 反缩放后更新；遇 inf/nan 自动跳过该步
   scaler.update()
```

- **为何要 GradScaler**：fp16 最小正规数约 $6\times10^{-5}$，小梯度会下溢为 0。把 loss 乘上大因子（如 $2^{16}$），梯度同步放大避开下溢，更新前再除回。
- **bf16 vs fp16**：bf16 指数位与 fp32 相同（动态范围大），通常**不需要** GradScaler，A100/H100 上更省心。
- **其它省显存招**：梯度检查点（前向不存激活、反向重算，时间换显存）、`channels_last` 内存格式（卷积更亲和 Tensor Core）。FP8 趋势见 [[llm-compression/quantization/fp8]]、量化基础见 [[llm-compression/quantization/量化基础]]。

---

## 10. 多卡数据并行 DDP + 通信量手算

单卡装不下大 batch / 想加速 → 多卡。CV 标配 **DistributedDataParallel (DDP)**：每卡一份完整模型、各喂一份数据子批，反向后 **AllReduce** 同步梯度求平均。

```
 GPU0      GPU1      GPU2      GPU3
 model     model     model     model    （各一份完整副本）
  │batch0   │batch1   │batch2   │batch3
  ▼         ▼         ▼         ▼
 dW0       dW1       dW2       dW3
  └────────── AllReduce 求和/平均 ──────────┘
  ▼         ▼         ▼         ▼
 同步后的 dW（四卡完全一致）→ 各自 step()
```

### 通信量手算

Ring-AllReduce 每卡收发约 $2(P-1)/P \times M$ 字节（$P$=卡数，$M$=梯度总字节）。

**手算**：ResNet50 约 $2.5\times10^7$ 参数，fp16 梯度 $M=2.5\times10^7\times2=50$ MB，$P=4$：
$$\text{每卡通信} \approx 2\times\frac{3}{4}\times50 = 75\ \text{MB / step}.$$
若每秒 10 个 step，单卡需 750 MB/s 有效带宽——NVLink 轻松满足，PCIe 则可能成瓶颈。AllReduce 原理见 [[ai-infra/网络/集合通信原语]]；与计算重叠以隐藏通信见 [[llm-optimizer/计算通信重叠]]。

> DDP 关键正确性点：① 每卡用 `DistributedSampler` 切分数据（不重叠）；② BatchNorm 跨卡不同步时各卡用各自小批统计，需更大 batch 或换 `SyncBatchNorm`；③ DDP 在反向时**边算边通信**（bucket 重叠），故比老的 DataParallel 高效得多。张量并行等更激进切法见 [[llm-inference/大模型推理张量并行]]。

---

## 11. ViT：CV 与 Transformer 的桥

Vision Transformer 把"卷积归纳偏置"换成"纯注意力"，让 CV 与 LLM 统一。

```
 图 [3,224,224]
   │ 切 16×16 patch（共 14×14=196 个）
   ▼
 196 个 patch，每个拉平 16·16·3=768 维 ─Linear→ token [196,768]
   │ + [CLS] token + 位置编码 → [197,768]
   ▼
 ┌── Transformer Encoder × L（多头自注意力 + MLP，全是残差） ──┐
   ▼
 取 [CLS] → Linear → [num_cls]
```

- **patch embedding 本质**：一个 `kernel=16, stride=16` 的卷积——所以 ViT 第一层仍可视作卷积。
- 自注意力让任意两 patch 直接交互（全局感受野），但缺卷积的平移等变与局部性，故需大数据（或蒸馏）才打得过 CNN。
- 注意力底层优化（省显存、提速）见 [[llm-optimizer/FlashAttention]]；KV-Cache 是推理侧概念，CV 分类无自回归一般用不到，了解见 [[llm-optimizer/kv-cache]]。位置编码思想见 [[llm-algo/旋转编码RoPE]]。

---

## 数值手算：ResNet18 跑一张 224×224 图

把前面的公式串起来，估算单图前向的算力与显存量级。

**① 第一层 conv7x7,s2，3→64**
- 输出尺寸：$\lfloor(224+6-6-1)/2\rfloor+1 = 112$ → `[64,112,112]`
- 参数：$64\times3\times7\times7 = 9{,}408$
- FLOPs：$2\times(64\cdot112\cdot112)\times(3\cdot7\cdot7)=2\times802816\times147\approx 0.236$ GFLOPs

**② 某中间 3×3,64→64,s1，特征 56×56**
- FLOPs：$2\times(64\cdot56\cdot56)\times(64\cdot9)\approx 0.231$ GFLOPs（见 §4.1）

**③ 全网累加** ≈ **1.8 GFLOPs/图**（标准值）。batch=256 时一次前向 $\approx 461$ GFLOPs；反向约为前向 2 倍 → 一步 $\approx 1.4$ TFLOPs。
A100 (fp16 峰值 ~312 TFLOPS，实测利用率约 40%) → 一步纯算 $\approx 1.4/124 \approx 11$ ms（理论下限，未计数据/通信）。

**④ 输入张量显存**：`[256,3,224,224]` fp32 $= 256\times3\times224\times224\times4 \approx 147$ MB；fp16 减半 73 MB。

**⑤ 权重显存**：ResNet18 约 $1.17\times10^7$ 参数，fp32 权重 $\approx 45$ MB；Adam 训练总占用 $\approx 45\times(4+4+8)/4 = 45\times4 = 180$ MB（权重+梯度+m+v），其余几 GB 全是**激活**——印证 §4.2 的"CV 训练显存由激活主导"。训练总览见 [[llm-train/README]]。

---

## 常见问题

| 现象 / 问题 | 根因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 激活随 batch/分辨率暴涨 | 减 batch、AMP、梯度检查点、`channels_last` |
| 推理精度比训练差 | 忘了 `model.eval()` | BN 用滑动统计、关 Dropout |
| 第二个 epoch loss 异常 | 忘了 `zero_grad()` | 每步先清梯度（或刻意梯度累积） |
| 多卡比单卡还慢 | 通信成瓶颈 / PCIe | NVLine/NVSwitch、bucket 重叠、增大计算量 |
| `view()` 报错 contiguous | permute 后内存非连续 | 用 `.reshape()` 或先 `.contiguous()` |
| 输入形状报错 | HWC vs CHW / 少了 batch 维 | `ToTensor` 转 CHW、`unsqueeze(0)` 加 N |
| fp16 训练出 NaN | 小梯度下溢 / 溢出 | 用 GradScaler 或换 bf16 |
| GPU 利用率低 | DataLoader 喂不上 | 加 `num_workers`、`pin_memory`、预取 |
| BN 在小 batch 不稳 | 单卡小批统计噪声大 | `SyncBatchNorm` 或 GroupNorm |
| 卷积输出尺寸不符预期 | pad/stride/dilation 算错 | 套 §3.1 公式逐项核对 |

> 工具与环境（PyTorch / torchvision / PyAV `pip install av` / CUDA 版本）随官方迭代，**版本与默认命令以官方文档为准**，本文只讲稳定不变的原理与机制。

---

## 🔗 跳转链接

- 总图谱：[[00-知识地图]]
- 架构与残差思想：[[llm-algo/transformer/模型架构]]
- 算力估算：[[llm-algo/FLOPs]]
- 注意力底层：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 位置编码：[[llm-algo/旋转编码RoPE]]
- 训练全景：[[llm-train/README]]
- 精度与量化：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 多卡通信：[[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]] · [[llm-inference/大模型推理张量并行]]
- 硬件底座：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
