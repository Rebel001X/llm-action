# PyTorch 原生张量并行（Tensor Parallel / DTensor）

> 一句话定位：用 PyTorch 原生的 `DTensor` + `DeviceMesh` 把一层（线性层 / 注意力 / MLP）的权重沿行或列切到多卡上，让单卡放不下、算不快的大矩阵乘法被多卡分摊。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[ai-framework/megatron-lm/README]] · [[ai-infra/网络/集合通信原语]] · [[B07:llm-inference/大模型推理张量并行]]

## 阅读地图

| 节 | 内容 | 你会得到 |
| --- | --- | --- |
| 0 | 一句话锚点 | TP 到底切什么、为什么省显存 |
| 1 | 地基：从单卡到多卡的显存账 | 原文两张 `nvidia-smi` 表的真实含义 |
| 2 | DTensor / DeviceMesh 是什么 | PyTorch TP 的两块地基 |
| 3 | Colwise（列切）原理 + ASCII 图 | 输入复制 → 输出分片 |
| 4 | Rowwise（行切）原理 + ASCII 图 | 输入分片 → 输出复制（需 all-reduce） |
| 5 | PairwiseParallel（Megatron 范式） | 为什么 Col→Row 配对一次通信 |
| 6 | 输入/输出布局准备 API | `make_input_replicate_1d` 等真实接口 |
| 实操 | 命令 / 代码 / 配置 | 原文 `ToyModel` 片段 + 跑法 |
| 坑 | 常见问题表 | 偶数层、布局不匹配、通信开销 |

---

## 0. 一句话锚点

**张量并行（Tensor Parallelism, TP）= 把一层内部的大权重矩阵切成几块，分给几张卡，每张卡只算自己那块，再用一次集合通信（all-gather / all-reduce）把结果拼回去。**

- 对照「数据并行 DP」：DP 是**每张卡都有完整模型**、切的是数据 batch；TP 是**每张卡只有部分权重**、切的是模型参数本身。
- 对照「流水并行 PP」：PP 把**不同层**放到不同卡（层间切）；TP 把**同一层**切到多卡（层内切）。
- 关键收益：单层权重显存、单层激活计算被 `N` 张卡分摊到约 `1/N`。

> 原文一句核心：**Tensor Parallelism(TP) 建立在 DistributedTensor(DTensor) 之上，并提供多种并行格式：Rowwise、Colwise 和 Pairwise Parallelism。**

---

## 1. 地基：从单卡到多卡的显存账（原文两张表的真实含义）

原文给了两张真实的 `nvidia-smi` 进程显存截图，它们正是 TP 「把负载摊到多卡」的直接证据。

**单卡（不并行）：一个进程，独占一张卡，吃 3820MiB。**

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   1521759      C   /opt/conda/bin/python            3820MiB |
+-----------------------------------------------------------------------------+
```

**4 卡张量并行：起 4 个进程（每卡一个），每张卡只占 ~1370–1382MiB。**

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   4112660      C   /opt/conda/bin/python            1382MiB |
|    1   N/A  N/A   4112661      C   /opt/conda/bin/python            1382MiB |
|    2   N/A  N/A   4112662      C   /opt/conda/bin/python            1370MiB |
|    3   N/A  N/A   4112663      C   /opt/conda/bin/python            1370MiB |
+-----------------------------------------------------------------------------+
```

**怎么读这两张表（为什么这是 TP 的证据）：**

- 单卡 1 个 PID 吃 3820MiB；4 卡是 **4 个 PID**（4112660~4112663），每张卡只吃 ~1.37GiB。
- 注意：**不是 3820 ÷ 4 = 955**。每卡 1.37GiB 里包含两部分：
  1. **被切分、随卡数缩小的部分**（权重分片、对应激活）≈ 随 `N` 下降；
  2. **固定开销**（CUDA context、NCCL 通信 buffer、PyTorch/cuDNN 库占用）≈ 每卡都有一份、不随 `N` 缩小，约几百 MiB。
- 这解释了「为什么 4 卡总占用（~5.5GiB）反而大于单卡（3.82GiB）」：TP 省的是**单卡峰值**（让放不下的模型放得下），不是**总显存**。每多一张卡就多一份固定开销。

> 数值直觉：设单层权重显存 `W=2.4GiB`，固定开销 `C=0.55GiB`。
> 单卡 ≈ `W + C = 2.95`（加上别的杂项凑到 3.82）。
> 4 卡每卡 ≈ `W/4 + C = 0.6 + 0.55 = 1.15`（加杂项凑到 ~1.37）。
> ⇒ **单卡峰值降下来了，但总量上升**，这正是 TP 的取舍本质。

---

## 2. 两块地基：DTensor 与 DeviceMesh

PyTorch 原生 TP 不是一套独立框架，而是构建在两个底层抽象之上。原文末尾点名了它们：**「DeviceMesh 设备网格」**、**「Tensor Parallelism 建立在 DistributedTensor(DTensor) 之上」**。

### 2.1 DTensor（Distributed Tensor，分布式张量）

把它理解为：**一个逻辑上完整、物理上分散在多卡的张量，外加一份「布局说明书（placement）」。**

三种核心布局（placement）：

| 布局 | 英文 | 含义 | 显存 |
| --- | --- | --- | --- |
| 复制 | `Replicate` | 每张卡都有**整份**相同数据 | 每卡 = 全量 |
| 分片 | `Shard(dim)` | 沿某维度切，每卡只有**一片** | 每卡 = 全量/N |
| 部分 | `Partial` | 每卡是部分和，需 all-reduce 才完整 | 每卡 = 全量 |

原文对 TP 风格的描述，本质就是在说「输入是哪种 placement、输出变成哪种 placement」：

- 原文：「**Rowwise，对模块的行进行分区。假设输入是分片的 DTensor，则输出是仿制的 DTensor。**」→ 输入 `Shard` → 输出 `Replicate`。
- 原文：「**Colwise，对张量或模块的列进行分区。假设输入是仿制的 DTensor，则输出是分片的 DTensor。**」→ 输入 `Replicate` → 输出 `Shard`。

（注：「仿制」是机翻，应理解为 **replicate=复制**。）

### 2.2 DeviceMesh（设备网格）

**DeviceMesh = 把一堆 GPU 排成一个 N 维网格，告诉框架「哪几张卡组成一个通信组」。**

```
8 卡，排成 2D mesh = [[0,1,2,3],
                     [4,5,6,7]]

         TP 维度 (列方向, 组内做张量并行) →
        +----+----+----+----+
DP 维度 | 0  | 1  | 2  | 3  |   ← 这 4 张卡一个 TP 组
(行)    +----+----+----+----+
 ↓      | 4  | 5  | 6  | 7  |   ← 这 4 张卡另一个 TP 组
        +----+----+----+----+
```

为什么需要它：TP 的 all-gather / all-reduce **只能在「同一个 TP 组」内做**。DeviceMesh 就是定义这个组的工具，也是 2D/3D 混合并行（DP×TP×PP）的坐标系。原文里 `make_input_replicate_1d(input, device_mesh=None)` 的第二个参数就是它。

---

## 3. Colwise（列切）：输入复制 → 输出分片

以一个线性层 `Y = X · A`（A 是权重）为例，A 沿**列**切成 `[A₁ | A₂]`。

```
         X (每卡都有完整 X, Replicate)
          |
   +------+------+
   |             |
 GPU0          GPU1
 X·A₁          X·A₂        ← 每卡算一半列，无需通信！
   |             |
 Y₁=[列前半]   Y₂=[列后半]   ← 输出天然是 Shard(沿列)
```

- **数学**：`X·[A₁|A₂] = [X·A₁ | X·A₂]`。列切后两块独立、**前向无通信**。
- **输入要求**：X 必须复制（每卡都要完整 X），对应原文「假设输入是**仿制的** DTensor」。
- **输出**：分片（`Y₁`/`Y₂` 各持一半列），对应原文「输出是**分片的** DTensor」。
- **数值手算**：X 是 `[1,2]`，A=`[[1,3],[2,4]]`，列切 A₁=`[[1],[2]]`、A₂=`[[3],[4]]`。GPU0 算 `1·1+2·2=5`，GPU1 算 `1·3+2·4=11`，拼起来 `Y=[5,11]` ✓。

## 4. Rowwise（行切）：输入分片 → 输出复制（需一次 all-reduce）

A 沿**行**切成 `[A₁; A₂]`（上下两块）。

```
 X₁(左半)      X₂(右半)      ← 输入本身已分片 Shard
   |             |
 GPU0          GPU1
 X₁·A₁         X₂·A₂        ← 各算部分积(Partial)
   |             |
   +------+------+
          |
     all-reduce(求和)        ← 必须通信一次！
          |
        Y(完整, Replicate)
```

- **数学**：`X·A = [X₁|X₂]·[A₁;A₂] = X₁·A₁ + X₂·A₂`。每卡只得到一个**部分和**（`Partial`），必须 **all-reduce 求和**才得到完整 Y。
- **输入要求**：分片，对应原文「假设输入是**分片的** DTensor」。
- **输出**：复制（all-reduce 后每卡都有完整 Y），对应原文「输出是**仿制的** DTensor」。
- **代价**：比 Colwise 多一次 all-reduce 通信。这就是下一节为什么要把 Col 和 Row **配对**。

## 5. PairwiseParallel：Megatron-LM 的经典配对（Col→Row）

原文：「**PairwiseParallel 将 colwise 和 rowwise 样式串联为固定对，就像 [Megatron-LM](https://arxiv.org/abs/1909.08053) 所做的那样。我们假设输入和输出都需要复制 DTensor。**」

**为什么是「Col 在前、Row 在后」这个固定顺序？** 因为它能把两层（如 MLP 的两个线性层、Attention 的 QKV+输出投影）之间的通信**省掉一半**：

```
 输入 X (Replicate)
     │
 ┌───▼──────────┐
 │ Linear1 列切  │  Colwise: 输入复制→输出分片, 前向无通信
 └───┬──────────┘
     │  中间激活天然分片 (Shard) —— 这里不用通信对齐!
 ┌───▼──────────┐
 │ Linear2 行切  │  Rowwise: 输入分片→输出复制
 └───┬──────────┘
     │  ← 仅在这里做 1 次 all-reduce
 输出 Y (Replicate)
```

- Col 的输出（分片）**正好**是 Row 需要的输入（分片）→ 中间**不需要任何通信**对齐布局。
- 整个 Col→Row 块**前向只需 1 次 all-reduce**（反向也只需 1 次 all-reduce），这正是 Megatron-LM 的核心技巧。
- **输入/输出都是复制**：所以这个「块」对外表现得像一个普通层，可以无缝串联多个块。

原文限制：「**PairwiseParallel 目前仅支持 `nn.MultiheadAttention`、`nn.Transformer` 或偶数层 `MLP`。**」

> 为什么必须**偶数层** MLP？因为要 Col、Row 成对出现：第 1 层 Col、第 2 层 Row、第 3 层 Col……层数为奇数就会剩一个无法配对的层，破坏「最后输出复制」的不变式。

| 风格 | 切分方向 | 输入布局 | 输出布局 | 前向通信 |
| --- | --- | --- | --- | --- |
| Colwise | 列 | Replicate | Shard | 无 |
| Rowwise | 行 | Shard | Replicate | 1× all-reduce |
| **PairwiseParallel** | Col→Row | Replicate | Replicate | 1× all-reduce / 对 |

---

## 6. 输入/输出布局准备 API（原文真实接口）

原文：「**由于 Tensor Parallelism 是建立在 DTensor 之上的，因此我们需要使用 DTensor 指定模块的输入和输出位置，以便它可以在前后与模块进行预期的交互。**」

给出的真实函数：

```python
torch.distributed.tensor.parallel.style.make_input_replicate_1d(input, device_mesh=None)
```

- **作用**：把一个普通张量 / DTensor 转成「**复制（Replicate）**」布局的 1D DTensor，让它符合 Colwise 风格对「输入要复制」的要求。
- **参数 `device_mesh`**：第 2 节讲的 DeviceMesh，指定在哪一组卡上复制；`None` 时用当前默认 mesh。
- **配套理解**：还存在对偶的 `make_input_shard_1d`（把输入转成分片，喂给 Rowwise 风格）。「准备函数」的职责就是**让上一层的输出布局对齐下一层风格所要求的输入布局**——这正是第 5 节配对能省通信的前提。

> 提示：上面这些 `torch.distributed.tensor.parallel` 接口在 PyTorch 演进中已逐步并入 `parallelize_module` + `ColwiseParallel`/`RowwiseParallel` 的新风格 API。本笔记**按原文给出的接口名原样保留**，实际编码请以你本机 PyTorch 版本的官方文档为准。

---

## 实操：命令 / 代码 / 配置（保留原文真料）

### 第一步：单卡放一个玩具模型（原文片段）

```python
from utils import cleanup, setup, ToyModel
model = ToyModel().cuda(0)
model
```

这就是第 1 节「单卡 3820MiB」那张表对应的最小起点：把 `ToyModel` 放到 `cuda:0` 上，单进程独占一张卡。

### 第二步：起多进程做张量并行

PyTorch 多卡 TP 是「**每张卡一个进程**」模型（对应第 1 节那 4 个 PID）。典型启动方式：

```bash
# 4 卡 = 4 进程，每进程绑定一张卡
torchrun --nproc_per_node=4 train_tp.py
```

每个进程内部的标准套路（与原文 `utils` 的 `setup/cleanup` 呼应）：

```python
setup(rank, world_size)          # 初始化进程组 (NCCL backend)
# 1) 建 DeviceMesh: 把 4 张卡组成一个 1D TP mesh
# 2) parallelize: 对 ToyModel 应用 Colwise/Rowwise/Pairwise 风格
# 3) 用 make_input_replicate_1d 准备输入布局
# 4) 正常 forward/backward, 框架自动插入 all-gather/all-reduce
cleanup()                        # 销毁进程组
```

### 验证：用 `nvidia-smi` 看显存被摊开

跑起来后 `nvidia-smi`，应能看到第 1 节那种**多 PID、每卡显存约为单卡分片**的格局，即 TP 生效的直接证据。

---

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 对策 |
| --- | --- | --- |
| 4 卡总显存反而比单卡大 | 每卡都有一份固定开销（CUDA context/NCCL buffer） | 正常现象；TP 省的是**单卡峰值**不是总量 |
| 用 PairwiseParallel 报错 / 行为异常 | MLP 层数为**奇数**，Col/Row 无法配对 | 改成**偶数层** MLP（原文限制） |
| 只支持部分模块 | Pairwise 仅支持 `nn.MultiheadAttention`/`nn.Transformer`/偶数层 MLP | 自定义结构需手动逐层指定 Col/Row 风格 |
| 形状/布局不匹配报错 | 上一层输出布局 ≠ 下一层风格要求的输入布局 | 用 `make_input_replicate_1d`/`*_shard_1d` 显式对齐 |
| Colwise 输入忘了复制 | 输入是 Shard 而非 Replicate | 喂入前先 `make_input_replicate_1d` |
| 卡数越多越不划算 | all-reduce 通信量随 N 上升，固定开销 ×N | TP 一般限在**单机内**（NVLink 高带宽），跨机用 DP/PP |
| 单卡也想跑通调试 | TP 依赖进程组 | 先用原文 `ToyModel().cuda(0)` 单卡验证逻辑，再上多卡 |

---

## 🔗 跳转链接

**枢纽**：[[00-知识地图]] · [[llm-train/README]] · [[llm-train/pytorch/distribution/README]]

**框架对照**：[[ai-framework/megatron-lm/README]]（Pairwise 范式的来源）· [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]] · [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]

**通信底座**：[[ai-infra/网络/集合通信原语]]（all-reduce / all-gather）· [[ai-infra/网络/NCCL]]

**模型结构**：[[llm-algo/transformer/模型架构]]（QKV/MLP 哪里被切）· [[B07:llm-inference/大模型推理张量并行]]（推理侧 TP）

**微调 / 对齐周边**：[[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-alignment/RLHF]] · [[llm-compression/quantization/量化基础]]
