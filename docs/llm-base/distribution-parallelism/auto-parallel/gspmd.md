# GSPMD：基于分片标注的通用自动并行

> 一句话定位：GSPMD 是 Google 提出的"只标注少量张量的分片方式、编译器自动补全整图分片并插入通信"的自动并行系统，是 SPMD（单程序多数据）范式在 ML 计算图上的通用化实现。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[auto-parallel/分布式训练自动并行概述]] · [[auto-parallel/Alpa]] · [[auto-parallel/Mesh-Tensorflow]] · [[tensor-parallel/tensor-parallel]] · [[ai-infra/网络/集合通信原语]] · [[llm-inference/大模型推理张量并行]]

---

## 阅读地图

| 节 | 内容 | 你将学到 |
|----|------|----------|
| 0 | 一句话锚点 | GSPMD 到底解决什么问题 |
| 1 | 地基：SPMD / Mesh / Sharding | 没有这些概念读不懂后文 |
| 2 | 分片标注 API：`mesh_split` | 用户只写几行 annotation |
| 3 | 分片传播（Sharding Propagation） | 编译器如何"补全"整图分片 |
| 4 | 通信插入：reshard 与集合通信 | 何时插 AllReduce/AllGather/AllToAll |
| 5 | 三大并行如何用分片表达 | DP/TP/PP 统一成 sharding |
| 6 | 算子级分片规则（matmul 为例） | 每个算子的分片怎么推 |
| 7 | 与 Megatron / Alpa / FSDP 对比 | 自动 vs 手工 |
| 数值手算 | 1.4B Transformer 在 8 卡上的通信/显存 | 逐数手算 |
| 常见问题 | 易错点表格 | 避坑 |

---

## 0. 一句话锚点

**手工并行**（如 Megatron-LM）需要工程师把每个矩阵乘法手动拆开、手动插入 `AllReduce`，改一层模型就要改一遍并行代码。

**GSPMD 的核心主张**：把"并行"降维成一个纯粹的**张量分片（sharding）标注问题**——

```
用户只标注：W 这个权重沿第 0 维切到 8 个设备上
   ↓ 编译器（XLA）自动做三件事
1. 分片传播：推断出所有中间张量、所有算子输入输出的分片
2. 通信插入：在分片不匹配处自动插入 AllGather/AllReduce/AllToAll
3. 单程序生成：每个设备跑同一份程序（SPMD），只处理自己那一片数据
```

一句话：**GSPMD = 分片标注 + 分片传播 + 自动通信插入，跑在 XLA 编译器里的 SPMD 划分器。**

---

## 1. 地基：SPMD / Mesh / Sharding 三个原子概念

### 1.1 SPMD（Single Program Multiple Data，单程序多数据）

与 MPMD（每个设备跑不同程序，如流水线并行的不同 stage）相对：

```
   MPMD（如 GPipe 手写）              SPMD（GSPMD）
┌─────────┐ ┌─────────┐        ┌─────────┐ ┌─────────┐
│设备0:    │ │设备1:    │        │设备0:    │ │设备1:    │
│ layer0-3│ │ layer4-7│        │ 同一份   │ │ 同一份   │
│ (程序A) │ │ (程序B) │        │ 程序     │ │ 程序     │
└─────────┘ └─────────┘        │ 切片A数据│ │ 切片B数据│
 程序不同，难扩展                 └─────────┘ └─────────┘
                                程序相同，规模无关
```

SPMD 的好处：**编译时间和设备数解耦**。1 个设备和 1 万个设备编译出来的程序是同一份，编译只做一次。这是 GSPMD 能扩展到数千 TPU 核的关键。

### 1.2 Device Mesh（设备网格）

把物理设备组织成一个**逻辑多维网格**。例如 8 张卡可以排成：

```
1D mesh: [d0 d1 d2 d3 d4 d5 d6 d7]              shape=(8,)

2D mesh:  ┌────┬────┬────┬────┐
   axis  │ d0 │ d1 │ d2 │ d3 │   ← "model" 轴 (size 4)
   "data"├────┼────┼────┼────┤
   (2)   │ d4 │ d5 │ d6 │ d7 │
          └────┴────┴────┴────┘   shape=(2, 4)
              ↑ "model" 轴
```

网格的每个**命名轴**（如 `data`、`model`）后面会被绑定到张量的某个维度上，从而表达"这个张量沿哪个网格轴切分"。

### 1.3 Sharding（分片）：张量的每个维度映射到 mesh 轴

一个张量分片的描述 = **每个张量维度 → 映射到哪个 mesh 轴（或 replicated / partial）**。三种状态：

```
张量 X 形状 [B, H]，mesh 轴 = {data, model}

① Sharded（切分）  : X 的 B 维沿 data 轴切 → 每卡持有 [B/2, H]
② Replicated（复制）: X 在 model 轴上不切 → 每卡持有完整 [B, H] 副本
③ Partial（部分和）: X 是各卡局部和，逻辑值 = 跨某轴 AllReduce 后才正确
                     （matmul 沿收缩维切分后的典型中间态）
```

`Partial` 是 GSPMD 区别于早期系统的精髓：它允许中间张量先保持"未归约的局部和"状态，把 `AllReduce` 延迟到真正需要完整值时才插入，从而合并通信、减少冗余。

---

## 2. 分片标注 API：`mesh_split` / `annotate`

GSPMD 暴露给用户的 API 极简，核心就一个意图："把张量 T 的第 i 维放到 mesh 的第 d 轴上"。伪代码：

```python
mesh = Mesh(devices=8, axis_names=("data", "model"), shape=(2, 4))

# 标注：激活 x 的 batch 维沿 data 轴切，权重 w 的输出维沿 model 轴切
x = mesh_split(x, mesh, dims_mapping=[0, -1])   # [B→data, H→不切]
w = mesh_split(w, mesh, dims_mapping=[-1, 1])   # [H→不切, O→model]
y = matmul(x, w)   # y 的分片由编译器自动推断，无需标注
```

要点：
- 用户**只标注"边界张量"**（输入、权重、loss 等少数几个），整图其余分片由编译器传播。
- 标注是一个 **identity 算子**（不改变数值，只附加分片元信息），编译期被消解。
- `dims_mapping[i] = d` 表示张量第 `i` 维沿 mesh 第 `d` 轴切；`-1`（或 `None`）表示该维 replicated。

```
  用户视角                          编译器视角
┌──────────────┐                ┌──────────────────────┐
│ 标注 x, w     │  ── 喂给 ──▶  │ XLA SPMD Partitioner │
│ （2~3 行）    │                │  传播 + 插通信 + 生成 │
└──────────────┘                └──────────────────────┘
                                          │
                                   每卡一份 SPMD 程序
```

---

## 3. 分片传播（Sharding Propagation）：编译器如何补全整图

这是 GSPMD 的算法核心。给定少量已标注张量，如何推断**所有**张量的分片？

### 3.1 传播是一个"在计算图上做约束求解"的过程

```
   标注的张量 = 已知分片（种子）
        │
        ▼  沿数据依赖边双向传播
  ┌─────────────────────────────────┐
  │ 前向传播：输入分片 → 推输出分片   │
  │ 反向传播：输出分片 → 推输入分片   │
  └─────────────────────────────────┘
        │ 反复迭代直到不动点（fixed point）
        ▼
   每个张量都被赋予一个分片
```

每个算子带一组**分片规则**：给定输入分片，输出该是什么分片；或给定输出分片，输入该是什么分片。传播就是按拓扑序（及反向）把这些规则连起来求解。

### 3.2 冲突与优先级

当一个张量被多条路径推出不同分片时，需要决策：

```
   path A 说 T 沿 data 切          path B 说 T replicated
            \                         /
             ▼                       ▼
          冲突！按优先级排序：
          ① 用户显式标注 > 传播得到的
          ② 切分（节省内存） > 复制
          ③ 更少 reshard 通信代价者优先
```

GSPMD 采用启发式优先级（用户标注优先、倾向保留切分、最小化通信），并通过迭代直到稳定。这就是论文标题中"General"的来源——规则与传播机制对算子和模型结构通用。

---

## 4. 通信插入：reshard 与集合通信原语

当某算子要求的输入分片与上游实际产出的分片**不一致**时，编译器插入一次 **reshard**（重分片），底层落到一个集合通信原语。映射关系：

```
 从 → 到                        插入的集合通信       语义
─────────────────────────────────────────────────────────
 Sharded   → Replicated        AllGather           收集各片拼全
 Partial   → Replicated        AllReduce           各卡局部和求总和
 Sharded(轴a) → Sharded(轴b)   AllToAll            重新打散维度
 Replicated → Sharded          (本地切片，无通信)   直接 slice
 Partial   → Sharded           ReduceScatter       求和并切分
```

ASCII 示意 `Partial → Replicated`（matmul 沿收缩维切分后必经的一步）：

```
收缩维 K 被切到 4 卡，各卡算出局部和 y_p（形状对但值不全）：

  d0: y0(partial)  d1: y1(partial)  d2: y2(partial)  d3: y3(partial)
        \              |                |              /
         └────────── AllReduce(sum) ─────────────────┘
                          ▼
        每卡得到 y = y0+y1+y2+y3  （Replicated，值正确）
```

GSPMD 的优化点：尽量让张量**保持 Partial 状态向后流动**，把多次 AllReduce 合并，或下沉到代价更低的位置（如和后续 ReduceScatter 融合成一次通信）。

---

## 5. 三大并行如何统一成"分片"

GSPMD 的威力在于：DP / TP / PP（及 ZeRO/FSDP）全都退化成 mesh 轴 + dims_mapping 的不同写法。

```
数据并行 DP   : 激活 x 的 batch 维 → 切到 data 轴；权重 W → replicated
              （反向时权重梯度是 Partial，AllReduce 求和）

张量并行 TP   : 权重 W 的某维 → 切到 model 轴；激活按 Megatron 列/行切法
              （前向后向各一次 AllReduce / AllGather）

全分片 FSDP   : 权重 W 也沿 data 轴切（参数、梯度、优化器态都切）
              （用前临时 AllGather 出全权重，用完即弃 → 省显存）

流水线 PP     : 严格说是 MPMD，GSPMD 主攻 SPMD 的 DP+TP，
              PP 通常由外层框架（GPipe/Megatron）配合，二者正交叠加
```

二维网格同时表达 DP + TP（这正是 Megatron-LM 的混合并行，但 GSPMD 用标注自动得到）：

```
mesh (2, 4): data 轴=2 做数据并行, model 轴=4 做张量并行

           model 轴（张量并行 4 路）
          ┌──────┬──────┬──────┬──────┐
 data 轴  │ d0   │ d1   │ d2   │ d3   │  data 组 0
 (DP 2路) ├──────┼──────┼──────┼──────┤
          │ d4   │ d5   │ d6   │ d7   │  data 组 1
          └──────┴──────┴──────┴──────┘
 → batch 沿 data 轴切；每个 W 沿 model 轴切；编译器自动插两类通信
```

---

## 6. 算子级分片规则：以 matmul 为例（最重要的算子）

矩阵乘 $Y_{[B,O]} = X_{[B,H]} \cdot W_{[H,O]}$ 有三个维度：batch 维 $B$、收缩维 $H$、输出维 $O$。沿不同维切对应不同并行：

### 6.1 沿 batch 维 B 切（= 数据并行）

```
X:[B/p, H] 切    W:[H, O] 复制    →    Y:[B/p, O] 切
每卡独立算自己 batch 子集，前向无通信；
反向时 dW 是各卡局部和 → Partial → AllReduce
```
$$\text{通信量(反向)}=2\cdot H\cdot O\cdot(p-1)/p \ \text{（AllReduce 权重梯度）}$$

### 6.2 沿输出维 O 切（= 列并行 Column-Parallel）

```
X:[B,H] 复制    W:[H, O/p] 切    →    Y:[B, O/p] 切
前向无需通信（各卡产出 Y 的不同列）；
若下游要完整 Y → AllGather
```

### 6.3 沿收缩维 H 切（= 行并行 Row-Parallel）

```
X:[B, H/p] 切    W:[H/p, O] 切    →    Y:[B,O] 是 Partial（局部和）
前向需 AllReduce 求和才得正确 Y
```
$$\text{通信量(前向)}=2\cdot B\cdot O\cdot(p-1)/p \ \text{（AllReduce 输出）}$$

Megatron 的"列并行 + 行并行"夹心结构（MLP 的 $W_1$ 列切、$W_2$ 行切，使中间 AllGather 省掉，只在最后一次 AllReduce），在 GSPMD 里就是**让中间张量保持 Sharded(O 轴)直接喂给行并行**，传播器自动只在末端插一次 AllReduce——这正是 Partial 延迟归约的价值。

```
 Megatron MLP（GSPMD 自动复现）：
 X ─列切W1→ H1[B, O/p]（切）─GELU(逐元素,不通信)→ ─行切W2→ Y[B,O](Partial)
                                                          │ AllReduce ×1
                                                          ▼ 正确 Y
 全程仅 1 次 AllReduce（前向），不需要中间 AllGather
```

---

## 7. 与 Megatron / Alpa / FSDP 的关系

| 系统 | 并行划分方式 | 自动化程度 | 关键差异 |
|------|--------------|-----------|----------|
| Megatron-LM | 手工改写算子、手插通信 | 低（人写） | 性能可控但不通用，换模型要重写 |
| GSPMD | 标注少量张量 + 编译器传播 | 中（标注+自动补全） | 通用、可组合，但分片策略仍需人给种子 |
| Alpa | 全自动搜索（ILP）层内 + DP 层间 | 高（自动找策略） | 连"切哪一维"都搜，GSPMD 默认靠人标注 |
| FSDP/ZeRO | 沿 data 轴切参数/梯度/优化器态 | 中 | 是 GSPMD 一种特例（W 也 sharded） |

记忆锚：**GSPMD 提供"机制"（如何把分片落地为通信与 SPMD 程序），Alpa 在 GSPMD 之上提供"策略"（自动决定怎么分片）**。二者互补——Alpa 的后端正是用类 GSPMD 的 SPMD 划分能力。

详见 [[auto-parallel/Alpa]] 与 [[auto-parallel/分布式训练自动并行概述]]。

---

## 数值手算：1.4B 类 GPT 单层 MLP 在 8 卡上的通信与显存

设定（取整便于手算）：
- 隐藏维 $H=2048$，MLP 扩展 4 倍即中间维 $4H=8192$，序列×批 token 数 $T = B\cdot S = 8192$
- mesh = (data=1, model=8)，即纯 8 路张量并行，单精度先按 fp16（2 字节）算
- MLP：$W_1\in\mathbb{R}^{H\times 4H}$ 列切，$W_2\in\mathbb{R}^{4H\times H}$ 行切

**① 参数量与每卡权重显存**

单层 MLP 参数：$W_1 + W_2 = H\cdot 4H + 4H\cdot H = 2\times(2048\times 8192) = 3.36\times10^7 \approx 33.6\text{M}$。

8 路张量并行后每卡持有 $1/8$：
$$33.6\text{M}/8 = 4.2\text{M 参数} \times 2\text{B(fp16)} = 8.4\text{MB / 卡}$$
（对比不切：33.6M × 2B = 67.2MB / 卡，省 8×）

**② 前向激活通信量（Megatron 风格只 1 次 AllReduce）**

行并行 $W_2$ 输出 $Y\in\mathbb{R}^{T\times H}$ 是 Partial，需 AllReduce。
AllReduce 单卡收发量 $\approx 2\cdot\dfrac{(p-1)}{p}\cdot(\text{张量大小})$（Ring-AllReduce）：

张量大小 $= T\cdot H = 8192\times 2048 = 1.68\times10^7$ 元素 $\times 2\text{B} = 33.6\text{MB}$。

$$\text{每卡通信量} = 2\times\frac{8-1}{8}\times 33.6\text{MB} = 2\times0.875\times33.6 \approx 58.8\text{MB}$$

**③ 反向再来一次对称的 AllReduce（梯度 / 激活）**

反向同样约 $58.8\text{MB/卡}$。单层 MLP 前向+反向合计 $\approx 117.6\text{MB/卡}$ 的张量并行通信。

**④ 通信时间估算（带宽 200 GB/s，如 NVLink 量级）**

$$t \approx \frac{117.6\text{MB}}{200\text{GB/s}} = \frac{0.1176\text{GB}}{200\text{GB/s}} \approx 0.59\text{ms / 层 / step}$$

24 层模型 $\Rightarrow 24\times0.59 \approx 14\text{ms}$ 纯张量并行通信开销/步。**结论**：张量并行只在卡间带宽足够高（NVLink/TPU ICI）时划算；跨节点（PCIe/以太网带宽低一个量级）则该开销暴涨，这就是 TP 通常限制在单机内、跨机改用 DP/PP 的根因。

**⑤ 切收缩维 vs 切 batch 维 的通信对比（直觉校验）**

- 切收缩维（行并行）：前向就要 AllReduce 输出 $T\cdot H$（与激活大小成正比）。
- 切 batch 维（数据并行）：前向零通信，反向 AllReduce 梯度 $\propto$ 参数量 $33.6\text{M}$。

当 **激活体量 > 参数体量** 时数据并行更省通信；反之张量并行更省。GSPMD 的传播器/Alpa 的搜索器本质就是在为每个算子做这类权衡——这也是"自动并行"要解决的核心数学问题。

---

## 常见问题

| 问题 | 答案 |
|------|------|
| GSPMD 是框架还是编译器 pass？ | 是 XLA 编译器中的 SPMD 划分器（partitioner），TensorFlow/JAX 都可用其能力。 |
| 用户要标注多少张量？ | 只需少数"种子"张量（输入/权重/loss），其余靠分片传播自动补全。 |
| Partial 状态有什么用？ | 让 AllReduce 延迟/合并，避免每个收缩维切分都立刻全归约，显著降低通信。 |
| 它能做流水线并行吗？ | PP 本质是 MPMD，GSPMD 专攻 SPMD 的 DP+TP+FSDP；PP 通常由外层框架叠加，二者正交。 |
| 和 FSDP 什么关系？ | FSDP 是"权重也沿 data 轴 sharded"的一个 GSPMD 特例。 |
| 分片传播会失败吗？ | 标注不足或冲突时可能推出次优分片（多余 reshard），需用户补标注或调优先级。 |
| 通信原语具体有哪些？ | AllGather / AllReduce / ReduceScatter / AllToAll，详见 [[ai-infra/网络/集合通信原语]]。 |
| 为什么能扩到数千 TPU？ | SPMD 使编译与设备数解耦，编译只做一次，规模无关。 |
| 它替代 Megatron 吗？ | 不完全。GSPMD 提供通用机制，Megatron 提供手调极致性能；二者思路在现代框架中常融合。 |
| 版本/API 细节？ | 具体 API 名称与默认值以官方 XLA / JAX 文档为准。 |

---

## 🔗 跳转链接

- 导航总入口：[[00-知识地图]]
- 自动并行概览：[[auto-parallel/分布式训练自动并行概述]] · [[auto-parallel/auto-parallel]]
- 自动并行同类系统：[[auto-parallel/Alpa]] · [[auto-parallel/Mesh-Tensorflow]] · [[auto-parallel/Flexflow]] · [[auto-parallel/Galvatron]] · [[auto-parallel/Unity]]
- 飞桨自动并行实践：[[auto-parallel/飞桨面向异构场景下的自动并行设计与实践]]
- 手工张量并行对照：[[tensor-parallel/tensor-parallel]] · [[llm-inference/大模型推理张量并行]]
- 底层通信：[[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]]
- 模型与算力地基：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/FLOPs]] · [[ai-infra/算力/GPU工作原理]]
- 训练总览：[[llm-train/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
