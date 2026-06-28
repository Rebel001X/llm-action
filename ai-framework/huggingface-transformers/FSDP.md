# FSDP (完全分片数据并行)

> FSDP = Fully Sharded Data Parallel：把"参数 / 梯度 / 优化器状态"沿数据并行维度切碎，每张卡只常驻 1/N，用时临时 all-gather，把"显存"换成"通信"，从而在普通 GPU 集群上训练远超单卡容量的大模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/deepspeed/README]] [[ai-framework/pytorch/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | 分片=切碎+按需聚合 |
| 1 | 地基：DDP 为什么会爆显存 | 复制 vs 分片、ZeRO 三级 |
| 2 | 分片到底切了什么 | 参数 P / 梯度 G / 优化器 O |
| 3 | 一层的完整生命周期 | all-gather → 算 → reduce-scatter |
| 4 | 两个核心集合通信原语 | all-gather / reduce-scatter |
| 5 | wrap 策略：以谁为单位分片 | unit、auto_wrap_policy |
| 6 | sharding_strategy 三档 | FULL / SHARD_GRAD_OP / HYBRID |
| 7 | 与 DDP / ZeRO / TP 对比 | 同与不同 |
| 8 | 显存账 + 通信账（手算） | 16 字节法则、7B 实例 |
| 9 | 实践要点与坑 | 混精、offload、ckpt |
| QA | 常见问题速查 | 表格 |

---

## 0. 一句话锚点

**DDP 的世界观**："每张卡都存一份完整模型，各算各的，最后把梯度对齐。" → 简单，但**每张卡的显存 = 单卡装满整个模型的状态**，模型一大就爆。

**FSDP 的世界观**："模型状态本来就太大，那就**别让任何一张卡存全量**。每张卡只保管 1/N 的碎片；真要用某一层时，临时向所有卡借齐这一层、算完立刻还回去。" → 用**多一点通信**换来**每张卡显存近似除以 N**。

一句话：**FSDP 是 PyTorch 原生实现的 ZeRO-3**。把"复制"改成"分片 + 按需聚合"。

---

## 1. 地基/前置

### 1.1 训练时显存到底被谁吃掉

不假设你记得，先把"训练一个模型要存哪些东西"拆到最原子。设模型参数量为 $\Psi$（个数），用**混合精度 + Adam** 训练，每一项的字节占用：

| 状态 | 内容 | 精度 | 字节/参数 |
|---|---|---|---|
| 参数 (fp16) | 前向/反向用的权重副本 | fp16 | $2\Psi$ |
| 梯度 (fp16) | 反向算出的梯度 | fp16 | $2\Psi$ |
| 参数 master (fp32) | 优化器维护的高精度主权重 | fp32 | $4\Psi$ |
| 动量 m (fp32) | Adam 一阶矩 | fp32 | $4\Psi$ |
| 方差 v (fp32) | Adam 二阶矩 | fp32 | $4\Psi$ |

合计 **$16\Psi$ 字节**（著名的"16 字节法则"）。这还**不含激活值**（前向中间结果，反向要用）。

- 7B 模型：$16 \times 7\times10^9 \approx 112\,\text{GB}$ —— 一张 80GB 的 A100 **单卡都放不下**，连推理之外的训练根本无从谈起。

### 1.2 DDP 的复制 = 浪费

DDP（DistributedDataParallel）下，**N 张卡各存一份完整的这 $16\Psi$**。8 卡训 7B，集群里躺着 8 份一模一样的 112GB —— 887GB 显存里有 7/8 是冗余复制。FSDP 的出发点就是：**把这份冗余切掉**。

### 1.3 ZeRO 三级：FSDP 的理论母体

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer）按"切哪三样"分三级，理解了这个就理解了 FSDP：

```
        每卡显存占用（N 卡，混精 Adam）
DDP   :  16Ψ                （全复制）
ZeRO-1:  4Ψ + 12Ψ/N        （只切优化器状态 O）
ZeRO-2:  2Ψ + 14Ψ/N        （切 O + 梯度 G）
ZeRO-3:  16Ψ/N             （切 O + G + 参数 P  ← FSDP 对应这一级）
```

**FSDP 默认对标 ZeRO-3**：三样全切，每卡显存随 N 线性下降。详细级别与 offload 见 [[ai-framework/deepspeed/README]]。

---

## 2. 分片切了什么 / 怎么切

### 2.1 三样东西都切

"完全分片"的"完全"指：**参数 P、梯度 G、优化器状态 O 三类全部分片**。设 N=4 卡，某一层有 12 个参数：

```
        参数 [p0 p1 p2 ... p11]   ←  逻辑上完整的一层
                  │  按 rank 平均切成 4 段（flatten 后等分）
   ┌──────────┬──────────┬──────────┬──────────┐
 rank0       rank1      rank2      rank3
[p0 p1 p2]  [p3 p4 p5] [p6 p7 p8] [p9 p10 p11]   ← 平时每卡只存自己这 3 个
   +O0        +O1        +O2        +O3            ← 对应的优化器状态也只存自己段
```

平时（idle 态）每张卡只持有 **1/N 的参数 + 1/N 的优化器状态 + 1/N 的梯度**。

### 2.2 "flatten 再切"：为什么不按层切而按字节切

FSDP 不是"把第 1 层给 rank0、第 2 层给 rank1"。它在一个 **wrap 单元（FSDP unit）** 内，把该单元所有参数**拉平成一维大向量（flatten）再等分**。好处：

- 切口均匀，每卡负载几乎相等（避免某层特别大导致不均）。
- 通信时是一段连续 buffer，集合通信效率高。

代价：参数被"打散"，所以 FSDP 内部维护 `FlatParameter` 来记录"原始 shape 如何从这段扁平 buffer 还原"。

---

## 3. 一层的完整生命周期（核心机制）

这是 FSDP 最关键的一节。**分片状态下怎么还能正常算前向反向？** 答案：**用之前临时聚齐，用完立即释放。**

### 3.1 前向：all-gather 聚齐 → 计算 → 释放

```
前向到达 Layer_k（此刻每卡只有 1/N 碎片）
        │
   ① all-gather：每张卡把自己的碎片广播，凑齐 Layer_k 的【完整参数】
        │     rank0:[p0p1p2] ─┐
        │     rank1:[p3p4p5] ─┼─► 每卡都得到 [p0..p11] 完整层（短暂占满显存）
        │     rank2:[p6p7p8] ─┤
        │     rank3:[p9p10p11]┘
        │
   ② compute：用完整 Layer_k 做前向，得到激活 → 传给 Layer_{k+1}
        │
   ③ free：立刻丢弃刚才聚齐的完整参数，只留回自己的 1/N 碎片
        │     （显存峰值 = 碎片 + "当前这一层"的完整参数，而非全模型）
        ▼
前向到达 Layer_{k+1}（重复 ①②③）
```

**关键洞察**：任意时刻，被"聚齐成完整"的只有**当前正在算的那一层**，不是整个模型。所以峰值显存 ≈ 全模型碎片(16Ψ/N) + 单层完整参数(很小)。

### 3.2 反向：all-gather 参数 → 算梯度 → reduce-scatter 梯度

反向同样需要完整参数（求导要用 W），所以**再 all-gather 一次**；算出该层完整梯度后，用 **reduce-scatter** 把梯度"求和并切回碎片"：

```
反向到达 Layer_k
        │
   ① all-gather 参数（前向已 free，需重新聚齐）
        │
   ② compute 梯度：得到 Layer_k 的【完整梯度】 g = [g0 g1 ... g11]（各卡对自己 batch 算的）
        │
   ③ reduce-scatter 梯度：跨卡【求和】+【切片】一步完成
        │     所有卡的 g 逐元素相加  →  再切成 N 段，第 i 段只发给 rank_i
        │     rank0 最终只拿到 Σg[0:3]，rank1 拿到 Σg[3:6] ...
        │
   ④ free 完整参数 & 完整梯度，只留自己那段【已聚合的梯度碎片】
        │
   ⑤ optimizer.step()：rank_i 只用本地碎片 (G_i, O_i) 更新本地参数碎片 P_i
        ▼                （无需任何额外通信，因为各管各的段）
```

**为什么用 reduce-scatter 而不是 all-reduce？**
DDP 用 all-reduce：每卡拿到**完整**的平均梯度（因为 DDP 每卡要更新完整参数）。FSDP 每卡只更新 1/N 参数，所以只需要**对应那 1/N 的梯度**。reduce-scatter = "求和 + 只把你负责的那段发给你"，刚好契合，**还省了一半梯度通信量**（见第 8 节通信账）。

---

## 4. 两个集合通信原语（拆到最底层）

FSDP 的全部魔法就建立在这两个原语上，必须讲清。

### 4.1 all-gather：聚齐

```
原语：每卡贡献自己的一块，结束后【每卡都拥有全部块的拼接】
        in            out
rank0  [A]   ──►   [A B C D]
rank1  [B]   ──►   [A B C D]
rank2  [C]   ──►   [A B C D]
rank3  [D]   ──►   [A B C D]
单卡发送量 ≈ 数据总量×(N-1)/N，结果体积放大 N 倍（短暂占显存）
```
FSDP 用它把"参数碎片"还原成"完整层参数"。

### 4.2 reduce-scatter：求和并切片

```
原语：每卡贡献一个【完整向量】，结束后【逐元素求和】再【按段分发】，每卡只收一段
        in (各卡同形)              out (只收自己那段)
rank0  [a0 a1 a2 a3]   ──►   [a0+b0+c0+d0]
rank1  [b0 b1 b2 b3]   ──►   [a1+b1+c1+d1]
rank2  [c0 c1 c2 c3]   ──►   [a2+b2+c2+d2]
rank3  [d0 d1 d2 d3]   ──►   [a3+b3+c3+d3]
```
**恒等式**：`all-reduce = reduce-scatter + all-gather`。DDP 做了完整 all-reduce；FSDP 反向只做前半段 reduce-scatter（梯度），前向单独做后半段 all-gather（参数）。这正是 FSDP 通信被精确拆分的原因。

---

## 5. wrap 策略：以谁为单位分片（关键参数）

FSDP 不是把整个模型当一个大块切，而是把模型递归地包成一个个 **FSDP unit**。**unit 是 all-gather / free 的最小粒度**，wrap 粒度直接决定"显存峰值 vs 通信次数"的权衡。

```
   整个模型只 wrap 一层（粗）         按 TransformerBlock wrap（推荐）
   ┌──────────────────┐            ┌──[Block1]──┐  ← 每个 block 一个 unit
   │   FSDP(whole)    │            │  FSDP      │
   │  一次性聚齐全模型 │            ├──[Block2]──┤  ← 算 Block_i 时只聚齐 Block_i
   │  → 峰值显存≈全量  │            │  FSDP      │
   │  → 通信次数极少    │            ├──[Block3]──┤
   └──────────────────┘            │  FSDP      │  → 峰值≈单block，通信次数↑
                                    └────────────┘
```

### 5.1 auto_wrap_policy（自动包装策略）

手动逐层 wrap 太累，FSDP 提供策略让它自动决定边界：

| 策略 | 含义 | 适用 |
|---|---|---|
| `transformer_auto_wrap_policy` | 指定 `transformer_layer_cls`（如 `LlamaDecoderLayer`），**每个 transformer block 包成一个 unit** | Transformer 训练首选 |
| `size_based_auto_wrap_policy` | 累积参数量超过 `min_num_params` 阈值就切一个 unit | 通用、非 transformer |
| 不 wrap | 整个模型一个 unit | 小模型/调试 |

**权衡的本质**：unit 越细 → 同时被聚齐的参数越少 → **峰值显存越低**，但 all-gather 调用次数越多 → **通信开销和 kernel launch 越多**。block 级是公认的甜点。

> HF Accelerate / Trainer 里通常通过 `fsdp_transformer_layer_cls_to_wrap` 指定要按哪个层类切。具体字段名以官方文档为准（见底部链接）。

---

## 6. sharding_strategy 三档（显存↔通信调节钮）

FSDP 允许你不必"全切到底"，按 ZeRO 等级选档：

```
FULL_SHARD     (≈ZeRO-3) 切 P+G+O，前向反向都 all-gather 参数
               每卡显存最低；通信最多          ← 默认、最省显存
SHARD_GRAD_OP  (≈ZeRO-2) 只切 G+O；参数前向后【不释放】，反向不重新 gather
               显存换通信：省一次反向 all-gather，但每卡常驻完整参数
NO_SHARD       (≈DDP)    不切，等价 DDP                ← 调试/对照基线
HYBRID_SHARD   节点内 FULL_SHARD + 节点间复制
               大集群：把昂贵的跨节点 all-gather 限制在节点内高速 NVLink
```

```
HYBRID_SHARD 拓扑（2 节点 ×4 卡）
 Node A: [g0 g1 g2 g3] 内部 FULL_SHARD（NVLink，快）
 Node B: [g0 g1 g2 g3] 内部 FULL_SHARD
 节点间：只对【梯度】做 all-reduce（跨节点慢链路，频率低）
 → 跨节点不传参数，只传梯度，省掉昂贵的跨机 all-gather
```

选择口诀：**显存够 → SHARD_GRAD_OP（更快）；显存紧 → FULL_SHARD；多机带宽差 → HYBRID_SHARD。**

---

## 7. 与 DDP / ZeRO / TP 对比

### 7.1 一张表看清

| 维度 | DDP | FSDP (FULL_SHARD) | DeepSpeed ZeRO-3 | 张量并行 TP |
|---|---|---|---|---|
| 切什么 | 不切（全复制） | P+G+O | P+G+O | 单层权重按行/列切 |
| 每卡参数显存 | $16\Psi$ | $\approx 16\Psi/N$ | $\approx 16\Psi/N$ | $\approx \Psi/N$（仅参数） |
| 梯度通信原语 | all-reduce | reduce-scatter | reduce-scatter | all-reduce（层内） |
| 参数通信 | 无（已全有） | 前向+反向 all-gather | 同左 | 层内 all-reduce 激活 |
| 通信触发点 | 反向结束一次 | 每个 unit 多次 | 每个 unit 多次 | 每层前向+反向 |
| 切分维度 | 数据 | 数据（沿 DP 维切状态） | 数据 | 模型（沿权重维切） |
| 实现 | PyTorch 原生 | PyTorch 原生 | 独立库 | Megatron 等 |
| 通信量级 | 低 | 中（约 1.5× DDP） | 中 | 高（每层激活） |

### 7.2 FSDP vs ZeRO-3：几乎等价

二者**算法等价**（都是 ZeRO-3 思想）。主要区别是工程层面：

- FSDP 是 **PyTorch 原生**，与 `torch.compile`、HF Trainer/Accelerate 集成更顺，无需额外引擎。
- ZeRO 是 **DeepSpeed** 独立库，offload / ZeRO-Infinity / 稀疏特性更成熟。详见 [[ai-framework/deepspeed/README]]。
- 选型：纯 PyTorch 栈、想用 `torch.compile` → FSDP；要极致 CPU/NVMe offload、已有 DeepSpeed 经验 → ZeRO。

### 7.3 FSDP 与 TP 正交，可叠加

FSDP 沿**数据维**切"模型状态"，TP 沿**权重维**切"单层计算"。两者维度不同，可组合成 2D 并行（FSDP + TP），是训练超大模型的常见配方。PyTorch 基础见 [[ai-framework/pytorch/README]]。

---

## 8. 数值例子 / 显存账 + 通信账（手算）

### 8.1 显存账：7B 模型，8 卡，混精 Adam

参数 $\Psi = 7\times10^9$，每参数总状态 $16$ 字节。

**模型状态显存（每卡）**：

| 方案 | 公式 | 每卡显存 |
|---|---|---|
| 单卡 / DDP | $16\Psi$ | $16 \times 7e9 = 112\,\text{GB}$ ❌ 放不下 |
| ZeRO-1 (切 O) | $4\Psi + 12\Psi/8$ | $28 + 10.5 = 38.5\,\text{GB}$ |
| ZeRO-2 (切 O+G) | $2\Psi + 14\Psi/8$ | $14 + 12.25 = 26.25\,\text{GB}$ |
| **FSDP FULL (切 O+G+P)** | $16\Psi/8$ | $\mathbf{14\,\text{GB}}$ ✅ |

$$\text{每卡模型状态} = \frac{16\Psi}{N} = \frac{16\times 7\times10^9}{8} = 14\times10^9\,\text{B} = 14\,\text{GB}$$

从 112GB → 14GB，**8 卡正好降到 1/8**。再加上 all-gather 临时聚齐"当前一层"的峰值（单层很小，几百 MB 级）和激活值，80GB A100 可从容训练。卡更多 → 每卡更低，这是 FSDP 的**线性可扩展性**。

### 8.2 通信账：FSDP 比 DDP 多传多少

设单步需通信的总参数体积为 $\Psi$（以 fp16 计为 $2\Psi$ 字节）。

- **DDP**：仅梯度 all-reduce。all-reduce 单卡通信量 ≈ $2\Psi$ 的 **$2\times\frac{N-1}{N}$** 倍 ≈ $2\Psi$（环形实现）。
- **FSDP FULL_SHARD**，单卡通信量拆三块：
  - 前向 all-gather 参数：≈ $1\Psi$
  - 反向 all-gather 参数：≈ $1\Psi$
  - 反向 reduce-scatter 梯度：≈ $1\Psi$

$$\frac{\text{FSDP 通信}}{\text{DDP 通信}} \approx \frac{1+1+1}{2} = 1.5\times$$

**结论**：FSDP 用约 **1.5 倍通信** 换来 **每卡显存 ÷ N**。所以 FSDP 强依赖**高带宽互联**（NVLink / IB）；带宽差时通信会成瓶颈 → 用 `SHARD_GRAD_OP`（省一次反向 all-gather）或 `HYBRID_SHARD` 缓解。

### 8.3 带宽直觉（约数，以实测为准）

7B 模型 fp16 参数 ≈ 14GB。若每步要 all-gather 三次量级的数据，在 NVLink（约 600 GB/s/卡级别）下通信可与计算重叠隐藏；在 PCIe（约几十 GB/s）下则可能裸露成为瓶颈。**这是为什么 FSDP 集群强烈推荐 NVLink + InfiniBand。**

---

## 9. 实践要点与坑

```
┌─ 训练一步的时间线（FSDP，理想重叠）────────────────────────┐
│ 计算 Layer_k 时，后台 prefetch all-gather Layer_{k+1} 参数 │
│ 反向算 Layer_k 时，后台 reduce-scatter Layer_{k+1} 梯度    │
│ → 通信被计算"盖住"，吞吐接近不分片                          │
└────────────────────────────────────────────────────────────┘
```

- **混合精度**：`MixedPrecision` 可分别设 `param_dtype` / `reduce_dtype` / `buffer_dtype`。常见 bf16 算、fp32 reduce（梯度求和用高精度防溢出）。
- **CPU offload**：`cpu_offload` 把碎片放 CPU，进一步省显存，但 PCIe 搬运慢，吞吐下降明显，显存实在不够才开。对标 ZeRO-Offload。
- **激活检查点（activation checkpointing）**：FSDP 省的是"模型状态"，**激活值另算**。长序列/大 batch 时激活才是显存大头，需配合 checkpoint（重算换显存）。
- **prefetch**：`forward_prefetch` / `backward_prefetch` 提前 gather 下一 unit，重叠通信与计算，对吞吐影响大，建议开。
- **wrap 粒度**：按 transformer block wrap 是甜点；wrap 太粗峰值爆显存，太细通信碎片化变慢。
- **保存权重**：分片状态下 state_dict 是分片的；保存完整 checkpoint 需用 `FullStateDictConfig`（或分片 checkpoint 格式），否则各卡只存到自己碎片。
- **不要和 DDP 同时包**：FSDP 已含数据并行语义，外面再套 DDP 会出错。
- **版本差异**：PyTorch 新版推出 FSDP2（基于 DTensor，per-parameter 分片，API 更简洁）。具体 API / 字段名 / 默认值**以对应 PyTorch 版本官方文档为准**。

---

## 常见问题（速查）

| 问题 | 答案 |
|---|---|
| FSDP 和 ZeRO-3 啥关系？ | 算法等价，FSDP 是 PyTorch 原生实现；ZeRO 是 DeepSpeed 实现 |
| 为什么前向、反向都要 all-gather？ | 前向算 y=Wx、反向求导都要完整 W；用完即 free，所以都得重新聚齐 |
| 梯度为什么 reduce-scatter 不是 all-reduce？ | 每卡只更新 1/N 参数，只需对应 1/N 梯度，省一半梯度通信 |
| 显存到底降多少？ | 模型状态从 $16\Psi$ → $16\Psi/N$，N 卡近似除以 N（FULL_SHARD） |
| 通信代价多大？ | 约 1.5× DDP；强依赖高带宽互联 |
| 激活值也被分片吗？ | 否。FSDP 只分片"模型状态"，激活需另配 activation checkpointing |
| FULL vs SHARD_GRAD_OP 怎么选？ | 显存紧用 FULL；显存够用 GRAD_OP 更快（省反向 all-gather） |
| 多机带宽差怎么办？ | HYBRID_SHARD：节点内分片、节点间只 all-reduce 梯度 |
| wrap 粒度怎么定？ | Transformer 按 decoder block（`transformer_auto_wrap_policy`） |
| 能和张量并行一起用吗？ | 能，二者正交，可组 FSDP+TP 的 2D 并行 |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-framework/deepspeed/README]] —— ZeRO 三级 / offload / 与 FSDP 对照
- [[ai-framework/pytorch/README]] —— 分布式基础 / 集合通信 / DTensor

### 官方与参考资料

- PyTorch FSDP 官方文档：https://pytorch.org/docs/stable/fsdp.html
- HF Accelerate FSDP 指南：https://huggingface.co/docs/accelerate/usage_guides/fsdp
- HF Transformers FSDP 配置：https://huggingface.co/docs/transformers/v4.41.0/en/fsdp#fsdp-configuration
- TrainingArguments（fsdp 参数）：https://huggingface.co/docs/transformers/v4.41.0/en/main_classes/trainer#transformers.TrainingArguments

**transformers 相关**
- 知乎讲解：https://zhuanlan.zhihu.com/p/648094197
- fsdp.json 示例：https://github.com/ifromeast/LLMTrainer/blob/main/02_fsdp/fsdp.json

**accelerate 相关**
- 用 PyTorch FSDP 微调 Llama 2 70B：https://zhuanlan.zhihu.com/p/671742753
- fsdp_config.yaml 示例：https://github.com/pacman100/LLM-Workshop/blob/main/chat_assistant/sft/training/configs/fsdp_config.yaml

> 注：本文 API 字段名 / 默认值 / 精确数字均"以对应版本官方文档为准"；显存与通信数字为帮助建立直觉的近似手算。
