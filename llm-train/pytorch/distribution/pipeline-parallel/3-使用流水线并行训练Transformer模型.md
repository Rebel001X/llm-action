# 使用流水线并行训练 Transformer 模型（PyTorch Pipe 实战）

> 用 PyTorch 内置的 `torch.distributed.pipeline.sync.Pipe`，把一个约 14 亿参数的 Transformer 切成两段、放到两块 GPU 上，用「流水线并行」单进程多卡训练。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[llm-algo/transformer/模型架构]] · [[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 你想知道 | 跳到 |
|---|---|
| 流水线并行到底解决什么问题 | §1 地基 |
| 为什么要拆 Encoder 层、怎么拆 | §2 拆模型 |
| 为什么 `Pipe` 要先起 RPC 框架 | §3 RPC 初始化 |
| micro-batch / chunks 是什么、气泡怎么算 | §4 流水线调度 |
| `Pipe` 的输入输出到底在哪块 GPU 上 | §5 数据流与设备 |
| 通信量 / 显存 / 加速比手算 | §「数值示例」 |
| 与张量并行、DDP 的取舍 | §「评价/对照」 |
| 代码骨架长啥样 | §6 最小示例 |

## 0. 一句话锚点

**流水线并行 = 把模型「按层」纵向切成几段（stage），每段放一块 GPU；再把一个 batch 切成多个 micro-batch，像工厂流水线一样让它们在各 stage 之间错峰流动，从而让多块 GPU 同时有活干。** 本文用的 `Pipe` 是 PyTorch 官方的「单机、同步」流水线实现。

## 1. 地基：为什么需要流水线并行

### 1.1 痛点：一块 GPU 装不下

一个 14 亿参数（1.4B）的模型，光是 fp32 权重就要：

$$
1.4\times10^9 \text{ 参数} \times 4\text{ B} \approx 5.6\text{ GB}
$$

训练时还要存：梯度（再 5.6 GB）、Adam 的一阶/二阶动量（再 11.2 GB）、以及**激活值**（activation，反向传播要用，随 batch×seq×layer 线性增长，往往是大头）。加起来轻松突破单卡显存。三条出路：

| 并行方式 | 切什么 | 一句话 |
|---|---|---|
| 数据并行 DDP | 切 batch，**每卡一份完整模型** | 装得下才行，解决不了"装不下" |
| 张量并行 TP | 切**单层内部**的矩阵（按行/列） | 通信频繁（每层 all-reduce），适合 NVLink 同机 |
| **流水线并行 PP** | 切**层**（把模型纵向分段） | 通信少（只在 stage 边界传激活），但有"气泡" |

本文聚焦 **PP**。它的好处是 stage 之间只在边界传一次激活张量，通信量小；代价是天然存在**流水线气泡（bubble）**——开头要等流水线灌满、结尾要等排空。

### 1.2 朴素分层（naive）为什么慢

如果只是简单把前一半层放 GPU0、后一半放 GPU1，串行执行：

```
朴素模型并行（无 micro-batch），时间轴 →
GPU0: [F0]........................[B0]...........
GPU1: ......[F1].........[B1]....................
         ↑ GPU1 在等 GPU0     ↑ GPU0 在等 GPU1
任意时刻只有一块卡在干活 → 利用率 ≈ 1/N，白买了卡
```

F=前向，B=反向。两块卡总有一块在空转。**流水线并行就是用 micro-batch 把这些空隙填满。**

## 2. 拆模型：从 `nn.Transformer` 到两段 `nn.Sequential`

### 2.1 哪一层最重，就拆哪一层

Transformer 里参数量最大的是 **`nn.TransformerEncoder`**，它内部是 `nlayers` 个 **`nn.TransformerEncoderLayer`** 堆叠。所以拆分策略是：

- **把 Embedding（含位置编码）单独抽成一个模块** → 放 stage 边界的开头；
- **把 N 个 EncoderLayer 平均分两半** → 一半给 GPU0，一半给 GPU1；
- **把输出投影（Decoder/Linear → 词表）单独抽成一个模块** → 放结尾。

把这些子模块按顺序塞进一个 `nn.Sequential`，`Pipe` 会沿着这个 Sequential 自动找切点。

### 2.2 拆分示意图

```
原始 Transformer（约 1.4B 参数，12 层 EncoderLayer 为例）
┌──────────────────────────────────────────────┐
│ Embedding+PosEnc → L0 L1 ... L11 → Decoder/Out │
└──────────────────────────────────────────────┘
            │  按层纵向切成 2 段
            ▼
 Stage 0  (GPU 0)              Stage 1  (GPU 1)
┌───────────────────────┐   ┌───────────────────────┐
│ Embedding+PosEnc      │   │ L6  L7  L8            │
│ L0  L1  L2  L3  L4  L5 │──▶│ L9  L10 L11           │
│                       │   │ Decoder/Linear→Vocab  │
└───────────────────────┘   └───────────────────────┘
        激活张量在边界从 GPU0 → GPU1（一次拷贝）
```

> 注：原教程文字描述用 4096 维、4096 隐藏、16 头、12 层、每卡 8 层——这里"每卡 8 层"对应的是把 Embedding/Decoder 也算进 Sequential 后的**模块数**平均，不必死抠数字，核心是"层切两半"。**以官方教程代码实际切点为准。**

### 2.3 为什么只切成「两段」很关键

教程特别强调：传给 `Pipe` 的 `nn.Sequential` **最好只含 2 个元素**（恰好对应 2 块 GPU）。原因：

- `Pipe` 会把 Sequential 的每个**顶层子模块**当作一个可放置单元。如果顶层只有 2 个子模块（每个子模块内部再嵌一串层），`Pipe` 就只产生 **2 个 partition**，stage 间只有 1 个边界 → 跨分区拷贝最少。
- 如果顶层散落很多小模块，`Pipe` 可能切出更多分区，产生不必要的跨设备拷贝开销。

所以工程上常把"GPU0 该跑的所有层"包进**一个** `nn.Sequential` 子模块，"GPU1 该跑的"包进**另一个**，再用外层 `nn.Sequential([part0, part1])` 交给 `Pipe`。

## 3. RPC 初始化：为什么流水线要先起 RPC

`Pipe` 的实现依赖 **RRef（Remote Reference，远程引用）**，而 RRef 属于 PyTorch 的 **RPC（远程过程调用）框架**。即便我们现在是**单机多卡、单进程**，也必须初始化 RPC——这是为了**未来能无缝扩展到跨主机流水线**（stage 分布在不同机器上）。

```
单进程 driver
   │  init_rpc(name="worker", rank=0, world_size=1)
   ▼
┌─────────────────────────────────────────┐
│  RPC 框架（提供 RRef）                    │
│   └─ Pipe 用 RRef 引用各 stage 的子模块  │
│        ├─ stage0 → cuda:0               │
│        └─ stage1 → cuda:1               │
└─────────────────────────────────────────┘
```

要点：

- **单进程驱动多 GPU** → 只需 1 个 worker，`world_size=1`、`rank=0`、`init_method` 指向 localhost 即可。
- 必须在构造 `Pipe(...)` **之前**调用 `torch.distributed.rpc.init_rpc(...)`，否则 `Pipe` 拿不到 RRef 会报错。
- 训练结束记得 `torch.distributed.rpc.shutdown()` 优雅退出。

> 命令/参数默认值以官方文档为准；不同 PyTorch 版本 `init_rpc` 的后端（`TensorPipe`）配置略有差异。

## 4. 流水线调度：micro-batch、chunks 与气泡

### 4.1 核心动作：把 mini-batch 再切成 micro-batch

`Pipe` 接收一个完整的 mini-batch，按 `chunks` 参数把它沿 batch 维切成若干 **micro-batch**，依次喂入流水线。这样当 micro-batch #1 在 stage1 做前向时，micro-batch #2 已经能在 stage0 开工，两块卡就**重叠**起来了。

```
chunks=4 的 GPipe 式调度（F=前向, B=反向, 数字=micro-batch 号）
时间 →
GPU0: F1 F2 F3 F4 ........ B4 B3 B2 B1
GPU1: .. F1 F2 F3 F4 .. B4 B3 B2 B1 ..
       └bubble┘            └bubble┘
       灌满期             排空期（仍有少量空转，但比朴素好太多）
```

灌满（warm-up）和排空（cool-down）阶段就是**气泡**——这段时间部分 GPU 闲着。

### 4.2 气泡占比公式（直觉版）

设 stage 数 $p$、micro-batch 数 $m$，每个 micro-batch 在每个 stage 上耗时近似相等，则 GPipe 式调度的**气泡时间占比**约为：

$$
\text{bubble fraction} \approx \frac{p-1}{m+p-1}
$$

含义：**micro-batch 越多（$m$ 越大），气泡占比越小**，流水线越接近满负荷。

### 4.3 chunks 的权衡

| chunks 增大 | 好处 | 代价 |
|---|---|---|
| 气泡变小、利用率升高 | ✅ | 每个 micro-batch 的 BatchNorm/统计噪声变大；过多 launch 开销 |
| | | micro-batch 太小，单个 GPU kernel 利用率下降 |

经验：让 $m \ge 4p$ 通常能把气泡压到可接受范围。**具体最优值需实测。**

## 5. 数据流与设备：输入输出在哪块 GPU 上

这是新手最容易踩的坑：

```
你的输入 batch（放在 cuda:0，即第一个 stage 所在卡）
        │  Pipe 内部切 micro-batch、逐 stage 前向
        ▼
   stage0 (cuda:0) → 激活拷到 cuda:1 → stage1 (cuda:1)
        │
        ▼
Pipe 的输出是 RRef，需 .local_value() 取回；
输出张量位于【最后一个 stage 所在卡】= cuda:1
        │
        ▼
计算 loss 时：target 必须也 .to(cuda:1)，否则设备不匹配报错
```

记忆口诀：**输入跟着第一段走，输出跟着最后一段走，loss 在最后那块卡上算。**

## 关键公式 / 数值示例（手算一遍才踏实）

以教程配置近似估算（取整、便于心算）：嵌入维度 $d=4096$，FFN 隐藏 $d_{ff}=4096$（教程值，通常 FFN=4×d，这里按教程取 4096），层数 $L=12$，注意力头 16，序列长 $S$，batch $B$。

### ① 参数量（约 1.4B 的来历）

单个 EncoderLayer 主要参数：
- 自注意力 Q/K/V/O 四个投影：$4 d^2 = 4\times4096^2 \approx 6.7\times10^7$
- FFN 两个线性：$2\, d\, d_{ff} = 2\times4096\times4096 \approx 3.4\times10^7$

每层约 $1.0\times10^8$ ≈ 0.1B。12 层 ≈ **1.2B**，加上 Embedding（词表×4096）等 ≈ **约 1.4B**，与教程吻合。

### ② 单卡放不下、两卡为何能放下

fp32 训练「权重+梯度+Adam 动量」约为参数量的 **16 字节/参数**：

$$
1.4\times10^9 \times 16\text{ B} \approx 22.4\text{ GB（仅优化器状态相关，还没算激活）}
$$

一块 16 GB 卡直接 OOM；切两段后每卡只持有约一半层 ≈ **11 GB 量级**，加上激活才放得下。**这就是流水线并行"省显存"的本质：每卡只存自己那段的权重+优化器状态+激活。**

### ③ 边界通信量（PP 通信省在哪）

stage 边界每个 micro-batch 只传一次激活张量，大小：

$$
B_\mu \times S \times d \times 2\text{B（fp16）}
$$

设 micro-batch $B_\mu=4$、$S=128$、$d=4096$：

$$
4\times128\times4096\times2 \approx 4.2\text{ MB / micro-batch / 方向}
$$

对比**张量并行**：每层前向+反向都要 all-reduce 一个 $B\times S\times d$ 张量，$L$ 层就是几十次集合通信。**PP 一个 batch 只在 1 个边界传 $m$ 次小张量，通信量比 TP 小一两个数量级**——这正是 PP 跨节点（NVLink 之外）也能用的原因。详见 [[ai-infra/网络/集合通信原语]]。

### ④ 加速比（含气泡）

$p=2$、$m=4$：气泡占比 $\approx (2-1)/(4+2-1)=1/5=20\%$。
理想 2 倍，实际 $\approx 2\times(1-0.2)=1.6\times$。
把 $m$ 提到 8：气泡 $\approx 1/9\approx11\%$，加速比 $\approx 1.78\times$。**结论：多块卡 + 多 micro-batch 才划算。**

## 6. 最小示例思路（骨架，非完整可运行代码）

```python
import torch, torch.nn as nn
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe

# 1) 起 RPC（单 worker 驱动多 GPU；Pipe 依赖 RRef）
rpc.init_rpc("worker", rank=0, world_size=1,
             rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method="file:///tmp/rpc"))

# 2) 拆模型：part0 放 cuda:0，part1 放 cuda:1
part0 = nn.Sequential(Embedding(...), *layers[:6]).to("cuda:0")
part1 = nn.Sequential(*layers[6:], OutProj(...)).to("cuda:1")
model = nn.Sequential(part0, part1)          # 顶层恰好 2 个子模块 → 2 个 partition

# 3) 交给 Pipe：chunks=micro-batch 数量
model = Pipe(model, chunks=8)                # chunks 越大气泡越小（见 §4）

# 4) 前向：输入放第一段卡；输出是 RRef，在最后一段卡
out = model(inputs.to("cuda:0")).local_value()
loss = criterion(out, targets.to("cuda:1"))  # target 也要搬到最后一段卡！
loss.backward()                              # Pipe 内部处理跨 stage 反向
optimizer.step()

rpc.shutdown()                               # 收尾
```

> 以上 API 名称/参数以你所用 PyTorch 版本官方文档为准；`torch.distributed.pipeline.sync.Pipe` 在较新版本可能被 `torch.distributed.pipelining` 取代，迁移以官方迁移指南为准。

## 评价 / 对照 / 局限

| 维度 | 流水线并行 PP（本文 Pipe） | 张量并行 TP | 数据并行 DDP |
|---|---|---|---|
| 切分粒度 | 按层（stage） | 层内矩阵 | 按 batch |
| 通信频率 | 低（仅 stage 边界，每 micro-batch 1 次） | 高（每层 all-reduce） | 中（梯度 all-reduce 1 次/step） |
| 省显存（权重） | ✅ 每卡只存一段 | ✅ 每卡只存一片 | ❌ 每卡整模型 |
| 主要开销 | 流水线气泡（bubble） | 频繁集合通信 | 模型须单卡装下 |
| 适合场景 | 模型层数多、跨节点 | 单层巨大、同机 NVLink | 模型不大、要扩 batch |

**Pipe 的局限：**
- **同步流水线**：所有 micro-batch 跑完才更新参数，气泡无法完全消除（需更激进的 1F1B / interleaved 调度，如 Megatron-LM 的实现）。
- **负载均衡敏感**：两段层数/计算量若不均，慢的那段拖累整体 → 切点要让两 stage 耗时尽量相等。
- **顶层 Sequential 结构有要求**：切点由顶层子模块决定，需手工把层包成恰好 N 个子模块。
- **激活显存仍在**：PP 省的是"权重+优化器状态"，激活仍随 batch 增长 → 常与**激活重计算（gradient checkpointing）**配合。
- **生产级常用组合**：PP × TP × DP 三维并行（3D parallelism）才是大模型训练主流，单用 PP 多见于教学与中等规模。详见 [[llm-train/pytorch/distribution/README]]。

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-train/pytorch/distribution/README]] — 分布式训练总览（DP/TP/PP 全家桶）
- [[llm-algo/transformer/模型架构]] — 被切分的 Transformer 本体
- [[B07:llm-inference/大模型推理张量并行]] — 对照另一种切法（层内 vs 层间）
- [[ai-infra/网络/集合通信原语]] — all-reduce / send-recv，PP 边界通信底座
- [[docs/transformer内存估算]] — 显存账怎么算（权重/激活/优化器）
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — 加速比/利用率/MFU 等指标口径
