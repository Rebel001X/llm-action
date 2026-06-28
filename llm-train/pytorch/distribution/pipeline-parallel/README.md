# PyTorch 流水线并行（Pipeline Parallelism）

> 一句话定位：把一个**单卡放不下的模型按层切成若干段**，分别放到不同 GPU 上，像工厂流水线一样让 micro-batch 在各段之间「流」起来 —— 这是 PyTorch 原生 `torch.distributed.pipeline.sync.Pipe` 的入门与实战。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[ai-framework/pytorch/README]] · [[ai-infra/网络/集合通信原语]] · [[B07:llm-inference/大模型推理张量并行]]

本目录围绕 PyTorch 官方四篇教程展开，配套脚本 `ddp_pipeline.py` 在 **4 张 GPU** 上把一个约 **10.6 亿参数** 的 Transformer 跑通了「数据并行 × 流水线并行」的 2D 并行。本文档中**所有数字与日志均来自本目录 `ddp_pipeline.py` 的真实运行**，未核实处标注「约/见原文」。

---

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 锚点 | 一句话讲清流水线并行 | 按层切、micro-batch |
| 1 地基 | 为什么需要 PP；和 DP/TP 的分工 | 显存墙、三种并行 |
| 2 朴素模型并行 | 单纯按层切为什么慢（GPU 轮流闲置） | 串行、利用率 25% |
| 3 流水线机制 | micro-batch 如何填满「气泡」 | GPipe、chunks |
| 4 气泡率 | 气泡公式 + 数值手算 | bubble、$\frac{p-1}{m+p-1}$ |
| 5 PyTorch API | `Pipe`、`partition_len`、`chunks` 怎么用 | `nn.Sequential`、RRef |
| 6 DDP×PP | 4 卡 2D 并行拓扑与跨进程同步 | NCCL、all-reduce |
| 实操 | 真实运行命令、日志、显存占用 | `python ddp_pipeline.py` |
| 坑 | 常见报错与限制 | `local_value()`、Windows |

---

## 0. 一句话锚点

> **流水线并行 = 把模型「沿层」竖着切成几段，每段放一张卡；再把一个 batch 拆成多个 micro-batch，让它们像传送带上的零件一样依次穿过各段，使本来轮流闲置的卡尽量同时干活。**

记住三个对照（同属「拆模型」的并行，但拆法不同）：

| 并行方式 | 切什么 | 一句话 |
|---|---|---|
| 数据并行 DP/DDP | 切 **数据** | 每卡放**整个模型副本**，各吃一份数据，梯度 all-reduce |
| 张量并行 TP | 切**单层内的矩阵** | 一个 Linear 的权重横/纵切到多卡，层内通信 |
| **流水线并行 PP** | 切**层（模型的深度方向）** | 第 1~k 层在卡0，第 k+1~2k 层在卡1，层间传激活 |

---

## 1. 地基：为什么需要流水线并行？

### 1.1 显存墙

一个模型要训练，单卡至少要装下：**参数 + 梯度 + 优化器状态 + 激活值**。以本目录脚本的 10.6 亿参数（`emsize=4096, nhid=4096, nlayers=8`）为例，仅用 Adam（fp32）粗算：

$$
\text{显存} \approx \underbrace{4P}_{\text{参数}} + \underbrace{4P}_{\text{梯度}} + \underbrace{8P}_{\text{Adam:m,v}} = 16P
$$

$16 \times 1.06\times10^9 \approx 17\text{GB}$（还没算激活值）。当模型继续放大到几十/几百亿参数，**单卡再大也装不下** → 必须把模型本身切开。这就是 PP 的存在理由：**DP 解决不了显存墙，因为 DP 要求每卡都放整份模型。**

### 1.2 三种并行的分工

```
        要解决的问题                推荐手段
   ┌──────────────────────┐
   │ 模型放得下，只是慢     │ →  数据并行 DP/DDP（复制模型）
   │ 单层太大放不下         │ →  张量并行 TP（切层内矩阵）
   │ 模型太深整体放不下     │ →  流水线并行 PP（切层）★本文
   └──────────────────────┘
   实际大模型训练 = DP × TP × PP「3D 并行」组合拳
```

---

## 2. 朴素「模型并行」为何慢？—— 串行的代价

最朴素的做法（对应官方 *SINGLE-MACHINE MODEL PARALLEL BEST PRACTICES*）：把第 1 段放 GPU0、第 2 段放 GPU1，前向时 GPU0 算完把激活传给 GPU1。问题在于**同一时刻只有一张卡在干活**：

```
朴素模型并行（4 段为例），时间从左到右 →
GPU0:  [F0]                          [B0]
GPU1:       [F1]                [B1]
GPU2:            [F2]      [B2]
GPU3:                 [F3][B3]
        ↑ 任一时刻只有 1 张卡忙，其余 3 张闲置 → 利用率 ≈ 1/4 = 25%
```

> 直觉：4 段流水线，朴素做法的 GPU 利用率只有 $1/p = 1/4$。卡越多，浪费越严重。**这不是真正的「并行」，只是把模型「摊」到多卡以求放得下。**

---

## 3. 流水线并行的核心机制：micro-batch 填气泡

GPipe 的关键洞察：把一个 mini-batch 再切成 $m$ 个 **micro-batch**，让它们错峰流过各段。当 micro-batch #1 在 GPU1 时，micro-batch #2 已经能在 GPU0 上算了 —— 各卡就**重叠**起来了。

```
流水线并行（4 段，micro-batch m=4），数字=第几个 micro-batch
时间 →
GPU0: F1 F2 F3 F4 ......... B4 B3 B2 B1
GPU1: .. F1 F2 F3 F4 .... B4 B3 B2 B1 ..
GPU2: .... F1 F2 F3 F4 .B4 B3 B2 B1 ....
GPU3: ...... F1 F2 F3 F4 B4 B3 B2 B1 ....
        ↑填充期(bubble)  ↑稳态:4卡同时忙   ↑排空期(bubble)
```

- **填充期 / 排空期**：流水线刚启动和收尾时仍有卡空闲，这段空闲叫**气泡（bubble）**。
- **稳态**：中间一段所有卡同时在算不同 micro-batch，这才是 PP 提速的来源。
- micro-batch 越多（$m$ 越大），气泡占比越小，但单个 micro-batch 太小又会让 GPU kernel 利用率下降 —— 需要权衡。

---

## 4. 气泡率：一个必须会手算的公式

设流水线段数（stage 数）为 $p$，micro-batch 数为 $m$，且每个 micro-batch 在每段耗时相等。GPipe 的气泡占比（理想模型）：

$$
\text{bubble fraction} = \frac{p-1}{m + p - 1}
$$

**数值手算**（对应本脚本：每条流水线 $p=2$ 段，`chunks=8` 即 $m=8$）：

$$
\frac{p-1}{m+p-1} = \frac{2-1}{8+2-1} = \frac{1}{9} \approx 11.1\%
$$

即理论上约 **11% 的时间在气泡里浪费**，约 89% 有效。对照表（直观感受 $m$ 的作用）：

| 段数 $p$ | micro-batch $m$ | 气泡率 $\frac{p-1}{m+p-1}$ | 解读 |
|---|---|---|---|
| 2 | 1 | 50% | = 朴素模型并行，一半时间空转 |
| 2 | 8 | **11.1%**（本脚本） | chunks=8 的效果 |
| 4 | 4 | 42.8% | 段多 micro-batch 少，气泡大 |
| 4 | 16 | 16.7% | 加大 $m$ 显著压气泡 |
| 8 | 32 | 18.4% | 段越多越需要更多 micro-batch |

> 经验法则：**让 $m \ge 4p$**，气泡率通常能压到 20% 以下。

---

## 5. PyTorch 原生 API：`torch.distributed.pipeline.sync.Pipe`

PyTorch 把流水线并行封装在 `Pipe` 里。用法分三步，下面**每行都对应 `ddp_pipeline.py` 的真实代码**。

### 5.1 把模型表达成 `nn.Sequential`

`Pipe` 要求模型是一个 `nn.Sequential`（这样它才知道「层」的边界从哪切）。脚本把 Transformer 拆成 `Encoder + N×TransformerEncoderLayer + Decoder` 装进列表：

```python
# 真实超参（ddp_pipeline.py）
emsize  = 4096   # embedding 维度
nhid    = 4096   # FFN 隐层维度
nlayers = 8      # TransformerEncoderLayer 层数
nhead   = 16     # 注意力头数
```

### 5.2 计算每段层数 `partition_len` 并按段搬卡

```python
num_gpus = 2
partition_len = ((nlayers - 1) // num_gpus) + 1   # = ((8-1)//2)+1 = 4
# 即：8 层均分到 2 张卡，每段 4 层

for i in range(nlayers):
    transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    device = i // partition_len     # 第 0~3 层→device0, 第 4~7 层→device1
    tmp_list.append(transformer_block.to(device))
```

这里 `partition_len: 4` 正是真实日志里打印的那行 `partition_len: 4 rank: 0`。

### 5.3 用 `Pipe` 包裹，设 `chunks`（即 micro-batch 数）

```python
chunks = 8
model = Pipe(torch.nn.Sequential(*module_list),
             chunks=chunks,            # 把每个 batch 切成 8 个 micro-batch
             checkpoint="never")       # 不做激活重计算
```

| 参数 | 含义 | 本脚本取值 | 为什么 |
|---|---|---|---|
| `chunks` | micro-batch 数 $m$ | 8 | 越大气泡越小（见§4），8 让 $p=2$ 时气泡≈11% |
| `checkpoint` | 是否激活重计算 | `"never"` | 省时间不省显存；`"always"/"except_last"` 省显存换算力 |

> 关键坑：`Pipe` 的输出是 **RRef**（远程引用），不是普通 Tensor。要拿到本地结果必须 `.local_value()`（见下文实操）。

---

## 6. DDP × PP：4 卡 2D 并行拓扑

脚本不止用 PP，还在外层套了 DDP：**2 个进程，各驱动一条 2 段流水线，两条流水线是同一模型的副本，用 NCCL all-reduce 同步梯度。**

```
                  全局 batch (DDP 按 rank 切成 2 半)
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
 进程 rank0                 进程 rank1
 ┌─────────────┐           ┌─────────────┐
 │   Pipe A    │           │   Pipe B    │  ← 同一模型的两个副本
 │ GPU0 → GPU1 │           │ GPU2 → GPU3 │  ← 每条 = 2 段流水线(PP)
 └─────────────┘           └─────────────┘
     │ 反向后梯度               │
     └──────── DDP all-reduce ──┘          ← NCCL 跨进程同步
```

坐标：**DP 维度=2（rank0/rank1），PP 维度=2（每条流水线 2 段），合计 2×2 = 4 张 GPU。**

关键代码（真实）：

```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group(backend="nccl", ...)   # 跨进程用 NCCL
model = DistributedDataParallel(model)         # 注意：不传 device_ids！
```

> 为什么 `DistributedDataParallel(model)` **不传 `device_ids`**？因为 `Pipe` 是「多设备模块」（一条流水线横跨 GPU0+GPU1），DDP 此时不能假设模型在单一设备上。这也是日志里出现
> `Warning: Cuda time stats are not collected for multi-device modules` 的原因 —— **属正常告警，非报错**。

---

## 实操：命令 / 日志 / 显存（原文真料）

### 运行命令

```bash
python ddp_pipeline.py
```

### 训练日志（真实输出，节选）

```
partition_len: 4 rank: 0
partition_len: 4 rank: 1
[RANK 1]: Total parameters in model: 1,061,924,974
[RANK 0]: Total parameters in model: 1,061,924,974
[W logger.cpp:317] Warning: Cuda time stats are not collected for multi-device modules. (function operator())
[RANK 1]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 517.85 | loss 45.48 | ppl 56432121366884458496.00
[RANK 0]: | epoch   1 |    10/   50 batches | lr 5.00 | ms/batch 517.64 | loss 44.56 | ppl 22504870874679681024.00
[RANK 1]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 385.36 | loss 46.63 | ppl 178457758192923508736.00
[RANK 0]: | epoch   1 |    20/   50 batches | lr 5.00 | ms/batch 385.36 | loss 45.77 | ppl 75770339191959027712.00
[RANK 0]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 385.49 | loss 41.88 | ppl 1546590899039937792.00
[RANK 1]: | epoch   1 |    30/   50 batches | lr 5.00 | ms/batch 385.49 | loss 42.50 | ppl 2856464247237597696.00
[RANK 1]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 386.09 | loss 38.95 | ppl 82371305929986336.00
[RANK 0]: | epoch   1 |    40/   50 batches | lr 5.00 | ms/batch 386.09 | loss 39.99 | ppl 233730045251711744.00
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 0]: | end of epoch   1 | time: 22.57s | valid loss  0.91 | valid ppl     2.48
[RANK 0]: -----------------------------------------------------------------------------------------
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 1]: | end of epoch   1 | time: 22.59s | valid loss  0.91 | valid ppl     2.48
[RANK 1]: -----------------------------------------------------------------------------------------
[RANK 1]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 425.29 | loss 39.88 | ppl 209605830679931392.00
[RANK 0]: | epoch   2 |    10/   50 batches | lr 4.75 | ms/batch 427.78 | loss 39.88 | ppl 208211260703358496.00
[RANK 0]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 386.57 | loss 28.43 | ppl 2212304159830.81
[RANK 1]: | epoch   2 |    20/   50 batches | lr 4.75 | ms/batch 386.57 | loss 28.97 | ppl 3807579905494.86
[RANK 0]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 387.16 | loss 26.35 | ppl 278391001398.60
[RANK 1]: | epoch   2 |    30/   50 batches | lr 4.75 | ms/batch 387.17 | loss 25.44 | ppl 111852774843.71
[RANK 1]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 387.81 | loss 19.52 | ppl 298987362.70
[RANK 0]: | epoch   2 |    40/   50 batches | lr 4.75 | ms/batch 387.82 | loss 19.41 | ppl 269058605.29
[RANK 0]: | end of epoch   2 | time: 21.62s | valid loss  0.23 | valid ppl     1.26
[RANK 1]: | end of epoch   2 | time: 21.67s | valid loss  0.23 | valid ppl     1.26
[RANK 0]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 433.81 | loss 11.59 | ppl 108360.86
[RANK 1]: | epoch   3 |    10/   50 batches | lr 4.51 | ms/batch 426.98 | loss 11.59 | ppl 107512.77
[RANK 0]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 386.89 | loss 10.04 | ppl 22813.06
[RANK 1]: | epoch   3 |    20/   50 batches | lr 4.51 | ms/batch 386.89 | loss  9.99 | ppl 21883.85
[RANK 0]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 387.63 | loss 10.69 | ppl 43941.38
[RANK 1]: | epoch   3 |    30/   50 batches | lr 4.51 | ms/batch 387.63 | loss 10.55 | ppl 38258.17
[RANK 1]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 387.79 | loss  9.80 | ppl 18089.39
[RANK 0]: | epoch   3 |    40/   50 batches | lr 4.51 | ms/batch 387.79 | loss  9.80 | ppl 18088.98
[RANK 0]: | end of epoch   3 | time: 21.76s | valid loss  0.31 | valid ppl     1.37
[RANK 1]: | end of epoch   3 | time: 21.83s | valid loss  0.31 | valid ppl     1.37
[RANK 0]: | End of training | test loss  0.27 | test ppl     1.31
[RANK 1]: | End of training | test loss  0.27 | test ppl     1.31
```

**怎么读这份日志**：

| 字段 | 含义 | 解读 |
|---|---|---|
| `partition_len: 4` | 每段流水线 4 层 | 8 层 ÷ 2 卡 = 4，§5.2 算出 |
| `Total parameters: 1,061,924,974` | 模型约 10.6 亿参 | 单卡放不下整模型→必须 PP |
| `ms/batch 385~517` | 每 batch 毫秒 | 第一个 batch（517）含流水线**填充期**，稳态降到 ~385 |
| `loss` 巨大、`ppl` 天文数字 | 初始未收敛 | epoch1→3 loss 45→9，ppl 从 $10^{19}$ 降到 $10^4$，**在学** |
| `valid ppl 2.48→1.26→1.37` | 验证困惑度 | 此为玩具数据，看趋势即可 |
| 两 rank 数字几乎相同 | DDP 在同步 | all-reduce 后两副本梯度一致 |

### 显存占用（真实 `nvidia-smi`）

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|=============================================================================|
|    0   N/A  N/A     29461      C   ...nv-py310-cu117/bin/python    11612MiB |
|    0   N/A  N/A     29462      C   ...nv-py310-cu117/bin/python      556MiB |
|    1   N/A  N/A     29461      C   ...nv-py310-cu117/bin/python    11748MiB |
|    2   N/A  N/A     29462      C   ...nv-py310-cu117/bin/python    11354MiB |
|    3   N/A  N/A     29462      C   ...nv-py310-cu117/bin/python    11748MiB |
+-----------------------------------------------------------------------------+
```

**怎么读显存表**：

- 环境是 **Python 3.10 + CUDA 11.7**（路径 `nv-py310-cu117`）。
- 进程 `29461`（rank0）占用 GPU0、GPU1；进程 `29462`（rank1）占用 GPU2、GPU3 —— **正好印证「一条流水线横跨 2 卡、2 条流水线 = 4 卡」的拓扑**。
- GPU0 上 `29461` 占 11612MiB 是流水线**第一段**（含 embedding），同卡上 `29462` 的 556MiB 是 DDP/NCCL 通信缓冲。
- 每卡约 11~12GB，说明 10.6 亿参的模型被**摊薄**到了多卡 —— 这正是 PP 的价值。

---

## 官方教程对照（原文链接保留）

| 主题 | 链接 | 本目录配套 |
|---|---|---|
| Pipeline Parallelism 基本 API | https://pytorch.org/docs/stable/pipeline.html | §5 |
| 单机模型并行最佳实践 | https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html | §2 |
| nn.Transformer + TorchText | https://pytorch.org/tutorials/beginner/transformer_tutorial.html | `2-使用torchtext...md` |
| 用流水线并行训练 Transformer | https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html | `3-使用流水线并行...md` |
| DDP + 流水线并行训练 Transformer | https://pytorch.org/tutorials/advanced/ddp_pipeline.html | `4-使用DDP与流水线并行...md` |

> 同目录细读文档：`1-流水线.md`、`2-使用torchtext训练transformer模型.md`、`3-使用流水线并行训练Transformer模型.md`、`4-使用DDP与流水线并行训练Transformer模型.md`；脚本 `ddp_pipeline.py`。

---

## 常见问题 / 坑（表格）

| 现象 / 坑 | 原因 | 解法 |
|---|---|---|
| `Warning: Cuda time stats are not collected for multi-device modules` | `Pipe` 是多设备模块，DDP 无法统计单设备耗时 | **正常告警，忽略** |
| `model(data)` 拿到的不是 Tensor，后续报错 | `Pipe` 输出是 RRef（远程引用） | 取本地结果用 `output.local_value()` |
| loss 计算报跨设备错 | 输出在最后一段卡上，target 还在第一段卡 | 把 `target` 搬到输出所在设备再算 loss |
| `DistributedDataParallel(model, device_ids=[x])` 报错 | `Pipe` 跨多卡，不能指定单一 device | DDP **不传** `device_ids`（见 §6） |
| 段数多但提速不明显 | micro-batch（`chunks`）太少，气泡占比高 | 增大 `chunks`，经验 $m\ge 4p$（见 §4） |
| 显存仍 OOM | `checkpoint="never"` 不省激活显存 | 改 `checkpoint="always"`/`"except_last"` 换算力省显存 |
| Windows 跑不起来 | `Pipe`/RPC 后端对 Windows 支持有限 | 用 Linux + NCCL |
| 第一个 batch 特别慢（517 vs 385 ms） | 流水线**填充期** + CUDA 预热 | 正常，看稳态 ms/batch |
| 各 rank 数字不一致 | 进程组未正确 all-reduce | 检查 `MASTER_ADDR/PORT(29500)`、`init_process_group(backend="nccl")` |

---

## 🔗 跳转链接

**枢纽**：[[00-知识地图]] · [[llm-train/README]] · [[llm-train/pytorch/distribution/README]]

**框架实现**：[[ai-framework/pytorch/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/huggingface-peft/README]]

**进阶并行（PP 在大模型里的角色）**：[[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]] · [[B07:llm-inference/大模型推理张量并行]]

**底层通信**：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]

**模型与微调**：[[llm-algo/transformer/模型架构]] · [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]

**延伸**：[[llm-alignment/RLHF]] · [[llm-compression/quantization/量化基础]]
