# HuggingFace Accelerate

> 一套训练代码不改逻辑，靠"包一层 + 一份配置"就能从单卡跑到多卡、多机，并按需切换 DDP / FSDP / DeepSpeed 与混合精度。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[ai-framework/deepspeed/README]]

## 阅读地图

| 节 | 主题 | 你将搞清楚的问题 |
|----|------|------------------|
| 0 | 一句话锚点 | Accelerate 到底是什么、不是什么 |
| 1 | 地基/前置 | 裸 PyTorch 分布式为什么烦，它替你做了什么 |
| 2 | 核心抽象 `Accelerator` | `prepare()` 这一行包了哪些东西 |
| 3 | 设备/进程模型 | rank / world_size / local_rank / 主进程 |
| 4 | 一套代码跑单卡/多卡/多机 | 配置文件 + `accelerate launch` 怎么切换 |
| 5 | 封装 DDP | 数据并行的最底层原理与 all-reduce |
| 6 | 封装 FSDP | 参数/梯度/优化器分片，显存怎么省 |
| 7 | 封装 DeepSpeed | ZeRO 三级 + offload，与 FSDP 的关系 |
| 8 | 混合精度 | fp16/bf16、AMP、为什么能又快又省 |
| 9 | `device_map` 大模型分片 | 推理时把一个模型摊到多卡/CPU/磁盘 |
| 10 | 数值例子 | 13B 模型显存手算、通信量手算 |
| — | 对照表 / 常见问题 / 跳转 | 选型与排错 |

## 0. 一句话锚点

**Accelerate 是 PyTorch 之上的一层"分布式胶水"**：你照常写 `model / optimizer / dataloader` 的训练循环，只在关键几行包一个 `Accelerator`，它就在**运行时**根据一份配置，把你的代码映射到单卡、多卡（DDP/FSDP/DeepSpeed）或多机上，并接管混合精度、梯度累积、设备搬运等杂活。

它**不是**一个新的训练框架，也**不**发明新算法——底层调用的仍是 PyTorch 的 `DistributedDataParallel`、`FullyShardedDataParallel` 或微软的 DeepSpeed。它的价值是**"一套代码、零侵入、配置驱动"**。

```
            你的训练脚本 (train.py)  ←—— 永远不用改
                       │
                ┌──────┴───────┐
                │  Accelerate  │  ← 薄薄一层胶水
                └──────┬───────┘
        ┌──────────┬───┴────┬───────────┐
       DDP        FSDP   DeepSpeed     单卡/CPU
        └──────────┴────────┴───────────┘
                  底层都是 PyTorch
```

## 1. 地基/前置：它解决什么问题

### 1.1 裸 PyTorch 写分布式有多烦

要把一个单卡脚本改成多卡 DDP，你**手动**得做这些事：

1. 调 `torch.distributed.init_process_group(backend="nccl")` 初始化通信组；
2. 读环境变量 `RANK / LOCAL_RANK / WORLD_SIZE`，`torch.cuda.set_device(local_rank)`；
3. 把模型搬到对应 GPU，再用 `DistributedDataParallel(model, device_ids=[local_rank])` 包起来；
4. 给 DataLoader 套 `DistributedSampler`，否则每张卡读到的是**同样的数据**（白算）；
5. 只在主进程（rank 0）打印日志、存 checkpoint，否则 8 张卡写 8 份；
6. 想用混合精度还要手搓 `torch.cuda.amp.GradScaler` 的 `scale/step/update`；
7. 想换 FSDP 或 DeepSpeed，上面几乎全得重写。

这些都是**与模型逻辑无关的样板代码（boilerplate）**，却容易写错（最常见 bug：忘了 `DistributedSampler` 导致数据重复、忘了只在 rank0 存盘导致文件损坏）。

### 1.2 Accelerate 的承诺

把上面 7 件事**全部收进 `Accelerator` 对象**。你的脚本里只剩这个结构（伪代码，API 以官方文档为准）：

```python
from accelerate import Accelerator
accelerator = Accelerator()                       # ① 替你 init_process_group + 选设备
model, optimizer, dataloader = accelerator.prepare(  # ② 替你包 DDP/分 sampler/搬设备
    model, optimizer, dataloader)

for batch in dataloader:                           # batch 已在正确的 GPU 上
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)                     # ③ 替你做 AMP 的 scale/反传
    optimizer.step(); optimizer.zero_grad()
```

**关键直觉**：单卡跑、8 卡 DDP 跑、FSDP 跑、DeepSpeed 跑——上面这段代码**一个字都不用改**，差别全在外部的配置文件和启动命令里。

## 2. 核心抽象：`Accelerator` 与 `prepare()`

`Accelerator` 是一切的入口。`prepare()` 是它的核心动作——把你传进去的对象"改造"成分布式版本：

```
            prepare(model, optimizer, dataloader, scheduler)
                              │
   ┌──────────────┬──────────┴───────┬──────────────────┐
   ▼              ▼                  ▼                  ▼
 model         optimizer         dataloader         scheduler
   │              │                  │                  │
 搬到本卡       包装成            塞进              按真实总步数
 +按后端包      AcceleratedOptim  DistributedSampler   缩放
 DDP/FSDP/DS    (处理AMP/累积)    每卡只见 1/N 数据    (多卡步数变化)
```

要点：
- 传进去几个、返回几个，**顺序一致**；没传的（如自定义 loss 函数）不需要 prepare。
- `prepare` 之后，`model` 可能已经是 `DDP` / `FSDP` 的包装对象，所以**存盘要先 `accelerator.unwrap_model(model)`** 拿回原始模型，否则 state_dict 的 key 会多一层 `module.` 前缀。
- `accelerator.backward(loss)` 取代 `loss.backward()`，因为混合精度的 loss 缩放、梯度累积的"是否真的 step"都藏在这里。

## 3. 设备 / 进程模型：先把名词搞清

分布式里这几个词必须分清，否则后面全乱：

| 名词 | 含义 | 例子（2 机 × 4 卡 = 8 进程） |
|------|------|------|
| **world_size** | 全局进程总数 | 8 |
| **rank** | 该进程的全局编号 | 0~7 |
| **local_rank** | 该进程在**本机**内的编号 | 每机各 0~3 |
| **node** | 物理机器 | node0, node1 |
| **主进程** | rank==0 的那个，负责日志/存盘 | rank 0 |

```
   node 0 (机器A)                  node 1 (机器B)
 ┌───────────────────┐          ┌───────────────────┐
 │ GPU0 GPU1 GPU2 GPU3│          │ GPU0 GPU1 GPU2 GPU3│
 │ rank0 r1   r2   r3 │          │ rank4 r5   r6   r7 │
 │ local 0 1  2    3  │          │ local 0 1  2    3  │
 └───────────────────┘          └───────────────────┘
         └─────────── NCCL 跨机通信 ───────────┘
                world_size = 8
```

常用守卫：`if accelerator.is_main_process:` 只在 rank0 执行；`accelerator.print(...)` 自动只在主进程打印；`accelerator.wait_for_everyone()` 是一个**屏障（barrier）**，让所有进程在此对齐（常用在存盘前后）。

## 4. 一套代码跑单卡 / 多卡 / 多机

这是 Accelerate 最招牌的能力。脚本不变，**变的只有两样**：配置文件 + 启动命令。

### 4.1 生成配置：`accelerate config`

交互式问答会把你的答案存成 `~/.cache/huggingface/accelerate/default_config.yaml`：

```
accelerate config            # 交互式回答：用几台机、几张卡、要不要混合精度、用 DDP/FSDP/DeepSpeed
accelerate config default    # 直接生成一份默认配置（不问问题）
accelerate env               # 打印当前环境与配置，排错第一步
```

配置文件大致长这样（字段以官方文档为准）：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU        # NO / MULTI_GPU / FSDP / DEEPSPEED
num_machines: 1
num_processes: 4                    # 总进程数 = 总 GPU 数
mixed_precision: bf16              # no / fp16 / bf16
```

### 4.2 启动：`accelerate launch`

```
accelerate launch train.py                    # 用默认配置
accelerate launch --config_file cfg.yaml train.py
accelerate launch --num_processes 4 train.py  # 命令行临时覆盖
```

它本质上是 `torchrun`（旧名 `torch.distributed.launch`）的友好封装：替你拉起 N 个进程、设好 `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` 这些环境变量。

```
   单卡：  distributed_type: NO        num_processes: 1
   多卡：  distributed_type: MULTI_GPU num_processes: 4   (一台机 4 卡)
   多机：  num_machines: 2  + 每台机器各跑一次 launch，
           指定 --machine_rank 0/1 和同一个 --main_process_ip
            ┌── node0: accelerate launch --machine_rank 0 ... train.py
            └── node1: accelerate launch --machine_rank 1 ... train.py
```

**核心结论**：单卡→多卡→多机，`train.py` 始终是同一份；切换发生在配置与命令层。

## 5. 封装 DDP（DistributedDataParallel）

### 5.1 数据并行在干什么

DDP = **每张卡放一份完整的模型副本**，各自喂不同的数据分片算梯度，然后用 **all-reduce** 把各卡梯度求平均，保证所有副本参数始终一致。

```
 step 内：
  GPU0: 数据片0 → 前向 → 反向 → 本地梯度 g0 ┐
  GPU1: 数据片1 → 前向 → 反向 → 本地梯度 g1 ├─ all-reduce 求平均 ḡ
  GPU2: 数据片2 → 前向 → 反向 → 本地梯度 g2 ┤   每卡都拿到同一个 ḡ
  GPU3: 数据片3 → 前向 → 反向 → 本地梯度 g3 ┘
  → 每卡用 ḡ 各自 optimizer.step()，参数仍然一致
```

**all-reduce** 的含义：把分布在各卡的张量逐元素求和（再除以卡数得平均），结果广播回每张卡。NCCL 用 ring-allreduce 实现，通信量与卡数几乎无关（见第 10 节手算）。

### 5.2 Accelerate 里怎么开

配置 `distributed_type: MULTI_GPU` 即默认 DDP。`prepare(model)` 自动用 `DistributedDataParallel` 包模型、给 dataloader 加 `DistributedSampler`。

**适用判断**：模型 + 优化器状态能塞进**单张卡**显存时，DDP 是首选——实现简单、通信少、扩展性好。塞不下，才考虑 FSDP/DeepSpeed。

## 6. 封装 FSDP（Fully Sharded Data Parallel）

### 6.1 DDP 的痛点：每张卡都存了一整份

DDP 下，模型参数 P、梯度 G、优化器状态 O **每张卡各存一份完整的**。用 Adam 训练 fp32 时，单参数的显存开销约：参数 4B + 梯度 4B + 一阶动量 4B + 二阶动量 4B ≈ **16 字节/参数**。模型一大，单卡就爆了。

### 6.2 FSDP 的思路：把 P/G/O 切片，谁用谁临时拼

FSDP 把参数、梯度、优化器状态**沿卡分片（shard）**，每卡只常驻 1/N。某层要前向/反向时，临时用 **all-gather** 把这层的完整参数拼起来，算完即丢。

```
 DDP:   每卡 = [完整P][完整G][完整O]            ← 显存 = 全量
 FSDP:  卡0=[P切片0][G切片0][O切片0]
        卡1=[P切片1][G切片1][O切片1]   常驻只有 1/N
        计算某层时：all-gather 临时拼出该层完整 P → 算 → 丢弃
```

代价：多了 all-gather 的通信。收益：单卡显存常驻量从"全量"降到约"全量 /N + 一层的临时量"。FSDP 是 PyTorch 原生实现，思想等价于 DeepSpeed ZeRO-3。

### 6.3 Accelerate 里怎么开

`accelerate config` 选 FSDP，会追问分片策略（`FULL_SHARD` 切 P+G+O / `SHARD_GRAD_OP` 只切 G+O / `NO_SHARD` 退化为 DDP）、按哪种层做 wrap、是否 CPU offload 等。脚本仍然不变。

## 7. 封装 DeepSpeed

DeepSpeed 是微软的训练引擎，核心是 **ZeRO（Zero Redundancy Optimizer）**，按消除冗余的程度分三级，并支持把数据卸载到 CPU/NVMe：

```
 ZeRO-1: 切【优化器状态 O】           省得最少，通信最省
 ZeRO-2: 切【O + 梯度 G】
 ZeRO-3: 切【O + G + 参数 P】          ≈ FSDP FULL_SHARD，省得最多
 + Offload: 把 O/P 进一步搬到 CPU 内存甚至 NVMe 磁盘，用速度换容量
```

Accelerate 通过两种方式对接：要么在 `accelerate config` 里直接配 ZeRO 级别，要么挂一个 DeepSpeed 的 `ds_config.json`（更细粒度，老手常用）。脚本依然不动，`prepare()` 内部会把 optimizer/scheduler 交给 DeepSpeed 引擎接管。

**FSDP vs DeepSpeed 一句话**：ZeRO-3 ≈ FSDP，机制几乎同源；DeepSpeed 生态更老更全（offload/NVMe/推理优化齐），FSDP 是 PyTorch 原生、依赖更轻。详见 [[ai-framework/deepspeed/README]]。

## 8. 混合精度（Mixed Precision）

### 8.1 为什么要混

默认 fp32 占 4 字节，慢且占显存。**混合精度** = 大部分运算用 16 位（fp16 或 bf16，2 字节，省一半显存、用上 Tensor Core 算得更快），少数对精度敏感的步骤（如权重累加、loss）保持 fp32，兼顾速度与稳定。

```
   前向/反向矩阵乘  →  fp16/bf16  （快、省显存）
   权重主副本/累加  →  fp32       （稳，不丢精度）
```

### 8.2 fp16 vs bf16

| | fp16 | bf16 |
|---|---|---|
| 指数位（动态范围） | 小，易上溢/下溢 | 大，约等于 fp32 范围 |
| 尾数位（精度） | 较多 | 较少 |
| 是否需要 loss scaling | **需要**（防梯度下溢） | 基本不需要 |
| 硬件要求 | 较老卡也支持 | 需 Ampere(A100)及以上 |

**loss scaling**：fp16 表示太小的梯度会变成 0（下溢）。把 loss 乘一个大因子 S 再反传，梯度同步放大 S 倍躲过下溢，更新前再除回 S。这套 `GradScaler` 的逻辑被 `accelerator.backward()` 自动接管——你只需在配置里写 `mixed_precision: fp16`。

## 9. `device_map` 大模型分片（推理侧）

前面 5~8 节都是**训练**的并行。`device_map` 解决的是另一个问题：**推理时一个模型大到单卡放不下，怎么把它"摊开"**。

```python
import torch
from transformers import AutoModelForCausalLM

checkpoint = "facebook/opt-13b"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",           # 自动把各层分配到 GPU0/GPU1/.../CPU/磁盘
    offload_folder="offload",    # 放不下的层落到这个磁盘目录
    offload_state_dict=True,
    torch_dtype=torch.float16,
)
```

`device_map="auto"` 的逻辑：按显存容量，把模型一层层往设备上塞，**先塞满 GPU，再用 CPU 内存，最后落磁盘**。推理时数据流经某层，就把该层参数搬到 GPU 算，算完按需换出。

```
   一个 13B 模型（fp16 ≈ 26 GB），单张 24G 卡放不下：
   ┌─ 第 0~20 层 → GPU0 (显存充足部分)
   ├─ 第 21~32 层 → CPU 内存 (offload)
   └─ 剩余 → 磁盘 offload_folder
   推理：tensor 流到哪层，就把哪层临时搬上 GPU 计算
```

这叫 **"big model inference"**，本质是**模型并行 + 分级 offload**，牺牲速度（频繁搬运 + 磁盘 IO 很慢）换"能跑起来"。它和训练并行是两回事，别混。

## 数值例子 / 典型场景

### 例 1：13B 模型用 Adam 训练，单卡能放下吗？

参数量 $N = 13\times10^9$。fp16 训练 + Adam，常用经验估算每参数约 16 字节（fp16 参数 2 + fp16 梯度 2 + fp32 参数副本 4 + Adam 一阶 4 + 二阶 4 ≈ 16）：

$$M_{\text{state}} = 16 \times 13\times10^9 \text{ B} = 208\times10^9 \text{ B} \approx 194\ \text{GiB}$$

单张 80GB 的 A100 远远放不下 → **必须** FSDP/ZeRO 分片。

用 **ZeRO-3 / FSDP 切到 8 张卡**，常驻状态降为约：

$$194 / 8 \approx 24.3\ \text{GiB}$$

再加上激活值与临时 all-gather 的一层参数，单卡约 30+ GiB → 80GB 卡可行。**这就是分片的价值。**

### 例 2：DDP 的 all-reduce 通信量

设梯度总大小 $\Phi$ 字节，$N$ 张卡。ring-allreduce 每卡收发的数据量约：

$$2 \times \frac{N-1}{N} \times \Phi$$

取 13B 模型 fp16 梯度 $\Phi = 2 \times 13\times10^9 = 26\ \text{GB}$，$N=8$：

$$2 \times \tfrac{7}{8} \times 26 \approx 45.5\ \text{GB（每卡每步收发）}$$

注意 $\frac{N-1}{N}$ 上限为 1，所以**通信量几乎不随卡数增长**——这正是 ring-allreduce 可扩展的原因。若 NVLink 带宽 300 GB/s，单步梯度同步耗时量级约 $45.5/300 \approx 0.15$ s（理想值，实际还要叠加 overlap/拥塞）。

### 例 3：混合精度省多少显存

某层激活 fp32 占 4 GB，切到 bf16：$4 \to 2$ GB，**省一半**。整网激活若 fp32 是 16 GB，bf16 后约 8 GB，常常正是"放不下→放得下"的临界点。

## 对照表（与同类对比）

| 维度 | 单卡裸 PyTorch | DDP | FSDP | DeepSpeed (ZeRO-3) |
|------|---------------|-----|------|----------------------|
| 每卡参数副本 | 1 份 | **完整 1 份** | 1/N 分片 | 1/N 分片 |
| 显存占用 | 全量 | 全量 | 全量/N+临时 | 全量/N+临时 |
| 通信开销 | 无 | 低(all-reduce) | 中(all-gather) | 中~高(可配 offload) |
| 能训多大模型 | 最小 | 单卡放得下即可 | 很大 | 最大(可 NVMe offload) |
| 实现来源 | — | PyTorch 原生 | PyTorch 原生 | 微软引擎 |
| 何时选 | 调试/小模型 | 模型能进单卡 | 大模型、依赖轻 | 超大模型、要 offload |

**Accelerate 在哪？** 它是这四列上面的统一壳：同一份脚本 + 改配置即可在它们之间切换。

| 工具 | 定位差异 |
|------|---------|
| **Accelerate** | 薄胶水，零侵入，配置驱动；适合"自己写训练循环但不想碰分布式样板" |
| **torchrun** | 只负责拉进程/设环境变量，分布式逻辑仍要自己写 |
| **Trainer (transformers)** | 更高层、连训练循环都封好；牺牲灵活换省心（其底层也用 Accelerate） |
| **DeepSpeed / FSDP** | 是被 Accelerate 调用的"后端"，不是竞品 |

## 常见问题

| 问题 | 原因 / 解法 |
|------|------------|
| 多卡训练比单卡慢/没加速 | 多半 batch 太小、通信占比高，或忘了用 `DistributedSampler`（prepare 会自动加，自己别再手动套） |
| 每张卡读到相同数据 | 没经过 `prepare(dataloader)`，sampler 没分片 |
| 存的 checkpoint key 多了 `module.` | 存盘前没 `accelerator.unwrap_model(model)` |
| 日志/文件被写了 N 份 | 没用 `is_main_process` 守卫或 `accelerator.print` |
| fp16 训练 loss 变 NaN | fp16 数值不稳，换 `bf16`（A100+）或确认 loss scaling 生效 |
| 多机连不上/卡住 | `MASTER_ADDR/PORT`、`machine_rank` 配错，或防火墙挡了端口；先跑 `accelerate env` 核对 |
| `device_map="auto"` 推理很慢 | 模型被 offload 到 CPU/磁盘，搬运是瓶颈——这是用速度换"能跑"，正常现象 |
| 切换 DDP→FSDP 要改代码吗 | 不用，改配置即可；只是存/读 checkpoint 的写法可能要按分片调整 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，先看这里定位
- [[ai-framework/pytorch/README]] — DDP/FSDP/AMP 的底层都来自 PyTorch，是本篇的地基
- [[ai-framework/deepspeed/README]] — ZeRO 三级 + offload 的深入原理，本篇第 7 节的延伸

> 说明：本文聚焦稳定机制与原理。具体 CLI 参数、配置字段名、API 签名可能随版本变化，**以官方文档为准**：https://huggingface.co/docs/accelerate
