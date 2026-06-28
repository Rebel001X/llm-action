# PyTorch 分布式训练

> 用 `torch.distributed` 把"一张卡跑不下/跑不快"的训练，拆成多进程协同：每个进程算自己的一份数据，靠**集合通信**把梯度对齐，效果等价于一张超大卡。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[llm-train/pytorch/distribution/多机多卡]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 标题 | 你将搞懂 | 难度 |
|----|------|----------|------|
| 0 | 一句话锚点 | 分布式训练到底在做什么 | ★ |
| 1 | 地基/前置 | 进程、GPU、梯度、为什么要并行 | ★ |
| 2 | torch.distributed 全景 | 这个库由哪几块组成 | ★★ |
| 3 | 进程组 / rank / world_size | 进程怎么被编号、怎么分组 | ★★ |
| 4 | init_process_group | 进程之间如何"握手"建连 | ★★ |
| 5 | 集合通信原语 | AllReduce 等是什么、算什么 | ★★ |
| 6 | DDP 原理 | 数据并行 + 梯度桶 + 计算通信重叠 | ★★★ |
| 7 | DistributedSampler | 数据怎么不重叠地切给各进程 | ★★ |
| 8 | 启动方式 | torchrun / launch / mp.spawn | ★★ |
| 9 | 数值例子 / 对照 / 实践 | 通信量手算、DP vs DDP、调参 | ★★★ |
| 10 | 常见问题 | 踩坑速查 | ★★ |

---

## 0. 一句话锚点

**分布式数据并行（DDP）= 把同一个模型复制到 N 个进程（通常 1 进程绑 1 GPU），每个进程喂不同的数据子集，各自反向传播得到自己的梯度，然后用 AllReduce 把 N 份梯度求平均，使所有副本的参数始终保持一致。** 它在数学上等价于用 N 倍的 batch size 在单卡上训练。

```
单卡:   [全部数据] → 1 个模型 → 1 份梯度 → 更新
        ─────────────────────────────────────
分布式: [数据/4] → 模型副本0 ┐
        [数据/4] → 模型副本1 ┤  AllReduce
        [数据/4] → 模型副本2 ┤  (梯度求平均) → 各自更新(结果相同)
        [数据/4] → 模型副本3 ┘
```

---

## 1. 地基 / 前置（不假设你记得任何概念）

**为什么需要分布式？** 两个动机，记住它们贯穿全文：
- **跑得下**：模型 + 优化器状态 + 激活值的显存超过单卡（如 24/48/80 GB）。
- **跑得快**：单卡算一个 epoch 太慢，想用更多卡线性加速。

本文聚焦最常用、最基础的**数据并行**（每张卡都有完整模型，只切数据）。当单卡连一份模型都放不下时，才需要张量并行/流水线并行/ZeRO 切分，那是 [[llm-train/pytorch/distribution/多机多卡]] 的话题。

**几个原子概念**：

| 概念 | 一句话 |
|------|--------|
| **进程 (process)** | 操作系统里独立运行的程序实例，有独立内存。分布式训练里"1 进程 = 1 个训练副本"。|
| **GPU** | 真正做矩阵运算的硬件。约定每个进程**独占一张** GPU，避免争抢。|
| **梯度 (gradient)** | `loss.backward()` 算出的、每个参数应该往哪个方向改的量。是分布式里**唯一需要跨进程同步**的东西（参数初始一致 + 梯度一致 ⇒ 更新后参数一致）。|
| **集合通信 (collective)** | 一组进程**共同参与**的通信操作（如所有进程的数据求和）。区别于"两两点对点"。详见 [[ai-infra/网络/集合通信原语]]。|
| **后端 (backend)** | 真正搬数据的通信库。GPU 用 **NCCL**（NVIDIA 集合通信库，走 NVLink/PCIe/IB），CPU 用 **Gloo**。|

**为什么"同步梯度"就够了？** 假设两个副本初始参数完全相同 $\theta$。副本 0 在数据 $B_0$ 上得到梯度 $g_0$，副本 1 在 $B_1$ 上得到 $g_1$。若两边都用**平均梯度** $\bar g = (g_0 + g_1)/2$ 更新：

$$\theta' = \theta - \eta \bar g$$

则更新后两边参数**仍然相同**。这等价于在合并 batch $B_0 \cup B_1$ 上单卡训练的梯度（损失对样本取均值时）。所以"参数同步"只需在初始化时做一次广播，之后每步只同步梯度即可。

---

## 2. `torch.distributed` 全景

`torch.distributed`（简称 `dist`）是 PyTorch 的分布式通信底座。它**不是**某种并行策略，而是提供"让多个进程互相通信"的能力，上层的 DDP / FSDP / 张量并行都建在它之上。

```
┌─────────────────────────────────────────────┐
│  上层封装: DistributedDataParallel (DDP)      │  ← 你常直接用这层
│            FullyShardedDataParallel (FSDP)    │
├─────────────────────────────────────────────┤
│  torch.distributed 核心 API                   │
│   - init_process_group / destroy             │  建连/拆连
│   - get_rank / get_world_size                │  问"我是谁/共几人"
│   - all_reduce / broadcast / all_gather ...  │  集合通信原语
├─────────────────────────────────────────────┤
│  通信后端 (backend)                           │
│   NCCL(GPU)   |   Gloo(CPU/GPU)   |   MPI     │
├─────────────────────────────────────────────┤
│  传输层: NVLink / PCIe / InfiniBand / TCP     │  实际物理链路
└─────────────────────────────────────────────┘
```

**关键心智模型**：你写的训练脚本会被**同时启动 N 份**（N = 进程数）。每一份代码**完全相同**，但通过 `rank` 知道"我是第几个"，从而拿不同数据、打印不同日志。这叫 **SPMD（单程序多数据）**。

---

## 3. 进程组 / rank / world_size

这三者是分布式的"身份证系统"。

| 术语 | 含义 | 类比 |
|------|------|------|
| **world** | 参与本次训练的**所有进程的集合** | 整个班级 |
| **world_size** | world 里进程总数 | 班级总人数 |
| **rank** | 当前进程在 world 里的**全局编号**，`0 .. world_size-1` | 你的学号（全校唯一）|
| **local_rank** | 当前进程在**本机内**的编号 | 你在本班的座位号 |
| **进程组 (process group)** | 一组进程的子集合，集合通信在组内进行 | 一个学习小组 |

默认存在一个包含所有进程的**默认进程组（WORLD）**；多数训练只用它。张量并行等高级场景才会切出多个子组。

`rank=0` 习惯上被当作**主进程（master）**，负责打印日志、保存 checkpoint、写 TensorBoard——因为这些动作只需做一次，否则 N 个进程会写同一个文件互相覆盖。

```
2 机 × 4 卡 = world_size 8 的编号布局
┌──────────── node 0 ────────────┐   ┌──────────── node 1 ────────────┐
│ GPU0   GPU1   GPU2   GPU3       │   │ GPU0   GPU1   GPU2   GPU3       │
│ rank0  rank1  rank2  rank3      │   │ rank4  rank5  rank6  rank7      │
│ lrank0 lrank1 lrank2 lrank3     │   │ lrank0 lrank1 lrank2 lrank3     │
└────────────────────────────────┘   └────────────────────────────────┘
        ↑ rank 全局唯一            local_rank 每机重新从 0 数 ↑
```

**为什么要区分 rank 和 local_rank？**
- 用 `local_rank` 选 GPU：`torch.cuda.set_device(local_rank)`，因为每台机器的 GPU 都从 0 编号。
- 用 `rank` 判断身份：`if rank == 0: save_checkpoint()`，因为全局只想要一个主进程。

```python
import torch.distributed as dist
rank        = dist.get_rank()          # 全局编号
world_size  = dist.get_world_size()    # 总进程数
local_rank  = int(os.environ["LOCAL_RANK"])  # torchrun 注入的环境变量
```

---

## 4. `init_process_group`：进程间"握手"

在做任何通信前，所有进程必须先**互相认识、建立连接**。这一步由 `dist.init_process_group()` 完成——把散落在各机器的 N 个进程"焊接"成一个能集合通信的整体。

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",        # GPU 选 nccl；纯 CPU 选 gloo
    init_method="env://",  # 从环境变量读取建连信息(最常用)
    world_size=world_size, # 总进程数(env:// 下可省, 由环境变量提供)
    rank=rank,             # 本进程编号(env:// 下可省)
)
```

**`init_method` 是什么？** 它告诉进程"去哪里碰头、怎么找到彼此"。最常用的 `env://` 表示从下面这组环境变量读取碰头信息（由启动器自动设置）：

| 环境变量 | 含义 |
|----------|------|
| `MASTER_ADDR` | rank 0 所在机器的 IP/主机名（碰头点）|
| `MASTER_PORT` | rank 0 监听的端口 |
| `WORLD_SIZE` | 总进程数 |
| `RANK` | 本进程全局编号 |
| `LOCAL_RANK` | 本进程机内编号 |

**握手过程（直觉）**：

```
所有进程都去 MASTER_ADDR:MASTER_PORT 这个"集合点"报到
        │
   rank0(master) ←── rank1 报到 ("我是1号, 共8人")
        ↑       ←── rank2 报到
        │       ←── ...  rank7 报到
        │
   集齐 world_size=8 人后 → rendezvous(集合)完成
        │
   建立 NCCL 通信子(communicator) → 之后可做 AllReduce
```

这是一个**阻塞式屏障**：任何进程的 `init_process_group` 都会卡住，直到**全部** world_size 个进程都到齐。所以若你启动了 8 个进程但只起来 7 个，程序会**静默卡死**——这是头号新手坑（见第 10 节）。

收尾时调用 `dist.destroy_process_group()` 释放资源。

---

## 5. 集合通信原语（DDP 的发动机）

DDP 同步梯度靠的就是 **AllReduce**。先建立直觉，详细对比见 [[ai-infra/网络/集合通信原语]]。

**AllReduce = Reduce（归约/求和）+ Broadcast（广播）**：把所有进程各自的张量按元素求和（或求平均），结果**每个进程都拿到一份**。

```
AllReduce(sum) 前              AllReduce(sum) 后
rank0: [1, 2]                  rank0: [10, 12]
rank1: [3, 4]      ────►       rank1: [10, 12]
rank2: [2, 2]                  rank2: [10, 12]   ← 三份求和: 1+3+2=6? 不,
rank3: [4, 4]                  rank3: [10, 12]      [1+3+2+4, 2+4+2+4]=[10,12]
```

DDP 用的是 **求平均** 版本（sum 后除以 world_size），让等效 batch 的梯度尺度正确。

现代实现用 **Ring-AllReduce（环形）** 算法：N 个进程排成环，数据分成 N 块，经过 $2(N-1)$ 步收发完成。它的妙处是**每个进程的通信量与 N 几乎无关**（恒定约 $2$ 倍参数量），不会因为加卡而让某个节点成为瓶颈——这正是它能扩展到上千卡的原因。

```
Ring 拓扑(4 进程):  r0 → r1 → r2 → r3 → (回到 r0)
每步只跟"右邻居"发、跟"左邻居"收，带宽被均摊。
```

其他会用到的原语：**Broadcast**（rank0 把初始参数发给所有人，DDP 初始化时用）、**AllGather**（FSDP 收集参数分片用）、**ReduceScatter**（ZeRO 用）。

---

## 6. DDP 原理：梯度桶 + 计算通信重叠

`DistributedDataParallel`（DDP）是数据并行的标准实现。用法极简：

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])   # 关键一行
# 之后 forward/backward/step 写法与单卡完全一样
```

但这一行背后有三个精妙设计，是面试高频考点。

### 6.1 初始化：参数广播

DDP 构造时，把 **rank0 的模型参数 broadcast 给所有进程**，保证所有副本起点完全一致。这一步只做一次。

### 6.2 反向传播：自动 AllReduce 梯度

DDP 在每个参数上注册了 **autograd hook（钩子）**。当 `backward()` 算出某个参数的梯度时，hook 被触发，自动发起对该梯度的 AllReduce。你**不需要手写任何通信代码**。

### 6.3 梯度桶（gradient bucketing）—— 为什么不一个一个发

朴素做法是"每算出一个参数的梯度就 AllReduce 一次"。但大模型有上千个参数张量，逐个发起通信会被**固定开销（latency）**淹没——发 1000 个小包远慢于发几个大包。

**解法：分桶。** DDP 把多个参数的梯度**攒进固定大小的桶（默认约 25 MB，以官方为准）**，桶满了才整桶 AllReduce 一次。这样把上千次小通信合并成几十次大通信，吞吐大幅提升。

```
参数梯度按"反向就绪顺序"装桶:
 ┌── bucket 0 (25MB) ──┐ ┌── bucket 1 ──┐ ┌── bucket 2 ──┐
 │ grad_a grad_b grad_c│ │ grad_d grad_e│ │ grad_f ...    │
 └─────────┬───────────┘ └──────┬───────┘ └──────┬────────┘
       满了→AllReduce        满了→AllReduce     满了→AllReduce
```

### 6.4 计算通信重叠（overlap）—— 性能核心

这是 DDP 真正快的原因。反向传播是**从输出层往输入层**算的；网络**末尾层的梯度最先算好**。所以 DDP 不等整个 backward 结束，而是：

**某个桶的梯度一就绪，立刻在后台开始 AllReduce 通信；与此同时，前面的层还在继续 backward 计算。** 通信（搬数据）和计算（算梯度）**并行进行**，把通信时间"藏"在计算时间里。

```
时间轴 →
计算: [layerN backward][layerN-1 backward][layer N-2 ...][layer0]
通信:        └→[AllReduce bucket0]──┐
                      └→[AllReduce bucket1]──┐
                                  └→[AllReduce bucket2]
        ↑ 通信与后续层的计算"重叠", 不串行等待 ↑
理想情况下总耗时 ≈ max(计算时间, 通信时间) 而非两者之和
```

> 注意：**桶按"梯度就绪顺序"而非参数定义顺序装桶**，所以 DDP 期望每次反向计算图大体一致；动态控制流可能触发"某些参数没收到梯度"的报错（用 `find_unused_parameters=True` 缓解，但有性能代价）。

---

## 7. DistributedSampler：数据不能重叠

如果 4 个进程都读同样的数据，那只是把同一份训练重复 4 遍，毫无意义。**`DistributedSampler` 负责把数据集不重叠地切成 world_size 份**，每个 rank 只拿自己那份。

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
loader  = DataLoader(dataset, batch_size=bs, sampler=sampler)  # 不要再传 shuffle=True

for epoch in range(E):
    sampler.set_epoch(epoch)   # 关键! 否则每个 epoch 的 shuffle 顺序都一样
    for x, y in loader:
        ...
```

```
dataset = [0 1 2 3 4 5 6 7]   world_size=4
 rank0 → 0 4      rank1 → 1 5      rank2 → 2 6      rank3 → 3 7
 (交错切分, 各拿 1/4, 互不重叠)
```

`set_epoch(epoch)` 易漏：不调它，sampler 的随机种子每个 epoch 不变，导致每轮看到的样本顺序完全相同，削弱泛化。

---

## 8. 启动方式：怎么把脚本同时跑 N 份

你不会手动开 8 个终端。PyTorch 提供启动器，自动拉起 N 个进程并注入 `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*` 环境变量。

| 方式 | 状态 | 说明 |
|------|------|------|
| **`torchrun`** | ✅ 当前推荐 | PyTorch 官方启动器，自带容错/弹性伸缩 |
| `python -m torch.distributed.launch` | ⚠️ 旧版 | 已被 torchrun 取代，新代码勿用 |
| `mp.spawn` | ✅ 可用 | 在脚本内用代码起进程，单机调试方便 |

**torchrun 单机 4 卡：**

```bash
torchrun --nproc_per_node=4 train.py
```

**torchrun 多机（2 机各 4 卡，共 8 进程）：**

```bash
# node 0 (主节点)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
         --master_addr=192.168.1.10 --master_port=29500 train.py
# node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
         --master_addr=192.168.1.10 --master_port=29500 train.py
```

| 参数 | 含义 | 权衡/要点 |
|------|------|-----------|
| `--nproc_per_node` | 每台机器起几个进程 | 通常 = 本机 GPU 数 |
| `--nnodes` | 机器总数 | 单机填 1 |
| `--node_rank` | 当前是第几台机器 | 每台机器**不同**，主节点为 0 |
| `--master_addr/port` | 碰头点地址/端口 | 所有机器填**同一个**（主节点的）|

> 用 `torchrun` 时，脚本里读 `LOCAL_RANK` 直接用 `int(os.environ["LOCAL_RANK"])`，不要再像旧版那样从 `--local_rank` 命令行参数取。

**mp.spawn（脚本内启动，适合单机调试）：**

```python
import torch.multiprocessing as mp
def worker(local_rank, world_size): ...   # 每个进程跑这个函数
mp.spawn(worker, args=(world_size,), nprocs=world_size)
```

### 环境准备（沿用仓库原有 Docker 提示）

```bash
docker run -dt --name pytorch_env_cu117 --restart=always --gpus all \
  --network=host --shm-size 4G \
  -v /home/gdong/workspace/code:/workspace/code \
  -v /home/gdong/workspace/model:/workspace/model \
  -w /workspace pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel /bin/bash
docker exec -it pytorch_env_cu117 bash
```

> `--shm-size 4G` 很关键：DataLoader 多 worker 通过共享内存传数据，默认 64MB 太小会报 "bus error / shared memory" 崩溃。多机训练务必 `--network=host`，否则容器网络隔离会让 NCCL 连不通。

---

## 9. 数值例子 / 对照 / 实践

### 9.1 手算：每步通信量与时间（以 7B 模型为例）

设模型 70 亿参数，梯度用 FP16（2 字节/参数）。

- 梯度总大小：$7 \times 10^9 \times 2\text{ B} = 14\text{ GB}$。
- **Ring-AllReduce 单卡收发量** ≈ $2 \times$ 数据量 $= 2 \times 14 = 28\text{ GB}$（与卡数 N 几乎无关，这是环算法的精髓）。
- 若用 InfiniBand，单卡有效带宽**约 100 GB/s（以实测为准）**，则一次梯度同步约：

$$t_{\text{comm}} \approx \frac{28\text{ GB}}{100\text{ GB/s}} = 0.28\text{ s}$$

- 若反向传播计算耗时约 $0.4$ s，由于**计算通信重叠**，理想每步开销 $\approx \max(0.4, 0.28) = 0.4$ s，而非 $0.4 + 0.28 = 0.68$ s——重叠省下约 41%。这就是 6.4 节"把通信藏进计算"的价值。

> 数字为量级估算，实际受拓扑、NCCL 算法、是否跨机等影响，**以实测/官方为准**。

### 9.2 等效 batch size

每卡 `batch_size=32`，8 卡 ⇒ **全局有效 batch = 32 × 8 = 256**。换算学习率时常用**线性缩放**经验法则：batch 放大 k 倍，学习率约放大 k 倍（配 warmup）。这解释了"为什么多卡要调 lr"。

### 9.3 DataParallel(DP) vs DistributedDataParallel(DDP)

| 维度 | `DataParallel` (DP) | `DistributedDataParallel` (DDP) |
|------|---------------------|----------------------------------|
| 进程模型 | **单进程多线程** | **多进程**（每卡 1 进程）|
| 受 GIL 影响 | 是，Python 全局锁成瓶颈 | 否 |
| 通信方式 | 主卡 gather/scatter，**主卡是瓶颈** | Ring-AllReduce，**无中心瓶颈** |
| 多机 | 不支持 | 支持 |
| 负载均衡 | 主卡显存/算力占用更高 | 均衡 |
| 现状 | **已不推荐** | **标准做法** |

一句话：**DP 已过时，新代码一律用 DDP。** DP 把所有梯度汇到主卡再分发，主卡又算又通信，扩展性极差。

### 9.4 与更大模型的并行策略对照（导引）

| 策略 | 切什么 | 何时用 |
|------|--------|--------|
| **DDP（数据并行）** | 切**数据**，每卡全量模型 | 模型放得下单卡 → 本文重点 |
| **ZeRO / FSDP** | 切**优化器状态/梯度/参数** | 模型放不下，但想保持数据并行语义 |
| **张量并行 (TP)** | 切**单层权重矩阵** | 单层都太大 |
| **流水线并行 (PP)** | 切**层（模型纵向分段）** | 层数极多 |

详见 [[llm-train/pytorch/distribution/多机多卡]]。

### 9.5 实践要点清单

- ✅ 用 `local_rank` 选 GPU（`set_device`），用 `rank==0` 控制日志/保存。
- ✅ `DistributedSampler` 每个 epoch 调 `set_epoch`。
- ✅ 保存 checkpoint 只在 `rank==0` 做，加载后用 `dist.barrier()` 等齐。
- ✅ 计算"平均 loss/指标"时用 `dist.all_reduce` 跨进程归约后再除以 world_size。
- ✅ 退出前 `dist.destroy_process_group()`。
- ⚠️ 多机务必确认防火墙放行 `MASTER_PORT`、各机器时间/环境一致。

---

## 10. 常见问题

| 现象 | 根因 | 处理 |
|------|------|------|
| 程序在 `init_process_group` **静默卡死** | 有进程没起来 / world_size 不匹配 / 端口不通 | 核对启动进程数；查 `MASTER_PORT` 是否被占用或被防火墙挡 |
| `Address already in use` | `MASTER_PORT` 被上次残留进程占用 | 换端口或 `pkill` 残留进程 |
| `RuntimeError: NCCL error` / 跨机连不通 | 网卡选错、容器网络隔离、IB 未就绪 | `--network=host`；设 `NCCL_DEBUG=INFO` 看日志；必要时指定 `NCCL_SOCKET_IFNAME` |
| 显存够却报 "bus error / shared memory" | DataLoader 共享内存不足 | 增大 `--shm-size`（如 8G/16G）|
| 多卡 loss/精度与单卡明显不同 | 没按等效 batch 调 lr；或 BN 跨卡未同步 | 线性缩放 lr + warmup；用 `SyncBatchNorm` |
| 各 epoch 数据顺序完全一样 | 漏了 `sampler.set_epoch()` | 每个 epoch 开头调用 |
| "Expected to mark a variable ready only once" / 部分参数无梯度 | 动态控制流导致某些参数本步未参与 | `DDP(..., find_unused_parameters=True)`（有性能代价，能避则避）|
| checkpoint 被多进程互相覆盖/重复写 | 没限定只在 rank0 写 | `if dist.get_rank()==0: save(...)` |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总入口
- [[ai-framework/pytorch/README]] — PyTorch 框架基础（张量/autograd/nn）
- [[llm-train/pytorch/distribution/多机多卡]] — 跨节点扩展、FSDP/TP/PP 进阶
- [[ai-infra/网络/集合通信原语]] — AllReduce/Broadcast/AllGather 等原理与 NCCL 实现
