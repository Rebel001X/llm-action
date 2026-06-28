# PyTorch 分布式 API

> PyTorch `torch.distributed` 的核心 API 速查与机制解剖：进程组初始化、集合通信原语（all_reduce/broadcast/barrier）、DDP 数据并行包装、DistributedSampler 数据切分，以及 rank/local_rank 的来龙去脉。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/pytorch/distribution/README]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 主题 | 你会带走什么 |
|----|------|-------------|
| 0 | 一句话锚点 | distributed 到底是"谁在跟谁通信" |
| 1 | 地基：它解决什么问题 | 单卡放不下/算不快 → 多进程协作 |
| 2 | 进程组与 init_process_group | rank/world_size/backend/rendezvous 怎么对齐 |
| 3 | rank / local_rank / world_size | 三个最容易混的整数，配 ASCII 拓扑图 |
| 4 | 集合通信原语 | all_reduce / broadcast / reduce / all_gather / barrier 语义 |
| 5 | DistributedSampler | 数据如何不重不漏地切给每个 rank |
| 6 | DDP 包装 | 反向传播里梯度怎么被自动 all_reduce |
| 7 | 启动器 torchrun | 环境变量怎么注入、谁拉起谁 |
| 8 | 端到端最小训练循环 | 把以上拼成一个能跑的骨架 |
| - | 常见问题 / 跳转链接 | 排错与延伸 |

---

## 0. 一句话锚点

`torch.distributed` 是 **PyTorch 的多进程协作层**：把 $N$ 个独立的 Python 进程组织成一个"进程组（process group）"，让它们通过 **集合通信原语**（基于 NCCL/Gloo/MPI 后端）交换张量，从而实现 **数据并行 / 模型并行 / 流水线并行**。

最核心的一句话：**"每张 GPU 一个进程，每个进程跑同一份脚本，靠 rank 区分身份，靠集合通信对齐数据。"** 这叫 **SPMD**（Single Program Multiple Data）。

---

## 1. 地基：它解决什么问题

单卡训练有两个天花板：

1. **算不快**：一个 batch 在一张卡上跑得慢，吞吐上不去。
2. **放不下**：模型参数 + 激活 + 优化器状态超过单卡显存（如 70B 模型）。

分布式的基本思路是 **"分而治之 + 周期性同步"**：

```
            单卡                          多卡数据并行(DDP)
   ┌──────────────────┐         ┌─────────┐  ┌─────────┐
   │  全部数据顺序跑    │         │ rank0   │  │ rank1   │
   │  慢，但简单        │         │ 1/N数据 │  │ 1/N数据 │
   └──────────────────┘         │ 同一份  │  │ 同一份  │
                                │ 模型副本│  │ 模型副本│
                                └────┬────┘  └────┬────┘
                                     │  梯度同步   │
                                     └──all_reduce─┘
                                  (每步反向后求平均，参数保持一致)
```

关键洞察：**数据并行下，每个进程持有完整模型副本，只处理一部分数据；反向传播算出的是"局部梯度"，必须通过 `all_reduce` 求和/平均，让所有副本的参数更新方向一致，否则副本会发散。**

集合通信原语的底层语义见 [[ai-infra/网络/集合通信原语]]，这里聚焦 PyTorch 的 API 封装。

---

## 2. 进程组与 `init_process_group`

### 2.1 它是什么

进程组（ProcessGroup）是 distributed 的"通信句柄"。所有集合通信都发生在某个进程组上。默认进程组在 `init_process_group()` 调用时建立，包含所有 `world_size` 个进程。

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",          # 通信后端
    init_method="env://",    # rendezvous 方式：从环境变量读取
    world_size=8,            # 总进程数（可省略，由 env 提供）
    rank=0,                  # 本进程全局编号（可省略，由 env 提供）
)
```

### 2.2 四个关键参数的含义与权衡

| 参数 | 含义 | 权衡 / 选择 |
|------|------|------------|
| `backend` | 通信库 | `nccl`：GPU 间，**训练首选**，走 NVLink/IB；`gloo`：CPU 或调试，跨平台；`mpi`：需自编译，少用 |
| `init_method` | rendezvous（会合）方式 | `env://`：读 `MASTER_ADDR/MASTER_PORT`，配合 torchrun 最常用；`tcp://ip:port`：手动指定；`file://`：共享文件系统 |
| `world_size` | 进程总数 | = 总 GPU 数（典型）；决定集合通信的参与方数量 |
| `rank` | 本进程编号 | $0 \le \text{rank} < \text{world\_size}$，全局唯一 |

### 2.3 rendezvous（会合）机制 —— 进程怎么"找到彼此"

所有进程启动时互不认识，必须先约定一个"集合点"。`env://` + `MASTER_ADDR/MASTER_PORT` 就是这个集合点：

```
         所有进程读取相同的 MASTER_ADDR:MASTER_PORT
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ rank0   │──┐    │ rank1   │──┐    │ rank2   │──┐
   └─────────┘  │    └─────────┘  │    └─────────┘  │
                ▼                 ▼                 ▼
         ┌──────────────────────────────────────────┐
         │   rank0 所在机器的 MASTER_PORT (TCP store) │
         │   每个进程注册自己的地址，握手完成后        │
         │   建立全连接通信拓扑（NCCL ring/tree）      │
         └──────────────────────────────────────────┘
```

`init_process_group` 是 **阻塞** 的：它会等到 `world_size` 个进程全部 join 才返回。如果少一个进程，其余进程会在这里挂起，常表现为"卡住无报错"——这是新手最常见的坑。

> 精确的超时默认值、各后端能力差异以官方文档/源码为准；机制上务必记住"它会等齐所有 rank 才返回"。

---

## 3. rank / local_rank / world_size —— 三个整数别搞混

这三个整数是分布式编程的"坐标系"，理解它们就理解了 SPMD。

```
            一个 2 机 × 4 卡 = 8 进程的集群

   ┌─────────────── Node 0 (机器A) ───────────────┐
   │  GPU0      GPU1      GPU2      GPU3           │
   │ rank=0   rank=1   rank=2   rank=3            │
   │ local=0  local=1  local=2  local=3          │
   └───────────────────────────────────────────────┘
   ┌─────────────── Node 1 (机器B) ───────────────┐
   │  GPU0      GPU1      GPU2      GPU3           │
   │ rank=4   rank=5   rank=6   rank=7            │
   │ local=0  local=1  local=2  local=3          │
   └───────────────────────────────────────────────┘

   world_size = 8   (全局进程总数)
   rank       = 全局唯一编号 0..7
   local_rank = 节点内编号 0..3  →  用来 set_device(本机第几张卡)
```

| 概念 | 范围 | 作用 | 怎么拿 |
|------|------|------|--------|
| `world_size` | 固定常数 | 集合通信参与方数 / 数据切分份数 | `dist.get_world_size()` 或 `env WORLD_SIZE` |
| `rank`（global rank） | $0..N-1$ | 全局身份，决定"谁是 master(rank0)" | `dist.get_rank()` 或 `env RANK` |
| `local_rank` | $0..(每机卡数-1)$ | **绑定本机第几张 GPU** | `env LOCAL_RANK`（torchrun 注入） |

**核心用法**：用 `local_rank` 选卡，用 `rank` 区分逻辑职责。

```python
import os, torch
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)          # 关键！否则所有进程挤到 GPU0
device = torch.device(f"cuda:{local_rank}")

rank = dist.get_rank()
if rank == 0:                              # 只让 rank0 打日志/存 ckpt
    print("I am the master")
```

**为什么必须 `set_device(local_rank)`**：NCCL 默认每个进程占一张卡。如果不设，多个进程都默认 `cuda:0`，会争抢同一张卡导致 OOM 或通信死锁。

---

## 4. 集合通信原语（Collective Operations）

集合通信 = "一组进程共同参与的一次通信操作"，**所有 rank 必须都调用同一个原语**，否则会死锁。下面是 PyTorch 最常用的几个。完整数学语义参见 [[ai-infra/网络/集合通信原语]]。

### 4.1 `all_reduce` —— 最重要的一个

把所有 rank 上的张量按某种 op（默认 SUM）规约，**结果广播回每个 rank**（每个人都拿到同一个最终值）。这是 DDP 梯度同步的核心。

```
   规约前                      all_reduce(SUM) 后
 rank0: [1]                   rank0: [10]
 rank1: [2]      ────────►    rank1: [10]
 rank2: [3]                   rank2: [10]
 rank3: [4]                   rank3: [10]    (1+2+3+4=10, 人人持有)
```

```python
t = torch.tensor([rank + 1.0], device=device)
dist.all_reduce(t, op=dist.ReduceOp.SUM)   # 原地修改 t
# 求平均：再除以 world_size
t /= dist.get_world_size()
```

数值例子：4 个 rank 各算出梯度分量 $g_0=0.1, g_1=0.3, g_2=0.2, g_3=0.4$。`all_reduce(SUM)` 后每人得 $1.0$，再 $/4$ 得平均梯度 $0.25$。所有副本用 $0.25$ 更新 → 参数保持一致。

常用 `op`：`SUM`、`AVG`（部分后端支持）、`MAX`、`MIN`、`PRODUCT`。

### 4.2 `broadcast` —— 一对多分发

由 `src` rank 把张量发给所有其他 rank。典型用途：rank0 加载初始权重后广播给所有人，保证起点一致。

```
   broadcast(src=0)
 rank0: [W]  ──┬──►  rank1: [W]
               ├──►  rank2: [W]
               └──►  rank3: [W]   (其余 rank 原内容被覆盖)
```

```python
weights = load_ckpt() if rank == 0 else torch.empty(shape, device=device)
dist.broadcast(weights, src=0)
```

### 4.3 `reduce` —— 多对一规约

与 all_reduce 类似，但 **结果只落在 `dst` rank**，其余 rank 的张量未定义。常用于把统计量汇总到 rank0 打印。

```
   reduce(dst=0, SUM)
 rank0:[1] rank1:[2] rank2:[3] ──► rank0:[6]  (仅 dst 持有)
```

### 4.4 `all_gather` —— 收集拼接

每个 rank 贡献一个张量，所有 rank 都拿到 **拼接后的完整列表**。常用于评估时汇总各卡预测结果。

```
 rank0:[a] rank1:[b] rank2:[c] ─► 每个 rank 都得到 [a,b,c]
```

```python
gathered = [torch.empty_like(t) for _ in range(world_size)]
dist.all_gather(gathered, t)
```

### 4.5 `scatter` / `gather` —— 分发与收集（对偶）

`scatter`：src 把一个列表拆成 $N$ 份，第 $i$ 份发给 rank $i$。`gather`：每个 rank 的张量汇集到 dst 的列表里。是 broadcast/all_gather 的"分片"版本。

### 4.6 `barrier` —— 同步栅栏

**所有 rank 都到达 barrier 才一起放行**，不传数据，只做时序对齐。

```
   时间 ──────────────────────────►
 rank0: ████ barrier ░░░░░ (先到，等待)
 rank1: ██████████ barrier  (后到)
                    ▲
                    └─ 全员到齐，同时继续
```

```python
if rank == 0:
    download_dataset()      # 只让 rank0 下载
dist.barrier()              # 其余 rank 在此等 rank0 下载完
load_dataset()              # 之后大家一起读
```

用途：避免竞态。比如只让 rank0 创建目录/下载数据/写 ckpt，其余 rank 用 barrier 等它完成再继续。

### 4.7 阻塞 vs 异步

多数原语默认 **阻塞**（同步）。传 `async_op=True` 返回一个 work handle，可 `work.wait()` 显式等待，用于通信/计算重叠优化。

| 原语 | 输入 | 输出落点 | 典型用途 |
|------|------|---------|----------|
| `all_reduce` | 各 rank 张量 | 全员相同 | 梯度同步（DDP 核心） |
| `broadcast` | src 张量 | 全员相同 | 初始权重对齐 |
| `reduce` | 各 rank 张量 | 仅 dst | 汇总统计到 rank0 |
| `all_gather` | 各 rank 张量 | 全员持完整列表 | 评估汇总预测 |
| `scatter` | src 列表 | 各 rank 一片 | 手动数据分发 |
| `barrier` | 无 | 无（仅同步） | 时序对齐/防竞态 |

---

## 5. `DistributedSampler` —— 数据不重不漏地切分

### 5.1 它解决什么

数据并行下每个 rank 只该看 **不同的** 那 $1/N$ 数据。若所有 rank 都用普通 sampler，会看同样的数据 → 等价于 batch_size 没变，白白浪费多卡。`DistributedSampler` 负责把数据集按 rank 切片。

```
   数据集索引 [0,1,2,3,4,5,6,7]  world_size=4
   ┌────────────────────────────────────┐
   │ (可选 shuffle，由 epoch 控制随机种子) │
   └────────────────────────────────────┘
              切分（interleave 或分块）
   rank0 → [0,4]   rank1 → [1,5]
   rank2 → [2,6]   rank3 → [3,7]
   每个 rank 各自的 DataLoader 只迭代自己那份
```

### 5.2 用法与关键点

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,   # 默认从进程组读
    rank=rank,                 # 默认从进程组读
    shuffle=True,
    drop_last=False,
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# 注意：用了 sampler 就不能再传 shuffle=True 给 DataLoader

for epoch in range(epochs):
    sampler.set_epoch(epoch)   # 关键！否则每个 epoch 洗牌顺序相同
    for x, y in loader:
        ...
```

**两个必记要点**：

1. **`set_epoch(epoch)` 必须每个 epoch 调用**：sampler 用 `epoch` 作为随机种子的一部分，不调用则每个 epoch 的 shuffle 完全一样，破坏训练。
2. **补齐（padding）行为**：当数据集大小不能被 world_size 整除时，默认会 **重复补齐** 末尾样本，让每个 rank 拿到等量数据（保证集合通信对齐）。若不希望重复，设 `drop_last=True` 丢弃尾部。

数值例子：dataset 大小 10，world_size 4。$10 / 4 = 2.5$，默认向上取整每 rank 拿 3 条，总需 12 条 → 补 2 条重复样本；每 rank 严格 3 条，保证后续 all_reduce 步数一致。

---

## 6. DDP 包装（`DistributedDataParallel`）

### 6.1 它是什么、解决什么

`DDP` 是一个 **模型包装器**：把你的 `nn.Module` 包一层后，在反向传播时 **自动对梯度做 all_reduce**，让你几乎不用改训练代码就实现数据并行。它取代了老旧的 `DataParallel`（单进程多线程，受 GIL 限制、负载不均，已不推荐）。

### 6.2 整体架构与梯度同步流程

```
   forward:  每个 rank 用自己那份数据独立前向（无通信）
                          │
                          ▼
   backward: autograd 逐层算梯度
             ┌──────────────────────────────────────┐
             │ DDP 在每个参数的 grad ready 时触发 hook │
             │ 把梯度装进 "bucket"（按大小分桶）       │
             │ 桶满 → 立即异步 all_reduce 该桶         │
             │ （通信与后续层的反向计算重叠！）         │
             └──────────────────────────────────────┘
                          │
                          ▼
   optimizer.step(): 各 rank 用"已平均的梯度"更新
                     → 所有副本参数严格一致
```

**核心机制 —— gradient bucketing（梯度分桶 + 通信重叠）**：DDP 不会等所有梯度都算完才一次性 all_reduce，而是把参数分成若干"桶"。反向是从输出层往输入层算，某个桶的所有梯度先 ready 就 **立刻** 异步发起该桶的 all_reduce，与还在计算的前面层重叠。这把通信时间"藏"在计算时间里，是 DDP 高效的关键。

### 6.3 标准用法

```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 训练循环里和单卡几乎一样：
out = model(x)
loss = criterion(out, y)
loss.backward()        # 这里自动触发梯度 all_reduce
optimizer.step()
optimizer.zero_grad()
```

### 6.4 关键参数与权衡

| 参数 | 含义 | 权衡 |
|------|------|------|
| `device_ids=[local_rank]` | 本进程绑定的 GPU | 单进程单卡标准写法 |
| `find_unused_parameters` | 是否检测未参与前向的参数 | `True` 兼容动态图（如部分分支不走），但有额外开销；能 `False` 尽量 `False` |
| `gradient_as_bucket_view` | 梯度复用 bucket 内存 | 省显存，推荐开 |
| `broadcast_buffers` | 是否每步广播 buffer（如 BN 统计） | 有 BN 时相关 |
| `static_graph` | 声明计算图固定 | 可做更激进的重叠优化 |

> 上述参数的精确默认值/行为细节以官方文档为准；务必掌握的是"DDP 在 backward 自动 all_reduce 梯度、并用分桶重叠通信"这一机制。

### 6.5 必须保证的两个一致性

1. **初始参数一致**：DDP 构造时会自动 broadcast rank0 的参数/buffer 到所有 rank（所以你不必手动 broadcast 初始权重）。
2. **每步参与的计算图一致**：所有 rank 每步必须调用相同的 forward/backward，否则集合通信对不齐会死锁。

### 6.6 DDP vs DataParallel vs FSDP

| 方案 | 进程模型 | 显存 | 适用 |
|------|---------|------|------|
| `DataParallel` | 单进程多线程 | 模型全复制，rank0 显存偏高 | 已过时，不推荐 |
| `DDP` | 多进程，每进程一卡 | 每卡一份完整模型 | **数据并行主力**，模型放得下单卡时 |
| `FSDP` | 多进程 + 参数/梯度/优化器分片 | 显存大幅降低 | 大模型放不下单卡时（详见仓库 FSDP 相关章节） |

---

## 7. 启动器 `torchrun`

你不会手动 `python train.py` 八次再手填 rank。`torchrun`（旧名 `python -m torch.distributed.launch`）负责 **拉起多个进程并注入环境变量**。

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py

# 2 机各 4 卡（在每台机器上分别执行，node_rank 不同）
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
         --master_addr=10.0.0.1 --master_port=29500 train.py
```

它为每个子进程注入这些环境变量，脚本里直接读：

```
   torchrun
      │ 为每个子进程注入：
      ├── RANK         (全局 rank)
      ├── LOCAL_RANK   (本机第几张卡)
      ├── WORLD_SIZE   (总进程数)
      ├── MASTER_ADDR  (rank0 机器地址)
      └── MASTER_PORT  (会合端口)
      │
      ▼
   于是脚本里：dist.init_process_group(backend="nccl")  # init_method 默认 env://
            local_rank = int(os.environ["LOCAL_RANK"])
```

| 参数 | 含义 |
|------|------|
| `--nproc_per_node` | 每台机器起几个进程（通常 = 每机 GPU 数） |
| `--nnodes` | 机器数 |
| `--node_rank` | 本机是第几台（0 开始） |
| `--master_addr/port` | 会合点，所有机器填同一个 |

torchrun 还提供 **弹性容错**（elastic）：某进程挂了可重启/重组，比老 launch 强。

---

## 8. 端到端最小训练循环（把上面拼起来）

```python
import os, torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    # 1) 读环境 → 选卡 → 建进程组
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")        # init_method 默认 env://
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # 2) 数据切分
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 3) 模型 → DDP 包装（自动同步初始参数）
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) 训练循环
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)                   # 每 epoch 重洗
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()                        # ←—— 此处自动 all_reduce 梯度
            optimizer.step()

        # 5) 只让 rank0 存 ckpt，其余等它
        if rank == 0:
            torch.save(model.module.state_dict(), "ckpt.pt")  # 注意 .module 脱壳
        dist.barrier()

    # 6) 收尾
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

调用链总览：

```
 torchrun ─► 起 N 进程，注入 env
            └─► init_process_group  (rendezvous，等齐 N 个)
                 └─► set_device(local_rank)  绑卡
                      └─► DDP(model)  广播初始权重一致
                           └─► 循环: sampler.set_epoch → forward
                                      → backward(自动 all_reduce 梯度)
                                      → step → (rank0 存ckpt + barrier)
                                 └─► destroy_process_group  退出
```

几个易错点：保存时用 `model.module.state_dict()`（DDP 包了一层 `.module`）；只在 rank0 落盘并 `barrier`；`loss` 打印若要全局平均需自己 `all_reduce`。

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| 程序卡在 `init_process_group` 不动 | 少了一个进程没 join / MASTER_ADDR 端口不通 | 检查 `--nproc_per_node`、防火墙、所有机器 MASTER 一致 |
| 所有进程挤 GPU0，OOM | 没 `set_device(local_rank)` | init 前先 `torch.cuda.set_device(local_rank)` |
| 多卡和单卡精度/loss 不对 | 没 `sampler.set_epoch`；或梯度没平均 | 每 epoch `set_epoch`；DDP 已自动平均，手动同步则记得 `/world_size` |
| 死锁、某些 rank 卡住 | 各 rank 调用的集合通信不一致（如 if 分支只有部分 rank 走 all_reduce） | 保证所有 rank 每步走相同集合通信调用路径 |
| `find_unused_parameters` 报错/慢 | 有参数未参与前向 | 改模型结构，或临时设 `find_unused_parameters=True`（有开销） |
| 存的 ckpt load 不回单卡 | DDP 多了 `module.` 前缀 | 存 `model.module.state_dict()` 或 load 时 strip 前缀 |
| 每张卡 batch=32，总有效 batch 是多少 | 数据并行 batch 线性放大 | 全局有效 batch $= 32 \times \text{world\_size}$，lr 通常相应调整 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-train/pytorch/distribution/README]] — 本目录分布式训练总览
- [[ai-infra/网络/集合通信原语]] — all_reduce/broadcast 等的底层算法（ring/tree、带宽分析）
