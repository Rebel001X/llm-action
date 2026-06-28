# torchrun(PyTorch 弹性分布式启动器)

> torchrun 是 PyTorch 官方的分布式训练「进程拉起器 + 弹性容错协调器」，把"启动 N 个进程、给每个进程发对的环境变量、让它们找到彼此组队"这件繁琐的事自动化。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] [[ai-infra/网络/NCCL]] [[ai-framework/megatron-lm/README]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 小节 | 你会得到什么 | 难度 |
| --- | --- | --- |
| 0. 一句话锚点 | torchrun 到底替你做了什么 | ★ |
| 1. 地基：为什么需要启动器 | 没有它你要手写一堆 env | ★ |
| 2. 进程组与 rank 体系 | world/local/global rank、node rank | ★★ |
| 3. torchrun 注入的环境变量 | 训练脚本怎么读到这些值 | ★★ |
| 4. Rendezvous(集合点)机制 | 多机怎么"组队握手" | ★★★ |
| 5. 弹性与容错 | restart、min/max nodes、缩扩容 | ★★★ |
| 6. 调用链与时序图 | 从命令到 NCCL 通信 | ★★ |
| 7. 参数说明 | 各参数做什么、怎么权衡 | ★★ |
| 8. 单机/多机示例 | 复制即用的骨架 | ★★ |
| 常见坑 | 端口冲突、网卡、hang | ★★★ |

参考：<https://pytorch.org/docs/stable/elastic/run.html>

---

## 0. 一句话锚点

**torchrun = 多进程拉起器(spawn N 个训练进程) + 环境变量注入器(告诉每个进程"你是几号、一共几个、去哪找老大") + 弹性协调器(节点挂了能重启/缩扩容)。**

它是旧命令 `python -m torch.distributed.launch` 的官方替代品。你的训练脚本几乎不用改，只需在脚本里读 torchrun 注入的环境变量来初始化进程组。

---

## 1. 地基：为什么需要一个"启动器"

分布式数据并行(DDP)的本质：**同一份训练脚本，在多张 GPU / 多台机器上各跑一个进程**，每个进程算自己那份数据的梯度，再用集合通信(AllReduce)把梯度求平均，保证所有副本的模型权重始终一致。

要让这些进程"组成一个团队"，每个进程必须知道四件事：

1. **我是谁**(我的全局编号 rank)。
2. **一共几个人**(world size)。
3. **我用本机第几张卡**(local rank)。
4. **去哪里和大家握手**(master 的 IP + 端口)。

如果手工启动，你得对每个进程手动 `export RANK=... WORLD_SIZE=... MASTER_ADDR=...` 再 `python train.py`，8 卡就要写 8 遍，2 机 16 卡更是噩梦，还容易写错。**torchrun 把这套环境变量的计算和注入全自动化**，并在此基础上加了容错。

```
没有 torchrun(手工)：              有 torchrun：
┌──────────────────────────┐      ┌──────────────────────────────┐
│ RANK=0 ... python train  │      │ torchrun --nproc-per-node=8 \│
│ RANK=1 ... python train  │  →   │          train.py             │
│ RANK=2 ... python train  │      │  ↳ 自动 fork 8 个进程         │
│ ...(写 8 遍,易错)        │      │  ↳ 自动注入 RANK/WORLD_SIZE..│
└──────────────────────────┘      └──────────────────────────────┘
```

---

## 2. 进程组与 rank 体系(核心概念,先吃透)

理解 torchrun 的关键是分清几个"编号"。设：2 台机器，每台 4 张 GPU，共 8 个进程。

| 概念 | 含义 | 取值范围(本例) |
| --- | --- | --- |
| **world size** | 全局进程总数 | 8 |
| **(global) rank** | 进程的全局唯一编号 | 0~7 |
| **local rank** | 进程在**本机内**的编号 | 0~3 |
| **node rank** | 机器(节点)的编号 | 0~1 |
| **nproc_per_node** | 每台机器起几个进程 | 4(通常=本机 GPU 数) |
| **nnodes** | 机器数 | 2 |

它们的关系（最常见的"每进程一卡"摆放）：

$$\text{global\_rank} = \text{node\_rank} \times \text{nproc\_per\_node} + \text{local\_rank}$$

```
        Node 0 (node_rank=0)            Node 1 (node_rank=1)
   ┌───────────────────────────┐  ┌───────────────────────────┐
   │ GPU0  GPU1  GPU2  GPU3     │  │ GPU0  GPU1  GPU2  GPU3     │
   │ rank0 rank1 rank2 rank3    │  │ rank4 rank5 rank6 rank7    │  ← global rank
   │ lr0   lr1   lr2   lr3      │  │ lr0   lr1   lr2   lr3      │  ← local rank
   └───────────────────────────┘  └───────────────────────────┘
                         world_size = 8
```

**为什么要区分 global rank 和 local rank？**
- **global rank**：决定数据分片(`DistributedSampler` 用它切数据)、决定谁是"主进程"(常约定 rank0 负责打日志、存 checkpoint、跑验证)。
- **local rank**：决定**绑哪张物理卡**——`torch.cuda.set_device(local_rank)`。卡是本机资源，所以要用本机编号。绝不能用 global rank 去 set_device，否则 rank4 会试图绑到本机不存在的 GPU4。

---

## 3. torchrun 注入的环境变量(脚本怎么读)

torchrun 在拉起每个进程前，会给它设置一组**环境变量**。你的脚本从 `os.environ` 读它们来初始化。常见的有：

| 环境变量(类别) | 作用 | 典型读法 |
| --- | --- | --- |
| `RANK` | 全局 rank | `int(os.environ["RANK"])` |
| `LOCAL_RANK` | 本机内 rank → 绑卡 | `int(os.environ["LOCAL_RANK"])` |
| `WORLD_SIZE` | 全局进程数 | `int(os.environ["WORLD_SIZE"])` |
| `MASTER_ADDR` / `MASTER_PORT` | 主节点地址/端口(握手用) | 一般 `init_process_group` 自动读 |
| `LOCAL_WORLD_SIZE` | 本机进程数 | 用于本机内逻辑 |

> 注意：上面是 torchrun 约定注入的变量类别，**确切名称与是否存在以官方文档/版本为准**。推荐脚本里读 `LOCAL_RANK` 而不是自己推算，因为弹性场景下编号可能变化。

典型脚本骨架（DDP）：

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ["LOCAL_RANK"])     # torchrun 注入
torch.cuda.set_device(local_rank)              # 关键：用 local_rank 绑卡
dist.init_process_group(backend="nccl")        # 自动读 RANK/WORLD_SIZE/MASTER_*

model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
# ... 训练循环 ...
dist.destroy_process_group()
```

注意 `init_process_group` **没有手动传 rank/world_size**——因为 torchrun 已把它们放进环境变量，PyTorch 会自动读取。这正是 torchrun 相对手工启动最省心的地方。

---

## 4. Rendezvous(集合点)：多机怎么"组队握手"

多机最难的一步是：分散在不同机器上的进程，如何在启动初期**互相发现、确认人数、选出谁是协调者**？这套机制叫 **Rendezvous(集合点 / 会合)**。

可以把它想成"线上开会前的点名"：所有人都连到同一个**会合点**(rendezvous endpoint)，报上**同一个会议号**(rendezvous id)，系统等够人数后宣布"开会"，并给每个人分配座位号(rank)。

```
       rdzv-endpoint = host0:29400   (会合点,通常在 node0)
                 ┌──────────────┐
                 │  Rendezvous  │  ← 大家都连到这里
                 │   (c10d store)│     报上同一个 rdzv-id
                 └──────┬───────┘
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Node0 进程们     Node1 进程们     Node2 进程们
   "我到了"         "我到了"         "我到了"
        └───── 凑够 nnodes → 分配 rank → 同时开跑 ─────┘
```

三个关键参数：

- **`--rdzv-id`**：本次任务的唯一 ID，**所有节点必须填同一个**，用来区分不同的训练任务(避免两个任务的进程串到一起)。
- **`--rdzv-backend`**：集合点的实现后端，即"用什么充当那个会合服务器"。
- **`--rdzv-endpoint`**：会合服务器所在的 `host:port`，通常指向 0 号节点。

**rdzv-backend 可选项(类别理解)：**

| backend | 说明 | 取舍 |
| --- | --- | --- |
| `c10d`(推荐) | PyTorch 内置的轻量 TCP store，无需额外部署 | 默认首选，省事 |
| `etcd-v2` / `etcd`(legacy) | 用外部 etcd 服务做协调，需自己搭 etcd | 大规模/已有 etcd 基础设施时考虑 |

> 用 `c10d` 时，endpoint 指向的那台机器(常是 node0)会在指定端口起一个 TCP store，其他节点连过来即可，**不必额外安装任何服务**。要用 etcd 系，需先搭好 etcd 服务器（如启用 v2 api：`--enable-v2`）。具体后端列表以官方文档为准。

---

## 5. 弹性与容错(elastic：torchrun 的"杀手锏")

旧的 `launch` 一旦某个进程崩了，整个任务就挂。torchrun 源自 **TorchElastic**，支持：

1. **进程失败重启**：`--max-restarts=N` 允许整组进程在失败后重新 rendezvous 并重启最多 N 次（配合脚本里定期存 checkpoint，可从最近状态恢复）。
2. **弹性伸缩(min/max nodes)**：`--nnodes=MIN:MAX` 形式，允许节点数在区间内变化。节点掉线后用剩余节点继续；新节点加入后自动并入。
3. **成员变动 = 重新 rendezvous**：每次伸缩/重启，都会重新点名、重新分配 rank。所以**rank 不是固定的**——这就是为什么要从环境变量读、不要硬编码。

```
        正常运行(3 nodes)
   N0 ── N1 ── N2     world_size = 24 (8/node)
            │
   N1 崩溃  ▼  触发 re-rendezvous (max-restarts 内)
   N0 ──────── N2     重新点名, world_size = 16, 从 checkpoint 续跑
            │
   N1 修复加回 ▼
   N0 ── N1 ── N2     再次 re-rendezvous, world_size = 24
```

**代价/注意**：弹性恢复依赖**你自己在脚本里实现 checkpoint 保存与加载**——torchrun 只负责重新拉进程和点名，**不会帮你恢复模型/优化器状态**。world_size 变化还会改变全局 batch size 与学习率缩放，需要训练逻辑能容忍。

---

## 6. 端到端调用链 / 时序图

```
你敲下命令
   │
   ▼
torchrun (agent 进程, 每台机器各一个)
   │  1. 解析 --nnodes/--nproc-per-node/--rdzv-*
   │  2. 各 agent 连到 rdzv-endpoint, 报 rdzv-id
   ▼
[ Rendezvous 点名 ]  ── 等够节点 → 分配 node_rank / 计算 global rank 段
   │
   ▼  每台机器的 agent 各 fork nproc_per_node 个 worker
worker 进程 (你的 train.py)
   │  3. 读 env: RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR ...
   │  4. torch.cuda.set_device(LOCAL_RANK)
   │  5. dist.init_process_group("nccl")  ←—— 建立通信子
   ▼
[ NCCL 通信子建立 ]  ── 各 rank 互联, 走 NVLink / IB / 网卡
   │
   ▼
训练循环: forward → backward → AllReduce 梯度 → optimizer.step
   │
   ▼ (若某 worker 崩) agent 感知 → 回到 [Rendezvous] 重来 (max-restarts 内)
```

要点：**每台机器跑一个 torchrun(agent)**，agent 负责本机 worker 的拉起与监控；多台 agent 通过 rendezvous 互联。NCCL 通信由 PyTorch 在 `init_process_group` 时建立，torchrun 只负责"把进程摆好、把环境变量发好"。

---

## 7. 参数说明(讲含义,默认值以官方为准)

| 参数(类别) | 作用 | 怎么权衡 |
| --- | --- | --- |
| `--nproc-per-node` | 本机起几个 worker | 通常 = 本机 GPU 数(每进程一卡)。也可填 `gpu`/`auto` 让其自动探测(以版本支持为准) |
| `--nnodes` | 机器数；支持 `MIN:MAX` 弹性区间 | 固定数=刚性任务；`2:4`=允许 2~4 台弹性伸缩 |
| `--node-rank` | 当前机器编号 | 静态(非弹性)多机时**每台手动指定不同值**(0,1,2…)；rendezvous 模式下自动分配 |
| `--rdzv-id` | 任务唯一 ID | 所有节点填**同一个**；不同任务用不同值避免串台 |
| `--rdzv-backend` | 集合点后端 | `c10d` 免部署、首选；etcd 系适合已有基础设施 |
| `--rdzv-endpoint` | 会合点 `host:port` | 指向 node0；端口要在所有节点间可达 |
| `--max-restarts` | 失败后最多重启次数 | 0=不容错；调大=更耐故障但可能反复重启掩盖真 bug |
| `--master-addr`/`--master-port`(旧式) | 主节点地址/端口 | 静态模式用；rendezvous 模式改用 `--rdzv-endpoint` |
| `--standalone` | 单机快捷模式 | 本机调试时省去手填 rdzv-* |

> 提醒：参数既有连字符写法(`--nproc-per-node`)也有下划线写法(`--nproc_per_node`)，确切拼写、默认值、是否仍保留旧参数，请以你使用版本的 `torchrun --help` 与官方文档为准，**不要凭记忆硬编默认值**。

---

## 8. 典型示例(复制即用骨架)

**(A) 单机多卡(最常用,调试首选)**

```bash
# standalone 模式:本机 8 卡,无需操心 rdzv
torchrun --standalone --nproc-per-node=8 train.py --epochs 3
```

**(B) 多机多卡(rendezvous 模式,推荐)**

在**每台机器**上跑同一条命令（`$HOST_NODE_ADDR` 指向 node0 的 `host:port`）：

```bash
torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_TRAINERS \
    --max-restarts=3 \
    --rdzv-id=$JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$HOST_NODE_ADDR \
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args ...)
```

多机训练你需要指定：

- **`--rdzv-id`**：唯一 job id（参与该任务的所有节点共享）。
- **`--rdzv-backend`**：`RendezvousHandler` 的一种实现(默认 `c10d`，推荐)。
- **`--rdzv-endpoint`**：rendezvous 后端运行的端点，通常形如 `host:port`。

目前开箱支持的 rendezvous 后端：`c10d`(推荐)、`etcd-v2` 和 `etcd`(legacy)。要使用 etcd-v2 / etcd，需先搭一个启用了 v2 api 的 etcd 服务器(如 `--enable-v2`)。

**(C) 弹性伸缩(允许 2~4 台)**

```bash
torchrun --nnodes=2:4 --nproc-per-node=8 \
         --rdzv-id=elastic-job --rdzv-backend=c10d \
         --rdzv-endpoint=node0:29400 --max-restarts=5 \
         train.py
```

---

## 常见问题 / 坑

| 现象 | 根因 | 解法 |
| --- | --- | --- |
| 启动卡住不动(hang) | 各节点 `rdzv-id` 不一致 / 防火墙挡了 rdzv 端口 / 凑不够 `nnodes` | 统一 id；放行端口；确认节点都启动 |
| `Address already in use` | rdzv/master 端口被占用(上次任务残留) | 换端口或清理残留进程 |
| 绑到错误的卡 / CUDA error | 用了 `RANK` 而非 `LOCAL_RANK` 去 `set_device` | 一律用 `LOCAL_RANK` 绑卡 |
| 多机但走错网卡(慢/不通) | NCCL 选了管理网卡而非 IB/高速网卡 | 设 `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` 指定网卡(详见 [[ai-infra/网络/NCCL]]) |
| 节点崩后没续上 | 脚本没存 checkpoint，restart 也无状态可恢复 | 脚本内定期保存并在启动时加载 checkpoint |
| 日志重复打印 8 遍 | 每个 rank 都打日志 | 仅在 `rank==0` 打印/存盘 |
| 旧脚本读不到 rank | 还在用 `torch.distributed.launch` 的 `--local_rank` 入参方式 | 迁移到从 `os.environ["LOCAL_RANK"]` 读 |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-train/pytorch/distribution/README]]
- [[ai-infra/网络/NCCL]]
- [[ai-infra/网络/集合通信原语]]
- [[ai-framework/megatron-lm/README]]
