# PyTorch 分布式训练 API：init_process_group 与 torchrun 启动

> 一句话定位：`torch.distributed` 的两个核心抓手——用 `init_process_group()` **建立进程间通信组**，用 `torchrun` **拉起并编号每个进程**；二者一上一下，缺一不可。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]] · [[ai-framework/pytorch/README]]

## 阅读地图

| 小节 | 你会得到 | 关键词 |
|---|---|---|
| 0. 一句话锚点 | 一句话记住全篇 | 通信组 + 进程编号 |
| 1. 地基/前置 | 进程、rank、world_size 的物理含义 | SPMD、rank |
| 2. init_process_group 全参数 | 每个参数干什么、为什么需要 | backend / init_method / rank / world_size |
| 3. 四种 backend 怎么选 | NCCL/Gloo/MPI/HCCL 对照决策 | 选型对照表 |
| 4. 为什么从 mpi 改到 torchrun | 历史演进与 slurm/mpi 的坑 | launch → torchrun |
| 5. torchrun 多机命令逐参数拆 | 跑通 3 机 12 卡 | nproc/nnodes/node_rank |
| 6. rank 是怎么被算出来的 | 全局 rank 公式 + 手算 | local_rank → global_rank |
| 实操命令 | 原文真料：完整命令 + 参数表 | 可直接抄 |
| 常见坑 | 卡住/连不上的排查表 | NCCL hang / 端口 |

## 0. 一句话锚点

分布式训练 = **N 个独立进程，各管一块（或几块）GPU，靠一张"通信组的花名册"互相找到对方做 all-reduce**。

- **谁来发花名册、点名编号**：`torchrun`（启动器，进程外）。
- **谁拿着花名册建立连接、约定用什么传输协议**：`init_process_group()`（进程内第一行分布式代码）。

记住这张图就够了：

```
        torchrun  (启动器/点名)
            │  注入环境变量 RANK / WORLD_SIZE / MASTER_ADDR ...
            ▼
   ┌───────────────────────────────────────────┐
   │  python 进程 0   进程 1   ...   进程 N-1     │
   │      │            │                │        │
   │  init_process_group(backend="nccl") 各自调用 │
   │      └──────┬─────┴────────┬───────┘        │
   │             ▼              ▼                 │
   │        握手成功，组成一个 "通信组"           │
   │        之后 all_reduce / broadcast 才能用    │
   └───────────────────────────────────────────┘
```

## 1. 地基/前置：rank、world_size、进程到底是什么

分布式训练用的是 **SPMD（Single Program Multiple Data）** 模型：**同一份脚本被启动成多个进程**，每个进程跑完全相同的代码，只是通过"自己的编号"决定处理哪一份数据 / 管哪一块 GPU。

把概念拆成原子：

| 概念 | 物理含义 | 谁告诉它 |
|---|---|---|
| **进程（process）** | 一个独立的 Python 解释器实例，通常绑定 1 块 GPU | OS / torchrun |
| **rank（全局排名）** | 这个进程在所有进程里的唯一编号，0 ~ world_size-1 | torchrun 注入 `RANK` |
| **local_rank** | 进程在**本机内部**的编号，0 ~ (本机GPU数-1)，用来选 `cuda:local_rank` | torchrun 注入 `LOCAL_RANK` |
| **world_size** | 全部进程总数（= 节点数 × 每节点进程数） | torchrun 注入 `WORLD_SIZE` |
| **node / node_rank** | 一台物理机器 / 机器编号 | 命令行 `--node_rank` |
| **master（主节点）** | 进程 rank 0 所在机器，负责"集合点"（rendezvous） | `--master_addr/port` |

为什么需要 rank？因为 SPMD 下所有进程代码相同，**唯一能区分"我是谁"的就是 rank**。`if rank == 0: save_checkpoint()` 这种写法就靠它。

```
    3 个节点，每节点 4 卡 → world_size = 12

  node02 (node_rank=0)   node03 (node_rank=1)   node04 (node_rank=2)
  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │ GPU0 GPU1 GPU2 GPU3│  │ GPU0 GPU1 GPU2 GPU3│  │ GPU0 GPU1 GPU2 GPU3│
  │ r0   r1   r2   r3 │   │ r4   r5   r6   r7 │   │ r8   r9   r10  r11│
  │ L0   L1   L2   L3 │   │ L0   L1   L2   L3 │   │ L0   L1   L2   L3 │
  └─────────────────┘    └─────────────────┘    └─────────────────┘
   r=global rank          L=local_rank
```

## 2. init_process_group()：建立通信组的那一行

> `dist.init_process_group()` 是 PyTorch 中用于初始化分布式训练的函数之一。它用于设置并行训练环境，连接多个进程以进行数据和模型的分布式处理。我们通过 `init_process_group()` 这个方法来进行初始化。

它做的事，本质是**让所有进程通过一个"集合点（master）"互相交换地址，然后建立 P2P 连接，组成一个通信组（process group）**。在这一行返回之前，任何 `all_reduce` / `broadcast` 都不能用。

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",      # 用什么库做通信（见第 3 节）
    init_method="env://" # 从环境变量读 MASTER_ADDR / RANK / WORLD_SIZE
)
```

### 2.1 参数逐个拆（原文真料保留 + 解释为什么）

| 参数 | 必需? | 取值 | 为什么需要它 |
|---|---|---|---|
| **backend** | ✅ 必需 | `gloo` / `mpi` / `nccl` / `hccl`（旧版还有 `tcp`） | 决定底层用哪套通信库；GPU 走 NCCL，CPU 走 Gloo |
| **init_method** | 可选 | `env://` / `file://...` / `tcp://ip:port` | 进程**初次如何找到彼此**的"集合方式"，默认 `env://` |
| **rank** | 可选 | 当前进程排名，从 0 开始 | 标识"我是谁"；用 `env://` 时可省，从 `RANK` 读 |
| **world_size** | 可选 | 总进程数 | 让每个进程知道"要等齐多少人"；省略时从 `WORLD_SIZE` 读 |
| **timeout** | 可选 | 初始化/集合操作超时时间 | 超时即报错，避免无限 hang（NCCL 默认很长） |
| **group_name** | 可选 | 进程组名称 | 多组通信时区分用 |

> 原文 backend 取值清单（保留）：
> - `tcp`：使用 TCP 协议进行通信。
> - `gloo`：使用 Gloo 库进行通信。
> - `mpi`：使用 MPI（Message Passing Interface）进行通信。
> - `nccl`：使用 NCCL 库进行通信（适用于多 GPU 的分布式训练）。
> - `hccl`：使用 HCCL 库进行通信（适用于华为昇腾 AI 处理器的分布式训练）。

> 原文 init_method 取值清单（保留）：
> - `env://`：使用环境变量中指定的方法进行初始化。
> - `file://`：使用本地文件进行初始化。
> - `tcp://:`：使用 TCP 地址和端口进行初始化。
> - `gloo://:`：使用 Gloo 地址和端口进行初始化。
> - `mpi://:`：使用 MPI 地址和端口进行初始化。

### 2.2 三种 init_method 的"集合点"机制对照

`init_method` 解决的核心问题：**进程刚启动时彼此还不认识，怎么"对暗号"完成第一次握手（rendezvous）？**

```
env://                        file://                   tcp://
─────────                     ─────────                 ─────────
每个进程读环境变量            所有进程读写同一            所有进程连到
MASTER_ADDR/PORT/RANK         共享文件做交换             同一个 ip:port
        │                          │                        │
   torchrun 帮你填好           需共享文件系统(NFS)       手动指定地址端口
   ✅ 最常用                   ⚠ 文件残留要清理          适合无共享盘
```

| init_method | 怎么集合 | 适用场景 | 坑 |
|---|---|---|---|
| `env://` | 读 `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE` 环境变量 | 配合 torchrun，最主流 | 环境变量没设全会卡住 |
| `file://path` | 多进程读写同一个共享文件 | 单机或有共享存储 | 上次跑崩的文件没删，下次复用会报错 |
| `tcp://ip:port` | 全部连到指定地址端口 | 无共享盘、手动控制 | 端口被占用/防火墙挡住 |

## 3. 四种 backend 怎么选：NCCL / Gloo / MPI / HCCL

这是工程里最常踩的选择题。原文给出的官方决策逻辑（**完整保留**）：

> - 对于分布式 **GPU** 训练，使用 **NCCL** 后端。
> - 对于分布式 **CPU** 训练，使用 **Gloo** 后端。
> - 如果你的主机是 GPU 主机并且具有 **InfiniBand** 互连：使用 **NCCL**，因为它是目前唯一支持 InfiniBand 和 GPUDirect 的后端。
> - 如果你的主机是 GPU 主机并且具有**以太网**互连：使用 **NCCL**，因为它目前提供了最好的分布式 GPU 训练性能，特别是对于多进程单节点或多节点分布式训练。
> - 如果你遇到 NCCL 的任何问题，使用 **Gloo** 作为备选选项。（注意，Gloo 目前运行速度比 NCCL 慢）
> - 如果你的主机是 **CPU** 主机并且具有 InfiniBand 互连：如果你的 InfiniBand 启用了 IP over IB，使用 **Gloo**，否则，使用 **MPI**。我们计划在即将发布的版本中为 Gloo 添加 InfiniBand 支持。
> - 如果你的主机是 CPU 主机并且具有以太网互连：使用 **Gloo**，除非你有特定的理由使用 MPI。

> 结论（原文）：**MPI 是最不推荐使用的一种方法，对于英伟达的显卡，最优先的还是使用 NCCL 方法。**

### 3.1 决策树（一图选完）

```
                你在做分布式训练，选 backend ?
                          │
            ┌─────────────┴─────────────┐
          GPU 训练?                    CPU 训练?
            │                            │
          NCCL ✅              ┌─────────┴──────────┐
   (IB / 以太网都用 NCCL)   IB 且开了 IP-over-IB?   以太网
            │                   │          │          │
   NCCL 出问题? → Gloo 兜底    Gloo      纯 IB→MPI    Gloo ✅
                                                    (除非特殊理由用 MPI)

   华为昇腾 NPU? → HCCL（NCCL 的昇腾对应物）
```

### 3.2 对照表

| backend | 设备 | 互连 | 速度 | 何时用 |
|---|---|---|---|---|
| **NCCL** | NVIDIA GPU | IB / 以太网 | 最快，支持 GPUDirect/IB | GPU 训练首选 |
| **Gloo** | CPU（也支持部分 GPU 操作） | 以太网 / IP-over-IB | 比 NCCL 慢 | CPU 训练、NCCL 兜底 |
| **MPI** | CPU | 纯 IB | 取决于实现 | 几乎不推荐；需自行编译 PyTorch |
| **HCCL** | 华为昇腾 NPU | — | — | 昇腾平台 |

> 为什么 NCCL 在 GPU 上无敌：它直接在 GPU 显存间做 ring/tree all-reduce，支持 **GPUDirect RDMA**（数据不经过 CPU 内存、不过 PCIe 拷到主机），所以 InfiniBand + NCCL 是大模型多机训练的事实标准。详见 [[ai-infra/网络/NCCL]] 与 [[ai-infra/网络/集合通信原语]]。

## 4. 为什么放弃 MPI、改用 torchrun（原文真实踩坑）

原文记录了一段真实的技术选型过程，**完整保留并解释**：

> 这里由于服务器采用的 **slurm 系统**，我们开始计划使用 mpi 去实现分布式分发，同时 torch 的初始化也支持 mpi，原始想法是通过 `mpirun` 来进行分布式计算。**但是，如果要使用 mpi 来实现分布式功能，必须要通过 GitHub 上的源代码进行编译，通过 conda 和 pip 进行下载的 PyTorch 自身是不携带 mpi 的。**

这是一个关键坑：**官方预编译的 PyTorch（pip/conda 装的）不带 MPI 后端**。想用 `backend="mpi"`，你必须在装好 MPI 的环境里从源码重新编译 PyTorch——成本极高、收益极低。

> 和 MPI 相匹配的有一种 torch 官方自带的方法，在 **torch 2.0 之前**使用的 API 叫 **`torch.distributed.launch`**，在使用时显示未来的版本将会弃用这个 API，取而代之的是 **`torchrun`**。因此我们将命令由 mpi 改为 torchrun 方法，在 dist 初始化使用 **nccl** 后端通信。

启动器的演进史：

```
  mpirun                torch.distributed.launch        torchrun
  (需自编译 PyTorch)  →  (torch<2.0, 已弃用警告)      →  (现役推荐)
       ✗ 麻烦              ⚠ deprecated                  ✅ 内置弹性容错
```

- `torch.distributed.launch` 和 `torchrun` 都是**启动器**，作用是：批量拉起 N 个进程，并给每个进程注入 `RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT` 等环境变量——这样进程里写 `init_method="env://"` 就能自动集合。
- `torchrun` 相比老的 `launch` 多了**弹性训练（elastic）/ 故障重启 / rendezvous** 能力，且不再需要手动传 `--use_env`。

## 5. torchrun 多机命令逐参数拆解

原文给了一个非常具体的 **3 节点 × 4 卡** 场景，逐字保留并图解。

> 假设我们有三个节点 node02、node03、node04，每个节点上有四张 GPU。现在我们将官方测试文档中的代码写为 `test_mpi.py`。最终通过 torchrun 实现的命令如下。

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=3 \
  --node_rank=0 \
  --master_addr=192.168.0.101 \
  --master_port=29500 \
  test_mpi.py
```

> 我们没有必要和 torchrun 的官方文档一样去设置 `--rdzv-backend` 和 `--rdzv-id`，因为这不是必须的，用默认的即可。我们只需要设置的参数只有上面这几个。

### 5.1 参数表（原文释义保留 + 补充）

| 参数 | 原文含义 | 本例值 | 补充说明 |
|---|---|---|---|
| `--nproc_per_node` | 每个节点上的进程数 | 4 | 通常 = 每机 GPU 数；每进程绑一块卡 |
| `--nnodes` | 总节点数 | 3 | 总进程 world_size = 4×3 = 12 |
| `--node_rank` | 当前节点排名，从 0 开始 | 0 | **每台机器上的值不同**，手动改 |
| `--master_addr` | 主节点 IP，用于协调 | 192.168.0.101 | 必须是 node_rank=0 那台机的 IP |
| `--master_port` | 主节点端口，用于通信 | 29500 | 所有机器填同一个；别被占用 |

### 5.2 三台机器分别敲什么（原文流程保留）

> 该命令将在 3 个节点的分布式环境中启动 4 个进程……主节点的地址的 `--node_rank` 必须设置为 0，也就是上述这行命令，必须要先在主节点上运行。

> 举个例子，假如我的主节点是 node02，那么我就要先在 node02 节点的终端上运行上述 torchrun 命令，同时 `--master_addr` 要为 node02 的 IP 地址（查看 IP 地址可以通过 `ip addr`），然后 node03、node04 的顺序就不重要了，在其节点的终端上将 `--node_rank=0` 改为 `--node_rank=1` 和 `--node_rank=2` 运行即可。

落地成三条命令（`--master_addr` 始终指向 node02，`--node_rank` 各不相同）：

```bash
# ① 先在主节点 node02 (IP=192.168.0.101) 运行
torchrun --nproc_per_node=4 --nnodes=3 --node_rank=0 \
         --master_addr=192.168.0.101 --master_port=29500 test_mpi.py

# ② 再在 node03 运行（只改 node_rank=1，master_addr 仍指 node02）
torchrun --nproc_per_node=4 --nnodes=3 --node_rank=1 \
         --master_addr=192.168.0.101 --master_port=29500 test_mpi.py

# ③ 再在 node04 运行（只改 node_rank=2）
torchrun --nproc_per_node=4 --nnodes=3 --node_rank=2 \
         --master_addr=192.168.0.101 --master_port=29500 test_mpi.py
```

```
       node02 先起（node_rank=0=master）
            ▲           ▲
            │ 连主节点   │ 连主节点
       node03 ────────┘   │
       node04 ────────────┘
   三台都连到 192.168.0.101:29500 完成 rendezvous → 12 进程组成通信组
```

为什么**主节点要先起**？因为 master 是"集合点"，其他节点启动后会去连 `master_addr:master_port`；如果主节点还没监听，从节点连不上会一直等（直到 rendezvous 超时）。

## 6. rank 是怎么被算出来的（手算示例）

torchrun 在每个进程里注入环境变量，**全局 rank 由 node_rank 和 local_rank 推出**：

$$\text{global\_rank} = \text{node\_rank} \times \text{nproc\_per\_node} + \text{local\_rank}$$

本例 `nproc_per_node = 4`，手算：

| 机器 | node_rank | local_rank | global rank = node_rank×4 + local_rank |
|---|---|---|---|
| node02 | 0 | 0,1,2,3 | 0, 1, 2, 3 |
| node03 | 1 | 0,1,2,3 | **4, 5, 6, 7** |
| node04 | 2 | 0,1,2,3 | **8, 9, 10, 11** |

例如 node03 的第 3 块卡：$1 \times 4 + 2 = 6$，全局 rank = 6。`world_size = nnodes × nproc_per_node = 3 × 4 = 12`。

进程内通常这样取值并绑卡：

```python
import os, torch, torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ["LOCAL_RANK"])   # torchrun 注入
torch.cuda.set_device(local_rank)            # 绑到 cuda:local_rank
rank       = dist.get_rank()                 # 全局 rank，如 0~11
world_size = dist.get_world_size()           # 12
```

## 实操：可直接抄的命令与配置（原文真料汇总）

**1）进程内初始化（NCCL + env://，配合 torchrun）**

```python
import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method="env://")
```

**2）单机多卡（4 卡，单节点）**

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
         --master_addr=127.0.0.1 --master_port=29500 test_mpi.py
```

**3）多机多卡（3 机 ×4 卡，原文场景）**

```bash
# 主节点 node02 (192.168.0.101) 先运行；从节点把 node_rank 改成 1 / 2
torchrun --nproc_per_node=4 --nnodes=3 --node_rank=0 \
         --master_addr=192.168.0.101 --master_port=29500 test_mpi.py
```

**4）查 IP（决定 master_addr）**

```bash
ip addr   # 取主节点对应网卡的 IP 填入 --master_addr
```

> `--rdzv-backend`、`--rdzv-id` 用默认即可，本场景不必显式设置。

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|---|---|---|
| `backend="mpi"` 报不支持 | pip/conda 装的 PyTorch **不带 MPI** | 改用 NCCL；非要 MPI 得从源码编译 PyTorch |
| 从节点启动后一直卡住 | 主节点（node_rank=0）还没起，连不上集合点 | **先在主节点运行**，再起从节点 |
| 各机连不上 / 超时 | `master_addr` 不是 node_rank=0 那台的 IP，或端口被占/被防火墙挡 | 三台填**同一** master_addr/port；`ip addr` 核对 IP，换未占用端口 |
| `env://` 初始化卡住 | `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` 没设全 | 用 torchrun 自动注入，别手动漏设 |
| NCCL 初始化 hang | 多网卡选错网卡 / IB 配置问题 | 设 `NCCL_SOCKET_IFNAME` 指定网卡，`NCCL_DEBUG=INFO` 看日志 |
| `file://` 复用报错 | 上次崩溃残留的共享文件没清 | 删掉旧文件或换路径 |
| 每个进程都用了 cuda:0 → 显存爆/冲突 | 没按 `LOCAL_RANK` 绑卡 | `torch.cuda.set_device(local_rank)` |
| 用了已弃用警告的 launch | `torch.distributed.launch`（torch<2.0）将弃用 | 换 `torchrun` |
| `torch.distributed.launch` 不会算 rank | 需读环境变量 | 改 torchrun，进程内用 `env://` + 读 `LOCAL_RANK` |

## 🔗 跳转链接

- 枢纽地图：[[00-知识地图]]
- 训练总览：[[llm-train/README]]
- 本目录分布式：[[llm-train/pytorch/distribution/README]]
- 框架解析：[[ai-framework/pytorch/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/huggingface-peft/README]]
- 并行训练框架：[[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]
- 网络与通信：[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- PEFT 高效微调：[[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]
- 推理张量并行：[[B07:llm-inference/大模型推理张量并行]]
- 进阶主题：[[llm-alignment/RLHF]] · [[llm-algo/transformer/模型架构]] · [[llm-compression/quantization/量化基础]]
