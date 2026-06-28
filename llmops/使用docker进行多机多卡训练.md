# 使用 Docker 进行多机多卡训练

> 把"环境复杂、机器异构"的多机多卡训练，封装进**可复现的容器镜像**，让每台机器跑出一模一样的环境，再用容器网络 + SSH/启动器把它们串成一个分布式作业。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llmops/README]] [[llmops/kubernetes]] [[llm-train/pytorch/distribution/README]] [[ai-infra/网络/NCCL]] [[ai-framework/deepspeed/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | 镜像 = 可复现环境 |
| 1 | 地基：为什么多机多卡训练这么难，Docker 解决了哪一半 | 环境漂移 / 依赖地狱 |
| 2 | 多机多卡的通信骨架（你得先懂它，才知道容器要"放行"什么） | NCCL / rank / IB |
| 3 | 容器化训练的整体架构（单机多卡 → 多机多卡） | host 网络 / shm / 设备直通 |
| 4 | GPU 怎么进容器：NVIDIA Container Toolkit | runtime / device 挂载 |
| 5 | 容器网络模式：bridge vs host vs overlay | NCCL 直连 / 端口 |
| 6 | 跨节点通信的硬件直通：IB / RDMA / `--ipc` / `--shm-size` | RDMA 设备 / 共享内存 |
| 7 | 启动方式：手工 SSH vs torchrun vs DeepSpeed/MPI launcher | rendezvous / hostfile |
| 8 | 一个端到端的 2 机 16 卡示例（参数讲含义，不背默认值） | NODE_RANK / MASTER_ADDR |
| 9 | 数值例子 + 实践要点 | 带宽 / shm / 拓扑 |
| QA | 常见坑表格 | 卡死 / 慢 / 找不到设备 |

## 0. 一句话锚点

**Docker 在多机多卡训练里只解决一件事：让每台机器的"软件环境"完全一致且可复现。** 它把 CUDA 运行时、cuDNN、NCCL、PyTorch、各种 Python 依赖打成一个镜像，`docker run` 出来的容器在任何机器上都长得一样，从此告别"我这台能跑、你那台报错"的环境漂移。

但 Docker **不负责**把多台机器变成一个分布式作业——那是 **NCCL（通信）+ 启动器（torchrun / deepspeed / mpirun，负责拉起进程并协调 rank）** 的事。容器化的关键难点，恰恰是**别让容器的隔离机制把分布式通信挡在外面**：GPU 要能进容器、跨机端口要通、IB/RDMA 设备要直通、共享内存要够大。本文就围绕"封装环境"和"打通通信"这两件事展开。

## 1. 地基：多机多卡为什么难，Docker 解决哪一半

先把问题拆成两块原子：

- **(A) 环境问题**：训练一个大模型，依赖链极长——CUDA Toolkit 版本、cuDNN、NCCL、PyTorch/编译器、apex/flash-attn 这类要编译的算子、几十个 Python 包。多台机器手动装，几乎必然出现版本错位（"环境漂移 / dependency hell"）。**这一半正是 Docker 的主场**：镜像把整套环境冻结成不可变的层，`docker pull` 到每台机器，环境就 100% 一致。
- **(B) 协同问题**：N 台机器、每台 M 张卡，要让 `N×M` 个进程互相发现、按 rank 编号、用 NCCL 做 AllReduce 同步梯度。**这一半 Docker 管不了**，需要启动器 + 通信库 + 网络打通。

```
        多机多卡训练 = 两个独立问题
   ┌─────────────────────┬──────────────────────┐
   │ (A) 环境一致性        │ (B) 进程协同 / 通信     │
   │  ▶ Docker 解决        │  ▶ NCCL + 启动器解决    │
   │  CUDA/cuDNN/NCCL      │  rank 编号 / rendezvous │
   │  PyTorch/依赖/算子     │  AllReduce / IB        │
   └─────────────────────┴──────────────────────┘
   容器化的活儿：让 (A) 落地，且不要破坏 (B)
```

理解这个分工，后面所有"为什么要加这个 flag"都有了答案：凡是 `--gpus`、`--ipc`、`--shm-size`、`--network host`、`--device /dev/infiniband/*` 这类参数，本质都是**在容器隔离的墙上，给 (B) 的通信开一道门**。

## 2. 先补通信骨架：你要"放行"的到底是什么

容器只是个壳，被它包住的真正主角是 **NCCL（NVIDIA Collective Communications Library）**。它负责 GPU 间的集合通信（AllReduce / AllGather / Broadcast 等）。多机多卡时，每个进程（rank）持有一张 GPU，梯度同步靠 NCCL 的 AllReduce 完成。

NCCL 选用的通信通道按"由快到慢"大致是：
- **机内**：NVLink / NVSwitch（GPU 直连，最快）→ 否则走 PCIe。
- **机间**：InfiniBand（IB）/ RoCE 上的 **GPUDirect RDMA**（GPU 显存直接走网卡，绕过 CPU，最快）→ 否则退化到 TCP/IP（走内核协议栈，慢且吃 CPU）。

```
   节点 A                              节点 B
 ┌──────────────┐                   ┌──────────────┐
 │ GPU0 ─NVLink─ GPU1 │             │ GPU0 ─NVLink─ GPU1 │
 │   │ GPUDirect RDMA │  IB/RoCE    │   │            │ │
 │  ┌┴───┐               网络       │  ┌┴───┐         │
 │  │ HCA│════════════════════════════│ HCA│         │  ← 机间高速链路
 │  └────┘               (后端网)    │  └────┘         │
 └──────────────┘                   └──────────────┘
   机内: NVLink/PCIe          机间: IB-RDMA(快) 或 TCP(慢)
```

> 关键结论：**容器化只要漏掉任何一环（GPU 没进容器、IB 设备没直通、共享内存太小、端口没通），NCCL 就会卡死或悄悄退化到 TCP，速度可能差一个数量级。** 第 6 节专门讲怎么把这些环节在容器里打通。相关原理见 [[ai-infra/网络/NCCL]] 与 [[ai-infra/网络/集合通信原语]]。

## 3. 容器化训练的整体架构

从单机多卡到多机多卡，容器的角色是逐步"放权"的：

```
单机多卡（1 容器 = 1 机的多卡）
┌─────────── Host (8×GPU) ───────────┐
│  docker run --gpus all --ipc=host  │
│  ┌──────── Container ───────────┐  │
│  │ torchrun --nproc_per_node=8  │  │
│  │  rank0 rank1 ... rank7        │  │
│  │  (NCCL 走 NVLink/PCIe)        │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘

多机多卡（每机 1 容器，容器间靠后端网通信）
┌──── Node0 ────┐        ┌──── Node1 ────┐
│ Container(同镜像)│  IB   │ Container(同镜像)│
│ NODE_RANK=0    │◄═════►│ NODE_RANK=1    │
│ rank 0..7      │  网络  │ rank 8..15     │
│ MASTER_ADDR ──┼────────┤ 连到 Node0     │
└────────────────┘        └────────────────┘
   一个全局通信域：world_size=16
```

设计要点（每条都说"为什么"）：
- **每台物理机起 1 个容器，容器里再用启动器拉起多个进程（每卡 1 进程）**。这是最常见、最省心的拓扑：容器内进程走机内高速通道，容器间走机间网络。
- **镜像必须完全一致**：所有节点 `docker pull` 同一个 tag（最好用 digest 锁定），否则 NCCL/PyTorch 版本不一致会导致通信协议对不上而 hang 死。这正是 Docker 的核心价值。
- **数据/checkpoint 用挂载卷（`-v`）或共享存储（NFS/对象存储）**，不要打进镜像——镜像只装环境，数据是变量。

## 4. GPU 怎么进容器：NVIDIA Container Toolkit

容器默认看不到宿主机的 GPU。让 GPU 进容器靠 **NVIDIA Container Toolkit**（旧称 nvidia-docker / nvidia-container-runtime）。它的机制：

- 宿主机装 **NVIDIA 驱动**（驱动留在宿主机，**镜像里只装 CUDA 运行时**，不装驱动——这就是为什么换驱动要动宿主机、而不是重建镜像）。
- 装 toolkit 后，容器运行时在启动容器时**自动把 `/dev/nvidia*` 设备节点、驱动用户态库、`nvidia-smi` 等注入容器**。
- 用户侧只需在 `docker run` 时加 **`--gpus`** 选项声明要哪些 GPU。

```
  容器内进程  →  CUDA 运行时(镜像自带)
       │
       ▼  (toolkit 注入)
  /dev/nvidia0..7  +  宿主机驱动用户态库(libcuda 等)
       │
       ▼
   宿主机 NVIDIA 驱动 ──► 物理 GPU
```

`--gpus` 的几种典型用法（讲含义，具体语法以官方文档为准）：
- **全部可见**：把宿主机所有 GPU 暴露给容器。
- **按数量**：只暴露 N 张（让运行时挑）。
- **按 ID/UUID**：精确指定哪几张卡，常用于在一台机器上隔离多个作业。
- 也可通过 `NVIDIA_VISIBLE_DEVICES` 环境变量达到类似效果。

> 实践注意：容器内的 `CUDA_VISIBLE_DEVICES` 是在**容器已能看到的 GPU 子集**里再做选择，二者是"先 toolkit 决定容器能看到哪些卡，再 CUDA 变量决定进程用其中哪几张"的两层关系，别混淆。

## 5. 容器网络模式：bridge / host / overlay

跨机通信的第一道门是**网络**。Docker 常见三种模式，对分布式训练影响很大：

| 模式 | 机制 | 对多机训练的影响 |
|---|---|---|
| **bridge（默认）** | 容器有独立网络命名空间，靠 NAT 与宿主机端口映射通信 | 跨机时端口要 `-p` 映射、NAT 转发，**额外开销大、易踩端口坑**，不推荐用于高性能训练 |
| **host** | 容器**直接用宿主机网络栈**，无隔离无 NAT | 容器直接拥有宿主机 IP 和高速网卡，**NCCL 能直连后端网，延迟/带宽最优**，多机训练最常用 |
| **overlay** | 跨主机的虚拟二层网络（多用于 Swarm/K8s） | 提供跨机扁平网络，但软件 overlay 通常有性能损耗，高性能场景多绕过它走 host/IB |

```
 bridge:  Container ──NAT──> Host eth0 ──> 网络   (多一层转换，慢)
 host:    Container ════════ Host eth0/IB ──> 网络 (零转换，快)  ★训练首选
```

> 结论：**多机多卡训练绝大多数情况用 `--network host`**。原因有三：① 省掉 NAT，延迟带宽最好；② NCCL 直接看到宿主机的高速网卡（含 IB）；③ 各 rank 用真实 IP/端口互联，配置最简单。代价是放弃网络隔离——在专用训练集群里通常可接受。`MASTER_ADDR` 此时填**宿主机的后端网 IP**。

## 6. 跨节点高速通信的直通：IB/RDMA、IPC、共享内存

这是容器化训练**最容易翻车**的一节。把 GPU 放进容器只完成了一半，要让 NCCL 跑满带宽，还得打通三样东西：

### 6.1 InfiniBand / RDMA 设备直通
IB 走 RDMA 需要容器能访问 IB 设备节点（如 `/dev/infiniband/*`）。常见做法：
- **`--network host`**：让容器直接用宿主机网络栈，IB 网卡自然可见（最省事）。
- **`--device /dev/infiniband/...`**：把 IB 字符设备直通进容器（按需挂载具体设备）。
- 有时还需放宽权限/能力（如内存锁定相关的 ulimit，RDMA 需要 pin 大量内存）。

如果这一步没做对，NCCL 会**悄悄退回 TCP**（`Socket` 传输），训练能跑但慢得离谱。可通过 NCCL 的调试输出（设置 `NCCL_DEBUG=INFO`）确认它选的是 `IB`/`NET/IB` 还是 `Socket`。

### 6.2 进程间通信（IPC）与共享内存（/dev/shm）
- **`--ipc=host`**：让容器与宿主机**共享 IPC 命名空间**。PyTorch 的 DataLoader 多进程、以及部分 NCCL 路径会用到 POSIX 共享内存，默认容器的 IPC 隔离 + 极小的 `/dev/shm` 会导致报错或卡死。
- **`--shm-size`**：调大容器 `/dev/shm` 大小（默认通常只有 64MB，对训练远远不够）。**没调大它，是新手最常见的"DataLoader worker 崩溃 / Bus error"原因。**

```
  容器隔离把这三道门关上 → 必须手动打开：
  ┌─ GPU ───────► --gpus / NVIDIA Container Toolkit (第4节)
  ┌─ 网络 ──────► --network host                     (第5节)
  ┌─ IB/RDMA ──► --device /dev/infiniband + host net (6.1)
  └─ 共享内存 ──► --ipc=host  且/或  --shm-size=...    (6.2)
        缺一个 → 卡死 / 退化 / 崩溃
```

> 记忆口诀：**"卡(GPU)、网(host)、IB(直通)、shm(放大)"** 四件套，多机训练容器缺一不可。

## 7. 启动方式：谁来把进程拉起来并编号

镜像和网络都通了，最后要有人**在每台机器上拉起进程、给每个进程分配 rank、并告诉它们去哪里集合（rendezvous）**。三种主流方式：

| 方式 | 机制 | 适用 |
|---|---|---|
| **手工 SSH + 环境变量** | 每台机器手动 `docker run`，手填 `MASTER_ADDR/MASTER_PORT/NODE_RANK/WORLD_SIZE` | 机器少、调试，最透明 |
| **torchrun（PyTorch 弹性启动器）** | 每节点跑一个 torchrun，它按 `--nnodes/--nproc_per_node/--node_rank` 拉起本机进程并完成 rendezvous | PyTorch 原生，最常用，见 [[llm-train/pytorch/distribution/README]] |
| **DeepSpeed / MPI launcher** | 用 `hostfile` 列出所有节点，launcher 通过 SSH（或 pdsh/mpirun）登录各机批量拉起，一条命令搞定全集群 | DeepSpeed/Megatron 训练，见 [[ai-framework/deepspeed/README]] |

关键变量（所有方式本质都在设置这几个，含义远比记默认值重要）：
- **`MASTER_ADDR` / `MASTER_PORT`**：rank 0 所在地址和端口，是所有进程"集合点"。容器里通常填宿主机后端网 IP。
- **`WORLD_SIZE`**：全局进程总数 = 节点数 × 每节点卡数（如 2×8=16）。
- **`NODE_RANK`**（机器编号）/ **`RANK`**（全局进程编号）/ **`LOCAL_RANK`**（本机内编号，决定用第几张卡）。

> DeepSpeed 的 `hostfile` 用法有个容器特有坑：launcher 默认靠 **SSH 登录每台机器**再启动进程。容器化时要么让容器内跑 SSH 服务且各容器能互相免密登录，要么改用"每机各自起容器 + torchrun"的对称方式。很多团队最终选后者，因为不必在容器里维护 SSH。

## 8. 端到端示例：2 机 16 卡（参数讲含义）

下面给一个**对称启动**（每台机器各跑一条命令）的骨架，重点理解每个参数为什么在这里，**具体 flag 名称与默认值以对应官方文档/源码为准**。

构建并分发同一镜像（保证环境一致）：
```bash
# 在任一机器构建，推到 registry，每台机器 pull 同一 digest
docker build -t myregistry/llm-train:v1 .
docker push myregistry/llm-train:v1
# 每台节点执行： docker pull myregistry/llm-train:v1
```

节点 0（`MASTER_ADDR` 指向本机后端网 IP，假设 10.0.0.10）：
```bash
docker run --rm -it \
  --gpus all \                  # 第4节：8 张 GPU 进容器
  --network host \              # 第5节：直连后端网，NCCL 最优
  --ipc=host \                  # 第6.2：共享 IPC，DataLoader/NCCL 需要
  --shm-size=16g \              # 第6.2：放大 /dev/shm，防 worker 崩溃
  --device /dev/infiniband \    # 第6.1：IB 设备直通（如有 IB）
  -v /data:/data \              # 数据走挂载，不打进镜像
  -e NCCL_DEBUG=INFO \          # 让 NCCL 打印选了哪条传输通道
  myregistry/llm-train:v1 \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
           --master_addr=10.0.0.10 --master_port=29500 \
           train.py
```

节点 1（只改 `--node_rank=1`，`master_addr` 仍指向节点 0）：
```bash
docker run --rm -it --gpus all --network host --ipc=host \
  --shm-size=16g --device /dev/infiniband -v /data:/data \
  -e NCCL_DEBUG=INFO myregistry/llm-train:v1 \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
           --master_addr=10.0.0.10 --master_port=29500 \
           train.py
```

两条命令一起跑，`torchrun` 会在各机拉起 8 个进程，共 16 个 rank 在 `10.0.0.10:29500` 完成 rendezvous，组成 `world_size=16` 的通信域开始训练。**两条命令的唯一区别是 `--node_rank`**——这正是"对称启动"的优雅之处。

```
 时间线：
  Node0: docker run ... node_rank=0 ─┐
                                     ├─► rendezvous @10.0.0.10:29500
  Node1: docker run ... node_rank=1 ─┘
                                     ▼
              16 个 rank 建立 NCCL 通信域 → AllReduce 同步梯度 → 训练
```

## 9. 数值例子与实践要点

- **带宽直觉**：NVLink 单卡双向带宽可达数百 GB/s 量级；机间 IB（如 200Gb/s ≈ 25GB/s）虽远低于机内，但仍远高于普通 TCP/万兆以太网（10Gb/s ≈ 1.25GB/s）。所以**让 NCCL 走上 IB-RDMA 而非退化到 TCP，往往就是吞吐差 10 倍以上**的分水岭——务必用 `NCCL_DEBUG=INFO` 确认通道。
- **`/dev/shm` 估算**：DataLoader 多 worker + pin memory 会占用共享内存，大模型大 batch 下几十 MB 的默认值瞬间爆掉。经验上设到 **几 GB ~ 几十 GB**（如 `--shm-size=16g`）较稳妥，按机器内存与 worker 数权衡。
- **镜像锁定**：用 **digest（`@sha256:...`）而非可变 tag** 拉镜像，避免"某台机器 pull 到了新版本导致 NCCL 协议不一致而 hang"。
- **拓扑亲和**：GPU、网卡（HCA）、NUMA 的物理拓扑会影响性能。NCCL 有拓扑感知，但容器若屏蔽了某些信息可能选不到最优路径——高性能场景需结合宿主机拓扑工具核对。
- **慢节点效应**：同步训练里 AllReduce 是同步屏障，**最慢的那个 rank 决定整体速度**（木桶效应）。一台机器网络/IO 异常，会拖垮整个作业。
- **与 K8s 的关系**：手动 `docker run` 适合少量机器和调试；规模化后通常交给 Kubernetes（用 Volcano/Kubeflow 做 gang 调度，把上面这些 flag 翻译成 Pod 的 `resources`/`securityContext`/`volumes`）。详见 [[llmops/kubernetes]]。

## 常见问题 / 坑（表格）

| 现象 | 根因 | 处理方向 |
|---|---|---|
| 容器内 `nvidia-smi` 找不到 GPU | 没装/没用 NVIDIA Container Toolkit，或漏了 `--gpus` | 装 toolkit，加 `--gpus`，确认宿主机驱动正常 |
| 训练能跑但**极慢** | NCCL 退化到 TCP（IB 没直通 / 没用 host 网络） | `NCCL_DEBUG=INFO` 看传输通道；补 `--network host` + IB 设备直通 |
| DataLoader worker 崩溃 / **Bus error** | `/dev/shm` 太小，或 IPC 被隔离 | 加 `--shm-size=...` 和 `--ipc=host` |
| 多机训练**启动即 hang**（卡在初始化） | `MASTER_ADDR/PORT` 不通、端口被防火墙挡、各机镜像版本不一致 | 检查后端网连通与端口；用同一 digest 镜像；确认 `NODE_RANK/WORLD_SIZE` 正确 |
| 偶发 hang 或协议错误 | 各节点 NCCL/PyTorch/CUDA 版本不一致 | 全集群统一镜像（Docker 的核心价值），用 digest 锁定 |
| DeepSpeed `hostfile` 启动失败 | launcher 走 SSH 但容器没 SSH/免密 | 容器内配 SSH 免密，或改用每机对称 `torchrun` |
| RDMA 报内存锁定失败 | 容器 ulimit（memlock）太小 | 放宽 memlock 限制（`--ulimit memlock=...`） |
| 端口冲突 / NAT 问题 | 用了 bridge 网络做多机 | 改 `--network host` |

## 🔗 跳转链接

- [[00-知识地图]]
- [[llmops/README]]
- [[llmops/kubernetes]]
- [[llm-train/pytorch/distribution/README]]
- [[ai-framework/deepspeed/README]]
- [[ai-infra/网络/NCCL]]
- [[ai-infra/网络/集合通信原语]]
- [[ai-infra/网络/InfiniBand]]

> 参考：Docker 容器中 DeepSpeed 多机多卡集群分布式训练大模型实践 https://cloud.baidu.com/article/3273769
