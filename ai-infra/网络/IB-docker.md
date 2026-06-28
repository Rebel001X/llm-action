# 在 Docker 容器中使用 InfiniBand

> 把宿主机上"插好、能跑满 400Gb/s"的 IB 网卡，正确地"借"进容器里，让容器内的 NCCL/MPI 也能走 RDMA 内核旁路与 GPUDirect——核心是把网卡设备节点、用户态 verbs 库、内核驱动版本三者对齐。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/网络/InfiniBand]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]] [[llmops/kubernetes]]

## 阅读地图

| 节 | 你会搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点：容器里跑 IB 到底要对齐什么 | 设备节点 / 用户态库 / 驱动版本 |
| 1 | 地基：容器为什么默认看不到 IB 网卡 | namespace / cgroup / 设备隔离 |
| 2 | RDMA 设备节点长什么样、要透传哪些 | /dev/infiniband, uverbs |
| 3 | 容器内必须装的用户态软件（含本文起点 libibverbs） | rdma-core, provider |
| 4 | 三种透传方式对比：privileged / --device / 插件 | 权限粒度 |
| 5 | 用户态-内核态版本必须匹配（最大的坑） | ABI, provider 与内核 |
| 6 | GPUDirect RDMA 在容器里的额外要求 | nv_peer_mem, NUMA |
| 7 | 镜像怎么构建（Dockerfile 思路） | base 镜像选择 |
| 8 | Kubernetes 下怎么做（RDMA device plugin） | k8s, SR-IOV, network operator |
| N | docker run 示例 / 排障流程 / checklist | 验证三步法 |

## 0. 一句话锚点

在容器里用 InfiniBand，本质是让**容器内进程**能完成宿主机上同样的那条链路：

```
  容器内 NCCL/MPI  ──▶  libibverbs(用户态)  ──▶  /dev/infiniband/uverbsN  ──▶  内核 ib_uverbs/mlx5  ──▶  HCA 硬件
        (容器装)         (容器装)                  (从宿主机透传)              (宿主机内核,容器共享)     (物理卡)
```

记住三句话，本文全部围绕它们展开：

1. **设备节点要透传**：`/dev/infiniband/*` 是 verbs 访问网卡的入口，容器默认看不见，必须显式给进去。
2. **用户态库要在容器里装**：内核驱动在宿主机，但 `libibverbs` 和厂商 provider（如 `libmlx5`）是**用户态库，跟着进程跑，所以要装进镜像**——这就是那行 `yum install libibverbs` 的由来。
3. **用户态与内核态版本要匹配**：容器里的 verbs 库和宿主机内核驱动通过一套 ABI 对话，版本错配是"容器里 `ibv_devinfo` 报错/看不到设备"的头号原因。

## 1. 地基：容器为什么默认"看不见"IB 网卡

容器不是虚拟机，它和宿主机**共享同一个 Linux 内核**，靠 namespace（隔离视图）+ cgroup（限制资源）+ 设备/能力裁剪做出"独立机器"的假象。这恰恰带来三道墙，把 IB 网卡挡在外面：

```
  宿主机 (一个内核)
  ┌──────────────────────────────────────────────┐
  │  IB 驱动 mlx5_core / ib_uverbs  (内核态,全机共享)│
  │  设备节点 /dev/infiniband/uverbs0 ...           │
  │                                                │
  │   ┌── 容器 namespace ───────────────┐          │
  │   │  墙①设备cgroup: 默认禁止访问任意 /dev 设备 │
  │   │  墙②mnt namespace: 看不到宿主机 /dev 节点  │
  │   │  墙③能力(capabilities)被裁剪              │
  │   │  → ibv_get_device_list() 返回空           │
  │   └────────────────────────────────┘          │
  └──────────────────────────────────────────────┘
```

- **墙①设备 cgroup**：Docker 默认只放行极少数设备（如 `/dev/null`），其它一律拒绝。IB 的字符设备不在白名单里。
- **墙②挂载 namespace**：容器有自己的 `/dev`，里面根本没有 `infiniband` 这个目录。
- **墙③ capabilities**：某些 verbs 操作（如锁页 `IBV_ACCESS_LOCAL_WRITE` 注册大块 pinned 内存）需要 `CAP_IPC_LOCK`，默认容器没有，会触发 `Cannot allocate memory` / `mlock failed`。

所以"容器里跑 IB"做的所有事，都是在**有选择地拆掉这三道墙**：透传设备节点（拆墙①②）、补能力（拆墙③）、再把用户态库装进去。

## 2. RDMA 在容器里需要哪些设备节点

verbs 库通过打开特定字符设备来和内核驱动通信。这些节点都在宿主机 `/dev/infiniband/` 下：

```
  /dev/infiniband/
  ├── uverbs0, uverbs1 ...   ← 每个 HCA 一个；verbs 数据/控制面的主入口(最关键)
  ├── rdma_cm               ← librdmacm 用,基于 IP 的建链(RoCE/IPoIB 场景常用)
  ├── umad0, issm0          ← 管理/诊断(ibstat、SM 相关),按需
  └── ...
```

- **`uverbsN` 是必须的**：`libibverbs` 打开它来创建 QP/CQ/MR、收发数据。没透传它，`ibv_devinfo` 直接看不到设备。
- **`rdma_cm`**：如果上层用 RDMA-CM 建链（很多 NCCL/UCX 配置、RoCE 场景会用），需要它。
- **`umad/issm`**：跑 `ibstat`、子网管理类工具才需要，纯数据传输可不给。

实践上最省事的做法是**整目录透传** `--device=/dev/infiniband/`（Docker 会把目录下所有节点放进容器并加进设备 cgroup 白名单），细粒度场景再单独点名某几个 `uverbsN`。

## 3. 容器内必须安装的用户态软件（本文的起点）

这是初学者最容易误解的一点：**内核驱动在宿主机，为什么容器里还要装东西？**

因为 RDMA 是"用户态直接驱动网卡"。`libibverbs` 不是内核模块，而是**和你的应用一起运行的用户态库**——它存在于进程的地址空间里，所以**必须存在于容器的文件系统里**。容器有独立 rootfs，宿主机装的库它看不到，于是要在镜像里装：

```
  容器内需要的用户态组件(逻辑分层)：
  ┌─────────────────────────────────────────────┐
  │ 上层  NCCL / OpenMPI / UCX                    │  ← 训练框架调它们
  ├─────────────────────────────────────────────┤
  │ 核心  libibverbs   librdmacm                  │  ← verbs / cm 接口库
  ├─────────────────────────────────────────────┤
  │ provider  libmlx5(ConnectX) / 其它厂商 .so    │  ← 把通用 verbs 翻译成本厂网卡指令
  ├─────────────────────────────────────────────┤
  │ 工具(可选) ibverbs-utils / infiniband-diags   │  ← ibv_devinfo / ibstat 排障
  └─────────────────────────────────────────────┘
```

两条安装路线（对应 [[ai-infra/网络/IB软件]] 第 2 节的 inbox vs OFED 之争）：

| 路线 | 装什么 | 适用 |
|------|--------|------|
| **发行版 inbox** | `rdma-core`（含 libibverbs/librdmacm/provider）+ `libibverbs-utils` | 用宿主机 inbox 驱动、版本不挑、云上常见 |
| **NVIDIA DOCA/MLNX_OFED** | 容器里装与宿主机**同版本**的 OFED 用户态包 | 大规模 GPU 集群、需 GPUDirect/性能调优 |

> 文件最初那行 `yum install libibverbs` 就是 RHEL/CentOS 系最小化安装 verbs 库；更完整通常装 `rdma-core libibverbs libibverbs-utils`（Ubuntu 下对应 `apt install rdma-core ibverbs-utils libibverbs1`）。**provider（如 libmlx5）一定要有**，否则库在但认不出 ConnectX 网卡。具体包名以发行版/官方文档为准。

**关键约束（铺垫第 5 节）**：如果宿主机用 MLNX_OFED，容器里**最好也装同一份 MLNX_OFED 用户态**，而不是混用 inbox `rdma-core`——版本/ABI 要对齐。

## 4. 三种透传方式：从"全放开"到"精细控制"

把网卡借进容器，业界有三档做法，权衡是**省事程度 vs 安全/隔离粒度**：

```
  权限粒度  粗 ◀──────────────────────────────▶ 细
  ┌────────────────┐ ┌───────────────────┐ ┌──────────────────────┐
  │ ① --privileged │ │ ② --device + --cap │ │ ③ RDMA device 插件    │
  │  整机设备全开   │ │  只点名 IB 设备     │ │  k8s 按需分配,带隔离   │
  └────────────────┘ └───────────────────┘ └──────────────────────┘
   最省事/最不安全     推荐:单机/手动 docker     生产/集群标准做法
```

**① `--privileged`（图省事，慎用生产）**

```
  docker run --privileged --network=host ...
```

把所有设备 cgroup 白名单全开、能力全给。IB 一定能用，但**等于放弃容器隔离**，生产不推荐。常见于快速验证或开发镜像。

**② `--device` + 显式补能力（推荐的手动方式）**

只透传 IB 设备节点，只补必要能力，隔离性好得多：

```
  docker run \
    --device=/dev/infiniband/ \        # 透传 IB 设备节点(拆墙①②)
    --cap-add=IPC_LOCK \               # 允许锁页(注册大块 pinned MR,拆墙③)
    --ulimit memlock=-1 \              # 解除 memlock 上限(否则 mlock 失败)
    --network=host \                   # IB 数据面不走容器 NAT;走宿主机网络栈最简单
    your-image
```

四个要素各司其职：`--device` 给设备、`--cap-add=IPC_LOCK` + `--ulimit memlock=-1` 让 MR 注册（锁页）能成功、`--network=host` 避免容器网络栈干扰（IB/IPoIB 走宿主机网络最省心）。**这是单机 docker run 跑 IB 的标准姿势。**

**③ RDMA device plugin / 网络插件（集群标准）**

Kubernetes 下不手写 `--device`，而是用 device plugin 把 RDMA 设备作为可调度资源分配给 Pod（见第 8 节）。

## 5. 最大的坑：用户态库与内核驱动版本必须匹配

这是容器跑 IB 区别于普通容器化的**最核心难点**，单独成节。

容器共享宿主机内核，所以"内核驱动版本"由**宿主机**决定，容器改不了；而"用户态 verbs 库版本"由**镜像**决定。两者通过 `ib_uverbs` 暴露的一套 ABI（应用二进制接口）对话：

```
   容器内(镜像决定)              宿主机(内核决定)
  ┌──────────────────┐        ┌──────────────────────┐
  │ libibverbs 用户态 │        │ ib_uverbs / mlx5 内核 │
  │ libmlx5 provider  │◀─ABI──▶│ 驱动                  │
  └──────────────────┘        └──────────────────────┘
       版本 A                        版本 B
        └────── A 与 B 必须兼容,否则握手失败 ──────┘
```

不匹配的典型症状：

- 容器里 `ibv_devinfo` / `ibv_get_device_list` **返回空或报错**，明明设备节点已经透传进去。
- provider `.so` 加载失败、`mlx5: unsupported` 之类错误。
- NCCL `NCCL_DEBUG=INFO` 日志里看不到 IB 传输，**悄悄回退到 TCP（Socket）**，带宽掉一个数量级却不报错——最隐蔽。

避坑原则：

| 场景 | 做法 |
|------|------|
| 宿主机用 inbox 驱动 | 容器装与之同代的 `rdma-core`（一般同发行版同 major 即可兼容） |
| 宿主机用 MLNX/DOCA-OFED | 容器**装同一版本** MLNX/DOCA-OFED 用户态（NVIDIA 提供基础镜像最稳） |
| 不想自己对齐 | 直接用 NVIDIA NGC 上预装好 OFED 的官方镜像（如 PyTorch/CUDA NGC 镜像）作为 base |

> 一句话：**内核态归宿主机，用户态归镜像，二者版本要锁死**。最稳的工程做法是镜像 OFED 版本 == 宿主机 OFED 版本。具体兼容矩阵以 NVIDIA 官方文档为准。

## 6. 容器里的 GPUDirect RDMA：额外要求

如果只是 CPU 内存间 RDMA，第 2~5 节就够了。但 AI 训练要的是 **GPU 显存直达**（GPUDirect RDMA，见 [[ai-infra/网络/IB软件]] 第 5 节），容器里还要再满足几条：

```
  容器跑 GPUDirect RDMA 的叠加要求：
  [IB 部分]                          [GPU 部分]
  --device=/dev/infiniband/   +   --gpus all (或 nvidia runtime)
  IPC_LOCK + memlock                NVIDIA Container Toolkit 注入驱动
       │                                  │
       └────── 还需要 ───────────┐        │
   宿主机加载 nv_peer_mem / DMA-BUF 内核模块(容器改不了,必须宿主机就绪)
       │
   GPU 与 NIC 同 PCIe switch / 同 NUMA(nvidia-smi topo -m 看 PIX/PXB)
```

要点：

1. **GPU 透传走 NVIDIA Container Toolkit**：用 `--gpus` 或 nvidia runtime，把宿主机 GPU 驱动库注入容器（和 IB 的设备透传是两套机制，要同时做对）。
2. **`nv_peer_mem`/DMA-BUF 是内核模块**：它让网卡能 DMA GPU 显存，**必须在宿主机加载好**，容器内无法加载内核模块。容器只是"用"它的能力。
3. **拓扑亲和性照样生效**：容器不改变物理 PCIe 拓扑，`nvidia-smi topo -m` 看到的 GPU↔NIC 关系（PIX/PXB 好，SYS 差）在容器里和宿主机一样重要。
4. **NCCL 配置透传进容器**：`NCCL_IB_HCA` 指定网卡、`NCCL_NET_GDR_LEVEL` 控制 GPUDirect 启用层级，这些环境变量要在容器里设。

## 7. 镜像怎么构建（Dockerfile 思路）

不背具体版本，讲分层逻辑（具体包名/版本以官方文档为准）：

```
  ┌─────────────────────────────────────────────┐
  │ FROM nvidia/cuda 或 NGC 镜像  ← 选 base(自带  │
  │                                CUDA/有时含OFED)│
  ├─────────────────────────────────────────────┤
  │ 装 RDMA 用户态:                                │
  │   inbox 路线: apt/yum install rdma-core        │
  │              ibverbs-utils libibverbs1         │
  │   OFED 路线: 安装与宿主机同版本 MLNX/DOCA-OFED  │
  ├─────────────────────────────────────────────┤
  │ 装诊断工具(可选): infiniband-diags(ibstat 等)  │
  ├─────────────────────────────────────────────┤
  │ 装/带 NCCL + OpenMPI/UCX                       │
  ├─────────────────────────────────────────────┤
  │ 你的训练代码 / 框架(PyTorch 等)                 │
  └─────────────────────────────────────────────┘
```

选 base 镜像的决策：**能用 NVIDIA NGC 官方镜像就别自己拼**——它把 CUDA、OFED 用户态、NCCL 版本组合都测过，省掉第 5 节的版本对齐地狱。只有当你必须用特定基础系统时，才自己 `FROM ubuntu` 一层层装，并务必让 OFED 版本对齐宿主机。

## 8. Kubernetes 下怎么做

手写 `docker run --device` 在集群里不可扩展。k8s 用**插件机制**把 RDMA 网卡变成可声明、可调度、可隔离的资源：

```
  Pod 声明: resources.limits: rdma/hca: 1
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │ RDMA device plugin (DaemonSet 跑在每个节点)    │
  │   - 把 /dev/infiniband 设备暴露为可调度资源     │
  │   - 调度到该 Pod 时,自动注入设备节点 + 权限     │
  ├──────────────────────────────────────────────┤
  │ (大集群) NVIDIA Network Operator                │
  │   - 统一部署 OFED 驱动 / device plugin /        │
  │     SR-IOV / 二级网络(Multus)                   │
  ├──────────────────────────────────────────────┤
  │ (隔离强需求) SR-IOV: 把一张物理 HCA 切成多个 VF │
  │   每个 Pod 拿独立 VF,做到真正的网卡级隔离       │
  └──────────────────────────────────────────────┘
```

- **RDMA device plugin**：让 Pod 像申请 GPU 一样申请 RDMA 设备，调度器据此放置 Pod，并自动完成第 4 节那套设备/能力注入。
- **SR-IOV**：把一张物理网卡虚拟成多个 VF（Virtual Function），每个 Pod 独占一个 VF，隔离性最好（多租户场景）；共享模式则多个 Pod 共用 HCA。
- **NVIDIA Network Operator**：把"装 OFED 驱动 + 部署 device plugin + 配 SR-IOV/Multus 二级网络"全自动化，是大规模 GPU 集群的推荐方式（具体能力以官方文档为准）。
- 别忘了 `IPC_LOCK` 能力和 `memlock` ulimit 在 Pod 的 securityContext / 节点配置里也要给到。

## docker run 完整示例与排障

### 单机最小可用示例（手动 docker run）

```
  docker run -it --rm \
    --network=host \                   # IB/IPoIB 走宿主机网络栈
    --device=/dev/infiniband/ \        # 透传所有 IB 设备节点
    --cap-add=IPC_LOCK \               # 允许锁页(MR 注册)
    --ulimit memlock=-1:-1 \           # 解除 memlock 软硬上限
    --gpus all \                       # (训练)透传 GPU
    your-ib-image:tag \
    bash
```

进容器后**先验证再训练**（验证三步法）：

```
  ① ibv_devinfo        看 verbs 是否认出网卡(state: PORT_ACTIVE 才对)
       │  报错/空 → 第5节版本对齐 或 第2节设备没透传
       ▼
  ② ib_write_bw        裸 RDMA 带宽测试(一端 server 一端 client)
       │  慢 → 物理/路由问题(不是容器问题)
       ▼
  ③ NCCL_DEBUG=INFO    跑一个 all_reduce,看日志里走的是 IB 还是 Socket
          走 Socket → 静默回退,回头查第5/6节
```

### 常见问题

| 问题 | 原因 / 排查 |
|------|------------|
| 容器里 `ibv_devinfo` 看不到设备 | (1) 没透传 `/dev/infiniband`（第2节）；(2) 用户态库/provider 没装或与内核版本不匹配（第3、5节） |
| 报错 `mlock failed` / `Cannot allocate memory` | 缺 `CAP_IPC_LOCK` 或 `memlock` ulimit 太小，无法锁页注册 MR（第4节补 `--cap-add=IPC_LOCK --ulimit memlock=-1`） |
| 设备透传了但 provider 报 `unsupported` | 容器里的 `libmlx5` 等 provider 与宿主机内核驱动版本错配（第5节，对齐 OFED 版本） |
| NCCL 莫名很慢，带宽掉一个量级 | **静默回退到 TCP**。`NCCL_DEBUG=INFO` 确认有没有走 IB；多半是版本错配或没透传设备 |
| GPUDirect RDMA 没生效 | 宿主机 `nv_peer_mem`/DMA-BUF 未加载（容器内无法加载内核模块），或 GPU↔NIC 跨 NUMA（第6节） |
| 用 `--network=bridge` 通信异常 | IB/IPoIB 走容器 NAT 很麻烦，改 `--network=host` 最省事 |
| inbox 还是 OFED 装哪个 | 跟宿主机走：宿主机 inbox 就装 `rdma-core`；宿主机 MLNX/DOCA-OFED 就装同版本 OFED（第3、5节） |
| 为什么内核在宿主机还要在容器装 libibverbs | verbs 库是**用户态**库，随进程跑、随 rootfs 走，所以必须装进镜像（第3节） |
| privileged 能跑但生产能用吗 | 不建议，等于放弃隔离。生产用 `--device`+`--cap-add` 或 k8s device plugin（第4、8节） |

### 一图总结：容器跑 IB 的"三对齐"

```
  ┌──────────── 容器跑 IB = 三件事对齐 ────────────┐
  │  ① 设备对齐  /dev/infiniband 透传进容器          │
  │  ② 软件对齐  容器装 libibverbs + provider(用户态) │
  │  ③ 版本对齐  容器用户态 == 宿主机内核 OFED 版本   │
  │  (训练再叠加) GPU 透传 + 宿主机 nv_peer_mem 就绪  │
  └────────────────────────────────────────────────┘
   缺①→看不到设备  缺②→库报错  缺③→静默回退TCP/provider失败
```

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引
- [[ai-infra/网络/InfiniBand]] — IB 硬件/协议/拓扑（本篇是"把它放进容器"）
- [[ai-infra/网络/IB软件]] — IB 用户态软件栈全貌（本篇装进容器的就是它）
- [[ai-infra/网络/NCCL]] — 容器内训练真正调用的集合通信层
- [[ai-infra/网络/集合通信原语]] — all-reduce 等原语，IB 是其传输底座
- [[llmops/kubernetes]] — 第 8 节 RDMA device plugin / SR-IOV 的上层平台
