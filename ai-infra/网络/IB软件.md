# InfiniBand 软件栈
> 把一块 IB 网卡（HCA）从"插上去能 ping 通"变成"GPU 显存里的张量绕过 CPU 直接飞到对端显存"的那一整层软件：驱动 + 用户态 verbs 库 + 子网管理 + 诊断工具 + GPUDirect + 上层 NCCL/MPI。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/网络/InfiniBand]] [[ai-infra/网络/通信软件]]

## 阅读地图

| 节 | 你会搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点：软件栈分几层 | 内核态 / 用户态 / 旁路 |
| 1 | 地基：RDMA / 内核旁路 / 零拷贝为什么快 | kernel bypass, zero-copy |
| 2 | OFED / MLNX_OFED 驱动包是什么 | OFED, inbox driver |
| 3 | verbs API：用户态怎么直接驱动网卡 | libibverbs, QP, CQ, MR |
| 4 | subnet manager：谁给整个 IB 网络配路由 | OpenSM, LID, routing |
| 5 | GPUDirect RDMA：显存直达，CPU 出局 | nvidia-peer-memory, P2P |
| 6 | perftest：怎么量出带宽/时延 | ib_write_bw, ib_send_lat |
| 7 | ibstat/ibstatus/诊断全家桶 | ibstat, ibping, perfquery |
| 8 | 与 NCCL/MPI 的关系：谁调谁 | NCCL, HPC-X, UCX |
| 9 | 数值例子 / 对照 / 实践清单 | 手算带宽时延 |

## 0. 一句话锚点

InfiniBand 软件栈是一个"**自下而上四层**"的结构。从硬件往应用方向：

```
  ┌─────────────────────────────────────────────────────────┐
  │  应用层    PyTorch / Megatron / vLLM                      │
  ├─────────────────────────────────────────────────────────┤
  │  集合通信  NCCL / MPI(HPC-X) / UCX / SHARP               │  ← 你训练时直接用的
  ├─────────────────────────────────────────────────────────┤
  │  用户态库  libibverbs / librdmacm + 厂商 provider(mlx5)   │  ← verbs API
  ├─────────────────────────────────────────────────────────┤
  │  内核驱动  ib_core / mlx5_core / ib_uverbs (OFED)         │  ← 控制路径走这
  ├─────────────────────────────────────────────────────────┤
  │  硬件      HCA 网卡(ConnectX) + Switch + Subnet Manager   │
  └─────────────────────────────────────────────────────────┘
        数据路径(data path)绕过内核，直接 用户态↔网卡↔显存
```

核心反直觉点：**数据真正传输时不经过内核**。内核只在建链、注册内存等"慢路径"出场；一旦链路建好，发包/收包由用户态库直接敲网卡寄存器（doorbell），这就是 **kernel bypass（内核旁路）**。

## 1. 地基：为什么 IB 软件能"快"

要理解整个软件栈的设计，先拆清三个最原子的概念。普通 TCP/IP 发一个包，数据要经历：`用户buffer → 内核socket buffer(拷贝1) → 协议栈处理 → 网卡DMA(拷贝2)`，每次系统调用还要陷入内核（上下文切换）。RDMA 把这些全砍掉：

**(1) 零拷贝(zero-copy)**：网卡直接从你注册过的用户内存 DMA 取数据，不经过内核中转 buffer。少了 2 次内存拷贝。

**(2) 内核旁路(kernel bypass)**：建链之后，发送命令直接写网卡的"门铃寄存器"，不走 `send()` 系统调用，没有用户态↔内核态切换。

**(3) CPU 卸载(offload)**：传输层协议（可靠性、重传、分片）由网卡硬件完成，CPU 不参与搬数据，可以去算别的。

```
  传统 TCP/IP                       RDMA (InfiniBand)
  app buffer                        app buffer (已注册MR)
     │ copy                              │  (无拷贝)
  kernel socket buf                      │
     │ 协议栈(CPU)                       ▼
  NIC DMA  ── 多次拷贝 + 多次陷内核   HCA DMA ── 一步到位，CPU不参与
```

代价：你必须**预先把内存"注册"给网卡**（告诉网卡这块物理页可以 DMA、并锁定不被换出），这就是后面 verbs 里的 MR（Memory Region）。这也是为什么 RDMA 编程比 socket 复杂——你要自己管内存和队列。

## 2. OFED / MLNX_OFED：驱动包

**OFED**（OpenFabrics Enterprise Distribution）是 OpenFabrics Alliance 维护的开源 RDMA 软件集合，包含内核驱动 + 用户态库 + 工具。它把"让操作系统认识 RDMA 网卡并提供统一 verbs 接口"这件事标准化，**一套 API 同时支持 InfiniBand、RoCE（以太网上的 RDMA）、iWARP**。

**MLNX_OFED**（现多称 **DOCA-OFED** / NVIDIA OFED）是 NVIDIA（原 Mellanox）测试、打包、调优过的 OFED 发行版，对 ConnectX 系列网卡做了深度优化，支持高达 400Gb/s 的 IB 和 10/25/40/50/100/200/400 GbE 的 RoCE（具体支持矩阵以官方文档为准）。

**两条路线对比**（实践中第一个决策点）：

| | inbox driver（发行版内置） | MLNX_OFED / DOCA-OFED |
|--|--------------------------|----------------------|
| 来源 | RHEL/Ubuntu/SLES 自带 `rdma-core` | 从 NVIDIA 官网单独下载安装 |
| 优点 | 随系统升级、与内核兼容好、省心 | 版本新、性能调优、含完整诊断工具、官方支持 |
| 缺点 | 工具/特性可能滞后 | 与特定内核版本绑定，升级内核要重装 |
| AI 训练推荐 | 小规模/云上可用 | 大规模 GPU 集群通常用它（GPUDirect 等需要配套） |

```
  MLNX_OFED 包内大致分层：
  ┌──────────── 用户态 ────────────┐
  │ libibverbs  librdmacm  libibmad │  库
  │ perftest    infiniband-diags    │  工具(perftest/ibstat...)
  │ opensm                          │  子网管理器
  ├──────────── 内核态 ────────────┤
  │ ib_core   ib_uverbs   rdma_cm   │  通用 RDMA 框架
  │ mlx5_core mlx5_ib                │  ConnectX 厂商驱动
  └─────────────────────────────────┘
```

安装后第一件事就是 `ibstat` 看网卡有没有被认出来、端口 State 是不是 Active（见第 7 节）。

## 3. verbs API：用户态直接驱动网卡

**verbs** 是 RDMA 的核心抽象，"verb（动词）"=一个操作动作，如 post_send、poll_cq。`libibverbs` 是它的用户态实现库，上层框架（NCCL、UCX、MPI）几乎都最终落到 verbs。它解决的问题是：**给应用一个能从用户态直接安全地驱动网卡、收发数据的接口**。

理解 verbs 只需四个核心对象：

```
  ┌─ PD  (Protection Domain) 保护域：一组资源的"权限沙箱"，MR/QP都挂在某个PD下
  │
  ├─ MR  (Memory Region)  内存注册：把一块用户内存锁页+登记给网卡，
  │                       得到 lkey(本地钥匙)/rkey(远端钥匙)。网卡只能DMA注册过的内存
  │
  ├─ QP  (Queue Pair)     队列对 = 发送队列(SQ) + 接收队列(RQ)，
  │                       是通信的端点(类似socket，但成对)。两端QP互联后才能收发
  │
  └─ CQ  (Completion Queue) 完成队列：操作完成后网卡往这里塞一个CQE，
                            你 poll_cq 得知"刚才那个发送/接收干完了"
```

一次发送的生命周期（数据路径全程不进内核）：

```
  发送端                                          接收端
  ─────                                          ─────
  ① reg_mr 注册buffer                            ① reg_mr + post_recv 挂好收buffer
  ② post_send(把WR放进SQ)
  ③ 敲 doorbell 寄存器 ───────HCA硬件传输──────▶  网卡DMA写入接收buffer
  ④ poll_cq 拿到CQE(发完了)                       ④ poll_cq 拿到CQE(收到了)
```

**两种语义（务必分清）**：

| 语义 | 谁参与 | 远端CPU是否感知 | 典型用途 |
|------|--------|----------------|----------|
| **SEND/RECV**（双边） | 两端都要 post | 感知（消耗一个 recv WR） | 控制消息、握手 |
| **WRITE/READ**（单边，RDMA本色） | 只有发起端 post | **不感知**（CPU 0 参与） | 大块数据搬运、NCCL 主力 |

单边 RDMA WRITE 是性能关键：发起端拿着远端的 `rkey + 地址`，直接把数据写进对端内存，对端 CPU 完全不知情、不中断。这就是 NCCL 能把 GPU 间通信开销压到极低的底层原因。

**QP 传输类型**：RC（Reliable Connection，可靠有序，最常用）/ UD（Unreliable Datagram）/ UC 等，权衡是"可靠性 vs 每连接资源开销"。大规模训练里 RC 连接数 ~ O(N²)，节点多时要考虑这个开销，这也是 SHARP/DC 等机制存在的动机。

## 4. subnet manager（子网管理器）

IB 与以太网最大的不同之一：**IB 子网必须有一个 SM 才能工作**。以太网交换机即插即用、自学习 MAC；IB 链路起来后端口还是 Down 状态，**必须由 SM 给每个端口分配 LID（Local ID）并下发交换机转发表，整个子网才"通电"**。

```
  SM 干的活：
   1. 扫描(sweep)整个子网拓扑：发现所有HCA端口和交换机
   2. 给每个端口分配 LID（IB的"二层地址"，16bit）
   3. 计算路由，给每台交换机下发 forwarding table（LFT）
   4. 持续监控，拓扑变化(插拔/故障)时重新配置
```

```
   ┌──────┐ LID=1   ┌──────────┐  LID=3 ┌──────┐
   │ HCA  ├─────────┤ IB Switch├────────┤ HCA  │
   │node A│         │  (LFT)   │        │node B│
   └──────┘    ┌────┴──────────┘        └──────┘
               │
          ┌────┴────┐  OpenSM 跑在某台主机或交换机上，
          │   SM    │  通过 SMP(子网管理包)配置全网
          └─────────┘
```

**实现**：`OpenSM`（OFED 自带，软件 SM，跑在主机上）或交换机内置硬件 SM（如 NVIDIA Quantum 系列管理型交换机；大集群常用 **UFM** 统一管理）。

**routing 算法**（影响 fat-tree 拓扑下的拥塞）：常见有 `minhop`、`updn`、`ftree`（专为 fat-tree 优化，配合拓扑文件做无阻塞路由）、`dor` 等。AI 集群是规则的 fat-tree，**选 ftree 路由 + 正确的拓扑文件**对避免链路热点很关键——选错会导致 all-reduce 带宽被某条挤爆的链路拖死。

**实践要点**：(1) 一个子网只能有一个 active SM（可配 standby 做高可用，按 priority 选主）；(2) 没有 SM，`ibstat` 端口会停在 `Initializing` 而非 `Active`——这是最常见的"网卡好像没问题但就是不通"。

## 5. GPUDirect RDMA：显存直达

普通流程里，要把 GPU A 的显存发到远端 GPU B，数据得：`GPU A显存 → (PCIe拷到)主机内存 → 网卡发出 → 对端主机内存 → (再拷到)GPU B显存`。两次跨 PCIe 的额外拷贝 + CPU 参与，又慢又占 CPU。

**GPUDirect RDMA** 让网卡直接 DMA 读写 GPU 显存，**主机内存和 CPU 彻底出局**：

```
  没有 GPUDirect RDMA：                有 GPUDirect RDMA：
  ┌─────┐                              ┌─────┐
  │GPU A│──PCIe──▶ Host RAM            │GPU A│──┐
  └─────┘            │                 └─────┘  │ 网卡直接DMA显存
                     ▼                          ▼
                  ┌────┐                     ┌────┐
                  │ HCA│──网络──▶            │ HCA│──网络──▶ 对端GPU
                  └────┘                     └────┘
   3跳 + CPU忙 + 高时延                 1跳 + CPU=0 + 低时延
```

**底层机制**：网卡通过 PCIe 的 **P2P（peer-to-peer）** 能力直接访问 GPU 的 BAR 空间（显存暴露的物理窗口）。Linux 侧由 **`nvidia-peer-memory`**（旧）/ **DMA-BUF**（新内核机制）把"GPU 显存可被网卡 DMA"这件事告诉 RDMA 子系统，使得 `ibv_reg_mr` 能注册一段显存指针拿到 lkey/rkey。NVIDIA 称之为 PeerDirect RDMA 与 PeerDirect ASYNC（让网卡和 GPU 直接握手 doorbell，连 CPU 触发都省了）。

**生效前提（实践踩坑高发区）**：
- 安装 `nvidia-peer-memory` / 内核支持 DMA-BUF，且 `nv_peer_mem` 模块已加载；
- **GPU 与网卡挂在同一 PCIe Switch / 同一 NUMA 节点下**最优——跨 CPU socket 的 P2P 可能不支持或要绕 QPI，性能骤降甚至失败；
- 用 `nvidia-smi topo -m` 看 GPU↔NIC 是 `PIX/PXB`（同 switch，最好）还是 `SYS`（跨 NUMA，差）。

这是 NCCL 在多机训练里能跑满 IB 带宽的硬件基础。

## 6. perftest：怎么量带宽和时延

`perftest` 是 OFED 自带的微基准套件，**绕过所有上层框架，直接用 verbs 测裸网络性能**——排障第一工具，用来回答"到底是网络慢还是 NCCL 配错了"。

命名规律：`ib_{操作}_{指标}`。

| 工具 | 测什么 | 语义 |
|------|--------|------|
| `ib_write_bw` | 带宽 | RDMA WRITE（最能跑满，最常用） |
| `ib_read_bw` | 带宽 | RDMA READ |
| `ib_send_bw` | 带宽 | SEND/RECV |
| `ib_write_lat` | 时延 | WRITE，半往返/单次延迟 |
| `ib_send_lat` | 时延 | SEND |

用法：一端当 server，一端当 client 连过去。

```
  # server 端（node1）
  ib_write_bw -d mlx5_0 -i 1 --report_gbits
  # client 端（node2，连 server IP）
  ib_write_bw -d mlx5_0 -i 1 --report_gbits <node1_ip>
```

关键参数含义（讲含义不背默认值，具体以 `ib_write_bw -h` 为准）：

| 参数 | 含义 | 权衡 |
|------|------|------|
| `-d` | 选网卡设备（如 mlx5_0） | 多网卡机器要选对 |
| `-i` | 端口号 | 双口卡选对端口 |
| `-s` | 消息大小（字节） | 小包测时延、大包测带宽峰值 |
| `-q` | QP 数量 | 多 QP 才能压满多通道/高带宽 |
| `--report_gbits` | 用 Gb/s 报告 | 对齐网卡标称 |
| `-a` | 扫描所有消息大小 | 一次看完带宽随包大小的曲线 |

测出来的数要和网卡标称对比（见第 9 节）。若 `ib_write_bw` 都跑不满，别折腾 NCCL，先查物理层/SM/路由。

## 7. ibstat / ibstatus 及诊断全家桶

排障从"看物理层状态"开始。`infiniband-diags` 包提供一整套工具。

**`ibstat`**：看本机每个 HCA 端口的详细状态——最常用。

```
  $ ibstat
  CA 'mlx5_0'
    Port 1:
      State:          Active        ◀── 必须 Active；Initializing=缺SM；Down=物理没通
      Physical state: LinkUp        ◀── 物理链路是否点亮
      Rate:           200           ◀── 协商速率(Gb/s)；比标称低 = 线/口降速
      Base lid:       12            ◀── SM 分到的 LID；为0 说明SM没配
      LMC:            0
      SM lid:         3             ◀── 当前 SM 的 LID
      Link layer:     InfiniBand    ◀── 区分 IB 还是 Ethernet(RoCE)
```

**`ibstatus`**：更简洁的端口状态速览（State / rate / link_layer），快速扫一眼。

**诊断三步法**：

```
  物理层OK?  ──▶  ibstat / ibstatus     看 State=Active, Rate达标
      │
  子网通?   ──▶  ibping / ibhosts       两端能不能互探；列出全网HCA
      │       iblinkinfo               看每条链路速率(找降速口)
      │       ibnetdiscover            导出全网拓扑
      │
  有错包?   ──▶  perfquery / ibqueryerrors  看端口错误计数器
                                            (SymbolErrors/PortRcvErrors↑=坏线/坏口)
  端到端带宽? ─▶ ib_write_bw            (见第6节)
```

| 工具 | 一句话 |
|------|--------|
| `ibstat` / `ibstatus` | 本机端口状态（State/Rate/LID/link层） |
| `ibhosts` / `ibswitches` | 列出子网内所有 HCA / 交换机 |
| `iblinkinfo` | 逐链路速率，最快找出"降速到一半"的口 |
| `ibping` | 两端 LID 级连通性测试 |
| `perfquery` | 读/清端口错误与流量计数器 |
| `ibnetdiscover` | 扫描并导出全网拓扑 |
| `ibdiagnet` | 一键全网体检（链路/错误/SM/拓扑综合） |

经验法则：**`State` 卡在 `Initializing` → 先查 SM**；`Rate` 不达标 → 查线缆/接口/对端；`perfquery` 错误计数持续涨 → 换线/换口。

## 8. 与 NCCL / MPI 的关系

这是 AI 工程师最该搞清的一层：**你写训练代码时几乎不直接碰 verbs，你碰的是 NCCL；NCCL 才去碰 verbs**。

```
  PyTorch DDP / FSDP / Megatron
        │  调用集合通信原语 all_reduce / all_gather ...
        ▼
  ┌──────────────┐        ┌──────────────────┐
  │    NCCL      │  或者  │  MPI(HPC-X) + UCX │   ← 集合通信层
  └──────┬───────┘        └─────────┬────────┘
         │  NCCL 的 IB plugin              │ UCX 的 ib transport
         ▼                                 ▼
  ┌───────────────────────────────────────────┐
  │           libibverbs (verbs API)           │  ← 都落到这
  └───────────────────────┬───────────────────┘
                          ▼
                  OFED 驱动 → HCA 硬件
   配合 GPUDirect RDMA：NCCL 直接在 GPU 显存间用 RDMA WRITE 传输
```

- **NCCL**：NVIDIA 的 GPU 集合通信库，深度学习多卡/多机训练的事实标准。它内部有 IB 传输后端，**直接调 verbs 做 RDMA WRITE**，并依赖 GPUDirect RDMA 实现显存直达。常用环境变量：`NCCL_IB_HCA`（指定用哪些网卡）、`NCCL_IB_GID_INDEX`（RoCE 时选 GID）、`NCCL_NET_GDR_LEVEL`（控制 GPUDirect 启用条件）。
- **HPC-X**：NVIDIA 的综合 HPC 包，含 OpenMPI、**UCX**（统一通信框架，做 verbs 之上的传输选择/抽象）、SHARP、HCOLL。MPI 程序（部分 HPC/科学计算、少数 LLM 框架）走这条路。
- **SHARP**：交换机内做 in-network reduction（在交换机芯片里直接把多份梯度求和），把 all-reduce 的一部分计算卸载到网络，进一步降通信量——大集群训练的加速点。

一句话关系：**OFED 给地基，verbs 是统一接口，NCCL/MPI 是你直接用的门面，GPUDirect 让显存直达，SM 让网络通电，perftest/ibstat 帮你证明这条链路是好的。**

## 9. 数值例子 / 对照 / 实践

### 手算 1：单条 IB 链路的理论带宽

NDR 单端口标称约 **400 Gb/s**。换算成有效数据吞吐（字节）：

$$
400\ \text{Gb/s} \div 8 = 50\ \text{GB/s（裸链路）}
$$

IB 用 **64b/66b** 编码（NDR），有效率约 $64/66 \approx 96.97\%$，再扣协议头开销，实际可用约：

$$
50 \times 0.97 \approx 48.5\ \text{GB/s（约，以 perftest 实测为准）}
$$

所以 `ib_write_bw --report_gbits` 测到接近 ~390+ Gb/s 才算健康；如果只有一半（~200），多半是协商降速或路由热点。

### 手算 2：all-reduce 通信量与耗时

环形 all-reduce（ring）传输的数据量约为：

$$
V_{\text{comm}} = 2 \times \frac{N-1}{N} \times S
$$

其中 $S$ 是梯度大小，$N$ 是参与的 GPU 数。设 7B 模型 fp16 梯度 $S \approx 14\ \text{GB}$，$N=8$：

$$
V \approx 2 \times \frac{7}{8} \times 14 = 24.5\ \text{GB}
$$

若每张卡有效带宽 $B \approx 45\ \text{GB/s}$，理想耗时：

$$
t \approx \frac{V}{B} = \frac{24.5}{45} \approx 0.54\ \text{s（理想，忽略时延与重叠）}
$$

结论：**带宽每降一半，all-reduce 时间翻倍**，直接吃掉训练吞吐——这就是为什么前面 SM 路由选错、链路降速这些"小事"在大规模训练里是大事。

### 手算 3：GPUDirect 省了什么

不开 GPUDirect 时每次传输多两跳 PCIe 拷贝。设单次 16 KB 消息，多两次 PCIe（约 16 GB/s）拷贝：

$$
t_{\text{额外}} \approx 2 \times \frac{16\ \text{KB}}{16\ \text{GB/s}} = 2 \times 1\ \mu s = 2\ \mu s
$$

外加 CPU 调度的不确定时延。小消息密集场景（如频繁同步），这 $\mu s$ 级开销累积起来很可观，开 GPUDirect 直接抹掉。

### 软件栈层次对照

| 层 | 组件 | 内核/用户态 | 在数据路径上? |
|----|------|------------|--------------|
| 应用 | PyTorch/Megatron | 用户 | 否（发起者） |
| 集合通信 | NCCL / HPC-X | 用户 | 是（编排） |
| verbs | libibverbs | 用户 | 是 |
| 驱动 | mlx5_core/ib_core | 内核 | **否（仅控制路径）** |
| 硬件 | HCA + Switch | — | 是（真正搬数据） |

### 实践 checklist

```
  [ ] ibstat：所有端口 State=Active, Rate 达标, Base lid≠0
  [ ] 子网有且仅一个 active SM（OpenSM/UFM），fat-tree 用 ftree 路由
  [ ] nvidia-smi topo -m：GPU↔NIC 是 PIX/PXB（同switch），不是 SYS
  [ ] nv_peer_mem 模块已加载（GPUDirect RDMA 生效）
  [ ] ib_write_bw 实测接近网卡标称（先证明裸网络好，再调 NCCL）
  [ ] perfquery：端口错误计数器不持续增长
  [ ] NCCL_IB_HCA 指定正确网卡；NCCL_DEBUG=INFO 确认走了 IB 而非回退 TCP
```

## 常见问题

| 问题 | 原因 / 答案 |
|------|------------|
| 网卡 LinkUp 但 State=Initializing，不通 | **缺 SM**。子网没有 active subnet manager，端口拿不到 LID。起 OpenSM 或检查 UFM |
| inbox driver 还是 MLNX_OFED？ | 小规模/云上 inbox 够用；大规模 GPU 训练用 MLNX/DOCA-OFED（配套 GPUDirect、工具全、调优好） |
| NCCL 慢，怀疑网络 | 先 `ib_write_bw` 测裸带宽。裸的就慢→查物理/SM/路由；裸的快但 NCCL 慢→查 NCCL 配置/GPUDirect/拓扑 |
| GPUDirect RDMA 没生效 | 查 `nv_peer_mem` 是否加载、`nvidia-smi topo -m` GPU 与 NIC 是否同 switch、是否跨 NUMA |
| Rate 比标称低一半 | 协商降速：线缆质量/接口脏/对端口速率不匹配；`iblinkinfo` 定位降速链路 |
| verbs 和 NCCL 啥关系 | verbs 是底层统一接口；NCCL 内部调 verbs 做 RDMA WRITE。你写代码用 NCCL，不直接写 verbs |
| RoCE 和 IB 软件栈通用吗 | OFED/verbs 接口通用（同一套 libibverbs）；但 RoCE 跑在以太网上，无需 SM，靠 GID/PFC/ECN 配 lossless，配置差异大 |
| perfquery 错误计数一直涨 | 物理层有坏线/坏口/坏光模块，先 clear 再观察，定位后换硬件 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引
- [[ai-infra/网络/InfiniBand]] — IB 硬件/协议/拓扑（本篇是它的"软件配套"）
- [[ai-infra/网络/通信软件]] — NCCL/MPI/UCX 等集合通信层（本篇第 8 节的上层）
