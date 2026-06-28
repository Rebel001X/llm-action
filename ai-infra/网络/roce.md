# RoCE (RDMA over Ethernet)

> 用「标准以太网」当物理网络，跑「RDMA(远程直接内存访问)」语义——让 GPU/网卡绕过 CPU 和内核，直接读写远端主机内存，做到 InfiniBand 级别的低时延高带宽，又复用了以太网的成本与运维生态。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/网络/InfiniBand]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | RDMA + 以太网 |
| 1 | 地基：什么是 RDMA、为什么要它 | kernel bypass / zero-copy / DMA |
| 2 | RoCEv1 vs RoCEv2 的本质区别 | L2 封装 vs UDP/IP 封装 |
| 3 | RoCEv2 报文是怎么一层层包起来的 | 协议栈分层 / BTH / 端口4791 |
| 4 | 一次 RDMA Write 在线路上发生了什么 | QP / WQE / CQ / 数据流 |
| 5 | 无损网络：PFC + ECN(DCQCN) | 流控 / 拥塞控制 / PAUSE |
| 6 | RoCE vs InfiniBand 全面对比 | 成本 / 性能 / 运维 |
| 7 | 大规模训练里 RoCE 的用途 | 集合通信 / 轨道拓扑 / 万卡 |
| 8 | OFED 驱动与 perftest 实操 | mlx5 / ib_send_bw |
| 数值 | 带宽/时延手算 | All-Reduce 时间 |
| FAQ | 易错点速查 | — |

---

## 0. 一句话锚点

**RoCE = RDMA over Converged Ethernet。** RDMA 是「能力」(让网卡直接搬内存、不打扰 CPU)，以太网是「载体」。RoCE 就是把原本跑在 InfiniBand 专网上的 RDMA 协议，**塞进以太网的帧里**，从而在普通(但高端)以太网交换机上获得近乎 IB 的性能。当前主流是 **RoCEv2**(基于 UDP/IP，可跨网段)。

---

## 1. 地基：先讲清 RDMA，再讲 RoCE

### 1.1 传统 TCP/IP 网络收发数据有多「贵」

传统 socket 发一段数据，CPU 要做的事：用户态 buffer → 拷到内核 socket buffer → 内核 TCP/IP 协议栈处理 → 拷到网卡 DMA 区。收方反着再来一遍。代价是：

- **多次内存拷贝**(user↔kernel)，吃内存带宽。
- **CPU 全程参与**：协议栈、中断、上下文切换都烧 CPU。
- **时延高**：软件路径长，微秒级甚至几十微秒抖动。

在万卡训练里，每一步都要做 All-Reduce 同步梯度，CPU 一旦成为瓶颈，GPU 就空转——这是不可接受的。

### 1.2 RDMA 三板斧

RDMA(Remote Direct Memory Access)用三个机制把上面的开销砍掉：

1. **Kernel Bypass(内核旁路)**：应用直接通过用户态 verbs 库把「工作请求」投递给网卡，不进内核协议栈。
2. **Zero-Copy(零拷贝)**：网卡用 DMA 直接从应用注册过的内存区(MR)搬数据，不经中转 buffer。
3. **CPU Offload(协议卸载)**：传输层(可靠性、重传、分段)由网卡硬件(NIC 上的 RDMA 引擎)完成，CPU 投递完请求就可以去干别的。

```
传统 TCP/IP 收发(CPU 全程在场):
 App ──copy──> Kernel socketbuf ──TCP/IP 栈──> NIC ──> 线
                  ^ 多次拷贝 ^ 协议栈烧 CPU ^ 中断

RDMA(CPU 投递后离场, 网卡直达内存):
 App ─注册内存(MR)─> [NIC RDMA 引擎] ═DMA═> 远端 App 内存
        投递 WQE          硬件做可靠传输       不打扰远端 CPU
```

> 关键认知：**RDMA 是一套语义/协议(QP、verbs、可靠传输)，它不绑定物理介质。** 跑在 IB 物理网上叫 InfiniBand；跑在以太网上就叫 RoCE；跑在 TCP 上叫 iWARP。本文讲的就是「以太网这条腿」。

---

## 2. RoCEv1 vs RoCEv2：本质是「封装在哪一层」

RDMA 协议本身有自己的传输头(BTH 等)，问题是：**怎么把它放进以太网传出去？** 这就是 v1 和 v2 的分水岭。

```
RoCEv1(链路层封装, 几乎被淘汰):
 ┌────────────┬──────────────┬─────────┐
 │ Ethernet头 │ IB 传输头+载荷 │  FCS    │   EtherType=0x8915
 └────────────┴──────────────┴─────────┘
 ✗ 没有 IP 头 → 不能路由 → 只能在同一个二层广播域(同网段)内通信
   → 规模受限, 基本无实际部署

RoCEv2(UDP/IP 封装, 当前主流):
 ┌────────┬───────┬──────────┬──────────────┬──────┐
 │Eth头   │ IP头  │ UDP头    │ IB 传输头+载荷 │ FCS  │   UDP 目的端口=4791
 └────────┴───────┴──────────┴──────────────┴──────┘
 ✓ 有 IP 头 → 可被三层路由 → 可跨网段, 可建大规模 Spine-Leaf 网络
 ✓ 有 UDP 头 → 源端口可做 ECMP 哈希熵, 多路径负载均衡
```

**结论(与原文一致)**：RoCEv1 基于网络链路层，无法跨网段，基本无应用；**RoCEv2 基于 UDP，可跨网段、扩展性好、吞吐与时延优秀，是大规模采用的方案。** 后文「RoCE」默认指 RoCEv2。

为什么选 **UDP 而不是 TCP**？因为 RDMA 的可靠性(确认、重传、保序)由网卡硬件按 IB 传输语义自己做了，不需要 TCP 那套软件可靠性叠加；UDP 只是借它的端口号穿越 IP 网络、并给 ECMP 提供哈希熵。目的端口固定 **4791**。

---

## 3. RoCEv2 协议栈：报文一层层是怎么包的

把一个 RoCEv2 包从外到内拆开，对照 OSI：

```
   层               RoCEv2 字段              作用
 ┌──────────────┬───────────────────────┬──────────────────────────┐
 │ L2 数据链路  │ Ethernet MAC 头        │ 局域寻址 + VLAN/PCP(给PFC)│
 │ L3 网络      │ IP 头(DSCP/ECN 位)     │ 跨网段路由 + 标记拥塞     │
 │ L4 传输      │ UDP 头(dport=4791)     │ 穿透 IP 网 + ECMP 熵      │
 │ RDMA 传输    │ IB BTH(Base Transport) │ 操作码/QP号/PSN(序号)     │
 │ (扩展头)     │ RETH/AETH/ImmDt 等     │ 远端虚拟地址/ACK/立即数   │
 │ 载荷         │ Payload(用户数据)      │ 真正要搬的内存内容        │
 │ 校验         │ ICRC + FCS             │ RDMA端到端校验 + 以太网帧校验│
 └──────────────┴───────────────────────┴──────────────────────────┘
```

几个要点(每个都说「为什么」)：

- **BTH(Base Transport Header)**：里面有 `OpCode`(是 SEND / WRITE / READ)、`Destination QP`(目标队列对)、`PSN`(Packet Sequence Number，丢包检测与保序的核心)。这是 RDMA 「传输层」的灵魂。
- **RETH**：RDMA Extended Transport Header，携带**远端虚拟地址 + R_Key + 长度**——这就是「直接写远端内存」的地址簿。
- **DSCP / ECN(在 IP 头)**：DSCP 用来做流量分类(把 RoCE 流量打进无损队列)；ECN 两位用来在交换机拥塞时打标记，触发拥塞控制(见第 5 节)。
- **ICRC**：端到端硬件校验，保证数据没被中间链路损坏，这是 RDMA 可靠性的一部分，独立于以太网 FCS。

> 一句话：RoCEv2 = 「以太网/IP/UDP 的外壳」+「InfiniBand 传输层(BTH 及扩展头)的内核」。外壳负责在以太网世界里送达，内核负责 RDMA 语义。

---

## 4. 一次 RDMA Write 在线路上发生了什么

RDMA 的编程模型核心对象：

- **QP(Queue Pair，队列对)** = 发送队列(SQ) + 接收队列(RQ)。一条「连接」就是两端各一个 QP 绑定。
- **WQE(Work Queue Element)** = 一条工作请求(「把这块内存写到对端那个地址」)。
- **CQ(Completion Queue)** = 完成队列，网卡干完活往里塞一个 CQE(完成事件)，应用来轮询/等待。
- **MR(Memory Region)** = 提前向网卡注册并 pin 住的内存区，配 L_Key/R_Key 做权限校验。

```
 主机 A(发起方)                              主机 B(目标方)
 ┌───────────────┐                          ┌───────────────┐
 │ App: 填 WQE   │                          │  App(不参与!) │
 │  目标地址/R_Key│                          │   CPU 空闲     │
 │      │post     │                          └──────▲────────┘
 │      ▼         │   RoCEv2 包(走以太网)          │ DMA 直写
 │  ┌────────┐    │ ═════════════════════════════> │
 │  │ SQ │QP │    │   Eth/IP/UDP/BTH/RETH/Payload  │
 │  └────────┘    │                                 │
 │   NIC 硬件做    │ <════════ ACK(AETH, 带 PSN)════ │ NIC 回 ACK
 │   可靠传输      │                                 │
 │      │         │                                 │
 │  ┌────────┐    │                                 │
 │  │  CQ    │<── 网卡塞 CQE(完成)                  │
 │  └────────┘    │                                 │
 └───────────────┘                          └───────────────┘
```

注意 **RDMA Write 是「单边操作」**：B 的 CPU 全程不知情，数据被网卡直接 DMA 进 B 预先注册的内存。这正是 GPU 训练里 GPUDirect RDMA 想要的——网卡可直接读写 **GPU 显存**(配合 nv_peer_mem / dmabuf)，数据从 A 的 GPU 显存直达 B 的 GPU 显存，全程不碰主机 CPU 与系统内存。

---

## 5. 无损网络：RoCE 跑得好的前提

### 5.1 为什么 RoCE「怕丢包」

InfiniBand 物理网天生是**无损**(link-level credit 流控，发送方有信用才发，从不溢出丢包)。但以太网默认是**有损**的——交换机缓冲满了就直接丢包。

而经典 RoCE 用的是 **Go-Back-N** 重传：丢一个包，要从该包开始**整段重传**，性能断崖式下跌。所以传统结论是：**RoCE 必须运行在「无损以太网」上**。无损靠两个机制叠加。

### 5.2 PFC(Priority Flow Control，优先级流控)——防止「缓冲溢出丢包」

PFC 是 IEEE 802.1Qbb，**逐跳、基于优先级**的反压：当下游交换机某个优先级队列(比如专门给 RoCE 的队列)快满了，它向上游发 **PAUSE 帧**，让上游**暂停发送该优先级**的流量,直到缓冲降下来。

```
PFC 反压链(只暂停 RoCE 那条优先级, 不影响普通流量):
 上游交换机 ──RoCE流量──> 下游交换机[队列将满!]
     ▲                          │
     └────── PAUSE 帧(优先级x)───┘   "停一下别发了"
 效果: 缓冲不会溢出 => 不丢包(无损)
```

- **为什么按优先级**：你不想因为 RoCE 拥塞把整条链路(包括管理流量)全停了。用 DSCP/PCP 把 RoCE 流量分到专属队列，只对它做 PFC。
- **PFC 的副作用(运维痛点)**：PAUSE 是逐跳的，可能层层往上传播，导致 **PFC 风暴 / 死锁 / 受害者流(victim flow)**——所以 PFC 是「兜底」,真正主力是下面的 ECN。

### 5.3 ECN + DCQCN(拥塞控制)——从源头降速,少触发 PFC

PFC 是「堵到家门口才喊停」，太粗暴。更优雅的是**端到端拥塞控制**：

1. 交换机队列开始堆积(超过阈值)→ 在经过的包 IP 头里**置 ECN 标记**(不是丢包，只是打个戳)。
2. 接收方收到带 ECN 标记的包 → 回一个 **CNP(Congestion Notification Packet)** 给发送方。
3. 发送方收到 CNP → **主动降低发送速率**(乘性减小)；一段时间没拥塞就慢慢加速(加性增大)。

这套算法在 Mellanox/NVIDIA 网卡上叫 **DCQCN(Data Center QCN)**，是 RoCE 大规模部署的事实标准。

```
ECN/DCQCN 闭环(在丢包之前就降速):
 发送方 ──数据──> [交换机:队列堆积] ──打ECN标记──> 接收方
    ▲                                                  │
    └──────────────── CNP(拥塞通知)────────────────────┘
         收到 CNP => 降速 (避免堆到要 PFC PAUSE)
```

**PFC 与 ECN 的分工**：ECN/DCQCN 是「主力调速」，平滑地把速率降到拥塞点以下；PFC 是「最后防线」，在 ECN 来不及反应、缓冲真要溢出时兜底，确保**绝不丢包**。两者配合才是合格的「无损以太网」。

> 趋势补充(以厂商方案为准)：近年出现 **DCN/Spectrum-X、自适应路由 + 改进重传(如 Selective Repeat)** 等技术，目标是让 RoCE 对少量丢包有更强容忍度，弱化对「严格无损」的依赖。但当前主流生产网仍以 PFC+ECN 无损为基线。

---

## 6. RoCE vs InfiniBand：成本 / 性能 / 运维

二者**应用层都是 RDMA verbs**(NCCL 之类上层软件几乎无感),区别在物理网络与生态。

```
            InfiniBand 专网                 RoCE(以太网)
 物理层    IB 专用交换机/线缆/HCA          标准以太网交换机/网卡
 流控      链路层 credit(天生无损)         PFC + ECN(后天调成无损)
 管理      子网管理器 SM(集中控制)         传统 IP/以太网运维(分布式)
 路由      IB 自带 LID/路由                IP 路由 + ECMP/自适应路由
 生态      相对封闭(头部厂商主导)          开放(多厂商以太网生态)
```

| 维度 | InfiniBand | RoCEv2 | 解读 |
|---|---|---|---|
| **裸性能** | 极佳，时延最低，原生无损 | 接近 IB，调好后差距小 | IB 是性能天花板；RoCE 调优后实战可打平 |
| **时延** | 端到端约 1~2μs(以官方为准) | 约 2~5μs 量级(以官方为准) | IB 略低；多数训练对这点差异不敏感 |
| **带宽** | 同代对齐(如 400G NDR) | 同代对齐(400G 以太网) | 单端口带宽两者同档次 |
| **成本** | 较高(专用交换机/线缆，绑定生态) | 较低(以太网规模效应，多供应商) | RoCE 的最大卖点之一 |
| **运维** | SM 集中、相对省心但需懂 IB | 复用以太网技能，但 PFC/ECN 调参难 | RoCE「门槛低、调优深」 |
| **扩展性** | 大规模成熟(超算/AI 集群) | Spine-Leaf 可达万卡级 | 两者都能上万卡 |
| **生态开放性** | 相对封闭 | 开放，避免单一供应商锁定 | 大厂自建网络常倾向 RoCE |

**一句话选型**：追求**极致、省心、预算充足**→ IB；追求**成本、开放、复用以太网生态、且有能力调 PFC/ECN**→ RoCE。当下两条路线在 AI 集群里都大规模存在。

---

## 7. 大规模训练里 RoCE 干什么

### 7.1 它服务的是「集合通信」

分布式训练每一步都要做 [[ai-infra/网络/集合通信原语]] 里的 **All-Reduce / All-Gather / Reduce-Scatter** 来同步梯度/参数。这些原语由 NCCL 实现，NCCL 底层就走 RDMA(RoCE 或 IB)。**RoCE 提供的就是这条「GPU 到 GPU、跨节点」的高速管道。**

```
数据并行一步的网络负载(梯度同步是大头):
 GPU 计算反向 → 产出梯度 → [All-Reduce 跨所有节点] → 更新参数 → 下一步
                              ▲ 这一步全靠 RoCE/IB 把梯度搬来搬去
 节点内: NVLink/NVSwitch     节点间: RoCE/IB
```

### 7.2 典型拓扑：轨道优化(Rail-optimized)的 Spine-Leaf

万卡集群常用 **rail-optimized** 设计：每台服务器 8 张 GPU、配 8 个 RoCE 网口，**第 i 个网口接到第 i 个 leaf(rail)交换机**。这样同号 GPU 之间一跳直达，NCCL 的环/树能高效铺在网络上。详见 [[ai-infra/网络/InfiniBand]] 与 Spine-Leaf 资料。

```
Rail-optimized(8 rail 示意, 每服务器每个网口归属一个 rail):
        Spine(脊)交换机层
        /    |    |    \
   Leaf0  Leaf1 ...  Leaf7     <- 8 个 rail
     |      |          |
  GPU0口  GPU1口 ... GPU7口     <- 服务器 A
  GPU0口  GPU1口 ... GPU7口     <- 服务器 B
 同号 GPU 走同一 rail, 一跳可达, 集合通信带宽利用率高
```

### 7.3 为什么训练特别在意 RoCE 调好

训练流量特点：**突发、同步、大象流(elephant flow)**——成千上万 GPU 几乎同时开始 All-Reduce，瞬间打满网络。这种「incast(多打一)」最容易触发拥塞→PFC PAUSE→性能塌方。所以**ECN 阈值、PFC 水线、DSCP 分类、ECMP/自适应路由**的调优，直接决定训练的「网络利用率」和扩展效率。RoCE 性能好不好，七分靠调。

---

## 8. OFED 驱动与 perftest 实操(承接原文)

### 8.1 OFED 软件栈

用 RoCEv2 之前要装 **OFED**(OpenFabrics Enterprise Distribution)——一个开源软件栈，面向 HPC 与数据中心的高性能网络，是一组用于 InfiniBand 与以太网 RDMA 的**软件包 + 驱动**集合。NVIDIA 网卡通常装 **MLNX_OFED / DOCA-OFED**(以官方为准)，提供 `mlx5` 驱动、用户态 `libibverbs` verbs 库、管理与诊断工具。

```
RoCE 软件栈(从上到下):
 ┌──────────────────────────────┐
 │ 应用 / NCCL / MPI            │
 ├──────────────────────────────┤
 │ libibverbs(用户态 verbs API) │  <- kernel bypass 在这层之下发生
 ├──────────────────────────────┤
 │ OFED 内核驱动(mlx5_core/ib)  │
 ├──────────────────────────────┤
 │ NIC 硬件(RoCE 引擎 + 以太网) │
 └──────────────────────────────┘
```

### 8.2 perftest 性能测试

`perftest` 是 OFED 自带的 RDMA 性能测试工具集，专门测 RDMA 性能。`-d` 指定 RDMA 设备(网卡)：

```bash
# 服务端(先起, 监听)
ib_send_bw -d mlx5_0

# 客户端(连服务端 IP)
ib_send_bw -d mlx5_1 10.251.30.207
```

常用工具(按操作语义区分)：

```
ib_send_bw  / ib_send_lat   : SEND 操作的带宽 / 时延
ib_write_bw / ib_write_lat  : RDMA WRITE(单边写)带宽 / 时延
ib_read_bw  / ib_read_lat   : RDMA READ(单边读)带宽 / 时延
```

实战排查清单：① `ibv_devinfo` 看网卡与端口是否 Active；② 确认两端 **DSCP/PCP 分类一致**、PFC/ECN 都开在同一优先级；③ 用 `ib_write_bw` 验证能否打满线速；④ 真业务用 `nccl-test`(见 [[ai-infra/网络/集合通信原语]] 与本目录 nccl-test 文档)看集合通信带宽。

---

## 数值例子 / 手算

### 例 1：单端口理论带宽 vs 实测

400G 以太网端口，线速 $400\,\text{Gbps} = 50\,\text{GB/s}$。RDMA 单向能跑到约 90~95% 线速，取 0.92：

$$
B_{\text{eff}} \approx 50 \times 0.92 = 46\ \text{GB/s}
$$

### 例 2：一次 All-Reduce 要多久(Ring 算法)

设 $N$ 个 GPU 跑 Ring-AllReduce，梯度总量 $S$，每端口有效带宽 $B$。Ring-AllReduce 每个节点收发的数据量约 $2\cdot\frac{N-1}{N}\cdot S$，总传输时间(带宽项)近似：

$$
T_{\text{bw}} \approx \frac{2(N-1)}{N}\cdot\frac{S}{B}
$$

代入：175B 参数模型，梯度按 FP16 计 $S = 175\times10^9 \times 2\,\text{B} = 350\,\text{GB}$；$N=8$，$B=46\,\text{GB/s}$：

$$
T_{\text{bw}} \approx \frac{2\times7}{8}\times\frac{350}{46}
= 1.75 \times 7.6 \approx 13.3\ \text{秒}
$$

> 解读：单看带宽项就要约 13 秒——所以真实训练会做**梯度分桶/重叠通信(overlap)**、用多端口聚合、用 BF16/梯度压缩来把这数压下去。也说明**端口数 × 单口带宽**是训练扩展的硬约束，RoCE 把带宽堆上去直接决定能不能扩。

### 例 3：时延量级直觉

跨节点一次 RDMA Write 单边时延约 $2\sim5\,\mu s$(以官方为准)。对比传统 TCP 往返动辄几十微秒并带抖动——在每步要做大量同步的训练里，这点差异乘以海量次数就很可观。

---

## 常见问题

| 问题 | 速答 |
|---|---|
| RoCE 和 RDMA 什么关系？ | RDMA 是能力/协议，RoCE 是「把 RDMA 跑在以太网上」的一种实现(另有 IB、iWARP)。 |
| v1 和 v2 选哪个？ | 一律 v2。v1 是二层封装、不能跨网段、基本无部署。 |
| RoCEv2 为什么用 UDP 不用 TCP？ | 可靠性由网卡按 IB 语义硬件实现，不需 TCP；UDP 只为穿透 IP 网 + 给 ECMP 提供哈希熵；端口固定 4791。 |
| 不开 PFC/ECN 行不行？ | 经典 RoCE 用 Go-Back-N，丢包代价极大，所以需要无损网络兜底；新方案在弱化此依赖，但生产基线仍是无损。 |
| PFC 和 ECN 谁是主力？ | ECN/DCQCN 端到端调速是主力；PFC 逐跳 PAUSE 是「绝不丢包」的最后防线。 |
| RoCE 能打平 InfiniBand 吗？ | 裸性能 IB 略优(时延更低、原生无损)；RoCE 调优后实战可接近，胜在成本与开放生态。 |
| GPU 显存能直达吗？ | 能，配合 GPUDirect RDMA，网卡直接 DMA 读写远端 GPU 显存，不碰主机 CPU/内存。 |
| 训练里 RoCE 最大的坑？ | incast(多打一)突发触发拥塞 → PFC 风暴/victim flow；要靠 ECN 阈值、水线、ECMP/自适应路由细调。 |
| 用什么测？ | OFED 的 perftest(ib_send/write/read_bw/lat)测点对点；nccl-test 测真集合通信带宽。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[ai-infra/网络/InfiniBand]] — RoCE 的「专网对手」，理解无损与拓扑的对照基准
- [[ai-infra/网络/集合通信原语]] — RoCE 真正服务的上层：All-Reduce/All-Gather 等

> 参考：AI场景下高性能网络技术 RoCE v2 介绍 — https://mp.weixin.qq.com/s/XyMFst3w-d65u4fU7cgLPA
