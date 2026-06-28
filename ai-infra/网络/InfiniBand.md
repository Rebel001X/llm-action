# InfiniBand 与 RoCE

> 一种为高性能计算（HPC）与大规模分布式训练而生的低时延、高带宽网络：靠 RDMA 让网卡直接读写远端内存，绕过 CPU 与内核，把"搬数据"这件事从软件成本里拿掉。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你会得到什么 | 关键直觉 |
|---|---|---|
| 0 | 一句话锚点 | IB ≈ 专用网络 + RDMA；RoCE ≈ 在以太网上跑 RDMA |
| 1 | 地基：传统 TCP/IP 为什么慢 | 拷贝多次 + 中断 + 内核态切换 |
| 2 | RDMA 原理 | 零拷贝、kernel bypass、CPU offload |
| 3 | Verbs / QP / WQE 编程模型 | 网卡是可编程引擎，不是哑管道 |
| 4 | IB vs RoCE vs 以太网 | 谁负责无损？谁负责拥塞？ |
| 5 | 带宽与时延数字 | SDR→NDR 的演进、单口/单机带宽手算 |
| 6 | 胖树（Fat-Tree）拓扑 | 为什么要"无收敛"，怎么数交换机 |
| 7 | GPUDirect RDMA | 让网卡直接读 GPU 显存，连主机内存都跳过 |
| 8 | 为什么大规模训练离不开它 | 通信占比、AllReduce 时间手算 |
| — | 数值示例 / 对照表 / 常见问题 | 把上面串成数字 |

## 0. 一句话锚点

- **InfiniBand（IB）**：一套**端到端的专用网络体系**（网卡 HCA + 交换机 + 线缆 + 子网管理器 SM），原生支持 RDMA，链路层天然**无损（lossless）**。
- **RDMA（Remote Direct Memory Access）**：远程直接内存访问。**一台机器的网卡可以直接把数据写进/读出另一台机器的内存，全程不打扰对方 CPU、不进对方内核**。这是 IB 的灵魂。
- **RoCE（RDMA over Converged Ethernet）**：把 RDMA 这套语义**搬到以太网**上跑。RoCEv2 用 UDP/IP 封装，可路由。好处是复用以太网生态，代价是要自己在以太网上"造"出无损环境（PFC/ECN 等）。

> 一句话区分：**IB 是"网络 + RDMA 一整套全家桶"；RoCE 是"借以太网的壳，跑 RDMA 的魂"。** 二者上层都长得像 RDMA，差别在链路/网络层谁来保证不丢包、谁来管拥塞。

## 1. 地基：传统 TCP/IP 网络为什么"慢"（从最底层讲起）

要理解 RDMA 的价值，先看传统 socket 收一个包，数据到底走了几趟。假设进程 A 要把内存里的一段数据发给进程 B：

```
传统 TCP/IP 发送路径（每个箭头都可能是一次内存拷贝或一次态切换）

  App Buffer (用户态)
        | ① copy 用户态->内核 socket buffer   (CPU 拷贝)
        v
  Kernel Socket Buffer
        | ② 协议栈处理 TCP/IP，加头           (CPU 算校验和等)
        v
  Driver / DMA 描述符
        | ③ DMA 到网卡                         (DMA)
        v
  NIC --------- 线缆 --------- NIC (对端)
                                  | ④ DMA 进内核 buffer
                                  v
                            Kernel Socket Buffer
                                  | ⑤ 协议栈处理 + ⑥ copy 到用户态
                                  v
                            App Buffer (对端用户态)
```

代价来自三处，每一处都"为什么慢"：

1. **多次内存拷贝**：①⑤⑥ 都是 CPU 参与的字节搬运。10 GB 数据拷一遍就要占满内存带宽一段时间，CPU 啥正事都干不了。
2. **内核态/用户态切换**：每次 `send()/recv()` 系统调用要陷入内核，上下文切换 + cache 污染。
3. **中断驱动**：包到了，网卡发中断，CPU 停下手头工作去处理。高速率下每秒上百万包，中断风暴直接吃光 CPU。

**结论（为什么需要新东西）**：在 100/200/400 Gb/s 这种速率下，"让 CPU 当搬运工"这条路根本走不通——CPU 会成为瓶颈，而且本该用来算梯度的算力被浪费在收发包上。RDMA 就是来拆掉这三座大山的。

## 2. RDMA 原理：零拷贝 + 内核旁路 + CPU 卸载

RDMA 的三个核心承诺，逐个拆到最原子：

### 2.1 Zero-Copy（零拷贝）
网卡（HCA）直接对**用户态注册过的内存**做 DMA，数据不再经过内核 socket buffer 中转。发送端的 App Buffer → 网卡 → 线缆 → 对端网卡 → 对端 App Buffer，**中间没有一次 CPU 字节拷贝**。

### 2.2 Kernel Bypass（内核旁路）
数据路径（data path）完全绕过内核协议栈。应用通过预先映射好的"门铃（doorbell）"寄存器直接通知网卡干活。**只有建立连接、注册内存这种慢速控制路径（control path）才进内核一次；之后每次收发都不进内核。**

### 2.3 CPU Offload / Transport Offload（卸载）
TCP 里"重传、排序、校验、拥塞控制"这些活，在 IB 里**由网卡硬件做**。CPU 发起一个操作后就可以走人，操作完成由网卡写一个完成事件（CQE）回来。

```
RDMA 数据路径（对端 CPU 全程不参与！）

  App Buffer A (用户态, 已注册)
        |  网卡直接 DMA 读 (零拷贝)
        v
  HCA(A) --- IB/RoCE 线缆 --- HCA(B)
                                |  网卡直接 DMA 写 (零拷贝)
                                v
                          App Buffer B (用户态, 已注册)
                                ↑
                        对端 CPU/内核：呼呼大睡 😴
```

**关键点："单边操作"**：`RDMA WRITE` / `RDMA READ` 是**单边（one-sided）**的——发起方知道对端的内存地址（virtual addr）和访问钥匙（rkey），就能直接读写，对端 CPU 完全无感。这与"双边（two-sided）"的 `SEND/RECV`（对端要预先 post 一个 recv buffer）不同。AllReduce 等集合通信大量用到单边语义。

## 3. 编程模型：Verbs / QP / WQE（网卡是可编程引擎）

RDMA 的软件接口叫 **Verbs**（动词）。核心抽象：

- **QP（Queue Pair，队列对）**：一对队列 = 发送队列 SQ + 接收队列 RQ。它是 RDMA 的"连接端点"，类似 socket。
- **WQE（Work Queue Element，读作 "wookie"）**：你想让网卡干的一件活（"把这块内存写到对端那个地址"）就是一个 WQE，post 进 SQ。
- **CQ（Completion Queue）/ CQE**：网卡干完后往 CQ 里放一个完成项 CQE，应用 poll CQ 得知"做完了"。
- **MR（Memory Region）**：内存必须先**注册（register）**，网卡才知道这块内存的物理页、并 pin 住不让换出，同时生成 `lkey`（本地钥匙）/`rkey`（远端钥匙）。

```
应用                    网卡 (HCA)
  |  post WQE 到 SQ        |
  |---------------------->| (敲 doorbell 寄存器, 不进内核)
  |                       |  硬件读 MR -> DMA -> 发线缆
  |                       |  ... 传输/重传/排序全硬件 ...
  |   poll CQ <-----------| 写 CQE 表示完成
  v                       v
```

QP 有几种**传输服务类型**，决定可靠性与是否面向连接：

| 类型 | 全称 | 可靠? | 连接? | 典型用途 |
|---|---|---|---|---|
| RC | Reliable Connection | 是 | 是 | 主力，支持 READ/WRITE/ATOMIC |
| UC | Unreliable Connection | 否 | 是 | 少用 |
| UD | Unreliable Datagram | 否 | 否 | 多播、管理、规模化场景 |
| RD | Reliable Datagram | 是 | 否 | 规范有、实现少 |

> 大规模训练里 NCCL 默认走 **RC**（可靠连接），用 `RDMA WRITE` 做点对点搬运。详见 [[ai-infra/网络/NCCL]]。

## 4. IB vs RoCE vs 传统以太网

三者的根本差异在"**谁来保证不丢包 + 谁来管拥塞**"。RDMA 硬件重传代价高（go-back-N 一退一大片），所以 RDMA **极度依赖"近乎无损"的网络**。

### 4.1 InfiniBand（专用网络）
- 链路层用**基于信用的流控（credit-based flow control）**：发送方只有在确认接收方有缓冲空间（有信用）时才发，**链路层天然不丢包**。
- 自带**子网管理器（Subnet Manager, SM）**：集中式地发现拓扑、算路由、配 LID（本地标识符）。一个 IB 子网像一台被统一编排的大交换机。
- 拥塞控制（CC）由硬件协议处理。

### 4.2 RoCE（在以太网上跑 RDMA）
- **RoCEv1**：RDMA 直接封在以太网帧里（Ethertype），二层、不可路由，几乎淘汰。
- **RoCEv2**：RDMA 封在 **UDP/IP** 里（UDP 目的端口 4791），**三层可路由**，是现在的主流。
- 以太网本身会丢包，所以要靠两套机制造无损：
  - **PFC（Priority Flow Control, 802.1Qbb）**：基于优先级的"暂停帧"，链路打满时让上游别发了——这是"链路级"防丢包。配错会引发**死锁/拥塞扩散（victim flow）**，是 RoCE 运维最大坑。
  - **ECN + DCQCN**：交换机在拥塞时给包打 ECN 标记，接收端回 CNP（拥塞通知），发送端降速——这是"端到端"拥塞控制。

```
                 谁保证无损 / 拥塞控制
 InfiniBand   |  链路: credit 流控(天生无损)  +  硬件 CC
 RoCEv2       |  链路: PFC(暂停帧)            +  端到端 ECN/DCQCN  ← 要精心调
 传统以太网    |  无! 丢了靠 TCP 重传(软件,慢)
```

| 维度 | InfiniBand | RoCEv2 | 传统以太网 + TCP |
|---|---|---|---|
| 传输 | RDMA(硬件) | RDMA(硬件) | TCP(软件) |
| 无损 | 链路天生(credit) | 靠 PFC 人工营造 | 无，丢了重传 |
| 可路由 | IB 路由(子网内 LID) | 是(UDP/IP) | 是 |
| 管理 | 集中式 SM | 标准以太网管理 | 标准 |
| 时延 | 最低(~1µs 级端到端) | 略高于 IB | 数十~上百 µs |
| 生态/成本 | 专用，常更贵 | 复用以太网，成本友好 | 最便宜 |
| 调优难度 | 相对省心 | 高(PFC/ECN 难调) | 低 |

> 直觉口诀：**IB 是"开箱无损"，RoCE 是"自己造无损"。** RoCE 省了专用设备的钱，花在了网络工程师的头发上。

## 5. 带宽与时延：从 SDR 到 NDR

InfiniBand 按**每条 lane 的信号速率**分代，链路通常是 **4 lane（4x）** 聚合。下表是常说的 4x 链路速率（保留原文档的分类口径并补全）：

| 代号 | 全称 | 4x 链路速率 | 编码 | 备注 |
|---|---|---|---|---|
| SDR | Single Data Rate | 8 Gb/s | 8b/10b | 每 10 bit 传 8 bit 有效 |
| DDR | Double Data Rate | 16 Gb/s | 8b/10b | |
| QDR | Quad Data Rate | 32 Gb/s | 8b/10b | |
| FDR | Fourteen Data Rate | 56 Gb/s | 64b/66b | 单 lane 14 Gb/s |
| EDR | Enhanced Data Rate | 100 Gb/s | 64b/66b | 单 lane 25 Gb/s |
| HDR | High Data Rate | 200 Gb/s | 64b/66b(+PAM4) | 单 lane 50 Gb/s |
| NDR | Next Data Rate | 400 Gb/s | PAM4 等 | 单 lane 100 Gb/s |
| XDR | (后续代) | 800 Gb/s | — | 演进中，以官方文档为准 |

**为什么编码会"吃掉"带宽——逐数手算**：
- **8b/10b**：每 10 个传输 bit 只有 8 bit 是数据 → 有效率 $8/10 = 80\%$。所以 SDR 信号率 10 Gb/s，有效数据率 $10 \times 0.8 = 8$ Gb/s（正是文档里的"8 Gb/s"）。
- **64b/66b**：每 66 bit 含 64 bit 数据 → 有效率 $64/66 \approx 96.97\%$，开销骤降。这就是 FDR 之后换编码的原因——8b/10b 那 20% 的损耗在高速率下太贵了。

**时延的量级直觉**：IB 端到端可做到 **~1 µs 级**（HCA 加交换机各贡献几百 ns）。对比 TCP/以太网常在 **几十到上百 µs**。一次 AllReduce 要跨很多步通信，时延被放大 $\log N$ 倍，差距会被指数级地暴露出来（见第 8 节）。

## 6. 胖树（Fat-Tree）拓扑：为什么要"无收敛"

大集群里几千张卡互联，不能用一台交换机（端口不够），得堆成多层。最常用的是 **Fat-Tree / Clos** 拓扑。核心思想：**越往上层，带宽越"胖"，保证任意两点间不被网络掐脖子。**

```
两层 Fat-Tree (无收敛, 简化示意)

      Spine0      Spine1      Spine2      Spine3     <- 脊交换机(核心)
       /|\         /|\         ...
      / | \       / | \
   ┌──┴─┴──┐  ┌──┴─┴──┐
   │ Leaf0 │  │ Leaf1 │   ...  Leaf k        <- 叶交换机(接入)
   └┬┬┬┬┬┬─┘  └───────┘
    GPU GPU ...                              <- 每个 leaf 接一批节点/GPU
```

关键术语与"为什么"：
- **收敛比（oversubscription）**：叶交换机"朝下接服务器的带宽" : "朝上接脊的带宽"。**1:1 = 无收敛（non-blocking）**，意味着哪怕全机所有节点同时满速互发，网络也不会成为瓶颈。AI 训练强烈追求接近 1:1。
- **为什么训练要无收敛**：AllReduce/AllToAll 这类集合通信会让**大量节点同时打满链路**。若上行收敛（比如 2:1），一半流量挤不上去，整轮通信被拖慢，所有 GPU 都得等——木桶效应。
- **路径多样性 + 自适应路由**：Fat-Tree 任意两叶之间有多条等价路径，配合**自适应路由（adaptive routing）**把流量摊开，避开热点链路。

**逐数手算（数交换机）**：设交换机有 $k$ 端口。两层无收敛 Fat-Tree 里，每个叶交换机一半端口朝下（接 $k/2$ 个节点）、一半朝上（连 $k/2$ 个脊）。
- 取 $k = 64$：每叶下接 $64/2 = 32$ 个节点。
- 若用 $L$ 个叶、$S$ 个脊，要无收敛则上下行端口对称，常见取 $S = k/2 = 32$ 个脊。
- 叶数受脊端口限制：每个脊有 64 口，可接 64 个叶 → 最多 $L = 64$ 个叶。
- 总节点数 $= L \times (k/2) = 64 \times 32 = 2048$ 个节点。

> 三层 Clos 可把规模继续放大到数万端口——代价是交换机/线缆数量与成本陡增，所以超大集群常用"轨道优化（rail-optimized）"等变体来省钱。

## 7. GPUDirect RDMA：连主机内存都跳过

普通 RDMA 已经绕过了 CPU 和内核，但数据如果在 **GPU 显存** 里（训练时梯度就在显存），还得先从显存拷到主机内存，再让网卡发——又多一趟。

**GPUDirect RDMA** 让 **网卡（HCA）直接对 GPU 显存做 DMA**，彻底省掉"显存↔主机内存"这一跳。

```
没有 GPUDirect RDMA:
  GPU 显存 --(拷贝1: GPU->Host)--> 主机内存 --(网卡DMA)--> 线缆
            ↑ 多一趟拷贝 + 占 PCIe + 占 CPU

有 GPUDirect RDMA:
  GPU 显存 ==(HCA 直接 DMA 读显存)==> 线缆
            ↑ 网卡和 GPU 走同一 PCIe Switch, 点对点直达
```

为什么能这么干（原理）：
- GPU 把一段显存"暴露"成可被其它 PCIe 设备访问的地址（BAR 空间），网卡拿到这个地址就能直接 DMA。
- **GPU 与网卡最好挂在同一个 PCIe Switch 下**（或同一 NUMA / NVLink 域），路径最短、不跨 CPU root complex。机房布线时讲究 GPU-NIC 亲和性就是这个道理。
- 配合 **GDRCopy、NCCL 的 net 插件** 等，把"显存里的张量直接发出去"做成常态。

> 收益：少一次拷贝 + 不占 CPU + 不占主机内存带宽 + 更低时延。对几百 GB/s 量级的梯度搬运，这一跳省下来非常可观。GPUDirect 在 NCCL 里如何被利用，见 [[ai-infra/网络/NCCL]]。

## 8. 为什么大规模训练离不开 IB/RDMA

数据并行每个 step 都要做一次梯度 **AllReduce**：把所有 GPU 的梯度求和再分发。这是**通信密集**操作，且**卡在关键路径上**（梯度没同步完，下一步不能更新权重）。

**Ring AllReduce 通信量手算**：$N$ 个 GPU、模型参数（梯度）共 $P$ 字节。Ring AllReduce 每张卡进/出网络的数据量约为：
$$
\text{每卡通信量} \approx 2 \times \frac{N-1}{N} \times P
$$
当 $N$ 很大时趋近 $2P$——**与卡数几乎无关**，这正是 ring 算法优雅之处。但"每卡 $2P$ 字节"必须在网络上跑完。

**代入数字**：取一个 70 亿参数模型，梯度按 fp16 = 2 字节/参数：
- $P = 7 \times 10^9 \times 2 = 1.4 \times 10^{10}$ 字节 $= 14$ GB。
- 每卡需搬运 $\approx 2P = 28$ GB。
- 用 **200 Gb/s（HDR）** 单口 = $25$ GB/s 有效带宽：
$$
t \approx \frac{28\ \text{GB}}{25\ \text{GB/s}} \approx 1.12\ \text{s}
$$
- 换成 **25 Gb/s 普通以太网** = $3.125$ GB/s：
$$
t \approx \frac{28}{3.125} \approx 8.96\ \text{s}
$$

**直觉冲击**：一个 step 假设计算只要 0.5 s，IB 下通信 1.1 s 已经让通信占比过半；而普通以太网 9 s 的通信会让 GPU **绝大部分时间在等网络**——几千万的算力空转。这就是"大规模训练为什么必须上 IB/RoCE + GPUDirect"的硬道理。

再叠加放大效应：
1. **时延放大**：AllReduce 有 $\sim 2(N-1)$ 步，每步都吃一次链路时延。1 µs 与 50 µs 的差距，乘以几百步、几千 step，累积成天级别的训练时间差。
2. **CPU 解放**：RDMA 让 CPU 不当搬运工，省下的 CPU 可以做数据预处理/调度。
3. **抖动敏感**：训练是 BSP 同步的，**最慢的那张卡决定整步速度**。丢一个包、触发一次软件重传，整步都被拖住——所以才如此追求"无损网络"。

## 数值示例 / 手算汇总

| 计算 | 公式 | 代入 | 结果 |
|---|---|---|---|
| 8b/10b 有效率 | $8/10$ | — | $80\%$ |
| 64b/66b 有效率 | $64/66$ | — | $\approx 97\%$ |
| SDR 有效数据率 | $10 \times 0.8$ | — | $8$ Gb/s |
| 200Gb/s 有效带宽 | $200/8$ | — | $25$ GB/s |
| 7B 模型梯度(fp16) | $7e9 \times 2$ | — | $14$ GB |
| Ring AllReduce 每卡量 | $2(N{-}1)/N \cdot P$ | $N$ 大 | $\approx 2P = 28$ GB |
| HDR 下 AllReduce 用时 | $28/25$ | — | $\approx 1.12$ s |
| 25GbE 下 AllReduce 用时 | $28/3.125$ | — | $\approx 8.96$ s |
| 2 层 Fat-Tree(k=64) 节点数 | $L\cdot k/2$ | $64\times32$ | $2048$ |

## 对照 / 复杂度表

| 概念 | 传统 TCP/IP | RDMA(IB/RoCE) |
|---|---|---|
| 内存拷贝次数 | 多次(用户↔内核) | 零拷贝 |
| 是否进内核(数据路径) | 每次收发都进 | 旁路，不进 |
| 谁做传输/重传 | CPU(软件) | 网卡(硬件) |
| 对端 CPU 是否参与 | 是 | 单边操作可完全不参与 |
| 典型端到端时延 | 数十~百 µs | ~1 µs 级 |
| 拥塞/无损 | TCP 软件控制 | IB credit / RoCE PFC+ECN |

## 常见问题

| 疑问 | 真相 |
|---|---|
| RoCE 是不是就是"慢一点的 IB"？ | 上层 RDMA 语义一样，差别在链路/网络层。RoCE 复用以太网、成本友好，但要靠 PFC/ECN 人工营造无损，调优难、易踩坑。 |
| RDMA 真的完全不用 CPU 吗？ | 数据路径不用。建连、注册内存、poll CQ 仍需少量 CPU；但相比传统收发，CPU 占用降一个数量级。 |
| 单边 READ/WRITE 不通知对端，对端怎么知道数据到了？ | 通常约定一个"完成标志位"或用 WRITE_WITH_IMM 携带立即数触发对端 CQE；纯单边时靠上层协议约定。 |
| 为什么 RDMA 这么怕丢包？ | 硬件重传机制（如 go-back-N）一旦丢包要回退重发一大片，性能断崖。所以宁可用 PFC 暂停也不丢包。 |
| 内存为什么要"注册/pin"？ | 网卡 DMA 用的是物理地址且不能让页被换出；注册会锁页并建立虚拟→物理映射 + 生成 rkey/lkey 钥匙。 |
| GPUDirect RDMA 一定更快吗？ | 当 GPU 与 NIC 在同一 PCIe Switch / 亲和拓扑下收益最大；跨 NUMA/跨 root complex 可能反而绕远，需关注 GPU-NIC 亲和。 |
| 训练里到底谁在用这些？ | NCCL 在底层选用 IB/RoCE Verbs + GPUDirect 做集合通信。见 [[ai-infra/网络/NCCL]] 与 [[ai-infra/网络/集合通信原语]]。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[ai-infra/网络/NCCL]] — 集合通信库，IB/RoCE/GPUDirect 的直接使用者
- [[ai-infra/网络/集合通信原语]] — AllReduce/AllGather/Broadcast 等原语，本文 AllReduce 手算的上层

---

### 参考
- 态路小课堂｜InfiniBand 网络相关内容简介：https://baijiahao.baidu.com/s?id=1760941961023057651&wfr=spider&for=pc
- InfiniBand Verbs 性能测试 perftest：https://github.com/linux-rdma/perftest
- 具体编码方式/速率代号以厂商与 IBTA 官方规范为准。
