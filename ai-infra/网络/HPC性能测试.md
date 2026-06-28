# HPC/网络性能测试

> 用「带宽 / 时延 / 消息率」三把尺子量出 RDMA 网络与集合通信的真实水位，从原始链路到 all-reduce 的瓶颈定位全链路方法论。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/网络/nccl-test-集合通讯的性能测试]] [[ai-infra/网络/InfiniBand]]

## 阅读地图

| 节 | 内容 | 你会得到什么 |
|----|------|------------|
| 0 | 一句话锚点 | 三个核心指标的直觉 |
| 1 | 地基：为什么要测、测什么 | 带宽/时延/消息率的定义与单位陷阱 |
| 2 | 测试金字塔(分层方法论) | 从链路层到框架层的分层定位 |
| 3 | perftest(ib_*_bw / ib_*_lat) | 裸 RDMA 点对点测试，参数逐个拆 |
| 4 | OSU Micro-Benchmarks | MPI 层 P2P 与集合通信测试 |
| 5 | all-reduce 带宽(algbw vs busbw) | 集合通信指标的正确解读 |
| 6 | 瓶颈定位决策树 | 看到数字后怎么往下查 |
| 7 | 数值例子 / 对照 / 实践 | 手算 + 典型公开数字 + 踩坑 |
| - | 常见问题 / 跳转 | 速查 |

## 0. 一句话锚点

性能测试就是回答三个问题：**搬一大块数据有多快(带宽)**、**搬一小块数据要等多久(时延)**、**一秒能搬多少个小块(消息率)**。AI 训练里真正卡你的，往往不是单链路带宽，而是**集合通信(all-reduce)在多机多卡叠加后的有效带宽**——所以必须分层往下测，逐层把"理论值 - 实测值"的缺口归因清楚。

## 1. 地基：为什么要测，测什么

### 1.1 为什么要测

买了 200Gb/s 的 InfiniBand 网卡，不代表训练就能跑满 200Gb/s。中间隔着：驱动、PCIe、CPU/GPU 内存拷贝、NUMA 亲和、交换机、拥塞控制、集合通信算法。**任何一环没配好，端到端就掉速**。性能测试的目的就是：

1. **验收**：硬件/驱动装好后，确认能到接近线速(line rate)。
2. **基线**：记录"健康水位"，后续异常时有对比基准。
3. **定位**：训练变慢时，自底向上排查到底是哪一层掉的。

### 1.2 三个核心指标(把单位讲透)

**(1) 带宽 Bandwidth** —— 单位时间搬运的数据量，适合**大消息**。

$$\text{带宽} = \frac{\text{传输的数据量}}{\text{传输耗时}}$$

单位陷阱(必须分清，否则差 8 倍)：
- **Gb/s**(小 b，bit)：网络厂商标称用这个。200Gb/s 网卡。
- **GB/s**(大 B，Byte)：程序里测吞吐常用这个。$1\,\text{GB/s} = 8\,\text{Gb/s}$。
- 还有 **GiB vs GB**：$1\,\text{GiB}=2^{30}$ 字节，$1\,\text{GB}=10^9$ 字节，差约 7%。
- `ib_*_bw` 的 `--report_gbits` 让它用 Gb/s 报告，方便对标网卡标称值。

**(2) 时延 Latency** —— 一个消息从发出到对端收到(或往返)的时间，适合**小消息**。

- 常报 **半往返(½ RTT)** 或 **单向延迟**，单位微秒 $\mu s$。
- RDMA 小包延迟可低至约 **1~3 $\mu s$**(以官方/实测为准)，远低于 TCP/IP(几十 $\mu s$)。
- 延迟决定了**小张量、小批次同步**的开销，对 MoE、流水线 bubble 很敏感。

**(3) 消息率 Message Rate** —— 每秒能完成的操作数，单位 **Mpps**(百万消息/秒)或 ops/s。

$$\text{消息率} = \frac{\text{完成的消息数}}{\text{耗时}}$$

- 小消息时，带宽很低但消息率是瓶颈(每个消息的固定开销主导)。
- 体现网卡/CPU 处理**很多小请求**的能力，对参数服务器、稀疏通信重要。

### 1.3 三者的关系：带宽-时延曲线

把"消息大小"从小到大扫一遍，画出带宽曲线，是性能测试最重要的一张图：

```
带宽(GB/s)
 ▲
线速 ┤                       ╭─────────────  ← 大消息：带宽受限(BW-bound)
     │                   ╭──╯
     │              ╭───╯      ← 拐点(N/2):达到半线速的消息大小
     │          ╭──╯              越靠左 = 小包效率越好
     │      ╭──╯
     │   ╭─╯
     │ ╭─╯   ← 小消息：时延/消息率受限(latency-bound)
     └─┴──┴───┴────┴──────┴──────────────▶ 消息大小(Byte, log)
      64B  1K   8K   64K   1M   16M

固定开销主导 ──────────►◄────────── 数据搬运主导
```

- **左半段**(小消息)：每条消息有固定开销(描述符、门铃 doorbell、网卡处理)，带宽爬不上去，此时看**时延**和**消息率**。
- **右半段**(大消息)：固定开销被摊薄，带宽趋近线速，此时看**峰值带宽**。
- **N/2 拐点**：达到"一半峰值带宽"所需的消息大小，越小说明小包处理越高效。

## 2. 测试金字塔(分层方法论)

性能问题的本质是"理论值 vs 实测值的缺口"。要定位缺口在哪层，必须**自底向上分层测**，每层都用上一层做"天花板"：

```
        ┌─────────────────────────────────────┐
  第4层  │  训练框架  (PyTorch DDP / FSDP / Megatron) │  端到端吞吐 tokens/s、step time
        │  ↑ 受集合通信 + 计算重叠 影响              │
        ├─────────────────────────────────────┤
  第3层  │  集合通信  (NCCL / nccl-tests)           │  busbw = all-reduce 有效带宽
        │  ↑ 受多链路拓扑 + 算法(ring/tree) 影响      │
        ├─────────────────────────────────────┤
  第2层  │  MPI 层    (OSU Micro-Benchmarks)        │  osu_bw / osu_latency / osu_allreduce
        │  ↑ 受 MPI 实现 + 进程映射 影响             │
        ├─────────────────────────────────────┤
  第1层  │  裸 RDMA   (perftest: ib_send_bw 等)     │  单 QP / 多 QP 点对点带宽与时延
        │  ↑ 受 网卡/驱动/PCIe/NUMA 影响            │
        ├─────────────────────────────────────┤
  第0层  │  链路/物理  (ibstatus, ibstat, ibdiagnet) │  链路状态、速率、误码、拓扑
        └─────────────────────────────────────┘

定位口诀：从下往上测，下层达标了再看上层；
         某层掉速 → 瓶颈就在「该层」或「该层到下层之间」。
```

**为什么这样分**：如果第 3 层 all-reduce 只有理论的 60%，你无法判断是 NCCL 算法问题还是底层链路问题——除非你已经知道第 1 层裸 RDMA 是满的。**先把地基测实，上层才有参照。**

## 3. perftest —— 裸 RDMA 点对点测试

### 3.1 是什么 / 解决什么

`perftest` 是 linux-rdma 社区的**裸 RDMA 微基准套件**，直接调用 verbs API(绕过 TCP/IP 内核栈)，测量**单对节点之间**的极限带宽与时延。它是测试金字塔的**第 1 层地基**：如果这里都跑不满，上层一定也跑不满。

工具家族(命名规律：`ib_{操作}_{指标}`)：

| 工具 | 操作类型 | 测什么 |
|------|---------|--------|
| `ib_send_bw` | SEND/RECV(双边) | 带宽 |
| `ib_send_lat` | SEND/RECV | 时延 |
| `ib_write_bw` | RDMA WRITE(单边) | 带宽(最常用) |
| `ib_write_lat` | RDMA WRITE | 时延 |
| `ib_read_bw` | RDMA READ(单边) | 带宽 |
| `ib_read_lat` | RDMA READ | 时延 |
| `ib_atomic_bw/lat` | 原子操作 | 带宽/时延 |

> **SEND vs WRITE 的区别(原子概念)**：`SEND` 是**双边**操作，对端要 post 一个 RECV 才能收，CPU 都参与；`WRITE` 是**单边**(one-sided) RDMA，发起方直接写到对端内存，对端 CPU 不感知。NCCL 等框架大量用单边的 WRITE/READ，所以验收带宽常用 `ib_write_bw`，但题目要求覆盖的 `ib_send_bw` 更能反映"含收发两端开销"的真实双边吞吐。

### 3.2 工作模型：服务端 + 客户端

perftest 是**两进程**模型，一端不带 IP(服务端，监听)，另一端带服务端 IP(客户端，连接)：

```
   节点 A (服务端)                      节点 B (客户端)
 ┌──────────────┐                    ┌──────────────┐
 │ ib_send_bw   │  ① 启动监听         │ ib_send_bw   │
 │   -d mlx5_0  │ ◄───────────────── │  -d mlx5_0   │
 │              │  ② TCP 交换 QP 信息 │  <A_的_IP>   │
 │   (无 IP)    │ ─────────────────► │ --report_gbits│
 │              │  ③ 走 RDMA 打流     │              │
 │  HCA ════════╪════════════════════╪════ HCA      │
 └──────────────┘   InfiniBand/RoCE  └──────────────┘
        先在 A 上启动，再在 B 上启动并指向 A
```

注意：**先启动服务端**(不带 IP 那一端)，否则客户端连不上。

### 3.3 关键参数的含义与权衡(讲含义，不背默认值)

| 参数 | 含义 | 调它影响什么 / 权衡 |
|------|------|--------------------|
| `-d <dev>` | 指定 HCA 设备(如 `mlx5_0`) | 多网卡机器必须选对，否则测到错网卡 |
| `-i <port>` | HCA 物理端口号 | 双口网卡选端口 |
| `-x <gid_index>` | GID 索引 | **RoCE 必填**；选 RoCEv2 对应的 GID。原文 `-x 3` 即此意 |
| `-s <size>` | 单消息大小(字节) | 大→测峰值带宽；小→测时延/消息率 |
| `-a` | 自动扫描所有消息大小 | 一次跑出完整带宽曲线(2B→8MB) |
| `-q <num>` | QP(队列对)数量 | 增加并发流，单 QP 跑不满线速时用多 QP 逼近线速 |
| `-n <iters>` | 迭代次数 | 越多越稳，太少噪声大 |
| `-D <sec>` | 按时长跑(秒) | 与 `-n` 二选一，定时打流 |
| `--report_gbits` | 用 Gb/s 报告 | 方便对标网卡标称(否则默认 MB/s) |
| `--run_infinitely` | 持续打流 | 配合监控看稳定带宽 |
| `-F` | 忽略 CPU 频率检查 | 跑容器/虚机里常加，避免误报 |
| `--use_cuda=<id>` | 用 GPU 显存做缓冲 | 测 **GPUDirect RDMA**(GDR)，反映训练真实路径 |
| `-R` | 用 RDMA-CM 建链 | 简化跨子网/RoCE 建链 |

**最重要的两条权衡直觉**：
- **单 QP 往往跑不满线速**：现代 200/400G 网卡的单 QP 受 PCIe、单核处理限制，需要 `-q 2/4/8` 多 QP 并发才逼近线速。**报"网卡能力"用多 QP，报"单流能力"用单 QP。**
- **`--use_cuda` 才是训练真实路径**：不带它测的是 CPU 内存到 CPU 内存；训练时数据在 GPU 显存，必须测 GDR 路径才有意义，否则会高估或低估。

### 3.4 典型命令(单机回环 / 多机)

```bash
# 0) 装工具 + 看链路状态(第 0 层先过)
apt install -y perftest infiniband-diags   # 或 yum install perftest infiniband-diags
ibstatus        # 看 state: ACTIVE, rate: 200 Gb/s
ibstat          # 更详细：链路层(IB/Ethernet)、GID

# 1) 单机回环自测(快速验证驱动/网卡，非真实链路带宽)
ib_send_bw -d mlx5_1 &                       # 服务端后台
ib_send_bw -d mlx5_1 127.0.0.1 --report_gbits  # 客户端

# 2) 多机带宽(两机须在同一 RDMA 子网/HPC 集群)
# A 机(服务端,RoCE 用 -x 选 GID):
ib_send_bw -d mlx5_1 -x 3 -a -F
# B 机(客户端,指向 A 的 RDMA 网卡 IP):
ib_send_bw -d mlx5_1 -x 3 -a -F <A_RDMA_IP> --report_gbits

# 3) 多 QP 逼近线速 + GDR 真实路径
ib_write_bw -d mlx5_1 -x 3 -q 4 --use_cuda=0 -F           # 服务端
ib_write_bw -d mlx5_1 -x 3 -q 4 --use_cuda=0 -F <A_IP> --report_gbits

# 4) 单独测时延(小包)
ib_send_lat -d mlx5_1 -x 3 -s 8 -F                        # 服务端
ib_send_lat -d mlx5_1 -x 3 -s 8 -F <A_IP>                 # 客户端
```

### 3.5 怎么读 perftest 输出

`ib_send_bw -a` 会按消息大小打印一张表，核心三列：

```
 #bytes  #iterations  BW peak[Gb/s]  BW average[Gb/s]  MsgRate[Mpps]
  2       1000          0.12            0.11               7.30      ← 小包:看 MsgRate
  64      1000          3.80            3.75               7.33
  8192    1000        180.5           179.2                2.73
  1048576 1000        197.4           197.1                0.024     ← 大包:看 BW,接近线速
  ...
```

- **BW average** 是你最该看的稳定带宽；BW peak 是瞬时峰值。
- 大包带宽接近线速(如 200G 网卡到约 197G，扣协议开销)→ 第 1 层健康。
- `ib_send_lat` 则报 `t_min / t_max / t_avg / t_99%` 等延迟分位数，关注 **t_avg 和 t_99%**(尾延迟)。

## 4. OSU Micro-Benchmarks(OMB)

### 4.1 是什么 / 解决什么

OSU Micro-Benchmarks(俄亥俄州立大学出品)是**MPI 层**(测试金字塔第 2 层)的标准微基准。perftest 测的是"两块网卡之间"的裸能力；OMB 测的是"经过 MPI 运行时之后"的能力，并且**首次引入集合通信测试**(allreduce/alltoall/bcast 等)。它是 HPC 圈验收互连的事实标准。

为什么需要它：训练框架(Megatron/DeepSpeed)和很多 HPC 应用是 MPI 启动的。MPI 实现(OpenMPI / MVAPICH2)、进程到核的映射、NUMA 亲和都会影响实测。**OMB 把"MPI 这一层的损耗"暴露出来。**

### 4.2 常用测试项

| 测试 | 类别 | 测什么 |
|------|------|--------|
| `osu_latency` | 点对点 | 两进程间往返延迟(随消息大小) |
| `osu_bw` | 点对点 | 两进程间单向带宽 |
| `osu_bibw` | 点对点 | 双向带宽 |
| `osu_mbw_mr` | 点对点 | 多对带宽 + **消息率**(messages/s) |
| `osu_allreduce` | **集合** | all-reduce 延迟(各 size) |
| `osu_alltoall` | 集合 | all-to-all 延迟(MoE/张量并行相关) |
| `osu_bcast` | 集合 | 广播延迟 |

### 4.3 运行模型 + 输出

OMB 用 `mpirun`/`mpiexec` 启动，进程数由集合通信测试决定：

```
   mpirun -np 16 -hostfile hosts ./osu_allreduce
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
  node0(rank0..7) node1(rank8..15) ...  ← 每 rank 一卡,跨节点走 RDMA
      └──── all-reduce 在所有 rank 间执行 ────┘

输出(osu_allreduce):  # Size  Avg Latency(us)
   4 → 2.81(小消息,延迟主导)   1048576 → 412.6(大消息,带宽主导)
```

```bash
# 点对点带宽/延迟(2 进程)
mpirun -np 2 -hostfile hosts ./osu_bw
mpirun -np 2 -hostfile hosts ./osu_latency

# 消息率(多对)
mpirun -np 2 -hostfile hosts ./osu_mbw_mr     # 关注 Messages/s 列

# 集合通信(全部 rank)
mpirun -np 16 -hostfile hosts ./osu_allreduce
```

**读法**：`osu_allreduce` 报的是**延迟(us)**而非带宽。要把它换算成带宽自己除：大消息时 $\text{有效带宽}\approx \frac{\text{消息大小}}{\text{延迟}}$，但集合通信的"有效带宽"概念在第 5 节用 busbw 才精确。OMB 主要用来对比"MPI 集合 vs NCCL 集合"以及不同算法/进程映射的差异。

## 5. all-reduce 带宽：algbw vs busbw(最容易读错)

### 5.1 为什么 all-reduce 的带宽要特殊定义

AI 训练里梯度同步几乎全靠 **all-reduce**。但"all-reduce 带宽"不能直接拿"数据量÷时间"草率算，因为 all-reduce 每个 rank 既发又收，且数据在环上转了多圈。这里有两个指标，**必须分清**(否则会误判网卡有没有跑满)：

**(1) algbw(算法带宽)** —— 用户视角，最朴素：

$$\text{algbw} = \frac{S}{t}$$

$S$ 是参与归约的数据大小，$t$ 是耗时。它**不随 GPU 数变化**地反映"我这块数据多久同步完"，但**不能直接对标网卡线速**。

**(2) busbw(总线带宽)** —— 硬件视角，可对标网卡：

$$\text{busbw} = \text{algbw} \times \frac{2(n-1)}{n}$$

其中 $n$ 是参与的 GPU 数。系数 $\frac{2(n-1)}{n}$ 来自 **ring all-reduce 算法**：每个 rank 实际要在网络上搬运约 $\frac{2(n-1)}{n}\cdot S$ 的数据(reduce-scatter + all-gather 两个阶段，每阶段 $n-1$ 步)。

### 5.2 ring all-reduce 数据流(理解系数从哪来)

```
4 个 GPU 的 ring all-reduce，数据被切成 4 块(chunk):

阶段一 Reduce-Scatter (n-1=3 步,每步每个 GPU 发 1 块):
   G0 ──► G1 ──► G2 ──► G3 ──► G0   (环形,逐块累加)
   3 步后:每个 GPU 各持有「一块的全局和」

阶段二 All-Gather (n-1=3 步,把各自的和传遍全环):
   G0 ──► G1 ──► G2 ──► G3 ──► G0
   3 步后:每个 GPU 都集齐全部 4 块的全局和

每个 GPU 网络上共搬运: (n-1)+(n-1) = 2(n-1) 块
每块大小 = S/n
∴ 单 GPU 网络流量 = 2(n-1)·(S/n) = S·2(n-1)/n
```

**关键直觉**：当 $n$ 很大时 $\frac{2(n-1)}{n}\to 2$，即 busbw ≈ 2×algbw。**busbw 才是该和网卡单向线速对比的量**——如果你的网卡 200Gb/s(=25GB/s)，busbw 能到约 20+ GB/s 就算健康；而 algbw 看起来只有一半，是正常的、不是掉速。

### 5.3 与 nccl-tests 的关系

第 3 层(集合通信)的标准工具是 **nccl-tests**(`all_reduce_perf`),它直接打印 algbw 和 busbw 两列,这正是 [[ai-infra/网络/nccl-test-集合通讯的性能测试]] 的主战场。perftest/OMB 是它的"下层地基对照"。

## 6. 瓶颈定位决策树

测出数字后，按下图自顶向下排查。**核心原则：每往上一层掉速，就回到下一层确认地基是否健康。**

```
训练慢(第4层:tokens/s 低)
       │
       ▼
通信占比高吗? (用 torch profiler / nsys 看 NCCL kernel 占时)
  ├─否─► 是计算/数据加载瓶颈,不是网络问题 → 查 GPU 利用率/dataloader
  └─是─►
       ▼
第3层 nccl-tests busbw 达标吗? (对标网卡线速的 ~80%+)
  ├─是─► 网络健康,是「通信/计算没重叠」或「并行策略」问题
  │       → 调 bucket size / 开 overlap / 换并行切分
  └─否─►
       ▼
第1层 perftest(多QP+GDR) 裸带宽达标吗?
  ├─否─► 地基就有问题,往下查物理层 ↓↓↓
  │        ├ ibstatus/ibstat: 链路 ACTIVE? 速率对吗? (掉到更低速率档?)
  │        ├ NUMA: 网卡和 GPU 在同一 NUMA? (numactl/nvidia-smi topo -m)
  │        ├ PCIe: 是不是 PCIe 跑到 x8/Gen3 没满? (lspci -vvv)
  │        ├ GDR: nvidia-peermem / dmabuf 装了吗? 没装则走 CPU 中转,慢
  │        └ MTU/PFC/ECN(RoCE): 拥塞控制没配 → 丢包重传,带宽塌方
  └─是─► 裸链路 OK,但 NCCL 不行 → NCCL 层配置问题
           ├ NCCL_IB_HCA / NCCL_SOCKET_IFNAME 选错网卡?
           ├ NCCL_IB_GID_INDEX(RoCE) 选错?
           ├ NCCL_ALGO / NCCL_PROTO 算法不优?
           ├ 拓扑没认对 → 看 NCCL_TOPO / 设 NCCL_DEBUG=INFO 看 Ring/Tree
           └ 跨机走了 TCP 没走 IB? (NCCL_DEBUG=INFO 看 [send] via 哪条路)
```

**最常见三大坑(经验)**：① 跨机偷偷走了 TCP socket 而不是 RDMA；② 网卡和 GPU 不在同一 NUMA 导致跨片拷贝；③ RoCE 没配好 PFC/ECN，一拥塞就丢包重传，带宽断崖。

## 7. 数值例子 / 对照 / 实践

### 7.1 手算：200Gb/s 网卡，all-reduce 同步 1.6GB 梯度要多久

设 8 卡($n=8$)、单机内 NVLink 不算、跨机网卡单向线速 200Gb/s = **25 GB/s**(理论)，实测裸 busbw 取约 **20 GB/s**(扣协议/效率约 80%)。

要同步的梯度 $S = 1.6\,\text{GB}$(约对应 7B 参数 fp16 的一部分量级，便于算)。

ring all-reduce 单 GPU 网络流量：

$$S \times \frac{2(n-1)}{n} = 1.6 \times \frac{2\times 7}{8} = 1.6 \times 1.75 = 2.8\,\text{GB}$$

耗时(用 busbw 即网络真实搬运速率)：

$$t = \frac{\text{流量}}{\text{busbw}} = \frac{2.8\,\text{GB}}{20\,\text{GB/s}} = 0.14\,\text{s} = 140\,\text{ms}$$

反推 algbw 验证：$\text{algbw} = S/t = 1.6/0.14 \approx 11.4\,\text{GB/s}$，而 $\text{busbw}=11.4\times\frac{2\times7}{8}=20\,\text{GB/s}$ ✓，自洽。

**结论直觉**：algbw(11.4) 看着只有线速(25)的不到一半，**但这是 ring 算法的固有特性，不是掉速**；要判断健康看 busbw(20≈线速 80%)。若 step 计算只要 100ms，那 140ms 的通信若不能与计算重叠，就会拖慢 1.4 倍——这正是要开 `overlap` / 梯度分桶的原因。

### 7.2 手算：消息率 vs 带宽，小包为什么慢

设单消息固定开销(网卡+CPU)约 $0.5\,\mu s$(以实测为准)，网卡 25GB/s。

- 发 **64B**：纯传输 $=64/25\text{e}9 \approx 0.003\,\mu s$，被 $0.5\,\mu s$ 固定开销淹没。实际带宽 $\approx 64\text{B}/0.5\mu s = 128\,\text{MB/s}$(线速的 0.5%！)，消息率 $\approx 1/0.5\mu s = 2\,\text{Mpps}$ 才是有意义指标。
- 发 **1MB**：传输 $=1\text{e}6/25\text{e}9 = 40\,\mu s \gg 0.5\mu s$，固定开销可忽略，带宽 ≈ 线速。

**这就是带宽曲线左低右高的根因**：小包看消息率/时延，大包看带宽。

### 7.3 工具对照表

| 工具 | 层级 | 测点 | 关键指标 | 何时用 |
|------|------|------|---------|--------|
| `ibstatus/ibstat` | L0 物理 | 链路 | state/rate/GID | 第一步验收 |
| `ibdiagnet` | L0 物理 | 全网拓扑 | 误码/拥塞/拓扑 | 整网体检 |
| `perftest`(ib_send_bw…) | L1 裸 RDMA | 点对点 | BW avg / lat / MsgRate | 验收单链路地基 |
| OSU OMB | L2 MPI | P2P+集合 | latency/bw/msgrate | MPI 应用对照 |
| **nccl-tests** | L3 集合 | all-reduce 等 | **algbw / busbw** | GPU 训练实测(主力) |
| torch profiler/nsys | L4 框架 | 端到端 | NCCL 占比/step time | 找通信-计算重叠问题 |

### 7.4 典型公开数字(约值，以官方/实测为准)

| 项 | 典型量级 |
|----|---------|
| InfiniBand HDR / NDR 单口线速 | 约 200 / 400 Gb/s |
| RDMA 小包单向延迟 / TCP 小包延迟 | 约 1~3 $\mu s$ / 约几十 $\mu s$(高一个数量级) |
| 单口 200G 实测裸大包带宽 | 约 180~197 Gb/s(扣协议) |
| 8 卡机内 NVLink 聚合 / "健康" busbw | 数百 GB/s 量级 / ≥ 线速 80% 即合格 |

### 7.5 实践要点(checklist)

1. **先过第 0 层**：`ibstatus` 看到 `ACTIVE` 且速率对再往上测。
2. **测训练真实路径**：perftest 加 `--use_cuda`，否则测的是 CPU 内存，不代表 GPU。
3. **单 QP 跑不满正常**：报网卡能力用 `-q 多QP`，报单流用单 QP，别混。
4. **RoCE 必填 GID** 并配好 PFC/ECN，否则一拥塞就丢包塌方。
5. **NUMA 亲和**：`nvidia-smi topo -m` 确认网卡和 GPU 同 NUMA/PCIe switch。
6. **busbw 才对标线速**，algbw 看起来低是 ring 固有，别误判；留好健康基线。
7. **跨机务必确认走 RDMA**：`NCCL_DEBUG=INFO` 看是否 `via NET/IB`，别偷走 TCP。

## 常见问题

| 问题 | 答案 |
|------|------|
| 带宽明明 200G 网卡，perftest 只到 100G？ | 多半是单 QP 限制，加 `-q 4`；或 PCIe 没跑满 Gen4 x16；或 NUMA 跨片。 |
| algbw 只有网卡线速一半，是掉速吗？ | 不是。看 busbw($\times\frac{2(n-1)}{n}$)，busbw 接近线速就健康。 |
| Gb/s 和 GB/s 老搞混 | 差 8 倍。网卡标称用 Gb/s(bit)，程序吞吐常用 GB/s(Byte)。 |
| 小消息带宽极低正常吗？ | 正常。小包看消息率(Mpps)和时延($\mu s$)，不看带宽。 |
| `ib_send_bw` vs `ib_write_bw` 选哪个？ | 验收单边路径(NCCL 类)用 write；要含双边收发开销用 send。 |
| RoCE 下带宽抖动、偶尔断崖 | 八成 PFC/ECN 拥塞控制没配好，丢包重传所致。 |
| 测出来 perftest 满但 NCCL 慢 | 地基 OK，查 NCCL 层：网卡选择、GID、算法、是否走了 TCP。 |
| 单机 `127.0.0.1` 回环带宽能代表真实吗？ | 不能。回环只验证驱动/网卡可用，真实带宽必须跨机测。 |

## 🔗 跳转链接

- [[00-知识地图]] —— 全局索引
- [[ai-infra/网络/nccl-test-集合通讯的性能测试]] —— 第 3 层集合通信实测(algbw/busbw 主战场)
- [[ai-infra/网络/InfiniBand]] —— 底层互连原理(RDMA/verbs/QP/GID)

参考资料：HPC 点对点 RDMA 测试 https://www.volcengine.com/docs/6419/164863 ｜ linux-rdma/perftest https://github.com/linux-rdma/perftest ｜ infiniband-diags https://github.com/linux-rdma/infiniband-diags ｜ OSU OMB https://mvapich.cse.ohio-state.edu/benchmarks/ ｜ RDMA 加速训练 https://www.volcengine.com/docs/6459/96563 ｜ 验证镜像 RDMA https://www.volcengine.com/docs/6459/119595
