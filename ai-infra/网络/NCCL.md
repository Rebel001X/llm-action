# NCCL 通信库

> NVIDIA Collective Communications Library：在多 GPU / 多机之间高效搬运梯度与张量的「集合通信」底座，是 PyTorch DDP、Megatron、DeepSpeed 等所有分布式训练框架的隐形发动机。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/网络/集合通信原语]] [[ai-infra/网络/InfiniBand]]

## 阅读地图

| 节 | 你会搞懂的问题 | 关键词 |
|----|----------------|--------|
| 0 | NCCL 到底是什么、一句话 | 集合通信库 |
| 1 | 为什么训练需要它（地基） | 数据并行 / AllReduce |
| 2 | 它提供哪些通信原语 | AllReduce/Broadcast/AllGather |
| 3 | AllReduce 怎么实现的（核心） | Ring / Tree |
| 4 | 它怎么"知道"硬件长什么样 | 拓扑感知 NVLink/PCIe/IB |
| 5 | comm / stream / 编程模型 | communicator / CUDA stream |
| 6 | 怎么塞进 PyTorch DDP | ProcessGroup / bucket |
| 7 | 调优要点与环境变量 | NCCL_* 调参 |
| 8 | 逐数手算：Ring 到底搬多少字节 | 带宽时间 |
| 9 | 对照表 / 复杂度 | Ring vs Tree |
| 10 | 常见问题 | 疑问→真相 |

## 0. 一句话锚点

**NCCL = 一组在 GPU 之间做「集合通信」的高度优化原语**。它不关心你训练什么模型，只负责一件事：当 N 个 GPU 上各自有一份数据、需要把它们「汇总 / 分发 / 重排」时，用最贴近硬件极限的方式把字节搬完。它对上层暴露 `ncclAllReduce` 之类的 C API，对下层直接驱动 NVLink / PCIe / InfiniBand。

> 历史上 NCCL 针对 NVIDIA 自家网络（NVLink、Spectrum-X 以太网、InfiniBand）做了深度优化；若用第三方网络（如博通 Tomahawk 5 以太网方案），客户需要自己有足够的工程能力为其适配并优化 NCCL，否则跑不满线速。

## 1. 地基：为什么训练一定要集合通信

### 1.1 数据并行的本质

数据并行（Data Parallel）下，**每个 GPU 持有一份完整的模型副本**，但只喂到一部分数据（一个 mini-batch 的 1/N）。每个 GPU 独立前向+反向，算出**自己这份数据对应的梯度** $g_i$。问题来了：要让 N 份模型保持一致，更新时必须用**全局平均梯度**：

$$\bar{g} = \frac{1}{N}\sum_{i=1}^{N} g_i$$

每个 GPU 都需要拿到这个 $\bar g$。这正是一个 **AllReduce**（先求和、再人人都拿到结果）操作。模型参数有几十亿个，梯度就是几 GB 的浮点数组——每一步训练都要在所有 GPU 之间把这几 GB 求和并广播回去。NCCL 就是干这个的。

```
  GPU0        GPU1        GPU2        GPU3
  g0          g1          g2          g3      ← 各算各的梯度
   \           |           |           /
    \__________AllReduce(SUM)_________/
   /           |           |           \
  ḡ           ḡ           ḡ           ḡ      ← 人人拿到相同的全局和
```

### 1.2 为什么不能用普通 socket / MPI 凑合

朴素做法（把数据拷到 CPU、走 TCP、CPU 求和、再拷回 GPU）有三宗罪：

1. **GPU→CPU→GPU 拷贝**白白浪费 PCIe 带宽，还卡住 CPU。
2. **不感知拓扑**：两个 GPU 明明插在同一块 NVLink 上（600+ GB/s），却走了 10 GB/s 的网卡。
3. **算法笨**：N 个 GPU 朴素两两通信是 $O(N^2)$ 流量。

NCCL 用 **GPUDirect**（GPU 显存直接 DMA，绕过 CPU）、**拓扑感知**、**Ring/Tree 算法**把这三点全解决。

## 2. NCCL 提供哪些通信原语

NCCL 实现了 MPI 风格的标准集合通信原语，全部直接在 GPU 显存上操作（详见 [[ai-infra/网络/集合通信原语]]）：

| 原语 | 语义 | 典型用途 |
|------|------|----------|
| **AllReduce** | 所有 rank 的输入按 op(SUM/MAX/...) 规约，结果人人一份 | 同步梯度（最常用） |
| **Broadcast** | 一个 root 的数据复制给所有 rank | 初始化时同步权重 |
| **Reduce** | 规约后只有 root 拿到结果 | 收集统计量 |
| **AllGather** | 每个 rank 一块数据，拼接后人人拿到全量 | ZeRO / 张量并行收集 |
| **ReduceScatter** | 规约后每个 rank 只拿结果的一段 | ZeRO 梯度分片 |
| **Send/Recv** (P2P) | 点对点收发 | 流水线并行传 activation |

一个关键恒等式（NCCL 内部就这么实现 AllReduce）：

$$\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$$

先让每个 GPU 把"一段"规约好（ReduceScatter），再把各段拼回完整结果（AllGather）。这就是 Ring AllReduce 分两个 phase 的根本原因。

```
ReduceScatter:                AllGather:
每人最后只拿到自己负责的       每人把自己那段广播给所有人
那一段的全局和                 最后拼成完整结果
[A][B][C][D]   →  [ΣA][·][·][·]   →   [ΣA][ΣB][ΣC][ΣD]
每人各一段                            人人完整
```

## 3. 核心算法：Ring vs Tree

### 3.1 Ring AllReduce（带宽最优）

把 N 个 GPU 排成一个**逻辑环**：0→1→2→…→N-1→0，每个 GPU 只和左右邻居通信。把梯度数组切成 N 块（chunk）。

**阶段一 ReduceScatter（N-1 步）**：第 k 步，GPU i 把"某一块"发给右邻居 i+1，同时从左邻居 i-1 收一块并就地累加。N-1 步后，每个 GPU 上恰好有**一块**是全局和。

```
4 个 GPU、梯度切 4 块 (a/b/c/d)。下面是 ReduceScatter 第 1 步：

 GPU0[a0 b0 c0 d0]   GPU1[a1 b1 c1 d1]   GPU2[...]   GPU3[...]
        └── 发 a0 ──►  收 a0 累加成 a0+a1
                              └── 发 b1 ──► ...      (环上同时进行)

每一步每条链路并行传 1 块；N-1 步后每人手里有一块"满和"。
```

**阶段二 AllGather（N-1 步）**：把那块"满和"沿环传一圈，让每个 GPU 都集齐 N 块。再走 N-1 步。

**为什么 Ring 带宽最优**：总共 $2(N-1)$ 步，每步每个 GPU 只发 $\frac{S}{N}$ 字节（S 是梯度总大小）。每个 GPU 总发送量：

$$2(N-1)\cdot\frac{S}{N} = 2S\cdot\frac{N-1}{N} \approx 2S$$

**和 N 无关！** 不管 8 卡还是 256 卡，每张卡搬的字节都约等于 $2S$，完美利用每条链路的带宽。代价是延迟随 N 线性增长（$2(N-1)$ 步）。

### 3.2 Tree AllReduce（延迟最优）

Ring 的软肋是**跨机大规模时延迟太高**（256 机要走 510 步）。Tree 把 rank 组成**二叉树**：

```
Reduce 阶段（叶→根，自底向上求和）：
            R0           ← 根：拿到全局和
          /    \
        R1      R2
       / \     / \
      R3 R4   R5 R6      ← 叶子先把数据往上汇

Broadcast 阶段（根→叶，自顶向下分发）把全局和发回每个节点。
```

树的高度是 $\log_2 N$，所以延迟只有 $O(\log N)$ 步，远小于 Ring 的 $O(N)$。**NCCL 实际用 "double binary tree"**：构造两棵互补的二叉树，让每个节点在一棵树是内部节点、在另一棵是叶子，这样上行/下行带宽都能用满，既保留 $\log N$ 延迟又不浪费带宽。

**选择规律**（NCCL 运行时自动决策，也可用 `NCCL_ALGO` 强制）：

| 场景 | 算法 | 原因 |
|------|------|------|
| 大消息、节点少 | **Ring** | 带宽最优，步数不怕 |
| 小消息、节点多 | **Tree** | 延迟 $\log N$ 取胜 |
| 单机 NVLink 全互联 | Ring/CollNet | 链路对称 |

## 4. 拓扑感知：NCCL 怎么"看见"硬件

这是 NCCL 比朴素实现快几十倍的关键。启动时 NCCL 会**扫描 PCIe / NVLink / 网卡拓扑**（读 `/sys`、NVML），构建一张机器内部的连接图，再据此决定环/树怎么排、用哪条链路。

### 4.1 带宽层级（差几个数量级）

```
速度从快到慢：
┌────────────────────────────────────────────────┐
│ NVLink / NVSwitch   ~300–900 GB/s   GPU↔GPU 同机 │ ★最快
│ PCIe Gen4 x16       ~32  GB/s       GPU↔CPU/网卡 │
│ InfiniBand HDR/NDR  ~25–50 GB/s     跨机          │
│ TCP/以太网 (socket) ~1–12 GB/s      兜底          │ ★最慢
└────────────────────────────────────────────────┘
```

NCCL 的核心策略：**机内走 NVLink，机间走 IB（配 GPUDirect RDMA）**，绝不让快链路上的数据降级到慢链路。

### 4.2 一个典型 8 卡机的拓扑环

```
        NVSwitch（全互联，每对 GPU 都有高带宽）
   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  GPU0   GPU1   GPU2   GPU3   GPU4   GPU5   GPU6   GPU7
   │                                                 │
   └────────── NCCL 排成逻辑 Ring ───────────────────┘
   0→1→2→3→4→5→6→7→0   （都在 NVLink 上，环极快）

机间：每台机的 GPU 通过本地 NIC（IB 网卡）+ GPUDirect RDMA
     直接把显存搬到对端显存，CPU 不参与拷贝。
```

### 4.3 GPUDirect RDMA：绕过 CPU

普通路径：GPU 显存 →(PCIe)→ CPU 内存 →(PCIe)→ 网卡。两次拷贝、CPU 全程参与。
GPUDirect RDMA 路径：**网卡直接 DMA 读 GPU 显存**，一步到位，CPU 只下发指令。前提是 GPU 和 NIC 挂在同一 PCIe Switch / 同一 NUMA 下，NCCL 拓扑扫描就是为了找到这种"近邻"配对。InfiniBand 细节见 [[ai-infra/网络/InfiniBand]]。

## 5. 编程模型：communicator、rank、stream

### 5.1 三个核心概念

- **communicator (comm)**：一个通信域。每个参与的 GPU 在域里有一个 `ncclComm_t` 句柄。所有集合通信调用都针对某个 comm。
- **rank**：GPU 在 comm 里的编号 0..N-1（root、环的位置都用 rank 表示）。
- **stream**：NCCL 操作是**异步**的，挂在一条 CUDA stream 上排队执行，和计算 kernel 共用 GPU 的执行流，从而**通信与计算可以重叠**。

```
建域流程（每个进程对应一块 GPU 的典型写法）：

  ncclGetUniqueId(&id)            ← rank0 生成唯一 ID
        │  （通过 TCP/文件/MPI 把 id 广播给所有进程）
        ▼
  ncclCommInitRank(&comm, N, id, myRank)   ← 每个 rank 建 comm
        │
        ▼
  ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream)
        │   ← 入队到 stream，立即返回（异步）
        ▼
  cudaStreamSynchronize(stream)   ← 需要结果时再等
```

### 5.2 异步 + 重叠是性能关键

因为 AllReduce 挂在 stream 上异步执行，框架可以**一边算反向、一边把已算好的梯度提前 AllReduce**：

```
时间 →
计算流: [反向层L]──[反向层L-1]──[反向层L-2]── ...
通信流:           [AllReduce层L]──[AllReduce层L-1]── ...
                  ↑ 层L梯度一就绪就开始传，和后续计算重叠
```

这就是 DDP "bucket + 反向钩子" 的物理基础。

### 5.3 Group 调用合并小操作

很多小 AllReduce 会被启动开销吃掉性能。`ncclGroupStart()/ncclGroupEnd()` 把多个调用合并成一次内核启动，减少 launch 开销——DDP 的 bucket 就是这个思想的上层封装。

## 6. 与 PyTorch DDP 集成

### 6.1 调用栈

```
你的代码:  loss.backward()
              │  （反向算梯度，触发每个参数的 hook）
              ▼
DDP:       梯度装入 bucket，bucket 满了就触发
              │
              ▼
ProcessGroup (backend="nccl")
              │
              ▼
NCCL:      ncclAllReduce(bucket, ..., ncclSum, comm, stream)
              │
              ▼
硬件:      NVLink / InfiniBand 实际搬字节
```

`torch.distributed.init_process_group(backend="nccl")` 时，PyTorch 用 NCCL 建好 comm；之后所有 `all_reduce` / `all_gather` 都落到 NCCL。

### 6.2 DDP 的两个关键优化（都依赖 NCCL 特性）

1. **Gradient Bucketing**：把许多小梯度张量打包成大桶（如 25MB）再一次 AllReduce。回顾第 8 节手算——小消息时延迟项占主导，合并成大消息才能跑满带宽。
2. **Computation–Communication Overlap**：反向是从输出层往输入层算的，先算完的层（靠近输出）的梯度可以**立刻**开始 AllReduce，与还在进行的前面层反向重叠。对应第 5.2 节的双流图。

> 经验：单机用 `DistributedDataParallel`（每进程一卡 + NCCL）几乎总是优于 `DataParallel`（单进程多线程，走不了 NCCL 的多进程通路）。

## 7. 调优要点与环境变量

NCCL 行为几乎全靠环境变量调（**以官方文档为准**，参数会随版本变化）：
官方环境变量文档：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

| 环境变量 | 作用 | 调优含义 |
|----------|------|----------|
| `NCCL_DEBUG=INFO`/`WARN` | 打印日志 | 排障第一步，看它选了哪条链路/算法 |
| `NCCL_SOCKET_IFNAME` | 指定走哪块网卡（如 `=ens1f0`） | 多网卡机器避免走错慢网卡 |
| `NCCL_IB_HCA` | 指定 IB 设备/端口 | 多 IB 卡时绑定正确 HCA |
| `NCCL_P2P_DISABLE` | 关 GPU 间 P2P | 仅排障，正常别关 |
| `NCCL_ALGO` | 强制 Ring/Tree | 自动决策不理想时手动指定 |
| `NCCL_PROTO` | LL/LL128/Simple 协议 | 小消息延迟 vs 大消息带宽权衡 |
| `NCCL_NTHREADS` / `NCCL_BUFFSIZE` | 线程数/缓冲区 | 微调吞吐 |

排障与安装速查：

```bash
# 看 NCCL 库是否在链接路径里
ldconfig -p | grep libnccl

# 补链接路径（找不到 libnccl 时）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# 安装（不同发行版包名不同，以官方为准）
yum install libnccl libnccl-devel libnccl-static

# 打开日志看实际选路
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens1f0
```

**调优心法（顺序）**：先 `NCCL_DEBUG=INFO` 确认它走的是 NVLink/IB 而不是悄悄降级到 socket；再确认网卡/HCA 绑对（`NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`）；再看消息大小是否够大（小 → 加大 bucket）；最后才考虑手动 `NCCL_ALGO`/`NCCL_PROTO`。

## 8. 数值示例 / 逐数手算

### 8.1 Ring AllReduce 到底搬多少字节、要多久

设：**N = 8** 卡，模型 **10 亿参数**，梯度用 **FP32（4 字节）**。

- 梯度总大小 $S = 10^9 \times 4 = 4\times10^9$ 字节 $= 4\,\text{GB}$。
- Ring 把 S 切成 N=8 块，每块 $S/N = 4\text{GB}/8 = 0.5\,\text{GB}$。
- 步数：ReduceScatter $N-1=7$ 步 + AllGather $7$ 步 $= 14$ 步。
- 每张卡总发送量：$2(N-1)\cdot S/N = 2\times7\times0.5\text{GB} = 7\,\text{GB}$。

验证近似公式 $2S\cdot\frac{N-1}{N} = 2\times4\times\frac{7}{8} = 7\,\text{GB}$ ✓。

**时间估算**：若机内 NVLink 单向有效带宽 $B = 250\,\text{GB/s}$：

$$t = \frac{\text{每卡发送量}}{B} = \frac{7\,\text{GB}}{250\,\text{GB/s}} = 0.028\,\text{s} = 28\,\text{ms}$$

每步还有固定延迟 $\alpha$（设 $1\,\mu s$），14 步只多 $14\,\mu s$，可忽略 → 大消息下**带宽主导**，Ring 合适。

### 8.2 为什么小消息要用 Tree（延迟主导）

把同样 8 卡换成传一个 **极小张量**（如 256 字节）：

- 传输时间 $\approx 256/(250\times10^9) \approx 1\,\text{ns}$，**忽略不计**。
- Ring 延迟项：$2(N-1)\times\alpha = 14\times1\mu s = 14\,\mu s$。
- Tree 延迟项：$2\log_2 8\times\alpha = 2\times3\times1\mu s = 6\,\mu s$。

此时**延迟主导**，Tree 的 6μs < Ring 的 14μs，Tree 胜。这就是 NCCL 按消息大小自动切换算法的量化依据。

### 8.3 DDP bucket 为什么有用

假设一层只有 256 字节梯度、有 1000 层。逐层 AllReduce：$1000\times14\mu s = 14\,\text{ms}$ 纯延迟。
打成一个 bucket（256KB）一次 AllReduce：传输 $\approx 256\text{KB}/250\text{GB/s}\approx1\mu s$ + 14μs 延迟 $\approx 15\,\mu s$。**快了约 900 倍**。这就是 bucketing 的威力。

## 9. 对照表 / 复杂度

| 维度 | Ring AllReduce | Tree(double-tree) AllReduce |
|------|----------------|------------------------------|
| 步数（延迟） | $2(N-1)$，$O(N)$ | $\sim 2\log_2 N$，$O(\log N)$ |
| 每卡发送量（带宽） | $\approx 2S$，与 N 无关 | $\approx 2S$（double-tree 也满带宽） |
| 适合 | 大消息、节点少 | 小消息、节点多 |
| 软肋 | 大规模时延迟高 | 实现更复杂 |

| 链路 | 量级带宽 | NCCL 何时用 |
|------|----------|-------------|
| NVLink/NVSwitch | 300–900 GB/s | 机内 GPU 互联，首选 |
| PCIe Gen4 x16 | ~32 GB/s | 无 NVLink 时机内兜底 |
| InfiniBand NDR | 25–50 GB/s | 机间，配 GPUDirect RDMA |
| 以太网 socket | 1–12 GB/s | 最后兜底，性能差 |

## 10. 常见问题

| 疑问 | 真相 |
|------|------|
| NCCL 是网络协议吗？ | 不是。它是**通信库/算法层**，底层仍走 NVLink/PCIe/IB/TCP；它决定"怎么搬、走哪条路"。 |
| Ring 比 Tree 永远好/差？ | 都不是。**大消息 Ring（带宽优），小消息/大集群 Tree（延迟优）**；NCCL 自动选。 |
| 训练突然很慢怎么排查？ | 先 `NCCL_DEBUG=INFO` 看它是不是悄悄降级到了 socket（没走 NVLink/IB），再查网卡绑定。 |
| 单机为啥 DDP 比 DataParallel 快？ | DDP 多进程 + NCCL 多卡通路，能重叠通信计算；DataParallel 单进程有 GIL/拷贝瓶颈。 |
| 为什么 DDP 要把梯度打包成 bucket？ | 小消息延迟主导（见 8.3），合并成大消息才能跑满带宽，减少启动开销。 |
| 非 NVIDIA 网络能用 NCCL 吗？ | 能跑，但 NCCL 主要为 NVIDIA 网络（NVLink/Spectrum-X/IB）优化；第三方以太网（如博通 Tomahawk 5）需自己投入工程做适配优化，否则吃不满线速。 |
| AllReduce 和 ReduceScatter+AllGather 啥关系？ | 恒等：$\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}$，Ring 就是这么两阶段实现的。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局总览，从这里找其它主题
- [[ai-infra/网络/集合通信原语]] — AllReduce/AllGather/ReduceScatter 等原语语义详解
- [[ai-infra/网络/InfiniBand]] — 机间高速网络与 GPUDirect RDMA 的底层支撑
