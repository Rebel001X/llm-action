# NCCL-Tests

> NCCL-Tests 是 NVIDIA 官方的集合通信基准测试套件：跑一遍就知道你的 GPU 互联（NVLink / PCIe / IB / RoCE）到底跑到了多少 GB/s，是不是被打满，瓶颈在哪。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]] [[ai-infra/网络/HPC性能测试]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点：为什么要测 | 带宽体检 |
| 1 | 地基：NCCL / 集合通信 / 总线带宽 | NCCL、AllReduce |
| 2 | 工具是什么、怎么编译 | make、MPI=1 |
| 3 | `all_reduce_perf` 等 7 个测试程序 | 程序清单 |
| 4 | 命令行参数逐个拆 | -b -e -f -g -n -w |
| 5 | 读懂输出：time / algbw / **busbw** | 两个带宽 |
| 6 | **algbw vs busbw** 公式推导（核心） | 系数 2(n-1)/n |
| 7 | 如何判断 NVLink / IB 跑满了 | 理论上限对照 |
| 8 | 多机测试：MPI + 网络环境变量 | NCCL_IB_HCA |
| 9 | 调优与诊断 checklist | NCCL_DEBUG |
| 数值 | AllReduce 手算 + 上限对照表 | 手算 |
| FAQ | 常见坑 | 排错 |

---

## 0. 一句话锚点

训练大模型时，多卡之间要不停地**同步梯度**（AllReduce）、**切分参数**（ReduceScatter / AllGather）。这些通信往往占总训练时间的 20%~50%。**NCCL-Tests 就是在不跑真模型的前提下，纯压通信链路**，量出"我的硬件互联实际能到多少带宽"，从而判断：

- 互联是否健康（NVLink 8 条全在？IB 卡都识别了？）
- 是否被打满（实测 busbw 接近理论上限 = 好）
- 多机比单机慢多少（跨节点 IB 是不是瓶颈）

> 它是 AI-Infra 工程师的"互联体检仪"。装机验收、故障定位、调优前后对比，都靠它。

---

## 1. 地基 / 前置

### 1.1 什么是 NCCL

**NCCL**（NVIDIA Collective Communications Library，读作 "Nickel"）是 NVIDIA 的多 GPU / 多节点集合通信库。它自动探测拓扑（NVLink、NVSwitch、PCIe、IB），构建**通信环（Ring）或树（Tree）**，让 N 张 GPU 高效地做集合操作。PyTorch DDP / FSDP、Megatron、DeepSpeed 底层全是它。详见 [[ai-infra/网络/NCCL]]。

### 1.2 什么是"集合通信原语"

不是点对点（A 发给 B），而是**一群进程一起参与**的通信模式。详见 [[ai-infra/网络/集合通信原语]]。最常用 5 个：

```
   AllReduce          ReduceScatter         AllGather
  各卡有一份 x_i      各卡有全量、求和后    各卡有一片，
  求和并广播回所有卡   每卡只留自己那片      拼成全量给所有卡
  (DDP 梯度同步)      (FSDP/ZeRO 后半)      (FSDP/ZeRO 前半)

   Broadcast                Reduce
  1 张卡 → 所有卡           所有卡 → 1 张卡求和
```

### 1.3 三个带宽概念（先建立直觉，第 6 节给公式）

- **链路带宽**：硬件物理上限。如单向 NVLink 4.0 一条 ≈ 25 GB/s，一张 H100 有 18 条。
- **algbw（算法带宽）**：`数据量 ÷ 耗时`。**用户视角**——传 N 字节用了多久。
- **busbw（总线带宽）**：algbw × **算法相关系数**，还原成**每条链路实际承载的速率**。**硬件视角**——能直接对照链路上限。

> 一句话：**algbw 是"我感觉传了多快"，busbw 是"线缆上真实跑了多快"。判断硬件是否打满，看 busbw。**

---

## 2. NCCL-Tests 是什么 / 如何编译

仓库：`https://github.com/NVIDIA/nccl-tests`。它是一组独立的可执行程序，每个对应一种集合操作，内部循环调用 NCCL 并精确计时。

### 2.1 单机编译

只需 `make`。若 CUDA 不在 `/usr/local/cuda`、NCCL 不在 `/usr`，用 `CUDA_HOME` / `NCCL_HOME` 指定：

```bash
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
# 产物在 ./build/  例如 ./build/all_reduce_perf
```

### 2.2 多机编译（带 MPI）

NCCL-Tests 依赖 **MPI** 来跨多个进程、从而跨多个节点。要编出 MPI 版本，设 `MPI=1` 并用 `MPI_HOME` 指向 MPI 安装路径：

```bash
$ make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

> MPI 在这里**只负责"拉起多进程 + 交换 NCCL 的连接信息（NCCL ID）"**，真正的数据传输还是 NCCL 走 NVLink/IB。MPI 不参与高速数据面。

```
            编译产物结构
  nccl-tests/
   └─ build/
       ├─ all_reduce_perf      ← 最常用
       ├─ all_gather_perf
       ├─ reduce_scatter_perf
       ├─ broadcast_perf
       ├─ reduce_perf
       ├─ alltoall_perf
       └─ sendrecv_perf
```

---

## 3. 七个测试程序（程序清单）

每个程序测一种原语。`all_reduce_perf` 是体检首选（DDP 训练就是它）。

| 程序 | 原语 | 典型场景 | busbw 系数（见第6节） |
|------|------|---------|---------------------|
| `all_reduce_perf` | AllReduce | DDP 梯度同步 | $2(n-1)/n$ |
| `all_gather_perf` | AllGather | FSDP 收集参数 | $(n-1)/n$ |
| `reduce_scatter_perf` | ReduceScatter | FSDP 切分梯度 | $(n-1)/n$ |
| `broadcast_perf` | Broadcast | 广播初始权重 | $1$ |
| `reduce_perf` | Reduce | 汇总到一卡 | $1$ |
| `alltoall_perf` | AllToAll | MoE 专家路由 | $(n-1)/n$ |
| `sendrecv_perf` | 点对点 | P2P 链路体检 | $1$ |

> 这里的 $n$ 是参与通信的 **GPU/rank 总数**，不是节点数。

---

## 4. 命令行参数逐个拆

```bash
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
#                          ^起   ^止    ^倍  ^GPU数
# 含义：在 8 张 GPU 上，消息大小从 8 字节 扫到 128 MB，每步 ×2
```

| 参数 | 含义 | 为什么重要 / 权衡 |
|------|------|------------------|
| `-b <size>` | 起始消息大小（begin），如 `8`、`1K` | 小消息测**延迟**（latency-bound） |
| `-e <size>` | 结束消息大小（end），如 `128M`、`8G` | 大消息测**带宽**（bandwidth-bound），busbw 在这里趋于上限 |
| `-f <factor>` | 每步乘的倍数，`-f 2` 即翻倍扫描 | 控制采样密度；2 是常用值 |
| `-g <ngpus>` | **单进程**内用几张 GPU | 单机单进程多卡时用；与 MPI 多进程模式二选一 |
| `-n <iters>` | 每个 size 计时迭代次数 | 越大越稳，去抖动 |
| `-w <iters>` | warmup 次数（不计时的预热） | 排除首次建链 / JIT 的开销，必须有 |
| `-d <type>` | 数据类型 float/half/... | 影响每元素字节数 |
| `-o <op>` | 归约操作 sum/prod/max/... | AllReduce 默认 sum |
| `-c <0/1>` | 是否校验结果正确性 | 验收开 1，纯测带宽关 0（省时间）|
| `-z <0/1>` | 是否用 blocking 模式 | 一般默认即可 |

> ⚠️ `-g N`（单进程 N 卡）和 `mpirun -np N ... -g 1`（N 进程各 1 卡）是**两种组织方式**。多机时**必须用 MPI 多进程**，每进程通常 `-g 1`，由 `mpirun -np` 控制总 rank 数。

---

## 5. 读懂输出：time / algbw / busbw

典型输出（节选，数字示意）：

```
#                                                     out-of-place                       in-place
#       size    count  type  redop    time   algbw   busbw   #wrong   time   algbw   busbw  #wrong
#        (B)  (elem)                  (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
     1048576   262144 float    sum   45.3   23.15   40.51       0    44.8   23.40   40.96      0
   134217728 33554432 float    sum 1380.2   97.24  170.18       0  1378.5   97.36  170.39      0
# Avg bus bandwidth : 165.42
```

逐列含义：

```
 size   = 消息字节数（本行测多大）
 count  = 元素个数 = size / sizeof(type)
 time   = 这一步集合操作的耗时
 algbw  = size / time              ← 算法带宽（用户视角）
 busbw  = algbw × 系数             ← 总线带宽（硬件视角，拿来比上限）
 #wrong = 校验错误数（应为 0）
 in/out-of-place = 输入输出是否同一块 buffer
 末行 Avg bus bandwidth = 各 size 的 busbw 平均，常作单一汇总指标
```

**关键认知：判断硬件是否打满，永远看最大消息那几行的 busbw，并与 Avg bus bandwidth 综合看。小消息 busbw 低是正常的（被延迟主导，没到带宽区）。**

```
 busbw 随 size 的典型曲线
  busbw │                ________________  ← 趋近硬件上限（饱和区，看这里）
 (GB/s) │             ／
        │          ／
        │      ／              ← 爬升区
        │ ／  ← 小消息：延迟主导，busbw 很低（正常）
        └──────────────────────────────► size
        8B    1KB   1MB   16MB   128MB
```

---

## 6. algbw vs busbw —— 公式推导（本文最核心一节）

这是面试与实战最常考的点。先记结论，再推导。

### 6.1 algbw 定义

$$
\text{algbw} = \frac{S}{t}
$$

其中 $S$ = 消息字节数（`-b/-e` 指定的那个 size），$t$ = 该集合操作耗时。**它只关心"用户提交了 S 字节、花了 t 秒"**，不管底层为完成这件事在链路上实际搬了多少。

### 6.2 为什么需要 busbw

问题：不同算法为传同样的 $S$，**链路上真实搬运的数据量不一样**。比如 AllReduce 比 Broadcast 累得多。如果都用 algbw 比较，会冤枉 AllReduce。于是定义：

$$
\text{busbw} = \text{algbw} \times C
\quad\text{其中 } C \text{ 是算法相关系数}
$$

$C$ 的设计目标是：**让 busbw 直接等于"每条链路上数据的实际速率"，从而可以跨算法横向比较、也可直接对照硬件链路上限。**

### 6.3 推导 AllReduce 的系数 $C = \dfrac{2(n-1)}{n}$

NCCL 的 AllReduce 用 **Ring 算法**：$n$ 个 GPU 排成环，分两个阶段。

```
       Ring AllReduce（n=4 示意）
   GPU0 ──► GPU1 ──► GPU2 ──► GPU3 ──┐
     ▲                               │
     └───────────────────────────────┘

  阶段一 ReduceScatter：n-1 步，每步每条链路传 S/n
  阶段二 AllGather     ：n-1 步，每步每条链路传 S/n
```

把数据切成 $n$ 片，每片 $S/n$ 字节：

- **ReduceScatter 阶段**：需要 $n-1$ 步，每步每条链路发送一片（$S/n$）。该阶段每条链路总传输 $= (n-1)\cdot \dfrac{S}{n}$。
- **AllGather 阶段**：同样 $n-1$ 步、每步 $S/n$。该阶段每条链路总传输 $= (n-1)\cdot \dfrac{S}{n}$。

两阶段相加，**每条链路实际传输的数据量**：

$$
S_{\text{bus}} = 2\,(n-1)\cdot \frac{S}{n} = \frac{2(n-1)}{n}\,S
$$

而这一切是在时间 $t$ 内完成的，所以每条链路的真实速率：

$$
\text{busbw} = \frac{S_{\text{bus}}}{t} = \frac{S}{t}\cdot\frac{2(n-1)}{n} = \text{algbw}\cdot\frac{2(n-1)}{n}
$$

得证 $C_{\text{AllReduce}} = \dfrac{2(n-1)}{n}$。

### 6.4 各算法系数一览（同理可推）

| 算法 | 系数 $C$ | $n\to\infty$ 时 | 直觉 |
|------|---------|----------------|------|
| AllReduce | $\dfrac{2(n-1)}{n}$ | $\to 2$ | ReduceScatter + AllGather，干两遍活 |
| ReduceScatter | $\dfrac{n-1}{n}$ | $\to 1$ | 只干一遍 |
| AllGather | $\dfrac{n-1}{n}$ | $\to 1$ | 只干一遍 |
| Broadcast / Reduce | $1$ | $\to 1$ | 数据沿环走一圈 |
| AllToAll | $\dfrac{n-1}{n}$ | $\to 1$ | 每卡给其余 n-1 卡各发一份 |

> **要点**：AllReduce 的 busbw 当 $n$ 较大时约等于 algbw 的 2 倍。所以你看到 algbw=97、busbw=170，并不矛盾——这是同一根线缆，只是 AllReduce 在它上面来回搬了约 2 份数据。

---

## 7. 如何判断 NVLink / IB 是否跑满

### 7.1 方法论三步

```
 ① 跑大消息（-e 几百 MB~几 GB），取 busbw 饱和值
 ② 查你的硬件链路理论单向带宽上限 B_link
 ③ 比较：busbw / B_link
       > 80%   健康、基本打满
       50~80%  偏低，查拓扑/参数/PCIe 降级
       < 50%   有问题，必排查（见第9节）
```

### 7.2 单机：判断 NVLine/NVSwitch 是否跑满

单机 8 卡若全走 NVLink/NVSwitch，busbw 应非常高（百 GB/s 级）。典型公开数字（**约值，以官方为准**）：

| 互联 | 单向单链路 ≈ | 单 GPU 聚合 ≈ | 8 卡 AllReduce busbw 期望 ≈ |
|------|-------------|--------------|---------------------------|
| PCIe 4.0 x16 | 32 GB/s | 32 GB/s | 几十 GB/s（明显受限）|
| NVLink 3.0（A100）| 25 GB/s/条 ×12 | ≈600 GB/s 双向 | 150~250 GB/s 量级 |
| NVLink 4.0（H100）| 25 GB/s/条 ×18 | ≈900 GB/s 双向 | 300~480 GB/s 量级 |

**判读**：8 卡 AllReduce busbw 只有几十 GB/s，远低于 NVLink 预期 → 八成走了 PCIe（NVLink 没用上 / NVSwitch 没识别），用 `nvidia-smi topo -m` 看拓扑矩阵确认 GPU 间是 `NV#`（NVLink）还是 `PIX/PHB`（PCIe）。

### 7.3 多机：判断 IB / RoCE 是否跑满

跨节点时 busbw 上限由**网卡数 × 单卡带宽**决定。典型（**约值，以官方为准**）：

| 网卡 | 单口理论 ≈ | 说明 |
|------|-----------|------|
| 100G IB/RoCE | 12.5 GB/s | EDR/部分 HDR |
| 200G IB（HDR）| 25 GB/s | |
| 400G IB（NDR）| 50 GB/s | H100 集群常见，每 GPU 配一张 |

```
 单机 vs 多机 busbw 对比（典型形态）
 busbw │ ███████████████  单机 8 卡（NVLink）  高
 (GB/s)│ ████              2 机 16 卡（跨 IB）  明显下降
       └────────────────────────────────────
       下降幅度 ≈ NVLink带宽 / IB聚合带宽
```

**判读**：若你有 8×400G NDR 网卡，跨节点理论聚合 ≈ 400 GB/s；实测跨机 AllReduce busbw 若只有 30~50 GB/s，说明只用了 1 张网卡或 IB 没配好 GPUDirect RDMA。要让多机打满，必须 **每 GPU 绑一张邻近网卡** 且开启 GPUDirect RDMA。

---

## 8. 多机测试：MPI + 网络环境变量

### 8.1 基本命令

10 个进程（可能跨多节点）、每进程 4 卡，共 40 GPU：

```bash
$ mpirun -np 10 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

更现实的写法（每进程 1 卡，靠 `-np` 和 hostfile 控制总数）：

```bash
$ mpirun -np 16 -H node1:8,node2:8 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

```
        多机数据面 / 控制面
  node1                        node2
  ┌──────────────┐            ┌──────────────┐
  │ GPU0..7      │  IB/RoCE   │ GPU0..7      │
  │  └ NVLink内连│◄══════════►│  NVLink内连  │  ← 数据面：NCCL 走 IB
  └──────┬───────┘  (RDMA)    └──────┬───────┘
         │ MPI over TCP(eth0)        │
         └───────── 控制面：交换 NCCL ID ┘
```

### 8.2 关键网络环境变量（`-x` 透传给所有进程）

| 变量 | 作用 | 不设的后果 |
|------|------|-----------|
| `NCCL_DEBUG=INFO` | 打印拓扑/通道/选路日志 | 排错全靠它 |
| `NCCL_IB_HCA=mlx5` | 指定用哪些 IB 网卡（前缀匹配）| 可能选错卡或走 socket |
| `NCCL_SOCKET_IFNAME=eth0` | 控制面/兜底用哪个网口 | 可能选到管理网/docker0 |
| `NCCL_IB_DISABLE=0/1` | 是否禁用 IB（1=强制走 socket）| 调试用，正式跑 0 |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 启用级别 | 影响是否绕过主机内存 |
| `NCCL_P2P_DISABLE` | 是否禁 P2P（NVLink/PCIe 直连）| 调试用 |
| `NCCL_ALGO` | 强制 Ring/Tree 算法 | 一般让它自适应 |

> 这些是**含义**，不要死记默认值——不同 NCCL 版本默认行为不同，**以官方文档为准**。原则：先用 `NCCL_DEBUG=INFO` 看它实际选了什么网卡和算法，再决定要不要覆盖。

---

## 9. 调优与诊断 checklist

```
 ┌─ busbw 偏低？按此顺序排查 ─────────────────────┐
 │ 1. nvidia-smi topo -m  → GPU 间真是 NVLink(NV#)? │
 │    还是退化成 PCIe(PIX/PHB/SYS)?                 │
 │ 2. NCCL_DEBUG=INFO     → 看选了 Ring/Tree、几个  │
 │    channel、用了哪些 IB 网卡                     │
 │ 3. 多机：每 GPU 是否绑了"最近"的网卡(NUMA 亲和)? │
 │ 4. GPUDirect RDMA 开了吗(NCCL_NET_GDR_LEVEL)?    │
 │ 5. -w warmup 够不够? -n 迭代够不够? 否则抖动     │
 │ 6. 消息够大吗? 小消息 busbw 低是正常的           │
 │ 7. PCIe 是否降速(lspci 看 LnkSta)? ACS 是否关?   │
 └───────────────────────────────────────────────┘
```

诊断要点：

- **基线对比**：调优前后跑同一条命令，比 `Avg bus bandwidth`，用数字说话。
- **延迟 vs 带宽分开看**：小消息行的 `time` 反映延迟（建链/启动开销），大消息行的 `busbw` 反映带宽上限。两类问题修法不同。
- **逐级缩小**：先单卡 sendrecv（链路）→ 单机 8 卡 AllReduce（NVLink）→ 双机 16 卡 AllReduce（IB）。哪一级掉就在哪一级查。
- **`-c 1` 仅验收用**：开校验会变慢，纯压带宽时关掉。
- **数值正确性**：`#wrong` 必须为 0，否则带宽数字没意义（可能 ECC/硬件故障）。

---

## 数值例子 / 对照 / 实践

### 例1：AllReduce 手算 busbw（验证第 6 节公式）

设：8 张 GPU（$n=8$），消息 $S=128\text{ MB}=128\times10^6$ B，耗时 $t=1.0\text{ ms}=10^{-3}$ s。

**第一步，algbw**（用户视角）：

$$
\text{algbw} = \frac{S}{t} = \frac{128\times10^6}{10^{-3}} = 1.28\times10^{11}\ \text{B/s} = 128\ \text{GB/s}
$$

**第二步，系数**（$n=8$）：

$$
C = \frac{2(n-1)}{n} = \frac{2\times7}{8} = \frac{14}{8} = 1.75
$$

**第三步，busbw**（硬件视角）：

$$
\text{busbw} = \text{algbw}\times C = 128 \times 1.75 = 224\ \text{GB/s}
$$

> 含义：用户感觉传了 128 GB/s，但每条链路实际跑了 224 GB/s。若硬件是 A100 NVLink（聚合约 600 GB/s 双向 / 300 GB/s 单向量级），224 GB/s 已接近打满——健康。

### 例2：判断是否走了 NVLink

某机 8 卡 AllReduce 大消息 busbw 实测仅 **35 GB/s**。

- NVLink 3.0 单 GPU 聚合期望几百 GB/s，35 GB/s 远低于此。
- 35 GB/s 恰好在 PCIe 4.0 x16（约 32 GB/s）量级附近。
- **结论**：NVLink 没用上，数据走了 PCIe。`nvidia-smi topo -m` 验证 → 若显示 `PIX/PHB/SYS` 而非 `NV#`，证实。修：检查 NVSwitch 驱动 / fabric-manager 服务 / `NCCL_P2P_DISABLE` 是否被误设为 1。

### 例3：算法选择对照

同样传 1 GB 数据、同样 $t$，谁的 busbw 数字大？

| 算法 | 系数 | 若 algbw=100 GB/s，则 busbw |
|------|------|----------------------------|
| Broadcast | 1 | 100 GB/s |
| AllGather | 7/8=0.875 | 87.5 GB/s |
| AllReduce | 1.75 | 175 GB/s |

> 这说明 **busbw 才能横向比硬件**：AllReduce 数字最大不是因为它"更快"，而是它在链路上搬的数据本就多——busbw 把这个差异归一化了。

### 实践模板（直接抄）

```bash
# 单机 8 卡体检（带宽区扫到 2G，开预热与多迭代）
./build/all_reduce_perf -b 1M -e 2G -f 2 -g 8 -n 50 -w 10

# 双机 16 卡体检（每进程 1 卡，IB）
mpirun -np 16 -H node1:8,node2:8 \
  -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=mlx5 \
  ./build/all_reduce_perf -b 1M -e 2G -f 2 -g 1 -n 50 -w 10

# 看到末行 "Avg bus bandwidth : XXX" 即为汇总指标
```

---

## 常见问题

| 问题 | 答案 |
|------|------|
| algbw 和 busbw 哪个判断打满？ | **busbw**，它是链路真实速率，可直接对照硬件上限 |
| 为什么 AllReduce 的 busbw 比 algbw 大近 2 倍？ | 系数 $2(n-1)/n$，$n$ 大时趋近 2，因为 ReduceScatter+AllGather 干两遍 |
| 小消息 busbw 很低正常吗？ | 正常，小消息被**延迟**主导，没进入带宽饱和区 |
| `-g 8` 和 `mpirun -np 8 ... -g 1` 区别？ | 前者单进程 8 卡，后者 8 进程各 1 卡；**多机必须用 MPI 多进程** |
| 多机比单机慢很多？ | 正常，跨节点走 IB（几十 GB/s）远慢于 NVLink（几百 GB/s）；但若 IB 没打满才是问题 |
| MPI 参与高速数据传输吗？ | **不**，MPI 只拉起进程、交换 NCCL ID（控制面），数据面是 NCCL 走 NVLink/IB |
| `#wrong` 非 0 怎么办？ | 硬件/ECC 问题，带宽数字作废，先修正确性 |
| busbw 上不去先看什么？ | `nvidia-smi topo -m` 确认走 NVLink 不是 PCIe；再看 GDR、NUMA 亲和、PCIe 降速 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[ai-infra/网络/NCCL]] — 通信库本体：Ring/Tree、拓扑探测、环境变量
- [[ai-infra/网络/集合通信原语]] — AllReduce/AllGather/ReduceScatter 等语义
- [[ai-infra/网络/HPC性能测试]] — 更广义的网络/HPC 基准方法论
- 官方仓库：https://github.com/NVIDIA/nccl-tests
- 百度云文档：https://cloud.baidu.com/doc/GPU/s/Yl3mr0ren
- 火山引擎《基于 NCCL 的多机 RDMA 网络性能测试》：https://www.volcengine.com/docs/6419/105002
