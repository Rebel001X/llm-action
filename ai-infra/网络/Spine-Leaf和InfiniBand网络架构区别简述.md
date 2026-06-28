# Spine-Leaf 与 InfiniBand 网络架构区别简述

> 一句话定位：**Spine-Leaf 讲的是"交换机怎么连成拓扑"（架构层），InfiniBand 讲的是"用什么协议/网卡跑数据"（技术层）——它们不是同级对手，常被放在一起对比是因为大模型训练集群里二者经常一起出现。** 📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 小节 | 你将搞懂的问题 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | 两者到底是不是一类东西？ | 架构层 vs 技术层 |
| 1. 前置：三层网络为什么不够用 | 大模型训练为什么逼着网络换架构 | 东西向流量、收敛比 |
| 2. Spine-Leaf 架构原理 | 扁平两层是怎么做到"无阻塞"的 | CLOS、等价多路径 |
| 3. InfiniBand 技术原理 | RDMA 为什么能把延迟压到 μs 级 | RDMA、内核旁路、SHARP |
| 4. 关键对比 | 维度逐条拆解二者差异 | 协议/延迟/成本 |
| 5. 它们如何同时存在 | 一个 GPU 集群里二者各管哪一段 | 计算网 vs 存储网 |
| 6. 数值手算 | 收敛比、二分带宽怎么算 | bisection bandwidth |
| 实操 | 厂商、速率代际、选型 | HDR/NDR、Ethernet |
| 常见问题/坑 | 容易搞混与翻车的点 | RoCE、收敛比、混淆层级 |

## 0. 一句话锚点

> **Spine-Leaf 和 InfiniBand 不在同一抽象层。**
>
> - **Spine-Leaf** = 一种**网络拓扑/架构**（怎么把交换机连起来），它本身不规定用什么协议，既可以跑 Ethernet，也可以跑 InfiniBand。
> - **InfiniBand** = 一整套**网络技术栈**（物理层 + 链路层 + 传输层 + RDMA 编程模型 + 专用网卡 HCA），它内部组网时**也常常用 Spine-Leaf（即 Fat-Tree）拓扑**。

把它们对立起来谈"区别"，本质是在对比两种**典型部署形态**：

- 形态 A：**Spine-Leaf + 以太网（Ethernet）** —— 通用数据中心。
- 形态 B：**InfiniBand（其组网拓扑通常也是 Spine-Leaf / Fat-Tree）** —— HPC 与大模型训练。

记住这点，下面所有"区别"才不会越读越乱。

## 1. 前置：传统三层网络为什么撑不住大模型

### 1.1 老架构：核心-汇聚-接入（三层树）

传统数据中心是**三层树形**：接入层(Access) → 汇聚层(Aggregation) → 核心层(Core)。它为"南北向流量"（用户↔服务器，进出数据中心）优化。

```
                 ┌─────────┐ ┌─────────┐
   Core 核心      │  Core   │ │  Core   │
                 └────┬────┘ └────┬────┘
                ┌─────┴───┐  ┌────┴────┐
   Aggregation │  Agg    │  │  Agg    │   汇聚
                └──┬───┬──┘  └──┬───┬──┘
   Access      ┌──┴┐ ┌┴──┐  ┌─┴─┐ ┌┴──┐
                │TOR│ │TOR│  │TOR│ │TOR│   接入(机柜顶交换机)
                └─┬─┘ └─┬─┘  └─┬─┘ └─┬─┘
                  服务器  ...
```

### 1.2 问题：东西向流量爆炸

大模型训练里，GPU 之间要做 **AllReduce / AllGather** 等[[ai-infra/网络/集合通信原语|集合通信]]，海量流量是**服务器↔服务器（东西向）**。

在三层树里，两台机柜的服务器通信要"上行到汇聚甚至核心，再下行"——

- **路径长** → 跳数多 → 延迟高、抖动大。
- **上层带宽收敛** → 接入层进来的总带宽远大于上行带宽（典型收敛比 4:1 甚至更高），东西向一拥塞就**丢包/排队**。
- 集合通信对**慢尾巴（straggler）**极敏感：1024 卡里只要 1 条链路掉速，整步 AllReduce 就被拖慢。

> **核心矛盾**：训练流量是东西向的、突发的、同步的；老架构是为南北向、平稳流量设计的。于是有了 **Spine-Leaf**。

## 2. Spine-Leaf 架构原理

### 2.1 结构：只有两层，全互联

Spine-Leaf 把网络压成**两层**：

- **Leaf（叶交换机）**：连服务器/GPU 节点（相当于 TOR）。
- **Spine（脊交换机）**：只连 Leaf，**不直接连服务器**。
- **规则**：**每一个 Leaf 都连到每一个 Spine**，Leaf 之间不互连，Spine 之间也不互连。

```
            ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  Spine     │ Sp 1 │   │ Sp 2 │   │ Sp 3 │   │ Sp 4 │
            └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
               │ ╲   ╳   ╱ │ ╲   ╳   ╱ │  全互联(每Leaf↔每Spine)
               │  ╲ ╱ ╲ ╱  │  ╲ ╱ ╲ ╱  │
            ┌──┴──┐    ┌──┴──┐    ┌──┴──┐    ┌──┴──┐
  Leaf      │Leaf1│    │Leaf2│    │Leaf3│    │Leaf4│
            └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘
              GPU节点    GPU节点    GPU节点    GPU节点
```

### 2.2 为什么这样设计——三个直接收益

1. **任意两节点固定 3 跳**：服务器 → Leaf → Spine → Leaf → 服务器，**最多两跳交换机**。无论集群多大，东西向延迟**一致且可预测**（解决了三层树"跳数随距离变化"的抖动）。
2. **等价多路径（ECMP）**：从 LeafA 到 LeafB，可以走任意一个 Spine——有几个 Spine 就有几条等价路径。流量按 5 元组 hash 分摊到所有 Spine 上，**天然负载均衡**。
3. **横向扩展（scale-out）**：带宽不够就**加 Spine**；端口不够就**加 Leaf**。加 Spine 时所有 Leaf 各连一根上行即可，不用动现有结构。

### 2.3 它的理论原型：CLOS / Fat-Tree

Spine-Leaf 是 1950 年代电话交换网 **CLOS 网络**的现代化身。当满足"上行带宽 ≥ 下行带宽"时，它是**无阻塞（non-blocking）/ 全二分带宽（full bisection bandwidth）**的——即任意一半节点同时给另一半发数据，网络都不会成为瓶颈。这正是集合通信梦寐以求的特性。

> ⚠️ 注意：Spine-Leaf 是**拓扑**，跑什么协议是另一回事。它最常配 **Ethernet**，但**InfiniBand 组网用的 Fat-Tree 本质也是 Spine-Leaf**。原文表格写"Spine-Leaf 以 Ethernet 为传输协议"是指**最常见的部署形态**，不是架构的硬性规定。

## 3. InfiniBand 技术原理

### 3.1 它是一整套栈，不是一根线

InfiniBand（IB）是面向 HPC 的**端到端网络技术**，包含：

- **HCA（Host Channel Adapter）**：IB 专用网卡（区别于以太网 NIC）。
- **IB 交换机**：跑 IB 链路层协议（**不是**以太网交换机）。
- **Subnet Manager（SM）**：集中式控制器，负责给每个端口分配 **LID（本地标识）** 并下发转发表——IB 是"集中管控"的，不像以太网靠分布式协议自学习。
- **RDMA 编程模型**：上层用 Verbs API（`ibv_*`）直接操作网卡队列。

### 3.2 杀手锏：RDMA（远程直接内存访问）

普通以太网 + TCP/IP 收发一个包，要经过：用户态→内核态拷贝→协议栈→网卡，**多次内存拷贝 + CPU 中断 + 上下文切换**，延迟几十~上百 μs，还吃 CPU。

RDMA 让网卡**直接读写远端主机内存**：

```
  传统 TCP/IP 路径                    RDMA 路径(内核旁路 Kernel Bypass)
  ┌──────────┐                        ┌──────────┐
  │ App 用户态│                        │ App 用户态│──┐ 直接提交WQE到网卡队列
  ├──────────┤  ←拷贝                  └──────────┘  │ (不进内核, CPU不参与搬数据)
  │ Kernel   │  ←协议栈+中断                          ▼
  │ TCP/IP   │                                  ┌─────────┐
  ├──────────┤  ←拷贝                            │  HCA网卡 │──DMA读写远端内存
  │ Driver   │                                  └─────────┘
  ├──────────┤
  │  NIC     │
  └──────────┘
```

收益：

- **内核旁路（kernel bypass）**：绕过内核协议栈。
- **零拷贝（zero-copy）**：数据直接从应用内存进网卡。
- **CPU 旁路**：搬数据不占 CPU，CPU 可专心算。
- 端到端延迟可压到 **~1 μs 量级**，远低于普通以太网。

> 这正是 [[ai-infra/网络/集合通信原语|集合通信]]（NCCL 底层）偏爱 IB 的原因：AllReduce 要在成千上万卡之间高频同步小消息，延迟每降 1 μs，整体训练吞吐都受益。

### 3.3 无损 + 在网计算

- **基于信用的流控（credit-based flow control）**：发送方只在确认接收方有缓冲空间时才发，**链路层天生无损（lossless）、不丢包**——以太网默认会丢包靠重传，IB 不需要。
- **SHARP（在网计算 In-Network Computing）**：把 AllReduce 的**求和操作下放到交换机里做**，数据在网络中"边走边算",减少数据往返。这是纯拓扑层面的 Spine-Leaf 给不了的。

## 4. 关键对比（逐维拆解）

| 维度 | Spine-Leaf（+Ethernet 形态） | InfiniBand |
| --- | --- | --- |
| **抽象层** | 网络**架构/拓扑** | 端到端**网络技术栈**（含网卡/交换机/协议/编程模型） |
| **典型协议** | Ethernet（也可承载 IB） | InfiniBand 链路层（非以太网） |
| **数据搬运** | TCP/IP 协议栈（或 RoCE 上的 RDMA） | 原生 RDMA，内核旁路 |
| **丢包模型** | 默认可丢、靠重传（需 PFC/ECN 调成无损） | 链路层信用流控，**天生无损** |
| **典型延迟** | 数十 μs（TCP）/ 个位数 μs（RoCE 调优后） | **~1 μs** 量级 |
| **管理模型** | 分布式（路由协议自学习） | 集中式 Subnet Manager |
| **在网计算** | 一般无 | 有（SHARP） |
| **典型场景** | 云计算、通用数据中心、虚拟化 | HPC、超算、大模型训练 |
| **成本** | 相对经济（生态成熟、通用） | 相对昂贵（专用硬件） |
| **常见厂商** | Cisco、Arista Networks、Juniper Networks | NVIDIA Mellanox、Intel、Cavium |

> 原文要点全部保留并细化：Spine-Leaf 由 Spine+Leaf 组成、提供低延迟高带宽可扩展、以 Ethernet 为协议、用于云/数据中心；InfiniBand 低延迟高并行、支持 RDMA、用于科学计算/数据分析/模拟；速率与厂商见下方"实操"节。

## 5. 它们如何在同一个 GPU 集群里同时存在

现实里二者**不是二选一**，而是分工：

```
   ┌──────────────────────── 一个大模型训练集群 ─────────────────────────┐
   │                                                                    │
   │   GPU服务器 ── (节点内) NVLink/NVSwitch ── GPU服务器                │
   │      │                                          │                  │
   │      │  计算网(后端网) = InfiniBand(NDR/HDR)      │  ← 跑集合通信     │
   │      │  拓扑 = Fat-Tree(就是 Spine-Leaf!)         │     AllReduce    │
   │      ▼                                          ▼                  │
   │  ┌──────IB Leaf/Spine 交换机(Fat-Tree)──────┐                       │
   │                                                                    │
   │   GPU服务器 ── 存储网/管理网 = Spine-Leaf + Ethernet                 │
   │              (访问对象存储/NFS、带外管理、监控)  ← 跑通用流量          │
   └────────────────────────────────────────────────────────────────────┘
```

- **后端计算网（GPU↔GPU）**：追求极致低延迟无损 → **InfiniBand**（其物理拓扑就是 Spine-Leaf/Fat-Tree）。
- **前端/存储/管理网**：追求通用、经济 → **Spine-Leaf + Ethernet**。
- **节点内 GPU 之间**：用 NVLink/NVSwitch（见 [[ai-infra/算力/GPU工作原理]]），那是更底层的另一回事。

所以"Spine-Leaf vs InfiniBand"更准确的说法是：**Ethernet 形态的 Spine-Leaf vs InfiniBand 形态的网络**——而后者内部往往**也是** Spine-Leaf。

## 6. 数值手算

### 6.1 收敛比（oversubscription ratio）

收敛比 = 下行带宽（连服务器）÷ 上行带宽（连 Spine）。

设一台 Leaf 有 48 个下行口接 GPU、各 200 Gbps，6 个上行口接 Spine、各 400 Gbps：

$$
\text{下行} = 48 \times 200 = 9600 \text{ Gbps},\quad
\text{上行} = 6 \times 400 = 2400 \text{ Gbps}
$$

$$
\text{收敛比} = \frac{9600}{2400} = 4{:}1
$$

含义：满负载东西向流量会被压到 1/4，存在瓶颈。**HPC/训练网通常要做到 1:1（无收敛/全无阻塞）**，即上行总带宽 = 下行总带宽，才能保证全二分带宽。

### 6.2 二分带宽（bisection bandwidth）

把网络从中间切成两半，跨切面的总带宽就是二分带宽，衡量"一半节点同时给另一半发数据"的能力。

设 N 个 Leaf、每 Leaf 连 K 个 Spine、每条上行 B：在 1:1 无阻塞设计下，二分带宽 ≈

$$
\text{Bisection BW} \approx \frac{N \times K \times B}{2}
$$

例：N=32 个 Leaf，每个 K=16 条上行，每条 B=400 Gbps：

$$
\frac{32 \times 16 \times 400}{2} = 102{,}400 \text{ Gbps} = 102.4 \text{ Tbps}
$$

AllReduce 的理论耗时正比于数据量 ÷ 可用带宽，二分带宽越大、收敛比越接近 1:1，[[ai-infra/网络/集合通信原语|集合通信]]越快。

### 6.3 IB 速率代际换算

原文提到 **HDR 可达 200 Gbps**。IB 速率以"通道数 × 单通道速率"计，常见 4x（4 通道）：

| 代际 | 单通道(Gbps) | 4x 端口(Gbps) |
| --- | --- | --- |
| FDR | 14 | 56 |
| EDR | 25 | 100 |
| **HDR** | **50** | **200** |
| NDR | 100 | 400 |
| XDR | 200 | 800 |

以太网常见 10 / 25 / 100 / 200 / 400 GbE。对应代际带宽已可比，IB 的优势更多在**延迟、无损、在网计算**，而非单纯峰值速率。

## 实操：选型与厂商速查（保留并扩展原文真料）

**速率（原文：HDR 200 Gbps；Ethernet 10 Gbps 或更高）**

- InfiniBand：FDR 56G → EDR 100G → **HDR 200G** → NDR 400G → XDR 800G。
- Ethernet：10 / 25 / 40 / 100 / 200 / 400 GbE，生态通用。

**常用厂商（原文保留）**

- Spine-Leaf（Ethernet）：**Cisco、Arista Networks、Juniper Networks**。
- InfiniBand：**NVIDIA Mellanox（原 Mellanox Technologies）、Intel、Cavium**。

**选型口诀**

| 你的诉求 | 选什么 |
| --- | --- |
| 极致低延迟 + 无损 + 大规模训练后端网 | InfiniBand（拓扑用 Fat-Tree/Spine-Leaf） |
| 通用、成本敏感、已有以太网生态 | Spine-Leaf + Ethernet；要低延迟可上 **RoCEv2**（见 [[ai-infra/网络/InfiniBand]] 相关） |
| 存储网 / 管理网 / 南北向 | Spine-Leaf + Ethernet |

> 想看 IB 的命令、监控、Docker 化部署，见同目录 `InfiniBand.md` / `IB流量监控.md` / `IB-docker.md` / `IB软件.md`。

## 常见问题 / 坑

| 坑 / 误区 | 真相 | 后果 / 建议 |
| --- | --- | --- |
| 把二者当同级对手 | Spine-Leaf 是**拓扑**，InfiniBand 是**技术栈**；IB 组网也用 Spine-Leaf | 对比时先明确"对的是 Ethernet 形态 vs IB 形态" |
| 以为 Spine-Leaf 一定跑以太网 | 拓扑与协议解耦；IB 的 Fat-Tree 就是 Spine-Leaf | 别用"协议"反推"架构" |
| 忽视收敛比 | 不是有 Spine-Leaf 就无阻塞，**上行<下行就有收敛** | 训练后端网要做 1:1 全无阻塞 |
| 以为以太网 = 不能 RDMA | **RoCE（RDMA over Converged Ethernet）** 让以太网也跑 RDMA | RoCEv2 需 PFC/ECN 调成无损，否则丢包性能崩 |
| IB 当成"插上就快" | IB 依赖 **Subnet Manager** 分配 LID、建路由 | SM 没配好整个子网不通 |
| 只看峰值带宽选网络 | 训练瓶颈常是**延迟/抖动/慢尾巴**，不是峰值 | 关注端到端延迟、无损、二分带宽 |
| 忽略一条慢链路 | 同步集合通信被最慢链路拖累（straggler） | 全网一致拓扑 + 监控（见 `IB流量监控.md`） |
| 把节点内 NVLink 也算进来对比 | NVLink 是**节点内** GPU 互联，与机间网络不同层 | 分清：NVLink(节点内) / IB·Ethernet(节点间) |

## 🔗 跳转链接

**枢纽**：[[00-知识地图]]

**本主题强相关**
- [[ai-infra/网络/InfiniBand]] —— IB 技术细节、命令与配置
- [[ai-infra/网络/集合通信原语]] —— AllReduce/AllGather，网络性能的最终消费者
- [[ai-infra/ai-hardware/硬件对比]] —— 网卡/交换机硬件选型
- [[ai-infra/算力/GPU工作原理]] —— 节点内 NVLink/NVSwitch 与机间网的边界
- [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]

**训练 / 推理为何依赖网络**
- [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- [[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- [[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/解码策略]]

**算法 / 优化 / 评测**
- [[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- [[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- [[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
