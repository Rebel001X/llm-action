# 昇腾集群网络(Ascend Network / 互联与集合通信网络)

> 昇腾 AI 集群里"芯片之间、服务器之间、机柜之间如何用高速网络连成一台超级计算机"的那一层。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]] [[ai-infra/ai-hardware/CUDA]]

## 阅读地图

| 节 | 你会学到 | 对标英伟达世界 |
|----|---------|---------------|
| 0 | 一句话锚点：网络是"被规模放大"的瓶颈 | 同理 |
| 1 | 昇腾软件/硬件栈里"网络"在哪一层 + 生态对照表 | NVLink/NVSwitch/IB/RoCE + NCCL |
| 2 | 三层互联：HCCS(片间) / PCIe(片-CPU) / RoCE(机间) | NVLink / PCIe / InfiniBand |
| 3 | RoCE 与无损以太(iLossless / PFC / ECN) | RoCE v2 + DCQCN |
| 4 | 网络拓扑:服务器内全互联 + 集群 Spine-Leaf | NVSwitch + Fat-Tree |
| 5 | HCCL 如何利用网络做集合通信(环/HD 算法) | NCCL Ring / Tree |
| 6 | 系统级调优:通信库×拓扑×并行的协同 | Topology-aware NCCL |
| — | 迁移要点与常见坑 | 从 IB+NCCL 迁到 RoCE+HCCL |
| — | 常见问题 / 跳转链接 | — |

## 0. 一句话锚点

单卡训练时网络不存在;一旦扩到几百上千张 NPU,**绝大部分"扩不上去"的性能损失都发生在网络上**——梯度同步、参数广播、流水线气泡的传递,全部走互联链路。所以理解昇腾集群,本质是理解它的**三层互联**(片间 / 片-CPU / 机间)和**集合通信怎么贴着拓扑跑**。一句话:

> 算力决定单卡上限,**网络决定集群能不能把单卡上限乘起来(线性度)**。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

昇腾软件栈自底向上大致是:**硬件(NPU / 互联) → CANN(驱动+运行时+算子+HCCL) → 框架(MindSpore / PyTorch+Ascend) → 套件(MindFormers / MindIE / ModelLink)**。

"网络"横跨两层:

- **物理/链路层**:HCCS 高速总线、PCIe、RoCE(100G/200G 以太)——属于**硬件 + 驱动**。
- **集合通信层**:HCCL(Huawei Collective Communication Library)——属于 **CANN**,它把"AllReduce/AllGather/Broadcast"这些原语映射到上面的物理链路上。

```
          昇腾软件栈            |        网络在哪一层
  ─────────────────────────────┼──────────────────────────────
  套件 MindFormers/MindIE      |  调用分布式并行(DP/TP/PP)
  框架 MindSpore / PyTorch     |  发起 all_reduce 等通信
  ─────────────────────────────┼──────────────────────────────
  CANN  ┌── HCCL 集合通信库 ───┼─►  环/HD 算法、通信域、流
        ├── Runtime / 驱动      |    管理 device、RDMA 队列
  ─────────────────────────────┼──────────────────────────────
  硬件  ┌── NPU(达芬奇)        |
        ├── HCCS  片间高速总线  ─┼─►  服务器内 NPU-NPU
        ├── PCIe  片↔CPU        |
        └── RoCE  100G/200G 以太 ┼─►  服务器↔服务器(跨机)
```

### 昇腾 ↔ 英伟达 网络生态对照表(迁移心智图)

| 维度 | 昇腾(Ascend) | 英伟达(NVIDIA) | 说明 |
|------|--------------|----------------|------|
| 片间高速互联(节点内) | **HCCS** 高速总线 | **NVLink** | NPU/GPU 直连,带宽远高于 PCIe |
| 节点内交换芯片 | HCCS Mesh / 全互联 | **NVSwitch** | 让节点内任意两卡满带宽 |
| 芯片↔CPU | **PCIe 4.0/5.0** | PCIe 4.0/5.0 | 这一层两家一样 |
| 跨节点网络 | **RoCE v2**(100/200G 无损以太) | **InfiniBand** 或 RoCE | 昇腾主推 RoCE+无损以太 |
| 无损/拥塞控制 | **iLossless**(PFC+ECN+AI 调参) | DCQCN / IB 信用流控 | 目标都是 0 丢包、低尾延迟 |
| 数据中心交换机 | CloudEngine 8800 等 | Spectrum / Quantum | Spine-Leaf 组网 |
| 集合通信库 | **HCCL** | **NCCL** | AllReduce/AllGather 等原语 |
| 通信原语接口 | hccl all_reduce... | nccl AllReduce... | 语义基本一一对应 |
| 拓扑感知 | HCCL 自动选环/HD | NCCL Ring/Tree 自选 | 都按拓扑选最优算法 |

**一句话记忆**:**HCCS↔NVLink、RoCE↔InfiniBand、HCCL↔NCCL、iLossless↔DCQCN**。把这四对刻在脑子里,昇腾集群网络的迁移地图就建好了。

## 2. 三层互联:HCCS / PCIe / RoCE

昇腾 AI 服务器(如 Atlas 系列)内部与集群层面,采用三类高速互联,**各管一段距离**:

1. **HCCS(片间,服务器内)**:昇腾 910 处理器之间通过 HCCS 高速总线直连,带宽显著高于 PCIe,用于节点内 NPU-NPU 的高频通信(如 TP 张量并行的 all-reduce)。对标 **NVLink**。
2. **PCIe 4.0(片↔CPU)**:NPU 与 CPU/主机内存之间用 PCIe 4.0(速率约 16 GT/s/lane),是 PCIe 3.0(约 8 GT/s)的两倍,负责数据装载、host-device 搬运。
3. **RoCE / 100G 以太(机间,跨服务器)**:集群层面用面向数据中心的交换机(如 CloudEngine 8800 系列),单端口 100Gbps,把所有 AI 服务器接入高速无损交换网络,承载跨机的梯度同步、参数广播。对标 **InfiniBand**。

```
   ┌───────────── AI 服务器(节点)─────────────┐
   │   NPU0 ══HCCS══ NPU1 ══HCCS══ NPU2 ══ ...  │   ← 节点内:HCCS 全互联(↔NVLink)
   │     ║PCIe         ║PCIe                     │
   │   ┌─┴───── CPU / Host Mem ─────┐            │   ← 片↔CPU:PCIe 4.0
   │   └────────── RoCE NIC ────────┘            │
   └──────────────────│─────────────────────────┘
                      │ 100/200G RoCE(↔InfiniBand)
              ┌───────┴────────┐
              │  CloudEngine    │   ← 集群:Spine-Leaf 无损以太交换
              │   交换机(Leaf) │
              └───────┬────────┘
                      │
            其他成百上千个节点 ...
```

> 距离越短、带宽越高:**HCCS > PCIe > RoCE**。集合通信算法的核心就是**尽量把高频流量压在 HCCS 内,只让必须跨机的流量走 RoCE**。

## 3. RoCE 与无损以太(iLossless)

跨机这一层昇腾走 **RoCE v2(RDMA over Converged Ethernet)**:让网卡直接读写远端内存,**绕过内核与 CPU 拷贝**,这正是 RDMA "零拷贝、内核旁路"的价值,效果上和 InfiniBand 类似,但跑在以太网上(更易复用数据中心基础设施)。

RoCE 的命门是:**以太网默认会丢包,而 RDMA 对丢包极其敏感**(一旦丢包要回退重传,尾延迟暴涨)。于是要把以太网做成"无损":

- **PFC(Priority Flow Control)**:基于优先级的逐跳反压,接收端快满了就让上游"暂停发",避免缓冲区溢出丢包。
- **ECN(Explicit Congestion Notification)**:交换机在拥塞早期给报文打标记,发送端据此**主动降速**(对标 NVIDIA 的 DCQCN)。
- **iLossless 智能无损**:在 PFC/ECN 之上,对集群内流量做**实时学习训练**自动调参,目标是网络 **0 丢包**与 **E2E μs 级时延**,缓解 PFC 死锁、参数难调等老问题。

```
   发送端 NPU                                接收端 NPU
      │  RDMA write(绕过 CPU/内核)            ▲
      ▼                                        │
   ┌──────┐   ECN 标记拥塞    ┌──────┐         │
   │ Leaf │──────────────────►│ Spine│──── ... │
   └──────┘◄── PFC 反压暂停 ──└──────┘         │
      │   iLossless 实时学习流量、动态调参      │
      └───────────► 目标:0 丢包 + μs 级时延 ───┘
```

## 4. 网络拓扑:节点内全互联 + 集群 Spine-Leaf

- **节点内**:NPU 之间 HCCS **全互联(full-mesh)**,使任意两卡都能高带宽直达——这与 NVSwitch 让节点内 GPU 全互联是同一思路。
- **集群层**:采用 **Spine-Leaf(脊叶)**两级 CLOS 组网,接近 Fat-Tree。Leaf 接服务器,Spine 连 Leaf,实现**百 TB 级全互联、无阻塞**的专属参数同步网络。这样任意两个节点之间都有多条等价路径,带宽可水平扩展。

```
            Spine1      Spine2      Spine3      ← 脊交换机(无阻塞核心)
             ╱ │ ╲       ╱ │ ╲       ╱ │ ╲
        ┌───┘  │  └──┐ ...  (full-mesh 连接所有 Leaf)
      Leaf1   Leaf2   Leaf3   Leaf4  ...        ← 叶交换机
       │ │     │ │     │ │     │ │
      节点 节点 节点 节点 节点 节点 ...           ← AI 服务器(节点内 HCCS 全互联)
```

> 关键收益:梯度同步时延缩短(华为给出 10~70% 量级的优化区间),为大规模数据/模型并行提供低延迟、无阻塞的同步通道。**具体性能数字以华为昇腾官方文档(Ascend 社区)为准。**

## 5. HCCL 如何利用网络做集合通信

HCCL 是昇腾的集合通信库(对标 NCCL),它把上层框架发起的 **AllReduce / AllGather / ReduceScatter / Broadcast / All2All** 等原语,**映射到第 2 节的三层物理链路上**,并按拓扑选择算法:

- **Ring(环)算法**:把参与设备排成环,数据分块沿环流转。带宽利用率高,**通信量与设备数无关**,适合大消息 AllReduce——和 NCCL 的 Ring 同源思路。
- **HD / Halving-Doubling(对半翻倍)**:按 2 的幂步数对半交换,**步数仅 log₂N**,小消息延迟更低(对标 NCCL Tree/递归对半)。
- **分层(Hierarchical)**:**先在节点内用 HCCS 做局部规约,再跨节点用 RoCE 做节点间规约,最后回灌**。这一步是性能关键——它把绝大多数流量留在高带宽的 HCCS 上,只让"每节点一份"的聚合结果走相对慢的 RoCE。

```
   分层 AllReduce(把流量压在 HCCS 内):
   ① 节点内 Reduce(HCCS,快)
      [NPU0..7] ──► 局部和  (每个节点各自得到本节点的部分和)
   ② 节点间 AllReduce(RoCE,慢,但每节点只发 1 份)
      节点A 部分和 ◄──RoCE──► 节点B 部分和 ◄──► ...
   ③ 节点内 Broadcast(HCCS,快)
      把全局和广播回节点内每张 NPU
```

> 选环还是选 HD、分几层,HCCL 会**根据通信域大小、消息大小、底层拓扑自动决策**(也支持手动调参)。这与 NCCL 自动在 Ring/Tree 间切换是同一设计哲学。详见 [[ai-infra/网络/集合通信原语]]。

## 6. 系统级调优:通信库 × 拓扑 × 并行

Atlas 900 这类 AI 集群的"线性度 >80%"不是靠单点堆料,而是**三者协同**:

- **通信库(HCCL)**:选对算法(环/HD/分层),让通信贴着拓扑跑。
- **网络拓扑**:HCCS 全互联 + Spine-Leaf 无阻塞,提供物理带宽与多路径。
- **训练算法/并行策略**:把通信量大的并行(如 TP)放进节点内 HCCS,把通信稀疏的(如 DP/PP)放到跨机 RoCE。

三者**联合调优**(通信库 + 网络拓扑 + 训练算法),才把昇腾处理器的算力真正"喂饱",实现高集群线性度与高作业调度效率。

## 迁移要点 / 注意事项与坑

从"InfiniBand + NCCL"世界迁到"RoCE + HCCL"时:

1. **NCCL → HCCL 的接口替换**:原语语义基本一一对应(AllReduce↔all_reduce),但**初始化通信域、设置 rank/通信域 ID、环境变量名都不同**。具体 API、环境变量与版本以华为昇腾官方文档为准,不要照搬 `NCCL_*` 变量名。
2. **并行映射要"拓扑感知"**:**把张量并行(TP,通信最密集)限制在节点内(HCCS)**,跨机优先放数据并行/流水线并行。若 TP 组被切到跨机 RoCE,带宽骤降会直接拖垮整体吞吐——这是最常见的性能塌方坑。
3. **RoCE 无损配置是第一坑**:PFC/ECN 没配好会导致丢包、PFC 死锁、尾延迟飙升,表现为"通信偶发性变慢/卡死"。无损以太的端到端配置(交换机 + 网卡)依赖运维协同,**具体配置以官方组网指南为准**。
4. **拓扑要让 HCCL 看见**:rank 编排若与物理拓扑不一致,分层算法会失效(把本该走 HCCS 的流量错放到 RoCE)。按官方建议的 rank-table / 拓扑文件生成方式来做。
5. **别照搬 GPU 的调优经验值**:HCCS 与 NVLink、RoCE 与 IB 的带宽/延迟特性不同,梯度桶大小(bucket size)、通信与计算 overlap 的最优点需在昇腾上**重新实测**。
6. **环境/驱动版本一致性**:CANN、HCCL、驱动、固件需配套,版本错配是分布式起不来的高频原因。**安装、镜像、docker、确切版本号一律以华为昇腾官方文档(Ascend 社区)为准。**

> 调优心法(机制层面,非具体数字):**先定位瓶颈在哪一层**(节点内 HCCS / 跨机 RoCE / host PCIe)——用通信 profiling 看各原语耗时;**再对症**:跨机慢就查 RoCE 无损与拓扑映射,节点内慢就查并行切分是否压在 HCCS。

## 常见问题

| 问题 | 解答 |
|------|------|
| HCCS 等于 NVLink 吗? | 定位等价(都是节点内片间高速直连,远快于 PCIe),实现与协议不同,不能混用工具链。 |
| 昇腾跨机为什么主推 RoCE 而不是 IB? | RoCE 跑在以太网上,易复用数据中心生态;配合 iLossless 做到无损,逼近 IB 体验。 |
| iLossless 解决什么? | 让以太网"不丢包":在 PFC/ECN 之上用 AI 实时学习流量动态调参,降尾延迟、防 PFC 死锁。对标 DCQCN。 |
| HCCL 和 NCCL 能互换吗? | 不能。语义对应但实现/接口不同;迁移要换库、改初始化与环境变量。 |
| 为什么 TP 要放节点内? | 张量并行通信最密集,放进高带宽 HCCS;若被切到跨机 RoCE,吞吐会断崖下跌。 |
| 集群线性度 >80% 靠什么? | 通信库(算法)× 网络拓扑(无阻塞)× 并行策略(拓扑感知)三者联合调优。 |
| 安装/版本怎么查? | 一律以华为昇腾官方文档(Ascend 社区)为准,本文不提供具体命令与版本号。 |

## 🔗 跳转链接

- 知识地图:[[00-知识地图]]
- 硬件算力:[[ai-infra/算力/昇腾NPU]]
- 软件生态:[[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 网络对标:[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 框架/套件:[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 训练/推理:[[llm-train/README]] · [[llm-inference/README]]
- 压缩:[[llm-compression/quantization/量化基础]]
- 模型结构:[[llm-algo/transformer/模型架构]]
