# NIXL (推理传输库)

> NIXL（NVIDIA Inference Xfer Library）是 NVIDIA 为大模型推理设计的**统一数据传输库**：用一套 API 把 GPU 显存 / 主机内存 / NVMe 存储 / 远端网络（RDMA/NVLink/TCP）抽象成可互相搬运的"内存段"，专门加速 PD 分离里的 **KV Cache 传输**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/PD分离]] [[llm-inference/Mooncake]] [[ai-infra/ai-hardware/GPU-network]]
> 源码：https://github.com/ai-dynamo/nixl

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点 | 统一传输抽象 / KV 搬运 |
| 1 | 地基：为什么推理需要一个传输库 | PD 分离 / KV Cache / 异构内存 |
| 2 | NIXL 是什么、解决什么 | Xfer Library / 后端插件 |
| 3 | 核心抽象：Agent / Memory / Backend | 注册段 / descriptor |
| 4 | 一次传输的完整生命周期 | register→prep→xfer→check |
| 5 | 后端插件：UCX / GDS / GPUDirect | RDMA / NVLink / NVMe |
| 6 | 与 Dynamo / PD 分离架构的关系 | Conductor / 调度 |
| 7 | 与 Mooncake / 同类对比 | Transfer Engine / 定位 |
| 8 | 数值例子 + 实践要点 | 带宽 / 显存 / 手算 |
| 9 | 常见问题 + 跳转 | FAQ |

---

## 0. 一句话锚点

把推理系统里"**把这块数据从 A 搬到 B**"这件事，无论 A/B 是本机 GPU、远端 GPU、主机内存还是磁盘，都收敛成**同一组 API**。NIXL 自己根据"源在哪、目的在哪"挑选最快的物理通道（NVLink / RDMA / PCIe / NVMe），上层（推理框架）**不必关心传输细节**。

一句话：**NIXL 之于"推理数据搬运"，就像 NCCL 之于"训练梯度同步"**——但 NIXL 面向的是点对点、异构、单边（one-sided）的推理流量，而非集合通信。

---

## 1. 地基：为什么推理需要一个"传输库"

### 1.1 先回顾 KV Cache（最原子的概念）

自回归生成时，每生成一个 token，Transformer 的每一层都要用到**之前所有 token** 的 Key 和 Value 向量。为了不重复计算，把它们缓存下来，就是 **KV Cache**。

单个请求的 KV Cache 大小（字节）：

$$
\text{KV} = 2 \times L \times S \times H \times d_{head} \times n_{kv} \times b
$$

- $2$：K 和 V 两份
- $L$：层数；$S$：序列长度（token 数）
- $H \times d_{head}$ 实际写成 $n_{kv}\times d_{head}$（GQA 下 KV 头数 $n_{kv}$ 远小于注意力头数）
- $b$：每个元素字节数（FP16=2，FP8=1）

它**随序列长度线性增长**，且**只在一个 GPU 上**。这就是后面所有麻烦的根源。

### 1.2 PD 分离（Prefill / Decode 分离）

详见 [[llm-inference/PD分离]]。核心矛盾：

```
Prefill 阶段：处理整段 prompt → 计算密集（compute-bound），吃满算力
Decode 阶段：一次只生成 1 token → 访存密集（memory-bound），吃满带宽
```

两种负载的硬件诉求相反，混在一台机器上互相拖累。**PD 分离**把它们拆到不同 GPU/节点：

```
        ┌───────────┐   KV Cache    ┌───────────┐
请求 ──▶ │ Prefill 机 │ ───────────▶ │ Decode 机 │ ──▶ 输出 token
        │ (算力密集) │  (要搬运!)    │ (带宽密集) │
        └───────────┘               └───────────┘
```

**问题来了**：Prefill 算完的 KV Cache 在 Prefill 机的显存里，必须**搬到** Decode 机的显存里，Decode 才能续写。这块搬运就是推理系统里最重、最频繁、最讲究的数据流。

### 1.3 异构内存的"传输地狱"

KV Cache 可能要在这些位置之间流动：

```
GPU 显存(HBM) ──NVLink──▶ 同机另一块 GPU 显存
GPU 显存      ──RDMA────▶ 远端节点 GPU 显存
GPU 显存      ──PCIe────▶ 主机内存(DRAM)
主机内存      ──NVMe────▶ 本地 SSD (KV 卸载/复用)
主机内存      ──RDMA────▶ 远端存储池
```

每条通道用的 API 完全不同：NVLink/RDMA 走 UCX，GPU↔SSD 走 GPUDirect Storage（cuFile），GPU↔GPU 同机走 cudaMemcpy/IPC……如果让每个推理框架（vLLM、SGLang、TensorRT-LLM、Dynamo）各自去对接这堆底层库，就是 **N 个框架 × M 个后端 = N×M 份重复又易错的代码**。

**NIXL 的存在意义**：把这 M 个后端收口成统一接口，N 个框架只学一套 API。把 $N\times M$ 降成 $N + M$。

---

## 2. NIXL 是什么、解决什么

**NIXL = NVIDIA Inference Xfer Library**（推理传输库），开源于 `ai-dynamo/nixl` 仓库，是 NVIDIA **Dynamo** 推理服务栈的底层传输组件，也可独立使用。

它解决三件事：

| 痛点 | NIXL 的答案 |
|------|------------|
| 异构内存/介质传输 API 五花八门 | 统一抽象成"内存段 + descriptor"，一套 `transfer` 调用 |
| 选哪条物理通道最快要人工判断 | 后端自动协商，按源/目的位置选最优插件 |
| 推理是点对点、单边、动态的，集合通信库（NCCL）不合适 | 专为 one-sided、异步、动态拓扑设计 |

### 2.1 它**不是**什么（划清边界）

- 不是 NCCL：NCCL 做 all-reduce/all-gather 这类**集合通信**，参与方对称、同步、拓扑固定；NIXL 做 **A→B 点对点搬运**，发起方单边发起（类似 RDMA write/read），无需对端 CPU 参与。
- 不是一个新网络协议：它复用现有传输（UCX、GDS、cudaMemcpy 等），只做**统一封装 + 调度**。
- 不是只搬 KV Cache：KV 是头号用例，但它能搬任意已注册内存段（如权重、激活、分布式 KV 池数据）。

```
        上层框架（vLLM / SGLang / Dynamo / 自研）
   ─────────────── NIXL 统一 API ───────────────
   ┌──────────┬──────────┬──────────┬──────────┐
   │ UCX 后端 │ GDS 后端 │ POSIX    │ (更多插件)│
   │ RDMA/    │ GPU↔SSD  │ 主机内存 │           │
   │ NVLink   │ cuFile   │ /文件    │           │
   └──────────┴──────────┴──────────┴──────────┘
       网卡        NVMe       DRAM/磁盘
```

---

## 3. 核心抽象：Agent / Memory / Backend / Descriptor

NIXL 用四个概念把"搬运"讲清楚。

### 3.1 Agent（代理）

一个 **Agent** 代表一个参与传输的进程/角色（通常一个推理 worker 一个 Agent）。Agent 有名字（如 `prefill-0`、`decode-3`），通过**元数据交换**互相认识：A 想往 B 写数据，必须先拿到 B 的内存段元数据。

### 3.2 Memory 注册（register）—— 为什么必须先注册？

要让 RDMA / GPUDirect 直接搬一块内存，硬件需要这块内存被**固定（pinned）并登记到网卡/IOMMU 的地址映射表**里，拿到一个可被远端寻址的"句柄"。这一步叫 **memory registration**。

```
GPU 显存里一块 KV buffer
   │ register()
   ▼
NIXL 把它登记成一个"内存段(segment)"
   → 拿到 memory descriptor: {地址, 长度, 设备类型(VRAM/DRAM/FILE), 后端句柄}
```

注册有开销（要锁页、建映射），所以**注册一次、复用多次**是性能关键：KV buffer 通常在启动时一次性注册成大池子。

### 3.3 Descriptor（描述符）与 Descriptor List

一次传输不是搬"一个地址"，而是搬"一组内存块"（KV Cache 按 block/page 分散存放，是很多小块）。NIXL 用 **descriptor list** 描述：

```
源 descriptor list           目的 descriptor list
[ {addr0, len, VRAM},        [ {addr0', len, VRAM},
  {addr1, len, VRAM},   ──▶    {addr1', len, VRAM},
  {addr2, len, VRAM} ]         {addr2', len, VRAM} ]
   ↑ 一一对应，可批量(batched)一次提交
```

批量提交（一次 `transfer` 带很多块）是关键优化：摊薄每次发起的固定开销，让小块也能跑出高带宽。

### 3.4 Backend（后端插件）

NIXL 把"具体怎么搬"做成**可插拔后端**。常见：

| 后端 | 负责的通道 | 底层 |
|------|-----------|------|
| UCX | 网络（RDMA/RoCE/IB）、本机 NVLink/共享内存 | UCX 库 |
| GDS | GPU 显存 ↔ NVMe SSD | GPUDirect Storage / cuFile |
| POSIX | 主机内存、文件 | 标准系统调用 |

NIXL 在传输时根据"源在 VRAM、目的在远端 VRAM"这样的组合**自动选后端**；上层只说"把这组块从 A 搬到 B"。

---

## 4. 一次传输的完整生命周期

把抽象串成时间线（以 Prefill→Decode 搬 KV 为例）：

```
[Prefill Agent]                          [Decode Agent]
     │                                          │
 1.  │ register(KV buffer)                       │ register(KV buffer)
     │ 拿到本地 descriptor list                  │
     │                                          │
 2.  │ ◀──── 交换元数据(metadata) ────────────▶ │
     │   (Decode 把它的内存段元数据给 Prefill)   │
     │                                          │
 3.  │ create_xfer_req(                          │
     │   local=源descs, remote=Decode的descs,    │
     │   op = WRITE )   ← 准备好一次传输请求      │
     │                                          │
 4.  │ post_xfer_req()  ── 异步发起 ──────────▶  │ (数据被单边写入,
     │   立刻返回,不阻塞                         │  Decode CPU 不介入)
     │                                          │
 5.  │ get_xfer_status() 轮询/等待完成            │
     │   == DONE ?                               │
     │      │ 是                                 │
 6.  │ 通知 Decode："KV 已就位,开始 decode"       │──▶ 续写 token
```

要点：
- **第 1-2 步是一次性的**（注册 + 元数据交换），可在连接建立时做好。
- **第 4 步是单边（one-sided）写**：Prefill 直接把数据写进 Decode 的显存，**不需要 Decode 的 CPU 参与收包**——这正是 RDMA write 的能力，省掉一次 CPU 中断和拷贝。
- **第 5 步异步**：发起后不阻塞计算线程，可与下一个 batch 的 prefill 重叠（overlap），把传输时间"藏"进计算里。

### 4.1 为什么"单边 + 异步 + 批量"是三大法宝

```
同步阻塞:   compute ──▶ [等KV搬完····] ──▶ next   （传输全暴露,GPU空转）
异步重叠:   compute ──▶ next compute ─────▶
                     └─KV搬运后台进行──┘            （传输被计算掩盖,≈免费）
```

---

## 5. 后端插件细看：UCX / GDS / GPUDirect

### 5.1 UCX 后端（网络与本机互联的主力）

UCX（Unified Communication X）是一个成熟的高性能通信框架，自动选择 InfiniBand verbs / RoCE / TCP / 共享内存 / CUDA IPC。NIXL 把它包成后端，于是：

- 同机两 GPU：UCX 走 **NVLink / CUDA IPC**（不出网卡）。
- 跨节点：UCX 走 **RDMA（IB 或 RoCE）**，绕过 CPU 和内核协议栈（kernel bypass），实现 **GPUDirect RDMA**——网卡直接读写 GPU 显存，数据不落主机内存。

```
跨节点 GPUDirect RDMA 路径（无 CPU 拷贝）:
[Prefill GPU HBM] ──▶ 网卡(NIC) ══网线══ 网卡 ──▶ [Decode GPU HBM]
       └────────── 全程不经过主机 DRAM ──────────┘
```

详见 [[ai-infra/ai-hardware/GPU-network]]。

### 5.2 GDS 后端（GPU ↔ SSD，KV 卸载/复用）

GPUDirect Storage（GDS，`cuFile` API）让 NVMe SSD 直接 DMA 进出 GPU 显存，**不经过主机内存中转**。用途：

- **KV 卸载**：显存不够时把不活跃请求的 KV 暂存到 SSD。
- **前缀缓存（prefix cache）落盘**：把热门 prompt 的 KV 存盘，命中即从盘读回（与 [[llm-inference/Mooncake]] 的 KVCache 池思路一致）。

```
传统(慢): SSD ──▶ 主机DRAM ──▶ (cudaMemcpy) ──▶ GPU HBM   (两跳+CPU)
GDS  (快): SSD ════════════ DMA ════════════▶ GPU HBM     (一跳,绕开CPU)
```

### 5.3 选谁？由"源/目的位置组合"决定

| 源 → 目的 | NIXL 选用 |
|-----------|----------|
| VRAM → 同机 VRAM | UCX(NVLink/IPC) |
| VRAM → 远端 VRAM | UCX(RDMA, GPUDirect) |
| VRAM → 本地 NVMe | GDS(cuFile) |
| VRAM → DRAM | UCX / cudaMemcpy |

---

## 6. 与 Dynamo / PD 分离架构的关系

**NVIDIA Dynamo** 是面向大规模分布式推理的服务框架（可理解为"推理界的 Kubernetes 调度层"）。它负责：请求路由、PD 分离编排、KV 感知调度、弹性扩缩。**NIXL 是 Dynamo 的传输地基**。

```
            ┌──────────────────────────────────────┐
            │            Dynamo  (编排层)            │
            │  路由 / PD 调度 / KV-aware 路由 / 扩缩  │
            └──────────────────────────────────────┘
                          │ 调用
            ┌──────────────────────────────────────┐
            │         NIXL  (统一传输层)             │
            │  register / transfer / 后端自动选择     │
            └──────────────────────────────────────┘
              │ UCX        │ GDS         │ ...
        ┌─────▼────┐ ┌─────▼─────┐ ┌─────▼────┐
        │RDMA/NVLink│ │NVMe(cuFile)│ │DRAM/文件 │
        └──────────┘ └───────────┘ └──────────┘
```

Dynamo 决定"**谁该把 KV 搬给谁**"（哪台 Prefill 的结果送到哪台 Decode、是否命中 KV 池），NIXL 负责"**把这次搬运高效执行掉**"。两者分工：**Dynamo 是大脑（决策），NIXL 是肌肉（执行）**。

vLLM、SGLang、TensorRT-LLM 也可以直接接 NIXL 做 PD 分离的 KV 传输，不一定要整套 Dynamo。

---

## 7. 与 Mooncake / 同类对比

| 维度 | NIXL | Mooncake Transfer Engine | NCCL |
|------|------|--------------------------|------|
| 定位 | 推理统一传输库 | 推理 KV 传输引擎 + 全局 KV 池 | 训练集合通信 |
| 通信模式 | 点对点、单边、异步 | 点对点、单边 | 集合(all-reduce 等)、同步 |
| 介质 | GPU/DRAM/NVMe/网络 全异构 | GPU/DRAM/SSD/RDMA | 主要 GPU↔GPU |
| 后端 | 插件式(UCX/GDS/...) | 自研 + RDMA | 自研集合算法 |
| 生态 | NVIDIA Dynamo | Kimi/月之暗面，vLLM 集成 | 训练全栈 |
| 关系 | 与 Mooncake **思路相通、可互补/竞争** | 见 [[llm-inference/Mooncake]] | 见 [[ai-infra/ai-hardware/GPU-network]] |

要点：NIXL 与 Mooncake 解决的是**同一类问题**（PD 分离/全局 KV 池的高效传输），NIXL 更偏"NVIDIA 官方、后端插件化、与 Dynamo 深绑"，Mooncake 更偏"算法系统一体、含全局调度与池化策略"。两者都强调 **one-sided RDMA + 异步 + 批量 descriptor**。详见 [[llm-inference/Mooncake]]。

---

## 8. 数值例子 + 实践要点

### 8.1 手算：搬一个长 prompt 的 KV 要传多少、要多久

设：Llama-3-70B 量级，$L=80$ 层，GQA 下 $n_{kv}=8$ 头、$d_{head}=128$，FP16（$b=2$），prompt 长度 $S=4096$。

每 token 每层 KV（K+V）字节：
$$
2 \times n_{kv} \times d_{head} \times b = 2 \times 8 \times 128 \times 2 = 4096 \text{ B} = 4\text{ KB}
$$

整段 prompt 的 KV：
$$
\text{KV} = 4\text{KB} \times L \times S = 4\text{KB} \times 80 \times 4096 \approx 1.28\ \text{GB}
$$

> 约值，随并行/量化/实现而变，精确以模型配置为准。

在不同通道搬这 1.28 GB（取**典型公开峰值带宽**，实际打折）：

| 通道 | 典型带宽(约) | 搬 1.28GB 耗时(理想) |
|------|-------------|---------------------|
| NVLink (同机) | ~900 GB/s（NVLink4 总线，约/以官方为准） | ~1.4 ms |
| RDMA 400Gb 网卡 | ~50 GB/s（400 Gbps≈50 GB/s） | ~26 ms |
| RDMA 200Gb 网卡 | ~25 GB/s | ~51 ms |
| PCIe Gen5 x16 | ~64 GB/s（约） | ~20 ms |

公式：$\text{时间} = \dfrac{\text{数据量}}{\text{带宽}\times \text{效率}}$，效率（实际/峰值）常在 0.6~0.9。

**洞察 1**：跨节点 RDMA 搬一次 KV 是 **几十毫秒级**，而 Decode 生成一个 token 也就**十几到几十毫秒**。所以传输若**不与计算重叠**，会直接吃掉一个 token 的时间——这正是 NIXL 坚持**异步 + 重叠**的原因。

**洞察 2**：同机 NVLink 比跨节点 RDMA 快约 **15~30 倍**。所以 Dynamo 的 KV-aware 调度会**尽量把 Decode 排到与 Prefill 同机/同 NVLink 域**，搬不动时才走网络——NIXL 让"同机走 NVLink、跨机走 RDMA"对上层透明。

### 8.2 注册开销 vs 复用

注册（pin + 建映射）可能是**毫秒级**一次性成本。若每次传输都注册一块新内存：

```
坏: register(1.4ms) + transfer(26ms) + dereg  每次都付注册税
好: 启动时 register 一个大 KV 池 → 之后 transfer 只付传输成本
```

**实践**：把 KV Cache 分配成**预注册的大池**（与 PagedAttention 的 block 池天然契合），传输时只提交 descriptor list，不重复注册。

### 8.3 实践要点清单

- **批量提交**：把同一请求的几十上百个 KV block 合成**一个** descriptor list 传输，别一块一块发。
- **重叠传输与计算**：post 后立刻返回，让 Prefill 继续算下一个 batch，传输在后台跑。
- **拓扑感知放置**：靠上层（Dynamo）尽量同 NVLink 域配对 PD，减少跨网传输。
- **网络要 RoCE/IB + GPUDirect RDMA**：否则数据要落主机内存中转，带宽腰斩、延迟翻倍（见 [[ai-infra/ai-hardware/GPU-network]]）。
- **版本与默认值以官方为准**：NIXL 仍在快速迭代，后端列表、API 名称、默认参数请查 `ai-dynamo/nixl` 文档，勿背具体版本号。

---

## 9. 常见问题

| 问题 | 答 |
|------|----|
| NIXL 和 NCCL 能互相替代吗？ | 不能。NCCL 做训练的对称集合通信；NIXL 做推理的点对点单边搬运。语义不同。 |
| 没有 RDMA 网卡能用吗？ | 能，UCX 会回退到 TCP，但带宽/延迟差很多，PD 分离收益大打折扣。 |
| 一定要配 Dynamo 吗？ | 不必。NIXL 可被 vLLM/SGLang/自研框架直接调用做 KV 传输；Dynamo 只是其上的编排层。 |
| 它只搬 KV Cache？ | 不。任何已注册内存段都能搬（权重、激活、KV 池数据等），KV 是头号场景。 |
| 为什么强调"单边"？ | 单边(RDMA write/read)发起方直接读写对端内存，对端 CPU 不介入，省中断省拷贝，延迟更低。 |
| 和 Mooncake 选哪个？ | 看生态：NVIDIA 栈/Dynamo 选 NIXL；已用 Mooncake KV 池/月之暗面方案则 Mooncake。思路相通，见 [[llm-inference/Mooncake]]。 |
| KV 卸载到 SSD 走什么？ | GDS 后端（GPUDirect Storage / cuFile），SSD 直接 DMA 进出 GPU 显存。 |
| 上层要关心走 NVLink 还是 RDMA 吗？ | 不用。NIXL 按源/目的位置自动选后端，对上层透明。 |

---

## 🔗 跳转链接

- [[00-知识地图]] —— 全局索引
- [[llm-inference/PD分离]] —— NIXL 的头号应用场景：Prefill/Decode 分离
- [[llm-inference/Mooncake]] —— 同类 KV 传输引擎 + 全局 KV 池，思路对照
- [[ai-infra/ai-hardware/GPU-network]] —— RDMA / NVLink / GPUDirect 的物理底座
- 源码与文档（以官方为准）：https://github.com/ai-dynamo/nixl
