# HCCL 集合通信库（华为昇腾）

> HCCL（Huawei Collective Communication Library）是昇腾平台上的集合通信库，对标英伟达 NCCL，负责多 NPU / 多机之间 AllReduce、AllGather 等通信原语的高性能实现。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]] [[ai-infra/ai-hardware/CUDA]]

## 阅读地图

| 章节 | 内容 | 你会得到 |
| --- | --- | --- |
| 0 | 一句话锚点 | 30 秒抓住 HCCL 是什么 |
| 1 | 在昇腾栈的定位 + 对标英伟达 | NCCL→HCCL 的迁移心智图 |
| 2 | 集合通信原语回顾 | AllReduce / AllGather / ReduceScatter 等 |
| 3 | HCCL 的两套接口（Python / C++ 单算子） | TF 图模式 vs PyTorch 后端嵌入 |
| 4 | 通信域、Rank、RankTable | 谁和谁通信、怎么编排 |
| 5 | 环算法与拓扑（Ring / 双环 / Mesh / HCCS） | 为什么这样设计、带宽从哪来 |
| 6 | 数据流：从张量到链路 | 一次 AllReduce 在硬件上发生了什么 |
| 7 | 在并行训练中的角色 | DP / TP / PP / EP 各用到哪些原语 |
| 迁移 | 从 NCCL 迁到 HCCL | 要改什么、易踩的坑 |
| FAQ | 常见问题 | 排错速查 |

## 0. 一句话锚点

**HCCL 之于昇腾，等同于 NCCL 之于英伟达。** 当你在 GPU 上写 `dist.all_reduce(tensor)`、底层走 NCCL；在昇腾 NPU 上同样写 `dist.all_reduce(tensor)`、底层就走 HCCL。它把"多张加速卡如何协同把梯度/激活同步起来"这件事，封装成一组高性能、拓扑感知的集合通信原语。

> 记忆钩子：**H**uawei **C**ollective **C**ommunication **L**ibrary。把 NCCL 的 N（NVIDIA）换成 HC（Huawei），其余心智模型几乎一一对应。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层

```
┌──────────────────────────────────────────────────────────┐
│  套件层   MindFormers / ModelLink / MindIE / msmodelslim   │
├──────────────────────────────────────────────────────────┤
│  框架层   MindSpore  /  PyTorch(torch_npu) / TensorFlow    │
├──────────────────────────────────────────────────────────┤
│  CANN 层  ┌────────────┬───────────┬──────────────────┐    │
│           │ 算子库      │  Runtime  │   ★ HCCL ★        │    │
│           │(类 cuDNN)  │(类 CUDART)│ (集合通信，类NCCL)│    │
│           └────────────┴───────────┴──────────────────┘    │
├──────────────────────────────────────────────────────────┤
│  驱动/固件  Driver + Firmware                              │
├──────────────────────────────────────────────────────────┤
│  硬件层    昇腾 NPU(达芬奇架构) + HCCS / RoCE 互联          │
└──────────────────────────────────────────────────────────┘
```

HCCL 处在 **CANN 层**，与算子库、Runtime 并列，是 CANN 对外提供"分布式通信能力"的那一块。它向上被框架（PyTorch 的 `torch_npu`、MindSpore）和套件（MindFormers、ModelLink）调用，向下依赖驱动把数据真正搬到 HCCS / RoCE 链路上。

### 1.2 昇腾 ↔ 英伟达生态对照表（迁移心智图）

| 维度 | 英伟达世界 | 昇腾世界 | 说明 |
| --- | --- | --- | --- |
| 加速器 | GPU | NPU（达芬奇架构） | 都是训练/推理加速卡 |
| 编程平台 | CUDA | CANN | 软件栈总称 |
| 运行时 | CUDA Runtime / Driver | Runtime / Driver | 设备管理、流、内存 |
| 算子库 | cuDNN / cuBLAS | CANN 算子库 | 卷积/矩阵乘等高性能算子 |
| **集合通信** | **NCCL** | **★ HCCL ★** | **本文主角** |
| 卡间高速互联 | NVLink / NVSwitch | HCCS | 同节点多卡直连 |
| 跨节点网络 | InfiniBand / RoCE | RoCE / IB | 多机互联 |
| 深度学习框架 | PyTorch / TensorFlow | MindSpore / PyTorch(torch_npu) | 昇腾上 PyTorch 需 torch_npu 适配层 |
| 训练大模型套件 | Megatron-LM | MindFormers / ModelLink | 并行训练 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 部署 |
| 量化工具 | GPTQ / AWQ | msmodelslim | 压缩 |
| 分布式后端名 | `nccl` | `hccl` | `init_process_group` 里填的 backend |

**一句话迁移结论**：原来 `torch.distributed.init_process_group(backend="nccl")`，到昇腾上改成 `backend="hccl"`（依赖 torch_npu）。上层并行逻辑大体不变，变的是后端实现和底层链路。具体接口名与参数以华为昇腾官方文档（Ascend 社区）为准。

## 2. 集合通信原语回顾

HCCL 实现的就是这些"大家耳熟能详"的原语，语义与 MPI / NCCL 一致：

| 原语 | 含义 | 典型用途 |
| --- | --- | --- |
| AllReduce | 所有 rank 的数据规约（求和等）后，结果广播回所有 rank | 数据并行梯度同步 |
| AllGather | 每个 rank 把自己的分片收集到所有 rank | 张量并行/ZeRO 参数聚合 |
| ReduceScatter | 规约后按分片散发到各 rank | ZeRO 梯度、TP 反向 |
| Broadcast | 一个 rank 把数据广播给所有 rank | 参数初始化分发 |
| Reduce | 规约结果只汇聚到一个 rank | 指标汇总 |
| AllToAll | 每个 rank 向其他每个 rank 各发一份分片 | MoE 专家并行（EP）路由 |
| Send / Recv | 点对点收发 | 流水线并行（PP）相邻 stage 传激活 |

> 关键等式：**AllReduce ≈ ReduceScatter + AllGather**。理解这条，就能理解 Ring-AllReduce 为什么能把通信量做到与卡数近似无关。

## 3. HCCL 的两套接口：Python 图模式 vs C++ 单算子嵌入

这是原文件已点明、也是最核心的一点：**HCCL 提供 Python 与 C++ 两套语言接口，对应两种使用范式。**

```
         ┌──────────────────────────────────────────┐
         │              使用 HCCL 的两条路           │
         └──────────────────────────────────────────┘
                 │                          │
    ┌────────────▼───────────┐   ┌──────────▼─────────────────┐
    │  ① Python 接口          │   │  ② C++ 单算子 API           │
    │  (图模式 / TensorFlow)  │   │  (OPBase 模式 / 框架适配)   │
    ├────────────────────────┤   ├────────────────────────────┤
    │ 把通信编进计算图，整图  │   │ 把 HCCL 单算子嵌入框架后端  │
    │ 在 NPU 上分布式执行优化 │   │ 例：嵌入 PyTorch 后端代码   │
    │                         │   │ 用户照常调 PyTorch 原生     │
    │ 适合 TF 静态图训练      │   │ 集合通信 API 即获分布式能力 │
    └─────────────────────────┘  └────────────────────────────┘
```

- **① Python 接口（图模式）**：面向 TensorFlow 等静态图框架。通信操作被编织进整张计算图，由图编译器整体优化后在昇腾 NPU 上分布式执行。优点是编译期可做通信-计算融合/重排。
- **② C++ 单算子 API（OPBase / 框架适配）**：把 HCCL 的单个通信算子（如 AllReduce 算子）嵌入到框架后端代码里。这正是 **PyTorch on 昇腾** 的做法——`torch_npu` 在 PyTorch 后端把集合通信对接到 HCCL 单算子 API，于是用户只需照常调用 PyTorch 原生的 `torch.distributed` 接口，无需感知底层是 HCCL。

> 迁移启示：PyTorch 用户几乎"零感知"——你写的还是 `dist.all_reduce`，只是 `backend="hccl"`，由 torch_npu 把调用翻译成 HCCL 单算子。这正是②的价值：**保留上层代码习惯，替换底层通信实现。**

## 4. 通信域、Rank 与 RankTable：谁和谁通信

集合通信要先回答"哪些卡组成一个通信组、各自编号是多少、走哪条链路"。

| 概念 | 含义 | NCCL 类比 |
| --- | --- | --- |
| 通信域(Communicator) | 一组参与同一次集合通信的设备集合 | NCCL communicator |
| Rank | 设备在通信域内的全局编号（0..N-1） | NCCL rank |
| Local Rank | 设备在本节点内的编号 | local rank |
| RankTable | 描述集群拓扑（每个 rank 的设备、IP、所在 server）的配置 | NCCL 通过 env/bootstrap |
| World Size | 通信域内设备总数 | world size |

```
   ┌─────────── Server 0 ───────────┐   ┌─────────── Server 1 ───────────┐
   │ NPU0   NPU1   NPU2   NPU3       │   │ NPU0   NPU1   NPU2   NPU3       │
   │ rank0  rank1  rank2  rank3      │   │ rank4  rank5  rank6  rank7      │
   │   └──HCCS 全互联(节点内)──┘     │   │   └──HCCS 全互联(节点内)──┘     │
   └──────────────┬─────────────────┘   └──────────────┬─────────────────┘
                  └─────────── RoCE / IB(节点间) ───────┘
                       一个 world_size=8 的通信域
```

RankTable（或等价的拓扑/环境配置）告诉 HCCL：这 8 个 rank 分布在哪些 server、哪几个走 HCCS（节点内高带宽）、哪几个跨节点走 RoCE。HCCL 据此选择合适的通信算法与链路。RankTable 的确切字段、生成方式与环境变量以华为昇腾官方文档（Ascend 社区）为准。

## 5. 环算法与拓扑：性能从哪来

HCCL 与 NCCL 思路一致：**拓扑感知地选择通信算法**，让数据尽量走高带宽链路、并把通信量摊到环上。

### 5.1 Ring-AllReduce（最经典）

把 N 个 rank 连成一个逻辑环，AllReduce 拆成两个阶段，各 N-1 步：

```
  Phase 1: ReduceScatter（边传边加，每步把一个分片传给下一跳并累加）
  Phase 2: AllGather（把累加好的各分片沿环传一圈，人人集齐）

      rank0 ──► rank1 ──► rank2 ──► rank3
        ▲                            │
        └────────────────────────────┘   (逻辑环)
```

每个 rank 任一时刻只与左右邻居通信，**单卡通信量 ≈ 2·(N-1)/N · 数据量，与卡数 N 几乎无关**，带宽利用率高。这是大规模数据并行的基石。

### 5.2 拓扑分层：节点内 HCCS + 节点间 RoCE

多机场景下，HCCL 通常做**两级/分层算法**：先在节点内沿 HCCS 高速环规约，再在节点间走 RoCE 规约，最后回灌。原理是"把跨节点这段昂贵链路上的数据量压到最小"，与 NCCL 的 tree/ring 分层思想一致。

| 算法/拓扑 | 适用场景 | 直觉 |
| --- | --- | --- |
| Ring（单环） | 中小规模、带宽敏感 | 摊平通信量 |
| 双环 / Mesh | 同节点多卡全互联(HCCS) | 充分利用全互联带宽 |
| 分层(节点内+节点间) | 多机 | 减少昂贵跨节点流量 |

> 具体在何种规模/数据量下走哪种算法，由 HCCL 内部按拓扑与消息大小自动选择，调优旋钮以官方文档为准。

## 6. 数据流：一次 AllReduce 在硬件上发生了什么

```
框架层    PyTorch: loss.backward() 产生各卡梯度
            │  dist.all_reduce(grad)   (backend=hccl)
            ▼
适配层    torch_npu 把调用翻译为 HCCL 单算子 API (C++ 接口②)
            │
            ▼
CANN/HCCL  根据通信域 + RankTable 选算法(Ring/分层)
            │  下发通信任务到设备执行流(Stream)
            ▼
驱动/硬件   节点内走 HCCS、节点间走 RoCE，沿环规约+回灌
            │
            ▼
框架层    每张卡拿到"同步后的平均梯度" → optimizer.step()
```

要点：通信被下发到 NPU 的**执行流（Stream）**上，可与计算算子流水重叠（通信-计算 overlap）。这与 CUDA 上"用独立 stream 跑 NCCL、与计算 kernel 并发"是同一套机制。

## 7. 在并行训练中的角色

大模型训练的各种并行，本质都是不同集合通信原语的组合，HCCL 全部承接：

| 并行方式 | 主要通信原语 | 通信发生时机 |
| --- | --- | --- |
| 数据并行 DP / ZeRO | AllReduce 或 ReduceScatter+AllGather | 梯度/参数同步 |
| 张量并行 TP | AllReduce / AllGather | 每层前向反向的激活切分聚合 |
| 流水线并行 PP | Send / Recv | 相邻 stage 传激活/梯度 |
| 专家并行 EP（MoE） | AllToAll | token 路由到不同专家 |

这也解释了为什么在昇腾上跑 MindFormers / ModelLink（对标 Megatron）时，HCCL 的健康与调优如此关键——它是所有并行维度的"通信底座"。

## 迁移要点 / 注意事项与坑

**从 NCCL 迁到 HCCL，要改什么：**

1. **后端名**：`init_process_group(backend="nccl")` → `backend="hccl"`，并确保安装了与 CANN 配套的 `torch_npu`。
2. **设备 API**：`torch.cuda.*` → 昇腾对应的设备接口（经 torch_npu）；`.cuda()` → 对应的 NPU 设备搬运。具体接口名以官方文档为准。
3. **集群配置**：NCCL 多走环境变量/bootstrap，HCCL 多依赖 **RankTable / 拓扑配置**描述集群，需正确生成并下发。
4. **环境变量**：HCCL 有自己的一套调优/超时/算法选择环境变量（与 `NCCL_*` 对应但名字不同），具体变量名以官方文档为准。

**常见坑（机制层面）：**

- **RankTable 与实际拓扑不符**：rank↔设备↔IP 映射写错，导致建链失败或性能塌方——这是最高频的坑。
- **节点内未走 HCCS**：拓扑配置不当使本应走 HCCS 高带宽的流量退化到低速链路，吞吐骤降。
- **通信-计算未重叠**：未让通信下发到独立流，串行执行使 NPU 算力空等。
- **超时/挂死**：某个 rank 卡住（数据加载慢、OOM）会让整个集合通信 hang，需结合超时设置与日志定位是哪个 rank。
- **CANN / torch_npu / 驱动版本不匹配**：三者需配套，错配会出现通信算子不可用。**具体配套关系与版本以华为昇腾官方文档（Ascend 社区）为准。**
- **框架范式选错**：TF 静态图用 Python 图模式接口、PyTorch 用 C++ 单算子嵌入，混用会导致接不上。

> 安装、镜像、版本、确切环境变量名一律以华为昇腾官方文档（Ascend 社区）为准——本文只讲清"每一步为什么要这么做"。

## 常见问题

| 问题 | 简答 |
| --- | --- |
| HCCL 对标 CUDA 世界的谁？ | NCCL。语义/原语一一对应，差别在底层实现与链路（HCCS/RoCE）。 |
| PyTorch 用户要重写通信代码吗？ | 基本不用。torch_npu 把 `dist.*` 翻译成 HCCL 单算子，改 `backend="hccl"` 即可。 |
| Python 接口和 C++ 接口什么区别？ | Python 接口面向 TF 图模式整图优化；C++ 单算子 API 用于框架后端适配（PyTorch 走这条）。 |
| 为什么要 RankTable？ | 它描述集群拓扑（rank↔设备↔server↔链路），HCCL 据此选算法和链路。 |
| 节点内/节点间链路分别是什么？ | 节点内 HCCS（类比 NVLink），节点间 RoCE/IB（类比 IB）。 |
| Ring-AllReduce 为什么高效？ | 单卡通信量与卡数近似无关，带宽利用率高；AllReduce=ReduceScatter+AllGather。 |
| MoE 训练靠哪个原语？ | AllToAll，做专家并行的 token 路由。 |
| 调优旋钮在哪？ | HCCL 环境变量 + 拓扑配置，具体名以官方文档为准。 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-infra/算力/昇腾NPU]]
- [[ai-infra/ai-hardware/AI芯片软件生态]]
- [[ai-infra/ai-hardware/CUDA]]
- [[ai-infra/网络/NCCL]]
- [[ai-infra/网络/集合通信原语]]
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/huggingface-transformers/README]]
- [[llm-compression/quantization/量化基础]]
- [[llm-inference/README]]
- [[llm-train/README]]
- [[llm-algo/transformer/模型架构]]

> 参考：HCCL API 参考（华为昇腾文档，CANN 社区版）— 具体接口、版本、命令以官方文档为准。
