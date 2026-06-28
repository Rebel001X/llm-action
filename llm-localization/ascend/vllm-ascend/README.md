# vLLM-Ascend(vLLM 昇腾 NPU 后端)

> 把社区最流行的高吞吐推理引擎 vLLM「插」到昇腾 NPU 上跑的官方硬件后端,让你几乎不改业务代码就能在国产卡上享受 PagedAttention + 连续批处理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/网络/NCCL]] [[llm-inference/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | vLLM 的 NPU plugin |
| 1 | 在昇腾栈里的定位 + 昇腾↔英伟达对照表 | CANN / Platform Plugin |
| 2 | vLLM 核心机制为什么能跨硬件 | PagedAttention / 连续批处理 |
| 3 | vLLM-Ascend 的插件化架构(ASCII 图) | Platform / Worker / Attention Backend |
| 4 | 图模式 vs 单算子模式 | torch_npu / ACL Graph |
| 5 | 张量并行与 HCCL 通信 | TP / HCCL ↔ NCCL |
| 6 | 部署流程的含义(不背命令) | 镜像 / CANN / 依赖链 |
| 迁移 | 从 CUDA vLLM 迁到昇腾要改什么、坑在哪 | device=npu / 量化 / 特性差异 |
| FAQ | 常见问题表 | — |

## 0. 一句话锚点

**vLLM-Ascend = vLLM 的「昇腾 NPU 硬件插件(out-of-tree platform plugin)」。** 上游 vLLM 把「设备相关」的部分抽象成插件接口,vLLM-Ascend 这个独立仓库实现了昇腾版的 Platform、Worker、Attention 后端和通信后端;于是你用的还是同一套 `vllm` API / OpenAI 兼容 Server,底层算力却换成了昇腾 NPU。它**不是**重写一个推理引擎,而是「让 vLLM 认识昇腾卡」。

> 类比:就像同一份 PyTorch 训练脚本,装上 `torch_npu` 后就能在昇腾上跑——vLLM-Ascend 之于 vLLM,等价于 `torch_npu` 之于 PyTorch。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 它在昇腾软件栈的哪一层

昇腾的软件栈自底向上大致是:**硬件(达芬奇架构 NPU)→ CANN(驱动+运行时+算子库,对标 CUDA)→ 框架层(PyTorch+torch_npu / MindSpore)→ 推理服务层(vLLM-Ascend / MindIE)**。vLLM-Ascend 坐在**最上面的服务层**,向下依赖 torch_npu 和 CANN,向上对业务暴露 vLLM 原生接口。

```
┌──────────────────────────────────────────────┐
│  业务 / OpenAI 兼容 API（offline & online）   │  ← 你写的代码几乎不变
├──────────────────────────────────────────────┤
│  vLLM 引擎核心（调度/PagedAttention/批处理）  │  ← 上游 vLLM，硬件无关
├──────────────────────────────────────────────┤
│  vLLM-Ascend 插件（Platform/Worker/Backend）  │  ← 本仓库，硬件相关
├──────────────────────────────────────────────┤
│  torch_npu（PyTorch 的昇腾适配层）            │
├──────────────────────────────────────────────┤
│  CANN（AscendCL 运行时 + 算子库 + HCCL）      │  ← 对标 CUDA Runtime+cuDNN+NCCL
├──────────────────────────────────────────────┤
│  驱动 + 固件 + 昇腾 NPU（达芬奇 Cube/Vector） │  ← 对标 GPU 驱动 + SM
└──────────────────────────────────────────────┘
```

### 1.2 昇腾 ↔ 英伟达 生态对照表(迁移心智图)

| 维度 | 英伟达世界 | 昇腾世界 | 一句话说明 |
|------|-----------|----------|-----------|
| 加速芯片 | GPU(SM/Tensor Core) | NPU(达芬奇 Cube/Vector 单元) | 都是矩阵加速硬件,微架构不同 |
| 底层软件栈 | CUDA Runtime/Driver | CANN + AscendCL | 运行时 + 设备管理 |
| 算子/数学库 | cuDNN / cuBLAS | CANN 算子库(AOL/aclnn) | 高性能算子集合 |
| 框架适配 | PyTorch 原生 CUDA | PyTorch + `torch_npu` | torch_npu 把 `cuda` 语义映射到 `npu` |
| 集合通信 | NCCL | HCCL | AllReduce/AllGather 等多卡通信 |
| **推理引擎(本篇)** | **vLLM(CUDA 后端)** | **vLLM-Ascend(NPU 后端)** | 同一引擎,换硬件插件 |
| 厂商自研推理引擎 | TensorRT-LLM | MindIE(LLM) | 厂商深度优化的闭环方案 |
| 训练套件 | Megatron-LM | MindFormers / ModelLink | 大模型并行训练 |
| 量化工具 | GPTQ/AWQ 工具链 | msModelSlim(昇腾量化) | 权重/激活量化 |
| 设备标识 | `cuda:0` | `npu:0` | 代码里最直观的差异 |

> 关键洞察:vLLM-Ascend 与 **MindIE** 是**两条不同路线**。vLLM-Ascend 走「社区引擎 + 昇腾后端」,生态/特性跟随上游 vLLM,迁移成本极低;MindIE 走「华为自研引擎」,在昇腾上往往能做更深的图融合与性能榨取。选型时:要复用 vLLM 生态/快速迁移 → vLLM-Ascend;要极致性能/官方深度支持 → MindIE。

## 2. 先回顾:vLLM 凭什么能跨硬件

vLLM 之所以快,核心是两件事——这两件事**本身与硬件无关**,正是它能被「插件化移植」的前提:

- **PagedAttention**:把 KV Cache 像操作系统分页一样切成固定大小的 block,按需分配、可共享(如多个并发请求共享同一前缀)。它解决了显存碎片与浪费,是 vLLM 高吞吐的地基。
- **连续批处理(Continuous Batching)**:不是等一个 batch 全部生成完再换,而是 token 级别地动态把新请求填进空位、把完成的请求踢出去,让算力时刻饱和。

引擎的**调度逻辑、KV block 管理、批处理策略**都在上游 vLLM 里;**真正落到硬件上的算子(Attention 计算、矩阵乘、KV 读写)**才是插件要替换的部分。这就是「核心逻辑共享、设备实现可插拔」的设计。

## 3. vLLM-Ascend 的插件化架构

vLLM 上游定义了几个「硬件可替换点」,vLLM-Ascend 逐一给出昇腾实现:

```
            vLLM 引擎核心（硬件无关）
                     │
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
 ┌──────────┐  ┌──────────┐  ┌──────────────┐
 │ Platform │  │  Worker  │  │  Attention   │
 │ (Ascend) │  │ (Ascend) │  │  Backend     │
 │ 设备探测/ │  │ 单卡执行/ │  │ (Ascend)     │
 │ 能力声明  │  │ 显存管理  │  │ PagedAttn→NPU │
 └────┬─────┘  └────┬─────┘  └──────┬───────┘
      │             │               │
      └──────┬──────┴───────┬───────┘
             ▼              ▼
        ┌─────────┐   ┌──────────────┐
        │torch_npu│   │ Communicator │
        │ 算子下发 │   │ (HCCL 多卡)   │
        └────┬────┘   └──────┬───────┘
             ▼               ▼
        ┌────────────────────────────┐
        │   CANN（算子库 + HCCL）     │
        └────────────────────────────┘
```

各组件职责(机制层面理解,不背 API):

| 插件点 | CUDA 版做什么 | 昇腾版怎么实现 |
|--------|-------------|--------------|
| Platform | 探测 GPU、声明数据类型/能力 | 探测 NPU 设备数、声明昇腾支持的 dtype/特性 |
| Worker | 在一张卡上跑模型、管 KV 显存 | 在一张 NPU 上跑,走 torch_npu 分配 NPU 内存 |
| Model Runner | 组织前向、捕获计算图 | 调用昇腾算子,支持图模式捕获 |
| Attention Backend | 用 CUDA kernel 算 PagedAttention | 用昇腾融合算子算 PagedAttention |
| Communicator | 走 NCCL | 走 **HCCL** |

> 这就是为什么「上层代码不变」:你 import 的还是 `vllm`,但当 vLLM 发现安装了 ascend 插件且设备是 NPU 时,会通过插件机制把这些可替换点全部换成昇腾实现。

## 4. 图模式 vs 单算子模式(性能关键开关)

昇腾上执行有两种典型方式,理解它对调优至关重要:

- **单算子(Eager)模式**:一个算子一个算子地下发到 NPU,像 PyTorch eager。灵活、好调试,但每次下发都有 Host 侧开销,小 shape 下「下发跟不上算」会让 NPU 空等。
- **图模式(Graph,基于 ACL Graph / torchair 等)**:把一段计算「录制」成图整体下发执行,大幅减少 Host-Device 交互开销,类似 CUDA Graph。decode 阶段(每步只生成一个 token、算子粒度小)从图模式获益尤其明显。

```
单算子模式:  Host ──下发op1──▶ NPU
            Host ──下发op2──▶ NPU   ← 每步都有下发开销，NPU 易空等
            Host ──下发op3──▶ NPU

图模式:      Host ──下发整张图──▶ NPU 连续执行 op1→op2→op3
                                   ← 一次下发，NPU 持续忙，吞吐↑
```

> 调优心智:**decode-heavy / 小 batch 场景优先尝试图模式**;遇到不支持的算子或动态 shape 时再回退单算子。是否启用、如何启用以华为昇腾官方文档(Ascend 社区)与 vLLM-Ascend 文档为准。

## 5. 张量并行与 HCCL 通信

单卡放不下的大模型靠**张量并行(TP)**切到多张 NPU 上,每层前向后需要 AllReduce 汇总——这步在昇腾上由 **HCCL** 完成(对标 CUDA 世界的 NCCL)。

```
          TP=4 的一层前向（昇腾 4 张 NPU）
   NPU0      NPU1      NPU2      NPU3
   [切片0]   [切片1]   [切片2]   [切片3]
     │         │         │         │
     └────────HCCL AllReduce（环/分层算法）────────┘
                    ▼
              完整激活，进入下一层
```

要点:
- HCCL 与 NCCL 角色对等,但**底层走昇腾互联(如 HCCS / RoCE)**,拓扑与环算法实现是华为自己的。
- 多卡跑需要正确的设备可见性与通信初始化,这部分在 vLLM-Ascend 里被封装进 Communicator;**出问题时优先排查 HCCL 连通性与卡间拓扑**,而不是改 vLLM 逻辑。
- 除 TP 外,数据并行/专家并行(MoE)等更高级并行的支持随版本演进,以官方特性矩阵为准。

## 6. 部署流程的含义(讲依赖,不背命令)

国产化部署最容易在「环境」上翻车。把流程拆成「为什么要这步」,具体命令/版本/镜像标签一律以华为昇腾官方文档(Ascend 社区)与 vLLM-Ascend 文档为准:

1. **驱动 + 固件**:先让操作系统认识 NPU 卡。这是地基,版本必须与后续 CANN 匹配。
2. **CANN**:安装运行时与算子库(对标装 CUDA Toolkit)。**CANN 版本是整条链的「公约数」**,torch_npu / vLLM-Ascend 都要与它对齐。
3. **torch_npu**:PyTorch 的昇腾适配层,版本须与 PyTorch、CANN 三方匹配。
4. **vLLM + vLLM-Ascend**:装上游 vLLM 与对应版本的 ascend 插件,二者版本要**配套**(插件通常对应特定 vLLM 版本)。
5. **(推荐)用官方镜像**:社区提供 vllm-ascend / cann 容器镜像,把上面这条「版本地狱」一次性打包好。生产部署优先用配套镜像,而不是手动逐层装。

> 依赖链口诀:**驱动 ↔ CANN ↔ torch_npu ↔ vLLM-Ascend ↔ vLLM**,任意一环版本错配都可能导致算子找不到、import 失败或运行时报错。**先查官方版本配套表,再动手**。

参考入口(以官方为准):
- vLLM-Ascend 文档(quick_start / 支持特性矩阵)
- vLLM 上游关于 NPU 支持的 issue 讨论
- quay.io 上的 vllm-ascend / cann 镜像仓库
- vLLM 推理输出(reasoning outputs)等特性文档

## 迁移要点 / 注意事项与坑

把一份在 A100 上跑的 vLLM 服务搬到昇腾,典型改动与坑:

| 类别 | 从 CUDA 迁到昇腾要注意 |
|------|----------------------|
| 设备标识 | `cuda` 语义改为 `npu`;但用 vLLM 高层 API 时通常自动识别,不必到处手改 |
| 版本配套 | 头号坑:CANN / torch_npu / vLLM / 插件四方版本必须配套,优先用官方镜像 |
| 特性差异 | 不是所有 vLLM 特性都已在昇腾就绪,**先查「支持特性矩阵」**(如某些量化格式、投机解码、特定 Attention 变体可能滞后) |
| 量化 | CUDA 侧的 GPTQ/AWQ 产物未必能直接用,昇腾侧量化建议走 **msModelSlim**;格式与支持范围以官方为准 |
| 图模式 | decode-heavy 场景开图模式提吞吐,但要确认模型涉及的算子都被图模式支持,否则回退单算子 |
| 多卡通信 | TP 走 HCCL,部署前确认卡间拓扑/HCCL 连通,通信报错先查 HCCL 而非 vLLM |
| 性能预期 | 同代不同架构,吞吐/时延不能简单照搬 GPU 数字,以实测为准(绝不照抄性能数字) |
| 精度对齐 | dtype 与算子实现差异可能带来数值差异,上线前做端到端精度回归 |

**性能调优思路(机制层面)**:① 优先图模式减少下发开销;② 合理设置 KV Cache 占比让 PagedAttention 充分利用 NPU 显存;③ 合适的 TP 度数,避免通信成为瓶颈;④ 关注 batch 组织,让连续批处理把 NPU 喂饱。具体参数与开关以官方文档为准。

## 常见问题

| 问题 | 回答 |
|------|------|
| vLLM-Ascend 是另一个推理引擎吗? | 不是。它是 vLLM 的昇腾硬件插件,引擎核心仍是上游 vLLM,业务代码基本不变。 |
| 它和 MindIE 怎么选? | 要复用 vLLM 生态/低成本迁移选 vLLM-Ascend;要极致性能/华为深度优化选 MindIE。两条路线。 |
| HCCL 是什么,和 NCCL 关系? | 昇腾的集合通信库,角色对标 NCCL,负责多卡 AllReduce 等,底层走昇腾互联。 |
| 为什么部署老报版本/算子错误? | 几乎都是 CANN/torch_npu/vLLM/插件版本错配,优先用官方配套镜像。 |
| CUDA 上的量化模型能直接跑吗? | 不一定。昇腾量化建议走 msModelSlim,支持格式以官方为准。 |
| 图模式一定更快吗? | decode-heavy/小 batch 通常更快;遇不支持算子或动态 shape 可能要回退单算子。 |
| 具体安装命令/版本号去哪查? | 一律以华为昇腾官方文档(Ascend 社区)与 vLLM-Ascend 文档为准,本文不给确切命令/版本。 |

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
