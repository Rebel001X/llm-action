# MindSpore 昇腾深度学习框架

> 华为自研的全场景 AI 框架，对标 PyTorch/TensorFlow，是昇腾软件栈中"框架层"的国产主力。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/huggingface-transformers/README]]

## 阅读地图

| 章节 | 你会得到什么 | 一句话 |
| --- | --- | --- |
| 0. 一句话锚点 | 最快理解 MindSpore 是什么 | 昇腾上的 PyTorch |
| 1. 地基:栈中定位 + 对标英伟达 | 迁移心智图 | MindSpore↔PyTorch、CANN↔CUDA |
| 2. 两种执行模式 | Graph vs PyNative 的取舍 | 静态图快、动态图好调 |
| 3. 计算图与图下沉机制 | 为什么图模式在 NPU 上快 | 整图下沉减少 Host-Device 交互 |
| 4. 自动微分与自动并行 | 训练超大模型的杀手锏 | 函数式微分 + 多维混合并行 |
| 5. 与 CANN/达芬奇的协作 | 框架如何落到硬件 | 算子下发到 Cube/Vector |
| 6. 迁移要点与常见坑 | 从 PyTorch 搬过来要改什么 | API 名、动静态、数据下沉 |
| 常见问题 | 速查 | 表格 |

## 0. 一句话锚点

MindSpore = **昇腾生态里的 PyTorch/TensorFlow**。它向上给算法工程师提供 Python 建模 API（构网、训练、推理），向下对接 CANN，把算子编译、下发到昇腾 NPU 的达芬奇核上执行。它的差异化卖点是：**统一的动静态图编程**、**函数式自动微分**和**原生的多维自动并行**（数据/模型/流水/优化器并行一把梭）。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

昇腾软件栈自下而上分层，MindSpore 处在"AI 框架层"，是连接算法与硬件的中枢：

```
            昇腾(Ascend)软件栈                       英伟达(NVIDIA)对照
 ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
 │ 应用/套件层                       │   │ 套件层                            │
 │  MindFormers / MindIE / ...      │ ↔ │  Megatron-LM / TensorRT-LLM /vLLM│
 ├─────────────────────────────────┤   ├─────────────────────────────────┤
 │ ★ AI 框架层                      │   │ AI 框架层                         │
 │   MindSpore (本文)               │ ↔ │  PyTorch / TensorFlow            │
 │   (也支持 PyTorch+torch_npu)     │   │                                  │
 ├─────────────────────────────────┤   ├─────────────────────────────────┤
 │ 异构计算架构 CANN                 │   │ 异构计算架构 CUDA                  │
 │  GE图引擎 / 算子库 / Runtime/驱动 │ ↔ │ cuDNN/cuBLAS / Runtime / Driver  │
 ├─────────────────────────────────┤   ├─────────────────────────────────┤
 │ 集合通信 HCCL                     │ ↔ │ 集合通信 NCCL                     │
 ├─────────────────────────────────┤   ├─────────────────────────────────┤
 │ 硬件:昇腾 NPU(达芬奇架构)         │ ↔ │ 硬件:GPU(SM/Tensor Core)         │
 └─────────────────────────────────┘   └─────────────────────────────────┘
```

**昇腾 ↔ 英伟达生态对照表（迁移必备）**

| 维度 | 昇腾世界 | 英伟达世界 | 说明 |
| --- | --- | --- | --- |
| 加速芯片 | NPU（昇腾，达芬奇架构） | GPU（SM + Tensor Core） | NPU 以 Cube 矩阵单元为核心 |
| 异构计算架构 | CANN | CUDA | 编程框架 + 运行时 + 驱动 |
| 算子加速库 | CANN 算子库（AOL/aclnn） | cuDNN / cuBLAS | 卷积、GEMM 等高性能算子 |
| 算子开发语言 | Ascend C | CUDA C++ | 写自定义高性能算子 |
| 集合通信库 | HCCL | NCCL | all-reduce/all-gather 等原语 |
| AI 框架 | **MindSpore** | PyTorch / TensorFlow | 本文主角 |
| PyTorch 适配层 | torch_npu（昇腾插件） | 原生 PyTorch | 让 PyTorch 跑在 NPU 上 |
| 训练套件 | MindFormers / ModelLink | Megatron-LM / HF Trainer | 大模型分布式训练 |
| 推理引擎 | MindIE / MindSpore Lite | TensorRT-LLM / vLLM | 推理加速 |
| 量化工具 | msModelSlim | GPTQ/AWQ 等 | 压缩量化 |
| 中间表示/编译 | MindIR / GE 图 | TorchScript / FX / TensorRT IR | 图级 IR |

> 关键认知：在昇腾上跑模型有**两条路**——① 直接用 **MindSpore**（端到端国产栈，自动并行最强）；② 用 **PyTorch + torch_npu**（沿用 PyTorch 习惯，迁移成本最低）。本文聚焦第一条路 MindSpore，但两条路最终都落到 CANN→NPU。

## 2. 两种执行模式:Graph 模式 vs PyNative 模式

MindSpore 最核心的编程概念是**动静统一**：同一套 API，既能用静态图（高性能），又能用动态图（好调试）。

- **Graph 模式（静态图 / 图模式）**：把整个神经网络模型**先编译成一整张计算图**，再整图下发执行。利用图优化（算子融合、常量折叠、内存复用等）和整图下沉技术提升性能，便于规模化部署和跨平台运行。适合**网络结构固定、追求高性能**的训练/推理场景。
- **PyNative 模式（动态图）**：神经网络中的各个算子**逐一下发执行**（即时执行 eager），方便编写和调试，支持单独对某步求梯度。

二者的主要区别：

| 对比项 | Graph 模式（静态图） | PyNative 模式（动态图） |
| --- | --- | --- |
| 执行方式 | 先编译整图再下发 | 算子逐个即时下发 |
| 性能 | 高（图优化 + 整图下沉） | 相对较低（频繁 Host-Device 交互） |
| 调试 | 难（无法打断点，靠算子打印 + 事后看输出） | 易（可下断点、取中间结果、用 pdb 调试） |
| 使用场景 | 网络固定、要高性能、要部署 | 脚本开发、流程调试、动态控制流 |
| 精度 | 与 PyNative 一致 | 与 Graph 一致 |
| 对标 PyTorch | 类似 `torch.compile` / TorchScript | 类似 PyTorch 默认 eager |

> 实践套路：**先用 PyNative 把网络调通调对，再切到 Graph 模式跑大规模训练/上线**。这正对应 PyTorch 用户"先 eager 开发、再 `torch.compile` 加速"的习惯。

## 3. 计算图与"整图下沉"机制

为什么图模式在 NPU 上能比 GPU 上的等价做法获得更大收益？关键在**整图下沉（Graph Sinking）**。

普通逐算子执行时，每个算子都要 Host（CPU）下发一次任务、Device（NPU）算完再回到 Host，**Host-Device 之间反复往返**，调度开销大：

```
 逐算子执行(PyNative):                整图下沉(Graph):
 Host: op1↓  op2↓  op3↓  ...          Host: [整张图一次性下沉]↓
        │     │     │                        │
 Device:└→■   └→■   └→■                Device:└→■■■■■■... (图内连续执行,中途不回Host)
   每个算子都来回一次               减少调度往返,数据/权重常驻NPU
```

整图下沉把**整张计算图、甚至训练循环的多步迭代**一次性下沉到 Device 端连续执行，配合**数据下沉（dataset sinking）**——数据通过通道直接送进 NPU，不必每步从 Host 喂——大幅减少调度与数据搬运开销。这是 MindSpore 在昇腾上能"喂饱"达芬奇核的核心机制。

图编译产物是 **MindIR**（MindSpore 的图级中间表示），由 CANN 的 **GE（Graph Engine，图引擎）** 接手做后端图优化与算子下发，最终落到达芬奇核。

## 4. 自动微分与自动并行

**函数式自动微分**：MindSpore 采用基于源码变换（source-to-source）的函数式微分，对函数求梯度得到的还是一个函数（如 `grad` 变换），天然契合静态图，便于做图级优化与高阶微分。这是它与 PyTorch 命令式 `loss.backward()` 的思路差异。

**多维自动并行**：训练千亿参数大模型时，单卡放不下，需要切分。MindSpore 把多种并行策略统一在一套语义里：

```
        一个超大模型的训练并行切分
 ┌──────────────────────────────────────────────┐
 │ 数据并行 DP   : 不同卡跑不同 batch,梯度 all-reduce │
 │ 模型/张量并行 TP: 把一个大矩阵按行/列切到多卡       │
 │ 流水线并行 PP : 把不同层放到不同卡,像流水线传递     │
 │ 优化器并行    : 优化器状态分片(类似 ZeRO)          │
 │ 专家并行 EP   : MoE 的不同专家分到不同卡           │
 └──────────────────────────────────────────────┘
   多种并行可叠加 = 多维混合并行,跨卡通信走 HCCL
```

对标英伟达世界：这套能力相当于 **Megatron-LM + DeepSpeed ZeRO** 合体，但 MindSpore 把它做成框架内置的"自动并行"——可在一定程度上由框架根据切分策略自动插入通信算子（all-reduce/all-gather/reduce-scatter，底层走 **HCCL**，对标 NCCL），降低手写并行代码的负担。

## 5. 与 CANN / 达芬奇架构的协作

MindSpore 自己不直接操作硬件，它把算子交给 CANN，再由 CANN 调度到达芬奇核的不同计算单元：

```
 MindSpore 算子(MatMul/Conv/Add/Softmax...)
        │  编译为 MindIR
        ▼
 CANN: GE 图引擎(图优化) + 算子库 + Runtime + 驱动
        │  选择最优算子实现 & 下发任务
        ▼
 昇腾 NPU 达芬奇核:
   ┌────────────┬────────────┬───────────┐
   │ Cube 单元   │ Vector 单元 │ Scalar 单元│
   │ 矩阵乘(GEMM)│ 向量逐元素   │ 标量/控制   │
   │ 卷积/Attn核 │ 激活/归一化  │ 流程调度    │
   └────────────┴────────────┴───────────┘
```

- **Cube 单元**：处理矩阵乘、卷积这类稠密线性代数，是算力主体（对标 GPU Tensor Core）。
- **Vector 单元**：处理激活、归一化、逐元素运算。
- **Scalar 单元**：负责标量计算与流程控制。

所以"框架算子能不能高效跑"取决于 CANN 算子库是否覆盖、是否融合得好。遇到缺失或慢的算子，可用 **Ascend C** 自定义算子（对标用 CUDA C++ 写 kernel）。

## 迁移要点与常见坑

从 PyTorch/CUDA 世界迁到 MindSpore + 昇腾，重点改这些：

1. **API 名称与语义差异**：`nn.Module`→`nn.Cell`，`forward`→`construct`，`tensor.cuda()` 的设备搬运换成 MindSpore 的设备/上下文设置。逐个对照官方 API 映射表，不要想当然。
2. **动静态心智切换**：Graph 模式下 Python 原生控制流、`print`、第三方库调用受限（需用框架算子或图内打印）；先在 PyNative 调通再切 Graph。
3. **数据下沉配置**：要发挥昇腾性能，通常需开启数据下沉/图下沉。下沉模式下取中间值、调试方式与 eager 不同，调试期可关掉、上线再开。
4. **算子覆盖度**：个别新颖算子 CANN 可能尚未支持或未融合，表现为报错或慢；先查官方算子支持列表，必要时用 Ascend C 自定义或替换等价算子。
5. **混合精度与溢出**：NPU 上 fp16/bf16 的数值行为、溢出检测（如溢出告警/loss scale）策略与 GPU 有差异，迁移大模型时务必校验精度对齐。
6. **分布式启动与通信**：多卡训练走 HCCL，需要正确的组网/rank 配置（昇腾常见 8 卡一机，跨机看网络拓扑）。通信慢多半是拓扑或 HCCL 算法选择问题。
7. **环境安装别硬记命令**：驱动、固件、CANN、MindSpore、Python 版本之间有**严格配套关系**，版本错配是头号坑。

> ⚠️ 护栏：以上**具体的安装命令、包名、版本号、镜像、配套表与确切性能数字，一律以华为昇腾官方文档（Ascend 社区 / MindSpore 官网）为准**，本文只讲流程含义、依赖关系与坑位，不给可能过期的命令。

**性能调优思路（机制层面，非具体参数）**：
- 优先用 **Graph 模式 + 数据下沉 + 整图下沉**，让 NPU 连续执行少回 Host。
- 关注**算子融合**是否生效（融合越多，Host 调度与中间内存搬运越少）。
- 让计算落在 **Cube 单元**友好的形状上（矩阵维度对齐到硬件分块尺寸，避免碎算）。
- 用 **profiling 工具**定位是算子慢、通信慢还是数据供给慢（对标 Nsight 的定位思路），再对症下药。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| MindSpore 对标英伟达谁？ | 对标 PyTorch / TensorFlow，是昇腾框架层国产主力。 |
| 在昇腾上一定要用 MindSpore 吗？ | 不一定。也可用 PyTorch + torch_npu，迁移成本更低；MindSpore 自动并行更强、国产端到端。 |
| Graph 和 PyNative 怎么选？ | PyNative 调试，Graph 跑性能/上线；二者精度一致。 |
| 整图下沉解决什么？ | 减少 Host-Device 往返调度，让 NPU 连续执行，喂饱达芬奇核。 |
| 自动并行对标什么？ | 类似 Megatron-LM + DeepSpeed ZeRO 的能力，框架内置。 |
| 底层通信用什么库？ | HCCL（对标 NCCL），多卡 all-reduce 等走它。 |
| 算子不支持怎么办？ | 查 CANN 算子支持列表，必要时用 Ascend C 自定义算子。 |
| 安装版本怎么配？ | 驱动/固件/CANN/MindSpore/Python 严格配套，**以官方配套表为准**。 |

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
