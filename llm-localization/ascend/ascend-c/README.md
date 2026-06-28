# Ascend C：昇腾算子编程语言

> Ascend C 是华为为昇腾 NPU 推出的**算子开发编程语言**——用 C/C++ 语法直接为达芬奇架构写自定义算子(Kernel)，对标英伟达世界的 **CUDA C**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/算子/算子开发]]

## 阅读地图

| 你想搞清楚的问题 | 看哪一节 |
| --- | --- |
| Ascend C 是什么、为啥需要它 | §0 一句话锚点、§1 定位 |
| 它在昇腾软件栈哪一层 | §1 地基 |
| CUDA 程序员的迁移心智图 | §1 对照表 |
| 达芬奇 Cube/Vector 硬件长啥样 | §2 硬件抽象 |
| 一个 Kernel 怎么写、SPMD 怎么并行 | §3 编程模型 |
| 数据怎么在片上搬运、为啥要流水 | §4 内存层级与流水(Tiling/Pipeline) |
| 从源码到部署的整条链路 | §5 开发部署流程 |
| 从 CUDA 迁移要改什么、踩什么坑 | §6 迁移要点与坑 |
| 常见疑问速查 | 常见问题表 |

## 0. 一句话锚点

**Ascend C = 昇腾上的 CUDA C。** 当你需要一个 CANN 内置算子库里**没有**的算子(比如一个融合算子、一个新激活函数、一个定制 Attention)，或者想把几个算子**手工融合**以省掉中间 HBM 读写时，就用 Ascend C 直接对达芬奇核(AI Core)编程，把计算映射到 Cube 和 Vector 单元上。它是「框架/套件之下、硬件之上」那一层手写极致性能的工具。

一句话区分三个容易混的概念：
- **CANN**：昇腾的异构计算软件栈(≈ CUDA Toolkit 整体)。
- **Ascend C**：CANN 里**写算子内核**的编程语言(≈ CUDA C 语言本身)。
- **aclnn / AOL**：算子被编译后对外暴露的**单算子调用接口**(≈ cuDNN/cuBLAS 那种 host 侧 API)。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(Ascend C 在哪)

```
┌─────────────────────────────────────────────────────────┐
│  套件层  MindFormers / MindIE / ModelLink / msmodelslim  │  ← 训练/推理/量化套件
├─────────────────────────────────────────────────────────┤
│  框架层  MindSpore / PyTorch(torch_npu) / TensorFlow     │  ← 写模型
├─────────────────────────────────────────────────────────┤
│  CANN(异构计算架构)                                       │
│   ├─ GE 图引擎 / AOL 算子加速库(aclnn 单算子接口)        │
│   ├─ 【★ Ascend C：自定义算子开发语言 ★】 ← 本文       │
│   ├─ 算子编译器(类比 nvcc，把 Kernel 编成 NPU 二进制)   │
│   └─ Runtime / Driver(任务下发、内存、Stream)           │
├─────────────────────────────────────────────────────────┤
│  硬件层  昇腾 NPU(达芬奇架构 AI Core：Cube + Vector)    │
└─────────────────────────────────────────────────────────┘
```

**关键定位**：Ascend C 是**CANN 内部、紧贴硬件**的那层。上面框架/套件调用的内置算子，绝大多数也是用 Ascend C(或更早期的 TIK/DSL)写出来的；你写的自定义算子编译后，会和内置算子一样被框架以 aclnn 接口调用。

### 1.2 「昇腾 ↔ 英伟达」对照表(迁移心智图)

| 维度 | 昇腾(华为) | 英伟达 | 一句话说明 |
| --- | --- | --- | --- |
| 加速芯片 | **NPU**(昇腾 910/310 系列) | GPU(A100/H100…) | 都是 AI 加速器，但微架构思路不同 |
| 微架构 | **达芬奇(Da Vinci)**：Cube + Vector + Scalar | SIMT + Tensor Core | 昇腾以矩阵 Cube 为核心 |
| 异构软件栈 | **CANN** | CUDA Toolkit | 整套驱动+运行时+编译器+库 |
| **算子编程语言** | **Ascend C** | **CUDA C/C++** | 本文主角，写 Kernel 的语言 |
| Kernel 编译器 | 算子编译器(bisheng/ccec 等) | nvcc | 把内核编成芯片二进制 |
| 内核启动语法 | `<<<...>>>`(Ascend C 核函数调用宏) | `kernel<<<grid,block>>>` | 都用三尖括号下发 |
| 数学/DNN 加速库 | AOL 算子库 / aclnn 接口 | cuDNN / cuBLAS | 内置高性能算子，host 侧直接调 |
| 集合通信库 | **HCCL** | NCCL | AllReduce/AllGather 等 |
| 深度学习框架 | MindSpore / torch_npu | PyTorch / TF | torch_npu 让 PyTorch 跑在 NPU |
| 大模型训练套件 | MindFormers / ModelLink | Megatron-LM / HF | 并行训练 |
| 大模型推理引擎 | MindIE | TensorRT-LLM / vLLM | 高性能推理 |
| 量化压缩工具 | msmodelslim | GPTQ/AWQ 工具链 | 权重量化 |
| 设备查询命令 | npu-smi | nvidia-smi | 看卡状态 |

> 记一句话：**「Ascend C 之于 CANN，正如 CUDA C 之于 CUDA Toolkit」**。会写 CUDA Kernel 的人迁移最快，难点在「微架构变了」(SIMT→Cube/Vector)和「内存搬运要显式编排」。

## 2. 硬件抽象：达芬奇 AI Core 长什么样

Ascend C 编程必须先在脑子里建立达芬奇 **AI Core** 的硬件模型。一个 AI Core 内部有三类计算单元 + 多级片上缓冲：

```
                 ┌──────────── 一个 AI Core ────────────┐
   HBM(片外)    │                                       │
  全局内存 GM ──┼──► [搬运单元 MTE] ──► 片上 Local Memory │
   (≈ GPU      │                       ┌──────────────┐ │
    Global)    │                       │ Cube 单元    │ │ ← 矩阵乘(MatMul)
               │                       │ (≈Tensor Core)│ │
               │                       ├──────────────┤ │
               │                       │ Vector 单元  │ │ ← 逐元素/规约(Add,ReLU,Softmax)
               │                       ├──────────────┤ │
               │                       │ Scalar 单元  │ │ ← 标量/循环控制流
               │                       └──────────────┘ │
               │   片上缓冲(队列管理):  Unified Buffer / L1 / L0A/L0B/L0C  │
               └───────────────────────────────────────┘
```

- **Cube 单元**：专做矩阵乘累加(MatMul)，是算力主力——大模型里 Linear/Attention 的 QK^T、PV、FFN 都压在它上面。对标 Tensor Core。
- **Vector 单元**：做逐元素运算与规约——激活、归一化、Softmax、element-wise add 等。
- **Scalar 单元**：标量计算与控制流，管循环、地址、分支。
- **MTE(Memory Transfer Engine)**：负责 GM(全局内存)↔ 片上缓冲之间的数据搬运。**Ascend C 编程的核心难点就是显式编排这些搬运**——这跟 CUDA 里「数据进 shared memory」类似，但更显式、更需要手工流水。

**对程序员的含义**：你写 Ascend C 时，要思考「这一步是 Cube 干还是 Vector 干」「数据现在在 GM 还是片上」。这与 CUDA 里只想「一个线程算什么」很不同——昇腾把**计算分工**和**数据搬运**都暴露给了你。

## 3. 编程模型：核函数与 SPMD

Ascend C 的核函数(Kernel)编程范式和 CUDA 高度神似，便于迁移：

- **核函数(`__global__` 类语义)**：标记为在 NPU 上执行的入口函数。
- **三尖括号启动**：host 侧用 `kernel<<<blockDim, ...>>>(args)` 风格的宏把任务下发到多个 AI Core，类似 `kernel<<<grid,block>>>`。其中 `blockDim` 表示**用多少个核(或核上的逻辑块)**，是 SPMD(单程序多数据)并行度。
- **block_idx**：每个并行实例用类似 `GetBlockIdx()` 拿到自己的编号，据此切分自己负责的数据分片——对应 CUDA 的 `blockIdx`。

一个典型 Ascend C 算子(以官方 AddCustom 这种逐元素加法为例)的骨架思路：

```
1) Init():   根据 block_idx 计算本核负责的数据偏移与长度，
             设置好输入/输出 GM 的 GlobalTensor，划分片上 Buffer 队列。
2) Process():按 Tiling 切成若干 tile，对每个 tile 走「搬入→计算→搬出」三段流水：
       CopyIn()  : MTE 把一个 tile 从 GM 搬到片上(Local)，入 Queue
       Compute() : Vector/Cube 在片上算(如 Add / MatMul)，结果入 Queue
       CopyOut() : MTE 把结果从片上搬回 GM
```

这套「CopyIn → Compute → CopyOut」三段式是 Ascend C 的**惯用结构**，几乎所有 Vector 类算子都长这样。

## 4. 内存层级与流水：Tiling + Pipeline(性能命门)

Ascend C 写得快不快，全看两件事：**Tiling(切分)** 和 **Pipeline(流水/双缓冲)**。

### 4.1 为什么要 Tiling

片上缓冲很小，装不下整块大张量。所以要把大数据**切成一个个 tile**，分批搬上片、算完搬下来。Tiling 参数(每个 tile 多大、切几块、每核分多少)既影响**正确性**也直接决定**性能**，通常在 host 侧算好后通过 TilingData 传给 kernel。这相当于 CUDA 里手工决定 tile size + 共享内存分块，但昇腾把它做成了显式的一等公民。

### 4.2 为什么要 Pipeline(双缓冲)

如果「搬入→计算→搬出」严格串行，那 Cube/Vector 在等数据搬运时就空转了。Ascend C 用 **Queue + 双缓冲(double buffer)** 把搬运和计算**重叠**起来：

```
朴素串行(慢)：  搬入T0 → 算T0 → 搬出T0 → 搬入T1 → 算T1 → 搬出T1 ...
                ████░░░░████  ░░░░████░░░░  ← 计算单元大量空闲

流水重叠(快)：  搬入T0 ┐
                       算T0 ┐  搬入T1 ┐
                            搬出T0    算T1 ┐ 搬入T2 ┐
                                          搬出T1   算T2 ...
                ↑ MTE 在搬下一块时，Vector/Cube 已经在算这一块 → 算力被喂饱
```

Ascend C 通过 `TQue`(队列)的 `EnQue/DeQue` 配合 `AllocTensor/FreeTensor`，让框架自动在搬运与计算单元间插入同步，实现这种重叠。**调优的本质就是：让 Cube/Vector 别饿着，让 MTE 别堵着。**

### 4.3 两种调用方式

- **Kernel 直调(KernelLaunch)**：纯算子工程，自己写 host 侧调用，适合快速验证/单算子。
- **框架调用(FrameworkLaunch)**：把算子注册成标准算子(含算子原型、InfoShape、Tiling、aclnn 接口),最终能被 PyTorch/MindSpore 当成普通算子调用。官方样例里 `FrameworkLaunch/AddCustom` 就是这条路。

## 5. 开发部署流程(讲清每步为什么)

> ⚠️ 以下只讲**流程含义与依赖关系**。涉及的精确命令、包名、镜像、版本号、路径，一律**以华为昇腾官方文档(Ascend 社区 / CANN 文档)为准**——本文不杜撰任何具体命令或版本。

整条链路的逻辑顺序与「为什么」：

```
①环境  →  ②写Kernel  →  ③写Tiling/原型  →  ④编译  →  ⑤打包部署  →  ⑥调用验证
```

| 步骤 | 做什么 | 为什么需要这步 | 常见坑 |
| --- | --- | --- | --- |
| ① 准备环境 | 装好 CANN 工具包，source 环境变量，确认 NPU 可见 | 算子编译器、头文件、Runtime 都在 CANN 里；环境变量没 source 则编译/运行找不到工具链 | 多版本 CANN 混装；环境变量没生效；驱动与 CANN 版本不配套 |
| ② 写 Kernel | 用 Ascend C 写 CopyIn/Compute/CopyOut | 这是算子的核心计算逻辑 | Vector/Cube 用错单元；片上 Buffer 越界 |
| ③ 写 Tiling + 算子原型 | 在 host 侧算 TilingData，定义输入输出 shape 推导 | 框架要知道怎么切数据、输出多大才能调度 | Tiling 算错导致越界或性能差；shape 推导写错 |
| ④ 编译(build) | 调 `build.sh` 之类脚本，把算子编成芯片二进制 | 类比 nvcc 把 .cu 编成 cubin | 编译目标架构(soc version)选错；交叉编译 aarch64 时工具链不对 |
| ⑤ 打包部署 | 生成自定义算子包(`*.run`)并安装到环境 | 让算子被注册进 CANN 的算子库，框架才找得到 | 算子包没装/装错位置；权限问题 |
| ⑥ 调用验证 | 用 aclnn 单算子接口或框架侧脚本跑一遍并比对结果 | 验证数值正确性与性能 | 没跟标杆(CPU/标准库)做精度比对；只看跑通不看精度 |

官方 `AddCustomSample` 就是这套流程的最小可运行示例：先 `build.sh` 编译并产出算子包(`custom_opp_*.run`)，安装后再用 `AclNNInvocation` 通过 aclnn 接口调用验证。这条「编译 → 装包 → aclnn 调用」的链路，是理解 Ascend C 工程结构的最佳入口。

## 6. 迁移要点：从 CUDA 到 Ascend C 改什么、坑在哪

### 6.1 心智迁移对照

| CUDA 里的概念/习惯 | 到 Ascend C 怎么变 |
| --- | --- |
| `__global__` 核函数 | Ascend C 核函数(语义类似，结构上多了 CopyIn/Compute/CopyOut 三段) |
| `blockIdx/threadIdx` SIMT | `GetBlockIdx()` 取核号；并行粒度是「核」而非「线程束」，**没有 SIMT 线程级并行的那套**，更像「每核一份数据」的 SPMD |
| `shared memory` 手工分块 | 片上 Local Buffer + **Tiling 显式切分**，且要自己编排 MTE 搬运 |
| Tensor Core(wmma) | **Cube 单元**(MatMul)，但 API 风格不同 |
| element-wise/规约 kernel | 交给 **Vector 单元**，写法是 Vector API |
| 异步拷贝 + 双缓冲(自己写) | `TQue` 队列 + double buffer，框架辅助同步 |
| cuDNN/cuBLAS 直接调 | 优先用 **AOL/aclnn 内置算子**，没有再自己写 Ascend C |

### 6.2 常见坑(机制层面)

1. **不要见算子就手写**：CANN 内置算子库已高度优化，**先查 aclnn 有没有现成的**；只有内置缺失、或要做融合(省 HBM 来回搬)时才上 Ascend C。盲目手写往往比内置慢。
2. **Cube/Vector 分工搞错**：把本该 Cube 做的 MatMul 用 Vector 硬算，或反过来，性能差几个数量级。先想清楚「这步是矩阵乘还是逐元素」。
3. **Tiling 拍脑袋**：tile 太大装不下片上缓冲(越界/报错)，太小则搬运开销占比高、算力喂不饱。Tiling 是性能命门，要按缓冲大小和数据量算。
4. **忘了做双缓冲**：串行的 CopyIn/Compute/CopyOut 会让计算单元空转。流水重叠是昇腾性能的关键。
5. **SoC 版本/目标架构选错**：编译时要对准目标昇腾芯片型号；910 和 310、不同代次的微架构能力不同，编错目标跑不起来或性能不对。
6. **精度不对齐**：迁移后只看「跑通」不看「数值对不对」。务必和 CPU/标准实现做精度比对(尤其是累加顺序、数据类型差异引起的误差)。
7. **环境/版本错配**：驱动、固件、CANN、框架插件(torch_npu)之间有版本配套关系，错配是最高频的「装不上/跑不起来」根因。**具体配套关系与版本以官方文档为准。**

### 6.3 调优思路(原则而非命令)

- **先用 profiling 找瓶颈**：是 Cube 算力打满，还是 MTE 搬运成为瓶颈，还是 Vector 卡住——不同瓶颈对策完全不同。
- **搬运瓶颈** → 优化 Tiling、加双缓冲、减少 GM 往返(算子融合)。
- **算力瓶颈** → 提高 Cube/Vector 利用率，合理排布数据格式(NPU 对数据 layout 敏感)。
- **融合优先**：把多个小算子融成一个 Ascend C 算子，省掉中间结果写回 HBM 再读出的开销——这是大模型场景里 Ascend C 最大的价值点(类似 GPU 上的 FlashAttention 思路)。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| Ascend C 等于 CUDA 吗? | 它对标的是 **CUDA C 语言**(写 Kernel 那层)；对标整套 CUDA Toolkit 的是 **CANN**。 |
| 什么时候才需要写 Ascend C? | 内置算子(aclnn/AOL)没有该算子、或想做**算子融合**省 HBM 搬运时。否则优先用内置。 |
| 不会 CUDA 能学吗? | 能，但有 CUDA Kernel 经验迁移最快，因为核函数/三尖括号/分块思路都相通。 |
| Cube 和 Vector 怎么选? | 矩阵乘(MatMul)用 Cube；逐元素/规约(激活、归一、Softmax)用 Vector。 |
| 为什么要写 Tiling? | 片上缓冲装不下大张量，必须切片分批搬算；Tiling 同时决定正确性和性能。 |
| 写完算子怎么被 PyTorch 用? | 走 FrameworkLaunch 注册成标准算子并产出 aclnn 接口，torch_npu 即可像普通算子一样调用。 |
| 精确命令/版本去哪查? | 一律以**华为昇腾官方文档(Ascend 社区 / CANN 文档)**为准，不要照抄网上零散命令。 |
| 跟 TIK/DSL 啥关系? | TIK 是更早期的算子开发方式；Ascend C 是当前主推的 C/C++ 风格方案，更接近 CUDA C 体验。 |

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
