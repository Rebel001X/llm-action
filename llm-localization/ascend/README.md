# 昇腾(Ascend)国产化 LLM 软件栈总览

> 一句话定位：昇腾是华为打造的"NPU 硬件 + CANN 异构计算 + 框架/套件"全栈 AI 基础设施，对标英伟达"GPU + CUDA + 训推套件"的国产替代闭环；本页是整个 `ascend/` 目录的入口与迁移心智图。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 你想知道 | 看哪一节 |
| --- | --- |
| 昇腾到底解决什么、整栈长什么样 | 0. 一句话锚点 / 1. 地基 |
| 从 CUDA/英伟达迁过来要建立的对照表 | 1. 昇腾 ↔ 英伟达生态对照 |
| 硬件层为何快(达芬奇/Cube) | 2. 硬件层:NPU 与达芬奇架构 |
| CANN 这层都包了什么 | 3. CANN:对标 CUDA 的异构计算底座 |
| 该用 PyTorch 还是 MindSpore | 4. 框架层:双框架并存 |
| 训练/推理/量化各用哪个套件 | 5. 套件层:训练·推理·压缩全家桶 |
| 从 GPU 把代码搬过来要改什么 | 6. 迁移要点与常见坑 |
| 每个子目录学什么 | 🔗 跳转链接(本目录地图) |

## 0. 一句话锚点

把英伟达世界的认知平移一次就懂：**显卡换成 NPU、CUDA 换成 CANN、NCCL 换成 HCCL、cuDNN/cuBLAS 换成 CANN 内置算子库、TensorRT-LLM/vLLM 换成 MindIE/vLLM-Ascend、Megatron 换成 ModelLink/MindFormers**。底层指令集与编程模型不同，但"分层解耦、上层尽量兼容生态"的思路一致——所以迁移的关键不是重学一切，而是把每一层一一对位。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(从下往上)

```
        ┌─────────────────────────────────────────────┐
 应用层 │  你的 LLM 训练/推理脚本、业务服务               │
        ├─────────────────────────────────────────────┤
 套件层 │ MindFormers │ ModelLink │ MindIE │ msmodelslim │  训练/推理/量化大礼包
        ├─────────────────────────────────────────────┤
 框架层 │   PyTorch(torch_npu 插件)  │   MindSpore       │  两套框架二选一/并存
        ├─────────────────────────────────────────────┤
 加速库 │  集合通信 HCCL │ 算子库(AOL) │ 图引擎 GE        │  ≈ NCCL + cuDNN/cuBLAS
        ├─────────────────────────────────────────────┤
 CANN   │  编译器 / 运行时(Runtime) / Driver 抽象        │  ≈ CUDA Toolkit + 驱动
        ├─────────────────────────────────────────────┤
 硬件层 │  昇腾 NPU(达芬奇架构:Cube/Vector/Scalar)      │  ≈ GPU(SM/Tensor Core)
        └─────────────────────────────────────────────┘
```

- **解耦思想**：每一层只依赖下一层的稳定接口。上层套件不该直接写硬件指令，硬件升级也尽量不破坏上层 API——这点和 CUDA 栈完全同构。
- **生态兼容策略**：昇腾刻意让 `torch_npu` 尽量"无感"接住 PyTorch 代码、让 vLLM-Ascend 复用 vLLM 上层逻辑，目的就是降低迁移摩擦。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 层 | 昇腾(国产) | 英伟达(对标) | 一句话说明 |
| --- | --- | --- | --- |
| 硬件 | 昇腾 NPU(910/310 系列,达芬奇架构) | GPU(A100/H100,SM+Tensor Core) | 都是"大量并行 + 矩阵专用单元" |
| 编程底座 | CANN(Compute Architecture for Neural Networks) | CUDA Toolkit + Driver | 编译器/运行时/驱动抽象 |
| 算子开发语言 | Ascend C | CUDA C++ | 写自定义高性能算子 → 见 [[ascend-c]] |
| 算子库 | CANN 内置算子(AOL/AICPU 等) | cuDNN / cuBLAS / cutlass | 卷积、GEMM、归一化等高性能库 |
| 集合通信 | HCCL(Huawei Collective Comm Lib) | NCCL | AllReduce/AllGather 等多卡通信 |
| 图编译/执行 | GE(Graph Engine,图模式) | CUDA Graph / TorchInductor | 整图下沉、算子融合 |
| 深度学习框架 | MindSpore / PyTorch(torch_npu) | PyTorch / TensorFlow | MindSpore 是原生自研栈 |
| 训练套件 | MindFormers / ModelLink(AscendSpeed) | Megatron-LM / HF Transformers + Trainer | 大模型并行训练流水线 |
| 推理引擎 | MindIE / vLLM-Ascend | TensorRT-LLM / vLLM | 高吞吐低延迟推理服务 |
| 量化压缩 | msModelSlim | GPTQ/AWQ/llm-compressor 工具链 | W8A8/W4A16 等量化 |
| 微调库 | PEFT(昇腾适配) / openMind | HF PEFT / TRL | LoRA 等参数高效微调 |
| 模型社区 | 魔乐(openMind 社区) / 魔搭 ModelScope | Hugging Face Hub | 模型/数据托管与一键加载 |

> 读法：左右两列是"同一格子里的功能"。当你看到一篇英伟达教程提到 NCCL，脑子里立刻替换成 HCCL，迁移成本就小了一大半。

## 2. 硬件层:NPU 与达芬奇(Da Vinci)架构

昇腾 NPU 的核心是**达芬奇架构**，一个 AI Core 内有三类计算单元，分工类似 GPU 里"Tensor Core + CUDA Core + 标量处理"：

```
        ┌──────────── 昇腾 AI Core(达芬奇) ────────────┐
        │  ┌──────────┐  矩阵主力,做 GEMM/卷积          │
        │  │ Cube 单元 │  ≈ 英伟达 Tensor Core          │
        │  └──────────┘                                 │
        │  ┌──────────┐  逐元素:激活/归一化/向量运算     │
        │  │ Vector   │  ≈ GPU 的 SIMD 向量通道          │
        │  └──────────┘                                 │
        │  ┌──────────┐  控制流/标量计算/地址            │
        │  │ Scalar   │  ≈ GPU 标量/控制逻辑             │
        │  └──────────┘                                 │
        │  片上缓冲(L0/L1 Buffer) + 多级数据搬运(MTE)   │
        └───────────────────────────────────────────────┘
```

- **Cube 单元**是吃满算力的关键：LLM 里绝大部分 FLOPs 来自矩阵乘(QKV 投影、FFN、Attention 打分)，能不能让 Cube 满载，直接决定 MFU(算力利用率)。
- **数据搬运(MTE)与片上 Buffer** 类似 GPU 的 shared memory + 显存层级：手写高性能算子时，"减少 HBM↔片上来回搬运"是第一优化原则，和 GPU 上"用好 shared memory、提高算术强度"完全同理。
- 详见 [[ascend-infra/达芬奇架构]] 与算子开发 [[ascend-c]]。

## 3. CANN:对标 CUDA 的异构计算底座

CANN 之于昇腾，等于 CUDA Toolkit 之于英伟达——它是上层框架与底层硬件之间的"全部胶水"：

1. **编译器**：把算子/图编译成 NPU 可执行的二进制(对位 nvcc/PTX 编译链)。
2. **Runtime 运行时**：管理 Device、Stream、内存、Kernel 下发(对位 CUDA Runtime/Driver API)。
3. **算子库 + 算子开发(Ascend C)**：内置高性能算子,缺的就自己用 Ascend C 写(对位 cuDNN/cuBLAS + 手写 CUDA Kernel)。
4. **图引擎 GE**：图模式下做整图下沉与算子融合,减少 Host-Device 交互(对位 CUDA Graph / 编译期融合)。

> CANN 的版本、与 Driver/Firmware 的配套关系是部署中最常见的坑源(见第 6 节)。**具体版本号与配套矩阵以华为昇腾官方文档(Ascend 社区)为准**，本仓库不写死版本。镜像可参考社区提供的 CANN 容器镜像(地址以官方/社区 registry 为准)。

## 4. 框架层:PyTorch 与 MindSpore 双轨

昇腾上跑 LLM 有两条路线，按团队既有技术栈选：

| 路线 | 入口 | 适合谁 | 对标 |
| --- | --- | --- | --- |
| **PyTorch + torch_npu** | [[pytorch]] | 已有大量 PyTorch 代码、想最小改动迁移的团队 | 直接对位英伟达 PyTorch 生态 |
| **MindSpore 原生栈** | [[mindspore]] | 想用华为自研全栈、深度图优化的团队 | 自研框架,类比 PyTorch/TF |

- **torch_npu 插件机制**：通过给 PyTorch 注册一个新后端设备(把 `cuda` 心智换成 `npu`),让 `tensor.to("npu")`、`torch.npu.xxx` 这类调用落到 CANN 上。大多数模型代码几乎"无感"运行，但凡是写死 `.cuda()`、依赖某些 GPU-only 算子或第三方 CUDA 扩展的地方就要改(见第 6 节)。
- **HF Transformers / PEFT 适配**：见 [[transformers]] 与 [[peft]]，思路是在昇腾上复用 HF 上层 API，把设备与个别算子替换为 NPU 版本。

## 5. 套件层:训练 · 推理 · 压缩全家桶

```
   训练 ───► MindFormers(MindSpore 系) │ ModelLink/AscendSpeed(PyTorch 系)
              对标 Megatron-LM:TP/PP/DP/SP 等并行 + 大模型预训练/微调流水线
   推理 ───► MindIE(昇腾原生推理引擎) │ vLLM-Ascend(vLLM 的昇腾后端)
              对标 TensorRT-LLM / vLLM:PagedAttention、连续批处理、KV Cache 管理
   压缩 ───► msModelSlim(昇腾量化工具)
              对标 GPTQ/AWQ 工具链:W8A8/W4A16 等,降显存提吞吐
   微调 ───► PEFT(昇腾适配) / openMind 套件
              对标 HF PEFT/TRL:LoRA 等参数高效微调
```

- **训练**：[[mindformers]](MindSpore 生态)与 [[modellink]](PyTorch 生态、原 AscendSpeed)都对标 Megatron-LM，提供张量并行 TP / 流水并行 PP / 数据并行 DP / 序列并行 SP 等组合,用来把超大模型切到多卡多机。
- **推理**：[[mindie]] 是华为原生推理引擎(对标 TensorRT-LLM),[[vllm-ascend]] 让你直接复用 vLLM 的调度/PagedAttention 上层逻辑、把算子后端换成昇腾。两者都围绕"KV Cache 显存管理 + 连续批处理"做吞吐优化。
- **量化压缩**：[[msmodelslim]] 对标 GPTQ/AWQ 量化工具,核心收益是省显存、提升带宽受限场景下的吞吐。量化原理见 [[llm-compression/quantization/量化基础]]。
- **微调与社区**：[[peft]]、[[openmind]] 负责 LoRA 等高效微调与魔乐社区生态。

各组件的 LLM 支持范围另见同目录 `昇腾LLM支持概览.md`。

## 6. 迁移要点与常见坑(从 CUDA 搬到昇腾)

**代码层面要改什么**

- 把硬编码的 `.cuda()` / `device="cuda"` 改为昇腾设备(经由 torch_npu 抽象);通信后端从 NCCL 换成 HCCL。
- 依赖 CUDA-only 第三方扩展(自定义 CUDA Kernel、FlashAttention 的 GPU 实现等)的部分,需要换成昇腾侧等价实现或社区适配版,不能直接编译。
- 混合精度/数据类型支持以昇腾实际支持为准(如某些 dtype/算子组合可能未覆盖),遇到不支持算子要么换写法,要么用 Ascend C 自定义。

**环境与部署层面的坑(高频)**

- **配套版本矩阵**：Driver/Firmware ↔ CANN ↔ torch_npu/MindSpore ↔ 套件,这条链必须版本互相匹配,错配是最常见的"装好了但跑不起来"。**具体配套关系以官方文档为准**,不要凭记忆拼版本。
- **环境变量与初始化**：CANN 需要 source 对应环境脚本后才能找到运行时与算子库;多卡训练需正确配置 HCCL 相关网络/rank 信息。具体脚本路径与变量名以官方文档为准。
- **镜像选择**：优先用官方/社区提供的 CANN/MindIE 容器镜像起步,减少手装 Driver 与依赖踩坑。镜像地址以官方/社区 registry 为准。

**性能调优思路(机制层面,不是调命令)**

- **喂满 Cube**：保证矩阵乘是主力负载,避免被小算子、频繁 Host-Device 同步、动态 shape 拖慢——和 GPU 上"提高 Tensor Core 利用率"同理。
- **用图模式**：能下沉整图就走图模式(GE),让 CANN 做算子融合、减少下发开销,类比 CUDA Graph/编译期融合的收益。
- **减少数据搬运**：优化片上 Buffer 复用、减少 HBM 往返,是手写算子和调优的第一性原则。
- **通信与计算重叠**：多卡训练让 HCCL 的 AllReduce 等通信与反向计算重叠,思路与 NCCL 上完全一致(原理见 [[ai-infra/网络/集合通信原语]])。

## 常见问题

| 问题 | 简答 |
| --- | --- |
| 昇腾上跑 LLM 必须用 MindSpore 吗? | 不必。PyTorch + torch_npu 路线可最小改动迁移;MindSpore 是另一条原生全栈路线。 |
| CANN 等于 CUDA 吗? | 定位等同(异构计算底座),但指令集/编程模型不同;迁移靠"按层对位"而非直接复用二进制。 |
| HCCL 和 NCCL 是一回事吗? | 角色一致(多卡集合通信库),接口与实现是昇腾自研版,环算法等思想可类比。 |
| 训练选 MindFormers 还是 ModelLink? | MindSpore 生态选 MindFormers;PyTorch 生态选 ModelLink(原 AscendSpeed)。 |
| 推理选 MindIE 还是 vLLM-Ascend? | 要华为原生深度优化选 MindIE;想复用 vLLM 生态与调度逻辑选 vLLM-Ascend。 |
| 为什么装好却跑不起来? | 多半是 Driver/Firmware↔CANN↔框架↔套件版本错配,核对官方配套矩阵。 |
| 具体安装命令/版本去哪查? | 一律以华为昇腾官方文档(Ascend 社区)为准,本仓库不写死命令与版本。 |

## 🔗 跳转链接

**本目录(ascend/)子组件地图**
- 硬件/底座：[[ascend-infra/达芬奇架构]] · [[ascend-c]](算子开发) · `ascend910-env-install.md`(环境安装) · `昇腾LLM支持概览.md`
- 框架：[[pytorch]] · [[mindspore]] · [[transformers]] · [[peft]]
- 训练：[[mindformers]] · [[modellink]]
- 推理：[[mindie]] · [[vllm-ascend]]
- 压缩/社区：[[msmodelslim]] · [[openmind]]

**枢纽(知识库主索引)**
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
