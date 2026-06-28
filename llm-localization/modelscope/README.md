# ModelScope(魔搭) + Swift:昇腾 NPU 上的模型仓库与训练/推理一体化

> 魔搭(ModelScope)是国产开源模型社区与生态平台,配套 ms-swift 框架,在昇腾 NPU 上一站式完成模型下载、微调与推理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]] [[llm-inference/README]]

## 阅读地图

| 小节 | 你会得到什么 | 适合谁 |
| --- | --- | --- |
| 0. 一句话锚点 | 一句话记住 ModelScope/Swift 是什么、在哪一层 | 所有人 |
| 1. 昇腾栈定位 + 对标英伟达 | 软件栈分层 + "昇腾↔英伟达"迁移心智图 | 从 GPU 迁移者 |
| 2. ModelScope 生态全景 | Hub / Library / Swift / Studio 各是什么 | 选型者 |
| 3. ms-swift 在 NPU 上的工作机制 | 框架如何对接 PyTorch+torch_npu+CANN | 训练/推理工程师 |
| 4. NPU 微调流程(配图) | 从下模型到拉起训练的全链路含义 | 调参者 |
| 5. NPU 推理与部署流程 | swift infer / deploy 链路与后端选择 | 部署者 |
| 6. 迁移要点与常见坑 | 从 GPU 脚本搬到 NPU 要改什么、易踩坑 | 踩坑者 |
| 常见问题 | 高频疑问速查 | 所有人 |

## 0. 一句话锚点

**ModelScope ≈ 中国版 Hugging Face Hub;ms-swift ≈ 把"下载 + 微调(SFT/LoRA/DPO/...) + 推理部署"打包成一个 CLI 的训练框架,并原生支持昇腾 NPU。** 你在 GPU 上习惯的 `transformers + peft + trl + vLLM` 这一整套,Swift 试图用一套统一命令在 NPU 上跑通。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(自底向上)

```
┌─────────────────────────────────────────────────────────┐
│  应用 / 套件层                                            │
│   ms-swift(本文)、MindFormers、MindIE、ModelLink ...     │
├─────────────────────────────────────────────────────────┤
│  AI 框架层                                                │
│   PyTorch + torch_npu(昇腾适配插件) / MindSpore         │
├─────────────────────────────────────────────────────────┤
│  异构计算架构 CANN                                        │
│   图编译(GE)、算子库、Runtime、HCCL 集合通信、Driver   │
├─────────────────────────────────────────────────────────┤
│  硬件层:昇腾 NPU(达芬奇架构 Cube/Vector/Scalar 单元)   │
└─────────────────────────────────────────────────────────┘
```

ms-swift 处在**最上层的套件层**:它本身不写算子、不管调度,而是站在 `PyTorch + torch_npu` 之上,把模型管理、数据处理、训练循环、并行策略、推理后端这些"流程"封装成统一的命令与配置。真正在 NPU 上跑的算子由 CANN 提供,设备通信由 HCCL 承担。

> 关键认知:**Swift 是"胶水 + 流程"层,不是底层加速库。** 它的 NPU 能力来自下方的 `torch_npu` 与 CANN;Swift 负责让你少写胶水代码。

### 1.2 对标英伟达生态(迁移心智图)

| 能力维度 | 昇腾(国产化) | 英伟达(CUDA 世界) | 一句话说明 |
| --- | --- | --- | --- |
| 加速硬件 | 昇腾 NPU(达芬奇架构) | GPU(SM/CUDA Core/Tensor Core) | 算力载体 |
| 底层计算架构 | CANN | CUDA | 驱动 + 运行时 + 编译器 |
| 算子/数学库 | CANN 算子库 | cuDNN / cuBLAS | 卷积/矩阵等高性能算子 |
| 框架适配层 | torch_npu(PyTorch 昇腾插件) | 原生 CUDA 后端 | 让 PyTorch 用上加速硬件 |
| 集合通信 | HCCL | NCCL | 多卡 AllReduce/AllGather 等 |
| 模型仓库/社区 | **ModelScope Hub(魔搭)** | Hugging Face Hub | 模型与数据集托管 |
| 模型加载库 | **modelscope library** | huggingface_hub / transformers | 拉模型、加载权重 |
| 训练/微调框架 | **ms-swift** | transformers Trainer + peft + trl / LLaMA-Factory | SFT/LoRA/DPO 一体化 |
| 推理后端 | **swift infer/deploy**(对接 vLLM-ascend / pt 后端) | vLLM / TensorRT-LLM | 高吞吐推理服务 |
| 量化工具 | msmodelslim(配合) | GPTQ / AWQ 工具链 | 权重量化 |

**心智图一句话**:把 GPU 上的 `HF Hub → transformers/peft/trl → vLLM` 这条链,平移成 NPU 上的 `ModelScope Hub → ms-swift → swift deploy`。Swift 同时兼容从 HF 下载,所以迁移成本主要在"设备后端"而非"工作流"。

## 2. ModelScope 生态全景

| 组件 | 角色 | 对标 |
| --- | --- | --- |
| ModelScope Hub | 在线模型/数据集仓库,国内访问快 | Hugging Face Hub |
| modelscope library | Python 库,`snapshot_download`/`pipeline` 拉模型与跑推理 | huggingface_hub + transformers pipeline |
| **ms-swift** | 训练/微调/推理的统一框架与 CLI(本文重点) | LLaMA-Factory / axolotl |
| ModelScope Studio | 在线创空间(Gradio 应用托管) | HF Spaces |

ms-swift 的价值在于:**一套命令覆盖 100+ 模型族 × 多种训练方法(全参/LoRA/QLoRA/DPO/ORPO/...)× 多种硬件(GPU/NPU)**。对昇腾用户而言,它把"在 NPU 上微调一个国产大模型"从"自己改一堆设备代码"降级成"换几个参数"。

## 3. ms-swift 在 NPU 上的工作机制

Swift 跑在 NPU 上的本质,是**让上层训练逻辑设备无关,把设备差异下沉给 torch_npu**:

```
  你的命令(swift sft ...)
        │
        ▼
  ms-swift:模型注册表 / 数据预处理 / Trainer 封装 / 并行配置
        │  调用标准 PyTorch API(.to(device)、autograd、optimizer)
        ▼
  torch_npu:把 CUDA 语义映射到 NPU(device='npu'、AllReduce 走 HCCL)
        │
        ▼
  CANN:图编译 + 算子下发 + Runtime 调度 + HCCL 通信
        │
        ▼
  昇腾 NPU:达芬奇 Cube 单元做矩阵乘,Vector 单元做激活/归一化
```

要点:
- **设备标识从 `cuda` 变成 `npu`**。Swift 在 NPU 环境下会把张量与模型放到 `npu` 设备;底层由 torch_npu 接管。
- **多卡训练通信走 HCCL**,等价于 GPU 上的 NCCL。分布式启动方式、`world_size` 概念一致,只是后端不同。
- **算子缺失时的回退**:某些前沿算子(如特定 attention/融合算子)在 CANN 上不一定有等价实现,可能回退到通用实现,这是性能差异的常见来源(见第 6 节)。
- 具体环境依赖(torch_npu 与 CANN 版本配套、镜像)以华为昇腾官方文档(Ascend 社区)与 ms-swift 仓库《NPU 推理与微调最佳实践》为准。

## 4. NPU 微调流程(全链路含义)

```
[1] 准备环境          [2] 取模型/数据        [3] 配置训练
昇腾驱动+CANN+       从 ModelScope Hub      选方法:LoRA/QLoRA/
torch_npu+ms-swift   下载模型与数据集       全参;设并行与超参
     │                     │                      │
     └──────────┬──────────┴───────────┬──────────┘
                ▼                       ▼
        [4] 拉起训练(swift sft)   设备=npu,通信=HCCL
                │
                ▼
        [5] 产出权重 / LoRA 适配器
                │
                ▼
        [6] (可选)合并权重 → 推理/部署(第 5 节)
```

每一步"为什么":
1. **环境**:NPU 与 GPU 最大差异在驱动与计算架构。必须先装好昇腾驱动 + 固件 + CANN,再装匹配的 torch_npu,最后装 ms-swift。**三者版本必须配套**——这是 NPU 上最常见的故障根源。具体版本与命令以华为昇腾官方文档(Ascend 社区)为准。
2. **取模型/数据**:用 ModelScope 下载在国内更快;Swift 也支持从 HF 拉,通过环境变量切换下载源。
3. **配置训练**:显存/HBM 紧张时优先 LoRA/QLoRA;多卡时配置并行(数据并行 + ZeRO 切分等),通信由 HCCL 承担。
4. **拉起训练**:核心是 `swift sft`。在 NPU 上需通过昇腾的分布式启动方式拉起多进程,每进程绑定一张 NPU。
5. **产出**:LoRA 只保存小体量适配器;全参保存完整权重。
6. **部署**:LoRA 可合并回基座再部署,或推理时动态挂载。

> 护栏:以上为**流程语义**;精确命令行、参数名、版本号一律以 ms-swift 官方《NPU 推理与微调最佳实践》文档为准。

## 5. NPU 推理与部署流程

Swift 提供两条路:
- `swift infer`:**本地交互/批量推理**,快速验证微调效果。
- `swift deploy`:**起一个 OpenAI 兼容的 API 服务**,对标 vLLM 的在线服务能力。

后端选择(机制层面):

```
            swift deploy / infer
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
   PyTorch(pt)后端        vLLM-ascend 后端
   通用、兼容性好          高吞吐、PagedAttention
   适合验证/小流量         适合生产/高并发
```

- **pt 后端**:直接用 transformers 风格逐 token 生成,稳但吞吐低。
- **vLLM-ascend**:vLLM 的昇腾适配版,带 continuous batching 与 PagedAttention,吞吐显著更高,但对模型/算子支持范围有限制。是否可用取决于该模型在昇腾上的 vLLM 适配进度——以官方文档为准。

## 6. 迁移要点与常见坑

**从 GPU 脚本搬到 NPU,你主要要改这些:**

| 类别 | GPU 习惯 | NPU 需要做的 |
| --- | --- | --- |
| 设备 | `device='cuda'` | `device='npu'`(Swift 多数自动处理) |
| 通信后端 | NCCL | HCCL(框架自动切换,但要装对 CANN) |
| 启动方式 | torchrun | 昇腾分布式启动(进程绑卡) |
| 依赖 | torch + CUDA | torch + torch_npu + CANN(三者配套) |
| 镜像 | nvidia/cuda 基础镜像 | 昇腾官方镜像(含 CANN) |

**高频坑(机制层面):**
1. **版本不配套**:torch_npu × CANN × 驱动固件三者版本必须严格匹配,错配会出现 `import torch_npu` 失败或算子加载失败。**这是第一大坑**。
2. **算子未适配 / 回退**:个别融合算子在 CANN 上缺等价实现,导致回退到慢路径或直接报错。表现为"GPU 跑得通、NPU 报算子不支持"。
3. **首次执行慢(图编译)**:昇腾走图编译,首个 step / 首次 shape 变化会触发编译,耗时长属正常;之后命中缓存即提速。不要误判为卡死。
4. **shape 动态化代价高**:NPU 对静态 shape 更友好,频繁变长 seq 会反复触发编译。可考虑分桶/padding 到固定长度。
5. **HBM 与显存语义差异**:OOM 信息形态不同,排查思路一致(减 batch、开梯度检查点、用 LoRA/量化)。
6. **下载源**:默认从 HF 可能慢/不可达,记得切到 ModelScope 源(环境变量,具体名称以官方文档为准)。

**性能调优思路(机制层面,非具体数字):**
- 优先让计算落在**静态 shape + 已适配的融合算子**路径上,避免回退。
- 多卡用 HCCL 时关注**通信拓扑**(参考 [[ai-infra/网络/集合通信原语]]),AllReduce 占比高时考虑梯度切分/重叠通信与计算。
- 训练吃 HBM 时,LoRA/QLoRA + 梯度检查点是性价比最高的组合。

## 常见问题

| 问题 | 简答 |
| --- | --- |
| ModelScope 和 Hugging Face 啥关系? | 定位相同(模型/数据社区),国内访问更快;Swift 两边都能拉。 |
| ms-swift 对标 GPU 上哪个工具? | 最接近 LLaMA-Factory:一个 CLI 覆盖 SFT/LoRA/DPO + 推理部署。 |
| NPU 上 Swift 性能能等于 GPU 吗? | 取决于算子适配度;适配好的路径接近,踩到回退算子会慢。 |
| 一定要用 MindSpore 吗? | 不用。Swift 走 PyTorch + torch_npu 路线,沿用 PyTorch 生态。 |
| 微调后怎么部署? | LoRA 合并或动态挂载,用 `swift deploy` 起 OpenAI 兼容服务。 |
| 推理用 pt 还是 vLLM? | 验证用 pt;高并发生产用 vLLM-ascend(看模型是否已适配)。 |
| 具体命令和版本去哪查? | ms-swift 仓库《NPU 推理与微调最佳实践》+ 华为昇腾官方文档(Ascend 社区)。 |

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

---

参考(原始链接,保留):
- ms-swift 仓库:https://github.com/modelscope/swift
- NPU 推理与微调最佳实践:https://github.com/modelscope/swift/blob/main/docs/source/LLM/NPU%E6%8E%A8%E7%90%86%E4%B8%8E%E5%BE%AE%E8%B0%83%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md
