# ModelLink:昇腾大模型分布式训练套件

> 一句话定位:ModelLink(后演进为 MindSpeed-LLM)是华为昇腾官方推出的、基于 PyTorch + Megatron-LM 的大模型分布式训练套件,是昇腾世界里对标 NVIDIA Megatron-LM 的训练框架。📍 导航:[[00-知识地图]]
> 🔗 相关:[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/megatron-lm/README]] [[llm-train/README]]

## 阅读地图

| 小节 | 你会得到什么 | 适合谁 |
|------|--------------|--------|
| 0. 一句话锚点 | ModelLink 是什么、不是什么 | 所有人 |
| 1. 地基与定位 | 昇腾软件栈分层 + 昇腾↔英伟达对照表 | 从 CUDA/Megatron 迁移者 |
| 2. 整体架构 | ModelLink 如何"嫁接"Megatron 到 NPU | 框架使用者 |
| 3. 并行机制 | TP/PP/DP/SP/EP 在昇腾上的落地 | 分布式训练工程师 |
| 4. 训练全流程 | 权重转换→预处理→训练→评估→推理 | 实操者 |
| 5. 环境与镜像 | Docker/CANN/依赖关系(讲含义不背命令) | 部署者 |
| 6. 迁移要点与坑 | 从 Megatron-GPU 搬到 NPU 改什么、踩什么 | 迁移者 |
| 7. 常见问题 | 快速排错 | 所有人 |

## 0. 一句话锚点

**ModelLink 让你"几乎不改训练脚本"地把 Megatron-LM 风格的大模型预训练 / 微调任务跑在昇腾 NPU 集群上。**

- 它**不是**一个全新的训练框架,而是 Megatron-LM 的**昇腾适配层 + 模型库 + 工具链**。
- 你写的依然是 Megatron 那套 `--tensor-model-parallel-size`、`--pipeline-model-parallel-size` 的参数;底层算子、通信、设备从 CUDA/NCCL 换成了 CANN/HCCL。
- 定位上等价于:**"昇腾版的 Megatron-LM + 一批开箱即用的国产/开源大模型配置 + 权重互转工具"**。

> 命名提示:ModelLink 是早期名称,昇腾后续将其整合进 **MindSpeed-LLM**(MindSpeed 套件的 LLM 子集)。本文用 ModelLink 泛指这一训练套件,具体仓库名/版本以华为昇腾官方文档(Ascend 社区 / Gitee 仓库)为准。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(ModelLink 在哪一层)

```
┌──────────────────────────────────────────────────────────┐
│  应用层:你的预训练/SFT/RLHF 任务脚本                       │
├──────────────────────────────────────────────────────────┤
│  训练套件层:★ ModelLink / MindSpeed-LLM ★  ← 本文主角     │
│      = Megatron-LM 适配 + 模型库 + 权重转换/数据预处理     │
├──────────────────────────────────────────────────────────┤
│  加速库层:MindSpeed(并行/融合算子/显存优化的底座)        │
├──────────────────────────────────────────────────────────┤
│  框架层:PyTorch + torch_npu(Ascend Extension for PyTorch)│
├──────────────────────────────────────────────────────────┤
│  异构计算架构:CANN(算子库 + 图编译 + HCCL 通信 + Runtime)│
│      AscendCL / aclnn 算子 / GE 图引擎 / HCCL              │
├──────────────────────────────────────────────────────────┤
│  驱动 + 固件:Driver / Firmware(npu-smi 管理)            │
├──────────────────────────────────────────────────────────┤
│  硬件:昇腾 NPU(910B/910C 等,达芬奇架构 Cube+Vector)     │
└──────────────────────────────────────────────────────────┘
```

关键认知:**ModelLink 不直接碰硬件**。它通过 `torch_npu` 把 PyTorch 的张量与算子调用重定向到 NPU,再由 CANN 把算子下发到达芬奇核心,集合通信交给 HCCL。ModelLink 自己负责的是"大模型训练的工程组装"——并行切分、权重格式、数据流水。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 能力维度 | 英伟达(NVIDIA)世界 | 昇腾(Ascend)世界 | 说明 |
|----------|----------------------|---------------------|------|
| 加速硬件 | GPU(A100/H100) | NPU(910B/910C) | 达芬奇 Cube 单元 ≈ Tensor Core |
| 计算架构/运行时 | CUDA + CUDA Runtime | CANN + AscendCL | 编程与下发的底座 |
| 算子库 | cuDNN / cuBLAS | CANN 算子(aclnn/TBE) | 卷积/GEMM 等高性能 kernel |
| 深度学习框架 | PyTorch(原生) | PyTorch + **torch_npu** | torch_npu 是关键适配层 |
| 集合通信 | NCCL | **HCCL** | AllReduce/AllGather 等 |
| 通信后端名 | `nccl` | `hccl` | `init_process_group` 的 backend |
| 大模型训练框架 | **Megatron-LM** | **ModelLink / MindSpeed-LLM** | 本文主角,接口高度相似 |
| 训练加速库 | Megatron-Core / TE | **MindSpeed** | 融合算子、并行、显存优化 |
| 推理引擎 | TensorRT-LLM / vLLM | **MindIE** | 部署侧 |
| 量化工具 | GPTQ / AWQ / TensorRT 量化 | **msModelSlim** | 压缩侧 |
| 设备查询/监控 | `nvidia-smi` | `npu-smi` | 卡状态/利用率 |
| 设备节点 | `/dev/nvidia*` | `/dev/davinci*` | 容器透传时需挂载 |
| 混合精度 | AMP / bf16 / fp16 | bf16 / fp16(NPU 支持) | 910B 对 bf16 友好 |

> 一句话记忆:**CUDA→CANN、NCCL→HCCL、Megatron→ModelLink、TensorRT-LLM→MindIE**。把这四对刻进脑子,迁移时就知道"谁替代了谁"。

## 2. 整体架构:ModelLink 如何把 Megatron 嫁接到 NPU

ModelLink 的工程巧思在于"**最大程度复用 Megatron 的代码与心智,最小程度暴露昇腾差异**"。

```
            你的训练脚本(Megatron 风格参数)
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   ModelLink / MindSpeed-LLM        │
        │   - 模型定义(LLaMA/Qwen/...)      │
        │   - 权重转换(HF ↔ Megatron 切分)  │
        │   - 数据预处理(打包成 idx/bin)    │
        │   - 训练入口(pretrain/finetune)   │
        └───────────────┬───────────────────┘
                        │  patch / 适配
                        ▼
        ┌───────────────────────────────────┐
        │   Megatron-LM(并行框架骨架)       │
        │   TP / PP / DP / SP / EP 切分逻辑   │
        └───────────────┬───────────────────┘
                        │  融合算子 / 显存优化
                        ▼
        ┌───────────────────────────────────┐
        │   MindSpeed(昇腾训练加速库)       │
        │   FlashAttention 类融合、重计算等  │
        └───────────────┬───────────────────┘
                        │  PyTorch 算子重定向
                        ▼
        ┌───────────────────────────────────┐
        │   PyTorch + torch_npu              │
        └───────────────┬───────────────────┘
                        ▼
              CANN(算子/图/HCCL/Runtime)
                        ▼
                 昇腾 NPU(达芬奇核)
```

要点:
- **Megatron-LM 被当作"骨架"依赖**(本目录现有片段里就有从 NVIDIA Megatron-LM 仓库安装 `megatron-core` 的步骤)。ModelLink 在其上做 patch/封装,让并行逻辑落到 NPU。
- **MindSpeed** 提供昇腾侧的融合算子(如 FlashAttention 的 NPU 实现)、选择性重计算、序列并行优化等,是性能的关键来源。
- **torch_npu** 是 PyTorch 的"昇腾插件",把 `torch.cuda.*` 语义映射到 `torch.npu.*` / NPU 设备。

## 3. 并行机制:TP/PP/DP/SP/EP 在昇腾上的落地

ModelLink 沿用 Megatron 的四/五维并行,概念**一一对应**,只是底层通信走 HCCL。

| 并行维度 | 含义 | 主要通信原语 | 昇腾侧承载 |
|----------|------|--------------|------------|
| DP 数据并行 | 不同卡跑不同数据,梯度同步 | AllReduce | HCCL |
| TP 张量并行 | 单层权重切到多卡 | AllReduce / AllGather | HCCL(对带宽最敏感) |
| PP 流水并行 | 不同层放不同卡,微批流水 | P2P Send/Recv | HCCL |
| SP 序列并行 | 切序列维,降激活显存 | ReduceScatter/AllGather | HCCL |
| EP 专家并行 | MoE 专家分布到多卡 | All-to-All | HCCL |

### 3.1 为什么 HCCL 的拓扑很关键

TP 通信量最大且在前向/反向的关键路径上,因此**TP 组通常放在同一台机器内**(走机内高速互联,如 HCCS/卡间总线),PP/DP 跨机(走 RoCE 网络)。这与 NVIDIA 上"TP 放 NVLink 内、PP/DP 走 IB"的布局思路完全同构。

```
单机 8 卡(TP=8 示意):
 NPU0─NPU1─NPU2─NPU3
   │    │    │    │      ← 机内高速互联(HCCS),HCCL 走环/全连接
 NPU7─NPU6─NPU5─NPU4
 TP 组在机内 ──► 通信最快
 PP/DP 跨机 ──► 走 RoCE 网络(需正确配置 HCCL 网卡/IP)
```

迁移含义:**机内 TP、跨机 PP/DP** 的并行布局原则不变;变的是要保证 HCCL 能正确发现机内互联与机间网卡(这是昇腾多机训练最常见的配置坑,见第 6 节)。

## 4. 训练全流程(五步骤的"含义",不是背命令)

ModelLink 的标准工作流分五步,理解每步"为什么"比记命令更重要:

```
①权重转换 ──► ②数据预处理 ──► ③启动训练 ──► ④评估 ──► ⑤推理/导出
  HF格式       原始语料         分布式          指标       回转HF/部署
  ↕切分        →token idx/bin   pretrain/SFT             给 MindIE
```

1. **权重格式转换(HF ↔ Megatron)**:Hugging Face 的权重是"整块"的,Megatron 训练需要按 TP/PP **切分后的分片**权重。ModelLink 提供转换脚本完成 `HF → Megatron(按你的 TP/PP 切)` 以及训练完 `Megatron → HF` 的回转。**坑**:转换时指定的 TP/PP 必须和训练时一致,否则加载报形状不匹配。

2. **数据预处理**:把原始文本 tokenize 并打包成 Megatron 的 `.idx/.bin` 索引格式,训练时可高效 mmap 读取。需指定与模型匹配的 tokenizer。

3. **启动训练**:用 Megatron 风格参数声明并行度(TP/PP/DP)、序列长度、global/micro batch、学习率调度等。多机时用分布式启动器拉起,每个 rank 绑定到一个 NPU(`/dev/davinci*`)。

4. **评估**:在验证集上算 loss / 下游任务指标,确认收敛与正确性。

5. **推理 / 导出**:可直接用套件做简单生成验证;生产部署则把权重回转为 HF 或对接 **MindIE** 推理引擎。

> 涉及具体命令、脚本名、参数默认值、CANN/torch_npu/镜像版本号:**一律以华为昇腾官方文档(Ascend 社区 / 对应 Gitee 仓库 README)为准**,本文只讲流程含义与依赖关系,不杜撰精确命令与版本。

## 5. 环境与镜像:讲依赖关系,不背命令

昇腾训练环境是一条**自底向上的依赖链**,任何一层版本错配都会导致算子报错或通信失败:

```
驱动 Driver / 固件 Firmware(宿主机装)
        │  必须与硬件型号匹配
        ▼
CANN(toolkit + kernels)
        │  CANN 版本决定可用算子集
        ▼
torch_npu(必须与 torch 版本 + CANN 版本三方匹配)
        │
        ▼
ModelLink / MindSpeed / Megatron-core(套件层)
```

容器化要点(本目录现有片段正是在做这件事):
- **基础镜像**:昇腾官方提供带 CANN + PyTorch + torch_npu 的预装镜像(如 `pytorch...cann...` 镜像),省去逐层装的痛苦。
- **设备透传**:`docker run` 要把 `/dev/davinci0..N`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` 等设备节点透传进容器,并挂载宿主机 `Driver/Firmware/npu-smi/dcmi`。**少挂一个 davinci 卡,容器内就看不到那张 NPU**。
- **多机网络**:多机训练常加 `--network=host`,让容器直接用宿主机网络栈,便于 HCCL 跨机通信。
- **Megatron 依赖**:需安装对应版本的 `megatron-core`(现有片段从 NVIDIA Megatron-LM 仓库按 tag 安装)。

> 镜像名、仓库地址、CANN/PyTorch 具体版本号、`pip` 安装命令:**以官方文档为准**。上面给出的是"每一步为什么要做"与"漏了会怎样",而非可复制的精确版本串。

## 6. 迁移要点与常见坑(从 Megatron-GPU 搬到 NPU)

### 6.1 必须改的几处

| 维度 | GPU 上 | NPU 上要改成 | 备注 |
|------|--------|--------------|------|
| 设备 | `cuda` | `npu` | 经 torch_npu,`device='npu:0'` |
| 通信后端 | `nccl` | `hccl` | `init_process_group(backend='hccl')` |
| 监控命令 | `nvidia-smi` | `npu-smi info` | 看卡占用/温度 |
| 设备可见性 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` | 限定可见 NPU |
| 融合算子 | TE/apex FlashAttn | MindSpeed 提供的 NPU 融合算子 | 开关参数不同 |
| 数据类型 | 常用 fp16 | 910B 上优先 **bf16** | 数值更稳 |

### 6.2 高频坑

1. **三方版本错配**:Driver/Firmware ↔ CANN ↔ torch_npu ↔ Megatron-core 必须配套。报"算子不存在 / undefined symbol"多半是 CANN 与 torch_npu 不匹配 → 用官方配套镜像最省心。
2. **HCCL 多机不通**:跨机训练时 HCCL 找不到正确网卡/IP,卡在通信初始化。需确认机间 RoCE 网络互通、HCCL 网卡环境变量配置正确、防火墙放行。
3. **权重切分与训练并行度不一致**:第 4 步转换时用的 TP/PP 必须等于训练时的 TP/PP,否则加载即报形状错。
4. **首个 step 极慢/疑似卡死**:昇腾首次执行会做**算子编译**(类似 JIT),首步耗时长是正常现象,不是死锁;后续会变快。
5. **容器看不到 NPU**:`/dev/davinci*` 没透传全,或 Driver 没挂载 → 容器内 `npu-smi` 报错。
6. **bf16/fp16 选择**:盲目沿用 GPU 上的 fp16 配置可能数值不稳;910B 上优先 bf16。
7. **`torch.cuda` 残留调用**:迁移代码里残留的硬编码 `torch.cuda.xxx` 需替换为 `torch.npu.xxx`(部分被 torch_npu 兼容,但不要全赌)。

### 6.3 性能调优思路(机制层面)

- **并行布局**:TP 优先放机内(高速互联),PP/DP 跨机,减少慢链路上的通信。
- **重计算(recompute)**:用算力换显存,序列长/模型大时打开选择性重计算。
- **序列并行 SP**:与 TP 搭配,进一步压激活显存。
- **开启 MindSpeed 融合算子**:FlashAttention 类融合能显著降显存、提吞吐。
- **micro-batch 与流水气泡**:增大梯度累积步数、调微批大小以摊薄 PP 气泡。

## 7. 常见问题

| 问题 | 答案 |
|------|------|
| ModelLink 和 Megatron-LM 是什么关系? | ModelLink 以 Megatron-LM 为骨架做昇腾适配,接口高度相似,可视为"昇腾版 Megatron"。 |
| ModelLink 和 MindSpeed-LLM 是一回事吗? | 大体是同一血脉:ModelLink 是早期名,后整合进 MindSpeed-LLM。以官方最新仓库命名为准。 |
| 用 MindSpore 还是 PyTorch? | ModelLink 走 **PyTorch + torch_npu** 路线;MindSpore 路线对应的是 MindFormers。 |
| 训练完怎么部署推理? | 权重回转 HF 或对接 **MindIE** 推理引擎(对标 TensorRT-LLM/vLLM)。 |
| 通信后端为什么是 hccl? | 昇腾的集合通信库是 HCCL,对标 NVIDIA 的 NCCL。 |
| 首步特别慢正常吗? | 正常,昇腾首次执行有算子编译开销,之后变快。 |
| 必须用官方镜像吗? | 强烈建议。三方版本配套很脆弱,官方镜像把 CANN/torch/torch_npu 配好,省大量踩坑。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局总览
- [[ai-infra/算力/昇腾NPU]] — 昇腾硬件与达芬奇架构
- [[ai-infra/ai-hardware/AI芯片软件生态]] — 国产 AI 芯片软件栈对比
- [[ai-infra/ai-hardware/CUDA]] — 对标的英伟达计算架构
- [[ai-infra/网络/NCCL]] — HCCL 对标对象
- [[ai-infra/网络/集合通信原语]] — AllReduce/AllGather/All-to-All
- [[ai-framework/megatron-lm/README]] — ModelLink 的骨架依赖
- [[ai-framework/huggingface-transformers/README]] — HF 权重互转
- [[llm-train/README]] — 大模型训练总览
- [[llm-inference/README]] — 推理侧(MindIE 衔接)
- [[llm-compression/quantization/量化基础]] — msModelSlim 量化
- [[llm-algo/transformer/模型架构]] — Transformer/MoE 结构
