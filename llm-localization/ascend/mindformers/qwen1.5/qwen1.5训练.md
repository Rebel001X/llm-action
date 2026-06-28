# 昇腾 MindFormers 训练 Qwen1.5

> 在华为昇腾 NPU 上, 用 MindFormers 套件完成 Qwen1.5 系列(0.5B/1.8B/4B/7B/14B/72B)的预训练、增量预训练与微调全流程。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/megatron-lm/README]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]] [[llm-algo/transformer/模型架构]]

## 阅读地图

| 你想知道的 | 看哪一节 |
| --- | --- |
| MindFormers 是什么、在昇腾栈哪一层 | §0 §1 |
| 昇腾生态 ↔ 英伟达生态怎么对应 | §1 对照表 |
| 软件栈纵向是怎么叠起来的 | §2 |
| 一次训练从准备到产出经历哪些步骤 | §3 |
| 并行切分(DP/MP/PP)在昇腾上怎么配 | §4 |
| 静态图(GRAPH)为什么比动态图快 | §5 |
| 权重在 HF 格式与 MindSpore 格式之间怎么转 | §6 |
| 从 PyTorch/Megatron 迁移要改什么、坑在哪 | §7 迁移要点与坑 |
| 常见报错与排查 | 常见问题表 |

## 0. 一句话锚点

**MindFormers = 昇腾上的 "Megatron + HuggingFace Transformers" 合体**: 它把 Qwen1.5 这类主流大模型的结构、并行切分、训练/微调/推理流程都做成了配置驱动(YAML)的开箱即用套件, 底层跑在 MindSpore 框架 + CANN 算子库 + 昇腾 NPU 硬件上。你的核心工作不是写训练循环, 而是**改 YAML、转权重、配并行、起任务**。

## 1. 地基: 在昇腾栈的定位 + 对标英伟达生态

昇腾软件栈自底向上分四层, MindFormers 处在最上层的"套件层":

```
┌──────────────────────────────────────────────┐
│  套件层    MindFormers (Qwen1.5/LLaMA/GLM... )  │ ← 你在这里写 YAML
├──────────────────────────────────────────────┤
│  框架层    MindSpore (静态图/动态图、自动微分)    │
├──────────────────────────────────────────────┤
│  异构计算   CANN (算子库 + 图编译器 GE + HCCL)    │
├──────────────────────────────────────────────┤
│  硬件层    昇腾 NPU (达芬奇架构: Cube/Vector)     │
└──────────────────────────────────────────────┘
```

### 昇腾 ↔ 英伟达 生态对照表(迁移心智图)

| 层次 | 昇腾(华为) | 英伟达(CUDA 世界) | 说明 |
| --- | --- | --- | --- |
| 训练/微调加速卡 | 昇腾 NPU(Ascend 910 系列) | GPU(A100/H100) | 算力硬件 |
| 芯片微架构 | 达芬奇架构 Cube/Vector 单元 | Tensor Core + CUDA Core | 矩阵+向量计算单元 |
| 异构计算软件栈 | CANN | CUDA | 驱动+运行时+编译器总集 |
| 算子/数学库 | CANN 算子库(AOL) | cuDNN / cuBLAS | 卷积/矩阵乘等高性能 kernel |
| 集合通信库 | HCCL | NCCL | AllReduce/AllGather 等原语 |
| 深度学习框架 | MindSpore | PyTorch | 张量+自动微分+图执行 |
| 大模型训练套件 | MindFormers | Megatron-LM / Transformers | 模型库+并行+训练流程 |
| 推理引擎 | MindIE | TensorRT-LLM / vLLM | 部署侧高性能推理 |
| 量化压缩工具 | msModelSlim | GPTQ / AWQ 工具链 | 权重量化 |
| 训练框架(另一支) | ModelLink | Megatron-LM | PyTorch+昇腾的训练方案 |

> 一句话记忆: **NPU↔GPU、CANN↔CUDA、HCCL↔NCCL、MindSpore↔PyTorch、MindFormers↔Megatron/HF**。看到 CUDA 世界的某个组件, 在昇腾世界基本都能找到一个对位件。

## 2. 纵向软件栈: 一条指令是怎么落到 NPU 的

```
你的 YAML 配置 (Qwen1.5-7B、并行度、学习率)
        │  MindFormers 解析配置, 构建模型与训练流程
        ▼
MindSpore 计算图 (前向 + 反向 + 优化器更新)
        │  GRAPH 模式: 整图编译优化(算子融合、内存复用)
        ▼
CANN / GE 图编译 → 切分为算子序列
        │  矩阵乘下发到 Cube, 激活/归一化下发到 Vector
        │  跨卡通信下发到 HCCL
        ▼
昇腾 NPU 达芬奇核 执行
```

关键点: MindFormers 不直接碰硬件, 它生成的是 MindSpore 的网络定义; 真正的"算子怎么在 Cube/Vector 上跑、多卡怎么通信"由 CANN 与 HCCL 负责。这与 PyTorch→cuDNN→GPU 的分工完全同构。

## 3. Qwen1.5 训练完整流程(步骤含义, 非逐字命令)

> ⚠️ 凡涉及精确镜像名/版本号/命令行, 一律以华为昇腾官方文档(Ascend 社区)与 MindFormers 仓库(gitee.com/mindspore/mindformers, 见文末参考)为准。下面只讲**每一步在做什么、为什么需要、容易踩什么坑**。

```
① 环境就位      ② 取模型与配置   ③ 数据准备
   CANN+MS+MF        Qwen1.5 YAML     原始语料→MindRecord
        \                |               /
         \               |              /
          ▼              ▼             ▼
        ④ 权重转换 (HF ckpt → MindSpore ckpt)
                         │
                         ▼
        ⑤ 配并行 (DP/MP/PP/优化器并行) + 改 YAML
                         │
                         ▼
        ⑥ 拉起分布式训练 (msrun / 多机 rank table)
                         │
                         ▼
        ⑦ 监控 loss/吞吐 → 保存 ckpt → 评估/转回 HF
```

### ① 环境就位
昇腾训练环境是一条**依赖链**: 固件与驱动(Driver/Firmware)→ CANN(含 toolkit、kernels)→ MindSpore → MindFormers, 版本必须**两两配套**。社区一般提供配套好的 Docker 镜像(如文末参考中的 `mindformers1.1_mindspore2.3rc2:...` 这类镜像), 直接拉取可省去手动对版本的痛苦。
- **坑**: 最高频的失败原因就是版本错配(CANN 与 MindSpore、MindSpore 与 MindFormers 不匹配), 表现为算子不存在、图编译失败。先用官方配套表对齐版本, 再动手。

### ② 取模型与配置
MindFormers 为 Qwen1.5 各规模提供独立的 YAML(`research/qwen1_5/` 目录), 涵盖 7B/14B/72B 等。YAML 里定义了模型结构(层数、隐藏维度、词表)、训练超参、并行策略、数据路径。**训练即改 YAML**, 这是 MindFormers 的核心范式。

### ③ 数据准备
原始文本要先经过 tokenizer, 再打包成 MindSpore 的 **MindRecord** 二进制格式(类比 Megatron 的 `.bin/.idx` 预处理产物)。这样训练时 IO 高效、可被数据并行均匀切分。
- **坑**: tokenizer 必须与 Qwen1.5 原始词表严格一致, 否则 token id 错位导致 loss 不收敛。

### ④ 权重转换
HuggingFace 上的 Qwen1.5 权重是 PyTorch 格式(`.safetensors`/`.bin`), 而 MindSpore 用 `.ckpt`。MindFormers 提供转换脚本, 完成两件事: **(a) 张量格式转换**, **(b) 参数命名映射**(HF 的 `model.layers.x.self_attn.q_proj` ↔ MindSpore 的命名)。
- **坑**: 转换方向要对(训练前 HF→MS, 产出后可 MS→HF 回流); QKV 是否合并、是否有 bias、RoPE 实现差异都会影响映射, 用官方脚本不要手写。

### ⑤ 配并行
见 §4。这是大模型训练最需要经验的一步: 模型放不下单卡就得切。

### ⑥ 拉起分布式训练
昇腾多卡训练靠**集群信息**协调: 单机多卡用 MindSpore 的 `msrun` 启动器一键拉起; 多机则需要 **rank table**(描述每张 NPU 的 IP、device_id、rank_id, 即设备拓扑), HCCL 据此建立通信域。
- **坑**: 多机训练 rank table 配错(IP/device 顺序错)是经典翻车点, 表现为 HCCL 初始化卡死或超时。

### ⑦ 监控与产出
观察 loss 曲线与吞吐(tokens/s), 定期保存 ckpt 断点续训; 训练完成后可评估, 并按需把 `.ckpt` 转回 HF 格式供下游推理(如 MindIE)使用。

## 4. 并行策略: 在昇腾上怎么切 Qwen1.5

Qwen1.5-72B 单卡装不下, 必须组合并行。MindFormers 通过 YAML 中的 `parallel_config` 配置, 概念与 Megatron 一一对应:

| 并行方式 | 含义 | 对标 Megatron |
| --- | --- | --- |
| 数据并行 DP | 复制模型, 切分数据 batch | Data Parallel |
| 张量并行 MP/TP | 把单层权重(如 QKV、FFN)切到多卡 | Tensor Parallel |
| 流水线并行 PP | 把不同层分到不同卡, 流水执行 | Pipeline Parallel |
| 优化器并行 | 切分优化器状态(类 ZeRO) | ZeRO / 优化器状态分片 |

```
        ┌── 张量并行(层内切) ──┐
卡0 ─ 卡1 |  一层权重切两半       |   ← 通信密集, 放同一节点(NVLink 类比: 昇腾用高速互联)
        └─────────────────────┘
卡2 ─ 卡3   流水线下一段 (PP stage 2)   ← 段间只传激活, 通信少, 可跨节点
   ↑
   └ 数据并行: 整个上面的组合再复制 N 份, 各喂不同数据, 末尾 AllReduce 梯度(HCCL)
```

**调优心法(机制层)**: 张量并行通信最重 → 优先放节点内(占满单机 NPU 互联带宽); 流水线并行通信轻 → 可跨机; 数据并行的梯度同步走 HCCL AllReduce。切分总卡数 = DP × MP × PP, 三者乘积要等于实际 NPU 数。

## 5. 静态图(GRAPH)模式: 为什么快

MindSpore 有两种执行模式, 训练大模型强烈建议 **GRAPH(静态图)**:

| 模式 | 机制 | 对标 |
| --- | --- | --- |
| PYNATIVE(动态图) | 逐算子下发, 边算边构图, 易调试 | PyTorch eager |
| GRAPH(静态图) | 先编译整图再执行, 做算子融合/内存复用/并行优化 | `torch.compile` / TF graph / CUDA Graph 思想 |

GRAPH 模式下, CANN 的图编译器(GE)能看到完整计算图, 从而把多个小算子**融合**成一个大 kernel(减少下发开销与中间显存)、复用内存、自动安排通信与计算重叠。代价是首次编译耗时长、动态 shape 支持弱、报错栈不如动态图直观。
- **坑**: GRAPH 模式不能像 Python 那样随意 print/断点; 调试期可临时切 PYNATIVE 定位逻辑问题, 跑大规模训练再切回 GRAPH 拿性能。

## 6. 权重格式与转换(HF ↔ MindSpore)

```
HuggingFace (PyTorch)              MindFormers (MindSpore)
 model-00001.safetensors    ──►     qwen1_5_7b.ckpt
 config.json / tokenizer            YAML 中的 model 配置
        │   ① 张量数值: torch.Tensor → ms.Parameter
        │   ② 命名映射: q_proj/k_proj/... → attention.* 命名
        │   ③ 结构差异: QKV 合并/RoPE/RMSNorm 实现对齐
        ▼
        官方转换脚本(双向)
```

理解这三步(数值、命名、结构)就能看懂任何"为什么转换后 loss 异常"的问题——几乎都是命名映射漏了某个参数, 或结构假设(是否有 bias、是否合并 QKV)不一致。

## 迁移要点 / 注意事项与坑(从 PyTorch/Megatron 来的同学)

1. **心智迁移**: 不要找"训练脚本", 找 **YAML 配置**。MindFormers 是配置驱动, 改 YAML 等价于改 Megatron 的命令行参数 + 模型定义。
2. **API 不是 PyTorch**: MindSpore 的张量/网络 API 命名与 PyTorch 不同(如 `nn.Cell` ≈ `nn.Module`), 但 Qwen1.5 已封装好, 一般无需手写网络层。
3. **数据格式必须转**: PyTorch 的 dataset → MindRecord, 这一步不可省。
4. **通信库换名**: 你心里的 NCCL = 这里的 HCCL; AllReduce/AllGather 原语语义一致, 但 rank table / 通信域建立方式不同。
5. **版本对齐是第一杀手**: Driver↔CANN↔MindSpore↔MindFormers 四级配套, 错一级就编译失败。优先用官方配套 Docker 镜像。
6. **GRAPH 模式调试反直觉**: 报错往往指向编译后的图而非源码行, 先在 PYNATIVE 下定位逻辑, 再切 GRAPH 跑性能。
7. **并行度乘积要对齐卡数**: DP×MP×PP 必须等于参与训练的 NPU 总数, 否则启动即报错。
8. **多机靠 rank table**: 跨节点训练的拓扑描述文件是高频翻车点, 务必核对 IP 与 device 顺序。
9. **精度对齐验证**: 迁移后先用小规模/少步数跑通, 对比 loss 量级与 HF 基线, 再放大规模, 避免烧大量算力后才发现转换出错。

> ⚠️ 再次强调: 本文不给精确命令、镜像 tag、版本号与路径。安装/Docker/启动的**确切命令与版本以华为昇腾官方文档(Ascend 社区)及 MindFormers 仓库为准**(见文末参考)。

## 常见问题

| 问题 | 可能原因 | 排查方向 |
| --- | --- | --- |
| 图编译报"算子不存在/不支持" | CANN 与 MindSpore 版本不配套 | 用官方配套表对齐版本, 或换配套 Docker 镜像 |
| HCCL 初始化卡死/超时(多机) | rank table 的 IP/device 顺序错 | 核对每张卡的 ip/device_id/rank_id 拓扑 |
| 转换权重后 loss 不收敛/为 NaN | 参数命名映射漏项或结构假设不符 | 用官方转换脚本; 检查 QKV 合并、bias、RoPE |
| loss 量级异常但能跑 | tokenizer/词表与原模型不一致 | 确认用 Qwen1.5 原始 tokenizer 重新生成 MindRecord |
| 启动即报并行度错误 | DP×MP×PP ≠ NPU 总数 | 调整 parallel_config 使乘积等于实际卡数 |
| 显存 OOM | 张量/流水线并行切分不足或 batch 过大 | 增大 MP/PP、开优化器并行、减 micro-batch、开重计算 |
| GRAPH 模式难调试 | 静态图不支持随意打印断点 | 临时切 PYNATIVE 定位逻辑, 再切回 GRAPH |
| 首个 step 极慢 | GRAPH 模式首次整图编译 | 正常现象, 后续 step 恢复正常吞吐 |

## 参考

- MindFormers Qwen1.5 说明: https://gitee.com/mindspore/mindformers/blob/r1.0/research/qwen1_5/qwen1_5.md
- 配套 Docker 镜像示例(版本以官方为准): `swr.cn-central-221.ovaijisuan.com/mindformers/mindformers1.1_mindspore2.3rc2:...`
- 确切命令、版本、镜像 tag 一律以**华为昇腾官方文档(Ascend 社区)** 与上述仓库为准。

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
