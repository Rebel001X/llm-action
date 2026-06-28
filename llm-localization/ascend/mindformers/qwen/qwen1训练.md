# 在昇腾上用 MindFormers 训练 Qwen-1（通义千问一代）

> 用华为昇腾(Ascend)NPU + MindSpore/MindFormers 套件，端到端把 Qwen-1 系列模型跑起来训练/微调的全景图与迁移指南。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/megatron-lm/README]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]]
> 官方参考：https://gitee.com/mindspore/mindformers/blob/r1.0/research/qwen/qwen.md （具体命令与版本以华为昇腾官方文档 / Ascend 社区为准）

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | MindFormers = 昇腾上的 Megatron+HF |
| 1 | 昇腾软件栈定位 + 昇腾↔英伟达对照表 | CANN/MindSpore/MindFormers 分层 |
| 2 | Qwen-1 模型本身要点 | RoPE / RMSNorm / SwiGLU / tiktoken 词表 |
| 3 | 训练全流程（权重转换→数据→配置→启动） | ckpt 转换、MindRecord |
| 4 | 并行机制（DP/MP/PP + 图模式） | sharding 策略、自动并行 |
| 5 | 从 HF/Megatron 迁移到 MindFormers 的要点 | 权重映射、词表对齐 |
| 6 | 常见坑与性能调优 | 内存/精度/编译/通信 |
| - | 常见问题表 + 跳转链接 | FAQ |

## 0. 一句话锚点

**MindFormers 之于昇腾，约等于 "Megatron-LM + HuggingFace Transformers" 之于英伟达。**
它是构建在 MindSpore 框架之上的大模型套件，内置了一批主流大模型（含 Qwen、LLaMA、GLM、Baichuan 等）的标准实现、分布式并行策略、权重转换脚本与训练/推理 pipeline。你在 GPU 上用 `transformers` + `deepspeed`/`megatron` 做的事，在昇腾上换成 MindSpore + MindFormers 做。Qwen-1 是阿里通义千问的第一代（7B/14B 等），用 tiktoken 风格 BPE 词表、RoPE 位置编码、RMSNorm、SwiGLU 这套现代 decoder-only 架构。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

昇腾软件栈从下到上分层，MindFormers 处在最上面的"套件层"：

```
┌───────────────────────────────────────────────┐
│  应用 / 套件层  MindFormers (Qwen 在此)         │ ← 模型库+并行+训练脚本
├───────────────────────────────────────────────┤
│  AI 框架层      MindSpore (亦支持 PyTorch+插件)  │ ← 自动微分/图编译/分布式
├───────────────────────────────────────────────┤
│  异构计算架构    CANN                            │ ← 算子库/图引擎/HCCL/Runtime
│                 (AOL算子 / GE图引擎 / HCCL / ACL)│
├───────────────────────────────────────────────┤
│  硬件层          昇腾 NPU (达芬奇架构 Cube+Vector)│ ← 910 训练 / 310 推理
└───────────────────────────────────────────────┘
```

**这层定位很关键**：当你训练 Qwen-1 报错时，要先判断锅在哪一层——是 MindFormers 配置写错（套件层），还是 MindSpore 图编译失败（框架层），还是 CANN 算子不支持某个 shape（算力库层），还是 HCCL 通信超时（网络层）。分层定位是排障第一直觉。

### 昇腾 ↔ 英伟达生态对照表（迁移心智图）

| 能力 / 角色 | 英伟达生态 | 昇腾生态 | 说明 |
|------------|-----------|---------|------|
| 加速硬件 | GPU (A100/H100) | NPU (昇腾 910 系列) | 910 用于训练，310 偏推理 |
| 底层编程/运行时 | CUDA + CUDA Runtime | CANN + ACL/Runtime | CANN 是昇腾的"CUDA" |
| 高性能算子库 | cuDNN / cuBLAS | CANN 算子库 (AOL/TBE) | 卷积/矩阵乘等融合算子 |
| 集合通信 | NCCL | HCCL | AllReduce/AllGather 等原语 |
| 通信硬件互联 | NVLink / NVSwitch | HCCS / 华为缓存一致总线 | 卡间高带宽互联 |
| 深度学习框架 | PyTorch / TensorFlow | MindSpore (PyTorch 亦可经适配) | 自动微分+图模式 |
| 大模型训练套件 | Megatron-LM / DeepSpeed | MindFormers / ModelLink | 并行+模型库 |
| 模型/权重生态 | HuggingFace Transformers | MindFormers 模型库 | Qwen 实现+ckpt |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 部署侧 |
| 量化工具 | GPTQ / AWQ / llm-compressor | msModelSlim | 压缩量化 |
| 混合精度 | AMP (fp16/bf16) | MindSpore 混合精度 (fp16/bf16) | 自动/手动混精 |
| 性能分析 | Nsight / nvprof | msprof / MindStudio Insight | profiling 工具 |

记住这张表，90% 的"GPU 上怎么做的，昇腾上对应啥"都能查到。

## 2. Qwen-1 模型本身的要点（与训练相关）

Qwen-1 是典型的现代 decoder-only Transformer，几个对训练/迁移有影响的细节：

- **位置编码 RoPE（旋转位置编码）**：无需可学习位置 embedding，外推长度时靠调 base/NTK 缩放；MindFormers 实现里 RoPE 的 dtype 与精度需对齐 HF，否则长序列下数值漂移。
- **归一化 RMSNorm**：相比 LayerNorm 去掉均值中心化，只做 RMS 缩放，省一次规约；昇腾上 RMSNorm 通常有融合算子。
- **激活 SwiGLU**：FFN 用门控线性单元，参数量比标准 FFN 略大（中间维度按比例缩放），加载权重时要对上 gate/up/down 三个投影。
- **词表/分词器**：Qwen-1 用 tiktoken 风格 BPE，词表约 15 万级别（远大于 LLaMA 的 3.2 万），训练前要确认 MindFormers 侧分词器与原始 `qwen.tiktoken` 一致，否则 token id 错位会让 loss 看似正常但学不出东西。
- **注意力**：标准多头注意力（Qwen-1 多数尺寸未用 GQA，GQA 是 Qwen2 起更普遍），QKV 可能合并为一个 fused 投影，权重转换时要正确切分。

> 这些"架构指纹"决定了**权重转换脚本必须严格按层名映射**，是迁移最容易翻车的地方（见第 5 节）。

## 3. 训练全流程（步骤的含义，而非具体命令）

整体四步，每步讲"为什么要这步"：

```
[1] 环境与依赖        [2] 权重/数据准备        [3] 写配置 YAML       [4] 分布式启动训练
  CANN+MindSpore   →   HF ckpt→MindSpore     →  并行/超参/路径    →  HCCL 多卡拉起
  +MindFormers         MindRecord 数据集         模型结构对齐         监控 loss/吞吐
```

**步骤 1：环境与依赖（只讲含义）**
- 装齐 CANN（含驱动/固件/toolkit）→ 装对应版本 MindSpore（Ascend 版）→ 装 MindFormers。三者**版本必须配套**，错配是最高频的安装坑。具体版本矩阵与安装命令以华为昇腾官方文档（Ascend 社区）为准。
- 通常用官方提供的 Ascend 训练镜像（docker）省去手装。镜像里 CANN/MindSpore 已对齐。
- 注意点：宿主机驱动版本要 ≥ 容器内 CANN 要求；多机训练要打通 HCCL 用的网络与 `rank table`（机器拓扑文件）。

**步骤 2：权重与数据准备**
- **权重转换**：把 HuggingFace 的 Qwen-1 `.safetensors/.bin` 转成 MindSpore 的 `.ckpt`。MindFormers 一般提供 `convert_weight` 类脚本，核心是**按层名做参数映射 + dtype 对齐**。
- **数据转换**：把原始语料 tokenize 后打包成 **MindRecord**（MindSpore 的高效数据格式，类比 Megatron 的 `.bin/.idx` 或 WebDataset），训练时按序列长度 pack。预训练用纯文本拼接；SFT 用对话模板（Qwen 的 ChatML：`<|im_start|>...<|im_end|>`），并做 loss mask 只在 assistant 段算 loss。

**步骤 3：写配置 YAML（MindFormers 的灵魂）**
MindFormers 高度配置驱动，一个 YAML 描述：模型结构（层数/隐藏维/头数/词表，须与 Qwen-1 对齐）、并行策略（dp/mp/pp/optimizer 切分）、优化器与学习率、数据集路径、ckpt 加载/保存、混合精度。**改模型规模、改并行、改数据，基本就是改 YAML**，不改代码——这是它和直接写 Megatron 脚本最大的体验差异。

**步骤 4：分布式启动训练**
- 单机多卡/多机多卡由启动脚本 + `rank table`（描述哪些卡参与、device id 与 ip 映射）驱动，底层用 HCCL 建立通信域。
- 训练中关注：loss 曲线是否正常下降、单步耗时（吞吐）、NPU 利用率（用 msprof 看）。
- 断点续训靠定期保存 ckpt；分布式下保存的是**切分后的分片 ckpt**，恢复时并行策略要一致或做 ckpt 重切分。

## 4. 并行机制与图模式（原理）

大模型放不进一张卡，靠多种并行切分。MindFormers/MindSpore 支持的并行维度与 Megatron 几乎一一对应：

```
        全局 batch 数据
              │
   ┌──────────┴──────────┐  数据并行 DP：每路看不同数据，梯度 AllReduce
   ▼                     ▼
 [模型副本0]          [模型副本1]
   │  张量并行 TP：把一层的矩阵按列/行切到多卡，前向后向插 AllReduce/AllGather
   │  ┌────┬────┐
   │  │卡A │卡B │  ← 一个 Transformer 层横切
   │  └────┴────┘
   │  流水并行 PP：把不同层放不同卡，micro-batch 流水起来减少气泡
   ▼
 Layer0-7 → Layer8-15 → ...（按 stage 切分到不同卡组）
```

- **数据并行 DP**：每卡完整模型副本、不同数据，反向后用 **HCCL AllReduce** 同步梯度（对标 NCCL AllReduce）。
- **张量/模型并行 MP/TP**：把单层大矩阵（如 attention 的 QKV、FFN 的投影）按维切到多卡，层内通信密集，通常放在卡间高带宽域（HCCS 内）。
- **流水并行 PP**：按层分 stage，配合 micro-batch 流水以减少"气泡"。
- **优化器并行**：类似 ZeRO，把优化器状态切分到 DP 组（MindSpore 的 optimizer parallel）。
- **自动并行**：MindSpore 特色——可由框架根据 sharding 策略自动推导切分与通信，减少手写并行代码。

**图模式（GRAPH_MODE）是昇腾性能关键**：MindSpore 默认把网络编译成静态计算图，由 CANN 的图引擎(GE)做算子融合、内存复用、并行编排后下发 NPU。相比 PyTorch 的 eager（逐算子下发），图模式启动有一次较慢的编译，但稳态吞吐更高。代价：调试不如 eager 直观、动态 shape 支持受限（变长序列要靠 padding/分桶）。

## 5. 从 HF / Megatron 迁移到 MindFormers 的要点

| 迁移项 | GPU 侧做法 | 昇腾 MindFormers 侧 | 注意点 |
|--------|-----------|---------------------|--------|
| 模型代码 | `transformers` 的 `QwenForCausalLM` | YAML 描述 + MindFormers 内置 Qwen 类 | 结构超参必须逐项对齐 |
| 权重格式 | `.safetensors` | `.ckpt`（需转换） | 层名映射 + dtype 对齐 |
| 词表 | `qwen.tiktoken` | 同源分词器 | token id 必须完全一致 |
| 数据 | jsonl/arrow + DataLoader | MindRecord | 提前 tokenize 打包 |
| 并行 | `deepspeed`/`megatron` 参数 | YAML 的 parallel 配置 | 概念对应但写法不同 |
| 启动 | `torchrun`/`deepspeed` | 启动脚本 + rank table | 多机靠 rank table |
| 混合精度 | `bf16`/`fp16` AMP | MindSpore 混精配置 | bf16 在 910 上更稳 |

**最关键的两个对齐**：
1. **权重层名映射**——HF 的 `transformer.h.{i}.attn.c_attn.weight` 之类，要正确映射到 MindFormers 的命名；QKV 若是 fused 投影，切分顺序错了会让模型"能跑但胡说"。
2. **分词器一致**——Qwen 用 tiktoken，词表 15 万级，任何不一致都会让训练数据的 label 错位。验证方法：拿同一句话，比对两侧 token id 序列完全相同。

## 6. 注意事项与常见坑

- **版本配套**：CANN ↔ MindSpore ↔ MindFormers 三件套版本必须匹配，且宿主机驱动/固件要够新。错配报错往往很底层、难懂。优先用官方配套镜像。
- **首次图编译慢**：图模式下第一步特别久（在编译），别误判为卡死；编译产物可缓存复用。
- **动态 shape 受限**：变长序列建议固定 `seq_length` + padding，或做长度分桶，避免反复重编译。
- **精度选择**：910 上优先 **bf16**（动态范围大、不易溢出），fp16 在大模型训练里更易出 NaN；混精下注意 RMSNorm/softmax/loss 等敏感算子保持高精度。
- **算子不支持/回退**：个别 shape 或新算子 CANN 可能不支持，导致回退到低效实现或报错；用 profiling 定位热点，必要时调整 shape 或反馈官方。
- **HCCL 通信**：多机训练 rank table 配错、网络不通、超时是高频坑；TP 尽量放在高带宽域内（HCCS），跨节点走 PP/DP。
- **内存(显存/NPU 内存) OOM**：先动 micro-batch、序列长度、重计算(recompute)、并行切分这几个旋钮；MindSpore 的重计算可显著省内存换算力。
- **loss 看着正常却学不好**：八成是词表/权重映射/loss mask 三者之一错了，回到第 5 节逐项核对。
- **性能调优思路（机制层面）**：① 提升 NPU 利用率（增大 batch、减少同步等待）；② 用算子融合（图模式自动 + 手动开关）；③ 平衡三种并行减少通信；④ 用 profiling(msprof) 找气泡和访存瓶颈，对症下药。

## 常见问题

| 问题 | 答案 |
|------|------|
| MindFormers 对标 GPU 上的什么？ | Megatron-LM + HuggingFace Transformers 的合体（套件层） |
| HCCL 对标什么？ | NCCL，提供 AllReduce 等集合通信原语 |
| CANN 对标什么？ | CUDA + cuDNN/cuBLAS，是昇腾的底层计算栈 |
| 为什么要把 HF 权重转 ckpt？ | MindSpore 用自己的 ckpt 格式，且需层名映射对齐 |
| 训练为什么默认图模式？ | CANN 图引擎要静态图做融合/调度，稳态性能更高 |
| fp16 还是 bf16？ | 910 上优先 bf16，动态范围大更稳，不易 NaN |
| 第一步特别慢正常吗？ | 正常，是图编译，不是卡死，编译产物可缓存 |
| 多机训练靠什么组网？ | rank table（拓扑文件）+ HCCL 建通信域 |
| Qwen-1 词表多大？为何重要？ | 约 15 万级 tiktoken BPE，分词器不一致会让 label 错位 |
| 具体安装命令/版本去哪查？ | 以华为昇腾官方文档（Ascend 社区）与 MindFormers 仓库 README 为准 |

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
