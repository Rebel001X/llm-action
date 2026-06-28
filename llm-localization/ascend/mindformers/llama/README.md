# 昇腾 MindFormers 之 LLaMA 训练

> 在华为昇腾 NPU 上用 MindFormers 套件训练/微调 LLaMA 系列大模型——它是昇腾世界里对标 Megatron-LM + HuggingFace 的"开箱即用大模型工厂"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/megatron-lm/README]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]] [[llm-algo/transformer/模型架构]]

## 阅读地图

| 节 | 你将搞清楚的问题 | 关键词 |
| --- | --- | --- |
| 0 | MindFormers + LLaMA 一句话是什么 | 套件 / 大模型工厂 |
| 1 | 它在昇腾软件栈哪一层、对标 CUDA 世界的谁 | CANN / Megatron / HF 对照表 |
| 2 | LLaMA 在昇腾上的算子是怎么落地的 | 达芬奇 Cube/Vector / 融合算子 |
| 3 | 四维并行如何在 NPU 集群上展开 | DP/TP/PP/SP + HCCL |
| 4 | 一次训练的端到端流程 | 权重转换 → 数据 → 配置 → 拉起 |
| 5 | 从 GPU/Megatron 迁移到昇腾要改什么 | 迁移要点 |
| 6 | 容易踩的坑 | 精度 / 内存 / 通信 |
| 7 | 常见问题速查 | FAQ |

## 0. 一句话锚点

**MindFormers 是昇腾官方的大模型全流程套件，LLaMA 是它内置支持的一个主流模型族。** 你给它一份 YAML 配置 + 数据集 + 转换好的权重，它就在昇腾 NPU 集群上帮你完成预训练 / 增量预训练 / SFT 微调 / 推理。你不需要手写并行切分、不需要手动调 HCCL 通信、不需要自己拼融合算子——这些都被套件和底层 CANN 封装好了。可以把它理解为"昇腾版的 Megatron-LM 训练脚本 + HuggingFace 模型库"二合一。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

昇腾软件栈自底向上分四层，MindFormers/LLaMA 处在最上面的"套件"层：

```
┌─────────────────────────────────────────────────────────┐
│  套件层  MindFormers(LLaMA/GLM/Qwen...) ← 本文在这里      │
│          提供：模型库 + 并行配置 + 训练/推理流水线        │
├─────────────────────────────────────────────────────────┤
│  框架层  MindSpore(图模式/动态图、自动微分、自动并行)     │
│          ←→ 对标 PyTorch                                  │
├─────────────────────────────────────────────────────────┤
│  异构计算层  CANN(算子库 AOL、图引擎 GE、HCCL、Runtime)   │
│          ←→ 对标 CUDA + cuDNN/cuBLAS + NCCL               │
├─────────────────────────────────────────────────────────┤
│  硬件层  昇腾 NPU(达芬奇架构 Cube/Vector/Scalar 单元)     │
│          ←→ 对标 GPU(SM / Tensor Core / CUDA Core)        │
└─────────────────────────────────────────────────────────┘
```

**昇腾 ↔ 英伟达 生态对照表（迁移心智图，最重要）**

| 维度 | 英伟达世界 | 昇腾世界 | 说明 |
| --- | --- | --- | --- |
| 加速硬件 | GPU（A100/H100） | NPU（昇腾 910 系列） | 训练用 910 系列 |
| 计算核心 | Tensor Core / CUDA Core | 达芬奇 Cube / Vector / Scalar | Cube 专吃矩阵乘 |
| 编程/计算平台 | CUDA | CANN | 异构计算架构 |
| 高性能算子库 | cuDNN / cuBLAS | CANN 算子库（AOL/aclnn） | 卷积、GEMM、融合算子 |
| 集合通信库 | NCCL | HCCL | AllReduce/AllGather 等 |
| 深度学习框架 | PyTorch | MindSpore | 也可用 PyTorch+torch_npu |
| 大模型训练框架 | Megatron-LM | MindFormers / ModelLink | 并行训练流水线 |
| 模型库/生态 | HuggingFace Transformers | MindFormers 模型库 | 内置 LLaMA 等 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 部署侧 |
| 量化工具 | GPTQ / AWQ | msModelSlim | 压缩侧 |
| 设备查看 | nvidia-smi | npu-smi | 看卡、看显存 |

> 一句话记忆：**MindFormers ≈ Megatron-LM × HuggingFace，跑在 MindSpore（≈PyTorch）上，底层踩着 CANN（≈CUDA）和 HCCL（≈NCCL）。**

## 2. LLaMA 在昇腾上：算子如何落地达芬奇架构

LLaMA 的核心计算无非是：RMSNorm、RoPE 旋转位置编码、注意力（QKV 投影 + Attention + 输出投影）、SwiGLU FFN。这些在昇腾上是怎么跑起来的？

**（1）矩阵乘交给 Cube 单元。** LLaMA 里绝大部分算力消耗在大矩阵乘法（QKV/O 投影、FFN 的两个大 GEMM）。达芬奇架构的 **Cube 单元**专门做矩阵乘累加（MAC），一拍完成一个固定尺寸的矩阵块乘法，这就是它对标 Tensor Core 的地方。所以"把 GEMM 切成 Cube 喜欢的块尺寸"是性能关键。

**（2）逐元素/归约交给 Vector 单元。** RMSNorm、Softmax、RoPE 的 sin/cos 乘加、激活函数这些"逐元素或沿某轴归约"的操作走 **Vector 单元**（对标 GPU 的 CUDA Core 通用算力）。

**（3）融合算子（Fused Op）减少搬运。** 昇腾上同样有 FlashAttention 类融合注意力算子，把"QK^T → mask → softmax → ×V"融成一个 kernel，避免中间大矩阵反复写回片上/片外内存。FFN 里也有 SwiGLU/Gelu 融合。融合的本质和 GPU 上一致：**减少 HBM 读写，让数据尽量停在片上缓冲区**。

```
LLaMA Decoder Layer 在昇腾上的算子分工
┌──────────────────────────────────────────────┐
│  x ──► RMSNorm (Vector) ──► QKV 投影 (Cube)     │
│        │                                        │
│        ▼                                        │
│   RoPE (Vector) ──► FlashAttention 融合算子      │
│        │              (Cube 做 QK^T/×V,         │
│        │               Vector 做 softmax)       │
│        ▼                                        │
│   O 投影 (Cube) ──► 残差(Vector) ──► RMSNorm     │
│        │                                        │
│        ▼                                        │
│   FFN: Gate/Up 投影(Cube) ─► SwiGLU(Vector)      │
│        ─► Down 投影(Cube) ─► 残差(Vector) ──► out │
└──────────────────────────────────────────────┘
```

**（4）图模式（Graph Mode）。** MindSpore 默认可把整张计算图编译后整体下发给昇腾，CANN 的图引擎 GE 会做算子融合、内存复用、流水编排等图级优化。这区别于 PyTorch 默认的逐算子下发（eager），更接近"编译执行"，吞吐通常更高，但调试不如动态图直观——这是从 PyTorch 过来最需要适应的思维转变。

## 3. 并行：四维切分如何在 NPU 集群展开

训 LLaMA 这种大模型，单卡放不下，必须切。MindFormers 把 Megatron 那套并行能力都搬过来了，靠 HCCL 做卡间通信：

| 并行方式 | 切什么 | 通信原语（走 HCCL） | 对标 |
| --- | --- | --- | --- |
| 数据并行 DP | 切 batch | AllReduce 梯度 | 同 Megatron |
| 张量并行 TP | 切单层权重矩阵 | AllReduce/AllGather | 同 Megatron-TP |
| 流水并行 PP | 切层（按 stage） | P2P send/recv | 同 Megatron-PP |
| 序列并行 SP | 切序列维（配合 TP） | AllGather/ReduceScatter | 省激活内存 |

```
   一个 8 卡节点上的并行布局示意（TP=2, PP=2, DP=2）
   ┌────────────┐   ┌────────────┐
   │ NPU0  TP0  │←─►│ NPU1  TP1  │  PP stage 0
   │  PP=0      │TP │  PP=0      │
   └─────┬──────┘   └─────┬──────┘
         │P2P             │P2P      ← 流水(PP)走 P2P
   ┌─────▼──────┐   ┌─────▼──────┐
   │ NPU2  TP0  │←─►│ NPU3  TP1  │  PP stage 1
   └────────────┘   └────────────┘
   （NPU4-7 是另一个 DP 副本，DP 间 AllReduce 同步梯度）
   节点内卡间走 HCCS/高速总线，跨节点走 RoCE 网络
```

**HCCL 的环算法直觉**：和 NCCL 一样，AllReduce 在环（Ring）拓扑上分两步——先 ReduceScatter 让每张卡拿到一段的归约结果，再 AllGather 把结果广播回所有卡。总通信量约 `2(N-1)/N × 数据量`，与卡数 N 弱相关，这是大集群能扩展的根本原因。MindFormers 用户一般不直接写 HCCL，但理解它有助于判断"通信是不是瓶颈"。

## 4. 端到端训练流程（讲含义，不背命令）

> 注意：以下只讲每一步"为什么要做、依赖谁、坑在哪"。**所有具体命令、包名、镜像、版本号，一律以华为昇腾官方文档（Ascend 社区 / MindFormers 官方仓库 README）为准。**

```
①环境  ②权重转换  ③数据预处理  ④写配置  ⑤拉起训练  ⑥续训/评估
  │        │           │           │          │          │
 CANN+    HF→ckpt    raw→mindrecord  YAML     分布式启动   ckpt
 MindSpore                           并行参数  脚本/ranktable
 +MindFormers
```

1. **环境准备**：装好匹配版本的固件/驱动 → CANN → MindSpore → MindFormers。**最大的坑是版本配套**：这四层有严格的版本对应关系，错配会出现算子不支持或导入报错。版本矩阵以官方文档为准，不要随意混搭。

2. **权重转换**：LLaMA 原始权重通常是 HuggingFace 格式，需要转成 MindFormers/MindSpore 的 ckpt 格式（套件一般自带转换脚本）。要点是**逐层名字映射要对齐**，尤其 RoPE、RMSNorm、合并/拆分 QKV 的处理，转错会"能跑但 loss 不对"。

3. **数据预处理**：把语料切 token 后打包成 MindSpore 的高效格式（mindrecord 之类），目的是让训练时 IO 不成为瓶颈。SFT 还要按对话模板拼 prompt/answer 并做 loss mask。

4. **写 YAML 配置**：这是 MindFormers 的灵魂。一个配置文件里声明模型结构（层数/隐藏维/头数）、序列长度、并行度（dp/tp/pp/sp）、优化器、学习率调度、重计算（recompute）、混合精度等。**并行度乘起来必须等于总卡数**，且要被层数/头数整除，这是新手最常配错的地方。

5. **分布式拉起**：多机多卡需要一份组网信息（rank table / 集群配置）告诉每张卡自己的 rank 和如何互联，再用分布式启动脚本把任务铺到各卡。HCCL 据此建立通信域。

6. **续训与评估**：定期存 ckpt（断点续训靠它），按 perplexity 或下游任务评估。

## 5. 迁移要点：从 GPU/Megatron 搬到昇腾

| 你在 GPU 上的做法 | 到昇腾要换成 | 注意 |
| --- | --- | --- |
| PyTorch + Megatron 脚本 | MindSpore + MindFormers + YAML | 从"写脚本"变"填配置" |
| `.from_pretrained()` 加载 HF 权重 | 先做权重格式转换 | 名字映射别错 |
| `torch.distributed`/NCCL 初始化 | HCCL + rank table 组网 | 组网文件是新概念 |
| eager 逐算子调试 | 优先图模式（编译执行） | 调试思路要变 |
| `nvidia-smi` 看卡 | `npu-smi` 看卡 | 命令名变了 |
| 自定义 CUDA kernel | 找 CANN 是否有对应算子/融合算子 | 没有则需适配，成本高 |
| AMP fp16/bf16 | 昇腾混合精度（bf16 优先） | 见下节精度坑 |

**核心心态**：迁移不是逐行翻译代码，而是"换一套等价的工具链 + 把模型描述成配置"。如果你坚持用 PyTorch，也可以走 `torch_npu` 适配路线（PyTorch 算子映射到 CANN），但训大模型用 MindFormers 这条原生路线生态更完整、并行更省心。

## 6. 注意事项与常见坑（机制层面）

- **版本配套坑**：固件/驱动 ↔ CANN ↔ MindSpore ↔ MindFormers 四层版本必须配套，这是 90% 安装类问题的根因。出错先查版本矩阵。
- **精度坑**：昇腾上训练优先用 **bf16**（动态范围大、对溢出更友好）而非 fp16。Loss scale、RMSNorm/Softmax 等归约建议保留 fp32 累加，否则容易 NaN 或精度漂移。和 GPU 复现 loss 曲线时，要接受小幅差异（算子实现不同）。
- **算子覆盖坑**：CANN 算子库很全但不是 100% 覆盖。遇到"某算子不支持"时，看是否能用等价算子替换、换数据类型、或升级 CANN；自定义算子开发成本高，尽量避免。
- **并行配置坑**：`dp×tp×pp = 总卡数`，且 tp 要能整除注意力头数、pp 要能整除层数。配不对直接起不来。TP 尽量限制在单节点内（用高速总线），跨节点用 PP/DP，减少慢速网络上的高频通信。
- **内存/重计算坑**：NPU 显存有限，长序列训练务必开重计算（recompute，用算力换显存）和合理的优化器并行/ZeRO 类切分，否则 OOM。
- **图模式调试坑**：图模式报错往往在编译期，信息不如动态图直观。可先用动态图（PyNative）跑通小规模，再切图模式追求吞吐。
- **数据格式坑**：直接喂原始文本会让 IO 拖垮训练，务必预处理成 mindrecord 类高效格式。

## 7. 常见问题

| 问题 | 答案 |
| --- | --- |
| MindFormers 和 ModelLink 啥关系？ | 都是昇腾大模型训练套件。MindFormers 基于 MindSpore；ModelLink 走 PyTorch+torch_npu 路线，更贴近 Megatron 习惯。二者都能训 LLaMA。 |
| 必须用 MindSpore 吗？能用 PyTorch 吗？ | MindFormers 原生是 MindSpore。想用 PyTorch 可走 torch_npu / ModelLink 路线。 |
| LLaMA 的 HF 权重能直接加载吗？ | 不能直接，需要做格式与层名映射转换，套件一般提供转换脚本。 |
| 单卡能训 LLaMA 吗？ | 小模型/微调可以；7B 及以上一般需要多卡并行（TP/PP/DP）。 |
| 如何看是不是通信瓶颈？ | 用昇腾 Profiling 工具看 HCCL 通信耗时占比；TP 跨节点、PP stage 不均衡都会放大通信。 |
| bf16 还是 fp16？ | 昇腾训练优先 bf16，溢出风险更低。 |
| 具体命令/版本去哪查？ | 一律以华为昇腾官方文档（Ascend 社区）与 MindFormers 官方仓库为准，本文不固化版本号。 |

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
