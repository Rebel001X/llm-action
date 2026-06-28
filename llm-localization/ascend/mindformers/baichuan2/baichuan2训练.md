# 昇腾 MindFormers 训练 Baichuan2

> 用 MindSpore + MindFormers 套件在昇腾 NPU 上完成 Baichuan2 的预训练 / 微调全流程。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/megatron-lm/README]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]] [[llm-algo/transformer/模型架构]]

> 参考：MindFormers research/baichuan2 官方文档（具体命令与版本以华为昇腾官方文档 / MindFormers 仓库为准）
> https://gitee.com/mindspore/mindformers/blob/r1.0/research/baichuan2/baichuan2.md

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|--------------|--------|
| 0 锚点 | 一句话说清这件事 | MindFormers / Baichuan2 / NPU |
| 1 地基 | 昇腾软件栈定位 + 昇腾↔英伟达对照 | CANN / MindSpore / MindFormers |
| 2 Baichuan2 模型要点 | 这模型和 LLaMA 差在哪 | NormHead / Alibi / max-z-loss |
| 3 训练全流程 | 从权重转换到拉起训练的步骤含义 | ckpt / yaml / 并行 |
| 4 分布式并行 | DP/MP/PP/优化器并行怎么配 | 自动并行 / sharding |
| 5 数据流水线 | 数据怎么变成 MindRecord | tokenizer / MindRecord |
| 迁移要点 | 从 HF/Megatron 迁到昇腾改什么 | 图模式 / 算子 / 坑 |
| 常见问题 | 高频报错与排查 | OOM / 精度 / loss |

## 0. 一句话锚点

**MindFormers 是华为昇腾官方的「大模型训练/推理套件」，对标 NVIDIA 生态里的 Megatron-LM + HF Transformers；Baichuan2 是百川智能的中英文大模型（7B/13B）。本文讲的是：如何在昇腾 NPU 上，用 MindFormers 提供的 Baichuan2 配方（research/baichuan2）完成权重转换、数据准备、分布式训练与微调。**

核心心智：你不写 PyTorch + CUDA，而是写 **MindSpore + CANN**；你不手搓 Megatron 并行，而是改 **一个 YAML 配置 + 一份 ckpt**，由 MindFormers 的 Trainer 把并行策略下发到昇腾集群。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层（Baichuan2 训练落在哪一层）

```
┌─────────────────────────────────────────────────────────┐
│  应用层：你的训练任务 = Baichuan2 配方 (research/baichuan2) │  ← 改 YAML / 准备数据
├─────────────────────────────────────────────────────────┤
│  套件层：MindFormers（Trainer / 并行 / 配置 / 权重转换）    │  ← 对标 Megatron-LM + HF
├─────────────────────────────────────────────────────────┤
│  框架层：MindSpore（计算图 / 自动微分 / 自动并行）          │  ← 对标 PyTorch
├─────────────────────────────────────────────────────────┤
│  异构计算架构：CANN（GE 图引擎 / 算子库 / HCCL 集合通信）    │  ← 对标 CUDA + cuDNN + NCCL
├─────────────────────────────────────────────────────────┤
│  硬件层：昇腾 NPU（达芬奇架构 Cube/Vector，910 系列）        │  ← 对标 GPU（A100/H100）
└─────────────────────────────────────────────────────────┘
              ↑ 通过 HCCL over RoCE/HCCS 组成多卡多机集群
```

要点：**Baichuan2 训练任务只是最上面两层的事**。模型结构、并行切分、混合精度由 MindFormers 封装；真正把矩阵乘搬上 Cube 单元、把 AllReduce 跑在网络上的，是 CANN 和 HCCL。理解分层后你就知道：调网络（HCCL）、调算子（CANN）、调结构与并行（MindFormers），各归各位。

### 1.2 昇腾 ↔ 英伟达 生态对照表（迁移心智图）

| 维度 | 英伟达生态 | 昇腾生态 | 在本任务中的角色 |
|------|-----------|----------|------------------|
| 加速硬件 | GPU (A100/H100) | NPU (Ascend 910 系列) | 跑 Baichuan2 前反向的算力 |
| 底层计算架构 | CUDA | CANN | 把算子下发到 NPU |
| 算子/数学库 | cuDNN / cuBLAS | CANN 算子库 (AICore 算子) | MatMul/FlashAttention 等实现 |
| 集合通信 | NCCL | HCCL | DP/MP 的 AllReduce/AllGather |
| 深度学习框架 | PyTorch | MindSpore | 自动微分 + 图执行 |
| 训练套件 | Megatron-LM | MindFormers | 并行 + 大模型配方 |
| 模型/权重库 | HF Transformers | MindFormers (research/ 配方) | Baichuan2 结构 + ckpt |
| 模型保存格式 | `.bin` / `.safetensors` | `.ckpt` (MindSpore) | 需做权重转换 |
| 数据格式 | WebDataset / Arrow / idxmap | MindRecord | 预处理产物 |
| 混合精度 | AMP (fp16/bf16) | MindSpore 混合精度 (fp16/bf16) | 省显存提吞吐 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 训完后部署 |
| 量化工具 | GPTQ / AWQ | msModelSlim | 部署前压缩 |

记住一条迁移主线：**HF/Megatron 权重 (.bin) → 转换脚本 → MindSpore .ckpt → 改 YAML → MindFormers 拉起多卡训练 → 训出 .ckpt → MindIE 部署**。

## 2. Baichuan2 模型要点（训练前必须懂的结构差异）

Baichuan2 在 LLaMA 类架构上有几处「非标准」改动，MindFormers 的 research/baichuan2 配方已经实现，但你迁移/调参时要心里有数：

| 结构点 | Baichuan2 的做法 | 训练影响 |
|--------|------------------|----------|
| 位置编码 | 7B 用 RoPE，13B 用 **ALiBi**（线性偏置） | 长度外推友好；注意力实现要选对 |
| 输出头 | **NormHead**：对 LM Head 权重做归一化 | 稳定训练、改善长尾词；不能简单当普通 Linear |
| 训练稳定 | **max-z-loss**（NormSoftmax 正则）抑制 logit 爆炸 | loss 里多一项，需在配置中开启 |
| 词表 | 约 12.5 万词表，中英混合更友好 | embedding 大、显存占用要算进去 |
| 激活 | SwiGLU；RMSNorm | 与 LLaMA 同 |

```
Baichuan2 一层 Decoder Block（概念图）
  x ──► RMSNorm ──► Attention(RoPE/ALiBi) ──► (+残差) ──┐
                                                        ▼
        RMSNorm ──► FFN(SwiGLU) ──► (+残差) ──────────► out
最后：RMSNorm ──► NormHead(对权重L2归一) ──► logits ──► (+max-z-loss)
```

## 3. 训练全流程（讲清每一步「为什么」，不造命令）

> 安装、镜像、确切命令与版本一律以华为昇腾官方文档（Ascend 社区 / MindFormers 仓库）为准。下面只讲流程含义、依赖关系与坑。

```
[1] 环境就绪      [2] 权重转换       [3] 数据准备        [4] 写/改 YAML      [5] 拉起训练
CANN+MindSpore   HF .bin → .ckpt    raw → MindRecord    并行/超参/路径      单机/多机分布式
+MindFormers     (tokenizer 对齐)   (tokenizer 编码)    (research 配方)     (Trainer 启动)
     │                │                  │                  │                  │
     └── 缺一不可 ────┴── 必须先于训练 ──┴── 决定吞吐与显存 ─┴── 决定能否跑起来 ┘
```

### 3.1 环境就绪
- 依赖链：**驱动/固件 → CANN → MindSpore → MindFormers**，版本必须配套（昇腾对版本耦合敏感，错配是头号坑）。
- 用 `npu-smi` 类工具确认卡可见、HCCL 通信可用（多机还要配 rank table / 组网）。
- 具体版本矩阵以官方「版本配套表」为准，不要凭记忆拼版本号。

### 3.2 权重转换（HF → MindSpore）
- 目的：HF 的 `.bin` 权重与 MindSpore `.ckpt` 的**张量命名、切分约定不同**，必须经转换脚本映射。
- 关键点：NormHead、词表 embedding 命名要对齐；转换时别漏掉 tokenizer（`tokenizer.model`），否则编码不一致。
- 若直接从头预训练，可跳过此步，用随机初始化。

### 3.3 数据准备（→ MindRecord）
- MindSpore 训练读 **MindRecord**（类似 Megatron 的 idxmap / HF 的 Arrow）。
- 流程：原始语料 → 用 Baichuan2 tokenizer 编码 → 打包成定长序列 → 写成 MindRecord 分片。
- 坑：tokenizer 必须用 Baichuan2 自带的（词表 12.5 万），用错词表训练全废。

### 3.4 写 / 改 YAML（核心）
- MindFormers 用一份 YAML 描述**模型结构、并行策略、优化器、数据路径、混合精度、检查点**。
- 你 90% 的工作是改 research/baichuan2 配方里现成的 YAML：换数据路径、改并行度、改 batch / 学习率 / 序列长度。
- 务必让 YAML 里的并行度 × 单卡 batch 与你的卡数自洽（见第 4 节）。

### 3.5 拉起训练
- 单机多卡用昇腾分布式启动器（基于 rank table / `msrun` 类工具）拉起；多机要配通信组网。
- Trainer 按 YAML 自动完成图编译、并行切分、混合精度、断点续训。
- 观察：首个 step 编译慢（图模式特性）属正常；loss 应平稳下降，NormHead+max-z-loss 让 loss 曲线更稳。

## 4. 分布式并行（昇腾上怎么切 Baichuan2）

MindSpore 支持**半自动/自动并行**：你在 YAML 里声明并行度，框架据此切分计算图并插入 HCCL 通信，相比手写 Megatron 并行更省心。

| 并行方式 | 含义 | 对标 Megatron | Baichuan2 何时用 |
|----------|------|---------------|------------------|
| 数据并行 DP | 复制模型、切分数据 | DP | 卡多、模型放得下时优先 |
| 模型/张量并行 MP/TP | 切分单层权重（如注意力、FFN） | Tensor Parallel | 13B 单卡放不下时 |
| 流水并行 PP | 按层切到不同卡 | Pipeline Parallel | 层多、跨机时 |
| 优化器并行 | 切分优化器状态（类 ZeRO） | ZeRO-1/2 | 省显存，几乎必开 |

```
Baichuan2-13B 在 8 卡上的一种切法（示意）
  全局 8 卡 = DP(2) × MP(4)
  ┌────── DP 组 0 ──────┐   ┌────── DP 组 1 ──────┐
  │ MP: 卡0 卡1 卡2 卡3 │   │ MP: 卡4 卡5 卡6 卡7 │
  │  └ 一份模型切 4 份 ┘ │   │  └ 一份模型切 4 份 ┘ │
  └─────────┬───────────┘   └──────────┬──────────┘
   组内 MP 用 AllReduce/AllGather       组间 DP 用 AllReduce 同步梯度
   全部走 HCCL（对标 NCCL）
```

调参直觉：**先 DP，放不下再加 MP，跨机再加 PP，优化器并行常驻**。MP 通信密集，尽量留在单机内（走 HCCS/卡间高带宽），DP 的梯度 AllReduce 可跨机。

## 5. 数据流水线（MindRecord 怎么来）

```
原始语料(txt/jsonl)
      │ ① 清洗/拼接
      ▼
Baichuan2 Tokenizer (词表~12.5万)
      │ ② 编码成 token id
      ▼
定长打包 (seq_len, 拼接 + pad/截断)
      │ ③ 加 attention mask / labels
      ▼
写出 MindRecord 分片  ──►  训练时由 MindSpore Dataset 流式读入 NPU
```

要点：序列长度、是否拼接（packing）直接影响吞吐与显存；MindRecord 分片数建议 ≥ 数据并行度，避免读不均。

## 迁移要点 / 注意事项与坑

**从 HF/Megatron+CUDA 迁到 MindFormers+昇腾，主要改这些：**

1. **框架范式变了**：PyTorch 动态图 → MindSpore 默认**图模式（Graph Mode）**。首 step 要编译，慢是正常的；但写自定义逻辑时，Python 控制流不能随便用（图模式有约束）。
2. **权重要转换**：`.bin/.safetensors` → `.ckpt`，张量命名映射别错；NormHead 特殊处理。
3. **数据要重做**：Arrow/idxmap → MindRecord，tokenizer 必须用 Baichuan2 原配。
4. **并行从「手写」变「声明」**：Megatron 手动切 → MindSpore 在 YAML 里声明并行度，框架自动切图插 HCCL。
5. **通信换成 HCCL**：NCCL → HCCL，多机要配 rank table / 组网，环组网与拓扑影响 AllReduce 性能。

**高频坑：**
- **版本错配**：驱动/CANN/MindSpore/MindFormers 版本不配套 → 各种诡异报错。永远查官方版本配套表。
- **tokenizer 用错**：拿 LLaMA 词表跑 Baichuan2，训练 loss 不收敛。
- **NormHead/max-z-loss 漏配**：直接套通用 LLaMA 配方会丢掉这两项，精度与稳定性下降。
- **MP 跨机**：把张量并行切到机器之间，通信被低带宽网络拖死，吞吐暴跌。
- **图模式踩 Python 控制流**：动态 shape、随意 if/print 调试，可能编译失败或退回低效路径。
- **混合精度溢出**：fp16 下大 logit 易溢出，Baichuan2 的 max-z-loss 正是为此；必要时切 bf16。

**性能调优思路（机制层面）：**
- 优先用 bf16 + 优化器并行降显存，再把省下的显存换成更大 batch 提吞吐。
- MP 尽量锁在单机（HCCS 高带宽域），DP 梯度同步可跨机。
- 开启重计算（recompute）换显存，开 packing 提有效 token 利用率。
- 关注首 step 编译耗时与稳态 step 时间分开看，别把编译时间当训练慢。

## 常见问题

| 问题 | 可能原因 | 排查方向 |
|------|----------|----------|
| 拉起即报算子/通信错误 | CANN/MindSpore/MindFormers 版本不配套 | 对照官方版本配套表重装 |
| loss 不下降 / 乱跳 | tokenizer 用错、学习率过大、漏 max-z-loss | 核对 tokenizer 与 YAML 配方项 |
| 显存 OOM | 并行度不足、batch/seq 过大、没开优化器并行 | 加 MP、开优化器并行/重计算、降 batch |
| 多机吞吐很低 | MP 被切到跨机、组网/rank table 配错 | 把 MP 锁单机，检查 HCCL 组网 |
| 首个 step 极慢 | 图模式编译（正常现象） | 看第二个 step 起的稳态时间 |
| fp16 出现 NaN/Inf | logit 溢出 | 确认开启 max-z-loss，或切 bf16 |
| 权重加载失败 | HF→ckpt 转换命名不匹配 | 核对转换脚本与 NormHead 处理 |
| 续训对不上 | 断点 ckpt 与并行策略不一致 | 保证续训并行配置与原训练一致 |

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 硬件与算力：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 网络与通信：[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 训练框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 模型与算法：[[llm-algo/transformer/模型架构]]
- 训练与推理：[[llm-train/README]] · [[llm-inference/README]]
- 压缩部署：[[llm-compression/quantization/量化基础]]
