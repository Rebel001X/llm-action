# ModelLink 数据集处理(Dataset)

> ModelLink 训练套件里"原始语料 → 可被高效喂入大模型训练的二进制数据"的预处理与加载链路,是昇腾上跑大模型训练绕不过去的第一步。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/megatron-lm/README]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | 预处理 / mmap / tokenize |
| 1 | 在昇腾栈的定位 + 昇腾↔英伟达对照表 | ModelLink / Megatron / CANN |
| 2 | 为什么要"预处理成二进制" | bin/idx / mmap / 不在训练时 tokenize |
| 3 | 预处理全链路(ASCII 图) | 下载→清洗→tokenize→序列化 |
| 4 | 预训练数据 vs 指令微调数据 | pretrain / SFT / handler |
| 5 | 数据加载与采样 | 多数据集混合 / blend / 权重 |
| 6 | Tokenizer 与 ModelLink | HF tokenizer / vocab 对齐 |
| — | 迁移要点 / 注意事项与坑 | 从 GPU 迁昇腾 |
| — | 常见问题 + 跳转链接 | FAQ / 双链 |

## 0. 一句话锚点

大模型训练不是把文本文件直接喂进去,而是先把语料**离线 tokenize 成整数 token、再序列化成可内存映射(mmap)的二进制文件**(一对 `.bin` + `.idx`),训练时按样本随机访问、零拷贝读取。ModelLink 沿用了 Megatron-LM 这套数据格式与流程,只是把后端算力换成了昇腾 NPU。**数据处理本身几乎不依赖 GPU/NPU,主要吃 CPU 与磁盘**——所以这一步在 GPU 世界和昇腾世界做法基本一致,这正是迁移最平滑的环节。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

ModelLink 是华为基于 Megatron-LM 在昇腾上的大模型分布式训练套件(后续演进有 MindSpeed-LLM 等名称,**具体名称/版本以华为昇腾官方文档为准**)。它处在"昇腾软件栈"的**训练套件层**:

```
┌───────────────────────────────────────────────┐
│  套件层   ModelLink / MindSpeed-LLM            │ ← 数据预处理脚本在这一层
│           (对标 Megatron-LM / NeMo)             │
├───────────────────────────────────────────────┤
│  框架层   PyTorch + torch_npu (Ascend Extension)│
│           (对标 原生 PyTorch + CUDA 后端)        │
├───────────────────────────────────────────────┤
│  加速库   CANN(算子库 + 图引擎 + HCCL)          │ ← 对标 CUDA + cuDNN + NCCL
├───────────────────────────────────────────────┤
│  硬件层   昇腾 NPU(达芬奇架构 Cube/Vector)       │ ← 对标 NVIDIA GPU(Tensor Core)
└───────────────────────────────────────────────┘
```

数据集处理脚本(如 `preprocess_data` 一类入口)位于**套件层**,它在 CPU 上跑,产出的二进制文件被框架/套件在训练时加载到 NPU。

### 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 维度 | 昇腾世界 | 英伟达世界 | 说明 |
|---|---|---|---|
| 训练加速卡 | 昇腾 NPU | GPU | NPU = Neural-network Processing Unit |
| 底层加速软件栈 | CANN | CUDA | 算子/运行时/编译 |
| 算子库 | CANN 算子(AOL/aclnn) | cuDNN / cuBLAS | 卷积/矩阵乘等 |
| 集合通信 | HCCL | NCCL | 多卡 all-reduce 等 |
| 深度学习框架 | PyTorch + torch_npu | 原生 PyTorch(CUDA) | torch_npu 提供 NPU 后端 |
| 大模型训练套件 | **ModelLink / MindSpeed-LLM** | **Megatron-LM** | 本文主角所在层 |
| 数据集格式 | `.bin` + `.idx`(mmap) | `.bin` + `.idx`(mmap) | **几乎一致**,沿用 Megatron 格式 |
| 数据预处理入口 | `preprocess_data`(套件脚本) | `tools/preprocess_data.py` | 接口风格高度相似 |
| Tokenizer | 复用 HuggingFace tokenizer | HuggingFace tokenizer | 跨生态通用,不依赖硬件 |
| 推理引擎 | MindIE | TensorRT-LLM / vLLM | 数据处理无关,补全心智图 |

**关键认知**:数据集预处理这一环,昇腾几乎"原样照搬"了 Megatron 的设计——`.bin/.idx` 格式、`preprocess_data` 脚本风格、tokenizer 复用 HF——所以从 Megatron 迁过来,**数据处理代码改动量最小**,真正的差异在后续的并行训练与算子层。

## 2. 为什么要"离线预处理成二进制"

直接拿原始 `.jsonl` / `.txt` 在训练循环里临时 tokenize,会遇到三个问题:

1. **CPU 成为瓶颈**:tokenize 是纯 CPU 计算,若放进训练步内,NPU 算得快、CPU 喂不上,卡算力空转。
2. **重复劳动**:同一份语料多轮训练 / 多次实验都要重 tokenize,浪费。
3. **随机访问难**:训练要按全局打乱的样本索引随机取第 N 个序列,文本文件做不到 O(1) 随机访问。

解决办法就是 Megatron(及 ModelLink)的 **mmap 二进制数据集**:

- `.bin`:把所有文档 tokenize 后的整数 token 顺序拼接,存成紧凑的二进制大文件。
- `.idx`:记录每篇文档在 `.bin` 中的起止偏移、长度、dtype 等元信息(索引)。

训练时通过 **内存映射(memory map)** 打开 `.bin`,操作系统按需把用到的页调入内存,**零拷贝、可随机寻址、跨进程共享**。多个数据并行 rank 共享同一份 mmap,内存开销不随并行度线性膨胀。

## 3. 预处理全链路(ASCII 图)

```
 原始语料                预处理(CPU,离线)              训练时加载(NPU)
┌──────────┐   下载/清洗   ┌───────────────────────┐   mmap 打开   ┌──────────────┐
│ 网页/书籍 │ ───────────▶ │ 1. 读取 jsonl/text     │              │ DataLoader    │
│ 代码/对话 │             │ 2. 按 key 取 text 字段  │              │  ├ 随机 index │
│ (jsonl)  │             │ 3. HF tokenizer 编码    │  ─────────▶  │  ├ 取序列     │
└──────────┘             │ 4. 加 eod/特殊 token    │   .bin/.idx  │  └ 组 batch  │
                         │ 5. 序列化 → .bin/.idx  │              │      │        │
                         └───────────────────────┘              │      ▼        │
                                                                │  喂入 NPU 计算 │
                                                                └──────────────┘
       └──── 纯 CPU + 磁盘,与硬件厂商无关 ────┘     └─ 这一步才上昇腾 NPU ─┘
```

要点:**虚线左侧整段几乎与 NVIDIA / 昇腾无关**,是普通的 CPU 数据工程;只有最右侧"喂入计算"才真正区分 GPU 与 NPU。这解释了为什么数据处理是迁移里最省心的部分。

## 4. 预训练数据 vs 指令微调数据

ModelLink 的预处理脚本通常用一个"数据处理器(handler)"参数来区分不同数据形态。**两类数据组织方式不同**:

### 4.1 预训练(Pretrain)数据

- 形态:海量无标注纯文本,目标是"预测下一个 token"。
- 处理:把每篇文档 tokenize,文档之间用 **end-of-document(eod)** 特殊 token 分隔,然后整体拼成一条长 token 流,训练时按固定 `seq_length` 切窗口。
- 关键:不需要 prompt/response 结构,**只关心连续 token**。

### 4.2 指令微调(SFT / Instruction)数据

- 形态:`instruction / input / output`(或 `messages` 对话)三元组。
- 处理:按**对话模板(chat template)**拼成单条序列,并生成 **loss mask**——只在 `output`(assistant 回复)部分计算损失,prompt 部分不回传梯度。
- 关键:必须正确套用与目标模型匹配的对话模板,模板错了 SFT 效果会显著退化。不同 handler 对应不同数据 schema(Alpaca 风格、ShareGPT 风格等)。

```
预训练:  [doc1 tokens][eod][doc2 tokens][eod]...  →  滑窗切 seq_length
SFT:     [系统][用户:instruction+input][助手:output][eod]
          └─ loss_mask = 0 ─────────────┘└─ loss_mask = 1 ─┘
```

**坑提醒**:SFT 时如果忘了对 prompt 段做 mask,模型会去"学着复述用户的问题",训练信号被污染。务必确认 handler 选对、模板与模型一致。

## 5. 数据加载、混合与采样

预处理产物可以是多个数据集(中文、英文、代码、数学……)。ModelLink 沿用 Megatron 的**数据混合(data blending)**机制:

- 给每个数据集一个**采样权重**,训练时按权重从各数据集随机抽样,实现"配比"。
- 权重影响模型能力分布(代码占比高 → 代码能力强),是数据配方的核心旋钮。
- 通过全局打乱的索引(shuffle index)保证不同 epoch 顺序不同,且各数据并行 rank 看到不重叠的切片。

```
        权重 0.5            权重 0.3           权重 0.2
   ┌─────────────┐    ┌─────────────┐   ┌─────────────┐
   │ 中文语料 bin │    │ 英文语料 bin │   │ 代码语料 bin │
   └──────┬──────┘    └──────┬──────┘   └──────┬──────┘
          └──────────────────┼─────────────────┘
                    按权重采样 + 全局 shuffle
                             │
                     ┌───────▼────────┐
                     │ 每个 DP rank 取 │ → batch → NPU
                     │  自己的索引切片  │
                     └────────────────┘
```

## 6. Tokenizer 与 ModelLink

- ModelLink 预处理通常**直接复用 HuggingFace 的 tokenizer**(指定模型目录,加载其 `tokenizer.json` / `tokenizer.model`),不自造分词器。
- **vocab 对齐是硬约束**:预处理用的 tokenizer 必须与训练所用模型的词表完全一致,否则 token id 与 embedding 错位,训练直接学崩。
- 特殊 token(eod、pad、bos/eos)的设置要与模型配置匹配,SFT 还需对话模板的特殊标记齐全。

**具体的脚本参数名、tokenizer 类型枚举、包名与版本号,一律以华为昇腾官方文档(Ascend 社区)为准。** 本文只讲机制与依赖关系,不杜撰命令行。

## 迁移要点 / 注意事项与坑

从 NVIDIA + Megatron 迁到昇腾 + ModelLink,数据处理这一环的实操经验:

1. **预处理阶段基本无需改硬件相关代码**。tokenize、序列化都在 CPU,产出的 `.bin/.idx` 与 Megatron 格式兼容性高;真正需要为昇腾改动的是后续训练的并行/算子/通信配置,不是数据。

2. **tokenizer 与模型词表一定要对齐**。跨生态最常见的事故是用了和权重不匹配的 tokenizer,现象是 loss 高得离谱或不收敛——先排查 tokenizer。

3. **SFT 的 loss mask 与对话模板**:选错 handler 或模板,会让模型学坏。迁移已有 GPU 上的 SFT 配方时,逐字核对模板。

4. **数据混合权重要重新审视**:更换硬件不改配方,但若同时换了模型尺寸/语料,权重需要重调,这是数据科学问题不是硬件问题。

5. **磁盘与内存**:mmap 依赖足够的页缓存,大规模语料预处理对磁盘吞吐和内存敏感;昇腾训练机的 CPU/磁盘配置若与原 GPU 集群不同,预处理吞吐会有差异,提前压测。

6. **CPU 多进程并行**:预处理通常用多 worker 并行 tokenize 提速,worker 数与机器 CPU 核数匹配即可,与 NPU 无关。

7. **路径/编码坑**:中文语料注意统一 UTF-8;jsonl 字段名(text/content)要与脚本的 `--json-key` 类参数一致,否则读到空文本。**具体参数名以官方文档为准。**

8. **不要在训练时临时 tokenize**:这是性能反模式——会让昂贵的 NPU 等廉价的 CPU。坚持"离线预处理 + mmap 加载"。

## 常见问题

| 问题 | 答案 |
|---|---|
| ModelLink 的数据格式和 Megatron 一样吗? | 高度一致,沿用 `.bin` + `.idx` 的 mmap 二进制格式,迁移改动极小。 |
| 数据预处理跑在 NPU 上吗? | 不。预处理是 CPU + 磁盘任务,与昇腾/英伟达硬件无关;只有训练才上 NPU。 |
| 为什么要离线 tokenize? | 避免训练时 CPU 喂不上 NPU、避免重复劳动、支持 O(1) 随机访问。 |
| 预训练和 SFT 数据处理差别? | 预训练拼连续 token 流;SFT 按对话模板拼,并对 prompt 段做 loss mask。 |
| tokenizer 从哪来? | 直接复用 HuggingFace tokenizer,**必须与目标模型词表一致**。 |
| 多数据集怎么配比? | 用数据混合权重(blend)按比例采样,是数据配方核心旋钮。 |
| 具体命令和版本去哪查? | 一律以华为昇腾官方文档(Ascend 社区)为准,本文不杜撰。 |
| 从 GPU 迁过来最容易踩什么坑? | tokenizer/词表不匹配、SFT 模板与 loss mask 出错。 |

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
