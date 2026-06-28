# 昇腾 + MindSpore 上的 BERT

> 在昇腾 NPU 上用 MindSpore 跑通 BERT 的预训练/微调全流程：从语料预处理、图模式编译、Cube/Vector 算子映射，到从 CUDA+PyTorch 迁移的对照与踩坑。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-algo/transformer/模型架构]]

## 阅读地图

| 节 | 你会得到什么 | 适合谁 |
| --- | --- | --- |
| 0 | 一句话锚点：BERT 在昇腾栈里到底是个什么角色 | 所有人 |
| 1 | 地基：昇腾软件栈分层 + 昇腾↔英伟达生态对照表 | 从 CUDA/PyTorch 迁过来的人 |
| 2 | 数据流水线：Wikipedia 语料 → 词表 → MindRecord | 准备预训练数据的人 |
| 3 | 模型机制：BERT 结构如何落到 Cube/Vector 单元 | 想懂底层算子的人 |
| 4 | 图模式(GRAPH_MODE) vs 动态图：编译与执行原理 | 调性能 / 调精度的人 |
| 5 | 分布式：HCCL 集合通信 + 数据并行 | 多卡训练的人 |
| 6 | 迁移要点与常见坑 | 正在迁移的工程师 |
| 7 | 常见问题速查 | 卡住了的人 |

---

## 0. 一句话锚点

**BERT** 是经典的双向 Transformer 编码器（Encoder-only），任务是「掩码语言建模 + 下一句预测」预训练，再在下游任务（分类、问答、NER）上微调。**在昇腾上**，BERT 由 **MindSpore** 框架描述模型，由 **CANN** 把算子编译并下发到 **昇腾 NPU（达芬奇架构）** 上执行——这条「框架→CANN→NPU」的链路，就是你要迁移和调优的全部对象。

把 BERT 当成「迁移练手第一题」：它结构规整（堆叠的 Self-Attention + FFN）、算子常见（MatMul、LayerNorm、Softmax、GELU），是验证「我的昇腾环境通不通、算子全不全、精度对不对」的最佳样例。

---

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层（BERT 在哪一层）

```
┌─────────────────────────────────────────────────────────┐
│  应用层：BERT 预训练脚本 / 下游微调脚本                    │  ← 你写的代码
├─────────────────────────────────────────────────────────┤
│  套件层：MindFormers（大模型套件，含 BERT 配置）           │  ← 可选，开箱即用
├─────────────────────────────────────────────────────────┤
│  框架层：MindSpore（定义网络、自动微分、图编译)            │  ← 等价于 PyTorch
├─────────────────────────────────────────────────────────┤
│  异构计算层：CANN（算子库 + 图编译器 GE + 运行时 + HCCL)   │  ← 等价于 CUDA 全家桶
├─────────────────────────────────────────────────────────┤
│  驱动 + 固件：Ascend Driver / Firmware                    │  ← 等价于 NVIDIA Driver
├─────────────────────────────────────────────────────────┤
│  硬件：昇腾 NPU（达芬奇架构 Cube + Vector + Scalar 单元)   │  ← 等价于 GPU
└─────────────────────────────────────────────────────────┘
```

要点：你的 BERT 代码只碰最上面两层（MindSpore / MindFormers）；越往下越「黑盒」，但**坑往往在 CANN 层**（算子不支持、精度差异、图编译失败）。理解分层，是为了在报错时知道「这是框架的锅还是 CANN 的锅」。

### 1.2 昇腾 ↔ 英伟达生态对照表（迁移心智图）

| 维度 | 英伟达世界 | 昇腾世界 | 一句话对应关系 |
| --- | --- | --- | --- |
| 加速硬件 | GPU | NPU（昇腾，达芬奇架构） | 都是「算力卡」，NPU 以矩阵 Cube 为核心 |
| 底层软件栈 | CUDA + Runtime | CANN（Compute Architecture for Neural Networks） | CANN 是昇腾的「CUDA」 |
| 算子/数学库 | cuDNN / cuBLAS | CANN 算子库（含 AscendCL 算子、AOE 调优） | 卷积/矩阵乘等高性能 kernel |
| 深度学习框架 | PyTorch / TensorFlow | MindSpore（也支持 PyTorch+昇腾插件） | 写模型的地方 |
| 大模型训练套件 | Megatron-LM / HF Transformers | MindFormers / ModelLink | 封装并行 + 配置化训练 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 部署加速、KV-Cache、量化推理 |
| 集合通信库 | NCCL | HCCL（Huawei Collective Comm Lib） | AllReduce/AllGather 等原语 |
| 量化工具 | GPTQ / AWQ 工具链 | msModelSlim | 权重/激活量化压缩 |
| 即时编译/图 | TorchScript / torch.compile | MindSpore GRAPH_MODE（图模式） | 整图编译优化 |
| 数据格式 | TFRecord / WebDataset | MindRecord | 高效训练数据容器 |
| 性能分析 | Nsight / nvprof | MindStudio / msprof | profiling 与可视化 |

> 记住这张表：迁移时遇到的每个英伟达组件，先在右列找对应物，再去查它的差异点。BERT 迁移就是把左列整套换成右列。

---

## 2. 数据流水线：从 Wikipedia 语料到 MindRecord

BERT 预训练需要海量纯文本。经典流程是用维基百科 dump 抽正文。**原始 stub 里的命令就是这一步的第一环：**

```
pip install wikiextractor
python -m wikiextractor.WikiExtractor -o <输出目录> -b <分块大小> <Wikipedia dump 文件>
```

> 注：`wikiextractor` 与昇腾无关，它只是把 `.xml.bz2` 维基 dump 抽成纯文本，是上游数据准备工具。具体命令与版本以官方仓库为准。

完整数据流水线（含义优先于命令）：

```
 Wikipedia dump (.xml.bz2)
        │  wikiextractor 抽正文
        ▼
 纯文本（一行一句/一段）
        │  清洗、分句、去噪
        ▼
 WordPiece 分词（用 BERT 词表 vocab.txt）
        │  生成 token id + segment id + mask
        ▼
 构造预训练样本（MLM 掩码 + NSP 句对）
        │  打包
        ▼
 MindRecord（昇腾/MindSpore 高效数据格式，对标 TFRecord）
        │  MindSpore Dataset 读取 → 喂给 NPU
        ▼
      训练
```

**为什么要转 MindRecord**：MindSpore 的数据管道（`mindspore.dataset`）对 MindRecord 做了零拷贝、并行预取、shard 切分优化，能让 NPU 不「饿着」。这等价于英伟达侧把数据转成 TFRecord/WebDataset 喂满 GPU。**坑**：如果直接用 Python 逐条喂 numpy，数据侧很容易成为瓶颈，NPU 利用率上不去。

> 转换脚本、词表、参数细节以华为昇腾官方文档（Ascend 社区）与 MindFormers/ModelZoo 对应样例为准。

---

## 3. 模型机制：BERT 如何落到达芬奇 Cube/Vector 单元

BERT 一层 = Multi-Head Self-Attention + 前馈网络（FFN），核心算子的硬件归属如下：

| BERT 中的运算 | 数学本质 | 落到达芬奇哪个单元 |
| --- | --- | --- |
| QKV 投影、Attention 打分、FFN 两个线性层 | 矩阵乘 MatMul | **Cube 单元**（矩阵计算核心，吞吐主力） |
| Softmax、LayerNorm、GELU、残差加 | 逐元素/归约 | **Vector 单元**（向量计算） |
| 索引、控制、循环 | 标量逻辑 | **Scalar 单元** |
| 数据搬运 HBM↔片上 Buffer | DMA | **MTE（存储转换引擎）** |

```
         一个 BERT 层在达芬奇上的执行
   ┌──────────────────────────────────────────────┐
   │ Input ─► [Cube: Q,K,V 投影 MatMul]             │
   │            │                                   │
   │            ▼                                   │
   │        [Cube: QKᵀ 打分]                        │
   │            │                                   │
   │            ▼                                   │
   │      [Vector: Softmax]                         │
   │            │                                   │
   │            ▼                                   │
   │        [Cube: 加权 V]  → [Cube: 输出投影]       │
   │            │                                   │
   │            ▼                                   │
   │  [Vector: 残差 + LayerNorm]                     │
   │            │                                   │
   │            ▼                                   │
   │   [Cube: FFN1] → [Vector: GELU] → [Cube: FFN2] │
   │            │                                   │
   │            ▼                                   │
   │  [Vector: 残差 + LayerNorm] ─► Output           │
   └──────────────────────────────────────────────┘
```

**关键洞察**：BERT 的算力几乎全压在 Cube 单元的 MatMul 上——这正是 NPU 设计的甜区。性能好不好，取决于 (1) 矩阵形状能否「打满」Cube（16×16 分形对齐）；(2) Vector 算子（LayerNorm/Softmax）会不会成为串行瓶颈；(3) 数据是否做了 NZ 等昇腾内部格式排布以利搬运。

---

## 4. 图模式 vs 动态图：编译与执行原理

MindSpore 有两种执行模式，BERT 训练强烈推荐**图模式**：

| | 动态图 PYNATIVE_MODE | 图模式 GRAPH_MODE |
| --- | --- | --- |
| 类比 | PyTorch eager | torch.compile / TF static graph |
| 执行方式 | 一行 Python 跑一个算子 | 整网编译成计算图再整体下发 |
| 调试 | 好调，能逐行打印 | 难调，但有报错图谱 |
| 性能 | 低（频繁 host-device 交互） | **高（算子融合、内存复用、下发开销摊薄）** |
| BERT 建议 | 调试期用 | **正式训练用** |

```
  GRAPH_MODE 下 BERT 的编译-执行链路
  MindSpore 网络定义
        │ 自动微分 + 构图
        ▼
  计算图 IR
        │ CANN GE（图引擎）：算子融合 / 常量折叠 / 内存规划 / 格式转换
        ▼
  下发到 NPU 的可执行 kernel 序列
        │ 一次下发，整图在 NPU 上跑
        ▼
       结果回 host
```

**机制要点**：图模式把「Python 解释 + 逐算子下发」的 host 开销几乎消掉，并允许 CANN 做**算子融合**（如 LayerNorm 内部几步合一、Attention 融合）。这是昇腾上拿性能的第一杠杆。代价是首次编译耗时（图编译 + AOE 自动调优），且对动态 shape 不友好——**BERT 要固定 seq_len、固定 batch，避免触发反复重编译**。

---

## 5. 分布式：HCCL 集合通信 + 数据并行

BERT-Large 单卡放得下，但预训练靠多卡数据并行加速。昇腾用 **HCCL**（对标 NCCL）做梯度同步。

```
   数据并行下的 AllReduce（Ring 环算法，HCCL ≈ NCCL）
   NPU0 ──grad──► NPU1 ──grad──► NPU2 ──grad──► NPU3
     ▲                                            │
     └────────────────grad────────────────────────┘
   每张卡算自己 batch 的梯度 → Ring AllReduce 求和平均 → 各卡更新一致权重
```

机制层面：
- **Ring AllReduce**：梯度切片在环上传递，带宽利用率高、与卡数弱相关，HCCL 与 NCCL 思路一致。
- **rank_table / 组网**：昇腾多卡需要 HCCL 的设备组网信息（卡间用 HCCS/RoCE 互联），这一步替代了 NCCL 的拓扑自发现。具体配置文件格式与生成方式以昇腾官方文档为准。
- **梯度累积 + 混合精度**：BERT 常用 FP16/BF16 计算 + FP32 主权重，配 Loss Scale 防下溢，机制与英伟达侧 AMP 一致，但 Loss Scale 策略和算子精度白名单可能不同。

> 多机多卡的 rank_table 生成、网卡配置、环境变量等具体命令与版本以华为昇腾官方文档（Ascend 社区）为准。

---

## 6. 迁移要点与常见坑（CUDA+PyTorch → 昇腾+MindSpore）

**改什么（心智清单）**：

1. **设备指定**：`cuda` → 昇腾设备（在 MindSpore 里通过 `set_context` 指定 device_target 为 Ascend，并设 GRAPH_MODE）。
2. **框架 API**：PyTorch 的 `nn.Module/optimizer/dataloader` → MindSpore 的 `nn.Cell` / `mindspore.dataset` / `Model.train`。语义相近但命名与默认行为有别。
3. **数据格式**：TFRecord/自定义 → MindRecord。
4. **通信库**：NCCL → HCCL（多卡组网换成 rank_table 思路）。
5. **混合精度**：AMP → MindSpore 的混合精度 + Loss Scale 管理。

**最容易踩的坑**：

| 坑 | 现象 | 应对思路 |
| --- | --- | --- |
| 动态 shape 反复重编译 | 训练前期奇慢、每个新 batch 卡顿 | 固定 seq_len/batch，padding 到统一长度 |
| 算子不支持/精度差异 | 报「算子未注册」或 loss 对不上 | 查 CANN 算子清单；用 BF16；对齐 LayerNorm/Softmax 实现 |
| 数据管道喂不满 NPU | NPU 利用率低、卡在 host | 转 MindRecord、开并行预取、增大 prefetch |
| 首个 step 极慢 | 图编译 + AOE 调优耗时 | 正常现象，编译结果可缓存复用 |
| Loss Scale 不当 | FP16 下 loss 变 NaN/不降 | 调整初始 scale、用动态 Loss Scale |
| 把 PyNative 当生产模式 | 性能远低于预期 | 训练切 GRAPH_MODE |

**性能调优思路（机制层面）**：优先保证算子在 Cube 上的形状对齐（维度凑 16 的倍数）；用图模式吃算子融合；用 msprof/MindStudio 看哪类单元（Cube/Vector/MTE）是瓶颈；数据侧确保不饿卡；多卡看 HCCL 通信是否与计算重叠。

---

## 7. 常见问题

| 问题 | 解答 |
| --- | --- |
| 必须用 MindSpore 吗？ | 不必。PyTorch 也能通过昇腾适配插件跑 BERT；但 MindSpore + 图模式在昇腾上最原生、调优空间最大。 |
| MindSpore 等于 PyTorch 吗？ | 定位等价（都是框架层），但 API、默认动态/静态图策略不同。迁移要逐 API 对照。 |
| BERT 是 Encoder-only，和 GPT 啥区别？ | BERT 双向编码做理解类任务；GPT 是 Decoder-only 做生成。算子构成类似，并行/KV-Cache 需求不同。 |
| 为什么强调固定 shape？ | 图模式按 shape 编译，变 shape 会重编译，BERT 这种定长任务正好适配。 |
| CANN 是什么？ | 昇腾的「CUDA」：算子库 + 图编译器 GE + 运行时 + HCCL，承接框架到硬件。 |
| Cube 和 Vector 单元区别？ | Cube 专做矩阵乘（MatMul 主力），Vector 做逐元素/归约（LayerNorm、Softmax、GELU）。 |
| wikiextractor 是昇腾工具吗？ | 不是，它只是上游抽维基语料的通用工具，与昇腾无关。 |
| 多卡训练换什么？ | NCCL 换 HCCL，组网用 rank_table 思路；AllReduce 算法（Ring）思想一致。 |

---

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

> 免责声明：本文聚焦机制、流程与迁移心智。所有精确命令行、包名、版本号、路径与性能数字，均以华为昇腾官方文档（Ascend 社区）及 MindSpore / MindFormers 官方仓库为准。
