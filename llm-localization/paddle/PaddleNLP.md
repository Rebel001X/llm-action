# PaddleNLP（飞桨大模型套件）

> PaddleNLP 是基于百度飞桨（PaddlePaddle）框架的大语言模型训练 / 微调 / 推理 / 部署一站式套件，是国产「框架 + 套件」自主路线在 NLP/LLM 方向的代表，与昇腾 NPU 等国产算力适配后形成「国产芯片 + 国产框架 + 国产套件」的全栈方案。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/huggingface-transformers/README]] [[ai-framework/megatron-lm/README]] [[llm-train/README]] [[llm-inference/README]]

## 阅读地图

| 章节 | 内容 | 你会得到 |
| --- | --- | --- |
| 0 | 一句话锚点 | 30 秒抓住 PaddleNLP 是什么 |
| 1 | 在国产栈/昇腾栈的定位 + 对标英伟达生态 | HF/Megatron/vLLM → Paddle 的迁移心智图 |
| 2 | 飞桨框架地基：动静统一与 IR | 为什么 Paddle 能既好调试又能图优化 |
| 3 | PaddleNLP 的三大能力面 | 训练 / 微调 / 推理部署一张图 |
| 4 | 4D 混合并行机制 | DP/TP/PP/Sharding 在 Paddle 怎么拼 |
| 5 | 推理与部署：FastDeploy / Paddle Inference | 训练完怎么落地服务 |
| 6 | 在昇腾 NPU 上跑：适配层原理 | CustomDevice 如何把 Paddle 接到 CANN |
| 迁移 | 从 PyTorch/HF 迁到 PaddleNLP | 要改什么、易踩的坑 |
| FAQ | 常见问题 | 排错速查 |

## 0. 一句话锚点

**PaddleNLP 之于飞桨，约等于 Transformers + Megatron + 部分 vLLM 之于 PyTorch 生态。** 它把「拿预训练模型 → 微调 → 大规模分布式训练 → 量化压缩 → 高性能推理服务」这一整条 LLM 流水线，封装在一个以飞桨为底座的套件里。

当你在英伟达 GPU + PyTorch 上习惯了 `from transformers import AutoModel`、`Trainer`、`vllm.LLM(...)` 这套链路；在国产路线里，对应的就是飞桨框架 + PaddleNLP 的 `AutoModelForCausalLM`、`Trainer`、`predictor`。心智模型几乎一一对应，只是把「PyTorch + HF + Megatron」这套换成「Paddle + PaddleNLP」这套。

> 记忆钩子：**Paddle** 是底座框架（对标 PyTorch），**PaddleNLP** 是跑在它上面的 NLP/LLM 套件（对标 HF Transformers + Megatron-LM）。一个是「发动机」，一个是「整车解决方案」。

## 1. 地基：在国产栈/昇腾栈的定位 + 对标英伟达生态

### 1.1 软件栈分层（以昇腾 NPU 为例）

PaddleNLP 不直接碰硬件，它站在「框架层之上的套件层」。从硬件往上看，一条完整的国产化链路长这样：

```
┌───────────────────────────────────────────────────────────────┐
│  套件层    ★ PaddleNLP ★   (训练/微调/压缩/推理，类 HF+Megatron) │
├───────────────────────────────────────────────────────────────┤
│  框架层    PaddlePaddle 飞桨   (动静统一，类 PyTorch)            │
│            └─ CustomDevice 适配层（把后端硬件接进来）            │
├───────────────────────────────────────────────────────────────┤
│  使能层    CANN（昇腾） / CUDA（英伟达） / 其他国产 SDK          │
│            算子库(类cuDNN) · Runtime(类CUDART) · HCCL(类NCCL)    │
├───────────────────────────────────────────────────────────────┤
│  硬件层    昇腾 NPU(达芬奇) / 英伟达 GPU / 其他国产 AI 芯片        │
└───────────────────────────────────────────────────────────────┘
```

关键认知：**PaddleNLP 是「框架无关于具体硬件」的**——它写在飞桨 API 上，飞桨再通过 CustomDevice 等机制把算子下发到 GPU 或 NPU。所以同一份 PaddleNLP 代码，理论上换个后端就能从 GPU 迁到昇腾 NPU，硬件差异被框架适配层吸收掉（第 6 节细讲）。

### 1.2 「飞桨生态 ↔ 英伟达/PyTorch 生态」对照表

这是迁移时最该记住的一张表：

| 角色 / 层次 | 英伟达 / PyTorch 世界 | 飞桨 / 国产世界 | 说明 |
| --- | --- | --- | --- |
| 深度学习框架 | PyTorch | **PaddlePaddle（飞桨）** | 都是动态图为主、支持静态图编译 |
| 张量 / 自动微分 | `torch.Tensor` / autograd | `paddle.Tensor` / autograd | API 命名高度相似 |
| 预训练模型库 | HF Transformers | **PaddleNLP（Transformers 模块）** | `AutoModel` / `AutoTokenizer` 同名同构 |
| 大模型分布式训练 | Megatron-LM / DeepSpeed | **PaddleNLP Trainer + Fleet** | 4D 混合并行、ZeRO/Sharding |
| 训练编排接口 | HF `Trainer` | PaddleNLP `Trainer` | 参数语义大体对齐 |
| 高性能推理 | vLLM / TensorRT-LLM | **PaddleNLP 高性能推理 + FastDeploy** | 算子融合、量化、PagedAttention 类机制 |
| 量化压缩 | GPTQ / AWQ / bitsandbytes | **PaddleSlim + PaddleNLP 量化** | PTQ/QAT、WInt8/WInt4 等 |
| 集合通信库 | NCCL | **HCCL（昇腾）** / NCCL（GPU） | 由框架后端选择，PaddleNLP 不感知 |
| 加速库 | cuDNN / cuBLAS | CANN 算子库（昇腾） / cuDNN（GPU） | 框架算子下沉到此层 |
| 计算平台 SDK | CUDA | CANN（昇腾） | 框架通过它访问硬件 |
| 加速硬件 | GPU | 昇腾 NPU / 其他国产 AI 芯片 | 达芬奇 Cube/Vector 架构 |

> 一句话总结这张表：**PaddleNLP 同时承担了 HF Transformers（模型库）、Megatron-LM（并行训练）、vLLM/TensorRT-LLM（推理服务）三件事，只是底座换成了飞桨**。这是它和「只做模型库」的 HF 最大的不同。

## 2. 飞桨框架地基：动静统一与中间表示

要理解 PaddleNLP 为什么既能像 PyTorch 一样好调试、又能做图优化加速，得先理解飞桨的「动静统一」。

- **动态图（命令式）**：和 PyTorch 一样逐行执行，调试直观、写起来灵活。开发、调参阶段用它。
- **静态图（声明式）**：先构建计算图 IR，再整体编译优化（算子融合、内存复用、并行调度），跑得更快。训练大规模 / 部署阶段用它。
- **动转静（dygraph to static）**：飞桨提供机制把动态图代码自动转成静态图，从而「开发用动态图，加速用静态图」。

```
   开发期（动态图）                    生产期（静态图）
 ┌──────────────────┐   动转静     ┌────────────────────────┐
 │ 逐行执行          │ ──────────▶ │ 整图编译                │
 │ 好调试/灵活       │             │ 算子融合·内存复用·调度   │
 │ 对标 PyTorch eager│             │ 对标 torch.compile/图模式│
 └──────────────────┘             └────────────────────────┘
```

这一点和昇腾的「图模式」理念天然契合：昇腾 CANN 也偏好把整张计算图下发给 GraphEngine 做整图优化，飞桨静态图正好能把完整 IR 交给后端，让 CANN 充分做算子融合和调度，发挥达芬奇架构的吞吐。

## 3. PaddleNLP 的三大能力面

PaddleNLP 的功能可以归成「三大面」，理解这三面就理解了整个套件：

```
                         ┌─────────────────────────┐
                         │       PaddleNLP          │
                         └─────────────────────────┘
            ┌───────────────────┼────────────────────┐
            ▼                   ▼                    ▼
   ① 模型与微调            ② 大规模训练          ③ 压缩与推理部署
   ────────────           ────────────         ────────────────
   AutoModel/Tokenizer     Trainer + Fleet       量化(PaddleSlim)
   预训练权重一键加载       4D 混合并行           高性能推理 predictor
   SFT/LoRA/PEFT 等微调     断点续训/混合精度     FastDeploy/Paddle Inference
   （类 HF Transformers）  （类 Megatron/DeepSpeed）（类 vLLM/TensorRT-LLM）
```

- **① 模型与微调面**：`AutoModelForCausalLM.from_pretrained(...)`、`AutoTokenizer`，与 HF 几乎同名。内置 SFT、LoRA/PEFT 等参数高效微调能力。
- **② 大规模训练面**：通过 `Trainer` + 飞桨分布式（Fleet）做 4D 混合并行（第 4 节），对应 Megatron-LM + DeepSpeed 的角色。
- **③ 压缩与部署面**：训练完的模型经量化压缩后，用高性能推理引擎服务化，对应 vLLM/TensorRT-LLM 的角色。

> 具体支持的模型清单、API 签名、可用的微调算法以 PaddleNLP 官方文档与对应版本 Release Notes 为准——这类清单随版本变化快，不要凭记忆套用。

## 4. 4D 混合并行机制（大模型训练的核心）

大模型训练放不进单卡，必须切。PaddleNLP 借飞桨分布式，把四种并行维度叠在一起，这套思路和 Megatron-LM + DeepSpeed ZeRO 同源：

| 并行维度 | 切什么 | 对标 | 通信原语 |
| --- | --- | --- | --- |
| 数据并行 DP | 切数据（每卡完整模型副本） | PyTorch DDP | AllReduce 同步梯度 |
| 张量并行 TP | 切单层权重矩阵（行/列切分） | Megatron TP | AllReduce / AllGather |
| 流水并行 PP | 切层（不同卡放不同层段） | Megatron PP / GPipe | P2P 点对点传激活 |
| 分组切片 Sharding | 切优化器状态/梯度/参数 | DeepSpeed ZeRO-1/2/3 | ReduceScatter + AllGather |

把四者叠起来，就是「4D 混合并行」：

```
        ┌──────── Pipeline 维度（切层，卡间传激活）────────┐
        │  stage0    stage1    stage2    stage3            │
        │  ┌────┐   ┌────┐   ┌────┐   ┌────┐               │
  TP →  │  │卡组│   │卡组│   │卡组│   │卡组│   每个卡组内再做  │
 (切权重)│  └────┘   └────┘   └────┘   └────┘   张量并行     │
        └──────────────────────────────────────────────────┘
                ▲ 整体再复制成多份做 DP（切数据）
                ▲ DP 组内再叠 Sharding（切优化器状态，省显存）
```

**机制要点**：这些通信（AllReduce / AllGather / ReduceScatter / P2P）在 GPU 上落到 NCCL，在昇腾 NPU 上落到 **HCCL**。PaddleNLP 本身不感知用的是哪个通信库——它只调飞桨的分布式集合通信 API，由框架后端按硬件选择。所以从 GPU 迁到昇腾，并行策略代码基本不动，换的是底层通信库（参见 [[ai-infra/网络/NCCL]] 与 HCCL 的对照）。

## 5. 推理与部署：从训练产物到线上服务

训练 / 微调出权重后，要落地成服务。PaddleNLP 这条链路对应英伟达世界的 vLLM / TensorRT-LLM + Triton：

```
  训练产物(权重)
       │
       ▼  ① 压缩（可选）：PTQ/QAT 量化、WInt8/WInt4，省显存提吞吐（类 GPTQ/AWQ）
       ▼  ② 导出：动转静 → 静态图模型（整图，便于后端优化）
       ▼  ③ 高性能推理：算子融合、KV-Cache、连续批处理（类 vLLM PagedAttention）
       ▼  ④ 服务化：FastDeploy / Paddle Inference 起服务，OpenAI 兼容接口等
  线上服务
```

- **量化压缩**：由 PaddleSlim / PaddleNLP 量化能力承担，机制上和 GPTQ/AWQ 一类——用少量校准数据做训练后量化（PTQ），把权重压到低比特，换显存和带宽。
- **高性能推理**：算子融合、KV-Cache 管理、连续批处理（continuous batching）等优化思路，与 vLLM/TensorRT-LLM 一致，目标都是把吞吐和显存利用率拉满。
- **服务化部署**：FastDeploy / Paddle Inference 负责把推理引擎包成可调用的服务。

> 具体的导出命令、量化配置项、服务启动参数、OpenAI 兼容接口的字段，以 PaddleNLP / FastDeploy 官方文档为准；版本间差异较大，不要硬记。

## 6. 在昇腾 NPU 上跑：适配层原理

这是「国产框架 + 国产芯片」全栈方案的关键一环。飞桨通过 **CustomDevice（自定义后端）** 机制把昇腾接进来：

```
  PaddleNLP 代码（写在飞桨 API 上，硬件无关）
        │  调 paddle.xxx 算子 / 分布式 API
        ▼
  PaddlePaddle 框架
        │  通过 CustomDevice 插件路由到具体后端
        ├──────────────┬─────────────────┐
        ▼              ▼                 ▼
   GPU 后端        昇腾 NPU 后端       其他国产后端
   (走 CUDA/cuDNN) (走 CANN 算子库)   (走各自 SDK)
                       │
                       ▼
              达芬奇架构 Cube/Vector 单元
              通信走 HCCL（替代 NCCL）
```

**机制要点**：

- CustomDevice 是飞桨的「后端插件接口」，硬件厂商按这套接口把自家算子库（昇腾就是 CANN）对接进来，框架上层无感。这等价于「换一套底层算子实现，上层模型代码不动」。
- 算子下沉：PaddleNLP 用到的矩阵乘、Attention 等算子，在昇腾后端会映射到 CANN 提供的高性能算子，由达芬奇架构的 Cube（矩阵）/ Vector（向量）单元执行——参见 [[ai-infra/算力/昇腾NPU]]。
- 通信下沉：分布式训练的集合通信在昇腾后端走 HCCL，环算法 / 拓扑感知由 HCCL 负责。

> 哪些昇腾型号 / CANN 版本 / 飞桨版本组合被官方验证支持、镜像怎么拉、环境怎么装，**这些一律以华为昇腾官方文档（Ascend 社区）与 PaddlePaddle/PaddleNLP 官方适配说明为准**。安装类操作重在理解依赖关系（驱动 → CANN → 飞桨 → PaddleNLP，版本必须互相匹配），而不是背命令。

## 迁移要点：从 PyTorch/HF 迁到 PaddleNLP（含常见坑）

理解了上面的对照，迁移就是「把心智模型平移」。要点与坑：

1. **API 平移而非照搬**：`torch.Tensor` → `paddle.Tensor`，`from transformers import AutoModel` → `from paddlenlp.transformers import AutoModel`。大量 API 同名，但**默认行为、参数名细节、张量布局未必完全一致**，逐个核对而非假设等价。
2. **权重格式不同**：HF 的 `.safetensors`/PyTorch checkpoint 与飞桨权重格式不同，跨生态迁移需要做**权重转换**（layout、命名映射、是否转置）。这是最常见的坑——转换不当会「能加载但结果错」。
3. **动静切换要早想清楚**：开发用动态图爽，但部署 / 大规模训练要走静态图（动转静）。代码里如果用了动态图特有的 Python 控制流，动转静时可能需要改写。
4. **并行策略是配置不是重写**：4D 并行靠配置切分维度，不需要手改模型层。但 TP/PP 的切分要和模型结构、卡数匹配，配错会 OOM 或通信死锁。
5. **从 GPU 迁到昇腾 NPU 时，瓶颈往往在「算子覆盖」而非框架**：某个算子若昇腾后端尚未支持或未优化，会回退或报错。排查思路：先确认该算子在 CANN 后端是否被支持、是否有等价融合算子，再考虑改写或换实现。
6. **性能调优在机制层**：优先吃满静态图整图优化（让 CANN 做融合）、用混合精度（BF16/FP16）、用上量化、把通信和计算重叠（流水并行 + 通信掩盖）。这些是机制层抓手，和具体厂商无关。
7. **版本匹配是头号环境坑**：驱动 / 固件 ↔ CANN ↔ 飞桨 ↔ PaddleNLP 四者版本必须互相兼容，错配会出现「能装上但跑不起来 / 算子报错」。**具体匹配矩阵以官方文档为准**。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| PaddleNLP 和 PaddlePaddle 什么关系？ | Paddle 是底座框架（类 PyTorch），PaddleNLP 是跑在它上面的 NLP/LLM 套件（类 HF Transformers + Megatron） |
| 它对标英伟达世界的谁？ | 同时覆盖 HF Transformers（模型库）+ Megatron-LM（并行训练）+ vLLM/TensorRT-LLM（推理部署）三个角色 |
| 能在昇腾 NPU 上跑吗？ | 能。通过飞桨 CustomDevice 接 CANN，算子下沉到达芬奇架构，通信走 HCCL |
| 从 HF/PyTorch 迁过来最大的坑？ | 权重格式转换 + 个别算子在昇腾后端的覆盖情况 |
| 大模型训练怎么切？ | 4D 混合并行：DP（切数据）+ TP（切权重）+ PP（切层）+ Sharding（切优化器状态，类 ZeRO） |
| 训练完怎么部署？ | 量化压缩 → 动转静导出 → 高性能推理 → FastDeploy/Paddle Inference 服务化 |
| 安装命令/版本去哪查？ | 一律以 PaddlePaddle/PaddleNLP 官方文档与华为昇腾官方文档（Ascend 社区）为准 |
| 通信用 NCCL 还是 HCCL？ | 套件不感知，由框架后端按硬件选：GPU→NCCL，昇腾→HCCL |

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 昇腾硬件与平台：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 通信相关：[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 训练框架与套件：[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 量化压缩：[[llm-compression/quantization/量化基础]]
- 训练与推理：[[llm-train/README]] · [[llm-inference/README]]
- 模型架构：[[llm-algo/transformer/模型架构]]
