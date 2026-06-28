# 大模型面试题库（LLM Interview）

> 面向大模型算法/工程/Infra 岗位的系统化面试复习索引：从「基础 → 结构 → 训练 → 微调 → 对齐 → 评估 → 压缩 → 推理 → 应用 → 综合」一条主线，把零散八股串成知识树。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] [[llm-train/README]] [[llm-inference/README]] [[llm-alignment/RLHF]]

## 阅读地图

| 你是谁 / 你想要 | 先看哪一节 | 配套子文件 |
| --- | --- | --- |
| 完全新手，想知道怎么用这个题库 | 第 0~1 节 | 全部 |
| 算法岗（建模/对齐方向） | 第 2 节「能力地图」之 基础/结构/对齐 | `base.md` `llm-algo.md` `llm-rlhf.md` |
| 工程/Infra 岗（训练/推理优化） | 第 2 节之 训练/压缩/推理 | `llm-train.md` `llm-compress.md` `llm-inference.md` |
| 应用/RAG/Agent 岗 | 第 2 节之 应用 | `llm-app.md` |
| 面试前一晚抱佛脚 | 第 4 节「高频考点速查」 | `comprehensive.md` |
| 想知道怎么答得有层次 | 第 5 节「答题方法论」 | — |

## 0. 一句话锚点

> **这个目录是一套「按大模型生命周期组织」的中文面试题库**：每个 `.md` 对应一个能力域，本 README 是总目录 + 复习路线图 + 答题方法论。它不替你背答案，而是给你一张「知识地图」，让你在面试官顺着任意一条线往下追问时，都知道自己站在树的哪根枝上、下一层是什么。

## 1. 地基：大模型面试到底在考什么？

很多人把大模型面试当成「背 100 道八股」，结果一被追问就崩。要先理解面试官的真实意图，再谈复习。

面试本质是在**有限时间内估计两件事**：

1. **知识的「深度」**——你是否把一个概念拆到了原子级（知其所以然），还是只记住了结论（知其然）。
2. **知识的「连通性」**——你能否在「训练时的某个选择」和「推理时的某个现象」之间建立因果链。

举例：面试官问「为什么用 RMSNorm 不用 LayerNorm」，

- 背书式答案：「RMSNorm 更快」——浅，且容易被追问打穿。
- 拆原子式答案：「LayerNorm 要算均值和方差两个统计量并做中心化；RMSNorm 去掉了减均值这一步，只用均方根缩放。去掉中心化后，少一遍 reduce、少存均值，**显存和带宽都省**；同时实践发现去掉中心化对收敛影响很小（因为残差流本身已带偏置信息）。代价是理论上对分布偏移的鲁棒性略弱。」——这就同时展示了「深度」（拆到算子级）和「连通性」（连到了显存/带宽这个 Infra 维度）。

所以题库的组织逻辑不是「知识点列表」，而是**沿大模型的生命周期串成一条因果链**：

```
                         大模型生命周期 = 面试主线
   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
   │  基础     │  结构     │  训练     │  对齐     │  应用     │
   │ base.md  │llm-algo  │llm-train │llm-rlhf  │ llm-app  │
   │ 注意力    │Transformer│ 并行策略 │ SFT/RLHF │ RAG/Agent│
   │ 位置编码  │ MoE/MLA  │ 混合精度 │ DPO/PPO  │ 提示工程  │
   └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
        │          │          │          │          │
        └──────────┴────► 横切关注点 ◄────┴──────────┘
                    评估(llm-eval) · 压缩(llm-compress) · 推理(llm-inference)
                    └─────────► 综合(comprehensive) ◄─────────┘
```

「横切」的意思是：评估、压缩、推理这三块**不属于某一个生命周期阶段，而是贯穿所有阶段**——训练完要评估，部署前要压缩，上线后要优化推理。综合题则把多条线交叉起来考。

## 2. 能力地图：九大题库逐个定位

下面把每个文件「考什么、为什么重要、典型追问链」说清楚。

### 2.1 大模型基础 —— `base.md`

> 定位：Transformer 之前的「数学与机制地基」。这一块答不好，后面全是空中楼阁。

核心议题：自注意力的 QKV 机制、缩放因子 $\frac{1}{\sqrt{d_k}}$ 为什么需要、多头注意力为什么要拆头、位置编码（绝对/相对/RoPE）、激活函数（GeLU/SwiGLU）、归一化（Pre-LN vs Post-LN、RMSNorm）、Softmax 数值稳定性、梯度消失/爆炸。

典型追问链（面试官常这样一路往下钻）：

```
自注意力是什么？
   └► 为什么要除以 √d_k？
        └► 不除会怎样？(点积方差随 d_k 增大 → Softmax 进入饱和区 → 梯度趋零)
             └► 那为什么是 √d_k 不是 d_k？(让点积方差归一化到 1)
                  └► 多头时每个头的 d_k 变小，缩放还成立吗？(成立，按每头维度算)
```

### 2.2 大模型结构 —— `llm-algo.md`

> 定位：把「积木」拼成「整机」。考的是具体模型的架构选型与演进。

核心议题：Decoder-only 为何成为主流（vs Encoder-Decoder）、GPT/LLaMA/Qwen 架构差异、MoE（专家混合）、GQA/MQA（分组/多查询注意力）、MLA（多头潜在注意力，DeepSeek）、KV Cache 的来由、长上下文方案（NTK/YaRN 外推）。

一个常被忽略的连通点：**GQA/MQA/MLA 这一组选型，本质是「推理时 KV Cache 显存」与「模型质量」之间的权衡**——它跨越了「结构」和「推理」两条线，是高频综合题。

### 2.3 大模型训练 —— `llm-train.md`

> 定位：Infra 岗的主战场。考的是「如何把一个塞不进单卡的模型训起来」。

核心议题：数据并行（DP/DDP/FSDP/ZeRO 三阶段）、张量并行（TP）、流水线并行（PP）、序列并行（SP）、3D 并行组合、混合精度（FP16/BF16/FP8）、梯度累积、梯度检查点（重计算换显存）、通信原语（AllReduce/AllGather/ReduceScatter）。

并行策略的「显存 vs 通信」权衡，可以用一张图记住：

```
   单卡装不下模型，怎么切？  ——三个维度，各有代价

   DP/ZeRO  ：按「数据样本」切，复制模型      → 通信省、显存费(ZeRO缓解)
   TP       ：按「权重矩阵列/行」切，切层内   → 显存省、通信极重(须高带宽NVLink)
   PP       ：按「层」切，切层间             → 显存省、有流水线气泡(bubble)
        │
        └─► 实践：大模型 = TP(机内NVLink) × PP(机间) × DP/ZeRO(扩batch)
                  谁在快网络谁切得细 —— TP 放机内，PP/DP 放机间
```

### 2.4 大模型微调 —— `llm-ft.md`

> 定位：让通用模型适配下游任务。算法+工程都会问。

核心议题：全量微调 vs 参数高效微调（PEFT）、LoRA（低秩分解 $W + BA$ 的原理与秩 $r$ 的权衡）、QLoRA（4bit 量化 + LoRA）、Adapter、Prefix/Prompt Tuning、灾难性遗忘、指令微调（SFT）数据构造。

LoRA 的核心直觉一定要会推：全量微调更新量 $\Delta W$ 是个满秩大矩阵；LoRA 假设 $\Delta W$ 是**低秩**的（$\Delta W \approx BA$，$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}$，$r \ll d$），于是可训练参数从 $d\times k$ 降到 $r\times(d+k)$，显存和存储都大幅下降，且推理时可把 $BA$ 合并回 $W$ 做到零额外延迟。

### 2.5 大模型对齐 / RLHF —— `llm-rlhf.md`

> 定位：从「会说话」到「说人话且安全」。近两年面试占比飙升。

核心议题：RLHF 三阶段（SFT → 奖励模型 RM → PPO）、奖励模型如何训（成对偏好 + Bradley-Terry）、PPO 的 actor/critic/reference/reward 四模型、KL 惩罚为什么必须有（防止策略漂离 SFT 模型）、DPO（绕过显式 RM 直接优化偏好）、GRPO（DeepSeek，去掉 critic 用组内相对优势）、奖励 hacking。

### 2.6 大模型评估 —— `llm-eval.md`

> 定位：横切关注点。「你怎么证明你的模型/优化是好的？」

核心议题：困惑度（PPL）、自动指标（BLEU/ROUGE 的局限）、基准（MMLU/GSM8K/HumanEval）、LLM-as-a-Judge 及其偏置、人工评估、推理性能指标（吞吐 throughput、首 token 延迟 TTFT、token 间延迟 TPOT/ITL）、评估的数据污染。

### 2.7 大模型压缩 —— `llm-compress.md`

> 定位：横切关注点，部署前的「瘦身」。Infra 岗高频。

核心议题：量化（PTQ vs QAT、INT8/INT4、对称/非对称、per-tensor/per-channel/per-group、GPTQ/AWQ/SmoothQuant 的思路差异、KV Cache 量化）、剪枝（结构化 vs 非结构化、2:4 稀疏）、知识蒸馏（logits/特征/数据蒸馏）。

量化的本质一句话：用更少的比特表示数值，**省显存、省带宽、（在支持的硬件上）省算力**，代价是精度损失——而各种算法（AWQ/GPTQ/SmoothQuant）的差异，全在「**如何在量化时尽量保住重要权重/激活的精度**」这一点上。

### 2.8 大模型推理 —— `llm-inference.md`

> 定位：横切关注点，上线后的「提速降本」。vLLM/TensorRT-LLM 等引擎都在这。

核心议题：Prefill vs Decode 两阶段、KV Cache 与 PagedAttention（vLLM）、Continuous Batching（连续批处理）、投机解码（Speculative Decoding）、FlashAttention（IO 感知的注意力）、量化推理、并行（TP/PP 推理部署）、计算密集 vs 访存密集（Decode 阶段是典型 memory-bound）。

一个必答的认知：**Prefill 是计算密集（compute-bound），Decode 是访存密集（memory-bound）**。这条结论解释了「为什么 Decode 要靠 batching 提吞吐」「为什么 KV Cache 量化对 Decode 增益大」等一连串现象。

### 2.9 大模型应用 & 综合 —— `llm-app.md` / `comprehensive.md`

> 定位：落地与交叉。RAG、Agent、提示工程，以及把多条线串起来的开放题。

核心议题：RAG（检索增强：分块、Embedding、向量库、重排、幻觉抑制）、Agent（ReAct、工具调用、记忆、规划）、提示工程（Few-shot/CoT）、Function Calling、长文本处理、幻觉成因与缓解。综合题则是「给定 X 资源/约束，你怎么训/部署一个模型」这类系统设计。

## 3. 推荐复习路线（按岗位）

不同岗位不必平均用力。下面给三条主路线，箭头表示「先后顺序」。

```
算法/建模岗:   base ─► llm-algo ─► llm-ft ─► llm-rlhf ─► llm-eval ─► comprehensive
工程/Infra岗:  base ─► llm-algo ─► llm-train ─► llm-compress ─► llm-inference ─► comprehensive
应用/RAG岗:    base ─► llm-algo ─► llm-ft ─► llm-app ─► llm-eval ─► comprehensive
                │
                └─ 共同起点：base + llm-algo 是所有岗位的「公共必修」，
                   因为任何深入追问最后都会落回 Transformer 的某个细节。
```

## 4. 高频考点速查（背完这张表能应付大半）

| 主题 | 一句话要点 | 易被追问处 |
| --- | --- | --- |
| 注意力缩放 | 除 $\sqrt{d_k}$ 防点积方差过大致 Softmax 饱和 | 为什么是平方根 |
| RoPE | 用旋转矩阵把相对位置编进 Q/K，支持外推 | NTK/YaRN 如何扩长上下文 |
| KV Cache | 缓存历史 K/V 避免 Decode 重算，空间换时间 | 显存随序列线性增长 → PagedAttention |
| GQA/MQA | 多个 Q 头共享少量 KV 头，省 KV Cache | 与质量的权衡、MLA 的改进 |
| ZeRO 三阶段 | 依次切优化器状态/梯度/参数，降冗余显存 | 与 FSDP 的关系、通信开销 |
| LoRA | 低秩 $\Delta W=BA$，省可训练参数 | 秩 $r$ 怎么选、推理可合并 |
| RLHF | SFT→RM→PPO，KL 约束防漂移 | DPO/GRPO 怎么简化掉哪些模块 |
| 量化 | 低比特表示省显存带宽，算法在保精度 | PTQ/QAT、AWQ vs GPTQ 思路 |
| FlashAttention | 分块 + 不落地中间矩阵，IO 感知省 HBM 读写 | 为什么省的是显存带宽不是 FLOPs |
| Prefill/Decode | 前者 compute-bound 后者 memory-bound | 各自的优化手段不同 |
| RAG 幻觉 | 检索质量 + 重排 + 提示约束共同抑制 | 分块策略、召回率/精确率权衡 |

> 表中所有「机制」都是稳定知识；涉及某个具体框架的**确切参数名、默认值、版本号**时，请以对应官方文档/源码为准，面试时讲清「这一类参数是做什么用的、怎么权衡」远比背默认值有用。

## 5. 答题方法论：怎么把题答出层次

把每道题用「**4 层漏斗**」组织，面试官追到哪一层你都接得住：

```
   ┌─────────────────────────────────────────────┐
 第1层  是什么 + 解决什么问题   ← 必答，30秒讲清定义与动机
   ├─────────────────────────────────────────────┤
 第2层  核心机制 / 怎么做      ← 拆到算子/数据流级，配公式或小例子
   ├─────────────────────────────────────────────┤
 第3层  权衡 / 代价 / 边界     ← 讲「省了什么、付出什么、何时不适用」
   ├─────────────────────────────────────────────┤
 第4层  连通 / 对比 / 实践坑    ← 连到训练或推理的另一条线 + 踩过的坑
   └─────────────────────────────────────────────┘
```

第 3、4 层是把「合格」拉到「优秀」的关键——大多数候选人只答到第 2 层。

## 常见问题 / 坑

| 坑 | 现象 | 纠正 |
| --- | --- | --- |
| 只背结论不拆原理 | 一被追问「为什么」就卡住 | 用第 1 节的「拆原子」思路，每个结论都问自己三个为什么 |
| 各知识点孤立 | 答不出「训练选择如何影响推理表现」 | 用第 4 层「连通」，刻意练习跨线因果链 |
| 死记默认值/版本号 | 记错了反而暴露不熟 | 讲机制与权衡，明确说「确切参数以官方文档为准」 |
| 公式硬背不懂含义 | 写得出 $\frac{1}{\sqrt{d_k}}$ 但说不出为什么 | 每个公式都要能用一句白话解释它在「修正什么」 |
| 不分岗位平均用力 | 时间不够、深度不足 | 按第 3 节选主路线，公共必修 + 岗位专修 |
| 忽视评估/Infra 横切 | 只会建模，答不了「怎么证明好」「怎么部署」 | 评估/压缩/推理三块每个岗位都要会基本盘 |

## 🔗 跳转链接

- 总览：[[00-知识地图]]
- 结构地基：[[llm-algo/transformer/模型架构]]
- 训练与并行：[[llm-train/README]]、[[llm-train/pytorch/distribution/README]]
- 框架：[[ai-framework/deepspeed/README]]、[[ai-framework/megatron-lm/README]]
- 对齐：[[llm-alignment/RLHF]]
- 压缩：[[llm-compression/quantization/量化基础]]、[[llm-compression/README]]
- 推理：[[llm-inference/README]]、[[llm-inference/vllm/README]]
- 优化算子：[[llm-optimizer/kv-cache]]、[[llm-optimizer/FlashAttention]]
- 评估：[[llm-eval/README]]、[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 应用：[[llm-application/rag/README]]
- Infra 网络：[[ai-infra/网络/集合通信原语]]、[[ai-infra/网络/NCCL]]
