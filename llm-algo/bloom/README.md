# BLOOM 模型详解

> BLOOM 是 BigScience 国际开放协作项目产出的 1760 亿参数、46 种自然语言 + 13 种编程语言的多语言自回归大模型，是「完全开放权重 + 开放训练全过程」的里程碑式开源 LLM。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] [[llm-algo/gpt3/README]] [[llm-algo/llama/README]] [[ai-framework/megatron-lm/README]] [[ai-framework/deepspeed/README]]

## 阅读地图

| 小节 | 你将搞懂 |
| --- | --- |
| 0. 一句话锚点 | BLOOM 到底是什么 |
| 1. 地基 | 它要解决「开源 LLM」的什么问题 |
| 2. 整体架构 | Decoder-only Transformer 的骨架 |
| 3. ALiBi 位置编码 | 为什么不用绝对/旋转位置编码 |
| 4. Embedding LayerNorm | 一个稳定训练的小改动 |
| 5. 词表与多语言分词 | 25 万词表为何这么大 |
| 6. 规模家族 | 560M → 176B 的参数配置 |
| 7. 训练数据 ROOTS | 1.6TB 多语言语料怎么来的 |
| 8. 训练基础设施 | Megatron-DeepSpeed 3D 并行 |
| 9. BLOOMZ / mT0 | 指令微调后的变体 |
| 流程示例 | 一次前向计算的数据流 |
| 常见坑 | 用 BLOOM 时容易踩的 |

## 0. 一句话锚点

BLOOM（BigScience Large Open-science Open-access Multilingual Language Model）= **GPT 式 Decoder-only Transformer** + **ALiBi 位置编码** + **Embedding 后接 LayerNorm** + **25 万级多语言词表**，由 Hugging Face 牵头、上千名研究者协作，在法国 Jean Zay 超算上训练完成，权重、代码、数据、训练日志全部公开。

它在模型「结构」上没有发明颠覆性新东西——核心价值在于 **把一个 100B+ 级模型的「整个炼制过程」第一次完全透明地摆出来**。

## 1. 地基：BLOOM 解决什么问题

在 BLOOM（2022 年）之前，百亿/千亿级语言模型有两类尴尬：

1. **闭源**：GPT-3、PaLM 等只能 API 访问，权重、数据、训练细节是黑箱，研究者无法复现、审计、二次开发。
2. **以英语为中心**：绝大多数大模型语料 90%+ 是英语，对中、法、阿拉伯、印地、非洲诸语等支持很差。

BLOOM 的目标就是两条：

```
        闭源 + 英语中心
              │
     ┌────────┴────────┐
     ▼                 ▼
  完全开放          多语言均衡
 (权重/代码/      (46 自然语言
  数据/日志)       +13 编程语言)
     │                 │
     └────────┬────────┘
              ▼
           BLOOM-176B
```

- **完全开放**：采用 RAIL（Responsible AI License）许可，权重可下载、可商用（带使用限制），训练 codebase（Megatron-DeepSpeed）、语料构建流程（ROOTS）、甚至每天的 loss 曲线和 tensorboard 都公开。
- **多语言**：刻意提高非英语语种占比，让模型对低资源语言也有可用能力。

> 一句话：BLOOM 在「模型科学」上偏保守（沿用成熟结构），在「开放科学」上极其激进。

## 2. 整体架构：Decoder-only Transformer

BLOOM 是标准的 **自回归 Decoder-only** 结构，和 GPT 系一脉相承：输入若干 token，预测下一个 token，靠因果掩码（causal mask）保证第 $i$ 个位置只能看到 $\le i$ 的 token。

```
   输入 token ids
        │
        ▼
 [Word Embedding]  (vocab≈250k × hidden)
        │
        ▼
 [Embedding LayerNorm]   ★ BLOOM 特有：emb 后立刻归一化
        │
        ▼
 ┌───────────────────────────────┐
 │  ×N 层 Transformer Block       │
 │  ┌─────────────────────────┐  │
 │  │ LayerNorm               │  │  (pre-LN)
 │  │ Multi-Head Self-Attn    │  │  ← ALiBi 加在注意力分数上
 │  │   + 残差                 │  │
 │  ├─────────────────────────┤  │
 │  │ LayerNorm               │  │
 │  │ MLP (GELU, 4×hidden)    │  │
 │  │   + 残差                 │  │
 │  └─────────────────────────┘  │
 └───────────────────────────────┘
        │
        ▼
 [Final LayerNorm]
        │
        ▼
 [LM Head]  (常与 Word Embedding 权重共享)
        │
        ▼
   logits → softmax → 下一个 token
```

关键设计点（与 vanilla GPT 的区别）：

| 部件 | BLOOM 的选择 | 为什么 |
| --- | --- | --- |
| 位置编码 | **ALiBi**（不加位置向量，在注意力分数上加斜率偏置） | 外推到比训练更长的序列更稳 |
| 归一化位置 | **Pre-LN**（子层前归一化） | 深层大模型训练更稳定 |
| 额外归一化 | **Embedding 后多一层 LayerNorm** | 进一步稳住超大模型的训练初期 |
| 激活函数 | GELU | 沿用 Transformer 主流 |
| 注意力 | 标准多头自注意力 | 保守、可靠 |

> 注：具体层数、隐藏维度、注意力头数等以官方 config（Hugging Face `bigscience/bloom`）和论文为准，本文给的是类别与机制。

## 3. ALiBi 位置编码（核心机制）

普通 Transformer 用「绝对位置嵌入」或「旋转位置编码（RoPE）」告诉模型 token 的先后顺序。**ALiBi（Attention with Linear Biases）** 换了一个更朴素的思路：

> 不给 token 加任何位置向量，而是在计算注意力分数时，**按 query 和 key 的距离，线性地扣分**——离得越远，分数被减得越多。

注意力分数从

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$$

变成

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} - m \cdot (i - j)$$

其中 $i \ge j$（因果），$(i-j)$ 是相对距离，$m$ 是一个**每个注意力头固定不同的斜率**（不可学习的超参，按头编号几何递减）。

可视化一个头的偏置矩阵（值越小越「不被关注」）：

```
       key位置→  0     1     2     3
 query  ┌───────────────────────────
   0    │  0     -∞    -∞    -∞     ← 因果掩码，看不到未来
   1    │ -m·1   0     -∞    -∞
   2    │ -m·2  -m·1   0     -∞
   3    │ -m·3  -m·2  -m·1   0
```

为什么这样设计：

1. **天然的「近因偏好」**：越近的 token 越重要，符合语言局部性。
2. **长度外推（extrapolation）**：因为偏置只依赖「相对距离」，训练时用 2048 长度，推理时喂更长序列，模型也能合理工作——这是 ALiBi 最大的卖点。
3. **省参数、省显存**：不需要存位置嵌入表。

不同头用不同斜率 $m$：有的头斜率大（强烈偏向近处），有的头斜率小（能看远处），让模型在不同尺度上都能建模依赖。

## 4. Embedding LayerNorm

BLOOM 在 **词嵌入之后、进入第一个 Transformer Block 之前**，额外插了一层 LayerNorm。

```
 ids → Embedding → [LayerNorm] → Block1 → ...
                      ↑
              别处没有的额外归一化
```

动机很实在：训练 176B 这种巨型模型，初期梯度/激活值容易爆炸或消失，BigScience 团队在消融实验中发现，在 embedding 后加一层归一化能**显著降低训练早期的不稳定（loss spike）**。这是个「小改动、大稳定」的工程经验，不是理论上的必需品。

## 5. 词表与多语言分词

BLOOM 用的是 **BPE（Byte-level BPE）分词器**，词表规模约 **25 万**（250680），远大于 GPT-2 的约 5 万。

为什么要这么大的词表：

```
  小词表(50k)         大词表(250k)
  ──────────         ────────────
  对英语友好          对 46 种语言都尽量友好
  非英语词被切成      每个语种都有足够 token
  很多碎片            预算 → fertility 更低
  序列变长、变慢       序列更短、信息密度高
```

- **fertility（分词膨胀率）**：一个词平均被切成几个 token。词表太小，对中文、阿拉伯文这类语言 fertility 很高（一个字/词被切成一堆字节碎片），既浪费 context 长度又拖慢推理。
- BLOOM 刻意把词表做大，让每种语言都「分到」足够多的子词单元，**摊平多语言的不公平**。
- 代价：大词表 → embedding 矩阵和 LM head 变大（$250680 \times \text{hidden}$），是显存/参数的大头之一。

## 6. 规模家族

BLOOM 不是单一模型，而是一组同结构、不同规模的家族（便于研究 scaling、便于小算力用户）：

```
  bloom-560m   ── 玩具/教学
  bloom-1b1
  bloom-1b7
  bloom-3b
  bloom-7b1    ── 单卡/少卡可玩
  bloom (176b) ── 旗舰
```

随规模增大，主要变化的是 **层数 N、隐藏维度 hidden、注意力头数 heads**；词表、ALiBi、Embedding-LN 等结构选择保持一致。

> 具体每档的精确层数/维度以官方模型卡为准；这里强调的是「同一套结构按 scaling law 放大」。

## 7. 训练数据 ROOTS

BLOOM 的语料叫 **ROOTS**（Responsible Open-science Open-collaboration Text Sources），约 **1.6 TB** 文本，覆盖 46 种自然语言 + 13 种编程语言。

```
  ROOTS 语料构成（示意，非精确占比）
  ┌──────────────────────────────────┐
  │  英语    法语   中文   西班牙语 ...  │  ← 自然语言（刻意均衡）
  │  阿拉伯  印地   越南   多种非洲语   │
  ├──────────────────────────────────┤
  │  Python  Java   C++   JS  ...      │  ← 编程语言
  └──────────────────────────────────┘
        │
   全程人工 + 自动 治理：
   去重 / 去个人信息(PII) / 过滤有害内容 / 质量筛选
```

特点：

- **治理透明**：数据来源、清洗规则、语言占比都有文档，强调「负责任」。
- **多语言均衡**：通过上采样/下采样调整各语言比例，避免英语一家独大。
- 数据准备工作本身（去重、PII 清理、毒性过滤）也作为开源成果发布。

## 8. 训练基础设施：Megatron-DeepSpeed 3D 并行

176B 模型放不进任何单卡，BLOOM 在数百张 A100（80GB）上、用 **Megatron-DeepSpeed**（Megatron-LM 的张量并行 + DeepSpeed 的 ZeRO/流水线）做 **3D 并行**：

```
        全局 GPU 集群
   ┌─────────────────────────────────┐
   │  数据并行 (Data Parallel)        │  不同 batch 切片
   │   ├ 流水线并行 (Pipeline)        │  按层切到不同 GPU 组
   │   │   ├ 张量并行 (Tensor)        │  单层矩阵切到组内 GPU
   │   │   │   GPU GPU GPU GPU        │
   │   │   └ ...                       │
   │   └ ...                           │
   └─────────────────────────────────┘
   显存优化：ZeRO 分片优化器状态/梯度
   精度：bf16 混合精度
```

- **张量并行（TP）**：把一个大矩阵乘法横向切到组内多卡，组内靠 NCCL all-reduce 拼结果（强通信，放同机 NVLink 内）。参见 [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]]。
- **流水线并行（PP）**：把不同层放到不同 GPU 组，像流水线一样传激活。
- **数据并行（DP）+ ZeRO**：把优化器状态/梯度分片到多卡，省显存。
- 训练历时数月，**全过程 loss 曲线、硬件故障、重启记录全部公开**——这是 BLOOM 最珍贵的科研资产。

机制细节见 [[ai-framework/megatron-lm/README]] 与 [[ai-framework/deepspeed/README]]。

## 9. BLOOMZ / mT0：指令微调变体

原始 BLOOM 是「续写式」基座模型，不会乖乖听指令。BigScience 在 **xP3（多语言、多任务的 prompt 数据集）** 上对 BLOOM 做指令微调，得到 **BLOOMZ**（mT0 是基于 mT5 的同类产物）。

```
  BLOOM (base，会续写)
        │  在 xP3 上做多任务/多语言指令微调
        ▼
  BLOOMZ (会跟随指令，且零样本跨语言泛化)
```

亮点：BLOOMZ 展现出 **跨语言指令泛化**——即使某任务只在英语 prompt 上训练过，也能在其他语言上零样本执行。

## 一次前向计算的数据流（示例）

以输入「`Bonjour le`」、要预测下一个 token 为例：

```
1. 分词:  "Bonjour le" → ids = [b1, b2, b3]   (BPE 子词)
2. 嵌入:  ids → E ∈ R^{3×hidden}
3. Emb-LN: E → LayerNorm(E)
4. 逐层:  for layer in 1..N:
            a) x' = LN(x)
            b) attn = Softmax( QKᵀ/√d  +  ALiBi偏置  +  因果掩码 ) · V
            c) x = x + attn                 ← 残差
            d) x'' = LN(x)
            e) x = x + MLP_GELU(x'')         ← 残差
5. 末归一: x → FinalLN(x)
6. 输出头: logits = x · Eᵀ        (权重共享)
7. 取最后位置 logits → softmax → 采样 → "monde" / "soleil" ...
```

注意第 4b 步：**ALiBi 偏置是在 softmax 之前、直接加到注意力分数上的**，没有任何位置嵌入参与第 2 步。

## 配置项与权衡（讲含义，不背默认值）

| 配置 | 作用 | 怎么权衡 |
| --- | --- | --- |
| `n_layer` / `hidden_size` / `n_head` | 决定模型容量 | 越大越强但越贵；按可用算力选档（560M→176B） |
| 词表大小（≈25万） | 多语言覆盖与序列长度 | 大词表对多语言友好，但 embedding/LM head 显存大 |
| ALiBi 斜率 | 各头的距离衰减强度 | 不可学习超参，按头几何递减；一般不动 |
| 精度（bf16/fp16） | 训练/推理数值范围 | bf16 动态范围大、训练稳；推理可再 int8/int4 量化 |
| max sequence length | 训练长度 | ALiBi 支持外推，可在更长序列上推理 |
| 量化（int8/int4） | 推理显存 | 176B 推理建议量化，见 [[llm-compression/quantization/量化基础]] |

> CLI/字段确切名字、默认值以 Hugging Face `transformers` 中 `BloomConfig` 与官方模型卡为准。

## 常见问题 / 坑

| 现象 / 误区 | 原因 | 应对 |
| --- | --- | --- |
| 直接问 BLOOM 不听指令 | 用的是 base 模型，只会续写 | 换 **BLOOMZ**，或自己做指令微调 |
| 中文/多语言效果不及预期 | base 模型未对齐，且各语种数据有限 | 用 BLOOMZ；必要时领域内继续训练 |
| 176B 显存装不下 | 单卡放不下千亿参数 | 多卡张量/流水并行 + int8/int4 量化 |
| 把 ALiBi 当成「可学位置嵌入」 | ALiBi 是固定偏置，无可训练位置参数 | 理解它是注意力分数上的距离惩罚 |
| 以为词表越大越好 | 大词表显著增加 embedding/head 参数与显存 | 权衡多语言覆盖与开销 |
| 期待它超过同期闭源 SOTA | BLOOM 优先「开放与多语言」而非刷榜 | 看重的是可复现性、透明度、低资源语言 |
| 把 BLOOM 和 LLaMA 混为一谈 | LLaMA 用 RoPE + RMSNorm + SwiGLU，BLOOM 用 ALiBi + LayerNorm + GELU | 对比见下方链接 |

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 同类基座模型：[[llm-algo/gpt3/README]] · [[llm-algo/llama/README]] · [[llm-algo/transformer/模型架构]]
- 训练框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 并行与通信：[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 推理优化：[[llm-compression/quantization/量化基础]] · [[llm-optimizer/FlashAttention]]
- 参考：[BLOOM模型结构详解](https://juejin.cn/post/7223305855923044409)
