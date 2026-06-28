# LLaMA 模型说明

> Meta 开源的 decoder-only 大语言模型家族，用"小参数 + 大数据"重塑了开源 LLM 生态，是 Alpaca/Vicuna/Llama-2-Chat 等几乎所有开源对话模型的"地基"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/llama/模型架构]] [[llm-algo/llama]]

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | LLaMA = 开源 GPT 风格 decoder |
| 1 | 地基/前置 | decoder-only、自回归、token、参数量 |
| 2 | 各代规模/许可/特点 | LLaMA1/2/3/3.1，许可证演变 |
| 3 | 架构为何这样设计 | RMSNorm/RoPE/SwiGLU/GQA |
| 4 | 权重怎么拿到手 | 申请、HF、转换、量化格式 |
| 5 | tokenizer 原理 | SentencePiece BPE → tiktoken |
| 6 | 与生态的关系 | Alpaca/Vicuna/指令微调谱系 |
| 7 | 怎么部署 | 显存估算、推理框架、量化 |
| 8 | 数值例子/对照/实践 | 显存手算、规模对照 |
| 9 | 常见问题 | FAQ 表 |

---

## 0. 一句话锚点

**LLaMA（Large Language Model Meta AI）是一组 decoder-only 的自回归 Transformer 语言模型。** 它的工作只有一件事：给定前面的 token 序列，预测"下一个 token 的概率分布"，然后采样、追加、再预测，循环往复地把文本"接龙"下去。

它之所以重要，不是因为架构有多新（架构和 GPT 系列大同小异），而是因为：

1. **开源可下载权重** —— 学术界和创业公司第一次能拿到"接近 GPT-3.5 质量"的模型权重做研究和二次开发。
2. **算力效率** —— 用更小的模型 + 更多的训练数据，证明了"7B/13B 也能很能打"，让消费级显卡能跑大模型。

```
        ┌─────────────────────────────────────────────┐
        │   LLaMA 在生态里的位置                        │
        │                                             │
        │   Meta 预训练 ──► LLaMA 基座(base model)     │
        │                      │                       │
        │       ┌──────────────┼──────────────┐        │
        │       ▼              ▼              ▼        │
        │   Alpaca         Vicuna         你的微调      │
        │  (指令微调)      (对话微调)      (LoRA/SFT)    │
        └─────────────────────────────────────────────┘
```

---

## 1. 地基/前置（不假设你记得这些）

在拆解 LLaMA 前，先把最原子的概念钉死。

### 1.1 什么叫 "decoder-only / 自回归"

Transformer 原始论文有 encoder（读输入）和 decoder（生成输出）两半。GPT/LLaMA 砍掉 encoder，**只保留 decoder**，所以叫 decoder-only。

"自回归（autoregressive）"指生成时**一个 token 一个 token 地吐**，每一步都把已生成的内容重新喂回去。数学上它建模的是联合概率的链式分解：

$$P(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \dots, x_{t-1})$$

模型每一步只学一件事：$P(x_t \mid x_{<t})$，即"看了前面所有 token，下一个最可能是谁"。

```
输入: "今天 天气 很"
            │
            ▼  (前向一次)
   ┌──────────────────────┐
   │ 下一个token概率分布    │
   │  好  : 0.62          │
   │  冷  : 0.18          │  ──► 采样得到 "好"
   │  热  : 0.09          │
   │  ... : ...           │
   └──────────────────────┘
            │
            ▼  把"好"接上去, 再来一次
输入: "今天 天气 很 好"  ──► 预测下一个 ...
```

### 1.2 什么叫 "token" 与 "参数量"

- **token**：模型的最小处理单位，不是字也不是词，而是 tokenizer 切出来的"子词片段"（见第 5 节）。一段中文/英文文本先被切成 token id 序列，模型只认数字。
- **参数量（7B/13B/70B）**：B = Billion（十亿）。7B 表示约 70 亿个浮点数权重。参数量大致决定了模型的"容量上限"和"显存/算力开销"。一个粗略心智模型：**FP16 下，每 1B 参数 ≈ 2 GB 权重显存**（因为每个参数 2 字节）。

### 1.3 "预训练 base" vs "指令/对话模型"

- **base 模型**：只做了"预测下一个 token"的预训练，擅长续写，但不会"听话回答问题"。
- **chat/instruct 模型**：在 base 之上做了 SFT（监督指令微调）甚至 RLHF，学会了"理解指令并回答"。LLaMA-2/3 官方既放 base 也放 chat 版。

---

## 2. 各代规模 / 许可 / 特点

LLaMA 不是一个模型，而是一个**不断迭代的家族**。下面按时间线讲清每一代"变了什么"。

```
时间线 (约)
2023.02 ── LLaMA 1     7B/13B/33B/65B   研究许可(不可商用)
2023.07 ── LLaMA 2     7B/13B/70B       社区许可(可商用, 有 MAU 门槛)
2024.04 ── LLaMA 3     8B/70B           社区许可, 改用 tiktoken 分词
2024.07 ── LLaMA 3.1   8B/70B/405B      128K 上下文, 多语言增强
2024.09+── LLaMA 3.2/3.3 (多模态/小模型/蒸馏) ...
```

### 2.1 LLaMA 1（2023.02）

- **规模**：7B / 13B / 33B(论文写 32.5B) / 65B。
- **许可**：**仅限非商业研究用途**，权重需逐个申请，且早期"意外泄露"到网上引发广泛传播。
- **特点**：核心卖点是"**用更多数据训练更小模型**"。论文遵循的思路源自 Chinchilla 的"算力最优"观察，但 Meta 更进一步：在推理成本敏感的现实里，宁可多训练让小模型更强。LLaMA-13B 在多数基准上超过了 175B 的 GPT-3。

### 2.2 LLaMA 2（2023.07）

- **规模**：7B / 13B / 70B（base 与 chat 两套）。
- **许可**：**Llama 2 Community License**，**首次允许商用**，但附带条件——例如月活超 7 亿（约 7 亿 MAU）的超大公司需另行向 Meta 申请授权；不得用 Llama 输出去改进其它竞争 LLM。
- **特点**：上下文从 2K 提升到 4K；70B 用了 **GQA（分组查询注意力）** 降低推理显存；chat 版做了 RLHF 对齐，发布了较完整的安全/红队报告。

### 2.3 LLaMA 3 / 3.1（2024）

- **规模**：3 代有 8B / 70B；3.1 增加了 **405B** 这一旗舰级稠密模型。
- **许可**：**Llama 3 / 3.1 Community License**（延续可商用 + 大厂门槛思路，3.1 起明确允许用其输出做合成数据训练其它模型）。
- **特点**：
  - **词表从 32K 暴涨到 128K**，分词器换成基于 **tiktoken** 的 BPE（见第 5 节），对多语言和代码更高效。
  - LLaMA 3.1 上下文窗口拉到 **128K tokens**。
  - 8B 也引入 GQA，所有规模统一注意力结构。
  - 训练数据规模大幅增加（官方称 15T+ tokens 量级，以官方为准）。

### 2.4 规模与许可对照表

| 代次 | 典型规模 | 上下文 | 词表 | 许可关键词 | 一句话 |
|------|----------|--------|------|------------|--------|
| LLaMA 1 | 7/13/33/65B | 2K | 32K | 仅研究 | 小模型大数据的起点 |
| LLaMA 2 | 7/13/70B | 4K | 32K | 可商用(7亿MAU门槛) | 首个商用友好开源基座 |
| LLaMA 3 | 8/70B | 8K | 128K | 可商用 | 换 tiktoken, 词表升级 |
| LLaMA 3.1 | 8/70/405B | 128K | 128K | 可商用(允许合成数据) | 旗舰 405B + 长上下文 |

> 注：上表的上下文/词表为各代官方主线说法，细节请以 Meta 官方 model card 为准。

---

## 3. 架构为何这样设计（每节配图）

LLaMA 是 GPT 风格的标准 decoder 堆叠，但在四个关键点上做了"务实改良"。这里只讲"为什么这么改"，细节公式见 [[llm-algo/llama/模型架构]]。

```
       LLaMA 单个 Transformer Block (Pre-Norm 结构)
   ┌──────────────────────────────────────────────┐
   │  x ──► RMSNorm ──► Self-Attn(RoPE+GQA) ──► (+)─┼─► 
   │  │                                         ▲  │
   │  └─────────────── 残差 ────────────────────┘  │
   │                                               │
   │  ──► RMSNorm ──► FFN(SwiGLU) ──► (+)──────────►│
   │  │                              ▲             │
   │  └────────── 残差 ──────────────┘             │
   └──────────────────────────────────────────────┘
        堆叠 N 层 (7B≈32层, 70B≈80层)
```

### 3.1 RMSNorm 替代 LayerNorm —— "为什么"

LayerNorm 要算均值和方差并做中心化；RMSNorm 砍掉减均值这步，只用均方根（RMS）缩放：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot g$$

**为什么**：计算更省（少一次求均值），实践证明效果不掉，训练更稳。$g$ 是可学习缩放向量。

### 3.2 RoPE 旋转位置编码 —— "为什么"

原始 Transformer 用"绝对位置编码"（给每个位置加一个固定向量）。RoPE 改成"**把位置信息编码成旋转**"：对 query/key 向量按位置施加旋转矩阵，使得两个 token 的注意力分数只依赖它们的**相对距离**。

**为什么**：相对位置更符合语言直觉（"前一个词"比"第 5 个词"更本质），且对**外推到更长上下文**更友好——这正是 LLaMA 3.1 能扩到 128K 的基础之一。

### 3.3 SwiGLU 前馈网络 —— "为什么"

标准 FFN 是 `Linear → ReLU → Linear`。SwiGLU 用"门控"结构：

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_1)\odot (xW_3)\big)W_2$$

一条路过激活函数当"门"，另一条路是"内容"，逐元素相乘。**为什么**：门控让网络能更灵活地控制信息流，同等算力下质量更好（代价是多了一个权重矩阵，所以 LLaMA 把中间维度调小以平衡参数量）。

### 3.4 GQA 分组查询注意力 —— "为什么"

```
  MHA(多头)        GQA(分组)         MQA(单KV)
 Q Q Q Q          Q Q Q Q          Q Q Q Q
 │ │ │ │          └┬┘ └┬┘          └─┬─┘
 K K K K           K   K              K
 V V V V           V   V              V
每头独立KV      每组共享KV        全部共享1组KV
显存最大        折中(LLaMA用)      显存最小但质量略降
```

推理时每个 token 的 Key/Value 要缓存（KV cache），显存开销和"KV 头数"成正比。**GQA** 让多个 Query 头共享一组 KV 头，**为什么**：把 KV cache 显存砍掉几倍，长上下文推理才扛得住，而质量几乎不掉。LLaMA-2-70B 起用，LLaMA-3 全系采用。

---

## 4. 权重怎么获取（实操路径）

```
   ┌─────────────┐   申请/同意许可   ┌──────────────┐
   │ Meta 官方   │ ───────────────► │ 原始权重(.pth)│
   │ llama.com   │                  │ + 分词器model │
   └─────────────┘                  └───────┬──────┘
                                            │ 转换脚本
   ┌─────────────┐   同样需同意许可   ┌──────▼───────┐
   │ HuggingFace │ ───────────────► │ HF 格式(.safe-│
   │ meta-llama/ │                  │  tensors)     │
   └─────────────┘                  └───────┬──────┘
                                            │ 量化
                                    ┌───────▼───────┐
                                    │ GGUF / GPTQ / │
                                    │ AWQ (社区量化)│
                                    └───────────────┘
```

### 4.1 三条主流路径

1. **Meta 官方申请**：到 Llama 官网填表同意许可，拿到下载链接/脚本，得到原始 `.pth` 权重 + `tokenizer.model`。这是"权威源头"。
2. **HuggingFace `meta-llama` 组织**：在 HF 上对相应仓库点"同意许可"，用 `transformers` 直接 `from_pretrained`。HF 提供已转好的 `safetensors`，最省事，也是工程实践首选。
3. **社区再分发/量化版**：如 GGUF（给 llama.cpp）、GPTQ/AWQ（4-bit 量化），适合消费级显卡或 CPU 推理。注意：**这些仍受 Llama 许可约束**。

### 4.2 三种权重格式分别给谁用

| 格式 | 后缀 | 给谁用 | 特点 |
|------|------|--------|------|
| 原始权重 | `.pth` | 官方/科研复现 | Meta 原生 checkpoint |
| HF 格式 | `.safetensors` | transformers/训练微调 | 生态最广, 安全张量 |
| GGUF | `.gguf` | llama.cpp/Ollama | CPU/Mac/低显存友好 |
| GPTQ/AWQ | 量化包 | vLLM/TGI 量化推理 | 4-bit, 省显存 |

**实践要点**：用 HF 训练/微调，用 GGUF 在本地 Mac/CPU 玩，用 AWQ+vLLM 上生产。许可证全程跟着权重走，商用前务必读一遍 model card。

---

## 5. tokenizer：SentencePiece BPE → tiktoken

tokenizer 是"文本 ↔ token id"的翻译官。模型只认整数，tokenizer 决定了"一句话被切成几块、每块是什么"。

### 5.1 BPE 到底在干什么（拆到最底层）

**BPE（Byte-Pair Encoding，字节对编码）** 的核心思想：从最细粒度开始，**反复把"最常一起出现的相邻片段"合并成一个新片段**，直到词表达到设定大小。

```
训练语料里 "low lower lowest" 反复出现 ...
初始: l o w   l o w e r   l o w e s t   (全是单字符)
统计最频繁相邻对: "l o" 出现最多 ──► 合并成 "lo"
再统计:           "lo w"  最多      ──► 合并成 "low"
... 直到词表满 32000(L1/L2) 或 128000(L3) 个 token
```

**为什么这样切**：既不像"按字母切"那样序列太长，也不像"按单词切"那样遇到生词就抓瞎。BPE 让常见词成为一个 token、罕见词拆成几个子词，兼顾**覆盖率**和**序列长度**。

### 5.2 LLaMA 1/2：SentencePiece 实现的 BPE

LLaMA 1/2 用 **SentencePiece** 库（Google 出品）训练 BPE，词表 **32K**。SentencePiece 的两个关键设计：

- **直接在原始字节流上训练**，不依赖空格分词，所以天然支持中文、日文这类无空格语言。
- 用一个特殊符号 **`▁`（U+2581）表示空格**，于是"分词→还原"是无损可逆的。

```
"Hello world"
  └─ SentencePiece ─►  ["▁Hello", "▁world"]
                        (▁ 记录了原本的空格位置)
解码时把 ▁ 换回空格即可无损还原
```

产物是一个二进制文件 `tokenizer.model`。

### 5.3 LLaMA 3：换成 tiktoken 风格 BPE，词表 128K

LLaMA 3 起改用 **tiktoken**（OpenAI 系）风格的 BPE，词表扩到 **128K**。

**为什么换**：

- 词表更大 → 同样一段文本切出的 token 更少 → **同样上下文窗口能装更多内容、推理更快**（尤其多语言和代码，原来 32K 词表对中文等切得很碎）。
- tiktoken 在长文本上编码速度快。

```
对比同一句中文 "人工智能正在改变世界"
LLaMA2(32K词表) :  切成 ~10+ 个零碎 token
LLaMA3(128K词表):  切成更少、更"整词"的 token
            ↓
   同样的 8K/128K 窗口能塞下更多实际内容
```

**实践坑**：换分词器意味着 LLaMA 2 和 LLaMA 3 的 token id **不通用**，词嵌入表大小也变了（32000→128256 量级），微调脚本/词表对齐时要特别注意。

---

## 6. 与生态的关系：Alpaca / Vicuna 谱系

LLaMA base 本身"不会聊天"。一整套开源对话模型，都是**站在 LLaMA 肩膀上**做指令/对话微调得来的。

```
        Meta LLaMA (base, 只会续写)
                  │
        ┌─────────┼──────────────┐
        ▼         ▼              ▼
   Alpaca     Vicuna        其它(Koala/
 (Stanford)  (LMSYS)         WizardLM...)
   │             │
   │ 52K 指令     │ 7万条 ShareGPT
   │ 数据(由       │ 真实对话(用户
   │ GPT 自动      │ 分享的 ChatGPT
   │ 生成)         │ 对话)
   │             │
   SFT          SFT(多轮对话)
```

### 6.1 Alpaca（斯坦福）

- **做法**：拿 LLaMA-7B base，用 **52K 条指令-回答数据**做 SFT。这 52K 数据是用"self-instruct"思路、让一个更强的模型（早期 GPT 系）**自动生成**的。
- **意义**：证明了"几百美元成本就能把 base 调成像样的指令模型"，点燃了开源微调浪潮。

### 6.2 Vicuna（LMSYS）

- **做法**：拿 LLaMA base，用约 **7 万条 ShareGPT 上用户分享的真实 ChatGPT 对话**做多轮对话 SFT。
- **意义**：因为训练数据是"真实优质对话"，Vicuna 的对话质量明显强于 Alpaca，一度被评估为"接近 ChatGPT 的开源天花板"，还催生了用 GPT-4 当裁判的评测范式（MT-Bench 的前身思路）。

### 6.3 谱系要点

| 模型 | 基座 | 微调数据 | 一句话 |
|------|------|----------|--------|
| Alpaca | LLaMA-7B | 52K 合成指令 | 低成本指令微调先驱 |
| Vicuna | LLaMA 7/13B | 7万真实对话 | 对话质量标杆 |
| Llama-2-Chat | LLaMA-2 | Meta 官方 SFT+RLHF | 官方对齐版 |

**核心认知**：Alpaca/Vicuna 的"聪明"来自 LLaMA 基座，它们贡献的是"**怎么把基座调得会听话**"的方法与数据，而非新能力。许可上，因其基于早期 LLaMA-1 且部分数据涉及 OpenAI 输出，**严格只可用于研究**。

---

## 7. 怎么部署（显存 / 框架 / 量化）

部署的第一问永远是：**这模型显存装得下吗？** 然后才是选框架。

### 7.1 显存从哪来（拆解）

推理显存 ≈ **权重** + **KV cache** + 少量激活/框架开销。

$$\text{权重显存} \approx N_{\text{params}} \times \text{每参数字节数}$$

- FP16：每参数 2 字节；INT8：1 字节；INT4：0.5 字节。
- KV cache 随"序列长度 × 层数 × KV 头维度"线性增长，长上下文时它会变成大头（GQA 就是为压它）。

```
   7B 模型权重显存随精度变化
   FP16  ████████████████  ≈14 GB
   INT8  ████████          ≈ 7 GB
   INT4  ████              ≈ 3.5 GB
        (再加 KV cache 与开销)
```

### 7.2 推理框架怎么选

| 框架 | 适合场景 | 核心机制 | 权衡 |
|------|----------|----------|------|
| transformers | 调试/微调 | 最通用、生态全 | 推理吞吐一般 |
| vLLM | 生产高并发 | PagedAttention 管理 KV cache | 吞吐高，部署稍重 |
| llama.cpp | 本地/Mac/CPU | GGUF + 量化, C++ 实现 | 极轻量, 单机友好 |
| Ollama | 个人一键跑 | 封装 llama.cpp | 体验最简单 |
| TGI | 服务化 | HF 官方推理服务 | 与 HF 生态贴合 |

**关键参数的含义（讲含义不背默认值）**：

- `max_seq_len / context length`：能处理的最长 token 数，越长 KV cache 越吃显存。
- `tensor parallel size`：把一个模型切到几张卡上（70B/405B 必须用），切得越多单卡显存压力越小但卡间通信越多。
- `quantization`：选 INT8/INT4 省显存，代价是可能掉一点精度——以官方/实测为准。
- `gpu memory utilization`（vLLM）：允许框架占用的显存比例，留太满容易 OOM。

### 7.3 部署决策图

```
        你要部署 LLaMA?
              │
   ┌──────────┴───────────┐
   │ 单机本地玩/Mac?       │──是─► llama.cpp / Ollama + GGUF量化
   └──────────┬───────────┘
              │否
   ┌──────────┴───────────┐
   │ 生产高并发服务?       │──是─► vLLM (+AWQ量化, +张量并行)
   └──────────┬───────────┘
              │否
              └─► transformers (调试/微调/原型)
```

---

## 8. 数值例子 / 对照 / 实践

### 8.1 手算：LLaMA-2-70B FP16 推理要多少卡？

- 权重显存：$70 \times 10^9 \times 2\ \text{B} = 140\ \text{GB}$。
- 一张 A100-80G 装不下 140GB，需 **张量并行切到 ≥2 张卡**（再留 KV cache 和开销，实务上常用 2~4 张 80G 卡）。
- 若改 **INT4 量化**：$70 \times 0.5 = 35\ \text{GB}$，**单张 A100-80G 就能跑**（质量略有损失，以实测为准）。

> 以上为数量级估算，未计入 KV cache、框架碎片与并行通信开销，实际以压测为准。

### 8.2 对照：为什么 405B 是"另一个世界"

| 模型 | FP16 权重 | 粗略所需 80G 卡数 |
|------|-----------|-------------------|
| 7B | ≈14 GB | 1 |
| 13B | ≈26 GB | 1 |
| 70B | ≈140 GB | ≥2 |
| 405B | ≈810 GB | ≥10（多机多卡）|

405B 的 810GB 权重必须**多机多卡 + 张量/流水线并行**，部署复杂度和成本是 7B 的两个数量级以上——这解释了为什么社区主力仍是 8B/70B。

### 8.3 token 效率对照（直观感受 128K 词表收益）

同一段中文，LLaMA-2（32K 词表）切出的 token 数可能是 LLaMA-3（128K 词表）的 **1.3~2 倍**（具体随文本而变）。token 更少意味着：同样上下文窗口装更多内容、单位文本推理更快、调用 API 更省。

---

## 常见问题

| 问题 | 回答 |
|------|------|
| LLaMA 和 GPT 架构差别大吗？ | 不大，都是 decoder-only。差异在 RMSNorm/RoPE/SwiGLU/GQA 这些务实改良和"开源权重"。 |
| LLaMA 能直接拿来聊天吗？ | base 版不能（只会续写），要用 chat/instruct 版或自己做 SFT（如 Alpaca/Vicuna 那样）。 |
| LLaMA 2/3 权重不通用？ | 是，分词器和词表都变了（32K→128K），token id 不互通，微调要对齐词表。 |
| 商用能用吗？ | LLaMA 2 起社区许可允许商用，但大厂(约7亿 MAU)有门槛；LLaMA 1 及 Alpaca/Vicuna 仅限研究。**以官方许可为准**。 |
| 消费级显卡能跑吗？ | 7B/8B 用 INT4 量化（GGUF/AWQ）+ llama.cpp，单张消费卡甚至高端 Mac 即可。 |
| GQA 到底省了什么？ | 省 KV cache 显存（让多 Query 头共享 KV 头），长上下文推理才扛得住。 |
| 为什么 LLaMA-13B 能打 GPT-3 175B？ | 用更多数据把小模型"喂饱"，在很多基准上质量反超，且推理成本低得多。 |

---

## 🔗 跳转链接

- [[00-知识地图]] —— 全局导航
- [[llm-algo/llama/模型架构]] —— RMSNorm/RoPE/SwiGLU/GQA 的逐公式拆解
- [[llm-algo/llama]] —— LLaMA 目录索引

### LLaMA3 深入参考资料

- Build Your Own Llama 3 Architecture from Scratch Using PyTorch：https://pub.towardsai.net/build-your-own-llama-3-architecture-from-scratch-using-pytorch-2ce1ecaa901c
- Decoding Llama3 microblogs series：https://hasgeek.com/simrathanspal/the-llama3-guide/sub/decoding-llama3-part-1-intro-to-llama3-RCehJkfUH348ryim1x6PLN#h:up-next-part-2-understanding-the-configuration
  - Part 1 - Intro to Llama3
  - Part 2 - Understanding the configuration
  - Part 3 - Normalisation
  - Part 4 - Rotary Positional Embeddings
  - Part 5 - Grouped Query Attention
  - Part 6 - Feed Forward Network
  - Part 7 - Transformer Block
- Deep dive into self-attention by hand：https://medium.com/data-science/deep-dive-into-self-attention-by-hand-%EF%B8%8E-f02876e49857
- Deep dive into Llama 3 by hand：https://medium.com/data-science/deep-dive-into-llama-3-by-hand-%EF%B8%8F-6c6b23dc92b2
- （不错）用 Transformer Engine 加速 HF Llama：https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
