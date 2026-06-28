# Transformers 推理

> Hugging Face `transformers` 是 LLM 推理的"参考实现"与事实基准：`model.generate()` 一行调用即可跑通自回归生成，正确、通用、易调试，但因缺少 PagedAttention/连续批处理而吞吐受限，是衡量 vLLM 等高性能引擎加速比的 baseline。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/huggingface-transformers/README]] [[llm-inference/vllm/README]] [[llm-inference/解码策略]]

## 阅读地图

| 节 | 主题 | 你将学到 | 一句话结论 |
|----|------|----------|-----------|
| 0 | 一句话锚点 | transformers 推理的本质 | 它是"会跑就行"的正确基准，不是"跑得快"的生产引擎 |
| 1 | 地基 | 自回归生成解决什么问题 | 把"预测下一个 token"循环成"生成一段文本" |
| 2 | generate 调用链 | 一次 `generate` 内部发生了什么 | prefill 一次 + decode 循环 N 次 |
| 3 | KV Cache | 为什么不重算、缓存什么 | 用显存换计算，把 $O(n^2)$ 退化成逐步 $O(n)$ |
| 4 | 解码策略 | greedy/beam/sampling 在哪一步生效 | 都只改"如何从 logits 选 token" |
| 5 | 为什么慢 | 显存碎片 + 静态批 + 串行 decode | 没有 PagedAttention 和连续批处理 |
| 6 | 作为 baseline | 为什么人人拿它对标 | 正确性金标准 + 加速比分母 |
| 7 | 何时够用 | 不必上 vLLM 的场景 | 离线/低并发/研究/调试 |
| 8 | 与 vLLM 差距 | 吞吐差几倍、差在哪 | 同卡同模型常见 5~24× 吞吐差 |
| — | 配置示例 | 关键参数含义与权衡 | 见"典型流程/配置示例" |

## 0. 一句话锚点

`transformers` 的推理 = **一个 PyTorch 前向 + 一个 Python 循环 + 一份 KV Cache**。

它的设计目标是**通用与正确**（覆盖所有模型架构、所有解码策略、便于研究与调试），不是**高吞吐**。所以：
- 学原理：它是最佳教科书，每一步都看得见。
- 跑生产：高并发在线服务请改用 vLLM / TGI / SGLang。
- 做评测：它是"加速比"的分母，是"输出是否正确"的金标准。

## 1. 地基：自回归生成解决什么问题

LLM 是一个**条件概率模型**：给定已有 token 序列 $x_{1..t}$，输出下一个 token 的概率分布。

$$
P(x_{t+1} \mid x_1, x_2, \dots, x_t) = \text{softmax}(W_o \cdot h_t)
$$

其中 $h_t$ 是最后一层 Transformer 在位置 $t$ 的隐藏状态，$W_o$ 是词表投影（lm_head），输出维度 = 词表大小 $|V|$（常见 3 万~15 万）。

模型**一次前向只能预测一个 token**。要生成一句话，就得把"预测下一个"循环起来——这就是**自回归（autoregressive）生成**：

```
输入: "今天天气"
  ↓ forward → logits → 选出 "真"
输入: "今天天气真"
  ↓ forward → logits → 选出 "好"
输入: "今天天气真好"
  ↓ forward → logits → 选出 <eos>  → 停止
输出: "今天天气真好"
```

**为什么这是难点**：生成 $N$ 个 token 就要做 $N$ 次前向，且每次前向都依赖上一次的输出，**天然串行**，无法在序列维度上并行。这是 LLM 推理慢的根因，也是后面所有优化的出发点。

## 2. `model.generate` 调用链：一次生成里发生了什么

`generate()` 把上面的循环封装好了。它内部分两个阶段：**Prefill（预填充）** 与 **Decode（解码）**。

```
 model.generate(input_ids, max_new_tokens=100, ...)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Prefill 阶段（一次性，处理整段 prompt）       │
 │  input: [t1 t2 t3 ... tP]   (P 个 prompt token)│
 │  一次 forward 并行算完所有位置的注意力          │
 │  → 写入 KV Cache（P 个位置的 K、V）            │
 │  → 取最后一个位置 logits → 选出第 1 个新 token  │
 └─────────────────────────────────────────────┘
        │  生成 token #1
        ▼
 ┌─────────────────────────────────────────────┐
 │  Decode 阶段（循环，每次只喂 1 个 token）       │
 │  step k:  input = [上一步生成的 1 个 token]     │
 │    forward（只算这 1 个位置的 Q）              │
 │    读 KV Cache 里前面所有位置的 K、V 做注意力    │
 │    → 追加新位置的 K、V 进 Cache                │
 │    → logits → 选出第 k+1 个 token             │
 │  直到 命中 eos / 达到 max_new_tokens          │
 └─────────────────────────────────────────────┘
        │
        ▼
   StoppingCriteria 判停 → 返回 sequences
```

关键对比：

| 阶段 | 输入长度 | 计算特征 | 瓶颈 |
|------|----------|----------|------|
| Prefill | 整个 prompt（P 个 token） | 算力密集（大矩阵乘） | **compute-bound**（GPU 算力） |
| Decode | 每步 1 个 token | 访存密集（读全部 KV + 权重） | **memory-bound**（显存带宽） |

> 一句话：**prefill 拼算力，decode 拼带宽**。LLM 服务的延迟和吞吐主要由 decode 阶段决定，因为它要循环很多步。

`generate()` 内部的核心组件（按职责）：
- **LogitsProcessor**：在采样前对 logits 动手脚（temperature、top-k、top-p、重复惩罚等）。
- **StoppingCriteria**：判断是否停止（eos、最大长度、自定义 stop string）。
- **Sampler / Search**：greedy / sampling / beam search 的选择逻辑（见第 4 节）。
- **Cache**：KV Cache 的存取（见第 3 节）。

## 3. KV Cache：用显存换计算

### 3.1 不缓存会重复算什么

注意力的核心是：每个位置的 Query 要和**前面所有位置**的 Key、Value 交互：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

在 decode 第 $k$ 步，新 token 的 $Q$ 要和位置 $1..k$ 的 $K、V$ 做注意力。**但位置 $1..k-1$ 的 $K、V$ 在前面的步骤里早就算过了**——它们只依赖各自的输入 token，不随新 token 变化。

如果不缓存，每步都要把前面所有 token 重新前向一遍：

```
不缓存（朴素）:  生成第 k 个 token 要前向 k 个位置 → 总计算 ∝ 1+2+...+N = O(N²)
有 KV Cache:    生成第 k 个 token 只前向 1 个位置 → 总计算 ∝ N = O(N)（逐步看是 O(1) 新增）
```

### 3.2 缓存的是什么、有多大

缓存的是每一层、每个 KV head 的 **K 张量和 V 张量**（注意：缓存 K/V，**不**缓存 Q，因为 Q 只对当前 token 有意义）。

单个请求的 KV Cache 显存（字节）估算：

$$
\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times b
$$

- $2$：K 和 V 两份
- $L$：层数
- $H_{kv}$：KV head 数（MQA/GQA 下远小于 attention head 数）
- $d_{head}$：每个 head 维度
- $S$：序列长度（prompt + 已生成）
- $b$：每个数的字节数（fp16=2，fp8=1）

**数值例子**（类 7B 模型：$L=32, H_{kv}=32, d_{head}=128$，fp16）：

每 token KV = $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ 字节 ≈ **0.5 MB/token**。

序列到 2048 token：$2048 \times 0.5\,\text{MB} ≈ 1\,\text{GB}$，**仅一个请求**。若用 GQA 把 $H_{kv}$ 降到 8，则缩到约 256 MB——这就是 GQA/MQA 省显存的来历。

```
KV Cache 增长（单请求）:

token:   1     2     3   ...   S
KV:    [k1]  [k1]  [k1]       [k1 k2 ... kS]   ← K 缓存逐步追加
       [v1]  [v2]  ...        [v1 v2 ... vS]   ← V 缓存逐步追加
显存:   0.5MB 1.0MB ...        S×0.5MB         ← 线性增长，停不下来
```

### 3.3 KV Cache 的代价 = transformers 慢的伏笔

KV Cache 解决了"重复计算"，但带来三个新麻烦，**transformers 默认实现都没解决好**：

1. **显存随长度线性涨**：长上下文或多并发时显存爆炸。
2. **必须预留连续显存**：朴素实现给每个请求按 `max_length` 一次性分配一块连续 Cache，**用不满就浪费**（内部碎片）。
3. **请求间无法共享**：相同前缀（如同一 system prompt）各存一份，重复占用。

vLLM 的 PagedAttention 正是针对 2、3 设计的（见第 5、8 节）。

## 4. 解码策略：只改"怎么选 token"

无论哪种解码策略，前面的 forward、KV Cache 都一样；区别只在**拿到 logits 后如何选下一个 token**。

```
            hidden_state h_t
                  │
                  ▼
            lm_head → logits  (维度 = 词表大小)
                  │
       ┌──────────┴───────────────┐
       ▼                          ▼
  LogitsProcessor           （beam search 维护多条候选束）
  temperature/top-k/top-p
  repetition_penalty
       │
       ▼
    选 token:
  ├ greedy        argmax，确定性，易复读
  ├ beam search   保留 num_beams 条候选，偏"高概率整体序列"
  └ sampling      按概率随机抽，配 temperature/top-k/top-p 控制多样性
```

| 策略 | `generate` 关键参数 | 特点 | 适用 |
|------|--------------------|------|------|
| Greedy | `do_sample=False` | 确定、最快、易陷入重复 | 抽取/分类/确定性任务 |
| Beam Search | `num_beams>1` | 找整体高概率序列、显存×beam | 翻译/摘要（追求"标准答案"） |
| Sampling | `do_sample=True` + `temperature/top_k/top_p` | 多样、有创造性 | 对话/写作 |

> 细节见 [[llm-inference/解码策略]]。这里只强调：**解码策略不改变 transformers 的性能瓶颈**——慢在循环和访存，不在选 token。

## 5. 为什么慢：没有 PagedAttention 和连续批处理

把 transformers 的"慢"拆成三个独立原因：

### 5.1 显存碎片化（无 PagedAttention）

朴素 KV Cache 按 `max_length` 预分配一整块连续显存。

```
请求实际只用 200 token，却按 max_length=2048 预留:

[████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ← 用 200，留 2048
 已用      浪费（内部碎片，别的请求也用不上）

PagedAttention（vLLM）做法: KV 切成固定大小的 block（像 OS 内存分页），
按需分配、非连续也能用、相同前缀可共享 block:

block表 → [blk7][blk2][blk9] ...  ← 按需取，几乎零浪费
```

结果：transformers 同样显存下能容纳的并发/序列远少于 vLLM。

### 5.2 静态批处理（无连续批处理 / continuous batching）

transformers 的 batch 是**静态**的：一个 batch 里所有请求**一起开始、一起结束**。

```
静态批（transformers 朴素）:  4 个请求，长度 5/50/8/100
 req A |████|.................（早结束，但要等）
 req B |████████████████████████████████|
 req C |██████|...............（早结束，但要等）
 req D |██████████████████████████████████████████|
          ↑ A、C 早就生成完了，GPU 却空转等最长的 D
          ↑ batch 内还要 padding 到最长，padding 也白算

连续批（vLLM continuous batching）:
 A 结束 → 立刻塞入新请求 E，GPU 不空转
 以"迭代"为粒度调度，每步都把空位填满 → 利用率拉满
```

静态批的两大浪费：**短请求等长请求（队头阻塞 + GPU 空转）** 和 **padding 算无用功**。

### 5.3 Decode 本质访存受限（memory-bound）

decode 每步只生成 1 个 token，却要**把整个模型权重 + 全部 KV Cache 从显存读一遍**。算的少、读的多，GPU 算力大量闲置：

```
decode 一步的时间 ≈ (模型权重字节 + KV字节) / 显存带宽
              而非 (浮点运算量) / 算力

→ batch=1 时，算力利用率常常只有个位数百分比
→ 把多个请求"凑成大 batch"才能摊薄权重读取，提高利用率
   但 transformers 静态批又凑不好（见 5.2）→ 恶性循环
```

> 三因素叠加：**碎片**让你装不下大 batch，**静态批**让你凑不出满 batch，于是 **memory-bound 的 decode** 始终在低利用率运行。这就是 transformers"慢"的完整链条。

## 6. 作为 baseline：为什么人人拿它对标

`transformers` 在工程与论文里被当作**默认基准**，原因有四：

1. **正确性金标准**：模型作者通常先在 transformers 上发布权重与参考实现。要验证 vLLM/TGI 输出是否正确，就和 transformers 逐 token 对齐（greedy 下应完全一致）。
2. **覆盖最全**：几乎所有新模型第一时间在 transformers 可用，没有"引擎还没适配"的尴尬。
3. **加速比的分母**：报告"vLLM 比 baseline 快 N 倍"时，baseline 多半就是 transformers。
4. **易读易改**：研究/调试时能逐层打印、单步跟踪，是理解机制的最佳载体。

```
评测里典型角色分工:
  transformers  →  正确性 oracle + 吞吐分母（baseline）
  vLLM/TGI/SGLang → 被测的高性能引擎（分子）
  指标: 吞吐(tokens/s)、TTFT(首token延迟)、TPOT(每token延迟)、显存占用
```

## 7. 何时够用：不必上 vLLM 的场景

不是所有场景都需要高性能引擎。以下情况 transformers 完全够用，甚至更合适：

| 场景 | 为什么够用 |
|------|-----------|
| 离线批处理 | 不在意延迟，跑完就行；可配合 `device_map` 切分多卡 |
| 低并发 / 单用户 | 没有排队和连续批的收益，引擎复杂度反成负担 |
| 研究与原型 | 要改 forward、插 hook、看中间量，transformers 最灵活 |
| 调试与对齐 | 验证新引擎输出正确性的 oracle |
| 冷门 / 自定义架构 | vLLM 可能尚未适配，transformers 现成可用 |
| 训练/微调内自带评估 | 训练栈本就是 transformers，复用最省事 |

判断口诀：**没有"高并发在线低延迟"硬需求时，先用 transformers**；遇到 QPS 压不住、延迟超标、显存装不下并发，再换 vLLM。

## 8. 与 vLLM 的差距：差几倍、差在哪

### 8.1 差距来源一一对应

| 维度 | transformers | vLLM | 谁赢 |
|------|--------------|------|------|
| KV 显存管理 | 连续预分配，有碎片 | PagedAttention 分页，近零浪费 | vLLM |
| 批处理 | 静态批，队头阻塞 | 连续批，迭代级调度 | vLLM |
| 前缀共享 | 各存一份 | Prefix Caching 共享 block | vLLM |
| 注意力 kernel | 通用实现（可接 SDPA/FlashAttention） | 高度优化 + Paged kernel | vLLM |
| 易用/灵活 | 极高，随便改 | 引擎黑盒，定制成本高 | transformers |
| 模型覆盖 | 最全、最快适配新模型 | 需引擎适配，稍滞后 | transformers |
| 适用 | baseline / 离线 / 研究 | 高并发在线服务 | 各有所长 |

### 8.2 吞吐差距量级

在**同卡、同模型、高并发**下，vLLM 相对 transformers 的吞吐提升常见在 **数倍到二十几倍**（官方与社区基准里常报 5~24×，具体数取决于模型、序列长度、并发与硬件）。

> ⚠️ 具体倍数随版本/场景变化很大，**以官方文档与你自己的复测为准**，不要把某个数字当定值。

```
直觉图（高并发吞吐, 越高越好）:

 vLLM         ████████████████████████  ← 连续批 + Paged，满载
 TGI/SGLang   ███████████████████
 transformers ██                        ← 静态批 + 碎片，多数时间空转
              └────────────────────────→ tokens/s
```

> 注意：**低并发 / batch=1 时差距会显著缩小**——连续批和分页的红利来自"高并发把 GPU 填满"。所以"差 N 倍"只在压力场景成立，单请求时两者接近。

## 典型流程/配置示例（讲含义，不背默认值）

下面是一段最小推理代码骨架，逐参数说明含义与权衡（**具体默认值/版本行为以官方文档为准**）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # 半精度: 省一半显存、提速; 数值范围比 bf16 小
    device_map="auto",           # 自动按显存把层分到多卡/CPU
    # attn_implementation="flash_attention_2",  # 装了就用, 省显存提速
)

inputs = tok("请解释 KV Cache:", return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=256,      # 最多新生成多少 token（控成本与延迟的主旋钮）
    do_sample=True,          # True=采样(多样) / False=greedy(确定)
    temperature=0.7,         # <1 更确定, >1 更发散; 仅采样时生效
    top_p=0.9,               # 核采样: 只在累计概率 0.9 的候选里抽
    top_k=50,                # 只在概率最高的 50 个候选里抽
    repetition_penalty=1.1,  # >1 抑制复读
    use_cache=True,          # 开 KV Cache（关掉会退化成 O(n²)，仅调试用）
    # num_beams=4,           # 启用 beam search; 显存与时间约 ×beam
)
print(tok.decode(out[0], skip_special_tokens=True))
```

参数权衡速记：

| 参数 | 调大的影响 | 调小的影响 | 权衡 |
|------|-----------|-----------|------|
| `max_new_tokens` | 输出更长、更慢、更贵 | 可能被截断 | 延迟/成本 vs 完整性 |
| `temperature` | 更发散、更易跑题 | 更确定、更易复读 | 创造性 vs 稳定性 |
| `top_p` / `top_k` | 候选更多、更随机 | 候选更少、更保守 | 多样性 vs 可控性 |
| `num_beams` | 序列更"高概率"、显存×beam | 退化到单路 | 质量 vs 成本 |
| `torch_dtype` (fp16/bf16) | — | — | bf16 数值更稳、fp16 范围小但更省 |
| `use_cache` | 必开（推理） | 仅调试关 | 速度 vs 内存 |

> 关于 `padding_side`、`pad_token` 在批量推理时要设对（通常生成任务左 padding），以及 `attention_mask` 必须传，否则结果错乱——**具体行为以官方文档/源码为准**。

## 常见问题

| 问题 | 解答 |
|------|------|
| 为什么 greedy 下 transformers 和 vLLM 输出应一致？ | 两者数学等价，差别只在 KV 管理/批处理/kernel 等"怎么算"，不改"算什么"。极少数因浮点累加顺序产生末尾分歧。 |
| `use_cache=True` 既然这么关键，为什么还会慢？ | KV Cache 解决"重复计算"，但 decode 仍是 memory-bound + 静态批 + 碎片，慢在这些地方，不在重复算。 |
| 为什么我 batch=1 测下来 vLLM 没快多少？ | 连续批/分页的红利来自高并发填满 GPU。单请求下两者都接近串行 decode，差距自然小。 |
| KV Cache 会不会爆显存？ | 会。它随"序列长度 × 并发数"线性增长。GQA/MQA 降 KV head、量化 KV、PagedAttention 都是为省它。 |
| Prefill 和 Decode 谁是延迟主因？ | TTFT（首 token）由 prefill 决定；TPOT（后续每 token）由 decode 决定。长输出场景 decode 累加是大头。 |
| 什么时候必须从 transformers 换成 vLLM？ | 出现"高并发压不住 / 延迟超标 / 显存装不下并发"任一信号时。否则 transformers 更省心。 |
| beam search 为什么吃显存？ | 同时维护 `num_beams` 条候选序列，每条都有独立 KV Cache，显存与计算约 ×beam。 |
| 关掉 KV Cache 会怎样？ | 退化成每步重算全部历史，复杂度回到 $O(n^2)$，只在调试/教学时这么做。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图与学习路径
- [[ai-framework/huggingface-transformers/README]] — transformers 框架本体（模型/分词/训练全貌）
- [[llm-inference/vllm/README]] — 高性能推理引擎，PagedAttention + 连续批处理
- [[llm-inference/解码策略]] — greedy / beam / sampling / top-k / top-p 详解

参考资料：
- https://huggingface.co/blog/how-to-generate
- https://huggingface.co/blog/zh/how-to-generate
