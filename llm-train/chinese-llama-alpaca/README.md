# 中文 LLaMA & Alpaca

> 在原版 LLaMA 上「扩中文词表 + 二次预训练 + 中文指令微调」，用 LoRA 低成本把一个英文为主的基座模型改造成懂中文、能对话的模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/llama/模型架构]] [[llm-train/peft/PEFT-API]] [[llm-data-engineering/README]]

---

## 阅读地图

| 节 | 你会搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | 词表扩充 / 二次预训练 / 指令微调 / LoRA |
| 1 | 它解决什么问题 | 原版 LLaMA 中文为什么差 |
| 2 | Token 与词表：为什么必须扩词表 | tokenizer / token 效率 / 上下文浪费 |
| 3 | 怎么扩词表(SentencePiece 合并) | BPE / merge / embedding 重采样 |
| 4 | 阶段一：中文二次预训练 | continual pre-train / CLM loss |
| 5 | 阶段二：中文指令微调(Alpaca) | SFT / instruction / Self-Instruct |
| 6 | LoRA：为什么能省钱 | 低秩 / 冻结 / 适配器合并 |
| 7 | 端到端全流程串讲 | 数据→词表→PT→SFT→合并→部署 |
| 8 | 典型配置示例(讲含义) | rank / alpha / 学习率 / 模块 |
| 9 | 常见问题 | FAQ |

源码与版本参考：
- 源码地址: https://github.com/ymcui/Chinese-LLaMA-Alpaca/
- commit id : 3e2f2529a4dc0d7567f46f1b2d3431a7d063588b
- 注：以下精确数字(词表大小/超参)以官方 README/脚本为准，本文重在讲清机制与取舍。

---

## 0. 一句话锚点

**Chinese-LLaMA-Alpaca = 词表换"中文键盘" + 二次预训练"补中文知识" + 指令微调"教会听话" + 全程用 LoRA"省显存省钱"。**

- **Chinese-LLaMA**：在原版 LLaMA 基座上扩中文词表 + 中文语料二次预训练得到的**基座**(只会续写，不会对话)。
- **Chinese-Alpaca**：在 Chinese-LLaMA 上再做**中文指令微调**得到的**对话模型**(会听指令、能回答)。
- 命名沿用斯坦福：LLaMA = 基座，Alpaca = 指令版(原始 Alpaca 是用 Self-Instruct 造的指令数据微调 LLaMA)。

```
原版 LLaMA(英文为主)
   │  ① 扩中文词表 + resize embedding
   ▼
中文 tokenizer 的 LLaMA(还没补知识)
   │  ② 中文大规模无标注语料 → 二次预训练(LoRA)
   ▼
Chinese-LLaMA(懂中文，但只会续写)
   │  ③ 中文指令数据 → 指令微调(LoRA)
   ▼
Chinese-Alpaca(懂中文，且会对话)
```

---

## 1. 地基/前置：原版 LLaMA 的中文为什么差

LLaMA 是 Meta 开源的英文为主基座模型，训练语料里中文占比极低。直接拿来用中文，有两个层面的问题：

1. **知识层面**：模型见过的中文文本少，中文世界知识、表达习惯都弱。
2. **编码层面(更隐蔽、更致命)**：LLaMA 的 tokenizer 词表里几乎没有完整中文词/字，中文会被拆成**字节(byte)级**或零碎片段。一个汉字往往要 2~3 个 token 才能表示。

第二点直接拖垮一切：

```
英文 "model"      → 1 个 token
中文 "模型"        → 原版可能 4~6 个 token(逐字节)
                     扩词表后 → 1~2 个 token
```

token 越多意味着：
- **上下文被浪费**：固定 4K/2K 上下文，原版可能只装下几百汉字。
- **推理更慢、更贵**：生成同样长度的中文，要预测更多步。
- **学习更难**：语义被打碎成字节，模型难以建立"模型"这个词的整体表示。

> 所以 Chinese-LLaMA-Alpaca 的第一刀就是**扩词表**——这是后续一切的地基。

---

## 2. Token 与词表：为什么必须扩词表(token 效率)

### 2.1 什么是 token 效率

定义"**压缩率**"：一段文本被切成的 token 数 ÷ 字符数，越小越好。

设一段中文有 $N$ 个字符，tokenizer 平均每字符切成 $r$ 个 token：

$$\text{tokens} = r \cdot N$$

- 原版 LLaMA 对中文：$r \approx 1.5\sim2.5$(很多汉字按字节切)。
- 扩词表后：$r \approx 0.6\sim1.0$(常用词/字一个 token)。

**数值例子**：写一篇 1000 字中文文章。

| 方案 | $r$ | token 数 | 2048 上下文能装下吗 |
|------|-----|----------|----------------------|
| 原版 LLaMA | 2.0 | ~2000 | 几乎装不下一篇 |
| 扩词表后 | 0.7 | ~700 | 还能塞进对话历史 |

同一段话，token 数从 2000 降到 700，**等效上下文长度扩大近 3 倍，推理成本降约 65%**。这就是"扩词表"最直接的收益。

### 2.2 为什么不直接把原版上下文调长

调长上下文要重训、显存平方级增长，且不解决"语义被字节打碎"的问题。扩词表是**更便宜、更治本**的做法。

---

## 3. 怎么扩词表(SentencePiece 词表合并)

LLaMA 用 **SentencePiece(BPE/Unigram)** 做分词。扩词表的核心是：**在中文语料上训练一个中文 tokenizer，再把它的词条"合并"进原版词表**。

```
原版 LLaMA 词表 (V0 ≈ 32000，英文+字节 fallback)
        +
中文 SentencePiece 词表 (在中文语料上新训，约 2 万中文 token)
        │ 去重合并(中文 token 不与原词表冲突的部分)
        ▼
扩充后词表 (V1 ≈ 49000+，多出的是中文字/词)
        │
        ▼
resize 模型的 embedding 矩阵 + 输出 lm_head
```

### 3.1 合并步骤(机制层面)

```
1. 准备中文语料(百科/新闻/网页等)。
2. 用 SentencePiece 训练一个中文分词模型 → 得到中文 token 列表。
3. 遍历中文 token：原词表里没有的 → 追加；已有的 → 跳过(去重)。
4. 词表大小 V0 → V1。
5. 改 embedding：旧 V0×d 矩阵 → 新 V1×d。
   - 前 V0 行：直接复制原版权重(保住英文能力)。
   - 新增 (V1−V0) 行：随机初始化 / 用已有 embedding 均值初始化。
6. lm_head(输出投影)同样从 V0 扩到 V1。
```

### 3.2 为什么新增行要"小心初始化"

新增的中文 token embedding 是随机的，模型完全不认识它们。如果随机值太离谱，初期 loss 会爆炸。常用做法是**用旧词表全部 embedding 的均值**初始化新行，让它们从"平均语义"出发，再靠二次预训练慢慢学。

> 关键直觉：**扩词表只是给了模型"中文键盘"，键盘上的键(新 token)还没有意义——意义要靠第 4 步的二次预训练注入。**

```
embedding 矩阵 resize 示意:
        d 维
   ┌──────────────┐
V0 │ 原版权重复制  │  ← 英文能力不丢
   ├──────────────┤
ΔV │ 新中文 token  │  ← 随机/均值初始化，待训练
   └──────────────┘
   V1 = V0 + ΔV
```

---

## 4. 阶段一：中文二次预训练(continual pre-training)

目标：**让模型真正"懂"中文**——把新加的中文 token 训出语义，补上中文世界知识。

- **任务形式**：标准自回归语言建模(Causal LM)，预测下一个 token。
- **损失函数**：交叉熵

$$\mathcal{L}_{PT} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})$$

- **数据**：大规模**无标注**中文纯文本(百科、新闻、网页、图书……见 [[llm-data-engineering/README]] 的清洗去重流程)。
- **"二次/继续"**：不是从零训，而是在原版 LLaMA 权重基础上接着训(continual pre-train)，所以叫"二次预训练"。

```
数据流(预训练):
原始中文文本
  → 清洗/去重/质量过滤
  → 中文 tokenizer 切成 token id
  → 打包成定长序列(如 512/1024)
  → 自回归预测下一个 token，算 CLM loss
  → 反向传播(只更新 LoRA + embedding/lm_head)
```

### 4.1 这一步通常分两阶段(工程经验)

很多实现会先**只训练新增的 embedding/lm_head**(让新 token 先"对齐"语义)，再**放开 LoRA 一起训**(让全网络适应中文)。这样更稳，避免一上来新 token 的乱梯度污染整个模型。

> 产出 = **Chinese-LLaMA 基座**：能流畅续写中文，但你问它问题，它可能继续"补全"而不是"回答"——因为它还没学会"听指令"。

---

## 5. 阶段二：中文指令微调(Chinese-Alpaca)

目标：**让模型从"会续写"变成"会听指令、会对话"**。这就是 SFT(Supervised Fine-Tuning)。

### 5.1 指令数据长什么样

每条样本是 (instruction, input, output) 三元组，套进固定**提示模板**：

```
### Instruction:
把下面这句话翻译成英文。
### Input:
今天天气真好。
### Response:
The weather is really nice today.
```

- 数据来源：翻译/改写的 Alpaca 指令、Self-Instruct 自动生成、人工标注、开源中文指令集等。
- **损失只算在 Response 部分**(instruction/input 作为条件，不计入 loss)——只教模型"该怎么答"，不教它"复读问题"。

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{Response}}\log p_\theta(y_t \mid \text{prompt}, y_{<t})$$

### 5.2 PT 和 SFT 的区别(一张表说清)

| 维度 | 二次预训练(PT) | 指令微调(SFT) |
|------|----------------|----------------|
| 数据 | 海量无标注纯文本 | 较少的(指令,回答)对 |
| 学什么 | 中文知识 + 新 token 语义 | 听指令、对话格式、有用性 |
| loss 算在哪 | 全序列 | 只算 Response |
| 产物 | Chinese-LLaMA(基座) | Chinese-Alpaca(对话) |
| 规模 | 大(GB~TB) | 小(几万~几十万条) |

> 顺序不能反：先有知识(PT)，再教表达(SFT)。直接在没补中文知识的模型上做 SFT，模型"会说中文的样子但没内容"。

---

## 6. LoRA：为什么全程用它(省钱的关键)

全参数微调 LLaMA 要更新数十亿参数，显存和算力都极高。LoRA(Low-Rank Adaptation)让普通显卡也能训。详见 [[llm-train/peft/PEFT-API]]。

### 6.1 核心思想

冻结原权重 $W \in \mathbb{R}^{d\times k}$，在旁边并联一个**低秩**增量 $\Delta W = BA$：

$$h = Wx + \Delta W x = Wx + \underbrace{B}_{d\times r}\underbrace{A}_{r\times k}\,x,\quad r \ll \min(d,k)$$

- $A$ 随机高斯初始化，$B$ 初始化为 0 → 训练开始时 $\Delta W = 0$，不破坏原模型。
- 实际用时常乘缩放系数 $\frac{\alpha}{r}$：$h = Wx + \frac{\alpha}{r}BAx$。

```
        x
        │
   ┌────┴─────┐
   ▼          ▼
 [ W ]      [ A ]  r×k   ← 只训这两个小矩阵
 冻结        │
   │        ▼
   │      [ B ]  d×r
   │        │ ×(α/r)
   └───►(+)◄┘
        │
        ▼  h
```

### 6.2 为什么省

只训 $A,B$，参数量从 $d\times k$ 降到 $r\times(d+k)$。

**数值例子**：某层 $d=k=4096$，取 $r=8$。

- 全参数：$4096\times4096 \approx 1678$ 万。
- LoRA：$8\times(4096+4096) = 6.5$ 万。
- 可训练参数降到约 **0.4%**，显存(优化器状态)随之骤降，单卡 24G 即可玩。

### 6.3 训练完怎么用

LoRA 权重很小(几十~几百 MB)，可以：
- **挂载**：推理时把 $\Delta W$ 加到 $W$ 上(`merge`)，得到完整模型，零额外延迟；
- **分发**：只发 LoRA 增量，用户自己合并到原版 LLaMA。

> 在本项目中，**词表扩充后的 embedding/lm_head 通常一并训练并随 LoRA 一起保存**，因为新 token 的 embedding 必须更新(LoRA 之外的部分)。这也是合并脚本要单独处理 embedding 尺寸的原因。

---

## 7. 端到端全流程串讲

把前面所有环节连成一条流水线：

```
┌──────────────────────────────────────────────────────────────┐
│  数据准备   [[llm-data-engineering/README]]                    │
│   中文纯文本(PT用) + 中文指令数据(SFT用)                        │
└───────────────┬──────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────┐
│  ① 词表扩充                                                    │
│   中文 SentencePiece + 原版词表 → 合并 → resize embedding      │
└───────────────┬──────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────┐
│  ② 二次预训练 (LoRA + 训 embedding/lm_head)                    │
│   无标注中文 → CLM loss → Chinese-LLaMA 基座                   │
└───────────────┬──────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────┐
│  ③ 指令微调 (LoRA)                                             │
│   (指令,回答)对 → 只算 Response 的 loss → Chinese-Alpaca       │
└───────────────┬──────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────┐
│  ④ 合并权重                                                    │
│   原版 LLaMA + LoRA(+扩充embedding) → 完整中文模型             │
└───────────────┬──────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────┐
│  ⑤ 部署/推理                                                   │
│   转 HF / GGUF(llama.cpp) → CPU/GPU 跑                         │
└──────────────────────────────────────────────────────────────┘
```

调用链(以指令微调为例)：

```
run_clm_sft.py
   ├─ load 原版 LLaMA(扩词表后的 config)
   ├─ resize_token_embeddings(V1)        # 对齐新词表
   ├─ get_peft_model(LoraConfig)         # 注入 LoRA
   ├─ load 中文指令数据 → 套模板 → 切 token
   ├─ Trainer.train()                    # 只更新 LoRA + embedding
   └─ save LoRA 适配器(+ tokenizer)
```

> 注：脚本名/参数以官方仓库为准，这里展示的是**典型调用结构**而非逐字命令。

---

## 8. 典型配置示例(讲含义，不是照抄)

下面是 LoRA 微调常见配置项及其权衡(**数值为示意，实际以官方脚本为准**)：

```yaml
# LoRA 配置
lora_rank: 8            # 秩 r。越大表达力越强、参数越多；中文任务常 8~64
lora_alpha: 32          # 缩放 α，实际放大系数 = α/r。常令 α=2r~4r
lora_dropout: 0.05      # 防过拟合
target_modules:         # 给哪些层加 LoRA
  - q_proj              # 注意力的 Q/K/V/O 投影最常见
  - v_proj
  - k_proj
  - o_proj
  - gate_proj          # 想更强可加 MLP 三个投影
  - up_proj
  - down_proj
modules_to_save:        # LoRA 之外、也要训练并保存的全量层
  - embed_tokens        # ★ 新增中文 token 的 embedding 必须训
  - lm_head             # ★ 输出层也要随词表扩充更新

# 训练超参
learning_rate: 2e-4     # LoRA 学习率通常比全参大(因可训参数少)
num_train_epochs: 1~3
per_device_batch_size: 微调小、PT 可大
gradient_accumulation: 用小 batch 模拟大 batch，省显存
max_seq_length: 512/1024  # 与上下文、显存权衡
fp16/bf16: true         # 混合精度省显存
```

### 关键取舍

| 参数 | 调大的好处 | 调大的代价 |
|------|-----------|-----------|
| `lora_rank` | 拟合能力强 | 参数多、易过拟合、慢 |
| `target_modules` 加 MLP | 效果上限高 | 显存/时间增加 |
| `learning_rate` | 收敛快 | 太大震荡/发散 |
| `max_seq_length` | 能学长文 | 显存平方级涨 |

> **必记坑**：扩词表后 `embed_tokens` 和 `lm_head` 一定要放进 `modules_to_save`，否则新中文 token 的 embedding 永远是随机初始化，模型学不会中文。

---

## 9. 与同类对比 / 常见问题

| 问题 | 答案 |
|------|------|
| Chinese-LLaMA 和 Chinese-Alpaca 区别？ | 前者是基座(只续写)，后者在前者上做指令微调(会对话)。 |
| 为什么非要扩词表，不能只做 SFT？ | 不扩词表，中文被切成字节，token 效率低、上下文浪费、学习困难——治标不治本。 |
| 二次预训练能省吗？ | 不建议。没补中文知识直接 SFT，模型"会说中文格式但没内容"。 |
| LoRA 会损失精度吗？ | 多数任务接近全参微调；追求极致可全参，但成本高得多。 |
| 新 token embedding 随机初始化会怎样？ | 初期 loss 高甚至发散，故常用均值初始化 + 先单独训 embedding。 |
| 和原版 Alpaca 的关系？ | 命名沿用斯坦福 Alpaca；本项目针对中文做了词表扩充与中文 PT/SFT。 |
| 训完怎么部署？ | 合并 LoRA → 转 HF 或 GGUF(llama.cpp)，可 CPU 量化推理。 |
| 后续版本？ | 社区有 Chinese-LLaMA-Alpaca-2 等改进(更长上下文/更优词表)，机制一脉相承，细节以官方为准。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航总入口
- [[llm-algo/llama/模型架构]] — LLaMA 的 RMSNorm/RoPE/SwiGLU 等结构细节
- [[llm-train/peft/PEFT-API]] — LoRA / PEFT 的 API 与实现
- [[llm-data-engineering/README]] — 预训练与指令数据的清洗/去重/构造
