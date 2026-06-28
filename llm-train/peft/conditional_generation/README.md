# PEFT × 条件生成（Conditional Generation / Seq2Seq）

> 把参数高效微调（PEFT：LoRA / Prefix-Tuning / Prompt-Tuning / P-Tuning）用在**编码器-解码器（Encoder-Decoder）**模型（T5 / mT5 / BART / FLAN-T5）上，做「给定输入 X，生成目标序列 Y」的条件生成任务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/peft/README]] [[llm-train/peft/Prompt-Tuning]] [[llm-train/peft/Prefix-Tuning]] [[llm-train/peft/PEFT-API]]

## 阅读地图

| 节 | 内容 | 你将带走 |
|----|------|---------|
| 0 | 一句话锚点 | 一句话记住「条件生成 + PEFT」的本质 |
| 1 | 地基：什么是条件生成、为什么用 Encoder-Decoder | seq2seq 的任务边界 |
| 2 | CLM vs Seq2Seq：PEFT 落点不一样 | 两类模型的差异原子级理解 |
| 3 | 数据流水线：从原始文本到 `labels` | tokenize / padding / -100 的为什么 |
| 4 | PEFT 怎么插进 Encoder-Decoder | LoRA 注入到哪几个矩阵 |
| 5 | Prefix / Prompt-Tuning 在 seq2seq 里的位置 | 软提示插在哪、影响 encoder 还是 decoder |
| 6 | 训练循环 & `Seq2SeqTrainer` | loss、teacher forcing、generate |
| 7 | 数值例子：T5 上 LoRA 的参数账 | 复现「<1% 可训练参数」 |
| 8 | 代码骨架 | 跑起来 |
| 坑表 | 常见问题 | 可背诵 |

## 0. 一句话锚点

- **条件生成**：输入一段文本 $X$，模型**生成**另一段文本 $Y$。典型任务：翻译、摘要、文本→SQL、问答、句子改写、`financial_phrasebank` 这类「句子 → 情感标签词」。
- **Encoder-Decoder（seq2seq）**：用一个 **Encoder** 把 $X$ 压成一组隐藏向量（带**双向**注意力，能看全句），再用一个 **Decoder** 一个 token 一个 token 地生成 $Y$（带**因果**注意力 + **cross-attention** 去看 Encoder 的输出）。代表：T5、mT5、BART。
- **PEFT**：冻结整个 Encoder-Decoder 主体，只训练**极少**的新增参数（LoRA 的低秩矩阵 / Prefix 的前缀向量 / Prompt 的软提示）。

> 本目录（`conditional_generation/`）= 把上面三件事**拼在一起**：用 PEFT 微调一个 seq2seq 模型来做生成任务。它与隔壁 `clm/`（causal LM，纯解码器，如 BLOOM/GPT）是**对照组**。

## 1. 地基：条件生成与 Encoder-Decoder

### 1.1 任务长什么样

```
任务: 把句子分类成情感词 (financial_phrasebank 风格)
输入 X: "The company reported strong quarterly profits ."
目标 Y: "positive"

任务: 翻译
输入 X: "translate English to German: How are you ?"
目标 Y: "Wie geht es dir ?"
```

注意：即使是「分类」，seq2seq 也把它当成**生成一个标签词**来做（生成式分类）。这正是 T5「Text-to-Text」的核心思想——**所有任务都统一成「文本进、文本出」**。

### 1.2 为什么要 Encoder-Decoder

```
       输入 X (可双向看全句)              目标 Y (只能看左边 + 看 X)
   ┌─────────────────────────┐      ┌──────────────────────────────┐
   │        ENCODER          │      │           DECODER            │
   │  双向 Self-Attention     │      │  ① 因果(masked) Self-Attn     │
   │  X -> H = [h1..hn]      │─────►│  ② Cross-Attention 看 H       │
   │                         │  H   │  ③ FFN -> 预测下一个 token     │
   └─────────────────────────┘      └──────────────────────────────┘
        理解 / 压缩输入                       条件 + 自回归 地生成
```

- Encoder 用**双向**注意力：第 1 个词能看到第 $n$ 个词 → 适合「理解」整段输入。
- Decoder 用**因果**注意力 + **cross-attention**：生成第 $t$ 个 token 时，只能看自己已生成的前 $t-1$ 个，但可以通过 cross-attention「回头看」整个输入 $H$ → 适合「在 $X$ 的条件下生成 $Y$」。

> 这就是它叫「**条件**生成」的原因：生成的每一步都**以 $X$（即 $H$）为条件**。纯解码器（CLM）则是把 $X$ 和 $Y$ 拼成一条序列直接续写，没有独立的 Encoder。

## 2. CLM vs Seq2Seq：PEFT 落点不一样

| 维度 | CLM（`clm/`，如 BLOOM/GPT/LLaMA） | Seq2Seq（本目录，如 T5/BART） |
|------|------|------|
| 结构 | 仅 Decoder | Encoder + Decoder |
| 任务表示 | `prompt + answer` 拼成一条，续写 | `X` 进 encoder，`Y` 进 decoder |
| 注意力 | 全程因果（单向） | encoder 双向 / decoder 因果 + cross |
| 注意力块种类 | self-attn | self-attn × 2（enc/dec）+ cross-attn |
| PEFT 可注入的矩阵更多 | q/k/v/o, ffn | encoder 的 q/k/v/o、decoder 的 self q/k/v/o、**cross-attn 的 q/k/v/o**、各自 ffn |
| HF 模型类 | `AutoModelForCausalLM` | `AutoModelForSeq2SeqLM` |
| PEFT TaskType | `CAUSAL_LM` | **`SEQ_2_SEQ_LM`** |
| 训练器 | `Trainer` | `Seq2SeqTrainer`（支持 `predict_with_generate`） |

> **最容易踩的概念坑**：在 PEFT 里给 seq2seq 配 `TaskType.SEQ_2_SEQ_LM`，给 CLM 配 `TaskType.CAUSAL_LM`。配错会导致 Prefix/Prompt-Tuning 的虚拟 token 插错地方、或 label 对齐错位。（确切枚举名以 `peft` 官方库为准。）

## 3. 数据流水线：从原始文本到 `labels`

这是 seq2seq 微调里**最容易写错**的一环。核心：**输入和目标分开 tokenize**。

```
原始样本                tokenize                         喂给模型
─────────              ──────────                        ─────────
X = "...profits ."  ──► input_ids      (encoder 的输入)  ─► model.encoder
Y = "positive"      ──► labels         (decoder 的目标)  ─► 算 loss

                       decoder_input_ids 由 labels 右移一位自动生成
                       (T5: 在最前面补一个 decoder_start_token / pad)
```

### 3.1 三件必须想清楚的事

1. **input 和 target 用不同的最大长度**：输入句子可能很长（`max_length`），目标（如标签词）很短（`max_target_length`）。分别截断，不要共用一个长度。
2. **padding 的 label 要置成 `-100`**：交叉熵损失里 `-100` 是「忽略位」。把 target 里 padding 出来的位置改成 `-100`，否则模型会去「学习预测 pad」，污染 loss。
3. **teacher forcing 自动化**：训练时 decoder 第 $t$ 步的输入是**真实**的 $y_{t-1}$（不是模型自己的预测），这叫 teacher forcing。HF 会用 `labels` 右移自动生成 `decoder_input_ids`，你一般不用手搓。

```
labels:            [ p_o_s , i_t_i , v_e , </s> , <pad> , <pad> ]
处理后(算loss用):   [ p_o_s , i_t_i , v_e , </s> , -100  , -100  ]
                                                   ↑ 被 CrossEntropy 忽略
```

> 数值直觉：若不置 `-100`，假设 batch 里一半 token 都是 pad，那一半的 loss 全在「学着输出 pad」，训练信号被严重稀释，验证集生成会冒出多余空白/结束符乱跳。

## 4. PEFT 怎么插进 Encoder-Decoder（以 LoRA 为例）

LoRA 的本质：冻结原权重 $W$，并联一条低秩支路 $\Delta W = BA$，前向变成

$$h = W x + \frac{\alpha}{r}\,B A\,x,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll \min(d,k)$$

只训练 $A,B$。在 seq2seq 里，可注入的「$W$」比 CLM 多一类——**cross-attention**：

```
            T5 Block (一层) 内的注意力矩阵
Encoder 侧:
   self-attn:   q  k  v  o     ← 可注入 LoRA
   ffn:         wi wo          ← 可注入 LoRA
Decoder 侧:
   self-attn:   q  k  v  o     ← 可注入 LoRA
   cross-attn:  q  k  v  o     ← 这是 seq2seq 独有，也可注入 LoRA
   ffn:         wi wo          ← 可注入 LoRA
```

```
       原冻结权重 W  (例: T5 的 attention q 投影)
                │
   x ──────────►│ W (frozen)
        │       │           \
        │       └──► W·x ──► (+) ──► h
        │                    /
        └──► A ──► (r 维) ──► B ──► (α/r)·BAx
              (train)        (train)
```

### 4.1 `target_modules`：注到哪几个矩阵

PEFT 用一个 `target_modules` 列表（**模块名的字符串匹配**）决定把 LoRA 加到哪些 `nn.Linear` 上。

- 在 T5 这类模型里，注意力投影常叫 `q`、`k`、`v`、`o`（具体名字以 `model.named_modules()` 实际打印为准，不要硬背）。
- **怎么权衡**：
  - 只注 `q,v`（经典 LoRA 配置）→ 参数最少，多数任务够用，最稳。
  - 加上 `k,o` → 表达力更强、可训练参数变多，小数据集上反而可能过拟合。
  - 加上 ffn（`wi,wo`）→ 进一步增容量，显存/算力成本上升。
- **rank $r$ 与 $\alpha$**：$r$ 越大容量越大（8/16/32 是常见档位）；缩放系数 $\alpha$ 控制 LoRA 支路的「响度」，有效缩放是 $\alpha/r$。一般经验是「先固定 $\alpha=2r$ 起步」，但**没有万能默认值，以任务实测为准**。

> 关键提醒：**不要凭记忆写死参数默认值**。`r`、`lora_alpha`、`lora_dropout`、`target_modules` 都要根据模型实际结构和任务调，具体字段名/默认值以你装的 `peft` 版本与官方文档为准。

## 5. Prefix / Prompt-Tuning 在 seq2seq 里的位置

LoRA 改的是「权重旁路」；Prefix/Prompt 改的是「输入序列」。在 Encoder-Decoder 里它们落点不同：

```
Prompt-Tuning (软提示):
   [P1 P2 ... Pk]  +  X(真实token嵌入)  ──► Encoder
   ↑ k 个可训练"假词向量"拼在输入最前面，只训练这 k×d 个数

Prefix-Tuning (前缀 KV):
   给每一层注意力的 Key/Value 都前置一段可训练的 "虚拟前缀"
   ┌ Encoder 各层 self-attn 的 K,V 前面挂前缀
   └ Decoder 各层 self-attn / cross-attn 的 K,V 前面挂前缀
   （前缀通常由一个小 MLP "重参数化"生成，更稳定）

P-Tuning:
   软提示 + 一个小型 prompt encoder(LSTM/MLP) 来生成提示嵌入
```

| 方法 | 改什么 | 直观理解 | seq2seq 里影响范围 |
|------|--------|---------|----------------|
| Prompt-Tuning | 输入嵌入前缀 | 学一段「咒语」放句首 | 主要影响 encoder 看到的内容 |
| Prefix-Tuning | 每层 KV 前缀 | 在每层注意力里都「塞私货」 | enc/dec 每层都受影响，表达力更强 |
| LoRA | 权重低秩旁路 | 给权重打「低秩补丁」 | 注入的那些矩阵 |
| P-Tuning | 软提示 + 编码器 | 用小网络生成更优软提示 | 类似 Prompt，但更灵活 |

> 选型直觉：**任务难/数据多 → LoRA 或 Prefix**（容量大）；**资源极紧/想最少新增参数 → Prompt-Tuning**（只有 $k\times d$ 个参数）。详见 [[llm-train/peft/README]] 的对比表。

## 6. 训练循环 & `Seq2SeqTrainer`

```
            一个训练 step 的数据流
  batch ──► input_ids ─────────► Encoder ──► H
            labels ──(右移)────► decoder_input_ids ─► Decoder(看 H) ─► logits
            logits + labels ──► CrossEntropy(忽略 -100) ──► loss
            loss.backward()  ──► 只更新 PEFT 的少量参数 ──► optimizer.step()
```

- **训练**：teacher forcing（拿真实 $y_{t-1}$ 当输入），一次前向算完整条 $Y$ 的 loss，快。
- **评估/推理**：必须**自回归 generate**（`model.generate(...)`），一步一步采样/beam search，没有真实 $y$ 可喂。`Seq2SeqTrainer` 用 `predict_with_generate=True` 在评估时切到 generate 模式，从而能算 BLEU/ROUGE/accuracy。
- **解码超参**：`num_beams`（beam search 宽度）、`max_new_tokens`、`length_penalty` 等控制生成质量与长度——这些是**生成**超参，不影响已训练好的权重，可推理时调。

> 训练 loss 低 ≠ 生成好。因为训练用 teacher forcing（有标准答案兜底），推理是自回归（错误会累积，叫 **exposure bias**）。所以一定要用 `generate` 看真实生成效果。

## 7. 数值例子：T5 上 LoRA 的参数账

假设一个 T5-base 量级模型：隐藏维 $d=768$，约 12 层 encoder + 12 层 decoder。给注意力的 `q,v` 注入 LoRA，秩 $r=8$。

单个被注入的 `q`（768×768）新增参数：

$$\underbrace{768\times 8}_{A} + \underbrace{8\times 768}_{B} = 6144 + 6144 = 12288 \text{ 个}$$

粗估被注入的线性层数量（enc 12 层 ×{q,v}=24，dec 12 层 ×{self q,v + cross q,v}=48，共约 72 个）：

$$72 \times 12288 \approx 8.8\times 10^{5}\ \text{个可训练参数}$$

而 T5-base 总参数约 $2.2\times 10^{8}$。可训练占比：

$$\frac{8.8\times 10^{5}}{2.2\times 10^{8}} \approx 0.4\%$$

> 即「冻结 99.6%，只训 0.4%」。这就是 PEFT 的卖点：显存（优化器状态只为这 0.4% 分配 Adam 动量）和 checkpoint 体积（只存 adapter，几 MB）都暴降。**以上为量级估算，确切层数/命名以实际模型为准。**

## 8. 代码骨架（伪代码，跑通顺序）

```python
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import get_peft_model, LoraConfig, TaskType

model_name = "t5-base"                      # 或 mt5 / flan-t5 / bart
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 1) 配 PEFT —— 注意 TaskType 是 SEQ_2_SEQ_LM
peft_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,        # ★ seq2seq 专用
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q", "v"],              # 名字以实际 named_modules 为准
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()          # 看「可训练参数 < 1%」

# 2) 数据处理：input/target 分开 tokenize，target 的 pad 置 -100
def preprocess(ex):
    mi = tok(ex["input"],  max_length=128, truncation=True)
    lb = tok(ex["target"], max_length=8,   truncation=True)["input_ids"]
    lb = [t if t != tok.pad_token_id else -100 for t in lb]   # ★ 忽略位
    mi["labels"] = lb
    return mi

collator = DataCollatorForSeq2Seq(tok, model=model)  # 动态 padding + 右移

# 3) 训练 —— 评估时用 generate
args = Seq2SeqTrainingArguments(
    output_dir="out", predict_with_generate=True,    # ★ 评估走 generate
    per_device_train_batch_size=8, learning_rate=1e-3,  # PEFT 学习率通常偏大
)
trainer = Seq2SeqTrainer(model=model, args=args, data_collator=collator, ...)
trainer.train()

# 4) 只保存 adapter（几 MB），推理时再叠回基座
model.save_pretrained("adapter_dir")
```

> 上面的字段名/默认值仅作示意，**确切 API 签名与默认值以你安装的 `transformers` / `peft` 版本官方文档为准**。PEFT 学习率往往比全量微调大（如 $10^{-3}$ 量级），因为只有少量参数在动。

## 常见问题 / 坑（表格）

| 现象 / 坑 | 根因 | 对策 |
|----------|------|------|
| loss 很低但生成是垃圾 | 只看了 teacher-forcing loss，没 `generate` | 评估开 `predict_with_generate=True`，看真实生成 |
| 生成里夹大量 pad / 提前 `</s>` | target 的 pad 没置 `-100` | label padding 位改成 `-100` |
| PEFT 不收敛 / 报对齐错误 | TaskType 配成了 `CAUSAL_LM` | seq2seq 必须用 `SEQ_2_SEQ_LM` |
| `target_modules` 匹配不到 | 模块名拼错（不同模型命名不同） | 先 `print(model)` / `named_modules()` 看真名 |
| 输入被截断丢信息 | input 和 target 共用一个 `max_length` | 拆成 `max_length` 与 `max_target_length` |
| 显存仍然爆 | 误以为 PEFT 省的是**激活**显存 | PEFT 省的是优化器状态/梯度；激活显存靠 batch/序列长度/梯度检查点控制 |
| adapter 加载后效果丢失 | 基座模型/版本对不上 | adapter 必须叠回**同一个**基座；记录 base model 名 |
| 多语言任务效果差 | 用了纯英文 T5 | 多语言选 mT5 / mBART |
| 学习率照搬全量微调（很小） | 只训少量参数需要更大步长 | PEFT 学习率适当调大（实测） |

> 一句话总结：**条件生成 = Encoder 理解 X + Decoder 在 X 条件下自回归生成 Y；PEFT = 冻结这套 Encoder-Decoder，只训练极少新增参数。** 难点全在「数据对齐（-100、teacher forcing）+ 选对 TaskType + 评估走 generate」三件事。

## 🔗 跳转链接

- 总图：[[00-知识地图]]
- PEFT 总览与方法对比：[[llm-train/peft/README]]
- 软提示家族：[[llm-train/peft/Prompt-Tuning]] ｜ 前缀法：[[llm-train/peft/Prefix-Tuning]]
- PEFT 库 API 用法：[[llm-train/peft/PEFT-API]]
- 训练全景：[[llm-train/README]]
- 模型架构基础（Encoder-Decoder / Attention）：[[llm-algo/transformer/模型架构]]
- 生成质量评测（BLEU/ROUGE）：[[llm-eval/README]]
