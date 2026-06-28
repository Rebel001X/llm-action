# HuggingFace Transformers

> 用一套统一的 `from_pretrained` / `AutoModel` / `Trainer` 接口，把成千上万种模型变成"换个名字字符串就能跑"的乐高积木——它是当今 NLP/大模型的事实标准框架。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/huggingface-peft/README]] [[ai-framework/pytorch/README]]

## 阅读地图

| 节 | 你会学到 | 为什么重要 |
| --- | --- | --- |
| 0 一句话锚点 | 它到底是什么 | 先建立心智模型 |
| 1 地基/前置 | 没有它之前世界有多痛 | 理解"解决什么问题" |
| 2 统一接口 Auto* | `AutoModel`/`AutoTokenizer` 怎么"自动" | 框架的灵魂 |
| 3 from_pretrained | 一行下载/加载/初始化权重 | 最常用的入口 |
| 4 Tokenizer | 文本 ↔ 数字怎么转 | 模型只懂数字 |
| 5 generate | 自回归生成的全套解码策略 | 推理核心 |
| 6 Trainer | 不写训练循环也能训 | 训练核心 |
| 7 量化 | 4bit/8bit 怎么省显存 | 落地必备 |
| 8 生态 | datasets/peft/accelerate/Hub | 为什么是"事实标准" |
| 数值例子 | 显存/吞吐手算 | 把抽象变具体 |
| 对照表 | 与 PyTorch 裸写 / vLLM 对比 | 何时选它 |

---

## 0. 一句话锚点

HuggingFace Transformers 是一个 **Python 库**，它把"加载模型、预处理文本、跑前向、生成文本、训练/微调"这些事，统一成 **同一套 API**。

它的最核心承诺是：**只要你知道模型在 Hub 上的名字（一个字符串），你就能用完全相同的几行代码把它跑起来**，无论它是 BERT、GPT-2、LLaMA、Qwen 还是 Whisper。

```
        "bert-base-chinese"  ─┐
        "Qwen/Qwen2-7B"      ─┤      AutoModel.from_pretrained(name)
        "openai/whisper"     ─┼──►   AutoTokenizer.from_pretrained(name)   ──►  跑起来
        "你自己微调的模型"     ─┘                  ▲
                                          同一套 API，零差别
```

---

## 1. 地基/前置：它解决什么问题

### 1.1 没有它之前的世界

在 Transformers 之前，每个团队发布模型时代码风格五花八门：论文 A 用自定义 tokenizer、权重 `.bin`；论文 B 另一套 tokenizer、权重 `.ckpt`、输入格式完全不同；论文 C 只有一个依赖删库 repo 的 colab。想对比 3 个模型，要读 3 套代码、装 3 套环境、调 3 套预处理。

问题的本质是：**模型 = 架构代码 + 权重 + 分词规则 + 预处理约定**，这四样东西散落各处、互不兼容。

### 1.2 Transformers 的解法

它做了三件根本性的事，把上面四样东西"打包并标准化"：

1. **统一架构实现**：用一套约定（`PreTrainedModel` 基类）重新实现了几百种模型架构，接口一致。
2. **统一权重分发**：通过 HuggingFace **Hub**（一个 Git LFS 仓库网站）托管权重，每个模型有唯一名字。
3. **统一加载入口**：`from_pretrained(名字)` 一行，自动下载 + 加载架构 + 灌入权重 + 拿到配套 tokenizer。

```
        ┌───────────────── 一个模型仓库（Hub 上） ─────────────────┐
        │  config.json     ← 架构超参（层数/隐藏维度/词表大小）        │
        │  model.safetensors ← 权重（几百 MB ~ 几百 GB）            │
        │  tokenizer.json  ← 分词规则 + 词表                        │
        │  generation_config.json ← 默认生成参数                    │
        └──────────────────────────────────────────────────────────┘
                              │  from_pretrained("仓库名")
                              ▼
                    架构 + 权重 + 分词器  全部就位
```

> 一句话：**它把"模型"从一堆零散文件，变成一个可以用名字寻址、即插即用的标准件。**

---

## 2. 统一模型接口：`Auto*` 类的魔法

### 2.1 为什么需要 "Auto"

没有 Auto 类时，你必须**事先知道**模型架构再导入对应的类（如 `BertModel.from_pretrained("bert-base-chinese")`，记错类名就报错）。`Auto*` 类的作用是：**读取仓库 `config.json` 里的 `model_type` 字段，自动帮你选对那个具体的类**。

```
   AutoModel.from_pretrained("某模型")
            │
            ▼
   下载 config.json  ──►  读到 "model_type": "qwen2"
            │
            ▼
   内部映射表查表：qwen2 → Qwen2Model 这个类
            │
            ▼
   等价于你手写  Qwen2Model.from_pretrained(...)，但你不需要知道这个类名
```

### 2.2 Auto 家族成员（按"任务头"区分）

模型主干（backbone）输出的是隐藏向量，要做具体任务还需要在上面接一个"头"（head）。Auto 家族按头的不同分成多个类：

| Auto 类 | 输出 | 典型用途 |
| --- | --- | --- |
| `AutoModel` | 隐藏状态（向量） | 拿 embedding、做特征提取 |
| `AutoModelForCausalLM` | 词表上的下一个词概率 | GPT/LLaMA 类生成 |
| `AutoModelForSeq2SeqLM` | 编码-解码生成 | T5、翻译、摘要 |
| `AutoModelForSequenceClassification` | 类别 logits | 情感分类、文本分类 |
| `AutoModelForTokenClassification` | 每个 token 的标签 | 命名实体识别 |
| `AutoModelForQuestionAnswering` | 答案起止位置 | 抽取式问答 |
| `AutoTokenizer` | 分词器 | 文本 ↔ id |
| `AutoConfig` | 配置对象 | 改架构超参 |
| `AutoProcessor` | 多模态预处理 | 图像/音频+文本 |

```
                ┌──────────────────────────┐
   输入 ids ──► │   Transformer 主干         │ ──► 隐藏向量 h (每个 token 一个)
                └──────────────────────────┘
                              │
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
   [分类头 Linear→C类]  [LM头 Linear→词表]  [NER头 Linear→标签]
   AutoModelFor         AutoModelFor        AutoModelFor
   SequenceClassif      CausalLM            TokenClassif
```

> 关键直觉：**主干是共享的，"Auto...For..." 只是告诉框架在主干上面接哪种头。** 选错头不会用错主干，只会用错最后一层。

---

## 3. `from_pretrained`：一行背后发生了什么

这是整个库最高频的一行。拆开看它做了 6 件事：

```
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B", torch_dtype=torch.bfloat16, device_map="auto")

  1. 解析名字 ──► 去本地缓存 ~/.cache/huggingface 找；没有就去 Hub 下载
  2. 读 config.json ──► 知道 model_type=qwen2，层数/维度等
  3. 按 config 构造一个"空壳"模型（随机初始化的权重骨架）
  4. 读 model.safetensors ──► 把磁盘上的权重张量灌进空壳
  5. 按 torch_dtype 转精度（如 bf16），按 device_map 分配到 GPU/CPU
  6. 返回 ready-to-use 的 model 对象
```

### 3.1 几个高频参数的"为什么"

| 参数 | 作用 | 为什么要它 |
| --- | --- | --- |
| `torch_dtype=torch.bfloat16` | 用 16 位加载 | fp32 权重显存翻倍，大模型放不下 |
| `device_map="auto"` | 自动把层分到多卡/CPU | 单卡放不下时按显存切分 |
| `low_cpu_mem_usage=True` | 边下边灌，不开两份 | 避免加载瞬间 CPU 内存 2× |
| `trust_remote_code=True` | 允许跑仓库自带的自定义建模代码 | 新架构未进库时需要；**有安全风险，仅对可信仓库开** |
| `revision="main"` | 锁定某个 commit/分支 | 复现实验，防权重被偷换 |

> 提示：具体参数名/默认值可能随版本变化，**以官方文档为准**；这里给的是稳定的"为什么"。

---

## 4. Tokenizer：文本 ↔ 数字的桥

模型只会做矩阵乘法，不认识汉字。Tokenizer 负责把字符串切成 **token**、再映射成整数 **id**；生成完再把 id 映射回字符串。

```
   "我爱AI"
      │  encode（分词 + 查词表）
      ▼
   tokens:  ["我", "爱", "AI"]
      │
      ▼
   ids:     [1037, 4521, 9912]
      │  ──► 喂给模型 ──► 模型输出 ids ──┐
      ▼                                  │ decode（查词表反向）
   "我爱AI ... 是未来"  ◄────────────────┘
```

### 4.1 子词分词：为什么不按字/按词

- 按词：词表爆炸（几十万），遇到没见过的词（OOV）就废了。
- 按字符：序列太长，丢失语义单元。
- **折中：子词（subword）**。常见词整体一个 token，罕见词拆片段，如 `unhappiness → un + happiness`。

主流算法：**BPE**（GPT 系）、**WordPiece**（BERT 系）、**Unigram/SentencePiece**（多语言/T5 系），都在"词表大小"和"序列长度"间找平衡。

### 4.2 调用时的关键产物

```python
enc = tokenizer("我爱AI", return_tensors="pt")
# enc["input_ids"]      ← token id 张量（喂给模型）
# enc["attention_mask"] ← 1=真 token，0=padding（告诉模型哪些是凑数的）
```

```
  批内对齐（padding）：
   句子1: [我][爱][AI]            → [我][爱][AI][PAD][PAD]   mask: 1 1 1 0 0
   句子2: [今][天][天][气][好]     → [今][天][天][气][好]      mask: 1 1 1 1 1
                                     └── 对齐成同长，方便堆成一个矩阵 ──┘
```

> `attention_mask` 的意义：注意力计算时把 PAD 位置打成 $-\infty$，softmax 后权重为 0，等于"看不见凑数的位置"。

聊天模型还有 **chat template**：把多轮对话按模型训练时的格式（角色标记、特殊 token）拼成一个字符串，避免你手拼格式拼错。

---

## 5. `generate`：自回归生成的全套解码

生成是**一次只产一个 token，把它接回输入，再产下一个**，循环到结束。

```
  输入: "今天天气"
   step1: 模型看 "今天天气"     → 预测下一个最可能是 "真"  → 接上
   step2: 模型看 "今天天气真"   → 预测 "好"               → 接上
   step3: 模型看 "今天天气真好" → 预测 <eos>（结束符）     → 停
   输出: "今天天气真好"
```

### 5.1 每一步：从 logits 到一个 token

模型每步输出词表上每个词的分数 logits，要从中"挑一个"。不同挑法 = 不同解码策略：

```
   logits (词表上每个词一个分数)
        │  /T  温度缩放：T<1 更尖锐(确定)，T>1 更平(随机)
        ▼
   过滤：top-k（只留分最高的 k 个） 或 top-p（累计概率到 p 的最小集合）
        ▼
   softmax → 概率分布
        ▼
   采样 or 取最大 → 选出下一个 token
```

### 5.2 解码策略对照

| 策略 | 怎么选 | 特点 | 适合 |
| --- | --- | --- | --- |
| greedy | 每步取概率最大 | 确定、可能重复枯燥 | 抽取式、要可复现 |
| beam search | 同时保留 b 条候选路径 | 全局更优、慢、易刻板 | 翻译、摘要 |
| top-k 采样 | 在前 k 个里随机 | 有创造性 | 开放写作 |
| top-p (nucleus) | 在累计概率 p 内随机 | 自适应候选数 | 通用对话 |
| temperature | 调随机程度 | 与上面叠加用 | 控发散度 |

```python
out = model.generate(**enc, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.7)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 5.3 KV Cache：为什么生成不会越来越慢到爆

朴素做法第 $n$ 步要重算前 $n$ 个 token 的注意力（$O(n^2)$ 累加）。**KV Cache** 把算过的 Key/Value 缓存起来，每步只算新 token（无 cache 总计算 ∝ Σ n² 爆炸；有 cache 每步 ∝ n 线性增长）。代价是显存：缓存大小 ≈ $2 \times n_{layers} \times n \times d_{model} \times \text{精度字节}$（$n$=序列长）。这是长上下文推理显存暴涨的主因——详见数值例子。

---

## 6. `Trainer`：不手写训练循环也能训

裸 PyTorch 训练要自己写前向、loss、反向、优化器 step、梯度清零、学习率调度、混合精度、多卡、保存、日志……几十行模板且容易出 bug。`Trainer` 把这套**封装成一个对象**。

```
   你提供：模型 + 数据集 + TrainingArguments(超参) + (可选)评估函数
                          │
                          ▼
   ┌─────────────────── Trainer 内部循环 ───────────────────┐
   │  for epoch:                                            │
   │    for batch in dataloader:                            │
   │       outputs = model(**batch)        # 前向            │
   │       loss = outputs.loss             # 自动取 loss     │
   │       loss.backward()                 # 反向            │
   │       optimizer.step(); scheduler.step()               │
   │       optimizer.zero_grad()                            │
   │       (混合精度 / 梯度累积 / 多卡 由 accelerate 处理)     │
   │    evaluate(); save_checkpoint()                       │
   └────────────────────────────────────────────────────────┘
```

### 6.1 三个核心输入

| 输入 | 是什么 | 关键字段（举例，以文档为准） |
| --- | --- | --- |
| `model` | 待训练模型 | 任意 `PreTrainedModel` |
| `TrainingArguments` | 所有超参的容器 | 学习率、batch、epoch、`fp16/bf16`、`gradient_accumulation_steps`、保存策略 |
| 数据集 | 已 tokenized 的 Dataset | 通常来自 `datasets` 库 |

### 6.2 它顺手解决的工程难题

- **混合精度**：一个开关 `bf16=True`，省一半显存、提速。
- **梯度累积**：显存放不下大 batch？`gradient_accumulation_steps=8` 用小步攒成大步（见数值例子）。
- **分布式**：底层用 `accelerate`，同一份代码单卡/多卡/多机不改。
- **断点续训、日志、评估、early stopping**：内置或回调（Callback）扩展。

> 想做参数高效微调（LoRA 等）？`Trainer` 与 [[ai-framework/huggingface-peft/README]] 无缝配合——把模型包成 PEFT 模型再交给 Trainer 即可。

---

## 7. 量化：用更少的比特装下大模型（沿用原文方向）

Transformers 原生集成了多种量化后端（如 bitsandbytes、auto-gptq、optimum 等），**加载时一个配置就能把权重压到 8bit/4bit**，显著降显存。

- 量化文档（以官方为准）：https://huggingface.co/docs/transformers/main_classes/quantization
- 更多量化/导出方案 optimum：https://github.com/huggingface/optimum

### 7.1 GPTQ 量化（训练后、基于校准数据）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
quantization_config = GPTQConfig(bits=4, group_size=128, dataset="c4", desc_act=False)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto")
```

### 7.2 LLM.int8() / 4bit —— bitsandbytes

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloomz-7b1-mt"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True)

# 4bit + 双量化（QLoRA 常用）
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=cfg)
```

权重精度 vs 每参字节：fp32=4B（基准）→ fp16/bf16=2B（一半）→ int8=1B（1/4）→ int4=0.5B（1/8，配 LoRA 即 QLoRA，可单卡微调 7B）。

> 量化是"用一点精度换大量显存"。推理通常 4bit 仍可用；训练则常配 LoRA（冻结量化主干，只训小适配器）。

---

## 8. 生态：为什么它是事实标准

单看库本身不足以解释它的统治力，关键在它周边形成了**自洽的工具链**，每一环都用同样的"名字即资源"哲学：

```
                      ┌──────────────── HuggingFace Hub ────────────────┐
                      │   模型权重 · 数据集 · Spaces 演示（Git LFS 托管）   │
                      └───────────────────────┬─────────────────────────┘
                                              │ from_pretrained / load_dataset
        ┌──────────────┬──────────────────────┼────────────────┬───────────────┐
        ▼              ▼                       ▼                ▼               ▼
  transformers     datasets               accelerate          peft          tokenizers
  (模型/训练/推理)  (一行加载数据集)        (单卡↔多机零改代码)  (LoRA等微调)   (Rust 极速分词)
        │                                                                       
        └──────────► optimum / TGI / safetensors（导出、部署、安全权重格式）
```

| 组件 | 一句话 | 解决什么 |
| --- | --- | --- |
| **Hub** | 模型/数据集的 GitHub | 统一分发与版本控制 |
| **datasets** | `load_dataset("名字")` | 标准化、可流式的数据加载 |
| **accelerate** | 设备/分布式抽象层 | 同代码跑单卡/多卡/多机 |
| **peft** | 高效微调 | LoRA/QLoRA，省显存 → [[ai-framework/huggingface-peft/README]] |
| **safetensors** | 安全的权重格式 | 替代可执行任意代码的 pickle |
| **optimum / TGI** | 导出与服务化 | ONNX/TensorRT、推理服务 |

**为什么成为事实标准（网络效应飞轮）**：更多模型上传 Hub → 用户用同一套 API 即可试用 → 体验好、迁移成本低 → 用户越来越多 → 作者想被人用就首发到 HF → 更多模型上传……自我强化。

> 注意：训练/微调它是王者，但**高吞吐在线推理**往往会换成 vLLM/TGI/SGLang（PagedAttention、连续批处理）。"事实标准"指的是**研究与训练入口**，不代表它在生产推理上一定最快。

---

## 数值例子：把抽象变具体

### 例 1：加载一个 7B 模型要多少显存？

参数量 $N = 7\times10^9$，**仅权重**显存 $= N \times \text{字节/参数}$：fp32→$28$ GB、bf16→$14$ GB、int8→$7$ GB、int4→$3.5$ GB。

结论：24GB 的消费级卡（如 4090）用 **bf16 勉强（14GB 权重 + 激活/KV cache）**，用 **4bit 则很宽裕**——这正是 4bit 加载受欢迎的原因。

### 例 2：KV Cache 在长上下文下吃多少显存？

设 32 层、隐藏维 4096、bf16（2 字节），上下文 $n=8192$ token，batch=1。每 token 每层缓存 K 和 V 各一份：

$$\text{KV} = 2 \times n_{layers} \times n \times d_{model} \times 2\text{字节}$$
$$= 2 \times 32 \times 8192 \times 4096 \times 2 \approx 8.6\ \text{GB}$$

也就是说，**光是 KV cache 就能吃掉 8GB+**——这解释了为什么长上下文推理常常显存先爆，以及为什么 PagedAttention 这类显存管理技术重要。

### 例 3：梯度累积凑出大 batch

显存只够 `per_device_batch=2`，但想要等效 batch=32：

$$\text{有效 batch} = \text{per\_device} \times \text{grad\_accum} \times \text{卡数}$$
$$32 = 2 \times 8 \times 2 \quad(\text{2 卡，累积 8 步})$$

即 `gradient_accumulation_steps=8`、2 张卡，就在不增加峰值显存的前提下，拿到和 batch=32 一样的更新效果（只是慢了 8×）。

---

## 对照表：何时选它

| 维度 | Transformers + Trainer | 裸 PyTorch [[ai-framework/pytorch/README]] | vLLM / TGI（推理引擎） |
| --- | --- | --- | --- |
| 定位 | 模型即插即用 + 训练/微调 | 自由造轮子 | 高吞吐在线推理 |
| 上手成本 | 低（几行跑通） | 高（全要自己写） | 中（部署为主） |
| 模型覆盖 | 极广（Hub 数十万） | 看你自己实现 | 主流生成模型 |
| 训练/微调 | ✅ 强（+peft/accelerate） | ✅ 但全手写 | ✗（不做训练） |
| 推理吞吐 | 一般（够用） | 看实现 | ✅ 极高（PagedAttn/连续批） |
| 何时选 | 研究、微调、原型、教学 | 要改架构/底层创新 | 生产高并发服务 |

> 决策直觉：**做研究/微调/快速验证 → Transformers；要发明新算子/新架构 → 退到裸 PyTorch；要把模型变成高并发 API → 上 vLLM/TGI。** 三者常组合：用 Transformers 微调出权重，用 vLLM 部署上线。

---

## 常见问题

| 问题 | 解答 |
| --- | --- |
| `AutoModel` 和 `AutoModelForCausalLM` 啥区别？ | 前者只给主干隐藏向量（拿 embedding），后者多了 LM 头能预测下一个词（能 generate）。 |
| 为什么我 `generate` 出来全是重复？ | 多半用了 greedy；试 `do_sample=True` + `top_p`/`temperature`，或加 `repetition_penalty`。 |
| 模型下载太慢/被墙？ | 设镜像端点（如 `HF_ENDPOINT` 环境变量），或离线下载后用本地路径传给 `from_pretrained`。 |
| `trust_remote_code=True` 安全吗？ | 它会执行仓库里的 Python 建模代码，**只对可信来源开**，否则有任意代码执行风险。 |
| 显存不够加载大模型？ | bf16 加载、`device_map="auto"` 切多卡/卸载到 CPU、或 8bit/4bit 量化。 |
| Trainer 太黑盒想自定义？ | 用 Callback 钩子，或重写 `compute_loss`/`training_step`，再不行就退回 accelerate 手写循环。 |
| pad token 报错？ | 很多生成模型没设 `pad_token`，手动 `tokenizer.pad_token = tokenizer.eos_token`。 |
| 版本/参数名对不上？ | API 随版本演进，**以官方文档为准**；本文讲的是稳定机制而非精确签名。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，从这里找其它主题
- [[ai-framework/huggingface-peft/README]] — PEFT/LoRA：在本框架上做参数高效微调
- [[ai-framework/pytorch/README]] — 底层 PyTorch：Transformers 的地基与"退路"
- 官方文档（以其为准）：https://huggingface.co/docs/transformers
- 量化文档：https://huggingface.co/docs/transformers/main_classes/quantization
- optimum（导出/部署/更多量化）：https://github.com/huggingface/optimum
