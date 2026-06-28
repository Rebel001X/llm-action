# Transformers 核心 API

> 一句话定位：HuggingFace `transformers` 的核心 API 是"用三类 Auto 对象（模型/分词器/配置）+ from_pretrained/save_pretrained 两个动词，把权重、词表、超参从磁盘/Hub 装进内存再吐回去"，再叠加 `generate`（采样解码）、`pipeline`（开箱即用封装）、`Trainer`（训练循环托管）四块上层能力。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/huggingface-transformers/README]] [[llm-inference/解码策略]]

## 阅读地图

| 节 | 主题 | 你将学到 | 关键词 |
|----|------|---------|--------|
| 0 | 一句话锚点 | 整个库的心智模型 | Auto / 三件套 |
| 1 | 地基/前置 | 模型=config+权重+tokenizer | checkpoint / 词表 |
| 2 | AutoConfig | 超参的"出生证" | 反序列化 |
| 3 | AutoTokenizer | 文本↔token id | BPE / 特殊符号 |
| 4 | AutoModel 家族 | 骨架 vs 带头 | *ForCausalLM |
| 5 | from_pretrained | 装载的全过程 | 解析→下载→实例化 |
| 6 | save_pretrained | 落盘的全过程 | safetensors |
| 7 | generate 参数语义 | 解码每个旋钮 | temperature / top-p |
| 8 | pipeline | 一行端到端 | 预处理+后处理 |
| 9 | Trainer 要点 | 训练循环托管 | TrainingArguments |
| 10 | 数值例子/对照/实践 | 显存/带宽手算 | KV cache |

---

## 0. 一句话锚点

记住这张"动词×名词"表，整个 API 就立起来了：

```
            from_pretrained()   save_pretrained()   __call__/generate
名词 ↓        (磁盘→内存)         (内存→磁盘)          (前向/解码)
─────────────────────────────────────────────────────────────────
AutoConfig    读 config.json      写 config.json       —
AutoTokenizer 读 tokenizer.*      写 tokenizer.*       text→ids / ids→text
AutoModel*    读 *.safetensors    写 *.safetensors     logits / 生成token
```

一句话：**"Auto" 不是某个模型，而是一个工厂/分发器**——它先偷看 `config.json` 里的 `model_type` 字段（比如 `"llama"`、`"qwen2"`），再去注册表里查到对应的具体类（`LlamaForCausalLM`），替你 new 出来。你不必记住几百个模型类名。

---

## 1. 地基/前置：一个"模型"到底由几块组成？

很多人以为"模型"就是一坨权重。实际上 HuggingFace 的一个 checkpoint 目录是**三类文件**的集合，缺一不可：

```
my-model/
├── config.json            ← 结构超参：层数、隐藏维度、词表大小、model_type...
├── model.safetensors      ← 权重张量（数字矩阵），可能分片 model-00001-of-00003.safetensors
├── tokenizer.json         ← 分词器：词表 + 合并规则（快速tokenizer）
├── tokenizer_config.json  ← 分词器超参：特殊token、padding方向、chat_template...
├── special_tokens_map.json← <bos>/<eos>/<pad> 等映射
└── generation_config.json ← 生成默认参数（可选）
```

**为什么要分三块？** 生命周期不同：**config** 决定"网络长什么样"（先有它才能 new 出正确形状的空壳去接权重）；**权重**是训练得到的具体数字（形状必须和 config 严丝合缝）；**tokenizer** 决定"文本怎么切成数字"（和模型绑死，换了就对不上）。

```
原始文本  ──tokenizer──▶  input_ids  ──model──▶  logits  ──采样/argmax──▶  下一个token id  ──decode──▶ 文本
"你好"                    [101, 872]              [词表大小维向量]
```

> 心智模型：**config 是图纸，权重是浇筑好的零件，tokenizer 是把文字翻译成整数的字典。三者通过 `from_pretrained` 从同一目录同时装入，保证彼此匹配。**

---

## 2. AutoConfig：超参的"出生证"

`AutoConfig` 负责把 `config.json` 反序列化成一个 Python 对象（`PretrainedConfig` 子类）。

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
config.num_hidden_layers      # 28      —— 多少个 Transformer block
config.hidden_size            # 3584    —— 每个token的向量维度 d_model
config.num_attention_heads    # 28      —— 注意力头数
config.vocab_size             # 152064  —— 词表大小
config.max_position_embeddings# 32768   —— 最长位置（上下文窗口）
```

```
config.json (文本)  ──AutoConfig.from_pretrained──▶  PretrainedConfig 对象
{                                                    config.hidden_size = 3584
  "model_type": "qwen2",         偷看这个字段 ───┐    config.num_hidden_layers = 28
  "hidden_size": 3584,                           │    ...
  ...                                            ▼
}                                          查注册表 → Qwen2Config 类
```

**为什么要单独有 config？** 三个高频用途：
1. **先看再下**：只下几 KB 的 config，就能判断模型大小、上下文长度，决定要不要下 14GB 权重。
2. **改结构再随机初始化**：`AutoModel.from_config(config)` 得到**随机权重**的同结构模型（不下预训练权重），用于从头训练。
3. **覆盖超参**：`from_pretrained(..., attn_implementation="flash_attention_2")` 这类 kwargs 会写进 config。

### RoPE 缩放：config 里最常被改的字段

RoPE（旋转位置编码）的"外推/插值"通过 config 的 `rope_scaling` 字典控制——这是把短上下文模型撑到长上下文的关键开关：

```python
config.rope_scaling = {"rope_type": "yarn", "factor": 4.0}
# factor=x 意为：让原本支持 L 长度的模型大致能处理 x*L 的序列
```

`rope_type` 取值语义（详细推导见 [[llm-inference/解码策略]] 相邻笔记）：

| rope_type | 做法 | 直觉 |
|-----------|------|------|
| `default` | 原始 RoPE | 不缩放 |
| `linear` | 位置索引整体除以 factor | 把刻度"压扁"，所有频率等比例插值 |
| `dynamic` (NTK) | 按当前长度动态调底数 | 短序列不损精度，长序列再插值 |
| `yarn` | 分频段处理（高频外推/低频插值） | 兼顾局部精度与长程外推，主流选择 |
| `longrope` / `llama3` | 各家定制曲线 | 见各模型论文 |

> 注意：改了 `factor` 通常也要相应更新 `max_position_embeddings`，否则模型自己不知道能跑更长（具体以官方文档为准）。

---

## 3. AutoTokenizer：文本 ↔ token id 的双向翻译

分词器把人类文本切成模型认识的整数 id，并补上特殊符号。

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

enc = tok("你好，世界", return_tensors="pt")
enc["input_ids"]       # tensor([[108386, 3837, 99489]])  —— 切成3个token
enc["attention_mask"]  # tensor([[1, 1, 1]])              —— 1=真实token,0=padding
tok.decode([108386, 3837, 99489])   # "你好，世界"
```

**编码内部三步：**

```
"你好世界"
   │ ① normalize  统一全角/半角、大小写、Unicode
   ▼
"你好世界"
   │ ② 分词 (BPE/WordPiece/Unigram)  按训练好的合并规则切子词
   ▼
["你好", "世", "界"]
   │ ③ 查词表 + 加特殊token
   ▼
[<bos>, 108386, 100, 101, <eos>]
```

**关键参数语义（讲含义不背默认值）：**

| 参数 | 含义 | 权衡 |
|------|------|------|
| `padding` | 补齐到同长（batch内对齐） | `True`/`"longest"` 补到批内最长；`"max_length"` 补到固定长。补多了浪费算力 |
| `truncation` | 超长截断 | 防止超过 `max_length` 报错；可能丢尾部信息 |
| `return_tensors` | 输出格式 | `"pt"`=PyTorch张量，`"np"`=numpy，`None`=python列表 |
| `add_special_tokens` | 是否加 `<bos>/<eos>` | 拼接已编码片段时常设 `False` |

**Fast vs Slow**：`use_fast=True`（默认尽量用）是 Rust 实现的 `tokenizer.json`，比纯 Python 的 slow 版快 1~2 个数量级；少数模型只有 slow 版。

**Chat template**：对话模型用 `tok.apply_chat_template(messages, ...)`，按 `tokenizer_config.json` 里的 Jinja 模板把 `[{"role":"user","content":...}]` 拼成模型期望的带角色标记字符串——**别手写拼接，不同模型格式不同**。

---

## 4. AutoModel 家族：骨架 vs 带头

最易混淆的点：`AutoModel` 和 `AutoModelForCausalLM` 不一样。区别在于**有没有"任务头"（task head）**。

```
                 ┌──────────────────────────────┐
input_ids ──────▶│  Transformer 主干 (encoder/   │──▶ last_hidden_state
                 │  decoder blocks) = "骨架"      │    [batch, seq, hidden]
                 └──────────────────────────────┘            │
                         ↑ AutoModel 到此为止                  │ 接不同的"头"
                                                              ▼
                              ┌───────────────────────────────────────────┐
                              │ ForCausalLM      → lm_head → vocab logits   │ (GPT/Llama, 文本生成)
                              │ ForSequenceClass → 分类头 → 类别 logits      │ (情感分类)
                              │ ForTokenClass    → 每token头 → NER 标签      │
                              │ ForQuestionAns   → span 头 → start/end       │
                              └───────────────────────────────────────────┘
```

| Auto 类 | 输出 | 典型用途 |
|---------|------|---------|
| `AutoModel` | 隐藏状态（无头） | 取 embedding、做特征 |
| `AutoModelForCausalLM` | 词表维 logits | GPT/Llama 文本生成 ← **LLM 最常用** |
| `AutoModelForSeq2SeqLM` | 词表维 logits | T5/BART 翻译摘要 |
| `AutoModelForSequenceClassification` | 类别 logits | 文本分类 |

```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    torch_dtype=torch.bfloat16,   # 半精度省一半显存
    device_map="auto",            # 自动把层切到多卡/CPU/磁盘
)
out = model(input_ids)            # 前向：得到 logits（不采样）
out.logits.shape                  # [batch, seq_len, vocab_size]
```

**`torch_dtype` 与 `device_map` 是装载阶段两个最值钱的旋钮**：
- `torch_dtype=torch.bfloat16` —— 权重以 bf16 载入，显存约为 fp32 的 1/2、fp16 同量但数值范围更稳。
- `device_map="auto"` —— 调 accelerate 做"按层切分"，单卡放不下时溢出到别的卡/CPU/磁盘（offload）。

---

## 5. from_pretrained：装载的全过程（重点拆解）

`from_pretrained` 是整个库使用频率最高的函数。它做的远不止"读文件"：

```
from_pretrained("Qwen/Qwen2-7B")
   │
   ① 解析标识符：是 Hub repo id 还是本地路径？
   │      "org/name" → 去 huggingface.co；"./path" → 读本地目录
   ▼
   ② 下载/定位文件（带缓存 ~/.cache/huggingface/hub）
   │      已缓存则跳过下载；只下缺的；支持断点续传
   ▼
   ③ 读 config → 决定实例化哪个具体类，按超参 new 出空壳模型
   │      此刻参数是"meta"占位，还没真正的数字
   ▼
   ④ 加载权重 safetensors，按名字对到每个张量（state_dict 映射）
   │      dtype 转换、缺失/多余键检查（warning）
   ▼
   ⑤ 放置到设备（device_map）、设为 eval 模式
   ▼
   返回 ready-to-use 的对象
```

**为什么要 safetensors？** 相比 PyTorch 原生 `.bin`（pickle 序列化）：
- **安全**：pickle 能在反序列化时执行任意代码（供应链攻击面）；safetensors 是纯张量+JSON 头，无代码执行。
- **快**：支持零拷贝 mmap，可直接映射到设备，大模型加载更快。

**分片（sharding）**：14GB 权重会切成多个 `model-0000x-of-0000n.safetensors`，外加一个 `model.safetensors.index.json` 记录"哪个张量在哪片"。好处：边下边载、断点续传、避免单文件过大。

**常用 kwargs：**

| kwargs | 作用 |
|--------|------|
| `torch_dtype` | 载入精度（bf16/fp16/auto） |
| `device_map` | 设备分布（"auto"/"cuda:0"/自定义dict） |
| `attn_implementation` | `"flash_attention_2"`/`"sdpa"`/`"eager"`，换更快的注意力核 |
| `revision` | 锁定 Hub 上的 commit/分支/tag（可复现） |
| `trust_remote_code` | 允许执行 repo 自带的建模代码（**有安全风险，需信任来源**） |
| `low_cpu_mem_usage` | 用 meta 设备避免装载时 CPU 内存翻倍 |

---

## 6. save_pretrained：落盘的全过程

`from_pretrained` 的逆操作。三件套各有自己的 `save_pretrained`，把内存对象写回磁盘目录，下次能原样读回。

```python
model.save_pretrained("./my-model")       # 写 config.json + *.safetensors
tok.save_pretrained("./my-model")         # 写 tokenizer.* 等
# 之后任何人都能：AutoModelForCausalLM.from_pretrained("./my-model")
```

```
内存对象                          ./my-model/ 目录
─────────                         ─────────────
model.config  ──to_json_file──▶   config.json
model 权重    ──save_file──────▶   model.safetensors (大则自动分片+index.json)
model.gen_cfg ───────────────▶    generation_config.json
tokenizer    ───────────────▶    tokenizer.json / tokenizer_config.json / ...
```

**实践要点：**
- `save_pretrained("dir")` 写的是**整个目录**，不是单文件——务必把 model 和 tokenizer **存到同一目录**，否则别人读不全。
- 也可单独写 config：`model.config.to_json_file("config.json")`（仅结构，不含权重）。
- `safe_serialization=True`（现默认）走 safetensors；设 `False` 退回 `.bin`。
- `push_to_hub=True` 可直接推到 HuggingFace Hub。
- 大模型可用 `max_shard_size="5GB"` 控制单片大小，便于分发与并行下载。

---

## 7. generate 参数语义：解码的每个旋钮

`model(...)` 只做**一步**前向得到 logits；`model.generate(...)` 替你做**自回归循环**：吐一个 token、拼回输入、再前向，直到遇到结束符或达到长度上限。**参数的本质是"如何从 logits 这个概率分布里挑下一个 token"**——这正是 [[llm-inference/解码策略]] 的核心。

```
input_ids ─▶ model ─▶ logits[最后位置] ─▶ 处理(温度/top-k/top-p) ─▶ 采样/argmax ─▶ next_id
    ▲                                                                              │
    └──────────────────────────── 把 next_id 拼到末尾 ◀───────────────────────────┘
              循环，直到 next_id==eos 或 长度达上限
```

**长度控制：**

| 参数 | 语义 |
|------|------|
| `max_new_tokens` | **只数新生成的** token 上限（推荐，不受输入长度干扰） |
| `max_length` | 输入+输出总长上限（易被长 prompt 误伤，少用） |
| `min_new_tokens` | 至少生成多少，防止过早 eos |

**采样开关与"温度"系列（核心）：**

| 参数 | 语义 | 调大的效果 |
|------|------|-----------|
| `do_sample` | `False`=贪心/束搜索（确定）；`True`=按概率随机采样 | 开启随机性 |
| `temperature` $T$ | logits 除以 $T$ 再 softmax：$p_i \propto \exp(z_i/T)$ | $T{>}1$ 分布变平→更发散；$T{<}1$ 变尖→更保守；$T{\to}0$≈贪心 |
| `top_k` | 只在概率最高的 $k$ 个候选里采样 | $k$ 越小越稳，越大越多样 |
| `top_p` (nucleus) | 累积概率刚超过 $p$ 的最小集合里采样 | $p$ 越小越保守；自适应候选数 |
| `repetition_penalty` | 对已出现 token 的 logit 打折，惩罚复读 | $>1$ 减少重复 |
| `no_repeat_ngram_size` | 禁止重复出现的 n-gram | 硬性防复读 |

温度直觉手算：logits $z=[2,1,0]$。
- $T=1$：softmax → $[0.665, 0.245, 0.090]$（差距明显）。
- $T=2$：除2后 $[1,0.5,0]$ → $[0.506, 0.307, 0.186]$（更平，低概率词更有机会）。
- $T=0.5$：乘2后 $[4,2,0]$ → $[0.867, 0.117, 0.016]$（更尖，几乎只挑第一个）。

**束搜索（beam search）：**

| 参数 | 语义 |
|------|------|
| `num_beams` | 同时保留 $b$ 条候选序列，最后选总分最高的 | 适合翻译/摘要等"求最优"；$b{=}1$ 即贪心 |
| `length_penalty` | 对长序列的偏好（>1 鼓励长，<1 鼓励短） |
| `early_stopping` | beam 全部到 eos 即停 |

**其它常用：**
- `eos_token_id` / `pad_token_id`：结束符与填充符（生成时缺 pad 会 warning）。
- `use_cache=True`：开 KV cache，把已算过的 K/V 缓存，避免每步重算前面所有 token——长文本生成的提速关键（见第 10 节算量）。
- `streamer=TextStreamer(tok)`：边生成边吐字到终端。
- `generation_config`：把上面这些打包成一个对象/`generation_config.json`，避免每次调用堆一长串 kwargs。

> 经验法则：**确定性输出**（代码、抽取）→ `do_sample=False`；**创意输出**（对话、写作）→ `do_sample=True, temperature≈0.7, top_p≈0.9`（具体甜点值因模型而异，以官方推荐为准）。

---

## 8. pipeline：一行端到端

`pipeline` 把"分词→前向→解码→后处理"全部封装，是最高层、最省心的 API。

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen2-7B", device_map="auto")
pipe("从前有座山，", max_new_tokens=50)
# [{'generated_text': '从前有座山，山里有座庙...'}]
```

```
原始输入  ──preprocess──▶  张量  ──forward──▶  logits  ──postprocess──▶  结构化结果
"图片/文本"   (tokenizer/                          (decode/argmax/        [{'label':..}]
              image_processor)                       softmax)
```

**它解决什么？** 新手不必懂 tokenizer 返回什么、logits 怎么变成标签——`pipeline` 替你把这套"样板代码"打包，按 task 字符串（`"text-generation"`/`"sentiment-analysis"`/`"automatic-speech-recognition"`...）自动选对的预/后处理。

**与底层 API 的取舍：**

| 维度 | pipeline | Auto* 手动 |
|------|----------|-----------|
| 上手 | 一行，最快 | 需写预处理/解码 |
| 控制 | 弱（隐藏细节） | 强（每步可改） |
| 批处理/部署 | 适合原型、小批 | 适合自定义训练/高吞吐服务 |

实践：原型验证用 pipeline；要精调 generate 参数、要自己管 batching/缓存，就降到 Auto* + generate。生产高并发推理一般另上 vLLM/TGI（见 README 与 [[llm-inference/解码策略]]）。

---

## 9. Trainer 要点（讲语义不背签名）

`Trainer` 把"训练循环"——前向、算 loss、反传、优化器 step、学习率调度、梯度累积、混合精度、保存 checkpoint、评估、日志——这些每次都要重写的样板，托管成一个对象。你只需提供：**模型 + 数据 + 一份超参（`TrainingArguments`）**。

```
       ┌─────────────────────── Trainer.train() 循环 ───────────────────────┐
       │  取一个 batch                                                        │
       │     ▼                                                                │
       │  前向 → 算 loss  ── (loss/累积步数) → backward 累积梯度               │
       │     ▼                                                                │
       │  达到累积步数？ ──否──▶ 继续取 batch                                  │
       │     │是                                                              │
       │  梯度裁剪 → optimizer.step → scheduler.step → 清零梯度               │
       │     ▼                                                                │
       │  到 logging/eval/save 间隔？ → 记日志 / 跑评估 / 存 checkpoint        │
       └────────────────────────────────────────────────────────────────────┘
```

```python
from transformers import Trainer, TrainingArguments
args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # 等效 batch=4*8*卡数
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    eval_strategy="steps", eval_steps=500,
    save_strategy="steps", save_steps=500,
    logging_steps=50,
)
trainer = Trainer(model=model, args=args,
                  train_dataset=ds_train, eval_dataset=ds_eval,
                  data_collator=collator, tokenizer=tok)
trainer.train()
```

**TrainingArguments 关键旋钮的"含义与权衡"（不背默认值）：**

| 参数 | 含义 | 权衡 |
|------|------|------|
| `per_device_train_batch_size` | **每张卡**的 batch | 越大越稳但吃显存；爆显存就调小 |
| `gradient_accumulation_steps` | 累积 N 个小 batch 再更新一次 | 用"时间换显存"模拟大 batch，等效 batch = 微批×累积×卡数 |
| `learning_rate` | 学习率 | 太大发散、太小学不动；微调常用小学习率 |
| `lr_scheduler_type` + `warmup_steps` | 学习率曲线 + 预热 | 预热避免开局梯度爆炸 |
| `bf16`/`fp16` | 混合精度 | 省显存+提速；bf16 范围更稳（A100/H100 优先） |
| `gradient_checkpointing` | 反传时重算激活而非全存 | 大幅省激活显存，换约 20~30% 额外算力（约，视模型） |
| `eval/save/logging_strategy` | 评估/保存/日志的触发节奏 | 越频繁越安全但越慢 |
| `deepspeed`/`fsdp` | 接分布式并行后端 | 训大模型时用，分摊优化器状态/参数显存 |

**核心语义点（面试常问）：**
1. **想自定义 loss**？继承 `Trainer` 重写 `compute_loss`，而不是改循环。
2. **等效大 batch** = `per_device_batch × grad_accum × 卡数`——三者乘积才是真正影响收敛的"全局批量"。
3. **Trainer vs 手写循环**：Trainer 省样板、内置分布式/混合精度/断点续训；但隐藏细节，特殊训练范式（如 RLHF/自定义采样）常需 `accelerate`/`trl` 或手写。
4. `SFTTrainer`（在 `trl` 库）是 Trainer 的指令微调特化版，自带打包/模板处理。

---

## 10. 数值例子 / 对照 / 实践

### 例 1：7B 模型推理显存怎么估？

参数 70 亿，权重精度 bf16（2 字节/参数）：

$$
\text{权重显存} \approx 7\times10^9 \times 2\,\text{B} = 1.4\times10^{10}\,\text{B} \approx 14\,\text{GB}
$$

所以 7B 模型 bf16 大约吃 **14GB** 显存（仅权重）；fp32 翻倍 ~28GB；int4 量化约 1/4 ~3.5GB。再加 KV cache 和激活，留 1.2~1.5 倍余量。

### 例 2：KV cache 为什么省算力？

生成第 $t$ 个 token，注意力要用到前面所有 token 的 K、V。**不缓存**：每步重算前 $t$ 个的 K/V，总注意力计算随序列呈 $O(L^2)$。**缓存**（`use_cache=True`）：每步只算**当前 1 个** token 的 K/V，追加进缓存，降为 $O(L)$ 步×常量。

KV cache 显存（单序列）：

$$
\text{KV} = 2 \times L \times n_{layer} \times n_{kv\_head} \times d_{head} \times \text{dtype}
$$

代入 7B 类模型（$L{=}4096, n_{layer}{=}28, n_{kv}\,d_{head}{=}$ 等效 hidden≈3584, bf16）：约
$2\times4096\times28\times3584\times2\,\text{B} \approx 1.6\,\text{GB}$（约，按 MHA 估；GQA 会显著更小）。

> 这解释了为什么"长上下文 + 大并发"先撑爆的往往是 KV cache 而非权重——也是 vLLM PagedAttention 要解决的问题。

### 例 3：梯度累积模拟大 batch

显存只够 `per_device_train_batch_size=2`，想要等效 batch=32（单卡）：$\text{grad\_accum}=\frac{32}{2\times1}=16$。设 `gradient_accumulation_steps=16`，每 16 个微批更新一次参数，数学上≈一次性 batch=32（BN 等少数层除外）。

### 三类 API 选型对照

| 场景 | 推荐 |
|------|------|
| 快速试个想法 | `pipeline` |
| 自定义解码/批处理/缓存 | `Auto* + generate` |
| 训练/微调 | `Trainer` + `TrainingArguments`（指令微调可上 `trl` 的 `SFTTrainer`） |
| 高并发线上推理 | 导出后上 vLLM / TGI（见 [[ai-framework/huggingface-transformers/README]]） |

---

## 常见问题

| 问题 | 答案 |
|------|------|
| `AutoModel` 和 `AutoModelForCausalLM` 啥区别？ | 前者只到主干输出隐藏状态；后者多了 `lm_head`，输出词表 logits，能 `generate` |
| 为什么读了 model 还要单独读 tokenizer？ | 二者是独立对象，分别 `from_pretrained`；务必同源同目录，否则 id 对不上 |
| `model()` 和 `model.generate()` 区别？ | 前者只前向一步出 logits；后者是自回归循环，替你逐 token 解码 |
| `max_length` vs `max_new_tokens`？ | 前者算输入+输出总长（易被长 prompt 误伤）；后者只数新增，推荐用它 |
| 输出每次不一样/太发散？ | 检查 `do_sample`、`temperature`、`top_p`；要确定性就 `do_sample=False` |
| `safetensors` 比 `.bin` 好在哪？ | 不执行代码（安全）、零拷贝 mmap（快），现为默认 |
| `device_map="auto"` 做了啥？ | 调 accelerate 按层把模型切到多卡/CPU/磁盘，单卡放不下时自动溢出 |
| `trust_remote_code=True` 安全吗？ | 会执行 repo 自带建模代码，仅对信任来源开启 |
| 等效 batch 怎么算？ | 微批 × `gradient_accumulation_steps` × 卡数 |
| 想改 loss 要重写训练循环吗？ | 不用，继承 `Trainer` 重写 `compute_loss` 即可 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引
- [[ai-framework/huggingface-transformers/README]] — 本目录总览（安装、生态、与推理框架的衔接）
- [[llm-inference/解码策略]] — generate 背后的贪心/束搜索/温度/top-k/top-p/RoPE 外推详解
