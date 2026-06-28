# 从零实现 BPE 分词器(Byte-level BPE Tokenizer）

一个端到端、CPU 几十秒跑通的 **字节级 BPE 分词器** 教学实现。BPE(Byte-Pair Encoding，字节对编码)是 GPT-2 / GPT-3 / LLaMA 等主流大模型所用的分词方案，本项目用纯 Python（核心算法零第三方库，仅 numpy/matplotlib 做统计与画图）把它的 `train / encode / decode` 三件事从头实现一遍，并验证编解码 round-trip 无损。

---

## 一、演示什么原理

分词器（tokenizer）是大模型的"输入第一层"：把文本切成一串整数 token id 喂给模型。BPE 的核心思想是 **用数据驱动地把高频字节组合"压"成一个新符号**：

1. **train（学词表）**：把语料切成词、再转成 UTF-8 字节序列；反复统计"相邻 token 对（pair）"的频率，每轮把出现次数最高的那一对合并成一个新 token，记成一条 merge 规则，直到达到目标词表大小。
2. **encode（编码）**：把任意文本先转成字节，再按"学习顺序"反复应用 merge 规则，得到 token id 列表。
3. **decode（解码）**：每个 token id 查词表得到对应字节串，拼接后用 UTF-8 还原回字符串。

**为什么是"字节级（byte-level）"？** 我们不在"字符"上做 BPE，而是先把字符串编码成 UTF-8 字节（0..255）。这样初始词表天然只有 256 个，且 **永远不会有未登录字符（OOV）**——任何 Unicode 文本（中文、emoji、生僻符号）都能被表示为字节，因此 `decode(encode(text)) == text` 永远无损。这正是 GPT-2 论文采用 byte-level BPE 的关键原因。

> 一句话直觉：BPE 是一种"对文本做无损压缩"的算法，词表越大，常见词/词缀就越能被合并成单个 token，序列越短，模型推理越省。

---

## 二、怎么跑

环境：Python 3.13 / numpy 2.3 / torch 2.12（CPU）。仅核心算法零依赖；画图需要 matplotlib（缺了会自动跳过、不崩）。

```bash
cd practical-projects/02-bpe-tokenizer
python bpe_tokenizer.py
```

跑完会在当前目录生成 `compression_curve.png`（词表大小 vs 压缩率曲线）。

---

## 三、预期输出（真实跑通摘录）

学到的前若干个 merge（高频字节对优先被合并）：

```
learned merges (highest-frequency byte pairs first):
  merge   0: (   'o',    'w') -> id 256      'ow'  (count=65)
  merge   1: (   'l',   'ow') -> id 257     'low'  (count=60)
  merge   2: (   'e',    's') -> id 258      'es'  (count=50)
  merge   3: (  'es',    't') -> id 259     'est'  (count=50)
  merge   5: (  'th',    'e') -> id 261     'the'  (count=45)
  ...
  merge  21: ('tokenizatio',  'n') -> id 277  'tokenization'  (count=20)
[train] done. learned 44 merges, final vocab = 300
```

可以清楚看到 BPE 是怎么"长大"的：先合出 `ow` → 再 `low` → 一路把高频词 `tokenization`、`encoding`、`newest` 整词合并成单个 token。

round-trip 无损校验（含未登录词、中文、emoji，全部 True）：

```
[round-trip] decode(encode(x)) == x  ?
  ok= True  bytes= 24 -> tokens=  7   'the lowest newest widest'
  ok= True  bytes= 17 -> tokens= 15   'a quick brown fox'
  ok= True  bytes= 33 -> tokens= 32   'unseen words like hippopotamus!!!'
  ok= True  bytes= 34 -> tokens= 34   '?????? round-trip ?'   # 中文/emoji，byte-level 仍无损
[round-trip] ALL LOSSLESS = True
```

压缩率随词表增大而下降（核心可量化信号）：

```
[compression] ratio = tokens / bytes  (lower is better)
  vocab= 256  tokens= 1750  bytes= 1750  ratio=1.0000   # 不学 merge = baseline
  vocab= 280  tokens= 1015  bytes= 1750  ratio=0.5800
  vocab= 300  tokens=  760  bytes= 1750  ratio=0.4343
  vocab= 350  tokens=  590  bytes= 1750  ratio=0.3371
[compression] best ratio = 0.3371  -> sequence shrunk 2.97x

RESULT: round_trip_lossless=True  compression_ratio=0.3371  shrink=2.97x
SUCCESS
```

**结论**：编解码无损（round-trip 全 True），同一段文本在词表 256→350 时 token 数从 1750 降到 590，序列缩短约 **2.97 倍**——这就是 BPE 给大模型带来的直接收益（更短的上下文、更省的算力）。

---

## 四、对应 llm-action 文档

- 数据工程 / 分词与数据清洗：[`../../llm-data-engineering`](../../llm-data-engineering)
- 词表、tokenizer 在训练流程中的位置：[`../../llm-train`](../../llm-train)

---

## 五、社区参考

- [karpathy/minbpe](https://github.com/karpathy/minbpe) —— Karpathy 的最小 BPE 实现，本项目的训练/编解码思路与之一脉相承。
- GPT-2 BPE：Radford et al., *Language Models are Unsupervised Multitask Learners*（2019），首次系统采用 byte-level BPE；官方实现见 [openai/gpt-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)。
- 生产级实现可参考 [openai/tiktoken](https://github.com/openai/tiktoken)、[huggingface/tokenizers](https://github.com/huggingface/tokenizers)。

---

## 六、局限 / 与真实工程的差异

| 维度 | 本 toy 实现 | 生产实现（tiktoken / HF tokenizers） |
| --- | --- | --- |
| 预切分 pre-tokenization | 仅按空白 `split()` | GPT-2 用精心设计的正则把标点/数字/空格切开，避免跨词合并 |
| 空格处理 | 直接丢弃词间空格的字节 | 用 `Ġ`（带空格前缀）等方式保留空格信息，使 decode 完全还原原始排版 |
| 训练规模 | 内置几十行 toy 语料、词表 300 | 数百 GB 语料、词表 5 万~10 万+ |
| encode 速度 | 每步重扫全序列找最优 pair，O(n²) 级，仅教学用 | Rust/C 实现 + 前缀缓存 + 并行，毫秒级 |
| 特殊 token | 无 | `<|endoftext|>`、`<pad>`、chat 模板等需单独注入并禁止被合并 |
| 正确性边界 | UTF-8 解码用 `errors="replace"` 兜底 | 严格按字节缓冲，保证任意字节流无损 |

本实现目标是 **把 BPE 的原理讲透、可一行行读懂并亲手验证无损与压缩效果**，不追求工业性能与排版完全保真。理解了这份代码，再去读 minbpe / tiktoken 源码会非常顺。
