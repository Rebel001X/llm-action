# 使用 torchtext 训练 Transformer 模型（数据与模型准备）

> 用 `torchtext` 把 WikiText-2 语料处理成「分词→建词表→展平→batchify」的张量，并定义一个标准的 Transformer Encoder 语言模型——这是后续「流水线并行训练」的**数据与模型地基**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[ai-framework/pytorch/README]] · [[llm-algo/transformer/模型架构]] · [[B07:llm-inference/大模型推理张量并行]]

## 阅读地图

| 你想知道 | 跳到 |
|---|---|
| 这篇笔记在整条流水线教程里处于什么位置 | §0 锚点 |
| 语言模型 / 自回归 / 因果掩码是什么 | §1 地基 |
| `TransformerModel` 每一层在干什么、为什么要 `*√d_model` | §2 模型结构 |
| 位置编码的 sin/cos 公式怎么来的、为什么 `register_buffer` | §3 位置编码 |
| `torchtext` 怎么分词、建词表、OOV 怎么处理 | §4 数据处理 |
| `batchify` 把一维语料拧成二维矩阵，形状怎么变 | §5 batchify |
| 输入-目标序列（`get_batch`）怎么错一位对齐 | §6 生成输入-目标 |
| 完整可运行代码 | §「实操：完整代码」 |
| 形状对不上 / OOV / 词表被消费 等坑 | §「常见问题/坑」 |

## 0. 一句话锚点

**本文做两件准备工作：(1) 用 `torchtext` 把原始文本 WikiText-2 变成模型能吃的整数张量；(2) 定义一个 Transformer Encoder 语言模型。** 这两样东西本身**还没有用到流水线并行**——它们是 [[3-使用流水线并行训练Transformer模型]] 与 [[4-使用DDP与流水线并行训练Transformer模型]] 的输入。理解了本文，后两篇就只剩「把模型切两段、放两块卡」这一步差异。

```
本目录 4 篇的关系（自底向上）
┌─────────────────────────────────────────────┐
│ 1-流水线.md          Pipe API 是什么、参数含义  │
│ 2-本文              数据 + 模型（单卡，无并行） │ ← 你在这
│ 3-流水线并行         把本文模型切 2 段 → 2 卡 PP │
│ 4-DDP+流水线并行      多组「2 卡 PP」再叠 DDP    │
└─────────────────────────────────────────────┘
```

## 1. 地基：这是个什么任务

### 1.1 任务 = 语言建模（Language Modeling）

给定一段词序列 $w_1, w_2, \dots, w_{t-1}$，预测**下一个词** $w_t$。整段话的概率被拆成逐词条件概率的连乘（自回归分解）：

$$
P(w_1,\dots,w_T)=\prod_{t=1}^{T} P(w_t \mid w_1,\dots,w_{t-1})
$$

模型对每个位置输出一个**词表大小**的分布，训练目标是让真实的下一个词概率最大（等价于最小化交叉熵）。

### 1.2 为什么用 Encoder + 因果掩码，而不是 Decoder

GPT 类用的是 Transformer Decoder。本教程用 `TransformerEncoder` 配一个**上三角因果掩码（causal mask）**来达到同样效果：让位置 $t$ 只能看到 $\le t$ 的词，看不到未来。掩码长这样（`True`/`-inf` 表示「禁止注意」）：

```
            被注意的列 (key) →
          w1   w2   w3   w4
查询  w1 [  0  -inf -inf -inf ]   w1 只能看自己
行    w2 [  0    0  -inf -inf ]   w2 能看 w1,w2
(query) w3 [  0    0    0  -inf ]   w3 能看 w1..w3
      w4 [  0    0    0    0  ]   w4 能看全部历史
                        ↑ 上三角 -inf：屏蔽未来，保证"不偷看答案"
```

没有这个掩码，模型在预测 $w_t$ 时能直接看到 $w_t$，训练就退化成「抄答案」，完全学不到东西。

### 1.3 前置依赖

| 包 | 作用 | 备注 |
|---|---|---|
| `torch` | 张量 / 自动微分 / `nn.Transformer` | 本系列的核心 |
| `torchtext` | 数据集 `WikiText2`、分词器、`build_vocab_from_iterator` | 数据处理专用，需单独装 |
| `math` | `sqrt`、`log`（位置编码与缩放） | 标准库 |

> torchtext 与 torch 版本需配套（torchtext 0.x 对应特定 torch 版本），版本错配是最常见的 `ImportError` 来源。

## 2. 模型结构：`TransformerModel` 逐层拆解

### 2.1 数据在模型里的流动路径

```
输入 src  [seq_len, batch]   ← 整数 token id
  │
  ▼ nn.Embedding(ntoken, d_model)        把 id 查成向量
  │  × math.sqrt(d_model)                 ← 缩放（见 §2.2）
  ▼  [seq_len, batch, d_model]
  │
  ▼ PositionalEncoding                    加上"第几个词"的信息
  ▼  [seq_len, batch, d_model]
  │
  ▼ TransformerEncoder(× nlayers)         多头自注意力 + FFN，叠 N 层
  │  （配 src_mask 因果掩码）
  ▼  [seq_len, batch, d_model]
  │
  ▼ nn.Linear(d_model, ntoken)            投影回词表大小
  ▼  [seq_len, batch, ntoken]             ← 每个位置一个词分布（logits）
```

注意整套都用 **`[seq_len, batch, ...]`**（seq 在前）的「时间优先」布局——这是 PyTorch `nn.Transformer` 的默认约定（`batch_first=False`），后面 `batchify` 的转置就是为了凑成这个布局。

### 2.2 为什么 embedding 之后要 `* math.sqrt(d_model)`

```python
src = self.embedding(src) * math.sqrt(self.d_model)
```

这一行直接出自原始 Transformer 论文。`init_weights` 把 embedding 初始化在 $[-0.1, 0.1]$ 这种小范围，量级偏小；而紧跟其后的位置编码 sin/cos 的取值范围是 $[-1, 1]$。若不放大 embedding，**位置信号会盖过词义信号**。乘上 $\sqrt{d_{model}}$ 把词向量的量级抬到与位置编码相当，二者相加才平衡。例如 $d_{model}=200$ 时 $\sqrt{200}\approx 14.1$。

### 2.3 权重初始化

```python
def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.linear.bias.data.zero_()
    self.linear.weight.data.uniform_(-initrange, initrange)
```

embedding 和输出 `linear` 用均匀分布 $U(-0.1, 0.1)$ 初始化，bias 清零。小范围初始化避免一开始 logits 过大导致 softmax 饱和、梯度消失。

### 2.4 完整模型代码（原文保留）

```python
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 单个EncoderLayer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # TransformerEncoder 包含多个 EncoderLayer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 嵌入层
        self.embedding = nn.Embedding(ntoken, d_model)

        self.d_model = d_model

        self.linear = nn.Linear(d_model, ntoken)

        # 初始化权重
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
```

构造函数 5 个超参的含义对照表：

| 参数 | 含义 | 典型值（本教程后续） |
|---|---|---|
| `ntoken` | 词表大小（vocab size），决定 embedding 行数与输出维度 | `len(vocab)` ≈ 28785 |
| `d_model` | 词向量 / 隐藏维度 | 200 |
| `nhead` | 多头注意力的头数（须整除 `d_model`） | 2 |
| `d_hid` | FFN 中间层维度 | 200 |
| `nlayers` | 堆叠多少个 `TransformerEncoderLayer` | 2 |
| `dropout` | 丢弃率 | 0.2 ~ 0.5 |

> `nhead` 必须整除 `d_model`：每个头维度 = `d_model / nhead`，例如 200/2 = 100。填了不能整除的值会直接报错。

## 3. 位置编码：为什么以及怎么算

### 3.1 为什么需要位置编码

自注意力本身是**置换不变（permutation-invariant）**的——把输入词打乱顺序，注意力的计算结果只是跟着换位，模型分不清「猫吃鱼」和「鱼吃猫」。必须显式把「第几个位置」的信息注入进去。

### 3.2 正弦位置编码公式

对位置 $pos$、维度 $i$：

$$
PE_{(pos,2i)}=\sin\!\Big(\frac{pos}{10000^{2i/d_{model}}}\Big),\qquad
PE_{(pos,2i+1)}=\cos\!\Big(\frac{pos}{10000^{2i/d_{model}}}\Big)
$$

代码里 `div_term` 就是 $\frac{1}{10000^{2i/d_{model}}}$，用 `exp(2i·(-log(10000)/d_model))` 这种等价写法算（数值更稳，避免直接幂运算溢出）。偶数维填 `sin`、奇数维填 `cos`：

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
```

### 3.3 频率随维度变化（直觉图）

低维（小 $i$）波长短、变化快，刻画近距离；高维波长长、变化慢，刻画远距离：

```
维度 i=0（高频）: pos →  /\/\/\/\/\/\/\/\   细分相邻位置
维度 i 中间     : pos →  /‾\__/‾\__/‾\__    中等尺度
维度 i 大（低频）: pos →  ___/‾‾‾\___/‾‾‾     粗粒度、长距离
```

不同频率的组合给每个位置一个**唯一**的向量指纹，且相对位置可由三角恒等式线性表示，模型容易学到「相距 k」的关系。

### 3.4 为什么用 `register_buffer` 而不是 `nn.Parameter`

位置编码是**固定的、不需要训练**的常量。`register_buffer`：
- 不会被加入 `parameters()`，优化器不会更新它，也不会有梯度；
- 但会随 `model.to(device)` 一起搬到 GPU，并保存进 `state_dict`。

如果误用 `nn.Parameter`，就变成可学习参数，既浪费显存又改变了「正弦位置编码」的语义。`forward` 里 `self.pe[:x.size(0)]` 只取前 `seq_len` 行相加（`max_len=5000` 预生成足够长，按需截断）。

## 4. 数据处理：torchtext 把文本变张量

### 4.1 三步走

```
原始文本(每行一句)  ──tokenizer──▶  词列表
   "You can install..."          ['you','can','install',...]
                                        │
                            build_vocab_from_iterator
                                        ▼
                                 vocab: 词 ↔ 整数id
                                        │
                                vocab(tokenizer(line))
                                        ▼
                                 [178, 112, 199, ...]  ← 整数序列
```

### 4.2 分词与建词表

```python
import torch
from torch import nn, Tensor
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')

tokenizer = get_tokenizer('basic_english')

# 构建词表
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])

# 设置默认索引值，当单词OOV时，使用默认索引
vocab.set_default_index(vocab['<unk>'])

# 0
print(vocab['<unk>'])
print(vocab.get_default_index())

item = "You can now install TorchText using pip!"
# [178, 112, 199, 19230, 0, 438, 0, 385]
print(vocab(tokenizer(item)))
```

要点：
- `get_tokenizer('basic_english')`：小写化 + 按空格/标点切分的简易英文分词器。
- `specials=['<unk>']`：把 `<unk>`（未知词 unknown）放进词表，且因为是 specials 的第一个，它的 id = **0**。
- `set_default_index(vocab['<unk>'])`：**关键一步**。设默认索引后，遇到词表里没有的词（OOV, Out-Of-Vocabulary）才返回 `<unk>` 的 id（0），否则会抛 `RuntimeError`。
- 看输出 `[178, 112, 199, 19230, 0, 438, 0, 385]`：里面两个 `0` 就是 `torchtext`、`pip` 这两个训练集没见过的词被映射成了 `<unk>`。

| 词 | id | 说明 |
|---|---|---|
| `<unk>` | 0 | 特殊符号，OOV 兜底 |
| `you` | 178 | 训练集见过 |
| `torchtext` | 0 | OOV → `<unk>` |
| `pip` | 0 | OOV → `<unk>` |

### 4.3 展平为一维张量

```python
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()

train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

print(train_data.shape, val_data.shape, test_data.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- `filter(lambda t: t.numel() > 0, data)`：过滤掉**空行**（WikiText 里有大量空行，分词后是 0 长度张量）。
- `torch.cat(...)`：把所有行**首尾拼成一条**长长的一维 token 流——语言模型不关心句子边界，整段语料当成一条连续序列。
- **坑提示（原文注释已点出）**：`train_iter` 在「建词表」时被迭代消费过一次，迭代器是一次性的，所以必须 `WikiText2()` 重新创建，否则得到空数据。

### 4.4 一维 → 二维：`batchify`

```python
def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """

    seq_len = data.size(0) // bsz
    # 移除无法整除的余数
    data = data[:seq_len * bsz]
    # torch.Size([2049980])
    print(data.shape)
    # t() 转置
    # contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
    data = data.view(bsz, seq_len).t().contiguous()
    # torch.Size([102499, 20])
    print(data.shape)
    return data.to(device)


batch_size = 20
eval_batch_size = 10
# torch.Size([102499, 20])
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
```

## 5. batchify 的形状变换（手算）

`batchify` 把一条长流切成 `bsz` 列、每列是一条独立子序列。以原文的真实数字为例：训练集展平后约 `2049980` 个 token，`bsz=20`：

$$
seq\_len = \lfloor 2049980 / 20 \rfloor = 102499,\qquad 102499 \times 20 = 2049980
$$

正好整除（余数 0 被裁掉）。形状演化：

```
原始一维流  data: [2049980]
      │  data[:seq_len*bsz]            裁掉余数（这里没余数）
      │  .view(bsz, seq_len)           → [20, 102499]  （每行一条子序列）
      │  .t()                          转置 → [102499, 20]
      │  .contiguous()                 转置后内存不连续，拷成连续布局
      ▼
   train_data: [102499, 20]  = [seq_len, batch_size]
                  ↑列与列之间相互独立、并行处理
```

直观图（拿 token 流 0,1,2,...,23，bsz=4 演示）：

```
长流: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
view(4,6):          切成 4 条，每条 6 个
  行0: 0  1  2  3  4  5
  行1: 6  7  8  9 10 11
  行2:12 13 14 15 16 17
  行3:18 19 20 21 22 23
.t() → [6,4]:    列 = 一条独立子序列（沿"时间"往下走）
  0  6 12 18
  1  7 13 19
  2  8 14 20      ← 每一列在时间上连续: 0→1→2→...
  3  9 15 21
  4 10 16 22
  5 11 17 23
```

> **为什么要 `.contiguous()`**：`.t()` 只是改了张量的「步长（stride）」视图，底层内存仍是转置前的排列。后续 `view`/某些算子要求内存连续，`.contiguous()` 重新拷贝一份按新语义排列的连续内存，避免报错。

| 张量 | bsz | 形状 `[seq_len, bsz]` |
|---|---|---|
| `train_data` | 20 | `[102499, 20]` |
| `val_data` | 10 | `[N_val // 10, 10]` |
| `test_data` | 10 | `[N_test // 10, 10]` |

## 6. 生成输入-目标序列（get_batch）

`batchify` 后我们有 `[seq_len_total, bsz]` 的大矩阵。训练时不会一次喂全部 `seq_len_total`（太长、显存炸），而是按 **`bptt`（back-prop through time，截断长度）** 一块块取。每块的「目标 = 输入错后一位」——这正是语言建模「预测下一个词」的体现：

```
取一段长度 bptt 的窗口（沿 seq 维），bptt=2 演示:
data(输入):  [w_i  , w_{i+1}]      ← 第 t 步喂这俩
target(目标): [w_{i+1}, w_{i+2}]   ← 期望输出（各错一位）

          位置:  t        t+1
  input :  w_i  ─pred→  应得 w_{i+1}
  input : w_{i+1}─pred→ 应得 w_{i+2}
                 ↑ 目标就是把输入整体左移一格
```

PyTorch 官方教程对应的 `get_batch`（标准实现，本文数据格式即为此服务）：

```python
bptt = 35

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int, 当前块的起始位置

    Returns:
        tuple (data, target):
          data   形状 ``[seq_len, batch_size]``
          target 形状 ``[seq_len * batch_size]`` （已展平，配合交叉熵）
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target
```

- `target` 比 `data` 整体右移一位（`i+1` 起），实现「下一个词」对齐。
- `seq_len = min(bptt, len(source)-1-i)`：靠近结尾不足 `bptt` 时取剩余长度，保证不越界（`-1` 留给 target 多取的那一位）。
- `target.reshape(-1)` 展平成一维，是为了直接喂 `nn.CrossEntropyLoss`（它要 logits `[N, C]` 配 target `[N]`）。

> 本文到此为止只准备好了 `train_data / val_data / test_data` 和 `TransformerModel`。**真正的训练循环、损失、优化器，以及把模型切段做流水线并行**，在 [[3-使用流水线并行训练Transformer模型]] 展开。

## 实操：完整代码（按运行顺序）

把上面分散的片段按可运行顺序串起来（均为原文真料，未改命令/参数）：

```python
# ===== 1. 模型定义 =====
import math, os
from tempfile import TemporaryDirectory
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

# ===== 2. 数据处理 =====
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_iter 已在建词表时被消费，必须重建
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 3. batchify =====
def batchify(data, bsz):
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)   # [102499, 20]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
```

关键参数速查：

| 参数 | 值 | 含义 |
|---|---|---|
| `batch_size` | 20 | 训练 batch（= batchify 的列数） |
| `eval_batch_size` | 10 | 验证/测试 batch |
| `bptt` | 35 | 每次反传的截断序列长度（在 get_batch 处） |
| `<unk>` id | 0 | OOV 兜底符号 |
| `train_data.shape` | `[102499, 20]` | `[seq_len, batch_size]` |

## 常见问题/坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `RuntimeError: Token not found` | 没调 `set_default_index`，遇到 OOV 词报错 | 调 `vocab.set_default_index(vocab['<unk>'])` |
| `data_process` 返回空 / 数据为 0 | `train_iter` 是一次性迭代器，建词表时已被消费 | 用 `WikiText2()` 重新创建 iter（原文注释已强调） |
| `view size is not compatible` | `.t()` 后内存不连续直接 `view` | 转置后加 `.contiguous()` |
| `nhead` 报错 | `nhead` 不整除 `d_model`（如 200/3） | 取整除值，如 `nhead=2`（200/2=100） |
| 位置信息「丢失」/学不到顺序 | 漏加位置编码，或位置编码量级盖过词义 | 保留 `* math.sqrt(d_model)` 缩放 |
| 训练 loss 不降、像背答案 | 没传因果 `src_mask`，模型偷看未来 | 训练时传上三角 `-inf` 掩码 |
| `import torchtext` 失败 | torchtext 与 torch 版本不匹配 | 安装与 torch 配套的 torchtext 版本 |
| 余数 token 被「丢失」 | `batchify` 裁掉了不能整除 `bsz` 的尾部 | 正常行为；想保留可改 padding 策略 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 上游 API：[[1-流水线]]（`torch.distributed.pipeline.sync.Pipe` 参数详解）
- 下游实战：[[3-使用流水线并行训练Transformer模型]] · [[4-使用DDP与流水线并行训练Transformer模型]]
- 本目录总览：[[llm-train/pytorch/distribution/README]]
- 框架：[[ai-framework/pytorch/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 训练总览：[[llm-train/README]] · [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]
- 模型原理：[[llm-algo/transformer/模型架构]]
- 并行进阶：[[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]
- 相关方向：[[llm-train/peft/PEFT-API]] · [[ai-framework/huggingface-peft/README]] · [[llm-alignment/RLHF]] · [[llm-compression/quantization/量化基础]]
