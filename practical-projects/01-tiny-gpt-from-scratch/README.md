# 01 · 从零实现并预训练一个字符级 GPT (Tiny Character-level GPT)

一个**端到端、纯 CPU 几十秒就能跑通**的最小 GPT 实战：用纯 PyTorch 从零搭出一个
decoder-only Transformer，在一小段内置文本上做**字符级**自回归预训练，亲眼看到
loss 下降、训练后采样出带"莎士比亚对话风格"的文本。

代码自包含，**不依赖任何外部数据集 / 网络 / 分词器**，适合作为理解 GPT 内部结构的第一站。

---

## ① 演示什么原理

GPT 是一个 **decoder-only Transformer**，训练目标是"预测下一个 token"
(autoregressive language modeling)。本项目把它的每一块都拆开做出来：

| 模块 | 作用 (对应 Transformer 原理) |
|------|------------------------------|
| **Token Embedding** | 把每个字符 (token) 映射成向量 |
| **Positional Embedding** | 注入位置信息——Transformer 本身对顺序不敏感，必须显式告诉它"第几个" |
| **Multi-Head Self-Attention + 因果 mask** | 缩放点积注意力 `softmax(QKᵀ/√d)V`；因果 mask 让位置 t 只能看到 ≤ t 的历史，这是"自回归"的关键 |
| **Position-wise FFN (MLP)** | 对每个位置独立做非线性变换 (中间层放大到 4×)；注意力负责"序列内交换信息"，FFN 负责"逐位置加工特征" |
| **LayerNorm + 残差 (Pre-LN)** | 残差是"梯度高速公路"让深层可训练；Pre-LN (先归一化再进子层) 训练更稳，与 GPT-2 一致 |
| **输出层 Linear → 词表 logits** | 投影到词表大小，对"下一个字符"打分；用交叉熵作损失 |

字符级 (character-level) 意味着词表就是文本里出现过的所有字符，无需训练分词器，
天然零依赖、零下载。

---

## ② 怎么跑

环境：Python 3.13 / PyTorch 2.x (CPU 即可) / 可选 matplotlib (画 loss 曲线，没有也不会崩)。

```bash
cd practical-projects/01-tiny-gpt-from-scratch
python tiny_gpt.py
```

全程 CPU，**几十秒内**跑完。固定了随机种子 (`SEED = 1337`)，结果可复现。

跑完会在当前目录生成 `loss_curve.png` (loss 下降曲线)。

---

## ③ 预期输出 (真实运行摘录)

模型仅约 **35 万参数**，loss 从随机水平 (~3.99) 一路降到 **0.078**，降幅 **98.1%**：

```
parameters      : 349870 (349.87K)
random-guess loss (ln vocab_size) ~ 3.8286
step    0 | eval loss 3.9875  (before training)
step  100 | eval loss 1.3142
step  200 | eval loss 0.2880
step  300 | eval loss 0.1230
step  400 | eval loss 0.0929
step  500 | eval loss 0.0815
step  600 | eval loss 0.0776
------------------------------------------------------------
SUMMARY
  init  eval loss : 3.9875
  final eval loss : 0.0776
  loss reduction  : 3.9099  (98.1% lower)
  beats 0.8x random-baseline : YES
```

训练后从换行符起手采样 300 个字符，可见模型**学到了训练文本的对话格式与用词风格**
(角色名 + 冒号 + 对白)：

```
First Citizen:
Before we proced any further, hear me speak.

All:
Speak, speak.
...
First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
```

> 量化"成功"信号：`beats 0.8x random-baseline : YES`——训练后 loss 远低于
> 随机猜测基线 `ln(vocab_size)`。因为数据集很小，模型会高度"背诵"训练文本，
> 这正是 toy demo 的预期 (见"局限"一节)。

---

## ④ 对应 llm-action 文档

- Transformer 原理与模型架构：[`../../llm-algo/transformer`](../../llm-algo/transformer)
- GPT 系列：[`../../llm-algo/gpt`](../../llm-algo/gpt)
- 多头注意力 / FFN 等细节：[`../../llm-algo/transformer.md`](../../llm-algo/transformer.md)
- 旋转位置编码 RoPE (本项目用的是可学习位置嵌入，RoPE 是更现代的替代)：[`../../llm-algo/旋转编码RoPE.md`](../../llm-algo/旋转编码RoPE.md)
- 基本概念：[`../../llm-algo/基本概念.md`](../../llm-algo/基本概念.md)

---

## ⑤ 社区参考

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 本项目结构与命名深受其影响，强烈推荐进阶阅读
- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) — 中文 LLM 从零入门系统教程
- [mlabonne LLM course](https://huggingface.co/blog/mlabonne/llm-course) — 体系化 LLM 学习路线

---

## ⑥ 局限 / 与真实工程的差异

本项目是**教学 toy**，刻意把规模压到最小，与真实生产级 LLM 有以下差异：

1. **字符级 vs 子词 (subword)**：真实模型用 BPE / SentencePiece 等分词器
   (词表数万)，字符级序列更长、语义粒度更粗。
2. **规模**：这里 ~35 万参数、3 层、96 维；真实模型动辄数十亿到上万亿参数。
3. **数据量**：数据只有 ~1KB，模型很容易"背书" (过拟合)；真实预训练用 TB 级语料，
   且严格区分 train/val 防过拟合。本 demo 为简洁起见未单独划分验证集。
4. **位置编码**：用的是可学习的绝对位置嵌入 (GPT-2 风格)；现代模型多用
   RoPE / ALiBi 等更利于长度外推的方案。
5. **注意力实现**：手写朴素注意力，便于教学；生产中用 FlashAttention 等
   IO 感知的高效实现，并配合 KV-Cache 加速推理。
6. **采样策略**：仅做了最朴素的概率采样；真实推理常用 temperature / top-k /
   top-p (nucleus) 等控制生成质量与多样性。
7. **工程化**：没有混合精度 (AMP)、梯度裁剪、学习率调度 (warmup+cosine)、
   分布式 (DDP / FSDP / TP / PP)、checkpoint 等——这些都是真实训练的必备项，
   可作为后续练习逐项加上。

---

*所属：llm-action / practical-projects (动手可跑实战系列)*
