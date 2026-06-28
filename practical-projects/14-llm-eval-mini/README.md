# 从零实现迷你 LLM 评测(困惑度 + 生成对比 + 下游准确率)

一个**自包含、CPU 几十秒可跑通**的教学脚本,用一个 toy 字符级语言模型(char-LM),
把"如何评测一个大模型"这件事拆开讲清楚。不依赖任何外部数据集 / 网络 / GPU。

> 真实 LLM 的评测(perplexity、生成质量、下游任务准确率)在**原理上**和一个几十 KB
> 的 char-LM 完全一致,只是规模不同。把模型缩到 CPU 上秒级训练,就能把**评测指标本身**
> (而不是工程规模)讲透。

---

## 一、演示什么原理

脚本围绕 LLM 评测的三个核心维度:

### 1. 困惑度 Perplexity(内在指标)
- 语言建模最核心的内在(intrinsic)指标。本质就是:
  **PPL = exp(平均交叉熵)**。
- 含义:模型对"持有的真实文本"平均有多"意外"。等价于"有效分支数"。
- PPL 越低越好。随机乱猜时 PPL ≈ 词表大小(本例词表 24,所以未训练模型 PPL ≈ 24)。
- 脚本打印 PPL 随训练步数下降,这就是"模型在学东西"的可量化信号。

### 2. 解码策略对比(生成质量评测的前提)
同一个模型,换不同采样方式,生成差异巨大。所以**评测生成必须固定解码策略**:
- **greedy(贪心)**:每步取 `argmax`,确定性最高、最稳但易重复。
- **temperature(温度采样)**:`logits / T` 后采样。`T<1` 更尖锐保守,`T>1` 更平更随机。
- **top-k**:只在概率最高的 `k` 个里采样,截断长尾,平衡多样性与质量。

### 3. 下游任务准确率(外在指标)
PPL 低 ≠ 下游好用,需要面向任务的外在(extrinsic)指标。
脚本构造一个 toy 任务:取语料中的真实前缀,让模型 `greedy` 预测下一个字符,
与真实字符比对,算 **next-char 准确率**(完形填空式)。

---

## 二、怎么跑

```bash
cd practical-projects/14-llm-eval-mini
python eval_mini.py
```

环境:Python 3.13 / numpy 2.3 / torch 2.12 (CPU)。约 **14 秒**跑完。
若装了 `matplotlib`,会额外存出 `ppl_curve.png`;没装则纯文本打印,不会崩。

---

## 三、预期输出(真实跑通摘录)

```
[data] corpus_chars=8520  vocab_size=24  train_tokens=7668  val_tokens=852
[data] random-guess perplexity ~= vocab_size = 24 (a useful sanity baseline)
[model] TinyCharLM params=27416
------------------------------------------------------------------------
[eval] step=   0  val_ce=3.1840  val_ppl=   24.14  downstream_acc=0.025  (baseline)
[eval] step= 100  train_loss=0.4241  val_ce=0.4092  val_ppl=    1.51  downstream_acc=0.990
[eval] step= 300  train_loss=0.0847  val_ce=0.0940  val_ppl=    1.10  downstream_acc=1.000
[eval] step= 600  train_loss=0.0686  val_ce=0.0712  val_ppl=    1.07  downstream_acc=1.000
------------------------------------------------------------------------
[result] perplexity: baseline=24.14 -> final=1.07  (22.5x lower)
[result] downstream next-char accuracy: baseline=0.025 -> final=1.000
------------------------------------------------------------------------
[decoding] same model, same prompt, different decoding strategies:
  greedy          : 'the mat. the dog ran in the park. a bird fle'
  temperature=0.5 : 'the lake. the sun is bright today. she likes'
  temperature=1.3 : 'the sun is bright today. she likes to rdat g'
  top_k=5         : 'the old town. i drink water in the morning. '
------------------------------------------------------------------------
[summary] perplexity dropped: True | downstream improved: True
[summary] OVERALL: PASS - the model measurably learned
```

**怎么看懂这几行:**
- **PPL 从 24.14 降到 1.07**(约 22.5 倍):接近"随机基线 = 词表大小"到"几乎完全确定"的两端,模型确实学到了语料结构。
- **下游准确率从 0.025 升到 1.000**:内在指标和外在指标同向改善。
- **解码对比**:`greedy` / `top_k` / 低温采样都吐出连贯的语料短语;
  **`temperature=1.3` 出现了 `rdat g` 这种噪声**——高温让分布变平、采到了低概率字符,直观展示了"温度越高越发散"。

---

## 四、对应 llm-action 文档

- 评测专题目录:[`../../llm-eval`](../../llm-eval)

本项目是该文档的"动手可跑"补充:文档讲评测体系与方法论,本脚本用最小可运行代码把
**困惑度计算、解码策略、下游准确率**三件事跑出来给你看。

---

## 五、社区参考

- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 业界最常用的开源 LLM 评测框架,支持数百个标准任务(本项目可视为它"困惑度 + 解码 + 任务准确率"思路的极简手写版)。

---

## 六、局限 / 与真实工程的差异

| 维度 | 本 toy 项目 | 真实 LLM 评测工程 |
| --- | --- | --- |
| 模型 | 单层 LSTM,~2.7 万参数 | Transformer,十亿~万亿参数 |
| 分词 | 字符级 | BPE / SentencePiece 子词 |
| 困惑度 | 自己手算 `exp(CE)` | 同原理,但需处理超长上下文、滑动窗口、跨文档边界 |
| 下游评测 | 自造 next-char toy 任务 | MMLU / GSM8K / HumanEval / C-Eval 等标准基准 |
| 生成评测 | 肉眼对比几条样本 | 自动指标(BLEU/ROUGE)+ LLM-as-judge + 人工标注 |
| 解码 | greedy / temperature / top-k | 还有 top-p(nucleus)、beam search、对比解码、投机解码等 |
| 数据污染 | 不涉及(toy 语料) | 必须严防测试集泄漏到训练集(contamination) |
| 复现性 | 固定种子即可 | 还需固定 prompt 模板、few-shot 示例、解码参数、框架版本 |

**一句话**:本项目让你亲手摸到"评测一个模型"的最小骨架——指标怎么算、解码怎么影响生成、
内在指标和外在指标如何对应。要做真实评测,请上 `lm-evaluation-harness` 这类成熟框架。
