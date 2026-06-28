# 大模型精度评估：困惑度·ROUGE·HELM·竞技场

> 一句话定位：**精度评估**回答"模型答得对不对、好不好"——从词级概率(困惑度)到文本重叠(ROUGE)，再到全维度榜单(HELM)和人类盲测(Chatbot Arena)。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[llm-inference/解码策略]] · [[llm-compression/quantization/量化基础]]

## 阅读地图

| 节 | 内容 | 你会得到 |
|----|------|---------|
| 0 | 一句话锚点 | 精度 vs 性能的区分 |
| 1 | 地基：什么是"评估" | 自动指标 / 人类评测两条路 |
| 2 | 困惑度 Perplexity | 交叉熵→PPL 的手算与坑 |
| 3 | ROUGE | n-gram 召回，中文分词坑 |
| 4 | HELM | 多维度 holistic 评测，59 指标全表 |
| 5 | lm-evaluation-harness | 学术界事实标准 harness |
| 6 | Chatbot Arena | 成对盲测 + Elo |
| 7 | CLEVA | 中文评测平台 |
| 实操 | 命令/链接/资源 | 原文真料汇总 |
| 坑 | 常见问题 | 对照表排错 |

## 0. 一句话锚点

> **精度评估(quality/accuracy)** = 衡量"输出内容好不好"；与之并列的 **性能评估(performance)** = 衡量"跑得快不快、省不省"（吞吐、延迟、显存，见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]）。本篇专攻**精度**。

精度评估的两大流派：

```
                  精度评估
                     │
        ┌────────────┴────────────┐
   自动指标(便宜、可复现)      人类/模型评测(贵、贴近真实)
        │                         │
  ┌─────┼─────┐            ┌──────┼───────┐
  PPL  ROUGE  Exact     人工打分  Arena   LLM-as-judge
 (内在) (重叠)  (匹配)    (绝对)   (成对Elo) (GPT-4当裁判)
```

- **内在指标(intrinsic)**：困惑度——只看模型对语料的概率拟合，不需要标准答案。
- **基于参考(reference-based)**：ROUGE / BLEU / Exact Match——拿模型输出和"金标准答案"比对。
- **基于人类/裁判(human/judge)**：Chatbot Arena、HELM 中的 HumanEval-*——评开放式问答。

> 入门综述（原文链接）：*How to Evaluate a Large Language Model (LLM)?* — https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/

## 1. 地基：为什么需要这么多指标？

一个 LLM 要做的事情五花八门：续写、翻译、摘要、问答、写代码、拒绝有害请求……**没有任何单一数字能概括"好坏"**。所以评估被拆成正交的维度：

```
任务类型      → 合适的指标
────────────────────────────
语言建模/续写  → Perplexity (越低越好)
摘要/翻译      → ROUGE / BLEU / BERTScore
选择题/抽取    → Exact Match / F1 / Accuracy
代码生成       → pass@k (单测通过率)
检索/排序      → RR@10 / NDCG@10
开放式对话      → 人类盲测 Elo / LLM-judge
安全/公平       → Toxicity / Bias / 扰动鲁棒性
```

HELM 的核心洞见就是：**把这些维度全摆出来**(holistic)，而不是只报一个 Accuracy。

## 2. 困惑度 Perplexity（原文核心料）

### 2.1 原文定义（保留）

> 语言模型效果好坏的常用评价指标是**困惑度(perplexity)**，在测试集上得到的 perplexity 越低，说明建模效果越好。
> PPL 用在 NLP 中衡量语言模型好坏。它根据每个词估计一句话出现的概率，并用句子长度做 normalize。
> **PPL 越小越好**：PPL 越小 → $p(w_i)$ 越大 → 句子中每个词的概率较高 → 这句话契合得越好。
>
> 参考：https://blog.csdn.net/hxxjxw/article/details/107722646

### 2.2 公式拆原子

对一个长度为 $N$ 的句子 $W = w_1 w_2 \dots w_N$，语言模型给出联合概率 $p(W)$。困惑度定义为该概率的**几何平均的倒数**：

$$\text{PPL}(W) = p(w_1, w_2, \dots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{\prod_{i=1}^{N} p(w_i \mid w_{<i})}}$$

**为什么是 $-1/N$ 次方？** 句子越长，$\prod p(w_i)$ 自然越小（概率连乘），直接比会"惩罚长句"。开 $N$ 次方根 = 用长度做 normalize，让不同长度的句子可比。

困惑度和**交叉熵(cross-entropy)** 是一对：

$$\text{PPL}(W) = 2^{H(W)} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\ln p(w_i \mid w_{<i})\right)$$

也就是说，**PPL = e 的(平均负对数似然)次方**。训练时 loss 用的就是这个平均 NLL，所以 `PPL = exp(loss)` 是工程里最常用的换算。

### 2.3 直觉：PPL 是"等效分支数"

PPL 可理解为模型在每一步**平均要在多少个等概率候选词里做选择**：

```
PPL = 1    →  完美预测，每步只有1个候选(确定)
PPL = 10   →  每步约等于在10个等概率词里猜
词表大小 V →  完全随机模型 PPL ≈ V (毫无知识)
```

所以 PPL 从词表大小 $V$（比如 5 万）一路降到几十、十几，就是模型"学会语言"的过程。

### 2.4 数值手算示例

假设词表只有 4 个等概率词，模型对每个位置都给 $p=0.25$，句子长 $N=3$：

$$\text{PPL} = \left(0.25 \times 0.25 \times 0.25\right)^{-1/3} = (0.015625)^{-1/3} = 4$$

PPL=4 = 词表大小，符合"完全随机=词表大小"的直觉。换成**学到东西**的模型，3 个词概率为 $0.5, 0.4, 0.5$：$\text{PPL}=(0.1)^{-1/3}\approx 2.15$，PPL 从 4 降到 2.15 → 模型更自信、更准。

### 2.5 PPL 的坑（必读）

```
┌──────────────────────────────────────────────────────────┐
│ 坑1：分词器(tokenizer)不同 → PPL 不可比！                   │
│   同一句话，BPE 切成 8 个 token vs WordPiece 切成 12 个，    │
│   N 不同，几何平均的基数就不同，跨模型比 PPL 是耍流氓。       │
├──────────────────────────────────────────────────────────┤
│ 坑2：测试集泄漏 → PPL 虚低。训练语料若混入测试集，           │
│   模型"背过答案"，PPL 低但泛化差。                          │
├──────────────────────────────────────────────────────────┤
│ 坑3：PPL 低 ≠ 对话好。RLHF/对齐后的模型 PPL 常常变高，       │
│   但人类更喜欢——因为对齐牺牲了"语料拟合"换"有用、安全"。      │
├──────────────────────────────────────────────────────────┤
│ 坑4：量化/不同推理引擎算的 PPL 会有微小差异，               │
│   见下方 vLLM vs HuggingFace 一致性链接。                   │
└──────────────────────────────────────────────────────────┘
```

> 坑4 与量化精度强相关：FP16→INT4 量化会让 PPL 略升，PPL 涨幅常被用作量化"掉点"的指标，详见 [[llm-compression/quantization/量化基础]] 与 [[llm-compression/quantization/GPTQ]]。

### 2.6 计算 PPL 的实操资源（原文真料保留）

- 一致性对比 *Comparing vLLM and Hugging Face Transformers*：https://medium.com/@kimdoil1211/ensuring-consistency-comparing-vllm-and-hugging-face-transformers-4cd88b83ed3c
- 权重量化与 PPL：https://www.analyticsvidhya.com/blog/2025/01/neural-network-weight-quantization/#h-advantages-of-weight-quantization
- vLLM 计算 PPL 的两个 issue 讨论：
  - https://github.com/vllm-project/vllm/issues/1019
  - https://github.com/vllm-project/vllm/issues/185

**最小实现思路**（HuggingFace 风格，原理代码）：

```python
import torch
# 滑窗法：把长文本切成 max_length 窗口，累加每个 token 的 NLL
nlls = []
for input_ids in windows:                      # 逐窗口
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        # out.loss 已是该窗口平均交叉熵(NLL)
        nlls.append(out.loss * input_ids.size(1))
ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
# 关键：用 stride 滑窗、忽略已计算过的前缀 token(label=-100)，避免重复计 + 上下文截断偏差
```

要点：长文本超过上下文窗口时用**滑动窗口**，重叠部分的 token 设 `label=-100` 不计入 loss，否则会重复计数导致 PPL 偏差。

## 3. ROUGE（摘要/生成的召回类指标）

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 衡量**模型输出与参考答案的 n-gram 重叠**，偏召回，常用于摘要/翻译。

```
ROUGE-N 召回 = (输出与参考共有的 N-gram 数) / (参考中的 N-gram 总数)

ROUGE-1 : 单词(unigram)重叠
ROUGE-2 : 二元词组(bigram)重叠 —— HELM 也用它(见59指标)
ROUGE-L : 最长公共子序列(LCS)，不要求连续，更宽松
```

**数值示例**：参考 = "猫 坐 在 垫子 上"（5 个 unigram），输出 = "猫 坐 垫子 上"。
共有 unigram = {猫,坐,垫子,上} = 4 个 → ROUGE-1 召回 = 4/5 = **0.8**。

### 3.1 中文 ROUGE 的坑

ROUGE 原本按**空格分词**，中文没有空格，直接套会把整句当一个 token，结果全错。必须先用分词器（jieba 等）切词。原文给的就是这个专用库：

- 中文 ROUGE：https://github.com/Isaac-JL-Chen/rouge_chinese

```
英文: "the cat sat"  --空格--> [the][cat][sat]   ✓直接可用
中文: "猫坐在垫子上"  --空格--> ["猫坐在垫子上"]   ✗整句一个token
中文: "猫坐在垫子上"  --jieba--> [猫][坐][在][垫子][上]  ✓ rouge_chinese
```

## 4. HELM（Holistic Evaluation of Language Models）

斯坦福 CRFM 出品，理念：**不只报准确率，而是把准确率、校准、鲁棒性、公平性、效率、毒性等全维度一起报**，形成"体检报告"而非"单科分数"。

- 官网：https://crfm.stanford.edu/helm/latest/

```
传统评测              HELM 体检式评测
─────────            ──────────────────────────────
只看 Accuracy   →    Accuracy + Calibration(校准)
                      + Robustness(抗扰动)
                      + Fairness(公平)
                      + Efficiency(能耗/延迟)
                      + Toxicity/Bias(毒性/偏见)
一个场景        →    多任务多场景横向铺开
```

### 4.1 HELM 的 59 个指标全表（原文真料，分组解读）

| 大类 | 指标 | 一句话含义 |
|------|------|-----------|
| **Accuracy** | Quasi-exact match / Exact match | (近似)完全匹配 |
| | F1 / F1 (set match) | 精确率召回率调和 |
| | RR@10 / NDCG@10 | 检索排序质量(倒数排名/折损增益) |
| | ROUGE-2 | bigram 重叠(见上节) |
| | Bits/byte | 压缩视角的语言建模(=PPL 同源) |
| | pass@1 | 代码一次通过率 |
| | Equivalent / Equivalent (chain of thought) | 答案等价(含 CoT) |
| **Calibration(校准)** | Max prob / 1-bin & 10-bin ECE | 置信度是否=真实正确率 |
| | (after Platt scaling) | Platt 缩放校准后的 ECE |
| | Selective coverage-accuracy area | 弃答-准确率曲线面积 |
| **Robustness(鲁棒)** | *(perturbation: typos)* 系列 | 加错别字后还对不对 |
| | *(perturbation: synonyms)* 系列 | 换同义词后稳不稳 |
| **Fairness(公平)** | *(perturbation: dialect/race/gender)* | 换方言/种族/性别表述后是否变化 |
| | Bias / Stereotypical associations | 刻板印象关联 |
| | Demographic representation | 人群表征是否均衡 |
| **Toxicity** | Toxic fraction | 有毒输出占比 |
| **Efficiency** | Observed/Idealized/Denoised inference runtime | 实测/理想/去噪推理耗时 |
| | Estimated training emissions/energy | 训练碳排/能耗 |
| **Summarization** | SummaC / QAFactEval / BERTScore | 摘要事实一致性/语义相似 |
| | Coverage/Density/Compression | 抽取度/压缩比 |
| | HumanEval-faithfulness/relevance/coherence | 人工评忠实/相关/连贯 |
| **APPS(代码)** | Avg. # tests passed / Strict correctness | 平均过测数/严格正确 |
| **BBQ(偏见QA)** | BBQ (ambiguous / unambiguous) | 歧义/非歧义下偏见 |
| **Copyright** | Longest common prefix / Edit distance/similarity | 是否逐字复述版权文本 |
| **Disinformation** | Self-BLEU / Entropy (Monte Carlo) | 生成多样性 |
| **Classification** | Macro-F1 / Micro-F1 | 分类宏/微平均 F1 |

原文完整 59 指标清单（保留原貌，便于对照官网）：

```
# Accuracy
none
Quasi-exact match
F1
Exact match
RR@10
NDCG@10
ROUGE-2
Bits/byte
Exact match (up to specified indicator)
Absolute difference
F1 (set match)
Equivalent
Equivalent (chain of thought)
pass@1

# Calibration
Max prob
1-bin expected calibration error
10-bin expected calibration error
Selective coverage-accuracy area
Accuracy at 10% coverage
1-bin expected calibration error (after Platt scaling)
10-bin Expected Calibration Error (after Platt scaling)
Platt Scaling Coefficient
Platt Scaling Intercept

# Robustness
Quasi-exact match (perturbation: typos)
F1 (perturbation: typos)
Exact match (perturbation: typos)
RR@10 (perturbation: typos)
NDCG@10 (perturbation: typos)
Quasi-exact match (perturbation: synonyms)
F1 (perturbation: synonyms)
Exact match (perturbation: synonyms)
RR@10 (perturbation: synonyms)
NDCG@10 (perturbation: synonyms)

# Fairness
Quasi-exact match (perturbation: dialect)
F1 (perturbation: dialect)
Exact match (perturbation: dialect)
RR@10 (perturbation: dialect)
NDCG@10 (perturbation: dialect)
Quasi-exact match (perturbation: race)
F1 (perturbation: race)
Exact match (perturbation: race)
RR@10 (perturbation: race)
NDCG@10 (perturbation: race)
Quasi-exact match (perturbation: gender)
F1 (perturbation: gender)
Exact match (perturbation: gender)
RR@10 (perturbation: gender)
NDCG@10 (perturbation: gender)
Bias
Stereotypical associations (race, profession)
Stereotypical associations (gender, profession)
Demographic representation (race)
Demographic representation (gender)
Toxicity
Toxic fraction
Efficiency
Observed inference runtime (s)
Idealized inference runtime (s)
Denoised inference runtime (s)
Estimated training emissions (kg CO2)
Estimated training energy cost (MWh)

---
# General information
# eval
# train
truncated
# prompt tokens
# output tokens
# trials

---
# Summarization metrics
SummaC
QAFactEval
BERTScore (F1)
Coverage
Density
Compression
HumanEval-faithfulness
HumanEval-relevance
HumanEval-coherence

# APPS metrics
Avg. # tests passed
Strict correctness

# BBQ metrics
BBQ (ambiguous)
BBQ (unambiguous)

# Copyright metrics
Longest common prefix length
Edit distance (Levenshtein)
Edit similarity (Levenshtein)

# Disinformation metrics
Self-BLEU
Entropy (Monte Carlo)

# Classification metrics
Macro-F1
Micro-F1
```

### 4.2 重点指标速通

- **ECE (Expected Calibration Error，校准误差)**：模型说"我 90% 确定"时，是不是真有 90% 答对？把预测按置信度分箱，比较"平均置信度"与"实际准确率"之差。ECE 越小越可信。
- **pass@k**：代码题生成 $k$ 个候选，只要有一个通过全部单测就算对——衡量"采样足够多次能不能做对"。
- **扰动(perturbation)指标**：同一题加错别字/换同义词/换性别表述，看分数掉多少 → 鲁棒性与公平性的核心思想。

## 5. lm-evaluation-harness（学术界事实标准）

EleutherAI 的统一评测框架，几乎所有论文跑 MMLU/HellaSwag/ARC 等基准都用它，保证**口径一致、可复现**。

- 仓库：https://github.com/EleutherAI/lm-evaluation-harness

```
lm-eval 的价值 = 统一 "怎么问 + 怎么判分"
  ┌─ 同一个 prompt 模板(few-shot 数量固定)
  ├─ 同一种答案抽取规则(取 logprob 最大的选项)
  └─ 同一个 metric 实现
→ 不同模型/不同人跑出的分数才能横向比较
```

与 HELM 的分工：**HELM 重"全维度体检+可视化报告"，lm-eval-harness 重"任务多、接入易、复现强"**，二者常配合使用。

## 6. Chatbot Arena（成对盲测 + Elo）

LMSYS 的众包平台：用户同时和**两个匿名模型**对话，投票选更好的那个，再用 **Elo 评分**算排名。

- 平台：https://chat.lmsys.org/

### 6.1 为什么需要 Arena（原文观点保留）

> 尽管存在 HELM 和 lm-evaluation-harness 等基准，但由于**缺乏成对比较兼容性**，它们在评估**自由形式问题**时存在不足。这就是 Chatbot Arena 等众包基准发挥作用的地方。

```
自动指标的盲区              Arena 的解法
──────────────             ───────────────
开放式问答没有唯一答案   →  人类成对投票"哪个更好"
Exact Match 判不了"更有用" →  相对偏好，不需金标准
单题打分有标准漂移        →  成对比较抵消主观偏差
                            → 汇总成 Elo / Bradley-Terry 排名
```

### 6.2 Elo 直觉

两模型对战，**胜率**决定分差。若 A 比 B 高 400 分，理论上 A 赢的概率约 91%（每 400 分对应 10 倍胜率）：

$$P(A\ \text{胜}) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

投票样本越多，排名越稳。这套机制把"成千上万次盲测"压缩成一个可比较的天梯分数。

> 也常用 **LLM-as-a-judge**（如 GPT-4 当裁判，MT-Bench）替代部分人工，成本低、可扩展，但要警惕裁判模型自身偏见（偏好长答案、偏好自家风格）。

## 7. CLEVA（中文评测平台）

针对中文的 holistic 评测平台，可理解为"中文版 HELM"。

- 仓库：https://github.com/LaVi-Lab/CLEVA
- 官网：http://www.lavicleva.com/#/homepage/overview

中文评测的特殊性：分词、成语/古文理解、繁简、中文安全合规——通用英文榜单覆盖不到，需要 CLEVA / C-Eval / CMMLU 这类本地化基准。

## 实操 / 命令 / 资源汇总（原文真料一站式）

| 类别 | 资源 | 链接 |
|------|------|------|
| 入门综述 | How to Evaluate an LLM | https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/ |
| PPL 原理 | CSDN 困惑度详解 | https://blog.csdn.net/hxxjxw/article/details/107722646 |
| PPL 一致性 | vLLM vs HF Transformers | https://medium.com/@kimdoil1211/ensuring-consistency-comparing-vllm-and-hugging-face-transformers-4cd88b83ed3c |
| PPL+量化 | Weight Quantization | https://www.analyticsvidhya.com/blog/2025/01/neural-network-weight-quantization/ |
| vLLM 算 PPL | issue #1019 | https://github.com/vllm-project/vllm/issues/1019 |
| vLLM 算 PPL | issue #185 | https://github.com/vllm-project/vllm/issues/185 |
| 中文 ROUGE | rouge_chinese | https://github.com/Isaac-JL-Chen/rouge_chinese |
| 全维度榜单 | HELM | https://crfm.stanford.edu/helm/latest/ |
| 评测 harness | lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness |
| 人类盲测 | Chatbot Arena | https://chat.lmsys.org/ |
| 中文平台 | CLEVA | https://github.com/LaVi-Lab/CLEVA |

## 常见问题 / 坑

| 现象 / 疑问 | 原因 | 应对 |
|------------|------|------|
| 两个模型 PPL 不能直接比 | 分词器不同，N 的口径不一致 | 只在同 tokenizer / 同测试集内比，或改用 bits/byte |
| 量化后 PPL 涨了一点 | INT4/INT8 引入量化误差 | 看涨幅是否在可接受阈值(常用 <0.1~0.5)，配合下游任务验证 |
| PPL 很低但对话很差 | PPL 只测语料拟合，不测有用/安全 | 对齐模型改用 Arena/LLM-judge/下游任务评 |
| 中文 ROUGE 全是 0 或异常 | 没分词，整句当一个 token | 用 rouge_chinese + jieba 先切词 |
| Exact Match 判不了开放式问答 | 开放题没有唯一答案 | 用成对比较(Arena) 或 LLM-as-judge |
| HELM 跑得慢 | 多场景多扰动，计算量大 | 选子集场景，或用 lm-eval-harness 跑特定任务 |
| Arena 排名波动 | 投票样本不足 | 等票数累积，看置信区间而非单点分 |
| LLM-judge 偏心 | 裁判偏好长答案/自家风格 | 加位置交换、用多裁判投票、校正长度偏置 |
| 滑窗算 PPL 偏高 | 重叠 token 重复计入 loss | 重叠前缀 label 设 -100 不计分 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同主题评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 精度与压缩强相关：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 推理引擎(算 PPL/吞吐)：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 对齐影响精度口径：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 训练与微调：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 架构原理：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 内核优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件/算力：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 内存估算：[[docs/transformer内存估算]]
