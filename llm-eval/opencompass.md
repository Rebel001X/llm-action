# OpenCompass 司南大模型评测体系

> OpenCompass（司南 2.0）是一套「数据集 × 模型 × 评测方式」三轴解耦的大模型能力评测框架：给一个模型，用上百个数据集、多种提示模板与打分器，量化出它在语言/知识/推理/考试/理解/长文本/安全/代码 8 个维度上的真实水平。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[llm-inference/vllm/README]] · [[llm-alignment/RLHF]]

## 阅读地图

| 你想知道 | 跳到 |
| --- | --- |
| 评测到底在测什么 | [0. 一句话锚点](#0-一句话锚点) |
| 为什么要有评测框架，benchmark 的本质 | [1. 地基：评测三要素](#1-地基评测三要素) |
| OpenCompass 的整体架构（怎么把模型跑成分数） | [2. 架构：从配置到分数的流水线](#2-架构从配置到分数的流水线) |
| 8 个能力维度分别测什么、有哪些数据集 | [3. 能力维度与数据集全景](#3-能力维度与数据集全景) |
| 客观题 vs 主观题怎么打分 | [4. 评测方式：客观打分 vs 主观打分](#4-评测方式客观打分-vs-主观打分) |
| 几个 shot、提示模板怎么影响分数 | [5. 提示工程对分数的影响](#5-提示工程对分数的影响) |
| 怎么真的跑起来（命令/配置） | [实操：安装与运行](#实操安装与运行) |
| 评测里最容易踩的坑 | [常见问题与坑](#常见问题与坑) |

## 0. 一句话锚点

> **评测 = 把「模型回答」和「标准答案」按规则比对，输出一个可比较的数字。** OpenCompass 干的事，就是把这个过程在 **上百个数据集 × 多个模型 × 多种打分规则** 上自动化、并行化、可复现地跑一遍。

一句话区分三个常被混淆的词：

- **数据集（Dataset）**：题目 + 标准答案，比如 GSM8K（小学数学应用题）。
- **指标（Metric）**：比对规则，比如 accuracy、pass@1、BLEU、ROUGE。
- **基准（Benchmark）**：固定下来的「数据集 + 指标 + 评测协议」组合，让不同模型的分数可横向对比。

## 1. 地基：评测三要素

要评一个大模型，本质上要回答三个问题，OpenCompass 把它们做成三条正交的轴：

```
        模型 (Model)              数据集 (Dataset)            评测方式 (Evaluator)
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │ 怎么拿到模型输出?  │  ×   │  题目从哪来?       │  ×   │  对错怎么判定?     │
   │ - HuggingFace 权重 │      │  - GSM8K / MMLU.. │      │  - 精确匹配        │
   │ - API (闭源模型)   │      │  - C-Eval / BBH.. │      │  - 选项抽取        │
   │ - vLLM/LMDeploy 加速│      │  - HumanEval...   │      │  - 代码执行/单测   │
   └──────────────────┘      └──────────────────┘      │  - LLM-as-Judge   │
                                                        └──────────────────┘
```

**为什么要三轴解耦？** 因为评测的痛点不是「测一次」，而是「测 N×M×K 次还要可复现」。如果模型、数据、打分逻辑耦在一起，换一个模型就要重写整套脚本。解耦后：

- 新增一个模型 → 只写一份 model config。
- 新增一个数据集 → 只写一份 dataset config（题目读取 + 提示模板 + 打分器）。
- 框架自动做笛卡尔积，把每个 (模型, 数据集) 组合排成一批推理任务。

> 类比：评测框架像「考场系统」。模型是考生，数据集是试卷，评测方式是阅卷标准。考场系统负责发卷、收卷、阅卷、登分，而不关心考生是谁、试卷是哪科。

## 2. 架构：从配置到分数的流水线

OpenCompass 把一次评测拆成 **划分（Partition）→ 推理（Infer）→ 评测（Eval）→ 汇总（Summarize）** 四个阶段：

```
  config (模型集 × 数据集集)
        │
        ▼
  ┌───────────┐   把 (模型,数据集) 笛卡尔积切成多个小任务
  │ Partition │   ── 大数据集再按样本数切片，便于并行
  └─────┬─────┘
        ▼
  ┌───────────┐   每个任务起一个进程/一张卡，调用模型生成答案
  │  Infer    │   ── 输出存到 predictions/  (原始回答落盘，可复用)
  └─────┬─────┘
        ▼
  ┌───────────┐   读 predictions，按 dataset 的 evaluator 打分
  │  Eval     │   ── 输出存到 results/      (每数据集一个分数)
  └─────┬─────┘
        ▼
  ┌───────────┐   把所有结果拼成一张大表 (模型为行，数据集为列)
  │ Summarize │   ── 输出 summary/  (csv/md/txt)
  └───────────┘
```

**为什么要把「推理」和「评测」分成两步并各自落盘？** 因为推理（跑模型）是最贵的一步（要 GPU、要时间）。把原始回答存进 `predictions/` 后：

- 你换一个打分器重新评，**不用再跑一遍模型**，直接读缓存。
- 推理中途崩了，可以断点续跑，已完成的任务跳过。
- 出了奇怪的分数，可以打开 `predictions/` 看模型到底答了什么，便于排错（这是排查评测 bug 的第一现场）。

**任务划分的意义**：MMLU 有上万道题，一张卡顺序跑要很久。Partition 把它切成多片，分发到多张卡/多进程并行，最后再合并——这就是评测框架相对「手写脚本」的核心价值之一。

## 3. 能力维度与数据集全景

OpenCompass 按 **8 个能力维度** 组织数据集。下面是原文给出的完整支持清单，按维度归类（这些是框架内置、开箱可用的数据集名）。

### 3.1 语言（Language）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 字词释义 | WiC、SummEdits | 同一个词在不同上下文是否同义 |
| 成语习语 | CHID | 中文成语完形填空 |
| 语义相似度 | AFQMC、BUSTM | 两句话是否同义 |
| 指代消解 | CLUEWSC、WSC、WinoGrande | 代词指向哪个名词 |
| 翻译 | Flores、IWSLT2017 | 机器翻译质量 |
| 多语种问答 | TyDi-QA、XCOPA | 跨语言问答 |
| 多语种总结 | XLSum | 跨语言摘要 |

### 3.2 知识（Knowledge）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 知识问答 | BoolQ、CommonSenseQA、NaturalQuestions、TriviaQA | 模型记住了多少世界知识（参数化知识） |

> 这一维度本质是在测「预训练语料里压进参数的事实性知识」。RAG（检索增强）正是为弥补这块短板而生——见 [[llm-inference/README]]。

### 3.3 推理（Reasoning）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 文本蕴含 | CMNLI、OCNLI、OCNLI_FC、AX-b、AX-g、CB、RTE、ANLI | 前提能否推出假设 |
| 常识推理 | StoryCloze、COPA、ReCoRD、HellaSwag、PIQA、SIQA | 物理/社会常识 |
| 数学推理 | MATH、GSM8K | 多步数学应用题 |
| 定理应用 | TheoremQA、StrategyQA、SciBench | 调用定理/科学知识解题 |
| 综合推理 | BBH（Big-Bench Hard） | 23 个高难子任务的硬骨头 |

> **GSM8K / MATH 是体现「思维链（CoT）增益」最明显的数据集。** 同一模型，不加 CoT 直接答 vs 加「Let's think step by step」后答，GSM8K 准确率常有数十个百分点的差距。这也是为什么评测必须固定提示模板（见第 5 节），否则分数不可比。

### 3.4 考试（Examination）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 初中/高中/大学/职业考试 | C-Eval、AGIEval、MMLU、GAOKAO-Bench、CMMLU、ARC、Xiezhi | 学科综合能力（多为多选题） |
| 医学考试 | CMB | 中文医学知识 |

> **MMLU（英文 57 学科）与 C-Eval / CMMLU（中文学科）是大模型「综合智力」的招牌指标**，几乎所有发布会都会报这几个分。它们是 4 选 1 的多选题，随机猜的基线准确率 = 25%，所以一个模型若 MMLU 只有 ~25% 说明几乎没学到东西。

### 3.5 理解（Understanding）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 阅读理解 | C3、CMRC、DRCD、MultiRC、RACE、DROP、OpenBookQA、SQuAD2.0 | 读一段文章回答问题 |
| 内容总结 | CSL、LCSTS、XSum、SummScreen | 抽取式/生成式摘要 |
| 内容分析 | EPRSTMT、LAMBADA、TNEWS | 情感/分类/末词预测 |

### 3.6 长文本（Long Context）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 长文本理解 | LEval、LongBench、GovReports、NarrativeQA、Qasper | 在很长上下文里检索/推理 |

> 长文本评测直接关联推理侧的 KV-Cache 与位置编码外推能力：见 [[llm-optimizer/kv-cache]] 与 [[llm-algo/旋转编码RoPE]]。「大海捞针（Needle-in-a-Haystack）」类测试就属于这一维度的代表玩法。

### 3.7 安全（Safety）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 安全 | CivilComments、CrowsPairs、CValues、JigsawMultilingual、TruthfulQA | 毒性/偏见/价值观/是否说真话 |
| 健壮性 | AdvGLUE | 对抗扰动下是否还稳 |

> **TruthfulQA 测「模型会不会一本正经地胡说」**——很多模型会顺着常见误解给出错误但流行的答案。安全与对齐紧密相关，是 RLHF/DPO 的优化目标之一：见 [[llm-alignment/RLHF]] 与 [[llm-alignment/DPO]]。

### 3.8 代码（Code）

| 子能力 | 数据集 | 测什么 |
| --- | --- | --- |
| 代码 | HumanEval、HumanEvalX、MBPP、APPs、DS1000 | 根据描述写出能通过单测的代码 |

> **代码评测与众不同：它不比对文本，而是真的执行生成的代码并跑单元测试。** 指标是 `pass@k`——下一节展开。

## 4. 评测方式：客观打分 vs 主观打分

同样一道题，「怎么判对错」决定了分数靠不靠谱。OpenCompass 支持多种打分器，核心分两大类：

```
                  ┌─ 客观评测 (有标准答案，可自动判) ──────────────┐
                  │   选择题  → 从模型输出里抽出 A/B/C/D 再比对      │
   评测方式 ───────┤   填空/QA → 精确匹配 / 包含匹配 / 归一化后匹配   │
                  │   翻译摘要 → BLEU / ROUGE 等重叠度指标          │
                  │   代码    → 执行 + 跑单测，统计 pass@k          │
                  └────────────────────────────────────────────┘
                  ┌─ 主观评测 (无唯一答案，要人或强模型评) ─────────┐
                  │   LLM-as-Judge → 用 GPT-4/更强模型当裁判打分     │
                  │   人工标注 / Arena 对战 → 两两胜率              │
                  └────────────────────────────────────────────┘
```

### 4.1 选择题为什么不能直接「字符串相等」

模型不会乖乖只吐一个「A」，它可能答「答案是 A，因为……」。所以选项抽取要用规则/正则从自由文本里**抠出选项字母**再比对。这一步没做好，会把答对的也判错，导致分数虚低——这是评测复现失败最常见的原因之一。

### 4.2 代码评测：pass@k 的数值直觉

`pass@k` 表示「给模型 k 次尝试，至少有一次通过全部单测」的概率。无偏估计公式（OpenAI HumanEval 论文）：

$$\text{pass@}k = \mathbb{E}_{\text{problems}}\left[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\right]$$

其中对每道题采样 $n$ 个解，其中 $c$ 个通过。

**手算一例**：某题采样 $n=10$ 个解，通过 $c=2$ 个，求 pass@1。

$$\text{pass@}1 = 1 - \frac{\binom{10-2}{1}}{\binom{10}{1}} = 1 - \frac{8}{10} = 0.2$$

即随便挑一个解，有 20% 概率通过——和直觉一致（10 个里 2 个对）。再看 pass@5：

$$\text{pass@}5 = 1 - \frac{\binom{8}{5}}{\binom{10}{5}} = 1 - \frac{56}{252} \approx 0.778$$

> 直觉：**给的尝试次数越多，pass@k 越高**（多挑几次更容易碰到对的）。所以比较模型时必须报同一个 k，pass@1 和 pass@10 不可混比。代码评测因为要真执行，务必在**沙箱/容器**里跑，防止生成的恶意代码破坏环境。

## 5. 提示工程对分数的影响

同一个模型，评测脚本不同，分数能差出十几分。关键变量：

| 变量 | 含义 | 对分数的影响 |
| --- | --- | --- |
| n-shot | 提示里给几个示例（few-shot） | 0-shot 通常显著低于 5-shot；MMLU 业界惯例报 5-shot |
| CoT | 是否引导「逐步思考」 | GSM8K/MATH 上加 CoT 可大幅提分 |
| 提示模板 | 题干如何拼成 prompt | 措辞、选项格式差异会让选项抽取成败不同 |
| 答案抽取规则 | 从输出里抠答案的正则 | 抠不准 → 答对也判错 |

```
   同一模型，同一 GSM8K 数据集
   0-shot 直接答            ▒▒▒▒▒░░░░░  ~低
   5-shot + CoT 示例        ▒▒▒▒▒▒▒▒▒░  ~高
   ↑ 分差可达数十个百分点 —— 所以「报分必须报评测设置」
```

> **核心原则：评测分数只有在「同样的提示设置」下才可比。** 看到一个 benchmark 排名，先问清楚它是几 shot、是否 CoT、用什么抽取规则。OpenCompass 把这些固化进 dataset config，正是为了保证可复现。

## 实操：安装与运行

> 以下为 OpenCompass 的标准用法范式，命令以官方仓库 [open-compass/opencompass](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md) 为准；具体子命令请对照你所装版本的文档。

**1）安装（建议独立 conda 环境，避免依赖冲突）**

```bash
conda create -n opencompass python=3.10 -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```

**2）准备数据集**（题目不在代码里，要单独下载解压到 `data/`）

```bash
# 下载官方打包好的数据集压缩包后解压到 ./data
# 解压后目录形如 data/<dataset_name>/...
```

**3）跑一次评测**（核心命令模式：指定模型 + 数据集）

```bash
# 用配置文件方式：configs/eval_xxx.py 里同时声明 models 和 datasets
python run.py configs/eval_demo.py

# 或命令行直传（HuggingFace 模型 + 内置数据集），便于快速试跑
python run.py \
  --datasets gsm8k_gen \
  --hf-path <huggingface模型路径> \
  --debug          # 单进程串行，方便看报错；正式跑去掉它以并行
```

命名约定要点：数据集 config 名常带后缀 **`_gen`（生成式）** 或 **`_ppl`（困惑度/逐选项打分）**——`gsm8k_gen` 表示让模型自由生成再抽答案，多选题常用 `_ppl` 比较各选项的困惑度。

**4）看结果**

```
outputs/<时间戳>/
├── predictions/   # 模型原始回答（排查问题的第一现场）
├── results/       # 每个数据集的分数
└── summary/       # 汇总大表 (csv / txt / md)
```

**用 vLLM / LMDeploy 加速推理**：评测动辄上万样本，原生 HF 生成很慢。OpenCompass 支持把后端切到 vLLM 等高吞吐推理引擎，吞吐可成倍提升——原理见 [[llm-inference/vllm/README]]。

## 常见问题与坑

| 现象 / 坑 | 根因 | 处理 |
| --- | --- | --- |
| 分数异常低（接近随机基线 25%） | 选项/答案抽取规则没匹配上模型输出格式 | 打开 `predictions/` 看原始回答，调整答案抽取正则或换 `_gen`/`_ppl` |
| 同一模型分数和别人对不上 | n-shot、是否 CoT、提示模板不同 | 报分必须连同评测设置一起说明，复现要用同一份 dataset config |
| 评测巨慢 | 用 HF 原生 generate 串行跑大数据集 | 切 vLLM/LMDeploy 后端 + 多卡 Partition 并行 |
| 改了打分器又要重跑一遍模型 | 没利用推理/评测分离 | 复用 `predictions/` 缓存，只重跑 Eval 阶段 |
| 代码评测把环境搞坏 / 卡死 | 直接在主机执行了模型生成的代码 | 必须放进沙箱/容器执行单测 |
| 找不到数据 / FileNotFound | 数据集没下载解压到 `data/` | 框架只含读取逻辑，题目需另行下载 |
| OOM | batch / 上下文长度对显存过大，长文本数据集尤甚 | 调小 batch、用量化模型、长文本任务单独配低并发 |
| pass@1 与 pass@10 混比 | 把不同 k 的代码分数横向比 | 比较时固定同一个 k |

> **一句话收尾**：OpenCompass 的价值不在「能算分」，而在「让上百个数据集 × 多个模型 × 多种打分规则的评测，变成一份可配置、可并行、可复现、可断点续跑的流水线」。评测做得对不对，决定了你对模型能力的判断准不准。

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同模块：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 模型架构（被评测对象的内核）：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 推理加速（评测吞吐的关键）：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 长文本相关：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 压缩量化（被评测的轻量模型）：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练与对齐（决定模型分数的来源）：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 估算：[[docs/transformer内存估算]]
