# LLM-QAT 复现实践日志（数据制备 / 训练 / 踩坑）

> facebookresearch/LLM-QAT 官方代码库的"动手复现"日志：从**数据来源选型（开源数据集 vs LLM 合成数据）**、环境搭建、合成数据生成、量化感知训练（QAT）到已知问题，一条龙记录。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/quantization/llm-qat/LLM-QAT]] [[llm-compression/quantization/量化基础]] [[llm-compression/quantization/SmoothQuant]] [[llm-train/pytorch/distribution/README]]

> 说明：本文是**实践复现笔记**，记录在自有机器上跑通 LLM-QAT 的过程与坑。算法原理（data-free 蒸馏、KV-cache 量化、Logit 蒸馏等）请看同目录 [[llm-compression/quantization/llm-qat/LLM-QAT]]。本文涉及的**具体脚本名、参数、目录结构**以官方仓库 `https://github.com/facebookresearch/LLM-QAT` 实际代码为准。

## 阅读地图

| 你想知道 | 跳到 |
|---|---|
| 这份日志在记录什么 | [§0](#0-一句话锚点) |
| QAT 为什么需要"训练数据"，从哪来 | [§1](#1-地基qat-为什么要数据数据从哪来) |
| 两种数据方案怎么选（开源 vs 合成） | [§2](#2-两种数据来源方案的取舍) |
| LLM 合成数据是怎么"自产自销"的 | [§3](#3-llm-合成数据自产自销的原理) |
| 环境怎么搭（torch / apex / venv） | [§4](#4-环境搭建虚拟环境--apex) |
| 合成数据多卡并行生成 + 合并 | [§5](#5-合成数据生成与合并多卡并行) |
| 训练阶段在做什么（QuantizeLinear / FSDP） | [§6](#6-训练阶段quantizelinear--fsdp) |
| 完整复现流程一图流 | [§7](#7-端到端复现流程一图流) |
| 版本变更记录（commit 时间线） | [版本时间线](#版本时间线本仓库记录) |
| 踩过的坑 | [常见问题/坑](#常见问题坑) |

---

## 0. 一句话锚点

**这份 `log.md` 是 LLM-QAT 官方代码的"复现实践流水账"**：核心结论是——LLM-QAT 训练阶段需要一批文本数据来做"量化学生网络 ← 全精度教师网络"的蒸馏，数据有两条路：**(A) 直接用开源公开数据集**；**(B) 用全精度模型自己"生成"一批文本作为 data-free 蒸馏语料**。本仓库的两个代码版本分别对应这两种能力。

```
                  本目录文件分工
  ┌──────────────────────────────────────────────────┐
  │ LLM-QAT.md  →  论文/算法原理（方法、消融、结论）       │
  │ README.md   →  环境/命令/模型结构（怎么跑）            │
  │ log.md (本文) → 复现流水账（数据选型/版本/踩坑）  ★你在这 │
  │ cfd70ff/    →  第一版代码快照（仅支持开源数据集）       │
  │ f4d873a/    →  第二版代码快照（新增 LLM 合成数据）      │
  └──────────────────────────────────────────────────┘
```

---

## 1. 地基：QAT 为什么要数据？数据从哪来

先把最底层的逻辑拆清楚。

**PTQ（训练后量化）** 不需要"训练"，最多用几百条样本做**校准（calibration）**——统计激活的数值范围，定出缩放因子 $\alpha$ 即可。代价小，但**低比特（如 4bit）会崩**。

**QAT（量化感知训练）** 不一样：它要在**前向里插入"伪量化"算子**，让网络"带着量化误差"再训练一段，把误差通过反向传播学回来。既然要"训练"，就必须有：

1. **一批输入文本**（喂给网络的 $X$）；
2. **一个监督信号**（告诉网络输出该长什么样）。

LLM-QAT 的关键设计是：监督信号**不用人工标签**，而是用**全精度教师模型的 logits**（知识蒸馏）。于是问题只剩——**那批输入文本从哪来？**

```
   QAT 训练一步的数据流（蒸馏视角）
   ┌─────────┐  X(文本)  ┌──────────────┐  教师 logits
   │  语料 X  │─────────▶│ 全精度教师模型 │───────────┐
   └─────────┘    │      └──────────────┘           │
                  │                                  ▼
                  │      ┌──────────────┐        交叉熵蒸馏
                  └─────▶│ 量化学生模型   │── logits ─▶ L_CE  ──► 反向更新
                         │ (QuantizeLinear)│
                         └──────────────┘
   ★ 问题核心：左上角的"语料 X"用什么？→ 这就是本文要讨论的"数据来源"。
```

> 量化公式（MinMax，详见 [[llm-compression/quantization/量化基础]]）：
> $\mathbf{X}_Q = \alpha \left\lfloor \dfrac{\mathbf{X}_R - \beta}{\alpha} \right\rceil + \beta,\quad \alpha=\dfrac{\max(|\mathbf{X}_R|)}{2^{N-1}-1},\ \beta=0$

---

## 2. 两种数据来源方案的取舍

本仓库 `log.md` 原始记录的核心就是这一句：**训练数据来源有两个方案**。

| 维度 | 方案一：开源公开数据集 | 方案二：LLM 合成数据（data-free） |
|---|---|---|
| 来源 | 现成语料（如通用 web 文本语料） | 用**全精度预训练模型自己生成** |
| 是否依赖原训练数据 | 否（用别的开源数据替代） | 否（完全 data-free，更彻底） |
| 分布匹配度 | 可能与教师模型预训练分布**有偏差** | **天然贴近**教师模型自身分布 ★ |
| 数据获取成本 | 低（下载即用） | 需要一次性推理生成（耗 GPU） |
| 适用场景 | 快速起步、数据可得 | 拿不到原数据、追求保真输出分布 |
| 论文倾向 | 基线 | **主推**：更好保留原始输出分布 |

**为什么论文主推合成数据？** 直觉是：你要让量化学生"模仿"教师，最好的"考题"就是教师**自己擅长、自己会说的话**。用教师自己采样出来的文本做蒸馏语料，能更忠实地复刻它的输出分布，并且**不依赖原始训练数据**（很多大模型的训练数据是闭源的，拿不到）。这正是 LLM-QAT 标题里 *data-free* 的含义。

```
   分布匹配示意（为什么合成数据更"对味"）
   开源数据集          教师模型分布
   ░░░░▓▓▓░░░    vs    ▒▒▒████▒▒▒
        └─ 偏移 ─┘          └─重合─┘
   开源语料的分布不一定罩住教师；
   教师自生成的语料 = 直接采样自教师分布，重合度更高。
```

---

## 3. LLM 合成数据"自产自销"的原理

"用 LLM 生成数据"听起来玄，拆到原子其实就是**自回归采样**：

1. 给一个起始 token（或几个起始 token）作为 prompt；
2. 全精度教师模型预测下一个 token 的概率分布；
3. **从分布里采样**（而不是永远取 top-1！）下一个 token；
4. 把采到的 token 接回输入，回到第 2 步，直到生成够长。

```
  自回归生成一条合成样本
  [起始tok] ─► 教师 ─► P(next) ─采样─► tok1
              tok1 ─► 教师 ─► P(next) ─采样─► tok2
              tok2 ─► 教师 ─► P(next) ─采样─► tok3  ...  → 拼成一条文本
                                                          存成 jsonl 一行
```

**为什么强调"采样"而不是"贪心取 top-1"？** 论文专门指出：
- 若总取 top-1，生成的文本会**高度重复、多样性差**，覆盖不到教师分布的尾部；
- 引入采样 → 文本更多样 → 蒸馏语料覆盖面更广。
- 但采样本身带噪声，所以**下一个 token 不能当硬标签**用——这正是要配合 **Logit 蒸馏（用教师完整 logits 做软标签）** 的原因（详见 [[llm-compression/quantization/llm-qat/LLM-QAT]] §3.3.3）。

> 合成数据格式：每条样本一行 JSON，形如 `{"text": "...生成的一段文本..."}`，落盘为 `.jsonl`。

---

## 4. 环境搭建：虚拟环境 + apex

> 以下命令为本仓库 README 记录的"机制级"步骤，**确切 wheel 文件名、CUDA/torch 版本号请以官方仓库 `requirements` 与你机器的 CUDA 为准**，不要照抄版本号。

核心三件套：**独立 Python 虚拟环境** + **匹配 CUDA 的 PyTorch** + **NVIDIA apex（混合精度/融合算子）**。

```bash
# 1) 建独立虚拟环境（隔离依赖，避免污染系统 python）
virtualenv -p /usr/bin/python3.10 llm-qat-venv
source .../llm-qat-venv/bin/activate

# 2) 装与本机 CUDA 匹配的 torch（cuXXX 要对上你的驱动）
pip install torch-<ver>+cu<XXX>-...whl
pip install torchvision-<ver>+cu<XXX>-...whl

# 3) 从源码编译 apex（带 C++/CUDA 扩展）
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./

# 4) 项目依赖
pip install -r requirement.txt
```

**为什么要 apex？** apex 提供融合的优化器/归一化算子和混合精度支持，QAT 训练大模型时能省显存、提吞吐。**坑点**：apex 必须**带 CUDA 扩展编译**（`--cuda_ext`），且 `pip` 版本不同，传扩展编译参数的写法不同（新版用 `--config-settings`，旧版用 `--global-option`）——传错就退化成纯 Python apex 甚至装不上。

---

## 5. 合成数据生成与合并（多卡并行）

生成 12000 条左右合成样本，做法是**按 GPU 分片并行**，每张卡跑一个分片编号，最后合并。

```
   多卡并行生成（数据并行思路：切分而非协作）
   GPU0 ─ generate_data.py 0 ─► gen.chunk.00.jsonl ┐
   GPU1 ─ generate_data.py 1 ─► gen.chunk.01.jsonl │
   GPU2 ─ generate_data.py 2 ─► gen.chunk.02.jsonl │  merge_gen_data.py
   ...                                   ...        ├──────────────► all_gen.jsonl
   GPU7 ─ generate_data.py 7 ─► gen.chunk.07.jsonl ┘   (≈12000 行)
   每卡独立生成 1500 条 → 8 卡 ≈ 12000 条
```

命令骨架（确切脚本名以仓库为准）：

```bash
mkdir -p gen_data/
# 每张卡负责一个分片编号，互不干扰
CUDA_VISIBLE_DEVICES=0 python generate_data.py 0
CUDA_VISIBLE_DEVICES=1 python generate_data.py 1
# ... 一直到 7
CUDA_VISIBLE_DEVICES=7 python generate_data.py 7

# 合并所有分片为一个总文件
python merge_gen_data.py        # → all_gen.jsonl
wc -l all_gen.jsonl             # 约 12000 行
head -n1 gen_data/gen.chunk.02.jsonl
# {"text": "...一段教师采样生成的文本..."}
```

**为什么按 GPU 切分而不是单卡跑？** 自回归生成是**串行**的（token 一个个出），单卡生成 12000 条很慢；按分片编号分到 8 卡**并行**，墙钟时间约缩到 1/8。这是典型的"**embarrassingly parallel**（无依赖可完美并行）"任务——各分片之间没有任何通信需求，所以不需要 NCCL/集合通信。

---

## 6. 训练阶段：QuantizeLinear + FSDP

训练时，模型里所有的线性层被换成 **`QuantizeLinear`**——这就是"伪量化算子"的载体：前向时对权重/激活做量化-反量化，反向时用 STE（直通估计）让梯度照常流过。

```
  LlamaDecoderLayer（QAT 改造后）
  self_attn:
    q_proj / k_proj / v_proj / o_proj  →  QuantizeLinear  (原本是 nn.Linear)
  mlp:
    gate_proj / up_proj / down_proj    →  QuantizeLinear
  其余（RMSNorm / RotaryEmbedding / 词表 Embedding / lm_head）保持原样
        │
        ▼  多机/多卡训练时，每个 DecoderLayer 再被 FSDP 包一层：
  FullyShardedDataParallel( FlattenParamsWrapper( LlamaDecoderLayer(...) ) )
```

两个要点：
- **只量化线性层**：注意力的 4 个投影 + MLP 的 3 个投影换成 `QuantizeLinear`；归一化、旋转位置编码、`lm_head` 通常保持高精度（它们对量化敏感且占比小）。
- **FSDP 分片**：训练 7B/13B/30B 这种规模，单卡放不下，用 **FSDP（完全分片数据并行）** 把每层参数切到多卡。原理见 [[llm-train/pytorch/distribution/README]]。

启动训练（脚本名/参数顺序以仓库为准，常见形如 `W A KV` 三个比特位）：

```bash
sh run_train.sh 8 8 8        # W8A8KV8
sh run_train_chunk.sh 8 8 8  # 分块读合成数据版本
```

> 训练产物：微调后的量化模型权重（如 `7B-finetuned/` 下的分片 `.bin` + index.json）+ TensorBoard 事件文件。

---

## 7. 端到端复现流程（一图流）

```
  ┌─ 0. 选数据来源 ──────────────────────────────────┐
  │   方案A 开源数据集            方案B LLM 合成数据      │
  │      │                          │                  │
  │   下载语料                generate_data.py × 8卡    │
  │      │                          │ → chunk.*.jsonl  │
  │      │                    merge_gen_data.py        │
  │      │                          │ → all_gen.jsonl  │
  └──────┴──────────────┬───────────┘                  │
                        ▼
  ┌─ 1. 环境 ─────────────────────────────────────────┐
  │   venv + torch(cuXXX) + apex(带cuda_ext) + 依赖     │
  └───────────────────────┬───────────────────────────┘
                          ▼
  ┌─ 2. 训练（QAT 蒸馏）──────────────────────────────┐
  │   全精度教师 ─logits─┐                              │
  │   语料X ─►           ├─► L_CE 蒸馏 ─► 更新量化学生   │
  │   量化学生(QuantizeLinear)+FSDP 分片                │
  │   run_train.sh  W A KV                            │
  └───────────────────────┬───────────────────────────┘
                          ▼
  ┌─ 3. 产物 ─────────────────────────────────────────┐
  │   7B-finetuned/*.bin + index.json + TB 日志        │
  │   ⚠ 评估/推理速度测量需自行补（见踩坑表）            │
  └───────────────────────────────────────────────────┘
```

---

## 版本时间线（本仓库记录）

| 时间 | commit | 变更内容 | 含义 |
|---|---|---|---|
| — | `cfd70ff` | 第一版 | 仅支持**开源公开数据集**（方案一） |
| 2023-08-03 | — | 记录 | 确认代码当前支持开源数据集训练 |
| 2023-08-14 | `f4d873a` | 第二版 | 新增**LLM 合成数据**能力（方案二，data-free） |

> 即：`cfd70ff/` 与 `f4d873a/` 两个子目录是这两个版本的代码快照，对应"开源数据 → 合成数据"的能力演进。

---

## 常见问题/坑

| 现象 / 坑 | 原因 | 应对 |
|---|---|---|
| 代码**没法直接跑通** | 官方代码偏研究性质，路径/参数硬编码多 | **需自行改源码**（模型路径、数据路径、启动参数）后再跑 |
| 缺**模型评估 / 推理速度**代码 | 仓库只在训练中用 HF `evaluate` 评效果 | 只能评"精度"，**测不了推理速度/吞吐**，需自己写 benchmark |
| apex 装不上 / 没用上 CUDA | 没带 `--cuda_ext` 编译，或 pip 版本导致参数写法错 | 新 pip 用 `--config-settings`，旧 pip 用 `--global-option`；确认带 `--cpp_ext --cuda_ext` |
| 合成数据多样性差、重复 | 生成时用了贪心 top-1 | 改为**从分布采样**下一个 token（论文明确要求） |
| 把"下一个 token"当硬标签训 | 采样引入噪声，硬标签非最优 | 用 **Logit 蒸馏**（教师完整 logits 当软标签） |
| 单卡生成 12000 条太慢 | 自回归生成串行 | 按 GPU **分片并行**（每卡一个 chunk 编号）再 merge |
| torch / CUDA 版本不匹配 | wheel 的 `cuXXX` 与本机驱动不符 | 按本机 CUDA 选 wheel，**不要照抄文档里的版本号** |
| 4bit 直接部署无硬件加速 | 论文成文时 4bit 缺开箱即用硬件支持 | 当时未做硬件实现；落地需结合后续推理框架/内核 |

> 提示：本文的具体脚本名、参数顺序、目录结构均为"机制级"描述，**确切命名以 `github.com/facebookresearch/LLM-QAT` 实际源码为准**。

---

## 🔗 跳转链接

- 算法原理（必读配套）：[[llm-compression/quantization/llm-qat/LLM-QAT]]
- 量化通用地基：[[llm-compression/quantization/量化基础]]
- 兼容技术（权重-激活重缩放）：[[llm-compression/quantization/SmoothQuant]]
- 大规模训练的分片并行：[[llm-train/pytorch/distribution/README]]
- 压缩总览：[[llm-compression/README]]
- 知识地图：[[00-知识地图]]
