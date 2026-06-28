# Firefly 微调

> Firefly（流萤）是一个**开源的中文大模型训练/微调项目**，以"用一套代码、低显存把主流中文 LLM 跑通指令微调"为目标，核心亮点是 **QLoRA + 多模型适配 + 数据 packing**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/peft/PEFT-API]] [[llm-train/chinese-llama-alpaca/README]]

仓库地址：<https://github.com/yangjianxin1/Firefly>

---

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | Firefly = QLoRA + 多模型 + packing |
| 1 | 地基：它解决什么问题 | 显存墙、中文指令数据、统一训练框架 |
| 2 | 整体架构与模块 | data / template / trainer / loss / merge |
| 3 | QLoRA 机制（为什么能省显存） | NF4、双重量化、分页优化器、LoRA 旁路 |
| 4 | 支持的模型与对话模板 | Qwen/Baichuan/ChatGLM/LLaMA/InternLM… |
| 5 | 数据格式与 firefly 指令集 | jsonl、多轮 conversation、loss mask |
| 6 | packing（多样本拼接） | 把短样本塞满 max_len，吞吐翻倍 |
| 7 | 训练循环 / loss 计算 | 只对 assistant 段算 loss |
| 8 | 启动脚本与配置项 | bootstrap.sh、train_args、deepspeed |
| 9 | 推理与权重合并 | adapter + base → 合并模型 |
| - | 典型配置示例（讲含义） | qlora json 字段逐个解释 |
| - | 常见问题 | 显存/复读/模板/合并 |

---

## 0. 一句话锚点

**Firefly = 一个"开箱即用"的中文 LLM 指令微调脚手架**：你准备好 `jsonl` 对话数据 + 一份训练配置 `json`，它就能用 **QLoRA**（4-bit 量化基座 + LoRA 旁路）在**单张消费级显卡**上微调 Qwen / Baichuan / ChatGLM / LLaMA 等模型，并通过 **packing** 把训练吞吐拉满。

可以把它理解成：**HuggingFace `Trainer` + `peft`(LoRA/QLoRA) + `bitsandbytes`(量化) + 一层"中文模型对话模板与数据处理"的胶水代码**。它没有发明新算法，价值在于**把这些零件正确地拼在一起并针对中文场景调好**。

---

## 1. 地基：它到底解决什么问题

要理解 Firefly，先看微调一个 7B/13B 模型直接撞到的三堵墙：

```
                ┌──────────────────────────────────────────┐
墙①  显存墙     │ 全参数微调 13B 需要的显存 ≈                │
                │   参数(fp16 26GB) + 梯度(26GB)            │
                │   + Adam 状态(m,v ≈ 52GB) ≈ 100GB+        │  → A100×多卡才行
                └──────────────────────────────────────────┘
                ┌──────────────────────────────────────────┐
墙②  数据墙     │ 中文指令/多轮对话数据格式不统一，          │
                │ loss 该算在哪些 token 上容易做错           │
                └──────────────────────────────────────────┘
                ┌──────────────────────────────────────────┐
墙③  适配墙     │ 每个国产模型对话模板(special token、角色   │
                │ 分隔符)都不一样，换模型要改一堆代码        │
                └──────────────────────────────────────────┘
```

Firefly 对应给出三把钥匙：

- **对墙①** → **QLoRA**：把基座权重量化到 4-bit（显存直接降到约 1/4），只训练极少量 LoRA 参数，让 **单卡 24GB 甚至 16GB 也能微调 7B/13B**。详见 [[llm-train/peft/LoRA-QLoRA]]。
- **对墙②** → **统一的 `conversation` 数据格式 + loss mask**：固定 jsonl 结构，框架自动只对"模型应该说的话（assistant 段）"计算 loss。
- **对墙③** → **template 注册表**：为每个模型登记一份对话模板，换模型只改配置里的 `model_name / template_name`，业务代码不动。

> 一句话：Firefly 把"省显存 + 标准化数据 + 多模型适配"这三件最容易踩坑的事封装好了。

---

## 2. 整体架构与核心模块

Firefly 本质是围绕 HF `Trainer` 搭的一圈"脚手架"，模块职责如下：

```
                       ┌─────────────────────────────────────────┐
配置 train_args.json → │              入口 train.py                │
                       │  解析参数 / 选模型 / 选模板 / 起 Trainer  │
                       └───────────────┬─────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐            ┌──────────────────┐           ┌──────────────────┐
│  component/   │            │   component/     │           │   component/     │
│  template     │            │   dataset        │           │   loss / collator│
│ (对话模板注册) │            │ (读 jsonl,packing)│           │ (只对 assistant  │
│  角色/分隔符   │            │  分词,loss_mask   │           │  段算 loss)      │
└───────┬───────┘            └────────┬─────────┘           └────────┬─────────┘
        │                            │                              │
        └────────────┬───────────────┴───────────────┬─────────────┘
                     ▼                               ▼
            ┌──────────────────┐            ┌──────────────────────┐
            │  bitsandbytes    │            │   peft (LoRA/QLoRA)  │
            │  4-bit 量化基座   │◄───插旁路──┤   只训练低秩矩阵 A,B  │
            └──────────────────┘            └──────────────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                          ┌──────────────────────┐
                          │  HF Trainer 训练循环  │
                          │  fwd→loss→bwd→optim   │
                          └──────────────────────┘
                                     │
                                     ▼
                      输出：LoRA adapter 权重 (几十~几百 MB)
                                     │
                          merge → 合并回 base，得到完整模型
```

核心模块（名称以仓库实际目录为准，思想稳定）：

| 模块 | 作用 | 关键点 |
|------|------|--------|
| 训练入口（train.py） | 串起整个流程 | 读 json 配置，决定 full / lora / qlora 模式 |
| template（对话模板） | 描述每个模型怎么拼对话 | system/user/assistant 的特殊 token 与分隔符 |
| dataset / collator | 读数据、分词、packing、补 mask | 决定哪些 token 参与 loss |
| loss | 自回归交叉熵 + mask | 只对回复段算 loss，详见第 7 节 |
| peft + bitsandbytes | 量化 + 低秩旁路 | QLoRA 的两大支柱 |
| merge / 推理脚本 | adapter 合并、加载推理 | 部署前的最后一步 |

---

## 3. QLoRA 机制：为什么能在单卡跑 13B

QLoRA = **Quantized LoRA**，它在普通 LoRA 之上再叠一层"把基座量化"。三层叠加缺一不可：

```
全参数微调      ：训练所有 W            → 显存爆炸
   │
   ▼ 加 LoRA：冻结 W，只训练旁路 ΔW=BA
LoRA           ：W(fp16) 冻结 + 小矩阵 B,A 可训
   │              基座仍是 fp16，13B 的 26GB 权重还在显存里
   ▼ 再加 4-bit 量化基座
QLoRA          ：W 以 4-bit(NF4) 存放(约 6.5GB) + B,A(fp16)可训
```

**(1) LoRA 旁路**：原始线性层 $h = Wx$，LoRA 改成

$$h = Wx + \frac{\alpha}{r}\,BAx,\qquad B\in\mathbb{R}^{d\times r},\;A\in\mathbb{R}^{r\times k},\; r\ll d,k$$

只训练 $A,B$。秩 $r$ 通常取 8/16/64，可训练参数往往不到全参的 1%。$\alpha$ 是缩放，等效学习率旋钮。

**(2) NF4 量化基座**：把冻结的 $W$ 用 **4-bit NormalFloat（NF4）** 存储。NF4 针对"权重近似正态分布"设计量化分位点，比普通 int4 精度更高。前向时按 block 反量化回 bf16 参与计算。

> 数值直觉：13B 权重 fp16 ≈ 26GB；NF4(4-bit) ≈ 26/4 ≈ **6.5GB**。这就是"单卡能装下"的关键。

**(3) 双重量化（Double Quantization）**：量化本身要存"缩放因子"，DQ 把这些缩放因子**再量化一次**，每参数再省约 0.3~0.5 bit。

**(4) 分页优化器（Paged Optimizer）**：用 NVIDIA 统一内存，在显存峰值（如长序列）时把优化器状态临时换到内存，避免 OOM 崩溃。

QLoRA 显存账（13B 量级，直觉量级，非精确值）：

```
基座 NF4         ~6.5 GB
LoRA 参数+梯度    ~小（< 1 GB，因为参数极少）
激活值/中间结果   随 batch×seq_len 变化（packing 会影响）
─────────────────────────────────
合计 ≈ 单张 24GB 卡可跑 13B QLoRA（开 gradient checkpointing 更稳）
```

更系统的对比见 [[llm-train/peft/LoRA-QLoRA]]，PEFT 各方法 API 见 [[llm-train/peft/PEFT-API]]。

---

## 4. 支持的模型与对话模板

Firefly 的一大卖点是"一套代码适配多模型"。**支持哪些、各模型模板细节以官方仓库 README/源码为准**，但思路是统一的：

```
常见适配模型（思路示意，具体清单看官方）:
  Qwen / Qwen1.5/2/2.5 系列
  Baichuan / Baichuan2
  ChatGLM2 / ChatGLM3
  LLaMA / LLaMA2 / 中文 LLaMA
  InternLM / Yi / Gemma / Mistral …
```

**为什么需要"对话模板"？** 因为每个模型在预训练/对齐时用的**角色分隔符和特殊 token 不一样**。比如同样一句"你好"，不同模型要拼成不同字符串：

```
模型A 模板:  <s>system\n...<sep>user\n你好<sep>assistant\n
模型B 模板:  [INST] 你好 [/INST]
ChatGLM 模板: [gMASK]<sop>...<|user|>\n你好<|assistant|>\n
```

Firefly 用一个 **Template 注册表**统一描述：system 提示、user/assistant 前后缀、停止 token 等。换模型时：

```
配置里改两处即可：
   model_name_or_path:  /path/to/Qwen2.5-7B
   template_name:       qwen     ← 决定怎么拼对话/在哪里算 loss
```

> 关键工程价值：**模板写错是新手最常见的坑**——会导致"训练时拼的格式"和"推理时的格式"不一致，模型学不会停、复读、答非所问。务必让训练模板与该模型官方对话模板一致。

---

## 5. 数据格式与 firefly 中文指令集

Firefly 同时也指**一个开源中文指令数据集**（firefly-train 系列，含多任务中文指令）。数据用 **jsonl**，一行一个样本，核心是 `conversation` 列表（支持多轮）：

```jsonc
// 一行 = 一个对话样本（多轮）
{
  "conversation_id": 1,
  "category": "Brainstorming",
  "conversation": [
    {"human": "帮我写一首关于秋天的诗", "assistant": "秋风起，落叶黄……"},
    {"human": "再短一点",            "assistant": "秋来叶落，凉意满城。"}
  ]
}
```

要点：

- **多轮**：`conversation` 是数组，第 2 轮能看到第 1 轮上下文。
- **human / assistant 角色**：框架据此拼模板、打 loss mask。
- **loss 只算 assistant**：human 段是"输入/上下文"，不应该让模型去拟合用户怎么提问（见第 7 节）。

数据流（从 jsonl 到一条训练样本）：

```
jsonl 一行
   │ 1. 解析 conversation
   ▼
[(human1,assist1),(human2,assist2)]
   │ 2. 套对话模板，拼成一个长字符串
   ▼
"<system>...<user>q1<assistant>a1<user>q2<assistant>a2"
   │ 3. tokenize → input_ids
   ▼
input_ids = [t0,t1,...,tn]
   │ 4. 生成 labels：human 段 = -100(忽略)，assistant 段 = 真 token
   ▼
labels    = [-100,-100,...,  a1_ids ,-100, ...,  a2_ids ]
   │ 5. (可选) packing：把多条短样本拼到 max_len
   ▼
送入 Trainer
```

---

## 6. packing：把短样本"塞满"，吞吐翻倍

**问题**：指令数据长短不一。若一条 80 token 的样本也补齐（pad）到 `max_len=1024`，那 ~94% 的算力都浪费在 pad token 上。

**packing 思路**：把多条短样本**首尾拼接**，凑满 `max_len` 再切一刀，几乎不浪费。

```
不开 packing（大量 pad，算力浪费）:
  样本1: [t t t][PAD PAD PAD PAD PAD PAD PAD ...]   ← 长度 1024，真内容很少
  样本2: [t t t t][PAD PAD PAD ...]

开 packing（拼满，几乎无 pad）:
  block1: [样本1 | 样本2 | 样本3 | 样本4...]  长度≈1024
  block2: [样本5 | 样本6 | 样本7 ...]         长度≈1024
        每个 block 内多条样本"挨着放"，labels 仍各自打 mask
```

收益与代价：

| 维度 | 不 packing | packing |
|------|-----------|---------|
| 有效 token 占比 | 低（大量 pad） | 高（接近 100%） |
| 训练吞吐 | 慢 | **快（常见 1.5~3×）** |
| 实现复杂度 | 简单 | 需正确处理边界 |
| 跨样本"串味"风险 | 无 | 需 attention 隔离或接受轻微影响 |

> 关键权衡：packing 把不相关样本拼到一个序列，理想情况应让 attention **不跨样本**（block-diagonal mask），否则样本 A 的结尾会"看到"样本 B 的开头。很多实现选择"接受这点轻微影响换吞吐"，是否启用隔离**以具体实现/版本为准**。无论如何，**labels 的 loss mask 必须按每条样本各自正确打**。

---

## 7. 训练循环与 loss：只对"该说的话"算损失

Firefly 复用 HF `Trainer` 的标准循环，差异在 **labels 的构造**（loss mask）。

```
for step, batch in dataloader:        # 一个 batch（可能已 packing）
    logits = model(batch.input_ids)   # 前向：4-bit 基座 + LoRA 旁路
    # 自回归：用第 i 个 token 预测第 i+1 个
    loss = CrossEntropy(
        logits[:, :-1, :],            # 预测
        labels[:, 1:],                # 目标（错位一格）
        ignore_index = -100           # human/system 段被忽略
    )
    loss.backward()                   # 只有 LoRA 的 A,B 有梯度
    optimizer.step()                  # 分页 AdamW 更新 A,B
    scheduler.step(); optimizer.zero_grad()
```

自回归语言建模目标（只在 assistant 段累加）：

$$\mathcal{L} = -\frac{1}{|M|}\sum_{i\in M}\log P_\theta(x_i \mid x_{<i}),\qquad M=\{\text{assistant 段 token 下标}\}$$

**为什么 human 段要 mask 成 -100？**
如果对 human 段也算 loss，模型会去"学习用户怎么提问"，这既浪费容量又偏离目标——我们要的是"给定问题，学会回答"。把 human/system 段标 `-100`，交叉熵直接跳过它们，梯度只来自 assistant 段。

```
input:  <user> 今天天气? <assistant> 晴，25 度 </s>
labels:  -100 -100 -100   -100      晴  ，25  度  </s>
                                    └────── 只在这里算 loss ──────┘
```

---

## 8. 启动脚本与关键配置项

仓库提供了若干 shell 入口（`bootstrap.sh` / `test_bash_getopts.sh` 等）来包装训练命令，便于在不同环境传"数据/模型/输出"路径。`bootstrap.sh` 用 `getopts` 解析四个路径参数：

```bash
# bootstrap.sh 的四个开关（取自脚本）
#   -d  TRAIN_DATASET_PATH   训练数据集路径
#   -p  PRE_MODEL_PATH       预训练(基座)模型路径
#   -o  MODEL_OUTPUT_PATH    模型/adapter 输出路径
#   -m  MODEL_METRICS_PATH   训练指标输出路径

sh bootstrap.sh -h          # 看帮助
sh bootstrap.sh -d /data/usw/web2 -p /opt/data/web2 -o /opt/data/web3 -m /opt/data/web4
```

真正的训练参数集中在**一份 `train_args/*.json` 配置**里（典型 QLoRA 配置字段，含义稳定，**具体字段名/默认值以官方为准**）：

| 配置项 | 含义 | 取值/权衡 |
|--------|------|-----------|
| `model_name_or_path` | 基座模型路径 | 决定加载哪个模型 |
| `template_name` | 对话模板 | 必须与基座官方模板一致，否则不收敛/复读 |
| `train_file` | 训练 jsonl | 见第 5 节格式 |
| `train_mode` | full / lora / qlora | 显存小用 qlora |
| `max_seq_length` | 序列最大长度 | 越长越占显存，配合 packing |
| `sft_packing` / `use_packing` | 是否开 packing | 开 → 吞吐↑（见第 6 节） |
| `lora_rank` (r) | LoRA 秩 | 8/16/64；大 → 容量↑、参数↑ |
| `lora_alpha` | LoRA 缩放 | 常设为 2×rank 量级 |
| `lora_dropout` | LoRA dropout | 防过拟合 |
| `lora_target_modules` | 注入旁路的层 | 常注入 q/k/v/o 及 mlp 投影 |
| `per_device_train_batch_size` | 单卡 batch | 受显存限制 |
| `gradient_accumulation_steps` | 梯度累积 | 小 batch 模拟大 batch |
| `learning_rate` | 学习率 | LoRA 常 1e-4~2e-4，比全参大 |
| `gradient_checkpointing` | 重算激活省显存 | 省显存、换时间 |
| `bf16` / `fp16` | 混合精度 | 新卡优先 bf16 |
| `deepspeed` | ZeRO 配置 | 多卡时分担显存 |

> 多卡/大模型时常配 **DeepSpeed ZeRO**（把优化器状态/梯度/参数分片到多卡），与 QLoRA 正交、可叠加，具体见仓库 deepspeed 目录与 [[00-知识地图]] 的并行训练章节。

---

## 9. 推理与权重合并

QLoRA 训练产出的是**很小的 LoRA adapter**（几十~几百 MB），不是完整模型。两种用法：

```
用法A：base(4-bit) + adapter 一起加载推理（省盘，但每次都要带 adapter）
  load base → load_adapter(adapter_dir) → generate

用法B：合并(merge) → 得到一个独立的完整模型，再正常部署
  base(fp16) + adapter →  W' = W + (α/r)·B·A  → 保存为完整权重
```

合并的本质就是把旁路加回主干：$W' = W + \frac{\alpha}{r}BA$。合并后推理无需 peft，部署最简单；但**合并要用 fp16 基座**做（不能在 4-bit 上直接合并后还指望无损），合并细节**以官方 merge 脚本为准**。

> 与 [[llm-train/chinese-llama-alpaca/README]] 的 `merge_llama_with_chinese_lora.py` 思路一致：都是"base + LoRA → 完整模型"，区别在 Chinese-LLaMA-Alpaca 还涉及**扩中文词表 + 二次预训练**，Firefly 更聚焦"指令微调脚手架"。

---

## 与同类项目对比

| 项目 | 定位 | 与 Firefly 关系 |
|------|------|-----------------|
| **Firefly** | 中文多模型 **指令微调脚手架** + 中文指令集 | 本文主角，QLoRA+packing+多模板 |
| [[llm-train/chinese-llama-alpaca/README]] | 给 LLaMA **扩中文词表 + 增量预训练 + SFT** | 更偏"把英文模型变中文"，含 PT 阶段 |
| [[llm-train/peft/PEFT-API]] | HuggingFace **PEFT 库本身** | Firefly 底层就用它实现 LoRA/QLoRA |
| LLaMA-Factory / Axolotl | 更大而全的微调框架 | 功能更广，Firefly 更轻、中文示例多 |

选型直觉：**只想快速把某个中文模型 QLoRA 微调跑通** → Firefly 很合适；**要扩词表/做继续预训练** → 看 Chinese-LLaMA-Alpaca；**只想理解 LoRA/QLoRA 原理与 API** → 看 PEFT 两篇。

---

## 常见问题

| 问题 | 原因 | 处理 |
|------|------|------|
| 单卡 OOM | 序列太长 / batch 太大 / 没开省显存 | 开 `qlora`+`gradient_checkpointing`，降 `max_seq_length`，加梯度累积 |
| 训练后模型复读 / 不停 | 对话模板与基座不一致，stop token 没学到 | 核对 `template_name` 与该模型官方模板严格一致 |
| loss 不降 / 学不会 | loss mask 错（对 human 段算了 loss）或学习率太小 | 确认只对 assistant 段算 loss；LoRA lr 用 1e-4~2e-4 |
| packing 后效果变差 | 跨样本注意力"串味" | 用 block-diagonal mask 隔离，或关 packing 对照 |
| 合并后精度掉 | 在 4-bit 基座上直接合并 | 用 fp16 基座 + adapter 合并 |
| 换模型要改一堆代码 | 没用模板机制 | 改 `model_name_or_path` + `template_name` 即可 |
| adapter 加载报维度不匹配 | base 与训练时的 base 不一致 | adapter 必须配对应同一基座版本 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航与训练/推理总图
- [[llm-train/peft/PEFT-API]] — PEFT 库 API（LoRA/QLoRA 怎么调）
- [[llm-train/peft/LoRA-QLoRA]] — LoRA 与 QLoRA 原理（NF4/双重量化/分页优化器）
- [[llm-train/chinese-llama-alpaca/README]] — 中文 LLaMA：扩词表 + 增量预训练 + SFT + 合并
- 官方仓库：<https://github.com/yangjianxin1/Firefly>（**支持模型清单/模板/字段默认值以此为准**）
