# HuggingFace TRL (对齐训练库)

> 一句话定位：TRL（Transformer Reinforcement Learning）是 HuggingFace 官方的"对齐后训练全家桶"，把 SFT → 奖励建模 → RLHF(PPO) / DPO / GRPO 等一整套"让模型听话"的训练范式，封装成与 `transformers`/`peft`/`accelerate` 无缝衔接的 `Trainer` 接口。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-alignment/RLHF]] [[llm-alignment/DPO]] [[ai-framework/huggingface-peft/README]]

---

## 阅读地图

| 序号 | 小节 | 你将搞懂的核心问题 | 难度 |
|------|------|--------------------|------|
| 0 | 一句话锚点 | TRL 到底解决"预训练之后"的什么问题 | ★ |
| 1 | 地基/前置 | 语言模型、损失、RL 三个最小概念 | ★ |
| 2 | 整体架构 | TRL 在 HF 生态里站在哪一层 | ★★ |
| 3 | 对齐三步走 | SFT→RM→RLHF 的全景数据流 | ★★ |
| 4 | SFTTrainer | 监督微调，所有对齐的起点 | ★★ |
| 5 | RewardTrainer | 用偏好对训练打分器（奖励模型） | ★★★ |
| 6 | PPOTrainer | 经典 RLHF，最复杂但最经典 | ★★★★ |
| 7 | DPOTrainer | 跳过奖励模型，直接用偏好优化策略 | ★★★ |
| 8 | GRPOTrainer | DeepSeek 同款，去掉 Critic 的群体相对优化 | ★★★★ |
| 9 | 与 PEFT/Accelerate 集成 | 单卡/多卡/LoRA 怎么省显存 | ★★★ |
| 10 | 数值例子/对照 | 手算 DPO/GRPO 损失，五大 Trainer 横评 | ★★★ |
| - | 常见问题 | OOM、不收敛、reward hacking… | ★★ |

---

## 0. 一句话锚点

一个基座大模型（base model）刚预训练完时，它只会"接着往下写"，不会"回答问题"，更不会"拒绝有害请求"。**TRL 就是把这块"毛坯模型"装修成"听人话、对齐人类偏好的助手"的施工队。**

它对应的是 LLM 训练的**后训练（post-training）/ 对齐（alignment）** 阶段：

```
        预训练                 后训练 = TRL 的地盘
   ┌──────────────┐    ┌───────────────────────────────────┐
   │  海量文本     │    │  SFT  →  奖励建模  →  RLHF/DPO/GRPO │
   │  自回归预测   │ →  │ 学格式    学偏好      学"更受欢迎"   │
   │ (会写)        │    │ (会答)   (会打分)     (会对齐)       │
   └──────────────┘    └───────────────────────────────────┘
     transformers              TRL
```

---

## 1. 地基/前置（把概念拆到最原子）

读 TRL 前，只需要三个最小积木。

### 1.1 语言模型在算什么

一个自回归语言模型，本质是对"下一个 token"的条件概率分布：给定前文 $x_{<t}$，输出词表上每个候选 token 的概率 $\pi_\theta(x_t \mid x_{<t})$。整句话的概率是连乘：

$$\pi_\theta(y \mid x) = \prod_{t=1}^{|y|} \pi_\theta(y_t \mid x, y_{<t})$$

其中 $x$ 是 prompt，$y$ 是模型生成的回答。$\theta$ 是模型参数。**"训练模型"就是改 $\theta$，让我们想要的句子概率变大。**

### 1.2 监督训练的损失：交叉熵

给一个目标句子，最简单的训练就是让模型对每个正确 token 的预测概率尽量接近 1。用负对数似然（交叉熵）：

$$\mathcal{L}_{\text{CE}} = -\sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})$$

概率越高，$-\log$ 越小，损失越小。这就是 SFT 的全部数学内核。

### 1.3 强化学习的最小词汇

RLHF 用到 RL 的几个词，先建立直觉：

| RL 术语 | 在 LLM 里的对应 | 大白话 |
|---------|----------------|--------|
| 策略 policy $\pi_\theta$ | 被训练的语言模型 | "下棋的人" |
| 动作 action | 生成一个 token | "落一子" |
| 状态 state | 当前已生成的前缀 | "当前棋盘" |
| 奖励 reward $r$ | 奖励模型给整句话打的分 | "这局赢没赢" |
| 价值 value / critic | 估计"从此处还能拿多少分" | "形势判断" |

**核心矛盾**：交叉熵需要"标准答案"，但"什么回答更好"往往没有唯一标准答案，只有"A 比 B 好"的相对偏好。RLHF/DPO/GRPO 就是为了能从**偏好**而不是**标准答案**里学习。

---

## 2. 整体架构：TRL 站在 HF 生态的哪一层

TRL 不重造轮子。它把模型加载、分布式、显存优化全部委托给底层库，自己只负责"对齐算法逻辑"。

```
┌──────────────────────────────────────────────────────────┐
│                     你的训练脚本 / CLI                      │
├──────────────────────────────────────────────────────────┤
│   TRL：对齐算法层                                          │
│   SFTTrainer  RewardTrainer  PPOTrainer  DPOTrainer        │
│   GRPOTrainer  +  AutoModelForCausalLMWithValueHead 等     │
├───────────────┬──────────────────────┬────────────────────┤
│  transformers │      peft            │     accelerate      │
│  (模型/分词器) │  (LoRA/QLoRA 等)     │ (多卡/混合精度/DS)  │
├───────────────┴──────────────────────┴────────────────────┤
│              PyTorch  +  DeepSpeed / FSDP                  │
├──────────────────────────────────────────────────────────┤
│                       GPU 硬件                             │
└──────────────────────────────────────────────────────────┘
```

设计要点（这是 TRL 的"卖点"）：

- **继承 `transformers.Trainer`**：所有 `SFTTrainer/DPOTrainer/...` 都是 `Trainer` 的子类，因此自动获得 checkpoint、日志、评估、`push_to_hub`、`accelerate launch` 多卡等能力。你会的 `Trainer` 用法直接复用。
- **配置即接口**：每个 Trainer 配一个 `XxxConfig`（如 `SFTConfig`、`DPOConfig`，均继承自 `TrainingArguments`），超参全在里面。
- **数据约定优先**：你只要把数据整成约定字段（如 SFT 的 `messages`/`text`、偏好的 `chosen`/`rejected`），TRL 自动处理模板套用与 tokenize。

> 具体类名、参数、默认值请以你安装版本的官方文档为准（TRL 迭代很快，API 在不同版本间会有调整）。本文聚焦"原理与心智模型"，这部分跨版本稳定。

---

## 3. 对齐三步走：全景数据流

经典 RLHF 是三段流水线，TRL 的五个 Trainer 正好覆盖这条线（DPO/GRPO 是对后两步的"压缩")。

```
 阶段1: SFT (监督微调)
   人写的高质量 (prompt, answer) 对
        │  交叉熵
        ▼
   π_sft  ──────────────┐  (作为后续的"参考模型 π_ref" / 初始化策略)
                        │
 阶段2: RM (奖励建模)    │
   人标注的偏好对         │
   (prompt, chosen ≻ rejected)
        │  pairwise loss │
        ▼                │
   奖励模型 r_φ           │
                        │
 阶段3: RLHF / 对齐       ▼
   ┌─────────────────────────────────────────┐
   │ 路线A (PPO):  策略采样 → r_φ 打分 → PPO 更新 │
   │ 路线B (DPO):  直接用偏好对 → 闭式损失更新     │ ← 不需要 r_φ
   │ 路线C (GRPO): 一题采多答 → 组内相对优势 → 更新 │ ← 不需要 critic
   └─────────────────────────────────────────┘
        │
        ▼
   对齐后的模型 π_aligned
```

**关键洞察：DPO 和 GRPO 都是对"阶段2+阶段3"的简化。** DPO 用数学把奖励模型"消掉"了；GRPO 保留 reward（可以是规则/RM），但用"组内比较"替代了价值网络（critic）。下面逐个拆。

---

## 4. SFTTrainer：监督微调，一切对齐的起点

### 4.1 解决什么问题

让"只会续写"的 base 模型学会"对话格式 + 跟随指令"。本质就是在高质量 `(指令, 回答)` 数据上做 §1.2 的交叉熵训练，但 TRL 帮你处理了三件麻烦事：套对话模板、只对"回答部分"算 loss（completion-only）、packing 拼接短样本提效率。

### 4.2 数据流与 loss masking（最易踩坑处）

```
原始样本: {"messages":[{"role":"user","content":"几大洲?"},
                      {"role":"assistant","content":"七大洲。"}]}
   │ 套 chat_template
   ▼
"<|user|>几大洲?<|assistant|>七大洲。<eos>"
   │ tokenize + 构造 labels
   ▼
input_ids:  [<|user|> 几 大 洲 ? <|assistant|> 七 大 洲 。 <eos>]
labels:     [ -100  -100 ... -100      七   大  洲  。  <eos>]
                └─ prompt 部分 mask 掉 ─┘ └── 只对回答算 CE ──┘
```

`labels = -100` 的位置不计入损失。**为什么只对回答算 loss**：我们不希望模型去"学习预测用户的问题"，只想让它学"给定问题该怎么答"。这一步配置错（对 prompt 也算 loss）是新手最常见的"训练了但效果怪"的原因。

### 4.3 心智模型

`SFTTrainer` ≈ `transformers.Trainer` + 自动套模板 + 自动 loss masking + 可选 packing + 一行接 LoRA。配置走 `SFTConfig`（继承 `TrainingArguments`，多了 `max_length`、`packing` 等字段）。

---

## 5. RewardTrainer：把"偏好"变成"分数"

### 5.1 解决什么问题

人类很难给"好回答"打绝对分（80 分还是 85 分？），但很容易判断"A 比 B 好"。奖励模型（RM）的任务：**学一个打分函数 $r_\phi(x,y)$，使得人类更喜欢的回答得分更高。**

### 5.2 结构与损失

拿 SFT 模型，把最后的"词表预测头"换成一个**输出标量的回归头**（value head），输入 `(prompt, response)`，输出一个实数分。

```
   (prompt, chosen )──► RM ──► r_chosen   (要它更大)
   (prompt, rejected)──► RM ──► r_rejected (要它更小)

   loss = -log σ( r_chosen − r_rejected )      σ 是 sigmoid
```

这是 **Bradley-Terry 偏好模型**：两回答得分差越大、且方向正确，sigmoid 越接近 1，loss 越接近 0。

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big]$$

$y_w$=chosen(win)，$y_l$=rejected(lose)。**注意：只有"差值"有意义，绝对分可以整体平移，所以 RM 分数不能跨 prompt 直接比较大小。** 数据字段约定为 `chosen` / `rejected`。

---

## 6. PPOTrainer：经典 RLHF，最复杂但最经典

### 6.1 解决什么问题

有了打分器 $r_\phi$，怎么把"高分"反传给策略？不能直接交叉熵（没有标准答案），用 RL：**让模型多生成、对高分回答的概率往上推、低分往下压**，同时别跑太偏。PPO（近端策略优化）是这一步的主力算法。

### 6.2 四个模型同台（PPO 显存大的根因）

```
            ┌──────────── prompt x ───────────┐
            ▼                                  ▼
   ┌─────────────────┐               ┌──────────────────┐
   │ Policy π_θ      │── 生成 y ──►   │ Reward r_φ       │── r(x,y)
   │ (训练，带value头)│               │ (冻结)           │
   └────────┬────────┘               └──────────────────┘
            │ logprob                          │
            ▼                                  ▼
   ┌─────────────────┐    KL 惩罚      ┌──────────────────┐
   │ Reference π_ref │◄───────────────│  组合奖励 R       │
   │ (冻结=SFT初值)  │                │  R = r − β·KL     │
   └─────────────────┘                └──────────────────┘
   Critic/Value(常与policy共享主干)：估计基线，降方差
```

四个角色：**策略**（训）、**参考模型**（冻结，防跑偏）、**奖励模型**（冻结，打分）、**价值网络/Critic**（训，估基线）。这就是 PPO 又慢又吃显存的原因——同时驻留多个模型。TRL 用 `AutoModelForCausalLMWithValueHead` 把 policy 和 value head 合在一个主干上以省显存。

### 6.3 为什么要 KL 惩罚

只追高分，模型会"钻奖励模型的空子"（reward hacking），输出怪话却拿高分。加一项"别离参考模型太远"的约束：

$$R(x,y) = r_\phi(x,y) - \beta \cdot \mathrm{KL}\big[\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big]$$

$\beta$ 是拉力系数：$\beta$ 大→更保守(贴近 SFT)，$\beta$ 小→更激进(更敢优化分数但易崩)。

### 6.4 PPO 的核心：裁剪目标

PPO 用"重要性采样比" $\rho_t = \dfrac{\pi_\theta(a_t)}{\pi_{\theta_{\text{old}}}(a_t)}$ 衡量新旧策略差异，并裁剪它防止一步走太大：

$$\mathcal{L}^{\text{clip}} = \mathbb{E}\Big[\min\big(\rho_t \hat A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]$$

$\hat A_t$ 是优势（这个动作比平均好多少，由 critic 估的基线算出）。$\epsilon$ 典型 0.2。直觉：**优势为正就提概率，但提太多会被 clip"封顶"，防止训练震荡。**

---

## 7. DPOTrainer：跳过奖励模型，直接优化偏好

### 7.1 解决什么问题

PPO 流程长（要先训 RM，再 RL，4 模型同台，调参敏感）。**DPO（Direct Preference Optimization）的洞见：可以用数学把 RM 和 RL 两步"合并"成一个像 SFT 一样稳定的监督损失。** 不再显式训练奖励模型，也不在线采样。

### 7.2 核心推导直觉

RLHF 的最优策略（带 KL 约束最大化奖励）有闭式解，反解可得"奖励 = 策略与参考策略的对数概率比再乘 $\beta$"：

$$r(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \text{(与 }y\text{ 无关的常数)}$$

把这个"隐式奖励"代回 §5.2 的 Bradley-Terry 损失，常数在差值里抵消，得到 DPO 损失：

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\Big[\log \sigma\Big(\beta \log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big]$$

### 7.3 数据流（对比 PPO 看简化在哪）

```
   偏好对 (x, y_w, y_l)
        │
        ├──► π_θ  : 算 logπ(y_w|x), logπ(y_l|x)   ← 训练
        └──► π_ref: 算 logπ_ref(y_w|x), logπ_ref(y_l|x) ← 冻结
        │
        ▼
   两个"对数比" 相减 → sigmoid → 损失 → 反传
   (无奖励模型、无采样、无 critic，只 2 个模型前向)
```

直觉：**推高 chosen 相对 ref 的概率，压低 rejected 相对 ref 的概率**，$\beta$ 控制偏离参考的力度。数据字段：`prompt`/`chosen`/`rejected`。TRL 内还提供 IPO、KTO、ORPO、CPO 等变体（多为换 loss 形态/换数据格式）。

---

## 8. GRPOTrainer：DeepSeek 同款，去掉 Critic

### 8.1 解决什么问题

PPO 要养一个和策略差不多大的 **Critic（价值网络）** 来估基线，贵且难训。**GRPO（Group Relative Policy Optimization）的招：对同一个 prompt 采样一组 $G$ 个回答，用"组内平均分"当基线，从而彻底删掉 critic。** 这是 DeepSeek-R1 推理训练走红的范式，尤其适配可程序化验证答案的任务（数学/代码，reward 可由规则给）。

### 8.2 机制

```
   prompt x
      │  采样 G 个回答 (如 G=8)
      ▼
   y_1 y_2 ... y_G ──► reward ──► r_1 r_2 ... r_G
                              (RM 或 规则: 答对=1 答错=0)
      │
      ▼  组内标准化做优势 (不要 critic!)
   A_i = (r_i − mean(r)) / std(r)
      │
      ▼  PPO 式裁剪目标 + KL(π_θ‖π_ref) 正则 → 更新
```

优势直接来自"我这条比这一组平均好多少"：

$$\hat A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

高于组均值→正优势→提概率；低于→压。**省掉 critic ≈ 省掉一个大模型的显存与训练复杂度**，代价是每步要为同一 prompt 多次采样（生成开销上升）。reward 可插规则函数，特别契合数学/代码这类有客观对错的场景。

---

## 9. 与 PEFT / Accelerate 集成

TRL 的实战价值很大一部分来自"省显存 + 易扩展"，这正是 PEFT 和 Accelerate 的职责。

### 9.1 PEFT / LoRA：让对齐能在小卡上跑

全参微调一个 7B 模型，光优化器状态就吃几十 GB。LoRA 冻结原权重，只训练注入的低秩矩阵 $\Delta W = BA$（详见 [[ai-framework/huggingface-peft/README]]）。在 TRL 里通常只需给 Trainer 传一个 `peft_config`（`LoraConfig`），它就会自动把模型包成 PEFT 模型再训练。

```
   全参 DPO:  base(冻不冻都占内存) + π_ref + 优化器状态 ── 显存爆炸
   LoRA DPO:  base 冻结(可量化为4bit=QLoRA) + 小 LoRA 适配器(训)
             π_ref 还能"省掉"——禁用 adapter 即得参考模型！
```

**一个精妙之处**：DPO/PPO 需要参考模型 $\pi_{\text{ref}}$。用 LoRA 时，"关掉 adapter 的当前模型"就等于原始 SFT 模型，所以**不必再单独加载一份 ref 模型**，进一步省一半显存。这是 LoRA+对齐的经典组合拳。

### 9.2 Accelerate / DeepSpeed / FSDP：多卡扩展

因为继承自 `transformers.Trainer`，TRL 脚本天然能用 `accelerate launch` 启动，并通过 accelerate 的配置切换：

```
单卡 ──► DDP 多卡数据并行 ──► DeepSpeed ZeRO-1/2/3 ──► FSDP
 易              ↑显存均摊          ↑切分优化器/梯度/参数   ↑参数分片
```

- **ZeRO-2**：切分优化器状态+梯度，常用、性价比高。
- **ZeRO-3 / FSDP**：连参数也切分，单卡放不下整模型时用，通信更重。
- 混合精度（bf16/fp16）、梯度累积、gradient checkpointing 全部通过 `XxxConfig`（即 `TrainingArguments`）字段直接开。

> 具体 `accelerate config` 选项、DeepSpeed json、显存数字随模型/序列长/卡型变化，请以官方文档与实测为准。

---

## 10. 数值例子 / 对照

### 10.1 手算 DPO 损失（一条样本）

设 $\beta=0.1$。某偏好对，模型与参考的对数概率（取自然对数）：

| | $\log\pi_\theta$ | $\log\pi_{\text{ref}}$ | 对数比 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ |
|--|--|--|--|
| chosen $y_w$  | $-5.0$ | $-6.0$ | $+1.0$ |
| rejected $y_l$ | $-4.0$ | $-3.0$ | $-1.0$ |

差值项：$\beta(1.0 - (-1.0)) = 0.1 \times 2.0 = 0.2$。

损失：$-\log \sigma(0.2) = -\log\frac{1}{1+e^{-0.2}} = -\log(0.5498) \approx 0.598$。

解读：chosen 相对 ref 更"被偏爱"（对数比 +1 > -1），方向正确，所以损失低于 $-\log\sigma(0)=0.693$。若把两者对数比对调（模型偏爱了 rejected），差值变 $-0.2$，损失升到 $-\log\sigma(-0.2)\approx 0.798$——模型被惩罚。

### 10.2 手算 GRPO 优势（一组 4 答）

同一道数学题采 4 个回答，规则打分（对=1 错=0）：$r = [1, 0, 1, 0]$。

- 均值 $\bar r = 0.5$；
- 标准差 $\sigma = \sqrt{\frac{(0.5)^2\times4}{4}} = 0.5$；
- 优势 $\hat A_i = (r_i-\bar r)/\sigma$：答对的 $=(1-0.5)/0.5=+1$，答错的 $=-1$。

于是两条对的回答概率被往上推（优势 +1），两条错的被往下压（优势 -1）——**完全不需要价值网络，组内一比即得方向。**

### 10.3 五大 Trainer 横评

| Trainer | 阶段 | 数据形态 | 需奖励模型 | 需参考模型 | 需 Critic | 在线采样 | 典型代价 | 一句话 |
|---------|------|----------|:--:|:--:|:--:|:--:|----------|--------|
| **SFTTrainer** | 监督微调 | (指令,回答) | ✗ | ✗ | ✗ | ✗ | 低 | 学格式与跟随 |
| **RewardTrainer** | 奖励建模 | 偏好对 chosen/rejected | — | ✗ | ✗ | ✗ | 低-中 | 训打分器 |
| **PPOTrainer** | RLHF | prompt(+RM) | ✓ | ✓ | ✓ | ✓ | 高(4模型) | 经典强大但难调 |
| **DPOTrainer** | 对齐 | 偏好对 | ✗ | ✓ | ✗ | ✗ | 中 | 稳、像SFT、最常用 |
| **GRPOTrainer** | 对齐/推理 | prompt + reward函数 | 可规则 | ✓ | ✗ | ✓ | 中-高 | 去critic，擅数学代码 |

### 10.4 何时用哪个（决策树）

```
 还没做过监督微调? ──是──► 先 SFTTrainer
        │否
        ▼
 答案有客观对错(数学/代码/可验证)? ──是──► GRPOTrainer (规则reward)
        │否
        ▼
 只有偏好对、想要稳定省事? ──是──► DPOTrainer (主流首选)
        │否(要在线探索/已有强RM/追极致效果)
        ▼
 RewardTrainer 训 RM ──► PPOTrainer
```

---

## 常见问题

| 现象 | 可能原因 | 方向性排查 |
|------|----------|------------|
| SFT 训了但答非所问 | loss 没 mask prompt / 模板没套对 | 检查 labels 是否 -100 掉 prompt，确认 chat_template |
| 显存 OOM | 全参 + 多模型同台(尤其 PPO) | 上 LoRA/QLoRA、ZeRO-3/FSDP、减 batch/序列长、开 gradient checkpointing |
| PPO 训练发散/输出胡言 | reward hacking、KL 太松 | 调大 $\beta$（KL 系数）、检查 RM 质量、降学习率 |
| DPO 两个 logp 都在跌 | 正常现象 | DPO 优化的是"差"，绝对概率下降不一定坏，看 reward margin/准确率 |
| DPO 不收敛 | $\beta$ 不当 / 数据噪声 | 调 $\beta$（常 0.1~0.5）、清洗偏好对、确认 ref 模型正确 |
| GRPO 生成巨慢 | 每 prompt 采 G 个回答 | 减小组大小 G、用 vLLM 等加速采样后端、缩短 max_new_tokens |
| LoRA 下 ref 模型咋来的 | 误以为要单独加载 | 禁用 adapter 即得 ref，无需第二份模型 |
| 多卡跑不起来 | accelerate/DeepSpeed 配置 | `accelerate config` 重配，对齐 `XxxConfig` 与启动方式 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全库总览与学习路径
- [[llm-alignment/RLHF]] — RLHF 三阶段原理（PPO 的上游理论）
- [[llm-alignment/DPO]] — DPO 推导与变体（IPO/KTO/ORPO）深入
- [[ai-framework/huggingface-peft/README]] — LoRA/QLoRA 等参数高效微调（TRL 省显存的底座）

> 免责声明：TRL 迭代频繁，文中类名/参数/默认值/CLI 仅为帮助理解原理的示意，落地请以你所装版本的官方文档为准。文中数值为讲解用的简化算例，非任何特定运行结果。
