# DeepSpeed-Chat

> 微软 DeepSpeed 团队推出的「一键式 RLHF 训练系统」：用一行命令把基座模型走完 SFT → Reward Model → PPO 三阶段，核心创新是 **Hybrid Engine（混合引擎）**，让同一份模型权重在 PPO 里能在「训练态」与「推理态」之间秒切，从而把端到端 RLHF 跑得又快又省。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-alignment/RLHF]] [[ai-framework/deepspeed/README]]
>
> 官方代码与博客：`microsoft/DeepSpeed` 仓库 `applications/DeepSpeed-Chat/`，Llama/Llama-2 支持见 `blogs/deepspeed-chat/ds-chat-release-8-31/README.md`（精确版本/参数以官方源码为准）。

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点 | 一键 RLHF + Hybrid Engine |
| 1 | RLHF 为什么难训、DS-Chat 解决什么 | 三阶段、四模型、显存墙 |
| 2 | 三阶段一键流程总览 | step1/2/3、run_*.sh |
| 3 | Stage1 SFT | 监督微调、对齐对话格式 |
| 4 | Stage2 Reward Model | pairwise loss、打分头 |
| 5 | Stage3 PPO 的四个模型 | actor/ref/critic/reward |
| 6 | PPO 算法实现细节 | GAE、ratio clip、KL 惩罚 |
| 7 | **Hybrid Engine（核心）** | 训练↔推理切换、为什么省 |
| 8 | ZeRO 与显存分层 | ZeRO-1/2/3、offload |
| 9 | 典型配置与命令（讲含义） | actor/critic LR、EMA、LoRA |
| 10 | 与 trlx/TRL/OpenRLHF 对比 | 工程定位差异 |
| — | 常见问题 + 跳转链接 | FAQ |

## 0. 一句话锚点

**DeepSpeed-Chat = RLHF 的「三阶段一键脚本」+「一个能在训练和推理两种模式间瞬切的引擎」。**
前者解决「易用性」（不用自己拼 SFT/RM/PPO 的脏活），后者解决「性能」（PPO 阶段 60%+ 时间花在生成上，Hybrid Engine 让生成飞起来）。

## 1. 地基：RLHF 为什么难，DS-Chat 解决什么问题

RLHF（基于人类反馈的强化学习）的标准三阶段（详见 [[llm-alignment/RLHF]]）：

```
 阶段1 SFT          阶段2 RM              阶段3 RLHF(PPO)
┌──────────┐      ┌──────────┐         ┌────────────────────┐
│ 预训练模型 │ ──▶ │ SFT 模型  │  ──▶   │ 用 RM 打分 + PPO 优化 │ ──▶ 对齐模型
│  + 人工示范│      │ +偏好对   │         │   (4 个模型同时在场)  │
└──────────┘      └──────────┘         └────────────────────┘
```

**难点在阶段3**，它同时把 4 个模型塞进显存：

```
            ┌─────────── PPO 训练循环 ───────────┐
  prompt ──▶│ Actor(策略,要训) ──生成──▶ response │
            │      │                              │
            │      ├─▶ Ref(参考,冻结) ──▶ KL 散度  │
            │      ├─▶ Reward(奖励,冻结)──▶ 标量奖励 │
            │      └─▶ Critic(价值,要训)──▶ V(s)    │
            └──────────────────────────────────────┘
```

- **Actor**：被优化的策略模型（通常 = SFT 模型初始化），既要训练又要做 **自回归生成**。
- **Reference（Ref）**：Actor 的冻结副本，算 KL 惩罚防止策略跑偏。
- **Reward**：阶段2 训出的奖励模型，给完整 response 打一个标量分。
- **Critic**：价值网络（常 = Reward 初始化），估计每个 token 的状态价值 $V(s)$，要训练。

**两大痛点，正是 DS-Chat 的设计动机：**
1. **易用性**：四模型 + 三阶段，自己手写训练循环极易出错。→ DS-Chat 给 `step1/step2/step3` 三套现成脚本，一行命令跑通。
2. **性能/显存**：4 个 7B 模型 ≈ 28B 参数同时在场，且 PPO 里 **Actor 既要训练（需优化器状态）又要生成（需 KV-Cache）**，两种模式对显存布局要求矛盾。→ DS-Chat 用 **Hybrid Engine + ZeRO** 解决。

## 2. 三阶段一键流程总览

DS-Chat 的目录心智模型（以官方源码为准）：

```
DeepSpeed-Chat/
├── train.py                 # 总入口，一行命令串起三阶段
└── training/
    ├── step1_supervised_finetuning/   run_*.sh (Llama 等各模型脚本)
    ├── step2_reward_model_finetuning/ run_*.sh
    └── step3_rlhf_finetuning/          run_*.sh  ← Hybrid Engine 在此生效
```

调用链：

```
 用户一行命令
     │
     ▼
 train.py ──┬─▶ step1 SFT  ──▶ 产出 actor 权重 ──┐
            ├─▶ step2 RM   ──▶ 产出 reward 权重 ─┤
            └─▶ step3 PPO  ◀────────────────────┘  (吃前两步产物)
                  │
                  └─▶ DeepSpeedEngine + DeepSpeedHybridEngine
```

每个 step 内部都用 `deepspeed --num_gpus=N main.py ...` 启动，配置走 ZeRO（stage 0/2/3）+ 可选 offload。

## 3. Stage 1：监督微调（SFT）

**目标**：把预训练 Llama 微调成「会按对话格式回话」的 SFT 模型，作为 Actor 与 Ref 的初始化。

机制：标准 causal LM 损失，只在 response token 上算 loss（prompt 部分常被 mask 掉）：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log p_\theta(x_t \mid x_{<t})$$

实践要点：
- 数据用「指令-回答」对，模板要和后续推理一致（Llama-2 用 `[INST] ... [/INST]`，以你的对话模板为准）。
- 学习率比预训练小一个量级；epoch 通常 1~3，过拟合会伤泛化。

## 4. Stage 2：奖励模型（Reward Model）

**目标**：训一个能给回答「打分」的模型。结构 = 基座 + 一个标量回归头（取最后一个 token 的 hidden 过线性层）。

训练用 **pairwise ranking loss**：人工标注「chosen 比 rejected 好」，让模型给 chosen 更高分：

$$\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\theta(x, y_c) - r_\theta(x, y_r)\big)$$

其中 $y_c$ 是更优回答、$y_r$ 是更差回答，$\sigma$ 是 sigmoid。
**数值直觉**：若 $r(y_c)-r(y_r)=2$，则 $\sigma(2)\approx0.88$，loss $\approx0.13$；若两者相等（$=0$），$\sigma(0)=0.5$，loss $=\ln 2\approx0.69$——模型被推着拉开分差。

实践要点：RM 常用比 Actor 小的模型即可；分数尺度无绝对意义，PPO 里只用相对差异，所以训练后常做归一化。

## 5. Stage 3：PPO 训练的四个模型与数据流

一次 PPO 迭代分两步：**(a) 经验采集（Rollout/Generation）→ (b) 学习（Optimization）**。

```
┌──────────────── (a) 经验采集：纯推理 ────────────────┐
│ prompt ─▶ Actor.generate() ─▶ response             │
│ (prompt+response) ─▶ Ref     ─▶ logπ_ref            │
│                   ─▶ Reward  ─▶ r (序列末标量)       │
│                   ─▶ Critic  ─▶ V(s_t) 每 token     │
│                   ─▶ Actor   ─▶ logπ_θ              │
└────────────────────────────────────────────────────┘
                         │ 攒成一个 batch 的经验
                         ▼
┌──────────────── (b) 学习：训练 ─────────────────────┐
│ 算 KL 罚: r_t = r - β·KL(π_θ‖π_ref)                 │
│ 算优势:  GAE → A_t,  回报 R_t                       │
│ 更新 Actor(PPO clip) + 更新 Critic(MSE)             │
└────────────────────────────────────────────────────┘
```

**为什么 (a) 是性能瓶颈**：自回归生成是逐 token 串行的，一个 256-token 的回答要前向 256 次；实测 PPO 里 **生成常占 60%+ 的墙钟时间**。这正是 Hybrid Engine 要攻的点（见第 7 节）。

## 6. PPO 算法实现细节

**奖励整形**：把 RM 的序列级标量分，减去逐 token 的 KL 惩罚，得到每步奖励：

$$r_t = \underbrace{r_{\text{RM}}\cdot \mathbb{1}[t=T]}_{\text{仅末 token 给分}} - \beta\,\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{ref}}(a_t|s_t)\big)$$

KL 罚把策略「拴」在 SFT 附近，$\beta$ 太小会 reward hacking（钻 RM 漏洞输出乱码），太大则学不动。

**优势估计（GAE）**：用 Critic 的 $V$ 算时序差分并指数加权：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad A_t = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}$$

**Actor 损失（带 ratio 裁剪）**：令 $\rho_t = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$，

$$\mathcal{L}^{\text{actor}} = -\,\mathbb{E}\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big]$$

裁剪保证单步更新不过猛（$\epsilon$ 常取 0.2）。**Critic 损失**为回报的 MSE：$\mathcal{L}^{\text{critic}}=\mathbb{E}[(V(s_t)-R_t)^2]$。

实现上的两个常见技巧：
- **PTX/预训练混合损失**：在 PPO 梯度里掺一点 SFT/预训练语言建模损失，缓解「对齐税」导致的通用能力退化。
- **EMA**：对 Actor 权重做指数滑动平均，输出更稳的最终模型。

## 7. Hybrid Engine（核心创新）：训练态↔推理态的瞬切

**问题**：Actor 在 PPO 里要扮两个角色，但两者对系统的要求是矛盾的：

```
        训练态(optimization)            推理态(generation)
   ┌──────────────────────┐      ┌──────────────────────┐
   │ ZeRO-3: 参数被切片分散  │      │ 生成要逐 token 前向，   │
   │ 到所有 GPU，前向要 all- │      │ 切片+反复通信 → 极慢；  │
   │ gather；为反向保留优化器 │      │ 想要：参数聚合、张量并行、│
   │ 状态/激活 → 显存重      │      │ KV-Cache、批量推理 kernel│
   └──────────────────────┘      └──────────────────────┘
```

若全程用 ZeRO-3 训练态去跑生成，慢到无法接受；若另起一份推理副本，显存翻倍。

**Hybrid Engine 的做法**：让同一份权重在两种「内存/并行布局」间无缝切换，复用同一块显存：

```
        ┌──────────── PPO 一次迭代 ────────────┐
        │                                       │
  ┌─────▼─────┐  切到推理态   ┌──────────────┐  │
  │ 训练布局   │ ───────────▶ │ 推理布局        │  │
  │ ZeRO 分片  │  (聚合参数、 │ 参数聚合/张量并行│  │
  │ 优化器状态 │   装推理 kernel│ +KV-Cache 高吞吐│  │
  └─────▲─────┘   +KV-Cache) └───────┬──────┘  │
        │           切回训练态         │ generate │
        └───────────────────────────◀─┘         │
        │ 用采到的经验做 PPO 反向更新             │
        └───────────────────────────────────────┘
```

切到推理态时：把 ZeRO 分片的参数 **聚合**、按 **张量并行** 重排、挂上 **DeepSpeed-Inference 的高性能融合 kernel** 和 **KV-Cache**，于是 `generate()` 快得多；
生成完 **切回训练态**：参数重新按 ZeRO 分片，恢复优化器状态，做反向。整个过程 **不复制第二份权重**，靠的是「布局变换」而非「再开一份模型」。

**为什么省（两个维度）：**
1. **省时间**：生成阶段用上了推理优化（kernel 融合 + KV-Cache + TP），而生成本就是 PPO 的大头，端到端因此提速明显（官方报告大幅吞吐提升，具体倍数以版本而定）。
2. **省显存**：不为「推理副本」额外开权重；训练态继续吃 ZeRO 的分片红利；空闲显存还能跑更大 batch。

一句话：**Hybrid Engine = 把「训练系统的省显存」和「推理系统的快生成」缝在同一份权重上。** 这是 DS-Chat 相对早期 RLHF 框架最大的差异化。

## 8. ZeRO 与显存分层：四模型怎么塞进去

ZeRO（Zero Redundancy Optimizer，详见 [[ai-framework/deepspeed/README]]）按「切什么」分级：

```
ZeRO-1 : 切 优化器状态(Optimizer States)
ZeRO-2 : 切 优化器状态 + 梯度(Gradients)
ZeRO-3 : 切 优化器状态 + 梯度 + 参数(Params)  ← 显存最省，通信最多
offload: 把上述部分搬到 CPU/NVMe，进一步省显存换带宽
```

DS-Chat 里常见搭配：
- **Actor/Critic（要训）**：ZeRO-2 或 ZeRO-3（大模型 + Hybrid Engine 配合）。
- **Ref/Reward（冻结，只前向）**：可用更轻的 ZeRO-3 推理或单纯分片放置，省下优化器状态。

显存账（直觉）：一个参数训练态约需 `参数 + 梯度 + Adam 二阶矩/一阶矩 + fp32 master` ≈ 每参 16 字节（混合精度）。4 个模型里只有 Actor/Critic 吃这份重账，Ref/Reward 只占「参数」那一份，所以把冻结模型尽量轻量化是关键省点。

## 9. 典型配置与命令（讲含义，不背默认值）

启动形态（示意，精确参数以官方 `run_*.sh` 为准）：

```
deepspeed main.py \
  --actor_model_name_or_path  <SFT 产物>      # 阶段1 输出，初始化 Actor/Ref
  --critic_model_name_or_path <RM 产物>       # 阶段2 输出，初始化 Critic/Reward
  --actor_zero_stage  3                       # Actor 的 ZeRO 级别
  --critic_zero_stage 3                       # Critic 的 ZeRO 级别
  --enable_hybrid_engine                      # ★ 开启混合引擎(加速生成)
  --inference_tp_size 2                        # 推理态张量并行宽度
  --actor_learning_rate  <小, 如 1e-5 量级>    # 策略更新步长, 过大易崩
  --critic_learning_rate <略大于 actor>       # 价值网络收敛需要
  --per_device_generation_batch_size <N>      # 采样吞吐
  --per_device_training_batch_size   <M>      # 反向更新吞吐
  --max_answer_seq_len  256                    # 生成长度, 直接决定 rollout 耗时
  --enable_ema                                 # Actor 权重滑动平均, 输出更稳
  --offload / --offload_reference_model        # 显存吃紧时把模型/参数搬 CPU
  # --only_optimize_lora                       # 配合 LoRA, 只训低秩增量, 大幅省显存
```

| 配置项 | 含义 | 权衡 |
|--------|------|------|
| `enable_hybrid_engine` | 生成走推理优化 | 提速核心；某些模型/并行组合需注意兼容性 |
| `inference_tp_size` | 推理态张量并行宽度 | 越大单卡显存越省、通信越多 |
| `actor/critic_zero_stage` | 各自 ZeRO 级别 | 3 最省显存、通信开销最大 |
| `max_answer_seq_len` | 生成 token 上限 | 越长 rollout 越慢、显存越大 |
| KL 系数 `β` | 拴住策略不跑偏 | 小→reward hacking；大→学不动 |
| `enable_ema` | 权重 EMA | 更稳；多一份权重显存 |
| LoRA / `only_optimize_lora` | 只训低秩增量 | 省显存省时；上限略低于全参 |

## 10. 与同类对比（工程定位）

| 框架 | 定位 | 特点 |
|------|------|------|
| **DeepSpeed-Chat** | 一键三阶段 + Hybrid Engine | 训推一体省显存、易上手；绑 DeepSpeed 生态 |
| HuggingFace **TRL** | 库式、与 HF 生态贴合 | 灵活、社区大；大规模需自行接 DeepSpeed/accelerate |
| **trlx**(CarperAI) | 早期工业级 RLHF | 支持大模型；工程偏重 |
| **OpenRLHF** | Ray + vLLM 解耦采样 | 用 vLLM 加速生成、分布式调度强 |

核心差异：DS-Chat 把「快生成」做进 **同一引擎**（Hybrid Engine）；OpenRLHF 等则倾向 **解耦**（独立 vLLM 推理集群 + Ray 调度）。前者部署简单、显存友好；后者在超大规模/异构采样上更灵活。

## 常见问题

| 问题 | 解答 |
|------|------|
| Hybrid Engine 到底省的是什么？ | 省时间（生成用推理优化 kernel+KV-Cache+TP）+ 省显存（不另开推理副本，复用同一份权重） |
| 为什么 PPO 比 SFT 慢这么多？ | 自回归生成串行逐 token，且要同时跑 4 个模型前向；生成常占 60%+ 时间 |
| 4 个模型都要全量显存吗？ | 不。只有 Actor/Critic 吃「参数+梯度+优化器」重账；Ref/Reward 冻结只占参数，可轻量放置 |
| KL 惩罚去掉行不行？ | 不行。没有 KL 罚策略会 reward hacking，钻 RM 漏洞输出乱码骗高分 |
| 显存还是不够怎么办？ | 提高 ZeRO 级别(→3)、开 offload、用 LoRA(`only_optimize_lora`)、缩短 `max_answer_seq_len` |
| Critic 必须和 Reward 同源吗？ | 常用 RM 初始化 Critic（结构相近），但两者职责不同：Reward 冻结打总分，Critic 要训估每步价值 |
| Llama/Llama-2 支持在哪看？ | 官方 `blogs/deepspeed-chat/ds-chat-release-8-31/README.md` 与各 `run_llama*.sh`（以源码为准） |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-alignment/RLHF]] — RLHF 三阶段与 PPO 原理详解
- [[ai-framework/deepspeed/README]] — ZeRO / offload / DeepSpeed 引擎机制
- 官方：`microsoft/DeepSpeed` → `applications/DeepSpeed-Chat/`
- Llama/Llama-2：`blogs/deepspeed-chat/ds-chat-release-8-31/README.md`（精确版本/参数以官方文档与源码为准）
