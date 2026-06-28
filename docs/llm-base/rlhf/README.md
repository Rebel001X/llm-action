# RLHF：基于人类反馈的强化学习

> 用人类偏好把"会接话"的语言模型调成"会好好接话"的助手——三阶段（SFT → 奖励模型 → PPO）把"人喜欢哪个回答"这种没法写成损失函数的目标，蒸馏进一个标量奖励，再用强化学习去最大化它。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-alignment/RLHF]]、[[llm-alignment/DPO]]、[[llm-algo/transformer/模型架构]]、[[llm-train/README]]、[[llm-algo/FLOPs]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：RLHF 到底在优化什么 | 偏好、标量奖励 |
| 1 | 地基：SFT / RL / 策略 / KL 散度 | 前置概念 |
| 2 | 为什么不能直接监督学习 | 不可微目标 |
| 3 | 阶段一 SFT：先学会"像人话" | 指令微调 |
| 4 | 阶段二 奖励模型 RM：把偏好压成标量 | Bradley-Terry、pairwise loss |
| 5 | 阶段三 PPO：用奖励去优化策略 | Actor/Critic/优势/裁剪 |
| 6 | KL 惩罚：别跑太远 | reference model |
| 7 | 四个模型同台：显存与数据流 | Actor/Ref/RM/Critic |
| 8 | 百川2 的 RM 数据工程（实战） | 三层分类、多样性 |
| 9 | RLHF vs DPO vs RLAIF | 路线对比 |
| 数值 | 奖励/优势/KL/显存逐步手算 | 手算 |
| FAQ | 常见坑与澄清 | 表格 |

---

## 0. 一句话锚点

**RLHF = 给语言模型找一个"打分老师"（奖励模型），再让模型用强化学习去拿高分，同时拿一根"皮筋"（KL 惩罚）拴住它别跑偏。** 形式化：策略 $\pi_\theta(y\mid x)$（输入 prompt $x$ 输出回答 $y$），要解

$$
\max_{\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\Big[\,r_\phi(x,y)\,\Big]\;-\;\beta\,\mathbb{D}_{\mathrm{KL}}\!\big[\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\big]
$$

- $r_\phi(x,y)$：奖励模型给"回答好不好"打的标量分。
- $\pi_{\mathrm{ref}}$：参考模型（一般就是 SFT 后冻结的那份），皮筋的另一端。
- $\beta$：皮筋松紧（KL 系数）。$\beta$ 大→保守贴着 SFT；$\beta$ 小→放飞，容易 reward hacking。

记住这一个式子，下面所有工程都是在**算这三项、求这个 max**。

```
              人类标注偏好
                  │
        ┌─────────▼──────────┐
 prompt │  奖励模型 r_φ(x,y)  │  ← 把"人更喜欢谁"压成一个数
   x ──►│   (打分老师)        │
        └─────────┬──────────┘
                  │ r(x,y)
        ┌─────────▼──────────┐
        │  PPO 优化策略 π_θ   │  拿高分 − β·KL(π_θ‖π_ref)
        │   (学生)            │
        └────────────────────┘
```

---

## 1. 地基 / 前置

把陌生词先拆成原子，后面才不会卡。

| 概念 | 一句话 | 在 RLHF 里是谁 |
|---|---|---|
| 策略 policy $\pi_\theta$ | 在状态下选动作的概率分布 | 语言模型本身 |
| 状态 state | 当前已生成的前缀 $x,y_{<t}$ | prompt + 已生成 token |
| 动作 action | 下一个 token $y_t$ | 词表里选一个 token |
| 轨迹 trajectory | 一整段生成 $y_1\dots y_T$ | 一条完整回答 |
| 奖励 reward | 动作/轨迹的好坏标量 | RM 在序列末尾给一个分 |
| 回报 return | 未来奖励的累加 | 这里几乎只有终止奖励 |
| 价值 value $V(s)$ | 从状态 $s$ 出发期望回报 | Critic 预测的 baseline |
| 优势 advantage $A$ | "这步比平均好多少" $=Q-V$ | 决定梯度方向与大小 |
| KL 散度 | 两个分布的差异度 | 拴住策略别离 ref 太远 |

**自回归视角**：语言生成天然是一个序列决策问题——每生成一个 token 就是一次"动作"，整段回答是一条"轨迹"。这正是能套上强化学习的根本原因。延伸：[[llm-algo/transformer/模型架构]]、[[llm-inference/解码策略]]。

**KL 散度复习**（后面手算要用）：对离散分布

$$
\mathbb{D}_{\mathrm{KL}}[p\|q]=\sum_i p_i\log\frac{p_i}{q_i}\ \ge 0,\quad \text{当且仅当}\ p=q\ \text{时为 }0.
$$

它不对称（$\mathrm{KL}[p\|q]\ne\mathrm{KL}[q\|p]$），量纲是"nat"（用 $\ln$）。

```
KL 当皮筋：
  π_ref ●────────elastic────────● π_θ
        固定锚点          越拉越远，β·KL 罚得越狠
```

---

## 2. 为什么不能直接监督学习？

最自然的想法："既然我知道哪个回答好，直接拿好回答当标签做 SFT 不就行了？" 行，但有三个硬伤：

1. **目标不可微**。"有用、无害、诚实"没法写成对 token 的可微损失。人类只会说"A 比 B 好"，给不出"理想 token 序列"。
2. **只见过正例，没见过负例的代价**。SFT 是"模仿学习"，告诉模型"这么写对"，但从不告诉它"那么写错得多离谱"。模型学不到偏好的**梯度**。
3. **暴露偏差 / 分布漂移**。SFT 训练时喂的是人写的前缀，推理时喂的是自己生成的前缀，误差累积。RL 让模型在**自己生成的分布上**被打分，直面这个问题。

RLHF 的解法：把"A 比 B 好"这种**相对偏好**用 Bradley-Terry 模型变成一个**标量奖励函数**（第 4 节），于是不可微目标变成"最大化标量"，可以用策略梯度优化。

```
监督学习:  标签是"唯一正确答案"   →  写不出来
RLHF:      标签是"A 好于 B"       →  能标 → 学奖励 → 优化
```

---

## 3. 阶段一：SFT（监督微调）

**目的**：先让基座模型"会听指令、说人话"，给后面的 RL 一个像样的起点。如果直接在原始预训练模型上做 PPO，策略太野，采样出来的轨迹质量太差，RM 打分和优化都会崩。

**做法**：标准的指令微调，损失就是 next-token 交叉熵，只在"回答"部分计算损失（prompt 部分 mask 掉）：

$$
\mathcal{L}_{\mathrm{SFT}}=-\sum_{t\in\text{response}}\log \pi_\theta(y_t\mid x,y_{<t}).
$$

```
prompt:  [介绍一下杭州]   ← loss masked (不算损失)
resp:    [杭州 是 浙江 ...] ← 只在这里算交叉熵
```

SFT 完成后这份权重有两个去向：① 作为 PPO 里 **Actor 的初始化**；② 复制冻结一份当 **reference model $\pi_{\mathrm{ref}}$**。延伸：[[llm-train/README]]、[[llm-train/peft/Prefix-Tuning]]。

---

## 4. 阶段二：奖励模型 RM（核心）

这是 RLHF 的"打分老师"，也是百川2 笔记里重点讲的部分。

### 4.1 结构

拿 SFT 模型，把最后的 LM head（输出词表 logits）换成一个**输出标量的线性头**：

```
                       ┌─ 原 LM head: hidden → |V| logits   (丢弃)
hidden(最后一个token) ─┤
                       └─ 新 value head: hidden → 1 标量 r  (新增)
```

输入 $(x,y)$，取最后一个 token（或 EOS）的隐藏态，过线性层得到标量 $r_\phi(x,y)$。

### 4.2 训练目标：Bradley-Terry + pairwise loss

人类标注的是 pair：对同一 prompt $x$，回答 $y_w$（win，更好）优于 $y_l$（lose）。Bradley-Terry 模型假设"$y_w$ 胜过 $y_l$"的概率是

$$
P(y_w\succ y_l)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),\quad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

最大化这个似然，等价于最小化

$$
\boxed{\ \mathcal{L}_{\mathrm{RM}}=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]\ }
$$

直觉：**只关心好坏回答的分差**，不关心绝对分值。分差越大、方向越对，loss 越小。

```
   r(y_w) ●───── gap = r(y_w)−r(y_l) ─────● r(y_l)
                  σ(gap) → 1 时 loss → 0
                  gap 为负(标反了) → loss 巨大
```

> 工程注意：RM 输出是**未校准的相对分**，加任意常数不改 loss（平移不变）。所以 PPO 里通常要对 reward 做白化/标准化。

### 4.3 多个回答的排序

很多团队（含 OpenAI）一次标 $K$ 个回答的排序，把 $\binom{K}{2}$ 个 pair 放进**同一个 batch** 算损失（而非拆成独立样本），避免同一 prompt 的回答被重复前向、也减少过拟合：

$$
\mathcal{L}=-\frac{1}{\binom{K}{2}}\,\mathbb{E}\Big[\textstyle\sum_{w<l}\log\sigma\big(r(x,y_w)-r(x,y_l)\big)\Big].
$$

延伸：[[llm-alignment/RLHF]]。

---

## 5. 阶段三：PPO（用奖励优化策略）

现在有了打分老师 $r_\phi$，让学生 $\pi_\theta$ 去拿高分。用的是 **PPO（Proximal Policy Optimization，近端策略优化）**。

### 5.1 同台四个模型

```
                     ┌──────────────┐
       prompt x ────►│  Actor π_θ   │── 采样回答 y (会更新)
                     └──────┬───────┘
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                    ▼
 ┌────────────┐     ┌──────────────┐     ┌──────────────┐
 │ Ref  π_ref │     │  RM  r_φ     │     │ Critic V_ψ   │
 │ (冻结)     │     │  (冻结,打分) │     │ (会更新,估值)│
 └─────┬──────┘     └──────┬───────┘     └──────┬───────┘
       │KL 项               │终止奖励 r          │baseline V(s)
       └──────────► 组合奖励 + 优势 A ◄──────────┘
```

| 模型 | 作用 | 是否更新 | 初始化自 |
|---|---|---|---|
| Actor $\pi_\theta$ | 被优化的策略 | ✅ | SFT |
| Reference $\pi_{\mathrm{ref}}$ | 算 KL 的锚 | ❌ 冻结 | SFT |
| Reward $r_\phi$ | 给整段打分 | ❌ 冻结 | 阶段二 RM |
| Critic $V_\psi$ | 估状态价值当 baseline | ✅ | 常用 RM 初始化 |

### 5.2 每步奖励怎么拼

只有序列**末尾**有 RM 给的真奖励，中间每个 token 的奖励是 **逐 token KL 惩罚**：

$$
r_t=
\begin{cases}
-\beta\big(\log\pi_\theta(y_t\mid\cdot)-\log\pi_{\mathrm{ref}}(y_t\mid\cdot)\big), & t<T\\[4pt]
r_\phi(x,y)\;-\;\beta\big(\log\pi_\theta(y_T\mid\cdot)-\log\pi_{\mathrm{ref}}(y_T\mid\cdot)\big), & t=T
\end{cases}
$$

KL 用 $\log\pi_\theta-\log\pi_{\mathrm{ref}}$ 在采样轨迹上做**逐 token 单样本估计**。

### 5.3 优势 GAE

用 Critic 的 $V_\psi$ 算 TD 残差 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，再做 GAE：

$$
\hat A_t=\sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l}.
$$

$\gamma$ 折扣（常 1.0），$\lambda$ 是 bias-variance 折中（常 0.95）。

### 5.4 PPO 裁剪目标

令重要性比 $\rho_t=\dfrac{\pi_\theta(y_t)}{\pi_{\theta_{\mathrm{old}}}(y_t)}$，PPO 的核心目标：

$$
\mathcal{L}^{\mathrm{CLIP}}=\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big].
$$

```
ρ 太大(>1+ε)且 A>0 → clip 住，不让一步迈太大
   ────┬────[1−ε ───── 1 ───── 1+ε]────┬────
       裁掉                            裁掉
```

裁剪保证每次更新都"近端"（proximal）——不偏离旧策略太远，这是 PPO 稳定的关键。Critic 同时用 $\mathcal{L}_V=(V_\psi(s_t)-\hat R_t)^2$ 更新。

延伸：[[llm-alignment/RLHF]]、[[ai-framework/deepspeed/README]]（DeepSpeed-Chat 把这四模型的并行/offload 工程化）。

---

## 6. KL 惩罚：为什么必须有

不加 KL 会发生 **reward hacking**：策略发现 RM 的漏洞（比如疯狂输出某些讨喜词、超长废话），把 RM 分刷爆，但人类看了直摇头——因为 RM 只是真实偏好的**有偏代理**。

```
无 KL:  π_θ ──────────────────────────► 漂到 RM 漏洞区
                                         (人类厌恶但RM给高分)
有 KL:  π_θ ──┐ β·KL 拉回 ┌── 在 SFT 附近的"可信域"内提升
              └──────────┘
```

$\beta$ 调参经验：太大→学不动（贴着 SFT）；太小→hacking、胡言乱语。常配 **自适应 KL 控制器**（目标 KL，超了就升 $\beta$，低了就降）。

---

## 7. 四模型同台的显存账（重要）

PPO 阶段同时驻留 **Actor + Ref + RM + Critic** 四份模型，这是 RLHF 工程最痛的地方。延伸：[[llm-algo/FLOPs]]、[[llm-train/README]]。

```
显存占用 ≈  Actor(训练态)  +  Critic(训练态)  +  Ref(推理态)  +  RM(推理态)
            ▲ 含优化器/梯度    ▲ 含优化器/梯度   ▲ 仅权重        ▲ 仅权重
            最贵               次贵
```

**训练态一份模型的显存**（Adam, 混合精度，每参数）：
- 权重 fp16：2 B
- 梯度 fp16：2 B
- Adam 状态（fp32 一阶+二阶）：8 B
- fp32 主权重：4 B
- 合计 ≈ **16 B/参数**

**推理态一份模型**（只放权重 fp16）≈ **2 B/参数**。

---

## 数值手算

### 手算 A：RM 的 loss

某 pair：$r(x,y_w)=2.0$，$r(x,y_l)=0.5$。分差 $=1.5$。

$$
\sigma(1.5)=\frac{1}{1+e^{-1.5}}=\frac{1}{1+0.2231}=0.8176.
$$
$$
\mathcal{L}_{\mathrm{RM}}=-\ln 0.8176=0.2014.
$$

若标注标反了（$r_w=0.5,r_l=2.0$，分差 $-1.5$）：$\sigma(-1.5)=0.1824$，loss $=-\ln0.1824=1.701$ —— 罚 8.4 倍，逼模型把分差掰正。

### 手算 B：逐 token KL 惩罚

某 token，$\pi_\theta=0.40$，$\pi_{\mathrm{ref}}=0.25$，$\beta=0.1$。

$$
\log\pi_\theta-\log\pi_{\mathrm{ref}}=\ln\frac{0.40}{0.25}=\ln1.6=0.4700.
$$
$$
r_t=-\beta\cdot0.4700=-0.047.
$$
策略比 ref 更自信地选了这个 token，被罚 $-0.047$；选了 ref 不爱的（比值<1）会得正奖励——皮筋在起作用。

### 手算 C：优势与 PPO 裁剪

设终止 $r_\phi=1.0$，$\gamma=1,\lambda=1$，某步 $V(s_t)=0.3$，简化后该步 $\hat A_t=1.0-0.3=0.7>0$（这一步比平均好）。

更新后比值 $\rho_t=1.5$，$\epsilon=0.2$ → clip 到 $1.2$。

$$
\rho\hat A=1.5\times0.7=1.05,\quad \mathrm{clip}(\rho)\hat A=1.2\times0.7=0.84.
$$
$$
\mathcal{L}^{\mathrm{CLIP}}=\min(1.05,\,0.84)=0.84.
$$
取了小的 $0.84$ → 梯度被裁，**不让这一步迈太大**，这就是"近端"。

### 手算 D：7B 模型 RLHF 的显存量级

四模型，Actor/Critic 训练态、Ref/RM 推理态（按上面单价）：

```
Actor   7e9 × 16 B = 112 GB
Critic  7e9 × 16 B = 112 GB
Ref     7e9 ×  2 B =  14 GB
RM      7e9 ×  2 B =  14 GB
────────────────────────────
权重/优化器小计   ≈ 252 GB
```

这还没算 KV-Cache、激活和 PPO 经验缓冲。结论：单卡 80GB **放不下**，必须 ZeRO/张量并行/offload。延伸：[[llm-inference/KV-Cache优化]]、[[ai-framework/megatron-lm/README]]、[[ai-framework/deepspeed/README]]。

> 省钱路线：用 LoRA 只训 Actor/Critic 的低秩增量，可把"16 B/参数"摊薄到接近推理态；或干脆走 DPO 砍掉 RM 与采样（第 9 节）。

---

## 8. 百川2 的 RM 数据工程（实战还原）

> 以下还原原始笔记要点，作为 RM 数据侧的工程范例。

**Prompt 多样性**：构造了一个 **200+ 细分类目** 的数据体系，尽可能覆盖用户需求，同时提升每类 prompt 的多样性，从而提升泛化能力。

**Response 多样性**：用**不同尺寸、不同训练阶段**的百川模型生成候选答案；**不使用其他开源模型**（经验证无法提升 RM 准确率，反而引入分布偏移）。

**三层分类系统**：全面覆盖所有类型的用户需求——

```
6 个一级类别
   └── 30 个二级类别
          └── 200+ 个三级类别
```

**两条原则**：
1. 训练时**保证每个类别内数据足够多样**，确保 RM 有更好的泛化性；
2. 奖励数据中结果需由 **Baichuan2 模型生成**，以确保**数据分布统一**（避免 RM 在与策略不同的分布上打分，导致 PPO 阶段奖励失真）。

为什么这么做有道理：RM 只在"自己见过的分布"上可靠。如果 RM 训练用别家模型的回答，而 PPO 里 Actor 生成的是百川风格回答，RM 就会在**没见过的分布**上瞎打分 → reward hacking 概率飙升。这正是第 6 节"代理有偏"的根因之一。

```
对齐分布：
  RM 训练分布 ≈ Actor 采样分布  →  RM 打分可靠
  RM 训练分布 ≠ Actor 采样分布  →  PPO 容易被骗
```

---

## 9. RLHF vs DPO vs RLAIF（路线对比）

| 路线 | 要不要 RM | 要不要在线采样+PPO | 一句话 |
|---|---|---|---|
| **RLHF (PPO)** | 要（显式标量 RM） | 要（四模型同台） | 效果强、最重、最难调 |
| **DPO** | 不要（隐式奖励） | 不要（离线 pair 即可） | 把"奖励+KL"解析地塞进一个分类损失，省掉 RM 和采样 |
| **RLAIF** | 要 | 要 | 偏好标注由 AI（如更强模型）产出，省人力 |

**DPO 的关键洞察**：RLHF 那个"max 奖励 − KL"问题有闭式最优解 $\pi^*\propto\pi_{\mathrm{ref}}\exp(r/\beta)$，反解出 $r$ 用 $\pi_\theta,\pi_{\mathrm{ref}}$ 表示，代回 Bradley-Terry，得到只依赖策略本身的损失：

$$
\mathcal{L}_{\mathrm{DPO}}=-\,\mathbb{E}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\Big)\Big].
$$

不用训 RM、不用在线采样、不用 Critic——四模型砍成两个（$\pi_\theta,\pi_{\mathrm{ref}}$）。代价：缺了在线探索，对分布外样本不如 PPO 鲁棒。详见 [[llm-alignment/DPO]]。

```
RLHF:  数据 → RM → 在线采样 → PPO        (4 模型, 重)
DPO :  偏好数据 ───────────► 一个分类损失  (2 模型, 轻)
```

---

## 常见问题

| 问题 | 回答 |
|---|---|
| 为什么要冻结 reference？ | 它是 KL 的固定锚点，更新了皮筋就没基准了 |
| RM 分数能跨 prompt 比较吗？ | 不能直接比，RM 是相对分、未校准，且有平移不变性 |
| 不加 KL 会怎样？ | reward hacking：刷爆 RM 分但人类反感，输出退化/复读 |
| Critic 为什么常用 RM 初始化？ | 二者都要"理解回答好坏"，共享表征收敛快 |
| $\beta$（KL 系数）怎么调？ | 自适应 KL 控制器盯目标 KL；太大学不动，太小会崩 |
| 一定要 PPO 吗？ | 否，可用 DPO/GRPO/RAFT 等；PPO 是经典但最重的一条 |
| 四个模型显存放不下怎么办？ | ZeRO-3/张量并行/参数 offload/LoRA，或换 DPO |
| GRPO 和 PPO 啥区别？ | GRPO 去掉 Critic，用同 prompt 一组采样的奖励均值当 baseline 估优势，省一份模型 |
| 为什么 response 要用自家模型生成？ | 让 RM 训练分布 ≈ Actor 采样分布，打分才可靠（百川2 经验） |
| 中间 token 的奖励哪来的？ | 只有末尾有 RM 真奖励，中间是逐 token 的 −β·KL |

---

## 🔗 跳转链接

- 导航总览：[[00-知识地图]]
- 对齐主线：[[llm-alignment/RLHF]]、[[llm-alignment/DPO]]
- 模型与生成：[[llm-algo/transformer/模型架构]]、[[llm-inference/解码策略]]、[[llm-algo/旋转编码RoPE]]
- 训练与微调：[[llm-train/README]]、[[llm-train/peft/Prompt-Tuning]]、[[llm-train/peft/Prefix-Tuning]]
- 算力与显存：[[llm-algo/FLOPs]]、[[llm-inference/KV-Cache优化]]、[[llm-compression/quantization/fp8]]
- 并行框架：[[ai-framework/deepspeed/README]]、[[ai-framework/megatron-lm/README]]、[[ai-infra/网络/集合通信原语]]
