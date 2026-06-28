# 从零实现 DPO 偏好对齐（CPU 可跑通）

在一个 **toy 字符级语言模型**上，用纯 PyTorch（CPU）从零实现 **DPO（Direct Preference Optimization，直接偏好优化）**。
全程不依赖任何外部数据集 / 网络 / RL 库，几十秒在普通笔记本上即可跑完，直观看到偏好对齐的收敛信号。

---

## 一、演示什么原理

传统 RLHF 的链路是：`SFT -> 训练奖励模型(RM) -> 用 PPO 让策略最大化 RM 奖励`，流程长、训练不稳、还要在线采样。

**DPO（[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)）** 的核心洞察是：
对"Bradley-Terry 偏好模型 + KL 正则的 RLHF 目标"，其**最优策略有解析解**。把解析解反代回去，可以把"奖励"
重参数化为 **策略 logp 与参考模型 logp 的对数比**。于是整个对齐过程坍缩成一个**简单的二分类式监督损失**，
直接在偏好对 `(prompt, chosen, rejected)` 上做梯度下降即可——**不需要显式奖励模型，也不需要 PPO**。

DPO 损失（本项目 `dpo_loss()` 的实现）：

```
L = - E[ log sigmoid( beta * ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                              - (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ) ) ]
```

- `y_w` = chosen（更优回答），`y_l` = rejected（更差回答）
- `π_θ` = 待训练的策略模型；`π_ref` = **冻结**的参考模型（通常是 SFT 初始权重的拷贝）
- `beta` 控制对参考模型的偏离强度（隐式 KL 约束，越大越"敢"偏离）
- 其中 **隐式奖励** `r(x,y) = beta * (logπ_θ(y|x) - logπ_ref(y|x))`，DPO 让 chosen 的隐式奖励高于 rejected

本 demo 的具体设定：
1. 字符级 toy LM（embedding -> 单层 GRU -> 线性头），词表就是字符集（约 25 个 token）。
2. 先做一个**极小 SFT**得到"中性"的参考模型（同时见过正/负续写），策略模型从同一起点深拷贝出发——这正是 DPO 的标准做法。
3. 合成偏好数据：同一 prompt 下，把"礼貌/正向"续写当 `chosen`，"粗鲁/负向"续写当 `rejected`。
4. 跑 DPO，观察 **loss 下降**、**偏好准确率（chosen 续写 logp > rejected 的比例）上升**、**隐式奖励间隔上升**，
   并对比训练前后的采样生成。

> 关键代码位置：偏好损失 `dpo_loss()`、只对续写计对数概率（屏蔽 prompt）的 `sequence_logprob()`、训练主循环 `train_dpo()`。

---

## 二、怎么跑

环境：Python 3.13 / numpy 2.3 / torch 2.12（CPU）。

```bash
cd practical-projects/09-dpo-from-scratch
python dpo.py
```

无需任何参数、数据下载或 GPU。若装了 matplotlib，会额外存一张 `dpo_curve.png`（loss 与偏好准确率曲线）；
没装也不会崩，会降级为文本提示。

---

## 三、预期输出

下面是一次真实运行的关键片段（随机种子已固定为 42，结果可复现）：

```
vocab_size=25  num_preference_pairs=24

[before DPO] pref_acc=0.583  reward_margin=+0.0000
[before DPO] samples:
   'the user said that isa great idean h'
   'i think that that is a bad ideat id'

-- Stage 2: DPO preference optimization --
[dpo] step   30/300  loss=0.0046  pref_acc=1.000  reward_margin=+5.8164
[dpo] step  150/300  loss=0.0003  pref_acc=1.000  reward_margin=+8.4747
[dpo] step  300/300  loss=0.0001  pref_acc=1.000  reward_margin=+9.5317

[after DPO]  pref_acc=1.000  reward_margin=+9.5317
[after DPO]  samples:
   'the user said t ank t that you ar th'   # 偏向 "thank / you are"
   'i think that that you are most t an'     # 偏向 "you are most (welcome)"
   'the answer is of course t am happy t'    # 偏向 "of course / am happy"

================================================================
SUMMARY (success signals)
================================================================
preference accuracy : 0.583 -> 1.000  (delta +0.417)
reward margin       : +0.0000 -> +9.5317  (delta +9.5317)
result              : PASS - DPO pushed policy toward chosen
```

可量化的"成功"信号：

| 指标 | 训练前 | 训练后 | 含义 |
| --- | --- | --- | --- |
| DPO loss | 高 | ~0.0001 | 偏好分类损失收敛 |
| 偏好准确率 | 0.583 | 1.000 | chosen 续写被判得比 rejected 更可能 |
| 隐式奖励间隔 | 0.0000 | +9.53 | chosen 的隐式奖励显著高于 rejected |

定性上：训练后的采样从"中性/会蹦出负向词"明显转向**礼貌正向续写**（thank / you are most / of course / happy）。

---

## 四、对应 llm-action 文档

- 直接偏好优化原理：[`../../llm-alignment/DPO.md`](../../llm-alignment/DPO.md)
- 对齐总览与 RLHF 对照：[`../../llm-alignment/RLHF.md`](../../llm-alignment/RLHF.md)、[`../../llm-alignment/基本概念.md`](../../llm-alignment/基本概念.md)
- 对齐专题目录：[`../../llm-alignment`](../../llm-alignment)

把本项目当作上述文档的"动手可跑"配套：文档讲清 DPO 的推导，这里给出能亲手改 `beta`、改偏好数据、看曲线的最小实现。

---

## 五、社区参考

- DPO 原始论文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model（arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)

---

## 六、局限 / 与真实工程的差异

本项目目标是"用最少代码讲清 DPO 的核心机制"，因此与生产级实现有意做了大量简化：

1. **模型与数据是 toy 规模**：字符级 GRR、词表约 25、序列很短、偏好对仅 24 条。真实 DPO 跑在
   SFT 过的大模型上，偏好数据是人工/AI 标注的成千上万条 `(prompt, chosen, rejected)`。
2. **过拟合明显**：本 demo 偏好集很小，几十步内偏好准确率就到 1.0、奖励间隔持续上升，说明策略在快速
   偏离参考模型。真实训练需要更大数据 + early stop + 在验证集上看准确率，避免 reward over-optimization。
3. **没有逐 token 归一化与 length penalty**：这里 `sequence_logprob` 直接对续写 token 的 logp 求和，
   长序列天然 logp 更小。工程实现常做长度归一化，或采用 IPO / DPO 变体来缓解长度偏置。
4. **batch 是 Python 循环逐样本前向**：为可读性牺牲了效率。真实实现会把 chosen/rejected padding 成同一
   batch 张量、用 attention mask 一次前向，并通常**共享一次前向**算策略与参考（或缓存参考 logp）。
5. **beta 固定、未调参**：`beta` 是 DPO 最重要的超参（隐式 KL 强度），生产中需要扫参。
6. **参考模型是 SFT 拷贝的简化版**：真实参考模型是完整 SFT 模型；本 demo 用一个极小 SFT 近似，
   只为让对数比有意义的起点。
7. **不含 DPO 的常见增强**：如 label smoothing（cDPO）、reference-free 变体、混合 SFT 正则项等，均未实现。

> 想进一步动手：试着把 `beta` 改成 0.5 / 1.0 看奖励间隔与采样风格的变化；或往偏好数据里加入"风格冲突"样本，
> 观察偏好准确率是否还能到 1.0。
