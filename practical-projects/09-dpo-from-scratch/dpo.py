"""
从零实现 DPO（Direct Preference Optimization，直接偏好优化）
=============================================================

这是什么
--------
DPO（论文 arXiv:2305.18290）是一种**绕过显式奖励模型与 PPO 强化学习**的偏好对齐方法。
传统 RLHF 流程是：SFT -> 训练奖励模型(RM) -> 用 PPO 让策略最大化 RM 奖励。
DPO 的核心洞察：对于 Bradley-Terry 偏好模型 + KL 正则的 RLHF 目标，其**最优策略**有解析形式，
反代回去可把"奖励"重参数化为 *策略 logp 与参考模型 logp 的对数比*。于是整个对齐过程坍缩成
一个**简单的二分类式监督损失**，直接在偏好对 (prompt, chosen, rejected) 上做梯度下降即可。

DPO 损失（本文件核心，见 dpo_loss 函数）：
    L = -E[ log sigmoid( beta * ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                                 -(logπ_θ(y_l|x) - logπ_ref(y_l|x)) ) ) ]
其中 y_w = chosen（更优回答），y_l = rejected（更差回答），
π_θ = 待训练的策略模型，π_ref = 冻结的参考模型（通常是 SFT 初始权重的拷贝），
beta 控制对参考模型的偏离强度（隐式 KL 约束）。

本 demo 做什么
--------------
1. 在一个**字符级 toy 语言模型**（单层 GRU + 线性头，纯 torch CPU）上跑通 DPO。
2. 合成一份"偏好数据"：同一个 prompt 下，把"礼貌/正向"的续写当作 chosen，
   把"粗鲁/负向"的续写当作 rejected。
3. 训练时打印：DPO loss 下降、隐式奖励间隔(reward margin)上升、
   偏好准确率(chosen 的序列对数概率 > rejected 的比例)上升。
4. 训练前后各采样几句生成，肉眼可见模型从"中性"偏向"礼貌正向"。

为什么 CPU 几十秒能跑完
-----------------------
词表 = 字符集（几十个），hidden=64，序列很短，几百步即可见收敛趋势。

注意（教学代码约定）
--------------------
- 所有 print 仅用 ASCII（Windows 控制台默认 GBK，打印中文会 UnicodeEncodeError）。
- 中文只出现在注释 / docstring / README。
- 固定随机种子，结果可复现。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 0. 可复现性：固定所有随机源
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------------
# 1. 合成偏好数据
#    设计思想：同一 prompt 下，chosen 用礼貌/正向词，rejected 用粗鲁/负向词。
#    模型若学到偏好，会更倾向于在 prompt 之后生成正向续写。
# ----------------------------------------------------------------------------
PROMPTS = [
    "the user said ",
    "my reply is ",
    "i think that ",
    "the answer is ",
]

# chosen 续写：礼貌 / 正向
POSITIVE = [
    "thank you very much",
    "you are most welcome",
    "i am happy to help",
    "that is a great idea",
    "of course i can help",
    "it is my pleasure",
]

# rejected 续写：粗鲁 / 负向
NEGATIVE = [
    "go away right now",
    "i do not care here",
    "that is a bad idea",
    "no i will not help",
    "stop bothering me now",
    "this is just useless",
]


@dataclass
class PrefExample:
    """一条偏好样本：相同 prompt，chosen 优于 rejected。"""
    prompt: str
    chosen: str     # 完整序列 = prompt + 正向续写
    rejected: str   # 完整序列 = prompt + 负向续写
    prompt_len: int  # prompt 的字符数（用于在算 logp 时屏蔽 prompt token，只对续写计损失）


def build_preference_dataset() -> list[PrefExample]:
    """
    笛卡尔式地把每个 prompt 与每对 (正向, 负向) 续写组合成偏好对。
    DPO 训练只需要相对偏好（chosen 比 rejected 好），不需要绝对奖励标签。
    """
    data: list[PrefExample] = []
    for p in PROMPTS:
        for pos, neg in zip(POSITIVE, NEGATIVE):
            data.append(
                PrefExample(
                    prompt=p,
                    chosen=p + pos,
                    rejected=p + neg,
                    prompt_len=len(p),
                )
            )
    random.shuffle(data)
    return data


# ----------------------------------------------------------------------------
# 2. 字符级词表（包含一个 BOS 用于自回归起始）
# ----------------------------------------------------------------------------
class CharVocab:
    """把字符 <-> 整数 id 互转。多一个特殊起始符 <bos>。"""

    BOS = "\x02"  # 用一个不会出现在文本里的控制字符当作 BOS

    def __init__(self, texts: list[str]):
        charset = set(self.BOS)
        for t in texts:
            charset.update(t)
        # 排序保证不同运行得到稳定的 id 映射（可复现）
        self.itos = [self.BOS] + sorted(c for c in charset if c != self.BOS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.stoi[c] for c in text]
        if add_bos:
            ids = [self.stoi[self.BOS]] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if self.itos[i] != self.BOS)


# ----------------------------------------------------------------------------
# 3. Toy 自回归语言模型：embedding -> GRU -> 线性头
#    与 GPT 的差别只在 backbone（这里用 GRU），DPO 逻辑与 backbone 无关。
# ----------------------------------------------------------------------------
class CharLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) 的 token id; 返回 logits: (B, T, V)。"""
        h = self.embed(x)
        out, _ = self.gru(h)
        return self.head(out)


# ----------------------------------------------------------------------------
# 4. 计算一条序列在某个模型下的对数概率（只对续写部分计，屏蔽 prompt）
#    这是 DPO 的关键输入：我们要的是 logπ(y | x)，即"给定 prompt x，
#    生成续写 y"的对数似然，而不是把 prompt 自身的似然也算进去。
# ----------------------------------------------------------------------------
def sequence_logprob(
    model: nn.Module,
    token_ids: torch.Tensor,   # (T,) 含 BOS 的完整序列
    response_start: int,       # 续写在 token_ids 中的起始下标（之前都是 prompt/BOS，要屏蔽）
) -> torch.Tensor:
    """
    自回归地用前缀预测下一个 token：logits[t] 预测 token_ids[t+1]。
    我们把"被预测的目标位置"落在续写区间内的对数概率累加起来。
    返回标量 tensor（保留计算图，供反向传播）。
    """
    x = token_ids.unsqueeze(0)            # (1, T)
    logits = model(x)[0]                  # (T, V)
    logp_all = F.log_softmax(logits, dim=-1)  # 每个位置对全词表的对数概率

    # 目标 token：位置 t 的预测目标是 token_ids[t+1]
    targets = token_ids[1:]              # (T-1,)
    pred_logp = logp_all[:-1]           # (T-1, V) 与 targets 对齐
    token_logp = pred_logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)

    # 屏蔽：只保留"目标位置 >= response_start"的项（即续写部分），prompt 部分不计损失。
    # token_logp[i] 对应的目标 token 是 token_ids[i+1]，其绝对位置为 i+1。
    positions = torch.arange(1, token_ids.size(0))   # 目标 token 的绝对位置
    mask = (positions >= response_start).float()
    return (token_logp * mask).sum()


# ----------------------------------------------------------------------------
# 5. DPO 损失
#    pi_logps_*  : 策略模型给 chosen / rejected 的续写对数概率
#    ref_logps_* : 参考模型（冻结）给 chosen / rejected 的续写对数概率
# ----------------------------------------------------------------------------
def dpo_loss(
    pi_logp_chosen: torch.Tensor,
    pi_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
):
    """
    返回 (loss, chosen_reward, rejected_reward)。
    这里的"隐式奖励" r(x,y) = beta * (logπ_θ(y|x) - logπ_ref(y|x))，
    DPO 让 chosen 的隐式奖励高于 rejected，等价于把偏好概率
    sigmoid(r_w - r_l) 推向 1。
    """
    # 策略相对参考的对数比（log-ratio），对应隐式奖励 / beta
    pi_logratio_chosen = pi_logp_chosen - ref_logp_chosen
    pi_logratio_rejected = pi_logp_rejected - ref_logp_rejected

    # DPO 的核心 logit：beta * ( log-ratio(chosen) - log-ratio(rejected) )
    logits = beta * (pi_logratio_chosen - pi_logratio_rejected)

    # -log sigmoid(logits)：希望 logits 越大越好（chosen 更受偏好）
    loss = -F.logsigmoid(logits)

    # 仅用于监控的隐式奖励（detach，不参与反传）
    chosen_reward = (beta * pi_logratio_chosen).detach()
    rejected_reward = (beta * pi_logratio_rejected).detach()
    return loss, chosen_reward, rejected_reward


# ----------------------------------------------------------------------------
# 6. 评估：在整个偏好集上算 (1) 平均隐式奖励间隔 (2) 偏好准确率
#    偏好准确率 = 策略给 chosen 的续写 logp > rejected 的比例（越接近 1 越好）
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(policy, ref, vocab, data, beta):
    correct = 0
    margin_sum = 0.0
    for ex in data:
        ids_c = torch.tensor(vocab.encode(ex.chosen), dtype=torch.long)
        ids_r = torch.tensor(vocab.encode(ex.rejected), dtype=torch.long)
        # +1：encode 在最前加了 BOS，使 prompt 字符整体后移一位
        start_c = ex.prompt_len + 1
        start_r = ex.prompt_len + 1

        pi_c = sequence_logprob(policy, ids_c, start_c)
        pi_r = sequence_logprob(policy, ids_r, start_r)
        ref_c = sequence_logprob(ref, ids_c, start_c)
        ref_r = sequence_logprob(ref, ids_r, start_r)

        # 偏好准确率用策略自身的续写 logp 比较（chosen 是否被判得更可能）
        if pi_c.item() > pi_r.item():
            correct += 1

        rc = beta * (pi_c - ref_c)
        rr = beta * (pi_r - ref_r)
        margin_sum += (rc - rr).item()

    acc = correct / len(data)
    avg_margin = margin_sum / len(data)
    return acc, avg_margin


# ----------------------------------------------------------------------------
# 7. 采样：从模型自回归生成，用于训练前后定性对比
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample(model, vocab, prompt: str, max_new: int = 22, temperature: float = 0.7) -> str:
    ids = vocab.encode(prompt)  # 含 BOS + prompt
    ids = torch.tensor(ids, dtype=torch.long)
    for _ in range(max_new):
        logits = model(ids.unsqueeze(0))[0, -1]      # 最后一步的 logits
        probs = F.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt])
    return vocab.decode(ids.tolist())


# ----------------------------------------------------------------------------
# 8. 先做一个最小 SFT，让参考模型/策略起点"会写英文字符"，
#    否则随机初始化下 DPO 的对数比噪声太大，收敛信号不直观。
#    （真实流程中参考模型就是 SFT 模型，这里我们也照此构造。）
# ----------------------------------------------------------------------------
def pretrain_sft(model, vocab, texts, steps=400, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
    model.train()
    for step in range(steps):
        seq = random.choice(seqs)
        logits = model(seq.unsqueeze(0))[0]          # (T, V)
        # 标准语言模型损失：用前 t 个预测第 t+1 个
        loss = F.cross_entropy(logits[:-1], seq[1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"[sft] step {step + 1:4d}/{steps}  ce_loss={loss.item():.4f}")
    model.eval()


# ----------------------------------------------------------------------------
# 9. DPO 训练主循环
# ----------------------------------------------------------------------------
def train_dpo(policy, ref, vocab, data, beta=0.1, steps=300, lr=2e-3, batch_size=8):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # 参考模型全程冻结（不计梯度、不更新）
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    history = []
    for step in range(steps):
        policy.train()
        batch = random.sample(data, k=min(batch_size, len(data)))

        loss_accum = 0.0
        for ex in batch:
            ids_c = torch.tensor(vocab.encode(ex.chosen), dtype=torch.long)
            ids_r = torch.tensor(vocab.encode(ex.rejected), dtype=torch.long)
            start = ex.prompt_len + 1  # +1 跳过 BOS

            pi_c = sequence_logprob(policy, ids_c, start)
            pi_r = sequence_logprob(policy, ids_r, start)
            # 参考模型 logp 不需要梯度
            with torch.no_grad():
                ref_c = sequence_logprob(ref, ids_c, start)
                ref_r = sequence_logprob(ref, ids_r, start)

            loss, _, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
            loss_accum = loss_accum + loss

        loss_mean = loss_accum / len(batch)
        opt.zero_grad()
        loss_mean.backward()
        opt.step()

        if (step + 1) % 30 == 0:
            acc, margin = evaluate(policy, ref, vocab, data, beta)
            history.append((step + 1, loss_mean.item(), acc, margin))
            print(
                f"[dpo] step {step + 1:4d}/{steps}  "
                f"loss={loss_mean.item():.4f}  "
                f"pref_acc={acc:.3f}  "
                f"reward_margin={margin:+.4f}"
            )
    return history


# ----------------------------------------------------------------------------
# 10. 可选画图（失败则纯文本回退，不崩）
# ----------------------------------------------------------------------------
def maybe_plot(history, out_path="dpo_curve.png"):
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境用 Agg 后端写文件
        import matplotlib.pyplot as plt

        steps = [h[0] for h in history]
        loss = [h[1] for h in history]
        acc = [h[2] for h in history]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(steps, loss, "o-", color="tab:red", label="DPO loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("DPO loss", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(steps, acc, "s-", color="tab:blue", label="pref acc")
        ax2.set_ylabel("preference accuracy", color="tab:blue")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        print(f"[plot] saved curve to {out_path}")
    except Exception as e:  # noqa: BLE001 - 教学代码：任何画图问题都降级为文本
        print(f"[plot] skipped (matplotlib unavailable: {e})")


# ----------------------------------------------------------------------------
# 11. demo 入口
# ----------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("DPO from scratch (toy char-level LM, CPU)")
    print("=" * 64)

    data = build_preference_dataset()
    all_texts = [ex.chosen for ex in data] + [ex.rejected for ex in data] + PROMPTS
    vocab = CharVocab(all_texts)
    print(f"vocab_size={vocab.size}  num_preference_pairs={len(data)}")

    # ---- 构造 SFT 参考模型：先在全部文本(正/负都见)上做语言建模 ----
    # 这样参考模型对 chosen/rejected 是"中性"的，DPO 才有偏移空间。
    print("\n-- Stage 1: SFT a neutral reference model --")
    base = CharLM(vocab.size)
    pretrain_sft(base, vocab, all_texts, steps=400)

    # 策略模型 = 参考模型的深拷贝（DPO 标准做法：从同一 SFT 起点出发）
    import copy
    policy = copy.deepcopy(base)
    ref = copy.deepcopy(base)

    beta = 0.1

    # ---- 训练前评估 + 采样 ----
    acc0, margin0 = evaluate(policy, ref, vocab, data, beta)
    print(f"\n[before DPO] pref_acc={acc0:.3f}  reward_margin={margin0:+.4f}")
    print("[before DPO] samples:")
    for p in PROMPTS:
        print("   '" + sample(policy, vocab, p) + "'")

    # ---- DPO 训练 ----
    print("\n-- Stage 2: DPO preference optimization --")
    history = train_dpo(policy, ref, vocab, data, beta=beta, steps=300)

    # ---- 训练后评估 + 采样 ----
    acc1, margin1 = evaluate(policy, ref, vocab, data, beta)
    print(f"\n[after DPO]  pref_acc={acc1:.3f}  reward_margin={margin1:+.4f}")
    print("[after DPO]  samples:")
    for p in PROMPTS:
        print("   '" + sample(policy, vocab, p) + "'")

    maybe_plot(history)

    # ---- 量化"成功"信号 ----
    print("\n" + "=" * 64)
    print("SUMMARY (success signals)")
    print("=" * 64)
    print(f"preference accuracy : {acc0:.3f} -> {acc1:.3f}  (delta {acc1 - acc0:+.3f})")
    print(f"reward margin       : {margin0:+.4f} -> {margin1:+.4f}  (delta {margin1 - margin0:+.4f})")
    improved = (acc1 >= acc0) and (margin1 > margin0)
    print("result              : " + ("PASS - DPO pushed policy toward chosen" if improved
                                       else "CHECK - no clear improvement"))


if __name__ == "__main__":
    main()
