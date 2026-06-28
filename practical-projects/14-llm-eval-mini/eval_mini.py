# -*- coding: utf-8 -*-
"""
从零实现迷你 LLM 评测(困惑度 + 解码策略对比 + 下游准确率)
=============================================================

这是什么
--------
一个**自包含、CPU 几十秒可跑通**的教学脚本,用一个 toy 字符级语言模型(char-LM)
把"如何评测一个 LLM"这件事拆开讲清楚。不依赖任何外部数据集 / 网络 / gym。

为什么用字符级 toy 模型
----------------------
真实 LLM 评测(perplexity、生成质量、下游任务准确率)在原理上和一个几十 KB 的
char-LM 完全一致,只是规模不同。把模型缩到能在 CPU 上秒级训练,就能把**评测指标
本身**(而不是工程规模)讲透。

本脚本演示三大评测维度
----------------------
1. 困惑度 Perplexity(PPL)
   - 语言建模的核心内在指标。本质 = exp(平均交叉熵)。
   - PPL 越低,模型对"持有的真实文本"越不感到意外。随机猜测时 PPL ≈ 词表大小。
   - 我们打印 PPL 随训练步数下降的曲线,这是模型"在学东西"的可量化信号。

2. 解码策略对比 Decoding strategies
   - 同一个模型,换不同采样方式,生成结果差异巨大。评测生成质量必须固定解码策略。
   - greedy(贪心):每步取 argmax,确定性最高、最"安全"、易重复。
   - temperature(温度采样):logits / T 后采样。T<1 更尖锐保守,T>1 更平、更随机。
   - top-k:只在概率最高的 k 个里采样,截断长尾、兼顾多样性与质量。

3. 下游任务准确率 Downstream accuracy
   - 内在指标(PPL)低 ≠ 下游好用。需要面向任务的外在指标。
   - 我们构造一个 toy 任务:给模型一个 "a+b=" 形式的前缀,让它续写,
     检查它生成的下一个字符是否正确(完形填空式 next-char 准确率)。

运行
----
    python eval_mini.py

所有 print 仅使用 ASCII(适配 Windows GBK 控制台)。中文只在注释 / docstring 里。
若安装了 matplotlib,会把 PPL 曲线存成 ppl_curve.png;否则纯文本打印,不会崩。
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. 可复现:固定所有随机源的种子。评测必须可复现,否则结论不可信。
# ----------------------------------------------------------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------------
# 1. 构造 toy 语料
#    用一段有结构、可学习的小语料(英文句子模板),让 char-LM 能在几秒内学出规律。
#    语料越有规律,PPL 下降越明显,越能展示"模型在学习"。
# ----------------------------------------------------------------------------
def build_corpus():
    # 一组结构化的小句子:模型能学到固定短语、空格、标点的转移规律。
    sentences = [
        "the cat sat on the mat. ",
        "the dog ran in the park. ",
        "a bird flew over the lake. ",
        "the sun is bright today. ",
        "she likes to read good books. ",
        "we play games every day. ",
        "they walk to the old town. ",
        "i drink water in the morning. ",
    ]
    # 重复拼接,得到足够长的训练流(toy 规模即可)。
    corpus = "".join(sentences) * 40
    return corpus


# ----------------------------------------------------------------------------
# 2. 词表 / 编码:字符级 tokenizer。char <-> id 双向映射。
# ----------------------------------------------------------------------------
class CharVocab:
    def __init__(self, text):
        chars = sorted(set(text))  # 排序保证可复现
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ----------------------------------------------------------------------------
# 3. 模型:一个极小的单层 LSTM 字符语言模型。
#    任务:给定前文,预测下一个字符的概率分布 P(x_t | x_<t)。
#    这正是 perplexity 评测所依赖的"对真实下一个 token 给多大概率"。
# ----------------------------------------------------------------------------
class TinyCharLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)  # 输出每个字符的 logit

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) 的字符 id
        emb = self.embed(x)                       # (B, T, E)
        out, hidden = self.lstm(emb, hidden)      # (B, T, H)
        logits = self.head(out)                   # (B, T, V) 未归一化分数
        return logits, hidden


# ----------------------------------------------------------------------------
# 4. 取训练 batch:把语料切成定长片段,输入 x 和目标 y 错开一位(标准 next-token)。
# ----------------------------------------------------------------------------
def get_batch(data_ids, seq_len, batch_size, device):
    n = len(data_ids)
    # 随机起点,采样 batch_size 个长度为 seq_len 的片段
    starts = torch.randint(0, n - seq_len - 1, (batch_size,))
    x = torch.stack([data_ids[s:s + seq_len] for s in starts])         # 输入
    y = torch.stack([data_ids[s + 1:s + 1 + seq_len] for s in starts]) # 目标 = 右移一位
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------
# 5. 困惑度评测:PPL = exp(平均交叉熵)。
#    在一段保留文本上前向,计算每个位置对真实下一个字符的交叉熵,平均后取 exp。
#    交叉熵单位是 nat;exp 后回到"等效有效分支数",即困惑度。
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_perplexity(model, data_ids, seq_len, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # 不重叠地遍历整段保留文本,稳定估计平均交叉熵
    step = seq_len
    for start in range(0, len(data_ids) - seq_len - 1, step):
        x = data_ids[start:start + seq_len].unsqueeze(0).to(device)
        y = data_ids[start + 1:start + 1 + seq_len].unsqueeze(0).to(device)
        logits, _ = model(x)
        # reduction='sum' 便于按 token 数做加权平均
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg_ce = total_loss / max(total_tokens, 1)   # 平均交叉熵 (nat/token)
    ppl = math.exp(avg_ce)                        # 困惑度 = exp(平均交叉熵)
    return avg_ce, ppl


# ----------------------------------------------------------------------------
# 6. 解码:从一个 prompt 出发自回归生成,支持 greedy / temperature / top-k。
#    评测生成质量时,解码策略必须明确且固定,否则结果不可比。
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(model, vocab, prompt, steps, device,
             strategy="greedy", temperature=1.0, top_k=None):
    model.eval()
    ids = torch.tensor(vocab.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    hidden = None
    # 先把 prompt 喂进去,拿到隐状态
    logits, hidden = model(ids, hidden)
    last_logits = logits[:, -1, :]  # 最后一个位置的 logits

    out_ids = []
    for _ in range(steps):
        if strategy == "greedy":
            # 贪心:直接取概率最大的字符(确定性,温度无关)
            next_id = torch.argmax(last_logits, dim=-1, keepdim=True)
        else:
            # 温度缩放:logits / T。T<1 分布更尖(更保守),T>1 更平(更随机)
            scaled = last_logits / max(temperature, 1e-6)
            if strategy == "top_k" and top_k is not None:
                # top-k:只保留概率最高的 k 个 logit,其余置 -inf,再采样
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                scaled = torch.where(scaled < thresh,
                                     torch.full_like(scaled, float("-inf")),
                                     scaled)
            probs = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 按概率采样
        out_ids.append(next_id.item())
        # 把新字符喂回去,推进自回归
        logits, hidden = model(next_id, hidden)
        last_logits = logits[:, -1, :]

    model.train()
    return prompt + vocab.decode(out_ids)


# ----------------------------------------------------------------------------
# 7. 下游 toy 任务准确率:next-char 完形填空。
#    给若干 prompt(本语料里的真实前缀),让模型 greedy 预测下一个字符,
#    与语料中真实的下一个字符比对,算准确率。这是一个"外在/下游"指标:
#    它衡量模型在具体任务上的对错,而不仅是分布上的困惑度。
# ----------------------------------------------------------------------------
@torch.no_grad()
def downstream_next_char_accuracy(model, vocab, corpus, device, num_probes=200, prefix_len=12):
    model.eval()
    rng = random.Random(SEED)  # 独立 RNG,保证探针集合可复现
    correct = 0
    total = 0
    for _ in range(num_probes):
        start = rng.randint(0, len(corpus) - prefix_len - 2)
        prefix = corpus[start:start + prefix_len]
        gold_next = corpus[start + prefix_len]  # 真实的下一个字符
        ids = torch.tensor(vocab.encode(prefix), dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(ids)
        pred_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        if vocab.itos[pred_id] == gold_next:
            correct += 1
        total += 1
    model.train()
    return correct / max(total, 1)


# ----------------------------------------------------------------------------
# 8. 训练循环:边训练边记录 PPL,展示"评测指标如何随训练改善"。
# ----------------------------------------------------------------------------
def train(model, train_ids, val_ids, vocab, corpus, device,
          steps=600, seq_len=48, batch_size=32, lr=3e-3, log_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []  # (step, train_loss, val_ce, val_ppl, downstream_acc)

    # 训练前先评一次:作为"随机初始化"基线,后面对比看下降
    base_ce, base_ppl = evaluate_perplexity(model, val_ids, seq_len, device)
    base_acc = downstream_next_char_accuracy(model, vocab, corpus, device)
    history.append((0, float("nan"), base_ce, base_ppl, base_acc))
    print("[eval] step=%4d  val_ce=%.4f  val_ppl=%8.2f  downstream_acc=%.3f  (baseline)"
          % (0, base_ce, base_ppl, base_acc))

    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, seq_len, batch_size, device)
        logits, _ = model(x)
        # 训练损失 = 交叉熵(next-token 预测)。这正是 PPL 的对数底。
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        # 梯度裁剪:LSTM 训练稳定性常用技巧
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps:
            val_ce, val_ppl = evaluate_perplexity(model, val_ids, seq_len, device)
            acc = downstream_next_char_accuracy(model, vocab, corpus, device)
            history.append((step, loss.item(), val_ce, val_ppl, acc))
            print("[eval] step=%4d  train_loss=%.4f  val_ce=%.4f  val_ppl=%8.2f  downstream_acc=%.3f"
                  % (step, loss.item(), val_ce, val_ppl, acc))
    return history


# ----------------------------------------------------------------------------
# 9. 可选画图:PPL 曲线。没有 matplotlib 就纯文本打印,绝不崩。
# ----------------------------------------------------------------------------
def plot_ppl(history, out_path="ppl_curve.png"):
    steps = [h[0] for h in history]
    ppls = [h[3] for h in history]
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境后端,纯存文件
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(steps, ppls, marker="o")
        plt.xlabel("training step")
        plt.ylabel("validation perplexity")
        plt.title("Perplexity goes down as the model learns")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print("[plot] saved perplexity curve to %s" % out_path)
    except Exception as e:
        # 文本版"曲线":用 PPL 列表代替图
        print("[plot] matplotlib unavailable (%s); printing PPL as text instead:" % type(e).__name__)
        print("[plot] step -> ppl: " + ", ".join("%d:%.1f" % (s, p) for s, p in zip(steps, ppls)))


# ----------------------------------------------------------------------------
# main demo
# ----------------------------------------------------------------------------
def main():
    device = torch.device("cpu")  # 教学脚本,固定 CPU

    print("=" * 72)
    print("MINI LLM EVALUATION DEMO: perplexity + decoding + downstream accuracy")
    print("=" * 72)

    # --- 数据 ---
    corpus = build_corpus()
    vocab = CharVocab(corpus)
    data = torch.tensor(vocab.encode(corpus), dtype=torch.long)
    n_train = int(len(data) * 0.9)
    train_ids, val_ids = data[:n_train], data[n_train:]
    print("[data] corpus_chars=%d  vocab_size=%d  train_tokens=%d  val_tokens=%d"
          % (len(corpus), vocab.size, len(train_ids), len(val_ids)))
    # 随机猜测时 PPL 约等于词表大小,作为"评测下界参照"
    print("[data] random-guess perplexity ~= vocab_size = %d (a useful sanity baseline)"
          % vocab.size)

    # --- 模型 + 训练 ---
    model = TinyCharLM(vocab.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] TinyCharLM params=%d" % n_params)
    print("-" * 72)

    history = train(model, train_ids, val_ids, vocab, corpus, device)

    # --- 评测结论 1:PPL 下降 ---
    print("-" * 72)
    base_ppl = history[0][3]
    final_ppl = history[-1][3]
    print("[result] perplexity: baseline=%.2f -> final=%.2f  (%.1fx lower)"
          % (base_ppl, final_ppl, base_ppl / max(final_ppl, 1e-9)))
    base_acc = history[0][4]
    final_acc = history[-1][4]
    print("[result] downstream next-char accuracy: baseline=%.3f -> final=%.3f"
          % (base_acc, final_acc))

    plot_ppl(history)

    # --- 评测结论 2:解码策略对比 ---
    # 同一个模型、同一个 prompt,不同解码策略的输出差异。
    print("-" * 72)
    print("[decoding] same model, same prompt, different decoding strategies:")
    prompt = "the "
    g = generate(model, vocab, prompt, steps=40, device=device, strategy="greedy")
    t_lo = generate(model, vocab, prompt, steps=40, device=device, strategy="temperature", temperature=0.5)
    t_hi = generate(model, vocab, prompt, steps=40, device=device, strategy="temperature", temperature=1.3)
    tk = generate(model, vocab, prompt, steps=40, device=device, strategy="top_k", temperature=1.0, top_k=5)
    # repr() 让换行/空格可见,且强制 ASCII 安全输出
    print("  greedy          : %s" % repr(g))
    print("  temperature=0.5 : %s" % repr(t_lo))
    print("  temperature=1.3 : %s" % repr(t_hi))
    print("  top_k=5         : %s" % repr(tk))

    # --- 量化"成功"信号 ---
    print("-" * 72)
    success = (final_ppl < base_ppl) and (final_acc > base_acc)
    print("[summary] perplexity dropped: %s | downstream improved: %s"
          % (final_ppl < base_ppl, final_acc > base_acc))
    print("[summary] OVERALL: %s" % ("PASS - the model measurably learned" if success
                                     else "CHECK - metrics did not improve as expected"))
    print("=" * 72)


if __name__ == "__main__":
    main()
