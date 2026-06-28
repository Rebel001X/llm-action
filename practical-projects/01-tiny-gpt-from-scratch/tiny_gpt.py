# -*- coding: utf-8 -*-
"""
从零实现并预训练一个字符级 GPT (Tiny Character-level GPT)
==========================================================

这是什么
--------
本文件用纯 PyTorch 从零实现一个最小可用的 GPT (decoder-only Transformer),
并在一小段内置文本上做字符级 (character-level) 预训练, 全程 CPU 几十秒内跑完。
目的不是性能, 而是把 GPT 的每一块"看得见、摸得着":

    Token Embedding + Positional Embedding   ->  把字符映射成向量, 并注入位置信息
    Multi-Head Self-Attention (因果 mask)     ->  让每个位置只能看到自己和左边的历史
    Position-wise Feed-Forward (MLP)          ->  对每个位置独立做非线性变换
    LayerNorm + 残差连接 (Pre-LN)              ->  稳定训练、让梯度顺畅回传
    最后一层 Linear -> 词表 logits             ->  预测"下一个字符"

训练目标就是经典的自回归语言建模 (next-token prediction):
给定前缀, 预测下一个字符, 用交叉熵作为损失。训练后从模型里采样,
就能看到它学到了训练文本的"风格"。

为什么是字符级 + toy 规模
------------------------
字符级不需要分词器 (tokenizer), 词表就是文本里出现过的所有字符, 自包含、零依赖、
零下载。模型只有几层、几十维, 数据只有几 KB, 因此在纯 CPU 上几十秒能跑完,
非常适合教学与在 CI 里冒烟验证。

注意 (Windows 控制台编码)
------------------------
本文件所有 print 只输出 ASCII。中文仅出现在注释/docstring 里, 避免在
GBK 控制台下打印中文触发 UnicodeEncodeError。

运行:
    python tiny_gpt.py
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0. 可复现设置: 固定随机种子, 保证每次运行结果一致 (教学/CI 友好)
# =============================================================================
SEED = 1337
torch.manual_seed(SEED)


# =============================================================================
# 1. 数据: 内置一小段文本 (英文, 因为是字符级建模, 风格更直观)
#    这里硬编码一段莎士比亚风格的独白片段, 几 KB, 不依赖任何外部数据集/网络。
# =============================================================================
TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""


# =============================================================================
# 2. 构建字符级"词表 (vocabulary)": 文本里出现过的所有不同字符
#    - stoi: char -> int (string-to-int)
#    - itos: int  -> char (int-to-string)
#    这一步对应 GPT 里的 "tokenizer", 只不过字符级最简单: 一个字符就是一个 token。
# =============================================================================
chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """字符串 -> 整数 id 列表 (每个字符查 stoi)。"""
    return [stoi[c] for c in s]


def decode(ids):
    """整数 id 列表 -> 字符串 (每个 id 查 itos)。"""
    return "".join(itos[int(i)] for i in ids)


# 把整段文本编码成一个 1D 张量, 后面切片采样训练样本。
data = torch.tensor(encode(TEXT), dtype=torch.long)


# =============================================================================
# 3. 超参数 (toy 规模, 保证 CPU 几十秒跑完)
# =============================================================================
block_size = 64      # 上下文长度 (context length): 一次最多看多少个历史字符
batch_size = 32      # 每步并行训练多少条序列
n_embd = 96          # 嵌入维度 (每个 token 向量的长度, d_model)
n_head = 4           # 多头注意力的头数 (n_embd 必须能被 n_head 整除)
n_layer = 3          # Transformer Block 堆叠层数
dropout = 0.1        # dropout 比例, 轻度正则
learning_rate = 3e-3
max_iters = 600      # 训练步数 (toy, 足够看到 loss 明显下降)
eval_interval = 100  # 每隔多少步评估并打印一次
eval_iters = 50      # 评估时平均多少个 batch, 让 loss 估计更稳

assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
head_size = n_embd // n_head  # 每个注意力头的维度


def get_batch():
    """
    随机采一个 mini-batch。
    - x: (batch_size, block_size) 输入序列
    - y: (batch_size, block_size) 目标序列, 即 x 整体右移一位
      含义: 在每个位置 t, 模型看 x[:t+1], 要预测 y[t] = x[t+1] (下一个字符)。
    这正是自回归语言建模的训练信号。
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


# =============================================================================
# 4. 模型组件
# =============================================================================
class MultiHeadSelfAttention(nn.Module):
    """
    多头因果自注意力 (Causal Multi-Head Self-Attention)。

    原理:
      对每个 token, 用三个线性投影得到 Query / Key / Value。
      注意力得分 = Q @ K^T / sqrt(head_size)  (缩放点积, 防止维度大时 softmax 饱和)
      因果 mask: 把每个位置"未来"的得分置为 -inf, 保证位置 t 只能看到 <= t,
                 这就是 GPT 这种 decoder-only 模型"自回归"的关键。
      softmax 后得到注意力权重, 再加权求和 Value, 得到每个位置的新表示。
      多头: 把 d_model 切成 n_head 份并行做注意力, 让模型在不同子空间学不同关系。
    """

    def __init__(self):
        super().__init__()
        # 一次性算出所有头的 Q,K,V (合并成一个大矩阵更高效)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)  # 多头拼回后的输出投影
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # 下三角因果 mask, 注册为 buffer (不是参数, 不参与训练但随模型保存/搬移)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, time(序列长度), channels(=n_embd)
        # 计算 Q,K,V 并拆成多头: (B, n_head, T, head_size)
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        # 缩放点积注意力得分: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        # 因果 mask: 未来位置填 -inf, softmax 后权重为 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # 加权求和 Value: (B, n_head, T, head_size)
        # 把多头拼回 (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise 前馈网络 (MLP)。
    对每个位置独立地做: Linear -> 非线性(GELU) -> Linear。
    通常中间层放大到 4 * n_embd, 给模型更多非线性表达能力。
    注意力负责"在序列内交换信息", FFN 负责"逐位置加工特征"。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    一个 Transformer Block (Pre-LN 结构, 与 GPT-2 一致):
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    残差连接让深层网络可训练 (梯度高速公路);
    Pre-LN (先归一化再进子层) 比 Post-LN 训练更稳定。
    """

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差 + 注意力子层
        x = x + self.ffn(self.ln2(x))   # 残差 + 前馈子层
        return x


class TinyGPT(nn.Module):
    """
    最小 GPT (decoder-only Transformer)。
    """

    def __init__(self):
        super().__init__()
        # token 嵌入: 每个字符 id -> n_embd 维向量
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # 位置嵌入: 每个位置 (0..block_size-1) -> n_embd 维向量
        # Transformer 本身对顺序不敏感, 必须显式注入位置信息。
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)            # 最终 LayerNorm
        self.head = nn.Linear(n_embd, vocab_size)   # 输出层: 投影到词表 logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        # 词义信息 + 位置信息相加, 得到每个 token 的初始表示
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size): 每个位置对下一个字符的打分

        loss = None
        if targets is not None:
            # 交叉熵: 把 (B,T,vocab) 拍平成 (B*T, vocab) 与 (B*T,) 目标对齐
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        自回归采样: 每次取最后 block_size 个 token 作上下文, 预测下一个字符的分布,
        从分布里采样一个字符, 拼接到序列末尾, 重复 max_new_tokens 次。
        这就是 GPT "生成文本"的过程。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # 截断到上下文窗口
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                # 只看最后一个位置的预测
            probs = F.softmax(logits, dim=-1)        # 转成概率分布
            next_id = torch.multinomial(probs, num_samples=1)  # 按概率采样
            idx = torch.cat((idx, next_id), dim=1)   # 拼接
        return idx


# =============================================================================
# 5. 评估: 在多个随机 batch 上平均 loss, 让打印的 loss 更平滑可信
# =============================================================================
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch()
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# =============================================================================
# 6. 画 loss 曲线 (可选): 用 Agg 后端存 png; 若无 matplotlib 则纯文本打印, 不崩。
# =============================================================================
def plot_losses(steps, losses, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面环境必须用 Agg, 直接写文件
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(steps, losses, marker="o")
        plt.xlabel("training step")
        plt.ylabel("cross-entropy loss")
        plt.title("TinyGPT training loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print("[plot] saved loss curve to %s" % out_path)
    except Exception as e:  # noqa: BLE001  教学代码: 画图失败不影响主流程
        print("[plot] matplotlib unavailable (%s); skip figure." % type(e).__name__)


# =============================================================================
# 7. 主流程 demo
# =============================================================================
def main():
    print("=" * 60)
    print("Tiny Character-level GPT  (pure PyTorch, CPU)")
    print("=" * 60)
    print("device          : cpu")
    print("vocab_size      : %d" % vocab_size)
    print("dataset chars   : %d" % len(data))
    print("block_size      : %d" % block_size)
    print("n_layer/n_head  : %d / %d" % (n_layer, n_head))
    print("n_embd          : %d" % n_embd)

    model = TinyGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters      : %d (%.2fK)" % (n_params, n_params / 1e3))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---- 训练前: 看一眼未训练模型的 loss 和生成 (应接近随机, loss ~ ln(vocab)) ----
    random_baseline = math.log(vocab_size)
    print("-" * 60)
    print("random-guess loss (ln vocab_size) ~ %.4f" % random_baseline)

    steps_log, loss_log = [], []
    init_loss = estimate_loss(model)
    print("step %4d | eval loss %.4f  (before training)" % (0, init_loss))
    steps_log.append(0)
    loss_log.append(init_loss)

    # ---- 训练循环 ----
    for it in range(1, max_iters + 1):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_interval == 0 or it == max_iters:
            eval_loss = estimate_loss(model)
            print("step %4d | eval loss %.4f" % (it, eval_loss))
            steps_log.append(it)
            loss_log.append(eval_loss)

    final_loss = loss_log[-1]
    print("-" * 60)
    print("SUMMARY")
    print("  init  eval loss : %.4f" % init_loss)
    print("  final eval loss : %.4f" % final_loss)
    print("  loss reduction  : %.4f  (%.1f%% lower)"
          % (init_loss - final_loss, 100.0 * (init_loss - final_loss) / init_loss))
    # 量化"成功"信号: 训练后 loss 应明显低于随机基线
    success = final_loss < random_baseline * 0.8
    print("  beats 0.8x random-baseline : %s" % ("YES" if success else "NO"))

    # ---- 训练后采样: 从一个换行符起手, 生成一段文本展示学到的风格 ----
    print("-" * 60)
    print("SAMPLE (generated after training):")
    start = torch.tensor([encode("\n")], dtype=torch.long)
    out = model.generate(start, max_new_tokens=300)[0]
    sample = decode(out)
    # 只打印 ASCII; 训练文本本身是英文, 这里直接输出即可
    print(sample.encode("ascii", "replace").decode("ascii"))
    print("-" * 60)

    # ---- 画 loss 曲线 (可选) ----
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_curve.png")
    plot_losses(steps_log, loss_log, out_png)

    print("DONE.")


if __name__ == "__main__":
    main()
