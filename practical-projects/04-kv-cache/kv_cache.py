"""
从零实现 KV Cache 推理加速 (KV Cache for Autoregressive Inference)
================================================================

这是什么
--------
本文件用一个 *toy* 规模的 GPT(decoder-only Transformer),在 CPU 上对比
自回归生成的两条路径,直观演示 KV Cache 为什么能加速大模型推理:

  (a) 无 cache(naive):每生成一个新 token,都把"已生成的整段序列"重新
      喂进模型,从头算一遍所有 token 的 Q/K/V 与注意力。第 t 步要算 t 个
      token,总计算量是 1+2+...+n ~ O(n^2)。

  (b) KV cache:注意到自回归是因果(causal)的——已经算过的 token,它们
      的 Key/Value 不会因为后面追加新 token 而改变。于是把每层每个历史
      token 的 K、V 缓存起来;每步只对"新来的 1 个 token"算 Q/K/V,再把
      它的 K/V 追加进 cache,与全部历史 K/V 做注意力。每步只算 1 个 token,
      总计算量 ~ O(n)。

核心结论:KV Cache 把"每步重算整段"变成"每步只算新 token",把生成阶段
(decode)的时间复杂度从 O(n^2) 降到 O(n)。两条路径在数学上等价,因此
*生成的 token 序列必须逐一相同*——本脚本会断言这一点,并打印随序列变长
而增大的加速比。

为什么这是 LLM 推理的基石
------------------------
真实的 LLM 服务(vLLM / TGI / TensorRT-LLM 等)都依赖 KV Cache。后续的
PagedAttention、KV Cache 量化、KV Cache offload、prefix caching 等优化,
本质都是"如何更省内存 / 更高吞吐地管理这块 cache"。理解了这里的 toy 版,
再看那些系统级优化就有了地基。

依赖:numpy, torch(CPU 即可)。无需数据集 / 网络。
运行:python kv_cache.py
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 复现性:固定随机种子,保证每次运行的随机初始化权重一致,数字可复现。
# ----------------------------------------------------------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)  # 单线程,让计时更稳定、更能反映算法复杂度差异


# ----------------------------------------------------------------------------
# 1. 一个最小的因果自注意力层 (causal self-attention)
#    它同时支持两种调用方式:
#      - 无 cache:传入整段序列 x,内部自己造因果 mask。
#      - 有 cache:传入"仅新 token" x_new,以及历史 (k_cache, v_cache);
#        本层把新 token 的 k/v 追加进 cache,然后对全部历史做注意力。
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # 一次性投影出 Q、K、V(拼在一起,效率更高,GPT 的常见写法)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def _split_heads(self, t):
        # (B, T, n_embd) -> (B, n_head, T, head_dim)
        B, T, C = t.shape
        return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x, kv_cache=None):
        """
        x: (B, T, n_embd)。
           - 无 cache 路径:T = 当前整段长度。
           - 有 cache 路径:T = 1(只传新 token)。
        kv_cache: None 或 (k_cache, v_cache),形状 (B, n_head, T_past, head_dim)。
        返回: (out, new_kv_cache)
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)          # 各 (B, T, C)
        q = self._split_heads(q)                        # (B, nh, T, hd)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if kv_cache is not None:
            # ---- KV Cache 的关键一步 ----
            # 历史 token 的 K/V 早已算好且不会改变(因果性),直接复用;
            # 只需把"新 token"的 k/v 拼到历史后面。这就是省下重复计算的地方。
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)           # 沿时间维拼接
            v = torch.cat([v_past, v], dim=2)
        new_kv_cache = (k, v)                            # 更新后的 cache 交回上层

        # 注意力打分:Q·K^T / sqrt(d)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nh,Tq,Tk)

        # 因果 mask:位置 i 只能看到 <= i 的位置。
        Tq, Tk = att.shape[-2], att.shape[-1]
        if kv_cache is None:
            # 无 cache:Tq == Tk == 整段长度,用标准下三角 mask。
            mask = torch.tril(torch.ones(Tq, Tk, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
        else:
            # 有 cache:Tq == 1(新 token),它在序列末尾,能看到所有 Tk 个历史
            # (含自己),因此本来就不需要再 mask 掉任何列。这正体现了"只算新
            # token 对全部历史的注意力"。
            pass

        att = F.softmax(att, dim=-1)
        y = att @ v                                     # (B, nh, Tq, hd)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)  # 合并多头
        return self.proj(y), new_kv_cache


# ----------------------------------------------------------------------------
# 2. 一个 Transformer block:注意力 + 前馈,均带残差与 LayerNorm(pre-norm)。
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, kv_cache=None):
        # 注意:LayerNorm/MLP 是 *逐 token* 的运算,对新 token 单独算与对整段
        # 算结果完全一致——这正是 KV Cache 能在数学上等价的前提之一。
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


# ----------------------------------------------------------------------------
# 3. 一个最小 GPT。提供两套前向:
#      - forward_full(idx):无 cache,喂整段,返回最后一个位置的 logits。
#      - forward_step(idx_new, caches):有 cache,只喂新 token + 历史 cache。
# ----------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # token 嵌入
        self.pos_emb = nn.Embedding(block_size, n_embd)   # 位置嵌入(绝对位置)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    # --- 路径 (a):无 cache,每步重算整段 ---
    def forward_full(self, idx):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)         # (B, T, n_embd)
        for blk in self.blocks:
            x, _ = blk(x, kv_cache=None)                  # 丢弃 cache
        x = self.ln_f(x)
        logits = self.head(x)                             # (B, T, vocab)
        return logits[:, -1, :]                           # 只要最后一步的预测

    # --- 路径 (b):有 cache,只算新 token ---
    def forward_step(self, idx_new, caches, pos_offset):
        """
        idx_new: (B, 1) 新 token。
        caches:  长度 n_layer 的列表,每项是该层的 (k_cache, v_cache) 或 None。
        pos_offset: 新 token 的绝对位置下标(= 已生成长度)。
        """
        B, T = idx_new.shape  # T == 1
        pos = torch.arange(pos_offset, pos_offset + T)
        x = self.tok_emb(idx_new) + self.pos_emb(pos)     # 只嵌入这 1 个 token
        new_caches = []
        for blk, cache in zip(self.blocks, caches):
            x, new_kv = blk(x, kv_cache=cache)            # 复用历史 K/V
            new_caches.append(new_kv)
        x = self.ln_f(x)
        logits = self.head(x)                             # (B, 1, vocab)
        return logits[:, -1, :], new_caches


# ----------------------------------------------------------------------------
# 4. 两种生成函数。为保证可比较,都用 *贪心解码*(取 argmax),这样输出确定。
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_no_cache(model, prompt, n_new):
    """无 cache:每步把已生成整段重新前向一遍。O(n^2) 计算量。"""
    idx = prompt.clone()
    for _ in range(n_new):
        idx_cond = idx[:, -model.block_size:]            # 截断到上下文窗口
        logits = model.forward_full(idx_cond)            # 整段重算
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


@torch.no_grad()
def generate_with_cache(model, prompt, n_new):
    """KV cache:先 prefill 一次(整段),之后每步只喂新 token。O(n) 计算量。"""
    idx = prompt.clone()
    B, T0 = idx.shape

    # ---- Prefill 阶段:把 prompt 整段过一遍,建立初始 KV cache ----
    # (真实系统里这一步叫 prefill;它仍是一次性的整段计算。)
    pos = torch.arange(T0)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    caches = []
    for blk in model.blocks:
        x, new_kv = blk(x, kv_cache=None)
        caches.append(new_kv)                            # 保存 prompt 的 K/V
    x = model.ln_f(x)
    logits = model.head(x)[:, -1, :]
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)

    # ---- Decode 阶段:每步只喂 1 个新 token,复用并扩展 cache ----
    for step in range(1, n_new):
        pos_offset = idx.shape[1] - 1                    # 新 token 的绝对位置
        logits, caches = model.forward_step(next_id, caches, pos_offset)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


# ----------------------------------------------------------------------------
# 5. 计时小工具:跑一次并返回 (输出, 耗时秒)。
# ----------------------------------------------------------------------------
def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def main():
    # ---- toy 超参:小到 CPU 几十秒内跑完,但足以看出 O(n^2) vs O(n) 趋势 ----
    vocab_size = 64
    block_size = 256
    n_embd = 64
    n_head = 4
    n_layer = 4

    model = TinyGPT(vocab_size, block_size, n_embd, n_head, n_layer).eval()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 64)
    print("KV Cache demo: naive (recompute all) vs cached (compute new only)")
    print("=" * 64)
    print(f"model: vocab={vocab_size} block={block_size} d={n_embd} "
          f"heads={n_head} layers={n_layer} params={n_params}")
    print()

    # 固定一个短 prompt(随机 token id),两条路径用同一个 prompt。
    prompt = torch.randint(0, vocab_size, (1, 4))

    # ---- 正确性 + 加速比:在多个生成长度上对比 ----
    print(f"{'gen_len':>8} | {'no_cache(s)':>12} | {'cache(s)':>10} | "
          f"{'speedup':>8} | {'match':>6}")
    print("-" * 64)

    lengths = [16, 32, 64, 128]
    speedups = []
    for n_new in lengths:
        out_no, t_no = timed(generate_no_cache, model, prompt, n_new)
        out_kv, t_kv = timed(generate_with_cache, model, prompt, n_new)

        # 正确性:两条路径必须产生逐 token 完全相同的序列。
        match = torch.equal(out_no, out_kv)
        speedup = t_no / t_kv if t_kv > 0 else float("inf")
        speedups.append(speedup)

        print(f"{n_new:>8} | {t_no:>12.4f} | {t_kv:>10.4f} | "
              f"{speedup:>7.2f}x | {str(match):>6}")

        # 硬断言:数学等价,任何不一致都说明实现有 bug。
        assert match, f"MISMATCH at gen_len={n_new}! KV cache is not equivalent."

    print("-" * 64)
    print("All sequences MATCH -> KV cache is mathematically equivalent. PASS")
    print()

    # ---- 复杂度直觉:打印每步处理的 token 数,直观看 O(n^2) vs O(n) ----
    n_demo = 8
    naive_tokens = sum(range(1, n_demo + 1))   # 1+2+...+n,二次增长
    cache_tokens = n_demo                       # 每步只算 1 个,线性增长
    print("Complexity intuition (work = #tokens whose Q/K/V are computed):")
    print(f"  generating {n_demo} tokens, no_cache attends to: "
          f"{list(range(1, n_demo + 1))}  -> total {naive_tokens} (O(n^2))")
    print(f"  generating {n_demo} tokens, with_cache computes:  "
          f"{[1] * n_demo}  -> total {cache_tokens} (O(n))")
    print()
    print(f"speedup grows with length: {[f'{s:.2f}x' for s in speedups]}")
    print(f"max speedup observed: {max(speedups):.2f}x at gen_len={lengths[-1]}")

    # ---- 可选:画加速比随长度变化图;matplotlib 缺失则纯文本,绝不崩 ----
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面后端,直接存 png
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lengths, speedups, marker="o")
        ax.set_xlabel("generated length n")
        ax.set_ylabel("speedup (no_cache_time / cache_time)")
        ax.set_title("KV Cache speedup grows with sequence length")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = "kv_cache_speedup.png"
        fig.savefig(out_png, dpi=120)
        print(f"saved plot -> {out_png}")
    except Exception as e:  # noqa: BLE001  教学代码:任何画图问题都降级为文本
        print(f"(matplotlib unavailable, skipped plot: {e})")

    print("=" * 64)
    print("DONE. Takeaways:")
    print("  1) KV cache outputs are IDENTICAL to naive -> correct.")
    print("  2) speedup increases with sequence length -> O(n^2) collapses to O(n).")
    print("=" * 64)


if __name__ == "__main__":
    main()
