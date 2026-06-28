"""
从零实现投机采样 / 投机解码（Speculative Decoding）—— toy 字符级语言模型版

================================================================================
这是什么？
--------------------------------------------------------------------------------
投机采样（Speculative Decoding, Leviathan et al. 2022, arXiv:2211.17192）是大模型
推理加速的核心技术之一。它的核心矛盾与解法：

  * 矛盾：大模型（target）自回归解码是串行的，一次前向只产出 1 个 token，
          访存/算力利用率低，延迟高。
  * 解法：用一个便宜的小模型（draft / 草稿模型）一口气“猜” k 个 token，
          然后让大模型“并行”地一次前向验证这 k 个候选 token。
          被接受的 token 直接采纳，第一个被拒绝的位置用一个修正分布重采样。

关键性质（本脚本会用实验验证）：
  * 无损（lossless）：经过“接受-拒绝 + 修正分布重采样”后，最终采样得到的 token
    序列分布与“纯 target 自回归采样”在统计上完全一致。小模型只影响速度，不影响分布。
  * 加速来源：当 draft 猜得准时，大模型一次前向能“确认”多个 token，
    平均每次 target 前向产出的 token 数 = 平均接受长度 > 1。

本脚本不依赖任何外部数据集 / 网络 / gym：
  * 自己造一份 toy 文本语料（重复的字符模式），训练两个字符级 LM：
      - target：稍大的 bigram/统计模型（作为“真理”分布）
      - draft ：更小/更糙的统计模型（猜测者）
  * 用 numpy 实现自回归采样基线与投机采样，逐项对齐验证“无损”。

为什么用统计字符模型而不是 Transformer？
  * 投机采样的“接受准则 + 修正分布”是与具体模型无关的概率算法。
  * 用可解析、可枚举的小模型，能在 CPU 上几十秒内严格验证“分布等价”，
    教学信号最干净。真实工程里把 draft/target 换成两个 Transformer 即可，算法不变。

对应 llm-action 文档：../../llm-inference/README.md 第 5.5 节「解码加速：投机解码」。
================================================================================
"""

import numpy as np

# 固定随机种子，保证可复现
SEED = 20240621


# ============================================================================
# 1) toy 语料 + 字符级 n-gram 语言模型
# ============================================================================
def build_corpus():
    """构造一份有明显局部规律的 toy 语料。

    规律越强，draft（小模型）越容易猜中 target，接受长度越高 —— 便于观察加速。
    """
    # 一段带重复结构的字符串：字符之间存在较强的条件依赖
    base = "the quick brown fox jumps over the lazy dog . "
    return base * 60


def build_vocab(text):
    """建立字符 <-> id 的双向映射。"""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


class CharNGramLM:
    """字符级 n-gram 统计语言模型。

    用前 (order) 个字符作为上下文，对下一个字符给出一个完整的概率分布
    P(next | context)。我们需要“完整分布”而不仅仅是 argmax，因为投机采样的
    接受准则要逐 token 比较 draft 与 target 的概率。
    """

    def __init__(self, order, vocab_size, smoothing):
        self.order = order              # 上下文长度（n-gram 的 n-1）
        self.V = vocab_size
        self.smoothing = smoothing      # 加性平滑系数；越大分布越“糊”（模型越糙）
        # counts[context_tuple] -> np.array(V,)  统计每个上下文下各字符出现次数
        self.counts = {}

    def fit(self, ids):
        """从 token id 序列里统计 n-gram 频次。"""
        order = self.order
        for i in range(order, len(ids)):
            ctx = tuple(ids[i - order:i])
            if ctx not in self.counts:
                self.counts[ctx] = np.zeros(self.V, dtype=np.float64)
            self.counts[ctx][ids[i]] += 1.0
        return self

    def dist(self, context_ids):
        """给定上下文（id 列表），返回下一个字符的概率分布 (V,)。

        这一步对应原理：语言模型的本质就是给出 P(x_t | x_<t)。
        投机采样需要 draft 和 target 都能吐出“分布”，而不仅是一个 token。
        """
        # 只取最后 order 个字符作为上下文；不足则左侧用 0 号 token 补齐
        ctx = list(context_ids[-self.order:])
        while len(ctx) < self.order:
            ctx = [0] + ctx
        ctx = tuple(ctx)

        # 加性（拉普拉斯）平滑：保证每个字符概率 > 0，
        # 这对投机采样很关键 —— 否则修正分布可能出现除零。
        counts = self.counts.get(ctx)
        if counts is None:
            # 没见过的上下文：退化成均匀分布
            probs = np.ones(self.V, dtype=np.float64)
        else:
            probs = counts + self.smoothing
        probs = probs / probs.sum()
        return probs


# ============================================================================
# 2) 基线：target 模型纯自回归采样
# ============================================================================
def autoregressive_sample(target_lm, prompt_ids, n_new, rng):
    """纯 target 自回归采样：每步前向一次，采 1 个 token。

    这是投机采样要对齐的“黄金分布”。统计 target 前向次数用于算加速比。
    """
    ids = list(prompt_ids)
    target_forwards = 0
    for _ in range(n_new):
        p = target_lm.dist(ids)     # 一次 target 前向
        target_forwards += 1
        nxt = rng.choice(len(p), p=p)
        ids.append(int(nxt))
    return ids, target_forwards


# ============================================================================
# 3) 投机采样核心
# ============================================================================
def speculative_sample(target_lm, draft_lm, prompt_ids, n_new, k, rng):
    """投机采样主循环。

    每一轮（while）：
      1) draft 从当前序列出发，自回归地“猜” k 个 token，并记录每个 token 处的
         draft 分布 q（猜测分布）。  —— 便宜的小模型串行 k 步。
      2) target 对这 k 个候选位置“并行”给出分布 p。
         （toy 实现里我们用循环逐位求 p，但概念上 target 只做 1 次批量前向，
          因此每一轮只计 1 次 target 前向。这正是加速的来源。）
      3) 逐个 token 应用接受准则：
            对候选 token x（draft 采出），以概率 min(1, p(x)/q(x)) 接受。
            - 若接受：采纳 x，继续看下一个候选。
            - 若拒绝：在第一个被拒位置，从“修正分布” norm((p - q)_+) 重采 1 个 token，
              然后丢弃该位置之后所有 draft 候选，结束本轮。
      4) 若 k 个候选全部被接受，则用 target 在第 k+1 个位置的分布额外采 1 个 token
         （“免费的”一步），这样每轮最多前进 k+1 个 token。

    这套“接受概率 min(1,p/q) + 修正分布 (p-q)_+”可证明：最终每个被产出的 token
    都服从 target 分布 p。这就是“无损”的数学保证（见 arXiv:2211.17192 定理 1）。
    """
    ids = list(prompt_ids)
    target_forwards = 0          # target 前向次数（加速比的分母核心）
    draft_forwards = 0           # draft 前向次数（仅供观察，便宜）
    accepted_lengths = []        # 每轮接受的 token 数，用于算平均接受长度
    produced = 0

    while produced < n_new:
        # ---- 步骤 1：draft 猜 k 个 token，记录 draft 分布 q ----
        draft_seq = list(ids)
        proposals = []           # draft 提议的 token
        q_list = []              # 每个提议处的 draft 分布
        steps = min(k, n_new - produced)
        for _ in range(steps):
            q = draft_lm.dist(draft_seq)   # 一次 draft 前向（便宜）
            draft_forwards += 1
            x = int(rng.choice(len(q), p=q))
            proposals.append(x)
            q_list.append(q)
            draft_seq.append(x)

        # ---- 步骤 2：target 对这 (steps) 个位置并行打分（概念上 1 次前向）----
        # p_list[j] = P_target(next | ids + proposals[:j])
        p_list = []
        ctx = list(ids)
        for j in range(steps):
            p_list.append(target_lm.dist(ctx))
            ctx.append(proposals[j])
        # 还需要“全接受后”那一位的 target 分布，用于免费多采一个 token
        p_extra = target_lm.dist(ctx)
        target_forwards += 1     # 本轮 target 只计 1 次前向（并行验证 + 1 个 bonus 位）

        # ---- 步骤 3：逐 token 接受 / 拒绝 ----
        n_accept = 0
        rejected = False
        for j in range(steps):
            x = proposals[j]
            p = p_list[j]
            q = q_list[j]
            # 接受概率 = min(1, p(x)/q(x))
            # 直觉：若 target 比 draft 更看好 x（p>=q），必接受；
            #       若 target 没那么看好，则按比例随机接受。
            accept_prob = min(1.0, p[x] / q[x])
            if rng.random() < accept_prob:
                ids.append(x)
                produced += 1
                n_accept += 1
                if produced >= n_new:
                    break
            else:
                # 第一个被拒位置：从修正分布 (p - q)_+ / Z 重采一个 token。
                # 这一步“纠偏”保证了无损：把 draft 多采的概率质量扣掉，
                # 把 draft 漏采的概率质量补回来。
                residual = np.clip(p - q, 0.0, None)
                s = residual.sum()
                if s <= 0:
                    # 数值兜底（平滑后几乎不会发生）：直接用 target 分布
                    residual = p
                    s = residual.sum()
                residual = residual / s
                x_fix = int(rng.choice(len(residual), p=residual))
                ids.append(x_fix)
                produced += 1
                rejected = True
                break

        accepted_lengths.append(n_accept)

        # ---- 步骤 4：若全部接受且还没采够，用 p_extra 免费多采 1 个 token ----
        if (not rejected) and produced < n_new and n_accept == steps:
            x_bonus = int(rng.choice(len(p_extra), p=p_extra))
            ids.append(x_bonus)
            produced += 1

    stats = {
        "target_forwards": target_forwards,
        "draft_forwards": draft_forwards,
        "avg_accept_len": float(np.mean(accepted_lengths)) if accepted_lengths else 0.0,
        "rounds": len(accepted_lengths),
    }
    return ids, stats


# ============================================================================
# 4) 无损性验证：投机采样 vs target 自回归，分布是否一致
# ============================================================================
def verify_distribution(target_lm, draft_lm, prompt_ids, k, rng, n_trials):
    """蒙特卡洛验证“无损”。

    固定 prompt，分别用 target 自回归 与 投机采样 各采很多次“下一个 token”，
    统计两者的经验分布，比较总变差距离（TV distance）。若投机采样无损，
    随采样次数增大，两个经验分布应收敛到一致（TV -> 0）。
    """
    V = target_lm.V
    counts_ar = np.zeros(V)
    counts_sp = np.zeros(V)

    plen = len(prompt_ids)
    for _ in range(n_trials):
        # 基线：纯 target 采 1 个 token，统计新采出的那个 token（位置 = plen）
        out_ar, _ = autoregressive_sample(target_lm, prompt_ids, 1, rng)
        counts_ar[out_ar[plen]] += 1
    for _ in range(n_trials):
        # 投机采样采 1 个 token（k 不影响第 1 个 token 的边缘分布）
        out_sp, _ = speculative_sample(target_lm, draft_lm, prompt_ids, 1, k, rng)
        counts_sp[out_sp[plen]] += 1

    emp_ar = counts_ar / counts_ar.sum()
    emp_sp = counts_sp / counts_sp.sum()
    # 同时给出 target 的真实理论分布作参照
    true_p = target_lm.dist(prompt_ids)

    tv_sp_vs_ar = 0.5 * np.abs(emp_sp - emp_ar).sum()
    tv_sp_vs_true = 0.5 * np.abs(emp_sp - true_p).sum()
    tv_ar_vs_true = 0.5 * np.abs(emp_ar - true_p).sum()
    return tv_sp_vs_ar, tv_sp_vs_true, tv_ar_vs_true


# ============================================================================
# 5) 画图（可选，失败则纯文本）
# ============================================================================
def maybe_plot(accept_hist, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境，存文件
        import matplotlib.pyplot as plt
        xs = sorted(accept_hist.keys())
        ys = [accept_hist[x] for x in xs]
        plt.figure(figsize=(6, 4))
        plt.bar(xs, ys, color="#4C72B0")
        plt.xlabel("accepted tokens per round")
        plt.ylabel("frequency")
        plt.title("Speculative decoding: accept-length distribution")
        plt.tight_layout()
        plt.savefig(out_path, dpi=110)
        plt.close()
        return out_path
    except Exception as e:
        print("[plot] skipped (matplotlib unavailable): %s" % type(e).__name__)
        return None


# ============================================================================
# 6) demo
# ============================================================================
def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("Speculative Decoding (toy char-level LM) demo")
    print("=" * 70)

    # ---- 数据 + 词表 ----
    text = build_corpus()
    stoi, itos = build_vocab(text)
    V = len(stoi)
    ids = [stoi[c] for c in text]
    print("corpus chars=%d  vocab_size=%d" % (len(text), V))

    # ---- 训练两个模型 ----
    # target：order=3 上下文更长、平滑更小 -> 分布更尖锐、更“聪明”（充当真理）
    target_lm = CharNGramLM(order=3, vocab_size=V, smoothing=0.01).fit(ids)
    # draft ：order=1 上下文更短、平滑更大 -> 更糙更快（充当猜测者）
    draft_lm = CharNGramLM(order=1, vocab_size=V, smoothing=0.30).fit(ids)
    print("target = 3-gram (sharp) | draft = 1-gram (coarse)")

    # ---- prompt ----
    prompt = "the quick brown "
    prompt_ids = [stoi[c] for c in prompt]
    n_new = 200
    k = 4  # draft 每轮提议的 token 数

    # ---- 基线：纯 target 自回归 ----
    ar_ids, ar_target_fwd = autoregressive_sample(target_lm, prompt_ids, n_new, rng)
    ar_text = "".join(itos[i] for i in ar_ids[len(prompt_ids):])
    print("-" * 70)
    print("[autoregressive] target_forwards=%d (one per token)" % ar_target_fwd)
    print("[autoregressive] sample: %r" % ar_text[:60])

    # ---- 投机采样 ----
    sp_ids, stats = speculative_sample(target_lm, draft_lm, prompt_ids, n_new, k, rng)
    sp_text = "".join(itos[i] for i in sp_ids[len(prompt_ids):])
    print("-" * 70)
    print("[speculative]  k=%d  rounds=%d" % (k, stats["rounds"]))
    print("[speculative]  target_forwards=%d  draft_forwards=%d"
          % (stats["target_forwards"], stats["draft_forwards"]))
    print("[speculative]  avg_accept_len=%.3f (tokens accepted per target forward, before bonus)"
          % stats["avg_accept_len"])
    print("[speculative]  sample: %r" % sp_text[:60])

    # ---- 加速比 ----
    # 衡量标准：产出同样多的 token，target 前向次数减少多少。
    # speedup = (target forwards if pure AR) / (target forwards in speculative)
    speedup = ar_target_fwd / stats["target_forwards"]
    tokens_per_forward = n_new / stats["target_forwards"]
    print("-" * 70)
    print("[speedup] target forwards: AR=%d -> SPEC=%d"
          % (ar_target_fwd, stats["target_forwards"]))
    print("[speedup] tokens per target forward = %.3f" % tokens_per_forward)
    print("[speedup] relative speedup (fewer target forwards) = %.2fx" % speedup)

    # ---- 无损性验证 ----
    print("-" * 70)
    print("[lossless check] Monte-Carlo: speculative vs target distribution ...")
    verify_prompt = [stoi[c] for c in "the "]
    tv_sp_ar, tv_sp_true, tv_ar_true = verify_distribution(
        target_lm, draft_lm, verify_prompt, k=k, rng=rng, n_trials=8000)
    print("[lossless check] TV(speculative , autoregressive) = %.4f" % tv_sp_ar)
    print("[lossless check] TV(speculative , target_true)    = %.4f" % tv_sp_true)
    print("[lossless check] TV(autoregress , target_true)    = %.4f" % tv_ar_true)
    # 判定：投机采样与基线的差距应与“基线本身的采样噪声”同量级
    ok_lossless = tv_sp_true <= tv_ar_true + 0.02
    print("[lossless check] speculative matches target within sampling noise: %s"
          % ("YES" if ok_lossless else "NO"))

    # ---- 接受长度直方图 + 画图 ----
    # 为了直方图，复跑一段轻量循环，逐轮收集“本轮接受了几个 token”。
    rng3 = np.random.default_rng(SEED + 2)
    accept_hist = {}
    produced = 0
    seq = list(prompt_ids)
    while produced < 400:
        draft_seq = list(seq)
        proposals, q_list = [], []
        steps = min(k, 400 - produced)
        for _ in range(steps):
            q = draft_lm.dist(draft_seq)
            x = int(rng3.choice(V, p=q))
            proposals.append(x); q_list.append(q); draft_seq.append(x)
        ctx = list(seq); p_list = []
        for j in range(steps):
            p_list.append(target_lm.dist(ctx)); ctx.append(proposals[j])
        n_acc = 0
        for j in range(steps):
            x = proposals[j]; p = p_list[j]; q = q_list[j]
            if rng3.random() < min(1.0, p[x] / q[x]):
                seq.append(x); produced += 1; n_acc += 1
            else:
                residual = np.clip(p - q, 0.0, None); residual /= residual.sum()
                seq.append(int(rng3.choice(V, p=residual))); produced += 1
                break
        accept_hist[n_acc] = accept_hist.get(n_acc, 0) + 1

    print("-" * 70)
    print("[accept-length histogram] (rounds grouped by #accepted tokens)")
    for x in sorted(accept_hist):
        print("  accepted=%d : %d rounds" % (x, accept_hist[x]))

    png = maybe_plot(accept_hist, "accept_length_hist.png")
    if png:
        print("[plot] saved %s" % png)

    # ---- 总结 ----
    print("=" * 70)
    print("SUMMARY")
    print("  - speculative produced %d tokens using only %d target forwards"
          % (n_new, stats["target_forwards"]))
    print("  - speedup vs pure autoregressive: %.2fx (target forwards)" % speedup)
    print("  - distribution lossless vs target: %s" % ("PASS" if ok_lossless else "CHECK"))
    print("=" * 70)


if __name__ == "__main__":
    main()
