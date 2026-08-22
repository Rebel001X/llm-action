"""
spec.py —— 标准投机采样（Speculative Sampling）的参考实现，纯 numpy，CPU 秒级。

对应论文：
  - Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative
    Decoding", arXiv:2211.17192 (2022-11), ICML 2023. —— Algorithm 1
  - Chen et al. (DeepMind), "Accelerating Large Language Model Decoding with
    Speculative Sampling", arXiv:2302.01318 (2023-02). —— Algorithm 2

本文件提供两条互相印证的路径：
  A. 采样路径 speculative_step()      —— 真的掷骰子，和引擎里跑的一样
  B. 精确枚举路径 iteration_dist()     —— 不掷骰子，把所有分支的概率加起来

B 让「无损」不再靠大样本统计去"看起来相等"，而是**逐点精确相等**（误差 ~1e-16）。
这是本库铁律四（数学断言必须可执行验证）的地基。

用法：
    python spec.py --demo     # 手算算例：|V|=5, gamma=3，打印每一步的具体数字
    python spec.py --proof    # 精确枚举 vs 目标分布，打印最大逐点误差
    python spec.py --bias     # 三种社区常见错法的精确偏差
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------
# 基础：残差分布
# --------------------------------------------------------------------------


def residual_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """被拒绝后重采样用的分布 p' = norm(max(0, p - q))。

    这是社区最高频写错的一处：拒绝之后**不能**直接从 p 重采，
    必须从残差分布采，否则输出分布有偏
    （见 `test_lossless.py::test_naive_resample_is_biased`）。

    退化情形与兜底分支（2026-08-22 对抗审稿更正，初稿这三句都写错了）：
      初稿说"p == q 时残差永远用不到，并在测试里断言这条路不会被走到"。**三处都不对**：
      ① 引的文件名是 `test_spec.py`，本库根本没有这个文件（在 `test_lossless.py`）；
      ② "永远用不到"不成立：1 - sum(min(p,p)) 是浮点噪声，**符号两边都可能**。
         实测 1000 组 (seed, state)：正 236 / 负 109 / 恰好 0 只有 655。
         为正时拒绝分支真的会被走到，这个兜底是**承重的活代码**，不是装饰；
      ③ 从来没有过"断言这条路不会被走到"的测试。
    现在的实现语义：sum(max(0,p-q)) <= 0 时退回 p。此时 p 与 q 逐点相等（两者都是分布），
    退回 p 与退回 q 等价，结果仍然正确 —— 见 test_lossless.py::test_residual_fallback_is_live_and_correct。
    """
    r = np.maximum(0.0, p - q)
    s = r.sum()
    if s <= 0.0:
        return p.copy()
    return r / s


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """全变差距离 TV(p,q) = 0.5 * sum|p-q|。"""
    return 0.5 * float(np.abs(p - q).sum())


def beta_overlap(p: np.ndarray, q: np.ndarray) -> float:
    """单步接受概率 beta = sum_x min(p(x), q(x))。

    恒等式 beta = 1 - TV(p,q)，见 accept.py 与 test_accept.py::test_beta_equals_one_minus_tv。
    """
    return float(np.minimum(p, q).sum())


# --------------------------------------------------------------------------
# 玩具模型：一阶马尔可夫链（让 target / draft 有"上下文"但仍可精确枚举）
# --------------------------------------------------------------------------


class ToyMarkov:
    """|V| 个 token 的一阶马尔可夫模型：next-token 分布只取决于上一个 token。

    为什么用马尔可夫而不是随便造两个固定分布？
      固定分布无法体现"接受与否会改变后续上下文"这件事，而正是这件事让
      Leviathan 论文里的 i.i.d. 假设在真实模型上不成立（见 accept.py）。
      马尔可夫既保留了上下文依赖，又小到可以精确枚举全部分支。
    """

    def __init__(self, V: int, seed: int, temp: float = 1.0):
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=(V, V))
        z = logits / temp
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.T = e / e.sum(axis=1, keepdims=True)  # (V, V), T[s] = p(.|s)
        self.V = V

    def dist(self, last: int) -> np.ndarray:
        return self.T[last]

    def chain_prob(self, s0: int, seq) -> float:
        """目标模型对序列 seq 的自回归概率 prod_i p(t_i | t_{i-1})。"""
        p = 1.0
        last = s0
        for t in seq:
            p *= self.T[last][t]
            last = t
        return float(p)


def perturb(model: ToyMarkov, eps: float, seed: int) -> ToyMarkov:
    """由 target 造一个"像但不完全像"的 draft：往 logit 上加噪。

    eps=0 -> draft == target（alpha = 1，全接受）
    eps 越大 -> 两个分布越远 -> alpha 越低
    """
    out = ToyMarkov.__new__(ToyMarkov)
    rng = np.random.default_rng(seed)
    logp = np.log(model.T + 1e-12) + eps * rng.normal(size=model.T.shape)
    logp = logp - logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    out.T = e / e.sum(axis=1, keepdims=True)
    out.V = model.V
    return out


# --------------------------------------------------------------------------
# A. 采样路径：一次投机迭代
# --------------------------------------------------------------------------


def speculative_step(target: ToyMarkov, draft: ToyMarkov, last: int, gamma: int,
                     rng: np.random.Generator, trace: list | None = None):
    """一次完整的投机迭代，返回本轮**确定产出**的 token 列表（长度 1..gamma+1）。

    严格按 Leviathan Algorithm 1：
      1) draft 自回归采 gamma 个 token（gamma 次串行前向）
      2) target 一次并行前向拿到 gamma+1 个分布
      3) 逐个用 min(1, p/q) 判接受；首次拒绝处从残差分布重采并**截断**
      4) 若 gamma 个全接受，额外从 p_{gamma+1} 采一个"赠品 token"
    """
    # ---- 1) 起草 ----
    xs, qs = [], []
    cur = last
    for _ in range(gamma):
        q = draft.dist(cur)
        x = int(rng.choice(len(q), p=q))
        xs.append(x)
        qs.append(q)
        cur = x

    # ---- 2) 目标模型并行验证：一次前向拿 gamma+1 个分布 ----
    # 关键：p_i 的条件是"前缀 + 前 i-1 个草稿 token"，与草稿的条件完全对齐
    ps, cur = [], last
    for i in range(gamma + 1):
        ps.append(target.dist(cur))
        if i < gamma:
            cur = xs[i]

    # ---- 3) 逐个判接受 ----
    out = []
    for i in range(gamma):
        x = xs[i]
        a = min(1.0, ps[i][x] / qs[i][x]) if qs[i][x] > 0 else 0.0
        u = float(rng.random())
        if trace is not None:
            trace.append({"i": i, "x": x, "p": float(ps[i][x]), "q": float(qs[i][x]),
                          "accept_prob": a, "u": u, "accepted": u < a})
        if u < a:
            out.append(x)
        else:
            r = residual_dist(ps[i], qs[i])
            y = int(rng.choice(len(r), p=r))
            out.append(y)
            if trace is not None:
                trace.append({"i": i, "resample_from_residual": y, "residual": r.copy()})
            return out  # 首次拒绝即截断，后面的草稿全部作废

    # ---- 4) 全接受，白拿一个 ----
    y = int(rng.choice(len(ps[gamma]), p=ps[gamma]))
    out.append(y)
    if trace is not None:
        trace.append({"bonus_token": y})
    return out


def speculative_generate(target, draft, s0, n_tokens, gamma, rng):
    """连续跑投机迭代直到产出 n_tokens 个 token。返回 (tokens, n_iters)。"""
    out, iters, last = [], 0, s0
    while len(out) < n_tokens:
        chunk = speculative_step(target, draft, last, gamma, rng)
        out.extend(chunk)
        last = out[-1]
        iters += 1
    return out[:n_tokens], iters


def target_generate(target, s0, n_tokens, rng):
    """朴素自回归（基线），用来做对照。"""
    out, last = [], s0
    for _ in range(n_tokens):
        p = target.dist(last)
        t = int(rng.choice(len(p), p=p))
        out.append(t)
        last = t
    return out


# --------------------------------------------------------------------------
# B. 精确枚举路径：不掷骰子，把概率加起来
# --------------------------------------------------------------------------


def iteration_dist(target: ToyMarkov, draft: ToyMarkov, last: int, gamma: int,
                   variant: str = "correct") -> dict:
    """一次投机迭代产出 token 元组的**精确**分布 {tuple: prob}。

    枚举全部分支：gamma 个草稿位置 × 每个位置 |V| 种取值 × 接受/拒绝二分。
    复杂度 O(|V|^gamma)，所以只在 |V|<=5, gamma<=4 的玩具规模用。

    variant 用来把**社区里常见的三种错误写法**也精确枚举出来，好证明它们真的有偏：
      "correct"    —— 标准算法：接受概率 min(1,p/q)，拒绝后从残差分布重采
      "naive_p"    —— 拒绝后直接从 p 重采（漏掉残差修正）。**有偏**
      "threshold"  —— 判据写成"p(x) >= q(x) 就接受"（把随机判据写成确定判据）。**有偏**
      "no_bonus"   —— 正确判据，但 gamma 个全接受时不发赠品 token。**无偏**，只是慢
    """
    res: dict = defaultdict(float)
    V = target.V

    def rec(i: int, prefix: tuple, prob: float, cur: int):
        if prob == 0.0:
            return
        if i == gamma:  # gamma 个草稿全被接受
            if variant == "no_bonus":
                if prefix:
                    res[prefix] += prob
                return
            pb = target.dist(cur)  # 赠品 token：白拿一个，从 p_{gamma+1} 采
            for y in range(V):
                if pb[y] > 0:
                    res[prefix + (y,)] += prob * pb[y]
            return
        p_i = target.dist(cur)
        q_i = draft.dist(cur)

        if variant == "threshold":
            # 错法二：确定性判据 p>=q 接受，否则拒绝
            for x in range(V):
                if q_i[x] <= 0:
                    continue
                if p_i[x] >= q_i[x]:
                    rec(i + 1, prefix + (x,), prob * q_i[x], x)
            p_rej = float(sum(q_i[x] for x in range(V) if p_i[x] < q_i[x]))
            if p_rej > 0:
                r = residual_dist(p_i, q_i)
                for y in range(V):
                    if r[y] > 0:
                        res[prefix + (y,)] += prob * p_rej * r[y]
            return

        # 接受分支：草稿抽到 x 且通过判据，概率 q(x)*min(1,p(x)/q(x)) = min(p(x),q(x))
        for x in range(V):
            if q_i[x] <= 0:
                continue
            a = min(1.0, p_i[x] / q_i[x])
            if a > 0:
                rec(i + 1, prefix + (x,), prob * q_i[x] * a, x)
        # 拒绝分支：残差分布与被拒的 x 无关，所以可以把拒绝概率合并起来一次算
        # 总拒绝概率 = sum_x q(x)*(1-min(1,p/q)) = 1 - sum_x min(p,q) = TV(p,q)
        p_rej = 1.0 - float(np.minimum(p_i, q_i).sum())
        if p_rej > 0:
            r = target.dist(cur) if variant == "naive_p" else residual_dist(p_i, q_i)
            for y in range(V):
                if r[y] > 0:
                    res[prefix + (y,)] += prob * p_rej * r[y]

    rec(0, (), 1.0, last)
    return dict(res)


def exact_stream_dist(target, draft, s0: int, n: int, gamma: int,
                      variant: str = "correct") -> dict:
    """投机采样输出流**前 n 个 token** 的精确分布 {tuple(len=n): prob}。

    把一次次迭代串起来（迭代产出长度可变，所以要递归到攒够 n 个）。
    这是本库对「L1 分布无损」的判定性验证：它应当逐点等于 target 的链式概率。
    """
    out: dict = defaultdict(float)

    def rec(emitted: tuple, prob: float):
        if prob < 1e-18:
            return
        if len(emitted) >= n:
            out[emitted[:n]] += prob
            return
        cur = emitted[-1] if emitted else s0
        for chunk, pc in iteration_dist(target, draft, cur, gamma, variant).items():
            rec(emitted + chunk, prob * pc)

    rec((), 1.0)
    return dict(out)


def exact_target_dist(target: ToyMarkov, s0: int, n: int) -> dict:
    """目标模型自身对长度 n 序列的精确分布（对照组）。"""
    out = {}

    def rec(seq: tuple, prob: float, cur: int):
        if len(seq) == n:
            out[seq] = prob
            return
        p = target.dist(cur)
        for t in range(target.V):
            if p[t] > 0:
                rec(seq + (t,), prob * p[t], t)

    rec((), 1.0, s0)
    return out


def max_pointwise_error(target, draft, s0, n, gamma, variant: str = "correct") -> float:
    """两个精确分布的最大逐点误差。L1 无损 <=> 这个数是机器精度量级。"""
    a = exact_stream_dist(target, draft, s0, n, gamma, variant)
    b = exact_target_dist(target, s0, n)
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


# --------------------------------------------------------------------------
# demo / proof
# --------------------------------------------------------------------------


def _demo():
    V, gamma = 5, 3
    tgt = ToyMarkov(V, seed=0, temp=1.0)
    drf = perturb(tgt, eps=0.8, seed=1)
    last = 0
    print("=" * 74)
    print("手算算例：|V|=%d, gamma=%d, 起始 token s0=%d" % (V, gamma, last))
    print("=" * 74)
    np.set_printoptions(precision=4, suppress=True)
    print("\ntarget p(.|s=0) =", tgt.dist(0))
    print("draft  q(.|s=0) =", drf.dist(0))
    b = beta_overlap(tgt.dist(0), drf.dist(0))
    print("单步接受概率 beta = sum min(p,q) = %.4f" % b)
    print("全变差 TV(p,q)            = %.4f" % total_variation(tgt.dist(0), drf.dist(0)))
    print("恒等式 beta + TV          = %.4f  (应为 1.0000)" % (b + total_variation(tgt.dist(0), drf.dist(0))))
    print("残差分布 p' = norm(max(0,p-q)) =", residual_dist(tgt.dist(0), drf.dist(0)))

    rng = np.random.default_rng(7)
    print("\n--- 跑 3 次投机迭代，逐步打印判据 ---")
    cur = last
    for it in range(3):
        tr = []
        chunk = speculative_step(tgt, drf, cur, gamma, rng, trace=tr)
        print("\n[迭代 %d] 起始 token=%d" % (it + 1, cur))
        for ev in tr:
            if "accept_prob" in ev:
                print("   位置%d 草稿token=%d  p=%.4f q=%.4f  min(1,p/q)=%.4f  u=%.4f  -> %s"
                      % (ev["i"], ev["x"], ev["p"], ev["q"], ev["accept_prob"], ev["u"],
                         "接受" if ev["accepted"] else "拒绝"))
            elif "resample_from_residual" in ev:
                print("   拒绝处从残差分布重采 -> token=%d" % ev["resample_from_residual"])
            elif "bonus_token" in ev:
                print("   gamma 个全接受，白拿赠品 token=%d" % ev["bonus_token"])
        print("   本轮产出 %d 个 token: %s" % (len(chunk), chunk))
        cur = chunk[-1]

    print("\n--- 一次迭代的精确产出分布（不掷骰子，枚举全部分支）---")
    d = iteration_dist(tgt, drf, 0, gamma=1)
    tot = sum(d.values())
    print("gamma=1 时共 %d 种产出，概率和 = %.12f" % (len(d), tot))
    first = defaultdict(float)
    for k, v in d.items():
        first[k[0]] += v
    print("产出的**第一个 token** 的边缘分布 =", np.array([first[i] for i in range(V)]))
    print("target 的 p(.|s=0)              =", tgt.dist(0))
    err = max(abs(first[i] - tgt.dist(0)[i]) for i in range(V))
    print("最大逐点误差 = %.3e   <- 这就是「L1 分布无损」" % err)


def first_token_dist(target, draft, last: int, gamma: int,
                     variant: str = "correct") -> np.ndarray:
    """一次迭代产出的**第一个 token** 的精确边缘分布。正确算法下它应当恒等于 p(.|last)。"""
    out = np.zeros(target.V)
    for chunk, pr in iteration_dist(target, draft, last, gamma, variant).items():
        out[chunk[0]] += pr
    return out


def _bias():
    """把三种常见错法的偏差算出来（不是估计，是精确枚举）。"""
    V, gamma = 5, 3
    tgt = ToyMarkov(V, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    print("=" * 88)
    print("三种社区常见写法的精确偏差  (|V|=%d, gamma=%d, 起始 token=0)" % (V, gamma))
    print("=" * 88)
    print("%-14s %-40s %-16s" % ("写法", "说明", "首 token 最大逐点误差"))
    rows = [
        ("correct", "min(1,p/q) 判据 + 残差分布重采", "correct"),
        ("naive_p", "拒绝后直接从 p 重采（漏残差修正）", "naive_p"),
        ("threshold", "判据写成 p>=q 就接受（确定性）", "threshold"),
        ("no_bonus", "正确判据但不发赠品 token", "no_bonus"),
    ]
    p_true = tgt.dist(0)
    for name, desc, var in rows:
        d = first_token_dist(tgt, drf, 0, gamma, var)
        err = float(np.abs(d - p_true).max())
        verdict = "无偏" if err < 1e-12 else "**有偏**"
        print("%-14s %-40s %.3e  %s" % (name, desc, err, verdict))
    print()
    print("注意 no_bonus：不发赠品 token 是**无偏**的，只是每轮少产出一个 token（变慢，不变错）。")
    print("      而 naive_p 与 threshold 都是**有偏**的 —— 它们悄悄改掉了模型的输出分布，")
    print("      而且不会报错、不会崩，只会让线上模型的行为与离线评测对不上。")


def _proof():
    print("=" * 74)
    print("L1 分布无损的判定性验证：精确枚举 vs 目标模型链式概率")
    print("=" * 74)
    print("%-6s %-7s %-6s %-8s %-12s %s" % ("V", "gamma", "n", "eps", "alpha(s0)", "最大逐点误差"))
    for V in (3, 4):
        for gamma in (1, 2, 3):
            for eps in (0.0, 0.5, 1.5, 4.0):
                tgt = ToyMarkov(V, seed=V, temp=1.0)
                drf = perturb(tgt, eps=eps, seed=V + 100)
                n = 3
                err = max_pointwise_error(tgt, drf, 0, n, gamma)
                a = beta_overlap(tgt.dist(0), drf.dist(0))
                print("%-6d %-7d %-6d %-8.1f %-12.4f %.3e" % (V, gamma, n, eps, a, err))
    print("\n读法：eps 越大草稿越烂、接受率越低、迭代次数越多，但**误差始终是机器精度**。")
    print("      草稿模型的好坏只影响速度，不影响分布 —— 这就是投机采样的全部卖点。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--bias", action="store_true")
    a = ap.parse_args()
    if a.demo:
        _demo()
    if a.proof:
        _proof()
    if a.bias:
        _bias()
    if not (a.demo or a.proof or a.bias):
        _demo()
