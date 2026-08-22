"""
accept.py —— 接受率 alpha、期望接受长度 E[tau]、最优 gamma 的数学与**证伪**。

三件事：
  1) 恒等式  beta = sum_x min(p,q) = 1 - TV(p,q)                       [可证]
  2) 公式    E[tau] = (1 - alpha^(gamma+1)) / (1 - alpha)              [有前提]
  3) 前提    上式要求"每个位置的接受事件 i.i.d."，**真实模型上不成立**   [本文件负责证伪]

第 3 条是本库和多数二手讲解的分界线：社区几乎都把 (1-a^(g+1))/(1-a) 当成
"接受率 a 就能算出加速比"的万能公式，但 Leviathan 论文 Theorem 3.8 的前提是
acceptance 独立同分布。真实模型里 beta 随上下文剧烈变化，Jensen 不等式立刻生效。

用法：
    python accept.py --table     # alpha x gamma 的 E[tau] 与最优 gamma 扫描表
    python accept.py --iid       # i.i.d. 假设的证伪实验（实测 vs 公式）
    python accept.py --temp      # 温度如何改变 alpha（非单调）
"""
from __future__ import annotations

import argparse

import numpy as np

from spec import (ToyMarkov, beta_overlap, perturb, speculative_step,
                  total_variation)

# --------------------------------------------------------------------------
# 1) 闭式公式
# --------------------------------------------------------------------------


def expected_tokens(alpha: float, gamma: int) -> float:
    """一次投机迭代的期望产出 token 数（含"赠品 token"）。

    Leviathan et al. 2022, Theorem 3.8:
        E[tau] = (1 - alpha^(gamma+1)) / (1 - alpha)
    alpha=1 的极限是 gamma+1（全接受 + 赠品）。

    前提（论文明写）：每个位置的接受事件独立同分布，接受概率均为 alpha。
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def speedup_ideal(alpha: float, gamma: int, c: float) -> float:
    """理想加速比（Leviathan 的口径）。

        speedup = E[tau] / (gamma * c + 1)

    c = 单次 draft 前向耗时 / 单次 target 前向耗时。
    分母的 "+1" 是那一次 target 验证前向，**它被假设与验证 1 个 token 同价**
    —— 这个假设只在 memory-bound 区成立，大 batch 下会崩（见 speedup.py）。
    """
    return expected_tokens(alpha, gamma) / (gamma * c + 1.0)


def optimal_gamma(alpha: float, c: float, gmax: int = 64) -> tuple[int, float]:
    """在 [1, gmax] 上暴力找最优 gamma，返回 (gamma*, speedup*)。"""
    best_g, best_s = 1, -1.0
    for g in range(1, gmax + 1):
        s = speedup_ideal(alpha, g, c)
        if s > best_s:
            best_g, best_s = g, s
    return best_g, best_s


def marginal_gamma(alpha: float, c: float, gmax: int = 512) -> int:
    """用**边际判据**求最优 gamma：一直加到 alpha^(g+1) <= c * speedup(g) 为止。

    判据的来历（写第 17 篇时纠正过一次）：
      直觉会写成 alpha^(g+1) > c —— "多拿一个 token 的概率 > 多花一次草稿的成本"。
      **这是错的**：speedup = N/D 是个**比值**，不是差值。比值最大化的条件是
      "边际比 > 当前平均比"，即 dN/dD > N/D，代入 dN = alpha^(g+1)、dD = c 得

          alpha^(g+1) > c * speedup(g)

      物理含义：多花 c 那份时间是有**机会成本**的 —— 本可以结束本轮、
      按当前速率继续产出。忽略机会成本会系统性地把 gamma 选得过大
      (alpha=0.9, c=0.05 时错判据给 28，实际最优 13，理想加速比损失约 15%)。

    与 optimal_gamma 的暴力搜索在 alpha x c 网格上逐点一致（并列除外），
    见 test_accept.py::test_marginal_criterion_matches_brute_force。
    """
    g = 1
    while g < gmax:
        if alpha ** (g + 1) <= c * speedup_ideal(alpha, g, c):
            return g
        g += 1
    return gmax


def naive_marginal_gamma(alpha: float, c: float, gmax: int = 512) -> int:
    """**错误**的边际判据 alpha^(g+1) > c —— 保留它是为了让测试证明它错。"""
    g = 1
    while g < gmax:
        if alpha ** (g + 1) <= c:
            return g
        g += 1
    return gmax


# --------------------------------------------------------------------------
# 2) 从模型算 alpha
# --------------------------------------------------------------------------


def stationary(T: np.ndarray, iters: int = 5000) -> np.ndarray:
    """马尔可夫链的平稳分布（幂法）。"""
    pi = np.ones(T.shape[0]) / T.shape[0]
    for _ in range(iters):
        pi = pi @ T
    return pi


def alpha_expected(target: ToyMarkov, draft: ToyMarkov) -> float:
    """alpha = E_{s ~ pi}[ beta(p(.|s), q(.|s)) ] —— 论文口径的"期望接受率"。"""
    pi = stationary(target.T)
    return float(sum(pi[s] * beta_overlap(target.dist(s), draft.dist(s))
                     for s in range(target.V)))


def beta_spread(target: ToyMarkov, draft: ToyMarkov):
    """各状态下的 beta，用来看它到底有多"不 i.i.d."。"""
    return np.array([beta_overlap(target.dist(s), draft.dist(s)) for s in range(target.V)])


# --------------------------------------------------------------------------
# 3) 证伪：实测 E[tau] vs 公式
# --------------------------------------------------------------------------


def measure_expected_tokens(target, draft, gamma, n_iters=200_000, seed=0):
    """实跑投机迭代，测每轮真实产出的 token 数的平均值。"""
    rng = np.random.default_rng(seed)
    last, tot = 0, 0
    for _ in range(n_iters):
        chunk = speculative_step(target, draft, last, gamma, rng)
        tot += len(chunk)
        last = chunk[-1]
    return tot / n_iters


def _table():
    print("=" * 78)
    print("表 1  E[tau] = (1-a^(g+1))/(1-a)   —— 一次迭代的期望产出 token 数")
    print("=" * 78)
    gammas = [1, 2, 3, 4, 5, 7, 10]
    print("%-8s" % "alpha", "".join("%8s" % ("g=%d" % g) for g in gammas))
    for a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print("%-8.2f" % a, "".join("%8.3f" % expected_tokens(a, g) for g in gammas))

    print()
    print("=" * 78)
    print("表 2  最优 gamma* 与理想加速比   speedup = E[tau]/(g*c+1)")
    print("=" * 78)
    print("%-8s %-10s %-10s %-10s" % ("alpha", "c", "gamma*", "speedup*"))
    for c in (0.02, 0.05, 0.1, 0.2):
        for a in (0.5, 0.7, 0.8, 0.9):
            g, s = optimal_gamma(a, c)
            print("%-8.2f %-10.2f %-10d %-10.3f" % (a, c, g, s))
    print()
    print("读法：c 是「草稿有多便宜」。同一个 alpha=0.8，c 从 0.02 涨到 0.20，最优 gamma* 从 11 掉到 4，")
    print("      理想加速比从 3.82x 掉到 1.87x —— 草稿是**串行**跑 gamma 次，成本线性涨，收益却指数饱和。")


def _iid():
    print("=" * 78)
    print("i.i.d. 假设证伪：公式用的 alpha 是「平均接受率」，但 beta 随上下文剧烈变化")
    print("=" * 78)
    print("%-4s %-6s %-8s %-9s %-9s %-9s %-10s %-9s" %
          ("V", "eps", "gamma", "alpha", "beta_min", "beta_max", "公式E[tau]", "实测E[tau]"))
    rows = []
    for V in (4, 8):
        for eps in (0.5, 1.5, 3.0):
            tgt = ToyMarkov(V, seed=V * 3, temp=1.0)
            drf = perturb(tgt, eps=eps, seed=V * 3 + 77)
            a = alpha_expected(tgt, drf)
            bs = beta_spread(tgt, drf)
            for gamma in (2, 4):
                pred = expected_tokens(a, gamma)
                meas = measure_expected_tokens(tgt, drf, gamma, n_iters=40_000, seed=1)
                rows.append((V, eps, gamma, a, bs.min(), bs.max(), pred, meas))
                print("%-4d %-6.1f %-8d %-9.4f %-9.4f %-9.4f %-10.4f %-9.4f  (实测/公式=%.3f)"
                      % (V, eps, gamma, a, bs.min(), bs.max(), pred, meas, meas / pred))
    print()
    print("结论：beta 在不同上下文下可以差出好几倍（beta_min vs beta_max），")
    print("      E[tau] 关于 alpha 是**凸函数**，由 Jensen 不等式 E[f(beta)] >= f(E[beta])，")
    print("      所以用平均 alpha 代入公式会**系统性低估**真实吞吐。")
    print("      工程含义：别用公式反推 alpha，也别用公式预测加速比 —— 直接测 acceptance length。")


def _retemp(m, T):
    """把一个模型按温度 T 重新归一化（对 log 概率除以 T 再 softmax）。"""
    z = np.log(m.T + 1e-12) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    o = ToyMarkov.__new__(ToyMarkov)
    o.T, o.V = e / e.sum(axis=1, keepdims=True), m.V
    return o


def _temp():
    print("=" * 88)
    print("温度如何改变 alpha —— 形状由「argmax 一致率」决定，不是一条固定曲线")
    print("=" * 88)
    V = 8
    base = ToyMarkov(V, seed=5, temp=1.0)
    drafts = {e: perturb(base, eps=e, seed=99) for e in (0.3, 1.2)}
    for e, d in drafts.items():
        agree = float(np.mean(np.argmax(base.T, axis=1) == np.argmax(d.T, axis=1)))
        print("  好/烂草稿：eps=%.1f 时 argmax 一致率 = %.3f" % (e, agree))
    print()
    print("%-9s %-14s %-14s" % ("温度T", "好草稿(eps=0.3)", "烂草稿(eps=1.2)"))
    prev = None
    for T in (0.02, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 20.0):
        a_good = alpha_expected(_retemp(base, T), _retemp(drafts[0.3], T))
        a_bad = alpha_expected(_retemp(base, T), _retemp(drafts[1.2], T))
        mark = "  <- 低谷" if prev is not None and a_good < prev else ""
        prev = a_good
        print("%-9.2f %-14.4f %-14.4f%s" % (T, a_good, a_bad, mark))
    print()
    print("读法（这段是实测推翻了我最初写法之后重写的）：")
    print("  1) T -> 0 时 alpha 收敛到**两个模型 argmax 相同的概率质量**，不是收敛到 0 或 1。")
    print("     好草稿 argmax 一致率 0.625 -> 贪心下 alpha 就有 0.70；烂草稿一致率 0.125 -> alpha 近 0。")
    print("  2) T -> inf 时两个分布都趋于均匀，alpha -> 1（两条曲线都往上收）。")
    print("  3) 于是**好草稿是 U 形**（贪心处高、中温有低谷、高温回升），")
    print("     **烂草稿是单调上升**。曲线形状不是常数，它取决于草稿与目标的 argmax 一致率。")
    print("  4) 工程含义（铁律二的硬证据）：拿 T=0 测出来的接受率**不能**外推到 T=0.7 的线上配置。")
    print("     好草稿会掉（0.70 -> 0.66 附近），烂草稿会涨。不报温度的接受率数字没有意义。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--temp", action="store_true")
    a = ap.parse_args()
    if a.table:
        _table()
    if a.iid:
        _iid()
    if a.temp:
        _temp()
    if not (a.table or a.iid or a.temp):
        _table()
