"""
regime.py —— 公式 E[tau]=(1-a^(g+1))/(1-a) 到底什么时候失效？（两次自我证伪的记录）

【猜想一】beta 随上下文波动 -> Jensen 不等式 -> 公式系统性低估。
【证伪一】accept.py --iid 实测：beta 在 0.176~0.449 之间变化（差 2.5 倍），
          公式却准到 2% 以内。猜想错了。

想清楚才发现推导本身就给了原因：
    E[tau] = sum_{k=0}^{gamma} P(前 k 个全接受) = sum_k E[ prod_{i<k} beta_i ]
  只要 beta_i **沿链互不相关**，E[prod beta_i] = prod E[beta_i] = alpha^k，
  公式**精确成立** —— 它根本不要求 beta 是常数，只要求不相关。
  (Leviathan 论文写的是 i.i.d.，那是充分条件，不是必要条件。)

【猜想二】那么正相关时 E[prod beta] > prod E[beta]，公式应当**低估**。
【证伪二】本文件 --sweep 实测：粘性 sticky 从 0.50 升到 0.99（beta 一阶自相关
          rho 从 +0.04 升到 +0.81），实测/公式 从 1.036 一路掉到 0.895。
          方向**相反** —— 公式是**高估**。

真正的机制（--bias 观察到，--renewal 查到底）：**更新过程的长度偏置**。
  公式里的 alpha 是平稳分布 pi 加权的平均接受率，这等价于默认"每轮迭代的起点是 pi 分布的"。
  但起点是上一轮结束的地方，而**难区的迭代更短**（更容易被拒绝截断）：
  实测 sticky=0.99 时 E[tau|难起点]=2.00，E[tau|易起点]=4.74。
  短迭代多 -> "按迭代抽样"把难区抽多了（起点落难区 0.38 -> 0.69），
  于是每轮的实际接受率低于 pi 加权的 alpha，公式**高估**。

  两条旁证：
  (1) 同一次实验里**按 token 抽样**的难区比例恒为 0.50 = pi —— 无损性在这里被独立验证；
      偏的只是"按迭代抽样"，不是 token 流本身。
  (2) 更新过程闭式预测在 sticky=0.99 时给 0.7033、实测 0.6938（差 1.4%），
      在 sticky=0.50 时给 0.6498、实测 0.3792（不准）—— 因为该闭式假设"整轮待在起点那个区"，
      这条只在高粘性下成立。模型在自己的假设成立处才准，这也是对本文件方法的一次自查。

结论（可复跑）：用全局平均接受率套公式，在"难区连片"的负载上会**高估** E[tau]，
本实验最大高估 10.5%（sticky=0.99, gamma=8：公式 3.633 vs 实测 3.253）。

用法：
    python regime.py --sweep    # 扫粘性：公式 vs 实测 + beta 自相关
    python regime.py --bias     # 诊断：迭代起点被偏置到难区的倍数
    python regime.py --renewal  # 机制：更新过程的长度偏置（难区迭代更短）
"""
from __future__ import annotations

import argparse

import numpy as np

from accept import expected_tokens, measure_expected_tokens, stationary
from spec import ToyMarkov, beta_overlap


def make_regime_pair(V: int = 8, sticky: float = 0.5, eps_hard: float = 3.0,
                     seed: int = 0):
    """构造一对 (target, draft)，token 前一半属"易区"、后一半属"难区"。

    - 易区状态：draft 与 target 几乎一致  -> beta 高
    - 难区状态：draft 被强噪声打乱        -> beta 低
    - sticky：从易区转到易区的概率（=从难区转到难区的概率）。
      sticky=0.5 表示下一个 token 落在哪个区与当前区无关（不相关）；
      sticky->1 表示区制粘住不动（强正相关）。
    """
    rng = np.random.default_rng(seed)
    half = V // 2
    easy = np.arange(half)          # 易区 token
    hard = np.arange(half, V)       # 难区 token

    # ---- target：块状转移矩阵，块内均匀随机，块间按 sticky 分配总质量 ----
    T = np.zeros((V, V))
    for s in range(V):
        same = easy if s < half else hard
        other = hard if s < half else easy
        w_same = rng.random(len(same)) + 0.2
        w_other = rng.random(len(other)) + 0.2
        T[s, same] = sticky * w_same / w_same.sum()
        T[s, other] = (1.0 - sticky) * w_other / w_other.sum()
    tgt = ToyMarkov.__new__(ToyMarkov)
    tgt.T, tgt.V = T, V

    # ---- draft：易区状态几乎不动，难区状态加大噪声 ----
    Q = np.zeros_like(T)
    for s in range(V):
        eps = 0.05 if s < half else eps_hard
        logq = np.log(T[s] + 1e-12) + eps * rng.normal(size=V)
        logq -= logq.max()
        e = np.exp(logq)
        Q[s] = e / e.sum()
    drf = ToyMarkov.__new__(ToyMarkov)
    drf.T, drf.V = Q, V
    return tgt, drf


def beta_autocorr(target, draft, n=200_000, seed=3):
    """沿目标链采样，测 beta_t 与 beta_{t+1} 的相关系数（即"难度粘不粘"）。"""
    rng = np.random.default_rng(seed)
    betas = np.array([beta_overlap(target.dist(s), draft.dist(s))
                      for s in range(target.V)])
    s, seq = 0, np.empty(n)
    for t in range(n):
        seq[t] = betas[s]
        s = int(rng.choice(target.V, p=target.dist(s)))
    a, b = seq[:-1], seq[1:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def iteration_start_bias(target, draft, gamma, n_iters=60_000, seed=5):
    """诊断：投机迭代的**起点 token** 落在难区的比例 vs 平稳分布下的比例。

    为什么要看这个：公式里的 alpha 用的是平稳分布 pi 下的平均 beta，
    它默认"每次迭代的起点是 pi 分布的"。但迭代的起点是**上一轮的最后一个 token**，
    而上一轮多半是被"拒绝"截断的 —— 拒绝更容易发生在难区。
    于是起点被系统性偏置到难区（一种更新过程里的长度偏置 length-biasing）。
    """
    from spec import speculative_step
    rng = np.random.default_rng(seed)
    V = target.V
    half = V // 2
    last, hard_starts = 0, 0
    for _ in range(n_iters):
        if last >= half:
            hard_starts += 1
        chunk = speculative_step(target, draft, last, gamma, rng)
        last = chunk[-1]
    pi = stationary(target.T)
    return hard_starts / n_iters, float(pi[half:].sum())


def renewal_check(target, draft, gamma, n_iters=60_000, seed=5):
    """把"起点偏置"的机制查到底：它是**更新过程的长度偏置**。

    思路：难区的投机迭代更短（更容易被拒绝截断）。于是"按迭代计数"时难区被过采样，
    而"按 token 计数"时不会 —— 后者由无损性保证等于平稳分布 pi。

    更新过程（renewal-reward）预测：
        P(迭代起点落难区) = (pi_hard / E[tau|难起点]) / (pi_easy/E[tau|易] + pi_hard/E[tau|难])
    这个公式**假设整轮迭代都待在起点所在的区**，所以它只在粘性接近 1 时才该准。
    返回的字典把预测与实测都给出来，好让读者看见"模型在自己假设成立的地方才准"。
    """
    from spec import speculative_step
    V = target.V
    half = V // 2
    rng = np.random.default_rng(seed)
    last = 0
    stat = {0: [0, 0], 1: [0, 0]}      # 区 -> [迭代数, token 数]
    hard_tok = tot_tok = 0
    for _ in range(n_iters):
        r = 1 if last >= half else 0
        chunk = speculative_step(target, draft, last, gamma, rng)
        stat[r][0] += 1
        stat[r][1] += len(chunk)
        for t in chunk:
            tot_tok += 1
            hard_tok += (t >= half)
        last = chunk[-1]
    e_easy = stat[0][1] / max(stat[0][0], 1)
    e_hard = stat[1][1] / max(stat[1][0], 1)
    pi = stationary(target.T)
    a, b = pi[:half].sum() / e_easy, pi[half:].sum() / e_hard
    return dict(e_tau_easy_start=e_easy, e_tau_hard_start=e_hard,
                iter_start_hard_measured=stat[1][0] / n_iters,
                iter_start_hard_renewal_pred=float(b / (a + b)),
                token_stream_hard=hard_tok / tot_tok,
                pi_hard=float(pi[half:].sum()))


def _renewal():
    print("=" * 104)
    print("起点偏置的机制：更新过程的长度偏置（难区的迭代更短 -> 按迭代计数时难区被过采样）")
    print("=" * 104)
    print("%-8s %-13s %-13s %-15s %-15s %-15s" %
          ("sticky", "E[tau|易起点]", "E[tau|难起点]", "token流落难区", "起点落难区实测", "更新过程预测"))
    for sticky in (0.50, 0.85, 0.99):
        tgt, drf = make_regime_pair(V=8, sticky=sticky, eps_hard=3.0, seed=11)
        r = renewal_check(tgt, drf, 4, n_iters=40_000)
        print("%-8.2f %-13.3f %-13.3f %-15.4f %-15.4f %-15.4f"
              % (sticky, r["e_tau_easy_start"], r["e_tau_hard_start"],
                 r["token_stream_hard"], r["iter_start_hard_measured"],
                 r["iter_start_hard_renewal_pred"]))
    print()
    print("三件事一起看：")
    print("  1) **token 流落难区恒为 0.50**（= 平稳分布 pi）—— 无损性在这里被独立验证了一次：")
    print("     投机采样吐出来的 token 流，其统计与目标模型自己跑出来的完全一致。")
    print("  2) 但**迭代起点**落难区的比例随粘性从 0.38 升到 0.69，远离 0.50。")
    print("     原因在第 2、3 列：难区起点的迭代只有 2.0 个 token，易区起点有 4.7 个。")
    print("     短的迭代多，于是「按迭代抽样」把难区抽多了 —— 经典的更新过程长度偏置。")
    print("  3) 更新过程公式在 sticky=0.99 时预测 0.7033、实测 0.6938（差 1.4%）；")
    print("     在 sticky=0.50 时预测 0.6498、实测 0.3792（完全不准）—— 因为该公式假设")
    print("     「整轮都待在起点那个区」，这条只在粘性接近 1 时成立。**模型在自己的假设成立处才准**。")
    print()
    print("对公式 E[tau]=(1-a^(g+1))/(1-a) 的含义：它用的 alpha 是 pi 加权的平均接受率，")
    print("等于默认「每轮起点是 pi 分布的」。上面第 2 行说明这条前提是错的，且偏向难区，")
    print("所以公式**高估**真实 E[tau]（--sweep 里最大高估 10.5%）。")


def _bias():
    print("=" * 88)
    print("诊断：为什么正相关下公式会**高估** —— 迭代起点被偏置到难区")
    print("=" * 88)
    print("%-9s %-8s %-16s %-16s %-10s" %
          ("sticky", "gamma", "起点落难区比例", "平稳分布难区比例", "偏置倍数"))
    for sticky in (0.50, 0.70, 0.85, 0.95, 0.99):
        tgt, drf = make_regime_pair(V=8, sticky=sticky, eps_hard=3.0, seed=11)
        for gamma in (4,):
            obs, base = iteration_start_bias(tgt, drf, gamma, n_iters=40_000)
            print("%-9.2f %-8d %-16.4f %-16.4f %-10.3f"
                  % (sticky, gamma, obs, base, obs / base))
    print()
    print("读法：粘性越强，迭代起点越集中在难区（偏置倍数 > 1）。")
    print("      因果链：拒绝多发生在难区 -> 下一轮就从难区起步 -> 粘性让它整轮都待在难区")
    print("           -> 实测 E[tau] 低于用全局平均 alpha 算出来的公式值。")
    print("      这与「Jensen 会让公式低估」的直觉**相反**，是本库自己跑出来推翻的第二条猜想。")


def _sweep():
    print("=" * 96)
    print("公式什么时候失效：不是「beta 有波动」，而是「beta 沿链正相关」")
    print("=" * 96)
    print("%-9s %-8s %-9s %-9s %-9s %-11s %-11s %-8s" %
          ("sticky", "gamma", "alpha", "beta_易", "beta_难", "公式E[tau]", "实测E[tau]", "实测/公式"))
    for sticky in (0.50, 0.70, 0.85, 0.95, 0.99):
        tgt, drf = make_regime_pair(V=8, sticky=sticky, eps_hard=3.0, seed=11)
        pi = stationary(tgt.T)
        betas = np.array([beta_overlap(tgt.dist(s), drf.dist(s)) for s in range(tgt.V)])
        alpha = float(pi @ betas)
        rho = beta_autocorr(tgt, drf, n=120_000)
        for gamma in (4, 8):
            pred = expected_tokens(alpha, gamma)
            meas = measure_expected_tokens(tgt, drf, gamma, n_iters=60_000, seed=5)
            print("%-9.2f %-8d %-9.4f %-9.4f %-9.4f %-11.4f %-11.4f %-8.3f"
                  % (sticky, gamma, alpha, betas[:4].mean(), betas[4:].mean(),
                     pred, meas, meas / pred))
        print("          (beta 的一阶自相关 rho = %+.3f)" % rho)
    print()
    print("读法：五行用的是同一对模型，只改转移的粘性；beta 的波动幅度几乎不变，")
    print("      变的只有「难度会不会连着出现」（rho 从 +0.04 到 +0.81）。")
    print("      结果：rho 越大，实测越**低于**公式（1.036 -> 0.895）。")
    print("      注意这与「正相关让 E[prod beta] 变大所以公式低估」的直觉相反 —— 见 --bias：")
    print("      迭代起点被偏置到难区，这一项压过了链内正相关项。")
    print("      => 公式的前提不是「接受率处处相同」，而是「沿链不相关 + 起点无偏」，")
    print("         两条在真实负载上都不成立。工程上别用公式预测收益，直接测 acceptance length。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--renewal", action="store_true")
    a = ap.parse_args()
    if a.bias:
        _bias()
    if a.renewal:
        _renewal()
    if a.sweep or not (a.bias or a.renewal):
        _sweep()
