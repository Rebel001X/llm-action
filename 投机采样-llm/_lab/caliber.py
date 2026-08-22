"""
caliber.py —— 把「三种无损口径」做成可判定的实验（铁律一的实验基础）。

  L1 分布无损：输出是目标分布 p 的精确样本（spec.py 已用精确枚举证明）
  L2 贪心等价：T=0 时输出序列与目标模型贪心解码**逐 token 相同**
  L3 近似     ：放宽接受判据（如 Medusa 的 typical acceptance），分布已经不等于 p

本文件负责 L2 与 L3：
  --greedy   验证 T=0 下投机输出与贪心逐 token 相同（且与草稿好坏无关）
  --typical  量化"绝对阈值接受"这类近似判据的偏差随阈值怎么涨
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from spec import ToyMarkov, perturb, residual_dist


def onehot_model(m: ToyMarkov) -> ToyMarkov:
    """把模型退化成 T->0 的极限：每行只在 argmax 处取 1。"""
    o = ToyMarkov.__new__(ToyMarkov)
    T = np.zeros_like(m.T)
    T[np.arange(m.V), m.T.argmax(axis=1)] = 1.0
    o.T, o.V = T, m.V
    return o


def greedy_generate(target: ToyMarkov, s0: int, n: int) -> list[int]:
    """朴素贪心解码（基线）。"""
    out, cur = [], s0
    for _ in range(n):
        cur = int(target.T[cur].argmax())
        out.append(cur)
    return out


def greedy_speculative(target: ToyMarkov, draft: ToyMarkov, s0: int, n: int,
                       gamma: int) -> list[int]:
    """T=0 下的投机采样：两侧都取 argmax，判据仍走 min(1,p/q) + 残差重采。

    T=0 时 p、q 都是 one-hot：
      - 两者 argmax 相同 -> p(x)=q(x)=1 -> min(1,p/q)=1 -> 必接受
      - 不同             -> p(x)=0      -> 接受概率 0   -> 必拒绝，
                            残差 norm(max(0,p-q)) 是 argmax(p) 处的 one-hot -> 吐出 argmax(p)
    两条路都吐出 argmax(p)，所以**输出与贪心逐 token 相同**，且判据里不再有随机性。
    """
    P, Q = onehot_model(target), onehot_model(draft)
    out, cur = [], s0
    while len(out) < n:
        xs, c = [], cur
        for _ in range(gamma):                      # 起草
            c = int(Q.T[c].argmax())
            xs.append(c)
        ps, c = [], cur
        for i in range(gamma + 1):                  # 并行验证
            ps.append(P.T[c])
            if i < gamma:
                c = xs[i]
        chunk = []
        for i in range(gamma):
            x = xs[i]
            qv = Q.T[xs[i - 1] if i else cur][x]
            a = min(1.0, ps[i][x] / qv) if qv > 0 else 0.0
            if a >= 1.0:
                chunk.append(x)
            else:
                chunk.append(int(residual_dist(ps[i], Q.T[xs[i - 1] if i else cur]).argmax()))
                break
        else:
            chunk.append(int(ps[gamma].argmax()))   # 赠品 token
        out.extend(chunk)
        cur = out[-1]
    return out[:n]


def typical_accept_first_token(target: ToyMarkov, draft: ToyMarkov, last: int,
                               gamma: int, eps: float) -> np.ndarray:
    """近似判据（L3）下，一次迭代产出的第一个 token 的**精确**边缘分布。

    判据：草稿 token x 只要满足 p(x) >= eps 就接受（绝对阈值，不看 q）。
    这是 Medusa 的 typical acceptance 的一个最简化身：用"够典型就放行"换接受长度。
    eps=0 时全部接受（分布完全变成 q）；eps 很大时退化成几乎全拒绝。
    """
    res = defaultdict(float)
    V = target.V

    def rec(i: int, prefix: tuple, prob: float, cur: int):
        if prob == 0.0:
            return
        if i == gamma:
            pb = target.dist(cur)
            for y in range(V):
                res[prefix + (y,)] += prob * pb[y]
            return
        p_i, q_i = target.dist(cur), draft.dist(cur)
        rej = 0.0
        for x in range(V):
            if q_i[x] <= 0:
                continue
            if p_i[x] >= eps:
                rec(i + 1, prefix + (x,), prob * q_i[x], x)
            else:
                rej += q_i[x]
        if rej > 0:
            r = residual_dist(p_i, q_i)
            for y in range(V):
                res[prefix + (y,)] += prob * rej * r[y]

    rec(0, (), 1.0, last)
    out = np.zeros(V)
    for k, v in res.items():
        out[k[0]] += v
    return out


def _greedy():
    print("=" * 86)
    print("L2 贪心等价：T=0 时投机输出与贪心解码逐 token 相同（与草稿好坏无关）")
    print("=" * 86)
    print("%-6s %-8s %-8s %-14s %-12s" % ("V", "eps", "gamma", "argmax一致率", "与贪心是否逐token相同"))
    for V in (5, 8):
        tgt = ToyMarkov(V, seed=V, temp=1.0)
        for eps in (0.0, 1.0, 3.0, 8.0):
            drf = perturb(tgt, eps=eps, seed=V + 7)
            agree = float(np.mean(tgt.T.argmax(1) == drf.T.argmax(1)))
            for gamma in (2, 4):
                a = greedy_generate(tgt, 0, 60)
                b = greedy_speculative(tgt, drf, 0, 60, gamma)
                print("%-6d %-8.1f %-8d %-14.3f %-12s"
                      % (V, eps, gamma, agree, "相同" if a == b else "**不同**"))
    print()
    print("读法：草稿再烂（本表最低 argmax 一致率 0.500），T=0 下的输出仍与贪心解码一字不差。")
    print("      原因：拒绝处的残差分布在 T=0 下退化成 argmax(p) 的 one-hot，")
    print("      于是接受与拒绝两条路吐出的是同一个 token。判据里的随机性完全消失。")
    print("      => L2 是 L1 在 T=0 的**退化特例**，不是另一条独立性质。")


def _typical():
    print("=" * 86)
    print("L3 近似判据的偏差：阈值换接受长度，代价是分布真的变了")
    print("=" * 86)
    V, gamma = 5, 3
    tgt = ToyMarkov(V, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p = tgt.dist(0)
    print("目标分布 p(.|s=0) =", np.round(p, 4))
    print()
    print("%-12s %-16s %-16s %-14s" % ("阈值 eps", "首token最大逐点误差", "与p的TV距离", "口径"))
    for eps in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50):
        d = typical_accept_first_token(tgt, drf, 0, gamma, eps)
        err = float(np.abs(d - p).max())
        tv = 0.5 * float(np.abs(d - p).sum())
        tag = "L3 近似" if err > 1e-12 else "无偏"
        print("%-12.2f %-16.4f %-16.4f %-14s" % (eps, err, tv, tag))
    print()
    print("读法（这段是按实测重写的，和「把阈值调保守就能减小偏差」的直觉相反）：")
    print("  1) eps=0 时无条件接受草稿，输出分布**就是草稿分布 q**，误差 0.1719 = max|p-q|。")
    print("  2) eps 从 0 涨到 0.10 期间误差不变 —— 因为本例 p 的最小分量是 0.1045，")
    print("     阈值还没够到任何 token，判据事实上没生效。")
    print("  3) eps 继续涨到 0.20、0.50，误差**反而变大**（0.2201 -> 0.6598）。")
    print("     原因：拒绝掉的质量走的是 norm(max(0,p-q)) 这条补偿路径，")
    print("     而那条补偿是**为 min(1,p/q) 判据配套设计的**。换了判据，补偿就不再匹配，")
    print("     拒绝得越多，走错路的质量越多，偏差越大。")
    print("  => 结论：偏差不是「阈值调一调就能抹掉」的，它是**结构性**的。")
    print("     判据与补偿必须成对更换；只改判据不改补偿，保守化反而更糟。")
    print("     这类做法用分布正确性换接受长度，是**设计选择**不是 bug，")
    print("     但必须如实标成 L3，不能和 L1 方法放进同一张表里比 acceptance length。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--typical", action="store_true")
    a = ap.parse_args()
    if a.greedy:
        _greedy()
    if a.typical:
        _typical()
    if not (a.greedy or a.typical):
        _greedy()
