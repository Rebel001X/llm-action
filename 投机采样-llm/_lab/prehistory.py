"""
prehistory.py —— 把「史前史三条分支为什么死/活」做成可判定的实验（第 08 篇专属）。

三条分支各配一个最小模型，回答一个具体问题：

  --exact    Blockwise（Stern 2018）那条线：**精确匹配判据**（accept iff 草稿 token
             == 打分模型的 argmax）在 T=0 下等价于贪心（L2），但把它原样搬进
             带温度的采样会怎样？答案：输出分布塌成 argmax(p) 的 one-hot，
             与 p 的 TV 距离恰好 = 1 - max(p)。**不是"稍微有偏"，是把采样关掉了。**
             这就是 2022-2023 那两篇必须换成 min(1,p/q) + 残差重采的原因。

  --nat      非自回归（NAT）那条线：条件独立假设的代价是**可算的信息量**。
             在所有"各位置独立"的分布里，离目标联合分布最近的那个就是边缘乘积，
             而它与目标的 KL 恰好等于 total correlation TC = ΣH(Y_t) - H(Y)。
             => 代价由**目标分布本身**决定，加参数/加算力不改变它。

  --jacobi   Jacobi 不动点那条线：在"前缀错一个则后面全错"的模型下，
             并行 Jacobi 迭代每轮净前进**恰好 1 个 token**，迭代次数与自回归相同，
             而每轮要算 T 个位置。这就是 Santilli 2023 的 PJ 全线 <1× 的机理模型。

全部纯 numpy，CPU 秒级，不联网、不下载数据。随机种子固定。
"""
from __future__ import annotations

import argparse

import numpy as np

from spec import ToyMarkov, perturb

# ---------------------------------------------------------------------------
# 分支一：Blockwise 的精确匹配判据（exact match / greedy 口径）
# ---------------------------------------------------------------------------


def exact_match_output_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """精确匹配判据下，一个位置的**精确**输出分布。

    判据（Stern 2018 的 verify 步骤，写成单位置形式）：
        草稿提出 x ~ q；
        若 x == argmax(p) 则接受 x；
        否则回退到打分模型自己的输出 argmax(p)。

    两条路都吐 argmax(p)，所以输出分布**与 q 无关**，恒为 argmax(p) 处的 one-hot。
    这正是它的卖点（相对贪心解码逐 token 相同 = 本库口径的 L2），
    也正是它的天花板（它没有任何办法产出 p 的样本 = 做不到 L1）。
    """
    out = np.zeros_like(p)
    out[int(np.argmax(p))] = 1.0
    return out


def exact_match_bias(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """返回 (最大逐点误差, TV 距离)，相对目标分布 p。"""
    d = exact_match_output_dist(p, q)
    return float(np.abs(d - p).max()), 0.5 * float(np.abs(d - p).sum())


# ---------------------------------------------------------------------------
# 分支二：NAT 的条件独立假设 —— 代价 = total correlation
# ---------------------------------------------------------------------------


def total_correlation(P: np.ndarray) -> float:
    """TC(Y_1..Y_T) = Σ_t H(Y_t) - H(Y_1..Y_T)，单位 nats。P 是联合分布张量。"""
    def H(x: np.ndarray) -> float:
        x = x[x > 0]
        return float(-(x * np.log(x)).sum())

    joint = H(P.ravel())
    marg = 0.0
    for axis in range(P.ndim):
        m = P.sum(axis=tuple(i for i in range(P.ndim) if i != axis))
        marg += H(m)
    return marg - joint


def marginals(P: np.ndarray) -> list[np.ndarray]:
    return [P.sum(axis=tuple(i for i in range(P.ndim) if i != a)) for a in range(P.ndim)]


def product_dist(ms: list[np.ndarray]) -> np.ndarray:
    out = ms[0]
    for m in ms[1:]:
        out = np.tensordot(out, m, axes=0)
    return out


def kl(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P||Q)，nats。Q 有零而 P 非零处返回 inf。"""
    p, q = P.ravel(), Q.ravel()
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def bimodal_example() -> tuple[np.ndarray, list[str], list[str]]:
    """Gu 等 2017 的 'Thank you.' 例子的最小数值化身。

    目标分布：'Danke schön.' 与 'Vielen Dank.' 各 0.5，其余组合 0。
    这是一个 2x2 的联合分布，两个位置**完全相关**（知道第一个词就知道第二个）。
    """
    P = np.zeros((2, 2))
    P[0, 0] = 0.5   # Danke  schön
    P[1, 1] = 0.5   # Vielen Dank
    return P, ["Danke", "Vielen"], ["schön", "Dank"]


# ---------------------------------------------------------------------------
# 分支三：Jacobi 不动点迭代
# ---------------------------------------------------------------------------


def jacobi_rounds(T: int, luck: float = 0.0, seed: int = 0) -> tuple[int, float]:
    """在"前缀一错全废"模型下跑并行 Jacobi 迭代，返回 (收敛轮数, 平均每轮净前进)。

    模型（对 attention 的一个最小刻画，取自 CLLM 论文对失败原因的表述：
    "a LLM can rarely yield a correct token when there are incorrection in
     its preceding tokens due to the attention mechanism"）：
        位置 t 的输出 = 正确 token   若 y[0..t-1] 全部正确；
                      = 正确 token   否则以概率 `luck` 蒙对（模拟"有些 token 不看前缀"）；
                      = 错误 token   其余情况。
    目标序列取 y*[t] = t。初值全部错（用 -1 表示）。

    luck=0 时可归纳证明：第 k 轮后恰好前 k 个位置正确 => 需要 T 轮，
    每轮净前进恰好 1 个 token —— 与自回归的迭代次数**完全相同**。
    """
    rng = np.random.default_rng(seed)
    y = -np.ones(T, dtype=int)
    star = np.arange(T)
    rounds = 0
    while not np.array_equal(y, star):
        rounds += 1
        prefix_ok = True
        new = np.empty(T, dtype=int)
        for t in range(T):
            if prefix_ok:
                new[t] = star[t]
            elif rng.random() < luck:
                new[t] = star[t]
            else:
                new[t] = -2                       # 一个恒错的 token
            if y[t] != star[t]:
                prefix_ok = False                 # 本轮读的是上一轮的 y（Jacobi 语义）
        y = new
        if rounds > 4 * T + 20:                   # 防呆
            break
    return rounds, T / rounds


def jacobi_scan(T: int = 32, lucks=(0.0, 0.1, 0.2, 0.3, 0.5), trials: int = 200):
    """扫 luck，返回 [(luck, 平均轮数, 理想加速比 T/rounds)]。

    "理想加速比" = 假设**并行验证完全免费**（memory-bound 假设，见第 03 篇）时的上界。
    真实系统里每轮要算 T 个位置，若已越过免费额度，这个上界还要再打折。
    """
    out = []
    for lk in lucks:
        rs = [jacobi_rounds(T, lk, seed=s)[0] for s in range(trials)]
        m = float(np.mean(rs))
        out.append((lk, m, T / m))
    return out


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def _exact():
    print("=" * 88)
    print("分支一 Blockwise：精确匹配判据搬进带温度采样 -> 输出分布塌成 one-hot")
    print("=" * 88)
    V, tgt = 5, ToyMarkov(5, seed=0, temp=1.0)
    p = tgt.dist(0)
    print("目标分布 p(.|s=0) =", np.round(p, 4), " max(p) =", round(float(p.max()), 4))
    print()
    print("%-10s %-14s %-18s %-14s %-10s" % ("草稿 eps", "argmax一致率", "最大逐点误差", "TV(out,p)", "口径"))
    for eps in (0.0, 0.5, 1.2, 3.0, 8.0):
        drf = perturb(tgt, eps=eps, seed=1)
        q = drf.dist(0)
        agree = float(np.mean(tgt.T.argmax(1) == drf.T.argmax(1)))
        err, tv = exact_match_bias(p, q)
        print("%-10.1f %-14.3f %-18.4f %-14.4f %-10s" % (eps, agree, err, tv, "L3(在 T>0 下)"))
    print()
    print("读法：")
    print("  1) 五行数字一模一样 —— 偏差**与草稿好坏完全无关**。这与 L1 判据的性质正好相反：")
    print("     L1 下草稿只影响速度不影响分布；exact-match 判据下草稿连速度都影响不到分布，")
    print("     因为输出压根不来自 q，它恒等于 argmax(p)。")
    print("  2) TV = 1 - max(p) = %.4f。这不是「稍微有偏」，是把采样这件事整个关掉了。" % (1 - p.max()))
    print("  3) 所以 Stern 2018 的判据只在 T=0 有意义（那时它就是 L2 贪心等价，是对的）。")
    print("     要支持 T>0，必须把判据换成随机的 min(1,p/q)、并把回退换成残差重采 —— ")
    print("     这正是 Leviathan 2022-11 与 Chen 2023-02 相对它新增的那一块。")


def _nat():
    print("=" * 88)
    print("分支二 NAT：条件独立假设的代价 = total correlation（可算的信息量）")
    print("=" * 88)
    P, w1, w2 = bimodal_example()
    ms = marginals(P)
    Q = product_dist(ms)
    print("目标联合分布（'Thank you.' 的两种合法德译，各 0.5）：")
    for i in range(2):
        for j in range(2):
            print("   P(%-6s %-6s) = %.2f" % (w1[i], w2[j], P[i, j]))
    print()
    print("最优独立近似（= 边缘乘积，NAT 的模型族里最好的那一个）：")
    for i in range(2):
        for j in range(2):
            print("   Q(%-6s %-6s) = %.2f%s" % (w1[i], w2[j], Q[i, j],
                                                "   <- 目标分布里概率为 0 的串" if P[i, j] == 0 else ""))
    leak = float(Q[P == 0].sum())
    print()
    print("  漏到「从未出现过的串」上的概率质量 = %.2f" % leak)
    print("  KL(P||Q) = %.4f nats = %.4f bits" % (kl(P, Q), kl(P, Q) / np.log(2)))
    print("  TC(P)    = %.4f nats  <- 两者相等（这是恒等式，不是巧合）" % total_correlation(P))
    print()
    print("随机联合分布上的复核（最优独立近似确实是边缘乘积）：")
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        R = rng.dirichlet(np.ones(3 * 3 * 3)).reshape(3, 3, 3)
        base = kl(R, product_dist(marginals(R)))
        tc = total_correlation(R)
        worst = max(worst, abs(base - tc))
        for _ in range(20):                        # 随便找些别的独立分布，都不该更好
            alt = [rng.dirichlet(np.ones(3)) for _ in range(3)]
            assert kl(R, product_dist(alt)) >= base - 1e-9
    print("  200 个随机 3x3x3 联合分布：|KL(P||边缘乘积) - TC(P)| 最大 = %.2e" % worst)
    print("  4000 次随机独立分布试探：无一优于边缘乘积。")
    print()
    print("读法：这条线的死因是**信息论的**不是工程的。")
    print("  TC 是目标分布自己的性质；只要模型族被限定成「各位置独立」，")
    print("  这份 KL 就一定要付，加参数、加算力、加数据都不改变它。")
    print("  投机采样的做法相反：它不改输出侧的分布族，只在**草稿侧**并行，")
    print("  最后由目标模型逐位置定稿 —— 于是 TC 一分不丢。")


def _jacobi():
    print("=" * 88)
    print("分支三 Jacobi：'前缀一错全废' 下每轮净前进恰好 1 个 token")
    print("=" * 88)
    print("模型：位置 t 只有在 y[0..t-1] 全对时才输出对（否则以概率 luck 蒙对）")
    print("      —— 这是 CLLM 论文对 vanilla Jacobi 失败原因的最小刻画。\n")
    print("%-10s %-16s %-22s %-16s" % ("luck", "平均收敛轮数", "理想加速比 T/rounds", "结论"))
    for lk, m, sp in jacobi_scan(T=32, lucks=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
        tag = "不如自回归" if sp <= 1.0 + 1e-9 else ("勉强" if sp < 1.5 else "才开始有意义")
        print("%-10.2f %-16.2f %-22.3f %-16s" % (lk, m, sp, tag))
    print()
    print("读法：")
    print("  1) luck=0 时轮数 = 序列长度 32，加速比恰好 1.000 —— 迭代次数一个都没省，")
    print("     而每轮要算 32 个位置。真实系统里这就是净亏损：")
    print("     Santilli 2023 Table 1 的纯 Jacobi (PJ) 在 CPU 上 0.73-0.75x、GPU 上 0.88x。")
    print("  2) 要拿到 1.5x，本模型需要 luck 到 0.3 以上（表里 0.30 才 1.393、0.50 才 1.936）")
    print("     —— 也就是每三个位置就有一个'不看前缀也能蒙对'。")
    print("     CLLM 实测原模型 fast-forward 只有 1.1 token/iter（其中 1.0 是不用")
    print("     Jacobi 也白拿的），净收益约 0.1，离这个门槛差一个数量级。")
    print("  3) 这条线没有白死：它的迭代轨迹后来被 Lookahead Decoding 拿去当 n-gram 池。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--nat", action="store_true")
    ap.add_argument("--jacobi", action="store_true")
    a = ap.parse_args()
    ran = False
    for flag, fn in ((a.exact, _exact), (a.nat, _nat), (a.jacobi, _jacobi)):
        if flag:
            fn()
            ran = True
            print()
    if not ran:
        _exact()
        print()
        _nat()
        print()
        _jacobi()
