"""
tree.py —— 树形草稿与树注意力：mask 是怎么造的，以及「同样的验证预算，该深还是该宽」。

两件事：

一、树注意力不是新算子，就是一张**不同形状的 mask**。
    把树按任意顺序拍平成一维序列，让每个节点只能看见自己的祖先链，
    就等价于"把每条根到叶的路径分别当成一条普通因果序列去跑"。
    verify_tree_equals_chains() 用真实的注意力前向把这条等价性算出来（误差 ~1e-16）。

二、给定验证预算 n 个节点（n 个 query token，成本几乎只取决于 n），
    是排成一条深度 n 的**链**好，还是排成一棵浅而宽的**树**好？
    这是 SpecInfer / Medusa / EAGLE-2 / Sequoia 一路在回答的问题。
    tree_vs_chain() 把它算成一张可比较的表。

用法：
    python tree.py --mask     # 打印一棵小树的 mask、深度、路径，并验证等价性
    python tree.py --shape    # 同预算下 链 vs 各种 k 叉树 的期望接受长度
"""
from __future__ import annotations

import argparse

import numpy as np

# --------------------------------------------------------------------------
# 一、树 -> mask
# --------------------------------------------------------------------------


def ancestors(parents: list[int], i: int) -> list[int]:
    """节点 i 的祖先链（含自己），从根到 i。parents[root] = -1。"""
    chain, cur = [], i
    while cur != -1:
        chain.append(cur)
        cur = parents[cur]
    return chain[::-1]


def build_mask(parents: list[int]) -> np.ndarray:
    """树注意力 mask：mask[i, j] = True 表示节点 i 可以看见节点 j。

    规则只有一条：**只能看见自己的祖先（含自己）**。
    普通因果 mask 是这条规则在"树退化成一条链"时的特例。
    """
    n = len(parents)
    m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in ancestors(parents, i):
            m[i, j] = True
    return m


def depths(parents: list[int]) -> np.ndarray:
    """每个节点的深度 —— 它同时就是这个节点该用的 **position id**。

    这是树形草稿最容易写错的地方：拍平后的下标 != 位置编码。
    同一层的兄弟节点下标不同，但 position id 必须相同。
    """
    return np.array([len(ancestors(parents, i)) - 1 for i in range(len(parents))])


def root_to_leaf_paths(parents: list[int]) -> list[list[int]]:
    n = len(parents)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p != -1:
            children[p].append(i)
    return [ancestors(parents, i) for i in range(n) if not children[i]]


# --------------------------------------------------------------------------
# 等价性：树 mask 的一次前向 == 每条路径各跑一次
# --------------------------------------------------------------------------


def _attention(X: np.ndarray, mask: np.ndarray, Wq, Wk, Wv) -> np.ndarray:
    """单头缩放点积注意力，mask 为 True 处可见。X: (n, d)。"""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    S = np.where(mask, S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S)
    A = A / A.sum(axis=-1, keepdims=True)
    return A @ V


def verify_tree_equals_chains(parents: list[int], d: int = 16, seed: int = 0) -> float:
    """把树拍平跑一次，与把每条路径当普通因果序列各跑一次，对比叶子节点的输出。

    返回最大逐元素误差。等价性成立 <=> 误差是机器精度量级。
    """
    rng = np.random.default_rng(seed)
    n = len(parents)
    tok_emb = rng.normal(size=(n, d))          # 每个节点自带的 token embedding
    pos_emb = rng.normal(size=(n + 4, d))      # 位置编码表，按**深度**取
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))

    dep = depths(parents)
    X_tree = tok_emb + pos_emb[dep]
    out_tree = _attention(X_tree, build_mask(parents), Wq, Wk, Wv)

    worst = 0.0
    for path in root_to_leaf_paths(parents):
        L = len(path)
        X_chain = tok_emb[path] + pos_emb[np.arange(L)]   # 链上位置 id = 0..L-1
        causal = np.tril(np.ones((L, L), dtype=bool))
        out_chain = _attention(X_chain, causal, Wq, Wk, Wv)
        # 链上第 t 个 token 对应树里的 path[t]
        for t, node in enumerate(path):
            worst = max(worst, float(np.abs(out_tree[node] - out_chain[t]).max()))
    return worst


# --------------------------------------------------------------------------
# 二、同预算下：链 vs 树
# --------------------------------------------------------------------------


def zipf_dist(V: int, s: float, rng) -> np.ndarray:
    """用 Zipf 近似真实 LLM 的 next-token 分布（少数 token 吃掉大部分概率）。"""
    p = 1.0 / np.power(np.arange(1, V + 1), s)
    p = p / p.sum()
    perm = rng.permutation(V)
    return p[perm]


def coverage_topk(p: np.ndarray, q: np.ndarray, k: int) -> float:
    """草稿取 top-k 候选时，目标模型下一个 token 落在候选集里的概率。

    这是"贪心/精确匹配"口径（L2）下，树的一层能被接受的概率。
    """
    idx = np.argsort(-q)[:k]
    return float(p[idx].sum())


def kary_depth(budget: int, k: int) -> int:
    """预算 budget 个节点时，k 叉树能长到多深（每层全展开）。"""
    d, used = 0, 0
    while True:
        nxt = k ** (d + 1)
        if used + nxt > budget:
            return d
        used += nxt
        d += 1


def expected_accept(c: float, D: int) -> float:
    """每层接受概率为 c、最大深度 D 时的期望接受长度 sum_{d=1..D} c^d。"""
    return float(sum(c ** dd for dd in range(1, D + 1)))


def _mask():
    #        0(root)
    #       /   \
    #      1     2
    #     / \     \
    #    3   4     5
    #        |
    #        6
    parents = [-1, 0, 0, 1, 1, 2, 4]
    print("=" * 74)
    print("一棵 7 节点草稿树：parents =", parents)
    print("=" * 74)
    m = build_mask(parents)
    print("\n树注意力 mask（行=query 节点，列=key 节点，1 表示可见）：")
    print("      " + " ".join("%2d" % j for j in range(len(parents))))
    for i in range(len(parents)):
        print("  %2d  " % i + " ".join(" %d" % int(v) for v in m[i]))
    print("\n每个节点的深度（= 它该用的 position id）：", depths(parents).tolist())
    print("根到叶的全部路径：")
    for p in root_to_leaf_paths(parents):
        print("   ", p)
    print("\n注意两件事：")
    print("  1) mask 每一行的 1 的个数 = 该节点深度+1，且**必是一条祖先链**，不是任意前缀；")
    print("  2) 兄弟节点（如 3 和 4）互相看不见 —— 它们是两个互斥的猜测，不能互相污染。")
    err = verify_tree_equals_chains(parents)
    print("\n等价性验证：树拍平跑一次 vs 每条路径各跑一次，最大逐元素误差 = %.3e" % err)
    print("=> 树注意力没有引入任何新算子，它就是一张换了形状的 mask。")


def make_draft(p: np.ndarray, agree: float, rng) -> np.ndarray:
    """造一个与 p"一致程度 = agree"的草稿分布：q = agree*p + (1-agree)*另一支 Zipf。

    比"往 log p 上加高斯噪声"更贴近真实草稿模型：真实草稿在多数位置与目标高度一致，
    只在少数位置整体跑偏；混合形式能把 top-1 一致率调到 0.4~0.9 的真实区间。
    """
    other = zipf_dist(len(p), 2.0, rng)
    q = agree * p + (1.0 - agree) * other
    return q / q.sum()


def _shape():
    rng = np.random.default_rng(0)
    print("=" * 96)
    print("同样的验证预算下：链深好，还是树宽好？（L2 贪心口径）")
    print("口径：|V|=2000，p~Zipf(2.0)（top-1 概率 0.61，接近真实 LLM）；q = agree*p + (1-agree)*另一支Zipf；")
    print("      每层接受概率 c = top-k 覆盖率；期望接受长度 = sum_{d=1..D} c^d")
    print("      预算 n = 一次验证喂给目标模型的 query token 数（成本几乎只取决于 n）")
    print("=" * 96)
    V = 2000
    p = zipf_dist(V, 2.0, rng)   # s=2.0 时 top-1 概率约 0.61，接近真实 LLM 的置信度
    for agree in (0.95, 0.80, 0.50):
        q = make_draft(p, agree, rng)
        cov = {k: coverage_topk(p, q, k) for k in (1, 2, 4, 8)}
        print("\n--- 草稿一致度 agree=%.2f ---" % agree)
        print("top-k 覆盖率： " + "  ".join("k=%d:%.3f" % (k, cov[k]) for k in (1, 2, 4, 8)))
        print("%-9s %-9s %-7s %-8s %-15s %s" % ("预算n", "形状", "k", "深度D", "期望接受长度", "最优"))
        for budget in (8, 16, 32, 64):
            rows, best = [], None
            for k in (1, 2, 4, 8):
                D = kary_depth(budget, k)
                if D == 0:
                    continue
                e = expected_accept(cov[k], D)
                rows.append((k, D, e))
                if best is None or e > best[2]:
                    best = (k, D, e)
            for k, D, e in rows:
                tag = "链" if k == 1 else "%d叉树" % k
                print("%-9d %-9s %-7d %-8d %-15.4f %s"
                      % (budget, tag, k, D, e, "<==" if k == best[0] else ""))
    print()
    print("读法（这段结论是按实测数据写的，和「草稿差就该加宽」的直觉不一样）：")
    print("  1) **预算是主导因素**：agree=0.95 时，预算 8 是链赢(1.52)，预算 16 起二叉树反超")
    print("     (1.78 -> 2.11 -> 2.36)。预算小的时候加宽度等于砍深度，砍得不值。")
    print("  2) **宽度值不值钱，取决于边际覆盖率增益 cov[k]-cov[1]，不是取决于草稿好不好**：")
    print("     agree=0.95 时 cov[2]-cov[1] = 0.760-0.608 = +0.152，加宽度买得到东西；")
    print("     agree=0.50 时 cov[2]-cov[1] = 0.608-0.608 = 0.000，第二候选是垃圾，")
    print("     于是**再大的预算也是链赢**（1.55 全场最高）。")
    print("  3) 所以「草稿不准就该把树加宽」是错的。草稿不准时，加宽只在「它的 top-k 确实")
    print("     还盖得住概率」时才有用；一个整体跑偏的草稿，宽度买不到任何东西。")
    print("  4) 最优形状同时随预算和边际覆盖率变 => **不存在放之四海皆准的静态树**，")
    print("     这正是 EAGLE-2 之后转向「按草稿置信度动态决定树形状」的原因。")
    print("注意本模型的简化：假设同一层各分支的接受概率相同、且层间独立；")
    print("  真实系统里兄弟分支的概率极不均匀（Zipf），所以真实最优树是**不规则**的，")
    print("  比这里的规则 k 叉树更好 —— 本表给的是规则树里的最优，不是全局最优。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--shape", action="store_true")
    a = ap.parse_args()
    if a.mask:
        _mask()
    if a.shape:
        _shape()
    if not (a.mask or a.shape):
        _mask()
