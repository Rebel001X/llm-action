"""树形草稿与树注意力的测试。第 16 篇引用本文件。"""
import numpy as np
import pytest

from tree import (ancestors, build_mask, coverage_topk, depths, expected_accept,
                  kary_depth, make_draft, root_to_leaf_paths,
                  verify_tree_equals_chains, zipf_dist)

TREE = [-1, 0, 0, 1, 1, 2, 4]          # 7 节点样例树
CHAIN = [-1, 0, 1, 2, 3]               # 退化成链


def test_mask_row_is_exactly_the_ancestor_chain():
    """mask 的每一行必须恰好是该节点的祖先集合，不多不少。"""
    m = build_mask(TREE)
    for i in range(len(TREE)):
        assert set(np.flatnonzero(m[i]).tolist()) == set(ancestors(TREE, i))


def test_mask_diagonal_is_true():
    """每个节点都能看见自己（否则第一层就没有 key 可用）。"""
    m = build_mask(TREE)
    assert m.diagonal().all()


def test_siblings_cannot_see_each_other():
    """兄弟节点是互斥猜测，必须互相不可见 —— 否则一条分支会污染另一条。"""
    m = build_mask(TREE)
    for a, b in [(1, 2), (3, 4)]:
        assert not m[a, b] and not m[b, a]


def test_chain_mask_is_ordinary_causal_mask():
    """树退化成链时，树 mask 就是普通下三角因果 mask。"""
    m = build_mask(CHAIN)
    assert np.array_equal(m, np.tril(np.ones_like(m, dtype=bool)))


def test_depth_equals_position_id():
    assert depths(TREE).tolist() == [0, 1, 1, 2, 2, 2, 3]
    assert depths(CHAIN).tolist() == [0, 1, 2, 3, 4]


def test_paths_cover_all_leaves():
    paths = root_to_leaf_paths(TREE)
    assert sorted(p[-1] for p in paths) == [3, 5, 6]
    for p in paths:
        assert p[0] == 0


@pytest.mark.parametrize("parents", [
    TREE, CHAIN,
    [-1, 0, 0, 0],                       # 一层全展开
    [-1, 0, 1, 1, 2, 2, 3, 3, 5],        # 不规则树
])
def test_tree_attention_equals_per_chain(parents):
    """**核心等价性**：树拍平跑一次 == 每条根到叶路径各按普通因果序列跑一次。

    这条断言支撑第 16 篇的论点"树注意力不是新算子，只是换了形状的 mask"。
    """
    assert verify_tree_equals_chains(parents, d=16, seed=0) < 1e-12
    assert verify_tree_equals_chains(parents, d=32, seed=7) < 1e-12


def test_tree_attention_breaks_if_position_ids_wrong():
    """反证：把 position id 从"深度"换成"拍平下标"，等价性立刻被破坏。

    这是树形草稿实现里最常见的一个 bug，本测试把它的后果量化出来。
    """
    rng = np.random.default_rng(0)
    parents, d = TREE, 16
    n = len(parents)
    tok = rng.normal(size=(n, d))
    pos = rng.normal(size=(n + 4, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    from tree import _attention
    X_bad = tok + pos[np.arange(n)]          # 错：用拍平下标当位置
    out_bad = _attention(X_bad, build_mask(parents), Wq, Wk, Wv)
    X_ok = tok + pos[depths(parents)]
    out_ok = _attention(X_ok, build_mask(parents), Wq, Wk, Wv)
    assert np.abs(out_bad - out_ok).max() > 1e-3


# ---------------------------------------------------------------- 形状选择

def test_coverage_increases_with_k():
    rng = np.random.default_rng(0)
    p = zipf_dist(500, 2.0, rng)
    q = make_draft(p, 0.9, rng)
    covs = [coverage_topk(p, q, k) for k in (1, 2, 4, 8, 16)]
    assert all(b >= a - 1e-12 for a, b in zip(covs, covs[1:]))
    assert covs[-1] <= 1.0 + 1e-12


def test_kary_depth_respects_budget():
    for k in (1, 2, 4, 8):
        for budget in (1, 5, 8, 16, 64, 100):
            D = kary_depth(budget, k)
            used = sum(k ** d for d in range(1, D + 1))
            assert used <= budget
            assert used + k ** (D + 1) > budget


def test_expected_accept_bounded_by_depth():
    for c in (0.1, 0.5, 0.9):
        for D in (1, 3, 8):
            assert 0 <= expected_accept(c, D) <= D


def test_tree_beats_chain_at_large_budget():
    """预算大时，二叉树的期望接受长度超过等预算的链（第 16 篇的表）。"""
    rng = np.random.default_rng(0)
    p = zipf_dist(2000, 2.0, rng)
    q = make_draft(p, 0.95, rng)
    chain = expected_accept(coverage_topk(p, q, 1), kary_depth(64, 1))
    binary = expected_accept(coverage_topk(p, q, 2), kary_depth(64, 2))
    assert binary > chain


def test_chain_beats_tree_at_small_budget():
    """预算小时反过来 —— 加宽度等于砍深度，砍得不值。"""
    rng = np.random.default_rng(0)
    p = zipf_dist(2000, 2.0, rng)
    q = make_draft(p, 0.95, rng)
    chain = expected_accept(coverage_topk(p, q, 1), kary_depth(8, 1))
    binary = expected_accept(coverage_topk(p, q, 2), kary_depth(8, 2))
    assert chain > binary


def test_expected_accept_ceiling_is_c_over_one_minus_c():
    r"""$\sum_{d\ge1}c^d = c/(1-c)$：**每种形状都有一个与预算无关的渐近天花板**。

    这条是下面几条的基础，也是第 16 篇 §5.3 改写后的核心原理。
    """
    for c in (0.3, 0.608, 0.828, 0.95):
        ceiling = c / (1 - c)
        assert expected_accept(c, 4000) == pytest.approx(ceiling, rel=1e-6)
        for D in (1, 5, 50):
            assert expected_accept(c, D) < ceiling


def test_higher_coverage_always_means_higher_ceiling():
    r"""**恒真的那一半**：$c_k>c_1 \Rightarrow c_k/(1-c_k) > c_1/(1-c_1)$。

    $x\mapsto x/(1-x)$ 在 $[0,1)$ 上严格增，所以这条**不需要任何阈值**，
    实测 40 种子 × 3 一致度 × 3 个 k 共 360 组，360/360 成立。
    """
    n = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        p = zipf_dist(2000, 2.0, rng)
        for agree in (0.95, 0.80, 0.50):
            q = make_draft(p, agree, rng)
            c1 = coverage_topk(p, q, 1)
            for k in (2, 4, 8):
                ck = coverage_topk(p, q, k)
                if ck <= c1:
                    continue
                n += 1
                assert ck / (1 - ck) > c1 / (1 - c1), (seed, agree, k, c1, ck)
    assert n > 300, n


def test_tree_actually_overtakes_when_gain_is_meaningful():
    r"""**需要条件的那一半**：相对增益 >1% 时，树在预算 4096 上真的反超。

    为什么要加"1%"这个门槛（这是跑数字跑出来的，不是拍的）：
    增益若只有浮点噪声量级（本库的 `make_draft` 在 agree=0.5/0.8 上恰好落在并列点，
    相对差约 1e-5），天花板确实更高，但 $k$ 叉树要爬到反超需要 $\sim2^{31}$ 的预算 ——
    **"迟早会赢"在工程上等于"不会赢"**。
    实测：阈值取 0 时 306/360 反超（漏的 54 组全是并列噪声）；取 1% 时 **306/306**。
    """
    ok = tot = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        p = zipf_dist(2000, 2.0, rng)
        for agree in (0.95, 0.80, 0.50):
            q = make_draft(p, agree, rng)
            c1 = coverage_topk(p, q, 1)
            for k in (2, 4, 8):
                ck = coverage_topk(p, q, k)
                if ck <= c1 * 1.01:          # 增益不足 1%，不在本命题范围内
                    continue
                tot += 1
                if (expected_accept(ck, kary_depth(4096, k))
                        > expected_accept(c1, kary_depth(4096, 1))):
                    ok += 1
    assert tot > 300, tot
    assert ok == tot, "%d/%d 组未反超" % (ok, tot)


def test_coverage_gain_is_always_strictly_positive():
    r"""**再深一层的更正**：$c_k>c_1$ 是**恒真**的，"边际增益为 0"根本不可能发生。

    理由很简单：top-$k$ 候选集包含 top-1 再加 $k-1$ 个 token，而 $p$ 是全支撑的
    （Zipf 每个分量都 >0），多加一个 token 必然多加一份正概率。

    本库初稿在第 16 篇看到 `c_2 = c_1 = 0.608` 就断言"边际贡献恰好是 0"，
    **那只是显示到三位小数的假象** —— 真实增益约 $10^{-5}$。
    实测 40 种子 × 2 一致度 × 3 个 k = 240 组，$c_k\le c_1$ 的样本数是 **0**。
    """
    zero_gain = 0
    total = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        p = zipf_dist(2000, 2.0, rng)
        for agree in (0.80, 0.50):
            q = make_draft(p, agree, rng)
            c1 = coverage_topk(p, q, 1)
            for k in (2, 4, 8):
                total += 1
                if coverage_topk(p, q, k) <= c1:
                    zero_gain += 1
    assert total == 240, total
    assert zero_gain == 0, "居然出现了 %d 组零增益" % zero_gain


def test_budget_needed_grows_as_the_gain_shrinks():
    """**所以真正的判据是"增益多大 vs 预算多大"，不是"有没有增益"。**

    同一个 $k=2$：增益 36% 时预算几十就反超；增益 $10^{-5}$ 时要爬到深度 30+
    才反超，而 $k=2$ 要到深度 30 需要 $2^{31}$ 量级的预算 —— 工程上等于永不反超。
    """
    rng = np.random.default_rng(0)
    p = zipf_dist(2000, 2.0, rng)
    q = make_draft(p, 0.80, rng)
    c1 = coverage_topk(p, q, 1)

    # 增益大（k=4，+36%）：小预算就反超
    c4 = coverage_topk(p, q, 4)
    assert c4 / c1 > 1.3
    assert expected_accept(c4, kary_depth(128, 4)) > expected_accept(c1, kary_depth(128, 1))

    # 增益极小（k=2，~1e-5）：现实预算内永不反超
    c2 = coverage_topk(p, q, 2)
    assert 1.0 < c2 / c1 < 1.001, c2 / c1
    for budget in (64, 1024, 65536):
        assert expected_accept(c2, kary_depth(budget, 2)) < expected_accept(c1, kary_depth(budget, 1))
    # 但它的天花板确实更高 —— 只是够不着
    assert c2 / (1 - c2) > c1 / (1 - c1)


def test_original_never_wins_claim_is_false_at_larger_budget():
    """把被推翻的那个具体反例钉住，防止将来又写回去。"""
    rng = np.random.default_rng(0)
    p = zipf_dist(2000, 2.0, rng)
    q = make_draft(p, 0.80, rng)
    c1, c4 = coverage_topk(p, q, 1), coverage_topk(p, q, 4)
    assert c4 > c1                                    # 4 叉确有边际增益
    chain_128 = expected_accept(c1, kary_depth(128, 1))
    tree_128 = expected_accept(c4, kary_depth(128, 4))
    assert tree_128 > chain_128 * 1.3, (chain_128, tree_128)   # 预算 128 就 +34%
