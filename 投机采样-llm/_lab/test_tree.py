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


def test_width_is_worthless_without_marginal_coverage():
    """当 cov[2] == cov[1]（第二候选是垃圾）时，无论预算多大，链都不输给树。

    这条推翻了"草稿不准就该把树加宽"的说法（第 16 篇记录）。
    """
    rng = np.random.default_rng(0)
    p = zipf_dist(2000, 2.0, rng)
    q = make_draft(p, 0.50, rng)
    gain = coverage_topk(p, q, 2) - coverage_topk(p, q, 1)
    assert gain / coverage_topk(p, q, 1) < 1e-5, gain   # 第二候选的边际贡献 ~= 0
    for budget in (8, 16, 32, 64):
        chain = expected_accept(coverage_topk(p, q, 1), kary_depth(budget, 1))
        best_tree = max(expected_accept(coverage_topk(p, q, k), kary_depth(budget, k))
                        for k in (2, 4, 8))
        assert chain >= best_tree
