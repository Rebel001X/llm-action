# -*- coding: utf-8 -*-
"""
跨模态检索单元测试
==================
覆盖 4 类硬性要求:
    1. 自检索 recall@1 == 1
    2. 加噪后 recall 下降
    3. recall@k 关于 k 单调不减
    4. rank 计算正确 (含手工构造的相似度矩阵)
外加: 归一化、温度不改排序、softmax 概率合法 等边界。
"""

import os
import sys

import numpy as np
import pytest

# 让测试能 import 到上一层的模块 (不依赖安装)。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval import (  # noqa: E402
    add_noise,
    l2_normalize,
    median_rank,
    ranks_of_ground_truth,
    recall_at_k,
    retrieval_report,
    similarity_matrix,
    softmax_retrieval_probs,
    topk_indices,
)
from toydata import make_easy_pairs, make_paired_embeddings  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. 自检索 recall@1 == 1                                                       #
# --------------------------------------------------------------------------- #
def test_self_retrieval_recall1_is_one():
    """一个向量集合对自己检索: 每个 query 最像的一定是它自己 -> recall@1=1。"""
    img, _ = make_paired_embeddings(n=100, dim=48, modality_sigma=0.3, seed=1)
    # 用 img 同时当 query 和 gallery,正确答案就是对角线。
    rep = retrieval_report(img, img, ks=(1,), normalize=True)
    assert rep["recall@1"] == 1.0
    assert rep["median_rank"] == 1.0


def test_easy_paired_recall1_is_one():
    """模态噪声极小的图文对: 文->图 自然对齐,recall@1 应为 1。"""
    img, txt = make_easy_pairs(n=60, dim=32, seed=7)
    rep = retrieval_report(txt, img, ks=(1, 5), normalize=True)  # 文 -> 图
    assert rep["recall@1"] == 1.0


# --------------------------------------------------------------------------- #
# 2. 加噪后 recall 下降                                                          #
# --------------------------------------------------------------------------- #
def test_noise_degrades_recall():
    """给文嵌入注入越来越大的噪声,recall@1 应单调不增 (总体下降)。"""
    img, txt = make_paired_embeddings(n=300, dim=64, modality_sigma=0.3, seed=3)
    rng = np.random.default_rng(0)
    prev = None
    recalls = []
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0]:
        noisy_txt = add_noise(txt, sigma, rng=rng)
        rep = retrieval_report(noisy_txt, img, ks=(1,), normalize=True)
        recalls.append(rep["recall@1"])
        if prev is not None:
            # 允许极小抖动,但整体不该上升。
            assert rep["recall@1"] <= prev + 1e-9
        prev = rep["recall@1"]
    # 首尾对比: 大噪声必然显著劣于无噪声。
    assert recalls[-1] < recalls[0]


def test_noise_increases_median_rank():
    """加噪不仅降 recall,还会抬高 median rank。"""
    img, txt = make_paired_embeddings(n=300, dim=64, modality_sigma=0.3, seed=11)
    rng = np.random.default_rng(1)
    clean = retrieval_report(txt, img, ks=(1,))["median_rank"]
    noisy = retrieval_report(add_noise(txt, 3.0, rng=rng), img, ks=(1,))["median_rank"]
    assert noisy >= clean


# --------------------------------------------------------------------------- #
# 3. recall@k 关于 k 单调不减                                                    #
# --------------------------------------------------------------------------- #
def test_recall_monotonic_in_k():
    """k 越大,能命中的 query 只多不少 -> recall@k 单调不减。"""
    img, txt = make_paired_embeddings(n=200, dim=64, modality_sigma=0.8, seed=5)
    sim = similarity_matrix(txt, img)
    ranks = ranks_of_ground_truth(sim)
    ks = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    vals = [recall_at_k(ranks, k) for k in ks]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12
    # k >= gallery 大小时,recall 必为 1 (正确项一定在里面)。
    assert recall_at_k(ranks, 200) == 1.0


# --------------------------------------------------------------------------- #
# 4. rank 计算正确 (手工构造)                                                    #
# --------------------------------------------------------------------------- #
def test_ranks_hand_crafted():
    """用一个能手算的相似度矩阵校验排名逻辑。

    sim 行 = query, 列 = gallery, gt 默认对角线:
        query0 对 gallery0 的分数 = 0.9, 是本行最大 -> rank 1
        query1 对 gallery1 的分数 = 0.2, 本行有 0.5 更大 -> rank 2
        query2 对 gallery2 的分数 = 0.1, 本行有 0.8 和 0.3 更大 -> rank 3
    """
    sim = np.array(
        [
            [0.9, 0.1, 0.2],  # gt=0 最大 -> rank1
            [0.5, 0.2, 0.1],  # gt=1(0.2), 只有 0.5 更大 -> rank2
            [0.8, 0.3, 0.1],  # gt=2(0.1), 0.8 与 0.3 更大 -> rank3
        ]
    )
    ranks = ranks_of_ground_truth(sim)
    assert list(ranks) == [1, 2, 3]
    assert recall_at_k(ranks, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranks, 2) == pytest.approx(2 / 3)
    assert recall_at_k(ranks, 3) == pytest.approx(1.0)
    assert median_rank(ranks) == 2.0


def test_ranks_with_custom_gt():
    """指定非对角线的 ground-truth,验证 gt 参数生效。"""
    sim = np.array(
        [
            [0.1, 0.9, 0.3],  # 正确答案指向列1(0.9) -> rank1
            [0.7, 0.2, 0.6],  # 正确答案指向列2(0.6), 只有 0.7 更大 -> rank2
        ]
    )
    gt = np.array([1, 2])
    ranks = ranks_of_ground_truth(sim, gt)
    assert list(ranks) == [1, 2]


def test_ranks_tie_handling():
    """并列时用严格 '>' 计数,正确项不该把自己也算进去 -> rank 至少为 1。"""
    sim = np.array([[0.5, 0.5, 0.5]])  # 三者并列
    ranks = ranks_of_ground_truth(sim, np.array([0]))
    # 没有任何一项严格大于正确项 -> rank = 1
    assert ranks[0] == 1


# --------------------------------------------------------------------------- #
# 5. top-k 下标正确                                                             #
# --------------------------------------------------------------------------- #
def test_topk_indices_order_and_shape():
    sim = np.array(
        [
            [0.1, 0.9, 0.3, 0.5],
            [0.8, 0.2, 0.7, 0.0],
        ]
    )
    top = topk_indices(sim, k=2)
    assert top.shape == (2, 2)
    # 第 0 行降序应是 列1(0.9) 然后 列3(0.5)
    assert list(top[0]) == [1, 3]
    # 第 1 行降序应是 列0(0.8) 然后 列2(0.7)
    assert list(top[1]) == [0, 2]


def test_topk_k_larger_than_gallery():
    """k 超过 gallery 大小时应自动裁剪,不报错。"""
    sim = np.array([[0.2, 0.5]])
    top = topk_indices(sim, k=10)
    assert top.shape == (1, 2)
    assert list(top[0]) == [1, 0]


# --------------------------------------------------------------------------- #
# 6. 归一化                                                                     #
# --------------------------------------------------------------------------- #
def test_l2_normalize_unit_norm():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    xn = l2_normalize(x)
    # 第一行 (3,4) 范数 5 -> (0.6, 0.8)
    np.testing.assert_allclose(xn[0], [0.6, 0.8])
    # 全 0 行不该产生 NaN
    assert np.isfinite(xn[1]).all()
    # 非零行范数应为 1
    assert np.isclose(np.linalg.norm(xn[2]), 1.0)


def test_normalize_changes_retrieval():
    """构造一个例子: 不归一化因长度偏置检索错,归一化后修正。"""
    # gallery: 两条候选,方向不同。
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
    # query 方向明明更贴近 gallery[1] (纯 y 方向),
    # 但我们把 gallery[0] 放大 10 倍制造长度偏置。
    gallery_biased = np.array([[10.0, 0.0], [0.0, 1.0]])
    query = np.array([[0.1, 1.0]])  # 主要是 y 方向

    # 不归一化: 点积 = [1.0, 1.0] 恰好并列, 长度偏置会干扰。
    sim_raw = similarity_matrix(query, gallery_biased, normalize=False)
    # 归一化后按方向: query 更像 gallery[1]。
    sim_norm = similarity_matrix(query, gallery_biased, normalize=True)
    assert np.argmax(sim_norm[0]) == 1
    # 说明两种口径确实可能给出不同结论 (值不同)。
    assert not np.allclose(sim_raw, sim_norm)


# --------------------------------------------------------------------------- #
# 7. 温度与 softmax                                                             #
# --------------------------------------------------------------------------- #
def test_temperature_does_not_change_ranking():
    """温度是单调变换,不改变 top-k 排序 -> recall / rank 与 T 无关。"""
    img, txt = make_paired_embeddings(n=120, dim=48, modality_sigma=0.6, seed=9)
    sim = similarity_matrix(txt, img)
    ranks_base = ranks_of_ground_truth(sim)
    for T in [0.05, 0.5, 1.0, 5.0, 100.0]:
        # softmax 后再按概率排名, 排序应与原始一致。
        probs = softmax_retrieval_probs(sim, temperature=T)
        ranks_T = ranks_of_ground_truth(probs)
        np.testing.assert_array_equal(ranks_base, ranks_T)


def test_softmax_probs_valid():
    sim = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax_retrieval_probs(sim, temperature=1.0)
    # 每行和为 1
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0])
    # 全 0 行 -> 均匀分布
    np.testing.assert_allclose(p[1], [1 / 3, 1 / 3, 1 / 3])
    # 都是合法概率
    assert (p >= 0).all() and (p <= 1).all()


def test_lower_temperature_sharper():
    """温度越低,分布越尖 (最大概率越大)。"""
    sim = np.array([[1.0, 2.0, 3.0]])
    p_low = softmax_retrieval_probs(sim, temperature=0.1)
    p_high = softmax_retrieval_probs(sim, temperature=10.0)
    assert p_low.max() > p_high.max()


def test_softmax_bad_temperature_raises():
    with pytest.raises(ValueError):
        softmax_retrieval_probs(np.array([[1.0, 2.0]]), temperature=0.0)


# --------------------------------------------------------------------------- #
# 8. 双向检索一致性 (i2t vs t2i 都能算)                                          #
# --------------------------------------------------------------------------- #
def test_both_directions_run():
    img, txt = make_easy_pairs(n=40, dim=32, seed=13)
    t2i = retrieval_report(txt, img, ks=(1, 5))  # 文 -> 图
    i2t = retrieval_report(img, txt, ks=(1, 5))  # 图 -> 文
    assert t2i["recall@1"] == 1.0
    assert i2t["recall@1"] == 1.0
