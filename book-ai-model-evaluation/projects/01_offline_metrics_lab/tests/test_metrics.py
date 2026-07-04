# -*- coding: utf-8 -*-
"""
test_metrics.py —— 离线评估指标库的单元测试
============================================

测试策略（对拍 + 边界 + 聚合）：
  1. 对拍手算值：在小例子上手推出精确答案，断言代码算出的就是它。
  2. 边界情况：全对 / 全错 / 单类 / 空输入 / 完美排序 / 完全逆序。
  3. 切片聚合：验证分组切片的正确性与整体值的一致性。

跑法： python -m pytest -q
"""

import math

import numpy as np
import pytest

import metrics as M


# =============================================================================
# 1. 分类指标测试
# =============================================================================


def test_confusion_matrix_basic():
    # 真实: [1,1,0,0]，预测: [1,0,0,1]
    # 逐个看： (1,1)=TP  (1,0)=FN  (0,0)=TN  (0,1)=FP
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    cm = M.confusion_matrix(y_true, y_pred)
    assert cm.tp == 1
    assert cm.fn == 1
    assert cm.tn == 1
    assert cm.fp == 1


def test_precision_recall_f1_handcomputed():
    # TP=3, FP=1, FN=2 时：
    #   precision = 3/(3+1) = 0.75
    #   recall    = 3/(3+2) = 0.6
    #   f1        = 2*0.75*0.6/(0.75+0.6) = 0.9/1.35 = 0.6666...
    y_true = [1, 1, 1, 1, 1, 0]      # 5 正 1 负
    y_pred = [1, 1, 1, 0, 0, 1]      # 预测 4 个正：命中 3，误报 1
    p, r, f1 = M.precision_recall_f1(y_true, y_pred)
    assert p == pytest.approx(0.75)
    assert r == pytest.approx(0.6)
    assert f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_precision_zero_division():
    # 一个都没预测为正 → precision 分母为 0 → 约定返回 0
    y_true = [1, 0, 1]
    y_pred = [0, 0, 0]
    p, r, f1 = M.precision_recall_f1(y_true, y_pred)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_roc_auc_perfect_separation():
    # 正样本分数全部高于负样本 → 完美区分 → AUC = 1.0
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert M.roc_auc_score(y_true, y_score) == pytest.approx(1.0)


def test_roc_auc_reversed():
    # 分数完全排反 → AUC = 0.0
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.2, 0.1]
    assert M.roc_auc_score(y_true, y_score) == pytest.approx(0.0)


def test_roc_auc_known_value():
    # 经典手算例子（与 sklearn 一致）：
    # y=[1,1,0,0], score=[0.9,0.4,0.6,0.3]
    # 正样本对(0.9,0.4)，负样本对(0.6,0.3)，看所有正负配对里正>负的比例
    #   pairs: (0.9>0.6)T (0.9>0.3)T (0.4>0.6)F (0.4>0.3)T → 3/4 = 0.75
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.4, 0.6, 0.3]
    assert M.roc_auc_score(y_true, y_score) == pytest.approx(0.75)


def test_roc_auc_single_class():
    # 全是正样本 → AUC 无定义 → 约定 0.5
    assert M.roc_auc_score([1, 1, 1], [0.2, 0.5, 0.9]) == pytest.approx(0.5)


def test_pr_auc_perfect():
    # 完美区分时：
    #   * Average Precision (阶梯求和) 应精确等于 1.0
    #   * 梯形法则版 pr_auc_score 会略低于 1.0（这是梯形积分在 recall≈1
    #     处线性插值造成的系统性低估，是已知性质而非 bug），应很高(>0.9)。
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert M.average_precision(y_true, y_score) == pytest.approx(1.0, abs=1e-9)
    assert M.pr_auc_score(y_true, y_score) > 0.9


def test_average_precision_known():
    # 排序 score 降序对应 rel: [1,0,1,0]
    #   命中在位置1(P=1/1=1.0)和位置3(P=2/3≈0.667)
    #   AP = (1.0 + 0.667)/2 = 0.8333
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.6]
    ap = M.average_precision(y_true, y_score)
    assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=1e-6)


def test_roc_curve_monotonic():
    # ROC 曲线的 fpr 和 tpr 都必须单调不减
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_score = rng.random(50)
    # 保证有两类
    y_true[0] = 0
    y_true[1] = 1
    fpr, tpr, _ = M.roc_curve(y_true, y_score)
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert fpr[0] == pytest.approx(0.0)
    assert tpr[0] == pytest.approx(0.0)
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)


# =============================================================================
# 2. 排序指标测试
# =============================================================================


def test_ndcg_perfect_ordering():
    # 模型分数与相关度同序 → NDCG = 1.0
    rel = [3, 2, 1, 0]
    score = [0.9, 0.8, 0.7, 0.6]
    assert M.ndcg_at_k(rel, score, k=4) == pytest.approx(1.0)


def test_ndcg_handcomputed():
    # rel（按模型排序后）= [0, 1, 1]，k=3
    #   DCG = 0/log2(2) + 1/log2(3) + 1/log2(4)
    #       = 0 + 1/1.585 + 1/2 = 0.6309 + 0.5 = 1.1309
    #   理想排序 = [1,1,0]：IDCG = 1/1 + 1/1.585 + 0 = 1 + 0.6309 = 1.6309
    #   NDCG = 1.1309 / 1.6309 = 0.6934
    rel = [0, 1, 1]
    score = [0.9, 0.8, 0.7]  # 模型排序即原序
    dcg = 0 + 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(rel, score, k=3) == pytest.approx(dcg / idcg, abs=1e-6)


def test_ndcg_all_zero_relevance():
    # 没有任何相关文档 → IDCG=0 → 约定返回 0（安全除法）
    assert M.ndcg_at_k([0, 0, 0], [0.5, 0.3, 0.1], k=3) == 0.0


def test_average_precision_at_k_handcomputed():
    # 模型排序后 rel = [1,0,1]，相关文档总数=2，k=3
    #   命中位置1 P=1/1=1.0；命中位置3 P=2/3≈0.667
    #   AP = (1.0 + 0.667)/min(2,3) = 1.667/2 = 0.8333
    rel = [1, 0, 1]
    score = [0.9, 0.8, 0.7]
    ap = M.average_precision_at_k(rel, score, k=3)
    assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=1e-6)


def test_map_multi_query():
    # 两个 query 的 AP 分别是 1.0 和 0.5 → MAP = 0.75
    rels = [[1, 0], [0, 1]]
    scores = [[0.9, 0.1], [0.9, 0.1]]
    # q1: rel排序[1,0]，命中位1 → AP=1.0
    # q2: rel排序[0,1]，命中位2 P=1/2=0.5 → AP=0.5
    m = M.mean_average_precision(rels, scores, k=2)
    assert m == pytest.approx(0.75)


def test_mrr_handcomputed():
    # q1: 第一个相关在位1 → 1/1=1.0
    # q2: 第一个相关在位2 → 1/2=0.5
    # q3: 没有相关 → 0
    # MRR = (1.0 + 0.5 + 0)/3 = 0.5
    rels = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    scores = [[0.9, 0.5, 0.1], [0.9, 0.5, 0.1], [0.9, 0.5, 0.1]]
    assert M.mean_reciprocal_rank(rels, scores) == pytest.approx(0.5)


def test_mrr_empty():
    assert M.mean_reciprocal_rank([], []) == 0.0


# =============================================================================
# 3. 校准指标测试
# =============================================================================


def test_brier_score_handcomputed():
    # p=[0.9,0.1], y=[1,0]  → ((0.9-1)^2 + (0.1-0)^2)/2 = (0.01+0.01)/2 = 0.01
    assert M.brier_score([1, 0], [0.9, 0.1]) == pytest.approx(0.01)


def test_brier_score_perfect():
    # 概率与标签完全一致 → Brier = 0
    assert M.brier_score([1, 0, 1], [1.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_ece_perfectly_calibrated():
    # 构造完美校准：某桶概率=0.5，其中恰好一半为正 → acc=conf → ECE=0
    y_prob = [0.5, 0.5, 0.5, 0.5]
    y_true = [1, 0, 1, 0]  # 该桶 acc=0.5，conf=0.5
    assert M.expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


def test_ece_miscalibrated():
    # 模型全说 0.9（很自信），但实际全错(y=0) → |acc-conf|=|0-0.9|=0.9
    y_prob = [0.9, 0.9, 0.9, 0.9]
    y_true = [0, 0, 0, 0]
    ece = M.expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.9)


def test_reliability_curve_shape():
    y_prob = [0.05, 0.15, 0.95]
    y_true = [0, 0, 1]
    conf, acc, count = M.reliability_curve(y_true, y_prob, n_bins=10)
    assert len(conf) == 10
    assert len(acc) == 10
    assert len(count) == 10
    assert int(np.nansum(count)) == 3  # 3 个样本都被分进某桶


# =============================================================================
# 4. 切片分析测试
# =============================================================================


def test_slice_metric_grouping():
    # 两组：A 组全对，B 组全错。用 accuracy 作为指标。
    def accuracy(yt, yp):
        yt = np.asarray(yt)
        yp = np.asarray(yp)
        return float(np.mean(yt == yp))

    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]  # A组(前2)全对，B组(后2)全错
    groups = ["A", "A", "B", "B"]
    res = M.slice_metric(y_true, y_pred, groups, accuracy)
    assert res["A"] == pytest.approx(1.0)
    assert res["B"] == pytest.approx(0.0)
    assert res["__overall__"] == pytest.approx(0.5)


def test_slice_report_multi_metric():
    def accuracy(yt, yp):
        return float(np.mean(np.asarray(yt) == np.asarray(yp)))

    def positive_rate(yt, yp):
        # 预测为正的比例
        return float(np.mean(np.asarray(yp) == 1))

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    groups = ["X", "X", "Y", "Y"]
    report = M.slice_report(
        y_true, y_pred, groups, {"acc": accuracy, "pos_rate": positive_rate}
    )
    assert "acc" in report and "pos_rate" in report
    assert report["acc"]["X"] == pytest.approx(1.0)
    assert report["acc"]["Y"] == pytest.approx(0.5)
    # 整体正预测率 = 2/4 = 0.5
    assert report["pos_rate"]["__overall__"] == pytest.approx(0.5)


def test_slice_metric_length_mismatch():
    # 长度不一致必须报错
    with pytest.raises(ValueError):
        M.slice_metric([1, 0], [1], ["A", "B"], lambda a, b: 0.0)


# =============================================================================
# 5. 与 numpy 交叉验证（用向量化蛮力法对拍 ROC-AUC）
# =============================================================================


def test_roc_auc_bruteforce_agreement():
    """用"所有正负配对里正分>负分的比例"这个 AUC 定义做蛮力对拍。"""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(10, 40)
        y_true = rng.integers(0, 2, size=n)
        if y_true.sum() == 0 or y_true.sum() == n:
            continue  # 跳过单类
        y_score = rng.random(n)

        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        # 蛮力：对每个正负配对判定，平局记 0.5
        wins = 0.0
        for p in pos:
            wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        brute_auc = wins / (len(pos) * len(neg))

        assert M.roc_auc_score(y_true, y_score) == pytest.approx(
            brute_auc, abs=1e-9
        )
