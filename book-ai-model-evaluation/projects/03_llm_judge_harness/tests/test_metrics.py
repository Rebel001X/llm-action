# -*- coding: utf-8 -*-
"""
test_metrics.py —— 一致性 / 校准指标的正确性。

重点验证 book 7.8.2 的 Cohen's kappa 数值示例、相关性、win-rate agreement,
以及这些指标在退化输入下的健壮性。
"""

import os
import sys

import numpy as np
import pytest

# 让测试能 import 上级目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (  # noqa: E402
    aggregate_scores,
    cohen_kappa,
    pearson_corr,
    raw_agreement,
    spearman_corr,
    win_rate_agreement,
)


# ---------------------------------------------------------------------------
# Cohen's kappa —— 复现 book 7.8.2 的数值示例(混淆矩阵 70/5/10/15)
# ---------------------------------------------------------------------------

def _build_book_example():
    """book 7.8.2: 100 样本,人类 vs 裁判 通过/不通过。
    人类通过 75(其中裁判判通过 70、不通过 5);人类不通过 25(裁判判通过 10、不通过 15)。
    """
    human = ["pass"] * 75 + ["fail"] * 25
    # 前 70 个 pass 被裁判判 pass,接着 5 个 pass 被判 fail
    judge = (["pass"] * 70 + ["fail"] * 5      # 对应人类 75 个 pass
             + ["pass"] * 10 + ["fail"] * 15)  # 对应人类 25 个 fail
    return human, judge


def test_kappa_book_example_values():
    """book 原文: p_o=0.85, p_e=0.65, kappa≈0.57。"""
    human, judge = _build_book_example()
    res = cohen_kappa(judge, human)
    assert res.n == 100
    assert res.p_observed == pytest.approx(0.85, abs=1e-9)
    assert res.p_expected == pytest.approx(0.65, abs=1e-9)
    assert res.kappa == pytest.approx(0.20 / 0.35, abs=1e-6)  # ≈0.5714
    assert 0.56 < res.kappa < 0.58
    assert res.interpretation.startswith("moderate")


def test_kappa_perfect_agreement():
    """完全一致 -> kappa=1(注意要有多于一个类别,否则 p_e 退化)。"""
    a = [1, 2, 3, 4, 5, 1, 2, 3]
    res = cohen_kappa(a, a)
    assert res.kappa == pytest.approx(1.0)
    assert res.p_observed == pytest.approx(1.0)


def test_kappa_below_chance_is_negative():
    """系统性相反的判决 -> kappa < 0(比瞎猜还差)。"""
    a = ["y", "y", "n", "n"]
    b = ["n", "n", "y", "y"]
    res = cohen_kappa(a, b)
    assert res.kappa < 0.0


def test_kappa_chance_level_near_zero():
    """当一方总是判同一类,raw agreement 可能很高但 kappa 应≈0(揭示无判断力)。"""
    # 人类 90% pass;一个「永远瞎猜 pass」的裁判
    human = ["pass"] * 90 + ["fail"] * 10
    lazy_judge = ["pass"] * 100
    res = cohen_kappa(lazy_judge, human)
    assert res.p_observed == pytest.approx(0.90)  # raw agreement 高达 0.90
    assert res.kappa == pytest.approx(0.0, abs=1e-9)  # 但 kappa≈0,揭穿它


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2, 3], [1, 2])


def test_kappa_empty_raises():
    with pytest.raises(ValueError):
        cohen_kappa([], [])


# ---------------------------------------------------------------------------
# raw agreement
# ---------------------------------------------------------------------------

def test_raw_agreement_basic():
    assert raw_agreement([1, 2, 3, 4], [1, 2, 3, 9]) == pytest.approx(0.75)


def test_raw_agreement_empty_raises():
    with pytest.raises(ValueError):
        raw_agreement([], [])


# ---------------------------------------------------------------------------
# 相关性
# ---------------------------------------------------------------------------

def test_pearson_perfect_positive():
    a = [1, 2, 3, 4, 5]
    b = [2, 4, 6, 8, 10]  # 完美线性
    assert pearson_corr(a, b) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1]
    assert pearson_corr(a, b) == pytest.approx(-1.0)


def test_pearson_zero_variance_returns_zero():
    """一方零方差(全相同分)-> 相关无定义,约定返回 0。"""
    a = [3, 3, 3, 3]
    b = [1, 2, 3, 4]
    assert pearson_corr(a, b) == 0.0


def test_spearman_monotonic_but_nonlinear():
    """单调非线性 -> Spearman=1,Pearson<1。"""
    a = [1, 2, 3, 4, 5]
    b = [1, 4, 9, 16, 25]  # 单调递增但非线性
    assert spearman_corr(a, b) == pytest.approx(1.0)
    assert pearson_corr(a, b) < 1.0


def test_spearman_handles_ties():
    """含并列值时应平均秩,不崩。"""
    a = [1, 1, 2, 3]
    b = [5, 5, 6, 7]
    assert spearman_corr(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# win-rate agreement
# ---------------------------------------------------------------------------

def test_win_rate_agreement():
    judge = ["A", "B", "tie", "A"]
    gold = ["A", "B", "tie", "B"]
    assert win_rate_agreement(judge, gold) == pytest.approx(0.75)


def test_win_rate_agreement_length_mismatch():
    with pytest.raises(ValueError):
        win_rate_agreement(["A"], ["A", "B"])


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def test_aggregate_scores():
    agg = aggregate_scores([5, 4, 4, 3, 3, 3])
    assert agg.n == 6
    assert agg.mean == pytest.approx((5 + 4 + 4 + 3 + 3 + 3) / 6)
    assert agg.distribution == {3: 3, 4: 2, 5: 1}
    assert agg.std >= 0.0


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_scores([])
