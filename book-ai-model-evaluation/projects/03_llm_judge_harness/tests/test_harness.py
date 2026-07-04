# -*- coding: utf-8 -*-
"""
test_harness.py —— 端到端管线 + 位置偏置度量。

覆盖 book 7.7.1(位置一致率 / 交换顺序检测)、7.8(验证闭环:kappa/相关/win-rate)。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import build_pairwise_dataset, build_pointwise_dataset  # noqa: E402
from judge import GoldOracleJudge, MockLLMJudge, Sample, TIE, WIN_A, WIN_B  # noqa: E402
from harness import (  # noqa: E402
    run_pairwise_validation,
    run_pairwise_with_swap,
    run_pointwise_validation,
    swap_sample,
)
from metrics import position_consistency, robust_pairwise_winner  # noqa: E402


# ---------------------------------------------------------------------------
# swap_sample:交换 A/B
# ---------------------------------------------------------------------------

def test_swap_sample_swaps_answers():
    s = Sample(sample_id="s", prompt="q", answer_a="AAA", answer_b="BBB",
               reference="ref")
    sw = swap_sample(s)
    assert sw.answer_a == "BBB"
    assert sw.answer_b == "AAA"
    assert sw.reference == "ref"
    assert sw.prompt == "q"


# ---------------------------------------------------------------------------
# 位置一致率:纯函数层面
# ---------------------------------------------------------------------------

def test_position_consistency_all_consistent():
    """无位置偏置的理想裁判:换序后胜者视角翻转,一致率=1.0。
    原判 A 赢,换序后应判 B 赢(因为 A/B 换了位置)-> 翻回视角一致。
    """
    original = [WIN_A, WIN_B, TIE]
    swapped = [WIN_B, WIN_A, TIE]  # 完美翻转
    res = position_consistency(original, swapped)
    assert res.consistency_rate == pytest.approx(1.0)
    assert res.flip_rate == pytest.approx(0.0)


def test_position_consistency_full_bias():
    """极端位置偏置:两次都判「第一个位置」(A)赢 -> 完全不一致。
    原判 A、换序也判 A(其实是原来的 B)-> 矛盾 -> 一致率=0。
    """
    original = [WIN_A, WIN_A, WIN_A]
    swapped = [WIN_A, WIN_A, WIN_A]
    res = position_consistency(original, swapped)
    assert res.consistency_rate == pytest.approx(0.0)
    assert res.prefers_first_rate == pytest.approx(1.0)  # 100% 选第一个


def test_position_consistency_partial():
    """部分翻转:book 7.7.1 的 0.6 一致率(40% 翻转)场景。"""
    # 5 对里 3 对一致(翻转),2 对不一致(都选第一个)
    original = [WIN_A, WIN_B, WIN_A, WIN_A, WIN_A]
    swapped = [WIN_B, WIN_A, WIN_B, WIN_A, WIN_A]  # 前 3 翻转, 后 2 都选 A
    res = position_consistency(original, swapped)
    assert res.n == 5
    assert res.n_consistent == 3
    assert res.consistency_rate == pytest.approx(0.6)


def test_position_consistency_length_mismatch():
    with pytest.raises(ValueError):
        position_consistency([WIN_A], [WIN_A, WIN_B])


# ---------------------------------------------------------------------------
# robust_pairwise_winner:交换顺序洗偏置
# ---------------------------------------------------------------------------

def test_robust_winner_consistent_A():
    # 原判 A,换序判 B(翻回是 A)-> 两次都说 A 赢 -> A 真赢
    assert robust_pairwise_winner(WIN_A, WIN_B) == WIN_A


def test_robust_winner_consistent_B():
    assert robust_pairwise_winner(WIN_B, WIN_A) == WIN_B


def test_robust_winner_contradiction_is_tie():
    # 原判 A,换序也判 A(翻回是 B)-> 矛盾 -> tie
    assert robust_pairwise_winner(WIN_A, WIN_A) == TIE


# ---------------------------------------------------------------------------
# 端到端:干净裁判 vs 带偏置裁判在数据集上的位置一致率对比
# ---------------------------------------------------------------------------

def test_clean_judge_has_high_position_consistency():
    samples, _ = build_pairwise_dataset()
    clean = MockLLMJudge(position_bias=0.0)
    rep = run_pairwise_with_swap(clean, samples)
    # 无偏置裁判在换序下应高度一致
    assert rep.position_bias.consistency_rate == pytest.approx(1.0)


def test_biased_judge_has_lower_position_consistency():
    samples, _ = build_pairwise_dataset()
    biased = MockLLMJudge(position_bias=0.6)
    rep = run_pairwise_with_swap(biased, samples)
    clean = MockLLMJudge(position_bias=0.0)
    clean_rep = run_pairwise_with_swap(clean, samples)
    # 带偏置的裁判位置一致率应明显更低
    assert rep.position_bias.consistency_rate < clean_rep.position_bias.consistency_rate
    # 且它更偏爱第一个位置
    assert rep.position_bias.prefers_first_rate > 0.5


# ---------------------------------------------------------------------------
# 端到端:pointwise 验证报告(与金标准的 kappa/相关)
# ---------------------------------------------------------------------------

def test_pointwise_validation_report():
    samples, gold_points = build_pointwise_dataset()
    judge = MockLLMJudge(position_bias=0.0)
    gold = GoldOracleJudge(gold_points, {})
    rep = run_pointwise_validation(judge, gold, samples, "helpfulness")
    assert len(rep.judge_scores) == len(samples)
    assert len(rep.gold_scores) == len(samples)
    # 裁判与金标准应有正相关趋势(方向对齐)
    assert rep.pearson > 0.5
    # kappa 应为正(优于瞎猜)
    assert rep.kappa.kappa > 0.0
    # 分歧数量 = judge != gold 的样本数,应与明细一致
    n_disagree = sum(1 for j, g in zip(rep.judge_scores, rep.gold_scores) if j != g)
    assert len(rep.disagreements) == n_disagree


def test_pointwise_self_agreement_is_perfect():
    """裁判和它自己比,一致率与 kappa 都应为满(确定性的直接推论)。"""
    samples, _ = build_pointwise_dataset()
    judge = MockLLMJudge()
    # 用同一个裁判当「金标准」的分回放
    scores = {s.sample_id: judge.score_pointwise(s, "d").score for s in samples}
    gold = GoldOracleJudge(scores, {})
    rep = run_pointwise_validation(judge, gold, samples, "d")
    assert rep.raw_agree == pytest.approx(1.0)
    assert rep.kappa.kappa == pytest.approx(1.0)
    assert len(rep.disagreements) == 0


# ---------------------------------------------------------------------------
# 端到端:pairwise 验证(稳健胜者 vs 金标准胜者)
# ---------------------------------------------------------------------------

def test_pairwise_validation_report():
    samples, gold_pairs = build_pairwise_dataset()
    judge = MockLLMJudge(position_bias=0.0)
    gold = GoldOracleJudge({}, gold_pairs)
    rep = run_pairwise_validation(judge, gold, samples, "helpfulness")
    assert len(rep.robust_winners) == len(samples)
    # 干净裁判在清晰对上应与金标准高度一致
    assert rep.win_rate_agree >= 0.75
    assert rep.kappa.kappa > 0.0


def test_biased_judge_hurts_pairwise_agreement():
    """位置偏置会拉低与金标准的一致性 —— 说明为什么必须先洗偏置再验证。

    对照:我们比较「不做交换顺序纠偏」的原始胜者 vs 金标准。
    带偏置裁判的原始胜者应比无偏置裁判更偏离金标准。
    """
    samples, gold_pairs = build_pairwise_dataset()
    from metrics import win_rate_agreement

    def raw_winrate(judge):
        winners = [judge.compare_pairwise(s, "d").winner for s in samples]
        gold = [gold_pairs[s.sample_id] for s in samples]
        return win_rate_agreement(winners, gold)

    clean = raw_winrate(MockLLMJudge(position_bias=0.0))
    biased = raw_winrate(MockLLMJudge(position_bias=0.7))
    assert biased <= clean
