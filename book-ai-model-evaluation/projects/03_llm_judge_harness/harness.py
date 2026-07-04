# -*- coding: utf-8 -*-
"""
harness.py —— 把「裁判 + 指标」编排成一条可复现的评估管线。

对应《AI Model Evaluation》第 7 章 7.8「验证闭环」:
    build 验证集 -> 收人类金标准 -> 跑裁判 -> 量化一致性 -> 分歧分析 -> 锁版本。

本模块提供三个高层函数,给 run_demo 和 tests 复用:
    run_pointwise_validation(): 裁判 pointwise 打分 vs 金标准 -> kappa/相关/分歧
    run_pairwise_with_swap():   裁判 pairwise + 交换顺序 -> 位置偏置度量
    run_pairwise_validation():  裁判(稳健裁决)vs 金标准胜者 -> win-rate agreement

设计:harness 完全不关心裁判内部是 Mock 还是真 LLM,只调 BaseJudge 的两个方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from judge import (
    BaseJudge,
    Sample,
    WIN_A,
    WIN_B,
    TIE,
)
from metrics import (
    KappaResult,
    PositionBiasResult,
    ScoreAggregate,
    aggregate_scores,
    cohen_kappa,
    pearson_corr,
    position_consistency,
    raw_agreement,
    robust_pairwise_winner,
    spearman_corr,
    win_rate_agreement,
)


def swap_sample(sample: Sample) -> Sample:
    """交换一个样本的 A/B 两个回答(book 7.7.1 第 2 步「swapped-order」)。

    其余字段保持不变;sample_id 加后缀便于日志区分。
    """
    return Sample(
        prompt=sample.prompt,
        answer_a=sample.answer_b,
        answer_b=sample.answer_a,
        reference=sample.reference,
        sample_id=sample.sample_id + "#swapped",
    )


# ----------------------------------------------------------------------------
# pointwise:裁判打分 vs 金标准,量化校准一致性
# ----------------------------------------------------------------------------

@dataclass
class PointwiseReport:
    judge_scores: List[int]
    gold_scores: List[int]
    kappa: KappaResult          # 把 1~5 分当有序类别做 kappa(book 7.8.2)
    pearson: float              # 连续趋势相关(book 7.8.2)
    spearman: float
    raw_agree: float
    judge_agg: ScoreAggregate   # 裁判打分聚合(book 7.5)
    gold_agg: ScoreAggregate
    disagreements: List[dict]   # 分歧样本明细(book 7.8.1 第 5 步「最有价值的一步」)


def run_pointwise_validation(
    judge: BaseJudge,
    gold: BaseJudge,
    samples: Sequence[Sample],
    dimension: str = "helpfulness",
) -> PointwiseReport:
    """在一批样本上跑裁判 pointwise 打分,并与金标准对齐。

    步骤对应 book 7.8.1:
        3) 在同批样本上跑裁判 -> judge_scores
        (金标准 gold 也走同一接口 -> gold_scores)
        4) 量化一致性 -> kappa / pearson / spearman / raw_agree
        5) 分歧分析 -> disagreements 明细
    """
    judge_scores: List[int] = []
    gold_scores: List[int] = []
    disagreements: List[dict] = []

    for s in samples:
        jv = judge.score_pointwise(s, dimension)
        gv = gold.score_pointwise(s, dimension)
        judge_scores.append(jv.score)
        gold_scores.append(gv.score)
        if jv.score != gv.score:
            disagreements.append({
                "sample_id": s.sample_id,
                "prompt": s.prompt,
                "answer": s.answer_a,
                "judge_score": jv.score,
                "gold_score": gv.score,
                "judge_justification": jv.justification,
            })

    return PointwiseReport(
        judge_scores=judge_scores,
        gold_scores=gold_scores,
        kappa=cohen_kappa(judge_scores, gold_scores),
        pearson=pearson_corr(judge_scores, gold_scores),
        spearman=spearman_corr(judge_scores, gold_scores),
        raw_agree=raw_agreement(judge_scores, gold_scores),
        judge_agg=aggregate_scores(judge_scores),
        gold_agg=aggregate_scores(gold_scores),
        disagreements=disagreements,
    )


# ----------------------------------------------------------------------------
# pairwise + swap:位置偏置检测(book 7.7.1)
# ----------------------------------------------------------------------------

@dataclass
class PairwiseSwapReport:
    original_winners: List[str]  # 原始顺序判决
    swapped_winners: List[str]   # 交换顺序判决(A 指换序后的第一个)
    robust_winners: List[str]    # 稳健裁决:只有正反一致才算真赢,否则 tie
    position_bias: PositionBiasResult
    details: List[dict]          # 每对的两次判决明细(便于回溯翻转对)


def run_pairwise_with_swap(
    judge: BaseJudge,
    samples: Sequence[Sample],
    dimension: str = "helpfulness",
) -> PairwiseSwapReport:
    """对每个样本判两次(原始顺序 + 交换顺序),量化位置偏置。

    book 7.7.1 三步:随机化/交换顺序 -> 两次判 -> 测翻转率。
    """
    original: List[str] = []
    swapped: List[str] = []
    robust: List[str] = []
    details: List[dict] = []

    for s in samples:
        ov = judge.compare_pairwise(s, dimension)
        sv = judge.compare_pairwise(swap_sample(s), dimension)
        original.append(ov.winner)
        swapped.append(sv.winner)
        rw = robust_pairwise_winner(ov.winner, sv.winner)
        robust.append(rw)
        details.append({
            "sample_id": s.sample_id,
            "original_winner": ov.winner,
            "swapped_winner": sv.winner,
            "robust_winner": rw,
            "flipped": robust_pairwise_winner(ov.winner, sv.winner) == TIE
            and ov.winner != TIE,
        })

    pb = position_consistency(original, swapped)
    return PairwiseSwapReport(
        original_winners=original,
        swapped_winners=swapped,
        robust_winners=robust,
        position_bias=pb,
        details=details,
    )


# ----------------------------------------------------------------------------
# pairwise 验证:稳健胜者 vs 人类金标准胜者
# ----------------------------------------------------------------------------

@dataclass
class PairwiseValidationReport:
    robust_winners: List[str]
    gold_winners: List[str]
    win_rate_agree: float       # book 7.8.2 win-rate agreement
    kappa: KappaResult          # 把胜者标签当分类做 kappa
    disagreements: List[dict]


def run_pairwise_validation(
    judge: BaseJudge,
    gold: BaseJudge,
    samples: Sequence[Sample],
    dimension: str = "helpfulness",
) -> PairwiseValidationReport:
    """用「交换顺序洗过位置偏置」的稳健胜者,去和人类金标准胜者对齐。

    这是 book 7.8 的完整精神:先缓解偏置(7.7.1),再验证一致性(7.8.2)。
    """
    robust_winners: List[str] = []
    gold_winners: List[str] = []
    disagreements: List[dict] = []

    for s in samples:
        ov = judge.compare_pairwise(s, dimension)
        sv = judge.compare_pairwise(swap_sample(s), dimension)
        rw = robust_pairwise_winner(ov.winner, sv.winner)
        gw = gold.compare_pairwise(s, dimension).winner
        robust_winners.append(rw)
        gold_winners.append(gw)
        if rw != gw:
            disagreements.append({
                "sample_id": s.sample_id,
                "robust_winner": rw,
                "gold_winner": gw,
                "original_winner": ov.winner,
                "swapped_winner": sv.winner,
            })

    return PairwiseValidationReport(
        robust_winners=robust_winners,
        gold_winners=gold_winners,
        win_rate_agree=win_rate_agreement(robust_winners, gold_winners),
        kappa=cohen_kappa(robust_winners, gold_winners),
        disagreements=disagreements,
    )
