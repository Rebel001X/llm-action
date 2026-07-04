# -*- coding: utf-8 -*-
"""
metrics.py —— 一致性 / 校准 / 位置偏置的度量。

对应《AI Model Evaluation》第 7 章:
    - 7.8.2 一致性指标:Cohen's kappa、与人类分的相关性、win-rate agreement
    - 7.7.1 位置偏置:交换顺序后的「位置一致率 position-consistency rate」
    - 7.5   Scoring 的聚合(mean / 分布)

设计原则:纯 numpy,函数式,输入 = 两个等长序列,输出 = 一个数或一个小 dataclass。
所有公式都在 docstring 里写出 LaTeX,和 book 的推导一一对应。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ----------------------------------------------------------------------------
# 1) 原始一致率(raw agreement)—— 故意也实现它,好和 kappa 对比暴露它的陷阱
# ----------------------------------------------------------------------------

def raw_agreement(a: Sequence, b: Sequence) -> float:
    """原始一致率 = 两个评审者判决相同的比例。

    p_o = (判决相同的样本数) / (总样本数)

    ⚠️ 陷阱(book 7.8.2):若某一类占绝对多数,一个「永远瞎猜多数类」的裁判
    也能拿到很高的 raw agreement,但它毫无判断力。所以要用 kappa 扣掉运气。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("两个序列长度必须一致")
    if a.size == 0:
        raise ValueError("空序列无法计算一致率")
    return float(np.mean(a == b))


# ----------------------------------------------------------------------------
# 2) Cohen's kappa —— 扣除「碰巧一致」后的净一致性(book 7.8.2 核心)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class KappaResult:
    kappa: float          # Cohen's kappa
    p_observed: float     # 观测一致率 p_o
    p_expected: float     # 随机期望一致率 p_e
    n: int                # 样本数
    interpretation: str   # 文字解读(Landis & Koch 经验阈值)


def _kappa_interpretation(k: float) -> str:
    """Landis & Koch(1977)经验阈值,book 7.8.2 提到的 >0.6 substantial、>0.8 almost perfect。"""
    if k < 0.0:
        return "poor(比瞎猜还差)"
    if k < 0.20:
        return "slight(几乎无一致)"
    if k < 0.40:
        return "fair(一般)"
    if k < 0.60:
        return "moderate(中等)"
    if k < 0.80:
        return "substantial(实质性一致)"
    return "almost perfect(几乎完美)"


def cohen_kappa(a: Sequence, b: Sequence) -> KappaResult:
    r"""计算两个评审者对同一批样本分类判决的 Cohen's kappa。

    公式(book 7.8.2):
        kappa = (p_o - p_e) / (1 - p_e)
    其中:
        p_o = 观测一致率(两人判决相同的比例)
        p_e = 随机期望一致率 = sum_k ( P(a=k) * P(b=k) )
              即把每个类别 k 的「a 判 k 的边际概率」乘「b 判 k 的边际概率」再求和。

    直觉:p_e 是「两人各自按自己的边际分布独立瞎猜时,恰好撞上的概率」。
    kappa 把这块「运气一致」从分子里减掉,再用 (1 - p_e) 归一化到 [-1, 1]。
        kappa=1 完美;kappa=0 等于瞎猜;kappa<0 比瞎猜还差。

    数值示例(book 7.8.2 表): 100 样本, p_o=0.85, p_e=0.65 -> kappa≈0.57。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("两个序列长度必须一致")
    n = a.size
    if n == 0:
        raise ValueError("空序列无法计算 kappa")

    # 全部出现过的类别(取两序列并集,保证列联表覆盖所有标签)
    classes = np.unique(np.concatenate([a, b]))

    p_o = float(np.mean(a == b))

    # 各自的边际分布 P(a=k)、P(b=k)
    p_e = 0.0
    for k in classes:
        pa = float(np.mean(a == k))
        pb = float(np.mean(b == k))
        p_e += pa * pb

    if np.isclose(p_e, 1.0):
        # 退化情形:两人都只用一个类别 -> 期望一致率=1,kappa 无定义。
        # 约定:若同时也完全一致则记 kappa=1,否则 0(避免除零)。
        kappa = 1.0 if np.isclose(p_o, 1.0) else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return KappaResult(
        kappa=float(kappa),
        p_observed=p_o,
        p_expected=float(p_e),
        n=int(n),
        interpretation=_kappa_interpretation(float(kappa)),
    )


# ----------------------------------------------------------------------------
# 3) 相关性 —— 连续打分(1~5)与人类分的趋势一致性(book 7.8.2)
# ----------------------------------------------------------------------------

def pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    r"""Pearson 相关系数,衡量两组连续分是否「同向变化」。

        r = cov(a,b) / (std(a) * std(b))

    适合 1~5 这种连续打分的趋势对齐(book 7.8.2:Correlation with human scores)。
    退化:任一方零方差(所有分相同)-> 相关无定义,返回 0.0。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("两个序列长度必须一致")
    if a.size < 2:
        raise ValueError("至少需要 2 个样本才能算相关")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: Sequence[float], b: Sequence[float]) -> float:
    r"""Spearman 秩相关 = 对「秩次」做 Pearson,衡量单调关系(对非线性更稳)。

    做法:把原始分转成排名(rank),再算 Pearson。book 7.8.2 提到 Pearson/Spearman
    都可用于连续打分的一致性。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("两个序列长度必须一致")
    if a.size < 2:
        raise ValueError("至少需要 2 个样本才能算相关")
    ra = _rankdata(a)
    rb = _rankdata(b)
    return pearson_corr(ra, rb)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """把数组转成秩次(1-based),并列取平均秩(average rank),等价 scipy.stats.rankdata。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # 处理并列:同值取平均秩
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    # 对每个唯一值,求它占据的秩位置的平均
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


# ----------------------------------------------------------------------------
# 4) win-rate agreement —— 成对比较里裁判和人对「谁赢」的一致比例(book 7.8.2)
# ----------------------------------------------------------------------------

def win_rate_agreement(judge_winners: Sequence[str], gold_winners: Sequence[str]) -> float:
    """裁判和金标准对「胜者」判断一致的比例(把 tie 也当一个类别一起比)。

    book 7.8.2:Win-rate agreement 适用于 pairwise 比较。
    """
    j = np.asarray(judge_winners)
    g = np.asarray(gold_winners)
    if j.shape != g.shape:
        raise ValueError("两个序列长度必须一致")
    if j.size == 0:
        raise ValueError("空序列无法计算 win-rate agreement")
    return float(np.mean(j == g))


# ----------------------------------------------------------------------------
# 5) 位置偏置:交换顺序后的「位置一致率」(book 7.7.1 —— 本章重点考点)
# ----------------------------------------------------------------------------

# 交换顺序后,胜者标签需要「翻回」到原始 A/B 视角才能比较。
# 原始判决 winner in {'A','B','tie'};换序后 A、B 互换了位置,
# 因此换序判 'A' 实际是原来的 B 赢,'B' 实际是原来的 A 赢,'tie' 不变。
def _flip_winner(w: str) -> str:
    from judge import WIN_A, WIN_B, TIE
    if w == WIN_A:
        return WIN_B
    if w == WIN_B:
        return WIN_A
    return TIE


@dataclass(frozen=True)
class PositionBiasResult:
    consistency_rate: float   # 位置一致率 C/N(换序后胜者不变的比例)
    flip_rate: float          # 翻转率 = 1 - 一致率
    n: int                    # 对数
    n_consistent: int         # 换序后仍判同一方赢的对数
    prefers_first_rate: float # 「偏爱第一个位置」的比例(诊断偏向方向)


def position_consistency(original_winners: Sequence[str],
                         swapped_winners: Sequence[str]) -> PositionBiasResult:
    r"""量化位置偏置(book 7.7.1「位置一致率」)。

    做法(book 7.7.1 三步):
        1) 原始顺序判一次 -> original_winners(A/B/tie,A 指「原始第一个」)
        2) 交换顺序再判一次 -> swapped_winners(A/B/tie,注意此时 A 指「换序后的第一个」)
        3) 把换序判决「翻回」原始视角(_flip_winner),再看两次是否判同一方赢。

    一致率:
        consistency_rate = C / N
    其中 N 为总对数,C 为「换序后翻回视角仍与原判一致」的对数。
    一致率明显 < 1.0 说明裁判受呈现顺序影响(book 7.7.1 的 40% 翻转示例)。

    额外诊断 prefers_first_rate:两次判决里「选了当前第一个位置」的频率。
        真实偏爱第一个(book 说常偏爱 Response A / 列表首项)时,该值会显著 > 0.5。
    """
    from judge import WIN_A
    o = list(original_winners)
    s = list(swapped_winners)
    if len(o) != len(s):
        raise ValueError("两次判决的对数必须一致")
    n = len(o)
    if n == 0:
        raise ValueError("空序列无法计算位置一致率")

    consistent = 0
    prefers_first = 0
    for ow, sw in zip(o, s):
        # 把换序判决翻回原始视角后,与原判比较是否一致
        if ow == _flip_winner(sw):
            consistent += 1
        # 诊断:原始那次判「A」= 选了原始第一个;换序那次判「A」= 选了换序第一个
        if ow == WIN_A:
            prefers_first += 1
        if sw == WIN_A:
            prefers_first += 1

    consistency_rate = consistent / n
    prefers_first_rate = prefers_first / (2 * n)  # 两次判决共 2N 个「首位选择机会」
    return PositionBiasResult(
        consistency_rate=consistency_rate,
        flip_rate=1.0 - consistency_rate,
        n=n,
        n_consistent=consistent,
        prefers_first_rate=prefers_first_rate,
    )


def robust_pairwise_winner(original_winner: str, swapped_winner: str) -> str:
    """book 7.7.1 实战框:只有「正反两次都判同一方赢」才算它真赢,否则记 tie。

    这是把位置偏置「洗掉」的稳健裁决法:
        - 原判 A 赢 且 换序翻回后也 A 赢 -> A 真赢
        - 原判 B 赢 且 换序翻回后也 B 赢 -> B 真赢
        - 两次矛盾 -> tie(不确定,交给人工或更强裁判)
    """
    from judge import WIN_A, WIN_B, TIE
    flipped = _flip_winner(swapped_winner)
    if original_winner == flipped and original_winner in (WIN_A, WIN_B):
        return original_winner
    return TIE


# ----------------------------------------------------------------------------
# 6) 聚合:pointwise 打分的均值 / 分布(book 7.5 Scoring:看聚合趋势)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreAggregate:
    mean: float
    std: float
    n: int
    distribution: dict  # {分值: 计数}


def aggregate_scores(scores: Sequence[int]) -> ScoreAggregate:
    """聚合一批 1~5 打分:均值、标准差、分布直方(book 7.5「跨大量样本看聚合趋势」)。"""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("空序列无法聚合")
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    dist = {int(v): int(c) for v, c in zip(vals, counts)}
    return ScoreAggregate(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        n=int(arr.size),
        distribution=dist,
    )
