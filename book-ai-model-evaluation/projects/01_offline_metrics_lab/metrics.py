# -*- coding: utf-8 -*-
"""
metrics.py —— 离线评估指标库（纯 numpy 实现）
================================================

本模块把机器学习离线评估里最常用的四大类指标从零手写一遍，
只依赖 numpy，不依赖 sklearn，方便你彻底看清每个公式的每一步在算什么。

四大类：
  1. 分类指标 (Classification):  precision / recall / F1 / PR-AUC / ROC-AUC
  2. 排序指标 (Ranking):         NDCG / MAP / MRR
  3. 校准指标 (Calibration):     ECE / Brier score
  4. 切片分析 (Slice analysis):  按分组维度聚合任意指标

设计约定
--------
* 所有函数输入统一用 numpy 数组或可被 np.asarray 转换的序列。
* y_true 一律是 0/1 标签（二分类）或相关性等级（排序）。
* y_score 一律是"越大越正"的分数（可以是概率，也可以是任意实数打分）。
* 除法一律走 _safe_divide，避免 0/0 抛异常或产生 nan 污染下游。

作者注：变量命名尽量贴近数学公式，读代码时对着 README 的 LaTeX 看。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# =============================================================================
# 0. 工具函数 (utilities)
# =============================================================================


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法：分母为 0 时返回 default，而不是抛异常或返回 nan。

    为什么需要它？评估里经常出现"某类样本一个都没有"的情况，
    比如 recall = TP / (TP + FN)，若正样本为 0，分母就是 0。
    我们约定这种"无定义"的情况返回 0.0（也可按需改成 nan）。
    """
    if denominator == 0:
        return default
    return numerator / denominator


def _as_1d_float(x: Sequence[float]) -> np.ndarray:
    """把输入转成一维 float64 数组，统一类型，避免整型除法等坑。"""
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr


def _check_same_length(*arrays: np.ndarray) -> None:
    """校验多个数组长度一致，不一致直接报错（评估里长度对不上一定是 bug）。"""
    lengths = {len(a) for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"输入数组长度不一致: {[len(a) for a in arrays]}")


# =============================================================================
# 1. 分类指标 (Classification metrics)
# =============================================================================


@dataclass
class ConfusionMatrix:
    """二分类混淆矩阵 (Confusion Matrix)。

    四个格子（约定 1 为正类 positive，0 为负类 negative）：
        tp: True  Positive  —— 真实为正，预测为正（预测对了的正样本）
        fp: False Positive  —— 真实为负，预测为正（误报，"假警报"）
        fn: False Negative  —— 真实为正，预测为负（漏报，"没抓到"）
        tn: True  Negative  —— 真实为负，预测为负（预测对了的负样本）
    """

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        """精确率 = TP / (TP + FP)：预测为正的里面，有多少是真的正。"""
        return _safe_divide(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """召回率 = TP / (TP + FN)：真实的正里面，被抓回来了多少。"""
        return _safe_divide(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        """F1 = 精确率与召回率的调和平均，二者都高才高。"""
        p, r = self.precision, self.recall
        return _safe_divide(2 * p * r, p + r)

    @property
    def accuracy(self) -> float:
        """准确率 = (TP + TN) / 全部：整体预测对的比例（类别不平衡时会骗人）。"""
        total = self.tp + self.fp + self.fn + self.tn
        return _safe_divide(self.tp + self.tn, total)


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> ConfusionMatrix:
    """由真实标签与预测标签（都是 0/1）计算混淆矩阵。

    参数
    ----
    y_true : 真实标签，元素 ∈ {0, 1}
    y_pred : 预测标签（已经用某阈值二值化过），元素 ∈ {0, 1}

    返回 ConfusionMatrix。
    """
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_pred)
    _check_same_length(yt, yp)

    # 用布尔掩码一次性把四个格子数出来，比循环快也更清晰
    yt_pos = yt == 1
    yp_pos = yp == 1
    tp = int(np.sum(yt_pos & yp_pos))
    fp = int(np.sum(~yt_pos & yp_pos))
    fn = int(np.sum(yt_pos & ~yp_pos))
    tn = int(np.sum(~yt_pos & ~yp_pos))
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def precision_recall_f1(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[float, float, float]:
    """一次性返回 (precision, recall, f1)，方便调用。"""
    cm = confusion_matrix(y_true, y_pred)
    return cm.precision, cm.recall, cm.f1


def _binary_clf_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算不同阈值下的累计 TP / FP，是 ROC 和 PR 曲线的公共骨架。

    核心思想（这是理解 AUC 的关键）：
      1. 把样本按 score 从大到小排序。
      2. 从上往下逐个"把阈值降低"，相当于逐个把样本判为正。
      3. 每遇到一个正样本，累计 TP +1；每遇到一个负样本，累计 FP +1。
      4. 只在"分数变化的位置"记录一个阈值点（相同分数要合并，否则曲线锯齿）。

    返回
    ----
    fps : 各阈值点处的累计假正数 (cumulative false positives)
    tps : 各阈值点处的累计真正数 (cumulative true positives)
    thresholds : 各阈值点对应的 score 值（从大到小）
    """
    # argsort 默认升序，用 [::-1] 反成降序 → score 大的排前面
    desc_order = np.argsort(y_score, kind="mergesort")[::-1]
    score_sorted = y_score[desc_order]
    y_sorted = y_true[desc_order]

    # 找出"分数发生变化"的位置（distinct value 的最后一个索引）
    # np.diff 相邻做差，非 0 处即分数变了；末尾必是一个阈值点
    distinct_mask = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct_mask, y_true.size - 1]

    # cumsum 累加正样本个数 → 每个阈值点处的累计 TP
    tps = np.cumsum(y_sorted)[threshold_idxs]
    # 阈值点的序号 +1 是"已判为正"的样本总数，减去 TP 就是 FP
    fps = 1 + threshold_idxs - tps
    return fps, tps, score_sorted[threshold_idxs]


def roc_curve(
    y_true: Sequence[int], y_score: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ROC 曲线的 (fpr, tpr, thresholds)。

    ROC = Receiver Operating Characteristic（受试者工作特征曲线）。
    横轴 FPR = FP / (FP + TN)  —— 负样本里被误判为正的比例（越低越好）
    纵轴 TPR = TP / (TP + FN)  —— 正样本里被正确抓回的比例（= recall，越高越好）

    曲线从 (0,0) 走到 (1,1)，越贴近左上角越好。
    """
    yt = _as_1d_float(y_true)
    ys = _as_1d_float(y_score)
    _check_same_length(yt, ys)

    fps, tps, thresholds = _binary_clf_curve(yt, ys)

    # 在最前面补一个 (0,0) 点，让曲线从原点出发
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]  # 一个"比最大分还大"的阈值

    total_pos = tps[-1]  # 总正样本数 = 最后累计的 TP
    total_neg = fps[-1]  # 总负样本数 = 最后累计的 FP

    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)
    return fpr, tpr, thresholds


def precision_recall_curve(
    y_true: Sequence[int], y_score: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 PR 曲线的 (precision, recall, thresholds)。

    PR 曲线在**正样本稀少**（类别极不平衡）时比 ROC 更能反映真实性能，
    因为它完全不看 TN（真负），而不平衡数据里 TN 巨多会让 ROC 看着很美。
    """
    yt = _as_1d_float(y_true)
    ys = _as_1d_float(y_score)
    _check_same_length(yt, ys)

    fps, tps, thresholds = _binary_clf_curve(yt, ys)

    # precision = TP / (TP + FP)；用 errstate 静默 0/0（在没有任何预测正时）
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0.0

    total_pos = tps[-1]
    recall = tps / total_pos if total_pos > 0 else np.zeros_like(tps)

    # sklearn 惯例：在末尾补 (precision=1, recall=0)，让曲线以 recall=0 收尾
    precision = np.r_[precision, 1.0]
    recall = np.r_[recall, 0.0]
    thresholds = np.r_[thresholds, thresholds[-1]]
    return precision, recall, thresholds


def _auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """用梯形法则 (trapezoidal rule) 求曲线下面积 AUC。

    梯形法则：把曲线离散成一串点，相邻两点连线，
    每一小段是一个梯形，面积 = (上底 + 下底) / 2 * 高。
    这里"高"是 x 的增量 dx，"上下底"是相邻两个 y。

    注意：x 必须是单调的；若整体递减我们先翻转成递增再算。
    """
    order = np.argsort(x)  # 保证 x 单调递增
    x_sorted = x[order]
    y_sorted = y[order]
    dx = np.diff(x_sorted)
    # 相邻 y 的平均 * dx，再求和
    area = np.sum((y_sorted[1:] + y_sorted[:-1]) / 2.0 * dx)
    return float(area)


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC 曲线下面积 (Area Under ROC Curve)。

    直观含义：随机取一个正样本和一个负样本，
    模型给正样本打的分高于负样本的概率。
    0.5 = 瞎猜；1.0 = 完美；<0.5 = 比瞎猜还差（可能标签反了）。
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(np.unique(fpr)) < 2:
        # 全是同一类样本，AUC 无定义，约定返回 0.5
        return 0.5
    return _auc_trapezoid(fpr, tpr)


def pr_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """PR 曲线下面积 (Area Under PR Curve)，也叫 Average Precision 的近似。

    这里用梯形法则对 (recall, precision) 积分。
    ⚠️ 注意：严格的 Average Precision (AP) 用的是阶梯求和而非梯形，
    两者在采样密时几乎相等；本函数返回梯形积分值，另有 average_precision 给 AP。
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return _auc_trapezoid(recall, precision)


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """平均精确率 (Average Precision, AP) —— PR 曲线的阶梯求和版本。

    AP = Σ_n (R_n - R_{n-1}) * P_n
    即：每当召回率往前跨一步 ΔR，就用"那一步之后（即当前阈值点）的精确率" P_n
    去加权累加。这是信息检索里最标准的 PR-AUC 定义，比梯形法则更常见于论文。

    ⚠️ 关键实现细节：必须直接用 _binary_clf_curve 得到的**逐阈值** (tps, fps)，
    不能用 precision_recall_curve 那个补了首尾端点的版本——否则端点会污染 ΔR×P 求和。
    在"完美区分"下，所有正样本排在前面，每一步 ΔR 对应的 P 都是 1，AP 恰好 = 1.0。
    """
    yt = _as_1d_float(y_true)
    ys = _as_1d_float(y_score)
    _check_same_length(yt, ys)

    fps, tps, _ = _binary_clf_curve(yt, ys)
    total_pos = tps[-1]
    if total_pos == 0:
        return 0.0

    # 各阈值点处的精确率 P_n = TP / (TP + FP)，召回率 R_n = TP / 总正数
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0.0
    recall = tps / total_pos

    # ΔR：相邻阈值点召回增量（首项相对 recall=0）
    dr = np.diff(recall, prepend=0.0)
    return float(np.sum(dr * precision))


# =============================================================================
# 2. 排序指标 (Ranking metrics)
# =============================================================================
#
# 排序场景（推荐、搜索）里，我们不止关心"预测对没对"，
# 更关心"把最相关的排在最前面没有"。以下三个指标衡量排序质量。


def _dcg(relevances: np.ndarray, k: int) -> float:
    """折损累计增益 (Discounted Cumulative Gain) 前 k 项。

    DCG@k = Σ_{i=1..k}  rel_i / log2(i + 1)

    直觉：排在越靠后的位置，即使相关也要"打折"，
    因为用户很少翻到后面。分母 log2(i+1) 就是位置折损因子。
    第 1 位折损 log2(2)=1（不打折），第 2 位除以 log2(3)≈1.585，越后越狠。
    """
    rel_k = relevances[:k]
    # 位置从 1 开始：i = 1,2,...,len。折损 = log2(i+1)
    discounts = np.log2(np.arange(2, len(rel_k) + 2))
    return float(np.sum(rel_k / discounts))


def ndcg_at_k(
    y_true_relevance: Sequence[float], y_score: Sequence[float], k: int
) -> float:
    """归一化折损累计增益 (Normalized DCG) @k。

    NDCG@k = DCG@k / IDCG@k

    IDCG 是"理想排序"（把相关度从高到低排）的 DCG，作为归一化上界，
    这样不同 query（相关文档数量不同）之间可以公平比较，结果落在 [0, 1]。

    参数
    ----
    y_true_relevance : 每个 item 的真实相关度（可 0/1，也可分级 0,1,2,3...）
    y_score          : 模型给每个 item 的打分（越大越靠前）
    k                : 只看前 k 个位置
    """
    rel = _as_1d_float(y_true_relevance)
    score = _as_1d_float(y_score)
    _check_same_length(rel, score)

    # 按模型打分从高到低排 → 得到"模型排序下"的相关度序列
    ranking = np.argsort(score, kind="mergesort")[::-1]
    rel_by_model = rel[ranking]
    dcg = _dcg(rel_by_model, k)

    # 理想排序：把真实相关度自己从大到小排
    ideal = np.sort(rel)[::-1]
    idcg = _dcg(ideal, k)

    return _safe_divide(dcg, idcg)


def average_precision_at_k(
    y_true_relevance: Sequence[int], y_score: Sequence[float], k: int
) -> float:
    """单个 query 的 AP@k（Average Precision）—— MAP 的组成单元。

    这里相关度按 0/1（相关/不相关）处理。
    AP@k = (Σ_{i=1..k}  P(i) * rel_i) / min(相关文档总数, k)
      其中 P(i) 是"到第 i 个位置为止的精确率"，rel_i ∈ {0,1}。

    直觉：每命中一个相关文档，就记一下"到这里为止命中率多高"，最后平均。
    命中越靠前，P(i) 越高，AP 越高。
    """
    rel = _as_1d_float(y_true_relevance)
    score = _as_1d_float(y_score)
    _check_same_length(rel, score)

    ranking = np.argsort(score, kind="mergesort")[::-1][:k]
    rel_ranked = (rel[ranking] > 0).astype(np.float64)  # 二值化为是否相关

    if rel_ranked.sum() == 0:
        return 0.0

    # 累计命中数 / 位置 = 每个位置的精确率
    cum_hits = np.cumsum(rel_ranked)
    positions = np.arange(1, len(rel_ranked) + 1)
    precision_at_i = cum_hits / positions

    # 只在"命中位置"处累加 precision
    ap = np.sum(precision_at_i * rel_ranked)
    # 分母：相关文档总数（全体，不只前 k）与 k 取小
    n_relevant = min(int((rel > 0).sum()), k)
    return _safe_divide(ap, n_relevant)


def mean_average_precision(
    relevances: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    k: int,
) -> float:
    """平均精度均值 MAP@k：对多个 query 的 AP@k 求平均。

    参数
    ----
    relevances : 列表的列表，relevances[q] 是第 q 个 query 各 item 的相关度
    scores     : 列表的列表，scores[q] 是第 q 个 query 各 item 的模型打分
    """
    if len(relevances) != len(scores):
        raise ValueError("relevances 与 scores 的 query 数不一致")
    if len(relevances) == 0:
        return 0.0
    aps = [
        average_precision_at_k(r, s, k) for r, s in zip(relevances, scores)
    ]
    return float(np.mean(aps))


def mean_reciprocal_rank(
    relevances: Sequence[Sequence[int]], scores: Sequence[Sequence[float]]
) -> float:
    """平均倒数排名 MRR (Mean Reciprocal Rank)。

    对每个 query，找到"第一个相关文档"的排名 rank，倒数记为 1/rank；
    没有相关文档则记 0。最后对所有 query 求平均。

    MRR 特别适合"只关心第一个正确答案在哪"的场景，
    比如问答、导航搜索——用户往往只想要第一个对的结果。
    """
    if len(relevances) != len(scores):
        raise ValueError("relevances 与 scores 的 query 数不一致")
    if len(relevances) == 0:
        return 0.0

    reciprocal_ranks: List[float] = []
    for rel, score in zip(relevances, scores):
        rel_arr = _as_1d_float(rel)
        score_arr = _as_1d_float(score)
        ranking = np.argsort(score_arr, kind="mergesort")[::-1]
        rel_ranked = rel_arr[ranking] > 0
        # np.argmax 返回第一个 True 的下标（0-based），+1 变成排名
        if rel_ranked.any():
            first_hit = int(np.argmax(rel_ranked)) + 1
            reciprocal_ranks.append(1.0 / first_hit)
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks))


# =============================================================================
# 3. 校准指标 (Calibration metrics)
# =============================================================================
#
# 校准关心的是：模型说"我有 80% 把握"时，是不是真的 80% 的时候对了？
# 一个 AUC 很高的模型，概率也可能完全没校准（比如全部预测挤在 0.9 附近）。


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Brier 分数 = 预测概率与真实标签的均方误差 (MSE)。

    Brier = (1/N) Σ (p_i - y_i)^2

    越小越好（0 最好）。它同时惩罚"过度自信"和"不够自信"，
    是一个既看区分度又看校准度的综合分数（可分解为校准项 + 精化项）。
    """
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_prob)
    _check_same_length(yt, yp)
    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """期望校准误差 ECE (Expected Calibration Error)。

    做法：
      1. 把预测概率 [0,1] 均匀切成 n_bins 个桶。
      2. 每个桶里算两个数：
           - 平均预测置信度 conf = 桶内预测概率均值
           - 实际准确率 acc = 桶内真实为正的比例
      3. ECE = Σ_桶  (桶样本占比) * |acc - conf|

    直觉：如果模型完美校准，每个桶里"说多少概率"就应"真有多少比例是正"，
    即 acc ≈ conf，各桶差值为 0，ECE = 0。差得越多 ECE 越大。
    """
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_prob)
    _check_same_length(yt, yp)

    n = len(yt)
    if n == 0:
        return 0.0

    # 桶的边界：0, 1/n_bins, 2/n_bins, ..., 1
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # 最后一个桶闭区间包含右端点 1.0，其余左闭右开
        if i == n_bins - 1:
            in_bin = (yp >= lo) & (yp <= hi)
        else:
            in_bin = (yp >= lo) & (yp < hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        conf = float(np.mean(yp[in_bin]))  # 桶内平均置信度
        acc = float(np.mean(yt[in_bin]))   # 桶内真实正比例
        ece += (count / n) * abs(acc - conf)
    return ece


def reliability_curve(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """可靠性曲线 (Reliability Diagram) 数据：返回每个桶的 (conf, acc, count)。

    画出来横轴是平均置信度 conf、纵轴是实际准确率 acc；
    完美校准的模型所有点都落在对角线 y = x 上。
    """
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_prob)
    _check_same_length(yt, yp)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (yp >= lo) & (yp <= hi)
        else:
            in_bin = (yp >= lo) & (yp < hi)
        count = int(np.sum(in_bin))
        if count == 0:
            confs.append(np.nan)
            accs.append(np.nan)
            counts.append(0)
        else:
            confs.append(float(np.mean(yp[in_bin])))
            accs.append(float(np.mean(yt[in_bin])))
            counts.append(count)
    return np.array(confs), np.array(accs), np.array(counts)


# =============================================================================
# 4. 切片分析 (Slice / subgroup analysis)
# =============================================================================
#
# 整体指标好，不代表每个子群都好。切片分析把样本按某个维度（如国家、
# 设备、年龄段）分组，在每组内单独算指标，暴露"平均值掩盖的短板"。


def slice_metric(
    y_true: Sequence,
    y_score_or_pred: Sequence,
    groups: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, float]:
    """按分组维度切片，在每个组内计算给定指标。

    参数
    ----
    y_true          : 真实标签
    y_score_or_pred : 预测分数或预测标签（取决于 metric_fn 需要什么）
    groups          : 与样本一一对应的分组键（如 ["US","CN","US",...]）
    metric_fn       : 一个函数 f(y_true_sub, y_pred_sub) -> float

    返回
    ----
    dict：{组名: 该组指标值}，外加一个 "__overall__" 表示全体指标。

    ⚠️ 面试常问：为什么不能只看整体指标？
        因为整体是各组的加权平均，可能"多数组很好 + 少数关键组很差"
        被平均掉。切片分析是发现公平性/长尾问题的第一道防线。
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_score_or_pred)
    grp = np.asarray(groups)
    _check_same_length(yt, yp, grp)

    result: Dict[str, float] = {}
    for g in np.unique(grp):
        mask = grp == g
        result[str(g)] = float(metric_fn(yt[mask], yp[mask]))
    # 附带整体指标做对照
    result["__overall__"] = float(metric_fn(yt, yp))
    return result


def slice_report(
    y_true: Sequence,
    y_score_or_pred: Sequence,
    groups: Sequence,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, Dict[str, float]]:
    """多指标切片报告：对每个指标都做一次切片，汇总成嵌套字典。

    返回结构：{ 指标名: { 组名: 值, ..., "__overall__": 值 } }
    方便直接喂给 pandas / 打印成表。
    """
    report: Dict[str, Dict[str, float]] = {}
    for name, fn in metrics.items():
        report[name] = slice_metric(y_true, y_score_or_pred, groups, fn)
    return report
