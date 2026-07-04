# -*- coding: utf-8 -*-
"""
跨模态检索核心库 (cross-modal retrieval core)
================================================

给一批「配对好」的 (图嵌入 image embedding, 文嵌入 text embedding),
本模块实现两个方向的 top-k 检索并计算标准指标:

    - 文 -> 图 (text-to-image, t2i): 用一条文本 query 去图库里捞最像的图
    - 图 -> 文 (image-to-text, i2t): 用一张图 query 去文库里捞最像的文

评价指标:
    - recall@k   : 正确项落在 top-k 里的比例 (召回率, 越大越好)
    - median rank: 正确项排名的中位数 (越小越好)

核心设计选择 (贯穿全库):
    - 相似度用「点积」;当向量做了 L2 归一化后,点积 == 余弦相似度 cosine。
      这正是 CLIP / BLIP 等模型对齐图文空间的标准做法。
    - 温度 temperature 只在把相似度转成概率 (softmax) 时才有意义;
      它不改变排序,因此对 recall@k / median rank 没有影响。
      但它决定检索置信度的「软硬」,本库单独提供函数演示。

全部纯 numpy 实现,CPU 秒级,不依赖 GPU / 网络 / 预训练模型。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "l2_normalize",
    "similarity_matrix",
    "topk_indices",
    "ranks_of_ground_truth",
    "recall_at_k",
    "median_rank",
    "retrieval_report",
    "softmax_retrieval_probs",
    "add_noise",
]


# --------------------------------------------------------------------------- #
# 1. 归一化                                                                     #
# --------------------------------------------------------------------------- #
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对每一行做 L2 归一化,使其长度(范数)变成 1。

    为什么要归一化?
        点积 a·b = |a| |b| cos(theta)。若不归一化,某个向量「特别长」
        就会天然获得更大的点积,从而在检索里不公平地排到前面 ——
        这叫「hubness / 长度偏置」问题。归一化后 |a|=|b|=1,
        点积就纯粹等于夹角余弦 cos(theta),只反映「方向是否一致」,
        这才是语义相似度真正想要的东西。

    参数
        x   : 形状 (N, D) 的矩阵,每行一个向量。
        eps : 防止除以 0 的下限(全 0 向量会得到 0 向量而非 NaN)。

    返回
        (N, D) 归一化后的矩阵。
    """
    x = np.asarray(x, dtype=np.float64)
    # 沿最后一维求 L2 范数,keepdims 让它能广播回 (N, 1)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    # np.maximum(norm, eps): 把 0 抬成 eps,避免 0/0 = NaN
    return x / np.maximum(norm, eps)


# --------------------------------------------------------------------------- #
# 2. 相似度矩阵                                                                  #
# --------------------------------------------------------------------------- #
def similarity_matrix(
    query: np.ndarray,
    gallery: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """计算 query 与 gallery 之间两两相似度 (点积 / 余弦)。

    参数
        query     : (Nq, D) 查询向量,每行一个 query。
        gallery   : (Ng, D) 被检索库,每行一个候选。
        normalize : 是否先做 L2 归一化。True 时点积即余弦相似度。

    返回
        (Nq, Ng) 矩阵,S[i, j] = query_i 与 gallery_j 的相似度。

    第一性原理: 检索 = 排序问题。我们不需要「距离」的绝对值,
    只需要一个「越大越像」的分数把候选排序即可。点积/余弦满足此性质
    且能用一次矩阵乘法 (query @ gallery.T) 批量算完,极快。
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if normalize:
        q = l2_normalize(q)
        g = l2_normalize(g)
    # (Nq, D) @ (D, Ng) = (Nq, Ng)
    return q @ g.T


# --------------------------------------------------------------------------- #
# 3. top-k 检索                                                                 #
# --------------------------------------------------------------------------- #
def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    """对每个 query,返回相似度最高的 k 个 gallery 下标 (按相似度降序)。

    参数
        sim : (Nq, Ng) 相似度矩阵。
        k   : 取前 k 个;自动裁剪到 <= Ng。

    返回
        (Nq, k) 的整型下标矩阵,每行是该 query 的 top-k 候选下标,
        第 0 列最像。

    实现细节: 用 np.argpartition 先 O(N) 把前 k 大的粗略拎出(不保证内部有序),
    再对这 k 个做精确排序。比全排序 O(N log N) 更省,是检索工程常用技巧。
    """
    sim = np.asarray(sim, dtype=np.float64)
    nq, ng = sim.shape
    k = int(min(k, ng))
    if k <= 0:
        return np.empty((nq, 0), dtype=np.int64)

    # argpartition: 把第 (ng-k) 位作为枢轴,右侧是最大的 k 个(无序)。
    # 取负号是因为 partition 默认找最小,我们要最大。
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]  # (Nq, k) 无序
    # 取出这 k 个的相似度,再在行内做降序精排。
    part_scores = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(-part_scores, axis=1)  # 行内降序
    return np.take_along_axis(part, order, axis=1)


# --------------------------------------------------------------------------- #
# 4. ground-truth 排名                                                          #
# --------------------------------------------------------------------------- #
def ranks_of_ground_truth(
    sim: np.ndarray,
    gt: np.ndarray | None = None,
) -> np.ndarray:
    """计算每个 query 的「正确答案」排在第几名 (1-based)。

    参数
        sim : (Nq, Ng) 相似度矩阵。
        gt  : (Nq,) 每个 query 对应的正确 gallery 下标。
              None 时默认第 i 个 query 的正确答案就是第 i 个 gallery
              (即对角线配对,自检索场景)。

    返回
        (Nq,) 数组,第 i 个元素 = query_i 的正确答案的排名,rank=1 表示最像。

    排名定义 (关键!):
        rank = 1 + (严格比正确项分数更高的候选个数)。
        「严格更高」用 > 而非 >=,这样并列时不会把自己也算进去,
        正确项至少能拿到 rank 1。这是 recall / mAP 里常见的稳健写法。
    """
    sim = np.asarray(sim, dtype=np.float64)
    nq, ng = sim.shape
    if gt is None:
        gt = np.arange(nq)
    gt = np.asarray(gt, dtype=np.int64)

    # 取出每个 query 对正确项的相似度分数,形状 (Nq, 1) 方便广播比较。
    gt_scores = np.take_along_axis(sim, gt[:, None], axis=1)  # (Nq, 1)
    # 逐行统计「比正确项严格更高」的候选数,+1 即为排名。
    higher = (sim > gt_scores).sum(axis=1)  # (Nq,)
    return higher + 1


# --------------------------------------------------------------------------- #
# 5. 指标: recall@k 与 median rank                                              #
# --------------------------------------------------------------------------- #
def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """recall@k = (正确项排名 <= k 的 query 比例)。

    直觉: 如果我给用户展示 top-k 条结果,有多大比例的 query
    能在这 k 条里看到「真正想要的那条」。k 越大 recall 越高(或持平),
    因此 recall@k 关于 k 单调不减 —— 这是本库一条 pytest 断言的由来。
    """
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        return 0.0
    return float((ranks <= k).mean())


def median_rank(ranks: np.ndarray) -> float:
    """median rank = 正确项排名的中位数。越小说明检索越准。

    为什么用中位数而非平均?
        排名分布是重尾的 —— 少数「怎么都检索不对」的难样本
        rank 可能高达几百,会把平均值拉爆。中位数对这种离群点稳健,
        更能反映「一半以上的 query 好到什么程度」。这是检索论文
        (如 CLIP、BLIP 的 R@1/R@5/R@10 + MedR) 的标准报告项。
    """
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        return float("nan")
    return float(np.median(ranks))


def retrieval_report(
    query: np.ndarray,
    gallery: np.ndarray,
    gt: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
    normalize: bool = True,
) -> dict:
    """一站式: 从原始嵌入直接算出 recall@k(多个 k) 与 median rank。

    返回一个字典,例如:
        {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0,
         "median_rank": 1.0, "ranks": array([...])}

    这是把上面几块拼起来的「门面函数」,run_demo 与测试都直接用它。
    """
    sim = similarity_matrix(query, gallery, normalize=normalize)
    ranks = ranks_of_ground_truth(sim, gt)
    report: dict = {}
    for k in ks:
        report[f"recall@{k}"] = recall_at_k(ranks, k)
    report["median_rank"] = median_rank(ranks)
    report["ranks"] = ranks
    return report


# --------------------------------------------------------------------------- #
# 6. 温度与 softmax 检索概率                                                     #
# --------------------------------------------------------------------------- #
def softmax_retrieval_probs(
    sim: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """把相似度矩阵按行转成检索概率分布 (softmax over gallery)。

    公式:  p_ij = exp(s_ij / T) / sum_j' exp(s_ij' / T)

    温度 T 的作用 (第一性原理):
        - T -> 0    : 分布趋近 one-hot,最像的那条概率 ~1,极度自信(锐化)。
        - T -> +inf : 分布趋近均匀,谁都差不多(平滑)。
        - CLIP 里学一个可训练的 logit_scale = 1/T,初始约 0.07 的温度,
          用来把余弦相似度([-1,1] 太挤)拉开、让对比损失更好优化。

    关键结论: 温度是「单调递增」变换 s -> s/T,不改变每行内部的大小顺序,
        因此不改变 top-k 排序 => recall@k 与 median rank 与 T 无关!
        它只改变「概率有多尖」。本库用它演示置信度,而非改变命中率。

    参数
        sim         : (Nq, Ng) 相似度矩阵。
        temperature : 温度 T > 0。

    返回
        (Nq, Ng) 概率矩阵,每行和为 1。
    """
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")
    sim = np.asarray(sim, dtype=np.float64)
    logits = sim / temperature
    # 减去每行最大值再取 exp,是数值稳定的标准 softmax 写法,防止 exp 溢出。
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 7. 加噪工具 (给测试与 demo 制造「检索变难」的场景)                              #
# --------------------------------------------------------------------------- #
def add_noise(
    x: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """给嵌入加高斯噪声,模拟「图文对齐没那么完美」。

    sigma 越大,图/文嵌入越对不上,检索越难 => recall 下降、median rank 上升。
    这正是测试里「加噪后 recall 单调下降」断言所依赖的机制。

    参数
        x     : (N, D) 原始嵌入。
        sigma : 噪声标准差 >= 0。0 表示不加噪。
        rng   : numpy 随机数生成器,传入可复现。
    """
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0:
        return x.copy()
    if rng is None:
        rng = np.random.default_rng()
    return x + rng.normal(0.0, sigma, size=x.shape)
