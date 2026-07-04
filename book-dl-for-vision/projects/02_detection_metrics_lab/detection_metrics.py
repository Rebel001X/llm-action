# -*- coding: utf-8 -*-
"""
detection_metrics.py
====================
目标检测评价指标 —— 纯 numpy 实现。

本模块只依赖 numpy，实现目标检测里最核心的一套评价与后处理工具：

    1. IoU (Intersection over Union)         —— 交并比，一切检测指标的基石
    2. NMS (Non-Maximum Suppression)          —— 非极大抑制，去掉重复框
    3. match_detections                        —— 把预测框和真值框做贪心匹配 (TP/FP)
    4. precision_recall_curve                  —— 由匹配结果算 PR 曲线
    5. average_precision                        —— 单类 AP（11 点插值 / 全点插值 COCO 风格）
    6. mean_average_precision                   —— 多类 mAP

设计原则（和书 Deep Learning for Vision Systems 第 7 章一致）：
    - 框统一用 [x1, y1, x2, y2] 表示（左上角、右下角，绝对像素坐标）。
    - x2 > x1、y2 > y1，即右下角坐标严格大于左上角。
    - 所有函数对 numpy 数组做向量化，尽量不写 Python 循环里的重活。

作者注释全部为中文，术语中英并列，便于零基础到进阶阅读。
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# 1. IoU —— 交并比
# =====================================================================
def box_area(boxes: np.ndarray) -> np.ndarray:
    """计算一批框的面积。

    参数
    ----
    boxes : (N, 4) 数组，每行 [x1, y1, x2, y2]

    返回
    ----
    (N,) 数组，每个框的面积。

    说明：面积 = 宽 * 高 = (x2 - x1) * (y2 - y1)。
    这里用 np.clip(..., 0, None) 把负的宽/高裁成 0，
    防止“退化框”（x2<x1）算出负面积污染后续 IoU。
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)  # 宽，不能为负
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)  # 高，不能为负
    return w * h


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算两组框两两之间的 IoU，返回 (Na, Nb) 矩阵。

    IoU (Intersection over Union) = 交集面积 / 并集面积。
    这是衡量“两个框重合程度”的标准指标，取值 [0, 1]，越大越重合。

    向量化推导（核心！面试常问）：
        设 A 有 Na 个框，B 有 Nb 个框。
        交集矩形的左上角 = 两框左上角的“逐元素最大值” (np.maximum)
        交集矩形的右下角 = 两框右下角的“逐元素最小值” (np.minimum)
        交集宽 = max(0, 右下角x - 左上角x)   （不重叠时为负 -> clip 到 0）
        交集面积 = 交集宽 * 交集高
        并集面积 = 面积A + 面积B - 交集面积
        IoU = 交集 / (并集 + eps)             （eps 防除零）

    利用 numpy 广播：boxes_a[:, None, :] 形状 (Na,1,4)，
    boxes_b[None, :, :] 形状 (1,Nb,4)，两者运算自动广播成 (Na,Nb,4)。
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        # 任意一组为空时，返回一个形状正确的空矩阵，方便上层统一处理
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    area_a = box_area(boxes_a)  # (Na,)
    area_b = box_area(boxes_b)  # (Nb,)

    # 交集矩形的左上角坐标：两框左上角取“更靠右下”的那个（即更大）
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (Na,Nb,2)
    # 交集矩形的右下角坐标：两框右下角取“更靠左上”的那个（即更小）
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # (Na,Nb,2)

    # 交集的宽高，负数（不重叠）裁成 0
    wh = np.clip(rb - lt, 0, None)                                # (Na,Nb,2)
    inter = wh[:, :, 0] * wh[:, :, 1]                             # (Na,Nb)

    # 并集 = A面积 + B面积 - 交集（广播成 (Na,Nb)）
    union = area_a[:, None] + area_b[None, :] - inter
    eps = np.finfo(np.float64).eps
    return inter / (union + eps)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两个单框的 IoU，返回一个标量。是 iou_matrix 的便捷封装。

    box_a, box_b : 长度为 4 的一维数组 [x1,y1,x2,y2]
    """
    return float(iou_matrix(np.asarray(box_a).reshape(1, 4),
                            np.asarray(box_b).reshape(1, 4))[0, 0])


# =====================================================================
# 2. NMS —— 非极大抑制
# =====================================================================
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """非极大抑制 (Non-Maximum Suppression)。

    问题背景：检测器（如 Faster R-CNN / YOLO）会对同一个物体吐出很多高度重叠、
    分数相近的框。NMS 的作用是“每个物体只保留一个最自信的框”。

    贪心算法（greedy）步骤：
        1. 按分数从高到低排序所有框。
        2. 取出当前分数最高的框，加入保留集合 keep。
        3. 计算这个框和剩余所有框的 IoU，
           把 IoU > 阈值 的框视为“同一物体的重复”，删掉。
        4. 在剩下的框里重复 2~3，直到没有框可选。

    参数
    ----
    boxes         : (N,4) 数组
    scores        : (N,) 数组，每个框的置信度
    iou_threshold : IoU 大于它就认为重复而抑制；越小抑制越狠

    返回
    ----
    keep : 一维 int 数组，保留下来的框在原数组中的“索引”，按分数从高到低。

    复杂度：最坏 O(N^2)。工业界对超大 N 会用分块或 GPU 版；这里教学用清晰版。
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    # argsort 默认升序，[::-1] 反转成降序 -> 分数从高到低的索引
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]          # 当前分数最高的框的索引
        keep.append(int(i))
        if order.size == 1:
            break
        # 计算 i 与其余所有框（order[1:]）的 IoU
        rest = order[1:]
        ious = iou_matrix(boxes[i][None, :], boxes[rest])[0]  # (len(rest),)
        # 保留 IoU <= 阈值 的框（即“不算重复”的），继续参与下一轮
        remain_mask = ious <= iou_threshold
        order = rest[remain_mask]

    return np.array(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray,
                scores: np.ndarray,
                labels: np.ndarray,
                iou_threshold: float = 0.5) -> np.ndarray:
    """按类别分别做 NMS（不同类别的框互不抑制）。

    工业做法（torchvision.ops.batched_nms 的思路）：给每个类别一个巨大的
    坐标偏移，让不同类的框在坐标上彻底错开，再做一次全局 NMS 即可。
    这里为清晰起见直接按类别分组循环。

    返回：保留索引（相对原数组），未排序合并前的顺序不保证全局有序。
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]           # 该类别的所有框索引
        local_keep = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[local_keep])         # 映射回全局索引
    if not keep_all:
        return np.empty((0,), dtype=np.int64)
    keep = np.concatenate(keep_all)
    # 合并后再按分数整体降序，输出更直观
    return keep[scores[keep].argsort()[::-1]]


# =====================================================================
# 3. 预测框与真值框匹配 —— 判定 TP / FP
# =====================================================================
def match_detections(pred_boxes: np.ndarray,
                     pred_scores: np.ndarray,
                     gt_boxes: np.ndarray,
                     iou_threshold: float = 0.5):
    """把一张图上单个类别的预测框与真值框匹配，判定每个预测是 TP 还是 FP。

    规则（PASCAL VOC / COCO 通用）：
        - 预测框按分数从高到低处理。
        - 每个预测框找和它 IoU 最大的、尚未被占用的真值框；
          若最大 IoU >= 阈值，则记为 TP (True Positive)，并占用该真值框；
        - 否则记为 FP (False Positive)。
        - 一个真值框最多只能被匹配一次（后来者即使 IoU 够也算 FP，即“重复检测”）。
        - 没被任何预测匹配上的真值框 = FN (False Negative)（漏检）。

    参数
    ----
    pred_boxes  : (P,4)
    pred_scores : (P,)
    gt_boxes    : (G,4)
    iou_threshold : 判为 TP 的 IoU 门槛

    返回
    ----
    tp     : (P,) 0/1 数组，按“分数降序”排列后每个预测是否 TP
    fp     : (P,) 0/1 数组，= 1 - tp
    scores : (P,) 对应上面排序后的分数（用于后续按分数扫阈值）
    n_gt   : int，真值框总数（算 recall 分母用）

    ⚠️ 注意 tp/fp 是“按分数降序”重排后的，和输入顺序不同。
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)

    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]

    order = pred_scores.argsort()[::-1]          # 分数从高到低
    sorted_scores = pred_scores[order]
    tp = np.zeros(n_pred, dtype=np.float64)
    fp = np.zeros(n_pred, dtype=np.float64)

    if n_gt == 0:
        # 没有真值 -> 所有预测都是误检 FP
        fp[:] = 1.0
        return tp, fp, sorted_scores, 0

    if n_pred == 0:
        return tp, fp, sorted_scores, n_gt

    gt_matched = np.zeros(n_gt, dtype=bool)      # 每个真值框是否已被占用
    ious = iou_matrix(pred_boxes[order], gt_boxes)  # (P,G)，已按分数序

    for p in range(n_pred):
        row = ious[p]                             # 该预测与所有真值的 IoU
        best_gt = int(np.argmax(row))             # IoU 最大的真值框
        best_iou = row[best_gt]
        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            tp[p] = 1.0                           # 命中且真值未被占用 -> TP
            gt_matched[best_gt] = True
        else:
            fp[p] = 1.0                           # 否则 -> FP（漏配或重复检测）
    return tp, fp, sorted_scores, n_gt


# =====================================================================
# 4. Precision-Recall 曲线
# =====================================================================
def precision_recall_curve(tp: np.ndarray, fp: np.ndarray, n_gt: int):
    """由（按分数降序的）TP/FP 序列算 PR 曲线。

    核心思想：想象把“置信度阈值”从高往低慢慢降。阈值越低，接受的预测越多。
    对已按分数降序排好的 TP/FP 做“累积和”(cumsum)，第 k 个元素就代表
    “取前 k 个最自信预测”时的累计 TP、FP：

        precision = 累计TP / (累计TP + 累计FP)   —— 查准率：预测里对的比例
        recall    = 累计TP / n_gt                 —— 查全率：真值被找回的比例

    返回
    ----
    precision : (P,) 数组
    recall    : (P,) 数组
    （若 P==0，返回长度为 0 的数组；上层 AP 会正确处理为 0）
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    eps = np.finfo(np.float64).eps

    recall = tp_cum / (n_gt + eps) if n_gt > 0 else np.zeros_like(tp_cum)
    precision = tp_cum / (tp_cum + fp_cum + eps)
    return precision, recall


# =====================================================================
# 5. AP —— Average Precision（两种插值口径）
# =====================================================================
def average_precision(precision: np.ndarray,
                      recall: np.ndarray,
                      method: str = "all_points") -> float:
    """由 PR 曲线计算 AP（PR 曲线下面积的近似）。

    method:
        - "11_points"  : PASCAL VOC 2007 口径。在 recall = 0,0.1,...,1.0 这 11
                          个点上取“该 recall 之后的最大 precision”，求平均。
        - "all_points" : PASCAL VOC 2010+/COCO 口径。先把 precision 变成
                          单调不增的“包络线”，再算阶梯下的精确面积。

    两种口径的共同关键动作：precision 的“右侧最大值包络”
        p_interp(r) = max_{r' >= r} precision(r')
    直觉：允许你“事后诸葛”——在某召回率处，可以用之后出现过的更高 precision 顶上去，
    这样曲线变成单调下降的阶梯，消除 PR 曲线本身的锯齿抖动。

    返回：AP 标量，范围 [0,1]。
    """
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)

    if precision.size == 0:
        return 0.0

    if method == "11_points":
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):       # 0,0.1,...,1.0
            mask = recall >= t
            p = precision[mask].max() if mask.any() else 0.0
            ap += p / 11.0
        return float(ap)

    elif method == "all_points":
        # 在两端补哨兵点：recall 前补 0、后补 1；precision 前补 0、后补 0
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        # 从右往左取累计最大值 -> precision 单调不增包络
        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        # 只在 recall 发生变化的位置累加矩形面积 sum((r_{i+1}-r_i)*p_{i+1})
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    else:
        raise ValueError(f"未知 method={method!r}，只支持 '11_points' 或 'all_points'")


def compute_ap(pred_boxes, pred_scores, gt_boxes,
               iou_threshold: float = 0.5,
               method: str = "all_points") -> float:
    """端到端算单类 AP：匹配 -> PR 曲线 -> AP。是最常用的高层入口。"""
    tp, fp, _, n_gt = match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold)
    if n_gt == 0:
        # 没有真值时 AP 约定为 0（该类不参与 mAP 平均由上层决定）
        return 0.0
    precision, recall = precision_recall_curve(tp, fp, n_gt)
    return average_precision(precision, recall, method=method)


# =====================================================================
# 6. mAP —— 多类平均
# =====================================================================
def mean_average_precision(detections: dict,
                           ground_truths: dict,
                           iou_threshold: float = 0.5,
                           method: str = "all_points"):
    """多类 mean Average Precision。

    参数
    ----
    detections : dict[class_id] -> (boxes(K,4), scores(K,))
    ground_truths : dict[class_id] -> boxes(M,4)
        注意：只在两个 dict 的类别并集上计算；某类没有真值则跳过（不计入平均）。
    iou_threshold, method : 传给单类 AP。

    返回
    ----
    mAP     : float，各类 AP 的算术平均
    per_class_ap : dict[class_id] -> AP
    """
    classes = set(detections.keys()) | set(ground_truths.keys())
    per_class_ap = {}
    for c in sorted(classes):
        gt = ground_truths.get(c, np.zeros((0, 4)))
        if np.asarray(gt).reshape(-1, 4).shape[0] == 0:
            # 该类无真值 -> 不纳入 mAP 平均（COCO 亦是如此处理）
            continue
        boxes, scores = detections.get(c, (np.zeros((0, 4)), np.zeros((0,))))
        per_class_ap[c] = compute_ap(boxes, scores, gt, iou_threshold, method)

    if not per_class_ap:
        return 0.0, {}
    mAP = float(np.mean(list(per_class_ap.values())))
    return mAP, per_class_ap


__all__ = [
    "box_area", "iou_matrix", "iou",
    "nms", "batched_nms",
    "match_detections", "precision_recall_curve",
    "average_precision", "compute_ap", "mean_average_precision",
]
