# -*- coding: utf-8 -*-
"""
test_detection_metrics.py
=========================
对 detection_metrics.py 的单元测试。覆盖：
    - IoU：已知框对拍（手算精确值）
    - NMS：去重正确 + 阈值敏感 + 分类别
    - 匹配/PR/AP/mAP：玩具检测结果对拍手算
    - 边界：无框 / 全重叠 / 无真值

运行： python -m pytest -q
"""

import os
import sys

import numpy as np
import pytest

# 让测试无论从哪运行都能 import 到上一级目录的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection_metrics import (  # noqa: E402
    box_area, iou, iou_matrix,
    nms, batched_nms,
    match_detections, precision_recall_curve,
    average_precision, compute_ap, mean_average_precision,
)


# ---------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------
def test_iou_identical_boxes_is_one():
    """两个完全相同的框 IoU = 1。"""
    b = [0, 0, 10, 10]
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    """完全不相交的框 IoU = 0。"""
    a = [0, 0, 10, 10]
    b = [20, 20, 30, 30]
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_touching_edge_is_zero():
    """仅边缘相接（交集面积为 0）IoU = 0。"""
    a = [0, 0, 10, 10]
    b = [10, 0, 20, 10]
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_half_overlap_known_value():
    """手算对拍：
    A=[0,0,10,10] 面积100；B=[5,0,15,10] 面积100。
    交集 x:[5,10] y:[0,10] -> 5*10=50。并集=100+100-50=150。
    IoU=50/150=1/3。
    """
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]
    assert iou(a, b) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_iou_contained_box_known_value():
    """一个框完全包含另一个：
    大框 [0,0,10,10]=100，小框 [0,0,5,5]=25，交集=25，并集=100。
    IoU=25/100=0.25。
    """
    big = [0, 0, 10, 10]
    small = [0, 0, 5, 5]
    assert iou(big, small) == pytest.approx(0.25)


def test_iou_matrix_shape_and_values():
    """iou_matrix 形状正确且对角为自身 IoU=1。"""
    boxes = np.array([[0, 0, 10, 10],
                      [5, 0, 15, 10],
                      [100, 100, 110, 110]], dtype=float)
    m = iou_matrix(boxes, boxes)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 1.0)
    assert m[0, 1] == pytest.approx(1.0 / 3.0)
    assert m[0, 2] == pytest.approx(0.0)


def test_box_area_clips_degenerate():
    """退化框（x2<x1）面积裁为 0，不出现负面积。"""
    boxes = np.array([[10, 10, 0, 0],      # 完全反了
                      [0, 0, 3, 4]], dtype=float)
    areas = box_area(boxes)
    assert areas[0] == 0.0
    assert areas[1] == pytest.approx(12.0)


def test_iou_matrix_empty_inputs():
    """空输入返回形状正确的空矩阵，不报错。"""
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


# ---------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------
def test_nms_removes_duplicates():
    """三个高度重叠的框（同一物体，两两 IoU 都 > 0.5）+ 一个远处的框。
    NMS 应保留：分数最高的重叠框(索引0) + 那个远处框(索引3)，共 2 个。
    这里用 [0,0,10,10] / [1,1,10,10] / [0,0,9,10]，与 0 号框 IoU 均 > 0.5，
    确保三者被视为同一物体的重复。
    """
    boxes = np.array([
        [0, 0, 10, 10],     # 分 0.9  <- 重叠簇里最高（保留）
        [1, 1, 10, 10],     # 分 0.8  <- 与 0 号 IoU=0.81>0.5，被抑制
        [0, 0, 9, 10],      # 分 0.7  <- 与 0 号 IoU=0.9 >0.5，被抑制
        [50, 50, 60, 60],   # 分 0.95 <- 远处独立物体，保留
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {3, 0}   # 远处框(3) 和 重叠簇最高分(0)


def test_nms_keeps_all_when_no_overlap():
    """互不重叠的框，全部保留。"""
    boxes = np.array([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [40, 40, 50, 50],
    ], dtype=float)
    scores = np.array([0.5, 0.6, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_nms_output_sorted_by_score():
    """输出索引按分数从高到低。"""
    boxes = np.array([
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [40, 40, 50, 50],
    ], dtype=float)
    scores = np.array([0.3, 0.9, 0.6])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [1, 2, 0]     # 0.9 > 0.6 > 0.3


def test_nms_threshold_sensitivity():
    """两个 IoU=1/3 的框：
    阈值 0.5 时不抑制（1/3<0.5）-> 保留 2 个；
    阈值 0.3 时抑制（1/3>0.3）-> 保留 1 个。
    """
    boxes = np.array([[0, 0, 10, 10],
                      [5, 0, 15, 10]], dtype=float)
    scores = np.array([0.9, 0.8])
    assert len(nms(boxes, scores, iou_threshold=0.5)) == 2
    assert len(nms(boxes, scores, iou_threshold=0.3)) == 1


def test_nms_empty_input():
    """空输入返回空数组。"""
    keep = nms(np.zeros((0, 4)), np.zeros((0,)), 0.5)
    assert keep.shape == (0,)


def test_nms_all_identical_boxes():
    """全部完全重叠（IoU=1）：只保留分数最高的 1 个。"""
    boxes = np.tile(np.array([0, 0, 10, 10], dtype=float), (5, 1))
    scores = np.array([0.1, 0.5, 0.9, 0.4, 0.3])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [2]           # 索引 2 分数 0.9 最高


def test_batched_nms_per_class():
    """两个高度重叠的框但属于不同类别，batched_nms 不应互相抑制。"""
    boxes = np.array([[0, 0, 10, 10],
                      [1, 1, 11, 11]], dtype=float)
    scores = np.array([0.9, 0.8])
    labels = np.array([0, 1])             # 不同类
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1]
    # 同类时则会抑制
    labels_same = np.array([0, 0])
    keep2 = batched_nms(boxes, scores, labels_same, iou_threshold=0.5)
    assert keep2.tolist() == [0]


# ---------------------------------------------------------------------
# 匹配 / PR / AP —— 玩具检测结果对拍手算
# ---------------------------------------------------------------------
def test_match_detections_basic_tp_fp():
    """2 个真值框；3 个预测：2 个命中、1 个误检。
    真值：[0,0,10,10], [20,20,30,30]
    预测（分数降序处理）：
        0.9 -> [0,0,10,10]      命中真值0 (IoU=1)  -> TP
        0.8 -> [20,20,30,30]    命中真值1 (IoU=1)  -> TP
        0.7 -> [100,100,110,110] 无匹配             -> FP
    """
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    pred = np.array([[0, 0, 10, 10],
                     [20, 20, 30, 30],
                     [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    tp, fp, sc, n_gt = match_detections(pred, scores, gt, 0.5)
    assert n_gt == 2
    assert tp.tolist() == [1, 1, 0]
    assert fp.tolist() == [0, 0, 1]
    assert sc.tolist() == [0.9, 0.8, 0.7]


def test_match_detections_double_detection_is_fp():
    """同一真值被两个预测命中 -> 第二个算 FP（重复检测）。"""
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    pred = np.array([[0, 0, 10, 10],
                     [0, 0, 10, 10]], dtype=float)   # 两个都命中同一真值
    scores = np.array([0.9, 0.8])
    tp, fp, sc, n_gt = match_detections(pred, scores, gt, 0.5)
    assert tp.tolist() == [1, 0]     # 高分的算 TP
    assert fp.tolist() == [0, 1]     # 低分重复的算 FP


def test_precision_recall_perfect_detector():
    """完美检测器：2 真值、2 预测全命中。
    cumTP=[1,2] cumFP=[0,0] -> precision=[1,1] recall=[0.5,1.0]。
    """
    tp = np.array([1.0, 1.0])
    fp = np.array([0.0, 0.0])
    p, r = precision_recall_curve(tp, fp, n_gt=2)
    assert np.allclose(p, [1.0, 1.0])
    assert np.allclose(r, [0.5, 1.0])


def test_ap_perfect_detector_is_one():
    """完美 PR 曲线（precision 恒 1，recall 到 1）AP=1（两种口径都应≈1）。"""
    p = np.array([1.0, 1.0])
    r = np.array([0.5, 1.0])
    assert average_precision(p, r, "all_points") == pytest.approx(1.0)
    assert average_precision(p, r, "11_points") == pytest.approx(1.0)


def test_ap_all_points_known_value():
    """手算对拍 all_points（VOC2010+）口径。
    序列（分数降序）TP/FP：[TP, FP, TP, TP]，n_gt=3。
        cumTP=[1,1,2,3] cumFP=[0,1,1,1]
        precision=[1/1, 1/2, 2/3, 3/4]=[1,0.5,0.6667,0.75]
        recall   =[1/3, 1/3, 2/3, 3/3]=[0.333,0.333,0.667,1.0]
    右侧最大 precision 包络（从右往左取 cummax）：
        recall=1.0 处包络 = 0.75
        recall=2/3 处包络 = max(0.6667, 0.75) = 0.75
        recall=1/3 处包络 = max(1.0, 0.5, 0.75) = 1.0
    只在 recall 跳变处累加矩形面积（补哨兵 recall 0 和 1）：
        0    -> 1/3 : 高度取 1/3 处包络 = 1.0
        1/3  -> 2/3 : 高度取 2/3 处包络 = 0.75
        2/3  -> 1   : 高度取 1   处包络 = 0.75
        AP = (1/3)*1.0 + (1/3)*0.75 + (1/3)*0.75 = 0.8333...
    """
    tp = np.array([1.0, 0.0, 1.0, 1.0])
    fp = np.array([0.0, 1.0, 0.0, 0.0])
    p, r = precision_recall_curve(tp, fp, n_gt=3)
    ap = average_precision(p, r, "all_points")
    assert ap == pytest.approx((1 / 3) * 1.0 + (1 / 3) * 0.75 + (1 / 3) * 0.75,
                               abs=1e-9)


def test_ap_11_points_known_value():
    """手算对拍 11 点口径，用上例同一 PR 曲线。
    precision=[1,0.5,0.6667,0.75]，recall=[1/3,1/3,2/3,1.0]。
    对每个 t 取 “recall>=t 的最大 precision”：
        t=0,0.1,0.2,0.3 (<=1/3)   -> recall>=t 含全部 -> max p = 1.0    (4个点)
        t=0.4,0.5,0.6   (<=2/3)   -> recall>=t 含 2/3、1.0 -> max p=0.75 (3个点)
        t=0.7,0.8,0.9,1.0 (<=1.0) -> recall>=t 只剩 1.0 -> p = 0.75      (4个点)
    AP = (4*1.0 + 3*0.75 + 4*0.75)/11 = 9.25/11 = 0.84090909...
    """
    tp = np.array([1.0, 0.0, 1.0, 1.0])
    fp = np.array([0.0, 1.0, 0.0, 0.0])
    p, r = precision_recall_curve(tp, fp, n_gt=3)
    ap = average_precision(p, r, "11_points")
    assert ap == pytest.approx((4 * 1.0 + 3 * 0.75 + 4 * 0.75) / 11.0, abs=1e-9)


def test_compute_ap_end_to_end():
    """端到端 compute_ap：3 真值，4 预测（3 命中 1 误检），all_points。
    真值：3 个分开的框；预测里 1 个误检落在空地。
    命中顺序（分数降序）=[TP,TP,FP,TP] -> 手算：
        cumTP=[1,2,2,3] cumFP=[0,0,1,1]
        precision=[1,1,2/3,3/4] recall=[1/3,2/3,2/3,1]
        包络后：AP = (1/3)*1 + (1/3)*1 + (1/3)*0.75 = 0.9167
    """
    gt = np.array([[0, 0, 10, 10],
                   [20, 20, 30, 30],
                   [40, 40, 50, 50]], dtype=float)
    pred = np.array([[0, 0, 10, 10],       # 0.9 TP
                     [20, 20, 30, 30],     # 0.85 TP
                     [200, 200, 210, 210], # 0.8 FP (空地)
                     [40, 40, 50, 50]],    # 0.7 TP
                    dtype=float)
    scores = np.array([0.9, 0.85, 0.8, 0.7])
    ap = compute_ap(gt_boxes=gt, pred_boxes=pred, pred_scores=scores,
                    iou_threshold=0.5, method="all_points")
    assert ap == pytest.approx((1 / 3) * 1 + (1 / 3) * 1 + (1 / 3) * 0.75, abs=1e-6)


def test_map_two_classes():
    """两类的 mAP = 两类 AP 的平均。
    类0：完美检测 -> AP=1.0
    类1：一半命中 -> 手算 AP。
    """
    detections = {
        0: (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),
        1: (np.array([[0, 0, 10, 10],
                      [100, 100, 110, 110]], dtype=float), np.array([0.9, 0.8])),
    }
    gts = {
        0: np.array([[0, 0, 10, 10]], dtype=float),          # 类0 一个真值，命中 -> AP=1
        1: np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float),  # 类1 两个真值
    }
    mAP, per = mean_average_precision(detections, gts, 0.5, "all_points")
    assert per[0] == pytest.approx(1.0)
    # 类1：预测 [0.9 命中真值A(TP), 0.8 空地(FP)]，n_gt=2
    # cumTP=[1,1] cumFP=[0,1] precision=[1,0.5] recall=[0.5,0.5]
    # 包络：AP = 0.5 * 1.0 = 0.5
    assert per[1] == pytest.approx(0.5, abs=1e-9)
    assert mAP == pytest.approx((1.0 + 0.5) / 2)


# ---------------------------------------------------------------------
# 边界情形
# ---------------------------------------------------------------------
def test_no_predictions():
    """有真值但无预测 -> AP=0（recall 永远 0）。"""
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    ap = compute_ap(np.zeros((0, 4)), np.zeros((0,)), gt, 0.5, "all_points")
    assert ap == pytest.approx(0.0)


def test_no_ground_truth():
    """无真值 -> compute_ap 约定返回 0；所有预测视为 FP。"""
    pred = np.array([[0, 0, 10, 10]], dtype=float)
    tp, fp, sc, n_gt = match_detections(pred, np.array([0.9]),
                                        np.zeros((0, 4)), 0.5)
    assert n_gt == 0
    assert fp.tolist() == [1.0]
    assert compute_ap(pred, np.array([0.9]), np.zeros((0, 4))) == 0.0


def test_map_skips_class_without_gt():
    """某类在真值里不存在 -> 不纳入 mAP 平均。"""
    detections = {
        0: (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),
        99: (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),  # 幽灵类
    }
    gts = {0: np.array([[0, 0, 10, 10]], dtype=float)}
    mAP, per = mean_average_precision(detections, gts, 0.5, "all_points")
    assert set(per.keys()) == {0}          # 99 被跳过
    assert mAP == pytest.approx(1.0)


def test_average_precision_empty_curve():
    """空 PR 曲线 -> AP=0。"""
    assert average_precision(np.array([]), np.array([]), "all_points") == 0.0
    assert average_precision(np.array([]), np.array([]), "11_points") == 0.0


def test_average_precision_bad_method_raises():
    """非法 method 抛 ValueError。"""
    with pytest.raises(ValueError):
        average_precision(np.array([1.0]), np.array([1.0]), method="nope")
