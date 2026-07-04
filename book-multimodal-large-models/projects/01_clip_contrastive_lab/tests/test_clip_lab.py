# -*- coding: utf-8 -*-
"""
test_clip_lab.py — CLIP 对比学习内核的单元测试
==============================================

覆盖四组硬性要求:
  1) InfoNCE 损失公式正确性 + 对称性;
  2) 温度对 logits/softmax 尖锐度的影响;
  3) 训练后配对相似度(相似度矩阵对角线)占优;
  4) 检索 recall@1 训练后显著提升。
全部离线、纯 CPU、秒级完成。
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

# 让测试可以直接 import 上级目录的 clip_lab.py(无需安装为包)。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_lab import (  # noqa: E402
    CLIPModel,
    info_nce_loss,
    l2_normalize,
    make_toy_pairs,
    recall_at_1,
    similarity_matrix,
    train_clip,
)


# ---------------------------------------------------------------------------
# 1. InfoNCE 损失:公式正确性
# ---------------------------------------------------------------------------
def test_info_nce_matches_manual_cross_entropy():
    """info_nce_loss 应等于"两方向交叉熵的平均"。"""
    torch.manual_seed(1)
    sim = torch.randn(5, 5)
    n = 5
    targets = torch.arange(n)
    expected = 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets))
    got = info_nce_loss(sim)
    assert torch.allclose(got, expected, atol=1e-6)


def test_info_nce_perfect_alignment_is_low():
    """当对角线远大于非对角线时,损失应趋近于 0(接近完美检索)。"""
    n = 6
    sim = torch.full((n, n), -10.0)
    sim.fill_diagonal_(10.0)  # 对角线极大,非对角线极小
    loss = info_nce_loss(sim)
    assert loss.item() < 1e-3


def test_info_nce_random_is_near_log_n():
    """全 0 logits(完全无区分)时,损失应等于 ln(N)。

    第一性原理:softmax(全 0) 是均匀分布,正样本概率 = 1/N,
    交叉熵 = -ln(1/N) = ln(N)。两个方向都是 ln(N),平均仍是 ln(N)。
    """
    n = 8
    sim = torch.zeros(n, n)
    loss = info_nce_loss(sim)
    assert abs(loss.item() - math.log(n)) < 1e-5


# ---------------------------------------------------------------------------
# 2. 对称性
# ---------------------------------------------------------------------------
def test_info_nce_is_symmetric_under_transpose():
    """损失对"图<->文"角色互换(即转置 sim)不变。

    因为损失 = 0.5*(CE(sim)+CE(sim.T)),转置后两项交换,和不变。
    """
    torch.manual_seed(2)
    sim = torch.randn(7, 7)
    assert torch.allclose(info_nce_loss(sim), info_nce_loss(sim.t()), atol=1e-6)


def test_similarity_matrix_shape_and_range():
    """相似度矩阵形状正确;温度=1 时,归一化后余弦相似度落在 [-1,1]。"""
    img = torch.randn(4, 10)
    txt = torch.randn(4, 12)  # 维度不同也应可算(各自归一化后点积)
    # 用同维度以便走 similarity_matrix(要求两侧 embed 维度相同)。
    img = torch.randn(4, 8)
    txt = torch.randn(4, 8)
    sim = similarity_matrix(img, txt, temperature=1.0)
    assert sim.shape == (4, 4)
    assert sim.max() <= 1.0 + 1e-5 and sim.min() >= -1.0 - 1e-5


# ---------------------------------------------------------------------------
# 3. 温度的影响
# ---------------------------------------------------------------------------
def test_temperature_scales_logits():
    """温度越小,logits 绝对值越大(被放大),softmax 越尖锐。"""
    img = torch.randn(5, 8)
    txt = torch.randn(5, 8)
    sim_small_t = similarity_matrix(img, txt, temperature=0.05)
    sim_large_t = similarity_matrix(img, txt, temperature=0.5)
    # 同一批数据,温度小的 logits 幅度应更大。
    assert sim_small_t.abs().max() > sim_large_t.abs().max()


def test_temperature_affects_softmax_sharpness():
    """温度越小,softmax 分布越尖锐(熵越低)。"""
    img = torch.randn(6, 8)
    txt = torch.randn(6, 8)

    def mean_entropy(temp):
        sim = similarity_matrix(img, txt, temperature=temp)
        p = F.softmax(sim, dim=1)
        ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1)  # 每行的熵
        return ent.mean().item()

    # 小温度 -> 更尖锐 -> 熵更低。
    assert mean_entropy(0.05) < mean_entropy(0.5)


def test_model_temperature_property():
    """CLIPModel.temperature 应与初始化温度一致(误差极小)。"""
    model = CLIPModel(img_dim=8, txt_dim=6, init_temperature=0.07)
    assert abs(float(model.temperature) - 0.07) < 1e-5


# ---------------------------------------------------------------------------
# 4. 训练后:配对对角占优 + recall@1 提升
# ---------------------------------------------------------------------------
def test_training_makes_diagonal_dominant():
    """训练后,相似度矩阵每一行的对角元素应是该行最大(配对占优)。"""
    imgs, txts, _ = make_toy_pairs(n_pairs=48, seed=3)
    _, hist = train_clip(imgs, txts, steps=300, seed=3)
    sim = hist["sim_after"]
    n = sim.shape[0]
    diag = sim.diag()
    # 对每一行,对角线元素应 >= 该行其它所有元素(允许极少数并列/噪声,用占比阈值)。
    row_max = sim.max(dim=1).values
    frac_row_ok = (diag >= row_max - 1e-6).float().mean().item()
    assert frac_row_ok >= 0.9, f"仅 {frac_row_ok:.2%} 的行对角占优"

    # 平均对角相似度应明显高于平均非对角相似度。
    mask = ~torch.eye(n, dtype=torch.bool)
    assert diag.mean().item() > sim[mask].mean().item() + 0.1


def test_recall_improves_after_training():
    """recall@1 训练后应显著高于训练前。"""
    imgs, txts, _ = make_toy_pairs(n_pairs=64, seed=0)
    _, hist = train_clip(imgs, txts, steps=300, seed=0)
    r_before = recall_at_1(hist["sim_before"])
    r_after = recall_at_1(hist["sim_after"])
    assert r_after > r_before, f"recall 未提升: {r_before} -> {r_after}"
    assert r_after >= 0.8, f"训练后 recall@1 偏低: {r_after}"


def test_loss_decreases_overall():
    """训练损失整体下降:末段平均损失应明显低于初段平均损失。"""
    imgs, txts, _ = make_toy_pairs(n_pairs=64, seed=1)
    _, hist = train_clip(imgs, txts, steps=300, seed=1)
    losses = hist["loss"]
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < early, f"损失未下降: {early:.3f} -> {late:.3f}"


# ---------------------------------------------------------------------------
# 5. 基础健壮性
# ---------------------------------------------------------------------------
def test_l2_normalize_unit_norm():
    """归一化后每个向量的 L2 范数应为 1。"""
    x = torch.randn(10, 7) * 5.0
    xn = l2_normalize(x)
    norms = xn.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_make_toy_pairs_shapes_and_reproducible():
    """玩具数据形状正确且可复现(同种子同结果)。"""
    a = make_toy_pairs(n_pairs=20, img_dim=32, txt_dim=24, seed=7)
    b = make_toy_pairs(n_pairs=20, img_dim=32, txt_dim=24, seed=7)
    assert a[0].shape == (20, 32) and a[1].shape == (20, 24) and a[2].shape == (20,)
    assert torch.allclose(a[0], b[0]) and torch.equal(a[2], b[2])


def test_forward_returns_finite_loss():
    """模型前向的损失应为有限值(无 NaN/Inf)。"""
    imgs, txts, _ = make_toy_pairs(n_pairs=16, seed=2)
    model = CLIPModel(img_dim=imgs.shape[1], txt_dim=txts.shape[1])
    sim, loss = model(imgs, txts)
    assert sim.shape == (16, 16)
    assert torch.isfinite(loss).item()
