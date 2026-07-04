# -*- coding: utf-8 -*-
"""
test_distill.py — 知识蒸馏核心库的单元测试
==========================================

覆盖三大类断言（对应任务要求）：
  1. 蒸馏损失公式正确（温度缩放 T²、KL 定义、alpha 端点行为、非负性…）。
  2. 训练后 student 与 teacher 的 KL 下降（蒸馏确实把二者拉近）。
  3. 温度影响软度（T 越大软标签越软 / 熵越大）。

外加若干「守门」测试：数据形状、模型可跑、确定性、蒸馏优于纯硬标签。

运行： python -m pytest -q
"""

import math

import torch
import torch.nn.functional as F

import sys
import os

# 让测试无论从哪个目录运行都能 import 到 distill.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill import (  # noqa: E402
    make_synthetic_data,
    make_teacher,
    make_student,
    subset,
    soft_targets,
    kl_divergence_loss,
    distillation_loss,
    train_teacher,
    train_student,
    student_teacher_kl,
    softness,
    accuracy,
)


# -----------------------------------------------------------------------------
# 固定一批小张量，供公式类测试反复使用
# -----------------------------------------------------------------------------
def _fixture_logits():
    torch.manual_seed(123)
    teacher = torch.randn(16, 3)
    student = torch.randn(16, 3)
    labels = torch.randint(0, 3, (16,))
    return student, teacher, labels


# =============================================================================
# A. 蒸馏损失公式正确性
# =============================================================================
def test_soft_targets_is_valid_distribution():
    """软标签必须是合法概率分布：非负、每行和为 1。"""
    logits = torch.randn(10, 4)
    p = soft_targets(logits, T=3.0)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_soft_targets_temperature_1_equals_softmax():
    """T=1 时 soft_targets 必须退化为普通 softmax。"""
    logits = torch.randn(8, 5)
    assert torch.allclose(soft_targets(logits, T=1.0), F.softmax(logits, dim=-1), atol=1e-6)


def test_kl_loss_matches_manual_formula():
    """
    对拍：kl_divergence_loss 必须等于手写的 T² * Σ p*(log p - log q) / B。
    这是「公式正确」最硬核的一条。
    """
    student, teacher, _ = _fixture_logits()
    T = 4.0
    got = kl_divergence_loss(student, teacher, T).item()

    # 手写参考实现
    p = F.softmax(teacher / T, dim=-1)
    logq = F.log_softmax(student / T, dim=-1)
    logp = F.log_softmax(teacher / T, dim=-1)
    manual = (T * T) * (p * (logp - logq)).sum(dim=-1).mean().item()

    assert math.isclose(got, manual, rel_tol=1e-5, abs_tol=1e-6)


def test_kl_loss_is_zero_when_identical():
    """teacher==student 时 KL 必须为 0（自己对自己没有散度）。"""
    logits = torch.randn(12, 3)
    kl = kl_divergence_loss(logits.clone(), logits.clone(), T=2.5)
    assert abs(kl.item()) < 1e-5


def test_kl_loss_non_negative():
    """KL 散度恒 >= 0（信息论基本性质）。"""
    for _ in range(5):
        s = torch.randn(20, 4)
        t = torch.randn(20, 4)
        assert kl_divergence_loss(s, t, T=3.0).item() >= -1e-6


def test_temperature_squared_scaling():
    """
    验证 T² 缩放确实存在：把 T 从 1 提到 2，
    「乘了 T²」的损失 应约等于 「同底 KL」乘以约 4 倍量级的补偿。
    这里用一个更稳的方式：直接检查我们的实现 = 4 * (不带 T²的 KL@T=2)。
    """
    s = torch.randn(16, 3)
    t = torch.randn(16, 3)
    T = 2.0
    with_scale = kl_divergence_loss(s, t, T).item()

    # 不带 T² 的裸 KL@T=2
    p = F.softmax(t / T, dim=-1)
    logq = F.log_softmax(s / T, dim=-1)
    bare = F.kl_div(logq, p, reduction="batchmean").item()

    assert math.isclose(with_scale, (T * T) * bare, rel_tol=1e-5, abs_tol=1e-6)


def test_distillation_loss_alpha_endpoints():
    """
    alpha 端点行为：
      alpha=1 → total 应 == 纯 CE（无蒸馏）。
      alpha=0 → total 应 == 纯 KL（纯模仿）。
    """
    student, teacher, labels = _fixture_logits()
    T = 3.0

    total1, ce1, kl1 = distillation_loss(student, teacher, labels, T=T, alpha=1.0)
    assert math.isclose(total1.item(), ce1.item(), rel_tol=1e-6, abs_tol=1e-7)

    total0, ce0, kl0 = distillation_loss(student, teacher, labels, T=T, alpha=0.0)
    assert math.isclose(total0.item(), kl0.item(), rel_tol=1e-6, abs_tol=1e-7)


def test_distillation_loss_components_returned():
    """复合损失应返回 (total, ce, kl) 三个标量，且 total 是二者的凸组合。"""
    student, teacher, labels = _fixture_logits()
    alpha, T = 0.5, 4.0
    total, ce, kl = distillation_loss(student, teacher, labels, T=T, alpha=alpha)
    expect = alpha * ce.item() + (1 - alpha) * kl.item()
    assert math.isclose(total.item(), expect, rel_tol=1e-6, abs_tol=1e-7)
    for x in (total, ce, kl):
        assert x.ndim == 0  # 都是标量


# =============================================================================
# B. 温度影响「软度」
# =============================================================================
def test_higher_temperature_increases_softness():
    """T 越大，软标签熵越大（越软）。这是温度的核心作用。"""
    torch.manual_seed(7)
    logits = torch.randn(64, 5) * 3.0  # 乘 3 让原分布偏尖，效果更明显
    ent_low = softness(logits, T=1.0)
    ent_mid = softness(logits, T=3.0)
    ent_high = softness(logits, T=10.0)
    assert ent_low < ent_mid < ent_high


def test_extreme_temperature_approaches_uniform():
    """T→很大 时软标签趋于均匀分布，熵趋于 log(C)。"""
    logits = torch.randn(32, 4) * 2.0
    ent = softness(logits, T=1000.0)
    assert abs(ent - math.log(4)) < 0.05  # 4 类均匀分布的熵 = ln4 ≈ 1.386


def test_low_temperature_sharpens():
    """T<1 时软标签变尖，熵应小于 T=1 时。"""
    torch.manual_seed(11)
    logits = torch.randn(64, 5)
    assert softness(logits, T=0.5) < softness(logits, T=1.0)


# =============================================================================
# C. 训练后 student-teacher KL 下降 + 蒸馏优于纯硬标签
# =============================================================================
def _make_setup(seed=0, n_classes=3, noise=0.5):
    """造一份数据 + 训练好 teacher，供集成测试复用。"""
    X, y = make_synthetic_data(n_per_class=200, n_classes=n_classes, noise=noise, seed=seed)
    teacher = make_teacher(in_dim=2, n_classes=n_classes)
    train_teacher(teacher, X, y, steps=200, lr=0.05)
    return X, y, teacher


def test_teacher_actually_learns():
    """先保证 teacher 本身训得动（准确率明显高于随机 1/3）。"""
    X, y, teacher = _make_setup()
    assert accuracy(teacher, X, y) > 0.75


def test_distillation_reduces_student_teacher_kl():
    """
    【核心断言 1】训练后 student 与 teacher 的分布 KL 应显著下降。
    对比「训练前的随机 student」与「蒸馏后的 student」。
    这条最硬：软标签 KL 项的直接目标就是拉近二者分布，因此几乎必然大幅下降。
    """
    X, y, teacher = _make_setup(seed=1)

    student = make_student(in_dim=2, n_classes=3)
    kl_before = student_teacher_kl(student, teacher, X, T=1.0)

    train_student(student, teacher, X, y, steps=300, lr=0.05, T=4.0, alpha=0.5)
    kl_after = student_teacher_kl(student, teacher, X, T=1.0)

    assert kl_after < kl_before          # 分布被拉近了
    assert kl_after < 0.5 * kl_before    # 且下降幅度明显（至少腰斩）


def test_distillation_reduces_kl_across_many_seeds():
    """
    【核心断言 1·加强版】跨多个 seed，蒸馏后的 KL 都应远小于训练前。
    避免「碰巧一个 seed 成功」，证明 KL 下降是稳健现象。
    """
    for seed in range(5):
        X, y, teacher = _make_setup(seed=seed, n_classes=4, noise=0.6)
        student = make_student(in_dim=2, n_classes=4)
        kl_before = student_teacher_kl(student, teacher, X, T=1.0)
        train_student(student, teacher, X, y, steps=300, lr=0.05, T=4.0, alpha=0.5)
        kl_after = student_teacher_kl(student, teacher, X, T=1.0)
        assert kl_after < 0.4 * kl_before, f"seed={seed}: {kl_after} 未显著小于 {kl_before}"


def test_distillation_beats_hard_label_on_test_accuracy():
    """
    【核心断言 2】数据稀缺时，蒸馏 student 的**测试集**准确率（跨 seed 平均）
    应优于「仅硬标签」student。

    这是 Hinton 的经典结论：软标签把 teacher 的「暗知识」注入 student，
    起到正则化作用，让 student 用很少的数据也能泛化得更好。

    ⚠️ 单个 seed 上二者差距可能被噪声淹没，所以这里对**多 seed 求平均**再比较，
    这才是科学、可复现的验证方式（也是论文里报均值±方差的原因）。
    """
    n_seeds = 8
    accs_distill, accs_hard = [], []
    for seed in range(n_seeds):
        # teacher 在「大」训练集上练好
        Xtr, ytr = make_synthetic_data(n_per_class=250, n_classes=4, noise=0.6, seed=seed)
        Xte, yte = make_synthetic_data(n_per_class=250, n_classes=4, noise=0.6, seed=seed + 100)
        teacher = make_teacher(in_dim=2, n_classes=4)
        train_teacher(teacher, Xtr, ytr, steps=300, lr=0.05)

        # student 只用少量样本（模拟数据稀缺）
        Xs, ys = subset(Xtr, ytr, n=30, seed=seed)
        s_distill = make_student(in_dim=2, n_classes=4, seed=42)
        s_hard = make_student(in_dim=2, n_classes=4, seed=42)

        train_student(s_distill, teacher, Xs, ys, steps=300, lr=0.03, T=4.0, alpha=0.3)
        train_student(s_hard, None, Xs, ys, steps=300, lr=0.03)

        accs_distill.append(accuracy(s_distill, Xte, yte))
        accs_hard.append(accuracy(s_hard, Xte, yte))

    mean_distill = sum(accs_distill) / n_seeds
    mean_hard = sum(accs_hard) / n_seeds
    # 蒸馏的平均测试准确率应 >= 硬标签（留一点数值容差）
    assert mean_distill >= mean_hard - 1e-6, f"distill={mean_distill:.4f} hard={mean_hard:.4f}"


def test_distillation_loss_kl_component_decreases_during_training():
    """训练过程中 KL 分量应总体下降（首段均值 > 末段均值）。"""
    X, y, teacher = _make_setup(seed=3)
    student = make_student(in_dim=2, n_classes=3)
    hist = train_student(student, teacher, X, y, steps=300, lr=0.05, T=4.0, alpha=0.5)

    head = sum(hist.kl[:30]) / 30
    tail = sum(hist.kl[-30:]) / 30
    assert tail < head


# =============================================================================
# D. 守门测试：数据 / 模型 / 确定性
# =============================================================================
def test_synthetic_data_shapes():
    X, y = make_synthetic_data(n_per_class=100, n_classes=3, n_features=2, seed=0)
    assert X.shape == (300, 2)
    assert y.shape == (300,)
    assert set(y.tolist()) == {0, 1, 2}


def test_synthetic_data_deterministic():
    """同 seed 必须产出完全一样的数据（可复现）。"""
    X1, y1 = make_synthetic_data(seed=5)
    X2, y2 = make_synthetic_data(seed=5)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)


def test_models_forward_shapes():
    X, _ = make_synthetic_data(n_per_class=10, seed=0)
    teacher = make_teacher(in_dim=2, n_classes=3)
    student = make_student(in_dim=2, n_classes=3)
    assert teacher(X).shape == (30, 3)
    assert student(X).shape == (30, 3)


def test_student_smaller_than_teacher():
    """Student 参数量必须明显小于 Teacher（否则谈不上『蒸馏到小模型』）。"""
    teacher = make_teacher(in_dim=2, n_classes=3)
    student = make_student(in_dim=2, n_classes=3)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    assert n_s < n_t / 3  # student 至少小 3 倍
