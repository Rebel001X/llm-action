# -*- coding: utf-8 -*-
"""
distill.py — 知识蒸馏（Knowledge Distillation, KD）最小实现核心库
==================================================================

配套《Rearchitecting LLMs》第 6 章「蒸馏恢复知识」。

本文件用纯 PyTorch（CPU 即可）实现一个**玩具级但公式完全正确**的知识蒸馏：
    - Teacher（稍大的 MLP 分类器）先在合成数据上训练好、然后冻结。
    - Student（更小的 MLP）用**复合损失** L = α·CE(hard) + β·T²·KL(soft) 去追平 Teacher。
    - 对照组：同一个 Student 只用 CE(hard) 训练，用来证明「有 Teacher 指导确实学得更好」。

核心概念（是什么 / 为什么 / 怎么用 / 代价），README 里逐行讲，这里给出可运行、可测试的实现。

设计原则
--------
1. **确定性**：所有随机都走一个可传入的 `torch.Generator`/`seed`，测试才能复现。
2. **纯函数式的损失**：`distillation_loss` 不含任何隐藏状态，方便单元测试直接喂张量。
3. **零外部依赖**：只用 torch，不下载任何预训练模型、不联网。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 合成数据集：两个同心圆环 / 高斯团（多分类），CPU 上一瞬间生成
# =============================================================================
def make_synthetic_data(
    n_per_class: int = 400,
    n_classes: int = 3,
    n_features: int = 2,
    noise: float = 0.35,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成一个「多高斯团」分类数据集（可线性/非线性分开，取决于噪声）。

    为什么自己造数据而不用 sklearn？——保证**零依赖、零下载、完全确定**。

    参数
    ----
    n_per_class : 每类样本数
    n_classes   : 类别数（默认 3 类）
    n_features  : 特征维度（默认 2，方便画图）
    noise       : 高斯噪声标准差，越大越难分
    seed        : 随机种子

    返回
    ----
    X : (N, n_features) float32
    y : (N,)            int64  标签
    """
    g = torch.Generator().manual_seed(seed)
    # 每个类的中心均匀分布在一个圆上，保证类间可分但有重叠
    angles = torch.arange(n_classes, dtype=torch.float32) / n_classes * 2 * torch.pi
    # 中心半径 2.0，让类团拉开一点距离
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 2.0  # (C, 2)
    if n_features > 2:
        pad = torch.zeros(n_classes, n_features - 2)
        centers = torch.cat([centers, pad], dim=1)

    xs, ys = [], []
    for c in range(n_classes):
        # 以类中心为均值撒高斯点
        pts = torch.randn(n_per_class, n_features, generator=g) * noise + centers[c]
        xs.append(pts)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    # 打乱（否则前 1/3 全是 0 类，batch 训练会偏）
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


def subset(X: torch.Tensor, y: torch.Tensor, n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 (X, y) 里确定性地抽 n 个样本，用于「小样本蒸馏」实验。

    为什么需要它？——知识蒸馏最经典的收益出现在**数据稀缺**时：软标签相当于
    teacher 额外注入的「暗知识」，起到正则化作用，让 student 用很少的数据也能泛化得更好。
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    return X[idx], y[idx]


# =============================================================================
# 2. 模型：Teacher（稍大）与 Student（小）都是简单 MLP
# =============================================================================
class MLP(nn.Module):
    """一个最普通的多层感知机分类器。宽度 `hidden` 决定它是 Teacher 还是 Student。"""

    def __init__(self, in_dim: int, hidden: int, n_classes: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, n_classes)]  # 最后一层输出 logits（不带 softmax）
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 返回 logits，形状 (B, n_classes)


def make_teacher(in_dim: int, n_classes: int, seed: int = 1) -> MLP:
    """Teacher：更宽更深（hidden=64, depth=3），容量大、拟合更好。"""
    torch.manual_seed(seed)
    return MLP(in_dim, hidden=64, n_classes=n_classes, depth=3)


def make_student(in_dim: int, n_classes: int, seed: int = 2) -> MLP:
    """Student：很小（hidden=8, depth=2），故意让它单独学不太动，凸显蒸馏收益。"""
    torch.manual_seed(seed)
    return MLP(in_dim, hidden=8, n_classes=n_classes, depth=2)


# =============================================================================
# 3. 蒸馏损失：本项目的心脏 —— 温度缩放的 KL + 交叉熵
# =============================================================================
def soft_targets(logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    把 logits 用温度 T 软化后得到「软标签」概率分布：softmax(logits / T)。

    T = 1  → 普通 softmax（尖锐）。
    T > 1  → 分布变「软」（平滑），暗知识（dark knowledge，即非目标类的相对概率）被放大。
    T < 1  → 分布变「尖」（更接近 one-hot）。

    ⚠️ 这里只除 T，不做别的缩放；缩放 T² 的补偿放在 kl_divergence_loss 里，别重复。
    """
    return F.softmax(logits / T, dim=-1)


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
) -> torch.Tensor:
    r"""
    温度缩放的 KL 散度蒸馏损失（Hinton et al. 2015 的标准形式）。

    数学定义
    --------
    记 p = softmax(teacher_logits / T)（软标签，teacher 的信念）
       q = softmax(student_logits / T)（student 的信念）

        L_KL = T^2 * KL(p || q) = T^2 * Σ_i p_i * (log p_i - log q_i)

    为什么乘 T² ？
    --------------
    softmax 里除以 T 会让梯度整体缩小约 1/T² 倍（对 logits 求导时链式法则带出 1/T）。
    乘回 T² 是为了让**软标签项的梯度量级与硬标签项可比**，这样调 α/β 时不会因为 T 变了
    就要跟着重调权重。这是 Hinton 原论文的经典处理，面试高频考点。

    实现细节
    --------
    - 用 `F.log_softmax(student/T)` 而不是 `log(softmax(...))`：数值更稳（避免 log(0)）。
    - `F.kl_div(input, target)` 约定 **input 要是 log 概率、target 是概率**，
      且计算的是 Σ target * (log target - input)。这里 target=p、input=log q，正好是 KL(p||q)。
    - `reduction="batchmean"`：对 batch 求平均（除以样本数），这才是数学上正确的 KL 期望。

    参数
    ----
    student_logits : (B, C) student 的原始 logits（未软化）
    teacher_logits : (B, C) teacher 的原始 logits（未软化）
    T              : 温度 (>0)

    返回
    ----
    标量 tensor：T² 缩放后的 KL 散度
    """
    # student 侧用 log_softmax（KL 的 input 需要 log 概率）
    log_q = F.log_softmax(student_logits / T, dim=-1)
    # teacher 侧用 softmax（target 需要普通概率），detach 保证不给 teacher 回传梯度
    p = F.softmax(teacher_logits / T, dim=-1)
    # F.kl_div: Σ p * (log p - log q)，batchmean = 除以 batch 大小
    kl = F.kl_div(log_q, p, reduction="batchmean")
    return (T * T) * kl


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    复合蒸馏损失（compound loss）：

        L = alpha * CE(student_logits, labels)          # 硬标签：贴近真值
          + (1 - alpha) * T^2 * KL(teacher || student)  # 软标签：模仿老师

    这与书里第 6 章的 `α·任务损失 + β·logits损失` 是同一个东西，
    只是我们用 (alpha, 1-alpha) 让两项权重和为 1，更直观。

    参数
    ----
    alpha : 硬标签权重 ∈ [0,1]。alpha=1 退化成普通监督训练（无蒸馏），
            alpha=0 变成纯模仿老师（忽略真值标签）。

    返回
    ----
    (total, ce, kl) 三个标量：总损失、硬标签 CE 分量、软标签 KL 分量（后两个用于日志/画图）
    """
    ce = F.cross_entropy(student_logits, labels)          # 硬标签交叉熵
    kl = kl_divergence_loss(student_logits, teacher_logits, T)  # 软标签 KL（已含 T²）
    total = alpha * ce + (1.0 - alpha) * kl
    return total, ce, kl


# =============================================================================
# 4. 训练循环：train_teacher / train_student（蒸馏 or 仅硬标签）
# =============================================================================
@dataclass
class History:
    """训练历史记录，便于测试断言与画图。"""
    loss: list[float] = field(default_factory=list)   # 总损失
    ce: list[float] = field(default_factory=list)     # CE 分量
    kl: list[float] = field(default_factory=list)     # KL 分量（蒸馏才有）
    acc: list[float] = field(default_factory=list)    # 训练集准确率


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """整体准确率（argmax logits == label）。"""
    model.eval()
    pred = model(X).argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_teacher(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
) -> History:
    """标准监督训练把 Teacher 练好（只用硬标签 CE）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = History()
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        hist.loss.append(loss.item())
        hist.acc.append(accuracy(model, X, y))
        model.train()
    return hist


def train_student(
    student: MLP,
    teacher: MLP | None,
    X: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
    T: float = 4.0,
    alpha: float = 0.5,
) -> History:
    """
    训练 Student。

    - 若 `teacher is None`：仅用硬标签 CE（对照组 / baseline）。
    - 若给了 `teacher`：用复合蒸馏损失（teacher 冻结、eval、no_grad 前向）。

    返回 History，方便对比「蒸馏 vs 仅硬标签」。
    """
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hist = History()

    # teacher 冻结：eval 模式 + 不需要梯度
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    student.train()
    for _ in range(steps):
        opt.zero_grad()
        s_logits = student(X)

        if teacher is None:
            # 对照组：纯硬标签
            loss = F.cross_entropy(s_logits, y)
            ce, kl = loss, torch.tensor(0.0)
        else:
            with torch.no_grad():
                t_logits = teacher(X)  # teacher 前向不建图
            loss, ce, kl = distillation_loss(s_logits, t_logits, y, T=T, alpha=alpha)

        loss.backward()
        opt.step()

        hist.loss.append(loss.item())
        hist.ce.append(ce.item())
        hist.kl.append(kl.item())
        hist.acc.append(accuracy(student, X, y))
        student.train()
    return hist


# =============================================================================
# 5. 评估工具：student 与 teacher 的分布有多接近？
# =============================================================================
@torch.no_grad()
def student_teacher_kl(
    student: MLP,
    teacher: MLP,
    X: torch.Tensor,
    T: float = 1.0,
) -> float:
    """
    衡量 student 与 teacher 输出分布的接近程度（越小越像）。

    注意：这里用 T=1 的「裸」KL 来度量最终分布差异（不带 T² 补偿，
    因为我们要的是「真实预测分布」有多像，而不是训练损失量级）。
    返回 KL(teacher || student) 的 batchmean，非负，越接近 0 越像。
    """
    student.eval()
    teacher.eval()
    s = student(X)
    t = teacher(X)
    log_q = F.log_softmax(s / T, dim=-1)
    p = F.softmax(t / T, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean").item()


@torch.no_grad()
def softness(logits: torch.Tensor, T: float) -> float:
    """
    「软度」的量化指标：软标签分布的平均熵（entropy）。

    熵越大 = 分布越平（越软）；熵越小 = 分布越尖（越接近 one-hot）。
    用来验证「温度 T 越高，软标签越软」。返回 nats 为单位的平均熵。
    """
    p = F.softmax(logits / T, dim=-1)
    # 加 eps 防 log(0)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    return ent.mean().item()
