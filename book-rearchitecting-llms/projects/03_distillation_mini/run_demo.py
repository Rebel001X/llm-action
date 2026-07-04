# -*- coding: utf-8 -*-
"""
run_demo.py — 知识蒸馏最小实现·端到端演示
==========================================

跑通一次完整实验并出 3 张图（保存到 figures/）：

  1. figures/training_curves.png
        「蒸馏 student」 vs 「仅硬标签 student」的训练损失 / 准确率 / 与 teacher 的 KL 对比。
        —— 证明：有 teacher 指导，student 学得更快、更像老师。

  2. figures/temperature_softness.png
        同一组 logits 在不同温度 T 下的软标签条形图 + 平均熵曲线。
        —— 证明：T 越大，软标签越「软」（暗知识被放大）。

  3. figures/decision_boundary.png
        Teacher / 蒸馏 Student / 硬标签 Student 三者的决策边界。
        —— 直观看到蒸馏 student 的边界更贴近 teacher。

运行：  python run_demo.py
无需联网、无需 GPU、无需任何预训练模型。
"""

import os
import sys

# Windows 控制台默认 GBK，emoji/中文会 UnicodeEncodeError；强制 stdout 用 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")  # 无界面后端，纯出图不弹窗（服务器/CI 友好）
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体 + 负号正常显示（否则中文变方块、负号变框）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill import (  # noqa: E402
    make_synthetic_data,
    make_teacher,
    make_student,
    subset,
    train_teacher,
    train_student,
    student_teacher_kl,
    softness,
    soft_targets,
    accuracy,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 实验 1：蒸馏 vs 仅硬标签 —— 训练曲线
# =============================================================================
def experiment_training(steps=300, T=4.0, alpha=0.5, seed=1):
    print(f"\n[实验1] 训练 teacher / 蒸馏 student / 硬标签 student（steps={steps}, T={T}, alpha={alpha}）")
    # 4 类 + 中等噪声：类间有重叠，teacher(容量大)能拟合好，小 student 吃力，蒸馏收益才看得出来
    X, y = make_synthetic_data(n_per_class=250, n_classes=4, noise=0.6, seed=seed)

    # 1) 训练 teacher
    teacher = make_teacher(in_dim=2, n_classes=4)
    train_teacher(teacher, X, y, steps=steps, lr=0.05)
    acc_t = accuracy(teacher, X, y)
    print(f"    teacher 准确率 = {acc_t:.3f}")

    # 2) 两个同初始化的 student，一个蒸馏一个纯硬标签
    s_distill = make_student(in_dim=2, n_classes=4, seed=42)
    s_hard = make_student(in_dim=2, n_classes=4, seed=42)

    h_distill = train_student(s_distill, teacher, X, y, steps=steps, lr=0.05, T=T, alpha=alpha)
    h_hard = train_student(s_hard, None, X, y, steps=steps, lr=0.05)

    acc_d = accuracy(s_distill, X, y)
    acc_h = accuracy(s_hard, X, y)
    kl_d = student_teacher_kl(s_distill, teacher, X, T=1.0)
    kl_h = student_teacher_kl(s_hard, teacher, X, T=1.0)
    print(f"    蒸馏  student 准确率 = {acc_d:.3f} | 与 teacher 的 KL = {kl_d:.4f}")
    print(f"    硬标签 student 准确率 = {acc_h:.3f} | 与 teacher 的 KL = {kl_h:.4f}")
    if kl_d < kl_h:
        print(f"    ✅ 蒸馏 student 与 teacher 分布更接近（KL 小 {(kl_h-kl_d)/kl_h*100:.1f}%）")

    # ---- 逐步记录「与 teacher 的 KL」曲线（重训一遍并每步测 KL）----
    kl_curve_d = _kl_trajectory(teacher, X, y, steps, T, alpha, distill=True)
    kl_curve_h = _kl_trajectory(teacher, X, y, steps, T, alpha, distill=False)

    # ---- 画图：3 联图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(h_distill.loss, label="蒸馏 student（总损失）", color="#d62728")
    axes[0].plot(h_hard.loss, label="硬标签 student（CE）", color="#1f77b4")
    axes[0].set_title("① 训练损失曲线")
    axes[0].set_xlabel("训练步 step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(h_distill.acc, label="蒸馏 student", color="#d62728")
    axes[1].plot(h_hard.acc, label="硬标签 student", color="#1f77b4")
    axes[1].axhline(acc_t, ls="--", color="#2ca02c", label=f"teacher 上限={acc_t:.2f}")
    axes[1].set_title("② 训练集准确率")
    axes[1].set_xlabel("训练步 step")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(kl_curve_d, label="蒸馏 student", color="#d62728")
    axes[2].plot(kl_curve_h, label="硬标签 student", color="#1f77b4")
    axes[2].set_title("③ 与 teacher 的 KL 散度（越低越像老师）")
    axes[2].set_xlabel("训练步 step")
    axes[2].set_ylabel("KL(teacher‖student)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("知识蒸馏：蒸馏 student vs 仅硬标签 student", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    图已保存 -> {out}")

    return X, y, teacher, s_distill, s_hard


def _kl_trajectory(teacher, X, y, steps, T, alpha, distill):
    """重训一个 student，每步记录它与 teacher 的 KL，用于画收敛轨迹。"""
    student = make_student(in_dim=2, n_classes=4, seed=42)
    opt = torch.optim.Adam(student.parameters(), lr=0.05)
    teacher.eval()
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        s_logits = student(X)
        if distill:
            with torch.no_grad():
                t_logits = teacher(X)
            log_q = F.log_softmax(s_logits / T, dim=-1)
            p = F.softmax(t_logits / T, dim=-1)
            kl = (T * T) * F.kl_div(log_q, p, reduction="batchmean")
            ce = F.cross_entropy(s_logits, y)
            loss = alpha * ce + (1 - alpha) * kl
        else:
            loss = F.cross_entropy(s_logits, y)
        loss.backward()
        opt.step()
        traj.append(student_teacher_kl(student, teacher, X, T=1.0))
        student.train()
    return traj


# =============================================================================
# 实验 2：温度对软标签软度的影响
# =============================================================================
def experiment_temperature():
    print("\n[实验2] 温度 T 对软标签软度的影响")
    # 一个偏尖的 logits 向量（3 类），best 类明显但存在暗知识
    logits = torch.tensor([[4.0, 1.5, 0.2]])
    Ts = [1.0, 2.0, 4.0, 8.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (左) 不同 T 下的软标签条形图
    width = 0.18
    x = np.arange(3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, T in enumerate(Ts):
        p = soft_targets(logits, T=T)[0].numpy()
        axes[0].bar(x + i * width, p, width, label=f"T={T:g}", color=colors[i])
        print(f"    T={T:>4g}  软标签 = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]  熵={softness(logits,T):.3f}")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(["类0（目标）", "类1", "类2"])
    axes[0].set_ylabel("软标签概率")
    axes[0].set_title("① 同一 logits 在不同温度下的软标签\n（T 越大越平 = 越软）")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    # (右) 熵随 T 连续变化曲线（在一批随机 logits 上求平均）
    torch.manual_seed(0)
    batch = torch.randn(200, 3) * 2.5
    T_grid = np.linspace(0.3, 12, 60)
    ent = [softness(batch, T=float(t)) for t in T_grid]
    axes[1].plot(T_grid, ent, color="#9467bd", lw=2)
    axes[1].axhline(np.log(3), ls="--", color="gray", label="均匀分布熵 ln3≈1.10")
    axes[1].axvline(1.0, ls=":", color="k", alpha=0.5, label="T=1（普通 softmax）")
    axes[1].set_xlabel("温度 T")
    axes[1].set_ylabel("平均熵（越大越软）")
    axes[1].set_title("② 软度（平均熵）随温度单调上升")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("温度 T 如何控制软标签的『软硬』", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "temperature_softness.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    图已保存 -> {out}")


# =============================================================================
# 实验 3：决策边界对比（teacher / 蒸馏 student / 硬标签 student）
# =============================================================================
@torch.no_grad()
def experiment_boundary(X, y, teacher, s_distill, s_hard):
    print("\n[实验3] 决策边界对比")
    # 构造网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    models = [("Teacher（大）", teacher), ("蒸馏 Student（小）", s_distill), ("硬标签 Student（小）", s_hard)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, model) in zip(axes, models):
        model.eval()
        Z = model(grid).argmax(dim=-1).numpy().reshape(xx.shape)
        n_cls = int(y.max().item()) + 1
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis", levels=np.arange(n_cls + 1) - 0.5)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="viridis", edgecolors="k", linewidths=0.2)
        ax.set_title(f"{name}\n准确率={accuracy(model, X, y):.3f}")
        ax.set_xlabel("特征 x1")
        ax.set_ylabel("特征 x2")

    fig.suptitle("决策边界：蒸馏 student 的边界更贴近 teacher", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "decision_boundary.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    图已保存 -> {out}")


# =============================================================================
# 实验 4：数据稀缺时蒸馏的「正则化」收益（测试集准确率，跨 seed 求均值）
# =============================================================================
def experiment_low_data(n_seeds=8, n_train=30, T=4.0, alpha=0.3):
    print(f"\n[实验4] 小样本蒸馏的泛化收益（student 只用 {n_train} 个样本，跨 {n_seeds} 个 seed 平均测试准确率）")
    accs_d, accs_h = [], []
    for seed in range(n_seeds):
        Xtr, ytr = make_synthetic_data(n_per_class=250, n_classes=4, noise=0.6, seed=seed)
        Xte, yte = make_synthetic_data(n_per_class=250, n_classes=4, noise=0.6, seed=seed + 100)
        teacher = make_teacher(in_dim=2, n_classes=4)
        train_teacher(teacher, Xtr, ytr, steps=300, lr=0.05)

        Xs, ys = subset(Xtr, ytr, n=n_train, seed=seed)  # 只给 student 少量样本
        sd = make_student(in_dim=2, n_classes=4, seed=42)
        sh = make_student(in_dim=2, n_classes=4, seed=42)
        train_student(sd, teacher, Xs, ys, steps=300, lr=0.03, T=T, alpha=alpha)
        train_student(sh, None, Xs, ys, steps=300, lr=0.03)

        accs_d.append(accuracy(sd, Xte, yte))
        accs_h.append(accuracy(sh, Xte, yte))

    md, mh = float(np.mean(accs_d)), float(np.mean(accs_h))
    sd_, sh_ = float(np.std(accs_d)), float(np.std(accs_h))
    print(f"    蒸馏  student 测试准确率 = {md:.3f} ± {sd_:.3f}")
    print(f"    硬标签 student 测试准确率 = {mh:.3f} ± {sh_:.3f}")
    print(f"    {'✅ 蒸馏泛化更好' if md >= mh else '≈ 二者接近'}（差值 {md - mh:+.4f}）")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = ["蒸馏 student", "硬标签 student"]
    means = [md, mh]
    errs = [sd_, sh_]
    colors = ["#d62728", "#1f77b4"]
    bars = ax.bar(labels, means, yerr=errs, capsize=8, color=colors, alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.005, f"{m:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("测试集准确率（越高越好）")
    ax.set_ylim(min(means) - 0.05, 1.0)
    ax.set_title(f"数据稀缺（每 student 仅 {n_train} 样本）时\n软标签作为正则化提升泛化 · {n_seeds} seed 平均")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "low_data_generalization.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    图已保存 -> {out}")


def main():
    torch.manual_seed(0)
    print("=" * 68)
    print(" 知识蒸馏最小实现 · 端到端演示（CPU，无需联网/GPU/预训练模型）")
    print("=" * 68)

    X, y, teacher, s_distill, s_hard = experiment_training()
    experiment_temperature()
    experiment_boundary(X, y, teacher, s_distill, s_hard)
    experiment_low_data()

    print("\n" + "=" * 68)
    print(" 全部完成 ✅   图片在 figures/ 目录：")
    print("   - training_curves.png          （蒸馏 vs 硬标签 训练曲线 + KL 收敛）")
    print("   - temperature_softness.png     （温度对软标签的影响）")
    print("   - decision_boundary.png        （三个模型决策边界）")
    print("   - low_data_generalization.png  （小样本蒸馏的泛化收益）")
    print("=" * 68)


if __name__ == "__main__":
    main()
