# -*- coding: utf-8 -*-
"""
run_demo.py —— 宽度剪枝实验室 · 可视化演示

跑完会在当前目录产出三张图（Agg 后端，无需显示器，中文用微软雅黑）：

    fig1_error_vs_ratio.png     剪枝比 vs 输出相对误差（三种评分策略对比）
    fig2_importance_hist.png    神经元重要性分布 + 剪枝阈值（谁被删一目了然）
    fig3_param_vs_error.png     参数下降 vs 输出误差（帕累托权衡曲线）
    fig4_attention_heads.png    注意力剪头：保留头数 vs 输出误差

并在终端打印一份剪枝报告表。

运行：
    python run_demo.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 无显示器 / 服务器 / CI 环境必须用 Agg
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False                        # 负号正常显示

import torch

import width_pruning as wp


# ----------------------------------------------------------------------------
# 构造演示用的玩具 MLP：神经元重要性有梯度（有的强、有的弱）
# ----------------------------------------------------------------------------

def build_demo_mlp(hidden=64, inter=512, seed=42):
    """构造一个「重要性有梯度」的 GluMLP，让剪枝效果清晰可见。

    第 i 个神经元乘一个从 1.0 平滑衰减到 0.05 的 scale，并打散顺序。
    这样：剪掉弱神经元误差小，剪到强神经元误差才快速上升 —— 曲线好看、有教学意义。
    """
    torch.manual_seed(seed)
    mlp = wp.GluMLP(hidden, inter)
    with torch.no_grad():
        scale = torch.linspace(1.0, 0.05, inter)
        scale = scale[torch.randperm(inter)]
        mlp.gate_proj.weight.mul_(scale.unsqueeze(1))
        mlp.up_proj.weight.mul_(scale.unsqueeze(1))
        mlp.down_proj.weight.mul_(scale.unsqueeze(0))
    return mlp


def build_calibration_data(hidden=64, n_batches=8, batch=4, seq=16, seed=7):
    """构造校准数据（hybrid 评分要用）。让输入在部分维度更有能量，制造激活差异。"""
    torch.manual_seed(seed)
    data = []
    for _ in range(n_batches):
        x = torch.randn(batch, seq, hidden)
        x[..., hidden // 2:] *= 0.3  # 后半维输入偏弱
        data.append(x)
    return data


# ----------------------------------------------------------------------------
# 图 1：剪枝比 vs 输出误差（三策略对比）
# ----------------------------------------------------------------------------

def fig_error_vs_ratio(mlp, x, dataloader):
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    methods = {
        "peak_to_peak (静态峰峰值)": "peak_to_peak",
        "l2 (静态L2范数)": "l2",
        "hybrid (数据驱动混合)": "hybrid",
    }
    plt.figure(figsize=(9, 5.5))
    for label, m in methods.items():
        errs = []
        for r in ratios:
            kw = {"dataloader": dataloader} if m == "hybrid" else {}
            new_mlp, _ = wp.prune_mlp(mlp, r, method=m, **kw)
            errs.append(wp.relative_output_error(mlp, new_mlp, x))
        plt.plot([r * 100 for r in ratios], errs, marker="o", label=label, linewidth=2)
    plt.xlabel("剪枝比例 prune ratio (%)")
    plt.ylabel("输出相对误差 relative L2 error")
    plt.title("宽度剪枝：剪枝比越大，输出误差越大\n（重要神经元先保留 → 小剪枝比误差小）")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_error_vs_ratio.png", dpi=120)
    plt.close()
    print("[saved] fig1_error_vs_ratio.png")


# ----------------------------------------------------------------------------
# 图 2：神经元重要性分布 + 剪枝阈值
# ----------------------------------------------------------------------------

def fig_importance_hist(mlp, prune_ratio=0.5):
    importance = wp.peak_to_peak_importance(mlp.gate_proj.weight.data, mlp.up_proj.weight.data)
    keep = wp.select_indices_to_keep(importance, prune_ratio)
    keep_set = set(keep.tolist())
    kept = importance[list(keep_set)].numpy()
    dropped = importance[[i for i in range(importance.numel()) if i not in keep_set]].numpy()

    plt.figure(figsize=(9, 5.5))
    plt.hist(dropped, bins=40, alpha=0.7, label=f"被剪掉 (n={len(dropped)})", color="#e07a5f")
    plt.hist(kept, bins=40, alpha=0.7, label=f"保留 (n={len(kept)})", color="#81b29a")
    # 阈值线：保留集里的最小重要性
    threshold = importance[list(keep_set)].min().item()
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"剪枝阈值 ≈ {threshold:.2f}")
    plt.xlabel("神经元重要性分数 (peak-to-peak)")
    plt.ylabel("神经元个数")
    plt.title(f"神经元重要性分布：{int(prune_ratio*100)}% 剪枝下谁去谁留\n"
              "（阈值右侧=高重要性保留，左侧=低重要性删除）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig2_importance_hist.png", dpi=120)
    plt.close()
    print("[saved] fig2_importance_hist.png")


# ----------------------------------------------------------------------------
# 图 3：参数下降 vs 输出误差（帕累托权衡）
# ----------------------------------------------------------------------------

def fig_param_vs_error(mlp, x):
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    reds, errs, labels = [], [], []
    for r in ratios:
        new_mlp, stats = wp.prune_mlp(mlp, r, method="peak_to_peak")
        reds.append(stats.param_reduction_pct)
        errs.append(wp.relative_output_error(mlp, new_mlp, x))
        labels.append(f"{int(r*100)}%")

    plt.figure(figsize=(9, 5.5))
    plt.plot(reds, errs, marker="s", color="#3d5a80", linewidth=2)
    for red, err, lab in zip(reds, errs, labels):
        plt.annotate(lab, (red, err), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("参数量下降 param reduction (%)")
    plt.ylabel("输出相对误差 relative L2 error")
    plt.title("压缩 vs 精度 的帕累托权衡\n（越靠右下角越好：省得多、误差小）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig3_param_vs_error.png", dpi=120)
    plt.close()
    print("[saved] fig3_param_vs_error.png")


# ----------------------------------------------------------------------------
# 图 4：注意力剪头 —— 保留头数 vs 误差
# ----------------------------------------------------------------------------

def fig_attention_heads():
    torch.manual_seed(0)
    attn = wp.ToyMultiHeadAttention(128, 16)
    # 制造头之间的重要性差异：给部分头的 q/k/v 权重放大
    with torch.no_grad():
        Dh = attn.head_dim
        scale = torch.linspace(1.0, 0.1, 16)
        for h in range(16):
            rows = slice(h * Dh, (h + 1) * Dh)
            attn.q_proj.weight[rows].mul_(scale[h])
            attn.k_proj.weight[rows].mul_(scale[h])
            attn.v_proj.weight[rows].mul_(scale[h])
    x = torch.randn(2, 10, 128)

    keep_nums = list(range(1, 16))
    errs = []
    for k in keep_nums:
        new_attn, _ = wp.prune_attention(attn, k, by="num")
        errs.append(wp.relative_output_error(attn, new_attn, x))

    plt.figure(figsize=(9, 5.5))
    plt.plot(keep_nums, errs, marker="^", color="#8338ec", linewidth=2)
    plt.xlabel("保留的注意力头数 (原始 16 头)")
    plt.ylabel("输出相对误差 relative L2 error")
    plt.title("注意力剪头：保留头越少，误差越大\n（重要头优先保留 → 保留多则误差小）")
    plt.gca().invert_xaxis()  # 从多到少，直观展示误差上升
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig4_attention_heads.png", dpi=120)
    plt.close()
    print("[saved] fig4_attention_heads.png")


# ----------------------------------------------------------------------------
# 终端报告
# ----------------------------------------------------------------------------

def print_report(mlp, x, dataloader):
    print("\n" + "=" * 68)
    print("宽度剪枝报告  (玩具 GLU-MLP: hidden=64, intermediate=512)")
    print("=" * 68)
    header = f"{'method':<14}{'ratio':>7}{'inter':>10}{'params':>12}{'省%':>8}{'误差':>10}"
    print(header)
    print("-" * 68)
    for m in ["peak_to_peak", "l2", "hybrid"]:
        for r in [0.2, 0.4, 0.6]:
            kw = {"dataloader": dataloader} if m == "hybrid" else {}
            new_mlp, st = wp.prune_mlp(mlp, r, method=m, **kw)
            err = wp.relative_output_error(mlp, new_mlp, x)
            print(f"{m:<14}{r:>7.2f}{st.original_inter:>5}->{st.pruned_inter:<4}"
                  f"{st.pruned_params:>12,}{st.param_reduction_pct:>7.1f}%{err:>10.4f}")
    print("-" * 68)
    print("观察：同一 ratio 下三种评分参数量一致（结构一样），但保留的神经元不同 → 误差不同。")
    print("      误差随 ratio 单调上升；重要神经元被优先保留，故小剪枝比误差很小。")
    print("=" * 68 + "\n")


def main():
    hidden, inter = 64, 512
    mlp = build_demo_mlp(hidden, inter)
    x = torch.randn(8, 16, hidden)                 # 评估用固定输入
    dataloader = build_calibration_data(hidden)    # hybrid 校准数据

    print_report(mlp, x, dataloader)

    fig_error_vs_ratio(mlp, x, dataloader)
    fig_importance_hist(mlp, prune_ratio=0.5)
    fig_param_vs_error(mlp, x)
    fig_attention_heads()

    print("\n完成！4 张图已保存到当前目录。用图片查看器打开 fig1~fig4 即可。")


if __name__ == "__main__":
    main()
