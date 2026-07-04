# -*- coding: utf-8 -*-
"""
run_demo.py — 深度剪枝 lab 的可视化 demo。

产出两张图(保存到 ./figures/):
  1) block 重要性柱状图(Block Influence),标出保护区 & 被选中删除的块。
  2) 剪枝比例 vs 参数量 / FLOPs 曲线,验证「删得越多、参数/计算越少」。

同时在终端打印:重要性分数、被选块、剪前后参数量/FLOPs 对比、保留层一致性校验。

⚠️ 全程离线、CPU、不下载任何模型。matplotlib 用 Agg 后端(无需 GUI),中文用微软雅黑。
"""
from __future__ import annotations

import os
import sys

# Windows 控制台默认 GBK,直接 print 中文/emoji 会 UnicodeEncodeError。
# 把 stdout/stderr 重配成 UTF-8(Python 3.7+ 支持 reconfigure)。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")                      # 无界面后端:服务器/CI 也能出图
import matplotlib.pyplot as plt
import numpy as np
import torch

# 中文字体 & 负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from toy_transformer import ToyConfig, build_toy_model, make_toy_batch
from depth_pruning import (
    calculate_layer_importance_cosine,
    count_params,
    estimate_flops,
    params_breakdown,
    protected_layer_set,
    prune_model,
    select_layers_to_prune,
    setup_layer_hooks,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def banner(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main():
    cfg = ToyConfig(n_layers=12, d_model=64, n_heads=4, seed=0)
    # vary_blocks=True:制造有高低差的重要性分布,让柱状图像书里图 4.6 那样直观。
    model = build_toy_model(cfg, vary_blocks=True)

    banner("① 玩具模型概况")
    bd = params_breakdown(model)
    print(f"block 数(深度)      : {model.n_layers}")
    print(f"隐藏维 d_model        : {cfg.d_model}")
    print(f"总参数量              : {bd['total']:,}")
    print(f"  其中 block 部分      : {bd['blocks_total']:,}  (每块 {bd['per_block']:,})")
    print(f"  其中非 block 部分    : {bd['non_block']:,}  (embed / lm_head / final norm)")

    # ---------------- 数据驱动重要性 ----------------
    banner("② 数据驱动重要性(Block Influence = 1 - 余弦相似度)")
    batches = [make_toy_batch(cfg, batch=4, seq=24, seed=s) for s in range(6)]
    importance = calculate_layer_importance_cosine(model, batches)
    for idx, score in importance.items():
        bar = "█" * int(score / max(importance.values()) * 30 + 0.5)
        print(f"  block {idx:2d}  BI={score:.4f}  {bar}")

    # ---------------- 选块 & 剪枝 ----------------
    banner("③ 选出「最不重要」的块并剪枝(保护首尾 + 避免相邻)")
    n_prune = 3
    to_prune = select_layers_to_prune(importance, num_layers_to_prune=n_prune)
    protected = protected_layer_set(model.n_layers)
    print(f"保护区(不可删)      : {sorted(protected)}")
    print(f"选中删除的 block      : {to_prune}  (共 {len(to_prune)} 块)")

    pruned = prune_model(model, to_prune)
    print(f"剪前层数              : {model.n_layers}   剪后层数: {pruned.n_layers}")
    print(f"剪前参数量            : {count_params(model):,}")
    print(f"剪后参数量            : {count_params(pruned):,}"
          f"   (降 {(1 - count_params(pruned) / count_params(model)) * 100:.1f}%)")
    f_full = estimate_flops(model, seq_len=24)
    f_pruned = estimate_flops(pruned, seq_len=24)
    print(f"剪前 FLOPs(单前向)   : {f_full:,}")
    print(f"剪后 FLOPs            : {f_pruned:,}"
          f"   (降 {(1 - f_pruned / f_full) * 100:.1f}%)")

    # ---------------- 保留层一致性校验 ----------------
    banner("④ 一致性校验:剪掉后面的块不影响前面保留块的输出")
    x = make_toy_batch(cfg, batch=2, seq=24, seed=999)
    h1, _, o1, n = setup_layer_hooks(model)
    with torch.no_grad():
        model(x)
    orig = {i: o1[i].clone() for i in range(n)}
    for h in h1:
        h.remove()
    h2, _, o2, _ = setup_layer_hooks(pruned)
    with torch.no_grad():
        pruned(x)
    first_cut = min(to_prune)
    ok = all(torch.equal(o2[i], orig[i]) for i in range(first_cut))
    print(f"最早被删块            : {first_cut}")
    print(f"block 0..{first_cut - 1} 与原模型逐位相等 : {ok}  "
          f"({'一致 [OK]' if ok else '不一致 [FAIL]'})")
    for h in h2:
        h.remove()

    # ================= 图 1:重要性柱状图 =================
    banner("⑤ 出图 1:block 重要性柱状图")
    idxs = list(importance.keys())
    vals = [importance[i] for i in idxs]
    colors = []
    for i in idxs:
        if i in to_prune:
            colors.append("#e53935")        # 红:被删
        elif i in protected:
            colors.append("#43a047")        # 绿:保护区
        else:
            colors.append("#90a4ae")        # 灰:候选但未删

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(idxs, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("block 编号")
    ax.set_ylabel("Block Influence (BI = 1 − cos)")
    ax.set_title("各 block 重要性(绿=保护首尾，红=选中删除，灰=候选未删)")
    ax.set_xticks(idxs)
    # 图例
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#43a047", label="保护区(首尾)"),
        Patch(color="#e53935", label="选中删除"),
        Patch(color="#90a4ae", label="候选未删"),
    ], loc="upper center")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "01_block_importance.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"已保存: {p1}")

    # ================= 图 2:剪枝比例 vs 参数量/FLOPs =================
    banner("⑥ 出图 2:剪枝比例 vs 参数量 / FLOPs")
    total_blocks = model.n_layers
    ks = list(range(0, total_blocks - protect_count(cfg) + 1))  # 从删 0 到删满候选
    ratios, param_list, flops_list = [], [], []
    base_params = count_params(model)
    base_flops = estimate_flops(model, seq_len=24)
    # 逐 k:按重要性升序、带保护地删 k 块,记录参数量/FLOPs。
    for k in ks:
        if k == 0:
            pk = model
        else:
            sel = select_layers_to_prune(importance, num_layers_to_prune=k)
            if len(sel) < k:                # 保护约束下可选块不够了,停止
                break
            pk = prune_model(model, sel)
        ratios.append(k / total_blocks * 100)
        param_list.append(count_params(pk) / base_params * 100)
        flops_list.append(estimate_flops(pk, seq_len=24) / base_flops * 100)

    fig2, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ratios, param_list, "o-", color="#1e88e5", label="参数量(占原始 %)")
    ax1.plot(ratios, flops_list, "s--", color="#fb8c00", label="FLOPs(占原始 %)")
    ax1.set_xlabel("剪枝比例(删除 block 数 / 总 block 数, %)")
    ax1.set_ylabel("相对原始模型(%)")
    ax1.set_title("剪枝比例 vs 参数量 / FLOPs(删得越多，越小越快)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    for x_, y_ in zip(ratios, param_list):
        ax1.annotate(f"{y_:.0f}%", (x_, y_), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7, color="#1e88e5")
    fig2.tight_layout()
    p2 = os.path.join(FIG_DIR, "02_pruning_ratio_vs_size.png")
    fig2.savefig(p2, dpi=120)
    plt.close(fig2)
    print(f"已保存: {p2}")

    banner("完成 ✅")
    print(f"两张图已写入: {FIG_DIR}")


def protect_count(cfg: ToyConfig) -> int:
    """保护区块数 = 前 4 + 后 1(与 protected_layer_set 默认一致)。"""
    return 5


if __name__ == "__main__":
    main()
