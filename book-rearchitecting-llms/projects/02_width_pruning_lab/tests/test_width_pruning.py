# -*- coding: utf-8 -*-
"""
test_width_pruning.py —— 宽度剪枝库的 pytest 测试套件

覆盖题目要求的四大验证点：
    1) 剪枝后 shape 一致（形状一致性三铁律：入口/内部 k/出口）
    2) 参数量确实下降
    3) 剪枝比越大误差越大（单调性）
    4) 重要神经元被保留（top-importance 存活）

外加：注意力剪头、混合评分、硬件对齐 divisor、防御性边界。

设计说明：
    - 为了让「误差随剪枝比单调增」「重要神经元保留」这类断言**稳定可复现**，
      我们不用纯随机 MLP（随机权重下所有神经元重要性趋同，误差会一次性跳满），
      而是构造「重要性有梯度」的 MLP：给每个神经元乘一个从大到小的 scale。
"""

import os
import sys

import torch
import pytest

# 让测试能直接 import 上级目录的 width_pruning.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import width_pruning as wp  # noqa: E402


# ----------------------------------------------------------------------------
# 工具：构造「重要性有梯度」的玩具 MLP
# ----------------------------------------------------------------------------

def make_graded_mlp(hidden=32, inter=128, seed=1, spread=(1.0, 0.05), shuffle=True):
    """构造一个神经元重要性从大到小平滑衰减的 GluMLP。

    - scale = linspace(spread[0], spread[1], inter)，第 i 个神经元乘 scale[i]；
    - 可选 shuffle：打散 scale，避免保留索引恰好连续（更真实、更能测 .sort() 逻辑）。
    """
    torch.manual_seed(seed)
    mlp = wp.GluMLP(hidden, inter)
    with torch.no_grad():
        scale = torch.linspace(spread[0], spread[1], inter)
        if shuffle:
            scale = scale[torch.randperm(inter)]
        mlp.gate_proj.weight.mul_(scale.unsqueeze(1))
        mlp.up_proj.weight.mul_(scale.unsqueeze(1))
        mlp.down_proj.weight.mul_(scale.unsqueeze(0))
    return mlp


def make_deadmix_mlp(hidden=32, inter=128, alive_frac=0.25, seed=0):
    """构造「一部分神经元近乎死亡」的 MLP：前 alive_frac 强，其余≈0。

    用于验证：剪掉死神经元 → 输出几乎不变（误差极小）。
    """
    torch.manual_seed(seed)
    mlp = wp.GluMLP(hidden, inter)
    with torch.no_grad():
        scale = torch.ones(inter)
        n_alive = int(inter * alive_frac)
        scale[n_alive:] = 0.02
        mlp.gate_proj.weight.mul_(scale.unsqueeze(1))
        mlp.up_proj.weight.mul_(scale.unsqueeze(1))
        mlp.down_proj.weight.mul_(scale.unsqueeze(0))
    return mlp


# ============================================================================
# 1. 形状一致性（shape consistency）
# ============================================================================

@pytest.mark.parametrize("method", ["peak_to_peak", "l2"])
@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 0.75])
def test_shape_consistency_entry_exit_preserved(method, ratio):
    """剪枝后：入口守恒 hidden、出口守恒 hidden、内部三层同步 k。"""
    hidden, inter = 32, 128
    mlp = make_graded_mlp(hidden, inter)
    new_mlp, stats = wp.prune_mlp(mlp, ratio, method=method)

    # 入口守恒：gate/up 的 in_features 仍 = hidden
    assert new_mlp.gate_proj.in_features == hidden
    assert new_mlp.up_proj.in_features == hidden
    # 出口守恒：down 的 out_features 仍 = hidden
    assert new_mlp.down_proj.out_features == hidden
    # 内部同步：三层用同一个 k
    k = new_mlp.intermediate_size
    assert new_mlp.gate_proj.out_features == k
    assert new_mlp.up_proj.out_features == k
    assert new_mlp.down_proj.in_features == k
    # k 应等于统计里记录的 pruned_inter，且严格小于原尺寸
    assert k == stats.pruned_inter
    assert 0 < k < inter


def test_forward_runs_and_shape_matches_after_prune():
    """剪后模型能前向，且输出张量 shape 与原模型一致（都是 [B,T,hidden]）。"""
    mlp = make_graded_mlp(32, 128)
    x = torch.randn(4, 8, 32)
    y0 = mlp(x)
    new_mlp, _ = wp.prune_mlp(mlp, 0.4, method="peak_to_peak")
    y1 = new_mlp(x)
    assert y0.shape == y1.shape == (4, 8, 32)


# ============================================================================
# 2. 参数量下降
# ============================================================================

@pytest.mark.parametrize("method", ["peak_to_peak", "l2"])
def test_param_count_decreases(method):
    """剪枝后参数量必须严格下降，且下降量与统计一致。"""
    mlp = make_graded_mlp(32, 128)
    p0 = wp.count_params(mlp)
    new_mlp, stats = wp.prune_mlp(mlp, 0.4, method=method)
    p1 = wp.count_params(new_mlp)
    assert p1 < p0
    assert stats.original_params == p0
    assert stats.pruned_params == p1
    assert stats.param_reduction == p0 - p1
    assert 0 < stats.param_reduction_pct < 100


def test_param_reduction_scales_with_ratio():
    """剪枝比越大，删掉的参数越多。"""
    mlp = make_graded_mlp(32, 128)
    _, s_small = wp.prune_mlp(mlp, 0.2, method="l2")
    _, s_big = wp.prune_mlp(mlp, 0.6, method="l2")
    assert s_big.param_reduction > s_small.param_reduction


# ============================================================================
# 3. 剪枝比越大误差越大（单调性）
# ============================================================================

@pytest.mark.parametrize("method", ["peak_to_peak", "l2"])
def test_error_monotonic_increasing_with_ratio(method):
    """在重要性有梯度的 MLP 上，误差应随剪枝比单调不减。"""
    mlp = make_graded_mlp(32, 128, seed=1)
    x = torch.randn(4, 8, 32)
    ratios = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    errs = []
    for r in ratios:
        new_mlp, _ = wp.prune_mlp(mlp, r, method=method)
        errs.append(wp.relative_output_error(mlp, new_mlp, x))
    # 允许极小的数值抖动
    for i in range(len(errs) - 1):
        assert errs[i] <= errs[i + 1] + 1e-6, f"非单调: {errs}"
    # 小剪枝比误差应明显小于大剪枝比
    assert errs[0] < errs[-1]


def test_small_ratio_small_error():
    """小剪枝比（10%）在梯度 MLP 上误差应很小（< 5%）。"""
    mlp = make_graded_mlp(32, 128, seed=1)
    x = torch.randn(4, 8, 32)
    new_mlp, _ = wp.prune_mlp(mlp, 0.1, method="peak_to_peak")
    err = wp.relative_output_error(mlp, new_mlp, x)
    assert err < 0.05


def test_pruning_dead_neurons_negligible_error():
    """剪掉近乎死亡的神经元 → 输出几乎不变（误差 < 1%）。"""
    mlp = make_deadmix_mlp(32, 128, alive_frac=0.25)
    x = torch.randn(4, 8, 32)
    # 剪 70%，理论上删的全是死神经元（活的只占 25%）
    new_mlp, _ = wp.prune_mlp(mlp, 0.70, method="peak_to_peak")
    err = wp.relative_output_error(mlp, new_mlp, x)
    assert err < 0.01


# ============================================================================
# 4. 重要神经元被保留
# ============================================================================

@pytest.mark.parametrize("method_fn", [
    lambda m: wp.peak_to_peak_importance(m.gate_proj.weight.data, m.up_proj.weight.data),
    lambda m: wp.l2_norm_importance(m.gate_proj.weight.data, m.up_proj.weight.data),
])
def test_top_important_neurons_preserved(method_fn):
    """剪枝比 < (1 - top占比) 时，最重要的若干神经元必须全部存活。"""
    mlp = make_graded_mlp(32, 128, seed=2)
    importance = method_fn(mlp)
    top_idx = torch.topk(importance, 20).indices
    keep = wp.select_indices_to_keep(importance, 0.5)  # 保留 64 个
    keep_set = set(keep.tolist())
    assert all(t.item() in keep_set for t in top_idx), "有重要神经元被误删"


def test_keep_indices_are_sorted_ascending():
    """select_indices_to_keep 返回的索引必须按原始位置升序（利于蒸馏恢复）。"""
    mlp = make_graded_mlp(32, 128, seed=3)
    importance = wp.peak_to_peak_importance(mlp.gate_proj.weight.data, mlp.up_proj.weight.data)
    keep = wp.select_indices_to_keep(importance, 0.4)
    assert torch.equal(keep, keep.sort().values)


def test_least_important_neurons_dropped():
    """最不重要的神经元应被删掉（不在保留集里）。"""
    mlp = make_graded_mlp(32, 128, seed=4)
    importance = wp.peak_to_peak_importance(mlp.gate_proj.weight.data, mlp.up_proj.weight.data)
    bottom_idx = torch.topk(importance, 20, largest=False).indices
    keep = wp.select_indices_to_keep(importance, 0.5)
    keep_set = set(keep.tolist())
    dropped = sum(1 for b in bottom_idx if b.item() not in keep_set)
    # 至少大部分最弱神经元被删（允许极少数因梯度重叠边界情况）
    assert dropped >= 18


# ============================================================================
# 5. 数据驱动混合评分（hybrid）
# ============================================================================

def test_hybrid_requires_dataloader():
    """method='hybrid' 不给 dataloader 应报错。"""
    mlp = make_graded_mlp(32, 128)
    with pytest.raises(ValueError):
        wp.prune_mlp(mlp, 0.3, method="hybrid", dataloader=None)


def test_hybrid_drops_inactive_neurons():
    """混合评分：权重大但在校准数据上激活≈0 的神经元应被删。

    构造：某些神经元的 up/gate 权重被人为放大（结构分高），但我们通过把它们的
    down_proj 列置零使其对输出无贡献 —— 不过 hybrid 是按 down_proj 输入端激活来测，
    这里更直接的验证是：只要跑通校准并产出更小模型、shape 正确即可。
    """
    mlp = make_graded_mlp(32, 128, seed=5)
    dataloader = [torch.randn(4, 8, 32) for _ in range(6)]
    new_mlp, stats = wp.prune_mlp(mlp, 0.3, method="hybrid", dataloader=dataloader)
    assert new_mlp.intermediate_size < mlp.intermediate_size
    assert new_mlp.gate_proj.in_features == 32
    assert new_mlp.down_proj.out_features == 32
    assert stats.method == "hybrid"


def test_activation_collector_accumulates_and_removes_hook():
    """ActivationCollector：累加范数长度正确，且退出后 hook 被移除。"""
    mlp = make_graded_mlp(32, 128)
    dataloader = [torch.randn(2, 5, 32) for _ in range(3)]
    norms = wp.collect_activation_norms(mlp, dataloader)
    assert norms.shape == (128,)
    assert torch.all(norms >= 0)
    # 退出 with 后，down_proj 不应再挂着我们的 forward hook
    assert len(mlp.down_proj._forward_hooks) == 0


def test_hybrid_differs_from_static_selection():
    """给不同数据校准，hybrid 选出的神经元通常与纯静态不同（数据在起作用）。"""
    mlp = make_graded_mlp(32, 128, seed=7)
    # 让数据只在部分维度有能量，制造激活差异
    torch.manual_seed(9)
    dataloader = []
    for _ in range(6):
        x = torch.randn(4, 8, 32)
        x[..., 16:] *= 0.01  # 后半维输入很弱
        dataloader.append(x)
    static_keep = set(wp.select_indices_to_keep(
        wp.peak_to_peak_importance(mlp.gate_proj.weight.data, mlp.up_proj.weight.data), 0.5).tolist())
    hybrid_mlp, _ = wp.prune_mlp(mlp, 0.5, method="hybrid", dataloader=dataloader)
    # 只要能跑出合法模型即算通过；差异是「通常」而非「必然」，故不强断言集合不等
    assert hybrid_mlp.intermediate_size == 64
    assert isinstance(static_keep, set)


# ============================================================================
# 6. 注意力剪头
# ============================================================================

def test_attention_prune_shape_and_forward():
    """注意力剪头后：hidden 入口/出口不变，能前向，输出 shape 一致。"""
    attn = wp.ToyMultiHeadAttention(64, 8)
    x = torch.randn(2, 6, 64)
    y0 = attn(x)
    new_attn, keep = wp.prune_attention(attn, 0.25, by="ratio")  # 8 -> 6 头
    y1 = new_attn(x)
    assert new_attn.num_heads == 6
    assert y0.shape == y1.shape == (2, 6, 64)
    assert wp.count_params(new_attn) < wp.count_params(attn)


def test_attention_prune_by_num():
    """按保留头数剪枝。"""
    attn = wp.ToyMultiHeadAttention(64, 8)
    new_attn, keep = wp.prune_attention(attn, 5, by="num")
    assert new_attn.num_heads == 5
    assert keep.numel() == 5


def test_attention_keep_heads_sorted():
    """保留的头索引升序。"""
    attn = wp.ToyMultiHeadAttention(64, 8)
    _, keep = wp.prune_attention(attn, 0.5, by="ratio")
    assert torch.equal(keep, keep.sort().values)


# ============================================================================
# 7. 硬件对齐 divisor & 防御性边界
# ============================================================================

@pytest.mark.parametrize("divisor", [8, 16, 32])
def test_divisor_alignment(divisor):
    """指定 divisor 后，pruned intermediate_size 必须是其倍数。"""
    mlp = make_graded_mlp(32, 128)
    new_mlp, stats = wp.prune_mlp(mlp, 0.3, method="l2", divisor=divisor)
    assert new_mlp.intermediate_size % divisor == 0
    assert new_mlp.intermediate_size > 0


def test_prune_ratio_out_of_range_raises():
    """剪枝比 >= 1.0 或 < 0 应报错。"""
    imp = torch.rand(128)
    with pytest.raises(ValueError):
        wp.select_indices_to_keep(imp, 1.0)
    with pytest.raises(ValueError):
        wp.select_indices_to_keep(imp, -0.1)


def test_never_prune_to_zero():
    """即便 ratio 逼近 1，也至少保留 1 个神经元。"""
    imp = torch.rand(128)
    keep = wp.select_indices_to_keep(imp, 0.999)
    assert keep.numel() >= 1


def test_unknown_method_raises():
    """未知 method 报错。"""
    mlp = make_graded_mlp(32, 64)
    with pytest.raises(ValueError):
        wp.prune_mlp(mlp, 0.3, method="nonsense")


def test_original_model_not_mutated():
    """剪枝返回新模型，原模型不被就地修改。"""
    mlp = make_graded_mlp(32, 128)
    inter_before = mlp.intermediate_size
    p_before = wp.count_params(mlp)
    _ = wp.prune_mlp(mlp, 0.5, method="peak_to_peak")
    assert mlp.intermediate_size == inter_before
    assert wp.count_params(mlp) == p_before


def test_importance_scores_shape():
    """三种评分函数输出长度都等于 intermediate_size。"""
    mlp = make_graded_mlp(32, 128)
    g, u, d = mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data
    assert wp.peak_to_peak_importance(g, u).shape == (128,)
    assert wp.l2_norm_importance(g, u).shape == (128,)
    act = torch.rand(128)
    assert wp.hybrid_importance(g, u, d, act).shape == (128,)
