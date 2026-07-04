# -*- coding: utf-8 -*-
"""
test_depth_pruning.py — 深度剪枝 lab 的 pytest 用例。

覆盖硬性要求:
  1) 删层后参数量按比例降(per-block 均摊)。
  2) 剪后模型形状正确、前向可跑。
  3) 重要性排序确定(determinism)。
外加:
  4) 保护启发式生效(首尾不被删)。
  5) 保留层输出一致性(剪掉后面的块不影响前面块的输出)。
  6) FLOPs 随删块下降;prune_model 不改原模型;越界报错等边界。
"""
import os
import sys

import numpy as np
import pytest
import torch

# 让测试无论从哪跑都能 import 到项目模块。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toy_transformer import ToyConfig, build_toy_model, make_toy_batch  # noqa: E402
from depth_pruning import (  # noqa: E402
    calculate_cosine_importance,
    calculate_layer_importance_cosine,
    count_params,
    count_block_params,
    estimate_flops,
    params_breakdown,
    protected_layer_set,
    prune_model,
    select_layers_to_prune,
    setup_layer_hooks,
)


@pytest.fixture
def cfg():
    return ToyConfig(n_layers=12, d_model=64, n_heads=4, seed=0)


@pytest.fixture
def model(cfg):
    return build_toy_model(cfg)


@pytest.fixture
def batches(cfg):
    return [make_toy_batch(cfg, batch=2, seq=16, seed=s) for s in range(4)]


# --------------------------------------------------------------------------------------
# 1) 参数量按比例下降
# --------------------------------------------------------------------------------------
def test_pruning_reduces_params_proportionally(model):
    """删 k 个 block => 参数量应精确下降 = 被删块参数之和;且约等于 k * per_block。"""
    bd = params_breakdown(model)
    per_block = bd["per_block"]

    full = count_params(model)
    to_remove = [4, 6, 8]
    # 被删块的真实参数和(玩具模型里每块结构相同,应精确等于 k * per_block)。
    removed_params = sum(count_block_params(model.layers[i]) for i in to_remove)

    pruned = prune_model(model, to_remove)
    pruned_total = count_params(pruned)

    assert full - pruned_total == removed_params
    # 结构同构 => 每块参数量相等 => 删 3 块正好等于 3 * per_block。
    assert removed_params == 3 * per_block
    # 剪后非 block 部分(embed/head/norm)完全不变。
    assert params_breakdown(pruned)["non_block"] == bd["non_block"]


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_param_drop_scales_with_k(model, k):
    """删的块越多,下降越多,且严格 = k * per_block。"""
    per_block = params_breakdown(model)["per_block"]
    full = count_params(model)
    pruned = prune_model(model, list(range(4, 4 + k)))  # 删中段连续 k 块(避开首尾保护无所谓,这里直接指定)
    assert full - count_params(pruned) == k * per_block


# --------------------------------------------------------------------------------------
# 2) 剪后形状正确 + 前向可跑
# --------------------------------------------------------------------------------------
def test_pruned_model_forward_shape(cfg, model):
    pruned = prune_model(model, [5, 7])
    assert pruned.n_layers == model.n_layers - 2

    x = make_toy_batch(cfg, batch=3, seq=20)
    out = pruned(x)
    # logits 形状必须是 [batch, seq, vocab],和层数无关。
    assert tuple(out.logits.shape) == (3, 20, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_pruned_model_loss_runs(cfg, model):
    pruned = prune_model(model, [6])
    x = make_toy_batch(cfg, batch=2, seq=12)
    out = pruned(x, labels=x)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert torch.isfinite(out.loss)


def test_layer_count_matches_module_list(model):
    """n_layers 属性必须始终等于 ModuleList 长度(剪枝后也是)。"""
    pruned = prune_model(model, [4, 5, 6])
    assert pruned.n_layers == len(pruned.layers) == 9


# --------------------------------------------------------------------------------------
# 3) 重要性排序确定(determinism)
# --------------------------------------------------------------------------------------
def test_importance_is_deterministic(cfg, batches):
    """同一模型、同一批数据,两次算出的重要性必须逐块相等。"""
    m1 = build_toy_model(cfg)
    m2 = build_toy_model(cfg)
    imp1 = calculate_layer_importance_cosine(m1, batches)
    imp2 = calculate_layer_importance_cosine(m2, batches)
    assert imp1.keys() == imp2.keys()
    for k in imp1:
        assert imp1[k] == pytest.approx(imp2[k], abs=1e-12)


def test_selection_is_deterministic(cfg, batches):
    """选块结果必须完全可复现(逐次调用相同)。"""
    m = build_toy_model(cfg)
    imp = calculate_layer_importance_cosine(m, batches)
    sel1 = select_layers_to_prune(imp, num_layers_to_prune=3)
    sel2 = select_layers_to_prune(imp, num_layers_to_prune=3)
    assert sel1 == sel2


def test_selection_orders_by_importance(cfg):
    """
    构造一个「已知重要性字典」,验证选块严格从低分往高分挑(排序正确)。
    人造分数:让块 5、9、7 最低,应被优先选中(受相邻保护影响不选相邻)。
    """
    scores = {i: float(i) for i in range(12)}   # 分数 = 块号,单调递增
    # 关掉保护,验证纯排序:最小的应是 0,1,2...
    sel = select_layers_to_prune(scores, num_layers_to_prune=3,
                                 heuristic_protection=False, adjacent_protection=False)
    assert sel == [0, 1, 2]


def test_selection_stable_on_ties(cfg):
    """分数完全相同时,必须用块号打破平局 => 结果确定(升序块号)。"""
    scores = {i: 0.5 for i in range(12)}        # 所有块分数一样
    sel = select_layers_to_prune(scores, num_layers_to_prune=3,
                                 heuristic_protection=False, adjacent_protection=False)
    assert sel == [0, 1, 2]                      # 平局 -> 按块号升序


# --------------------------------------------------------------------------------------
# 4) 保护启发式
# --------------------------------------------------------------------------------------
def test_protection_set():
    prot = protected_layer_set(12, protect_front=4, protect_back=1)
    assert prot == {0, 1, 2, 3, 11}


def test_heuristic_protection_never_prunes_ends(cfg):
    """开启保护时,选出的块绝不包含首 4 或末 1。"""
    # 人造:让首尾块分数最低,诱使算法想删它们 —— 保护应挡住。
    scores = {i: 1.0 for i in range(12)}
    for i in (0, 1, 2, 3, 11):
        scores[i] = 0.0
    sel = select_layers_to_prune(scores, num_layers_to_prune=3,
                                 heuristic_protection=True, adjacent_protection=False)
    protected = protected_layer_set(12)
    assert all(s not in protected for s in sel)


def test_adjacent_protection_avoids_consecutive(cfg):
    """开启相邻保护时,选出的块两两不相邻。"""
    scores = {i: float(i) for i in range(12)}    # 单调,最小的是 4,5,6...(0-3 被保护)
    sel = select_layers_to_prune(scores, num_layers_to_prune=3,
                                 heuristic_protection=True, adjacent_protection=True)
    sel_sorted = sorted(sel)
    for a, b in zip(sel_sorted, sel_sorted[1:]):
        assert b - a >= 2                        # 任意两选中块间隔 >= 2


# --------------------------------------------------------------------------------------
# 5) 保留层输出一致性
# --------------------------------------------------------------------------------------
def test_retained_layers_before_cut_are_bit_identical(cfg, model):
    """
    第一性质:剪掉某些后面的块,不会影响它们「之前」的块的输出。
    即最早被删块之前的所有保留块,输出应和原模型逐位相等。
    """
    x = make_toy_batch(cfg, batch=2, seq=16)

    h1, ins1, outs1, n = setup_layer_hooks(model)
    with torch.no_grad():
        model(x)
    orig = {i: outs1[i].clone() for i in range(n)}
    for h in h1:
        h.remove()

    cut = [4, 6, 9]           # 最早被删的是 block 4
    pruned = prune_model(model, cut)
    h2, ins2, outs2, n2 = setup_layer_hooks(pruned)
    with torch.no_grad():
        pruned(x)
    # 剪后 block 0..3 对应原模型 block 0..3(未受任何影响)。
    for i in range(min(cut)):
        assert torch.equal(outs2[i], orig[i]), f"保留块 {i} 输出应与原模型完全一致"
    for h in h2:
        h.remove()


def test_cosine_importance_bounds(cfg, batches):
    """BI = 1 - cos,cos ∈ [-1,1] => 重要性 ∈ [0,2];随机初始残差模型应接近 0。"""
    m = build_toy_model(cfg)
    imp = calculate_layer_importance_cosine(m, batches)
    for v in imp.values():
        assert 0.0 <= v <= 2.0


def test_calculate_cosine_importance_edge_cases():
    """空张量返回 0;相同输入输出 => cos=1 => 重要性≈0;方向相反 => 重要性≈2。"""
    empty = torch.zeros(0, 4)
    assert calculate_cosine_importance(empty, empty) == 0.0

    v = torch.randn(3, 8)
    assert calculate_cosine_importance(v, v) == pytest.approx(0.0, abs=1e-5)
    assert calculate_cosine_importance(v, -v) == pytest.approx(2.0, abs=1e-5)


# --------------------------------------------------------------------------------------
# 6) FLOPs / 边界 / 不改原模型
# --------------------------------------------------------------------------------------
def test_flops_decrease_with_pruning(model):
    full = estimate_flops(model, seq_len=16)
    pruned = prune_model(model, [4, 5, 6])
    less = estimate_flops(pruned, seq_len=16)
    assert less < full
    # 删 3/12 块 => block 部分 FLOPs 降约 25%(非 block 部分不变,故总降略小于 25%)。
    ratio = (full - less) / full
    assert 0.15 < ratio < 0.25


def test_prune_does_not_mutate_original_by_default(model):
    before = model.n_layers
    _ = prune_model(model, [3, 4, 5])           # inplace=False
    assert model.n_layers == before             # 原模型层数不变


def test_prune_inplace_mutates(model):
    before = model.n_layers
    ret = prune_model(model, [3, 4], inplace=True)
    assert ret is model
    assert model.n_layers == before - 2


def test_prune_out_of_range_raises(model):
    with pytest.raises(IndexError):
        prune_model(model, [999])


def test_prune_duplicate_indices_dedup(model):
    """重复块号应去重,只删一次。"""
    before = model.n_layers
    pruned = prune_model(model, [5, 5, 5])
    assert pruned.n_layers == before - 1


def test_vary_blocks_profile_creates_spread(cfg, batches):
    """vary_blocks=True 时,首尾块 BI 应显著高于中段冗余块(证明重要性有梯度)。"""
    m = build_toy_model(cfg, vary_blocks=True)
    imp = calculate_layer_importance_cosine(m, batches)
    front_back = np.mean([imp[i] for i in (0, 1, 2, 11)])
    middle_low = np.mean([imp[i] for i in (6, 8)])
    assert front_back > middle_low * 5          # 首尾至少比中段冗余块高 5 倍
