# -*- coding: utf-8 -*-
"""
depth_pruning.py — 深度剪枝(depth pruning)核心逻辑。

对齐《Rearchitecting LLMs》第 4 章的四件套:
  1) PyTorch forward hook 探针        —— 不改模型代码,捕获每个 block 的输入/输出激活。
  2) Block Influence(BI)= 1 - cos    —— 数据驱动重要性指标(= ShortGPT 的 BI)。
  3) 保护启发式 + 相邻保护 选块        —— 从「最不重要」里挑,但保护首尾、避免删连续段。
  4) prune_model:重建更短的 layers    —— 深度剪枝在代码上就是「切片 + 拼接」。

外加两个可量化的收益指标:
  - count_params:参数量(总量 & 每 block 均摊),验证「删 k 块 => 参数按比例降」。
  - estimate_flops:单次前向的近似 FLOPs,验证「删块 => 计算量下降」。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 1) Hook 探针:捕获每个 block 的输入 & 输出激活
# ======================================================================================
def setup_layer_hooks(model):
    """
    给 model.layers 里的每个 block 挂一个 forward hook,把「输入激活/输出激活」存进字典。

    返回 (hooks, layer_inputs, layer_outputs, num_layers):
      - hooks:        句柄列表,用完要 hook.remove() 摘掉。
      - layer_inputs: {块号 -> 该块的输入张量}
      - layer_outputs:{块号 -> 该块的输出张量}
    """
    num_layers = len(model.layers)
    layer_inputs: dict[int, torch.Tensor] = {}
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = []

    # 工厂函数(闭包):每次调用「记住」当时的 idx,避免 28 个 hook 都写进同一个键。
    def create_hook(layer_idx: int) -> Callable:
        def hook(module, inp, out):
            # forward hook 固定签名 (module, input, output)。
            # input 一定是元组,output 可能是张量或元组 —— 都要判断后取张量。
            input_tensor = inp[0] if isinstance(inp, (tuple, list)) else inp
            output_tensor = out[0] if isinstance(out, (tuple, list)) else out
            # .detach():脱离计算图,阻止梯度累积、省显存(我们只观测不训练)。
            layer_inputs[layer_idx] = input_tensor.detach()
            layer_outputs[layer_idx] = output_tensor.detach()
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(create_hook(i)))
    return hooks, layer_inputs, layer_outputs, num_layers


# ======================================================================================
# 2) Block Influence:importance = 1 - 平均余弦相似度(输入 vs 输出)
# ======================================================================================
def calculate_cosine_importance(input_tensor: torch.Tensor,
                                output_tensor: torch.Tensor,
                                layer_idx: int = -1) -> float:
    """
    衡量「一个 block 把信息改动了多少」:
        importance = 1 - mean(cos(input, output))
    相似度高(≈1) => block 几乎没改动 => 重要性低(打酱油,可删)。
    相似度低      => 改动大           => 重要性高(干实事,别删)。

    这正是 ShortGPT(Men et al., 2024)里的 Block Influence(BI)指标。
    """
    if input_tensor.numel() == 0 or output_tensor.numel() == 0:
        return 0.0

    # 展平成 [batch, 特征]:每个样本变成一个长向量,逐样本算余弦。
    input_flat = input_tensor.reshape(input_tensor.size(0), -1)
    output_flat = output_tensor.reshape(output_tensor.size(0), -1)

    similarities = F.cosine_similarity(input_flat, output_flat, dim=1)

    # 数值鲁棒:滤掉 inf/nan(高维运算偶尔溢出)。
    finite = similarities[torch.isfinite(similarities)]
    if finite.numel() == 0:
        return 0.0

    return 1.0 - finite.mean().item()


def calculate_layer_importance_cosine(model, batches, device: str = "cpu") -> dict[int, float]:
    """
    完整重要性流水线:挂钩子 -> 逐 batch 前向(钩子自动填激活)-> 逐 block 算 BI ->
    跨 batch 求平均 -> 返回 {块号: 重要性}。

    `batches`:可迭代对象,每个元素是 input_ids 张量 [B, S](玩具版直接喂 token id)。
    """
    hooks, layer_inputs, layer_outputs, num_layers = setup_layer_hooks(model)
    scores: dict[int, list[float]] = {i: [] for i in range(num_layers)}

    with torch.no_grad():                      # 不要梯度
        for input_ids in batches:
            input_ids = input_ids.to(device)
            model(input_ids)                   # 跑前向,hook 自动把激活填进字典

            for idx in range(num_layers):
                if idx not in layer_inputs or idx not in layer_outputs:
                    scores[idx].append(0.0)
                    continue
                scores[idx].append(
                    calculate_cosine_importance(layer_inputs[idx], layer_outputs[idx], idx)
                )

            layer_inputs.clear()               # 清掉本 batch 激活,省内存
            layer_outputs.clear()

    for h in hooks:
        h.remove()                             # 用完摘钩子,别让它每次前向都白跑

    final: dict[int, float] = {}
    for idx, lst in scores.items():
        valid = [s for s in lst if np.isfinite(s)]
        if not valid:
            raise RuntimeError(f"Block {idx} 没被捕获到,hook 可能失败了")
        final[idx] = float(np.mean(valid))
    return final


# ======================================================================================
# 3) 选块:从「最不重要」里挑,保护首尾 + 可选避免相邻
# ======================================================================================
def protected_layer_set(num_layers: int, protect_front: int = 4, protect_back: int = 1) -> set[int]:
    """保护启发式圈定的「不可删」块号集合:前 protect_front 个 + 后 protect_back 个。"""
    front = set(range(min(protect_front, num_layers)))
    back = set(range(max(0, num_layers - protect_back), num_layers))
    return front | back


def select_layers_to_prune(importance_scores: dict[int, float],
                           num_layers_to_prune: int = 2,
                           heuristic_protection: bool = True,
                           adjacent_protection: bool = True,
                           protect_front: int = 4,
                           protect_back: int = 1) -> list[int]:
    """
    按重要性升序(最不重要在前)挑要删的 block。

    - heuristic_protection:保护首尾块(保护启发式,防灾)。
    - adjacent_protection: 已选块的相邻位不再选(避免删连续段;困惑度指标下分散删更优)。

    返回的块号列表按「被选中的顺序」= 大致按重要性从低到高。
    ⚠️ 排序里对相同分数用 (score, idx) 做二次键,保证结果确定(determinism)。
    """
    total = len(importance_scores)
    protected = protected_layer_set(total, protect_front, protect_back) if heuristic_protection else set()

    # 关键:排序键用 (分数, 块号) —— 分数相同也有稳定顺序,保证「排序确定」。
    sorted_layers = sorted(importance_scores.items(), key=lambda kv: (kv[1], kv[0]))

    selected: list[int] = []
    for layer, _score in sorted_layers:
        if layer in protected:
            continue
        if adjacent_protection and any(abs(layer - s) == 1 for s in selected):
            continue
        selected.append(layer)
        if len(selected) >= num_layers_to_prune:
            break
    return selected


# ======================================================================================
# 4) 真正剪枝:重建一个更短的 layers 列表
# ======================================================================================
def prune_model(model, layer_indices: list[int], inplace: bool = False):
    """
    深度剪枝:从 model.layers 里删掉 layer_indices 指定的 block。

    实现就是书里那句朴素的话 —— 「删 block = 重建一个更短的 ModuleList」。

    - inplace=False(默认):深拷贝一份再删,不动原模型(方便对照实验)。
    - layer_indices:要删的块号列表,允许乱序/重复,内部会去重。
    """
    target = model if inplace else deepcopy(model)
    to_remove = set(layer_indices)

    total = len(target.layers)
    for idx in to_remove:
        if idx < 0 or idx >= total:
            raise IndexError(f"块号 {idx} 越界(合法范围 0..{total - 1})")

    # 保留「不在删除集合里」的 block,按原顺序重建 ModuleList。
    kept = [blk for i, blk in enumerate(target.layers) if i not in to_remove]
    target.layers = nn.ModuleList(kept)
    return target


# ======================================================================================
# 收益量化:参数量 & FLOPs
# ======================================================================================
def count_params(model) -> int:
    """模型总参数量(所有可训练张量的元素个数之和)。"""
    return sum(p.numel() for p in model.parameters())


def count_block_params(block) -> int:
    """单个 block 的参数量。"""
    return sum(p.numel() for p in block.parameters())


def params_breakdown(model) -> dict[str, int]:
    """
    参数量拆解:总量 / 所有 block 合计 / 平均每 block / 非 block 部分(embed+head+norm)。
    用来验证「每删一块,参数按每块均摊比例下降」。
    """
    total = count_params(model)
    block_total = sum(count_block_params(b) for b in model.layers)
    n = len(model.layers)
    return {
        "total": total,
        "blocks_total": block_total,
        "per_block": block_total // n if n else 0,
        "non_block": total - block_total,
        "n_layers": n,
    }


def estimate_flops(model, seq_len: int, batch: int = 1) -> int:
    """
    近似估计单次前向的 FLOPs(只数 nn.Linear 的矩阵乘 —— 占绝对大头)。

    约定:一个把 [.., in] 投影到 [.., out] 的 Linear,对 (batch*seq) 个 token,
    近似 2 * batch * seq * in * out FLOPs(乘加各算一次,即 2 MACs)。

    这样估出来的量级足以体现「删块 => FLOPs 按比例下降」的趋势,
    而不追求和真实 profiler 精确对齐(那需要点每个 elementwise 算子,意义不大)。
    """
    tokens = batch * seq_len
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += 2 * tokens * module.in_features * module.out_features
    return flops
