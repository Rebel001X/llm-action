# -*- coding: utf-8 -*-
"""
width_pruning.py —— 宽度剪枝实验室的核心库（配套《Rearchitecting LLMs》第 5 章）

本模块把「宽度剪枝（width pruning）」从一堆散代码，收敛成一套干净、可测、纯 CPU 可跑的 API：

    1) 玩具 GLU-MLP（GluMLP）             —— 复刻 Llama-3.2 的 gate/up/down 三层门控结构
    2) 玩具多头注意力（ToyMultiHeadAttention）—— 用来演示「按注意力头剪枝」
    3) 重要性评分（importance scoring）      —— 权重 L2 范数 / peak-to-peak 幅度 / 数据驱动激活范数
    4) 结构化重建（rebuild）               —— topk 选神经元/头，物理上重建更小的权重矩阵
    5) 形状一致性保证                       —— 入口守恒 / 内部同步 k / 出口守恒

设计原则（和 llm-action 中文教程一致）：
    - 每个函数「是什么 / 为什么 / 怎么用 / 代价」都在 docstring 里讲清楚；
    - 纯 torch CPU，不依赖 transformers、不联网、不下模型；
    - 剪枝是「无梯度的结构手术」，全程 torch.no_grad()。

作者视角：AI-Infra 工程师，面向国内面试 & 生产落地。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 一、玩具模型：GLU-MLP 与 多头注意力
# =============================================================================

class GluMLP(nn.Module):
    """玩具版 GLU（Gated Linear Unit，门控线性单元）前馈层。

    结构完全对齐 Llama / Qwen 的 SwiGLU：

        MLP(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    维度约定（记牢，是剪枝的地基）：
        - hidden_size        : 层间对接尺寸（入口=出口），**剪枝时永不改**；
        - intermediate_size  : MLP 内部扩张维度，**这就是我们要剪的「宽度」**。

    PyTorch Linear(in, out) 的 .weight 形状是 (out_features, in_features)：
        - gate_proj / up_proj : Linear(hidden -> inter)，weight = [inter, hidden]，**行 = 神经元**；
        - down_proj           : Linear(inter -> hidden)，weight = [hidden, inter]，**列 = 神经元**。

    ⚠️ 坑：gate 和 up 的第 i 行、down 的第 i 列，共同构成「第 i 个神经元对」，
           必须**一起删或一起留**，否则门控同步（gating synchronization）被打破。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) 充当「带调节的开关」，⊙ 逐元素乘到 up 的「值」上，再降维回 hidden。
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyMultiHeadAttention(nn.Module):
    """玩具多头自注意力（Multi-Head Attention），用于演示「按头剪枝」。

    为聚焦剪枝逻辑，这里做了教学化简化：
        - 单一 batch 内自注意力，无 causal mask（可选加），无 KV cache；
        - q/k/v 各自 Linear(hidden -> num_heads*head_dim)，o_proj 收回 hidden。

    剪一个头 = 同时删掉 q/k/v 对应的 head_dim 行 + o_proj 对应的 head_dim 列。
    删头后 hidden 入口/出口不变，只是 num_heads 变小 —— 与 MLP 剪神经元同构。
    """

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden]
        B, T, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        # 缩放点积注意力
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)      # [B, H, T, T]
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v                                        # [B, H, T, Dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * Dh)  # 拼回 [B, T, H*Dh]
        return self.o_proj(ctx)


# =============================================================================
# 二、重要性评分（importance scoring）
# =============================================================================

def peak_to_peak_importance(gate_weight: torch.Tensor,
                            up_weight: torch.Tensor) -> torch.Tensor:
    """静态 peak-to-peak（峰峰值）重要性评分 —— 书中 Listing 5.0。

    是什么：对每个神经元，取它一行权重的 max(w) + |min(w)|（正向峰 + 负向峰）。
    为什么：范围大 → 能产生更大幅度的变换 → 表达能力强 → 更重要。
    代价：完全 data-free（不看数据），零成本、秒出；但「一刀切」，不知你的任务真正需要谁。

    参数：
        gate_weight, up_weight : [intermediate_size, hidden_size]，行=神经元。
    返回：
        [intermediate_size] 的重要性向量（gate 峰峰 + up 峰峰）。
    """
    gate_range = torch.max(gate_weight, dim=1).values + torch.abs(torch.min(gate_weight, dim=1).values)
    up_range = torch.max(up_weight, dim=1).values + torch.abs(torch.min(up_weight, dim=1).values)
    return gate_range + up_range


def l2_norm_importance(gate_weight: torch.Tensor,
                       up_weight: torch.Tensor) -> torch.Tensor:
    """静态 L2 范数重要性评分（另一条常见静态路线）。

    是什么：每个神经元一行权重的 L2 范数 sqrt(Σ w_i²)，gate 和 up 相加。
    为什么：L2 范数衡量「这个神经元连接权重的总能量」，能量大→影响大→更重要。
             相比 peak-to-peak 更平滑、抗单点极值（一个巨大离群权重不会独占分数）。
    代价：同样 data-free；对「权重大但方向抵消」的情况不如激活敏感。

    返回：[intermediate_size] 的重要性向量。
    """
    gate_l2 = torch.norm(gate_weight, p=2, dim=1)
    up_l2 = torch.norm(up_weight, p=2, dim=1)
    return gate_l2 + up_l2


def hybrid_importance(gate_weight: torch.Tensor,
                      up_weight: torch.Tensor,
                      down_weight: torch.Tensor,
                      act_norm: torch.Tensor) -> torch.Tensor:
    """数据驱动混合评分（结构分 × 激活分）—— 书中 Listing 5.3。

        Importance = StructuralScore  ×  X_d_norm
                     └ 三层归一化范围和   └ 真实激活 L2 范数

    关键点：
        - down_proj 的神经元是「列」，故 dim=0（gate/up 是「行」dim=1）；
        - 三层各自归一化到 [0,1] 再相加，避免某层权重尺度大几个数量级而压过其它两层；
        - 用「乘法」融合：任一项低（比如权重大但激活≈0），最终分就低 → 该删。

    参数：
        gate_weight, up_weight : [inter, hidden]
        down_weight            : [hidden, inter]
        act_norm               : [inter]，来自 hook 累加的激活 L2 范数
    返回：
        [inter] 的混合重要性向量。
    """
    gate_weight = gate_weight.float()
    up_weight = up_weight.float()
    down_weight = down_weight.float()
    act_norm = act_norm.float().to(gate_weight.device)

    gate_score = torch.max(gate_weight, dim=1).values + torch.abs(torch.min(gate_weight, dim=1).values)
    up_score = torch.max(up_weight, dim=1).values + torch.abs(torch.min(up_weight, dim=1).values)
    # down_proj 神经元在「列」上 → dim=0
    down_score = torch.max(down_weight, dim=0).values + torch.abs(torch.min(down_weight, dim=0).values)

    gate_norm = gate_score / (gate_score.max() + 1e-8)
    up_norm = up_score / (up_score.max() + 1e-8)
    down_norm = down_score / (down_score.max() + 1e-8)

    structural = gate_norm + up_norm + down_norm  # [inter]
    return structural * act_norm


def head_importance_by_weight(attn: ToyMultiHeadAttention) -> torch.Tensor:
    """按注意力头计算重要性（静态，基于 q/k/v 权重 L2 范数）。

    每个头占据 q/k/v 权重矩阵里连续的 head_dim 行。把该头所有 q/k/v 权重的
    L2 范数加起来，作为这个头的重要性。范数大 → 该头承载的变换更强 → 更重要。

    返回：[num_heads] 的重要性向量。
    """
    H, Dh = attn.num_heads, attn.head_dim
    # 每个投影 weight = [num_heads*head_dim, hidden]，reshape 成 [H, Dh, hidden] 后按头求范数
    def per_head_norm(w: torch.Tensor) -> torch.Tensor:
        return torch.norm(w.view(H, Dh, -1), p=2, dim=(1, 2))  # [H]

    return (per_head_norm(attn.q_proj.weight.data)
            + per_head_norm(attn.k_proj.weight.data)
            + per_head_norm(attn.v_proj.weight.data))


# =============================================================================
# 三、激活抓取：用 forward hook 累加 down_proj 输入的 L2 范数
# =============================================================================

class ActivationCollector:
    """用 PyTorch forward hook 抓 GluMLP 中 down_proj 的**输入端**激活 L2 范数。

    为什么抓 down_proj 输入端（门控相乘之后、降维之前）？
        - 抓 up_proj 输出（门控之前）：看到的是「潜力」，若被门静音则不是真实贡献；
        - 抓 down_proj 输出（降维之后）：信息已压回 hidden，无法追溯是哪个中间神经元；
        - ✅ 抓 down_proj 输入：门 + SiLU 过滤后，inter 维还在，每个神经元真实贡献一目了然。

    为什么用 L2 范数（而非求和/L1）？
        - 简单求和会正负抵消（[+5,-5,+5,-5]→0，明明在工作却显示不活跃）；
        - L2 = sqrt(Σx²) 平方放大大值，**奖励有尖峰的专才神经元，惩罚低水平背景噪声**。

    ⚠️ 坑：用完必须 remove()，否则 hook 在后续所有 forward 里继续乱触发。
    """

    def __init__(self, mlp: GluMLP):
        self.mlp = mlp
        self.acc = torch.zeros(mlp.intermediate_size, dtype=torch.float32)  # 存 CPU，防 OOM
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _hook(self, module, inp, out):
        # inp 是元组，inp[0] = down_proj 的输入 = [B, T, inter]
        x_d = inp[0].detach()                                   # detach 断开计算图
        act_l2 = torch.norm(x_d.to(torch.float32), p=2, dim=(0, 1))  # 沿 batch&seq 压 → [inter]
        self.acc += act_l2.cpu()                                # 累加到 CPU

    def __enter__(self) -> "ActivationCollector":
        self._handle = self.mlp.down_proj.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()                               # ★ 务必摘 hook
            self._handle = None

    @property
    def norms(self) -> torch.Tensor:
        return self.acc


@torch.no_grad()
def collect_activation_norms(mlp: GluMLP, dataloader) -> torch.Tensor:
    """跑一遍校准数据，用 hook 累加 down_proj 输入的 L2 范数。

    参数：
        mlp        : 要校准的 GluMLP；
        dataloader : 可迭代对象，每个元素是形如 [B, T, hidden] 的输入张量。
    返回：
        [intermediate_size] 的累加激活范数（可直接喂给 hybrid_importance）。
    """
    mlp.eval()
    with ActivationCollector(mlp) as collector:
        for batch in dataloader:
            mlp(batch)
    return collector.norms


# =============================================================================
# 四、结构化重建：物理上做出更小的权重矩阵
# =============================================================================

def _num_to_prune(prune_ratio: float, total: int) -> int:
    """把剪枝比例换算成「删几个」，并封顶保护（至少留 1 个）。

    ⚠️ 坑：prune_ratio=1.0 会把神经元删光 → 模型报废。min(..., total-1) 保证至少留 1。
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio 必须 ∈ [0, 1)，收到 {prune_ratio}")
    return min(int(round(prune_ratio * total)), total - 1)


def select_indices_to_keep(importance: torch.Tensor,
                           prune_ratio: float,
                           divisor: Optional[int] = None) -> torch.Tensor:
    """根据重要性分数，选出要**保留**的神经元/头索引（已按原始位置升序）。

    步骤：
        1. total = 分数长度；算删几个 → 得到保留数 k；
        2. （可选）硬件对齐：把 k 向下取整到 divisor 的倍数（Tensor Core 友好）；
        3. topk 选分数最高的 k 个；
        4. ★ .sort() 把索引按**原始位置**升序重排 —— 保持权重内部结构接近原始，利于后续蒸馏恢复。

    参数：
        importance  : [total] 重要性向量；
        prune_ratio : 剪枝比例 ∈ [0,1)；
        divisor     : 若给定，强制 k 为其倍数（如 8/16/32），演示硬件对齐。
    返回：
        LongTensor，保留的索引（升序），长度 = k。
    """
    total = importance.numel()
    n_prune = _num_to_prune(prune_ratio, total)
    k = total - n_prune
    if divisor is not None and divisor > 0:
        k_aligned = (k // divisor) * divisor
        k = max(k_aligned, divisor)          # 至少留一个 divisor 块
        k = min(k, total)
    _, idx = torch.topk(importance, k, largest=True, sorted=True)
    return idx.sort().values                 # ★ 按原始位置升序


@torch.no_grad()
def rebuild_mlp(mlp: GluMLP, keep_idx: torch.Tensor) -> GluMLP:
    """按保留索引，物理重建一个更小的 GluMLP（形状一致性手术）。

    三条铁律（守住则 Transformer 外部察觉不到内部动过刀）：
        1. 入口守恒：new gate/up 的 in_features 仍 = hidden_size；
        2. 内部同步：gate.out = up.out = down.in = k（三层用同一个 k）；
        3. 出口守恒：new down 的 out_features 仍 = hidden_size。

    权重搬运（★ 行列别搞反）：
        - gate/up  选**行**：weight[keep_idx, :]（神经元是行）；
        - down     选**列**：weight[:, keep_idx]（神经元是列）。

    返回：全新的、更小的 GluMLP（不修改原 mlp）。
    """
    k = keep_idx.numel()
    has_bias = mlp.gate_proj.bias is not None
    new_mlp = GluMLP(mlp.hidden_size, k, bias=has_bias)

    new_mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[keep_idx, :].clone()
    new_mlp.up_proj.weight.data = mlp.up_proj.weight.data[keep_idx, :].clone()
    new_mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, keep_idx].clone()

    if has_bias:
        # gate/up 的 bias 长度 = intermediate（每个神经元一个）→ 按行选；
        # down 的 bias 长度 = hidden（输出维）→ 不变。
        new_mlp.gate_proj.bias.data = mlp.gate_proj.bias.data[keep_idx].clone()
        new_mlp.up_proj.bias.data = mlp.up_proj.bias.data[keep_idx].clone()
        new_mlp.down_proj.bias.data = mlp.down_proj.bias.data.clone()

    new_mlp.intermediate_size = k
    return new_mlp


@torch.no_grad()
def rebuild_attention(attn: ToyMultiHeadAttention,
                      keep_heads: torch.Tensor) -> ToyMultiHeadAttention:
    """按保留的头索引，物理重建一个更少头的注意力（结构化剪头）。

    每个头占 q/k/v 的连续 head_dim 行、o_proj 的连续 head_dim 列。
    我们把「头索引」展开成「行/列索引」，再选出来搬运。

    形状一致性同样成立：hidden 入口/出口不变，只有 num_heads 变小。
    """
    H, Dh = attn.num_heads, attn.head_dim
    new_H = keep_heads.numel()
    # 头索引 → 行/列索引：第 h 个头占 [h*Dh, (h+1)*Dh)
    row_idx = torch.cat([torch.arange(h * Dh, (h + 1) * Dh) for h in keep_heads.tolist()])

    has_bias = attn.q_proj.bias is not None
    # ⚠️ 关键：剪头后 new_H*Dh 可能不整除 hidden（如 hidden=64 保 6 头，6 不整除 64），
    #   所以**不能**走 ToyMultiHeadAttention 的构造函数（那里有整除断言）。
    #   用 __new__ 造一个空壳，再手动把字段填好，强制沿用旧 head_dim，保证与旧头一一对应。
    new_attn = ToyMultiHeadAttention.__new__(ToyMultiHeadAttention)
    nn.Module.__init__(new_attn)
    new_attn.hidden_size = attn.hidden_size
    new_attn.num_heads = new_H
    new_attn.head_dim = Dh
    proj_out = new_H * Dh
    # 重新创建对齐旧 head_dim 的投影层（q/k/v 输出 = new_H*Dh，o_proj 收回 hidden）
    new_attn.q_proj = nn.Linear(attn.hidden_size, proj_out, bias=has_bias)
    new_attn.k_proj = nn.Linear(attn.hidden_size, proj_out, bias=has_bias)
    new_attn.v_proj = nn.Linear(attn.hidden_size, proj_out, bias=has_bias)
    new_attn.o_proj = nn.Linear(proj_out, attn.hidden_size, bias=has_bias)

    new_attn.q_proj.weight.data = attn.q_proj.weight.data[row_idx, :].clone()
    new_attn.k_proj.weight.data = attn.k_proj.weight.data[row_idx, :].clone()
    new_attn.v_proj.weight.data = attn.v_proj.weight.data[row_idx, :].clone()
    new_attn.o_proj.weight.data = attn.o_proj.weight.data[:, row_idx].clone()

    if has_bias:
        new_attn.q_proj.bias.data = attn.q_proj.bias.data[row_idx].clone()
        new_attn.k_proj.bias.data = attn.k_proj.bias.data[row_idx].clone()
        new_attn.v_proj.bias.data = attn.v_proj.bias.data[row_idx].clone()
        new_attn.o_proj.bias.data = attn.o_proj.bias.data.clone()
    return new_attn


# =============================================================================
# 五、一站式 API：给定策略与比例，直接产出剪好的模型 + 统计
# =============================================================================

@dataclass
class PruneStats:
    """剪枝统计结果。"""
    method: str
    prune_ratio: float
    original_inter: int
    pruned_inter: int
    original_params: int
    pruned_params: int

    @property
    def param_reduction(self) -> int:
        return self.original_params - self.pruned_params

    @property
    def param_reduction_pct(self) -> float:
        return 100.0 * self.param_reduction / self.original_params


def count_params(module: nn.Module) -> int:
    """数一个 module 的参数总量。"""
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def prune_mlp(mlp: GluMLP,
              prune_ratio: float,
              method: str = "peak_to_peak",
              dataloader=None,
              divisor: Optional[int] = None) -> tuple[GluMLP, PruneStats]:
    """一站式 MLP 宽度剪枝入口。

    参数：
        mlp         : 原始 GluMLP（不会被修改）；
        prune_ratio : 剪枝比例 ∈ [0,1)；
        method      : "peak_to_peak" | "l2" | "hybrid"（hybrid 需要 dataloader 做校准）；
        dataloader  : method="hybrid" 时必填，校准数据（可迭代 [B,T,hidden] 张量）；
        divisor     : 可选硬件对齐（k 取该值倍数）。
    返回：
        (剪后的新 GluMLP, PruneStats)。
    """
    gate_w = mlp.gate_proj.weight.data
    up_w = mlp.up_proj.weight.data
    down_w = mlp.down_proj.weight.data

    if method == "peak_to_peak":
        importance = peak_to_peak_importance(gate_w, up_w)
    elif method == "l2":
        importance = l2_norm_importance(gate_w, up_w)
    elif method == "hybrid":
        if dataloader is None:
            raise ValueError("method='hybrid' 需要提供 dataloader 做激活校准")
        act_norm = collect_activation_norms(mlp, dataloader)
        importance = hybrid_importance(gate_w, up_w, down_w, act_norm)
    else:
        raise ValueError(f"未知 method: {method}")

    keep_idx = select_indices_to_keep(importance, prune_ratio, divisor=divisor)
    new_mlp = rebuild_mlp(mlp, keep_idx)

    stats = PruneStats(
        method=method,
        prune_ratio=prune_ratio,
        original_inter=mlp.intermediate_size,
        pruned_inter=new_mlp.intermediate_size,
        original_params=count_params(mlp),
        pruned_params=count_params(new_mlp),
    )
    return new_mlp, stats


@torch.no_grad()
def prune_attention(attn: ToyMultiHeadAttention,
                    keep_ratio_or_num,
                    by: str = "ratio") -> tuple[ToyMultiHeadAttention, torch.Tensor]:
    """一站式注意力剪头入口（静态，按 q/k/v 权重范数选头）。

    参数：
        attn             : 原始注意力（不被修改）；
        keep_ratio_or_num: by="ratio" 时是剪枝比例(0~1)；by="num" 时是保留头数；
        by               : "ratio" | "num"。
    返回：
        (剪后的新注意力, 保留的头索引升序)。
    """
    importance = head_importance_by_weight(attn)
    H = attn.num_heads
    if by == "ratio":
        n_prune = min(int(round(keep_ratio_or_num * H)), H - 1)
        k = H - n_prune
    elif by == "num":
        k = int(keep_ratio_or_num)
        k = max(1, min(k, H))
    else:
        raise ValueError(f"未知 by: {by}")
    _, idx = torch.topk(importance, k, largest=True, sorted=True)
    keep_heads = idx.sort().values
    new_attn = rebuild_attention(attn, keep_heads)
    return new_attn, keep_heads


# =============================================================================
# 六、误差度量工具
# =============================================================================

@torch.no_grad()
def relative_output_error(model_a: nn.Module,
                          model_b: nn.Module,
                          x: torch.Tensor) -> float:
    """两个模型在同一输入下的**相对输出误差**（L2 相对范数）。

        err = ||f_a(x) - f_b(x)||_2 / (||f_a(x)||_2 + eps)

    用来验证「剪枝后前向近似原输出」：小剪枝比 → 小误差。
    """
    ya = model_a(x)
    yb = model_b(x)
    num = torch.norm((ya - yb).float(), p=2)
    den = torch.norm(ya.float(), p=2) + 1e-8
    return (num / den).item()
