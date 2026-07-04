"""
calc.py —— Transformer 训练"显存 + 算力"计算器(纯 Python/numpy,完全离线)

复刻《Ultra-Scale Playbook》第 2 章(单卡显存解剖)与第 9 章(把训练步塞进显存)的算账逻辑:
给定模型结构与并行配置,算出

  - 参数量(parameters)
  - 训练时的"模型状态"显存:参数 + 梯度 + Adam 优化器状态(m, v)
  - 激活(activations)显存(用 Megatron 的经典公式)
  - 前向/反向 FLOPs(≈ 6 * N * D)与 MFU
  - ZeRO-1/2/3、张量并行 TP、流水线并行 PP、激活重算 对显存的影响

所有公式都能在 tests/ 里用已知模型(GPT-2 124M、7B)对拍验证。
中英并列:参数 parameters、梯度 gradients、优化器状态 optimizer states、激活 activations。
"""
from __future__ import annotations
from dataclasses import dataclass, field

# 每种 dtype 的字节数(bytes per element)
DTYPE_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


@dataclass
class ModelConfig:
    """一个标准 decoder-only Transformer 的结构超参。"""
    num_layers: int          # 层数 L
    hidden: int              # 隐藏维 h(d_model)
    num_heads: int           # 注意力头数 a
    vocab: int               # 词表大小 V
    seq_len: int             # 序列长度 s
    ffn_mult: int = 4        # FFN 中间维 = ffn_mult * hidden(GPT 系常用 4)
    tie_embeddings: bool = True   # 输入嵌入与输出投影是否共享权重


# ----------------------------------------------------------------------------
# 1) 参数量
# ----------------------------------------------------------------------------
def param_count(cfg: ModelConfig) -> dict:
    """
    返回 {'embedding', 'non_embedding', 'total'}。

    单层参数(忽略 bias 的低阶项):
      - 注意力:Q/K/V 投影 3*h^2 + 输出投影 h^2 = 4*h^2
      - MLP:两个线性层 h->(m*h)->h,共 2*m*h^2
      - 两个 LayerNorm:≈ 4*h(低阶项)
    ffn_mult=4 时单层 ≈ 4h^2 + 8h^2 = 12 h^2 —— 这就是"12 L h^2"经验公式的来历。
    """
    h = cfg.hidden
    per_layer = 4 * h * h + 2 * cfg.ffn_mult * h * h + 4 * h
    non_embed = cfg.num_layers * per_layer
    # token 嵌入 V*h + 可学习位置嵌入 s*h;输出头与输入嵌入共享则不额外计
    embed = cfg.vocab * h + cfg.seq_len * h
    if not cfg.tie_embeddings:
        embed += cfg.vocab * h
    return {"embedding": embed, "non_embedding": non_embed, "total": embed + non_embed}


# ----------------------------------------------------------------------------
# 2) 模型状态显存(参数 + 梯度 + 优化器)
# ----------------------------------------------------------------------------
def model_state_bytes(
    num_params: int,
    *,
    mixed_precision: bool = True,
    optimizer: str = "adam",
    zero_stage: int = 0,
    dp: int = 1,
) -> dict:
    """
    混合精度 + Adam 的经典账(每参数字节数):
      fp16/bf16 参数 2 + fp16/bf16 梯度 2 + fp32 主参数 4 + Adam m 4 + Adam v 4 = 16 B/param

    ZeRO 沿数据并行维度 dp 分片:
      stage1 分片优化器状态;stage2 再分片梯度;stage3 再分片参数。
    返回各部分与合计的字节数(bytes)。
    """
    if mixed_precision:
        p_bytes, g_bytes = 2, 2                 # 计算用低精度参数/梯度
        master = 4                              # fp32 主参数
    else:
        p_bytes, g_bytes = 4, 4
        master = 0                              # 纯 fp32 时参数本身即 fp32,无额外主副本
    if optimizer == "adam":
        opt = master + 4 + 4                    # 主参数 + m + v
    elif optimizer == "sgd":
        opt = master                            # 无一阶/二阶动量(简化)
    else:
        raise ValueError(optimizer)

    dp = max(1, dp)
    shard = lambda x, on: (x / dp if (on and zero_stage >= 1) else x)
    params = num_params * p_bytes
    grads = num_params * g_bytes
    opt_states = num_params * opt

    # 分片开关:stage1 分优化器,stage2 分梯度,stage3 分参数
    params_e = params / dp if zero_stage >= 3 else params
    grads_e = grads / dp if zero_stage >= 2 else grads
    opt_e = opt_states / dp if zero_stage >= 1 else opt_states
    total = params_e + grads_e + opt_e
    return {
        "params": params_e, "grads": grads_e, "optimizer": opt_e,
        "total": total, "bytes_per_param": (params + grads + opt_states) / num_params,
    }


# ----------------------------------------------------------------------------
# 3) 激活显存(Megatron / Korthikanti 等公式)
# ----------------------------------------------------------------------------
def activation_bytes(
    cfg: ModelConfig,
    batch: int,
    *,
    recompute: str = "none",   # none | selective | full
    tp: int = 1,
    act_bytes: int = 2,        # 激活多为 16-bit
) -> float:
    """
    单层激活(约,单位:元素数)≈ s*b*h*(34 + 5*a*s/h)  —— Megatron 论文公式。
    * 34*s*b*h:各线性层/LN/dropout 的中间激活
    * 5*a*s^2*b:注意力分数矩阵相关(随 s^2 增长,长序列杀手)
    张量并行 tp 把大部分激活按 tp 切分;激活重算大幅降低存储:
    * selective:只重算注意力那部分(去掉 5as/h 项)
    * full:每层只存输入 ≈ 2*s*b*h,反向时整层重算
    """
    s, b, h, a, L = cfg.seq_len, batch, cfg.hidden, cfg.num_heads, cfg.num_layers
    if recompute == "full":
        per_layer_elems = 2 * s * b * h            # 只存每层输入
    elif recompute == "selective":
        per_layer_elems = 34 * s * b * h           # 去掉注意力 s^2 项
    else:
        per_layer_elems = s * b * h * (34 + 5 * a * s / h)
    # TP 切分(近似:大部分激活可切)
    per_layer_elems = per_layer_elems / max(1, tp)
    return per_layer_elems * L * act_bytes


# ----------------------------------------------------------------------------
# 4) FLOPs 与 MFU
# ----------------------------------------------------------------------------
def training_flops(num_params: int, num_tokens: int) -> float:
    """一次训练(前向+反向)总 FLOPs ≈ 6 * N * D(N=参数量,D=token 数)。
    前向 2ND,反向 4ND(反向约为前向 2 倍)。"""
    return 6.0 * num_params * num_tokens


def mfu(achieved_flops_per_s: float, peak_flops_per_s: float) -> float:
    """Model FLOPs Utilization = 实测吞吐算力 / 峰值算力。"""
    return achieved_flops_per_s / peak_flops_per_s


# ----------------------------------------------------------------------------
# 5) 汇总一张"账单"
# ----------------------------------------------------------------------------
GB = 1024 ** 3


def bill(
    cfg: ModelConfig, batch: int, *,
    mixed_precision=True, optimizer="adam",
    zero_stage=0, dp=1, tp=1, pp=1, recompute="none",
) -> dict:
    """综合一台卡上的显存账单(GB)。pp 把层平摊到 pp 个 stage(每 stage L/pp 层)。"""
    pc = param_count(cfg)
    N = pc["total"]
    # 张量并行 tp、流水线并行 pp 都减少单卡承载的参数
    params_per_device = N / max(1, tp) / max(1, pp)
    ms = model_state_bytes(params_per_device, mixed_precision=mixed_precision,
                           optimizer=optimizer, zero_stage=zero_stage, dp=dp)
    # 单卡只算它承载的层数(pp 切层)
    cfg_stage = ModelConfig(**{**cfg.__dict__, "num_layers": max(1, cfg.num_layers // max(1, pp))})
    act = activation_bytes(cfg_stage, batch, recompute=recompute, tp=tp)
    total = ms["total"] + act
    return {
        "params_total": N,
        "params_per_device": params_per_device,
        "model_state_GB": ms["total"] / GB,
        "activation_GB": act / GB,
        "total_GB": total / GB,
        "breakdown": {k: v / GB for k, v in ms.items() if k != "bytes_per_param"},
        "bytes_per_param": ms["bytes_per_param"],
    }


# 一些常见模型预设,方便 demo / 测试
PRESETS = {
    "gpt2-124m": ModelConfig(num_layers=12, hidden=768, num_heads=12, vocab=50257, seq_len=1024),
    "gpt2-xl-1.5b": ModelConfig(num_layers=48, hidden=1600, num_heads=25, vocab=50257, seq_len=1024),
    "llama-7b": ModelConfig(num_layers=32, hidden=4096, num_heads=32, vocab=32000, seq_len=4096, ffn_mult=4),
    "llama-70b": ModelConfig(num_layers=80, hidden=8192, num_heads=64, vocab=32000, seq_len=4096, ffn_mult=4),
}
