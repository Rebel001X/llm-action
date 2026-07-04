# -*- coding: utf-8 -*-
"""
DDP vs FSDP/ZeRO 显存计算器 —— 核心模块 (memory_calculator.py)
================================================================

本模块用「第一性原理」把大模型训练时**单卡显存占用**拆成 4 大块:
    1. 参数 (Parameters, P)
    2. 梯度 (Gradients, G)
    3. 优化器状态 (Optimizer States, O)
    4. 激活值 (Activations, A)  —— 本计算器聚焦前 3 块(模型状态 Model States),
                                    激活值单独用一个粗略估算函数给出。

核心记忆点 💡:混合精度 (mixed precision) 训练下,Adam 优化器对**每个参数**
需要约 **16 字节 (16 Bytes/param)** 的「模型状态」显存:

    ┌─────────────────────────────────────────────────────────────┐
    │  fp16 参数        : 2 B                                        │
    │  fp16 梯度        : 2 B                                        │
    │  fp32 参数副本    : 4 B   ┐                                    │
    │  fp32 动量  m     : 4 B   ├── 优化器状态 (Adam) = 12 B         │
    │  fp32 方差  v     : 4 B   ┘                                    │
    ├─────────────────────────────────────────────────────────────┤
    │  合计            : 16 B/param                                  │
    └─────────────────────────────────────────────────────────────┘

这就是著名的 "16 Bytes per parameter" 定律 (见 ZeRO 论文 Rajbhandari et al. 2020,
《Distributed AI Systems》一书中反复引用)。

ZeRO / FSDP 的思想:把上面这 3 块中的一部分或全部,**沿数据并行 (DP) 维度切片**,
每张卡只存 1/DP,从而线性降低单卡显存:

    ZeRO-1 : 只切「优化器状态 O」                    → 省得最多(O 占大头)
    ZeRO-2 : 切「优化器状态 O + 梯度 G」
    ZeRO-3 : 切「优化器状态 O + 梯度 G + 参数 P」全切 → FSDP 本质等价于 ZeRO-3
    DDP    : 什么都不切,每卡全量 16 B/param         → 最费显存,但通信最简单

本模块只用 Python 标准库 + dataclass,**零第三方依赖、纯离线**,不加载任何真实模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


# ============================================================================
# 1. 基础常量:各 dtype 每个元素占多少字节
# ============================================================================
# 说明:训练时"计算精度"和"存储精度"可以不同。混合精度里,前向/反向用 fp16(或 bf16),
# 但为了数值稳定,优化器内部维护一份 fp32 的"主参数副本"(master copy)。
BYTES_PER_DTYPE: Dict[str, int] = {
    "fp32": 4,   # 单精度
    "fp16": 2,   # 半精度
    "bf16": 2,   # bfloat16,和 fp16 一样 2 字节,但指数位更多、数值范围更大
    "fp8": 1,    # 8-bit 浮点(H100 起支持),前沿训练/推理用
    "int8": 1,   # 8-bit 整型,主要用于推理量化
}


class ParallelStrategy(str, Enum):
    """支持的并行/分片策略。继承 str 方便直接当字符串用、打印友好。"""
    DDP = "DDP"          # 纯数据并行,不分片,每卡全量
    ZERO1 = "ZeRO-1"     # 分片优化器状态
    ZERO2 = "ZeRO-2"     # 分片优化器状态 + 梯度
    ZERO3 = "ZeRO-3"     # 分片优化器状态 + 梯度 + 参数
    FSDP = "FSDP"        # PyTorch FSDP,全分片,本质 ≈ ZeRO-3


# ============================================================================
# 2. 优化器"每参数额外倍数"表
# ============================================================================
# 这里的数字 = 优化器状态(O)相对于"一份 fp32 参数"的字节倍数。
# 以 Adam 为例:它需要 fp32 主副本(4B) + 动量 m(4B) + 方差 v(4B) = 12B,
#   其中 4B 的 fp32 主副本我们单独算,故 Adam 的"纯状态" = m + v = 8B = 2 份 fp32。
# 为避免歧义,下面 OPTIMIZER_STATE_BYTES 直接给"每参数优化器状态总字节数(含 fp32 主副本)"。
#
# 记忆:混合精度下
#   Adam  优化器状态(含 fp32 主副本 + m + v) = 4 + 4 + 4 = 12 B/param
#   SGD-m 优化器状态(含 fp32 主副本 + 动量 momentum) = 4 + 4 = 8 B/param
#   SGD   优化器状态(仅 fp32 主副本,无动量)         = 4 B/param
OPTIMIZER_STATE_BYTES: Dict[str, int] = {
    "adam": 12,     # AdamW / Adam:主副本 4 + m 4 + v 4
    "adamw": 12,    # 同 Adam
    "sgd_momentum": 8,   # SGD + momentum:主副本 4 + 动量 4
    "sgd": 4,       # 纯 SGD:仅 fp32 主副本
    "adafactor": 8, # Adafactor 近似(按行列存储,粗略取 8;实际更省,这里给保守估计)
}


@dataclass
class ModelConfig:
    """一次显存估算所需的全部输入。

    Attributes
    ----------
    num_params : float
        模型参数量(个数,不是字节)。例如 7B 模型传 7e9。
    optimizer : str
        优化器名,取值见 OPTIMIZER_STATE_BYTES 的 key(小写)。
    param_dtype : str
        前向/反向计算时参数与梯度的 dtype,通常 'fp16' 或 'bf16'。
    dp_degree : int
        数据并行度 = 参与切片的卡数(ZeRO/FSDP 沿这个维度分片)。
    """
    num_params: float
    optimizer: str = "adam"
    param_dtype: str = "fp16"
    dp_degree: int = 8

    def __post_init__(self) -> None:
        # —— 入参校验:提前把非法输入拦下来,报清晰错误(面试常问"你怎么做健壮性") ——
        if self.num_params <= 0:
            raise ValueError(f"num_params 必须 > 0,收到 {self.num_params}")
        if self.dp_degree < 1:
            raise ValueError(f"dp_degree 必须 >= 1,收到 {self.dp_degree}")
        self.optimizer = self.optimizer.lower()
        if self.optimizer not in OPTIMIZER_STATE_BYTES:
            raise ValueError(
                f"未知优化器 '{self.optimizer}',支持:{list(OPTIMIZER_STATE_BYTES)}"
            )
        if self.param_dtype not in BYTES_PER_DTYPE:
            raise ValueError(
                f"未知 dtype '{self.param_dtype}',支持:{list(BYTES_PER_DTYPE)}"
            )


@dataclass
class MemoryBreakdown:
    """一次估算的输出:各部分显存(单位:字节 Bytes)+ 便捷的 GB 换算。"""
    strategy: str
    param_bytes: float        # 参数占用(单卡)
    grad_bytes: float         # 梯度占用(单卡)
    optimizer_bytes: float    # 优化器状态占用(单卡)
    dp_degree: int

    @property
    def model_state_bytes(self) -> float:
        """模型状态总和(不含激活值)= P + G + O。"""
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes

    @property
    def model_state_gb(self) -> float:
        """换算成 GB(用 1 GB = 1024^3 字节的二进制口径)。"""
        return self.model_state_bytes / (1024 ** 3)

    def as_gb_dict(self) -> Dict[str, float]:
        """把四项(P/G/O/总)都换成 GB,方便画图/打印。"""
        g = 1024 ** 3
        return {
            "参数 P": self.param_bytes / g,
            "梯度 G": self.grad_bytes / g,
            "优化器 O": self.optimizer_bytes / g,
            "总计": self.model_state_bytes / g,
        }


# ============================================================================
# 3. 核心函数:给定 config + 策略,算单卡模型状态显存
# ============================================================================
def estimate_model_state(cfg: ModelConfig, strategy: ParallelStrategy) -> MemoryBreakdown:
    """计算某并行策略下**单卡**的模型状态显存(P + G + O)。

    数学模型
    --------
    设参数量为 N,数据并行度为 D。定义"全量"(未分片)每卡占用:

        param_full = N * bytes(param_dtype)          # 如 fp16 → 2N 字节
        grad_full  = N * bytes(param_dtype)          # 梯度和参数同 dtype → 2N
        opt_full   = N * OPTIMIZER_STATE_BYTES[opt]  # Adam → 12N

    然后按策略决定"哪几块除以 D":

        ┌────────┬──────────┬──────────┬────────────┐
        │ 策略    │  参数 P   │  梯度 G   │  优化器 O   │
        ├────────┼──────────┼──────────┼────────────┤
        │ DDP    │  全量     │  全量     │   全量      │
        │ ZeRO-1 │  全量     │  全量     │   /D        │
        │ ZeRO-2 │  全量     │  /D       │   /D        │
        │ ZeRO-3 │  /D       │  /D       │   /D        │
        │ FSDP   │  /D       │  /D       │   /D  (≈Z3) │
        └────────┴──────────┴──────────┴────────────┘

    Parameters
    ----------
    cfg : ModelConfig
        模型/优化器/并行配置。
    strategy : ParallelStrategy
        并行策略枚举。

    Returns
    -------
    MemoryBreakdown
        单卡各部分显存明细(字节)。
    """
    N = cfg.num_params
    D = cfg.dp_degree
    b_param = BYTES_PER_DTYPE[cfg.param_dtype]      # 参数/梯度每元素字节
    b_opt = OPTIMIZER_STATE_BYTES[cfg.optimizer]    # 优化器状态每参数字节

    # —— 全量(每卡若不分片)各块字节 ——
    param_full = N * b_param
    grad_full = N * b_param
    opt_full = N * b_opt

    # —— 按策略决定分片因子(1 表示不切,D 表示沿 DP 维切成 1/D) ——
    if strategy == ParallelStrategy.DDP:
        p_shard, g_shard, o_shard = 1, 1, 1
    elif strategy == ParallelStrategy.ZERO1:
        p_shard, g_shard, o_shard = 1, 1, D
    elif strategy == ParallelStrategy.ZERO2:
        p_shard, g_shard, o_shard = 1, D, D
    elif strategy in (ParallelStrategy.ZERO3, ParallelStrategy.FSDP):
        # FSDP 全分片 == ZeRO-3:参数、梯度、优化器状态全部 /D
        p_shard, g_shard, o_shard = D, D, D
    else:  # pragma: no cover —— 枚举已穷尽,这里只是防御
        raise ValueError(f"未知策略 {strategy}")

    return MemoryBreakdown(
        strategy=strategy.value,   # 用 .value 得到 "DDP" 而非 "ParallelStrategy.DDP"
        param_bytes=param_full / p_shard,
        grad_bytes=grad_full / g_shard,
        optimizer_bytes=opt_full / o_shard,
        dp_degree=D,
    )


# ============================================================================
# 4. 激活值(Activations)粗略估算 —— 补充,不属于"模型状态"
# ============================================================================
def estimate_activation_bytes(
    num_layers: int,
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    param_dtype: str = "fp16",
    use_activation_checkpointing: bool = False,
) -> float:
    """极粗略估算 Transformer 激活值显存(单位:字节)。

    经验公式(数量级近似,来自 Megatron-LM / 《Distributed AI Systems》讨论):

        activations ≈ L * B * S * H * bytes * k

    其中 L=层数, B=batch, S=序列长, H=隐藏维, k 为经验常数(这里取 ~34/H 的简化,
    直接用 k≈12 表示每 token 每层的中间张量倍数)。开启激活重计算(checkpointing)
    后,激活显存约降到 1/√L 量级,这里简化为乘 0.3。

    ⚠️ 注意:这是**数量级估算**,不追求精确;精确值依赖具体实现(注意力是否 flash、
    是否存 dropout mask 等)。面试时能说清"激活值和 B*S*H*L 成正比"即可。
    """
    if min(num_layers, hidden_size, seq_len, batch_size) <= 0:
        raise ValueError("num_layers/hidden_size/seq_len/batch_size 必须都 > 0")
    b = BYTES_PER_DTYPE[param_dtype]
    k = 12  # 经验倍数:qkv、attn score、mlp 中间态等
    raw = num_layers * batch_size * seq_len * hidden_size * b * k
    if use_activation_checkpointing:
        raw *= 0.3  # 激活重计算:用算力换显存,近似降到 30%
    return float(raw)


# ============================================================================
# 5. 便捷入口:一次算齐所有策略 + 判断能否放进显存
# ============================================================================
def compare_all_strategies(cfg: ModelConfig) -> Dict[str, MemoryBreakdown]:
    """对同一 config,把 DDP / ZeRO-1/2/3 / FSDP 全算一遍,返回 {策略名: 明细}。"""
    strategies = [
        ParallelStrategy.DDP,
        ParallelStrategy.ZERO1,
        ParallelStrategy.ZERO2,
        ParallelStrategy.ZERO3,
        ParallelStrategy.FSDP,
    ]
    return {s.value: estimate_model_state(cfg, s) for s in strategies}


def can_fit(breakdown: MemoryBreakdown, gpu_mem_gb: float,
            activation_gb: float = 0.0, reserved_gb: float = 2.0) -> bool:
    """判断某策略下单卡显存"能否放进" GPU。

    Parameters
    ----------
    breakdown : MemoryBreakdown
        某策略的模型状态明细。
    gpu_mem_gb : float
        单卡总显存(GB),如 A100-40G 传 40,H100-80G 传 80。
    activation_gb : float
        额外激活值显存(GB),默认 0(只看模型状态)。
    reserved_gb : float
        预留给 CUDA context / 通信 buffer / 碎片的余量,默认 2GB。

    Returns
    -------
    bool
        True 表示放得下(留有余量),False 表示 OOM 风险。
    """
    need = breakdown.model_state_gb + activation_gb + reserved_gb
    return need <= gpu_mem_gb


def summarize(cfg: ModelConfig, gpu_mem_gb: float = 80.0,
              activation_gb: float = 0.0) -> str:
    """生成一段人类可读的对比报告(纯文本表格),给 CLI / demo 打印用。"""
    results = compare_all_strategies(cfg)
    lines = []
    lines.append(f"模型: {cfg.num_params/1e9:.1f}B 参数 | 优化器: {cfg.optimizer} "
                 f"| dtype: {cfg.param_dtype} | DP 度: {cfg.dp_degree} "
                 f"| 单卡显存: {gpu_mem_gb:.0f}GB")
    lines.append("-" * 78)
    lines.append(f"{'策略':<8}{'参数(GB)':>12}{'梯度(GB)':>12}"
                 f"{'优化器(GB)':>14}{'合计(GB)':>12}{'能否放下':>10}")
    lines.append("-" * 78)
    for name, bd in results.items():
        gb = bd.as_gb_dict()
        fit = can_fit(bd, gpu_mem_gb, activation_gb)
        flag = "✅ 放得下" if fit else "❌ OOM"
        lines.append(f"{name:<8}{gb['参数 P']:>12.2f}{gb['梯度 G']:>12.2f}"
                     f"{gb['优化器 O']:>14.2f}{gb['总计']:>12.2f}{flag:>12}")
    lines.append("-" * 78)
    return "\n".join(lines)


# ============================================================================
# 6. 直接运行本文件:打印一个 7B 模型的对比示例
# ============================================================================
if __name__ == "__main__":
    # ⚠️ Windows 控制台默认 GBK 编码,打印 emoji/中文会崩;重设为 UTF-8。
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    demo_cfg = ModelConfig(num_params=7e9, optimizer="adam",
                           param_dtype="fp16", dp_degree=8)
    print(summarize(demo_cfg, gpu_mem_gb=40.0))
