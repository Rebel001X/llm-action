# -*- coding: utf-8 -*-
"""
memory_calculator 的 pytest 测试集
==================================
覆盖:
  1. "16 字节/参数" 定律 (DDP + Adam + fp16)
  2. ZeRO-k 按 DP 度正确分片
  3. FSDP ≈ ZeRO-3(数值完全一致)
  4. 分片单调性(切得越多单卡越省)
  5. can_fit 放得下/放不下 判定
  6. 边界与非法输入
运行:  python -m pytest -q
"""

import math
import os
import sys

import pytest

# 让测试无论从哪个目录运行,都能 import 到上一级的 memory_calculator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_calculator import (  # noqa: E402
    BYTES_PER_DTYPE,
    OPTIMIZER_STATE_BYTES,
    ModelConfig,
    ParallelStrategy,
    can_fit,
    compare_all_strategies,
    estimate_activation_bytes,
    estimate_model_state,
)

GB = 1024 ** 3


# ---------------------------------------------------------------------------
# 1. 核心:16 字节/参数 定律
# ---------------------------------------------------------------------------
def test_ddp_adam_fp16_is_16_bytes_per_param():
    """DDP + Adam + fp16 下,单卡模型状态应恰为 16 B/param。

    2(fp16参数) + 2(fp16梯度) + 12(Adam:主副本4+m4+v4) = 16 B/param
    """
    N = 1_000_000_000  # 1B 参数,取整方便验算
    cfg = ModelConfig(num_params=N, optimizer="adam", param_dtype="fp16", dp_degree=8)
    bd = estimate_model_state(cfg, ParallelStrategy.DDP)
    total_bytes = bd.model_state_bytes
    assert total_bytes == pytest.approx(16 * N), "DDP+Adam+fp16 必须是 16 B/param"
    # 分块也要对得上
    assert bd.param_bytes == pytest.approx(2 * N)
    assert bd.grad_bytes == pytest.approx(2 * N)
    assert bd.optimizer_bytes == pytest.approx(12 * N)


def test_ddp_independent_of_dp_degree():
    """DDP 不分片:改 DP 度不改变单卡显存。"""
    N = 5e8
    bd1 = estimate_model_state(
        ModelConfig(N, "adam", "fp16", dp_degree=2), ParallelStrategy.DDP)
    bd2 = estimate_model_state(
        ModelConfig(N, "adam", "fp16", dp_degree=64), ParallelStrategy.DDP)
    assert bd1.model_state_bytes == pytest.approx(bd2.model_state_bytes)


def test_bf16_same_as_fp16_bytes():
    """bf16 与 fp16 都是 2 字节,模型状态应完全相等。"""
    N = 3e8
    a = estimate_model_state(ModelConfig(N, "adam", "fp16", 8), ParallelStrategy.DDP)
    b = estimate_model_state(ModelConfig(N, "adam", "bf16", 8), ParallelStrategy.DDP)
    assert a.model_state_bytes == pytest.approx(b.model_state_bytes)


# ---------------------------------------------------------------------------
# 2. ZeRO-k 按 DP 度分片
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("D", [1, 2, 4, 8, 16, 64])
def test_zero1_shards_only_optimizer(D):
    """ZeRO-1:只有优化器状态 /D,参数和梯度保持全量。"""
    N = 7e9
    cfg = ModelConfig(N, "adam", "fp16", dp_degree=D)
    ddp = estimate_model_state(cfg, ParallelStrategy.DDP)
    z1 = estimate_model_state(cfg, ParallelStrategy.ZERO1)
    assert z1.param_bytes == pytest.approx(ddp.param_bytes)          # 参数没切
    assert z1.grad_bytes == pytest.approx(ddp.grad_bytes)            # 梯度没切
    assert z1.optimizer_bytes == pytest.approx(ddp.optimizer_bytes / D)  # 优化器切了


@pytest.mark.parametrize("D", [1, 2, 8, 32])
def test_zero2_shards_grad_and_optimizer(D):
    """ZeRO-2:梯度 + 优化器 /D,参数仍全量。"""
    N = 7e9
    cfg = ModelConfig(N, "adam", "fp16", dp_degree=D)
    ddp = estimate_model_state(cfg, ParallelStrategy.DDP)
    z2 = estimate_model_state(cfg, ParallelStrategy.ZERO2)
    assert z2.param_bytes == pytest.approx(ddp.param_bytes)               # 参数没切
    assert z2.grad_bytes == pytest.approx(ddp.grad_bytes / D)             # 梯度切了
    assert z2.optimizer_bytes == pytest.approx(ddp.optimizer_bytes / D)   # 优化器切了


@pytest.mark.parametrize("D", [1, 2, 8, 128])
def test_zero3_shards_everything(D):
    """ZeRO-3:三块全部 /D。"""
    N = 7e9
    cfg = ModelConfig(N, "adam", "fp16", dp_degree=D)
    ddp = estimate_model_state(cfg, ParallelStrategy.DDP)
    z3 = estimate_model_state(cfg, ParallelStrategy.ZERO3)
    assert z3.param_bytes == pytest.approx(ddp.param_bytes / D)
    assert z3.grad_bytes == pytest.approx(ddp.grad_bytes / D)
    assert z3.optimizer_bytes == pytest.approx(ddp.optimizer_bytes / D)
    # 全切后总量应为 DDP 的 1/D
    assert z3.model_state_bytes == pytest.approx(ddp.model_state_bytes / D)


# ---------------------------------------------------------------------------
# 3. FSDP ≈ ZeRO-3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("D", [1, 4, 8, 16, 256])
def test_fsdp_equals_zero3(D):
    """FSDP 全分片,数值应与 ZeRO-3 完全一致。"""
    cfg = ModelConfig(13e9, "adamw", "bf16", dp_degree=D)
    z3 = estimate_model_state(cfg, ParallelStrategy.ZERO3)
    fsdp = estimate_model_state(cfg, ParallelStrategy.FSDP)
    assert fsdp.param_bytes == pytest.approx(z3.param_bytes)
    assert fsdp.grad_bytes == pytest.approx(z3.grad_bytes)
    assert fsdp.optimizer_bytes == pytest.approx(z3.optimizer_bytes)
    assert fsdp.model_state_bytes == pytest.approx(z3.model_state_bytes)


# ---------------------------------------------------------------------------
# 4. 分片单调性:显存 DDP >= ZeRO1 >= ZeRO2 >= ZeRO3
# ---------------------------------------------------------------------------
def test_memory_monotonic_across_strategies():
    """在 D>1 时,越激进的分片单卡显存越小(非严格递减,但不增)。"""
    cfg = ModelConfig(7e9, "adam", "fp16", dp_degree=8)
    r = compare_all_strategies(cfg)
    ddp = r["DDP"].model_state_bytes
    z1 = r["ZeRO-1"].model_state_bytes
    z2 = r["ZeRO-2"].model_state_bytes
    z3 = r["ZeRO-3"].model_state_bytes
    assert ddp >= z1 >= z2 >= z3
    # 且都严格小于 DDP(因为 D=8>1 且各块非零)
    assert z1 < ddp and z2 < z1 and z3 < z2


def test_optimizer_state_bytes_table():
    """校验优化器状态字节表:Adam=12, SGD-m=8, SGD=4。"""
    assert OPTIMIZER_STATE_BYTES["adam"] == 12
    assert OPTIMIZER_STATE_BYTES["sgd_momentum"] == 8
    assert OPTIMIZER_STATE_BYTES["sgd"] == 4


def test_sgd_cheaper_than_adam():
    """同配置下 SGD 的优化器状态显存应小于 Adam。"""
    N = 1e9
    adam = estimate_model_state(
        ModelConfig(N, "adam", "fp16", 8), ParallelStrategy.DDP)
    sgd = estimate_model_state(
        ModelConfig(N, "sgd", "fp16", 8), ParallelStrategy.DDP)
    assert sgd.optimizer_bytes < adam.optimizer_bytes
    # SGD 无动量:仅 fp32 主副本 4 B/param
    assert sgd.optimizer_bytes == pytest.approx(4 * N)


# ---------------------------------------------------------------------------
# 5. can_fit 判定
# ---------------------------------------------------------------------------
def test_can_fit_ddp_7b_oom_on_40g():
    """7B 模型 DDP 需 ~104GB,放不进 40GB 卡。"""
    cfg = ModelConfig(7e9, "adam", "fp16", dp_degree=8)
    ddp = estimate_model_state(cfg, ParallelStrategy.DDP)
    assert can_fit(ddp, gpu_mem_gb=40.0) is False


def test_can_fit_zero3_7b_fits_on_40g():
    """7B 模型 ZeRO-3 (DP=8) 单卡 ~13GB,放得进 40GB 卡。"""
    cfg = ModelConfig(7e9, "adam", "fp16", dp_degree=8)
    z3 = estimate_model_state(cfg, ParallelStrategy.ZERO3)
    assert can_fit(z3, gpu_mem_gb=40.0) is True


def test_can_fit_respects_reserved_and_activation():
    """预留 + 激活值会占用余量,可能把'原本放得下'挤成 OOM。"""
    cfg = ModelConfig(7e9, "adam", "fp16", dp_degree=8)
    z3 = estimate_model_state(cfg, ParallelStrategy.ZERO3)  # ~13GB 模型状态
    # 给 16GB 卡,再加 5GB 激活 + 2GB 预留 → 13+5+2=20 > 16 → 放不下
    assert can_fit(z3, gpu_mem_gb=16.0, activation_gb=5.0, reserved_gb=2.0) is False
    # 换 40GB 卡就放得下
    assert can_fit(z3, gpu_mem_gb=40.0, activation_gb=5.0, reserved_gb=2.0) is True


# ---------------------------------------------------------------------------
# 6. 激活值估算
# ---------------------------------------------------------------------------
def test_activation_scales_linearly_with_batch():
    """激活值应与 batch_size 成正比:batch 翻倍,激活翻倍。"""
    base = estimate_activation_bytes(num_layers=32, hidden_size=4096,
                                     seq_len=2048, batch_size=1)
    dbl = estimate_activation_bytes(num_layers=32, hidden_size=4096,
                                    seq_len=2048, batch_size=2)
    assert dbl == pytest.approx(2 * base)


def test_activation_checkpointing_reduces_memory():
    """开启激活重计算后显存应下降。"""
    full = estimate_activation_bytes(32, 4096, 2048, 4,
                                     use_activation_checkpointing=False)
    ckpt = estimate_activation_bytes(32, 4096, 2048, 4,
                                     use_activation_checkpointing=True)
    assert ckpt < full


# ---------------------------------------------------------------------------
# 7. 边界 & 非法输入
# ---------------------------------------------------------------------------
def test_dp_degree_one_equals_no_sharding():
    """DP=1 时,ZeRO-3 应和 DDP 完全一样(分母为 1,等于没切)。"""
    cfg = ModelConfig(1e9, "adam", "fp16", dp_degree=1)
    ddp = estimate_model_state(cfg, ParallelStrategy.DDP)
    z3 = estimate_model_state(cfg, ParallelStrategy.ZERO3)
    assert ddp.model_state_bytes == pytest.approx(z3.model_state_bytes)


def test_invalid_num_params_raises():
    with pytest.raises(ValueError):
        ModelConfig(num_params=0, optimizer="adam", param_dtype="fp16", dp_degree=8)
    with pytest.raises(ValueError):
        ModelConfig(num_params=-1e9, optimizer="adam", param_dtype="fp16", dp_degree=8)


def test_invalid_dp_degree_raises():
    with pytest.raises(ValueError):
        ModelConfig(num_params=1e9, optimizer="adam", param_dtype="fp16", dp_degree=0)


def test_invalid_optimizer_raises():
    with pytest.raises(ValueError):
        ModelConfig(num_params=1e9, optimizer="lion_unknown", param_dtype="fp16", dp_degree=8)


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        ModelConfig(num_params=1e9, optimizer="adam", param_dtype="fp64", dp_degree=8)


def test_gb_conversion_consistency():
    """model_state_gb 应等于 model_state_bytes / 1024^3。"""
    cfg = ModelConfig(2e9, "adam", "fp16", 8)
    bd = estimate_model_state(cfg, ParallelStrategy.DDP)
    assert bd.model_state_gb == pytest.approx(bd.model_state_bytes / GB)


def test_dtype_table_values():
    """dtype 字节表健全性。"""
    assert BYTES_PER_DTYPE["fp32"] == 4
    assert BYTES_PER_DTYPE["fp16"] == 2
    assert BYTES_PER_DTYPE["bf16"] == 2
    assert BYTES_PER_DTYPE["fp8"] == 1
