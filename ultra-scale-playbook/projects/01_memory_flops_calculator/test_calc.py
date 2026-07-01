"""
test_calc.py —— 用已知模型对拍验证计算器(完全离线,纯 python)。
运行:python -m pytest -q
金标准:参数量对齐公开数字、显存公式自洽、并行/ZeRO/重算按预期缩放。
"""
import math
import pytest
from calc import (
    ModelConfig, param_count, model_state_bytes, activation_bytes,
    training_flops, bill, PRESETS, GB,
)


def test_gpt2_param_count_matches_124m():
    pc = param_count(PRESETS["gpt2-124m"])
    # GPT-2 small 公认约 124M
    assert 120e6 < pc["total"] < 128e6


def test_llama7b_param_count_near_7b():
    pc = param_count(PRESETS["llama-7b"])
    assert 6.0e9 < pc["total"] < 7.0e9


def test_non_embedding_follows_12Lh2():
    cfg = ModelConfig(num_layers=10, hidden=1000, num_heads=10, vocab=1, seq_len=1)
    pc = param_count(cfg)
    # 12 * L * h^2 是主导项;误差来自 4h 低阶项
    approx = 12 * cfg.num_layers * cfg.hidden ** 2
    assert math.isclose(pc["non_embedding"], approx, rel_tol=1e-3)


def test_mixed_adam_is_16_bytes_per_param():
    ms = model_state_bytes(1_000_000, mixed_precision=True, optimizer="adam", zero_stage=0, dp=1)
    assert ms["bytes_per_param"] == 16
    assert ms["total"] == 16 * 1_000_000


def test_zero3_shards_model_state_by_dp():
    base = model_state_bytes(1e9, zero_stage=0, dp=8)["total"]
    z3 = model_state_bytes(1e9, zero_stage=3, dp=8)["total"]
    assert math.isclose(z3, base / 8, rel_tol=1e-9)


def test_zero_stage_monotonic():
    kw = dict(num_params=1e9, dp=4)
    t0 = model_state_bytes(zero_stage=0, **kw)["total"]
    t1 = model_state_bytes(zero_stage=1, **kw)["total"]
    t2 = model_state_bytes(zero_stage=2, **kw)["total"]
    t3 = model_state_bytes(zero_stage=3, **kw)["total"]
    assert t0 > t1 > t2 > t3   # 分片越多,单卡显存越省


def test_activation_recompute_reduces_memory():
    cfg = PRESETS["llama-7b"]
    none = activation_bytes(cfg, batch=1, recompute="none")
    sel = activation_bytes(cfg, batch=1, recompute="selective")
    full = activation_bytes(cfg, batch=1, recompute="full")
    assert none > sel > full


def test_activation_scales_linearly_with_batch():
    cfg = PRESETS["gpt2-124m"]
    a1 = activation_bytes(cfg, batch=1)
    a4 = activation_bytes(cfg, batch=4)
    assert math.isclose(a4, 4 * a1, rel_tol=1e-9)


def test_activation_tp_shards():
    cfg = PRESETS["llama-7b"]
    a1 = activation_bytes(cfg, batch=1, tp=1)
    a8 = activation_bytes(cfg, batch=1, tp=8)
    assert math.isclose(a8, a1 / 8, rel_tol=1e-9)


def test_training_flops_6ND():
    assert training_flops(7e9, 1e12) == 6 * 7e9 * 1e12


def test_bill_tp_reduces_params_per_device():
    cfg = PRESETS["llama-70b"]
    b1 = bill(cfg, batch=1, tp=1, pp=1)
    b8 = bill(cfg, batch=1, tp=8, pp=1)
    assert math.isclose(b8["params_per_device"], b1["params_per_device"] / 8, rel_tol=1e-9)


def test_bill_total_is_positive_and_sums():
    cfg = PRESETS["llama-7b"]
    b = bill(cfg, batch=2, zero_stage=3, dp=8, tp=1, recompute="selective")
    assert b["total_GB"] > 0
    assert math.isclose(b["total_GB"], b["model_state_GB"] + b["activation_GB"], rel_tol=1e-9)


def test_7b_fits_after_zero3_but_not_before():
    # 7B 模型状态 ≈ 6.6e9 * 16 / GB ≈ 98 GB,单张 80GB 卡放不下;ZeRO-3 dp=8 后 ≈ 12 GB 可放下
    cfg = PRESETS["llama-7b"]
    before = bill(cfg, batch=1, zero_stage=0, dp=1, recompute="full")["model_state_GB"]
    after = bill(cfg, batch=1, zero_stage=3, dp=8, recompute="full")["model_state_GB"]
    assert before > 80
    assert after < 20
