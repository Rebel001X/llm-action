# -*- coding: utf-8 -*-
"""
test_calculator.py
==================
优化技术收益计算器的 pytest 测试。

三条主线（对应题目要求）：
  1) 量化降显存**严格按位宽比例**（FP16→INT8 减半，→INT4 砍到 1/4）；
  2) 连续批处理提吞吐但**有上限**（受显存 max_batch 与带宽 roofline 约束）；
  3) 公式**自洽**（KV 公式、单调性、量化激活提 FLOPS、瀑布叠加守恒 等）。
"""

import math
import os
import sys

import pytest

# 让测试能 import 上一级目录的核心模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_calculator import (  # noqa: E402
    BITS,
    BYTES_PER_PARAM,
    HardwareConfig,
    ModelConfig,
    Metrics,
    OptimizationCalculator,
    OptimizationConfig,
    WorkloadConfig,
    continuous_batching,
    default_setup,
    kv_quantize,
    quantize,
)


# =============================================================================
# 夹具
# =============================================================================
@pytest.fixture
def calc():
    c, _, _, _ = default_setup()
    return c


@pytest.fixture
def model():
    return ModelConfig()


# =============================================================================
# 一、量化降显存严格按位宽比例
# =============================================================================
def test_weight_memory_fp16_matches_book(calc):
    """7B × 2B(FP16) 应约等于书里的 ~13–14 GB（用 GiB 换算约 13.04）。"""
    w = calc.weight_memory_gb("fp16")
    assert 12.5 < w < 14.0


def test_int8_halves_weight_memory(calc):
    """FP16(16bit) → INT8(8bit)：位宽减半 → 显存严格减半。"""
    w16 = calc.weight_memory_gb("fp16")
    w8 = calc.weight_memory_gb("int8")
    assert math.isclose(w8, w16 / 2, rel_tol=1e-9)


def test_int4_quarters_weight_memory(calc):
    """FP16 → INT4(4bit)：位宽 1/4 → 显存 1/4。"""
    w16 = calc.weight_memory_gb("fp16")
    w4 = calc.weight_memory_gb("int4")
    assert math.isclose(w4, w16 / 4, rel_tol=1e-9)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16", "fp8", "int8", "int4", "fp4"])
def test_weight_memory_scales_with_bit_width(calc, dtype):
    """通用断言：显存 / 参考显存 == 位宽 / 参考位宽（严格按位宽比例）。"""
    ref = "fp16"
    ratio_mem = calc.weight_memory_gb(dtype) / calc.weight_memory_gb(ref)
    ratio_bits = BITS[dtype] / BITS[ref]
    assert math.isclose(ratio_mem, ratio_bits, rel_tol=1e-9)


def test_kv_int8_halves_kv_memory(calc):
    """KV 量化 FP16→INT8：KV 显存严格减半（位宽比例同样成立）。"""
    kv16 = calc.kv_memory_gb(batch=8, kv_dtype="fp16")
    kv8 = calc.kv_memory_gb(batch=8, kv_dtype="int8")
    assert math.isclose(kv8, kv16 / 2, rel_tol=1e-9)


def test_kv_per_token_matches_book_formula(model):
    """
    书 p.412：Llama-7B FP16 每 token KV = 2×32×32×128×2 = 524,288 B = 0.5 MB。
    """
    per_tok = model.kv_bytes_per_token("fp16")
    assert per_tok == 2 * 32 * 32 * 128 * 2
    assert math.isclose(per_tok, 0.5 * 1024 * 1024)


# =============================================================================
# 二、连续批处理提吞吐，但有上限
# =============================================================================
def test_batching_increases_throughput(calc):
    """batch 1→8：吞吐必须显著提升（权重搬运成本被摊薄）。"""
    t1, _ = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=1))
    t8, _ = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=8))
    assert t8 > t1 * 1.5  # 至少 1.5× 以上，确实在提吞吐


def test_throughput_is_monotonic_nondecreasing_in_batch(calc):
    """吞吐关于 batch 单调不减（在显存允许范围内），符合 roofline 直觉。"""
    prev = -1.0
    for b in [1, 2, 4, 8, 16, 32, 64]:
        t, eff = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=b))
        if eff == 0:
            break
        assert t >= prev - 1e-6
        prev = t


def test_throughput_has_upper_bound_saturates(calc):
    """
    吞吐有上限：batch 极大时，KV 搬运项主导，吞吐趋于饱和。
    验证「翻倍 batch 时吞吐增益递减」——大 batch 的边际收益 < 小 batch。
    用一张大显存卡避免过早被显存夹断，纯看带宽 roofline 的饱和。
    """
    big_hw = HardwareConfig(vram_gb=100000.0, mem_bandwidth_tb_s=2.0, fp16_tflops=312.0)
    c = OptimizationCalculator(ModelConfig(), big_hw, WorkloadConfig())
    t_small_a, _ = c.decode_throughput_tok_s(OptimizationConfig(batch_size=2))
    t_small_b, _ = c.decode_throughput_tok_s(OptimizationConfig(batch_size=4))
    t_big_a, _ = c.decode_throughput_tok_s(OptimizationConfig(batch_size=256))
    t_big_b, _ = c.decode_throughput_tok_s(OptimizationConfig(batch_size=512))
    gain_small = t_small_b / t_small_a   # batch 2→4 的吞吐倍数
    gain_big = t_big_b / t_big_a         # batch 256→512 的吞吐倍数
    # 小 batch 翻倍时吞吐接近翻倍；大 batch 翻倍时增益已明显衰减 → 有上限
    assert gain_small > gain_big
    assert gain_big < 1.3  # 大 batch 段几乎饱和


def test_batch_clamped_by_memory(calc):
    """
    显存夹紧：请求 batch=100000 但显存装不下 → effective_batch 被夹到 max_batch。
    这是吞吐上限的另一半（显存墙）。
    """
    max_b = calc.max_batch_by_memory("fp16", "fp16")
    _, eff = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=100000))
    assert eff == max_b
    assert eff < 100000


def test_kv_quant_raises_memory_batch_ceiling(calc):
    """KV 量化让每 token KV 更小 → 显存能装下更多 batch → 吞吐天花板抬高。"""
    max_fp16 = calc.max_batch_by_memory("fp16", "fp16")
    max_int8 = calc.max_batch_by_memory("fp16", "int8")
    assert max_int8 > max_fp16
    # KV 减半，理论上 KV-batch 上限约翻倍
    assert math.isclose(max_int8 / max_fp16, 2.0, rel_tol=0.15)


# =============================================================================
# 三、公式自洽性
# =============================================================================
def test_total_memory_is_sum_of_parts(calc):
    """总显存 == 权重 + KV + 激活余量（无遗漏、无重复计）。"""
    opt = OptimizationConfig(weight_dtype="fp16", batch_size=4, kv_dtype="fp16")
    m = calc.compute(opt)
    act = m.weight_mem_gb * 0.10
    assert math.isclose(m.total_mem_gb, m.weight_mem_gb + m.kv_mem_gb + act, rel_tol=1e-6)


def test_weight_and_activation_quant_speeds_prefill(calc):
    """W8A8/FP8（量化激活）提 FLOPS → TTFT 应比 FP16 更低；W4A16 则不变。"""
    ttft_fp16 = calc.ttft_ms(OptimizationConfig())
    ttft_w8a8 = calc.ttft_ms(OptimizationConfig(weight_dtype="fp8", activation_dtype="fp8"))
    ttft_w4a16 = calc.ttft_ms(OptimizationConfig(weight_dtype="int4", activation_dtype="fp16"))
    assert ttft_w8a8 < ttft_fp16                 # 量化激活 → prefill 提速
    assert math.isclose(ttft_w4a16, ttft_fp16, rel_tol=1e-9)  # weight-only 不提算力


def test_ttft_scales_with_prompt_length():
    """TTFT ∝ prompt_len（prefill 计算量随输入线性增长）—— 公式自洽。"""
    model, hw = ModelConfig(), HardwareConfig()
    c1 = OptimizationCalculator(model, hw, WorkloadConfig(prompt_len=1000, gen_len=100))
    c2 = OptimizationCalculator(model, hw, WorkloadConfig(prompt_len=2000, gen_len=100))
    t1 = c1.ttft_ms(OptimizationConfig())
    t2 = c2.ttft_ms(OptimizationConfig())
    assert math.isclose(t2, t1 * 2, rel_tol=1e-9)


def test_latency_rises_slightly_with_batch(calc):
    """
    吞吐 vs 延迟权衡（书 p.44）：batch 变大，单请求 decode 延迟不降反略升
    （单步要搬更多 KV）。验证方向正确。
    """
    _, eff1 = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=1))
    _, eff32 = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=32))
    lat1 = calc.decode_latency_ms_per_token(OptimizationConfig(batch_size=1), eff1)
    lat32 = calc.decode_latency_ms_per_token(OptimizationConfig(batch_size=32), eff32)
    assert lat32 >= lat1


def test_weight_quant_lowers_decode_latency(calc):
    """权重量化让每步搬运的权重更少 → decode 单 token 延迟下降（带宽受限）。"""
    _, eff = calc.decode_throughput_tok_s(OptimizationConfig(batch_size=1))
    lat_fp16 = calc.decode_latency_ms_per_token(OptimizationConfig(weight_dtype="fp16"), eff)
    lat_int4 = calc.decode_latency_ms_per_token(OptimizationConfig(weight_dtype="int4"), eff)
    assert lat_int4 < lat_fp16


def test_oom_flag_when_model_too_big():
    """把 405B 模型塞进 24G 卡 → OOM=True，effective_batch=0。"""
    big_model = ModelConfig(name="Llama-405B", num_params_b=405, num_layers=126,
                            num_heads=128, head_dim=128, hidden_size=16384)
    small_hw = HardwareConfig(name="RTX4090-24G", vram_gb=24.0)
    c = OptimizationCalculator(big_model, small_hw, WorkloadConfig())
    m = c.compute(OptimizationConfig())
    assert m.oom is True
    assert m.effective_batch == 0
    assert m.throughput_tok_s == 0.0


def test_convenience_transforms_compose(calc):
    """
    便捷变换 quantize/continuous_batching/kv_quantize 可链式叠加，
    且结果等价于直接构造同样的 OptimizationConfig（自洽）。
    """
    opt = OptimizationConfig()
    opt = quantize(opt, "int4")
    opt = continuous_batching(opt, 16)
    opt = kv_quantize(opt, "int8")
    direct = OptimizationConfig(weight_dtype="int4", batch_size=16, kv_dtype="int8")
    assert opt == direct


# =============================================================================
# 四、瀑布叠加（waterfall）
# =============================================================================
def test_waterfall_stages_have_deltas(calc):
    """瀑布每个阶段（除第一个）都带相对上一阶段的增量字段。"""
    stages = [
        ("baseline", OptimizationConfig()),
        ("+W4A16", OptimizationConfig(weight_dtype="int4")),
        ("+batch32", OptimizationConfig(weight_dtype="int4", batch_size=32)),
        ("+KV-int8", OptimizationConfig(weight_dtype="int4", batch_size=32, kv_dtype="int8")),
    ]
    rows = calc.waterfall(stages)
    assert len(rows) == 4
    assert rows[0]["speedup_vs_prev"] == 1.0
    for r in rows[1:]:
        assert "d_mem_gb" in r and "d_throughput" in r and "speedup_vs_prev" in r


def test_waterfall_quant_reduces_weight_memory(calc):
    """瀑布中「+量化」这一步，权重显存必须下降（delta 为负方向合理）。"""
    stages = [
        ("baseline", OptimizationConfig()),
        ("+W4A16", OptimizationConfig(weight_dtype="int4")),
    ]
    rows = calc.waterfall(stages)
    assert rows[1]["metrics"].weight_mem_gb < rows[0]["metrics"].weight_mem_gb


def test_full_stack_beats_baseline_throughput(calc):
    """全优化叠加后的吞吐应远高于 baseline（叠加收益为正）。"""
    base = calc.compute(OptimizationConfig())
    full = calc.compute(OptimizationConfig(weight_dtype="int4", batch_size=32, kv_dtype="int8"))
    assert full.throughput_tok_s > base.throughput_tok_s * 5


def test_as_row_keys_stable(calc):
    """Metrics.as_row 的字段稳定，供表格/绘图消费。"""
    m = calc.compute(OptimizationConfig())
    row = m.as_row()
    expected = {"weight_mem_gb", "kv_mem_gb", "total_mem_gb", "throughput_tok_s",
                "latency_ms_per_token", "ttft_ms", "effective_batch", "oom"}
    assert set(row.keys()) == expected
