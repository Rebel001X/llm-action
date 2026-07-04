# -*- coding: utf-8 -*-
"""
test_estimator.py —— 对 estimator.py 每条核心公式做精确/性质验证。

测试分五组：
    1) 显存公式（权重 / KV）—— 与手算精确对齐
    2) Little 定律（并发 = QPS × 延迟）
    3) 吞吐与瓶颈（decode 带宽瓶颈 / prefill 算力瓶颈）
    4) 成本单调性（成本随 QPS 线性、$/1M token 与 QPS 无关）
    5) 边界与健壮性（零、极端、量化、GQA vs MHA）

运行：  python -m pytest -q
"""

import math
import os
import sys

import pytest

# 让测试无论从哪跑都能 import 到上级目录的 estimator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimator import (  # noqa: E402
    DType,
    GPUSpec,
    ModelSpec,
    Workload,
    GPU_CATALOG,
    MODEL_CATALOG,
    BYTES_PER_GB,
    BYTES_PER_TB,
    SECONDS_PER_MONTH,
    weight_memory_gb,
    kv_cache_bytes_per_token,
    little_law_concurrency,
    decode_throughput_per_gpu,
    prefill_throughput_per_gpu,
    estimate,
    estimate_by_name,
    format_report,
)


# ---------------------------------------------------------------------------
# 组 1：显存公式
# ---------------------------------------------------------------------------

def test_weight_memory_fp16_exact():
    """8B 参数 × FP16(2字节) = 16e9 字节 = 14.90 GiB（精确）。"""
    m = MODEL_CATALOG["Llama3-8B"]
    gb = weight_memory_gb(m, DType.FP16)
    expected = 8.0 * 1e9 * 2 / BYTES_PER_GB
    assert gb == pytest.approx(expected, rel=1e-12)
    assert gb == pytest.approx(14.9, abs=0.1)


def test_weight_memory_int8_is_half_of_fp16():
    """INT8(1字节) 权重显存恰为 FP16(2字节) 的一半。量化省显存的第一性原理。"""
    m = MODEL_CATALOG["Llama3-70B"]
    fp16 = weight_memory_gb(m, DType.FP16)
    int8 = weight_memory_gb(m, DType.INT8)
    assert int8 == pytest.approx(fp16 / 2, rel=1e-12)


def test_weight_memory_int4_is_quarter():
    """INT4(0.5字节) 是 FP16 的四分之一。"""
    m = MODEL_CATALOG["Llama3-8B"]
    assert weight_memory_gb(m, DType.INT4) == pytest.approx(
        weight_memory_gb(m, DType.FP16) / 4, rel=1e-12)


def test_kv_per_token_exact_formula():
    """KV/token = 2 × L × kv_heads × head_dim × bytes，逐项对齐手算。

    Llama3-8B: 2 × 32 × 8 × 128 × 2 = 131072 字节 = 128 KB。
    """
    m = MODEL_CATALOG["Llama3-8B"]
    b = kv_cache_bytes_per_token(m, DType.FP16)
    assert b == 2 * 32 * 8 * 128 * 2
    assert b == 131072
    assert b / 1024 == pytest.approx(128.0)


def test_kv_uses_kv_heads_not_q_heads():
    """GQA 的核心：KV 用 num_kv_heads 而非 num_q_heads。

    构造两个除 kv_heads 外完全相同的模型，KV 应严格按 kv_heads 比例缩放。
    """
    mha = ModelSpec("mha", 13, 40, 5120, num_q_heads=40, num_kv_heads=40)
    gqa = ModelSpec("gqa", 13, 40, 5120, num_q_heads=40, num_kv_heads=8)
    kv_mha = kv_cache_bytes_per_token(mha, DType.FP16)
    kv_gqa = kv_cache_bytes_per_token(gqa, DType.FP16)
    # kv_heads 40 → 8，比例应恰为 5 倍
    assert kv_mha / kv_gqa == pytest.approx(40 / 8, rel=1e-12)


def test_kv_int8_half_of_fp16():
    """KV 量化到 INT8，每 token KV 减半。"""
    m = MODEL_CATALOG["Llama3-70B"]
    assert kv_cache_bytes_per_token(m, DType.INT8) == pytest.approx(
        kv_cache_bytes_per_token(m, DType.FP16) / 2, rel=1e-12)


def test_resolved_head_dim_default():
    """未显式给 head_dim 时，用 hidden_dim / num_q_heads 推出。"""
    m = ModelSpec("x", 8, 32, 4096, num_q_heads=32, num_kv_heads=8)
    assert m.resolved_head_dim() == 4096 // 32 == 128


# ---------------------------------------------------------------------------
# 组 2：Little 定律
# ---------------------------------------------------------------------------

def test_little_law_basic():
    """并发 = QPS × 延迟(秒)。100 QPS × 2s = 200。"""
    assert little_law_concurrency(100, 2.0) == pytest.approx(200.0)


def test_little_law_scales_with_both_factors():
    """QPS 翻倍或延迟翻倍，并发都应翻倍（各自线性）。"""
    base = little_law_concurrency(50, 1.5)
    assert little_law_concurrency(100, 1.5) == pytest.approx(2 * base)
    assert little_law_concurrency(50, 3.0) == pytest.approx(2 * base)


def test_concurrency_in_estimate_matches_little_law():
    """estimate() 内部算出的并发数应与 Little 定律一致。"""
    wl = Workload(qps=40, avg_input_tokens=500, avg_output_tokens=200,
                  target_p99_ms=2500)
    est = estimate(wl, MODEL_CATALOG["Llama3-8B"], GPU_CATALOG["A100-80G"])
    assert est.concurrency == pytest.approx(40 * 2.5)


# ---------------------------------------------------------------------------
# 组 3：吞吐与瓶颈
# ---------------------------------------------------------------------------

def test_decode_scales_with_batch():
    """decode 聚合吞吐随 batch 线性增长（连续批处理摊薄带宽墙）。"""
    g, m = GPU_CATALOG["A100-80G"], MODEL_CATALOG["Llama3-8B"]
    t1 = decode_throughput_per_gpu(g, m, DType.FP16, mfu=0.4, batch_size=1)
    t32 = decode_throughput_per_gpu(g, m, DType.FP16, mfu=0.4, batch_size=32)
    assert t32 == pytest.approx(32 * t1, rel=1e-12)


def test_decode_scales_with_bandwidth():
    """decode 是带宽瓶颈：带宽翻倍，吞吐翻倍（其它相同）。"""
    m = MODEL_CATALOG["Llama3-8B"]
    slow = GPUSpec("slow", 300, 80, 1.0, 2.0)
    fast = GPUSpec("fast", 300, 80, 2.0, 2.0)
    ts = decode_throughput_per_gpu(slow, m, DType.FP16)
    tf = decode_throughput_per_gpu(fast, m, DType.FP16)
    assert tf == pytest.approx(2 * ts, rel=1e-12)


def test_prefill_scales_with_flops():
    """prefill 是算力瓶颈：TFLOPS 翻倍，吞吐翻倍。"""
    m = MODEL_CATALOG["Llama3-8B"]
    weak = GPUSpec("weak", 300, 80, 2.0, 2.0)
    strong = GPUSpec("strong", 600, 80, 2.0, 2.0)
    assert prefill_throughput_per_gpu(strong, m) == pytest.approx(
        2 * prefill_throughput_per_gpu(weak, m), rel=1e-12)


def test_smaller_model_has_higher_throughput():
    """同卡下，模型越小，decode/prefill 吞吐越高（读/算的权重更少）。"""
    g = GPU_CATALOG["H100-80G"]
    small, big = MODEL_CATALOG["Llama3-8B"], MODEL_CATALOG["Llama3-70B"]
    assert decode_throughput_per_gpu(g, small, DType.FP16) > \
        decode_throughput_per_gpu(g, big, DType.FP16)
    assert prefill_throughput_per_gpu(g, small) > \
        prefill_throughput_per_gpu(g, big)


def test_memory_bound_when_huge_concurrency():
    """长上下文 + 高并发（大延迟）+ 短输出 → KV 堆积把瓶颈推成 memory。

    构造：小模型（算力便宜）+ 3.2万 input（KV 巨大）+ 32 output（decode 算力小）
        + 极长延迟（并发数飙升，KV 随并发线性膨胀）。
    此时「显存约束需卡数」超过「算力约束需卡数」，瓶颈为 memory。
    """
    wl = Workload(qps=100, avg_input_tokens=32000, avg_output_tokens=32,
                  target_p99_ms=50000)
    est = estimate(wl, MODEL_CATALOG["Llama3-8B"], GPU_CATALOG["A100-40G"])
    assert est.bottleneck == "memory"
    assert est.gpus_for_memory > est.gpus_for_compute


def test_hbm_read_bounded_by_bandwidth():
    """decode 的 HBM 读带宽需求不应超过 GPU 标称带宽（batch 摊薄后 = 带宽×mfu）。"""
    g = GPU_CATALOG["A100-80G"]
    est = estimate_by_name(qps=50, avg_input_tokens=1024, avg_output_tokens=256,
                           gpu_name="A100-80G", mfu=0.4)
    nominal_gbps = g.hbm_bandwidth_tbs * BYTES_PER_TB / BYTES_PER_GB
    assert est.hbm_read_gbps_per_gpu <= nominal_gbps + 1e-6
    # 且应约等于 带宽 × mfu（batch 摊薄后逼近带宽墙）
    assert est.hbm_read_gbps_per_gpu == pytest.approx(nominal_gbps * 0.4, rel=1e-9)


# ---------------------------------------------------------------------------
# 组 4：成本性质
# ---------------------------------------------------------------------------

def test_cost_month_matches_hour():
    """月成本 = 时成本 × (一个月的小时数)。"""
    est = estimate_by_name(qps=30, avg_input_tokens=800, avg_output_tokens=200)
    hours = SECONDS_PER_MONTH / 3600.0
    assert est.cost_per_month_usd == pytest.approx(
        est.cost_per_hour_usd * hours, rel=1e-12)


def test_cost_scales_roughly_linear_with_qps():
    """成本随 QPS 近似线性：QPS×10 → GPU 数与月成本约 ×10（不含 ceil 抖动）。

    用较大 QPS 避免 ceil 取整误差主导；断言比值落在 [8, 12] 的宽松线性带内。
    """
    lo = estimate_by_name(qps=100, avg_input_tokens=1000, avg_output_tokens=300)
    hi = estimate_by_name(qps=1000, avg_input_tokens=1000, avg_output_tokens=300)
    ratio = hi.cost_per_month_usd / lo.cost_per_month_usd
    assert 8.0 <= ratio <= 12.0
    # GPU 数也应大致 ×10
    assert 8.0 <= hi.num_gpus / lo.num_gpus <= 12.0


def test_cost_per_million_tokens_independent_of_qps():
    """$/1M token 是**单位成本**，理论上与 QPS 无关（分子分母同比例放大）。

    在 compute 瓶颈、无 ceil 抖动的区间，两个 QPS 的 $/1M 应几乎相等。
    """
    a = estimate_by_name(qps=200, avg_input_tokens=1000, avg_output_tokens=300)
    b = estimate_by_name(qps=2000, avg_input_tokens=1000, avg_output_tokens=300)
    # 允许 ceil 带来的小幅偏差
    assert a.cost_per_million_tokens_usd == pytest.approx(
        b.cost_per_million_tokens_usd, rel=0.05)


def test_quantization_reduces_cost():
    """把权重从 FP16 量化到 INT8：decode 吞吐翻倍 → 需卡数下降 → 成本下降。"""
    fp16 = estimate_by_name(qps=100, avg_input_tokens=1000, avg_output_tokens=300,
                            model_name="Llama3-70B", gpu_name="H100-80G",
                            weight_dtype=DType.FP16)
    int8 = estimate_by_name(qps=100, avg_input_tokens=1000, avg_output_tokens=300,
                            model_name="Llama3-70B", gpu_name="H100-80G",
                            weight_dtype=DType.INT8)
    assert int8.cost_per_month_usd < fp16.cost_per_month_usd


def test_cheaper_gpu_can_lower_unit_cost():
    """给定同一负载，$/1M token 会随所选 GPU 的性价比变化（这里只验证函数对 GPU 敏感）。"""
    a100 = estimate_by_name(qps=50, avg_input_tokens=1000, avg_output_tokens=300,
                            gpu_name="A100-80G")
    h100 = estimate_by_name(qps=50, avg_input_tokens=1000, avg_output_tokens=300,
                            gpu_name="H100-80G")
    assert a100.cost_per_million_tokens_usd != h100.cost_per_million_tokens_usd


# ---------------------------------------------------------------------------
# 组 5：边界与健壮性
# ---------------------------------------------------------------------------

def test_at_least_one_gpu():
    """即使流量极小，也至少需要 1 张卡（不能是 0）。"""
    est = estimate_by_name(qps=0.001, avg_input_tokens=10, avg_output_tokens=10)
    assert est.num_gpus >= 1


def test_num_gpus_is_ceil_of_max_constraint():
    """实际卡数 = ceil(max(算力约束, 显存约束))。"""
    est = estimate_by_name(qps=50, avg_input_tokens=1024, avg_output_tokens=256)
    raw = max(est.gpus_for_compute, est.gpus_for_memory)
    assert est.num_gpus == max(1, math.ceil(raw))


def test_higher_qps_needs_more_or_equal_gpus():
    """QPS 单调增 → 需卡数单调不减。"""
    prev = 0
    for qps in [10, 50, 100, 500, 1000]:
        est = estimate_by_name(qps=qps, avg_input_tokens=1000, avg_output_tokens=300)
        assert est.num_gpus >= prev
        prev = est.num_gpus


def test_longer_output_needs_more_gpus():
    """输出 token 变长 → decode 工作量增加 → 需卡数不减。"""
    short = estimate_by_name(qps=100, avg_input_tokens=1000, avg_output_tokens=100)
    long = estimate_by_name(qps=100, avg_input_tokens=1000, avg_output_tokens=2000)
    assert long.num_gpus >= short.num_gpus


def test_unknown_model_raises():
    """未知模型名应抛 KeyError，防止静默用错模型。"""
    with pytest.raises(KeyError):
        estimate_by_name(qps=10, avg_input_tokens=100, avg_output_tokens=100,
                         model_name="不存在的模型")


def test_unknown_gpu_raises():
    with pytest.raises(KeyError):
        estimate_by_name(qps=10, avg_input_tokens=100, avg_output_tokens=100,
                         gpu_name="不存在的GPU")


def test_weight_too_big_triggers_tp_note():
    """70B FP32 权重 = 280GB，远超单卡 → 应给出张量并行提示。"""
    est = estimate(Workload(10, 500, 200),
                   MODEL_CATALOG["Llama3-70B"], GPU_CATALOG["A100-40G"],
                   weight_dtype=DType.FP32)
    assert any("张量并行" in n for n in est.notes)
    assert est.num_gpus >= 1


def test_report_renders_without_error():
    """报表渲染不应抛异常，且包含关键字段。"""
    est = estimate_by_name(qps=50, avg_input_tokens=1024, avg_output_tokens=256)
    text = format_report(est)
    assert "GPU" in text and "每月" in text and "百万" in text


def test_all_catalog_models_and_gpus_run():
    """所有目录里的模型 × GPU 组合都能跑通、不崩、卡数为正。"""
    for mname in MODEL_CATALOG:
        for gname in GPU_CATALOG:
            est = estimate_by_name(qps=20, avg_input_tokens=512,
                                   avg_output_tokens=128,
                                   model_name=mname, gpu_name=gname)
            assert est.num_gpus >= 1
            assert est.cost_per_month_usd > 0
            assert est.cost_per_million_tokens_usd > 0


def test_gqa_saves_kv_memory_vs_mha():
    """相同规模下，GQA 模型的并发 KV 显存应显著小于 MHA。"""
    common = dict(qps=100, avg_input_tokens=4000, avg_output_tokens=1000,
                  gpu_name="A100-80G")
    gqa = estimate_by_name(model_name="Llama3-70B", **common)   # 8 KV 头
    mha = estimate_by_name(model_name="MHA-13B", **common)      # 40 KV 头
    # 归一到「每 token KV」比较更公平，直接比 kv_per_token
    assert gqa.kv_per_token_kb < mha.kv_per_token_kb
