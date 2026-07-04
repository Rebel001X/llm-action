"""
test_quantize.py —— 验证量化实现抓住了本质与理论界。
运行:python -m pytest -q

覆盖:
  1) 对称零点=0、非对称边界、整数码落在 [qmin,qmax]、最大值映射到 qmax
  2) 量化往返最大误差 <= scale/2(理论界);MSE ≈ s^2/12(均匀模型)
  3) 位宽越低误差越大(单调)
  4) 有 outlier 时 per-channel / per-group 明显优于 per-tensor
  5) 非对称对"全正激活"优于对称
  6) numpy 与 torch 两套实现数值对拍一致
"""
import numpy as np
import pytest

from quantize import (
    QScheme, int_range, fake_quantize, quantize_dequantize,
    compute_qparams, mse, max_abs_err, theoretical_max_error, theoretical_mse,
    make_outlier_matrix, fake_quantize_torch, quant_report,
)


# ---------------- 基础:范围 / 零点 / 边界 ----------------
def test_int_range_symmetric_vs_asymmetric():
    assert int_range(8, True) == (-127, 127)
    assert int_range(4, True) == (-7, 7)
    assert int_range(8, False) == (0, 255)
    assert int_range(4, False) == (0, 15)


def test_symmetric_zero_point_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 64))
    for gran in ("per_tensor", "per_channel", "per_group"):
        _, _, scale, zp = fake_quantize(
            x, QScheme(8, symmetric=True, granularity=gran, group_size=32))
        assert np.allclose(zp, 0.0)   # 对称量化零点恒为 0


def test_codes_within_range_all_schemes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 128)) * 3.0
    for bits in (8, 4, 2):
        for sym in (True, False):
            for gran in ("per_tensor", "per_channel", "per_group"):
                sc = QScheme(bits, sym, gran, group_size=32)
                _, q, _, _ = fake_quantize(x, sc)
                qmin, qmax = int_range(bits, sym)
                assert q.min() >= qmin and q.max() <= qmax


def test_symmetric_max_maps_to_qmax():
    # 绝对值最大的元素,对称量化后应落在 ±qmax
    x = np.array([[0.1, -0.2, 5.0, -5.0, 0.0]])
    _, q, _, _ = fake_quantize(x, QScheme(8, symmetric=True, granularity="per_tensor"))
    assert q.max() == 127 or q.min() == -127


def test_zero_is_exactly_representable_asymmetric():
    # 非对称:强制把 0 纳入范围,浮点 0 应精确还原为 0
    x = np.linspace(-1.0, 3.0, 50).reshape(5, 10)
    x[0, 0] = 0.0
    x_hat = quantize_dequantize(x, num_bits=8, symmetric=False, granularity="per_tensor")
    assert abs(x_hat[0, 0]) < 1e-9


# ---------------- 理论界:往返误差 <= scale/2,MSE ≈ s^2/12 ----------------
SCHEMES = [
    QScheme(8, True, "per_tensor"),
    QScheme(8, False, "per_tensor"),
    QScheme(4, True, "per_channel"),
    QScheme(4, False, "per_channel"),
    QScheme(4, True, "per_group", group_size=32),
    QScheme(4, False, "per_group", group_size=32),
]


@pytest.mark.parametrize("scheme", SCHEMES, ids=[str(s) for s in SCHEMES])
def test_roundtrip_within_theoretical_max_bound(scheme):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 128)) * 2.0
    x_hat, _, scale, _ = fake_quantize(x, scheme)
    # 数据范围完全落在量化区间内(无截断)-> 每点误差 <= 对应 scale/2
    bound = theoretical_max_error(scale)
    assert max_abs_err(x, x_hat) <= bound * (1 + 1e-6)


def test_mse_matches_uniform_model_for_fine_grid():
    # 8bit、细网格下,量化误差近似 U(-s/2,s/2),MSE ≈ s^2/12
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 256))
    sc = QScheme(8, True, "per_tensor")
    x_hat, _, scale, _ = fake_quantize(x, sc)
    observed = mse(x, x_hat)
    model = theoretical_mse(scale)
    ratio = observed / model
    assert 0.2 < ratio < 3.0   # 量级吻合(gaussian 非严格均匀,留足余量)


# ---------------- 位宽越低误差越大 ----------------
def test_lower_bits_larger_error():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(64, 128))
    errs = {b: mse(x, quantize_dequantize(x, num_bits=b, granularity="per_tensor"))
            for b in (8, 6, 4, 3, 2)}
    assert errs[2] > errs[3] > errs[4] > errs[6] > errs[8]


def test_max_error_halves_when_adding_a_bit():
    # 每多 1 bit,量化格子翻倍,scale(≈最大误差)约减半
    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 128)) * 1.5
    _, _, s8, _ = fake_quantize(x, QScheme(8, True, "per_tensor"))
    _, _, s7, _ = fake_quantize(x, QScheme(7, True, "per_tensor"))
    ratio = theoretical_max_error(s7) / theoretical_max_error(s8)
    assert 1.7 < ratio < 2.3   # (2^7-1)/(2^6-1)=127/63 ≈ 2.02


# ---------------- outlier:per-channel / per-group 优于 per-tensor ----------------
def test_per_channel_beats_per_tensor_with_outliers():
    W, _ = make_outlier_matrix(rows=64, cols=256, n_outlier_rows=2,
                               outlier_scale=40.0, seed=7)
    e_pt = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_tensor"))
    e_pc = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_channel", axis=0))
    assert e_pt > e_pc * 5      # 离群通道下 per-tensor 明显更差


def test_per_group_beats_per_tensor_with_outliers():
    W, _ = make_outlier_matrix(rows=64, cols=256, n_outlier_rows=2,
                               outlier_scale=40.0, seed=8)
    e_pt = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_tensor"))
    e_pg = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_group", group_size=64))
    assert e_pt > e_pg * 5


def test_finer_granularity_never_worse_on_clean_data():
    # 无 outlier 的干净数据上,更细粒度也不应更差(通常更好或相当)
    rng = np.random.default_rng(9)
    W = rng.normal(size=(64, 256)) * 0.1
    e_pt = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_tensor"))
    e_pc = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_channel"))
    e_pg = mse(W, quantize_dequantize(W, num_bits=4, granularity="per_group", group_size=64))
    assert e_pc <= e_pt * 1.05
    assert e_pg <= e_pc * 1.05


# ---------------- 非对称 vs 对称:全正激活 ----------------
def test_asymmetric_beats_symmetric_for_nonneg_activation():
    # 模拟 ReLU/GELU 后的激活:恒为正且有偏。对称浪费掉一半负区间。
    rng = np.random.default_rng(10)
    a = np.abs(rng.normal(size=(32, 128))) + 0.5     # 全正
    e_sym = mse(a, quantize_dequantize(a, num_bits=8, symmetric=True, granularity="per_tensor"))
    e_asym = mse(a, quantize_dequantize(a, num_bits=8, symmetric=False, granularity="per_tensor"))
    assert e_asym < e_sym


# ---------------- numpy / torch 对拍 ----------------
@pytest.mark.parametrize("scheme", SCHEMES, ids=[str(s) for s in SCHEMES])
def test_numpy_torch_parity(scheme):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(16, 64)) * 2.0
    x_hat_np, _, _, _ = fake_quantize(x, scheme)
    x_hat_th = fake_quantize_torch(x, scheme)
    assert np.allclose(x_hat_np, x_hat_th, atol=1e-8, rtol=0)


# ---------------- 报告字段完整 ----------------
def test_quant_report_fields_and_scale_count():
    W = np.random.default_rng(12).normal(size=(64, 256))
    r_pt = quant_report(W, QScheme(4, True, "per_tensor"))
    r_pc = quant_report(W, QScheme(4, True, "per_channel"))
    r_pg = quant_report(W, QScheme(4, True, "per_group", group_size=64))
    assert r_pt["n_scales"] == 1
    assert r_pc["n_scales"] == 64             # 每行一个
    assert r_pg["n_scales"] == 64 * (256 // 64)   # 每组一个
    # per-group 有效位宽应高于名义位宽(要额外存 scale)
    assert r_pg["eff_bits"] > 4.0
    assert r_pt["eff_bits"] == 4.0
