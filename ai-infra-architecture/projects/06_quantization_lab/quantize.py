"""
quantize.py —— 量化实验室核心模块(对应 ../../08_量化与低精度_....md)。

用纯 numpy 实现"量化-反量化"(fake / simulated quantization),覆盖:
  · 位宽 num_bits:8bit / 4bit / 任意 b
  · 对称 symmetric(zero-point=0)/ 非对称 asymmetric(带 zero-point)
  · 粒度 granularity:per-tensor(全张量 1 个 scale)
                      per-channel(每行/每输出通道 1 个 scale)
                      per-group(每 G 个元素 1 个 scale,GPTQ/AWQ 的 group_size)
并提供误差度量(MSE / 最大绝对误差)与理论界工具。

核心公式(定点量化 / uniform affine quantization):
    q      = clip( round(x / s) + z ,  qmin, qmax )     # 浮点 -> 整数码
    x_hat  = (q - z) * s                                 # 整数码 -> 近似浮点
其中 s 为 scale(步长),z 为 zero-point(零点偏移)。
本模块只做"模拟量化":量化后立刻反量化回 float,便于逐点测误差——这正是
PTQ/QAT 里 fake-quant 节点的做法。真实部署会把 q 以 int8/int4 打包存储。

另附一个 torch 版 fake_quantize_torch,用于和 numpy 版做数值对拍(parity)。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EPS = 1e-12  # 防止全零张量导致 scale=0 除零


# ============================================================
# 0) 量化方案描述
# ============================================================
@dataclass
class QScheme:
    """一套量化配置。granularity ∈ {per_tensor, per_channel, per_group}。
    per_channel 沿 `axis`(默认 0,即每个输出通道/每行一个 scale)。
    per_group 沿 2D 权重的列(输入维)每 group_size 个元素一个 scale。"""
    num_bits: int = 8
    symmetric: bool = True
    granularity: str = "per_tensor"
    group_size: int = 128
    axis: int = 0

    def __str__(self) -> str:
        sym = "sym" if self.symmetric else "asym"
        g = self.granularity.replace("per_", "")
        tag = f"g{self.group_size}" if self.granularity == "per_group" else ""
        return f"INT{self.num_bits}-{sym}-{g}{tag}"


# ============================================================
# 1) 整数码范围:对称 vs 非对称
# ============================================================
def int_range(num_bits: int, symmetric: bool):
    """返回 (qmin, qmax)。
    对称:有符号且去掉一个极端值,保证零点严格为 0,范围 [-(2^{b-1}-1), 2^{b-1}-1]
          INT8 -> [-127, 127];INT4 -> [-7, 7]。
    非对称:无符号全范围 [0, 2^b - 1]
          INT8 -> [0, 255];INT4 -> [0, 15]。"""
    if symmetric:
        qmax = 2 ** (num_bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** num_bits - 1


# ============================================================
# 2) 计算 scale / zero-point(在某个 reduce 轴上取 min/max)
# ============================================================
def _params(x: np.ndarray, reduce_axes, symmetric: bool, qmin: int, qmax: int):
    """在 reduce_axes 上做归约,得到可广播的 scale、zero-point。
    reduce_axes=None 表示对整个张量归约(per-tensor)。"""
    if symmetric:
        # 对称:范围由绝对值最大值决定,零点固定 0
        amax = np.max(np.abs(x), axis=reduce_axes, keepdims=True)
        scale = np.maximum(amax / qmax, EPS)
        zp = np.zeros_like(scale)
    else:
        # 非对称:用真实 [min, max],并强制把浮点 0 纳入范围(便于精确表示 0)
        xmin = np.minimum(np.min(x, axis=reduce_axes, keepdims=True), 0.0)
        xmax = np.maximum(np.max(x, axis=reduce_axes, keepdims=True), 0.0)
        scale = np.maximum((xmax - xmin) / (qmax - qmin), EPS)
        zp = np.round(qmin - xmin / scale)  # 使浮点 xmin 映射到整数 qmin
    return scale, zp


def _view_and_axes(x: np.ndarray, scheme: QScheme):
    """按粒度把 x 变形成"归约视图",并返回 (视图, 归约轴, 原始形状)。
    per_group:2D -> (rows, n_group, group_size),沿最后一维归约。"""
    orig_shape = x.shape
    g = scheme.granularity
    if g == "per_tensor":
        return x, None, orig_shape
    if g == "per_channel":
        reduce_axes = tuple(i for i in range(x.ndim) if i != scheme.axis)
        return x, reduce_axes, orig_shape
    if g == "per_group":
        assert x.ndim == 2, "per_group 实现针对 2D 权重矩阵 [out, in]"
        rows, cols = x.shape
        assert cols % scheme.group_size == 0, (
            f"列数 {cols} 必须能被 group_size {scheme.group_size} 整除")
        xw = x.reshape(rows, cols // scheme.group_size, scheme.group_size)
        return xw, (2,), orig_shape
    raise ValueError(f"未知 granularity: {g}")


def compute_qparams(x, scheme: QScheme):
    """给定张量与方案,返回 (scale, zp, qmin, qmax)。scale/zp 为按粒度归约后的数组。"""
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = int_range(scheme.num_bits, scheme.symmetric)
    xw, reduce_axes, _ = _view_and_axes(x, scheme)
    scale, zp = _params(xw, reduce_axes, scheme.symmetric, qmin, qmax)
    return scale, zp, qmin, qmax


# ============================================================
# 3) 量化 / 反量化(fake-quant:量化后立刻反量化回 float)
# ============================================================
def fake_quantize(x, scheme: QScheme):
    """核心接口。返回 (x_hat, q, scale, zp):
        x_hat  反量化后的近似浮点(与 x 同形状)
        q      整数码(与 x 同形状,int64;取值在 [qmin, qmax])
        scale  按粒度的 scale 数组
        zp     按粒度的 zero-point 数组"""
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = int_range(scheme.num_bits, scheme.symmetric)
    xw, reduce_axes, orig_shape = _view_and_axes(x, scheme)
    scale, zp = _params(xw, reduce_axes, scheme.symmetric, qmin, qmax)

    q = np.clip(np.round(xw / scale) + zp, qmin, qmax)  # 量化
    x_hat = (q - zp) * scale                             # 反量化
    return (x_hat.reshape(orig_shape),
            q.reshape(orig_shape).astype(np.int64),
            scale, zp)


def quantize_dequantize(x, num_bits=8, symmetric=True,
                        granularity="per_tensor", group_size=128, axis=0):
    """便捷封装:只要 x_hat。等价于用 QScheme 调 fake_quantize。"""
    scheme = QScheme(num_bits, symmetric, granularity, group_size, axis)
    x_hat, _, _, _ = fake_quantize(x, scheme)
    return x_hat


# ============================================================
# 4) 误差度量与理论界
# ============================================================
def mse(x, x_hat) -> float:
    """均方误差 Mean Squared Error。"""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean((x - x_hat) ** 2))


def max_abs_err(x, x_hat) -> float:
    """最大绝对误差 max |x - x_hat|。"""
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x - x_hat)))


def theoretical_max_error(scale) -> float:
    """未截断时,四舍五入误差上界 = scale/2(取所有 scale 的最大值)。"""
    return float(np.max(scale) / 2.0)


def theoretical_mse(scale) -> float:
    """均匀量化误差模型:误差近似 U(-s/2, s/2),方差 = s^2/12。
    per-tensor 时取标量 scale;多 scale 时给出一个代表性上界(用最大 scale)。"""
    return float((np.max(scale) ** 2) / 12.0)


def effective_bits(scheme: QScheme, scale_bits: int = 16) -> float:
    """有效位宽:per-group 每 group_size 个元素要额外存一个 scale(+zp),
    真实占用 = num_bits + scale_bits/group_size(对称;非对称再加零点)。
    per-tensor/per-channel 的元数据摊到每元素后可忽略。"""
    if scheme.granularity == "per_group":
        extra = scale_bits / scheme.group_size
        if not scheme.symmetric:
            extra += scale_bits / scheme.group_size  # 还要存 zero-point
        return scheme.num_bits + extra
    return float(scheme.num_bits)


def quant_report(x, scheme: QScheme) -> dict:
    """一次性给出误差报告:MSE、最大误差、理论界、scale 个数(元数据开销)。"""
    x_hat, q, scale, zp = fake_quantize(x, scheme)
    return {
        "scheme": str(scheme),
        "mse": mse(x, x_hat),
        "max_abs": max_abs_err(x, x_hat),
        "theo_max": theoretical_max_error(scale),
        "theo_mse": theoretical_mse(scale),
        "n_scales": int(scale.size),
        "eff_bits": effective_bits(scheme),
    }


# ============================================================
# 5) outlier 权重构造(演示 per-channel 为何更好)
# ============================================================
def make_outlier_matrix(rows=64, cols=256, n_outlier_rows=2, n_outlier_cols=6,
                        normal_std=0.05, outlier_scale=40.0, seed=0):
    """构造一个"含离群"的权重矩阵,复现真实 LLM 里的 outlier 现象。
    绝大多数元素是小幅正态(std=normal_std);只有极少数位置(位于 n_outlier_rows
    行、且只落在前 n_outlier_cols 列)幅值极大(~outlier_scale × normal_std)。

    为什么这样构造:outlier 只占几个元素,却把 |W| 的最大值抬高几十倍。
      · per-tensor:全局共用一个 scale,被这几个大值撑爆 → 其余正常元素被压到 0,
        大面积掉精度。
      · per-channel:每行独立 scale,只有"含 outlier 的行"被牺牲,其余行不受连累。
      · per-group:每 group 独立 scale,连"含 outlier 的行"里也只牺牲那一个组。
    返回 (W, out_rows)。"""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, normal_std, size=(rows, cols))
    out_rows = rng.choice(rows, size=n_outlier_rows, replace=False)
    big = normal_std * outlier_scale
    # 只在这几行的前 n_outlier_cols 列放大值(集中在少数组内)
    signs = rng.choice([-1.0, 1.0], size=(n_outlier_rows, n_outlier_cols))
    W[np.ix_(out_rows, np.arange(n_outlier_cols))] = signs * big
    return W, out_rows


# ============================================================
# 6) torch 版 fake_quantize(用于和 numpy 版对拍)
# ============================================================
def fake_quantize_torch(x, scheme: QScheme):
    """与 numpy 版逐点等价的 torch 实现(CPU,float64)。返回 x_hat(numpy)。
    证明"量化是纯算术",框架无关;也是把逻辑搬进训练图(QAT)的雏形。"""
    import torch
    t = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
    qmin, qmax = int_range(scheme.num_bits, scheme.symmetric)
    orig_shape = t.shape
    g = scheme.granularity
    if g == "per_tensor":
        t2, dims = t, tuple(range(t.ndim))
    elif g == "per_channel":
        t2 = t
        dims = tuple(i for i in range(t.ndim) if i != scheme.axis)
    elif g == "per_group":
        rows, cols = t.shape
        t2 = t.reshape(rows, cols // scheme.group_size, scheme.group_size)
        dims = (2,)
    else:
        raise ValueError(g)

    if scheme.symmetric:
        amax = t2.abs().amax(dim=dims, keepdim=True)
        scale = torch.clamp(amax / qmax, min=EPS)
        zp = torch.zeros_like(scale)
    else:
        xmin = torch.clamp(t2.amin(dim=dims, keepdim=True), max=0.0)
        xmax = torch.clamp(t2.amax(dim=dims, keepdim=True), min=0.0)
        scale = torch.clamp((xmax - xmin) / (qmax - qmin), min=EPS)
        zp = torch.round(qmin - xmin / scale)

    q = torch.clamp(torch.round(t2 / scale) + zp, qmin, qmax)
    x_hat = (q - zp) * scale
    return x_hat.reshape(orig_shape).numpy()
