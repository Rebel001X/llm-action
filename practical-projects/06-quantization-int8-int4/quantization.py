"""
从零实现权重量化(INT8 / INT4)—— 教学版,CPU 可跑通
===================================================================

这个脚本不依赖任何外部数据集 / 网络 / gym,只用 numpy + torch(CPU),
在一个 toy 规模的小 MLP 上,从最底层把"权重量化"的核心原理走一遍:

    1) absmax 对称量化(per-tensor):整张权重共用一个 scale。
    2) per-channel 对称量化(INT8):每个输出通道(行)各自一个 scale,
       这正是 LLM 权重量化里能显著降低误差的关键工程做法。
    3) 分组对称量化(INT4,group-wise):把每一行再切成若干小组,
       每组一个 scale —— 这是 GPTQ / AWQ 等 4bit 方案的常见布局。

核心概念(量化-反量化 / scale):
    - 量化(quantize):  q = round(clip(w / scale, -Qmax, Qmax))   -> 整数
    - 反量化(dequant): w_hat = q * scale                          -> 近似浮点
    - scale 决定了"浮点范围"如何映射到"有限的整数格点"。对称量化里
      scale = absmax / Qmax,Qmax = 2^(bits-1) - 1(INT8 是 127,INT4 是 7)。
    - 反量化得到的 w_hat 与原始 w 之间的差,就是"量化误差"。位宽越低、
      分组越粗,误差越大;但存储/带宽越省。这就是精度 vs 压缩的权衡。

脚本会打印:不同方案下的 量化误差(MSE / 余弦相似度)、压缩比,
以及在一次 toy 前向中"量化前 vs 量化后"输出的差异,
让你直观看到 INT8 几乎无损、INT4 误差变大但仍可用、per-channel/分组的收益。

运行:
    python quantization.py

只用 ASCII 打印(Windows 控制台 GBK 下打印中文会 UnicodeEncodeError),
中文只出现在注释 / docstring / README 里。
"""

import numpy as np
import torch

# 固定随机种子,保证每次运行结果可复现(教学代码必须可复现)
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===================================================================
# 1. 量化的最小内核:对称量化 / 反量化
# ===================================================================
#
# 对称量化:整数区间关于 0 对称,即 [-Qmax, +Qmax]。
# 给定 bits 位,Qmax = 2^(bits-1) - 1。
#   INT8 -> Qmax = 127     INT4 -> Qmax = 7
# (留出一个码位不用,换取对称、零点为 0,反量化只需一次乘法,硬件友好。)

def qmax_for_bits(bits: int) -> int:
    """返回该位宽下对称量化能表示的最大正整数。"""
    return (1 << (bits - 1)) - 1  # 2^(bits-1) - 1


def symmetric_quantize(weight: torch.Tensor, scale: torch.Tensor, bits: int):
    """
    对称量化:把浮点 weight 按给定 scale 映射成有限位宽的整数。
    q = round(weight / scale),再裁剪到 [-Qmax, Qmax]。
    scale 可以是标量(per-tensor)或可广播的向量(per-channel / group)。
    """
    qmax = qmax_for_bits(bits)
    # 这一步对应"把浮点投影到整数格点":先缩放到整数尺度,四舍五入,再夹紧
    q = torch.round(weight / scale)
    q = torch.clamp(q, min=-qmax, max=qmax)
    return q  # 这里返回的是"整数值",真实推理里会以 int8/int4 存储


def symmetric_dequantize(q: torch.Tensor, scale: torch.Tensor):
    """
    反量化:整数 * scale,得到对原始浮点的近似 w_hat。
    推理时算子真正吃的是这个 w_hat(或等价的整数乘 + scale 缩放)。
    """
    return q * scale


# ===================================================================
# 2. 三种 scale 计算策略
# ===================================================================
#
# scale 怎么选,直接决定误差大小。对称量化里都用 absmax 思路:
# 让权重里绝对值最大的那个数,恰好落在整数边界 Qmax 上。
#   scale = absmax / Qmax
# 区别只在于"absmax 在多大范围内取":整张张量 / 每行 / 每个小组。

def per_tensor_scale(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """per-tensor:整张权重共用一个 absmax,得到一个标量 scale。最省、最糙。"""
    absmax = weight.abs().max()
    scale = absmax / qmax_for_bits(bits)
    return scale  # 标量


def per_channel_scale(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    per-channel(逐输出通道 / 逐行):每行各自一个 absmax 和 scale。
    LLM 权重不同行的数值范围差异很大,逐行 scale 能大幅降低误差。
    weight 形状 [out, in] -> scale 形状 [out, 1](可广播回 weight)。
    """
    absmax = weight.abs().amax(dim=1, keepdim=True)  # 每行取一个最大绝对值
    scale = absmax / qmax_for_bits(bits)
    return scale  # [out, 1]


def grouped_quantize(weight: torch.Tensor, bits: int, group_size: int):
    """
    分组对称量化(group-wise):把每行切成长度 group_size 的小组,每组一个 scale。
    这是 4bit 量化(GPTQ / AWQ 等)的常见布局:粒度比 per-channel 更细,
    在低位宽下能把误差压下来,代价是要多存一些 scale(int4 时仍很划算)。

    返回:整数 q(形状同 weight)、scale(形状 [out, n_groups])、以及 group_size。
    要求 in 维能被 group_size 整除(教学起见,toy 权重已对齐)。
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    n_groups = in_features // group_size

    # 把每行 reshape 成 [out, n_groups, group_size],在最后一维上算 absmax
    w_g = weight.reshape(out_features, n_groups, group_size)
    absmax = w_g.abs().amax(dim=2, keepdim=True)          # [out, n_groups, 1]
    scale_g = absmax / qmax_for_bits(bits)                # 每组一个 scale

    q_g = torch.round(w_g / scale_g)
    q_g = torch.clamp(q_g, -qmax_for_bits(bits), qmax_for_bits(bits))

    q = q_g.reshape(out_features, in_features)            # 还原成原形状
    scale = scale_g.reshape(out_features, n_groups)       # [out, n_groups]
    return q, scale, group_size


def grouped_dequantize(q: torch.Tensor, scale: torch.Tensor, group_size: int):
    """分组反量化:把整数按所属组的 scale 还原成近似浮点。"""
    out_features, in_features = q.shape
    n_groups = in_features // group_size
    q_g = q.reshape(out_features, n_groups, group_size)
    scale_g = scale.reshape(out_features, n_groups, 1)
    w_hat_g = q_g * scale_g
    return w_hat_g.reshape(out_features, in_features)


# ===================================================================
# 3. 误差 / 压缩比度量
# ===================================================================

def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """均方误差:越小说明反量化后的权重越接近原始权重。"""
    return torch.mean((a - b) ** 2).item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    余弦相似度(把权重拉平成向量):衡量"方向"是否一致。
    越接近 1 越好。即使数值整体缩放,余弦也很稳健,常用来判断量化是否"伤筋动骨"。
    """
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()


def compression_ratio(bits: int, scale_numel: int, weight_numel: int,
                      fp_bits: int = 32) -> float:
    """
    估算压缩比(原始 FP 字节 / 量化后字节)。
    量化后字节 = 权重整数位 + scale 占用(scale 仍按 FP16 存,这里粗略按 16 位算)。
    这是教学级估算,真实框架还有 zero-point / pack 等细节。
    """
    original_bits = weight_numel * fp_bits
    quantized_bits = weight_numel * bits + scale_numel * 16  # scale 按 FP16 估
    return original_bits / quantized_bits


# ===================================================================
# 4. toy MLP:用一个真实的小线性层来观察"量化对前向输出的影响"
# ===================================================================

class ToyMLP(torch.nn.Module):
    """一个极小的两层 MLP,纯粹用来做量化前后前向对比(不训练)。"""

    def __init__(self, in_dim=64, hidden=128, out_dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden, bias=False)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def clone_with_weights(model: ToyMLP, w1: torch.Tensor, w2: torch.Tensor) -> ToyMLP:
    """复制一个结构相同的 MLP,但把两层权重换成给定(可能是量化后的)权重。"""
    new = ToyMLP(model.fc1.in_features, model.fc1.out_features, model.fc2.out_features)
    with torch.no_grad():
        new.fc1.weight.copy_(w1)
        new.fc2.weight.copy_(w2)
    return new


# ===================================================================
# 5. 把一种"量化方案"封装成统一接口,便于横向对比
# ===================================================================

def quantize_weight(weight: torch.Tensor, method: str, bits: int,
                    group_size: int = 32):
    """
    给定权重和方案名,返回反量化后的权重 w_hat,以及 scale 的元素个数(算压缩比用)。
    method:
      'per_tensor'  -> per-tensor absmax 对称量化
      'per_channel' -> per-channel(逐行)对称量化
      'group'       -> 分组对称量化(常用于 INT4)
    """
    if method == "per_tensor":
        scale = per_tensor_scale(weight, bits)
        q = symmetric_quantize(weight, scale, bits)
        w_hat = symmetric_dequantize(q, scale)
        scale_numel = 1
    elif method == "per_channel":
        scale = per_channel_scale(weight, bits)            # [out, 1]
        q = symmetric_quantize(weight, scale, bits)
        w_hat = symmetric_dequantize(q, scale)
        scale_numel = scale.numel()
    elif method == "group":
        q, scale, gs = grouped_quantize(weight, bits, group_size)
        w_hat = grouped_dequantize(q, scale, gs)
        scale_numel = scale.numel()
    else:
        raise ValueError(f"unknown method: {method}")
    return w_hat, scale_numel


# ===================================================================
# 6. demo
# ===================================================================

def run_demo():
    print("=" * 70)
    print(" Weight Quantization from Scratch : INT8 / INT4 (CPU, toy)")
    print("=" * 70)

    # ---- 构造 toy 模型,取出第一层权重作为主要量化对象 -------------
    model = ToyMLP(in_dim=64, hidden=128, out_dim=32)
    W1 = model.fc1.weight.detach().clone()  # 形状 [128, 64]
    W2 = model.fc2.weight.detach().clone()  # 形状 [32, 128]
    print(f"\n[model] toy MLP  fc1.weight={tuple(W1.shape)}  "
          f"fc2.weight={tuple(W2.shape)}")
    print(f"[weight] fc1: min={W1.min():.4f} max={W1.max():.4f} "
          f"absmax={W1.abs().max():.4f}")

    # ---- A. 在 fc1 权重上对比各种方案的"误差 + 压缩比" --------------
    # 表头:方案 / 位宽 / MSE(越小越好) / 余弦(越接近1越好) / 压缩比(越大越省)
    print("\n[A] quantization error & compression on fc1.weight "
          "(FP32 baseline)")
    header = f"{'method':<14}{'bits':>5}{'MSE':>14}{'cosine':>12}{'compress':>11}"
    print(header)
    print("-" * len(header))

    configs = [
        ("per_tensor",  8,  None),   # INT8 最朴素:整张一个 scale
        ("per_channel", 8,  None),   # INT8 逐行:LLM 常用,误差小很多
        ("per_tensor",  4,  None),   # INT4 朴素:误差明显变大
        ("per_channel", 4,  None),   # INT4 逐行
        ("group",       4,  32),     # INT4 分组(group=32):4bit 的实用做法
        ("group",       4,  16),     # INT4 更细分组:误差更低、scale 略多
    ]

    rows = []
    for method, bits, gs in configs:
        kwargs = {"group_size": gs} if gs else {}
        w_hat, scale_numel = quantize_weight(W1, method, bits, **(kwargs or {}))
        err = mse(W1, w_hat)
        cos = cosine_sim(W1, w_hat)
        cr = compression_ratio(bits, scale_numel, W1.numel())
        tag = method + (f"/g{gs}" if gs else "")
        rows.append((tag, bits, err, cos, cr))
        print(f"{tag:<14}{bits:>5}{err:>14.3e}{cos:>12.6f}{cr:>10.2f}x")

    # ---- B. 量化整张网络(fc1+fc2),看 toy 前向输出差异 -------------
    # 这一步把"权重误差"传导到"模型输出",更贴近真实关心的指标。
    print("\n[B] end-to-end forward: FP32 output vs quantized-weight output")
    x = torch.randn(8, 64)                      # 一个 toy batch
    with torch.no_grad():
        y_fp = model(x)                         # FP32 参考输出

    fwd_header = (f"{'scheme':<22}{'out-MSE':>14}{'out-cosine':>14}"
                  f"{'max|dy|':>12}")
    print(fwd_header)
    print("-" * len(fwd_header))

    fwd_configs = [
        ("INT8 per_tensor",   "per_tensor",  8, None),
        ("INT8 per_channel",  "per_channel", 8, None),
        ("INT4 per_channel",  "per_channel", 4, None),
        ("INT4 group=32",     "group",       4, 32),
    ]
    for name, method, bits, gs in fwd_configs:
        kwargs = {"group_size": gs} if gs else {}
        w1_hat, _ = quantize_weight(W1, method, bits, **(kwargs or {}))
        w2_hat, _ = quantize_weight(W2, method, bits, **(kwargs or {}))
        qmodel = clone_with_weights(model, w1_hat, w2_hat)
        with torch.no_grad():
            y_q = qmodel(x)
        out_mse = mse(y_fp, y_q)
        out_cos = cosine_sim(y_fp, y_q)
        max_dy = (y_fp - y_q).abs().max().item()
        print(f"{name:<22}{out_mse:>14.3e}{out_cos:>14.6f}{max_dy:>12.4f}")

    # ---- C. 直观展示一次"量化-反量化"在几个权重值上的轨迹 ----------
    # 让读者亲眼看到:浮点 -> 整数 -> 近似浮点,以及 scale 的作用。
    print("\n[C] quantize-dequantize trace on first 6 weights of fc1 row 0")
    row0 = W1[0]
    scale8 = per_channel_scale(W1, 8)[0]        # 该行的 INT8 scale(标量)
    q8 = symmetric_quantize(row0, scale8, 8)
    w8 = symmetric_dequantize(q8, scale8)
    scale4 = per_channel_scale(W1, 4)[0]
    q4 = symmetric_quantize(row0, scale4, 4)
    w4 = symmetric_dequantize(q4, scale4)
    print(f"  INT8 scale={scale8.item():.5f}   INT4 scale={scale4.item():.5f}")
    print(f"  {'fp32':>10}{'int8_q':>9}{'int8_hat':>11}"
          f"{'int4_q':>9}{'int4_hat':>11}")
    for i in range(6):
        print(f"  {row0[i].item():>10.4f}{int(q8[i].item()):>9d}"
              f"{w8[i].item():>11.4f}{int(q4[i].item()):>9d}"
              f"{w4[i].item():>11.4f}")

    # ---- 可选:画一张误差 vs 方案的柱状图(失败则纯文本,不崩) ------
    try:
        import matplotlib
        matplotlib.use("Agg")                   # 无界面后端,直接存 png
        import matplotlib.pyplot as plt
        labels = [r[0] + f"\n{r[1]}b" for r in rows]
        mses = [r[2] for r in rows]
        plt.figure(figsize=(9, 4))
        plt.bar(range(len(rows)), mses, color="#4C72B0")
        plt.yscale("log")
        plt.xticks(range(len(rows)), labels, fontsize=8)
        plt.ylabel("MSE (log scale)")
        plt.title("Quantization error by scheme (fc1.weight)")
        plt.tight_layout()
        out_png = "quant_error.png"
        plt.savefig(out_png, dpi=110)
        plt.close()
        print(f"\n[plot] saved bar chart -> {out_png}")
    except Exception as e:  # noqa: BLE001 (教学代码:画图失败不影响主流程)
        print(f"\n[plot] skipped (matplotlib unavailable): {e}")

    # ---- 结论性信号:让人一眼看出"成功" -----------------------------
    print("\n[summary]")
    print("  - INT8 per_channel: near-lossless (cosine ~1.0, tiny MSE).")
    print("  - INT4 error is larger but group-wise quant brings it back down.")
    print("  - finer groups -> lower error, slightly more scale storage.")
    print("  - compression ratio grows as bits drop (FP32 -> INT8 ~4x, "
          "INT4 ~8x).")
    print("\nDONE. Quantization demo finished successfully.")


if __name__ == "__main__":
    run_demo()
