"""
run_demo.py —— 量化实验室演示:打印误差报告 + 生成两张配图。
运行:python run_demo.py

产出:
  · error_vs_bitwidth.png   量化误差 MSE vs 位宽(8/6/4/3/2),三种粒度对比
  · granularity_outlier.png outlier 权重下 per-tensor/channel/group 的误差对比 + 权重热力图
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 无显示环境,离线出图
import matplotlib.pyplot as plt

from quantize import (
    QScheme, quantize_dequantize, fake_quantize, mse, max_abs_err,
    make_outlier_matrix, quant_report,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BITS = [8, 6, 4, 3, 2]
GRANS = ["per_tensor", "per_channel", "per_group"]
GLABEL = {"per_tensor": "per-tensor(1 个 scale)",
          "per_channel": "per-channel(每行 1 个)",
          "per_group": "per-group(g=64)"}
COLOR = {"per_tensor": "#C44E52", "per_channel": "#DD8452", "per_group": "#55A868"}


def _err_vs_bits(W):
    """返回 {granularity: {bits: mse}}。"""
    out = {}
    for g in GRANS:
        row = {}
        for b in BITS:
            row[b] = mse(W, quantize_dequantize(
                W, num_bits=b, granularity=g, group_size=64, axis=0))
        out[g] = row
    return out


def fig_error_vs_bitwidth(W):
    data = _err_vs_bits(W)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # 左:干净权重上,误差随位宽下降而指数上升(y 对数轴)
    for g in GRANS:
        ys = [data[g][b] for b in BITS]
        ax1.plot(BITS, ys, "o-", color=COLOR[g], label=GLABEL[g], lw=2, ms=7)
    ax1.set_yscale("log")
    ax1.set_xticks(BITS)
    ax1.invert_xaxis()                       # 从高位宽(左)到低位宽(右),误差递增
    ax1.set_xlabel("位宽 num_bits(越右越激进)")
    ax1.set_ylabel("量化误差 MSE(对数轴)")
    ax1.set_title("误差 vs 位宽:每砍 1 bit,误差约 ×4", fontsize=12, weight="bold")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=9)

    # 右:固定 4bit,三种粒度的 MSE 柱状(干净数据上差别小)
    b = 4
    vals = [data[g][b] for g in GRANS]
    ax2.bar([GLABEL[g] for g in GRANS], vals, color=[COLOR[g] for g in GRANS])
    ax2.set_ylabel(f"INT{b} 量化误差 MSE")
    ax2.set_title("干净权重上:粒度越细,误差越小(差距不大)", fontsize=12, weight="bold")
    ax2.tick_params(axis="x", rotation=12)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig("error_vs_bitwidth.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return data


def fig_granularity_outlier():
    W, out_rows = make_outlier_matrix(rows=48, cols=256, n_outlier_rows=2,
                                      outlier_scale=40.0, seed=7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # 左:权重热力图,几行离群(幅值极大)
    im = ax1.imshow(np.abs(W), aspect="auto", cmap="magma")
    ax1.set_title(f"含离群的权重 |W|(第 {sorted(int(r) for r in out_rows)} 行有大值)",
                  fontsize=11, weight="bold")
    ax1.set_xlabel("输入维 in_features"); ax1.set_ylabel("输出通道 out(行)")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    # 右:4bit 下三种粒度的 MSE(log 轴),per-tensor 被 outlier 拖爆
    res = {}
    for g in GRANS:
        res[g] = mse(W, quantize_dequantize(
            W, num_bits=4, granularity=g, group_size=64, axis=0))
    bars = ax2.bar([GLABEL[g] for g in GRANS], [res[g] for g in GRANS],
                   color=[COLOR[g] for g in GRANS])
    ax2.set_yscale("log")
    ax2.set_ylabel("INT4 量化误差 MSE(对数轴)")
    ax2.set_title("有 outlier 时:per-channel/group 完胜 per-tensor",
                  fontsize=12, weight="bold")
    ax2.tick_params(axis="x", rotation=12)
    for rect, g in zip(bars, GRANS):
        ax2.text(rect.get_x() + rect.get_width() / 2, res[g],
                 f"{res[g]:.1e}", ha="center", va="bottom", fontsize=8)
    ratio = res["per_tensor"] / res["per_channel"]
    ax2.text(0.97, 0.90, f"per-tensor ≈ {ratio:.0f}× per-channel",
             transform=ax2.transAxes, ha="right", color="#C44E52",
             fontsize=11, weight="bold")

    fig.tight_layout()
    fig.savefig("granularity_outlier.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return res, ratio


def main():
    print("=" * 66)
    print("量化实验室:INT8/INT4 · 对称/非对称 · per-tensor/channel/group")
    print("=" * 66)

    rng = np.random.default_rng(0)
    W = rng.normal(0.0, 0.1, size=(64, 256))    # 干净权重

    print("\n[1] 干净权重(64x256, N(0,0.1)),per-tensor 误差 vs 位宽:")
    print(f"    {'bits':>5} | {'MSE':>12} | {'max_abs':>10} | 理论max = scale/2")
    for b in BITS:
        xh, _, scale, _ = fake_quantize(W, QScheme(b, True, "per_tensor"))
        print(f"    {b:>5} | {mse(W, xh):>12.3e} | {max_abs_err(W, xh):>10.3e} "
              f"| {float(scale.reshape(-1)[0]) / 2:.3e}")

    print("\n[2] 对称 vs 非对称(对全正激活 |N|+0.5,per-tensor,8bit):")
    a = np.abs(rng.normal(size=(32, 128))) + 0.5
    for sym in (True, False):
        r = quant_report(a, QScheme(8, sym, "per_tensor"))
        tag = "对称  sym " if sym else "非对称 asym"
        print(f"    {tag}: MSE={r['mse']:.3e}  max={r['max_abs']:.3e}")

    print("\n[3] 生成 error_vs_bitwidth.png ...")
    fig_error_vs_bitwidth(W)

    print("[4] outlier 权重(2 行幅值 ×40),各粒度 INT4 误差:")
    res, ratio = fig_granularity_outlier()
    for g in GRANS:
        print(f"    {g:<12}: MSE = {res[g]:.3e}")
    print(f"    -> per-tensor 误差是 per-channel 的 {ratio:.0f} 倍(outlier 撑爆全局 scale)")
    print("    生成 granularity_outlier.png")

    print("\n[5] per-group 的元数据开销(有效位宽):")
    for gs in (128, 64, 32):
        r = quant_report(W, QScheme(4, True, "per_group", group_size=gs))
        print(f"    INT4 group_size={gs:>3}: 有效位宽 ≈ {r['eff_bits']:.3f} bit "
              f"(scales={r['n_scales']})")

    print("\n[OK] demo 结束,已生成 2 张图。")


if __name__ == "__main__":
    main()
