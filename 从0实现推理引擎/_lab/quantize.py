"""quantize.py —— 量化：为什么它在推理里是**访存优化**，不是算力优化。

先把最容易被说反的一条摆出来：

    **权重量化省的是「读权重的字节数」，不是「乘法的次数」。**
    weight-only 量化下，权重按 int8/int4 存，**算的时候还是要反量化回浮点**，
    FLOPs 一点没少，甚至还多了反量化那几下。
    它之所以有用，是因为 decode 本来就是访存受限（见 `minigpt.py` 的 `analytic_cost`）——
    **分母小了，分子没动，于是快了。**

推论有两条，都很硬：
  * **batch 很大、已经算力受限的时候，weight-only 量化的收益会塌**，
    因为那时瓶颈已经不在读权重上了。
  * **长上下文 + 大 batch 的时候，主项变成 KV**，该量化的是 KV 缓存而不是权重。
    ⚠️ 这条我第一版写窄了，写成"上下文一长 KV 就占主导"，被自检打脸：
    ctx=16384 但 batch=1 时，KV 只占访存的 39.5%，权重仍是主项 ——
    **单条请求的 KV 再长也只有一份，而权重是固定的一大坨。**
    KV 反超需要**两个条件同时成立**。
  `intensity_with_quant()` 把这两条算给你看。

本文件实现三档量化并逐一量误差：
  * per-tensor（整个矩阵一个 scale）—— 最省事，**一个离群值就毁掉它**
  * per-channel（每个输出通道一个 scale）—— 工程上的默认选择
  * group-wise int4（每 G 个输入维一个 scale）—— 更省字节，误差更大

跑法：
    python quantize.py --selftest    # 13 项断言
    python quantize.py               # 误差表 + 访存账 + 端到端 top-1 一致率

!!! 本文件不产任何硬件性能数字。误差是真算的，字节数是真数的，
    **"快多少"一律只给解析比值，并标明它在什么条件下不成立。**
"""
from __future__ import annotations

import numpy as np

import minigpt as M


# ═════════════════════════════════════════════════ 一、三档量化

def _qdq(w: np.ndarray, amax: np.ndarray, bits: int) -> np.ndarray:
    """对称量化再反量化。`amax` 会广播回 `w` 的形状。"""
    qmax = 2 ** (bits - 1) - 1
    scale = np.where(amax == 0, 1.0, amax / qmax)
    q = np.clip(np.rint(w / scale), -qmax - 1, qmax)
    return (q * scale).astype(np.float32)


def qdq_per_tensor(w: np.ndarray, bits: int = 8) -> np.ndarray:
    """整个矩阵共用一个 scale。

    **它的致命弱点是离群值**：scale 由全局最大绝对值定，
    只要有一个异常大的权重，其余所有权重的有效位数就被它吃掉了。
    `selftest()` 里注入一个离群值，误差会跳一个数量级。
    """
    return _qdq(w, np.abs(w).max(), bits)


def qdq_per_channel(w: np.ndarray, bits: int = 8, axis: int = 0) -> np.ndarray:
    """每个输出通道一个 scale（`w` 形状 (in, out)，沿 `axis=0` 取最大）。

    **这是工程上的默认选择**：额外开销只有每通道一个 fp16 的 scale
    （相对于成千上万个权重可以忽略），却把离群值的影响关进了它自己那一列。
    """
    return _qdq(w, np.abs(w).max(axis=axis, keepdims=True), bits)


def qdq_group(w: np.ndarray, bits: int = 4, group: int = 64) -> np.ndarray:
    """沿输入维每 `group` 个元素一个 scale。int4 常用这一档。

    **粒度换字节**：group 越小误差越小，但 scale 越多、有效位宽越接近 int8。
    `bytes_per_weight()` 把这笔账算清楚 —— int4/group=32 的实际位宽不是 4，是 4.5。
    """
    n_in, n_out = w.shape
    pad = (-n_in) % group
    if pad:
        w = np.concatenate([w, np.zeros((pad, n_out), w.dtype)], 0)
    g = w.reshape(-1, group, n_out)
    out = _qdq(g, np.abs(g).max(axis=1, keepdims=True), bits)
    return out.reshape(-1, n_out)[:n_in]


def bytes_per_weight(bits: int, group: int | None, scale_bytes: int = 2) -> float:
    """**实际**每权重字节数：位宽本身 + 摊到每个权重头上的 scale。

    这是宣传口径和真实口径的差别所在 ——
    "int4" 在 group=32 时实际是 `4/8 + 2/32 = 0.5625` 字节/权重，即 **4.5 bit**。
    group=128 时是 4.125 bit。**group 越小，"int4"越名不副实。**
    """
    base = bits / 8.0
    return base if group is None else base + scale_bytes / group


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """相对 Frobenius 误差。"""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


# ═════════════════════════════════════════════════ 二、访存账

def intensity_with_quant(d_model: int, n_layer: int, n_head: int, d_head: int,
                         vocab: int, ctx: int, batch: int = 1,
                         w_bytes: float = 4.0, kv_bytes: float = 4.0) -> dict:
    """一步 decode 的解析成本，**权重与 KV 的每元素字节数可以分别设**。

    口径与 `minigpt.analytic_cost` 一致（多请求批处理：每条请求各有一份 KV），
    只是把两处 `dtype_bytes` 拆成了两个独立旋钮，好回答两个问题：

      * 量化权重在什么时候有用？—— 权重那一项在访存里占比大的时候（短上下文、小 batch）。
      * 量化 KV 在什么时候有用？—— KV 那一项占比大的时候（长上下文、大 batch）。

    **注意 FLOPs 一项完全不随位宽变**：weight-only 量化不改变乘加次数。
    """
    mm = n_layer * (4 * d_model ** 2 + 8 * d_model ** 2) + d_model * vocab
    flops = 2 * batch * mm + batch * n_layer * n_head * 4 * ctx * d_head
    b_w = w_bytes * (mm + 2 * batch * d_model)
    b_kv = kv_bytes * batch * ctx * 2 * n_layer * d_model
    total = b_w + b_kv
    return {"flops": flops, "bytes": total,
            "arithmetic_intensity": round(flops / total, 3),
            "weight_share": round(b_w / total, 4),
            "kv_share": round(b_kv / total, 4)}


# ═════════════════════════════════════════════════ 三、端到端：量化整个玩具模型

MATS = ("wq", "wk", "wv", "wo", "w1", "w2")


def quantize_model(w: dict, fn) -> dict:
    """把每层的 6 个矩阵过一遍 `fn`。**LayerNorm 的 gain/bias 与 embedding 不动** ——

    它们参数量小、对误差敏感，量化它们收益微乎其微而风险不小。
    真实引擎也是这么做的：**量化只覆盖大矩阵。**
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in w.items() if k != "layers"}
    out["layers"] = []
    for lw in w["layers"]:
        nl = dict(lw)
        for k in MATS:
            nl[k] = fn(lw[k])
        out["layers"].append(nl)
    return out


def top1_agreement(cfg, w_ref: dict, w_q: dict, n_prompt: int = 60,
                   seed: int = 0) -> float:
    """量化前后 **top-1 预测一致的比例**。

    这是本文件唯一的"质量"指标，而且必须说清它是什么：
    **它衡量的是"和未量化的自己有多像"，不是"答得对不对"。**
    本库的模型是随机权重，没有"对不对"可言。
    真实评测要跑真实任务，**任何拿这个数去谈精度损失的说法都是越界的**。
    """
    rng = np.random.default_rng(seed)
    same = 0
    for _ in range(n_prompt):
        p = [int(x) for x in rng.integers(0, cfg.vocab, 6)]
        a = int(M.forward_naive(cfg, w_ref, p).argmax())
        b = int(M.forward_naive(cfg, w_q, p).argmax())
        same += (a == b)
    return same / n_prompt


# ═════════════════════════════════════════════════ 四、自检

def selftest() -> int:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.05, (256, 256)).astype(np.float32)

    # ---- 基本性质 ----
    e8t = rel_err(w, qdq_per_tensor(w, 8))
    e8c = rel_err(w, qdq_per_channel(w, 8))
    e4g = rel_err(w, qdq_group(w, 4, 64))
    # 别拍脑袋定阈值 —— 对称量化的误差有解析预测：
    # 量化噪声近似均匀分布，标准差 = scale/sqrt(12)，而 scale = amax/qmax。
    # 相对误差 ≈ (amax/qmax/sqrt(12)) / std(w)。**先算出来再断言，不要猜。**
    pred = (np.abs(w).max() / 127 / np.sqrt(12)) / float(w.std())
    chk(abs(e8t - pred) / pred < 0.15,
        f"int8 per-tensor 相对误差 {e8t:.5f}，解析预测 {pred:.5f}（差 "
        f"{abs(e8t - pred) / pred:.1%}）—— **误差量级是可以先算出来的**")
    chk(e4g > e8c, f"int4/group64 误差 {e4g:.5f} > int8 per-channel {e8c:.5f}"
                   f" —— **位宽是主导项**")

    q1 = qdq_per_channel(w, 8)
    q2 = qdq_per_channel(q1, 8)
    chk(np.allclose(q1, q2, atol=1e-7),
        "量化是幂等的：对已量化的权重再量化一次不变（否则说明 scale 算法有偏）")

    chk(np.allclose(qdq_per_tensor(w, 32), w, rtol=1e-4, atol=1e-6),
        "位宽足够大时量化退化为恒等（数值健全性检查）")

    zero = np.zeros((8, 8), np.float32)
    chk(np.isfinite(qdq_per_tensor(zero, 8)).all(),
        "全零矩阵不产生除零（scale 为 0 时必须兜底）")

    # ---- 离群值：per-tensor 的死穴 ----
    w_out = w.copy()
    w_out[0, 0] = 50.0                      # 注入一个离群值
    ot = rel_err(w_out, qdq_per_tensor(w_out, 8))
    oc = rel_err(w_out, qdq_per_channel(w_out, 8))
    chk(ot > 20 * e8t,
        f"注入 1 个离群值后 per-tensor 误差从 {e8t:.5f} 涨到 {ot:.5f}"
        f"（{ot / e8t:.0f} 倍）—— **一个数毁掉整个矩阵**")
    chk(oc < ot / 5,
        f"per-channel 把损害关在那一列里：误差 {oc:.5f}，仅为 per-tensor 的 {oc / ot:.1%}")

    # ---- 位宽的真实口径 ----
    b4_32 = bytes_per_weight(4, 32)
    b4_128 = bytes_per_weight(4, 128)
    chk(abs(b4_32 - 0.5625) < 1e-9 and abs(b4_128 - 0.515625) < 1e-9,
        f"所谓 int4：group=32 时实际 {b4_32 * 8:.2f} bit、group=128 时 {b4_128 * 8:.3f} bit"
        f" —— **不是 4**")

    # ---- 访存账 ----
    args = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000)
    short = intensity_with_quant(**args, ctx=128, batch=1)
    long_solo = intensity_with_quant(**args, ctx=16384, batch=1)
    long_batch = intensity_with_quant(**args, ctx=16384, batch=64)
    chk(short["weight_share"] > 0.9,
        f"ctx=128、batch=1 时权重占访存 {short['weight_share']:.1%}")
    # **这一条我一开始写错了。** 原以为"上下文一长 KV 就占主导"，
    # 于是断言 ctx=16384 时 kv_share > 0.5 —— 实测只有 39.5%，断言当场挂掉。
    # 真相：**batch=1 时权重几乎永远是主项**，因为权重是固定的一大坨，
    # 而单条请求的 KV 再长也只有一份。KV 要反超必须同时有长上下文**和**大 batch。
    chk(long_solo["kv_share"] < 0.5 < long_batch["kv_share"],
        f"ctx=16384：batch=1 时 KV 只占 {long_solo['kv_share']:.1%}，"
        f"batch=64 时才涨到 {long_batch['kv_share']:.1%}"
        f" —— **KV 占主导要「长上下文 + 大 batch」两个条件同时成立**")

    w8 = intensity_with_quant(**args, ctx=128, batch=1, w_bytes=1.0)
    chk(w8["flops"] == short["flops"] and w8["bytes"] < short["bytes"],
        f"权重量化到 int8：FLOPs **一点没变**，字节数降到 {w8['bytes'] / short['bytes']:.1%}"
        f" —— **它是访存优化，不是算力优化**")

    kv8_long = intensity_with_quant(**args, ctx=16384, batch=64, kv_bytes=1.0)
    kv8_short = intensity_with_quant(**args, ctx=128, batch=1, kv_bytes=1.0)
    gain_long = 1 - kv8_long["bytes"] / long_batch["bytes"]
    gain_short = 1 - kv8_short["bytes"] / short["bytes"]
    chk(gain_long > 10 * gain_short,
        f"KV 量化到 int8：ctx=16384/batch=64 省 {gain_long:.1%} 访存，"
        f"ctx=128/batch=1 只省 {gain_short:.2%} —— **短上下文小 batch 下量化 KV 基本白干**")

    big_batch = intensity_with_quant(**args, ctx=128, batch=256)
    big_batch_q = intensity_with_quant(**args, ctx=128, batch=256, w_bytes=1.0)
    small_gain = 1 - big_batch_q["bytes"] / big_batch["bytes"]
    solo_gain = 1 - w8["bytes"] / short["bytes"]
    chk(small_gain < solo_gain,
        f"batch=1 时权重量化省 {solo_gain:.1%} 访存，batch=256 时只省 {small_gain:.1%}"
        f" —— **批大了权重被摊薄，量化收益跟着塌**")

    # ---- 端到端 ----
    cfg = M.Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=64, seed=1)
    wm = M.init_weights(cfg)
    a8 = top1_agreement(cfg, wm, quantize_model(wm, lambda x: qdq_per_channel(x, 8)))
    a4 = top1_agreement(cfg, wm, quantize_model(wm, lambda x: qdq_group(x, 4, 16)))
    chk(a8 >= a4,
        f"端到端 top-1 一致率：int8/通道 {a8:.0%} >= int4/组16 {a4:.0%}"
        f"（**这是「与自己有多像」，不是精度**）")
    chk(a8 >= 0.9, f"int8 per-channel 下 top-1 一致率 {a8:.0%}，量化没有把模型打散")

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


# ═════════════════════════════════════════════════ 五、演示

def _demo() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.05, (512, 512)).astype(np.float32)
    w_out = w.copy()
    w_out[0, 0] = 50.0

    print("一、量化误差（相对 Frobenius）")
    print(f"  {'方案':<26}{'每权重字节':>12}{'正态权重':>12}{'含1个离群值':>14}")
    rows = [
        ("fp32（不量化）", 4.0, lambda x: x),
        ("int8 per-tensor", bytes_per_weight(8, None), lambda x: qdq_per_tensor(x, 8)),
        ("int8 per-channel", bytes_per_weight(8, None) + 2 / 512,
         lambda x: qdq_per_channel(x, 8)),
        ("int4 group=128", bytes_per_weight(4, 128), lambda x: qdq_group(x, 4, 128)),
        ("int4 group=32", bytes_per_weight(4, 32), lambda x: qdq_group(x, 4, 32)),
    ]
    for name, b, f in rows:
        print(f"  {name:<26}{b:>12.4f}{rel_err(w, f(w)):>12.5f}"
              f"{rel_err(w_out, f(w_out)):>14.5f}")
    print("  最后一列是本表的重点：**per-tensor 被一个离群值打穿，per-channel 没事。**")

    print("\n二、量化谁取决于工作点（解析访存账，非实测）")
    args = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000)
    print(f"  {'ctx':>7}{'batch':>7}{'权重占访存':>12}{'KV占访存':>11}"
          f"{'权重int8省':>12}{'KV int8省':>11}")
    for ctx, bs in [(128, 1), (2048, 1), (16384, 1), (128, 64), (2048, 64),
                    (16384, 64), (2048, 256)]:
        base = intensity_with_quant(**args, ctx=ctx, batch=bs)
        qw = intensity_with_quant(**args, ctx=ctx, batch=bs, w_bytes=1.0)
        qk = intensity_with_quant(**args, ctx=ctx, batch=bs, kv_bytes=1.0)
        print(f"  {ctx:>7}{bs:>7}{base['weight_share']:>12.1%}"
              f"{base['kv_share']:>11.1%}"
              f"{1 - qw['bytes'] / base['bytes']:>12.1%}"
              f"{1 - qk['bytes'] / base['bytes']:>11.1%}")
    print("  两列「省」加起来接近对应的占比 —— **量化只能省它占的那部分，一分不多。**")
    print("  所以：小 batch 短上下文量权重；大 batch 长上下文量 KV。**没有通用答案。**")

    print("\n三、端到端 top-1 一致率（玩具模型，60 个随机 prompt）")
    cfg = M.Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=64, seed=1)
    wm = M.init_weights(cfg)
    for name, f in [("int8 per-tensor", lambda x: qdq_per_tensor(x, 8)),
                    ("int8 per-channel", lambda x: qdq_per_channel(x, 8)),
                    ("int4 group=16", lambda x: qdq_group(x, 4, 16)),
                    ("int4 per-tensor", lambda x: qdq_per_tensor(x, 4))]:
        a = top1_agreement(cfg, wm, quantize_model(wm, f))
        print(f"  {name:<22}{a:>8.0%}")
    print("  [注意] 这是**与未量化的自己有多像**，不是精度。")
    print("         本库的权重是随机初始化的，没有「答得对不对」可言；")
    print("         真实精度损失必须跑真实评测，**拿这张表谈精度就是越界**。")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _demo()
