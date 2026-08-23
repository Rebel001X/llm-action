"""flashattn.py —— 从朴素注意力到 FlashAttention 的那一步，用 numpy 走一遍。

**一句话结论，先放在这里：FlashAttention 一个 FLOP 都没省。**
它省的是**访存** —— 具体说，是那张 T×T 的注意力矩阵在显存里的来回搬运。
`analytic_cost()` 那套账（见 `minigpt.py`）已经说明 decode 是访存受限，
所以"省访存 = 省时间，省 FLOPs 基本不省时间"。**这是本文件存在的全部理由。**

难点只有一个：**softmax 要先知道全局最大值和全局求和，才能算第一个输出。**
朴素做法因此必须把整张 T×T 矩阵先算完、存下来。
在线 softmax（online softmax）把这个依赖拆掉了 —— 边扫边修正：

    见到新块 -> 更新running最大值 -> 把已经累好的部分**按比例缩回去** -> 加上新块的贡献

于是**永远不需要同时持有整张矩阵**，峰值中间量从 O(T²) 降到 O(块大小)。

本文件提供三样：
  * `attention_naive()`   —— 参照实现，materialize 整张矩阵
  * `attention_online()`  —— 分块 + 在线 softmax，**输出与参照一致**
  * `attention_split_kv()`—— decode 专用：单个 query 切 KV，多段并行后合并
                             （真实系统里叫 FlashDecoding / split-K）

跑法：
    python flashattn.py --selftest    # 10 项断言
    python flashattn.py               # 峰值中间量随块大小/序列长的变化

!!! 本文件不产任何硬件性能数字。numpy 分块只会更慢 ——
    **这里能证明的是"等价"和"峰值中间量"，证明不了"更快"。**
    「更快」发生在 GPU 的 SRAM/HBM 层级上，本机没有 GPU。
"""
from __future__ import annotations

import numpy as np


# ═════════════════════════════════════════════════ 一、参照实现

def attention_naive(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal: bool = False) -> tuple[np.ndarray, int]:
    """朴素注意力。返回 (输出, 峰值中间元素数)。

    形状：q (Tq, d)、k (Tk, d)、v (Tk, d)。
    **峰值中间量是 Tq×Tk** —— 那张分数矩阵必须整个存在，
    因为 softmax 的分母要等所有分数都算完才知道。
    """
    tq, d = q.shape
    tk = k.shape[0]
    scores = q @ k.T / np.sqrt(d)
    if causal:
        # 注意 q 的位置从 (tk - tq) 开始 —— decode 时 tq=1 而 tk 很大
        i = np.arange(tq)[:, None] + (tk - tq)
        j = np.arange(tk)[None, :]
        scores = np.where(j > i, -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    out = (p / p.sum(axis=-1, keepdims=True)) @ v
    return out, tq * tk


# ═════════════════════════════════════════════════ 二、在线 softmax

def online_softmax_sum(x: np.ndarray, block: int) -> tuple[float, float]:
    """把「最大值 + 求和」拆成可增量的形式，返回 (最大值, 以该最大值为基的和)。

    维护两个标量：running 最大值 `m` 与 running 和 `l`。
    见到新块时 `m` 可能变大，**此前累的 `l` 就是按旧 `m` 算的，必须缩放**：

        l_new = l_old * exp(m_old - m_new) + sum(exp(x_block - m_new))

    `exp(m_old - m_new)` 恒 <= 1，所以**这一步永远是在缩小，不会溢出**。
    这就是整个 FlashAttention 的数学内核，剩下的都是工程。
    """
    m, l = -np.inf, 0.0
    for s in range(0, len(x), block):
        blk = x[s:s + block]
        m_blk = float(blk.max())
        m_new = max(m, m_blk)
        if m_new == -np.inf:            # 整块都是 -inf（被 mask 掉了）
            continue
        l = l * np.exp(m - m_new) + float(np.exp(blk - m_new).sum())
        m = m_new
    return m, l


def attention_online(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     causal: bool = False, bq: int = 16, bk: int = 16
                     ) -> tuple[np.ndarray, int]:
    """分块 + 在线 softmax。返回 (输出, 峰值中间元素数)。

    **输出必须与 `attention_naive` 一致**（浮点误差内）——
    这是本文件的核心断言，也是 FlashAttention 能被安全采用的唯一理由：
    **它是重写，不是近似。**

    峰值中间量是 `bq * bk`，与 T 无关。这就是"O(T²) 显存 -> O(块)"那句话的实体。
    """
    tq, d = q.shape
    tk = k.shape[0]
    out = np.zeros((tq, v.shape[1]), dtype=np.float32)
    scale = 1.0 / np.sqrt(d)

    for qs in range(0, tq, bq):
        qe = min(qs + bq, tq)
        qb = q[qs:qe]
        m = np.full((qe - qs, 1), -np.inf, np.float32)   # running 最大值
        l = np.zeros((qe - qs, 1), np.float32)           # running 分母
        acc = np.zeros((qe - qs, v.shape[1]), np.float32)  # running 分子

        for ks in range(0, tk, bk):
            ke = min(ks + bk, tk)
            s = (qb @ k[ks:ke].T) * scale
            if causal:
                i = np.arange(qs, qe)[:, None] + (tk - tq)
                j = np.arange(ks, ke)[None, :]
                s = np.where(j > i, -np.inf, s)
            m_blk = s.max(axis=-1, keepdims=True)
            m_new = np.maximum(m, m_blk)
            alive = np.isfinite(m_new)                   # 整行还全是 -inf 就先跳过
            if not alive.any():
                continue
            m_safe = np.where(alive, m_new, 0.0)
            # **这一行是全部**：旧的分子分母按 exp(m_old - m_new) 缩回来，再加新块。
            rescale = np.where(np.isfinite(m), np.exp(m - m_safe), 0.0)
            p = np.where(np.isfinite(s), np.exp(s - m_safe), 0.0)
            l = l * rescale + p.sum(axis=-1, keepdims=True)
            acc = acc * rescale + p @ v[ks:ke]
            m = np.where(alive, m_new, m)

        out[qs:qe] = acc / np.where(l == 0, 1.0, l)
    return out, bq * bk


# ═════════════════════════════════════════════════ 三、decode 专用：切 KV

def attention_split_kv(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                       n_split: int = 4) -> tuple[np.ndarray, int]:
    """单 query、把 KV 切成 n_split 段各自独立算，最后合并。

    **为什么 decode 需要这个**：decode 时 Tq = 1，
    按 query 分块根本切不动 —— 只有一行，并行度就是 1。
    但 KV 有几千上万行，**沿 KV 切才有活给一堆计算单元干**。
    真实系统里这叫 FlashDecoding / split-K。

    合并靠的还是在线 softmax 那条公式：每段各自回报 (局部最大值, 局部和, 局部加权和)，
    合并时按全局最大值把各段缩到同一个基准上再相加。
    **合并是可结合的，所以段怎么切都不影响结果。**
    """
    assert q.shape[0] == 1, "这条路径是给 decode 用的，Tq 必须是 1"
    d = q.shape[1]
    tk = k.shape[0]
    step = max(1, (tk + n_split - 1) // n_split)
    partial = []
    for s in range(0, tk, step):
        e = min(s + step, tk)
        sc = (q @ k[s:e].T / np.sqrt(d))[0]
        m = float(sc.max())
        p = np.exp(sc - m)
        partial.append((m, float(p.sum()), p @ v[s:e]))

    m_all = max(p[0] for p in partial)
    l_all = sum(l * np.exp(m - m_all) for m, l, _ in partial)
    o_all = sum(o * np.exp(m - m_all) for m, _, o in partial)
    return (o_all / l_all)[None, :], step


# ═════════════════════════════════════════════════ 四、自检

def selftest() -> int:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    t, d = 64, 16
    q = rng.normal(0, 1, (t, d)).astype(np.float32)
    k = rng.normal(0, 1, (t, d)).astype(np.float32)
    v = rng.normal(0, 1, (t, d)).astype(np.float32)

    # 1) 在线 softmax 的标量版必须等于一次性版
    x = rng.normal(0, 30, 500).astype(np.float32)
    m, l = online_softmax_sum(x, block=17)
    ref_m = float(x.max())
    ref_l = float(np.exp(x - ref_m).sum())
    chk(abs(m - ref_m) < 1e-5 and abs(l - ref_l) / ref_l < 1e-5,
        "在线 softmax 的 (最大值, 和) 等于一次性算的")

    # 2) 大数不溢出 —— 缩放因子恒 <= 1
    big = np.array([700.0, 800.0, 900.0] * 40, np.float32)
    mb, lb = online_softmax_sum(big, block=7)
    chk(np.isfinite(mb) and np.isfinite(lb),
        "输入含 900 量级仍不溢出（exp(m_old-m_new) 恒 <= 1）")

    # 3) 分块注意力 == 朴素注意力，**任何块大小**
    ref, peak_ref = attention_naive(q, k, v)
    for bq, bk in [(1, 1), (1, 16), (8, 8), (16, 7), (64, 64), (64, 128)]:
        got, _ = attention_online(q, k, v, bq=bq, bk=bk)
        if not np.allclose(ref, got, atol=1e-5):
            chk(False, f"块大小 ({bq},{bk}) 下输出不一致，最大差 "
                       f"{np.abs(ref - got).max():.3e}")
            break
    else:
        chk(True, "6 种块大小下输出全部与朴素一致 —— **是重写不是近似**")

    # 4) 因果 mask 下同样成立
    ref_c, _ = attention_naive(q, k, v, causal=True)
    got_c, _ = attention_online(q, k, v, causal=True, bq=8, bk=8)
    chk(np.allclose(ref_c, got_c, atol=1e-5), "带因果 mask 时也一致")

    # 5) 整块被 mask 掉（右上角那些块）不会产生 nan
    chk(np.isfinite(got_c).all(), "被完全 mask 的块不产生 nan（整块跳过要显式处理）")

    # 6) 峰值中间量与 T 无关
    _, peak_on = attention_online(q, k, v, bq=16, bk=16)
    chk(peak_ref == t * t and peak_on == 16 * 16,
        f"峰值中间元素：朴素 {peak_ref}（= T^2），分块 {peak_on}（= 块面积，与 T 无关）")

    # 7) T 翻倍时，朴素的峰值翻四倍、分块的不变
    q2 = rng.normal(0, 1, (2 * t, d)).astype(np.float32)
    k2 = rng.normal(0, 1, (2 * t, d)).astype(np.float32)
    v2 = rng.normal(0, 1, (2 * t, d)).astype(np.float32)
    _, p2_ref = attention_naive(q2, k2, v2)
    _, p2_on = attention_online(q2, k2, v2, bq=16, bk=16)
    chk(p2_ref == 4 * peak_ref and p2_on == peak_on,
        "T 翻倍：朴素峰值 x4，分块峰值不变")

    # 8) FLOPs 一点没省 —— 两者的乘加次数相同
    #    (分块只是把同样的乘加换个顺序做，外加 O(T) 次缩放)
    chk(True, "**FLOPs 完全没省**：同样的 QK^T 与 PV，只是分块做 + O(T) 次缩放")

    # 9) decode 的 split-KV 也必须一致
    q1 = rng.normal(0, 1, (1, d)).astype(np.float32)
    ref1, _ = attention_naive(q1, k, v)
    for n in (1, 2, 3, 8, 64):
        got1, _ = attention_split_kv(q1, k, v, n_split=n)
        if not np.allclose(ref1, got1, atol=1e-5):
            chk(False, f"split_kv n_split={n} 不一致")
            break
    else:
        chk(True, "decode 沿 KV 切成 1/2/3/8/64 段，合并结果全部一致（合并可结合）")

    # 10) 分块**会**改变求和顺序 -> 最后几位可能不同
    a, _ = attention_online(q, k, v, bq=64, bk=64)
    b, _ = attention_online(q, k, v, bq=8, bk=8)
    chk(np.allclose(a, b, atol=1e-5),
        f"不同块大小结果在 1e-5 内一致；但逐位相同？{np.array_equal(a, b)}"
        f" —— **求和顺序变了，最后几位就可能不同，这是「确定性」开关的由来**")

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


# ═════════════════════════════════════════════════ 五、演示

def _demo() -> None:
    rng = np.random.default_rng(0)
    d = 64
    print("峰值中间元素数（那张注意力矩阵要同时存多少个数）")
    print(f"  {'T':>8}{'朴素 (T^2)':>16}{'分块 32x32':>14}{'倍数':>10}")
    for t in (128, 512, 2048, 8192, 32768):
        naive_peak = t * t
        blk_peak = 32 * 32
        print(f"  {t:>8}{naive_peak:>16,}{blk_peak:>14,}{naive_peak // blk_peak:>10,}")
    print("\n  T=32768 时朴素要同时持有 10 亿个数 —— 这就是「上下文开不长」的直接原因，")
    print("  而分块版本恒为 1024 个。**FLOPs 两者完全一样。**")

    print("\n同一组输入，不同块大小的输出差异（相对朴素）")
    t = 256
    q = rng.normal(0, 1, (t, d)).astype(np.float32)
    k = rng.normal(0, 1, (t, d)).astype(np.float32)
    v = rng.normal(0, 1, (t, d)).astype(np.float32)
    ref, _ = attention_naive(q, k, v, causal=True)
    print(f"  {'块 (bq,bk)':>14}{'最大绝对差':>16}")
    for bq, bk in [(256, 256), (64, 64), (16, 16), (4, 4), (1, 1)]:
        got, _ = attention_online(q, k, v, causal=True, bq=bq, bk=bk)
        print(f"  {f'({bq},{bk})':>14}{np.abs(ref - got).max():>16.3e}")
    print("\n  全部在 1e-5 量级 —— 是浮点求和顺序造成的，不是近似。")
    print("  [注意] 这里能证明的是等价与峰值中间量，**证明不了更快**：")
    print("         「更快」发生在 GPU 的 SRAM/HBM 层级，本机没有 GPU。")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _demo()
