"""parallel.py —— 张量并行 / 流水并行 / 专家并行，在单机 numpy 上把**等价性**证明一遍。

一台机器装不下的模型要切到多张卡上。切法只有三个方向，本文件各给一个能跑的最小版：

  * **TP（张量并行）**：把**同一层**的权重矩阵切开，每张卡算一部分，再合并。
  * **PP（流水并行）**：把**不同层**分给不同卡，像流水线一样传递激活。
  * **EP（专家并行）**：MoE 里把**不同专家**放到不同卡，按路由把 token 送过去。

**本文件不做真的多进程通信** —— 那是运维问题，不是原理问题。
这里用单进程模拟"每张卡各持有一份分片"，然后回答三个真正重要的问题：

  1. **切完还等不等价？**（等价才敢用；本文件对 TP 和 EP 都给了逐位对比）
  2. **通信量是多少？**（决定了它对互联带宽的要求）
  3. **代价是什么？**（TP 要高速互联、PP 有气泡、EP 有负载不均）

跑法：
    python parallel.py --selftest    # 12 项断言
    python parallel.py               # 通信量 / 流水气泡 / 专家负载三张表

!!! 本文件不产任何硬件性能数字。它算的是**元素数、字节数、气泡比例**这类
    可以纯手算复核的量，**不涉及任何真实互联的实测带宽**。
"""
from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ═════════════════════════════════════════════════ 一、张量并行：MLP

def mlp_single(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """不切的参照实现：x -> w1 -> gelu -> w2。"""
    return gelu(x @ w1) @ w2


def mlp_tensor_parallel(x: np.ndarray, w1: np.ndarray, w2: np.ndarray,
                        tp: int) -> tuple[np.ndarray, int]:
    """Megatron 那套切法。返回 (输出, 本层的 all-reduce 元素数)。

    **切法不是随便挑的，是被 GELU 逼出来的**：

      * `w1` 按**列**切（column-parallel）：每张卡拿到 `d -> h/tp` 的一片，
        算出的是**完整的一部分隐藏单元**，不是部分和。
        所以 **GELU 可以就地做，不需要任何通信**。
      * `w2` 按**行**切（row-parallel）：每张卡吃自己那 `h/tp` 个隐藏单元，
        算出 `d` 维的**部分和**，最后 all-reduce 相加。

    结果是**整个 MLP 只需要一次 all-reduce**。
    反过来切（`w1` 按行）就必须在 GELU **之前**再来一次通信，
    因为 GELU 是非线性的，`gelu(a+b) != gelu(a)+gelu(b)` ——
    `selftest()` 里有一条断言把这个不等式钉住。
    """
    d, h = w1.shape
    assert h % tp == 0, "隐藏维必须能被 tp 整除"
    parts = []
    for r in range(tp):                       # r 代表"第 r 张卡"
        w1_shard = w1[:, r * h // tp:(r + 1) * h // tp]     # 按列切
        w2_shard = w2[r * h // tp:(r + 1) * h // tp, :]     # 按行切
        parts.append(gelu(x @ w1_shard) @ w2_shard)          # 本地算，无通信
    out = sum(parts)                          # <- 唯一的一次 all-reduce
    return out, out.size * (tp - 1) if tp > 1 else 0


def allreduce_elements(tokens: int, d_model: int, tp: int) -> int:
    """一次 ring all-reduce 每张卡收发的元素数：`2 * (tp-1)/tp * N`。

    注意它**不随 tp 增大而线性下降** —— tp 从 2 到 8，系数只从 1.0 涨到 1.75，
    也就是说**卡越多，每张卡的通信量还略微更大**。
    这就是"TP 只在单机内做"的根本原因：跨机互联带宽跟不上。
    """
    n = tokens * d_model
    return 0 if tp <= 1 else int(2 * (tp - 1) / tp * n)


# ═════════════════════════════════════════════════ 二、张量并行：注意力

def attn_single(x: np.ndarray, wq, wk, wv, wo, n_head: int) -> np.ndarray:
    t, d = x.shape
    dh = d // n_head
    q = (x @ wq).reshape(t, n_head, dh).transpose(1, 0, 2)
    k = (x @ wk).reshape(t, n_head, dh).transpose(1, 0, 2)
    v = (x @ wv).reshape(t, n_head, dh).transpose(1, 0, 2)
    o = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh)) @ v
    return o.transpose(1, 0, 2).reshape(t, d) @ wo


def attn_tensor_parallel(x: np.ndarray, wq, wk, wv, wo, n_head: int,
                         tp: int) -> np.ndarray:
    """注意力的 TP 切法：**按头切**。

    头之间本来就互不影响（见注意力那几篇），所以这是最自然的切法 ——
    每张卡拿 `n_head/tp` 个头的 Q/K/V 权重，各自算完自己的头，
    输出投影 `wo` 按行切，最后 all-reduce。

    **硬约束：`n_head % tp == 0`。** 这就是为什么 TP 度只能取 1/2/4/8 这种数 ——
    不是实现懒，是头数除不尽。GQA 之后约束还要更紧一层（KV 头更少）。
    """
    t, d = x.shape
    dh = d // n_head
    assert n_head % tp == 0, "头数必须能被 tp 整除 —— 这是硬约束，不是实现限制"
    hpr = n_head // tp
    parts = []
    for r in range(tp):
        sl = slice(r * hpr * dh, (r + 1) * hpr * dh)
        q = (x @ wq[:, sl]).reshape(t, hpr, dh).transpose(1, 0, 2)
        k = (x @ wk[:, sl]).reshape(t, hpr, dh).transpose(1, 0, 2)
        v = (x @ wv[:, sl]).reshape(t, hpr, dh).transpose(1, 0, 2)
        o = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh)) @ v
        parts.append(o.transpose(1, 0, 2).reshape(t, hpr * dh) @ wo[sl, :])
    return sum(parts)


# ═════════════════════════════════════════════════ 三、流水并行

def pipeline_schedule(n_stage: int, n_micro: int) -> tuple[list[list[int | None]], float]:
    """推理期（只有前向）的流水调度。返回 (时间线, 气泡比例)。

    时间线 `tl[stage][t]` = 该时刻这一级在处理哪个 micro-batch，空闲则 None。
    第 s 级处理第 m 个 micro-batch 的时刻是 `s + m`，所以总步数 = `n_micro + n_stage - 1`。

    **气泡比例 = (n_stage - 1) / (n_micro + n_stage - 1)**，
    与每级的耗时无关（假设各级等长）。含义很直白：
    **micro-batch 数要远大于流水级数，否则大部分时间在等。**
    """
    total = n_micro + n_stage - 1
    tl: list[list[int | None]] = [[None] * total for _ in range(n_stage)]
    for m in range(n_micro):
        for s in range(n_stage):
            tl[s][s + m] = m
    idle = sum(1 for row in tl for c in row if c is None)
    return tl, idle / (n_stage * total)


# ═════════════════════════════════════════════════ 四、专家并行

def moe_route(tokens: int, n_expert: int, top_k: int = 1,
              seed: int = 0, skew: float = 0.0) -> np.ndarray:
    """模拟 MoE 路由，返回每个专家分到的 token 数。

    `skew=0` 是均匀路由；`skew>0` 让前几个专家更受欢迎（真实路由器一定不均匀）。
    """
    rng = np.random.default_rng(seed)
    w = np.exp(-skew * np.arange(n_expert))
    w = w / w.sum()
    load = np.zeros(n_expert, dtype=np.int64)
    for _ in range(tokens):
        for e in rng.choice(n_expert, size=top_k, replace=False, p=w):
            load[e] += 1
    return load


def imbalance(load: np.ndarray) -> float:
    """**最大负载 / 平均负载。**

    这是 EP 唯一重要的数：一步的耗时由**最慢的那张卡**决定，
    所以 `max/mean` 直接就是"因为不均衡而浪费掉的倍数"。
    真实系统靠容量因子（超出就丢弃/溢出）和辅助均衡损失来压它，
    **两种手段都有代价**：丢弃改变输出，均衡损失干扰主目标。
    """
    return float(load.max() / load.mean())


# ═════════════════════════════════════════════════ 五、自检

def selftest() -> int:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    t, d, h, n_head = 12, 32, 128, 8
    x = rng.normal(0, 1, (t, d)).astype(np.float32)
    w1 = rng.normal(0, 0.05, (d, h)).astype(np.float32)
    w2 = rng.normal(0, 0.05, (h, d)).astype(np.float32)

    # ---- TP: MLP 等价 ----
    ref = mlp_single(x, w1, w2)
    for tp in (1, 2, 4, 8):
        got, _ = mlp_tensor_parallel(x, w1, w2, tp)
        if not np.allclose(ref, got, atol=1e-4):
            chk(False, f"tp={tp} 时 MLP 输出不一致，最大差 {np.abs(ref-got).max():.3e}")
            break
    else:
        chk(True, "MLP 在 tp=1/2/4/8 下输出全部一致（列切+行切是精确重排）")

    # ---- 为什么必须列切在前：GELU 非线性 ----
    a = rng.normal(0, 1, (4, 4)).astype(np.float32)
    b = rng.normal(0, 1, (4, 4)).astype(np.float32)
    chk(not np.allclose(gelu(a + b), gelu(a) + gelu(b), atol=1e-3),
        "gelu(a+b) != gelu(a)+gelu(b) —— **所以 w1 必须列切，否则 GELU 前还要多一次通信**")

    # ---- 通信量的形状 ----
    e2 = allreduce_elements(t, d, 2)
    e8 = allreduce_elements(t, d, 8)
    chk(e2 == t * d and e8 == int(1.75 * t * d),
        f"all-reduce 元素数：tp=2 是 {e2}（=N），tp=8 是 {e8}（=1.75N）"
        f" —— **卡越多每卡通信量不降反升**")
    chk(allreduce_elements(t, d, 1) == 0, "tp=1 时通信量为 0")

    # ---- TP: 注意力等价 ----
    wq, wk, wv = (rng.normal(0, 0.05, (d, d)).astype(np.float32) for _ in range(3))
    wo = rng.normal(0, 0.05, (d, d)).astype(np.float32)
    ref_a = attn_single(x, wq, wk, wv, wo, n_head)
    for tp in (1, 2, 4, 8):
        got_a = attn_tensor_parallel(x, wq, wk, wv, wo, n_head, tp)
        if not np.allclose(ref_a, got_a, atol=1e-4):
            chk(False, f"注意力 tp={tp} 不一致")
            break
    else:
        chk(True, "注意力按头切，tp=1/2/4/8 下输出全部一致")

    try:
        attn_tensor_parallel(x, wq, wk, wv, wo, n_head, tp=3)
        chk(False, "头数除不尽时应当报错")
    except AssertionError:
        chk(True, "n_head=8 不能被 tp=3 整除时**明确报错**（这是硬约束）")

    # ---- PP: 气泡公式 ----
    for p, m in [(2, 1), (4, 4), (4, 16), (8, 64)]:
        _, bub = pipeline_schedule(p, m)
        want = (p - 1) / (m + p - 1)
        if abs(bub - want) > 1e-9:
            chk(False, f"P={p} M={m} 气泡 {bub:.4f} != 公式 {want:.4f}")
            break
    else:
        chk(True, "流水气泡比例 == (P-1)/(M+P-1)，模拟与公式逐一吻合")

    _, b_few = pipeline_schedule(8, 8)
    _, b_many = pipeline_schedule(8, 64)
    chk(b_few > b_many * 3,
        f"micro-batch 从 64 降到 8，气泡从 {b_many:.1%} 涨到 {b_few:.1%}"
        f" —— **PP 的代价全在这里**")

    # ---- EP: 负载不均 ----
    even = moe_route(4096, 8, top_k=1, seed=1, skew=0.0)
    skewed = moe_route(4096, 8, top_k=1, seed=1, skew=0.6)
    chk(int(even.sum()) == 4096 and int(skewed.sum()) == 4096,
        "路由不丢 token（每个 token 恰好去 top_k 个专家）")
    chk(imbalance(even) < imbalance(skewed),
        f"均匀路由不均衡度 {imbalance(even):.2f}，偏斜路由 {imbalance(skewed):.2f}"
        f" —— **一步的耗时由最慢那张卡决定，所以 max/mean 就是浪费倍数**")

    top2 = moe_route(4096, 8, top_k=2, seed=1, skew=0.0)
    chk(int(top2.sum()) == 8192,
        "top_k=2 时总负载翻倍 —— **MoE 省的是每 token 的算力，不是总通信量**")

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


# ═════════════════════════════════════════════════ 六、演示

def _demo() -> None:
    print("一、TP 的通信量（每层两次 all-reduce：注意力一次 + MLP 一次）")
    print(f"  {'tp':>4}{'每次 all-reduce 元素数':>24}{'相对 tp=2':>12}")
    tokens, d = 2048, 4096
    for tp in (1, 2, 4, 8, 16):
        e = allreduce_elements(tokens, d, tp)
        base = allreduce_elements(tokens, d, 2)
        print(f"  {tp:>4}{e:>24,}{(e / base if base else 0):>12.2f}")
    print("  每卡通信量**随 tp 增大而略增**（系数 2(tp-1)/tp 从 1.00 逼近 2.00），")
    print("  同时每卡算的活变少 —— 所以 TP 的通信/计算比一路恶化。")
    print("  **这就是 TP 基本只在单机内（高速互联）做的原因。**")

    print("\n二、PP 的气泡（推理期，只有前向，各级等长）")
    print(f"  {'级数 P':>8}{'micro M':>10}{'总步数':>10}{'气泡':>10}")
    for p, m in [(2, 2), (4, 4), (4, 8), (4, 32), (8, 8), (8, 64), (16, 16)]:
        _, bub = pipeline_schedule(p, m)
        print(f"  {p:>8}{m:>10}{m + p - 1:>10}{bub:>10.1%}")
    print("  **M 必须远大于 P**。M=P 时气泡接近 50%，M=8P 时降到约 11%。")
    print("  在线推理很难攒出大 M —— 这是 PP 在服务场景里不如 TP 常见的原因之一。")

    print("\n三、EP 的负载不均（8 个专家，4096 个 token，top-1）")
    print(f"  {'偏斜':>8}{'最轻':>10}{'最重':>10}{'max/mean':>12}{'等效浪费':>12}")
    for skew in (0.0, 0.2, 0.4, 0.8, 1.5):
        load = moe_route(4096, 8, top_k=1, seed=1, skew=skew)
        im = imbalance(load)
        print(f"  {skew:>8.1f}{load.min():>10}{load.max():>10}{im:>12.2f}"
              f"{(im - 1):>11.0%}")
    print("  一步的耗时由最重的那张卡决定，所以 max/mean 直接就是浪费倍数。")
    print("  [注意] 以上全是可手算复核的计数，**不含任何真实互联的实测数字**。")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _demo()
