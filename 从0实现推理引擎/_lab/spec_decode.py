"""spec_decode.py —— 投机解码：**它不是近似，输出分布逐点等于原模型。**

想法很简单：用一个便宜的小模型（draft）一口气猜 γ 个 token，
再让大模型（target）**一次前向**把这 γ+1 个位置全部验一遍。
猜对了就白赚，猜错了就从错的那个位置重来。

两个问题必须回答，否则这套东西不能用：

  1. **验完之后输出的还是不是原模型的分布？**
     是，而且是**精确**相等，不是近似。本文件用 20 万次采样把它测出来，
     并同时给一个**看起来很合理但有偏**的朴素验收版做反例。
  2. **为什么"多验 γ 个位置"几乎不要钱？**
     因为 decode 是访存受限（见 `minigpt.py` 的 `analytic_cost`）。
     **关键在于这 γ+1 个 token 属于同一条请求，共用同一份 KV** ——
     所以 KV 只读一遍，而不是像多请求批处理那样每条各读一份。
     `verify_cost()` 把这笔账算给你看。

**接受-拒绝规则**（Leviathan 等人的形式），逐条都有对应代码：

    从 q 采一个 x；以概率 min(1, p(x)/q(x)) 接受；
    否则从 **norm(max(0, p - q))** 里重采一个，并就此打住。

它成立的一行证明：
    P(输出 x) = min(q(x), p(x)) + max(0, p(x)-q(x)) = p(x)   ← 恒等式，与 q 无关

跑法：
    python spec_decode.py --selftest   # 12 项断言（含 20 万次采样的分布检验）
    python spec_decode.py              # 接受率 / 期望产出 / 成本账 三张表

!!! 本文件不产任何硬件性能数字。所有"加速比"都是**按解析成本模型算的估计**，
    并且明确标注了它在什么条件下不成立。
"""
from __future__ import annotations

import numpy as np


# ═════════════════════════════════════════════════ 一、核心：接受-拒绝

def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`norm(max(0, p - q))` —— 被拒绝之后要从这里重采。

    直觉：`min(p, q)` 那部分已经被"接受"这条路径覆盖了，
    剩下没被覆盖的正是 `p - q` 的正部。从它里面补一发，总量刚好补齐成 p。
    """
    r = np.maximum(p - q, 0.0)
    s = r.sum()
    if s <= 0:                      # p 完全被 q 覆盖（例如 p == q），理论上到不了这里
        return p / p.sum()
    return r / s


def spec_sample_one(p: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> int:
    """γ=1 的一步：**输出分布精确等于 p**。

    注意接受概率用的是 `p(x)/q(x)`，**分母是 draft 的概率**。
    写反成 `q(x)/p(x)` 不会崩，只会让分布悄悄偏掉 —— `selftest()` 有反例。
    """
    x = int(rng.choice(len(q), p=q))
    if rng.random() < min(1.0, float(p[x] / q[x])):
        return x
    return int(rng.choice(len(p), p=residual(p, q)))


def spec_sample_one_naive(p: np.ndarray, q: np.ndarray,
                          rng: np.random.Generator) -> int:
    """**故意写错的版本**：`p(x) >= q(x)` 就收，否则直接从 p 重采。

    它看起来很合理 —— "draft 高估了就重来" —— 而且**跑起来完全正常**，
    生成的文本也通顺。但它的分布不是 p。
    只留给自检当反例，不要在别处调用。
    """
    x = int(rng.choice(len(q), p=q))
    if p[x] >= q[x]:
        return x
    return int(rng.choice(len(p), p=p))


def spec_step(p_rows: np.ndarray, q_rows: np.ndarray,
              rng: np.random.Generator) -> tuple[list[int], int]:
    """完整一轮：draft 先猜 γ 个，target 一次验完。

    `p_rows` 形状 (γ+1, V)：target 在 γ+1 个位置上的分布
    （**最后一行是"全接受时用来多出一个 token"的那一位**，这就是 γ+1 的来源）。
    `q_rows` 形状 (γ, V)：draft 在前 γ 个位置上的分布。

    返回 (本轮产出的 token, 接受了几个 draft token)。
    **产出至少 1 个**：即使第一个就被拒，也会从残差里补一个。
    这条保证很重要 —— 它意味着投机解码**不可能比逐个解码更慢地推进**（步数意义上）。
    """
    gamma = len(q_rows)
    out: list[int] = []
    for i in range(gamma):
        x = int(rng.choice(q_rows.shape[1], p=q_rows[i]))
        if rng.random() < min(1.0, float(p_rows[i][x] / q_rows[i][x])):
            out.append(x)
            continue
        out.append(int(rng.choice(p_rows.shape[1], p=residual(p_rows[i], q_rows[i]))))
        return out, i                     # 拒了就停，后面猜的全丢
    # γ 个全中：可以**免费**再多出一个（第 γ+1 位的分布 target 已经算出来了）
    out.append(int(rng.choice(p_rows.shape[1], p=p_rows[gamma])))
    return out, gamma


# ═════════════════════════════════════════════════ 二、产出与成本

def expected_tokens(alpha: float, gamma: int) -> float:
    """接受率为 α 时，一轮的期望产出 `(1 - α^(γ+1)) / (1 - α)`。

    α=1 时退化为 γ+1（全中）。α→0 时趋于 1（全靠残差那一发兜底）。
    **它对 γ 是次线性的**：α=0.8 时 γ 从 4 加到 8，期望产出只从 3.36 涨到 4.16，
    而 draft 的成本是线性涨的 —— **所以 γ 有最优值，不是越大越好。**
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return float((1 - alpha ** (gamma + 1)) / (1 - alpha))


def verify_cost(d_model: int, n_layer: int, n_head: int, d_head: int,
                vocab: int, ctx: int, n_tok: int, dtype_bytes: int = 4) -> dict:
    """一次前向验 `n_tok` 个 token 的解析成本。

    **和多请求批处理的关键区别在 KV 这一项**：
    这 n_tok 个 token 属于**同一条请求**，共用同一份历史 KV，
    所以 `bytes_kv` **与 n_tok 无关**（读一遍就够）；
    而多请求批处理时每条请求各有一份 KV，那一项是要乘 batch 的
    （对比 `minigpt.py` 的 `analytic_cost`，它建模的是后者）。

    **这就是投机解码"验证几乎白送"的全部原因**：
    分子随 n_tok 线性涨，分母几乎不动。
    """
    matmul_params = n_layer * (4 * d_model ** 2 + 8 * d_model ** 2) + d_model * vocab
    flops = 2 * n_tok * matmul_params + n_tok * n_layer * n_head * 4 * ctx * d_head
    bytes_w = dtype_bytes * (matmul_params + 2 * n_tok * d_model)
    bytes_kv = ctx * 2 * n_layer * d_model * dtype_bytes      # <- 不乘 n_tok
    total_b = bytes_w + bytes_kv
    return {"n_tok": n_tok, "flops": flops, "bytes": total_b,
            "arithmetic_intensity": round(flops / total_b, 3)}


def speedup_estimate(alpha: float, gamma: int, draft_ratio: float,
                     verify_overhead: float = 0.0) -> float:
    """粗估加速比。**这是解析估计，不是实测。**

        每轮产出 = expected_tokens(α, γ)
        每轮成本 = 1（target 验一次） + γ·draft_ratio（draft 跑 γ 步）
                   + verify_overhead（验 γ+1 个位置比验 1 个多出来的那点）

    `draft_ratio` = draft 一步 / target 一步 的成本比。
    `verify_overhead` 在**小 batch**时接近 0（访存受限，多算几个位置几乎白送），
    但在**大 batch**时会显著大于 0 —— 那时 target 已经算力受限，
    多出来的 FLOPs 是要真花时间的。**这就是"高负载下投机解码可能反而变慢"的机制。**
    """
    return expected_tokens(alpha, gamma) / (1.0 + gamma * draft_ratio + verify_overhead)


# ═════════════════════════════════════════════════ 三、自检

def _rand_dist(rng: np.random.Generator, v: int, temp: float = 1.0) -> np.ndarray:
    x = rng.normal(0, temp, v)
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float64)


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    """全变差距离，0 表示完全相同。"""
    return float(0.5 * np.abs(a - b).sum())


def selftest() -> int:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    V, N = 8, 200_000
    p = _rand_dist(rng, V, 1.5)
    q = _rand_dist(rng, V, 1.5)

    # ---- 分布不变性：本文件的核心命题 ----
    emp = np.bincount([spec_sample_one(p, q, rng) for _ in range(N)],
                      minlength=V) / N
    tv = _tv(emp, p)
    chk(tv < 0.005,
        f"20 万次采样的经验分布与 target 的全变差 = {tv:.5f} —— **投机解码不是近似**")

    emp_n = np.bincount([spec_sample_one_naive(p, q, rng) for _ in range(N)],
                        minlength=V) / N
    tv_n = _tv(emp_n, p)
    chk(tv_n > 20 * tv,
        f"朴素验收版的全变差 = {tv_n:.5f}，是正确版的 {tv_n / tv:.0f} 倍"
        f" —— **它跑得很正常，只是分布错了**")

    # ---- 恒等式本身 ----
    Z = float(np.maximum(p - q, 0).sum())
    rec = np.minimum(p, q) + Z * residual(p, q)
    chk(np.allclose(rec, p, atol=1e-12),
        "恒等式 min(q,p) + Z·norm(max(0,p-q)) == p 精确成立（与 q 无关）")
    chk(abs(float(np.minimum(p, q).sum()) - (1 - Z)) < 1e-12,
        f"整体接受率 = sum min(p,q) = {1 - Z:.4f}，正好是 1 - 残差质量")

    # ---- draft 越像 target，接受率越高 ----
    chk(_tv(p, p) == 0.0 and float(np.minimum(p, p).sum()) == 1.0,
        "draft 与 target 完全一致时接受率 = 1（永不拒绝）")
    far = _rand_dist(rng, V, 6.0)
    chk(float(np.minimum(p, far).sum()) < float(np.minimum(p, q).sum()),
        "draft 分布离 target 越远，接受率越低（接受率 = 两个分布的重叠面积）")

    # ---- 多 token 版 ----
    gamma = 4
    same = np.stack([p] * (gamma + 1))
    outs = [spec_step(same, np.stack([p] * gamma), rng) for _ in range(200)]
    chk(all(n == gamma for _, n in outs) and all(len(o) == gamma + 1 for o, _ in outs),
        f"draft==target 时每轮必然全接受，产出 γ+1 = {gamma + 1} 个 token")

    bad = np.stack([_rand_dist(rng, V, 8.0) for _ in range(gamma)])
    outs2 = [spec_step(np.stack([p] * (gamma + 1)), bad, rng) for _ in range(400)]
    chk(all(len(o) >= 1 for o, _ in outs2),
        "**即使一个都没猜中，每轮也至少产出 1 个 token**（残差那一发兜底）")
    chk(np.mean([n for _, n in outs2]) < np.mean([n for _, n in outs]),
        "draft 很差时平均接受数明显更低")

    # ---- 期望产出公式 ----
    for a, g in [(0.5, 4), (0.8, 4), (0.9, 8)]:
        sim = np.mean([min(sum(1 for _ in iter(
            lambda: rng.random() < a, False)), g) for _ in range(20000)]) + 1
        want = expected_tokens(a, g)
        if abs(sim - want) > 0.05:
            chk(False, f"α={a} γ={g}：模拟 {sim:.3f} vs 公式 {want:.3f}")
            break
    else:
        chk(True, "期望产出 (1-α^(γ+1))/(1-α) 与伯努利模拟一致")

    chk(expected_tokens(1.0, 4) == 5.0 and abs(expected_tokens(1e-9, 4) - 1.0) < 1e-6,
        "两个端点：α=1 时产出 γ+1；α→0 时产出 1")

    # ---- 成本账：验证为什么几乎白送 ----
    args = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000, ctx=2048)
    c1 = verify_cost(**args, n_tok=1)
    c5 = verify_cost(**args, n_tok=5)
    grow_f = c5["flops"] / c1["flops"]
    grow_b = c5["bytes"] / c1["bytes"]
    chk(grow_f > 4.9 and grow_b < 1.05,
        f"验 5 个 token vs 验 1 个：FLOPs x{grow_f:.2f}，访存仅 x{grow_b:.3f}"
        f" —— **同一条请求共用一份 KV，所以验证几乎白送**")

    # ---- 什么时候反而变慢 ----
    fast = speedup_estimate(0.8, 4, draft_ratio=0.05, verify_overhead=0.0)
    slow = speedup_estimate(0.8, 4, draft_ratio=0.05, verify_overhead=4.0)
    chk(fast > 2.0 > slow,
        f"小 batch 估计加速 {fast:.2f}x；一旦 target 转为算力受限"
        f"（verify_overhead=4）就跌到 {slow:.2f}x —— **高负载下可能得不偿失**")

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


# ═════════════════════════════════════════════════ 四、演示

def _demo() -> None:
    rng = np.random.default_rng(1)
    V = 32
    p = _rand_dist(rng, V, 2.0)

    print("一、接受率 = draft 与 target 两个分布的重叠面积 sum(min(p,q))")
    print(f"  {'draft 温度':>12}{'重叠面积 α':>14}{'γ=4 期望产出':>16}{'γ=8':>10}")
    for temp in (2.0, 2.5, 3.0, 4.0, 6.0):
        q = _rand_dist(np.random.default_rng(7), V, temp)
        a = float(np.minimum(p, q).sum())
        print(f"  {temp:>12.1f}{a:>14.3f}{expected_tokens(a, 4):>16.2f}"
              f"{expected_tokens(a, 8):>10.2f}")
    print("  **产出对 γ 是次线性的，而 draft 成本对 γ 是线性的 —— 所以 γ 有最优值。**")

    print("\n二、γ 的最优值（draft 成本 = target 的 5%，小 batch）")
    print(f"  {'γ':>4}" + "".join(f"{f'α={a}':>10}" for a in (0.5, 0.7, 0.8, 0.9)))
    best = {a: (0, 0.0) for a in (0.5, 0.7, 0.8, 0.9)}
    for g in (1, 2, 4, 6, 8, 12, 16):
        row = f"  {g:>4}"
        for a in (0.5, 0.7, 0.8, 0.9):
            s = speedup_estimate(a, g, 0.05)
            row += f"{s:>10.2f}"
            if s > best[a][1]:
                best[a] = (g, s)
        print(row)
    print("  各列最优 γ：" + "  ".join(f"α={a} -> γ={g}(x{s:.2f})"
                                    for a, (g, s) in best.items()))

    print("\n三、为什么「多验几个」几乎白送（解析成本，非实测）")
    args = dict(d_model=4096, n_layer=32, n_head=32, d_head=128,
                vocab=32000, ctx=2048)
    print(f"  {'验几个':>8}{'FLOPs 倍数':>14}{'访存倍数':>12}{'算术强度':>12}")
    base = verify_cost(**args, n_tok=1)
    for n in (1, 2, 4, 8, 16):
        c = verify_cost(**args, n_tok=n)
        print(f"  {n:>8}{c['flops'] / base['flops']:>14.2f}"
              f"{c['bytes'] / base['bytes']:>12.3f}"
              f"{c['arithmetic_intensity']:>12.2f}")
    print("  访存几乎不动，因为这几个 token **同属一条请求、共用一份 KV**。")
    print("  （多请求批处理不是这样 —— 那时每条请求各有一份 KV，见 minigpt.analytic_cost）")
    print("  [注意] 以上全是解析模型算出来的，**不是任何硬件上的实测**。")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _demo()
