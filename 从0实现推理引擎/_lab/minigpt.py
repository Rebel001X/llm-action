"""minigpt.py —— 一个能跑的最小 Transformer 推理内核（纯 numpy，CPU）。

这个库讲「从 0 实现推理引擎」，所以第一件事必须是**有一个真的能跑的模型**，
否则后面讲 KV 缓存、分页、连续批处理全是空的。

设计约束（都是有意的）：
  * **只用 numpy**，不装 torch。目的是让每一步矩阵乘都露在外面，
    而不是藏在框架的一个算子里 —— 这是教学库，不是性能库。
  * **权重随机但固定 seed**。我们要验证的是「同一套权重下，
    朴素前向和 KV 缓存前向输出必须逐位一致」这类**正确性**命题，
    跟权重是不是训练出来的无关。
  * **CPU、小尺寸**。默认 4 层 / 4 头 / d=128 / 词表 512，
    在笔记本上一秒能跑几十步，足够把机制跑出来。

⚠️ 本文件里**可以**出现实测数字（它测的是我们自己这个玩具引擎在 CPU 上的行为），
但**绝不能**把这些数字外推到 vLLM/SGLang 这类真引擎上 —— 那需要 GPU，本机没有。
"""
from __future__ import annotations

import numpy as np

# ────────────────────────────────────────────────── 配置与权重


class Config:
    def __init__(self, n_layer=4, n_head=4, d_model=128, vocab=512,
                 max_seq=512, seed=0):
        assert d_model % n_head == 0
        self.n_layer, self.n_head, self.d_model = n_layer, n_head, d_model
        self.d_head = d_model // n_head
        self.vocab, self.max_seq, self.seed = vocab, max_seq, seed

    def __repr__(self) -> str:
        return (f"Config(L={self.n_layer}, H={self.n_head}, d={self.d_model}, "
                f"V={self.vocab})")

    # 一条请求每个 token 的 KV 字节数：2(K和V) × 层数 × d_model × 4字节(float32)
    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layer * self.d_model * 4

    # 参数量（不含 embedding 的近似：每层 4 个 d×d 投影 + FFN 两个 d×4d）
    def n_params(self) -> int:
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * 4 * self.d_model
        return self.n_layer * per_layer + self.vocab * self.d_model * 2


def init_weights(cfg: Config) -> dict:
    """随机但可复现的权重。scale 用 1/sqrt(d) 免得 softmax 一上来就饱和。"""
    rng = np.random.default_rng(cfg.seed)
    s = 1.0 / np.sqrt(cfg.d_model)

    def rnd(*shape):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    w = {"wte": rnd(cfg.vocab, cfg.d_model), "wpe": rnd(cfg.max_seq, cfg.d_model),
         "layers": []}
    for _ in range(cfg.n_layer):
        w["layers"].append({
            "wq": rnd(cfg.d_model, cfg.d_model), "wk": rnd(cfg.d_model, cfg.d_model),
            "wv": rnd(cfg.d_model, cfg.d_model), "wo": rnd(cfg.d_model, cfg.d_model),
            "w1": rnd(cfg.d_model, 4 * cfg.d_model), "w2": rnd(4 * cfg.d_model, cfg.d_model),
            "ln1_g": np.ones(cfg.d_model, np.float32), "ln1_b": np.zeros(cfg.d_model, np.float32),
            "ln2_g": np.ones(cfg.d_model, np.float32), "ln2_b": np.zeros(cfg.d_model, np.float32),
        })
    w["lnf_g"] = np.ones(cfg.d_model, np.float32)
    w["lnf_b"] = np.zeros(cfg.d_model, np.float32)
    return w


# ────────────────────────────────────────────────── 基本算子

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    """减最大值再取指数 —— 这不是"优化"，是**不这样会溢出**。

    exp(800) 在 float32 里直接是 inf，然后 inf/inf = nan。
    这一步就是 online softmax / FlashAttention 那一整套东西的起点。
    """
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ────────────────────────────────────────────────── 朴素前向（无缓存）

def forward_naive(cfg: Config, w: dict, tokens: list[int]) -> np.ndarray:
    """把整段序列重新算一遍。**每生成一个 token 就调用一次它，就是 O(n²) 的来源。**

    返回最后一个位置的 logits，形状 (vocab,)。
    """
    T = len(tokens)
    x = w["wte"][tokens] + w["wpe"][:T]                     # (T, d)
    mask = np.triu(np.full((T, T), -1e9, np.float32), k=1)  # 因果掩码

    for lw in w["layers"]:
        h = layernorm(x, lw["ln1_g"], lw["ln1_b"])
        q = (h @ lw["wq"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
        k = (h @ lw["wk"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
        v = (h @ lw["wv"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
        att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(cfg.d_head) + mask)
        o = (att @ v).transpose(1, 0, 2).reshape(T, cfg.d_model)
        x = x + o @ lw["wo"]
        h = layernorm(x, lw["ln2_g"], lw["ln2_b"])
        x = x + gelu(h @ lw["w1"]) @ lw["w2"]

    x = layernorm(x, w["lnf_g"], w["lnf_b"])
    return x[-1] @ w["wte"].T                                # 权重共享的 lm_head


# ────────────────────────────────────────────────── KV 缓存前向

class KVCache:
    """最朴素的 KV 缓存：每层一对连续数组，按位置追加。

    这是**第一版**。它有一个致命问题：必须为 max_seq 预分配，
    而每条请求实际长度差别巨大 —— 那正是分页 KV（`paged.py`）要解决的事。
    """

    def __init__(self, cfg: Config, max_seq: int | None = None):
        self.cfg = cfg
        n = max_seq or cfg.max_seq
        self.k = [np.zeros((cfg.n_head, n, cfg.d_head), np.float32) for _ in range(cfg.n_layer)]
        self.v = [np.zeros((cfg.n_head, n, cfg.d_head), np.float32) for _ in range(cfg.n_layer)]
        self.length = 0

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)


def forward_cached(cfg: Config, w: dict, new_tokens: list[int],
                   cache: KVCache) -> np.ndarray:
    """只算新来的 token，历史 K/V 从缓存里读。

    `new_tokens` 可以是一整段 prompt（prefill），也可以是一个 token（decode）——
    **两者走的是同一段代码**。这一点是理解"prefill 和 decode 不是两种算法"的关键。
    """
    T = len(new_tokens)
    p0 = cache.length                                       # 已有多少历史
    x = w["wte"][new_tokens] + w["wpe"][p0:p0 + T]

    # 新 token 之间仍要因果掩码；对历史全部可见，所以历史那段不加掩码
    causal = np.triu(np.full((T, T), -1e9, np.float32), k=1)

    for li, lw in enumerate(w["layers"]):
        h = layernorm(x, lw["ln1_g"], lw["ln1_b"])
        q = (h @ lw["wq"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
        kn = (h @ lw["wk"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
        vn = (h @ lw["wv"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)

        cache.k[li][:, p0:p0 + T] = kn                      # 追加写
        cache.v[li][:, p0:p0 + T] = vn
        k_all = cache.k[li][:, :p0 + T]
        v_all = cache.v[li][:, :p0 + T]

        scores = q @ k_all.transpose(0, 2, 1) / np.sqrt(cfg.d_head)
        scores[:, :, p0:] += causal                         # 只对新 token 段加因果掩码
        o = (softmax(scores) @ v_all).transpose(1, 0, 2).reshape(T, cfg.d_model)
        x = x + o @ lw["wo"]
        h = layernorm(x, lw["ln2_g"], lw["ln2_b"])
        x = x + gelu(h @ lw["w1"]) @ lw["w2"]

    cache.length = p0 + T
    x = layernorm(x, w["lnf_g"], w["lnf_b"])
    return x[-1] @ w["wte"].T


# ────────────────────────────────────────────────── 生成

def generate(cfg: Config, w: dict, prompt: list[int], n_new: int,
             use_cache: bool = True, greedy: bool = True) -> list[int]:
    out = list(prompt)
    if use_cache:
        cache = KVCache(cfg, max_seq=len(prompt) + n_new + 1)
        logits = forward_cached(cfg, w, out, cache)
        for _ in range(n_new):
            nxt = int(logits.argmax()) if greedy else int(np.argmax(logits))
            out.append(nxt)
            logits = forward_cached(cfg, w, [nxt], cache)
    else:
        for _ in range(n_new):
            logits = forward_naive(cfg, w, out)
            out.append(int(logits.argmax()))
    return out


# ────────────────────────────────────────────────── 解析计算（不是实测）

def analytic_cost(cfg: Config, ctx_len: int, batch: int = 1,
                  dtype_bytes: int = 4) -> dict:
    """一步 decode 的**解析**成本：算力 vs 访存。不需要 GPU 也能算清楚。

    ⚠️ 这里有一条我写第一版时算错、被自检打脸的关键结论，务必看懂：

    **batch=1 时，算术强度恒等于 `2/dtype_bytes`（fp32 下就是 0.5 FLOP/字节），
    与上下文长度完全无关。**
    为什么：权重侧每个参数做 1 次乘加（2 FLOP）、读 4 字节 → 0.5；
    KV 侧每个 KV 元素也是做 1 次乘加、读 4 字节 → 还是 0.5。
    两边都是 0.5，加权平均当然还是 0.5。
    我原本以为"上下文越长越访存受限"，其实**它一开始就已经到底了，没有更低可去**。

    真正能抬高算术强度的只有一件事：**增大 batch**。
    权重只读一遍却被 batch 条请求共用，权重侧强度变成 `0.5×batch`；
    而 KV 是每条请求私有的，强度还是 0.5，抬不动。
    → **连续批处理的根本理由就在这条式子里**；
    → 也解释了为什么长上下文场景下 batch 开不大（KV 吃显存），收益跟着塌。
    """
    d, L, H, dh = cfg.d_model, cfg.n_layer, cfg.n_head, cfg.d_head
    # **必须把"矩阵乘用到的权重"和"只被 gather 的 embedding"分开算**
    # （第一版我把 embedding 的字节算进了分母、却没算 lm_head 的 FLOPs，
    #   于是强度算出来是 0.43~0.50 一路上升，看着像"ctx 越长越不受限"，全是口径错）。
    matmul_params = L * (4 * d * d + 8 * d * d) + d * cfg.vocab   # 层权重 + lm_head
    flops_weights = 2 * batch * matmul_params                     # 每参数 1 次乘加 = 2 FLOP
    # 注意力读已有 ctx：每层每头 QK^T 与 AV 各 ctx×dh 次乘加
    flops_attn = batch * L * H * 4 * ctx_len * dh
    # 权重**只读一遍**被整个 batch 共用 —— 这是批处理省下的东西
    # wte/wpe 是 gather，只读用到的那几行，量级可忽略但仍如实计入
    bytes_weights = dtype_bytes * (matmul_params + 2 * batch * d)
    # KV 每条请求私有，随 batch 线性增长 —— 这是批处理省不掉的东西
    bytes_kv = batch * ctx_len * (2 * L * d * dtype_bytes)
    total_flops = flops_weights + flops_attn
    total_bytes = bytes_weights + bytes_kv
    return {
        "ctx_len": ctx_len, "batch": batch,
        "flops_per_step": total_flops,
        "bytes_per_step": total_bytes,
        # 算术强度：每读 1 字节能做多少次浮点运算。**越低越是访存瓶颈。**
        "arithmetic_intensity": round(total_flops / total_bytes, 3),
        "kv_share_of_bytes": round(bytes_kv / total_bytes, 4),
    }


def selftest() -> int:
    ok = True
    cfg = Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=64, seed=1)
    w = init_weights(cfg)
    prompt = [3, 14, 15, 62, 6]

    # 1) **本库最重要的一条正确性命题**：KV 缓存不许改变输出。
    #    朴素前向与缓存前向在同一权重下必须逐位一致（浮点误差在 1e-4 内）。
    a = forward_naive(cfg, w, prompt)
    c = KVCache(cfg)
    b = forward_cached(cfg, w, prompt, c)
    if not np.allclose(a, b, atol=1e-4):
        print(f"FAIL prefill 一致性: 最大差 {np.abs(a - b).max():.3e}"); ok = False

    # 2) 逐 token 喂 vs 一次喂整段，结果也必须一致（增量正确性）
    c2 = KVCache(cfg)
    for t in prompt:
        b2 = forward_cached(cfg, w, [t], c2)
    if not np.allclose(a, b2, atol=1e-4):
        print(f"FAIL 逐token增量一致性: 最大差 {np.abs(a - b2).max():.3e}"); ok = False

    # 3) 整条生成结果必须一致 —— 这是端到端的"缓存不改变语义"
    g1 = generate(cfg, w, prompt, 8, use_cache=False)
    g2 = generate(cfg, w, prompt, 8, use_cache=True)
    if g1 != g2:
        print(f"FAIL 生成不一致:\n  无缓存 {g1}\n  有缓存 {g2}"); ok = False

    # 4) softmax 必须能扛住大数（不减最大值就是 nan）
    big = np.array([[800.0, 801.0, 802.0]], np.float32)
    s = softmax(big)
    if not np.isfinite(s).all() or abs(s.sum() - 1) > 1e-5:
        print(f"FAIL softmax 数值稳定性 -> {s}"); ok = False

    # 5) **batch=1 时算术强度恒为 0.5（fp32），与 ctx 无关。**
    #    这条是我第一版算错的地方：原以为"ctx 越长越访存受限"，
    #    其实 batch=1 时它一开始就已经到底，没有更低可去。
    for ctx in (16, 256, 4096, 65536):
        ai = analytic_cost(cfg, ctx, batch=1)["arithmetic_intensity"]
        if abs(ai - 0.5) > 2e-3:
            print(f"FAIL batch=1 算术强度应恒为 0.5，ctx={ctx} 得到 {ai}"); ok = False

    # 6) **增大 batch 必须抬高算术强度** —— 这是连续批处理的根本理由
    a1 = analytic_cost(cfg, 128, batch=1)["arithmetic_intensity"]
    a64 = analytic_cost(cfg, 128, batch=64)["arithmetic_intensity"]
    if not a64 > a1 * 1.5:
        print(f"FAIL 增大 batch 应显著抬高算术强度 -> {a1} → {a64}"); ok = False

    # 7) **但上下文越长，batch 的收益越被 KV 吃掉** —— 长文本下批处理红利塌陷
    short = analytic_cost(cfg, 16, batch=64)["arithmetic_intensity"]
    long_ = analytic_cost(cfg, 8192, batch=64)["arithmetic_intensity"]
    if not long_ < short:
        print(f"FAIL 长 ctx 下 batch 收益应被 KV 稀释 -> {short} → {long_}"); ok = False

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    cfg = Config()
    w = init_weights(cfg)
    print(cfg, f"参数量≈{cfg.n_params():,}", f"每token KV={cfg.kv_bytes_per_token()}B")
    print("\n一步 decode 的算术强度 = FLOPs / Bytes（越低越访存受限）")
    print("看第一列：**batch=1 恒为 0.50，与 ctx 完全无关** —— 它一开始就已经到底了。")
    print("看每一行往右：**只有增大 batch 能抬高它**，因为权重被摊薄；")
    print("再看行与行之间：**ctx 越长往右抬得越少** —— KV 每条请求私有，摊不薄。\n")
    batches = (1, 4, 16, 64, 256)
    print(f"{'ctx':>8}" + "".join(f"{'bs=' + str(x):>9}" for x in batches)
          + f"{'KV占访存(bs=64)':>18}")
    for ctx in (16, 64, 256, 1024, 4096, 16384):
        row = f"{ctx:>8}"
        for x in batches:
            row += f"{analytic_cost(cfg, ctx, batch=x)['arithmetic_intensity']:>9.2f}"
        row += f"{analytic_cost(cfg, ctx, batch=64)['kv_share_of_bytes'] * 100:>17.1f}%"
        print(row)
