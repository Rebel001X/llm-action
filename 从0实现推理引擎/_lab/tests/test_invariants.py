"""不变量测试 —— 用来补差分测试的**结构性盲区**。

本库前面所有测试的主力判据是**差分**：算两遍，逐位比。
它极其有效，但有一个盲区，而且这个盲区是结构性的、不是写得不够仔细：

    **同时出现在两侧的 bug，差分测试永远抓不到。**

实测（`05-工程与陷阱/01-正确性怎么保证.md` 里有完整复现）：
把 `minigpt.py` 里 4 处 `x = x + ...` 的残差改成 `x = ...`，
两条前向路径**同时**被改到 ——
`test_kv_cache_matches_naive` 照样通过，`test_generation_identical_with_and_without_cache`
也照样通过，而模型输出的 argmax 已经从 36 变成了 44。
**整套自检全绿，模型已经错了。**

补法只有一种：**不比"两个实现"，比"实现和它必须满足的性质"。**
这些性质来自数学，不来自另一份代码，所以两侧同错也躲不过去。
本文件收的就是这类断言 —— 每一条都是**退化情形**或**结构不变量**。

跑法： cd _lab && python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minigpt as M  # noqa: E402
from minigpt import Config  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    cfg = Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=128, seed=1)
    return cfg, M.init_weights(cfg)


# ────────────────────────────────────────── 一、退化情形：把层压成恒等

def test_zeroed_output_projections_reduce_model_to_embedding(tiny):
    """**本文件最重要的一条。**

    把每层的 `wo` 与 `w2` 置零后，两个子层的输出都是 0，
    于是 `x = x + 0` ⇒ **每一层都必须退化成恒等映射**，
    整个模型退化成 `lm_head(LayerNorm(embedding))`。

    这条断言之所以关键：**它是差分测试抓不到的那个盲区的解药。**
    残差写成 `x = ...`（丢掉加号）时，两条前向路径会一起错、差分测试全绿，
    但这里 x 会被清成 0，最大差从 2.4e-07 跳到 3.26 —— 立刻暴露。
    """
    cfg, w0 = tiny
    w = {k: v for k, v in w0.items() if k != "layers"}
    w["layers"] = []
    for lw in w0["layers"]:
        nl = dict(lw)
        nl["wo"] = np.zeros_like(lw["wo"])
        nl["w2"] = np.zeros_like(lw["w2"])
        w["layers"].append(nl)

    prompt = [3, 14, 15, 62, 6, 35]
    got = M.forward_naive(cfg, w, prompt)
    x = w["wte"][prompt] + w["wpe"][:len(prompt)]
    want = M.layernorm(x, w["lnf_g"], w["lnf_b"])[-1] @ w["wte"].T
    assert np.allclose(got, want, atol=1e-4), f"最大差 {np.abs(got - want).max():.3e}"


def test_single_token_attention_returns_the_value_itself(tiny):
    """序列长度为 1 时，softmax 只有一项 ⇒ 恒等于 1 ⇒ 注意力输出恰好是 V 本身。

    **与分数的数值完全无关** —— 所以缩放因子写错、mask 方向反了、
    甚至 QK^T 整个算错，这条都照样成立……**正因如此它才有用**：
    它把"注意力的加权结构"和"分数怎么算"这两件事分开验了。
    """
    s = M.softmax(np.array([[[12345.0]]], np.float32))
    assert np.allclose(s, 1.0)
    v = np.array([[[1.0, 2.0, 3.0]]], np.float32)
    assert np.allclose(s @ v, v)


def test_identical_keys_give_uniform_attention():
    """所有 K 相同 ⇒ 所有分数相同 ⇒ 注意力输出 = V 的**算术平均**。

    这条能抓到"某个位置被漏算/重复算"这类权重分配错误。
    """
    k = np.ones((1, 5, 4), np.float32)
    q = np.full((1, 1, 4), 0.7, np.float32)
    v = np.arange(20, dtype=np.float32).reshape(1, 5, 4)
    p = M.softmax(q @ k.transpose(0, 2, 1) / np.sqrt(4))
    assert np.allclose(p, 1 / 5)
    assert np.allclose(p @ v, v.mean(axis=1, keepdims=True))


# ────────────────────────────────────────── 二、结构不变量

def test_softmax_rows_sum_to_one(tiny):
    for shape in [(3, 7), (1, 1), (4, 128)]:
        x = np.random.default_rng(0).normal(0, 50, shape).astype(np.float32)
        assert np.allclose(M.softmax(x).sum(axis=-1), 1.0, atol=1e-5)


def test_layernorm_normalizes():
    """g=1、b=0 时输出必须零均值单位方差 —— LayerNorm 的定义本身。"""
    x = np.random.default_rng(0).normal(3, 7, (5, 32)).astype(np.float32)
    y = M.layernorm(x, np.ones(32, np.float32), np.zeros(32, np.float32))
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(y.std(axis=-1), 1.0, atol=1e-3)


def test_future_tokens_cannot_change_past_logits(tiny):
    """**因果性**：位置 k 的 logits 不许因为 k 之后的 token 变了而变。

    这是输入空间上的不变量，和"哪个实现"无关 ——
    所以它对"两侧同错"也免疫。掩码方向写反时这一条会立刻挂。
    """
    cfg, w = tiny
    head = [3, 14, 15, 62]

    def stepwise(toks):
        c = M.KVCache(cfg)
        return [M.forward_cached(cfg, w, [t], c).copy() for t in toks]

    a = stepwise(head + [6, 35])
    b = stepwise(head + [40, 41])          # 只改后两个 token
    for i in range(len(head)):
        assert np.allclose(a[i], b[i], atol=1e-6), f"位置 {i} 的 logits 被未来影响了"


def test_permuting_the_prompt_changes_the_output(tiny):
    """**反向断言**：换了 token 顺序结果必须变。

    如果它没变，说明位置编码根本没起作用（或者被覆盖掉了）——
    **这类"少做了一件事"的 bug，正向断言是抓不到的。**
    """
    cfg, w = tiny
    a = M.forward_naive(cfg, w, [3, 14, 15, 62])
    b = M.forward_naive(cfg, w, [62, 15, 14, 3])
    assert not np.allclose(a, b, atol=1e-3)


# ────────────────────────────────────────── 三、解析模型也要验退化情形

def test_analytic_intensity_hits_the_theoretical_floor_exactly(tiny):
    """**先验证退化情形等于理论值，再看趋势。**

    batch=1 时算术强度必须恰好是 `2/dtype字节`，因为权重侧和 KV 侧
    都是"每元素 1 次乘加(2 FLOP)、读 dtype 字节"。
    本库第一版这里算出 0.43 一路升到 0.50，看着像个漂亮的趋势，
    其实是口径错了（embedding 字节进了分母、lm_head 的 FLOPs 没进分子）。
    **任何 roofline 类模型都该先过这一关。**
    """
    cfg, _ = tiny
    for db in (1, 2, 4):
        for ctx in (16, 1024, 65536):
            ai = M.analytic_cost(cfg, ctx, batch=1,
                                 dtype_bytes=db)["arithmetic_intensity"]
            assert abs(ai - 2 / db) < 4e-3 * (2 / db), f"dtype={db} ctx={ctx} -> {ai}"


def test_analytic_bytes_scale_the_way_the_formula_says(tiny):
    """KV 字节必须**同时**对 ctx 和 batch 线性 —— 两个方向都要验。

    只验一个方向时，把 batch 写到错误的位置也能蒙混过关。
    """
    cfg, _ = tiny
    base = M.analytic_cost(cfg, 1024, batch=1)
    kv = lambda r: r["bytes_per_step"] * r["kv_share_of_bytes"]
    assert abs(kv(M.analytic_cost(cfg, 2048, batch=1)) / kv(base) - 2) < 0.02
    assert abs(kv(M.analytic_cost(cfg, 1024, batch=8)) / kv(base) - 8) < 0.02
