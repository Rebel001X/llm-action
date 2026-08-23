"""进阶主题四件套的回归测试：FlashAttention / 并行 / 投机解码 / 结构化输出 / 量化。

判据在这里又换了一套 —— 这是本库第三种测试哲学，值得先说清：

| 模块 | 判据 | 为什么 |
|---|---|---|
| `flashattn` | **输出恒等** | 分块只是重排，不是近似 |
| `parallel` | **输出恒等** | 切权重也只是重排 |
| `spec_decode` | **分布恒等**（统计检验） | 单次输出本来就是随机的，只能测分布 |
| `structured` | **约束恒成立** | 它**故意**改变分布，恒等测试没有意义 |
| `quantize` | **误差有界 + 单调** | 它是有损的，只能测"损得可控" |

**把这四种判据用错地方，是这个领域最常见的测试错误。**
比如拿"输出恒等"去测量化（永远挂），或者拿"跑起来没报错"去测投机解码
（**它错了也照样跑，只是分布悄悄偏了**）。

跑法： cd _lab && python -m pytest tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashattn as F  # noqa: E402
import parallel as P  # noqa: E402
import quantize as Q  # noqa: E402
import spec_decode as S  # noqa: E402
import structured as ST  # noqa: E402


def test_all_advanced_selftests_pass():
    assert F.selftest() == 0
    assert P.selftest() == 0
    assert S.selftest() == 0
    assert ST.selftest() == 0
    assert Q.selftest() == 0


# ────────────────────────────────────────── 一、FlashAttention：输出恒等

@pytest.fixture(scope="module")
def qkv():
    rng = np.random.default_rng(0)
    t, d = 48, 16
    return tuple(rng.normal(0, 1, (t, d)).astype(np.float32) for _ in range(3))


@pytest.mark.parametrize("bq,bk", [(1, 1), (1, 8), (7, 5), (16, 16), (64, 64)])
def test_tiling_does_not_change_output(qkv, bq, bk):
    """**分块是重写，不是近似。** 块大小是纯实现细节，不许影响结果。"""
    q, k, v = qkv
    ref, _ = F.attention_naive(q, k, v, causal=True)
    got, _ = F.attention_online(q, k, v, causal=True, bq=bq, bk=bk)
    assert np.allclose(ref, got, atol=1e-5)


def test_peak_intermediate_is_independent_of_sequence_length(qkv):
    """朴素峰值是 T²，分块峰值是块面积 —— **这才是 FlashAttention 省下的东西**。"""
    q, k, v = qkv
    _, naive = F.attention_naive(q, k, v)
    _, tiled = F.attention_online(q, k, v, bq=8, bk=8)
    assert naive == q.shape[0] * k.shape[0]
    assert tiled == 64


def test_online_softmax_survives_huge_inputs():
    """缩放因子 exp(m_old - m_new) 恒 <= 1，所以永远向下缩，不会溢出。"""
    x = np.array([700.0, 800.0, 900.0] * 50, np.float32)
    m, l = F.online_softmax_sum(x, block=13)
    assert np.isfinite(m) and np.isfinite(l) and l > 0


def test_split_kv_merge_is_associative(qkv):
    """decode 时 Tq=1，只能沿 KV 切；合并可结合，所以切几段都一样。"""
    _, k, v = qkv
    q1 = np.random.default_rng(3).normal(0, 1, (1, k.shape[1])).astype(np.float32)
    ref, _ = F.attention_naive(q1, k, v)
    for n in (1, 2, 5, 16, 48):
        got, _ = F.attention_split_kv(q1, k, v, n_split=n)
        assert np.allclose(ref, got, atol=1e-5)


# ────────────────────────────────────────── 二、并行：输出恒等

@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_tensor_parallel_mlp_is_exact(tp):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (8, 32)).astype(np.float32)
    w1 = rng.normal(0, 0.05, (32, 128)).astype(np.float32)
    w2 = rng.normal(0, 0.05, (128, 32)).astype(np.float32)
    got, _ = P.mlp_tensor_parallel(x, w1, w2, tp)
    assert np.allclose(P.mlp_single(x, w1, w2), got, atol=1e-4)


def test_gelu_is_not_additive():
    """`w1` 必须**列**切的全部理由：GELU 非线性，不能作用在部分和上。"""
    rng = np.random.default_rng(2)
    a, b = (rng.normal(0, 1, (4, 4)).astype(np.float32) for _ in range(2))
    assert not np.allclose(P.gelu(a + b), P.gelu(a) + P.gelu(b), atol=1e-3)


def test_head_count_must_divide_tp():
    """**硬约束，不是实现限制。** 所以 TP 度只能取能整除头数的值。"""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (4, 32)).astype(np.float32)
    ws = [rng.normal(0, 0.05, (32, 32)).astype(np.float32) for _ in range(4)]
    with pytest.raises(AssertionError):
        P.attn_tensor_parallel(x, *ws, n_head=8, tp=3)


def test_allreduce_volume_grows_with_tp():
    """系数 2(tp-1)/tp 单调上升 —— **卡越多，每卡通信量不降反升**。"""
    vols = [P.allreduce_elements(1024, 4096, t) for t in (2, 4, 8, 16)]
    assert vols == sorted(vols) and vols[0] < vols[-1]


@pytest.mark.parametrize("p,m", [(2, 1), (4, 4), (4, 16), (8, 64), (16, 8)])
def test_pipeline_bubble_matches_formula(p, m):
    _, bub = P.pipeline_schedule(p, m)
    assert abs(bub - (p - 1) / (m + p - 1)) < 1e-9


def test_moe_topk_does_not_lose_tokens():
    for k in (1, 2, 4):
        assert int(P.moe_route(1000, 8, top_k=k, seed=0).sum()) == 1000 * k


# ────────────────────────────────────────── 三、投机解码：分布恒等

@pytest.fixture(scope="module")
def pq():
    rng = np.random.default_rng(0)
    return S._rand_dist(rng, 8, 1.5), S._rand_dist(rng, 8, 1.5)


def test_acceptance_identity_is_exact(pq):
    """min(q,p) + Z·norm(max(0,p-q)) == p。**这是恒等式，与 q 无关。**"""
    p, q = pq
    z = float(np.maximum(p - q, 0).sum())
    assert np.allclose(np.minimum(p, q) + z * S.residual(p, q), p, atol=1e-12)


def test_output_distribution_equals_target(pq):
    """**投机解码不是近似。** 8 万次采样，全变差应当落在采样噪声量级。"""
    p, q = pq
    rng = np.random.default_rng(7)
    emp = np.bincount([S.spec_sample_one(p, q, rng) for _ in range(80_000)],
                      minlength=len(p)) / 80_000
    assert S._tv(emp, p) < 0.01


def test_naive_verifier_is_biased(pq):
    """看起来合理的朴素验收版**跑得很正常，只是分布错了** —— 这类 bug 最难发现。"""
    p, q = pq
    rng = np.random.default_rng(7)
    emp = np.bincount([S.spec_sample_one_naive(p, q, rng) for _ in range(80_000)],
                      minlength=len(p)) / 80_000
    assert S._tv(emp, p) > 0.05


def test_always_produces_at_least_one_token(pq):
    """一个都没猜中也要产出 1 个 —— 所以投机解码在步数上不会更差。"""
    p, _ = pq
    rng = np.random.default_rng(1)
    bad = np.stack([S._rand_dist(rng, len(p), 8.0) for _ in range(4)])
    for _ in range(200):
        out, _ = S.spec_step(np.stack([p] * 5), bad, rng)
        assert len(out) >= 1


def test_expected_tokens_endpoints():
    assert S.expected_tokens(1.0, 4) == 5.0
    assert abs(S.expected_tokens(1e-12, 4) - 1.0) < 1e-6


def test_verification_is_nearly_free_in_bytes():
    """γ+1 个 token **同属一条请求、共用一份 KV** —— 这才是"验证白送"的机制。"""
    a = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000, ctx=4096)
    c1, c8 = S.verify_cost(**a, n_tok=1), S.verify_cost(**a, n_tok=8)
    assert c8["flops"] / c1["flops"] > 7.9
    assert c8["bytes"] / c1["bytes"] < 1.05


# ────────────────────────────────────────── 四、结构化输出：约束恒成立

@pytest.fixture(scope="module")
def index():
    return ST.build_index()


def _noise(st, tx):
    import zlib
    return np.random.default_rng(zlib.crc32(f"{st}|{tx}".encode())).normal(
        0, 3, len(ST.VOCAB))


@pytest.mark.parametrize("seed", range(20))
def test_constrained_output_is_always_parseable(index, seed):
    """**合法性来自掩码，不来自模型。** 模型给的是纯噪声，结果照样合法。"""
    o = ST.generate(_noise, seed=seed, index=index)
    assert o["state"][0] == "done"
    json.loads(o["text"])          # 解析失败会抛，就是断言


def test_token_legality_is_state_dependent():
    """同一个 token 换个状态就不合法 —— **所以不能按 token 写规则**。"""
    assert ST.step_token(("kb", 1, 0), '":') == ("val", 0, 0)
    assert ST.step_token(("kq", 0, 0), '":') is None


def test_jump_forward_changes_cost_not_output(index):
    """跳过的都是本来就没得选的步，所以**产出必须完全一样**，只是少问了几次模型。"""
    for seed in range(10):
        a = ST.generate(_noise, seed=seed, jump_forward=True, index=index)
        b = ST.generate(_noise, seed=seed, jump_forward=False, index=index)
        assert a["text"] == b["text"]
        assert a["model_calls"] + a["jumped"] == b["model_calls"]
        assert a["model_calls"] < b["model_calls"]


def test_empty_allowed_set_raises_loudly():
    """静默的话 softmax 变 nan、argmax 取 0，模型开始吐垃圾而毫无报错。"""
    with pytest.raises(ValueError):
        ST.apply_mask(np.zeros(len(ST.VOCAB)), np.zeros(len(ST.VOCAB), bool))


def test_mask_renormalizes_onto_allowed_set(index):
    """结构化输出**故意**改变分布 —— 与投机解码的等价变换不是一回事。"""
    st = ("val", 0, 0)
    lg = ST.apply_mask(np.zeros(len(ST.VOCAB)), index[st])
    e = np.exp(lg - lg.max())
    probs = e / e.sum()
    assert probs[~index[st]].sum() == 0.0
    assert abs(probs[index[st]].sum() - 1.0) < 1e-12


# ────────────────────────────────────────── 五、量化：误差有界且单调

@pytest.fixture(scope="module")
def wmat():
    return np.random.default_rng(0).normal(0, 0.05, (256, 256)).astype(np.float32)


def test_error_matches_analytic_prediction(wmat):
    """量化噪声近似均匀分布，误差量级**可以先算出来**，不用拍脑袋定阈值。"""
    got = Q.rel_err(wmat, Q.qdq_per_tensor(wmat, 8))
    pred = (np.abs(wmat).max() / 127 / np.sqrt(12)) / float(wmat.std())
    assert abs(got - pred) / pred < 0.15


def test_one_outlier_destroys_per_tensor(wmat):
    """**一个数毁掉整个矩阵** —— per-channel 的存在理由。"""
    clean = Q.rel_err(wmat, Q.qdq_per_tensor(wmat, 8))
    w = wmat.copy()
    w[0, 0] = 50.0
    dirty = Q.rel_err(w, Q.qdq_per_tensor(w, 8))
    assert dirty > 20 * clean
    assert Q.rel_err(w, Q.qdq_per_channel(w, 8)) < dirty / 5


def test_quantization_is_idempotent(wmat):
    q = Q.qdq_per_channel(wmat, 8)
    assert np.allclose(q, Q.qdq_per_channel(q, 8), atol=1e-7)


def test_error_is_monotonic_in_bitwidth(wmat):
    errs = [Q.rel_err(wmat, Q.qdq_per_channel(wmat, b)) for b in (4, 6, 8, 12)]
    assert errs == sorted(errs, reverse=True)


def test_int4_is_not_actually_four_bits():
    """group=32 时实际 4.5 bit。**宣传口径和真实口径差在 scale 上。**"""
    assert abs(Q.bytes_per_weight(4, 32) * 8 - 4.5) < 1e-9
    assert Q.bytes_per_weight(4, 32) > Q.bytes_per_weight(4, 128)


def test_weight_quant_changes_bytes_not_flops():
    """**它是访存优化，不是算力优化。** FLOPs 一位都没动。"""
    a = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000,
             ctx=256, batch=1)
    base = Q.intensity_with_quant(**a)
    q = Q.intensity_with_quant(**a, w_bytes=1.0)
    assert q["flops"] == base["flops"]
    assert q["bytes"] < base["bytes"] * 0.3


def test_kv_dominance_needs_long_context_and_large_batch():
    """**我第一版写窄了**：以为长上下文就够，实测 batch=1 时权重仍是主项。"""
    a = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000)
    assert Q.intensity_with_quant(**a, ctx=16384, batch=1)["kv_share"] < 0.5
    assert Q.intensity_with_quant(**a, ctx=16384, batch=64)["kv_share"] > 0.9


def test_weight_quant_benefit_shrinks_with_batch():
    """批大了权重被摊薄，weight-only 量化的收益跟着塌。"""
    a = dict(d_model=4096, n_layer=32, n_head=32, d_head=128, vocab=32000, ctx=256)
    def gain(bs):
        b = Q.intensity_with_quant(**a, batch=bs)
        return 1 - Q.intensity_with_quant(**a, batch=bs, w_bytes=1.0)["bytes"] / b["bytes"]
    assert gain(1) > gain(64) > gain(256)
