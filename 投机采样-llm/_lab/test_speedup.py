"""物理账：roofline、batch 崩塌、保本线。第 03、18、19、22 篇引用本文件。"""
import pytest

from speedup import (H100, MODELS, breakeven_accept_len, crossover_batch,
                     feasible, fwd_time, scale, spec_throughput, tokens_to_saturate)


# ---------------------------------------------------------------- roofline 基本性质

def test_ridge_point_value():
    """H100 的屋脊点 = 989.5e12 / 3.35e12 ≈ 295 FLOP/Byte。第 03 篇引用这个数。"""
    ridge = H100["peak"] / H100["bw"]
    assert 290 < ridge < 300


def test_decode_is_memory_bound_at_batch_one():
    r = fwd_time(MODELS["llama3-70b"], scale(H100, 4), batch=1, seqlen=1024, q_per_seq=1)
    assert r["bound"] == "memory"
    assert r["intensity"] < 5.0          # 算术强度远低于屋脊点 295


def test_verify_is_nearly_free_in_memory_bound_region():
    """**投机采样成立的全部理由**：memory 区里，验证 5 个 token 与 1 个几乎同价。"""
    hw = scale(H100, 4)
    t1 = fwd_time(MODELS["llama3-70b"], hw, 1, 1024, 1)
    t5 = fwd_time(MODELS["llama3-70b"], hw, 1, 1024, 5)
    assert t1["bound"] == t5["bound"] == "memory"
    assert t5["t"] / t1["t"] < 1.01


def test_verify_stops_being_free_in_compute_bound_region():
    """**投机采样崩塌的全部理由**：compute 区里，验证时间随 query token 数线性涨。"""
    hw = scale(H100, 8)
    t1 = fwd_time(MODELS["llama3-70b"], hw, 512, 1024, 1)
    t5 = fwd_time(MODELS["llama3-70b"], hw, 512, 1024, 5)
    assert t5["bound"] == "compute"
    assert t5["t"] / t1["t"] > 2.0


def test_weights_read_once_regardless_of_query_tokens():
    """访存字节与 q_per_seq 无关 —— 权重只搬一次。"""
    hw = scale(H100, 4)
    a = fwd_time(MODELS["llama3-8b"], hw, 8, 2048, 1)
    b = fwd_time(MODELS["llama3-8b"], hw, 8, 2048, 9)
    assert abs(a["t_mem"] - b["t_mem"]) < 1e-15


# ---------------------------------------------------------------- batch 崩塌

def test_speedup_at_batch_one_is_large():
    hw = scale(H100, 8)
    r = spec_throughput("llama3-70b", "llama3.2-1b", hw, 1, 1024, 4, 3.0)
    assert r["speedup"] > 2.5


def test_speedup_collapses_at_large_batch_short_context():
    """短上下文 + 大 batch：同一个 E[tau]=3.0，加速比翻转成 <1。第 19 篇引用。"""
    hw = scale(H100, 8)
    small = spec_throughput("llama3-70b", "llama3.2-1b", hw, 1, 1024, 4, 3.0)
    large = spec_throughput("llama3-70b", "llama3.2-1b", hw, 512, 1024, 4, 3.0)
    assert small["speedup"] > 2.5
    assert large["speedup"] < 1.0
    assert large["verify_bound"] == "compute"


def test_speedup_is_monotone_decreasing_in_batch():
    hw = scale(H100, 8)
    ss = [spec_throughput("llama3-70b", "llama3.2-1b", hw, b, 1024, 4, 3.0)["speedup"]
          for b in (1, 16, 64, 128, 256, 384, 512)]
    assert all(b <= a + 1e-9 for a, b in zip(ss, ss[1:])), ss


def test_long_context_keeps_speedup_at_large_batch():
    """长上下文下**不翻转** —— KV 读把前向钉死在 memory 区。第 22 篇引用。

    所以"投机解码在大 batch 下没用"这句流行说法的隐含前提是短上下文。
    """
    hw = scale(H100, 8)
    r = spec_throughput("llama3-70b", "llama3.2-1b", hw, 512, 16384, 4, 3.0)
    assert r["verify_bound"] == "memory"
    assert r["speedup"] > 2.0


def test_long_context_large_batch_may_not_fit():
    """但长上下文 + 大 batch 常常根本装不下 —— 另一个维度的约束。"""
    hw = scale(H100, 8)
    ok, gb = feasible(MODELS["llama3-70b"], hw, 512, 16384,
                      draft=MODELS["llama3.2-1b"])
    assert not ok and gb > hw["hbm"] / 1e9


def test_70b_does_not_fit_on_one_gpu():
    """基本体检：70B fp16 权重 141 GB > 单卡 80 GB。表里不许出现这种配置。"""
    ok, gb = feasible(MODELS["llama3-70b"], H100, 1, 128)
    assert not ok and gb > 140


# ---------------------------------------------------------------- 保本线

def test_breakeven_always_above_one():
    """草稿总要花时间，所以保本所需的 E[tau] 恒 >1：E[tau]=1 时投机必然是纯亏。"""
    hw = scale(H100, 4)
    for batch in (1, 8, 32, 64):
        for seqlen in (1024, 8192):
            assert breakeven_accept_len("llama3-70b", "llama3.2-1b",
                                        hw, batch, seqlen, 4) > 1.0


def test_lighter_draft_lowers_breakeven():
    """草稿越轻，保本线越低 —— 这就是从"小模型草稿"转向"单层草稿头"的动力。"""
    hw = scale(H100, 4)
    heavy = breakeven_accept_len("llama3-70b", "llama3.2-1b", hw, 8, 1024, 4)
    light = breakeven_accept_len("llama3-70b", "eagle-head", hw, 8, 1024, 4)
    assert light < heavy


def test_breakeven_can_exceed_theoretical_max():
    """大 batch 短上下文下，保本线会超过 gamma+1 的理论上限 -> 无论如何都不可能赚。"""
    hw = scale(H100, 8)
    be = breakeven_accept_len("llama3-70b", "llama3.2-1b", hw, 1024, 1024, 4)
    assert be > 5.0


@pytest.mark.parametrize("model", ["llama3-8b", "llama3-70b", "llama3.2-1b"])
def test_model_specs_self_consistent(model):
    m = MODELS[model]
    assert m["P"] > 0 and m["kv_per_tok"] > 0 and m["d_model"] > 0
    # KV/token = 2(K,V) * layers * kv_dim * 2字节，必是 4*layers 的整数倍
    assert m["kv_per_tok"] % (4 * m["layers"]) == 0


def test_compute_bound_asymptote_equals_wasted_compute_ratio():
    """**compute-bound 极限下，加速比收敛到 E[tau]/(gamma+1)，恒小于 1。**

    机制：进入 compute 区后 t_verify 正比于 batch*(gamma+1)，于是
    spec_tps = batch*E[tau]/t_iter 与 batch 无关（封顶），而基线也封顶但更高。
    比值的极限就是"有效产出 / 消耗的算力份额" = E[tau]/(gamma+1)，
    再打一个草稿开销的折扣。这说明大 batch 下的翻转**不是调参能救的**。
    """
    hw = scale(H100, 8)
    gamma, etau = 4, 3.0
    r = spec_throughput("llama3-70b", "llama3.2-1b", hw, 1024, 1024, gamma, etau)
    ideal = etau / (gamma + 1)                       # 0.6
    assert r["verify_bound"] == "compute"
    # 实测加速比 ≈ 理想上限 × (1 - 草稿占迭代的比例)
    assert abs(r["speedup"] - ideal * (1 - r["t_draft_share"])) < 0.01, r
    assert r["speedup"] < 1.0


def test_spec_throughput_saturates_while_baseline_keeps_climbing():
    """compute 区里投机吞吐几乎封顶，基线却继续涨 —— 所以交叉点必然出现。"""
    hw = scale(H100, 8)
    specs, bases = [], []
    for b in (128, 256, 512, 1024):
        r = spec_throughput("llama3-70b", "llama3.2-1b", hw, b, 1024, 4, 3.0)
        specs.append(r["spec_tps"])
        bases.append(r["base_tps"])
    assert specs[-1] / specs[0] < 1.10, specs      # 投机吞吐 8 倍 batch 只涨不到 10%
    assert bases[-1] / bases[0] > 2.5, bases       # 基线吞吐仍在大幅增长


def test_higher_accept_length_raises_but_cannot_save_asymptote():
    """把 E[tau] 顶到理论上限 gamma+1，compute 极限下也只能打平，不能盈利。"""
    hw = scale(H100, 8)
    gamma = 4
    r = spec_throughput("llama3-70b", "llama3.2-1b", hw, 1024, 1024, gamma, float(gamma + 1))
    assert r["speedup"] < 1.0          # 草稿开销吃掉了最后那点余量
    assert r["speedup"] > 0.9          # 但已经很接近打平


# ---------------------------------------------------------------- 量化 x 投机

def test_weight_quantization_shrinks_the_usable_batch_range():
    """**量化与投机采样是竞争关系，不是叠加关系。**

    权重量化把访存项砍小 -> 前向更早离开 memory-bound 区 -> "验证几乎免费"更早失效
    -> 投机可用的 batch 区间缩小。实测交叉点：fp16 在 273，fp8 在 137，int4 在 69。
    """
    hw = scale(H100, 8)
    cb = [crossover_batch("llama3-70b", "llama3.2-1b", hw, 1024, 4, 3.0, wb)
          for wb in (2.0, 1.0, 0.5)]
    assert all(x is not None for x in cb), cb
    assert cb[0] > cb[1] > cb[2], cb          # 精度越低，交叉点越靠左
    assert cb[0] / cb[2] > 3.0, cb            # 缩小 3 倍以上


def test_kv_quantization_also_shrinks_the_range():
    """KV 量化砍的是 batch*seqlen 那一项，同样把前向往 compute 区推。"""
    hw = scale(H100, 8)
    full = crossover_batch("llama3-70b", "llama3.2-1b", hw, 1024, 4, 3.0, 2.0, 1.0)
    half = crossover_batch("llama3-70b", "llama3.2-1b", hw, 1024, 4, 3.0, 2.0, 0.5)
    assert half < full, (full, half)


def test_quantization_speeds_up_the_baseline_too():
    """但量化让**基线本身**变快 —— 所以该比的是绝对吞吐，不是加速比。"""
    hw = scale(H100, 8)
    base = [spec_throughput("llama3-70b", "llama3.2-1b", hw, 1, 1024, 4, 3.0, wb)["base_tps"]
            for wb in (2.0, 1.0, 0.5)]
    assert base[0] < base[1] < base[2], base
    assert base[2] / base[0] > 3.5, base


def test_long_context_still_no_crossover_even_quantized():
    """长上下文下即使量化到 int4 + KV fp8，本模型扫到 8192 仍未见交叉点。"""
    hw = scale(H100, 8)
    assert crossover_batch("llama3-70b", "llama3.2-1b", hw, 16384, 4, 3.0, 0.5, 0.5) is None


def test_free_budget_is_a_few_hundred_query_tokens():
    """一批在翻进 compute-bound 前的"免费额度"是几百个 query token 量级。

    这个量级决定了 chunked prefill 与投机采样必然抢额度：
    主流引擎的 prefill chunk 通常是 512~2048，一个 chunk 就够把它吃光。
    """
    hw = scale(H100, 8)
    cap = tokens_to_saturate(MODELS["llama3-70b"], hw, 1, 1024)
    assert 200 < cap < 400, cap


def test_spec_alone_can_exhaust_the_free_budget():
    """batch 够大时，**投机解码自己就把免费额度透支了**，一个 prefill token 都塞不下。"""
    hw = scale(H100, 8)
    gamma = 4
    cap = tokens_to_saturate(MODELS["llama3-70b"], hw, 128, 1024)
    assert 128 * (gamma + 1) > cap, (cap, 128 * (gamma + 1))
    # batch 小时则还有富余
    cap8 = tokens_to_saturate(MODELS["llama3-70b"], hw, 8, 1024)
    assert 8 * (gamma + 1) < cap8, (cap8, 8 * (gamma + 1))


def test_free_budget_grows_with_context_length():
    """长上下文把免费额度撑大（KV 读变多）—— 这是长上下文对投机友好的量化说法。"""
    hw = scale(H100, 8)
    caps = [tokens_to_saturate(MODELS["llama3-70b"], hw, 32, s)
            for s in (1024, 8192, 32768)]
    assert caps[0] < caps[1] < caps[2], caps


def test_free_budget_shrinks_with_context_at_batch_one():
    """**长上下文撑大免费额度这条只在 batch 够大时成立。**

    batch=1 时 KV 访存项只有 1*seqlen*kv，而 attention 算力项按 L*seqlen 涨，
    分母涨得比分子快 -> 额度反而变小。这个反转是修正 attention FLOPs 漏乘层数
    之后才显出来的（漏乘时算力项被低估，看不出反转）。
    """
    hw = scale(H100, 8)
    caps = [tokens_to_saturate(MODELS["llama3-70b"], hw, 1, s) for s in (1024, 8192)]
    assert caps[0] > caps[1], caps


def test_attention_flops_include_layer_count():
    """回归测试：attention 浮点数必须乘层数 L。

    历史 bug：早期写成 4*seqlen*d_model（漏了 L），s=1024 时几乎无害，
    s=32768 时该项被低估到只有真值的 1/80。这条断言把它钉住。
    """
    m = MODELS["llama3-70b"]
    hw = scale(H100, 8)
    s_long = 32768
    r = fwd_time(m, hw, 1, s_long, 1)
    attn = 4 * m["layers"] * s_long * m["d_model"]
    expect = (2 * m["P"] + attn) / hw["peak"]
    assert abs(r["t_cmp"] - expect) < 1e-15, (r["t_cmp"], expect)
    assert attn / (2 * m["P"]) > 0.5, "长上下文下 attention 项应当已经可观"
