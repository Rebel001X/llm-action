"""_lab 的回归测试。

本教学库的测试哲学与性能库不同，只有一条：
**任何优化都不许改变输出。**

KV 缓存、分页、连续批处理、前缀缓存 —— 这四样都是"应该完全等价"的变换。
它们一旦出错，**几乎不会崩，只会静默给出不一样的结果**。
所以这里绝大多数测试是「优化前 vs 优化后逐位对比」，而不是测速度。

跑法： cd _lab && python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine as E  # noqa: E402
import minigpt as M  # noqa: E402
import paged as P  # noqa: E402
from minigpt import Config  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    cfg = Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=128, seed=1)
    return cfg, M.init_weights(cfg)


# ────────────────────────────────────────── 各模块自检必须全绿

def test_all_selftests_pass():
    assert M.selftest() == 0
    assert P.selftest() == 0
    assert E.selftest() == 0


# ────────────────────────────────────────── 一、KV 缓存不改变输出

def test_kv_cache_matches_naive(tiny):
    cfg, w = tiny
    prompt = [3, 14, 15, 62, 6, 35]
    a = M.forward_naive(cfg, w, prompt)
    b = M.forward_cached(cfg, w, prompt, M.KVCache(cfg))
    assert np.allclose(a, b, atol=1e-4), f"最大差 {np.abs(a - b).max():.3e}"


def test_incremental_feeding_matches_bulk(tiny):
    """逐 token 喂 == 一次喂整段。位置编码偏移写错就会在这里露馅。"""
    cfg, w = tiny
    prompt = [3, 14, 15, 62, 6, 35]
    want = M.forward_naive(cfg, w, prompt)
    c = M.KVCache(cfg)
    for t in prompt:
        got = M.forward_cached(cfg, w, [t], c)
    assert np.allclose(want, got, atol=1e-4)


def test_generation_identical_with_and_without_cache(tiny):
    cfg, w = tiny
    p = [3, 14, 15, 62]
    assert M.generate(cfg, w, p, 8, use_cache=False) == \
           M.generate(cfg, w, p, 8, use_cache=True)


# ────────────────────────────────────────── 二、数值稳定性

def test_softmax_survives_large_inputs():
    """不减最大值就是 nan。这不是理论风险。"""
    s = M.softmax(np.array([[800.0, 801.0, 802.0]], np.float32))
    assert np.isfinite(s).all()
    assert abs(s.sum() - 1.0) < 1e-5


def test_softmax_shift_invariance():
    """softmax(x) == softmax(x - c) —— 减最大值之所以合法的全部理由。"""
    x = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert np.allclose(M.softmax(x), M.softmax(x - 12345.0), atol=1e-6)


# ────────────────────────────────────────── 三、算术强度模型

def test_batch1_intensity_is_constant_half(tiny):
    """**fp32、batch=1 时算术强度恒为 0.5，与上下文无关。**

    这是本库最重要的一条解析结论，也是我第一版算错的地方
    （embedding 字节进了分母、lm_head FLOPs 没进分子）。
    """
    cfg, _ = tiny
    for ctx in (16, 256, 4096, 65536):
        ai = M.analytic_cost(cfg, ctx, batch=1)["arithmetic_intensity"]
        assert abs(ai - 0.5) < 2e-3, f"ctx={ctx} 得到 {ai}"


def test_fp16_doubles_the_floor(tiny):
    """强度下界是 `2/dtype字节`，所以 fp16 是 1.0。"""
    cfg, _ = tiny
    ai = M.analytic_cost(cfg, 1024, batch=1, dtype_bytes=2)["arithmetic_intensity"]
    assert abs(ai - 1.0) < 5e-3


def test_batching_raises_intensity(tiny):
    """只有增大 batch 能抬高强度 —— 连续批处理的根本理由。"""
    cfg, _ = tiny
    a1 = M.analytic_cost(cfg, 128, batch=1)["arithmetic_intensity"]
    a64 = M.analytic_cost(cfg, 128, batch=64)["arithmetic_intensity"]
    assert a64 > a1 * 1.5


def test_long_context_dilutes_batching_benefit(tiny):
    """ctx 越长，batch 的收益越被 KV 吃掉 —— 长上下文杀吞吐的机制。"""
    cfg, _ = tiny
    short = M.analytic_cost(cfg, 16, batch=64)["arithmetic_intensity"]
    long_ = M.analytic_cost(cfg, 8192, batch=64)["arithmetic_intensity"]
    assert long_ < short
    assert M.analytic_cost(cfg, 8192, batch=64)["kv_share_of_bytes"] > 0.9


# ────────────────────────────────────────── 四、分页不改变输出

@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
def test_paging_is_output_preserving(tiny, block_size):
    """**块大小是纯粹的实现细节，不许影响一个 bit。**"""
    cfg, w = tiny
    prompt = [3, 14, 15, 62, 6, 35, 8, 9, 7, 12]
    want = M.forward_naive(cfg, w, prompt)
    tab = P.SeqTable(P.PagedKVCache(cfg, n_blocks=128, block_size=block_size))
    got = P._forward_paged(cfg, w, prompt, tab)
    assert np.allclose(want, got, atol=1e-4)


def test_internal_fragmentation_below_one_block(tiny):
    """分页的核心卖点：内碎片被压到不足一个块。"""
    cfg, w = tiny
    pool = P.PagedKVCache(cfg, n_blocks=128, block_size=8)
    tab = P.SeqTable(pool)
    P._forward_paged(cfg, w, list(range(20)), tab)
    assert tab.fragmentation()["internal_waste"] < pool.block_size


def test_blocks_are_returned_on_free(tiny):
    cfg, w = tiny
    pool = P.PagedKVCache(cfg, n_blocks=32, block_size=4)
    tab = P.SeqTable(pool)
    P._forward_paged(cfg, w, list(range(12)), tab)
    assert pool.alloc.n_used > 0
    tab.free()
    assert pool.alloc.n_used == 0


def test_out_of_blocks_raises_loudly(tiny):
    """块耗尽必须**明确报错**。静默乱写是推理引擎最危险的失败模式。"""
    cfg, w = tiny
    tab = P.SeqTable(P.PagedKVCache(cfg, n_blocks=1, block_size=2))
    with pytest.raises(MemoryError):
        P._forward_paged(cfg, w, list(range(20)), tab)


def test_refcount_guards_shared_blocks(tiny):
    """块共享成立的前提：引用计数归零才归还。"""
    cfg, _ = tiny
    pool = P.PagedKVCache(cfg, n_blocks=8, block_size=4)
    b = pool.alloc.alloc()
    pool.alloc.incref(b)
    pool.alloc.decref(b)
    assert pool.alloc.n_used == 1
    pool.alloc.decref(b)
    assert pool.alloc.n_used == 0


# ────────────────────────────────────────── 五、批处理与前缀缓存不改变输出

def _reference(cfg, w, prompt, n):
    out, toks = [], list(prompt)
    for _ in range(n):
        nxt = int(M.forward_naive(cfg, w, toks).argmax())
        out.append(nxt)
        toks.append(nxt)
    return out


@pytest.mark.parametrize("prefix_cache", [False, True])
def test_batching_preserves_per_request_output(tiny, prefix_cache):
    """**多条请求同批跑，每条的输出必须和它单独跑时一模一样。**"""
    cfg, w = tiny
    shared = list(range(7, 23))
    p1, p2 = shared + [30, 31], shared + [40, 41]
    want1, want2 = _reference(cfg, w, p1, 5), _reference(cfg, w, p2, 5)
    e = E.Engine(cfg, w, n_blocks=256, block_size=4, max_running=4,
                 enable_prefix_cache=prefix_cache)
    r1, r2 = E.Request(1, p1, 5), E.Request(2, p2, 5)
    e.add(r1); e.add(r2)
    e.run()
    assert r1.out == want1
    assert r2.out == want2


def test_prefix_cache_actually_saves_prefill(tiny):
    """前缀缓存唯一的收益来源：少算 prefill token。省不下来就等于没开。"""
    cfg, w = tiny
    shared = list(range(7, 23))

    def prefill(pc):
        e = E.Engine(cfg, w, n_blocks=256, block_size=4, max_running=4,
                     enable_prefix_cache=pc)
        e.add(E.Request(1, shared + [30, 31], 4))
        e.add(E.Request(2, shared + [40, 41], 4))
        return e.run()["prefill_tokens"]

    assert prefill(True) < prefill(False)


def test_chained_hash_prevents_key_collision(tiny):
    """**不做链式哈希就会静默给出错误答案。**

    "1,2|5,6" 与 "3,4|5,6" 的第二块内容相同但历史不同，K/V 完全不同。
    只 hash 本块 token 会让它们撞成同一个 key。
    """
    cfg, _ = tiny
    pc = E.PrefixCache(P.PagedKVCache(cfg, n_blocks=32, block_size=2))
    assert pc.block_hashes([1, 2, 5, 6])[1] != pc.block_hashes([3, 4, 5, 6])[1]


def test_short_request_finishes_before_long_one(tiny):
    """连续批处理的意义：短请求不必等长请求。"""
    cfg, w = tiny
    e = E.Engine(cfg, w, n_blocks=512, block_size=4, max_running=4)
    short, long_ = E.Request(1, [1, 2, 3, 4], 2), E.Request(2, [5, 6, 7, 8], 12)
    e.add(short); e.add(long_)
    e.run()
    assert short.done < long_.done


def test_no_block_leak_across_many_requests(tiny):
    cfg, w = tiny
    e = E.Engine(cfg, w, n_blocks=64, block_size=4, max_running=2,
                 enable_prefix_cache=False)
    for i in range(6):
        e.add(E.Request(i, [1, 2, 3, 4, 5, 6, 7, 8], 3))
    e.run()
    assert e.pool.alloc.n_used == 0
