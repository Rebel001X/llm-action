"""MoE 下投机采样账本的测试。第 18、23 篇引用本文件。"""
import pytest

from moe import (MOE_MODELS, active_params, expected_active_experts, expert_size,
                 fwd_time_moe, moe_spec_speedup, tokens_to_saturate_moe, weight_bytes)
from speedup import H100, MODELS, scale, tokens_to_saturate


@pytest.mark.parametrize("name", list(MOE_MODELS))
def test_expert_decomposition_is_self_consistent(name):
    """规格自洽：稠密部分 + 全部专家 == 总参数量。"""
    m = MOE_MODELS[name]
    assert 0 < expert_size(m) < m["P_total"]
    assert m["P_dense"] + m["n_experts"] * expert_size(m) == pytest.approx(m["P_total"])
    assert active_params(m) < m["P_total"] / 5      # MoE 的定义性质：激活远小于总参


@pytest.mark.parametrize("name", list(MOE_MODELS))
def test_expert_decomposition_matches_published_active(name):
    """**用官方公布的激活参数量反查本文件的分解**。

    这条是补的：原来只查"P_dense + 全部专家 = 总参"这个内部恒等式，
    那是**怎么填 P_dense 都成立**的空检查 —— 一个太松的测试。
    加上这条之后才抓出 DeepSeek-V3 的 P_dense 填错了（13.0B 反推出 33.6B 激活，
    而官方是 37B），已改为 17.0B。
    """
    m = MOE_MODELS[name]
    assert active_params(m) == pytest.approx(m["P_active"], rel=0.03),         (name, active_params(m) / 1e9, m["P_active"] / 1e9)


def test_active_experts_saturates():
    """激活专家数随 token 数单调增且以专家总数为上界（券收集）。"""
    m = MOE_MODELS["gpt-oss-120b"]
    vals = [expected_active_experts(m, n) for n in (1, 4, 16, 64, 256, 4096)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert vals[0] == pytest.approx(m["top_k"], rel=0.05)   # 1 个 token 就是 top_k 个
    assert vals[-1] < m["n_experts"] + 1e-9
    assert vals[-1] / m["n_experts"] > 0.999                # 大 N 时贴住上限


def test_moe_free_budget_far_exceeds_dense():
    """**MoE 的免费额度比同量级稠密模型大一个数量级** —— 因为每 token 算力极低。"""
    hw = scale(H100, 8)
    dense = tokens_to_saturate(MODELS["llama3-70b"], hw, 64, 1024, 2.0)
    moe = tokens_to_saturate_moe(MOE_MODELS["gpt-oss-120b"], hw, 64, 1024, 0.53)
    assert moe > 5 * dense, (dense, moe)


def test_small_batch_penalty_exists_only_in_moe():
    """**MoE 特有**：小 batch 下投机要多读权重（稠密模型上这一项恒为 1）。"""
    hw = scale(H100, 8)
    r = moe_spec_speedup("gpt-oss-120b", hw, 1, 1024, 4, 3.0, 0.53)
    assert r["mem_ratio"] > 3.0, r          # batch=1 时多读 3 倍以上
    assert r["speedup"] < 1.0, r            # 于是小 batch 反而亏

    # 对照：稠密模型的访存与 query token 数无关
    a = fwd_time_moe(MOE_MODELS["gpt-oss-120b"], hw, 1, 1024, 1, 0.53)
    b = fwd_time_moe(MOE_MODELS["gpt-oss-120b"], hw, 1, 1024, 5, 0.53)
    assert b["mem_bytes"] > a["mem_bytes"] * 3


def test_penalty_vanishes_at_moderate_batch():
    """batch 到几十以后两边都激活满，访存比回到 1 —— '权重只读一次'在 MoE 上是大 batch 性质。"""
    hw = scale(H100, 8)
    r = moe_spec_speedup("gpt-oss-120b", hw, 200, 1024, 4, 3.0, 0.53)
    assert r["mem_ratio"] < 1.01, r
    assert r["experts_base"] / MOE_MODELS["gpt-oss-120b"]["n_experts"] > 0.99


def test_moe_still_profitable_at_concurrency_200():
    """**解释那个反例**：gpt-oss-120b + MXFP4 在并发 200 时仍是 memory-bound、仍然赚。"""
    hw = scale(H100, 8)
    r = moe_spec_speedup("gpt-oss-120b", hw, 200, 1024, 4, 3.0, 0.53)
    assert r["verify_bound"] == "memory"
    assert r["speedup"] > 1.5, r


def test_moe_sweet_spot_is_mid_batch_unlike_dense():
    """**与稠密恰好相反**：MoE 的最优区间在中等 batch，不在 batch=1。"""
    hw = scale(H100, 8)
    s = {b: moe_spec_speedup("gpt-oss-120b", hw, b, 1024, 4, 3.0, 0.53)["speedup"]
         for b in (1, 8, 32, 128, 512, 2048)}
    assert s[1] < s[32] and s[8] < s[32], s        # 小 batch 不是最优
    assert max(s, key=s.get) not in (1, 8), s
    # 稠密模型则是 batch 越小越好
    from speedup import spec_throughput
    d = {b: spec_throughput("llama3-70b", "llama3.2-1b", hw, b, 1024, 4, 3.0)["speedup"]
         for b in (1, 8, 32, 128, 512)}
    assert max(d, key=d.get) == 1, d


def test_moe_outlasts_dense_by_an_order_of_magnitude():
    """MoE 在稠密模型早已翻转的 batch 上仍然盈利。"""
    hw = scale(H100, 8)
    from speedup import spec_throughput
    dense_1024 = spec_throughput("llama3-70b", "llama3.2-1b", hw, 1024, 1024, 4, 3.0)["speedup"]
    moe_1024 = moe_spec_speedup("gpt-oss-120b", hw, 1024, 1024, 4, 3.0, 0.53)["speedup"]
    assert dense_1024 < 1.0 < moe_1024, (dense_1024, moe_1024)
