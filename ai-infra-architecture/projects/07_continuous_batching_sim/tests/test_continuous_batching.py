"""
test_continuous_batching.py —— 验证连续批处理模拟器抓住了本质。
运行(在项目根目录):python -m pytest -q

覆盖:
  · 不变量:所有请求完成、token 数正确、ttft/done 合法、时间戳单调。
  · 连续吞吐 ≥ 静态(work-conserving → makespan 更短)。
  · 连续 GPU 空闲 slot-steps ≤ 静态(空 slot 立刻回填)。
  · 连续的尾延迟 / 平均排队 优于静态。
  · 静态的"批内长度方差"会造成明显空转;连续把它吃掉。
  · 极端 case(单请求、batch=1、大 batch)不崩、语义正确。
"""
import random

from continuous_batching import (
    Request, simulate_static, simulate_continuous, summarize, _percentile,
)


# ---------------- 工作负载构造 ----------------
def mixed_workload(n=48, seed=0):
    """混合负载:多数短输出(交互),少数长输出(长文生成)→ 批内长度方差大。"""
    rnd = random.Random(seed)
    reqs = []
    for i in range(n):
        if rnd.random() < 0.25:
            out = rnd.randint(60, 100)          # 长生成
        else:
            out = rnd.randint(2, 10)            # 短交互
        reqs.append(Request(i, arrival=rnd.uniform(0, 25),
                            prompt_len=rnd.randint(16, 256), output_len=out))
    return reqs


def variance_workload():
    """确定性负载:1 条超长 + 一堆超短,全部 t=0 到达,batch=2 时差异最大化。"""
    reqs = [Request(0, 0.0, prompt_len=16, output_len=40)]     # 长
    reqs += [Request(i, 0.0, prompt_len=16, output_len=1) for i in range(1, 9)]  # 8 条短
    return reqs


def _assert_invariants(res):
    for r in res.requests:
        assert len(r.token_times) == r.output_len          # 每条都产出全部 token
        assert r.done > 0 and r.ttft > 0 and r.start >= r.arrival - 1e-9
        assert r.token_times == sorted(r.token_times)      # token 时刻单调不减
        assert abs(r.done - r.token_times[-1]) < 1e-9       # done = 最后一个 token


# ---------------- 测试 ----------------
def test_all_requests_complete_both():
    wl = mixed_workload()
    for res in (simulate_static(wl, batch_size=4),
                simulate_continuous(wl, batch_size=4)):
        _assert_invariants(res)
        assert res.total_output_tokens() == sum(r.output_len for r in wl)


def test_continuous_throughput_ge_static():
    wl = mixed_workload()
    s = summarize(simulate_static(wl, batch_size=4))
    c = summarize(simulate_continuous(wl, batch_size=4))
    # 连续是 work-conserving:makespan ≤ 静态 → 吞吐 ≥ 静态
    assert c["throughput_tok_s"] >= s["throughput_tok_s"] - 1e-6
    assert c["makespan_ms"] <= s["makespan_ms"] + 1e-6


def test_continuous_less_idle_and_higher_util():
    wl = mixed_workload()
    s = simulate_static(wl, batch_size=4)
    c = simulate_continuous(wl, batch_size=4)
    # 连续把先完成请求的空 slot 立刻回填 → 空转更少、利用率更高
    assert c.idle_slot_steps() <= s.idle_slot_steps()
    assert c.gpu_util() >= s.gpu_util() - 1e-9


def test_continuous_better_tail_and_queue():
    wl = mixed_workload()
    s = summarize(simulate_static(wl, batch_size=4))
    c = summarize(simulate_continuous(wl, batch_size=4))
    # 连续消除队头阻塞 → 平均排队更短、尾延迟更低
    assert c["mean_queue"] <= s["mean_queue"] + 1e-6
    assert c["p99_latency"] <= s["p99_latency"] + 1e-6


def test_variance_makes_static_strictly_worse():
    """1 长 + 8 短、batch=2:静态被最慢者拖住且不能中途补入 → 严格差于连续。"""
    wl = variance_workload()
    s = simulate_static(wl, batch_size=2)
    c = simulate_continuous(wl, batch_size=2)
    _assert_invariants(s)
    _assert_invariants(c)
    # 连续:长请求跑的同时,短请求源源不断补进那 1 个空 slot → makespan 明显更短
    assert c.makespan < s.makespan
    assert c.idle_slot_steps() < s.idle_slot_steps()
    assert summarize(c)["throughput_tok_s"] > summarize(s)["throughput_tok_s"]


def test_util_in_valid_range_and_idle_nonneg():
    wl = mixed_workload(seed=3)
    for res in (simulate_static(wl, batch_size=8),
                simulate_continuous(wl, batch_size=8)):
        assert 0.0 <= res.gpu_util() <= 1.0 + 1e-9
        assert res.idle_slot_steps() >= 0
        assert res.busy_slot_steps() == sum(r.service_steps() for r in res.requests)


def test_single_request_equivalent():
    """只有一条请求时,静态与连续应完全等价(没有并发/回填可言)。"""
    wl = [Request(0, 0.0, prompt_len=64, output_len=20)]
    s = summarize(simulate_static(wl, batch_size=4))
    c = summarize(simulate_continuous(wl, batch_size=4))
    assert abs(s["makespan_ms"] - c["makespan_ms"]) < 1e-9
    assert abs(s["throughput_tok_s"] - c["throughput_tok_s"]) < 1e-6


def test_batch_size_one_is_sequential():
    """batch=1 时两种策略都退化为串行,makespan 相同(逐条排队)。"""
    wl = mixed_workload(n=12, seed=7)
    s = simulate_static(wl, batch_size=1)
    c = simulate_continuous(wl, batch_size=1)
    assert abs(s.makespan - c.makespan) < 1e-9
    _assert_invariants(s)
    _assert_invariants(c)


def test_larger_batch_improves_continuous_throughput():
    """并发度更高时,连续批处理吞吐应不降(通常上升),直到被负载/长度上限约束。"""
    wl = mixed_workload(seed=1)
    small = summarize(simulate_continuous(wl, batch_size=2))
    big = summarize(simulate_continuous(wl, batch_size=16))
    assert big["throughput_tok_s"] >= small["throughput_tok_s"] - 1e-6
    assert big["gpu_util"] <= 1.0 + 1e-9


def test_percentile_helper():
    xs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _percentile(xs, 0.5) == 50
    assert _percentile(xs, 1.0) == 100
    assert _percentile([], 0.9) == 0.0
    assert _percentile([42], 0.99) == 42
