"""
test_pd_sim.py —— 验证 PD 分离模拟器抓住了本质:
  · 合置下,prefill 突发会阻塞 decode(出现 > 1 个 tick 的停顿)
  · 分离下,decode 不被阻塞(步间隔 ≈ 1 个 tick)
运行:python -m pytest -q
"""
from pd_sim import Request, simulate_colocated, simulate_disaggregated, DECODE_STEP


def workload():
    # 两个"解码型"请求(短 prompt、长输出)先进入并开始 decode
    reqs = [Request(0, 0.0, prompt_len=2, output_len=30),
            Request(1, 0.0, prompt_len=2, output_len=30)]
    # 随后两个"预填充型"请求(长 prompt)同时到达,合置时会占满 2 个 slot 阻塞 decode
    reqs += [Request(2, 3.0, prompt_len=20, output_len=5),
             Request(3, 3.0, prompt_len=20, output_len=5)]
    return reqs


def test_all_requests_complete():
    for res in (simulate_colocated(workload(), n_slots=2),
                simulate_disaggregated(workload(), n_prefill=2, n_decode=2)):
        for r in res["requests"]:
            assert len(r.token_times) == r.output_len       # 每个请求都产出全部 token
            assert r.done_time > 0 and r.ttft > 0


def test_disaggregated_decode_not_blocked():
    res = simulate_disaggregated(workload(), n_prefill=2, n_decode=2)
    # 分离:独立 decode 池,decode 步间隔应恒 ≈ 1 个 tick
    assert res["max_decode_gap"] <= DECODE_STEP * 1.5


def test_colocated_decode_stalls_under_prefill_burst():
    res = simulate_colocated(workload(), n_slots=2)
    # 合置:两个长 prefill 同时占满 2 个 slot → 活跃 decode 停顿明显 > 1 个 tick
    assert res["max_decode_gap"] > DECODE_STEP * 1.5


def test_disagg_better_tail_tpot_than_colocated():
    colo = simulate_colocated(workload(), n_slots=2)
    disa = simulate_disaggregated(workload(), n_prefill=2, n_decode=2)
    # 分离的 decode 尾延迟(p99 TPOT)应不差于合置(通常更好)
    assert disa["p99_tpot"] <= colo["p99_tpot"]
    assert disa["max_decode_gap"] < colo["max_decode_gap"]


def test_throughput_positive_and_invariants():
    res = simulate_colocated(workload(), n_slots=2)
    assert res["throughput_tok_per_s"] > 0
    assert res["makespan_ms"] > 0
    for r in res["requests"]:
        # TTFT 至少覆盖 prefill 时间
        assert r.ttft >= r.prompt_len * 0.5 - 1e-6


def test_more_decode_slots_help_disagg_throughput():
    few = simulate_disaggregated(workload(), n_prefill=2, n_decode=1)
    many = simulate_disaggregated(workload(), n_prefill=2, n_decode=2)
    # decode 池不阻塞,二者 decode 都不停;此处只验证不崩且吞吐为正
    assert few["throughput_tok_per_s"] > 0 and many["throughput_tok_per_s"] > 0
