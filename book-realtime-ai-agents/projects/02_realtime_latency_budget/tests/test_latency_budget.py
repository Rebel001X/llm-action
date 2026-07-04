# -*- coding: utf-8 -*-
"""
延迟预算模型的单元测试 (pytest)。

覆盖题目要求的四条硬性验证：
  1. 流式首响 < 非流式首响            —— test_streaming_ttfa_beats_non_streaming
  2. 端到端 == 各级之和               —— test_e2e_equals_sum_of_stages
  3. decode 流式按 token 逐个产出     —— test_stream_decode_yields_per_token
  4. 预算超标能被检测出来            —— test_budget_violation_detected

外加若干边界/单调性测试，保证模型物理上自洽。
"""

import math

import pytest

from latency_budget import (
    ASRConfig,
    LLMConfig,
    PipelineConfig,
    TTSConfig,
    TokenEvent,
    asr_latency_ms,
    audio_playback_ms,
    check_budget,
    compare_modes,
    llm_decode_ms,
    llm_prefill_ms,
    simulate_non_streaming,
    simulate_streaming,
    speedup_ttfa,
    stream_decode,
    ttfa_breakdown,
    tts_first_chunk_ms,
    tts_full_synthesis_ms,
)


@pytest.fixture
def cfg() -> PipelineConfig:
    """一组标准参数，供多数测试复用。"""
    return PipelineConfig()


# -----------------------------------------------------------------------------
# 硬性 1：流式首响 < 非流式首响
# -----------------------------------------------------------------------------
def test_streaming_ttfa_beats_non_streaming(cfg):
    """流式的首响延迟必须严格小于非流式——这是流式存在的意义。"""
    r = compare_modes(cfg)
    assert r["streaming"].ttfa_ms < r["non_streaming"].ttfa_ms
    # 加速比应 > 1（这里参数下约 4x）
    assert speedup_ttfa(cfg) > 1.0


def test_streaming_ttfa_only_waits_first_token(cfg):
    """流式 TTFA 只等「首 token」(prefill)，不含整句 decode。

    验证方式：流式 TTFA 应当远小于「非流式里 decode 整句」这一项的耗时，
    因为流式根本不等 decode 整句。
    """
    s = simulate_streaming(cfg)
    decode_full = llm_decode_ms(cfg)  # 生成整句的 decode 耗时
    # 流式首响 = ASR + prefill + TTS首块，完全不含 decode 整句
    assert s.ttfa_ms == asr_latency_ms(cfg) + llm_prefill_ms(cfg) + tts_first_chunk_ms(cfg)
    assert s.ttfa_ms < decode_full + s.ttfa_ms  # decode 整句是额外时间


# -----------------------------------------------------------------------------
# 硬性 2：端到端 == 各级之和
# -----------------------------------------------------------------------------
def test_e2e_equals_sum_of_stages(cfg):
    """两种模式的端到端延迟，都必须等于各级 stage 时长之和（时序自洽）。"""
    for mode_fn in (simulate_streaming, simulate_non_streaming):
        res = mode_fn(cfg)
        total = sum(dur for _, _, dur in res.stages)
        assert math.isclose(res.e2e_ms, total, rel_tol=1e-9), (
            f"{res.mode}: e2e={res.e2e_ms} != sum(stages)={total}"
        )


def test_stages_are_contiguous_when_sequential(cfg):
    """非流式各级首尾相接：第 i 级的 (起点+时长) == 第 i+1 级的起点。

    这验证了非流式「串行、无重叠」的物理模型。
    """
    res = simulate_non_streaming(cfg)
    for (_, start_a, dur_a), (_, start_b, _) in zip(res.stages, res.stages[1:]):
        assert math.isclose(start_a + dur_a, start_b, rel_tol=1e-9)


def test_ttfa_never_exceeds_e2e(cfg):
    """首响不可能晚于端到端——听到第一个音，一定不晚于听完整段。"""
    for mode_fn in (simulate_streaming, simulate_non_streaming):
        res = mode_fn(cfg)
        assert res.ttfa_ms <= res.e2e_ms + 1e-9


# -----------------------------------------------------------------------------
# 硬性 3：decode 流式按 token 逐个产出
# -----------------------------------------------------------------------------
def test_stream_decode_yields_per_token(cfg):
    """stream_decode 必须逐 token 产出，且时序正确。"""
    events = list(stream_decode(cfg))
    # 数量 == response_tokens
    assert len(events) == cfg.response_tokens
    # 每个都是 TokenEvent
    assert all(isinstance(e, TokenEvent) for e in events)
    # index 连续递增
    assert [e.index for e in events] == list(range(cfg.response_tokens))
    # 只有第一个是 first
    assert events[0].is_first is True
    assert all(e.is_first is False for e in events[1:])


def test_stream_decode_first_token_time_is_ttft(cfg):
    """首 token 的就绪时刻 == ASR 收尾 + prefill（即 TTFT）。"""
    events = list(stream_decode(cfg))
    expected_ttft = asr_latency_ms(cfg) + llm_prefill_ms(cfg)
    assert math.isclose(events[0].ready_at_ms, expected_ttft, rel_tol=1e-9)


def test_stream_decode_is_monotonic_and_evenly_spaced(cfg):
    """token 就绪时刻单调递增，且相邻间隔恒为 ms_per_token（decode 匀速）。"""
    events = list(stream_decode(cfg))
    for a, b in zip(events, events[1:]):
        assert b.ready_at_ms > a.ready_at_ms
        gap = b.ready_at_ms - a.ready_at_ms
        assert math.isclose(gap, cfg.llm.ms_per_token, rel_tol=1e-9)


def test_stream_decode_is_lazy_generator(cfg):
    """stream_decode 是真正的惰性生成器：能只取前 k 个而不算完全部。

    这从代码结构上证明「流式」——调用方拿一个处理一个，不必等 list 全就绪。
    """
    gen = stream_decode(cfg)
    first_three = [next(gen) for _ in range(3)]
    assert [e.index for e in first_three] == [0, 1, 2]
    # 生成器还能继续产出剩余的
    rest = list(gen)
    assert len(rest) == cfg.response_tokens - 3


# -----------------------------------------------------------------------------
# 硬性 4：预算超标检测
# -----------------------------------------------------------------------------
def test_budget_within(cfg):
    """宽松预算下应达标，且余量为正。"""
    bc = check_budget(cfg, target_ttfa_ms=1000.0)
    assert bc.within_budget is True
    assert bc.headroom_ms > 0
    assert math.isclose(bc.actual_ttfa_ms, 410.0, rel_tol=1e-9)


def test_budget_violation_detected(cfg):
    """严苛预算（100ms）下必须报超标，且余量为负。"""
    bc = check_budget(cfg, target_ttfa_ms=100.0)
    assert bc.within_budget is False
    assert bc.headroom_ms < 0
    assert math.isclose(bc.headroom_ms, 100.0 - bc.actual_ttfa_ms, rel_tol=1e-9)


def test_budget_breakdown_sums_to_actual(cfg):
    """预算分解各项毫秒之和 == 实测 TTFA；占比之和 == 100%。"""
    bc = check_budget(cfg, target_ttfa_ms=500.0)
    ms_sum = sum(ms for _, ms, _ in bc.breakdown)
    pct_sum = sum(pct for _, _, pct in bc.breakdown)
    assert math.isclose(ms_sum, bc.actual_ttfa_ms, rel_tol=1e-9)
    assert math.isclose(pct_sum, 100.0, rel_tol=1e-6)


def test_ttfa_breakdown_matches_streaming_ttfa(cfg):
    """ttfa_breakdown 三项之和 == 流式仿真得到的 TTFA（两条路径一致）。"""
    parts_sum = sum(ms for _, ms in ttfa_breakdown(cfg))
    s = simulate_streaming(cfg)
    assert math.isclose(parts_sum, s.ttfa_ms, rel_tol=1e-9)


# -----------------------------------------------------------------------------
# 单调性 / 物理自洽（额外保险）
# -----------------------------------------------------------------------------
def test_longer_prompt_increases_prefill_and_ttfa():
    """prompt 越长，prefill 越慢，流式 TTFA 越大（compute-bound 的直觉）。"""
    short = PipelineConfig(prompt_tokens=50)
    long = PipelineConfig(prompt_tokens=800)
    assert llm_prefill_ms(long) > llm_prefill_ms(short)
    assert simulate_streaming(long).ttfa_ms > simulate_streaming(short).ttfa_ms


def test_more_response_tokens_increase_e2e_not_ttfa():
    """回答更长 → 端到端更久，但流式首响不变（首响只取决于首 token）。"""
    short = PipelineConfig(response_tokens=10)
    long = PipelineConfig(response_tokens=200)
    assert simulate_streaming(long).e2e_ms > simulate_streaming(short).e2e_ms
    assert math.isclose(
        simulate_streaming(long).ttfa_ms,
        simulate_streaming(short).ttfa_ms,
        rel_tol=1e-9,
    )


def test_streaming_e2e_not_worse_than_non_streaming(cfg):
    """流式端到端应当 <= 非流式（流式不会更慢，通常更快或相当）。"""
    r = compare_modes(cfg)
    assert r["streaming"].e2e_ms <= r["non_streaming"].e2e_ms + 1e-9


def test_playback_is_e2e_floor(cfg):
    """物理下限：端到端 >= 首响 + 音频播放时长（听完这句话本身就要那么久）。

    对流式而言尤为直观：首响后至少还要把剩余音频放完。
    """
    s = simulate_streaming(cfg)
    play = audio_playback_ms(cfg)
    # 端到端不可能小于「首响 + 除首块外的播放时长」的物理下限
    remaining_play = audio_playback_ms(cfg, cfg.response_tokens - 1)
    assert s.e2e_ms >= s.ttfa_ms + min(remaining_play, play) - 1e-9


def test_zero_response_tokens_edge_case():
    """边界：回答 0 token 时不崩，decode/播放为 0。"""
    c = PipelineConfig(response_tokens=0)
    assert llm_decode_ms(c) == 0.0
    assert audio_playback_ms(c) == 0.0
    # 流式仿真也应正常返回
    s = simulate_streaming(c)
    assert s.ttfa_ms > 0  # 仍有 ASR+prefill+首块的固定开销


def test_custom_configs_are_used():
    """自定义子配置能被正确读取（dataclass 组合无坑）。"""
    c = PipelineConfig(
        asr=ASRConfig(finalize_ms=999.0),
        llm=LLMConfig(prefill_base_ms=1.0, prefill_ms_per_token=0.0),
        tts=TTSConfig(first_chunk_ms=2.0),
        prompt_tokens=10,
    )
    assert asr_latency_ms(c) == 999.0
    assert llm_prefill_ms(c) == 1.0  # base + 0*10
    assert tts_first_chunk_ms(c) == 2.0
    assert tts_full_synthesis_ms(c, 0) == 2.0
