# -*- coding: utf-8 -*-
"""
实时交互延迟预算模型 (Real-Time Interaction Latency Budget Model)
================================================================

配套《Multimodal Real-Time AI Agent Systems》第 2 章「为实时 AI 交互做架构」。

核心命题：一句「你好」传进麦克风，到用户耳朵里听见第一个音节回音，
中间到底经历了哪些延迟？为什么「流式（streaming）」能把「首响延迟
（TTFA, Time-To-First-Audio）」砍到非流式的几分之一？

本模块把 ASR → LLM(prefill+decode) → TTS 这条实时语音流水线建成一个
**可计算、可验证、可分解**的延迟模型，回答三个工程问题：

    1. 首响延迟 (TTFA)      —— 用户开口后多久听到第一个字的声音？
    2. 端到端延迟 (E2E)     —— 整句话说完到整句回答播完，一共多久？
    3. 流式 vs 非流式       —— 流式凭什么快？快在哪个环节？代价是什么？

并给出「延迟预算分解 (latency budget breakdown)」：把总预算像切蛋糕一样
分给每一级，超标了立刻报警。

设计原则：纯 CPU、无网络、无外部模型。所有「模型」都是**确定性的延迟仿真器**，
不真的跑 ASR/LLM/TTS，而是用「每 token 多少毫秒」这类物理参数精确复现时序。
这正是延迟工程 (latency engineering) 在纸面上该做的事：先把预算算清楚，
再去堆工程实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


# =============================================================================
# 一、单级组件的延迟参数 (Per-stage latency parameters)
# =============================================================================
# 我们把流水线拆成三级：ASR（语音转文字）、LLM（大模型推理）、TTS（文字转语音）。
# 每一级都用几个「物理参数」刻画其延迟行为，参数含义见各字段中文注释。


@dataclass
class ASRConfig:
    """自动语音识别 (Automatic Speech Recognition) 的延迟参数。

    真实世界里 ASR 有两种形态：
      - 非流式：等你把整句话说完，再一次性转成文字（准，但慢）。
      - 流式：边说边转，说完瞬间就吐出最终文字（快，靠增量解码）。

    我们用两个参数刻画：
      - finalize_ms：从「用户说完最后一个音」到「ASR 交付最终文本」的收尾延迟。
        流式 ASR 的 finalize_ms 很小（因为大部分识别在你说话时已经做完了）。
      - realtime_factor：ASR 处理音频的速度相对于音频时长的倍率。
        =1.0 表示「处理 1 秒音频恰好花 1 秒」（实时）；<1 表示比实时更快。
        仅在非流式估算「处理整段音频」时用到。
    """

    finalize_ms: float = 150.0       # 收尾延迟（毫秒）：说完到出最终文本
    realtime_factor: float = 0.3     # 处理倍率：0.3 = 处理速度是音频时长的 3.3 倍


@dataclass
class LLMConfig:
    """大语言模型 (Large Language Model) 推理的延迟参数。

    LLM 推理分两个物理上完全不同的阶段，这是理解实时延迟的**核心**：

      1. Prefill（预填充 / 首 token 前的计算）：
         把整段输入 prompt（system + 历史 + 用户这句话）一次性喂进模型，
         并行地算完所有输入 token 的注意力，产出「第一个输出 token」。
         这一步是**计算密集（compute-bound）**的，耗时随输入长度增长。
         prefill 结束的时刻 = TTFT（Time-To-First-Token，首 token 延迟）。

      2. Decode（解码 / 逐 token 生成）：
         之后每生成一个新 token，都要读一遍 KV cache，是**访存密集
         （memory-bound）**的。每个 token 花的时间近似恒定 = ms_per_token。
         输出多少 token，就重复多少次。

    这就是为什么「流式输出」有意义：decode 是一个 token 一个 token 出来的，
    我们不必等整句话生成完，第一个 token 一出来就能送去 TTS。
    """

    prefill_base_ms: float = 40.0        # prefill 固定开销（毫秒）
    prefill_ms_per_token: float = 0.5    # prefill 每个输入 token 的增量耗时
    ms_per_token: float = 25.0           # decode 每个输出 token 的耗时（≈40 tok/s）


@dataclass
class TTSConfig:
    """文字转语音 (Text-To-Speech) 合成的延迟参数。

    TTS 也有两种形态：
      - 非流式：等 LLM 把整句话写完，再一次性合成整段音频。
      - 流式：LLM 每吐出一个「可合成单元」（一个词 / 一个句子片段），
        TTS 立刻合成对应的一小段音频并开始播放。

    参数：
      - first_chunk_ms：合成「第一小段音频」的延迟（流式关键指标）。
        这决定了「文本准备好」之后多久能发出第一个声音。
      - ms_per_token：每个文本 token 对应的合成耗时（非流式整段合成时累加）。
      - audio_ms_per_token：每个文本 token 合成出的**音频时长**（毫秒）。
        注意：这是「播放时长」，不是「计算耗时」。它决定端到端里
        「把话说完」本身要占多久墙钟时间。
    """

    first_chunk_ms: float = 120.0        # 首个音频块的合成延迟（毫秒）
    ms_per_token: float = 8.0            # 每 token 的合成计算耗时
    audio_ms_per_token: float = 60.0     # 每 token 对应的音频播放时长（毫秒）


@dataclass
class PipelineConfig:
    """整条实时语音流水线的输入规模 + 三级组件参数。

    输入规模（决定各级要处理多少活）：
      - user_speech_ms：用户这句话说了多长（音频毫秒数）。
      - prompt_tokens：喂给 LLM 的完整 prompt 有多少 token（含 system+历史+用户）。
      - response_tokens：LLM 预计生成多少个输出 token。
    """

    user_speech_ms: float = 2000.0       # 用户说话时长（毫秒）
    prompt_tokens: int = 200             # LLM 输入 prompt 的 token 数
    response_tokens: int = 40            # LLM 输出 response 的 token 数

    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


# =============================================================================
# 二、单级延迟的计算函数 (Per-stage latency computation)
# =============================================================================
# 每个函数只回答「这一级要花多少毫秒」，不掺流水线编排，方便单元测试。


def asr_latency_ms(cfg: PipelineConfig) -> float:
    """ASR 收尾延迟：用户说完最后一个音，到 ASR 交付最终文本，花多少毫秒。

    在流式 ASR 里，识别工作在用户说话的同时就并行做掉了，所以这里只算
    「收尾」那一小段，即 finalize_ms。我们不把「用户说话时长」算进 ASR 延迟，
    因为那是用户自己占的时间，不是系统的账。
    """
    return cfg.asr.finalize_ms


def llm_prefill_ms(cfg: PipelineConfig) -> float:
    """LLM prefill 延迟 = TTFT（首 token 延迟）。

    prefill 耗时 = 固定开销 + 每输入 token 的增量 × prompt token 数。
    prompt 越长，prefill 越慢——这就是为什么「精简 system prompt / 裁剪历史」
    是降低实时延迟的头号手段。
    """
    return cfg.llm.prefill_base_ms + cfg.llm.prefill_ms_per_token * cfg.prompt_tokens


def llm_decode_ms(cfg: PipelineConfig, n_tokens: int | None = None) -> float:
    """LLM decode 延迟：生成 n_tokens 个输出 token 花多少毫秒。

    decode 每 token 近似恒定耗时，所以总耗时 = ms_per_token × token 数。
    n_tokens 缺省时用 response_tokens（生成整句）。
    """
    if n_tokens is None:
        n_tokens = cfg.response_tokens
    return cfg.llm.ms_per_token * n_tokens


def tts_first_chunk_ms(cfg: PipelineConfig) -> float:
    """TTS 首块延迟：文本准备好后，合成出第一个可播放音频块花多少毫秒。"""
    return cfg.tts.first_chunk_ms


def tts_full_synthesis_ms(cfg: PipelineConfig, n_tokens: int | None = None) -> float:
    """TTS 整段合成的**计算**耗时（非流式用）：把 n_tokens 文本全合成完花多少毫秒。

    注意区分两个「时间」：
      - 合成计算耗时：first_chunk_ms + ms_per_token × token 数（这里算的）。
      - 音频播放时长：audio_ms_per_token × token 数（见 audio_playback_ms）。
    """
    if n_tokens is None:
        n_tokens = cfg.response_tokens
    return cfg.tts.first_chunk_ms + cfg.tts.ms_per_token * n_tokens


def audio_playback_ms(cfg: PipelineConfig, n_tokens: int | None = None) -> float:
    """回答音频的**播放时长**：把 n_tokens 文本对应的语音播完，需要多少墙钟毫秒。

    这是物理下限：哪怕你把所有计算延迟压到 0，用户听完这句话本身也要花
    这么久。端到端延迟不可能小于「首响延迟 + 播放时长」。
    """
    if n_tokens is None:
        n_tokens = cfg.response_tokens
    return cfg.tts.audio_ms_per_token * n_tokens


# =============================================================================
# 三、流水线编排：非流式 vs 流式 (Pipeline orchestration)
# =============================================================================


@dataclass
class LatencyResult:
    """一次流水线仿真的结果。所有时间单位均为毫秒 (ms)。

    - ttfa_ms：首响延迟 (Time-To-First-Audio)。用户说完 → 听到第一个音。
      这是实时体验**最关键**的指标：人对「对方开始回应」的等待极其敏感。
    - e2e_ms：端到端延迟 (End-to-End)。用户说完 → 整段回答播放完毕。
    - stages：分级延迟明细（用于画瀑布图 + 预算分解），有序的 (名称, 起点ms, 时长ms)。
    - mode：'streaming' 或 'non_streaming'。
    """

    ttfa_ms: float
    e2e_ms: float
    stages: list[tuple[str, float, float]]
    mode: str


def simulate_non_streaming(cfg: PipelineConfig) -> LatencyResult:
    """非流式流水线：每一级都等上一级**完全做完**才开始。

    时序（从「用户说完」这一刻 t=0 起算）：
        [ASR 收尾] → [LLM prefill] → [LLM decode 整句] → [TTS 整段合成] → [播放整段]

    关键特征：TTFA 极大——因为第一个音频要等到 LLM 把**整句话**生成完、
    TTS 把**整段**合成完，才可能播放。用户会经历一段死寂的「空气时间
    （dead air）」，这正是第一代语音助手体验糟糕的技术根源之一。
    """
    stages: list[tuple[str, float, float]] = []
    t = 0.0

    d_asr = asr_latency_ms(cfg)
    stages.append(("ASR 收尾", t, d_asr))
    t += d_asr

    d_prefill = llm_prefill_ms(cfg)
    stages.append(("LLM prefill", t, d_prefill))
    t += d_prefill

    d_decode = llm_decode_ms(cfg)                      # 生成整句
    stages.append(("LLM decode(整句)", t, d_decode))
    t += d_decode

    d_tts = tts_full_synthesis_ms(cfg)                 # 合成整段
    stages.append(("TTS 整段合成", t, d_tts))
    t += d_tts

    # 非流式：第一个音频 = 整段合成完那一刻才开始播 → TTFA 就是这里的 t
    ttfa = t

    d_play = audio_playback_ms(cfg)                    # 播放整段
    stages.append(("音频播放", t, d_play))
    t += d_play

    return LatencyResult(ttfa_ms=ttfa, e2e_ms=t, stages=stages, mode="non_streaming")


def simulate_streaming(cfg: PipelineConfig) -> LatencyResult:
    """流式流水线：各级**尽早开工**，第一个输出 token 一出来就往下游送。

    时序（t=0 = 用户说完）：
        [ASR 收尾] → [LLM prefill 出首 token] → [TTS 合成首块] → 【第一个音频！】
                                              ↘ 与此同时 LLM 继续 decode 剩余 token，
                                                TTS 继续合成剩余音频，边合成边播放。

    TTFA（首响）= ASR 收尾 + LLM prefill + TTS 首块。
    注意：这里 **只等 LLM 出「第一个 token」**（即 prefill 结束），
    不等它把整句写完！这就是流式把首响砍到几分之一的根本原因。

    E2E（端到端）：首响之后，音频要么被「计算」卡住（decode/合成跟不上），
    要么被「播放」卡住（音频本身要放那么久）。理想流式系统中，只要
    decode+合成的速度 ≥ 播放速度，播放就不会断流，E2E ≈ 首响 + 播放整段时长。
    我们按更真实的方式取二者上界：E2E = 首响 + max(剩余计算时间, 剩余播放时间)。
    """
    stages: list[tuple[str, float, float]] = []
    t = 0.0

    d_asr = asr_latency_ms(cfg)
    stages.append(("ASR 收尾", t, d_asr))
    t += d_asr

    d_prefill = llm_prefill_ms(cfg)                    # 出第一个 token
    stages.append(("LLM prefill(首token)", t, d_prefill))
    t += d_prefill

    # 第一个 token 已就绪，TTS 合成第一小段音频
    d_tts_first = tts_first_chunk_ms(cfg)
    stages.append(("TTS 首块合成", t, d_tts_first))
    t += d_tts_first

    # 到这里，第一个音频块可以播放了 → 这就是 TTFA
    ttfa = t

    # 首响之后：剩余 token 的 decode + 合成 与 播放 并行进行。
    # 剩余「计算」时间：把剩下的 (response_tokens - 1) 个 token decode 完 + 合成完。
    remaining_tokens = max(cfg.response_tokens - 1, 0)
    remaining_compute = (
        llm_decode_ms(cfg, remaining_tokens)
        + cfg.tts.ms_per_token * remaining_tokens
    )
    # 剩余「播放」时间：整段音频总时长，减去第一块已开始播放的部分。
    # 简化：第一块对应 1 个 token 的音频，后续是 remaining_tokens 个 token 的音频。
    remaining_play = audio_playback_ms(cfg, remaining_tokens)

    # 播放不断流的前提下，尾巴由二者较大者决定（谁慢谁卡脖子）。
    tail = max(remaining_compute, remaining_play)
    stages.append(("首响后并行(计算/播放)", ttfa, tail))

    e2e = ttfa + tail
    return LatencyResult(ttfa_ms=ttfa, e2e_ms=e2e, stages=stages, mode="streaming")


# =============================================================================
# 四、真正的「流式 decode 逐 token 出」生成器 (Streaming decode as generator)
# =============================================================================


@dataclass
class TokenEvent:
    """decode 过程中吐出的一个 token 事件（用于验证「逐 token 流式产出」）。

    - index：这是第几个 token（从 0 开始）。
    - ready_at_ms：这个 token 在墙钟上的就绪时刻（相对 t=0=用户说完）。
    - is_first：是否为首 token（首 token 的就绪时刻 = TTFT）。
    """

    index: int
    ready_at_ms: float
    is_first: bool


def stream_decode(cfg: PipelineConfig) -> Iterator[TokenEvent]:
    """以生成器形式**逐个**产出 decode token，复现「流式一个一个出」的时序。

    首 token 在 ASR 收尾 + prefill 之后就绪；之后每隔 ms_per_token 出一个。
    这个生成器是「流式」区别于「非流式」的本质：调用方可以在 for 循环里
    **拿到一个就处理一个**（送 TTS），而不必等 list 全部 return。

    ⚠️ 用 yield 而非 return list，是为了让「流式」在代码结构上就成立：
    非流式对应「先算完整个 list 再返回」，流式对应「算一个 yield 一个」。
    """
    first_ready = asr_latency_ms(cfg) + llm_prefill_ms(cfg)
    for i in range(cfg.response_tokens):
        ready_at = first_ready + cfg.llm.ms_per_token * i
        yield TokenEvent(index=i, ready_at_ms=ready_at, is_first=(i == 0))


# =============================================================================
# 五、延迟预算分解与超标检测 (Latency budget breakdown & violation check)
# =============================================================================


@dataclass
class BudgetCheck:
    """预算检查结果。

    - target_ttfa_ms：为 TTFA 设定的预算上限（工程 SLO，如「首响必须 < 500ms」）。
    - actual_ttfa_ms：实测/仿真得到的 TTFA。
    - within_budget：是否达标（actual <= target）。
    - breakdown：TTFA 内部各级贡献 (名称, 毫秒, 占比%)，帮你定位「谁吃掉了预算」。
    - headroom_ms：预算余量（正=还有空间，负=超标多少毫秒）。
    """

    target_ttfa_ms: float
    actual_ttfa_ms: float
    within_budget: bool
    breakdown: list[tuple[str, float, float]]
    headroom_ms: float


def ttfa_breakdown(cfg: PipelineConfig) -> list[tuple[str, float]]:
    """把流式 TTFA 拆成三个可归因的部分：ASR 收尾 / LLM prefill / TTS 首块。

    返回 [(名称, 毫秒), ...]，三者之和 == 流式 TTFA。这是延迟工程里最有用的
    一张表：它告诉你「要把首响再砍 100ms，该去优化哪一级」。
    """
    return [
        ("ASR 收尾", asr_latency_ms(cfg)),
        ("LLM prefill", llm_prefill_ms(cfg)),
        ("TTS 首块", tts_first_chunk_ms(cfg)),
    ]


def check_budget(cfg: PipelineConfig, target_ttfa_ms: float) -> BudgetCheck:
    """检查流式 TTFA 是否在预算内，并给出各级占比。

    这是把「延迟预算 (latency budget)」变成一个**可执行的守卫**：
    在 CI 里跑它，一旦某次改动（比如 system prompt 变长）把 TTFA 顶破预算，
    测试立刻红灯。这正是实时系统防止「延迟回归 (latency regression)」的手段。
    """
    parts = ttfa_breakdown(cfg)
    actual = sum(ms for _, ms in parts)
    breakdown = [
        (name, ms, (ms / actual * 100.0 if actual > 0 else 0.0)) for name, ms in parts
    ]
    return BudgetCheck(
        target_ttfa_ms=target_ttfa_ms,
        actual_ttfa_ms=actual,
        within_budget=actual <= target_ttfa_ms,
        breakdown=breakdown,
        headroom_ms=target_ttfa_ms - actual,
    )


# =============================================================================
# 六、便捷汇总 (Convenience summary)
# =============================================================================


def compare_modes(cfg: PipelineConfig) -> dict[str, LatencyResult]:
    """一次性跑出流式与非流式两种结果，方便对比。"""
    return {
        "streaming": simulate_streaming(cfg),
        "non_streaming": simulate_non_streaming(cfg),
    }


def speedup_ttfa(cfg: PipelineConfig) -> float:
    """流式相对非流式在**首响延迟**上的加速比（非流式TTFA / 流式TTFA）。

    >1 表示流式更快。这个数字往往非常大（几倍到十几倍），是流式最有说服力的卖点。
    """
    r = compare_modes(cfg)
    s = r["streaming"].ttfa_ms
    ns = r["non_streaming"].ttfa_ms
    return ns / s if s > 0 else float("inf")


if __name__ == "__main__":
    # 快速自检：直接 python latency_budget.py 会打印一份延迟报告
    # ⚠️ Windows 控制台默认 GBK，直接 print emoji/中文会崩，强制 stdout 用 utf-8
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cfg = PipelineConfig()
    res = compare_modes(cfg)
    print("=== 实时语音流水线延迟预算 ===")
    print(f"输入: 用户说话 {cfg.user_speech_ms:.0f}ms, "
          f"prompt {cfg.prompt_tokens} tok, 回答 {cfg.response_tokens} tok")
    for mode in ("streaming", "non_streaming"):
        r = res[mode]
        print(f"\n[{mode}] 首响 TTFA={r.ttfa_ms:.1f}ms  端到端 E2E={r.e2e_ms:.1f}ms")
    print(f"\n流式首响加速比: {speedup_ttfa(cfg):.1f}x")
    bc = check_budget(cfg, target_ttfa_ms=500.0)
    print(f"\n预算检查 (目标 TTFA<500ms): "
          f"{'达标 ✅' if bc.within_budget else '超标 ❌'}, "
          f"实测 {bc.actual_ttfa_ms:.1f}ms, 余量 {bc.headroom_ms:.1f}ms")
