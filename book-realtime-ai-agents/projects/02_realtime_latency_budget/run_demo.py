# -*- coding: utf-8 -*-
"""
延迟预算演示 (Latency Budget Demo)
==================================

跑一遍实时语音流水线的延迟仿真，并画三张图：

  1. latency_waterfall.png   —— 流式 vs 非流式 的延迟「瀑布图」，
     直观看到非流式的首个音频要等多久、流式凭什么快。
  2. ttfa_breakdown.png      —— 流式首响 TTFA 的预算分解（ASR/prefill/TTS 各吃多少）。
  3. prompt_sweep.png        —— prompt 长度扫描：prompt 越长，TTFA 怎么涨、预算何时被顶破。

运行:
    python run_demo.py

⚠️ 全程离线、纯 CPU，不联网、不下模型。matplotlib 用 Agg 后端，不弹窗，直接存 png。
"""

import sys

import matplotlib

matplotlib.use("Agg")  # ⚠️ 无界面后端，服务器/CI 也能出图，必须在 pyplot 之前设
import matplotlib.pyplot as plt

# 中文字体 + 负号正常显示（Windows 上 YaHei，Linux 回退 SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from latency_budget import (
    PipelineConfig,
    check_budget,
    compare_modes,
    simulate_streaming,
    speedup_ttfa,
    ttfa_breakdown,
)

# Windows 控制台默认 GBK，强制 utf-8 才能打印中文与 emoji
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# 每一级用固定颜色，三张图保持一致，便于对照
STAGE_COLORS = {
    "ASR 收尾": "#4C72B0",
    "LLM prefill": "#DD8452",
    "LLM prefill(首token)": "#DD8452",
    "LLM decode(整句)": "#C44E52",
    "TTS 整段合成": "#8172B3",
    "TTS 首块合成": "#937860",
    "音频播放": "#55A868",
    "首响后并行(计算/播放)": "#55A868",
}


def _color(name: str) -> str:
    return STAGE_COLORS.get(name, "#999999")


def plot_waterfall(cfg: PipelineConfig, path: str) -> None:
    """画流式 vs 非流式的延迟瀑布图。

    每一行是一种模式，横轴是墙钟时间（ms），每个色块是一级 stage，
    从左到右按发生顺序拼接；竖虚线标出「第一个音频（TTFA）」的时刻。
    一眼就能看出：非流式的 TTFA 竖线被推到很右边，流式的靠得很左。
    """
    res = compare_modes(cfg)
    modes = ["non_streaming", "streaming"]
    labels = {"non_streaming": "非流式\n(non-streaming)", "streaming": "流式\n(streaming)"}

    fig, ax = plt.subplots(figsize=(12, 4.2))
    bar_h = 0.55

    seen_legend: set[str] = set()
    for row, mode in enumerate(modes):
        r = res[mode]
        for name, start, dur in r.stages:
            legend_label = name if name not in seen_legend else None
            if legend_label is not None:
                seen_legend.add(name)
            ax.barh(
                row,
                dur,
                left=start,
                height=bar_h,
                color=_color(name),
                edgecolor="white",
                linewidth=1.2,
                label=legend_label,
            )
            # 在够宽的块上标注毫秒
            if dur > 90:
                ax.text(
                    start + dur / 2, row, f"{dur:.0f}",
                    va="center", ha="center", color="white", fontsize=9, fontweight="bold",
                )
        # TTFA 竖线：第一个音频出现的时刻
        ax.axvline(r.ttfa_ms, color="#333333", linestyle="--", linewidth=1.3, alpha=0.0)
        ax.annotate(
            f"首个音频 TTFA\n{r.ttfa_ms:.0f} ms",
            xy=(r.ttfa_ms, row),
            xytext=(r.ttfa_ms, row + 0.42),
            ha="center", va="bottom", fontsize=9, color="#B00020", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B00020", linewidth=1.4),
        )

    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([labels[m] for m in modes], fontsize=11)
    ax.set_xlabel("墙钟时间 wall-clock time (ms)，t=0 = 用户说完这句话", fontsize=11)
    sp = speedup_ttfa(cfg)
    ax.set_title(
        f"实时语音流水线 · 延迟瀑布图（流式 vs 非流式）\n"
        f"流式首响仅需等「首个 token」，把 TTFA 从 "
        f"{res['non_streaming'].ttfa_ms:.0f}ms 砍到 {res['streaming'].ttfa_ms:.0f}ms（{sp:.1f}× 更快）",
        fontsize=12, pad=14,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.6, len(modes) - 0.1 + 0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ttfa_breakdown(cfg: PipelineConfig, target_ttfa_ms: float, path: str) -> None:
    """画流式首响 TTFA 的预算分解（水平堆叠条 + 预算线）。"""
    parts = ttfa_breakdown(cfg)
    bc = check_budget(cfg, target_ttfa_ms)

    fig, ax = plt.subplots(figsize=(11, 2.8))
    left = 0.0
    for name, ms in parts:
        ax.barh(0, ms, left=left, height=0.5, color=_color(name),
                edgecolor="white", linewidth=1.5, label=f"{name} ({ms:.0f}ms)")
        ax.text(left + ms / 2, 0, f"{ms:.0f}", va="center", ha="center",
                color="white", fontsize=10, fontweight="bold")
        left += ms

    # 预算线
    ax.axvline(target_ttfa_ms, color="#B00020", linestyle="--", linewidth=2)
    ax.text(target_ttfa_ms, 0.32, f"预算上限 {target_ttfa_ms:.0f}ms",
            color="#B00020", ha="center", fontsize=10, fontweight="bold")

    # ⚠️ 图上不用 emoji：YaHei 没有 ✅/❌ 字形会显示成豆腐块，改用纯文字
    status = "达标" if bc.within_budget else "超标"
    ax.set_title(
        f"流式首响 TTFA 预算分解 —— 实测 {bc.actual_ttfa_ms:.0f}ms，"
        f"余量 {bc.headroom_ms:+.0f}ms（{status}）",
        fontsize=12, pad=10,
    )
    ax.set_yticks([])
    ax.set_xlabel("TTFA 组成 (ms)", fontsize=11)
    ax.set_xlim(0, max(target_ttfa_ms, bc.actual_ttfa_ms) * 1.15)
    ax.legend(loc="lower right", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_prompt_sweep(target_ttfa_ms: float, path: str) -> None:
    """扫描 prompt 长度，画 TTFA 随 prompt token 数增长的曲线 + 预算线。

    这张图讲一个残酷的工程真相：**你的 system prompt 和历史每长一点，
    首响就慢一点**。曲线穿过预算线的那个点，就是「prompt 预算上限」。
    """
    prompt_sizes = list(range(0, 1201, 50))
    ttfas = []
    for p in prompt_sizes:
        cfg = PipelineConfig(prompt_tokens=p)
        ttfas.append(simulate_streaming(cfg).ttfa_ms)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(prompt_sizes, ttfas, marker="o", markersize=3, color="#4C72B0",
            linewidth=2, label="流式首响 TTFA")
    ax.axhline(target_ttfa_ms, color="#B00020", linestyle="--", linewidth=2,
               label=f"预算上限 {target_ttfa_ms:.0f}ms")

    # 找到第一个越界的 prompt 长度并标注
    breach = next((p for p, t in zip(prompt_sizes, ttfas) if t > target_ttfa_ms), None)
    if breach is not None:
        ax.axvline(breach, color="#DD8452", linestyle=":", linewidth=1.8)
        ax.annotate(
            f"prompt≈{breach} tok 起\n首响超预算",
            xy=(breach, target_ttfa_ms),
            xytext=(breach + 60, target_ttfa_ms - 120),
            fontsize=10, color="#DD8452", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#DD8452"),
        )

    ax.set_xlabel("prompt 长度 (token 数：system + 历史 + 用户当前话)", fontsize=11)
    ax.set_ylabel("流式首响 TTFA (ms)", fontsize=11)
    ax.set_title("prompt 越长，首响越慢 —— prefill 是 compute-bound 的\n"
                 "（裁剪历史 / 精简 system prompt 是降低实时延迟的头号手段）",
                 fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = PipelineConfig()
    target = 500.0  # 实时对话常见 SLO：首响 < 500ms

    print("=" * 60)
    print("实时交互延迟预算 · 演示")
    print("=" * 60)
    print(f"输入规模: 用户说话 {cfg.user_speech_ms:.0f}ms | "
          f"prompt {cfg.prompt_tokens} tok | 回答 {cfg.response_tokens} tok")
    print("-" * 60)

    res = compare_modes(cfg)
    for mode in ("non_streaming", "streaming"):
        r = res[mode]
        print(f"[{mode:>14}] 首响 TTFA = {r.ttfa_ms:7.1f} ms | "
              f"端到端 E2E = {r.e2e_ms:7.1f} ms")
    print(f"\n>>> 流式首响加速比: {speedup_ttfa(cfg):.1f}×")

    bc = check_budget(cfg, target)
    print(f"\n预算检查 (目标 TTFA < {target:.0f}ms): "
          f"{'达标 ✅' if bc.within_budget else '超标 ❌'} | "
          f"实测 {bc.actual_ttfa_ms:.1f}ms | 余量 {bc.headroom_ms:+.1f}ms")
    print("  TTFA 分解:")
    for name, ms, pct in bc.breakdown:
        print(f"    - {name:<14}: {ms:6.1f} ms  ({pct:4.1f}%)")

    # 出图
    print("-" * 60)
    plot_waterfall(cfg, "latency_waterfall.png")
    print("已保存: latency_waterfall.png  (流式 vs 非流式 瀑布图)")
    plot_ttfa_breakdown(cfg, target, "ttfa_breakdown.png")
    print("已保存: ttfa_breakdown.png     (首响预算分解)")
    plot_prompt_sweep(target, "prompt_sweep.png")
    print("已保存: prompt_sweep.png       (prompt 长度扫描)")
    print("=" * 60)
    print("完成 ✅  用图片查看器打开 png 即可。")


if __name__ == "__main__":
    main()
