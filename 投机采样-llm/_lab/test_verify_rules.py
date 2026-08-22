"""test_verify_rules.py —— 给自检器**自己**上测试。

为什么需要它：铁律一/二的体检里加了五条豁免（verify.py 里的 (a)–(e)）。
豁免每放宽一次，就多一次"为了消 WARN 而把真违规一起放过"的风险。
所以本文件用一张**正例 / 对照组**表把两边都钉住：

  - 正例（must be flagged）：真裸奔的加速比、真没标口径的"无损"，一条都不许漏；
  - 对照组（must be exempt）：已经人工复核确认的假阳性，一条都不许再报。

这些句子全部改写自本库真实出现过的行（见 _meta/建库审计.md 的三分类记录）。
"""
from __future__ import annotations

import pytest

import verify


# ---- 铁律一：没标口径的"无损" -------------------------------------------------

LOSSLESS_MUST_FLAG = [
    "这套判据是无损的，可以放心上线。",
    '我们的实现是"无损"的，和不开投机没区别。',      # 加引号≠提及：这仍是断言
    "[[04-拒绝采样修正-无损性的完整证明]] 已经说明这是无损的。",  # 有双链但无指路动词
    "把接受判据放宽以后仍然是无损的。",
]

LOSSLESS_MUST_EXEMPT = [
    "本篇不重复无损性证明（在 [[04-拒绝采样修正-无损性的完整证明]]）。",   # (d) 指路句
    '- "无损"措辞冲突的完整辨析：[[07-无损的三种口径-分布无损不等于结果相同]]',  # (d)+(e)
    '把"无损"讲成"输出完全一样"是最高频的错。',                       # (e) use–mention
    '为什么并行验证几乎免费、"无损"到底保证了什么。',                   # (e)
]


def _lossless_flagged(raw: str) -> bool:
    """复刻 check_rules 里铁律一的判定（不含上下文窗口，只判这一行本身）。"""
    line = verify.strip_for_rules(raw)
    if not verify.LOSSLESS_PAT.search(line):
        return False
    if verify.CALIBER_EXEMPT.search(line):
        return False
    if verify.caliber_exempt_line(raw, line):
        return False
    return not any(t in line for t in verify.CALIBER_TOKENS)


@pytest.mark.parametrize("raw", LOSSLESS_MUST_FLAG)
def test_lossless_without_caliber_is_flagged(raw):
    assert _lossless_flagged(raw), "铁律一漏抓：%s" % raw


@pytest.mark.parametrize("raw", LOSSLESS_MUST_EXEMPT)
def test_lossless_pointer_and_mention_are_exempt(raw):
    assert not _lossless_flagged(raw), "铁律一误报：%s" % raw


def test_caliber_marked_line_is_not_flagged():
    assert not _lossless_flagged("这套判据给的是 L1 分布无损，不是结果相同。")


# ---- 铁律二：裸奔的加速比 -----------------------------------------------------

SPEEDUP_MUST_FLAG = [
    "实测端到端加速 3.2x 以上，非常划算。",
    "理想加速比从 1.38 变成 6.37，差 4.6 倍。",        # 有"差"但同行有"加速比"→不豁免
    "换个草稿模型后提速 2.3 倍。",
    r"$\alpha$ 从 0.6 提到 0.8 之后是 2.3x 的水平。",   # 倍数旁没有非速度主语
    "平均接受 token 数从 1.4 涨到 7.3，差 5.2 倍。",    # 接受长度也归铁律二管
    "12% 相对于 EAGLE 那条线 4.6 倍的差距。",           # "差"在数字右侧，不算主语线索
]

SPEEDUP_MUST_EXEMPT = [
    "草稿头的参数量只有骨干的 10 倍不到。",                       # (c) 参数量
    "训练用 4×A100 (40G)，模型是 Mixtral 8×7B。",              # (a) 乘号
    "41×8 的命中矩阵。",                                       # (a) 乘号
    r"$\beta$ 在 0.176 到 0.449 之间变化（差 2.5 倍），公式仍准。",  # (c) β 的波动
    "NVIDIA 3.6x 博客的标题与正文自相矛盾。",                     # (b) 材料代称
    "头的 lr 取骨干的 4 倍。",                                  # (c) 学习率
    "相对 EAGLE-1 约 8 倍数据量。",                             # (c) 数据量
    "正确值是 70 272 字节（原值高估 1.78 倍）。",                  # (c) 字节数
]


@pytest.mark.parametrize("raw", SPEEDUP_MUST_FLAG)
def test_real_speedup_number_is_detected(raw):
    assert verify.speedup_matches(verify.strip_for_rules(raw)), "铁律二漏抓：%s" % raw


@pytest.mark.parametrize("raw", SPEEDUP_MUST_EXEMPT)
def test_non_speedup_multiplier_is_exempt(raw):
    assert not verify.speedup_matches(verify.strip_for_rules(raw)), "铁律二误报：%s" % raw


def test_qps_counts_as_load_caliber():
    """给了 QPS 就是给了负载口径，与"并发"同类，不算裸奔。"""
    assert "QPS" in verify.BATCH_TOKENS


# ---- 铁律二的表格豁免：一个数字的主语在**它那一列的表头**里 --------------------
#
# 起因：第 21 篇 §6.3「等收益线」那张表，第三列整列是"草稿成本 $c$ 相对基准的倍数"，
# 主语是成本不是速度，但表格行本身只剩 `| 0.80 | 0.0442 | **2.21×** | 8 | 3.1989 |`，
# 按行看不出主语。豁免的判据与护栏见 verify.py 里 NON_SPEED_LEFT 上方的注释。

COST_TABLE = [
    "### 6.3 等收益线",
    "",
    "| 目标 $\\alpha$ | 允许的最大 $c$ | 相对基准的**成本倍数** | 该点的 $\\gamma$ | 达成的加速比 |",
    "|---|---|---|---|---|",
    "| 0.80 | 0.0442 | **2.21×** | 8 | 3.1989 |",
    "| 0.85 | 0.0752 | **3.76×** | 8 | 3.1989 |",
]

SPEED_TABLE = [
    "### 某篇 §1 实测表",
    "",
    "| 方法 | 加速比 |",
    "|---|---|",
    "| EAGLE-3 | **2.83×** |",
    "| DFlash | 1.6× |",
]

# 列头点名了非速度主语，**但同一行出现了速度词** —— 行级护栏优先，不许豁免。
MIXED_TABLE = [
    "### 某篇 §2 混合表",
    "",
    "| 方法 | 相对基准的成本倍数 | 端到端加速 |",
    "|---|---|---|",
    "| A | **2.21×** | 快 |",
]


def _row_flagged(lines: list, idx: int) -> bool:
    hdr = verify.table_header_cells(lines, idx)
    return bool(verify.speedup_matches(
        verify.strip_for_rules(lines[idx]), hdr, lines[idx]))


@pytest.mark.parametrize("idx", [4, 5])
def test_cost_multiple_column_is_exempt(idx):
    """列头写明是"成本倍数"，整列主语不是速度 -> 豁免。"""
    assert not _row_flagged(COST_TABLE, idx), COST_TABLE[idx]


@pytest.mark.parametrize("idx", [4, 5])
def test_speedup_column_is_still_flagged(idx):
    """列头就叫"加速比"，表头里的速度词让豁免失效 -> 必须仍被抓。"""
    assert _row_flagged(SPEED_TABLE, idx), SPEED_TABLE[idx]


def test_speed_word_in_row_defeats_column_exemption():
    """同一行只要出现速度词，列头的非速度主语也救不了它。"""
    assert _row_flagged(MIXED_TABLE, 4)


def test_table_header_lookup_stops_at_blank_line():
    """向上找表头时遇到空行/标题就放弃，不会张冠李戴上一张表的表头。"""
    lines = ["| a | b |", "|---|---|", "| 1 | 2 |", "", "散文行 加速 3.2×。"]
    assert verify.table_header_cells(lines, 4) == []


def test_strip_for_rules_preserves_offsets():
    """列号是按原始行里的 `|` 数出来的，所以 strip 必须**等长**。"""
    raw = "| [[04-拒绝采样修正-无损性的完整证明]] | `code` | 2.21× |"
    assert len(verify.strip_for_rules(raw)) == len(raw)


# ---- 全库现状：这两条铁律必须保持 0 -------------------------------------------

def test_repo_has_no_rule_warnings():
    warn_lossless, warn_speedup, _ = verify.check_rules(verbose=False)
    assert warn_lossless == [], "铁律一新增 WARN：%s" % warn_lossless[:5]
    assert warn_speedup == [], "铁律二新增 WARN：%s" % warn_speedup[:5]
