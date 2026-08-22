"""
verify.py —— 全库自检：把五条铁律变成可执行的检查，而不是靠自觉。

检查项
  --links   双链死链：正文里的 [[X]] 必须指向 §5 清单里真实存在的文件
  --tests   测试虚引：正文里引用的 `_lab/test_x.py::test_y` 必须真实存在（零虚引）
  --rules   铁律体检：
              铁律一 每处"无损"附近必须标 L1/L2/L3 口径
              铁律二 加速比数字附近必须出现 batch（口径不许裸奔）
              铁律三 每篇必须有失效条件小节
              铁律五 历史篇的每个工作要有年月与 arXiv/URL
  --struct  结构体检：每篇必须有「本篇验证」与「本篇来源」小节；清单齐不齐
  --consistency 跨篇一致性：同一 arXiv 编号在不同篇目里数字打架、缺年月、无主语句式
  --stats   统计：篇数、行数、双链数、来源 URL 数、测试引用数
  --all     全部

退出码：有 ERROR 则非 0（可以挂 CI / 收尾时一把梭）。

自检器自身的坑（本仓库踩过，写在这里防止重犯）：
  1) 把 `test_x.py` 去掉 .py 当成函数名去查 -> 一片假失败。本文件用 `::` 严格切分。
  2) 正则不覆盖表格单元里的写法（如 `| **2.3x** |`）-> 漏检。本文件按行扫描并剥掉表格分隔符。
  3) 只在一个样本上判统计命题 -> 结论随机。本文件所有"比例"类判断都给出分母。
  4) **假阳性要留档、豁免要写理由**：见下面 (a)–(e) 五条豁免的注释，每条都注明
     它豁免的是哪一类句子、以及为什么那类句子不适用该铁律。反例测试见
     `_lab/test_verify_rules.py`（正例必须仍被抓，对照组必须被豁免）。

2026-08-22 **已修**的一个假阴性（留档，因为它是本库最贵的一课）：
  SPEEDUP_PAT 原来写作 `[×xX]\b`，对 `×` 这个符号要求右邻是词字符，
  于是 `加速 3.2×，` `1.9×（最差）` `2.4× vs` 这类**× 后面跟标点或空格**的写法
  一律漏检 —— 而那正是本库表格里最常见的写法（全库 894 处 `N.N×` 只命中 1 处，漏 99.9%）。
  修法是三件事一起做，缺一不可：
    ① 去掉 `\b`（乘号形态改由 MULT_NOT_TIMES 单独兜）；
    ② 口径窗口从固定 ±6 行改成**所在小节**（口径常写在小节开头或表注里）；
    ③ 逐条复核修完后暴出的 27 处，A 类补口径 / B 类加豁免 / C 类留 WARN（见 _meta/建库审计.md）。
  **教训**：在这之前 `test_repo_has_no_rule_warnings` 一直是绿的 —— 它守的是一个
  **本身就漏检的检查器**。"全绿"只能证明检查器没报警，不能证明库是干净的；
  所以本文件的每条豁免都必须在 test_verify_rules.py 里同时钉住"正例仍被抓"。
r"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "_lab"

# --- 铁律一：口径标记 --------------------------------------------------------
CALIBER_TOKENS = ("L1", "L2", "L3", "分布无损", "贪心等价", "greedy-exact", "近似口径")
LOSSLESS_PAT = re.compile(r"无损")
# 允许的例外：在讲"口径"本身、在标题里、在引用别人说法并当场纠正的句子里
CALIBER_EXEMPT = re.compile(r"口径|误解|错法|纠正|社区|所谓|铁律|本库|三种|标注")

# 铁律一的两类**已确认假阳性**（逐条人工复核过，留档见 _meta/建库审计.md）。
# 豁免的原则只有一条：**那句话根本不是在下"它是无损的"这个断言**。
#
# (d) 指路句：句子在指**别的篇目**（"本篇不重复无损性证明（在 [[04-…]]）"、
#     "'无损'措辞冲突的完整辨析：[[07-…]]"）。它指代的是篇目内容，不是断言，
#     "这里的无损是 L1 还是 L3"对它不适用。判据要求**同时**满足两条：
#     ① 原始行里有双链；② 有指路动词。少一条就照常检查 —— 例如第 16 篇
#     "于是 [[04-…]] 的对齐前提被破坏，无损性悄悄失效"有双链但无指路动词，
#     它是断言，必须被抓（反例测试确认仍然抓得到）。
POINTER_VERB = re.compile(r"不重复|详见|参见|另见|完整辨析|指路|见本库")
# (e) 引号里的**词本身**：把"无损"讲成…、"无损"措辞冲突、"无损"到底保证了什么。
#     use–mention 里的 mention：在谈这个词被怎么用，不是在用它下断言。
#     两个条件缺一不可：引号内**只有"无损(性)"二字**、且句中有讨论用词 ——
#     这样 `我们的实现是"无损"的` 这种加了引号的真断言不会被放过。
MENTION_PAT = re.compile(r"[\"“”「『]无损性?[\"“”」』]")
MENTION_VERB = re.compile(r"讲成|说成|措辞|到底|这个词|叫做|称为|写成|读成")

# --- 铁律二：加速比 ----------------------------------------------------------
# 2026-08-22 修一个**假阴性**（对抗审稿实测）：原来是 `[×xX]\b`，要求 × 右邻是词字符，
# 于是 `加速 3.2×，`、`1.9×（最差）`、`2.4× vs` 这类 **× 后跟标点或空格**的写法全部漏检 ——
# 而那正是本库表格里最常见的写法（全库 894 处 `N.N×` 只命中 1 处，漏 99.9%）。
# 现在不再要求右邻是词字符；乘号形态（4×A100、8×7B）由 MULT_NOT_TIMES 单独兜住。
SPEEDUP_PAT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:[×xX]|倍(?!数|率))")
# QPS 与"并发"是同一件事的两种写法：给了 QPS 的来源（如 vLLM 官方博客）
# 就是给了负载口径，不算裸奔。
BATCH_TOKENS = ("batch", "bs=", "BS=", "批大小", "并发", "bs ", "Batch", "QPS", "qps")

# 铁律二的三类**已确认假阳性**（同样逐条复核并留档）。豁免的原则只有一条：
# **那个倍数的主语根本不是"速度"**，所以"报加速比必须给 batch"对它不适用。
#
# (a) `N×M` 里的 × 是**乘号**不是"倍"：4×A100（卡数）、Mixtral 8×7B（专家数×规模）、
#     41×8 命中矩阵（矩阵尺寸）。判据：× 右边紧跟数字或型号（大写字母开头）。
#     真加速比的右邻不是数字也不是型号（"2.8x 的"、"1.25X"），不会被误伤。
#     （"1.5×，"这种右邻是标点的，是上面那条**已知假阴性**，SPEEDUP_PAT 根本没匹配到，
#     轮不到这条豁免。）
MULT_NOT_TIMES = re.compile(r"[×xX]\s*(?:\d|[A-Z])")
# (b) 材料代称：按标题里的数字给一份材料起名，如"NVIDIA 3.6x 博客"。
#     这是在**指代一份材料**，不是本库在报一个加速比（该文的口径评点另有专节）。
MATERIAL_ALIAS = re.compile(r"[×xX]\s*(?:那篇|这篇)?\s*(?:博客|文章|一文|长文|帖|blog|Blog)")
# (c) 倍数的主语是**非速度量**：β/α 的波动、参数量、数据量、学习率、显存、
#     成本预算、分子分母。铁律二管的是"多快"，不管"多大/多贵/多分散"。
#     判据要求这些词紧贴倍数，**并且**同一行不出现任何速度类词 —— 只要出现就
#     一律不豁免，宁可多报。所以 "理想加速比…差 4.6 倍"（同行有"加速比"）仍会被抓。
#     左右两侧用的词表不同，这是必须的：中文里主语在倍数**左边**（"参数量…10 倍"、
#     "差 2.5 倍"），只有名词能跟在倍数**右边**（"8 倍数据量"）。
#     若右侧也认"差"，"…那条线 4.6 倍的差距"就会被误豁免（反例测试钉住了这一条）。
#     2026-08-22 补一条**表格内的主语位置**：在 markdown 表里，一个数字的主语不在同一行的
#     左边，而在**它那一列的表头**（如「相对基准的**成本倍数**」那一列，整列都是草稿成本 $c$
#     的倍数，不是速度）。所以 (c) 的左侧线索里额外算上**匹配所在列的表头单元格**。
#     两条护栏保证它不会放过真违规：
#       ① 只取**匹配所在那一列**的表头，不取整行表头 —— 否则同表里只要有一列叫"加速比"，
#          整张表都会被那三个字牵着走（那正好是反过来的错）；
#       ② 表头单元格里只要出现任何速度类词，就**一律不豁免**（与同行的判定同权），
#          所以 `| 加速比 | 2.8× |` 这种列头永远抓得到。
NON_SPEED_LEFT = ("差", "相差", "波动", "乘了", "vs", "参数量", "数据量", "显存",
                  "预算", "成本", "lr", "学习率", "分母", "分子", "字节")
NON_SPEED_RIGHT = ("参数量", "数据量", "显存", "预算", "学习率", "字节")
SPEED_WORDS = ("加速", "提速", "吞吐", "throughput", "tok/s", "tokens/s", "token/s",
               "TPOT", "TTFT", "延迟", "latency", "speedup", "speed",
               "接受长度", "接受 token", "acceptance", "快", "慢",
               "wall-clock", "walltime", "ms/t")


HEADING = re.compile(r"^#{1,6}\s")


def section_of(lines: list, idx: int) -> list:
    """第 idx 行（0-based）所在的小节：上一个 markdown 标题到下一个标题之间。

    铁律二用它当口径窗口 —— 口径声明常写在小节开头（"口径：batch=…"）或表注里，
    固定 ±N 行会把长表的中后部全判成"裸奔"。按小节取才符合作者的书写习惯。
    """
    start = 0
    for j in range(idx, -1, -1):
        if HEADING.match(lines[j]):
            start = j
            break
    end = len(lines)
    for j in range(idx + 1, len(lines)):
        if HEADING.match(lines[j]):
            end = j
            break
    return lines[start:end]


def caliber_exempt_line(raw: str, line: str) -> bool:
    """铁律一：这一行的"无损"是不是**不在下断言**（(d) 指路句 / (e) 引号提及）。"""
    if "[[" in raw and POINTER_VERB.search(line):
        return True
    if MENTION_PAT.search(line) and MENTION_VERB.search(line):
        return True
    return False


TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$")


def _split_cells(row: str) -> list:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def table_header_cells(lines: list, idx: int) -> list:
    """第 idx 行（0-based）若是 markdown 表的**数据行**，返回表头各单元格；否则 []。

    表头 = 分隔行 `|---|---|` 的上一行。向上查找时遇到空行或标题就放弃
    （说明已经离开这张表），因此不会把上一张表的表头张冠李戴。
    """
    if "|" not in lines[idx] or TABLE_SEP.match(lines[idx]):
        return []
    for j in range(idx - 1, max(-1, idx - 200), -1):
        ln = lines[j]
        if not ln.strip() or HEADING.match(ln):
            return []
        if "|" in ln and TABLE_SEP.match(ln):
            if j - 1 >= 0 and "|" in lines[j - 1]:
                return _split_cells(lines[j - 1])
            return []
    return []


def _column_of(line: str, pos: int) -> int:
    """匹配落在表格的第几列（0-based）。要求 line 与原始行**等长**，
    这正是 strip_for_rules 用等长空白替换的理由。"""
    return line[:pos].count("|") - 1


def speedup_matches(line: str, headers: list | None = None,
                    raw: str | None = None) -> list:
    """铁律二：返回本行里**真正算加速比**的倍数（剔除 (a)(b)(c) 三类假阳性）。

    headers 给的是本行所在表的表头单元格列表（见 table_header_cells），
    raw 是**未经 strip_for_rules 的原始行**（列分隔符 `|` 只在它里面还在）。
    两者都给了，(c) 的"主语"判定才会额外看**匹配所在那一列的表头**；
    否则退化成原来的纯按行判定，行为与加这条豁免之前完全一致。
    """
    out = []
    for m in SPEEDUP_PAT.finditer(line):
        rest = line[m.end() - 1:]              # 从 ×/倍 这个字符本身起算
        if MULT_NOT_TIMES.match(rest):         # (a) 乘号
            continue
        if MATERIAL_ALIAS.match(rest):         # (b) 材料代称
            continue
        head = ""
        if headers and raw is not None and len(raw) == len(line):
            ci = _column_of(raw, m.start())
            if 0 <= ci < len(headers):
                head = headers[ci]
        left, right = line[max(0, m.start() - 20):m.start()], line[m.end():m.end() + 12]
        if ((any(c in head + left for c in NON_SPEED_LEFT)   # (c) 非速度主语
             or any(c in right for c in NON_SPEED_RIGHT))
                and not any(w in line + " " + head for w in SPEED_WORDS)):
            continue
        out.append(m)
    return out

# --- 铁律三/结构 -------------------------------------------------------------
REQUIRED_SECTIONS = ("本篇验证", "本篇来源")
FAILURE_TOKENS = ("失效条件", "什么时候不", "什么时候反而", "负收益", "什么时候是错的",
                  "不适用", "何时不该")

TEST_REF_PAT = re.compile(r"`?_lab/(test_[A-Za-z0-9_]+\.py)::(test_[A-Za-z0-9_]+)`?")
LINK_PAT = re.compile(r"\[\[([^\]|#]+)")
URL_PAT = re.compile(r"https?://[^\s)>\]，。；、]+")


def chapters() -> list[Path]:
    return sorted(p for p in ROOT.glob("*.md") if re.match(r"^\d\d-", p.name))


def all_md() -> list[Path]:
    return sorted(ROOT.glob("*.md"))


def stems() -> set[str]:
    return {p.stem for p in all_md()}


def strip_code_blocks(text: str) -> str:
    """去掉 ``` 围栏代码块，避免把代码里的东西当正文检查。"""
    return re.sub(r"```.*?```", "", text, flags=re.S)


def strip_for_rules(line: str) -> str:
    """铁律体检专用：把不该被当成"正文表述"的部分挖掉。

    踩过的坑：双链的**文件名**里含"无损"二字（如 [[04-拒绝采样修正-无损性的完整证明]]），
    会被铁律一的检查大量误判成"未标口径的无损"。行内代码同理。
    这是自检器自身的假阳性，不是被检对象的问题 —— 修检查器，不是修正文。

    **等长替换**：挖掉的部分用同样长度的空白填回，而不是塌缩成一个空格。
    这样处理后的行与原始行**逐字符对齐**，铁律二才能把一个匹配的偏移量换算回
    "它在表格的第几列"（见 _column_of / table_header_cells）。
    """
    blank = lambda m: " " * len(m.group(0))
    line = re.sub(r"\[\[[^\]]*\]\]", blank, line)   # 双链整体挖掉
    line = re.sub(r"`[^`]*`", blank, line)          # 行内代码挖掉
    line = line.replace("|", " ")                   # 表格分隔符
    return line


# ---------------------------------------------------------------- 双链

def _link_body(p) -> str:
    """双链检查用的正文：去掉围栏代码块**与行内代码**。

    行内代码里的 `[[...]]` 是在演示语法（如写作规范、README、附录篇），
    不是真链接；不剥掉会被判成死链（踩过）。
    """
    body = strip_code_blocks(p.read_text(encoding="utf-8"))
    return re.sub(r"`[^`]*`", " ", body)


def check_links(verbose=True):
    known, errors = stems(), []
    for p in all_md():
        body = _link_body(p)
        for m in LINK_PAT.finditer(body):
            target = m.group(1).strip()
            if target not in known:
                errors.append((p.name, target))
    n = sum(len(LINK_PAT.findall(_link_body(p))) for p in all_md())
    if verbose:
        print("[双链] 共 %d 条，死链 %d 条" % (n, len(errors)))
        for f, t in errors:
            print("  ERROR 死链  %s -> [[%s]]" % (f, t))
    return errors, n


# ---------------------------------------------------------------- 测试虚引

def collect_test_functions() -> dict[str, set[str]]:
    out = {}
    for f in LAB.glob("test_*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            print("  ERROR 测试文件语法错误 %s: %s" % (f.name, e))
            out[f.name] = set()
            continue
        out[f.name] = {n.name for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    return out


def check_tests(verbose=True):
    funcs, errors, total = collect_test_functions(), [], 0
    for p in all_md():
        body = p.read_text(encoding="utf-8")
        for m in TEST_REF_PAT.finditer(body):
            total += 1
            fname, tname = m.group(1), m.group(2)
            if fname not in funcs:
                errors.append((p.name, fname, tname, "文件不存在"))
            elif tname not in funcs[fname]:
                errors.append((p.name, fname, tname, "函数不存在"))
    if verbose:
        n_have = sum(len(v) for v in funcs.values())
        print("[测试引用] 正文引用 %d 处，_lab 实有测试函数 %d 个，虚引 %d 处"
              % (total, n_have, len(errors)))
        for f, fn, tn, why in errors:
            print("  ERROR 虚引  %s -> %s::%s (%s)" % (f, fn, tn, why))
    return errors, total


# ---------------------------------------------------------------- 结构

def check_struct(verbose=True):
    errors = []
    for p in chapters():
        body = p.read_text(encoding="utf-8")
        for sec in REQUIRED_SECTIONS:
            if sec not in body:
                errors.append((p.name, "缺小节：" + sec))
        if not any(t in body for t in FAILURE_TOKENS):
            errors.append((p.name, "缺失效条件（铁律三）"))
    if verbose:
        print("[结构] 章节 %d 篇，结构问题 %d 处" % (len(chapters()), len(errors)))
        for f, why in errors:
            print("  ERROR 结构  %s：%s" % (f, why))
    return errors


def check_manifest(verbose=True):
    """对照 _meta/写作规范.md §5 的清单，看哪些篇目还没写。"""
    spec = (ROOT / "_meta" / "写作规范.md").read_text(encoding="utf-8")
    block = re.search(r"## 5\. 本库文件清单.*?```(.*?)```", spec, re.S)
    if not block:
        print("  WARN 没在写作规范里找到 §5 清单")
        return []
    want = [ln.strip() for ln in block.group(1).splitlines()
            if re.match(r"^\d\d-", ln.strip())]
    have = stems()
    missing = [w for w in want if w not in have]
    extra = [h for h in have if re.match(r"^\d\d-", h) and h not in want]
    if verbose:
        print("[清单] 应有 %d 篇，已有 %d 篇，缺 %d 篇" % (len(want), len(want) - len(missing), len(missing)))
        for m in missing:
            print("  TODO 未写  %s" % m)
        for e in extra:
            print("  WARN 清单外文件  %s" % e)
    return missing


# ---------------------------------------------------------------- 铁律

def check_rules(verbose=True):
    """铁律体检。

    只检查**解释性正文**：标题行、以及「本篇验证 / 本篇来源」两个元信息小节里
    的表述不参与铁律一检查 —— 那里的"无损"是在指代篇目或测试名，不是在下断言。
    这条豁免是踩过假阳性之后加的（见 strip_for_rules 的注释）。
    """
    warn_lossless, warn_speedup, errors = [], [], []
    for p in chapters():
        lines = strip_code_blocks(p.read_text(encoding="utf-8")).splitlines()
        meta_from = len(lines)
        for j, ln in enumerate(lines):
            if ln.startswith("###") and ("本篇验证" in ln or "本篇来源" in ln):
                meta_from = min(meta_from, j)
        for i, raw in enumerate(lines, 1):
            if raw.lstrip().startswith("#"):
                continue                      # 标题不参与
            in_meta = (i - 1) >= meta_from
            line = strip_for_rules(raw)
            # 铁律一
            if (not in_meta) and LOSSLESS_PAT.search(line) \
                    and not CALIBER_EXEMPT.search(line) \
                    and not caliber_exempt_line(raw, line):
                ctx = " ".join(strip_for_rules(x) for x in lines[max(0, i - 4):i + 3])
                if not any(t in ctx for t in CALIBER_TOKENS):
                    warn_lossless.append((p.name, i, raw.strip()[:70]))
            if speedup_matches(line, table_header_cells(lines, i - 1), raw):
                # 口径窗口 = **所在小节**（上一个标题到下一个标题），而不是固定 ±N 行。
                # 理由：口径声明通常写在小节开头或表注里，固定窗口会把整张长表判成裸奔。
                ctx = " ".join(strip_for_rules(x) for x in section_of(lines, i - 1))
                if not any(t in ctx for t in BATCH_TOKENS):
                    warn_speedup.append((p.name, i, raw.strip()[:70]))
    if verbose:
        print("[铁律一 无损口径] 未标口径的「无损」 %d 处" % len(warn_lossless))
        for f, i, s in warn_lossless[:25]:
            print("  WARN %s:%d  %s" % (f, i, s))
        print("[铁律二 加速比裸奔] 附近没提 batch 的倍数 %d 处" % len(warn_speedup))
        for f, i, s in warn_speedup[:25]:
            print("  WARN %s:%d  %s" % (f, i, s))
    return warn_lossless, warn_speedup, errors



# ---------------------------------------------------------------- 跨篇一致性

ARXIV_PAT = re.compile(r"(?:arXiv[:\s]*|arxiv\.org/abs/)(\d{4}\.\d{4,5})", re.I)
YM_NEAR = re.compile(r"(20\d\d)[-年/](\d{1,2})")
SPEEDUP_NEAR = re.compile(r"(\d+(?:\.\d+)?)\s*[×xX](?![\w])")
# 铁律五禁止的无主语句式
NO_SUBJECT = [
    "最近有研究表明", "最近有工作",
    "业界普遍认为", "大家普遍认为",
    "有研究表明", "有工作表明",
    "相关研究表明", "众所周知",
]


def check_consistency(verbose=True):
    """跨篇一致性审计。

    起因：本库建库时，两份调研笔记（RS-4 与 RS-5）对**同一篇论文**给出了互相矛盾的
    记录，是写第 25 篇的过程中人工发现的。这类问题必须由工具兜住，不能靠运气。

    三项检查：
      1) 同一个 arXiv 编号在不同篇目里被搭配了**不同的加速比数字** -> 需要人工核对口径
      2) 提到 arXiv 编号但附近没有年月 -> 违反铁律五
      3) 无主语句式（"最近有研究表明"等） -> 违反铁律五
    """
    files = all_md() + sorted((ROOT / "_research").glob("*.md"))
    by_arxiv: dict[str, dict[str, set]] = {}
    ym_seen: dict[tuple, bool] = {}
    no_ym, no_subj = [], []

    for p in files:
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, raw in enumerate(lines, 1):
            # 例外：正在**声明禁用**这些句式的行（规范/自检说明），不算违规。
            # 这是踩过的假阳性：RS-1 的免责声明里逐字列了禁用句式，被判成了违规。
            meta_line = any(k in raw for k in
                            ("禁止", "不使用", "不许",
                             "避免", "这类", "红线",
                             "铁律", "冗余句式"))
            if not meta_line:
                for ph in NO_SUBJECT:
                    if ph in raw:
                        no_subj.append((p.name, i, ph))
            for m in ARXIV_PAT.finditer(raw):
                aid = m.group(1)
                ctx = " ".join(lines[max(0, i - 2):i + 1])   # 窄窗口：同行 + 上一行，降噪
                by_arxiv.setdefault(aid, {}).setdefault(p.name, set())
                for sm in SPEEDUP_NEAR.finditer(ctx):
                    by_arxiv[aid][p.name].add(sm.group(1))
                # 铁律五按 (文件, 编号) 聚合：同一篇里只要**有一处**给了年月就算合规，
                # 后续行内再提不必重复标注。逐处计数会把交叉引用全判成违规（踩过）。
                dated = YM_NEAR.search(" ".join(lines[max(0, i - 3):i + 3])) is not None
                key = (p.name, aid)
                ym_seen[key] = ym_seen.get(key, False) or dated

    no_ym = [(f, 0, aid) for (f, aid), ok in sorted(ym_seen.items()) if not ok]

    conflicts = []
    for aid, per_file in by_arxiv.items():
        vals = {f: v for f, v in per_file.items() if v}
        if len(vals) >= 2:
            allv = set().union(*vals.values())
            # 只有当不同文件给出的数字集合互不相同时才提示
            if len(allv) > 1 and len({frozenset(v) for v in vals.values()}) > 1:
                conflicts.append((aid, vals))

    if verbose:
        print("[一致性] 出现过的 arXiv 编号 %d 个" % len(by_arxiv))
        print("[一致性] 同一编号在不同篇目里搭配了不同加速比数字：%d 处（需人工核对口径）"
              % len(conflicts))
        for aid, vals in conflicts[:12]:
            print("  CHECK arXiv:%s" % aid)
            for f, v in vals.items():
                print("        %-46s %s" % (f, sorted(v)))
        print("[铁律五] 整篇从未给年月的 arXiv 编号：%d 个（按 篇目x编号 聚合）" % len(no_ym))
        for f, _, aid in no_ym[:15]:
            print("  WARN %-46s arXiv:%s" % (f, aid))
        print("[铁律五] 无主语句式：%d 处" % len(no_subj))
        for f, i, ph in no_subj[:12]:
            print("  ERROR %s:%d  出现禁用表述" % (f, i))
    return conflicts, no_ym, no_subj


# ---------------------------------------------------------------- 统计

def check_stats():
    md = all_md()
    total_lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in md)
    links = sum(len(LINK_PAT.findall(strip_code_blocks(p.read_text(encoding="utf-8"))))
                for p in md)
    urls = set()
    for p in md:
        urls.update(URL_PAT.findall(p.read_text(encoding="utf-8")))
    trefs = sum(len(TEST_REF_PAT.findall(p.read_text(encoding="utf-8"))) for p in md)
    funcs = collect_test_functions()
    labs = sorted(f.name for f in LAB.glob("*.py") if not f.name.startswith("test_"))
    print("=" * 66)
    print("统计")
    print("=" * 66)
    print("  Markdown 篇数        %d（其中编号章节 %d）" % (len(md), len(chapters())))
    print("  总行数               %d" % total_lines)
    print("  双链                 %d" % links)
    print("  去重后来源 URL       %d" % len(urls))
    print("  正文引用测试         %d 处" % trefs)
    print("  _lab 测试函数        %d 个（%d 个测试文件）"
          % (sum(len(v) for v in funcs.values()), len(funcs)))
    print("  _lab 实现模块        %d 个：%s" % (len(labs), ", ".join(labs)))


def main():
    ap = argparse.ArgumentParser()
    for f in ("links", "tests", "rules", "struct", "stats", "consistency", "all"):
        ap.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    if not any(vars(a).values()):
        a.all = True
    n_err = 0
    if a.all or a.links:
        e, _ = check_links()
        n_err += len(e)
    if a.all or a.tests:
        e, _ = check_tests()
        n_err += len(e)
    if a.all or a.struct:
        n_err += len(check_struct())
        check_manifest()
    if a.all or a.rules:
        check_rules()
    if a.all or a.consistency:
        _, _, ns = check_consistency()
        n_err += len(ns)
    if a.all or a.stats:
        check_stats()
    if n_err:
        print("\n=> %d 个 ERROR" % n_err)
    else:
        print("\n=> 无 ERROR")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
