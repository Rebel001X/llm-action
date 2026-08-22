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
"""
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

# --- 铁律二：加速比 ----------------------------------------------------------
SPEEDUP_PAT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:[×xX]\b|倍(?!数|率))")
BATCH_TOKENS = ("batch", "bs=", "BS=", "批大小", "并发", "bs ", "Batch")
# 不算加速比的场合：纯倍数比较（如"参数量大 10 倍"）很难自动区分，
# 因此这一项报 WARN 不报 ERROR，由人复核。

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
    """
    line = re.sub(r"\[\[[^\]]*\]\]", " ", line)   # 双链整体挖掉
    line = re.sub(r"`[^`]*`", " ", line)            # 行内代码挖掉
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
            if (not in_meta) and LOSSLESS_PAT.search(line) and not CALIBER_EXEMPT.search(line):
                ctx = " ".join(strip_for_rules(x) for x in lines[max(0, i - 4):i + 3])
                if not any(t in ctx for t in CALIBER_TOKENS):
                    warn_lossless.append((p.name, i, raw.strip()[:70]))
            # 铁律二
            if SPEEDUP_PAT.search(line):
                ctx = " ".join(strip_for_rules(x) for x in lines[max(0, i - 6):i + 6])
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
