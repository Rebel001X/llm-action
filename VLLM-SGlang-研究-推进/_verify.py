#!/usr/bin/env python
"""_verify.py —— 本库的体检器。不通过就别提交。

本库的诚实标准是「每条代码断言都能回到一行真实源码」，所以验的是：

  1. 取证基准头   每篇正文开头必须声明 `本篇取证基准：<engine> @ <sha>`，
                  且 sha 必须与 _lab/out/*.json 里记录的 clone sha 一致 ——
                  防止「拿 A 版本的代码讲 B 版本的行号」。
  2. 代码引用     形如 `vllm/v1/core/sched/scheduler.py:123` 的行内代码，
                  必须在对应引擎源码里真实存在，且文件行数 >= 引用行号。
  3. 双链         [[xxx]] 必须能落到本库某个 .md。
  4. 占位符       TODO / TBD / 待补 / XXX / ??? 一律不许留在正文。
  5. 结构         每篇要有一级标题、要有「延伸阅读」或「自测」出口（KB 硬指标）。

三种模式：
    python _verify.py            # 全量（有 _src 就验行号，没有就跳过并明说跳了多少条）
    python _verify.py --links    # 只验双链与占位符（最快）
    python _verify.py --strict   # 缺 _src 也当失败（CI 用）
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "_src"
LAB_OUT = ROOT / "_lab" / "out"

# 引擎名 -> _src 下目录名（与 _lab/common.py 的 ENGINES 保持一致）
ENGINE_DIRS = {
    "vllm": "vllm", "sglang": "sglang", "lmdeploy": "lmdeploy", "tgi": "tgi",
    "lightllm": "lightllm", "tensorrt-llm": "tensorrt-llm", "dynamo": "dynamo",
    "ktransformers": "ktransformers", "mooncake": "mooncake",
    "llama.cpp": "llama.cpp", "mlc-llm": "mlc-llm", "tokasaurus": "tokasaurus",
}

CODE_SUFFIX = r"(?:py|pyi|rs|cpp|cc|cu|cuh|h|hpp|toml|yaml|yml|json|md|sh)"
# 行内代码里的代码引用：[engine:]path/to/file.py:123  或 ...:123-145
CITE_RE = re.compile(
    r"`(?:(?P<engine>[a-z0-9.\-]+):)?(?P<path>[A-Za-z0-9_][A-Za-z0-9_./\-]*\."
    + CODE_SUFFIX + r"):(?P<line>\d+)(?:-(?P<end>\d+))?`")
BASELINE_RE = re.compile(r"本篇取证基准[^`\n]*`(?P<engine>[a-z0-9.\-]+)`\s*@\s*`(?P<sha>[0-9a-f]{7,40})`")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
# 占位符扫描：故意**不收**「占位」二字 —— 正文里「占位 token」「占位符」是正当术语，
# 收了就会大面积误伤（这个坑在 vLLM 解剖库踩过一次）。
#
# 第二层误伤：本库**本身就在统计 TODO 注释**，正文里「TODO 认领率」
# 「TODO/未实现/废弃 的跨引擎扫描」都是把 TODO 当研究对象谈，不是施工残留。
# 所以后面跟中日韩字符（允许中间夹一个 / 、 ， · 之类的分隔符）时不算 ——
# 真正的残留标记后面跟的是冒号、括号、英文或行尾，不会是中文。
PLACEHOLDER_RE = re.compile(
    r"(?<![A-Za-z])(TODO|TBD|FIXME|XXX|待补|待填|\?\?\?)(?![A-Za-z])"
    r"(?!\s*[/、，,·]?\s*[一-鿿])")
FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
ROSTER_RE = re.compile(r"`(\d{2}-[^`]+?)\.md`")

SKIP_DIRS = {"_src", "_lab", ".git", "__pycache__"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.stats: dict[str, int] = defaultdict(int)

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


def md_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        out.append(p)
    return sorted(out)


def strip_fences(text: str) -> str:
    """去掉围栏代码块 —— 里面的 TODO / [[..]] 是示例，不算正文。"""
    return FENCE_RE.sub("", text)


def roster() -> set[str]:
    """从 _PLAN.md §4 篇目表里读出「计划中的文件名」。

    为什么需要：多 agent 并行施工时，先写完的篇会链到还没落盘的兄弟篇。
    那是**前向引用**，不是死链 —— 判成错误会逼 agent 去掉正确的链接。
    但链到名册以外的名字仍然是硬错误（那是编出来的）。
    """
    plan = ROOT / "_PLAN.md"
    if not plan.exists():
        return set()
    return {m.group(1) for m in ROSTER_RE.finditer(plan.read_text(encoding="utf-8"))}


def clone_shas() -> dict[str, str]:
    shas: dict[str, str] = {}
    for jf in sorted(LAB_OUT.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for name, d in (data.get("engines") or {}).items():
            sha = (d.get("ref") or {}).get("sha")
            if sha:
                shas.setdefault(name, sha)
    return shas


def file_line_count(p: Path, cache: dict[Path, int]) -> int:
    if p in cache:
        return cache[p]
    try:
        n = sum(1 for _ in p.open("r", encoding="utf-8", errors="replace"))
    except OSError:
        n = -1
    cache[p] = n
    return n


def check(rep: Report, mode: str) -> None:
    files = md_files()
    rep.stats["md_files"] = len(files)
    stems = {p.stem for p in files}
    planned = roster()
    shas = clone_shas()
    have_src = SRC.exists() and any((SRC / d).exists() for d in ENGINE_DIRS.values())
    line_cache: dict[Path, int] = {}

    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        raw = p.read_text(encoding="utf-8", errors="replace")
        body = strip_fences(raw)

        # `_PLAN.md` 之类的元文件：它写的是**引用格式的示范**和跨库笔记链接，
        # 不是正文断言，只查占位符，别拿正文的尺子量它。
        if p.name.startswith("_"):
            for m in PLACEHOLDER_RE.finditer(INLINE_CODE_RE.sub("", body)):
                rep.err(f"{rel}: 残留占位符 {m.group(1)!r}")
            rep.stats["meta_files"] += 1
            continue

        # --- 5. 结构 ---
        if not re.search(r"^#\s+\S", raw, re.M):
            rep.err(f"{rel}: 缺一级标题")
        is_index = p.name.startswith(("00-", "_", "99-", "README"))
        if not is_index and not re.search(r"(延伸阅读|自测|练习|检查清单|参考)", body):
            rep.warn(f"{rel}: 没有「延伸阅读/自测」出口")

        # --- 4. 占位符（行内代码里的不算：那是在引用源码里的标记）---
        prose = INLINE_CODE_RE.sub("", body)
        for m in PLACEHOLDER_RE.finditer(prose):
            line = prose[:m.start()].count("\n") + 1
            rep.err(f"{rel}:~{line}: 残留占位符 {m.group(1)!r}")

        # --- 3. 双链：名册内但未落盘 = 前向引用（warn）；名册外 = 死链（err）---
        for m in WIKILINK_RE.finditer(body):
            target = m.group(1).strip()
            rep.stats["wikilinks"] += 1
            if target in stems:
                continue
            line = body[:m.start()].count("\n") + 1
            if target in planned:
                rep.stats["forward_refs"] += 1
                rep.warn(f"{rel}:{line}: 前向引用 [[{target}]]（在 _PLAN 名册里，尚未落盘）")
            else:
                rep.err(f"{rel}:{line}: 死链 [[{target}]]（不在 _PLAN.md §4 名册里）")

        # --- 1. 取证基准头 ---
        bm = BASELINE_RE.search(raw)
        default_engine = None
        if bm:
            default_engine = bm.group("engine")
            declared = bm.group("sha")
            actual = shas.get(default_engine)
            if default_engine not in ENGINE_DIRS:
                rep.err(f"{rel}: 取证基准引擎名 {default_engine!r} 不在登记表里")
            elif actual and not actual.startswith(declared):
                rep.err(f"{rel}: 取证基准 sha {declared} != 实际 clone {actual[:len(declared)]}")
            rep.stats["baselined"] += 1
        elif not is_index:
            rep.err(f"{rel}: 缺「本篇取证基准：`<engine>` @ `<sha>`」声明")

        # --- 2. 代码引用 ---
        for m in CITE_RE.finditer(body):
            rep.stats["citations"] += 1
            engine = m.group("engine") or default_engine
            path = m.group("path")
            lineno = int(m.group("line"))
            end = int(m.group("end")) if m.group("end") else lineno
            at = body[:m.start()].count("\n") + 1
            # 本库可以引用**自己的**工具（讲清楚某个数字是怎么算出来的、
            # 或者指出某个抽取脚本的缺陷时必须能指到行）。这类路径按库根解析，不走引擎。
            if path.startswith(("_lab/", "_verify.py")):
                rep.stats["self_citations"] += 1
                target = ROOT / path
                if not target.exists():
                    rep.err(f"{rel}:{at}: 引用的本库文件不存在 → {path}")
                    continue
                n = file_line_count(target, line_cache)
                if n >= 0 and end > n:
                    rep.err(f"{rel}:{at}: 行号越界 {path}:{end}，该文件只有 {n} 行")
                else:
                    rep.stats["citations_ok"] += 1
                continue
            if engine is None:
                rep.err(f"{rel}:{at}: 代码引用 `{path}:{lineno}` 无法判定引擎（本篇无取证基准头，也没写 engine: 前缀）")
                continue
            if engine not in ENGINE_DIRS:
                rep.err(f"{rel}:{at}: 引用了未登记的引擎 {engine!r}")
                continue
            if end < lineno:
                rep.err(f"{rel}:{at}: 行号区间倒置 {lineno}-{end}")
            if not have_src:
                rep.stats["citations_skipped"] += 1
                continue
            base = SRC / ENGINE_DIRS[engine]
            if not base.exists():
                rep.stats["citations_skipped"] += 1
                continue
            target = base / path
            if not target.exists():
                rep.err(f"{rel}:{at}: 引用的文件不存在 → {engine}:{path}")
                continue
            n = file_line_count(target, line_cache)
            if n >= 0 and end > n:
                rep.err(f"{rel}:{at}: 行号越界 {engine}:{path}:{end}，该文件只有 {n} 行")
            else:
                rep.stats["citations_ok"] += 1

    if not have_src:
        msg = (f"未找到 _src/ 源码，{rep.stats['citations_skipped']} 条代码引用只验了格式没验行号。"
               f"要全验请按 _PLAN.md 的 clone 命令拉源码。")
        (rep.err if mode == "strict" else rep.warn)(msg)


def main(argv: list[str]) -> int:
    mode = "full"
    if "--links" in argv:
        mode = "links"
    if "--strict" in argv:
        mode = "strict"

    rep = Report()
    check(rep, mode)

    print("=" * 68)
    print(f"  md 文件 {rep.stats['md_files']}  |  双链 {rep.stats['wikilinks']}"
          f"（前向引用 {rep.stats['forward_refs']}）  |  取证基准头 {rep.stats['baselined']}")
    print(f"  代码引用 {rep.stats['citations']}（核过行号 {rep.stats['citations_ok']}，"
          f"跳过 {rep.stats['citations_skipped']}）")
    print("=" * 68)
    for w in rep.warns:
        print(f"  [warn] {w}")
    for e in rep.errors:
        print(f"  [ERR ] {e}")
    print("-" * 68)
    if rep.errors:
        print(f"FAIL：{len(rep.errors)} 个错误，{len(rep.warns)} 个警告")
        return 1
    print(f"PASS：0 错误，{len(rep.warns)} 个警告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
