#!/usr/bin/env python
"""_verify.py —— 本教学库的体检器。不通过就别提交。

和隔壁研究库（`../VLLM-SGlang-研究-推进/`）的体检器**规则不同**，因为定位不同：

  研究库：每条断言必须回到**别人源码**的一行。
  本  库：每条原理必须有**本库自己能跑的代码**，外加可选的"真实引擎是怎么做的"对照。

所以这里验四件事：

  1. 本库自引用   `_lab/minigpt.py:123` 必须在本库里真实存在且行号不越界。
                  **这是本库最重要的一条** —— 讲原理却指不到可跑代码，等于没讲。
  2. 跨库引用     `vllm:vllm/v1/core/sched/scheduler.py:123` 这种带引擎前缀的，
                  对着 `../VLLM-SGlang-研究-推进/_src/<engine>/` 核。
                  **那份 clone 不在时自动跳过并明说跳了多少条**，不伪装通过。
  3. 双链         [[xxx]] 必须落到本库某个 .md，或在 `_PLAN.md` 名册里（前向引用记 warn）。
  4. 占位符与结构 TODO/FIXME 之类不许留；每篇要有一级标题与自测出口。

用法：
    python _verify.py            # 全量
    python _verify.py --strict   # 缺隔壁 clone 也当失败（CI 用）
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 隔壁研究库的源码 clone —— 本库只读，不管理它
SIBLING_SRC = ROOT.parent / "VLLM-SGlang-研究-推进" / "_src"

ENGINE_DIRS = {
    "vllm": "vllm", "sglang": "sglang", "lmdeploy": "lmdeploy", "tgi": "tgi",
    "lightllm": "lightllm", "tensorrt-llm": "tensorrt-llm", "dynamo": "dynamo",
    "ktransformers": "ktransformers", "mooncake": "mooncake",
    "llama.cpp": "llama.cpp", "mlc-llm": "mlc-llm", "tokasaurus": "tokasaurus",
}

CODE_SUFFIX = r"(?:py|pyi|rs|cpp|cc|cu|cuh|h|hpp|toml|yaml|yml|json|md|sh)"
CITE_RE = re.compile(
    r"`(?:(?P<engine>[a-z0-9.\-]+):)?(?P<path>[A-Za-z0-9_][A-Za-z0-9_./\-]*\."
    + CODE_SUFFIX + r"):(?P<line>\d+)(?:-(?P<end>\d+))?`")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
# 与研究库同样的两层免伤：不收「占位」二字；后面跟中日韩字符（可夹分隔符）时不算
PLACEHOLDER_RE = re.compile(
    r"(?<![A-Za-z])(TODO|TBD|FIXME|XXX|待补|待填|\?\?\?)(?![A-Za-z])"
    r"(?!\s*[/、，,·]?\s*[一-鿿])")
FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
ROSTER_RE = re.compile(r"`(\d{2}-[^`]+?)\.md`")
SKIP_DIRS = {"_lab", ".git", "__pycache__"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.stats: dict[str, int] = defaultdict(int)


def md_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.md")
                  if not any(x in p.relative_to(ROOT).parts for x in SKIP_DIRS))


def roster() -> set[str]:
    plan = ROOT / "_PLAN.md"
    if not plan.exists():
        return set()
    return {m.group(1) for m in ROSTER_RE.finditer(plan.read_text(encoding="utf-8"))}


def line_count(p: Path, cache: dict[Path, int]) -> int:
    if p in cache:
        return cache[p]
    try:
        n = sum(1 for _ in p.open("r", encoding="utf-8", errors="replace"))
    except OSError:
        n = -1
    cache[p] = n
    return n


def check(rep: Report, strict: bool) -> None:
    files = md_files()
    rep.stats["md_files"] = len(files)
    stems = {p.stem for p in files}
    planned = roster()
    have_sibling = SIBLING_SRC.exists()
    cache: dict[Path, int] = {}

    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        raw = p.read_text(encoding="utf-8", errors="replace")
        body = FENCE_RE.sub("", raw)
        prose = INLINE_CODE_RE.sub("", body)

        if p.name.startswith("_"):        # _PLAN.md 是蓝图，只查占位符
            for m in PLACEHOLDER_RE.finditer(prose):
                rep.errors.append(f"{rel}: 残留占位符 {m.group(1)!r}")
            continue

        if not re.search(r"^#\s+\S", raw, re.M):
            rep.errors.append(f"{rel}: 缺一级标题")
        is_index = p.name.startswith(("00-", "99-", "README"))
        if not is_index and not re.search(r"(自测|延伸阅读|练习|检查清单)", body):
            rep.warns.append(f"{rel}: 没有「自测/延伸阅读」出口")

        for m in PLACEHOLDER_RE.finditer(prose):
            rep.errors.append(f"{rel}:~{prose[:m.start()].count(chr(10)) + 1}: "
                              f"残留占位符 {m.group(1)!r}")

        for m in WIKILINK_RE.finditer(body):
            t = m.group(1).strip()
            rep.stats["wikilinks"] += 1
            if t in stems:
                continue
            at = body[:m.start()].count("\n") + 1
            if t in planned:
                rep.stats["forward_refs"] += 1
                rep.warns.append(f"{rel}:{at}: 前向引用 [[{t}]]（名册内，尚未落盘）")
            else:
                rep.errors.append(f"{rel}:{at}: 死链 [[{t}]]（不在 _PLAN.md §4 名册里）")

        for m in CITE_RE.finditer(body):
            rep.stats["citations"] += 1
            engine, path = m.group("engine"), m.group("path")
            lo = int(m.group("line"))
            hi = int(m.group("end")) if m.group("end") else lo
            at = body[:m.start()].count("\n") + 1
            if hi < lo:
                rep.errors.append(f"{rel}:{at}: 行号区间倒置 {lo}-{hi}")

            if engine is None:                      # 本库自引用
                rep.stats["self_citations"] += 1
                target = ROOT / path
                if not target.exists():
                    rep.errors.append(f"{rel}:{at}: 本库文件不存在 → {path}"
                                      f"（本库自引用不加引擎前缀）")
                    continue
                n = line_count(target, cache)
                if n >= 0 and hi > n:
                    rep.errors.append(f"{rel}:{at}: 行号越界 {path}:{hi}，该文件只有 {n} 行")
                else:
                    rep.stats["citations_ok"] += 1
            else:                                    # 跨库引用真实引擎
                rep.stats["engine_citations"] += 1
                if engine not in ENGINE_DIRS:
                    rep.errors.append(f"{rel}:{at}: 未登记的引擎 {engine!r}")
                    continue
                if not have_sibling:
                    rep.stats["citations_skipped"] += 1
                    continue
                base = SIBLING_SRC / ENGINE_DIRS[engine]
                if not base.exists():
                    rep.stats["citations_skipped"] += 1
                    continue
                target = base / path
                if not target.exists():
                    rep.errors.append(f"{rel}:{at}: 引擎源码里没有这个文件 → {engine}:{path}")
                    continue
                n = line_count(target, cache)
                if n >= 0 and hi > n:
                    rep.errors.append(f"{rel}:{at}: 行号越界 {engine}:{path}:{hi}，"
                                      f"该文件只有 {n} 行")
                else:
                    rep.stats["citations_ok"] += 1

    if not have_sibling:
        msg = (f"未找到隔壁 {SIBLING_SRC}，{rep.stats['citations_skipped']} 条引擎引用"
               f"只验了格式没验行号。要全验请先按研究库 _PLAN.md §2 拉源码。")
        (rep.errors if strict else rep.warns).append(msg)


def main(argv: list[str]) -> int:
    rep = Report()
    check(rep, strict="--strict" in argv)
    s = rep.stats
    print("=" * 70)
    print(f"  md 文件 {s['md_files']}  |  双链 {s['wikilinks']}（前向 {s['forward_refs']}）")
    print(f"  代码引用 {s['citations']}：本库自引用 {s['self_citations']}、"
          f"引擎引用 {s['engine_citations']}")
    print(f"  核过行号 {s['citations_ok']}，跳过 {s['citations_skipped']}")
    print("=" * 70)
    for w in rep.warns:
        print(f"  [warn] {w}")
    for e in rep.errors:
        print(f"  [ERR ] {e}")
    print("-" * 70)
    if rep.errors:
        print(f"FAIL：{len(rep.errors)} 个错误，{len(rep.warns)} 个警告")
        return 1
    print(f"PASS：0 错误，{len(rep.warns)} 个警告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
