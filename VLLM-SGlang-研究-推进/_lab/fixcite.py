"""fixcite.py —— 修「路径写短了」的代码引用。

反复出现的同一种错：正文里把引用写成 `serving.py:123` 或 `cohere/api_router.py:45`，
少了从仓库根算起的前缀。`_verify.py` 会判它「文件不存在」，但不会替你补。

这个脚本干的事：对每条解析不到的引用，在对应引擎源码里**按后缀唯一匹配**找回全路径。
  * 恰好匹配到 1 个文件 → 可以自动补（`--apply`）
  * 匹配到多个 → 只报告，让人自己定（**绝不瞎选**，选错等于制造假引用）
  * 匹配到 0 个 → 报告为真错误，可能是文件名也写错了

行号不做任何猜测：补的只是路径前缀，行号原样保留，补完仍由 `_verify.py` 核。

用法:
    python fixcite.py                  # 只报告
    python fixcite.py --apply          # 唯一匹配的就地改写
    python fixcite.py --selftest
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from common import SRC

ROOT = Path(__file__).resolve().parent.parent
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
BASELINE_RE = re.compile(r"本篇取证基准[^`\n]*`(?P<engine>[a-z0-9.\-]+)`")
SKIP_DIRS = {"_src", "_lab", ".git", "__pycache__"}


def build_index(engine: str) -> dict[str, list[str]]:
    """后缀路径 -> 完整相对路径列表。键是所有可能的尾段，便于按写短了的形式查。"""
    root = SRC / ENGINE_DIRS[engine]
    idx: dict[str, list[str]] = defaultdict(list)
    if not root.exists():
        return idx
    for fp in root.rglob("*"):
        if not fp.is_file() or ".git" in fp.parts:
            continue
        rel = fp.relative_to(root).as_posix()
        parts = rel.split("/")
        for i in range(len(parts)):
            idx["/".join(parts[i:])].append(rel)
    return idx


def md_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.md")
                  if not any(x in p.relative_to(ROOT).parts for x in SKIP_DIRS)
                  and not p.name.startswith("_"))


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    apply = "--apply" in argv
    indexes: dict[str, dict[str, list[str]]] = {}
    n_ok = n_fixed = n_ambig = n_missing = 0

    for p in md_files():
        text = p.read_text(encoding="utf-8")
        bm = BASELINE_RE.search(text)
        default_engine = bm.group("engine") if bm else None
        rel_md = p.relative_to(ROOT).as_posix()
        edits: list[tuple[str, str]] = []

        for m in CITE_RE.finditer(text):
            engine = m.group("engine") or default_engine
            path = m.group("path")
            if engine not in ENGINE_DIRS:
                continue
            if engine not in indexes:
                indexes[engine] = build_index(engine)
            idx = indexes[engine]
            root = SRC / ENGINE_DIRS[engine]
            if (root / path).exists():
                n_ok += 1
                continue
            cands = idx.get(path, [])
            if len(cands) == 1:
                n_fixed += 1
                old = m.group(0)
                new = old.replace(path, cands[0])
                edits.append((old, new))
                print(f"  [fix ] {rel_md}: {path} -> {cands[0]}")
            elif len(cands) > 1:
                n_ambig += 1
                print(f"  [多义] {rel_md}: {path} 有 {len(cands)} 个候选，需人工定：")
                for c in cands[:5]:
                    print(f"           {c}")
            else:
                n_missing += 1
                print(f"  [缺失] {rel_md}: {path} 在 {engine} 源码里找不到任何同名文件")

        if apply and edits:
            for old, new in dict(edits).items():
                text = text.replace(old, new)
            p.write_text(text, encoding="utf-8")

    print(f"\n本来就对 {n_ok}  |  可唯一补全 {n_fixed}  |  多义 {n_ambig}  |  查无此文件 {n_missing}")
    if not apply and n_fixed:
        print("加 --apply 才会真的改写文件。")
    return 0


def selftest() -> int:
    """自检：索引要能按任意尾段命中，且多义时必须报多义而不是随便选一个。"""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel in ["a/b/serving.py", "a/c/serving.py", "a/b/only.py"]:
            fp = root / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text("x = 1\n", encoding="utf-8")
        idx: dict[str, list[str]] = defaultdict(list)
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            r = fp.relative_to(root).as_posix()
            parts = r.split("/")
            for i in range(len(parts)):
                idx["/".join(parts[i:])].append(r)
        if len(idx["serving.py"]) != 2:
            print(f"FAIL serving.py 应是多义 -> {idx['serving.py']}"); ok = False
        if idx["b/serving.py"] != ["a/b/serving.py"]:
            print(f"FAIL 带一层目录应唯一 -> {idx['b/serving.py']}"); ok = False
        if idx["only.py"] != ["a/b/only.py"]:
            print(f"FAIL only.py 应唯一 -> {idx['only.py']}"); ok = False

    # 正则要能吃带引擎前缀和不带前缀两种
    m1 = CITE_RE.search("见 `vllm:vllm/v1/core/sched/scheduler.py:686` 处")
    m2 = CITE_RE.search("见 `serving.py:12` 处")
    if not m1 or m1.group("engine") != "vllm" or m1.group("line") != "686":
        print(f"FAIL 带前缀解析 -> {m1}"); ok = False
    if not m2 or m2.group("engine") is not None or m2.group("path") != "serving.py":
        print(f"FAIL 不带前缀解析 -> {m2}"); ok = False
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
