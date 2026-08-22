"""repo_stats.py —— 从真实 clone 出来的源码里算「代码结构」的确定性事实。

回答的问题：
  * 这个引擎到底多大？各语言占比多少？（判断"它是 Python 项目还是 C++ 项目"）
  * 代码量集中在哪几个子目录？（判断"作者把复杂度放在哪"）
  * 自研 kernel 有多少？CUDA / Triton / 汇编各多少？（判断"它是编排层还是算子层"）

用法:
    python repo_stats.py            # 全部已 clone 的引擎
    python repo_stats.py vllm sglang
    python repo_stats.py --selftest
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from common import (ENGINES, OUT, available_engines, count_lines, dump, dump_engines,
                    engine_path, read_text, rel, repo_ref, require_engines,
                    walk_files)

# 后缀 -> 语言。只登记我们真正会拿来下结论的那些。
LANG_BY_SUFFIX = {
    ".py": "Python", ".pyi": "Python",
    ".cu": "CUDA", ".cuh": "CUDA",
    ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".h": "C/C++ header",
    ".hpp": "C/C++ header", ".hh": "C/C++ header", ".c": "C",
    ".rs": "Rust",
    ".go": "Go",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".md": "Markdown",
    ".rst": "reST",
    ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".sh": "Shell",
    ".proto": "Protobuf",
    ".mojo": "Mojo",
}

# Triton kernel 的判定。
# 第一版只认 `@triton.jit` 装饰器 —— 这是**假阴性**：MLC-LLM 把 kernel 写成
# `triton.jit(triton_kernel)` 的**函数调用形式**再交给 TVM 的 `T.call_kernel(...)`
# （`python/mlc_llm/op/triton.py:335` / `:453`），语义上就是手写 Triton kernel，
# 却因为没用装饰器语法被判成 0。所以两种写法都要认。
# 一般教训：**"仓库里搜不到某个模式"和"这个仓库没有这个功能"是两回事**，
# 匹配规则本身就是需要被审计的对象。
TRITON_MARKS = ("@triton.jit", "triton.jit(")
CUTLASS_MARK = "cutlass"


def _subdir_key(relpath: str, depth: int = 2) -> str:
    parts = relpath.split("/")
    if len(parts) <= 1:
        return "<root>"
    return "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1])


def analyze(name: str) -> dict:
    root = engine_path(name)
    assert root is not None, name
    ref = repo_ref(name)

    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "lines": 0, "code": 0})
    by_dir: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "lines": 0})
    total_files = 0
    total_lines = 0
    triton_files: list[str] = []
    cuda_files: list[str] = []
    cutlass_hits = 0
    py_test_lines = 0
    py_src_lines = 0

    for fp in walk_files(root):
        suffix = fp.suffix.lower()
        lang = LANG_BY_SUFFIX.get(suffix)
        if lang is None:
            continue
        r = rel(fp, root)
        total, code = count_lines(fp)
        total_files += 1
        total_lines += total
        by_lang[lang]["files"] += 1
        by_lang[lang]["lines"] += total
        by_lang[lang]["code"] += code
        d = _subdir_key(r)
        by_dir[d]["files"] += 1
        by_dir[d]["lines"] += total

        if lang == "CUDA":
            cuda_files.append(r)
        if suffix == ".py":
            text = read_text(fp)
            if any(m in text for m in TRITON_MARKS):
                triton_files.append(r)
            # 测试代码 vs 产品代码：路径里带 test 的算测试
            low = r.lower()
            if "/test" in low or low.startswith("test") or "/tests/" in low:
                py_test_lines += total
            else:
                py_src_lines += total
        if suffix in (".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp"):
            if CUTLASS_MARK in read_text(fp).lower():
                cutlass_hits += 1

    top_dirs = sorted(by_dir.items(), key=lambda kv: -kv[1]["lines"])[:20]
    langs = sorted(by_lang.items(), key=lambda kv: -kv[1]["lines"])

    return {
        "ref": ref.as_dict() if ref else {},
        "totals": {
            "files_counted": total_files,
            "lines_counted": total_lines,
            "python_src_lines": py_src_lines,
            "python_test_lines": py_test_lines,
            # ⚠️ 这个比值**只统计 Python**，且只认路径里带 test 的文件。
            # 对 Rust/C++ 项目严重失真：Rust 的 `#[cfg(test)]` 单元测试就写在源文件里，
            # 完全不会被算进来。实测例子：TGI 报 0.108，但它的 `router/src/` 里有
            # 9 个 `#[cfg(test)]` 模块、71 个 `#[test]` 函数（`queue.rs` 约三分之一是测试）。
            # **跨语言比这个数没有意义**，正文引用时必须写明"Python 口径"。
            "test_to_src_ratio": round(py_test_lines / py_src_lines, 3) if py_src_lines else None,
            "test_ratio_caveat": "仅统计 Python 且只认路径含 test 的文件；对 Rust/C++ 项目严重低估",
        },
        "languages": [{"lang": k, **v} for k, v in langs],
        "top_dirs": [{"dir": k, **v} for k, v in top_dirs],
        "kernels": {
            "cuda_files": len(cuda_files),
            "triton_jit_files": len(triton_files),
            "files_mentioning_cutlass": cutlass_hits,
            "triton_examples": sorted(triton_files)[:15],
            "cuda_examples": sorted(cuda_files)[:15],
        },
    }


def selftest() -> int:
    """自检：不依赖 _src，用临时目录造一个假仓库验证统计逻辑。"""
    import tempfile, os, json
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pkg").mkdir()
        (root / "pkg" / "a.py").write_text("import triton\n\n@triton.jit\ndef k():\n    pass\n", encoding="utf-8")
        (root / "pkg" / "b.cu").write_text("// c\n__global__ void f(){}\n", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "test_a.py").write_text("def test_x():\n    assert 1\n", encoding="utf-8")

        by_lang = defaultdict(int)
        triton = 0
        for fp in walk_files(root):
            lang = LANG_BY_SUFFIX.get(fp.suffix.lower())
            if not lang:
                continue
            by_lang[lang] += 1
            if fp.suffix == ".py" and any(m in read_text(fp) for m in TRITON_MARKS):
                triton += 1
        if by_lang["Python"] != 2:
            print(f"FAIL python files: {by_lang['Python']}"); ok = False
        if by_lang["CUDA"] != 1:
            print(f"FAIL cuda files: {by_lang['CUDA']}"); ok = False
        if triton != 1:
            print(f"FAIL triton files: {triton}"); ok = False
        t, c = count_lines(root / "pkg" / "b.cu")
        if (t, c) != (2, 1):
            print(f"FAIL count_lines b.cu -> {(t, c)} (期望 2 行总计, 1 行代码)"); ok = False
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    names = [a for a in argv if not a.startswith("-")] or available_engines()
    names = require_engines(names)
    if not names:
        print("没有可分析的引擎，先把 _src/ 下的仓库 clone 下来。", file=sys.stderr)
        return 2
    out = {}
    for n in names:
        print(f"[repo_stats] {n} ...", flush=True)
        out[n] = analyze(n)
    fp = dump_engines("repo_stats.json", out)
    print(f"-> {fp}")
    for n, d in out.items():
        t = d["totals"]
        print(f"  {ENGINES[n]['label']:<14} {t['lines_counted']:>9,} 行  "
              f"py_src={t['python_src_lines']:>8,}  cuda={d['kernels']['cuda_files']:>4}  "
              f"triton={d['kernels']['triton_jit_files']:>4}  @{d['ref'].get('sha_short','?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
