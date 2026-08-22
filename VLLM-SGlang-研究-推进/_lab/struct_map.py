"""struct_map.py —— 画「代码结构图」：每个引擎的关键子系统落在哪个文件、有哪些类、多大。

为什么要有它：讲代码结构最容易犯的错是**照着 README 抄目录树**。
这里改成从源码里按子系统关键词定位真实文件，再用 AST 把类/方法抽出来，
每一条都带 `文件:行号`，正文引用时可被 verify.py 逐条核对。

子系统切分（对所有引擎用同一把尺，好横向比）：
    entrypoint / scheduler / kv_cache / attention / model_exec /
    distributed / quantization / speculative / structured_output / disagg

用法:
    python struct_map.py vllm
    python struct_map.py               # 全部
    python struct_map.py --selftest
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

from common import (ENGINES, available_engines, count_lines, dump, dump_engines, engine_path,
                    read_text, rel, repo_ref, require_engines, walk_files)

# 子系统 -> 路径关键词。命中即归类；一个文件可归多类（例如 v1/core/sched/scheduler.py）。
#
# ⚠️ 已知口径缺陷（写正文的人必须知道）：这是**纯文件名匹配**，会有假阳性。
# 实测例子：vLLM 的 "scheduler" 桶里混进了 KV-connector 的 offloading/scheduler.py
# 和 EPLB 相关文件，报出 20 文件 / 8,424 行，而真正的连续批处理调度器只有
# 6 文件 / 4,012 行。所以本工具的数字只配当**导航线索**，
# 一旦要写进正文当结论，必须回源码点数复核，并在正文里写复核后的数。

# 第二个已知缺陷（TGI 那篇写作时发现）：关键词表是照 **Python 项目的命名习惯**列的，
# 对 Rust/C++ 项目有系统性盲区。TGI 真正的批处理引擎叫
# `queue.rs` / `backend.rs` / `radix.rs` / `block_allocator.rs`，一个都不命中，
# 于是 scheduler 桶报 0 行；而它的顶层目录恰好叫 `server/`，
# 于是几乎全部 Python 产品码被 entrypoint 桶吸走。下面补了这些命名，但**盲区不可能补全** ——
# 用这些数字前，永远先回源码点一次。
SUBSYSTEMS: dict[str, list[str]] = {
    "entrypoint":        ["entrypoint", "server", "api_server", "http_server", "openai", "cli"],
    "scheduler":         ["sched", "scheduler", "batch_sched", "policy", "waiting",
                          "queue", "batcher", "batching"],
    "kv_cache":          ["kv_cache", "block_manager", "cache_engine", "memory_pool",
                          "radix_cache", "prefix_cache", "block_pool", "kv_pool", "paged",
                          "radix", "block_allocator", "allocator"],
    "attention":         ["attention", "attn", "flashinfer", "flash_attn"],
    "model_exec":        ["model_runner", "model_executor", "worker", "executor", "model_loader"],
    "distributed":       ["distributed", "parallel_state", "tensor_parallel", "pipeline",
                          "communicat", "nccl", "all_reduce"],
    "quantization":      ["quantization", "quant", "gptq", "awq", "fp8", "int8", "compressed"],
    "speculative":       ["spec_decode", "speculative", "eagle", "medusa", "ngram", "draft"],
    "structured_output": ["structured_output", "guided", "grammar", "outlines", "xgrammar",
                          "constrained", "json_schema"],
    "disagg":            ["disagg", "kv_connector", "kv_transfer", "pd_", "prefill_decode",
                          "nixl", "mooncake"],
}

# 每个引擎的「代码根」——只在这个子树里找，避免把 tests/benchmarks/docs 混进来
PKG_ROOTS: dict[str, list[str]] = {
    "vllm": ["vllm"],
    "sglang": ["python/sglang"],
    "lmdeploy": ["lmdeploy"],
    "lightllm": ["lightllm"],
    "tensorrt-llm": ["tensorrt_llm"],
    # KTransformers 2026 版把老 python 包整体挪进 archive/，新增 kt-kernel/（C++/CUDA 算子）
    "ktransformers": ["archive/ktransformers", "kt-kernel", "ktransformers"],
    # MLC-LLM 的调度器（EngineImpl::Step）与 PD 分离全在 C++ 侧，
    # 只扫 python/mlc_llm 会得到 scheduler=0 / disagg=0 这种假的空结果
    "mlc-llm": ["python/mlc_llm", "cpp"],
    "tokasaurus": ["tokasaurus"],
    "tgi": ["server/text_generation_server", "router/src", "backends"],
    "dynamo": ["lib", "components"],
    "llama.cpp": ["src", "tools", "ggml/src", "common"],
    "mooncake": ["mooncake-transfer-engine", "mooncake-store"],
}

EXCLUDE_PARTS = ("/test", "/tests/", "/benchmark", "/docs/", "/examples/", "/third_party/")


def _classify(relpath: str) -> list[str]:
    low = relpath.lower()
    hits = []
    for sub, keys in SUBSYSTEMS.items():
        if any(k in low for k in keys):
            hits.append(sub)
    return hits


def _classes_in(fp: Path, root: Path) -> list[dict]:
    text = read_text(fp)
    if not text or fp.suffix != ".py":
        return []
    try:
        tree = ast.parse(text, filename=str(fp))
    except SyntaxError:
        return []
    lines = text.splitlines()
    out = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        methods = [m.name for m in node.body
                   if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        doc = ast.get_docstring(node) or ""
        out.append({
            "name": node.name,
            "file": rel(fp, root),
            "line": node.lineno,
            "lines": end - node.lineno + 1,
            "bases": [ast.unparse(b) for b in node.bases][:4],
            "n_methods": len(methods),
            "methods": methods[:25],
            "doc_first_line": " ".join(doc.split())[:160] if doc else None,
        })
        del lines  # 只是提醒：行数用 AST 的 end_lineno，不再二次数
        lines = text.splitlines()
    return out


def analyze(name: str) -> dict:
    root = engine_path(name)
    assert root is not None
    ref = repo_ref(name)
    roots = PKG_ROOTS.get(name, ["."])

    per_sub: dict[str, dict] = {s: {"files": [], "lines": 0, "classes": []} for s in SUBSYSTEMS}
    tree_lines: dict[str, int] = defaultdict(int)
    tree_files: dict[str, int] = defaultdict(int)
    n_scanned = 0

    for sub_root in roots:
        base = root / sub_root
        if not base.exists():
            continue
        for fp in walk_files(base, {".py", ".rs", ".cpp", ".cc", ".cu", ".h", ".hpp"}):
            r = rel(fp, root)
            if any(x in "/" + r.lower() for x in EXCLUDE_PARTS):
                continue
            n_scanned += 1
            total, _ = count_lines(fp)
            parts = r.split("/")
            key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1]) or "<root>"
            tree_lines[key] += total
            tree_files[key] += 1
            for sub in _classify(r):
                per_sub[sub]["files"].append({"file": r, "lines": total})
                per_sub[sub]["lines"] += total

    # 只对最大的若干文件做 AST 抽类，控制产物体积
    for sub, d in per_sub.items():
        d["files"].sort(key=lambda x: -x["lines"])
        d["n_files"] = len(d["files"])
        for f in d["files"][:12]:
            fp = root / f["file"]
            for c in _classes_in(fp, root):
                if c["lines"] >= 20 or c["n_methods"] >= 3:
                    d["classes"].append(c)
        d["classes"].sort(key=lambda c: -c["lines"])
        d["classes"] = d["classes"][:25]
        d["files"] = d["files"][:25]

    top_tree = sorted(tree_lines.items(), key=lambda kv: -kv[1])[:30]

    return {
        "ref": ref.as_dict() if ref else {},
        "pkg_roots": roots,
        "files_scanned": n_scanned,
        "subsystems": {s: {"n_files": d["n_files"], "lines": d["lines"],
                           "top_files": d["files"], "key_classes": d["classes"]}
                       for s, d in per_sub.items()},
        "dir_tree": [{"dir": k, "lines": v, "files": tree_files[k]} for k, v in top_tree],
    }


def selftest() -> int:
    import tempfile
    ok = True
    # 1) 分类器：一个路径可命中多个子系统，且不能瞎命中
    got = set(_classify("vllm/v1/core/sched/scheduler.py"))
    if "scheduler" not in got:
        print(f"FAIL classify scheduler -> {got}"); ok = False
    got2 = set(_classify("python/sglang/srt/mem_cache/radix_cache.py"))
    if "kv_cache" not in got2:
        print(f"FAIL classify kv_cache -> {got2}"); ok = False
    if _classify("vllm/utils/__init__.py"):
        print(f"FAIL classify should be empty -> {_classify('vllm/utils/__init__.py')}"); ok = False

    # 2) AST 抽类：行数、方法数、基类都要对
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f = root / "sched.py"
        f.write_text(
            "class Scheduler(SchedulerInterface):\n"
            '    """Core scheduler."""\n'
            "    def schedule(self):\n"
            "        pass\n"
            "    def update(self):\n"
            "        pass\n",
            encoding="utf-8")
        cs = _classes_in(f, root)
        if len(cs) != 1:
            print(f"FAIL n classes -> {len(cs)}"); ok = False
        else:
            c = cs[0]
            if c["name"] != "Scheduler" or c["line"] != 1 or c["lines"] != 6:
                print(f"FAIL class meta -> {c}"); ok = False
            if c["n_methods"] != 2 or c["bases"] != ["SchedulerInterface"]:
                print(f"FAIL methods/bases -> {c}"); ok = False
            if c["doc_first_line"] != "Core scheduler.":
                print(f"FAIL docstring -> {c['doc_first_line']}"); ok = False
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    names = [a for a in argv if not a.startswith("-")] or available_engines()
    names = require_engines(names)
    if not names:
        print("没有可分析的引擎。", file=sys.stderr)
        return 2
    out = {}
    for n in names:
        print(f"[struct_map] {n} ...", flush=True)
        out[n] = analyze(n)
    fp = dump_engines("struct_map.json", out)
    print(f"-> {fp}")
    for n, d in out.items():
        subs = d["subsystems"]
        top = sorted(((s, v["lines"]) for s, v in subs.items()), key=lambda kv: -kv[1])[:4]
        desc = "  ".join(f"{s}={v:,}" for s, v in top)
        print(f"  {ENGINES[n]['label']:<14} scanned={d['files_scanned']:>5}  {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
