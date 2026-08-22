"""improve.py —— 从源码里挖「可改进点」的**客观信号**，供 05-改进机会/ 取证。

为什么要有它：谈"哪里能改进"最容易变成拍脑袋的意见。这里只收**能指到行**的信号，
每条都带 `文件:行` 和原文片段，正文引用时可被 _verify.py 核对。

收六类信号（都用 AST，不用正则猜）：
  1. todo          —— TODO / FIXME / HACK / XXX 注释（带认领人的单独标出来）
  2. not_impl      —— raise NotImplementedError（抽象接口没铺满的地方）
  3. deprecated    —— DeprecationWarning / @deprecated / 名字里带 legacy|_v1|old_
  4. silent_except —— except 后面只有 pass / continue / return None：**静默吞异常**
  5. bare_except   —— except: 或 except Exception: 不重抛也不记日志
  6. god_file      —— 单文件超过阈值行数（改一处要读一万行的地方）

第 4、5 类是重点：推理引擎里"静默降级"比"报错"危险得多 ——
用户看不见自己掉进了慢路径或错路径。

用法:
    python improve.py                 # 全部引擎
    python improve.py vllm sglang
    python improve.py --selftest
"""
from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import (ENGINES, available_engines, dump_engines, engine_path,
                    read_text, rel, repo_ref, require_engines, walk_files)
from struct_map import EXCLUDE_PARTS, PKG_ROOTS, _classify

# 「探测型」文件：吞异常在这里是**正当**的 —— 探测可选依赖/硬件/环境本来就该失败即跳过。
# 不把它们摘出去，密度指标会被这类文件主导，跨引擎排名毫无意义。
# （这条是跑完第一版之后发现的：vLLM/SGLang 的静默 except 热点前几名全是
#  collect_env.py / check_env.py / platforms/cuda.py 这种探测文件。）
PROBE_PATTERNS = (
    "check_env", "collect_env", "/platforms/", "_ops.py", "/utils/",
    "utils.py", "env_override", "version", "/compat", "import_utils",
    "device_utils", "/logger", "diagnos", "_probe", "capability",
)

# 「热路径」子系统：请求真正流经的地方。这里静默吞异常 = 用户看不见的降级或错答案。
HOT_SUBSYSTEMS = {"scheduler", "kv_cache", "attention", "model_exec",
                  "disagg", "speculative", "structured_output"}


def path_role(relpath: str, exc: str | None = None) -> str:
    """给一处静默 except 定角色。

    probe —— 吞它是正当的：探测型文件，或**捕的就是 ImportError**（可选依赖）。
             `except ImportError: pass` 是 Python 生态里处理可选依赖的标准写法，
             哪怕它出现在热路径文件的模块头部，也不该算进"危险的静默降级"。
    hot   —— 请求真正流经的子系统里，吞掉的是运行时异常：用户看不见的降级或错答案。
    other —— 其余。

    这两层过滤是跑完前两版之后加的：第一版按全仓密度排名，前几名全是
    check_env/collect_env；第二版按文件角色过滤后，剩下的又大半是 ImportError。
    每过滤一层，这个指标才更接近"真的值得去看的地方"。
    """
    low = relpath.lower()
    if exc and ("ImportError" in exc or "ModuleNotFoundError" in exc):
        return "probe"
    if any(p in low for p in PROBE_PATTERNS):
        return "probe"
    if HOT_SUBSYSTEMS & set(_classify(relpath)):
        return "hot"
    return "other"

TODO_TAGS = ("TODO", "FIXME", "HACK", "XXX")
GOD_FILE_LINES = 3000
MAX_SAMPLES = 25


def _todo_comments(text: str, r: str) -> list[dict]:
    """扫注释里的 TODO 系标记。只看注释行，避免把字符串常量算进来。"""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if "#" not in line:
            continue
        comment = line[line.index("#"):]
        for tag in TODO_TAGS:
            if tag not in comment:
                continue
            body = " ".join(comment.split())[:180]
            # TODO(alice): ... 这种带认领人的，比裸 TODO 有价值得多
            owned = f"{tag}(" in comment
            out.append({"tag": tag, "line": i, "file": r, "owned": owned, "text": body})
            break
    return out


def _is_silent_handler(handler: ast.ExceptHandler) -> str | None:
    """判断 except 块是不是「吞掉了异常」。返回吞的方式，或 None。"""
    body = handler.body
    # 只有一条语句，且是 pass / continue / return None / return
    if len(body) == 1:
        s = body[0]
        if isinstance(s, ast.Pass):
            return "pass"
        if isinstance(s, ast.Continue):
            return "continue"
        if isinstance(s, ast.Return) and (s.value is None
                                          or (isinstance(s.value, ast.Constant) and s.value.value is None)):
            return "return None"
    # 多条语句：只要整块里既没有 raise、也没有任何 log/print/warn 调用，就算吞
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(handler))
    if has_raise:
        return None
    for n in ast.walk(handler):
        if isinstance(n, ast.Call):
            fn = n.func
            name = (fn.attr if isinstance(fn, ast.Attribute) else
                    fn.id if isinstance(fn, ast.Name) else "")
            if any(k in name.lower() for k in ("log", "warn", "print", "error", "debug",
                                               "info", "exception", "critical")):
                return None
    return "no-raise-no-log" if body else "empty"


def _exc_name(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "<bare>"
    try:
        return ast.unparse(handler.type)
    except Exception:
        return "<?>"


def _scan_file(fp: Path, root: Path) -> dict:
    r = rel(fp, root)
    text = read_text(fp)
    if not text:
        return {}
    n_lines = text.count("\n") + 1
    res: dict = {"todo": _todo_comments(text, r), "not_impl": [], "deprecated": [],
                 "silent_except": [], "bare_except": [], "god_file": [], "lines": n_lines}
    if n_lines >= GOD_FILE_LINES:
        res["god_file"].append({"file": r, "line": 1, "lines": n_lines})
    try:
        tree = ast.parse(text, filename=str(fp))
    except SyntaxError:
        return res

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            exc = node.exc
            nm = ""
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                nm = exc.func.id
            elif isinstance(exc, ast.Name):
                nm = exc.id
            if nm == "NotImplementedError":
                res["not_impl"].append({"file": r, "line": node.lineno})
        elif isinstance(node, ast.ExceptHandler):
            kind = _is_silent_handler(node)
            name = _exc_name(node)
            entry = {"file": r, "line": node.lineno, "exc": name}
            if kind:
                entry["swallow"] = kind
                res["silent_except"].append(entry)
            if name in ("<bare>", "Exception", "BaseException"):
                res["bare_except"].append(entry)
        elif isinstance(node, ast.Call):
            fn = node.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if nm == "warn":
                for a in node.args + [k.value for k in node.keywords]:
                    if isinstance(a, ast.Name) and "Deprecat" in a.id:
                        res["deprecated"].append({"file": r, "line": node.lineno,
                                                  "kind": "warnings.warn"})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in node.decorator_list:
                try:
                    s = ast.unparse(d)
                except Exception:
                    continue
                if "deprecat" in s.lower():
                    res["deprecated"].append({"file": r, "line": node.lineno,
                                              "kind": f"@{s[:60]}", "name": node.name})
    return res


def analyze(name: str) -> dict:
    root = engine_path(name)
    assert root is not None
    ref = repo_ref(name)
    buckets: dict[str, list] = defaultdict(list)
    file_lines: dict[str, int] = {}
    files_scanned = 0
    total_lines = 0

    for sub_root in PKG_ROOTS.get(name, ["."]):
        base = root / sub_root
        if not base.exists():
            continue
        for fp in walk_files(base, {".py"}):
            r = rel(fp, root)
            if any(x in "/" + r.lower() for x in EXCLUDE_PARTS):
                continue
            got = _scan_file(fp, root)
            if not got:
                continue
            files_scanned += 1
            total_lines += got["lines"]
            file_lines[r] = got["lines"]
            for k in ("todo", "not_impl", "deprecated", "silent_except",
                      "bare_except", "god_file"):
                buckets[k].extend(got[k])

    # 给每条静默 except 打上角色标签，并按角色分桶统计
    for e in buckets["silent_except"]:
        e["role"] = path_role(e["file"], e.get("exc"))
    by_role = Counter(e["role"] for e in buckets["silent_except"])
    hot_items = [e for e in buckets["silent_except"] if e["role"] == "hot"]
    hot_lines = sum(l for f, l in file_lines.items() if path_role(f) == "hot")
    # 注意：热路径行数按**文件**判定（不传 exc），命中数按**每处**判定（传 exc）。
    # 分母是"热路径文件有多大"，分子是"其中真正可疑的处数"，口径不同是有意的。

    summary = {k: len(v) for k, v in buckets.items()}
    summary["files_scanned"] = files_scanned
    summary["python_lines_scanned"] = total_lines
    summary["hot_path_lines_scanned"] = hot_lines
    per10k = (lambda n, d: round(n * 10000 / d, 1) if d else None)
    summary["silent_except_per_10k_lines"] = per10k(len(buckets["silent_except"]), total_lines)
    summary["silent_except_by_role"] = dict(by_role)
    # **这个才是有意义的指标**：热路径上每万行有多少处静默吞异常。
    # 全仓密度会被 check_env/collect_env 这类探测文件主导，排名没有意义。
    summary["silent_except_hot_per_10k_hot_lines"] = per10k(len(hot_items), hot_lines)
    summary["todo_per_10k_lines"] = per10k(len(buckets["todo"]), total_lines)
    summary["todo_owned"] = sum(1 for t in buckets["todo"] if t["owned"])

    top_all = Counter(e["file"] for e in buckets["silent_except"]).most_common(10)
    top_hot = Counter(e["file"] for e in hot_items).most_common(10)

    return {
        "ref": ref.as_dict() if ref else {},
        "summary": summary,
        "hotspots_silent_except": [{"file": f, "count": c} for f, c in top_all],
        "hotspots_silent_except_hot_path": [{"file": f, "count": c} for f, c in top_hot],
        "samples": {k: v[:MAX_SAMPLES] for k, v in buckets.items()},
        "samples_silent_except_hot_path": hot_items[:MAX_SAMPLES],
        "god_files": sorted(buckets["god_file"], key=lambda x: -x["lines"])[:20],
        "caveat": (
            "静默 except 是**信号不是判决**。第一版只看全仓密度是错的：热点前几名全是 "
            "check_env / collect_env / platforms 这类**探测型**文件，那里吞异常本来就正当。"
            "所以本产物把每条按 role 分成 probe / hot / other，"
            "真正该看的是 `silent_except_hot_per_10k_hot_lines`（热路径口径）。"
            "即便是热路径命中，写进正文前也必须逐条打开看上下文。"),
    }


def selftest() -> int:
    import tempfile
    ok = True
    src = "\n".join([
        "import warnings",
        "def f():",
        "    try:",
        "        g()",
        "    except ValueError:",
        "        pass",                      # silent, 非 bare
        "def h():",
        "    try:",
        "        g()",
        "    except Exception as e:",
        "        logger.warning(e)",         # bare 但不静默
        "def i():",
        "    try:",
        "        g()",
        "    except Exception:",
        "        raise RuntimeError",        # 既不静默，重抛了
        "def j():",
        "    raise NotImplementedError",
        "# TODO(alice): fix this",
        "# FIXME bare one",
        "x = 'TODO in a string, must not count'",
    ])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fp = root / "m.py"
        fp.write_text(src, encoding="utf-8")
        got = _scan_file(fp, root)

        if len(got["silent_except"]) != 1 or got["silent_except"][0]["swallow"] != "pass":
            print(f"FAIL silent_except -> {got['silent_except']}"); ok = False
        if len(got["bare_except"]) != 2:      # h() 和 i() 都是 except Exception
            print(f"FAIL bare_except -> {got['bare_except']}"); ok = False
        if len(got["not_impl"]) != 1:
            print(f"FAIL not_impl -> {got['not_impl']}"); ok = False
        tags = sorted(t["tag"] for t in got["todo"])
        if tags != ["FIXME", "TODO"]:
            print(f"FAIL todo tags -> {tags}（字符串里的 TODO 不该被算进来）"); ok = False
        owned = {t["tag"]: t["owned"] for t in got["todo"]}
        if owned.get("TODO") is not True or owned.get("FIXME") is not False:
            print(f"FAIL todo owned -> {owned}"); ok = False
        if got["god_file"]:
            print("FAIL 小文件不该被判为 god_file"); ok = False

    # 角色判定：ImportError 一律 probe（可选依赖），哪怕在热路径文件里
    if path_role("vllm/v1/core/sched/scheduler.py", "ImportError") != "probe":
        print("FAIL ImportError 应判 probe"); ok = False
    if path_role("vllm/v1/core/sched/scheduler.py", "ValueError") != "hot":
        print("FAIL 调度器里的 ValueError 应判 hot"); ok = False
    if path_role("vllm/collect_env.py", "ValueError") != "probe":
        print("FAIL 探测型文件应判 probe"); ok = False
    if path_role("vllm/entrypoints/cli/main.py", "ValueError") != "other":
        print(f"FAIL 非热非探测应判 other -> {path_role('vllm/entrypoints/cli/main.py', 'ValueError')}"); ok = False

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
        print(f"[improve] {n} ...", flush=True)
        out[n] = analyze(n)
    fp = dump_engines("improve.json", out)
    print(f"-> {fp}")
    print(f"  {'引擎':<14}{'静默except':>10}{'其中热路径':>11}{'热路径/万行':>12}"
          f"{'TODO':>7}{'(认领)':>7}{'未实现':>7}{'超大文件':>9}")
    for n, d in out.items():
        s = d["summary"]
        hot = s["silent_except_by_role"].get("hot", 0)
        print(f"  {ENGINES[n]['label']:<14}{s['silent_except']:>10}{hot:>11}"
              f"{str(s['silent_except_hot_per_10k_hot_lines']):>12}"
              f"{s['todo']:>7}{s['todo_owned']:>7}{s['not_impl']:>7}{s['god_file']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
