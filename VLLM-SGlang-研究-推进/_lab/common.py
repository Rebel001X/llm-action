"""_lab 公共层：定位 _src/ 下的引擎源码、锁定 commit、统一 JSON 落盘。

设计原则（本库红线）：
  1. 任何写进正文的数字，必须由 _lab 里的脚本从**真实源码**算出来，落到 out/*.json；
  2. out/*.json 入库（可审计），_src/ 不入库（体积 & 上游 license）；
  3. 每份产物都带 engine 的 commit sha + 抓取时刻，别人 clone 到同一 sha 可复现。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
SRC = ROOT / "_src"
OUT = LAB / "out"
OUT.mkdir(exist_ok=True)

# 引擎登记表：name -> (相对 _src 的目录, 主要 python 包路径, 展示名)
ENGINES: dict[str, dict] = {
    "vllm":         {"dir": "vllm",         "pkg": "vllm",                  "label": "vLLM"},
    "sglang":       {"dir": "sglang",       "pkg": "python/sglang",         "label": "SGLang"},
    "lmdeploy":     {"dir": "lmdeploy",     "pkg": "lmdeploy",              "label": "LMDeploy"},
    "tgi":          {"dir": "tgi",          "pkg": "server/text_generation_server", "label": "TGI"},
    "lightllm":     {"dir": "lightllm",     "pkg": "lightllm",              "label": "LightLLM"},
    "tensorrt-llm": {"dir": "tensorrt-llm", "pkg": "tensorrt_llm",          "label": "TensorRT-LLM"},
    "dynamo":       {"dir": "dynamo",       "pkg": "components",            "label": "NVIDIA Dynamo"},
    "ktransformers":{"dir": "ktransformers","pkg": "ktransformers",         "label": "KTransformers"},
    "mooncake":     {"dir": "mooncake",     "pkg": "mooncake-wheel",        "label": "Mooncake"},
    "llama.cpp":    {"dir": "llama.cpp",    "pkg": "src",                   "label": "llama.cpp"},
    "mlc-llm":      {"dir": "mlc-llm",      "pkg": "python/mlc_llm",        "label": "MLC-LLM"},
    "tokasaurus":   {"dir": "tokasaurus",   "pkg": "tokasaurus",            "label": "Tokasaurus"},
}

SKIP_DIRS = {".git", "__pycache__", ".github", "node_modules", ".idea", ".vscode",
             ".pytest_cache", ".mypy_cache", "build", "dist", ".eggs"}


@dataclass
class RepoRef:
    """一个引擎源码 checkout 的身份证。"""
    name: str
    label: str
    path: str
    sha: str
    sha_short: str
    commit_date: str
    remote: str

    def as_dict(self) -> dict:
        return asdict(self)


def engine_path(name: str) -> Path | None:
    """源码目录。两种合法形态：git clone（有 .git）或 zipball 解压（有 .clone_meta.json）。

    为什么要两种：LightLLM / Dynamo 这类仓库的文件名里带 `:`（如
    `moe_sum_reduce:v1/...json`），NTFS 不允许，`git checkout` 在读 index 阶段
    就报 invalid path，sparse-checkout 也救不了 —— 只能走 zipball 逐条解压跳过。
    跳过了什么记在 `.clone_skipped.txt`，可审计。
    """
    meta = ENGINES.get(name)
    if not meta:
        return None
    p = SRC / meta["dir"]
    if (p / ".git").exists() or (p / ".clone_meta.json").exists():
        return p
    return None


def available_engines() -> list[str]:
    return [n for n in ENGINES if engine_path(n) is not None]


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), *args],
                              capture_output=True, text=True, timeout=60,
                              encoding="utf-8", errors="replace").stdout.strip()
    except Exception:
        return ""


def repo_ref(name: str) -> RepoRef | None:
    p = engine_path(name)
    if p is None:
        return None
    meta_fp = p / ".clone_meta.json"
    if meta_fp.exists():
        m = json.loads(meta_fp.read_text(encoding="utf-8"))
        return RepoRef(
            name=name, label=ENGINES[name]["label"], path=str(p),
            sha=m["sha"], sha_short=m["sha_short"],
            commit_date=m["commit_date"],
            remote=f"https://github.com/{m['repo']}.git (zipball)",
        )
    sha = _git(p, "rev-parse", "HEAD")
    return RepoRef(
        name=name,
        label=ENGINES[name]["label"],
        path=str(p),
        sha=sha,
        sha_short=sha[:8],
        commit_date=_git(p, "log", "-1", "--format=%cI"),
        remote=_git(p, "config", "--get", "remote.origin.url"),
    )


def walk_files(root: Path, suffixes: Iterable[str] | None = None) -> Iterator[Path]:
    """遍历仓库文件，跳过 .git / 构建产物。suffixes 为 None 表示全部。"""
    sufs = set(suffixes) if suffixes else None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.startswith(".clone_"):   # 我们自己放的取证元数据，不算上游代码
                continue
            if sufs is not None and Path(fn).suffix.lower() not in sufs:
                continue
            yield Path(dirpath) / fn


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def count_lines(p: Path) -> tuple[int, int]:
    """返回 (总行数, 非空非纯注释行数)。注释判定按 # 与 // 两种前缀，够用即可。"""
    text = read_text(p)
    if not text:
        return 0, 0
    total = 0
    code = 0
    for line in text.splitlines():
        total += 1
        s = line.strip()
        if not s or s.startswith(("#", "//", "/*", "*", "*/")):
            continue
        code += 1
    return total, code


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def dump(name: str, payload: dict) -> Path:
    """落盘 JSON。键排序 + 固定缩进，保证同输入同输出（diff 友好）。"""
    fp = OUT / name
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
    return fp


def load(name: str) -> dict:
    fp = OUT / name
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def dump_engines(name: str, engines: dict) -> Path:
    """按引擎**增量合并**后落盘。

    为什么不能直接覆盖：`python struct_map.py ktransformers` 这种只跑一个引擎的用法很常见，
    如果整份覆盖，其余 11 个引擎的结果会被静默清空 —— 而下游 compare.py 读到的
    「只有 2 个引擎」看起来像事实，其实是被自己的工具擦掉的。这个坑真踩过一次。
    """
    prev = load(name).get("engines", {})
    prev.update(engines)
    return dump(name, {"engines": prev})


def require_engines(names: list[str]) -> list[str]:
    """过滤出真的 clone 下来的引擎，并把缺的打到 stderr —— 缺就是缺，不假装有。"""
    have, missing = [], []
    for n in names:
        (have if engine_path(n) else missing).append(n)
    if missing:
        print(f"[warn] 未找到源码，跳过: {', '.join(missing)}", file=sys.stderr)
    return have
