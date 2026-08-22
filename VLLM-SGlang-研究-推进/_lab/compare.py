"""compare.py —— 把各引擎的抽取结果对齐成「可直接贴进正文的横向对比表」。

三张表：
  A. 路由表     —— 谁有 /v1/xxx，谁没有；谁有独家的非标准端点
  B. 请求字段表 —— ChatCompletionRequest 这一个类，各家到底认哪些字段
                   （这是"OpenAI 兼容"这句话唯一能被证伪的地方）
  C. 旋钮表     —— 服务端配置对象的字段数与重名字段的默认值差异

铁律：这里只做**集合运算**，不做任何"谁更好"的判断。
      判断留给正文，且必须能追回本表。

依赖 api_surface.json / repo_stats.json，先跑它们。

用法:
    python compare.py                 # 生成 compare.json + compare.md
    python compare.py --selftest
"""
from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict

from common import ENGINES, OUT, dump, load

# 各家 chat 请求体的类名（同一个概念，各家叫法一致，但字段集不同）
CHAT_REQ_NAMES = ["ChatCompletionRequest"]
COMPLETION_REQ_NAMES = ["CompletionRequest"]
SAMPLING_NAMES = ["SamplingParams", "GenerationConfig"]
SERVER_CFG_NAMES = ["EngineArgs", "ServerArgs", "PytorchEngineConfig",
                    "TurbomindEngineConfig", "LlmArgs", "Config"]

# OpenAI 官方 chat/completions 请求体的字段（用于区分"标准字段"与"各家私货"）。
# 来源：platform.openai.com/docs/api-reference/chat/create（2026-08 查阅）。
# 这份清单是**人工录入的外部事实**，不是从源码抽的 —— 在产物里单独标注，别混淆证据等级。
OPENAI_CHAT_FIELDS = {
    "model", "messages", "frequency_penalty", "logit_bias", "logprobs",
    "top_logprobs", "max_tokens", "max_completion_tokens", "n", "modalities",
    "prediction", "audio", "presence_penalty", "response_format", "seed",
    "service_tier", "stop", "store", "stream", "stream_options", "temperature",
    "top_p", "tools", "tool_choice", "parallel_tool_calls", "user",
    "function_call", "functions", "metadata", "reasoning_effort",
}


def classify_default(expr: str | None) -> str:
    """给「默认值表达式」定性，决定它能不能拿来跨引擎比。

    三类：
      literal   —— 常量（`256` / `'auto'` / `False` / `None`），**唯一可比的**
      factory   —— `dataclasses.field(default_factory=list)` / `get_field(...)`，值要运行才知道
      delegated —— `ModelConfig.dtype` 这种把默认值转交给子配置对象的写法

    为什么必须分：vLLM 的 `EngineArgs` 大量用 delegated 写法，
    直接和 SGLang 的字面量比，会得到「41 个旋钮默认值不同」这种**假结论** ——
    实际上多数只是"一个写字面量、一个写引用"，值本身可能完全一样。
    真要比，得去把被引用的那个 Config 展开，本工具不做，就老实标出来不比。
    """
    if expr is None:
        return "literal"                       # 无默认值，视作 None，可比
    e = expr.strip()
    if e in ("None", "True", "False") or e.startswith(("'", '"')):
        return "literal"
    try:
        ast.literal_eval(e)
        return "literal"
    except (ValueError, SyntaxError):
        pass
    if "field(" in e or "get_field(" in e or "factory" in e:
        return "factory"
    if re.match(r"^[A-Z]\w*(Config|Args)\.\w+$", e):
        return "delegated"
    return "expression"


def _find_class(engine_data: dict, names: list[str], buckets=("protocol_classes",
                                                             "config_classes",
                                                             "engine_classes")) -> dict | None:
    best = None
    for bucket in buckets:
        for c in engine_data.get(bucket, []):
            if c["name"] in names:
                if best is None or c["n_fields"] > best["n_fields"]:
                    best = c
    return best


def build(api: dict, stats: dict) -> dict:
    engines = api.get("engines", {})
    names = list(engines)

    # ---- A. 路由 ----------------------------------------------------------
    route_owner: dict[str, set[str]] = defaultdict(set)
    for n, d in engines.items():
        for p in d.get("summary", {}).get("unique_paths", []):
            route_owner[p].add(n)
    routes = []
    for p in sorted(route_owner):
        owners = sorted(route_owner[p])
        routes.append({"path": p, "engines": owners, "n": len(owners),
                       "exclusive_to": owners[0] if len(owners) == 1 else None})

    # ---- B. chat 请求字段 --------------------------------------------------
    chat_fields: dict[str, dict] = {}
    for n, d in engines.items():
        c = _find_class(d, CHAT_REQ_NAMES)
        if not c:
            continue
        fields = {f["name"]: f for f in c["fields"]}
        chat_fields[n] = {
            "class": c["name"], "file": c["file"], "line": c["line"],
            "n_fields": len(fields),
            "fields": sorted(fields),
            "openai_standard": sorted(set(fields) & OPENAI_CHAT_FIELDS),
            "engine_specific": sorted(set(fields) - OPENAI_CHAT_FIELDS),
            "missing_vs_openai": sorted(OPENAI_CHAT_FIELDS - set(fields)),
        }
    field_owner: dict[str, set[str]] = defaultdict(set)
    for n, d in chat_fields.items():
        for f in d["fields"]:
            field_owner[f].add(n)
    field_matrix = [{"field": f, "engines": sorted(v), "n": len(v),
                     "is_openai_standard": f in OPENAI_CHAT_FIELDS}
                    for f, v in sorted(field_owner.items())]

    # ---- C. 服务端旋钮 -----------------------------------------------------
    knobs: dict[str, dict] = {}
    for n, d in engines.items():
        c = _find_class(d, SERVER_CFG_NAMES, buckets=("config_classes",))
        if not c:
            continue
        knobs[n] = {"class": c["name"], "file": c["file"], "line": c["line"],
                    "n_fields": c["n_fields"],
                    "defaults": {f["name"]: f["default"] for f in c["fields"]}}
    knob_owner: dict[str, set[str]] = defaultdict(set)
    for n, d in knobs.items():
        for k in d["defaults"]:
            knob_owner[k].add(n)
    shared_knobs = []
    for k, owners in sorted(knob_owner.items()):
        if len(owners) < 2:
            continue
        defaults = {n: knobs[n]["defaults"].get(k) for n in sorted(owners)}
        kinds = {n: classify_default(v) for n, v in defaults.items()}
        row = {"knob": k, "engines": sorted(owners), "defaults": defaults,
               "default_kinds": kinds}
        # 只有**所有参与方都是字面量**时，"默认值不同"这句话才成立
        comparable = all(v == "literal" for v in kinds.values())
        row["comparable"] = comparable
        row["default_differs"] = comparable and len(set(defaults.values())) > 1
        if not comparable:
            row["why_not_comparable"] = (
                "至少一方的默认值不是字面量（"
                + ", ".join(f"{n}={kinds[n]}" for n in sorted(kinds) if kinds[n] != "literal")
                + "），跨引擎比字面量会得出假结论，故不判定")
        shared_knobs.append(row)

    # ---- D. 规模 -----------------------------------------------------------
    scale = []
    for n, d in stats.get("engines", {}).items():
        t = d["totals"]
        langs = {l["lang"]: l["lines"] for l in d["languages"]}
        scale.append({
            "engine": n, "label": ENGINES.get(n, {}).get("label", n),
            "sha": d["ref"].get("sha_short"), "date": d["ref"].get("commit_date"),
            "lines_total": t["lines_counted"],
            "python_src": t["python_src_lines"],
            "python_test": t["python_test_lines"],
            "test_ratio": t["test_to_src_ratio"],
            "cuda_files": d["kernels"]["cuda_files"],
            "triton_files": d["kernels"]["triton_jit_files"],
            "rust_lines": langs.get("Rust", 0),
            "cpp_lines": langs.get("C++", 0) + langs.get("C/C++ header", 0),
        })
    scale.sort(key=lambda r: -r["lines_total"])

    return {
        "engines_compared": names,
        "routes": routes,
        "chat_request": chat_fields,
        "chat_field_matrix": field_matrix,
        "server_knobs": {n: {k: v for k, v in d.items() if k != "defaults"}
                         for n, d in knobs.items()},
        "shared_knobs": shared_knobs,
        "scale": scale,
        "notes": {
            "openai_field_list_source": "platform.openai.com/docs/api-reference/chat/create, 2026-08 人工录入",
            "caveat": "路由集合来自装饰器静态抽取；条件注册（if 分支里 add_api_route）可能漏。",
        },
    }


def to_markdown(cmp: dict) -> str:
    L: list[str] = []
    L.append("# 横向对比·机器生成表（compare.py 产物，勿手改）\n")
    L.append(f"对比引擎：{', '.join(cmp['engines_compared'])}\n")

    L.append("\n## A. 工程规模\n")
    L.append("| 引擎 | commit | 总行数 | Python 产品码 | Python 测试码 | 测试/产品 | CUDA 文件 | Triton 文件 | Rust 行 | C/C++ 行 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in cmp["scale"]:
        L.append(f"| {r['label']} | `{r['sha']}` | {r['lines_total']:,} | {r['python_src']:,} | "
                 f"{r['python_test']:,} | {r['test_ratio']} | {r['cuda_files']} | "
                 f"{r['triton_files']} | {r['rust_lines']:,} | {r['cpp_lines']:,} |")

    L.append("\n## B. `/v1/*` 路由覆盖\n")
    v1 = [r for r in cmp["routes"] if r["path"].startswith("/v1/")]
    engines = cmp["engines_compared"]
    L.append("| 路径 | " + " | ".join(engines) + " |")
    L.append("|---|" + "---|" * len(engines))
    for r in v1:
        cells = ["✅" if e in r["engines"] else "—" for e in engines]
        L.append(f"| `{r['path']}` | " + " | ".join(cells) + " |")

    L.append("\n## C. ChatCompletionRequest 字段数\n")
    L.append("| 引擎 | 类 | 文件:行 | 字段总数 | 其中 OpenAI 标准 | 引擎私有 | 缺席的 OpenAI 字段 |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for n, d in cmp["chat_request"].items():
        L.append(f"| {ENGINES.get(n, {}).get('label', n)} | `{d['class']}` | `{d['file']}:{d['line']}` | "
                 f"{d['n_fields']} | {len(d['openai_standard'])} | {len(d['engine_specific'])} | "
                 f"{len(d['missing_vs_openai'])} |")

    L.append("\n## D. 服务端配置对象\n")
    L.append("| 引擎 | 类 | 文件:行 | 字段数 |")
    L.append("|---|---|---|---:|")
    for n, d in cmp["server_knobs"].items():
        L.append(f"| {ENGINES.get(n, {}).get('label', n)} | `{d['class']}` | "
                 f"`{d['file']}:{d['line']}` | {d['n_fields']} |")

    diff = [k for k in cmp["shared_knobs"] if k["default_differs"]]
    same = [k for k in cmp["shared_knobs"] if k["comparable"] and not k["default_differs"]]
    skipped = [k for k in cmp["shared_knobs"] if not k["comparable"]]
    L.append(f"\n## E. 同名旋钮的默认值（可比 {len(diff) + len(same)} 个：不同 {len(diff)}、"
             f"相同 {len(same)}；**不可比 {len(skipped)} 个已排除**）\n")
    L.append("> 口径：只有当**所有参与引擎的默认值都是字面量**时才判定异同。"
             "vLLM 的 `EngineArgs` 大量把默认值转交给子配置（写成 `ModelConfig.dtype` 这种），"
             "拿它和别家的字面量直接比会得出假结论，所以这类一律排除、不下判断。\n")
    if diff:
        L.append("| 旋钮 | " + " | ".join(engines) + " |")
        L.append("|---|" + "---|" * len(engines))
        for k in diff[:60]:
            cells = [str(k["defaults"].get(e, "—")) for e in engines]
            L.append(f"| `{k['knob']}` | " + " | ".join(cells) + " |")
        if len(diff) > 60:
            L.append(f"\n> 表已截断，完整 {len(diff)} 行见 `compare.json` 的 `shared_knobs`。")
    else:
        L.append("（本次没有可比且不同的旋钮。）")
    L.append(f"\n被排除的 {len(skipped)} 个旋钮及排除原因见 `compare.json` 的 "
             "`shared_knobs[*].why_not_comparable`。")

    L.append(f"\n---\n\n> 口径说明：{cmp['notes']['caveat']}  \n"
             f"> OpenAI 字段清单来源：{cmp['notes']['openai_field_list_source']}\n")
    return "\n".join(L) + "\n"


def selftest() -> int:
    ok = True
    api = {"engines": {
        "a": {"summary": {"unique_paths": ["/v1/chat/completions", "/health", "/a_only"]},
              "protocol_classes": [{"name": "ChatCompletionRequest", "file": "a.py", "line": 1,
                                    "n_fields": 3,
                                    "fields": [{"name": "model", "default": None},
                                               {"name": "temperature", "default": "1.0"},
                                               {"name": "top_k", "default": "-1"}]}],
              "config_classes": [{"name": "EngineArgs", "file": "c.py", "line": 5, "n_fields": 2,
                                  "fields": [{"name": "max_num_seqs", "default": "256"},
                                             {"name": "dtype", "default": "'auto'"}]}],
              "engine_classes": []},
        "b": {"summary": {"unique_paths": ["/v1/chat/completions", "/health"]},
              "protocol_classes": [{"name": "ChatCompletionRequest", "file": "b.py", "line": 2,
                                    "n_fields": 2,
                                    "fields": [{"name": "model", "default": None},
                                               {"name": "temperature", "default": "0.7"}]}],
              "config_classes": [{"name": "ServerArgs", "file": "d.py", "line": 9, "n_fields": 2,
                                  "fields": [{"name": "max_num_seqs", "default": "128"},
                                             {"name": "tp_size", "default": "1"}]}],
              "engine_classes": []},
    }}
    stats = {"engines": {
        "a": {"ref": {"sha_short": "aaa"}, "totals": {"lines_counted": 100, "python_src_lines": 80,
                                                      "python_test_lines": 20, "test_to_src_ratio": 0.25},
              "languages": [{"lang": "Python", "lines": 100}],
              "kernels": {"cuda_files": 1, "triton_jit_files": 2}},
    }}
    c = build(api, stats)

    excl = {r["path"]: r["exclusive_to"] for r in c["routes"]}
    if excl.get("/a_only") != "a":
        print(f"FAIL exclusive route -> {excl}"); ok = False
    if excl.get("/health") is not None:
        print("FAIL shared route marked exclusive"); ok = False

    ca = c["chat_request"]["a"]
    if ca["engine_specific"] != ["top_k"]:
        print(f"FAIL engine_specific -> {ca['engine_specific']}"); ok = False
    if "model" not in ca["openai_standard"] or "temperature" not in ca["openai_standard"]:
        print(f"FAIL openai_standard -> {ca['openai_standard']}"); ok = False
    if "seed" not in ca["missing_vs_openai"]:
        print("FAIL missing_vs_openai should include seed"); ok = False

    shared = {k["knob"]: k for k in c["shared_knobs"]}
    if "max_num_seqs" not in shared:
        print(f"FAIL shared knob missing -> {list(shared)}"); ok = False
    elif not shared["max_num_seqs"]["default_differs"]:
        print("FAIL 256 vs 128 应判为不同"); ok = False
    if "dtype" in shared:
        print("FAIL 单引擎旋钮不该进 shared"); ok = False

    md = to_markdown(c)
    for must in ("## A. 工程规模", "## C. ChatCompletionRequest", "max_num_seqs"):
        if must not in md:
            print(f"FAIL markdown missing {must}"); ok = False

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    api = load("api_surface.json")
    stats = load("repo_stats.json")
    if not api:
        print("缺 api_surface.json，先跑 python api_surface.py", file=sys.stderr)
        return 2
    c = build(api, stats)
    dump("compare.json", c)
    md = OUT / "compare.md"
    md.write_text(to_markdown(c), encoding="utf-8")
    print(f"-> {OUT / 'compare.json'}")
    print(f"-> {md}")
    print(f"  引擎 {len(c['engines_compared'])} 个；路由 {len(c['routes'])} 条；"
          f"chat 字段并集 {len(c['chat_field_matrix'])} 个；"
          f"同名不同默认值旋钮 {sum(1 for k in c['shared_knobs'] if k['default_differs'])} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
