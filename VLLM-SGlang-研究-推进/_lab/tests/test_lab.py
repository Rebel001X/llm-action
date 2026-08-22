"""_lab 的回归测试。

分两层，**故意分开**：
  * 结构性测试（不带 mark）—— 只用临时目录造的假仓库，任何机器上都能跑，
    验的是「解析逻辑对不对」；
  * 取证测试（`@pytest.mark.corpus`）—— 需要 `_src/` 真源码存在，
    验的是「产物里的数字自洽不自洽」。没有 _src 就 skip，**不伪装通过**。

跑法： cd _lab && python -m pytest tests -q
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api_surface  # noqa: E402
import common  # noqa: E402
import compare  # noqa: E402
import improve  # noqa: E402
import repo_stats  # noqa: E402
import struct_map  # noqa: E402


# ------------------------------------------------------------------ 结构性

def test_selftests_all_pass():
    """每个脚本的 --selftest 必须绿。它们是各自解析逻辑的第一道闸。"""
    assert repo_stats.selftest() == 0
    assert api_surface.selftest() == 0
    assert struct_map.selftest() == 0
    assert compare.selftest() == 0
    assert improve.selftest() == 0


def test_engine_registry_consistent():
    """登记表与各脚本的路径表要对得上，别出现"登记了但没人认"的引擎。"""
    known = set(common.ENGINES)
    assert set(struct_map.PKG_ROOTS) <= known
    assert set(api_surface.ENTRY_HINTS) <= known
    assert set(api_surface.REGEX_ROUTES) <= known
    # 每个引擎至少要有一种抽 API 的办法，否则它在 api_surface 里必然是空壳
    for n in known:
        assert n in api_surface.ENTRY_HINTS or n in api_surface.REGEX_ROUTES, n


def test_count_lines_comment_and_blank_handling():
    """注释与空行不算代码行 —— 这个口径被正文引用，必须钉死。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.py"
        p.write_text("# c\n\nimport os\n\n# tail\nx = 1\n", encoding="utf-8")
        total, code = common.count_lines(p)
        assert total == 6
        assert code == 2


def test_route_extraction_ignores_non_http_decorators():
    """@app.on_event / @lru_cache 之类不能被误当成路由。"""
    import tempfile
    src = "\n".join([
        "@app.on_event('startup')",
        "async def boot():",
        "    pass",
        "",
        "@functools.lru_cache",
        "def f():",
        "    pass",
        "",
        "@router.post('/v1/x')",
        "def real():",
        "    pass",
    ])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f = root / "a.py"
        f.write_text(src, encoding="utf-8")
        got = api_surface._scan_python(root, [f], want_classes=False, want_flags=False)
        assert [r["path"] for r in got["routes"]] == ["/v1/x"]


def test_route_extraction_reads_path_keyword():
    """FastAPI 也允许 path= 关键字传参，别只认位置参数。"""
    import tempfile
    src = "@router.get(path='/health')\ndef h():\n    pass\n"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f = root / "a.py"
        f.write_text(src, encoding="utf-8")
        got = api_surface._scan_python(root, [f], want_classes=False, want_flags=False)
        assert [r["path"] for r in got["routes"]] == ["/health"]


def test_class_fields_skips_dunder_and_constants():
    """类里的 CONST 和 _private 不是"用户可见字段"，不该进字段表。"""
    src = "\n".join([
        "class C(BaseModel):",
        "    MAX = 5",
        "    _hidden = 1",
        "    model: str",
        "    temperature: float = 1.0",
    ])
    cls = ast.parse(src).body[0]
    names = [f["name"] for f in api_surface._class_fields(cls)]
    assert names == ["model", "temperature"]


def test_subsystem_classifier_is_multilabel_and_precise():
    assert "scheduler" in struct_map._classify("a/core/sched/scheduler.py")
    assert "kv_cache" in struct_map._classify("a/mem_cache/radix_cache.py")
    both = struct_map._classify("a/disaggregation/kv_transfer/conn.py")
    assert "disagg" in both
    # 纯工具文件不该被任何子系统认领
    assert struct_map._classify("a/utils/logger.py") == []


def test_compare_marks_default_difference_not_presence():
    """同名旋钮只有在**默认值真的不同**时才该被标红，别把"都有"当"不同"。"""
    api = {"engines": {
        "a": {"summary": {"unique_paths": []}, "protocol_classes": [], "engine_classes": [],
              "config_classes": [{"name": "EngineArgs", "file": "a", "line": 1, "n_fields": 2,
                                  "fields": [{"name": "k", "default": "1"},
                                             {"name": "same", "default": "'x'"}]}]},
        "b": {"summary": {"unique_paths": []}, "protocol_classes": [], "engine_classes": [],
              "config_classes": [{"name": "ServerArgs", "file": "b", "line": 1, "n_fields": 2,
                                  "fields": [{"name": "k", "default": "2"},
                                             {"name": "same", "default": "'x'"}]}]},
    }}
    c = compare.build(api, {"engines": {}})
    by = {k["knob"]: k["default_differs"] for k in c["shared_knobs"]}
    assert by == {"k": True, "same": False}


def test_compare_refuses_to_judge_non_literal_defaults():
    """vLLM 把默认值转交给子配置（`ModelConfig.dtype`），拿它和字面量比会得出假结论。

    这类必须被判为「不可比」而不是「不同」—— 否则表格会凭空造出几十条假差异。
    """
    assert compare.classify_default("256") == "literal"
    assert compare.classify_default("'auto'") == "literal"
    assert compare.classify_default(None) == "literal"
    assert compare.classify_default("ModelConfig.dtype") == "delegated"
    assert compare.classify_default("dataclasses.field(default_factory=list)") == "factory"
    assert compare.classify_default("get_field(ModelConfig, 'hf_overrides')") == "factory"

    api = {"engines": {
        "a": {"summary": {"unique_paths": []}, "protocol_classes": [], "engine_classes": [],
              "config_classes": [{"name": "EngineArgs", "file": "a", "line": 1, "n_fields": 2,
                                  "fields": [{"name": "dtype", "default": "ModelConfig.dtype"},
                                             {"name": "n", "default": "1"}]}]},
        "b": {"summary": {"unique_paths": []}, "protocol_classes": [], "engine_classes": [],
              "config_classes": [{"name": "ServerArgs", "file": "b", "line": 1, "n_fields": 2,
                                  "fields": [{"name": "dtype", "default": "'auto'"},
                                             {"name": "n", "default": "2"}]}]},
    }}
    rows = {k["knob"]: k for k in compare.build(api, {"engines": {}})["shared_knobs"]}
    assert rows["dtype"]["comparable"] is False
    assert rows["dtype"]["default_differs"] is False, "转交式默认值不许被判成'不同'"
    assert "why_not_comparable" in rows["dtype"]
    assert rows["n"]["comparable"] is True and rows["n"]["default_differs"] is True


def test_compare_openai_field_split_is_disjoint():
    """标准字段与私有字段必须互斥且并集等于全集 —— 防止分类漏项。"""
    api = {"engines": {"a": {
        "summary": {"unique_paths": []}, "config_classes": [], "engine_classes": [],
        "protocol_classes": [{"name": "ChatCompletionRequest", "file": "a", "line": 1,
                              "n_fields": 3,
                              "fields": [{"name": "model", "default": None},
                                         {"name": "top_k", "default": "-1"},
                                         {"name": "stream", "default": "False"}]}]}}}
    d = compare.build(api, {"engines": {}})["chat_request"]["a"]
    assert set(d["openai_standard"]) & set(d["engine_specific"]) == set()
    assert set(d["openai_standard"]) | set(d["engine_specific"]) == set(d["fields"])


# ------------------------------------------------------------------ 取证层

def _out(name: str):
    fp = common.OUT / name
    if not fp.exists():
        pytest.skip(f"缺 {name}，先跑对应脚本")
    return json.loads(fp.read_text(encoding="utf-8"))


@pytest.mark.corpus
def test_every_engine_has_a_pinned_sha():
    """产物里每个引擎都必须带 sha，否则正文的取证基准头无从核对。"""
    data = _out("repo_stats.json")["engines"]
    assert data, "repo_stats.json 是空的"
    for name, d in data.items():
        sha = d["ref"].get("sha", "")
        assert len(sha) >= 7, f"{name} 没有 sha"
        assert d["ref"].get("commit_date"), f"{name} 没有 commit 日期"


@pytest.mark.corpus
def test_line_totals_are_internally_consistent():
    """各语言行数之和必须等于总行数 —— 抓分类遗漏。"""
    for name, d in _out("repo_stats.json")["engines"].items():
        s = sum(l["lines"] for l in d["languages"])
        assert s == d["totals"]["lines_counted"], f"{name}: {s} != {d['totals']['lines_counted']}"
        assert sum(l["files"] for l in d["languages"]) == d["totals"]["files_counted"], name


@pytest.mark.corpus
def test_vllm_and_sglang_expose_openai_chat_endpoint():
    """两个主角必须都抽到 /v1/chat/completions。抽不到 = glob 过期了，是 bug 不是事实。"""
    eng = _out("api_surface.json")["engines"]
    for name in ("vllm", "sglang"):
        if name not in eng:
            pytest.skip(f"{name} 未分析")
        paths = eng[name]["summary"]["unique_paths"]
        assert "/v1/chat/completions" in paths, f"{name} 抽不到 chat 端点，检查 ENTRY_HINTS"


@pytest.mark.corpus
def test_no_engine_silently_yields_empty_api():
    """已登记入口线索的引擎，路由和协议类不能同时为 0 —— 那是 glob 失效的信号。"""
    for name, d in _out("api_surface.json")["engines"].items():
        if name not in api_surface.ENTRY_HINTS:
            continue
        s = d["summary"]
        assert s["n_routes"] + s["n_protocol_classes"] + s["n_cli_flags"] > 0, \
            f"{name} 抽了个空，八成是路径变了"


@pytest.mark.corpus
def test_struct_map_citations_resolve():
    """struct_map 报出的每个类的行号，必须在真实文件里存在 —— 这是正文引用的地基。"""
    data = _out("struct_map.json")["engines"]
    checked = 0
    for name, d in data.items():
        root = common.engine_path(name)
        if root is None:
            continue
        for sub in d["subsystems"].values():
            for c in sub["key_classes"][:5]:
                fp = root / c["file"]
                assert fp.exists(), f"{name}:{c['file']} 不存在"
                n = sum(1 for _ in fp.open("r", encoding="utf-8", errors="replace"))
                assert c["line"] <= n, f"{name}:{c['file']}:{c['line']} 越界（共 {n} 行）"
                checked += 1
    if checked == 0:
        pytest.skip("没有可核的类")


@pytest.mark.corpus
def test_improve_signals_point_at_real_lines():
    """可改进点是要写进正文并被引用的，每条样本的 file:line 必须真实存在。"""
    data = _out("improve.json")["engines"]
    checked = 0
    for name, d in data.items():
        root = common.engine_path(name)
        if root is None:
            continue
        for bucket, items in d["samples"].items():
            for e in items[:4]:
                fp = root / e["file"]
                assert fp.exists(), f"{name}:{e['file']} 不存在（{bucket}）"
                n = sum(1 for _ in fp.open("r", encoding="utf-8", errors="replace"))
                assert e["line"] <= n, f"{name}:{e['file']}:{e['line']} 越界（共 {n} 行，{bucket}）"
                checked += 1
    if checked == 0:
        pytest.skip("没有可核的信号")


@pytest.mark.corpus
def test_improve_density_is_comparable_across_engines():
    """密度口径必须是「每万行」而不是绝对条数，否则大仓库天然吃亏，跨引擎不可比。"""
    for name, d in _out("improve.json")["engines"].items():
        s = d["summary"]
        if not s["python_lines_scanned"]:
            continue
        expect = round(s["silent_except"] * 10000 / s["python_lines_scanned"], 1)
        assert s["silent_except_per_10k_lines"] == expect, name
