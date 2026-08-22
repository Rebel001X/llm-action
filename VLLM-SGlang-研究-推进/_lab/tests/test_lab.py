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
import prefix_sim  # noqa: E402
import sched_sim  # noqa: E402
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
    assert prefix_sim.selftest() == 0
    assert sched_sim.selftest() == 0


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


# ------------------------------------------------------ 前缀缓存差分模拟器

def test_chained_hash_kills_everything_after_a_miss():
    """链式哈希的定义性质：中间断一块，后面全部作废。这是 vLLM 与基数树的根本差别。"""
    v = prefix_sim.VllmBlockCache(block_size=4, capacity_blocks=9999)
    base = list(range(40))
    v.process(base)
    # 只改第 21 个 token（落在第 6 块），前 5 块应仍命中，后面全丢
    changed = base[:20] + [777] + base[21:]
    assert v.process(changed) == 20


def test_block_granularity_loses_the_unaligned_tail():
    """共享前缀不是 block_size 整数倍时，块粒度会丢掉尾巴 —— 差异的唯一来源。"""
    a = list(range(100))
    b = list(range(70)) + [777] * 30           # 共享 70，不是 16 的倍数
    v = prefix_sim.VllmBlockCache(block_size=16, capacity_blocks=9999)
    v.process(a)
    assert v.process(b) == 64                  # 70 → 向下取整到 64

    sg = prefix_sim.SglangRadixCache(page_size=1, capacity_tokens=999999)
    sg.process(a)
    assert sg.process(b) == 70                 # page=1 才是 token 粒度


def test_radix_tree_is_not_automatically_token_granular():
    """把 page_size 拉到 16，基数树同样只命中 64 —— 粒度来自 page_size，不是"树"。"""
    a = list(range(100))
    b = list(range(70)) + [777] * 30
    sg = prefix_sim.SglangRadixCache(page_size=16, capacity_tokens=999999)
    sg.process(a)
    assert sg.process(b) == 64


def test_vllm_fine_grained_hashing_closes_the_matching_gap():
    """vLLM 的细粒度 hash 单元能把块粒度损失完全补回来 —— hash 单元=1 时追平基数树。"""
    a = list(range(100))
    b = list(range(70)) + [777] * 30
    v = prefix_sim.VllmBlockCache(block_size=16, capacity_blocks=9999, hash_block_size=1)
    v.process(a)
    assert v.process(b) == 70


def test_vllm_eviction_is_recency_based_not_insertion_fifo():
    """**这条锁住一个我自己犯过的错**。

    vLLM 的空闲队列在命中时 touch() 把块摘掉、释放时追加到队尾
    （block_pool.py:702/:714/:737），所以顺序是「按最近释放」而非「按首次插入」。
    把它建模成 insert-FIFO 会凭空造出 SGLang 的优势 —— 这里用一条构造 trace 钉死：
    反复命中的热前缀在真实策略下必须活下来，在稻草人策略下会被冷流量冲掉。
    """
    hot = list(range(64))
    real = prefix_sim.VllmBlockCache(block_size=16, capacity_blocks=8, evict="freequeue")
    straw = prefix_sim.VllmBlockCache(block_size=16, capacity_blocks=8, evict="insert_fifo")
    for i in range(12):
        for c in (real, straw):
            c.process(hot)                                   # 热前缀反复访问
            c.process([9000 + i * 100 + j for j in range(64)])  # 冷流量冲刷
    assert real.st.hit_tokens > straw.st.hit_tokens, (
        f"真实策略应保住热前缀：real={real.st.hit_tokens} straw={straw.st.hit_tokens}")


def test_no_shared_prefix_means_no_hits_for_either():
    """无共享前缀时两边都必须是 0 —— 否则说明模拟器在自己制造命中。"""
    v = prefix_sim.VllmBlockCache(block_size=16, capacity_blocks=9999)
    sg = prefix_sim.SglangRadixCache(page_size=1, capacity_tokens=999999)
    for t in ([1] * 80, [2] * 80, [3] * 80):
        v.process(t)
        sg.process(t)
    assert v.st.hit_tokens == 0 and sg.st.hit_tokens == 0


def test_workloads_are_deterministic():
    """同 seed 必须逐位可复现，否则实验结论不可复算。"""
    for kind in ("shared_system", "tree_branch", "misaligned_share", "no_share", "hot_cold"):
        assert prefix_sim.gen_workload(kind, 20, 7) == prefix_sim.gen_workload(kind, 20, 7)


@pytest.mark.corpus
def test_prefix_sim_results_match_committed_json():
    """产物里的结论必须能被重跑复现（同 seed 同参数 → 同数字）。"""
    fp = common.OUT / "prefix_sim.json"
    if not fp.exists():
        pytest.skip("先跑 python prefix_sim.py")
    saved = json.loads(fp.read_text(encoding="utf-8"))["results"]
    for r in saved[:3]:
        again = prefix_sim.run_case(r["workload"], r["n_req"], r["block_size"],
                                    r["page_size"], r["capacity_tokens"],
                                    r["hash_block_size"] if r["hash_block_size"] != r["block_size"] else None,
                                    seed=7)
        assert again["vllm"]["hit_tokens"] == r["vllm"]["hit_tokens"], r["workload"]
        assert again["sglang"]["hit_tokens"] == r["sglang"]["hit_tokens"], r["workload"]


# ------------------------------------------------------ 调度公平性差分模拟器

def _gap(policy, seeds=(11, 23, 37, 53, 71, 97), cap=512):
    gs = []
    for sd in seeds:
        proto = sched_sim.gen_arrivals(150, 2, 50, 128, sd)
        gs.append(sched_sim.simulate(proto, sched_sim._arm(policy), cache_tokens=cap).fairness_gap)
    return sum(gs) / len(gs)


def _hit(policy, seeds=(11, 23, 37, 53, 71, 97), cap=512):
    hs = []
    for sd in seeds:
        proto = sched_sim.gen_arrivals(150, 2, 50, 128, sd)
        hs.append(sched_sim.simulate(proto, sched_sim._arm(policy), cache_tokens=cap).hit_rate)
    return sum(hs) / len(hs)


def test_lpm_buys_nothing_when_cache_is_ample():
    """**最强的一条结论**：缓存充裕时 LPM 一分命中率都不多赚，只是重新分配延迟。"""
    proto = sched_sim.gen_arrivals(150, 2, 50, 128, 11)
    a = sched_sim.simulate(proto, "lpm", cache_tokens=65536)
    b = sched_sim.simulate(proto, "fcfs", cache_tokens=65536)
    assert a.hit_rate == b.hit_rate
    assert a.fairness_gap > b.fairness_gap + 50      # 但公平性天差地别


def test_lpm_pays_off_only_under_cache_pressure():
    """LPM 的价值完全是缓存压力的函数。"""
    assert _hit("lpm") > _hit("fcfs") + 0.05


def test_anti_starvation_is_nearly_free_in_hit_rate():
    """给 SGLang 加防饥饿的核心论据：公平性大幅改善，命中率几乎不掉。"""
    h_lpm, h_aged = _hit("lpm"), _hit("aged_T32")
    g_lpm, g_aged = _gap("lpm"), _gap("aged_T32")
    assert g_aged < g_lpm * 0.75, f"公平差应显著下降 {g_lpm} -> {g_aged}"
    assert h_aged > h_lpm - 0.02, f"命中率不该掉太多 {h_lpm} -> {h_aged}"


def test_the_fairness_actually_comes_from_the_tiebreak():
    """**本实验最反直觉的一条**：T 不变、只换并列时的 tiebreak，公平性收益就大部分消失
    —— 真正在防饥饿的是 tiebreak，不是「等得久就加分」这条公式本身。"""
    assert _gap("aged_T32") < _gap("aged_T32_antitie") * 0.8


def test_fine_quantization_loses_the_tiebreak_effect():
    """T 越细 → 并列越少 → tiebreak 越难生效 → 越不公平。上一条的推论。"""
    assert _gap("aged_T32") < _gap("aged_T1")
