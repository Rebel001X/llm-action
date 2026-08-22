"""api_surface.py —— 用 AST 从源码里抽「对外 API 表面」，不靠读文档、不靠记忆。

抽四类东西：
  1. HTTP 路由      —— FastAPI/starlette 的 @app.post("/v1/...") 装饰器
  2. 协议模型       —— pydantic BaseModel 子类的字段名/类型/默认值（OpenAI 兼容层的真实字段集）
  3. 配置对象       —— dataclass / pydantic 的 EngineArgs、ServerArgs、SamplingParams 等
  4. CLI 开关       —— argparse 的 add_argument("--xxx")

非 Python 引擎（TGI 的 Rust router、llama.cpp 的 C++ server）走正则兜底，
产物里用 method="regex" 明确标注，**不冒充 AST 精度**。

用法:
    python api_surface.py             # 全部
    python api_surface.py vllm sglang
    python api_surface.py --selftest
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from common import (ENGINES, available_engines, dump, dump_engines, engine_path, read_text,
                    rel, repo_ref, require_engines)

HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options",
                "api_route", "websocket"}

# 每个引擎的「入口文件线索」：路由文件 / 协议文件 / 配置文件。
# 用 glob 匹配；上游挪了文件就匹配不到，那就诚实地记 0，不猜。
ENTRY_HINTS: dict[str, dict[str, list[str]]] = {
    # 2026 版 vLLM 把 entrypoints 拆成了「每个端点一个包」：
    #   vllm/entrypoints/<family>/<endpoint>/{api_router,protocol,serving}.py
    # 旧的单文件 openai/protocol.py 已经不存在，hints 必须跟着走。
    "vllm": {
        "routes": ["vllm/entrypoints/**/api_router.py",
                   "vllm/entrypoints/openai/api_server.py",
                   "vllm/entrypoints/**/api_server.py"],
        "protocol": ["vllm/entrypoints/**/protocol.py"],
        "config": ["vllm/engine/arg_utils.py", "vllm/sampling_params.py",
                   "vllm/entrypoints/openai/cli_args.py",
                   "vllm/config/*.py", "vllm/config.py"],
        "pyapi": ["vllm/entrypoints/llm.py", "vllm/v1/engine/async_llm.py",
                  "vllm/engine/async_llm_engine.py"],
    },
    "sglang": {
        "routes": ["python/sglang/srt/entrypoints/http_server.py",
                   "python/sglang/srt/entrypoints/**/*server*.py"],
        "protocol": ["python/sglang/srt/entrypoints/openai/protocol.py",
                     "python/sglang/srt/openai_api/protocol.py"],
        "config": ["python/sglang/srt/server_args.py",
                   "python/sglang/srt/sampling/sampling_params.py"],
        "pyapi": ["python/sglang/srt/entrypoints/engine.py",
                  "python/sglang/lang/interpreter.py"],
    },
    "lmdeploy": {
        "routes": ["lmdeploy/serve/openai/api_server.py",
                   "lmdeploy/serve/openai/endpoints/*.py",
                   "lmdeploy/serve/openai/**/serving.py",
                   "lmdeploy/serve/anthropic/endpoints/*.py",
                   "lmdeploy/serve/proxy/proxy.py"],
        "protocol": ["lmdeploy/serve/openai/protocol.py",
                     "lmdeploy/serve/**/protocol.py"],
        "config": ["lmdeploy/messages.py", "lmdeploy/cli/*.py"],
        "pyapi": ["lmdeploy/api.py", "lmdeploy/serve/async_engine.py"],
    },
    "lightllm": {
        "routes": ["lightllm/server/api_http.py", "lightllm/server/api_server.py"],
        "protocol": ["lightllm/server/api_models.py", "lightllm/server/**/protocol.py"],
        "config": ["lightllm/server/api_cli.py",
                   "lightllm/server/core/objs/sampling_params.py"],
        "pyapi": ["lightllm/server/httpserver/manager.py"],
    },
    "tensorrt-llm": {
        "routes": ["tensorrt_llm/serve/openai_server.py",
                   "tensorrt_llm/serve/openai_disagg_server.py",
                   "tensorrt_llm/serve/coordinator_server.py",
                   "tensorrt_llm/serve/cluster_storage.py"],
        "protocol": ["tensorrt_llm/serve/openai_protocol.py"],
        "config": ["tensorrt_llm/llmapi/llm_args.py", "tensorrt_llm/sampling_params.py",
                   "tensorrt_llm/commands/serve.py"],
        "pyapi": ["tensorrt_llm/llmapi/llm.py"],
    },
    # KTransformers 2026 版把原来的 python 包整体挪进了 archive/，
    # 仓库根改成 kt-kernel/ + ktransformers.py。两处都扫，扫不到就是真没有。
    "ktransformers": {
        "routes": ["archive/ktransformers/server/api/**/*.py",
                   "ktransformers/server/api/**/*.py"],
        "protocol": ["archive/ktransformers/server/schemas/**/*.py",
                     "ktransformers/server/schemas/**/*.py"],
        # 新的 kt-cli 用 click 而不是 argparse，得把它的命令模块也扫进来
        "config": ["archive/ktransformers/server/config/config.py",
                   "archive/ktransformers/server/args.py",
                   "ktransformers/server/config/config.py",
                   "kt-kernel/python/cli/commands/*.py"],
        "pyapi": ["archive/ktransformers/server/backend/interfaces/*.py"],
    },
    "mlc-llm": {
        "routes": ["python/mlc_llm/serve/entrypoints/openai_entrypoints.py",
                   "python/mlc_llm/**/entrypoints/*.py"],
        "protocol": ["python/mlc_llm/protocol/openai_api_protocol.py"],
        # 第一版只指 serve/config.py + interface/serve.py，两个都不是 CLI 入口，
        # 于是 cli_flags 抽出 0 —— 假的。真正的 argparse 散在 python/mlc_llm/cli/*.py，
        # 光 cli/serve.py 就有 36 处 add_argument。
        "config": ["python/mlc_llm/serve/config.py", "python/mlc_llm/interface/serve.py",
                   "python/mlc_llm/cli/*.py"],
        "pyapi": ["python/mlc_llm/serve/engine.py"],
    },
    "tokasaurus": {
        "routes": ["tokasaurus/server/**/*.py", "tokasaurus/entry.py"],
        "protocol": ["tokasaurus/common_types.py"],
        "config": ["tokasaurus/config.py"],
        "pyapi": ["tokasaurus/manager/**/*.py"],
    },
    # Dynamo 的 HTTP 层是 Rust（axum），Python 侧只有 worker 组件。
    # 协议定义在 .rs 里，**不能喂给 Python AST**（会得到一堆假的 SyntaxError），
    # 所以 protocol 留空，路由改走下面的 REGEX_ROUTES。
    "dynamo": {
        "routes": ["components/src/**/*.py"],
        "protocol": [],
        "config": ["components/src/**/*.py"],
        "pyapi": ["lib/bindings/python/src/**/*.py"],
    },
}

# 非 Python 的路由，正则兜底
REGEX_ROUTES: dict[str, dict] = {
    "tgi": {
        "files": ["router/src/server.rs"],
        # axum: .route("/generate", post(generate))  /  utoipa: path = "/generate"
        "patterns": [r'\.route\(\s*"([^"]+)"\s*,\s*(get|post|put|delete)\(',
                     r'path\s*=\s*"(/[^"]*)"'],
    },
    # llama-server 用 ctx_http.get("/path", handler) / .post(...) 注册路由
    "llama.cpp": {
        "files": ["tools/server/server.cpp", "tools/server/server-http.cpp",
                  "examples/server/server.cpp"],
        "patterns": [r'ctx_http\.(get|post)\(\s*"([^"]+)"',
                     r'srv->(Get|Post)\(\s*"([^"]+)"'],
    },
    # Dynamo：axum 的 .route(&path, post(h)) 用的是变量，路径本身写在
    # `unwrap_or("/v1/xxx")` 的默认值或 `const X: &str = "/v1/xxx"` 里。
    # 抓的是**默认路径**，实际可被启动参数改写 —— 正文引用时要说明这一点。
    "dynamo": {
        "files": ["lib/llm/src/http/service/*.rs"],
        "patterns": [r'unwrap_or(?:_else)?\(\s*(?:\|\|\s*)?"(/[^"]*)"',
                     r'const\s+\w+\s*:\s*&str\s*=\s*"(/[^"]*)"'],
    },
    # Mooncake 是传输引擎/KV 存储，不是 HTTP 推理服务；这里保留空 pattern，
    # 抽到 0 条是**事实**而不是 bug（它的对外接口是 C++/Python 库调用与 RPC）。
    "mooncake": {
        "files": [],
        "patterns": [],
    },
}


# ---------------------------------------------------------------- AST helpers

def _unparse(node) -> str:
    try:
        s = ast.unparse(node)
    except Exception:
        return "<?>"
    return s if len(s) <= 160 else s[:157] + "..."


def _decorator_routes(fn) -> list[dict]:
    """从函数装饰器里抽 HTTP 路由。只认 <obj>.<method>("/path") 这一种形状。"""
    found = []
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
            continue
        method = dec.func.attr.lower()
        if method not in HTTP_METHODS:
            continue
        obj = dec.func.value
        obj_name = obj.id if isinstance(obj, ast.Name) else _unparse(obj)
        path = None
        if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
            path = dec.args[0].value
        for kw in dec.keywords:
            if kw.arg == "path" and isinstance(kw.value, ast.Constant):
                path = kw.value.value
        if path is None:
            continue
        found.append({
            "path": path,
            "method": method.upper(),
            "handler": fn.name,
            "app_obj": obj_name,
            "line": fn.lineno,
            "is_async": isinstance(fn, ast.AsyncFunctionDef),
        })
    return found


def _call_routes(tree) -> list[dict]:
    """抽 `app.add_api_route("/path", handler, methods=["GET"])` 这种**函数调用式**注册。

    为什么单独写一份：TensorRT-LLM 的 `OpenAIServer.register_routes()` 全部用这种写法
    （`tensorrt_llm/serve/openai_server.py`），只认装饰器会得到「0 条路由」——
    而 0 是错的，不是事实。
    """
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("add_api_route", "add_route", "add_websocket_route")):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        path = node.args[0].value
        if not isinstance(path, str) or not path.startswith("/"):
            continue
        handler = _unparse(node.args[1]) if len(node.args) > 1 else "?"
        methods = ["POST"]           # add_api_route 的 FastAPI 默认是 GET，但这里显式取
        for kw in node.keywords:
            if kw.arg == "methods":
                try:
                    vals = ast.literal_eval(kw.value)
                    methods = [str(v).upper() for v in vals]
                except (ValueError, SyntaxError):
                    methods = ["?"]
        if not any(kw.arg == "methods" for kw in node.keywords):
            methods = ["GET"]        # FastAPI 默认
        for m in methods:
            found.append({"path": path, "method": m, "handler": handler,
                          "app_obj": _unparse(node.func.value), "line": node.lineno,
                          "is_async": None, "style": "add_api_route"})
    return found


def _router_prefixes(tree) -> dict[str, str]:
    """找出 `router = APIRouter(prefix="/v1")` 这类前缀声明，返回 {变量名: 前缀}。

    为什么必须有：FastAPI 允许把公共前缀挂在 router 上，处理函数上的装饰器只写
    `@router.post("/chat/completions")`。只看装饰器会得到 `/chat/completions`，
    于是"有几条 /v1/* 端点"会被算成 0 —— KTransformers 就是这么被误判的。
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("APIRouter", "Router"):
            continue
        prefix = ""
        for kw in node.value.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                prefix = kw.value.value
        if not prefix:
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                out[t.id] = prefix.rstrip("/")
    return out


def _click_flags(tree) -> list[dict]:
    """抽 `@click.option("--foo", ...)` / `@option("--foo", ...)` 装饰器式 CLI 开关。

    为什么必须有：新一代 CLI（KTransformers 的 `kt-cli`）用 click 而不是 argparse，
    只认 argparse 会把它的开关数算成 0。
    """
    flags = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            f = dec.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if nm not in ("option", "argument"):
                continue
            names = [a.value for a in dec.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            opts = [n for n in names if n.startswith("-")]
            if not opts:
                continue
            kw = {k.arg: k.value for k in dec.keywords if k.arg}
            entry = {
                "flags": opts,
                "primary": next((n for n in opts if n.startswith("--")), opts[0]),
                "type": _unparse(kw["type"]) if "type" in kw else None,
                "default": _unparse(kw["default"]) if "default" in kw else None,
                "action": None,
                "line": dec.lineno,
                "style": "click",
            }
            if "help" in kw and isinstance(kw["help"], ast.Constant) \
                    and isinstance(kw["help"].value, str):
                entry["help"] = " ".join(kw["help"].value.split())[:200]
            flags.append(entry)
    return flags


def _class_fields(cls: ast.ClassDef) -> list[dict]:
    """抽类体里的带注解字段（pydantic / dataclass 都是这个形状）。"""
    fields = []
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append({
                "name": stmt.target.id,
                "type": _unparse(stmt.annotation),
                "default": _unparse(stmt.value) if stmt.value is not None else None,
                "line": stmt.lineno,
            })
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_") and not t.id.isupper():
                    fields.append({
                        "name": t.id, "type": None,
                        "default": _unparse(stmt.value), "line": stmt.lineno,
                    })
    return fields


def _public_methods(cls: ast.ClassDef) -> list[dict]:
    out = []
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and not stmt.name.startswith("_"):
            args = [a.arg for a in stmt.args.args if a.arg != "self"]
            args += [a.arg for a in stmt.args.kwonlyargs]
            out.append({"name": stmt.name, "line": stmt.lineno,
                        "is_async": isinstance(stmt, ast.AsyncFunctionDef),
                        "params": args})
    return out


def _base_names(cls: ast.ClassDef) -> list[str]:
    return [_unparse(b) for b in cls.bases]


def _argparse_flags(tree) -> list[dict]:
    """抽 parser.add_argument("--foo", ...) 的 flag 名、type、default、choices、help 首句。"""
    flags = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not names:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        entry = {
            "flags": names,
            "primary": next((n for n in names if n.startswith("--")), names[0]),
            "type": _unparse(kw["type"]) if "type" in kw else None,
            "default": _unparse(kw["default"]) if "default" in kw else None,
            "action": _unparse(kw["action"]) if "action" in kw else None,
            "line": node.lineno,
        }
        if "choices" in kw:
            entry["choices"] = _unparse(kw["choices"])
        if "help" in kw and isinstance(kw["help"], ast.Constant) and isinstance(kw["help"].value, str):
            entry["help"] = " ".join(kw["help"].value.split())[:200]
        flags.append(entry)
    return flags


def _resolve(root: Path, patterns: list[str], cap: int = 400) -> list[Path]:
    seen: list[Path] = []
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p not in seen:
                seen.append(p)
                if len(seen) >= cap:
                    return seen
    return seen


def _scan_python(root: Path, files: list[Path], want_routes=True, want_classes=True,
                 want_flags=True) -> dict:
    routes, classes, flags, parsed, failed = [], [], [], [], []
    uses_include_router = False
    for fp in files:
        text = read_text(fp)
        if not text:
            continue
        try:
            tree = ast.parse(text, filename=str(fp))
        except SyntaxError as e:
            failed.append({"file": rel(fp, root), "error": f"SyntaxError line {e.lineno}"})
            continue
        parsed.append(rel(fp, root))
        r = rel(fp, root)
        prefixes = _router_prefixes(tree)
        if "include_router" in text:
            uses_include_router = True
        if want_routes:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for rt in _decorator_routes(node):
                        # router = APIRouter(prefix="/v1") 时，装饰器上的路径要补前缀
                        pre = prefixes.get(rt.get("app_obj", ""))
                        if pre and not rt["path"].startswith(pre):
                            rt["path"] = pre + rt["path"]
                            rt["prefix_from_router"] = pre
                        rt["file"] = r
                        routes.append(rt)
            for rt in _call_routes(tree):
                rt["file"] = r
                routes.append(rt)
        if want_classes:
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    fields = _class_fields(node)
                    classes.append({
                        "name": node.name, "file": r, "line": node.lineno,
                        "bases": _base_names(node), "n_fields": len(fields),
                        "fields": fields,
                        "methods": _public_methods(node),
                        "decorators": [_unparse(d) for d in node.decorator_list],
                    })
        if want_flags:
            for f in _argparse_flags(tree) + _click_flags(tree):
                f["file"] = r
                flags.append(f)
    return {"routes": routes, "classes": classes, "flags": flags,
            "files_parsed": parsed, "parse_failures": failed,
            "uses_include_router": uses_include_router}


def _scan_regex(root: Path, spec: dict) -> dict:
    routes, files_seen = [], []
    for pat in spec["files"]:
        for fp in sorted(root.glob(pat)):
            if not fp.is_file():
                continue
            files_seen.append(rel(fp, root))
            text = read_text(fp)
            for rx in spec["patterns"]:
                for m in re.finditer(rx, text):
                    groups = [g for g in m.groups() if g]
                    path = next((g for g in groups if g.startswith("/")), None)
                    if not path:
                        continue
                    verb = next((g.upper() for g in groups
                                 if g.lower() in {"get", "post", "put", "delete"}), "?")
                    line = text[:m.start()].count("\n") + 1
                    routes.append({"path": path, "method": verb,
                                   "file": rel(fp, root), "line": line})
    dedup, seen = [], set()
    for r in routes:
        key = (r["path"], r["method"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return {"routes": dedup, "files_parsed": files_seen, "method": "regex",
            "classes": [], "flags": [], "parse_failures": []}


def analyze(name: str) -> dict:
    root = engine_path(name)
    assert root is not None
    ref = repo_ref(name)
    result: dict = {"ref": ref.as_dict() if ref else {}, "method": "ast"}

    if name in ENTRY_HINTS:
        h = ENTRY_HINTS[name]
        rs = _scan_python(root, _resolve(root, h["routes"]), want_classes=False, want_flags=False)
        ps = _scan_python(root, _resolve(root, h["protocol"]), want_routes=False, want_flags=False)
        cs = _scan_python(root, _resolve(root, h["config"]), want_routes=False)
        api = _scan_python(root, _resolve(root, h.get("pyapi", [])), want_flags=False)

        result["routes"] = rs["routes"] + api["routes"]
        result["route_files"] = rs["files_parsed"]
        result["protocol_classes"] = ps["classes"]
        result["protocol_files"] = ps["files_parsed"]
        result["config_classes"] = cs["classes"]
        result["config_files"] = cs["files_parsed"]
        result["cli_flags"] = cs["flags"]
        result["engine_classes"] = api["classes"]
        result["parse_failures"] = (rs["parse_failures"] + ps["parse_failures"]
                                    + cs["parse_failures"] + api["parse_failures"])
        result["_uses_include_router"] = rs.get("uses_include_router", False) or             api.get("uses_include_router", False)
        # 混合型引擎（Python worker + Rust/C++ HTTP 层，如 Dynamo）：两种抽法都跑，合并
        if name in REGEX_ROUTES:
            extra = _scan_regex(root, REGEX_ROUTES[name])
            if extra["routes"]:
                result["method"] = "ast+regex"
                result["routes"] = result["routes"] + extra["routes"]
                result["route_files"] = result["route_files"] + extra["files_parsed"]
    elif name in REGEX_ROUTES:
        result.update(_scan_regex(root, REGEX_ROUTES[name]))
        for k in ("protocol_classes", "config_classes", "cli_flags", "engine_classes"):
            result.setdefault(k, [])
    else:
        result.update({"routes": [], "protocol_classes": [], "config_classes": [],
                       "cli_flags": [], "engine_classes": [], "note": "未登记入口线索"})

    paths = sorted({r["path"] for r in result.get("routes", [])})
    # 前缀解析的已知边界：本工具只在**单个文件内**解析 APIRouter(prefix=...)。
    # 像 KTransformers 那样用 include_router 跨文件组合、把 /v1 挂在更上层的写法，
    # 这里补不出来 —— 于是 openai_compat_paths 会偏少。**这是工具的边界，不是引擎没有该端点。**
    # 只有当「用了 include_router **且** 一条 /v1/* 都没抽到」时才报警 ——
    # 光看用没用 include_router 会对 vLLM/SGLang 这类前缀写在装饰器里的项目误报，
    # 一个对所有人都亮的警告等于没有警告。
    uses_include = bool(result.pop("_uses_include_router", False))
    _paths_tmp = {r["path"] for r in result.get("routes", [])}
    suspicious = uses_include and bool(_paths_tmp) and not any(
        p.startswith("/v1/") for p in _paths_tmp)
    result["summary"] = {
        "route_prefix_caveat": (
            "该引擎用 include_router 跨文件组合路由，且本次一条 /v1/* 都没抽到 —— "
            "上层前缀很可能挂在别处，本工具只在单文件内解析 APIRouter(prefix=...)，补不出来。"
            "**这是工具的边界，不能据此说该引擎没有 /v1 端点。**" if suspicious else None),
        "route_prefix_maybe_undercounted": suspicious,
        "n_routes": len(result.get("routes", [])),
        "n_unique_paths": len(paths),
        "unique_paths": paths,
        "n_protocol_classes": len(result.get("protocol_classes", [])),
        "n_cli_flags": len({f["primary"] for f in result.get("cli_flags", [])}),
        "openai_compat_paths": [p for p in paths if p.startswith("/v1/")],
    }
    return result


SELFTEST_SRC = "\n".join([
    "import argparse",
    "from pydantic import BaseModel",
    "from fastapi import FastAPI",
    "app = FastAPI()",
    "",
    "class ChatCompletionRequest(BaseModel):",
    "    model: str",
    "    messages: list = []",
    "    temperature: float = 1.0",
    "",
    "@app.post('/v1/chat/completions')",
    "async def create_chat(req: ChatCompletionRequest):",
    "    return req",
    "",
    "@app.get('/health')",
    "def health():",
    "    return 'ok'",
    "",
    "v1 = APIRouter(prefix='/v1')",
    "",
    "@v1.post('/rerank2')",
    "def rr():",
    "    pass",
    "",
    "@click.option('--kt-flag', default=3, help='click style')",
    "def cli():",
    "    pass",
    "",
    "def make():",
    "    p = argparse.ArgumentParser()",
    "    p.add_argument('--max-num-seqs', type=int, default=256, help='Maximum sequences.')",
    "    p.add_argument('--enable-prefix-caching', action='store_true')",
    "    return p",
    "",
    "def register(app):",
    "    app.add_api_route('/v1/models', get_models, methods=['GET'])",
    "    app.add_api_route('/v1/rerank', do_rerank, methods=['POST'])",
    "    app.add_api_route('/version', ver)",
    "",
])


def selftest() -> int:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f = root / "srv.py"
        f.write_text(SELFTEST_SRC, encoding="utf-8")
        got = _scan_python(root, [f])

        paths = sorted(r["path"] for r in got["routes"])
        if paths != ["/health", "/v1/chat/completions", "/v1/models",
                     "/v1/rerank", "/v1/rerank2", "/version"]:
            print(f"FAIL routes -> {paths}"); ok = False
        # add_api_route：显式 methods 要被读出来，不写 methods 时按 FastAPI 默认 GET
        by_path = {r["path"]: r for r in got["routes"]}
        if by_path["/v1/rerank"]["method"] != "POST":
            print(f"FAIL add_api_route methods -> {by_path['/v1/rerank']}"); ok = False
        if by_path["/version"]["method"] != "GET":
            print(f"FAIL add_api_route default method -> {by_path['/version']}"); ok = False
        if by_path["/v1/models"]["style"] != "add_api_route":
            print("FAIL style tag missing"); ok = False
        if by_path["/health"]["method"] != "GET" or by_path["/v1/chat/completions"]["method"] != "POST":
            print("FAIL decorator methods"); ok = False

        cls = [c for c in got["classes"] if c["name"] == "ChatCompletionRequest"]
        if not cls or cls[0]["n_fields"] != 3:
            print(f"FAIL protocol fields -> {cls}"); ok = False
        elif cls[0]["fields"][2]["default"] != "1.0":
            print(f"FAIL default -> {cls[0]['fields'][2]}"); ok = False

        # APIRouter(prefix=...) 的前缀必须被补上，否则 /v1 端点数会被算成 0
        if by_path["/v1/rerank2"].get("prefix_from_router") != "/v1":
            print(f"FAIL router prefix -> {by_path['/v1/rerank2']}"); ok = False
        flags = sorted(x["primary"] for x in got["flags"])
        if flags != ["--enable-prefix-caching", "--kt-flag", "--max-num-seqs"]:
            print(f"FAIL flags -> {flags}"); ok = False
        else:
            mns = next(x for x in got["flags"] if x["primary"] == "--max-num-seqs")
            if mns["default"] != "256" or mns["type"] != "int":
                print(f"FAIL flag meta -> {mns}"); ok = False

        chat = next(r for r in got["routes"] if r["path"].startswith("/v1"))
        if not chat["is_async"]:
            print("FAIL async detection"); ok = False

        # 正则兜底也要自检：造一个假 rust router
        (root / "server.rs").write_text(
            '.route("/generate", post(generate))\n.route("/health", get(health))\n',
            encoding="utf-8")
        rr = _scan_regex(root, {"files": ["server.rs"],
                                "patterns": [r'\.route\(\s*"([^"]+)"\s*,\s*(get|post)\(']})
        if sorted(r["path"] for r in rr["routes"]) != ["/generate", "/health"]:
            print(f"FAIL regex routes -> {rr['routes']}"); ok = False

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
        print(f"[api_surface] {n} ...", flush=True)
        out[n] = analyze(n)
    fp = dump_engines("api_surface.json", out)
    print(f"-> {fp}")
    for n, d in out.items():
        s = d["summary"]
        print(f"  {ENGINES[n]['label']:<14} routes={s['n_routes']:>3} "
              f"(/v1 {len(s['openai_compat_paths']):>2})  proto_cls={s['n_protocol_classes']:>3}  "
              f"cli_flags={s['n_cli_flags']:>4}  method={d.get('method')}")
        if d.get("parse_failures"):
            print(f"      parse_failures: {d['parse_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
