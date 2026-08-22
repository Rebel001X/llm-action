# SGLang HTTP API 表面全解

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：四层兼容协议，最终都收口成一个内部请求

## 0. 结论先行

- SGLang 的 HTTP 表面不是"一套 API"，是**四层兼容协议叠在一个内部结构上**：OpenAI 兼容（`/v1/*`）、Anthropic 兼容（`/v1/messages`）、Ollama 兼容（`/api/*`）、SageMaker 兼容（`/ping` `/invocations`），全部最终转换成同一个内部请求对象 `GenerateReqInput`（`python/sglang/srt/managers/io_struct.py:162`），再走同一条 `tokenizer_manager.generate_request()` 通道。`/generate`（`python/sglang/srt/entrypoints/http_server.py:894`）不是"又一个兼容层"，它是**唯一的原生入口**——其余所有协议层最终都在替 `/generate` 拼装同一份 `GenerateReqInput`。
- 包装关系不是并列的三条独立路径，是**俄罗斯套娃**：`AnthropicServing.handle_messages()`（`python/sglang/srt/entrypoints/anthropic/serving.py:206`）先把 Anthropic Messages 请求转换成 `ChatCompletionRequest`（`_convert_to_chat_completion_request()`，`python/sglang/srt/entrypoints/anthropic/serving.py:228`），然后直接调用 `OpenAIServingChat` 的内部方法 `_convert_to_internal_request()`/`_handle_non_streaming_request()`（`python/sglang/srt/entrypoints/anthropic/serving.py:759, 766`）——**Anthropic 层没有自己的 tokenize/模板逻辑，是纯翻译层，源码类文档字符串原话就是"Acts as a translation layer between Anthropic's Messages API and SGLang's OpenAI-compatible chat completion infrastructure"**（`python/sglang/srt/entrypoints/anthropic/serving.py:184-188`）。
- SGLang 共抽到 **83 条路由**，但这不是全部——`_lab/api_surface.py` 是 AST 静态扫描，遇到 `@app.post(os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"))` 这种**路径参数不是字符串字面量而是函数调用**的写法会直接漏抽。Ollama 兼容的 4 条路由（`/api/chat` `/api/generate` `/api/tags` `/api/show`，`python/sglang/srt/entrypoints/http_server.py:1973,1979,1987,1993`）全部用这种写法，**真实路由数至少是 87，不是 83**——这是 `_lab/out/compare.json` 里"条件注册可能漏"这条 caveat 的一个可复现实例，细节见 `## 7`。
- SGLang 相对 vLLM 最不一样的地方：**把 RLHF 训练场景要用到的权重热更新/显存让出族（`init_weights_update_group`、`update_weights_from_tensor`、`release_memory_occupation`…共 16 条）当成主服务器的一等公民路由**，只用 `@auth_level(AuthLevel.ADMIN_OPTIONAL)` 标记权限级别，默认无 key 配置时**不设防**（`python/sglang/srt/utils/auth.py:82`）。vLLM 有功能对等的端点（`/sleep` `/wake_up` `/update_weights`…），但它们被物理隔离在 `vllm/entrypoints/serve/dev/` 目录下，默认**完全不挂载**，需要显式设置从不默认打开的 `VLLM_SERVER_DEV_MODE=1`（默认 `False`，`vllm:vllm/envs.py:168`）才会注册，注册时还会打印"SECURITY WARNING: Development endpoints are enabled! This should NOT be used in production!"（`vllm:vllm/entrypoints/serve/__init__.py:44-46`）。同一件事，SGLang 当成生产特性，vLLM 当成开发者调试后门——这不是谁更安全，是两边对"谁会把推理引擎当 RL rollout worker 用"这件事的产品定位完全不同。
- gRPC 入口是"薄壳"：`grpc_server.py` 文件头文档字符串写明"Thin gRPC server wrapper — delegates to smg-grpc-servicer package"（`python/sglang/srt/entrypoints/grpc_server.py:2`），真正的 gRPC servicer 实现在一个**外部 pip 包**（`smg-grpc-servicer`）里，不在本仓库源码范围内；`grpc_server.py` 自己只启动一个基于 `aiohttp` 的 HTTP 旁路（sidecar），暴露 `/metrics` `/start_profile` `/stop_profile` 三个调试端点，且用的是 `app.router.add_get/add_post`（`python/sglang/srt/entrypoints/grpc_server.py:64,152,153`）而非 FastAPI 装饰器——这也是为什么这三条端点**完全不在** `_lab` 抽出的 83 条里（AST 抽取器只认 `@app.get/@app.post` 这种 FastAPI 装饰器语法）。

---

## 1. 它在系统里的位置

一次进来的 HTTP 请求，无论走哪条协议，最终都要收口到同一个漏斗：

```
Client
  │
  ├─ POST /v1/chat/completions ─┐
  ├─ POST /v1/messages (Anthropic) ─┼─→ ChatCompletionRequest ─→ OpenAIServingChat
  ├─ POST /api/chat (Ollama) ────┘         ._convert_to_internal_request()
  ├─ POST /invocations (SageMaker) ────────────────┐         │
  │                                                  ▼         ▼
  └─ POST /generate (原生) ─────────────────→  GenerateReqInput
                                                      │
                                                      ▼
                                    TokenizerManager.generate_request()
                                                      │  (ZMQ → Scheduler 进程)
                                                      ▼
                                              Scheduler / TpModelWorker
```

- **顶层挂载点**只有一个 FastAPI 实例 `app`（`python/sglang/srt/entrypoints/http_server.py:456`），所有兼容协议的路由都注册在这同一个 `app` 上，通过 `app.state.<xxx_serving>` 挂具体的 handler 对象——这些 handler 对象的初始化发生在 `launch_server()` 内（`python/sglang/srt/entrypoints/http_server.py:2766` 起），例如 `fast_api_app.state.openai_serving_chat = _global_state.tokenizer_manager.serving_chat_class(...)`（`python/sglang/srt/entrypoints/http_server.py:306-309`）、`fast_api_app.state.anthropic_serving = AnthropicServing(fast_api_app.state.openai_serving_chat)`（`python/sglang/srt/entrypoints/http_server.py:337-339`，**注意这里直接把已经建好的 `openai_serving_chat` 传给 `AnthropicServing` 的构造函数**，从代码摆放顺序上就能看出依赖关系）。
- **`TokenizerManager`**（`python/sglang/srt/managers/tokenizer_manager.py`，本篇不展开，见 `_PLAN.md` 篇目 `02-SGLang-Scheduler事件循环`）是唯一真正跨进程通信的边界——`http_server.py` 里所有 handler 最终都调用 `_global_state.tokenizer_manager.xxx()`，这一层往下才是 ZMQ + Scheduler 进程，属于另一篇的范围。
- **PD 分离**在 HTTP 表面上留下两处痕迹：一是 `ChatCompletionRequest`/`GenerateReqInput` 里的 `bootstrap_host/bootstrap_port/bootstrap_room` 字段（用于 prefill 侧告诉 decode 侧去哪取 KV），二是一个**完全独立的 FastAPI 应用** `EngineInfoBootstrapServer`（`python/sglang/srt/entrypoints/engine_info_bootstrap_server.py:26`），文档字符串写"Runs in a daemon thread on node_rank==0. Each ModelRunner registers its info via HTTP PUT after model initialization"（`python/sglang/srt/entrypoints/engine_info_bootstrap_server.py:26-31`）——这是给 KV 传输引擎（transfer engine）做跨 TP-rank 信息注册用的旁路服务，跑在**另一个端口**，和主 `app` 是两个不同的 FastAPI 实例。这部分留给 `[[11-SGLang-PD分离与HiCache分层]]` 详细展开，本篇只在代码地图里给出坐标。

这篇只看"HTTP 表面长什么样、协议之间怎么互相包装"，不看 `TokenizerManager` 往下怎么跟 Scheduler 通信（那是 `[[02-SGLang-Scheduler事件循环]]` 的范围），也不看 RadixAttention/HiCache 内部实现（那是 `[[03-SGLang-RadixAttention与前缀缓存]]` 和 `[[11-SGLang-PD分离与HiCache分层]]` 的范围）。

---

## 2. 代码地图（文件 → 职责，带行号）

### 2.1 路由分类表（83 条 AST 抽取结果，按族分组）

来源：`_lab/out/api_surface.json` 的 `engines.sglang.routes`（AST 抽取），逐条核对过行号。除特别标注外文件都是 `python/sglang/srt/entrypoints/http_server.py`。

**A 族 · 健康检查与元信息只读**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/health`、`/health_generate` | GET | `health_generate` | `python/sglang/srt/entrypoints/http_server.py:656`（同一函数注册两个路径） |
| `/get_model_info` | GET | `get_model_info` | `python/sglang/srt/entrypoints/http_server.py:729` |
| `/model_info` | GET | `model_info` | `python/sglang/srt/entrypoints/http_server.py:739` |
| `/get_weight_version`、`/weight_version` | GET | `weight_version` | `python/sglang/srt/entrypoints/http_server.py:778` |
| `/get_server_info` | GET | `get_server_info` | `python/sglang/srt/entrypoints/http_server.py:787` |
| `/server_info` | GET | `server_info` | `python/sglang/srt/entrypoints/http_server.py:797` |
| `/get_load` | GET | `get_load` | `python/sglang/srt/entrypoints/http_server.py:830` |
| `/`（默认根） | GET/HEAD | `sglang_root` | `python/sglang/srt/entrypoints/http_server.py:1968` |
| `/ping` | GET | `sagemaker_health` | `python/sglang/srt/entrypoints/http_server.py:2024` |

**B 族 · 原生推理入口**（本篇 `## 4` 详细走读）

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/generate` | API_ROUTE | `generate_request` | `python/sglang/srt/entrypoints/http_server.py:894` |
| `/encode` | API_ROUTE | `encode_request` | `python/sglang/srt/entrypoints/http_server.py:943` |
| `/classify` | API_ROUTE | `classify_request` | `python/sglang/srt/entrypoints/http_server.py:955` |

**C 族 · 缓存与语料库管理**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/flush_cache` | API_ROUTE | `flush_cache` | `python/sglang/srt/entrypoints/http_server.py:968` |
| `/add_external_corpus` | POST | `add_external_corpus` | `python/sglang/srt/entrypoints/http_server.py:986` |
| `/remove_external_corpus` | POST | `remove_external_corpus` | `python/sglang/srt/entrypoints/http_server.py:1011` |
| `/list_external_corpora` | GET | `list_external_corpora` | `python/sglang/srt/entrypoints/http_server.py:1029` |
| `/clear_hicache_storage_backend`（deprecated） | GET/POST | `clear_hicache_storage_backend_deprecated` | `python/sglang/srt/entrypoints/http_server.py:1044` |
| `/hicache/storage-backend/clear` | POST | `clear_hicache_storage_backend` | `python/sglang/srt/entrypoints/http_server.py:1060` |
| `/hicache/storage-backend`（同路径三方法） | PUT=attach / DELETE=detach / GET=status | `attach_hicache_storage_backend`/`detach_hicache_storage_backend`/`hicache_storage_backend_status` | `python/sglang/srt/entrypoints/http_server.py:1080,1114,1141` |

**D 族 · Profiling / Debug**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/start_profile` | GET/POST | `start_profile_async` | `python/sglang/srt/entrypoints/http_server.py:1159` |
| `/stop_profile` | API_ROUTE | `stop_profile_async` | `python/sglang/srt/entrypoints/http_server.py:1170` |
| `/set_trace_level` | API_ROUTE | `set_trace_level` | `python/sglang/srt/entrypoints/http_server.py:1180` |
| `/freeze_gc` | API_ROUTE | `freeze_gc_async` | `python/sglang/srt/entrypoints/http_server.py:1191` |
| `/start_expert_distribution_record` | API_ROUTE | `start_expert_distribution_record_async` | `python/sglang/srt/entrypoints/http_server.py:1204` |
| `/stop_expert_distribution_record` | API_ROUTE | `stop_expert_distribution_record_async` | `python/sglang/srt/entrypoints/http_server.py:1215` |
| `/dump_expert_distribution_record` | API_ROUTE | `dump_expert_distribution_record_async` | `python/sglang/srt/entrypoints/http_server.py:1226` |
| `/configure_logging` | API_ROUTE | `configure_logging` | `python/sglang/srt/entrypoints/http_server.py:1599` |
| `/dumper/{method}` | API_ROUTE | `_dumper_control_handler` | `python/sglang/srt/entrypoints/http_server.py:872` |

**E 族 · RLHF 权重热更新 / 显存让出**（本篇 `## 5` 重点）

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/update_weights_from_disk` | POST | `update_weights_from_disk` | `python/sglang/srt/entrypoints/http_server.py:1237` |
| `/init_weights_send_group_for_remote_instance` | POST | `init_weights_send_group_for_remote_instance` | `python/sglang/srt/entrypoints/http_server.py:1266` |
| `/send_weights_to_remote_instance` | POST | `send_weights_to_remote_instance` | `python/sglang/srt/entrypoints/http_server.py:1285` |
| `/get_remote_instance_transfer_engine_info`（deprecated） | GET | `get_remote_instance_transfer_engine_info` | `python/sglang/srt/entrypoints/http_server.py:1303` |
| `/remote_instance_transfer_engine_info` | GET | `remote_instance_transfer_engine_info` | `python/sglang/srt/entrypoints/http_server.py:1314` |
| `/init_weights_update_group` | POST | `init_weights_update_group` | `python/sglang/srt/entrypoints/http_server.py:1341` |
| `/destroy_weights_update_group` | POST | `destroy_weights_update_group` | `python/sglang/srt/entrypoints/http_server.py:1357` |
| `/update_weights_from_tensor` | POST | `update_weights_from_tensor` | `python/sglang/srt/entrypoints/http_server.py:1373` |
| `/update_weights_from_distributed` | POST | `update_weights_from_distributed` | `python/sglang/srt/entrypoints/http_server.py:1395` |
| `/update_weights_from_ipc` | POST | `update_weights_from_ipc` | `python/sglang/srt/entrypoints/http_server.py:1415` |
| `/update_weight_version` | POST | `update_weight_version` | `python/sglang/srt/entrypoints/http_server.py:1434` |
| `/get_weights_by_name` | GET/POST | `get_weights_by_name` | `python/sglang/srt/entrypoints/http_server.py:1468` |
| `/release_memory_occupation` | GET/POST | `release_memory_occupation` | `python/sglang/srt/entrypoints/http_server.py:1484` |
| `/resume_memory_occupation` | GET/POST | `resume_memory_occupation` | `python/sglang/srt/entrypoints/http_server.py:1496` |
| `/weights_checker` | GET/POST | `check_weights` | `python/sglang/srt/entrypoints/http_server.py:1508` |
| `/slow_down` | GET/POST | `slow_down` | `python/sglang/srt/entrypoints/http_server.py:1527` |

**F 族 · LoRA 热加载**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/load_lora_adapter` | API_ROUTE | `load_lora_adapter` | `python/sglang/srt/entrypoints/http_server.py:1541` |
| `/load_lora_adapter_from_tensors` | API_ROUTE | `load_lora_adapter_from_tensors` | `python/sglang/srt/entrypoints/http_server.py:1551` |
| `/unload_lora_adapter` | API_ROUTE | `unload_lora_adapter` | `python/sglang/srt/entrypoints/http_server.py:1564` |

**G 族 · 会话与请求控制**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/open_session` | API_ROUTE | `open_session` | `python/sglang/srt/entrypoints/http_server.py:1574` |
| `/close_session` | API_ROUTE | `close_session` | `python/sglang/srt/entrypoints/http_server.py:1588` |
| `/abort_request` | POST | `abort_request` | `python/sglang/srt/entrypoints/http_server.py:1609` |
| `/parse_function_call` | POST | `parse_function_call_request` | `python/sglang/srt/entrypoints/http_server.py:1621` |
| `/separate_reasoning` | POST | `separate_reasoning_request` | `python/sglang/srt/entrypoints/http_server.py:1649` |
| `/pause_generation` | POST | `pause_generation` | `python/sglang/srt/entrypoints/http_server.py:1687` |
| `/continue_generation` | POST | `continue_generation` | `python/sglang/srt/entrypoints/http_server.py:1700` |

**I 族 · OpenAI 兼容（`/v1/*`，17 条）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/v1/completions` | POST | `openai_v1_completions` | `python/sglang/srt/entrypoints/http_server.py:1715` |
| `/v1/chat/completions` | POST | `openai_v1_chat_completions` | `python/sglang/srt/entrypoints/http_server.py:1723` |
| `/v1/embeddings` | POST | `openai_v1_embeddings` | `python/sglang/srt/entrypoints/http_server.py:1737` |
| `/v1/classify` | POST | `openai_v1_classify` | `python/sglang/srt/entrypoints/http_server.py:1749` |
| `/v1/tokenize`、`/tokenize` | POST | `openai_v1_tokenize` | `python/sglang/srt/entrypoints/http_server.py:1767` |
| `/v1/detokenize`、`/detokenize` | POST | `openai_v1_detokenize` | `python/sglang/srt/entrypoints/http_server.py:1785` |
| `/v1/audio/transcriptions` | POST | `openai_v1_audio_transcriptions` | `python/sglang/srt/entrypoints/http_server.py:1793` |
| `/v1/realtime` | WEBSOCKET | `openai_v1_realtime_transcription` | `python/sglang/srt/entrypoints/http_server.py:1833` |
| `/v1/models` | GET | `available_models` | `python/sglang/srt/entrypoints/http_server.py:1844` |
| `/v1/models/{model:path}` | GET | `retrieve_model` | `python/sglang/srt/entrypoints/http_server.py:1876` |
| `/v1/score` | POST | `v1_score_request` | `python/sglang/srt/entrypoints/http_server.py:1901` |
| `/v1/responses` | POST | `v1_responses_request` | `python/sglang/srt/entrypoints/http_server.py:1909` |
| `/v1/responses/{response_id}` | GET | `v1_retrieve_responses` | `python/sglang/srt/entrypoints/http_server.py:1928` |
| `/v1/responses/{response_id}/cancel` | POST | `v1_cancel_responses` | `python/sglang/srt/entrypoints/http_server.py:1936` |
| `/v1/rerank` | POST/PUT | `v1_rerank_request` | `python/sglang/srt/entrypoints/http_server.py:1946` |

**J 族 · Anthropic 兼容**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/v1/messages` | POST | `anthropic_v1_messages` | `python/sglang/srt/entrypoints/http_server.py:2003` |
| `/v1/messages/count_tokens` | POST | `anthropic_v1_count_tokens` | `python/sglang/srt/entrypoints/http_server.py:2013` |

**L 族 · SageMaker 兼容**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/invocations` | POST | `sagemaker_chat_completions`（内部直接调用 `openai_serving_chat.handle_request`） | `python/sglang/srt/entrypoints/http_server.py:2030` |

**M 族 · PD Bootstrap 旁路服务（独立 FastAPI 实例，非主 `app`）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/health` | GET | `health` | `python/sglang/srt/entrypoints/engine_info_bootstrap_server.py:48` |
| `/register_transfer_engine_info` | PUT | `register_transfer_engine_info` | `python/sglang/srt/entrypoints/engine_info_bootstrap_server.py:52` |
| `/get_transfer_engine_info` | GET | `get_transfer_engine_info` | `python/sglang/srt/entrypoints/engine_info_bootstrap_server.py:75` |

### 2.2 AST 漏抽的第 K 族：Ollama 兼容（动态路径，`_lab` 统计不到）

以下 4 条真实存在、可正常访问，但因为路径参数是 `os.environ.get(...)` 而非字符串字面量，**没有出现在 83 条统计里**：

| 默认路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/api/chat`（可用 `SGLANG_OLLAMA_CHAT_ROUTE` 覆盖） | POST | `ollama_chat` | `python/sglang/srt/entrypoints/http_server.py:1973` |
| `/api/generate`（可用 `SGLANG_OLLAMA_GENERATE_ROUTE` 覆盖） | POST | `ollama_generate` | `python/sglang/srt/entrypoints/http_server.py:1979` |
| `/api/tags`（可用 `SGLANG_OLLAMA_TAGS_ROUTE` 覆盖） | GET | `ollama_tags` | `python/sglang/srt/entrypoints/http_server.py:1987` |
| `/api/show`（可用 `SGLANG_OLLAMA_SHOW_ROUTE` 覆盖） | POST | `ollama_show` | `python/sglang/srt/entrypoints/http_server.py:1993` |

另外根路径 `/` 的注册本身是**条件分支**：若设置了 `SGLANG_OLLAMA_ROOT_ROUTE` 环境变量（`python/sglang/srt/entrypoints/http_server.py:1955-1960`），根路径会改注册成 `ollama_root`（返回 `"Ollama is running"`）而不是默认的 `sglang_root`——两个函数**互斥注册，永远不会同时存在**，这也是为什么 83 条统计里只看到 `sglang_root` 一份（AST 静态扫描时 `if` 分支两边都会被遍历到，但运行时只有一支真的注册进 `app.router`）。

### 2.3 协议类文件

- 唯一的协议定义文件：`python/sglang/srt/entrypoints/openai/protocol.py`（2116 行，共 86 个 pydantic/msgspec 类，来自 `api_surface.json` 的 `n_protocol_classes`）。
- `ChatCompletionRequest` 类定义于 `python/sglang/srt/entrypoints/openai/protocol.py:823`，一路到 `965` 行左右结束（含 `_DEFAULT_SAMPLING_PARAMS` 类属性与两个 `@model_validator`），本篇 `## 3` 展开。
- Anthropic 协议在**独立文件** `python/sglang/srt/entrypoints/anthropic/protocol.py`，不在 `protocol_files` 抽取范围内（`_lab/api_surface.py` 只 glob 了 `entrypoints/openai/protocol.py` 这一个文件，Anthropic/Ollama 的协议类因此也没被 `n_protocol_classes=86` 统计进去——这是另一处"统计口径比实际窄"的地方，和 2.2 的路由漏抽是同一类问题，见 `## 7`）。

### 2.4 gRPC 协议契约与 HTTP 的关系

`## 0` 已经点出 `grpc_server.py` 是"薄壳"，这里把契约本身摊开看。**协议定义确实在本仓库里**：`proto/sglang/runtime/v1/sglang.proto`（335 行）定义了一个 `SglangService`（`proto/sglang/runtime/v1/sglang.proto:4-34`），共 29 个 RPC，分三组，注释直接写明分组意图：

- **"SGLang-native RPCs (typed proto)"**（`proto/sglang/runtime/v1/sglang.proto:5`）——16 个，逐条对应 HTTP A/B/C 族的核心能力：`Generate`/`TextGenerate`（对应 `/generate`）、`Embed`/`TextEmbed`（对应 `/encode`）、`Classify`（对应 `/classify`）、`Tokenize`/`Detokenize`、`HealthCheck`、`GetModelInfo`/`GetServerInfo`/`ListModels`/`GetLoad`、`Abort`（对应 `/abort_request`）、`FlushCache`、`PauseGeneration`/`ContinueGeneration`。这些 RPC 都有**独立定义的 protobuf message**（比如 `GenerateRequest`，`proto/sglang/runtime/v1/sglang.proto:116-134`），字段是强类型的。
- **"OpenAI-compatible RPCs (JSON pass-through)"**（`proto/sglang/runtime/v1/sglang.proto:23`）——6 个：`ChatComplete`/`Complete`/`OpenAIEmbed`/`OpenAIClassify`/`Score`/`Rerank`。但它们的输入输出类型 `OpenAIRequest`/`OpenAIResponse`（`proto/sglang/runtime/v1/sglang.proto:294-297,304` 起）**根本不是结构化 message**——`OpenAIRequest` 只有一个字段 `bytes json_body = 1`（`proto/sglang/runtime/v1/sglang.proto:295`），说白了就是把 HTTP 那份 `ChatCompletionRequest` 的 JSON 序列化结果原样塞进一个 `bytes` 字段，通过 gRPC 通道传输，服务端拿到后大概率还是走同一套 pydantic 解析——**gRPC 在这一组 RPC 上完全没有提供"强类型"这个通常认为的核心卖点**，只是换了个传输层（HTTP/1.1+JSON → HTTP/2+protobuf 帧），协议内容本身还是 JSON 文本。
- **"Admin/Ops RPCs"**（`proto/sglang/runtime/v1/sglang.proto:31`）——只有 3 个：`StartProfile`/`StopProfile`/`UpdateWeightsFromDisk`。**`## 2` E 族 16 条 RLHF 相关端点里，只有 `update_weights_from_disk` 一个有 gRPC 对应物**——`update_weights_from_tensor`/`update_weights_from_distributed`/`release_memory_occupation`/`resume_memory_occupation`/`init_weights_update_group` 这些用于**co-located 训练/推理**场景的端点，在 gRPC 契约里全部缺席。这意味着如果一个 RLHF 训练框架决定用 gRPC 而不是 HTTP 对接 SGLang，co-located 场景下的权重热更新/显存让出这条路是走不通的，只能退回 `UpdateWeightsFromDisk`（也就是牺牲掉"不落盘直接内存传权重"这个 co-located 场景最看重的优势）——这是 gRPC 表面比 HTTP 表面**功能更窄**的一个具体、可核实的例子，不是泛泛而谈。
- **契约与实现分离**：`.proto` 契约在本仓库，但真正把这个契约跑起来的 servicer 代码在外部包 `smg-grpc-servicer`（`grpc_server.py:2,159`），本仓库只负责：① 定义协议（`.proto`）；② 提供一个启动脚手架 `serve_grpc()`（`python/sglang/srt/entrypoints/grpc_server.py:156`）去 `import` 外部包并把 `server_args`/`model_info` 传进去；③ 顺带起一个 `aiohttp` 的调试旁路（`/metrics` `/start_profile` `/stop_profile`，`grpc_server.py:64,152-153`，和 gRPC 端口不是同一个端口）。**"SGLang 支持 gRPC"这句话需要拆成两半理解**：协议契约是开源的、可审计的；能不能真的跑起来取决于一个不在 `_src/sglang` 范围内、版本号独立演进的外部依赖（`python/sglang/srt/entrypoints/grpc_server.py:236-250` 里有专门处理"装的 `smg-grpc-servicer` 版本太旧不支持某个参数"的兼容代码，说明这个外部依赖的版本漂移是真实存在的运维问题）。

---

## 3. 核心数据结构

- **`GenerateReqInput`**（`python/sglang/srt/managers/io_struct.py:162`）——真正意义上的"内部协议"。核心字段是二选一的 `text: Optional[Union[List[str], str]]`（`python/sglang/srt/managers/io_struct.py:170`）和 `input_ids: Annotated[Optional[Union[List[List[int]], List[int]]], ...]`（`python/sglang/srt/managers/io_struct.py:173-176`），**没有 `messages` 字段，也没有任何 chat template 相关的东西**——这就是为什么 `/generate` 是"原生"的：它假设调用方已经把对话渲染成了纯文本或 token id。
- **`ChatCompletionRequest`**（`python/sglang/srt/entrypoints/openai/protocol.py:823`）——OpenAI 兼容层的输入协议，字段总数 68 个（`_lab/out/compare.json` 的 `chat_request.sglang.n_fields`），其中 22 个是 OpenAI 标准字段，其余 46 个是 SGLang 私有扩展。本篇 `## 5` 决策 3 展开字段差异。
- **`AnthropicMessagesRequest`**（`python/sglang/srt/entrypoints/anthropic/protocol.py`，未被 `_lab` 统计但源码真实存在，见 2.3）——Anthropic Messages API 的输入协议，`AnthropicServing._convert_to_chat_completion_request()`（`python/sglang/srt/entrypoints/anthropic/serving.py:228`）负责把它转成 `ChatCompletionRequest`。
- **`AuthLevel`**（`python/sglang/srt/utils/auth.py:20`）——三档：`NORMAL`（默认所有端点走的旧逻辑，配了 `api_key` 就都要）、`ADMIN_OPTIONAL`（RLHF 那一族用这个，`python/sglang/srt/utils/auth.py:82` 的文档写"can be accessed without any key (if no keys configured)"）、`ADMIN_FORCE`（必须配了 `admin_api_key` 才允许访问，哪怕 `api_key` 对了也不行）。
- **`StreamChunk` / `StreamDelta` / `StreamChoice`**（`python/sglang/srt/entrypoints/openai/sse_utils.py:13,27,35`）——专门为 SSE 分片设计的 `msgspec.Struct`，`build_sse_content()` 函数负责拼装成一行 `data: {...}\n\n`。用 `msgspec` 而不是 `ChatCompletionResponse` 这个 pydantic 类本身直接序列化流式分片，是因为流式路径每秒钟要序列化几十次，`msgspec.json.Encoder()`（`python/sglang/srt/entrypoints/openai/sse_utils.py:49`，模块级单例，避免每次分片都重新构造编码器）比 pydantic 的 `.model_dump_json()` 快。
- **`ReleaseMemoryOccupationReqInput` / `ResumeMemoryOccupationReqInput`**（`python/sglang/srt/managers/io_struct.py:1937,1947`）——字段只有一个 `tags: Optional[List[str]]`，源码注释直接写"Optional tags to identify the memory region, which is primarily used for RL / Currently we only support `weights` and `kv_cache`"（`python/sglang/srt/managers/io_struct.py:1938-1939`）——这是源码自己承认这一族端点是为 RL 场景设计的，不是本库的推断。

---

## 4. 主流程走读

以 `/v1/chat/completions` 一次非流式请求为例，跳数从 HTTP 层到 `TokenizerManager` 边界为止（往下属于 `[[02-SGLang-Scheduler事件循环]]`）：

**跳 1（路由分发）**：`openai_v1_chat_completions(request: ChatCompletionRequest, raw_request: Request)`（`python/sglang/srt/entrypoints/http_server.py:1723-1729`）拿到 FastAPI 已经用 pydantic 校验、解析好的 `ChatCompletionRequest`，直接转发给 `raw_request.app.state.openai_serving_chat.handle_request(request, raw_request)`（`python/sglang/srt/entrypoints/http_server.py:1727-1729`）——路由函数本身不做任何业务逻辑，纯粹是"从 `app.state` 取出正确的 handler 对象再转发"，这个模式对所有 `/v1/*` OpenAI 端点是一致的。

**跳 2（统一入口）**：`OpenAIServingBase.handle_request()`（`python/sglang/srt/entrypoints/openai/serving_base.py:73`）先做 `_validate_request()`（`python/sglang/srt/entrypoints/openai/serving_base.py:82`），再调 `_convert_to_internal_request()`（`python/sglang/srt/entrypoints/openai/serving_base.py:93`，这是抽象方法，`OpenAIServingChat` 覆写了它），拿到 `(adapted_request, processed_request)` 之后按 `request.stream` 分流到 `_handle_streaming_request()`（`python/sglang/srt/entrypoints/openai/serving_base.py:104`）或 `_handle_non_streaming_request()`（`python/sglang/srt/entrypoints/openai/serving_base.py:108`）。

**跳 3（chat template 渲染，只在这一层发生）**：`OpenAIServingChat._convert_to_internal_request()`（`python/sglang/srt/entrypoints/openai/serving_chat.py:965`）内部调 `self._process_messages(request, is_multimodal)`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1003`），这一步里 `self.tokenizer_manager.tokenizer.apply_chat_template(...)`（`python/sglang/srt/entrypoints/openai/serving_chat.py:598` 或 `1385`，取决于是否走"先渲染成文本再单独 encode"的拆分路径）把 `messages` 列表变成一段文本或 token id 序列。**`/generate` 本身没有这一步**——`GenerateReqInput` 直接要求调用方传 `text`/`input_ids`，这也是为什么"原生 API 能做而 OpenAI 层做不到"这个问题的答案其实是反过来的：不是原生层功能更少，是**原生层把模板渲染这个决定权交还给调用方**（比如某些 agent 框架想自己控制 few-shot 拼接格式，绕开引擎自带的 chat template，就得走 `/generate`）。

**跳 4（组装内部请求）**：`_convert_to_internal_request()` 继续构造 `sampling_params = request.to_sampling_params(...)`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1005`），然后 `adapted_request = GenerateReqInput(**prompt_kwargs, sampling_params=sampling_params, return_logprob=request.logprobs, ..., bootstrap_host=request.bootstrap_host, ...)`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1051-1092`）——**每一个 `ChatCompletionRequest` 私有扩展字段（`bootstrap_host`、`lora_path`、`priority`、`session_id`……）都在这里被原样透传进 `GenerateReqInput`**，这是"OpenAI 层能做 PD 分离/LoRA/优先级调度"这些原生功能的真正原因：不是 OpenAI 协议本身支持，是 SGLang 在 `ChatCompletionRequest` 里加了这些私有字段再透传下去。

**跳 5（非流式：一次性拿结果）**：`_handle_non_streaming_request()`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1808`）里 `ret = await self.tokenizer_manager.generate_request(adapted_request, raw_request).__anext__()`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1816-1818`）——这行代码和 `/generate` 原生路径里的 `_global_state.tokenizer_manager.generate_request(obj, request).__anext__()`（`python/sglang/srt/entrypoints/http_server.py:933-935`）**调用的是同一个方法**，唯一区别是 `obj` 的构造来源不同（一个来自 `GenerateReqInput` 直接反序列化，一个来自 `ChatCompletionRequest` 转换）。

**跳 6（流式：SSE 逐块拼装）**：`_handle_streaming_request()`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1514`）调 `_generate_chat_stream()`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1542`），内部 `include_usage, continuous_usage_stats = should_include_usage(request.stream_options, self.tokenizer_manager.server_args.stream_response_default_include_usage)`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1575-1578`）——这个函数（`python/sglang/srt/entrypoints/openai/utils.py:92`）的逻辑是"请求里显式要 usage，或者服务端配了默认给 usage，两者取或"（`python/sglang/srt/entrypoints/openai/utils.py:97-98`），不是标准 OpenAI 语义里"不给就不给"的纯 opt-in，SGLang 加了一个服务端级别的默认值开关。逐 token `async for content in self.tokenizer_manager.generate_request(adapted_request, raw_request)`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1580-1582`）拿到增量结果后调用 `build_sse_content()`（定义在 `sse_utils.py`）拼成 `data: {...}\n\n` 字节串，`FastAPI` 的 `StreamingResponse` 直接把这些字节写回连接。

**对照跳（`/generate` 原生流式）**：`generate_request()`（`python/sglang/srt/entrypoints/http_server.py:894`）里流式分支直接 `yield b"data: " + dumps_json(out) + b"\n\n"`（`python/sglang/srt/entrypoints/http_server.py:905`）——**没有走 `sse_utils.py` 的 `StreamChunk`/`StreamDelta` 结构体**，是把 `TokenizerManager` 吐出来的原始字典 `dumps_json` 之后直接塞进 SSE frame，格式和 OpenAI 的 `ChatCompletionStreamResponse` 完全不同（字段是 `text`、`meta_info` 这些原生字段，不是 `choices[].delta.content`）。这印证了"两条协议不共享同一套响应序列化"，只在请求侧共享 `GenerateReqInput`，响应侧是各自独立拼装的。

**跳 7（Anthropic 层，套一层壳）**：`anthropic_v1_messages()`（`python/sglang/srt/entrypoints/http_server.py:2003`）→ `AnthropicServing.handle_messages()`（`python/sglang/srt/entrypoints/anthropic/serving.py:206`）→ `_convert_to_chat_completion_request()`（`python/sglang/srt/entrypoints/anthropic/serving.py:228`，把 Anthropic 的 `messages`/`system`/`tools` 结构翻译成 OpenAI 的 `messages`/`tools` 结构，包括 `_convert_anthropic_image_source_to_openai_part()` 这样专门处理多模态字段名差异的小函数，`python/sglang/srt/entrypoints/anthropic/serving.py:233` 起）→ 得到一个 `ChatCompletionRequest` 后**直接调用跳 2～6 的同一套 `OpenAIServingChat` 方法**（`python/sglang/srt/entrypoints/anthropic/serving.py:748,759,766` 分别调用 `_validate_request`/`_convert_to_internal_request`/`_handle_non_streaming_request`）。Anthropic 层完全没有自己往下走到 `TokenizerManager` 的路径，是纯粹的请求格式翻译 + 复用。

### 4.1 RLHF 调用时序：把 SGLang 当 rollout worker 用

`## 2` E 族的 16 条端点不是互相独立的开关，是有先后依赖的一套协议。以下两条时序线来自源码 + 上游文档 `docs/docs/advanced_features/sglang_for_rl.mdx`（**文档所述**，逐条核对过对应端点在源码里确实存在，未查证的地方另行标注）。

**时序 A · co-located（训练进程和推理进程共享同一批 GPU，典型是 verl 的默认部署方式）**：

```
① 启动阶段：--enable-memory-saver 建服务（server_args.py，见 [[09-SGLang-ServerArgs旋钮全景]] 核实具体旗标）
② 每个 RL step 循环：
   rollout 阶段  → 正常调 /generate 或 /v1/chat/completions 采样
   训练阶段前     → POST /release_memory_occupation {"tags": ["kv_cache"]}
                    （http_server.py:1484，让出 KV cache 显存给训练进程用，
                     调用前必须保证没有在途请求——源码断言"no ongoing requests"）
   训练进程更新权重（引擎外，不经过 SGLang）
   权重传回      → POST /update_weights_from_tensor
                    （http_server.py:1373，同机内存直传，不落盘；
                     tensor 必须留在 GPU 上，搬到 CPU 会破坏更新路径——docs 原话）
   训练阶段后     → POST /resume_memory_occupation {"tags": ["kv_cache"]}
                    （http_server.py:1496，KV cache 显存要回来了，
                     之后的请求会按需重建缓存，flush 掉的旧前缀缓存不会自动恢复）
   下一轮 rollout → 回到 /generate
```

**时序 B · disaggregated（训练集群和推理集群物理分离，权重经 NCCL/IB 广播）**：

```
① 建组（一次性）  → POST /init_weights_update_group
                     （http_server.py:1341，指定 master_address/master_port/world_size，
                      训练侧 TP rank 0 和所有 rollout worker 加入同一个 NCCL 通信组）
② 每次权重同步    → 训练侧 all-gather 权重后发起 broadcast
                     → 各 rollout worker 各自 POST /update_weights_from_distributed
                       （http_server.py:1395，收到广播后按 TP shard 各自加载自己需要的那部分参数）
③ 训练结束/重建组 → POST /destroy_weights_update_group
                     （http_server.py:1357，释放 NCCL 通信组资源）
```

两条时序线的分野点在 `## 5` 决策 4 已经点过：**是否要求训练进程和推理进程共享 GPU 显存地址空间**。co-located 图省事（同机内存直传，不需要额外通信组），代价是训练和推理抢同一批显卡的显存，需要 `/release_memory_occupation` 这种"临时腾地方"的机制；disaggregated 图伸缩性（推理集群可以独立于训练集群扩缩容），代价是要多维护一个 NCCL 通信组的生命周期（建组/广播/拆组三段式，对应上面三个端点）。**第三条策略"从磁盘更新"（`/update_weights_from_disk`，`python/sglang/srt/entrypoints/http_server.py:1237`）不属于这两条时序线**，它是两者都能退化到的兜底方案——docs 原话"the safest option for high availability because the checkpoint itself is the source of truth"，代价是要多一轮磁盘 I/O，`## 2.4` 里也提到这是唯一同时出现在 HTTP 和 gRPC 两个协议表面上的权重更新方式。

---

## 5. 设计决策与代价

### 决策 1：`GenerateReqInput` 作为唯一内部协议，所有兼容层向它收口

- **为什么这么设计**：让 `TokenizerManager`/`Scheduler` 只需要认识一种请求格式，不管前端来的是 OpenAI 请求、Anthropic 请求、Ollama 请求还是原生请求，调度器代码完全不用感知协议差异。新增一种兼容协议（比如未来加个 Google Gemini 兼容层）只需要写一个"翻译成 `GenerateReqInput`"的转换函数，不用碰调度/执行任何一行代码。
- **不这样会怎样**：如果 `TokenizerManager`/`Scheduler` 要同时认识 `ChatCompletionRequest`、`AnthropicMessagesRequest`、`GenerateReqInput` 三种格式，每加一种协议就要在最底层的调度逻辑里加一次分支判断，调度器的稳定性会被表层协议的迭代速度拖着走——协议层加字段的频率远高于调度器改动的频率（`ChatCompletionRequest` 68 个字段，调度器核心逻辑相对稳定）。
- **什么时候可以不这样**：如果引擎从一开始就只打算支持一种协议（比如专用的内部 RPC 服务，不对外暴露 OpenAI 兼容层），维护一层转换的成本就是纯浪费，不如让 `Scheduler` 直接认请求本身的格式。

### 决策 2：Anthropic 兼容用组合（持有 `OpenAIServingChat` 引用）而不是重新实现

- **为什么这么设计**：`AnthropicServing.__init__(self, openai_serving_chat: OpenAIServingChat)`（`python/sglang/srt/entrypoints/anthropic/serving.py:191-192`）只存一个引用，复用对方几乎所有内部方法（`_validate_request`、`_convert_to_internal_request`、`_handle_non_streaming_request`、`_process_messages`、`supports_native_reasoning_history()`……）。这样 chat template 渲染、多模态处理、工具调用约束这些复杂逻辑只维护一份，Anthropic 层要做的只是"字段名翻译"。
- **不这样会怎样**：如果重新实现一遍 tokenize/模板渲染/工具调用逻辑，两套实现会逐渐产生行为差异（比如 OpenAI 层修了一个多模态图片处理的 bug，Anthropic 层如果是独立实现就不会自动同步修复），维护成本翻倍还容易出现"同一个模型，两种协议行为不一致"的用户可见问题。
- **什么时候可以不这样**：如果 Anthropic 协议和 OpenAI 协议在语义上分歧很大（比如 Anthropic 未来加入一个 OpenAI 完全没有对应概念的核心特性），硬套皮翻译层会变得别扭，这时候独立实现反而更干净——**目前 SGLang 还没走到这一步**，`python/sglang/srt/entrypoints/anthropic/serving.py:184-188` 的文档字符串坦承自己就是"translation layer"。

### 决策 3：`ChatCompletionRequest` 用一个大 pydantic 类塞下所有私有扩展字段，而不是拆成"标准子集 + 扩展 mixin"

- **为什么这么设计**：`ChatCompletionRequest`（`python/sglang/srt/entrypoints/openai/protocol.py:823`）68 个字段里 46 个是私有扩展（`_lab/out/compare.json` 的 `chat_request.sglang.engine_specific`），包括 PD 分离用的 `bootstrap_host/bootstrap_port/bootstrap_room`（`python/sglang/srt/entrypoints/openai/protocol.py:938-940`）、DP 路由用的 `routed_dp_rank/disagg_prefill_dp_rank`（`python/sglang/srt/entrypoints/openai/protocol.py:943-945`）、结构化输出用的 `regex/ebnf`（`python/sglang/srt/entrypoints/openai/protocol.py:894-895`）、推理控制用的 `separate_reasoning/stream_reasoning`（`python/sglang/srt/entrypoints/openai/protocol.py:908-909`）。全塞进一个类，客户端只需要理解"这是一个大的可选字段集合"，任何标准 OpenAI SDK 发请求时这些字段全部留空也能正常工作（pydantic 默认值兜底）。
- **不这样会怎样**：如果严格区分"标准子协议"和"SGLang 扩展协议"两个类，标准 OpenAI 客户端库（比如官方 `openai` Python SDK）序列化请求体时可能会因为多出的扩展字段被某些严格模式的客户端库拒绝，或者服务端要多写一层"扩展字段单独解析"的逻辑——现状是简单粗暴但工程上最省事。
- **什么时候可以不这样**：如果 SGLang 要把这份协议提交给 OpenAI 或做成官方标准的一部分，私有扩展和标准字段混在一起会让协议审查变得困难；`_lab/out/compare.json` 显示 vLLM 的 `ChatCompletionRequest`（`vllm:vllm/entrypoints/openai/chat_completion/protocol.py:213`）也是同样的"一个大类"策略（同样 68 个字段），说明这是这一类引擎的共性选择，不是 SGLang 独有的权宜之计。
- **缺什么**：`ChatCompletionRequest` 相对 OpenAI 官方标准（`platform.openai.com/docs/api-reference/chat/create`，2026-08 人工核对，见 `compare.json` 的 `openai_field_list_source`）缺 8 个字段：`audio`、`function_call`、`functions`（旧版工具调用接口，OpenAI 自己都标 deprecated）、`metadata`、`modalities`（多模态输出模态选择）、`prediction`（predicted output 特性）、`service_tier`、`store`（对话持久化开关）。这 8 个字段**在 SGLang/vLLM/LMDeploy 三家里完全一致地缺失**（`compare.json` 三家的 `missing_vs_openai` 列表相同），推断（`本库推断`）是这些字段依赖 OpenAI 后端专有的账户/存储/优先级基础设施（`store`/`service_tier`/`metadata` 明显是云端概念），开源自托管引擎没有对应的底层能力去实现，不是遗漏而是无法实现。

### 决策 4：RLHF 权重更新族直接挂在主 `app` 上，只用 `AuthLevel` 区分权限，不做独立路由前缀/独立端口

- **为什么这么设计**：`http_server_engine.py` 里的 `HttpServerEngineAdapter`（`python/sglang/srt/entrypoints/http_server_engine.py:49`）类文档字符串写"You can use this class to launch a server from a VerlEngine instance"（`python/sglang/srt/entrypoints/http_server_engine.py:52-53`，`verl` 是字节跳动开源的 RLHF 训练框架），它的 `generate()`/`update_weights_from_tensor()`/`release_memory_occupation()`/`resume_memory_occupation()`（`python/sglang/srt/entrypoints/http_server_engine.py:104,78,138,141`）全部通过 HTTP POST 打到同一个 base URL 的不同路径——**训练框架把整个 SGLang server 当作一个"既能推理又能被控制显存/权重的 rollout worker"来使用**，如果控制端点和推理端点分属不同端口/不同服务，`VerlEngine` 这类调用方要多维护一份地址簿，纯粹增加复杂度。
- **不这样会怎样**：如果这批端点默认关闭（像 vLLM 的 `VLLM_SERVER_DEV_MODE` 那样），每次拉起一个 RLHF 训练用的 SGLang rollout worker 都要多传一个环境变量/命令行参数，训练脚本的可移植性变差；对 SGLang 的定位（原生支持在线 RL 训练的推理后端）而言，这批端点不是"调试专用"，是核心使用场景之一。
- **什么时候可以不这样**：给 `--api-key`/`--admin-api-key` 配置好之后，`AuthLevel.ADMIN_OPTIONAL` 会要求提供对应 key（`auth.py` 的 `decide_request_auth()` 逻辑），此时暴露面被收紧；如果部署场景是"公网直接暴露推理服务，压根不会有 RL 训练流量"，理论上可以在反向代理层直接拦截这批路径前缀，SGLang 本身没有提供"编译期直接去掉这些路由"的开关（`本库推断`：未在 `ServerArgs` 里找到类似 `--disable-rlhf-endpoints` 的旋钮，留给 `[[09-SGLang-ServerArgs旋钮全景]]` 核实）。

### 决策 5：`ChatCompletionRequest.stream_options.include_usage` 之外再加一个服务端默认值 `stream_response_default_include_usage`

- **为什么这么设计**：`should_include_usage()`（`python/sglang/srt/entrypoints/openai/utils.py:92-106`）的逻辑是"请求要 usage 或服务端默认给 usage，两者取或"（`python/sglang/srt/entrypoints/openai/utils.py:97-98`）。对某些内部平台场景，运营方想强制所有流式响应都带 token 用量统计（比如用于计费），但又不想要求每个调用方都手动传 `stream_options: {include_usage: true}`，加一个服务端默认值可以在不改客户端代码的前提下统一打开。
- **不这样会怎样**：如果严格遵循 OpenAI 标准语义（纯粹按请求里的 `stream_options` 决定），想统一采集用量数据的运营方只能在网关层做请求体注入，多一层运维复杂度。
- **什么时候可以不这样**：默认值不设置时（`stream_response_default_include_usage=False`，需要去 `[[09-SGLang-ServerArgs旋钮全景]]` 核实这个旋钮是否可从 CLI 配置），行为和标准 OpenAI 语义完全一致——这是一个"默认关闭、按需打开"的加法特性，不影响标准客户端的期望行为。

---

## 6. 同位对照（vLLM 在同一位置怎么做）

同一取证会话核对了 vLLM（sha `7ca49fbe`，2026-08-22，取证基准与本篇不同引擎，单独标注）：

**规模速览**（数字均来自 `_lab/out/api_surface.json`，两家取证时间相隔几小时内）：

| 维度 | SGLang | vLLM |
|---|---|---|
| 路由总数（AST 抽取） | 83（另有 4 条 Ollama 路由被漏抽，见 `## 2.2`） | 63 |
| `/v1/*` 路由数 | 17 | 21 |
| 协议类数量 | 86（单文件 `protocol.py`） | 188（按 endpoint 拆包到多文件） |
| Anthropic 兼容 | 有（`/v1/messages`，组合方式） | 有（`/v1/messages`，继承方式） |
| Ollama 兼容 | 有（`/api/*`，AST 漏抽） | 无（`unique_paths` 里未出现 `/api/*` 系列） |
| RLHF 端点默认可用性 | 始终挂载，`AuthLevel` 分级 | 默认不挂载，需 `VLLM_SERVER_DEV_MODE=1` |
| gRPC 入口的协议契约 | 部分在仓库内（`proto/sglang/runtime/v1/sglang.proto`） | 不在仓库内（`smg_grpc_proto` 外部包） |

- **规模对比**：vLLM 抽到 **63** 条路由（21 条 `/v1/*`），SGLang **83**（17 条 `/v1/*`，另有 4 条 Ollama 路由因动态路径被漏抽，见 `## 2.2`）；协议类数量 vLLM **188** 个远多于 SGLang 的 **86** 个——vLLM 的 `entrypoints` 已经按 family 拆包成 `vllm/entrypoints/<family>/<endpoint>/{api_router,protocol,serving}.py` 的目录结构（`_PLAN.md` §7 已知踩坑 3），每个 endpoint 独立一个 protocol 文件，粒度天然更细；SGLang 把几乎所有 OpenAI 协议类塞进一个 2116 行的 `protocol.py`，粒度更粗但文件更集中。**协议类数量差 2 倍以上不代表功能差 2 倍，是文件组织粒度不同**，这是读这两个数字时最容易踩的坑。
- **两家都在做 Anthropic 兼容，但包装方式完全相反**：SGLang 用**组合**（`AnthropicServing` 持有一个 `openai_serving_chat` 引用，`python/sglang/srt/entrypoints/anthropic/serving.py:191-192`）；vLLM 用**继承**——`class AnthropicServingMessages(OpenAIServingChat)`（`vllm:vllm/entrypoints/anthropic/serving.py:99`），Anthropic 的 serving 类直接是 OpenAI serving 类的子类，复用父类方法靠 `super()` 或直接调用继承来的方法，而不是持有引用转发调用。两种方式最终效果类似（都是"复用 OpenAI 层的 tokenize/模板逻辑"），但组合更容易在未来把 Anthropic 逻辑迁移到不依赖 `OpenAIServingChat` 具体实现的独立类（只要接口不变），继承则把两者的生命周期和内部状态耦合得更紧——**两家都在做同一件事这个事实本身说明 Anthropic Messages API 已经变成事实上的"第二标准"，不只 Claude 生态用它**，这是本篇给出的一个跨引擎观察，不是某一家的设计选择。
- **RLHF 控制面的物理隔离程度不同**（`## 0`/`## 5` 已展开核心结论，这里补充路由细节）：vLLM 功能对等的端点（`/sleep` `/wake_up` `/is_sleeping`、`vllm:vllm/entrypoints/serve/dev/sleep/api_router.py:22,33,46`；`/init_weight_transfer_engine` `/start_weight_update` `/update_weights` `/finish_weight_update` `/weight_info`、`vllm:vllm/entrypoints/serve/dev/rlhf/api_router.py:158,176,188,206,224`；外加一个通用的 `/collective_rpc`、`vllm:vllm/entrypoints/serve/dev/rpc/api_router.py:24`，可以对 worker 任意方法发起 RPC）全部注册在 `register_vllm_dev_api_routers()`（`vllm:vllm/entrypoints/serve/__init__.py:43`）里，只有 `envs.VLLM_SERVER_DEV_MODE` 为真（默认 `False`，`vllm:vllm/envs.py:168`）时才会在 `register_api_routers()`（`vllm:vllm/entrypoints/launchers/api_server/routers.py:33-36`）里被挂载，挂载时打印安全警告。SGLang 侧对等的 16 条端点（`## 2` E 族）**始终挂载**，只用 `AuthLevel` 做权限分级，不做"开发模式/生产模式"的二元开关。
- **`/collective_rpc` 这种"一个端点打任意 worker 方法"的通用 RPC 通道，SGLang 没有对应物**：SGLang 的每个 RLHF 操作都是独立的强类型端点（`UpdateWeightsFromTensorReqInput` 等各自有专门的 pydantic/msgspec schema），vLLM 除了专用端点外还留了一条"万能后门"——`collective_rpc(method, args, kwargs)`（`vllm:vllm/entrypoints/serve/dev/rpc/api_router.py:24-52`）可以调用 worker 上任意已注册的方法，注释里写"For security reason, only serialized string args/kwargs are passed"（`vllm:vllm/entrypoints/serve/dev/rpc/api_router.py:36-37`）——这条路径的攻击面明显更大（能调用的方法集合取决于 worker 类实现，不是 HTTP 层能审计的），vLLM 自己也把它归进默认关闭的 `dev` 路由里，等于承认了这一点。
- **两家的 Python gRPC 入口，用的是同一个第三方厂商的同一族包**：`## 2.4` 已经展开 SGLang 的 `grpc_server.py` 委托给外部包 `smg-grpc-servicer`（`python/sglang/srt/entrypoints/grpc_server.py:2`）；核对 vLLM 之后发现它的 `vllm/entrypoints/grpc_server.py`（文档字符串"Starts a gRPC server backed by AsyncLLM, using the VllmEngineServicer from the smg-grpc-servicer package"）**连协议定义都不在自己仓库里**——`from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc`（`vllm:vllm/entrypoints/grpc_server.py:31`）、`from smg_grpc_servicer.vllm.servicer import VllmEngineServicer`（`vllm:vllm/entrypoints/grpc_server.py:33`）——同一个 `smg-grpc-servicer` 包下分出 `smg_grpc_servicer.sglang.*` 和 `smg_grpc_servicer.vllm.*` 两个子模块，分别给两家引擎当 gRPC 后端。**两大开源推理引擎的 gRPC 支持事实上外包给了同一个第三方**，谁维护、发版节奏是否与引擎主干同步，都不在 `_src/vllm`、`_src/sglang` 任何一边的源码范围内，属于「未查证」——但比较起来 SGLang 至少把协议契约（`.proto` 文件）留在自己仓库里可审计，vLLM 连契约本身都是外部包的一部分。**vLLM 仓库里另有一份完全不相关的原生 gRPC 定义**——`rust/proto/inference.proto`（`package vllm`，`service Inference { rpc Generate / rpc GenerateStream }`），只有两个 RPC，规模和用途都与 `entrypoints/grpc_server.py` 面向的完整推理服务差远了，推断（`本库推断`）这是给 Rust 侧某个路由/网关组件用的极简接口，不能拿它当作"vLLM 有原生 gRPC 支持"的证据——两份 gRPC 相关代码字面意义上都叫"gRPC"，指向的却是完全不同的两个系统。

---

## 7. 踩坑与反直觉

1. **83 条不是真实数字，至少是 87**——`## 2.2` 已展开：Ollama 兼容的 4 条路由用 `os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat")` 这种非字面量路径注册（`python/sglang/srt/entrypoints/http_server.py:1973,1979,1987,1993`），AST 静态扫描（`_lab/api_surface.py`）解析不出字符串常量就不会记进 `unique_paths`。**这正是 `_lab/out/compare.json` 的 `notes.caveat`（"路由集合来自装饰器静态抽取；条件注册（if 分支里 add_api_route）可能漏"）预警的情况被我在这篇取证里实打实撞上了**——空口白牙的"可能漏"变成了有行号的"确实漏了这 4 条"。
2. **`grpc_server.py` 在 `route_files` 列表里，但对 83 条贡献是 0**——不是文件没被扫，是这个文件里的三个端点（`/metrics` `/start_profile` `/stop_profile`）用的是 `aiohttp` 的 `app.router.add_get/add_post`（`python/sglang/srt/entrypoints/grpc_server.py:64,152-153`），AST 抽取器只认 `@app.get/@app.post` 这种 FastAPI 装饰器语法，两种注册方式长得完全不同，抽取器**不是漏看了这个文件，是压根不认识这种写法**。更进一步，真正的 gRPC servicer 实现根本不在这个仓库里——`python/sglang/srt/entrypoints/grpc_server.py:2` 的文档字符串直接写"Thin gRPC server wrapper — delegates to smg-grpc-servicer package"，这意味着**"SGLang 的 gRPC 入口"这个说法本身需要打折扣**：SGLang 仓库只提供一个启动脚手架 + HTTP 调试旁路，实际的 gRPC 协议实现是一个外部依赖包，装不装、装哪个版本都会改变 gRPC 路径的行为，这部分**在 `_src/sglang` 里查不到，标注「未查证」**。
3. **同一个函数被注册成两个路径，读代码容易以为是两个独立实现**：`/health` 和 `/health_generate` 都指向 `health_generate`（`python/sglang/srt/entrypoints/http_server.py:656`），`/get_weight_version` 和 `/weight_version` 都指向 `weight_version`（`python/sglang/srt/entrypoints/http_server.py:778`），`/get_server_info`/`server_info`、`/v1/tokenize`/`/tokenize`、`/v1/detokenize`/`/detokenize` 都是这个模式——这是"新路径 + 保留旧路径做向后兼容别名"的常见做法，但如果只搜索 `def xxx` 定位实现，会漏看到它其实还服务另一条路径。
4. **管理端点默认"无 key 即无认证"**：`AuthLevel.ADMIN_OPTIONAL` 的语义是"如果没配置任何 key，直接放行"（`python/sglang/srt/utils/auth.py:82-84` 文档），也就是说一个**没传 `--api-key`/`--admin-api-key` 的 SGLang server**，任何能连到端口的人都可以调 `/release_memory_occupation`（让服务瞬间失去 GPU 显存）、`/update_weights_from_disk`（换掉正在提供服务的模型权重）——这不是 bug，是"信任内网部署"的默认假设（RL 训练集群通常在隔离网络里），但如果不小心把这样配置的 server 暴露到公网，后果是实打实的拒绝服务/数据污染，**没有代码层面的强制提醒**，仅在文档层面存在。
5. **原生 `/generate` 的流式响应和 OpenAI `/v1/chat/completions` 的流式响应是两套完全不同的 JSON 结构**：前者是 `dumps_json(out)` 直接序列化 `TokenizerManager` 内部字典（字段名 `text`、`meta_info`），后者走 `sse_utils.py` 的 `StreamChunk`（字段名 `choices[].delta.content`，OpenAI 标准形状）。如果调用方以为"反正都是 SSE，格式应该差不多"直接复用同一套解析代码，会在切换端点时直接解析失败——两条路径除了共享 `TokenizerManager` 这个数据源之外，响应序列化是完全独立维护的两套代码。
6. **gRPC 上的"OpenAI 兼容"是名不副实的**：`.proto` 契约（`## 2.4`）把 `ChatComplete`/`Complete`/`OpenAIEmbed`/`OpenAIClassify`/`Score`/`Rerank` 标成"OpenAI-compatible RPCs"，听起来像是和 HTTP `/v1/chat/completions` 平级的强类型协议，但它们的请求体 `OpenAIRequest` 只有一个字段 `bytes json_body`（`proto/sglang/runtime/v1/sglang.proto:295`）——**gRPC 在这六个 RPC 上完全没有兑现"强类型契约"这个 gRPC 本该提供的核心价值**，客户端还是要自己拼一份和 HTTP 请求体一模一样的 JSON 再塞进 bytes 字段，唯一区别是传输帧从 HTTP/1.1 换成了 HTTP/2。第一次看到 `.proto` 里这几个 RPC 名字，很容易误以为 gRPC 路径对 `ChatCompletionRequest` 也做了字段级类型校验，实际上校验（如果有的话）还是要等到 `bytes` 被反序列化成 JSON 之后，在 `smg-grpc-servicer` 内部某处发生——这部分具体怎么做的，本仓库看不到，标注「未查证」。
7. **`tool_server.py` 在 `route_files` 里，但和 `grpc_server.py` 一样对 83 条路由贡献是 0**——原因不同于 grpc_server.py（那是注册方式 AST 认不出），`tool_server.py`（191 行）里压根**没有任何 HTTP 路由注册代码**，`ToolServer`/`MCPToolServer`/`DemoToolServer`/`NativeToolServer` 这些类是给 `/v1/responses` 端点内建工具调用（built-in tool calling，MCP 协议客户端）用的业务逻辑模块，不是一个独立的 HTTP 入口——`_lab/api_surface.py` 把它收进 `route_files` 是因为文件路径匹配了 glob 模式（`entrypoints/openai/*.py`），但扫描器扫完发现零路由，这是**扫描范围包含了本不该被当作"路由候选文件"的文件**，和 `grpc_server.py` 那种"该抽的没抽到"是两种不同性质的空结果，混在一起统计容易掩盖问题的真实原因。

---

## 8. 可改进点

以下均按 `本库推断` 标注，基于本篇读到的代码结构，不涉及编造 PR/issue 号（遵守 `_PLAN.md` §1.6）：

- **`_lab/api_surface.py` 的路由抽取器应该对 `os.environ.get("XXX", "default")` 这种模式做特判**：只要第二个参数是字符串字面量，就可以提取出默认路径记入统计，并额外标注"动态路径，可被环境变量覆盖"——这样能同时修好 `## 7` 第 1 条踩坑，且不需要真正执行代码（AST 层面就能识别这个固定模式，`ast.Call` 里 `func.attr == "get"` 且 `args[1]` 是 `ast.Constant`）。
- **`_lab/api_surface.py` 应该识别 `aiohttp` 的 `add_get`/`add_post`/`add_route` 调用**，即使只是额外统计一个"非 FastAPI 路由"计数，也能避免 `route_files` 列表里出现"扫了但贡献 0 条"这种误导性的空结果（`_PLAN.md` §7 明确警告过"空结果要当 bug 查，不许当事实写"，`grpc_server.py` 目前是介于"真空"和"漏抽"之间的灰色地带，值得单独归类而不是沉默地记 0）。
- **`n_protocol_classes` 的统计口径应该扩展到 `entrypoints/anthropic/protocol.py` 和 Ollama 相关协议文件**，否则"86 个协议类"这个数字会让读者误以为 Anthropic/Ollama 兼容层没有自己的协议定义（实际上有，只是没被这次 glob 命中）。
- **`AuthLevel.ADMIN_OPTIONAL` 的"无 key 即放行"默认行为，值得在 `ServerArgs` 启动日志里显式打印一行警告**（类比 vLLM 的 `VLLM_SERVER_DEV_MODE` 打印"SECURITY WARNING"那样），目前只在源码注释里说明，运行时没有任何提示——这对第一次部署的用户是一个容易踩的信任假设落差。
- **`.proto` 里"OpenAI-compatible RPCs"这组注释分类容易造成误导**（`## 7` 第 6 条），建议要么把这六个 RPC 的分组注释改成更准确的措辞（比如"JSON pass-through RPCs, no field-level typing"），要么真的给它们定义结构化 message（哪怕只覆盖最常用的 `model`/`messages`/`stream` 几个字段，其余走一个 `extra: bytes` 兜底字段）——现状是协议文件的分组名字暗示了一种契约强度，实际代码没有兑现。
- **`route_files` 的统计口径应该把"贡献 0 条路由"的文件单独列一类，并标注零路由的原因**（`grpc_server.py` 是"注册方式不认识"，`tool_server.py` 是"压根没有路由，只是业务逻辑模块"）——目前两种情况在 `api_surface.json` 里长得一模一样（都只是在 `route_files` 数组里挂个名，`routes` 数组里查不到任何一条来自这个文件），下游读者没法只看 JSON 区分这是"漏抽 bug"还是"文件本来就不该被扫进来"，容易造成不必要的怀疑或者相反的盲目信任。

---

## 9. 自测题与延伸阅读

**闭卷自测题**（先合上本篇再答，答完再翻回去对）：

1. `/generate`、`/v1/chat/completions`、`/v1/messages` 三条路径，最终都收口到哪一个内部数据结构？chat template 渲染发生在这三层里的哪一层？
2. 为什么 `AuthLevel.ADMIN_OPTIONAL` 的端点在没有配置 `--api-key`/`--admin-api-key` 时是完全开放的？这是 bug 还是设计假设？依据是什么？
3. `_lab/api_surface.py` 统计出的 83 条路由里，为什么少了 Ollama 兼容的 4 条？这类问题在 AST 静态抽取工具里还可能以什么形式出现（提示：`aiohttp` 的例子）？
4. SGLang 和 vLLM 都实现了 Anthropic Messages API 兼容层，但一个用组合一个用继承，各自的取舍是什么？
5. `stream_options.include_usage` 在 SGLang 里除了请求方显式指定之外，还有什么途径能被打开？这和纯 OpenAI 标准语义有什么区别？
6. `ChatCompletionRequest` 相对 OpenAI 标准缺的 8 个字段，为什么三家开源引擎（SGLang/vLLM/LMDeploy）都缺同样这几个？

**延伸阅读**（仅从 `_PLAN.md` §6 名册挑选）：

- [[01-SGLang-全景与代码地图]] —— 先建立 SGLang 整体坐标系，再回来看本篇的 HTTP 层细节
- [[09-SGLang-ServerArgs旋钮全景]] —— 本篇多处标注"需要去 ServerArgs 核实"的旋钮（`stream_response_default_include_usage`、admin key 配置方式），在那篇有系统性梳理
- [[08-vLLM-HTTP-API表面全解]] —— vLLM 侧的对应篇章，`## 6` 的跨引擎对照细节可以在那篇找到 vLLM 视角的镜像描述
