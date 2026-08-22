# vLLM HTTP API 表面全解

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：63 条路由散在 30 个文件里，`--api-key` 只护住四个前缀

## 0. 结论先行

- **`vllm/entrypoints/openai/api_server.py` 已经不是主装配文件了，是一个已废弃的转发壳**：整个文件体只剩 59 行，开头就是 `warnings.warn("...api_server\` is deprecated and will likely be unsupported in a future version. Use the corresponding function from \`vllm.entrypoints.launchers\` instead.")`（`vllm/entrypoints/openai/api_server.py:24-30`），真正的 `build_app`/`register_api_routers`/`main` 都是从 `vllm.entrypoints.launchers.*` 原样 re-export 过来的（`vllm/entrypoints/openai/api_server.py:5-22`）。**这意味着"看文件名猜装配点"这个直觉在这一版会直接踩空**——真正的主装配点是 `vllm/entrypoints/launchers/app.py` 的 `build_app()`（`vllm/entrypoints/launchers/app.py:19`），本篇 `## 1`/`## 2` 会给出完整链路。
- **`--api-key` 不是全局认证，只保护四个路径前缀**：`AuthenticationMiddleware` 的 `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")`（`vllm/entrypoints/serve/middleware/authenticate.py:11`）——凡是不落在这四个前缀下的路由（`/invocations`、`/classify`、`/score`、`/rerank`、`/pooling`、`/generative_scoring`、`/sleep`、`/collective_rpc`、`/scale_elastic_ep`、`/fault_tolerance/*`、`/tokenize`……）配了 `--api-key` 也照样不设防。这不是本库读代码读出的"隐藏 bug"，是 vLLM 自己在 CLI 帮助文本里明写的："Warning: this only authenticates endpoints under the \`/v1\`, \`/v2\`, and \`/inference\` path prefixes... Do not rely on \`--api-key\` alone to secure vLLM"（`vllm/entrypoints/openai/cli_args.py:287-292`），并且专门用一整篇文档 `docs/usage/security.md` 逐条列出哪些端点在哪些条件下不受保护。`## 5` 决策 1、`## 7` 第 1 条会展开。
- **63 条路由不是一族，而是叠着四种互不相同的"要不要注册"闸门**：① 按 `supported_tasks`（`generate`/`pooling`/`transcription`/`realtime`/`render` 等启动时确定的任务集合）决定要不要注册整族路由；② 全局二元开关 `VLLM_SERVER_DEV_MODE`（默认 `False`，一次性打开/关闭 dev 族全部 5 个文件）；③ 单个 CLI 旗标（`--enable-fault-tolerance`、`--tokens-only`、`--enable-tokenizer-info-endpoint`）逐一控制单条路由；④ 环境变量 + 可选 SDK 探测的组合（`VLLM_ENABLE_COHERE_API=1` 且 `pip install cohere` 都满足才会注册 `/cohere/v2/chat`，两个条件差一个都是静默不注册，只打不同级别的日志）。`## 2.2`、`## 5` 决策 6 逐条给出。
- **vLLM 的"第三套协议"其实是两套不同粒度的 Cohere 兼容，不是一套**：`/cohere/v2/chat`（完整 Cohere Chat v2 语义，`vllm/entrypoints/cohere/api_router.py:103`）需要显式 `VLLM_ENABLE_COHERE_API=1` 才注册，默认关闭；但 `/v2/embed`（`vllm/entrypoints/pooling/embed/api_router.py:46`）和 `/v2/rerank`（`vllm/entrypoints/pooling/scoring/api_router.py:105`）是 pooling 族的一部分，只要模型支持对应任务就**始终注册，没有任何开关**。同一个厂商的兼容协议，三个端点走两种完全不同的默认可用性策略。
- **`/v1/chat/completions/render` 不是"调试用的旁路"，是内部生成路径本来就在用的同一份代码被单独开了个口子**：`OpenAIServingChat.render_chat_request()`（`vllm/entrypoints/openai/chat_completion/serving.py:217`）和独立的 `ServingRender.render_chat_request()`（`vllm/entrypoints/scale_out/render/serving.py:64`）内部都只是转发到同一个 `self.online_renderer.render_chat(...)`（分别在 `vllm/entrypoints/openai/chat_completion/serving.py:240`、`vllm/entrypoints/scale_out/render/serving.py:80`），`ServingRender` 的文档字符串原话是"This is the authoritative implementation used directly by the GPU-less render server and delegated to by OpenAIServingChat"（`vllm/entrypoints/scale_out/render/serving.py:69-70`）——render 端点暴露的就是 `/v1/chat/completions` 内部第一步在做的事，vLLM 甚至专门做了一个只跑这一步、不需要 GPU 的独立进程（`build_and_serve_renderer()`，`vllm/entrypoints/launchers/render/entry.py:20`）。`## 4`、`## 5` 决策 4 展开。
- **`/collective_rpc` 叠了两层暴露面**：既需要 `VLLM_SERVER_DEV_MODE=1` 才会注册（`vllm/entrypoints/serve/__init__.py:43,57-59`），注册之后又**不受 `--api-key` 保护**（不落在 `GUARDED_PREFIX` 里）——`docs/usage/security.md:231` 原话把它标成"extremely dangerous"。这是本篇能给出的、证据链最完整的一条"开发者后门"案例。
- **`supported_tasks`（决定一大批路由生死的核心变量）不是 CLI 旗标，是运行时问引擎要来的事实**：`build_and_serve()`（`vllm/entrypoints/launchers/api_server/entry.py:132`）里 `supported_tasks = await engine_client.get_supported_tasks()` 要等模型权重加载完才能确定——同一条 `vllm serve` 命令换一个模型，挂出来的路由集合可能完全不同，这不是能从命令行参数静态推断的属性。
- **OpenAI / Anthropic / Cohere 三套协议的错误处理接入深度完全不同**：OpenAI 原生 `ErrorResponse` 不需要转换；Anthropic 每个 handler 手动调用 `translate_error_response()`（`vllm/entrypoints/anthropic/api_router.py:39-48`）逐个转换；Cohere 干脆用一个全局 `BaseHTTPMiddleware`（`vllm/entrypoints/cohere/api_router.py:146-171`）拦截重写——三种做法对应三种协议各自接入 vLLM 内部错误体系的深度，`## 3` 展开。

---

## 1. 它在系统里的位置

`vllm serve` 命令最终落到 `vllm.entrypoints.launchers.api_server.entry` 的 `main()`（旧壳文件转发处见 `vllm/entrypoints/openai/api_server.py:57-59`），一路调用到 `build_app()`（`vllm/entrypoints/launchers/app.py:19-54`）——这是全仓库唯一构造 `FastAPI()` 实例的地方：

```python
if args.disable_fastapi_docs:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)
elif args.enable_offline_docs:
    app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
else:
    app = FastAPI(lifespan=lifespan)
app.state.args = args
app.root_path = args.root_path

register_api_routers(args, app, supported_tasks, model_config)   # <- 63 条路由从这里长出来
attach_endpoint_plugins(app, supported_tasks)                    # <- 第三方插件路由，注册顺序在核心路由之后
init_exception_handler(app)
init_entrypoints_middleware(args, app, supported_tasks)          # <- --api-key 中间件在这里挂上
app = sagemaker_standards_bootstrap(app)
```
（`vllm/entrypoints/launchers/app.py:34-54`，逐行对应上面注释）

`register_api_routers()`（`vllm/entrypoints/launchers/api_server/routers.py:12`）是本篇 `## 2` 要展开的核心函数——它不直接注册路由，而是按条件把注册工作**转包**给 9 个不同子系统各自的 `attach_router`/`register_xxx_api_routers` 函数。这个函数本身只有 63 行，但决定了 63 条路由里哪些会真的出现在这次启动的 `app` 上。

`register_api_routers()` 第二个参数 `supported_tasks` 贯穿全篇 `## 2.2`/`## 2.5` 的条件表——它**不是一个 CLI 旗标，是启动时问引擎要来的运行时事实**：`build_and_serve()`（`vllm/entrypoints/launchers/api_server/entry.py:115-136`）里 `supported_tasks = await engine_client.get_supported_tasks()`（`:132`）先向引擎查询"这个模型实际支持哪些任务"，再拿着这个结果调 `build_app(args, supported_tasks, model_config)`（`:135`）。这意味着"这次启动到底会挂哪些路由"这件事，**要等模型权重加载完、引擎能回答"我支持什么任务"之后才能确定**——同一份 `vllm serve` 命令行、同一组 CLI 旗标，换一个模型（比如从一个纯 chat 模型换成一个同时支持 chat 和 embedding 的模型）挂出来的路由集合可能完全不同，这也是为什么本篇 `## 2.2`/`## 2.5` 反复强调"随任务注册"的路由无法只从 CLI 旗标推断,必须知道加载的是什么模型。

值得单独说明的是**渲染专用进程**这条不常见的路径：`build_and_serve_renderer()`（`vllm/entrypoints/launchers/render/entry.py:20-38`）调用的也是同一个 `build_app()`，但传入 `supported_tasks=("render",)`（`vllm/entrypoints/launchers/render/entry.py:38`），这样 `register_api_routers()` 里所有依赖 `"generate"` 任务的条件分支都不会触发，最终这个进程上只会挂 `/v1/chat/completions/render`、`/v1/completions/render`、`/v1/models`、`/ping` 这类不需要 GPU 的路由——这是一个**完全独立的部署形态**，不是"同一个 server 多加几条路由"。

本篇只看 HTTP 表面长什么样、哪些路由在什么条件下存在、协议之间怎么互相复用；不看 `AsyncLLM`/`EngineCore` 内部的调度和执行逻辑（那是 `[[02-vLLM-V1架构与EngineCore循环]]` 的范围），也不展开 KV 传输/PD 分离的内部机制（那是 `[[12-vLLM-PD分离与KV-Connector]]` 的范围，本篇只在 `kv_transfer_params` 字段和 `/inference/v1/generate` 处标出坐标）。

---

## 2. 代码地图（文件 → 职责，带行号）

### 2.1 装配链（从 CLI 到 63 条路由，按调用顺序）

| 步骤 | 函数 | 文件:行 | 职责 |
|---|---|---|---|
| 1 | `build_app()` | `vllm/entrypoints/launchers/app.py:19` | 唯一构造 `FastAPI()` 实例的地方，随后依次调路由注册、插件挂载、异常处理器、中间件 |
| 2 | `register_api_routers()` | `vllm/entrypoints/launchers/api_server/routers.py:12` | 63 条路由的总闸门，按 `supported_tasks`/`VLLM_SERVER_DEV_MODE`/CLI 旗标分派给 9 个子注册函数 |
| 3 | `register_vllm_serve_api_routers()` | `vllm/entrypoints/serve/__init__.py:19` | **始终注册**：instrumentator（健康检查/指标）、LoRA、profile、tokenize |
| 4 | `register_vllm_dev_api_routers()` | `vllm/entrypoints/serve/__init__.py:43` | 仅 `VLLM_SERVER_DEV_MODE=1` 时注册：cache/rlhf/rpc/server_info/sleep 五族，注册时打印 `SECURITY WARNING`（`:44-47`） |
| 5 | `register_generate_api_routers()` | `vllm/entrypoints/generate/api_router.py:21` | `"generate"` 任务下注册：chat/responses/completion/anthropic/cohere/generative_scoring |
| 6 | `register_scale_out_api_routers()` | `vllm/entrypoints/scale_out/factories.py:61` | `"generate"` 或 `"render"` 任务下注册：render/derender **无条件**，token-in-token-out 需 `"generate"` |
| 7 | `register_pooling_api_routers()` | `vllm/entrypoints/pooling/factories.py`（本篇未展开内部） | 任一 pooling 任务下注册：classify/embed/pooling/score/rerank |
| 8 | `register_fault_tolerance_api_router()` | `vllm/entrypoints/serve/fault_tolerance/api_router.py:99` | 仅 `args.enable_fault_tolerance` 为真时注册 |
| 9 | `attach_endpoint_plugins()` | `vllm/plugins/endpoint_plugins/interface.py`（本篇未展开） | 第三方插件路由，**注册顺序在所有核心路由之后**，且不在 `GUARDED_PREFIX` 白名单设计的考虑范围内（`docs/usage/security.md:343` 专门警告过这一点） |

### 2.2 路由分族表（63 条 AST 抽取结果，按注册条件分组）

来源：`_lab/out/api_surface.json` 的 `engines.vllm.routes`，逐条核对过行号。63 条对应 61 个 unique path——两处路径复用：`/ping` 同时注册 `GET`/`POST` 指向同一个 `ping` 函数（`vllm/entrypoints/serve/sagemaker/api_router.py:50`）；`/abort_requests` 由**两个不同文件的两个不同函数**分别注册（见 `## 7` 第 2 条）。

**A 族 · 核心生成协议（`"generate"` 任务，无额外开关）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/v1/chat/completions` | POST | `create_chat_completion` | `vllm/entrypoints/openai/chat_completion/api_router.py:53` |
| `/v1/chat/completions/batch` | POST | `create_batch_chat_completion` | `vllm/entrypoints/openai/chat_completion/api_router.py:90` |
| `/v1/completions` | POST | `create_completion` | `vllm/entrypoints/openai/completion/api_router.py:46` |
| `/v1/responses` | POST | `create_responses` | `vllm/entrypoints/openai/responses/api_router.py:60` |
| `/v1/responses/{response_id}` | GET | `retrieve_responses` | `vllm/entrypoints/openai/responses/api_router.py:82` |
| `/v1/responses/{response_id}/cancel` | POST | `cancel_responses` | `vllm/entrypoints/openai/responses/api_router.py:112` |
| `/generative_scoring` | POST | `create_generative_scoring` | `vllm/entrypoints/generate/generative_scoring/api_router.py:39` |

**B 族 · Anthropic 兼容（无额外开关，随 `"generate"` 任务注册）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/v1/messages` | POST | `create_messages` | `vllm/entrypoints/anthropic/api_router.py:63` |
| `/v1/messages/count_tokens` | POST | `count_tokens` | `vllm/entrypoints/anthropic/api_router.py:101` |

**C 族 · Cohere 兼容（两种不同的注册条件，见 `## 5` 决策 6）**

| 路径 | 方法 | 处理函数 | 文件:行 | 注册条件 |
|---|---|---|---|---|
| `/cohere/v2/chat` | POST | `chat_v2` | `vllm/entrypoints/cohere/api_router.py:115` | `VLLM_ENABLE_COHERE_API=1` 且 `cohere` SDK 已安装（`:229-241`），两者缺一不注册 |
| `/v2/embed` | POST | `create_cohere_embedding` | `vllm/entrypoints/pooling/embed/api_router.py:56` | 无开关，随 pooling `embed` 任务注册 |
| `/v2/rerank` | POST | `do_rerank_v2` | `vllm/entrypoints/pooling/scoring/api_router.py:114` | 无开关，随 pooling `score` 任务注册 |

**D 族 · Pooling / 打分 / 排序（随对应 pooling 任务注册）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/classify` | POST | `create_classify` | `vllm/entrypoints/pooling/classify/api_router.py:29` |
| `/v1/embeddings` | POST | `create_embedding` | `vllm/entrypoints/pooling/embed/api_router.py:38` |
| `/pooling` | POST | `create_pooling` | `vllm/entrypoints/pooling/pooling/api_router.py:37` |
| `/score` | POST | `create_score` | `vllm/entrypoints/pooling/scoring/api_router.py:47` |
| `/v1/score` | POST | `create_score_v1` | `vllm/entrypoints/pooling/scoring/api_router.py:62` |
| `/rerank` | POST | `do_rerank` | `vllm/entrypoints/pooling/scoring/api_router.py:81` |
| `/v1/rerank` | POST | `do_rerank_v1` | `vllm/entrypoints/pooling/scoring/api_router.py:95` |

D 族每条路由的注册条件比其他族更细粒度——`register_pooling_api_routers()`（`vllm/entrypoints/pooling/factories.py:110-140`）先整体判断 `model_config is None` 就直接返回、什么都不注册（`:115-116`），再逐条判断：`/pooling` 只要 `model_config.get_pooling_task(supported_tasks)` 不是 `None` 就注册（`:118-123`）；`/classify` 单独看 `"classify" in supported_tasks`（`:125-130`）；`/v1/embeddings`/`/v2/embed` 单独看 `"embed" in supported_tasks`（`:132-135`）；`/score`/`/v1/score`/`/rerank`/`/v1/rerank`/`/v2/rerank` 这一组共用 `enable_scoring_api(supported_tasks, model_config)`（`:137-140`）——而这个函数内部还藏着一个和任务集合无关的额外条件：如果模型的 pooling 任务是 `classify` 但 `model_config.hf_config.num_labels != 1`，打分 API 会被关掉（`vllm/entrypoints/pooling/utils.py:243-247`，"Scoring API is only enabled for num_labels == 1"）——**这条路由是否存在不仅取决于启动参数，还取决于当前加载的模型本身的分类头配置**，是本篇所有条件注册里唯一一处依赖模型权重元信息（而不是纯 CLI/环境变量）的例子。

**E 族 · Render / Derender / Token-in-Token-out（"scale-out" 家族，见 `## 4`/`## 5` 决策 4）**

| 路径 | 方法 | 处理函数 | 文件:行 | 注册条件 |
|---|---|---|---|---|
| `/v1/chat/completions/render` | POST | `render_chat_completion` | `vllm/entrypoints/scale_out/render/api_router.py:37` | `"generate"` **或** `"render"` 任务，无条件注册 |
| `/v1/completions/render` | POST | `render_completion` | `vllm/entrypoints/scale_out/render/api_router.py:62` | 同上 |
| `/v1/chat/completions/derender` | POST | `derender_chat_completion` | `vllm/entrypoints/scale_out/derender/api_router.py:40` | 同上 |
| `/v1/completions/derender` | POST | `derender_completion` | `vllm/entrypoints/scale_out/derender/api_router.py:90` | 同上 |
| `/inference/v1/generate` | POST | `generate` | `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:58` | 仅 `"generate"` 任务 |
| `/abort_requests`（scale-out 版） | POST | `abort_requests` | `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:80` | 仅 `"generate"` 任务 **且** `args.tokens_only=True`（`:77`） |

**F 族 · 始终挂载的运维基础端点（`register_vllm_serve_api_routers`）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/v1/models` | GET | `show_available_models` | `vllm/entrypoints/openai/models/api_router.py:21` |
| `/v1/load_lora_adapter` | POST | `load_lora_adapter` | `vllm/entrypoints/serve/lora/api_router.py:44`（**内部再判一次** `VLLM_ALLOW_RUNTIME_LORA_UPDATING`，见下） |
| `/v1/unload_lora_adapter` | POST | `unload_lora_adapter` | `vllm/entrypoints/serve/lora/api_router.py:62`（同上） |
| `/start_profile` | POST | `start_profile` | `vllm/entrypoints/serve/profile/api_router.py:22` |
| `/stop_profile` | POST | `stop_profile` | `vllm/entrypoints/serve/profile/api_router.py:30` |
| `/tokenize` | POST | `tokenize` | `vllm/entrypoints/serve/tokenize/api_router.py:46` |
| `/detokenize` | POST | `detokenize` | `vllm/entrypoints/serve/tokenize/api_router.py:71` |
| `/tokenizer_info` | GET | `get_tokenizer_info` | `vllm/entrypoints/serve/tokenize/api_router.py:91`，仅 `--enable-tokenizer-info-endpoint` 时注册（`:87`） |
| `/ping` | GET/POST | `ping` | `vllm/entrypoints/serve/sagemaker/api_router.py:50` |
| `/invocations` | POST | `invocations` | `vllm/entrypoints/serve/sagemaker/api_router.py:66` |

**G 族 · Dev-only（仅 `VLLM_SERVER_DEV_MODE=1`，`## 5` 决策 2/5 重点）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/reset_prefix_cache`、`/reset_mm_cache`、`/reset_encoder_cache` | POST | 同名函数 | `vllm/entrypoints/serve/dev/cache/api_router.py:21,48,59` |
| `/pause` | POST | `pause_generation` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:30` |
| `/resume` | POST | `resume_generation` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:75` |
| `/abort_requests`（dev 版） | POST | `abort_requests` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:95` |
| `/is_paused` | GET | `is_paused` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:140` |
| `/init_weight_transfer_engine` | POST | `init_weight_transfer_engine` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:158` |
| `/start_weight_update` | POST | `start_weight_update` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:176` |
| `/start_draft_weight_update` | POST | `start_draft_weight_update` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:182` |
| `/update_weights` | POST | `update_weights` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:188` |
| `/finish_weight_update` | POST | `finish_weight_update` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:206` |
| `/update_weight_version` | POST | `update_weight_version` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:215` |
| `/weight_info` | GET | `weight_info` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:224` |
| `/get_world_size` | GET | `get_world_size` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:230` |
| `/collective_rpc` | POST | `collective_rpc` | `vllm/entrypoints/serve/dev/rpc/api_router.py:24` |
| `/server_info` | GET | `show_server_info` | `vllm/entrypoints/serve/dev/server_info/api_router.py:44` |
| `/sleep`、`/wake_up`、`/is_sleeping` | POST/GET | `sleep`/`wake_up`/`is_sleeping` | `vllm/entrypoints/serve/dev/sleep/api_router.py:22,33,46` |

**H 族 · 弹性伸缩与故障恢复（独立 CLI/任务门控，不属于 dev 族）**

| 路径 | 方法 | 处理函数 | 文件:行 | 注册条件 |
|---|---|---|---|---|
| `/scale_elastic_ep` | POST | `scale_elastic_ep` | `vllm/entrypoints/serve/elastic_ep/api_router.py:39` | `"generate"` 任务，无需 dev 模式 |
| `/is_scaling_elastic_ep` | POST | `is_scaling_elastic_ep` | `vllm/entrypoints/serve/elastic_ep/api_router.py:86` | 同上 |
| `/fault_tolerance/apply` | POST | `process_fault_tolerance_instruction` | `vllm/entrypoints/serve/fault_tolerance/api_router.py:45` | `args.enable_fault_tolerance` 为真 |
| `/fault_tolerance/status` | GET | `get_status` | `vllm/entrypoints/serve/fault_tolerance/api_router.py:94` | 同上 |

**I 族 · 语音（随 `transcription`/`realtime` 任务）**

| 路径 | 方法 | 处理函数 | 文件:行 |
|---|---|---|---|
| `/v1/audio/transcriptions` | POST | `create_transcriptions` | `vllm/entrypoints/speech_to_text/transcription/api_router.py:42` |
| `/v1/audio/translations` | POST | `create_translations` | `vllm/entrypoints/speech_to_text/translation/api_router.py:42` |
| `/v1/realtime` | WEBSOCKET | `realtime_endpoint` | `vllm/entrypoints/speech_to_text/realtime/api_router.py:18` |

### 2.3 协议类文件（188 个类，19 个 `protocol.py` 文件）

和 SGLang 把几乎所有协议类塞进一个 2116 行的 `entrypoints/openai/protocol.py` 不同，vLLM 按 `<family>/<endpoint>/protocol.py` 拆到 19 个独立文件（`_lab/out/api_surface.json` 的 `engines.vllm.protocol_files`），核心的 `ChatCompletionRequest`（68 个字段，本篇 `## 3` 展开）定义在 `vllm/entrypoints/openai/chat_completion/protocol.py:213`，类体一路到 `:1046` 行（下一个类 `BatchChatCompletionRequest` 从 `:1047` 开始）。这套"每个 endpoint 一个 `{api_router,protocol,serving}.py` 三件套"的目录结构是 `_PLAN.md` §7 已知踩坑 3 提到的重构结果——旧的单文件 `vllm/entrypoints/openai/protocol.py` 在这个 sha 已经不存在了。

### 2.4 中间件栈：`--api-key` 认证只是其中一层，注册顺序在路由之后

`init_entrypoints_middleware()`（`vllm/entrypoints/serve/middleware/register.py:19-75`）在 `build_app()` 里于路由全部注册完之后才被调用（`vllm/entrypoints/launchers/app.py:53`），依次挂上：`CORSMiddleware`（`vllm/entrypoints/serve/middleware/register.py:24-30`，始终挂载，`allow_origins` 默认 `["*"]`）→ 条件挂载的 `AuthenticationMiddleware`（`:32-36`，仅当 `args.api_key` 或 `envs.VLLM_API_KEY` 非空时才挂，这也是为什么完全不配 `--api-key` 时 `GUARDED_PREFIX` 讨论的"是否受保护"根本无意义——中间件压根不存在）→ 可选的 `XRequestIdMiddleware`（`:38-41`）→ 仅 `"generate"` 任务下挂载的 `ScalingMiddleware`（`:43-47`，检查请求是否落在 `/scale_elastic_ep` 触发的弹性伸缩窗口期）→ 仅 `"realtime"` 任务下挂载的 `WebSocketMetricsMiddleware`（`:49-55`）→ 可选的调试用响应体日志中间件（`:57-63`，`envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE` 打开时会打印一条"CAUTION"警告，因为响应体可能含敏感信息）→ 最后是 `args.middleware` 里用户自定义的任意中间件（`:65-74`，支持类和协程函数两种形态）。**Starlette 的 `add_middleware()` 是往 `user_middleware` 列表头部插入（`insert(0, ...)`），构建调用栈时又是 `reversed(middleware)` 逐层往外包**——两次反转的结果是"先注册的中间件反而离路由更近、在请求路径上更晚处理请求"。这意味着虽然 `CORSMiddleware` 最先注册（`:24-30`），实际请求路径上**它比之后才注册的 `AuthenticationMiddleware` 更晚处理请求**——一个跨域预检请求（`OPTIONS`）会先进 `AuthenticationMiddleware.__call__()`，这也是为什么这个中间件要专门显式放行 `OPTIONS` 方法（`vllm/entrypoints/serve/middleware/authenticate.py:48-51`）：如果不做这个特判，预检请求会在真正走到 `CORSMiddleware` 补上跨域响应头之前，先被认证逻辑拒绝掉。

### 2.5 闸门速查表：63 条路由背后的全部条件汇总

本篇正文散落提到的所有"要不要注册/要不要认证"的判断条件，汇总在一张表里方便回查：

| 闸门 | 类型 | 默认值 | 影响哪些路由 | 判断位置 |
|---|---|---|---|---|
| `supported_tasks` 里是否含 `"generate"` | 启动时任务集合 | 取决于加载的模型 | A/B/C(`/cohere/v2/chat`)/E(`/inference/v1/generate`) 族的大多数 | `vllm/entrypoints/launchers/api_server/routers.py:39,73` |
| `supported_tasks` 里是否含 `"render"` | 启动时任务集合 | 取决于启动方式 | E 族的 render/derender 4 条（`"generate"` 或 `"render"` 二选一即可） | `vllm/entrypoints/launchers/api_server/routers.py:52` |
| `supported_tasks` 里是否含 pooling 任务 | 启动时任务集合 | 取决于加载的模型 | D 族全部 | `vllm/entrypoints/launchers/api_server/routers.py:64` |
| `VLLM_SERVER_DEV_MODE` | 环境变量 | `False` | G 族全部（cache/rlhf/rpc/server_info/sleep，13 条路由） | `vllm/envs.py:168`，判断点 `vllm/entrypoints/launchers/api_server/routers.py:34` |
| `VLLM_ENABLE_COHERE_API` + `cohere` SDK 是否已装 | 环境变量 + 依赖探测 | `False` / 取决于环境 | `/cohere/v2/chat` 单条 | `vllm/envs.py:264`，判断点 `vllm/entrypoints/cohere/api_router.py:229-241` |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | 环境变量 | `False` | `/v1/load_lora_adapter`、`/v1/unload_lora_adapter` | 判断点 `vllm/entrypoints/serve/lora/api_router.py:27` |
| `args.enable_fault_tolerance` | CLI 旗标 | `False` | H 族 `/fault_tolerance/*` 2 条 | 判断点 `vllm/entrypoints/launchers/api_server/routers.py:69` |
| `args.tokens_only` | CLI 旗标 | `False` | scale-out 版 `/abort_requests` 1 条 | 判断点 `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:77` |
| `args.enable_tokenizer_info_endpoint` | CLI 旗标 | `False` | `/tokenizer_info` 单条 | 判断点 `vllm/entrypoints/serve/tokenize/api_router.py:87` |
| `model_config.hf_config.num_labels == 1`（仅 `classify` 任务） | 模型元信息 | 取决于模型 | `/score`、`/v1/score`、`/rerank`、`/v1/rerank`、`/v2/rerank` | `vllm/entrypoints/pooling/utils.py:243-247` |
| `args.api_key` 或 `VLLM_API_KEY` 非空 | CLI/环境变量 | 未配置 | 决定 `AuthenticationMiddleware` 存不存在，而不是某条具体路由存不存在 | `vllm/entrypoints/serve/middleware/register.py:33` |
| 请求路径是否落在 `GUARDED_PREFIX` | 中间件内部逻辑 | 只含 `/v1`、`/v2`、`/inference`、`/cohere` | 决定已挂载的 `AuthenticationMiddleware` 要不要检查这条请求 | `vllm/entrypoints/serve/middleware/authenticate.py:11,59` |

这张表本身就是 `## 0` 第三条结论"四种互不相同的闸门"的完整证据清单——11 行里覆盖了 4 种不同的判断依据（任务集合、环境变量、CLI 旗标、模型元信息），没有一种是"只要看 `_lab/out/api_surface.json` 就能确定的静态属性"。

---

## 3. 核心数据结构

- **`AuthenticationMiddleware`**（`vllm/entrypoints/serve/middleware/authenticate.py:14`）——纯 ASGI 中间件，`GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")`（`:11`）是一个**白名单前缀元组**，只有请求路径以这四者之一开头才会走 `verify_token()`（`:30-45`，Bearer token 用 SHA-256 摘要做 `secrets.compare_digest` 常量时间比较）。`__call__()`（`:47-62`）里唯一的判断逻辑就是 `if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers)`——不匹配前缀直接放行，不做任何认证。
- **`ChatCompletionRequest`**（`vllm/entrypoints/openai/chat_completion/protocol.py:213`）——68 个字段（`_lab/out/compare.json` 的 `chat_request.vllm.n_fields`），22 个标准 OpenAI 字段 + 46 个 vLLM 私有扩展（`n_fields`/`openai_standard`/`engine_specific` 三个列表长度分别是 68/22/46），本篇 `## 5` 决策 3 展开代表性字段。按用途分组，46 个私有字段里比较能代表"vLLM 特有能力"的一批：

| 字段 | 行号 | 解决什么问题 |
|---|---|---|
| `priority` | `:380` | 请求优先级调度，配合调度器的优先级队列（本篇不展开调度细节） |
| `session_id` | `:398` | 多轮对话之间复用 KV cache / 会话状态 |
| `routed_experts_prompt_start`、`return_token_offsets` | `:425,430` | MoE 专家路由决策的可观测性——把每个 token 实际路由到哪个专家暴露出来 |
| `cache_salt` | `:467` | 给前缀缓存加盐，让本来会命中前缀缓存的两个相同请求强制不共享缓存（多租户隔离场景） |
| `kv_transfer_params`、`ec_transfer_params` | `:480,485` | PD 分离场景下，调用方指定 KV/encoder-cache 传输的目标参数，详见 `[[12-vLLM-PD分离与KV-Connector]]` |
| `vllm_xargs` | `:492` | 任意扩展参数透传的兜底通道，字段本身不做语义解释，交给下游插件/自定义采样逻辑处理 |
| `structured_outputs` | `:376` | 结构化输出（JSON schema/正则/语法约束），详见 `[[11-vLLM-结构化输出]]` |
| `repetition_detection` | `:500` | 检测并处理生成过程中的重复模式，是采样参数之外的独立后处理开关 |
| `stream_interval` | `:510` | 控制流式响应每隔多少个 token 才 flush 一次，用于在延迟和吞吐之间取舍 |
| `thinking_token_budget` | `:258` | 给推理/思考过程设置 token 预算上限，和 `include_reasoning` 配合控制"思考"内容的输出行为 |
- **`WeightTransferInitRequest` / `WeightTransferUpdateRequest`**（`vllm/distributed/weight_transfer/base.py`，被 `vllm/entrypoints/serve/dev/rlhf/api_router.py:11-14` 引入）——RLHF 权重传输族的请求体，字段只有一个不透明的 `init_info`/`update_info` 字典（分别见 `vllm/entrypoints/serve/dev/rlhf/api_router.py:163-171` 和 `vllm/entrypoints/serve/dev/rlhf/api_router.py:193-201`），具体结构由权重传输引擎的实现决定，HTTP 层不做强类型校验。
- **`FaultToleranceRequest`**（`vllm/v1/fault_tolerance/utils.py`，被 `vllm/entrypoints/serve/fault_tolerance/api_router.py:14` 引入）——字段 `instruction`/`params`/`request_id`，且 `instruction` 目前被硬编码白名单成只允许 `"retry"` 一个值（`_ALLOWED_INSTRUCTIONS = {"retry"}`，`vllm/entrypoints/serve/fault_tolerance/api_router.py:20`），其余值直接 400。
- **`GenerateRequest`**（`vllm/entrypoints/scale_out/token_in_token_out/protocol.py`）——token-in-token-out 的原生请求体，是 render 端点的输出类型、也是 `/inference/v1/generate` 的输入类型，`vllm/entrypoints/scale_out/render/api_router.py:29` 的 `response_model=GenerateRequest` 直接把这个类型关系写进了 FastAPI 的 OpenAPI schema 里。
- **`AnthropicMessagesRequest` / `AnthropicMessagesResponse`**（`vllm/entrypoints/anthropic/protocol.py`，被 `vllm/entrypoints/anthropic/api_router.py:10-17` 引入）——Anthropic Messages API 的请求/响应类型，`AnthropicServingMessages._convert_anthropic_to_openai_request()`（`vllm/entrypoints/anthropic/serving.py:194`）负责把它转成 `ChatCompletionRequest`，`messages_full_converter()`（`:620`）负责把 `ChatCompletionResponse` 转回来。
- **`OnlineRenderer`**（`vllm/renderers/online_renderer.py:63`）——`## 4`/`## 5` 决策 4 反复提到的"渲染逻辑的唯一实现"，构造参数里的 `chat_template`/`chat_template_content_format`/`trust_request_chat_template`/`enable_auto_tools`/`tool_parser`/`reasoning_parser`（`:66-78`）正是 `ChatCompletionRequest` 里那些和模板渲染、工具调用格式化相关字段的服务端配置对应物——每次启动只构造一个实例（`vllm/entrypoints/launchers/api_server/app_state.py:82` 或 `vllm/entrypoints/launchers/render/app_state.py:50`，两处二选一，取决于是完整 server 还是 render-only server），被 `OpenAIServingChat`、`ServingRender`、`AnthropicServingMessages`、`CohereServingChatV2` 共享。
- **`RequestResponseMetadata`**（`vllm/entrypoints/openai/engine/protocol.py:164`）——每条请求在 HTTP 层内部流转时挂在 `raw_request.state.request_metadata` 上的元数据对象（`vllm/entrypoints/openai/chat_completion/serving.py:281-282` 是典型赋值点），Cohere 层的 `_request_id()` 辅助函数（`vllm/entrypoints/cohere/api_router.py:70-84`）会优先从这里读 `request_id`、读不到再退回 `X-Request-Id` 请求头——这是一个跨协议族共享的"请求身份"载体，不属于任何一个具体协议的 pydantic 模型。
- **三套错误信封（`ErrorResponse` / `AnthropicErrorResponse` / `CohereError`），三种不同的"谁来翻译"策略**：核心 `ErrorResponse`（`vllm/entrypoints/openai/engine/protocol.py`，被几乎所有 `api_router.py` 引入）是 `{"error": {"message": ..., "code": ...}}` 形状，全局异常处理器（`init_exception_handler()`，`vllm/entrypoints/launchers/app.py:52`）原生产出的就是这个形状，OpenAI 族路由不需要任何转换。Anthropic 族反过来——`translate_error_response()`（`vllm/entrypoints/anthropic/api_router.py:39-48`）在**每一个** handler 函数内部手动调用（`:70,76,79,108,114,117`），把 `ErrorResponse` 逐字段映射成 `AnthropicErrorResponse`。Cohere 族又是第三种做法——`CohereErrorEnvelopeMiddleware(BaseHTTPMiddleware)`（`vllm/entrypoints/cohere/api_router.py:146-171`）不在 handler 里做转换，而是**全局拦截 `/cohere/*` 路径下所有状态码 ≥400 的 JSON 响应**（`:162-171`）统一重写成 `CohereError` 形状，文档字符串直接说明动机："globally-registered exception handlers... produce vLLM's internal `ErrorResponse` shape. That shape doesn't match the `CohereError` schema... so clients... would see a mismatch"（`:150-157`）——因为 pydantic 请求体校验失败这类错误根本不会经过 `chat_v2()` handler 内部的 try/except，只有中间件能兜住这条路径。**三种协议、三种错误翻译的实现位置**（原生无需转换 / 逐 handler 手动转换 / 全局中间件兜底转换），恰好对应三种协议各自接入 vLLM 内部错误体系的深度不同。

---

## 4. 主流程走读

以 `/v1/chat/completions` 一次非流式请求为主线，`/v1/messages`（Anthropic）作为对照跳：

**跳 1（路由分发）**：`create_chat_completion(request: ChatCompletionRequest, raw_request: Request)`（`vllm/entrypoints/openai/chat_completion/api_router.py:53-74`）先从 `raw_request.app.state.openai_serving_chat` 取出 handler（`:57`，`chat()` 辅助函数定义在 `:32-33`），再转发给 `handler.create_chat_completion(request, raw_request)`（`:61`）。路由函数本身不做业务逻辑，只负责"取 handler + 转发 + 按返回类型包成 `JSONResponse`/`StreamingResponse`"（`:63-74`）。

**跳 2（外层包装 + kv 传输清理）**：`OpenAIServingChat.create_chat_completion()`（`vllm/entrypoints/openai/chat_completion/serving.py:244-257`）本体只有一句话——`return await self._with_kv_transfer_rejection_cleanup(self._create_chat_completion(request, raw_request), request, raw_request)`（`:256-258`），真正的逻辑全在 `_create_chat_completion()`（`:260`）里，外层这一包是为了保证 PD 分离场景下即使请求半途被拒绝，KV 传输相关的资源也会被正确释放。

**跳 3（渲染：tokenize + chat template，和 render 端点走同一份代码）**：`_create_chat_completion()`（`:260-275`）第一步就是 `result = await self.render_chat_request(request)`（`:275`），而 `render_chat_request()`（`:217-241`）内部只做两件事——`_check_model()` 校验模型/LoRA（`:231`）、引擎存活检查（`:236-238`），然后 `return await self.online_renderer.render_chat(request)`（`:240`）。**这一行和独立的 `/v1/chat/completions/render` 端点最终调用的是同一个 `OnlineRenderer.render_chat()`**（对照见 `vllm/entrypoints/scale_out/render/serving.py:80`）——区别只是 `ServingRender.render_chat_request()`（`vllm/entrypoints/scale_out/render/serving.py:64-93`）额外做了"必须恰好 1 个 engine prompt"这类 render-only 场景才需要的校验（`:86-90`），`OpenAIServingChat` 这边不需要，因为它接下来直接把结果送进调度。

**跳 4（组装引擎输入、发起生成）**：`render_chat_request()` 返回 `(conversation, engine_inputs)`（`vllm/entrypoints/openai/chat_completion/serving.py:277`）之后，`_create_chat_completion()` 继续为每个 `engine_input` 算 `max_tokens`（`:308-317`）、解析 LoRA 适配器（`:290`）、生成 `request_id`（`:281-282`），最终按 `request.stream` 分流（`:394`）：`True` 走 `chat_completion_stream_generator()`（`:395-403`，定义在 `:452`），`False` 走 `chat_completion_full_generator()`（`:407-415`）——本篇不展开引擎内部怎么产出 `RequestOutput` 序列（属于 `[[02-vLLM-V1架构与EngineCore循环]]` 范围），只看 HTTP 层怎么把这个异步序列变成响应字节。

**跳 4.1（流式：SSE 逐块拼装）**：`chat_completion_stream_generator()`（`vllm/entrypoints/openai/chat_completion/serving.py:452-508` 起）是一个 `AsyncGenerator[str, None]`，每收到一个增量结果就 `yield f"data: {data}\n\n"`（`:507`，`data` 是 `ChatCompletionStreamResponse` 序列化后的 JSON 字符串），流结束再 `yield "data: [DONE]\n\n"`（`:508`）——这是标准 OpenAI SSE 收尾哨兵。路由函数拿到这个生成器后直接 `return StreamingResponse(content=generator, media_type="text/event-stream")`（`vllm/entrypoints/openai/chat_completion/api_router.py:74`），FastAPI 逐块把 `yield` 出来的字符串写回连接，不会等生成器耗尽再一次性发送。

**跳 5（Anthropic 层：不重新实现，直接调用继承来的同名方法）**：`create_messages(request: AnthropicMessagesRequest, raw_request)`（`vllm/entrypoints/anthropic/api_router.py:63-86`）转发给 `AnthropicServingMessages.create_messages()`（`vllm/entrypoints/anthropic/serving.py:592-618`）。这个方法只做三步：① `chat_req = self._convert_anthropic_to_openai_request(request, ...)`（`:605`，把 Anthropic 的 `messages`/`system`/`tools` 结构翻译成 `ChatCompletionRequest`）；② `generator = await self.create_chat_completion(chat_req, raw_request)`（`:610`）——注意这里调用的 `self.create_chat_completion` 就是跳 2 里那个方法本身，因为 `class AnthropicServingMessages(OpenAIServingChat)`（`vllm/entrypoints/anthropic/serving.py:99`）是**继承**关系，不是持有一个引用再转发；③ 把 `ChatCompletionResponse` 转回 `AnthropicMessagesResponse`（`messages_full_converter()`，`:620-660`，逐字段映射 `finish_reason`→`stop_reason`、`reasoning`→`thinking` 内容块等）。**Anthropic 层完全没有自己的渲染/生成逻辑**，跳 3、跳 4 对它是透明复用的。

**跳 6（Cohere 层：独立文件，但同样复用生成通道）**：`chat_v2(request: CohereChatV2Request, raw_request)`（`vllm/entrypoints/cohere/api_router.py:115-140` 起）转发给 `raw_request.app.state.cohere_serving_chat_v2`（`:67-68`），这个 handler 对象在 `init_generate_state()` 里构造时同样传入 `state.openai_serving_models`/`online_renderer` 等共享依赖（`vllm/entrypoints/generate/api_router.py:215-234`），但**只有 `VLLM_ENABLE_COHERE_API=1` 且 `cohere` SDK 可导入时这个对象才会被真正构造并挂到 `app.state`**（`vllm/entrypoints/generate/api_router.py:75-81`）；`attach_router()` 侧也做了对称的双重检查（`vllm/entrypoints/cohere/api_router.py:229-241`），两处检查独立存在，任一处失败都会让 `/cohere/v2/chat` 返回 404（路由压根没注册）而不是 501。

**跳 7（render 独立端点：反过来印证跳 3）**：`render_chat_completion(request: ChatCompletionRequest, raw_request)`（`vllm/entrypoints/scale_out/render/api_router.py:37-49`）拿到的是**和 `/v1/chat/completions` 完全同一种类型**的请求体，转发给 `ServingRender.render_chat_request()`（`vllm/entrypoints/scale_out/render/serving.py:64`），返回值是 `GenerateRequest`（token-in-token-out 格式，`response_model=GenerateRequest` 声明见 `vllm/entrypoints/scale_out/render/api_router.py:29`）——调用方拿到的就是"如果这条请求走 `/v1/chat/completions`，引擎实际会收到的 token 序列长什么样"，不发起任何实际生成。`derender_chat_completion()`（`vllm/entrypoints/scale_out/derender/api_router.py:40`）做相反的事：接收一个原始 generate 响应，转成标准 `ChatCompletionResponse` 形状。两者搭配 `/inference/v1/generate`（`vllm/entrypoints/scale_out/token_in_token_out/api_router.py:58`），构成一条完整的"协议翻译在一处、token 生成在另一处"的旁路——`## 5` 决策 4 展开这套架构解决的问题。

### 4.1 运维 / 弹性族：各自解决什么问题，风险在哪

这四组端点解决的是四个不同的问题，不能当成同一类"管理接口"笼统看待：

| 端点 | 解决什么问题 | 关键实现细节 | 风险 |
|---|---|---|---|
| `/sleep`、`/wake_up`、`/is_sleeping` | 让引擎在不销毁进程的前提下把显存让给别的任务（同机 RLHF 训练、临时切模型） | `sleep(level, mode)`（`vllm/engine/protocol.py:165-167` 抽象方法，`vllm/v1/engine/async_llm.py:969-972` 实现）分两档：**level 1** 把模型权重卸到 CPU 内存、丢弃 KV cache，唤醒后权重从 CPU 恢复；**level 2** 权重和 KV cache 都直接丢弃（只留 `named_buffers()`，即 rope-scaling 之类的小张量，见 `vllm/v1/worker/gpu_worker.py:208-218` 的 `_sleep_saved_buffers`），唤醒后要靠 `collective_rpc("reload_weights")` 重新加载——这两档语义是 `docs/features/sleep_mode.md:21` 明文写的，不是本库猜的 | 需要 `VLLM_SERVER_DEV_MODE=1` 才注册，但注册之后**不受 `--api-key` 保护**：任何能连上端口的请求都能把一个正在服务的引擎打睡（拒绝服务） |
| `/scale_elastic_ep`、`/is_scaling_elastic_ep` | 运行时改变数据并行（DP）规模，给 MoE 场景做弹性专家并行伸缩，不用重启进程 | `scale_elastic_ep()`（`vllm/entrypoints/serve/elastic_ep/api_router.py:39-82`）接受 `new_data_parallel_size` 和 `drain_timeout`（默认 120 秒，`:46`），先把在途请求排空再切换规模，超时会返回 408 而不是硬切（`:72-77`） | **不需要 dev 模式**，只要 `"generate"` 任务支持就注册——且同样不受 `--api-key` 保护；`docs/usage/security.md:246` 把它列进"可发起拒绝服务"的一类操作 |
| `/fault_tolerance/apply`、`/fault_tolerance/status` | 让外部编排器（K8s controller 之类）在检测到某个 rank 挂掉后触发跨 rank 的恢复流程，而不是整集群重启 | `instruction` 目前被硬编码只允许 `"retry"` 一个值（`vllm/entrypoints/serve/fault_tolerance/api_router.py:20`）；由于恢复要跑"跨 rank 集合操作，只有每个 rank 都收到指令才会完成"（`:61-64` 注释原话），HTTP 层用 `202 Accepted` + 后台任务（`BackgroundTasks`，`:45-73`）立即返回，调用方要轮询 `/fault_tolerance/status` 才能确认完成 | 仅 `args.enable_fault_tolerance` 为真时注册（不是 dev 模式旗标），但同样不在 `GUARDED_PREFIX` 内 |
| `/collective_rpc` | 给"没有专用端点覆盖的操作"留一条通用后门——直接在 worker 进程上调用任意已注册方法（比如上面提到的 `reload_weights`） | `collective_rpc(method, args, kwargs, timeout)`（`vllm/entrypoints/serve/dev/rpc/api_router.py:24-54`），注释明写"For security reason, only serialized string args/kwargs are passed. User-defined \`method\` is responsible for deserialization if needed."（`:38-39`）——**这句注释本身说明了安全边界画在哪**：HTTP 层只保证参数是字符串，不保证 `method` 本身是不是一个安全的操作，能调用的方法集合完全取决于 worker 类当前实现了哪些方法，HTTP 层没有方法白名单 | 需要 `VLLM_SERVER_DEV_MODE=1`，且不受 `--api-key` 保护，`docs/usage/security.md:231` 标注"extremely dangerous"——是本篇四组里唯一同时叠了"dev-only 开关"和"任意方法调用"两层风险的端点 |

四组端点的共同点是**没有一个受 `--api-key` 保护**（都不落在 `/v1`、`/v2`、`/inference`、`/cohere` 这四个前缀下），区别只在于要不要 `VLLM_SERVER_DEV_MODE=1` 这道额外闸门——`/sleep`、`/collective_rpc` 需要，`/scale_elastic_ep`、`/fault_tolerance/*` 不需要。这个"部分运维端点默认打开、部分需要显式开关"的不一致，本身就是 `## 7` 要挑出来的一个踩坑点。

---

## 5. 设计决策与代价

### 决策 1：`--api-key` 只用一个路径前缀白名单来决定要不要认证，不是给每条路由单独打标签

- **为什么这么设计**：`AuthenticationMiddleware`（`vllm/entrypoints/serve/middleware/authenticate.py:14-62`）是一个纯 ASGI 中间件，只在 `app.add_middleware()` 时接收一次 `tokens` 列表（`vllm/entrypoints/serve/middleware/register.py:33-36`），`__call__()` 里只做一次 `url_path.startswith(GUARDED_PREFIX)` 判断（`:59`）——这样加一条新路由完全不需要碰认证逻辑，只要新路由的路径以 `/v1`/`/v2`/`/inference`/`/cohere` 开头，它就自动被保护；反过来，任何插件路由（`attach_endpoint_plugins()`，`vllm/entrypoints/launchers/app.py:50`）不需要知道 vLLM 内部的认证机制存不存在，只要按惯例把路径挂在这四个前缀下就能"免费"获得保护。这是一种用命名约定代替显式声明的设计，接线成本几乎为零。
- **不这样会怎样**：如果改成给每条路由单独声明"要不要认证"（比如 SGLang 用 `AuthLevel` 三档分级），新增一条路由时必须记得显式标注，漏标就是裸奔——`_lab/out/api_surface.json` 里 63 条路由分布在 30 个文件、9 个不同的注册函数里，如果认证是逐路由声明的，遗漏的概率会随文件数量线性上升。而按前缀白名单，遗漏的代价反过来变成"新路由如果没有以这四个前缀开头，会被开发者自己发现"（因为不加前缀通常意味着这条路由本来就不打算走标准客户端），但也正是这个"通常"，造成了 `/invocations`、`/generative_scoring`、`/pooling`、`/classify`、`/score`、`/rerank` 这些**同样暴露推理能力**、只是路径没挂在 `/v1` 下的端点被误伤性地排除在保护之外——`docs/usage/security.md:246` 原话把这一点列成头号安全隐患："Bypass authentication by using non-`/v1` endpoints like `/invocations`... to run arbitrary inference without credentials"。
- **什么时候可以不这样**：如果部署场景本来就打算把整个 HTTP 服务放在反向代理/API 网关后面，由网关统一做认证（`docs/usage/security.md` 通篇的建议方向就是这个），`--api-key` 前缀白名单的粒度问题就不重要——网关层可以对所有路径统一鉴权，不区分前缀。vLLM 自己的文档没有把"网关层认证"当成默认假设，而是当成"你必须自己加的一层"，这也是为什么文档要专门列一节把不受保护的端点点清楚。

### 决策 2：entrypoints 拆成 `<family>/<endpoint>/{api_router,protocol,serving}.py` 三件套，取代旧的单文件 `openai/protocol.py`

- **为什么这么设计**：每个 endpoint 独立三个文件后，`api_router.py` 只关心 HTTP 层（路径、依赖注入、序列化成什么响应类型）、`protocol.py` 只关心 pydantic 请求/响应模型、`serving.py` 只关心业务逻辑——三者的变更半径互不重叠。以 `chat_completion/` 目录为例（`vllm/entrypoints/openai/chat_completion/{api_router.py,protocol.py,serving.py}`），改 `ChatCompletionRequest` 加一个字段只需要碰 `protocol.py`，不会牵动 `api_router.py` 里的路由声明。188 个协议类分散到 19 个文件（`_lab/out/api_surface.json` 的 `protocol_files`），每个文件的改动都是局部的。
- **不这样会怎样**：SGLang 走的是相反的路线——几乎所有 OpenAI 协议类塞进一个 2116 行的 `entrypoints/openai/protocol.py`（`sglang:python/sglang/srt/entrypoints/openai/protocol.py`），好处是"要看所有协议长什么样，打开一个文件就够了"；代价是这个文件成了高频合并冲突的焦点——任何 endpoint 的字段变更都要改同一个文件，多人并行开发时冲突概率更高。vLLM 反过来把这个"单文件视图"的便利性换成了"改动隔离"，`_lab/api_surface.py` 的抽取器也因此要跟着改成按 `entrypoints/<family>/<endpoint>/protocol.py` 的 glob 模式去找协议类（`_PLAN.md` §7 已知踩坑 3 记录过这次重构曾经让抽取器一度得到"0 条路由"的假空结果）。
- **什么时候可以不这样**：如果引擎的协议表面本来就小（比如只支持 `/v1/chat/completions` 一个端点），拆三件套的开销（19 个文件、30 个目录）远大于收益，单文件反而更容易读——这也是为什么 vLLM 自己在更早版本、协议表面更小的时候也是单文件；拆包是协议表面膨胀到一定规模之后的重构结果，不是从第一天就该有的设计。

### 决策 3：`ChatCompletionRequest` 用一个大类塞下全部 46 个私有扩展字段，不拆"标准子集 + 扩展 mixin"

- **为什么这么设计**：`ChatCompletionRequest`（`vllm/entrypoints/openai/chat_completion/protocol.py:213-1046`）68 个字段里 46 个是 vLLM 私有扩展（`_lab/out/compare.json` 的 `chat_request.vllm.engine_specific`），包括结构化输出用的 `structured_outputs`（`:376`）、优先级调度用的 `priority`（`:380`）、会话保持用的 `session_id`（`:398`）、专家路由可观测性用的 `routed_experts_prompt_start`/`return_token_offsets`（`:425,430`）、缓存复用用的 `cache_salt`（`:467`）、PD 分离用的 `kv_transfer_params`/`ec_transfer_params`（`:480,485`）、任意扩展参数透传用的 `vllm_xargs`（`:492`）。全塞进一个类的好处是标准 OpenAI SDK 发请求时这些字段全部留空也能正常序列化（pydantic 默认值兜底），客户端不需要知道"这是标准字段还是扩展字段"这个区分。
- **不这样会怎样**：如果严格拆分"标准协议类"和"vLLM 扩展 mixin"两层，`OpenAIServingChat._create_chat_completion()` 里大量直接访问 `request.priority`/`request.kv_transfer_params` 这类扩展字段的代码就要多一层"先判断扩展 mixin 是否存在"的防御逻辑，或者要求调用方总是传入组合类型——现状是简单地把 46 个字段当一等公民对待，运行时代码不需要关心某个字段"是不是标准的"。
- **什么时候可以不这样**：`_lab/out/compare.json` 显示 SGLang 的 `ChatCompletionRequest`（`sglang:python/sglang/srt/entrypoints/openai/protocol.py:823`）同样是 68 个字段的单一大类策略，且两家相对 OpenAI 官方标准缺失的 8 个字段（`audio`/`function_call`/`functions`/`metadata`/`modalities`/`prediction`/`service_tier`/`store`）**完全一致**——这说明"一个大类塞所有扩展字段"和"这 8 个字段两家都做不到"都不是某一家的权宜之计，而是这一类自托管开源引擎的共性选择：这 8 个字段依赖 OpenAI 后端专有的账户/存储/分层服务基础设施（`store`/`service_tier`/`metadata` 明显是云端概念），开源引擎没有对应底层能力去实现，不是遗漏是不可实现（**本库推断**，依据是这 8 个字段的语义本身）。如果 vLLM 未来要把这份协议提交给上游标准化，私有扩展和标准字段混在一起会让协议审查变得困难——但目前没有这个诉求。

### 决策 4：把"渲染"（tokenize + chat template）从生成路径里剥出来，做成独立可复用的 `OnlineRenderer` + 独立进程

- **为什么这么设计**：`OpenAIServingChat.render_chat_request()`（`vllm/entrypoints/openai/chat_completion/serving.py:217-241`）和 `ServingRender.render_chat_request()`（`vllm/entrypoints/scale_out/render/serving.py:64-93`）都只是薄壳，真正的渲染逻辑集中在 `OnlineRenderer.render_chat()`（`vllm/renderers/online_renderer.py:117`）一处。这样"渲染"这个 CPU 密集、不需要 GPU 的步骤可以独立扩缩容——`build_and_serve_renderer()`（`vllm/entrypoints/launchers/render/entry.py:20-38`）允许启动一个完全不加载模型权重的进程，只做协议翻译；`/inference/v1/generate`（`vllm/entrypoints/scale_out/token_in_token_out/api_router.py:58`）则是一个只接受已经 token 化好的输入、不做任何模板渲染的纯生成端点。两者搭配就是文档里"Disaggregated Everything"（`vllm/entrypoints/scale_out/token_in_token_out/api_router.py:83` 的 docstring 原话）说的场景：协议解析/模板渲染跑在一批廉价的 CPU 实例上，token 生成跑在昂贵的 GPU 实例上，两者可以独立按各自的负载特征扩缩容。
- **不这样会怎样**：如果渲染逻辑耦合在 `OpenAIServingChat` 内部、不对外暴露成独立端点，想要拆分部署的团队只能自己重新实现一遍 tokenize + chat template 逻辑（模板语法、多模态占位符处理、工具调用格式化都要对齐），并且要持续跟上游同步，行为漂移的风险很高；vLLM 直接把内部用的那份实现原样暴露出来，杜绝了"外部重新实现出行为不一致的翻译层"这个问题。
- **什么时候可以不这样**：如果部署规模小、GPU 和 CPU 资源本来就绑定在同一台机器上（没有独立扩缩容的诉求），维护 render/derender 这两族额外端点（4 条路由 + 2 个额外的 serving 类）纯粹是增加代码面积——单机部署场景下这套拆分带来的收益基本兑现不了，`register_scale_out_api_routers()` 却仍然会在 `"generate"` 或 `"render"` 任务下无条件注册这 4 条路由（`vllm/entrypoints/scale_out/factories.py:65-71`），是本篇 `## 8` 会提到的一个可优化点。

### 决策 5：`VLLM_SERVER_DEV_MODE` 用一个全局二元开关控制 5 个文件的全部路由，不做 SGLang 式的按端点分级权限

- **为什么这么设计**：`register_vllm_dev_api_routers()`（`vllm/entrypoints/serve/__init__.py:43-69`）把 cache/rlhf/rpc/server_info/sleep 五个子系统的路由一次性全部挂载或全部不挂载，中间没有"只开一部分"的选项。这是最简单的实现——一个 `if envs.VLLM_SERVER_DEV_MODE:`（`vllm/entrypoints/launchers/api_server/routers.py:34`）就控制了 `/reset_prefix_cache`、`/pause`、`/collective_rpc`、`/server_info`、`/sleep` 等十几条路由的生死，运维只需要记住一个环境变量，不需要理解每条路由各自的风险级别。打开时还会打印一条统一的 `SECURITY WARNING`（`vllm/entrypoints/serve/__init__.py:44-47`），提醒操作者"这不是给生产环境用的"。
- **不这样会怎样**：SGLang 把功能上对等的 RLHF 权重更新/显存让出族**始终挂载**，只用 `AuthLevel.ADMIN_OPTIONAL` 三档权限区分（`sglang:python/sglang/srt/utils/auth.py:82`），因为 SGLang 把这批端点当成"生产场景的一等公民"（同机 RLHF 训练是它的核心使用场景之一）。vLLM 反过来把语义上对等的端点（`/init_weight_transfer_engine`、`/update_weights`、`/pause`/`/resume`）标成"开发调试专用"，默认不给任何用户使用——如果一个 RLHF 训练框架想用 vLLM 做 rollout worker，必须显式设置 `VLLM_SERVER_DEV_MODE=1` 并接受"这本来是为调试设计的"这个定位落差，还要接受这批端点连 `--api-key` 都保护不了的现实。
- **什么时候可以不这样**：如果 vLLM 未来想把 RLHF 场景当成官方支持的一等场景（而不是"碰巧能用的调试后门"），值得学 SGLang 的分级权限模型，给这批端点单独配一套认证（哪怕只是要求配置一个独立的 `--rlhf-api-key`，而不是复用打了折扣的全局开关）——现状是"要么完全不能用，要么打开之后完全不设防"，中间没有过渡地带。

### 决策 6：Cohere 兼容用「环境变量 + SDK 探测」双重 opt-in，而不是像 Anthropic 一样默认随 `"generate"` 任务开

- **为什么这么设计**：`vllm/entrypoints/cohere/api_router.py:1-25` 的文件头文档字符串说得很直接——Cohere v2 协议模型来自官方 `cohere` SDK（一个可选依赖），为了不强制所有 vLLM 用户安装这个包，`attach_router()` 先做一次性的 SDK 导入探测（`:49-54`），再检查 `VLLM_ENABLE_COHERE_API`（`:229-241`）。两级把关是因为它们要防的是两种不同的误配置：SDK 探测防的是"包没装但被要求启用"（这种情况打 WARNING，因为用户明确表达了意图却没达成），环境变量防的是"包因为别的原因被装了但不代表想用这个端点"（比如测试依赖顺带装了 `cohere`，这种情况默认不开、打 DEBUG 级日志）。
- **不这样会怎样**：如果像 Anthropic 一样默认打开（不需要环境变量），`cohere` SDK 就必须变成一个硬依赖或者要在运行时优雅降级——但 vLLM 选择把"是否要 Cohere 兼容"完全交给部署方决定，这样做的代价是 Cohere Chat v2 的可发现性变差（不看文档不会知道要设这个环境变量），`_lab/out/api_surface.json` 里这条路由本身是能被 AST 抽到的（因为装饰器是静态字面量），但**运行时是否真的存在取决于环境变量和 SDK 安装情况，这两点 AST 抽取器都看不出来**——这是"路由存在于源码"和"路由存在于这次启动的 app 上"之间的落差,`## 7` 会展开。
- **什么时候可以不这样**：注意到 `/v2/embed`、`/v2/rerank` 这两个同样是 Cohere 兼容协议的端点走的是完全不同的策略——它们是 pooling 族的一部分，没有任何开关,只要模型支持对应任务就注册（`vllm/entrypoints/pooling/embed/api_router.py:46-61`、`vllm/entrypoints/pooling/scoring/api_router.py:105-114`）。这说明"要不要给 Cohere 兼容加开关"不是一个统一的产品决策，而是按端点各自的"依赖是否可选"来定的——`/v2/embed`/`/v2/rerank` 复用的是 vLLM 自己已有的 pooling 基础设施、不依赖 `cohere` SDK，所以没有必要加开关；`/cohere/v2/chat` 依赖外部 SDK 解析请求体，所以需要。**同一个"Cohere 兼容"标签下，三个端点的可用性策略并不统一**，这本身是本篇一个值得记录的不一致点。

### 决策 7：`/invocations` 用运行时类型嗅探在多种请求形状之间分发，而不是要求调用方显式声明请求类型

- **为什么这么设计**：SageMaker 的模型服务契约规定所有推理请求都打到同一个 `/invocations` 路径，请求体形状本身要能自解释是"这是一个 chat 请求"还是"这是一个 embedding 请求"。`attach_router()`（`vllm/entrypoints/serve/sagemaker/api_router.py:30-98`）在启动时把 `get_generate_invocation_types()` 和 `get_pooling_invocation_types()` 返回的所有候选请求类型各自包一层 `pydantic.TypeAdapter`（`INVOCATION_VALIDATORS`，`:42-45`），运行时收到请求后依次尝试每个 `validator.validate_python(body)`（`:84`），**第一个不抛 `ValidationError` 的类型就是命中的类型**（`:82-88`），再转发给这个类型对应的 `endpoint` 函数。这样 SageMaker 的模型托管标准和 vLLM 内部按类型分派的路由完全解耦——`/v1/chat/completions`、`/v1/embeddings` 各自的类型系统不用为了兼容 SageMaker 改一行代码。
- **不这样会怎样**：如果不做运行时类型嗅探，要么强迫 SageMaker 场景的调用方在请求体里显式加一个 `"task": "chat"` 这样的判别字段（但 SageMaker 标准本身不允许 vLLM 强制要求这个字段，请求体格式由上游模型托管框架决定，不受 vLLM 控制），要么退化成"`/invocations` 只支持一种任务类型"（这样多任务模型——比如同时支持 chat 和 embedding 的服务——就没法用同一个 SageMaker endpoint 同时提供两种能力）。
- **什么时候可以不这样**：这套运行时嗅探的代价是**每次请求最坏情况下要跑 `len(INVOCATION_VALIDATORS)` 次 pydantic 校验才能确定类型**（校验失败的开销通常不小，因为 pydantic 要走完所有字段约束才报错），如果一个部署场景明确知道自己只服务一种任务类型（比如专门起一个纯 embedding 服务），完全可以不通过 `/invocations` 这条通用入口，直接用 `/v1/embeddings`，跳过类型嗅探的开销——这也是为什么 `/invocations` 从来不是 vLLM 推荐的默认调用方式，只是"要接 SageMaker 生态就必须实现"的兼容层。

---

## 6. 同位对照（SGLang 在同一位置怎么做）

同一取证会话核对了 SGLang（sha `15a43983`，2026-08-22）：

**规模速览**（数字来自 `_lab/out/api_surface.json`）：

| 维度 | vLLM | SGLang |
|---|---|---|
| 路由总数（AST 抽取） | 63（61 个 unique path） | 83（另有 4 条 Ollama 路由被漏抽，见 `sglang:python/sglang/srt/entrypoints/http_server.py:1973`） |
| `/v1/*` 路由数 | 21 | 17 |
| 协议类数量 | 188（拆到 19 个文件） | 86（单文件 `protocol.py`） |
| Anthropic 兼容 | 有（`/v1/messages`，**继承**：`AnthropicServingMessages(OpenAIServingChat)`，`vllm/entrypoints/anthropic/serving.py:99`） | 有（`/v1/messages`，**组合**：`AnthropicServing.__init__(self, openai_serving_chat)`，`sglang:python/sglang/srt/entrypoints/anthropic/serving.py:191-192`） |
| Cohere/其他厂商兼容 | 有，`/cohere/v2/chat` 默认关闭（opt-in），`/v2/embed`/`/v2/rerank` 默认开 | 无 Cohere；有 Ollama（`/api/*`）+ SageMaker（`/ping`/`/invocations`）兼容 |
| `--api-key`/`AuthLevel` 默认覆盖范围 | 只覆盖 `/v1`、`/v2`、`/inference`、`/cohere` 四个前缀 | 默认覆盖**所有**端点（`AuthLevel.NORMAL` = "legacy behavior: api_key protects all endpoints when configured"），只有显式标 `ADMIN_OPTIONAL`/`ADMIN_FORCE` 或 `/health`、`/metrics` 才例外 |
| RLHF 端点默认可用性 | 默认不挂载，需 `VLLM_SERVER_DEV_MODE=1`，且不受 `--api-key` 保护 | 始终挂载，`AuthLevel.ADMIN_OPTIONAL`（默认无 key 即放行，配了 key 就要） |

- **认证覆盖范围是两家最根本的分歧点，比 RLHF 端点默认开不开更本质**：vLLM 的 `AuthenticationMiddleware`（`vllm/entrypoints/serve/middleware/authenticate.py:11`）是一张**白名单**——只有匹配 `GUARDED_PREFIX` 的路径才会被检查，默认姿态是"不保护"；SGLang 的 `decide_request_auth()`（`sglang:python/sglang/srt/utils/auth.py:76-141`）文档字符串直接写明 `NORMAL`（默认级别）的语义是"legacy behavior (api_key protects all endpoints when configured)"（`sglang:python/sglang/srt/utils/auth.py:80`）——默认姿态是**保护**，只有显式标注 `/health`/`/metrics` 前缀（`:100-101`）或者用 `@auth_level(...)` 装饰器改成 `ADMIN_OPTIONAL`/`ADMIN_FORCE` 的端点才例外。两边都用同一句英文短语描述各自模型里"最宽松"的那一档——vLLM 的 `GUARDED_PREFIX` 之外是"完全不查"，SGLang 的 `ADMIN_OPTIONAL` 在没配 key 时也是"完全不查"——但**这一档在 vLLM 是默认状态覆盖大多数端点，在 SGLang 是需要显式装饰器标注、只覆盖一小撮管理端点**。同样是"给管理端点开个方便之门"，两家实现出来的默认暴露面大小完全不是一个数量级。
- **两家的 Anthropic 兼容包装方式相反**（`## 5` 决策未展开这条，这里补全）：vLLM 用**继承**——`AnthropicServingMessages.create_messages()`（`vllm/entrypoints/anthropic/serving.py:592-618`）里 `self.create_chat_completion(chat_req, raw_request)`（`:610`）调用的就是父类 `OpenAIServingChat` 定义的同一个方法，Python 方法解析顺序（MRO）保证了这一点；SGLang 用**组合**——`AnthropicServing` 持有一个 `openai_serving_chat` 引用（`sglang:python/sglang/srt/entrypoints/anthropic/serving.py:191-192`），要调用父类逻辑得显式 `self.openai_serving_chat._convert_to_internal_request()`（`sglang:python/sglang/srt/entrypoints/anthropic/serving.py:759`）。继承的耦合更紧（`AnthropicServingMessages` 的实例本身就是一个 `OpenAIServingChat`，两者生命周期完全绑定），但代码量更小；组合更松（哪天 `OpenAIServingChat` 的内部实现换掉，只要接口不变 `AnthropicServing` 不用改），但要多写"转发到持有对象的哪个方法"这一层。**两家都在做同一件事这个事实本身说明 Anthropic Messages API 已经变成事实上的"第二标准"**，这是跨引擎共性，不是某一家的设计选择。
- **协议类数量差 2 倍以上（188 对 86）不代表功能差 2 倍**：这是文件组织粒度不同的产物，`## 5` 决策 2 已经展开——vLLM 拆到 19 个文件、每个 endpoint 独立一个 `protocol.py`，粒度天然更细，一个字段的类型注解、一个嵌套的枚举都可能单独算一个类；SGLang 把几乎所有协议塞进一个 2116 行的文件。用类的绝对数量比较两家协议表面的丰富程度是本篇要提醒的一个误读陷阱。
- **两家的错误响应翻译策略也不对称**：vLLM 的 Cohere 层用全局 `BaseHTTPMiddleware` 兜底翻译（`## 3`/`## 0` 已展开，`vllm/entrypoints/cohere/api_router.py:146-171`），Anthropic 层则是逐 handler 手动调用 `translate_error_response()`（`vllm/entrypoints/anthropic/api_router.py:39-48`）；SGLang 的 Anthropic 层同样需要处理协议错误形状差异，但走的是"翻译层里统一处理"的路线（`sglang:python/sglang/srt/entrypoints/anthropic/serving.py:206` 起的 `handle_messages()` 本身就包一层 try/except），不像 vLLM 这样把"是否需要全局中间件兜底"按协议分别决定——这背后的原因和 `## 0` 里 Cohere 中间件文档字符串说的一样：pydantic 请求体校验失败这类错误根本不会经过 handler 内部逻辑，只有全局注册的异常处理器和中间件能拦到，SGLang 目前没有 Cohere 兼容层，没有遇到这个具体问题，无法类比。
- **两家都在"路由是否真的存在"这个问题上让 AST 静态抽取器吃了亏，但吃亏的方式不同**：SGLang 的 Ollama 兼容路由用 `os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat")` 这种非字面量路径注册（`sglang:python/sglang/srt/entrypoints/http_server.py:1973`），AST 解析不出字符串常量就漏抽，属于"注册方式抽取器认不出"；vLLM 这边则是**注册方式抽取器完全认得出**（`@router.post("/cohere/v2/chat")` 是标准字面量装饰器），但**这条路由是否真的挂在这次启动的 `app` 上取决于两个运行时条件**（`VLLM_ENABLE_COHERE_API` + `cohere` SDK 是否安装），AST 层面完全看不出来。两者都指向同一条 `_PLAN.md` 认知锚点——"63 条路由"这个数字描述的是**源码里写了多少条路由声明**，不是"这次启动实际会响应多少条路径"，`## 7` 第 1 条继续展开。

---

## 7. 踩坑与反直觉

1. **"63 条路由"是源码里声明的数量，不是任意一次启动实际会响应的数量**：光是 `VLLM_SERVER_DEV_MODE`（默认关）就影响 13 条路由的生死（`## 2.2` G 族），`VLLM_ENABLE_COHERE_API` + SDK 探测影响 1 条，`args.tokens_only` 影响 1 条（`/abort_requests` scale-out 版），`args.enable_fault_tolerance` 影响 2 条，`--enable-tokenizer-info-endpoint` 影响 1 条，`VLLM_ALLOW_RUNTIME_LORA_UPDATING` 影响 2 条——**一次典型的默认配置启动（不开任何这些旗标）实际暴露的路由数是 63 减去至少 20 条**。这不是 AST 抽取器的 bug（每条路由的装饰器都是合法的字面量，抽取器该抽的都抽到了），是"路由存在于源码"和"路由存在于运行时"这两个概念本身就不该混为一谈——本篇 `## 2.2` 每张表都专门标了"注册条件"这一列，正是为了避免读者把 63 当成"随便起个 server 就有这么多攻击面"。
2. **`/abort_requests` 这个路径被两个完全不同文件的两个不同函数注册，谁生效取决于注册顺序**：dev 族的版本（`vllm/entrypoints/serve/dev/rlhf/api_router.py:95`，需要 `VLLM_SERVER_DEV_MODE=1`）在 `register_api_routers()` 里于第 37 行（`vllm/entrypoints/launchers/api_server/routers.py:34-37`）就注册了；scale-out 族的版本（`vllm/entrypoints/scale_out/token_in_token_out/api_router.py:80`，需要 `args.tokens_only=True`）要等到第 55 行（`:52-55`）才注册。FastAPI/Starlette 的路由匹配是按注册顺序取第一个精确匹配，**如果一次启动同时把 `VLLM_SERVER_DEV_MODE=1` 和 `--tokens-only` 都打开**（两个条件互不冲突，完全可能同时为真），dev 族版本的 `/abort_requests`（无 `request_ids` 时 abort 全部在途请求）会先注册、拿到路由匹配优先权，scale-out 族那个专门为"Disaggregated Everything"场景写的、语义略有差异的 `/abort_requests`（同样逻辑但直接调 `engine_client(raw_request).abort(request_ids)`，`vllm/entrypoints/scale_out/token_in_token_out/api_router.py:88-89`）会变成永远匹配不到的死代码（**本库推断**：基于 Starlette 路由匹配"先注册先匹配"的通用行为，未实测验证，仅从注册顺序推出结论）。
3. **`vllm/entrypoints/openai/api_server.py` 这个文件名本身就是一个陷阱**：任何习惯了"OpenAI 兼容层入口在 `openai/api_server.py`"这个心智模型的人（这也是本篇任务书最初给的假设），第一次打开这个文件会发现它只有 59 行、通篇是 `warnings.warn` 和 re-export（`:5-22,24-30,50-59`），真正的逻辑已经全部搬到 `vllm/entrypoints/launchers/` 下。这不是死代码——`__all__` 里导出的每个名字仍然可以被外部代码 `from vllm.entrypoints.openai.api_server import build_app` 这样导入且能正常工作（因为是原样转发），只是**这个文件已经不是"读源码找主装配点"时该打开的第一个文件**了。
4. **`--api-key` 保护的端点列表和"标准 OpenAI 客户端能访问的端点列表"并不完全重合**：`docs/usage/security.md:153-176` 列出的"受保护端点"里包含 `/v1/load_lora_adapter`（"only available when `--enable-lora` is set and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`"）——这条路由即使配了 `--api-key`，也要先满足另外两个独立的开关才会真的存在（`vllm/entrypoints/serve/lora/api_router.py:26-29`）。"受 `--api-key` 保护"和"这条路由存在"是两个独立的问题，一个查 `GUARDED_PREFIX`，一个查各自的注册条件，两者互不蕴含。
5. **`/server_info`（dev-only）会把服务器完整配置和全部 `VLLM_*` 环境变量原样吐出来**：`show_server_info()`（`vllm/entrypoints/serve/dev/server_info/api_router.py:43-59`）把 `PydanticVllmConfig.dump_python(vllm_config, ...)`（`:53`）、`_get_vllm_env_vars()`（`:25-35`，遍历 `dir(envs)` 里所有 `VLLM_` 开头且不含 `KEY` 的变量）、`get_env_info()`（`:39-40`，系统环境信息）三块一起塞进响应体。`KEY` 过滤只挡住了变量名里带 `KEY` 的（比如不会泄漏 `VLLM_API_KEY` 本身），但完整的并行策略、模型路径、量化配置等信息全部暴露——这条端点需要 `VLLM_SERVER_DEV_MODE=1` 才存在，但如同 `## 4.1` 说的，同样不受 `--api-key` 保护。
6. **`ChatCompletionResponse` 里也有 `kv_transfer_params`/`ec_transfer_params`（`vllm/entrypoints/openai/chat_completion/protocol.py:143,146`），和 `ChatCompletionRequest` 里同名的两个字段（`:480,485`）容易被当成同一处定义**——前者出现在 `ChatCompletionResponse` 类体内（`:127-150`），是引擎回传给调用方的 PD 分离元数据；后者出现在 `ChatCompletionRequest` 类体内（`:213-1046`），是调用方主动传入的 PD 分离参数。两者字段名相同、语义相反（一个是"引擎告诉你"，一个是"你告诉引擎"），只看字段名不看所在类容易搞反方向。
7. **`stream_options.include_usage` 之外，vLLM 也有一个不严格遵循标准 OpenAI 语义的服务端级默认值**——`enable_force_include_usage` 出现在 `init_generate_state()` 构造 `OpenAIServingChat`/`OpenAIServingCompletion`/`AnthropicServingMessages`/`CohereServingChatV2` 时统一传入的参数（`vllm/entrypoints/generate/api_router.py:145,168,209,229`），命名和语义都与 SGLang 的 `stream_response_default_include_usage`（`sglang:python/sglang/srt/entrypoints/openai/utils.py:97-98`）同构——**两家各自独立地在标准协议之外加了一个"服务端强制打开 usage 统计"的开关**，本篇没有深入这个字段具体从哪个 CLI 旗标控制（留给 `[[09-vLLM-Python-API与EngineArgs]]` 核实），但两家收敛到同一种"运营方想统一采集计费数据，不想依赖每个客户端配合"的加法特性，是又一处跨引擎共性观察。
8. **`/ping` 用同一个函数堆叠两个装饰器注册成两条方法记录，容易被误读成"这里一定有两个实现"**：`@router.post("/ping", ...)` 和 `@router.get("/ping", ...)`（`vllm/entrypoints/serve/sagemaker/api_router.py:47-48`）连续两行分别装饰同一个 `async def ping(raw_request)`（`:50`），这是 `_lab/out/api_surface.json` 里 `n_routes=63` 但 `n_unique_paths=61` 之所以只差 2 的一半原因（另一半是 `/abort_requests` 的双文件重复，见第 2 条）。只看 `api_surface.json` 的 JSON 数组本身分不清这两种"重复"——`/ping` 是"同一个 handler、两个方法"，`/abort_requests` 是"两个 handler、同一条路径、注册条件互斥"，语义完全不同，但都会让 `n_routes - n_unique_paths = 2` 这个数字算出来一样，必须回到 `routes` 数组本身逐条核对文件路径和 `handler` 字段才能分辨——这是"一个汇总数字掩盖了两种不同性质的重复"的具体案例，和 SGLang 篇 `## 7` 提到的"扫了但贡献 0 条路由"是同一类"数字骗人"的坑。
9. **`OnlineRenderer` 在完整 server 和 render-only server 两种部署形态下，是两处独立的构造点，不是同一个跨进程共享的单例**：`vllm/entrypoints/launchers/api_server/app_state.py:82` 和 `vllm/entrypoints/launchers/render/app_state.py:50` 各自 `state.online_renderer = OnlineRenderer(...)`，两个进程各自持有自己的实例——`## 5` 决策 4 说"render 和生成路径共用同一份渲染逻辑"指的是**同一个类的两份独立实例、跑在两个不同进程里**，不是字面意义上的"同一个对象"。如果两个进程的 `chat_template`/`tool_parser` 等构造参数没有严格保持一致（比如只在完整 server 的启动命令里传了 `--chat-template`，render-only 进程漏传），两边渲染出来的 token 序列会出现细微不一致，而这种不一致不会在任何一边单独测试时暴露出来，只有把两边的输出拿去对比才能发现。
10. **`n_cli_flags=233`（`_lab/out/api_surface.json` 的 `engines.vllm.summary.n_cli_flags`）这个数字里混着影响 HTTP 表面和完全不影响 HTTP 表面的旗标，不能直接拿来跟 SGLang 的 CLI 旗标数对比"谁的旋钮更复杂"**：本篇提到的 `--tokens-only`、`--enable-fault-tolerance`、`--enable-tokenizer-info-endpoint`、`--api-key` 这些直接决定路由生死或认证范围，但 233 个旗标的绝大多数（并行策略、量化配置、调度器参数……）跟 HTTP 层完全无关，只是恰好和路由注册逻辑共享同一个 `argparse.Namespace`。把"CLI 旗标总数"当成"HTTP API 复杂度"的代理指标是一种常见但不成立的简化——这一点留给 `[[09-vLLM-Python-API与EngineArgs]]` 按子系统拆分统计。

---

## 8. 可改进点

以下均按 `本库推断` 标注，基于本篇读到的代码结构，不涉及编造 PR/issue 号（遵守 `_PLAN.md` §1.6）：

- **`AuthenticationMiddleware` 的 `GUARDED_PREFIX` 白名单模型值得改成 SGLang 式的默认保护、显式豁免**：现状是"不在白名单里就不查"，任何新增的、没有刻意挂在 `/v1`/`/v2`/`/inference`/`/cohere` 下的路由（无论是核心团队自己加的还是第三方插件加的）都会**默认不设防**，这个默认值的方向和大多数安全实践（"默认拒绝，显式放行"）是反的。改成"默认所有非 `/health` 路径都要 `--api-key`，只有显式标注的端点（health/metrics/ping 这类）例外"，能一次性堵住 `/invocations`、`/generative_scoring`、`/pooling`、`/classify`、`/score`、`/rerank` 这批已经在文档里被点名的暴露面，且未来新增路由不需要开发者记得"要不要挂在受保护前缀下"。
- **`/abort_requests` 的双重注册应该在其中一个注册点加一次"路径已存在"检测并主动报错**：`register_api_routers()`（`vllm/entrypoints/launchers/api_server/routers.py:12`）目前对 `VLLM_SERVER_DEV_MODE=1` 和 `args.tokens_only=True` 是否同时为真没有任何校验，`## 7` 第 2 条推断出的"其中一个版本会变成永远匹配不到的死代码"这种情况，本该在启动时就用一次路由表遍历检查出来并报错或至少打印警告，而不是留给运行时静默发生。
- **`/tokenizer_info`（`--enable-tokenizer-info-endpoint`）和 dev 族的 `/server_info` 都涉及"配置/模板信息泄露"这一类风险，但门控方式不统一**：前者是独立 CLI 旗标，后者绑在 `VLLM_SERVER_DEV_MODE` 这个大开关上——如果一个运维只想开 `/server_info` 用于内部监控，却不想连带打开 `/collective_rpc` 这种"extremely dangerous"级别的端点，目前没有办法只开一部分 dev 族路由，只能全开或全不开（`## 5` 决策 5 已经展开这一点）。可以考虑把 `/server_info`、`/reset_prefix_cache` 这类"只读或低风险"操作从 `VLLM_SERVER_DEV_MODE` 里拆出来，单独给一个风险级别更低的开关。
- **Cohere 兼容三个端点（`/cohere/v2/chat`、`/v2/embed`、`/v2/rerank`）的可用性策略不统一，容易让用户以为"配了 `VLLM_ENABLE_COHERE_API` 就等于完整 Cohere 兼容"**：实际上 `/v2/embed`/`/v2/rerank` 从来不受这个环境变量影响（`## 5` 决策 6 已展开）。值得在文档或 `/v1/models`/`/server_info` 之类的自省端点里显式列出"当前这次启动实际生效的 Cohere 兼容端点有哪些"，而不是让用户靠翻源码才能确认。
- **`_lab/api_surface.py` 的路由抽取器应该额外标注每条路由的"注册条件"字段**：目前 `api_surface.json` 的每条路由记录只有 `path`/`method`/`handler`/`file`/`line`，没有记录这条路由是否被包在 `if envs.VLLM_SERVER_DEV_MODE:`、`if "generate" in supported_tasks:` 这类条件语句内部——本篇 `## 2.2` 的"注册条件"列完全是人工读代码补上去的。如果抽取器能顺带记录"这个 `@router.post` 调用所在的最近一层 `if` 条件表达式的源码文本"（AST 层面可以做，遍历到装饰器所在函数时记录外层 `ast.If` 节点的 `test` 字段），下游文章和自动化审计都能省掉这一步人工核对。
- **`n_routes`/`n_unique_paths` 之间的差值应该拆成"同 handler 多方法"和"多 handler 同路径"两类分别计数，而不是合成一个数字**：`## 7` 第 8 条已经展开这个问题——当前 `summary` 里 `n_routes - n_unique_paths = 2` 这个差值无法直接看出到底是良性的方法复用（`/ping`）还是需要注意注册顺序的路径冲突（`/abort_requests`）。抽取器完全可以在遇到路径重复时，判断命中的是同一个 `(file, line, handler)` 还是不同的，分别计入两个独立字段（比如 `n_multi_method_paths` 和 `n_conflicting_paths`），后者出现时甚至可以直接在抽取阶段打印警告，因为这正是 `_PLAN.md` §1.1 强调的"空结果/异常结果要当 bug 查"的同类情形。

---

## 9. 自测题与延伸阅读

**闭卷自测题**（先合上本篇再答，答完再翻回去对）：

1. `vllm/entrypoints/openai/api_server.py` 这个文件现在还是不是 vLLM HTTP 层的主装配点？如果不是，真正的装配链从哪个函数开始，经过哪几层？
2. 配置了 `--api-key` 之后，`/invocations` 和 `/v1/chat/completions` 谁需要认证、谁不需要？依据是哪一行代码判定的？
3. `/cohere/v2/chat`、`/v2/embed`、`/v2/rerank` 这三个 Cohere 兼容端点，各自的注册条件是什么？为什么不统一？
4. `/v1/chat/completions/render` 和 `/v1/chat/completions` 内部调用的是不是同一份渲染逻辑？证据是哪个方法、哪个共享对象？
5. `/sleep` 的 level 1 和 level 2 语义分别是什么？为什么 level 2 唤醒后需要额外调用 `collective_rpc("reload_weights")`？
6. `/abort_requests` 这个路径在源码里被注册了几次？如果 `VLLM_SERVER_DEV_MODE=1` 和 `--tokens-only` 同时打开，哪个版本会真正生效？
7. `ChatCompletionRequest` 相对 OpenAI 标准缺的 8 个字段，为什么 vLLM 和 SGLang 缺的是完全一样的 8 个？
8. `/invocations` 端点怎么判断一个请求体应该转发给 chat completion 逻辑还是 embedding 逻辑？这种做法的代价是什么？
9. 为什么 `CORSMiddleware` 明明最先注册，实际请求路径上却比后注册的 `AuthenticationMiddleware` 更晚处理请求？这对 `OPTIONS` 预检请求意味着什么？

**延伸阅读**（仅从 `_PLAN.md` §6 名册挑选）：

- [[01-vLLM-全景与代码地图]] —— 先建立 vLLM 整体坐标系（三层语言栈、model_executor 与 v1 的规模对比），再回来看本篇的 HTTP 层细节
- [[09-vLLM-Python-API与EngineArgs]] —— 本篇多处标注"需要去 EngineArgs/CLI 核实"的旗标（`enable_force_include_usage` 具体由哪个 CLI 选项控制、`--tokens-only` 的完整语义），在那篇有系统性梳理
- [[12-vLLM-PD分离与KV-Connector]] —— 本篇 `kv_transfer_params`/`ec_transfer_params` 字段和 `/inference/v1/generate` 只在 HTTP 表面给出坐标，PD 分离的内部机制在那篇展开
- [[08-SGLang-HTTP-API表面全解]] —— SGLang 侧的对应篇章，`## 6` 的跨引擎认证模型对照（白名单 vs 默认保护）可以在那篇找到 SGLang 视角的镜像描述
- [[02-vLLM-V1架构与EngineCore循环]] —— `/sleep`、`/collective_rpc` 最终都要落到 `AsyncLLM`/`EngineCore` 的方法调用，本篇只看到 HTTP 层这一半，另一半在那篇
