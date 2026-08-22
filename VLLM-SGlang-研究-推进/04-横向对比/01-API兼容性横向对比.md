# API 兼容性横向对比 —— "OpenAI 兼容"四个字兼容到什么程度

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）；其余引擎的 sha 见 `_lab/out/repo_stats.json` 的 `ref` 字段（sglang `15a43983`/2026-08-22、tensorrt-llm `75b023cd`/2026-08-22、lmdeploy `1263d5cd`/2026-08-21、lightllm `87881ae7`/2026-08-21、mlc-llm `9fa644f5`/2026-08-17、llama.cpp `2c6b141e`/2026-08-22、tgi `b4adbf2f`/2026-03-21、dynamo `2a6da14c`/2026-08-22、ktransformers `95009ea6`/2026-08-21、mooncake `3b5a5941`/2026-08-22、tokasaurus `72689052`/2025-08-19）。本篇引用跨引擎源码时一律加 `引擎名:` 前缀；不带前缀的引用默认指 `vllm`。
> **一句话**：12 家都说"OpenAI 兼容"，字段集合的交集只有 22 个。

## 0. 结论先行

- **"OpenAI 兼容"不是一个二元判定，是一条从"路由名字一样"到"字段语义一样"逐层变窄的漏斗。** 12 个引擎在 `_lab` 里共抽到 **236 条不同路径**，`/v1/chat/completions` 这一条 10 家都有（`_lab/out/compare.md` §B）；但把镜头拉到请求体字段，vLLM 与 SGLang 的 `ChatCompletionRequest` 都是 **68 个字段**，其中只有 **22 个**在 OpenAI 官方字段表里出现——也就是说，两家"兼容"程度最高的引擎，各自资产负债表上私有字段（46 个）都是标准字段（22 个）的**两倍还多**。
- **vLLM 与 SGLang 缺的 8 个 OpenAI 标准字段字面上完全相同**：`audio` `function_call` `functions` `metadata` `modalities` `prediction` `service_tier` `store`（`_lab/out/compare.json` 的 `chat_request.vllm.missing_vs_openai` 与 `chat_request.sglang.missing_vs_openai`）。这不是巧合，`function_call`/`functions` 是 OpenAI 自己已经用 `tools`/`tool_choice` 取代的旧字段，另外 6 个（音频输入输出、多模态开关、预测输出、服务分级、持久化 metadata、`store` 落库）才是两家都还没跟上的**真实功能缺口**——这条判断是**本库推断**，依据是 OpenAI 官方字段表的字段命名与两家现存字段的功能覆盖对照，未在源码里找到显式声明。
- **同一个字段名在不同引擎里可以是完全相反的语义**：`priority` 这个词，vLLM 默认"数值越小越先处理"（`vllm/entrypoints/openai/chat_completion/protocol.py:380`），TensorRT-LLM 默认"数值越大越先处理"（`tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:1020`），SGLang 默认行为与 TRT-LLM 一致但留了一个开关能切换成 vLLM 那种（`sglang:python/sglang/srt/server_args.py:892`）。把同一个 `priority: 1` 发给三家，效果可能是"优先处理"也可能是"垫底处理"。
- **`--api-key` 保护面是四个路径前缀，不是"整个服务器"**：vLLM 官方文档明写 `--api-key` 只保护 `/v1` `/v2` `/inference` `/cohere` 四个前缀（`docs/usage/security.md:147`），`/sleep`、`/collective_rpc`、`/scale_elastic_ep` 这类运维端点即便配了 key 也不设防；不过 `/sleep`/`/collective_rpc` 还叠了一层"默认根本不注册"的开关（`VLLM_SERVER_DEV_MODE=1` 才挂载），`/scale_elastic_ep` 没有这层开关，只要支持 `generate` 任务就默认挂着——同一句"无鉴权"背后是两种不同的暴露程度，`## 5.5` 展开。
- **本篇数据本身有三种不同根因的漏抽**，这不是"顺带一提"，是理解整张对比表之前必须知道的口径：llama.cpp 的正则漏抽了真实存在的 `/v1/models`（源码里确实写了这条路由，只是正则没认出来），KTransformers 的 AST 漏抽了真实存在的 `/v1/chat/completions`（源码里通过 `APIRouter(prefix='/v1')` 组合出这条路径，AST 抽取器不追踪路由前缀的层层拼接），两者的漏抽原因完全不同，`## 6` 有完整复现。

---

## 1. 它在系统里的位置

`[[08-vLLM-HTTP-API表面全解]]` 和 `[[08-SGLang-HTTP-API表面全解]]` 已经把各自引擎的路由装配链、条件注册闸门、协议转换细节讲清楚了——那两篇是"单引擎纵切"，本篇不重复它们的内部机制，只做**跨引擎的那一层**：把 12 个引擎的路由表和字段集当成集合来运算，回答"从 A 换到 B，`/v1/chat/completions` 这五个字到底能不能直接照抄请求体"这个问题。

数据来源是 `_lab/` 确定性层的四个产物：
- `_lab/out/api_surface.json` —— 每个引擎用 AST/正则/混合三种方法抽出的路由表与 pydantic 协议类字段；
- `_lab/out/compare.json` / `compare.md` —— 在 `api_surface.json` 之上做的跨引擎集合运算（路由并集、字段并集、同名字段的引擎归属）；
- `_lab/out/repo_stats.json` —— 各引擎 clone 时的 sha 与日期，本篇开头的取证基准表就是从这里抄的。

本篇只看 HTTP 请求/响应这一层的"表面兼容性"：路由存不存在、字段叫什么、字段语义是否一致、鉴权覆盖到哪。不看各引擎内部怎么把这个请求体转换成调度器能理解的对象（那是 `[[03-SGLang-RadixAttention与前缀缓存]]`、`[[03-vLLM-调度器解剖]]` 这类篇目的范围），也不产出任何性能数字。

之所以值得单独写一篇跨引擎对比，而不是让读者自己去读 12 份单引擎文档再脑内做差集，是因为**"兼容"这个判断本身具有欺骗性的复合结构**：路由层面兼容（都有 `/v1/chat/completions`）不代表字段层面兼容（22/68 字段重叠），字段名相同不代表字段语义相同（`## 5.3`），协议标准相同（都叫"OpenAI 兼容"）不代表鉴权模型相同（`## 5.5`）。任何一层单独拿出来看都会给出过于乐观的结论，只有把四层叠在一起，才能回答"我现在用 vLLM 客户端代码，能不能直接指向 SGLang 的 endpoint"这个实际问题——这正是本篇存在的理由。

---

## 2. 代码地图（文件 → 职责，带行号）

| 主题 | 文件:行 | 说明 |
|---|---|---|
| vLLM `ChatCompletionRequest` 类定义 | `vllm/entrypoints/openai/chat_completion/protocol.py:213` | 68 字段，22 个 OpenAI 标准 |
| vLLM `priority` 字段（数值越小越先） | `vllm/entrypoints/openai/chat_completion/protocol.py:380` | 与 TRT-LLM/SGLang 默认方向相反 |
| vLLM `session_id` 字段（纯观测用途） | `vllm/entrypoints/openai/chat_completion/protocol.py:398` | 与 `trace_headers` 并列传给 `generate()`，不驱动任何服务端状态 |
| vLLM `--api-key` 保护前缀 | `vllm/entrypoints/serve/middleware/authenticate.py:11` | `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")` |
| vLLM `/sleep` 路由注册 | `vllm/entrypoints/serve/dev/sleep/api_router.py:21` | 仅 `VLLM_SERVER_DEV_MODE=1` 时挂载 |
| vLLM `/collective_rpc` 路由注册 | `vllm/entrypoints/serve/dev/rpc/api_router.py:23` | 同上，且不落在 `GUARDED_PREFIX` 内 |
| vLLM `/scale_elastic_ep` 路由注册 | `vllm/entrypoints/serve/elastic_ep/api_router.py:29` | 无 dev-mode 闸门，`generate` 任务下默认挂载 |
| vLLM Anthropic 错误转换 | `vllm/entrypoints/anthropic/api_router.py:39` | `translate_error_response()`，逐 handler 手动调用 |
| vLLM Cohere 端点开关 | `vllm/entrypoints/cohere/api_router.py:212` | 需 `VLLM_ENABLE_COHERE_API=1` 且装了 `cohere` SDK 才注册 |
| SGLang `ChatCompletionRequest` 类定义 | `sglang:python/sglang/srt/entrypoints/openai/protocol.py:823` | 同样 68 字段，22 个 OpenAI 标准 |
| SGLang `min_tokens` 字段 | `sglang:python/sglang/srt/entrypoints/openai/protocol.py:895` | 公开字段名与 vLLM/TRT-LLM 对齐 |
| SGLang `session_id` 字段（驱动 Radix 原生会话） | `sglang:python/sglang/srt/entrypoints/openai/protocol.py:906` | 与 vLLM 同名不同义 |
| SGLang `session_id` 落到调度器的真实分支 | `sglang:python/sglang/srt/managers/scheduler.py:2419` | `session_controller` 按 `session_id` 找会话状态 |
| SGLang 优先级调度方向开关 | `sglang:python/sglang/srt/server_args.py:892` | `schedule_low_priority_values_first`，默认 `False`（即默认数值越大越先，与 TRT-LLM 一致） |
| SGLang Anthropic 翻译层文档字符串 | `sglang:python/sglang/srt/entrypoints/anthropic/serving.py:187` | 明写"纯翻译层"，无自己的 tokenize/模板逻辑 |
| LMDeploy `ChatCompletionRequest` 类定义 | `lmdeploy:lmdeploy/serve/openai/protocol.py:171` | 43 字段，22 个 OpenAI 标准 |
| LMDeploy `session_id` 字段（int，驱动真实 Session 对象） | `lmdeploy:lmdeploy/serve/openai/protocol.py:211` | 与 vLLM/SGLang 同名三义中的第三种 |
| LMDeploy `min_new_tokens` 字段 | `lmdeploy:lmdeploy/serve/openai/protocol.py:217` | 与 vLLM/SGLang/TRT-LLM 的 `min_tokens` 同义不同名 |
| LMDeploy Anthropic Messages 端点 | `lmdeploy:lmdeploy/serve/anthropic/endpoints/messages.py:69` | 独立 `anthropic/` 子包实现 |
| TensorRT-LLM `ChatCompletionRequest` 类定义 | `tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:882` | 53 字段，20 个 OpenAI 标准 |
| TensorRT-LLM `min_tokens` 字段 | `tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:939` | 与 vLLM/SGLang 同名同义 |
| TensorRT-LLM `priority` 字段（数值越大越先） | `tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:1020` | `[0.0, 1.0]`，与 vLLM 反向 |
| TensorRT-LLM `cache_salt` 字段校验 | `tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:1147` | 与 vLLM/SGLang 语义一致的正例 |
| LightLLM `ChatCompletionRequest` 类定义 | `lightllm:lightllm/server/api_models.py:191` | 31 字段，21 个 OpenAI 标准 |
| LightLLM Anthropic 层依赖外部包 | `lightllm:lightllm/server/api_anthropic.py:76` | 缺 `litellm` 包时请求期抛 `RuntimeError` |
| MLC-LLM `ChatCompletionRequest` 类定义 | `mlc-llm:python/mlc_llm/protocol/openai_api_protocol.py:258` | 20 字段，19 个 OpenAI 标准，私有字段仅 1 个 |
| llama.cpp `/v1/models` 真实注册但被本库正则漏抽 | `llama.cpp:tools/server/server.cpp:240` | 源码里确有此路由，`## 6` 复现漏抽原因 |
| llama.cpp `/infill`、`/props` 注册 | `llama.cpp:tools/server/server.cpp:237` `llama.cpp:tools/server/server.cpp:252` | 单机文本编辑器场景的原生端点 |
| llama.cpp Anthropic Messages 路由 | `llama.cpp:tools/server/server.cpp:251` | C++ 侧同样实现了 `/v1/messages` |
| KTransformers 真实路由是 `/v1/chat/completions` | `ktransformers:archive/ktransformers/server/api/openai/__init__.py:7` | `APIRouter(prefix='/v1')`，AST 未追踪此拼接 |
| KTransformers 端点文件本身只写 `/chat/completions` | `ktransformers:archive/ktransformers/server/api/openai/endpoints/chat.py:136` | 装饰器里看不到 `/v1` 前缀 |
| KTransformers 官方声明该服务器已归档 | `ktransformers:README.md:159` | "the original integrated KTransformers framework has been archived" |
| Mooncake 定位声明 | `mooncake:README.md:32` | "the serving platform for Kimi"——是被集成的传输/KV 引擎，不是独立推理服务 |
| SGLang RLHF 权重热更新族起点 | `sglang:python/sglang/srt/entrypoints/http_server.py:1339` | `/init_weights_update_group`，权限标记见下条 |
| SGLang 运维端点权限标记 | `sglang:python/sglang/srt/utils/auth.py:128-137` | `AuthLevel.ADMIN_OPTIONAL`，未配 key 时默认不设防 |
| Dynamo Anthropic 端点实验性开关 | `dynamo:components/src/dynamo/frontend/frontend_args.py:489` | `--enable-anthropic-api`，默认关闭且标注 `[EXPERIMENTAL]` |

以上 30 条覆盖 9 个引擎、10 个源文件类型（`.py`/`.cpp`/`.md`），远超 `## 2` 硬指标要求的 10 条。

---

## 3. 核心数据结构

### 3.1 `ChatCompletionRequest` 字段规模（抄自 `_lab/out/compare.md` §C）

| 引擎 | 字段总数 | OpenAI 标准 | 引擎私有 | 缺席的 OpenAI 字段 |
|---|---:|---:|---:|---:|
| vLLM | 68 | 22 | 46 | 8 |
| SGLang | 68 | 22 | 46 | 8 |
| TensorRT-LLM | 53 | 20 | 33 | 10 |
| LMDeploy | 43 | 22 | 21 | 8 |
| LightLLM | 31 | 21 | 10 | 9 |
| MLC-LLM | 20 | 19 | 1 | 11 |

**口径提醒**：这张表只覆盖 6 个引擎——TGI（Rust）、llama.cpp（C++）、Dynamo（Rust 为主）三家在各自仓库里**不存在 Python `class ChatCompletionRequest`**（`_lab/out/api_surface.json` 里三家的 `protocol_classes` 长度均为 0），本库的字段抽取器只解析 Python pydantic 类，天然覆盖不到它们，`## 6` 详细说明。这不代表这三家字段更少或更兼容，只代表**本库现在的工具够不到那一层**，如实标"未查证"而不是拿别的数字顶替。

### 3.2 `/v1/*` 路由覆盖矩阵（节选，完整表见 `_lab/out/compare.md` §B）

12 个引擎共抽到 **236 条不同路径**（`compare.json` 的 `routes` 数组长度）。下面按"覆盖广度"分三档摘录：

**全员或近全员覆盖（核心兼容面）**

| 路径 | 覆盖数 | 未覆盖的引擎 |
|---|---:|---|
| `/v1/chat/completions` | 10/12 | ktransformers、mooncake |
| `/v1/completions` | 10/12 | ktransformers、mooncake |
| `/v1/models` | 9/12 | ktransformers（见 `## 6`）、llama.cpp（见 `## 6`）、mooncake |
| `/v1/embeddings` | 7/12 | ktransformers、lightllm、mooncake、tgi、tokasaurus |
| `/v1/responses` | 7/12 | ktransformers、mlc-llm、mooncake、tgi、tokasaurus |
| `/v1/messages`（Anthropic） | 6/12 | dynamo/lightllm/llama.cpp/lmdeploy/sglang/vllm 有；ktransformers、mlc-llm、mooncake、tensorrt-llm、tgi、tokasaurus 无 |

**中间档（3~6 家覆盖，能看出"第二梯队"共识）**

| 路径 | 覆盖数 | 覆盖引擎 |
|---|---:|---|
| `/generate`（各家自己的原生入口，非 OpenAI 协议） | 6/12 | dynamo、ktransformers、lightllm、lmdeploy、sglang、tgi |
| `/health` | 6/12 | dynamo、lightllm、lmdeploy、sglang、tensorrt-llm、tgi |
| `/v1/messages/count_tokens`（Anthropic 计数端点） | 5/12 | lightllm、llama.cpp、lmdeploy、sglang、vllm |
| `/v1/audio/transcriptions` | 3/12 | llama.cpp、sglang、vllm |
| `/v1/rerank` | 3/12 | llama.cpp、sglang、vllm |
| `/v1/realtime` | 3/12 | dynamo、sglang、vllm |
| `/v1/responses/{response_id}`（Responses 状态查询/取消） | 3/12 | sglang、tensorrt-llm、vllm |
| `/v1/score` | 2/12 | sglang、vllm |

值得单独一提：`/generate` 这条"各家自己最早的原生接口"如今只剩 6 家还留着，**vLLM 和 TensorRT-LLM 都没有保留**——两家都已经把 OpenAI 协议当成唯一对外入口，原生接口要么从未存在（TensorRT-LLM 走的是自家 Executor API，走 gRPC/C++ 层，不经过这层 HTTP 原生端点），要么已经在某个历史版本里被移除（vLLM 现存的最接近"原生"的端点是 `/inference/v1/generate`，已经是 OpenAI 化之后的产物）。这从侧面印证 `## 5.1` 的判断：路由表是产品定位的化石记录，**留没留原生端点，本身就是"这个引擎多大程度上把 OpenAI 协议当唯一门面"的一个信号**。

**只有一到两家的独占端点（挑几个讲清楚为什么存在）**

| 路径 | 独占引擎 | 为什么存在 |
|---|---|---|
| `/cohere/v2/chat` | vLLM | 第三套协议兼容，默认关闭，见 `## 5.4` |
| `/v1/audio/speech` | Dynamo | Dynamo 定位是"多后端统一网关"，语音合成是它接入的众多后端能力之一 |
| `/v1/data_transceiver_state` | TensorRT-LLM | PD 分离场景下查询 KV 传输引擎状态，NVIDIA 自家 Executor API 的直接映射 |
| `/infill` | llama.cpp | 代码补全场景的"中间填空"，llama.cpp 面向单机文本编辑器集成，其余引擎面向 Web 服务没有这个场景 |
| `/v1/chat/completions/render` | vLLM | 把"组装 prompt 但不跑推理"这一步单独开成端点，服务于无 GPU 的 render-only 进程，见 `08-vLLM-HTTP-API表面全解` |
| `/v1/batches` | Dynamo、Tokasaurus | OpenAI Batch API 语义，两家都在做"异步批处理任务队列"，其余引擎没有对应的任务持久化层 |
| `/slots/:id_slot` | llama.cpp | 单进程多 slot 并发槽位的调试接口，对应 llama.cpp 的"进程内槽位"并发模型，其余引擎没有这个并发单元 |
| `/hicache/storage-backend` | SGLang | 分层前缀缓存（HiCache）的存储后端查询，对应 `[[11-SGLang-PD分离与HiCache分层]]` 描述的多级缓存架构，其余引擎没有对等的分层缓存管理面 |
| `/distserve/p2p_connect` 等 6 条 | LMDeploy | PD 分离场景下节点间点对点连接管理，是 LMDeploy 自家 DistServe 实现的内部管控面，命名风格（`p2p_connect`/`p2p_drop_connect`/`p2p_initialize`）看得出是独立于 vLLM/SGLang PD 实现的另一套协议 |
| `/v2/models/:model_name/versions/:model_version/infer` | TGI | KServe v2 推理协议的直接映射，服务于已经在用 Kubernetes/KServe 生态做模型服务编排的团队，与 `/v1/*` 是两条完全独立的协议栈 |

### 3.3 字段级重叠度（抄自 `_lab/out/compare.json` 的 `chat_field_matrix`）

124 个字段（6 引擎并集）里，恰好被 2~3 家共有、且不是 OpenAI 标准字段的有 **24 个**——这批字段是"看起来眼熟但最该核实语义"的高危区，`## 5.3` 展开三个具体例子。

---

## 4. 主流程走读

本篇没有"一次请求怎么被处理"这种单引擎主流程，取而代之的是**本篇数据本身是怎么产生的**——这条流水线决定了下面所有表格的可信边界：

```
_src/<engine>/            （--depth 1 clone 的真实源码，12 个仓库）
      │
      ▼
_lab/api_surface.py       按 method 字段选择 AST 或正则抽取
      │  ├─ AST 方法（vllm/sglang/lmdeploy/lightllm/tensorrt-llm/mlc-llm/tokasaurus/ktransformers）
      │  │     遍历 @app.get/@router.post 装饰器 + pydantic BaseModel 子类
      │  ├─ 正则方法（tgi/llama.cpp）
      │  │     ctx_http\.(get|post)\(\s*"([^"]+)"  /  \.route\("...", (get|post)(...
      │  └─ 混合方法（dynamo）
      │        ast 找 handler，regex 从 unwrap_or("/v1/xxx") 默认值里抠路径
      ▼
_lab/out/api_surface.json  （逐引擎：routes[]、protocol_classes[]、cli_flags[]…）
      │
      ▼
_lab/compare.py           跨引擎集合运算：路由并集/差集、字段并集/差集、同名旋钮默认值
      │
      ▼
_lab/out/compare.json / compare.md   （本篇 `## 3` 的表全部照抄自这里）
```

三种抽取方法对应三种系统性漏抽风险，`## 6` 用真实案例逐一复现。这条流水线里**唯一手工判断的环节**是"OpenAI 标准字段清单"——`compare.json` 的 `notes.openai_field_list_source` 记录为 `platform.openai.com/docs/api-reference/chat/create, 2026-08 人工录入`，也就是「文档所述」证据，不是「源码为证」，本篇 `## 3.1` 的"缺席字段"判断继承这条口径。

---

## 5. 设计决策与代价

### 5.1 独占端点：路由覆盖矩阵背后是产品定位差异

**决策：每个引擎在 `/v1/*` 之外的独占端点，几乎都能一一映射回它的目标部署场景。**

- **为什么这样**：`/infill`（llama.cpp）对应"单机跑在代码编辑器背后"，`/v1/data_transceiver_state`（TensorRT-LLM）对应"数据中心级 PD 分离查询传输引擎状态"，`/v1/batches`（Dynamo、Tokasaurus）对应"异步任务队列"——路由表本质上是产品定位的化石记录，不是随手加的功能。
- **不这样会怎样**：如果每家都严格只实现 OpenAI 官方端点集合，那么 PD 分离、批处理队列、代码补全这些**场景特化能力**就无处安放，要么塞进已有端点的字段里（造成 `## 5.2` 的字段爆炸），要么干脆做不出来。独占端点是把"这家引擎最独特的能力"暴露成 HTTP 接口的必然结果。
- **什么时候可以不这样**：如果一个团队的目标就是"纯粹的 OpenAI 协议网关"（比如只做多引擎路由转发，不暴露引擎特有能力），那就应该只挂载 `/v1/*` 标准端点、把独占端点全部隐藏在内部——这正是许多"LLM 网关"类项目（不在本库范围内）的做法。

### 5.2 字段级兼容：46 个私有字段分族，8 个共同缺席字段

vLLM 与 SGLang 都是 68 字段、22 个 OpenAI 标准、46 个私有——但**"46 个私有"不是一族**，本库按用途人工归类如下（**本库推断**，依据字段名与代码位置的合理判断，非引擎自身声明的分类；数字可在 `compare.json` 的 `chat_request.<engine>.engine_specific` 里核对总数）：

**vLLM 46 个私有字段**

| 族 | 字段数 | 举例 |
|---|---:|---|
| 采样扩展 | 13 | `min_p` `top_k` `repetition_penalty` `use_beam_search` `truncate_prompt_tokens` |
| 可观测性/调试/追踪 | 12 | `request_id` `session_id` `return_token_ids` `prompt_logprobs` `stream_interval` |
| Prompt 模板构造 | 11 | `chat_template` `add_generation_prompt` `continue_final_message` `thinking_token_budget` |
| KV 缓存/PD 传输控制 | 3 | `cache_salt` `kv_transfer_params` `ec_transfer_params` |
| 多模态 | 2 | `mm_processor_kwargs` `media_io_kwargs` |
| 结构化输出 | 2 | `structured_outputs` `_grammar_from_parser` |
| 调度 | 1 | `priority` |
| 逃生舱/内部 | 2 | `vllm_xargs`（透传给内部采样参数的兜底通道）、`_DEFAULT_SAMPLING_PARAMS` |

**SGLang 46 个私有字段**

| 族 | 字段数 | 举例 |
|---|---:|---|
| 会话/调度 | 9 | `session_id` `priority` `data_parallel_rank` `lora_path` `task` |
| 可观测性/调试 | 10 | `return_hidden_states` `return_routed_experts` `rid` `return_meta_info` |
| 采样扩展 | 6 | `min_p` `top_k` `repetition_penalty` `min_tokens` |
| 结构化输出/语法 | 5 | `ebnf` `regex` `stop_regex` `custom_logit_processor` |
| 多模态 | 5 | `images_config` `video_config` `use_audio_in_video` |
| 推理/输出控制 | 5 | `separate_reasoning` `stream_reasoning` `skip_special_tokens` |
| PD 分离/KV 传输 | 4 | `bootstrap_host` `bootstrap_port` `bootstrap_room` |
| 前缀缓存控制 | 2 | `cache_salt` `extra_key` |

**第三个样本：TensorRT-LLM 的 33 个私有字段**（用来验证上面的分族模式不是 vLLM/SGLang 两家的巧合）

| 族 | 字段数 | 举例 |
|---|---:|---|
| 采样扩展 | 13 | `best_of` `early_stopping` `length_penalty` `top_p_min` `use_beam_search` |
| Prompt 模板构造 | 10 | `add_generation_prompt` `chat_template` `conversation_params` `thinking_token_budget` |
| Prompt token 直传 | 3 | `prompt_token_ids` `prompt_token_ids_b64` `prompt_ignore_length` |
| 多模态 | 2 | `media_io_kwargs` `mm_processor_kwargs` |
| PD 分离/KV 传输 | 2 | `disaggregated_params` `cache_salt` |
| 调度 | 1 | `priority` |
| LoRA | 1 | `lora_request` |
| Agent/工具编排 | 1 | `agent_hierarchy` |

三个样本（vLLM 46、SGLang 46、TRT-LLM 33）里，"采样扩展"和"Prompt 模板构造"两族**始终是最大的两族**——这不是巧合：OpenAI 标准字段集里的采样参数只有 `temperature`/`top_p`/`frequency_penalty`/`presence_penalty` 四个，而 HuggingFace `generate()` 系的采样参数（`top_k`/`min_p`/`repetition_penalty`/`length_penalty`/`use_beam_search`……）本来就有十几个，三家不约而同地把这批"HF 系但非 OpenAI 系"的采样参数原样透传成私有字段，说明**私有字段膨胀的第一大驱动力是"OpenAI 标准的采样参数集合本来就比开源推理框架的习惯集合窄"**，而不是各家发明了多少全新概念。

- **为什么这样**：两家都在 OpenAI 协议之上叠了自己的调度器（vLLM 的 `kv_transfer_params`/SGLang 的 `bootstrap_*` 对应各自的 PD 分离实现）、自己的结构化输出后端（vLLM 走 `structured_outputs` 统一开关，SGLang 直接暴露 `ebnf`/`regex` 两种语法）、自己的可观测性诉求（两家的调试字段数量都排进前二，这是"生产环境需要比 OpenAI 协议给得更多的可观测性"这一诉求的直接产物）。这些能力如果不放进请求体字段，就得放进只有本引擎知道的 header 或者单独端点，字段化是成本最低的暴露方式。
- **不这样会怎样**：如果拒绝加私有字段、坚持"纯 OpenAI 协议"，PD 分离、Radix 会话续接、专家路由观测这些能力就必须挪到非 `/v1/*` 的独占端点或者环境变量里配置——SGLang 的 `bootstrap_host`/`bootstrap_port`/`bootstrap_room` 恰好证明了这条路径是可行的（它们本可以做成独立端点，但选择了放进请求体，理由是 PD 分离要求这些参数**逐请求**变化，做成端点意味着每个请求都要多一次 HTTP 调用）。
- **什么时候可以不这样**：当引擎只服务单一确定场景（比如 MLC-LLM 面向端侧部署，私有字段只有 1 个 `debug_config`）时，没有必要复刻 vLLM/SGLang 这种"大而全"的私有字段集——**私有字段数量本身就是"这个引擎背后场景有多复杂"的一个粗略代理指标**，不是越多越好。

**共同缺席的 8 个 OpenAI 标准字段**：`audio` `function_call` `functions` `metadata` `modalities` `prediction` `service_tier` `store`。这直接决定"能不能无缝替换"——如果客户端代码用到了这 8 个字段中的任何一个（比如用 `metadata` 做请求打标、用 `store` 落库到 OpenAI 平台、用 `modalities`/`audio` 做语音输出），vLLM 与 SGLang 都会**静默忽略这些字段**（pydantic 模型默认丢弃未声明字段，不会报错），这是本篇 `## 7` 迁移风险表的核心信息来源之一。

### 5.3 私有字段的兼容陷阱：同名不同义、同义不同名

从 `chat_field_matrix` 里挑出恰好被 2~3 家共有的字段（`## 3.3`），三个具体例子：

**例一：`session_id`，三家三种语义**

| 引擎 | 类型 | 真实作用 |
|---|---|---|
| vLLM | `str \| None` | 纯观测用途，和 `trace_headers` 一起传给 `generate()`（`vllm/entrypoints/openai/chat_completion/serving.py:340-346`），**不驱动任何服务端状态**，文档字符串原话是"Stable session identity shared by related requests"（`vllm/entrypoints/openai/chat_completion/protocol.py:398-404`） |
| SGLang | `Optional[str]` | 驱动真实的会话控制器：`scheduler.py` 里按 `session_id` 查 `self.session_controller`（`sglang:python/sglang/srt/managers/scheduler.py:2419-2513`），配合 Radix 原生会话可以复用之前请求的 KV 状态 |
| LMDeploy | `int \| None = -1` | 类型都不一样（int 不是 str），驱动真正的 `Session` 对象生命周期管理（`lmdeploy:lmdeploy/serve/openai/protocol.py:211`），`-1` 是"未指定"的哨兵值 |

同一个字段名，从"纯装饰性 metadata"到"决定服务端要不要保留会话状态"横跨了两个数量级的功能重要性——把 vLLM 客户端代码里"顺手传个 session_id 方便查日志"的习惯直接搬到 SGLang，如果多个不相关请求复用了同一个字符串，会**意外共享 KV 缓存状态**。

**例二：`min_tokens` vs `min_new_tokens`，同一件事两个名字**

vLLM（`vllm/entrypoints/openai/chat_completion/protocol.py:274`）、SGLang（`sglang:python/sglang/srt/entrypoints/openai/protocol.py:895`）、TensorRT-LLM（`tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:939`）三家字段名都是 `min_tokens`；LMDeploy 管同一件事叫 `min_new_tokens`（`lmdeploy:lmdeploy/serve/openai/protocol.py:217`）。值得一提的是 SGLang 自己内部把这个值透传进采样参数字典时用的 key 恰恰是 `"min_new_tokens"`（`sglang:python/sglang/srt/entrypoints/openai/protocol.py:1103`）——公开字段名跟着 vLLM 走，内部实现命名却跟 LMDeploy 一样，说明这确实是社区里两套并存的命名习惯，不是谁抄谁。

**例三：`priority`，同一个名字方向相反**

- vLLM：`int`，默认 `0`，"lower means earlier handling"（`vllm/entrypoints/openai/chat_completion/protocol.py:380-389`）——**数值越小越先处理**。
- TensorRT-LLM：`Optional[float]`，范围 `[0.0, 1.0]`，"higher is served first"（`tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:1020-1026`）——**数值越大越先处理**，与 vLLM 完全相反。
- SGLang：`Optional[int]`，默认行为由 `schedule_low_priority_values_first` 决定，默认值 `False`（`sglang:python/sglang/srt/server_args.py:892-896`），即默认"数值越大越先"（与 TRT-LLM 一致），但可以显式打开开关切换成 vLLM 那种方向。

同一个 `priority: 1` 发给三家默认配置，vLLM 会当"接近最高优先级"处理，TRT-LLM/SGLang 会当"接近最低优先级"处理——这是三个例子里**唯一没有任何一方"更对"**的情形，纯粹是命名共识没有达成。

- **为什么会有这类陷阱**：这些引擎并非从统一规范演化而来，各自的调度器/会话机制是独立设计的，字段名选择服从各自代码库的既有命名习惯（LMDeploy 偏爱贴近 HuggingFace `generate()` 参数名如 `min_new_tokens`；vLLM/TRT-LLM 的调度优先级命名各自贴合自己已有的调度器实现）。
- **不这样会怎样（即"字段名强制统一"的代价）**：任何一家为了跨引擎兼容而重命名自己的字段，都要么破坏向后兼容（已有客户端代码全部要改），要么维护一套双字段名的兼容层（增加自身协议的复杂度）——目前没有引擎选择付这个代价。
- **什么时候可以不这样**：如果客户端代码从一开始就走 OpenAI 标准字段子集（22 个），这类陷阱完全不会触发；只有当客户端为了拿到某个引擎的私有能力（比如 vLLM 的 `priority` 调度）而写了引擎特定代码，再原样把这段代码套到另一个引擎上时，陷阱才会触发。**这类陷阱的根本对策不是"记住每个字段的方向"，是"迁移引擎时把所有私有字段当成全新 API 重新读文档，不要假设名字相同就语义相同"。**

反例（说明不是所有共享字段都危险）：`cache_salt` 在 vLLM、SGLang、TensorRT-LLM 三家语义一致——都是"给 KV 前缀缓存的哈希掺盐，限制跨请求复用范围"（TRT-LLM 的校验器文档字符串："KV cache will be salted with the provided string to limit the kv cache reuse"，`tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:1014-1019`），且这背后对应同一个真实安全问题——vLLM 文档明确把 `cache_salt` 列为 CVE-2025-46570（前缀缓存时序侧信道）的缓解手段（`docs/usage/security.md:534-544`）。**共享字段不必然是陷阱，`cache_salt` 这种由同一个安全问题倒逼出的字段反而是三家少有的真正对齐点。**

### 5.4 OpenAI 之外的第二标准：Anthropic Messages 的六家实现

`/v1/messages` 的真实名单（源码核实，非路由表照抄）：

| 引擎 | 实现方式 | 默认可用性 |
|---|---|---|
| vLLM | 独立 `entrypoints/anthropic/` 子包，逐 handler 手动调用 `translate_error_response()`（`vllm/entrypoints/anthropic/api_router.py:39`） | 默认开启 |
| SGLang | 纯翻译层，内部直接调用 `OpenAIServingChat` 的私有方法转换请求（`sglang:python/sglang/srt/entrypoints/anthropic/serving.py:187`） | 默认开启 |
| LMDeploy | 独立 `serve/anthropic/` 子包 | 默认开启 |
| llama.cpp | C++ 侧原生路由注册（`llama.cpp:tools/server/server.cpp:251`） | 默认开启 |
| LightLLM | 依赖外部 `litellm` 包的适配器，未安装时请求期抛 `RuntimeError`（`lightllm:lightllm/server/api_anthropic.py:76`） | 需要额外 `pip install 'lightllm[anthropic]'` |
| Dynamo | 需要显式 `--enable-anthropic-api`，标注 `[EXPERIMENTAL]`（`dynamo:components/src/dynamo/frontend/frontend_args.py:489`） | 默认关闭 |

vLLM 还多一层 Cohere Chat v2 兼容（`/cohere/v2/chat`，`vllm/entrypoints/cohere/api_router.py:104`），但同样默认关闭，需要 `VLLM_ENABLE_COHERE_API=1` 且装了 `cohere` SDK 才注册（`vllm/entrypoints/cohere/api_router.py:212-243`）——vLLM 的 `/v2/embed`、`/v2/rerank` 则是 pooling 族的一部分，只要模型支持对应任务就始终注册，没有任何开关，同一个厂商的三个兼容端点走两种完全不同的默认可用性策略。

细看 `/v1/messages` 的姊妹端点 `/v1/messages/count_tokens`（`## 3.2` 中间档表已给出 5/12 覆盖），6 家实现 `/v1/messages` 的引擎里只有 Dynamo 没有配套实现 `count_tokens`——与 `/v1/messages` 本身一样，Dynamo 的 Anthropic 支持标注为 `[EXPERIMENTAL]`（见上表），配套的计数端点没有跟进并不意外，这也印证了"实验性功能"这个标签是准确的，不是自谦。

- **为什么这样**：Anthropic Messages API 在 Claude 生态里的地位类似 OpenAI Chat Completions，一旦某个客户端 SDK（比如 Claude Code 本身、各类 Agent 框架）已经针对 Anthropic 协议写好了工具调用/流式解析逻辑，引擎侧实现 `/v1/messages` 就能让这批客户端**零改动**接入——这是"多标准并存"对客户端 SDK 生态最大的意义：客户端不需要为每个后端引擎写一份适配代码，只要引擎实现了它认的那套协议。
- **不这样会怎样**：如果引擎只做 OpenAI 协议，Anthropic 生态的客户端要么自己写一层 Anthropic→OpenAI 的转换 proxy（LightLLM 选的就是这条路，只不过它把转换逻辑委托给了 `litellm` 这个第三方库，而不是自己实现），要么干脆不支持——这也是为什么 6 家里有 2 家（TGI、TensorRT-LLM）选择完全不做：如果目标客户群本来就是走 OpenAI 协议进来的企业客户，额外维护一套协议转换层的收益不足以覆盖维护成本。
- **什么时候可以不这样**：一个引擎如果目标场景高度垂直（比如 MLC-LLM 面向端侧单机部署，Tokasaurus 是研究用途的极简服务器），维护多标准兼容层的边际收益很低，不实现是合理选择。

### 5.5 运维端点的分化：谁把什么样的能力做成一等公民路由

| 引擎 | 运维端点族 | 目标场景 |
|---|---|---|
| vLLM | `/sleep` `/wake_up` `/is_sleeping`（`vllm/entrypoints/serve/dev/sleep/api_router.py:21`）、`/scale_elastic_ep`（`vllm/entrypoints/serve/elastic_ep/api_router.py:29`）、`/collective_rpc`（`vllm/entrypoints/serve/dev/rpc/api_router.py:23`） | 显存让出（RLHF rollout 之间腾空间）、弹性扩缩容、任意 RPC 调试——但**物理隔离**在 `serve/dev/` 目录下，多数需要 `VLLM_SERVER_DEV_MODE=1` |
| SGLang | `/init_weights_update_group` 起（`sglang:python/sglang/srt/entrypoints/http_server.py:1339`）共 16 条权重热更新/显存让出端点 | 同样服务 RLHF 场景，但当成**主服务器的一等公民路由**，只用 `AuthLevel.ADMIN_OPTIONAL` 标记权限级别（`sglang:python/sglang/srt/utils/auth.py:128-137`），未配 key 时默认不设防 |
| llama.cpp | `/infill` `/props` `/apply-template` `/slots/:id_slot`（`llama.cpp:tools/server/server.cpp:237`、`llama.cpp:tools/server/server.cpp:252`） | 面向"本地单机 + 编辑器集成"场景的调试/编辑能力，没有分布式运维端点 |
| TensorRT-LLM | `/cluster_info`（`tensorrt-llm:tensorrt_llm/serve/coordinator_server.py:67`）、`/kv_cache_events`、`/energy_metrics`、`/release_memory`（`tensorrt-llm:tensorrt_llm/serve/openai_server.py:988`、`tensorrt-llm:tensorrt_llm/serve/openai_server.py:977`、`tensorrt-llm:tensorrt_llm/serve/openai_server.py:1039`） | 面向数据中心级集群运维：跨节点集群状态查询、KV 缓存事件流、能耗指标——这是本篇 12 个引擎里唯一暴露"能耗"这个维度的端点，对应 NVIDIA 自己数据中心运维体系的诉求 |

vLLM 与 SGLang 在**功能对等**的 RLHF 权重更新能力上做出了相反的产品判断：vLLM 把它当开发者调试后门（默认不挂载，挂载时打印 `SECURITY WARNING`），SGLang 把它当生产特性（默认挂载，只做权限标记）。这不是谁更安全，是两边对"谁会把推理引擎当 RL rollout worker 用"这件事的产品定位不同——这一判断沿用自 `08-SGLang-HTTP-API表面全解` 的对照结论，本篇核实过两处引用行号均成立。

**已核实的安全事实（上游已知并有文档，不是本库读代码读出的隐藏 bug）**：vLLM `--api-key` 只保护 `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")` 四个前缀（`vllm/entrypoints/serve/middleware/authenticate.py:11`），`docs/usage/security.md` 用整整一节列出哪些端点在哪些条件下不设防（`docs/usage/security.md:143-235`）。细看这份清单会发现**"不设防"内部还分两层**：

1. `/sleep` `/collective_rpc` `/reset_prefix_cache` 等标注为"仅 `VLLM_SERVER_DEV_MODE=1` 时可用"（`docs/usage/security.md:220-231`）——默认部署下这些端点**根本不存在**，谈不上鉴权与否；
2. `/scale_elastic_ep` `/pause` `/abort_requests` `/update_weights` 等标注为"仅当支持 `generate` 任务时可用"（`docs/usage/security.md:192-203`）——这些端点**默认就挂载**，且不受 `--api-key` 保护，是需要部署方自己用反向代理/网络隔离补的真实缺口。

`/collective_rpc` 同时踩中两条（既要 dev-mode 才挂载，挂载后又不受 `--api-key` 保护），文档原话把它标成"extremely dangerous"（`docs/usage/security.md:231`）。

- **为什么这样**：`--api-key` 中间件在 vLLM 里是按路径前缀做的粗粒度拦截（`GUARDED_PREFIX` 是一个元组做 `startswith` 判断），这是实现成本最低的方案——新增一个 `/v1/*` 下的端点自动被保护，不需要逐个端点声明鉴权装饰器；代价是任何**不小心**注册在 `/v1` 之外的端点默认就是裸奔的。
- **不这样会怎样（即"逐端点声明鉴权"的代价）**：如果每个端点都要显式声明"是否需要鉴权"，多出的工程量是每次新增端点都要过一遍安全检查清单，历史上这类逐端点声明的方案容易出现"忘了标注"的疏漏——前缀白名单的失误模式相反，是"忘了把端点放进受保护前缀"，两种方案都依赖人工纪律，只是失误的方向不同。
- **什么时候可以不这样**：单机本地开发、网络完全隔离（不暴露公网端口）的场景下，`--api-key` 覆盖面不完整这件事本身不构成风险——这也是为什么 vLLM 文档反复强调"不要仅依赖 `--api-key`"而不是"修复覆盖面"：生产部署本来就应该叠加反向代理 + 网络隔离，`--api-key` 只是最后一道非必须的软保险。

### 5.6 容易被漏掉的第三个标准：OpenAI 自己的 Responses API

`## 0`/`## 5.4` 讲的是"OpenAI 之外"的第二标准（Anthropic），但还有一个更容易被忽略的事实：**OpenAI 自己在 Chat Completions 之外又推出了一套 Responses API（`/v1/responses`），这是第三套需要单独兼容的协议，不是 Chat Completions 的简单包装**。`## 3.2` 已经给出覆盖数字——7/12 家实现了 `/v1/responses`（dynamo、lightllm、llama.cpp、lmdeploy、sglang、tensorrt-llm、vllm），vLLM、SGLang、TensorRT-LLM 三家都各自定义了独立的 `ResponsesRequest` 类，不是复用 `ChatCompletionRequest`：

- vLLM：`vllm/entrypoints/openai/responses/protocol.py:136`
- SGLang：`sglang:python/sglang/srt/entrypoints/openai/protocol.py:1577`
- TensorRT-LLM：`tensorrt-llm:tensorrt_llm/serve/openai_protocol.py:1204`

**口径提醒（未查证部分）**：`_lab/compare.py` 目前只对 `ChatCompletionRequest` 做了字段级差集运算，`ResponsesRequest` 没有对应的字段对比——本篇因此**不产出** Responses API 的字段级兼容性数字，只能确认"三家都各自定义了独立协议类"这一结构性事实。想知道 Responses API 层面私有字段有多少、缺席哪些 OpenAI 标准字段，需要先给 `compare.py` 补一条 `responses_request` 的差集运算，这是 `## 8` 可改进点之外一个具体的、本篇没有完成的后续工作项。

- **为什么会有 Responses API 这第三个标准**：OpenAI 自己的产品文档把 Responses API 定位为面向"有状态多轮 Agent 工作流"的下一代接口（支持内建工具调用、服务端会话状态），Chat Completions 则保留为"无状态单次补全"接口——两者不是替代关系，是并存关系。开源引擎跟进 Responses API，本质上是在追一个**还在演进中的、OpenAI 自己都没有停止修改的协议**，这与追 Chat Completions（相对稳定）的确定性完全不同。
- **不这样会怎样**：如果引擎只实现 Chat Completions、不跟进 Responses API，任何基于 OpenAI 官方 Agent SDK（默认走 Responses API）构建的客户端就无法直接接入——这是 5 家（ktransformers/mlc-llm/mooncake/tgi/tokasaurus）目前的状态，它们的客户端要么手动转换成 Chat Completions 调用，要么完全接不上这类新款 Agent 框架。
- **什么时候可以不跟进**：Responses API 目前主要服务"官方 Agent SDK 生态"这一个细分场景，如果引擎的目标用户群本来就是通过 LangChain/自建 Chat Completions 客户端接入（不依赖官方 Agent SDK），不跟进 Responses API 的代价很小——这也是为什么面向端侧（MLC-LLM）、面向研究极简部署（Tokasaurus）的引擎选择不做的合理原因，而不是"做不到"。

---

## 6. 口径差异与抽取限制

`## 0` 已经预告：本篇的路由覆盖矩阵和字段对比表**不是无损的**。三种抽取方法各自有系统性盲区，下面用四个**已复现**的真实案例逐一说明——"抽到 0 条"或者"抽不到某条明明存在的路由"要当 bug 去查，不能当事实写进正文，这是 `_PLAN.md` 第 3 节踩过的坑，本篇专门验证过。

### 6.1 正则方法：方法名与括号之间多一个空格就漏抽（llama.cpp）

`_lab/api_surface.py` 里 llama.cpp 的路由正则是 `` ctx_http\.(get|post)\(\s*"([^"]+)" ``——方法名 `get`/`post` 后面**紧跟**左括号，中间不允许空格。但 llama.cpp 源码里同一个文件的路由注册对齐风格不统一：

```cpp
ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));    // ← get 和 ( 之间有空格
```

（`llama.cpp:tools/server/server.cpp:239-240`）。`ctx_http.get (` 这种写法（为了让后面的路径字符串对齐）不匹配 `ctx_http\.(get|post)\(`，于是 `/v1/models` 在 `compare.json` 里被记成"llama.cpp 不支持"——**但源码里这条路由确实存在，而且和 `/models` 指向同一个 handler `routes.get_models`**。这条路由在 `## 3.2` 表里被错误地划进"llama.cpp 未覆盖"一栏，本篇写正文时已按源码事实订正，不能直接抄 `compare.md` 的原始计数。

### 6.2 AST 方法：能抓到字符串字面量，但不追踪 `include_router(prefix=...)` 的拼接（KTransformers）

KTransformers 的 `/chat/completions` 端点装饰器写的确实只有 `/chat/completions`（`ktransformers:archive/ktransformers/server/api/openai/endpoints/chat.py:136`），AST 抽取器如实记录了这个字符串。但这个端点所在的 `router` 在上一层被显式加了前缀：

```python
router = APIRouter(prefix='/v1')          # ktransformers:archive/.../api/openai/__init__.py:7
router.include_router(chat_router)
```

FastAPI 在装配时会把 `prefix` 和子路由路径拼接成 `/v1/chat/completions`，这是运行时真实存在的路径。`_lab/api_surface.py` 的路由抽取只看单个装饰器调用里的字符串字面量，**不追踪 `APIRouter(prefix=...)` 到 `include_router()` 之间的组合关系**——这是和 6.1 完全不同的另一类盲区：6.1 是"字符串没抓到"，6.2 是"字符串抓到了，但少拼了一段"。`_PLAN.md` 任务书原本问"KTransformers 没有 `/v1/chat/completions` 的原因"，核实结论是**它其实有，只是这份对比表因为抽取方法的这个限制而记漏了**。

顺带一提：即便修正这条漏抽，这段 HTTP 服务器代码本身也已经不是 KTransformers 当前主推的用法——仓库自己的 `README.md` 写明"the original integrated KTransformers framework has been archived to the `archive/` directory for reference. The project now organizes the two capabilities above from the `kt-kernel` source tree"（`ktransformers:README.md:159`），项目现在的定位是给 SGLang 等框架提供"Clean Python API"的算子/内核库（`ktransformers:README.md:71`），而不是自带一个独立的 OpenAI 兼容服务器。**两条原因叠加**：一条是本库工具的抽取盲区，一条是上游自己的产品方向迁移，`## 3.2` 表格里"ktransformers 未覆盖 `/v1/chat/completions`"这个结论本身其实是**巧合地"对"**——只是理由和任务书最初的猜测（"端点可能挂别的前缀"）不完全一致：它不是挂在别的前缀下，是挂在**正确的 `/v1` 前缀下、只是这份代码路径已被上游归档**。

### 6.3 混合方法：Dynamo 的路径是运行时默认值，不是固定字符串（已知限制，此处复核）

Dynamo 的路由不是 `@app.post("/v1/xxx")` 这种装饰器写法，而是 `let path = path.unwrap_or("/v1/chat/completions".to_string())`（`dynamo:lib/llm/src/http/service/openai.rs:3896`）——`unwrap_or(...)` 里的字符串是**没有显式传参时的默认值**，Dynamo 支持通过启动参数改写实际挂载路径。`_lab/api_surface.py` 的正则只能抓到这个默认值，抓不到"这个路径本次启动是否被改写"，所以 `## 3.2` 表里 Dynamo 那一列的每一个 ✅ 都要读成"默认配置下如此"，不是"这次部署保证如此"。

### 6.4 非字面量路径参数：环境变量兜底写法完全跳过抽取（SGLang 的 Ollama 兼容层）

SGLang 的 Ollama 兼容路由写成 `@app.post(os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"))`（`sglang:python/sglang/srt/entrypoints/http_server.py:1973`）——装饰器的参数是一个函数调用 `os.environ.get(...)` 而不是字符串字面量。`_lab/api_surface.py` 的 AST 抽取要求装饰器第一个参数是 `ast.Constant`（第 179 行 `isinstance(dec.args[0], ast.Constant)` 的判断条件），遇到 `ast.Call` 节点直接跳过——这类路由**完全不出现**在 `api_surface.json` 里，不是记错，是压根没被数到。`08-SGLang-HTTP-API表面全解` 已经指出 SGLang 至少有 4 条 Ollama 兼容路由（`/api/chat` `/api/generate` `/api/tags` `/api/show`）用这种写法漏抽，本篇复核确认同一根因还影响到 `/vertex_generate`（`sglang:python/sglang/srt/entrypoints/http_server.py:2040`，同样用 `os.environ.get(...)` 兜底）——**这条盲区不是"少数一两条"的偶然，是"用环境变量做路径兜底"这种写法本身系统性地躲过 AST 字面量抽取**。

### 6.5 字段级对比的覆盖边界：没有 Python pydantic 类就没有字段数据

`## 3.1` 的字段规模表只覆盖 6 个引擎，TGI（Rust `struct`）、llama.cpp（C++ `struct`）、Dynamo（Rust 为主）三家的 `protocol_classes` 均为空数组——这不是三家"字段更简单"，是本库的字段抽取器**只解析 Python `class Xxx(BaseModel)` 定义**，用 AST 找 `ast.ClassDef` 节点，遇到非 Python 语言的结构体定义没有对应解析器。这条边界决定了 `## 5.3` 的"私有字段陷阱"例子只能取自 6 个 Python 引擎——TGI/llama.cpp/Dynamo 是否有类似的同名不同义字段，**本篇未查证**，需要人工读 Rust/C++ 源码或另写专门的解析器才能回答。

**小结**：四类漏抽机制互不相同（正则格式敏感 / AST 不追踪路由组合 / 默认值可被运行时改写 / 非字面量参数完全跳过），任何一条单独出现都可能被误读成"这家引擎不支持某功能"。**读这份对比表的正确姿势是：`_lab/out/compare.md` 给的是下界，不是精确值——某个引擎在某一格显示"—"，代表"抽取脚本没找到"，不代表"源码里没有"，本篇 `## 3.2`/`## 5` 已对能核实到的漏抽逐条订正，未核实的維持原表数字并保留这条警告。**

---

## 7. 踩坑与反直觉

### 7.1 迁移风险表

从 A 引擎的客户端代码原样切到 B 引擎，实测会踩到什么——以下每一行都对应上文已核实的源码事实，不猜测未核实的行为。

| 迁移方向 | 具体会踩什么 | 后果 | 证据 |
|---|---|---|---|
| vLLM → SGLang | `session_id` 从"日志观测字段"变成"功能字段"：两个无关请求复用同一个字符串会**意外共享 KV 会话状态**，而不只是日志里看着像同一个会话 | 静默的正确性风险，不会报错 | `## 5.3` 例一 |
| vLLM → SGLang | `priority` 默认方向可能相反：vLLM 里数值小的请求优先，切到 SGLang 默认配置下数值大的请求优先 | 请求排队顺序与预期相反，压测时表现为"高优先级请求反而更慢" | `## 5.3` 例三 |
| vLLM → SGLang | `kv_transfer_params` 与 `bootstrap_host/port/room` 是两套完全不同的 PD 分离配置协议，字段名、语义都不通用 | 携带 vLLM 的 PD 配置直接发给 SGLang，这些字段会被当成未声明字段静默丢弃，PD 分离功能整体失效但请求本身不报错 | `## 5.2` 字段分族表 |
| SGLang → vLLM | `regex`/`ebnf`/`no_stop_trim` 这类结构化输出/停止符控制字段在 vLLM 里不存在，需要改走 vLLM 统一的 `structured_outputs` 参数 | 字段被静默丢弃，约束解码功能整体失效（不是格式错误，是完全不生效） | `## 5.2` 字段分族表 |
| vLLM → TGI | 发送 `model`、`logit_bias`、`n` 字段：TGI 的 `ChatRequest` 确实声明了这三个字段，但文档字符串明写 `[UNUSED]`/`UNUSED`（`tgi:router/src/lib.rs:831`、`tgi:router/src/lib.rs:844`、`tgi:router/src/lib.rs:870`），服务端**接收但完全不使用** | 客户端以为设了 `n=3` 会拿到 3 个候选，实际只回 1 个；`logit_bias` 完全不生效；都不会报错 | `## 2` 代码地图 |
| vLLM → TGI | vLLM 的 46 个私有字段（`priority` `session_id` `kv_transfer_params` `cache_salt`…）在 TGI 的 `ChatRequest` 里全部不存在，且 TGI 的 struct 没有加 `#[serde(deny_unknown_fields)]`（`tgi:router/src/lib.rs` 全文搜索无此属性） | 所有 vLLM 私有字段发给 TGI 都被 serde **静默丢弃**，不会报 422，看起来"请求成功"但对应功能完全没有发生 | 本篇核实（见上） |
| vLLM → TGI | TGI 强制 `response_format` 与 `tools` 互斥（"A request can use `response_format` OR `tools` but not both"，`tgi:router/src/lib.rs:929`），vLLM 没有这条限制 | 同时携带两个字段的请求在 vLLM 上能跑，切到 TGI 后行为未定义/可能报错，需要额外核实 TGI 具体校验逻辑（本篇未深入 TGI 校验代码，仅核实到文档字符串这一层） | `## 2` 代码地图 |
| vLLM → llama.cpp | `/classify` `/score` `/pooling` 在 llama.cpp 的 236 条路径并集里完全没有对应端点（`## 3.2` 路由矩阵） | 分类、非 `/v1` 打分、pooling 三类能力在 llama.cpp 上**没有任何形式的替代端点**，不是换个名字的问题，是能力缺口 | `_lab/out/compare.md` §B |
| vLLM → llama.cpp | LoRA 管理 API 形状不同：vLLM 是 `/v1/load_lora_adapter`/`/v1/unload_lora_adapter` 两个动作端点，llama.cpp 是 `/lora-adapters` 一个资源端点（GET 查询、POST 更新，`llama.cpp:tools/server/server.cpp:270-271`） | 不是字段级差异，是整个调用模式（RPC 风格 vs REST 资源风格）要重写 | `## 2` 代码地图 |
| llama.cpp → vLLM | `/infill` `/props` `/apply-template` 这类面向本地编辑器集成的端点在 vLLM 里没有对应物 | 依赖代码补全"中间填空"语义的客户端整体功能缺失，不是接口不兼容，是场景本身在 vLLM 的产品定位里不存在 | `## 3.2` 独占端点表 |
| vLLM → LMDeploy | `session_id` 类型从 `str \| None` 变成 `int \| None = -1`：如果客户端习惯传 UUID 字符串当 `session_id`，直接发给 LMDeploy 会在字段校验层被拒绝（类型不匹配），而不是像 `## 7.1` 前两行那样静默生效 | 这是本表里**少数会直接报错、而不是静默失败**的迁移风险，报错本身反而比静默失败更容易在测试阶段就发现 | `## 2` 代码地图 |
| vLLM → LMDeploy | `min_tokens` 要改叫 `min_new_tokens`；vLLM 的 46 个私有字段绝大多数在 LMDeploy 的 21 个私有字段里没有对应物（`## 5.2` 字段分族表），尤其是 PD 分离相关的 `kv_transfer_params` 完全没有等价字段 | 同 `## 7.1` 的"静默丢弃"模式：字段名不对的直接被丢弃，字段类型不对的（如上一行）才会报错 | `## 3.1` 字段规模表 |

**为什么迁移风险普遍是"静默失败"而不是"报错失败"**：pydantic（Python 侧）和 serde（Rust 侧）的默认行为都是"忽略请求体里未声明的字段"，这是为了向前兼容——新客户端给旧服务端发新字段不应该导致整个请求失败。但这条对 API 演进友好的设计，恰好也是"引擎间迁移最危险的陷阱都不报错"的根因：**私有字段被静默丢弃，比字段类型错误被 422 拒绝更难发现**，因为请求"看起来成功了"。

**不这样会怎样（即"未声明字段就报错"的代价）**：如果 pydantic/serde 默认拒绝未声明字段，任何客户端在切换引擎版本、甚至切换同引擎的小版本（新增了字段）时都可能因为发送了旧版本没有的字段而报错——这在实践中比"字段被忽略"更容易造成大规模生产事故，所以两边都不这么做。

**什么时候可以不担心这条**：如果客户端严格只使用 OpenAI 标准的 22 个字段（`## 3.1`），完全不触碰任何引擎私有字段，这条"静默失败"的迁移风险基本不存在——代价是放弃了 46 个私有字段背后的能力（结构化输出细粒度控制、PD 分离、调度优先级……）。

### 7.2 三个容易被误读的反直觉点

- **"抽不到"不等于"不存在"，`## 6` 的四个案例里有三个（llama.cpp 的 `/v1/models`、KTransformers 的 `/v1/chat/completions`、SGLang 的 4 条 Ollama 路由）都是"源码里真实存在，工具没抓到"。** 任何只读 `compare.md` 原始数字、不回源码核实的横向对比文章，都会把这几条误报成"该引擎不支持"。
- **KTransformers 的案例是双重巧合**：抽取工具漏抓了真实存在的 `/v1/chat/completions`，但这段代码本身又恰好是上游已经归档、不再是主推方向的遗留实现——两个独立成立的事实拼在一起，让 `## 3.2` 表格里"KTransformers 未覆盖"这个最终展示结论意外地"蒙对了"，但支撑它的中间推理链是错的，`## 6.2` 已经把两条原因拆开说清楚。
- **共享字段名不必然是陷阱**：`## 5.3` 三个反例（`session_id`/`min_tokens` vs `min_new_tokens`/`priority`）容易让人觉得"跨引擎同名字段都不可信"，但 `cache_salt` 恰恰是反例的反例——三家实现语义一致，且背后对应同一个真实安全问题（CVE-2025-46570）。**判断一个共享字段是否可信，不能只看名字，要看它是否对应一个双方都独立认可的、外部世界的具体问题**（时序侧信道是密码学/安全领域的公认问题，`priority` 的调度方向则纯粹是各家内部约定，没有外部标准可对齐）。

### 7.3 切引擎之前，给自己列一张检查清单

把本篇的结论压缩成一个迁移前能照着走一遍的检查清单，不是新结论，是对 `## 5`/`## 7.1` 的重新编排：

1. **先分清客户端用到的字段落在哪个圈层**：OpenAI 标准 22 字段（`## 3.1`）？目标引擎的私有字段（`## 5.2`）？还是恰好两边都有但语义可能不同的共享字段（`## 5.3`）？只有第三类需要真正逐字段核实，前两类分别是"稳"和"注定要重写"。
2. **搜一遍客户端代码里有没有硬编码 `priority` 的具体数值**，如果有，先确认目标引擎的方向约定（`## 5.3` 例三），不要假设数值语义可移植。
3. **确认客户端是否依赖 `session_id` 做跨请求状态延续**（不只是日志打标），如果是，目标引擎必须原生支持"会话"这个概念（SGLang/LMDeploy 有，vLLM 没有，`## 5.3` 例一），否则这段业务逻辑要重新设计，不是换个字段名能解决的。
4. **不要相信"请求返回 200 就说明字段生效了"**——`## 7.1` 已经证明 pydantic/serde 的默认行为是静默丢弃未声明字段，200 只代表"格式合法"，不代表"每个字段都被使用"。迁移后第一件事应该是**对照 `## 5.2` 的私有字段清单，逐个检查目标引擎的响应里是否体现了预期效果**，而不是只看状态码。
5. **鉴权配置不能跨引擎照抄**：`--api-key` 在 vLLM 上只护住 4 个前缀（`## 5.5`），换一个引擎要重新确认它的鉴权中间件覆盖面，不能假设"配了 key 就等于配了鉴权"这件事在所有引擎上程度一致。
6. **`_lab/out/compare.md` 上显示"不支持"的格子，先按 `## 6` 的四种漏抽模式排除一遍工具误报，再下结论**——本篇亲自订正过 3 处，说明这不是小概率事件。

---

## 8. 可改进点

以下均为**本库推断**，不是任何上游 issue/PR 的转述（本库不编 issue 号）：

1. **给"OpenAI 兼容"配一份机器可读的差异声明，而不是一句营销话**。本篇 `## 3.1`/`## 5.2` 能写出来，前提是 `_lab/compare.py` 把两份 pydantic 类做了字段级差集运算——这套方法论本身可以反向输出成一个每个引擎仓库都能自带的 `compat.json`（字段并集、缺席的 OpenAI 标准字段、私有字段清单），维护成本远低于人工写一篇兼容性文档，且能跟着代码自动更新，不会像文档那样过期。
2. **`priority` 这类方向性字段，建议在 OpenAPI schema 层面就用 enum 或显式描述消除歧义，而不是只在 Python 类的 `description=` 里写一句话**。当前 vLLM（`:380`）、TRT-LLM（`:1020`）的字段描述其实都写清楚了方向，问题不是文档缺失，是**客户端开发者切换 base_url 时默认"字段名一样=行为一样"，不会去重新读每个字段的 description**——这是人的习惯问题，不是文档问题，所以更彻底的解法是让方向不一致的字段**改名**（比如 TRT-LLM/SGLang 把 `priority` 改成 `priority_higher_first` 这种自解释命名），而不是指望使用者读文档。
3. **pydantic/serde 的"静默丢弃未声明字段"行为，建议加一个可选的严格模式**（比如 `X-Strict-Fields: true` 请求头或环境变量），命中未声明字段时至少打一条 WARNING 日志而不是完全无声——这不需要改变默认行为（`## 7.1` 已经论证了默认静默有其合理性），只是给想要"迁移时发现哪些字段被丢了"的开发者一个可选的调试开关。据 `## 6.5` 的口径限制，TGI/llama.cpp/Dynamo 三家甚至可能更需要这个开关，因为它们没有 Python 层面 pydantic 校验器那种"至少能看到未声明字段列表"的中间产物。
4. **本库自己的抽取工具存在三处可修的漏抽**（`## 6.1`-`## 6.4`），修复优先级建议：① `_lab/api_surface.py` 的 llama.cpp 正则加 `\s*` 容忍方法名与括号间的空格（一行改动，直接修复 `/v1/models` 这类漏抽）；② AST 抽取器给 `APIRouter(prefix=...)` 到 `include_router()` 的组合关系做一层浅层追踪（收益是修复 KTransformers 这类漏抽，但需要处理跨文件的路由组合树，工作量明显大于①）；③ 非字面量路径参数（`os.environ.get(...)` 这类）至少输出一条"检测到动态路径，未抽取"的占位记录，而不是完全静默跳过——让"抽取脚本主动承认抽不到"比"表格显示为空"更诚实。
5. **给 `_lab/compare.py` 补一条 `ResponsesRequest` 的字段级差集运算**（`## 5.6` 已指出这是缺口）。当前只对 `ChatCompletionRequest` 做了字段对比，但 `/v1/responses` 已经有 7 家实现、3 家（vLLM/SGLang/TensorRT-LLM）各自定义了独立协议类——这条数据目前完全空缺，落地成本和现有 `chat_request` 差集运算是同一套代码换一个类名，性价比很高。

---

## 9. 自测题与延伸阅读

### 9.1 闭卷自测题

1. vLLM 与 SGLang 的 `ChatCompletionRequest` 字段总数、OpenAI 标准字段数、私有字段数分别是多少？两家共同缺席的 8 个 OpenAI 标准字段里，哪两个其实是 OpenAI 自己已经用 `tools`/`tool_choice` 取代的旧字段？
2. `_lab` 的路由抽取表显示 llama.cpp 不支持 `/v1/models`，但源码 `llama.cpp:tools/server/server.cpp:240` 明明写了这条路由注册。漏抽的根因是什么？和 KTransformers 漏抽 `/v1/chat/completions` 的根因是同一类问题吗？
3. 把同一个 `priority: 1` 分别发给 vLLM、TensorRT-LLM、SGLang（默认配置），三家会分别把这个请求当成"高优先级"还是"低优先级"处理？靠哪个 SGLang 启动参数能让它的默认行为对齐 vLLM？
4. `session_id` 字段在 vLLM、SGLang、LMDeploy 三家的类型和真实作用分别是什么？如果把 vLLM 客户端里"用同一个字符串标记同一用户的多次请求"这个习惯原样搬到 SGLang，会引发什么问题？
5. vLLM 的 `--api-key` 保护哪四个路径前缀？`/collective_rpc` 为什么被本篇称为"叠了两层暴露面"？这两层分别是什么？
6. 为什么 TGI、llama.cpp、Dynamo 三家没有出现在 `## 3.1` 的 `ChatCompletionRequest` 字段规模对比表里？这是否代表它们的请求字段确实更少？
7. `cache_salt` 字段在 vLLM/SGLang/TensorRT-LLM 三家语义一致，本篇为什么把它当成"共享字段里的正例"而不是像 `session_id`/`priority` 那样的陷阱？判断标准是什么？
8. OpenAI 自己的 Responses API（`/v1/responses`）和 Chat Completions API 是什么关系？12 个引擎里有几家实现了前者？本篇为什么没有给出 Responses API 的字段级兼容性数字？
9. vLLM 的 46 个私有字段和 TensorRT-LLM 的 33 个私有字段里，哪两族始终是占比最大的两族？这个共性说明了私有字段膨胀的第一大驱动力是什么？

### 9.2 延伸阅读

- `[[08-vLLM-HTTP-API表面全解]]` —— vLLM 单引擎纵切，63 条路由的完整装配链与条件注册闸门
- `[[08-SGLang-HTTP-API表面全解]]` —— SGLang 单引擎纵切，四层兼容协议如何收口成同一个内部请求
- `[[01-TensorRT-LLM]]` —— TensorRT-LLM 全景，本篇多处私有字段（`agent_hierarchy`/`disaggregated_params`/`prompt_token_ids_b64`）的完整上下文在这篇里
- `[[09-vLLM-Python-API与EngineArgs]]` —— 本篇只看 HTTP 请求字段，服务端启动参数（`EngineArgs` 233 个字段）的完整旋钮清单在这篇
- `[[09-SGLang-ServerArgs旋钮全景]]` —— 同上，SGLang 侧的 `ServerArgs`（476 个字段），`## 5.3` 例三引用的 `schedule_low_priority_values_first` 在这篇有完整上下文

### 9.3 原始数据速查

| 文件 | 用途 | 本篇主要引用的键 |
|---|---|---|
| `_lab/out/compare.md` | 人读版对比表 | §B 路由覆盖、§C 字段规模 |
| `_lab/out/compare.json` | 机器读版对比表 | `chat_field_matrix`、`routes`、`chat_request` |
| `_lab/out/api_surface.json` | 逐引擎原始抽取结果 | `<engine>.routes`、`<engine>.protocol_classes`、`<engine>.method` |
| `_lab/out/repo_stats.json` | 各引擎 clone sha 与日期 | `<engine>.ref`，本篇开头取证基准表照抄自此 |

`## 6` 的四个漏抽案例（llama.cpp 正则空格敏感、KTransformers 前缀拼接盲区、Dynamo 默认值可改写、SGLang 非字面量路径）均可在 `api_surface.json` 与对应引擎源码里复现，复现方法见各小节引用的文件行号。
