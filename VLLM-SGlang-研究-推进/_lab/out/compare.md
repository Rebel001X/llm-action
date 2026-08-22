# 横向对比·机器生成表（compare.py 产物，勿手改）

对比引擎：dynamo, ktransformers, lightllm, llama.cpp, lmdeploy, mlc-llm, mooncake, sglang, tensorrt-llm, tgi, tokasaurus, vllm


## A. 工程规模

| 引擎 | commit | 总行数 | Python 产品码 | Python 测试码 | 测试/产品 | CUDA 文件 | Triton 文件 | Rust 行 | C/C++ 行 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TensorRT-LLM | `75b023cd` | 2,452,198 | 792,216 | 603,872 | 0.762 | 429 | 91 | 0 | 598,382 |
| SGLang | `15a43983` | 2,317,985 | 1,254,674 | 605,010 | 0.482 | 246 | 249 | 156,078 | 51,979 |
| NVIDIA Dynamo | `2a6da14c` | 2,220,830 | 193,827 | 215,985 | 1.114 | 2 | 0 | 608,542 | 1,533 |
| vLLM | `7ca49fbe` | 1,914,984 | 932,277 | 505,654 | 0.542 | 169 | 202 | 109,867 | 55,159 |
| llama.cpp | `2c6b141e` | 992,377 | 58,859 | 11,033 | 0.187 | 273 | 0 | 0 | 663,024 |
| Mooncake | `3b5a5941` | 548,895 | 30,573 | 33,413 | 1.093 | 38 | 0 | 4,405 | 420,627 |
| KTransformers | `95009ea6` | 403,076 | 152,237 | 29,114 | 0.191 | 21 | 8 | 0 | 127,399 |
| LMDeploy | `1263d5cd` | 339,214 | 155,986 | 59,422 | 0.381 | 130 | 34 | 0 | 67,110 |
| TGI | `b4adbf2f` | 189,926 | 98,255 | 10,645 | 0.108 | 31 | 4 | 23,443 | 1,251 |
| LightLLM | `87881ae7` | 148,574 | 103,709 | 24,334 | 0.235 | 0 | 129 | 0 | 0 |
| MLC-LLM | `9fa644f5` | 100,240 | 59,516 | 10,673 | 0.179 | 0 | 1 | 0 | 22,690 |
| Tokasaurus | `72689052` | 13,219 | 11,405 | 1,251 | 0.11 | 0 | 0 | 0 | 0 |

## B. `/v1/*` 路由覆盖

| 路径 | dynamo | ktransformers | lightllm | llama.cpp | lmdeploy | mlc-llm | mooncake | sglang | tensorrt-llm | tgi | tokasaurus | vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `/v1/audio/speech` | ✅ | — | — | — | — | — | — | — | — | — | — | — |
| `/v1/audio/transcriptions` | — | — | — | ✅ | — | — | — | ✅ | — | — | — | ✅ |
| `/v1/audio/translations` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/batches` | ✅ | — | — | — | — | — | — | — | — | — | ✅ | — |
| `/v1/batches/{batch_id}` | — | — | — | — | — | — | — | — | — | — | ✅ | — |
| `/v1/chat/completions` | ✅ | — | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/v1/chat/completions/batch` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/chat/completions/control` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/chat/completions/derender` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/chat/completions/input_tokens` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/chat/completions/render` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/classify` | ✅ | — | — | — | — | — | — | ✅ | — | — | — | — |
| `/v1/completions` | ✅ | — | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/v1/completions/derender` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/completions/render` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/data_transceiver_state` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/detokenize` | — | — | — | — | — | — | — | ✅ | — | — | — | — |
| `/v1/embeddings` | ✅ | — | — | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | ✅ |
| `/v1/encode` | — | — | — | — | ✅ | — | — | — | — | — | — | — |
| `/v1/files` | ✅ | — | — | — | — | — | — | — | — | — | ✅ | — |
| `/v1/files/{file_id}` | — | — | — | — | — | — | — | — | — | — | ✅ | — |
| `/v1/files/{file_id}/content` | — | — | — | — | — | — | — | — | — | — | ✅ | — |
| `/v1/health` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/images/edits` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/images/generations` | ✅ | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/images/{image_id}/content` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/load_lora_adapter` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/messages` | ✅ | — | ✅ | ✅ | ✅ | — | — | ✅ | — | — | — | ✅ |
| `/v1/messages/count_tokens` | — | — | ✅ | ✅ | ✅ | — | — | ✅ | — | — | — | ✅ |
| `/v1/models` | ✅ | — | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/v1/models/{model:path}` | — | — | — | — | — | — | — | ✅ | — | — | — | — |
| `/v1/pooling` | ✅ | — | — | — | — | — | — | — | — | — | — | — |
| `/v1/realtime` | ✅ | — | — | — | — | — | — | ✅ | — | — | — | ✅ |
| `/v1/rerank` | — | — | — | ✅ | — | — | — | ✅ | — | — | — | ✅ |
| `/v1/reranking` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/responses` | ✅ | — | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | — | — | ✅ |
| `/v1/responses/input_tokens` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/responses/{response_id}` | — | — | — | — | — | — | — | ✅ | ✅ | — | — | ✅ |
| `/v1/responses/{response_id}/cancel` | — | — | — | — | — | — | — | ✅ | — | — | — | ✅ |
| `/v1/score` | — | — | — | — | — | — | — | ✅ | — | — | — | ✅ |
| `/v1/stream` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/streams/lookup` | — | — | — | ✅ | — | — | — | — | — | — | — | — |
| `/v1/tokenize` | — | — | — | — | — | — | — | ✅ | — | — | — | — |
| `/v1/unload_lora_adapter` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/videos` | ✅ | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/videos/generations` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/videos/{video_id}` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/videos/{video_id}/content` | — | — | — | — | — | — | — | — | ✅ | — | — | — |

## C. ChatCompletionRequest 字段数

| 引擎 | 类 | 文件:行 | 字段总数 | 其中 OpenAI 标准 | 引擎私有 | 缺席的 OpenAI 字段 |
|---|---|---|---:|---:|---:|---:|
| LightLLM | `ChatCompletionRequest` | `lightllm/server/api_models.py:191` | 31 | 21 | 10 | 9 |
| LMDeploy | `ChatCompletionRequest` | `lmdeploy/serve/openai/protocol.py:171` | 43 | 22 | 21 | 8 |
| MLC-LLM | `ChatCompletionRequest` | `python/mlc_llm/protocol/openai_api_protocol.py:258` | 20 | 19 | 1 | 11 |
| SGLang | `ChatCompletionRequest` | `python/sglang/srt/entrypoints/openai/protocol.py:823` | 68 | 22 | 46 | 8 |
| TensorRT-LLM | `ChatCompletionRequest` | `tensorrt_llm/serve/openai_protocol.py:882` | 53 | 20 | 33 | 10 |
| vLLM | `ChatCompletionRequest` | `vllm/entrypoints/openai/chat_completion/protocol.py:213` | 68 | 22 | 46 | 8 |

## D. 服务端配置对象

| 引擎 | 类 | 文件:行 | 字段数 |
|---|---|---|---:|
| KTransformers | `Config` | `archive/ktransformers/server/config/config.py:20` | 0 |
| LMDeploy | `PytorchEngineConfig` | `lmdeploy/messages.py:371` | 48 |
| SGLang | `ServerArgs` | `python/sglang/srt/server_args.py:473` | 476 |
| vLLM | `EngineArgs` | `vllm/engine/arg_utils.py:424` | 233 |

## E. 同名旋钮的默认值（可比 5 个：不同 4、相同 1；**不可比 37 个已排除**）

> 口径：只有当**所有参与引擎的默认值都是字面量**时才判定异同。vLLM 的 `EngineArgs` 大量把默认值转交给子配置（写成 `ModelConfig.dtype` 这种），拿它和别家的字面量直接比会得出假结论，所以这类一律排除、不下判断。

| 旋钮 | dynamo | ktransformers | lightllm | llama.cpp | lmdeploy | mlc-llm | mooncake | sglang | tensorrt-llm | tgi | tokasaurus | vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `block_size` | — | — | — | — | 64 | — | — | — | — | — | — | None |
| `enable_lora` | — | — | — | — | — | — | — | None | — | — | — | False |
| `enable_metrics` | — | — | — | — | True | — | — | False | — | — | — | — |
| `enable_prefix_caching` | — | — | — | — | False | — | — | — | — | — | — | None |

被排除的 37 个旋钮及排除原因见 `compare.json` 的 `shared_knobs[*].why_not_comparable`。

---

> 口径说明：路由集合来自装饰器静态抽取；条件注册（if 分支里 add_api_route）可能漏。  
> OpenAI 字段清单来源：platform.openai.com/docs/api-reference/chat/create, 2026-08 人工录入

