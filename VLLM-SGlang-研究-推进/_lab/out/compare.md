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
| MLC-LLM | `9fa644f5` | 100,240 | 59,516 | 10,673 | 0.179 | 0 | 0 | 0 | 22,690 |
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
| `/v1/images/edits` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/images/generations` | ✅ | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/images/{image_id}/content` | — | — | — | — | — | — | — | — | ✅ | — | — | — |
| `/v1/load_lora_adapter` | — | — | — | — | — | — | — | — | — | — | — | ✅ |
| `/v1/messages` | ✅ | — | ✅ | ✅ | ✅ | — | — | ✅ | — | — | — | ✅ |
| `/v1/messages/count_tokens` | — | — | ✅ | ✅ | ✅ | — | — | ✅ | — | — | — | ✅ |
| `/v1/models` | ✅ | — | ✅ | — | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ |
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

## E. 同名旋钮但默认值不同（共 41 个）

| 旋钮 | dynamo | ktransformers | lightllm | llama.cpp | lmdeploy | mlc-llm | mooncake | sglang | tensorrt-llm | tgi | tokasaurus | vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `allowed_media_domains` | — | — | — | — | — | — | — | dataclasses.field(default_factory=list) | — | — | — | ModelConfig.allowed_media_domains |
| `attention_backend` | — | — | — | — | — | — | — | None | — | — | — | AttentionConfig.backend |
| `block_size` | — | — | — | — | 64 | — | — | — | — | — | — | None |
| `cpu_offload_gb` | — | — | — | — | — | — | — | 0 | — | — | — | UVAOffloadConfig.cpu_offload_gb |
| `dcp_comm_backend` | — | — | — | — | — | — | — | 'ag_rs' | — | — | — | ParallelConfig.dcp_comm_backend |
| `disable_custom_all_reduce` | — | — | — | — | — | — | — | False | — | — | — | ParallelConfig.disable_custom_all_reduce |
| `distributed_executor_backend` | — | — | — | — | None | — | — | — | — | — | — | ParallelConfig.distributed_executor_backend |
| `download_dir` | — | — | — | — | None | — | — | None | — | — | — | LoadConfig.download_dir |
| `dtype` | — | — | — | — | 'auto' | — | — | 'auto' | — | — | — | ModelConfig.dtype |
| `enable_eplb` | — | — | — | — | False | — | — | False | — | — | — | ParallelConfig.enable_eplb |
| `enable_lora` | — | — | — | — | — | — | — | None | — | — | — | False |
| `enable_mamba_cache_stochastic_rounding` | — | — | — | — | — | — | — | False | — | — | — | MambaConfig.enable_stochastic_rounding |
| `enable_metrics` | — | — | — | — | True | — | — | False | — | — | — | — |
| `enable_mfu_metrics` | — | — | — | — | — | — | — | False | — | — | — | ObservabilityConfig.enable_mfu_metrics |
| `enable_prefix_caching` | — | — | — | — | False | — | — | — | — | — | — | None |
| `enable_return_routed_experts` | — | — | — | — | False | — | — | False | — | — | — | ModelConfig.enable_return_routed_experts |
| `hf_overrides` | — | — | — | — | None | — | — | — | — | — | — | get_field(ModelConfig, 'hf_overrides') |
| `kv_cache_dtype` | — | — | — | — | — | — | — | 'auto' | — | — | — | CacheConfig.cache_dtype |
| `language_model_only` | — | — | — | — | False | — | — | False | — | — | — | MultiModalConfig.language_model_only |
| `load_format` | — | — | — | — | — | — | — | 'auto' | — | — | — | LoadConfig.load_format |
| `logprobs_mode` | — | — | — | — | None | — | — | — | — | — | — | ModelConfig.logprobs_mode |
| `lora_target_modules` | — | — | — | — | — | — | — | None | — | — | — | LoRAConfig.target_modules |
| `mamba_backend` | — | — | — | — | — | — | — | 'triton' | — | — | — | MambaBackendEnum.TRITON |
| `mamba_cache_philox_rounds` | — | — | — | — | — | — | — | 0 | — | — | — | MambaConfig.stochastic_rounding_philox_rounds |
| `max_lora_rank` | — | — | — | — | — | — | — | None | — | — | — | LoRAConfig.max_lora_rank |
| `model_impl` | — | — | — | — | — | — | — | 'auto' | — | — | — | ModelConfig.model_impl |
| `model_loader_extra_config` | — | — | — | — | — | — | — | '{}' | — | — | — | get_field(LoadConfig, 'model_loader_extra_config') |
| `nnodes` | — | — | — | — | — | — | — | 1 | — | — | — | ParallelConfig.nnodes |
| `node_rank` | — | — | — | — | — | — | — | 0 | — | — | — | ParallelConfig.node_rank |
| `offload_group_size` | — | — | — | — | — | — | — | -1 | — | — | — | PrefetchOffloadConfig.offload_group_size |
| `offload_num_in_group` | — | — | — | — | — | — | — | 1 | — | — | — | PrefetchOffloadConfig.offload_num_in_group |
| `offload_prefetch_step` | — | — | — | — | — | — | — | 1 | — | — | — | PrefetchOffloadConfig.offload_prefetch_step |
| `otlp_traces_endpoint` | — | — | — | — | — | — | — | 'localhost:4317' | — | — | — | ObservabilityConfig.otlp_traces_endpoint |
| `quantization` | — | — | — | — | — | — | — | None | — | — | — | ModelConfig.quantization |
| `reasoning_parser` | — | — | — | — | — | — | — | None | — | — | — | StructuredOutputsConfig.reasoning_parser |
| `revision` | — | — | — | — | None | — | — | None | — | — | — | ModelConfig.revision |
| `served_model_name` | — | — | — | — | — | — | — | None | — | — | — | ModelConfig.served_model_name |
| `skip_tokenizer_init` | — | — | — | — | — | — | — | False | — | — | — | ModelConfig.skip_tokenizer_init |
| `stream_interval` | — | — | — | — | — | — | — | 1 | — | — | — | SchedulerConfig.stream_interval |
| `tokenizer_mode` | — | — | — | — | — | — | — | 'auto' | — | — | — | ModelConfig.tokenizer_mode |
| `trust_remote_code` | — | — | — | — | — | — | — | False | — | — | — | ModelConfig.trust_remote_code |

---

> 口径说明：路由集合来自装饰器静态抽取；条件注册（if 分支里 add_api_route）可能漏。  
> OpenAI 字段清单来源：platform.openai.com/docs/api-reference/chat/create, 2026-08 人工录入

