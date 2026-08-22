# MLC-LLM

> **本篇取证基准**：`mlc-llm` @ `9fa644f5`（2026-08-17）
> **一句话**：别家优化"服务"，它优化"能不能跑"

## 0. 结论先行

- **MLC-LLM 走的是编译器路线，这条路线的核心资产是 `python/mlc_llm/support/auto_target.py` 里的一张目标表**：`AUTO_DETECT_DEVICES = ["cuda", "rocm", "metal", "vulkan", "opencl", "cpu"]`（`python/mlc_llm/support/auto_device.py:17`）加上 `PRESET` 字典里的 `iphone:generic` / `macabi:generic` / `android:generic` / `android:adreno` 等条目（`python/mlc_llm/support/auto_target.py:423`起）。同一份用 `nn.Module`（TVM Relax 前端）写的模型定义，靠切换 `target.kind.name` 就能落到 NVIDIA GPU、AMD GPU、Apple Silicon、Android 手机、iPhone、浏览器（WebGPU/WASM）——不用为每个平台重写一遍算子。
- **代价对称地摆在那里**：桌面平台（CUDA/ROCm/Vulkan/OpenCL/CPU）全部共用同一个 `_build_default()`（`python/mlc_llm/support/auto_target.py:310`），真正需要平台特化代码的只有三类打包动作——iOS/macOS Catalyst 走 Xcode 工具链导出 `.dylib`/`.tar`（`_build_iphone`，`python/mlc_llm/support/auto_target.py:180`）、Android 走 NDK 导出 `.so`/`.tar`（`_build_android`/`_build_android_so`，`python/mlc_llm/support/auto_target.py:205`/`228`）、浏览器走 Emscripten 导出 `.wasm` 并静态链接 `mlc_wasm_runtime.bc`（`_build_webgpu`，`python/mlc_llm/support/auto_target.py:251`）。这三类加起来不到 150 行 Python，说明"多平台"这件事的重活被扔给了 TVM 的 codegen 后端，MLC-LLM 自己只写"怎么打包产物"。
- **`_lab/repo_stats.json` 记的"CUDA 文件 0、Triton 文件 0"是真的，但需要一条重要澄清**：仓库里确实没有 `.cu` 文件，但 `python/mlc_llm/op/triton.py:335` 和 `:453` 里各有一处 `triton.jit(triton_kernel)` 调用，用来生成 FP8 分块量化矩阵乘法（`get_tir_w8a8_block_fp8_matmul`，`python/mlc_llm/op/triton.py:271`）——这是不折不扣的**手写 Triton kernel**，只是它是以 `triton.jit(fn)` 函数调用的形式嵌进 `T.call_kernel(...)`（TVM TIR 的调用点），而不是标准的 `@triton.jit` 装饰器写法。`_lab/repo_stats.py`（`TRITON_MARK` 定义处） 的判定标记 `TRITON_MARK = "@triton.jit"` 只认装饰器语法，这一处漏计了——**这是抽取脚本的假阴性，不是仓库里真没有**。详见 `## 7`。
- **`## 4` 会给出的另一条纠偏**：`_lab/api_surface.json` 记的 `cli_flags: 0` 同样不是"配置不走 argparse"，而是抽取脚本的扫描范围只指向 `serve/config.py` + `interface/serve.py` 两个文件（`_lab/api_surface.py`（`ENTRY_HINTS["mlc-llm"]` 登记处）），而这两个文件根本不是 CLI 入口——真正的 argparse 藏在 `cli/*.py` 里，`cli/serve.py` 一家就有 36 处 `add_argument(`。详见 `## 4`。
- **0 个手写 kernel 不等于"CUDA 上什么手工优化都没有"**：`compile` 的默认优化等级 `O2`（`python/mlc_llm/cli/compile.py:96` 默认值）会在 CUDA 且架构 `>= sm_80` 时打开 `flashinfer=True`（`python/mlc_llm/interface/compiler_flags.py:212`），这条开关把 attention 算子外包给 FlashInfer——一个**外部的、手写 CUDA 的算子库**（`python/mlc_llm/op/extern.py` 的 `ExternModuleStore`）。所以准确说法是：**MLC-LLM 自己的仓库里没有手写 kernel，但它默认的 CUDA 编译产物里可能含有别人写的手写 kernel**——纯度不是 100%，但主干算子（matmul/norm/dequant/...)确实全由 TVM 从 Relax IR 编译生成。

**速查：关键问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| 从 HF 模型到可执行产物走哪几步，产物是什么？ | `## 2`、`## 4` |
| `compiler_pass/` 里的 pass 具体做什么？ | `## 2`、`## 4` |
| 一份代码怎么落到 Metal/Vulkan/WebGPU/CUDA？ | `## 0`、`## 3`（见下）、`## 5` 决策 1 |
| serve 层的 13 条路由和调度成熟度？ | `## 4` |
| "0 手写 kernel"到底什么意思？ | `## 0`、`## 6`、`## 7` |
| 编译器路线值不值、什么时候是负担？ | `## 5` |

## 1. 它在系统里的位置

MLC-LLM（Machine Learning Compilation for LLM）是陈天奇团队主导、依托 Apache TVM 生态（准确说是 TVM 的下一代 IR 栈 Relax/TVM Unity）的 LLM 部署工具链。它不是一个"服务框架"，而是一条**从 HuggingFace 权重到平台原生可执行产物的编译流水线**，外加一层围绕这条流水线的最小可用 OpenAI 兼容服务。这个定位在体量上就能看出来：`_lab/repo_stats.json` 记录的 12 个引擎里，MLC-LLM 总行数 100,240，是倒数第二小的（仅比 5 个月没更新的 Tokasaurus 大），Python 产品码 59,516，C/C++ 22,690 行，且 **CUDA 文件数为 0**（`_lab/out/repo_stats.json` → `engines.mlc-llm.kernels.cuda_files`）。对照 vLLM 的 169 个、SGLang 的 246 个、llama.cpp 的 273 个,这一栏本身就是全篇的入口线索。

它在系统里有三个身份，对应仓库里的三条主线：

1. **一个模型编译器**：`mlc_llm compile` 把某个模型定义（`python/mlc_llm/model/`）+ 量化方案（`python/mlc_llm/quantization/`）+ 目标平台（`--device`）编译成一份可执行产物。这是 `## 2`/`## 4` 的主体。
2. **一个推理运行时**：`cpp/serve/` 里 17,101 行 C++（`_lab/out/repo_stats.json` → `top_dirs` 里 `cpp/serve` 排第二）实现了一个连续批处理引擎（`Engine`/`ThreadedEngine`），Python 侧的 `serve/engine.py` 只是它的 FFI 薄封装。
3. **一个 OpenAI 兼容 HTTP 服务**：`mlc_llm serve` 起一个 FastAPI 应用，13 条路由，其中 4 条 `/v1/*`。

`python/mlc_llm/model/` 下有 42 个模型架构子目录(不含 `model.py`/`model_preset.py`/`model_utils.py`/`__init__.py`),按结构大致分四类:

| 类别 | 代表架构 |
|---|---|
| 稠密 Transformer(占大多数) | `llama`/`qwen`/`qwen2`/`qwen3`/`mistral`/`gemma`/`gemma2`/`gemma3`/`phi`/`phi3`/`baichuan`/`internlm`/`internlm2`/`chatglm3`/`cohere`/`nemotron`/`minicpm`/`ministral3`/`olmo`/`olmo2`/`orion`/`stable_lm`/`gpt2`/`gpt_j`/`gpt_neox`/`gpt_bigcode`/`starcoder2` |
| MoE | `mixtral`/`deepseek`/`deepseek_v2`(MLA)/`qwen2_moe`/`qwen3_moe` |
| RNN/状态空间(非标准 KV cache) | `rwkv5`/`rwkv6`(对应 `## 4.1` `_infer_kv_state_kind()` 里 `kv_state_kind="rnn_state"` 的特判) |
| 混合注意力(线性注意力+全注意力交替) | `qwen35`——`python/mlc_llm/model/qwen35/__init__.py:1` 原话"Qwen3.5 GatedDeltaNet hybrid model",对应 `_infer_kv_state_kind()` 里 `kv_state_kind="hybrid"` 的特判 |
| 多模态/视觉 | `llava`/`phi3v`/`qwen2_5_vl`/`vision`/`bert` |
| 仅投机解码用 | `eagle`/`medusa`(`## 4.4` 已展开) |

这张表和 `## 5` 决策 1 提到的"新颖模型结构仍需专门代码"是同一件事的正面例子——每一类结构差异越大(稠密 Transformer vs MoE vs RNN 状态 vs 混合注意力),`nn.Module` 定义里需要区别对待的地方就越多,`_infer_kv_state_kind()`(`python/mlc_llm/interface/compile.py:98`-`105`)对 `rwkv`/`medusa`/`qwen3_5` 三种特判的存在本身就是证据——**KV cache 的组织方式(`## 3.3`)不仅要按硬件分派,还要按模型架构分派,`hybrid`/`rnn_state`/`kv_cache` 三选一发生在编译期,不是运行时自动识别的**。

**本篇范围边界**：不展开 `model/` 目录下具体模型结构（Llama/Qwen/RWKV 等各家 `nn.Module` 定义）、不展开 `quantization/` 每种量化算法的数值细节、不展开投机解码（EAGLE/Medusa）内部对齐逻辑——本篇聚焦"编译流水线怎么走、TVM 在其中扮演什么角色、多平台怎么落地、serve 层做到什么程度"这四件事,以及贯穿全篇的"0 手写 kernel"这条主线。

## 2. 代码地图（文件 → 职责，带行号）

按"CLI 入口 → 编译流水线 → TVM 编译期 pass → 多平台构建 → C++ 运行时 → HTTP 服务"六层组织：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `python/mlc_llm/__main__.py:11` | 顶层 CLI 分发：`compile`/`convert_weight`/`gen_config`/`chat`/`serve`/`package`/`calibrate`/`router` 八个子命令 | — |
| `python/mlc_llm/interface/gen_config.py:89` | `gen_config()`——生成 `mlc-chat-config.json`（编译流水线第 1 步） | 产物写盘：`python/mlc_llm/interface/gen_config.py:289`-`290` |
| `python/mlc_llm/interface/convert_weight.py:214` | `convert_weight()`——权重转换+量化（第 2 步），落盘用 `tvmjs.dump_tensor_cache` | `python/mlc_llm/interface/convert_weight.py:186`-`194` |
| `python/mlc_llm/interface/compile.py:228` | `compile()`——模型编译为可执行产物（第 3 步）,内部调 `_compile()` | `python/mlc_llm/interface/compile.py:108` |
| `python/mlc_llm/interface/compile.py:160`-`166` | `model.quantize[...]` 建量化模型 → `model.export_tvm(...)` 导出到 Relax IRModule | — |
| `python/mlc_llm/interface/compile.py:205`-`223` | 进入 TVM `PassContext`,调用 `args.build_func(mod, args, pipeline=...)` | — |
| `python/mlc_llm/compiler_pass/pipeline.py:81` | `@register_pipeline("mlc_llm")` 装饰的 `_mlc_llm_pipeline`——整条编译流水线的 pass 编排 | Phase 0-5 见 `python/mlc_llm/compiler_pass/pipeline.py:104`-`204` |
| `python/mlc_llm/compiler_pass/fuse_transpose_matmul.py:10` | `FuseTransposeMatmul`——算子融合示例：transpose+matmul 融合成一个 op | — |
| `python/mlc_llm/compiler_pass/blas_dispatch.py:16` | `BLASDispatch`——后端选择示例：CUDA 走 cuBLAS、ROCm 走 hipBLAS，其余 `raise` | `python/mlc_llm/compiler_pass/blas_dispatch.py:20`-`29` |
| `python/mlc_llm/compiler_pass/estimate_memory_usage.py:17` | `AttachMetadataWithMemoryUsage`——内存规划示例：把静态内存估算结果写进模块元数据 | `python/mlc_llm/compiler_pass/estimate_memory_usage.py:40` `_MemoryEstimator` |
| `python/mlc_llm/support/auto_target.py:31` | `detect_target_and_host()`——`--device` 字符串到 `(Target, BuildFunc)` 的解析入口 | — |
| `python/mlc_llm/support/auto_target.py:310` | `_build_default()`——CUDA/ROCm/Vulkan/OpenCL/CPU 共用的构建函数,产出 `.so`/`.dylib`/`.dll` | — |
| `python/mlc_llm/support/auto_target.py:423` | `PRESET` 字典——iPhone/macOS-Catalyst/Android(generic/adreno) 的目标+构建函数映射 | — |
| `python/mlc_llm/op/extern.py:40` | `enable()`——按 `--opt` 等级决定是否外挂 FlashInfer/CUTLASS 手写算子库 | — |
| `python/mlc_llm/serve/config.py:9` | `EngineConfig`——Python 侧引擎配置 dataclass,24 个字段 | `speculative_mode` 默认值见 `:154`,`prefix_cache_mode` 默认 `'radix'` 见 `:157` |
| `cpp/serve/config.h:287` | `EngineConfigNode::prefix_cache_mode` 默认 `PrefixCacheMode::kRadix`——C++ 侧与 Python 默认值一致 | — |
| `cpp/serve/engine.cc:749` | `EngineImpl::Step()`——连续批处理主循环,按序尝试 `actions_` 里每个 `EngineAction` | — |
| `cpp/serve/engine_actions/action_commons.cc:16` | `CreateEngineActions()`——按 speculative/disaggregation 模式组装 Step() 里那条 action 链 | 普通模式:`:113`-`122`（`NewRequestPrefill`→`BatchJumpForward`→`BatchDecode`） |
| `python/mlc_llm/serve/engine_base.py:556` | `MLCEngineBase.__init__`——Python 侧通过 `tvm.get_global_func("mlc.serve.create_threaded_engine")` 拿到 C++ 引擎的 FFI 句柄 | `python/mlc_llm/serve/engine_base.py:613` |
| `python/mlc_llm/interface/serve.py:109`-`125` | FastAPI `app` 组装:`openai_entrypoints`+`metrics_entrypoints`+`microserving_entrypoints` 恒挂载,`debug_entrypoints` 需 `--enable-debug` | — |
| `python/mlc_llm/serve/entrypoints/openai_entrypoints.py:237` | `request_chat_completion`——`/v1/chat/completions` 处理函数 | 其余 3 条 `/v1/*` 见 `:46`(embeddings)/`:121`(models)/`:133`(completions) |
| `python/mlc_llm/serve/entrypoints/microserving_entrypoints.py:23` | `/microserving/prep_recv`——PD 分离的 KV 接收准备接口 | `:49`(remote_send) `:67`(start_generate) |
| `python/mlc_llm/router/router.py:18` | `Router`——PD 分离/多副本负载均衡的协调进程,`router_mode` 二选一 | `:151`(`_pick_endpoint`) `:218`(`_handle_completion_disagg`) |
| `python/mlc_llm/interface/package.py:350` | `package()`——把编译产物打包成移动 App 工程骨架(`compile` 之后的第四步) | `:265`(Android)/`:309`(iPhone)/`:326`(macOS Catalyst) |
| `python/mlc_llm/interface/calibrate.py:131` | `calibrate()`——静态激活值 FP8 量化的校准步骤,需要已编译的 `model_lib` | `:148`-`151`(构造 `AsyncMLCEngine`) |
| `python/mlc_llm/quantization/quantization.py:31` | `QUANTIZATION` 字典——18 个量化方案登记表,`## 3.4` 详细展开 | — |
| `python/mlc_llm/loader/loader.py:9` | `LOADER` 字典——权重源格式登记表,3 个键全部指向 `HuggingFaceLoader` | — |

**必须回原文核对的说明**：以上引用均已用 `grep -n` 在 `_src/mlc-llm` 里核过一次，`## 9` 前会再跑 `_verify.py` 做行号复核。

## 3. 核心数据结构

### 3.1 编译期：`Target` + `BuildFunc` 这对二元组是多平台的枢纽

`python/mlc_llm/support/auto_target.py:28` 定义 `BuildFunc = Callable[[IRModule, "CompileArgs", Pass], None]`——多平台支持在类型层面就是"一个 TVM `Target`（描述目标硬件的 IR）配一个把 `IRModule` 落盘的函数"。`detect_target_and_host()`（`python/mlc_llm/support/auto_target.py:31`）分两条路径产出这对二元组:

- **桌面/服务器设备**(`cuda`/`rocm`/`metal`/`vulkan`/`opencl`/`cpu`,`python/mlc_llm/support/auto_device.py:17`):走 `Target.from_device(hint)` 自动探测,`BuildFunc` 统一是 `_build_default()`。
- **移动/嵌入式/浏览器设备**(`iphone`/`macabi`/`android`/`webgpu`/`mali`/`opencl`,`_detect_target_gpu` 里的字符串重写 `hint += ":generic"`,`python/mlc_llm/support/auto_target.py:84`-`85`):查 `PRESET` 字典拿到手写好的 `target` 字典（比如 iPhone 的 `max_threads_per_block: 256`、`max_shared_memory_per_block: 32768`,针对 Metal 的硬件限制精调）和专属 `BuildFunc`。

这个设计把"给 TVM 什么样的 target 描述"和"怎么把编译结果打包成这个平台能加载的文件格式"彻底分离——前者复用 TVM 已有的后端(LLVM/NVPTX/Metal codegen/SPIR-V/...),后者是 MLC-LLM 自己维护的、按平台数量线性增长的一小撮胶水代码。

### 3.2 `EngineConfig`——Python dataclass 与 C++ struct 的一份配置、两处定义

`python/mlc_llm/serve/config.py:9` 的 `EngineConfig` 是纯 Python `@dataclass`,24 个字段,`asjson()`/`from_json()`(`:162`/`:167`)负责跨语言序列化。`cpp/serve/config.h:236` 的 `EngineConfigNode` 是它在 C++ 侧的镜像,字段名基本一一对应(`prefix_cache_mode`、`speculative_mode`、`prefill_mode` 等)。两边的默认值也保持一致——Python 侧 `prefix_cache_mode` 默认 `'radix'`(`python/mlc_llm/serve/config.py:157`),C++ 侧默认 `PrefixCacheMode::kRadix`(`cpp/serve/config.h:287`)。这不是巧合而是约定:Python 只是把 `EngineConfig.asjson()` 序列化后传给 C++ 的 `EngineConfig::FromJSON`,真正被引擎读取和使用的是 C++ 那份。**这意味着 `EngineConfig` 的"权威定义"其实在 C++,Python 侧的 dataclass 是给用户填写用的镜像**——改配置字段要两边一起改,这也是一个真实的维护成本(见 `## 5`)。

### 3.3 `PagedKVCache`——继承自 TVM 运行时对象,不是引擎自己维护的 Python/C++ 数据结构

`python/mlc_llm/nn/kv_cache.py:13`:

```python
class PagedKVCache(TVMPagedKVCache):
```

它继承自 TVM Relax 前端提供的 `nn.kv_cache.PagedKVCache`(即 `TVMPagedKVCache`),本身只是加了一个 `create_generic()` 工厂方法。真正的分页 KV Cache 读写逻辑(block 分配、attention kernel 怎么读这块内存)是在 `python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py:82` 的 `DispatchKVCacheCreation` pass 里,在**编译期**把一个抽象的 "KV cache creation" Relax 函数,重写成具体的 TIR 分页 KV cache 实现(`create_tir_paged_kv_cache`/`create_flashinfer_paged_kv_cache`,同文件方法列表)。也就是说 MLC-LLM 里"KV cache 怎么组织"这件事本身就是一段**要被编译的代码**,而不是运行时一个独立维护的内存池模块——这与 vLLM/SGLang 用纯 Python/C++ 手写 `BlockManager`/`TokenToKVPoolAllocator` 是完全不同的抽象层级。

### 3.4 `Quantization` 注册表——18 种量化方案、6 个实现类

`python/mlc_llm/quantization/quantization.py:31` 的 `QUANTIZATION: Dict[str, Quantization]` 字典登记了 18 个具名量化方案,背后只有 6 个实现类。逐条列全(方案名→`kind`→关键参数,`python/mlc_llm/quantization/quantization.py:32`-`201`):

| 方案名 | `kind` | 关键参数 |
|---|---|---|
| `q0f16`/`q0bf16`/`q0f32` | `no-quant` | 仅转 `model_dtype`(float16/bfloat16/float32),不量化 |
| `q3f16_0`/`q3f16_1` | `group-quant` | `group_size=40`,`quantize_dtype=int3`,`linear_weight_layout` 分别为 `KN`/`NK` |
| `q4f16_0`/`q4f16_1`/`q4bf16_0`/`q4bf16_1`/`q4f32_1` | `group-quant` | `group_size=32`,`quantize_dtype=int4`,`model_dtype` 分别覆盖 float16/bfloat16/float32,`linear_weight_layout` 分 `KN`/`NK` 两派 |
| `q4f16_2` | `group-quant` | 同 `q4f16_1` 但 `quantize_embedding=False`/`quantize_final_fc=False`——embedding 和输出层不量化 |
| `q4f16_autoawq` | `awq` | `group_size=128`,`quantize_dtype=int4`——group size 明显大于自家 `group-quant` 方案 |
| `q4f16_ft` | `ft-quant` | `storage_dtype=int8`(FasterTransformer 风格,不是 `uint32`) |
| `e5m2_e5m2_f16`/`e4m3_e4m3_f16`/`e4m3_e4m3_f16_max_calibrate` | `per-tensor-quant` | 按 tensor 的 FP8,后两者 `use_scale=True`,`calibration_mode` 分别是 `inference`/`max` |
| `fp8_e4m3fn_bf16_block_scale`/`..._static_activation` | `block-scale-quant` | 分块 FP8,后者 `use_activation_scale=True`——即 `## 4.1` 提到需要 `mlc_llm calibrate` 的那个方案 |

6 个实现类分别是 `NoQuantize`、`GroupQuantize`(`python/mlc_llm/quantization/group_quantization.py:28`)、`AWQQuantize`、`FTQuantize`(`python/mlc_llm/quantization/ft_quantization.py:29`)、`PerTensorQuantize`(`python/mlc_llm/quantization/per_tensor_quantization.py:31`)、`BlockScaleQuantize`。`## 4.1` 的 `mlc_llm convert_weight --quantization q4f16_1` 就是把上表最左列的字符串键传给 `model.quantize[quantization.kind]`(`python/mlc_llm/interface/compile.py:160`)——`kind` 才是内部分发用的字符串,不是方案名本身,一个常见的读代码陷阱是把 `q4f16_1`(方案名,面向用户)和 `group-quant`(`kind`,面向内部分发)当成同一个东西。`GroupQuantize.quantize_weight()`(`python/mlc_llm/quantization/group_quantization.py:28` 起的方法列表)在 `## 4.1` 描述的 `convert_weight` 流程里被逐参数调用,产出的是量化后的整数权重+per-group scale,这些 scale 张量本身也会被反量化融合 pass(`## 4.2` Phase 3 的 `FuseDequantizeMatmulEwise`/`FuseDequantizeTake`)在编译期融合进矩阵乘法算子里,而不是运行时反量化。

### 3.5 `ChatCompletionRequest`——20 个字段,19 个是 OpenAI 标准字段,只有 1 个私货

`python/mlc_llm/protocol/openai_api_protocol.py:258` 的 `ChatCompletionRequest` 是 `## 4.3` 那 4 条 `/v1/*` 路由里最核心的一个协议类,`_lab/out/compare.json` 里 `chat_request.mlc-llm` 记录了它的字段构成:`n_fields: 20`,其中 `openai_standard` 命中 19 个(`frequency_penalty`/`logit_bias`/`logprobs`/`max_tokens`/`messages`/`model`/`n`/`presence_penalty`/`response_format`/`seed`/`stop`/`stream`/`stream_options`/`temperature`/`tool_choice`/`tools`/`top_logprobs`/`top_p`/`user`),`engine_specific`(即私货字段)只有 1 个——`debug_config`。`missing_vs_openai` 记录了它缺的 11 个官方字段(`audio`/`function_call`/`functions`/`max_completion_tokens`/`metadata`/`modalities`/`parallel_tool_calls`/`prediction`/`reasoning_effort`/`service_tier`/`store`)。

这个"20 个字段、19 个标准、1 个私货"的构成,和 `00-总览与阅读地图.md` 记录的 vLLM/SGLang(各自 68 个字段,只有 22 个命中官方字段,46 个私货)对照极为鲜明——**MLC-LLM 的 OpenAI 兼容层几乎是"标准子集",不是"标准 + 大量私有扩展"**。这个反差本身也解释了为什么 `## 4.3` 里 `/v1/*` 只有 4 条路由:MLC-LLM 没有走"在 OpenAI 协议上叠加自己的调度/多模态/Agent 相关私有字段"这条路,协议层的目标就是"覆盖 OpenAI 标准里最常用的那部分,不多做"——这和它把大部分工程投入都花在编译流水线、而不是 API 表面丰富度上的整体定位是一致的。

`_lab/out/api_surface.json` 记录 `openai_api_protocol.py` 里一共定义了 26 个协议类(`n_protocol_classes: 26`),按用途可以分四组(`python/mlc_llm/protocol/openai_api_protocol.py:25`-`436` 范围内,`## 2` 已引用其中最核心的两个):

| 分组 | 成员 | 字段数量级 |
|---|---|---|
| Completion 家族 | `CompletionRequest`(`:144`)/`CompletionResponse`(`:212`)/`CompletionResponseChoice`(`:205`)/`CompletionLogProbs`(`:47`)/`CompletionUsage`(`:56`) | 请求 20 字段,其余多为 2-6 字段的响应壳 |
| Chat 家族 | `ChatCompletionRequest`(`:258`)/`ChatCompletionResponse`(`:422`)/`ChatCompletionStreamResponse`(`:436`)/`ChatCompletionMessage`(`:250`)/`ChatTool`/`ChatFunction`/`ChatFunctionCall`/`ChatToolCall`(`:228`-`244`) | 请求同样 20 字段,工具调用相关的四个类各自 2-3 字段 |
| Embedding 家族 | `EmbeddingRequest`(`:72`)/`EmbeddingResponse`(`:107`)/`EmbeddingObject`(`:96`)/`EmbeddingUsage`(`:102`) | 请求 5 字段,响应壳 2-4 字段 |
| 共享基础类型 | `LogProbs`(`:43`)/`LogProbsContent`(`:36`)/`TopLogProbs`(`:30`)/`StreamOptions`(`:65`)/`ListResponse`(`:25`)/`ModelResponse`(`:121`)/`RequestResponseFormat`(`:135`) | 1-5 字段,被前三组复用 |

这个分组本身印证了 `/v1/*` 只有 4 条路由(`## 4.3`)对应的协议表面确实"小而完整"——26 个类里超过一半是字段数个位数的响应壳/共享类型,真正承担用户输入复杂度的只有 `CompletionRequest`/`ChatCompletionRequest`/`EmbeddingRequest` 三个请求类。

## 4. 主流程走读

### 4.1 编译流水线:三步产出可执行产物

MLC-LLM 把"部署一个模型"拆成三条独立的 CLI 子命令,对应 `interface/` 下三个入口函数:

1. **`mlc_llm gen_config`**(`python/mlc_llm/interface/gen_config.py:89`)——读 HuggingFace 的 `config.json`,结合 `--quantization`,产出 `mlc-chat-config.json`(写盘动作在 `python/mlc_llm/interface/gen_config.py:289`-`290`)。这份 JSON 是后两步和运行时共同的配置源。
2. **`mlc_llm convert_weight`**(`python/mlc_llm/interface/convert_weight.py:214`)——用 `LOADER[source_format]`(`loader/`)读原始权重(HF safetensors/PyTorch bin),按 `Quantization.quantize_weight` 逐参数量化,最后用 `tvmjs.dump_tensor_cache(...)`(`python/mlc_llm/interface/convert_weight.py:186`-`194`)写出分片的权重文件 + `ndarray-cache.json` 元数据——这是 TVM 生态自己的权重存储格式,不是 `.safetensors`。
3. **`mlc_llm compile`**(`python/mlc_llm/interface/compile.py:228`)——这是"编译"真正发生的地方:
   - `model.quantize[quantization.kind](model_config, quantization)`(`python/mlc_llm/interface/compile.py:160`)构造带量化算子的模型定义;
   - `model.export_tvm(spec=model.get_default_spec(), ...)`(`python/mlc_llm/interface/compile.py:163`)把 `nn.Module` 形式的模型**导出为 TVM Relax `IRModule`**——这一步是从"PyTorch 风格的模型定义"跨到"TVM 编译器能处理的 IR"的关键分界线;
   - `args.build_func(mod, args, pipeline=relax.get_pipeline("mlc_llm", ...))`(`python/mlc_llm/interface/compile.py:206`-`223`)把 IRModule 交给 `## 3.1` 里选定的 `BuildFunc`,跑完整条 `_mlc_llm_pipeline`(`python/mlc_llm/compiler_pass/pipeline.py:81`)后导出成目标平台的原生文件。

产物格式由目标平台决定,不是单一的 `.so`:桌面(CUDA/ROCm/Vulkan/OpenCL/CPU)是 `.so`/`.dylib`/`.dll`(`_build_default()`,`python/mlc_llm/support/auto_target.py:310`-`330`);iPhone 是 `.dylib`(`_build_metal_x86_64`)或 `.tar`(`_build_iphone`,系统库形式,`python/mlc_llm/support/auto_target.py:191`);Android 是 `.tar`(系统库,`_build_android`,`python/mlc_llm/support/auto_target.py:209`)或 `.so`(非系统库,`_build_android_so`,`python/mlc_llm/support/auto_target.py:232`);浏览器是 `.wasm`(`_build_webgpu`,`python/mlc_llm/support/auto_target.py:255`,需要静态链接 `mlc_wasm_runtime.bc`)。

`convert_weight` 支持的原始权重格式相当窄:`python/mlc_llm/loader/loader.py:9` 的 `LOADER` 字典只登记了三个键(`huggingface-torch`/`huggingface-safetensor`/`awq`),背后全部指向同一个 `HuggingFaceLoader` 类(`loader/huggingface_loader.py`)——**MLC-LLM 只认 HuggingFace 生态的权重格式,不像 llama.cpp 有自己独立的 GGUF 转换生态**。

对 `## 3.4` 提到的静态激活值 FP8 方案(`fp8_e4m3fn_bf16_block_scale_static_activation`),流水线还多一步:`mlc_llm calibrate`(`python/mlc_llm/interface/calibrate.py:131` 的 `calibrate()`)。这一步和前三步不是纯线性的——它的函数签名直接接收 `model_lib`(`python/mlc_llm/interface/calibrate.py:134`),内部构造一个真正的 `AsyncMLCEngine`(`python/mlc_llm/interface/calibrate.py:148`-`151`)跑一批校准 prompt,靠 `CalibrationObserver`(`python/mlc_llm/interface/calibrate.py:18`,单例模式收集各张量的激活值范围)观察运行时激活值分布,再把结果写回给下一次 `convert_weight`/`compile` 使用。**这意味着静态激活值量化不是一次编译就能搞定,而是需要先有一份能跑的编译产物,用它采集统计量,再重新走一遍量化+编译**——`## 5` 决策 1 提到的"编译期本身进入部署流程"这条负担,在这条量化方案上被放大成了两轮。

### 4.2 编译期 pass:Phase 0-5 把 Relax IR 一路降到可执行代码

`_mlc_llm_pipeline`(`python/mlc_llm/compiler_pass/pipeline.py:102`-`207`)是一条用 `tvm.transform.Sequential` 串起来的 pass 链,分五个阶段(注释直接写在代码里,`python/mlc_llm/compiler_pass/pipeline.py:106`/`121`/`134`/`144`/`151`/`168`):

- **Phase 0**(`python/mlc_llm/compiler_pass/pipeline.py:107`-`120`):挂上 KV cache 创建(`DispatchKVCacheCreation`)、softmax 温度处理、变量边界、CUDA Graph 符号捕获提示、流水并行 stage 信息、logit 处理、采样函数等——本质是把"这个模型该怎么跑"的元信息全部挂到 IRModule 的属性上。
- **Phase 1**(`python/mlc_llm/compiler_pass/pipeline.py:122`-`133`,"Relax 图级优化"):`DispatchTritonKernel`(把 `mlc.triton.*` 的占位调用替换成真正的 Triton kernel,见 `## 0`)、`FuseFTDequantizeEpilogue`、`FuseDequantizeTranspose`、条件性的 `BLASDispatch`(仅 `cublas_gemm=True` 时)、`FuseAddRMSNorm`(仅非 `llvm` target,即非纯 CPU)、`FuseTransposeMatmul`。
- **Phase 2**(`python/mlc_llm/compiler_pass/pipeline.py:135`-`143`,"降到 TIR"):走 TVM Relax 官方的"zero pipeline"——`DispatchSampling`/`DispatchSortScan`/`LegalizeOps`/`AnnotateTIROpPattern`/`FoldConstant`/`FuseOps`/`FuseTIR`,这几个是 TVM 本体提供的通用 pass,不是 MLC-LLM 自己写的。
- **Phase 3**(`python/mlc_llm/compiler_pass/pipeline.py:145`-`150`,TIR 级优化):`FuseDequantizeMatmulEwise`、`FuseDequantizeTake`、死代码消除。
- **Phase 4**(`python/mlc_llm/compiler_pass/pipeline.py:151`-`167`,"Dlight 低级优化"):`LowBatchGemvSpecialize`,再按 `target.kind.name` 分叉——非 `llvm`(即 GPU 系目标)套 `dl.gpu.Matmul()`/`GEMV()`/`Reduction()`/`GeneralReduction()`/`Fallback()` 一组 schedule 模板,`llvm`(CPU)只套 `dl.cpu.GEMV()`(`python/mlc_llm/compiler_pass/pipeline.py:154`-`166`)——**这是"dispatch"这个词在 MLC 里最字面的体现:同一段高层算子描述,按目标硬件种类套不同的底层调度模板,而不是走两套完全独立的代码路径**。
- **Phase 5**(`python/mlc_llm/compiler_pass/pipeline.py:168`-`203`,"降到 VM bytecode"):内存规划的核心在这里——`tvm.relax.transform.StaticPlanBlockMemory()`(`python/mlc_llm/compiler_pass/pipeline.py:190`)做静态内存复用分析,紧跟着 `AttachMetadataWithMemoryUsage(metadata)`(`python/mlc_llm/compiler_pass/pipeline.py:191`,即 `## 2` 提到的 `_MemoryEstimator`)把估算结果写回模块;再往后是 CUDA Graph 重写(`RewriteCUDAGraph`,`python/mlc_llm/compiler_pass/pipeline.py:193`)、张量分配下沉、VM 形状下沉等一整套把 Relax 计算图变成可以被 TVM Relax VM 直接跑的字节码的收尾工作。

`compiler_pass/` 目录一共 22 个 pass 文件(不含 `__init__.py`,`## 4.2` 上面五条 Phase 只挑了有代表性的几个展开;完整名单——每个都是 MLC-LLM 自己写的、不属于 TVM 本体——按在 `pipeline.py` 里出现的顺序列全:

| Pass 类 | 文件:行 | 所属 Phase |
|---|---|---|
| `DispatchKVCacheCreation` | `python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py:82` | 0 |
| `AttachSoftmaxWithTemperature` | `python/mlc_llm/compiler_pass/attach_softmax_with_temperature.py:15` | 0 |
| `AttachVariableBounds` | `python/mlc_llm/compiler_pass/attach_support_info.py:13` | 0 |
| `AttachCUDAGraphSymbolicCaptureHints` | `python/mlc_llm/compiler_pass/attach_support_info.py:58` | 0 |
| `AttachPipelineParallelStages` | `python/mlc_llm/compiler_pass/attach_support_info.py:79` | 0 |
| `AttachLogitProcessFunc` | `python/mlc_llm/compiler_pass/attach_logit_processor.py:14` | 0 |
| `AttachAdditionalPrimFuncs` | `python/mlc_llm/compiler_pass/attach_support_info.py:32` | 0 |
| `AttachAllocEmbeddingTensorFunc` | `python/mlc_llm/compiler_pass/attach_embedding_allocator.py:10` | 0 |
| `AttachGPUSamplingFunc` | `python/mlc_llm/compiler_pass/attach_sampler.py:15` | 0 |
| `AttachSpecDecodeAuxFuncs` | `python/mlc_llm/compiler_pass/attach_spec_decode_aux_funcs.py:10` | 0 |
| `AttachMemoryPlanAttr` | `python/mlc_llm/compiler_pass/attach_support_info.py:46` | 0 |
| `AttachSequenceLengthPaddingFactor` | `python/mlc_llm/compiler_pass/attach_support_info.py:108` | 0 |
| `DispatchTritonKernel` | `python/mlc_llm/compiler_pass/dispatch_triton_kernel.py:160` | 1 |
| `FuseFTDequantizeEpilogue` | `python/mlc_llm/compiler_pass/fuse_ft_dequantize_matmul_epilogue.py:13` | 1 |
| `FuseDequantizeTranspose` | `python/mlc_llm/compiler_pass/fuse_dequantize_transpose.py:11` | 1 |
| `BLASDispatch` | `python/mlc_llm/compiler_pass/blas_dispatch.py:16` | 1(条件) |
| `FuseAddRMSNorm` | `python/mlc_llm/compiler_pass/fuse_add_norm.py:150` | 1(条件) |
| `FuseTransposeMatmul` | `python/mlc_llm/compiler_pass/fuse_transpose_matmul.py:10` | 1 |
| `FuseDequantizeMatmulEwise` | `python/mlc_llm/compiler_pass/fuse_dequantize_matmul_ewise.py:9` | 3 |
| `FuseDequantizeTake` | `python/mlc_llm/compiler_pass/fuse_dequantize_take.py:15` | 3 |
| `CleanUpTIRAttrs` | `python/mlc_llm/compiler_pass/clean_up_tir_attrs.py:10` | 3 |
| `LowBatchGemvSpecialize` | `python/mlc_llm/compiler_pass/low_batch_specialization.py:11` | 4 |
| `LiftTIRGlobalBufferAlloc` | `python/mlc_llm/compiler_pass/lift_global_buffer_alloc.py:13` | 5 |
| `ScatterTupleGetItem` | `python/mlc_llm/compiler_pass/scatter_tuple_get_item.py:14` | 5 |
| `PipelineParallelRewrite` | `python/mlc_llm/compiler_pass/pipeline_parallel_rewrite.py:12` | 5 |
| `AttachCUDAGraphAllocInitFunc` | `python/mlc_llm/compiler_pass/attach_cuda_graph_alloc_init_func.py:8` | 5 |
| `AttachMetadataWithMemoryUsage` | `python/mlc_llm/compiler_pass/estimate_memory_usage.py:17` | 5 |

这张表本身也是"编译器路线的代价"的一个直观证据:22 个自定义 pass,大多数是单一职责的小类(`attach_support_info.py` 一个文件里就塞了 6 个,平均每个不到 30 行),阅读顺序还必须跟着 `pipeline.py` 的 `Sequential` 列表走——这正是 `## 5` 决策 1 里"调试要懂 TVM 整套概念栈"这条负担的具体密度。

### 4.3 serve 层:13 条路由 + C++ 连续批处理引擎

`mlc_llm serve` 启动的 FastAPI 应用(`python/mlc_llm/interface/serve.py:109`-`131`)恒挂三个路由组、按 `--enable-debug` 再挂一个:

| 路由前缀 | 路由数 | 用途 |
|---|---|---|
| `/v1/*` | 4 | `embeddings`/`models`/`completions`/`chat/completions`——OpenAI 兼容主干 |
| `/microserving/*` | 3 | `prep_recv`/`remote_send`/`start_generate`——PD 分离的 KV 传输协议 |
| `/debug/*` | 5 | 仅 `--enable-debug` 时挂载:dump 事件轨迹、CUDA profiler 开关、dump 引擎指标、重置引擎统计 |
| `/metrics` | 1 | Prometheus 风格指标 |

`/v1/*` 只有 4 条,比 vLLM/SGLang 常见的十几条(还有 `/tokenize`、`/v1/responses`、`/v1/messages` 等)窄得多——**MLC-LLM 没有做 Anthropic Messages 兼容,也没有独立的 `/tokenize` 端点**(未查证是否在其它模块里存在同名但未被抽取到的路由;以 `_lab/api_surface.json` 记录的 `openai_entrypoints.py`+`**/entrypoints/*.py` 扫描范围为准)。

真正值得注意的是 `/microserving/*` 这三条:它们不是给终端用户用的,而是**PD(Prefill/Decode)分离场景下,协调器节点用来指挥 prefill 节点和 decode 节点互传 KV 数据的控制面**。`prep_recv`(`python/mlc_llm/serve/entrypoints/microserving_entrypoints.py:23`)让接收方在前缀缓存里匹配已有前缀、为剩余 KV 分配好接收位置;`remote_send`(`:49`)让发送方计算指定窗口的 KV 并推送到目标;`start_generate`(`:67`)让接收方在指定窗口内 prefill 并开始 decode。三个接口内部都是复用同一个 `request_completion`(`openai_entrypoints.py`)加不同的 `DisaggConfig`(`debug_protocol.py`)标记完成的,C++ 侧对应 `cpp/serve/engine_actions/disagg_prepare_recv.cc`(447 行)和 `disagg_remote_send.cc`(503 行)两个 `EngineAction`。

**调度层面**:C++ 的 `EngineImpl::Step()`(`cpp/serve/engine.cc:749`)是标准的连续批处理(continuous batching)主循环——每次调用按固定顺序遍历 `actions_` 列表,某个 action 的 `Step()` 返回非空(即处理了至少一批请求)就立刻收尾返回,不会在一次 `Step()` 里跑多个 action。普通(非投机、非 PD 分离)模式下这条链是 `NewRequestPrefill → BatchJumpForward → BatchDecode`(`cpp/serve/engine_actions/action_commons.cc:113`-`122`)——新请求优先 prefill,然后处理跳跃前向(jump-forward,用于约束解码提前吐出确定 token),最后批量 decode。这个"每步优先吃新请求"的策略和 vLLM/SGLang 的连续批处理是同一类设计。**前缀缓存**方面,`cpp/serve/prefix_cache.cc`(442 行)+ `cpp/serve/radix_tree.cc`(845 行)实现了一棵基数树(radix tree),对应 `EngineConfig.prefix_cache_mode` 默认值 `'radix'`(`python/mlc_llm/serve/config.py:157`,C++ 侧 `cpp/serve/config.h:287`)——**前缀缓存不是可选功能,是默认打开的**。

### 4.4 分布式并行度与投机解码的草稿配对:同样是编译期决定,不是启动参数

`## 4.1`/`## 4.2` 讲的"KV cache 怎么组织"是编译期决定的一个例子,但不是唯一一个。张量并行度(`tensor_parallel_shards`)和流水并行阶段数(`pipeline_parallel_stages`)是另外两个:

- **张量并行**:`python/mlc_llm/interface/compile.py:70`-`81` 的 `_apply_preproc_to_params_and_check_pipeline()` 在编译期就调用 `shard_strategy.gen_shard_info(shards=model_config.tensor_parallel_shards, weight=param)`(`python/mlc_llm/interface/compile.py:72`-`75`),对应实现在 `python/mlc_llm/support/tensor_parallel.py:12` 的 `ShardSingleDim`。也就是说**每个参数该怎么按 TP 度切分,是在编译产物里就写死的元信息**,不是运行时加载权重时才决定的切分方式。
- **流水并行**:`python/mlc_llm/compiler_pass/pipeline_parallel_rewrite.py:25` 的 `_PipelineParallelRewriter`(继承 `PyExprMutator`,289 行)在编译期把整张计算图按 `pipeline_parallel_stages` 重写成多个 stage 子函数(`_create_stage_func`,同文件方法列表),每个参数还带着 `pipeline_stages` 属性标记它属于哪个 stage(`python/mlc_llm/interface/compile.py:90`-`94`)。
- **投机解码**:`python/mlc_llm/model/eagle/eagle_model.py:76` 的 `EagleForCausalLM` 和 `python/mlc_llm/model/medusa/medusa_model.py:44` 的 `MedusaModel` 都是独立的 `nn.Module`,需要在**编译时**和主模型一起被指定(`compile` 需要知道要不要为草稿模型也生成一份编译产物),对应 C++ 侧 `EngineConfig.speculative_mode`/`spec_draft_length`(`python/mlc_llm/serve/config.py:154`-`156`)决定 `CreateEngineActions()`(`## 2`)组装出哪一条 action 链(`EagleNewRequestPrefill`+`EagleBatchDraft`+`EagleBatchVerify`,或 `NewRequestPrefill`+`BatchDraft`+`BatchVerify`,`cpp/serve/engine_actions/action_commons.cc:34`-`88`)。

这条线索连起来看,是"编译器路线倾向于把尽可能多的运行时决策挪到编译期"这个大主题的第三个具体样本(前两个是 `## 5` 决策 1 的目标硬件、决策 2 的 KV cache)——**在 vLLM/SGLang 里,`--tensor-parallel-size` 是一个启动参数,同一份 HF checkpoint 换个 TP 度直接重启服务就行;在 MLC-LLM 里,换 TP 度等于换了一份编译产物,要回到 `mlc_llm compile` 重新走一遍**(本库推断,依据是 TP 切分信息在编译期就已经烘焙进导出的 IRModule 元数据里,`python/mlc_llm/interface/compile.py:198` 的 `metadata["params"]` 记录了每个参数的 `preprocs`/`shard_strategy` 信息)。`## 5` 决策 4 展开这条取舍。

### 4.5 `mlc_llm router`:把 `/microserving/*` 三条路由串成一条完整的 PD 分离流程

`## 4.3` 已经列出 `/microserving/prep_recv`/`remote_send`/`start_generate` 这三条路由,但没说清楚**谁来调用它们、按什么顺序调用**——答案是 `python/mlc_llm/router/router.py:18` 的 `Router` 类,它是 `mlc_llm router` 子命令(`cli/router.py`→`python/mlc_llm/interface/router.py:16` 的 `serve()`)启动的一个独立协调进程,前面挂了 N 个用 `Popen` 拉起的 `mlc_llm serve` 端点(`python/mlc_llm/router/router.py:33` 附近的注释"Spawn len(host_list) server endpoints with Popen")。`Router.router_mode` 有两个取值(`python/mlc_llm/router/router.py:29`,`Literal["disagg", "round-robin"]`,`Router.__init__` 默认 `"disagg"`,CLI `--router-mode` 默认同为 `"disagg"`,`python/mlc_llm/cli/router.py:31`;但 `python/mlc_llm/interface/router.py:26` 的 `serve()` 函数签名自己写的默认值是 `"round-robin"`——这个函数级默认值在 CLI 路径下永远不会被用到,因为 `python/mlc_llm/cli/router.py:88` 总是显式传参,是一处无害但容易读串的默认值不一致)。

`router_mode="disagg"` 下,`_handle_completion_disagg()`(`python/mlc_llm/router/router.py:218`)按官方三步协议依次调用:`send_prepare_receive()`(`python/mlc_llm/router/router.py:316`-`326`,POST 到 `/microserving/prep_recv`,docstring 原话"Performs step 1 of disaggregated serving: ask D to prepare metadata")→`send_remote_send()`(`python/mlc_llm/router/router.py:338`-`350`,"step 2: ask P to prefill and transfer KV to D")→`send_start_generate()`(`python/mlc_llm/router/router.py:357`-`368`,"step 3: ask D to decode and return normal response")。**这三个方法名里的 P/D 就是 Prefill/Decode**——`Router` 在多个 `mlc_llm serve` 进程之间选出一个当 P、一个当 D(`_pick_endpoint()`,`python/mlc_llm/router/router.py:151`),把原本一条完整的补全请求拆成"P 上 prefill 并把 KV 送到 D"+"D 上 decode 并把结果流式返回给客户端"两段,`pd_balance_factor` 参数(`python/mlc_llm/router/router.py:30`)则用于在多副本之间做负载权衡。**这意味着 PD 分离在 MLC-LLM 里不是靠某个引擎内部开关打开的,而是靠额外起一层 `Router` 进程、由它去调度多个普通 `mlc_llm serve` 进程实现的**——和 `## 6` 提到的 TensorRT-LLM 用独立的 `OpenAIDisaggServer` 网关进程做同一件事,是同一种"PD 分离=独立协调进程,不是引擎内部状态机分支"的架构选择。

### 4.6 `mlc_llm package`:把编译产物打包成可安装的移动 App,是 `compile` 之后的第四步

`## 4.1` 的三步流水线产出的是模型库文件(`.so`/`.dylib`/`.tar`/`.wasm`)加权重分片——对桌面/服务器场景这已经是终点,但对 iOS/Android 场景还差一步:把这份产物和一个真正能被 Xcode/Android Studio 编译成安装包的项目骨架拼在一起。这一步是 `python/mlc_llm/interface/package.py:350` 的 `package()` 函数,`SUPPORTED_DEVICES = ["iphone", "macabi", "android"]`(`python/mlc_llm/interface/package.py:17`)限定了它只对这三种移动/桌面 App 场景生效。核心是三个平台专属的绑定生成函数:`build_android_binding()`(`python/mlc_llm/interface/package.py:265`,把 `android/mlc4j` 下的模板工程复制过去,替换 `build.gradle` 等文件,`python/mlc_llm/interface/package.py:292`-`293`)、`build_iphone_binding()`(`python/mlc_llm/interface/package.py:309`)、`build_macabi_binding()`(`python/mlc_llm/interface/package.py:326`)。这三个函数各自只负责"把已经编译好的模型库塞进对应平台的原生工程模板里",不重新触发任何编译期 pass——**`compile` 负责"能不能跑",`package` 负责"怎么变成用户能安装的东西",两件事在代码里是完全分开的两步,`package` 甚至不需要重新导入 TVM 的编译流水线**(`package.py` 顶部导入列表里没有 `tvm`,只有 `mlc_llm.interface.jit` 和文件系统操作)。这条边界再次印证了 `## 0`/`## 5` 决策 1 的说法:多平台支持里"打包"这部分工作量是和"编译"分离的、随平台数量线性增长的胶水代码,不是编译器本身要解决的问题。

## 5. 设计决策与代价

### 决策 1:整条流水线的终点是"用 TVM 把模型编译成目标平台的原生代码",而不是"启动一个跑在某种运行时里的服务进程"

**为什么这么设计**——`python/mlc_llm/support/auto_device.py:17` 的 `AUTO_DETECT_DEVICES` 只有 6 个桌面设备类型,但真正决定"能不能支持一个新硬件"的不是这张表,而是 TVM 有没有针对它的 codegen 后端(LLVM/NVPTX/AMDGPU/Metal/SPIR-V/...)。只要 TVM 有,MLC-LLM 只需要在 `## 3.1` 的 `PRESET` 字典或 `AUTO_DETECT_DEVICES` 里加一条目标描述+一个打包函数(`_build_xxx()`,平均几十行),不需要为这个新平台重写 attention/matmul/norm 等任何一个算子——`python/mlc_llm/compiler_pass/pipeline.py:154`-`166` 那段 `dl.gpu.*` vs `dl.cpu.GEMV()` 分叉,分叉的是"用哪套调度模板生成代码",分叉点数量是 O(硬件大类),不是 O(算子数 × 硬件大类)。

**不这样会怎样**——对照 llama.cpp:它的 `ggml` 后端库是手写路线的典型代表,`_lab/repo_stats.json` 记录它有 273 个 CUDA 文件(`engines.llama.cpp.kernels.cuda_files`),这些文件对应的是每个算子在每种硬件上分别手写一份实现(`ggml-cuda/`、`ggml-metal/`、`ggml-vulkan/` 等各自独立的算子库)。新增一个硬件平台意味着新增一整套 kernel 实现和调优工作;新增一个算子(比如新的 attention 变体)意味着要在已支持的每个平台上都补一份。这条路线的好处是每份实现都能针对具体硬件精调到极致(见 `## 6`),坏处是工作量随"算子数 × 平台数"增长。

**什么时候这是负担**——四条,都有具体证据支撑,不是空泛的"编译慢":①**编译期本身进入了部署流程**——`gen_config`→`convert_weight`→`compile` 三步(`## 4.1`)在换模型、换量化方案、换目标平台的任意一个维度变化时都要重新走一遍,不像 vLLM/SGLang 那样直接加载 HF checkpoint 就能跑;②**编译器生成的 kernel 不一定打得过手工调优的**——最直接的证据是 MLC-LLM 自己在默认 `O2` 优化等级、CUDA 架构 `>= sm_80` 时会打开 `flashinfer=True`(`python/mlc_llm/interface/compiler_flags.py:212`,`## 0` 已展开),把 attention 计算外包给一个手写 CUDA 库,而不是相信 TVM/dlight 生成的 kernel 已经够用——**如果编译生成的 kernel 已经足够快,这个默认开关就没有存在的必要**;③**调试要懂 TVM 的整套概念栈**——Relax IR、TIR、dlight schedule、Relax VM bytecode,这比读一段直接调用 PyTorch 算子的 Python 代码门槛高得多,`compiler_pass/pipeline.py` 里 `_DebugDump` 在每个 Phase 后落一份中间 IR(`python/mlc_llm/compiler_pass/pipeline.py:120`/`133`/`143`/`150`/`167`/`192`)这件事本身就说明"不看中间产物就很难排查编译期问题"是团队自己的共识;④**新颖模型结构仍然需要专门代码**——`python/mlc_llm/model/deepseek_v2/deepseek_v2_model.py` 为 DeepSeek-V2 的 MLA(多头潜在注意力)结构写了 873 行专属的 `nn.Module` 定义,`model/` 目录下四十余种架构(`python/mlc_llm/model/` 下 43 个模型子目录,不含 `model.py`/`model_preset.py`/`model_utils.py`/`__init__.py`)每一种都是独立实现——**编译器路线省下的是"为每个平台重写 kernel",省不下"为每种新模型结构重写模型定义"这部分工作量**,这两件事经常被混为一谈。

### 决策 2:KV cache 的组织方式是编译期 pass 的产物,不是运行时可插拔的独立模块

**为什么这么设计**——`python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py:82` 的 `DispatchKVCacheCreation` 在编译期把一个抽象的"创建 KV cache"的 Relax 函数,重写成具体的分页 KV cache 实现(`create_tir_paged_kv_cache`/`create_flashinfer_paged_kv_cache`)。这样做的好处是 KV cache 相关的内存分配能被后续 `StaticPlanBlockMemory`(`python/mlc_llm/compiler_pass/pipeline.py:190`)等内存规划 pass 一起纳入统一的静态分析,产出的代码里没有"运行时判断该用哪种 KV cache 实现"这种分支,执行路径更精简。

**不这样会怎样**——像 vLLM(`BlockPool`)、SGLang(`RadixCache`/`TokenToKVPoolAllocator`)那样把 KV cache 做成一个独立的 Python/C++ 运行时类,可以在服务已经启动、模型已经加载完的情况下调整策略(比如实验性地改 page size、换一种驱逐算法),不需要停机重新编译。

**什么时候可以不这样**——当你需要"不重新编译产物就切换 KV cache 策略"这种运行时灵活性时,编译期方案做不到:任何这类改动都要退回 `mlc_llm compile` 重新走一遍流水线,产出一份新的 `.so`/`.tar`/`.wasm`。这也是"编译器路线牺牲运行时灵活性换执行效率"这条通用取舍在 KV cache 这一个子系统上的具体体现。

### 决策 3:`EngineConfig` 在 Python 和 C++ 里各定义一份

**为什么这么设计**——C++ 侧的 `EngineConfigNode`(`cpp/serve/config.h:236`)是运行时真正读取的配置,不需要经过 Python 解释器;Python 侧的 `EngineConfig` dataclass(`python/mlc_llm/serve/config.py:9`)给终端用户一个类型友好、带 IDE 补全和文档字符串(`python/mlc_llm/serve/config.py:10`-`30`)的构造接口,通过 `asjson()`(`python/mlc_llm/serve/config.py:162`)序列化后传给 C++。两边字段名和默认值人工保持一致(`## 3.2` 已用 `prefix_cache_mode` 默认值核对过一次)。

**不这样会怎样**——如果只保留 C++ 一份,Python 用户没有类型提示,只能手写 JSON 字符串构造引擎,出错只能等运行时报错;如果只保留 Python 一份,C++ 运行时每次访问配置字段都要经过 FFI 调用 Python 对象,在一个以性能为核心目标的引擎里这个开销不可接受。

**什么时候可以不这样**——当配置字段数量少、变化不频繁时,人工同步两份定义的维护成本可以接受(当前 24 个字段,规模上明显小于 SGLang 的 `ServerArgs`——476 个字段,或 vLLM 的 `EngineArgs`——233 个字段,`00-总览与阅读地图.md` 已有记录)。但这不是免费的安全:两份定义没有共享的单一事实来源,一旦字段数量随功能扩张(参照 SGLang/vLLM 的量级),人工同步会出现遗漏或默认值漂移——这正是 `## 7` 要展开的一条真实风险。

### 决策 4:张量并行度/流水并行阶段数在编译期确定,不是启动参数

**为什么这么设计**——`## 4.4` 已给出证据链:`python/mlc_llm/interface/compile.py:70`-`81` 在编译期就为每个参数生成分片信息(`ShardSingleDim.gen_shard_info`,`python/mlc_llm/support/tensor_parallel.py:12`),`python/mlc_llm/compiler_pass/pipeline_parallel_rewrite.py:25` 的 `_PipelineParallelRewriter` 在编译期把计算图重写成多个 stage 子函数。把并行切分方式在编译期就确定下来,能让 TVM 的内存规划(`StaticPlanBlockMemory`,`## 4.2` Phase 5)和算子融合 pass 把"这个参数分布在哪几个设备上"也纳入优化范围,产出的执行代码不需要运行时再判断分片布局。

**不这样会怎样**——vLLM/SGLang 的做法是运行时按 `--tensor-parallel-size` 加载权重时动态切分(`ColumnParallelLinear`/`RowParallelLinear` 风格的运行时切分逻辑),同一份 HF checkpoint 换一个 TP 度只需要换一个启动参数,不需要预先准备一份"TP=2 专用"和另一份"TP=4 专用"的产物。

**什么时候可以不这样**——当部署环境的并行拓扑会频繁变化(比如同一份权重要在不同集群规模、不同 GPU 数量的机器上灵活伸缩)时,编译期烧死并行度的代价会被放大:每种 `(tensor_parallel_shards, pipeline_parallel_stages)` 组合都需要一份独立的编译产物和一次独立的 `compile` 调用。这条取舍和决策 1、决策 2 是同一种模式在不同子系统上的重复——**MLC-LLM 的编译器路线,统一的取舍公式是"运行时会变的东西,尽量在编译期就问清楚",换来的是执行路径更精简,付出的是任何一维配置变化都要回到编译这一步**。

## 6. 同位对照

### 三角对比:解释式(vLLM)、编译式但单一硬件商(TensorRT-LLM)、编译式且跨硬件(MLC-LLM)

| | 执行方式 | 目标硬件 | 新硬件接入成本 |
|---|---|---|---|
| vLLM | 默认走 `torch.compile`(V1 `CompilationConfig.mode` 为 `None` 时解析成 `VLLM_COMPILE`=3,`` `vllm:vllm/config/compilation.py:447`-`452` ``),但模型前向本身仍是 Python 逐层调度(`` `vllm:vllm/v1/worker/gpu_model_runner.py:4284` `` 的 `execute_model`) | 只要 PyTorch 后端支持(CUDA/ROCm/XPU/TPU 等),但每个新硬件需要独立适配 attention 后端等关键 kernel | 中——依赖 PyTorch 生态 + 自研 attention 后端,不是"改一个 target 描述就行" |
| TensorRT-LLM | 当前默认也是 **eager PyTorch**,`torch_compile_config` 默认 `None`(文档所述/前作已核,`tensorrt-llm:tensorrt_llm/llmapi/llm_args.py:5407`-`5408`),历史上曾有 AOT TensorRT engine 编译路线但已整体移除(`tensorrt-llm:tensorrt_llm/llmapi/llm.py:391`-`399` 只接受 `pytorch`/`_autodeploy` 两个 backend) | **仅 NVIDIA GPU**——无论是已移除的 TensorRT engine 路线还是当前的 PyTorch 路线,都锁死在 CUDA 生态里 | 高——即使走"编译"路线,面向的也只是 NVIDIA 一家硬件商的工具链,不解决"支持 AMD/Apple/浏览器"这个问题 |
| MLC-LLM(本篇) | **AOT 编译**:`mlc_llm compile` 把 Relax IR 走完 Phase 0-5(`## 4.2`)后导出成目标平台原生代码,`compile()` 本身就是产出可执行产物,不是"启动引擎前顺手编译一下" | CUDA/ROCm/Metal/Vulkan/OpenCL/CPU/Android/iOS/WebGPU,由 TVM 的 codegen 后端广度决定(`## 5` 决策 1) | 低(相对而言)——目标平台层面只需要新增 `Target` 描述+打包函数;但"低"是有前提的,见 `## 5` 决策 1 的负担清单 |

这张表最容易被误读的一点是:**"编译式"和"跨硬件"是两个独立的轴,不能因为一家是编译式就默认它跨硬件**。TensorRT-LLM 曾经的 AOT engine 编译路线,编译目标是"某块具体 NVIDIA GPU 上的 `.engine` 二进制",不是"任意硬件";MLC-LLM 的编译目标是"任意 TVM 支持后端的原生代码"——同样是"提前把模型变成一份可执行产物"这件事,面向的硬件广度可以完全不同。反过来,vLLM 虽然被归类为"解释式",V1 默认也在用 `torch.compile` 做图级优化——"解释式引擎不编译"和"编译式引擎必然跨硬件"这两句常见简化,在这三家的当前快照上都不成立。

### KV cache:编译期烧死 vs 运行时可插拔

MLC-LLM 的 KV cache 实现在编译期确定(`## 5` 决策 2)。vLLM 的 `BlockPool`(见 `00-总览与阅读地图.md` 已记录的 LIFO/FIFO 双路驱逐机制)和 SGLang 的 `RadixCache`/`UnifiedRadixCache` 都是运行时 Python/C++ 对象,可以在服务运行期间被查询、被统计、甚至(某些实现里)被动态调参而不需要重新构建执行图。这不是"MLC-LLM 的 KV cache 更差",而是它继承了编译器路线本身的取舍——一旦选择把尽可能多的决策挪到编译期,KV cache 这个子系统自然也被卷入,不会因为"KV cache 恰好是个例外"就单独留一个运行时接口。

### Attention 内核:三家对"手写 vs 生成"给出的不同答案,MLC-LLM 自己都不是单一答案

vLLM 把 paged attention 的手写 kernel 在 NVIDIA 路径上外包给 FlashInfer/FlashAttention(`00-总览与阅读地图.md` 已记录,`csrc/attention/` 只剩 dtype 头文件);MLC-LLM 默认在 CUDA sm_80+ 上同样外挂 FlashInfer(`## 0`/`## 5` 决策 1),但在其它平台(ROCm/Metal/Vulkan/WebGPU)没有这个选项,只能依赖 TVM/dlight 生成的 attention 实现(`python/mlc_llm/op/attention.py:20` 的 `attention()` 函数,`## 3` 未展开的纯 Relax 算子定义)。**同一个引擎在不同平台上,"attention 是手写的还是编译生成的"这个问题的答案是不一样的**——这条细节经常在"MLC-LLM = 纯编译生成"这种一句话总结里被抹掉。

### 调度决策在哪个语言实现:MLC-LLM 和 TensorRT-LLM 站在同一边,vLLM/SGLang 站在另一边

`00-总览与阅读地图.md` 与 [[01-TensorRT-LLM]] `## 6` 已经指出 vLLM 的调度器是纯 Python(`` `vllm:vllm/v1/core/sched/scheduler.py` `` 的 `schedule()`),SGLang 同理(`` `sglang:python/sglang/srt/managers/scheduler.py` `` 的 `class Scheduler`)。MLC-LLM 的连续批处理决策则和 TensorRT-LLM 一样默认在 **C++**:`cpp/serve/engine.cc:749` 的 `EngineImpl::Step()` 和 `cpp/serve/engine_actions/action_commons.cc:16` 的 `CreateEngineActions()`,Python 侧的 `serve/engine.py` 只是通过 `tvm.get_global_func("mlc.serve.create_threaded_engine")`(`python/mlc_llm/serve/engine_base.py:613`)拿到一个 FFI 句柄,不参与任何一步调度决策(`## 3.1`/`## 4.3` 已展开)。

这个分组方式(MLC-LLM、TensorRT-LLM 把调度放 C++;vLLM、SGLang 把调度放 Python)和"谁走编译器路线、谁走解释器路线"这条轴并不重合——MLC-LLM 是编译器路线里唯一跨硬件的,TensorRT-LLM 现在其实是 eager PyTorch 路线(`## 6` 第一个表格),两者调度层语言选择却一致。这提示了一件容易被"编译式 vs 解释式"这个标签掩盖的事实:**"模型计算怎么跑"(编译 or 解释)和"该给哪些请求跑"(调度决策放哪个语言)是两个独立的架构决策,四家引擎在这两个维度上分别做了不完全相关的选择**,不能用一个标签同时预测两件事。

### 量化的落地方式:编译期融合 vs 运行时 kernel 选择

`## 3.4` 已经展开 MLC-LLM 的量化方案在 `convert_weight` 阶段产出整数权重+scale,`FuseDequantizeMatmulEwise`/`FuseDequantizeTake`(`## 4.2` Phase 3)在**编译期**把反量化计算融合进矩阵乘法算子——产出的可执行文件里,"这个权重是 int4 还是 fp8"这件事已经固化成了具体的算子选择,不存在运行时判断。vLLM/SGLang 的量化路径正相反:两家都支持在同一个进程里通过 `--quantization` 启动参数在多种量化 kernel 之间切换(比如 AWQ/GPTQ/FP8 各自对应一个已编译好、随进程一起加载的 CUDA/Triton kernel),权重格式的选择发生在**加载时**,不需要为每种量化方案单独产出一份不同的可执行程序。这条差异和 `## 5` 决策 1/2/4 是同一个大主题的第四个样本:**MLC-LLM 把量化方案也归入"编译期该问清楚的事情"之列**,换来的是编译产物里没有反量化分支判断的运行时开销,代价是换量化方案等于换编译产物,不是换一个启动参数。

## 7. 踩坑与反直觉

1. **"Triton 文件 0"是抽取脚本的假阴性,仓库里确实有手写 Triton kernel。**`_lab/repo_stats.py`（`TRITON_MARK` 定义处） 的判定标记 `TRITON_MARK = "@triton.jit"` 只匹配装饰器写法,而 `python/mlc_llm/op/triton.py:335`/`453` 里的用法是把普通函数包一层 `triton.jit(triton_kernel)` 再传给 TVM 的 `T.call_kernel(...)`(`python/mlc_llm/op/triton.py:271` 起的 `get_tir_w8a8_block_fp8_matmul`)——语义上仍然是"写了一个 Triton kernel",只是没有用装饰器语法。这条踩坑提醒了一件更通用的事:**"仓库里搜不到某个模式"和"这个仓库没有对应功能"是两回事**,尤其是像本库这种靠正则/AST 关键字匹配做规模统计的场景,匹配规则本身就是需要被审计的对象。
2. **"CLI 开关 0"同样是扫描范围问题,不是"配置不走 argparse"。**`_lab/api_surface.py`（`ENTRY_HINTS["mlc-llm"]` 登记处） 给 mlc-llm 登记的 "config" 扫描文件只有 `serve/config.py`(纯 dataclass,没有 `add_argument`)和 `interface/serve.py`(内部函数,也没有);真正的 argparse 逻辑分散在 `python/mlc_llm/cli/*.py` 十几个文件里,`cli/serve.py` 一家就有 36 处 `add_argument(`,`cli/compile.py` 有 11 处,`cli/gen_config.py` 有 13 处——粗略加总远超"0"。而且这些 CLI 全部通过 `mlc_llm.support.argparse.ArgumentParser`(`python/mlc_llm/support/argparse.py:7`,继承标准库 `argparse.ArgumentParser` 并覆写 `error()` 方法)构造,不是什么自定义配置系统绕开了 argparse。
3. **"0 手写 kernel"配上"默认打开 FlashInfer",两句话放在一起读才完整。**只读 `_lab/repo_stats.json` 会得出"这个引擎的所有 GPU 代码都是编译生成的"这个印象,但 `python/mlc_llm/interface/compiler_flags.py:212`(`O2`/`O3` 默认 `flashinfer=True`)+ `python/mlc_llm/op/extern.py:40`-`53`(按 target 决定是否真正启用)说明:默认编译产物在满足条件的 CUDA 硬件上,attention 部分调用的是 MLC-LLM 仓库之外、FlashInfer 项目手写的 CUDA kernel。这不是自相矛盾,而是"仓库自身 0 手写 kernel"(可验证,`_lab` 数字为证)和"默认编译产物的运行时行为"(需要读 `compiler_flags.py`+`extern.py` 才知道)是两个不同粒度的问题,不能互相替代。
4. **前缀缓存默认是打开的,而且是一棵基数树,不是"编译器工具懒得做服务层功能"。**`python/mlc_llm/serve/config.py:157` 的 `prefix_cache_mode` 默认值是 `'radix'`,C++ 侧 `cpp/serve/config.h:287` 同步默认 `PrefixCacheMode::kRadix`,底层是 `cpp/serve/radix_tree.cc`(845 行)。"编译器路线的引擎在服务层通常比较简陋"是一种常见的先入为主,MLC-LLM 的 `cpp/serve/` 目录本身有 17,101 行(`_lab/out/repo_stats.json` → `top_dirs`),连续批处理+前缀缓存+投机解码+PD 分离一样不少,只是它们都写在 C++ 里,不像 vLLM/SGLang 那样能直接在 Python 里读到。
5. **`--device` 的字符串重写规则容易在读代码时被忽略。**`python/mlc_llm/support/auto_target.py:84`-`85` 里 `if hint in ["iphone", "macabi", "android", "webgpu", "mali", "opencl"]: hint += ":generic"`——用户传 `--device iphone`,内部立刻改写成 `iphone:generic` 去查 `PRESET` 字典;但 `opencl` 同时出现在这条重写列表和 `AUTO_DETECT_DEVICES` 里,意味着 `--device opencl` 和 `--device android`(重写前不含 `opencl`,但 `android:generic`/`android:adreno` 都在用 opencl 作为底层 `Target.kind`)最终走的是同一种 target kind、不同的 `PRESET` 条目——不跟着这条重写逻辑走一遍,容易把"opencl 桌面设备自动探测"和"android 走 opencl 后端"这两条完全不同的路径搞混。
6. **`kv_cache_page_size` 表面上是个可配置字段,实际上只有一个合法值。**`EngineConfig.kv_cache_page_size` 在 Python dataclass 里的类型是普通的 `int`,默认值 `16`(`python/mlc_llm/serve/config.py:145`);但 `python/mlc_llm/serve/engine_base.py:92`-`96` 的 `_check_engine_config()` 会在构造引擎时硬校验:只要这个字段不等于 16 就直接 `raise ValueError`。也就是说这个字段名字长得像一个可调旋钮,实际可用取值集合的大小是 1——这类"类型允许但运行时收窄到单一值"的字段,只看 dataclass 定义会误判成"这是可以自由调整的性能参数"。
7. **`EngineConfig` 里不少字段的"默认值"其实要等到运行时结合硬件信息才能算出来,写 `None` 只是占位。**`max_num_sequence`/`max_total_sequence_length`/`prefill_chunk_size` 在 Python 侧全部是 `Optional[int] = None`(`python/mlc_llm/serve/config.py:146`-`149`);真正的数值是 C++ 侧 `InferEngineConfig()`(`cpp/serve/config.cc` 一系列 `if (mode == EngineMode::kLocal) {...} else if (mode == EngineMode::kInteractive) {...}`分支,例如 `:664`-`670` 那段对 `max_num_sequence` 的推导:`local` 模式取 `min(4, model_max_batch_size)`,`interactive` 模式固定为 `1`,`server` 模式取 `model_max_batch_size`)结合 `mode` 和探测到的 GPU 显存/`gpu_memory_utilization` 算出来的。这和 `00-总览与阅读地图.md` 记录的 vLLM"默认值不同"踩坑是同一类陷阱——**拿 Python 侧的字面量 `None` 去跨引擎比较默认值,或者假设它和 SGLang/vLLM 某个写死的数字直接可比,都会得到误导性结论**;真正的默认值需要跟进 `mode` 参数去 C++ 里查。
8. **`_lab/struct_map.json` 里 `scheduler: 0`、`disagg: 0` 是同一类扫描盲区的第三个例子,而且这次是整个子系统被完全漏掉,不只是漏计几处。**`_lab/struct_map.py`（`PKG_ROOTS["mlc-llm"]` 登记处） 给 mlc-llm 登记的 `PKG_ROOTS["mlc-llm"]` 只有一条:`["python/mlc_llm"]`——扫描器压根不会走进 `cpp/` 目录。但 `## 4.3` 已经证实调度主循环在 `cpp/serve/engine.cc:749` 的 `EngineImpl::Step()`,`## 4.5` 证实 PD 分离核心逻辑在 `cpp/serve/engine_actions/disagg_prepare_recv.cc`(447 行)+ `disagg_remote_send.cc`(503 行)——这些代码是真实存在的,只是全部写在 C++ 里,不在 `python/mlc_llm/` 这棵树下。**这条踩坑和踩坑 1、2 加起来构成一个模式**:凡是靠"只扫某个目录/某几个文件"做统计的工具,对一个 Python+C++ 混合仓库,只要真正的实现落在扫描范围之外的语言/目录里,统计结果就会是"0",而"0"极易被误读成"没有这个功能",不是"我们没扫到"。三条踩坑指向同一条更高层的提醒:**任何一个"0"或"没有"的统计结论,动手前先确认扫描范围覆盖了整个实现,而不是只覆盖了一种语言**。

## 8. 可改进点

1. **修正 `_lab/repo_stats.py` 的 Triton 判定标记**(对应 `## 7` 踩坑 1)——把 `TRITON_MARK = "@triton.jit"` 放宽成同时匹配 `@triton.jit` 装饰器和 `triton.jit(` 函数调用两种形式(比如改成对 `triton\.jit\b` 做正则搜索而不是精确子串匹配),这样 `mlc-llm` 的 `triton_jit_files` 才能反映仓库里真实存在的手写 Triton kernel。这是本库自己工具链能落地的改进,不涉及改 mlc-llm 上游代码。
2. **扩大 `_lab/api_surface.py`（`ENTRY_HINTS["mlc-llm"]` 登记处） 里 mlc-llm 的 `"config"` 扫描范围**(对应 `## 7` 踩坑 2)——把 `python/mlc_llm/cli/*.py` 加进扫描 glob,这样 `cli_flags`/`n_cli_flags` 才能反映这个引擎真实的旋钮数量,而不是恒为 0。当前的登记(`_lab/api_surface.py` 的 `ENTRY_HINTS["mlc-llm"]` 条目)只覆盖了 serve 相关的两个文件,对一个"CLI 驱动的编译工具"这个定位而言,遗漏了最大的一块 CLI 表面。
3. **`interface/compiler_flags.py` 里 `--flashinfer` 的默认行为建议在 `mlc_llm compile --help` 的输出里更显式地提示"这会引入仓库外的手写 CUDA kernel"**(对应 `## 7` 踩坑 3,本库推断)——当前 `O2`/`O3` 默认打开这个开关(`python/mlc_llm/interface/compiler_flags.py:212`/`220`),但用户如果只关心"我要一份纯编译生成的产物"(比如为了审计供应链、确认代码来源),需要读到 `op/extern.py` 才能意识到默认路径不是纯编译生成的。加一行 `--opt` 帮助文本说明 `O2`/`O3` 默认外挂 FlashInfer,能降低这个认知落差。
4. **`python/mlc_llm/support/auto_target.py:84`-`85` 的 `hint += ":generic"` 重写逻辑,建议加一条日志说明改写后的最终 hint**(对应 `## 7` 踩坑 5,本库推断)——当前 `_detect_target_gpu` 对显式命中 `PRESET` 的分支有 `FOUND`/`NOT_FOUND` 日志(`python/mlc_llm/support/auto_target.py:95`-`101` 附近的 `logger.info`),但字符串重写这一步本身没有留痕,用户传 `--device iphone` 看不到日志里出现过 `iphone:generic` 这个中间态,排查"为什么用了这份 target 配置"时不够直观。
5. **`EngineConfig.kv_cache_page_size` 建议改成只读常量或加类型级约束,而不是一个"填了别的数会在构造期报错"的 `int` 字段**(对应 `## 7` 踩坑 6)——`python/mlc_llm/serve/config.py:145` 当前的类型标注 `int = 16` 完全没有传达"这个值目前只能是 16"这条约束,用户在 `## 3.2` 提到的 `asjson()`/`from_json()` 序列化路径上很容易顺手把它当成一个可调参数写进配置文件模板;要么去掉这个字段直接在文档里写死,要么在 dataclass 层面就用 `Literal[16]` 标注类型,让 IDE/类型检查器能在构造前而不是构造时(`_check_engine_config()`)就发现问题。
6. **`cpp/serve/config.cc` 里按 `mode` 推导 `max_num_sequence` 等字段的逻辑,建议在 Python `EngineConfig` 的字段文档字符串里补一句明确指向**(对应 `## 7` 踩坑 7)——当前 `python/mlc_llm/serve/config.py:25`-`30` 的 `mode` 字段文档已经说了"决定 `max_num_sequence` 等字段未显式指定时的取值",但 `max_num_sequence` 自己的字段文档没有反向链接回这句话,读者容易先看到某个字段是 `Optional[int] = None` 就止步,不继续去找"那 None 到底是什么值"的答案在哪。
7. **`_lab/struct_map.py`（`PKG_ROOTS["mlc-llm"]` 登记处） 的 `PKG_ROOTS["mlc-llm"]` 应该加上 `cpp`**(对应 `## 7` 踩坑 8)——当前只登记 `["python/mlc_llm"]`,导致 `scheduler`/`disagg` 两个子系统的统计恒为 0,是本库自己产出的一份误导性数据。把 `PKG_ROOTS["mlc-llm"]` 改成 `["python/mlc_llm", "cpp"]`(参照 `tgi`/`dynamo`/`llama.cpp` 已经登记多个 root 目录的先例,`_lab/struct_map.py` 的 `PKG_ROOTS` 字典),`_classify()`(`_lab/struct_map.py` 的 `_classify()`)按路径关键词分类的逻辑不用改,重跑一遍就能让这两个子系统的行数反映 `cpp/serve/engine.cc`、`cpp/serve/engine_actions/disagg_*.cc` 的真实规模。这是本库自己工具链能落地的改进,且是本篇踩坑 8 里优先级最高的一条——它影响的不是某几行数字,而是两个整个子系统的"有没有"判断。

## 9. 自测题与延伸阅读

**自测题**

1. 从一个 HuggingFace 模型仓库到能被加载执行的产物,要依次跑哪三条 CLI 子命令?每一步各自的核心产物是什么?(提示:`## 4.1`)
2. `_build_default()` 和 `PRESET` 字典里那些平台专属的 `_build_xxx()`,分工边界画在哪里?为什么 CUDA/ROCm/Vulkan/OpenCL/CPU 能共用一个构建函数?(提示:`## 3.1`、`## 2`)
3. `_lab/repo_stats.json` 记的"CUDA 文件 0、Triton 文件 0"分别在什么意义上是"真"、在什么意义上需要澄清?(提示:`## 0`、`## 7` 踩坑 1)
4. 默认 `O2` 优化等级在 CUDA sm_80+ 上会做一件什么事,这件事和"0 手写 kernel"这个说法如何共存?(提示:`## 5` 决策 1、`## 7` 踩坑 3)
5. `mlc_llm serve` 起的 13 条路由分几组挂载?哪一组是可选的、由什么开关控制?`/microserving/*` 三条路由分别对应 PD 分离流程里的哪个环节?(提示:`## 4.3`)
6. `compiler_pass/pipeline.py` 的 Phase 4 里,GPU 目标和 CPU(`llvm`)目标分别套用了哪些 dlight schedule 模板?这处分叉体现了"dispatch"这个词在 MLC-LLM 里的什么含义?(提示:`## 4.2`)
7. 为什么说"TensorRT-LLM 也是编译式,但和 MLC-LLM 不是同一种编译式"?两者编译产物面向的硬件广度分别是什么决定的?(提示:`## 6` 第一个表格)
8. `EngineConfig` 为什么要在 Python 和 C++ 里各留一份定义?这个设计在字段数量增长时会遇到什么风险?(提示:`## 5` 决策 3、`## 7` 踩坑 4 的对照)
9. 张量并行度(`tensor_parallel_shards`)在 MLC-LLM 里是编译期决定还是启动参数?这和 vLLM/SGLang 的 `--tensor-parallel-size` 有什么本质区别?换并行度要付出什么代价?(提示:`## 4.4`、`## 5` 决策 4)
10. `EngineImpl::Step()` 的调度决策用什么语言实现?这个选择和 TensorRT-LLM、vLLM、SGLang 三家分别是同一类还是不同类?"调度语言"和"模型计算是编译还是解释"这两个维度是否总是绑在一起?(提示:`## 6` "调度决策在哪个语言实现"一节)
11. MLC-LLM 的 `ChatCompletionRequest` 有多少个字段?其中多少个是 OpenAI 标准字段、多少个是私货?这和 vLLM/SGLang 的字段构成呈现出怎样相反的比例?(提示:`## 3.5`)
12. `qwen35` 目录对应的模型架构在 `_infer_kv_state_kind()` 里被归到哪一类 `kv_state_kind`?这说明"按硬件分派"和"按模型架构分派"是不是同一件事?(提示:`## 1`、`## 3.3`)

**延伸阅读**

- [[00-总览与阅读地图]]——本库立场、12 个引擎的取证基准表(MLC-LLM 一行的规模数字出处)、以及"跨项目对比里最容易造假数据的是自己的脚本"这条纪念踩坑,和本篇 `## 7` 踩坑 1/2 是同一类教训
- [[01-TensorRT-LLM]]——`## 6` 三角对比的另一角:同样标榜"编译式"但只面向单一硬件商的对照样本
- [[11-开源推理引擎谱系图]]——把 MLC-LLM 放回 12 个引擎的谱系里,看编译器路线在整个生态中的位置

