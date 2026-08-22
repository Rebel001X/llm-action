# vLLM 全景与代码地图

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：一次请求要穿过三个进程、两套模型布局、一场正在进行的语言迁移。

## 0. 结论先行

- **默认路径下，一次 `/v1/chat/completions` 请求至少穿过三个操作系统进程**：HTTP/协议解析进程（front-end）→ 引擎核心进程（EngineCore，独立子进程，跑自己的 `run_busy_loop`）→ 至少一个 GPU worker 进程（`WorkerProc`，每个并行 rank 一个）。这不是"多进程可选"，是 V1 架构的默认拓扑，`## 4` 会逐跳给出文件与行号。
- **这个仓库里同时存在两套模型代码布局**：老的 `vllm/model_executor/models/`（293 个文件，扁平目录，一个模型一个 `.py`）和新的 `vllm/models/`（目前只装 6 个模型家族，但每个家族下面按 `nvidia/` `amd/` `xpu/` 拆成硬件专属子目录）。源码注释直接把老布局称为"legacy"（`vllm/model_executor/models/registry.py:1005`-`1008`）——这是一场肉眼可见、尚未完成的重构，`## 1` `## 5` 会给证据。
- **这不是模型层独有的现象，算子绑定层在做同一件事的第三份拷贝**：`csrc/` 下除了传统的 `attention/`、`moe/`、`quantization/`、`rocm/` 这些扁平子目录，还有一个 `csrc/libtorch_stable/`（184 文件、69,144 行），头文件里明确写着"stable ABI compatibility"（`csrc/libtorch_stable/torch_utils.h:21`）——这是 PyTorch 的稳定 ABI 扩展机制，同一批算子（`cutlass_extensions/`、`attention/` 在两边都各有一份）正在被迁往这套不依赖 PyTorch 内部头文件、跨版本更稳定的新绑定方式。模型布局、算子绑定各自独立走出了"扁平旧目录 + 隔离新目录"这条同构的重构路径。
- **这个仓库正在长出第三种语言栈**：除了 Python（系统装配）和 CUDA/C++（算子），`rust/` 目录下有 11 万行 Rust，实现了一个独立的 HTTP 前端二进制，会话内可以托管（spawn）一个"headless"的 Python 引擎子进程（`rust/src/cmd/src/main.rs:123`-`145`）。但这条路径默认关闭（`vllm/envs.py:165`：`VLLM_USE_RUST_FRONTEND: bool = False`），是 opt-in，不是默认行为——把"正在发生"和"已经发生"混为一谈是本篇要避免的第一个坑。
- **前端进程和 EngineCore 进程之间不是传对象引用，是过一遍手写的二进制协议**：Python 侧用 `msgspec`/`msgpack` 编解码（`vllm/v1/serial_utils.py:136`、`:313`），Rust 侧独立定义了镜像结构体（`rust/src/engine-core-client/src/protocol/handshake.rs:50` 的 `EngineCoreReadyResponse`，对应 Python `vllm/v1/engine/__init__.py:73` 的同名类）——这意味着 Rust 前端能接进来，靠的不是"调用同一份 Python 代码"，而是两边各自实现了同一套线上协议。
- **代码规模不是均匀分布的**：`vllm/model_executor/` 一个目录就占 424,532 行、1339 个文件（据 `_lab/out/repo_stats.json`），超过整个仓库统计口径（6286 文件、1,914,984 行）的五分之一。但"最大"不等于"最复杂"——`## 1` 会拆开看这 42 万行里有多少是"293 个模型文件互相不通用的重复劳动"。
- **测试代码比生产代码少（测试/产品 = 0.542）**——这不是"测试不足"的证据，而是要结合"哪些子系统靠 CI 里的真实 GPU 跑分回归、哪些靠 pytest 断言"分开看，本篇不下结论，留给后续篇目用具体子系统的测试目录核验。

## 1. 它在系统里的位置

本库根目录 `_src/vllm` 是 `https://github.com/vllm-project/vllm.git` 的一次 `--depth 1` 浅克隆，sha 为 `7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6`（`_lab/out/repo_stats.json` 的 `ref` 字段）。这是本库（`01-vLLM/`）后面 11 篇文章的唯一取证来源——`02` 到 `12` 都是从本篇画出的地图里挑一个子系统往下钻。

### 1.1 三层语言栈，量级都不小

仓库的语言构成不是"一个 Python 项目带一点 CUDA"，而是三层叠在一起（据 `_lab/out/repo_stats.json`）：

| 语言 | 行数 | 文件数 | 在系统里的角色 |
|---|---|---|---|
| Python | 1,437,931 | 4294 | 系统装配层：调度、KV 管理、HTTP 协议、模型定义的 Python 胶水部分 |
| JSON | 157,056 | 646 | 主要是模型配置样例、CUDA Graph/量化的调优表 |
| Rust | 109,867 | 309 | 新前端：HTTP 服务、tokenizer、chat 模板渲染、引擎客户端 |
| CUDA | 70,442 | 169 | 手写 GPU kernel（attention、量化、MoE、通信） |
| Markdown | 44,253 | 285 | 文档 |
| C/C++ header | 32,312 | 88 | kernel 头文件、pybind 声明 |
| C++ | 22,847 | 43 | kernel 宿主代码、CPU 后端 |
| Shell | 21,176 | 132 | 构建/CI 脚本 |
| YAML | 17,579 | 289 | CI workflow、benchmark 配置 |

Python 产品码 932,277 行、测试码 505,654 行（测试/产品 = 0.542，同样出自 `repo_stats.json`）——这个比例本身不能单独说明"测试够不够"，因为 vLLM 大量的正确性验证发生在跑真实模型的 CI（跑分对齐、logprob 对齐），不会体现在 pytest 行数里；这一点标注为**本库推断**，未去核对 CI workflow 就不展开。

再看目录规模（`_lab/out/repo_stats.json` 的 `top_dirs`）：`vllm/model_executor` 424,532 行/1339 文件是全仓库最大的单一目录，其次是 `tests/v1`（149,220/411）、`vllm/v1`（148,226/360）、`rust/src`（114,299/358）、`tests/kernels`（93,398/262）、`vllm/models`（77,076/195）、`vllm/kernels`（69,744/33）、`csrc/libtorch_stable`（69,144/184）、`tests/entrypoints`（62,540/260）、`vllm/distributed`（54,754/138）。把 `vllm/v1`（V1 引擎本体）和 `vllm/model_executor`（模型 + 算子层）加起来，572,758 行——超过 Python 产品码的一半都堆在"引擎循环"和"模型执行"这两块，这也是本库把 `02-vLLM-V1架构与EngineCore循环`、`06-vLLM-模型执行与CUDA-Graph` 单独拆成两篇的原因。

按子系统口径（`_lab/out/struct_map.json` 的 `subsystems`，用关键词 + AST 定位而非目录路径，所以和 `top_dirs` 的数字不是同一种切法）：`model_exec` 392,324 行遥遥领先，其后是 `attention` 81,107、`quantization` 56,317、`distributed` 54,851、`entrypoint` 45,972、`disagg`（PD 分离/KV 传输）33,900、`speculative` 18,360、`kv_cache` 9,581、`scheduler` 8,424、`structured_output` 2,808。**调度器只有 8,424 行，是十个子系统里倒数第二小的**——这和"调度器是 vLLM 的核心创新"这种常见印象是反的：真正堆代码量的不是调度算法本身，是"要在 N 个硬件后端上分别实现同一个算子""要接住新模型的各种变体"这类横向复杂度。`## 7` 会展开这一点。

### 1.2 算子绑定层：一份 kernel，两套注册方式并存

`csrc/` 顶层不是一个平铺的 kernel 文件夹，而是先按"功能"分：`attention/`、`moe/`、`quantization/`、`quickreduce/`、`rocm/`、`cpu/`、`cutlass_extensions/`。

这批目录相对小——比如 `csrc/attention/` 只有 6 个头文件，`csrc/moe/` 目前只剩 1 个 CPU 兜底实现（`dynamic_4bit_int_moe_cpu.cpp`），大部分 MoE kernel 已经不在这里了。

真正的大头是 `csrc/libtorch_stable/`（184 文件、69,144 行，`_lab/out/repo_stats.json` 的 `top_dirs` 已给出），下面同样有一份 `attention/`、`cutlass_extensions/`、`core/`——**同一批功能在仓库里出现了两份目录结构**。`csrc/libtorch_stable/torch_utils.h:21` 的注释把这份新目录的动机写得很直接："Device properties cache for stable ABI compatibility"（第 53 行同样出现"stable ABI compatible"字样）。

绑定入口也分裂成两处：`csrc/torch_bindings.cpp:21` 走的是老的 `TORCH_LIBRARY_FRAGMENT` 注册方式，`csrc/libtorch_stable/torch_bindings.cpp`（另一份同名文件，路径不同）走新的 stable-ABI 注册。

Python 侧对应的是 `vllm/kernels/`（33 文件、69,744 行）这个新的 IR 派发层，用 `vllm.ir.ops.<name>.register_impl(...)` 把逻辑算子路由到具体实现（`vllm/kernels/vllm_c.py:23`），而不是像老代码那样直接 `torch.ops._C.xxx(...)` 硬编码调用。

### 1.3 入口层：一个模型服务要说好几种协议方言

`vllm/entrypoints/`（211 文件、40,549 行）本身就分了十几个子目录，各自对应一种"客户端方言"或一类任务：

- `openai/` —— OpenAI 协议本体（chat/completion/responses/models）
- `anthropic/` —— Anthropic Messages API
- `cohere/` —— Cohere Chat V2
- `mcp/` —— Model Context Protocol 工具服务器
- `pooling/` —— embedding/rerank/classify 这类非生成任务
- `speech_to_text/` —— 转录（transcription/translation）
- `scale_out/` —— 弹性伸缩相关路由
- `serve/` —— `vllm serve` 自身的管理端点（sleep/wake/权重更新）
- `launchers/` —— 本篇 `## 2` 引用的路由装配层
- `cli/` —— 命令行子命令

这不是"功能蔓延"，是因为同一个推理引擎要同时兼容多家云厂商的客户端 SDK——`_lab/out/api_surface.json` AST 出 188 个协议类（pydantic model），对应的正是这些目录各自定义的请求/响应 schema；同一份 `api_surface.json` 里还 AST 出 64 个配置类（`ModelConfig`/`SchedulerConfig` 这类）——协议类数量是配置类的近 3 倍，说明这一层大部分复杂度花在"怎么把外部请求翻译成内部统一表示"，不是"内部有多少种配置组合"。

### 1.4 十个子系统一览

`_lab/out/struct_map.json` 用关键词 + AST 把仓库切成十个子系统（和 `## 1.1` 的目录级统计是两套不同的切法，一个文件可能同时被算进多个子系统，比如 KV 传输相关代码同时计入 `distributed` 和 `disagg`）。每个子系统挑一个规模最大的文件和一个有代表性的类，作为后续 07/05/10/11/12 等篇目的入口锚点：

| 子系统 | 行数 | 文件数 | 代表文件（最大） | 代表类 |
|---|---|---|---|---|
| `model_exec` | 392,324 | 901 | `vllm/v1/worker/gpu_model_runner.py`（7752 行） | `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py:501`） |
| `attention` | 81,107 | 158 | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`（3246 行） | `MLAAttention`（`vllm/model_executor/layers/attention/mla_attention.py:388`） |
| `quantization` | 56,317 | 170 | `vllm/model_executor/layers/quantization/modelopt.py`（2451 行） | `CompressedTensorsConfig`（`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:82`） |
| `distributed` | 54,851 | 138 | `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`（2640 行） | `GroupCoordinator`（`vllm/distributed/parallel_state.py:380`） |
| `entrypoint` | 45,972 | 210 | `vllm/entrypoints/chat_utils.py`（2057 行） | `LLM`（`vllm/entrypoints/llm.py:67`） |
| `disagg`（PD 分离/KV 传输） | 33,900 | 79 | `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`（2640 行） | `NixlBaseConnectorWorker`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:99`） |
| `speculative` | 18,360 | 64 | `vllm/v1/spec_decode/llm_base_proposer.py`（1896 行） | `SpecDecodeBaseProposer`（`vllm/v1/spec_decode/llm_base_proposer.py:71`）、`SpeculativeConfig`（`vllm/config/speculative.py:85`） |
| `kv_cache` | 9,581 | 13 | `vllm/v1/core/kv_cache_utils.py`（2323 行） | `KVCacheManager`（`vllm/v1/core/kv_cache_manager.py:118`） |
| `scheduler` | 8,424 | 20 | `vllm/v1/core/sched/scheduler.py`（3037 行） | `Scheduler`（`vllm/v1/core/sched/scheduler.py:73`） |
| `structured_output` | 2,808 | 10 | `vllm/v1/structured_output/utils.py`（574 行） | `StructuredOutputManager`（`vllm/v1/structured_output/__init__.py:36`） |

三个观察：一是 `distributed` 和 `disagg` 两行的"代表文件"是同一个文件（`moriio_connector.py`）——KV 传输连接器代码天生横跨"分布式通信"和"PD 分离"两个概念，脚本按关键词分类时会重复计入，这也是为什么 `## 1.1` 的 `subsystems` 总和不等于全仓库总行数，读数字时不能直接相加。二是 `structured_output` 只有 10 个文件、2,808 行，是十个子系统里最小的，但对应的功能（JSON schema/正则约束解码）在生产环境里出问题的概率不低于任何一个大子系统——**行数小不代表不重要**，只代表"这个功能可以用较少代码实现"。三是 `entrypoint` 子系统的代表类选了 `LLM` 而不是某个具体的 Serving 类，是因为 `chat_utils.py`（2057 行，专门处理多模态/工具调用消息的规范化）本身没有一个"主类"，它是一堆被各协议 Serving 类共用的纯函数集合——这也提示"最大文件"不一定对应"最重要的类"，两者要分开找。

## 2. 代码地图（文件 → 职责，带行号）

下面按"一次请求会依次经过的层"排列，而不是按目录字母序——这样读的时候能直接对上 `## 4` 的走读顺序。

| 层 | 文件:行 | 干什么 |
|---|---|---|
| 进程入口 | `pyproject.toml:44` | `console_scripts`：`vllm` 命令指向 `vllm.entrypoints.cli.main:main` |
| 进程入口 | `vllm/entrypoints/cli/main.py:17` | `main()`：遍历 `CMD_MODULES` 把 `serve`/`bench`/… 分发给对应子命令类 |
| 进程入口 | `vllm/entrypoints/cli/serve.py:59`-`60` | 判断是否启用 Rust 前端（`VLLM_USE_RUST_FRONTEND`），决定接下来起 Python 多进程 API server 还是 Rust 二进制 |
| Rust 前端（可选） | `rust/src/cmd/src/main.rs:123`-`145` | `Command::Serve` 分支：先拉起 `ManagedEngineHandle`（托管 Python headless 引擎子进程），再起 Rust HTTP server |
| Rust 前端（可选） | `rust/src/server/src/routes.rs:76`-`123` | Rust 侧独立实现的路由表：`/v1/chat/completions`、`/v1/completions`、`/v1/models` 等约 26 条 `.route()` |
| Rust 前端（可选） | `rust/src/engine-core-client/src/protocol/handshake.rs:50` | Rust 侧手写的 `EngineCoreReadyResponse` 结构体，与 Python 侧同名类型对拍握手协议 |
| HTTP 路由装配 | `vllm/entrypoints/launchers/api_server/routers.py:12` | `register_api_routers`：按 `supported_tasks` 决定挂哪些子路由（generate / pooling / transcription / …） |
| HTTP 路由装配 | `vllm/entrypoints/generate/api_router.py:21` | 生成类任务再扇出到 OpenAI / Anthropic / Cohere / Responses 各协议自己的路由模块 |
| HTTP 路由（OpenAI 协议） | `vllm/entrypoints/openai/chat_completion/api_router.py:40`-`53` | `@router.post("/v1/chat/completions")` 装饰器与 `create_chat_completion` 处理函数 |
| Serving 层 | `vllm/entrypoints/openai/chat_completion/serving.py:244` | `OpenAIServingChat.create_chat_completion`：请求校验、渲染 chat template、拼 `SamplingParams` |
| Serving 层 | `vllm/entrypoints/openai/chat_completion/serving.py:370` | 调用 `self.engine_client.generate(...)`，从 HTTP 世界进入引擎世界的最后一行 |
| 前端进程内引擎门面 | `vllm/v1/engine/async_llm.py:550`-`578` | `AsyncLLM.generate`：四步走（建 `AsyncStream` → 处理输入 → 挂进 Detokenizer → 发给独立进程的 EngineCore），docstring 原话见 `## 4` |
| 进程间通信 | `vllm/v1/engine/core_client.py:78`-`87` | `EngineCoreClient` 抽象基类，docstring 列出三种子类的适用场景 |
| 进程间通信 | `vllm/v1/engine/core_client.py:503`-`514` | `MPClient`：base class，用 ZMQ `input_socket`/`output_socket` 与后台 EngineCore 进程收发 |
| 进程间通信 | `vllm/v1/serial_utils.py:136`、`:313` | `MsgpackEncoder`/`MsgpackDecoder`：跨进程契约对象在 ZMQ 管道两端的编解码实现 |
| 引擎核心进程（启动） | `vllm/v1/engine/core.py:254` | `EngineCore._initialize_kv_caches`：跑一次 profile 前向传播定出可用显存，再算出 KV block 数 |
| 引擎核心进程（忙循环） | `vllm/v1/engine/core.py:1405`-`1416` | `EngineCoreProc.run_busy_loop`：轮询输入队列 → 跑一步 → 发布统计，循环到收到关停信号 |
| 引擎核心进程（一步） | `vllm/v1/engine/core.py:597`-`627` | `EngineCore.step`：`schedule()` → `execute_model()`（异步 future）→ `sample_tokens()` → `update_from_output()` |
| 调度层 | `vllm/v1/core/sched/scheduler.py:484`-`495` | `Scheduler.schedule`：源码 NOTE 原话说明"没有 prefill/decode 阶段划分"，统一按 `num_computed_tokens` 记账 |
| 调度层（异步变体） | `vllm/v1/core/sched/async_scheduler.py:12` | `AsyncScheduler(Scheduler)`：`async_scheduling` 开启时使用的子类，允许调度提前于上一步模型输出完成前推进 |
| 执行层 | `vllm/v1/executor/multiproc_executor.py:111` | `MultiprocExecutor`：EngineCore 之下再起 N 个 GPU worker 子进程 |
| 执行层 | `vllm/v1/executor/multiproc_executor.py:600` | `WorkerProc`：单个 worker 子进程的类，真正持有 GPU 显存和模型权重 |
| 模型执行 | `vllm/v1/worker/gpu_model_runner.py:4284` | `GPUModelRunner.execute_model`：拿到 `SchedulerOutput`，跑一次真正的前向传播 |
| 算子绑定（legacy） | `csrc/torch_bindings.cpp:21` | `TORCH_LIBRARY_FRAGMENT`：老式 C++/CUDA kernel 注册为 `torch.ops._C.*` |
| 算子绑定（stable ABI） | `csrc/libtorch_stable/torch_utils.h:21`、`:53` | 新绑定层的头文件注释，明确写"stable ABI compatibility" |
| 算子绑定（Python IR 派发） | `vllm/kernels/vllm_c.py:23`-`26` | Python 侧用 `vllm.ir.ops.rms_norm.register_impl` 把逻辑算子派发到 `torch.ops._C` 的具体实现 |
| 输出回路 | `vllm/v1/engine/output_processor.py:603` | `OutputProcessor.process_outputs`：把 `EngineCoreOutputs` 转成面向用户的 `RequestOutput` |
| 输出回路 | `vllm/v1/engine/detokenizer.py:31`-`69` | `IncrementalDetokenizer`：增量解码新 token，不必每步重跑整段文本的 tokenizer |
| 模型层重构信号 | `vllm/model_executor/models/registry.py:1005`-`1008` | 注释原文：非默认位置的模块（`vllm.models.<name>` 布局）被称为"legacy"之外的新路径 |
| 模型层重构信号 | `vllm/model_executor/models/registry.py:1475`-`1481` | `_resolve_module_name`：以 `vllm.` 开头的模块名直接用，否则默认拼到 `vllm.model_executor.models.` 下——这个 if/else 就是新旧两套布局共存的具体判据 |

以上引用只是骨架；`## 3` `## 4` 会把其中若干条的上下文摊开细读。

## 3. 核心数据结构

一次请求在三个进程之间流转，靠的是几个关键结构体做"契约"，而不是靠直接传对象引用（隔着进程边界，对象引用没有意义，必须先序列化）：

- **`VllmConfig`**（`vllm/config/vllm.py:357`）——启动时构造一次的总配置对象，`ModelConfig`/`SchedulerConfig`/`SpeculativeConfig`/`ParallelConfig` 等子配置全挂在它下面，EngineCore、Scheduler、GPUModelRunner 三层都拿同一份 `VllmConfig` 初始化。它不跨进程重建：Rust/Python 前端进程解析完 CLI 参数后序列化传给 EngineCore 子进程，而不是每层各自读一遍命令行。
- **`EngineArgs`**（`vllm/engine/arg_utils.py:424`）——CLI 参数到 `VllmConfig` 的转换层，`vllm serve` 的几百个 `--xxx` 旗标最终都在这里被解析、校验、组装。据 `_lab/out/api_surface.json`，AST 抽取到 233 个 CLI 旗标——这个数字会在 `09-vLLM-Python-API与EngineArgs` 篇细拆。
- **`EngineCoreRequest`**（`vllm/v1/engine/__init__.py:107`）——前端进程发给 EngineCore 进程的请求载荷，是**跨进程边界的第一个契约对象**：prompt token id、`SamplingParams`、LoRA 请求、优先级等字段都在这里。配套的 `EngineCoreRequestType`（`vllm/v1/engine/__init__.py:284`）是个枚举，标记这条消息是新请求、还是 abort、还是控制指令；`EngineCoreReadyResponse`（`vllm/v1/engine/__init__.py:73`）则是反方向——EngineCore 进程启动完成后回给前端进程的握手消息，Rust 前端在 `rust/src/engine-core-client/src/protocol/handshake.rs:50` 里独立定义了同名结构体来解这条消息，这就是"两边各自实现同一套协议"的具体样子。
- **`Request`**（`vllm/v1/request.py:59`）——`EngineCoreRequest` 到达 EngineCore 进程后，被包装成的**进程内**运行时对象，调度器操作的就是这个类型的实例；它比 `EngineCoreRequest` 多了 `num_computed_tokens`、KV block 归属这些只有调度器关心的运行时状态。
- **`SchedulerOutput`**（`vllm/v1/core/sched/output.py:207`）——**调度层到执行层的契约**：`Scheduler.schedule()` 产出它，`GPUModelRunner.execute_model()` 消费它。它不包含"调度器怎么想的"，只包含"这一步谁跑、跑多少 token、KV block 表长什么样"这类执行器需要的最终结果。
- **`SamplerOutput`**（`vllm/v1/outputs.py:254`）与**`ModelRunnerOutput`**（`vllm/v1/outputs.py:310`）——**执行层到调度层的契约**（反方向）：前向传播 + 采样跑完之后，worker 进程把新生成的 token id、logprobs 打包成这两个结构体，经 `MultiprocExecutor` 收集回 EngineCore 进程，喂给 `Scheduler.update_from_output`。
- **`Executor`**（`vllm/v1/executor/abstract.py:38`）——EngineCore 持有的执行器抽象基类，`MultiprocExecutor`（多进程/多卡）、`UniprocExecutor`（单进程，调试或单卡场景）都实现它；EngineCore 自己不关心执行器内部是几个进程，只调用统一的 `execute_model`/`sample_tokens` 接口——这层抽象是"调度器完全不知道 GPU worker 是几个进程"这件事的具体实现位置。
- **`LLM`**（`vllm/entrypoints/llm.py:67`）——离线批量推理的入口类，和在线 `AsyncLLM` 是姊妹关系：两者都是 `EngineClient` 协议的实现，`LLM` 内部用的是同步的 `SyncMPClient` 而不是 `AsyncMPClient`，但底下的 `EngineCoreRequest`/`SchedulerOutput`/`ModelRunnerOutput` 三层契约完全共用，这也是为什么本库要单独用一篇（`09-vLLM-Python-API与EngineArgs`）讲离线 API 而不是把它塞进本篇。

跨进程序列化本身也不是"随便 pickle 一下"：`vllm/v1/serial_utils.py:136` 的 `MsgpackEncoder` 和 `:313` 的 `MsgpackDecoder`，配合 `vllm/v1/engine/core_client.py:632`-`633` 的实际用法（`self.encoder = MsgpackEncoder(...)`、`self.decoder = MsgpackDecoder(EngineCoreOutputs)`），说明 vLLM 选的是 `msgspec`/`msgpack` 而不是 Python 标准库的 `pickle`——这个选择本身也值得记一笔：`msgpack` 是跨语言的二进制格式，Rust 前端能解出同一份消息（`rust/src/engine-core-client/src/protocol/`下一整套 `.rs` 文件对应 Python 侧的协议类），如果用 `pickle`，这条路会直接堵死，因为 `pickle` 格式是 Python 专属的。

一个容易忽略的细节（**源码为证**）：`vllm/engine/llm_engine.py` 整个文件只有 7 行，内容是 `from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine` 后接 `LLMEngine = V1LLMEngine`——也就是说 `vllm/engine/` 这个目录名字虽然还在，但已经不装真正的引擎实现了，只是给 V0 时代的 import 路径留的别名层。V0 的 `LLMEngine`/`AsyncLLMEngine` 实现已经被完全删除，`vllm/engine/async_llm_engine.py` 同理，也是 7 行的别名文件。

## 4. 主流程走读

### 4.0 冷启动：在第一个请求到达之前

请求走读通常默认"引擎已经在跑"，但这里先补一步容易被忽略的前置流程，因为它决定了后面每一步能用多少 KV cache（`04-vLLM-KV缓存与前缀缓存` 篇细讲具体的块分配算法，这里只讲它在启动序列里的位置）。

`EngineCore.__init__`（`vllm/v1/engine/core.py:108`）里会调用 `_initialize_kv_caches`（同文件 254 行），这个方法按顺序做四件事：

1. 让 worker 进程跑一次真实的前向传播来 profile 峰值显存占用——`available_gpu_memory = self.model_executor.determine_available_memory()`（第 308 行）。
2. 拿这个显存数字反算能建多少 KV cache block（具体公式留给 `04-vLLM-KV缓存与前缀缓存` 篇）。
3. 如果这次 profile 导致 `max_model_len` 被自动下调，专门发一次 `collective_rpc("update_max_model_len", ...)`（第 330 行）把新值同步给已经启动、缓存了旧值的 worker 们。
4. 调用 `self.model_executor.initialize_from_config(kv_cache_configs)`（第 343 行）把块表真正建起来。

源码注释里有一处很直白的时序说明："workers were spawned before memory profiling"（326 行附近）——也就是说 `WorkerProc` 子进程的创建，比"这个进程能用多少显存"这个问题的答案，还要早一步。这个先后顺序也解释了第 3 步为什么必须存在：worker 起来的那一刻还不知道真实可用显存，只能先用一个乐观估计跑，等 profile 结果出来才可能需要纠正。

### 4.1 一次请求怎么走

以 `POST /v1/chat/completions`、非流式、单机单卡、不开 Rust 前端为基准路径，逐跳给出文件和行号：

1. **进程 0：命令行启动。** `vllm serve <model>` 触发 `pyproject.toml:44` 声明的 `console_scripts` 入口，跑到 `vllm/entrypoints/cli/main.py:17` 的 `main()`，按子命令名分发到 `vllm/entrypoints/cli/serve.py`。这里第 59-60 行判断 `envs.VLLM_USE_RUST_FRONTEND`，默认 `False`（`vllm/envs.py:165`），走 Python 多进程 API server 路径。

2. **HTTP 路由装配（仍在同一个前端进程里）。** FastAPI `app` 由 `vllm/entrypoints/launchers/api_server/routers.py:12` 的 `register_api_routers` 统一挂路由：先挂 `vllm serve` 自身的管理路由，再判断 `"generate" in supported_tasks`，走到 `vllm/entrypoints/generate/api_router.py:21`，扇出到 OpenAI（chat/completion/responses）、Anthropic、Cohere 各自的 `api_router.py`。这一层解释了为什么 `_lab/out/api_surface.json` 能 AST 出 63 条路由、188 个协议类——一个模型服务同时要说 OpenAI、Anthropic、Cohere 三种"方言"（`## 1.3` 已给出目录层面的证据）。

3. **HTTP 请求进入具体路由。** `vllm/entrypoints/openai/chat_completion/api_router.py:40` 的 `@router.post("/v1/chat/completions")` 接住请求，`create_chat_completion`（同文件 53 行）从 `request.app.state` 取出预先建好的 `OpenAIServingChat` 实例（`chat()` 辅助函数，35 行附近），调用 `handler.create_chat_completion(request, raw_request)`。

4. **Serving 层：协议对象 → 引擎输入。** `vllm/entrypoints/openai/chat_completion/serving.py:244` 的 `OpenAIServingChat.create_chat_completion` 做请求校验、chat template 渲染、工具调用/推理解析器初始化，最终在 370 行调用 `self.engine_client.generate(engine_input, sampling_params, sub_request_id, ...)`——这一行是"HTTP 世界"和"引擎世界"的分界线。

5. **前端进程内的引擎门面。** `engine_client` 是 `AsyncLLM` 实例，`vllm/v1/engine/async_llm.py:550` 的 `generate()` 方法自己的 docstring（570-578 行）写得很直白，原话四步是："1) Making an AsyncStream corresponding to the Request. 2) Processing the Input. 3) Adding the Request to the Detokenizer. 4) Adding the Request to the EngineCore (separate process)."——第 4 步明确写了 `separate process`：这是**源码自己承认**的进程边界，不是本库的推断。

6. **跨进程：序列化 + ZMQ 发送。** `add_request` 内部经 `EngineCoreClient`（`vllm/v1/engine/core_client.py:78`-`87`，docstring 列出 `InprocClient`（V0 兼容/调试用）、`SyncMPClient`（给同步 `LLM` 用）、`AsyncMPClient`（给 `AsyncLLM` 用）三种子类）把 `EngineCoreRequest` 用 `MsgpackEncoder`（`vllm/v1/serial_utils.py:136`）编码成字节流，通过 `MPClient`（同文件 503 行）持有的 `input_socket` ZMQ 套接字推给后台的 EngineCore 进程。

7. **进程 1：EngineCore 忙循环。** `vllm/v1/engine/core.py:1405` 的 `run_busy_loop` 是这个独立进程的主循环：轮询输入队列（`_process_input_queue`）、跑一步（`_process_engine_step`）、发布统计，循环往复直到收到关停信号。

8. **调度 + 执行一步。** `_process_engine_step` 最终落到 `vllm/v1/engine/core.py:597` 的 `EngineCore.step`：先 `scheduler_output = self.scheduler.schedule(...)`（调 `vllm/v1/core/sched/scheduler.py:484`；如果开了异步调度，实际跑的是它的子类 `AsyncScheduler`，`vllm/v1/core/sched/async_scheduler.py:12`），再 `future = self.model_executor.execute_model(scheduler_output, non_block=True)` 异步发起执行，接着算结构化输出的语法位掩码，最后 `future.result()` 拿到模型输出、`self.scheduler.update_from_output(...)` 把结果写回调度器状态。

9. **进程 2+：GPU worker 执行。** `self.model_executor` 是 `MultiprocExecutor`（`vllm/v1/executor/multiproc_executor.py:111`），它自己不跑模型，而是把 `SchedulerOutput` 转发给一个或多个 `WorkerProc` 子进程（同文件 600 行）；每个 worker 里的 `GPUModelRunner.execute_model`（`vllm/v1/worker/gpu_model_runner.py:4284`）才是真正调用 PyTorch 前向传播、走到 `torch.ops._C.*`（由 `csrc/torch_bindings.cpp:21` 注册，或 stable-ABI 新绑定层）算子的地方。

10. **结果沿原路返回。** `ModelRunnerOutput` 从 worker 进程传回 EngineCore 进程 → `EngineCoreOutputs` 用 `MsgpackDecoder`（`vllm/v1/serial_utils.py:313`）在前端进程解码 → `vllm/v1/engine/output_processor.py:603` 的 `process_outputs` 把它转成 `RequestOutput`，途中调用 `vllm/v1/engine/detokenizer.py:31` 的 `IncrementalDetokenizer` 做增量 detokenize（不是每步都把全部已生成 token 重新过一遍 tokenizer）。`AsyncLLM.generate` 里的 `async for` 循环把结果 `yield` 回 Serving 层，Serving 层再包成 SSE 流或一次性 JSON 响应给 HTTP 客户端。

十步里跨了两次进程边界（前端 ↔ EngineCore、EngineCore ↔ worker），如果开启 Rust 前端，第 1-4 步会换成 `rust/src/server/src/routes.rs` 里独立实现的 Rust handler（比如 85 行的 `/v1/chat/completions` 路由），但第 5 步开始（Rust 进程通过 `engine-core-client` crate 与 Python EngineCore 用同一套 msgpack 协议对话，握手消息见 `rust/src/engine-core-client/src/protocol/handshake.rs:50`）不变——这也是为什么 Rust 前端能做到"只换前端，模型执行代码一行不用动"。

## 5. 设计决策与代价

### 决策一：EngineCore 是独立进程，不是前端进程里的一个对象

- **为什么这么设计**：Python 的 GIL 让"处理 HTTP 请求 + JSON 序列化 + chat template 渲染"这类 CPU 密集工作和"驱动 GPU kernel launch 的忙循环"抢一个锁。把 EngineCore 拆到独立进程，用 ZMQ + msgpack 做 IPC（`vllm/v1/engine/core_client.py:503`-`514`、`vllm/v1/serial_utils.py:136`），可以让前端的 CPU 工作不阻塞 GPU 侧的调度节奏，反之亦然。
- **不这样会怎样**：`EngineCoreClient` 确实留了一个同进程实现 `InprocClient`（`vllm/v1/engine/core_client.py:306`），`make_client` 静态方法上方的注释说明这条路径主要是给调试场景用的（`vllm/v1/engine/core_client.py:97`）——换句话说，vLLM 团队自己验证过"同进程"是可行的技术选项，但没有把它当成生产默认值，暗示他们观察到了同进程下的干扰问题（**本库推断**，未见到该取舍的量化说明，未查证具体量级）。
- **什么时候可以不这样**：单机调试、单元测试、或者请求量极低到 GIL 竞争完全不构成瓶颈的场景，`InprocClient` 仍然可用，省掉序列化/反序列化和跨进程调度的开销。

### 决策二：EngineCore 之下还有一层独立的 worker 进程

- **为什么这么设计**：多 GPU 并行（张量并行/流水并行）天然需要多进程——每张卡一个进程持有自己的那部分模型权重和显存。`MultiprocExecutor`（`vllm/v1/executor/multiproc_executor.py:111`）把"EngineCore 只管调度决策，worker 只管执行"这个职责边界，做成了单卡也不例外的统一路径。
- **不这样会怎样**：如果单卡场景走一条特殊的"同进程执行"捷径、多卡场景才用多进程，会出现两套 `execute_model` 调用路径，调试和维护成本翻倍——`Executor` 抽象基类（`vllm/v1/executor/abstract.py:38`）统一了这个接口，`UniprocExecutor` 和 `MultiprocExecutor` 对上层暴露相同的方法签名，让 EngineCore 完全不用关心"下面是几个进程"。
- **什么时候可以不这样**：如果目标场景是"纯 CPU 推理""极小模型调试"，`UniprocExecutor` 就是这个"不这样"的落地实现，进程内直接跑模型，省掉进程间通信的序列化开销，但放弃了多卡扩展能力。

### 决策三：模型代码新旧两套布局并存（`model_executor/models/` vs `models/`）

- **为什么这么设计**：`registry.py` 的注释原文把这种新路径称为"hardware-isolated"布局（`vllm/model_executor/models/registry.py:1005`-`1008`）——新一代超大 MoE 模型（DeepSeek V4、Kimi K3、MiniMax M3 等）需要为 NVIDIA/AMD/XPU 分别写 profile 专属的 kernel 路径（比如 FlashMLA vs AITER 版 gated-delta-net vs XPU 稀疏 kernel），继续塞进旧布局的单文件模式意味着一个模型文件里堆三份平台分支。
- **不这样会怎样**：旧布局下"一个文件塞多硬件分支"已经在失控——`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 单文件 2055 行、`vllm/model_executor/layers/attention/mla_attention.py` 单文件 3062 行（均据 `_lab/out/struct_map.json` 的 `attention` 子系统 `top_files`），这两个文件本身就是"继续硬撑扁平布局"的活例子。
- **什么时候可以不这样**：绝大多数模型不需要按硬件拆目录——293 个 legacy 模型里，Bert、Bloom 这类结构简单、不需要平台专属融合算子的模型完全没必要迁移，继续单文件扁平布局反而更容易读；新布局的代价（多层目录、`common/`+`nvidia/`+`amd/`+`xpu/` 至少 4 份 `__init__.py`）只有在真的需要平台分叉时才划算。

### 决策四：算子绑定层同样保留新旧两套注册机制

- **为什么这么设计**：`csrc/torch_bindings.cpp:21` 代表的老式 `TORCH_LIBRARY_FRAGMENT` 绑定方式依赖 PyTorch 内部（不稳定）的 C++ ABI，每次 PyTorch 大版本升级都可能要重新编译甚至改代码；`csrc/libtorch_stable/` 这套新绑定（`csrc/libtorch_stable/torch_utils.h:21`、`:53` 明确写"stable ABI"）改用 PyTorch 官方提供的稳定 ABI 层，理论上可以做到"一次编译，跨 PyTorch 次版本兼容"。
- **不这样会怎样**：如果全仓库一次性切换到 stable ABI，意味着要同时重写全部 169 个 CUDA 文件和配套的 pybind 声明，且要验证每一个 kernel 在新绑定方式下数值行为不变——这个工作量本身就是"分批迁移、新老并存"这个决策的直接代价来源；目前能看到的信号是新模型家族（`vllm/models/` 下那 6 个）的算子更倾向于挂在新绑定层，但没有找到一条"全部 kernel 何时完成迁移"的时间表（**本库推断**，未查证 roadmap）。
- **什么时候可以不这样**：CPU-only 后端（`csrc/cpu/`，38,110 行，`_lab/out/repo_stats.json` 的 `top_dirs` 已给出）对 ABI 稳定性的敏感度远低于 CUDA/ROCm（没有那么频繁的驱动/编译器版本组合要适配），继续用老式绑定方式的收益/成本比可能反而更高。

### 决策五：Rust 前端可选，不是默认替换 Python 前端

- **为什么这么设计**：Rust 的 axum/tokio 栈处理 HTTP 路由、JSON 序列化、tokenizer（Rust 侧用 `tokenizers`/`fastokens`/`tekken` 这类 crate，据 `rust/Cargo.toml` 依赖声明）能绕开 Python GIL 和 CPython 对象开销，这部分恰好是"HTTP 前端最费 CPU、离 GPU 计算最远"的路径；而模型执行继续留在 Python，因为 PyTorch/CUDA 生态仍然是 Python-first。
- **不这样会怎样**：`rust/src/server/src/routes.rs` 目前只注册了约 26 条路由（`.route(` 出现 27 次），而 Python 侧的路由表有 63 条（`_lab/out/api_surface.json` 的 `n_routes`）——对照两边路径清单，Rust 侧覆盖了核心的 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、权重更新、sleep/wake 这些管理端点，但没有看到 `/v1/embeddings`、`/v1/responses`、`/v1/messages`（Anthropic 协议）、`/v1/rerank`、`/v2/embed`、Cohere 端点在 Rust 侧的对应实现。如果把 `VLLM_USE_RUST_FRONTEND` 设成默认开启，这些端点会直接不可用而不是优雅降级——这也是为什么它现在是 `False`（`vllm/envs.py:165`）。
- **什么时候可以不这样**：单机低 QPS 调试、或者需要用到还没被 Rust 覆盖的功能（多协议 profile、MCP tool server 等）时，应该继续用默认的 Python 前端；只有在"HTTP 层可验证是瓶颈、且用到的端点都在 Rust 覆盖范围内"的高吞吐部署场景，才值得打开这个开关。

### 决策六：调度器不分 prefill/decode 两个阶段，统一按 token 预算记账

- **为什么这么设计**：`Scheduler.schedule`（`vllm/v1/core/sched/scheduler.py:484`）上方的 NOTE 注释原话写道——"There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec"（486-495 行）。用一套统一的"追赶目标 token 数"逻辑，可以让 chunked prefill、prefix cache 复用、投机解码的 draft token 都走同一条记账路径，不用给每种场景写专门分支。
- **不这样会怎样**：如果保留"prefill 阶段 vs decode 阶段"两套调度路径，新特性（chunked prefill、投机解码）落地时往往要同时改两条路径，容易出现只在一条路径生效的不一致——这类"两条平行代码路径必须手动保持同步"的维护负担，正是本库另一篇 [[03-vLLM-调度器解剖]] 里用具体 bug 场景展开的话题（此处不重复取证，避免和该篇内容重叠）。
- **什么时候可以不这样**：如果工作负载高度单一（比如只做 decode-only、batch size 固定的推理服务），针对性写一条简化的 decode-only 快路径，跳过通用调度器的分支判断，理论上能省掉一些调度开销——这更接近 PD 分离部署里 decode 节点的场景，具体是否真的这样实现，留给篇目表里的「12-vLLM-PD分离与KV-Connector」一篇核实，此处标注**本库推断**。

## 6. 同位对照：SGLang 在同一位置怎么做

`_src/sglang` 同样存在一个 `rust_extensions/` 目录和 `python/sglang/srt/managers/rust_server.py`，但和 vLLM 的摆法不一样：这个模块自己的 docstring 写得非常直接——"The Rust server replaces the Python api-server + `TokenizerManager` + `DetokenizerManager` stack (hence this module sits beside them in `managers/`), running them as Rust threads **inside the scheduler process**"（`sglang:python/sglang/srt/managers/rust_server.py:1`-`8`）。

两边的取舍方向一致（把 HTTP/tokenize/detokenize 这类 CPU 密集、非 GPU 路径迁出纯 Python），但迁移单位不同：

- **vLLM**：Rust 前端是独立的可执行文件、独立的操作系统进程，通过 `ManagedEngineHandle::spawn` 去托管一个 Python "headless" 引擎子进程（`rust/src/cmd/src/main.rs:145`），进程边界清晰，两边只靠 ZMQ + msgpack 协议对话。
- **SGLang**：Rust 代码是"线程"，嵌进已经存在的 Python 调度器进程内部（`sglang:python/sglang/srt/managers/rust_server.py:5`），替换的是同一进程里的三个 Python 组件，而不是新增一个独立进程。

另一处可以对照的差异：vLLM 把"调度决策"（`Scheduler`，跑在 EngineCore 进程）和"模型执行"（`GPUModelRunner`，跑在独立的 `WorkerProc` 进程）切成两个隔着 IPC 的进程（`vllm/v1/executor/multiproc_executor.py:600`）；而 SGLang 的 `Scheduler` 类在单个 rank 内直接把 `TpModelWorker` 当成自己的属性持有并同步调用——`self.tp_worker = TpModelWorker(**worker_kwargs)`（`sglang:python/sglang/srt/managers/scheduler.py:924`），调度和单卡内的模型执行在同一个进程对象图里，不经过 IPC。（多卡场景下 SGLang 同样会为每个 TP rank 起独立进程，这里对照的是"调度器与它直接管理的那一路模型执行"之间是否跨进程，不是"整个系统是否单进程"。）

但"进程拓扑"这一层，两边其实是同构的：SGLang 的 `Engine._launch_subprocesses` 明确用 `multiprocessing` 拉起调度器进程（docstring 原话"Launch scheduler processes using multiprocessing"，`sglang:python/sglang/srt/entrypoints/engine.py:864`，紧接着多处 `mp.Process(...)` 调用），也就是说 SGLang 默认同样是"HTTP/Tokenizer 进程 + 一个或多个调度器/GPU 进程"的多进程拓扑，和 vLLM"前端进程 + EngineCore 进程 + worker 进程"的三层结构，本质上是同一个"把 GPU 忙循环和 Python Web 框架拆到不同进程"的答案，只是切分粒度不同（vLLM 多切了一刀，把调度和执行也分进了两个进程；SGLang 把调度和单卡执行留在同一个进程里）。

两边都还留着经典的 FastAPI Python 服务器作为默认路径没有删掉：`sglang:python/sglang/srt/entrypoints/http_server.py:456` 的 `app = FastAPI(...)` 和 `vllm/entrypoints/launchers/api_server/routers.py:12` 的 `register_api_routers` 是同一层级的对应物，Rust 都是 opt-in 而非默认。这种"请求面往 Rust 迁、模型执行留在 Python 生态"是不是两个项目组各自独立收敛出的行业共识，还是互相观望的结果，本库不下结论，留给 `_PLAN.md` 篇目表里的「04-工程规模与代码结构对比」一篇系统化验证。

## 7. 踩坑与反直觉

- **"调度器是核心"是错觉，至少不是代码量意义上的核心。** 十个子系统里 `scheduler` 只有 8,424 行，比 `structured_output`（2,808 行）大不了多少量级差异，但比 `model_exec`（392,324 行）小了近 47 倍（据 `_lab/out/struct_map.json`）。真正堆代码的地方是"同一个能力要在 N 个硬件后端、M 个模型家族上分别实现一遍"，不是调度算法本身有多精巧。
- **"V0 已经被删掉"和"V0 的名字还在"是两回事。** `vllm/engine/llm_engine.py`、`vllm/engine/async_llm_engine.py` 这两个文件都还在仓库里、都还能 `import`，但内容只是指向 `vllm.v1.engine.*` 的别名（各 7 行）。如果只看 `import` 语句判断"这个项目是不是还在维护 V0 兼容层"，会得出错误结论——真正的实现早就只剩 V1 一份。
- **"多进程"不代表"两个进程"。** 直觉上容易以为 vLLM 是"一个 HTTP 进程 + 一个 GPU 进程"两层结构，但源码显示是三层：前端进程、EngineCore 进程、（每个并行 rank 一个的）worker 进程（`vllm/v1/executor/multiproc_executor.py:111` 与 `:600`）。开 Rust 前端后会变成四层（Rust 进程额外托管 Python headless 引擎）。
- **"新模型都该往新的 `vllm/models/` 布局写"是过度推断。** `vllm/models/` 目前只装了 6 个模型家族（deepseek_v32、deepseek_v4、dots3_note、inkling、kimi_k3、minimax_m3），293 个 legacy 模型完全没有跟进——本库没有找到一条"多大规模的模型必须迁移"的显式判据（`## 8` 会把这个观察写成可改进点，而不是当作既定事实）。
- **重构信号不止一处，是三处同构的模式在同时发生。** 模型层（`model_executor/models/` vs `models/`）、算子绑定层（`csrc/torch_bindings.cpp` vs `csrc/libtorch_stable/`）、算子派发层（老式 `torch.ops._C.xxx` 硬编码 vs 新的 `vllm/kernels/` IR 派发）——三个独立的"扁平旧目录 + 隔离新目录"并存局面，各自的迁移进度还不一样（模型层只覆盖 6 个家族，算子绑定层覆盖了 184 个文件/69,144 行）。如果只读到其中一处就下结论"vLLM 在重构 X"，会漏掉这其实是一场更大范围、跨多个子系统同时进行的架构演进。
- **测试/生产代码比 0.542 不能直接读成"测试覆盖不到六成"。** 这个比例是全仓库 Python 行数的粗口径，`tests/kernels`（93,398 行）这类目录本身就包含大量跑分脚本和对照实现，不是纯断言代码；反过来 `csrc/`（C++/CUDA，未计入这个 Python 专属比例）的正确性验证很大程度上靠 `tests/kernels` 下的 Python 测试跑，两边不是简单的一一对应关系。

## 8. 可改进点

以下几条标注为**本库推断**，未提交 issue 或 PR 核实维护者是否已经在计划中——单纯是读代码时观察到的、可以在后续篇目里继续深挖的方向：

1. **新旧模型布局之间缺一条显式的迁移判据。** `_resolve_module_name`（`vllm/model_executor/models/registry.py:1475`-`1481`）只负责"识别一个模块名是不是已经在新布局里"，但仓库里没有找到"什么条件下一个模型应该从 `model_executor/models/` 迁到 `models/`"的书面规则——现在看起来更像是"哪个模型的作者愿意就迁"。一条可核验的判据（比如"需要 ≥2 个平台专属 kernel 实现即迁移"）会比隐性约定更容易长期维护，也更方便新贡献者判断该往哪写。同样的问题在算子绑定层（`torch_bindings.cpp` vs `libtorch_stable/`）也存在，可以用同一条判据统一处理。
2. **Rust 路由表和 Python 路由表没有单一真源（single source of truth）。** 两边各自硬编码：Python 在 `vllm/entrypoints/*/api_router.py` 里分散声明，Rust 在 `rust/src/server/src/routes.rs` 里集中声明。`## 5` 决策五已经指出两边路径集合存在真实差异（Rust 缺 `/v1/embeddings`、`/v1/responses` 等）。一个从两侧分别抽取路径集合、跑 diff 的 CI 检查，能在"漂移"发生时给出明确报错，而不是让打开 `VLLM_USE_RUST_FRONTEND` 的用户遇到静默的 404。
3. **`InprocClient` 的定位偏模糊。** 代码里存在（`vllm/v1/engine/core_client.py:306`），但 `make_client` 上方只有一行简短说明这是给调试用的（`vllm/v1/engine/core_client.py:97`），没有更详细的文档说明它和生产路径在正确性上是否完全等价——如果两条路径（同进程 vs 跨进程）在某些边界条件下行为不同，调试时用 `InprocClient` 复现的问题可能在生产的 `AsyncMPClient` 路径下并不存在，反之亦然。
4. **冷启动阶段"先起 worker 再 profile 显存"这个时序，对失败场景的暴露不够直接。** `_initialize_kv_caches`（`vllm/v1/engine/core.py:254`）里 profile 出的显存不够、或者需要下调 `max_model_len` 时，走的是事后用 `collective_rpc` 同步新值（同文件 330 行）这条补救路径，而不是在 worker 启动前先做一次轻量估算、把大概率会失败的配置提前拦下——对于"模型太大、显存明显不够"这类一眼能看出来的配置错误，用户体验上要多等一轮 worker 启动 + profile 才能看到报错。
5. **本篇给出的所有路由/文件数字是静态 AST 抽取，不是运行时验证。** `_lab/out/api_surface.json` 的路由计数依赖脚本能正确识别 `@router.post(...)` 这类装饰器模式；如果未来某个协议改用不同的注册方式（比如动态 `add_api_route`），计数会静默漏掉而不是报错——这是 `_lab/api_surface.py` 本身的已知局限，不是 vLLM 的问题，但读本篇数字时要记住这一点。

## 9. 自测题与延伸阅读

**闭卷自测**（不看正文，能答上来才算过）：

1. 默认配置（不开 Rust 前端）下，一个 `/v1/chat/completions` 请求从 HTTP 到 token 生成，最少要穿过几个操作系统进程？分别在做什么？
2. `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 方法上方注释为什么强调"没有 prefill 阶段和 decode 阶段"？这句话是在防止哪一类未来可能出现的 bug？
3. `vllm/model_executor/models/` 和 `vllm/models/` 都装模型定义代码，`_resolve_module_name` 函数是怎么判断一个模型架构名该走哪条导入路径的？除了模型层，本篇还在哪一层找到了同构的"扁平旧目录 + 隔离新目录"现象？
4. `VLLM_USE_RUST_FRONTEND` 的默认值是什么？打开它之后，Python 侧的 `EngineCore` 进程会被 Rust 实现替换掉吗？前端进程和 EngineCore 进程之间靠什么格式交换数据？
5. `EngineCoreClient` 有哪三个具体子类？分别对应 `LLM`、`AsyncLLM`、调试场景里的哪一个？
6. 为什么 `csrc/torch_bindings.cpp` 里的 `TORCH_LIBRARY_FRAGMENT` 宏对"vLLM 支持多硬件后端"这件事很关键？如果没有这一层，Python 的 `vllm/kernels/vllm_c.py` 要怎么调用到具体的 CUDA/ROCm 实现？
7. EngineCore 启动时，"worker 进程被创建"和"这个进程能用多少显存被确定"，哪一步先发生？如果后一步的结果要求调低 `max_model_len`，源码里是怎么把新值同步给已经起来的 worker 的？

**延伸阅读**（双链只取自 `_PLAN.md` §6 名册）：

- [[03-vLLM-调度器解剖]] —— 调度器本体的逐行拆解：`schedule()` 的 running/waiting 双扫描、抢占策略、`SchedulerOutput` 十四个字段各自的用途；本篇 `## 5` 决策六提到的"NOTE 注释"在那篇会被完整展开。
- [[04-vLLM-KV缓存与前缀缓存]] —— KV 缓存/前缀缓存的哈希链、块池驱逐算法、混合注意力的不动点计算；本篇 `## 4.0` 只讲了"冷启动时 KV 块数怎么被算出来"这一步在流程里的位置，具体算法是刻意让给那一篇的。
- [[_PLAN]] —— 第 1、2 节的诚实标准和取证基准定义；本篇给出的所有取证基准、双链、代码引用是否真实可核验，都以它和仓库根目录的 `_verify.py` 为准。
