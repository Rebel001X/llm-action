# SGLang 全景与代码地图

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：运行时六十万行，前端 DSL 不到五千行，"S" 早就不是重点。

## 0. 结论先行

- **SGLang 的名字来自"Structured Generation Language"，但这门语言现在只剩 4,644 行**（`python/sglang/lang/`，14 个文件，本库对 `_src/sglang` 实测统计），而它要驱动的运行时 `python/sglang/srt/` 有 704,134 行——**运行时比语言前端大 151 倍**。`lang/` 甚至不直接跑模型：它的 `RuntimeEndpoint` 后端就是个 HTTP 客户端，把 `srt` 当成远程服务来调（`python/sglang/lang/backend/runtime_endpoint.py:82`）。`## 1` `## 7` 会展开这个反差。
- **`python/sglang/kernels/` 不是一整块自研代码，是三种不同来源拼起来的命名空间**：`kernels/aot/` 是原先独立发布的 `sgl-kernel`（PyPI 包名 `sglang-kernel`，Python import 路径仍是 `sgl_kernel`，见 `python/sglang/kernels/aot/README.md:1`-`13`）迁进主仓库后的样子，是 SGLang 团队自己维护的 C++/CUDA/CMake 工程；而 `kernels/ops/attention/flash_attn/cute/` 与 `kernels/ops/attention/fa4_sm120/` 是**原样拷贝**（vendoring）自 Tri Dao 团队的 FlashAttention-4（CuTeDSL 版本），版权头至今写着"Copyright (c) 2025, Tri Dao."（`python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py:1`-`2`），既不是 git submodule（仓库里没有 `.gitmodules`），也不是 pip 依赖。`## 5` 第二、三条决策会给出为什么这么拼。
- **`server_args.py` 是一个 10,142 行的单文件，`ServerArgs` 这一个类就占了 9,401 行、476 个字段**（类体从 `python/sglang/srt/server_args.py:473` 到第 9,873 行，`PortArgs` 紧接着在 `:9970` 起头）。这不是历史包袱式的失控——类文档字符串里专门写了"Adding new arguments"分节指南（同文件 476-480 行附近），`__post_init__` 编排了 63 个 `_handle_*` 方法组成的校验/归一化流水线（`python/sglang/srt/server_args.py:3646` 的 `_run_resolution_pipeline`）。是刻意的集中式设计，代价在 `## 5` 决策一里拆开算。
- **`python/sglang/multimodal_gen/` 是一条完全独立的第二运行时**，官方称呼是"SGLang diffusion"，服务的是图像/视频生成模型（Wan、FLUX、Qwen-Image 等，见 `python/sglang/multimodal_gen/README.md:5`-`9`），有自己的 `Scheduler`（`python/sglang/multimodal_gen/runtime/managers/scheduler.py:80`）、自己的 `ServerArgs`（`python/sglang/multimodal_gen/runtime/server_args/server_args.py:212`）、自己的 FastAPI app（`python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:397`），312,776 行代码，和服务 LLM 的 `srt` 互不调用。`## 1` `## 6` 会展开这条流水线是不是"第二个 SGLang"。
- **一次 `/generate` 请求默认至少跨 3 个操作系统进程**：HTTP 进程（`TokenizerManager` 常驻，`python/sglang/srt/managers/tokenizer_manager.py:386`）→ ZMQ → 调度器进程（每个 TP×PP rank 一个，或数据并行下的 `DataParallelController`，`python/sglang/srt/entrypoints/engine.py:856`）→ ZMQ → detokenizer 进程（`python/sglang/srt/managers/detokenizer_manager.py:92`）。多机部署时最前面还可能站着 `sgl-model-gateway`（Rust 写的独立网关进程，约 12.4 万行含测试），把请求路由到多个 SGLang 实例。`## 4` 逐跳给出文件和行号。
- **复杂度不是均匀摊开的**：按 `_lab/out/struct_map.json` 的十个子系统口径，`attention` 217,907 行遥遥领先，是 `scheduler`（35,054 行）的 6.2 倍；但这 21.8 万行里相当一部分是刚才提到的 vendored FlashAttention-4 CuTeDSL 内核（单文件 `flash_fwd_sm100.py` 就有 5,610 行）——"attention 子系统最大"和"attention 是 SGLang 自研代码量最大的部分"是两个不同的结论，`## 7` 会把这个坑挑明。

## 1. 它在系统里的位置

`_src/sglang` 是 `https://github.com/sgl-project/sglang.git` 的一次 `--depth 1` 浅克隆，sha 为 `15a439832054c1809a2bf59f7b94bf9dd71de282`（`_lab/out/repo_stats.json` 与 `_lab/out/struct_map.json` 的 `ref` 字段一致）。仓库根目录不只是一个 Python 包：

```
sglang/
├── python/sglang/     # 本篇主战场：srt / lang / kernels / multimodal_gen / cli
├── rust/               # 三个 crate：sglang-server（嵌入式）/ sglang-grpc / sglang-mm
├── sgl-model-gateway/  # 独立 Rust 网关：多 worker 路由、PD 编排、OpenAI 兼容代理
├── 3rdparty/           # 第三方子仓库（AMD wheel 等）
├── test/、benchmark/、docs/、examples/、scripts/
```

语言构成（`_lab/out/repo_stats.json` 的 `languages`）：

| 语言 | 行数 | 文件数 | 角色 |
|---|---|---|---|
| Python | 1,859,684 | 5,750 | 系统装配层：调度、内存池、HTTP 协议、模型定义、大部分 kernel 的 Python 侧封装 |
| Rust | 156,078 | 434 | `sgl-model-gateway` 独立网关 + 嵌入式 `rust/sglang-server` PyO3 扩展 |
| CUDA | 98,845 | 246 | `kernels/aot`（自研 + 迁入）与 `kernels/jit` 的手写 GPU kernel |
| JSON | 94,696 | 588 | 主要是 MoE/量化的调优配置表（如按 `E=...,N=...,device_name=...` 命名的 tuning JSON） |
| C/C++ header | 28,105 | 93 | kernel 头文件、pybind/torch extension 声明 |
| Markdown | 27,278 | 158 | 文档 |
| C++ | 23,874 | 52 | kernel 宿主代码 |
| Go | 7,598 | 22 | （范围较小，未深入核实用途，标注**未查证**） |

全仓库总计 7,532 个被统计文件、2,317,985 行；Python 产品码 1,254,674 行、测试码 605,010 行，测试/产品 = 0.482（均据 `repo_stats.json` 的 `totals`）——和 vLLM 的 0.542（见 `[[01-vLLM-全景与代码地图]]`）量级接近，两边都不适合直接读成"测试覆盖六成"，理由同样是大量正确性验证靠 CI 里跑真实模型对齐，不体现在 pytest 行数里（**本库推断**，未去核对 CI workflow）。

`_lab/out/repo_stats.json` 的 `top_dirs` 只统计到 `python/sglang`（4,435 文件/1,543,126 行）这一层，不再往下拆——这也是本篇要做的事。本库对 `_src/sglang/python/sglang` 各子包做了一次实测（`find + cat | wc -l`，统计的是含空行/注释的原始行数，口径不同于 `repo_stats.json` 里"code"字段的净代码行，下表标注**源码为证**，方法可复现）：

| 子包 | 文件数 | 行数 | 占 `python/sglang` 主要包总量的比例 | 角色 |
|---|---|---|---|---|
| `srt/` | 1,646 | 704,134 | 54.4% | LLM 推理运行时：调度、KV 缓存、HTTP/gRPC 协议、模型定义、分布式 |
| `multimodal_gen/` | 951 | 312,776 | 24.2% | 独立的图像/视频生成运行时（"SGLang diffusion"） |
| `kernels/` | 594 | 207,602 | 16.0% | 统一 kernel 命名空间（不含 `.cu`/`.cuh`，那部分在 `languages` 表的 CUDA 行里） |
| `test/` | 178 | 54,975 | 4.2% | `python/sglang` 包内自带的测试（不同于仓库顶层 `test/registered` 那 41 万行） |
| `benchmark/` | 23 | 9,793 | 0.76% | kernel/调度器微基准 |
| `cli/` | 7 | 1,125 | 0.09% | `sglang serve` / `sglang generate` 命令行分发 |
| `lang/` | 14 | 4,644 | 0.36% | 前端 DSL：`sgl.gen()`、chat 模板、多种远端 LLM 后端客户端 |

（占比分母为以上七项之和 1,295,049 行，不含 `python/sglang` 顶层散文件。）

`srt/` 这 704,134 行本身也不是铁板一块，本库对它内部的主要子目录同样做了一次实测（口径同上，含空行/注释）：

| `srt/` 子目录 | 文件数 | 行数 | 角色 |
|---|---|---|---|
| `layers/` | 317 | 143,618 | `nn.Module` 层：注意力后端、量化方法、MoE、LoRA 等——是"模型怎么算"和"kernel 怎么调"之间的胶水层 |
| `models/` | 254 | 160,964 | 具体模型架构定义，一个模型家族大体一个文件（类似 vLLM 的 `model_executor/models/`） |
| `mem_cache/` | 126 | 62,850 | KV 缓存池、RadixAttention 前缀缓存、分层缓存（HiCache） |
| `managers/` | 49 | 37,514 | `TokenizerManager`/`Scheduler`/`DetokenizerManager`/`DataParallelController`/`ScheduleBatch` 等进程级组件 |
| `disaggregation/` | 33 | 26,908 | PD 分离：NIXL/Mooncake 等传输后端、encoder 侧接收器 |
| `entrypoints/` | 60 | 25,908 | HTTP/gRPC 协议族：OpenAI、Anthropic、Ollama、原生协议 |
| `speculative/` | 48 | 23,655 | EAGLE、DFlash 等投机解码 worker |
| `configs/` | 64 | 16,012 | 模型 config 解析（`transformers`-style `config.json` 到内部配置对象） |
| `distributed/` | 28 | 10,012 | 进程组、通信原语封装（大部分张量并行通信走 `kernels/` 里的算子，这里是编排层） |

`layers/` 和 `models/` 加起来 304,582 行，占 `srt/` 总量的 43.3%——"模型怎么定义、层怎么组装"仍然是运行时里最重的部分，这一点和 vLLM 的 `model_executor/`（据 `[[01-vLLM-全景与代码地图]]`，单目录 424,532 行、超五分之一仓库体量）方向一致，只是 SGLang 把"层"（`layers/`）和"模型骨架"（`models/`）分成了两个目录，vLLM 目前是合在 `model_executor/` 一个目录下。

按 `_lab/out/struct_map.json` 的十个子系统口径（关键词 + AST 定位，切法与上表不同，两者不必对得上）：`attention` 217,907 行最大，其后是 `distributed` 86,399、`quantization` 75,769、`entrypoint` 62,513、`kv_cache` 32,786、`model_exec` 43,047、`disagg` 38,075、`speculative` 37,259、`scheduler` 35,054、`structured_output` 2,933 最小。和 vLLM 一样，**调度器不是代码量意义上的核心**——这是两个引擎共有的模式，`## 6` 会展开对照。

## 2. 代码地图（文件 → 职责，带行号）

按"一次请求会依次经过的层"排列：

| 层 | 文件:行 | 干什么 |
|---|---|---|
| 进程入口 | `python/sglang/launch_server.py:16` | `run_server(server_args)`：按 `encoder_only`/gRPC/Rust 开关分发到具体的服务器实现 |
| HTTP 服务装配 | `python/sglang/srt/entrypoints/http_server.py:456` | `app = FastAPI(lifespan=lifespan, ...)`：进程内唯一的 FastAPI 实例 |
| HTTP 路由 | `python/sglang/srt/entrypoints/http_server.py:894` | `@app.api_route("/generate", methods=["POST", "PUT"])` 的 `generate_request`：原生协议入口 |
| HTTP 服务启动 | `python/sglang/srt/entrypoints/http_server.py:2766` | `launch_server(server_args)`：起 uvicorn、跑警启（warmup）`/generate`、再对外宣布就绪 |
| 引擎门面 | `python/sglang/srt/entrypoints/engine.py:207` | `class Engine`：`TokenizerManager` + 调度器进程 + detokenizer 进程三件套的装配点 |
| 进程编排 | `python/sglang/srt/entrypoints/engine.py:856` | `_launch_scheduler_processes`：按 `tp_size`/`pp_size`/`dp_size` 决定起几个 `mp.Process`，还是改起 `DataParallelController` |
| 请求契约 | `python/sglang/srt/managers/io_struct.py:162` | `class GenerateReqInput`：HTTP JSON → 内部对象的第一层 |
| 请求契约 | `python/sglang/srt/managers/io_struct.py:944` | `class TokenizedGenerateReqInput`：跨进程送去调度器的载荷，token id 已经算好 |
| Tokenizer 层 | `python/sglang/srt/managers/tokenizer_manager.py:386` | `class TokenizerManager`：常驻 HTTP 进程内，负责分词、经 ZMQ 转发给调度器、收 detokenizer 的结果 |
| 调度器 | `python/sglang/srt/managers/scheduler.py:384` | `class Scheduler(SchedulerWarmupMixin, SchedulerPostTrainingMixin, SchedulerDisaggMixin)`：单个 TP rank 的事件循环主体，134 个方法 |
| 调度器事件循环 | `python/sglang/srt/managers/scheduler.py:1748` / `:1783` | `event_loop_normal` / `event_loop_overlap`：两种事件循环，后者把下一步的 CPU 调度和当前步的 GPU 执行重叠起来 |
| 调度器进程入口 | `python/sglang/srt/managers/scheduler.py:5140` | `run_scheduler_process`：`mp.Process` 的 `target`，子进程里真正跑起来的顶层函数 |
| 数据并行 | `python/sglang/srt/managers/data_parallel_controller.py:132` / `:811` | `class DataParallelController` 与 `run_data_parallel_controller_process`：`dp_size > 1` 时替代一对一的调度器进程拓扑 |
| Detokenizer | `python/sglang/srt/managers/detokenizer_manager.py:92` / `:516` | `class DetokenizerManager` 与 `run_detokenizer_process`：独立进程，把 token id 流式解码回文本 |
| 嵌入式 Rust | `python/sglang/srt/managers/rust_server.py:1`-`8` | 模块 docstring 原话："The Rust server replaces the Python api-server + `TokenizerManager` + `DetokenizerManager` stack ... running them as Rust threads **inside the scheduler process**" |
| 嵌入式 Rust 开关 | `python/sglang/srt/environ.py:1519` | `SGLANG_RUST_SERVER = EnvBool(False)`：默认关闭，opt-in |
| 全局配置 | `python/sglang/srt/server_args.py:473` | `class ServerArgs`：476 个字段的单一配置类 |
| IPC 地址簿 | `python/sglang/srt/server_args.py:9970` | `class PortArgs`：各进程间 ZMQ socket 名字的集中定义 |
| 调度批次 | `python/sglang/srt/managers/schedule_batch.py:816` / `:2011` | `class Req`（单请求运行时状态）与 `class ScheduleBatch`（一批请求的运行时状态） |
| 前缀缓存 | `python/sglang/srt/mem_cache/unified_radix_cache.py:148` | `class UnifiedRadixCache(BasePrefixCache)`：133 个方法，RadixAttention 的核心数据结构 |
| KV 缓存池 | `python/sglang/srt/mem_cache/memory_pool.py:1759` | `class MHATokenToKVPool(KVCache)`：标准 MHA 模型的物理 KV 存储 |
| kernel 注册表 | `python/sglang/kernels/registry.py:17` | `class KernelRegistry`：进程级单例，记录 `KernelSpec` 但不触发导入/编译 |
| kernel 描述 | `python/sglang/kernels/spec.py:204` | `class KernelSpec(msgspec.Struct, frozen=True)`：一个 op + 一个 backend + 一条 import path |
| kernel 选择 | `python/sglang/kernels/selector.py:38` / `:92` | `select_kernel` / `get_kernel`：无优先级排序，单 backend 直接解析，多 backend 必须显式点名 |
| 自研 kernel 包 | `python/sglang/kernels/aot/README.md:1`-`13` | 前身 `sgl-kernel`：独立 PyPI 包 `sglang-kernel`，Python import 路径仍是 `sgl_kernel` |
| vendored kernel | `python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py:1`-`2` | 版权头 `Copyright (c) 2025, Tri Dao.` + `Copyright (c) 2026, Colfax International.`——原样拷贝，不是自研 |
| vendored kernel | `python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py:1` | 版权头 `Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, ... Tri Dao.` |
| 前端 DSL | `python/sglang/lang/backend/runtime_endpoint.py:82` | `RuntimeEndpoint` 把 `srt` 当 HTTP 远端调用：`self.base_url + "/generate"` |
| 第二运行时 | `python/sglang/multimodal_gen/runtime/managers/scheduler.py:80` | `class Scheduler(SchedulerWarmupMixin, SchedulerPostTrainingMixin, SchedulerDisaggMixin)`：diffusion 版调度器，和 `srt` 的 `Scheduler` 同名但完全独立 |
| 第二运行时 | `python/sglang/multimodal_gen/runtime/server_args/server_args.py:212` | `class ServerArgs(DisaggServerArgsMixin)`：diffusion 版配置类，3,084 行 |
| 第二运行时 | `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:397` | `app = FastAPI(lifespan=lifespan)`：diffusion 独立的 FastAPI 实例，和 `srt` 的那个不是同一个进程 |

以上 30 条覆盖了"一次普通 `/generate` 请求"会经过的层。下面 6 条补上本篇 `## 0` 提到的十个子系统里、这次没在主流程出现但代码量不小的几块，方便对上后续篇目：

| 子系统 | 文件:行 | 干什么 |
|---|---|---|
| 结构化输出 | `python/sglang/srt/constrained/base_grammar_backend.py:58` | `class BaseGrammarObject`：约束解码的抽象基类，`xgrammar`/`llguidance`/`outlines` 等具体后端都实现它 |
| 结构化输出 | `python/sglang/srt/constrained/xgrammar_backend.py:208` | `class XGrammarGrammarBackend(BaseGrammarBackend)`：默认的语法后端实现之一 |
| 投机解码 | `python/sglang/srt/speculative/eagle_worker_v2.py:1050` | `class EAGLEWorkerV2(BaseSpecWorker)`：EAGLE 投机解码的 worker 实现 |
| PD 分离 | `python/sglang/srt/disaggregation/nixl/conn.py:152` | `class TransferInfo`：基于 NVIDIA NIXL 的 KV 传输连接管理，PD 分离场景下 prefill 节点向 decode 节点搬 KV |
| 分布式 | `python/sglang/srt/distributed/parallel_state.py:237` | `class GroupCoordinator`：张量/流水/专家并行各种进程组的统一封装 |
| 量化 | `python/sglang/srt/layers/quantization/fp8.py:225` | `class Fp8Config(QuantizationConfig)`：FP8 量化配置，`Fp8LinearMethod`/`Fp8MoEMethod` 是它派生出的两个执行方法类 |

以上 36 条只是骨架，`## 3` `## 4` 会把其中若干条的上下文摊开细读。

## 3. 核心数据结构

- **`ServerArgs`**（`python/sglang/srt/server_args.py:473`）——启动时构造一次的全局配置对象，476 个字段全部平铺在一个 `@dataclasses.dataclass` 里，`__post_init__`（同文件 3643 行）调用 `_run_resolution_pipeline`（3646 行），依次跑 63 个 `_handle_*` 方法做校验、归一化、跨字段推导（比如 PD 分离、CUDA Graph 配置、模型能力自动调整）。类文档字符串专门有"Adding new arguments"分节，要求新字段"放进正确的分组注释块，或新建一个分组"——集中但不是失控，`## 5` 会拆代价。
- **`PortArgs`**（`python/sglang/srt/server_args.py:9970`）——各进程间 ZMQ/管道地址的集中定义：`tokenizer_ipc_name`（detokenizer→tokenizer）、`scheduler_input_ipc_name`（tokenizer→scheduler rank 0）、`detokenizer_ipc_name`（scheduler→detokenizer）、`nccl_port`、`rpc_ipc_name`（Engine↔Scheduler 的 RPC 调用）、`metrics_ipc_name`、`tokenizer_worker_ipc_name`（多 tokenizer worker 场景）、`decoupled_spec_ipc_config`（投机解码 verifier/drafter 之间）——这是"进程之间怎么找到彼此"的唯一真源，`init_new` 静态方法（`:10002`）按 `ServerArgs` 算出所有具体地址。
- **`GenerateReqInput`**（`python/sglang/srt/managers/io_struct.py:162`）——HTTP JSON body 反序列化后的第一层对象，字段还是"用户友好"的形态（`text`/`input_ids`/`sampling_params`/`stream` 等）。
- **`TokenizedGenerateReqInput`**（`python/sglang/srt/managers/io_struct.py:944`，`msgspec.Struct`）——`TokenizerManager` 分词后打包、经 ZMQ 送去调度器进程的载荷，是**跨进程边界的第一个契约对象**；用 `msgspec.Struct` 而不是 `dataclass`，仓库自己的风格规则要求新数据容器一律走 `msgspec.Struct`（原因是要配合严格类型检查，也为未来 Rust 迁移铺路）。
- **`Req`**（`python/sglang/srt/managers/schedule_batch.py:816`）——请求到达调度器进程后包装成的**进程内**运行时对象，携带 `seqlen`、KV 已提交长度、语法 FSM 状态、投机解码计数等只有调度器关心的字段。
- **`ScheduleBatch`**（`python/sglang/srt/managers/schedule_batch.py:2011`）——一批 `Req` 的运行时状态容器，`prepare_for_extend`/`prepare_for_decode`/`retract_decode`/`merge_batch` 等方法定义了 prefill、decode、显存不足时抢占（retraction）分别怎么改写这个批次。
- **`UnifiedRadixCache`**（`python/sglang/srt/mem_cache/unified_radix_cache.py:148`）——RadixAttention 的核心数据结构，133 个方法，`match_prefix`/`insert`/`evict` 是前缀缓存三个最基本的操作；名字里的"Unified"意味着它要同时应付普通 KV、混合 Mamba、滑窗注意力等多种缓存布局，这也是为什么它比朴素版 `RadixCache`（`python/sglang/srt/mem_cache/radix_cache.py`，544 行）大 5 倍还多。
- **`MHATokenToKVPool`**（`python/sglang/srt/mem_cache/memory_pool.py:1759`）——标准多头注意力模型的物理 KV 存储实现，`KVCache` 抽象基类的具体子类之一（同文件里还有 MLA、Mamba 等专用池）。
- **`KernelSpec`**（`python/sglang/kernels/spec.py:204`，`msgspec.Struct(frozen=True)`）与 **`KernelRegistry`**（`python/sglang/kernels/registry.py:17`）——kernel 层的元数据契约：一个算子 id（`"<group>.<name>"`）+ 一个 backend 枚举 + 一条 `"module:attr"` 形式的导入路径，`register_kernel` 只记录元数据，不触发 `torch` 导入也不触发 JIT 编译，直到真正被调用。
- **`AttentionBackend`**（`python/sglang/srt/layers/attention/base_attn_backend.py:36`，`ABC`）——所有注意力后端（`FlashAttentionBackend`/`AiterAttnBackend`/`AscendAttnBackend`/`DeepseekSparseAttnBackend` 等）共同实现的抽象基类，统一了 `init_forward_metadata`/`forward_extend`/`forward_decode`/`init_cuda_graph_state` 这套方法签名——`Scheduler` 和 `ModelRunner` 只认这一层接口，不关心具体是哪个硬件后端。
- **`BasePrefixCache`**（`python/sglang/srt/mem_cache/base_prefix_cache.py:235`，`ABC` + `PrefixCacheTrait`）——`UnifiedRadixCache`/`RadixCache`/`SWARadixCache`/`MambaRadixCache` 共同的抽象基类，统一 `match_prefix`/`insert`/`evict`/`cache_finished_req` 接口；有多少种"缓存要感知的额外结构"（滑窗、Mamba 状态、混合注意力），就有多少个具体子类，但调度器只对着这一层接口编程。
- **`KVCache`**（`python/sglang/srt/mem_cache/memory_pool.py:1628`，`abc.ABC`）——物理 KV 存储的抽象基类，`MHATokenToKVPool`（`:1759`）只是众多子类之一，同文件里还有面向 MLA、Mamba 状态、DeepSeek V4 混合缓存的专用实现（`kv_cache_configurator.py` 按模型架构在启动时选择具体走哪一个）。

## 4. 主流程走读

以 `POST /generate`、非流式、单机单卡、不开嵌入式 Rust、不经 `sgl-model-gateway` 为基准路径：

1. **进程 0：命令行启动。** `sglang serve <model>`（或 `python -m sglang.launch_server`）走到 `python/sglang/launch_server.py:16` 的 `run_server`，按 `server_args.encoder_only`/`smg_grpc_mode`/`grpc_mode` 分支，默认场景落到 `sglang.srt.entrypoints.http_server.launch_server`。
2. **HTTP 服务装配。** 模块加载期就建好了唯一的 `app = FastAPI(...)`（`python/sglang/srt/entrypoints/http_server.py:456`），随后一长串 `@app.api_route(...)` 装饰器把 79 个不同路径（据 `_lab/out/api_surface.json` 的 `n_unique_paths`）挂上去——原生协议（`/generate`/`/encode`/`/classify`）、OpenAI 兼容（`/v1/chat/completions` 等）、Anthropic 兼容（`/v1/messages`）、管理端点（`/flush_cache`/`/start_profile`）都在同一个 app 上。
3. **`launch_server` 起服务。** `python/sglang/srt/entrypoints/http_server.py:2766` 的 `launch_server` 内部会先构造 `Engine`（下一步），再用 uvicorn 把 FastAPI app 跑起来，启动过程里还会发一次警启用的 `/generate` 请求，确认整条链路都通了之后才对外宣布就绪。
4. **`Engine` 装配三件套。** `python/sglang/srt/entrypoints/engine.py:207` 的 `Engine.__init__` 依次调用 `_launch_scheduler_processes`（`:856`）与 `_launch_detokenizer_subprocesses`（`:973`），再构造 `TokenizerManager`。`_launch_scheduler_processes` 内部按 `dp_size > 1` 或专家并行的"scale"模式二选一：不满足就按 `pp_rank × tp_rank` 逐个起 `mp.Process`（`target=run_scheduler_process`），满足就只起一个 `DataParallelController` 进程，由它再去管理下面的调度器进程。
5. **HTTP 请求进来。** `generate_request`（`python/sglang/srt/entrypoints/http_server.py:894`）把请求体反序列化成 `GenerateReqInput`（`io_struct.py:162`），交给 `_global_state.tokenizer_manager.generate_request`。
6. **Tokenizer 层：分词 + 发往调度器。** `TokenizerManager`（`tokenizer_manager.py:386`）持有一个 `zmq.asyncio.Context`，`recv_from_detokenizer` 是 `zmq.PULL` 套接字（连到 `port_args.tokenizer_ipc_name`），`send_to_scheduler` 是 `zmq.PUSH` 套接字（连到 `port_args.scheduler_input_ipc_name`，`tokenizer_manager.py:548`-`560`）。分词完成后打包成 `TokenizedGenerateReqInput`（`io_struct.py:944`），经 `send_to_scheduler.send_pyobj(...)` 推给调度器进程——这是**第一次跨进程边界**。
7. **进程 1：调度器事件循环。** `run_scheduler_process`（`scheduler.py:5140`）在子进程里构造 `Scheduler`（`scheduler.py:384`）后跑 `event_loop_normal`（`:1748`）或 `event_loop_overlap`（`:1783`，把"准备下一步 CPU 侧调度"和"当前步 GPU 前向传播"两件事重叠执行的版本）。循环体反复做：从 ZMQ 收新请求 → 调度策略选一批 `Req` 组成 `ScheduleBatch`（`schedule_batch.py:2011`）→ 前缀缓存查找（`UnifiedRadixCache.match_prefix`，`unified_radix_cache.py` 内）→ 分配/回收 KV 槽位。
8. **模型执行。** `Scheduler` 把 `ScheduleBatch` 交给它持有的 `TpModelWorker`（`self.tp_worker = TpModelWorker(**worker_kwargs)`，`scheduler.py:924`）——这一步**不跨进程**：单个 TP rank 内，调度和模型执行是同一个 Python 进程里的直接方法调用，不经 IPC；attention 后端（`srt/layers/attention/` 下具体的 `FlashAttentionBackend` 等）在这一步被调用，进而可能落到 `kernels/ops/` 下的具体算子实现。
9. **结果回填 + 增量 detokenize 分发。** 一步前向传播产出的新 token id 被打包，经 `send_to_detokenizer`（`zmq.PUSH`，连到 `port_args.detokenizer_ipc_name`）推给独立的 detokenizer 进程。
10. **进程 2：detokenizer。** `run_detokenizer_process`（`detokenizer_manager.py:516`）跑起 `DetokenizerManager`（`:92`），把新 token id 增量解码成文本片段，再经 `zmq.PUSH`（连到 `port_args.tokenizer_ipc_name`）推回 `TokenizerManager` 所在的 HTTP 进程——**第三次跨进程边界**（tokenizer→scheduler→detokenizer→tokenizer，构成一个环，不是直线）。
11. **流式响应。** `TokenizerManager.recv_from_detokenizer`（`zmq.PULL`）收到结果，`generate_request` 的 `async for` 循环把它 `yield` 成 SSE 数据帧（`b"data: " + dumps_json(out) + b"\n\n"`）或攒成一次性 JSON 响应，返回给 HTTP 客户端。
12. **可选：嵌入式 Rust 接管前三层。** 若 `SGLANG_RUST_SERVER=True`（`environ.py:1519`，默认 `False`），第 5-6、9-11 步的 Python 实现会被 `python/sglang/srt/managers/rust_server.py` 里挂进来的 Rust 线程替换——注意这些 Rust 线程**运行在调度器进程内部**（该文件 docstring 原话），不是新开一个操作系统进程；对应的 PyO3 扩展模块由 `rust/sglang-server` crate 编译而来，`python-module = "sglang.srt.rust_extensions._server"`（`rust/sglang-server/Cargo.toml`）。
13. **可选：多实例部署经 `sgl-model-gateway`。** 大规模部署下，客户端先打到 `sgl-model-gateway`（独立 Rust 进程/服务，README 自称"control plane"+"data plane"），由它做 worker 注册发现、健康检查、PD 分离路由、多协议代理，再把请求转发给某一个具体的 SGLang 实例的 HTTP 端口——这时候上面第 1-11 步是在被选中的那个实例内部重演一次，网关本身不跑模型。

十二步（含两个可选分支）里，默认路径至少跨 3 次进程边界（tokenizer↔scheduler、scheduler↔detokenizer 各算一次收发），如果算上多实例部署的网关转发，最多可以看到 5 层进程：网关 → tokenizer 进程 → 调度器进程（每 TP rank 一个）→（原路）detokenizer 进程 → 网关 → 客户端。

## 5. 设计决策与代价

### 决策一：`ServerArgs` 是一个 476 字段的单一巨类，不拆成多个子配置对象

- **为什么这么设计**：一个平铺的 `dataclass` 意味着任何地方要读配置只需要 `server_args.xxx`，不用先弄清楚这个字段属于 `ModelConfig` 还是 `ParallelConfig` 还是 `CacheConfig`；`__post_init__` 的 63 个 `_handle_*` 方法可以在同一个命名空间里做跨领域的联动校验（比如 PD 分离的开关会联动 CUDA Graph 的默认值，`_apply_inkling_prefill_cuda_graph_default` 这类方法名就是这种联动的产物）——如果字段分散在十几个子配置对象里，跨对象联动就要么在更外层写胶水代码，要么用回调/观察者模式，复杂度不会消失只会转移。
- **不这样会怎样**：代价是文件本身变成审阅（review）的物理瓶颈——10,142 行的单文件，`git diff` 稍微改大一点就很难看出改动边界；类文档字符串里专门写"Place the field in the right section"这条规则，说明维护者已经意识到"平铺"和"可维护"之间的张力，用注释分组（`# Model and tokenizer`、`# LoRA` 等）当轻量级的软边界，而不是用真正的子类型系统强制边界。
- **什么时候可以不这样**：如果字段之间跨领域联动很少、大多数字段只被一个子系统读取，拆成 `ModelConfig`/`ParallelConfig`/`CacheConfig` 这类强类型子对象（`vllm/config/` 那种做法，`## 6` 细讲）能让每个子对象保持在几百行的可读规模，代价是构造 `VllmConfig` 时要多一层"哪个字段该放进哪个子对象"的设计决策，且新增一个跨对象联动规则要么塞进更外层，要么在两个子对象之间开一条读写通道。

### 决策二：把原本独立发布的 `sgl-kernel` 迁进主仓库、变成 `kernels/aot/`

- **为什么这么设计**：`kernels/README.md`（`python/sglang/kernels/README.md:1`-`4`）明确引用了 RFC #29630，把 `kernels/` 定位为"unified kernel namespace"——统一的可调用 kernel 导入面（`from sglang.kernels.ops.layernorm import rmsnorm`）。在此之前，`sgl-kernel`（PyPI 上仍叫 `sglang-kernel`）是独立版本号、独立发布节奏的包，运行时代码要跟着它的版本更新走；迁进主仓库后，运行时和它调用的 kernel 版本天然锁定在同一个 commit，不再有"运行时新特性依赖的 kernel 版本还没发布"的错位窗口。
- **不这样会怎样**：如果继续保持独立仓库/独立发布，`kernels/README.md` 里描述的 `BaseFusedOp` 统一调度契约（`forward_native`/`forward_triton`/`forward_cuda` 等按 backend 分发）就必须靠版本号锁定 + CI 交叉测试来保证两边同步，而不是靠"同一次 git commit 天然同步"——版本锁定这件事本身就是额外的工程负担，也是不少多仓库项目（kernel 库 + 运行时分离）长期头疼的同步问题。
- **什么时候可以不这样**：如果 kernel 库的目标用户主要是"别的推理框架也想复用这批 kernel"（对外部生态友好），保持独立发布、独立版本号反而更合适——`sgl-kernel` 迁入之后仍然保留了独立的 PyPI 包名 `sglang-kernel` 和 `pyproject.toml`（`python/sglang/kernels/aot/pyproject.toml`），说明维护者并没有放弃"让外部项目单独 `pip install sglang-kernel`"这条路，只是把开发时的源码位置和运行时代码放到了同一个仓库里。

### 决策三：直接拷贝 Tri Dao 团队的 FlashAttention-4（CuTeDSL）源码进仓库，而不是走 pip 依赖

- **为什么这么设计**：`flash_attn/cute/README.md` 里写的开发方式是 `git clone https://github.com/Dao-AILab/flash-attention.git` 后 `pip install -e "flash_attn/cute[dev]"`——上游本身还在快速迭代（CuTeDSL 这套 API 面向 Hopper/Blackwell，跟着 CUTLASS/CuTe 版本演进），SGLang 要用到未发布的修复或者要针对自己的调用方式做适配（`fa4_sm120/` 目录名暗示这是给 SM120，即 Blackwell GeForce/DGX Spark 消费级卡型做的专门适配，上游未必有对应发布），拷贝一份能立刻拿到最新代码并本地打补丁，不用等上游发版、也不用等上游接受自己的 PR。
- **不这样会怎样**：拷贝的代价是**版权与更新责任都转移到了 SGLang 自己身上**——版权头写着 2025/2026 年份和 Tri Dao、Jay Shah 等原作者姓名（`flash_fwd_sm100.py:1`-`2`、`fa4_sm120/flash_fwd.py:1`），上游后续的 bug 修复不会自动同步过来，需要 SGLang 团队自己盯着上游改动、手动回合并；仓库里没有找到自动化的上游同步脚本或 vendoring 清单（**本库推断**，只检索了常见位置，未穷尽全仓库搜索），这意味着"这份拷贝和上游差了多少个 commit"目前只能靠人工经验判断。
- **什么时候可以不这样**：如果目标硬件的 kernel 已经在上游发布稳定版本、且 SGLang 不需要任何本地修改，直接 `pip install flash-attn-4` 走依赖声明更省心——`flash_attn/cute/README.md` 里给出的正是这条"外部用户"路径（`pip install flash-attn-4`），只是 SGLang 自己作为高频修改方选择了拷贝而不是依赖。

### 决策四：`multimodal_gen` 是独立于 `srt` 的第二套调度器/配置/HTTP 服务，不复用

- **为什么这么设计**：图像/视频生成模型（扩散模型）的执行模式和自回归 LLM 本质不同——没有逐 token 的 KV 缓存增长和前缀复用，取而代之的是多步去噪（denoising steps）、按时间步调度的 pipeline stage（`multimodal_gen/runtime/pipelines_core/stages/`），`Scheduler` 要管理的资源单位是"整条流水线的一次调用"而不是"一个 token 的生成"；`multimodal_gen/runtime/managers/scheduler.py:80` 的 `Scheduler` 类文档字符串写的是"Runs the main event loop for the rank 0 worker"，方法列表里是 `_dispatch_generation`/`_execute_generation_grouped` 这类批量生成动作，和 `srt` 的 `Scheduler` 方法列表（`init_all_attention_backends`/`init_memory_pools`）几乎没有交集。硬套同一个 `Scheduler` 类，等于要在一个类里同时塞两种完全不同的资源模型。
- **不这样会怎样**：如果强行复用 `srt` 的调度器和 KV 缓存框架，扩散模型的"多步去噪、无 KV 增长"场景要么被迫伪装成某种退化的自回归批次（浪费大量为 KV 缓存设计的代码路径），要么在 `srt.Scheduler` 内部再分叉出一整套 `if is_diffusion_model` 分支——比新建一个独立类的认知负担更高，因为读者要在同一个类里区分两套完全不同的心智模型。
- **什么时候可以不这样**：如果只是"LLM 模型支持读取图片/视频作为输入"（多模态输入，不是生成图片/视频），并不需要独立运行时——那属于 `srt/multimodal/` 和 `srt/managers/mm_utils.py` 的范畴，仍然复用 `srt` 的调度器和 KV 缓存，只是在 prefill 阶段多一步视觉编码。两者不要混淆：`multimodal_gen` 的"multimodal"指的是**生成**多模态内容，不是**理解**多模态输入。

### 决策五：Rust 有两种集成方式，且刻意做成两条互相独立的路径

- **为什么这么设计**：嵌入式 Rust（`rust/sglang-server` crate → PyO3 扩展 → `rust_server.py` 在调度器进程内跑 Rust 线程）解决的是"单个 SGLang 实例内，HTTP 解析/tokenize/detokenize 这些 CPU 密集工作要不要用更快的语言做"；`sgl-model-gateway` 解决的是完全不同层面的问题——"多个 SGLang 实例之间怎么做流量路由、健康检查、PD 分离编排"，这天然是一个独立于任何单个推理实例的控制/数据平面，装进单个实例的进程里没有意义。两个问题的作用域不同，做成两个独立组件而不是一个大而全的 Rust 层，避免了"单机加速"和"集群编排"这两种关注点纠缠在一起。
- **不这样会怎样**：如果把两者合并成一个组件，要么 `sgl-model-gateway` 要反过来嵌入每个 worker 进程（失去"独立进程做路由决策，某个 worker 挂了不影响路由层"的隔离性），要么嵌入式 Rust 服务器要长出跨实例路由的能力（职责越界，一个本该只管"单实例内加速协议解析"的组件被迫理解集群拓扑）。
- **什么时候可以不这样**：单机单实例部署、且实测 HTTP 层不是瓶颈时，两者都可以不开——嵌入式 Rust 默认关闭（`SGLANG_RUST_SERVER=False`），`sgl-model-gateway` 本身就是可选的独立部署组件，不部署就是默认的"客户端直接打单个 SGLang HTTP 端口"路径（本篇 `## 4` 走读的正是这条最简路径）。

## 6. 同位对照：vLLM 在同一位置怎么做

（vLLM 一侧引用取自 `_src/vllm` @ `7ca49fbe`，与 `[[01-vLLM-全景与代码地图]]` 使用同一次 clone。）

- **配置对象的拆分粒度完全相反。** SGLang 把几乎所有配置字段塞进一个类：`ServerArgs`（`python/sglang/srt/server_args.py:473`），9,401 行、476 字段。vLLM 的 `EngineArgs`（`vllm:vllm/engine/arg_utils.py:424`）本身只是 CLI 参数解析层，只有 2,909 行，真正的配置值分散存放在 `vllm/config/` 目录下约 32 个文件、合计 14,083 行的多个 `dataclass`（`ModelConfig`/`CacheConfig`/`ParallelConfig`/`SchedulerConfig` 等）里，`EngineArgs.create_engine_config()` 之类的方法负责把 CLI 参数组装成这些子对象。两边字段总量级接近（vLLM 的配置代码行数其实更多），差别在于 **SGLang 选择了"一处平铺"，vLLM 选择了"多处强类型拆分"**——`## 5` 决策一分析的正是这组权衡的两端各自落在哪一边。
- **两边都在往运行时里"收 kernel"，但收的东西不同。** vLLM 这次 clone 里也有一个 `vllm/kernels/` 目录（`vllm/kernels/helion/`、`vllm/kernels/triton/`），但里面是 Helion/Triton 这类**用 Python 写、JIT 编译**的 kernel；CUDA/C++ 编写的算子仍然独立放在仓库顶层的 `csrc/`，通过 `TORCH_LIBRARY_FRAGMENT` 注册成 `torch.ops._C.*`（`vllm:csrc/torch_bindings.cpp:21`），不在 `vllm/kernels/` 这个 Python 命名空间里。SGLang 的 `kernels/` 则把 **CUDA/C++ 源码本身**（`kernels/aot/csrc/`、`kernels/jit/csrc/`）也收进了同一个目录树，Python 层的 `kernels/ops/*` 只是分发外壳——SGLang 的"收纳"覆盖到了编译产物的源头，vLLM 目前的"收纳"止步于 Python/JIT 这一层。
- **Rust 集成点的数量和性质不同。** vLLM 只有一种 Rust 路径：`rust/` 下的独立可执行文件，作为**独立操作系统进程**跑 HTTP 前端，通过 `ManagedEngineHandle::spawn` 托管一个 Python "headless"引擎子进程（`vllm:rust/src/cmd/src/main.rs:123`-`145`），默认关闭（`vllm:vllm/envs.py:165` 的 `VLLM_USE_RUST_FRONTEND: bool = False`）。SGLang 有两种：`rust_server.py` 是**进程内的 PyO3 扩展**（Rust 代码以线程形式跑在 Python 调度器进程里，不是独立进程），同样默认关闭（`SGLANG_RUST_SERVER = EnvBool(False)`，`environ.py:1519`）；`sgl-model-gateway` 才是独立进程，但它对应的不是 vLLM 的"HTTP 前端"，而是更接近"多实例负载均衡器"这个更高的层级，vLLM 这次 clone 里没有找到对应组件（**未查证**是否存在于其他分支或子项目，只是没在这次 clone 范围内找到）。两边"默认关闭 Rust 加速路径"这个决策是一致的，但集成的解剖位置不同。
- **两边都存在"新旧模型代码布局并存"的重构信号，但没在 SGLang 侧深入核实细节。** vLLM 侧的证据是 `vllm/model_executor/models/`（老布局）与 `vllm/models/`（新的按硬件拆分布局）并存（`vllm:vllm/model_executor/models/registry.py:1005`-`1008`）；SGLang 侧 `srt/models/` 目录是单一扁平布局（254 个文件、160,964 行，一个模型家族一个 `.py` 文件），本次没有找到类似的新旧并存证据（**未查证**是否存在于更细的子目录里）。
- **"第二运行时"这件事只在 SGLang 侧看到。** `multimodal_gen/` 是完全独立于 LLM 服务运行时的图像/视频生成引擎，vLLM 这次 clone 的顶层目录里没有找到同等规模、独立 `Scheduler`/`ServerArgs`/HTTP app 的第二套运行时（`find . -iname "*diffus*"` 在 `_src/vllm` 下无匹配）——**这不代表 vLLM 生态完全没有扩散模型支持**（vLLM 有独立的 `diffusers`-integration 类项目，不在这次 clone 的仓库范围内），只是在 `vllm-project/vllm` 这一个仓库里没有对应体量的实现，标注为**未查证**上游是否另有仓库承载。

## 7. 踩坑与反直觉

- **"SGLang"这个名字最容易造成的误解**：以为这个项目的核心是"一门给 LLM 编程的结构化生成语言"。实测下来，`lang/` 只有 4,644 行，占 `python/sglang` 主要包总量的 0.36%；`srt/` 才是真正的重头戏。今天大多数生产部署根本不会碰 `sgl.gen()` 这套 DSL，直接打 `srt` 暴露的 OpenAI 兼容 HTTP 接口。
- **`kernels/` 目录名字暗示"这里都是 SGLang 自己写的"，其实至少两类子目录是原样拷贝。** `kernels/ops/attention/flash_attn/cute/` 和 `kernels/ops/attention/fa4_sm120/` 的每个文件顶部都还留着 Tri Dao 等原作者的版权声明——如果只看目录路径 `python/sglang/kernels/...` 判断"这是 SGLang 的自研 kernel"，会得出错误结论。判断一个 kernel 文件是自研还是 vendored，目前只能靠打开文件读版权头，仓库里没有统一的清单（`## 8` 会把这个写成可改进点）。
- **`server_args.py` 一万行不是"没人管的技术债"。** 类文档字符串里专门有贡献指南，`__post_init__` 编排了 63 个命名规范的 `_handle_*` 方法——这是刻意为之的集中式设计，不是失控增长后的产物；批评它"应该拆分"之前，至少要先承认它是经过设计的，代价是可衡量的（review 成本），不是"混乱"两个字能概括的。
- **Rust 出现在这个仓库的两个位置，性质完全不同，不要混着讲。** `rust/sglang-server`（嵌入式 PyO3 扩展，跑在调度器进程内的线程）和 `sgl-model-gateway`（独立部署的网关服务）都叫"Rust"，但一个是"给单实例加速协议解析"，一个是"给多实例做流量路由"——审计"SGLang 用了多少 Rust"时，把两者的行数简单相加（156,078 行）会掩盖它们完全不同的部署形态和开关方式。
- **`multimodal_gen` 不是"让 LLM 支持看图/生成图文混排回复"。** 那是 `srt/multimodal/` 的范畴，仍然在 `srt` 运行时内、复用同一个调度器和 KV 缓存。`multimodal_gen` 是完全独立的**图像/视频生成**引擎（扩散模型），两者名字都带"multimodal"，职责完全不同，容易在快速浏览目录树时搞混。
- **"attention 子系统 217,907 行"读成"SGLang 的注意力实现有 21.8 万行独有代码"是过度推断。** `struct_map.json` 的子系统统计按关键词+AST 定位，不区分文件来源；这 21.8 万行里，`flash_fwd_sm100.py`（5,610 行）、`fa4_sm120/flash_fwd.py`（4,552 行）这类 vendored 文件占了相当比例。读这类统计数字时，要意识到"某子系统行数最大"和"SGLang 团队在这个子系统上投入的自研工作量最大"是两个不同的结论，中间隔着一层"这些代码到底是谁写的"。

## 8. 可改进点

以下几条标注为**本库推断**，未提交 issue 或 PR 核实维护者是否已在计划中，是读代码时观察到的方向：

1. **vendored kernel 目录缺一份集中清单。** 现在判断"`kernels/ops/` 下哪些子目录是原样拷贝、来自哪个上游仓库、对应哪个上游 commit/版本"只能逐文件打开看版权头。一份 `VENDORED.md`（列出目录、上游仓库地址、拷贝时的上游版本/commit、license）能让贡献者和安全审计者不用逐文件排查就知道"这段代码谁负责同步"。
2. **`ServerArgs` 的分组约束目前只停留在文档层面，没有运行时校验。** 类文档字符串要求新字段"放进正确的分组注释块"，但这是靠 review 时人工检查的软约束——没看到自动化脚本检查"新增字段是否落在某个 `# XXX` 注释块下面"。476 个字段的规模下，人工约束的可靠性会随时间推移而下降；一个轻量的 lint 脚本（检查每个字段紧邻的注释块归属）能把这条规则变成可自动执行的。
3. **两条 Rust 集成路径之间缺少选型指南。** `SGLANG_RUST_SERVER`（单实例内加速）和 `sgl-model-gateway`（多实例路由）功能上有一定重叠（两者都涉及请求路由、协议处理），但本次检索没找到一篇文档明确回答"单机部署该不该开 `SGLANG_RUST_SERVER`，多机部署是否还需要额外部署 `sgl-model-gateway`，两者能不能同时开"——一张决策表能减少用户在两个开关之间的选型困惑。
4. **`srt` 和 `multimodal_gen` 各自维护一份 `ServerArgs`，字段有概念重叠但没有共享基类。** 两边都有 `attention_backend`、量化相关配置的字段（`multimodal_gen/runtime/server_args/server_args.py:212` 的 `_adjust_quant_config`、`_normalize_attention_backend_name` 方法名和 `srt/server_args.py` 里同类字段的处理逻辑在概念上高度相似），但 `struct_map.json` 显示这是两个完全独立的类定义，没有共享的基类或 mixin。长期看，两边配置语义独立演进，容易在字段名相同但含义细微差异的场景下产生混淆（比如某个量化选项在两边的合法取值范围不一致而没有文档说明）。

## 9. 自测题与延伸阅读

**闭卷自测**（不看正文，能答上来才算过）：

1. `python/sglang` 下 `srt/`、`lang/`、`kernels/`、`multimodal_gen/` 四个子包里，哪一个是原本给这个项目起名"Structured Generation Language"的那部分？它现在占整体代码量的大致比例是多少？
2. `python/sglang/kernels/ops/attention/flash_attn/cute/` 目录下的代码是 SGLang 团队自己写的吗？怎么用一行代码内的证据判断？
3. 一次默认配置下的 `/generate` 请求，从 HTTP 进来到文本返回，最少跨几次进程边界？分别发生在哪两个 manager 之间？
4. `SGLANG_RUST_SERVER` 打开之后，Rust 代码是作为独立操作系统进程运行，还是运行在某个已有 Python 进程内部？和 `sgl-model-gateway` 的部署形态有什么本质区别？
5. `ServerArgs`（`srt/server_args.py`）和 `multimodal_gen` 自己的 `ServerArgs` 是同一个类吗？为什么 `multimodal_gen` 需要一套独立的调度器而不是复用 `srt.Scheduler`？
6. `_lab/out/struct_map.json` 里 `attention` 子系统的行数最大，这是否等价于"SGLang 团队在 attention 上投入的自研代码量最大"？为什么？

**延伸阅读**（双链只取自 `_PLAN.md` §6 名册）：

- [[02-SGLang-Scheduler事件循环]] —— 本篇 `## 4` 只走读到 `event_loop_normal`/`event_loop_overlap` 两个入口，`Scheduler` 134 个方法、`ScheduleBatch` 的 prefill/decode/抢占具体怎么改写批次，留给那一篇逐行拆。
- [[09-SGLang-ServerArgs旋钮全景]] —— 本篇 `## 5` 决策一只分析了"为什么集中成一个类"这个架构选择，476 个字段具体分几组、每组解决什么问题、63 个 `_handle_*` 方法各自的校验逻辑，留给那一篇按分组过一遍。
- [[07-SGLang-前端DSL与编程模型]] —— 本篇 `## 0` `## 7` 只提到 `lang/` 只有 4,644 行、是个 HTTP 客户端外壳，`sgl.gen()`/`sgl.select()` 这套编程模型具体怎么把 Python 控制流编译成对 `srt` 的调用序列，留给那一篇展开。
- [[01-vLLM-全景与代码地图]] —— `## 6` 同位对照的另一半来源，两篇建议对照读：同样是"全景与代码地图"定位，同样按"一次请求穿过几个进程"的顺序画代码地图，方便直接比较两个引擎在同一个决策点上的取舍。
- [[_PLAN]] —— 第 1、2 节的诚实标准和取证基准定义；本篇给出的所有取证基准、双链、代码引用是否真实可核验，都以它和仓库根目录的 `_verify.py` 为准。
