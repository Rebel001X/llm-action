# TensorRT-LLM

> **本篇取证基准**：`tensorrt-llm` @ `75b023cd`（2026-08-22）
> **一句话**：编译式引擎已经不编译了，默认反而是 eager PyTorch

## 0. 结论先行

- **它不再是"编译式"引擎了。** 这个快照里，把模型提前编译成 `.engine` 文件的传统 TensorRT 路线已经被整体移除，不是"降权"、不是"仍可选但不推荐"——是代码里已经找不到了：没有 `trtllm-build` 命令行入口（`pyproject.toml:558`-`559` 的 `console_scripts` 只剩 `trtllm-bench` / `trtllm-serve` / `trtllm-eval`），没有 `TrtLlmArgs` 类，`tensorrt_llm/models/` 下也只剩通用基类。官方migration文档写得直白：`docs/source/legacy/tensorrt-backend-removal.md:4`-`6`"The TensorRT engine backend has been removed. PyTorch is now the sole execution backend"（文档所述）。
- **`LLM` 类现在只接受两个 `backend` 值**：`"pytorch"`（默认）和 `"_autodeploy"`（beta，且已在被弃用——`tensorrt_llm/commands/serve.py:950`-`952` 的 CLI 帮助文本直接写"deprecated and will be discontinued"）。`tensorrt_llm/llmapi/llm.py:391`-`399` 的 `if/elif/else` 里，任何其它 `backend` 值都会 `raise ValueError`——包括曾经的 `"tensorrt"`。
- **`_engine_dir` 恒为 `None`。** `tensorrt_llm/llmapi/llm_utils.py:448` 的 `CachedModelLoader.__call__()` 对 pytorch/`_autodeploy` 两条可达路径都是 `return None, self._hf_model_dir`——没有编译产物,直接读 HF checkpoint。`tensorrt_llm/llmapi/llm.py:1586`-`1630` 里那段读 `self.args.build_config`、校验 `max_seq_len`/`max_beam_width` 的旧逻辑,由于前面 `tensorrt_llm/llmapi/llm.py:1575`-`1584` 已经对 `backend in ["pytorch", "_autodeploy"]` 直接 `return`,在当前唯一可达的两条 backend 上是**永远执行不到的死代码**（`TorchLlmArgs`/`AutoDeployLlmArgs` 都没有 `build_config` 字段,真跑到那里会 `AttributeError`）。
- **in-flight batching 的默认实现在 C++,不在 Python。** `tensorrt_llm/_torch/pyexecutor/_util.py:3082`-`3094` 默认构造的是 `BindCapacityScheduler`/`BindMicroBatchScheduler`,内部包的是 nanobind 绑定的 `tb_internal.algorithms.CapacityScheduler`（绑定点 `cpp/tensorrt_llm/nanobind/batch_manager/algorithms.cpp:87`,C++ 实现 `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:668`）。存在纯 Python 的 `PyCapacityScheduler`/`SimpleUnifiedScheduler`,但要显式打开 `scheduler_config.use_python_scheduler` 才会走那条路（`tensorrt_llm/_torch/pyexecutor/_util.py:3059`）。
- **"编译"这个词今天基本等于 CUDA Graph 捕获,而不是 AOT 引擎构建。** `torch_compile_config` 默认是 `None`（`tensorrt_llm/llmapi/llm_args.py:5407`-`5408`,status="prototype"）,真正 torch.compile/Inductor 是选装项;默认打开的只有 decode 阶段的 CUDA Graph 捕获（`cuda_graph_config` 的 `default_factory=CudaGraphConfig`,`tensorrt_llm/llmapi/llm_args.py:5180`,status="beta"）。
- **这条线索最反直觉的地方是它和 vLLM 的对照会反过来**:vLLM V1 的 `CompilationConfig.mode` 默认解析成 `VLLM_COMPILE`（`vllm:vllm/config/compilation.py:447`-`459` 的文档字符串原话:"None: If None, we will select the default compilation mode. For V1 engine this is 3"（对应 `VLLM_COMPILE`）——也就是说"号称解释式"的 vLLM 默认在编译,"号称编译式"的 TensorRT-LLM 默认不编译。见 `## 6`。

**速查:关键问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| TensorRT engine 编译路线还在不在?现在走哪条? | `## 0`、`## 5` 决策 2、`## 7` 踩坑 1 |
| `LLM` → Executor → C++ runtime 怎么分层? | `## 4` |
| in-flight batching 在 Python 还是 C++ 实现? | `## 3.2`、`## 4` ⑤、`## 5` 决策 3 |
| 60 条路由怎么分族?`add_api_route` 这种注册方式利弊? | `## 2`、`## 5` 决策 4 |
| 为什么要编译/不编译会怎样/什么时候编译是负担? | `## 5` |
| 和 vLLM/SGLang 比,"编译式 vs 解释式"到底谁更编译? | `## 6` |

## 1. 它在系统里的位置

TensorRT-LLM(以下简称 TRT-LLM)是 NVIDIA 官方的旗舰推理栈,定位是"NVIDIA GPU 上的优化 LLM 推理库"（`AGENTS.md` 开篇原话,文档所述）。这个仓库在体量上是 `_lab/repo_stats.json` 记录的 12 个引擎里最大的一个:总行数 2,452,198,Python 产品码 792,216,CUDA 文件 429 个（全场最多),C/C++ 合计 598,382 行。这个体量本身就是一条线索——一个真正只做"薄封装 + eager PyTorch"的项目不需要 429 个 CUDA 文件和近 60 万行 C/C++。

它在系统里扮演两个角色:

1. **一个 Python 库**:`from tensorrt_llm import LLM` 之后本地跑 `generate()`,单进程或 MPI/Ray 多进程管理多 GPU。
2. **一个独立服务进程**:`trtllm-serve <hf_model>` 启动一个 OpenAI 兼容 HTTP 服务,底下可能是单机、张量并行、甚至 PD 分离的多节点集群。`tensorrt_llm/serve/openai_server.py`（单机/聚合服务）、`openai_disagg_server.py`（PD 分离网关）、`coordinator_server.py`（多副本路由协调器)是三个不同粒度的 FastAPI 应用,各自独立注册路由,详见 `## 2`。

`pyproject.toml:557`-`559` 声明了三个 console_scripts 入口,对应 `tensorrt_llm/commands/` 下三个各司其职的 CLI 命令(`AGENTS.md` 的 Common Commands 表,文档所述):`trtllm-serve`(启动 HTTP 服务,入口 `tensorrt_llm/commands/serve.py`)、`trtllm-bench`(吞吐/延迟基准测试驱动,入口 `tensorrt_llm/commands/bench.py:64` 的 `main()`)、`trtllm-eval`(跑标准评测集,入口 `tensorrt_llm/commands/eval.py:143` 的 `main()`)。这三个命令共享同一套 `LLM`/`TorchLlmArgs` 构造逻辑,不是三套独立实现——`trtllm-eval` 直接 `from .. import LLM as PyTorchLLM`(`tensorrt_llm/commands/eval.py:21`,`llm_cls = PyTorchLLM` 在 `tensorrt_llm/commands/eval.py:189`);`trtllm-bench` 本身只是参数解析层,真正的 `LLM` 构造下沉到 `tensorrt_llm/bench/benchmark/utils/asynchronous.py:29` 的 `from tensorrt_llm import LLM, SamplingParams`。三个命令最终殊途同归到同一个 `LLM` 类,和 `## 4` 走读的主流程只是"谁来发起请求"这一层不同。

**两条后端路线,现在的答案**(源码为证,`## 0` 已给出核心引用):历史上 TRT-LLM 有两条根本不同的执行路线——(a) 把 HF checkpoint 转换成 TRT-LLM checkpoint 格式,再用 `trtllm-build` 编译出针对特定 GPU 架构、特定并行度、特定精度的 `.engine` 文件,运行时加载这个二进制 engine 跑;(b) 直接用 PyTorch eager/graph 模式跑 HF 权重,不产出任何编译产物。**当前快照里只剩 (b)。** (a) 整条链路——`trtllm-build`/`trtllm-refit`/`trtllm-prune` 命令、`TrtLlmArgs` 类、`tensorrt_llm._tensorrt_engine.LLM`、per-model 的 `convert_checkpoint.py`——已经被整体删除,只在 `docs/source/legacy/` 下留了交叉引用文档,首行就是警告 banner(`docs/source/legacy/architecture/workflow.md:3`-`4`:"The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.",文档所述)。这条演进本身也留了痕迹:`docs/source/legacy/torch.md:5`-`9` 是 PyTorch 后端还处于"currently in beta"阶段时写的旧文档("The PyTorch backend of TensorRT LLM is available in version 0.17 and later"),对照 `AGENTS.md` 里当前的架构表把 PyTorch 标成"Default",能看出这是一条从 beta 走到唯一主线的完整轨迹。

**编译产物的历史生命周期(文档所述,`docs/source/legacy/architecture/checkpoint.md`)**:虽然这条路线已经从当前可达代码里删除,但理解它曾经"重"在哪里,才能理解"移除它"这个决策换来了什么。旧流程分两步——① `convert_checkpoint.py` 把 HF checkpoint 转换成 TRT-LLM 自己的 checkpoint 格式,并行度在这一步就写死:文档示例(`docs/source/legacy/architecture/checkpoint.md:187`-`191`)`python3 convert_checkpoint.py --model_dir ./opt-125m --dtype float16 --tp_size 2 --output_dir ./opt/125M/trt_ckpt/fp16/2-gpu/` 产出的 `config.json` 里直接带 `"mapping": {"world_size": 2, "tp_size": 2}`(`docs/source/legacy/architecture/checkpoint.md:204`-`218`,文档所述)——**这个 checkpoint 只能用于 TP=2 的部署,换并行度必须回到这一步重新转换**。② `trtllm-build` 把这个 checkpoint 编译成 engine,文档示例(`docs/source/legacy/architecture/checkpoint.md:230`-`236`)`trtllm-build --checkpoint_dir ./opt/125M/trt_ckpt/fp16/2-gpu/ --gemm_plugin float16 --max_batch_size 8 --max_input_len 924 --max_seq_len 1024 --output_dir ./opt/125M/trt_engines/fp16/2-gpu/`——`max_batch_size`/`max_seq_len` 在这一步同样被编译进 engine,不是运行时可调参数。这条证据链解释了 `## 5` 决策 2 提到的死代码为什么长那样:`tensorrt_llm/llmapi/llm.py:1586`-`1630` 里对 `build_config.max_batch_size`/`max_beam_width`/`max_seq_len` 的校验,校验的正是"用户这次请求有没有超出编译时就已经固定死的上限"——引擎产物存放路径按惯例是"精度/并行度"分层的目录(如上例 `trt_engines/fp16/2-gpu/`),一个模型想同时支持 fp16/int8、TP=1/TP=2/TP=4 三种精度两种并行度组合,就要维护 6 份独立的 engine 目录。**具体编译耗时的量级:未查证**——本库翻遍能找到的 legacy 文档都没有给出量化的构建时长数字,按 `_PLAN.md` §1 第 4 条的口径要求不编造这个数字;能确定的只是"这是一次需要跑完整 TensorRT 图优化流程的操作,和加载一个已有 checkpoint 不是同一数量级的动作",但具体是几分钟还是几十分钟,源码与文档都没有给出,不把印象当结论写。

**AutoDeploy 不是第三条独立路线,是 PyTorch 路线内部的一个变体**——`AGENTS.md` 的架构表把它列成单独一行(Entry Point `_torch/auto_deploy/` shim),但它的 Key Path 写的是"adapts `PyExecutor`"(文档所述),也就是说它复用的还是同一个 `PyExecutor` 事件循环,只是在模型构造阶段多了一层 `torch.export` + 图变换。而且它自己正在被官方劝退:`tensorrt_llm/commands/serve.py:950`-`952` 的 CLI 帮助文本直接写"the '_autodeploy' backend is deprecated and will be discontinued in a future release; please use the 'pytorch' backend instead"。把这条线索和 `## 0` 的 backend 校验逻辑放在一起看,当前快照实际上只有**一条**长期路线——`"pytorch"`——`"_autodeploy"` 是一个正在退场的实验分支,不是"第二个平起平坐的选项"。

**规模画像(源码为证,`_lab/out/struct_map.json`,粗口径需回源码核)**:按目录名/文件名关键词匹配出四个子系统——

| 子系统 | 粗口径行数 | 文件数 | 最大文件(粗口径,可能混入无关代码) |
|---|---:|---:|---|
| `model_exec` | 76,346 | 89 | `tensorrt_llm/_torch/pyexecutor/py_executor.py` 8,827 行、`tensorrt_llm/_torch/pyexecutor/model_engine.py` 8,404 行 |
| `attention` | 74,697 | 127 | `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/attention/mla/mla_decode_fp16.py` 4,504 行 |
| `distributed` | 25,966 | 48 | `tensorrt_llm/_torch/visual_gen/models/cosmos3/pipeline_cosmos3.py` 2,339 行 |
| `kv_cache` | 25,021 | 44 | `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` 3,929 行 |

这个统计口径和 [[03-vLLM-调度器解剖]] 里"scheduler 子系统统计混进了 KV 传输调度"遇到的问题是同一类——**按关键词匹配目录名会把不相关的东西混进来**:`attention` 子系统里排前几的文件里不少其实是 `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/attention/fmha_blockscaled.py`(3,768 行)这类 VisualGen(图像/视频生成)专用的融合多头注意力 kernel,不是 LLM 文本推理路径会用到的代码;`distributed` 子系统同理混进了上表里那个 Cosmos3 视频生成 pipeline 文件。

**本篇讲的"attention/distributed/model_exec"专指 `_torch/pyexecutor/`、`_torch/attention_backend/`、`_torch/distributed/` 这几个目录,不是上表粗口径统计的全部**——粗口径数字适合做体量感的第一印象,精确论证要落到具体文件路径,这条经验在跨引擎对比里反复出现,值得每次用统计数字前都自问一遍"这个数字有没有被无关文件污染"。

**部署形态也不止"单进程 `LLM()`"一种。** `GenerationExecutor.create()`(`## 4` ⑤ 展开)会按 `orchestrator_type` 在 MPI、Ray、RPC 三种进程编排方式之间选择,`tensorrt_llm/executor/executor.py:614`-`625` 是 Ray 分支的入口,`:640`-`646` 是 RPC 分支。HTTP 服务层还叠加了一层独立的形态维度——`OpenAIServer` 按 `ServerRole` 枚举在同一个类里切换成聚合(LLM)、多模态编码器、embedding、视觉生成四种角色之一(`## 2.1` 展开),`OpenAIDisaggServer`/`CoordinatorServer` 则是 PD 分离与多副本路由这两种更粗粒度形态各自独立的进程。理解 TRT-LLM 的"服务形态"要同时分清两条正交的轴:**进程编排**(MPI/Ray/RPC,决定 GPU worker 怎么被拉起)和**服务角色**(聚合/PD分离/协调器/多模态编码/embedding/视觉生成,决定 HTTP 层暴露什么)。

## 2. 代码地图(文件 → 职责,带行号)

按"公开 API 层 → 编排层 → 执行层 → C++ 运行时"四层组织,共 25 条引用:

| 文件 | 职责 | 关键行 |
|---|---|---|
| `tensorrt_llm/llmapi/llm.py:350` | `BaseLLM`——所有 LLM 类的公共基类,决定 backend 校验、参数收敛 | — |
| `tensorrt_llm/llmapi/llm.py:391`-`399` | backend 只认 `"pytorch"`/`"_autodeploy"`,其它一律 `ValueError` | — |
| `tensorrt_llm/llmapi/llm.py:1782` | `_TorchLLM.__init__`:`backend = kwargs.pop("backend", "pytorch")`——默认值就是 pytorch | — |
| `tensorrt_llm/llmapi/llm.py:1933` | `class LLM(_TorchLLM)`——公开的 `LLM` 类本身就是 `_TorchLLM` 的子类,没有独立的"TRT engine LLM"分支 | — |
| `tensorrt_llm/llmapi/llm_args.py:5160` | `class TorchLlmArgs(BaseLlmArgs)`——当前唯一的主力配置类 | — |
| `tensorrt_llm/llmapi/llm_args.py:5407`-`5408` | `torch_compile_config` 默认 `None`,status=prototype | — |
| `tensorrt_llm/llmapi/llm_args.py:5180`-`5188` | `cuda_graph_config` 默认 `CudaGraphConfig()`(非 None,即默认开),status=beta | — |
| `tensorrt_llm/llmapi/llm_utils.py:362` | `CachedModelLoader`——加载 HF checkpoint 的唯一入口 | — |
| `tensorrt_llm/llmapi/llm_utils.py:448` | `return None, self._hf_model_dir`——`_engine_dir` 恒为 `None` | — |
| `tensorrt_llm/executor/executor.py:543` | `GenerationExecutor.create()`——决定用 MPI/Ray/RPC/IPC 哪种编排方式起 worker 进程 | — |
| `tensorrt_llm/executor/base_worker.py:165`-`216` | `_create_py_executor()`——worker 进程内真正调用 `create_py_executor` | — |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py:337` | `create_py_executor()`——组装 ModelEngine/Scheduler/KVCacheManager,产出 `PyExecutor` | — |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py:549` | `class PyExecutor`——事件循环本体,全库最大类之一(8,199 行) | — |
| `tensorrt_llm/_torch/pyexecutor/_util.py:3082`-`3094` | 默认调度器构造:`BindCapacityScheduler` + `BindMicroBatchScheduler` | — |
| `tensorrt_llm/_torch/pyexecutor/llm_request.py:883` | `class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest)`——请求对象本身继承自 C++ 绑定类 | — |
| `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:668` | `CapacityScheduler::CapacityScheduler` 构造函数(C++ 实现本体) | — |
| `cpp/tensorrt_llm/nanobind/batch_manager/algorithms.cpp:87` | nanobind 绑定点:`nb::class_<CapacityScheduler>(m, ...)` | — |
| `tensorrt_llm/serve/openai_server.py:962`-`963` | `OpenAIServer.register_routes()`——60 条路由里的 47 条在这个方法里注册 | — |
| `tensorrt_llm/commands/serve.py:29` | `from tensorrt_llm import LLM as PyTorchLLM`——CLI 层给公开 `LLM` 类起的别名,暴露了它就是 pytorch backend 这件事 | — |
| `tensorrt_llm/commands/serve.py:663` | `llm_args.pop("build_config", None)`——CLI 构造 pytorch LLM 前先丢掉遗留字段 | — |
| `tensorrt_llm/serve/openai_server.py:578`-`594` | `ServerRole` 四分支,决定这个进程注册哪一组路由(`## 2.1` 展开) | — |
| `tensorrt_llm/serve/openai_disagg_server.py:154`-`156` | PD 网关自身"自己管协调"还是"委托给外部 CoordinatorServer"的分支注释 | — |
| `tensorrt_llm/llmapi/llm_args.py:4770`-`4774` | `orchestrator_type` 字段:默认 `None` 即 MPI,`"rpc"`/`"ray"` 是 status=prototype 的备选 | — |
| `tensorrt_llm/llmapi/llm_args.py:4299`-`4306` | `CacheTransceiverConfig.backend`:`DEFAULT`/`UCX`/`NIXL`/`MOONCAKE`/`MPI` 五选一 | — |
| `tensorrt_llm/llmapi/llm_args.py:3860`-`3865` | `KvCacheConfig.enable_block_reuse` 默认 `True`——前缀缓存默认打开 | — |

代表性的调度器默认构造代码(`tensorrt_llm/_torch/pyexecutor/_util.py:3082`-`3096`,源码为证,和 `## 0`/`## 5` 决策 3 的结论对应):

```python
capacity_scheduler = BindCapacityScheduler(
    scheduler_capacity,
    kv_cache_manager.impl if kv_cache_manager is not None else None,
    peft_cache_manager.impl if peft_cache_manager is not None else None,
    scheduler_config.capacity_scheduler_policy,
    cross_kv_cache_manager=cross_kv_cache_manager.impl
    if cross_kv_cache_manager is not None else None,
    two_step_lookahead=mapping.has_pp(),
    no_schedule_until_state=no_schedule_until_state,
    enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
)

mb_scheduler = BindMicroBatchScheduler(
    max_batch_size,
    max_num_tokens,
    ctx_chunk_config,
    no_schedule_until_state=no_schedule_until_state,
)
```

这段代码是`else`分支(既不是 `KVCacheV2Scheduler` 路径也不是 `scheduler_config.use_python_scheduler` 路径时的兜底),`BindCapacityScheduler`/`BindMicroBatchScheduler` 两个类名里的"Bind"前缀就是"这是一层 C++ 绑定的薄包装"最直白的自我说明。

**本篇范围边界**:PyExecutor 内部的 KV 缓存分配细节、注意力后端矩阵（TRTLLM/FlashInfer/VANILLA 等具体 kernel 选择)、投机解码实现,不在本篇展开——本篇只回答"这个引擎的整体分层与编译式/解释式这条根本设计选择",子系统细节留给本库后续篇目补齐(参考 `_PLAN.md` §4 的篇目占位,当前均为规划中)。

### 2.1 API 表面:60 条路由分族

`_lab/out/api_surface.json` 用 AST 抽取到 60 条 `add_api_route` 调用,`openai_compat_paths`(即 `/v1/*`)有 14 条。逐条核对源码后,这 60 条分属 **4 个 Python 文件、7 个互斥的注册方法**——理解这一点很关键:**没有任何一个运行中的 `trtllm-serve` 进程会同时暴露 60 条路由**,每个进程只会走到下面 7 组里的一组。

**组 1~4 都在 `tensorrt_llm/serve/openai_server.py` 的 `OpenAIServer` 类里,由 `ServerRole` 枚举二选一(`tensorrt_llm/serve/openai_server.py:578`-`594`):**

| 组 | 注册方法 | 触发条件 | 路由数 | 代表路由(路径 \| 方法 \| handler \| 文件:行) |
|---|---|---|---:|---|
| 1 聚合/默认(最常见) | `register_routes()`,入口 `tensorrt_llm/serve/openai_server.py:963` | `server_role` 非以下三者(默认分支) | 20 | `/v1/chat/completions` \| POST \| `self.openai_chat` \| `tensorrt_llm/serve/openai_server.py:1021`;`/v1/responses` \| POST \| `self.openai_responses` \| `tensorrt_llm/serve/openai_server.py:1025`;`/server_info` \| GET \| `self.get_server_info` \| `tensorrt_llm/serve/openai_server.py:1048` |
| 2 多模态编码器 | `register_mm_encoder_routes()`,入口 `tensorrt_llm/serve/openai_server.py:1077` | `ServerRole.MM_ENCODER` | 8 | `/v1/chat/completions` \| POST \| `self.openai_mm_encoder` \| `tensorrt_llm/serve/openai_server.py:1085` |
| 3 Embedding | `register_embedding_routes()`,入口 `tensorrt_llm/serve/openai_server.py:1128` | `ServerRole.EMBEDDING` | 4 | `/v1/embeddings` \| POST \| `self.openai_embedding` \| `tensorrt_llm/serve/openai_server.py:1132` |
| 4 视觉生成(VisualGen) | `register_visual_gen_routes()`,入口 `tensorrt_llm/serve/openai_server.py:1234` | `ServerRole.VISUAL_GEN` | 13 | `/v1/images/generations` \| POST \| `self.openai_image_generation` \| `tensorrt_llm/serve/openai_server.py:1245`;`/v1/videos/{video_id}` \| DELETE \| `self.delete_video` \| `tensorrt_llm/serve/openai_server.py:1272` |

**组 5~7 是三个完全独立的服务进程,各自的 `app` 对象互不相干:**

| 组 | 文件:类 | 角色 | 路由数 | 代表路由 |
|---|---|---|---:|---|
| 5 PD 分离网关 | `tensorrt_llm/serve/openai_disagg_server.py:139` `OpenAIDisaggServer` | 转发请求到 context/generation 两组后端 | 5 | `/v1/completions` \| POST \| `self._wrap_entry_point(...)` \| `tensorrt_llm/serve/openai_disagg_server.py:264` |
| 6 多副本协调器 | `tensorrt_llm/serve/coordinator_server.py:52` `CoordinatorServer` | 多副本间 `/select`(选副本)`/finish`(释放) | 5 | `/select` \| POST \| `self.select` \| `tensorrt_llm/serve/coordinator_server.py:65` |
| 7 集群键值存储 | `tensorrt_llm/serve/cluster_storage.py` | 多节点间共享少量元数据(non-FastAPI,走另一个 `server` 对象) | 5 | `/set` \| POST \| `jsonify(self._set)` \| `tensorrt_llm/serve/cluster_storage.py:226` |

决定走哪一组的分支就是`## 0`已提到的 `ServerRole` 判断(`tensorrt_llm/serve/openai_server.py:578`-`594`,源码为证):

```python
if self.server_role is ServerRole.VISUAL_GEN:
    assert self._is_visual_gen, \
        "generator must be a VisualGen for VISUAL_GEN server"
    self.register_visual_gen_routes()
elif self.server_role is ServerRole.MM_ENCODER:
    assert isinstance(
        self.generator, MultimodalEncoder
    ), "generator must be a MultimodalEncoder for multimodal encoder"
    self.register_mm_encoder_routes()
elif self.server_role is ServerRole.EMBEDDING:
    assert getattr(self.generator.args, "encode_only", False), (
        "generator must be an encode_only=True LLM for the embedding "
        "server")
    self._init_embedding_batcher()
    self.register_embedding_routes()
else:
    self.register_routes()
```

四个分支互斥,而且每个分支进去之前还带了一个 `assert`,校验 `self.generator`(背后是`## 4`⑤~⑥组装出来的 `GenerationExecutor`/`MultimodalEncoder`)确实是这个角色期望的类型——路由注册和运行时类型校验被绑在了一起,选错角色不是"路由 404",而是构造阶段直接 `AssertionError`。

20+8+4+13+5+5+5 = 60,和 `api_surface.json` 的 `n_routes` 对上。**这个"60"本身是一个需要小心的数字**——它是静态扫描"这个仓库里一共写了多少条 `add_api_route`",不是"一个进程运行时暴露多少接口"。单机标准服务(组 1)实际只暴露 20 条,`/v1/*` 里也只有 5 条落在这一组(`/v1/models`、`/v1/data_transceiver_state`、`/v1/completions`、`/v1/chat/completions`、`/v1/responses` 及其子路径)。

## 3. 核心数据结构

### 3.1 `TorchLlmArgs`(配置的单一事实来源)

`tensorrt_llm/llmapi/llm_args.py:4448` 定义 `BaseLlmArgs(StrictBaseModel)`,`tensorrt_llm/llmapi/llm_args.py:5160` 派生出 `TorchLlmArgs(BaseLlmArgs)`——这是一个 Pydantic 模型,字段上普遍标注 `status="beta"`/`status="prototype"`/`status="deprecated"`(比如上面提到的 `torch_compile_config`、`cuda_graph_config`)。这个 status 标注本身就是一种诚实机制:字段没写 status 的隐含"stable",标了的等于官方在告诉你"这块随时可能变"。构造 `LLM` 对象时,`tensorrt_llm/llmapi/llm.py:411`-`420` 把用户传入的所有 kwargs 塞进 `TorchLlmArgs(...)`,Pydantic 校验失败会在构造期直接报错,而不是留到跑到某个分支才发现参数不对——这比很多引擎"运行时才发现某个 flag 组合不支持"的体验更早暴露问题。

### 3.2 `LlmRequest`——继承自 C++ 绑定类

`tensorrt_llm/_torch/pyexecutor/llm_request.py:883`:

```python
class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):
```

这行代码是理解"in-flight batching 到底在哪实现"的关键证据:Python 侧的请求对象**不是**独立定义的 dataclass,而是直接继承自 nanobind 暴露出来的 C++ `batch_manager::LlmRequest`。也就是说请求的核心状态(token 序列、KV block 归属、beam 状态机等)本来就活在 C++ 对象里,Python 子类只是加了一些 Python 侧才需要的便利方法。对照 vLLM 的 `Request`(纯 Python dataclass,`vllm:vllm/v1/request.py`)或 SGLang 的 `Req`(同样纯 Python),这是三个引擎里唯一一个"请求对象骨架在 C++"的设计。

### 3.3 调度产出:`(scheduled, paused, fitting_disagg_gen_init)` 三元组

`BindCapacityScheduler.schedule_request()`(`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py:421`-`463`)返回一个三元组——本地能上场的请求、被暂停(抢占)的请求、以及等待跨机 KV 传输落地的 PD 分离生成请求。这不是一个像 vLLM `SchedulerOutput` 那样字段丰富的 dataclass 契约,而是更接近 C++ 那边 `CapacityScheduler::operator()` 的原始返回签名——Python 只是原样转发。

### 3.4 `SamplingParams` 与 `Mapping`

`tensorrt_llm/sampling_params.py:186` 的 `SamplingParams` 承担采样超参(温度、top-p、beam width 等);`tensorrt_llm/mapping.py:464` 的 `class Mapping(MappingBase)` 承担并行拓扑——TP/PP/CP(context parallel)/EP(expert parallel)各维度的 rank 映射,是构造 `LLM(tensor_parallel_size=..., pipeline_parallel_size=...)` 时真正落地成"谁跟谁通信"的地方。

### 3.5 `CapacitySchedulerPolicy`——Python 枚举镜像 C++ 枚举

`tensorrt_llm/llmapi/llm_args.py:3432`-`3438`:

```python
class CapacitySchedulerPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)
```

这个 `_to_pybind()` 方法(命名沿用了从 pybind11 迁移到 nanobind 之前的旧叫法,`## 7` 会展开这个命名遗留)是"配置在 C++、决策也在 C++"这条主线的又一处证据:用户在 Python 侧设置的 `CapacitySchedulerPolicy.MAX_UTILIZATION`,构造 `BindCapacityScheduler` 时(`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py:451` 的 `scheduler_policy._to_pybind()`)立刻被转换成 C++ 侧同名枚举 `_CapacitySchedulerPolicy` 再传进 nanobind 绑定的构造函数——Python 枚举本身只是一层贴着 C++ 枚举值的包装,和 `## 3.2` 的 `LlmRequest` 是同一种设计哲学:配置对象与决策算法都尽量贴近 C++ 实现,Python 侧提供的是类型安全的镜像而不是独立实现。`SchedulerConfig` 本身(`tensorrt_llm/llmapi/llm_args.py:3489` 的 `class SchedulerConfig(StrictBaseModel, PybindMirror)`)整个类都继承自 `PybindMirror`,这不是孤例,是 TRT-LLM 配置层的通用模式。

### 3.6 `ResourceManager`——资源管理器的注册表模式

`tensorrt_llm/_torch/pyexecutor/resource_manager.py:2861` 的 `class ResourceManager` 本身很薄——一个按 `ResourceManagerType` 枚举索引的 `OrderedDict`(`tensorrt_llm/_torch/pyexecutor/resource_manager.py:2863`-`2867`),真正的资源分配逻辑分散在各个 `BaseResourceManager` 子类里(KV 缓存、PEFT/LoRA 缓存、Mamba 状态缓存等各自一个)。文档所述(`docs/source/torch/arch_overview.md`)把它总结成三个接口:`prepare_resources`(每步 model forward 之前调)、`update_resources`(每步结束后调)、`free_resources`(请求结束时调)。这种"薄容器 + 多个独立管理器"的模式和 `## 3.1` 的 `TorchLlmArgs`(单一大配置对象)形成对照——配置层选择了"一个大类装所有字段",资源管理层选择了"一个注册表 + 多个小类",同一个代码库里两种组织哲学并存,分别服务于"用户要方便地设置几百个旋钮"和"内部要能按类型独立扩展资源种类"这两个不同的目标。

### 3.7 `ModelEngine` / `PyTorchModelEngine`

`tensorrt_llm/_torch/pyexecutor/model_engine.py:190` 的 `class ModelEngine(ABC)` 是模型前向的抽象接口(`forward()` 方法),`tensorrt_llm/_torch/pyexecutor/model_engine.py:381` 的 `class PyTorchModelEngine(ModelEngine)` 是当前唯一的具体实现——持有真正的 `nn.Module`(HF 架构直接构造出来的 PyTorch 模型)、CUDA Graph 捕获/重放逻辑(`## 5` 决策 1 展开)、以及 `torch_compile_config` 非 `None` 时对 `self.model` 做 `torch.compile()` 包裹的分支(`tensorrt_llm/_torch/pyexecutor/model_engine.py:822`-`833`)。文档所述(`docs/source/torch/arch_overview.md`):"The core component of `PyExecutor` is the `ModelEngine`... The key method of `ModelEngine` is `forward`"。

### 3.8 `KvCacheConfig`——`enable_block_reuse` 默认打开,`sink_token_length` 已废弃

`tensorrt_llm/llmapi/llm_args.py:3860`-`3865` 的 `KvCacheConfig(StrictBaseModel, PybindMirror)` 同样继承 `PybindMirror`(和`## 3.5`的 `SchedulerConfig` 同一个模式)。`enable_block_reuse: bool = Field(default=True, ...)` 是 TRT-LLM 版本的前缀缓存开关,默认打开,对应`00-总览与阅读地图`里 vLLM 前缀缓存驱逐策略那条硬结论的同位概念——具体驱逐策略、块级复用的实现细节不在本篇范围(见`## 2`范围边界)。同一个类里还能看到`## 5`决策 2、`## 7` 踩坑 6 那种"字段留着但已经不代表原意"的例子:`sink_token_length` 字段的 description 原文是"Deprecated and ignored on the PyTorch backend. StreamingLLM is not supported..."(`tensorrt_llm/llmapi/llm_args.py:3877`-`3880`,源码为证)——StreamingLLM(注意力汇聚 token 机制)这个特性本身在 PyTorch 后端已经不支持,但字段没有被删除,只是被标注成"deprecated and ignored"。这类"字段还在、行为已经不在"的模式在 TRT-LLM 配置层反复出现,读者遇到任何一个字段之前最好先看它的 `status`/description,而不是假设字段存在就等于功能存在。

## 4. 主流程走读

一次 `trtllm-serve <hf_model>` 到能接受 HTTP 请求,要穿过下面这条链路:

① **CLI 解析**:`tensorrt_llm/commands/serve.py` 的 `serve` 命令(`click.command`)解析 `--backend`(默认 `pytorch`,`tensorrt_llm/commands/serve.py:948`)等参数,调用 `get_llm_args()`(`tensorrt_llm/commands/serve.py:197`)把 CLI flag 收敛成一个 dict。

② **构造 `LLM` 对象**:`tensorrt_llm/commands/serve.py:662`-`664`——`if backend == 'pytorch': llm_args.pop("build_config", None); llm = PyTorchLLM(**llm_args)`。注意这里显式 `pop` 掉 `build_config`——这是给旧配置文件/脚本兼容用的清理动作,再次印证 `build_config` 是遗留字段(`## 5` 决策 2 会展开)。`PyTorchLLM` 就是 `tensorrt_llm.LLM`(`tensorrt_llm/commands/serve.py:29` 的别名导入)。

③ **`LLM.__init__` → `BaseLLM.__init__`**(`tensorrt_llm/llmapi/llm.py:1933` → `tensorrt_llm/llmapi/llm.py:1769` → `tensorrt_llm/llmapi/llm.py:354`):校验 backend、把 kwargs 收敛成 `TorchLlmArgs` 实例、设置 MPI session。

④ **`_build_model()`**(`tensorrt_llm/llmapi/llm.py:1835`,`_TorchLLM` 覆写):调 `CachedModelLoader`(`tensorrt_llm/llmapi/llm_utils.py:362`)下载/定位 HF checkpoint,返回 `(None, hf_model_dir)`(`tensorrt_llm/llmapi/llm_utils.py:448`);随后创建 tokenizer、`input_processor`;最后调 `self._executor_cls.create(...)`(`tensorrt_llm/llmapi/llm.py:1896`,即 `GenerationExecutor.create`)。

⑤ **`GenerationExecutor.create()`**(`tensorrt_llm/executor/executor.py:543`):这里是编排层的分岔口——按 `orchestrator_type` 决定用 Ray(`_create_ray_executor`)、RPC(`_create_rpc_executor`)、还是 MPI/IPC(`_create_ipc_executor`,`tensorrt_llm/executor/executor.py:648`)拉起 worker 进程/线程。多机部署时这一层还要处理 `TLLM_EXECUTOR_ATTACH_INFO` 这种"附着到已运行 worker"的场景(`tensorrt_llm/executor/executor.py:579`-`603`),用于多前端(multi-frontend)共享一个后端 executor。

⑥ **worker 进程内构造 `PyExecutor`**:`tensorrt_llm/executor/base_worker.py:165`-`216` 的 `_create_py_executor()` 调 `create_py_executor()`(`tensorrt_llm/_torch/pyexecutor/py_executor_creator.py:337`),后者依次:加载模型配置、构造 `ModelEngine`(持有实际的 `nn.Module`)、构造调度器(默认 `BindCapacityScheduler`+`BindMicroBatchScheduler`,`tensorrt_llm/_torch/pyexecutor/_util.py:3082`-`3094`)、构造 `KVCacheManager`、构造 `Sampler`,最后 `return PyExecutor(...)`(`tensorrt_llm/_torch/pyexecutor/py_executor.py:549`)。

⑦ **`PyExecutor` 事件循环**:文档所述(`docs/source/torch/arch_overview.md` "The single-step flow of PyExecutor")——取新请求→调度→模型前向→解码→处理完成请求,循环往复。调度这一步就是⑤⑥里组装好的 `BindCapacityScheduler`/`BindMicroBatchScheduler`,底层调用的是 C++ `CapacityScheduler::operator()`(`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:708`)。这就是 in-flight batching(TRT-LLM 对连续批处理的官方叫法,`InflightBatchingStats` 类名与统计字段贯穿全代码库,例如 `tensorrt_llm/_torch/pyexecutor/py_executor.py:2012`)真正决策发生的地方——**默认路径是 C++,不是 Python**。

⑧ **HTTP 服务层**:`OpenAIServer.__init__` 持有一个 `GenerationExecutor`(通过 `self.generator`),按 `## 2.1` 的 `ServerRole` 分支选择一组 `register_*_routes()` 方法把对应路由挂到 FastAPI `app` 上;请求进来后 `openai_chat`/`openai_completion` 等 handler 把 OpenAI 协议对象转换成 TRT-LLM 内部的 `SamplingParams` + prompt token id,喂给 `self.generator.generate_async(...)`,本质上是往 `PyExecutor` 的请求队列里塞了一条 `LlmRequest`。

### 4.1 一次 decode 步:CUDA Graph 命中与不命中的两条路

`PyExecutor` 的事件循环每一步都要决定"这一批请求要不要走 CUDA Graph 重放"。这条判断逻辑值得单独走一遍,因为它是"编译式今天变成了什么"这个问题最具体的落地:

1. **判断能不能重放**:只有当这一步调度出来的批次**全部是 decode 阶段**(没有 prefill/chunked-prefill 请求混在里面)、且这个批次的 batch size 落在 `cuda_graph_config.batch_sizes` 预先捕获过的桶里,才会走重放路径。`AGENTS.md` 里对应到 `docs/source/torch/scheduler.md` 描述的 `CapacityScheduler`/`MicroBatchScheduler` 两步调度产出的批次信息,决定了这一步是否满足条件。
2. **命中**:直接 replay 之前录制好的 CUDA Graph——一次 kernel 启动序列的重放,跳过 Python 侧逐算子调度的开销,这是 decode 阶段(每步只算一个新 token,矩阵很小,launch overhead 相对计算量占比高)收益最大的地方。
3. **不命中**(prefill、chunked-prefill、batch size 没预先捕获、或者开了 `enable_piecewise_cuda_graph`/`torch_compile_config` 走 piecewise 编译路径):退回普通 PyTorch eager 前向,一个算子一个算子地正常调度、正常启动 kernel。**不命中不是错误,是设计里就有的常态路径**——第 ①、④ 步的 prefill 恒定走这条路(除非开了 `prefill_cuda_graph_backend`,`tensorrt_llm/llmapi/llm_args.py:5410`-`5416`,默认 `DISABLED`)。
4. **warmup 阶段**:CUDA Graph 的捕获动作不是发生在服务启动瞬间就对所有配置的 batch size 桶捕获完——`model_engine.py` 里能看到 `warmup_with_kv_cache_cleanup`、`_get_max_shape_warmup_requests` 这类方法名(`_lab/out/struct_map.json` 抽取到的 `PyTorchModelEngine` 方法列表),说明捕获动作发生在服务真正接受流量之前的一段专门 warmup 期——这也是为什么"改一次 `cuda_graph_config.batch_sizes` 要重启服务"这件事的开销不是"编译耗时"而是"warmup 耗时",两者本质是同一类代价(重新录制),只是没有旧 AOT 引擎构建那么久。

这条路径选择逻辑本身完全在 Python 里做(`PyTorchModelEngine.forward()`),和 `## 5` 决策 3 里"调度决策在 C++"是两回事——**决定"喂哪些请求"的调度算法在 C++,决定"这批请求怎么跑模型前向"(重放 or eager)的逻辑在 Python**。这是理解 TRT-LLM 分层时容易混淆的一点:C++/Python 的分界线不是沿着"调度 vs 执行"画的,而是更细粒度地按具体职责逐个划定的。

### 4.2 PD 分离场景下多一层网关

如果部署形态是 PD 分离(`## 1` 提到的"服务角色"轴),请求的路径要多绕一层:客户端先打到 `OpenAIDisaggServer`(`## 2.1` 组 5),它把 `/v1/completions`/`/v1/chat/completions` 通过 `self._wrap_entry_point(...)` 转发给内部的 `context`/`generation` 两组独立 worker 集群——每组 worker 本质上还是`## 4`①~⑧那一整条链路各跑一份,只是一组只做 prefill(context)、一组只做 decode(generation),中间通过 `## 6`"KV 传输运行时"一节说的 NIXL/UCX/MOONCAKE/MPI 后端搬运 KV cache。`OpenAIDisaggServer.__init__` 里有一段注释(`tensorrt_llm/serve/openai_disagg_server.py:154`-`156`,源码为证)把这层网关自身的两种角色说清楚了:"When set, this is a forked worker: routing/readiness are delegated to the coordinator at coordinator_url... Otherwise this process owns the routers + cluster state"——也就是说 `OpenAIDisaggServer` 既可以自己承担路由/集群状态管理,也可以把这部分委托给`## 2.1` 组 6 的独立 `CoordinatorServer`,取决于是否传了 `coordinator_url`。三种服务角色(聚合、PD 网关、协调器)不是三选一的静态部署形态,而是可以按"网关是否需要外部协调器"这个开关再组合出一种变体。

## 5. 设计决策与代价

### 决策 1:PyTorch 是唯一后端,不再提供"engine 编译"作为可选项

`docs/source/legacy/tensorrt-backend-removal.md:11`-`19`(文档所述)用一张表列完了整条链路被拿掉了什么、换成了什么,照抄如下,是理解这个决策范围最直接的证据:

| Removed | Replacement / new behavior |
| --- | --- |
| `LLM(backend="tensorrt")` | Raises `ValueError` — PyTorch is the only backend; omit `backend` |
| `TrtLlmArgs` | Use `TorchLlmArgs`(the default) |
| `tensorrt_llm._tensorrt_engine.LLM` | Use `tensorrt_llm.LLM` |
| `trtllm-build` / `trtllm-refit` / `trtllm-prune` | No engine-build step — HuggingFace checkpoints load directly |
| Per-model `convert_checkpoint.py` | Not needed — no checkpoint conversion |
| `--backend tensorrt`(CLI) | Omit, or pass `--backend pytorch` |
| `tensorrt` pip dependency | Dropped(no longer installed) |

最后一行值得单独拎出来:连 `tensorrt` 这个 pip 依赖本身都被去掉了,不是"默认不用但还装着",是构建产物、构建工具、构建依赖三层一起清空。

**为什么这么设计**——文档所述(`docs/source/legacy/tensorrt-backend-removal.md`)给出了动机链:HF checkpoint 可以直接加载,不需要 per-model 的 `convert_checkpoint.py`;新模型架构接入不再需要重新实现 TRT-LLM 自己的模型定义语言,直接复用 HF `transformers` 的模型代码路径更快。`docs/source/legacy/architecture/workflow.md:24`(文档所述)也提前写过旧架构的痛点:"TensorRT-LLM evolves so quickly that the model's definition code might have changed for better performance; which means the `convert_checkpoint.py` is out of date"——维护两条模型定义(HF 原始定义 + TRT-LLM checkpoint 转换脚本)本身就是持续的技术债。

**不这样会怎样**(即维持双后端会怎样,本库推断,依据是 legacy 文档里描述的具体痛点)——两套模型实现要保持同步,新模型/新论文架构要接入两次;引擎构建产物要按每个(模型、精度、量化方案、TP/PP 度、max_batch_size/max_seq_len)组合分别编译和存盘,组合数随部署场景数量组合爆炸;每次 checkpoint 或依赖库升级都可能让已编译的 `.engine` 失效,需要重新构建。

**什么时候可以不这样**(即什么时候维持一个独立编译产物仍然值得)——本库推断:超长期、单模型、单一部署形态、极致延迟敏感的场景(比如车载/边缘设备上一个固定模型跑一辈子),AOT 编译换来的稳定可预测延迟仍然有价值。但 TRT-LLM 项目本身的选择是不再把这个选项内置在主线产品里,把它整体移出到了"legacy,仅供交叉引用"的文档区(`docs/source/legacy/tensorrt-backend-removal.md:47`-`50`)——这是项目方替所有用户做的取舍,不是"你可以选 A 也可以选 B"。

### 决策 2:`build_config`/`LlmBuildStats.engine_dir` 等字段仍留在代码里但已不可达

**为什么这么设计**——`tensorrt_llm/llmapi/llm_utils.py:462`-`477` 的 `LlmBuildStats` dataclass 里还留着 `cache_hitted`(注释"Whether the cache is hit for the engine")、`engine_dir: Optional[Path] = None`(注释"The path to the trt-llm engine")这些字段;`tensorrt_llm/commands/serve.py:663`/`669` 在构造 `PyTorchLLM`/`AutoDeployLLM` 前都要显式 `llm_args.pop("build_config", None)`。这是大型代码库去除一个历史支柱功能时的常见现象——完全物理删除所有引用点风险高(可能有内部工具、序列化格式、遥测统计依赖这些字段名),保留字段名但让主路径不再触达是更安全的渐进式清理。

**不这样会怎样**(即彻底删干净会怎样)——本库推断:如果连 `LlmBuildStats.engine_dir` 这种统计字段也物理删除,任何反序列化旧版本 `LlmBuildStats`(比如历史日志、监控面板)的代码都会直接报错;`tensorrt_llm/llmapi/llm.py:1586`-`1630` 那段读 `build_config` 的死代码如果直接删除,理论上更干净,但如果还有别的分支(比如未来重新加回某种 build 前置校验)复用这段逻辑的一部分,删除会丢失可参考的实现细节。

**什么时候可以不这样**——当一个大版本号明确宣布"删除所有 legacy 兼容层"时(参照 `docs/source/legacy/tensorrt-backend-removal.md` 标题里的"Breaking change"字样,这次 removal 本身就是一次breaking change,但仍选择保留了字段名而非清空整个 legacy 目录)。这提示了一条经验规律:一个功能被"removed and no longer supported",不代表它在代码库里的所有痕迹都会被清除——读源码判断一条路径是否可达,要看**入口处的分支逻辑**(比如 `tensorrt_llm/llmapi/llm.py:391`-`399` 的 `backend` 校验),而不能只看某个字段/类名是否还"存在"。

### 决策 3:in-flight batching 的核心算法用 C++ 实现,通过 nanobind 暴露给 Python

**为什么这么设计**——`AGENTS.md`(文档所述)明确写了架构分层:"Both backends share these C++ components: Scheduling pipeline: Scheduler → BatchManager (in-flight batching) → KV Cache Manager"。这是历史延续:C++ `BatchManager` 是 TRT-LLM 早年为(现已移除的)C++ runtime 写的调度实现,PyTorch 后端接入后选择直接复用这套已经打磨过的调度策略实现,而不是在 Python 里重写一遍——`Scheduler` 操作发生在**每个迭代步**,要扫描全部 active requests 判断谁能获得 KV 资源,是一个天然的性能敏感热路径。

**不这样会怎样**(即改成纯 Python 调度会怎样)——TRT-LLM 自己就提供了这个对照组:`PyCapacityScheduler`(`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py:1909`)和 `SimpleUnifiedScheduler`,通过 `scheduler_config.use_python_scheduler` 开关切入(`tensorrt_llm/_torch/pyexecutor/_util.py:3059`-`3075`)。这条路径的存在本身说明:C++ 实现不是不可替代,而是"默认更快/更省心,但留了逃生舱"。走 Python 路径的代价是:失去了和历史 C++ runtime 共享同一套调度策略实现的红利,策略变更要在两个语言里分别维护(如果哪天两边都要支持)。

**什么时候可以不这样**(即什么时候该切到 Python 调度器)——本库推断:研究/实验自定义调度策略(比如接入新的 batching 启发式、和外部资源管理器联动)时,Python 路径改起来不需要重新编译 C++、不需要 nanobind 重新生成绑定,迭代速度快得多——这正是它被保留而不是被删除的理由。生产环境如果没有自定义调度需求,默认的 C++ 路径是更省心的选择。

### 决策 4:`add_api_route()` 函数调用式注册,而不是 `@app.get(...)` 装饰器式

**为什么这么设计**(源码为证,`tensorrt_llm/serve/openai_server.py:962`-`1048`)——`register_routes()` 是一个独立方法,里面连续调用 `self.app.add_api_route(path, handler, methods=[...])`。这种写法把"路由表"聚合成了一段可以整体阅读的代码块,而不是分散在类的各个方法定义前面(装饰器写法下,一个 handler 的路由信息和函数体绑死在一起,要找"这个 app 一共注册了哪些路由"得翻遍整个文件)。`register_routes()` 里还能看到条件注册的例子——`tensorrt_llm/serve/openai_server.py:988`-`1010` 只有 `resource_governor_queue is not None` 时才会额外注册 `/_resource_governor/*` 路由,这种"路由集合本身依赖运行时状态"的逻辑用装饰器写法几乎没法表达(装饰器在类定义时就固定了路由表,函数调用式可以放在任何条件分支里)。

**不这样会怎样**(即改回装饰器写法会怎样)——本库推断:失去条件注册的能力(除非改用更复杂的路由注册后置技巧);IDE 里"从路径找 handler"会变得更容易(装饰器写法下路径字符串紧贴函数定义,`Ctrl+F` 搜路径就能定位实现),这是函数调用式风格付出的可读性代价——要找 `/v1/completions` 的实现,得先在 `register_routes()` 里找到它绑定的 handler 名字(`self.openai_completion`),再去搜这个方法名。

**什么时候可以不这样**——本库推断:如果一个服务的路由表是静态的、不需要按运行时条件增减,装饰器写法在"定义即注册"这点上更符合直觉,多数 Python Web 框架教程也默认教装饰器写法。TRT-LLM 选择函数调用式,大概率是因为它的路由表**确实**不是静态的(条件注册的 `/_resource_governor/*`,以及不同服务类——`OpenAIServer`/`OpenAIDisaggServer`/`CoordinatorServer`——各自独立的路由集合)。

### 决策 5:一个 `OpenAIServer` 类里塞四种互斥角色,而不是拆成四个类

**为什么这么设计**(源码为证,`tensorrt_llm/serve/openai_server.py:578`-`594`)——`ServerRole` 的四个分支(`VISUAL_GEN`/`MM_ENCODER`/`EMBEDDING`/默认)共享同一个 `OpenAIServer.__init__` 里已经做好的一大堆前置工作:FastAPI `app` 构造、中间件挂载、`GenerationExecutor`/`MultimodalEncoder` 生命周期管理、错误处理、`_collect_perf_metrics` 相关的 `PerfMetricsMiddleware`(`tensorrt_llm/serve/openai_server.py:596`-`598`)。这些是四种角色共用的"服务基础设施",拆成四个类意味着这部分要么重复写四遍,要么再抽一层公共基类——TRT-LLM 选择了更直接的办法:一个类,`register_routes` 换四套。

**不这样会怎样**(即拆成四个独立类会怎样,本库推断)——四个类各自更薄、更容易单独阅读("这个类到底暴露哪些路由"一眼看完,不用先确认 `ServerRole` 分支);但会牺牲共享基础设施的便利性,后续要给"所有角色都加一个中间件"这种需求就要改四处而不是一处。当前 60 条路由里有 20+8+4+13=45 条都挤在同一个文件的同一个类里,`openai_server.py` 因此成为 TRT-LLM 服务层里最大的单文件——这正是"共享基础设施"选择的直接代价:阅读一个角色的路由表,肉眼还是要先跳过另外三组。

**什么时候可以不这样**——本库推断:如果四种角色的共享前置逻辑本来就很薄(比如只是"建一个 `FastAPI()` 对象"这种一行代码级别的共享),拆开成四个独立类反而更清晰,没必要为了省几行初始化代码把互斥的路由表混进同一个文件。TRT-LLM 的 `OpenAIServer.__init__` 显然不是这种"很薄"的情况——它涉及执行器生命周期、中间件、错误处理这些跨角色都要一致的行为,所以合并成一个类是这个具体场景下更划算的选择,不是普适真理。

### 决策 6:进程编排默认 MPI,RPC/Ray 是标了 prototype 的新选项

**为什么这么设计**(源码为证)——`tensorrt_llm/llmapi/llm_args.py:4770`-`4774`:`orchestrator_type: Optional[Literal["rpc", "ray"]] = Field(default=None, description="The orchestrator type to use. Defaults to None, which uses MPI.", status="prototype")`。MPI 是 HPC 领域数十年的成熟技术,TRT-LLM 面向的多 GPU/多节点张量并行场景和 MPI 的设计目标高度重合(固定拓扑、SPMD 执行模型),选它做默认符合"不重新发明轮子"的一般原则。

**不这样会怎样**(即只支持 MPI,不做 RPC/Ray 选项会怎样,本库推断)——MPI 的进程模型是静态的:worker 数量、rank 分配在启动时就固定,不适合和 Ray 这种支持动态扩缩容、和更大的分布式计算生态(比如同一个 Ray 集群里跑数据处理 + 训练 + 推理)集成的场景。只支持 MPI 会让 TRT-LLM 更难嵌入到已经用 Ray 做资源调度的 MLOps 平台里——`orchestrator_type="ray"` 分支(`tensorrt_llm/llmapi/llm.py:385`-`389`)专门处理了这种场景,还顺带把 `TLLM_DISABLE_MPI` 环境变量设成 `1`,说明 Ray 模式下是完全绕开 MPI 通信层的。

**什么时候可以不这样**(即什么时候该用 RPC/Ray 而不是默认 MPI)——本库推断:两个都还标着 `status="prototype"`,官方自己没有把它们当作生产就绪的默认选项。需要和外部 Ray 集群共享资源调度、或者需要比 MPI 更灵活的动态 worker 管理时值得评估,但要接受"prototype"状态意味着接口可能变、覆盖测试可能不如 MPI 路径完整——这是 TRT-LLM 用字段级 `status` 标注做的显式风险提示(`## 3.1` 已经讨论过这个机制),不是本库自己的推测。

## 6. 同位对照

### "编译式 vs 解释式"的标签,在这个快照上是反的

`00-总览与阅读地图.md` 把 vLLM/SGLang 定义为"运行时解释式",TRT-LLM 定义为"编译式"——这是这个仓库还叫得出"TensorRT"时的历史印象。逐行核对默认配置后,结论要反过来说:

| | 默认执行模式 | 依据 |
|---|---|---|
| TensorRT-LLM(本篇) | **eager PyTorch**,`torch_compile_config` 默认 `None` | `tensorrt_llm/llmapi/llm_args.py:5407`-`5408` |
| vLLM | **默认走 `torch.compile`**,`CompilationConfig.mode` 为 `None` 时解析成 `VLLM_COMPILE`(=3) | `vllm:vllm/config/compilation.py:447`-`459` |

vLLM 的 `CompilationConfig.mode` 字段文档字符串原话(`vllm:vllm/config/compilation.py:452`-`453`):"None: If None, we will select the default compilation mode. For V1 engine this is 3"(3 = `VLLM_COMPILE`,"Custom vLLM Inductor-based backend with caching, piecewise compilation, shape specialization, and custom passes")。也就是说 vLLM V1 默认会用 Inductor 把模型图编译一遍并缓存;TRT-LLM 的 PyTorch 后端默认**不**做这件事,只做 CUDA Graph 捕获(录制固定 batch size 下的 kernel 调用序列并重放,不涉及图变换/算子融合意义上的"编译")。

两边唯一还残留"编译式"血统的,是各自的调度决策位置——但这也反过来了:TRT-LLM 的调度决策默认在 **C++**(`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:668`,通过 nanobind 暴露),vLLM 的调度决策在纯 **Python**(`vllm:vllm/v1/core/sched/scheduler.py:484` 的 `schedule()`),SGLang 同样在纯 Python(`sglang:python/sglang/srt/managers/scheduler.py:384` 的 `class Scheduler`)。把"谁把执行热路径下沉到了更底层的语言"作为衡量"编译式思维"的标准,TRT-LLM 其实比 vLLM/SGLang 更彻底——只是它下沉的是**调度算法**,不是**模型计算图**。

### 请求对象的归属:C++ 继承 vs 纯 Python dataclass

TRT-LLM 的 `LlmRequest`(`tensorrt_llm/_torch/pyexecutor/llm_request.py:883`)直接继承 nanobind 绑定类;vLLM 的 `Request`(`vllm:vllm/v1/request.py`)是独立的纯 Python 类。这个差异和上面调度算法归属的差异是同一件事的两面——TRT-LLM 选择让"状态"和"决策"都尽量靠近 C++,Python 层更像一层瘦封装;vLLM/SGLang 选择让状态和决策都留在 Python,C++/CUDA 只负责算子本身。两种架构哲学都能撑起一个生产级引擎,代价分别在`## 5`决策 3 已经展开(改调度策略要不要重新编译 C++)。

### API 表面:三种服务粒度 vs vLLM/SGLang 的单体 server

vLLM 与 SGLang 各自的 HTTP 层基本是单一入口文件族(`vllm:vllm/entrypoints/openai/`、`sglang:python/sglang/srt/entrypoints/http_server.py`)。TRT-LLM 在 `tensorrt_llm/serve/` 下拆成了三个独立 FastAPI 应用——`OpenAIServer`(`tensorrt_llm/serve/openai_server.py:347`,聚合/单机场景,`## 2.1` 展开的四种角色)、`OpenAIDisaggServer`(`tensorrt_llm/serve/openai_disagg_server.py:139`,PD 分离网关,转发 `/v1/completions`/`/v1/chat/completions` 到 context/generation 两组后端)、`CoordinatorServer`(`tensorrt_llm/serve/coordinator_server.py:52`,多副本间的 `/select`/`/finish` 路由协调)——这和 TRT-LLM 把 PD 分离、多前端 attach(`## 4` ⑤ 提到的 `TLLM_EXECUTOR_ATTACH_INFO`)当作一等公民的产品定位是一致的:vLLM/SGLang 的 PD 分离更多是在同一个 server 类里加分支,TRT-LLM 是从进程粒度就切开了。

### KV 传输运行时:又一处"默认 C++,Python 是逃生舱"

PD 分离场景下,KV cache 要从 context 节点搬到 generation 节点。TRT-LLM 这部分的默认实现和`## 5`决策 3 里调度器的模式几乎一模一样:`CacheTransceiverConfig.transceiver_runtime` 字段(`tensorrt_llm/llmapi/llm_args.py:4308`-`4314`)默认是 `"auto"`,字段说明原文——"'auto' (default) adopts the model's preferred runtime when the effective backend supports it, and falls back to the C++ transceiver otherwise"——也就是说除非模型显式声明偏好 Python 传输,否则最终落地的还是 C++ 实现;显式传 `transceiver_runtime="PYTHON"` 才能强制切到 Python 路径(`tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py:53`-`55` 的判断条件里能看到 `!= "PYTHON"` 这个显式排除)。传输后端本身在 `DEFAULT`/`UCX`/`NIXL`/`MOONCAKE`/`MPI` 之间选择(`tensorrt_llm/llmapi/llm_args.py:4302`-`4306`),`AGENTS.md`(文档所述)把 NIXL 称为默认,`MOONCAKE` 这一项还说明 TRT-LLM 的 KV 传输层能直接对接 Mooncake 传输引擎(参见 [[08-Mooncake传输引擎]])。

对照 vLLM 的 PD 分离抽象——`KVConnectorBase_V1`(`vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`)——是一个纯 Python 的 `ABC`,具体后端(NIXL、Mooncake 等)各自是 Python 类实现这个接口。两边都支持 NIXL 作为传输后端,但**默认落地的语言层**又一次呈现出和调度器相同的分野:TRT-LLM 默认 C++、留 Python 逃生舱;vLLM 默认 Python 接口,把 C 层的活儿限制在真正做数据搬运的那一小块。

### 配置稳定性标注:字段级 status vs 没有统一机制

`## 3.1` 提到 TRT-LLM 的每个 Pydantic 字段都可以带 `status="stable"/"beta"/"prototype"/"deprecated"`(默认隐含 stable,本篇引用过的例子包括 `torch_compile_config`、`cuda_graph_config`、`orchestrator_type`、`--backend` CLI 选项本身)。这是一种把"这个功能有多成熟"焊进类型系统的做法,用户 `TorchLlmArgs(...)` 构造期就能感知到自己在用一个 prototype 字段(如果工具链把 status 暴露成警告的话)。对照检查后,vLLM 的 `vllm/config/*.py` 里没有找到同类的字段级 `status=` 元数据(直接 grep 零命中),SGLang 的 `server_args.py` 里对废弃参数是靠散落的注释和运行时 `deprecated alias`/警告文案个案处理(比如 `sglang:python/sglang/srt/server_args.py:358` 的 `# deprecated alias` 行内注释),不是一套统一的、可以被工具链批量扫描的机制。三个引擎里,只有 TRT-LLM 把"成熟度"当成配置 schema 的一等属性——这大概率和 `## 5` 决策 2/决策 6 反复出现的"大量字段处于过渡期"这个现实有关:一个正在把整条后端路线从 TensorRT 迁移到 PyTorch 的项目,比一个架构相对稳定的项目更需要显式标注"这块还在变"。

### 规模对照:体量最大,但"编译式"标签站不住脚

`00-总览与阅读地图.md` §3 的取证基准表(源码为证,`_lab/out/repo_stats.json`)给了三个引擎的完整规模画像:

| 引擎 | 总行数 | Python 产品码 | CUDA 文件 | C/C++ 行 |
|---|---:|---:|---:|---:|
| TensorRT-LLM(本篇) | 2,452,198 | 792,216 | 429 | 598,382 |
| SGLang | 2,317,985 | 1,254,674 | 246 | 51,979 |
| vLLM | 1,914,984 | 932,277 | 169 | 55,159 |

TRT-LLM 是三者里 C/C++ 代码量级唯一一个和 Python 代码量同一个数量级的(59.8 万 vs 79.2 万行,C/C++ 相当于 Python 的 75%);SGLang 的 C/C++(5.2 万行)只占 Python 产品码(125 万行)的 4%,vLLM 是 5.9%。这条数字支持 `## 6` 前面几节反复论证的结论——TRT-LLM 不是"C++/Python 二选一编译式引擎"退化成了"纯 Python 引擎",而是把 C++ 集中用在了调度决策、KV 传输这类历史上打磨过的算法模块,**"编译式"这个标签的历史指向(把整个模型计算图编译成引擎)已经不成立了,但"重度使用 C++ 实现关键路径"这个特征并没有消失,只是换了个作用对象**。这也是本篇标题"编译式引擎已经不编译了"这句话背后更精确的意思:不编译的是**模型计算图**,C++ 依赖程度并没有降到和 vLLM/SGLang 同一水平。

### 一个综合观察:两种引擎在"往哪个方向下沉复杂度"上选择不同

把`## 5`决策 3(调度器)、`## 3.5`(`CapacitySchedulerPolicy._to_pybind()`)、上面的 KV 传输运行时三件事放在一起看,能看出一条贯穿全篇的主线:**TRT-LLM 把"决策"类逻辑(调度策略、KV 传输策略)默认下沉到 C++,把"计算图变换"类逻辑(torch.compile)留在 Python 且默认关闭;vLLM 则正相反——调度决策留在 Python,计算图编译默认打开。** 这不是谁更"编译式"的简单排序,而是两个团队对"哪类逻辑值得用更底层的语言换稳定性能,哪类逻辑值得用更高层的语言换迭代速度"给出了不同答案。TRT-LLM 押注"batching/传输策略这种历史上被反复打磨过的算法,C++ 实现的边际收益更大";vLLM 押注"模型计算图这种随模型架构频繁变化的部分,用 Python 生态的 torch.compile 换来更快的新模型接入速度更划算"。

## 7. 踩坑与反直觉

1. **`attn_backend` 默认值就叫 `"TRTLLM"`,但它不是被移除的那个 TensorRT engine 后端。**`tensorrt_llm/llmapi/llm_args.py:5314`-`5321` 的 `attn_backend: str = Field(default='TRTLLM', ...)`,对应 `tensorrt_llm/_torch/attention_backend/trtllm.py:1396` 的 `class TrtllmAttention(AttentionBackend[...])`,内部通过 `torch.ops.trtllm.*`(比如 `tensorrt_llm/_torch/attention_backend/trtllm.py:2202` 的 `torch.ops.trtllm.mla_rope_append_paged_kv_assign_q`)调用 TRT-LLM 自家 CUDA/cutlass kernel 库(`libtensorrt_llm.so`)。这是一个**纯粹的命名巧合陷阱**:"TRTLLM"在这里是一个 PyTorch 自定义算子后端的名字,和已经被删除的"TensorRT engine 编译后端"没有任何关系,读代码/读配置时很容易被这个字符串带回旧印象。
2. **在 `_check_arguments` 里能看到"两个时代"的代码并存,但只有一半可达。**`tensorrt_llm/llmapi/llm.py:1571`-`1636` 这整个方法前半段(pytorch/`_autodeploy` 分支,`1575`-`1584`)和后半段(读 `build_config` 的旧逻辑,`1586`-`1630`)物理上写在同一个方法体里,后半段引用的 `self.args.build_config` 在 `TorchLlmArgs`/`AutoDeployLlmArgs` 上根本不存在——这是"死代码不等于代码被删除"的一个具体样本,`_verify.py` 式的静态检查工具不会告诉你这一点,只有跟着分支逻辑走一遍才知道。
3. **CLI 里 `--backend` 选项本身被标了 `status="deprecated"`**(`tensorrt_llm/commands/serve.py:952`),而不是只有 `_autodeploy` 这个取值被弃用——说明项目的长期方向是连"选择 backend"这个动作本身都可能被收掉(将来只剩 pytorch 一条路,不需要选)。
4. **CUDA Graph 捕获默认开着,但每个捕获的 batch size 桶要占用到 200MB 显存。**`tensorrt_llm/llmapi/llm_args.py:5180`-`5188` 的字段说明原文:"Note that each CUDA graph can use up to 200 MB of extra memory"——这不是免费的默认行为,`cuda_graph_config.batch_sizes` 配置的桶越多,静态占用的显存越多,这个代价在配置项的 docstring 里就写明了,不用等到 OOM 才发现。
5. **"60 条路由"这个数字如果不看源码,很容易被误解成"这个服务有 60 个能力"。**实际上任何一个 `trtllm-serve` 进程按角色最多暴露其中一组(`## 2.1`),标准聚合服务只有 20 条。`_lab/api_surface.json` 的静态扫描口径(AST 扫全仓库 `add_api_route` 调用)对"这个仓库支持多少种 HTTP 行为"是准确的,但对"某一个具体部署暴露多少接口"是一个上界,不是实际值——这条经验对本库其它篇目统计"路由数"时同样适用,拿到一个数字先确认它是静态扫描口径还是运行时口径。
6. **`docs/source/legacy/` 目录本身就是一份"removal 有多彻底"的证据**——这个目录下有 29 个 `.md` 文件(`find docs/source/legacy -name "*.md" | wc -l` 数出来的),其中至少 6 个文件直接含有"legacy TensorRT backend has been removed"或 caution 警告文本(grep 口径,数值会随措辞变化略有出入,但量级说明这不是一两个孤立页面的事)。这些文档没有被物理删除,而是整体搬进了一个明确标注"仅供交叉引用"的子目录——是本篇反复强调的"移除≠物理删干净,但入口分支已经不可达"这条原则在文档层面的对应物。
7. **`_to_pybind()` 这个方法名是另一处命名遗留**(`## 3.5`)——TRT-LLM 的 C++ 绑定技术已经从 pybind11 迁移到了 nanobind(`cpp/tensorrt_llm/nanobind/` 目录,`## 2` 已引用其中两个文件;仓库里已经找不到 `cpp/tensorrt_llm/pybind/` 目录),但配置类里转换到 C++ 枚举的方法还叫 `_to_pybind`,不是 `_to_nanobind`。这是纯粹的历史命名惯性,不影响功能,但和 `## 7` 踩坑 1 的 `attn_backend='TRTLLM'` 是同一类陷阱——字符串/方法名没有跟着底层实现的变迁而更新,读代码时不能望文生义。

## 8. 可改进点

1. **清理 `_check_arguments` 里已不可达的 `build_config` 分支**(`tensorrt_llm/llmapi/llm.py:1586`-`1630`)。既然 `backend` 已经在方法入口处收窄成只有两个可达取值,这段代码除了增加阅读负担和误导("`self.args.build_config` 看起来像是一个存在的属性")之外没有实际作用,可以直接删除或改成一个明确的 `assert False, "unreachable: legacy engine path removed"`。
2. **`LlmBuildStats` 里的 `engine_dir`/`cache_hitted` 字段命名具有误导性**(`tensorrt_llm/llmapi/llm_utils.py:466`、`474`)——字段名和注释都在说"engine",但当前唯一可达路径下这两个字段永远是默认值,不会被赋值。如果保留是为了向后兼容序列化格式,至少可以在字段旁补一行注释说明"pytorch/`_autodeploy` 路径下恒为默认值,仅为兼容历史序列化格式保留"。
3. **文档层面,`attn_backend='TRTLLM'` 这个默认值容易和已移除的 TensorRT engine 后端混淆**(见 `## 7` 踩坑 1)——`tensorrt_llm/llmapi/llm_args.py:5314`-`5321` 的字段 description 目前只写"Attention backend to use.",可以补一句明确区分"这是一个 PyTorch 自定义算子内核族的名字,与已移除的 TensorRT engine 编译后端无关"。
4. **`api_surface.json` 这类静态扫描工具的输出,建议加一个"互斥分组"标注**(对应 `## 7` 踩坑 5)——`_lab/api_surface.py` 目前把 `add_api_route` 调用当成扁平列表抽取,如果能顺带记录"这条路由是在哪个 `register_*` 方法里、这个方法是否受 `if/elif` 互斥分支保护",下游做跨引擎对比时就不会拿一个上界数字当实际运行时数字用——这个改进不需要改 TRT-LLM 上游代码,只需要改本库 `_lab/api_surface.py` 的抽取逻辑,属于本库自己能落地的部分。
5. **`_to_pybind()` 方法名可以在下一次涉及绑定层的重构里顺手改成中性名字**(比如 `_to_cpp_enum()`),消除`## 7` 踩坑 6 里描述的命名惯性——这是一个典型的"低风险、高可读性收益"的小改动,不涉及行为变化,适合和其它清理工作捆绑提交(`AGENTS.md` 明确写了"No low-value busywork PRs"的类似原则,单独为改名开 PR 大概率不划算,但可以搭车)。
6. **`KvCacheConfig.sink_token_length`(`## 3.8`)这类"deprecated and ignored"字段,建议和 `## 5` 决策 2 的 `build_config` 清理一起批量盘点**——两者是同一类技术债(功能移除后字段留存),分别在配置系统的两个不同角落独立出现,说明这不是孤立的一次疏漏,而是这个代码库在快速移除功能时的一贯做法。与其逐个发现逐个修,不如写一个一次性的静态检查脚本,扫描所有 `description` 里含 "deprecated"/"removed"/"ignored" 字样但字段类型本身没有标 `status="deprecated"` 的情况,统一补上 status 标注——这比人工一个个找更不容易漏。

## 9. 自测题与延伸阅读

**自测题**

1. `tensorrt_llm.LLM(model=..., backend="tensorrt")` 在当前快照下会发生什么?为什么?(提示:`## 2` 表格第 2 行)
2. `CachedModelLoader.__call__()` 对 pytorch backend 返回的 `_engine_dir` 是什么值?这意味着什么样的"编译产物生命周期"?(提示:`## 4` ④)
3. in-flight batching 的默认调度决策,是在哪个语言实现的?怎么切换成另一种实现?切换的代价是什么?(提示:`## 5` 决策 3)
4. 为什么说"vLLM 默认在编译,TRT-LLM 默认不编译"这句话在这个快照上成立?各自的证据分别在哪一行?(提示:`## 6` 第一个表格)
5. `register_routes()` 用 `add_api_route()` 函数调用式而不是装饰器注册路由,举一个只有这种写法才能自然表达的场景。(提示:`## 5` 决策 4)
6. `attn_backend` 的默认值是什么字符串?为什么这个名字容易造成误解?
7. `_lab/api_surface.json` 统计的"60 条路由",和一个标准聚合服务实际暴露的路由数,为什么不是同一个数字?差异来自哪个运行时判断?(提示:`## 2.1`)
8. KV 传输运行时默认走 C++ 还是 Python?这和调度器的默认语言选择是同一种模式还是不同模式?(提示:`## 6` "KV 传输运行时"一节)
9. `KvCacheConfig.sink_token_length` 字段还在不在配置类里?它对应的 StreamingLLM 功能在 PyTorch 后端还支不支持?这和 `build_config` 的情况是不是同一类模式?(提示:`## 3.8`、`## 8` 改进点 6)

**延伸阅读**

- [[00-总览与阅读地图]]——本库立场与已知硬结论,理解本篇"反直觉"结论为何值得强调的背景
- [[_PLAN]]——十段式骨架与双链名册,续写本库其它篇目前先读这个
- [[03-vLLM-调度器解剖]]——同位对照里提到的 vLLM 纯 Python 调度器实现细节,想深入对比可从这篇入手
