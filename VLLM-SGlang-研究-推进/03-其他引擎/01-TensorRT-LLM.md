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

**两条后端路线,现在的答案**(源码为证,`## 0` 已给出核心引用):历史上 TRT-LLM 有两条根本不同的执行路线——(a) 把 HF checkpoint 转换成 TRT-LLM checkpoint 格式,再用 `trtllm-build` 编译出针对特定 GPU 架构、特定并行度、特定精度的 `.engine` 文件,运行时加载这个二进制 engine 跑;(b) 直接用 PyTorch eager/graph 模式跑 HF 权重,不产出任何编译产物。**当前快照里只剩 (b)。** (a) 整条链路——`trtllm-build`/`trtllm-refit`/`trtllm-prune` 命令、`TrtLlmArgs` 类、`tensorrt_llm._tensorrt_engine.LLM`、per-model 的 `convert_checkpoint.py`——已经被整体删除,只在 `docs/source/legacy/` 下留了交叉引用文档,首行就是警告 banner(`docs/source/legacy/architecture/workflow.md:3`-`4`:"The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.",文档所述)。这条演进本身也留了痕迹:`docs/source/legacy/torch.md:5`-`9` 是 PyTorch 后端还处于"currently in beta"阶段时写的旧文档("The PyTorch backend of TensorRT LLM is available in version 0.17 and later"),对照 `AGENTS.md` 里当前的架构表把 PyTorch 标成"Default",能看出这是一条从 beta 走到唯一主线的完整轨迹。

## 2. 代码地图(文件 → 职责,带行号)

按"公开 API 层 → 编排层 → 执行层 → C++ 运行时"四层组织,共 16 条引用:

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

**本篇范围边界**:PyExecutor 内部的 KV 缓存分配细节、注意力后端矩阵（TRTLLM/FlashInfer/VANILLA 等具体 kernel 选择)、投机解码实现,不在本篇展开——本篇只回答"这个引擎的整体分层与编译式/解释式这条根本设计选择",子系统细节留给本库后续篇目补齐(参考 `_PLAN.md` §4 的篇目占位,当前均为规划中)。

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

## 4. 主流程走读

一次 `trtllm-serve <hf_model>` 到能接受 HTTP 请求,要穿过下面这条链路:

① **CLI 解析**:`tensorrt_llm/commands/serve.py` 的 `serve` 命令(`click.command`)解析 `--backend`(默认 `pytorch`,`tensorrt_llm/commands/serve.py:948`)等参数,调用 `get_llm_args()`(`tensorrt_llm/commands/serve.py:197`)把 CLI flag 收敛成一个 dict。

② **构造 `LLM` 对象**:`tensorrt_llm/commands/serve.py:662`-`664`——`if backend == 'pytorch': llm_args.pop("build_config", None); llm = PyTorchLLM(**llm_args)`。注意这里显式 `pop` 掉 `build_config`——这是给旧配置文件/脚本兼容用的清理动作,再次印证 `build_config` 是遗留字段(`## 5` 决策 2 会展开)。`PyTorchLLM` 就是 `tensorrt_llm.LLM`(`tensorrt_llm/commands/serve.py:29` 的别名导入)。

③ **`LLM.__init__` → `BaseLLM.__init__`**(`tensorrt_llm/llmapi/llm.py:1933` → `tensorrt_llm/llmapi/llm.py:1769` → `tensorrt_llm/llmapi/llm.py:354`):校验 backend、把 kwargs 收敛成 `TorchLlmArgs` 实例、设置 MPI session。

④ **`_build_model()`**(`tensorrt_llm/llmapi/llm.py:1835`,`_TorchLLM` 覆写):调 `CachedModelLoader`(`tensorrt_llm/llmapi/llm_utils.py:362`)下载/定位 HF checkpoint,返回 `(None, hf_model_dir)`(`tensorrt_llm/llmapi/llm_utils.py:448`);随后创建 tokenizer、`input_processor`;最后调 `self._executor_cls.create(...)`(`tensorrt_llm/llmapi/llm.py:1896`,即 `GenerationExecutor.create`)。

⑤ **`GenerationExecutor.create()`**(`tensorrt_llm/executor/executor.py:543`):这里是编排层的分岔口——按 `orchestrator_type` 决定用 Ray(`_create_ray_executor`)、RPC(`_create_rpc_executor`)、还是 MPI/IPC(`_create_ipc_executor`,`tensorrt_llm/executor/executor.py:648`)拉起 worker 进程/线程。多机部署时这一层还要处理 `TLLM_EXECUTOR_ATTACH_INFO` 这种"附着到已运行 worker"的场景(`tensorrt_llm/executor/executor.py:579`-`603`),用于多前端(multi-frontend)共享一个后端 executor。

⑥ **worker 进程内构造 `PyExecutor`**:`tensorrt_llm/executor/base_worker.py:165`-`216` 的 `_create_py_executor()` 调 `create_py_executor()`(`tensorrt_llm/_torch/pyexecutor/py_executor_creator.py:337`),后者依次:加载模型配置、构造 `ModelEngine`(持有实际的 `nn.Module`)、构造调度器(默认 `BindCapacityScheduler`+`BindMicroBatchScheduler`,`tensorrt_llm/_torch/pyexecutor/_util.py:3082`-`3094`)、构造 `KVCacheManager`、构造 `Sampler`,最后 `return PyExecutor(...)`(`tensorrt_llm/_torch/pyexecutor/py_executor.py:549`)。

⑦ **`PyExecutor` 事件循环**:文档所述(`docs/source/torch/arch_overview.md` "The single-step flow of PyExecutor")——取新请求→调度→模型前向→解码→处理完成请求,循环往复。调度这一步就是⑤⑥里组装好的 `BindCapacityScheduler`/`BindMicroBatchScheduler`,底层调用的是 C++ `CapacityScheduler::operator()`(`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:708`)。这就是 in-flight batching(TRT-LLM 对连续批处理的官方叫法,`InflightBatchingStats` 类名与统计字段贯穿全代码库,例如 `tensorrt_llm/_torch/pyexecutor/py_executor.py:2012`)真正决策发生的地方——**默认路径是 C++,不是 Python**。

⑧ **HTTP 服务层**:`OpenAIServer.__init__` 持有一个 `GenerationExecutor`(通过 `self.generator`),`register_routes()`(`tensorrt_llm/serve/openai_server.py:962`)把 47 条路由挂到 FastAPI `app` 上;请求进来后 `openai_chat`/`openai_completion` 等 handler 把 OpenAI 协议对象转换成 TRT-LLM 内部的 `SamplingParams` + prompt token id,喂给 `self.generator.generate_async(...)`,本质上是往 `PyExecutor` 的请求队列里塞了一条 `LlmRequest`。

## 5. 设计决策与代价

### 决策 1:PyTorch 是唯一后端,不再提供"engine 编译"作为可选项

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

vLLM 与 SGLang 各自的 HTTP 层基本是单一入口文件族(`vllm:vllm/entrypoints/openai/`、`sglang:python/sglang/srt/entrypoints/http_server.py`)。TRT-LLM 在 `tensorrt_llm/serve/` 下拆成了三个独立 FastAPI 应用——`OpenAIServer`(`tensorrt_llm/serve/openai_server.py:347`,聚合/单机场景,47 条路由)、`OpenAIDisaggServer`(`tensorrt_llm/serve/openai_disagg_server.py:139`,PD 分离网关,转发 `/v1/completions`/`/v1/chat/completions` 到 context/generation 两组后端)、`CoordinatorServer`(`tensorrt_llm/serve/coordinator_server.py:52`,多副本间的 `/select`/`/finish` 路由协调)——这和 TRT-LLM 把 PD 分离、多前端 attach(`## 4` ⑤ 提到的 `TLLM_EXECUTOR_ATTACH_INFO`)当作一等公民的产品定位是一致的:vLLM/SGLang 的 PD 分离更多是在同一个 server 类里加分支,TRT-LLM 是从进程粒度就切开了。

## 7. 踩坑与反直觉

1. **`attn_backend` 默认值就叫 `"TRTLLM"`,但它不是被移除的那个 TensorRT engine 后端。**`tensorrt_llm/llmapi/llm_args.py:5314`-`5321` 的 `attn_backend: str = Field(default='TRTLLM', ...)`,对应 `tensorrt_llm/_torch/attention_backend/trtllm.py:1396` 的 `class TrtllmAttention(AttentionBackend[...])`,内部通过 `torch.ops.trtllm.*`(比如 `tensorrt_llm/_torch/attention_backend/trtllm.py:2202` 的 `torch.ops.trtllm.mla_rope_append_paged_kv_assign_q`)调用 TRT-LLM 自家 CUDA/cutlass kernel 库(`libtensorrt_llm.so`)。这是一个**纯粹的命名巧合陷阱**:"TRTLLM"在这里是一个 PyTorch 自定义算子后端的名字,和已经被删除的"TensorRT engine 编译后端"没有任何关系,读代码/读配置时很容易被这个字符串带回旧印象。
2. **在 `_check_arguments` 里能看到"两个时代"的代码并存,但只有一半可达。**`tensorrt_llm/llmapi/llm.py:1571`-`1636` 这整个方法前半段(pytorch/`_autodeploy` 分支,`1575`-`1584`)和后半段(读 `build_config` 的旧逻辑,`1586`-`1630`)物理上写在同一个方法体里,后半段引用的 `self.args.build_config` 在 `TorchLlmArgs`/`AutoDeployLlmArgs` 上根本不存在——这是"死代码不等于代码被删除"的一个具体样本,`_verify.py` 式的静态检查工具不会告诉你这一点,只有跟着分支逻辑走一遍才知道。
3. **CLI 里 `--backend` 选项本身被标了 `status="deprecated"`**(`tensorrt_llm/commands/serve.py:952`),而不是只有 `_autodeploy` 这个取值被弃用——说明项目的长期方向是连"选择 backend"这个动作本身都可能被收掉(将来只剩 pytorch 一条路,不需要选)。
4. **CUDA Graph 捕获默认开着,但每个捕获的 batch size 桶要占用到 200MB 显存。**`tensorrt_llm/llmapi/llm_args.py:5180`-`5188` 的字段说明原文:"Note that each CUDA graph can use up to 200 MB of extra memory"——这不是免费的默认行为,`cuda_graph_config.batch_sizes` 配置的桶越多,静态占用的显存越多,这个代价在配置项的 docstring 里就写明了,不用等到 OOM 才发现。

## 8. 可改进点

1. **清理 `_check_arguments` 里已不可达的 `build_config` 分支**(`tensorrt_llm/llmapi/llm.py:1586`-`1630`)。既然 `backend` 已经在方法入口处收窄成只有两个可达取值,这段代码除了增加阅读负担和误导("`self.args.build_config` 看起来像是一个存在的属性")之外没有实际作用,可以直接删除或改成一个明确的 `assert False, "unreachable: legacy engine path removed"`。
2. **`LlmBuildStats` 里的 `engine_dir`/`cache_hitted` 字段命名具有误导性**(`tensorrt_llm/llmapi/llm_utils.py:466`、`474`)——字段名和注释都在说"engine",但当前唯一可达路径下这两个字段永远是默认值,不会被赋值。如果保留是为了向后兼容序列化格式,至少可以在字段旁补一行注释说明"pytorch/`_autodeploy` 路径下恒为默认值,仅为兼容历史序列化格式保留"。
3. **文档层面,`attn_backend='TRTLLM'` 这个默认值容易和已移除的 TensorRT engine 后端混淆**(见 `## 7` 踩坑 1)——`tensorrt_llm/llmapi/llm_args.py:5314`-`5321` 的字段 description 目前只写"Attention backend to use.",可以补一句明确区分"这是一个 PyTorch 自定义算子内核族的名字,与已移除的 TensorRT engine 编译后端无关"。

## 9. 自测题与延伸阅读

**自测题**

1. `tensorrt_llm.LLM(model=..., backend="tensorrt")` 在当前快照下会发生什么?为什么?(提示:`## 2` 表格第 2 行)
2. `CachedModelLoader.__call__()` 对 pytorch backend 返回的 `_engine_dir` 是什么值?这意味着什么样的"编译产物生命周期"?(提示:`## 4` ④)
3. in-flight batching 的默认调度决策,是在哪个语言实现的?怎么切换成另一种实现?切换的代价是什么?(提示:`## 5` 决策 3)
4. 为什么说"vLLM 默认在编译,TRT-LLM 默认不编译"这句话在这个快照上成立?各自的证据分别在哪一行?(提示:`## 6` 第一个表格)
5. `register_routes()` 用 `add_api_route()` 函数调用式而不是装饰器注册路由,举一个只有这种写法才能自然表达的场景。(提示:`## 5` 决策 4)
6. `attn_backend` 的默认值是什么字符串?为什么这个名字容易造成误解?

**延伸阅读**

- [[00-总览与阅读地图]]——本库立场与已知硬结论,理解本篇"反直觉"结论为何值得强调的背景
- [[_PLAN]]——十段式骨架与双链名册,续写本库其它篇目前先读这个
- [[03-vLLM-调度器解剖]]——同位对照里提到的 vLLM 纯 Python 调度器实现细节,想深入对比可从这篇入手
