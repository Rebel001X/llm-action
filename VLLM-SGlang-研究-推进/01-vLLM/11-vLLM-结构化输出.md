# vLLM 结构化输出与受限解码

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：语法编译异步在线程池、bitmask 在 CPU 与 GPU 前向重叠算。

## 0. 结论先行

- 结构化输出子系统的真正入口不是一个"过滤器"，而是一台**双时钟机器**：`SamplingParams.verify()`（`vllm/sampling_params.py:782-798`）在请求进入引擎**之前**做一次"能不能编译"的探测性校验（不产出可用的语法对象，只是把 `_backend` 定死并在 `backend="auto"` 时试出真正要用哪个后端）；真正被采样阶段消费的语法对象，在请求被 `EngineCore.preprocess_add_request()` 接收时才第二次编译（`vllm/v1/structured_output/__init__.py:115-176`），且这次编译默认扔进线程池异步跑，不占 EngineCore 的忙循环。
- **后端不是运行时可选项，是进程级单例。** `StructuredOutputManager.backend` 只在第一次遇到结构化输出请求时惰性创建一次（`vllm/v1/structured_output/__init__.py:130-165`），"NOTE: We only support a single backend" 是源码自己写的注释——同一个 vLLM 实例里，xgrammar 和 guidance 不能同时给两个请求分别服务。
- **bitmask 的计算在 CPU，应用在 GPU，两者之间夹着一次跨进程 RPC 的等待窗口，这个窗口就是"重叠"发生的地方。** `EngineCore.step()`（`vllm/v1/engine/core.py:597-627`）先用 `non_block=True` 把 `execute_model()` 扔给 worker 进程（此调用立即返回一个 `Future`，不等 GPU 算完），然后**在同一行代码之后**立刻调用 `self.scheduler.get_grammar_bitmask(...)`——这是纯 CPU 计算，在 GPU 跑 forward 的同时，EngineCore 自己的进程在算语法状态机要不要往前推、bitmask 每一位该填 0 还是 1。算完才 `future.result()` 等 GPU，再把算好的 `grammar_output` 传给 `sample_tokens()`。
- **投机解码的草稿 token 会先被语法"预审"再被调度，而不是等生成完再秋后算账。** `Scheduler.update_draft_token_ids()`（`vllm/v1/core/sched/scheduler.py:2264-2283`）在草稿 token 送去参与验证之前，先用 `grammar.validate_tokens(...)`（只读检查，不推进 FSM）把不合语法的后缀砍掉；`update_draft_token_ids_in_output()`（`vllm/v1/core/sched/scheduler.py:2284-2320`）把被砍掉的位置用 `-1` 填齐，`grammar_bitmask()` 遇到 `-1` 直接跳过该位置的约束（`vllm/v1/structured_output/__init__.py:304-306`）。
- **前缀缓存命中不需要"恢复"语法状态，因为语法状态从来没有被 KV 缓存管过。** 语法 FSM（`matcher`/`guide`/`ll_matcher` 等）是 Python 侧挂在 `Request.structured_output_request.grammar` 上的**长生命周期对象**，只在 `Scheduler.update_from_output()` 里被真实输出 token 推进一次（`vllm/v1/core/sched/scheduler.py:1918-1943`）；即使请求被抢占导致 `num_computed_tokens` 清零、KV 块被释放、重新排队后靠前缀缓存命中跳过大段重算（`vllm/v1/core/sched/scheduler.py:1347-1388` 的 `_preempt_request` 完全不碰 `structured_output_request`），语法对象本身还在原地，从未被重建——这不是一个需要小心维护的正确性机制，而是两套状态**物理上没有交叠**的自然结果。
- **vLLM 没有跳跃式解码（jump-forward decoding）**，尽管所用的两个后端库（xgrammar、llguidance）都原生支持它——`vllm/v1/structured_output/backend_xgrammar.py:146-147` 的注释直接写着"for jump-forward decoding"却从未调用对应 API，`vllm/v1/structured_output/backend_guidance.py:177-182` 更直白地留了一条 `TODO - Add jump decoding support in the future`。这条线在 `## 6` 展开。

证据分级：本篇除标注"文档所述"或"本库推断"外，均为「源码为证」，带 `文件:行`。不产出任何实测数字（本机无 GPU），压缩率/加速比只讲机制不编数字。

## 1. 它在系统里的位置

结构化输出横跨三个进程边界，理解这条子系统必须先认清这三段：

1. **前端/输入处理线程**：`InputProcessor._validate_params()`（`vllm/v1/engine/input_processor.py:84-102`）在请求变成 `EngineCoreRequest` 之前调用 `SamplingParams.verify()`（`vllm/sampling_params.py:782-798`），内部走到 `_validate_structured_outputs()`（`vllm/sampling_params.py:1020-1063`）。这一步只做**语法能不能被目标后端解析**的探测编译（例如 `xgr.Grammar.from_json_schema(schema)`），编译出的对象立刻丢弃，不进入运行时——它的唯一产出是把 `sampling_params.structured_outputs._backend` 钉死成一个具体后端名字。
2. **EngineCore 的"输入处理线程"**（与主忙循环并行的一条独立线程，见 `vllm/v1/engine/core.py:985-987` 的注释"could be directly used in input processing thread to allow request initialization running in parallel with Model forward"）：`preprocess_add_request()`（`vllm/v1/engine/core.py:982-1004`）调用 `StructuredOutputManager.grammar_init()`（`vllm/v1/structured_output/__init__.py:115-176`），这里才真正编译出运行时要用的语法对象（`StructuredOutputGrammar` 实例），默认异步扔进线程池。
3. **EngineCore 的主忙循环**（`vllm/v1/engine/core.py:597-627`）：每步 `schedule()` 之后，`Scheduler.get_grammar_bitmask()`（`vllm/v1/core/sched/scheduler.py:1720-1742`）在 CPU 上把本步所有结构化输出请求的语法状态推进、bitmask 填好；随后这个 bitmask 被送到 worker 进程，在 `sample_tokens()` 里通过 `apply_grammar_bitmask()`（`vllm/v1/structured_output/utils.py:87-176`）叠加到 logits 上，再交给 `Sampler` 采样。

调用链自上而下：`Scheduler`（持有 `StructuredOutputManager` 实例，见 `vllm/v1/core/sched/scheduler.py:78,103`）决定"这一步谁需要 bitmask"；`GPUModelRunner.sample_tokens()`（`vllm/v1/worker/gpu_model_runner.py:4663-4696`，[[06-vLLM-模型执行与CUDA-Graph]] 已详细解剖这个方法的整体结构）消费 bitmask 并应用到 logits；`GPUModelRunner.execute_model()`——纯 forward，不碰结构化输出——与 `sample_tokens()` 之间的两次 RPC 拆分，正是本篇 `## 4` 要讲的重叠窗口的物理载体。

## 2. 代码地图（文件 → 职责，带行号）

| 文件:行 | 职责 |
|---|---|
| `vllm/v1/structured_output/__init__.py:36` | `class StructuredOutputManager`——引擎级管理器，持有唯一的后端实例 |
| `vllm/v1/structured_output/__init__.py:115` | `grammar_init()`——请求进来时创建/选定后端、提交编译（同步或异步） |
| `vllm/v1/structured_output/__init__.py:220` | `grammar_bitmask()`——每步在 CPU 上批量推进 FSM 并填充 bitmask 张量 |
| `vllm/v1/structured_output/backend_types.py:19` | `class StructuredOutputOptions`——JSON/JSON_OBJECT/REGEX/GRAMMAR/CHOICE/STRUCTURAL_TAG 六种约束类型 |
| `vllm/v1/structured_output/backend_types.py:31` | `class StructuredOutputGrammar(ABC)`——单请求语法对象的抽象接口 |
| `vllm/v1/structured_output/backend_types.py:99` | `class StructuredOutputBackend(ABC)`——引擎级后端的抽象接口，`compile_grammar()`/`allocate_token_bitmask()` |
| `vllm/v1/structured_output/backend_xgrammar.py:37` | `class XgrammarBackend`——默认后端，`GrammarCompiler` 自带 `cache_enabled=True` |
| `vllm/v1/structured_output/backend_guidance.py:89` | `class GuidanceBackend`——基于 `llguidance`，唯一原生支持全部 6 种约束类型的后端 |
| `vllm/v1/structured_output/backend_outlines.py:54` | `class OutlinesBackend`——把 JSON/CHOICE 都转成正则表达式再编译 |
| `vllm/v1/structured_output/backend_lm_format_enforcer.py:96` | `class LMFormatEnforcerBackend`——不支持 GRAMMAR，且显式拒绝与投机解码同时使用 |
| `vllm/v1/structured_output/request.py:22` | `class StructuredOutputRequest`——请求级容器，`grammar` 字段可以是 `Future`/对象/`Exception` 三态之一 |
| `vllm/v1/structured_output/request.py:50` | `_check_grammar_completion()`——用 `future.result(timeout=0.0001)` 做非阻塞轮询 |
| `vllm/v1/structured_output/utils.py:87` | `apply_grammar_bitmask()`——把 bitmask 重排到 batch 顺序，异步拷到 GPU，调用 `xgr.apply_token_bitmask_inplace` |
| `vllm/v1/structured_output/utils.py:49` | `compile_regex_with_timeout()`——正则编译加超时墙，防 ReDoS |
| `vllm/v1/worker/gpu/structured_outputs.py:39` | `class StructuredOutputsWorker`——**实验性** Model Runner V2 里用 Triton kernel 应用 bitmask 的替代实现 |
| `vllm/config/structured_outputs.py:12` | `StructuredOutputsBackend` 类型——`"auto"/"xgrammar"/"guidance"/"outlines"/"lm-format-enforcer"` |
| `vllm/sampling_params.py:1020` | `_validate_structured_outputs()`——前端侧的探测性编译校验，`auto` 的回退链在这里 |
| `vllm/v1/engine/input_processor.py:97` | `params.verify(...)` 调用点——结构化输出校验被触发的地方 |
| `vllm/v1/engine/core.py:982` | `preprocess_add_request()`——在"输入处理线程"里调用 `grammar_init()` |
| `vllm/v1/engine/core.py:609` | `step()` 里 `execute_model(..., non_block=True)` 与 `get_grammar_bitmask()` 相邻两行——重叠窗口的物理位置 |
| `vllm/v1/core/sched/scheduler.py:1720` | `get_grammar_bitmask()`——调度器侧的 bitmask 计算入口 |
| `vllm/v1/core/sched/scheduler.py:1918` | `update_from_output()` 里 `grammar.accept_tokens(...)`——语法状态机被真实输出 token 推进的唯一位置 |
| `vllm/v1/core/sched/scheduler.py:2264` | `update_draft_token_ids()`——投机解码草稿 token 的语法预审 |
| `vllm/v1/core/sched/output.py:301` | `class GrammarOutput`——跨 RPC 传递的 bitmask 载荷 |
| `vllm/v1/request.py:114` | 新请求若带结构化输出参数，初始状态直接是 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` |

以上 24 条均可在 `_src/vllm/` 对应文件里核实；下文 `## 4`~`## 7` 还会引用更多具体行号。

## 3. 核心数据结构

**`StructuredOutputOptions`**（`vllm/v1/structured_output/backend_types.py:19-25`）是六种约束类型的封闭集合：

```python
# vllm/v1/structured_output/backend_types.py:19-25
class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()
```

`StructuredOutputKey = tuple[StructuredOutputOptions, str]`（`vllm/v1/structured_output/backend_types.py:28`）——类型 + 序列化后的语法文本，是 `compile_grammar()` 的入参，也是 outlines 后端内部缓存的 key 组成部分（`## 5.2` 展开）。

**`StructuredOutputGrammar`**（`vllm/v1/structured_output/backend_types.py:31-96`）是请求级语法对象的抽象接口，六个抽象方法定死了所有后端必须实现的能力面：`accept_tokens()`（推进 FSM）、`validate_tokens()`（只读检查，不推进）、`rollback()`（回滚 N 个 token）、`fill_bitmask()`（往指定 batch 行写 bitmask）、`is_terminated()`、`reset()`。四个后端（xgrammar/guidance/outlines/lm-format-enforcer）各自的 `XgrammarGrammar`/`GuidanceGrammar`/`OutlinesGrammar`/`LMFormatEnforcerGrammar` 都是这个接口的具体实现，内部包一个自己库的匹配器对象（`xgr.GrammarMatcher`、`llguidance.LLMatcher`、`oc.Guide`、`lmformatenforcer.TokenEnforcer`）。

**`StructuredOutputRequest`**（`vllm/v1/structured_output/request.py:22-79`）是请求级的语法状态容器：

```python
# vllm/v1/structured_output/request.py:21-37（节选字段）
class StructuredOutputRequest:
    params: StructuredOutputsParams
    _grammar: (
        Future[StructuredOutputGrammar] | StructuredOutputGrammar | Exception | None
    ) = None
    reasoning_ended: bool | None = None
    reasoning_end_token_index: int | None = None
    reasoner: "ReasoningParser | None" = None
```

`_grammar` 字段的类型标注本身就说明了这个对象的三态生命周期：**编译中**（`Future`）→ **编译完成**（`StructuredOutputGrammar`）或**编译失败**（`Exception`）。`grammar` 属性的 getter（`vllm/v1/structured_output/request.py:66-69`）每次访问都调用 `_check_grammar_completion()`（`:50-59`），用 `self._grammar.result(timeout=0.0001)` 做**非阻塞**探测——超时说明还没编译完，直接返回 `False`；这是调度器判断"这个请求能不能开始跑"的唯一依据，不需要任何回调或事件通知机制。

**`GrammarOutput`**（`vllm/v1/core/sched/output.py:300-305`）是跨 RPC 边界传递的 bitmask 载荷：

```python
# vllm/v1/core/sched/output.py:300-306
class GrammarOutput:
    structured_output_request_ids: list[str]
    grammar_bitmask: "npt.NDArray[np.int32]"
```

之所以用 `np.ndarray` 而不是 `torch.Tensor` 跨进程传（`vllm/v1/structured_output/__init__.py:365-368` 的注释直说"much more efficient for serialization and deserialization when sending this to the GPU workers"），是因为 EngineCore 进程和 worker 进程之间走的是共享内存队列 RPC（`vllm/v1/executor/multiproc_executor.py:429` 的 `rpc_broadcast_mq.enqueue(...)`），`torch.Tensor` 的序列化开销比 numpy 数组高。

**`SchedulerOutput` 的两个结构化输出标记**（`vllm/v1/core/sched/output.py:249-255`）：

```python
# vllm/v1/core/sched/output.py:249-255
has_structured_output_requests: bool = False
pending_structured_output_tokens: bool = False
```

前者是"本步有没有请求需要 bitmask"的快速短路（`get_grammar_bitmask()` 第一行就检查它，`vllm/v1/core/sched/scheduler.py:1725-1726`）；后者只在启用异步调度时才可能为真（`vllm/v1/core/sched/async_scheduler.py:31-33`），标记"本步的输出 token 还是占位符，语法还不能推进"，`## 5.4` 展开它驱动的延迟采样逻辑。

**`StructuredOutputsConfig`**（`vllm/config/structured_outputs.py:17-74`）是引擎级、启动时定死的配置，`backend` 只是其中一个字段：`disable_any_whitespace`（`:26-30`）要求 JSON 输出全程无空白，仅 xgrammar/guidance 支持；`disable_additional_properties`（`:31-34`）让 guidance 后端在编译 JSON Schema 前主动补全 `additionalProperties: false`（`_walk_json_for_additional_properties()`，`vllm/v1/structured_output/backend_guidance.py:37-47`），目的是让 guidance 的默认行为和 xgrammar/outlines 对齐（后两者默认就不允许 schema 之外的字段）。这两个字段都有一个 `model_validator`（`vllm/config/structured_outputs.py:62-74`）在配置加载时就校验"配的后端支不支持这个开关"，配错了直接在引擎启动阶段报错，不会拖到第一个请求才发现。

## 4. 主流程走读

### 4.1 请求进入：两次编译，两个进程边界

一个带 `response_format` / `guided_json` 的请求要经过两次"编译尝试"：

第一次在 **API 请求校验阶段**，`SamplingParams.verify()`（`vllm/sampling_params.py:782-798`）调用 `_validate_structured_outputs()`（`:1020-1063`）。这一步先解析 `backend` 配置：

```python
# vllm/sampling_params.py:1046-1063（节选）
backend = structured_outputs_config.backend
if _backend := self.structured_outputs._backend:
    if backend != _backend and not (
        backend == "auto" and self.structured_outputs._backend_was_auto
    ):
        raise VLLMValidationError(...)
else:
    self.structured_outputs._backend = backend
```

若引擎启动时配置的是具体后端（`xgrammar`/`guidance`/`outlines`/`lm-format-enforcer`），直接调用对应的 `validate_xgrammar_grammar()` 等函数（`vllm/sampling_params.py:1118-1148`）——这些函数会**真的调用一次编译**（例如 `xgr.Grammar.from_json_schema(schema)`，见 `vllm/v1/structured_output/backend_xgrammar.py:356`），编译失败直接 `raise VLLMValidationError`，请求还没进引擎就被 400 拒绝。

若配置是 `"auto"`（默认值，`vllm/config/structured_outputs.py:21`），走一条显式的回退链（`vllm/sampling_params.py:1150-1193`）：

```python
# vllm/sampling_params.py:1156-1191（节选逻辑）
try:
    validate_xgrammar_grammar(self)
    self.structured_outputs._backend = "xgrammar"
except VLLMValidationError:
    skip_guidance = _is_non_tekken_mistral(tokenizer)
    if not skip_guidance and so_params.json:
        skip_guidance = has_guidance_unsupported_json_features(schema)
    if skip_guidance:
        validate_structured_output_request_outlines(self)
        self.structured_outputs._backend = "outlines"
    else:
        validate_guidance_grammar(self, tokenizer=_get_llg_tokenizer(tokenizer))
        self.structured_outputs._backend = "guidance"
self.structured_outputs._backend_was_auto = True
```

**优先级是 xgrammar > guidance > outlines**，且这个选择是**每个请求各自试一遍**（`_backend` 字段在 `SamplingParams` 上，不是全局单例）——但因为 `StructuredOutputManager.backend` 全局只建一次（`## 5.1` 展开），如果一个引擎实例先后收到两个 schema、一个能被 xgrammar 编译一个不能，第二个请求的 `_backend` 会被设成 `"guidance"`，可 `StructuredOutputManager.backend` 此时已经是 `XgrammarBackend` 实例了——`grammar_init()` 里 `if self.backend is None` 的判断（`vllm/v1/structured_output/__init__.py:130`）只在第一次建后端，后续请求即使 `_backend` 字段写着别的名字也不会重建后端对象，`_create_grammar()` 里 `assert self.backend is not None` 之后直接用已有的 `self.backend.compile_grammar(...)`，实际用的仍是最先建立的那个后端。这是"进程级单例"设计的一个直接推论：**`auto` 的每请求回退逻辑只在同一批请求的后端选择一致时才如预期工作**，`## 7` 展开这个坑。

第二次编译发生在请求真正被 EngineCore 接收时：`preprocess_add_request()`（`vllm/v1/engine/core.py:982-1004`）调用 `grammar_init()`：

```python
# vllm/v1/structured_output/__init__.py:167-176
grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar
if self._use_async_grammar_compilation:
    grammar = self.executor.submit(self._create_grammar, request)
else:
    try:
        grammar = self._create_grammar(request)
    except Exception as e:
        grammar = Future()
        grammar.set_exception(e)
request.structured_output_request.grammar = grammar
```

`_use_async_grammar_compilation` 默认为 `True`，只有 `distributed_executor_backend == "external_launcher"` 时才关闭（`vllm/v1/structured_output/__init__.py:47-56` 的注释解释：external_launcher 模式下每个 TP rank 各跑一个调度器，异步编译会让不同 rank 的 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING` 转换发生在不同时刻，打破 TP rank 间必须同步推进的确定性假设）。默认路径下，`self.executor.submit(...)` 把 `_create_grammar()`（`vllm/v1/structured_output/__init__.py:178-200`）扔进 `ThreadPoolExecutor`（线程数是 `(cpu_count()+1)//2`，`:77`），**不阻塞 `preprocess_add_request()` 这次调用本身**，`request.structured_output_request.grammar` 立刻被设成一个还没完成的 `Future`。

**编译不会卡住 EngineCore 的主忙循环**——这是设计上的硬保证，理由分两层：第一层，`grammar_init()` 本身跑在"输入处理线程"而不是主忙循环线程（`vllm/v1/engine/core.py:985-987` 的注释）；第二层，即使是这条输入处理线程，默认配置下也只是提交任务到线程池就返回，真正的编译在另一条工作线程上跑。请求在编译完成前的状态是 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`（`vllm/v1/request.py:114-115` 在构造时直接赋值，而不是先给 `WAITING` 再改），调度器每步用 `_try_promote_blocked_waiting_request()`（`vllm/v1/core/sched/scheduler.py:2817-2822`）非阻塞地探一次 `grammar is None`/是否是 `Exception`，编译完就转 `WAITING`，没编译完就跳过——**其他请求的调度完全不受影响**。

### 4.2 每步：CPU 算 bitmask 与 GPU 跑 forward 重叠

`EngineCore.step()`（`vllm/v1/engine/core.py:597-627`）是重叠机制的物理位置：

```python
# vllm/v1/engine/core.py:608-617
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with (...):
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
```

`execute_model(..., non_block=True)` 底层是 `MultiProcExecutor.collective_rpc()`（`vllm/v1/executor/multiproc_executor.py:373-448`）：把 `(method, args, kwargs)` 塞进共享内存队列 `rpc_broadcast_mq`（`:429`），立刻返回一个 `FutureWrapper`（`:445-448`），**不等 worker 进程回消息**。worker 进程收到消息后才真正开始跑模型 forward（可能落进 CUDA Graph replay，参见 [[06-vLLM-模型执行与CUDA-Graph]]）。

紧接着 `get_grammar_bitmask(scheduler_output)`（`vllm/v1/core/sched/scheduler.py:1720-1742`）在 EngineCore **自己的进程**里执行——这是纯 CPU 工作，和 worker 进程里跑的 GPU forward 是两个操作系统进程，物理上并行：

```python
# vllm/v1/core/sched/scheduler.py:1720-1742
def get_grammar_bitmask(self, scheduler_output):
    if not scheduler_output.has_structured_output_requests:
        return None
    structured_output_request_ids = [
        req_id for req_id in scheduler_output.num_scheduled_tokens
        if (req := self.requests.get(req_id))
        and (req.use_structured_output and not req.is_prefill_chunk)
    ]
    if not structured_output_request_ids:
        return None
    bitmask = self.structured_output_manager.grammar_bitmask(
        self.requests, structured_output_request_ids,
        scheduler_output.scheduled_spec_decode_tokens,
    )
    return GrammarOutput(structured_output_request_ids, bitmask)
```

`req.is_prefill_chunk` 的判断把还在做分块预填充的请求排除在外（`vllm/v1/core/sched/scheduler.py:1411-1413` 在 `_update_after_schedule()` 里维护这个标记）——**预填充阶段不需要 bitmask**，因为语法约束只作用于"这一步真的会采样出下一个 token"的位置，预填充的中间 chunk 不采样。

`grammar_bitmask()`（`vllm/v1/structured_output/__init__.py:220-368`）内部按批大小走两条路径：批量大且没开投机解码时，用 `ThreadPoolExecutor`（`self.executor_for_fillmask`，`:62-69`）把填充任务切成 16 个一组并行提交（`:250-279`）；否则串行填充，同时处理投机解码场景下每个请求要填多行 bitmask（一个草稿位置一行 + 一个 bonus token 位置一行）。填完之后转成 `numpy` 数组返回（`:365-368`）。

等 `future.result()` 拿到 worker 的 forward 结果（`model_output is None`，因为 `execute_model()` 在 worker 侧只算到 logits 就返回，[[06-vLLM-模型执行与CUDA-Graph]] `## 4.2` 已详细拆过这一步），再调用 `sample_tokens(grammar_output)`——这时候 `grammar_output` 早就算好在手边了，**不需要再等**。

### 4.3 GPU 侧：bitmask 怎么应用到 logits

`GPUModelRunner.sample_tokens()`（`vllm/v1/worker/gpu_model_runner.py:4663-4696`）收到 `grammar_output` 后：

```python
# vllm/v1/worker/gpu_model_runner.py:4693-4696
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

`apply_grammar_bitmask()`（`vllm/v1/structured_output/utils.py:87-176`）分三步：

1. **重排**：`grammar_output.grammar_bitmask` 里请求的顺序是调度器决定的顺序，未必和 `input_batch.req_ids`（GPU 侧 batch 的实际槽位顺序）一致，先按槽位顺序重新排一遍（`:118-142`），顺带把投机解码场景下每个请求占用的多行（bonus + spec 位置）对齐好。
2. **异步拷贝**：`grammar_bitmask.to(logits.device, non_blocking=True)`（`:145`）——CPU 侧的 pinned 内存（`sorted_bitmask_tensor` 构造时 `pin_memory=PIN_MEMORY`，`:131`）配 `non_blocking=True`，是 [[06-vLLM-模型执行与CUDA-Graph]] `## 5.2` 讲过的同一套 H2D 重叠手法。
3. **应用**：GPU 上调用 `xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)`（`:162`）——这是 xgrammar 库自带的 CUDA kernel（vLLM 不管后端是不是 xgrammar，bitmask 的应用统一走 xgrammar 的 kernel，因为 bitmask 的打包格式——每 32 个 vocab 位打包成一个 `int32`——是所有后端共享的约定，见 `## 5.3`）；CPU 张量场景下（`logits.is_cpu`）还要处理旧版 xgrammar CPU kernel 只认 `float32` 的兼容问题（`:167-176`）。

bitmask 形状是 `(num_logit_rows, cdiv(vocab_size, 32))`，`dtype=torch.int32`——`XgrammarBackend.allocate_token_bitmask()`（`vllm/v1/structured_output/backend_xgrammar.py:133-134`）直接委托给 `xgr.allocate_token_bitmask(max_num_seqs, vocab_size)`；`num_logit_rows` 的分配额度是 `max_batch_size * (1 + max_num_spec_tokens)`（`vllm/v1/structured_output/__init__.py:240-242`）——每个请求最多占 `1+num_speculative_tokens` 行，覆盖投机解码每一个候选位置。

`vllm/v1/worker/gpu/structured_outputs.py:39-163` 是**实验性** Model Runner V2（`vllm/v1/worker/gpu/README.md` 自称"under active development"）里的另一套实现——不依赖 xgrammar 的 kernel，改用 vLLM 自己写的 Triton kernel `_apply_grammar_bitmask_kernel`（`:126-163`），按 `(request, position)` 而不是绝对 batch 索引做映射，配合该 Runner 的"自适应验证"（adaptive verification）机制在 GPU 上动态决定每个请求的 logit 偏移。这条路径本篇不展开，只作为"未来 vLLM 可能不再依赖 xgrammar kernel 应用 bitmask"的线索标注。

### 4.4 投机解码里的语法预审

`Scheduler.update_draft_token_ids()`（`vllm/v1/core/sched/scheduler.py:2264-2283`）在草稿模型产出候选 token 之后、这些 token 被正式排进 `scheduler_output` 之前：

```python
# vllm/v1/core/sched/scheduler.py:2279-2283
if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

`validate_tokens()`（各后端实现，例如 `vllm/v1/structured_output/backend_xgrammar.py:181-201`）只读检查——**不推进** FSM，返回被接受的最长前缀。一旦某个草稿 token 不满足语法，后面的位置全部被砍掉，`spec_token_ids` 变短。`update_draft_token_ids_in_output()`（`vllm/v1/core/sched/scheduler.py:2284-2320`）在拿到真实模型输出后再做一次同样的过滤，把被砍掉的位置补 `-1` 占位（保持张量形状不变），`grammar_bitmask()` 遇到 `-1` 直接把 `apply_bitmask` 置 `False`（`vllm/v1/structured_output/__init__.py:304-306`）——**不给已经确定不会被采用的位置算约束**，省掉无意义的 bitmask 填充工作。

### 4.5 失败模式：报错路径与 token 化边界

结构化输出会在三个不同时刻失败，报错方式各不相同：

1. **显式指定后端 + schema 不被该后端支持**：在 `## 4.1` 描述的探测性编译阶段直接失败，`raise VLLMValidationError(...)`（例如 `vllm/v1/structured_output/backend_xgrammar.py:325-327` 正则失败、`:351-353` 不支持的 JSON 特性、`:358-360` schema 编译失败），请求还没进引擎就被 API 层拒绝——这是**同步报错**，客户端立刻拿到一个 400 响应。
2. **`backend="auto"` 且所有候选后端都编译不了**：回退链（`## 4.1`）最终仍然抛出 `VLLMValidationError`（最后一次尝试的异常没有再被吞掉），效果和第 1 种一样是同步 400，只是多试了两次。
3. **探测性编译通过，但引擎侧的运行时编译真的失败**：这种情况理论上应该很少见（同一段 schema、同一个后端，探测性编译能过运行时编译却过不了），但代码没有假设它不会发生——`_create_grammar()`（`vllm/v1/structured_output/__init__.py:178-200`）用 `try/except` 包住实际编译调用，异常被 `logger.exception(...)` 记录后重新抛出，向上层传播成 `Future` 的异常态或直接赋给 `_grammar`（`## 4.1` 的 `grammar_init()` 代码块）。调度器在 `_try_promote_blocked_waiting_request()`（`vllm/v1/core/sched/scheduler.py:2817-2823`）里探测到 `structured_output_req.grammar` 是 `Exception` 实例时，把请求 ID 记进 `self.grammar_compile_error_reqs`（`:2822`）而不是放行；`update_from_output()` 的收尾阶段（`vllm/v1/core/sched/scheduler.py:2068-2076`）把这个集合里的请求统一 `finish_requests(..., RequestStatus.FINISHED_ERROR)`，只让**这一个请求**带着错误结束，不影响同批次其他请求，更不会让 EngineCore 崩溃——这是**异步报错**，但仍然是"每请求隔离失败"而不是"静默放行不合法输出"或者"整个引擎级联失败"。

三种失败路径共同点：**vLLM 不存在"schema 编译不了就静默退化成无约束生成"这条路径**——找不到任何代码在编译失败时把请求当成普通生成请求继续跑；失败要么在提交时挡下来，要么让这一个请求带着明确的错误原因结束。

**token 化边界问题**（一个 JSON 字符——比如某个多字节 UTF-8 字符——横跨两个 token 的边界）不是靠"检测边界、特殊处理"解决的，而是从根上避免了这个问题的存在：`XgrammarBackend.__post_init__()` 构造 `TokenizerInfo` 时（`vllm/v1/structured_output/backend_xgrammar.py:51-65`），非 Mistral tokenizer 走 `xgr.TokenizerInfo.from_huggingface(...)`（`:61-65`），Mistral tokenizer 则手动指定 `vocab_type=xgr.VocabType.BYTE_FALLBACK`（非 tekken 分词器）或 `RAW`（tekken 分词器，`:54-56`）。`BYTE_FALLBACK` 词表类型告诉 xgrammar："每个 token 最终要在**字节**层面被拼接、匹配"，语法自动机本身是在字节流上定义的（JSON Schema 的字符串规则被编译成 UTF-8 字节序列上的自动机），而不是在"每个 token 必须恰好对应零个或多个完整字符"这个假设上定义的——一个多字节字符被切成两个 token 时，自动机在第一个 token 结束时停在一个"部分字节序列已消费、尚未凑齐一个完整 UTF-8 字符"的中间状态，第二个 token 到来后继续在字节层面推进，最终仍然能正确判断这个字符是否合法。这个结论是**源码为证**（vLLM 侧确实按字节级词表类型构造 `TokenizerInfo`）+ **本库推断**（"为什么这样能解决边界问题"是对 xgrammar 自动机语义的合理推断，xgrammar 库本身的实现细节不在 `_src/` 范围内，未查证其内部具体如何处理不完整字节序列的中间状态）。outlines 后端走的是相似的字节级路径：`_reduced_vocabulary()`（`vllm/v1/structured_output/utils.py:321-387`）把每个 token 映射成 `bytes` 而不是 `str`（`:348-383` 的循环体逐 token 转 `token_bytes`），同样是在字节而不是字符层面建立词表到自动机的映射。

## 5. 设计决策与代价

### 5.1 后端进程级单例，不支持按请求切换

**为什么这么设计**：编译好的 `Grammar` 对象要绑定一个具体的 `TokenizerInfo`（词表 + 特殊 token 元信息），构造这个绑定本身有初始化开销（`XgrammarBackend.__post_init__()`，`vllm/v1/structured_output/backend_xgrammar.py:38-71`，要么 `TokenizerInfo.from_huggingface(...)` 要么手动构造）；如果每个请求可以选不同后端，这份初始化要为每种后端各做一遍，且四个后端库（xgrammar/llguidance/outlines_core/lmformatenforcer）互相之间没有共享的中间表示，混用会让 `StructuredOutputManager` 的状态管理复杂度直接翻倍。

**不这样会怎样**：如果放开按请求选后端，`grammar_init()` 里 `if self.backend is None` 的惰性初始化就要改成按后端类型查字典、缺失才建，`_fill_bitmasks`/`grammar_bitmask` 里所有假设"这一批请求的 bitmask 用同一种打包格式"的代码都要重新审视——`apply_grammar_bitmask()` 统一走 `xgr.apply_token_bitmask_inplace`（`## 4.3`）这件事本身就隐含了"bitmask 打包格式跨后端一致"的假设，这个假设目前成立是因为所有后端的 `allocate_token_bitmask()` 都产出同样的 `(rows, cdiv(vocab,32))` int32 打包格式，但不代表可以安全假设未来所有后端都这样打包。

**什么时候可以不这样**：如果所有请求都用同一个后端（这是当前唯一支持的用法），这个限制不构成实际代价。上游注释"We do NOT support different backends on a per-request basis in V1 (for now, anyway...)"（`vllm/v1/structured_output/__init__.py:127-128`）里的"for now"暗示这是一个已知的范围收窄而非架构死角。

### 5.2 探测性编译 + 运行时编译，两次编译不复用

**为什么这么设计**：探测性编译（`SamplingParams.verify()` 阶段）发生在**前端进程**（或 `AsyncLLM` 的输入处理线程），此时甚至不确定这个请求最终会不会被 EngineCore 接受（可能因为其他校验失败被拒绝）；运行时编译发生在 **EngineCore 进程**，两者本来就不在同一个地址空间，编译出的对象（`xgr.CompiledGrammar` 等）绑定了 C++/Rust 扩展的句柄，天然不能跨进程直接传递——唯一能传的是"编译是否成功"这个布尔结论和错误信息，对象本身传不过去。

**不这样会怎样**：如果只做一次编译（去掉前端探测性校验），语法错误（比如 JSON Schema 用了 xgrammar 不支持的 `multipleOf`，见 `has_xgrammar_unsupported_json_features()`，`vllm/v1/structured_output/backend_xgrammar.py:239-297`）就要等到请求真正进了 EngineCore、`grammar_init()` 异步编译失败之后才能发现——用户体验从"提交请求立刻收到 400"退化成"请求先被引擎接受，过一会儿才收到一个异步错误"，对写 SDK/客户端代码的人更不友好。

**什么时候可以不这样**：如果引擎侧的 `_create_grammar()` 编译失败能做到足够快、足够可靠地转成同步错误返回给客户端（当前确实能做到，只是**慢一步**——因为默认异步），单纯为了"更快报错"而在前端重复编译一次就是可以省略的优化；具体到内部实现，编译结果本身两次都被丢弃重算（探测性编译产出的 `xgr.Grammar` / `xgr.CompiledGrammar` 对象不会被 `_create_grammar()` 复用），这是**本库推断**的一处可优化空间（`## 8` 展开），源码没有解释为什么不复用。

Outlines 后端在自己的编译结果上做了显式缓存（`OutlinesBackend._compile_index()`，`vllm/v1/structured_output/backend_outlines.py:59-72`）：`cache_key = f"{vocabulary._hash}_{regex_string}"`，命中就直接返回缓存的 `oc.Index`，默认用内存 `LRUCache(maxsize=128)`（`get_outlines_cache()`，`vllm/v1/structured_output/utils.py:283-302`），只有设置 `VLLM_V1_USE_OUTLINES_CACHE=1`（默认 `False`，`vllm/envs.py:195`）才会换成无上限的磁盘缓存（`OutlinesDiskCache`，`vllm/v1/structured_output/utils.py:219-280`，用 SQLite + outlines 自己的二进制序列化，规避 pickle 反序列化的任意代码执行风险）。xgrammar 后端则完全委托给 `xgr.GrammarCompiler` 自带的 `cache_enabled=True`（`vllm/v1/structured_output/backend_xgrammar.py:66-70`），容量上限 `VLLM_XGRAMMAR_CACHE_MB`（默认 512 MB，`vllm/envs.py:222`）。**guidance 和 lm-format-enforcer 后端没有任何显式缓存**（`compile_grammar()` 每次都从零构造 `llguidance.LLMatcher`/`lmformatenforcer.TokenEnforcer`，`vllm/v1/structured_output/backend_guidance.py:109-135`、`vllm/v1/structured_output/backend_lm_format_enforcer.py:102-143`）——**同一个 JSON Schema 被两个不同请求各发一次，guidance 后端会编译两遍**，这是四个后端之间一处不对称的、源码可验证的行为差异，`## 7` 展开。

### 5.3 bitmask 用位打包格式而不是逐 token 布尔数组

**为什么这么设计**：词表通常在几万到二十万量级，逐 token 用一个 `bool`/`int8` 存的话，一个批次几百个请求乘上这个宽度，光是 H2D 拷贝的字节数就是位打包方案的 8~32 倍；`cdiv(vocab_size, 32)` 个 `int32`（每个承载 32 个 vocab 位）把这个数组压到最小，`xgr.apply_token_bitmask_inplace` 在 GPU 上按位解包、原地把被禁 token 的 logit 设为 `-inf`。

**不这样会怎样**：如果用未压缩的布尔掩码，`apply_grammar_bitmask()` 里那次 `non_blocking=True` 的 H2D 拷贝（`vllm/v1/structured_output/utils.py:145`）传输的数据量会成倍增长，在结构化输出请求占比高、batch size 大的场景下，这次拷贝本身可能从"可忽略的重叠开销"变成"值得关注的瓶颈"——具体倍数取决于选择的整数宽度，本库不编具体数字。

**什么时候可以不这样**：CPU 推理路径（`vllm/v1/structured_output/utils.py:152-176` 的 `logits.is_cpu` 分支）依然用同样的打包格式，只是走 xgrammar 的 CPU kernel 而不是 CUDA kernel——没有为 CPU 场景单独放宽格式，因为打包/解包本身的 CPU 开销相对 H2D 拷贝节省的部分并不构成需要权衡的瓶颈（本库推断）。

### 5.4 async scheduling 下延迟采样，等占位符被真实 token 填满

**为什么这么设计**：开启异步调度后，`AsyncScheduler._update_after_schedule()`（`vllm/v1/core/sched/async_scheduler.py:19-49`）会在**还不知道上一步真实采样结果**的情况下就为下一步排产（`request.num_output_placeholders` 记录还没到位的占位 token 数）。而语法 FSM 的推进依赖"上一个真实 token 是什么"——占位符期间无法计算 bitmask（不知道 FSM 现在在哪个状态），所以 `pending_structured_output_tokens` 标记为真时，`step_with_batch_queue()`（`vllm/v1/engine/core.py:638-752`）会跳过立即计算 bitmask，改成把 `scheduler_output` 存进 `deferred_scheduler_output`（`:687-690`），等上一步的真实输出（含草稿 token 的验证结果）到位后才补算 `get_grammar_bitmask()` 并调用 `sample_tokens()`（`:744-750`）。

**不这样会怎样**：如果不管占位符状态、直接对着还是占位符的"假设 token"算 bitmask，FSM 会推进到一个错误的状态——异步调度 + 投机解码叠加时尤其危险，因为草稿 token 可能被拒绝（`## 4.4`），如果 bitmask 是按"假设草稿全部被接受"的状态算出来的，实际验证结果不同则这一步的约束就是错的，轻则约束失效放行不合法 token，重则 FSM 状态彻底跑偏后续全错。

**什么时候可以不这样**：没开异步调度（`self.use_async_scheduling` 为假，普通 `step()` 路径）或没有结构化输出请求时，`pending_structured_output_tokens` 恒为 `False`（`vllm/v1/core/sched/async_scheduler.py:31-33` 的赋值只在 `AsyncScheduler` 子类里发生），延迟采样这条分支不会被触发，`step_with_batch_queue()` 退化成"每步都立即算 bitmask、立即采样"。

### 5.5 语法状态与 KV/前缀缓存完全解耦

**为什么这么设计**：语法 FSM 描述的是"输出 token 序列合不合法"，这是一个**逻辑序列位置**上的性质，和"这个 token 的 attention KV 是重新算的还是从缓存读的"（一个**物理存储**层面的性质）没有依赖关系——只要 `accept_tokens()` 被调用的次数和顺序与真实输出 token 序列一致，FSM 状态就是对的，不管背后的 KV 是缓存命中还是重新计算出来的。把两者放在同一个对象里管理（比如让语法状态成为 KV 块元数据的一部分）反而会人为制造耦合，让 KV 缓存驱逐/前缀共享的逻辑必须额外考虑"要不要连带处理语法状态"。

**不这样会怎样**：如果语法状态被错误地和 KV 缓存生命周期绑定（比如误以为"KV 块被释放就该重置语法状态"），会导致预填充命中前缀缓存、或请求被抢占后重新排队的场景下语法状态被意外清空——FSM 从初始状态重新开始，但 `Scheduler.update_from_output()` 不会重新调用 `accept_tokens()` 把已经生成的历史 token 重放一遍（`vllm/v1/core/sched/scheduler.py:1918-1943` 只在 `new_token_ids` 非空、即本步有新输出时才推进），于是 FSM 状态和真实已输出内容永久错位，后续每一个 bitmask 都算错。当前源码没有这个 bug，是因为 `_preempt_request()`（`vllm/v1/core/sched/scheduler.py:1347-1388`）确实没有触碰 `structured_output_request`——但这依赖的是"没人写这段代码"，而不是有显式的隔离机制强制这件事，`## 7` 会展开为什么这是本库认为值得警惕的隐式正确性依赖。

**什么时候可以不这样**：如果未来给结构化输出加"支持请求级的语法状态迁移/序列化"（例如 PD 分离场景下把请求从一个实例迁移到另一个实例，见 [[12-vLLM-PD分离与KV-Connector]] 涉及的场景，本篇未验证该场景下语法状态如何处理，标注**未查证**），那时候语法状态就必须显式序列化/反序列化，不能再依赖"Python 对象活得比 KV 缓存久"这个隐式假设。

## 6. 同位对照：SGLang 的约束解码

SGLang 的等价物是 `GrammarManager`（`sglang:python/sglang/srt/constrained/grammar_manager.py:26`），和 vLLM 的 `StructuredOutputManager` 在整体形状上高度相似：也是异步编译 + `concurrent.futures.Future` + 请求级语法对象。`process_req_with_grammar()`（`sglang:python/sglang/srt/constrained/grammar_manager.py:131-182`）用 `(type, spec_string)` 元组做 key 显式查缓存：

```python
# sglang:python/sglang/srt/constrained/grammar_manager.py:153-160（节选）
value, cache_hit = self.grammar_backend.get_cached_or_future_value(
    key, req.require_reasoning
)
req.grammar = value
if not cache_hit:
    req.grammar_key = key
    add_to_grammar_queue = True
```

这一点和 vLLM 不同：vLLM 的 `StructuredOutputManager._create_grammar()`（`vllm/v1/structured_output/__init__.py:178-200`）**没有**自己的请求间去重缓存，是否复用完全取决于各后端库自己的内部缓存（xgrammar 有、outlines 有、guidance/lm-format-enforcer 没有，`## 5.2` 已展开）；SGLang 把这层缓存做在了 `GrammarManager` 自己的层级，对所有后端一视同仁——两个请求用同一个 JSON Schema，无论后端是什么，第二个请求都直接拿到第一个的编译结果或其 `Future`。就绪检查上，SGLang 用固定间隔的忙轮询（`get_ready_grammar_requests()`，`sglang:python/sglang/srt/constrained/grammar_manager.py:184-220`，`while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL: ... if req.grammar.done(): ...`），vLLM 用单次超时探测（`future.result(timeout=0.0001)`，`vllm/v1/structured_output/request.py:54`）——前者在调度循环里主动花一段时间等，后者一次只探一下、探不到就先跳过这个请求继续处理别的。

**jump-forward 解码**是这次对照里最值得记录的一条发现，而且和最初的预期不一致：**SGLang 的语法后端确实实现了 jump-forward 的接口**——`BaseGrammarObject.try_jump_forward()`/`jump_forward_str_state()`/`jump_and_retokenize()`（`sglang:python/sglang/srt/constrained/base_grammar_backend.py:120-146`）是抽象方法，`XGrammarGrammar` 有具体实现：

```python
# sglang:python/sglang/srt/constrained/xgrammar_backend.py:164-172
def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
    s = self.matcher.find_jump_forward_string()
    if s:
        return [], s
    return None
```

`llguidance_backend.py`、`outlines_backend.py`、`outlines_jump_forward.py`、`reasoner_grammar_backend.py` 都各自有 jump-forward 相关代码。**但本库在 `python/sglang/srt/managers/` 下逐文件搜索 `try_jump_forward`/`jump_and_retokenize` 的调用点，一处都没找到**——唯一的调用点是 `test/manual/lang_frontend/test_jump_forward.py`（`lang_frontend` 目录名暗示这是给旧版 SGLang 前端 DSL 用的手动测试，不是当前主服务路径）。也就是说：**在本篇取证的这个快照里，SGLang 的主 scheduler 请求处理循环并没有实际调用 jump-forward 逻辑**——这条能力在后端接口层面还在，但看起来已经和当前的主干调度器脱钩（源码为证：未找到调用点；这条"脱钩"的判断本身是**本库推断**，因为不存在调用点不代表未来不会重新接上，也可能是本库检索遗漏，标注**未查证**其历史原因）。

vLLM 这边则更直接：两个后端库都原生支持这个能力（xgrammar 的 `find_jump_forward_string()`，llguidance 的 `compute_ff_bytes()`/`compute_ff_tokens()`），但 vLLM 自己的代码里**从未调用**——`vllm/v1/structured_output/backend_xgrammar.py:146-147` 只是一条指向 xgrammar 文档的注释，`vllm/v1/structured_output/backend_guidance.py:177-182` 是一条明确的 `TODO`。为什么两边都没有真正跑起来？**本库推断**（源码没有直接给出原因）：跳跃式解码要求"跳过的那几个 token 必须是当前 batch 里唯一确定的下一批输出"，这与 persistent batch（[[06-vLLM-模型执行与CUDA-Graph]] `## 5.1`）、CUDA Graph 固定 batch 形状（同篇 `## 5.3`）、以及"每步为整个 batch 统一决定 padding 后的 token 数"这套批处理基础设施天然冲突——跳跃意味着这个请求这一步"多算了几个 token 但不需要重新过一遍完整 forward"，而当前的批处理执行模型是每步所有请求过一次统一形状的 forward，没有"这个请求这步跳 3 个 token、那个请求正常推进 1 个"的位置留给这种异构处理。

## 7. 踩坑与反直觉

**陷阱一：`backend="auto"` 时，后端选择是"整个引擎生命周期里第一个成功的请求说了算"，不是"每个请求各自决定"。** `## 5.1` 已经指出后端是进程级单例；这里更具体的坑是：如果引擎启动后收到的第一个结构化输出请求恰好用了一个 xgrammar 编译不了的 schema（比如带 `multipleOf` 的 JSON Schema），`_validate_structured_outputs()` 的 `auto` 回退链（`vllm/sampling_params.py:1150-1193`）会把这第一个请求的 `_backend` 设成 `"guidance"`，`grammar_init()` 第一次调用时就把 `StructuredOutputManager.backend` 建成了 `GuidanceBackend` 实例。**之后所有请求，哪怕它们的 schema 本来能被 xgrammar 完美编译，也会被 `_create_grammar()` 里 `self.backend.compile_grammar(...)` 用 guidance 编译**——`_backend` 字段虽然在每个请求自己的 `SamplingParams` 上仍然正确地写着 `xgrammar`（因为每个请求各自跑一遍 `## 4.1` 描述的回退链），但真正被使用的 `StructuredOutputManager.backend` 早就定死了。这不是一个断言失败或者报错的场景——两个后端库对同一个 JSON Schema 生成的合法输出集合、空白符处理策略（`disable_any_whitespace`）等细节并不完全一致，用户会观察到"同一个 schema，不同时间发的请求，输出风格不完全一样"却查不到日志报错。

**陷阱二：`guidance`/`lm-format-enforcer` 后端没有请求间去重缓存，重复 schema 会被反复编译。** `## 5.2` 已经指出这一点，这里补一句更直接的后果：如果一个服务大量请求共享同一个（或近似的）系统级 JSON Schema（常见于"每个请求都要求同一种输出格式"的应用场景），选择 `backend="auto"` 大概率落到 xgrammar 或 guidance（取决于 schema 特性），如果落到 guidance，**每一个请求都要重新走一遍 `llguidance.LLMatcher(...)` 构造**（`vllm/v1/structured_output/backend_guidance.py:122-126`），这部分开销在高 QPS 场景下会被反复摊销而不是被摊销一次——这不是 bug，是当前代码结构下的一个已知不对称,`## 8` 会给出可改进的方向。

**陷阱三（反直觉但验证下来是"安全"的）：抢占 + 前缀缓存命中不会让语法状态错位，但这依赖的是隐式假设而不是显式契约。** `## 5.5` 已经详细展开机制本身，这里强调的是"为什么容易让人误判有 bug"：直觉上，KV 前缀缓存能让请求"跳过一大段重新计算"，很容易联想到"那语法状态是不是也要跟着跳"——但语法约束从来不作用于 prompt 部分（`## 4.2` 提到 `is_prefill_chunk` 排除逻辑），且 FSM 状态活在 Python 对象里、由输出 token 序列驱动而非由 KV 计算驱动，所以两者根本不会打架。本库没有在测试或注释里找到任何显式声明"这两套状态是解耦的、可以放心不管"——这是读代码推出的结论,如果未来有人往 `_preempt_request()` 或 KV 缓存驱逐路径里加"清理请求相关状态"的通用逻辑,很容易在不知情的情况下把 `structured_output_request` 也清了,从而真正制造出一个当前并不存在的 bug。

**陷阱四：`lm-format-enforcer` 后端和投机解码硬冲突,是显式报错而不是静默失效。** `LMFormatEnforcerBackend.compile_grammar()` 里有一条硬编码检查：

```python
# vllm/v1/structured_output/backend_lm_format_enforcer.py:128-137
max_rollback_tokens = (
    self.vllm_config.speculative_config.num_speculative_tokens
    if self.vllm_config.speculative_config is not None else 0
)
if max_rollback_tokens > 0:
    raise ValueError(
        "LM Format Enforcer backend does not support speculative tokens"
    )
```

这是四个后端里唯一一个显式拒绝"结构化输出 + 投机解码"组合的——xgrammar/guidance/outlines 都在 `compile_grammar()` 里接受 `max_rollback_tokens`/`num_speculative_tokens` 参数并据此配置匹配器的回滚能力（分别是 `max_rollback_tokens=self.num_speculative_tokens`、`max_rollback=max_rollback_tokens`，`vllm/v1/structured_output/backend_xgrammar.py:127`、`vllm/v1/structured_output/backend_outlines.py:100`），只有 lm-format-enforcer 选择直接报错。这条报错在**编译时**触发（也就是 `grammar_init()` 异步编译失败,请求以 `FINISHED_ERROR` 结束,`## 8`/失败模式一节展开),不是在运行期悄悄跳过投机解码或悄悄跳过语法约束。

## 8. 可改进点

以下均为本库阅读代码后的推断，未在上游 issue/PR 中核实是否已有讨论（诚实标准第 6 条：不编 issue 号）。

1. **前端探测性编译（`SamplingParams.verify()`）和引擎侧运行时编译（`grammar_init()`）完全独立，编译结果不复用。** `## 5.2` 已指出这一点。如果两次编译发生在同一进程内（比如 in-process 模式、非多进程 executor 的部署形态），理论上可以把探测性编译的产物缓存下来，`grammar_init()` 优先查这个缓存——目前两次编译无论是否同进程都各自从头来过，对编译本身开销较大的复杂 schema（尤其是大型嵌套 JSON Schema）是双倍浪费。
2. **`guidance`/`lm-format-enforcer` 后端补一层和 outlines 对齐的请求间缓存。** `outlines_core.Index` 有基于 `(vocabulary_hash, regex_string)` 的显式缓存（`## 5.2`），xgrammar 委托给自己库的 `GrammarCompiler` 缓存，这两个后端目前每个请求都从零构建匹配器对象——给 `GuidanceBackend`/`LMFormatEnforcerBackend` 各加一层基于 `StructuredOutputKey` 的 `LRUCache`（缓存的是可复用的中间表示，比如 `serialized_grammar` 字符串或已解析的 `character_level_parser`，而不是绑定了每请求状态的匹配器实例本身),能补上 `## 7` 陷阱二指出的不对称。
3. **`should_fill_bitmask`/`should_advance` 里 reasoning 相关的分支逻辑高度依赖多个布尔标记的时序一致性。** `StructuredOutputManager.should_fill_bitmask()`（`vllm/v1/structured_output/__init__.py:370-392`）和 `should_advance()`（`:394-452`）都要在"是否在推理内容里"（`reasoning_ended`）、"这一步是否检测到推理结束标记"、"是否开启 `enable_in_reasoning`"之间做判断，`grammar_bitmask()` 内部（`:298-359`）还有一层 `post_reasoning_end_in_window` 状态跟踪跨窗口的推理结束时刻——这段逻辑目前只靠代码注释（例如 `:210-212` 提到的"for thinking support, we will need to reset..."）维护正确性,没有看到针对"推理恰好在投机解码窗口中间结束"这类边界场景的专门测试文件（本库在 `tests/v1/structured_output/` 下没有找到对应用例，**未查证**是否在别处有覆盖）。给这个状态机补显式的状态图注释或者专门的边界测试,能降低未来改动这段代码引入 regression 的风险。

## 9. 自测题与延伸阅读

**自测题**（闭卷回答，答案均可在 `## 2`~`## 7` 用到的行号里核实）：

1. `SamplingParams.verify()` 阶段的探测性编译和 `StructuredOutputManager.grammar_init()` 的运行时编译分别发生在哪个进程/线程？为什么两次编译的产物不能互相复用？
2. `backend="auto"` 时，回退链的优先级是什么？如果引擎收到的第一个结构化输出请求触发了回退，后续能被 xgrammar 正常编译的请求会用哪个后端处理？为什么？
3. `EngineCore.step()` 里，`get_grammar_bitmask()` 为什么要写在 `future.result()` 之前而不是之后？如果调换这两行的顺序，"CPU 与 GPU 前向重叠"这个效果还成立吗？
4. 投机解码的草稿 token 在什么时候第一次被拿去和语法做比对？`validate_tokens()` 和 `accept_tokens()` 的关键区别是什么，为什么草稿 token 预审阶段用前者而不是后者？
5. 请求被抢占（`_preempt_request()`）之后，`num_computed_tokens` 被清零、KV 块被释放，`structured_output_request.grammar` 会发生什么？为什么这不会导致语法状态和已输出内容错位？
6. `lm-format-enforcer` 后端在检测到投机解码开启时的行为，和 xgrammar/guidance/outlines 在同样场景下的行为有什么本质不同？
7. SGLang 的 `try_jump_forward()` 在哪些后端里有具体实现？本库为什么判断这些实现在当前的主 scheduler 循环里没有被实际调用？这个判断的证据边界（能证明什么、不能证明什么）是什么？
8. vLLM 的 bitmask 打包格式是什么？为什么统一走 `xgr.apply_token_bitmask_inplace` 应用，即使当前请求用的是 guidance 或 outlines 后端编译的语法？

**延伸阅读**：

- [[03-vLLM-调度器解剖]]——`SchedulerOutput`/`schedule()` 的完整流程，本篇的 `get_grammar_bitmask()` 挂在这条主线之后
- [[06-vLLM-模型执行与CUDA-Graph]]——`execute_model()`/`sample_tokens()` 两次 RPC 拆分的完整解剖，本篇 `## 4.2` 复用了这条主线
- [[10-vLLM-投机解码]]——草稿 token 生成与验证的完整流程，本篇 `## 4.4` 只讲了其中和语法相关的切片
- [[06-SGLang-约束解码与语法后端]]——SGLang 侧 `GrammarManager`/jump-forward 的完整解剖（若已成文）
- [[00-总览与阅读地图]]
