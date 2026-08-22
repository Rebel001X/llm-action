# vLLM 模型执行与 CUDA Graph

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：持久化 batch 省重建开销，CUDA Graph 靠"形状归一"换零 CPU 开销。

## 0. 结论先行

- `GpuModelRunner`（`vllm/v1/worker/gpu_model_runner.py`）是 vLLM V1 执行栈里离 GPU 最近的一层：调度器只决定"这一步谁跑、跑几个 token"，真正把这个决定翻译成张量、丢进模型、采出 token 的活全在这一个类里，长达 7752 行、117 个方法，是全仓库单文件行数最大的文件（`_lab/out/struct_map.json` 的 `subsystems.model_exec.top_files` 第一条）。
- **"persistent batch" 不是缓存优化，是避免每步重新分配 CPU/GPU 张量**——它省的主要是 CPU 端的分配与整理开销，其次才是借 `pin_memory` 让 H2D 拷贝能非阻塞地和其他工作重叠；代价是一套增量增删的簿记逻辑（`condense`/`swap_states`），状态错位是这类代码最容易踩的坑。
- **CUDA Graph 不是"给模型录一段视频"，是给一小撮*固定形状*的批次各录一段**——capture 只发生在预先选定的若干 `num_tokens`（叫 capture sizes），运行时的真实 batch 会被**填充（pad）**到最近的一个已捕获尺寸上，多算的部分白算但不引发形状不匹配。
- **decode 能上 FULL graph，prefill 通常只能上 PIECEWISE 甚至完全不上**——根子在 attention backend 的 `AttentionCGSupport` 声明：大多数后端只承诺"query 长度一致的批次"（`UNIFORM_BATCH` / `UNIFORM_SINGLE_TOKEN_DECODE`）才能整图捕获，decode 步天然满足（每个请求恰好推进 1 个 token，或 spec decode 场景下恰好 `1+num_speculative_tokens` 个），prefill/mixed 批次逐请求 token 数不同，天然不满足。
- **torch.compile 不在 `load_model()` 里发生，而是被 `@support_torch_compile` 装饰器懒触发在"第一次真正调用模型 forward"的那一刻**——这一刻是 `profile_run()`（GPU 显存探测阶段），比 `capture_model()`（CUDA Graph 捕获）早一步，两者常被误认为同一件事。
- `execute_model()` 和 `sample_tokens()` 是两次独立可调用的 RPC，不是一次调用内部顺序完成"forward → sample"——这个拆分是本篇要反复回头讲的一条主线。

证据分级：本篇除标注"文档所述"或"本库推断"外，均为「源码为证」，带 `文件:行`。不产出任何实测数字（本机无 GPU），CUDA Graph/编译带来的收益只讲机制不编数字。

## 1. 它在系统里的位置

调用链自上而下：EngineCore 的忙循环每一步产出一个 `SchedulerOutput`（详见 [[03-vLLM-调度器解剖]]），交给 `Worker`。`Worker.execute_model()` 和 `Worker.sample_tokens()` 本身只是一层带了 `@torch.inference_mode()`、`@with_gpu_sync_check` 装饰器的转发壳，原样把调用转给 `model_runner`：

```python
# vllm/v1/worker/gpu_worker.py:1055-1066（节选）
@torch.inference_mode()
@with_gpu_sync_check
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
    return self.model_runner.sample_tokens(grammar_output)

@torch.inference_mode()
@with_gpu_sync_check
def execute_model(
    self, scheduler_output: "SchedulerOutput"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    ...
```

也就是说 `GpuModelRunner` 是 Worker 进程里唯一真正碰 GPU 张量、碰模型权重、碰 CUDA Graph 的对象。Ray 执行器走的是另一条路径但落点相同：`vllm/v1/executor/ray_utils.py:145` 与 `:171` 分别调用 `self.worker.model_runner.execute_model(...)` 与 `self.worker.model_runner.sample_tokens(...)`，说明无论走 multiproc 还是 Ray executor，最终都收敛到同一个 `GpuModelRunner` 实例上的这两个方法。

`execute_model()` 和 `sample_tokens()` 在 vLLM V1 里被拆成了两个独立可调用的 RPC（而不是一个 `execute_model()` 内部顺序完成"forward → sample"），这个拆分是本篇要反复回头讲的一条主线，见 `## 4`。它的直接动机是给结构化输出的语法状态机计算（见 [[11-vLLM-结构化输出]]）和异步调度让出窗口——`sample_tokens()` 的入参就是 `grammar_output: "GrammarOutput | None"`（`vllm/v1/worker/gpu_model_runner.py:4663-4664`），这意味着 EngineCore 在 `execute_model()` 之后、`sample_tokens()` 之前，可以并行算完当前 batch 每个请求的语法约束比特掩码。

`GpuModelRunner` 的上游依赖是 KV 缓存分配结果（见 [[04-vLLM-KV缓存与前缀缓存]]）与 attention backend 选型（见 [[05-vLLM-注意力后端与算子层]]）——它自己不决定用哪种 attention kernel，只是持有 `self.attn_groups` 并在每步查询它们的 CUDA Graph 支持能力（`## 5` 会展开）。投机解码的草稿模型前向也挂在同一个类里（`self.drafter`，`## 4.5` 提及），详见 [[10-vLLM-投机解码]]。

## 2. 代码地图（文件 → 职责，带行号）

| 文件:行 | 职责 |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py:485` | `class ExecuteModelState(NamedTuple)`——横跨两次 RPC 调用的暂存态 |
| `vllm/v1/worker/gpu_model_runner.py:501` | `class GPUModelRunner` 定义起点，`_lab/out/struct_map.json` 记录该类 117 个方法、跨度 7252 行 |
| `vllm/v1/worker/gpu_model_runner.py:1097` | `_make_buffer()`——创建 `CpuGpuBuffer` 的统一入口 |
| `vllm/v1/worker/gpu_model_runner.py:1242` | `_update_states()`——persistent batch 的增量更新入口，本步开始前先把上一步的状态"打齐" |
| `vllm/v1/worker/gpu_model_runner.py:2015` | `_prepare_inputs()`——把 `SchedulerOutput` 翻译成本步要喂进模型的 GPU 张量 |
| `vllm/v1/worker/gpu_model_runner.py:3608` | `_preprocess()`——组装 `input_ids`/`positions`/`inputs_embeds` |
| `vllm/v1/worker/gpu_model_runner.py:3755` | `_sample()`——调用 `Sampler` 的入口 |
| `vllm/v1/worker/gpu_model_runner.py:3952` | `_model_forward()`——真正调模型 `forward()` 的一层薄封装 |
| `vllm/v1/worker/gpu_model_runner.py:3985` | `_is_uniform_decode()`——判定本步是否为"整齐的" decode 批次 |
| `vllm/v1/worker/gpu_model_runner.py:4050` | `_determine_batch_execution_and_padding()`——本步该用 FULL/PIECEWISE/NONE 里的哪种、padding 到多少 token |
| `vllm/v1/worker/gpu_model_runner.py:4284` | `execute_model()`——一步的主入口，只做到 forward+算 logits 为止，不采样 |
| `vllm/v1/worker/gpu_model_runner.py:4663` | `sample_tokens()`——真正采样、更新状态、（可选）跑投机解码草稿模型 |
| `vllm/v1/worker/gpu_model_runner.py:5409` | `load_model()`——权重加载入口 |
| `vllm/v1/worker/gpu_model_runner.py:6549` | `profile_run()`——显存探测用的一次 dummy forward，**这是 torch.compile 实际触发编译的地方** |
| `vllm/v1/worker/gpu_model_runner.py:5931` | `_dummy_run()`——不依赖真实请求、纯为 warmup/profile/capture 服务的假前向 |
| `vllm/v1/worker/gpu_model_runner.py:6937` | `capture_model()`——CUDA Graph 捕获入口，按 batch size 从大到小依次 capture |
| `vllm/v1/worker/gpu_input_batch.py:35` | `class CachedRequestState`——每个请求在 Python 侧的非张量元信息快照 |
| `vllm/v1/worker/gpu_input_batch.py:92` | `class InputBatch`——persistent batch 的核心容器，一堆按 slot 索引的 CPU/GPU 双份张量 |
| `vllm/v1/worker/gpu_input_batch.py:708` | `condense()`——请求被移除后把空洞往前压缩，这是状态错位风险最集中的地方 |
| `vllm/v1/worker/gpu_worker.py:456` | `Worker.load_model()`——转发给 `model_runner.load_model()` |
| `vllm/v1/worker/gpu_worker.py:481` | `determine_available_memory()`——调用 `profile_run()` 探测显存，间接触发编译 |
| `vllm/v1/worker/gpu_worker.py:705` | `compile_or_warm_up_model()`——启动阶段的额外 warmup + 调 `capture_model()` |
| `vllm/config/compilation.py:37` | `class CompilationMode`——NONE/STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE/VLLM_COMPILE 四档 |
| `vllm/config/compilation.py:53` | `class CUDAGraphMode`——NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE |
| `vllm/config/compilation.py:463` | `cache_dir` 字段——编译产物缓存目录，默认按内容哈希自动生成 |
| `vllm/forward_context.py:30` | `class BatchDescriptor`——本步的"形状签名"，`CudagraphDispatcher` 用它做查表 key |
| `vllm/v1/cudagraph_dispatcher.py:15` | `class CudagraphDispatcher`——运行时按 batch 形状查表决定实际用哪种 Graph 模式 |
| `vllm/v1/cudagraph_dispatcher.py:235` | `dispatch()`——给定形状返回 `(CUDAGraphMode, BatchDescriptor)` |
| `vllm/compilation/cuda_graph.py:233` | `CUDAGraphWrapper.__call__()`——真正做 capture/replay/eager-passthrough 三选一判断的地方 |
| `vllm/compilation/decorators.py:502` | `@support_torch_compile` 生成类的 `__call__()`——torch.compile 懒触发的物理位置 |
| `vllm/compilation/backends.py:1059` | 编译缓存目录哈希的拼装逻辑（无 `cache_dir` 时自动生成） |
| `vllm/v1/sample/sampler.py:21` | `class Sampler(nn.Module)`——采样器本体，是个 `nn.Module`，全程 GPU 张量运算 |
| `vllm/v1/sample/logits_processor/state.py:148` | `class LogitsProcessors`——按 argmax-invariant 拆成两条链 |
| `vllm/model_executor/model_loader/base_loader.py:43` | `BaseModelLoader.load_model()`——加载三段式：建骨架 → 灌权重 → 后处理 |
| `vllm/model_executor/model_loader/default_loader.py:415` | `DefaultModelLoader.load_weights()`——safetensors 流式读盘、按模型自定义映射灌进模块 |
| `vllm/v1/attention/backend.py:567` | `class AttentionCGSupport`——attention backend 对 CUDA Graph 形状约束的自我声明 |
| `vllm/v1/utils.py:110` | `class CpuGpuBuffer`——pinned CPU + GPU 双份张量的统一封装 |

以上 30+ 条均可用 `## 9` 自测题反查；下文引用的行号均可在 `_src/vllm/` 对应文件里核实。

## 3. 核心数据结构

**`CpuGpuBuffer`**（`vllm/v1/utils.py:110-149`）是 persistent batch 的地基：

```python
# vllm/v1/utils.py:110-142（节选）
class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(self, *size, dtype, device, pin_memory=PIN_MEMORY, with_numpy=True):
        with torch.inference_mode(False):
            self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
            self.gpu = torch.zeros_like(self.cpu, device=device)
        ...

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
```

一次性分配一对形状相同的张量，CPU 侧 `pin_memory=True`，GPU 侧普通显存；`copy_to_gpu(n)` 用 `non_blocking=True` 做 H2D 拷贝。`GpuModelRunner` 里几乎所有喂给模型的输入（`input_ids`、`positions`……）都是这种双份缓冲区，`_make_buffer()`（`vllm/v1/worker/gpu_model_runner.py:1097`）是创建它们的统一入口。

**`InputBatch`**（`vllm/v1/worker/gpu_input_batch.py:92`）是这些缓冲区的容器和索引层，本质是"槽位数组"：`max_num_reqs` 个槽位，每个请求占一个槽，槽内是该请求的 token id、采样参数、block table 等。它不是每步重建，而是维护 `req_id_to_index` 映射，跨步复用同一批底层张量（`## 4`/`## 5` 展开为什么）。

**`CachedRequestState`**（`vllm/v1/worker/gpu_input_batch.py:35-58`）是每个请求在 Python 侧（非张量）的元信息快照：

```python
# vllm/v1/worker/gpu_input_batch.py:35-58（节选字段）
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    mrope_positions: torch.Tensor | None = None
    lora_request: LoRARequest | None = None
    prompt_embeds: torch.Tensor | None = None
    prev_num_draft_len: int = 0
    pooling_params: PoolingParams | None = None
```

`GpuModelRunner.requests: dict[str, CachedRequestState]` 持有全部在飞请求，是 `InputBatch` 槽位内容的"真相来源"。

**`BatchDescriptor`**（`vllm/forward_context.py:30-51`）是"这一步的形状签名"：

```python
# vllm/forward_context.py:29-51（节选）
@dataclass(frozen=True)
class BatchDescriptor:
    """Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph."""

    num_tokens: int
    num_reqs: int | None = None
    """Can be None for PIECEWISE cudagraphs where the cudagraphs can handle
    any number of requests."""
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
```

padding 后的 token 数、是否 uniform decode、LoRA 状态都在这个 key 里，`CudagraphDispatcher.dispatch()` 用它去查有没有对应的已捕获 Graph（`## 5.3`）。

**`ExecuteModelState`**（`vllm/v1/worker/gpu_model_runner.py:485-498`）是横跨 `execute_model()` 与 `sample_tokens()` 两次调用的"暂存态"：

```python
# vllm/v1/worker/gpu_model_runner.py:485-498
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
```

`sample_tokens()` 开头 unpack 之后立刻清空（`vllm/v1/worker/gpu_model_runner.py:4676-4690`）。如果 `execute_model()` 还没返回 `None`（即上一次调用没走完）就再调一次，会直接抛 `RuntimeError`（`vllm/v1/worker/gpu_model_runner.py:4289-4293`）——这是这套双阶段调用协议唯一的运行时保险丝。

**`CUDAGraphMode`**（`vllm/config/compilation.py:53-104`）用 `enum.Enum` 的技巧让组合模式的取值直接是元组：

```python
# vllm/config/compilation.py:53-69（节选）
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self
```

`decode_mode()`/`mixed_mode()` 靠 `self.value[0]`/`[1]` 取出各自子模式（`vllm/config/compilation.py:65-69`），一个枚举同时管住"decode 步用什么、prefill/mixed 步用什么"两条决策。

## 4. 主流程走读

### 4.1 启动阶段：权重先落地，图后捕获

顺序是 `Worker.load_model()`（`vllm/v1/worker/gpu_worker.py:456-463`）→ `Worker.determine_available_memory()`（`vllm/v1/worker/gpu_worker.py:481`）→（KV 缓存按探测结果分配，属 [[04-vLLM-KV缓存与前缀缓存]] 范畴）→ `Worker.compile_or_warm_up_model()`（`vllm/v1/worker/gpu_worker.py:705`）。

- `load_model()` 只做"建骨架 + 灌权重"，不触发任何 CUDA kernel 编译（`## 4.6` 展开）。
- `determine_available_memory()` 内部调用 `self.model_runner.profile_run()`（`vllm/v1/worker/gpu_worker.py:521-525` 的 `memory_profiling` 上下文里），这是模型第一次被真正 `forward()` 调用——如果 `CompilationMode` 是 `VLLM_COMPILE`，torch.compile 在这一刻懒触发（`## 4.4`）。
- `compile_or_warm_up_model()`（`vllm/v1/worker/gpu_worker.py:705-735`）先为一批"补充尺寸"（不在 CUDA Graph capture 列表里但用户仍想编译，比如 `max_num_batched_tokens`）跑 `_dummy_run()` 触发/复用编译缓存，再调 `kernel_warmup()` 预热 kernel autotune，最后才在 `enforce_eager` 为假时调用 `self.model_runner.capture_model()`（`vllm/v1/worker/gpu_worker.py:742-743`）去真正捕获 CUDA Graph。**编译和捕获是两件先后发生的事，编译发生得更早。**

### 4.2 每步的主入口：`execute_model()` 只走到算完 logits

`execute_model()`（`vllm/v1/worker/gpu_model_runner.py:4284-4645`）的骨架（省略大量分支细节，完整版见源码）：

```python
# vllm/v1/worker/gpu_model_runner.py:4284 起（骨架节选，行号见正文）
def execute_model(self, scheduler_output, intermediate_tensors=None):
    if self.execute_model_state is not None:
        raise RuntimeError(...)                                   # L4289-4293
    deferred_state_corrections_fn = self._update_states(scheduler_output)  # L4323
    logits_indices, spec_decode_metadata, max_num_sampled_tokens = (
        self._prepare_inputs(scheduler_output, num_scheduled_tokens_np)
    )                                                              # L4365-4367
    cudagraph_mode, batch_desc, ... = self._determine_batch_execution_and_padding(...)  # L4385
    attn_metadata, ... = self._build_attention_metadata(...)       # L4495
    input_ids, inputs_embeds, positions, ... = self._preprocess(...)  # L4519
    with set_forward_context(attn_metadata, self.vllm_config,
                              cudagraph_runtime_mode=cudagraph_mode,
                              batch_descriptor=batch_desc, ...):    # L4542-4553
        model_output = self._model_forward(input_ids=input_ids, positions=positions, ...)  # L4560
    logits = self.model.compute_logits(hidden_states[logits_indices])  # L4594-4595
    self.execute_model_state = ExecuteModelState(scheduler_output, logits, ...)  # L4626
    return None                                                    # L4645
```

按顺序做：

1. `_update_states(scheduler_output)`（`vllm/v1/worker/gpu_model_runner.py:4323`）——见 `## 4.3`，本步先把 `InputBatch` 校正成和 `SchedulerOutput` 一致。
2. `_prepare_inputs(scheduler_output, num_scheduled_tokens_np)`（`vllm/v1/worker/gpu_model_runner.py:4365-4367`）——从 `InputBatch` 的 CPU 缓冲区里挑出本步要用的 token，写进另一组"本步专用"的 CPU→GPU 缓冲区。
3. `_determine_batch_execution_and_padding(...)`（`vllm/v1/worker/gpu_model_runner.py:4385-4395`）——决定 `cudagraph_mode`、`batch_desc`（含 padding 后的 token 数）——见 `## 5.3`。
4. 构建 attention metadata、slot mapping（`vllm/v1/worker/gpu_model_runner.py:4484-4510`），走 `_preprocess()`（定义于 `vllm/v1/worker/gpu_model_runner.py:3608`）拿到 `input_ids`/`positions`/`inputs_embeds`。
5. 用 `set_forward_context(...)`（`vllm/v1/worker/gpu_model_runner.py:4542-4553`）把 `cudagraph_runtime_mode`、`batch_descriptor` 挂进线程局部的 forward context，供 `CUDAGraphWrapper` 在真正 forward 时读取。
6. `self._model_forward(...)`（`vllm/v1/worker/gpu_model_runner.py:4560-4566`，定义在 `vllm/v1/worker/gpu_model_runner.py:3952`）——这才是模型真正跑一遍的地方，内部可能落进已捕获的 CUDA Graph 里 replay，也可能落进 eager/编译后的 Python 代码路径，取决于 `## 5` 的判定结果。
7. 取出 `logits_indices` 对应的 hidden state，`self.model.compute_logits(...)`（`vllm/v1/worker/gpu_model_runner.py:4594-4595`）算出 logits。
8. **不采样**，把 `scheduler_output`、`logits`、spec-decode 元数据等打包成 `ExecuteModelState`（`vllm/v1/worker/gpu_model_runner.py:4626-4637`）存起来，`return None`（`vllm/v1/worker/gpu_model_runner.py:4645`）。

### 4.3 `_update_states`：persistent batch 的增量更新

`_update_states()`（`vllm/v1/worker/gpu_model_runner.py:1242-1251` 的 docstring 直接写明"updates the cached states and the persistent batch"）按顺序：

```python
# vllm/v1/worker/gpu_model_runner.py:1252-1303（节选逻辑）
for req_id in scheduler_output.finished_req_ids:
    req_state = self.requests.pop(req_id, None)          # L1252-1256
    ...
for req_id in scheduler_output.finished_req_ids:
    self.input_batch.remove_request(req_id)               # L1266-1267

scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
cached_req_ids = self.input_batch.req_id_to_index.keys()
resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
# NOTE(woosuk): The persistent batch optimization assumes that
# consecutive batches contain mostly the same requests. If batches
# have low request overlap ... this optimization becomes very inefficient.
for req_id in unscheduled_req_ids:
    self.input_batch.remove_request(req_id)                # L1302-1303
```

1. 把 `scheduler_output.finished_req_ids` 里的请求从 `self.requests` 弹出，并从 `self.input_batch` 里 `remove_request()`（`vllm/v1/worker/gpu_model_runner.py:1252-1267`）。
2. 把本步**没被调度**但仍在追踪的请求（被抢占、或本步暂不轮到）也从 `InputBatch` 里挪走，但保留其 `CachedRequestState`——它们还会回来（`vllm/v1/worker/gpu_model_runner.py:1283-1303`）。这里有一条直白的注释：*"The persistent batch optimization assumes that consecutive batches contain mostly the same requests. If batches have low request overlap ... this optimization becomes very inefficient."*（`vllm/v1/worker/gpu_model_runner.py:1298-1301`）——这是本篇 `## 5.1` 要展开的代价来源。
3. 新请求按 `CachedRequestState` 构造、塞进 `self.requests`（`vllm/v1/worker/gpu_model_runner.py:1315-1381`）。
4. 已在跑的请求按 `scheduler_output.scheduled_cached_reqs` 更新 `num_computed_tokens`、block table 等（`vllm/v1/worker/gpu_model_runner.py:1382-1421` 起）。

### 4.4 torch.compile 何时真正编译

`@support_torch_compile` 装饰器把模型类的 `forward` 包了一层，生成的 `__call__()`（`vllm/compilation/decorators.py:502`）在**每次**被调用时先做两道短路检查：

```python
# vllm/compilation/decorators.py:502-513
def __call__(self: type[_T], *args, **kwargs) -> Any:
    if self.do_not_compile or torch.compiler.is_compiling():
        return self.forward(*args, **kwargs)

    if is_forward_context_available() and get_forward_context().skip_compiled:
        return self.forward(*args, **kwargs)
    ...
```

如果已经在编译过程中或该模型被标记 `do_not_compile`，直接走原始 `forward`（`vllm/compilation/decorators.py:506-507`）；如果 forward context 要求 `skip_compiled`（编码器-解码器模型第一遍带 encoder 输入的步——`## 7` 展开），同样直接走原始 `forward`（`vllm/compilation/decorators.py:512-513`）。真正首次落进 Dynamo/Inductor 编译路径，是第一次满足以上条件都不成立的调用——按 `## 4.1` 的顺序，这一刻是 `profile_run()` 内部第一次 `_dummy_run()`，不是 `load_model()`。

编译产物的磁盘缓存目录如果用户没手动指定 `cache_dir`（`vllm/config/compilation.py:463-466`），会按环境哈希、`VllmConfig.compute_hash()`、编译器哈希、被追踪源码文件内容的哈希拼出一个 10 位十六进制 key：

```python
# vllm/compilation/backends.py:1059-1079（节选）
if not self.compilation_config.cache_dir:
    factors = [env_hash, config_hash, code_hash, compiler_hash]
    hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
    cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "torch_compile_cache", hash_key)
    self.compilation_config.cache_dir = cache_dir
...
rank = vllm_config.parallel_config.rank
dp_rank = vllm_config.parallel_config.data_parallel_index
local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)
```

落在 `envs.VLLM_CACHE_ROOT/torch_compile_cache/<hash_key>/`（`VLLM_CACHE_ROOT` 默认 `~/.cache/vllm`，`vllm/envs.py:34`）；再往下按 `rank_<r>_<dp_rank>/<prefix>` 分子目录，多进程/多卡各写各的，不互相踩。**冷启动代价的机制**：`code_hash` 覆盖的是"被追踪到的源码文件内容"（`vllm/compilation/backends.py:1042-1053` 逐个读取 `forward_code_files` 拼哈希），任何一个哈希因子变了（环境变量、vLLM 配置、被追踪的模型源码文件），缓存目录名就变，等于重新编译——这解释了"改一行看似无关的代码/升级一个环境变量就要重新等编译"这种体感，具体等待时长本篇不给数字（未查证，且因硬件/模型而异）。

### 4.5 `sample_tokens`：真正采样、更新状态、驱动投机解码

`sample_tokens()`（`vllm/v1/worker/gpu_model_runner.py:4663-4720` 起）先 unpack `ExecuteModelState`（`vllm/v1/worker/gpu_model_runner.py:4677-4690`），若传入了 `grammar_output` 就调用 `apply_grammar_bitmask(...)` 把结构化输出的约束叠加到 logits 上（`vllm/v1/worker/gpu_model_runner.py:4693-4696`），然后 `self._sample(logits, spec_decode_metadata)`（`vllm/v1/worker/gpu_model_runner.py:4699`，定义于 `vllm/v1/worker/gpu_model_runner.py:3755`）调用 `Sampler`。采完之后 `_update_states_after_model_execute(...)`（`vllm/v1/worker/gpu_model_runner.py:4701-4703`）把新采出的 token 写回 `InputBatch`；如果开了投机解码，视 drafter 类型决定是否要在这一步顺带跑草稿模型的前向（`vllm/v1/worker/gpu_model_runner.py:4721-4800` 起的 `propose_draft_token_ids` 闭包与后续分支），详见 [[10-vLLM-投机解码]]。

采样本身在 GPU 上完成——`Sampler`（`vllm/v1/sample/sampler.py:21`）是个 `nn.Module`，`forward()` 输入输出都是 `torch.Tensor`，全程没有 `.item()`/`.cpu()` 之类的同步点（除非请求了 logprobs 之类需要具体数值的路径）。它的 docstring 把整条流水线的顺序写得很直白：

```text
# vllm/v1/sample/sampler.py:21-59（docstring 节选，按执行顺序）
1. 若请求了 logprobs：按 logprobs_mode 计算/拷贝原始 logprobs
2. logits 转 float32
3. 应用 allowed token ids 白名单
4. 应用 bad words 排除
5. 应用非 argmax-invariant 的 logits processor（min_tokens、logit_bias）
6. 应用惩罚项（repetition/frequency/presence penalty）
7. 采样：先判断 greedy，再温度、argmax-invariant 的 logits processor（如 min_p）、
   top_k/top_p，最后按温度是否 >= 1e-5 决定用随机采样还是退回贪心结果
8. 收集 top-k logprobs
9. 返回 SamplerOutput
```

logits processor 链在 `LogitsProcessors`（`vllm/v1/sample/logits_processor/state.py:148-160`）里被分成两条列表：

```python
# vllm/v1/sample/logits_processor/state.py:151-160
def __init__(self, logitsprocs=None):
    self.argmax_invariant: list[LogitsProcessor] = []
    self.non_argmax_invariant: list[LogitsProcessor] = []
    if logitsprocs:
        for logitproc in logitsprocs:
            (self.argmax_invariant if logitproc.is_argmax_invariant()
             else self.non_argmax_invariant).append(logitproc)
```

"argmax-invariant"（不改变贪心采样结果的处理器，比如 min_p）和"non-argmax-invariant"（会改变贪心结果的处理器，比如 min_tokens/logit_bias）被分别记账、在采样流程的不同阶段插入——这是为什么 `Sampler.forward()` 的 docstring 步骤 5 和步骤 7c 分别引用了两条不同的链。

### 4.6 权重怎么加载

`GpuModelRunner.load_model()`（`vllm/v1/worker/gpu_model_runner.py:5409`）调用 `get_model_loader(self.load_config)` 拿到一个 `BaseModelLoader` 子类实例（默认是 `DefaultModelLoader`），再调它的 `load_model()`——这是三段式流程（`vllm/model_executor/model_loader/base_loader.py:43-82`）：

```python
# vllm/model_executor/model_loader/base_loader.py:43-82（节选）
def load_model(self, vllm_config, model_config, prefix="") -> nn.Module:
    target_device = torch.device(load_device)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = initialize_model(vllm_config=vllm_config,
                                      model_config=model_config, prefix=prefix)  # 建骨架
        self.load_weights(model, model_config)                                  # 灌权重
        if _has_online_quant(model):
            finalize_layerwise_processing(model, model_config)
        process_weights_after_loading(model, model_config, target_device)       # 后处理
    return model.eval()
```

1. **建骨架**：`initialize_model()`（`vllm/model_executor/model_loader/utils.py:38`）在 `with target_device:` 上下文里直接 `model_class(vllm_config=vllm_config, prefix=prefix)` 构造模型——注意这一步是**直接在目标设备（通常是 GPU）上创建带随机初始值的参数**，不是先建在 meta device 再搬过去。
2. **灌权重**：`DefaultModelLoader.load_weights()`（`vllm/model_executor/model_loader/default_loader.py:415-433`）调用 `model.load_weights(self.get_all_weights(model_config, model))`（`vllm/model_executor/model_loader/default_loader.py:427`）——每个模型类自己实现 `load_weights()`，负责把 checkpoint 里的参数名映射到 vLLM 内部模块树的参数名，一般通过流式读取 safetensors 文件完成（`vllm/model_executor/model_loader/default_loader.py:25-35` 引入的几种 `*_weights_iterator`）。
3. **严格性检查**：非量化模型默认开启"哪些参数被加载过"的追踪（`vllm/model_executor/model_loader/default_loader.py:436-445`），如果模型声明的参数里有没被 checkpoint 覆盖到的，直接 `raise ValueError`（`vllm/model_executor/model_loader/default_loader.py:465-469`）——这是防止"某个权重悄悄留在随机初始值上却没人发现"的安全网。
4. **后处理**：`process_weights_after_loading(...)` 处理量化打包、权重转置等需要在权重就位后才能做的操作，量化配置见 [[05-vLLM-注意力后端与算子层]]。

## 5. 设计决策与代价

### 5.1 persistent batch：跨步复用槽位而非每步重建

**为什么这么设计**：请求集合在连续两步之间通常高度重叠——同一批请求继续 decode，只有极少数完成/新增。每步重新分配 `InputBatch` 里那一堆 `(max_num_reqs, max_model_len)` 形状的 CPU 张量本身就是一笔 CPU 分配/清零开销；更关键的是这些张量的 `pin_memory=True`，pinned 内存分配比 pageable 内存贵得多，每步重新申请会显著拖慢 CPU 端调度节奏。持久化之后只需要"增删差集"——`_update_states()` 只处理本步真正变化的请求（`## 4.3`）。

**不这样会怎样**：如果改成每步从零构建输入张量，省掉的是 `_update_states`/`condense`/`swap_states` 这套簿记逻辑的复杂度，换来的是每步都要重新做 pinned 内存分配 + 全量 CPU 端 token/位置/block table 计算——CPU 端开销会随 batch size 线性增长且没有"仅算增量"的空间，在高并发、小 decode 步长为主的场景下这部分 CPU 开销很容易成为整条流水线的瓶颈（本库推断，依据是 `pin_memory` 分配在多数 CUDA 运行时实现里比 pageable 分配慢一到两个数量级，属通用工程常识，非本仓库实测数字）。

**什么时候可以不这样**：请求集合跨步几乎不重叠的场景（源码注释直接点名，`vllm/v1/worker/gpu_model_runner.py:1298-1301`）——比如两组完全不相关的请求交替调度——这时增量更新退化成"整批删、整批加"，`condense()` 反而要做和重建等价的搬运工作，持久化不再省钱。此外单请求、低并发、CPU 从不成为瓶颈的场景，重建的简单性可能比持久化的正确性收益更划算（本库推断）。

### 5.2 CpuGpuBuffer + non_blocking H2D 拷贝

**为什么这么设计**：`pin_memory=True` 的 CPU 张量做 H2D 拷贝时可以用 DMA 异步搬运，`non_blocking=True`（`vllm/v1/utils.py:141-142`）让这次拷贝提交后立刻返回，不阻塞 CPU 线程去做下一步的 Python 逻辑（比如继续准备 attention metadata）。`_prepare_inputs()` 开头第一行就是 `self.input_batch.block_table.commit_block_table(num_reqs)`（`vllm/v1/worker/gpu_model_runner.py:2035`），注释写明"Start copying the block table first. This way, we can overlap the copy with the following CPU operations."（`vllm/v1/worker/gpu_model_runner.py:2033-2034`）——先发起拷贝，再继续算别的，是这套机制存在的直接理由。

**不这样会怎样**：如果用普通 pageable 内存 + 阻塞拷贝，`_prepare_inputs()` 里每一次 H2D 拷贝都会让 CPU 干等到拷贝完成才能往下走，前面攒起来的"边拷贝边算"的重叠窗口全部消失，CPU 端准备输入的墙钟时间会直接叠加到每步延迟上（本库推断，机制层面成立，具体幅度未查证）。

**什么时候可以不这样**：输入张量本来就很小（比如极短的 decode 步、batch size 极小）时，拷贝本身的耗时可能远小于 CUDA kernel launch 开销，重叠收益有限；这类场景下用不用 pinned memory 差别不大，但 vLLM 的实现并没有为此单独分支——`CpuGpuBuffer` 是统一路径，这里没有"什么时候可以不这样"的代码开关，是本库推断的理论边界而非源码里的实际分支。

### 5.3 CUDA Graph：capture 固定尺寸集合 + 运行时 padding

**为什么这么设计**：一个 `torch.cuda.CUDAGraph` 只能重放和捕获时**完全一致**的输入形状/输入地址（`vllm/compilation/cuda_graph.py` 的 `CUDAGraphEntry.input_addresses` 字段专门用来在调试模式下核对这一点）。vLLM 不可能为每一种可能出现的 `num_tokens` 都捕获一份图，所以只为一个预先算好的尺寸子集（`cudagraph_capture_sizes`）捕获，运行时把真实 `num_tokens` **向上填充**到最近的一个已捕获尺寸：

```python
# vllm/v1/cudagraph_dispatcher.py:72-91（节选）
def _compute_bs_to_padded_graph_size(self) -> None:
    max_size = self.compilation_config.max_cudagraph_capture_size
    capture_sizes = self.compilation_config.cudagraph_capture_sizes
    self._bs_to_padded_graph_size = [0] * (max_size + 1)
    for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
        for bs in range(start, end):
            self._bs_to_padded_graph_size[bs] = start if bs == start else end
```

`capture_model()`（`vllm/v1/worker/gpu_model_runner.py:6937-7041`）按"先捕获大尺寸、小尺寸复用大尺寸的显存池"的顺序（`vllm/v1/worker/gpu_model_runner.py:6952-6954` 注释）依次捕获，`for runtime_mode, batch_descs in self.cudagraph_dispatcher.get_capture_descs(): self._capture_cudagraphs(...)`（`vllm/v1/worker/gpu_model_runner.py:6999-7007`）。

**不这样会怎样**：如果不做 padding、要求每个 batch size 精确匹配一份已捕获的图，捕获数量会随最大 batch size 线性增长，启动时捕获耗时和显存占用都会随之膨胀（源码注释 `vllm/v1/worker/gpu_model_runner.py:7035` 处日志"This usually takes 5~20 secs"暗示了捕获本身已有实打实的时间成本，此处不引申为具体数字，因为这是该仓库自己的注释估计而非本库实测）；反过来如果完全不 capture，退化为每步都是 eager Python 前向 + 独立 kernel launch，`## 0` 提到的"不产出实测数字"红线不允许我在此断言具体倍数收益，但从机制上看，eager 路径每层都要过一遍 Python 调度器 + kernel launch 排队，CUDA Graph replay 是把整段调用序列录进一次 driver 提交，理论上能省掉这些 launch 开销（文档所述，来源：`vllm/config/compilation.py:621-623` 的字段文档提到"Can be good for small models or workloads with small prompts"，隐含小 workload 下 launch 开销占比更高）。

**什么时候可以不这样**：`cudagraph_mode` 显式设为 `NONE`（`vllm/v1/worker/gpu_model_runner.py:6938-6943` 会打印警告直接跳过 `capture_model()`），或者 `enforce_eager=True`（`## 4.1` 提到 `compile_or_warm_up_model` 里的判断）——都会让整条服务纯 eager 跑。此外 cascade attention 打开、encoder-decoder 模型带 encoder 输入的步（`## 7` 展开）都会在**某些具体批次**上被动退回非 FULL 模式，不是全局开关但效果类似。

### 5.4 FULL 只覆盖 decode，PIECEWISE 兜底 prefill/mixed

**为什么这么设计**：attention backend 通过 `AttentionCGSupport` 枚举（`vllm/v1/attention/backend.py:567-581`）声明自己能在什么形状约束下支持整图捕获：

```python
# vllm/v1/attention/backend.py:567-581
class AttentionCGSupport(Enum):
    """cudagraph support of the attention backend. Here we do not consider
    the cascade attention, as currently it is never cudagraph supported."""
    ALWAYS = 3               # 支持任意 mixed-prefill-decode
    UNIFORM_BATCH = 2        # 批内 query 长度一致即可（可覆盖 spec decode）
    UNIFORM_SINGLE_TOKEN_DECODE = 1   # 仅 query_len==1 的纯 decode
    NEVER = 0                 # 完全不支持（抽象基类默认值）
```

decode 步天然满足"每个请求推进固定数量 token"这一条件，是 `_is_uniform_decode()`（`vllm/v1/worker/gpu_model_runner.py:3985-4003`）判定的核心逻辑：

```python
# vllm/v1/worker/gpu_model_runner.py:3996-4000
return (
    (max_num_scheduled_tokens == uniform_decode_query_len)
    and (num_tokens == max_num_scheduled_tokens * num_reqs)
) if force_uniform_decode is None else force_uniform_decode
```

prefill 或 chunked-prefill 混合批次里每个请求的 token 数天然不同，绝大多数 backend 都声明不支持整图捕获，只能靠 PIECEWISE——把 attention 这类"形状依赖强"的算子排除在被捕获的子图之外（`vllm/config/compilation.py:617-619` 字段文档："PIECEWISE mode build piecewise cudagraph only, keeping the cudagraph incompatible ops ... outside the cudagraph"），图之间的部分照常走 eager/编译后的 Python 代码，靠 padding 到同一个 `num_tokens` 就能复用。

**不这样会怎样**：如果强行对 prefill 批次也用 FULL 图（`cudagraph_mode=FULL`，字段文档明说"not supported by many backends"，`vllm/config/compilation.py:621-622`），要么在不支持的 backend 上直接报错/退化，要么捕获数量会随请求可能出现的 token 分布组合爆炸——`FULL` 模式的默认捕获形状本来就假设是 uniform decode 场景，硬套到 ragged prefill 上等于要为几乎每一种长度组合都捕获一份图，不现实。

**什么时候可以不这样**：字段文档列出的 `FULL_DECODE_ONLY`——P/D 分离部署里的 D（decode-only）实例，"prefill is not as important so we can save some memory"（`vllm/config/compilation.py:625-627`），可以只捕获 decode 图、mixed 批次一律不上图；或者选一个 `AttentionCGSupport.ALWAYS` 的 backend（少数支持任意形状的实现），这样 prefill 也能进 FULL 图，但这是 backend 能力决定的，不是 `GpuModelRunner` 层面能凭空打开的开关。

### 5.5 execute_model / sample_tokens 拆成两次调用

**为什么这么设计**：结构化输出的语法约束比特掩码计算在 CPU（甚至独立线程/进程）上进行，和 GPU 上的 forward 计算天然可以重叠——`execute_model()` 提交完 forward 就返回 `None`，EngineCore 拿这个窗口去算 `grammar_output`，算完再传给 `sample_tokens()`（`## 4.5`）。这个拆分同时天然支持异步调度（`self.use_async_scheduling`，`vllm/v1/worker/gpu_model_runner.py:4704`）——下一步的 `execute_model()` 甚至可以在上一步的采样结果还没算完前就开始准备。

**不这样会怎样**：如果合并成一次调用内部顺序做"forward → 等语法约束算完 → sample"，两段计算就从并行变成了严格串行，语法约束计算的 CPU 耗时会直接叠加到每步延迟上，异步调度这类需要"提前发起下一步"的优化也失去了可插入的缝隙。

**什么时候可以不这样**：没有结构化输出、也没开异步调度的最简单场景下，两次调用之间事实上没有别的工作可插入，拆分带来的"可重叠窗口"是空的，此时两次调用退化成等价于一次——但源码没有为这种情况提供合并路径，`ExecuteModelState` 这一层状态传递始终存在（`## 3`），复杂度是固定成本，不随场景简化（本库推断：合并会破坏协议一致性，非源码显式声明的取舍点）。

### 5.6 权重直接建在目标设备上，而不是先 meta 再迁移

**为什么这么设计**：`initialize_model()`（`vllm/model_executor/model_loader/utils.py:38-61`）在 `with target_device:` 上下文里直接实例化模型类（`## 4.6`）——这让模型构造代码本身保持简单：`nn.Linear`、`nn.Embedding` 之类的标准 PyTorch 组件不需要额外感知"我现在是在 meta device 上"这件事，模型作者写的仍是普通的 `__init__`。

**不这样会怎样**：如果先在 meta device 上建骨架（不分配真实存储）、再在加载阶段把每个参数 materialize 到目标设备，理论上能省掉"先随机初始化一遍参数、再整体被 checkpoint 覆盖"这一步浪费的显存带宽和计算——但代价是模型定义代码必须处处小心 meta tensor 的特殊语义（不能做真正的数值运算），vLLM 选择了用严格性检查（`## 4.6` 第 3 步）去兜底"权重没被覆盖"的正确性风险，而不是从设备语义上直接杜绝这类问题。

**什么时候可以不这样**：`load_config.load_format == "dummy"`（`vllm/v1/worker/gpu_model_runner.py:5427-5428` 的 `load_dummy_weights` 分支）时本来就不需要真实权重，直接用随机初始化的目标设备张量跑，meta device 的收益（省去"先分配真实存储再被覆盖"的浪费）在这个场景里也不体现——因为反正没有第二次真实覆盖发生。

## 6. 同位对照：SGLang 在同一位置怎么做

SGLang 把"给 CUDA Graph 分阶段"这件事做得比 vLLM 更显式——它直接拆成两个类：`DecodeCudaGraphRunner`（`sglang:python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:200`）和 `PrefillCudaGraphRunner`（`sglang:python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:245`），共享 `BaseCudaGraphRunner`（`sglang:python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:105`）的调度骨架，而不是像 vLLM 那样用一个 `CUDAGraphMode` 枚举 + 一个 `CudagraphDispatcher` 在运行时查表决定当前是 FULL 还是 PIECEWISE。

`ModelRunner.forward()`（`sglang:python/sglang/srt/model_executor/model_runner.py:1520`）里能看到明确的三级 fallback：

```python
# sglang:python/sglang/srt/model_executor/model_runner.py:1681-1749（节选）
can_run_graph = bool(
    mode_check()
    and self.decode_cuda_graph_runner
    and self.decode_cuda_graph_runner.can_run_graph(forward_batch)   # L1683-1684
)
if can_run_graph:
    ret = self.decode_cuda_graph_runner.execute(forward_batch, ...)  # L1696-1701
    return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)
...
elif (
    forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
    and self.prefill_cuda_graph_runner is not None
    and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)  # L1726-1730
):
    # Prefill cuda graph (piecewise).                                # L1735
    ret = self.prefill_cuda_graph_runner.execute(forward_batch, **kwargs)  # L1746-1748
```

先判断 `decode_cuda_graph_runner.can_run_graph(...)`，能跑就直接 `execute()` 并早返回；不能跑就落到 `prefill_cuda_graph_runner.can_run_graph(...)`——代码注释直接写"Prefill cuda graph (piecewise)"（`sglang:python/sglang/srt/model_executor/model_runner.py:1735`），说明 **SGLang 的 prefill 图同样是 piecewise 风格**，和 vLLM 的判断收敛到了同一个结论：prefill 批次的动态形状迫使 attention 之类的算子留在被捕获的图之外，两者都不能奢望 prefill 拿到和 decode 一样的整图待遇。这不是巧合，是同一个物理约束（attention kernel 的输入形状随请求长度变化）逼出来的收敛设计（源码为证 + 本库推断：收敛性判断基于两边代码结构的直接对照）。

vLLM 用一个类（`GpuModelRunner`）内部按 `cudagraph_mode` 枚举分支，SGLang 用两个独立 Runner 类分工，是"一个执行器管所有形状 vs 按阶段拆执行器"两种工程取舍——本篇不评判孰优孰劣，具体权衡留给 [[05-SGLang-注意力后端矩阵]] 或跨引擎对比篇处理。

## 7. 踩坑与反直觉

**陷阱：cascade attention 打开会静默把 FULL 图降级掉，不报错、不崩溃，只是那一步偷偷变慢。**

`_determine_batch_execution_and_padding()` 里判定要不要用 FULL 图时，直接把 `use_cascade_attn`（以及 encoder-decoder 场景的 `has_encoder_output`）作为 `disable_full` 的条件传进去：

```python
# vllm/v1/worker/gpu_model_runner.py:4095-4107
def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
    return self.cudagraph_dispatcher.dispatch(
        num_tokens=num_tokens, has_lora=has_lora, uniform_decode=uniform_decode,
        num_active_loras=num_active_loras,
        valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
        invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
    )

cudagraph_mode, batch_descriptor = dispatch_cudagraph(
    num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
)
```

`AttentionCGSupport` 的枚举文档本身就写死了"cascade attention... currently it is never cudagraph supported"（`vllm/v1/attention/backend.py:568-570`）——这不是某个 backend 的临时限制，是整个 cascade attention 特性和 CUDA Graph 结构性冲突：cascade attention 需要按共享前缀长度动态分两段算 attention，这个"分几段、每段多长"本身就是运行时才知道的动态形状，没法固定进一份预先捕获好的图里。

`CudagraphDispatcher.dispatch()` 的兜底逻辑在满足 `num_tokens > max_size`、或 `allowed_modes` 被削减到只剩 `NONE` 等条件时，直接返回 `CUDAGraphMode.NONE`（`vllm/v1/cudagraph_dispatcher.py:274-281`）；`CUDAGraphWrapper.__call__()` 里一旦发现运行时模式和自己的模式不匹配，**直接把调用转发给底层 `self.runnable(*args, **kwargs)`**：

```python
# vllm/compilation/cuda_graph.py:244-254
if (
    cudagraph_runtime_mode == CUDAGraphMode.NONE
    or cudagraph_runtime_mode != self.runtime_mode
):
    # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
    # running without cudagraphs.
    return self.runnable(*args, **kwargs)
```

没有异常、没有默认打印的日志，这一步就是安安静静地跑了一次没有 Graph 加速的 forward。

**怎么在日志里认出来**：默认情况下这条路径完全不出声。要看见它，得显式打开 `--cudagraph-metrics`（对应 `observability_config.cudagraph_metrics`，默认 `False`，`vllm/config/observability.py:68`）——打开之后 `_determine_batch_execution_and_padding()` 每步都会构造一个 `CUDAGraphStat(runtime_mode=str(cudagraph_mode), ...)`（`vllm/v1/worker/gpu_model_runner.py:4147-4154`），`CUDAGraphLogging.generate_metric_table()`（`vllm/compilation/cuda_graph.py:71-118`）会把所有观测到的 `(unpadded tokens, padded tokens, paddings, runtime mode)` 组合聚合成一张 Markdown 表格打到日志里——如果这张表里某一行的 `Runtime Mode` 列写的是 `NONE`，且出现频次不低，那就是有一批批次一直在走 eager，而不是你以为的"图已经生效"。另一条独立线索是 `_prepare_kv_sharing_fast_prefill()`（`vllm/v1/worker/gpu_model_runner.py:2994-3016`）——它对 KV sharing fast-prefill 场景显式传了 `invalid_modes={CUDAGraphMode.FULL}`（`vllm/v1/worker/gpu_model_runner.py:3009-3010`），说明这是一个已知会被排除在 FULL 图之外的特性组合，不是 bug，是设计使然。

**次要陷阱：encoder-decoder 模型第一遍（带 encoder 输入）的步骤会被强制走 eager。** `set_forward_context(..., skip_compiled=has_encoder_input)`（`vllm/v1/worker/gpu_model_runner.py:4552`，`has_encoder_input` 定义于 `vllm/v1/worker/gpu_model_runner.py:4526-4528`）把这个标志一路传到 `support_torch_compile` 生成的 `__call__()`，命中就直接走原始 `forward`（`## 4.4` 已展开代码），源码注释解释原因："tensor shapes/types vary across invocations, preventing the capture of a single computational graph"（`vllm/compilation/decorators.py:510-511`）。这条路径同样不报错，只是这一类模型的这一类步骤永远吃不到编译/图加速的红利——查代码前光看日志很难分辨"这个模型本来就不支持"还是"配置错了"。

## 8. 可改进点

以下均为本库阅读代码后的推断，未在上游 issue/PR 中核实是否已有讨论（诚实标准第 6 条：不编 issue 号）。

1. **`--cudagraph-metrics` 默认关闭，观测门槛偏高。** cascade attention/encoder-decoder 这类"静默降级"场景（`## 7`）只有在用户主动打开这个 flag 之后才能在日志里看见；对于排障，默认打开一个采样频率很低（比如每 N 步一次）的轻量版统计、或者在检测到某个 batch shape 连续多次落进 `NONE` 分支时打一条 `logger.warning_once`，会显著降低用户"以为开了图、实际一直在吃 eager"的踩坑概率。
2. **`condense()`（`vllm/v1/worker/gpu_input_batch.py:708-840` 起）的正确性高度依赖多处状态数组同步移动**——`token_ids_cpu`、`spec_token_ids`、`block_table`、`num_computed_tokens_cpu` 等十几处字段都要在同一次 `empty_index`/`last_req_index` 交换里各自搬一遍，任何一次新增字段忘了在这里补一行都是一个隐蔽的状态错位 bug，且大概率只在"请求被移除导致压缩"这种非最常见路径上触发，日常小规模测试很难覆盖到。给 `InputBatch` 补一个"字段清单自检"（比如遍历 `__init__` 里所有 `(max_num_reqs,)` 形状的属性，断言 `condense()` 确实处理过每一个），能把这类疏漏从运行时踩坑前移到 CI 阶段。
3. **`load_weights()` 的"严格性检查"（`## 4.6` 第 3 步）默认只在非量化模型上开启**（`vllm/model_executor/model_loader/default_loader.py:436-438` 的 `default_enable_weights_track` 判断），量化模型的权重覆盖情况因此少了这道安全网——量化路径本身逻辑更复杂（在线量化、layerwise 处理），恰恰更需要这类"谁没被加载"的检查，而不是默认豁免。把追踪逻辑扩展到能正确处理量化场景的"预期被跳过的参数名单"（而不是整体关闭检查），能补上这块盲区。

## 9. 自测题与延伸阅读

**自测题**（闭卷回答，答案均可在 `## 2`~`## 7` 用到的行号里核实）：

1. `execute_model()` 和 `sample_tokens()` 为什么要拆成两次独立调用？拆分之后 `ExecuteModelState` 起什么作用？
2. `InputBatch` 里的 `condense()` 在什么条件下才会真的搬动数据？如果所有被移除的请求槽位后面没有活跃请求需要往前压，它会做什么？
3. 一个 attention backend 声明自己的 `AttentionCGSupport` 是 `UNIFORM_SINGLE_TOKEN_DECODE`，这意味着它能否支持开了投机解码（每步验证 `1+num_speculative_tokens` 个 token）的 decode 步整图捕获？为什么？
4. `cudagraph_mode=NONE`（全局关闭）和某个具体 batch 在运行时被 `CudagraphDispatcher.dispatch()` 判定回退到 `CUDAGraphMode.NONE`（局部降级），这两种"NONE"在用户可观测的效果上有什么区别？
5. `cache_dir` 没有手动指定时，它的哈希由哪几个因子拼成？如果只改了模型权重文件（不改推理代码、不改 vLLM 配置、环境变量不变），编译缓存会不会失效？为什么？
6. `_is_uniform_decode()` 判定"这是一个 uniform decode 批次"用了哪两个条件？如果只满足其中一个会发生什么？
7. `Sampler.forward()` 的文档把 logits processor 分成"argmax-invariant"和"non-argmax-invariant"两类分别在流程的哪两步插入？为什么不能都放在同一步？
8. `load_model()` 三段式（建骨架 → 灌权重 → 后处理）里，哪一步会因为参数没被 checkpoint 覆盖而直接抛异常？这个检查默认对哪类模型不生效？

**延伸阅读**：

- [[03-vLLM-调度器解剖]]——`SchedulerOutput` 从哪来
- [[04-vLLM-KV缓存与前缀缓存]]——block table 与 slot mapping 的上游
- [[05-vLLM-注意力后端与算子层]]——`AttentionCGSupport` 各后端实现细节
- [[10-vLLM-投机解码]]——`sample_tokens()` 里的 drafter 分支
- [[11-vLLM-结构化输出]]——`grammar_output` 的产出方
- [[00-总览与阅读地图]]
