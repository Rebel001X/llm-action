# vLLM 注意力后端与算子层

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：选后端看能力声明，MLA 另起契约

## 0. 结论先行

- 注意力子系统按 `_lab/out/struct_map.json` 的统计是 158 个文件 / 81,107 行，占了 vLLM 里相当大的一块表面积——但真正的"后端清单"是一个**单一枚举**：`vllm/v1/attention/backends/registry.py:34`-`131` 的 `AttentionBackendEnum`，本篇实测数出 37 个具名成员（含一个留给第三方注册的 `CUSTOM` 空位），不是散落在各处的 `if backend == "xxx"` 字符串比较。
- **`VLLM_ATTENTION_BACKEND` 这个环境变量在本篇取证的这一版已经不存在了**——本库对整个 `_src/vllm` 做了全文检索，零命中。选后端现在只有三个入口：CLI `--attention-backend`、结构化的 `--attention-config.backend`（二者互斥，`vllm/engine/arg_utils.py:2401`-`2406` 会在同时设置时直接报错）、以及 Python API 里的 `AttentionConfig(backend=...)`。这条反直觉的事实值得放在结论最前面，因为网上大量教程仍在教你 `export VLLM_ATTENTION_BACKEND=FLASHINFER`。
- "自动选择"的本质不是一份写死的性能排行榜，而是**每个硬件平台各自维护一份优先级列表 + 每个 backend 类自己声明一组正交能力谓词**（head_size、dtype、kv_cache_dtype、block_size、是否 MLA、是否支持 sink、是否稀疏……共 16 项，`vllm/v1/attention/backend.py:263`-`348`），选择时按优先级顺序找第一个"全部谓词通过"的 backend。这是本篇 `## 5` 要重点讲的"组合爆炸"控制机制。
- 契约分三层，生命周期完全不同：`AttentionBackend`（构建期确定一次的静态能力声明+类型注册表）、`AttentionMetadataBuilder`（每个 attention group 一个长驻实例，可能持有 CUDA Graph 用的持久化缓冲区）、`AttentionMetadata`（**几乎每个 decode/prefill step 都要重新 `build()` 一次**的临时数据，`vllm/v1/worker/gpu_model_runner.py:2608`）。选 backend 便宜（结果被 `@functools.cache` 记住），建 metadata 不便宜（每步都要跑一遍）。
- MLA 不是"多头注意力换一个 kernel"：它把 KV cache 压缩成单个 latent 向量，`num_kv_heads=1`、`head_size = kv_lora_rank + qk_rope_head_dim`（`vllm/model_executor/layers/attention/mla_attention.py:438`，DeepSeek-V3 上是 512+64=576），接口上直接换成 `forward_mha`/`forward_mqa` 两个方法而不是一个 `forward`（`vllm/v1/attention/backend.py:983`-`1037`），所以 MLA 走的是完全独立的 `MLAAttentionImpl` 抽象基类，不是 `AttentionImpl` 的一个特例。
- 算子来源是"三明治"结构：vLLM 自己手写的主要是 KV cache 读写胶水（C++/CUDA，如 `concat_and_cache_mla`）和 18 个含 `@triton.jit` 的 Triton kernel 文件（`vllm/v1/attention/ops/` 目录下），真正的注意力计算主力（FlashAttention、FlashInfer、CUTLASS）都是 vendor 集成——但集成方式不同：FlashAttention 是 vLLM 自己维护的 fork（`vllm-project/flash-attention`），通过 CMake `ExternalProject` 拉取指定 commit 编译进 vLLM 自己的 `.so`；FlashInfer 是纯 pip 依赖（`flashinfer-python==0.6.17`）。而 vLLM 早期"招牌"的手写 `paged_attention_v1`/`v2` CUDA kernel，在本篇取证的这一版 CUDA 路径上已经**完全消失**，全仓唯一残留的字符串引用是 ROCm AITER 库自己的算子名，和 vLLM 自己的 kernel 无关。

## 1. 它在系统里的位置

注意力后端处在模型定义和物理硬件之间的一个中间层，横切两条不同粒度的流程：

**构建期（选一次）**：模型加载时，每个 `Attention`/`MLAAttention` 层的 `__init__` 会调用一次 `get_attn_backend()`（普通注意力见 `vllm/model_executor/layers/attention/attention.py:339`-`350`，`use_mla=False` 硬编码传入），把选出来的 backend 类存成 `self.attn_backend`。这个决定对这一层的生命周期是**永久的**——不会在推理过程中重新选择。选择结果本身还被 `vllm/v1/attention/selector.py:194` 的 `@cache` 装饰器记住，同一份 `AttentionSelectorConfig` 只算一次。

**运行期（每步重建 metadata，调用 kernel）**：模型 forward 时，`Attention.forward()` 不接收 metadata 作为参数，而是通过一个全局 `forward_context` 拿——文档字符串原话是"Attention metadata (`attn_metadata`) is set using a context manager in the model runner's `execute_model` method... accessed via forward context using `vllm.forward_context.get_forward_context().attn_metadata`"（`vllm/model_executor/layers/attention/attention.py:489`-`496`）。也就是说 backend 的**类型**在构建期就已经固定死了，但 backend 需要的**数据**（block table、slot mapping、seq_lens……）是每一步由 `GPUModelRunner` 现算现填进 context 的。

在整个请求生命周期里，注意力后端上游是 [[03-vLLM-调度器解剖]] 决定的"这一步谁上场、吃几个 token"和 [[04-vLLM-KV缓存与前缀缓存]] 决定的"逻辑块分到了哪些物理块号"；下游是 [[06-vLLM-模型执行与CUDA-Graph]] 的 CUDA Graph 捕获（要求 metadata 的 shape 在图内保持稳定，这也是 `AttentionMetadataBuilder` 要单独提供 `build_for_cudagraph_capture()` 的原因）。注意力后端本身既不决定"批多大"也不决定"KV 放哪个块"，它只负责：把调度器和 KV 缓存管理器已经决定好的东西，翻译成某个具体 kernel 能听懂的张量布局，然后调用它。

## 2. 代码地图（文件 → 职责，带行号）

按"契约定义 → 后端登记 → 具体实现 → 底层算子 → 平台选择逻辑"五层来看：

**契约定义层**（一份抽象，所有后端共用）：
- `vllm/v1/attention/backend.py:59`-`348` —— `AttentionBackend` 抽象基类：能力声明（`supports_head_size`/`supports_dtype`/…）+ `validate_configuration()` 的组合校验逻辑，是本篇 `## 5` 的核心。
- `vllm/v1/attention/backend.py:584`-`748` —— `AttentionMetadataBuilder` 抽象基类：`build()` 是唯一必须实现的抽象方法（`656`-`673`）。
- `vllm/v1/attention/backend.py:869`-`982` —— `AttentionImpl`（普通注意力的 `forward()` 契约）。
- `vllm/v1/attention/backend.py:983`-`1072` —— `MLAAttentionImpl`（`forward_mha`/`forward_mqa` 双方法契约，MLA 专用）。

**后端登记层**（一份枚举，37 个具名成员）：
- `vllm/v1/attention/backends/registry.py:34`-`131` —— `AttentionBackendEnum`，每个成员的值是一条可被 `resolve_obj_by_qualname` 动态 `import` 的类路径字符串；`register_backend()`（`243`-`295`）允许运行时覆盖或注册第三方后端到 `CUSTOM` 槽位。

**具体实现层**（`vllm/v1/attention/backends/` 下 22 个顶层文件 + `mla/` 子目录 19 个文件）：
- `vllm/v1/attention/backends/flash_attn.py:78`-`147` —— `FlashAttentionBackend`，`get_name()` 返回 `"FLASH_ATTN"`，`supports_head_size()` 声明 `head_size % 8 == 0` 且 ≤256（FA4 可到 ≤512）。
- `vllm/v1/attention/backends/flashinfer.py`（全文件 2,648 行，本篇取证版本最大的单文件之一）—— `FlashInferMetadataBuilder`（`struct_map.json` 记录起于第 660 行）。
- `vllm/model_executor/layers/attention/mla_attention.py:1435`-`1464` —— `MLACommonBackend`，`is_mla()` 恒真、`get_supported_head_sizes()` 返回 `[320, 576]`。
- `vllm/v1/attention/backends/rocm_attn.py:163`-`218` —— `RocmAttentionBackend`，ROCm 平台自己的一条非-AITER 路径（`## 5` 决策里"要不要依赖 AITER 库"是 ROCm 优先级列表的一个分叉点）。
- `vllm/v1/attention/backends/cpu_attn.py:45`-`74` —— `CPUAttentionBackend`，CPU 平台的默认非 MLA 后端，`get_name()` 与 `get_supported_head_sizes()` 定义都在同一小段。
- `vllm/v1/attention/backends/turboquant_attn.py:1`-`17` —— `TurboQuantAttentionBackend`，文件头注释直接给出了 cache 物理布局：每个 slot 是 `[压缩后的 K（若干字节）| FP16 的 V]` 拼接，prefill 阶段用标准稠密注意力算完再压缩存 K，decode 阶段直接在压缩域里算注意力分数——这是本篇遇到的**唯一一个自定义了非标准 KV cache 物理布局**的后端，其余后端复用的都是标准 paged KV cache 布局。
- `vllm/v1/attention/backends/flash_attn_diffkv.py:35`-`52` —— `FlashAttentionDiffKVBackend`，`DiffKV` 指 K、V 的 head dim 不相等（`hdim_qk != hdim_v`）这种非对称场景，要求 FA3 或 FA4——命名容易让人误以为是"另一种 KV cache 差分压缩算法"，其实只是"Q/K 头维度和 V 头维度不同"的字面缩写（本篇 `## 7` 会展开这条容易读错的命名）。
- `vllm/v1/attention/backends/registry.py:112` —— `NO_ATTENTION` 枚举条目指向的 `no_attention.py` 文件在这版快照里**不存在**（`## 7` 详述），是本篇在核对代码地图时顺带发现的死引用，不是本库凭空猜测。

**算子层**（真正跑在 GPU 上的代码）：
- `vllm/v1/attention/ops/`（34 个文件）——`_lab` 统计显示这个目录下有 18 个文件含 `@triton.jit`，是 vLLM 自己写的 Triton kernel 集中地，例如 `triton_unified_attention.py`（1,189 行）、`chunked_prefill_paged_decode.py`。
- `csrc/libtorch_stable/cache_kernels.cu`、`csrc/libtorch_stable/attention/merge_attn_states.cu` —— vLLM 自己手写的 C++/CUDA，负责 KV cache 写入（`reshape_and_cache`、`concat_and_cache_mla`）和分块结果合并，**不是**注意力矩阵乘本身。
- `csrc/attention/`（仅 6 个 `.cuh` 头文件：`attention_generic.cuh`/`dtype_*.cuh`）—— 只剩数据类型转换辅助代码，没有一个 `.cu` kernel 源文件。

**平台选择层**：
- `vllm/v1/attention/selector.py:102`-`212` —— `get_attn_backend()` 入口 + `_cached_get_attn_backend()`（`@cache` 记忆化）。
- `vllm/platforms/cuda.py:82`-`163` —— `_get_backend_priorities()`，NVIDIA 平台的优先级列表，按 `device_capability.major`（Blackwell=10/12 等）和 MLA/非 MLA 分叉。
- `vllm/platforms/cuda.py:403`-`502` —— `CudaPlatform.get_attn_backend_cls()`，手动选择 vs 自动选择两条路径。
- `vllm/platforms/rocm.py:459`-`495`、`617`-`663` —— ROCm 平台的独立优先级列表和选择逻辑（依赖 AITER 库是否可用）。
- `vllm/platforms/cpu.py:82`-`119` —— CPU 平台，`CPU_ATTN`/`CPU_MLA`/`AMX_MLA` 三选一，没有优先级列表，纯 if/elif。
- `vllm/platforms/tpu.py:9`-`20` —— TPU 平台把整个实现委托给外部包 `tpu_inference`（本篇 `## 7` 详述）。

以上 10+ 条引用已覆盖构建期选择、运行期实现、底层算子三个层次，行号均已用 `Read` 工具逐条核对。

## 3. 核心数据结构

**`AttentionBackendEnum`**（`vllm/v1/attention/backends/registry.py:34`）——一个 `Enum`，值是类路径字符串而不是类对象本身，好处是**引用一个后端不需要 `import` 它**（很多后端依赖只有对应硬件才装得上的包，比如 FlashInfer）。真正拿到类靠 `get_class()`（`151`-`161`），内部调用 `resolve_obj_by_qualname` 做懒加载 `import`。

**`AttentionSelectorConfig`**（`vllm/v1/attention/selector.py:21`-`59`）——一个 `NamedTuple`，把选后端要看的所有轴打包成一份可哈希的配置：`head_size`、`dtype`、`kv_cache_dtype`、`block_size`、`use_mla`、`has_sink`、`use_sparse`、`use_mm_prefix`、`use_per_head_quant_scales`、`attn_type`、`has_sliding_window`、`use_non_causal`、`use_batch_invariant`、`use_kv_connector`、`use_pcp`、`use_adaptive_verification`、`use_dcp`——共 17 个字段。可哈希是关键：正因为它是 `NamedTuple` 而不是普通 dataclass，`@cache` 才能把它当字典键用。

**`AttentionConfig`**（`vllm/config/attention.py:20`-`39`）——用户可见的配置对象：`backend: AttentionBackendEnum | None`（`None`/`"auto"` 触发自动选择，`158`-`170` 的 `field_validator` 负责把字符串 `"auto"` 转成 `None`）；`backend_per_kind: dict[str, AttentionBackendEnum]`（按 `KVCacheSpecKind` 分组覆盖，见下）。

**`KVCacheSpecKind`**（`vllm/v1/kv_cache_interface.py:130`-`140`）——`FULL_ATTENTION`/`MLA_ATTENTION`/`SLIDING_WINDOW`/`SLIDING_WINDOW_MLA`/`MAMBA`/`CHUNKED_LOCAL_ATTENTION`/`SINK_FULL_ATTENTION`/`ENCODER_ONLY_ATTENTION`/`CROSS_ATTENTION`/`UNKNOWN` 十种。`backend_per_kind` 的 key 就是这个枚举的字符串值——这意味着同一个模型如果混了全注意力层和滑窗层（比如 gpt-oss 那种交替结构），可以给两组分别指定不同 backend，而不是被迫用同一个。

**`AttentionMetadata`**（`vllm/v1/attention/backend.py:361`-`362`）——本体是个空类（`pass`），真正干活的是子类和 `CommonAttentionMetadata`（`368`-`562`）：`query_start_loc`/`seq_lens`/`num_reqs`/`block_table_tensor`/`slot_mapping` 等字段，其中不少同时保留 GPU 和 CPU 两份副本（比如已弃用但仍在用的 `_seq_lens_cpu`），代价是每步都要维护两份同步的张量。`CommonAttentionMetadata` 是所有 backend 共用的"半成品"，各 backend 的 `AttentionMetadataBuilder.build()` 在此基础上加工出自己需要的最终形态（比如 FlashInfer 需要的 `FlashInferMetadata`）。

**`AttentionCGSupport`**（`vllm/v1/attention/backend.py:567`-`582`）——一个四级枚举：`NEVER`(0)/`UNIFORM_SINGLE_TOKEN_DECODE`(1)/`UNIFORM_BATCH`(2)/`ALWAYS`(3），描述这个 backend 的 metadata builder 能不能被 CUDA Graph 捕获、能捕获到哪种批次形状（比如投机解码的批次里每个请求的 query 长度都是 `1+num_speculative_tokens`，属于 `UNIFORM_BATCH` 而非 `ALWAYS`）。

**`AttentionGroup`**（`vllm/v1/worker/utils.py:249`-`296`）——运行期把"哪些层共享同一个 backend + 同一个 KV cache spec"归成一组的容器，持有 `metadata_builders: list[AttentionMetadataBuilder]`（列表是因为 micro-batching 场景下每个 ubatch 需要独立的 builder 实例，避免持久化缓冲区互相打架）。这是 backend 类型（构建期定死）和 metadata（每步重建）之间真正的"中间人"。

**`MultipleOf`**（`vllm/v1/attention/backend.py:52`-`57`）——一个只有一个字段 `base: int` 的极简包装类，`AttentionBackend.get_supported_kernel_block_sizes()`（`72`-`74`，默认返回 `[MultipleOf(1)]`）用它表达"kernel 只要求 block_size 是某个数的倍数"而不是"精确等于某个数"。`supports_block_size()`（`116`-`133`）拿用户设置的 `block_size` 对着这个列表逐项取模检查——这是"kernel 物理约束"和"框架逻辑块大小"之间的一层解耦，本篇 `## 5` 决策 7 会展开这个不起眼但同样在做组合控制的小类。

**`AttentionLayer` Protocol**（`vllm/v1/attention/backend.py:749`-`767`）——`forward()` 方法之外只声明了一组量化用的标量/张量字段（`_q_scale`/`_k_scale`/`_v_scale`/`_prob_scale` 及其 `_float` 变体），这是 `AttentionImpl.forward()` 的第一个参数 `layer` 的类型约束：每个 backend 的 `forward()` 实现不需要关心"这一层怎么被创建的"，只需要知道它能读到这几个量化 scale。

**`AttentionImplBase`**（`vllm/v1/attention/backend.py:770`-`864`）——`AttentionImpl` 和 `MLAAttentionImpl` 的共同父类，携带一批与"这个 kernel 在分布式并行下怎么表现"相关的声明式字段：`supports_pcp`/`supports_dcp`（是否支持 prefill/decode context parallelism）、`lse_base_on_e`（这个 backend 返回的 softmax log-sum-exp 是自然对数底还是 2 为底——`803`-`804` 的注释原话点名 DCP 的合并 kernel 会按这个标志分支，"getting it wrong silently corrupts the cross-shard softmax denominator"，选错了不会报错只会算错）、`dcp_world_size`/`dcp_rank`/`pcp_world_size`/`pcp_rank`（`837`-`858` 的 `__new__` 里统一初始化，取自分布式进程组，取不到就退化成 `1`/`0`）。

## 4. 主流程走读

### 4.1 构建期：选一次

`Attention.__init__`（`vllm/model_executor/layers/attention/attention.py:339`-`350`）直接调用 `get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla=False, ...)`。这条调用链：

```python
# vllm/v1/attention/selector.py:102 附近（节选）
def get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla=False, ...):
    vllm_config = get_current_vllm_config()
    attn_selector_config = AttentionSelectorConfig(...)          # 打包 17 个轴
    backend = vllm_config.attention_config.backend                # 用户指定的，或 None
    if attention_config.backend_per_kind:                         # 按 KV-cache-group 覆盖
        kind = get_attn_spec_kind(use_mla, has_sliding_window, attn_type)
        backend = attention_config.backend_per_kind.get(kind.value, backend)
    return _cached_get_attn_backend(backend, attn_selector_config, num_heads)  # L187-191
```

`_cached_get_attn_backend`（`vllm/v1/attention/selector.py:194`-`212`）把工作甩给 `current_platform.get_attn_backend_cls()`——这是一个按平台分发的多态调用，`CudaPlatform`/`RocmPlatform`/`CpuPlatform` 各自有自己的实现。以 CUDA 为例（`vllm/platforms/cuda.py:403`-`502`）：

- **手动选择**（`selected_backend is not None`）：只验证这一个 backend 的 `validate_configuration()`，通过就直接用，不通过就抛 `ValueError` 并把不通过的原因打进异常信息（`422`-`430`）——不会静默降级。
- **自动选择**（默认）：调 `get_valid_backends()`（`362`-`401`），拿到 `_get_backend_priorities()`（`82`-`163`）给出的优先级列表，逐个跑 `validate_configuration()`，收集所有"合法候选"，用 `min(..., key=priority)` 选优先级数字最小（即最靠前）的那个（`463`-`469`）。如果 `--block-size` 排除了本来优先级更高的候选，还会专门打一条 warning 告诉用户"你的 block-size 设置把更快的 backend 挤掉了"（`473`-`490`）。

优先级列表本身是**声明式**的分支表，不是运行时探测硬件后现算的：MLA 在 SM100（Blackwell）上又按 KV cache dtype 是否量化、`num_heads` 是否 ≤16 进一步拆成两套 sparse backend 排序（`vllm/platforms/cuda.py:98`-`116`），非 MLA 情况下 SM100 又因为"cutlass 非因果路径有问题"而把 `FLASHINFER` 排到 `FLASH_ATTN` 前面、`use_non_causal=True` 时反过来（`144`-`163`）——这些都是维护者根据已知问题手写进列表顺序里的，不是自动 benchmark 出来的。

### 4.2 运行期：每步重建 metadata

`GPUModelRunner._prepare_inputs()` 在每个 step 开始时构造一份 `CommonAttentionMetadata`（`vllm/v1/worker/gpu_model_runner.py:2497`-`2516`），塞满 `query_start_loc`/`seq_lens`/`block_table_tensor`/`slot_mapping` 等这一步的真实数据。随后对每个 `AttentionGroup` 调用其 `metadata_builder.build()`（`2608`-`2612`）：

```python
# vllm/v1/worker/gpu_model_runner.py:2594-2614（节选，逻辑简化）
if for_cudagraph_capture:
    attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
elif cache_key in cached_attn_metadata and builder.supports_update_block_table:
    # 同一步内，多个 hybrid KV-cache-group 共享同一个 builder 类型时，
    # 只更新 block table，不整个重新 build
    attn_metadata_i = builder.update_block_table(cached_attn_metadata[cache_key], ...)
else:
    attn_metadata_i = builder.build(common_prefix_len=..., common_attn_metadata=...)
```

关键点：`cached_attn_metadata` 这个字典是**函数局部变量**，每次调用 `_prepare_inputs()` 都会重新初始化为空——它只在"同一步内、多个 KV-cache-group 共享同一个 builder 类型"这个场景下省一次 build，**跨 step 完全不缓存**。也就是说，除非命中 CUDA Graph 捕获路径（复用固定 buffer）或 `update_block_table` 这个特殊优化通道，`AttentionMetadata` 本质上是每个 decode/prefill step 都要重新构造的一次性对象——`AttentionMetadataBuilder` 实例本身是长驻的（在 `AttentionGroup.create_metadata_builders()`，`vllm/v1/worker/utils.py:261`-`291` 里模型加载时创建一次），但它产出的 `AttentionMetadata` 不是。

之后模型 forward 时，`Attention.forward()`（`vllm/model_executor/layers/attention/attention.py:478`）不显式接收 metadata，而是从 `forward_context` 里取（`489`-`496`），调用 `self.impl.forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)`——`self.impl` 就是构建期选好的那个 `AttentionImpl` 实例，`forward()` 内部才真正调用 kernel（自研 Triton、还是 vendor 的 FlashAttention/FlashInfer C++ 扩展）。

### 4.3 MLA 的岔路

MLA 层在构建期走的是同一套 `get_attn_backend(use_mla=True, ...)` 入口（`MLAAttention.get_attn_backend()`，`vllm/model_executor/layers/attention/mla_attention.py:1191`-`1193`），但拿到的 backend 类实现的是 `MLAAttentionImpl` 而不是 `AttentionImpl`，`forward_impl()`（`vllm/model_executor/layers/attention/mla_attention.py:744`）内部会根据这一步是 prefill 还是 decode 分别调 `forward_mha`（计算友好，MHA 语义）或 `forward_mqa`（数据搬运友好，MQA 语义）——这两条路径甚至可能对应**两个不同的 backend 实现**（MLA 的 prefill backend 和 decode backend 是分开选的，见 `## 5` 决策 5）。

### 4.4 优先级列表全貌：四条分叉路径

`_get_backend_priorities()`（`vllm/platforms/cuda.py:82`-`163`）实际展开是四条互斥路径，本篇把verified 到的顺序摘成一张表（"优先"列数字越小越靠前）：

| 场景 | 触发条件 | 优先级顺序（节选） |
|---|---|---|
| 非 MLA，SM100 且因果 | `device_capability.major == 10 and not use_non_causal` | `FLASHINFER` → `FLASH_ATTN` → `TRITON_ATTN` → `FLEX_ATTENTION` → `TURBOQUANT` |
| 非 MLA，其余情况 | 上面条件不满足（含 SM100 非因果） | `FLASH_ATTN` → `FLASHINFER` → `TRITON_ATTN` → `FLEX_ATTENTION` → `TURBOQUANT` |
| MLA，SM100 | `use_mla and device_capability.major == 10` | `FLASHINFER_MLA` → `TOKENSPEED_MLA` → `CUTLASS_MLA` → `FLASH_ATTN_MLA` → `FLASHMLA` → `TRITON_MLA` → 两个 sparse 变体（顺序按 kv dtype/num_heads 再分叉） |
| MLA，SM120 | `use_mla and device_capability.major == 12` | `TRITON_MLA` → `FLASHINFER_MLA_SPARSE_SM120` |
| MLA，其余算力 | 上面两条都不满足 | `FLASH_ATTN_MLA` → `FLASHMLA` → `FLASHINFER_MLA` → `TRITON_MLA` → `FLASH_ATTN_MLA_SPARSE` → `FLASHMLA_SPARSE` |

（数据来源：`vllm/platforms/cuda.py:93`-`163` 逐条读出，非因果 SM100 与"其余情况"共用一套顺序是本篇核对时发现的一个容易忽略的细节——"SM100" 不是一个统一分支，因果与否会导致完全不同的优先级表。）

### 4.5 选完 backend 之后，某些 backend 内部还有一层版本选择

选中 `FLASH_ATTN` 只是决定了"用哪一族 kernel"，具体用 FlashAttention 2、3 还是 4，是**backend 内部的第二层选择**：`get_flash_attn_version()`（`vllm/v1/attention/backends/fa_utils.py:70`-`119`）按 `device_capability.major` 给出默认值（SM90 Hopper 优先 FA3、SM100+ Blackwell 优先 FA4、其余退回 FA2），再看 `AttentionConfig.flash_attn_version` 是否有显式覆盖（`105`-`110`），最后还有一条"SM100 上选了 FA3 但不支持，自动降级到 FA4 或 FA2"的兜底（`112`-`118`）。这条选择逻辑完全在 `AttentionBackendEnum.FLASH_ATTN` 这一个枚举成员内部发生，`_get_backend_priorities()` 那张表根本看不到——也就是说"选后端"和"选后端版本"是**两层独立的自动选择**，各自有各自的降级路径，理解 backend 选型时如果只看外层优先级列表，会漏掉这一层。

### 4.6 多模态编码器（ViT）走的是另一套独立选择逻辑

容易被忽略的一点：如果模型带视觉编码器（ViT），编码器里的注意力层**不走本篇 `## 4.1` 那套 `get_attn_backend()`/`_get_backend_priorities()` 流程**，而是单独的 `get_supported_vit_attn_backends()` + `get_vit_attn_backend()`（`vllm/platforms/cuda.py:504`-`558`）。这套逻辑更简单：`get_supported_vit_attn_backends()` 按 `has_device_capability(80)` 直接返回一个固定的四元素列表（`FLASH_ATTN`/`TRITON_ATTN`/`TORCH_SDPA`/`FLASHINFER`，具体顺序还会按算力翻转，`506`-`519`），`get_vit_attn_backend()` 遍历这个列表，只检查 `supports_head_size()` + `supports_dtype()` + `supports_compute_capability()` 三项就选定——**完全不走 `validate_configuration()` 的完整 16 项校验**，因为 ViT 编码器场景没有 KV cache 分页、没有因果 mask、不涉及 MLA/sink/sparse 这些解码器特有的复杂度，能省略的检查维度天然就少。这意味着"这个 vLLM 版本支持哪些 attention backend"这个问题本身没有唯一答案——要看你问的是解码器侧还是编码器侧。

### 4.7 端到端走一遍：一个 DeepSeek 系 MLA 模型在 Blackwell（SM100）上会发生什么

把前面几节串起来，假设部署一个 `use_mla=True`、`num_heads=8`、BF16 KV cache、没有显式设置 `--attention-backend` 的 DeepSeek 系模型：

1. **构建期**：`MLAAttention.__init__` 调 `get_attn_backend(use_mla=True, ...)`（`vllm/v1/attention/selector.py:102`），`AttentionConfig.backend` 是 `None`（用户没指定），`backend_per_kind` 也是空——直接进自动选择。
2. `current_platform.get_attn_backend_cls()`（CUDA 平台，`vllm/platforms/cuda.py:403`）发现 `selected_backend is None`，调 `get_valid_backends()`。
3. `_get_backend_priorities(use_mla=True, device_capability.major=10, num_heads=8, kv_cache_dtype="auto", ...)`（`vllm/platforms/cuda.py:93`-`129`）：走 `## 4.4` 表格里"MLA，SM100"这一支——`kv_cache_dtype` 不是量化格式（`is_quantized_kv_cache` 为假）且 `num_heads=8 <= 16`，所以 sparse 变体的顺序是 `FLASHINFER_MLA_SPARSE` 优先于 `FLASHMLA_SPARSE`（`vllm/platforms/cuda.py:107`-`111`）；主列表最终是 `FLASHINFER_MLA → TOKENSPEED_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA → FLASHINFER_MLA_SPARSE → FLASHMLA_SPARSE`。
4. 逐个跑 `validate_configuration()`：假设 `FLASHINFER_MLA` 全部谓词通过（BF16、`auto` kv dtype、`use_mla=True` 匹配 `is_mla()`……），它是列表里优先级数字最小的合法候选，直接选定，不再往后看。
5. 选定结果被 `_cached_get_attn_backend` 的 `@cache` 记住——同一进程里，另一层 `num_heads`/`dtype` 相同的 MLA 层会直接命中缓存，不重新跑一遍上面 4 步。
6. **运行期**：每个 step，`GPUModelRunner._prepare_inputs()` 建好 `CommonAttentionMetadata`，对这一层所在的 `AttentionGroup` 调 `FlashInferMLAMetadataBuilder.build()`（`## 4.2`），产出这一步专属的 `AttentionMetadata`。
7. `MLAAttention.forward_impl()`（`vllm/model_executor/layers/attention/mla_attention.py:744`）按这一步是 prefill 还是 decode 分流：prefill 调 `forward_mha`（如果 `mla_prefill_backend` 没显式指定，`## 5` 决策 5 提到的"tries FlashAttention first"规则在这里生效，和第 3 步选定的 `FLASHINFER_MLA` decode backend 是**两个独立的选择结果**）；decode 调 `forward_mqa`，内部才真正调用 FlashInfer 的 MLA CUDA kernel。

这条链路里，"选 backend"（步骤 1-5）只在模型加载时发生一次，"填数据 + 调 kernel"（步骤 6-7）每个 step 都要重新走——这正是 `## 0` 结论里"选 backend 便宜，建 metadata 不便宜"这句话的具体样子。

## 5. 设计决策与代价

### 决策 1：37 个后端塞进一个枚举，而不是"每个 vendor 一个 if 分支"

- **为什么这么设计**：`AttentionBackendEnum` 把"后端名字"和"后端实现"解耦成一个字符串到类路径的映射（`vllm/v1/attention/backends/registry.py:34`-`131`），新增一个后端只需要在枚举里加一行 + 写一个实现类，不需要去改任何选择逻辑的中心代码；`register_backend()`（`243`-`295`）还允许运行时覆盖已有条目或者往 `CUSTOM` 槽位挂第三方后端，第三方硬件厂商（比如某个 NPU 供应商）可以在不 fork vLLM 的情况下接入。
- **不这样会怎样**：如果后端名字和类路径散落在各处字符串比较里（`if backend_name == "flash_attn": from xxx import Yyy`），每加一个后端就要去改 N 个判断点，且没有一个"这个名字合不合法"的单一真相来源——`_AttentionBackendEnumMeta.__getitem__`（`vllm/v1/attention/backends/registry.py:18`-`31`）能在用户传错名字时给出"有效选项有哪些"的友好报错，字符串比较版本很难做到这一点。
- **什么时候可以不这样**：如果一个引擎从一开始就只打算支持一两种硬件、不考虑第三方扩展（比如某些专为单一硬件设计的推理引擎），维护一个大枚举的收益不明显，直接写死映射表更省事。

### 决策 2：`validate_configuration()` 声明式校验，而不是一张 backend × 轴的显式真值表

这是本篇被要求重点讲清楚的"组合爆炸"控制机制。37 个后端 × 若干量化格式 × 若干模型结构特征（MLA/sink/sliding-window/sparse/mm-prefix/non-causal……）× 多种硬件计算能力，如果要手写一张"哪些组合合法"的真值表，组合数会迅速失控。vLLM 的做法是把这件事**拆成正交轴**：

```python
# vllm/v1/attention/backend.py:263-348（节选，validate_configuration 主体）
def validate_configuration(cls, head_size, dtype, kv_cache_dtype, block_size,
                            use_mla, has_sink, use_sparse, use_mm_prefix, ...):
    invalid_reasons = []
    if not cls.supports_head_size(head_size): invalid_reasons.append(...)
    if not cls.supports_dtype(dtype): invalid_reasons.append(...)
    if use_mla != cls.is_mla(): invalid_reasons.append(...)
    if has_sink and not cls.supports_sink(): invalid_reasons.append(...)
    ...
    combination_reason = cls.supports_combination(...)   # 逃生舱：处理非正交的例外
    if combination_reason is not None: invalid_reasons.append(combination_reason)
    return invalid_reasons
```

每个 backend 类只需要实现十几个独立的能力谓词（`supports_head_size`/`supports_dtype`/`supports_sink`/…），`validate_configuration()` 用固定的 16 条 if 语句逐一检查——这是一个 O(轴数) 的校验，不是 O(backend 数 × 轴数) 的真值表。但正交假设并不总成立：`FlashAttentionBackend.supports_combination()`（`vllm/v1/attention/backends/flash_attn.py:173`-`207`）就是处理"非正交例外"的逃生舱，比如"attention sink 只在 compute capability ≥ 9.0 时支持"、"FP8 KV cache 需要 SM90 上的 FA3 或 SM100 上的 FA4"、"mm_prefix 要求 FA 版本精确解析到 4"——这些是轴与轴之间的联合约束，硬塞进独立谓词会丢信息。
- **为什么这么设计**：谓词化把"这个 backend 支不支持某个特征"变成一个可独立测试、可独立文档化的方法；`docs/mkdocs/gen_files/generate_attention_backends.py` 甚至用 Python `ast` 模块静态解析每个 backend 文件里这些方法的返回值，自动生成 `docs/design/attention_backends.md` 里的能力矩阵表——文档和代码用同一个源头，不会因为有人改了代码却忘了同步文档而失真（`docs/design/attention_backends.md:1`-`6` 的文件头原话）。
- **不这样会怎样**：如果每加一个新特征（比如未来加一种新的量化格式）都要去改一张巨大的 `Dict[Backend, Dict[Feature, bool]]` 真值表，这张表会变成谁都不敢碰的"上帝对象"；`supports_combination` 这类联合约束如果也硬塞进真值表，表的维度会继续爆炸（"head_size × dtype × cc" 这种三元联合条件没法用二维表表示）。
- **什么时候可以不这样**：如果一个引擎的后端数量长期停留在个位数（比如只支持 1-2 种硬件），维护一张手写真值表反而更直观，没必要为了"未来的可扩展性"引入这套谓词框架——vLLM 走到今天这一步，是 37 个后端的规模倒逼出来的设计，不是提前设计好的。

### 决策 3：`backend_per_kind` 按 KV-cache-group 而不是全局唯一

- **为什么这么设计**：混合结构模型（全注意力层 + 滑窗层交替，比如某些长上下文优化模型）本身就会产生多个 `KVCacheSpecKind` 分组（`vllm/v1/kv_cache_interface.py:130`-`140`），每组的最优 backend 可能不同（比如滑窗层想用支持 sliding window 的 backend，全注意力层想用吞吐更高的另一个）。`get_attn_backend()` 在有 `backend_per_kind` 覆盖时，会先算出这一层属于哪个 `KVCacheSpecKind`（`get_attn_spec_kind()`，`vllm/v1/attention/selector.py:62`-`99`），再查表覆盖全局 `backend`（`176`-`183`）。
- **不这样会怎样**：如果只能设一个全局 backend，混合结构模型被迫让所有层共用同一个 backend——要么牺牲滑窗层的特化优化，要么全注意力层被迫迁就滑窗 backend 的能力上限（比如某些 backend 不支持 sliding window，`supports_sliding_window()` 直接返回 `False`）。
- **什么时候可以不这样**：模型结构里所有层的 `KVCacheSpecKind` 都相同（绝大多数纯 decoder-only 模型是这样）时，`backend_per_kind` 是空字典，退化成"只看全局 `backend`"——这也是为什么大多数用户从来没听说过这个配置项。

### 决策 4：Metadata 每步重建，而不是缓存复用

- **为什么这么设计**：`block_table_tensor`/`slot_mapping`/`seq_lens` 这些量在每个 decode step 都会变（新 token 写进新的物理槽位、序列变长），语义上就是"这一步专属"的数据，缓存整份 `AttentionMetadata` 意义不大——除非命中两个特例：CUDA Graph 捕获阶段用固定 shape 的持久化 buffer（`build_for_cudagraph_capture()`，`vllm/v1/attention/backend.py:690`-`701`），或者同一步内多个 hybrid KV-cache-group 共享同一个 builder 类型时只更新 block table（`supports_update_block_table`，`vllm/v1/worker/gpu_model_runner.py:2598`-`2606`）。
- **不这样会怎样**：如果试图跨 step 缓存 metadata，需要在每步开始时先做一次"这份缓存是否还对得上当前 batch"的一致性检查（batch 组成变了、序列长度变了、抢占/驱逐发生了……），这个检查本身的复杂度未必比直接重建更低，还会引入"缓存失效但没被正确识别"这类隐蔽 bug 的风险。
- **什么时候可以不这样**：如果 batch 组成和序列长度在多个连续 step 之间保证不变（比如某些确定性长度的离线批处理场景），`supports_draft_decode_metadata_update`（`vllm/v1/attention/backend.py:599`）这类"部分字段可以原地更新而不必整个重建"的机制就是为这种场景准备的——但这仍然是"更新"而不是"零成本复用"，本质上还是承认了 metadata 是 step 相关的。

### 决策 5：MLA prefill 和 decode 用不同的 backend 选择

- **为什么这么设计**：MLA 的 `forward_mha`（prefill，计算友好）和 `forward_mqa`（decode，数据搬运友好）是两种完全不同的计算模式（见 `## 6` 前的公式对比），没有理由假设同一个 vendor kernel 在两种模式下都最优。`docs/design/attention_backends.md:145`-`165` 文档所述：prefill backend 靠 `-ac.mla_prefill_backend` 单独选（默认自动，"tries FlashAttention first"），decode backend 靠标准的 `-ac.backend` 选（比如 `FLASHMLA`/`TRITON_MLA`）——两个旋钮互不干扰。
- **不这样会怎样**：如果强制 prefill 和 decode 用同一个 backend，要么牺牲 decode 阶段数据搬运友好算法带来的收益，要么牺牲 prefill 阶段计算友好算法的收益，两头不讨好。
- **什么时候可以不这样**：如果一个 MLA 实现的 prefill 和 decode 本来就共享同一套 kernel（没有区分 MHA/MQA 两种模式的必要，比如某些简化实现直接对 decode 也用小 batch 的稠密计算），拆开两个旋钮就是过度设计——但 DeepSeek 系模型的两种模式在计算特征上差异足够大，拆分是有真实收益的（本库推断，依据是官方文档明确把这当成两个独立可调项而不是合并成一个）。

### 决策 6：FlashAttention 走 CMake 编译进自己的 `.so`，FlashInfer 走纯 pip 依赖

- **为什么这么设计**：`cmake/external_projects/vllm_flash_attn.cmake` 里 `GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git`（vLLM 自己维护的 fork，锁定特定 commit），编译进 `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`/`_vllm_fa3_C.abi3.so`/`_vllm_fa4_cutedsl_C.abi3.so` 三个扩展模块（`setup.py:1371`-`1380`）——这样 vLLM 可以在自己的 fork 里打上游还没合并的补丁、控制确切版本，不受上游 FlashAttention 发版节奏约束。FlashInfer 则通过 `requirements/cuda.txt:16`-`18` 声明为普通 pip 包（`flashinfer-python==0.6.17`，额外需要 `flashinfer-cubin` 提供预编译 cubin），用户 `pip install vllm[cuda]` 时跟着装。
- **不这样会怎样**：如果 FlashAttention 也走纯 pip 依赖，vLLM 就没法在不等上游发版的情况下修复紧急 bug 或加自定义特性（历史上 vLLM 确实往 fork 里加过上游没有的功能）；反过来如果 FlashInfer 也走 fork+编译，vLLM 要额外承担一份 CUDA 编译产物的构建和分发成本，而 FlashInfer 团队本身已经在维护独立发版节奏，没必要重复造轮子。
- **什么时候可以不这样**：CUTLASS 走的是第三种模式——`CMakeLists.txt:502` 里 `GIT_REPOSITORY https://github.com/nvidia/cutlass.git` 拉的是 NVIDIA 官方仓库（不是 vLLM fork），但 vLLM 自己在这之上写 kernel（比如 `csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu`，用 CUTLASS 的模板库拼出 MLA decode kernel）——当 vendor 库本身足够稳定、vLLM 只是"使用其模板/API 编写自己的 kernel"而不需要改 vendor 源码时，直接拉官方仓库编译比维护 fork 更省心。

### 决策 7：kernel block size 用"倍数"声明，不要求精确匹配框架 block_size

- **为什么这么设计**：框架层的逻辑 block_size（KV cache 一个块存多少 token）和某个具体 kernel 内部最适配的 tile 大小未必相等——`MultipleOf`（`vllm/v1/attention/backend.py:52`-`57`）让 backend 只声明"我要求 block_size 是 N 的倍数"（比如 `FlashAttentionBackend.get_supported_kernel_block_sizes()` 声明 `[MultipleOf(16)]`），`get_preferred_block_size()`（`149`-`158`）会在用户没有显式指定 `--block-size` 时，自动把默认值调整到 kernel 能接受的最小合法值。这比要求"kernel block size 必须精确等于框架 block_size"多出一个自由度：一个 kernel 的最优 tile 大小可能随硬件、随 head_size 变化，用"倍数约束"而不是"精确值"能让同一个 kernel 兼容多种框架配置。
- **不这样会怎样**：如果每个 kernel 都要求精确匹配某个固定值，用户设置的 `--block-size` 稍微不对就直接选不到这个 backend（校验会在 `supports_block_size()` 那一步就失败），而且不同 kernel 的"精确值"要求大概率互不相同，会进一步加剧 `## 5` 决策 2 讨论的组合爆炸——block_size 这一个轴自己就可能长出几十种离散取值。
- **什么时候可以不这样**：如果一个 backend 的 kernel 实现本身就要求逻辑块大小和物理 tile 大小严格一一对应（比如某些为特定 block_size 手工展开、去掉了循环开销的极致优化 kernel），`get_supported_kernel_block_sizes()` 完全可以返回一个只含离散整数、不用 `MultipleOf` 包装的列表——`supports_block_size()`（`116`-`133`）的实现本身就同时兼容两种写法，这条约束不是框架强加的，是 kernel 作者自己的选择。

## 6. 同位对照：SGLang 在同一位置怎么做

（对应 [[05-SGLang-注意力后端矩阵]]，本节引用的 SGLang 行号已用 `_src/sglang` 逐条核对，SGLang 完整的后端矩阵和调度细节留给对应篇章展开。）

- **后端清单的组织方式不同**：vLLM 用一个平台无关的 `AttentionBackendEnum`（37 个成员，每个成员自带类路径）；SGLang 用一份**扁平字符串列表** `ATTENTION_BACKEND_CHOICES`（`sglang:python/sglang/srt/server_args.py:181`-`209`，本篇实测 22 个条目，含 `"triton"`/`"fa3"`/`"fa4"`/`"flashinfer"`/`"flashmla"`/`"aiter"`/`"hpc_ops"` 等），backend 名字直接作为 CLI 字符串参数值，没有 vLLM 那种"枚举成员 → 动态 import 类路径"的中间层。有趣的是两边都集成了同一个第三方库——腾讯的 [Tencent/hpc-ops](https://github.com/Tencent/hpc-ops)：vLLM 叫它 `HPC_ATTN`（`vllm/v1/attention/backends/registry.py:119`），SGLang 叫它 `"hpc_ops"`，两边都限定"仅 Hopper（SM90）+ 特定 block/page size"。
- **"自动选择"的实现方式不同**：vLLM 是"声明式优先级列表 + 每个 backend 自报能力谓词"（本篇 `## 4`/`## 5`）；SGLang 的默认选择 `_get_default_attn_backend()`（`sglang:python/sglang/srt/server_args.py:5914`-`5986`）是一段**命令式的 if/elif 决策树**，直接在函数体里判断"Hopper + CUDA 12.3 + 非投机解码或 top-1 → `fa3`"、"SM100 支持 + 无 asymmetric KV → `trtllm_mha`"、"HIP → `aiter`" 等具体条件，把"为什么选这个"的理由写成代码注释而不是让每个 backend 类自报能力。这不是谁更优越的问题，是两种可维护性权衡：vLLM 的方式让"加一个新 backend"不用碰选择逻辑，但要求每个 backend 类完整声明能力谓词；SGLang 的方式让"为什么在这种硬件上选这个"一目了然（决策树本身就是文档），但每加一种新硬件条件就要去改这一个中心函数。
- **兼容性修正的组织方式不同**：vLLM 把"这个组合是否合法"收敛成每个 backend 类的一个方法（`validate_configuration`/`supports_combination`）；SGLang 的 `_handle_attention_backend_compatibility()`（`sglang:python/sglang/srt/server_args.py:5988`-`6025`）是一串**具名补丁函数**的顺序调用（`_attention_backend_fa3_fp8_fallback`/`_fa4_page_constraint`/`_mla_backend_page_constraints`……），每个函数处理一种具体的不兼容场景并直接改写配置对象。这种"打补丁"风格的好处是每个特例都能就近写清楚触发条件和处理方式，代价是要理解"选出来的 backend 到底是不是真的能跑"，需要通读整条补丁链，不像 vLLM 那样能只看一个类的方法就知道它支不支持某个特征。
- **decode/prefill 拆分的粒度不同**：SGLang 在 CLI 层面就直接暴露 `decode_attention_backend`/`prefill_attention_backend` 两个独立旋钮（`sglang:python/sglang/srt/server_args.py:1737`-`1754`，注释写明"have priority over --attention-backend"）——这是按**推理阶段**（prefill vs decode）拆分，对所有模型都适用；vLLM 目前只有 MLA 单独拆了 prefill/decode 两个 backend（`## 5` 决策 5），非 MLA 的普通注意力层没有这个粒度的拆分，`backend_per_kind` 拆分的维度是 **KV-cache-group 的结构类型**（全注意力 vs 滑窗），不是推理阶段。
- **多模态编码器独立配置这件事，两边都做了，但暴露方式不同**：vLLM 把 ViT 选择逻辑完全藏在平台类内部（`## 4.6` 的 `get_vit_attn_backend()`），用户没有一个专门的 CLI 旋钮去覆盖它；SGLang 则直接在 `ServerArgs` 上暴露了 `mm_attention_backend` 字段（`sglang:python/sglang/srt/server_args.py:1777`-`1782`，可选 `"sdpa"` 等），是用户可见、可显式覆盖的一个独立配置项。两边都认识到"多模态编码器的注意力约束比解码器简单，值得单独处理"这件事，但 vLLM 选择把这个决定权留在框架内部（自动挑，不给旋钮），SGLang 选择把决定权交给用户——这是"框架自动化程度"上的一个具体分歧点，本库倾向于认为没有绝对优劣（**本库推断**：vLLM 的路径依赖是 ViT 场景的合法后端集合本来就窄，自动选就够用；SGLang 暴露旋钮的收益取决于用户是否真的有调整它的场景，未查证两边各自的实际调整频率）。

## 7. 踩坑与反直觉

1. **`VLLM_ATTENTION_BACKEND` 环境变量已经不存在了，但网上大量教程还在用它。** 本库对整个 `_src/vllm` 做了 `grep -rln "VLLM_ATTENTION_BACKEND"`，零命中——不是某个文件里没写，是这一版代码库里彻底没有这个环境变量了。现在唯一入口是 `--attention-backend` CLI 参数（`vllm/engine/arg_utils.py:976`）或 `AttentionConfig.backend`，二者互斥（`vllm/engine/arg_utils.py:2401`-`2406`）。如果你在这一版 vLLM 上设了这个环境变量，它会被安静地忽略——不会报错，也不会生效，这是最容易踩的一个坑。
2. **`AttentionBackendEnum.NO_ATTENTION` 指向一个不存在的文件。** 枚举里写的是 `"vllm.v1.attention.backends.no_attention.NoAttentionBackend"`（`vllm/v1/attention/backends/registry.py:112`），但本库对整个 `_src/vllm` 搜索 `NoAttentionBackend`/`no_attention.py`，唯一命中就是这一行声明本身——对应的实现文件根本不存在。如果代码路径真的选到了 `NO_ATTENTION`，`resolve_obj_by_qualname` 会在 `import` 时抛 `ModuleNotFoundError`。这条不是"本库读代码读错了"，是这版快照里真实存在的死引用（本库推断：大概率是一个正在废弃或从未完成的预留条目，**未查证**具体是历史遗留还是开发中特性）。
3. **vLLM 的招牌手写 kernel `paged_attention_v1`/`v2` 在 CUDA 路径上已经消失。** 全仓搜索这两个函数名，唯一命中在 `vllm/v1/attention/backends/rocm_aiter_fa.py:1164,1243,1362`，且那里调用的是 `torch.ops.aiter.paged_attention_v1`——这是 AMD AITER 库自己的算子，恰好同名，和 vLLM 论文里那个手写 CUDA kernel没有任何关系。原来的 `csrc/attention/attention_kernels.cu` 已经不在这版代码里，`csrc/attention/` 目录现在只剩 6 个数据类型转换的 `.cuh` 头文件。这个反直觉之处在于：很多人心目中"vLLM = PagedAttention 的手写实现"这个印象，在当前这版代码里已经不成立——PagedAttention 描述的**内存管理思想**（分块、间接寻址）还在，但**计算 kernel 本身**已经全面外包给了 FlashAttention/FlashInfer/Triton。
4. **TPU 支持被整体抽出到了一个外部包。** `vllm/platforms/tpu.py` 只有 20 行，核心逻辑是 `try: from tpu_inference.platforms import TpuPlatform as TpuInferencePlatform`（`9`-`14`），失败就打一条"请安装 tpu_inference"的错误日志然后什么都不做。这意味着本篇 `## 2` 列出的整套 CUDA/ROCm/CPU 平台选择逻辑（`_get_backend_priorities`、`get_attn_backend_cls` 的具体实现）在 `_src/vllm` 这份快照里对 TPU **完全不存在**——TPU 的注意力后端选择逻辑活在一个本库没有 clone 的独立仓库里，**未查证**其内部实现是否遵循同一套 `AttentionBackend` 契约。
5. **选 backend 便宜，建 metadata 不便宜——但这个不对称容易被"selector 有缓存"这句话误导。** `_cached_get_attn_backend`（`vllm/v1/attention/selector.py:194`）和 `_get_backend_priorities`（`vllm/platforms/cuda.py:82`）都套了 `@cache`，容易让人以为"注意力这块都是缓存友好的"；但 `builder.build()` 这个真正每步都要跑的调用完全没有等价的跨步缓存（`## 4.2`/`## 5` 决策 4 已详细展开）——"选型"和"填数据"是两件成本量级完全不同的事，不能因为前者便宜就假设后者也便宜。
6. **`DiffKV` 这个名字第一眼容易读成"差分 KV 压缩"，其实是"K 和 V 的 head dim 不一样"。** `FlashAttentionDiffKVBackend`（`vllm/v1/attention/backends/flash_attn_diffkv.py:35`-`52`）的文档字符串明确写"DiffKV (hdim_qk != hdim_v) requires FA3 or FA4"——这里的 "Diff" 是 "different"（不同）的缩写，不是 "differential"（差分/增量）。同样的命名陷阱在 `TRITON_ATTN_DIFFKV` 上也存在。这条容易在跨团队沟通时产生误解：以为这是又一种类似 TurboQuant 那样的 KV 压缩方案，实际上它和压缩完全无关，只是放宽了"Q/K 头维度必须等于 V 头维度"这个假设。
7. **"vLLM 支持哪些 attention backend" 这个问题在这版代码里没有唯一答案。** `## 4.6` 已经展开：解码器侧的普通/MLA 注意力走 `_get_backend_priorities()` 的完整声明式校验（37 个后端参与竞争），ViT 编码器侧走 `get_supported_vit_attn_backends()`/`get_vit_attn_backend()` 一条独立的、只查三项谓词的简化路径（`vllm/platforms/cuda.py:504`-`558`，固定备选集只有 4 个）。读文档或者读代码时如果不区分"问的是哪一侧"，很容易把两份不同的清单混着引用。

## 8. 可改进点

（以下均为本库基于源码走读的推断，标注证据依据；**未查证**是否已有官方 issue 在跟踪。）

- **`NO_ATTENTION` 这类死引用应该有 CI 兜底。** 目前没有发现任何测试遍历 `AttentionBackendEnum` 全部成员并尝试 `get_class()` 来确认每条都能真正 `import` 成功（`CUSTOM` 除外，它本来就要求先注册）。加一个这样的测试，`registry.py` 里出现死引用会在 CI 阶段就暴露，而不是等到用户真的选中那个从未存在的 backend 才在运行时炸掉。
- **组合校验只在真正构建这一层时才跑，用户可能要等模型加载到一半才发现某个 backend 选不了。** `validate_configuration()`（`vllm/v1/attention/backend.py:263`）是在 `Attention.__init__` 阶段才被调用的（间接经由 `get_attn_backend`），如果一个大模型有几十层，理论上可以在 CLI 参数解析完成、模型权重还没开始下载/加载之前，先用模型配置文件里已知的 `head_size`/`dtype`/结构信息跑一遍"影子校验"，把不兼容的 backend 选择在几秒内报出来，而不是让用户等到显存分配、权重加载这些更耗时的步骤都做完才失败。
- **AST 静态解析生成的文档矩阵可能跟不上运行时才能确定的能力。** `docs/mkdocs/gen_files/generate_attention_backends.py` 用 Python `ast` 模块解析每个 backend 文件的方法体来抽取能力矩阵，这个思路本身很好（文档和代码同源），但 `supports_combination()`（比如 `vllm/v1/attention/backends/flash_attn.py:173`-`207`）里调用的 `flash_attn_supports_kv_cache_dtype()` 这类需要运行时查询已安装 FlashAttention 版本才能确定结果的函数，静态 AST 解析未必能完整还原其语义（**本库推断**，未逐行验证该生成脚本对这类运行时依赖分支的具体处理方式，也未实际跑过这个脚本比对输出）。
- **`MultipleOf` 只声明"倍数关系"，没有声明"倍数选大了会不会浪费显存"。** `get_preferred_block_size()`（`vllm/v1/attention/backend.py:149`-`158`）在用户没显式设置时，只要满足 `supports_block_size(default_block_size)` 就直接用默认值，否则退化到 `min(supported_sizes)`——这中间没有一个信号告诉用户"选了这个 block_size 之后，这个 kernel 内部实际会按多大的物理 tile 对齐、是否存在因为对齐而多分配的显存"。对于显存紧张、想精细控制 KV cache 大小的部署场景，这部分信息目前只能靠读 kernel 源码或者实测显存占用来倒推（**未查证**是否有专门的日志或诊断接口暴露这个信息）。
- **ViT 侧的选择逻辑（`## 4.6`）和解码器侧完全独立维护，容易在新增一个 backend 时漏掉一边。** `get_supported_vit_attn_backends()`（`vllm/platforms/cuda.py:504`-`519`）里的四个候选是手写的固定列表，不是从 `AttentionBackendEnum` 全量派生再过滤出"支持 encoder-only 场景"的子集——如果未来给 37 个后端里的某一个加上了 ViT 场景支持，需要有人记得手动把它加进这个列表，`AttentionBackendEnum` 本身的注册不会自动让它出现在 ViT 候选集里。

## 9. 自测题与延伸阅读

**闭卷自测题**（合上本文，尝试不看源码回答）：

1. `VLLM_ATTENTION_BACKEND` 环境变量在本篇取证的这一版代码里还存在吗？现在选后端有哪几个入口，彼此是什么关系？
2. `AttentionBackend`、`AttentionMetadataBuilder`、`AttentionMetadata` 三者的生命周期分别是"构建期一次"、"每步一次"，还是别的？各自的实例数量（每模型一个/每 attn group 一个/每 step 一个）分别是多少？
3. MLA 的 `forward_mha` 和 `forward_mqa` 分别对应 prefill 还是 decode？为什么 MLA 的 KV cache `num_kv_heads` 恒为 1？
4. `validate_configuration()` 里的 16 条独立 if 检查和 `supports_combination()` 这个"逃生舱"分别解决什么问题？为什么不能只靠前者？
5. FlashAttention 和 FlashInfer 在 vLLM 里的集成方式（源码获取、编译方式）有什么不同？这个不同带来了什么工程上的取舍？
6. `AttentionBackendEnum.NO_ATTENTION` 这个死引用说明了什么？如果你要给这类问题设计一道 CI 检查，会怎么写？
7. `DiffKV` 这个后端名字里的 "Diff" 指的是什么？它和 KV cache 压缩（比如 TurboQuant）有关系吗？
8. ViT 编码器的注意力后端选择和解码器侧的 `get_attn_backend()` 是同一套流程吗？两者在校验的严格程度上有什么区别，为什么可以这样简化？

**延伸阅读（本库内）**：

- [[04-vLLM-KV缓存与前缀缓存]] —— 本篇 `slot_mapping`/`block_table_tensor` 这些字段"逻辑块归谁"的上游决策，在那一篇里展开
- [[06-vLLM-模型执行与CUDA-Graph]] —— `AttentionCGSupport`/`build_for_cudagraph_capture()` 如何与 CUDA Graph 捕获配合，本篇只讲了 metadata builder 侧的接口
- [[07-vLLM-分布式与并行策略]] —— `AttentionImplBase` 里的 `dcp_world_size`/`pcp_world_size`（decode/prefill context parallelism）如何影响 backend 的可用性判断，本篇 `supports_pcp`/`supports_non_causal_dcp` 只带过一笔
