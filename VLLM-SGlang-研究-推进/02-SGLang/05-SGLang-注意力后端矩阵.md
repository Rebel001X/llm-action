# SGLang 注意力后端矩阵与自研算子

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：选型靠决策树，MLA 复用同一契约

## 0. 结论先行

- **注意力子系统是全仓最大的一块表面积**：`_lab/out/struct_map.json` 记录 407 个文件 / 217,907 行，是 `scheduler` 子系统（35,054 行）的 6.2 倍（同一数字 [[01-SGLang-全景与代码地图]] 已引用过，本篇只在需要处复用，不重复展开"attention 最大 ≠ 自研代码量最大"这条已经讲过的坑）。
- **backend 清单不是一个类型化枚举，是一个字符串到工厂函数的字典**：`ATTENTION_BACKENDS`（`python/sglang/srt/layers/attention/attention_registry.py:31`）由 22 次 `@register_attention_backend("名字")` 装饰器调用堆出来（`42`-`517`），对应 21 个具体类（`"nsa"` 是 `"dsa"` 的别名，指向同一个工厂函数）。CLI 层面的 `ATTENTION_BACKEND_CHOICES`（`python/sglang/srt/server_args.py:181`-`209`）多列了一个 `"compressed"`（`dsv4` 的别名），一共 23 个字符串，但走的是两条完全不同的别名机制（`## 7` 展开）。
- **"自动选择"不是一张声明式优先级表，是一段写死在函数体里的 if/elif 决策树**：`_get_default_attn_backend()`（`python/sglang/srt/server_args.py:5914`-`5986`）用大约 30 行硬编码逻辑判断"Hopper+CUDA12.3 上用 fa3""Blackwell 上非对称 KV 用 fa4 否则 trtllm_mha""ROCm 上用 aiter"……每条分支后面跟着一条注释解释"为什么"（比如 `5944`-`5946` 行引用了一个 GitHub issue 号说明 flashinfer 0.6.1 在 Hopper 上有性能回退）。这与后面 `## 6` 要对照的 vLLM 声明式谓词表是两种完全不同的可维护性哲学。
- **静默 fallback 不是个例，是这套系统里反复出现的模式**：本篇实测数出至少 6 处会在不报错的情况下悄悄改写用户的配置或决策——FA3 撞上 `fp8_e5m2` 自动切到 triton（`python/sglang/srt/arg_groups/overrides.py:2287`-`2296`）、7 个 MLA 系 backend 的 `page_size` 不满足内核约束时被自动改写（`python/sglang/srt/arg_groups/overrides.py:2129`-`2196`）、Intel AMX/XPU 硬件检测失败时自动降级到 `torch_native`/`triton`（`python/sglang/srt/arg_groups/overrides.py:2320`-`2341`）、`cutedsl_mla` 只设了 decode 没设 prefill 时自动填 `trtllm_mla`（`python/sglang/srt/arg_groups/overrides.py:2254`-`2285`）、投机解码草稿 backend 不在专属白名单里时静默换成 `triton`/`flashinfer`（`python/sglang/srt/speculative/draft_worker_common.py:39`-`48`，`## 4.7` 详述）、以及一个更隐蔽的——DeepSeek 系模型选了一个没在 `AttentionBackendRegistry` 里注册处理函数的 backend（比如 `cutedsl_mla`、`hpc_ops`、`wave`）时，决定"这一步走 MHA 还是走吸收态 MLA"的策略会静默退化成 triton 的策略，即便真正跑 kernel 的仍是你指定的那个 backend（`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:45`-`46`，本库推断其实际影响范围，`## 7` 详述）。
- **MLA 没有独立的抽象基类**：SGLang 不像本库对照的 vLLM 那样为 MLA 单开一个 `MLAAttentionImpl` 契约，而是让 MLA 系 backend 直接实现同一个 `AttentionBackend` ABC——`FlashInferMLAAttnBackend(AttentionBackend)`（`python/sglang/srt/layers/attention/flashinfer_mla_backend.py:208`）自己就是 MLA 的"契约起点"，`FlashMLABackend`、`CutlassMLABackend`、`TRTLLMMLABackend` 全部靠**类继承**而不是**类型系统**表达"这是一个 MLA 后端"。真正决定某一步该用普通 MHA 还是吸收态 MLA 的逻辑，甚至不在 attention 子系统里，而在模型定义文件 `python/sglang/srt/models/deepseek_v2.py:2000`-`2027` 的 `dispatch_attn_forward_method()`。
- **算子"收进自己仓库"这件事拆开看是三层不同来源，靠一个统一的 `KernelBackend` 枚举编目，不是简单的"vendor 了别人的代码"**：pip 依赖（`flash-attn-4>=4.0.0b18`、`flashinfer_python==0.6.17`，`python/pyproject.toml:35`-`36`）、原样拷贝且保留上游版权头的 vendor 代码（`python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py:1`-`2` 写着 "Copyright (c) 2025, Tri Dao."）、以及挂着 "Copyright 2023-2024 SGLang Team" 的自研代码（`python/sglang/kernels/ops/attention/decode_attention.py:1`）——三者共存于同一个 `python/sglang/kernels/` 命名空间下，用 `KernelSpec`/`KernelBackend`（`python/sglang/kernels/spec.py:29`-`42`）显式区分"这个算子的血统"。

## 1. 它在系统里的位置

注意力后端处在模型层定义（`RadixAttention`/`DeepseekV2AttentionMLA` 这类模块）和物理 kernel 之间，横切构建期和运行期两条流程：

**构建期（进程启动时选一次）**：`ModelRunner.init_attention_backends()`（`python/sglang/srt/model_executor/model_runner.py:942`-`963`）在模型加载完成后调用，内部先 `resolve_attention_backend_strs()` 拿到这一次运行该用的 `(prefill, decode)` 两个 backend 字符串（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:158`-`178`），再 `build_attention_backends()` 真正实例化出 backend 对象（`69`-`143`），存到 `self.attn_backend`。

**运行期（每次 forward 读一次全局上下文）**：与 [[01-SGLang-全景与代码地图]] 提到的整体架构一致，SGLang 用一个**模块级全局变量**而不是 contextvar 承载"当前 forward 用哪个 backend"——`ForwardContext`（`python/sglang/srt/model_executor/forward_context.py:34`-`41`）是个只有一个字段 `attn_backend` 的冻结 dataclass，`ModelRunner._forward_raw` 在每次 forward 前把它发布出去，模型层里所有调用 `get_attn_backend()`（`66`-`67`）拿到的都是**同一个 backend 对象**，不是像 vLLM 那样每层在构建期各自缓存一份。这个差异是 `## 6` 的第一条对照。

上游是 [[02-SGLang-Scheduler事件循环]] 决定的"这一步谁上场"和 [[04-SGLang-内存池与KV布局]] 决定的"KV 数据在物理池里的哪个槽位"；下游是 CUDA Graph 捕获（`init_forward_metadata_in_graph`/`out_graph` 的划分就是为它服务的，`## 3` 展开）。注意力后端自己不决定批多大、KV 放哪，只负责把已经决定好的东西翻译成某个具体 kernel 认识的张量形状，然后调用它——这一点和 vLLM 完全一致。

**边界**：本篇只讲全/滑窗/MLA 这条"softmax 注意力"主线的 backend 矩阵。Mamba2/GDN/Lightning 这类线性注意力有自己独立的 `AttentionBackend` 实现族（`python/sglang/srt/layers/attention/linear/` 目录，`## 4.3` 提到的 `attn_backend_wrapper()` 是它们和主线 backend 的唯一交汇点），本篇不展开其内部算法；`## 3` 已经讲过的 RadixCache 前缀复用逻辑属于 [[03-SGLang-RadixAttention与前缀缓存]] 的范畴，本篇涉及 KV 物理布局的部分只讲"attention backend 怎么读写它"，不讲"哪些块被驱逐"。

## 2. 代码地图（文件 → 职责，带行号）

按"契约定义 → CLI 枚举 → 决策与兼容性 → 构建与编排 → 具体实现 → MLA 岔路 → 算子层 → 多模态独立路径"八层组织：

**契约定义层**：
- `python/sglang/srt/layers/attention/base_attn_backend.py:36`-`308` —— `AttentionBackend` ABC，整个文件只定义这一个类，三件套核心方法（`init_forward_metadata`/`forward_extend`/`forward_decode`）之外，一半篇幅在描述 CUDA Graph capture/replay 的钩子方法（`## 3` 展开）。
- `python/sglang/srt/layers/attention/base_attn_backend.py:22`-`33` —— `SharedReadEnds` 枚举，描述一个 backend 在 CUDA Graph replay 的哪个阶段读完调度器共享数据，四级：`PRE_REPLAY`/`IN_REPLAY`/`POST_REPLAY`/`UNKNOWN`。

**CLI 枚举层**（字符串常量表，不是类型化枚举）：
- `python/sglang/srt/server_args.py:181`-`209` —— `ATTENTION_BACKEND_CHOICES`，本篇实测 23 个字符串（含 `nsa`/`compressed` 两个 deprecated 别名），按源码里的注释分四组：Common / NVIDIA specific / AMD specific / Other platforms。
- `python/sglang/srt/server_args.py:213`-`220` —— `DRAFT_ATTENTION_BACKEND_CHOICES`，6 个值（`flashinfer`/`fa3`/`fa4`/`triton`/`ascend`/`trtllm_mha`），投机解码草稿模型自己的 backend 只能从这里选。
- `python/sglang/srt/server_args.py:222`-`233` —— `CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`，8 个值，能读分块前缀缓存布局的 backend 白名单。
- `python/sglang/srt/server_args.py:235`-`243` —— `DETERMINISTIC_ATTENTION_BACKEND_CHOICES`（5 个）与 `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND`（4 个），确定性推理场景的两层白名单。
- `python/sglang/srt/server_args.py:1737`-`1793` —— `decode_attention_backend`/`prefill_attention_backend`/`mm_attention_backend` 三个 CLI 字段，前两个注释明确写"have priority over --attention-backend"。

**决策与兼容性层**：
- `python/sglang/srt/server_args.py:5914`-`5986` —— `_get_default_attn_backend()`，命令式 if/elif 决策树，本篇 `## 4.1` 逐分支拆开读。
- `python/sglang/srt/server_args.py:5988`-`6127` —— `_handle_attention_backend_compatibility()`，140 行的补丁编排函数，本身不写具体校验逻辑，而是按固定顺序调用一串从 `arg_groups/overrides.py` 导入的"post process pass"。
- `python/sglang/srt/arg_groups/overrides.py:2112`-`2394` —— 9 个具名补丁函数（`_attention_backend_default`/`_mla_backend_page_constraints`/`_mla_kv_cache_dtype_checks`/`_cutedsl_prefill_backend_fill`/`_attention_backend_fa3_fp8_fallback`/`_fa4_page_constraint`/`_attention_backend_platform_fallbacks`/`_intel_xpu_page_constraint`/`_attention_backend_dual_chunk`），`## 5` 决策 1 逐个对应。
- `python/sglang/srt/arg_groups/overrides.py:277`-`290` —— `attention_backends_of()`，"split 字段没设就退回 base backend"这条归一化逻辑的唯一实现，`_resolved_attention_backends()`（`python/sglang/srt/server_args.py:9142`-`9150`）和运行期的 `dispatch_attn_forward_method()` 都调它，是决策结果的单一真相来源。

**构建与编排层**：
- `python/sglang/srt/model_executor/model_runner.py:942`-`963` —— `init_attention_backends()`，构建期入口。
- `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:26`-`37` —— `ResolvedAttentionBackendStr`/`AttentionBackends` 两个 `msgspec.Struct`。
- `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:181`-`235` —— `_build_resolved_backend()`，prefill/decode 字符串不同时组一个 `HybridAttnBackend`（`193`-`228`），相同时走单一 backend（`230`-`234`）。
- `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:251`-`257` —— `_build_full_attention_backend_from_str()`，`ATTENTION_BACKENDS[backend_str](model_runner)` 真正实例化的地方；字符串不在字典里直接 `raise ValueError`（`254`-`255`），这一步**不会**静默。
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py:20`-`52` —— `HybridAttnBackend.__init__` 与 `_select_backend()`（`61`-`80`），组合模式实现 prefill/decode 拆分。

**具体实现层**（21 个类，对应 22 个注册名字，覆盖 `ATTENTION_BACKEND_CHOICES` 四组）。下表以源码枚举为准逐条核对，"滑窗"列的"是"均能指到一个具体字段名，弱信号或没查到确切证据一律标"未查证"，不猜：

| backend 名 | 实现文件:行 | 适用硬件/模型 | MLA | 滑窗(SWA) | spec decode(草稿) |
|---|---|---|---|---|---|
| `triton` | `python/sglang/srt/layers/attention/triton_backend.py:136` | 全平台兜底 | 是（内部分支） | 是（`use_sliding_window_kv_pool`，`188`） | 是（在 `DRAFT_ATTENTION_BACKEND_CHOICES`） |
| `torch_native` | `python/sglang/srt/layers/attention/torch_native_backend.py:19` | 任意设备兜底，禁 CUDA Graph | 未查证 | 未查证 | 未查证 |
| `flex_attention` | `python/sglang/srt/layers/attention/torch_flex_backend.py:17` | CUDA，禁 CUDA Graph | 未查证 | 否（0 处 sliding_window 匹配） | 否（`python/sglang/srt/server_args.py:6023`-`6025` 的 `assert` 显式拒绝） |
| `dsa`（含别名 `nsa`） | `python/sglang/srt/layers/attention/dsa_backend.py:285` | DeepSeek V3.2 稀疏注意力专属 | 特化（非标准 MLA，是 MLA 的稀疏变体） | 未查证 | 否（不在 draft 列表） |
| `dsv4`（含别名 `compressed`） | `python/sglang/srt/layers/attention/deepseek_v4_backend.py:504`（CUDA）/`deepseek_v4_backend_hip_radix.py`（HIP）/`ascend_dsv4_backend.py`（NPU） | DeepSeek V4 压缩注意力专属，三平台三个类 | 特化（DeepSeek V4 自有方案，非标准 MLA） | 未查证 | 否 |
| `cutlass_mla` | `python/sglang/srt/layers/attention/cutlass_mla_backend.py:51` | Blackwell，`page_size` 强制 128 | 是（继承 `FlashInferMLAAttnBackend`） | 未查证 | 否（不在 draft 列表） |
| `fa3` | `python/sglang/srt/layers/attention/flashattention_backend.py:127`（`fa_impl_ver=3`） | SM80 仅非 MLA / SM90 MLA 与非 MLA 均可（`python/sglang/srt/layers/attention/attention_registry.py:214`） | 部分（SM90 可，SM80 拒绝） | 是（`FlashAttentionMetadata.swa_page_table` 字段） | 是（draft 列表） |
| `fa4` | `python/sglang/srt/layers/attention/flashattention_backend.py:127`（`fa_impl_ver=4`） | Blackwell 主打，同一个类 | 是（非对称 KV 场景默认选它，`python/sglang/srt/server_args.py:5956`-`5959`） | 是（同一实现类共享该字段） | 是（draft 列表） |
| `flashinfer` | `python/sglang/srt/layers/attention/flashinfer_backend.py:289`（非 MLA）/`python/sglang/srt/layers/attention/flashinfer_mla_backend.py:208`（MLA） | NVIDIA 通用 | 是（独立类 `FlashInferMLAAttnBackend`） | 是（`use_sliding_window_kv_pool`，`313`） | 是（draft 列表） |
| `flashmla` | `python/sglang/srt/layers/attention/flashmla_backend.py:58` | Hopper/Blackwell，`page_size` 强制 64 | 是（继承 `FlashInferMLAAttnBackend`） | 未查证 | 否 |
| `trtllm_mla` | `python/sglang/srt/layers/attention/trtllm_mla_backend.py:187` | Blackwell，`page_size` 限 32/64 | 是（工厂函数强制，`python/sglang/srt/layers/attention/attention_registry.py:71`-`72`） | 未查证 | 否（不在 draft 列表，但在分块前缀白名单） |
| `cutedsl_mla` | `python/sglang/srt/layers/attention/cutedsl_mla_backend.py:59` | Blackwell，仅 decode（prefill 会被 `assert` 拒绝） | 是（继承 `TRTLLMMLABackend`） | 未查证 | 否 |
| `tokenspeed_mla` | `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py:114` | Blackwell，KV cache 限定 `fp8_e4m3` | 是（继承 `TRTLLMMLABackend`） | 未查证 | 否 |
| `trtllm_mha` | `python/sglang/srt/layers/attention/trtllm_mha_backend.py:103` | Hopper/Blackwell/SM120，工厂函数拒绝 MLA | 否（`python/sglang/srt/layers/attention/attention_registry.py:253`-`254` 显式 raise） | 是（`73` 处 sliding_window 信号） | 是（draft 列表，注释标注"decode-only dense-MQA drafts"） |
| `dual_chunk_flash_attn` | `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py:106` | 长上下文 Dual Chunk 模型专属，检测到即自动开启 | 否 | 特化（自有 chunk 掩码机制，非标准 SWA） | 未查证 |
| `hpc_ops` | `python/sglang/srt/layers/attention/hpc_ops_backend.py:119` | 仅 Hopper，`page_size=64`，工厂函数拒绝 MLA 与 spec decode | 否（`python/sglang/srt/layers/attention/attention_registry.py:262`-`263` 显式 raise） | 未查证（弱信号） | 否（`python/sglang/srt/layers/attention/attention_registry.py:268`-`271` 显式 raise） |
| `aiter` | `python/sglang/srt/layers/attention/aiter_backend.py:133` | AMD ROCm，内部按 `self.use_mla` 分支 | 是（同一个类内部两条路径） | 是（`use_sliding_window_kv_pool`，`234`） | 否（不在 draft 列表） |
| `wave` | `python/sglang/srt/layers/attention/wave_backend.py:40` | AMD（Wave DSL） | 未查证 | 未查证 | 未查证 |
| `intel_amx` | `python/sglang/srt/layers/attention/intel_amx_backend.py:18` | Intel CPU（无 AMX 指令集会被静默降级到 `torch_native`） | 未查证 | 未查证 | 未查证 |
| `ascend` | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py:298`（`hardware_backend/npu/attention/`） | 华为昇腾 NPU | 是（区分 `MHA_NPU`/`MLA_NPU`/`DSA_NPU` 三态） | 未查证 | 是（draft 列表） |
| `intel_xpu` | `python/sglang/srt/layers/attention/xpu_backend.py:28` | Intel XPU（无 XMX 会被静默降级到 `triton`） | 是（`page_size` 按 MLA/非 MLA 分别限定 `[16,32,64,128]`/`[64,128]`） | 是（`59` 处 sliding_window 信号） | 未查证 |

**MLA 岔路层**：
- `python/sglang/srt/models/deepseek_v2.py:1896`-`1928` —— `DeepseekV2AttentionMLA.__init__` 里同时建两个 `RadixAttention` 实例：`attn_mqa`（吸收态，`num_kv_heads=1`，`head_size=kv_lora_rank+qk_rope_head_dim`）与 `attn_mha`（常规多头，`num_kv_heads=num_local_heads`）。
- `python/sglang/srt/models/deepseek_v2.py:2000`-`2027` —— `dispatch_attn_forward_method()`，模型层自己决定这一步走哪条路，不是 backend 决定的。
- `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:35`-`45` —— `AttentionBackendRegistry`，backend 名字到"MHA/MLA 切换策略函数"的映射，只覆盖 13 个名字，未覆盖的名字 `get_handler()`（`44`-`45`）兜底退到 `"triton"` 的策略。

**算子层**：
- `python/sglang/kernels/ops/attention/decode_attention.py:1`-`14`、`python/sglang/kernels/ops/attention/extend_attention.py:1`-`14` —— 版权头写 "Copyright 2023-2024 SGLang Team"，triton 后端用的自研 kernel。
- `python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py:1`-`2` —— 版权头写 "Copyright (c) 2025, Tri Dao." + "Copyright (c) 2026, Colfax International. (modifications)"，5,610 行，vendor 且带二次修改。
- `python/sglang/kernels/spec.py:29`-`42` —— `KernelBackend` 枚举，11 个 provenance 值（`TORCH`/`TRITON`/`JIT`/`AOT`/`CUTE_DSL`/`FLASHINFER`/`AITER`/…），把三种来源统一编目。
- `python/sglang/kernels/registry.py:74`-`76` —— `register_kernel(spec)`，进程级登记入口，只存元数据不触发 import；`python/sglang/kernels/selector.py:38`-`50` —— `select_kernel(op, backend=None)`，按 op id 解析出固定调用路径，多实现时要求显式指定 backend，没有优先级排序或自动 benchmark 择优。
- `python/sglang/kernels/aot/README.md:1`-`13` —— `kernels/aot/` 即历史上独立发布的 `sgl-kernel`（PyPI 包名 `sglang-kernel`，import 路径仍是 `sgl_kernel`），[[01-SGLang-全景与代码地图]] 已引用此文件。
- `python/sglang/srt/layers/attention/swa_mla_fallback/forward.py`、`ops.py` —— MLA×滑窗组合没有原生 kernel，靠一条 Triton/PyTorch 手写 fallback 路径顶上。

**多模态独立路径**：
- `python/sglang/srt/layers/attention/vision.py:1003`-`1017` —— `QKV_BACKEND_IMPL`（9 个 ViT 专用 backend 名，如 `triton_attn`/`aiter_attn`，命名空间与主 backend 表不同）。
- `python/sglang/srt/layers/attention/vision.py:1222`-`1260` —— `_determine_attention_backend()`，ViT 侧独立的平台默认值判断。

以上引用已覆盖 CLI、决策、构建、执行、MLA 分流、算子来源、多模态七个层次，超过 40 条 `文件:行`，均已用 `Read`/`grep -n` 逐条核对。

## 3. 核心数据结构

**`AttentionBackend`**（`python/sglang/srt/layers/attention/base_attn_backend.py:36`）——抽象基类，docstring 明确写了"Forward-data init contract (3 methods)"：`init_forward_metadata`（eager 入口，默认是 `_out_graph` + `_in_graph` 的组合，`65`-`71`）、`init_forward_metadata_out_graph`（graph capture/replay **之外**跑的动态形状准备，`73`-`93`）、`init_forward_metadata_in_graph`（graph capture **内部**跑的静态形状 GPU op，lint 契约明确禁止在里面调 `.item()`/`.cpu()`/`.tolist()`，`95`-`107`）。旧版本的 `init_forward_metadata_capture_cuda_graph`/`init_forward_metadata_replay_cuda_graph` 已经从 ABC 里删除（`53`-`56` 的类 docstring 原话："fully deprecated and removed from the ABC"）——这是本篇取证版本里一次已完成的接口迁移，不是本库的推测。

**`ATTENTION_BACKENDS`**（`python/sglang/srt/layers/attention/attention_registry.py:31`）——一个普通 `dict[str, Callable]`，key 是 CLI 字符串，value 是接受 `runner: ModelRunner` 返回 `AttentionBackend` 实例的工厂函数。这与 vLLM 的 `AttentionBackendEnum`（`vllm:vllm/v1/attention/backends/registry.py:34`）有一个关键差异：vLLM 存的是"类路径字符串"，实例化逻辑统一在基类外部；SGLang 存的是"工厂函数本身"，每个工厂函数可以在实例化前后插入任意平台特定逻辑（比如 `create_dsv4_backend` 按 NPU/HIP/CUDA 三分支选不同的类，`python/sglang/srt/layers/attention/attention_registry.py:151`-`174`），代价是这层间接性让"这个名字最终会变成哪个类"不能只看一张表，得跳进函数体里读。

**`ResolvedAttentionBackendStr` / `AttentionBackends`**（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:26`-`37`）——两个 `msgspec.Struct`（frozen，仓库风格约定 `.claude/rules/no-dataclasses.md` 要求新代码用 `msgspec.Struct` 而非 `@dataclass`）。前者只装 `(prefill, decode, is_draft_override)` 三个字符串/布尔字段；后者是构建结果的容器，除了最终生效的 `attn_backend` 外还带 `decode_attn_backend_group: list[AttentionBackend]`——这个列表只在 PD-multiplexing（`get_disagg().enable_pdmux`）场景下有多于一个元素（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:85`-`96`），普通场景下是空列表。

**`HybridAttnBackend`**（`python/sglang/srt/layers/attention/hybrid_attn_backend.py:20`）——不是一个新的算子实现，是一个**组合器**：持有 `prefill_backend`/`decode_backend` 两个完整的 `AttentionBackend` 实例，`_select_backend()`（`61`-`80`）按 `ForwardMode` 决定这一步该问谁——`decode_or_idle` 恒定问 `decode_backend`，`extend`（prefill）恒定问 `prefill_backend`，`target_verify`（投机解码验证步）看 `speculative_attention_mode` 配置动态选。这个类自己也实现了完整的 `AttentionBackend` 接口（`init_forward_metadata`/`forward_extend`/`forward_decode` 等），对上层模型代码完全透明——模型层拿到的永远是"一个 backend"，感知不到背后其实是两个。

**`KernelBackend`**（`python/sglang/kernels/spec.py:29`-`42`）——`str, Enum`，11 个值：`TORCH`/`TORCH_COMPILE`/`TRITON`/`JIT`/`AOT`/`CUTE_DSL`/`FLYDSL`/`FLASHINFER`/`DEEPGEMM`/`AITER`/`TORCH_NPU`。类文档字符串明确区分"backend"（怎么构建/来自哪里）与"device"（跑在什么硬件上）两个正交维度——同一个 `JIT` provenance 既能编给 CUDA 也能编给 ROCm，设备支持范围由另一个 `CapabilityRequirement` 描述（`python/sglang/kernels/spec.py:12`-`16`）。

**`AttnForwardMethod`**（`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py:4`-`42`）——`IntEnum`，12 个成员：`MHA`/`MLA`/`MHA_CHUNKED_KV`/`MHA_ONE_SHOT`/`MLA_FUSED_ROPE_ROCM`/`MLA_FUSED_ROPE_CPU`/`MHA_NPU`/`MLA_NPU`/`DSA_NPU`/`MHA_ROCM`/`MHA_ONE_SHOT_ROCM`/`MLA_ROCM`。这是 DeepSeek 系模型自己的"这一步该怎么算"决策结果类型，比 backend 层面的"用哪个 kernel 库"更细一级——同一个 `trtllm_mla` backend 在不同 forward 步骤上可能对应 `MLA` 或 `MHA_CHUNKED_KV` 两种完全不同的计算路径。

**`AttentionBackendRegistry`**（`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:35`-`45`）——一个极简类属性字典 `_handlers: dict[str, Callable]`，`register()` 写、`get_handler()` 读，`get_handler()` 找不到时兜底成 `_handlers.get("triton")`（`44`-`45`）而不是抛异常——这行代码是 `## 0`/`## 7` 反复提到的那个静默 fallback 的确切位置，完整定义只有 11 行：

```python
# python/sglang/srt/models/deepseek_common/attention_backend_handler.py:35-45
class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))
```

**`TboAttnBackend`**（`python/sglang/srt/layers/attention/tbo_backend.py:17`）——第三个"组合器"类，和 `HybridAttnBackend`（按 prefill/decode 组合）、`HybridLinearAttnBackend`（按层 id 组合全注意力/线性注意力）并列。它不组合两个不同种类的 backend，而是用同一个工厂函数 `init_new()`（`31`）造出两份**同类型**的 backend 实例，服务于 Two-Batch-Overlap（TBO，把一个大批次拆两半、通信和计算错峰）——`## 2` 提到的 `build_attention_backends()` 只在 `get_exec().overlap.enable_two_batch_overlap` 打开且不是草稿 worker 时才会走这条分支（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:98`-`107`）。三个组合器类的共同点：都完整实现 `AttentionBackend` 接口，对上层模型代码透明，模型层永远只看到"一个 backend"——这是 `## 5` 决策 2/3 反复出现的同一个设计模式在三个不同场景下的复用。

## 4. 主流程走读

### 4.1 CLI 解析：命令式决策树，不是优先级表

`_get_default_attn_backend()`（`python/sglang/srt/server_args.py:5914`-`5986`）在 `attention_backend`/`decode_attention_backend`/`prefill_attention_backend` 三个字段都没设时才会被调用（由 `_attention_backend_default()` 触发，`python/sglang/srt/arg_groups/overrides.py:2112`-`2125`）。它的 docstring 直接写了决策摘要：

```text
1. Models with MHA Architecture (e.g: Llama, QWen)
   1.1 Hopper 上默认 fa3（除非 spec decode 且 topk>1 或 page_size>1）
   1.2 SM100/SM103 上默认 trtllm_mha（不支持 SM120，SM120 落到 flashinfer）
   1.3 其余情况 flashinfer 可用则用 flashinfer，否则 triton
2. Models with MLA Architecture
   2.1 Hopper 上 fa3
   2.2 Blackwell 上 flashinfer
   2.3 其余 triton
```

函数体本身是一串 `if/elif`，不是遍历一张表：`is_hopper_with_cuda_12_3()`、`is_sm100_supported()`、`is_hip()`、`is_mps()` 这些平台探测函数被直接内联进条件判断（`5938`-`5986`，非 MLA 与 MLA 两个分支体）。一个容易漏看的细节：非 MLA 的 SM100 分支里，如果 `model_config.has_asymmetric_kv` 为真会选 `fa4` 而不是 `trtllm_mha`（`5956`-`5959`，注释解释"trtllm_mha requires equal K/V row widths; fa4 carries v_head_dim through"）——这是一条模型结构（K/V 头宽是否对称）反过来否决默认硬件优先级的分支，和 `## 5` 决策 1 要讲的"白名单+补丁链"风格一致：例外情况就地加一个 if，不是维护一张正交能力矩阵。

Out-of-tree 平台（`current_platform.is_out_of_tree()`）在函数最开头就整体绕过这套逻辑，改问平台自己的 `get_default_attention_backend()`（`5929`-`5930`）——这条与 `## 6` 要对照的 vLLM TPU 平台委托模式是同一个思路。

### 4.2 兼容性补丁链：9 个函数依次改写同一份配置

`_handle_attention_backend_compatibility()`（`python/sglang/srt/server_args.py:5988`-`6127`）不写具体校验逻辑，而是按固定顺序调用 `run_post_process_pass(self, fn)`（`python/sglang/srt/arg_groups/overrides.py:179`-`210`），依次跑：

1. `_attention_backend_default` —— 填充默认值（`## 4.1`）；如果 `prefill_attention_backend == decode_attention_backend` 且都设了，反向覆盖 `attention_backend`（`python/sglang/srt/arg_groups/overrides.py:2113`-`2117`）。
2. `_mla_backend_page_constraints` —— **7 个** MLA/TRTLLM 系 backend（`flashmla`/`cutlass_mla`/`trtllm_mla`/`tokenspeed_mla`/`cutedsl_mla`/`trtllm_mha`/`hpc_ops`）各自的 `page_size` 内核约束，不满足就打 warning 自动改写（`python/sglang/srt/arg_groups/overrides.py:2129`-`2196`）。
3. `_mla_kv_cache_dtype_checks` —— trtllm_mla/tokenspeed_mla 的 `kv_cache_dtype` 只读校验，不满足**直接抛异常**（`python/sglang/srt/arg_groups/overrides.py:2209`-`2233`，这是本节唯一不静默的一步）。
4. `_cutedsl_prefill_backend_fill` —— `cutedsl_mla` 只支持 decode，`prefill_attention_backend` 没设时自动填 `trtllm_mla`（`python/sglang/srt/arg_groups/overrides.py:2254`-`2285`）。
5. `_attention_backend_fa3_fp8_fallback` —— `fa3` + `fp8_e5m2` 自动切 `triton`（`python/sglang/srt/arg_groups/overrides.py:2287`-`2296`）。
6. `_fa4_page_constraint` —— 非 MLA 模型 + SM100 + `fa4` 强制 `page_size=128`（`python/sglang/srt/arg_groups/overrides.py:2298`-`2318`）。
7. `_attention_backend_platform_fallbacks` —— `intel_amx` 缺 AMX 指令集自动切 `torch_native`，`intel_xpu` 缺 XMX 自动切 `triton`（`python/sglang/srt/arg_groups/overrides.py:2320`-`2341`）。
8. `_intel_xpu_page_constraint` —— Intel XPU 的 `page_size` 约束（`python/sglang/srt/arg_groups/overrides.py:2343`-`2359`）。
9. `_attention_backend_dual_chunk` —— 模型带 `dual_chunk_attention_config` 时自动切 `dual_chunk_flash_attn`，如果用户手动指定了别的 backend 则**抛异常**（`python/sglang/srt/arg_groups/overrides.py:2361`-`2374`）。

九步里有七步是"打 warning 后静默改写"，只有两步（第 3、9 步）选择直接抛异常。这个比例本身就是 `## 5` 决策 5 要讲的设计取舍：绝大多数不兼容组合被当作"用户大概率不关心具体数值、只关心能不能跑起来"来处理。

第 2 步 `_mla_backend_page_constraints`（`python/sglang/srt/arg_groups/overrides.py:2129`-`2196`）是九步里体量最大的一个，值得看一眼它的重复结构——7 个 backend 各自一段近乎相同的 if 块，逐个改写同一个局部变量 `page_size`，最后统一 diff 出结果：

```python
# python/sglang/srt/arg_groups/overrides.py:2129-2196（节选，仅保留结构骨架）
def _mla_backend_page_constraints(view):
    page_size = view.page_size
    if view.attention_backend == "flashmla" or view.decode_attention_backend == "flashmla":
        logger.warning("FlashMLA only supports a page_size of 64, ...")
        page_size = 64
    if view.attention_backend == "cutlass_mla" or view.decode_attention_backend == "cutlass_mla":
        logger.warning("Cutlass MLA only supports a page_size of 128, ...")
        page_size = 128
    if view.attention_backend == "trtllm_mla" or view.decode_attention_backend == "trtllm_mla":
        if page_size not in [32, 64]:
            page_size = 64
    # ... tokenspeed_mla / cutedsl_mla / trtllm_mha / hpc_ops 各一段同构 if 块
    if page_size != view.page_size:
        return {"page_size": page_size}
    return {}
```

这段代码本身就是"白名单+补丁链"路线（`## 5` 决策 1）的一个缩影：正确性靠 7 段几乎重复的 if 块保证，不是靠一张"backend → 合法 page_size 集合"的查找表——多写几行换来的是每一段判断条件都可以独立读懂、独立修改，不需要理解一个通用的查表机制。

### 4.3 构建期：字符串变成对象，prefill/decode 可能分叉成两个实例

`build_attention_backends()`（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:69`-`143`）拿到 `## 4.2` resolve 出的 `(prefill, decode)` 字符串对，交给 `_build_resolved_backend()`（`181`-`235`）：

```python
# attention_backend_setup.py:181-235（节选，逻辑简化）
if resolved.is_draft_override:
    attn_backend = _build_backend_from_str(..., resolved.prefill, ...)   # 草稿模型：单一 backend
elif resolved.decode != resolved.prefill:
    attn_backend = attn_backend_wrapper(model_runner, HybridAttnBackend(
        model_runner=model_runner,
        decode_backend=_build_full_attention_backend_from_str(..., resolved.decode, ...),
        prefill_backend=_build_full_attention_backend_from_str(..., resolved.prefill, ...),
    ))
else:
    attn_backend = _build_backend_from_str(..., resolved.prefill, ...)   # 相同：单一 backend
```

`_build_full_attention_backend_from_str()`（`251`-`257`）是真正查 `ATTENTION_BACKENDS[backend_str]` 的地方——字符串不在表里直接 `raise ValueError(f"Invalid attention backend: {backend_str}")`（`254`-`255`），这一步没有静默降级。`attn_backend_wrapper()`（`python/sglang/srt/layers/attention/attention_registry.py:324`-`510`）在拿到基础 backend 之后还会按模型结构再包一层：如果是 hybrid GDN / Mamba2 / MiniMax 稀疏这类混合线性注意力模型，会把刚构建好的 full-attention backend 和一个独立构建的线性注意力 backend 一起塞进 `HybridLinearAttnBackend`（`460`-`508`）——这意味着**同一个模型里，不同层可能对应不同的 backend 组合**，`full_attention_layer_ids` 决定哪些层用哪一个（`503`-`508`）。

最终 `attn_backend.prefill_attention_backend_str`/`decode_attention_backend_str` 两个字符串字段被显式写回 backend 对象自身（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:134`-`135`）——这两个字符串后面在 `## 4.5` 的 MLA 分流逻辑里会被 `dispatch_attn_forward_method()` 读回去，是决策阶段和执行阶段之间少有的、被显式传递的桥。

### 4.4 运行期：全局 forward context，三件套调用顺序

每次 forward，模型层调 `RadixAttention.forward()`（`python/sglang/srt/layers/radix_attention.py:150`），内部调 `get_attn_backend().forward(q, k, v, self, forward_batch, ...)`（`182`、`279` 等多处调用点）。`get_attn_backend()`（`python/sglang/srt/model_executor/forward_context.py:66`-`67`）读的是模块级全局变量 `_current`（`43`），由 `ModelRunner` 在每次 `_forward_raw` 开始时用 `forward_context()` 上下文管理器发布（`78`-`83`）——所有层，无论多少个 `RadixAttention` 实例，这一次 forward 里读到的都是**同一个** backend 对象引用。

`AttentionBackend.forward()`（`python/sglang/srt/layers/attention/base_attn_backend.py:216`-`258`）按 `forward_batch.forward_mode` 三路分发：`is_idle()` 直接返回空、`is_decode()` 调 `forward_decode()`、其余（含 `is_extend()`）调 `forward_extend()`——`forward_mixed()` 只在 NPU 平台的混合模式下才会被调用（`239`-`248`）。metadata 的准备发生在 `forward()` 调用之前，由调用方（`ModelRunner`/`HybridAttnBackend`）先调 `init_forward_metadata()` 或它的 out_graph/in_graph 两段式版本——这与 vLLM "每步重建 metadata、backend 实例长驻"的模式在生命周期上是一致的，差异在于 SGLang 把 CUDA Graph capture/replay 的静态/动态形状拆分做成了**两个独立方法**（`init_forward_metadata_out_graph`/`init_forward_metadata_in_graph`）而不是一个 `build_for_cudagraph_capture()`，这是 `## 6` 的第二条对照。

如果这一步的 backend 恰好是 `## 3`/`## 5` 决策 3 提到的 `HybridAttnBackend`，`forward()`/`init_forward_metadata()` 实际转发给的子 backend 由同一个私有方法决定，三处调用（`forward`/`init_forward_metadata`/`init_forward_metadata_out_graph`）全部复用它，保证"这一步该问谁"这件事只判断一次：

```python
# python/sglang/srt/layers/attention/hybrid_attn_backend.py:61-80（节选）
def _select_backend(self, forward_mode: ForwardMode) -> AttentionBackend:
    if forward_mode.is_decode_or_idle():
        return self.decode_backend
    elif forward_mode.is_target_verify():
        return (
            self.decode_backend if self.spec_attn_is_decode
            else self.prefill_backend
        )
    else:
        return self.prefill_backend
```

`target_verify`（投机解码验证步）是唯一一个不由 `forward_mode` 单独决定的分支——还要看 `speculative_attention_mode` 这个独立配置项（`## 4.7` 提到的投机解码路径在这里和主选型逻辑短暂交汇了一次）。

### 4.5 MLA 的岔路：模型层决定走哪条路，backend 层只管执行

以 DeepSeek 系模型为例，`DeepseekV2AttentionMLA` 在构造时同时建好两个 `RadixAttention`：`attn_mqa`（吸收态，`kv_lora_rank+qk_rope_head_dim=576` 维，`num_kv_heads=1`，`python/sglang/srt/models/deepseek_v2.py:1896`-`1905`）和 `attn_mha`（常规多头，`python/sglang/srt/models/deepseek_v2.py:1919`-`1928`）。每次 forward，`dispatch_attn_forward_method()`（`2000`-`2027`）先从 `get_attn_backend()` 读出这一步该用的 `prefill_attention_backend_str`/`decode_attention_backend_str`（读的正是 `## 4.3` 写回的那两个字段），再查 `AttentionBackendRegistry.get_handler(attention_backend)`（`2026`）拿到一个"这个 backend 在这一步该返回 `MHA` 还是 `MLA`"的判断函数。

以 `trtllm_mla` 为例，`handle_attention_trtllm_mla` 最终落到 `_handle_attention_backend()`（`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:97`-`131` 附近）的通用逻辑：CUDA Graph capture 中恒定用 `MLA`（吸收态，因为 MHA 的形状随 KV 长度变化，无法捕获，`102`-`104`）；否则看这一步的"待计算前缀长度"是否超过阈值——短则用一次性 `MHA_ONE_SHOT`（仅 `fa3`/`flashinfer`/`flashmla` 支持，`MHA_ONE_SHOT_SUPPORTED_BACKENDS`，`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:16`），长则退回吸收态 `MLA`。选完 `AttnForwardMethod` 之后，模型层调用 `self.attn_mha.forward(...)` 或 `self.attn_mqa.forward(...)`——这两个调用最终还是落回 `## 4.4` 那套 `get_attn_backend().forward_extend/forward_decode`，只是 `q`/`k`/`v` 的形状（是否吸收）已经在模型层被决定好了。

这条链路解释了"MLA 为什么不能复用普通 paged 路径"：吸收态下，参与 attention 计算的 K/V 不是每个 head 各自解压出来的张量，而是**所有 head 共享的同一个压缩 latent 向量**（`num_kv_heads=1`），物理 KV cache 的存储格式因此和常规 MHA 完全不同——一个假设 `num_kv_heads` 与 query head 数成固定比例（GQA/MQA 常规配置）的通用 paged attention kernel 没有办法直接处理"1 个 KV head，但这个 head 的维度是 576"这种极端形状，需要专门理解"吸收"这个代数变换（`W_UK` 折进 Q 投影、`W_UV` 折进 O 投影）的 kernel 才行。滑窗和 MLA 的组合是这条约束的一个极端后果：常规 MLA kernel 不处理滑窗掩码，滑窗和吸收态 MLA 同时出现时只能退到 `swa_mla_fallback/` 下手写的 Triton/PyTorch 路径（`python/sglang/srt/layers/attention/swa_mla_fallback/forward.py`）——这不是某个 backend 的疏忽，是"吸收态 MLA 的 KV 物理布局"和"滑窗需要按位置掩码"这两个约束天然难以在同一个高性能 kernel 里同时满足的直接证据。

### 4.6 多模态编码器：另一张表，另一套默认值

ViT 编码器（`VisionAttention`，`python/sglang/srt/layers/attention/vision.py`）完全不走 `## 4.1`-`## 4.5` 这套流程。它有自己的 backend 名字空间 `QKV_BACKEND_IMPL`（`1003`-`1012`，9 个值：`triton_attn`/`sdpa`/`fa3`/`fa4`/`flashinfer_cudnn`/`ascend_attn`/`aiter_attn`/`amx_attn`/`xpu_attn`——注意 `triton_attn` 不是 `triton`，`aiter_attn` 不是 `aiter`，与主 backend 表的字符串不通用），自己的默认值判断 `_determine_attention_backend()`（`1222`-`1260`，docstring 直接写"Platform defaults: Hopper→fa3, Blackwell→fa4, 其余 CUDA→triton_attn, NPU→ascend_attn"）。用户可以用 `--mm-attention-backend` 覆盖（`python/sglang/srt/server_args.py:1777`-`1793`），但这个覆盖只对 ViT 侧生效，对主干 LLM 的 attention backend 没有任何影响——两套配置互不干扰，也互不感知对方的存在。

### 4.7 投机解码草稿模型：第三条独立的选型路径，带自己的静默兜底

草稿模型（EAGLE/DFLASH 等算法用的小模型）的 attention backend **不是**从目标模型的 `(prefill, decode)` 二元组直接继承来的，走的是第三条选型路径。`resolve_attention_backend_strs()`（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:158`-`178`）一开始就检查 `is_draft_worker and draft_attn_backend`（`169`）：如果这是一个草稿 `ModelRunner` 且它自己的 `draft_attention_backend` 属性非空，直接返回 `ResolvedAttentionBackendStr(prefill=draft_attn_backend, decode=draft_attn_backend, is_draft_override=True)`（`172`-`176`）——`## 4.3` 提到的 `_build_resolved_backend()` 一看到 `is_draft_override` 为真就直接走单一 backend 分支（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:187`-`192`），**完全跳过** `HybridAttnBackend` 组合逻辑，即便目标模型本身配置了不同的 prefill/decode backend。

`ModelRunner.draft_attention_backend` 这个属性本身的值，由 `resolve_draft_attention_backend()`（`python/sglang/srt/model_executor/model_runner.py:267`-`282`）算出来：非草稿 runner 直接返回 `None`；是草稿 runner 则优先用构造参数传入的值，其次退回 `server_args.speculative_draft_attention_backend`（`282`）。而 `speculative_draft_attention_backend` 这个 CLI 字段本身（`python/sglang/srt/server_args.py:2224`-`2227`）**没有** `choices=DRAFT_ATTENTION_BACKEND_CHOICES` 约束——`DRAFT_ATTENTION_BACKEND_CHOICES` 这张白名单实际生效的地方在另一个独立函数 `_resolve_draft_attention_backend_fallback()`（`python/sglang/srt/speculative/draft_worker_common.py:31`-`49`）里：先看 `speculative_draft_attention_backend` 是否设置，没设置就退回目标模型自己的 `attention_backend`（`35`），如果两者都是 `None` 才用硬编码默认值（`"triton"`（HIP）或 `"flashinfer"`（其他），`36`-`37`）；算出候选值后检查是否在 `DRAFT_ATTENTION_BACKEND_CHOICES` 白名单里（`39`），**不在**就打一条 warning 静默换成同一个硬编码默认值（`40`-`48`）——这是本篇找到的第 6 处静默 fallback，`## 0`/`## 7` 统一计数时把它算进去。

这条路径解释了两件事：一是为什么 `DRAFT_ATTENTION_BACKEND_CHOICES` 只有 6 个值（`## 2` 已列出）——它不是"所有 backend 的子集"，而是"经过验证能在草稿模型场景下正确工作"的一个独立小名单；二是为什么 `HybridAttnBackend`（`## 5` 决策 3）的复杂度不会渗透进投机解码路径——草稿模型的 attention backend 选型，从 CLI 到实例化，是一条与目标模型平行、几乎不共享代码的独立管线，只在"没显式设置草稿 backend 时退回目标模型 backend 作为候选之一"这一处发生了浅层耦合。

`_resolve_draft_attention_backend_fallback()` 的完整逻辑摘录如下（三层候选 + 一次白名单检查 + 一次静默替换）：

```python
# python/sglang/srt/speculative/draft_worker_common.py:31-49（节选）
def _resolve_draft_attention_backend_fallback(*, server_args, algo_label):
    draft_backend = server_args.speculative_draft_attention_backend  # 候选 1：用户显式指定
    if draft_backend is None:
        draft_backend, _ = server_args.get_attention_backends()      # 候选 2：目标模型自己的 prefill backend
    if draft_backend is None:
        return "triton" if torch.version.hip else "flashinfer"       # 候选 3：硬编码默认值
    if draft_backend not in DRAFT_ATTENTION_BACKEND_CHOICES:          # 白名单检查
        fallback = "triton" if torch.version.hip else "flashinfer"
        logger.warning(...)                                          # 只打日志，不报错
        return fallback                                              # 静默替换
    return draft_backend
```

三层候选叠一次白名单检查——这是本篇找到的选型逻辑里嵌套层数最多的一处，也是最容易在读代码时漏看"候选 2 其实是目标模型的 backend"这一步的地方。

### 4.8 端到端走一遍：DeepSeek 系模型在 Blackwell 上不设任何 backend 参数会发生什么

把 `## 4.1`-`## 4.5` 串起来，假设部署一个 `use_mla_backend=True` 的 DeepSeek 系模型，SM100（Blackwell）硬件，用户没有传 `--attention-backend`/`--decode-attention-backend`/`--prefill-attention-backend` 中的任何一个：

1. **CLI 解析阶段**：`_attention_backend_default()`（`python/sglang/srt/arg_groups/overrides.py:2112`-`2125`）发现三个字段都是 `None`，调用 `_get_default_attn_backend(use_mla_backend=True, model_config)`（`python/sglang/srt/server_args.py:5914`）。
2. `_get_default_attn_backend` 走 MLA 分支（`5972`-`5986`）：`is_hopper_with_cuda_12_3()` 为假（这是 Blackwell 不是 Hopper），`is_sm100_supported()` 为真——命中 `elif is_sm100_supported(): return "flashinfer"`（`5974`-`5975`）。三个字段被统一填成 `"flashinfer"`。
3. **兼容性补丁链**（`## 4.2`）依次跑：`_mla_backend_page_constraints` 检查 `attention_backend == "flashmla"/"cutlass_mla"/"trtllm_mla"/...`，`"flashinfer"` 都不在这几个条件里，`page_size` 不受影响；`_mla_kv_cache_dtype_checks`、`_cutedsl_prefill_backend_fill`、`_fa4_page_constraint` 同理全部跳过（条件都是别的 backend 名字）；9 步补丁链对这个具体组合实际只是空跑一遍。
4. **构建期**（`## 4.3`）：`resolved.prefill == resolved.decode == "flashinfer"`，`_build_resolved_backend()` 走"相同"分支，不会构造 `HybridAttnBackend`；`create_flashinfer_backend(runner)`（`python/sglang/srt/layers/attention/attention_registry.py:42`-`66`）看到 `runner.use_mla_backend` 为真，实例化的是 `FlashInferMLAAttnBackend`（`python/sglang/srt/layers/attention/flashinfer_mla_backend.py:208`）而不是普通的 `FlashInferAttnBackend`。
5. **模型层**：`DeepseekV2AttentionMLA.dispatch_attn_forward_method()`（`## 4.5`）读到 `prefill_attention_backend_str == decode_attention_backend_str == "flashinfer"`，`AttentionBackendRegistry.get_handler("flashinfer")` 命中已注册的 `handle_attention_flashinfer`（`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:234`）——这是 `## 7` 提到的表覆盖问题在这个具体场景下**不会**触发（`"flashinfer"` 在 13 个已注册名字里）。
6. **运行期**：非 CUDA Graph capture 的短前缀 extend 步可能选中 `AttnForwardMethod.MHA_ONE_SHOT`（`"flashinfer"` 在 `MHA_ONE_SHOT_SUPPORTED_BACKENDS` 里，`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:16`），调用 `self.attn_mha.forward(...)`；decode 步固定用 `AttnForwardMethod.MLA`，调用 `self.attn_mqa.forward(...)`——两者最终都落回 `get_attn_backend().forward_extend`/`forward_decode`（`## 4.4`），backend 对象自始至终是同一个 `FlashInferMLAAttnBackend` 实例。

这条链路里，"用户什么都没传"到"最终跑起来的具体 kernel 调用序列"之间，实际经过了 CLI 决策树、9 步兼容性补丁、构建期 backend 实例化、模型层 MHA/MLA 分流四道独立的选择，本篇 `## 5`/`## 6` 讨论的每一处设计决策，在这一条具体路径上都至少生效了一次。

## 5. 设计决策与代价

### 决策 1：用白名单常量 + 命令式决策树 + 补丁函数链控制组合爆炸，而不是每个 backend 类自报能力谓词

这是本篇被要求重点讲清楚的机制。22 个 backend × 若干量化格式（`bf16`/`fp8_e4m3`/`fp8_e5m2`/`fp4_e2m1`/`mxfp8`…）× 模型结构特征（MLA/滑窗/spec decode/dual-chunk/asymmetric-KV…）× 硬件平台（NVIDIA 多代/AMD/Intel/NPU），如果要穷举合法组合，数量同样会失控。SGLang 的做法和 vLLM 完全不同：**没有一个类似 `validate_configuration()` 的统一入口**，控制机制拆成了三层松散耦合的东西——`python/sglang/srt/server_args.py:181`-`243` 的一组静态字符串白名单常量（`ATTENTION_BACKEND_CHOICES`/`DRAFT_...`/`CHUNKED_PREFIX_CACHE_SUPPORTED_...`/`DETERMINISTIC_...`，声明"哪些名字在这个场景下允许出现"）、`_get_default_attn_backend()` 里硬编码的 if/elif 决策树（声明"没指定时该选哪个"）、以及 `_handle_attention_backend_compatibility()` 编排的 9 个补丁函数（声明"选完之后哪些组合需要被纠正"）。

- **为什么这么设计**：这三层各自都很薄、很容易读——想知道"投机解码草稿模型能用哪些 backend"，直接看 `DRAFT_ATTENTION_BACKEND_CHOICES` 这 6 行就完了，不需要去某个类里找一个 `supports_draft_decode()` 方法的实现。新增一条特例（比如"某 backend 在某种 kv_cache_dtype 下需要报错"）只需要在 `arg_groups/overrides.py` 里加一个新的 `@register_post_process` 函数，插进 `_handle_attention_backend_compatibility()` 的调用序列——改动是局部的、可以单独测试的。
- **不这样会怎样**：如果把这些检查都塞进每个 backend 类自己的方法（vLLM 式路线），SGLang 这种"同一个 backend 名字在 MLA/非 MLA/NPU/HIP 下对应完全不同的类"（比如 `dsv4` 三个平台三个类）的场景会很别扭——能力声明应该挂在哪个类上？三个类各自声明一遍，还是在一个共同基类上声明再靠子类覆盖？现在的白名单+补丁链路线绕开了这个问题：白名单只关心字符串，不关心字符串背后具体是哪个类。
- **什么时候可以不这样**：如果 backend 数量长期停留在个位数、平台分支很少，白名单和 if/elif 决策树足够直观，没必要引入 vLLM 那种谓词框架的复杂度——但代价是，白名单之间**没有自动一致性检查**：`ATTENTION_BACKEND_CHOICES` 有 23 个字符串，`DRAFT_ATTENTION_BACKEND_CHOICES` 只有 6 个，`AttentionBackendRegistry` 的 MLA/MHA 切换表只覆盖 13 个——三张表谁该和谁保持子集关系，全靠人工维护，没有一处代码断言"如果 X 加入了主表，Y 表是否也该同步"。**这正是"支持得多"的反面**：22 个 backend 意味着至少 5 张需要人工同步的清单（`ATTENTION_BACKEND_CHOICES`/`DRAFT_...`/`CHUNKED_PREFIX_CACHE_SUPPORTED_...`/`DETERMINISTIC_...`/`AttentionBackendRegistry._handlers`），任何一张漏更新，产生的都不是启动时的报错，而是运行时的静默行为偏差。

### 决策 2：全局单例 backend（模块级变量），而不是每层持久化引用

- **为什么这么设计**：`ForwardContext`（`python/sglang/srt/model_executor/forward_context.py:34`）只有一个字段，`_current` 是模块级普通全局变量而不是 `contextvars.ContextVar`——docstring 里明确写了理由（`17`-`20`）："每个 forward 在单个 Python 线程上同步运行"，普通全局变量比 contextvar 更便宜、更直接。既然一次 forward 里所有层用的都是同一个 backend（`## 4.3` 提到的 hybrid-linear 混合模型例外后面单独说），没必要让每一层各自持有一份引用——那只是同一个指针的 N 份拷贝，徒增维护成本（比如模型热更新 backend 时要挨个改 N 个地方）。
- **不这样会怎样**：如果像 vLLM 那样让每层在构建期各自缓存 `self.attn_backend`，遇到"运行时需要换 backend"的场景（比如 PD-multiplexing 场景下 `decode_attn_backend_group` 要在多个预建好的 backend 实例间轮换，`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:89`-`96`）就要逐层修改引用，或者在每层里再包一层间接查找——全局单例模式恰好省掉了这层间接。
- **什么时候可以不这样**：如果模型结构要求不同层**永久性**地使用不同 backend（不是像 hybrid-linear 那样按层 id 分组，而是每层可能各不相同），全局单例就不够用了——`attn_backend_wrapper()` 处理 hybrid GDN/Mamba/MiniMax 稀疏这些场景时，实际上是在全局单例外面再包一层"内部按 `full_attention_layer_ids` 分发"的组合器（`HybridLinearAttnBackend`，`python/sglang/srt/layers/attention/attention_registry.py:506`-`508`），本质上是用组合而不是"放弃全局单例"来解决这个例外——全局单例仍然成立，只是这一个单例内部变复杂了。

### 决策 3：prefill/decode 用两个独立 backend 实例（`HybridAttnBackend` 组合），而不是同一个类里两个方法

- **为什么这么设计**：`decode_attention_backend`/`prefill_attention_backend` 是两个独立的 CLI 参数（`python/sglang/srt/server_args.py:1737`-`1753`），拆分粒度是"推理阶段"，对**所有模型**都适用，不像 vLLM 只在 MLA 场景才拆 prefill/decode backend。`HybridAttnBackend`（`python/sglang/srt/layers/attention/hybrid_attn_backend.py:20`）把这个拆分实现成一个**组合器**：它自己实现完整的 `AttentionBackend` 接口，内部持有两个真正的 backend 实例，按 `forward_mode` 转发调用（`61`-`80`）。这样任何两个 backend 只要都实现了标准契约，理论上都能被拼成一对 prefill/decode 组合，不需要每对组合单独写一个类。
- **不这样会怎样**：如果像 vLLM 的 MLA 处理方式那样，在一个类内部用两个方法（`forward_mha`/`forward_mqa`）表达"prefill 用一种算法，decode 用另一种"，这种拆分就被锁死在"同一个 kernel 库内部的两种模式"——`fa3` 用于 prefill、`trtllm_mla` 用于 decode 这种**跨厂商组合**是表达不出来的。组合器模式的代价在 `HybridAttnBackend.init_mha_chunk_metadata()` 那条注释里说得很直白（`134`-`142`）：分块前缀 MHA 的元数据准备必须显式委托给 `prefill_backend`，否则"prefix-cache-hit 的 extend batch 会跑在过期的 plan 上"（原文引用了一个真实报错信息）——组合带来的灵活性需要在每一个跨 backend 边界的细节上手动打补丁去缝合，接缝处的复杂度不会消失，只是从"写在一个类里"变成"写在组合器的转发逻辑里"。
- **什么时候可以不这样**：绝大多数用户从未显式设置过 `decode_attention_backend`/`prefill_attention_backend`（`attention_backends_of()` 在两者皆空时直接退回 `attention_backend`，`python/sglang/srt/arg_groups/overrides.py:277`-`290`），这种情况下 `_build_resolved_backend()` 走的是 `resolved.decode == resolved.prefill` 分支（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:230`-`234`），压根不会构造 `HybridAttnBackend`——这层组合复杂度只在用户主动要求"两阶段用不同 kernel"时才会被引入，源码里专门打了一条 warning 提醒"这个功能是 experimental and unstable"（`python/sglang/srt/layers/attention/attention_registry.py:225`-`227`）。

### 决策 4：MLA 复用同一个 `AttentionBackend` 契约（靠类继承表达"这是 MLA"），不新增专门抽象基类

- **为什么这么设计**：`FlashInferMLAAttnBackend(AttentionBackend)` 自己就是一个完整可用的 `AttentionBackend` 实现（`forward_extend`/`forward_decode` 齐全，`python/sglang/srt/layers/attention/flashinfer_mla_backend.py:597`/`675`），后续的 `FlashMLABackend`/`CutlassMLABackend`/`TRTLLMMLABackend` 全靠 `class Xxx(FlashInferMLAAttnBackend)` 继承它，复用它已经写好的 metadata 管理和 KV 池接口，只覆盖真正需要换的那部分（比如换一个不同的 CUDA kernel 调用）。好处是不需要在类型系统层面维护"这是不是 MLA"这件事——MLA-ness 是继承链的自然结果，不是一个显式声明的接口标记。
- **不这样会怎样**：如果像 vLLM 那样新增一个 `MLAAttentionImpl` 抽象基类、要求 MLA 后端实现 `forward_mha`/`forward_mqa` 两个方法而不是 `forward_extend`/`forward_decode`，`AttentionBackend.forward()` 里那段按 `forward_mode` 三路分发的通用逻辑（`python/sglang/srt/layers/attention/base_attn_backend.py:216`-`258`）就要为 MLA 单独复制一份，或者在契约里插入分支判断"这是不是 MLA 类型"——SGLang 选择让 MLA 和非 MLA 共用同一套分发逻辑，两者的差异被下推到"具体某个 `forward_extend` 实现内部要不要处理吸收态形状"这个更细的层次。
- **什么时候可以不这样**：MLA 场景下"这一步该用吸收态还是常规 MHA"这个决策本身，SGLang 并没有下沉到 backend 契约里——它被放在了模型层的 `dispatch_attn_forward_method()`（`## 4.5`）。如果一个团队希望这个决策对所有模型架构都可复用、不想在每个新模型类里重新实现一遍分流逻辑，把它上移成 backend 契约的一部分（类似 vLLM 的 `forward_mha`/`forward_mqa`）反而是更省事的选择——SGLang 目前只有 DeepSeek 系模型需要这套分流，复用价值还没有大到值得把它提升成一个跨模型的抽象层。

### 决策 5：绝大多数不兼容组合选择静默改写而不是报错

- **为什么这么设计**：`## 4.2` 列出的 9 个补丁函数里 7 个选择"打 warning + 自动改成一个能跑的值"。以 `page_size` 为例——它是一个横跨调度器、KV 池、attention backend 三个子系统的共享参数，普通用户很少真的关心它的具体数值，只关心"我选的这个 backend 能不能正常跑起来"。与其在 CLI 解析阶段就因为一个数值不匹配而拒绝启动，`_mla_backend_page_constraints` 选择直接把它改成 backend 能接受的值，让服务先跑起来。
- **不这样会怎样**：如果每一处不匹配都直接抛异常，用户光是让一个 `trtllm_mla` 模型跑起来，可能要在命令行上手动试错好几轮（先撞见 page_size 报错改成 64，再撞见 kv_cache_dtype 报错改成 fp8_e4m3……）。静默改写把这个试错过程内置进了启动流程本身。但代价是**可观测性下降**：这些改写只在日志里留一条 `logger.warning`，如果用户没细看启动日志（尤其是在容器化部署、日志被截断或过滤的场景），最终跑起来的服务配置和用户以为自己传的配置可能已经不是同一回事——`page_size` 这种会影响 KV 显存占用和调度粒度的参数被静默改写，排查性能异常时如果没想到去翻这几行日志，很容易南辕北辙。
- **什么时候可以不这样**：`_mla_kv_cache_dtype_checks`（决策路径里第 3 步）和 `_attention_backend_dual_chunk`（第 9 步，用户显式指定了冲突 backend 时）选择直接抛异常而不是静默改写——共同点是这两处的"正确值"没有一个显而易见的默认猜测（kv_cache_dtype 影响的是数值精度，改错了会安静地产出错误结果而不是报错；dual-chunk attention 是否启用是一个模型架构级别的二元判断，猜错了后果是"整个 attention 计算方式都不对"）。这两处的判断标准可以概括为：**当自动改写的后果是"性能差一点"时选择改写，当后果是"结果错但看起来正常"时选择报错**。

### 决策 6：把算子收进 `python/sglang/kernels/` 统一命名空间，用 `KernelBackend` 枚举编目三种不同来源，而不是各自散落

RFC #29630（`python/sglang/kernels/README.md:3`）把历史上分散的几处算子代码收进了同一个包：曾经独立发布的 PyPI 包 `sgl-kernel`（现在是 `kernels/aot/`，import 路径仍保留 `sgl_kernel` 以保持向后兼容，`python/sglang/kernels/aot/README.md:1`-`13`）、曾经存在的 `sglang.jit_kernel`（README 原话"has been **removed**"，迁移进 `kernels/jit/`）、以及 vendor 自 FlashAttention/CUTLASS 的 CuTe-DSL 内核和自研 Triton kernel（都在 `kernels/ops/<group>/` 下按算子分组，attention 只是 18 个分组之一）。

- **为什么这么设计**：`kernels/registry.py`/`selector.py`/`spec.py` 提供了一层统一的元数据登记（`register_kernel(KernelSpec(...))`），`README.md` 明确写"只记录元数据，不 import torch 或触发 JIT 编译，直到真正被调用"——三种截然不同来源（pip 依赖、vendor 代码、自研代码）的算子因此可以被同一套工具审计（"这个 op 有几种实现"）、同一套选择逻辑分发（`select_kernel(op, backend=None)`），而不需要调用方关心某个具体 kernel 到底是自己写的还是抄来的。这对 `## 0` 提到的"attention 子系统最大但不等于自研最多"这条已在 [[01-SGLang-全景与代码地图]] 讲过的坑，提供了一个可验证的分类账本——想知道某个具体算子的血统，读它的 `KernelSpec.backend` 字段就知道，不需要去翻 git blame 或版权头。
- **不这样会怎样**：如果 vendor 代码、pip 依赖调用、自研代码分散在互不知情的三套目录结构和三套 import 习惯里（这正是 SGLang 这次重构之前的状态——`sgl-kernel` 是独立仓库/独立 wheel，`jit_kernel` 是另一套体系），新增一种硬件后端支持时很容易"各写各的"：同一个算子在 AOT 路径下实现了一遍 ROCm 支持，JIT 路径下完全没跟进，两条路径的能力矩阵逐渐失去同步而没有人能一眼看出差异。
- **什么时候可以不这样**：如果一个引擎从第一天起就只服务单一硬件、算子数量有限（不像 SGLang 这样横跨 CUDA/ROCm/NPU/CPU/XPU 五种设备族），维护这样一套统一注册表的工程成本本身可能超过它带来的收益——直接按硬件平台分目录、各自为政，反而更直观。SGLang 走到需要这套机制的规模，本身就是 22 个 attention backend + 五种硬件平台叠加出来的复杂度倒逼的结果，不是提前设计好的。

### 决策 7：投机解码草稿模型的 backend 走一条独立的小白名单 + 独立的静默兜底，不复用主模型的选型结果

`## 4.7` 已经展开了机制：`DRAFT_ATTENTION_BACKEND_CHOICES`（`python/sglang/srt/server_args.py:213`-`220`，6 个值）和 `_resolve_draft_attention_backend_fallback()`（`python/sglang/srt/speculative/draft_worker_common.py:31`-`49`）构成了一条与 `## 5` 决策 1 描述的主选型逻辑完全平行的独立管线。

- **为什么这么设计**：草稿模型通常远小于目标模型（本篇 `## 4.7` 引用的 `speculative_draft_kv_cache_dtype` 帮助文本里提到"a 5-layer DFLASH draft"），它的 attention 计算特征（batch 极小、每步 token 数固定）和目标模型不同，能正确处理这种极端形状的 backend 集合本来就比目标模型的候选集合窄——6 个而不是 22 个是"筛选过的能力子集"，不是"随手选的一个更短列表"。独立管线还带来一个好处：目标模型换了 backend 不会连带影响草稿模型已经验证过能用的选择。
- **不这样会怎样**：如果草稿模型直接复用目标模型解析出的 `(prefill, decode)` 二元组（`## 4.3` 的常规路径），一旦目标模型选中了 `dsa`/`cutedsl_mla` 这类没在 `DRAFT_ATTENTION_BACKEND_CHOICES` 里出现过的 backend，草稿模型大概率会在实例化阶段直接报错或者产出错误结果——因为这些 backend 的 metadata 构建逻辑可能从没在"目标批一个 token、草稿批多个候选 token"这种投机解码特有的张量形状下测试过。
- **什么时候可以不这样**：如果一个团队确信自己的草稿模型和目标模型架构高度相似（比如同系列模型的小尺寸版本），且已经人工验证过某个不在白名单里的 backend 在草稿场景下工作正常，`speculative_draft_attention_backend` 这个 CLI 字段本身没有 `choices` 约束（`python/sglang/srt/server_args.py:2224`-`2227`）——**理论上**可以手动指定；实际拦下不合法组合的是 `_resolve_draft_attention_backend_fallback()` 这一处独立校验，而不是字段定义本身的类型约束，这也是为什么这条白名单能在不改 CLI 参数定义的情况下随版本迭代扩充。

### 决策 8：ROCm 平台用一张独立的重映射表处理 `AttnForwardMethod`，而不是让每个决策函数都写一遍平台分支

`## 3` 提到的 `AttnForwardMethod` 12 个成员里，`MHA_ROCM`/`MHA_ONE_SHOT_ROCM`/`MLA_ROCM` 三个是 CUDA 版本（`MHA`/`MHA_ONE_SHOT`/`MLA`）的 ROCm 对应物。`resolve_rocm_forward_method()`（`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:31`-`34`）在 `dispatch_attn_forward_method()`（`## 4.5`）的返回值外面再包一层：非 HIP 平台直接原样返回，HIP 平台查一张三项映射表 `_ROCM_FORWARD_METHODS`（`24`-`28`）做替换。

- **为什么这么设计**：源码注释直接给出了理由（`19`-`23`）——ROCm 有自己专门的 `forward_mha_rocm.py`/`forward_mla_rocm.py` 实现，"the shared CUDA paths carry no AMD branches"，也就是说每个 backend handler 函数（`handle_attention_fa3`/`handle_attention_flashinfer`……）内部完全不需要关心自己是不是跑在 ROCm 上，只管返回"这一步该用哪种计算模式"这个平台无关的答案；平台特定的替换收敛成这一张表、一次查询。
- **不这样会怎样**：如果每个 backend handler 函数自己判断"是不是 HIP，是的话返回 `_ROCM` 后缀的枚举值"，`## 2` 列出的每一个 `handle_attention_*` 函数体内都要重复一遍同样的 if-HIP 分支——13 个已注册 handler 函数，13 遍重复代码，任何一个漏写都会导致这一个 backend 在 ROCm 上走错计算路径。
- **什么时候可以不这样**：源码注释同样交代了这张表为什么不是四项而是三项——`MHA_CHUNKED_KV` 没有 ROCm 对应条目，因为它的累加步骤需要 CUDA-only 的 `merge_state_v2` kernel（`22`-`23`）。这意味着"统一重映射表"这个模式本身有一个隐含前提：所有需要平台特化的枚举值都能在目标平台上找到对应实现；一旦某个模式在某个平台上根本没有对应 kernel，这张表就没法覆盖它，只能让上层调用方自己规避（比如 ROCm 上永远不会选出一个会触发 `MHA_CHUNKED_KV` 的 backend 组合）——重映射表解决的是"翻译"问题，不解决"这个平台压根不支持某种计算模式"这个更底层的能力缺口。

## 6. 同位对照：vLLM 在同一位置怎么做

（对应 [[05-vLLM-注意力后端与算子层]]，本节涉及的 vLLM 行号已在 `_src/vllm` 独立核对，不转抄该篇未验证过的结论。）

- **backend 登记方式不同**：vLLM 用一个类型化的 `AttentionBackendEnum`（`vllm:vllm/v1/attention/backends/registry.py:34`），每个成员的值是一条**可 import 的类路径字符串**，实例化逻辑统一在基类外部（`resolve_obj_by_qualname`）；SGLang 用一个普通 `dict[str, Callable]`（`python/sglang/srt/layers/attention/attention_registry.py:31`），每个值是一个**工厂函数**，可以在实例化前后插入平台分支逻辑（`## 3` 已展开）。两边都不要求调用方在选型阶段就 import 具体 backend 的重依赖包，但 vLLM 靠"懒加载 import"做到这一点，SGLang 靠"把 import 语句写在工厂函数体内部"做到同一件事——效果相近，机制不同。本库独立核对确认：vLLM 这一版代码里 `VLLM_ATTENTION_BACKEND` 环境变量已经从仓库中完全消失（`grep -rl VLLM_ATTENTION_BACKEND _src/vllm` 零命中），选型入口是 CLI `--attention-backend`；SGLang 对应的入口自始至终就是 CLI `--attention-backend`/`--decode-attention-backend`/`--prefill-attention-backend`，没有经历过"曾经有环境变量后来去掉"这段历史。
- **"自动选择"的实现方式不同**：vLLM 是"每个平台一份声明式优先级列表 + 每个 backend 类自报能力谓词，选择器逐个跑 `validate_configuration()`"；SGLang 是一段写死在 `_get_default_attn_backend()`（`python/sglang/srt/server_args.py:5914`-`5986`）里的命令式 if/elif，`## 5` 决策 1 已经展开这两条路线各自的取舍——本质是"能力自证的表驱动"和"决策树直接编码进函数体"之分，前者新增 backend 更省事，后者新增一条判断分支更直接。
- **兼容性修正的组织方式接近，但落地位置不同**：vLLM 把大部分"这个组合是否合法"收进每个 backend 类自己的 `validate_configuration`/`supports_combination` 方法；SGLang 的 `_handle_attention_backend_compatibility()` 走的也是一串具名补丁函数（`## 4.2` 的 9 个 `_attention_backend_*`/`_mla_*`/`_fa4_*`），风格上和 vLLM 的"逃生舱"（`supports_combination`）异曲同工，但组织位置完全不同：vLLM 的补丁贴在**产生这个能力声明的 backend 类内部**，SGLang 的补丁贴在**与任何具体 backend 类都无关的、单独的 `arg_groups/overrides.py` 模块**里，靠字符串比较（`view.attention_backend == "trtllm_mla"`）而不是方法调用来判断"这条补丁该不该生效"。前者的好处是补丁离能力声明近，后者的好处是所有补丁能在一个文件里线性读完，不用跳进 21 个 backend 类里分别找。
- **prefill/decode 拆分的粒度不同**：SGLang 在 CLI 层面对**所有模型**暴露 `--decode-attention-backend`/`--prefill-attention-backend` 两个独立旋钮，`## 5` 决策 3 已展开其组合器实现；vLLM 目前只在 MLA 场景才拆 prefill/decode backend（`-ac.mla_prefill_backend` 与 `-ac.backend`），非 MLA 的普通注意力层没有这个粒度的拆分。反过来，vLLM 有一个 SGLang 没有的拆分维度——`backend_per_kind`，按 `KVCacheSpecKind`（全注意力/滑窗/…）分组覆盖 backend，SGLang 目前没有等价的"按层结构类型分组选 backend"的 CLI 级配置，混合结构模型（滑窗+全注意力交替）在 SGLang 里靠模型代码自己在 `attn_backend_wrapper()` 里手工判断（`python/sglang/srt/layers/attention/attention_registry.py:352`-`509`），不是一个通用配置项。
- **MLA 契约的抽象层级不同**：这是本篇的核心对照点。vLLM 新增了一个独立的 `MLAAttentionImpl` 抽象基类，要求 MLA 后端实现 `forward_mha`/`forward_mqa` 两个专门方法；SGLang 没有新增契约，MLA 后端复用同一个 `AttentionBackend.forward_extend`/`forward_decode`，"这一步该用吸收态还是常规 MHA"这个判断被完全下推到模型层的 `dispatch_attn_forward_method()`（`## 4.5`，`## 5` 决策 4 已展开两条路线各自代价）。换句话说：vLLM 把"MLA 有两种计算模式"这件事焊进了 attention 子系统的类型系统里；SGLang 把它当成"DeepSeek 系模型自己的业务逻辑"留在了模型定义文件里，attention 子系统本身对"MLA 有几种模式"毫不知情。
- **多模态编码器独立选择这件事，两边都做了，暴露方式也接近**：vLLM 把 ViT 选择逻辑（`get_vit_attn_backend()`）藏进平台类内部，没有专门 CLI 旋钮；SGLang 用 `--mm-attention-backend`（`python/sglang/srt/server_args.py:1777`-`1793`）显式暴露给用户，但两边都有一个共同点——ViT 侧的候选集合比主干侧小得多、校验也简化得多（SGLang 的 `_determine_attention_backend()` 只做"平台探测优先"，没有类似主干侧那 9 个补丁函数的组合校验链）：编码器场景没有 KV 分页、没有 MLA、没有投机解码，组合爆炸的维度天然就少了几个，两个引擎不约而同地选择了更简单的选型逻辑。
- **CUDA Graph 静态/动态形状的拆分粒度不同**：SGLang 把这件事拆成两个方法——`init_forward_metadata_out_graph`（capture/replay 之外，动态形状允许）与 `init_forward_metadata_in_graph`（capture 内部，`python/sglang/srt/layers/attention/base_attn_backend.py:95`-`107` 的 lint 契约明确禁止在里面调 `.item()`/`.cpu()`/`.tolist()`）；vLLM 走的是"一个 `build()` 方法 + 一个专门的 `build_for_cudagraph_capture()` 变体"外加 `AttentionCGSupport` 四级枚举描述这个 backend 能被捕获到哪种批次形状。两边都承认"capture 期间不能有动态 host-device 同步"这条硬约束，但 SGLang 把它做成方法签名层面的强制拆分（写错方法会在 capture 阶段直接报错），vLLM 把它做成一个描述性枚举（写错更可能是运行时行为异常而不是显式报错）——前者对新写 backend 的人更"防呆"，后者的信息更集中在一个字段里方便查询。
- **算子来源治理的"元层"有无不同**：`## 5` 决策 6 展开的 `KernelBackend` 枚举（`python/sglang/kernels/spec.py:29`-`42`）把 pip 依赖、vendor 代码、自研代码统一编目进同一套 `KernelSpec` 登记表；vLLM 对照篇 `## 5` 决策 6 描述的"FlashAttention 走 CMake fork、FlashInfer 走纯 pip、CUTLASS 走官方仓库自写 kernel"是同样的"三明治"结构，但**没有**一个类似的统一元数据层——三种来源各自体现在 `cmake/external_projects/`、`requirements/cuda.txt`、`CMakeLists.txt` 三处不同的构建配置文件里，靠人读构建脚本才能拼出完整的来源图谱。这是"是否值得建一个统一编目"这件事上两个引擎给出的不同答案——SGLang 22 个 backend 的规模显然已经越过了这个门槛，vLLM 目前主要是 3-4 种来源，尚未看到类似的统一登记诉求。
- **从"用户什么都不传"到"具体 kernel 被调用"要经过几层独立决策，两边数目不同**：本篇对照篇 `## 4` 给出的 vLLM 链路是三层——选 backend（声明式优先级表）、backend 内部再选版本（比如 FA2/FA3/FA4）、ViT 独立选型；本篇 `## 4.8` 给出的 SGLang 链路是六层——CLI 决策树（`## 4.1`）、9 步兼容性补丁链（`## 4.2`）、构建期实例化（`## 4.3`）、模型层 MHA/MLA 分流（`## 4.5`）、ViT 独立选型（`## 4.6`）、以及只在投机解码场景下才会插入的草稿 backend 独立管线（`## 4.7`）。层数更多不直接等于更复杂——`## 4.8` 的具体例子里，大部分组合下中间几层其实是"空跑"（9 步补丁对不匹配的 backend 名字直接跳过）——但层数本身是一个可以量化比较的"选型链路长度"指标，值得在评估两个引擎的可调试性时纳入考虑。

## 7. 踩坑与反直觉

1. **两个 deprecated backend 别名用了两种完全不同的实现机制。** `"nsa"` 是 `"dsa"` 的别名，靠 `@register_attention_backend("nsa")` 注册一个独立的 `_create_nsa_compat` 包装函数，运行时打 `DeprecationWarning` 再转调 `create_dsa_backend`（`python/sglang/srt/layers/attention/attention_registry.py:140`-`148`）；`"compressed"` 是 `"dsv4"` 的别名，靠 `ServerArgs.__post_init__` 里一段循环遍历四个 attention-backend 相关字段、发现值等于 `"compressed"` 就地 `setattr` 改成 `"dsv4"`（`python/sglang/srt/server_args.py:4208`-`4219`）——前者是"注册一个新工厂函数走正常注册流程"，后者是"配置解析阶段的字符串替换，压根不经过 `ATTENTION_BACKENDS` 表"。两条路径殊途同归，但如果要新增第三个 deprecated 别名，选哪条模式没有文档说明，只能照抄其中一个。
2. **`AttentionBackendRegistry` 的覆盖面比实际可用的 MLA backend 少。** `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:233`-`246` 只注册了 13 个名字的 MHA/MLA 切换策略（`ascend`/`flashinfer`/`fa3`/`flashmla`/`cutlass_mla`/`fa4`/`trtllm_mla`/`tokenspeed_mla`/`aiter`/`dsa`/`nsa`/`triton`/`intel_xpu`），但 `cutedsl_mla`、`hpc_ops`、`wave`、`torch_native`、`flex_attention`、`intel_amx`、`dual_chunk_flash_attn` 都不在这张表里。如果给一个 DeepSeek 系模型选了 `cutedsl_mla` 这种未注册的 backend，`get_handler()` 会静默退回 `"triton"` 的判断策略——真正跑 kernel 的仍然是 `CuteDslMLABackend`（`## 4.3` 已确认字符串到类的映射不受这张表影响），受影响的只是"这一步该走 MHA 还是吸收态 MLA"的判断逻辑本身可能不是针对 `cutedsl_mla` 优化过的（比如 `MHA_ONE_SHOT_SUPPORTED_BACKENDS` 不含 `cutedsl_mla`，一次性 MHA 优化不会触发）。**本库推断**：由于 `cutedsl_mla` 本身是 decode-only backend（`## 5` 决策/`## 4.2` 第 4 步已确认），实际受影响的窗口可能很窄；但这是一处源码里真实存在的表覆盖不全，不是本库编造的假设性 bug，具体运行时后果需要在真实硬件上复现才能下最终结论（**未查证**）。
3. **`hpc_ops` 这个第三方库名字容易和"高性能计算"泛称混淆。** 它特指 [Tencent/hpc-ops](https://github.com/Tencent/hpc-ops) 这一个具体项目（`python/sglang/srt/layers/attention/attention_registry.py:260` 的注释直接给出了链接），只在 Hopper（SM90）且 `page_size=64` 下可用（工厂函数 `create_hpc_ops_backend`，`260`-`274`），且显式拒绝投机解码（`268`-`271`，`raise ValueError`）——这是本篇 21 个具体实现类里唯一一个在工厂函数阶段就用异常拒绝投机解码组合的 backend，而不是等到运行时才报错。
4. **`triton_attn`/`aiter_attn` 与 `triton`/`aiter` 是两套不同命名空间下的字符串，容易看混。** `## 4.6` 已经展开：ViT 侧 `QKV_BACKEND_IMPL` 用的是 `triton_attn`/`aiter_attn`/`amx_attn`/`xpu_attn` 这套带 `_attn` 后缀的命名，主干侧 `ATTENTION_BACKEND_CHOICES` 用的是不带后缀的 `triton`/`aiter`/`intel_amx`/`intel_xpu`。这两套字符串**不能互相传**——`--attention-backend triton_attn` 会在 `ATTENTION_BACKEND_CHOICES` 校验阶段直接报错（因为 `triton_attn` 不在这张表里），`--mm-attention-backend triton` 同理会在 `mm_attention_backend` 的 `choices` 校验阶段报错。
5. **`sgl-kernel` 这个名字现在指的不是一个独立仓库/独立进程，而是同一个 monorepo 里的一个子目录。** 如果只看 PyPI 页面或者旧文档，容易以为 `sgl-kernel` 是和 `sglang` 主包分开维护、分开发版的独立项目；但在本篇取证的这一版仓库里，`kernels/aot/README.md` 自己承认包名已经改叫 `sglang-kernel`（`"sgl-kernel (prior sgl-kernel)"`，`1`），源码树就在 `python/sglang/kernels/aot/` 下，和 attention backend 的其余代码同仓库、同 sha、同一次 clone 就能拿到——不需要额外再 clone 一个仓库。
6. **草稿模型的静默 fallback 有一条容易被忽略的触发路径：目标模型自己的 backend 字符串会被拿去当草稿候选值试。** `## 4.7` 已经展开：`_resolve_draft_attention_backend_fallback()`（`python/sglang/srt/speculative/draft_worker_common.py:35`）在用户没显式设置 `--speculative-draft-attention-backend` 时，第一候选不是硬编码默认值，而是 `server_args.get_attention_backends()` 的返回值——也就是**目标模型自己解析出的 backend**。如果目标模型选中了 `dsa`（DeepSeek 稀疏注意力，`## 2` 已确认它不在 `DRAFT_ATTENTION_BACKEND_CHOICES` 6 个值里），投机解码会先把 `"dsa"` 当候选值，白名单检查失败后才静默换成 `"triton"`/`"flashinfer"`——中间那一步"试了一下目标模型的 backend 但没通过"完全体现在日志的一行 warning 里，不细看容易以为草稿 backend 是凭空选出来的默认值。
7. **算子目录名字保留了一段命名历史，容易和已经废弃的 CLI 别名对不上号。** `python/sglang/kernels/ops/attention/nsa_triton_decode/` 这个目录名字面上对应的是 `"nsa"`（Native Sparse Attention 的缩写），但 `## 0`/`## 7` 第 1 条已经确认 `"nsa"` 在 CLI 层面只是 `"dsa"` 的 deprecated 别名——目录名没有跟着改成 `dsa_triton_decode`。这不是本库找到的一个 bug，只是提醒：仓库里的目录名、文件名不总是和当前有效的 CLI 字符串保持同步重命名，读代码定位算子实现时，历史名字和当前名字要分开对待。
8. **`attn_backend_wrapper()` 只包一次而不是包两次，是为了不重复初始化混合模型的线性/稀疏侧 backend。** `## 4.3` 提到构造 `HybridAttnBackend` 之后还要再包一层 `attn_backend_wrapper()`；源码注释在这一处专门解释了顺序为什么不能反过来（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:198`-`203`）："Wrapping each child independently duplicates the linear/sparse side backend for hybrid models (for example, two GDN dispatchers for Qwen3.5 when prefill and decode use different MHA backends)"——如果对 `prefill_backend`/`decode_backend` 两个子 backend 分别调用 `attn_backend_wrapper()`，Qwen3.5 这类混合 GDN 模型会被初始化出**两份**独立的线性注意力 dispatcher（一份挂在 prefill 子 backend 上，一份挂在 decode 子 backend 上），而任意一次 forward 里只有其中一份真正会被用到——多出来的那一份纯粹是浪费的初始化开销和状态。这是本篇找到的又一处"组合器模式"里容易踩的顺序坑：先组合两个同类 backend，再统一包一层模型级 wrapper，不能反过来。

## 8. 可改进点

（以下均为本库基于源码走读的推断，标注证据依据；**未查证**是否已有官方 issue 在跟踪。）

- **`AttentionBackendRegistry` 的覆盖面应该有一个启动期一致性检查。** 目前没有发现任何测试遍历 `ATTENTION_BACKENDS`（全部 22 个字符串）并断言"如果这个 backend 支持 MLA 模型，`AttentionBackendRegistry._handlers` 里也应该有对应条目"——`## 7` 第 2 条提到的 `cutedsl_mla` 缺口本可以在 CI 阶段就被一个简单的集合差集检查捕获，而不必等到有人真的用这个组合跑起来才发现。
- **五张需要人工同步的白名单（`## 5` 决策 1 提到）可以收敛成一份。** `ATTENTION_BACKEND_CHOICES`/`DRAFT_ATTENTION_BACKEND_CHOICES`/`CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`/`DETERMINISTIC_ATTENTION_BACKEND_CHOICES`/`AttentionBackendRegistry._handlers` 目前互相独立维护；如果给每个 backend 的工厂函数（或者一个新增的轻量装饰器参数）挂上"支持草稿解码/支持分块前缀/支持确定性推理"这几个布尔标记，这几张表可以从"人工列出的字符串列表"变成"从单一登记表按标记过滤出来的视图"，新增一个 backend 时不会再有遗漏某张表的风险。
- **9 个补丁函数的静默改写缺少一个"这次启动到底改了什么"的汇总。** `## 5` 决策 5 提到的 7 处静默改写各自打了一条独立的 `logger.warning`，分散在启动日志的不同位置；一个专门的"配置解析摘要"（比如启动完成时打印一张"用户传入值 → 最终生效值"的差异表）可以让这些改写在一处集中可见，而不需要用户翻遍整个启动日志去拼凑"我的 page_size 到底被谁改了"。
- **ViT 侧和主干侧的 backend 命名空间可以考虑加前缀而不是后缀区分。** `## 7` 第 4 条提到的 `triton_attn` vs `triton` 容易混淆，如果两套命名从设计时就用更醒目的前缀区分（比如 `mm:triton` vs `triton`），错传参数时的报错信息也能更早、更准确地提示"你可能传错了命名空间"，而不是一条通用的"不在 choices 列表里"。
- **草稿模型 backend 解析目前分散在两个文件的两个函数里，值得合并成一个入口。** `resolve_draft_attention_backend()`（`python/sglang/srt/model_executor/model_runner.py:267`-`282`）和 `_resolve_draft_attention_backend_fallback()`（`python/sglang/srt/speculative/draft_worker_common.py:31`-`49`）都在做"草稿模型该用哪个 backend"这件事，前者决定 `ModelRunner.draft_attention_backend` 这个属性的初始值，后者在真正构建草稿 worker 时做白名单校验和兜底——本篇 `## 4.7` 需要跨两个文件才能拼出完整链路，说明这条决策目前没有单一入口可以一眼读完；合并成一个函数（或者至少让其中一个显式调用另一个而不是两条平行存在的路径）能降低下一个想理解"草稿 backend 到底怎么定下来的"的人的阅读成本。

## 9. 自测题与延伸阅读

**闭卷自测题**（合上本文，尝试不看源码回答）：

1. `ATTENTION_BACKENDS`（`attention_registry.py`）和 `ATTENTION_BACKEND_CHOICES`（`server_args.py`）是同一张表吗？两者的 key 集合为什么不完全相同？
2. SGLang 的"自动选择默认 backend"逻辑（`_get_default_attn_backend`）和 vLLM 的优先级列表在实现风格上有什么本质区别？各自的新增成本落在哪里？
3. `HybridAttnBackend` 是一个新的算子实现，还是一个组合器？它是怎么让"prefill 用一个 backend，decode 用另一个"这件事对模型层透明的？
4. 举出本篇提到的至少 3 处静默 fallback，说明各自"改的是什么值"以及"为什么选择不报错"。
5. MLA 的 `forward_mha`/`forward_mqa` 式分流在 vLLM 里发生在 backend 契约层，在 SGLang 里发生在哪一层？这个差异导致了什么后果？
6. `python/sglang/kernels/` 下的算子来自几种不同的"血统"？分别举一个文件路径作为证据。
7. 为什么滑窗和吸收态 MLA 的组合需要一条单独的 `swa_mla_fallback` 路径，而不能直接复用常规 MLA kernel？
8. `AttentionBackendRegistry.get_handler()` 找不到对应 backend 时会发生什么？这个行为对最终跑起来的 kernel 调用有没有影响？
9. 投机解码草稿模型的 attention backend 是从目标模型的 `(prefill, decode)` 结果直接继承的吗？如果不是，它经过了哪几层独立的解析和白名单校验？
10. `HybridAttnBackend`、`HybridLinearAttnBackend`、`TboAttnBackend` 这三个"组合器"类各自组合的是什么？它们的共同点是什么？
11. 一个没设任何 `--attention-backend` 参数的 DeepSeek 系模型部署在 Blackwell 上，最终会跑在哪个具体的 backend 类上？中间经过了几道独立的选择？

**延伸阅读（本库内）**：

- [[01-SGLang-全景与代码地图]] —— `python/sglang/kernels/` 的三层来源拼图与 `attention` 子系统 217,907 行的全貌统计，本篇复用了这两条已核验的结论
- [[04-SGLang-内存池与KV布局]] —— `page_size`/KV 物理布局如何被具体 attention backend 的内核约束反向钉死，本篇 `## 5` 决策 5 提到的静默改写在那一篇能看到"改了之后 KV 池怎么响应"的下游影响
- [[05-vLLM-注意力后端与算子层]] —— `## 6` 的完整对照对象，MLA 契约、prefill/decode 拆分粒度、多模态编码器独立选型三条对照均以该篇的已核验行号为准
