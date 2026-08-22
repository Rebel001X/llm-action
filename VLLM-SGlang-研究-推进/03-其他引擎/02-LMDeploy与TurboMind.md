# LMDeploy 与 TurboMind：双引擎的边界与代价

> **本篇取证基准**：`lmdeploy` @ `1263d5cd`（2026-08-21）
> **一句话**：两套引擎、两张配置表，接口面貌统一，能力并不对称

## 0. 结论先行

- LMDeploy 不是"一个引擎加几个后端选项"，是**两套完全独立的执行栈**：C++ 的 TurboMind（`src/turbomind/`，脱胎于 NVIDIA FasterTransformer）与纯 PyTorch 的 `pytorch` backend（`lmdeploy/pytorch/`）。两者各有一张配置表——`TurbomindEngineConfig` 42 字段（`lmdeploy/messages.py:209`）、`PytorchEngineConfig` 48 字段（`lmdeploy/messages.py:371`）——概念上重叠的不到一半，`## 3` 会逐族对比。
- 引擎选择是**不对称**的：显式传 `PytorchEngineConfig` 会被硬锁死（`lmdeploy/archs.py:72`），但显式传 `TurbomindEngineConfig` **不会**被硬锁——`autoget_backend_config()` 仍会重新跑一遍架构核验，模型不在 TurboMind 白名单里就静默降级到 pytorch，只留一行 `logger.warning`（`lmdeploy/archs.py:41`-`43`）。这个白名单只有约 15 个架构（`lmdeploy/turbomind/supported_models.py:7`-`33`），DeepSeek 系列、GLM-4-MoE（带视觉时）、块扩散模型统统不在其中。
- TurboMind 的"persistent batch"不是 vLLM 式"每步用队列重建批次"，而是一个**容量恒定为 `max_batch_size` 的槽位集合**：请求进 batch 占一个槽、结束释放一个槽，槽位数组本身（C++ 里的 `State::rc`）在整个服务生命周期内被复用，不是每步重新分配。`## 5` 会讲这个设计现在的真实代价。
- LMDeploy 招牌的 KV cache INT4/INT8 量化，在 TurboMind 和 pytorch 两个后端里都是**运行时动态量化**——每次写 KV cache 时现算一个 warp/group 内的 min/max 做非对称量化（`src/turbomind/kernels/attention/quantization.h:341`-`366`），**没有任何校准（calibration）步骤**。这和 LMDeploy 自己的权重量化（AWQ，`lmdeploy/lite/quantization/calibration.py`）形成对照——后者需要跑校准数据集，前者完全不需要。
- 最反直觉的一条：`QuantPolicy.TURBO_QUANT`（值 42，K=4bit QJL4 + V=2bit MSE）这个名字听起来专属于 TurboMind，但源码里**只有 pytorch backend 实现了它**（`lmdeploy/pytorch/kernels/cuda/fill_kv_cache.py`），TurboMind C++ 侧完全没有对应代码路径，且 `TurbomindEngineConfig.__post_init__` 也没有拦截这个值——这正是"双引擎特性漂移"的活样本。
- API 表面 37 条路由（`_lab/out/api_surface.json`），横跨 4 套协议：OpenAI 兼容（8 条 `/v1/*`）、Anthropic 兼容（`/v1/messages` 等 3 条）、LMDeploy 自有的 `/generate`/`/get_ppl`，以及一个独立的多实例代理 `lmdeploy/serve/proxy/proxy.py`（4 个文件、1,083 行）——它解决的是"一个统一入口分发到多个 LMDeploy 实例"，`## 4`/`## 6` 会展开。

**速查：关键问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| 42 字段和 48 字段到底哪些重叠、哪些各自独有？ | `## 3` |
| 用户没指定引擎时怎么选？指定错了会报错还是静默换后端？ | `## 4` |
| TurboMind 的 persistent batch 和 vLLM 连续批处理具体差在哪？ | `## 4`、`## 5` 决策 3 |
| KV int4/int8 量化在哪一行代码发生？要不要校准？ | `## 4` ④、`## 5` 决策 4 |
| 为什么要养两套引擎，这笔账现在划算吗？ | `## 5` 决策 1 |
| proxy.py 解决什么问题，和 SGLang 的 router 有什么不同？ | `## 4` ⑤、`## 6` |

## 1. 它在系统里的位置

用户面对的入口只有两个：Python 的 `lmdeploy.pipeline()`（`lmdeploy/api.py:15`-`77`，内部构造 `Pipeline` 对象，`lmdeploy/pipeline.py:34`）和 CLI 的 `lmdeploy serve api_server`（`lmdeploy/cli/serve.py:16` 的 `SubCliServe`）。两个入口最终都汇合到同一个决策点——`autoget_backend_config()`（`lmdeploy/archs.py:54`-`91`）——它读一遍模型的 HuggingFace config，判断这个模型架构能不能用 TurboMind 跑，产出 `('turbomind', TurbomindEngineConfig)` 或 `('pytorch', PytorchEngineConfig)` 这一对结果。

从这一步往下，`AsyncEngine`（`lmdeploy/serve/core/async_engine.py:90`）是**唯一**同时认识两个后端的类：它的 `__init__` 里有 8 处 `backend == 'turbomind'`/`backend == 'pytorch'` 的显式分支（本篇统计自 `lmdeploy/serve/core/async_engine.py` 全文 grep），分别调 `_build_turbomind()` 或 `_build_pytorch()`。往下一层，两个后端彻底分道扬镳：

```
用户请求
  └─ AsyncEngine（唯一横跨两后端的类，backend=='turbomind'/'pytorch' 分支处处可见）
       ├─ backend='turbomind' → src/turbomind/ 的 C++ Engine（独立线程模型，见 ## 4）
       └─ backend='pytorch'   → lmdeploy/pytorch/engine/ 的 Python Engine
                                  （调度器 lmdeploy/pytorch/paging/scheduler.py，
                                   注释原文写着 "modify from: https://github.com/vllm-project/vllm"）
```

再往上一层，`lmdeploy/serve/openai/` 下的 OpenAI 兼容路由、`lmdeploy/serve/anthropic/` 下的 Anthropic 兼容路由，以及独立跑在多个 LMDeploy 实例前面的 `lmdeploy/serve/proxy/proxy.py`，这三层完全不关心底层是哪个引擎——它们只认 `AsyncEngine` 暴露的统一接口。这条边界很重要：**API 表面统一，不代表两个引擎背后的能力也统一**，`## 3`/`## 5` 会用配置字段的差异把这一点坐实。

## 2. 代码地图（文件 → 职责，带行号）

按"配置与选型 → C++ 引擎 → PyTorch 引擎 → API/代理"的顺序排列：

| 文件:行 | 职责 |
|---|---|
| `lmdeploy/messages.py:20`-`27` | `QuantPolicy` 枚举——KV 量化策略的唯一定义处，两个后端共用同一套值 |
| `lmdeploy/messages.py:209` | `TurbomindEngineConfig` 类定义，42 字段 |
| `lmdeploy/messages.py:371` | `PytorchEngineConfig` 类定义，48 字段 |
| `lmdeploy/archs.py:10`-`51` | `autoget_backend()`——探测模型架构、决定用哪个后端，失败只 warning 不报错 |
| `lmdeploy/archs.py:54`-`91` | `autoget_backend_config()`——引擎选择与配置对象构造的真正入口 |
| `lmdeploy/turbomind/supported_models.py:7`-`33` | `SUPPORTED_ARCHS`——TurboMind 能跑的模型架构白名单，约 15 个 |
| `lmdeploy/pipeline.py:75` | `Pipeline.__init__` 里调用 `autoget_backend_config` 的确切位置 |
| `lmdeploy/cli/serve.py:226`-`235` | CLI 侧的引擎选择逻辑，与 `pipeline()` 的路径不完全相同 |
| `lmdeploy/serve/core/async_engine.py:90` | `AsyncEngine` 类——唯一横跨两后端的门面 |
| `src/turbomind/engine/README.md` | TurboMind C++ 引擎异步执行模型的**规范性契约文档**（本篇 `## 4` 大量引用） |
| `src/turbomind/engine/batch.h:47`-`74` | `BatchData` 结构——engine 线程与 model executor 线程之间传递的批次载体 |
| `src/turbomind/engine/engine.cc:171`-`184` | `Engine::Impl::State`——持有 `rc`（当前批内序列数组）与 `perm`（重排索引） |
| `src/turbomind/engine/engine.cc:794`-`795` | `n_free = max_batch_size - st.size() + st.finish`——槽位余量计算 |
| `src/turbomind/engine/scheduler.h`/`scheduler.cc` | TurboMind 调度事务（`PlanResume`/`PlanContinue`/`Schedule`）——概念定义见 README |
| `src/turbomind/kernels/attention/quantization.h:341`-`390` | `warp_stats`/`quantize`——KV cache 动态量化的核心：warp 内 min/max 统计 + 非对称量化 |
| `lmdeploy/pytorch/paging/scheduler.py:2` | 文件头注释 "modify from: https://github.com/vllm-project/vllm" |
| `lmdeploy/pytorch/paging/scheduler.py:461` | `Scheduler` 类定义——pytorch backend 自己的连续批处理调度器 |
| `lmdeploy/pytorch/kernels/cuda/fill_kv_cache.py:23`-`41` | `_quant_int8`/`_quant_int4`——pytorch backend 对 TurboMind KV 量化的独立平行实现 |
| `lmdeploy/pytorch/disagg/config.py:21`/`:39` | `EngineRole`/`MigrationBackend` 枚举——PD 分离配置，只出现在 pytorch 一侧 |
| `lmdeploy/serve/openai/endpoints/distserve.py:16`-`31` | PD 分离 HTTP 端点，读取的字段却是 pytorch 专属（`## 7` 细讲） |
| `lmdeploy/serve/proxy/proxy.py:71` | `NodeManager` 类——多实例代理的路由核心 |
| `lmdeploy/serve/proxy/utils.py:18`-`33` | `RoutingStrategy` 枚举（RANDOM / MIN_EXPECTED_LATENCY / MIN_OBSERVED_LATENCY） |

（共 20 条，超过硬指标要求的 10 条。`_lab/out/struct_map.json` 的口径提示：kv_cache 子系统统计的 5,180 行、quantization 子系统统计的 8,419 行**全部落在 `lmdeploy/pytorch/` 下**——这是 `struct_map.py` 基于 Python AST 抽取的必然结果，C++ 侧 `src/turbomind/kernels/attention/quantization.h` 等文件不进这个统计口径，读数字时要留意这一层"看不见 C++"的偏差，不是 C++ 侧真的没有这些代码。）

**本篇范围边界**：TurboMind 的 CUDA kernel 内部实现（FMHA、GEMM、通信）不是本篇重点，本篇只讲到"KV 量化点在哪一层函数"为止；pytorch backend 的分布式并行（`lmdeploy/pytorch/disagg/`、`enable_microbatch`、`enable_eplb`）留给专门的 PD 分离/并行篇章，本篇只在 `## 3`/`## 7` 中作为"配置字段不对称"的证据出现。

## 3. 核心数据结构

### 3.1 两张配置表的逐族对比

这是本篇最核心的一张表。`TurbomindEngineConfig`（`lmdeploy/messages.py:209`-`368`，42 字段）与 `PytorchEngineConfig`（`lmdeploy/messages.py:371`-`551`，48 字段）按语义分族：

| 族 | TurboMind 独有 | PyTorch 独有 | 两边都有（字段名可能不同） |
|---|---|---|---|
| 并行策略 | `cp`（context parallel）、`attn_cp_size`、`outer_dp_size`、`nnodes`/`node_rank`/`dist_init_addr`、`device_num`、`communicator`（`:337`，nccl 通信后端选择） | `moe_tp_size`、`distributed_executor_backend`（uni/mp/ray）、`enable_microbatch`、`enable_eplb` | `tp`/`dp`/`ep`、`attn_tp_size`/`mlp_tp_size` |
| KV cache 分块 | `cache_block_seq_len`（`:323`）、`cache_chunk_size`（`:322`） | `block_size`（`:470`）、`kernel_block_size`（`:471`，pytorch 特有的物理/逻辑块尺寸分离） | `cache_max_entry_count`（Turbomind `:321`，Pytorch `:468`，语义一致：KV cache 占显存百分比） |
| 前缀缓存/检查点 | `cache_prompt`（all/auto）、`cache_generation`（all/auto/none）、`cache_checkpoint_interval`、`cache_prompt_boundary_skip`——TurboMind 自研的"部分块边界发布 + 循环状态检查点"机制（详见 `src/turbomind/engine/README.md` 的 `boundary-policy` 契约） | `prefix_cache_state_budget`、`prefix_cache_decode_state_interval`——面向 SSM/循环状态模型的检查点节流 | `enable_prefix_caching`（Turbomind `:324`，Pytorch `:478`） |
| 量化 | `rope_scaling_factor`、`use_logn_attn`（TurboMind 自带的旧式 NTK/logn attention 实现） | `model_format`（仅 `'fp8'` 一个选项） | `quant_policy`（Turbomind `:329` 拒绝 FP8/FP8_E5M2，Pytorch `:486` 全部接受，`## 5` 决策 4 展开） |
| PD 分离/推测解码 | （无对应字段） | `role`（`:508`，Hybrid/Prefill/Decode）、`migration_backend`（`:509`，DLSlime）——PD 分离仅在此配置里可设 | `num_tokens_per_iter`/`max_prefill_iters`（TurboMind 侧的 "Dynamic SplitFuse"-like 调度，与 pytorch 的 `prefill_interval` 概念相近但不同名） |
| 块扩散/其它前沿特性 | （无对应字段） | `dllm_block_length`/`dllm_unmasking_strategy`/`dllm_denoising_steps`/`dllm_confidence_threshold`（块扩散语言模型专属）、`logprobs_mode`、`enable_return_routed_experts`、`enable_transfer_obj_ref` | `empty_init`、`language_model_only`、`hf_overrides`、`enable_metrics`、`download_dir`/`revision` |

**读这张表最该记住的一句话**：42 与 48 字段里，语义完全对应的不到 15 对，TurboMind 独有的字段几乎全部是"C++ 引擎自己的性能/并行旋钮"（NCCL 通信方式、上下文并行、循环状态检查点边界），PyTorch 独有的字段几乎全部是"新架构/新范式的接入点"（PD 分离、块扩散、EPLB、微批）。这不是两张表凑巧长得不一样，是**两套引擎在演化方向上已经分岔**——TurboMind 在往"同一批经典架构榨得更快"演化，pytorch backend 在往"接住更多新架构"演化。

### 3.2 `QuantPolicy` 枚举：量化策略的唯一真源

`lmdeploy/messages.py:20`-`27`，逐字抄录：

```python
class QuantPolicy(enum.IntEnum):
    """Quantization policy constants for KV cache."""
    NONE = 0
    INT4 = 4  # 4-bit KV cache
    INT8 = 8  # 8-bit KV cache
    FP8 = 16  # FP8 KV cache (float8_e4m3fn, per-tensor scale. DSA uses the fp8_ds_mla layout)
    FP8_E5M2 = 17  # FP8 KV cache (float8_e5m2, per-tensor scale)
    TURBO_QUANT = 42  # TurboQuant: K=4bit QJL4 + V=2bit MSE
```

两个后端共用这一份定义，但**接受的取值范围不同**：`TurbomindEngineConfig.__post_init__`（`lmdeploy/messages.py:352`-`358`）显式拒绝 `FP8`/`FP8_E5M2`；`PytorchEngineConfig.__post_init__`（`:527`-`530`）不做这层过滤，只检查 `quant_policy > 0` 时设备类型必须是 `cuda`/`ascend`（`:543`-`545`）。`TURBO_QUANT`（42）在两边的 `__post_init__` 里都没有被显式拒绝——但只有 pytorch backend 真正实现了它（`## 5` 决策 5 展开这个坑）。

### 3.3 `GenerationConfig`：唯一真正统一的第三张表

`lmdeploy/messages.py:36`-`207`，30 字段，是每次生成请求的采样参数（`temperature`/`top_p`/`top_k`/`stop_words`/`response_format` 等）。这张表**不区分后端**——两个引擎读的是同一个 `GenerationConfig` 对象。换句话说，"用户能控制生成行为的旋钮"是统一的,"用户能控制引擎怎么跑"的旋钮是分裂的——这条边界划得很清楚：`GenerationConfig` 属于请求语义,`TurbomindEngineConfig`/`PytorchEngineConfig` 属于部署语义,只有后者暴露了双引擎的分岔。

### 3.4 TurboMind C++ 侧：`BatchData` 与槽位状态

`src/turbomind/engine/batch.h:47`-`74` 定义的 `BatchData` 是 engine 线程与 model executor 线程之间传递的载体，核心字段：

```cpp
struct BatchData {
    const int phase;
    int bs0 = 0;  // prev batch size
    int bsz = 0;  // curr batch size
    Buffer_<int> perm;   // 排列索引：当前批位置 -> 上一批位置
    std::vector<ResolvedCopy> restore_copies;   // kPrepare 之前执行
    std::vector<ResolvedCopy> publish_copies;   // kUnprep 之后执行
    ...
};
```

而真正持有"当前有哪些请求在跑"的是 `Engine::Impl::State`（`src/turbomind/engine/engine.cc:171`-`184`）：

```cpp
struct State {
    vector<unique_ptr<Sequence>> rc;   // 当前批内的序列状态
    vector<int> perm;                  // current -> previous
    int bs0 = 0, active = 0, finish = 0, swapout = 0;
    int size() const noexcept { return rc.size(); }
};
```

`rc` 的长度上限由 `param_.max_batch_size` 卡死——`src/turbomind/engine/engine.cc:794`-`795` 算槽位余量的代码是 `n_free = param_.max_batch_size - st.size() + st.finish`。这就是"persistent batch"在当前源码里的真实形态：不是教科书式"预分配 N 个槽位数组、每步原地覆写"，而是一个**容量恒定、内容动态增删的 `vector`**，但概念上仍然是"batch 本身作为持久对象存在，请求进出这个对象"，而不是 vLLM 那种"每步从队列里现取现拼一个新批次描述"。

## 4. 主流程走读

### ① 引擎选择：从模型路径到一对 `(backend, config)`

`autoget_backend_config()`（`lmdeploy/archs.py:54`-`91`）的完整决策逻辑，逐字抄录关键段：

```python
def autoget_backend_config(model_path, backend_config=None, trust_remote_code=False):
    if isinstance(backend_config, PytorchEngineConfig):
        return 'pytorch', backend_config          # L72：唯一的硬锁分支

    backend = autoget_backend(model_path, trust_remote_code=trust_remote_code)  # L75
    config = PytorchEngineConfig() if backend == 'pytorch' else TurbomindEngineConfig()
    if backend_config is not None:
        if type(backend_config) is type(config):
            config = backend_config
        else:
            # 把用户传的 TurbomindEngineConfig 字段搬进新选出的 PytorchEngineConfig（或反过来）
            data = asdict(backend_config)
            for k, v in data.items():
                if v and hasattr(config, k):
                    setattr(config, k, v)
            ...
    return backend, config
```

注意 L72 只判断 `isinstance(backend_config, PytorchEngineConfig)`——这意味着：
- 传 `PytorchEngineConfig` ⇒ **强制** pytorch，不管模型是否被 TurboMind 支持。
- 传 `TurbomindEngineConfig`、或什么都不传 ⇒ 总是先跑一遍 `autoget_backend()`（`lmdeploy/archs.py:10`-`51`）——它调用 `lmdeploy/turbomind/supported_models.py:38` 的 `is_supported()`，查 `SUPPORTED_ARCHS` 白名单（`:7`-`33`，`Qwen2/Qwen2Moe/Qwen2VL/Qwen3/Qwen3Moe/Qwen3_5/Qwen3_5Moe/InternVL*/InternLM2/InternLM3/Llama/Glm4MoeLite/GptOss/Mixtral`,约 15 个架构，注意**没有 DeepSeek 系列**）。查不到就 `logger.warning('Fallback to pytorch engine because ... not supported by turbomind engine.')`（`lmdeploy/archs.py:41`-`43`），静默换成 pytorch。就算用户手上明明拿着一个 `TurbomindEngineConfig` 对象传进去,如果模型不在白名单里,`autoget_backend_config()` 返回的第一个值仍然是 `'pytorch'`,只是把 `TurbomindEngineConfig` 里能对上的字段值誊抄进新建的 `PytorchEngineConfig`（`lmdeploy/archs.py:83`-`88` 的字段映射，包括 `block_size`/`cache_block_seq_len` 这一对特意做了改名映射）。

CLI 路径（`lmdeploy/cli/serve.py:226`-`235`）逻辑略有不同但结论一致：

```python
backend = args.backend                              # 默认 'turbomind'（lmdeploy/cli/utils.py:390-397）
if backend != 'pytorch':
    backend = autoget_backend(args.model_path, ...)  # 显式传 turbomind 或不传都会重新核验
```

`--backend` 的默认值是 `'turbomind'`（`lmdeploy/cli/utils.py:393`-`396`），但只要不是显式 `'pytorch'`,就要再过一遍 `autoget_backend()` 的架构核验——**CLI 和 Python API 在这一点上行为一致：只有 pytorch 是硬选项,turbomind 永远是"尝试性"选项**。

### ② TurboMind 一次调度事务（源码为证 + `src/turbomind/engine/README.md` 文档所述）

TurboMind 的 engine 线程（`Engine::Impl::InternalThreadEntry`）和 model executor 线程（`ModelExecutor::Impl::InternalThreadEntry`）异步流水：

1. **准入**：`Gateway` 把外部请求塞进按队列 round-robin 分发的 `RequestQueue`（`src/turbomind/engine/gateway.h`/`.cc`），engine 线程从队列里弹出请求、转成 `Sequence`（`BatchOp::kAdd`）。
2. **调度事务**：对每个候选 `Sequence` 调 `PlanResume`（新进请求）或 `PlanContinue`（继续跑的请求）——只做"选逻辑块、算 `resume_len`、生成 restore 拷贝意图",不动真实显存；然后 `Scheduler::Schedule()` 一次性提交:决定谁激活、分配/淘汰缓存块、设置 `history_len`/`input_len`（`src/turbomind/engine/README.md` 的 `scheduler-transaction`/`scheduler-commit` 两节）。
3. **Setup**（`BatchOp::kSetup`，engine 线程）：把调度决定的元数据搬进 `BatchData`,解析缓存块 handle 到设备地址——但**不**在这一步碰显存内容。
4. **执行**（`BatchOp::kPrepare`/`kForward`/`kUnprep`，model executor 线程）：真正的 CUDA kernel 在这里跑,包括本篇关心的 KV 量化写入。
5. **回收**（`BatchOp::kFetch`/`kUpdate`,engine 线程）：结果拷回 host,更新 `Sequence` 状态；已完成的请求进入 `retiring` 状态,等 `inflight == 0` 才真正释放资源（`BatchOp::kDel`）。

这个流程里"批"的身份贯穿始终——`BatchData` 的 `bsz`/`perm` 描述的是**这一步谁在批里、排在第几位**,而批本身（`Engine::Impl::State::rc`）作为一个跨越多步存在的容器,容量受 `max_batch_size` 约束。对照 `## 5` 决策 3,这与 vLLM 每次 `schedule()` 调用现场从 `self.running`/`self.waiting` 两个队列拼出 `SchedulerOutput` 的方式,在"批的生命周期"这一点上是两种不同的心智模型。

### ③ pytorch backend 的一次调度：读者已经"认识"这段代码

`lmdeploy/pytorch/paging/scheduler.py:461` 的 `Scheduler` 类,文件头第 2 行原话是 `# modify from: https://github.com/vllm-project/vllm`。它的核心结构——`SchedulerSession`/`SchedulerSequence`、`block_manager` 分页分配、`block_trie` 前缀树匹配——和 vLLM V1 调度器（见 [[03-vLLM-调度器解剖]]）在概念骨架上高度相似:都是"每步从等待/运行两类请求里挑出这一步能塞进预算的子集,用分页 KV block 管理显存"。LMDeploy 自己也在源码注释里承认了血缘关系,这意味着:**同一家公司同一个代码库里,两个后端对"什么是连续批处理"给出了两种不同的实现**——TurboMind 是"槽位容器 + 异步流水线"的 C++ 原生设计,pytorch backend 是"每步重新调度"的 vLLM 同款设计。

### ④ KV cache 量化:从哪一行代码开始变成 INT4/INT8

以 TurboMind 为例,`UnifiedAttentionLayer`（`src/turbomind/models/llama/unified_attention_layer.cc`）构造时读 `quant_policy_`（`:137`）,决定 KV cache 每个元素存几个 bit（`:151` 的 `qaunt_bits = quant_policy_ ? quant_policy_ : dtype_bits`）。真正的量化数学在 `src/turbomind/kernels/attention/quantization.h`:

```cpp
// :341-366，节选核心逻辑
template<...>
__device__ void warp_stats(Array<P, 2> (&param)[S], const Array<T, N> (&x)[S][C], B n_bits) {
    // 对 warp 内的一组 KV 向量求 min / max
    ...
    const float scale = ((float)stats[1] - (float)stats[0]) * inv_q_max;
    param[s][0] = (P)scale;   // scale
    param[s][1] = (P)stats[0]; // zero point = min
}

template<...>
__device__ void quantize(Array<Q, N> (&dst)[S][C], const Array<T, N> (&src)[S][C],
                         const Array<P, 2> (&params)[S], B n_bits) {
    // (x - zero) / scale，再按 n_bits 饱和取整
    ...
}
```

**没有任何一行读取离线校准文件或模型 checkpoint 里预存的 scale**——scale 和 zero point 是每次写 KV cache 时,在这一小组（warp 内的若干 token）数据上现算出来的,量化完就地丢弃,下次再算。这是纯粹的**运行时动态非对称量化**,`ConvertKvCache<T, uint8_t>`/`ConvertKvCache<T, uint4_t>`（`:427`-`490`）把这套统计结果应用到实际的类型转换。

pytorch backend 有一份独立的平行实现:`lmdeploy/pytorch/kernels/cuda/fill_kv_cache.py:23`-`41` 的 `_quant_int8`/`_quant_int4`,是 Triton 版本,数学逻辑相同（除以 scale 加零点后取整),同样没有校准输入。两个后端各写了一遍同一套数学,分属两种 kernel 语言（CUDA C++ / Triton）——这本身就是"双引擎"要多付的维护税之一。

与之对照,LMDeploy 自己的**权重**量化（AWQ）走的是完全不同的路线:`lmdeploy/lite/quantization/calibration.py:19` 的 `CalibrationContext` 需要真的跑一批校准样本（`lmdeploy lite auto_awq` 命令),用 `ActivationObserver` 记录激活统计,离线算出量化参数再写进模型权重。KV cache 量化"零校准",权重量化"必须校准"——这是同一个项目里两种不同的量化哲学,原因是二者优化目标不同:权重量化图的是把误差降到最低（一次性代价,值得校准);KV cache 量化图的是"对任何输入都能开箱即用"（在线代价,校准反而不划算,因为 KV 内容本身就是高度动态的)。

### ⑤ `proxy.py`:一个统一入口背后的多实例路由

`lmdeploy serve proxy`（子命令注册在 `lmdeploy/cli/serve.py:186`,处理函数在 `:372`）启动一个独立的 FastAPI 应用（`lmdeploy/serve/proxy/proxy.py`）,核心是 `NodeManager`（`:71`）:

- `add_node`/`remove_node`（路由 `/nodes/add`、`/nodes/remove`,`proxy.py:493`/`:514`）动态注册/摘除后端 LMDeploy 实例,每个 `Node`（`:55`）带一个 `Status`,其中 `role: EngineRole`（`:48`,Hybrid/Prefill/Decode）——proxy 层面天生知道 PD 分离的角色划分。
- 请求到来时,`get_node_url()`（`:251`-`...`）按 `RoutingStrategy`（`lmdeploy/serve/proxy/utils.py:18`-`23`:`RANDOM`/`MIN_EXPECTED_LATENCY`/`MIN_OBSERVED_LATENCY`）三选一决定转发到哪个实例——`MIN_OBSERVED_LATENCY` 靠每个 `Status.latency`（一个定长 `deque`）滑动窗口实测延迟做决策,不是静态权重轮询。
- proxy 自己也重新实现了一份 `/v1/chat/completions`/`/v1/completions`（`:574`/`:747`）,内部转发给挑中的后端节点——对客户端来说,proxy 和单实例 `api_server` 的 API 面完全一样,多实例这件事是透明的。

它解决的问题很明确:**LMDeploy 没有把多实例负载均衡这件事完全甩给外部组件(nginx/k8s Service)**,而是自带了一个能感知 PD 角色、能按实测延迟路由的轻量代理。`## 6` 会对比 vLLM/SGLang 在同一问题上的选择。

## 5. 设计决策与代价

### 决策 1:为什么要养两套引擎(为什么/不这样/什么时候错)

- **为什么这么设计**:TurboMind 2023 年从 FasterTransformer fork 而来,目标是在一小撮当时最主流的架构(Llama 系)上榨干显存带宽和调度开销——自己写 CUDA/cutlass FMHA、自己管理 KV cache 池、自己实现 persistent batch,不依赖 PyTorch 运行时的算子分发开销。后来 DeepSeek-MoE、多模态、块扩散模型、EPLB 这些新范式接连出现,在 C++ 里手写每一种新架构的成本远高于 Python(`SUPPORTED_ARCHS` 白名单十来个架构、几年只涨到这个数,就是这层成本的证据),于是长出了 pytorch backend 去接住这些新东西,同时保留 TurboMind 服务已有的窄集合模型。
- **不这样会怎样**:若只留 TurboMind,DeepSeek 系列、块扩散模型、EPLB 场景全部无法部署,新架构接入周期以"重写一遍 C++ forward"计;若只留 pytorch backend,会失去 TurboMind 那条专门为极致时延设计的路径(显式槽位管理、无 Python 解释器开销的调度循环、自定义 FMHA)——多少收益本库不产出实测数字,但这条路径存在的唯一理由就是性能,砍掉它意味着放弃这个理由。
- **什么时候这个选择是错的**:当维护两套引擎的代价开始超过收益时。三个可观测的代价信号:(a) 配置面分裂,42/48 字段里概念对应不到 15 对(`## 3.1`);(b) 特性在两边漂移而不是同步——`TURBO_QUANT` 挂着"Turbo"的名字却只有 pytorch 实现,推测解码明确"not supported by turbomind"(`lmdeploy/serve/core/async_engine.py:145`-`146`),PD 分离的 `role`/`migration_backend` 只存在于 `PytorchEngineConfig`;(c) 统一的 API 表面之下藏着后端相关的隐性契约——`## 7` 会讲 `/distserve/*` 端点对 TurboMind 后端可能直接不可用。当用户已经分不清"我现在用的到底是哪个引擎、它支持什么"时,双引擎从"各展所长"变成了"认知负担",这正是本篇标题想说的"代价"。

### 决策 2:引擎选择"一边硬锁一边软选"

- **为什么这么设计**:pytorch backend 是万能兜底(覆盖面最广,几乎总能跑),所以用户显式传 `PytorchEngineConfig` 没必要再校验,直接锁定;TurboMind 是窄而快的特例,永远存在"这个模型架构它其实跑不了"的可能,所以哪怕用户传了 `TurbomindEngineConfig`,`autoget_backend_config()`(`lmdeploy/archs.py:72`-`91`)也要重新核验一遍架构支持情况。
- **不这样会怎样**:如果 `TurbomindEngineConfig` 也被当成硬锁,用户传了一个 TurboMind 不支持的模型 + `TurbomindEngineConfig`,要么在模型加载阶段才于 C++ 侧报错(比 Python 层的架构检查晚得多),要么该行为从一开始就没有防御,直接崩溃或产出不可预测结果。
- **什么时候这是错的**:当这份"善意的静默降级"本身造成误解时——用户以为自己在用 TurboMind(配置对象类型都传对了),实际请求跑在 pytorch backend 上,唯一线索是启动日志里一行 `logger.warning`(`lmdeploy/archs.py:41`-`43`)。如果日志级别被调到 `ERROR` 或没人盯着启动输出,这次降级完全不可见。对性能敏感、需要确定自己用上了 TurboMind 特性(比如它的 KV cache 检查点前缀缓存、`cache_prompt`/`cache_generation` 这套机制)的部署场景,这个静默 fallback 就是错的——理想情况下应该有一个"strict"选项让选型失败直接报错而不是自动降级(本库推断当前不存在这样的开关,**未查证**是否有相关 issue 在跟踪)。

### 决策 3:TurboMind persistent batch = 固定容量槽位,而非每步重建

- **为什么这么设计**:`Engine::Impl::State::rc`(`src/turbomind/engine/engine.cc:171`-`184`)作为一个跨步存在的容器,好处是请求的执行状态(`Sequence`)只需要在进入/离开批次时构造/析构一次,批内其余步骤只是"原地更新"——不需要像 vLLM 那样每次 `schedule()` 都重新遍历 `self.running`/`self.waiting` 两个队列、重新计算预算分配(见 [[03-vLLM-调度器解剖]] `## 4`)。配合异步流水线(engine 线程与 model executor 线程分离,`BatchData` 的多个 `phase` 允许调度提前于执行),TurboMind 能把"决定下一步跑什么"和"真正执行上一步"这两件事重叠起来。
- **不这样会怎样**:如果批次每步重建(像 vLLM 那样用轻量 dataclass `SchedulerOutput` 现场拼批),调度逻辑更容易推理、更容易加新的旁路功能(结构化输出、投机解码只需要在裁剪 `num_new_tokens` 的地方加一层判断,vLLM 的这套做法见同名文章`## 4.2`),但每步都要重新遍历请求集合、重新计算预算——对 TurboMind 这种以"单一模型族、极致吞吐"为目标的引擎,这份重建开销被认为不值得付。
- **什么时候可以不这样**:当引擎需要频繁支持新的旁路调度策略(投机解码的动态 K、结构化输出的语法约束、PD 分离的角色感知)时,vLLM 式"每步现场拼批"的架构更容易长出新分支而不破坏既有代码——这也是为什么 pytorch backend(承担了 LMDeploy 里几乎全部"新特性"的接入任务:PD 分离、EPLB、块扩散、推测解码)选择follow vLLM 的调度器结构,而不是照抄 TurboMind 的槽位模型。**槽位固定与现场重建不是谁更"对",是"极致吞吐的窄场景"与"快速迭代的宽场景"两种优先级的具体化**。

### 决策 4:KV 量化在配置层"共享值域、后端各自过滤"

- **为什么这么设计**:`QuantPolicy` 枚举统一定义(`lmdeploy/messages.py:20`-`27`),但 `TurbomindEngineConfig.__post_init__`(`:355`-`358`)显式拒绝 `FP8`/`FP8_E5M2`,`PytorchEngineConfig` 不拒绝——这样共享 CLI 的 `--quant-policy` 参数(`lmdeploy/cli/utils.py:256`-`277`,一份解析逻辑同时挂在 `pt_group` 和 `tb_group` 两个参数组下,`lmdeploy/cli/serve.py:133`/`:159`)不用为两个后端各写一份,配置对象的 `__post_init__` 各自负责本后端的合法性边界。
- **不这样会怎样**:如果每个后端都定义自己的量化枚举,CLI 层要么重复一份 `--quant-policy` 帮助文本,要么用户在两个后端之间切换时要记两套数字/名字映射——现在的设计下,`0/4/8/16/17/42` 这几个数字在哪个后端都是同一个意思,只是"支持不支持"的问题,不是"是什么"的问题。
- **什么时候可以不这样(即这个设计的边界在哪)**:当某个取值在配置层"合法"但在执行层完全没有实现时,这套"值域共享、行为分裂"的设计就会制造陷阱——`TURBO_QUANT`(42)正是这种情况:两边 `__post_init__` 都没有拦截它,但只有 pytorch backend 真正实现了对应 kernel。**本库推断**:一个用户在 TurboMind 后端上设置 `quant_policy=42`,配置构造阶段不会报错(`QuantPolicy(42)` 合法、不在 FP8/FP8_E5M2 黑名单里),但 `quant_bits = quant_policy_` 会把 42 当作"42 比特"传进 `BlockConfig` 相关的显存布局计算,`ConvertKvCache` 模板家族里没有任何针对 42 的特化——具体会在哪一步以什么形式报错,**未查证**(本机无 GPU,无法实际启动一次这个配置来复现)。这正是"配置层校验"和"执行层能力"没有绑在一起的典型坑,`## 8` 会把它列进可改进点。

### 决策 5:proxy.py 内建路由策略,而不是完全依赖外部负载均衡

- **为什么这么设计**:`NodeManager`(`lmdeploy/serve/proxy/proxy.py:71`)把"PD 角色感知"(`role: EngineRole`)和"按实测延迟路由"(`MIN_OBSERVED_LATENCY`,靠每个节点的滑动窗口延迟 `deque`)直接写进代理层,免去用户自己搭一层理解 PD 语义的负载均衡器——尤其是 PD 分离场景下,"这个请求该发给 Prefill 节点还是 Decode 节点"这件事本身就需要业务语义,通用型负载均衡器(nginx 轮询)做不到。
- **不这样会怎样**:如果完全依赖外部 LB(比如只提供 `/health` 让 nginx/k8s 探活、路由策略甩给基础设施层),LMDeploy 就不用维护 `proxy.py` 这 1,083 行代码和它自己的一套请求转发/流式响应重写逻辑(`lmdeploy/serve/proxy/streaming_response.py`)——但用户要么自己再实现一层 PD 角色感知的路由,要么放弃 PD 分离在多实例场景下的自动路由能力。
- **什么时候可以不这样**:单实例部署,或者多实例之间完全同构(没有 PD 角色区分、请求可以均匀分布)时,`proxy.py` 提供的路由策略价值有限,这时候更轻量的外部 LB(甚至 DNS 轮询)完全够用——`proxy.py` 的真正价值在 PD 分离 + 多实例这个交集场景,场景越窄,自建代理这层投入的性价比就越低。

## 6. 同位对照:vLLM / SGLang 在同一位置怎么做

| 维度 | LMDeploy | vLLM | SGLang |
|---|---|---|---|
| 引擎数量 | 2 套(TurboMind C++ + pytorch backend) | 1 套(Python 编排 + CUDA/Triton 算子) | 1 套(Python 编排 + Triton/CUDA 算子) |
| 批处理机制 | TurboMind:容量恒定的槽位容器 + 异步双线程流水线;pytorch backend:每步重建,结构上"modify from vllm" | 每次 `schedule()` 现场从 `waiting`/`running` 两个队列拼 `SchedulerOutput`(`vllm:vllm/v1/core/sched/scheduler.py:484`,详见 [[03-vLLM-调度器解剖]]) | 概念上同属"连续批处理"家族,调度器结构与 vLLM 接近(**未查证**具体实现细节,留给 [[02-SGLang-Scheduler事件循环]]) |
| KV cache 量化选项 | `NONE`/`INT4`/`INT8`/`FP8`/`FP8_E5M2`/`TURBO_QUANT`,**运行时动态量化,零校准**(`src/turbomind/kernels/attention/quantization.h:341`-`366`) | 经典路径是**校准出的静态 per-tensor scale**(`` `vllm:vllm/model_executor/layers/quantization/kv_cache.py:186` `` 的 "Using uncalibrated q_scale" 告警印证这一点);较新的 `CacheDType` 列表(`` `vllm:vllm/config/cache.py:39` ``)里也出现了 `int4_per_token_head`/`int8_per_token_head`(`:52`-`53`)与 `turboquant_k8v4` 等选项——vLLM 正在把 per-token 动态量化和"TurboQuant"这个量化方案本身也纳入自己的 `CacheDType` 里,**这条赛道上"LMDeploy 独有 INT4/INT8"这句话正在过时**(本库推断,依据是这两处字面量确实存在于当前快照,但未逐行核实其运行时是否也是零校准) | `kv_cache_dtype` 支持 `fp8_e5m2`/`fp8_e4m3`/`mxfp8`/`bf16`/`nvfp4` 等(`` `sglang:python/sglang/srt/server_args.py:705` ``),**没有** INT4/INT8 选项;`quantization_param_path`(`:692`)明确写着"KV cache dtype 是 FP8 时通常需要提供这个校准文件,否则 scale 默认 1.0 可能有精度问题"——同样是"静态校准优先"的路线 |
| 多实例路由 | 内建 `proxy.py`(FastAPI app,`lmdeploy serve proxy` 一条命令起,RANDOM/MIN_EXPECTED_LATENCY/MIN_OBSERVED_LATENCY 三种策略,天然感知 PD 角色) | 有 `DPSupervisor`(`vllm/entrypoints/openai/dp_supervisor.py`)负责多端口起多个数据并行副本,但函数命名里的 `infer_multi_port_external_lb_start_rank`/`validate_multi_port_external_lb_args` 暗示路由决策本身仍交给**外部**负载均衡器,vLLM 侧只做进程管理和健康探活(本库推断,依据是函数名里的 `external_lb` 字样,**未逐行核实**是否存在内建路由策略) | 有独立的 `sgl-router`(Rust crate,`_src/sglang/experimental/sgl-router/`),与主 Python server 分属不同代码库/语言,**未查证**其具体路由策略与 LMDeploy `RoutingStrategy` 的对应关系 |
| API 协议覆盖 | OpenAI 兼容(8 条 `/v1/*`)+ Anthropic 兼容(`lmdeploy/serve/anthropic/endpoints/messages.py:70` 等 3 条)+ 自有 `/generate`/`/get_ppl` | 同样做了 OpenAI 兼容,且这份快照里 vLLM 也已经有 `vllm/entrypoints/anthropic/api_router.py`——Anthropic 兼容**不是** LMDeploy 独有(本库推断:两边都已实现,行业趋同,**未查证**谁先做的) | OpenAI 兼容为主,**未查证**是否已有 Anthropic 兼容端点 |

三点小结:①"KV int4/int8 量化"曾经是 LMDeploy 最鲜明的招牌,但从这份快照看,vLLM 已经在追平这条路径,**差异化正在被时间抹平**;②多实例路由这件事上,LMDeploy 把它做成了自己代码库里的一等公民,vLLM/SGLang 则倾向于外置(交给外部 LB 或独立 Rust 组件)——这也是"要不要多养一块代码"的同类权衡,只是发生在不同的子系统上;③连续批处理这件事,LMDeploy 自己的两个后端就出现了两种不同答案,读者不需要再去 vLLM/SGLang 找对照——**LMDeploy 内部本身就是一个绝佳的对照组**。

## 7. 踩坑与反直觉

1. **传了 `TurbomindEngineConfig` 不代表一定用 TurboMind。** `autoget_backend_config()`(`lmdeploy/archs.py:72`-`91`)只在检测到 `PytorchEngineConfig` 时硬锁,`TurbomindEngineConfig` 仍要过一遍架构白名单核验,不支持就静默换成 pytorch,字段值搬过去,只留一行 warning。第一次读代码容易以为"传对象类型 = 选定后端",实际上只有一个方向(锁定 pytorch)成立。

2. **`TURBO_QUANT` 这个名字最容易让人以为它是 TurboMind 专属。** 恰恰相反,`src/turbomind/` 全目录搜不到 `TURBO_QUANT`/`TurboQuant`/`QJL4` 任何字样,它只在 `lmdeploy/pytorch/kernels/cuda/fill_kv_cache.py` 里有实现。配置层的 `QuantPolicy` 枚举定义(`lmdeploy/messages.py:20`-`27`)对两个后端一视同仁,但真正的算子实现完全不对称。

3. **PD 分离(`/distserve/*`)的路由注册和后端能力可能对不上。** `lmdeploy/serve/openai/endpoints/distserve.py:16`-`31` 的 `engine_info()` 读取 `engine_config.dp_rank`/`engine_config.block_size`/`engine_config.num_cpu_blocks`/`engine_config.num_gpu_blocks`——这四个字段全部只存在于 `PytorchEngineConfig`,`TurbomindEngineConfig` 的 42 个字段里没有一个同名项(TurboMind 用的是 `cache_block_seq_len`/`cache_chunk_size`,没有显式的 `num_cpu_blocks`/`num_gpu_blocks`,也没有 `dp_rank`)。这组端点在通用 `api_server` 里对所有后端一视同仁地注册,**但源码读出的字段只对 pytorch 后端存在**——`lmdeploy/serve/core/async_engine.py:232` 里 `dp_rank = self.backend_config.dp_rank if self.backend == 'pytorch' else 0` 这一行本身就是对这层不对称的显式补丁。**本库推断**:若用 TurboMind 后端起服务并调用 `/distserve/engine_info`,大概率会在读取 `dp_rank` 等属性时抛 `AttributeError`,但**未实测验证**(本机无 GPU,无法真正跑一次这个组合)。

4. **推测解码明确"turbomind 不支持",但配置层不拦截。** `lmdeploy/serve/core/async_engine.py:145`-`146` 的判断是 `if speculative_config is not None and backend == 'turbomind': logger.warning('speculative decoding is not supported by turbomind ')`——只警告,不报错,也不强制切换后端。用户传了 `speculative_config` 又用 TurboMind,大概率是"配置被静默忽略",而不是"程序拒绝执行"。

5. **KV cache 量化"零校准"不等于"没有精度代价"。** `warp_stats()`(`src/turbomind/kernels/attention/quantization.h:341`-`366`)在一个 warp 范围内(几十个 token 量级)现算 min/max——统计窗口天然比全量校准集小得多,离群值更容易把 scale 拉宽、压缩有效量化区间。本篇没有做也不会做精度实测(本机无 GPU),只从源码结构指出:**"零校准"是"部署简单"和"统计窗口小"之间的权衡,不是免费的**。

6. **`_lab/out/struct_map.json` 的 kv_cache/quantization 子系统统计口径完全看不见 C++ 代码。** `struct_map.py` 基于 Python AST 解析,`kv_cache` 子系统统计到的 5,180 行、`quantization` 子系统统计到的 8,419 行,全部落在 `lmdeploy/pytorch/` 下——`src/turbomind/kernels/attention/quantization.h` 这类 C++ 头文件不进这个统计。读这两个数字时容易误以为"LMDeploy 的量化代码主要在 pytorch 里",实际只是统计工具的语言盲区,C++ 侧同样有一整套独立的量化实现,只是没被这个特定脚本数进去。

## 8. 可改进点

(以下均为本库基于源码走读的推断,标注证据依据;**未核实**是否已有官方 issue 在跟踪。)

**改进点 1:引擎选型缺一个"strict"模式**
- 现状:`autoget_backend_config()`(`lmdeploy/archs.py:54`-`91`)遇到不支持的架构总是静默降级,唯一线索是一行 `logger.warning`。
- 影响:性能敏感场景下,用户可能一直以为自己用的是 TurboMind,直到某天检查启动日志才发现请求全程跑在 pytorch backend 上——这类"性能预期落空"的问题排查成本很高,因为服务本身运行正常,不会报错、不会崩溃。
- 建议方向:加一个 `strict_backend: bool` 之类的开关,设为真时,`TurbomindEngineConfig` 遇到不支持的架构应该直接抛异常而不是降级——把"要不要接受降级"的决定权交还给调用方。

**改进点 2:`quant_policy` 的配置层校验和执行层能力没有对齐**
- 现状:`TurbomindEngineConfig.__post_init__`(`lmdeploy/messages.py:352`-`358`)只拦截了 `FP8`/`FP8_E5M2`,没有拦截 `TURBO_QUANT`(42)——而 `src/turbomind/` 全目录没有任何 `TURBO_QUANT` 相关实现。
- 影响:用户在 TurboMind 后端设置 `--quant-policy turbo_quant` 能通过配置构造阶段的校验,失败会发生在更晚、更难定位的地方(`## 5` 决策 4 已展开,**未查证**具体报错形式)。
- 建议方向:`TurbomindEngineConfig.__post_init__` 的黑名单应该和 C++ 侧实际支持的 `quant_policy` 取值同步维护,理想情况下由一个共享的"每个后端支持哪些 QuantPolicy"的映射表驱动,而不是在 Python 配置类里手写排除项。

**改进点 3:`/distserve/*` 端点的后端假设没有在路由注册层显式声明**
- 现状:`lmdeploy/serve/openai/endpoints/distserve.py` 的端点无条件注册进通用 `api_server`,内部却直接访问只有 `PytorchEngineConfig` 才有的字段(`## 7` 踩坑 3)。
- 影响:TurboMind 后端的用户理论上能在 API 文档/OpenAPI schema 里看到这些端点,调用后才发现不可用——这是一种"文档可见但实际不可用"的接口设计,容易造成困惑。
- 建议方向:在路由注册阶段按 `backend` 条件性挂载这组端点(FastAPI 支持基于条件动态 `include_router`),或者至少在 `engine_info()` 里对非 pytorch 后端提前用一个明确的 4xx 错误替代 `AttributeError`。

**改进点 4:两套 KV 量化 kernel(CUDA C++ / Triton)没有共享的正确性对拍**
- 现状:`src/turbomind/kernels/attention/quantization.h` 的 `warp_stats`/`quantize` 和 `lmdeploy/pytorch/kernels/cuda/fill_kv_cache.py` 的 `_quant_int8`/`_quant_int4` 各自独立实现同一套"非对称 min/max 量化"数学,分属两种 kernel 语言。
- 影响:数学逻辑的任何一次修正(比如量化舍入方式、饱和边界处理)都需要在两处分别改、分别测——本篇没有找到任何脚本或测试用例交叉比对两个后端在相同输入下的量化输出是否一致(**未查证**是否存在,只是没在 `tests/` 目录下看到明显对应的对拍测试)。
- 建议方向:哪怕不合并实现,至少加一组"同一 KV 张量输入,分别过两个后端的量化路径,比对反量化误差是否落在同一量级"的测试,把"两边应该数学等价"这条隐性假设显式测出来。

## 9. 自测题与延伸阅读

**闭卷自测题**(合上本文,尝试不看源码回答):

1. `TurbomindEngineConfig` 有 42 个字段,`PytorchEngineConfig` 有 48 个,概念完全对应的大约有多少对?两边各自独有的字段分别集中在哪几类能力上?
2. 显式传 `PytorchEngineConfig` 和显式传 `TurbomindEngineConfig`,在引擎选择这件事上,两者的"锁定力度"是否对称?为什么不对称?
3. `QuantPolicy.TURBO_QUANT`(值 42)这个名字容易让人误以为它属于哪个引擎?实际实现在哪个文件?
4. TurboMind 的 KV cache INT4/INT8 量化需不需要离线校准?量化的 scale/zero point 是在哪一层函数、按什么粒度现算出来的?
5. `/distserve/engine_info` 这个 HTTP 端点读取的字段,是两个后端配置都有,还是只有其中一个有?
6. TurboMind 的"persistent batch"和 vLLM 的连续批处理,在"批"这个概念的生命周期上有什么本质区别?
7. LMDeploy 自己的 pytorch backend 调度器(`lmdeploy/pytorch/paging/scheduler.py`)和 TurboMind 的调度模型,哪一个在源码注释里承认了与 vLLM 的血缘关系?
8. `proxy.py` 的三种路由策略分别是什么?`MIN_OBSERVED_LATENCY` 依据什么数据做决策?
9. vLLM 当前快照的 `CacheDType` 里出现了哪些和 LMDeploy KV 量化命名相似的选项?这说明"INT4/INT8 KV 量化是 LMDeploy 独有优势"这句话现在还成立吗?
10. 推测解码在 LMDeploy 的哪个后端不被支持?这件事是通过报错还是警告体现的?

(题 1-2 对应 `## 3.1`、`## 4` ①、`## 5` 决策 2;题 3 对应 `## 5` 决策 4、`## 7` 踩坑 2;题 4 对应 `## 4` ④;题 5 对应 `## 7` 踩坑 3;题 6-7 对应 `## 4` ②③、`## 5` 决策 3、`## 6`;题 8 对应 `## 4` ⑤;题 9 对应 `## 6`;题 10 对应 `## 7` 踩坑 4——答不上来就回对应小节重读,不用从头翻。)

**延伸阅读(本库内)**:

- [[03-vLLM-调度器解剖]] —— 本篇 `## 5` 决策 3、`## 6` 反复对照的"每步现场重建批次"式连续批处理,是理解 TurboMind persistent batch 差异化定位的参照系
- [[04-工程规模与代码结构对比]] —— LMDeploy 339,214 总行数、130 个 CUDA 文件、34 个 Triton 文件在全部 12 个引擎里处于什么位置,本篇 `## 2` 引用的 `_lab/out/repo_stats.json` 数字在这里有横向坐标
- [[05-选型决策树]] —— "什么场景选 LMDeploy、选它的哪个后端"这个决策本身,本篇只讲清楚了两个后端各自的能力边界,没有给出选型建议,留给这篇整合

---

读完本篇应该能回答的一句话总结:LMDeploy 的"双引擎"不是营销话术,是两套配置表、两套调度模型、两套 KV 量化 kernel 的真实并存——TurboMind 用固定容量的槽位容器和异步双线程流水线换极致性能,pytorch backend 照抄 vLLM 的调度器骨架换架构覆盖面,统一的 API 表面之下,能力边界一直在,只是被藏进了配置字段的有无、CLI 警告的字面、以及"这个端点读的字段另一个后端根本没有"这类需要读源码才能发现的细节里。
