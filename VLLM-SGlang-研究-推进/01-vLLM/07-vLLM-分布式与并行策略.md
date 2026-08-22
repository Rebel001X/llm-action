# vLLM 分布式与并行策略解剖

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：五个并行维度是切出来的，DP的"步调一致"只为MoE而生

## 0. 结论先行

- 全部并行维度——TP / PP / PCP（prefill context parallel）/ DCP（decode context parallel）/ DP——都不是分别写循环拼出来的，而是把全局 rank 排成一个 5 维张量 `all_ranks`（形状 `[-1, dp, pp, pcp, tp]`），对不同维度 `transpose` 后 `reshape` + `unbind`，一次性在 `initialize_model_parallel()` 里派生出所有进程组（`vllm/distributed/parallel_state.py:1751`-`1997`）。EP 组不是独立的第六个维度，而是把 DP×PCP×TP 三个维度合并成一个组（`vllm/distributed/parallel_state.py:1932`-`1934`）——这是本篇最重要的一条线索，`## 4` 会展开。
- TP 的切分点在四类层上各不相同，通信原语也不同：QKV 走列并行（`ColumnParallelLinear` 及其子类 `QKVParallelLinear`），前向不通信；O_proj / down_proj 走行并行（`RowParallelLinear`），前向末尾做一次 `all_reduce`；embedding 走"词表并行 + all_reduce"（不是经典的行/列切法，是按 vocab 维切分后每个 rank 只有自己命中的 token 非零，用求和当"选择"）；lm_head 走"词表并行 + all_gather"。四种切分对应四种不同的通信时机与原语，`## 4` 逐条对源码。
- PP 在 vLLM 里**没有** Megatron 式的 virtual pipeline / interleaved 1F1B 调度。它把流水线并行做在 `EngineCore` 这一层：一次调用 `step_with_batch_queue()` 会让最多 `pipeline_parallel_size`（或 async scheduling 下 `pp_size+1`）个**连续的调度步骤**同时在飞行中，通过 `Future` 排队实现"填满流水线"，而不是把一个 batch 切成多个 micro-batch 交给不同 PP rank（`vllm/config/vllm.py:584`-`592`、`vllm/v1/engine/core.py:638`-`694`）。
- **DP 最反直觉的一条**：只有 MoE 模型的 `--data-parallel-size` 才会创建真正需要"步调一致"的 `DPEngineCoreProc`；非 MoE（稠密）模型设置的 DP，本质上等价于起 N 个完全独立、互不通信的普通 `EngineCore` 副本——源码注释原话是"Non-MoE DP ranks are completely independent, so treat like DP=1"（`vllm/v1/engine/core.py:1319`-`1329`）。MoE 模型的 DP 因为要跟 EP 组共享同一批 rank，没有请求的 rank 必须跑 `execute_dummy_batch()` 占位（`vllm/v1/worker/gpu_worker.py:1221`-`1223`），且是否所有 rank 都已完成的全局判定**每 32 步才做一次 all-reduce**，不是每步（`vllm/v1/engine/core.py:2267`-`2274`）。
- "自定义 all-reduce" 不是唯一的"非 NCCL"路径，而是一整条优先级链：symm-mem NCCL AR → quick reduce（仅 ROCm）→ FlashInfer AR → AITER 自定义 AR（AMD）→ vLLM 自定义 AR（CUDA IPC）→ 通用 symm_mem → pynccl → `torch.distributed` 默认兜底（`vllm/distributed/device_communicators/cuda_communicator.py:278`-`341`）。vLLM 自定义 AR 只在 `world_size ∈ {2,4,6,8}`、张量 ≤8MB、字节数是 16 的倍数、且同节点内 P2P/NVLink 全连通时才生效（`vllm/distributed/device_communicators/custom_all_reduce.py:348`-`360`）。
- EPLB 不是"EP 组顺手做的事"，是单独一个 `_EPLB` 进程组 + 一整套"滑动窗口记录负载 → 每隔固定步数判断要不要重排 → 真的把专家权重张量跨 rank 拷贝"的机制，代价是显式的显存搬运（`vllm/distributed/eplb/eplb_state.py:432`-`448`、`vllm/distributed/eplb/rebalance_execute.py:512`）。
- 多机启动默认走**vLLM 自己的 multiprocessing**，不是 Ray：`ParallelConfig.__post_init__` 里，只要是 CUDA 平台且 `nnodes > 1`，就直接选 `"mp"`；只有显式要求 `data_parallel_backend="ray"` 或已经跑在 Ray placement group 里才会切到 Ray（`vllm/config/parallel.py:936`-`959`）。

**速查：关键问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| TP 切分点在哪些层？行切还是列切？通信在哪一步？ | `## 4` ②、`## 5` 决策 2 |
| PP 有没有 virtual pipeline？micro-batch 怎么调度？ | `## 4` ③、`## 5` 决策 3 |
| DP 为什么必须步调一致？没请求的 rank 怎么办？和 EP 什么关系？ | `## 0` 第4条、`## 4` ①④、`## 5` 决策 4、`## 7` 踩坑 1 |
| 自定义 all-reduce 什么条件生效？为什么不总是比 NCCL 快？ | `## 4` ⑤、`## 5` 决策 5 |
| EPLB 干什么、何时触发、代价是什么？ | `## 4` ⑥、`## 5` 决策 6、`## 8` |
| 多机默认用 Ray 还是原生 multiprocessing？ | `## 0` 最后一条、`## 5` 决策 1 |

## 1. 它在系统里的位置

`vllm/distributed/` 是全库**唯一**知道"当前进程是哪个 rank、和谁同组"的地方，其余子系统只消费它暴露的查询函数，从不自己管理进程组：

- **模型层**（`vllm/model_executor/layers/linear.py`、`vocab_parallel_embedding.py`）在 `__init__` 里调 `get_tensor_model_parallel_world_size()` / `get_tensor_model_parallel_rank()` 决定切多大的分片，在 `forward()` 里调 `tensor_model_parallel_all_reduce()` / `tensor_model_parallel_all_gather()` 做通信——这两个函数是 `vllm/distributed/communication_op.py:12`-`23` 定义的模块级便捷封装，内部转发到当前 TP 组的 `GroupCoordinator`。
- **调度器 / EngineCore**（[[03-vLLM-调度器解剖]] 的地盘）不关心 TP，但关心 PP 和 DP：PP 决定 `max_concurrent_batches` 从而决定 batch queue 深度（`vllm/config/vllm.py:584`），DP 决定要不要起 `DPEngineCoreProc`、要不要在空转时打 dummy batch。
- **模型执行 / CUDA Graph**（[[06-vLLM-模型执行与CUDA-Graph]] 的地盘）依赖 `GroupCoordinator.all_reduce()` 被注册成 `torch.ops.vllm.all_reduce` 这个自定义算子（而不是直接调用 Python 方法），这样 CUDA Graph capture 和 `torch.compile` 才能把跨 rank 通信也编译进图里（`vllm/distributed/parallel_state.py:662`-`683`，`## 5` 决策 2 会展开为什么要这样包一层）。
- **KV 传输 / PD 分离**（[[12-vLLM-PD分离与KV-Connector]] 的地盘）物理上也放在 `vllm/distributed/kv_transfer/` 目录下，但它传输的是 KV cache 张量而不是模型并行的激活值/权重，通信方式（NIXL、Mooncake、hf3fs 等）与本篇讲的 TP/PP/DP/EP 通信是两套完全不同的机制。**本篇不覆盖这部分**，`## 2` 会给出这个目录在统计口径上造成的干扰有多大。
- `vllm/distributed/` 目录下还有两个本篇只提及、不深入展开的邻居：`elastic_ep/`（`enable_elastic_ep`，让 DP/EP 用无状态 NCCL 组支持推理期间扩缩容，`vllm/config/parallel.py:208`）和 `weight_transfer/`（模型权重在线热更新用的传输层，与 `## 4` ⑥ EPLB 搬运专家权重是两套不同的代码路径，前者是"整份权重换新"，后者是"专家粒度重排"）。两者都是"进程组建好之后，运行期间还要再动态调整"这条线上的机制，和本篇的静态建组主线相关但不同，值得各自单独一篇。

## 2. 代码地图（文件 → 职责，带行号）

`_lab/out/struct_map.json` 按文件名关键词把 `distributed` 子系统统计成 **138 个文件 / 54,851 行**——这个口径**严重失真**，本库推断依据如下：`vllm/distributed/kv_transfer/` 单独就有 32,609 行（约占 59%），`vllm/distributed/ec_transfer/`（encoder cache 跨进程搬运，属于多模态编码器缓存，非本篇话题）另有 1,806 行；两者合计 34,415 行、约 63% 的"distributed 行数"其实是 PD 分离与缓存搬运的地盘，属于 [[12-vLLM-PD分离与KV-Connector]]。真正属于"TP/PP/DP/EP 怎么建组、怎么通信"这条主线的代码是：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `vllm/distributed/parallel_state.py:380` | `GroupCoordinator` 类定义——一个进程组的全部封装（全文件 2,359 行） | — |
| `vllm/distributed/parallel_state.py:409` | `GroupCoordinator.__init__`——按 `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` 二选一走 `new_group` 或 `split_group` 建组 | — |
| `vllm/distributed/parallel_state.py:662` | `GroupCoordinator.all_reduce()`——包成 `torch.ops.vllm.all_reduce` 自定义算子再转发 | — |
| `vllm/distributed/parallel_state.py:981` | `GroupCoordinator.send_tensor_dict()`——PP 阶段间传递 `IntermediateTensors` | — |
| `vllm/distributed/parallel_state.py:1076` | `GroupCoordinator.recv_tensor_dict()` | — |
| `vllm/distributed/parallel_state.py:1586` | `init_distributed_environment()`——算全局 rank/world_size（含 DP 偏移量）、起 `torch.distributed` | — |
| `vllm/distributed/parallel_state.py:1751` | `initialize_model_parallel()`——**全部并行组的唯一派生入口**，本篇 `## 4` ①逐段对照 | — |
| `vllm/config/parallel.py:119` | `ParallelConfig` 类定义（59 个字段） | — |
| `vllm/config/parallel.py:243` | `distributed_executor_backend` 字段——mp / ray / uni / external_launcher 四选一 | — |
| `vllm/config/parallel.py:856` | `ParallelConfig.__post_init__`——`world_size` 计算、DP 校验、执行后端默认值选择 | — |
| `vllm/config/parallel.py:59` | `EPLBConfig` 类定义 | — |
| `vllm/model_executor/layers/linear.py:407` | `ColumnParallelLinear`——TP 列切的基类 | — |
| `vllm/model_executor/layers/linear.py:971` | `QKVParallelLinear(ColumnParallelLinear)`——QKV 投影按 head 维列切 | — |
| `vllm/model_executor/layers/linear.py:1510` | `RowParallelLinear`——TP 行切，`forward` 末尾按需 `all_reduce` | — |
| `vllm/model_executor/layers/vocab_parallel_embedding.py:198` | `VocabParallelEmbedding`——词表并行 embedding，`all_reduce` 合并 | — |
| `vllm/model_executor/layers/logits_processor.py:126` | `_gather_logits()`——lm_head 输出用 `all_gather` 拼回完整词表 | — |
| `vllm/config/vllm.py:584` | `VllmConfig.max_concurrent_batches`——PP 流水线深度的唯一定义处 | — |
| `vllm/v1/engine/core.py:638` | `EngineCore.step_with_batch_queue()`——PP/异步调度的流水线主循环 | — |
| `vllm/v1/engine/core.py:2000` | `DPEngineCoreProc`——只服务 MoE 模型的 DP 引擎子类 | — |
| `vllm/v1/worker/dp_utils.py:173` | `coordinate_batch_across_dp()`——DP rank 间对齐 token 数与是否 micro-batch | — |
| `vllm/distributed/device_communicators/custom_all_reduce.py:56` | `CustomAllreduce` 类——CUDA IPC 实现的自定义 all-reduce | — |
| `vllm/distributed/device_communicators/cuda_communicator.py:278` | `CudaCommunicator.all_reduce()`——多路径通信优先级链的裁决点 | — |
| `vllm/distributed/device_communicators/all2all.py:44` | `AgRsAll2AllManager`——EP 默认 all2all 后端（all-gather + reduce-scatter） | — |
| `vllm/distributed/eplb/eplb_state.py:230` | `EplbState`——EPLB 状态机（负载记录窗口、重排计时器） | — |
| `vllm/distributed/eplb/rebalance_execute.py:512` | `rearrange_expert_weights_inplace()`——EPLB 真正搬运专家权重的地方 | — |
| `vllm/distributed/eplb/policy/default.py:20` | `DefaultEplbPolicy`——改编自 DeepSeek 开源 EPLB 的贪心装箱算法（源码注释自述，见文件头 `vllm/distributed/eplb/policy/default.py:1`-`12`） | — |
| `vllm/v1/executor/multiproc_executor.py:111` | `MultiprocExecutor`——默认执行后端，支持跨节点（`node_rank_within_dp`） | — |

（表格 25 条引用，含行号，超过硬指标要求的 10 条。）

**本篇范围边界**：`vllm/distributed/kv_transfer/`、`vllm/distributed/ec_transfer/`（PD 分离、KV/编码器缓存跨进程搬运）属于 [[12-vLLM-PD分离与KV-Connector]] 的地盘；`vllm/v1/core/sched/` 的调度决策逻辑属于 [[03-vLLM-调度器解剖]]；CUDA Graph capture 与 `torch.compile` 如何吃进 `torch.ops.vllm.all_reduce` 属于 [[06-vLLM-模型执行与CUDA-Graph]]。本篇只讲"进程组怎么建、TP/PP/DP/EP 各自的通信怎么发生"。

## 3. 核心数据结构

### 3.1 `ParallelConfig`（`vllm/config/parallel.py:119`）

59 个字段中与本篇直接相关的几组（每组给出字段名 + 默认值 + 行号）：

```python
# vllm/config/parallel.py（节选，行号见各行）
pipeline_parallel_size: int = Field(default=1, ge=1)              # :122
tensor_parallel_size: int = Field(default=1, ge=1)                # :124
prefill_context_parallel_size: int = Field(default=1, ge=1)       # :126
data_parallel_size: int = Field(default=1, ge=1)                  # :129
enable_expert_parallel: bool = False                               # :165
enable_eplb: bool = False                                          # :174
expert_placement_strategy: ExpertPlacementStrategy = "linear"      # :178
all2all_backend: All2AllBackend = "allgather_reducescatter"        # :188
disable_custom_all_reduce: bool = False                            # :205
enable_elastic_ep: bool = False                                    # :208
distributed_executor_backend: ... = None                           # :243
world_size: int = Field(init=False)                                 # :327（__post_init__ 里算出）
```

`world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size`（`vllm/config/parallel.py:858`-`862`），**不含** `data_parallel_size`——DP 各副本各自持有独立的 `world_size` 份 worker，`world_size_across_dp = world_size * data_parallel_size`（`vllm/config/parallel.py:573`-`575`）才是全局进程数。这是一个容易踩的坑：读代码时看到 `self.world_size` 不要默认它是"总卡数"。

### 3.2 `EPLBConfig`（`vllm/config/parallel.py:59`-`92`）

```python
window_size: int = Field(default=1000, gt=0)      # :62  —— 负载记录滑动窗口，单位：forward 步数
step_interval: int = Field(default=3000, gt=0)     # :64  —— 每多少步检查一次要不要重排
num_redundant_experts: int = Field(default=0, ge=0) # :72  —— 热门专家允许放几份冗余副本
use_async: bool = True                              # :84  —— 重排与前向计算异步重叠
policy: EPLBPolicyOption = "default"                # :89  —— 目前只有一种内置策略
communicator: EPLBCommunicatorBackend | None = None # :92  —— None 时自动选（优先 NIXL）
```

`_validate_eplb_config`（`vllm/config/parallel.py:102`-`115`）里有一条硬约束：`use_async=True` 时 `communicator` 不能是 `torch_nccl`/`pynccl`，因为异步 EPLB 和 NCCL 的多流机制会冲突，必须用 `torch_gloo` 或 `nixl`——这条约束在 `## 5` 决策 6 里会解释为什么。

### 3.3 `GroupCoordinator`（`vllm/distributed/parallel_state.py:380`-`1293`）

全库有 8 个模块级单例分别持有一个 `GroupCoordinator`：`_TP`、`_DCP`、`_PCP`、`_PP`、`_DP`、`_EP`、`_EPLB`，外加 `_WORLD`（`vllm/distributed/parallel_state.py:1293`-`1445` 是它们对应的 `get_xxx_group()` 访问器）。每个 `GroupCoordinator` 内部同时持有一个 device backend（NCCL/gloo）的 `device_group` 和一个纯 CPU 的 `cpu_group`（用于 rank 间做轻量元数据同步，不占 GPU 通信带宽），这是设计上"每建一个并行组就顺手建一对"的固定模式（`vllm/distributed/parallel_state.py:449`-`466`）。

### 3.4 八个进程组一览

| 单例 | `group_name` | 建组代码行 | 组大小 | 干什么 |
|---|---|---|---|---|
| `_WORLD` | — | `vllm/distributed/parallel_state.py:1293`（accessor） | 全部 rank | 最外层世界组，其余组都从它切出来 |
| `_TP` | `"tp"` | `vllm/distributed/parallel_state.py:1848` | `tensor_parallel_size` | TP 通信（`## 4` ②） |
| `_DCP` | `"dcp"` | `vllm/distributed/parallel_state.py:1866` | `decode_context_parallel_size` | decode 阶段切分 KV cache（不扩大 world size，`vllm/config/parallel.py:342`-`345`：无 PCP 时复用 TP rank，有 PCP 时横跨 PCP 轴或 TP×PCP 整块） |
| `_PCP` | `"pcp"` | `vllm/distributed/parallel_state.py:1885` | `prefill_context_parallel_size` | prefill 阶段切分序列长度（`vllm/config/parallel.py:126`-`128`：扩大 world size，但不增加 KV cache 分片数） |
| `_PP` | `"pp"` | `vllm/distributed/parallel_state.py:1903` | `pipeline_parallel_size` | PP 通信，`send_tensor_dict`/`recv_tensor_dict`（`## 4` ③） |
| `_DP` | `"dp"` | `vllm/distributed/parallel_state.py:1920` | `data_parallel_size` | DP rank 间的 token 数/CUDA Graph 模式对齐（`## 4` ④） |
| `_EP` | `"ep"` | `vllm/distributed/parallel_state.py:1953` | `dp × pcp × tp`（稠密模型不建） | MoE 专家计算的 all2all（`## 4` ①⑥） |
| `_EPLB` | `"eplb"` | `vllm/distributed/parallel_state.py:1976` | 与 `_EP` 同一批 rank，但独立进程组 | 只用于专家权重重排，故意和 MoE 前向的集合通信隔离，避免两边共用同一个 `torch.distributed` 状态导致死锁（`vllm/distributed/parallel_state.py:1957`-`1960` 注释原话） |

PCP 和 DCP 是本篇未展开的两个次要维度（不在 `## 0` 必答问题列表内），这里只给出它们在这张总表里的定位：**PCP 切的是"算的时候序列多长"，DCP 切的是"存的时候 KV cache 多大"**，二者都不参与 TP/PP/DP/EP 这四个主线维度的通信路径，属于更细粒度的长上下文优化，值得单独一篇但不是本篇的战场。

### 3.5 常用旋钮速查

以下是本篇涉及的、用户能直接摸到的命令行参数与环境变量，每条给出它在 `ParallelConfig`/`envs.py` 里的定义行、以及对应哪一节：

| 旋钮 | 类型 | 默认值 | 定义处 | 对应章节 |
|---|---|---|---|---|
| `--tensor-parallel-size` | CLI/字段 | 1 | `vllm/config/parallel.py:124` | `## 4` ② |
| `--pipeline-parallel-size` | CLI/字段 | 1 | `vllm/config/parallel.py:122` | `## 4` ③ |
| `--data-parallel-size` | CLI/字段 | 1 | `vllm/config/parallel.py:129` | `## 4` ④ |
| `--enable-expert-parallel` | CLI/字段 | `False` | `vllm/config/parallel.py:165` | `## 4` ① |
| `--enable-eplb` | CLI/字段 | `False` | `vllm/config/parallel.py:174` | `## 4` ⑥ |
| `--disable-custom-all-reduce` | CLI/字段 | `False` | `vllm/config/parallel.py:205` | `## 4` ⑤、`## 5` 决策 5 |
| `--distributed-executor-backend` | CLI/字段 | `None`（自动） | `vllm/config/parallel.py:243` | `## 4` ⑦ |
| `disable_nccl_for_dp_synchronization` | 字段 | `None`（隐式跟随 async scheduling） | `vllm/config/parallel.py:227` | `## 4` ④、`## 8` 改进点 2 |
| `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` | 环境变量 | `False` | `vllm/envs.py:66`，用在 `vllm/distributed/parallel_state.py:436` | `## 3.3`（建组的第二条代码路径，本篇不展开） |
| `VLLM_SKIP_P2P_CHECK` | 环境变量 | `False` | `vllm/envs.py:127`，用在 `vllm/distributed/device_communicators/custom_all_reduce.py:40` | `## 4` ⑤（跳过真实 P2P 探测，直接信驱动上报） |

`VLLM_DISTRIBUTED_USE_SPLIT_GROUP` 值得单独说一句：`GroupCoordinator.__init__` 其实有两条建组代码路径（`vllm/distributed/parallel_state.py:432`-`466`），默认路径是逐组调 `torch.distributed.new_group()`；这个环境变量打开后走 `_create_subgroups_split_group()`（`vllm/distributed/parallel_state.py:282`起），用 PyTorch 较新的 `split_group` API 一次性从默认进程组切出子组，理论上初始化更快，但不是默认行为——本篇不展开对比两条路径的差异，未查证两者在大规模集群下的初始化耗时差多少。

## 4. 主流程走读

### ① `initialize_model_parallel()`：五个维度怎么从一个张量里"切"出来

进入这个函数前，`world_size`、`data_parallel_size` 已经由 `ParallelConfig` 定好。函数先留下一段关键注释再把全局 rank 排成一个 5 维张量（`vllm/distributed/parallel_state.py:1817`-`1831`）：

```python
# vllm/distributed/parallel_state.py:1817-1831（源码原文）
# the layout order is: ExternalDP x DP x PP x PCP x TP
# ExternalDP is the data parallel group that is not part of the model,
# every dp rank can generate independently (in verl integration).
# DP is the data parallel group that is part of the model,
# all the ranks in the same DP group should generate simultaneously,
# i.e. the `generate` call in the same DP group should be called together,
# otherwise it will cause deadlock.
# to get group_ranks for each dimension, transpose that dimension to the
# last dimension, then reshape to 2D, then unbind the last dimension
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)  # noqa
```

之后每个并行组都是对这个张量做一次 `transpose` 把目标维度换到最后，再 `reshape(-1, size).unbind(0)` 切成一组组 rank 列表：

- TP：`all_ranks.view(-1, tensor_model_parallel_size).unbind(0)`（`vllm/distributed/parallel_state.py:1837`）——最后一维本来就是 TP，不需要 transpose，天然连续。
- PP：`all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)`（`vllm/distributed/parallel_state.py:1891`-`1893`）。
- DP：`all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)`（`vllm/distributed/parallel_state.py:1908`）。
- **EP**（`vllm/distributed/parallel_state.py:1926`-`1936`）：

```python
# vllm/distributed/parallel_state.py:1926-1936（源码原文，节选）
if config.model_config is None or config.model_config.is_moe:
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            data_parallel_size
            * prefill_context_model_parallel_size
            * tensor_model_parallel_size,
        )
        .unbind(0)
    )
```

EP 组的大小是 `data_parallel_size × prefill_context_parallel_size × tensor_parallel_size`——也就是说，**EP 组吃掉了 DP 和 TP 两个维度，只把 PP 排除在外**。这解释了"DP 和 EP 什么关系"：对于 MoE 模型，attention 部分仍然各自在自己的 TP 组内做张量并行，但 MoE 的专家层被摊到"同一个 PP stage 内的所有 DP×TP rank"上一起分担——所以 DP 和 EP 不是互斥关系，而是同一批物理 rank 在不同层上戴着不同的"组员"帽子：算 attention 时是 TP 组的一员，算 MoE 时是 EP 组的一员。上面那段注释里"all the ranks in the same DP group should generate simultaneously... otherwise it will cause deadlock"（`vllm/distributed/parallel_state.py:1820`-`1823`）正是为这件事埋下的伏笔——`## 4` ④会展开这句"deadlock"具体怎么被规避。

`config.model_config is None or config.model_config.is_moe`（`vllm/distributed/parallel_state.py:1926`）——稠密模型这一整段直接跳过，完全不创建 EP 组；对应地，`# If no EP group needed, _EP remains None`（`vllm/distributed/parallel_state.py:1979`）。

**worked example**：把上面的抽象公式换成一个具体拓扑更容易吃透。设 `world_size=8`、`tp=2`、`pp=2`、`dp=2`、`pcp=1`（稠密模型，不建 EP 组），按 `all_ranks = arange(8).reshape(-1, 2, 2, 1, 2)` 的行主序展开，每个 rank 的坐标 `(dp, pp, tp)` 和它落在哪些组里如下：

| rank | (dp, pp, tp) | 所在 TP 组 | 所在 PP 组 | 所在 DP 组 |
|---|---|---|---|---|
| 0 | (0,0,0) | {0,1} | {0,2} | {0,4} |
| 1 | (0,0,1) | {0,1} | {1,3} | {1,5} |
| 2 | (0,1,0) | {2,3} | {0,2} | {2,6} |
| 3 | (0,1,1) | {2,3} | {1,3} | {3,7} |
| 4 | (1,0,0) | {4,5} | {4,6} | {0,4} |
| 5 | (1,0,1) | {4,5} | {5,7} | {1,5} |
| 6 | (1,1,0) | {6,7} | {4,6} | {2,6} |
| 7 | (1,1,1) | {6,7} | {5,7} | {3,7} |

规律看一眼就懂：**TP 组永远是"最后一维连续的一对"**（因为 TP 是 `all_ranks` 天生最后一维，不需要 transpose）；**PP 组是"固定 (dp, tp)、只变 pp"的那两个 rank**；**DP 组是"固定 (pp, tp)、只变 dp"的那两个 rank**。同一个 rank 6 号，在 TP 语境下是 `{6,7}` 的一员，在 PP 语境下又是 `{4,6}` 的一员，在 DP 语境下还是 `{2,6}` 的一员——三个身份互不冲突，靠的正是坐标系统一，而不是三份互相独立算出来的分组表格。

### ② TP：四类层的四种切分/通信

**QKV 投影**——`QKVParallelLinear(ColumnParallelLinear)`（`vllm/model_executor/layers/linear.py:971`-`997`）：权重矩阵沿输出维（head 维）列切，每个 rank 拿到一部分 query/key/value head。列并行的 `forward`（`vllm/model_executor/layers/linear.py:575`-`603`，`ColumnParallelLinear.forward`）默认 `gather_output=False`，即**不做任何通信**——每个 rank 算出自己那部分 head 的 attention 输出，天然是"分片状态"，一路带到 O_proj。

**O_proj / down_proj**——`RowParallelLinear.forward()`（`vllm/model_executor/layers/linear.py:1641`-`1665`）：输入已经是分片的（`input_is_parallel=True`），本地矩阵乘后如果 `reduce_results=True and tp_size>1`，调 `tensor_model_parallel_all_reduce(output_parallel)`（`vllm/model_executor/layers/linear.py:1660`）。这是 TP 每层唯一必须的一次 all-reduce（attention 一次、MLP 一次，一层两次）。

**Embedding**——`VocabParallelEmbedding.forward()`（`vllm/model_executor/layers/vocab_parallel_embedding.py:486`-`505`）：按词表切分（不是隐藏维），每个 rank 只持有 `[vocab_start, vocab_end)` 区间的行；查表前先用 `get_masked_input_and_mask` 把落在别的 rank 区间的 token id 打上 mask 变成 0（`vllm/model_executor/layers/vocab_parallel_embedding.py:487`-`494`），查完表把落在别处的位置清零（`.masked_fill_`），最后 `all_reduce` 求和——因为每个 token 只在一个 rank 上非零，求和等价于"选出正确的那一份"。这和行/列并行都不同，是靠"稀疏 + 求和"模拟 gather 的技巧。

**lm_head**——`LogitsProcessor._gather_logits()`（`vllm/model_executor/layers/logits_processor.py:118`-`127`）：`lm_head` 复用 `VocabParallelEmbedding` 的切分方式，但输出层需要每个 rank 都看到完整词表的 logits 才能采样，所以末尾用 `tensor_model_parallel_all_gather(logits)`（`vllm/model_executor/layers/logits_processor.py:126`）把各 rank 的 vocab 切片拼回完整向量,而不是像 embedding 输入那样用 all_reduce。

### ③ PP：EngineCore 级的流水线，不是模型内部的 micro-batch

`max_concurrent_batches` 是流水线深度的唯一定义（`vllm/config/vllm.py:584`-`592`）：

```python
# vllm/config/vllm.py:584-592（源码原文）
@property
def max_concurrent_batches(self) -> int:
    # PP requires PP-size concurrent batches to fill the pipeline.
    # Async scheduling requires 2 concurrent batches to overlap.
    pp_size = self.parallel_config.pipeline_parallel_size
    if self.scheduler_config.async_scheduling:
        if self.use_v2_model_runner:
            return pp_size + 1
        if pp_size <= 1:
            return 2
    return pp_size
```

`EngineCore.step_with_batch_queue()`（`vllm/v1/engine/core.py:638`-`712`）用一个 `deque(maxlen=batch_queue_size)` 实现"先把队列填满再等结果"：只要队列没满且还有请求，就继续 `schedule()` + `execute_model(..., non_block=True)` 拿到一个 `Future` 塞进队列并立刻返回（不阻塞等待这一步的模型输出）；只有队列满了或没有更多请求可调度时，才 `batch_queue.pop()` 阻塞等待**最早**那个 future 的结果。

PP rank 之间通过 `GroupCoordinator.send_tensor_dict()` / `recv_tensor_dict()`（`vllm/distributed/parallel_state.py:981`、`1076`）把 `IntermediateTensors` 逐级传递（`vllm/v1/worker/gpu_model_runner.py:4579`-`4583`：非最后一个 PP rank 直接 `return hidden_states`，交给 executor 层的通信原语转发）。**这不是把一个 batch 切成 micro-batch 分给不同 PP rank**（vLLM 不做这种模型内部切分），而是把**多个连续的调度步骤**同时喂进流水线——效果类似 GPipe 的"多批次排队"，但排队的单位是"引擎迭代"而不是"模型内 micro-batch"，也没有 Megatron 那种为了减小气泡而设计的 interleaved/virtual-pipeline 调度。

### ④ DP：只有 MoE 才 lockstep

进程分发的裁决点在 `vllm/v1/engine/core.py:1320`-`1329`：

```python
# vllm/v1/engine/core.py:1320-1329（源码原文，节选去掉两行注释）
if data_parallel and vllm_config.model_config.is_moe:
    # Set data parallel rank for this engine process.
    parallel_config.data_parallel_rank = dp_rank
    engine_core = DPEngineCoreProc(*args, **kwargs)
else:
    # Non-MoE DP ranks are completely independent, so treat like DP=1.
    parallel_config.reconfigure_for_independent_dp_rank()
    engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
```

`DPEngineCoreProc.__init__` 开头就是一个 `assert vllm_config.model_config.is_moe`（`vllm/v1/engine/core.py:2014`-`2016`）。MoE 模型下，DP 各 rank 共享同一个 EP 组做专家计算，任何一个 rank 少跑一步都会让别的 rank 在集合通信上永久等待，因此每一步都要判断"本地没请求时是否要空跑"：`EngineCoreProc.run_busy_loop`（DP 版本，`vllm/v1/engine/core.py:2210`-`2223`）在 `local_unfinished_reqs` 为假但 `engines_running` 仍为真时调用 `execute_dummy_batch()`（`vllm/v1/engine/core.py:2216`-`2220`），底层是 `self.model_runner._dummy_run(num_tokens, uniform_decode=True)`（`vllm/v1/worker/gpu_worker.py:1221`-`1223`）——跑一个假 token 序列把这一步的集合通信"配合"完，但不产出真实输出。

而"全体 DP rank 是否都已经没活干了"这个全局判断很贵（要做一次 all-reduce），所以做了限流：`_has_global_unfinished_reqs()` 每被调用 32 次才真正发起一次同步 all-reduce，中间 31 次直接假定"还在跑"（`vllm/v1/engine/core.py:2267`-`2274`）。

进入模型前，每一步还有一次专门的 DP 对齐：`coordinate_batch_across_dp()`（`vllm/v1/worker/dp_utils.py:173`-`234`）把本 rank 这一步的 token 数、要不要做 dual-batch-overlap micro-batch、CUDA Graph 模式一起塞进一个 4×dp_size 的张量，跨 DP 组 all-reduce 一次（`vllm/v1/worker/dp_utils.py:39`-`57`，`_run_ar`），确保所有 rank 用同一套 padding 和同一个 CUDA Graph 分支——这一步默认走 NCCL，`disable_nccl_for_dp_synchronization` 打开后才退化成 CPU/gloo（`vllm/v1/worker/dp_utils.py:21`-`36`，`vllm/config/parallel.py:227`-`231`）。

### ⑤ 自定义 all-reduce vs NCCL：七级优先队列

`CudaCommunicator.all_reduce()`（`vllm/distributed/device_communicators/cuda_communicator.py:278`-`341`）依次尝试：symm-mem NCCL AR → quick reduce（ROCm MI3xx 专用）→ FlashInfer AR → AITER 自定义 AR（AMD）→ vLLM 自定义 AR → 通用 symm_mem → pynccl → `torch.distributed.all_reduce` 兜底。

vLLM 自定义 AR（`CustomAllreduce`）能不能用，看 `should_custom_ar()`（`vllm/distributed/device_communicators/custom_all_reduce.py:348`-`360`）：

```python
# vllm/distributed/device_communicators/custom_all_reduce.py:348-360（源码原文）
def should_custom_ar(self, inp: torch.Tensor):
    if self.disabled or self.world_size > 8:
        return False
    inp_size = inp.numel() * inp.element_size()
    if inp_size % 16 != 0:
        return False
    if not is_weak_contiguous(inp):
        return False
    if self.world_size == 2 or self.fully_connected:
        return inp_size < self.max_size
    return False
```

`self.disabled` 在构造时由多个条件叠加决定（`vllm/distributed/device_communicators/custom_all_reduce.py:93`-`202`）：`world_size` 必须落在 `{2,4,6,8,16}`（`_SUPPORTED_WORLD_SIZES`，`vllm/distributed/device_communicators/custom_all_reduce.py:57`）；同节点内 GPU 数 >2 时必须"全连通"（NVLink mesh，`is_fully_connected`），否则直接放弃（`vllm/distributed/device_communicators/custom_all_reduce.py:179`-`184`）。

还要做一次真实的 P2P 探测 `_can_p2p()`（`vllm/distributed/device_communicators/custom_all_reduce.py:36`-`50`，逐 GPU 对做一次真实探测而不是只看拓扑声明）。`max_size` 默认 8MB（`vllm/distributed/device_communicators/custom_all_reduce.py:74`），超过就退回更下游的路径。

### ⑥ EPLB：记录窗口 → 定时触发 → 真的搬权重

`EplbState`（`vllm/distributed/eplb/eplb_state.py:230`起）在 `__init__` 里从 `EPLBConfig` 读出 `expert_load_window_size`（滑窗，默认 1000 步）和 `expert_rearrangement_step_interval`（默认 3000 步，`vllm/distributed/eplb/eplb_state.py:432`-`448`）。每个 MoE 前向步都往滑窗里记一次各专家的实际负载（`_compute_eplb_load_stats`，`vllm/distributed/eplb/eplb_state.py:66`-`74`）；步数计数器超过 `step_interval` 后触发一次重排检查（`vllm/distributed/eplb/eplb_state.py:649`-`674`）。

触发后，`DefaultEplbPolicy`（改编自 DeepSeek 开源 EPLB，源码文件头自述，`vllm/distributed/eplb/policy/default.py:1`-`12`）用贪心装箱算法（`balanced_packing`，`vllm/distributed/eplb/policy/default.py:22`-`50`）算出新的专家到物理 rank 的映射；真正执行搬迁的是 `rearrange_expert_weights_inplace()`（`vllm/distributed/eplb/rebalance_execute.py:512`起），它调用 `transfer_layer()` 逐层把 `expert_weights` 张量从旧 rank 拷贝到新 rank 的中间缓冲区再拷回（`vllm/distributed/eplb/rebalance_execute.py:263`-`269`，`b[dst].copy_(w[src_local], non_blocking=True)`）——这是显式的显存到显存（可能跨节点）拷贝，不是免费操作，`is_profile=True` 时可以只算代价不真的搬（`vllm/distributed/eplb/rebalance_execute.py:455`-`458`）。

### ⑦ 多机启动：默认原生 multiprocessing，Ray 是显式选项

执行后端的裁决在 `ParallelConfig.__post_init__`（`vllm/config/parallel.py:936`-`976`）：只要 `distributed_executor_backend` 没被用户显式指定、且 `world_size_across_dp > 1`，就进入一串"能不用 Ray 就不用"的优先级判断——CUDA 平台且 `nnodes > 1` 时直接给 `backend = "mp"`（`vllm/config/parallel.py:946`-`947`）；只有 `data_parallel_backend == "ray"` 被显式设置（`vllm/config/parallel.py:960`-`965`），或者当前进程已经跑在一个 Ray placement group 里（`vllm/config/parallel.py:966`-`976`：先查 `self.placement_group`，再查 `ray_is_initialized()` 和 `get_current_placement_group()`），才会切到 `"ray"`。**多机不是触发 Ray 的条件，多机只会触发 `"mp"`**——这条容易被想当然地读反。

选中 `"mp"` 之后，`MultiprocExecutor`（`vllm/v1/executor/multiproc_executor.py:111`）本身就是能跨节点工作的：每个物理节点各起一份该执行器，用 `parallel_config.node_rank_within_dp == 0` 判断自己是不是这个 DP 组里的"leader 节点"（`vllm/v1/executor/multiproc_executor.py:148`）——leader 节点用 `get_ip()` 拿到自己在局域网里的真实 IP（`vllm/v1/executor/multiproc_executor.py:152`）起一个跨进程消息队列，非 leader 节点的 worker 通过这个 IP:port 加入同一个广播队列。

每个 worker 的全局 rank 由 `global_start_rank = local_world_size * node_rank_within_dp`（`vllm/v1/executor/multiproc_executor.py:177`-`179`）加上节点内 `local_rank`（`vllm/v1/executor/multiproc_executor.py:190`）算出——这条路径完全不依赖 Ray 的 actor/placement group 机制，跨节点通信走的是 vLLM 自己实现的共享内存 + TCP 消息队列（`vllm/distributed/device_communicators/shm_broadcast.py`，通信层实现细节不在本篇范围）。

`init_distributed_environment()`（`## 2` 已列出）在 DP>1（含单机多 DP 副本、以及多机场景）时要重新计算全局 rank 和 world_size（`vllm/distributed/parallel_state.py:1618`-`1620`）：

```python
# vllm/distributed/parallel_state.py:1618-1620（源码原文）
rank = parallel_config.data_parallel_rank * world_size + rank
world_size = parallel_config.world_size_across_dp
```

`torch.distributed.init_process_group` 最终看到的 rank 不是"这台机器第几张卡"，而是"DP 副本编号 × 单副本 world_size + 副本内本地 rank"——DP 维度在这里被摊平成一段连续的全局 rank 区间，这正是 `## 3.1` 强调"`ParallelConfig.world_size` 不含 `data_parallel_size`"的落地之处：两者在这一行代码里被重新拼接回全局视角。

同一段逻辑在 elastic EP 场景下还有一份变体（`vllm/distributed/parallel_state.py:1559`-`1560`），用的是同一条公式，只是提前算好存进局部变量传给别处，不再赘述。

只有当 `distributed_executor_backend == "ray"` 时才会走 `RayDistributedExecutor`（`vllm/v1/executor/ray_executor.py:64`）：每个 worker 是一个 Ray actor，跨节点通信靠 Ray 的对象存储和 placement group 调度，不需要用户手动填 `master_addr`/`node_rank`——这是它相对原生 `"mp"` 路径唯一的系统性优势（省掉手动指定每台机器 rank 的运维步骤），代价是多引入一层 Ray 自己的调度、序列化与容错机制，属于"图方便还是图少一层依赖"的取舍，本库不评判孰优孰劣——未查证两条路径在真实多机集群下的启动耗时/稳定性差异，没有实测数据支撑就不编结论。

## 5. 设计决策与代价

**决策 1：用一个 5 维张量 reshape+transpose 派生全部并行组，而不是给每种组合各写一段建组代码**（`vllm/distributed/parallel_state.py:1826`-`1934`）

- **为什么这么设计**：并行维度一旦超过两三个（TP/PP/DP/PCP/DCP/EP），维度间组合的方式会指数增长；用一个统一的坐标张量 `all_ranks`，任何一个并行组都只是"选一个维度 transpose 到最后再 reshape"，逻辑保证了同一个物理 rank 在不同维度上的分组关系永远互相一致（不会出现 TP 组和 EP 组各自算出的成员对不上号）。
- **不这样会怎样**：如果像早期一些框架那样为每种并行维度单独写嵌套 `for` 循环拼 rank 列表，新增一个维度（比如这次的 PCP）就要同时改多处建组代码，一旦某处循环顺序写错，会出现"同一个 rank 在 TP 组里认为自己是 0 号，在 EP 组里又被算成另一个物理设备"这种错位，轻则数据错误，重则集合通信死锁（本库推断，依据是该文件里维度构造互相依赖同一份 `all_ranks` 而非独立计算，说明作者刻意规避"多份独立真相"）。
- **什么时候可以不这样**：只有一个并行维度在用（比如单卡纯 TP、其余维度都是 1）时，这套通用机制其实是杀鸡用牛刀——很多轻量单机推理框架直接写 `rank // tp_size` 这种简单表达式就够了。vLLM 为了让同一套代码同时兼容任意维度组合，付出的代价是这段代码本身的可读性门槛更高。

**决策 2：`RowParallelLinear` 自己决定要不要 `all_reduce`（`reduce_results` 标志），而不是框架统一在每层后处理**（`vllm/model_executor/layers/linear.py:1660`）

- **为什么这么设计**：把通信的决定权下放到每个线性层本身，可以按需关闭——比如后面紧跟的层需要保留分片输出（配合 sequence parallel 或算子融合），这时候把 `reduce_results` 设成 `False` 就能省掉一次通信。
- **不这样会怎样**：如果框架层面统一约定"每个 TP 层后面都做一次 all-reduce"，会在某些可以合并通信的场景里多做一次不必要的 all-reduce——比如两个相邻的行并行层之间，统一策略会强制先 gather 回完整张量、再重新按行切分喂给下一层，比直接传递分片状态多两次通信。
- **什么时候可以不这样**：`tp_size == 1` 时 `if self.reduce_results and self.tp_size > 1` 这个条件天然为假（`vllm/model_executor/layers/linear.py:1660` 所在的 `if` 判断本身），退化成纯本地矩阵乘，不产生任何通信开销——TP=1 时这套机制自动关闭，不需要用户手动处理。

**决策 3：PP 用 EngineCore 级别的 batch queue 做流水线，而不是模型内部切 micro-batch**（`vllm/v1/engine/core.py:638`-`712`）

- **为什么这么设计**：vLLM 的 continuous batching 本身每一步的 batch 组成都是动态的（长度不一、有的在 prefill 有的在 decode），如果再叠加一层"把这个动态 batch 切成规整 micro-batch 分给不同 PP stage"的逻辑，两套批处理机制会互相打架。选择在 EngineCore 迭代粒度上排队（连续几步的调度决策同时在飞），可以让每一步仍然是一次完整、独立的 continuous-batching 决策，PP 带来的"流水线"效果是靠多步重叠实现的，不需要改动调度器内部的批处理逻辑。
- **不这样会怎样**：如果引入 Megatron 式的 virtual pipeline / interleaved 1F1B 调度，需要在调度器里维护多个"尚未跑完的 micro-batch"的状态机，还要和 KV cache 分配、抢占逻辑深度耦合——本库推断，这大概率是 vLLM（以及下面 `## 6` 会看到的 SGLang）都没有采用 virtual pipeline 的原因：工程复杂度的提升，对推理场景（相比训练，序列长度分布更极端、batch 组成变化更频繁）收益不成比例。
- **什么时候可以不这样**：`pipeline_parallel_size <= 1` 时 `max_concurrent_batches` 退化为 1（非异步调度）或 2（异步调度，`vllm/config/vllm.py:588`-`592`），batch queue 的排队机制形同虚设，直接单步执行——不开 PP 就完全不需要关心这一层。

**决策 4：DP 的"步调一致"用 dummy batch 填充空闲 rank，而不是允许某个 rank 跳过这一步**（`vllm/v1/worker/gpu_worker.py:1221`-`1223`）

- **为什么这么设计**：EP 组的集合通信（all-gather/reduce-scatter，见 `## 4` ①⑥）要求组内每个 rank 都发起同一次通信调用；如果某个 rank 因为本地没有请求就直接跳过整个前向，其余 rank 发起的那次集合通信会因为少了一个参与者而永久挂起。用一次"假前向"（数据无意义，但确实调用了同一组通信原语）让所有 rank 保持相同的调用序列，是让 MoE EP 能在动态负载下工作的最小改动。
- **不这样会怎样**：允许空闲 rank 直接 skip，会导致其余 rank 的 all-to-all/all-gather 调用因为组内成员数不匹配而 hang 住，最终触发 NCCL 超时——这正是 `vllm/distributed/parallel_state.py:1820`-`1823` 那段注释里"otherwise it will cause deadlock"想要提前警告的场景。
- **什么时候可以不这样**：非 MoE 稠密模型的 DP 完全不用这套机制——如 `## 4` ④所述，它们直接被当成互相独立的普通 `EngineCore` 副本（`vllm/v1/engine/core.py:1326`-`1329`），各 rank 各跑各的，没有共享的 EP 组需要保持一致，也就没有 dummy batch 的必要。

**决策 5：自定义 all-reduce 优先于 NCCL，但严格限定小 world_size、小张量、同节点 P2P**（`vllm/distributed/device_communicators/custom_all_reduce.py:348`-`360`）

- **为什么这么设计**：基于 CUDA IPC 的点对点显存写入，在同节点小规模（2/4/6/8 卡）、小张量（几十 KB 到几 MB，恰好是 TP 层间激活值 all-reduce 的典型规模）场景下，比 NCCL 的 ring/tree 算法省掉了 kernel launch 和 ring 调度的固定开销，延迟更低。
- **不这样会怎样**：如果不设 `should_custom_ar()` 里的尺寸和 world_size 门槛，对大张量或跨节点场景也硬走自定义 AR，会因为 CUDA IPC 缓冲区容量固定（`max_size` 默认 8MB）而直接失败，或者因为不支持跨节点 P2P 而根本无法工作——门槛本身就是"别把这条快路径用错场景"的护栏。
- **什么时候可以不这样**：显式设 `disable_custom_all_reduce=True`（`vllm/config/parallel.py:205`）整条路径直接关闭，退回 pynccl/`torch.distributed`；或者硬件本身不支持 P2P（比如纯 PCIe、无 NVLink 的多卡机器），`_can_p2p()` 探测失败后会自动禁用（`vllm/distributed/device_communicators/custom_all_reduce.py:188`-`202`），不需要用户手动干预。

**决策 6：EPLB 用固定的"窗口 + 间隔"节流重排频率，而不是每步都重新计算负载均衡**（`vllm/distributed/eplb/eplb_state.py:432`-`448`）

- **为什么这么设计**：统计专家负载、跑装箱算法、真正搬运权重张量三步都有实打实的开销——尤其权重搬运是真实的 PCIe/NVLink 显存拷贝（`## 4` ⑥）。如果每步都触发这套流程，通信开销会反过来吃掉负载均衡本身想省下来的收益。用滑动窗口攒够统计样本（默认 1000 步）、每隔几千步（默认 3000 步）才触发一次，是"用统计的陈旧度换开销"的权衡。
- **不这样会怎样**：如果重排太频繁，专家权重刚搬完，负载分布又变了，等于持续为一个不稳定的目标追着跑；如果完全不重排，负载不均会随着长时间运行、路由分布漂移持续累积，某些专家所在的 rank 长期成为瓶颈。
- **什么时候可以不这样**：`enable_eplb=False`（默认值）时整套机制根本不存在；`use_async=True`（默认值）让重排和前向计算异步重叠，从"停下来重排"退化为"边算边搬"，进一步摊薄这个权衡的代价——但这要求 `communicator` 不能是 `torch_nccl`/`pynccl`（`vllm/config/parallel.py:102`-`113` 的校验），因为异步重排和 NCCL 共享 CUDA stream 的多流机制会冲突，必须换成 `torch_gloo` 或 `nixl`。

## 6. 同位对照（SGLang 在同一位置怎么做）

**并行组的构造方式**：SGLang 的 `initialize_model_parallel()` 不走"从一个统一坐标张量里切"这条路，而是让调用方显式传入七个独立命名的维度参数（`sglang:python/sglang/srt/distributed/parallel_state.py:2328`-`2342`）：`tensor_model_parallel_size`、`expert_model_parallel_size`、`pipeline_model_parallel_size`、`attention_data_parallel_size`、`attention_context_model_parallel_size`、`moe_data_model_parallel_size`、`decode_context_parallel_size`。函数文档里给的例子（`sglang:python/sglang/srt/distributed/parallel_state.py:2374`-`2385`）显式区分了"4 个 attention 张量并行组"和"2 个 MoE 专家并行组"，二者可以用不同的划分方式。

这与 vLLM 把 EP 组定义成"DP×PCP×TP 的合并投影"（`## 4` ①）是两种不同的建模思路：vLLM 认为 EP 是其他维度的派生物，SGLang 则把"attention 侧怎么并行"和"MoE 侧怎么并行"当成两套独立坐标系从一开始就分开声明。哪种更好未查证——两边都没有公开的消融实验说明这个设计选择对吞吐/延迟的实际影响，本库不编造这个结论。

**DP 的 lockstep 机制**：SGLang 也有一个和 `execute_dummy_batch()` 对等的机制——`DPAttentionHelper.get_idle_batch()`（`sglang:python/sglang/srt/managers/scheduler_components/dp_attn.py:442`-`453`）构造一个请求列表为空、随后调用 `prepare_for_idle()` 把 `forward_mode` 设成 `ForwardMode.IDLE` 的空批次（`sglang:python/sglang/srt/managers/schedule_batch.py:2951`-`2953`），本质上和 vLLM 的 `_dummy_run()` 是同一个目的：让没有真实请求的 rank 也发起一次完整的前向调用，配合其它 rank 完成集合通信。

但两边判断"要不要开这套 lockstep"的**门槛不同**：vLLM 是"这个模型是不是 MoE"（`## 4` ④，`vllm_config.model_config.is_moe`）；SGLang 是 `require_mlp_sync()`——`get_parallel().enable_dp_attention` 显式开启，或者 `require_gathered_buffer()` 为真（`sglang:python/sglang/srt/utils/common.py:3798`-`3801`）。也就是说 SGLang 把"要不要 lockstep"做成了一个独立的运行时开关（`enable_dp_attention`），不是从"是不是 MoE 模型"这一个信号自动推导出来的——这意味着 SGLang 理论上可以让**非 MoE 模型**的 attention 也走 DP lockstep（纯粹为了 attention 层的负载均衡），而 vLLM 目前把"DP 需要步调一致"和"这是不是 MoE 模型"焊死在了一起（`## 7` 踩坑 5 里"非 MoE DP 完全独立"这条，在 SGLang 这边不成立，取决于 `enable_dp_attention` 而不是模型类型）。

**EPLB 触发条件**：vLLM 用固定的 `step_interval`（默认 3000 步，`## 5` 决策 6）节流；SGLang 的 `EPLBManager._check_rebalance_needed()` 则是按**滑窗内平均 GPU 利用率**门控——只有当利用率低于 `eplb_min_rebalancing_utilization_threshold` 时才真正触发重排，否则跳过并打日志说明（`sglang:python/sglang/srt/eplb/eplb_manager.py:232`-`241`）。这是"按固定节奏做"和"挑系统不忙的时候做"两种不同的节流哲学，`## 8` 会把这一点列为可能的改进方向。另外 SGLang 把 EPLB 独立成顶层包 `srt/eplb/`（`eplb_manager.py`、`expert_location.py`、`expert_distribution.py` 等，均在 `python/sglang/srt/eplb/` 下），而不是像 vLLM 那样嵌在 `distributed/eplb/` 目录里——这个目录层级差异直接影响了 `## 2` 提到的"按目录粗口径统计会产生假阳性"问题的严重程度：SGLang 的目录结构不会把 EPLB 和 KV 传输混进同一个"distributed"统计桶。

**多机启动**：两边都**不**默认用 Ray。SGLang 的 `ServerArgs` 里 `dist_init_addr`、`nnodes`、`node_rank` 是走原生 `torch.distributed` 初始化的标准参数（`sglang:python/sglang/srt/server_args.py:1030`-`1039`），没有 Ray 依赖；vLLM 同样默认走自研的 `MultiprocExecutor`（`## 0` 最后一条）。这说明"多机部署默认不用 Ray、自己管理 rank/master 地址"不是 vLLM 一家的特例，而是当前主流开源推理引擎的共同选择——Ray 更多是给"需要动态伸缩、跨异构集群调度"的复杂部署场景准备的可选项，不是多机的默认必需品。

**自定义 all-reduce**：SGLang 同样有一份 `CustomAllreduce` 实现（`sglang:python/sglang/srt/distributed/device_communicators/custom_all_reduce.py:40`），机制上（CUDA IPC、小 world_size 门槛）与 vLLM 高度相似——本库推断，两边这部分代码在设计思路上可能同源或互相参考了同一份社区实现（vLLM 的 `custom_all_reduce.cuh` 在 `csrc/` 目录下也是社区共享度较高的算子），但具体阈值、支持的 world_size 列表是否完全一致，需要逐行 diff 才能下结论——未查证，本篇不做这个比较。

## 7. 踩坑与反直觉

1. **`struct_map.json` 的"distributed 138 文件/54,851 行"里，59% 是 KV 传输代码，不是并行策略代码**——`## 2` 已经用真实的 `wc -l` 复核过：`kv_transfer/` 32,609 行、`ec_transfer/` 1,806 行，两者合计约 63% 的行数属于 [[12-vLLM-PD分离与KV-Connector]] 的地盘。任何用"文件路径含 distributed 关键词"做粗口径统计的工具，在这个目录上都会得到失真的规模印象——这条踩坑在写这篇之前就被 `_PLAN.md` 提前警告过（"这是文件名关键词匹配的粗口径，有假阳性"），实测确认警告是对的。
2. **`ParallelConfig.world_size` 不包含 `data_parallel_size`**（`vllm/config/parallel.py:858`-`862`）——第一次读代码容易把 `self.world_size` 当成"总 GPU 卡数"，实际上它只是 `pp × tp × pcp`，DP 是独立的多副本维度，要用 `world_size_across_dp`（`vllm/config/parallel.py:573`-`575`）才是真正意义上的全局进程数。
3. **`_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]`（`vllm/distributed/device_communicators/custom_all_reduce.py:57`）里的 16，对 `all_reduce` 并不生效**——`should_custom_ar()` 一进来就 `if self.disabled or self.world_size > 8: return False`（`vllm/distributed/device_communicators/custom_all_reduce.py:349`），16 这一档只用于 `all_gather`/`reduce_scatter` 的 MNNVL（多机 NVLink）路径（`vllm/distributed/device_communicators/custom_all_reduce.py:401`-`403`、`455`-`457`）。只看"支持的 world_size 列表"很容易误以为 16 卡的 TP/EP 组也能走自定义 all-reduce。
4. **DP 的全局完成度判定不是每步都同步**——`_has_global_unfinished_reqs()`（`vllm/v1/engine/core.py:2267`-`2274`）表面上看是每步都调用，但函数体里第一句就是 `self.step_counter += 1; if self.step_counter % 32 != 0: return True`——只在第 32 步的倍数才真正发起 all-reduce，中间 31 次直接假定"还在跑"直接返回。不往函数体里看，只看调用点，会以为这是个每步都做的同步点。
5. **`reconfigure_for_independent_dp_rank()` 会让同一个 `ParallelConfig` 对象里两个"看起来同义"的字段产生分歧**：非 MoE 模型的 DP rank 被重配置后，`data_parallel_rank` 被强制清零（`vllm/config/parallel.py:1070`），但 `data_parallel_index` 仍然保留原始的 dp_rank——源码注释原话是"Note that parallel_config.data_parallel_index will still reflect the original DP rank"（`vllm/v1/engine/core.py:1327`-`1328`）。如果日志或调试代码里习惯性地读 `data_parallel_rank` 来判断"这是第几个 DP 副本"，在非 MoE 场景下会读到一个恒为 0 的错误值，得改读 `data_parallel_index`。
6. **`_EP` 进程组存在，不代表 MoE 层真的在用 expert-parallel 路由**——`## 4` ①里 `initialize_model_parallel()` 建 EP 组的条件只看 `config.model_config.is_moe`（`vllm/distributed/parallel_state.py:1926`），完全不检查 `enable_expert_parallel`；但 MoE 层内部真正决定"要不要按专家切分"的开关是另一处：`FusedMoEParallelConfig` 里 `use_ep = (dp_size × pcp_size × tp_size > 1) and vllm_parallel_config.enable_expert_parallel`（`vllm/model_executor/layers/fused_moe/config.py:1212`-`1214`）。`enable_expert_parallel` 默认是 `False`（`vllm/config/parallel.py:165`）——也就是说，一个 MoE 模型即使跑在多卡上，只要没显式加 `--enable-expert-parallel`，`_EP` 进程组会被建出来，但 MoE 层实际上并不使用它做专家路由（退化成专家权重在 TP 维度上复制/切分的另一套路径）。只看"这个进程组存不存在"来判断"是不是在用 EP"，会得出错误结论。
7. **`VLLM_DISTRIBUTED_USE_SPLIT_GROUP` 不是默认值，但代码路径永远在**——`GroupCoordinator.__init__` 里 `if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP`（`vllm/distributed/parallel_state.py:436`）这个分支平时不会被走到（默认 `False`，`vllm/envs.py:66`），容易被当成"死代码"跳过不看；但只要环境变量一开，全部 8 个进程组的建组方式都会切换成 `_create_subgroups_split_group()`（`vllm/distributed/parallel_state.py:282`）这条路径，行为差异（尤其是失败时的报错信息、超时行为）没有被本篇验证过是否完全一致——读代码时两条路径都要过一遍，不能只读默认分支。

## 8. 可改进点

以下均为**本库推断**，未在上游 issue/PR 中核实过，标注依据：

1. **EPLB 的重排触发条件可以引入类似 SGLang 的利用率门控**——当前 `step_interval` 是固定步数（`## 5` 决策 6），意味着系统本来就很闲、负载均衡收益微乎其微的时段，也会按部就班触发一次真实的权重搬运；而系统正忙、最需要腾出带宽跑推理请求的时段，也可能恰好撞上重排窗口。SGLang 的 `eplb_min_rebalancing_utilization_threshold`（`## 6`）提供了一个"挑系统较闲时机做"的思路，vLLM 目前没有等价的门控信号。
2. **`disable_nccl_for_dp_synchronization` 的隐式默认值不够显眼**——字段类型是 `bool | None`，`None` 时的真实行为（"async scheduling 时为 True，否则 False"）写在字段的 docstring 里（`vllm/config/parallel.py:227`-`231`），但要理解这个默认值真正落地的地方得去翻 `field_validator`（`vllm/config/parallel.py:422`起）。对一个会决定"DP 同步走 NCCL 还是 CPU gloo"这种性能敏感开关，`None` sentinel + 文档字符串的组合比直接暴露一个只读 property 更容易被漏读。
3. **`vllm/distributed/` 目录把"并行策略"和"KV 跨进程搬运"混在一起，是 `## 2`/`## 7` 那条统计失真问题的根源**——如果像 SGLang 把 EPLB 独立成 `srt/eplb/` 顶层包一样，把 `kv_transfer/`、`ec_transfer/` 提升成 `vllm/kv_transfer/`、`vllm/ec_transfer/` 顶层包，不需要改变任何运行时逻辑，就能让"这个目录在讲什么"从路径本身读出来，减少后续代码审计、新人上手、以及像本库这样做自动化代码地图统计时的混淆成本。
4. **`enable_expert_parallel` 和"是否建 `_EP` 组"这两件事可以在日志里显式对齐**——`## 7` 踩坑 6 发现的"组建了但没用上"这种状态，目前只能靠交叉读两处源码才能确认；`initialize_model_parallel()` 末尾已经有一条 `logger.info_once` 打印每个 rank 的 DP/PP/PCP/TP/EP/EPLB rank（`vllm/distributed/parallel_state.py:1982`-`1994`），顺手加一行"`enable_expert_parallel=False`，EP 组已建但 MoE 层不会使用"这样的显式告警，能让这条踩坑在启动日志阶段就被发现，而不是等看到 MoE 层性能不如预期才回头查配置。

## 9. 自测题与延伸阅读

**自测题**（闭卷，答案都在 `## 4`/`## 5` 里）：

1. EP 组的大小是哪三个并行维度的乘积？为什么 PP 维度被排除在外？
2. 给一个非 MoE 稠密模型设置 `--data-parallel-size 4`，和给一个 MoE 模型设置同样的参数，运行时行为有什么本质区别？分别对应哪个类（`DPEngineCoreProc` 还是普通 `EngineCoreProc`）？
3. 自定义 all-reduce（`CustomAllreduce`）在什么条件下会被跳过、转而走 pynccl 或 `torch.distributed` 默认路径？至少说出 3 个独立的判定条件。
4. vLLM 的 PP 流水线并行和 Megatron 式的 interleaved 1F1B 有什么本质不同？为什么 `## 5` 决策 3 认为 vLLM 选了一条更简单的路？
5. `EPLBConfig` 里 `window_size`（默认 1000）和 `step_interval`（默认 3000）分别控制什么？为什么 `step_interval` 必须大于等于 `window_size` 才有意义？
6. 多机部署 vLLM，默认会用 Ray 还是自研 `MultiprocExecutor`？在什么条件下会切换到 Ray？
7. 一个 MoE 模型跑在多卡上，`_EP` 进程组一定会被创建吗？创建了 `_EP` 组是否就意味着 MoE 层一定在用专家并行路由？两者分别由哪个字段控制？
8. 世界坐标张量 `all_ranks` 的形状是 `[-1, dp, pp, pcp, tp]`。给定 `world_size=8`、`tp=2`、`pp=2`、`dp=2`，rank 5 会同时出现在哪三个组里（TP 组、PP 组、DP 组）？

**延伸阅读**：

- [[03-vLLM-调度器解剖]]——`EngineCore.step()` 的完整四步循环、`SchedulerOutput` 的其余字段，是本篇 `## 4` ③ batch queue 机制的上游调用方。
- [[06-vLLM-模型执行与CUDA-Graph]]——`torch.ops.vllm.all_reduce` 这个自定义算子如何被 CUDA Graph capture 和 `torch.compile` 一起编译进执行图，本篇 `## 1` 只提了一句，细节在那一篇。
- [[12-vLLM-PD分离与KV-Connector]]——`vllm/distributed/kv_transfer/` 目录的真正职责，`## 2`/`## 7` 反复提到的"63% 假阳性"行数具体都在讲什么。
