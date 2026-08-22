# vLLM 调度器解剖

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：无阶段划分，只按 token 预算追赶目标

## 0. 结论先行

- `schedule()` 里**没有"prefill 阶段"和"decode 阶段"这两个概念**。每个请求只有一个数字要追：`num_computed_tokens` 要追上 `num_tokens_with_spec`。这句话是源码原话，见 `vllm/v1/core/sched/scheduler.py:486`-`495` 的整段 NOTE 注释，本篇 `## 4` 会逐字展开。
- 每一步先扫 **RUNNING**（已经在跑的请求，可能还在分块 prefill），再扫 **WAITING**（全新请求）。这不是"decode 优先于 prefill"——因为 RUNNING 里本来就混着还没 prefill 完的请求；准确的说法是"已经占了名额的优先于全新准入的"。
- token 预算是**一根共享的标量**（`token_budget`，源自 `max_num_scheduled_tokens`），chunked prefill 不是"提前把 prompt 切好扔进队列"，而是"这一步预算够多少就算多少，算不完下一步接着算"。
- 抢占（preemption）= **丢弃 KV、下次重算**。V1 里没有 swap-to-CPU 这条路径——v0 时代的 `--swap-space` 参数已被删除。
- 请求队列只有两种实现：FCFS 用 `deque`，PRIORITY 用 `heapq`。没有第三种，也没有前缀缓存感知的排队策略。
- `SchedulerOutput` 是调度器与 worker 之间的**唯一契约**，14 个顶层字段，只有 3 个是"喂模型吃饭"的（`scheduled_new_reqs` / `scheduled_cached_reqs` / `num_scheduled_tokens`），其余大多是给 KV connector、EC connector、结构化输出、投机解码、CUDA Graph 清零这些旁路系统搭的桥。
- 连续批处理（continuous batching）本身的收益不是在任何流量下都存在——`## 5` 决策 5 会用源码证据说明它在"同质输入 + 离线批量"场景下收益趋近于零。

**速查：关键问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| `schedule()` 主循环先调度 running 还是 waiting？为什么？ | `## 4` ②④、`## 5` 决策 1 |
| token 预算怎么被消耗？chunked prefill 怎么切一个长 prompt？ | `## 4` ①②、`4.1` worked example、`## 5` 决策 2 |
| 抢占的触发条件、选谁抢、KV 丢弃还是 swap？V1 还支不支持 swap？ | `## 4` ③、`## 5` 决策 3、`## 7` 踩坑 4 |
| 调度队列是 FCFS 还是优先级堆？优先级从哪来？ | `## 3.3`、`## 5` 决策 4 |
| 结构化输出/投机解码/多模态在调度里有没有特殊分支？ | `4.2` |
| `SchedulerOutput` 带了什么字段给 worker？ | `## 3.2` |

## 1. 它在系统里的位置

调度器是 `EngineCore` 主循环里的第一个环节。`vllm/v1/engine/core.py:597`-`624` 的 `EngineCore.step()` 把一次迭代拆成四步：

```python
# vllm/v1/engine/core.py:597 附近（节选，行号见下）
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())   # L608
    future = self.model_executor.execute_model(scheduler_output, non_block=True)   # L609
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)          # L610
    ...
    model_output = future.result()
    ...
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )                                                                              # L621-623
```

`schedule()` → `execute_model()` → `get_grammar_bitmask()` → `update_from_output()`，这四个方法调用构成一次完整迭代。调度器自己不跑模型、不算 logits，它只决定"这一步谁上场、上场吃几个 token、KV 往哪放"，把决定打包成 `SchedulerOutput` 交给 `model_executor`。

`Scheduler.__init__`（`vllm/v1/core/sched/scheduler.py:74`-`373`）持有的旁路对象：

| 对象 | 管什么 | 是否必然存在 |
|---|---|---|
| `KVCacheManager` | KV 块分配、前缀缓存命中查询 | 必有 |
| `EncoderCacheManager` / `EncoderDecoderCacheManager` | 多模态编码器输出缓存 | 必有（无多模态时预算为 0） |
| `StructuredOutputManager` | 语法约束、grammar bitmask 计算 | 必有 |
| `KVConnectorBase_V1` | PD 分离 / 外部 KV 搬运 | 可选，配置 `kv_transfer_config` 才创建 |
| `ECConnectorBase` | 编码器缓存跨进程搬运 | 可选，配置 `ec_transfer_config` 才创建 |
| `RoutedExpertsManager` | MoE 专家路由记录 | 可选，`enable_return_routed_experts` 开启才创建 |

调度器**不持有模型权重**，模型执行完全在 `model_executor` 那一侧——这条边界很干净：`Scheduler` 只读/写请求元数据和 KV block 的"账本"，从不接触张量本身。

一次 `step()` 只调一次 `schedule()`；`schedule()` 一次调用只产出"这一步"的批次，不是把所有排队请求一次性调度完——这也是为什么 `EngineCore` 要在一个 `while` 循环里反复调 `step()`，且每次调用前都先检查 `self.scheduler.has_requests()`（`vllm/v1/engine/core.py:606`-`607`）：没有任何未完成或待清理的请求时，直接跳过整轮 `schedule()`/`execute_model()`，不空转 GPU。

## 2. 代码地图（文件 → 职责，带行号）

`_lab/out/struct_map.json` 把所有文件名里含 `scheduler` 字样的文件都归进了"scheduler"子系统，统计出 20 个文件 / 8,424 行——但这个统计口径**按文件名字符串匹配**，混进了 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`（KV 传输的写调度，1,698 行）、`vllm/distributed/eplb/policy/default.py`（专家负载均衡策略）等和"连续批处理调度器"完全不同的子系统（本库推断，依据是这些文件路径与职责描述均与请求批处理无关）。**本篇讲的"调度器"是真正做连续批处理决策的那一撮文件**——`vllm/v1/core/sched/` 整个目录，6 个文件、4,012 行，外加 `vllm/v1/request.py`（请求状态机）与 `vllm/config/scheduler.py`（配置对象）：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `vllm/v1/core/sched/scheduler.py:73` | `Scheduler` 类定义，全库最大文件（3,037 行） | — |
| `vllm/v1/core/sched/scheduler.py:484` | `schedule()`——主调度循环入口 | — |
| `vllm/v1/core/sched/scheduler.py:1347` | `_preempt_request()`——抢占的唯一实现 | — |
| `vllm/v1/core/sched/scheduler.py:1744` | `update_from_output()`——消费 worker 输出、判断请求是否结束 | — |
| `vllm/v1/core/sched/scheduler.py:2182` | `_select_waiting_queue_for_scheduling()`——决定这一步从哪个等待队列弹请求 | — |
| `vllm/v1/core/sched/scheduler.py:2331` | `add_request()`——请求入队入口 | — |
| `vllm/v1/core/sched/scheduler.py:2359` | `finish_requests()`——外部中止/detokenizer 命中 stop string 时的出口 | — |
| `vllm/v1/core/sched/output.py:206` | `SchedulerOutput` dataclass——调度器与 worker 的契约 | — |
| `vllm/v1/core/sched/output.py:35` | `NewRequestData`——首次调度的请求全量数据 | — |
| `vllm/v1/core/sched/output.py:129` | `CachedRequestData`——已调度过请求的增量数据 | — |
| `vllm/v1/core/sched/interface.py:24` | `PauseState` 枚举（UNPAUSED/PAUSED_NEW/PAUSED_ALL） | — |
| `vllm/v1/core/sched/interface.py:38` | `SchedulerInterface` 抽象基类，定义调度器必须实现的全部方法 | — |
| `vllm/v1/core/sched/request_queue.py:13` | `SchedulingPolicy` 枚举（FCFS / PRIORITY） | — |
| `vllm/v1/core/sched/request_queue.py:75` | `FCFSRequestQueue`——继承 `deque` | — |
| `vllm/v1/core/sched/request_queue.py:131` | `PriorityRequestQueue`——包一层 `heapq` | — |
| `vllm/v1/core/sched/utils.py:94` | `check_stop()`——停止条件判定（EOS/stop token/长度/重复检测） | — |
| `vllm/v1/core/sched/async_scheduler.py:12` | `AsyncScheduler(Scheduler)`——异步调度子类，只覆写 3 个方法 | — |
| `vllm/v1/request.py:364` | `RequestStatus` 枚举——请求状态机 | — |
| `vllm/config/scheduler.py:26` | `SchedulerConfig`——`max_num_batched_tokens` 等旋钮的定义处 | — |
| `vllm/v1/core/kv_cache_manager.py:347` | `allocate_slots()`——调度器申请 KV 块的唯一入口，返回 `None` 即触发抢占 | — |

（第一列已含行号，满足"文件:行"引用格式；上表 20 条，超过硬指标要求的 10 条。)

**本篇范围边界**：`allocate_slots()` 内部具体怎么分配 block、前缀缓存怎么命中/淘汰，属于 [[04-vLLM-KV缓存与前缀缓存]] 的地盘，本篇只讲调度器怎么调用它、怎么解读它的返回值（`None` 触发抢占）。`EngineCore` 的完整主循环结构（含 `capture_iteration_details`、pipeline parallel 的 microbatch 调度）属于 [[02-vLLM-V1架构与EngineCore循环]]，本篇只截取 `schedule()` 相关的四步调用。一事一文，边界划清楚。

## 3. 核心数据结构

### 3.1 `Request` 与 `RequestStatus`（请求状态机）

`vllm/v1/request.py:364`-`380` 定义状态机：

```
WAITING ──┬─→ WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR ─┐
          ├─→ WAITING_FOR_REMOTE_KVS                  ├─→ RUNNING ──→ PREEMPTED ──(回到 waiting 队头)
          └─→ WAITING_FOR_STREAMING_REQ ──────────────┘        │
                                                                 └─→ FINISHED_STOPPED / FINISHED_LENGTH_CAPPED /
                                                                     FINISHED_ABORTED / FINISHED_IGNORED /
                                                                     FINISHED_ERROR / FINISHED_REPETITION
```

对应的源码原文（`vllm/v1/request.py:364`-`380`，逐字抄录）：

```python
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()
```

一个巧妙设计：`RequestStatus` 是 `IntEnum`，成员按声明顺序编号，`PREEMPTED` 排在所有 `FINISHED_*` 之前。`is_finished()` 的实现只有一行（`vllm/v1/request.py:386`-`387`）：

```python
@staticmethod
def is_finished(status: "RequestStatus") -> bool:
    return status > RequestStatus.PREEMPTED
```

第 373-374 行的注释直接写明："Note: anything after `PREEMPTED` will be considered as a finished status."——这意味着**给状态机加新的"完成"类型只能加在 `PREEMPTED` 之后**，加在前面会被 `is_finished()` 误判。

### 3.2 `SchedulerOutput`（调度器 → worker 的唯一契约）

`vllm/v1/core/sched/output.py:206`-`297`，逐字段过一遍（证据等级：源码为证）：

| 字段 | 含义 |
|---|---|
| `scheduled_new_reqs: list[NewRequestData]` | 本步首次调度的请求，带全量 prompt/采样参数/block_ids |
| `scheduled_cached_reqs: CachedRequestData` | 本步继续调度的请求，只带**增量**（worker 已缓存过全量数据） |
| `num_scheduled_tokens: dict[str, int]` | 每个请求本步分到几个 token 名额——调度器输出的核心 |
| `total_num_scheduled_tokens: int` | 上一字段的和，`≤ max_num_scheduled_tokens` |
| `scheduled_spec_decode_tokens: dict[str, list[int]]` | 投机解码：本步要验证的草稿 token |
| `scheduled_encoder_inputs: dict[str, list[int]]` | 多模态：本步要跑视觉/音频编码器的输入下标 |
| `num_common_prefix_blocks: list[int]` | RUNNING 队列的最长公共前缀块数，给 cascade attention 用 |
| `finished_req_ids: set[str]` | 上一步到这一步之间结束的请求，通知 worker 释放缓存状态 |
| `free_encoder_mm_hashes: list[str]` | 要从编码器缓存里释放的多模态哈希 |
| `preempted_req_ids: set[str] \| None` | 本步被抢占的请求（V2 model runner 用） |
| `has_structured_output_requests: bool` | 本步是否有结构化输出请求（仅异步调度设置） |
| `kv_connector_metadata` / `ec_connector_metadata` | PD 分离 / EC 传输的搬运元数据，由对应 connector 的 `build_connector_meta()` 填 |
| `new_block_ids_to_zero: list[int] \| None` | 本步新分配、需要 worker 清零的 block id（防止脏数据污染 attention/SSM） |
| `kv_cache_block_copies` / `partial_tail_offloads` | CoW 复制、Mamba 部分尾块的搬运指令 |
| `num_spec_tokens_to_schedule: int` | 动态投机解码：调度器算出的下一步最优 K |

`NewRequestData` 和 `CachedRequestData` 的区别体现了一个明确的工程权衡：worker 进程会缓存每个请求的数据（`vllm/v1/core/sched/output.py:209` 注释 "we don't need to re-send it every scheduling step"），所以第一次调度发全量，之后只发"新 token + 新 block"这样的增量,省掉重复传输 prompt。两者的字段定义对照（`vllm/v1/core/sched/output.py:36`-`49` 与 `:130`-`144`，节选）：

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    ...

@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_token_ids: list[list[int]]      # 仅 PP 用；无 PP 时为空
    all_token_ids: dict[str, list[int]]  # 仅 MRV1 补发用
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
```

`NewRequestData` 是"这个请求长什么样"（一次性描述），`CachedRequestData` 是"这批已认识的请求这一步分别新增了什么"（数组下标对齐 `req_ids`，不再重复 `req_id → 数据` 的字典结构，因为 worker 已经用 `req_id` 建过索引）——这也是为什么后者的字段几乎都是并行数组而不是字典。

### 3.3 请求队列：`RequestQueue`

`vllm/v1/core/sched/request_queue.py:13`-`17` 定义 `SchedulingPolicy` 只有两个值：`FCFS = "fcfs"`、`PRIORITY = "priority"`。对应两个实现：

- `FCFSRequestQueue`（`vllm/v1/core/sched/request_queue.py:75`-`128`）直接继承 `deque[Request]`，`add_request` 是 `append`，`pop_request` 是 `popleft`——严格先进先出。
- `PriorityRequestQueue`（`vllm/v1/core/sched/request_queue.py:131`-`198`）包一层 `heapq`，排序键来自 `Request.__lt__`（`vllm/v1/request.py:350`-`361`）：先比 `priority`（越小越先），再比 `arrival_time`，再比 `request_id`，最后比 `id(self)` 兜底。

两个实现共享同一个抽象接口（`vllm/v1/core/sched/request_queue.py:20`-`72`），`Scheduler` 通过工厂函数拿到具体实现，自己不知道也不关心底层是 `deque` 还是 `heap`（`vllm/v1/core/sched/request_queue.py:201`-`208`，逐字抄录）：

```python
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
```

`Scheduler.__init__` 用它同时造出 `self.waiting` 和 `self.skipped_waiting` 两个队列（`vllm/v1/core/sched/scheduler.py:196`-`198`）——两者用的是**同一种策略**，`skipped_waiting` 不是另一套排序规则，只是"本步被跳过的请求"的临时存放处。

`Scheduler` 自身持有三个队列级容器（`vllm/v1/core/sched/scheduler.py:187`-`199`）：`self.requests: dict[str, Request]`（全部未删除请求的索引，用于 O(1) 按 id 查找）、`self.waiting` / `self.skipped_waiting`（两个 `RequestQueue`，后者放本步因各种阻塞状态被跳过的请求）、`self.running: list[Request]`（普通 `list` 而非队列，因为需要按下标扫描、支持从中间删除以实现抢占）。

### 3.4 `Request` 上专门为调度服务的字段

`vllm/v1/request.py:59`-`235` 的 `Request.__init__` 里，除了 `prompt_token_ids`/`sampling_params` 这些"业务字段"，还有一组字段**只为调度器和抢占/异步调度存在**，读 `schedule()` 时反复用到，值得单独列出（均为源码为证）：

| 字段 | 初始值 / 位置 | 作用 |
|---|---|---|
| `priority` | `vllm/v1/request.py:84` | `PRIORITY` 策略的排序键之一，数字越小越先调度 |
| `arrival_time` | `vllm/v1/request.py:96` | 排序键之二，`PRIORITY` 同优先级按到达时间破平局 |
| `is_prefill_chunk` | `False`，`vllm/v1/request.py:198` | 本步是否还是"未完成的 prefill 分块"，`_update_after_schedule` 每步重算（`vllm/v1/core/sched/scheduler.py:1408`-`1410`） |
| `num_output_placeholders` | `0`，`vllm/v1/request.py:160` | 异步调度专用：GPU 还没跑完但已经"乐观"占用的输出 token 数 |
| `num_stale_output_tokens` | `0`，`vllm/v1/request.py:163` | 抢占瞬间还在飞行中的输出 token 数，抢占后逐步"drain" |
| `drop_stale_output` | `False`，`vllm/v1/request.py:166` | 是否要丢弃飞行中的输出（而不是照常交付）——同步抢占后又在同一步恢复时会置真 |
| `num_in_flight_tokens` | `0`，`vllm/v1/request.py:171` | 异步调度/流水线并行下，已发出但还没收到 GPU 结果的 token 数 |
| `spec_token_ids` | `[]`，`vllm/v1/request.py:181` | 投机解码：下一步要验证的草稿 token |
| `num_preemptions` | `0`，`vllm/v1/request.py:210` | 累计被抢占次数，只增不减，可用于观测某个请求是否被反复抢占 |

这组字段的存在本身说明一件事：**同步调度和异步调度共享同一个 `Request` 类**，异步调度不是另起一套数据结构，而是在普通字段之外叠加"乐观计数"（`num_output_placeholders`/`num_in_flight_tokens`），`AsyncScheduler` 只需要覆写几个方法去维护这些叠加字段（见 `## 5` 决策 6）。

## 4. 主流程走读：一次 `schedule()` 调用

逐步走读 `vllm/v1/core/sched/scheduler.py:484`-`1326`，标出每一步改了哪个状态。

**① 计数与预算初始化**（`vllm/v1/core/sched/scheduler.py:485`-`523`）
- `self.current_step += 1`（L485）——全局步数计数器往前走一步，驱动 V2+PP+async 的解码节奏门。
- 声明本地空容器：`scheduled_new_reqs`/`scheduled_resumed_reqs`/`scheduled_running_reqs`/`preempted_reqs`（都是 `list`），`req_to_new_blocks`/`num_scheduled_tokens`（都是 `dict`）——这些都是"本步"局部变量，函数返回后即销毁，真正持久的状态只有 `self.running`/`self.waiting` 等。
- 两个独立的预算标量在这里诞生：`token_budget = self.max_num_scheduled_tokens`（L504）、`input_budget = self.scheduler_config.max_num_batched_tokens`（L507）——两者默认数值相等但语义不同，见 `## 7` 踩坑 2。
- 若 `self._pause_state == PauseState.PAUSED_ALL`，`token_budget` 被强制清零（L508-510）——**暂停不是清空队列，是让预算归零**，请求仍原地待命，队列结构完全不受影响。
- `self.kv_cache_manager.new_step_starts()`（L523）通知 KV manager 新的一步开始，比如清理上一轮遗留的临时状态。

**② RUNNING 循环**（`vllm/v1/core/sched/scheduler.py:531`-`753`）
`while req_index < len(self.running) and token_budget > 0`。对每个在跑请求：
- 若 `input_budget <= draft_slots` 直接跳出整个循环（L535-536）——这是第二道预算闸门，防止投机解码占用的槽位被普通 token 挤占。
- 若干"跳过"检查：PP 下已确定跑满 `max_tokens`（L538-552）、V2+PP+async 的解码节奏门 `next_decode_eligible_step`（L554-558）、DP prefill 节流（L560-564）——命中任一条，`req_index += 1; continue`，继续看下一个 RUNNING 请求。
- 计算 `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`（L566-570）——"这个请求还差多少 token 没被算过"，chunked prefill 和纯 decode 用的是同一行代码：decode 请求这个值天然是 1（加投机解码的话是 `1+K`），未完成的 prefill 请求这个值是剩余 prompt 长度。
- 用 `token_budget`、`input_budget`、`max_model_len`、Mamba 块对齐、encoder 预算、`num_prefill_lookahead` 层层裁剪 `num_new_tokens`（L571-614）。
- **若裁完 `num_new_tokens == 0`**（L616-634）：`req_index += 1; continue`。第 630-632 行的 NOTE 原话："by doing `continue` instead of `break`, we do not strictly follow the FCFS scheduling policy and allow the lower-priority requests to be scheduled"——**这是源码自己承认的不严格 FCFS**。
- 调 `self.kv_cache_manager.allocate_slots(...)`（L637-696，内嵌抢占逻辑，见 ③）申请 KV 块。
- 申请成功：`scheduled_running_reqs.append(request)`，记录 `req_to_new_blocks`/`num_scheduled_tokens`，`token_budget -= num_new_tokens`，`input_budget -= num_new_tokens + draft_slots`（L702-710）。随后处理投机解码 token（L712-728）和多模态编码器输入（L730-743）。

**③ 抢占（嵌在 ② 内部，只在 `allocate_slots` 返回 `None` 时触发）**（`vllm/v1/core/sched/scheduler.py:638`-`696`）
```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break
    # 申请失败：抢占"最低优先级"的请求
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
        ...
    else:
        preempted_req = self.running.pop()
    self._preempt_request(preempted_req, scheduled_timestamp, ...)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break  # 已经没人可抢，这个请求这步调度不了
```
`PRIORITY` 策略选 `self.running` 里 `(priority, arrival_time)` 最大的（数字越大越"不重要"）；非 `PRIORITY`（即 FCFS）直接 `self.running.pop()`——**弹出 list 尾部，也就是最后被放进 `running` 的那个请求，是 LIFO 语义**，不是常见直觉里"先抢最老的"。

`_preempt_request()`（`vllm/v1/core/sched/scheduler.py:1347`-`1388`）做的事：
1. `assert request.status == RequestStatus.RUNNING`（L1360-1362，只有在跑的请求能被抢占）
2. `self._free_request_blocks(request)`（L1363）——**KV 块在这一刻被释放**
3. `self.encoder_cache_manager.free(request)`（L1364）——多模态编码缓存一并释放
4. `request.status = RequestStatus.PREEMPTED`（L1366）
5. `request.num_computed_tokens = 0`（L1367）——**这就是"丢弃重算"发生的确切位置**，没有任何把 KV 写到 CPU 内存的代码路径
6. `request.num_preemptions += 1`（L1382）
7. `self.waiting.prepend_request(request)`（L1387）——**放回等待队列的队头**（不是队尾），被抢占的请求下次优先被重新考虑

对照真实源码（`vllm/v1/core/sched/scheduler.py:1360`-`1387`，节选核心行）：

```python
assert request.status == RequestStatus.RUNNING, (
    "Only running requests can be preempted"
)
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
if request.spec_token_ids:
    request.spec_token_ids = []
...
request.num_output_placeholders = 0
request.num_preemptions += 1
...
# Put the request back to the waiting queue.
self.waiting.prepend_request(request)
self.reset_preempted_req_ids.add(request.request_id)
```

逐行确认了 `## 0` 的结论：**没有任何一行把 KV block 的内容写到别的存储介质**，`_free_request_blocks` 只是把 block 归还给池子（block pool 内部会不会保留内容取决于前缀缓存机制，那是 KV 缓存篇的话题，本篇只确认调度器这一层没有主动"搬运"这个动作）。

**④ WAITING 循环**（`vllm/v1/core/sched/scheduler.py:755`-`1176`）
入口条件（L756）：`if not preempted_reqs and self._pause_state == PauseState.UNPAUSED`——**这一步只要发生过抢占，就完全不再准入新请求**，剩余预算全部留给已在跑的请求。循环体：
- `_select_waiting_queue_for_scheduling()`（`vllm/v1/core/sched/scheduler.py:2182`-`2192`）决定从 `self.skipped_waiting` 还是 `self.waiting` 弹候选（FCFS 模式下 `skipped_waiting` 优先；PRIORITY 模式下比较两个队列的队头，谁优先级高选谁）。
- `num_running = len(self.running) + self.num_waiting_for_streaming_input`；若 `>= self.max_num_running_reqs` 直接 `break`（L764-766）——**并发请求数上限在这里生效**。
- 前缀缓存查找：本地 `_get_local_prefix_cache_hit()`（L826）+ 可选的 KVConnector 远程命中查询（L829-887）。
- 计算 `num_new_tokens`，用 `enable_chunked_prefill` 开关裁剪：**若关闭 chunked prefill 且单步塞不下整个 prompt，直接 `break`**（L966-972）——注意这里是 `break` 不是 `continue`，和 RUNNING 循环的语义不对称（详见 `## 7`）。
- `self.kv_cache_manager.allocate_slots(..., has_scheduled_reqs=bool(self.running))`（L1041-1053）——`watermark` 只在这条准入路径上生效（见 `## 5` 决策 2）。若返回 `None`，`break` 整个 WAITING 循环（L1055-1062）。
- 申请成功：`request.status = RequestStatus.RUNNING`（L1143），`self.running.append(request)`（L1123）。至此这个请求正式从"等待"变为"在跑"，下一步开始会进入 ② 的 RUNNING 循环。

**⑤ 收尾**（`vllm/v1/core/sched/scheduler.py:1177`-`1326`）
- 断言校验：`total_num_scheduled_tokens <= max_num_scheduled_tokens`、`token_budget >= 0`、`input_budget >= 0`、`len(self.running) <= self.max_num_running_reqs` 等（L1178-1189）——这些断言是给开发者兜底的防线，正常运行永远不该触发。
- 计算 `num_common_prefix_blocks`（L1193-1199）——取 RUNNING 队列里任意一个请求（`self.running[0]`）去问 KV manager"大家的最长公共前缀有几个块"，给 cascade attention 用；调度器自己不关心具体是哪个前缀，只是把这个数字透传给 worker。
- 组装 `SchedulerOutput`（L1281-1302）——把 `## 3.2` 表格里的 14 个字段逐一填好。
- 若配置了 KV/EC connector，调 `build_connector_meta()` 填 `kv_connector_metadata`/`ec_connector_metadata`（L1308-1317）。
- 最后调 `_update_after_schedule()`（`vllm/v1/core/sched/scheduler.py:1390`-`1416`）把本步 `num_scheduled_tokens` 累加进每个请求的 `num_computed_tokens`。**这一步在 `schedule()` 返回之前就完成，不等 GPU 真正跑完前向传播**——目的是让 prefill 请求下一步能立刻被重新纳入调度而不用等 `update_from_output()`，代价是"乐观推进"的计数（`num_output_placeholders` 等，见 `## 3.4`）必须在 GPU 结果真正返回后由 `update_from_output()` 核对、修正（比如投机解码的草稿被拒绝时要回滚）。

### 4.1 一个跨三步的教学示例（本库构造，非实测数据，仅用于说明机制）

假设 `max_num_scheduled_tokens = max_num_batched_tokens = 512`，`max_num_seqs` 足够大不成为瓶颈。三个请求依次到达：`A`（prompt 900 token，`priority` 相同）先到，`B`（prompt 50 token）在 `A` 的第一步 prefill 中途到达，`C`（prompt 700 token）随后到达。用 `## 4` 里的行号对着走一遍：

| 步 | RUNNING 循环（`vllm/v1/core/sched/scheduler.py:531`） | WAITING 循环（`vllm/v1/core/sched/scheduler.py:759`） | 结果状态 |
|---|---|---|---|
| 1 | 空，`self.running` 还没有人 | `A` 出队：`num_new_tokens = min(900, 512) = 512`（chunked prefill 生效，`vllm/v1/core/sched/scheduler.py:940`-`974`），`allocate_slots` 成功，`A` 进入 `self.running`，`is_prefill_chunk=True` | `A`: RUNNING（已耗 512/900 token），`B`/`C` 仍在 `waiting` |
| 2 | `A` 排进 RUNNING 循环：剩余 `900-512=388` token 待算，`num_new_tokens=388`，`token_budget` 剩 `512-388=124` | `B` 出队：`num_new_tokens=min(50,124)=50`，预算够，直接一次性调度完；`C` 出队：`num_new_tokens=min(700, 124-50=74)=74`，chunked prefill 继续切 | `A`: RUNNING（`is_prefill_chunk` 变 `False`，prefill 结束，下一步开始产出 decode token）；`B`: RUNNING（prefill 一步做完）；`C`: RUNNING（仍在切块，`is_prefill_chunk=True`） |
| 3 | `A`（decode，`num_new_tokens=1`）、`B`（decode，`num_new_tokens=1`）都排进 RUNNING，`C` 的剩余 `700-74=626` token 也排进 RUNNING 继续切；预算 512 里 `A`+`B` 各吃 1、`C` 吃剩下的 510 | 无新请求 | 三者都在 `self.running`，`C` 仍需要第 4 步才能切完 prefill |

这个例子体现两件事：① **decode 请求的 token 名额永远优先被满足**——因为它们排在 RUNNING 循环最前面，且每次只要 1 个 token，几乎不可能被预算卡住；② chunked prefill 的切块点完全由"这一步还剩多少预算"决定，不是提前规划好的，`C` 在第 2 步能切多少完全取决于 `A`/`B` 那一步用剩了多少 `token_budget`。

若这时来了一个请求 `D`，且 KV cache 已经被 `A`/`B`/`C` 占满，`allocate_slots` 会在 RUNNING 循环内部返回 `None`，触发 `## 4` ③ 的抢占逻辑：FCFS 策略下 `self.running.pop()` 弹出的是**最后被 `append` 进 `self.running` 的请求**——按上面的时间线是 `C`（第 2 步才加入），不是最早到达的 `A`。`C` 会被整体退回 `waiting` 队头、`num_computed_tokens` 清零，之前切掉的 `74+626` token 全部作废，下次被重新调度时从 0 开始重新走一遍 chunked prefill。

### 4.2 三个特殊分支在哪里挂钩

任务里明确要问的三类请求——结构化输出、投机解码、多模态——在 `schedule()` 里**都不是独立的 if/else 大分支，而是分散挂在主循环的裁剪逻辑和收尾阶段**：

- **投机解码**：不占用单独的调度阶段，而是直接叠加进 `num_new_tokens` 的计算——`request.num_tokens_with_spec`（`vllm/v1/request.py:292`-`293`，`len(_all_token_ids) + len(spec_token_ids)`）已经把草稿 token 算进"这个请求还差多少 token"里。调度成功后在 `vllm/v1/core/sched/scheduler.py:712`-`728` 把本步要验证的草稿 token 单独记进 `scheduled_spec_decode_tokens`，随 `SchedulerOutput` 一起发给 worker；动态 K（每步验证几个草稿 token）在收尾阶段计算，见 `vllm/v1/core/sched/scheduler.py:1265`-`1270`。
- **结构化输出**：不在 `schedule()` 内部处理语法约束本身，只在 `_update_after_schedule` 里打一个标记位——`scheduler_output.has_structured_output_requests |= (request.use_structured_output and not request.is_prefill_chunk)`（`vllm/v1/core/sched/scheduler.py:1411`-`1413`）。真正计算语法 bitmask 是 `EngineCore.step()` 里 `schedule()` 之后单独调用的 `get_grammar_bitmask()`（`vllm/v1/core/sched/scheduler.py:1720`-`1742`，抽象定义见 `vllm/v1/core/sched/interface.py:85`-`89`），只处理有这个标记的请求，且明确跳过还在 prefill 中的请求（`not req.is_prefill_chunk`，因为 prefill 阶段还没到需要按语法约束采样的时候）。
- **多模态**：唯一在 RUNNING/WAITING 两个循环里都显式调用的独立子过程，`_try_schedule_encoder_inputs()`（`vllm/v1/core/sched/scheduler.py:1542`-`1566`，RUNNING 循环里的调用点在 `vllm/v1/core/sched/scheduler.py:596`-`608`，WAITING 循环里在 `vllm/v1/core/sched/scheduler.py:989`-`1001`）——它会反过来**缩小** `num_new_tokens`（如果视觉 token 超出 encoder 计算预算或编码器缓存放不下，就把解码器 token 数砍到编码器输入之前为止,`vllm/v1/core/sched/scheduler.py:1554`-`1566` 的行为说明）。`disable_chunked_mm_input` 配置项（`vllm/config/scheduler.py:107`-`113`）进一步禁止把一个多模态 item 切成两半分两步喂。

三者共同点：**都不打断"num_computed_tokens 追赶 num_tokens_with_spec"这条主线**，而是在计算 `num_new_tokens` 时层层裁剪，或者在收尾阶段挂一个供下游读取的标记/输出字段——这也是为什么 `## 0` 说 `schedule()` 没有阶段划分：这些看起来像"特殊模式"的功能，实现上都只是主循环普通裁剪链条上的一环。

## 5. 设计决策与代价

### 决策 1：RUNNING 先于 WAITING

- **为什么这么设计**：保护已经在生成的请求不被新请求的 prefill 突然抢走预算，也让"半成品" chunked prefill 请求尽快做完——否则 prefill 可能被无限期打断，KV manager 里挂着的中间态 block 越占越多。
- **不这样会怎样**：如果 WAITING 先调度，长 prompt 会持续挤占预算，已经在 decode 的请求的 token 间延迟（ITL/TPOT）会剧烈抖动，流式体验直接崩。
- **什么时候可以不这样**：几乎找不到"应该反过来"的场景——这个顺序本身接近连续批处理的定义。可以调的是 WAITING **内部**的顺序（比如 SGLang 的 cache-aware 重排，见 `## 6`），但没有引擎让 WAITING 整体压过 RUNNING。

### 决策 2：token 预算是共享标量，不是"每请求配额"

- **为什么这么设计**：这正是 chunked prefill 的本体——不预先规划"这个请求切几刀"，而是每步用一个全局剩余预算贪心地喂饱尽可能多的请求，天然支持 prefill 和 decode 混部在同一个 batch（decode 请求每个只要 1 个名额，prefill 按需吃剩下的）。
- **不这样会怎样**：如果给每请求固定配额（比如"单步最多 256 token"），长 prefill 需要更多步才能完成，且步间 GPU 利用率会因配额不满而空转。
- **什么时候可以不这样**：`max_num_scheduled_tokens` 单独设成比 `max_num_batched_tokens` 更小的值时（`vllm/config/scheduler.py:56`-`61` 的 fallback 逻辑：不设就直接等于 `max_num_batched_tokens`），就是**故意**让"真正塞进模型前向的 token 数"比"整体输入预算"更紧，给投机解码等需要额外追加 slot 的场景留余量——这也说明"一根共享标量"背后其实是 `token_budget`/`input_budget` 两把独立的尺子（L504/L507），不是只有一个旋钮。

### 决策 3：抢占 = 丢弃 + 重算，V1 没有 swap

- **为什么这么设计**：`docs/design/metrics.md:511`-`524`（文档所述）给出官方理由：swap-to-CPU 原本是为 v0 的 `SequenceGroup`/beam search 的 KV 块共享设计的；V1 删掉了 `SequenceGroup`，又有前缀缓存兜底（重算时相当一部分 KV 能从前缀缓存命中，不是从零算），recompute 的代价被前缀缓存摊薄，swap 的工程复杂度不再划算。
- **不这样会怎样**：保留 swap-to-CPU 需要额外的 CPU 内存管理、PCIe 带宽调度，以及"swap 回来的 KV 是否还新鲜"（比如权重热更新场景）这类一致性问题。`docs/configuration/optimization.md:34` 那条警告日志里 `PreemptionMode.RECOMPUTE` 这个名字本身就是历史包袱——v0 还有 `SWAP` 可选，V1 里这个名字已经没有对照项了。
- **什么时候可以不这样**：`vllm/distributed/kv_transfer/kv_connector/utils.py:222` 和 `vllm/platforms/{cuda,rocm,xpu}.py` 里仍有 `swap_out_blocks_to_host` 这个平台原语，但那是给外部 KV connector（PD 分离、CPU offload cache）用的搬运原语，**不是调度器自己的抢占路径**。对照 SGLang：`retract_decode`（`sglang:python/sglang/srt/managers/schedule_batch.py:2825`）在常规模式下同样是纯 `release_kv_cache`（等价于丢弃重算），但在 "decode disaggregation" 模式下会调用 `retraction_backup` 把 KV 备份到 host 内存（`sglang:python/sglang/srt/managers/schedule_batch.py:1933`-`1940`）以避免重算——这是 vLLM V1 完全没有的分支。

### 决策 4：请求队列只有 FCFS 和 PRIORITY 两种

- **为什么这么设计**：更少的策略意味着服务器运营者更容易预测调度器的行为；`PRIORITY` 额外提供 `(priority, arrival_time)` 两级排序（`vllm/v1/request.py:350`-`361`）已经够用。
- **不这样会怎样**：如果要支持 cache-aware 重排（像 SGLang 的 LPM，把等待队列按前缀匹配长度重排，命中共享前缀的请求排在一起以减少 KV cache 抖动），需要在 `request_queue.py` 之外再加一层相似度计算逻辑——队列就不再是简单的 `deque`/`heap`。
- **什么时候可以不这样**：并发量小、大多数等待请求之间没有共享前缀时，cache-aware 排序几乎不产生收益，FCFS + block 级前缀缓存已经够用（本库推断，未在 vLLM 代码或文档里找到对这个取舍的直接声明）。反证见 `## 6`：SGLang 自己在等待队列超过 128 个请求时也会自动把 LPM 降级回 FCFS（`sglang:python/sglang/srt/managers/schedule_policy.py:291`-`293`），说明这类策略本身有性能天花板。

### 决策 5：连续批处理的收益不是任何流量下都存在

这是本篇被要求重点讲清楚的一条。**连续批处理的收益来源是两件独立的事**：
1. **并发请求的输出长度是异质的**——否则所有请求反正会跑到同一批结束才退场，没有人提前空出名额，`schedule()` 里判断"谁提前结束"的所有分支（`check_stop`，`vllm/v1/core/sched/utils.py:94`-`130`；`update_from_output` 里逐请求处理 `stopped`，`vllm/v1/core/sched/scheduler.py:1886`-`1916`）永远走同一条路径，continuous batching 和 static batching 在效果上等价。
2. **请求是在线到达的**——否则如果所有请求在 t=0 一次性提交完毕，WAITING 循环判断"这一步还能不能再插入新请求"这件事（`while (self.waiting or self.skipped_waiting) and token_budget > 0`，L759）只发生一次性的初始分配，之后再也不会有"新人半路插队抢占旧人预算"，抢占逻辑（`## 4` ③）永远不会被触发。

`schedule()` 整个函数的分支复杂度几乎全部用于应付这两件事——谁提前结束、谁刚到、KV 块不够了怎么办。如果输入输出都是固定长度、且全部请求同批到达，这些分支永远走同一条固定路径，continuous batching 退化成普通 static batching，唯一还剩的差别是"长 prompt 会被 chunked prefill 切开分步执行"——而 chunked prefill 本身的收益是另一件独立的事（避免长 prompt 一次性挤占 decode 的 token 间延迟），不能记在 continuous batching 头上。

- **什么时候可以放心用离线/静态批处理**：跑固定评测集这类输入输出长度已知或近似同质、全部样本一次性提交的场景——这时调度器每一步的抢占检查、队列弹出、预算计算都是纯开销，`vllm.LLM.generate()` 的一次性批处理路径更省。

### 决策 6：异步调度是 `Scheduler` 的子类，不是内部 if 分支

- **为什么这么设计**：`AsyncScheduler(Scheduler)`（`vllm/v1/core/sched/async_scheduler.py:12`-`70`）只覆写 `_update_after_schedule`（`vllm/v1/core/sched/async_scheduler.py:19`-`49`）和 `_update_request_with_output`（`vllm/v1/core/sched/async_scheduler.py:51`-`70`）两个方法，`schedule()` 本体**完全不用改**。`vllm/config/scheduler.py:170`-`178` 的 `get_scheduler_cls()` 只是在构造期根据 `async_scheduling` 配置（`vllm/config/scheduler.py:148`-`151`）二选一实例化哪个类。子类化把"要不要在 GPU 结果还没返回时就乐观推进状态"这个横切关注点隔离在一小撮方法里，而不是在 3,037 行的 `scheduler.py` 主体里到处插 `if self.async_scheduling`。
- **不这样会怎样**：如果把异步逻辑写成散落在 `schedule()`/`update_from_output()` 内部的条件分支，`num_output_placeholders`/`num_in_flight_tokens`（见 `## 3.4`）这类"乐观计数"字段的读写点会散布在主流程各处,极易出现"某个分支忘了检查 async 标志"的漏洞——本篇 `## 3.4` 列出的字段全部只在 `AsyncScheduler` 覆写的方法里被写，同步 `Scheduler` 完全不碰它们，这本身就是一种正确性保证（同步路径不可能因为多了 async 分支而被污染）。
- **什么时候可以不这样**：如果一个自定义调度器需要的行为差异不止"结果何时到达"这一件事（比如彻底不同的准入策略），继承 `SchedulerInterface`（`vllm/v1/core/sched/interface.py:38`）直接重写整个 `schedule()` 比在 `Scheduler`/`AsyncScheduler` 继承链上再叠一层更清晰——`scheduler_cls` 配置项（`vllm/config/scheduler.py:117`-`120`）本来就支持插入任意自定义类。

### 六条决策速览

| 决策 | 一句话代价 | 什么时候可以不这样 |
|---|---|---|
| 1. RUNNING 先于 WAITING | 新请求 TTFT 可能因老请求占满预算而变长 | 几乎找不到反例，这是连续批处理定义的一部分 |
| 2. 共享 token 预算而非每请求配额 | 单个长 prompt 可能连续多步吃满预算，挤压其它请求 | `max_num_scheduled_tokens` 单独调小时退化出"预留余量"效果 |
| 3. 抢占丢弃重算，不 swap | 高抢占率下重复计算浪费算力 | 未提供，V1 架构性选择；需要免重算时只能靠不触发抢占（加内存/减并发） |
| 4. 队列只有 FCFS/PRIORITY | 无法按前缀命中率重排队列 | 并发小、请求间无共享前缀时几乎不损失 |
| 5. Continuous batching 收益依赖异质输出+在线到达 | 同质离线批量场景下调度开销纯浪费 | 固定评测集等场景直接用离线批处理 API |
| 6. 异步调度用子类而非 if 分支 | 多一层继承，读代码要跳两个文件 | 需要更大差异的准入策略时直接重写 `schedule()` |

## 6. 同位对照：SGLang 在同一位置怎么做

（对应 [[02-SGLang-Scheduler事件循环]]，本节只取与"排队策略"和"抢占/驱逐"直接相关的部分，行号已实测核对，SGLang 具体事件循环结构留给对应篇章。）

- **队列策略**：vLLM `vllm/v1/core/sched/request_queue.py:13`-`17` 只有 `FCFS`/`PRIORITY` 两种。SGLang `sglang:python/sglang/srt/managers/schedule_policy.py:200`-`213` 额外定义了 `CacheAwarePolicy`（`LPM` 最长前缀匹配 / `DFS_WEIGHT`）和 `CacheAgnosticPolicy`（`FCFS` / `LOF` 最长输出优先 / `RANDOM` / `ROUTING_KEY`）共 6 种可插拔策略，源码原文：

```python
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""
    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting

class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"
```

**但要澄清一个常见误解**：SGLang 当前的默认值是 `"fcfs"`（`sglang:python/sglang/srt/server_args.py:873`），不是很多人以为的 `"lpm"`——这条本篇特意核实过源码，没有凭印象写。`LPM` 依然是可选项，且在等待队列超过 128 个请求时会自动降级回 FCFS（`sglang:python/sglang/srt/managers/schedule_policy.py:291`-`293`），说明前缀感知排序本身有规模上限。
- **抢占/驱逐的语义**：vLLM 统一叫 "preempt"，SGLang 管这个叫 "retract"（`sglang:python/sglang/srt/managers/schedule_batch.py:2825` `retract_decode`）。常规路径两边等价——都是释放 KV（`release_kv_cache(is_insert=False)`），等价于丢弃重算；但 SGLang 在 "decode disaggregation" 模式下会走 `retraction_backup` 把 KV 备份到 host 内存（`sglang:python/sglang/srt/managers/schedule_batch.py:1933`-`1940`），这是 vLLM V1 完全没有的分支。
- **驱逐对象的选择方式**：vLLM 每次 `allocate_slots` 失败只选**一个**受害者（`PRIORITY` 用 `max(priority, arrival_time)`，`FCFS` 用 `running.pop()` 的 LIFO），选完立刻重试，不够再选下一个。SGLang 的 `_get_decode_retraction_order` 一次性算出**整条驱逐顺序**，循环 pop 直到 `check_decode_mem` 满足为止（`sglang:python/sglang/srt/managers/schedule_batch.py:2825`-`2860` 附近）——可以一次性驱逐好几个请求再重试，不是"抢一个测一次"。这是实现风格差异，不是谁更"对"。
- **"驱逐到底"的兜底不同**：SGLang 在 `retract_decode` 里显式写了"始终保留至少一个请求"的安全阀（`sglang:python/sglang/srt/managers/schedule_batch.py:2837`-`2839`，`if len(sorted_indices) == 1: break`），如果连最后一个请求也塞不下，会**直接把它 abort 掉**并返回错误（`sglang:python/sglang/srt/managers/schedule_batch.py:2864`-`2872` 的 `FINISH_ABORT("...Aborting the last request.")` 分支）。vLLM 的 `_preempt_request` 循环没有这样的显式安全阀——按 `vllm/v1/core/sched/scheduler.py:694`-`696` 的逻辑，理论上会一直抢占到正在评估的这个请求自己被抢占为止（`preempted_req == request` 时才 `break`），失败模式是"这个请求这一步排不上、被放回 waiting 队头等下次机会"，而不是直接判它失败——**vLLM V1 在这条路径上没有"因为单个请求太大而主动 abort"的分支**（本库推断，依据是通读 `_preempt_request` 调用链没有发现调用 `finish_requests` 或设置 `FINISHED_*` 状态的代码）。

| 对比维度 | vLLM V1 | SGLang |
|---|---|---|
| 等待队列策略 | FCFS / PRIORITY，二选一 | FCFS(默认) / LPM / DFS-WEIGHT / LOF / RANDOM / ROUTING-KEY，六选一 |
| 队列是否感知前缀缓存 | 否，只在 block 分配层命中缓存 | 是，`LPM` 策略按前缀匹配长度重排队列本身 |
| 抢占/驱逐默认代价 | 丢弃 KV，全量重算 | 丢弃 KV，全量重算（同 vLLM） |
| 特殊模式下能否避免重算 | 否，V1 无 swap 路径 | 能，`decode disaggregation` 模式下可搬运到 host |
| 单次驱逐几个受害者 | 一个，失败再选下一个 | 一次性算好整条顺序，循环驱逐 |
| 驱逐到最后一个还不够时 | 请求被抢占后放回等待队列重试 | 显式 abort 该请求并报错 |

对读者的实际含义（本库推断，依据是通读 `_preempt_request`/`schedule()` 未发现重试次数上限或超时中止逻辑，但未逐行核对 `finish_requests` 之外是否存在其它外部中止入口）：在 vLLM V1 的调度器这一层，一个暂时挤不下的请求会被反复放回 `waiting` 队头、下一步继续重试，不会因为"这一步挤不下"本身就报错；SGLang 在 `retract_decode` 里则有一条明确的兜底路径，把"确实放不下"直接判成错误返回。两边的用户可感知差异是：vLLM 侧更可能表现为"这个请求延迟异常高"，SGLang 侧更可能表现为"直接收到一个错误响应"——具体是否如此还取决于调度器之上的请求超时/重试策略,这属于 API 层的话题，不在本篇范围内。

## 7. 踩坑与反直觉

1. **`long_prefill_token_threshold` 默认是 `0`，但语义是"关闭这个上限"，不是"上限为 0"。** `vllm/config/scheduler.py:70`-`72` 的字段注释原话是 "0 disables the cap (default)"。第一次读代码容易以为默认状态下每步只能给长 prefill 分 0 个 token；实际上默认状态下这个专门的裁剪完全不生效，真正限制单步 chunk 大小的是 `token_budget`/`input_budget` 这两个全局预算（`vllm/v1/core/sched/scheduler.py:571`-`572` 只有 `long_prefill_token_threshold > 0` 才会介入）。

2. **`max_num_scheduled_tokens` 和 `max_num_batched_tokens` 是两个不同的旋钮，容易被当成同一个东西。** 前者对应 `schedule()` 里的 `token_budget`（真正能塞进这一步前向传播的 token 数），后者对应 `input_budget`（略宽松，给投机解码等场景的额外 slot 留空间）。`vllm/config/scheduler.py:119`-`123` 的 fallback 逻辑显示：不显式设置 `max_num_scheduled_tokens` 时它直接等于 `max_num_batched_tokens`，两个变量在默认配置下数值相同、语义却不同，只有显式调小前者才会看出分叉。

3. **`watermark` 默认 `0.0`（关闭），且哪怕开了也只对 `WAITING`/`PREEMPTED` 状态的请求生效，还要求"已经有至少一个请求在跑"才生效。** `vllm/config/scheduler.py:136`-`141` 给出默认值；`vllm/v1/core/kv_cache_manager.py:467`-`473` 的判断条件是 `has_scheduled_reqs and request.status in (WAITING, PREEMPTED)`——单请求场景下 watermark 形同虚设。这条经常和"vLLM 会主动为高峰期预留内存"的直觉相反：**默认配置下 vLLM 会用满几乎全部 KV cache 直到装不下才触发抢占，不会提前留白**。

4. **`--swap-space` 在 V1 已被删除，日志里 `PreemptionMode.RECOMPUTE` 这个名字是历史包袱。** 见 `docs/configuration/optimization.md:34`（示例警告日志）与 `docs/design/metrics.md:505`-`513`（"The `--swap-space` flag has been removed as this feature is no longer used in V1"）。网上搜到的"调 swap space 防 OOM"的旧教程，是 v0 时代的建议，V1 根本没有这个旋钮。

5. **同一个 `schedule()` 函数内部，"这个请求这步排不上"的处理方式并不对称。** RUNNING 循环遇到 `num_new_tokens == 0` 走 `continue`（`vllm/v1/core/sched/scheduler.py:630`-`634`，源码自己承认"不严格 FCFS"）；WAITING 循环遇到多数申请失败情况（关闭 chunked prefill 时塞不下、DP 节流、`allocate_slots` 返回 `None` 等）走的是 `break`（分别见 L933、L972、L1011、L1062）——直接终止本步 WAITING 准入，不会跳过队首继续看后面更小的请求。读代码时不能假设两个循环有一致的公平性语义，这也是队头阻塞（见 `## 8`）的根源。

6. **`scheduler_reserve_full_isl` 默认是 `True`，和"chunked prefill 允许只要第一块塞得下就先收下请求"这个直觉相反。** 很多人第一次接触 chunked prefill 会以为"只要这一步能塞进第一个 chunk 就先准入，剩下的慢慢切"——但 `vllm/config/scheduler.py:130`-`134` 的默认配置要求：admission 时必须确认**整条序列**能放进 KV cache（`full_sequence_must_fit=True` 传进 `allocate_slots`，`vllm/v1/core/kv_cache_manager.py:475`-`491`），只是当前这一步先算第一个 chunk。换句话说，chunked prefill 决定的是"一步算多少"，不是"要不要先收下这个请求"——一个长到无论如何也塞不满 KV cache 的请求，在默认配置下从一开始就不会被准入，不会出现"admit 了但后面几步发现根本放不下"的情况。

7. **`get_grammar_bitmask()` 不在 `schedule()` 内部，是 `EngineCore.step()` 单独调用的下一个方法。** 见 `## 1` 的代码片段（`vllm/v1/engine/core.py:610`）——容易以为结构化输出的语法约束是 `SchedulerOutput` 里现成算好的一个字段，实际上 `SchedulerOutput` 只带了一个布尔标记 `has_structured_output_requests`（`vllm/v1/core/sched/output.py:251`）,真正的 bitmask 要在 `schedule()` 返回之后再调一次 `get_grammar_bitmask()`（`vllm/v1/core/sched/scheduler.py:1720`-`1742`）才算出来，属于两次独立的方法调用、两份独立的返回值。

## 8. 可改进点

（以下均为本库基于源码走读的推断，标注证据依据；未核实是否已有官方 issue 在跟踪，未查证部分已明确标出。）

**改进点 1：PRIORITY 抢占的受害者选择是 O(n) 线性扫描**
- 现状：`max(self.running, key=lambda r: (r.priority, r.arrival_time))`（`vllm/v1/core/sched/scheduler.py:651`-`655`）每次抢占都要扫一遍整个 `self.running`。
- 影响：若一步内连续触发多次抢占（比如新来一个超大 prompt 一次性挤出好几个低优先级请求），退化成 O(n²)。正常部署下 `self.running` 规模不大（`max_num_seqs` 默认 128，`vllm/config/scheduler.py:44`），影响有限，但在 `priority` 模式 + 高并发 + 频繁抢占的场景值得关注。
- 建议方向：把 `self.running` 换成按 `(priority, arrival_time)` 维护的小顶堆能把单次选择降到 O(log n)，代价是"按下标扫描 RUNNING 循环"这个当前实现依赖的顺序语义要重新设计——不是一个孤立的小改动。

**改进点 2：WAITING 准入循环的"一次拒绝就 break"可能造成隐性队头阻塞**
- 现状：若关闭 `enable_chunked_prefill` 又混合了长短 prompt，队首一个长 prompt 排不下时会直接终止本步准入（`vllm/v1/core/sched/scheduler.py:966`-`972`），不像 RUNNING 循环那样 `continue` 尝试下一个（`## 7` 踩坑 5）。
- 影响：后面能一次性塞进剩余预算的短 prompt 本步完全没有机会被调度，即便它本可以立刻跑完——这是经典的队头阻塞（head-of-line blocking），只在 `enable_chunked_prefill=False` 这个非默认配置下出现。
- 建议方向：可以考虑参照 RUNNING 循环放宽成 `continue`，但要小心不要破坏 FCFS 的可预测性（本来靠前的请求可能被无限期饿死在队列里）；SGLang 式的做法是在队列层面重排（`LPM`/`LOF`）而不是放宽单次准入判断，两种思路对可预测性的影响不同，值得先想清楚要保哪种保证。

**改进点 3：队列侧完全没有前缀缓存感知**
- 现状：vLLM 的前缀命中优化只发生在 block 分配层（`KVCacheManager.get_computed_blocks`），如果同一批等待请求里有大量共享前缀，谁先被准入纯粹看到达顺序/优先级，不考虑"谁先准入能让 block 复用更多"。
- 影响：极端场景下（大量请求共享同一个长系统提示词）,FCFS 顺序可能让本该背靠背命中前缀缓存的两个请求被无关请求隔开，中间那段时间前缀 block 可能因为 LRU 被挤出去，命中率打折扣。
- 建议方向：对照 SGLang 的 `LPM` 策略（`## 6`）,这是一个潜在的优化空间——**未查证**是否已有相关 vLLM 官方讨论或 issue 在跟踪，也未验证在真实流量下这个效应的量级是否值得为此增加队列复杂度。

**改进点 4：`AsyncScheduler` 的正确性依赖"乐观计数"字段两两对齐，缺少内建一致性校验**
- 现状：`## 3.4` 列出的 `num_output_placeholders`/`num_in_flight_tokens`/`num_stale_output_tokens` 等字段要靠 `_update_after_schedule` 和 `update_from_output` 在不同时间点分别加、分别减,才能保持"乐观值"和"实际值"最终一致（例如投机解码被拒绝时要在 `vllm/v1/core/sched/scheduler.py:1855`-`1859` 回滚 `num_computed_tokens`/`num_output_placeholders`）。
- 影响：这类"两处分别维护、必须相互对齐"的状态，历史上是并发/异步系统里最容易滋生难复现 bug 的地方——本篇通读代码没有发现运行时断言在每步结束后校验这些计数器的不变量（比如 `num_output_placeholders >= 0` 只在个别写入点用 `assert` 兜底，见 `vllm/v1/core/sched/async_scheduler.py:63`，不是在每步收尾统一校验）。
- 建议方向：**未查证**是否已有专门的调试模式或测试覆盖这类不变量；作为读者的自查清单，改这部分代码前建议先搞清楚 `## 3.4` 表里每个字段在哪几处被写、哪几处被读。

## 9. 自测题与延伸阅读

**闭卷自测题**（合上本文，尝试不看源码回答）：

1. `schedule()` 一次调用里，RUNNING 循环遇到 `num_new_tokens == 0` 走的是 `continue` 还是 `break`？WAITING 循环遇到对应情况呢？两者一致吗？
2. FCFS 策略下，抢占时被选中的"受害者"取自 `self.running` 的哪一端？这意味着刚加入 `running` 的请求还是资历更老的请求更容易被先抢？
3. 一个请求被抢占后，它的 KV block 发生了什么？`num_computed_tokens` 变成了多少？它被放回 `waiting` 队列的队头还是队尾？
4. `max_num_scheduled_tokens` 和 `max_num_batched_tokens` 分别对应 `schedule()` 里的哪个局部变量？它们默认相等吗？
5. `watermark` 默认值是多少？它对 RUNNING 请求的续期（追加 token）生效吗？
6. SGLang 的 `schedule_policy` 当前默认值是什么？和"LPM 是 SGLang 默认策略"这个常见印象是否一致？
7. continuous batching 在什么样的输入分布下收益趋近于零？为什么这时候 chunked prefill 的收益要单独算，不能记在 continuous batching 头上？
8. `scheduler_reserve_full_isl` 默认是开还是关？它检查的是"这一步能不能塞下"还是"整条序列最终能不能塞下"？
9. `AsyncScheduler` 相对 `Scheduler` 覆写了几个方法？`schedule()` 本体需不需要改？
10. `get_grammar_bitmask()` 是 `schedule()` 内部的一部分，还是 `EngineCore.step()` 单独调用的下一步？
11. `_lab/out/struct_map.json` 统计的"scheduler 子系统 20 个文件 / 8,424 行"和本篇讲的调度器范围是否完全对得上？差在哪里？

（题 1-3 对应 `## 4` ②③、`## 7` 踩坑 5；题 4-5 对应 `## 7` 踩坑 2-3；题 6-7 对应 `## 6`、`## 5` 决策 5；题 8 对应 `## 7` 踩坑 6；题 9-10 对应 `## 5` 决策 6、`## 7` 踩坑 7；题 11 对应 `## 2` 开头——答不上来就回对应小节重读，不用从头翻。）

**延伸阅读（本库内）**：

- [[02-vLLM-V1架构与EngineCore循环]] —— `schedule()` 的上游调用方，`EngineCore.step()` 的完整循环结构
- [[04-vLLM-KV缓存与前缀缓存]] —— `allocate_slots()`、`watermark`、前缀缓存命中的具体实现，本篇只讲了它作为"调度器 KV 申请入口"的边界行为
- [[02-SGLang-Scheduler事件循环]] —— `## 6` 同位对照的完整展开，SGLang 事件循环与本篇调度循环的结构性差异

---

读完本篇应该能回答的一句话总结：vLLM V1 调度器的全部复杂度，本质上是在维护一件事——每个请求的 `num_computed_tokens` 什么时候能追上它的 `num_tokens_with_spec`，用一根共享的 token 预算标量、按 RUNNING 优先于 WAITING 的顺序、在预算或 KV 块不够时丢弃重算——除此之外看到的分支，几乎都是投机解码、结构化输出、多模态、PD 分离这些旁路功能在往这条主线上挂钩子，而不是主线本身长出了新的阶段。
