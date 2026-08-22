# SGLang Scheduler 与事件循环

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：调度策略与前缀缓存共享同一棵树，且默认"能 prefill 就先 prefill"。

## 0. 结论先行

- **三进程模型**：`TokenizerManager` 活在主/HTTP 进程里（不是独立 `mp.Process`），`Scheduler` 每个 TP/PP rank 一个独立进程，`DetokenizerManager` 也是独立进程。三者之间只用 ZMQ 的 PUSH/PULL 单向管道串联，没有共享内存、没有 RPC 调用。
- **批类型判据是硬编码的优先级，不是预算竞争**：`get_next_batch_to_run()` 里只要能凑出新的 prefill 批，就无条件先跑它；decode 批只在"这一步没有新 prefill 可跑"时才被选中。这与 vLLM V1 完全不同——vLLM 根本没有"prefill 批 / decode 批"的类型区分，一切请求在同一个 token 预算循环里被统一处理。
- **调度策略与前缀缓存共用同一棵树**：LPM（Longest Prefix Match）策略直接查询 `tree_cache`（也就是 RadixAttention 的那棵 radix 树）来给等待队列打分排序。这是 SGLang 与 vLLM 调度器最本质的架构差异——vLLM 的调度队列（FCFS/PRIORITY）完全不看前缀缓存的内容。
- **overlap（重叠）调度靠多 CUDA stream + 事件，不是独立线程**：单个 Python 线程把不同工作发射到 `schedule_stream`/`forward_stream`/`copy_stream` 三条 CUDA 流上，靠 `Event` 表达依赖关系，物理上的并发执行发生在 GPU 侧的流调度器里。
- **retract（显存不足时的回退）默认不给受害请求任何插队特权**：被退回的请求重新排到等待队列队尾，和全新请求一视同仁重新排序；这与 vLLM 抢占后把请求插到等待队列**最前面**的做法正好相反。
- **LPM 策略没有防饥饿机制**：唯一的兜底是等待队列超过 128 时自动退化为 FCFS，但这条判据的动机是省掉昂贵的前缀匹配计算（性能考虑），不是显式的公平性设计。

**速查：任务要求的问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| 三进程模型、IPC 机制、为什么拆出 detokenizer 进程 | `## 1`、`## 6` 进程模型对照 |
| overlap 重叠的到底是什么、怎么实现、打开后受什么限制 | `## 4`、`## 5` 决策三 |
| `get_next_batch_to_run()` 判据、`new_token_ratio` 启发式 | `## 4` 第 2 步、`## 5` 决策一/决策六 |
| LPM 怎么排序、和 RadixAttention 的关系 | `## 5` 决策二、`## 6` 调度策略对照 |
| retract 怎么退请求、和 vLLM preemption 的区别 | `## 5` 决策四、`## 6` retract vs preempt |
| SchedulePolicy 有没有防饥饿机制 | `## 7` 反直觉一 |

## 1. 它在系统里的位置

一次请求的完整路径，跨越三个操作系统进程：

```
HTTP 层 (FastAPI)
   │  异步协程，与 TokenizerManager 同一个进程
   ▼
TokenizerManager（主进程，tokenizer_manager.py）
   │  编码 prompt → TokenizedGenerateReqInput
   │  ZMQ PUSH → scheduler_input_ipc_name
   ▼
Scheduler（独立进程，每 TP/PP rank 一个，scheduler.py）
   │  排队 → 批调度 → 模型前向 → 采样
   │  ZMQ PUSH → detokenizer_ipc_name
   ▼
DetokenizerManager（独立进程，detokenizer_manager.py）
   │  token id → 增量文本
   │  ZMQ PUSH → tokenizer_ipc_name
   ▼
TokenizerManager（同一个主进程，绕回来）
   │  ZMQ PULL 收到，唤醒对应请求的 asyncio Future
   ▼
HTTP 响应 / SSE 流
```

这张图里容易被忽略的一点：`TokenizerManager` 出现了两次，但它们是**同一个进程**——请求编码和结果接收都发生在主进程的 asyncio 事件循环里，`TokenizerManager` 从来不是一个独立的 `mp.Process`（`## 6` 会详细对照 vLLM 的等价角色）。

三个组件的职责边界很清晰：

- **`Scheduler`** 是这条链路里唯一同时持有 GPU 显存管理权（KV cache 分配/回收）、请求排队权（`waiting_queue`）和批构造权（`ScheduleBatch`）的组件。本篇只讲它的内部机制。
- **`TokenizerManager`** 只做 CPU 侧的文本↔token 转换和请求路由，不参与任何调度决策——它甚至不知道一个请求现在是在 waiting、running 还是被 retract。
- **`DetokenizerManager`** 只做增量 detokenize（token id 流 → 增量文本片段），同样不碰调度，也不知道 KV cache 的存在。

三条 ZMQ 管道上流动的消息类型（均是 `msgspec.Struct`，定义在 `python/sglang/srt/managers/io_struct.py`）：

| 管道 | 典型消息类型 | 方向 |
|---|---|---|
| TokenizerManager → Scheduler | `TokenizedGenerateReqInput`（`python/sglang/srt/managers/io_struct.py:944`） | 一个已编码好的生成请求 |
| Scheduler → DetokenizerManager | `BatchTokenIDOutput`（`python/sglang/srt/managers/io_struct.py:1397`）、`BatchEmbeddingOutput`（`python/sglang/srt/managers/io_struct.py:1585`） | 一批请求这一步产出的 token id / embedding |
| DetokenizerManager → TokenizerManager | `BatchStrOutput`（`python/sglang/srt/managers/io_struct.py:1497`） | 增量解码出的文本片段 |
| Scheduler → TokenizerManager（跳过 detokenizer 的直连通道） | `AbortReq`（`python/sglang/srt/managers/io_struct.py:1998`） | 中止通知等不需要 detokenize 的旁路消息 |

最后一行值得注意：`SchedulerIpcChannels` 里 `send_to_tokenizer` 和 `send_to_detokenizer` 是**两个独立的 socket**（`python/sglang/srt/managers/scheduler_components/ipc_channels.py:20-21`），Scheduler 可以绕过 DetokenizerManager 直接给 TokenizerManager 发消息——`retract_decode()` 里对被中止请求发送 `AbortReq` 就是走这条直连通道，因为中止通知不含 token id、不需要 detokenize。

## 2. 代码地图（文件 → 职责，带行号）

按 `_lab/struct_map.py` 的子系统划分，SGLang 的 scheduler 子系统一共 81 个文件、35,054 行（数字见 `_lab/out/struct_map.json` 里 `engines.sglang.subsystems.scheduler.n_files` / `.lines`），本篇只挑其中和"事件循环、批调度、调度策略"直接相关的部分：

- `python/sglang/srt/managers/scheduler.py:384` —— `class Scheduler`，本体，5228 行的巨型类。
  含多个 mixin：`SchedulerPPMixin`、`SchedulerMultiplexMixin`、`SchedulerDisaggregationDecodeMixin`、`SchedulerDisaggregationPrefillMixin`，每个 mixin 对应一种事件循环变体。
- `python/sglang/srt/managers/scheduler.py:1696` —— `run_event_loop()`。
  进程启动后的入口，负责建 CUDA stream（含"避免 schedule_stream 与 forward_stream 撞车"的 redraw 逻辑），再调 `dispatch_event_loop()`。
- `python/sglang/srt/managers/scheduler.py:1748` —— `event_loop_normal()`。
  非重叠调度：收请求 → 定批 → 跑批 → 处理结果，同步顺序执行，每步都等 GPU 算完。
- `python/sglang/srt/managers/scheduler.py:1783` —— `event_loop_overlap()`。
  重叠调度（默认路径）：用 `result_queue` 把"处理上一批结果"推迟到"启动当前批前向"之后，是 `## 4` 逐步走读的对象。
- `python/sglang/srt/managers/scheduler.py:1857` —— `is_disable_overlap_for_batch()`。
  判断这一步是否要临时打断重叠（连续两个 prefill 批、投机解码语法同步等情形）。
- `python/sglang/srt/managers/scheduler.py:3080` —— `get_next_batch_to_run()`。
  每轮迭代的调度核心：合并上一批到 running_batch、决定 prefill 还是 decode、返回 `NextBatchPlan`。
- `python/sglang/srt/managers/scheduler.py:3225` / `python/sglang/srt/managers/scheduler.py:3252` —— `get_new_batch_prefill()` / `_get_new_batch_prefill_raw()`。
  用 `PrefillAdder` 从 `waiting_queue` 里挑请求组 prefill 批，含 chunked prefill 的截断逻辑。
- `python/sglang/srt/managers/scheduler.py:3566` —— `update_running_batch()`。
  decode 批的显存检查与 `retract_decode()` 的调用点。
- `python/sglang/srt/managers/scheduler.py:5045` —— `dispatch_event_loop()`。
  按 `enable_overlap` / `enable_pdmux` / `pp_size` / disaggregation 模式挑选事件循环变体的分发函数。
- `python/sglang/srt/managers/scheduler.py:5140` —— `run_scheduler_process()`。
  `mp.Process` 的目标函数，构造 `Scheduler` 并调用 `run_event_loop()`；异常时会给父进程发 `SIGQUIT`。
- `python/sglang/srt/managers/schedule_batch.py:2011` —— `class ScheduleBatch`。
  一批请求在 GPU 上的完整状态（张量字段 + `Req` 列表）。
- `python/sglang/srt/managers/schedule_batch.py:2825` —— `retract_decode()`。
  显存不足时的回退逻辑，返回被退请求列表、新的 `new_token_ratio`、需要中止的请求列表。
- `python/sglang/srt/managers/schedule_batch.py:3429` —— `class NextBatchPlan(msgspec.Struct)`。
  `get_next_batch_to_run()` 的返回类型，只有 `batch_to_run` 和 `running_batch` 两个字段。
- `python/sglang/srt/managers/schedule_policy.py:216` —— `class SchedulePolicy`。
  LPM / DFS-weight / FCFS / LOF / RANDOM / ROUTING_KEY 六种等待队列排序策略的分发点。
- `python/sglang/srt/managers/schedule_policy.py:381` —— `_sort_by_longest_prefix()`。
  LPM 的实现：按 `-num_matched_prefix_tokens` 排序，越匹配越靠前。
- `python/sglang/srt/managers/schedule_policy.py:511` —— `class PrefillAdder`。
  把 `waiting_queue` 里的请求逐个装进本轮 prefill 批的预算记账器，含 chunked prefill 截断（`add_chunked_req` 在 `python/sglang/srt/managers/schedule_policy.py:1004`）。
- `python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:14` —— `class NewTokenRatioTracker`。
  "预留多少显存给运行中请求的未来 decode"这个启发式的状态机。
- `python/sglang/srt/managers/scheduler_components/ipc_channels.py:17` —— `class SchedulerIpcChannels`。
  Scheduler 侧的 ZMQ socket 集合：收 tokenizer、发 tokenizer（直连，用于 abort 等旁路消息）、发 detokenizer。
- `python/sglang/srt/managers/tokenizer_manager.py:547` —— `TokenizerManager.init_ipc_channels()`。
  主进程侧的 ZMQ socket：收 detokenizer、发 scheduler。
- `python/sglang/srt/managers/detokenizer_manager.py:92` —— `class DetokenizerManager`。
  `python/sglang/srt/managers/detokenizer_manager.py:167` 是它的 `event_loop()`，单线程 `recv → dispatch → send`，不碰任何调度或 GPU 状态。

## 3. 核心数据结构

#### `Req`

单个请求的状态机（定义在 `schedule_batch.py`，字段众多，未在此逐一展开）。调度器最常读写的几个字段：

| 字段 | 作用 |
|---|---|
| `origin_input_ids` | 原始 prompt 的 token id 列表，不含已生成的部分 |
| `output_ids` | 已经生成出来的 token id 列表，随 decode 步数增长 |
| `prefix_indices` | 在 RadixAttention 树里已匹配的前缀，对应的**设备端** KV 索引 |
| `fill_ids` | 本轮前向需要真正计算的那部分 token（`origin_input_ids + output_ids` 减去已缓存的前缀） |
| `retraction_count` / `retracted_stain` | 被 retract 过几次、是否处于"回退后重算"状态，供调度与统计口径使用 |
| `num_matched_prefix_tokens` | LPM 排序读的分数，`match_prefix_for_req()` 写入 |
| `sampling_params` | `max_new_tokens` 等采样参数，`PrefillAdder`/`NewTokenRatioTracker` 用它估算未来 KV 占用 |

调度器不直接操作张量——它先在 `Req` 级别做决策（谁能进这一批、进多少 token），再由 `ScheduleBatch` 把一组决策好的 `Req` "编译"成一批物理张量。

#### `ScheduleBatch`（`python/sglang/srt/managers/schedule_batch.py:2011`）

一批请求在 GPU 上的完整快照：

- 张量字段：`input_ids`、`seq_lens`、`req_pool_indices`、`out_cache_loc`、`forward_mode` 等。
- `reqs: List[Req]`——这一批包含哪些请求。
- 既是"这一步要跑什么"的物化形式，也是 overlap 调度里被 `.copy()` 快照、塞进 `result_queue` 的对象。

它的字段按仓库约定（`.claude/rules/schedule-batch-out-of-place-mutation.md`）只能整体重绑定，不能原地 mutate——这是为了让 `.copy()` 出去的快照在 overlap 队列里保持冻结，不被下一轮迭代意外改写。

#### `ForwardMode`（`python/sglang/srt/model_executor/forward_batch_info.py:100`）

一个 `IntEnum`，告诉模型执行层这批张量该走哪条前向路径：

| 取值 | 含义 |
|---|---|
| `EXTEND` | 标准 prefill（含 chunked prefill 的每一块） |
| `DECODE` | 单步 decode，每个请求生成 1 个新 token |
| `MIXED` | chunked prefill 与 decode 混跑在同一批（`enable_mixed_chunk` 打开时） |
| `IDLE` | 没有请求要跑（数据并行下某些 rank 空闲时用到） |
| `TARGET_VERIFY` / `DRAFT_EXTEND_V2` | 投机解码专用 |
| `PREBUILT` | disaggregation decode worker，KV 已就绪等待开始解码 |
| `SPLIT_PREFILL` | PD-Multiplexing 专用的切分 prefill |

#### `NextBatchPlan`（`python/sglang/srt/managers/schedule_batch.py:3429`）

`get_next_batch_to_run()` 的返回值，只有两个字段：

- `batch_to_run: Optional[ScheduleBatch]`——这一步真正要跑的批，可能是 prefill 批、decode 批，也可能是 `None`（空闲）。
- `running_batch: ScheduleBatch`——合并/过滤后的新 running_batch，供下一轮迭代使用。

这个小 struct 把"调度决策"和"调度器状态更新"拆成了两个显式返回值，避免调度函数隐式地直接改写 `self.running_batch`。

#### `PrefillAdder`（`python/sglang/srt/managers/schedule_policy.py:511`）

不是持久状态，而是**单轮 prefill 批构造**的一次性记账器：

- `rem_total_tokens`：整个 KV 池还能装多少 token，已经扣掉了运行中请求按 `new_token_ratio` 估算的未来 decode 占用。
- `rem_chunk_tokens`：chunked prefill 单块的 token 预算。
- `can_run_list`：本轮真正能跑的请求列表，逐个 `add_one_req()` 累加出来。

调度器每轮都会新建一个 `PrefillAdder`，用完即弃——它不跨迭代持久化。

#### `NewTokenRatioTracker`（`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:14`）

贯穿多轮迭代的持久状态，只有一个可变字段 `current`。它是"给运行中请求预留多少未来 decode 空间"这个估计值，本质是一个自适应的保守系数，详见 `## 5` 决策六。

## 4. 主流程走读：event_loop_overlap 的一次迭代

以 `event_loop_overlap()`（`python/sglang/srt/managers/scheduler.py:1783`）为例逐步走读。重叠调度是默认路径——`disable_overlap_schedule` 默认为 `False`（`python/sglang/srt/server_args.py:993`）。

**第 1 步：收请求。**
`recv_reqs = self.request_receiver.recv_requests()`（`python/sglang/srt/managers/scheduler.py:1799`）从 ZMQ PULL socket 里非阻塞取出 `TokenizerManager` 转发来的新请求。
`process_input_requests()` 把它们塞进 `self.waiting_queue`（最终落到 `_add_request_to_queue()`，`python/sglang/srt/managers/scheduler.py:2783`）。

**第 2 步：定批。**
`plan = self.get_next_batch_to_run(running_batch=self.running_batch, last_batch=self.last_batch)`（`python/sglang/srt/managers/scheduler.py:1805`）。这一步内部（从 `python/sglang/srt/managers/scheduler.py:3080` 起）依次做：

1. 把上一轮的 `last_batch`（如果是 prefill/extend 批）过滤掉已结束的请求后合并进 `running_batch`（`python/sglang/srt/managers/scheduler.py:3132-3157`）——**这是 prefill 批"晋升"为 decode 批的地方**。一个 prefill 批跑完一步之后，它的请求就混进了 `running_batch`，下一轮作为候选参与 decode。
2. 调 `get_new_batch_prefill(running_batch)`（`python/sglang/srt/managers/scheduler.py:3174`）尝试从 `waiting_queue` 里凑一个新的 prefill 批。
3. 用一个非常直白的条件分支决定这一步真正跑什么：

节选自 `python/sglang/srt/managers/scheduler.py:3191`-`3200`（原文保留注释）：

```python
if new_batch is not None:
    # Run prefill first if possible
    ret = new_batch
else:
    # Run decode (skip for prefill-only batches)
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        running_batch = self.update_running_batch(running_batch)
        ret = running_batch if not running_batch.is_empty() else None
    else:
        ret = None
```

只要 `get_new_batch_prefill` 凑出了新批（哪怕只有一个请求），就直接把它作为这一步要跑的批，**完全不去看 `running_batch` 里还有多少 decode 请求在排队**。只有 `new_batch is None` 时才会走 `else` 分支跑 `update_running_batch(running_batch)`——即 decode。这就是"prefill 严格优先"的全部实现（`## 5` 决策一展开代价）。

**第 3 步：判断是否需要断开 overlap。**
`is_disable_overlap_for_batch()`（`python/sglang/srt/managers/scheduler.py:1857`）检查两种会明显拉低体验的情形：连续两个 prefill 批（会拉高第一个批的 TTFT）、投机解码语法约束需要同步 FSM。命中其一，就先同步处理掉 `last_batch` 的结果（`pop_and_process()`），打断这一步的流水线重叠。

**第 4 步：发射当前批的前向计算。**
如果 `batch` 非空，`self.run_batch(batch)`（`python/sglang/srt/managers/scheduler.py:1830`）把前向计算发到 `forward_stream` 上。这是一条独立于 `schedule_stream`（当前 Python 主线程的调度逻辑跑在这条流上）的 CUDA stream，通过显式插入流依赖后异步发射，不阻塞 CPU：

```python
# 节选自 python/sglang/srt/managers/scheduler.py:3750-3751 附近
with self.forward_stream_ctx:
    self.forward_stream.wait_stream(self.schedule_stream)
    ...
    batch_result = self.model_worker.forward_batch_generation(batch, **fwd_kwargs)
```

返回的 `batch_result`（连同 `batch.copy()` 快照）被 `append` 进 `self.result_queue`（`python/sglang/srt/managers/scheduler.py:1833`），**而不是立刻处理**。

**第 5 步：处理上一轮的结果。**
`if self.last_batch: pop_and_process()`（`python/sglang/srt/managers/scheduler.py:1839-1841`）——注意处理的是 `result_queue` 里最早的那一条，也就是**上一轮**（第 N-1 轮）批的结果，不是刚发射的这一批。`pop_and_process()` 内部的 `result.copy_done.synchronize()`（例如 `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:250`）才会真正阻塞等待 GPU 完成对应批次的 D2H 拷贝。

因为这个同步点等待的是上一轮而非当前这一轮的 GPU 工作，当前这一轮的前向（第 4 步刚发射的）大概率已经在 GPU 上并行跑着——这就是"重叠"的来源：**CPU 侧为第 N+1 轮做调度准备（下一次循环的第 1、2 步）时，GPU 侧还在算第 N 轮的前向**。

**第 6 步：采样。**
`launch_batch_sample_if_needed(batch_result, batch)`（`python/sglang/srt/managers/scheduler.py:1849`）在结果处理之后跑，因为采样可能依赖语法约束（grammar FSM）等只有在上一批处理完才确定的状态。

**第 7 步：滚动状态。**
`self.last_batch = batch`（`python/sglang/srt/managers/scheduler.py:1852`），下一轮迭代从第 1 步重新开始。

### 4.1 一个具体例子（示意，非实测数据）

用一个虚构的三请求场景把 `get_next_batch_to_run()` 的判据串起来——这里的数字纯粹用来说明控制流，不是任何 benchmark 结果：

- **迭代 N**：`waiting_queue = [A]`，`running_batch` 为空。`get_new_batch_prefill()` 凑出只含 A 的 prefill 批，`new_batch is not None` 成立，`ret = new_batch`（`python/sglang/srt/managers/scheduler.py:3193`）。这一步跑 prefill，`forward_mode = EXTEND`。
- **迭代 N+1**：请求 B 抵达，`waiting_queue = [B]`；A 已经在上一轮迭代的第 2 步里被合并进 `running_batch`（因为 `last_batch.forward_mode.is_extend()` 为真，见 `python/sglang/srt/managers/scheduler.py:3132-3157`）。这一轮 `get_new_batch_prefill()` 又凑出含 B 的新 prefill 批，于是**继续跑 prefill**——A 虽然已经进了 `running_batch`，但因为 B 能组出新 prefill 批，A 的 decode 被又推迟了一轮。
- **迭代 N+2**：`waiting_queue` 为空，`get_new_batch_prefill()` 返回 `None`。这时 `ret` 分支走 `else`，`update_running_batch(running_batch)` 把 A、B 一起做一步 decode（`python/sglang/srt/managers/scheduler.py:3197`），`forward_mode = DECODE`。
- **迭代 N+3**：假设此时显存紧张，`check_decode_mem()` 返回 `False`（`python/sglang/srt/managers/schedule_batch.py:2818-2823`）——`update_running_batch()` 触发 `retract_decode()`，按"已生成 token 数最少者优先退"选出受害请求（比如 B 刚开始生成没几个 token，被选中），B 的 KV 被释放，重新 `append` 回 `waiting_queue` 队尾。
- **迭代 N+4**：`waiting_queue = [B]`（重新排队，`retracted_stain=True`），如果同时有全新请求 C 抵达且 LPM 分数比 B 高，C 会排在 B 前面——这正是 `## 5` 决策四提到的"回退请求没有插队特权"在时间线上的具体样子。

这个例子里最容易被忽略的一点：A 从"prefill 批"变成"decode 批的一员"不是一个显式的状态迁移动作，而是**合并到 `running_batch` 之后自然发生的**——`running_batch` 里的请求既可能是纯 decode 请求，也可能是刚合并进来、还没做过任何 decode 步的前 prefill 请求，`Req` 本身没有一个"我是不是刚从 prefill 转过来"的显式布尔字段，全靠 `ScheduleBatch.forward_mode` 这个批级别的字段区分当前这一步在做什么。

### 4.2 `event_loop_normal` 与 `event_loop_overlap` 的结构性差异

把两个事件循环并排读会发现它们的第 1、2 步（收请求、定批）**完全共用同一套调用**——`get_next_batch_to_run()` 不知道、也不关心自己是被哪个事件循环调用的。差异全部集中在"跑批"和"处理结果"这两步怎么编排：

| | `event_loop_normal`（`python/sglang/srt/managers/scheduler.py:1748`） | `event_loop_overlap`（`python/sglang/srt/managers/scheduler.py:1783`） |
|---|---|---|
| 结果处理时机 | 跑完 `run_batch(batch)` **立刻** `process_batch_result(batch, result)`（`python/sglang/srt/managers/scheduler.py:1770-1771`） | 结果先 `append` 进 `result_queue`（`python/sglang/srt/managers/scheduler.py:1833`），下一轮迭代才 `popleft()` 处理 |
| 是否需要额外的队列 | 不需要，`batch`/`result` 用完即弃 | 需要 `self.result_queue: Deque[...]`（`python/sglang/srt/managers/scheduler.py:1785-1787`），跨迭代持有引用 |
| CPU 是否等待 GPU | 每轮都等：`run_batch()` 返回前 GPU 前向已经跑完（或至少发起了同步拷贝） | 不等：`run_batch()` 只是发射，真正的同步点在**下一轮**处理上一批结果时才触发 |
| 空闲时的行为 | `batch` 为空则直接 `self.on_idle()`（`python/sglang/srt/managers/scheduler.py:1774-1775`） | 需要分别处理"当前批为空"和"上一批为空"两种空闲情形（`python/sglang/srt/managers/scheduler.py:1834-1844`），因为两者不是同一轮 |

也就是说 `event_loop_normal` 不是"少做点什么"的简化版 overlap，而是把 `batch` 和 `result` 的生命周期从"跨两轮"压缩成"当轮结束"——这正是它更简单、但也更慢的原因：少了一层间接，也就少了一层可以让 CPU 和 GPU 各自往前走的空间。

## 5. 设计决策与代价

### 决策一：prefill 严格优先于 decode

*为什么这么设计*
- prefill 决定 TTFT（首 token 延迟），而 TTFT 是用户能直接感知的指标。
- SGLang 的默认策略是"来了就尽快让它开始生成"，代价转嫁给已经在生成中的请求（略微推迟它们的下一个 token）。

*不这样会怎样*
- 如果改成 decode 优先，新请求要等 running_batch 排空或显存腾出空间才能插入。
- TTFT 会随并发请求数线性变差，在高并发短生成场景（比如大量短答案请求）体验明显劣化。

*什么时候可以不这样*
- `prefill_decode_interval`（`server_args` 定义，调度器侧读取点在 `python/sglang/srt/managers/scheduler.py:1203` 的 `_should_defer_prefill()`）可以把"跑完一个 prefill 批之后必须先跑 N 轮 decode 才能再排下一个 prefill"这个节奏显式配出来。
- 本质是把"prefill 优先"降级为"prefill 有限打断 decode"，牺牲 TTFT 换 TPOT（每 token 间隔）的稳定性。
- 默认值是 `0`（关闭，即本决策开头的严格优先，详见 `## 7` 反直觉四）。

### 决策二：调度策略与前缀缓存树耦合（LPM）

*为什么这么设计*
- SGLang 的核心卖点是 RadixAttention——不同请求间共享的 KV 前缀存在一棵 radix 树里。
- 如果调度器按 FCFS 随机顺序把请求塞进同一批，两个共享长前缀的请求可能被分到相隔很远的两轮，中间夹杂的其它请求会把树里的公共前缀节点挤出去（LRU 逐出），下次这两个请求各自都要重新算一遍本可共享的前缀。
- LPM 排序（`python/sglang/srt/managers/schedule_policy.py:381`）让"和树里已有内容匹配度高的请求"优先出队，本质是在**最大化本轮批次内 / 相邻批次间的前缀复用率**，减少重复计算和树逐出压力。

*不这样会怎样*
- 退化成 FCFS 时（`python/sglang/srt/managers/schedule_policy.py:291` 的自动降级，或用户显式设置 `--schedule-policy fcfs`），前缀命中率完全靠运气。
- 如果客户端把共享前缀的请求打散发送（比如多用户各自的 system prompt 相同但穿插到达），批次内几乎没有前缀复用，RadixAttention 退化成普通的每请求独立 KV cache。

*什么时候可以不这样*
- 等待队列很短时（几乎不用排序，先来后到差别不大）。
- 请求之间本来就没有前缀重叠（比如每个请求都是完全独立的用户输入，没有共享 system prompt/few-shot 模板）。
- `tree_cache.disable` 为真时（未启用前缀缓存，`python/sglang/srt/managers/schedule_policy.py:311` 会自动把 LPM 降级为 FCFS）。

### 决策三：overlap 用多 CUDA stream + 事件，不是独立线程

*为什么这么设计*
- 早期实现思路（在别的推理引擎里常见）是开一个独立的 worker 线程跑前向、主线程只管调度，通过线程间队列传递"future" token id。
- SGLang 现在的做法（`python/sglang/srt/managers/scheduler.py:3741-3862`）是单线程发射到多个 CUDA stream 上，靠 `Event.record()`/`wait_stream()` 表达依赖：`schedule_stream` 负责 CPU 侧准备下一批的输入张量，`forward_stream` 跑模型前向，`copy_stream` 做结果的 D2H 拷贝，三者并发执行，由 CUDA 的流依赖图而不是 Python 线程调度器来保证正确性。
- 它还引入了一个 `future_map`（`python/sglang/srt/managers/scheduler.py:3746`、`python/sglang/srt/managers/scheduler.py:3767`）：当前批的输出 token id 在还没真正算出来时就先占好"未来位置"，下一轮批构造可以引用这个占位符而不必等 GPU 算完。

*不这样会怎样*
- 用独立线程 + 显式 future 队列，需要处理线程间的 GIL 争抢、异常传播、优雅退出等一整套并发原语。
- 用单线程 + 多 stream + 事件的方式把"重叠"完全表达成 CUDA 依赖图，Python 侧仍是单线程顺序执行，心智负担小得多——但要求所有跨 stream 的张量生命周期管理必须非常小心（`record_batch_in_overlap()`，`python/sglang/srt/managers/scheduler.py:3651` 专门给出了为什么要手动 snapshot 所有字段引用的注释）。

*什么时候可以不这样*
- `disable_overlap_schedule=True` 会退回 `event_loop_normal()`（同步顺序执行），代价是每轮迭代都要等 GPU 前向完全结束才能开始下一轮的 CPU 调度，吞吐显著下降。
- 但在几种场景下**必须**关掉重叠：
  - 流水线并行（`pp_size > 1` 时 `python/sglang/srt/server_args.py:9260-9263` 直接 assert 必须 `disable_overlap_schedule`）。
  - PD-Multiplexing（`python/sglang/srt/server_args.py:9319-9321` 同样 assert）。
  - Apple `mps` 设备（`python/sglang/srt/server_args.py:4444-4447` 自动强制关闭，因为 MPS 没有 SGLang 依赖的那套 CUDA stream/event 语义）。

### 决策四：retract 不给"回退请求"排队特权

*为什么这么设计*
- `_add_request_to_queue(req, is_retracted=True)` 在非 PD-disaggregation 模式下走的还是 `self.waiting_queue.append(req)`（`python/sglang/srt/managers/scheduler.py:2790`），跟全新请求一样排到队尾。
- 唯一的区别只是打了个 `retracted_stain=True` 的标记用于统计口径（`python/sglang/srt/managers/schedule_batch.py:1691`，`python/sglang/srt/managers/schedule_policy.py:909-911` 用它把这部分 token 归类到"重复计算"而不是"缓存命中"）。
- 这样实现简单：调度器不用维护"回退请求"和"新请求"两条不同优先级的队列，LPM/FCFS 等排序逻辑对它们一视同仁。

*不这样会怎样*
- 现状的风险是：如果系统持续高负载、新请求源源不断到达，一个被退回的请求理论上可能被反复排到队尾、迟迟等不到资源，而它已经消耗过一次计算（生成到一半被打断）。
- 这是纯粹的"公平性"代价，不是正确性问题——请求最终一定会被服务，只是延迟没有上界保证。

*什么时候可以不这样*
- `retraction_policy="priority"`（`python/sglang/srt/managers/schedule_batch.py:2904`）可以让退回顺序按业务优先级而非"已生成 token 数"排序，但这只改变**谁被退**，不改变退回后**排队位置**。
- 真正想给回退请求插队特权，需要走 PD-disaggregation 的 decode 模式——那条路径下 `release_req(..., offload_kv=True)` 会把 KV 备份到 host 内存（`python/sglang/srt/managers/schedule_batch.py:1928-1940`），配合 `disagg_decode_prealloc_queue.add(req, is_retracted=True)`（`python/sglang/srt/managers/scheduler.py:2799`）有专门的重入队列，能避免从零重新预填。

### 决策五：chunked prefill 按 token 预算而非请求数切块

*为什么这么设计*
- 一个超长 prompt（比如 32K token）如果整块塞进一次前向，会独占很长时间的 GPU，把同批或紧随其后的短请求（尤其是 decode 请求）的延迟都拖长。
- `PrefillAdder.add_one_req()` 在 `input_tokens > chunk_tokens_limit` 时会把请求截断到 `trunc_len`（页对齐，`python/sglang/srt/managers/schedule_policy.py:1386-1422`），本轮只算这一部分，剩下的记在 `self.chunked_req`，留到下一轮继续（`add_chunked_req`，`python/sglang/srt/managers/schedule_policy.py:1004`）。

*不这样会怎样*
- 不开 chunked prefill（`chunked_prefill_size=-1`）时，长 prompt 请求会让当前这一步的前向计算时间正比于 prompt 长度。
- 其它并发请求的尾延迟会出现明显的"长尾干扰"——这在多租户在线服务场景是比较致命的。

*什么时候可以不这样*
- 吞吐优先、不关心尾延迟的离线批处理场景（比如跑评测集），关掉 chunked prefill 反而能省掉多轮调度的固定开销、减少 KV 前缀反复 match 的成本。
- `--chunked-prefill-size -1` 显式关闭。

### 决策六：`new_token_ratio` 用保守系数代替精确预留

*为什么这么设计*
- 调度器在决定"还能不能再收一个 prefill 请求"时，需要知道 running_batch 里现有请求未来还要占多少 KV 空间。
- 精确值是"每个请求的 `max_new_tokens - 已生成数`"，但这是最坏情况估计（几乎没有请求会真的生成到 `max_new_tokens` 才停），全按最坏情况预留会让能同时跑的请求数远低于实际可行的水平。
- `NewTokenRatioTracker`（`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:14`）维护一个 `current` 系数（初始 `0.7 * schedule_conservativeness`，`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:22-25`），`PrefillAdder._get_running_request_total_token_offset()`（`python/sglang/srt/managers/schedule_policy.py:661-668`）用 `(max_new_tokens - 已生成数) * current` 而不是全额预留。

*不这样会怎样*
- 如果 `current` 恒为 `1.0`（全额保守预留），系统会显著低估可用并发度，牺牲吞吐换取"绝不发生显存不够而 retract"的确定性。
- 如果 `current` 恒定很低（激进预留），会频繁触发 `retract_decode()`，已经开始生成的请求被反复打断重排，尾延迟反而更差。

*什么时候可以不这样*
- `current` 本身是自适应的：正常跑（没有触发 retract）时逐步 `decay_step()` 往 `min` 靠拢（`python/sglang/srt/managers/scheduler.py:3639`，`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:34-35`，600 步内衰减完）。
- 一旦真的发生 retract 就用 `estimate_new_token_ratio_after_retract()`（`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:41-49`）按实际观测到的"已生成 token / max_new_tokens"重新校准，所以大部分时候用户不需要手动干预。
- 只有对延迟极度敏感、宁可牺牲吞吐也要把 retract 概率压到接近零的场景，才值得手动调低 `SGLANG_INIT_NEW_TOKEN_RATIO` 或抬高 `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR`。

## 6. 同位对照：vLLM 的 `vllm/v1/core/sched/scheduler.py`

一句话概括四个维度的差异：

| 维度 | SGLang | vLLM V1 |
|---|---|---|
| 进程模型 | tokenizer+API 一个进程、scheduler 一个进程、detokenizer 单独一个进程（三段式） | tokenizer+detokenizer+API 一个进程、EngineCore 一个进程（两段式） |
| 批类型判据 | 非此即彼：prefill 批或 decode 批，由"是否存在可调度的新 prefill"决定 | 无类型区分：统一按 `num_computed_tokens` 追赶 `num_tokens_with_spec`，由 token 预算决定 |
| 调度队列排序 | LPM（查询 RadixAttention 树）/ FCFS / DFS-weight 等六种 | 仅 FCFS / PRIORITY，不看前缀缓存内容 |
| 显存不足时的应对 | retract：退回等待队列**队尾**，默认不备份 KV | preempt：退回等待队列**队首**，同样丢弃 KV 重算 |

#### 进程模型对照

vLLM V1 的进程边界和 SGLang 不完全对应。API/`AsyncLLM` 进程内直接持有 `OutputProcessor`（`vllm:vllm/v1/engine/output_processor.py:443`）和 `IncrementalDetokenizer`（`vllm:vllm/v1/engine/detokenizer.py:31`），detokenize 是**同进程内的 CPU 计算**，不是独立 OS 进程；真正独立出去的是 `EngineCoreProc`（`vllm:vllm/v1/engine/core.py:1021`），对应 SGLang 的 `Scheduler` 进程。

SGLang 单独拆出 detokenizer 进程的动机（**本库推断**，依据：`DETOKENIZER_MAX_STATES` 等状态维护逻辑独立成一个纯 CPU 循环，`python/sglang/srt/managers/detokenizer_manager.py:167` 的 `event_loop()` 只做 `recv → dispatch → send`，不碰任何 GPU/调度状态）大概率是为了让"高吞吐场景下大量小字符串处理"这种纯 CPU、GIL 敏感的工作不占用 API 进程或 Scheduler 进程的 CPU 时间片——尤其是 Scheduler 进程的调度逻辑本身也是纯 Python/CPU，如果 detokenize 和调度挤在同一个进程里，两者会直接竞争 GIL。vLLM 选择不拆，可能是因为 `AsyncLLM` 进程本身以 asyncio 为主、CPU 密集的 detokenize 工作量相对可控（**本库推断**，未查证具体取舍讨论）。

#### prefill/decode 判据对照——两个引擎调度哲学的分水岭

SGLang 的 `get_next_batch_to_run()` 是"非此即彼"：一轮迭代要么跑纯 prefill 批，要么跑纯 decode 批（`python/sglang/srt/managers/scheduler.py:3191-3200`，`enable_mixed_chunk` 打开时决定预留多少 mixed decode token 名额，但主体逻辑依然是"prefill 优先，decode 兜底"）。

vLLM 的 `schedule()`（`vllm:vllm/v1/core/sched/scheduler.py:484`）**完全没有**"prefill 批"和"decode 批"的概念区分——源码注释直接写明（`vllm:vllm/v1/core/sched/scheduler.py:487`）:"There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec."

它的循环先遍历 `self.running`（包含正在 decode 的请求和正在做 chunked prefill 的请求，两者用同一个 `num_computed_tokens` 追赶 `num_tokens_with_spec` 的语义统一处理，`vllm:vllm/v1/core/sched/scheduler.py:531-704`），消耗 `token_budget`，**只有 running 队列没耗尽预算时**才继续从 `waiting` 队列拉新请求进批（`vllm:vllm/v1/core/sched/scheduler.py:749` 起）。

换句话说：vLLM 的一个批次天然是 prefill 和 decode 的混合体，由 token 预算（`max_num_batched_tokens`）而不是"批类型"决定谁能进批；SGLang 的一个批次默认是纯质的（除非显式开 `enable_mixed_chunk`），由"是否存在可调度的新 prefill 请求"这个布尔判据决定批类型。

#### 调度策略对照：LPM vs FCFS

vLLM 的 `SchedulingPolicy`（`vllm:vllm/v1/core/sched/request_queue.py:13`）只有两个值——`FCFS` 和 `PRIORITY`，没有任何策略会去看 KV 前缀缓存的匹配长度来决定调度顺序。这不是疏漏，而是架构上的必然：vLLM 的前缀缓存是一个基于哈希的 block pool + LRU 驱逐（不在本篇展开），调度器和前缀缓存管理器是两个解耦的子系统，调度器只关心 token 预算和到达顺序/优先级。

SGLang 反过来，`SchedulePolicy.calc_priority()`（`python/sglang/srt/managers/schedule_policy.py:237`）在 LPM 模式下直接调用 `tree_cache.match_prefix()`（经由 `match_prefix_for_req()`，`python/sglang/srt/managers/schedule_policy.py:138`）来给每个等待请求打分，调度顺序**由前缀缓存树的当前形状决定**。这正是 RadixAttention 作为 SGLang 核心设计的自然延伸——树既是缓存结构，也是调度输入。代价在 `## 5` 决策二里已经展开：LPM 天然没有时间公平性保证，而 vLLM 的 FCFS/PRIORITY 天然是有序、可预测等待时间的。

#### retract vs preempt：都是"丢显存"，但重排队策略相反

两者的相同点：都是在无法分配新 KV block 时，选一个受害者释放它的 KV，让出空间；都保留已生成的 `output_ids`/token 内容，只是要求重新计算对应 KV（不是把已生成的文本丢弃）。

不同点有两处：

1. **选谁**——SGLang 默认按"已生成 token 数最少者"优先退（`_get_decode_retraction_order`，`python/sglang/srt/managers/schedule_batch.py:2889-2927`，逻辑是"投入最少的最不可惜"）；vLLM 默认（FCFS 模式无 `PRIORITY` 策略时）弹出 `self.running` 列表的**最后一个**元素（`vllm:vllm/v1/core/sched/scheduler.py:684`，等价于"最近才被调度上的请求先被抢占"，更接近 LIFO）。
2. **退回队列的位置**——如 `## 5` 决策四所述，这是最直接影响用户感知延迟的差异：SGLang 的默认路径退回 `waiting_queue` 队尾（`python/sglang/srt/managers/scheduler.py:2790`），要和所有新到达请求重新按调度策略排队；vLLM 明确 `self.waiting.prepend_request(request)`（`vllm:vllm/v1/core/sched/scheduler.py:1387`），插到等待队列**最前面**，下一轮调度大概率立刻重新捡起。

这意味着同样是"被牺牲"，vLLM 的受害请求恢复得更快；SGLang 的受害请求恢复时间不确定，取决于 LPM 排序下它的前缀匹配分数、以及新请求的到达速率。

#### chunked prefill 记账方式对照

两个引擎都支持 chunked prefill，但记账的"账本"结构不一样。SGLang 用一个**专用的一次性对象** `PrefillAdder`（`python/sglang/srt/managers/schedule_policy.py:511`）——每轮迭代新建、跑完即弃，内部维护 `rem_chunk_tokens`（本轮 chunk 预算）与 `rem_total_tokens`（整个 KV 池的剩余空间估计）两套独立的计数器，一个长 prompt 超出 `chunk_tokens_limit` 就被截断，剩下的部分记在 `self.chunked_req` 上等下一轮继续（`python/sglang/srt/managers/schedule_policy.py:1386-1422`）。

vLLM 没有对应的"prefill 专用记账对象"：`token_budget` 是 `schedule()` 函数体内的一个局部标量（`vllm:vllm/v1/core/sched/scheduler.py:504`），在同一个循环里被 running 请求（含正在做 chunked prefill 的）和 waiting 请求共享着消耗；一个请求这一步能拿到多少 token，取决于 `min(num_new_tokens, token_budget, ...)`（`vllm:vllm/v1/core/sched/scheduler.py:573-575`）这个 running 循环体内联的算式，没有独立出一个"批构造器"类。换句话说，SGLang 把"prefill 批怎么攒出来"这件事封装成了一个可以单独阅读、单独测试的对象；vLLM 把它摊平在 `schedule()` 的主循环里，和 decode 请求的调度逻辑物理上写在同一段代码里——这与 `## 6` 开头对照表里"批类型判据"那一行是同一个设计选择的两种代码组织形式。

（对照使用的 vLLM 版本：`vllm` @ `7ca49fbe`，2026-08-22，仅作同位参考，不是本篇取证基准；关于 vLLM 调度器的完整解剖见 [[03-vLLM-调度器解剖]]。）

## 7. 踩坑与反直觉

**反直觉一：LPM 排序没有防饥饿机制，且 128 的阈值是性能兜底不是公平性设计。**
直觉上"调度策略"应该内建某种防止某类请求永远排不上号的保护，但读 `schedule_policy.py` 通篇找不到任何基于等待时长的加权或老化（aging）机制——`_sort_by_longest_prefix()`（`python/sglang/srt/managers/schedule_policy.py:381-391`）纯粹是按 `-num_matched_prefix_tokens` 排序，一个前缀完全不匹配（`num_matched_prefix_tokens=0`）的请求会被排到最后，且没有随等待轮数增加而被"提权"的逻辑。唯一的兜底是等待队列长度超过 128 时自动退化到 FCFS（`_determine_active_policy`，`python/sglang/srt/managers/schedule_policy.py:290-294`），但这条判据的动机写在注释里是"关掉昂贵的前缀匹配和排序计算"（性能），只是顺带带来了公平性的副作用，不能当成显式的防饥饿设计来依赖。

**反直觉二：overlap 调度不是"另开一个线程跑 GPU"，而是单线程发射到多个 CUDA stream。**
直觉上"CPU 与 GPU 重叠"容易联想到经典的生产者-消费者线程模型（一个线程管前向计算、一个线程管其它一切），但 SGLang 现在的实现（`python/sglang/srt/managers/scheduler.py:1783-1855`、`python/sglang/srt/managers/scheduler.py:3741-3862`）完全在单个 Python 线程里跑，靠给不同工作分配不同 CUDA stream（`schedule_stream`/`forward_stream`/`copy_stream`）加显式 `Event` 依赖来实现物理上的并发执行。仓库约定文件里对应的坑：`schedule-batch-out-of-place-mutation.md` 明确写了"`copy()` 快照和重叠调度器排队的引用依赖老对象保持冻结"——这条约束就是因为单线程发射到多 stream 的设计要求 CPU 侧不能过早原地改写还被 GPU 侧引用着的张量。

**反直觉三：默认情况下"退回"（retract）并不会把 KV 备份到 host，而是直接丢弃重算。**
容易望文生义地以为"回退"意味着有某种断点续算机制，但 `release_req()`（`python/sglang/srt/managers/schedule_batch.py:1913-1948`）里明确写着：只有 `disaggregation_mode == "decode"` 时才会走 `retraction_backup()` 把 KV 拷到 host（`python/sglang/srt/managers/schedule_batch.py:1932-1940`），普通单机模式下 `release_kv_cache(req, tree_cache, is_insert=False)`（`python/sglang/srt/managers/schedule_batch.py:1942`）直接释放显存且**不**插入 radix 树（`is_insert=False`），意味着这部分 KV 连"下次可能命中缓存"的机会都没有。已生成的 `output_ids` 文本内容本身没丢，但对应的 KV 需要从零重新计算（相当于把已生成的文本重新当成一段更长的 prompt 去做 prefill）。

**反直觉四：`prefill_decode_interval` 的默认值是 0（关闭），这意味着"prefill 优先"在默认配置下没有任何限速阀。**
很多读者会假设生产环境默认配置已经考虑了 TTFT/TPOT 的权衡，但 `server_args` 里 `prefill_decode_interval` 默认值为 `0`（表示禁用节流），配合 `_should_defer_prefill()`（`python/sglang/srt/managers/scheduler.py:1203-1208`）在 `remaining == 0` 时直接返回 `False`（不推迟），也就是说**默认配置下每一步都会先检查是否有新 prefill 可跑，有就跑，不受任何"最近跑过几次 decode"的约束**——高并发场景下如果 prefill 请求持续到达，decode 请求的单步间隔（直接影响 TPOT）理论上可以被无限期推迟，直到 prefill 队列暂时清空。

**反直觉五："三进程模型"里其实只有两个真正意义上独立跑起来的新进程。**
容易把"三进程"理解成三个对等的、由 `mp.Process` 启动的实体，但实际只有 `Scheduler`（`python/sglang/srt/managers/scheduler.py:5140` 的 `run_scheduler_process`）和 `DetokenizerManager`（`python/sglang/srt/managers/detokenizer_manager.py:516` 的 `run_detokenizer_process`）是这样启动的；`TokenizerManager` 是在主进程里直接 `TokenizerManagerClass(server_args, port_args)` 构造出来的普通对象（`python/sglang/srt/entrypoints/engine.py:162`），和承载 HTTP 服务的 FastAPI/uvicorn 事件循环共享同一个进程。称呼上说"三进程"是从"三个逻辑角色"的角度说的，不是"三个 `mp.Process`"。

**反直觉六："prefill 批和 decode 批互斥"这句话不总是成立。**
`## 4` 第 2 步说的"非此即彼"是默认行为，但 `self.is_mixed_chunk`（`python/sglang/srt/managers/scheduler.py:1185-1187`，由 `chunked_prefill_size` 已设置且 `enable_mixed_chunk` 打开共同决定）打开之后，`PrefillAdder` 构造时会把当前 `running_batch` 的请求数 `running_bs` 作为 `num_mixed_decode_tokens` 传进去（`python/sglang/srt/managers/scheduler.py:3333`）——意味着这一步的 prefill 批预算里，会预留出正在 decode 的请求继续往下走一步所需的 token 名额，两类请求被塞进**同一个物理批次**里一起前向。这是为什么 `ForwardMode` 里专门有一个 `MIXED` 取值（见 `## 3`）：它不是 `EXTEND` 和 `DECODE` 的中间态，而是"两者都在同一批里发生"的显式标记。默认 `enable_mixed_chunk=False`，所以本篇 `## 4`/`## 5` 决策一描述的"严格非此即彼"是绝大多数部署的实际行为，但不是源码里唯一的路径。

## 8. 可改进点

以下判断标注为「本库推断」，未在上游 issue/PR 中核实，仅基于代码读到的行为提出、供讨论：

1. **LPM 策略缺少等待时长因子。**
   目前的排序 key 只有 `num_matched_prefix_tokens`（`python/sglang/srt/managers/schedule_policy.py:386`），可以考虑改成 `(某种前缀匹配得分) - w * 等待时长` 的复合 key，在不牺牲太多前缀命中率的前提下给长时间等待的请求一个逐渐提升的优先级，避免理论上的无限期推迟（`## 7` 反直觉一提到的问题）。
   风险：需要调好权重 `w`，调不好可能反而破坏 LPM 本来想最大化的前缀复用率，需要用真实流量分布做 A/B 才能验证收益。

2. **retract 后的重排队没有区分"刚被打断"和"全新到达"。**
   让 `is_retracted=True` 的请求在重新计算 `SchedulePolicy` 排序分数时获得一个小的固定加成（而不是像现在完全一视同仁），可以在不引入专门优先队列的前提下，缓解"回退请求可能被反复退回"的尾部风险。
   vLLM 的 `prepend_request`（`vllm:vllm/v1/core/sched/scheduler.py:1387`）是更激进的版本，直接给最高优先级——SGLang 如果照搬这个做法，需要考虑它和 LPM 排序的交互（插到队首会绕过前缀匹配排序，可能反而降低批内前缀复用率）。

3. **单机（非 PD-disaggregation）模式下的 retract 直接丢弃 KV，损失了本可利用的复用机会。**
   `release_req()`（`python/sglang/srt/managers/schedule_batch.py:1913`）目前只在 decode-disaggregation 模式下走 host 备份路径；把这条路径的条件放宽到"host 内存池有空间"而不局限于 disaggregation 模式，理论上能降低单机高负载场景下 retract 的重算成本。
   代价：要权衡额外的 D2H/H2D 拷贝开销是否划算——这本身就是一个需要实测数据才能下结论的问题（本库不产出性能数字，留待读者验证）。

4. **`prefill_decode_interval` 默认关闭，缺少一个自适应版本。**
   现状是用户要么完全不管（默认 `0`，无限制 prefill 优先），要么手动配一个固定的"跑 N 轮 decode 才能插一次 prefill"。可以考虑让这个间隔根据当前 running_batch 里 decode 请求的实际尾延迟动态调整，而不是让用户猜一个静态值。
   代价：多了一个反馈回路，需要证明它比"用户手动调参"更稳定，不会引入震荡。

5. **`event_loop_normal` 和 `event_loop_overlap` 的公共前半段（收请求、定批）目前是两份几乎相同的代码。**
   对比 `python/sglang/srt/managers/scheduler.py:1748-1782` 和 `python/sglang/srt/managers/scheduler.py:1783-1812` 会发现"收请求 → `process_input_requests` → `get_next_batch_to_run`"这几行在两个函数里逐字重复。抽成一个共享的私有辅助方法（`_recv_and_schedule()` 之类）能减少后续改动时"改了一处忘了另一处"的风险。
   代价：几乎没有——这是一处纯粹的重复代码消除，唯一的成本是需要小心两个循环各自后续分支读取的局部变量命名保持一致。

## 9. 自测题与延伸阅读

**自测题**

1. `get_next_batch_to_run()` 在什么条件下会返回一个 decode 批而不是 prefill 批？请指出判据所在的具体行号。
2. LPM 排序策略在等待队列长度超过多少时会自动退化为 FCFS？这个退化机制解决的是公平性问题还是性能问题？
3. `event_loop_overlap()` 里，`result_queue.popleft()` 弹出的结果对应的是"当前刚发射的批次"还是"上一轮的批次"？为什么这样设计能实现 CPU/GPU 重叠？
4. SGLang 的 `retract_decode()` 默认按什么顺序选择要退回的请求？这个顺序和 vLLM 默认抢占策略选择的受害者顺序有什么不同？
5. `new_token_ratio` 这个系数如果恒定为 `1.0`（不衰减）会对系统的哪个指标产生负面影响？如果恒定为一个很低的值又会怎样？
6. 为什么 `pp_size > 1` 时 SGLang 强制要求关闭 overlap 调度？（提示：结合 `## 5` 决策三里 `future_map`/多 stream 依赖图的实现方式思考。）
7. chunked prefill 打开之后，一个超长 prompt 的请求在 `waiting_queue` 里等待时，它算作"waiting"状态还是已经进入了某种中间态？`self.chunked_req` 这个字段起什么作用？
8. `TokenizerManager`、`Scheduler`、`DetokenizerManager` 三者中，哪个不是通过 `mp.Process` 启动的独立操作系统进程？这对理解"三进程模型"这个说法意味着什么？
9. `enable_mixed_chunk` 打开之后，`ForwardMode.MIXED` 这个取值和 `EXTEND`/`DECODE` 是什么关系？它出现的前提条件是什么？
10. Scheduler 给 TokenizerManager 发消息时，为什么有的消息（比如生成结果）要经过 DetokenizerManager 中转，有的消息（比如 `AbortReq`）却可以直接发送？这个设计选择的判据是什么？

**延伸阅读**

- [[01-vLLM-全景与代码地图]] —— vLLM 的整体代码结构与本篇对照用的坐标系。
- [[03-vLLM-调度器解剖]] —— vLLM V1 `Scheduler.schedule()` 的完整走读，本篇 `## 6` 的对照细节均可在其中找到更深的展开。
- [[04-vLLM-KV缓存与前缀缓存]] —— vLLM 的哈希 block pool + LRU 前缀缓存实现，用于理解"为什么 vLLM 的调度策略不需要看前缀匹配"这一设计选择的另一半。
