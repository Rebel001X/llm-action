# TGI：Rust router + Python server 的双语言架构

> **本篇取证基准**：`tgi` @ `b4adbf2f`（2026-03-21）
> **一句话**：调度用 Rust 躲 GIL，前向用 Python 认模型

## 0. 结论先行

- TGI 不是"用 Rust 重写了一遍 vLLM"，是**两个语言分管两件事**：Rust 一侧（`router/` + `backends/v2`、`backends/v3`）管请求排队、token 预算记账、连续批处理决策，跑成一个 axum HTTP 服务进程；Python 一侧（`server/text_generation_server/`）管模型加载与真正的一次前向计算，每个 TP shard 一个独立 OS 进程。两边不共享内存，靠 gRPC 通信，协议契约是 `proto/v3/generate.proto` 里的 `service TextGenerationService`（`proto/v3/generate.proto:5`），核心只有两个 RPC：`Prefill`（`proto/v3/generate.proto:18`）和 `Decode`（`proto/v3/generate.proto:20`）。这不是随手的技术选型，是"调度是 CPU 密集、对延迟敏感的工作，和 Python 的 GIL 放在一起会互相拖累"这条判断的直接产物——`## 5` 展开这条判断的代价。
- 连续批处理"下一批该装哪些请求、给多少 token 预算"这件事，完整地实现在 Rust 里：`State::next_batch()`（`backends/v3/src/queue.rs:237`）按 `prefill_token_budget`/`token_budget` 两个预算逐个从队首 pop 请求，装不下就整批 `break` 并把当前请求塞回队首（`backends/v3/src/queue.rs:297`、`346`）；外层 `batching_task()`（`backends/v3/src/backend.rs:129`）是一个 `loop {}`（`backends/v3/src/backend.rs:141`），prefill 一批之后持续用 decode 阶段腾出来的预算尝试拼接新请求进来。这条链路（`Entry` / `Queue` / `State` / `batching_task`）从头到尾没有一行 Python——vLLM 和 SGLang 把功能对等的逻辑放在各自的 Python `Scheduler` 类里，对照见 `## 6`。
- TGI 后期把执行层做成了可插拔的 `Backend` trait（`router/src/infer/mod.rs:29`），已知实现至少 4 种：`BackendV3`（`backends/v3/src/backend.rs:77`，gRPC 到 Python shard，默认路径）、上一代协议 `backends/v2`（同构，未展开）、`LlamacppBackend`（`backends/llamacpp/src/backend.rs:642`，直接 FFI 调 llama.cpp C 库，**不经过 Python、不经过 gRPC**）、`TensorRtLlmBackendV2`（`backends/trtllm/src/looper.rs:314`，靠 `cxx` crate FFI 直调 TensorRT-LLM 的 C++ Executor API，同样不经过 Python）。另外 `backends/gaudi/`、`backends/neuron/` 没有自己的 Rust crate，是 `server/text_generation_server` 针对特定硬件的 Python fork，复用 v2/v3 协议接入同一个 `Backend` trait。这意味着 TGI 把自己从"一个引擎"重新定位成了"一个可插多种执行后端的路由/调度框架"——`## 3` 展开。
- HTTP 表面一共抽到 **32 条路由**（`_lab/out/api_surface.json`，`method=regex`——TGI 的路由是 axum `.route()` 链式调用而非 AST 友好的装饰器语法，抽取器改用正则逐行扫描 `router/src/server.rs`，**这个精度限制意味着同一路径可能被多次命中或漏抽，下文列表逐条人工核对过行号**），去重后 **20 条唯一路径**，其中命中 `/v1/*` 前缀的只有 **3 条**：`/v1/chat/completions`、`/v1/completions`、`/v1/models`。真正的原生入口是 `/generate`（`router/src/server.rs:273`）和 `/generate_stream`（`router/src/server.rs:476`），外加一个更老的历史包袱——挂在根路径的 `POST /`（`compat_generate`，`router/src/server.rs:126`），这是 OpenAI 协议成为事实标准之前 TGI 用来兼容旧版 HuggingFace Inference API 的响应形状转换层。`## 2`/`## 7` 展开。
- 测试/产品代码比 **0.108**（`_lab/out/repo_stats.json` 的 `test_to_src_ratio`）算的是 **Python** 路径里带 `test` 关键词的行数占比，对 Rust 完全不可见——直接 grep 源码能数出 **9 个** `#[cfg(test)] mod tests` 块、**71 个** `#[test]`/`#[tokio::test]` 函数，例如 `backends/v3/src/queue.rs:560` 起的 `mod tests` 一路到 836 行文件结尾，**这一个文件近三分之一是测试**，但这些行一行都没有进 0.108 这个比值的分子。反过来，`_lab/struct_map.py` 的子系统关键词表也漏了 TGI 的批处理核心文件——`backends/v3/src/queue.rs`、`backends/v3/src/backend.rs` 因为文件名不含 `sched`/`scheduler`/`queue` 这几个关键词，被统计成 `scheduler` 子系统 **0 行**；真正的连续批处理逻辑因此在 `_lab/out/struct_map.json` 里"查无此文件"。这不是 TGI 代码的问题，是本库统计口径本身的盲区，`## 2`/`## 7` 会用直接跑 `_classify()` 得到的结果复核。
- 这份快照最后一次提交是 **2026-03-21**（`b4adbf2f`），本库抓取日是 **2026-08-22**，中间隔了 **5 个月**——这是 `git log` 的客观事实。仓库根 `README.md` 第 2 行原话（**文档所述**）：「text-generation-inference is now in maintenance mode. Going forward, we will accept pull requests for minor bug fixes, documentation improvements and lightweight maintenance tasks.」（`README.md:2`），第 4 行进一步写明官方现在推荐的下游引擎是 vLLM、SGLang，以及 llama.cpp/MLX（`README.md:4`）。这是项目自己的公开声明，不是本库的判断；本篇不对此做任何"已过时/被淘汰"式的延伸解读，只在 `## 7` 摆出这两行原文和完整上下文。

---

## 1. 它在系统里的位置

一次请求进来，无论走原生接口还是 OpenAI 兼容接口，都要跨一次语言边界：

```
Client
  │
  ├─ POST /generate, /generate_stream ──┐
  ├─ POST /v1/chat/completions ─────────┼─→ Rust axum handler（router/src/server.rs）
  ├─ POST /v1/completions ──────────────┘         │
  │                                                ▼
  │                                     Infer::generate_stream()
  │                                     （router/src/infer/mod.rs:100）
  │                                                │
  │                                                ▼
  │                                    Backend::schedule()（trait，infer/mod.rs:29）
  │                                                │
  │                     ┌──────────────────────────┼───────────────────────────┐
  │                     ▼                           ▼                          ▼
  │              BackendV3::schedule         LlamacppBackend::schedule   TensorRtLlmBackendV2::schedule
  │           (backends/v3/src/backend.rs:79) (backends/llamacpp/.../backend.rs:644) (backends/trtllm/.../looper.rs:315)
  │                     │                           │                          │
  │                     ▼                     进程内 FFI 直调              cxx FFI 直调
  │              Queue::append()              llama.cpp C 库             TensorRT-LLM C++ Executor
  │           (backends/v3/src/queue.rs:76)   （不经过 Python/gRPC）      （不经过 Python/gRPC）
  │                     │
  │                     ▼
  │              batching_task() 循环（backend.rs:129/141）
  │              State::next_batch() 预算记账（queue.rs:237）
  │                     │  gRPC：Prefill / Decode
  │                     ▼
  │              ShardedClient → N 个 Python 子进程（每 TP rank 一个，Unix socket）
  │                     │
  │                     ▼
  │              TextGenerationService.Prefill / Decode
  │              （server/text_generation_server/server.py:151, 193）
  │                     │
  │                     ▼
  │              self.model.generate_token(batch) ← 真正的 GPU 前向
```

- **进程边界**：路由是一个 OS 进程（`text-generation-router` 二进制，`backends/v3/Cargo.toml:12-13`），每个 TP shard 是另一个独立的 OS 进程（`text-generation-server`）。两者不共享地址空间。`launcher/src/main.rs` 负责统一编排：`shard_manager()`（`launcher/src/main.rs:923`）拼出 `Command::new("text-generation-server")`（`launcher/src/main.rs:1150`）并把这个 rank 专属的 Unix domain socket 路径 `{uds_path}-{rank}`（`launcher/src/main.rs:962`）传给它；`spawn_shards()`（`launcher/src/main.rs:1579`）为每个 rank 都起一个这样的子进程；所有 shard 健康检查通过之后，`main()`（`launcher/src/main.rs:2038`）才拉起路由进程本身（`Command::new("text-generation-router")`，`launcher/src/main.rs:1975`）。
- **语言边界**：Rust 侧只看得到 `proto/v3/generate.proto` 定义的强类型消息（`Batch`/`CachedBatch`/`Generation` 等），看不到、也不需要看到 PyTorch tensor 长什么样；Python 侧只看得到反序列化后的 batch 结构，不需要知道请求是怎么被排进这个 batch 的。这条边界同时也是**故障边界**——一个 Python shard 崩溃不会直接杀死 Rust 路由进程，路由会在下一次 gRPC 调用或健康检查里发现连接断开。
- 这篇只看"请求怎么被排队、攒批、跨语言送到执行层"，不展开模型前向本身怎么做量化/注意力选择（那是 `server/text_generation_server/layers/` 的范围，本库暂未单独成篇），也不展开 TensorRT-LLM/llama.cpp 后端内部怎么管理自己的 KV cache（那是各自上游项目的范围，TGI 这边只提供一层 `Backend` trait 适配）。

---

## 2. 代码地图（文件 → 职责，带行号）

| 文件 | 职责 | 关键行 |
|---|---|---|
| `router/src/server.rs` | axum HTTP 入口：路由注册 + 请求处理函数 | 基础路由表 `2203-2211`；info 路由表 `2237-2244`；路由合并 `2250`；kserve v2 路由 `2273-2284`；`compat_generate`（历史包袱）`126`；`get_model_info` `161`；`openai_get_model_info` `176`；`get_chat_tokenize` `199`；`health` `230`；`generate` `273`；`generate_stream` `476`；`generate_stream_internal` `508`；`completions` `715`；`chat_completions` `1168`；`run()` 泛型服务入口 `1502` |
| `router/src/infer/mod.rs` | 跨 backend 统一的推理门面，定义 `Backend` trait | `Backend` trait `29`；`Infer` 结构体 `49`；`generate_stream` `100`，内部 `self.backend.schedule()` 调用 `134`；`generate` `245` |
| `router/src/lib.rs` | 顶层协议结构体、`Tokenizer` 枚举、PyO3 桥接（tokenize 场景专用，非批处理路径） | `Tokenizer::Python` 变体 `30`；`PyTokenizer` `38`；`CompletionRequest` `492` |
| `backends/v3/src/backend.rs` | v3 后端：批处理主循环、prefill/decode 派发、指标上报 | `BackendV3` `17`；`::new` `27`；`impl Backend for BackendV3` `77`；`schedule` `79`；`batching_task` `129`；主循环 `loop {}` `141`；`prefill()` `297`；`decode()` `342`；`filter_batch()` `389`；`filter_send_generations()` `423`；`send_errors()` `540` |
| `backends/v3/src/queue.rs` | 请求队列 + token 预算连续批处理决策 | `Entry` `20`；`Queue` `41`；`::new` `46`；`append` `76`；`next_batch`（Queue 外壳） `86`；`queue_task` `118`；`State` `166`；`State::append` `226`；`State::next_batch`（真正的预算记账） `237`；预算超限判断 `297`、`346`；`mod tests` `560` |
| `backends/v3/src/block_allocator.rs` | Paged block 分配器（非前缀感知） | `BlockAllocator` `28`；`Allocator` trait `131` |
| `backends/v3/src/radix.rs` | 前缀感知的 radix-tree block 分配器 | `RadixAllocator` `20`；`impl Allocator for RadixAllocator` `81` |
| `proto/v3/generate.proto` | Rust↔Python 的 gRPC 契约 | `service TextGenerationService` `5`；`rpc Prefill` `18`；`rpc Decode` `20` |
| `server/text_generation_server/server.py` | Python 侧 gRPC servicer，持有真实模型与 KV batch 缓存 | `class TextGenerationService` `56`；`Warmup` `100`；`Prefill` `151`（内部 `generate_token`/`cache.set` `181-182`）；`Decode` `193`（内部同上 `216-217`）；`serve_inner` `242` |
| `launcher/src/main.rs` | 统一进程编排：拉起 N 个 Python shard + 1 个 Rust router | `shard_uds_path` 默认值 `782-783`；`shard_manager` `923`；UDS 路径拼接 `962`；shard 子进程 spawn `1150`；`spawn_shards` `1579`；router 子进程 spawn `1975`；`main` `2038`；`spawn_shards` 调用点 `2294` |
| `backends/llamacpp/src/backend.rs` | llama.cpp 后端：跳过 Python，直接 FFI | `LlamacppBackend` `164`；`impl Backend for LlamacppBackend` `642`；`schedule` `644` |
| `backends/trtllm/src/looper.rs` | TensorRT-LLM 后端：跳过 Python，`cxx` FFI 到 C++ Executor | `TensorRtLlmBackendV2` `259`；`impl Backend` `314`；`schedule` `315` |
| `backends/v3/Cargo.toml` | v3 crate 元信息 | crate 名 `2`；`[[bin]] name = "text-generation-router"` `12-13` |
| `README.md` | 项目状态声明（文档所述） | 维护模式声明 `2`；下游推荐 `4`；"A Rust, Python and gRPC server" 定语 `21` |
| `Cargo.toml`（仓库根） | Cargo workspace 成员表 | `members` 列表 `2-10`；版本号 `3.3.6-dev0` `24` |

共 20+ 条独立 `文件:行` 引用，超过硬指标要求的 6 条。

**统计口径的自我复核**（对应 `## 0` 最后两条结论）：直接跑 `_lab/struct_map.py` 的 `_classify()` 函数验证关键词命中情况：

```
backends/v3/src/queue.rs      -> []          # 想找 scheduler，关键词表没有 "queue"
backends/v3/src/backend.rs    -> []          # 同上
backends/v3/src/radix.rs      -> []          # 想找 kv_cache，关键词表有 "radix_cache" 但文件叫 "radix.rs"，不命中
backends/v3/src/block_allocator.rs -> []     # 关键词表有 "block_manager"/"block_pool"，没有 "block_allocator"
router/src/server.rs          -> ['entrypoint']
server/text_generation_server/models/flash_causal_lm.py -> ['entrypoint']
```

`entrypoint` 桶的关键词表包含字面量 `"server"`（`_lab/struct_map.py:35`），而 TGI 的 Python 产品码几乎全部住在路径里带 `server` 这个词的目录下——`server/text_generation_server/`、`backends/gaudi/server/text_generation_server/`、`backends/neuron/server/text_generation_server/`——这直接导致 `_lab/out/struct_map.json` 里 `entrypoint` 子系统报出 **98,821 行**，逼近全库 Python 产品码总量 **98,255 行**（`_lab/out/repo_stats.json`）；而真正做连续批处理决策的 `queue.rs`/`backend.rs`、真正做前缀缓存的 `radix.rs`/`block_allocator.rs`，因为是 Rust 文件、文件名又不含匹配关键词，在这份子系统统计里全部显示为 0。**这不是 TGI 代码规模小，是这把统计尺子对这几个文件名瞎眼。**`## 7` 继续展开。

---

## 3. 核心数据结构

- **`Backend` trait**（`router/src/infer/mod.rs:29`）——整个双语言/多后端架构的抽象边界，只有 4 个方法：`schedule()`（接一个校验过的请求，返回一个 token 流）、`health()`、`start_health()`、`name()`。`Infer` 只认识这个 trait，不知道背后具体是哪种执行方式。
- **`Infer`**（`router/src/infer/mod.rs:49`）——HTTP 层与调度层之间的门面，持有 `validation: Validation`、`backend: Arc<dyn Backend + Send + Sync>`、`chat_template: Option<ChatTemplate>`、并发限流用的 `limit_concurrent_requests: Arc<Semaphore>`。它本身不持有任何"请求正在排队/正在跑第几个 token"这类状态——那些状态都下沉在具体的 `Backend` 实现里。
- **`Entry`**（`backends/v3/src/queue.rs:20`）——一条排队中的请求：`request: ValidGenerateRequest`（校验过的载荷）、`response_tx: mpsc::UnboundedSender<...>`（这条请求专属的响应回传通道，`Infer::generate_stream()` 消费的就是这个通道另一端的 `UnboundedReceiverStream`）、`queue_time`/`batch_time` 两个 `Instant`（排队耗时可观测性）、`block_allocation: Option<BlockAllocation>`（KV block 预分配结果）。
- **`Queue` / `State`**（`backends/v3/src/queue.rs:41`、`166`）——典型的 actor 模式：`Queue` 只是一个 `mpsc::UnboundedSender<QueueCommand>` 的薄壳，真正的队列状态 `State`（内部是 `VecDeque<(u64, Entry)>` 加一个可选的 `BlockAllocator`）活在后台的 `queue_task()`（`queue.rs:118`）协程里，靠 channel 串行化所有读写——好处是不需要在 `await` 点上还拿着一把锁。
- **`Batch` / `CachedBatch` / `Generation`**（`proto/v3/generate.proto`）——Rust↔Python 的线上格式。`Batch` 描述"要 prefill 哪些请求"；`CachedBatch` 是 Python 端返回的**不透明句柄**（只有 id、size、max_tokens、current_tokens 这些标量字段），Rust 侧靠这个句柄在后续 `Decode` 调用里继续引用同一批 KV 状态，但**真实的 KV tensor 从未跨过这条 gRPC 边界**——它们只活在 Python 进程的显存里。
- **`RadixAllocator` / `BlockAllocator`**（`backends/v3/src/radix.rs:20`、`backends/v3/src/block_allocator.rs:28`）——TGI 自己在 Rust 里从零实现的前缀缓存感知 block 分配器，和 SGLang 的 `RadixCache`/vLLM 的 `BlockPool` 解决的是同一个问题（哪些 KV block 可以被新请求复用），但树的遍历本身发生在 Rust 进程里，不需要过一次 gRPC 才能知道能不能复用。

---

## 4. 主流程走读

以 `/generate` 一次非流式请求为例，从 HTTP 层一路走到 Python 端的一次真实前向：

**跳 1（路由分发）**：`generate()`（`router/src/server.rs:273`）拿到 axum 已解析好的 `GenerateRequest`，转调 `infer.generate(...)`（`router/src/infer/mod.rs:245`）。

**跳 2（校验 + 派发）**：`Infer::generate_stream()`（`router/src/infer/mod.rs:100`）先经 `Validation` 把原始请求变成 `ValidGenerateRequest`，然后 `self.backend.schedule(valid_request)`（`router/src/infer/mod.rs:134`）——这一行是**trait 对象的动态派发**，`Infer` 完全不知道背后是 `BackendV3` 还是 `LlamacppBackend`。

**跳 3（v3 路径：入队）**：`BackendV3::schedule()`（`backends/v3/src/backend.rs:79`）把请求包成 `Entry`，调 `Queue::append()`（`backends/v3/src/queue.rs:76`）——这一步只是往 mpsc channel 发一条 `QueueCommand::Append`，函数立刻返回、不阻塞；随后 `batching_task_notifier.notify_one()` 唤醒批处理循环，`schedule()` 把这条请求专属的 `UnboundedReceiverStream` 作为返回值交回 `Infer`。

**跳 4（预算记账）**：`batching_task()`（`backends/v3/src/backend.rs:129`）平时睡在 `notifier.notified().await`，被唤醒后调 `queue.next_batch(...)`（`backends/v3/src/backend.rs:148-154`），真正的预算判断落在 `State::next_batch()`（`backends/v3/src/queue.rs:237`）：从队首逐个 `pop_front`，若开了前缀缓存就先问 `block_allocator.allocate()`（内部会走到 `radix.rs` 的树匹配），拿到 `prefix_len` 之后算出这条请求还需要多少 `postfix_len` 的新 token 预算；`prefill_tokens + postfix_len > prefill_token_budget`（`backends/v3/src/queue.rs:346`）一旦超限，要么按 chunking 只截一部分塞进这批（`queue.rs:349-360`），要么把整条请求原样塞回队首、结束这一轮攒批（`queue.rs:368-370`）。

**跳 5（跨语言调用）**：拿到一批 `(entries, batch, span)` 后，`prefill(&mut client, batch, None, &mut entries)`（`backends/v3/src/backend.rs:297`）调用 `client.prefill(batch, cached_batch)`（`backend.rs:307`）——这是一次真正的 gRPC 调用，跨过 Unix socket 打到 Python 侧的 `TextGenerationService.Prefill()`（`server/text_generation_server/server.py:151`）。

**跳 6（Python 端：真正的前向）**：`self.model.batch_type.from_pb(...)`（`server.py:130`/`165`，视是否为视觉模型走不同分支）把 protobuf 反序列化成 tensor batch，`generations, next_batch, timings = self.model.generate_token(batch)`（`server.py:181`）才是真正跑一次 GPU 前向；`self.cache.set(next_batch)`（`server.py:182`）把返回的 batch 状态（含 KV tensor）留在 Python 进程内存里，序列化回 Rust 的只是标量元信息包成的 `CachedBatch`。

**跳 7（拆分回各自的响应流）**：Rust 收到 `(generations, next_batch, timings)` 后，`filter_send_generations()`（`backends/v3/src/backend.rs:423`）把每个 token 通过它所属 `Entry.response_tx` 送回各自独立的 mpsc channel——这就是"一批请求共享一次前向计算，但各自拿到自己专属的流式响应"的实现方式；`filter_batch()`（`backend.rs:389`）过滤掉已经命中停止条件的请求，剩下的 `CachedBatch` 继续喂给内层 `while let Some(batch) = cached_batch` 循环反复调用 `decode()`（`backend.rs:342` → `server.py:193`），直到没有活跃请求为止。

**跳 8（OpenAI 兼容路径复用同一条通道）**：`chat_completions()`（`router/src/server.rs:1168`）先把 `ChatRequest` 渲染成 `GenerateRequest`（chat 模板拼接），随后直接复用跳 2～7 的同一条 `Infer::generate_stream`/`Backend::schedule` 通道——这一点和 SGLang 把所有协议收口到 `GenerateReqInput`（`sglang:python/sglang/srt/managers/io_struct.py:162`）是同一种模式：不同协议层最终收口到同一个内部调度对象，只是 TGI 的内部对象叫 `ValidGenerateRequest`（`router/src/validation.rs`），SGLang 叫 `GenerateReqInput`。

**对照跳（`compat_generate`，历史包袱）**：`compat_generate()`（`router/src/server.rs:126`）是三条路径里最老的一条，直接把请求转给 `generate`/`generate_stream`，唯一区别是把返回值包进一个 `Vec` 里以匹配 HuggingFace 旧版 Inference API 的响应形状（源码注释原话"wrap generation inside a Vec to match api-inference"，紧邻 `router/src/server.rs:126` 函数体）。这条路由今天仍然挂在根路径 `POST /` 上（`router/src/server.rs:2204`），是 OpenAI 兼容协议成为事实标准之前留下的、至今没有摘掉的历史包袱。

---

## 5. 设计决策与代价

### 决策 1：把连续批处理调度放进 Rust，不放进 Python

- **为什么这么设计**：调度这件事本质是 CPU 密集、对延迟敏感的整数/指针运算（`State::next_batch()` 要在每次攒批时对队列里的每条 `Entry` 做预算判断、`VecDeque` 的 push/pop），而且必须能和"等 Python 那边一次 gRPC 前向调用返回"这件事**真正并发**，不能被谁的解释器锁卡住。Rust 侧完全没有 GIL，`Queue`/`State` 用 actor 模式串行化写入 + tokio 异步运行时，入队/唤醒都是不涉及锁竞争的纳秒级操作。
- **不这样会怎样**：如果调度逻辑也写在 Python 里、和模型前向共用一个解释器（vLLM/SGLang 的做法），就需要精心保证"调度器这段代码本身很轻、不做重计算"，并且依赖 CPython 在等待 GPU kernel 异步执行返回时会释放 GIL 这个事实——这条路径vLLM/SGLang 都走通了（`## 6` 详述），说明"调度必须用 Rust"从来不是唯一解，只是 TGI 在自己的时间点上做出的选择。
- **什么时候这是负担**：双语言意味着**两套构建链、两套测试体系、两套贡献者技能要求**——改一条完整的请求路径（比如新增一个采样参数）往往要同时改 `proto/v3/generate.proto`（契约）、Rust 端 `ValidGenerateRequest`（校验/透传）、Python 端 batch 反序列化逻辑（`server.py`）三处，缺一不可。贡献者需要同时懂 Rust 的所有权系统和 Python 的 tensor 代码才能完整地改动这条链路，这是纯 Python 或纯 Rust 单语言引擎不需要付的税。

### 决策 2：用 `Backend` trait 抽象执行层，而不是让路由直接认识某一种推理运行时

- **为什么这么设计**：调度逻辑已经写死在 Rust 里之后，让它去指挥不同的执行引擎（PyTorch shard / llama.cpp / TensorRT-LLM）只需要新写一个 `impl Backend for XxxBackend`，`Infer`、HTTP 路由、validation、chat 模板渲染完全不用碰——`router::server::run()`（`router/src/server.rs:1502`）本身是 `pub async fn run(backend: impl Backend + Send + Sync + 'static, ...)` 泛型函数，每个 backend crate（v3/llamacpp/trtllm）的 `main.rs` 各自构造好自己的 `Backend` 实现后调用同一份 `run()`。
- **不这样会怎样**：如果每种执行后端都要重新实现一遍"HTTP 路由 + validation + chat 模板 + SSE 拼装"，这些和执行引擎无关的逻辑会在 4 个以上的 backend crate 里各抄一份——`router/src` 目录被四个后端共用，重复的代价是修一个 bug 要在多份代码里各修一次。
- **什么时候可以不这样**：如果一个引擎从一开始就只打算支持一种执行运行时（vLLM/SGLang 都是"一个 Python 进程内一种调度器认一种 GPU 执行方式"），这层 trait 抽象就是纯粹的间接层开销，付出的心智成本换不回对应的收益。

### 决策 3：每个 TP shard 一个独立 Python 进程，靠本地 gRPC（Unix socket）通信，不做进程内直接调用

- **为什么这么设计**：张量并行本身要求每个 rank 有独立的 CUDA context / NCCL 通信组，多进程是最自然的隔离单元；gRPC + protobuf 给了 Rust 和 Python 一条语言无关、带版本化 schema 的边界（`proto/v3/generate.proto`），Python 侧可以独立升级模型代码而不用碰 Rust 一行。用 Unix domain socket 而不是 TCP，是因为两端本来就在同一台机器上，省掉完整网络栈的开销。
- **不这样会怎样**：如果两端要共享内存/进程内直接调用（比如靠 PyO3 直接把 tensor 传进 Python），调度进程会和某个具体的 Python 解释器绑死生命周期，GIL 又会变成问题——**TGI 在别的地方确实用了 PyO3**（`router/src/lib.rs:38` 的 `PyTokenizer`，用于某些需要走 Python `transformers.AutoTokenizer` 的 tokenizer 场景），但那是极小范围（只做 tokenize，不在批处理主循环上），说明 TGI 团队清楚"进程内 PyO3 调用"和"进程间 gRPC"该分别用在哪种场景。
- **什么时候可以不这样**：单卡、无需 TP 的部署下，多进程 + gRPC 的序列化/IPC 开销是纯负担——这正是 `LlamacppBackend`/`TensorRtLlmBackendV2` 选择进程内 FFI（`cxx`/C 绑定）而不是再起一个 Python 子进程的原因：C++/C 运行时没有 GIL，进程内调用反而更简单、更快、没有额外的序列化成本。

---

## 6. 同位对照

vLLM 和 SGLang 的连续批处理调度器至今都还在 Python 里：`vllm:vllm/v1/core/sched/scheduler.py`（本库 `[[03-vLLM-调度器解剖]]` 的范围）、`sglang:python/sglang/srt/managers/scheduler.py`（`[[02-SGLang-Scheduler事件循环]]` 的范围）——两家都没有把"下一步该选谁进 batch"这个决策挪到另一种语言里，靠的是"GPU kernel launch 是异步的，launch 完 Python 很快把控制权交还给事件循环"这个事实，让调度决策和张量计算在同一个 Python 进程里也不至于被 GIL 卡死。

但 vLLM 现在确实有一个不小的 Rust 代码库：`00-总览与阅读地图.md` §3 统计出 vLLM 有 **109,867 行 Rust**，体量上和 TGI 全部 23,443 行 Rust 相比反而是数倍。打开 `_src/vllm/rust/` 看：这**不是**调度器的 Rust 化，是**前端服务层**的 Rust 化。`README.md` 原话（`vllm:rust/README.md:3`）：「This is a Rust drop-in alternative frontend for vLLM. The current goal is to rebuild the northbound serving layer in Rust while still talking to the core Python vLLM engine process(es) via ZMQ over the existing engine boundary.」——要重建的是"north-bound"（面向客户端）的服务层：HTTP API（`vllm-server`）、chat 模板渲染与工具调用解析（`vllm-chat`）、tokenizer（`vllm-text`），往下通过 `vllm-engine-core-client` 走 ZMQ 连到**完全没有改变的 Python EngineCore**（分层图见 `vllm:rust/README.md:20-31`）。README 同时写明这仍是「experimental...not feature-complete」（`vllm:rust/README.md:5`），`Cargo.toml` 的版本号还停在 `0.1.0`（`vllm:rust/Cargo.toml:22`）。

也就是说：vLLM 的 10 万行 Rust 和 TGI 的 2.3 万行 Rust 虽然都叫"Rust"，落在系统里的位置完全不同——TGI 的 Rust 做的是"batch 怎么攒"（本篇 `## 0`/`## 4` 讲的那条链路）；vLLM 的 Rust 做的是"请求怎么被解析、渲染、tokenize"，批处理决策仍然锁在 Python EngineCore 里没挪窝。两者不能拿"都有 Rust"简单类比。

SGLang 这边也有类似的东西，落点又不一样：`experimental/sgl-router/`（`sglang:experimental/sgl-router/README.md:3` 原话「Slim, KV-aware, OpenAI-compatible router for SGLang workers.」）和更完整的 `sgl-model-gateway/`（`sglang:sgl-model-gateway/README.md:8` 原话「Industry-first gRPC pipeline with native Rust tokenization, reasoning, and tool-call execution for high-throughput OpenAI-compatible serving.」）——这两个都是**跨多个 SGLang server 实例的舰队级负载均衡/路由**（worker 发现、KV 感知路由到该去哪个副本），不是单实例内部"这一步该给哪些请求跑前向"的调度器；单实例内部的批处理决策仍然在 Python 的 `Scheduler` 里。

收束成一句话：三家引擎里，**只有 TGI 把"单实例内、逐请求的 token 预算记账"这一步做进了 Rust**；vLLM 的 Rust 停在请求解析这一层、再往下交给 Python；SGLang 的 Rust 停在多实例路由这一层、再往下也交给 Python。这不是谁更先进，是三个团队在"Rust 值得投在系统的哪个边界"这件事上给出了三个不同答案——TGI 诞生更早（2022 年，那时候"用 Python 异步事件循环写 LLM 调度器"这套后来被 vLLM/SGLang 验证可行的工程经验还没有被证明过），选了把整条服务前台都下沉到 Rust 的做法。

---

## 7. 踩坑与反直觉

**"5 个月没提交"是 `git log` 的事实，"进入维护模式"是上游自己的公开声明**——两者不是一回事，本篇分开摆：快照 `b4adbf2f` 停在 2026-03-21，本库抓取日 2026-08-22，中间 5 个月没有新提交，这是可以用 `git log` 复现的客观事实。与此同时，`README.md` 顶部用 GitHub 的 `[!CAUTION]` 告示框写着（**文档所述**，`README.md:2`）：「text-generation-inference is now in maintenance mode. Going forward, we will accept pull requests for minor bug fixes, documentation improvements and lightweight maintenance tasks.」，紧接着一句（`README.md:4`）点名现在推荐用 vLLM、SGLang，以及 llama.cpp/MLX。这两句话是项目方自己写的，不是本库的推断——本篇不据此再往前多说一句"已经过时"，只是把这两处原文和它们的行号钉在这里，留给读者自己判断这句声明和实际的 issue/PR 活跃度是否吻合（**未查证**：本库没有网络权限去核对 GitHub 上 2026-03-21 之后是否仍有 issue 回复或 PR 合并，只核实了本地快照里 README 文案本身）。

**统计工具在 TGI 身上系统性失灵，而不是碰巧算少了**——`_lab/struct_map.py` 的子系统关键词表是给 12 个引擎共用的同一把尺子，对大多数引擎（Python 为主、文件名多带 `scheduler`/`sched`）是好用的，但 TGI 恰好在两个地方撞上了这把尺子的盲区：① `entrypoint` 桶收 `"server"` 这个关键词，TGI 恰好把自己的顶层目录取名叫 `server/`，导致几乎全部 98,255 行 Python 产品码都落进这一个桶（`## 2` 已用 `_classify()` 直接复核）；② `scheduler`/`kv_cache` 桶要靠 `scheduler`/`radix_cache`/`block_manager` 这类关键词命中，TGI 把对应文件分别取名 `queue.rs`/`radix.rs`/`block_allocator.rs`，一个都不命中，于是这两个桶报出的数字和 TGI 真实的调度/前缀缓存代码量几乎没有关系。**这条踩坑值得记住的不是"TGI 代码少"，是"跨引擎统计工具的关键词表本身携带了对某一类引擎命名习惯的隐性假设"。**

**测试比例 0.108 只覆盖了半个语言**——`_lab/repo_stats.py` 判定一行 Python 是不是测试代码，靠的是路径里有没有 `test`（`_lab/repo_stats.py:94`），这条规则完全没有考虑 Rust 的 `#[cfg(test)]` 惯例。直接 grep 出 `router/`、`backends/`、`launcher/` 三个目录下共 9 个文件带 `#[cfg(test)] mod tests` 块（`router/src/chat.rs`、`router/src/config.rs`、`router/src/infer/chat_template.rs`、`router/src/lib.rs`、`router/src/validation.rs`、`router/src/vertex.rs`、`backends/v2/src/queue.rs`、`backends/v3/src/queue.rs`、`backends/v3/src/radix.rs`），共 71 个 `#[test]`/`#[tokio::test]` 函数。最集中的例子就是本篇反复引用的 `backends/v3/src/queue.rs`：文件总长 836 行，`mod tests` 从第 560 行开始一直到文件结尾，**近 33% 的篇幅是测试**，包括直接针对 `## 4` 走读过的预算记账逻辑写的 `test_next_batch_token_budget`/`test_queue_next_batch_token_budget`。这些行一行都没有计入 0.108——所以这个数字准确的说法是"Python 产品码里测试占比 10.8%"，不是"TGI 测试覆盖薄弱"，两者不能划等号。

**`compat_generate` 是活着的历史化石**——`router/src/server.rs:126` 这个函数存在的唯一理由是兼容一个更早的、非 OpenAI 形状的 HuggingFace Inference API 响应格式（返回值要包一层 `Vec`），今天它仍然挂在最显眼的根路径 `POST /` 上（`router/src/server.rs:2204`），排在路由表的第一条，比 `/v1/chat/completions`（`router/src/server.rs:2207`）还靠前。这是一处直接能在代码里读出来的"协议演化痕迹"：先有专用 API，后补 OpenAI 兼容层，两层至今并存，谁都没有被移除。

---

## 8. 可改进点

1. **`launcher` 默认只知道怎么拉起 v3 后端**：`launcher/src/main.rs:1975` 硬编码 `Command::new("text-generation-router")`，而只有 `backends/v3/Cargo.toml:12-13` 的 `[[bin]]` 恰好取了这个名字；`backends/llamacpp/Cargo.toml` 的 crate 名是 `text-generation-router-llamacpp`（产出的二进制名字不同），`backends/trtllm/Cargo.toml` 的 crate 名是 `text-generation-backends-trtllm`（本身是 lib，自己另有 `main.rs` 构建独立二进制）。这意味着统一的 `launcher` 目前实际只能真正驱动 v3 一种后端，其余后端要跑起来需要绕开 `launcher` 自己拼命令行。如果这是有意为之（v3 是唯一"一键路径"，其余是需要手动接线的 opt-in），值得在 `launcher --help` 或顶层 README 里显式写明——现在这条边界只能靠读源码才能确认。
2. **本库自己的 `_lab/struct_map.py` 关键词表该扩一版**（这是本库工具的问题，不是 TGI 上游的问题，但因为本篇高度依赖它才发现，顺手记在这里）：`SUBSYSTEMS` 字典（`_lab/struct_map.py:34-45`）应该加入 `queue`/`radix`/`block_alloc` 这几个关键词，否则任何一个像 TGI 这样把调度器文件直接命名成 `queue.rs`（而不是随大流叫 `scheduler.py`）的引擎，都会被这把尺子系统性漏统计。以后接入新引擎时，应该先用 `## 2` 里演示的方式跑一遍 `_classify()` 对着该引擎的几个关键文件名做一次自测，而不是直接相信 `out/struct_map.json` 的数字。
3. **README 的维护模式声明留了一个没说清的问题**：`README.md:2` 用词是"lightweight maintenance tasks"，没有明确这是否包含安全补丁——对一个自称"Used in production at Hugging Face"（`README.md:21`）的项目，这行文案没有回答"现在还能不能提安全漏洞修复的 PR、大概多久会被合并"这个实际问题。本库不代替上游回答，只指出这处文案本身存在歧义，值得上游用一句更明确的话补上。

---

## 9. 自测题与延伸阅读

**闭卷自测**

1. TGI 的 Rust router 和 Python server 之间用什么协议通信？协议契约文件在哪，定义了哪两个核心 RPC？
2. 连续批处理里"这一批该装哪些请求"的预算判断函数叫什么、在哪个文件、大致在哪一行？它同时比较的是哪两个预算量？
3. `Backend` trait 目前有哪几种已知实现？其中哪两种完全不经过 Python 进程？它们各自怎么跟自己的推理运行时通信？
4. TGI 的 HTTP 表面一共抽到多少条路由、去重后多少条唯一路径，其中 `/v1/*` 前缀的有几条？这个比例说明了 TGI 协议演化史上的什么历史包袱？
5. 测试/产品代码比 0.108 这个数字是怎么算出来的？它对 Rust 代码可见吗？举出本篇引用过的一处具体反例。
6. `_lab/struct_map.py` 把 TGI 的 `scheduler` 子系统统计成多少行？为什么会这样，问题出在关键词表还是出在 TGI 代码本身？
7. vLLM 现在也有十万行级的 Rust 代码，它和 TGI 的 Rust 分别落在系统的哪个位置？两者能不能简单类比？

**延伸阅读（双链）**

- [[03-vLLM-调度器解剖]] —— vLLM 把功能对等的连续批处理逻辑放在 Python 里长什么样，和本篇 `## 6` 的对照点对应
- [[02-SGLang-Scheduler事件循环]] —— SGLang 的 Python 调度器事件循环，`## 6` 的另一个对照点
- [[08-SGLang-HTTP-API表面全解]] —— 同类型 API 表面全解的姊妹篇，可以直接对比"路由数量 vs `/v1/*` 占比"这两个数字在两个引擎之间的差距有多大
- [[11-开源推理引擎谱系图]] —— 本篇属于 `03-其他引擎/` 系列，谱系图篇会把 TGI 放进整个开源推理引擎的时间线里定位
