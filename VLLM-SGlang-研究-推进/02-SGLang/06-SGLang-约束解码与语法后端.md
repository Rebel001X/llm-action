# SGLang 约束解码与语法后端

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：编译异步不卡主循环，但等结果的忙等会；跳跃解码接口活着，调用点已经死了。

## 0. 结论先行

- **默认后端是 xgrammar**，可选 `outlines`/`llguidance`/`none`，还留了一个运行时注册表允许插自定义后端（`create_grammar_backend` 走 `GRAMMAR_BACKEND_REGISTRY` 优先）。三个内置后端支持的约束类型不对等：outlines 不支持 EBNF、不支持 structural_tag；llguidance 的 structural_tag 只认旧版 `structures`/`triggers` 格式；只有 xgrammar 四种类型（json/regex/ebnf/structural_tag，新旧两种 structural_tag 写法）全支持。
- **语法编译本身不卡 Scheduler 主循环——它被丢进 `ThreadPoolExecutor`**，但**检查编译是否完成这一步会卡**：`get_ready_grammar_requests()` 在主循环里跑一个有限忙等（默认 5ms，`SGLANG_GRAMMAR_POLL_INTERVAL`），每轮调度只要 `grammar_queue` 非空就会执行这个忙等，这是常被忽略的细节。
- **bitmask 的生命周期是 CPU 分配 → CPU 逐行填充 → H2D 拷贝 → GPU kernel 应用**，全部塞在 `model_runner.sample()` 的 `_preprocess_logits()` 里，每个前向步都要跑一遍；不同后端的"逐行填充"并行度不同：xgrammar 用的是 SGLang 自己写的 Python for 循环（`BaseGrammarObject.fill_vocab_mask_batched`），llguidance 用它自带的原生并行 executor。
- **跳跃式解码（jump-forward）的接口在四个后端类里都完整实现了**（`try_jump_forward` / `jump_forward_str_state` / `jump_and_retokenize`），xgrammar 和 llguidance 的实现是真实可用的（分别包着 xgrammar 的 `find_jump_forward_string()` 和 llguidance 的 `compute_ff_tokens()`），但**在 `python/sglang/srt/managers/` 整个目录里找不到任何一处调用这三个方法**——这是本篇最重要的发现，`## 4.4`、`## 7` 详细展开。
- **语法状态不靠 RadixAttention 的前缀缓存恢复，而是作为 `Req.grammar` 这个 Python 对象本身在 retract / PD 传输过程中被原样带着走**；真正的正确性机关是 `grammar.current_token is None` 这个哨兵，防止同一个 token 被 `accept_token()` 两次——代码注释里明确写着这条防线是为了堵一个真实发生过的 "Tokens not accepted → FINISH_ABORT" bug。
- **不支持的 schema 不会静默退化成无约束生成，会显式中止请求**：`dispatch_*` 对不支持的类型返回 `InvalidGrammarObject`，之后统一在 `req.set_finish_with_abort()` 里报错，客户端能看到明确的失败原因。
- **"严格推理模式"（思考预算/强制结束思考）走的是另一条更轻量的 token filter 通道，不经过完整的语法自动机**：只有 xgrammar 后端声明支持（`is_support_token_filter`），配 `--enable-strict-thinking` 和其它后端会在服务启动阶段（不是等到具体请求）就直接拒绝启动。

**速查：任务要求的问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| 后端清单、默认值、支持类型差异表 | `## 2`、`## 3.4` |
| `BaseGrammarBackend` 逐方法、编译同步/异步 | `## 3.3`、`## 4.1`、`## 5` 决策一 |
| bitmask 生命周期（CPU/GPU 分工） | `## 4.3`、`## 5` 决策二 |
| jump-forward 原理、实现位置、正确性风险 | `## 4.4`、`## 7` 反直觉一 |
| 与 RadixAttention 的交互、前缀命中后语法状态怎么恢复 | `## 4.5`、`## 5` 决策五 |
| 失败模式：报错还是静默降级 | `## 4.1`、`## 5` 决策六 |

## 1. 它在系统里的位置

一个语法约束请求从进入系统到影响采样，跨越四个已有子系统（[[02-SGLang-Scheduler事件循环]] 讲过的调度主循环、[[03-SGLang-RadixAttention与前缀缓存]] 讲过的前缀树、采样层、模型执行层），本篇是把"约束解码"这条线单独抽出来走读：

```
HTTP 请求（response_format / regex / ebnf 等参数）
   │
   ▼
SamplingParams.json_schema / .regex / .ebnf / .structural_tag（互斥，四选一或都不选）
   │  python/sglang/srt/sampling/sampling_params.py:71-74
   ▼
Scheduler.handle_generate_request()
   │  → GrammarManager.process_req_with_grammar(req)
   │    python/sglang/srt/managers/scheduler.py:2722
   ▼
命中缓存 ─┬─ 是 → 直接拿到编译好的 BaseGrammarObject.copy()，进 waiting_queue
          │
          └─ 否 → 丢进 ThreadPoolExecutor 后台编译，req 进 grammar_queue（不是 waiting_queue）
                     │
                     ▼
             每轮调度前先 poll grammar_queue（有限忙等）
             python/sglang/srt/managers/scheduler.py:3258-3259
                     │  编译完成/超时 → 转入 waiting_queue
                     ▼
             正常 prefill/decode 批调度（不在本篇展开，见 [[02-SGLang-Scheduler事件循环]]）
                     │
                     ▼
             SamplingBatchInfo.update_regex_vocab_mask()（每个前向步）
             python/sglang/srt/sampling/sampling_batch_info.py:240-265
                     │  分配 → 填充 → 搬到 GPU → 应用到 logits
                     ▼
             sampler 采样出 next_token_id
                     │
                     ▼
             req.grammar.accept_token(next_token_id)（每生成一个 token 就推进一次 FSM）
                     │
                     └─ 循环回到"正常 prefill/decode 批调度"，直到 grammar.is_terminated()
```

约束解码在这条链路上不是一个独立进程或独立线程，而是**寄生在 Scheduler 单进程的几个固定挂钩点上**：请求入队时（编译）、每轮定批前（轮询编译结果）、每个前向步采样前（填 bitmask）、每个 token 生成后（推进 FSM）。它天生要跟 Scheduler 的主循环节奏对齐，这也是 `## 5` 决策一要讨论"为什么编译要丢进线程池"的直接原因。

约束解码还和推理模型的"思考预算"功能有一处显式接口：请求可以在 `sampling_params.custom_params` 里带一个 `thinking_budget` 字段，`GrammarManager._get_request_thinking_budget()`（`python/sglang/srt/constrained/grammar_manager.py:117`-`122`）把它读出来，`_apply_request_reasoning_budget()`（`python/sglang/srt/constrained/grammar_manager.py:124`-`129`）在语法对象类型是 `ReasonerGrammarObject` 时把它写进 `req.grammar.max_think_tokens`——这是一个按请求覆盖全局默认思考预算（`## 4.7` 会展开 `SGLANG_MAX_THINK_TOKENS` 这个全局默认值）的旁路，调用点同时出现在缓存命中路径（`process_req_with_grammar()` 里，`python/sglang/srt/constrained/grammar_manager.py:170`）和异步编译完成路径（`get_ready_grammar_requests()` 里，`python/sglang/srt/constrained/grammar_manager.py:287`），确保无论语法对象是同步拿到的还是异步编译出来的，per-request 的思考预算都会被应用一次。

## 2. 代码地图（文件 → 职责，带行号）

`_lab/struct_map.py` 把这个子系统统计为 14 个文件、2,933 行（`_lab/out/struct_map.json` 里 `engines.sglang.subsystems.structured_output.n_files` / `.lines`），是 SGLang 十个子系统里代码量最小的一个（[[01-SGLang-全景与代码地图]] 已经点过这个数字）。按职责分组：

**接口与调度层**（`python/sglang/srt/constrained/`）：

- `python/sglang/srt/constrained/base_grammar_backend.py:58` —— `class BaseGrammarObject`，单个请求的语法状态抽象基类，规定了 `accept_token`/`rollback`/`allocate_vocab_mask`/`fill_vocab_mask`/`try_jump_forward` 等接口。
- `python/sglang/srt/constrained/base_grammar_backend.py:202` —— `class BaseGrammarBackend`，负责编译与缓存的抽象基类，持有 `self.executor = ThreadPoolExecutor()`（`python/sglang/srt/constrained/base_grammar_backend.py:206`）和 `self.cache: Dict[Tuple[str,str], BaseGrammarObject]`（`python/sglang/srt/constrained/base_grammar_backend.py:207`）。
- `python/sglang/srt/constrained/base_grammar_backend.py:351` —— `create_grammar_backend()`，按 `--grammar-backend` 旋钮实例化具体后端的工厂函数。
- `python/sglang/srt/constrained/grammar_manager.py:26` —— `class GrammarManager`，挂在 `Scheduler` 上的语法子系统总控，持有 `grammar_queue`（等编译完成的请求队列）。
- `python/sglang/srt/managers/scheduler.py:1996` —— `Scheduler.init_grammar_manager()`，Scheduler 构造期间创建 `GrammarManager` 的地方。

**三个内置后端**：

- `python/sglang/srt/constrained/xgrammar_backend.py:73` / `python/sglang/srt/constrained/xgrammar_backend.py:208` —— `XGrammarGrammar` / `XGrammarGrammarBackend`，默认后端，包装 [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)。
- `python/sglang/srt/constrained/llguidance_backend.py:98` / `python/sglang/srt/constrained/llguidance_backend.py:211` —— `GuidanceGrammar` / `GuidanceBackend`，包装 [guidance-ai/llguidance](https://github.com/guidance-ai/llguidance)。
- `python/sglang/srt/constrained/outlines_backend.py:42` / `python/sglang/srt/constrained/outlines_backend.py:114` —— `OutlinesGrammar` / `OutlinesGrammarBackend`，包装 [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)，是三者里支持类型最少的。
- `python/sglang/srt/constrained/outlines_jump_forward.py:142` —— `OutlinesJumpForwardMap`，2024 年"压缩 FSM"博客配套的跳跃表构造器，`## 4.4` 会说明它现在处于什么状态。

**推理时思考模式包装**：

- `python/sglang/srt/constrained/reasoner_grammar_backend.py:35` / `python/sglang/srt/constrained/reasoner_grammar_backend.py:243` —— `ReasonerGrammarObject` / `ReasonerGrammarBackend`，把"思考中不生效、思考结束后才生效"这层语义包在任意底层语法对象外面。

**GPU/CPU 算子层**：

- `python/sglang/kernels/ops/grammar/bitmask_ops.py:13` —— `apply_token_bitmask_inplace_kernel`，应用 bitmask 到 logits 的 Triton kernel（改自 xgrammar 自带的同名 kernel）。
- `python/sglang/kernels/aot/csrc/grammar/apply_token_bitmask_inplace_cuda.cu`（271 行，HIP/ROCm 路径用）。
- `python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py:27` —— `set_token_filter_torch`，非 CUDA 设备上"只放行/屏蔽几个 token"的 torch 兜底实现。

**与其它子系统的挂钩点**：

- `python/sglang/srt/managers/scheduler.py:2722` —— `handle_generate_request()` 里调用 `process_req_with_grammar()` 的地方，请求刚从 tokenizer 那边到达时。
- `python/sglang/srt/managers/scheduler.py:3258`-`3259` —— `_get_new_batch_prefill_raw()` 里轮询 `grammar_queue` 的地方。
- `python/sglang/srt/managers/scheduler.py:1879`-`1893` —— `is_disable_overlap_for_batch()` 里判断语法是否需要强制打断 overlap 流水线的分支。
- `python/sglang/srt/managers/schedule_batch.py:1112`-`1116` —— `Req.grammar`/`Req.grammar_key`/`Req.grammar_wait_ct` 三个字段的定义处。
- `python/sglang/srt/managers/schedule_batch.py:2247`-`2250` —— `ScheduleBatch.grammar_needs_sync()`。
- `python/sglang/srt/sampling/sampling_batch_info.py:240`-`265` —— `update_regex_vocab_mask()`，bitmask 生命周期的入口。
- `python/sglang/srt/model_executor/model_runner.py:1764`-`1790` —— `_preprocess_logits()`，采样前把 bitmask 应用到 logits 的地方。
- `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py:113`-`139` —— PD 分离 decode 侧 `process_prebuilt()`，含 `current_token is None` 那条正确性哨兵。
- `python/sglang/srt/speculative/spec_utils.py:440`-`489` —— `traverse_tree()`，投机解码草稿树的 DFS 遍历里逐节点推进语法 FSM、逐节点生成 bitmask。
- `python/sglang/srt/speculative/spec_info.py:137`-`142` —— `SpeculativeAlgorithm.supports_grammar_overlap()`，决定哪些投机解码算法允许语法与 overlap 调度共存。
- `python/sglang/srt/sampling/sampling_params.py:71`-`74`、`python/sglang/srt/sampling/sampling_params.py:201`-`210` —— `SamplingParams` 上四个互斥字段的定义与互斥性校验，语法约束从 HTTP 请求参数落到调度器可读字段的最后一跳。

## 3. 核心数据结构

### 3.1 `BaseGrammarObject`（`python/sglang/srt/constrained/base_grammar_backend.py:58`）

单个请求的语法状态机，字段很少：`_finished`（是否终止）、`grammar_stats`（`GrammarStats`，记编译耗时等指标）、`current_token`（最近一次 `accept_token()` 接受的 token，`__init__` 里初始为 `None`，`## 4.5` 会说明它的正确性用途）。接口方法分四组：

| 组 | 方法 | 作用 |
|---|---|---|
| 推进/回滚 | `accept_token(token)`、`rollback(k)`、`is_terminated()` | 逐 token 推进/回退 FSM |
| bitmask | `allocate_vocab_mask`、`fill_vocab_mask`、`fill_vocab_mask_batched`（静态方法，`python/sglang/srt/constrained/base_grammar_backend.py:88`-`94`）、`move_vocab_mask`、`apply_vocab_mask` | `## 4.3` 逐步走读 |
| 复制 | `copy()` | 请求级实例化：缓存里存的是"编译产物模板"，每个请求 `.copy()` 出自己的独立 matcher |
| 跳跃解码 | `try_jump_forward`、`jump_forward_str_state`、`jump_and_retokenize` | `## 4.4` 详细展开 |

### 3.2 `GrammarStats`（`python/sglang/srt/constrained/base_grammar_backend.py:39`-`48`）

一个纯记账用的 dataclass：`compilation_time`、`schema_count`、`ebnf_size`、`is_cache_hit`、`is_grammar_aborted`、`tree_traversal_time`（投机解码树遍历耗时列表）、`dispatch_type`（"json"/"regex"/"ebnf"/"structural_tag"）、`num_timeout`。这些字段目前只在源码里能看到定义和赋值点，本篇没有找到它们被导出成 Prometheus 指标或 HTTP 响应字段的位置——**未查证**它们最终流向哪里，只能确认它们存在且被 `_init_value_dispatch()`（`python/sglang/srt/constrained/base_grammar_backend.py:259`-`282`）在编译完成后填了 `compilation_time`。

### 3.3 `BaseGrammarBackend`（`python/sglang/srt/constrained/base_grammar_backend.py:202`）

进程级单例（每个 Scheduler 进程一个，不是每请求一个）。核心字段：

- `self.executor: ThreadPoolExecutor`（`python/sglang/srt/constrained/base_grammar_backend.py:206`）—— 编译任务的执行池，默认参数（未传 `max_workers`，走 Python 默认值）。
- `self.cache: Dict[Tuple[str, str], BaseGrammarObject]`（`python/sglang/srt/constrained/base_grammar_backend.py:207`）—— 缓存 key 是 `(key_type, key_string)` 二元组，比如 `("json", '{"type": "object", ...}')`，**不做任何归一化**（两个语义等价但字符串不同的 JSON Schema 会被当成两个缓存条目分别编译）。
- `dispatch_json`/`dispatch_regex`/`dispatch_ebnf`/`dispatch_structural_tag`（`python/sglang/srt/constrained/base_grammar_backend.py:247`-`257`）—— 基类默认实现全部是 `self._not_supported(...)`（`python/sglang/srt/constrained/base_grammar_backend.py:219`-`221`），子类按自己支持的类型 override。
- `get_cached_or_future_value(key, require_reasoning)`（`python/sglang/srt/constrained/base_grammar_backend.py:284`-`293`）—— 缓存命中就同步返回 `value.copy()`；不命中就 `self.executor.submit(self._init_value_dispatch, key, require_reasoning)`，返回一个 `concurrent.futures.Future`。**这一行是"编译是否阻塞主循环"这个问题的分水岭**，`## 4.1` 展开。

### 3.4 三个内置后端的支持类型矩阵

综合源码（`dispatch_*` 系列方法是否 override 基类的 `_not_supported`）和文档（`docs/docs/advanced_features/structured_outputs.mdx:8`-`18`，「文档所述」）两个证据源：

| 后端 | json_schema | regex | ebnf | structural_tag |
|---|---|---|---|---|
| **xgrammar**（默认） | 支持 | 支持 | 支持 | 支持（新旧两种写法，`python/sglang/srt/constrained/xgrammar_backend.py:368`-`399`） |
| **llguidance** | 支持 | 支持 | 支持 | 只支持旧版 `structures`/`triggers` 格式，新版 `format` 写法会在 `assert is_legacy_structural_tag(...)`（`python/sglang/srt/constrained/llguidance_backend.py:286`）处直接抛异常 |
| **outlines** | 支持（内部转成 regex 再编译） | 支持 | **不支持**——`dispatch_ebnf` 直接 `return super().dispatch_ebnf(...)`（`python/sglang/srt/constrained/outlines_backend.py:160`-`161`），落到基类的 `_not_supported` | **不支持**，同理（`python/sglang/srt/constrained/outlines_backend.py:163`-`164`） |

`--grammar-backend none` 是第四个"选项"，但它不是一个后端类——`create_grammar_backend()` 对 `name == "none"` 直接 `return None`（`python/sglang/srt/constrained/base_grammar_backend.py:415`-`423`），`GrammarManager.grammar_backend` 整个是 `None`，任何带 `json_schema`/`regex`/`ebnf`/`structural_tag` 的请求会在 `process_req_with_grammar()` 里被立即 `set_finish_with_abort`（`python/sglang/srt/constrained/grammar_manager.py:140`-`142`）。

默认值来自 `python/sglang/srt/server_args.py:6235`-`6237` 的 `_handle_grammar_backend()`：`self.grammar_backend is None` 时赋值为 `"xgrammar"`；候选集合是 `GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]`（`python/sglang/srt/server_args.py:256`），并且 `add_grammar_backend_choices()`（`python/sglang/srt/server_args.py:440`-`441`）留了扩展口——配合 `python/sglang/srt/constrained/base_grammar_backend.py:36` 的模块级字典 `GRAMMAR_BACKEND_REGISTRY` 和 `python/sglang/srt/constrained/base_grammar_backend.py:347`-`348` 的 `register_grammar_backend(name, init_func)`，第三方可以注册一个自定义后端名，`create_grammar_backend()`（`python/sglang/srt/constrained/base_grammar_backend.py:361`-`364`）里自定义后端的优先级高于四个内置选项。

### 3.5 批量填充用的三个 `NamedTuple`

`fill_vocab_mask_batched()`（`## 4.3` 第 2 步）不是简单地对每个语法对象调一次 `fill_vocab_mask`，而是先把"哪些行需要填、用哪个语法对象填"打包成列表，三个轻量 `NamedTuple` 分别对应三种打包粒度：

- `GrammarRow`（`python/sglang/srt/constrained/base_grammar_backend.py:51`-`56`）—— 最基础的单位：`row`（这个请求在批内的行号）+ `grammar`（对应的语法对象）。`SamplingBatchInfo.update_regex_vocab_mask()`（`## 4.3`）构造的 `entries` 列表就是一串 `GrammarRow`。
- `GrammarMask`（`python/sglang/srt/constrained/base_grammar_backend.py:149`-`159`）—— 填充完成后的产物：`grammar`（批内任选一个语法对象，只是用来调用 `apply_vocab_mask` 这个静态/类方法，不代表整批共享同一个语法状态）+ `vocab_mask`（已经填好、搬到 GPU 的整批张量）。`.apply(logits)` 方法（`python/sglang/srt/constrained/base_grammar_backend.py:158`-`159`）就是 `## 4.3` 第 4 步的入口。
- `GrammarDraftRow`（`python/sglang/srt/constrained/llguidance_backend.py:46`-`51`）—— llguidance 独有，用于投机解码场景：在 `GrammarRow` 的基础上多带一个 `draft_tokens: List[int]` 字段，配合 `fill_token_bitmask_with_draft_tokens()`（`python/sglang/srt/constrained/llguidance_backend.py:59`-`76`）把"这一行的草稿 token 链"也交给原生 executor，一次性算出链上每个节点各自的合法 token 集合，对应 `## 4.6` 提到的 xgrammar 路径里用 CPU DFS（`traverse_tree()`）手动做的事情——llguidance 把它下沉到了库内部的并行实现里。

三者的共同点是**都不携带张量数据本身**，只携带"往批内哪一行填、用谁的状态填"这份轻量索引，真正的大张量（`vocab_mask`）作为参数单独传递——这是一种常见的"索引和数据分离"打包方式，避免了在构造这些小对象时拷贝或引用大张量的子视图。

## 4. 主流程走读

### 4.1 编译：异步提交，同步等待

`GrammarManager.process_req_with_grammar(req)`（`python/sglang/srt/constrained/grammar_manager.py:131`-`182`）是入口。核心逻辑：按 `req.sampling_params` 上四个互斥字段中非 `None` 的那个拼出 `key = (key_type, key_string)`（`python/sglang/srt/constrained/grammar_manager.py:144`-`151`），调用 `self.grammar_backend.get_cached_or_future_value(key, req.require_reasoning)`（`python/sglang/srt/constrained/grammar_manager.py:153`）。这一步分两条路：

- **缓存命中**（`value` 是已编译好的 `BaseGrammarObject`）：`req.grammar = value`（其实是 `value.copy()`，`python/sglang/srt/constrained/base_grammar_backend.py:289`），**同步**完成，请求直接可以进 `waiting_queue`。
- **缓存未命中**：`self.executor.submit(self._init_value_dispatch, key, require_reasoning)`（`python/sglang/srt/constrained/base_grammar_backend.py:292`）把真正的编译工作（`dispatch_json`/`dispatch_regex`/`dispatch_ebnf`/`dispatch_structural_tag`，调用 xgrammar/llguidance/outlines 各自的编译器）扔进后台线程池，`req.grammar` 暂时是一个 `Future`，请求被塞进 `self.grammar_queue`（`python/sglang/srt/constrained/grammar_manager.py:179`-`180`）而不是 `waiting_queue`——**这一步不阻塞调用者**，`process_req_with_grammar()` 立刻返回。

真正会占用主循环时间片的是下一步：`get_ready_grammar_requests()`（`python/sglang/srt/constrained/grammar_manager.py:184`-`311`），每轮调度只要 `has_waiting_grammars()`（`python/sglang/srt/constrained/grammar_manager.py:69`-`70`）为真就会被 `_get_new_batch_prefill_raw()`（`python/sglang/srt/managers/scheduler.py:3258`-`3259`）调用。它的核心是一段**有限忙等**（`python/sglang/srt/managers/scheduler.py:205`-`225`）：

```python
start_time = time.perf_counter()
while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
    for i, req in enumerate(self.grammar_queue):
        ...
        if req.grammar.done():
            ready_req_idxs.add(i)
    if len(ready_req_idxs) == len(self.grammar_queue):
        break
    time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)
```

`SGLANG_GRAMMAR_POLL_INTERVAL` 默认 `0.005`（5ms，`python/sglang/srt/environ.py:377`）。也就是说：**只要 `grammar_queue` 里还有没编译完的请求，Scheduler 每轮定批前最多会在这个 while 循环里原地等最多 5ms**（内部用 0.5ms 粒度的 `time.sleep` 轮询，不是真正的忙转），这段时间调度器什么别的事都不做。这不是"编译卡住了主循环"，而是"检查编译有没有完成"这件事本身有一个有意为之的、有上限的阻塞窗口——`## 5` 决策一详细讨论这个权衡。超过 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS`（默认 10000 次，`python/sglang/srt/environ.py:378`，对应最坏情况约 50 秒）还没编译完的请求会被判定超时（`python/sglang/srt/environ.py:227`-`239`），走 `failed_req_idxs` 分支，`req.grammar.cancel()` 后以 `InvalidGrammarObject("Grammar preprocessing timed out")` 写入缓存并中止请求（`python/sglang/srt/environ.py:293`-`303`）。

TP/DP 多卡场景下还有一层同步：`torch.distributed.all_gather_object()`（`python/sglang/srt/environ.py:246`-`251`）让所有 rank 对"哪些请求编译好了/哪些超时了"取交集/并集共识（`python/sglang/srt/environ.py:252`-`255`），因为同一批请求必须在所有 rank 上以完全相同的方式进批——这是分布式推理里"控制流必须在所有 rank 上一致"这条通用约束的一个具体实例。PP（流水线并行）场景下，PP0 的判定结果通过 `_pp_sync_ready_failed()`（`python/sglang/srt/environ.py:77`-`107`）异步 P2P 传给后续 PP rank。

编译失败或不支持的类型，走的都是 `InvalidGrammarObject`（`python/sglang/srt/constrained/base_grammar_backend.py:191`-`199`）这一条路，携带 `error_message`。它被当成一个正常缓存值存进 `self.cache`（意味着同一个坏 schema 之后的请求也会立刻命中"已知无效"而不必重新编译一次去发现它无效），消费端统一在两处检查它并调用 `req.set_finish_with_abort(error_msg)`：缓存命中路径在 `python/sglang/srt/constrained/grammar_manager.py:162`-`168`，异步编译路径在 `python/sglang/srt/constrained/grammar_manager.py:288`-`290`。**结论：不支持的 schema / 编译失败的 schema 都不会静默退化成无约束生成，都会带着具体错误信息中止请求。**

### 4.2 请求进批之后：token 级别的推进

一旦请求进入 `waiting_queue`，剩下的调度逻辑（哪个请求进哪一批、prefill 还是 decode）和语法无关，见 [[02-SGLang-Scheduler事件循环]]。语法子系统只在两处继续起作用：

1. 每个前向步采样前，`SamplingBatchInfo.update_regex_vocab_mask()` 把批内所有带语法的请求的"下一步合法 token 集合"编码成 bitmask（`## 4.3`）。
2. 采样出 `next_token_id` 之后，`req.grammar.accept_token(next_token_id)` 推进 FSM 一步（普通 decode 路径的调用点是 `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:641`、`python/sglang/srt/managers/scheduler_components/batch_result_processor.py:779`，chunked prefill 的调用点是 `_apply_prefill_grammar()`，`python/sglang/srt/managers/scheduler_components/batch_result_processor.py:632`-`650`）。

### 4.3 bitmask 生命周期：CPU 分配、CPU 填充、H2D、GPU 应用

四步全部串在 `ModelRunner._preprocess_logits()`（`python/sglang/srt/model_executor/model_runner.py:1764`-`1790`）里，被 `sample()`（`python/sglang/srt/model_executor/model_runner.py:1792`-`1846`）在**每个前向步、采样之前**同步调用：

**第 1 步：分配（CPU，pinned）。** `SamplingBatchInfo.update_regex_vocab_mask()`（`python/sglang/srt/sampling/sampling_batch_info.py:240`-`265`）从批内第一个非空语法对象拿到 `allocate_vocab_mask(vocab_size, batch_size, device)`。xgrammar 的实现（`python/sglang/srt/constrained/xgrammar_backend.py:61`-`70`，`_allocate_token_bitmask`）用 `torch.full(..., pin_memory=is_pin_memory_available())` ——**没有传 `device` 参数，张量默认建在 CPU 上**，只是钉了页（pinned），为的是后面第 3 步能做真正的 non-blocking H2D 拷贝（不钉页的话 `non_blocking=True` 会被静默降级成同步拷贝）。

**第 2 步：填充（CPU，逐行）。** `first_grammar.fill_vocab_mask_batched(entries, vocab_mask)`（`python/sglang/srt/sampling/sampling_batch_info.py:261`）。`entries` 只包含"未结束且未终止"的行（`python/sglang/srt/sampling/sampling_batch_info.py:256`-`260`）——已结束/未使用语法的行保持分配时的全 -1（xgrammar 语义里代表"全部放行"）不动。基类的默认实现（`python/sglang/srt/constrained/base_grammar_backend.py:88`-`94`）就是一个 Python for 循环，一行一行调用每个请求自己的 `fill_vocab_mask(vocab_mask, idx)`；xgrammar 的 `fill_vocab_mask()`（`python/sglang/srt/constrained/xgrammar_backend.py:119`-`120`）直接转发给 `self.matcher.fill_next_token_bitmask(vocab_mask, idx)`——这是 xgrammar 库自己的 C++/Rust 实现在填，SGLang 这边只负责按行调度，**没有对这个循环本身做并行化**。llguidance 是唯一 override 了 `fill_vocab_mask_batched`（`python/sglang/srt/constrained/llguidance_backend.py:151`-`159`）的后端：如果整批都是 `GuidanceGrammar`，走它自带的原生并行 executor `fill_next_token_bitmask_par()`（`python/sglang/srt/constrained/llguidance_backend.py:79`-`87`，底层是 `LLExecutor`，一个跨请求并行处理多个 matcher 的 Rust 线程池）。

**第 3 步：搬到 GPU（H2D，non-blocking）。** `first_grammar.move_vocab_mask(vocab_mask, self.device)`（`python/sglang/srt/sampling/sampling_batch_info.py:264`），三个后端的实现都是 `vocab_mask.to(device, non_blocking=True)`。

**第 4 步：应用到 logits（GPU，kernel）。** `sampling_info.apply_logits_bias(logits_output.next_token_logits)`（`python/sglang/srt/model_executor/model_runner.py:1782`）→ `GrammarMask.apply()`（`python/sglang/srt/constrained/base_grammar_backend.py:158`-`159`）→ `grammar.apply_vocab_mask(logits, vocab_mask)`。xgrammar 在 CUDA/XPU/MUSA 上走 `apply_token_bitmask_inplace_triton`（`python/sglang/srt/constrained/xgrammar_backend.py:126`-`131`，kernel 定义在 `python/sglang/kernels/ops/grammar/bitmask_ops.py:13`-`24`，改自 xgrammar 自己的同名 Triton kernel）；HIP（AMD ROCm）走 `apply_token_bitmask_inplace_cuda`（来自 `sgl_kernel`，对应 271 行的 `apply_token_bitmask_inplace_cuda.cu`）；NPU 走 `torch.ops.npu.apply_token_bitmask`（`python/sglang/kernels/ops/grammar/bitmask_ops.py:132`-`135`）；CPU 走 xgrammar 库自带的 `apply_token_bitmask_inplace(..., backend="cpu")`（`python/sglang/kernels/ops/grammar/bitmask_ops.py:136`-`141`，用于 MLX 后端）。masked 掉的 token 被置为 `-inf`（llguidance/outlines 的路径类似，outlines 直接用 `logits.masked_fill_`）。

用完立刻释放：`_preprocess_logits()` 末尾 `sampling_info.grammar_mask = None`（`python/sglang/srt/model_executor/model_runner.py:1789`），注释明确写了原因（`python/sglang/srt/model_executor/model_runner.py:1784`-`1788`）——overlap 调度下 `sampling_info` 可能被 `delay_sample_func` 闭包和 `batch_record_buf` 一路带到下一轮迭代，不主动释放会造成"用了语法约束就稳定涨显存"的 VRAM 泄漏。

**这四步在什么设备上跑，是本篇一个容易被简化掉的细节**：只有第 4 步是 GPU kernel，第 1-3 步都是 CPU 工作（分配、填充）加一次 H2D 拷贝——"填 bitmask"这件事本质上是 CPU 密集型的，`## 5` 决策二展开为什么设计成这样。

### 4.4 跳跃式解码（jump-forward）：接口俱全，调用点已经消失

这是本篇取证过程中最意外的发现，分三层说清楚。

**第一层：这个机制原本要解决什么。** 源头是 2024-02-05 的 SGLang 博客"[压缩 FSM](https://lmsys.org/blog/2024-02-05-compressed-fsm/)"，代码里的引用留在 `python/sglang/srt/constrained/outlines_jump_forward.py:15`-`16`。核心想法：约束解码的正则/文法自动机里，很多状态转移是"唯一确定的"（比如 JSON 里 `"name": "` 后面几乎必然跟着字段名，`http://` 后面几乎必然跟着一段固定或高度受限的字符）——如果当前自动机状态到接下来若干个字符的路径是唯一确定的，理论上可以**跳过对应的多次前向计算，直接把这段确定的字符串当作输出吐出去**，只在路径重新变得不确定的地方恢复正常的"跑前向 → 采样 → 推进 FSM"循环。

**第二层：接口和实现确实存在，而且不是占位符。** `BaseGrammarObject` 定义了三个方法（`python/sglang/srt/constrained/base_grammar_backend.py:120`-`146`）：`try_jump_forward(tokenizer)` 返回"能不能跳、跳到哪"；`jump_forward_str_state(helper)` 把跳跃结果转成字符串和自动机的下一个状态；`jump_and_retokenize(old_output_ids, new_output_ids, next_state)` 处理**跳跃后重新分词的结果**（用模型自己的 tokenizer 把跳过去的那段字符串重新分词，得到的 token id 序列不一定等于自动机路径上"应该"产生的 token id 序列——这正是任务里点名的"tokenizer 边界"正确性风险）。

四个实现里，**xgrammar 是活的**：`try_jump_forward()`（`python/sglang/srt/constrained/xgrammar_backend.py:164`-`168`）直接调用 xgrammar 库自己的 `self.matcher.find_jump_forward_string()`；`jump_and_retokenize()`（`python/sglang/srt/constrained/xgrammar_backend.py:174`-`196`）是这四个实现里唯一认真处理了 tokenizer 边界问题的版本——用最长公共前缀找到 `old_output_ids` 和重新分词后的 `new_output_ids` 从哪个位置开始分叉（`python/sglang/srt/constrained/xgrammar_backend.py:178`-`182`），把自动机 `rollback()` 到分叉点，再把 `new_output_ids` 分叉之后的部分逐个 `accept_token()` 重新喂给自动机（`python/sglang/srt/constrained/xgrammar_backend.py:184`-`196`），任何一个 token 喂不进去就抛 `ValueError`。这是"重新分词可能产生不同 token 序列"这个风险的正确应对方式：不假设跳跃后的字符串重新分词等于原路径的 token 序列，而是老老实实拿真实分词结果去对自动机做一次回滚重放。

**llguidance 也是活的，但用了完全不同的策略绕开了这个风险**：`try_jump_forward()`（`python/sglang/srt/constrained/llguidance_backend.py:191`-`196`）调用 `self.ll_matcher.compute_ff_tokens()`，返回的直接是**已经在这个 matcher 自己词表体系下合法的 token id 列表**（`ff_tokens`），不是一段需要重新分词的字符串——所以它的 `jump_and_retokenize()`（`python/sglang/srt/constrained/llguidance_backend.py:201`-`204`）是空实现（`pass`）：**没有"重新分词"这一步，因为压根不需要**。这是这三个后端里对"跳跃解码正确性风险"处理得最干净的一个：llguidance 把"下一段确定的 token 是什么"这个问题留在 token id 空间里解决，而不是先降到字符串再升回 token id。

**outlines 名义上支持，实际上从来没有真正跑起来过**：`OutlinesGrammar.try_jump_forward()`（`python/sglang/srt/constrained/outlines_backend.py:80`-`102`）依赖 `self.jump_forward_map`，但 `OutlinesGrammarBackend._compile_regex()`（`python/sglang/srt/constrained/outlines_backend.py:145`-`158`）在构造 `OutlinesGrammar` 时**硬编码 `jump_forward_map = None`**（`python/sglang/srt/constrained/outlines_backend.py:157`），所以 `try_jump_forward()` 开头 `if not self.jump_forward_map: return None`（`python/sglang/srt/constrained/outlines_backend.py:81`-`82`）永远短路返回 `None`。全仓库唯一真正 `OutlinesJumpForwardMap(...)` 构造调用在 `python/sglang/srt/constrained/outlines_jump_forward.py:182`，位于该文件自己的 `__main__` 演示函数 `test_main()` 里，不在任何请求处理路径上。

**第三层，也是最关键的：即使 xgrammar/llguidance 的实现是活的，调度器也从来不调用它们。** 在整个 `python/sglang/srt/managers/` 目录（`scheduler.py`、`schedule_batch.py`、`io_struct.py`、各 `scheduler_components/`、`disaggregation/`）里搜索 `try_jump_forward`、`jump_forward_str_state`、`jump_and_retokenize`、`find_jump_forward_string`，**零命中**。把搜索范围扩大到整个 `python/` 目录（含前端 DSL、`docs/`、`test/`），命中的文件只有定义这三个方法的 5 个 `constrained/` 后端文件本身，以及 `test/manual/lang_frontend/test_jump_forward.py`——这是一个不在自动 CI 里跑的手工测试脚本（目录名 `manual`），而且它本身也没有直接调用这三个方法，只是通过旧版 `sgl.gen(regex=...)` 前端跑了几个回归示例，跳跃解码是否被触发完全取决于底层是否被调度器调用——而调度器不调用。

也就是说：**`BaseGrammarObject` 定义的跳跃解码接口，在当前这个 commit 的请求处理主链路里没有任何一个入口方法会被触发。** 本库浅克隆（`--depth 1`）拿不到 git 历史，无法确认这个调用点是什么时候、因为什么原因从调度器里消失的——这是**未查证**的部分，`## 7` 反直觉一会给出几个有依据支撑、但仍标注为「本库推断」的可能原因。

### 4.5 与 RadixAttention 的交互：语法状态不从缓存恢复，而是被"带着走"

任务里点名的正确性陷阱——"前缀缓存命中时语法状态怎么恢复"——在 SGLang 里答案出人意料地简单：**它不需要恢复，因为它从来没有被 RadixAttention 的树"拥有"过。**

`process_req_with_grammar()`（`## 4.1`）在请求刚到达 Scheduler、还没有做任何前缀匹配之前就已经决定了这个请求要不要编译语法、编译成什么——语法对象的生命周期完全独立于 `tree_cache.match_prefix()` 这套机制。对一个全新请求，无论它的 prompt 前缀在 radix 树里命中了多少 token，它的 `req.grammar` 永远是从状态 0 开始的一个全新 matcher（`.copy()` 出来的，参见 `## 3.3`）——因为语法约束的对象是**这个请求自己将要生成的输出**，不是被复用的 prompt KV，两者本来就是不同维度的东西：KV cache 缓存的是"输入 token 的 attention 中间结果"，语法状态是"输出 token 的自动机位置"，共享 KV 前缀完全不意味着共享生成路径。

真正需要处理的场景是**同一个请求自己被打断又恢复**：显存不足时的 `retract_decode()`（[[03-SGLang-RadixAttention与前缀缓存]] 里详细讲过 retract 机制）会释放这个请求的 KV、把它退回 `waiting_queue`，但**不会**把 `req.grammar` 这个 Python 对象清空或重建——它就是同一个对象，连着它内部已经推进到的自动机状态，一起被"冻结"在 `Req` 上等下一次被重新调度。风险在于：这个请求恢复处理时，如果代码路径又对它已经接受过的最后一个 token 重复调用了一次 `accept_token()`，自动机会因为"这个 token 在当前状态下不合法（因为状态已经前进过了）"而抛异常，进而中止请求。

这条防线就是 `current_token` 哨兵。`decode_schedule_batch_mixin.py:process_prebuilt()`（`python/sglang/srt/constrained/outlines_jump_forward.py:113`-`139`）里：

```python
if req.grammar.current_token is None:
    req.grammar.accept_token(req.output_ids[-1])
```

注释原文（`python/sglang/srt/constrained/outlines_jump_forward.py:125`-`126`）："if it is not None, then the grammar is from a retracted request, and we should not accept the token as it's already accepted"。`current_token` 在每次 `accept_token()` 里被写入（xgrammar：`python/sglang/srt/constrained/xgrammar_backend.py:95`；`ReasonerGrammarObject`：`python/sglang/srt/constrained/reasoner_grammar_backend.py:121`），充当"这个 grammar 对象上一次接受的 token 是否已经被这条代码路径消费过"的标记。`python/sglang/srt/constrained/reasoner_grammar_backend.py:113`-`120` 的注释更进一步，直接记录了这个哨兵是为了堵一个**真实发生过**的 bug："Without this, a ReasonerGrammarObject's current_token stays None forever (the inner grammar's is updated, not the wrapper's), so the guard never fires and the token is accepted twice -> 'Tokens not accepted' -> FINISH_ABORT."——也就是说，这个哨兵机制本身还在 `ReasonerGrammarObject` 包装层上被重新实现过一次（`python/sglang/srt/constrained/reasoner_grammar_backend.py:121` 的 `self.current_token = token`），因为最初的实现只更新了内层 `self.grammar.current_token`，包装对象自己的 `current_token` 永远是 `None`，导致哨兵形同虚设。

PD 分离场景下有对称的另一半：prefill 侧在把 KV 交给 decode 侧之前，如果本地已经产出了第一个 token，会先在自己这份 `req.grammar` 上调用一次 `accept_token()`（`python/sglang/srt/disaggregation/prefill.py:765`-`767`）。这两处调用点合起来说明一个更大的设计原则：**SGLang 从不假设语法状态可以从 KV/字符串内容反推出来，它选择显式地把"这个 token 有没有被语法自动机吃过"这件事记在语法对象自己身上，用一个哨兵字段而不是重新扫描历史来维持一致性。**

### 4.6 投机解码里的"多 token 一步"：不是 jump-forward，是草稿树 DFS

如果把"一步吃掉多个 token"理解成一个抽象目标，SGLang 里真正活着、每天都在跑的实现路径不是 jump-forward，而是投机解码的草稿树验证。`traverse_tree()`（`python/sglang/srt/speculative/spec_utils.py:440`-`489`）在 CPU 上 DFS 遍历草稿模型生成的候选树，对每个被目标模型接受的节点调用一次 `grammar.accept_token()`（`python/sglang/srt/speculative/spec_utils.py:480`）推进自动机，再调用 `grammar.fill_vocab_mask()`（`python/sglang/srt/speculative/spec_utils.py:483`）为这个节点生成对应的 bitmask，然后递归访问子节点——这是在一步验证里，把"语法约束"和"投机解码的树验证"编织在一起的地方：树的每一层都要检查"这个候选 token 是否同时满足草稿接受条件和语法合法性"。

这和 jump-forward 是两种不同的思路：jump-forward 的假设是"自动机路径唯一确定，不需要模型参与，跳过前向计算"；草稿树验证的假设是"仍然需要模型参与（草稿模型 + 目标模型验证），但把多个候选打包进一次前向，用树形结构摊薄验证成本"。两者都能达到"平均每次前向产出不止一个 token"的效果，但后者不依赖语法自动机本身有没有"确定性游程"，适用面更广，这也是本库推断*为什么投机解码路径持续在维护、jump-forward 路径无人问津*的一个合理解释——`## 7` 反直觉一继续展开。

### 4.7 严格推理模式下的另一条路径：token filter，不走完整 bitmask

`## 1` 提到的思考预算之外，SGLang 还有一个 `--enable-strict-thinking` 开关，管的是"推理模型在思考阶段/思考结束后，哪些 token 该被强制允许或禁止"，比如强制在思考预算用完时立刻吐出思考结束标记。这条路径**不需要一个完整的语法自动机**，只需要"允许/禁止几个具体的 token id"，所以 SGLang 没有把它接到 `fill_vocab_mask` 的自动机查询逻辑上，而是单独开了一条更轻量的 `set_token_filter` 通道：

- `BaseGrammarBackend` 默认 `is_support_token_filter` 为 `False`（`python/sglang/srt/constrained/base_grammar_backend.py:227`-`229`），`set_token_filter(...)` 默认是空操作（`python/sglang/srt/constrained/base_grammar_backend.py:231`-`235`），`init_strict_reasoning_grammar(...)` 默认返回 `None`（`python/sglang/srt/constrained/base_grammar_backend.py:237`-`239`）——也就是说这条能力默认不存在，后端要显式声明支持。
- 目前只有 `XGrammarGrammarBackend` override 了 `is_support_token_filter` 为 `True`（`python/sglang/srt/constrained/xgrammar_backend.py:245`-`247`），并实现了真正的 `set_token_filter`（`python/sglang/srt/constrained/xgrammar_backend.py:267`-`290`）：按 CUDA/非 CUDA 分别派发到 `set_token_filter_triton`（GPU）或 `set_token_filter_torch`（CPU 兜底，即 `## 2` 提到的 `python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py:27`-`64`，用 `torch.bitwise_or`/`bitwise_and` 直接在 int32 打包的 bitmask 上按位置位/清位）。
- `ReasonerGrammarObject.fill_vocab_mask()`（`python/sglang/srt/constrained/reasoner_grammar_backend.py:150`-`169`）在 `_is_thinking()` 为真时**完全跳过内层语法对象**，改成调用 `self._do_token_filter(...)`（`python/sglang/srt/constrained/reasoner_grammar_backend.py:163`-`168`，转发到 `token_filter_fn`）——思考阶段的 token 合法性判断压根不经过 JSON Schema/正则自动机，只是"思考预算够不够、是否要强制吐出结束标记"这个简单条件分支。
- `enable_strict_thinking` 打开且需要 token filter（`think_excluded_token_ids` 非空或 `max_think_tokens >= 0`）时，`ReasonerGrammarBackend.__init__` 会检查底层后端是否支持（`python/sglang/srt/constrained/reasoner_grammar_backend.py:270`-`278`），不支持就直接 `raise ValueError`；`create_grammar_backend()` 里还有一层更早的检查（`python/sglang/srt/constrained/base_grammar_backend.py:390`-`397`：xgrammar 初始化失败且开了严格推理就直接报错，不允许静默退化到不支持 token filter 的后端；`python/sglang/srt/constrained/base_grammar_backend.py:415`-`422`：`--grammar-backend none` 配 `--enable-strict-thinking` 同样直接拒绝）——这是决策六"不支持就显式报错"这条原则在配置校验阶段的又一次体现，只是这次拦在了服务启动时，比等到具体请求进来才失败更早。

## 5. 设计决策与代价

### 决策一：编译丢进线程池，但"检查是否编译完成"用有限忙等而不是纯粹的事件通知

*为什么这么设计*
- Scheduler 是单进程、单线程的（[[02-SGLang-Scheduler事件循环]]），语法编译（尤其是复杂 JSON Schema 或长 EBNF）耗时可能到几十甚至上百毫秒，同步编译会让这段时间里同一个 Scheduler 进程上其它请求的 decode 步全部停摆——`ThreadPoolExecutor`（`python/sglang/srt/constrained/base_grammar_backend.py:206`）把这部分 CPU 工作挪出主线程，是必须做的。
- 但完全异步（比如用 `asyncio` 的回调或者 `Future.add_done_callback`）需要把 Scheduler 的事件循环改造成能被后台线程唤醒的形态，而 Scheduler 现有的主循环是一个纯同步的 `while True` 轮询结构（`run_event_loop()`，见 [[02-SGLang-Scheduler事件循环]] `## 1`），插入一个跨线程回调意味着要给这个纯同步循环引入线程安全的状态写入点。`get_ready_grammar_requests()` 选择了更简单的方案：主循环每轮定批前，用一个有界的 `while ... time.sleep(...)` 循环去 `poll` 一次 `Future.done()`（`python/sglang/srt/constrained/grammar_manager.py:206`-`225`），5ms 内查完就提前 `break`，查不完就放弃这一轮，下一轮调度再查。
- 这个设计让"语法编译"这件事对 Scheduler 主循环的其它逻辑完全透明——不需要修改事件循环结构，只在一个已有的调用点（`_get_new_batch_prefill_raw()`）里插入一段轮询代码。

*不这样会怎样*
- 如果完全不轮询、只在编译完成后才把请求放回 `waiting_queue`（比如用 `add_done_callback` 直接从后台线程往 `waiting_queue` 里塞），需要给 `waiting_queue`（一个普通 Python `list`/`deque`）加锁或者改成线程安全的队列，而 Scheduler 的几乎所有其它状态（`running_batch`、`tree_cache`、`req_to_token_pool`）都假设只有主线程会碰它——引入跨线程写入会把这条"单线程状态机"的隐含假设打破，牵一发动全身。
- 如果轮询间隔设得比 5ms 大很多，编译好的请求要等更久才能重新进入调度，直接拉长这类请求的 TTFT；设得比 5ms 小很多，忙等本身占用的 CPU 时间片增多，在没有语法请求排队时这个循环也不会跑（`has_waiting_grammars()` 为假时整段代码直接跳过），但只要队列里还有一个未完成的编译，这 5ms 就是每轮调度都要付的固定税。

*什么时候可以不这样*
- 缓存命中路径本来就是同步的（`## 4.1`）——已经见过的 schema 之后的请求根本不会进 `grammar_queue`，也就不会触碰这段忙等逻辑，代价只在"新 schema 第一次出现"时才发生。
- `--grammar-backend none` 完全跳过整个编译/轮询链路。
- 可以调大 `SGLANG_GRAMMAR_POLL_INTERVAL`（环境变量）把单次忙等上限压得更低，用"语法请求 TTFT 稍微多等一两轮调度"换主循环更少的固定开销——这本质是同一枚硬币的两面，没有免费的选项。

### 决策二：bitmask 用"CPU 分配+CPU 填充+GPU 应用"三段式，而不是端到端塞进 GPU kernel

*为什么这么设计*
- "下一步哪些 token 合法"这个判断，本质上是在查询一个自动机（正则 DFA、EBNF 编译出的下推自动机、JSON Schema 转成的状态机）当前状态的出边——这是一个天然串行、依赖具体语法结构的状态查询，不是一个规整的、能直接映射成 GPU 并行原语的数值计算；xgrammar/llguidance 把这部分实现成了 C++/Rust 库，运行在 CPU 上。
- 真正适合放 GPU 的部分只有最后一步：拿到 bitmask 之后，把它按位应用到一个 `[batch, vocab_size]` 的 logits 张量上（该置 `-inf` 的位置置 `-inf`）——这是逐元素操作，天然向量化，`apply_token_bitmask_inplace_triton`（`python/sglang/kernels/ops/grammar/bitmask_ops.py:13`-`24`）把这一步单独做成 kernel。
- 中间的 H2D 拷贝用 pinned memory + `non_blocking=True`（`## 4.3` 第 1、3 步）是把"CPU 填充"和"拷贝到 GPU"两件事尽量并行化的标准手法——填充仍然是 CPU 时间，但至少不用等拷贝完成才能开始填下一批。

*不这样会怎样*
- 如果把自动机状态查询也搬到 GPU 上，需要把 xgrammar/llguidance 的整套自动机遍历逻辑重写成 GPU kernel（本质是把一个任意深度、任意分支结构的状态机遍历并行化），工程复杂度远高于收益——批内每个请求的自动机通常规模不大（相对 `vocab_size` 而言），CPU 遍历本身不是 GPU 算力能显著加速的那类问题。
- 现状的代价是：xgrammar 后端的 `fill_vocab_mask_batched` 用未并行化的 Python for 循环逐行调用（`python/sglang/srt/constrained/base_grammar_backend.py:88`-`94`），批内语法请求数越多，这一步占用的 CPU 时间越接近线性增长，在大并发、全部请求都带语法约束的场景下可能成为瓶颈——这正是 `## 6` 里 vLLM 用阈值化线程池并行这一步、llguidance 用原生并行 executor 这一步的动机所在。

*什么时候可以不这样*
- llguidance 已经示范了一种"不这样"：把填充这一步也并行化（`fill_next_token_bitmask_par`，`python/sglang/srt/constrained/llguidance_backend.py:79`-`87`），代价是需要底层库自己提供线程安全的并行遍历接口，SGLang 侧的 Python 代码没有增加复杂度，只是换了个后端。
- 语法约束请求在批内占比很低（大多数请求是普通生成）时，这一步的绝对耗时本来就小，三段式设计的"没有充分并行"缺点不会成为实际瓶颈。

### 决策三：overlap 调度默认放行语法约束，只对个别投机解码算法强制同步

*为什么这么设计*
- `ScheduleBatch.grammar_needs_sync()`（`python/sglang/srt/managers/schedule_batch.py:2247`-`2250`）只在 `self.has_grammar and not self.spec_algorithm.supports_grammar_overlap()` 时为真，`is_disable_overlap_for_batch()`（`python/sglang/srt/managers/scheduler.py:1879`-`1893`）只在这个条件**且**当前批是 decode**且**存在还未处理的上一轮结果时才强制打断 overlap 流水线（`python/sglang/srt/managers/scheduler.py:1895`-`1903` 的 `_advance_pending_grammar()` 是配套的"语法屏障"，在正常情况下把 FSM 推进这件 CPU 工作藏在目标前向计算之下）。
- `SpeculativeAlgorithm.supports_grammar_overlap()`（`python/sglang/srt/speculative/spec_info.py:137`-`142`）对 EAGLE 系、standalone、dflash 系返回 `True`，注释直接写明原因：这些算法有一个真实的 GPU 草稿阶段，语法 FSM 的 CPU 推进工作可以"藏"在这段 GPU 时间之下，不需要额外同步点；而 `NGRAM`（宿主端 n-gram 词表查找，没有 GPU 草稿阶段可藏）以及不开投机解码时的默认注册表实现（`python/sglang/srt/speculative/spec_registry.py:98`-`100`，基类默认 `False`）都没有这个"可藏身的 GPU 窗口"，所以语法推进必须同步完成才能保证下一步 bitmask 是对的。
- 换句话说，这不是"语法约束和 overlap 调度天生冲突"，而是"语法 FSM 推进这件 CPU 工作需要一段可以并发隐藏它的 GPU 时间，有就不打断，没有就必须同步"。

*不这样会怎样*
- 如果对所有开了投机解码的场景都无条件因为语法约束打断 overlap（不做算法级别的区分），NGRAM 之外那些本可以overlap 的算法（EAGLE 等）会白白损失本可以拿到的重叠收益。
- 如果反过来对所有场景都不打断（包括 NGRAM），语法 bitmask 有可能在自动机状态还没推进完的中间态上被计算出来，产生错误的合法 token 集合——这是正确性问题，不是性能问题。

*什么时候可以不这样*
- 不使用投机解码（`spec_algorithm.is_none()` 为真）时，`need_grammar_sync` 这个判据的前提条件 `not batch.spec_algorithm.is_none()` 本身就不成立，语法约束和 overlap 调度默认就能共存，不需要额外配置。
- `disable_overlap_schedule=True` 整体关掉 overlap（[[02-SGLang-Scheduler事件循环]] 决策三），语法约束自然也不需要考虑这层交互。

### 决策四：跳跃式解码保留完整接口，但调度器不再调用它

*为什么当初要做这件事*
- 2024 年（`python/sglang/srt/constrained/outlines_jump_forward.py:15`-`16` 引用的博客）约束解码普遍用逐 token 的 mask-and-sample 循环实现，每个确定性字符仍然要走一次完整的前向计算——如果自动机路径确定，理论收益很直接：省掉这些前向。

*不这样（继续调用）会怎样*
- 从当前证据只能反过来说：现状就是**不这样**——`## 4.4` 的 grep 结果表明调度器没有调用点，用户配置了任何正则/JSON Schema，无论其中有多少确定性游程，都不会获得跳跃解码带来的前向节省，纯粹是逐 token 走完整流程。这不是配置开关关闭的结果（没有找到任何 `--enable-jump-forward` 之类的旋钮），而是代码路径本身缺失调用点。

*什么时候可以恢复*
- 只是猜测性的技术判断（**本库推断**，未查证）：要重新接上这条链路，至少要解决"一步生成多个 token"和现有调度模型的几处结构性冲突——`ScheduleBatch` 的批张量（`seq_lens`、`out_cache_loc` 等）按"这一步每个请求算 1 个新 token"的假设组织；CUDA Graph 捕获的 decode 路径要求固定形状；overlap 调度的 `future_map` 按"每个请求每步占一个未来 token 位置"的语义设计（[[02-SGLang-Scheduler事件循环]] 决策三）。跳跃解码要在同一步里让某一行"凭空"多出好几个 token，天然是变长的，和这几处的定长假设都冲突。这不代表不可能做，只是解释了为什么"接口都写好了"和"重新接上调度器"之间还有相当的工程距离。
- 从代码本身看，`xgrammar`/`llguidance` 两个后端的实现细节仍然完整、没有被标记废弃（没有 `DeprecationWarning`，也没有留下待办类注释），意味着这更像是"随着调度架构演进而自然掉线"，而不是"决定不再需要它"的显式产品决策——但这一点同样是推断，浅克隆看不到相关的 PR 讨论。

### 决策五：语法状态用请求自带的哨兵字段维持一致性，而不是从 KV/字符串反推

*为什么这么设计*
- `Req.grammar` 是一个附着在请求对象上、贯穿其整个生命周期（含 retract、PD 传输）的 Python 对象，天然比"重新从已生成文本反推自动机应该处于什么状态"更直接、更不容易出错——重新反推需要对整个语法自动机做一次"回放"（从状态 0 开始，把所有已生成 token 重新喂一遍），而 `## 4.5` 描述的 `current_token` 哨兵只需要一个 `is None` 判断。
- 这个设计把"KV cache 的生命周期"和"语法状态的生命周期"彻底解耦：RadixAttention 的树只管 KV block 的复用和淘汰，完全不需要知道某个 KV 块背后有没有语法约束——这是关注点分离带来的简单性。

*不这样会怎样*
- 如果语法状态被设计成可以从 radix 树的内容反推（比如把每个 token 在自动机里的状态也存进树节点），RadixAttention 的树结构会被迫携带语法子系统的私有状态，任何不使用语法约束的请求路径也要为这份额外簿记买单，而且两个子系统的淘汰/失效策略要保持同步（KV 块被逐出时语法状态怎么办？两个请求共享同一段 KV 前缀但用不同语法约束怎么办？）——`## 4.5` 已经说明了为什么语法状态和 prompt 前缀本来就不共享同一份数据，这样做是给一个不存在的问题设计方案。

*什么时候可以不这样*
- 这条设计几乎没有"可以不这样"的场景——只要语法约束按请求粒度生效（而不是按 KV 块粒度生效），状态就必须挂在请求对象上，不存在把它挂到 KV 树上更简单的替代方案。唯一的例外是**不使用**语法约束（`--grammar-backend none`）时，这整套哨兵机制根本不会被触碰。

### 决策六：不支持的 schema 显式中止请求，而不是静默退化成无约束生成

*为什么这么设计*
- `dispatch_*` 系列方法在不支持的类型上统一走 `_not_supported()`（`python/sglang/srt/constrained/base_grammar_backend.py:219`-`221`），产出 `InvalidGrammarObject` 而不是 `None` 或者直接跳过约束——`process_req_with_grammar()`（`python/sglang/srt/constrained/grammar_manager.py:131`-`182`）和 `get_ready_grammar_requests()`（`python/sglang/srt/constrained/grammar_manager.py:184`-`311`）两条消费路径都把它当"确定失败"处理，走 `req.set_finish_with_abort(error_msg)`。
- 静默退化（用户要 EBNF 约束，后端不支持就当没有约束一样生成）是一类特别危险的失败模式：**输出格式看起来仍然正常（因为没有约束地生成本来也能生成"看起来正常"的文本），但没有任何硬性保证**，下游如果直接 `json.loads()` 解析输出会在生产环境里随机失败，而且很难把这个失败关联回"其实是当时选错了 grammar backend"。显式中止请求虽然对用户体验更硬，但错误发生的时间点和错误信息都是确定的。

*不这样会怎样*
- 如果选择静默退化，用户很容易在换后端（比如从 xgrammar 切到 outlines）之后才发现自己一直依赖的 EBNF 约束早就没生效——`## 6`/`## 7` 会提到 llguidance 对新版 `structural_tag` 格式的处理方式（`assert` 直接抛异常）恰好走的是相反的极端：完全不打算优雅处理，直接让整个 dispatch 失败，效果上和 SGLang 现在选的"显式 abort"殊途同归，但是更粗暴（`assert` 在生产环境如果被 `-O` 优化掉会被跳过，是否发生在这条路径上**未查证**）。

*什么时候可以不这样*
- 这条设计基本没有"可以不这样"的合理场景——静默降级在约束解码这个领域几乎总是错误的取舍，因为它把"有没有约束"这个二元问题变成了一个只有仔细读日志才能发现的隐藏状态。唯一的软化空间是提前校验：如果服务端能在请求接收阶段（而不是编译阶段）就用一份"这个后端支持哪些类型"的静态表提前拒绝，可以把失败时间点从"编译完成后"提前到"请求刚到达时"，减少无谓的排队和资源占用——但这已经是"更早报错"而不是"不报错"。

## 6. 同位对照：vLLM 的 `vllm/v1/structured_output/`

一句话概括四个维度的差异：

| 维度 | SGLang | vLLM |
|---|---|---|
| 内置后端数量 | 3 个（xgrammar / outlines / llguidance） | 4 个（xgrammar / guidance / outlines / lm-format-enforcer） |
| 编译异步机制 | 一个 `ThreadPoolExecutor` 管编译；填 bitmask 内嵌在采样路径，未额外并行 | 两个 `ThreadPoolExecutor`：一个管编译，一个专门管"填 bitmask"，后者有阈值化并行 |
| `accept` 接口粒度 | `accept_token(token: int)`，多 token 场景靠调用方在 Python 循环里逐个调 | `accept_tokens(request_id, tokens: list[int])`，批量语义下推到接口层 |
| jump-forward | 接口存在、部分实现是活的，但调度器不调用（`## 4.4`） | 接口层压根没有这个方法，只在源码注释里提了一句指向底层库的 API |

#### 后端选择与"单一全局后端"

vLLM 的 `StructuredOutputManager.grammar_init()`（`vllm:vllm/v1/structured_output/__init__.py:115`-`176`）按 `request.sampling_params.structured_outputs._backend` 惰性实例化 `self.backend`，四选一：`XgrammarBackend`（`vllm:vllm/v1/structured_output/__init__.py:134`-`139`）、`GuidanceBackend`（`vllm:vllm/v1/structured_output/__init__.py:140`-`145`）、`OutlinesBackend`（`vllm:vllm/v1/structured_output/__init__.py:146`-`153`）、`LMFormatEnforcerBackend`（`vllm:vllm/v1/structured_output/__init__.py:154`-`163`）。源码注释直接点出一条限制（`vllm:vllm/v1/structured_output/__init__.py:127`-`128`）："We only support a single backend. We do NOT support different backends on a per-request basis in V1"——这和 SGLang 是一样的：`GrammarManager.grammar_backend`（`python/sglang/srt/constrained/grammar_manager.py:32`-`38`）同样是 Scheduler 进程启动时按 `--grammar-backend` 一次性创建的单例，不支持同一进程内不同请求用不同后端。两个引擎在这一点上做出了相同的取舍，值得记一笔的是**为什么两边都这么选**（**本库推断**，依据：两边都把"编译后的语法对象"设计成可以 `.copy()` 复用的缓存条目，如果后端可以按请求切换，缓存 key 至少要多一维"用哪个后端编译的"，且不同后端的自动机格式互不兼容，缓存复用的收益会打折扣）——这是"缓存友好"这个共同设计目标自然导出的限制，不是各自独立踩中同一个坑。

#### 编译异步机制：都异步，但 vLLM 把"填 bitmask"也异步化了

vLLM 的 `StructuredOutputManager.__init__` 里有两个线程池：`self.executor`（`vllm:vllm/v1/structured_output/__init__.py:77`-`78`）专门管语法编译，和 SGLang 的 `BaseGrammarBackend.executor` 角色一致；额外多出的 `self.executor_for_fillmask`（`vllm:vllm/v1/structured_output/__init__.py:61`-`69`）专门用来并行"填 bitmask"这一步——但只在 `max_num_seqs > self.fill_bitmask_parallel_threshold`（硬编码 `128`，`vllm:vllm/v1/structured_output/__init__.py:62`）时才创建这个线程池，填充时按 `fill_bitmask_parallel_batch_size = 16`（`vllm:vllm/v1/structured_output/__init__.py:64`）分块提交给它（提交点在 `vllm:vllm/v1/structured_output/__init__.py:218`）。

SGLang 没有对应的"阈值化并行填充"机制——`## 4.3`/`## 5` 决策二已经说明，xgrammar 后端的 `fill_vocab_mask_batched` 是未并行化的 Python for 循环，唯一并行化的路径是 llguidance 后端自带的原生 executor（对所有批大小无条件并行，不是阈值触发）。两边对同一个"CPU 密集的逐行填充"问题给出了不同的应对：vLLM 选择"批大小超过阈值才值得付线程池调度开销"的显式阈值判断；SGLang 把这个决定权交给后端库自己有没有实现并行接口。

#### 接口粒度：单 token vs 批量 token

vLLM 的 `StructuredOutputGrammar.accept_tokens(request_id, tokens: list[int])`（抽象方法定义在 `vllm:vllm/v1/structured_output/backend_types.py:33`-`44`）从接口层面就是批量的——`XgrammarGrammar.accept_tokens()`（`vllm:vllm/v1/structured_output/backend_xgrammar.py:157`-`169` 起）内部对 `tokens` 做 `for token in tokens: self.matcher.accept_token(token)` 循环，但"这是一批"这件事在接口签名里就写明了，调用方（投机解码验证等场景）不需要自己写循环。

SGLang 的 `BaseGrammarObject.accept_token(token: int)`（`python/sglang/srt/constrained/base_grammar_backend.py:68`-`72`）只接受单个 token，`## 4.6` 提到的投机解码路径（`python/sglang/srt/speculative/spec_utils.py:480`）和普通多 token 提交路径（`python/sglang/srt/managers/scheduler_components/batch_result_processor.py:762`-`789` 的 `_accept_grammar_tokens()`）都是在**调用方**自己写 for 循环、逐个调用 `accept_token()`，接口本身没有"这是一批"的概念。两种设计能做的事情等价，区别只在于"批量"这个语义放在接口层还是调用方——vLLM 把它放进接口意味着未来所有后端都必须支持批量语义；SGLang 把它留在调用方意味着接口更小，但每个新增的"多 token 一次性推进"场景都要在调用方重新写一遍循环（目前至少有两处几乎相同的循环：`python/sglang/srt/speculative/spec_utils.py:477`-`480` 和 `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:778`-`782`）。

#### jump-forward：一个"造了又弃用"，一个"从没造过"

`## 4.4` 已经确认 SGLang 四个后端都实现了跳跃解码的完整接口，只是调度器不再调用。vLLM 这边情况更彻底：`StructuredOutputGrammar` 这个抽象基类一共只有 6 个抽象方法——`accept_tokens`、`validate_tokens`、`rollback`、`fill_bitmask`、`is_terminated`、`reset`（`vllm:vllm/v1/structured_output/backend_types.py:33`-`91`），**没有任何一个和跳跃解码相关**。唯一的痕迹是 `XgrammarGrammar` 这个 dataclass 头部的一段纯注释（`vllm:vllm/v1/structured_output/backend_xgrammar.py:146`-`147`）：

```python
# https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
# for jump-forward decoding
```

这行注释只是指出"xgrammar 库本身有这个能力"，没有配套的方法实现、没有调用点、也没有被写进抽象接口——是纯粹的文档性提示，比 SGLang 那种"造了完整接口、写了正确性处理逻辑、最后没人调用"的状态更接近"从一开始就没打算做"。

两个引擎现在都不在生产路径上跑跳跃解码，但走到这个结果的历史路径不同：vLLM 大概率是**评估过这个技术方向、留了个指路的注释，但没有投入实现**（**本库推断**，依据：只有注释、没有任何接口残留）；SGLang 是**完整实现过，现在处于调用链路缺失的状态**（`## 4.4` 的直接证据）。这个差异本身提示一个更大的判断：跳跃解码在两个当前最主流的开源推理引擎里都不是一个被积极维护的优化方向，读者如果在评估"要不要为跳跃解码去写自定义语法后端"，这个事实值得纳入考量。

这也回应了 `## 5` 决策四提出的推断——两个引擎的调度器都朝着"批内每个请求每步产出定长 token 数"的方向演进（SGLang 有 CUDA Graph 捕获和 overlap 调度的 `future_map`，vLLM V1 的 `schedule()` 同样按 `num_computed_tokens` 逐步推进，参见 [[02-SGLang-Scheduler事件循环]] `## 6`），这类调度模型天然更容易和"多 token 一步但来自可预测的草稿树"（投机解码）而不是"多 token 一步但长度不可预知"（jump-forward）共存——后者需要在批张量已经按定长分配好之后，再对某一行插入若干个额外 token，这和两个引擎当前调度层的核心假设都不太兼容。

（对照使用的 vLLM 版本：`vllm` @ `7ca49fbe`，2026-08-22，仅作同位参考，不是本篇取证基准；vLLM 结构化输出的完整解剖见 [[11-vLLM-结构化输出]]。）

## 7. 踩坑与反直觉

**反直觉一：跳跃式解码不是"性能优化默认关闭"，而是调度器里根本没有调用它的代码。**
容易望文生义地以为 SGLang 的"约束解码快"部分归功于跳跃解码这个 2024 年博客里介绍的技术，但 `## 4.4` 的 grep 证据表明：`python/sglang/srt/managers/` 整个目录找不到 `try_jump_forward`/`jump_forward_str_state`/`jump_and_retokenize`/`find_jump_forward_string` 的任何调用；四个后端里 xgrammar 和 llguidance 的实现是"活的"（真实包装了底层库对应的 API），outlines 的实现是"双重死代码"——`jump_forward_map` 在构造时被硬编码成 `None`（`python/sglang/srt/constrained/outlines_backend.py:157`），即使调度器调用了 `try_jump_forward()` 也会立即因为这个短路返回 `None`。当前的约束解码性能收益来自 bitmask 机制本身（xgrammar 编译出高效自动机 + GPU 上并行应用 bitmask），而不是跳过前向计算。

**反直觉二："语法编译不阻塞主循环"这句话只对了一半。**
编译本身确实丢进了 `ThreadPoolExecutor`（`python/sglang/srt/constrained/base_grammar_backend.py:206`），不会同步卡住 Scheduler；但 `## 4.1` 展示的 `get_ready_grammar_requests()` 里那段 `while ... time.sleep(...)` 忙等循环（`python/sglang/srt/constrained/grammar_manager.py:206`-`225`），只要 `grammar_queue` 非空，每一轮调度前最多会原地等 `SGLANG_GRAMMAR_POLL_INTERVAL`（默认 5ms）。这段等待不影响吞吐上限（因为它有硬上限且命中即提前退出），但会给"正在等语法编译完成的请求所在的这个 Scheduler 进程"的每轮调度周期性地叠加最多 5ms 的固定延迟，直到编译完成或超时。

**反直觉三：语法约束和 overlap 调度默认并不冲突，只有特定投机解码算法组合才会强制同步。**
容易联想"语法约束需要精确的逐 token 状态，overlap 调度是异步重叠的，两者听起来天然矛盾"，但 `## 5` 决策三说明真实情况更精细：不开投机解码时两者可以自由共存（`grammar_needs_sync()` 的前提条件 `not spec_algorithm.is_none()` 不成立）；开了投机解码之后，只有 `NGRAM`（宿主端草稿，没有 GPU 阶段可以隐藏 FSM 推进的 CPU 开销）会触发强制同步，EAGLE/standalone/dflash 系恰恰因为有 GPU 草稿阶段可以"藏"住这部分 CPU 工作，被允许留在 overlap 路径里。

**反直觉四：`ReasonerGrammarObject` 曾经复现过一次"哨兵字段没跟着包装层走"的 bug，这条防线目前是两层实现。**
`## 4.5` 引用的 `python/sglang/srt/constrained/reasoner_grammar_backend.py:113`-`121` 注释直接写明历史：最初只有内层语法对象（比如 `XGrammarGrammar`）在 `accept_token()` 里更新自己的 `current_token`，包装它的 `ReasonerGrammarObject`（用于"思考模式下语法暂不生效"的场景）没有同步更新自己的 `current_token`，导致这个字段在包装对象上永远是 `None`——PD 分离/retract 场景下的哨兵判断 `if req.grammar.current_token is None` 因此永远为真，重复调用 `accept_token()`，报出 "Tokens not accepted" 后中止请求。修复方式是让 `ReasonerGrammarObject.accept_token()`（`python/sglang/srt/constrained/reasoner_grammar_backend.py:113`-`124`）自己也写一遍 `self.current_token = token`（`python/sglang/srt/constrained/reasoner_grammar_backend.py:121`），这是一个"包装类必须显式转发被包装对象状态"的具体教训——`BaseGrammarObject` 的多数方法（`fill_vocab_mask`、`allocate_vocab_mask` 等）在 `ReasonerGrammarObject` 里都能看到类似的显式转发模式（`python/sglang/srt/constrained/reasoner_grammar_backend.py:170`-`191`），说明这类包装类每加一个字段/方法都要记得同步一次，容易漏。

**反直觉五：llguidance 后端对新版 `structural_tag` 格式的处理是"直接失败"，不是"优雅降级到旧版格式"。**
`python/sglang/srt/constrained/llguidance_backend.py:283`-`310` 的 `dispatch_structural_tag()` 里，`assert is_legacy_structural_tag(structural_tag)`（`python/sglang/srt/constrained/llguidance_backend.py:286`）意味着如果用户传入的是新版 `format` 写法的 structural_tag（xgrammar 支持这两种写法，`## 3.4`），llguidance 后端会在这一行直接抛 `AssertionError`，被外层 `except Exception`（`python/sglang/srt/constrained/llguidance_backend.py:308`）捕获后转成 `InvalidGrammarObject`（走 `## 4.1`/决策六描述的显式 abort 路径，不是静默降级）——但用 `assert` 而不是显式 `raise ValueError(...)` 来做这层校验，意味着**理论上**如果运行环境用 `python -O` 启动会跳过所有 `assert` 语句，届时这行校验会被整体跳过，后续代码会尝试按 legacy 格式的字段名（`structure["begin"]`/`structure["schema"]`/`structure["end"]`）去解析一个实际是新版格式的字典，大概率抛 `KeyError` 而不是给出清晰的"格式不匹配"错误信息——这条链路本篇标注为**未查证**：没有找到 SGLang 是否在生产部署脚本里显式禁止 `-O` 模式的证据，但代码依赖 `assert` 做输入校验这件事本身在任何 Python 项目里都是一个值得注意的模式。

## 8. 可改进点

以下判断标注为「本库推断」，未在上游 issue/PR 中核实，仅基于代码读到的行为提出、供讨论：

1. **要么删掉跳跃解码的死接口，要么把调用点接回去，两者都好过现在的中间状态。**
   `## 4.4`/反直觉一描述的现状——四个后端里三个实现了完整接口、一个双重死代码、调度器完全不调用——对后续维护者是一个陷阱：新来的贡献者读到 `try_jump_forward()` 的完整正确性处理逻辑（尤其是 xgrammar 那段 `jump_and_retokenize()` 的回滚重放），很容易误以为这是一个正在生效的优化路径，进而在它之上继续叠加功能或者当作参考实现复制到别处。风险：如果要接回调度器，需要先解决 `## 5` 决策四提到的变长 token/CUDA Graph/`future_map` 定长假设冲突，工作量不小；如果选择删除，至少要先确认没有下游（比如 SGLang 的某个 fork 或者外部项目）依赖这几个公开方法。

2. **xgrammar 后端的 bitmask 填充可以参考 vLLM 的阈值化并行方案。**
   `## 6` 已经指出 vLLM 用 `fill_bitmask_parallel_threshold=128` 决定是否把填充工作分给一个专用线程池，SGLang 的 `python/sglang/srt/constrained/base_grammar_backend.py:88`-`94` 目前是无条件的单线程 for 循环。给 xgrammar 后端补一条类似的阈值判断，在"批内语法约束请求数很多"的场景下可能有实际收益。代价：需要引入额外的线程池管理和阈值调参，小批量场景下线程调度本身的开销可能反而不划算——这正是 vLLM 设置阈值而不是无条件并行的原因，照搬时不能省掉这层判断。

3. **语法编译缓存的 key 没有做任何归一化，语义等价的 schema 会被重复编译。**
   `## 3.3` 提到 `BaseGrammarBackend.cache` 的 key 就是原始字符串 `(key_type, key_string)`，两个字段顺序不同但语义完全等价的 JSON Schema（比如 `{"a": ..., "b": ...}` 和 `{"b": ..., "a": ...}`）会被当成两个不同的缓存条目，各自触发一次完整编译。在 key 生成前对 JSON 类型的 schema 做一次 `json.dumps(..., sort_keys=True)` 归一化，可以提高缓存命中率。代价：归一化本身有 CPU 开销（虽然远小于编译），而且需要确认这个开销不会抵消掉缓存命中带来的收益——大 schema 场景下应该是净赚。

4. **`GrammarStats` 里的字段目前找不到导出到 Prometheus/HTTP 响应的路径。**
   `## 3.2` 已经说明 `compilation_time`/`is_cache_hit`/`tree_traversal_time` 等字段被赋值但本篇没有找到消费点。如果这些指标已经有其它导出路径，本条不成立（**未查证**部分）；如果确实没有，接入现有的可观测性体系（SGLang 本身有 Prometheus 指标导出，这里未展开）能让"语法编译耗时是不是当前部署的瓶颈"这类问题从"读源码猜测"变成"看 dashboard"。

5. **llguidance 后端可以把 `assert is_legacy_structural_tag(...)`（`python/sglang/srt/constrained/llguidance_backend.py:286`）换成显式的 `raise ValueError`。**
   `## 7` 反直觉五已经说明用 `assert` 做输入校验在 `-O` 模式下存在理论上被跳过的风险，即使这个风险在当前部署方式下从未真正触发，把校验换成不受解释器优化开关影响的显式异常也是零成本的加固——同时把错误信息从默认的 `AssertionError`（不带任何上下文）换成"llguidance 后端不支持新版 structural_tag format 写法，请改用 xgrammar 或转换成 legacy 格式"这类可操作的提示，能省下用户一次去读 SGLang 源码才能定位问题的时间。

## 9. 自测题与延伸阅读

**自测题**

1. `get_cached_or_future_value()` 在缓存命中和未命中两条路径上，哪一条是同步返回，哪一条会返回一个 `Future`？分别对应源码哪几行？
2. `SGLANG_GRAMMAR_POLL_INTERVAL` 默认是多少？它控制的是"语法编译耗时"还是"检查编译是否完成"这一步的开销上限？两者有什么区别？
3. bitmask 从分配到应用到 logits 一共四步，哪一步是纯 GPU kernel 工作？其余几步各自发生在什么设备上，为什么第 1 步要用 pinned memory？
4. xgrammar 和 llguidance 两个后端的 `try_jump_forward()` 分别包装了底层库的哪个 API？它们对"跳跃后重新分词可能产生不同 token 序列"这个正确性风险分别是怎么处理的？
5. 在 `python/sglang/srt/managers/` 目录下搜索 `try_jump_forward` 会得到什么结果？这个结果对"SGLang 是否真的在用跳跃解码"这个问题意味着什么？
6. `current_token` 这个哨兵字段是为了防止哪种场景下的重复 `accept_token()` 调用？`reasoner_grammar_backend.py` 里记录的历史 bug 具体是什么，修复方式是什么？
7. `SpeculativeAlgorithm.supports_grammar_overlap()` 对哪些投机解码算法返回 `True`？NGRAM 为什么返回 `False`？这个区分依据是什么？
8. 一个请求的 JSON Schema 里用了 EBNF 约束但当前配置的 `--grammar-backend` 是 `outlines`，会发生什么？是静默生成不受约束的输出，还是会看到明确的错误？
9. vLLM 的 `StructuredOutputGrammar` 抽象基类一共有几个抽象方法？其中有没有跟跳跃解码相关的？这和 SGLang 的对应接口相比说明了什么？
10. `--enable-strict-thinking` 打开、`--grammar-backend` 是 `outlines` 时会发生什么？这条校验发生在服务启动阶段还是请求处理阶段？和"不支持的 schema 类型会怎样"这条失败模式相比，哪个失败得更早？

**延伸阅读**

- [[02-SGLang-Scheduler事件循环]] —— 理解 `## 4.1`/`## 5` 决策一/三里"忙等"和"overlap 调度"发生在哪个更大的循环结构里，本篇很多时序细节的背景。
- [[03-SGLang-RadixAttention与前缀缓存]] —— 理解 `## 4.5`/决策五"语法状态为什么不需要从前缀缓存恢复"，需要先知道 RadixAttention 的树实际缓存的是什么、retract 时具体丢弃了什么。
- [[11-vLLM-结构化输出]] —— `## 6` 同位对照的完整展开，vLLM 侧的四个后端、`StructuredOutputManager` 的完整调度集成细节本篇只挑了跟本文对照相关的部分。
