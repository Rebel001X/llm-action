# Tokasaurus

> **本篇取证基准**：`tokasaurus` @ `72689052`（2025-08-19）
> **一句话**：13,219 行验证引擎最小骨架

**速查：关键问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| 一个推理引擎最少要拆成几个进程/模块？ | `## 1`、`## 2`（本篇题眼） |
| "调度器只有 684 行"这个数字为什么会误导人？ | `## 0`、`## 6` |
| Hydragen 到底是什么、和前缀缓存/RadixAttention 什么关系？ | `## 5` 决策 4、`## 6` |
| 它为什么敢用 `fullgraph=True` 整图编译？ | `## 5` 决策 7、`## 6` |
| "OpenAI 兼容"这四个字兼容到什么程度？ | `## 5` 决策 5、`## 7` |
| 它到底缺哪些生产级引擎该有的功能？ | `## 8`、`## 5` 决策 5/6 |

## 0. 结论先行

- **它是本库 12 个对象里最小的一个，也是唯一一个纯 Python、零自研 kernel 的引擎。** `_lab/repo_stats.json` 记录的总行数是 13,219，其中 Python 产品码 11,405、测试码 1,251（测试/产品比 0.11），CUDA/Triton/Rust/C/C++ 文件数全部是 **0**。它不是"还没来得及写 kernel"，而是把所有性能敏感算子外包给了 `flashinfer-python==0.2.0.post2`（`pyproject.toml:33`）——这是本篇要讲的第一个反直觉点：一个号称高吞吐的推理引擎,自己一行 CUDA/Triton 都没写。
- **一个推理引擎最少需要几个零件，这个仓库给了一份可以直接数出来的答案：三个进程。** `tokasaurus/server/`（HTTP 层,1,338 行）、`tokasaurus/manager/`（CPU 侧编排/调度/KV 管理,3,823 行）、`tokasaurus/model/`（GPU 侧前向执行,3,698 行）,三者用裸的 `multiprocessing.Queue` 直接串起来,不经过任何 RPC 框架。README 原话（`README.md:149`,文档所述）："Tokasaurus has three major components"。见 `## 1`、`## 2`。
- **"调度器只有 684 行"这句话本身会误导人。** `tokasaurus/manager/scheduler.py` 确实只有 684 行,但它只是调度**算法**（一段纯函数式的"未来 KV 用量模拟"数学）,真正完成一次调度决策还要靠 `manager/allocator.py`（431 行,KV 块分配+前缀树）和 `manager/manager.py`（1,064 行,把算法结果落地成状态变更+提交给模型）协同——三者相加 2,179 行,仍然只是 vLLM 对应部分（`vllm/v1/core/sched/` 6 个文件共 4,012 行,外加 `block_pool.py` 831 行、`kv_cache_manager.py` 887 行,合计 5,730 行）的三分之一多一点。差在哪、哪些是必要复杂度，见 `## 6`。
- **Hydragen 省的是访存效率，不是显存容量。** 它把共享同一段前缀的多条解码序列的 query 打包成一次"无因果掩码的批量 prefill"，只读一遍共享前缀的 KV（`tokasaurus/model/attention_utils.py:156`），再用在线 softmax 合并（`cascade.merge_state`）拼回各自独有后缀的 decode 结果。这和 vLLM 的块哈希前缀缓存、SGLang 的 RadixAttention（省的是"要不要重算/要不要多存一份 KV"）是不同维度的优化，三者可以叠加，不是互相替代。详见 `## 5` 决策 4、`## 6`。
- **协议层写着"OpenAI 兼容"，但 `validate_args` 主动拒绝了大半个采样面。** `tokasaurus/server/utils.py:503`-`531` 显式拒绝 `stream=True`、`top_p≠1.0`、`frequency_penalty≠0`、`presence_penalty≠0`、`logit_bias≠None`；`response_format` 字段虽然存在于 pydantic 模型（`tokasaurus/server/types.py:82`），但全仓库再没有第二处代码读取它。这不是遗漏，是为它的目标工作点（`tokasaurus/benchmarks/monkeys_*.py` 里 n=512~1024 的重复采样/best-of-k）刻意收窄的结果。见 `## 5` 决策 5、`## 7`。
- **这是本库 12 个对象里最不活跃的一个（客观事实，非评价）。** 取证快照日是 2026-08-22，`_lab/out/repo_stats.json` 记录 Tokasaurus 唯一可见的 clone 提交日期是 2025-08-19（浅克隆只能看到这一个提交），相差 12 个月。README/`CLAUDE.md` 均未提及任何继任项目，`pyproject.toml` 里 `classifiers` 自报 `"Development Status :: 3 - Alpha"`（`pyproject.toml:13`）。有没有后继项目、是否已停止维护——**未查证**，本篇只陈述这几条可核实的事实，不下"已废弃"的判断。

## 1. 它在系统里的位置

Tokasaurus 由 Stanford Scaling Intelligence Lab 开发（作者列表见 `README.md` 末尾的 bibtex 引用块：Jordan Juravsky、Ayush Chakravarthy、Ryan Ehrlich、Sabri Eyuboglu、Bradley Brown、Joseph Shetaye、Christopher Ré、Azalia Mirhoseini——文档所述），定位不是"vLLM/SGLang 的又一个替代品"，而是**面向一个被主流引擎不太在意的工作点**：README 原话（`README.md:24`，文档所述）"Very low CPU overhead (important for small models/fast GPUs)"——当模型小、GPU 快到一步 forward 只要几毫秒时，CPU 侧的调度/序列化/Python 解释开销会变成新的瓶颈，这和"模型大、一步 forward 要几十上百毫秒"的主流服务场景（多轮对话、RAG）是完全不同的优化方向。

这个定位在代码里能找到具体证据：`tokasaurus/benchmarks/` 目录下的 `monkeys_gsm8k.py`、`monkeys_math500.py`、`monkeys_chat.py`（"monkeys"对应同实验室论文 *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*，作者 Bradley Brown 与本项目重合，源码为证但论文本身**未查证**是否为同一批人所写）实现的都是"对同一个 prompt 采样 `n=512`（`tokasaurus/benchmarks/monkeys_gsm8k.py:25`）甚至更多份，再算 pass@k"这种重复采样评测。这正是"大批量、小模型、高吞吐"工作点的典型代表：RL rollout、best-of-k 拒绝采样、蒙特卡洛搜索这类场景都需要对同一个/相似的 prompt 打出成百上千份样本，且不需要流式体验、不需要精细的采样后处理——只需要"改温度、批量跑、等结果"。HTTP 层专门给这种用法开了一条非 OpenAI 标准的口子：`/custom/synchronous-batch-completions`（`tokasaurus/server/endpoints.py:242`），一次提交一批 `ChatCompletionRequest`，`asyncio.gather` 等全部跑完再一次性返回。

在这个工作点上，主流引擎"哪里不够好"，源码层面没有直接证据（这属于对比其他引擎的判断，本篇不做，留给 `## 6`／`04-横向对比/`），但可以从 Tokasaurus 自己的设计取舍反推它认为哪里重要：**CPU 侧编排延迟**（`## 5` 决策 3 的多步预调度）、**GPU 端到手就不停**（README 原话 "the manager works to ensure that there are multiple items in the model input queue, so that the model can always be running forward passes"，`README.md:153`，文档所述）、**KV 显存利用率**（`## 5` 决策 3 的完成长度预测），三者都是"重复采样/rollout"场景里格外敏感、但对话场景里不那么突出的指标。

README 用一整节（"Managing GPU Memory with KV Cache Limits and Concurrency Controls"，`README.md:112`-`122`，文档所述）讲显存预算怎么在三个池子间取舍：`kv_cache_num_tokens` 决定 KV 缓存能装多少 token，`max_tokens_per_forward`/`max_seqs_per_forward` 联合决定每步前向的激活显存上限——原话特别点出"Prefill tokens don't run through the LM head... so they take less activation memory"（`README.md:116`，文档所述），也就是说 `max_seqs_per_forward` 实际限制的是**能同时跑语言模型头的 token 数**，这个数偏小会不成比例地压低激活显存占用。这三个旋钮要联合调、不能只加大 KV 缓存而不同步加大并发控制（否则装得下的序列数受限于并发上限，KV 缓存再大也用不满）——这是"把批大小做到最大"这个单一目标在配置层面留下的具体印记，对话服务型引擎的调优文档很少会用整节篇幅只讲这一件事。

"最小骨架"这个题眼不只体现在三进程拆分上，也体现在"加一个新模型架构要花多少代码"这件事上：`Qwen2ForCausalLM`（`tokasaurus/model/qwen.py:23`）整个文件只有 **43 行**——全部通过继承 `LlamaAttention`/`LlamaBlock`/`LlamaModel`/`LlamaForCausalLM`（`tokasaurus/model/qwen.py:3`-`8`）实现，真正新写的代码只有一个类属性 `qkv_bias: bool = True`（`tokasaurus/model/qwen.py:12`）和一个把 QKV bias 参数加进张量并行映射表的方法覆写（`make_tp_map()`，`tokasaurus/model/qwen.py:27`-`43`）。`Qwen3ForCausalLM`（`tokasaurus/model/qwen3.py:145`）因为要多支持 QK-Norm，代码量涨到 165 行，但依然是同一套继承结构（`Qwen3Attention(LlamaAttention)`，`tokasaurus/model/qwen3.py:32`）。`contributing/models.md`（文档所述）把这套"继承基类、只覆写差异点"的模式写成了官方的接入新模型指南——这是"骨架够小"在另一个维度上的体现：不仅运行时的进程拓扑最小，连"扩展一个新模型"这个开发者动作的成本也被压到了几十行代码。

测试规模（1,251 行，占 11.9%）本身也值得看一眼分布，而不只是看比例。`tests/` 下 6 个文件里，`test_block_allocator.py`（`prefix_match`/`free_and_update` 的纯逻辑测试，如 `test_basic_caching()`，`tests/test_block_allocator.py:41`）和 `test_scheduler.py`（`calc_block_usage_over_time`/`try_onboarding_seqs` 的数值模拟测试，`test_calc_block_usage_over_time()`，`tests/test_scheduler.py:100`）是不需要 GPU、不需要下载模型的纯单元测试；但 `test_basic.py`、`test_bumping.py`、`test_topk.py` 里的用例（比如 `test_bumping()`，`tests/test_bumping.py:36`）全部通过 `client: OpenAI` fixture 启动一个真实的三进程服务、下载一个真实的小模型（默认 `meta-llama/Llama-3.2-1B-Instruct`，见 `tests/test_bumping.py:11`）再发真实 HTTP 请求验证——是端到端集成测试，不是 mock 出来的单测。这个仓库的快照里也**没有** `.github/workflows/` 之类的 CI 配置文件（源码为证，浅克隆下 `find` 不到任何 `.yml`/`.yaml`）——测试写得不少，但这批需要 GPU+真实权重的集成测试有没有在什么地方被自动跑过、多久跑一次，**未查证**。

## 2. 代码地图（文件 → 职责，带行号）

按"HTTP 层 → CPU 编排层 → GPU 执行层 → 跨进程胶水"四层组织：

| 文件:行 | 职责 |
|---|---|
| `tokasaurus/entry.py:114` | `start()`——唯一的进程启动入口：拉起除本进程角色外的全部子进程，本进程内联跑 `config.local_proc_name` 对应的那一个 |
| `tokasaurus/entry.py:21` | `make_engine()`——为一个 dp_rank 建一套 server↔manager↔model 的 4 条 `mp.Queue`（server→manager、manager→server、manager→model、model→manager） |
| `tokasaurus/common_types.py:52` | `class ServerConfig(pydra.Config)`——全局唯一配置源，49 个字段，覆盖模型/并行度/调度/Hydragen/CUDA Graph 全部旋钮 |
| `tokasaurus/common_types.py:37` | `class Engine`——只包一组队列的哑对象，佐证"三进程+队列"骨架里没有第四种抽象 |
| `tokasaurus/server/endpoints.py:56` | `app = FastAPI(lifespan=lifespan)`——10 条路由的唯一挂载点，函数调用式装饰器注册（`@app.post`/`@app.get`），不像 TensorRT-LLM 那样有独立的 `register_routes()` |
| `tokasaurus/server/utils.py:570` | `process_request()`——OpenAI 请求 → 内部 `TokasaurusRequest` 的唯一转换点，含参数校验 |
| `tokasaurus/server/utils.py:503` | `validate_args()`——`stream`/`top_p`/`frequency_penalty`/`logit_bias` 非默认值直接 400 |
| `tokasaurus/manager/manager.py:981` | `manager_loop()`——CPU 侧事件循环本体，全仓库唯一的"主循环" |
| `tokasaurus/manager/manager.py:826` | `schedule_steps()`——一次性调度 `scheduling_steps_ahead` 步，只在第 0 步重新模拟 KV 用量 |
| `tokasaurus/manager/manager.py:702` | `onboard_new_seqs()`——粗筛(`coarse_onboard`)+精筛(`precise_onboard`)两阶段把排队请求收进批次 |
| `tokasaurus/manager/scheduler.py:576` | `schedule()`——真正的批调度算法，函数体开头的注释是原作者自嘲"Bro trust me, it's a good scheduler"（`tokasaurus/manager/scheduler.py:588`） |
| `tokasaurus/manager/scheduler.py:47` | `calc_block_usage_over_time()`——把"未来每个时间步 KV 块占用是多少"算成一条时间线，供调度与 onboard 复用 |
| `tokasaurus/manager/allocator.py:129` | `class BlockAllocator`——分页 KV 分配器，内部即一棵前缀树 |
| `tokasaurus/manager/allocator.py:196` | `prefix_match()`——KV 前缀缓存命中的唯一入口 |
| `tokasaurus/manager/hydragen.py:66` | `group_for_hydragen()`——遍历前缀树，找出满足最小组大小/最小前缀长度的共享前缀分组 |
| `tokasaurus/manager/stopping_predictor.py:152` | `class PredictionMap`——用历史完成长度分布预测在跑序列还需要多少 token（供 KV 预留用） |
| `tokasaurus/manager/input_building.py:152` | `seqs_to_input()`——把调度结果拍平成模型能吃的 `ModelInput`（prefill/hydragen/decode 三段顺序） |
| `tokasaurus/model/entry.py:130` | `get_model_process_dict()`——按 `pp_size`/`tp_size` 决定起几个模型进程、怎么接队列 |
| `tokasaurus/model/basic_worker.py:39` | `basic_model_loop()`——单卡（或纯 TP）模型进程的主循环 |
| `tokasaurus/model/pipeline_worker.py:111` | `pipeline_worker_model_loop()`——流水并行模型进程的主循环，多一层 `dist.send`/`dist.recv` |
| `tokasaurus/model/attention_utils.py:108` | `tokasaurus_attention()`——prefill/hydragen/decode 三段 attention 分别调 flashinfer，再拼接输出 |
| `tokasaurus/model/types.py:196` | `class AttentionInfo`——docstring 原话："tokens are in order of prefill, hydragen, decode" |
| `tokasaurus/model/kv_cache.py:7` | `class LayerKVCache`——单层 KV 缓存的 `nn.Module`，就是两个 `register_buffer` |
| `tokasaurus/model/utils.py:378` | `model = torch.compile(model, fullgraph=True, dynamic=True)`——端到端 torch.compile 的唯一触发点 |
| `tokasaurus/model/utils.py:513` | `class CUDAGraphInfo`——CUDA Graph 捕获与回放的状态容器 |
| `tokasaurus/utils.py:487` | `block_on_queues()`——用 `multiprocessing.connection.wait` 直接卡在 `mp.Queue` 私有属性 `_reader` 上，是整个"事件循环"能不用轮询的底层原因 |
| `tokasaurus/utils.py:492` | `error_propogation_decorator`——因为子进程异常有时会被静默吞掉，这里手动 `print`+`traceback.print_exc()` 再重新抛出 |
| `tokasaurus/server/utils.py:129` | `class ServerState`——server 进程持有的多副本（DP）视图，含 `requests_per_engine` 负载计数 |
| `tokasaurus/server/utils.py:229` | `submit_request()`——按"在途请求数最少"选一个 DP 副本，是全仓库唯一的负载均衡逻辑 |
| `tokasaurus/model/llama.py:162` | `LlamaAttention.make_attn_fn()` 里的 `torch.library.custom_op` 注册——把不可被 `torch.compile` 追踪的 flashinfer 调用包装成一个自定义算子，是 `fullgraph=True` 能成立的关键机制 |
| `tokasaurus/model/llama.py:56` / `:75` | `all_gather()`/`reduce_scatter()`——张量并行下手写的序列并行式集合通信，`torch.compile` 开关会切到 `funcol` 版本 |
| `tokasaurus/model/utils.py:387` | `run_warmup_batches()`——启动期送一批代表性 batch 探测 OOM/触发重编译 |
| `tokasaurus/model/utils.py:631` | `create_cudagraph()`——CUDA Graph 捕获，仅覆盖纯 decode 批次 |
| `tokasaurus/model/utils.py:723` | `class ModelRunner`——统一封装"要不要走 CUDA Graph 回放"的分发逻辑 |
| `pyproject.toml:33` | `"flashinfer-python==0.2.0.post2"`——全仓库唯一的注意力/分页 KV 算子来源 |
| `tokasaurus/model/qwen.py:23` | `class Qwen2ForCausalLM(LlamaForCausalLM)`——全文件只有 43 行，接入一个新模型架构的最小样本 |
| `tokasaurus/manager/monitoring.py:281` | `step_stats()`——每步循环结束后打印/上报吞吐、Hydragen 命中率、KV 利用率等统计 |
| `tokasaurus/manager/monitoring.py:97` | `log_to_statsd()`——全仓库唯一的运行时指标出口，走 statsd 推送而非 Prometheus 拉取 |
| `tokasaurus/utils.py:345` / `:376` | `sglang_manager()`/`vllm_manager()`——拉起真实 vLLM/SGLang 服务做对比基线的 context manager，当前快照里没有调用点 |
| `tokasaurus/common_types.py:80` | `torch_compile: bool = False`——端到端编译默认关闭，需要显式打开 |

## 3. 核心数据结构

**`ServerConfig`**（`tokasaurus/common_types.py:52`，继承 `pydra.Config`）是唯一的配置源，49 个字段，用 `key=value` 的 CLI 语法直接赋值（Pydra 的机制，不是 argparse——`_lab/out/api_surface.json` 里这篇引擎的 `n_cli_flags` 是 0，正是因为 Pydra 不走 `add_argument`）。`finalize()`（`tokasaurus/common_types.py:148`）里做的交叉校验值得一读：比如 `tokasaurus/common_types.py:171`-`174` 断言"开 Hydragen 又开 CUDA Graph 时，`cudagraph_max_size` 必须小于 `hydragen_min_group_size`"——两个特性的实现互相不兼容达到某个批大小之后，靠一个 `assert` 在启动期挡住而不是靠代码兼容性设计避免，是这种小团队研究引擎常见的"用断言代替泛化"取舍。

**`Sequence`**（`tokasaurus/manager/types.py:28`）是调度侧唯一的请求单位——一个 `TokasaurusRequest`（用户的一次 `n=k` 请求）会被拆成 `k` 个独立 `Sequence`（`tokasaurus/manager/manager.py:82` 的 `sids = [f"{req.id}-part-{i}-of-{req.n}" ...]`），调度器完全不知道"这几个 Sequence 属于同一个用户请求"，只在 `finish_sequences()`（`tokasaurus/manager/manager.py:212`）里按 `req_id_to_seq_ids` 反查、凑齐 `n` 份才真正把 `RequestOutput` 推回 server。`SchedulingQueue`（`tokasaurus/manager/types.py:151`）用三个普通字典（`decoding_seqs`/`prefilling_seqs`/`queued_seqs`）当插入顺序队列，没有优先级、没有多种排队策略——这是和 `## 6` 里 vLLM `RequestQueue` 的关键差异点。

**`ModelInput`/`AttentionInfo`/`PageInformation` 用了"Builder 与 Tensor 分离"的模式**：`AttentionInfoBuilder`（`tokasaurus/model/types.py:163`）在 manager 进程里只用 Python `list` 攒数据（`PageInformationBuilder.add_sequence()`，`tokasaurus/model/types.py:122`），这个 builder 对象本身跨进程通过 `mp.Queue` pickle 传给 model 进程后，才在 model 进程里调用 `.build()`（`tokasaurus/model/types.py:179`）把 list 转成 `torch.Tensor`。这么做的理由能直接从类型上看出来：`list[int]` 的 pickle/unpickle 比 `torch.Tensor`（要走 CUDA/共享内存那一套序列化协议）轻得多，而 manager 进程本来就没有 GPU 上下文，没必要在那一侧构造张量。

**`HydragenGroup`**（`tokasaurus/manager/types.py:342`）只有两个字段：`block_ids: list[int]`（共享前缀在前缀树里对应的块序列）和 `seq_ids: set[str]`（属于这组的序列 id）——它本身不携带任何张量，是纯粹的"分组结果"记录，真正的张量构造发生在 `input_building.py` 把 `HydragenGroup` 转成 `PageInformationBuilder.add_sequence()` 调用的那一步（`tokasaurus/manager/input_building.py:227`-`239`）。

**`ServerState`/`Engine`**（`tokasaurus/server/utils.py:129`、`tokasaurus/common_types.py:37`）是数据并行（`dp_size > 1`）场景下 server 进程持有的多副本视图：一个 server 进程对应 `config.dp_size` 个 `Engine`，每个 `Engine` 各自一套 manager+model 进程和队列。`submit_request()`（`tokasaurus/server/utils.py:229`-`233`）的负载均衡策略非常直白——`min_requests = min(state.requests_per_engine); engine_index = state.requests_per_engine.index(min_requests)`，选当前在途请求数最少的那个副本，不看 prompt 长度、不看该副本当前 KV 占用率。这个"哪个副本队列短就发哪个"的贪心策略，是能用一行代码说清楚的负载均衡实现——代价见 `## 7`。

**`ExtraModelConfig`**（`tokasaurus/model/types.py:492`）携带模型层不该从 HF `config` 里读、但张量并行/流水并行必须知道的信息：`pp_rank`/`tp_rank`/`tp_group`/`torch_compile`。张量并行的实现方式值得单独说一句：`LlamaAttention.forward()`（`tokasaurus/model/llama.py:218`）在算 QKV 投影前先 `all_gather(hidden_states, extra_config)`（`tokasaurus/model/llama.py:56`），把被切成 `tp_size` 份的 token 批次先聚合成完整批次，再各自算被切成 `tp_size` 份的注意力头；`o_proj` 之后再 `reduce_scatter()`（`tokasaurus/model/llama.py:75`）把结果按 token 切回各 rank——这是"序列并行+张量并行"的组合写法（RMSNorm/残差加法只在切分后的 token 子集上算，只有矩阵乘那部分聚合成全量 token 再切分头），不是最朴素的"全程 all-reduce"式张量并行。`all_gather`/`reduce_scatter` 内部还按 `extra_config.torch_compile` 分了两条实现（`tokasaurus/model/llama.py:62`-`63`、`:81`-`82`）：不开编译时直接调 `torch.distributed.all_gather_into_tensor`/`reduce_scatter_tensor`，开编译时改用 `torch.distributed._functional_collectives`（`funcol`）的版本——因为原始 NCCL 集合通信在 `torch.compile` 追踪时表现不稳定，这是决策 7（`fullgraph=True`）向张量并行实现渗透出的具体代价。

**权重加载和 `tp_map` 是同一件事的两端。** `Qwen2ForCausalLM.make_tp_map()`（`tokasaurus/model/qwen.py:27`-`43`，`## 0`/`## 1` 已提到）产出的是一个"参数名 → 该参数在哪个维度上被张量并行切分"的字典；这个字典真正被消费的地方是 `load_safetensors_repo()`（`tokasaurus/model/safetensors_utils.py:34`），它拿着 `tp_map` 和 `compute_shard_bounds()`（`tokasaurus/model/safetensors_utils.py:12`-`26`）算出的切片边界，直接用 `safe_open`（`tokasaurus/model/safetensors_utils.py:6`）从磁盘上的 safetensors 文件里**只读取当前 rank 需要的那一份权重**，不需要先把完整参数加载进内存再切片丢弃——张量并行的"分片"这个概念从模型定义（`make_tp_map`）一路贯穿到磁盘 I/O 层，是同一份元数据在两处被复用，而不是两套独立实现。

**`ModelRunner`/`CUDAGraphInfo`**（`tokasaurus/model/utils.py:723`、`:513`）是模型进程侧"要不要走 CUDA Graph 回放"的统一分发点：`create_cudagraph()`（`tokasaurus/model/utils.py:631`）为一组预设的纯 decode 批大小（`prefill_tokens=0`，与 `tokasaurus/manager/manager.py:321` 的 `model_will_use_cudagraphs()` 判据一致）各自捕获一张图，`ModelRunner` 在真正跑一个批次时，按这个批次是否命中某个已捕获的形状来决定是重放静态图还是走普通 eager/compiled 前向。

## 4. 主流程走读

一次 `POST /v1/chat/completions` 到拿到响应,要穿过下面 9 步(跨三个进程):

① **HTTP 层接收**：`oai_chat_completions`（`tokasaurus/server/endpoints.py:74`）调 `generate_output` → `process_request()`（`tokasaurus/server/utils.py:570`）先 `validate_args()`（`tokasaurus/server/utils.py:503`）拒绝不支持的参数，再用 `state.tokenizer.apply_chat_template(...)` 把消息转成 `input_ids`，组装成 `TokasaurusRequest`（`tokasaurus/server/types.py:143`）。

② **提交给 manager**：`submit_request()`（`tokasaurus/server/utils.py:229`）先按"当前在途请求数最少"选一个数据并行副本（`tokasaurus/server/utils.py:230`-`233`，`dp_size=1` 时这一步没有意义），把 `TokasaurusRequest` 放上该副本的 `q_server_to_manager`，返回一个 `asyncio.Event`，HTTP 协程在这里挂起等待。

③ **manager 主循环唤醒**：`manager_loop()`（`tokasaurus/manager/manager.py:981`）平时卡在 `block_on_queues([q_server_to_manager, q_model_to_manager])`（`tokasaurus/manager/manager.py:989`，底层是 `tokasaurus/utils.py:487`），新请求到达时被唤醒，`handle_new_server_commands()`（`tokasaurus/manager/manager.py:71`）把请求拆成 `n` 个 `Sequence`，塞进 `SchedulingQueue.queued_seqs`。

④ **多步调度**：`schedule_steps(state, num_steps_to_schedule)`（`tokasaurus/manager/manager.py:826`，`num_steps_to_schedule = scheduling_steps_ahead - num_inflight_batches`）在第 0 步调用 `onboard_new_seqs()`（`tokasaurus/manager/manager.py:702`）尽量把排队请求收进批次，再调 `schedule()`（`tokasaurus/manager/scheduler.py:576`）产出 `ScheduleDecision`；后续步复用同一个 `num_prefill` 速率，只调 `make_scheduling_decision()`（`tokasaurus/manager/scheduler.py:541`），不重新模拟 KV 用量。

⑤ **Hydragen 分组**（若开启）：`group_for_hydragen()`（`tokasaurus/manager/manager.py:879`，即 `tokasaurus/manager/hydragen.py:66`）只在第 0 步对当前解码队列做一次分组，后续步用 `restrict_hydragen_groups()`（`tokasaurus/manager/manager.py:909`，即 `tokasaurus/manager/hydragen.py:142`）在原分组基础上按"这一步还活着的 seq_id"做子集裁剪，避免每步重新遍历前缀树。

⑥ **提交给模型进程**：`prepare_and_submit_to_model()`（`tokasaurus/manager/manager.py:328`）按流水并行阶段数切片（`slice_decision()`，`tokasaurus/manager/input_building.py:103`），每个切片经 `seqs_to_input()`（`tokasaurus/manager/input_building.py:152`）拍平成 `ModelInput`，放上 `q_manager_to_model`；随后 `apply_decision()`（`tokasaurus/manager/scheduler.py:640`）推进各 `Sequence` 的 `prompt_scheduled`/`completion_scheduled` 计数。

⑦ **模型进程执行**：`basic_model_loop()`（`tokasaurus/model/basic_worker.py:39`，单卡/纯 TP）或 `pipeline_worker_model_loop()`（`tokasaurus/model/pipeline_worker.py:111`，流水并行，多一层 `dist.send`/`dist.recv` 传递跨阶段隐状态）从 `input_q` 取出 `ModelInput`，`ModelRunner.plan`/`run`（`model/utils.py`）驱动模型前向；`tokasaurus_attention()`（`tokasaurus/model/attention_utils.py:108`）内部把 prefill/hydragen/decode 三段分别喂给 flashinfer 的三个 wrapper，再用 `cascade.merge_state` 拼接 hydragen 分支和对应的 decode 分支。

⑧ **结果回传**：模型进程把 `ModelOutput`（含采样出的 token id、可选 logprobs）放上 `q_model_to_manager`；`handle_new_model_outputs()`（`tokasaurus/manager/manager.py:249`）→`handle_output()`（`tokasaurus/manager/manager.py:142`）把 token 写回对应 `Sequence.seq_output`，`check_for_stop_strings()`（`tokasaurus/manager/manager.py:108`）批量检查停止串，`finish_sequences()`（`tokasaurus/manager/manager.py:212`）把凑齐 `n` 份输出的请求打包成 `RequestOutput` 放上 `q_manager_to_server`。

⑨ **HTTP 层返回**：`receive_from_manager_loop()`（`tokasaurus/server/utils.py:154`，一个常驻 `asyncio.Task`）从 `q_manager_to_server` 取出 `RequestOutput`，`set()` 对应的 `asyncio.Event`，挂起的 HTTP 协程被唤醒，`process_chat_completions_output()`（`tokasaurus/server/utils.py:694`）拼出 OpenAI 格式的 JSON 响应体返回给客户端。

## 5. 设计决策与代价

八条决策先给一张总览，细节在各自小节：

| # | 决策 | 省下了什么 | 直接的代价 |
|---|---|---|---|
| 1 | 三进程+裸 `mp.Queue` | 序列化/RPC 框架开销 | 没有跨机部署能力 |
| 2 | 调度算法与状态管理物理分离 | 算法可单独单测 | 一次调度决策要跨 3 个文件才能读全 |
| 3 | 多步预调度+完成长度预测 | KV 显存不必按 `max_tokens` 保守预留 | 需要 bump 机制兜底预测偏差 |
| 4 | Hydragen 批量共享前缀 attention | 共享前缀的重复访存 | 分组/重排本身的调度开销 |
| 5 | 协议层主动收窄采样面 | 采样 kernel/流式状态管理复杂度 | 不能服务交互式聊天场景 |
| 6 | 零自研 kernel，外包 flashinfer | 专职 kernel 工程投入 | 性能天花板被上游库锁死 |
| 7 | 端到端 `fullgraph=True` 编译+启动期预热 | 生产期编译延迟/意外 OOM | 只能覆盖收窄过的模型族 |
| 8 | TP 用序列并行+头并行组合 | 非矩阵乘操作的 `tp_size` 倍冗余 | 通信从一次 all-reduce 变两次 |

### 决策 1：三进程 + 裸 `mp.Queue`，不用任何 RPC/Actor 框架

**为什么这么设计**——三个角色（HTTP、CPU 编排、GPU 执行）职责边界固定、拓扑不变，不需要服务发现、不需要跨语言互操作，`block_on_queues()`（`tokasaurus/utils.py:487`）直接调用 `multiprocessing.connection.wait()` 卡在 `mp.Queue` 的私有 `_reader` 管道上，零额外依赖、零序列化框架开销，这和 README 强调的"Very low CPU overhead"（`README.md:24`，文档所述）目标直接对应。

**不这样会怎样**——引入 gRPC/Ray/ZeroMQ 之类的框架会在每条消息上多一层序列化协议和一层间接调度，对"CPU 开销要压到最低"这个目标是净损耗；但代价是这套骨架**没有跨机部署能力**——`entry.py` 里 manager/model 进程全部通过本地 `mp.Process` 拉起（`tokasaurus/entry.py:126`），队列也是本机 `multiprocessing.Queue`，`tp_size`/`pp_size` 再大也只能在一台机器的多张卡之间跑，无法像 vLLM/SGLang/TRT-LLM 那样借助 Ray/MPI 做多机编排。

**什么时候可以不这样**——当模型大到必须多机 PP/TP（比如百亿参数以上模型分布在跨节点的卡上）时，裸 Queue 就不够用了，这也是为什么它的官方示例最多只给到"8 卡流水并行跑 70B"（`README.md` Quickstart 部分），而不是多机集群部署方案。

### 决策 2：调度算法（684 行）与状态管理（allocator+manager）物理分离

**为什么这么设计**——`scheduler.py` 里的函数（`calc_block_usage_over_time()`、`try_onboarding_seqs()`、`schedule()`）几乎不修改任何长期存活对象的生命周期，只读 `SchedulingQueue` 的快照、产出一个 `ScheduleDecision`（`tokasaurus/manager/types.py:254`）——这让调度算法可以脱离并发状态被单独测试，`tests/test_scheduler.py` 正是直接构造 `SchedulingQueue` 传进 `schedule()` 断言结果，不需要起真实的三进程集群。

**不这样会怎样**——如果把"要不要抢占""分配哪些 KV 块""要不要触发 Hydragen 分组"全部糅进同一段有状态代码里，`calc_block_usage_over_time()` 这种数值密集的模拟逻辑（`tokasaurus/manager/scheduler.py:47`-`197`）几乎不可能写单元测试，只能靠端到端集成测试（起真实 GPU）间接验证，调试成本会高得多。

**什么时候可以不这样**——当调度策略本身需要感知运行时状态（比如按 LoRA adapter 亲和性分组、按 SLO 优先级抢占）时，把决策逻辑收回到有状态对象里反而更自然——这正是 vLLM 把状态和策略都放进同一个 `Scheduler` 类（`vllm:vllm/v1/core/sched/scheduler.py:73`）的原因，见 `## 6`。

### 决策 3：提前调度 `scheduling_steps_ahead` 步 + 用历史分布预测完成长度

**为什么这么设计**——目标工作点是"大批量、小模型、快 GPU"，一步 forward 可能只要几毫秒，如果每步都要等 manager 现算一次调度决策再发车，CPU 编排延迟会直接吃掉 GPU 利用率。`scheduling_steps_ahead: int = 8`（`tokasaurus/common_types.py:89`）让 `schedule_steps()`（`tokasaurus/manager/manager.py:826`）一次算好未来 8 步的批次；配合 `EarlyStoppingTracker`/`PredictionMap`（`tokasaurus/manager/stopping_predictor.py:152`，用历史完成序列"实际生成长度 / max_tokens"的分布做分桶插值预测），KV 块预留不必按每个请求的 `max_tokens` 保守分配，而是按"大概率会在哪里停"来分配，从而能塞进更多并发序列。

**不这样会怎样**——按 `max_tokens` 保守预留是更简单、更常见的默认假设，但在"大量请求实际完成长度远小于 `max_tokens`"的场景（重复采样时很多样本会提前触发 EOS）下会系统性浪费 KV 显存，直接压低能同时在跑的批大小——而批大小正是这个引擎唯一要优化的指标。反过来，"提前算好的调度决策，运行时才发现 KV 真不够"这种预测失误必须有兜底，于是有了 `allocate_tokens_for_decode_bumping_seqs_if_necessary()`（`tokasaurus/manager/manager.py:746`）——把预测偏差造成的复杂度转移成了"bump"（把序列踢回队首、换新 seq id 重排）这条补救路径，这是多步预调度必须付出的必要复杂度，不是可以白拿的免费午餐。

**什么时候可以不这样**——当完成长度高度可预测（比如几乎所有请求都会跑满 `max_tokens`，典型如强制长度的摘要任务）或 GPU 慢到 CPU 调度延迟本来就不是瓶颈时，单步调度（多数引擎的默认做法）更简单、也不会因预测偏差触发额外的 bump 开销。

### 决策 4：Hydragen——把共享前缀的多次小 attention 合成一次批量 attention

**为什么这么设计**——当很多解码序列共享同一段长前缀、每条每步只贡献 1 个 query token 时，朴素做法是每条序列各自对"共享前缀+独有后缀"的完整 KV 做一次 attention——这是 G 条序列各自读一遍同一段共享 KV，是纯粹的访存冗余（同一批数据被读 G 次）。Hydragen 的做法（`tokasaurus/manager/hydragen.py:66` 的 `group_for_hydragen()` 先在前缀树里找分组，`tokasaurus/manager/input_building.py:227`-`239` 把组内序列的 query 拼成一批）把这 G 次读，压成 1 次批量、无因果掩码的"伪 prefill"调用（`tokasaurus/model/attention_utils.py:156` 的 `hydragen_wrapper.run_return_lse`），共享前缀的 KV 只在物理上和逻辑上都只被读一次；再用在线 softmax 合并（`cascade.merge_state`，`tokasaurus/model/attention_utils.py:165`）把这部分结果和各序列独有后缀的 decode attention 结果按正确的归一化常数拼回去。

**不这样会怎样**——不开 Hydragen，共享前缀这部分依然会走对每条序列各自的 decode attention（`decode_wrapper.run_return_lse`），显存里存的 KV 块数量和有没有 Hydragen 完全一样（前缀共享这件事由 `BlockAllocator` 的前缀树独立完成，见 `## 6` 的三线对比），差的只是"读这些已经在显存里的 KV 时算子内部的访存效率"。这一点必须讲清楚，否则容易把 Hydragen 和"省显存"混为一谈：**它优化的是算子执行时的内存带宽利用率，不是 KV 缓存占用量。**

**什么时候可以不这样**——当序列间几乎没有共享前缀（比如完全独立的多轮对话），或共享前缀太短、能凑够 `hydragen_min_group_size`（默认 32，`tokasaurus/common_types.py:61`）条序列的分组太少时，分组、重排（`reorder_decoding_seqs_for_hydragen()`，`tokasaurus/manager/hydragen.py:10`）本身的调度开销会抵消掉访存收益——这也是为什么 `hydragen_min_group_size`/`hydragen_min_prefix_len` 两个阈值存在的原因。另外 README 明确提到（`README.md`，"Hydragen"一节，文档所述）开启 Hydragen 会因为 bf16 合并带来"slight numerical impact"，不是零代价的开关。

### 决策 5：协议层故意收窄——拒绝 streaming、拒绝除温度外的大多数采样旋钮

**为什么这么设计**——`validate_args()`（`tokasaurus/server/utils.py:503`-`531`）明确拒绝 `stream=True`（400 "Streaming is not supported"）、`top_p not in [None, 1.0]`、`frequency_penalty≠0`、`presence_penalty≠0`、`logit_bias≠None`；`SamplingParams`（`tokasaurus/server/types.py:137`）本身就只有 `temperature`/`top_p` 两个字段。这不是功能没做完，是刻意的范围收窄：`## 1` 已经确认它的招牌工作负载是"对同一个 prompt 打出成百上千份样本、只靠改温度制造多样性"的重复采样场景（`benchmarks/monkeys_*.py`），这类场景既不需要流式体验（反正要等全部样本跑完才能统计 pass@k），也不需要精细的重复惩罚/logit 偏置这类为"单次高质量生成"设计的采样后处理。

**不这样会怎样**——实现完整的采样管线（top_k/top_p 截断、frequency/presence 惩罚、logit_bias、流式 SSE）会显著增加 model worker 侧采样 kernel 的复杂度，以及 server 侧为每个请求维护增量输出通道的状态管理成本，而这些复杂度在它默认的使用方式（`/custom/synchronous-batch-completions`，一次提交一批、等全部跑完再拿结果）下完全用不上——徒增维护负担换不来目标场景里的收益。

**什么时候可以不这样**——一旦要服务面向终端用户的交互式聊天产品（用户盯着屏幕等首字延迟、需要流式体验、需要重复惩罚来对抗车轱辘话），这套收窄就直接不够用了。这也是它没有被定位成"vLLM/SGLang 的替代品"，而是"某一类特定工作负载的专用引擎"的根本原因——协议兼容的广度从来不是它要争取的指标。

### 决策 6：零自研 kernel，注意力/分页 KV 全部外包给 flashinfer

**为什么这么设计**——`pyproject.toml:33` 是全仓库唯一一处 attention 相关依赖声明（`flashinfer-python==0.2.0.post2`），`attention_utils.py`/`model/types.py` 里所有的 `BatchPrefillWithPagedKVCacheWrapper`/`BatchDecodeWithPagedKVCacheWrapper`/`cascade.merge_state` 都是直接调用这个库的公开 API。一个由研究组维护、贡献者个位数（`README.md` bibtex 列出 8 位作者，文档所述）的项目，把"注意力算子怎么写才能榨干硬件"这个需要专职 kernel 工程师长期投入的问题让渡给上游通用库，能把有限的工程精力全部投入到调度/Hydragen 这些真正体现该引擎差异化价值的地方——`_lab/repo_stats.json` 里 CUDA/Triton/Rust/C/C++ 文件数全部是 0，不是"没有性能敏感代码"，而是"性能敏感代码的维护责任被转移给了 flashinfer 团队"。

**不这样会怎样**——如果都要自研，13,219 行的规模至少要再加一个数量级（参照 vLLM 的 169 个 CUDA 文件、202 个 Triton 文件），且需要专职 kernel 工程师长期投入——这和"研究组维护的教学/研究性引擎"这个体量、这个团队规模完全不匹配，会拖慢它真正想验证的调度/Hydragen 这些想法的迭代速度。

**什么时候可以不这样**——当需要 flashinfer 尚未覆盖的 attention 变体（比如某些新的稀疏模式、非标准的 MLA 变体），或者要榨干最后一点延迟（生产级 SLA 场景，容不下通用库为覆盖更多场景做的抽象开销）时，外包给通用库的性能天花板会先于"调度策略更聪明"这个优化方向被打穿——这也是为什么 vLLM/SGLang 在接入 FlashInfer/FlashAttention 之余，还各自维护自己的 Triton/CUDA kernel 作为"极限性能"的备选路径。

### 决策 7：端到端 `fullgraph=True` 编译 + 启动期主动预热探测 OOM/触发重编译

**为什么这么设计**——`torch.compile(model, fullgraph=True, dynamic=True)`（`tokasaurus/model/utils.py:378`）对整个模型只编译一张图，不允许中途图断裂（`fullgraph=True` 一旦遇到不能追踪的操作就直接报错，而不是像默认模式那样退回 eager 子图）；配合 `run_warmup_batches()`（`tokasaurus/model/utils.py:387`，docstring 原话"Send a max-sized batch to the model to check if it's gonna OOM. Can also send more batches to try to trigger recompiles ahead of time."，`tokasaurus/model/utils.py:398`-`401`）在服务真正接客之前，主动把一组代表性的 batch 形状（不同的 prefill/decode token 数组合，`decode_sizes`/`prefill_sizes`，`tokasaurus/model/utils.py:411`-`426`）都喂一遍模型，把 `torch.compile` 因为形状变化触发的重编译、以及最大批量下的显存 OOM，全部提前到启动期暴露。这条路径直接对应 README 的承诺（`README.md:26`，文档所述）"No OOMs or recompiles in production"。

**不这样会怎样**——不做启动期预热，第一次遇到某个新 batch 形状时才现场触发 `torch.compile` 重编译，会让线上流量的某几个请求（很不巧撞上一个此前没跑过的形状）承受几秒到几十秒的编译延迟，在"重复采样、大批量吞吐"这个目标场景里，这类长尾延迟会直接拖累整批任务的完成时间；同理，如果不提前用最大批量探测显存，OOM 会在服务运行中期、批量恰好冲到峰值时才发生，此时中断的代价（已经在跑的所有序列全部报错）远大于启动期失败重试。`fullgraph=True` 本身也是一种偏保守的正确性优先设计：一旦模型代码里出现新的 `torch.compile` 不支持的操作，会在编译期直接报错，而不是悄悄退回一个性能打折的子图，逼着开发者显式处理，不产生"图断裂了但没人注意到"的隐性性能衰退。

**什么时候可以不这样**——`fullgraph=True` 只在模型族收窄（`tokasaurus/model/` 目前只有 Llama/Qwen2/Qwen3 三种架构，`_lab/out/struct_map.json` 的 `entrypoint` 子系统之外没有更多模型文件）、且团队能随时跟进新架构的兼容性问题时才可行；一旦要支持的模型架构、注意力变体、量化方案组合数远超一个研究组能逐个验证 `torch.compile` 兼容性的量级（vLLM/SGLang 面对的正是这种局面），piecewise 编译（允许部分子图退回 eager）就成了必须的妥协——`## 6` 会具体对照 vLLM 在这一点上的取舍。

补一句"为什么 `fullgraph=True` 在这个仓库里能落地"的具体机制：`attn_fn`（`tokasaurus/model/llama.py:162`）用 `torch.library.custom_op` 把整段 flashinfer 调用（`tokasaurus_attention()`）注册成一个自定义算子，并配一个 `register_fake`（`tokasaurus/model/llama.py:206`-`214`）告诉 Dynamo 这个算子的输出形状是什么。同样的手法在采样步骤上又用了一次：`sample_from_probs`（`tokasaurus/model/llama.py:362`-`367`）把随机采样也包成 `torch.library.custom_op`——一个模型里两处不可追踪的操作（注意力、采样）各自被包成一个"黑盒但形状已知"的算子，是"整图编译"能够兼容"外部不可追踪 kernel"这个通用矛盾的具体解法，不是碰运气碰出来的。`torch.compile` 追踪到这个算子时只需要相信"它返回一个和 `ragged_q` 同形状的张量"，不需要真的把 flashinfer 内部那些不可追踪的 CUDA 调用摊开在计算图里——这是"整图编译"和"调用不可追踪的外部 kernel 库"这两个互相冲突的目标能同时成立的具体接口手法，不是靠运气。

### 决策 8：张量并行用"序列并行+头并行"组合，而不是朴素的全程 all-reduce

**为什么这么设计**——`LlamaAttention.forward()`（`tokasaurus/model/llama.py:218`）在算 QKV 投影前先 `all_gather(hidden_states, extra_config)`（`tokasaurus/model/llama.py:56`）把切分的 token 批次聚合回全量，o_proj 后再 `reduce_scatter()`（`tokasaurus/model/llama.py:75`）切回各 rank——RMSNorm/残差加法这些非矩阵乘操作全程只在切分后的 token 子集上算，只有真正需要跨 rank 通信的矩阵乘部分才处理全量 token。相比"每个 rank 全程持有完整 token 批次，只在 o_proj 后 all-reduce"的朴素张量并行，这个写法能省下 `tp_size - 1` 份的重复 LayerNorm/残差加法计算和相应的激活显存——在"CPU/显存开销要压到最低"这个目标下，这种省法虽然只是矩阵乘之外的边角计算，但对小模型来说边角计算占比本来就不小。

**不这样会怎样**——朴素张量并行实现更简单（只需要在 o_proj 后插一次 all-reduce），但每个 rank 都要为 RMSNorm、残差加法这些操作重复处理全量 token，激活显存和这部分计算的 wall-clock 时间都是 `tp_size` 倍冗余；序列并行式写法把这部分冗余消掉了，代价是通信原语从一次 all-reduce 变成一次 all-gather+一次 reduce-scatter（两次通信而不是一次，总通信量理论上相当，但多了一次同步点）。

**什么时候可以不这样**——`tp_size == 1` 时 `all_gather`/`reduce_scatter` 直接原样返回输入（`tokasaurus/model/llama.py:57`-`58`、`:76`-`77`），这套机制完全不介入；当 `torch_compile=True` 时还要切到 `funcol.all_gather_tensor`/`funcol.reduce_scatter_tensor`（`tokasaurus/model/llama.py:63`-`64`、`:82`-`85`）而不是原始 `torch.distributed` 集合通信——注释直接写明原因（`tokasaurus/model/llama.py:62`，源码为证）"funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash"，也就是说这个分支本身是为了绕开一个已知的 NCCL 崩溃问题，不是设计初衷，是决策 7（整图编译）向分布式实现渗透出的又一处具体代价。

## 6. 同位对照

### 调度+KV 管理的代码规模：684 行是错的读法，2,179～3,823 行才是可比的数字

把"调度"理解成"能不能对一批候选请求做出批处理决策"这一件事，vLLM 对应的是 `vllm/v1/core/sched/` 目录 6 个文件共 4,012 行（`vllm:vllm/v1/core/sched/scheduler.py:73` 的 `class Scheduler` 本体就有 3,037 行），如果再算上 KV 块管理（`vllm:vllm/v1/core/block_pool.py:143` 的 `class BlockPool`，831 行，加 `kv_cache_manager.py` 887 行），合计 5,730 行。Tokasaurus 完成同等范围的工作（批调度算法+KV 块分配+前缀树），最小可比口径是 `scheduler.py`（684）+`allocator.py`（431）+`manager.py`（1,064，因为 onboard/bump/hydragen 编排这些在 vLLM 里也算进 `Scheduler` 类方法）= **2,179 行**；如果算上 Hydragen 分组、完成长度预测、批构建、监控统计这些 vLLM 没有对应物或者放在别处的辅助逻辑，整个 `tokasaurus/manager/` 目录是 **3,823 行**。无论按哪个口径，都明显小于 vLLM 的 5,730 行，但差距不是"vLLM 写得啰嗦"，下一段具体拆解差在哪。

### 差的是什么：策略可插拔性 + 部署形态覆盖面，不是"调度这件事本身"

vLLM 的调度器背后有一层抽象接口 `SchedulerInterface`（`vllm:vllm/v1/core/sched/interface.py:38`），允许整体替换调度器实现（比如异步调度器 `vllm:vllm/v1/core/sched/async_scheduler.py:1`）；请求排队策略本身也是可插拔的——`RequestQueue` 抽象基类下有 `FCFSRequestQueue` 和 `PriorityRequestQueue` 两种实现（`vllm:vllm/v1/core/sched/request_queue.py:20`、`:75`、`:131`），支持按优先级抢占。Tokasaurus 的 `SchedulingQueue`（`tokasaurus/manager/types.py:151`）是一个具体 dataclass，三个普通字典，只有 FCFS 一种排队方式，没有接口层、没有可插拔实现——**vLLM 多出来的行数里，相当一部分是"允许调度策略被替换/扩展"这件事本身的抽象成本**，这是它作为通用生产引擎必须付的"泛化税"；而 Tokasaurus 只服务一种工作负载，不需要为"以后可能要换一种调度策略"预留接口。另外 vLLM 的 KV 管理还要覆盖 PD 分离场景下的跨机 KV 传输（`vllm/distributed/kv_transfer/kv_connector/v1/` 目录，`_lab/out/struct_map.json` 把这部分也计入了"scheduler"子系统，光是 `nixl/`、`mooncake/`、`offloading/` 几个连接器就有超过 3,000 行）——这是 Tokasaurus 完全没有涉猎的能力（`## 7`/`## 8` 会具体展开缺口）。

### 前缀共享的三条线：块哈希缓存 / RadixAttention / Hydragen，省的是三件不同的事

| 引擎 | 机制 | 省什么 | 对应位置 |
|---|---|---|---|
| Tokasaurus（本篇） | 前缀树（`PrefixTreeBlock`）+ 前缀匹配 | 显存容量 + prefill 重算算力（复用已缓存的 KV 块） | `tokasaurus/manager/allocator.py:196` |
| Tokasaurus（本篇） | Hydragen：共享前缀分组批量 attention | attention 算子内部的访存效率（同一段 KV 从"读 G 遍"变成"读 1 遍" | `tokasaurus/manager/hydragen.py:66`、`tokasaurus/model/attention_utils.py:156` |
| SGLang | RadixCache：基数树管理跨请求共享的 KV 块 | 与 Tokasaurus 前缀树同一件事——显存容量 + prefill 重算算力 | `sglang:python/sglang/srt/mem_cache/radix_cache.py:303` |
| vLLM | 块哈希前缀缓存（`BlockPool`） | 同上，显存容量 + prefill 重算算力 | `vllm:vllm/v1/core/block_pool.py:143` |

三个引擎在"要不要为共享前缀多存一份 KV/要不要重算"这个问题上给出的是**同一类答案**（缓存已算好的 KV 块，靠某种树结构做前缀匹配）；但 Tokasaurus 独有的是在"KV 已经确定共享、已经缓存好"之后，**还要不要为每条序列单独发起一次读它的 attention 调用**这个更细粒度的问题——这是 vLLM 的块哈希缓存和 SGLang 的 RadixCache 都没有覆盖的层面（两者解决的是"存不存、算不算"，Hydragen 解决的是"已经决定要读了，读的方式够不够高效"）。三者理论上可以叠加：SGLang/vLLM 拿到共享前缀的判断后，同样可以在 attention 算子层面做类似 Hydragen 的批量化——这属于"友商可改进点"的范畴，本篇不展开判断。

### 张量并行的通信原语：reduce-scatter+all-gather vs 标准 all-reduce

`## 5` 决策 8 已经讲过 Tokasaurus 在 o_proj 之后用 `reduce_scatter()`（`tokasaurus/model/llama.py:75`）把结果按 token 切回各 rank，而不是让每个 rank 都持有全量 token。vLLM 通用的 `RowParallelLinear.forward()`（`vllm:vllm/model_executor/layers/linear.py:1641`-`1660`）在 `tp_size > 1` 时走的是标准 `tensor_model_parallel_all_reduce(output_parallel)`——每个 rank 在 all-reduce 之后都拿到完整的、未切分的 token 批次输出。这不是"谁更先进"的问题：vLLM 的 `RowParallelLinear` 是一个要被几十种模型架构复用的通用层，模型定义代码不需要关心"当前 hidden_states 是全量的还是按 token 切分的"这种细节，用全量 all-reduce 换来的是这个抽象对上层完全透明；Tokasaurus 只需要在 `LlamaBlock` 这一条路径上手动维护"hidden_states 现在是切分状态还是全量状态"，愿意承担这份心智负担来换激活显存和非矩阵乘计算量的节省，是因为它的模型定义代码总共只有几个文件、改起来成本可控——同一个技术选择（要不要用序列并行式的 reduce-scatter），在"通用层库"和"手写单一模型定义"这两种场景下的性价比是反过来的。

### 可观测性：statsd 推送 vs Prometheus 拉取

Tokasaurus 的全部运行时指标走 `statsd`（`tokasaurus/manager/monitoring.py:97`-`100` 的 `log_to_statsd()`，内部是 `statsd.StatsClient(server_url).gauge/incr` 这类推送调用），10 条 HTTP 路由里没有一条是抓取式的 `/metrics`。vLLM 则内置了标准的 Prometheus 拉取端点（`vllm:vllm/entrypoints/serve/instrumentator/metrics.py:78` 的 `Mount("/metrics", make_asgi_app(registry=registry))`），运维方只要把 Prometheus 指向这个端口就能采集，不需要额外部署一个 statsd 收集器。两种模式各有取舍：推送式对"部署环境里本来就有 statsd/dogstatsd 收集链路"的团队几乎零配置，缺点是没有收集器在跑时,`StatsTracker`（`tokasaurus/manager/monitoring.py:121`）算出来的数据无处可看；拉取式不需要额外基础设施就能临时 `curl` 一次看到当前值，但需要引擎自己维护一套 Prometheus 客户端库依赖和指标注册表。`## 8` 已经给出针对这一点的具体改进建议。

### 抢占/bump 语义：都是 LIFO，但踢法不同

vLLM 的抢占受害者选择是 `self.running.pop()`（`vllm:vllm/v1/core/sched/scheduler.py:686`）——从 `running` 列表尾部弹出，即最后加入的请求先被踢，`00-总览与阅读地图.md` §4 已指出这等价于"保护最老的请求"。Tokasaurus 的 `allocate_tokens_for_decode_bumping_seqs_if_necessary()`（`tokasaurus/manager/manager.py:746`）同样是 `running_seqs.pop()`（`tokasaurus/manager/manager.py:764`，同样 LIFO），但踢法更重：不是简单地把请求挪回队列重排，而是**创建一个带 `-bumped` 后缀的全新 `Sequence`**（`tokasaurus/manager/manager.py:770`-`779`），并显式把旧 id 从 `req_id_to_seq_ids` 换成新 id（`tokasaurus/manager/manager.py:789`-`790`），原因是防止跨进程异步返回的、属于旧序列的过期输出被张冠李戴地追加到新分配的批次位置上（`tokasaurus/manager/manager.py:766`-`769` 的注释）。这是一个具体的"多进程异步系统"的正确性细节，比 vLLM 单进程内直接操作对象引用的抢占实现多付出了"必须换身份证"这层代价。

### 编译策略：一张图 vs 分片图

Tokasaurus 对模型只编译一张完整的图（`torch.compile(model, fullgraph=True, dynamic=True)`，`tokasaurus/model/utils.py:378`），中途一旦有算子追踪不了就直接编译失败，靠 `## 5` 决策 7 的启动期预热把这类失败提前暴露。vLLM V1 的默认编译模式是 `VLLM_COMPILE`（`vllm:vllm/config/compilation.py:452`-`459`，字段文档字符串原话"Custom vLLM Inductor-based backend with caching, piecewise compilation, shape specialization, and custom passes"）——piecewise（分片）编译允许模型被切成多段分别编译、互不连累，某一段追踪失败可以单独退回 eager 而不必让整个前向都跑不起来。这个差异背后是模型覆盖面的差异：Tokasaurus 只需要在 Llama/Qwen2/Qwen3 这几种确定的架构上验证过 `fullgraph=True` 能过，而 vLLM 要覆盖的模型架构、注意力后端、量化方案组合数远超单人/小团队能逐个验证"能不能整图编译"的量级，piecewise 是它必须付的通用性代价，也是它能对"随便一个新模型接进来就能跑"这个承诺负责的方式。

## 7. 踩坑与反直觉

1. **`response_format` 是个死字段。** `ChatCompletionRequest.response_format`（`tokasaurus/server/types.py:82`）声明了 `json_schema`/`json_object` 两种结构化输出模式，但全仓库搜索这个字段名，除了 `server/types.py` 里的定义之外**没有第二处代码读取它**——协议层"看起来支持"结构化输出，实际上传了也不会生效，也不会报错提醒你它被忽略了。这是"协议兼容"和"功能实现"脱节的典型样本，和 `00-总览与阅读地图.md` §4 里 vLLM/SGLang 的 `ChatCompletionRequest` 三分之二字段是"各家私货"是同一类问题的另一种表现——这里更进一步，字段声明了但**连自家都不用**。
2. **"OpenAI 兼容"这四个字要打个问号。** `validate_args()`（`tokasaurus/server/utils.py:503`-`531`）会主动拒绝 `stream=True`、非默认 `top_p`/`frequency_penalty`/`presence_penalty`/`logit_bias`——这些都是 OpenAI 官方字段，客户端传了非默认值会直接收到 400，而不是静默忽略。`## 5` 决策 5 已经论证这是合理的范围收窄，但如果只看 `## 2` 的路由表以为它是"又一个 OpenAI 兼容服务"，实际接入时会在这几个参数上踩坑。
3. **`block_on_queues()` 依赖 `mp.Queue` 的私有属性 `_reader`。**（`tokasaurus/utils.py:488`）标准库 `multiprocessing.Queue` 并没有公开暴露"给我这个队列底层的可等待 fd/连接对象"这个接口，Tokasaurus 直接读了 CPython 实现细节里的 `_reader` 属性来配合 `multiprocessing.connection.wait()` 做多队列多路复用。能工作，但是脆弱的私有 API 依赖——`## 8` 会给出具体建议。
4. **调度算法的正式性声明很诚实。** `tokasaurus/manager/scheduler.py:588` 那句注释——"Bro trust me, it's a good scheduler. I sketched it out on my iPad bro. Graphs and everything. This scheduler slaps."——不是本篇瞎编，是原作者自己写在源码里的。这句玩笑话背后的信息是准确的：这套调度算法没有附带正式的最优性证明，是一套经验上有效的启发式模拟（`calc_block_usage_over_time`/`try_onboarding_seqs`），读代码时不要误以为它有理论保证。
5. **Hydragen 和 CUDA Graph 不能同时开到任意批大小。** `tokasaurus/common_types.py:171`-`174` 的 `finalize()` 里有一条断言：`cudagraph_max_size` 必须小于 `hydragen_min_group_size`，否则直接在启动期报错退出。这是两个特性各自实现时没有考虑对方边界情况的产物——工程上选择用一个启动期断言挡住不兼容组合，而不是让两条路径在实现层面真正兼容，是小团队研究引擎里典型的"用契约代替泛化"取舍。
6. **仓库里留着两个从没被调用过的"拉起 vLLM/SGLang 对比服务器"的函数。** `sglang_manager()`（`tokasaurus/utils.py:345`）和 `vllm_manager()`（`tokasaurus/utils.py:376`）都是完整实现的 `@contextmanager`，用 `subprocess.Popen` 分别拉起 `python -m sglang.launch_server`/`python vllm_server.py` 作为对照基线服务——但在这个快照的全仓库范围内搜索这两个函数名，除定义处外**没有任何调用点**（源码为证）。这两个函数的存在本身就是证据：作者确实搭过和 vLLM/SGLang 做直接对比的基准测试基础设施；它们为什么在当前快照里成了无人调用的死代码——是被哪个基准脚本移除了、还是从未提交调用方——**未查证**。
7. **数据并行的负载均衡只数"在途请求数"，不看请求有多重。** `submit_request()`（`tokasaurus/server/utils.py:230`-`233`）选副本的唯一依据是 `state.requests_per_engine`（一个纯计数数组），一个 `n=1024` 的重复采样请求和一个 `n=1` 的单次请求在这个计数器里权重完全一样。在"大批量"这个目标场景下，如果客户端把几个大 `n` 请求集中发给同一个副本，各副本实际算力负载可能严重不均——但计数器上看起来是"均衡"的。`## 8` 给出具体改进方向。

## 8. 可改进点

1. **清理或补全 `response_format` 字段**（`tokasaurus/server/types.py:56`-`66`、`:82`）。既然协议层已经声明了 `json_schema`/`json_object` 两种结构化输出模式，要么接入 flashinfer 或其他库的约束解码实现让它真正生效，要么在字段旁加一行说明"当前版本忽略此字段"、或者在 `validate_args()` 里像拒绝 `logit_bias` 一样显式拒绝非 `None` 的 `response_format`——现状是"传了不报错也不生效"，对使用者是最容易踩的坑。
2. **补一个只读的 HTTP metrics 端点。** 当前的可观测性完全依赖 `statsd`（`tokasaurus/manager/monitoring.py:86`-`100` 的 `_statsd_client()`/`log_to_statsd()`），10 条路由（`## 2`）里没有一条是 Prometheus 风格的 `/metrics`。`StatsTracker`（`tokasaurus/manager/monitoring.py:121`）已经把吞吐/Hydragen 命中率/KV 利用率这些数据都算出来了，只是没有暴露一个不依赖外部 statsd 基础设施就能看到的读接口——对没有搭 statsd 收集链路的用户，运行时几乎是黑盒。
3. **给 `block_on_queues()` 的私有 API 依赖加一层显式检查或替代实现。**（`tokasaurus/utils.py:488`）依赖 `mp.Queue._reader` 这种 CPython 实现细节，一旦某个 Python 版本改了 `multiprocessing` 内部实现就可能静默失效（不报错，只是不再正确等待）。可以在启动期做一次自检（确认 `_reader` 属性存在且行为符合预期），或者迁移到标准库公开支持的多路复用方式（比如给每个队列配一对 `os.pipe` 做唤醒信号）。
4. **DP 负载均衡引入请求"重量"而不只是计数。**（`tokasaurus/server/utils.py:230`-`233`）`requests_per_engine` 目前对一个 `n=1` 的请求和一个 `n=1024` 的重复采样请求一视同仁地计数为 1，可以简单地把计数换成"请求预计贡献的 token 量"（`prompt tokens × n` 或结合 `max_tokens`），或者直接复用 manager 侧已经在算的 KV 占用/在跑序列数作为负载信号——这些数据 `StatsTracker`（`tokasaurus/manager/monitoring.py:121`）已经有，只是没有反馈回 server 侧的选路逻辑。

## 9. 自测题与延伸阅读

**自测题**

1. Tokasaurus 跑一个单卡请求需要几个进程角色？各自对应哪个代码目录，分别多少行？三者之间怎么通信？（提示：`## 1`、`## 2`）
2. Hydragen 到底省的是什么、不省的是什么？它和 vLLM 前缀缓存、SGLang RadixAttention 的分工差异，分别对应哪一行代码？（提示：`## 5` 决策 4、`## 6`）
3. `scheduling_steps_ahead` 存在的原因是什么？如果把它设成 1（每步都重新调度），最先受影响的会是哪个指标？为什么需要配合 `EarlyStoppingTracker` 一起用？（提示：`## 5` 决策 3）
4. "调度器只有 684 行"这句话为什么会误导人？要完整回答"调度到底花了多少行代码"，需要把哪几个文件加起来？和 vLLM 对应功能的行数差距，有多少是"泛化税"、有多少是"没做这件事"？（提示：`## 6`）
5. `validate_args()` 拒绝了哪几类采样参数？把这几类参数和 `benchmarks/monkeys_*.py` 里的典型工作负载对照，为什么拒绝反而是合理的设计取舍？（提示：`## 5` 决策 5、`## 7` 踩坑 2）
6. Tokasaurus 有没有自己的 CUDA/Triton kernel？它的 attention 算子最终调用的是哪个外部库、哪几个函数？这个选择在 `_lab/repo_stats.json` 里留下了什么可验证的痕迹？（提示：`## 2`、`## 5` 决策 6）
7. `torch.compile(model, fullgraph=True, dynamic=True)` 和 vLLM 默认的 `VLLM_COMPILE` 编译模式，根本差异是什么？为什么 Tokasaurus 敢用 `fullgraph=True`，而 vLLM 不敢？（提示：`## 5` 决策 7、`## 6`"编译策略"一节）
8. `submit_request()` 选数据并行副本时用的是什么信号？这个信号在什么情况下会失真？（提示：`## 3`、`## 7` 踩坑 7、`## 8`）
9. `sglang_manager()`/`vllm_manager()` 这两个函数说明了什么？为什么它们的存在本身就是一条证据，即便现在没人调用？（提示：`## 7` 踩坑 6）

**延伸阅读**

- [[00-总览与阅读地图]]——本库立场与已知硬结论，理解本篇多处"反直觉"结论为何值得强调的背景
- [[_PLAN]]——十段式骨架与双链名册，续写本库其它篇目前先读这个
- [[03-vLLM-调度器解剖]]——`## 6` 同位对照里 vLLM `Scheduler`/`SchedulerInterface`/`RequestQueue` 的完整实现细节，想深入对比可从这篇入手（规划中）
- [[03-SGLang-RadixAttention与前缀缓存]]——`## 6` 三线对比里 SGLang `RadixCache` 的完整实现细节（规划中）
