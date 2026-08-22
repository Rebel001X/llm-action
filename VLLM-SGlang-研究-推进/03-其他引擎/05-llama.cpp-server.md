# llama.cpp 的 server：给单机和端侧的推理引擎

> **本篇取证基准**：`llama.cpp` @ `2c6b141e`（2026-08-22）
> **一句话**：slot 是固定车位，不是 vLLM 那种能随时改道的弹性车道。

## 0. 结论先行

- **这是本库 12 个引擎里唯一一个"跑起来可以不装 Python"的服务进程。** `_lab/out/repo_stats.json` 统计整个仓库 992,377 行代码里 Python 只有 69,892 行（产品码 58,859 行），而且这些 Python **全部是离线脚本**——`convert_hf_to_gguf.py` 之类的格式转换工具和测试。`llama-server` 二进制本身由 C++（437,312 行）+ C/C++ 头文件（225,712 行）+ C（74,545 行）编译而成，请求处理路径上不存在解释器。这不是"轻量化"这种形容词能带过的事，`## 5` 会把它拆成一个真正的设计决策来核代价。
- **ggml 的后端注册表里有 16 条独立的 `register_backend()` 调用**（`ggml/src/ggml-backend-reg.cpp:119`-`173`），对应 CUDA、Metal、SYCL、Vulkan、WebGPU、zDNN、virtgpu、OpenCL、ZenDNN、Hexagon、CANN、BLAS、RPC、OpenVINO、ET、CPU 十六条硬件路径。但目录数其实是 18 个（`ggml/src/ggml-hip/`、`ggml/src/ggml-musa/` 也在），因为 AMD HIP 和摩尔线程 MUSA 走的是"把 CUDA 源码转译后仍然编译进 `GGML_USE_CUDA` 宏"这条路（`ggml/src/ggml-hip/CMakeLists.txt:92`、`ggml/src/ggml-musa/CMakeLists.txt:74` 都写着 `target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)`），不是各自独立的注册函数——这是本篇第一个"数字对不上直觉"的地方。
- **server 的并发单位是固定数量的 `server_slot`（`-np`/`--parallel`），不是按显存块动态生长的请求池。** 默认 `n_parallel = -1`（自动），server 启动时如果没显式指定就定死成 4（`tools/server/server.cpp:152`-`156`），且总上下文长度会被硬性均分给每个 slot：`cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max`（`src/llama-context.cpp:293`）——除非开 `--kv-unified`，否则一个 slot 用不完的上下文，另一个 slot 借不到。
- **slot 之间的 KV 复用靠一次线性最长公共前缀（LCP）扫描，不是基数树或哈希表。** `get_available_slot`（`tools/server/server-context.cpp:1500`）遍历所有空闲 slot，对每个 slot 现存的 token 序列和新请求做 `tokens.get_common_prefix(task.tokens)`（`:1531`），选相似度最高且过阈值（默认 0.1，`common/common.h:678`）的那个；找不到就退化成 LRU（`:1565`-`1579`）。这一步是 `O(n_slots)` 的，n_slots 通常个位数到几十，量级上和 SGLang RadixCache 的树查找不是一回事，但因为 slot 数天生就小，这个"笨办法"反而够用。
- **还有一层容易被忽略的兜底：8 GiB 默认开启的主机内存提示词缓存。** `--cache-ram` 默认值是 `8192`（MiB）（`common/common.h:616`），不是 0——也就是说 llama-server **默认就会**把被挤出 slot 的旧提示词的 KV 状态序列化进普通内存（`server_prompt_cache`，`tools/server/server-task.h:613`），下次同一个前缀回来时优先从这里恢复而不是重算，代价是主机 RAM 而非显存。
- **`_lab/api_surface.py` 用正则而非 AST 抽 llama.cpp 的路由（`method=regex`），这份统计低估了真实路由数。** 它只识别 `ctx_http.post(...)` 这一种调用形态，抽出 36 条、全部标成 `POST`——但 `tools/server/server.cpp` 里同样密集地写着 `ctx_http.get (...)` 和 `ctx_http.del (...)`（比如 `GET /health` 在 `:234`、`GET /props` 在 `:237`、`GET /slots` 在 `:273`），这些一条都没被正则捕到。`## 2` `## 7` 会把这个精度缺口摊开讲。
- **`server-mcp.cpp` 让 llama-server 长出了 MCP 能力，但方向和"MCP server"这个名字暗示的相反。** 它实现的是**MCP 客户端/宿主**：按 Cursor 兼容的 JSON 配置拉起外部 MCP server 子进程（stdio 传输），列出它们的工具、在模型调用工具时按需转发（`tools/server/server-mcp.h:50` 的 `server_mcp_transport`、`:132` 的 `server_mcp` 类），再把这些工具和内置工具一起挂到 `GET /tools`（`tools/server/server.cpp:347`-`348`）。**llama-server 本身并不对外暴露一个 MCP 协议端点供别的 MCP 客户端连接**——全仓库搜不到任何 `/mcp` 这样的 JSON-RPC over HTTP 路由。这是本篇取证过程里最反直觉的一点。
- **GGUF 的量化类型远不止"Q4 还是 Q8"这么简单。** `ggml_type` 枚举定义了 35 个存活类型（`GGML_TYPE_COUNT=43` 是数组哨兵，中间跳过 8 个历史废弃占位，`ggml/include/ggml.h:392`-`433`），其中 27 个是权重量化格式，分属三个机制完全不同的家族（传统线性量化、K-quant 分层量化、I-quant 码本量化，`## 4` 逐一拆）；上层暴露给 CLI 的 `llama_ftype` 有 36 个预设（如 `Q4_K_M`），每个预设不是"全模型统一比特宽度"，而是一张按张量角色分配不同类型的**混合表**（`src/llama-quant.cpp:608`-`613`）。
- **它和 vLLM 不是同一类东西，源码层面能给出至少四条独立证据。** 没有跨进程/跨机器的调度器-执行器分层（一次 `update_slots()` 里只有一处 `llama_decode()` 调用，`:3612`，所有活跃 slot 的下一步 token 拼进同一个 batch 一起跑）；没有 PD 分离的代码痕迹；仓库里唯一的分布式路径（`ggml-rpc`）在文档里被作者自己标注为"proof-of-concept…fragile and insecure…Never run on an open network"（`tools/rpc/README.md:4`-`5`）；"router mode"做的是同一台机器上管理多个模型进程的生命周期（`tools/server/server-models.h:113` 的 `server_models`），不是跨节点的集群编排。`## 6` 会逐条对照 vLLM 给出反例。

## 1. 它在系统里的位置

`llama.cpp` 的 README 把目标说得很直接："The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware"，紧接着第一条特性就是"Plain C/C++ implementation without any dependencies"（`README.md:54`、`:57`）。快速上手的两条命令也写在 README 里——`llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF` 直接从 Hugging Face 拉模型跑命令行，`llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF` 拉起 OpenAI 兼容的 API server（`README.md:19`-`24`）——一条命令、一个二进制、不需要提前起 Python 环境或装 CUDA。

这决定了它在本库 12 个引擎里的坐标：vLLM、SGLang、TensorRT-LLM、Dynamo 这些引擎默认工作点是"一台或多台装好驱动的服务器 GPU，追求高并发吞吐"；llama.cpp 的默认工作点是"一台可能没有独立显卡的个人电脑或边缘设备，追求装得上、跑得起来、单用户体验可接受的延迟"。这两个目标不是同一目标函数下的两个刻度，而是两个不同的优化方向——`## 6` 会把这句话落到具体代码证据上，不停留在口号层面。

仓库规模上，`_lab/out/repo_stats.json` 的 `top_dirs` 显示最大的目录是 `ggml/src`（364,133 行/819 文件）——这是硬件后端层，比第二名 `tools/ui`（110,392 行/428 文件，自带的 TypeScript Web UI）大三倍还多；`tools/server`（34,611 行/71 文件）才是本篇主角所在的目录，规模上排第七。这个排序本身就说明了 llama.cpp 的资产重心：**打通尽可能多的硬件**比**server 本身的调度复杂度**占的代码量大得多，这和 vLLM 把最大代码量堆在 `model_executor`+`v1` 引擎循环（据 [[01-vLLM-全景与代码地图]]）是完全不同的分布形状。

测试/产品比 0.187（`_lab/out/repo_stats.json` 的 `totals.test_to_src_ratio`）——这个数字本身不能孤立地读成"测试不够"，因为 `tests/`（45,555 行/56 文件）里相当一部分是逐算子的数值对拍脚本（`test-quantize-fns.cpp`、`test-backend-ops.cpp` 这类），验证的是"CPU 实现和 CUDA 实现算出同一个答案"，不是传统意义上的业务逻辑单测,这一点标注为**本库推断**，未逐文件核对每个测试文件的性质。

## 2. 代码地图（文件 → 职责，带行号）

按"一次请求会依次经过的层"排列：

| 层 | 文件:行 | 干什么 |
|---|---|---|
| 路由装配 | `tools/server/server.cpp:234`-`240` | `ctx_http.get("/health", ...)`、`GET /props`、`GET /models` 等只读端点注册；这些调用**没有**被 `_lab/api_surface.py` 的正则抽到 |
| 路由装配 | `tools/server/server.cpp:245` | `ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions))`——OpenAI 协议主入口 |
| 路由装配 | `tools/server/server.cpp:251` | `ctx_http.post("/v1/messages", ...)`——Anthropic Messages API 兼容端点，注释直接写"anthropic messages API" |
| 路由装配 | `tools/server/server.cpp:270`-`274` | LoRA 热插拔（`GET`/`POST /lora-adapters`）与 slot 存盘/恢复（`GET /slots`、`POST /slots/:id_slot`） |
| 路由装配（工具） | `tools/server/server.cpp:347`-`348` | 仅当 `--tools` 或配置了 MCP server 时才注册 `GET`/`POST /tools`，否则返回 403（`:359`-`360`） |
| 启动期参数解析 | `tools/server/server.cpp:152`-`156` | `n_parallel < 0` 时自动定为 4 且强制 `kv_unified = true`——这是"不显式设置 `-np` 时到底有几个 slot"的唯一真源 |
| slot 数据结构 | `tools/server/server-context.cpp:196` | `struct server_slot` 定义：一个 slot 同时持有目标模型上下文 `ctx_tgt`、可选的投机解码草稿上下文 `ctx_dft`、采样器、生成状态 |
| slot 生命周期 | `tools/server/server-context.cpp:58`-`63` | `slot_state` 枚举：`IDLE → WAIT_OTHER/STARTED → PROCESSING_PROMPT → DONE_PROMPT → GENERATING` |
| slot 选择/复用 | `tools/server/server-context.cpp:1500` | `get_available_slot`：先按 `task.id_slot` 精确指定，再按 LCP 相似度找可复用 slot，最后退化到 LRU |
| slot 选择/复用 | `tools/server/server-context.cpp:1565`-`1579` | LRU 兜底分支：`slot.t_last_used` 最小者胜出 |
| 任务派发 | `tools/server/server-context.cpp:2328`-`2339` | 拿不到可用 slot 时 `queue_tasks.defer(std::move(task))`，**不是拒绝请求** |
| 任务队列 | `tools/server/server-queue.cpp:90`-`108` | `pop_deferred_task`：slot 一释放，优先弹出等着这个 slot id 的任务，否则弹队首（FIFO） |
| 主循环 | `tools/server/server-queue.cpp:278`-`299` | `server_queue::start_loop`：单一循环，处理新任务 → `callback_update_slots()` 跑一次推理步 → 等待新任务 |
| 批处理执行 | `tools/server/server-context.cpp:3612` | 一次 `update_slots()` 里**唯一**的 `llama_decode(ctx_tgt, batch_view)` 调用——所有活跃 slot 的下一步 token 被拼进同一个 batch |
| HTTP 并发模型 | `tools/server/server-http.cpp:309`-`320` | `n_threads_http` 默认 `max(n_parallel+4, hardware_concurrency()-1)`，用 `httplib::ThreadPool` 处理 HTTP 层；但推理主循环仍是单线程 |
| 提示词缓存 | `tools/server/server-task.h:613`-`630` | `server_prompt_cache`：主机内存里的 KV 状态归档，`limit_size`/`limit_tokens` 两个上限 |
| 提示词缓存开关 | `common/common.h:616` | `cache_ram_mib = 8192`——**默认开启**，不是 opt-in |
| MCP 客户端 | `tools/server/server-mcp.h:50` | `server_mcp_transport`：一个 MCP server 子进程会话（stdio 读写 JSON-RPC） |
| MCP 客户端 | `tools/server/server-mcp.h:132` | `class server_mcp`：管理多个 `server_mcp_transport`，聚合它们各自暴露的工具 |
| 多模型进程管理 | `tools/server/server-models.h:113`、`:323` | `server_models`（子进程生命周期状态机 DOWNLOADING→LOADED→SLEEPING）与 `server_models_routes`（router 模式下的代理路由） |
| ggml 后端注册 | `ggml/src/ggml-backend-reg.cpp:119`-`173` | `ggml_backend_registry()` 构造函数：16 条 `#ifdef GGML_USE_* / register_backend(...)` |
| 设备类型定义 | `ggml/include/ggml-backend.h:134`-`144` | `ggml_backend_dev_type` 枚举：`CPU`/`GPU`/`IGPU`/`ACCEL`/`META` 五种设备角色 |
| GPU→CPU 兜底 | `src/llama-model.cpp:924`、`:986`、`:1322` | `make_cpu_buft_list`、`make_gpu_buft_list` 两个函数，及"add CPU buffer types as a fallback"的实际落点 |
| 上下文分片 | `src/llama-context.cpp:288`-`293` | `kv_unified` 为假时 `n_ctx_seq = n_ctx / n_seq_max`，把总上下文硬性均分给每个 slot |
| 模型架构注册 | `src/llama-model.cpp:46`-`50` | `switch (arch)` 按 `LLM_ARCH_LLAMA` 等分发到 `src/models/*.cpp` 里手写的图构建代码 |
| 量化混合表 | `src/llama-quant.cpp:608`-`613` | `LLAMA_FTYPE_MOSTLY_Q4_K_M` 预设下，不同层的 `attn_v`/`ffn_down` 张量按 `use_more_bits` 分配 `Q4_K`/`Q5_K`/`Q6_K` |
| 分布式（PoC） | `tools/rpc/README.md:4`-`5` | `ggml-rpc-server` 的官方警告："proof-of-concept…never run on an open network" |

以上 28 条只是骨架，`## 3` `## 4` 会把其中若干条的上下文摊开细读。

## 3. 核心数据结构

- **`server_slot`**（`tools/server/server-context.cpp:196`）——server 并发的基本单位。除了 `ctx_tgt`/`ctx_dft`（目标/草稿模型的 `llama_context`）、`smpl`（采样器）这些显而易见的字段，还有两个容易被忽略但很关键的：`t_last_used`（LRU 判据的唯一依据，`:220`）和 `prompt`（`server_prompt` 类型，装着这个 slot 当前缓存的 token 序列和检查点，`:253`）。`can_batch_with`（`:390`-`396`）判断两个 slot 能否被合并进同一次 `llama_decode`（要求任务类型相同、embedding 维度相同、LoRA 配置相同），是"多 slot 共享一次 batch"这件事在代码里的直接开关。
- **`slot_state`**（`tools/server/server-context.cpp:58`-`63`）——六态状态机：`IDLE`、`WAIT_OTHER`（子任务等父任务先处理完 prompt，用于多任务并发场景）、`STARTED`、`PROCESSING_PROMPT`、`DONE_PROMPT`、`GENERATING`。和 vLLM 请求状态（`WAITING`/`RUNNING`/`PREEMPTED`/`FINISHED` 等，见 [[03-vLLM-调度器解剖]]）比,llama.cpp 把"prompt 处理"和"逐 token 生成"拆成了两个显式状态,这是因为 chunked prefill 在这里是**同一个 slot 内部**的多步循环,不像 vLLM 那样和 decode 请求混进同一次调度决策。
- **`server_prompt` / `server_prompt_cache`**（`tools/server/server-task.h:567`、`:613`）——前者是"一个 slot 当前记住的 token 序列 + checkpoint 列表"，后者是**主机内存**里的归档区,`alloc`/`load`/`update` 三个方法实现"slot 被新请求挤占前先把旧状态序列化进这里,下次同前缀请求回来时尝试恢复"。这是 GPU 显存之外的第二层缓存,默认容量 8 GiB（`common/common.h:616`）。
- **`ggml_backend_dev_type`**（`ggml/include/ggml-backend.h:134`-`144`）——五值枚举:`CPU`、`GPU`（独显）、`IGPU`（核显,用主机内存）、`ACCEL`（配合 CPU 使用的加速器,如 BLAS/AMX）、`META`（包装多个设备做张量并行的"元设备"）。这个枚举是整个后端选择/回退逻辑的类型基础——`## 4` 会看到它在张量分配时怎么被用。
- **`ggml_type`**（`ggml/include/ggml.h:392`-`433`）——张量的存储类型,35 个存活值,27 个是量化格式。三个子族的代表结构体:
  - `block_q4_0`（`ggml/src/ggml-common.h:194`-`198`）——**传统线性量化**:32 个元素一块,一个 `ggml_half`（fp16）当缩放因子 `d`,`x = a*q`,没有偏移量。
  - `block_q4_K`（`ggml/src/ggml-common.h:323`-`338`）——**K-quant 分层量化**:256 个元素为一个超级块（`QK_K=256`,`ggml/src/ggml-common.h:89`）,内部再切成 8 个 32 元素子块;超级块级别存两个 fp16（`d`/`dmin`,分别是"子块缩放的缩放"和"子块偏移的缩放"）,子块级别的具体缩放/偏移量只用 6 bit 存进 `scales[K_SCALE_SIZE]`——**缩放本身也被量化了**,这是"K"系列比传统线性量化多出的一层间接。公式变成 `x = a*q + b`,比 `Q4_0` 多了偏移项。
  - `block_iq2_xxs`（`ggml/src/ggml-common.h:381`-`385`）——**I-quant 码本量化**:每个 `qs` 元素不是直接存的量化值,而是一个索引,指向编译期烧进二进制的定长网格表（`kgrid_2bit_256`/`kgrid_2bit_512` 等,`ggml/src/ggml-quants.c:2859`起）;`kgrid = type == GGML_TYPE_IQ2_XXS ? kgrid_2bit_256 : ...`（`ggml/src/ggml-quants.c:3111`-`3113`）按类型选表。这是三个家族里**唯一需要额外静态数据**的一种,换来的是在极低比特（2 bit 上下）时比线性量化更接近真实权重分布。
- **`llama_ftype`**（`include/llama.h:116`-`158`）——CLI 层面的量化预设枚举,36 个可用值（如 `LLAMA_FTYPE_MOSTLY_Q4_K_M = 15`）。每个预设不直接等于某个 `ggml_type`,而是在 `src/llama-quant.cpp` 里驱动一张"哪个张量用哪个 `ggml_type`"的分配表——细节见 `## 4`。

## 4. 主流程走读

### 4.1 启动:从命令行参数到"能收请求"

1. **模式判定。** `main()`（`tools/server/server.cpp`）先判断是不是 router 模式:`is_router_server`——没给 `--model`/`--hf-repo`/`--docker-repo` 时为真(此时进程只做多模型代理,不碰 GPU,注释原文"router server never loads a model and must not touch the GPU",`server.cpp:135`-`137`)。
2. **slot 数量落定。** 非 router 模式下,`params.n_parallel < 0`(默认值)会被改写成 4,同时强制 `kv_unified = true`(`server.cpp:152`-`156`)——这两行决定了"什么都不配置直接起服务"时到底有几路并发、上下文怎么分。
3. **路由表装配。** `server_routes routes(params, ctx_server)` 构造出所有 handler,随后是一长串 `ctx_http.get(...)`/`ctx_http.post(...)`/`ctx_http.del(...)` 调用(`server.cpp:234`起)。是否注册 `/tools`、`/cors-proxy` 取决于运行时开关(`--tools`、`--ui-mcp-proxy`),关闭时对应路径返回 403 而不是 404(`server.cpp:329`-`330`、`:359`-`360`)——**路由表本身是静态的,但 handler 行为按配置分叉**。
4. **模型加载与后端选择。** 模型加载路径(`src/llama-model.cpp`)会先枚举 `ggml_backend_dev_count()` 遍历到的所有设备,分别调用 `make_gpu_buft_list`(`:986`)和 `make_cpu_buft_list`(`:924`)为每个设备建一份"这个设备支持存哪些 buffer 类型"的列表;关键的一步是 `make_gpu_buft_list` 返回之后,调用方紧接着把 `pimpl->cpu_buft_list` 拼接到每个 GPU 设备自己的 buft 列表末尾,注释原话"add CPU buffer types as a fallback"(`:1322`-`1324`)——**每个张量在放不进 GPU 显存时,天然有一条退到 CPU 内存的路,不需要额外的异常处理分支**,这是本篇 `## 5` 要展开的第一个设计决策。
5. **slot 初始化。** `for (int i = 0; i < params_base.n_parallel; i++) slots.emplace_back()`(`server-context.cpp:1233`起),每个 slot 拿到相同的 `n_ctx_slot = llama_n_ctx_seq(ctx_tgt)`(`:1211`,若超过训练长度会被截断并打警告),意味着**同一次启动里所有 slot 的容量是一致的**,没有"给这个 slot 多分点上下文"的运行时调节。

### 4.2 一次 `/v1/chat/completions` 请求怎么走

1. **HTTP 层接住请求。** `httplib::ThreadPool` 里的某个工作线程(池大小由 `n_threads_http` 决定,默认 `max(n_parallel+4, hardware_concurrency()-1)`,`server-http.cpp:309`-`320`)解析出 JSON body,调用挂在 `/v1/chat/completions` 上的 `routes.post_chat_completions`(`server.cpp:245`)。这一层**是**多线程的——HTTP 解析、JSON 反序列化、chat template 渲染可以在多个请求间并行进行。
2. **打包成任务、入队。** handler 把请求转成一个或多个 `server_task`,通过 `server_queue::post` 塞进 `queue_tasks`(`std::deque`,`tools/server/server-queue.h:24`),然后阻塞等待结果(通过 `server_response_reader` 轮询)。
3. **单一主循环被唤醒。** `server_queue::start_loop`(`server-queue.cpp:278`)是**全进程唯一**跑推理的循环:`process_new_tasks(false)` 把新任务从队列搬进内部状态,然后调用 `callback_update_slots()`——这一步是同步的、单线程的,"HTTP 层多线程、推理主循环单线程"是这个进程真实的并发形状。
4. **挑 slot。** `update_slots()` 内部对每个待处理任务调 `get_available_slot(task)`(`server-context.cpp:1500`):
   - 若 `slot_prompt_similarity`(默认 0.1,`common/common.h:678`)不为 0,对所有空闲 slot 算 `tokens.get_common_prefix(task.tokens)` 得到最长公共前缀长度,取相似度(前缀长度/请求长度)最高且过阈值的那个(`:1513`-`1544`);
   - 否则(或没找到)退化成"选 `t_last_used` 最小的空闲 slot"(LRU,`:1565`-`1579`);
   - 若命中的 slot 即将丢失超过一半的既有上下文(`f_keep < 0.5`),先把它当前状态存进 `server_prompt_cache`(主机内存)再覆盖(`:1554`-`1558`、`:1582`-`1601`)。
   - 若所有 slot 都在忙,`get_available_slot` 返回空,任务被 `queue_tasks.defer(...)` 送进 `queue_tasks_deferred`(`server-context.cpp:2334`-`2337`),**不是被拒绝**;下次某个 slot 释放时,`pop_deferred_task(id_slot)`(`server-queue.cpp:90`)优先把等着这个 slot id 的任务捞回来,否则捞队首那个(FIFO,`:104`-`108`)。默认没有队列深度上限,只有显式带 `fail_on_no_slot=true` 查询参数请求 `GET /slots` 时才会看到 503(`server-context.cpp:4675`-`4681`)——**默认路径下请求永远排队,不会因为忙就被拒**。
5. **一次批处理。** 拿到 slot 的任务被推进(prompt 处理或采样一步),所有当前活跃 slot 各自贡献的 token 通过 `common_batch_add`(`server-queue.cpp:164`)拼进同一个 `llama_batch`,最终整个 `update_slots()` 里**只有一处** `llama_decode(ctx_tgt, batch_view)` 调用(`server-context.cpp:3612`)——这一步用 `queue_tasks.yield_to_queue(...)` 包起来,好让 metrics 一类的元任务能在解码等待期间被处理,但解码本身是这一步唯一的重活。
6. **结果流式返回。** 采样出新 token 后,对应 slot 的 handler 通过 SSE 把增量结果写回 HTTP 响应流;当某个 slot 的任务完成(命中停止条件或达到 `n_predict_max`),`state` 回到 `SLOT_STATE_IDLE`(`server-context.cpp:508`),等待下一次 `get_available_slot` 把它选中。

### 4.3 GGUF 量化:三条路线,一张混合表

三个量化家族在 `## 3` 已经给出结构体证据,这里补机制层面的关键差异:

- **传统线性量化**(`Q4_0`/`Q5_0`/`Q8_0` 等)是最简单的每块一个缩放因子,`x = a*q`,解码只需一次乘法,没有分层结构。
- **K-quant**(`Q2_K`…`Q6_K`)把 256 个元素分成 8 个子块,子块各自的缩放/偏移量本身又被压缩进 6 bit(`K_SCALE_SIZE`,`ggml/src/ggml-common.h:337` 附近),换来的是"用更少的额外存储,逼近逐通道量化的精度",代价是解码路径多一层间接寻址。
- **I-quant**(`IQ1_S`…`IQ4_XS`)在 2-4 bit 区间用预训练好的码本网格代替逐元素线性量化,`kgrid_2bit_256`/`kgrid_1bit_2048` 这些常量表(`ggml/src/ggml-quants.c:2859`起)是编译期烧进二进制的,不随模型变化——这意味着**量化精度和这些码本的设计强相关**,不是简单调一个缩放因子能改善的。

而 CLI 层面用户敲的 `Q4_K_M` 这种预设名,并不对应"整模型统一用 `GGML_TYPE_Q4_K`"。`src/llama-quant.cpp:608`-`613` 的分支显示,`LLAMA_FTYPE_MOSTLY_Q4_K_M` 下,注意力/FFN 的某些张量按层深度(`use_more_bits`,`:430`-`431`:处于最前 1/8、最后 1/8,或每 3 层里的特定一层)被升到 `Q5_K`,再往上一级的 `attn_v`/`ffn_down` 甚至可能升到 `Q6_K`——这是一张**按张量角色和层位置分配精度**的混合表,`M`/`S`/`L` 后缀描述的是这张表整体偏保守还是偏激进,不是一个统一比特宽度的代号。`--pure` 选项(`tools/quantize/README.md` 描述)会关掉这套混合逻辑,强制所有张量用同一种类型。

## 5. 设计决策与代价

### 决策一:纯 C/C++ 单体,零 Python 依赖

- **为什么这么设计**:README 把这条列为第一特性——"Plain C/C++ implementation without any dependencies"(`README.md:57`)。目标场景是"minimal setup"(`README.md:54`),一个不依赖运行时解释器、不需要提前装 CUDA/cuDNN/PyTorch 的二进制,可以直接拷到一台没联网、没 Python 环境、甚至没有独立显卡的机器上跑。这不是性能优化,是**部署面**的优化:`llama serve -hf <repo>` 一条命令背后不需要 `pip install` 任何东西,C++ 编译产物本身就是完整的可执行文件。
- **不这样会怎样**:vLLM/SGLang 的部署单位事实上是"Python 环境 + CUDA 驱动 + 一堆 pip 依赖"(可参照 [[01-vLLM-全景与代码地图]] 里 Python 产品码 932,277 行的规模),换一台机器意味着要么装同样的环境,要么发一个几 GB 的容器镜像。llama.cpp 反过来:`llama.cpp` 的 Python 代码(69,892 行)只在**转换模型格式**(`convert_hf_to_gguf.py`)和跑测试时才被用到——这些是构建期/准备期工作,不在 server 的请求处理路径上。**推理请求从进入到离开,不经过任何 Python 代码**,这是本篇开头说"和本库其他引擎完全不同"的最直接来源。
- **什么时候这是负担**:模型架构支持完全靠手写 C++。`src/models/` 下有 151 个文件(`_lab/out/repo_stats.json` 的 `top_dirs`),每个新架构都要一个新的 `.cpp` 文件手写图构建代码(比如 `src/models/llama.cpp`,250 行,定义 `load_arch_hparams`/`load_arch_tensors` 两个函数),再到 `src/llama-model.cpp:46`-`50` 这样的 `switch (arch)` 分支里手动接线。vLLM/SGLang 加一个新模型通常是抄一份 PyTorch 的 `transformers` 实现,Python 层的胶水代码相对轻;llama.cpp 加一个新模型意味着要懂 ggml 的计算图 API、手写量化感知的张量创建代码——`docs/development/HOWTO-add-model.md`(196 行)本身的存在就说明这不是一件"随手改改配置"的事。这也直接抬高了贡献门槛:能审到"这个新模型的图搭得对不对"的人,必须同时懂模型结构和 ggml 底层 API,比"读一遍 HuggingFace 实现"要求更高。

### 决策二:slot 是固定数量、上下文均分的静态资源,不是动态增长的请求池

- **为什么这么设计**:llama.cpp 的目标场景是单机、常常是单用户或少量并发用户,`n_parallel` 一旦定下就对应一份确定的内存/显存预算(`n_ctx_slot = n_ctx / n_parallel`,`src/llama-context.cpp:293`)——这让资源用量在**启动时就能算清楚**,不需要运行时的显存碎片管理、块表、驱逐算法这套复杂机制。对于端侧和消费级硬件,可预测的资源占用比"极限情况下的最优吞吐"更重要。
- **不这样会怎样**:如果要做到 vLLM 那种"按 block 动态分配、任意请求都能借用任意空闲显存"的粒度,需要引入完整的分页 KV cache 管理(块表、引用计数、驱逐策略)——这套机制本身在 vLLM 里对应了 `_lab/out/struct_map.json` 统计的 `kv_cache` 子系统(据 [[01-vLLM-全景与代码地图]],9,581 行/13 文件)。llama.cpp 选择不引入这层复杂度,代价是**上下文利用率在请求长度分布不均时会打折**——一个 slot 的一半上下文空着,另一个 slot 的请求因为长度不够用而被截断或拒绝,两者互不相通(除非手动开 `--kv-unified`)。
- **什么时候可以不这样**:`--kv-unified` 打开后(`common/common.h:563` 默认 `false`,但 `n_parallel` 自动模式下会被强制设成 `true`,`server.cpp:156`)`n_ctx_seq = n_ctx`(`src/llama-context.cpp:291`),意味着序列之间可以共享同一份完整长度的上下文预算而不是硬性均分——这是 llama.cpp 在"简单均分"和"完全动态"之间给出的一个折中开关,但仍然不等于逐 block 的动态分配,`未查证` `kv_unified=true` 时具体的底层内存布局是否已经做到块级共享,这一点留给以后核实,此处不下结论。

### 决策三:MCP 集成做成客户端/宿主,不做成服务端

- **为什么这么设计**:llama-server 本身已经是"一个能调用工具的 LLM 服务",让它去**连接**外部 MCP server(文件系统、搜索等能力提供方)、把工具结果喂给模型,比反过来**实现**一个 MCP 服务端等外部 Agent 框架来调用 llama-server,更贴合它"单机跑一个能干活的助手"这个使用场景。README 原话:"the server can expose tools coming from MCP servers"(`tools/server/README.md:347`-`349`),配置格式直接兼容 Cursor 的 `mcpServers` JSON(`:350`-`361`)——这是抄现成生态的配置格式,不是自造协议。
- **不这样会怎样**:如果做成 MCP 服务端,llama-server 要多维护一套 MCP 协议的 HTTP/SSE 传输层和会话状态,且这套服务端能力和它已有的 `/v1/chat/completions` 之类端点在语义上是两回事(一个是"被调用做工具",一个是"调用别人做工具")——现状是两者都没做,只做了后者(客户端角色)。README 也明确划了一条边界:"Note: `--ui-mcp-proxy` is unrelated, it only lets the Web UI reach remote MCP servers from the browser"(`tools/server/README.md:379`),说明连 Web UI 里那个"MCP 代理"都只是浏览器侧转发,不是 server 自己的 MCP 服务端实现。
- **什么时候可以不这样**:如果目标是把 llama-server 接入一个更大的多 Agent 编排系统(比如让 Claude Desktop 或某个 Agent 框架把 llama-server 当成一个可调用的 MCP 资源),现状的代码是做不到的——这需要额外实现 MCP 服务端协议,目前仓库里没有这部分代码,标注为**本库推断**下的一个明确缺口,`## 8` 会再提一次。

### 决策四:分布式仅有一条"proof-of-concept"级别的 RPC 路径,没有 PD 分离

- **为什么这么设计**:llama.cpp 的默认工作点是单机,`ggml-rpc`(`ggml/src/ggml-rpc/ggml-rpc.cpp`,2065 行)提供的是"把 ggml 设备通过 TCP 暴露给远程调用方"这个能力,用途是让一个大模型能跨机器切分张量/流水线并行来跑——这是"单次推理会话需要的显存超过一台机器"这个具体问题的答案,不是"多租户高并发服务"的答案。作者自己在文档里把它的成熟度标得很低:"currently in a proof-of-concept development stage…fragile and insecure. **Never run the RPC server on an open network or in a sensitive environment!**"(`tools/rpc/README.md:4`-`5`)。
- **不这样会怎样**:vLLM/SGLang/Dynamo 这类面向集群的引擎会实现请求级别的 KV 缓存跨节点传输(PD 分离,可参照篇目表里的 [[12-vLLM-PD分离与KV-Connector]]),让 prefill 和 decode 分别跑在专用节点上、通过高速互联传 KV 状态。llama.cpp 完全没有这层——`docs/`、`tools/` 全仓库搜不到"disaggregat"或"prefill/decode separation"相关的文档或代码痕迹(**源码为证**:grep 结果为空)。这意味着它没法在多机场景下做请求粒度的负载均衡与角色分工,只能做"router mode"这种同一台机器上多个模型进程的生命周期管理(`tools/server/server-models.h:113`)。
- **什么时候可以不这样**:如果确实需要多机分布式,`ggml-rpc` 是仓库里唯一的选项,但要接受它明确标注的"实验性、不安全"状态,且只解决"模型太大装不进一台机器"这个问题,不解决"这台集群要服务多少并发用户"的问题——这两个问题在 vLLM 的世界观里是分开被解决的(张量并行 vs PD 分离/多副本),在 llama.cpp 这里只有前者有对应实现。

## 6. 同位对照:vLLM 在同一位置怎么做

**并发单位:固定 slot vs 动态 block 池。** llama.cpp 的 `server_slot`(`tools/server/server-context.cpp:196`)数量在启动时由 `-np` 定死,每个 slot 的上下文容量也是启动时算好的固定值(`n_ctx / n_parallel`,`src/llama-context.cpp:293`)。vLLM 的 `Scheduler`(`` `vllm:vllm/v1/core/sched/scheduler.py:73` ``)不预分配"槽位",而是维护一个 `BlockPool`(`` `vllm:vllm/v1/core/block_pool.py:143` ``)——所有请求共享同一个物理块池,`free_block_queue`(`` `vllm:vllm/v1/core/block_pool.py:181` ``)按需分配、按 LRU 近似策略驱逐,一个请求能用多少块只取决于当前显存里还有多少空闲块,不取决于它落在"第几个 slot"。这是两种资源模型的根本区别:llama.cpp 是**固定车位**(车位数量和大小开工前定死,某个车位空着别的车也停不进去),vLLM 是**弹性车道**(车道总容量固定,但怎么分给谁是每一步都在重算的动态决策)。

**调度粒度:整批一次 decode vs 逐步细粒度调度。** llama.cpp 的 `update_slots()` 每一步只做一件事:把所有活跃 slot 的下一步 token 拼进一个 batch,调一次 `llama_decode`(`server-context.cpp:3612`)。vLLM 的 `Scheduler.schedule()`(`` `vllm:vllm/v1/core/sched/scheduler.py:484` ``)在每一步要决定"这一步谁的 prompt 继续处理、谁的 decode 继续、要不要触发抢占",本身就是一个比"把活跃 slot 的 token 拼一起"复杂得多的决策过程——因为 vLLM 面对的请求数可以远超"能同时塞进一次 batch"的 slot 数量,必须有一层显式的准入控制;llama.cpp 因为 slot 数量本来就小(通常个位数到几十),不需要这层复杂度,"活跃的都拼一起跑"就是全部逻辑。

**前缀复用:线性 LCP 扫描 vs 结构化索引。** llama.cpp 的 `get_available_slot` 对每个空闲 slot 做一次 `get_common_prefix` 比对(`server-context.cpp:1531`),是 `O(n_slots × prompt_len)` 的线性扫描。SGLang 用基数树(`RadixCache`,可参照 [[03-SGLang-RadixAttention与前缀缓存]])做结构化前缀索引,vLLM 用块哈希表(可参照 [[04-vLLM-KV缓存与前缀缓存]])。三者的复杂度差异在 n 很大时会拉开明显差距,但 llama.cpp 的 n(slot 数)天生就小,线性扫描在这个规模下并不是真实瓶颈——**这不是"llama.cpp 的前缀缓存实现得简陋",而是"这个规模下更复杂的数据结构收益不明显"**,是和它的目标场景匹配的选择,不是短板。

**满载时的行为:无限期排队 vs 显式准入控制。** llama.cpp 在所有 slot 都忙时把请求丢进 `queue_tasks_deferred`(`server-queue.cpp:79`),没有默认的队列深度上限,请求会一直等到有 slot 释放(`## 4.2` 已给出证据)。vLLM 的 `SchedulerConfig` 里的 `max_num_seqs`(`` `vllm:vllm/config/scheduler.py:63` ``)从准入层面就限制了同时处理的请求数,超出预算的请求走抢占(踢出正在跑的请求腾资源)或在调度器内部排队,这套逻辑本身是 `Scheduler` 代码量的重要组成部分。两边都"不会无理由拒绝请求",但 vLLM 的准入控制是运行时动态博弈的一部分,llama.cpp 的排队更接近"先来后到,能者上"的简单队列语义。

## 7. 踩坑与反直觉

- **`_lab/out/api_surface.json` 里的 36 条路由,方法全标成 `POST`——这是抽取脚本的盲区,不是 llama.cpp 只支持 POST。** 正则只匹配了 `ctx_http.post(...)` 这一种调用形态(`_lab/out/api_surface.json` 的 `"method": "regex"` 字段就是明确的自曝),`server.cpp` 里同样数量级的 `ctx_http.get(...)`(`GET /health`、`GET /props`、`GET /models`、`GET /lora-adapters`、`GET /slots`、`GET /v1/stream`、`GET /tools` 等)和 `ctx_http.del(...)`(`DELETE /models`、`DELETE /v1/stream`)一条都没被统计进去。读这份 JSON 时如果直接拿"36"当作"llama.cpp 一共暴露 36 个能力点",会明显低估。
- **`server-mcp.cpp` 这个文件名本身是个陷阱。** 第一反应容易读成"llama.cpp 实现了一个 MCP server",但代码显示它做的是反方向:llama-server 是 MCP **客户端**,去连接、调用外部 MCP server 提供的工具(`## 0` `## 5` 已给证据)。这个命名习惯(用被集成的协议名字命名"集成这个协议的模块")在很多项目里都存在,读源码时不能只看文件名猜方向,要看谁发起连接、谁实现协议的哪一端。
- **`--cache-ram` 默认是 8192(MiB),不是 0。** 容易假设"额外的缓存机制默认关闭,要显式开启才有",但 `common/common.h:616` 显示提示词缓存**默认就占用 8 GiB 主机内存**——在内存本就紧张的端侧设备上,这是一笔容易被忽略的隐性开销,需要显式传 `--cache-ram 0` 才能关掉。
- **`Q4_K_M` 这种命名格式暗示"4 bit,一种类型",实际是一张混合表。** `src/llama-quant.cpp:608`-`613` 显示同一个 `Q4_K_M` 预设下,不同层、不同角色的张量可能被分配 `Q4_K`、`Q5_K`、甚至 `Q6_K` 三种不同类型——量化后的模型文件大小和精度,是这张混合表和模型结构共同决定的结果,不能只凭预设名字里的"4"字推算出统一的比特占用。
- **`n_parallel` 的"自动"档不是"按硬件自动探测出一个合理值",只是硬编码成 4。** `server.cpp:152`-`156` 的自动逻辑就是一个固定常数加上强制打开 `kv_unified`,不涉及探测显存、探测 CPU 核数之类的运行时判断——这和"auto"这个词通常暗示的"智能适配"有落差,读代码前不应该预设它做了更复杂的事。
- **测试/产品比 0.187 是全仓库口径,不能直接读成"这个仓库测试写得少"。** 和 vLLM 篇([[01-vLLM-全景与代码地图]])提到的同一类陷阱一样,`tests/` 目录里有相当比例是跨后端数值对拍脚本(验证 CPU 和 CUDA 算出同一个数),这类测试的性质和"业务逻辑单测"不同,直接套用比例数字下结论容易失真。

## 8. 可改进点

以下几条标注为**本库推断**,未提交 issue 或 PR 核实维护者是否已有计划:

1. **`_lab/api_surface.py` 对 llama.cpp 的抽取应该补上 `.get(`/`.del(` 两种调用形态。** 现状只认 `.post(`,导致这份统计给出的路由数字系统性偏小、且方法字段全错——这是本库工具自身的已知局限(`## 0` `## 7` 已经指出),修起来只需要把正则扩展到 `ctx_http\.(get|post|del)\s*\(`,不需要真的上 AST。
2. **`GET /slots?fail_on_no_slot` 这个"满载信号"是 opt-in 的,默认路径下客户端拿不到任何背压提示。** 请求满载时唯一的默认行为是无限期排队(`## 6` 已指出),客户端如果不主动轮询 `/slots` 就无法区分"排队中"和"卡住了"。给 `/v1/chat/completions` 这类端点在队列长度超过某个阈值时主动返回一个 429 加 `Retry-After`(而不是要求客户端另外发一个 `fail_on_no_slot` 查询),会让默认体验更接近其他推理服务的习惯用法。
3. **`slot_prompt_similarity` 的 LCP 扫描目前是纯线性的,场景一旦扩大到几十上百个 slot(比如未来某个大内存服务器场景把 `-np` 开得很大),`O(n_slots)` 的每请求扫描成本会线性增长。** 一个轻量级的改进是给每个 slot 的前缀维护一个短哈希摘要,先用哈希做粗筛,只对通过粗筛的候选做完整的 `get_common_prefix` 比对——不需要引入完整的基数树,量级上比现在更适合"slot 数量变大"这个假设性场景。
4. **MCP 只做了客户端方向,server-mcp.cpp 里没有 MCP 服务端实现。** 如果 llama-server 反过来暴露一个标准 MCP 服务端接口(把 `/completion`、`/embedding` 这些能力包装成 MCP resource/tool),会让它更容易被现有的 MCP 生态(Claude Desktop、各类 Agent 框架)直接当作后端接入,而不需要额外写一层适配代码——这是 `## 5` 决策三里提到的明确缺口。

## 9. 自测题与延伸阅读

**闭卷自测**(不看正文,能答上来才算过):

1. `llama-server` 默认(不显式传 `-np`)会起几个 `server_slot`?这个数字是运行时探测出来的还是写死的?对应哪一行代码?
2. 一个新请求到达、所有 slot 都在忙时,llama.cpp 的默认行为是什么?和显式带 `fail_on_no_slot=true` 查询 `GET /slots` 时的行为有什么不同?
3. `get_available_slot` 选 slot 有几层优先级?每一层各自的判据是什么?如果 `slot_prompt_similarity` 被设成 0,会跳过哪一层?
4. `server-mcp.cpp` 让 llama-server 扮演的是 MCP 协议里的客户端角色还是服务端角色?给出一条能支撑这个判断的具体代码或文档证据。
5. K-quant(如 `Q4_K`)和传统线性量化(如 `Q4_0`)在块结构上的核心区别是什么?"缩放因子本身也被量化"这句话对应哪个字段?
6. `Q4_K_M` 这个量化预设名,是"整模型统一用 `GGML_TYPE_Q4_K`"吗?如果不是,真实的分配逻辑在哪个文件、按什么维度分配?
7. llama.cpp 仓库里唯一和"跨机器"沾边的代码路径叫什么名字?它的文档对自己的成熟度给出了什么样的原话评价?

**延伸阅读**(双链只取自 `_PLAN.md` §6 名册):

- [[04-工程规模与代码结构对比]]——把本篇给出的 llama.cpp 规模数字(992,377 行、Python 只占 5.9%)放进 12 个引擎的横向对比里,看它在整个谱系里处在什么位置。
- [[11-开源推理引擎谱系图]]——本篇只深入讲了 llama.cpp 一家,这篇会把它和 TensorRT-LLM、LMDeploy、MLC-LLM 等"其他引擎"放进同一张谱系图,理清楚"端侧优先"这条产品线和"集群优先"那条产品线的分野。
- [[06-性能口径与基准陷阱]]——本篇按 `_PLAN.md` 第 1 节的红线完全没有给出任何实测吞吐/延迟数字(本机无 GPU),这篇讲的是"如果要谈 llama.cpp 的性能,一个负责任的说法需要带哪些口径"。
