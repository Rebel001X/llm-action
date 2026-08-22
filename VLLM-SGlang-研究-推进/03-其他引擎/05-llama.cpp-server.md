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
- **HTTP 连接断开不等于生成中止，这是一个 vLLM 侧找不到对应实现的能力。** 每次生成对应一个按 `conversation_id` 索引的环形缓冲区（`tools/server/server-stream.h:11`-`12`），客户端断线重连后可以从任意偏移量继续读；`_src/vllm/vllm/entrypoints/` 下对同名机制的 grep 是零命中，`## 6` `## 7` 会给出这条能力在两边目标场景下的价值差异。
- **默认配置下这是一个"不设防"的服务，不是"弱设防"。** `middleware_validate_api_key`（`tools/server/server-http.cpp:206`-`208`）在没传 `--api-key` 时直接放行所有请求；如果同时开了 `--tools all`，模型能调用的 `exec_shell_command` 工具默认又是在 `llama-server` 进程本身的权限下跑（`tools/server/server-tools.cpp:1250` 的 `permission_write = true`）——两条默认行为叠在一起，是本篇 `## 7` 要重点展开的风险面。
- **它和 vLLM 不是同一类东西，源码层面能给出至少四条独立证据。** 没有跨进程/跨机器的调度器-执行器分层（一次 `update_slots()` 里只有一处 `llama_decode()` 调用，`:3612`，所有活跃 slot 的下一步 token 拼进同一个 batch 一起跑）；没有 PD 分离的代码痕迹；仓库里唯一的分布式路径（`ggml-rpc`）在文档里被作者自己标注为"proof-of-concept…fragile and insecure…Never run on an open network"（`tools/rpc/README.md:4`-`5`）；"router mode"做的是同一台机器上管理多个模型进程的生命周期（`tools/server/server-models.h:113` 的 `server_models`），不是跨节点的集群编排。`## 6` 会逐条对照 vLLM 给出反例。

## 1. 它在系统里的位置

`llama.cpp` 的 README 把目标说得很直接："The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware"，紧接着第一条特性就是"Plain C/C++ implementation without any dependencies"（`README.md:54`、`:57`）。快速上手的两条命令也写在 README 里——`llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF` 直接从 Hugging Face 拉模型跑命令行，`llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF` 拉起 OpenAI 兼容的 API server（`README.md:19`-`24`）——一条命令、一个二进制、不需要提前起 Python 环境或装 CUDA。

这决定了它在本库 12 个引擎里的坐标：vLLM、SGLang、TensorRT-LLM、Dynamo 这些引擎默认工作点是"一台或多台装好驱动的服务器 GPU，追求高并发吞吐"；llama.cpp 的默认工作点是"一台可能没有独立显卡的个人电脑或边缘设备，追求装得上、跑得起来、单用户体验可接受的延迟"。这两个目标不是同一目标函数下的两个刻度，而是两个不同的优化方向——`## 6` 会把这句话落到具体代码证据上，不停留在口号层面。

仓库规模上，`_lab/out/repo_stats.json` 的 `top_dirs` 显示最大的目录是 `ggml/src`（364,133 行/819 文件）——这是硬件后端层，比第二名 `tools/ui`（110,392 行/428 文件，自带的 TypeScript Web UI）大三倍还多；`tools/server`（34,611 行/71 文件）才是本篇主角所在的目录，规模上排第七。这个排序本身就说明了 llama.cpp 的资产重心：**打通尽可能多的硬件**比**server 本身的调度复杂度**占的代码量大得多，这和 vLLM 把最大代码量堆在 `model_executor`+`v1` 引擎循环（据 [[01-vLLM-全景与代码地图]]）是完全不同的分布形状。

测试/产品比 0.187（`_lab/out/repo_stats.json` 的 `totals.test_to_src_ratio`）——这个数字本身不能孤立地读成"测试不够"，因为 `tests/`（45,555 行/56 文件）里相当一部分是逐算子的数值对拍脚本（`test-quantize-fns.cpp`、`test-backend-ops.cpp` 这类），验证的是"CPU 实现和 CUDA 实现算出同一个答案"，不是传统意义上的业务逻辑单测,这一点标注为**本库推断**，未逐文件核对每个测试文件的性质。

### 1.1 `ggml/src` 内部构成:CPU 后端比 CUDA 后端代码更多

`ggml/src` 下每个 `ggml-<backend>/` 子目录对应 `## 0` 提到的一条硬件路径,逐目录数行数(**源码为证**,本库现场统计,未见于 `_lab/out/repo_stats.json` 的既有字段):

| 后端目录 | 文件数 | 行数 | 说明 |
|---|---:|---:|---|
| `ggml-cpu/` | 67 | 91,460 | **全仓库最大的单一后端**,超过 CUDA 的两倍还多 |
| `ggml-opencl/` | 176 | 66,002 | 跨厂商 GPU 的兜底路径,规模上排第二 |
| `ggml-sycl/` | 167 | 44,712 | Intel oneAPI |
| `ggml-cuda/` | 276 | 44,232 | 文件数最多(276,和 `_lab/out/repo_stats.json` 统计的 `cuda_files: 273` 接近但不完全相等,差异来自 `.cu`/`.cuh` 之外还有少量 `.h`/`CMakeLists.txt` 被一并计入本表) |
| `ggml-vulkan/` | 177 | 42,763 | 含大量 `.comp` shader 源码 |
| `ggml-hexagon/` | 80 | 34,251 | 高通 Hexagon DSP |
| `ggml-metal/` | 13 | 25,657 | 文件数最少但单文件密度高,Apple 芯片专属 |
| `ggml-et/` | 78 | 23,705 | 具体全称`未查证`,`ggml/src/ggml-et/CMakeLists.txt` 显示它依赖一个外部 `ET_PLATFORM` SDK |
| `ggml-openvino/` | 74 | 14,532 | Intel OpenVINO |
| `ggml-cann/` | 7 | 9,976 | 华为昇腾 CANN |
| `ggml-webgpu/` | 55 | 20,394 | 浏览器/跨平台 WebGPU |
| `ggml-virtgpu/` | 42 | 4,989 | 虚拟机/容器场景的 virtio-gpu 透传 |
| `ggml-rpc/` | 4 | 2,815 | `## 5` 决策四讨论的 PoC 级分布式路径 |
| `ggml-zendnn/` | 2 | 929 | AMD ZenDNN(CPU 专属优化库) |
| `ggml-zdnn/` | 8 | 924 | IBM Z 架构 NNPA 加速器 |
| `ggml-blas/` | 2 | 631 | 通用 BLAS 库桥接 |
| `ggml-musa/` | 3 | 248 | 摩尔线程,壳文件——真正的算子实现复用 `ggml-cuda/` |
| `ggml-hip/` | 1 | 153 | AMD HIP,同样是壳文件复用 `ggml-cuda/` |

**CPU 后端代码量反直觉地排第一**,原因不难理解:CUDA/SYCL/Vulkan 这些后端能调用厂商提供的 BLAS/cuBLAS/oneMKL 之类的库做矩阵乘法这类重头运算,`ggml-cpu/` 没有这种依赖,每一种量化格式(`## 3` 提到的 27 种)乘每一种 SIMD 指令集(AVX2/AVX512/NEON/…)的组合都要手写核函数——`ggml-cpu/` 事实上同时扮演了"通用兜底路径"和"手工优化最彻底的路径"两个角色。`ggml-hip/`(153 行)和 `ggml-musa/`(248 行)体量极小,印证了 `## 0` 已给出的判断:它们只是把 `GGML_USE_CUDA` 宏和转译脚本接起来的壳,真正的核函数实现只写了一份,在 `ggml-cuda/` 里。

这份规模不是纸面上"支持"两个字带过的——CI 配置里能看到对应的独立构建验证:`.github/workflows/` 下有 25 个 `build-*.yml`/`ui-build*.yml` 文件,`build-cann.yml`、`build-cpu.yml`、`build-cuda-ubuntu.yml`、`build-cuda-windows.yml`、`build-opencl.yml`、`build-openvino.yml`、`build-sycl.yml`、`build-virtgpu.yml`、`build-vulkan.yml`、`build-wasm.yml`、`build-webgpu.yml` 逐个后端各有一份(**源码为证**:`ls .github/workflows/` 现场统计,未见于 `_lab/out/repo_stats.json` 的既有字段)。这意味着 `## 0` 提到的"16 条硬件路径"不是声明式的宏列表,每条路径在合并代码前都要过一次独立的 CI 构建。

### 1.2 `tools/server` 内部构成:一个曾经的单文件已经拆成十二个

`tools/server`(34,611 行/71 文件,`_lab/out/repo_stats.json` 的 `top_dirs`)内部按职责拆分,规模从大到小:

| 文件 | 行数 | 职责 |
|---|---:|---|
| `server-context.cpp` | 5,475 | slot 状态机、`update_slots()` 主推理循环、批处理组装 |
| `server-models.cpp` | 2,554 | router 模式下的多模型子进程生命周期管理 |
| `server-tools.cpp` | 2,172 | 内置工具(文件系统等)与 MCP 工具的统一调用层 |
| `server-task.cpp` | 1,903 | `server_task`/`server_task_result` 及提示词缓存的实现 |
| `server-common.cpp` | 1,821 | 跨文件共享的工具函数(JSON 序列化、错误格式化等) |
| `server-http.cpp` | 838 | HTTP 层封装(基于 `cpp-httplib`)、线程池、CORS |
| `server-chat.cpp` | 692 | chat template 渲染、多轮对话消息规整 |
| `server-stream.cpp` | 668 | 可恢复流式传输(`/v1/stream`、`/v1/streams/lookup`) |
| `server-schema.cpp` | 659 | JSON Schema/结构化输出相关 |
| `server-queue.cpp` | 621 | 任务队列、主循环调度(`start_loop`)、yield 机制 |
| `server-mcp.cpp` | 820 | `## 0` `## 5` 反复提到的 MCP 客户端实现 |
| `server.cpp` | (主文件) | `main()`、路由表装配、启动期参数处理 |

这个拆分本身是新近发生的架构演进信号:文件名前缀统一为 `server-*`,说明它们是从一个更早的单体 `server.cpp` 里陆续拆出来的独立编译单元——`## 0` 提到的"server 已拆多文件"这条已知事实,在这张表里能看到拆分粒度已经细到"MCP 客户端"“可恢复流式传输”“路由到多模型的代理逻辑”这种功能级别,而不是简单按行数对半切。

### 1.3 Web UI 与多模态:两块不在"server 逻辑"里但同样真实的资产

`tools/ui`(110,392 行/428 文件,几乎全是 TypeScript,`_lab/out/repo_stats.json` 的 `top_dirs`)是内嵌进同一个仓库的 Web 前端——**这不是外部项目,是这个仓库自己维护的一整套 SPA**。构建期它被编译(或从 Hugging Face 的 `ggml-org/llama-ui` 桶下载预构建产物,`tools/ui/CMakeLists.txt:3`)成生成的 `ui.cpp`/`ui.h`(`tools/ui/CMakeLists.txt:36`-`37`),再作为 `llama-ui` 静态库链接进 `llama-server-impl`(`tools/server/CMakeLists.txt` 里 `target_link_libraries` 一行同时列出 `server-context llama-ui cpp-httplib`)——最终 Web UI 的静态资源以字节数组的形式被编译进同一个可执行文件,不需要单独部署一个前端服务。`tools/mtmd`(29,271 行/75 文件)是多模态(图像/音频输入)的编码器与投影层实现,和 `## 3` 提到的 `mctx`/`mbatch`(`server_slot` 里的多模态字段)对应。两者合计接近 14 万行,提醒读者:即便只看 `tools/server`,也漏掉了这个仓库相当一部分"能力"层——Web UI 和多模态支持都不在 `tools/server` 目录下,但都是 `llama-server` 实际服务的功能。

### 1.4 模型架构支持是两侧对称的手工劳动

`## 5` 决策一会展开这一点的代价分析,这里先摆出规模证据:支持一个新模型架构,llama.cpp 需要在**两个独立的目录、两种语言**里各写一份代码。Python 侧,`conversion/`(89 文件、20,377 行,`_lab/out/repo_stats.json` 的 `top_dirs`)下每个架构对应一个 `@ModelBase.register` 装饰的转换类(**源码为证**:`grep -rc '@ModelBase.register' conversion/` 现场统计得到 222 处命中),负责把 HuggingFace 权重的张量命名、形状映射到 GGUF 要求的布局;`convert_hf_to_gguf.py` 本身只有 307 行,是个瘦身过的 CLI 入口,不装真正的转换逻辑。C++ 侧,`src/models/`(151 文件,`## 5` 已给出)每个架构对应一个手写图构建 `.cpp` 文件。222 和 151 这两个数字不完全相等,一部分原因是同一个 GGUF 架构类型可能被多个 HuggingFace 模型家族共用(转换层按 HF 模型类名区分,C++ 图构建层按 GGUF 的 `LLM_ARCH_*` 枚举区分,颗粒度不同)——但两个数字量级接近这件事本身,已经说明"加一个新模型"在这个项目里从来不是单侧的工作量。

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
| 鉴权中间件 | `tools/server/server-http.cpp:206`-`208` | `middleware_validate_api_key`：`params.api_keys` 为空时直接放行——**默认不鉴权**，需要显式传 `--api-key` 才会校验 |
| 鉴权中间件 | `tools/server/server-http.cpp:218`-`221` | 同时接受 OpenAI 风格的 `Authorization: Bearer <key>` 和 Anthropic 风格的 `X-Api-Key` 两种请求头 |
| 可恢复流式传输 | `tools/server/server-stream.h:11`-`12` | 环形缓冲区按 `conversation_id` 索引，生成过程和某一次具体的 HTTP 连接解耦 |
| 内置工具 | `tools/server/server-tools.cpp:1247`-`1250` | `server_tool_exec_shell_command`：`permission_write = true`，默认在宿主机权限下执行 shell 命令 |
| 内置工具隔离 | `tools/server/server-tools.cpp:1861` | `server_tools_container_runtime`：`--tools-runtime` 配置后，工具调用被路由进容器或远程 SSH 主机 |
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
| 异常统一处理 | `tools/server/server.cpp:54`-`84` | `ex_wrapper`：包裹每个路由 handler，把 `std::invalid_argument`/`std::exception`/未知异常统一翻译成 400/500 JSON 错误体 |
| 任务类型枚举 | `tools/server/server-task.h:16`起 | `server_task_type`：`SERVER_TASK_TYPE_COMPLETION`/`_INFILL`/`_EMBEDDING`/`_RERANK`/`_SLOT_GET` 等 |
| GGUF 魔数/版本 | `ggml/include/gguf.h:41`-`42` | `#define GGUF_MAGIC "GGUF"`、`#define GGUF_VERSION 3` |
| 量化预设枚举 | `include/llama.h:116`-`158` | `llama_ftype`：36 个 CLI 可用的量化预设值 |
| 传统线性量化块 | `ggml/src/ggml-common.h:194`-`198` | `block_q4_0`：32 元素一块，单一 fp16 缩放因子，无偏移量 |
| K-quant 超级块 | `ggml/src/ggml-common.h:323`-`338` | `block_q4_K`：256 元素超级块，缩放/偏移量本身也被 6 bit 量化 |
| I-quant 码本块 | `ggml/src/ggml-common.h:381`-`385` | `block_iq2_xxs`：每个量化码是一个索引，指向编译期烧进二进制的定长网格表 |
| I-quant 码本表 | `ggml/src/ggml-quants.c:3111`-`3113` | 按 `ggml_type` 选用哪张 `kgrid_*` 常量表 |
| API 鉴权中间件 | `tools/server/server-http.cpp:206`-`231` | `middleware_validate_api_key`：默认放行、支持 `Bearer` 与 `X-Api-Key` 两种请求头 |
| CORS 安全警告 | `tools/server/server.cpp:310`-`316` | `cors_origins == "*" && api_keys.empty()` 时打印醒目警告 |
| 内置工具隔离 | `tools/server/README.md:201` | `--tools-runtime`：把内置工具调用路由进 Docker/Podman 容器或远程 SSH 主机 |

| 重要性矩阵 | `tools/imatrix/README.md:3` | `llama-imatrix`：跑一遍校准文本、收集每个张量各通道的激活统计，供量化时参考 |
| 重要性矩阵门槛 | `src/llama-quant.cpp:179`、`:593` | `qs.has_imatrix`：有无 imatrix 会切换同一个量化预设下的实际分配分支 |
| 嵌入端点 | `tools/server/server-context.cpp:5056`-`5062` | `post_embeddings`/`post_embeddings_oai`：同一个 `handle_embeddings_impl`，两种响应体格式 |
| 重排序端点 | `tools/server/server-context.cpp:5064`-`5066` | `post_rerank`：要求 `--reranking`（`params.embedding && pooling_type == RANK`），否则报错 |
| 重排序协议兼容 | `tools/server/server-context.cpp:5072`-`5075` | 同时兼容 TEI 与 Jina 两种重排序请求体格式，按请求体是否含 `"texts"` 字段自动判断 |

以上 44 条只是骨架，`## 3` `## 4` 会把其中若干条的上下文摊开细读。

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
- **`server_task` / `server_task_result`**（`tools/server/server-task.h:137`、`:272`）——HTTP 请求被翻译成的内部任务对象与其对应结果对象;`server_task_type`（`:16`起）枚举了 `SERVER_TASK_TYPE_COMPLETION`/`_INFILL`/`_EMBEDDING`/`_RERANK`/`_SLOT_GET` 等任务种类,`## 4.2` 走读的调度分支就是按这个枚举 `switch`。一个任务不一定对应一个 slot——`task.is_parent()`/`child_tasks`(`tools/server/server-context.cpp:2348`起)支持"一个父任务扇出多个子任务、各自抢一个 slot"的模式,这是 `slot_state::WAIT_OTHER` 存在的原因。
- **`buft_list_t`**(`src/llama-model-loader.h:21`,定义为 `std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>`)与 `ggml_backend_buffer_type_t`(`ggml/include/ggml-backend.h:24`,一个不透明指针类型)——模型加载时每个设备对应一份"这个设备能把张量放进哪些 buffer 类型"的**优先级列表**,`## 4.1` 会展示它如何在 GPU 分配失败时提供退路。这一层是 `## 5` 决策一"CPU 兜底不需要异常处理分支"的具体数据结构基础。
- **GGUF 文件布局**(`ggml/include/gguf.h:1`-`30` 的头部注释是权威定义)——一个 GGUF 文件从头到尾是:4 字节魔数 `"GGUF"`(`GGUF_MAGIC`,`:41`)→ 4 字节版本号(当前 `GGUF_VERSION 3`,`:42`)→ 张量计数 → KV 元数据计数 → 逐个 KV 对(超参数、tokenizer 词表、量化信息等都塞在这里)→ 逐个张量的名字/维度/类型/偏移量 → 最后是对齐过的张量数据二进制块。**整个模型是一个自描述的单文件**,不需要旁边配一个 `config.json` 或 tokenizer 文件才能加载——这是 `## 5` 决策五要展开的对比点。
- **`ex_wrapper`**(`tools/server/server.cpp:54`-`84`)——一个包裹每个路由 handler 的统一异常翻译层。`## 2` 表格里几乎每一行注册路由都长成 `ctx_http.post("/path", ex_wrapper(handler))` 这个形状:`ex_wrapper` 内部 `try/catch` 住 `std::invalid_argument`(翻译成 400)、其他 `std::exception`(翻译成 500)、以及裸的 `catch (...)`(未知异常同样归为 500),统一格式化成 JSON 错误体返回,还会把异常信息打进 `SRV_WRN` 日志(`:76`)。**这意味着单个 handler 内部完全不需要写自己的 try/catch 来处理"参数不对就报错"这类场景**——直接 `throw std::invalid_argument(...)`,顶层的 `ex_wrapper` 会接住。这是 C++ 单体架构下用异常传播替代逐层显式错误码检查的一个具体例子。

## 4. 主流程走读

### 4.1 启动:从命令行参数到"能收请求"

1. **模式判定。** `main()`（`tools/server/server.cpp`）先判断是不是 router 模式:`is_router_server`——没给 `--model`/`--hf-repo`/`--docker-repo` 时为真(此时进程只做多模型代理,不碰 GPU,注释原文"router server never loads a model and must not touch the GPU",`tools/server/server.cpp:135`-`137`)。
2. **slot 数量落定。** 非 router 模式下,`params.n_parallel < 0`(默认值)会被改写成 4,同时强制 `kv_unified = true`(`tools/server/server.cpp:152`-`156`)——这两行决定了"什么都不配置直接起服务"时到底有几路并发、上下文怎么分。
3. **路由表装配。** `server_routes routes(params, ctx_server)` 构造出所有 handler,随后是一长串 `ctx_http.get(...)`/`ctx_http.post(...)`/`ctx_http.del(...)` 调用(`tools/server/server.cpp:234`起)。是否注册 `/tools`、`/cors-proxy` 取决于运行时开关(`--tools`、`--ui-mcp-proxy`),关闭时对应路径返回 403 而不是 404(`tools/server/server.cpp:329`-`330`、`:359`-`360`)——**路由表本身是静态的,但 handler 行为按配置分叉**。
4. **模型加载与后端选择。** 模型加载路径(`src/llama-model.cpp`)会先枚举 `ggml_backend_dev_count()` 遍历到的所有设备,分别调用 `make_gpu_buft_list`(`:986`)和 `make_cpu_buft_list`(`:924`)为每个设备建一份"这个设备支持存哪些 buffer 类型"的列表;关键的一步是 `make_gpu_buft_list` 返回之后,调用方紧接着把 `pimpl->cpu_buft_list` 拼接到每个 GPU 设备自己的 buft 列表末尾,注释原话"add CPU buffer types as a fallback"(`:1322`-`1324`)——**每个张量在放不进 GPU 显存时,天然有一条退到 CPU 内存的路,不需要额外的异常处理分支**,这是本篇 `## 5` 要展开的第一个设计决策。
5. **slot 初始化。** `for (int i = 0; i < params_base.n_parallel; i++) slots.emplace_back()`(`tools/server/server-context.cpp:1233`起),每个 slot 拿到相同的 `n_ctx_slot = llama_n_ctx_seq(ctx_tgt)`(`:1211`,若超过训练长度会被截断并打警告),意味着**同一次启动里所有 slot 的容量是一致的**,没有"给这个 slot 多分点上下文"的运行时调节。

### 4.2 一次 `/v1/chat/completions` 请求怎么走

1. **HTTP 层接住请求。** `httplib::ThreadPool` 里的某个工作线程(池大小由 `n_threads_http` 决定,默认 `max(n_parallel+4, hardware_concurrency()-1)`,`tools/server/server-http.cpp:309`-`320`)解析出 JSON body,调用挂在 `/v1/chat/completions` 上的 `routes.post_chat_completions`(`tools/server/server.cpp:245`)。这一层**是**多线程的——HTTP 解析、JSON 反序列化、chat template 渲染可以在多个请求间并行进行。
2. **打包成任务、入队。** handler 把请求转成一个或多个 `server_task`,通过 `server_queue::post` 塞进 `queue_tasks`(`std::deque`,`tools/server/server-queue.h:24`),然后阻塞等待结果(通过 `server_response_reader` 轮询)。
3. **单一主循环被唤醒。** `server_queue::start_loop`(`tools/server/server-queue.cpp:278`)是**全进程唯一**跑推理的循环:`process_new_tasks(false)` 把新任务从队列搬进内部状态,然后调用 `callback_update_slots()`——这一步是同步的、单线程的,"HTTP 层多线程、推理主循环单线程"是这个进程真实的并发形状。
4. **挑 slot。** `update_slots()` 内部对每个待处理任务调 `get_available_slot(task)`(`tools/server/server-context.cpp:1500`):
   - 若 `slot_prompt_similarity`(默认 0.1,`common/common.h:678`)不为 0,对所有空闲 slot 算 `tokens.get_common_prefix(task.tokens)` 得到最长公共前缀长度,取相似度(前缀长度/请求长度)最高且过阈值的那个(`:1513`-`1544`);
   - 否则(或没找到)退化成"选 `t_last_used` 最小的空闲 slot"(LRU,`:1565`-`1579`);
   - 若命中的 slot 即将丢失超过一半的既有上下文(`f_keep < 0.5`),先把它当前状态存进 `server_prompt_cache`(主机内存)再覆盖(`:1554`-`1558`、`:1582`-`1601`)。
   - 若所有 slot 都在忙,`get_available_slot` 返回空,任务被 `queue_tasks.defer(...)` 送进 `queue_tasks_deferred`(`tools/server/server-context.cpp:2334`-`2337`),**不是被拒绝**;下次某个 slot 释放时,`pop_deferred_task(id_slot)`(`tools/server/server-queue.cpp:90`)优先把等着这个 slot id 的任务捞回来,否则捞队首那个(FIFO,`:104`-`108`)。默认没有队列深度上限,只有显式带 `fail_on_no_slot=true` 查询参数请求 `GET /slots` 时才会看到 503(`tools/server/server-context.cpp:4675`-`4681`)——**默认路径下请求永远排队,不会因为忙就被拒**。
5. **一次批处理。** 拿到 slot 的任务被推进(prompt 处理或采样一步),所有当前活跃 slot 各自贡献的 token 通过 `common_batch_add`(`tools/server/server-queue.cpp:164`)拼进同一个 `llama_batch`,最终整个 `update_slots()` 里**只有一处** `llama_decode(ctx_tgt, batch_view)` 调用(`tools/server/server-context.cpp:3612`)——这一步用 `queue_tasks.yield_to_queue(...)` 包起来,好让 metrics 一类的元任务能在解码等待期间被处理,但解码本身是这一步唯一的重活。
6. **结果流式返回。** 采样出新 token 后,对应 slot 的 handler 通过 SSE 把增量结果写回 HTTP 响应流;当某个 slot 的任务完成(命中停止条件或达到 `n_predict_max`),`state` 回到 `SLOT_STATE_IDLE`(`tools/server/server-context.cpp:508`),等待下一次 `get_available_slot` 把它选中。

`## 6` 提到"一次 `update_slots()` 只有一处 `llama_decode`",这句话需要一个补充:**一次调用不等于一个长 prompt 一步吃完**。`update_slots()` 内部注释直接写"process in chunks of params.n_batch"(`tools/server/server-context.cpp:3033`),`-b`/`--batch-size`(逻辑批大小,`common/arg.cpp:1659`-`1660`)和 `-ub`/`--ubatch-size`(物理批大小,`common/arg.cpp:1666`-`1667`)两个旋钮共同决定这个"块"有多大。一个长 prompt 进入 `SLOT_STATE_PROCESSING_PROMPT` 后,会被拆成多个不超过 `n_batch` 的块,**每个块占用一次独立的 `update_slots()` 迭代**,直到整段 prompt 处理完才切到 `SLOT_STATE_DONE_PROMPT`。所以准确的说法是:任意一次 `llama_decode` 调用里,确实只服务"当前这一批活跃 slot 各自贡献的一个块",但一个 slot 要吃完自己的长 prompt,可能要跨越好几次这样的调用——这是 chunked prefill 在 llama.cpp 里的具体落点,和 vLLM 的 chunked prefill(`` `vllm:vllm/config/scheduler.py:70` `` 的 `long_prefill_token_threshold`,超过这个 token 数的 prompt 会被调度器主动拆块)解决的是同一个问题,只是控制旋钮和触发时机在两边分别绑定在不同的层——llama.cpp 是"每次不超过 n_batch 就必然分块",vLLM 是"超过一个可配置阈值才分块"。
### 4.3 GGUF 量化:三条路线,一张混合表

三个量化家族在 `## 3` 已经给出结构体证据,这里补机制层面的关键差异:

- **传统线性量化**(`Q4_0`/`Q5_0`/`Q8_0` 等)是最简单的每块一个缩放因子,`x = a*q`,解码只需一次乘法,没有分层结构。
- **K-quant**(`Q2_K`…`Q6_K`)把 256 个元素分成 8 个子块,子块各自的缩放/偏移量本身又被压缩进 6 bit(`K_SCALE_SIZE`,`ggml/src/ggml-common.h:337` 附近),换来的是"用更少的额外存储,逼近逐通道量化的精度",代价是解码路径多一层间接寻址。
- **I-quant**(`IQ1_S`…`IQ4_XS`)在 2-4 bit 区间用预训练好的码本网格代替逐元素线性量化,`kgrid_2bit_256`/`kgrid_1bit_2048` 这些常量表(`ggml/src/ggml-quants.c:2859`起)是编译期烧进二进制的,不随模型变化——这意味着**量化精度和这些码本的设计强相关**,不是简单调一个缩放因子能改善的。

而 CLI 层面用户敲的 `Q4_K_M` 这种预设名,并不对应"整模型统一用 `GGML_TYPE_Q4_K`"。`src/llama-quant.cpp:608`-`613` 的分支显示,`LLAMA_FTYPE_MOSTLY_Q4_K_M` 下,注意力/FFN 的某些张量按层深度(`use_more_bits`,`:430`-`431`:处于最前 1/8、最后 1/8,或每 3 层里的特定一层)被升到 `Q5_K`,再往上一级的 `attn_v`/`ffn_down` 甚至可能升到 `Q6_K`——这是一张**按张量角色和层位置分配精度**的混合表,`M`/`S`/`L` 后缀描述的是这张表整体偏保守还是偏激进,不是一个统一比特宽度的代号。`--pure` 选项(`tools/quantize/README.md` 描述)会关掉这套混合逻辑,强制所有张量用同一种类型。

这张混合表还有第二个输入维度:**重要性矩阵(imatrix)**。`tools/imatrix`(独立工具 `llama-imatrix`)先用一段校准文本跑一遍模型,收集每个张量各通道的激活统计(`tools/imatrix/README.md:3`:"Compute an importance matrix for a model and given text dataset"),量化时把这份统计喂给 `llama-quantize` 的 `--imatrix` 选项。`src/llama-quant.cpp` 里 `qs.has_imatrix`(`:179`)这个标志位直接影响分配逻辑:比如 `LLAMA_FTYPE_MOSTLY_IQ3_XXS` 在**没有** imatrix 时会整体退化到更保守的类型(`:593`),`ffn_down` 层的量化在**有** imatrix 时才会启用一段额外的"防止某些层量化崩掉"的特殊处理(`:624`-`627` 的注释原话:"Guard against craziness in the first few ffn_down layers that can happen even with imatrix")。换句话说,**同一个 `Q4_K_M` 预设名,搭配不搭配 imatrix,实际选出来的 `ggml_type` 分配都可能不同**——预设名本身只决定"要不要混合"和"混合的大致激进程度",真正逐张量的取舍还要看有没有、以及用了什么样的校准数据算出来的 imatrix。

### 4.4 一个张量怎么决定放在哪块设备上

`## 1.1` 已经给出 16 条硬件后端,但一个具体的模型权重张量最终落在哪个设备的显存/内存里,是加载期一段级联逻辑决定的,不是"整模型一股脑塞进 GPU":

1. **枚举设备,分设备建候选列表。** `make_gpu_buft_list(dev, split_mode, tensor_split)`(`src/llama-model.cpp:986`)为每个 GPU 设备建一份它自己能用的 buffer 类型列表(如果 `split_mode == LLAMA_SPLIT_MODE_ROW`,还会尝试拿到设备的行切分 buffer 类型,通过 `ggml_backend_reg_get_proc_address` 动态查找该后端是否实现了 `ggml_backend_split_buffer_type`);`make_cpu_buft_list(devices, use_extra_bufts, no_host)`(`:924`)则单独建一份 CPU 侧的列表,里面优先塞 `ACCEL` 类型设备(BLAS/AMX 这类"跟 CPU 配合用"的加速器,判据是 `ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL`,`:930`),再塞一个"host buffer"(GPU 需要把大批量数据从主机搬过去时用的锁页内存,`:930`起注释解释了原因),最后才是普通 CPU 内存本身。
2. **GPU 列表末尾拼接 CPU 列表。** `make_gpu_buft_list` 返回后,调用方紧接着执行 `buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end())`,注释原话"add CPU buffer types as a fallback"(`src/llama-model.cpp:1322`-`1324`)——**每个 GPU 设备自己的候选列表,天然以 CPU 列表结尾**,意味着某个张量如果在这个设备上分配失败(比如显存不够),下一步会自动尝试 CPU 而不需要额外写一层 `try/except` 式的补救代码。
3. **显存不够时的默认切分策略。** 多 GPU 场景下,如果用户没有手动指定 `tensor_split`,`load_tensors` 会按各设备当前的空闲显存比例(`ggml_backend_dev_memory(dev, &free, &total)`,`src/llama-model.cpp:1336`起"default split, by free memory")算出默认切分权重——如果某个设备的驱动不报告显存数字(`free == 0 && total == 0`,`src/llama-model.cpp:1346`),会退化成用 CPU 的内存数字顶替,注释直接标了对应的上游 issue 号,应对的是某些没有实现显存查询接口的后端。

这套逻辑解释了 `## 0` 提到的"ggml 一份代码打通所有硬件"具体是怎么落地的:**不是每个后端单独写一套加载流程,而是所有后端共用同一套 buft 级联分配逻辑**,后端之间的差异被封装进 `ggml_backend_dev_buffer_type`/`ggml_backend_dev_host_buffer_type` 这类统一接口背后。

### 4.5 slot 内部的另外两条支线:投机解码与多模态

`server_slot`(`## 3`)里的字段其实透露了两条没有在主流程走读里展开的支线:

- **投机解码**:`ctx_dft`(草稿模型上下文,`tools/server/server-context.cpp:200`)、`spec`(`common_speculative*`)、`spec_draft`/`spec_prompt`/`spec_i_batch` 这些字段(`tools/server/server-context.cpp:208`-`215`)说明投机解码不是一个独立的旁路系统,而是**嵌进了同一个 slot 状态机内部**——一个 slot 在 `SLOT_STATE_GENERATING` 时,除了目标模型的 `llama_decode`,还会驱动草稿模型多生成几个候选 token,校验后一次性接受或回退。这意味着开启投机解码不会增加 slot 数量或改变调度逻辑,只是让单个 slot 内部的"一步"变得更重。
- **多模态**:`mctx`(`mtmd_context*`)、`mbatch`(`mtmd::batch_ptr`)两个字段(`tools/server/server-context.cpp:204`-`206`)对接 `tools/mtmd`(`## 1.3` 提到的 29,271 行多模态编码器)。图像/音频输入在进入 `llama_decode` 之前,要先经过 `mtmd` 的编码器转换成 embedding,这条路径和纯文本路径共享同一个 slot 状态机,但 `server_slot::can_split()`(`tools/server/server-context.cpp:394`-`397`)上方专门有一段注释说明"如果这个 context 没有 memory 模块,所有 embedding 必须在单个 ubatch 内算完"——多模态输入对批处理粒度提出了比纯文本更严格的约束。

### 4.6 可恢复流式传输:HTTP 连接断了,生成不断

`server-stream.h`(`## 2` 已列出该文件)实现的是一个和 slot 模型正交但同样值得记一笔的能力:**HTTP 连接断开不等于生成中止**。头部注释写得很直接:"streaming buffer for one generation, survives HTTP disconnect. the producer appends SSE bytes, readers drain from any offset via read_from. keyed by conversation_id"(`tools/server/server-stream.h:11`-`12`)。具体机制:

- 每次生成对应一个 `stream_session`,由 `stream_pipe_producer`(`tools/server/server-stream.h:31`-`40`)持有并写入一个环形缓冲区,这个缓冲区**不属于某个具体的 HTTP 连接**,而是独立存在,按 `conversation_id` 索引。
- 客户端断线重连后,`GET /v1/stream`(由 `server_stream_make_get_handler` 生成的 handler,`:51`)可以带着同一个 conversation id 重新接上,从环形缓冲区里已经写到的任意偏移量继续读——**生成本身在服务端一直没停**,断的只是 HTTP 层的字节流。
- `POST /v1/streams/lookup`(`:52`-`55`)让客户端一次性查询多个自己知道的 conversation id 是否还活着,但注释明确写了一条安全边界:"only answers for ids the caller already owns…the server never lists ids it has not been asked about, so a random caller cannot enumerate live sessions"——**服务端不会主动泄露还有哪些会话在跑**,查询是"验证你已知道的东西",不是"探测有什么"。

这个能力和 `## 6` 讨论的"固定 slot"模型是两个不同层面的问题:slot 决定的是"计算资源怎么分配",可恢复流式传输解决的是"HTTP 这条易断的连接,如何不成为生成过程的单点故障"——router 模式下(`## 2` 提到的 `server_models_routes`),这一层还要再解决一个问题:请求可能落在多个子进程中的某一个,`router_stream_get`/`router_streams_lookup`/`router_stream_delete`(`tools/server/server-models.h:358`-`360`)三个 handler 的职责就是"先找到拥有这个会话的子进程,再把请求转发过去"——**HTTP 连接可以断,但"这个会话归哪个进程"这份记账不能丢**。

### 4.7 内置工具:比 MCP 更直接的一条能力扩展路径

`## 0` `## 5` 反复强调 `server-mcp.cpp` 让 llama-server 扮演 MCP 客户端角色,但 MCP 不是模型调用工具的唯一路径——`server-tools.cpp`(2,172 行,`## 1.2` 已列出)自己就实现了一批**内置**工具,不依赖任何外部进程。`tools/server/README.md:200` 把可用工具名单写得很直白:`read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info`,分别对应源码里逐个继承 `server_tool` 的实现类(`tools/server/server-tools.cpp` 里 `name = "read_file"`、`name = "exec_shell_command"` 等赋值语句各自标记了一个类的入口)。用 `--tools all` 一次性打开全部,或者 `--tools name1,name2` 挑着开。

这份清单里 `exec_shell_command` 值得单独拿出来讲:这是一个**允许模型直接在宿主机上执行任意 shell 命令**的工具,`server_tool_exec_shell_command` 的构造函数把 `permission_write` 标成 `true`(`tools/server/server-tools.cpp:1250`),命令输出上限 16 KB、超时上限 60 秒(`tools/server/server-tools.cpp:1242`-`1243`)。README 对这整个特性的提醒毫不含糊:"do not enable in untrusted environments"(`tools/server/README.md:200`),而且开启后会**自动把 `--cors-origins` 收紧到只允许 localhost**(同一行)——这是一条把"默认更安全"直接做进代码里的设计,不依赖用户自己记得关。这批能力和 `## 5` 决策三讨论的 MCP 集成方向一致(让模型能干活),但路径更短:不用先起一个外部 MCP server 进程,风险和收益都更直接地暴露在同一个二进制里。

### 4.8 嵌入与重排序:又一处"身兼多个协议方言"

`POST /embeddings`(legacy)、`POST /embeddings`/`POST /v1/embeddings`(`## 2` 已列出路由)最终都走到 `handle_embeddings_impl`,`post_embeddings`(`tools/server/server-context.cpp:5056`-`5058`)和 `post_embeddings_oai`(`:5060`-`5062`)是同一个实现函数的两层薄包装,区别只在于 `TASK_RESPONSE_TYPE_NONE` 还是 `TASK_RESPONSE_TYPE_OAI_EMBD`——即**同一份计算逻辑,两种响应体格式**。`POST /rerank` 系的路由(`/rerank`、`/reranking`、`/v1/rerank`、`/v1/reranking`)则有一段前置校验:`post_rerank`(`tools/server/server-context.cpp:5064`)一开始就检查 `!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK`(`:5066`),不满足就直接返回"This server does not support reranking. Start it with `--reranking`"的 `NOT_SUPPORTED` 错误——**这不是一个总是可用的端点,依赖启动参数**。源码注释(`:5072`-`5074`)还标注这批端点同时兼容两种业界格式:TEI(HuggingFace Text Embeddings Inference)和 Jina Reranker 各自的请求体形状,由请求体里出现的字段自动判断走哪种解析路径。这和 `## 0` 已给出的"OpenAI + Anthropic 双标准"是同一个模式在另一个功能面上的重复:**能力越基础(嵌入、重排序这类相对通用的任务),越容易长出多套事实标准的兼容层**。

### 4.9 Router 模式:同机多进程,不是同进程多模型

`## 0` `## 5` 已经反复强调 router 模式做的是"同一台机器上管理多个模型进程",这里补一层具体机制,避免和"一个进程内切换模型权重"这类轻量方案混为一谈。`is_router_server` 为真时(`## 4.1` 已给出判据),router 进程本身**不加载任何模型**,它持有的 `server_models`(`## 2` 已列出)负责为每个要跑起来的模型拉起一个**独立的操作系统子进程**:日志原话"spawning server instance with name=%s on port %d"(`tools/server/server-models.cpp:1022`),用 `subprocess_option_no_window`(`:1044`)这类选项配置的子进程 API 真正 spawn 出去(`:1046` 失败时抛 `"failed to spawn server instance"`)——**每个模型是一个完整的、独立监听自己端口的 `llama-server` 进程**,不是同一个进程里切换权重指针。router 自己只做请求转发和状态代理(`## 2` 提到的 `models_routes->proxy_get`/`proxy_post`)。

这个设计直接决定了 `server_model_status`(`## 7` 已给出六态状态机)描述的是"子进程本身的生命周期",不是某种模型内部的缓存状态——`SLEEPING` 意味着子进程还活着但模型权重可能已经从显存换出(具体换出机制`未查证`,未继续深挖该状态的底层实现),`UNLOADED` 则对应子进程可能已经退出。和 vLLM 一个引擎实例服务一个模型、多模型靠起多个独立 vLLM 实例外加一层反向代理的部署方式(**本库推断**,未在 vLLM 源码里找到内建的"router"概念)相比,llama.cpp 把这层进程管理直接内建进了同一个二进制,不需要额外部署 nginx 或类似组件去做路由。

## 5. 设计决策与代价

### 决策一:纯 C/C++ 单体,零 Python 依赖

- **为什么这么设计**:README 把这条列为第一特性——"Plain C/C++ implementation without any dependencies"(`README.md:57`)。目标场景是"minimal setup"(`README.md:54`),一个不依赖运行时解释器、不需要提前装 CUDA/cuDNN/PyTorch 的二进制,可以直接拷到一台没联网、没 Python 环境、甚至没有独立显卡的机器上跑。这不是性能优化,是**部署面**的优化:`llama serve -hf <repo>` 一条命令背后不需要 `pip install` 任何东西,C++ 编译产物本身就是完整的可执行文件。
- **不这样会怎样**:vLLM/SGLang 的部署单位事实上是"Python 环境 + CUDA 驱动 + 一堆 pip 依赖"(可参照 [[01-vLLM-全景与代码地图]] 里 Python 产品码 932,277 行的规模),换一台机器意味着要么装同样的环境,要么发一个几 GB 的容器镜像。llama.cpp 反过来:`llama.cpp` 的 Python 代码(69,892 行)只在**转换模型格式**(`convert_hf_to_gguf.py`)和跑测试时才被用到——这些是构建期/准备期工作,不在 server 的请求处理路径上。**推理请求从进入到离开,不经过任何 Python 代码**,这是本篇开头说"和本库其他引擎完全不同"的最直接来源。
- **什么时候这是负担**:模型架构支持完全靠手写 C++,而且这份手写工作量是**两侧同时存在**的(`## 1.4` 已给出规模证据:C++ 图构建 151 个文件、Python 转换层 222 处 `@ModelBase.register`)。`src/models/` 下每个新架构都要一个新的 `.cpp` 文件手写图构建代码(比如 `src/models/llama.cpp`,250 行,定义 `load_arch_hparams`/`load_arch_tensors` 两个函数),再到 `src/llama-model.cpp:46`-`50` 这样的 `switch (arch)` 分支里手动接线——**支持一个新模型架构,要在 Python 转换层和 C++ 图构建层各写一遍**。vLLM/SGLang 加一个新模型通常是抄一份 PyTorch 的 `transformers` 实现,Python 层的胶水代码相对轻;llama.cpp 加一个新模型意味着要懂 ggml 的计算图 API、手写量化感知的张量创建代码——`docs/development/HOWTO-add-model.md`(196 行)本身的存在就说明这不是一件"随手改改配置"的事。这也直接抬高了贡献门槛:能审到"这个新模型的图搭得对不对"的人,必须同时懂模型结构和 ggml 底层 API,比"读一遍 HuggingFace 实现"要求更高。

### 决策二:slot 是固定数量、上下文均分的静态资源,不是动态增长的请求池

- **为什么这么设计**:llama.cpp 的目标场景是单机、常常是单用户或少量并发用户,`n_parallel` 一旦定下就对应一份确定的内存/显存预算(`n_ctx_slot = n_ctx / n_parallel`,`src/llama-context.cpp:293`)——这让资源用量在**启动时就能算清楚**,不需要运行时的显存碎片管理、块表、驱逐算法这套复杂机制。对于端侧和消费级硬件,可预测的资源占用比"极限情况下的最优吞吐"更重要。
- **不这样会怎样**:如果要做到 vLLM 那种"按 block 动态分配、任意请求都能借用任意空闲显存"的粒度,需要引入完整的分页 KV cache 管理(块表、引用计数、驱逐策略)——这套机制本身在 vLLM 里对应了 `_lab/out/struct_map.json` 统计的 `kv_cache` 子系统(据 [[01-vLLM-全景与代码地图]],9,581 行/13 文件)。llama.cpp 选择不引入这层复杂度,代价是**上下文利用率在请求长度分布不均时会打折**——一个 slot 的一半上下文空着,另一个 slot 的请求因为长度不够用而被截断或拒绝,两者互不相通(除非手动开 `--kv-unified`)。
- **什么时候可以不这样**:`--kv-unified` 打开后(`common/common.h:563` 默认 `false`,但 `n_parallel` 自动模式下会被强制设成 `true`,`tools/server/server.cpp:156`)`n_ctx_seq = n_ctx`(`src/llama-context.cpp:291`),意味着序列之间可以共享同一份完整长度的上下文预算而不是硬性均分——这是 llama.cpp 在"简单均分"和"完全动态"之间给出的一个折中开关,但仍然不等于逐 block 的动态分配,`未查证` `kv_unified=true` 时具体的底层内存布局是否已经做到块级共享,这一点留给以后核实,此处不下结论。

### 决策三:MCP 集成做成客户端/宿主,不做成服务端

- **为什么这么设计**:llama-server 本身已经是"一个能调用工具的 LLM 服务",让它去**连接**外部 MCP server(文件系统、搜索等能力提供方)、把工具结果喂给模型,比反过来**实现**一个 MCP 服务端等外部 Agent 框架来调用 llama-server,更贴合它"单机跑一个能干活的助手"这个使用场景。README 原话:"the server can expose tools coming from MCP servers"(`tools/server/README.md:347`-`349`),配置格式直接兼容 Cursor 的 `mcpServers` JSON(`:350`-`361`)——这是抄现成生态的配置格式,不是自造协议。
- **不这样会怎样**:如果做成 MCP 服务端,llama-server 要多维护一套 MCP 协议的 HTTP/SSE 传输层和会话状态,且这套服务端能力和它已有的 `/v1/chat/completions` 之类端点在语义上是两回事(一个是"被调用做工具",一个是"调用别人做工具")——现状是两者都没做,只做了后者(客户端角色)。README 也明确划了一条边界:"Note: `--ui-mcp-proxy` is unrelated, it only lets the Web UI reach remote MCP servers from the browser"(`tools/server/README.md:379`),说明连 Web UI 里那个"MCP 代理"都只是浏览器侧转发,不是 server 自己的 MCP 服务端实现。
- **什么时候可以不这样**:如果目标是把 llama-server 接入一个更大的多 Agent 编排系统(比如让 Claude Desktop 或某个 Agent 框架把 llama-server 当成一个可调用的 MCP 资源),现状的代码是做不到的——这需要额外实现 MCP 服务端协议,目前仓库里没有这部分代码,标注为**本库推断**下的一个明确缺口,`## 8` 会再提一次。

### 决策四:分布式仅有一条"proof-of-concept"级别的 RPC 路径,没有 PD 分离

- **为什么这么设计**:llama.cpp 的默认工作点是单机,`ggml-rpc`(`ggml/src/ggml-rpc/ggml-rpc.cpp`,2065 行)提供的是"把 ggml 设备通过 TCP 暴露给远程调用方"这个能力,用途是让一个大模型能跨机器切分张量/流水线并行来跑——这是"单次推理会话需要的显存超过一台机器"这个具体问题的答案,不是"多租户高并发服务"的答案。作者自己在文档里把它的成熟度标得很低:"currently in a proof-of-concept development stage…fragile and insecure. **Never run the RPC server on an open network or in a sensitive environment!**"(`tools/rpc/README.md:4`-`5`)。
- **不这样会怎样**:vLLM/SGLang/Dynamo 这类面向集群的引擎会实现请求级别的 KV 缓存跨节点传输(PD 分离,可参照篇目表里的 [[12-vLLM-PD分离与KV-Connector]]),让 prefill 和 decode 分别跑在专用节点上、通过高速互联传 KV 状态。llama.cpp 完全没有这层——`docs/`、`tools/` 全仓库搜不到"disaggregat"或"prefill/decode separation"相关的文档或代码痕迹(**源码为证**:grep 结果为空)。这意味着它没法在多机场景下做请求粒度的负载均衡与角色分工,只能做"router mode"这种同一台机器上多个模型进程的生命周期管理(`tools/server/server-models.h:113`)。
- **什么时候可以不这样**:如果确实需要多机分布式,`ggml-rpc` 是仓库里唯一的选项,但要接受它明确标注的"实验性、不安全"状态,且只解决"模型太大装不进一台机器"这个问题,不解决"这台集群要服务多少并发用户"的问题——这两个问题在 vLLM 的世界观里是分开被解决的(张量并行 vs PD 分离/多副本),在 llama.cpp 这里只有前者有对应实现。

### 决策五:GGUF 是自描述单文件格式,不是"权重+配置"分离的目录

- **为什么这么设计**:`## 3` 已经给出 GGUF 的文件布局——魔数、版本、KV 元数据、张量信息、张量数据全部顺序写进同一个文件(`ggml/include/gguf.h:1`-`30`)。这直接服务于"minimal setup"这条目标(`README.md:54`):一个 `.gguf` 文件可以单独复制、单独分发、单独校验完整性,加载时不需要额外去读一个旁路的 `config.json` 或 tokenizer 词表文件——**模型的结构超参数、tokenizer 词表、量化方式,这些原本分散在多个文件里的信息,都被塞进了同一个文件的 KV 元数据区**。对于"下载一个文件就能跑"这个使用体验,单文件格式是必要条件而不是可选项。
- **不这样会怎样**:HuggingFace 生态的标准格式是"一个模型目录",权重在 `*.safetensors`(可能还分片),结构超参数在 `config.json`,tokenizer 相关信息分散在 `tokenizer.json`/`tokenizer_config.json`/`special_tokens_map.json` 等多个文件里——vLLM 的默认加载器就是按这个假设写的:`DefaultModelLoader` 用 `safetensors` 作为其中一种登记的加载格式(`` `vllm:vllm/model_executor/model_loader/__init__.py:44` ``、`` `:60` ``),`` `vllm:vllm/model_executor/model_loader/default_loader.py:23` `` 专门处理"先下载 safetensors 索引文件、再按索引拉分片"这个多文件协同的场景。这套方案的好处是**复用了整个 HuggingFace 生态的工具链**(`transformers`、`accelerate` 等都认这个目录结构),但代价是加载一个模型天然涉及多个文件的一致性(版本对不对得上、缺文件时报错是否清晰)——GGUF 把这类一致性问题在转换阶段(`convert_hf_to_gguf.py`)就解决掉了,运行时不用再操心。
- **什么时候这是负担**:GGUF 的元数据 schema 本身在演进(`gguf.h` 头部注释描述的是当前 `GGUF_VERSION 3` 下的布局,`:42`),如果某个新模型架构引入了 GGUF 元数据 KV 表达不了的结构(比如某些非标准的层间连接方式),要么等上游给这个架构写转换脚本和加载代码(`## 5` 决策一已经指出这一步是手写 C++),要么根本转不成 GGUF——这是"自描述单文件"在**新模型追新速度**上要付的代价,HuggingFace 目录格式因为直接复用 `transformers` 的 Python 实现,理论上新模型发布当天就能跑,GGUF 往往要等社区补上转换脚本和图构建代码。

### 决策六:默认不鉴权、不加密,把安全配置完全交给使用者

- **为什么这么设计**:`middleware_validate_api_key`(`tools/server/server-http.cpp:206`-`208`)在 `api_keys` 为空时直接放行,`## 0` `## 7` 已给出证据。这个默认值服务的场景是"本机跑、只有本机用户能连上"——个人电脑上跑一个 `llama-server` 给自己写的脚本调用,或者局域网里一台可信设备之间通信,额外强制配一个 API key 只是增加使用摩擦,不增加实际安全性(攻击者如果已经能碰到这台机器,鉴权挡不住多少)。这和云端多租户服务"默认必须鉴权"的假设前提完全不同:云端服务默认对公网开放,本机服务默认只有本机能连。
- **不这样会怎样**:如果反过来默认强制鉴权(比如启动时自动生成一个随机 key 并打印到终端),对"本机跑着玩"这个场景会变成纯粹的摩擦——用户还要再复制一遍 key 传给客户端。但代价在 `## 7` 已经写清楚:一旦用户把 `--host` 从 `127.0.0.1` 改成 `0.0.0.0`(哪怕只是为了让同一局域网的另一台设备连上),不设防的默认值就从"没问题"变成"局域网内任何人都能调用,包括调用像 `exec_shell_command` 这样的高风险内置工具"。这个由使用场景切换触发的风险跃迁,代码层面没有任何提示或阻拦——唯一的提示是 `## 7` 提到的那条 `cors_origins == "*"` 警告,而且这条警告的触发条件是 CORS 配置,不是监听地址本身。
- **什么时候可以不这样**:任何"服务可能被除操作者以外的第三方访问到"的场景——多用户共享的开发机、暴露在公司内网、更不用说公网——都应该显式传 `--api-key`,并且如果开了内置工具,认真考虑 `## 5` 决策三提到的 `--tools-runtime` 隔离。这不是 llama.cpp 特有的取舍,但因为它的目标场景默认偏向"单用户本机",默认值选择了对这类场景更友好的一侧,把风险判断的责任转移给了使用者。

## 6. 同位对照:vLLM 在同一位置怎么做

**并发单位:固定 slot vs 动态 block 池。** llama.cpp 的 `server_slot`(`tools/server/server-context.cpp:196`)数量在启动时由 `-np` 定死,每个 slot 的上下文容量也是启动时算好的固定值(`n_ctx / n_parallel`,`src/llama-context.cpp:293`)。vLLM 的 `Scheduler`(`` `vllm:vllm/v1/core/sched/scheduler.py:73` ``)不预分配"槽位",而是维护一个 `BlockPool`(`` `vllm:vllm/v1/core/block_pool.py:143` ``)——所有请求共享同一个物理块池,`free_block_queue`(`` `vllm:vllm/v1/core/block_pool.py:181` ``)按需分配、按 LRU 近似策略驱逐,一个请求能用多少块只取决于当前显存里还有多少空闲块,不取决于它落在"第几个 slot"。这是两种资源模型的根本区别:llama.cpp 是**固定车位**(车位数量和大小开工前定死,某个车位空着别的车也停不进去),vLLM 是**弹性车道**(车道总容量固定,但怎么分给谁是每一步都在重算的动态决策)。

**调度粒度:整批一次 decode vs 逐步细粒度调度。** llama.cpp 的 `update_slots()` 每一步只做一件事:把所有活跃 slot 的下一步 token 拼进一个 batch,调一次 `llama_decode`(`tools/server/server-context.cpp:3612`)。vLLM 的 `Scheduler.schedule()`(`` `vllm:vllm/v1/core/sched/scheduler.py:484` ``)在每一步要决定"这一步谁的 prompt 继续处理、谁的 decode 继续、要不要触发抢占",本身就是一个比"把活跃 slot 的 token 拼一起"复杂得多的决策过程——因为 vLLM 面对的请求数可以远超"能同时塞进一次 batch"的 slot 数量,必须有一层显式的准入控制;llama.cpp 因为 slot 数量本来就小(通常个位数到几十),不需要这层复杂度,"活跃的都拼一起跑"就是全部逻辑。

**前缀复用:线性 LCP 扫描 vs 结构化索引。** llama.cpp 的 `get_available_slot` 对每个空闲 slot 做一次 `get_common_prefix` 比对(`tools/server/server-context.cpp:1531`),是 `O(n_slots × prompt_len)` 的线性扫描。SGLang 用基数树(`RadixCache`,可参照 [[03-SGLang-RadixAttention与前缀缓存]])做结构化前缀索引,vLLM 用块哈希表(可参照 [[04-vLLM-KV缓存与前缀缓存]])。三者的复杂度差异在 n 很大时会拉开明显差距,但 llama.cpp 的 n(slot 数)天生就小,线性扫描在这个规模下并不是真实瓶颈——**这不是"llama.cpp 的前缀缓存实现得简陋",而是"这个规模下更复杂的数据结构收益不明显"**,是和它的目标场景匹配的选择,不是短板。

**满载时的行为:无限期排队 vs 显式准入控制。** llama.cpp 在所有 slot 都忙时把请求丢进 `queue_tasks_deferred`(`tools/server/server-queue.cpp:79`),没有默认的队列深度上限,请求会一直等到有 slot 释放(`## 4.2` 已给出证据)。vLLM 的 `SchedulerConfig` 里的 `max_num_seqs`(`` `vllm:vllm/config/scheduler.py:63` ``)从准入层面就限制了同时处理的请求数,超出预算的请求走抢占(踢出正在跑的请求腾资源)或在调度器内部排队,这套逻辑本身是 `Scheduler` 代码量的重要组成部分。两边都"不会无理由拒绝请求",但 vLLM 的准入控制是运行时动态博弈的一部分,llama.cpp 的排队更接近"先来后到,能者上"的简单队列语义。

**进程拓扑:单进程 vs 三层进程。** llama.cpp 的 HTTP 线程池、任务队列、`update_slots()` 主循环、`llama_decode` 全都活在**同一个操作系统进程**里(`## 4.2` 已给出证据:HTTP 层多线程、推理主循环单线程,但没有进程边界)。vLLM 默认拓扑是前端进程 → 独立的 EngineCore 进程 → 每个并行 rank 一个的 worker 进程,进程之间靠 ZMQ + msgpack 序列化通信(细节见 [[02-vLLM-V1架构与EngineCore循环]])。这个差异不是"谁更先进",而是直接服务于各自的目标场景:vLLM 拆进程是为了让 Python GIL 不去抢占 GPU 忙循环的调度节奏(前端的 JSON 解析、chat template 渲染这类 CPU 工作不阻塞 GPU 侧);llama.cpp 完全没有 GIL 这个问题(它本来就不是解释型语言跑主循环),拆进程带来的序列化开销纯粹是负资产,所以选择不拆。

**模型格式:自描述单文件 vs 目录+索引。** `## 5` 决策五已经展开 GGUF 单文件 vs HuggingFace safetensors 目录的机制差异,这里补一个具体后果:llama.cpp 的 `llama-server` 启动命令只需要一个 `.gguf` 路径(或者一个 HF repo 名,内部转换成对应的 GGUF 下载),vLLM 的 `--model` 参数指向的是一个目录(或 HF repo 名,内部按 `safetensors`/`config.json`/`tokenizer.json` 等多个文件的组合去加载)。这个差异直接影响两边"支持一个新模型架构"要做的事:GGUF 路线下,新架构需要先有人写转换脚本把 HF 格式转成 GGUF、再有人在 `src/models/` 写图构建代码(`## 5` 决策一);vLLM 路线下,新架构如果已经在 `transformers` 里实现了,通常能直接复用其 Python 定义,不需要额外的格式转换步骤。

**硬件投入的重心不同。** `## 1.1` 给出的数字显示 llama.cpp 里代码量最大的后端是 `ggml-cpu`(91,460 行),超过 `ggml-cuda`(44,232 行)一倍还多。vLLM 的算子层(`csrc/`)几乎完全是为 CUDA/ROCm 写的(据 [[01-vLLM-全景与代码地图]],CPU 后端 `csrc/cpu/` 只有 38,110 行,是所有硬件后端目录里偏小的一个),因为 vLLM 的目标场景默认假设有 GPU。这组数字方向相反,恰好印证了 `## 1` 给出的判断:两个引擎的"硬件投入重心"分别指向了各自默认工作点最依赖的那类硬件。

**连接断开后的行为:可恢复流式传输 vs 无对应机制。** `## 4.6` 描述的"HTTP 连接断了、生成不断,靠 `conversation_id` 重新接上"这套机制(`tools/server/server-stream.h:11`-`12`),在 vLLM 的 `entrypoints/` 目录下找不到同名或同构的实现(**源码为证**:对 `resumable`/`conversation_id` 关键字的 grep 在 `_src/vllm/vllm/entrypoints/` 下零命中)。这不代表 vLLM"更差"——vLLM 的默认工作点是多副本、高并发的集群部署,单个请求的 HTTP 连接断开后由客户端重试通常足够,不必在服务端为每个生成维护一个断线重连缓冲区;llama.cpp 面向的是"一个用户、一次可能不太稳的本地/远程连接"这类场景,断线后不想直接丢掉已经生成到一半的长输出,这套机制才有意义。**同一个"连接会断"的现实,两个目标场景下值不值得专门解决,答案不一样**。

**异常/错误统一处理:手动包裹每条路由 vs 框架级全局注册。** llama.cpp 的 `ex_wrapper`(`tools/server/server.cpp:54`-`84`,`## 3` 已介绍)是手写的高阶函数,每次注册路由时显式包一层:`ctx_http.post("/path", ex_wrapper(handler))`,漏包一个路由,这个路由就没有统一错误处理。vLLM 用的是 FastAPI 框架自带的机制:`init_exception_handler`(`` `vllm:vllm/entrypoints/serve/exception_handling/register.py:27` ``)按异常**类型**注册一次性全局 handler(`HTTPException`、`RequestValidationError`、`VLLMError`、`ValueError` 等各自绑定一个处理函数),新增路由天然继承这层保护,不需要每个路由单独声明。这个差异本质上是"C++ 手写显式包裹"和"Python 框架级钩子"两种语言生态默认给出的能力差——llama.cpp 没有 FastAPI 这样的框架可以复用,`ex_wrapper` 是在没有这类基础设施时最直接的等价物。

**鉴权覆盖的协议面、比较方式都不同。** llama.cpp 的 `middleware_validate_api_key`(`tools/server/server-http.cpp:206`-`231`)同时接受 `Authorization: Bearer` 和 `X-Api-Key` 两种请求头(`## 2` 已给出证据),比较方式是普通的 `std::find`。vLLM 的 `AuthenticationMiddleware.verify_token`(`` `vllm:vllm/entrypoints/serve/middleware/authenticate.py:29` ``)只校验 `Authorization` 头的 `Bearer` scheme,但比较前先对 token 做 SHA-256 哈希、再用 `secrets.compare_digest`(常量时间比较,防时序侧信道)。这个细节对应 `## 0` 已给出的硬结论——vLLM 和 llama.cpp 都实现了 `/v1/messages`(Anthropic 协议兼容),但 llama.cpp 在鉴权这一层也顺带接住了 Anthropic 生态惯用的 `X-Api-Key` 请求头,协议兼容面比"多一个路由"更往前走了一步;而 vLLM 在密钥比较这一个更窄的点上,多做了一层时序攻击防护。**两边各自更用心的地方不一样**,不是哪边全面更严谨。

## 7. 踩坑与反直觉

- **`_lab/out/api_surface.json` 里的 36 条路由,方法全标成 `POST`——这是抽取脚本的盲区,不是 llama.cpp 只支持 POST。** 正则只匹配了 `ctx_http.post(...)` 这一种调用形态(`_lab/out/api_surface.json` 的 `"method": "regex"` 字段就是明确的自曝),`server.cpp` 里同样数量级的 `ctx_http.get(...)`(`GET /health`、`GET /props`、`GET /models`、`GET /lora-adapters`、`GET /slots`、`GET /v1/stream`、`GET /tools` 等)和 `ctx_http.del(...)`(`DELETE /models`、`DELETE /v1/stream`)一条都没被统计进去。读这份 JSON 时如果直接拿"36"当作"llama.cpp 一共暴露 36 个能力点",会明显低估。
- **`server-mcp.cpp` 这个文件名本身是个陷阱。** 第一反应容易读成"llama.cpp 实现了一个 MCP server",但代码显示它做的是反方向:llama-server 是 MCP **客户端**,去连接、调用外部 MCP server 提供的工具(`## 0` `## 5` 已给证据)。这个命名习惯(用被集成的协议名字命名"集成这个协议的模块")在很多项目里都存在,读源码时不能只看文件名猜方向,要看谁发起连接、谁实现协议的哪一端。
- **`--cache-ram` 默认是 8192(MiB),不是 0。** 容易假设"额外的缓存机制默认关闭,要显式开启才有",但 `common/common.h:616` 显示提示词缓存**默认就占用 8 GiB 主机内存**——在内存本就紧张的端侧设备上,这是一笔容易被忽略的隐性开销,需要显式传 `--cache-ram 0` 才能关掉。
- **`Q4_K_M` 这种命名格式暗示"4 bit,一种类型",实际是一张混合表。** `src/llama-quant.cpp:608`-`613` 显示同一个 `Q4_K_M` 预设下,不同层、不同角色的张量可能被分配 `Q4_K`、`Q5_K`、甚至 `Q6_K` 三种不同类型——量化后的模型文件大小和精度,是这张混合表和模型结构共同决定的结果,不能只凭预设名字里的"4"字推算出统一的比特占用。
- **`n_parallel` 的"自动"档不是"按硬件自动探测出一个合理值",只是硬编码成 4。** `tools/server/server.cpp:152`-`156` 的自动逻辑就是一个固定常数加上强制打开 `kv_unified`,不涉及探测显存、探测 CPU 核数之类的运行时判断——这和"auto"这个词通常暗示的"智能适配"有落差,读代码前不应该预设它做了更复杂的事。
- **测试/产品比 0.187 是全仓库口径,不能直接读成"这个仓库测试写得少"。** 和 vLLM 篇([[01-vLLM-全景与代码地图]])提到的同一类陷阱一样,`tests/` 目录里有相当比例是跨后端数值对拍脚本(验证 CPU 和 CUDA 算出同一个数),这类测试的性质和"业务逻辑单测"不同,直接套用比例数字下结论容易失真。
- **`--ctx-size` 设置超过训练长度不会报错,只会被静默下调并打一行警告。** `n_ctx_slot > n_ctx_train` 时,代码直接执行 `n_ctx_slot = n_ctx_train`(`tools/server/server-context.cpp:1213`-`1216`),日志里打一句 `SRV_WRN`。如果只看 `--ctx-size` 传参、不盯着启动日志,容易误以为服务真的按自己指定的长度在跑,实际吃到的上下文比预期短——这类"警告而非报错"的降级路径,和 `## 0` 提到的"CUDA Graph 掉回 eager 是完全静默的"（vLLM 篇的说法）是同一类问题:**降级本身没错,错的是它默认不够显眼**。
- **`ggml-hip/`、`ggml-musa/` 这两个目录名容易让人误判成和 `ggml-cuda/` 平级的独立实现。** `## 1.1` 的表格把它们单独列了行,但代码层面(`ggml/src/ggml-hip/CMakeLists.txt:92`、`ggml/src/ggml-musa/CMakeLists.txt:74`)两者都只是把各自厂商的转译工具接进构建系统、再定义 `GGML_USE_CUDA` 宏——**真正的算子实现只有一份,写在 `ggml-cuda/` 里**。目录并列的观感和"独立维护两套 AMD/摩尔线程专属核函数"这个印象容易对不上号,读目录列表时要多留意 `CMakeLists.txt` 里到底 `target_compile_definitions` 了哪个宏,不能只数目录个数。
- **`--tools all` 打开的不只是"读文件"这类温和能力,`exec_shell_command` 是货真价实的任意 shell 命令执行——但"没有隔离"是一个容易下得太快的结论。** `## 4.7` 已给出源码证据(`tools/server/server-tools.cpp:1250` 的 `permission_write = true`)。第一次看到"给模型加个工具"这个说法,容易联想到沙箱化、权限受限的执行环境;llama.cpp 的默认行为确实是"在 `llama-server` 进程本身的权限下直接跑"(README 用"do not enable in untrusted environments"划了红线,`tools/server/README.md:200`),但**隔离机制本身是存在的,只是默认关闭**:`--tools-runtime`(`tools/server/README.md:201`)可以把工具调用路由进一个 Docker/Podman 容器,甚至是一台通过 SSH 连接的远程主机执行(`server_tools_container_runtime`,`tools/server/server-tools.cpp:1861`起)。真正的陷阱不是"完全没有隔离方案",而是**默认值是"host environment"**——不显式配置 `--tools-runtime`,工具就是在宿主机本机权限下跑,这条红线现状确实是靠文档提醒,不是靠默认配置强制。
- **`llama-server` 默认不做任何鉴权,不是"默认用一个弱密钥"。** `middleware_validate_api_key`(`tools/server/server-http.cpp:206`-`208`)在 `params.api_keys` 为空时直接 `return true`——也就是不校验、任何人都能调用。这和很多服务"默认生成一个随机密钥、提示你去改"的习惯不同,llama.cpp 是彻底不设防,完全依赖用户自己想起来传 `--api-key`。`server.cpp` 里单独有一处运行时检查会在 `cors_origins == "*" && api_keys.empty()` 同时成立时打印醒目警告:"CORS is set to allow all origins('\*') and no API key is set…this can be a security risk"(`tools/server/server.cpp:312`)——这条警告和上面的鉴权中间件是同一个风险面的两个表现,凑在一起看:**默认配置下,一台把 `llama-server` 绑定到 `0.0.0.0` 的机器,几乎是对局域网内所有人开放的**。
- **同一个能力经常注册两条甚至三条路径,不是文档遗漏,是刻意保留的兼容层。** `/completion`(legacy)和 `/completions`/`/v1/completions`(`tools/server/server.cpp:241`-`243`)指向同一族 handler,`/embedding`(legacy)和 `/embeddings`/`/v1/embeddings`(`:253`-`255`)同理,`GET /models` 和 `GET /v1/models`(`:239`-`240`)也一样。第一次看 `## 2` 的代码地图容易以为这是路由表没整理干净,实际是 llama.cpp 早期(`/completion`/`/embedding` 这些没有 `/v1` 前缀的端点是它自己最早期的原生 API)和后来补上的 OpenAI 兼容层并存的结果——**新代码应该走哪条路径,取决于要不要 OpenAI 兼容,不是历史包袱该不该删**。
- **router 模式下"模型"这个词有一个独立于 GGUF 文件的生命周期状态机,和 slot 的生命周期是两回事。** `server_model_status`(`tools/server/server-models.h:30`-`38`)有 `DOWNLOADING`/`DOWNLOADED`/`UNLOADED`/`LOADING`/`LOADED`/`SLEEPING` 六个状态,管理的是"这个模型的子进程现在处于什么阶段",和 `## 3` 描述的 `slot_state`(管理"这个 slot 现在在干什么")是两套完全独立的状态机,分别对应 router 进程和某个具体模型子进程内部——读代码时看到"状态"两个字要先确认这是哪一层的状态,不能默认它们是同一个概念的不同叫法。
- **`## 4.9` 提到的"router 每个模型一个独立子进程",容易被误读成"和 vLLM 的多副本部署是一回事"。** 表面上都是"多个独立进程各自跑一份模型",但触发扩容/缩容的机制完全不同:vLLM 的多副本通常靠外部编排(K8s、`--data-parallel-size` 之类的启动期静态配置)决定副本数,运行时不会自己去起停副本;llama.cpp 的 router 子进程数量是**请求触发**的动态行为——`POST /models/load`(`## 2` 已列出)按需拉起一个新子进程,`SLEEPING` 状态(`server_model_status` 已给出)则是子进程存活但模型权重被换出。这更接近"按需唤醒的模型仓库",不是"预先配置好副本数的负载均衡集群"。
- **`server_task` 的 `is_parent()`/`child_tasks` 看起来像"高级调度信号",实际服务的是相对朴素的一对多协同场景,不是抢占或优先级调度。** `## 3` 已经指出这对字段支持"一个父任务扇出多个子任务、各自抢一个 slot"的模式;第一次看到"父子任务"容易联想到 vLLM 里请求优先级或者投机解码草稿-验证这类复杂调度概念,但 llama.cpp 这里对应的是更朴素的场景(比如同一次请求需要多个 slot 协同处理),背后不存在一个"谁的优先级更高"的判断逻辑——所有子任务平权,一起等资源,`slot_state::WAIT_OTHER` 纯粹是"等其他子任务的 slot 先处理完 prompt"这个同步点,不涉及优先级比较。


## 8. 可改进点

以下几条标注为**本库推断**,未提交 issue 或 PR 核实维护者是否已有计划:

1. **`_lab/api_surface.py` 对 llama.cpp 的抽取应该补上 `.get(`/`.del(` 两种调用形态。** 现状只认 `.post(`,导致这份统计给出的路由数字系统性偏小、且方法字段全错——这是本库工具自身的已知局限(`## 0` `## 7` 已经指出),修起来只需要把正则扩展到 `ctx_http\.(get|post|del)\s*\(`,不需要真的上 AST。
2. **`GET /slots?fail_on_no_slot` 这个"满载信号"是 opt-in 的,默认路径下客户端拿不到任何背压提示。** 请求满载时唯一的默认行为是无限期排队(`## 6` 已指出),客户端如果不主动轮询 `/slots` 就无法区分"排队中"和"卡住了"。给 `/v1/chat/completions` 这类端点在队列长度超过某个阈值时主动返回一个 429 加 `Retry-After`(而不是要求客户端另外发一个 `fail_on_no_slot` 查询),会让默认体验更接近其他推理服务的习惯用法。
3. **`slot_prompt_similarity` 的 LCP 扫描目前是纯线性的,场景一旦扩大到几十上百个 slot(比如未来某个大内存服务器场景把 `-np` 开得很大),`O(n_slots)` 的每请求扫描成本会线性增长。** 一个轻量级的改进是给每个 slot 的前缀维护一个短哈希摘要,先用哈希做粗筛,只对通过粗筛的候选做完整的 `get_common_prefix` 比对——不需要引入完整的基数树,量级上比现在更适合"slot 数量变大"这个假设性场景。
4. **MCP 只做了客户端方向,server-mcp.cpp 里没有 MCP 服务端实现。** 如果 llama-server 反过来暴露一个标准 MCP 服务端接口(把 `/completion`、`/embedding` 这些能力包装成 MCP resource/tool),会让它更容易被现有的 MCP 生态(Claude Desktop、各类 Agent 框架)直接当作后端接入,而不需要额外写一层适配代码——这是 `## 5` 决策三里提到的明确缺口。
5. **`--ctx-size` 超过训练长度时的降级路径应该更显眼。** `## 7` 已指出这条路径目前只打一句 `SRV_WRN`(`tools/server/server-context.cpp:1215`),日志级别和普通信息性日志混在一起。既然 `GET /props` 这类端点本来就会把当前生效的上下文长度上报给客户端,一个低成本的改进是在这种降级发生时,把"实际生效值和用户请求值不一致"这件事也放进 `GET /props` 的响应体里显式标注,而不是只依赖启动日志——这样即使运维没盯着启动输出,也能在运行时通过 API 发现配置被悄悄改写了。
6. **`ggml/src/ggml-hip/`、`ggml/src/ggml-musa/` 的目录命名和它们的实际角色(转译壳而非独立实现)不匹配,容易误导第一次读这部分代码的贡献者。** `## 7` 已指出这个落差。一个低成本的文档改进是在这两个目录各自的 `CMakeLists.txt` 顶部加一句注释级别的说明("this target transpiles CUDA sources via hipify/musify, see ggml-cuda/ for the actual kernels"),或者在项目的后端说明文档(`docs/backend/`)里显式画出"16 个目录、14 份独立实现"这条区分线,减少"数目录数等于数后端实现数"这种容易犯的误读。
7. **`--tools-runtime` 的隔离能力存在,但默认值是"不隔离"(host environment)。** `## 7` 已指出这条:`exec_shell_command` 这类高风险工具默认在宿主机本机权限下跑,要靠用户自己读文档、自己想起来去配 `--tools-runtime docker:<image>`(`tools/server/README.md:201`)才会走容器隔离。一个更安全的默认行为是:只要 `--tools` 里包含了 `permission_write = true` 的工具(`exec_shell_command`/`write_file`/`edit_file`),且没有显式配置 `--tools-runtime`,就在启动日志里把警告级别提到比现在更显眼的位置(甚至可以要求显式传一个 `--tools-runtime none` 才能跳过隔离),而不是让"隔离机制存在"和"隔离机制默认开启"这两件事被用户混为一谈。
8. **`middleware_validate_api_key` 的密钥比较不是常量时间的。** `## 6` 已给出对照:vLLM 的 `AuthenticationMiddleware.verify_token`(`` `vllm:vllm/entrypoints/serve/middleware/authenticate.py:29` ``)对 token 先哈希再用 `secrets.compare_digest` 做常量时间比较,llama.cpp 这边是普通的 `std::find`(`tools/server/server-http.cpp:231`)。对绝大多数"本机跑、只有自己用"的场景,这个差异几乎不构成实际威胁;但既然 `--api-key` 这个选项本身就是为"有必要设防"的场景准备的,补一个常量时间比较的成本很低(标准库 `std::equal` 配合按位或累加,或直接引入现成的实现),用一致的心智模型对待"启用鉴权"这件事,比只保护一半更清楚。
9. **`GET /props` 目前不区分"用户请求的配置"和"实际生效的配置"。** `## 8` 第 5 条已经指出 `--ctx-size` 被截断这一个具体场景,但同样的落差(用户传的参数值和运行时实际生效值不一致)理论上不止这一处。一个更系统性的改进是让 `server_context` 在启动期把"每个曾经被静默调整过的参数"统一记进一张小表,`GET /props` 响应体里加一个 `adjusted_params` 字段列出来,而不是每发现一处降级就单独打一次补丁式的日志——这样以后再新增一种静默降级路径,客户端可观测性是自动继承的,不需要每次都记得手动加一处上报。
10. **Router 模式下,子进程的资源账本(每个模型占多少显存/内存)没有在 `## 2` 提到的路由表面暴露出来。** `server_model_status`(`## 7` 已给出六态状态机)只回答"这个模型现在处于哪个阶段",不直接回答"这台机器现在还能不能再多起一个模型"。如果 router 能在生成状态响应时顺带汇总"当前已加载模型的显存占用总量 vs 硬件总量",对于同机多模型场景的运维决策(现在能不能再 `POST /models/load` 一个新模型)会比现在"试了才知道"更可预测。


## 9. 自测题与延伸阅读

**闭卷自测**(不看正文,能答上来才算过):

1. `llama-server` 默认(不显式传 `-np`)会起几个 `server_slot`?这个数字是运行时探测出来的还是写死的?对应哪一行代码?
2. 一个新请求到达、所有 slot 都在忙时,llama.cpp 的默认行为是什么?和显式带 `fail_on_no_slot=true` 查询 `GET /slots` 时的行为有什么不同?
3. `get_available_slot` 选 slot 有几层优先级?每一层各自的判据是什么?如果 `slot_prompt_similarity` 被设成 0,会跳过哪一层?
4. `server-mcp.cpp` 让 llama-server 扮演的是 MCP 协议里的客户端角色还是服务端角色?给出一条能支撑这个判断的具体代码或文档证据。
5. K-quant(如 `Q4_K`)和传统线性量化(如 `Q4_0`)在块结构上的核心区别是什么?"缩放因子本身也被量化"这句话对应哪个字段?
6. `Q4_K_M` 这个量化预设名,是"整模型统一用 `GGML_TYPE_Q4_K`"吗?如果不是,真实的分配逻辑在哪个文件、按什么维度分配?
7. llama.cpp 仓库里唯一和"跨机器"沾边的代码路径叫什么名字?它的文档对自己的成熟度给出了什么样的原话评价?
8. `ggml/src` 下代码量最大的后端目录是哪一个?比第二大的 CUDA 后端(以文件数计)大多少?为什么会是这个结果,而不是通常印象里的 CUDA?
9. GGUF 和 HuggingFace 的 safetensors 目录,哪一个是"自描述单文件"?这个差异如何影响两边"支持一个新模型架构"要做的工作?
10. `/v1/stream` 和 `/v1/streams/lookup` 这两个端点解决的是什么问题?`POST /v1/streams/lookup` 的接口设计里,哪一条安全边界防止了"随便一个调用方枚举出所有存活会话"?
11. `--tools all` 打开 `exec_shell_command` 之后,这个工具默认运行在什么权限下?要让它跑在隔离环境(容器或远程主机)里,需要额外配置哪个参数?
12. `POST /rerank` 在什么条件下会直接返回错误,不进入实际的重排序逻辑?这个端点同时兼容哪两种业界重排序 API 格式,靠请求体里的什么特征区分?
13. `qs.has_imatrix` 这个标志位会不会改变同一个量化预设(比如 `Q4_K_M`)下的张量类型分配?举一个源码里能找到的具体分支作为证据。

**延伸阅读**(双链只取自 `_PLAN.md` §6 名册):

- [[04-工程规模与代码结构对比]]——把本篇给出的 llama.cpp 规模数字(992,377 行、Python 只占 5.9%)放进 12 个引擎的横向对比里,看它在整个谱系里处在什么位置。
- [[11-开源推理引擎谱系图]]——本篇只深入讲了 llama.cpp 一家,这篇会把它和 TensorRT-LLM、LMDeploy、MLC-LLM 等"其他引擎"放进同一张谱系图,理清楚"端侧优先"这条产品线和"集群优先"那条产品线的分野。
- [[06-性能口径与基准陷阱]]——本篇按 `_PLAN.md` 第 1 节的红线完全没有给出任何实测吞吐/延迟数字(本机无 GPU),这篇讲的是"如果要谈 llama.cpp 的性能,一个负责任的说法需要带哪些口径"。
