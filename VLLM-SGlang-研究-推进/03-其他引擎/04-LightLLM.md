# LightLLM：纯 Python + 纯 Triton 的推理引擎

> **本篇取证基准**：`lightllm` @ `87881ae7`（2026-08-21）
> **一句话**：把算子交给 Triton、把进程边界交给 rpyc + 共享内存，赌的是"可读性换极限性能"这笔账划算。

## 0. 结论先行

- **这是本库 12 个引擎里唯一一个 CUDA 文件数为 0 的**：全仓 `.cu`/`.cuh`/`.cpp`/`.cc` 计数为 0，同时 129 个 `.py` 文件里含 `@triton.jit`（`_lab/out/repo_stats.json` 的 `kernels.cuda_files`/`kernels.triton_jit_files` 字段；本库现场复核 `grep -rl "@triton.jit" --include=*.py lightllm/` 同样得到 129）。vLLM 是 169 个 CUDA 文件 + 202 个 Triton 文件、SGLang 是 246 + 249——两家都是"CUDA 兜底 + Triton 加速某些算子"的混合路线，LightLLM 是本库里唯一把算子层**整体**押在 Triton 上的引擎，仓库里根本不存在 `csrc/` 这类 C++/CUDA 扩展目录。
- **多进程拓扑比"httpserver/router/model_rpc 三段论"更细**：真实结构是 HTTP 进程（`hypercorn` 跑 `lightllm.server.api_http:app`）→ 路由进程（`RouterManager`，ZMQ）→ **每个张量并行 rank 一个**的模型 worker 子进程（`ModelRpcServer`，走 **rpyc**，不是 ZMQ）→ 独立的去 token 化进程（`DeTokenizationManager`，ZMQ PUB/SUB）。四类角色、至少四个操作系统进程，`## 2` `## 4` 会给出逐跳的文件和行号。
- **KV 管理的粒度是"一个 token 一个格子"，不是块（block）**。`MemoryManager`（`lightllm/common/kv_cache_mem_manager/mem_manager.py:25`）把整份 KV 缓存开成一块 4D 张量，`KvCacheAllocator`（`lightllm/common/kv_cache_mem_manager/allocator.py:11`）用一个 `torch.arange` 数组做指针平移式分配——没有"块大小"这个概念，分配单位天然是 1 个 token。这正是 README 自称的 "token-level KC Cache management"（`README.md:63`，原文如此，疑似 KV 的笔误）——本库把这个设计叫法坐实为源码层面的 `MemoryManager`/`KvCacheAllocator`，而不是停在营销术语上。
- **请求对象本身直接是 C 结构体，摆在 POSIX 共享内存里**：`Req`（`lightllm/server/core/objs/req.py:95`）、`SamplingParams`（`lightllm/server/core/objs/sampling_params.py:266`）都继承 `ctypes.Structure`，由 `ShmReqManager`（`lightllm/server/core/objs/shm_req_manager.py:18`，内部 `from multiprocessing import shared_memory`）统一分配。跨进程真正走 `pickle`/ZMQ 的，只是一个指向共享内存槽位的整数索引（`GroupReqIndexes`），不是请求全部字段——这和 vLLM/SGLang"把整个请求对象 msgpack 一遍"的做法是两条不同路线，`## 5` `## 6` 会展开代价。
- **两套互相独立的 Triton 自动调优缓存机制同时存在**，一套是老的 `KernelConfigs`（`lightllm/common/kernel_config.py:11`，缓存目录 `lightllm/common/all_kernel_configs/<kernel_name>/`），一套是新的 `Autotuner`（`lightllm/common/triton_utils/autotuner.py:79`，缓存目录按 **Triton 版本 + 设备名 + kernel_name** 三层展开，`lightllm/common/triton_utils/autotuner.py:203`）。两套机制都把"参数=值"拼进文件名（`{block_seq=256,...}_NVIDIA_H200.json`），其中新机制的 `kernel_name` 里还带版本号（如 `"moe_sum_reduce:v1"`），拼进目录名后在 Windows 上就是非法字符——这正是本库 clone 这个仓库时 **360 个文件跳过**（`.clone_meta.json` 的 `files_skipped: 360`）的根因，`## 7` 会展开。
- **本地缺失的是「自动调优的结果」，不是「自动调优的机制」**：`_src/lightllm/.clone_skipped.txt` 里全部 360 条都是 `all_kernel_configs/` 或 `triton_utils/autotune_kernel_configs/` 下面的 `.json` 配置文件，产生这些文件的 Python 代码（`autotuner.py`、`kernel_config.py`，以及各 kernel 文件里的 `@autotune(...)` 调用点）本篇全部读到、全部能引用行号；但具体某个 GPU 型号在某个形状下调出来的 `BLOCK_M`/`num_warps` 数值，本篇**没有**看到，也不假装看到过。
- **API 表面不小**：29 条路由（22 个去重路径，`_lab/out/api_surface.json` 的 `summary`）同时兼容 OpenAI（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/v1/responses`）、Anthropic（`/v1/messages`、`/v1/messages/count_tokens`）、外加 LightLLM 自己的原生接口（`/generate`、`/generate_stream`）与 TGI 兼容层；CLI 开关 142 个（`api_cli.py`，`## 2` `## 5` 展开族群）。"轻量"指的是代码实现方式（纯 Python/Triton），不是功能面窄。

## 1. 它在系统里的位置

LightLLM（ModelTC 出品）是本库 `03-其他引擎/` 系列里第一篇写的文章，也是十二个引擎快照里代码量最小的"正经"服务化引擎之一——总行数 148,574（`_lab/out/repo_stats.json` 的 `totals.lines_counted`），Python 生产码 103,709 行、测试码 24,334 行，测试/生产比 0.235（同一文件 `totals.test_to_src_ratio`）。这个比例本身不能读成"测试不足"或"测试充分"，本篇不下结论——但值得记一笔：`test/kernel`（7 文件/2,032 行）、`unit_tests/common`（42 文件/5,396 行）、`test/advanced_config`（13 文件/7,303 行，`_lab/out/repo_stats.json` 的 `top_dirs`）三块测试目录规模都不小，说明"代码量小"不等于"没人测"。

**这份快照有两点必须先说清楚，否则后面所有引用都建立在错误前提上**：

1. **取证方式是 GitHub zipball，不是 `git clone`**。元数据落在 `_src/lightllm/.clone_meta.json`：`"source": "github-zipball"`、`"sha": "87881ae7ca9aa774db744ad38df22c95d80c0eb0"`、`"files_written": 1323`、`"files_skipped": 360`。这意味着这份快照没有 `.git` 目录，本篇的取证基准头写的 sha 是 zipball 声明的对应 commit，不是本地 `git log` 能反查的。
2. **360 个文件因为文件名带冒号（`:`）在 Windows NTFS 上无法落盘**，清单在 `_src/lightllm/.clone_skipped.txt`（360 行，与 `.clone_meta.json` 的 `files_skipped` 字段一致）。这些文件清一色是 Triton kernel 的自动调优配置，形如：

   ```
   lightllm/common/all_kernel_configs/_fwd_kernel_flash_decode_diverse_stage1:v1/{block_seq=256,gqa_group_size=16,...}_NVIDIA_H200.json
   lightllm/common/triton_utils/autotune_kernel_configs/triton_3.4.0/NVIDIA_H200/moe_sum_reduce:v1/{...}.json
   ```

   本篇但凡涉及"自动调优配置里具体存了什么数值"的地方，一律标注**本地不可见**，去 `.clone_skipped.txt` 核实清单本身，不假装读过内容——这一条既是取证纪律，也是 `## 7` 要展开的一个真实工程教训。

在 `_lab/out/struct_map.json` 按关键词切出的十个子系统里，LightLLM 的分布形状和 vLLM/SGLang 明显不同：`entrypoint` 34,076 行/168 文件遥遥领先（`function_call_parser.py` 2,108 行、`api_anthropic.py` 1,356 行、`api_openai.py` 1,166 行是该子系统三个最大文件），其次是 `attention` 8,256 行、`quantization` 8,232 行、`kv_cache` 5,778 行、`disagg`（PD 分离）2,685 行、`distributed` 1,967 行、`structured_output` 226 行；`scheduler` 和 `speculative` 两个子系统关键词匹配为 **0 行**——不是 LightLLM 没有调度器和投机解码（`req_queue/` 目录、`mtp_mode` 系列 CLI 开关都是实打实存在的），而是这套关键词+AST 抽取脚本按文件名/类名模式识别子系统，LightLLM 把调度逻辑摊在 `router/manager.py`、`router/req_queue/*.py` 这类不含"scheduler"字样的文件名里——**读数字要先确认脚本的识别口径，0 不代表"不存在"，可能代表"叫法不一样"**，这一条本身就值得记进 `## 7`。

## 2. 代码地图（文件 → 职责，带行号）

按"一次请求从命令行启动到真正跑 kernel"的顺序排列：

| 层 | 文件:行 | 干什么 |
|---|---|---|
| 打包声明 | `setup.py:3` | `package_data` 显式列出两套自动调优配置的 glob：`common/all_kernel_configs/*/*.json` 与 `common/triton_utils/*/*/*/*/*.json`——证明这些配置文件被当作要随包分发的产物，不是临时缓存 |
| 启动脚本 | `docs/CN/source/getting_started/quickstart.rst:51` | 官方文档给出的启动命令是 `python -m lightllm.server.api_server --model_dir ...`——没有 `console_scripts` 入口（`setup.py` 未声明），不像 vLLM 的 `vllm serve` 或 SGLang 的独立命令 |
| 进程入口 | `lightllm/server/api_server.py:9` | `launch_server`：按 `args.run_mode` 分派到 `pd_master_start`/`config_server_start`/`visual_only_start`/`normal_or_p_d_start` 四条启动路径之一 |
| 冷启动配置 | `lightllm/server/api_start.py:33` | `_set_envs_and_config`：`mp.set_start_method("spawn", force=True)`——多进程一律用 `spawn`，不用 `fork`（`## 5` 展开原因） |
| 冷启动拉起 | `lightllm/server/api_start.py:384`-`397` | `_launch_subprocesses` 尾段：先拉 `start_metric_manager`，再**同一批**拉起 `start_router_process` 与 `start_detokenization_process` |
| HTTP 进程 | `lightllm/server/api_start.py:412`-`429` | `normal_or_p_d_start`：用 `subprocess.Popen` 起独立的 `hypercorn` 进程跑 `lightllm.server.api_http:app`——**HTTP 服务本身是又一个独立操作系统进程**，不在 `_launch_subprocesses` 的 `mp.Process` 体系里 |
| 子进程握手工具 | `lightllm/utils/start_utils.py:18` | `SubmoduleManager.start_submodule_processes`：`mp.Process` 起子进程 + `mp.Pipe` 单向管道等一次 `"init ok"` 握手，失败就 `kill` 全部已起进程退出 |
| CLI 定义 | `lightllm/server/api_cli.py:7` | `--run_mode` 参数：`choices=['normal','prefill','decode','pd_master','config_server','visual_only']`，决定走哪条启动路径 |
| 启动参数镜像 | `lightllm/server/core/objs/start_args_type.py:9`-`12` | `StartArgs.run_mode` dataclass 字段，默认值/choices 与 `lightllm/server/api_cli.py:7` 手工保持一致——两处独立声明，`## 8` 会指出这是一个可改进点 |
| HTTP 路由 | `lightllm/server/api_http.py:365`、`:385`、`:405`、`:438`、`:461` | `/v1/chat/completions`、`/v1/completions`、`/v1/messages`（Anthropic）、`/v1/responses`、`/v1/models` 五条协议路由的装饰器与处理函数 |
| HTTP 请求门面 | `lightllm/server/httpserver/manager.py:46` | `HttpServerManager` 类定义：HTTP 进程里持有的、面向路由进程的请求管理器 |
| HTTP→路由 | `lightllm/server/httpserver/manager.py:314` | `HttpServerManager.generate`：接住一次生成请求的入口方法 |
| HTTP→路由（发送） | `lightllm/server/httpserver/manager.py:683` | `self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)`——发的是共享内存索引，不是完整请求体 |
| 路由进程入口 | `lightllm/server/router/manager.py:475` | `start_router_process`：路由进程的 `mp.Process` 目标函数 |
| 路由主循环 | `lightllm/server/router/manager.py:222`、`:277` | `RouterManager.loop_for_fwd`（外层循环）与 `_step`（单步调度+发起执行） |
| 路由收请求 | `lightllm/server/router/manager.py:411`-`426` | `_add_req`：从共享内存按索引取出 `Req` 对象，塞进 `req_queue`，同时把索引对象 pickle 发给去 token 化进程 |
| 拉起模型 worker | `lightllm/server/router/manager.py:128`-`139` | 按 `world_size`（tp 并行度）逐 rank 调 `start_model_process`，`asyncio.gather` 并行等待全部就绪 |
| 模型 worker 入口 | `lightllm/server/router/model_infer/model_rpc.py:194`、`:208` | `start_model_process`：`mp.Process(target=_init_env, ...)` 起一个独立 GPU 子进程，每个张量并行 rank 一个 |
| RPC 服务端 | `lightllm/server/router/model_infer/model_rpc.py:42`、`:187` | `ModelRpcServer(rpyc.Service)` 定义；`ThreadedServer(model_rpc_server, socket_path=socket_path, ...)`——rpyc 服务挂在 Unix domain socket 上，不走 ZMQ |
| RPC 客户端 | `lightllm/server/router/model_infer/model_rpc.py:130`、`:236` | `ModelRpcClient`：路由进程侧的 rpyc 客户端封装；`unix_connect(socket_path, ...)` 建立连接 |
| 去 token 化进程 | `lightllm/server/detokenization/manager.py:171`、`:33`-`37` | `start_detokenization_process` 入口；`zmq.PULL` 收路由进程结果，`zmq.PUB` 广播给 HTTP 进程 |
| KV 内存管理 | `lightllm/common/kv_cache_mem_manager/mem_manager.py:25`、`:86` | `MemoryManager` 类；`kv_buffer = torch.empty((layer_num, size+1, 2*head_num, head_dim), ...)`——一整块按层排布的 4D 张量 |
| KV 分配器 | `lightllm/common/kv_cache_mem_manager/allocator.py:11`、`:31` | `KvCacheAllocator`；`alloc(need_size)`——`torch.arange` 数组上做指针平移，分配单位是 1 个 token |
| 前缀缓存 | `lightllm/server/router/dynamic_prompt/radix_cache.py:22`、`:101` | `TreeNode`；`RadixCache`——基数树覆盖在 `MemoryManager` 的 token 索引之上做前缀复用 |
| 共享内存请求对象 | `lightllm/server/core/objs/req.py:95`、`lightllm/server/core/objs/shm_req_manager.py:18` | `Req(ctypes.Structure)`；`ShmReqManager`——请求对象本体常驻共享内存，不逐字段序列化传递 |
| Triton kernel 示例 | `lightllm/common/basemodel/triton_kernel/norm/rmsnorm.py:10` | `_rms_norm_fwd_fused`：RMSNorm 的 `@triton.jit` 实现 |
| Triton kernel 示例 | `lightllm/common/basemodel/triton_kernel/att/decode_att/gqa/flash_decoding/gqa_flash_decoding_stage1.py:8` | `_fwd_kernel_flash_decode_stage1`：GQA flash-decoding 第一阶段 kernel |
| Triton kernel 示例 | `lightllm/common/basemodel/triton_kernel/fused_moe/moe_sum_reduce.py:8`、`:63` | `_moe_sum_reduce_kernel`（`@triton.jit`）；`@autotune(kernel_name="moe_sum_reduce:v1", ...)` 装饰器把它接进自动调优系统 |
| 自动调优（新） | `lightllm/common/triton_utils/autotuner.py:79`、`:203`-`217` | `Autotuner` 类；`cache_dir` 属性——按 `triton_version/device_name/kernel_name` 三层建缓存目录 |
| 自动调优（旧） | `lightllm/common/kernel_config.py:11`、`:17`、`:44` | `KernelConfigs` 抽象基类；`get_config_file_name`（`{参数=值}_设备名.json` 命名规则）；`store_config`（写盘） |
| 自动调优使用方 | `lightllm/common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding_config.py:7`-`8` | `MlaDecodeAttentionKernelConfig(KernelConfigs)`：`kernel_name = "mla_decode_attentnion"`（原文拼写如此），MLA 解码 attention 走旧的 `KernelConfigs` 机制 |

以上 30 条只是骨架，`## 3` `## 4` 会把其中若干条摊开细读。

## 3. 核心数据结构

- **`StartArgs`**（`lightllm/server/core/objs/start_args_type.py:8`）——启动参数的 dataclass 化身，与 `api_cli.py` 里 142 个 `argparse` 定义**手工保持字段/默认值一致**，两处各自独立维护（`## 8` 展开风险）。
- **`SamplingParams`**（`lightllm/server/core/objs/sampling_params.py:266`）——不是普通 Python 类，是 `ctypes.Structure`：`_do_sample`/`_temperature`/`_top_p`/`_top_k` 等 7 个字段（`n_fields: 7`，`_lab/out/api_surface.json`）直接以 C 兼容内存布局存在，配套 `verify()`（`lightllm/server/core/objs/sampling_params.py:422`）做合法性校验、`to_dict()`（`:483`）转成 Python 字典供日志/调试用。它和同文件里的 `StopSequence`（`:26`）、`GuidedGrammar`（`:135`）、`AllowedTokenIds`（`:195`）等结构体是"一组小 C 结构体拼成一个采样配置"的设计。
- **`Req`**（`lightllm/server/core/objs/req.py:95`）——同样是 `ctypes.Structure`，是"一条请求"在共享内存里的真身；`ChunkedPrefillReq`（`:472`）、`TokenHealingReq`（`:526`）分别是它针对分块预填充、token healing 场景的子类扩展。
- **`ShmReqManager`**（`lightllm/server/core/objs/shm_req_manager.py:18`）——`Req` 对象的分配器，内部 `from multiprocessing import shared_memory`（同文件 4 行）打开/映射共享内存段，`alloc_req_index`（`:91`）分配一个槽位、`get_req_obj_by_index`（`:123`）按索引取回对象引用（多个进程各自 `attach` 到同一段共享内存，取到的是同一份数据，不是拷贝）。
- **`GroupReqIndexes`**（引用于 `lightllm/server/router/manager.py:411` 的 `_add_req` 参数类型）——**真正跨进程用 `pickle`/ZMQ 传递的对象**，本质是若干个 `shm_req_indexes`（整数）加一点元信息（`multimodal_params`、`time_mark`）；请求的大字段（prompt token id 列表、生成参数结构体）全部留在共享内存里，跨进程传的只是"去哪个槽位取"这一句话。
- **`MemoryManager`**（`lightllm/common/kv_cache_mem_manager/mem_manager.py:25`）——持有真正的 GPU KV 显存（`kv_buffer`，`:86`），`get_cell_size`（`:58`）算出"一个 token 的 KV 占多少字节"，`profile_size`（`:61`）在 `size` 未显式指定时跑一次真实显存查询反算最大 token 数；`ReadOnlyStaticsMemoryManager`（`:249`）是给路由进程用的**只读**镜像，路由进程不持有 GPU 显存，靠它读共享的统计数字做调度估算。
- **`KvCacheAllocator`**（`lightllm/common/kv_cache_mem_manager/allocator.py:11`）——`mem_state`（一个 `torch.arange(0, size)` 数组）+ `mark_start`/`mark_end` 两个指针，`alloc`（`:31`）和 `free`（`:56`）都是纯粹的指针平移操作，不涉及块表、不涉及位图扫描。
- **`TreeNode` / `RadixCache`**（`lightllm/server/router/dynamic_prompt/radix_cache.py:22`、`:101`）——基数树节点存的 `token_mem_index_value` 就是 `KvCacheAllocator` 分配出来的那些 token 索引；`RadixCache` 是覆盖在 `MemoryManager` 之上的前缀复用层，`insert`（`:129`）、`evict_tree_set`（`:118`，按 `SortedSet` 维护驱逐优先级）实现的思路和 SGLang 的 RadixAttention 前缀树同源（LightLLM README 自己列出"SGLang（部分借用了 LightLLM 的 kernel）"，`README.md:55`）。
- **`ModelRpcServer` / `ModelRpcClient`**（`lightllm/server/router/model_infer/model_rpc.py:42`、`:130`）——rpyc 服务/客户端一对儿，`exposed_init_model`（`:56`）、`exposed_get_max_total_token_num`（`:113`）是服务端暴露的方法，客户端用 `rpyc.async_(f)`（`:135`）包一层异步适配，让 `asyncio` 的路由进程可以 `await` 一个本质是同步 RPC 调用的操作。
- **`Autotuner`**（`lightllm/common/triton_utils/autotuner.py:79`）——`cached_configs`（`static_key → {run_key: config}` 两层字典）、`fast_match_configs`（同结构的一层缓存加速）、`cache_dir`（`:203`，按 `triton_version/device_name/kernel_name` 分层的磁盘缓存路径）；`KernelConfigs`（`lightllm/common/kernel_config.py:11`）是更早的同类机制，缓存目录只按 `kernel_name` 一层展开，不区分 Triton 版本。

## 4. 主流程走读

以 `POST /v1/chat/completions`、非流式、单机单卡、`run_mode=normal` 为基准路径，逐跳给出文件和行号：

1. **命令行启动。** `python -m lightllm.server.api_server --model_dir ...`（`docs/CN/source/getting_started/quickstart.rst:51`）触发 `lightllm/server/api_server.py:9` 的 `launch_server`，先 `torch.multiprocessing.set_start_method("spawn")`（`lightllm/server/api_server.py:14`），再按 `run_mode` 分派——默认路径走 `normal_or_p_d_start`（`lightllm/server/api_start.py:408`）。

2. **`_launch_subprocesses` 冷启动。** `lightllm/server/api_start.py:36` 的 `_launch_subprocesses` 先做一长串参数自动推导与合法性断言（多模态检测、显存比例校验、`chunked_prefill_size` 与 `batch_max_tokens` 的关系推导等，`lightllm/server/api_start.py:39`-`321`），再依次拉起可选的多模态缓存/视觉/音频/CPU 缓存子进程，最后 `lightllm/server/api_start.py:384`-`397` 用同一个 `process_manager.start_submodule_processes` 调用**同时**拉起 `start_router_process` 和 `start_detokenization_process` 两个进程。

3. **HTTP 进程单独起。** 回到 `normal_or_p_d_start`（`lightllm/server/api_start.py:408`），`_launch_subprocesses` 跑完之后，`lightllm/server/api_start.py:429` 用 `subprocess.Popen(command)` 起一个**外部**的 `hypercorn` 进程，命令行里指定跑 `lightllm.server.api_http:app`（`lightllm/server/api_start.py:412`-`425`）——这是第五个进程角色（前面已经有：主进程本身、路由进程、去 token 化进程、N 个模型 worker 进程），HTTP 服务不在 `mp.Process`/`SubmoduleManager` 体系内，是单独 `subprocess.Popen` 出来的。

4. **路由进程内部再拉模型 worker。** `RouterManager.wait_to_model_ready`（`lightllm/server/router/manager.py:114`）在路由进程内部执行：按 `world_size`（等于 `--tp`）逐 rank 调用 `start_model_process`（`lightllm/server/router/manager.py:128`-`139`，`asyncio.gather` 并行等待），每次调用都在 `lightllm/server/router/model_infer/model_rpc.py:208` 起一个新的 `mp.Process`，目标函数 `_init_env`（`lightllm/server/router/model_infer/model_rpc.py:169`）在子进程里构造 `ModelRpcServer` 并用 `ThreadedServer` 挂到一个随机生成的 Unix socket 路径上（`lightllm/server/router/model_infer/model_rpc.py:186`-`190`，`_generate_unix_socket_path` 在 `:241`）。路由进程随后用 `unix_connect` 连回去（`lightllm/server/router/model_infer/model_rpc.py:236`），拿到一个 `ModelRpcClient`。这一步之后，路由进程调用每个 worker 的 `exposed_init_model`（`lightllm/server/router/model_infer/model_rpc.py:56`）完成模型加载与显存 profile。

5. **HTTP 请求进来。** `lightllm/server/api_http.py:365` 的路由接住 `/v1/chat/completions`，内部走到 `HttpServerManager.generate`（`lightllm/server/httpserver/manager.py:314`）——这是 HTTP 进程内的请求门面，不是模型执行本身。

6. **请求对象写共享内存，索引发路由。** `generate` 内部通过 `ShmReqManager` 在共享内存里分配一个 `Req` 槽位（`ShmReqManager.alloc_req_index`，`lightllm/server/core/objs/shm_req_manager.py:91`），把 prompt token id、`SamplingParams` 结构体写进去，然后把**索引**打包成 `GroupReqIndexes`，通过 `self.send_to_router.send_pyobj(...)`（`lightllm/server/httpserver/manager.py:683`）用 ZMQ PUSH 发给路由进程——这一步传输的字节量只和"多少个索引整数"有关，跟 prompt 有多长无关。

7. **路由进程收请求、入队调度。** `RouterManager._add_req`（`lightllm/server/router/manager.py:411`）从共享内存按索引取回 `Req` 对象（`shm_req_manager.get_req_obj_by_index`，`lightllm/server/core/objs/shm_req_manager.py:123`），塞进 `self.req_queue`（`lightllm/server/router/manager.py:424`），同时把 `group_req_indexes` 转手 pickle 一份发给去 token 化进程（`lightllm/server/router/manager.py:426`）——**去 token 化进程和路由进程在这一刻拿到的是同一份共享内存里的同一个 `Req`**，路由发的这条 ZMQ 消息只是告诉对方"去哪个槽位读"。

8. **调度循环推进一步。** `RouterManager.loop_for_fwd`（`lightllm/server/router/manager.py:222`）是路由进程的主循环，每轮调 `_step`（`lightllm/server/router/manager.py:277`）：从 `req_queue` 生成一个新批次（`_generate_new_batch`，`lightllm/server/router/manager.py:428`）或推进正在跑的批次，然后通过 `ModelRpcClient` 把这一步的批次信息 RPC 发给全部模型 worker 并发执行。

9. **模型 worker 真正跑 kernel。** 每个 `ModelRpcServer`（`lightllm/server/router/model_infer/model_rpc.py:42`）持有的 `backend`（`ChunkedPrefillBackend` 等，`lightllm/server/router/model_infer/model_rpc.py:99` 附近按运行模式选择具体子类）驱动模型前向：注意力/RMSNorm/MoE 等算子全部经由 `@triton.jit` kernel（如 `lightllm/common/basemodel/triton_kernel/norm/rmsnorm.py:10`、`lightllm/common/basemodel/triton_kernel/att/decode_att/gqa/flash_decoding/gqa_flash_decoding_stage1.py:8`）执行，KV 写入/读取通过 `MemoryManager.kv_buffer`（`lightllm/common/kv_cache_mem_manager/mem_manager.py:86`）按 `KvCacheAllocator.alloc` 分配出的 token 索引（`lightllm/common/kv_cache_mem_manager/allocator.py:31`）直接定位；如果开了动态前缀缓存，`RadixCache`（`lightllm/server/router/dynamic_prompt/radix_cache.py:101`）会先尝试复用已有前缀对应的 token 索引，减少新分配量。

10. **结果沿共享内存 + ZMQ 回流。** 新生成的 token 写回共享内存里的 `Req` 对象；去 token 化进程（`DeTokenizationManager`，`lightllm/server/detokenization/manager.py:25`）通过 `zmq.PULL`（`:33`）收到路由进程转发的索引消息后，从共享内存读出新 token id、跑 tokenizer 增量解码，再用 `zmq.PUB`（`:36`）广播出去。HTTP 进程里的 `HttpServerManager` 用 `zmq.SUB`（`lightllm/server/httpserver/manager.py:100`-`101`，`setsockopt(zmq.SUBSCRIBE, b"")` 订阅全部消息）收到解码结果，包成 SSE 流或一次性 JSON 响应返回给客户端。

十步里一共跨了 HTTP 进程 → 路由进程 → N 个模型 worker 进程 → 去 token 化进程 → 回到 HTTP 进程这条链，其中 HTTP↔路由、路由↔去 token 化用 ZMQ，路由↔模型 worker 用 rpyc，请求本体大字段全程留在共享内存里不挪窝——这条路径的取舍在 `## 5` `## 6` 展开。

## 5. 设计决策与代价

### 决策一：算子层整体押在 Triton 上，不写手工 CUDA/C++

- **为什么这么设计**：一份 kernel 代码能跨硬件后端复用——`lightllm/utils/device_utils.py:92`-`93` 的 `is_musa()` 判定 `torch.version.musa`，`:343` 断言 `platform_name in ["cuda", "musa"]`，CLI 里对应 `--hardware_platform`（`lightllm/server/api_cli.py:884`，取值 `cuda | musa`）——同一份 `@triton.jit` kernel 靠 Triton 自己的后端切换，不需要为 NVIDIA CUDA 和摩尔线程 MUSA 各写一份 `.cu`/`.cpp`。这也是 Python 开发者能直接改 kernel（不用装 `nvcc`/配 C++ 编译工具链、不用等 C++ 扩展重新编译链接）的直接来源——`## 2` 列出的 129 个含 `@triton.jit` 的文件全部是纯 `.py`，改一行代码、重启进程就生效。
- **不这样会怎样**：参照 vLLM 的对照组——`csrc/` 下按功能（`attention/`、`moe/`、`quantization/`、`rocm/`、`cpu/`）和 ABI 新旧两套（`csrc/torch_bindings.cpp` vs `csrc/libtorch_stable/`）拆出的目录结构，169 个 CUDA 文件之外还要单独维护 ROCm、CPU 兜底实现（`01-vLLM-全景与代码地图.md` `## 1.2`）——每加一个新硬件后端，要新增一整套编译产物和绑定声明；LightLLM 把这层复杂度让给了 Triton 编译器本身。
- **什么时候这是负担**：三点都有源码为证。**编译期开销真实存在且需要专门处理**——`lightllm/common/basemodel/basemodel.py:1107`-`1120` 的 `_autotune_warmup` 方法会在服务真正接客之前，按 `[1, 4, 8, 16, 32, ..., 4096]` 一串固定长度（外加 `batch_max_tokens` 本身）逐个跑一遍前向传播来触发 JIT 编译和自动调优（`Autotuner.start_autotune_warmup()`/`end_autotune_warmup()` 包住整个过程，`lightllm/common/basemodel/basemodel.py:1110`、`:1177`）——如果没有这一步，线上第一批真实请求会撞上编译延迟。**极限性能通常够不到手写 CUDA/CUTLASS 的水位**（本库推断：本机无 GPU，未做实测对比，仅从"Triton 是更高层的 DSL，编译器自动排的指令调度不一定优于专家手工调优的 CUTLASS/PTX"这个业界共识推断，未见到 LightLLM 自己发布的量化对比数据）。**依赖 Triton 版本**——`get_triton_version()`（`lightllm/common/triton_utils/autotuner.py:443`-`444`，返回 `f"triton_{triton.__version__}"`）直接拼进缓存目录路径（`lightllm/common/triton_utils/autotuner.py:216`），意味着升级 Triton 版本后，历史调好的配置全部找不到，要么退回默认 `run_config`（性能下降但不报错），要么重新跑一遍 autotune。**autotune 结果要随硬件分发**——`setup.py:3` 的 `package_data` 显式声明要把 `common/all_kernel_configs/*/*.json` 和 `common/triton_utils/*/*/*/*/*.json` 打进包里，这意味着"kernel 在什么硬件上跑得快"这份知识，要么随发布物一起分发（覆盖不到用户的新硬件型号），要么用户自己现场跑一遍 autotune（`## 7` 会展开这条路径在 Windows 上直接翻车的真实案例）。

### 决策二：路由进程与模型 worker 进程之间用 rpyc，不是 ZMQ + 自定义消息协议

- **为什么这么设计**：rpyc 把跨进程调用包装成"看起来像调本地方法"的语义——`ModelRpcServer.exposed_init_model`（`lightllm/server/router/model_infer/model_rpc.py:56`）在服务端就是一个普通方法，客户端 `ModelRpcClient` 用 `rpyc.async_(f)`（`lightllm/server/router/model_infer/model_rpc.py:135`）包一层就能 `await`，不需要像消息队列那样手工定义一套请求类型枚举、在两端各自写 dispatch 分支。对一个把"纯 Python、易读、方便做研究"当卖点的引擎（`README.md:63`）来说，这条路省了不少样板代码。
- **不这样会怎样**：vLLM 的对照做法是显式协议——`EngineCoreRequestType`（`vllm:vllm/v1/engine/__init__.py:284`）是个专门的枚举，标记每条 ZMQ 消息是新请求/abort/控制指令，`MsgpackEncoder`/`MsgpackDecoder`（`vllm:vllm/v1/serial_utils.py:136`、`:313`）负责编解码；SGLang 同样是"没有共享内存、没有 RPC 调用"的纯 ZMQ push/pull 管道（`02-SGLang-Scheduler事件循环.md` `## 0`）。两家都要为"这条消息是什么类型"这件事单独建模；LightLLM 用 rpyc 把这层折叠进了"调用哪个 `exposed_*` 方法"里，代价是路由进程和模型 worker 进程之间不再是纯粹的消息传递，而是带了一层 RPC 框架（`ThreadedServer`，`lightllm/server/router/model_infer/model_rpc.py:187`）。
- **什么时候可以不这样／是负担**：rpyc 的连接配置显式打开了 `allow_pickle=True`（`lightllm/server/router/model_infer/model_rpc.py:187`、`:236`），这意味着这条 Unix socket 上收发的对象可以是任意 Python 对象（通过 `pickle` 反序列化）——本篇不对安全性下结论（**本库推断**，未做安全评估），但如果未来某个部署形态把这条 rpyc 连接暴露到不受信的网络（当前默认是本机 Unix socket，风险面有限），`allow_pickle=True` 会是需要重新审视的一条配置。另外，rpyc 底层是同步阻塞语义，`ModelRpcClient` 靠 `asyncio.to_thread(ans.wait)`（`lightllm/server/router/model_infer/model_rpc.py:144`）把它塞进 asyncio 事件循环，这一层线程池转发在高并发多 worker 场景下的开销特性和原生异步 ZMQ 不同——具体量级本篇未做压测，留作待验证的观察点。

### 决策三：请求对象整体常驻共享内存，跨进程只传一个索引

- **为什么这么设计**：`Req`（`lightllm/server/core/objs/req.py:95`）、`SamplingParams`（`lightllm/server/core/objs/sampling_params.py:266`）都是 `ctypes.Structure`，由 `ShmReqManager`（`lightllm/server/core/objs/shm_req_manager.py:18`）统一分配在 POSIX 共享内存段上。HTTP 进程写入一次、路由进程和去 token 化进程各自按索引 `attach` 读取（`shm_req_manager.get_req_obj_by_index`，`:123`）——三个进程看到的是同一份内存，不是三份互相独立的拷贝。跨进程真正用 `pickle` 传的 `GroupReqIndexes`（`lightllm/server/router/manager.py:411` 的参数类型）只装了整数索引和少量元信息，序列化开销和 prompt 长度无关。
- **不这样会怎样**：vLLM 的对照做法是每次跨进程都重新编码一遍完整请求负载——`EngineCoreRequest`（`vllm:vllm/v1/engine/__init__.py:107`）经 `MsgpackEncoder`（`vllm:vllm/v1/serial_utils.py:136`）序列化后通过 ZMQ 发送；SGLang 的 `TokenizerManager`/`Scheduler`/`DetokenizerManager` 之间同样是每条消息各自编码。对短 prompt 而言这点开销可以忽略，但 LightLLM 的设计意味着"路由进程要读、去 token 化进程也要读"这类**同一份数据被多个进程重复访问**的场景，完全不产生额外的序列化/反序列化成本——因为压根不需要传数据本体。
- **什么时候可以不这样**：共享内存方案要求 `Req`/`SamplingParams` 这类结构体的字段类型在 `ctypes.Structure` 定义时就**提前钉死**（`lightllm/server/core/objs/sampling_params.py:266` 往下的字段列表是编译期确定的 C 结构体布局），不能像 Python `dict`/`msgspec` 结构体那样运行时动态增删字段——每加一个新的采样参数字段，都要改这个 `ctypes.Structure` 定义并保证跨进程的内存布局一致。如果一个引擎的请求负载本身很小、进程数不多（比如单机单进程部署，不涉及"多个进程重复读同一份数据"这个场景），共享内存管理的复杂度（生命周期、清理、跨平台的 `/dev/shm` 依赖）可能就得不偿失，直接走 vLLM/SGLang 那种"每次都序列化"的路子反而更简单。

### 决策四：KV 缓存按单 token 粒度分配，不做块（block/page）抽象

- **为什么这么设计**：`KvCacheAllocator`（`lightllm/common/kv_cache_mem_manager/allocator.py:11`）只用一个 `torch.arange` 数组和 `mark_start`/`mark_end` 两个指针，`alloc`（`:31`）、`free`（`:56`）都是纯粹的区间平移，没有块表、没有位图、没有"块内部分命中"这类需要额外处理的边界情况。配合 `RadixCache`（`lightllm/server/router/dynamic_prompt/radix_cache.py:101`）做前缀复用时，`TreeNode.token_mem_index_value` 直接存的就是这些 token 级别的索引——不需要像"块粒度"设计那样考虑一个块内一部分 token 命中、另一部分没命中的折中处理。
- **不这样会怎样**：vLLM 默认 `block_size=16`（`vllm:vllm/config/cache.py:79`），一个块只有整块存满才能被缓存复用，短前缀命中要么退化成"部分块"特判、要么干脆不命中；`04-vLLM-KV缓存与前缀缓存.md` 专门用一整节（`_block_hash`/`_block_hash_num_tokens`）处理这类"块粒度和缓存粒度不对齐"的复杂度。LightLLM 从根上没有块的概念，天然不用面对这类问题——但代价是每次分配/释放都是标量级别的操作，没有"批量搬一个块"的摊销效果。
- **什么时候可以不这样／是负担**：token 粒度的指针平移分配器假设分配和释放大体符合"后来先释放"的栈式模式（`lightllm/common/kv_cache_mem_manager/allocator.py:31`-`49` 里专门用一个三倍大小的 `_mem_state_return` 换手缓冲区来避免异步场景下的内存竞争，注释原话"避免异步情况下的内存竞争"）——这本身就是为了让"极简的指针平移"能撑住并发场景打的补丁。SGLang 的默认 `page_size` 同样是 1（`sglang:python/sglang/srt/arg_groups/overrides.py:2396`，token 粒度），但它的分配器抽象里保留了"page"这个概念，方便某些 kernel（如 FlashMLA）反过来把 `page_size` 强制改写成 64（`02-SGLang-Scheduler事件循环.md` 之外的取证见 `04-SGLang-内存池与KV布局.md` `## 0`）；LightLLM 的 `KvCacheAllocator` 从命名到实现都没有留"页"这层抽象，如果未来要接入要求大页对齐的 kernel（某些稀疏注意力实现），大概率需要重新设计这一层，不是加个参数就能切换。

### 决策五：两套自动调优缓存机制并存，没有统一到一套

- **为什么这么设计**：这是历史演进的产物——`KernelConfigs`（`lightllm/common/kernel_config.py:11`）是更早的机制，缓存目录只按 `kernel_name` 一层展开（`lightllm/common/all_kernel_configs/<kernel_name>/`）；`Autotuner`（`lightllm/common/triton_utils/autotuner.py:79`）是后来引入的更完善的版本，额外按 **Triton 编译器版本**分层（`lightllm/common/triton_utils/autotuner.py:203`-`217`）——这条改进本身是必要的：不分版本的话，Triton 升级后旧配置可能因为编译器行为变化而不再是最优解甚至跑出错误结果。新 kernel（如 `moe_sum_reduce`，`lightllm/common/basemodel/triton_kernel/fused_moe/moe_sum_reduce.py:63`）陆续切到新机制，旧 kernel（如 MLA 解码 attention，`lightllm/common/basemodel/triton_kernel/mla_att/decode_att/gqa_flash_decoding_config.py:7`-`8`）还留在老机制上，两套并存反映的是**渐进式迁移**而非一次性重写。
- **不这样会怎样**：如果坚持只留一套，要么放弃"按 Triton 版本分层"这个正确性改进（回退到 `KernelConfigs` 的旧做法，冒着"用不匹配的 Triton 版本调出来的 config 硬跑"的风险），要么一次性把所有仍用 `KernelConfigs` 的 kernel 全部改写并重新在各硬件型号上采集一遍配置数据——对一个开源项目而言，后者的一次性工作量和"贡献者要在真实 GPU 上重新跑一遍 benchmark"这个门槛，都是不小的迁移成本。
- **什么时候可以不这样**：新写的 kernel 应该直接接入 `Autotuner`（按版本分层更安全）；只有在维护存量、不打算大改的旧 kernel（本篇看到的例子是 MLA 解码 attention 这类相对稳定、迭代频率低的模块）上，继续沿用 `KernelConfigs` 才是省事的选择——两套机制的选择本质上是"新代码用新标准、旧代码不为了统一而统一去动它"这条常见工程原则的具体体现。

## 6. 同位对照

**进程拓扑与 IPC 机制**：三个引擎都走"多进程"，但切法和粘合方式都不同。vLLM 是前端进程 → EngineCore 进程 → N 个 `WorkerProc` 进程三层，边界之间统一用 ZMQ + `msgpack`（`vllm:vllm/v1/executor/multiproc_executor.py:111`、`:600`；`vllm:vllm/v1/serial_utils.py:136`）；SGLang 是 `TokenizerManager`（留在 HTTP 主进程里，不是独立 `mp.Process`，`sglang:python/sglang/srt/entrypoints/engine.py:162`）+ 每个 TP/PP rank 一个的 `Scheduler` 进程（`sglang:python/sglang/srt/managers/scheduler.py:5140` 的 `run_scheduler_process`）+ 独立的 `DetokenizerManager` 进程，全程 ZMQ push/pull，源码 docstring 原话"没有共享内存、没有 RPC 调用"（`02-SGLang-Scheduler事件循环.md` `## 0`）。LightLLM 是 HTTP 进程（`hypercorn` 子进程）+ 路由进程 + N 个模型 worker 进程 + 去 token 化进程四层，其中路由↔worker 这一跳**恰恰是 rpyc RPC**（`lightllm/server/router/model_infer/model_rpc.py:187`），和另外两家"拒绝 RPC、只用消息传递"的选择正相反；同时请求体本身常驻共享内存（`lightllm/server/core/objs/req.py:95`），也和另外两家"每次都完整序列化请求"的做法不同——三家在"IPC 到底该长什么样"这道题上给出了三个方向都不一样的答案，没有哪个是显然正确的默认项。

**KV 缓存的分配粒度**：vLLM 默认 `block_size=16`（`vllm:vllm/config/cache.py:79`），SGLang 默认 `page_size=1`（`sglang:python/sglang/srt/arg_groups/overrides.py:2396`，部分 kernel 会强制改成更大值），LightLLM 干脆没有"块/页"这层命名，`KvCacheAllocator`（`lightllm/common/kv_cache_mem_manager/allocator.py:11`）直接就是 token 粒度的指针平移。三者的位置可以理解成一个连续谱系：vLLM 用较大的固定块粒度换分配效率、代价是块内碎片和"部分命中"的复杂度；SGLang 默认贴近 token 粒度但保留了"页"这个可伸缩概念给特定 kernel 用；LightLLM 从命名到实现都彻底不留这层抽象，是这个谱系上最激进的一端。

**算子实现路线**：`_lab/out/repo_stats.json` 给出的三家 CUDA/Triton 文件数——vLLM 169/202，SGLang 246/249，LightLLM 0/129（`00-总览与阅读地图.md` `## 3` 的总表）——直观呈现了这道题的三种答案：vLLM 和 SGLang 都是"CUDA 文件数量和 Triton 文件数量同一个数量级"的混合路线（CUDA 负责极限性能路径，Triton 负责能快速迭代的路径），LightLLM 是唯一把这道选择题的答案定死在"全 Triton"这一端的引擎——`## 5` 决策一已经展开了这条选择背后的收益与代价，此处只做横向定位：**在"要不要为了性能上限保留一条手写 CUDA 路径"这件事上，LightLLM 和另外两家给出的是相反答案**。

## 7. 踩坑与反直觉

- **"360 个文件缺失"不是网络问题，是这套自动调优机制的命名方式本身在 Windows 上不可行。** 缺失清单（`.clone_skipped.txt`）全部落在 `all_kernel_configs/<kernel_name>/` 或 `autotune_kernel_configs/<triton_version>/<device>/<kernel_name>/` 目录下，其中不少 `kernel_name` 自带版本号且用冒号分隔（如 `"moe_sum_reduce:v1"`，`lightllm/common/basemodel/triton_kernel/fused_moe/moe_sum_reduce.py:64`），`os.path.join` 把它直接拼进目录名（`lightllm/common/triton_utils/autotuner.py:213`-`217`）——这在 Linux/macOS 上完全合法，在 Windows NTFS 上冒号是驱动器盘符的保留字符，目录创建/checkout 直接失败。`setup.py:3` 的 `package_data` 还把这两套目录声明为要打进发布包的数据文件，说明这不是临时产物，是这个项目预期会被分发、被用户在自己机器上直接落盘的东西——如果哪天有用户在 Windows 上尝试从源码安装 LightLLM，会撞上和本库 clone 时一样的问题。
- **"0 个 CUDA 文件"不等于"完全不用 CUDA 能力"。** `CudaGraph` 类（`lightllm/common/basemodel/cuda_graph.py:21`）用的是 `torch.cuda.CUDAGraph()`（`:89`、`:125`）——这是 PyTorch 官方 Python API，不是手写的 `.cu`/`.cuh` 源文件，所以不计入"CUDA 文件"这个统计口径，但 CUDA Graph 捕获/回放这个 GPU 特性本身完全在用（对应 CLI 开关 `--disable_cudagraph`、`--graph_max_batch_size` 等，`lightllm/server/api_cli.py:579`-`620`）。"全 Triton"准确的说法是"不写手工 CUDA/C++ 源码"，不是"不接触任何 CUDA 特性"。
- **`_lab/out/struct_map.json` 里 `scheduler` 和 `speculative` 两个子系统显示 0 行，不代表 LightLLM 没有调度器和投机解码。** 调度逻辑真实存在于 `router/manager.py`（`_step`、`loop_for_fwd` 等）和 `router/req_queue/*.py` 里，投机解码对应 `--mtp_mode`/`--mtp_draft_model_dir`/`--mtp_step` 三个 CLI 开关（`lightllm/server/api_cli.py:733`-`756`）——这套关键词+AST 的子系统识别脚本按文件名/类名模式匹配，LightLLM 恰好没有叫 `scheduler.py`、`speculative*.py` 这类"标准命名"的文件，于是被计成了 0。读这类自动化统计产物时，**先确认识别口径再下"不存在"的结论**，这条本身也适用于本库其他篇目的类似数字。
- **"轻量"说的是实现方式，不是接口面积。** README 用"lightweight"形容自己（`README.md:18`），但 `_lab/out/api_surface.json` 的 `summary` 显示 29 条路由（22 个去重路径）、142 个 CLI 开关——CLI 开关数量甚至超过一些"重"引擎的量级。"轻量"准确指向的是"纯 Python + Triton、没有 C++ 扩展编译步骤"这件事本身，不是功能范围小。
- **`StartArgs` dataclass 的默认值从未真正参与运行时取值。** `lightllm/server/api_server.py:36` 先用 `argparse` 的 `parser.parse_args()` 解析命令行（默认值来自 `api_cli.py` 里各个 `field(default=...)`/`add_argument(default=...)`），再用 `StartArgs(**vars(args))`（`lightllm/server/api_server.py:38`）把解析结果**逐字段**灌进 dataclass 构造函数——也就是说只要 `argparse` 那边成功解析出了一个值（哪怕是它自己的 `default`），`StartArgs` 字段声明里写的 `default=...` 就完全不会被用到，它的实际作用只是给类型标注和 IDE 补全提供信息。如果两处默认值出现不一致，不会有任何运行时报错，只会让读代码的人（以为默认值是 dataclass 里写的那个）和实际跑起来的行为（以 `argparse` 侧为准）产生认知偏差。

## 8. 可改进点

以下几条标注为**本库推断**，未提交 issue/PR 核实维护者是否已在计划中：

1. **自动调优缓存目录名不要用冒号做版本分隔符。** `"moe_sum_reduce:v1"` 这类 `kernel_name` 一旦拼进目录路径（`lightllm/common/triton_utils/autotuner.py:213`-`217`），就在 Windows 上不可用。换成 `moe_sum_reduce__v1` 或者把版本号单独作为路径的一层（`.../moe_sum_reduce/v1/...`）而不是拼进字符串，能在不改变功能的前提下解决跨平台 checkout 问题——这条本篇是亲身撞到的，不是理论推测。
2. **`StartArgs` dataclass 与 `api_cli.py` 的 `argparse` 定义应该有单一真源。** 目前两处各自维护 142 个开关的名字、默认值、`choices`，`## 7` 已指出不一致时不会报错、只会造成认知偏差——可以考虑从 `argparse` 的 `ArgumentParser` 对象反射生成 dataclass 字段，或者反过来从 dataclass 的 `field(metadata=...)` 生成 `add_argument` 调用，二选一，但不要两份手工同步的独立声明。
3. **两套自动调优缓存机制（`KernelConfigs` vs `Autotuner`）建议给出显式的迁移路线图或至少一份转换脚本。** 目前从代码里能看到"新 kernel 用新机制、旧 kernel 留在老机制"的既成事实，但没有找到书面的迁移判据或截止时间表（**本库推断**，未查证是否存在未公开的内部计划）。
4. **rpyc 连接的 `allow_pickle=True` 配置（`lightllm/server/router/model_infer/model_rpc.py:187`、`:236`）建议在文档里明确写清楚信任边界。** 目前从代码看这条 Unix socket 默认只在本机可达（`_generate_unix_socket_path` 落在 `/tmp` 下），风险面有限，但如果未来支持跨主机的路由-worker 拆分，这条配置需要被重新评估——本篇不下安全结论，只指出这是一个应该被显式记录的设计决策点，而不是隐含在代码里等人读到。
5. **`_lab/struct_map.py` 的子系统识别口径对 LightLLM 这类命名习惯不同的项目会产出误导性的 0 值**（`scheduler`/`speculative` 两个子系统），这是本库自己的分析工具需要补的口径，不是 LightLLM 代码的问题，但会影响后续读者对"LightLLM 到底有没有调度器"这类问题的第一印象，值得在 `_lab/struct_map.py` 里为 LightLLM 补充命名模式或者干脆换成基于导入关系的识别方式。

## 9. 自测题与延伸阅读

**闭卷自测**（不看正文，能答上来才算过）：

1. HTTP 进程把一个新请求交给路由进程时，ZMQ 消息里装的是完整的 prompt + 采样参数，还是别的什么？完整请求数据实际存放在哪里？
2. 路由进程和模型 worker 进程之间用什么 IPC 机制？这和 vLLM、SGLang 在同一位置的选择有什么本质区别？
3. LightLLM 的 KV 缓存分配单位是什么？把它和 vLLM 默认 `block_size=16`、SGLang 默认 `page_size=1` 放在一起比较，三者在"是否需要一层块/页抽象"这道题上分别怎么选的？
4. 为什么这份仓库 clone 下来会有 360 个文件缺失？这暴露了 LightLLM 自动调优缓存机制的哪个具体设计（提示：目录名里的一个特殊字符）？
5. LightLLM 全仓 CUDA 文件数是 0，这是否意味着它完全不使用任何 CUDA 特性？举出本篇给出的一个反例。
6. `StartArgs` dataclass 里每个字段写的 `default=...` 在真实运行时会不会生效？`api_server.py` 里 `StartArgs` 是怎么被构造出来的？
7. `_lab/out/struct_map.json` 里 LightLLM 的 `scheduler` 子系统显示 0 行代码，这说明 LightLLM 没有调度逻辑吗？如果不是，真正的调度代码在哪个文件里？

**延伸阅读**（双链只取自 `_PLAN.md` §6 名册）：

- [[04-vLLM-KV缓存与前缀缓存]] —— vLLM `block_size=16` 的块粒度设计、`_block_hash`/"部分块"缓存的复杂度来源；本篇 `## 5` 决策四、`## 6` 对照的 KV 粒度谱系另一端。
- [[04-SGLang-内存池与KV布局]] —— SGLang `ReqToTokenPool`/`TokenToKVPool` 两级映射、`page_size` 默认 1 但保留"页"抽象供特定 kernel 强制改写的设计；本篇 `## 6` KV 粒度对照的中间态。
- [[00-总览与阅读地图]] —— 十二个引擎 CUDA/Triton 文件数总表（`## 3`）、"每个'它很好'后面必须跟'什么时候这是错的'"这条红线的出处；本篇 `## 5` 决策一的三段式论证遵循的就是这条红线。
