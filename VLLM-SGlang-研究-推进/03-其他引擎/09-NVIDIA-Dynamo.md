# 09 · NVIDIA Dynamo

> **本篇取证基准**：`dynamo` @ `2a6da14c`（2026-08-22）
> **一句话**：自己不推理，只负责把 vLLM/SGLang/TRT-LLM 指挥成一个集群。

## 0. 结论先行

- **Dynamo 不是本库第 13 个"推理引擎"，是长在其他引擎之上的编排层。** 全仓库 2,220,830 行代码里 CUDA 文件只有 **2 个**（`_lab/out/repo_stats.json` 的 `engines.dynamo.kernels.cuda_files`），而且这 2 个文件都不做数学——`copy_blocks_kernel`（`lib/llm/src/kernels/block_copy.cu:41`-`42`）和 `kvbm_kernels_block_to_universal_kernel`（`lib/kvbm-kernels/cuda/tensor_kernels.cu:151`-`152`）从函数名就能看出来只是内存搬运/布局转换 kernel，不是 attention 或 GEMM。一个"推理引擎"不摸计算，这本身就是它的定位声明。
- **三语言分工不是"Rust 前端 + Python 后端 + Go 运维"这么简单。** Rust（`lib/`，608,542 行）扛的是需要强契约、低延迟、跨进程共享状态的部分：HTTP 服务、KV 前缀索引、PD 分离的路由决策；Python（`components/`，193,827 行产品码）扛的是"和某个具体推理引擎的 Python API 对话"这种天然易变、要跟着上游引擎版本走的胶水层；Go（`deploy/operator/`）扛的是"把一份 K8s CRD 变成一组 Pod"这件事，完全不碰推理逻辑。`## 1` `## 5` 会展开这个边界。
- **KV-aware routing 是 Dynamo 相对"轮询/最少请求数"这类朴素负载均衡的核心增量。** `lib/kv-router/src/indexer/radix_tree.rs` 里的 `RadixTree`（`:49`）用压缩基数树维护"哪个 worker 的哪个 dp_rank 持有哪些 KV block hash"；worker 侧的 `KvEventPublisher`（`lib/llm/src/kv_router/publisher/mod.rs:190`）把 prefix cache 的 store/remove 事件发布出来喂给这棵树；路由决策时 `DefaultWorkerSelector::worker_cost`（`lib/kv-router/src/scheduling/selector/default.rs:333`）把"前缀命中了多少块"和"这个 worker 当前有多忙"合成一个代价函数，选代价最低的 worker。这套机制在 `## 2` `## 3` 逐层拆开。
- **worker 抽象目前是"双轨制"，不是一套统一接口打天下。** 仓库里确实存在一份文档化的统一契约——Python 侧 `LLMEngine` ABC（`components/src/dynamo/common/backend/engine.py:347`）和镜像的 Rust `LLMEngine` trait（`lib/backend-common/src/engine.rs:174`），但**vLLM/SGLang/TRT-LLM 三个正式引擎集成目前都没有直接实现它们**——用的是各自更老的 `BaseWorkerHandler` 类体系（`components/src/dynamo/vllm/handlers.py:1095`）。反而是一个新出现、体量已经接近老 handler（3,921 行 vs 4,220 行）的 Rust "sidecar"（`dynamo-vllm-sidecar`，`lib/sidecar/vllm/src/main.rs:7`）在真正实现 Rust 版 trait。`## 4` `## 7` 会把这条正在发生、尚未收敛的迁移线摊开讲。
- **PD 分离的路由决策在 Rust，KV 字节搬运在引擎自己手里。** `PrefillRouter`（`lib/llm/src/kv_router/prefill_router/mod.rs:182`）是跑在前端进程里的 Rust `Operator`，决定"要不要先打一个 prefill worker、打哪个"；但它只传递 `BootstrapInfo`（host/port/room，`lib/llm/src/protocols/common/preprocessor.rs:94`）这类握手信息和 `PrefillResult.disaggregated_params`（同文件 `:195`、`:199`，源码注释写明"engine-owned；the framework reads this through … without interpretation"），真正的 KV 字节搬运下放给引擎自己的 connector（vLLM 侧 NIXL 拉取式 / Mooncake 推送式，见 `components/src/dynamo/vllm/kv_connector_protocols.py:4`-`11`）。
- **API 表面 20 条默认路由，其中 14 条是 `/v1/*`（`_lab/out/api_surface.json` 的 `openai_compat_paths`），但"默认"两个字要较真。** 这些路径全部来自 Rust 代码里的 `path.unwrap_or(...)` 或 `unwrap_or_else(...)`，例如 `lib/llm/src/http/service/openai.rs:3896` 的 `let path = path.unwrap_or("/v1/chat/completions".to_string());`——正文引用它们时必须说明这是**默认值不是硬编码路径**，`service_v2.rs` 里能查到至少三个环境变量（`DYN_HTTP_SVC_HEALTH_PATH`、`DYN_HTTP_SVC_CHAT_PATH`、`DYN_HTTP_SVC_CMP_PATH`，`lib/llm/src/http/service/service_v2.rs:1030`、`:1034`、`:1036`）会在构建路由树时整体覆盖这些默认值。

## 1. 它在系统里的位置

`_src/dynamo` 是 `https://github.com/ai-dynamo/dynamo.git` 的一次 GitHub zipball 快照，sha 为 `2a6da14c83617f5412caba5777dbf0e2399e9c5f`（`_lab/out/repo_stats.json` 的 `ref` 字段，与 `.clone_meta.json` 一致）。

### 1.1 语言构成：JSON 最多，但那不是"配置很复杂"

`_lab/out/repo_stats.json` 的 `engines.dynamo.languages`（按行数降序）：

| 语言 | 行数 | 文件数 | 在系统里的角色 |
|---|---:|---:|---|
| JSON | 756,357 | 103 | 见 `## 1.2`——绝大部分不是产品数据 |
| Rust | 608,542 | 1186 | 核心运行时：HTTP 服务、KV 路由、PD 编排、发现机制 |
| Python | 414,353 | 1453 | 各引擎的 worker 胶水层、planner、profiler、示例 |
| Go | 143,773 | 360 | Kubernetes operator |
| YAML | 141,891 | 653 | K8s manifest、CI、CRD schema |
| Markdown | 97,956 | 597 | 文档（`docs/fern/`） |
| Shell | 21,072 | 186 | 构建/CI 脚本 |
| TypeScript | 18,023 | 42 | 控制台/前端工具 |
| TOML | 6,173 | 60 | Cargo/项目配置 |
| Protobuf | 5,045 | 11 | gRPC 接口定义 |
| JavaScript | 4,356 | 4 | — |
| CUDA | 1,329 | 2 | 见 `## 0`，KV block 内存搬运 |
| C++ | 1,290 | 6 | 配合 CUDA 的宿主代码 |
| C | 427 | 2 | — |
| C/C++ header | 243 | 4 | — |

四条腿都不小：Rust 是最大的单一语言，Python 次之，Go 单独撑起一整个 K8s operator（14 万行不是一个小项目的体量），JSON 反而排第一——但 `## 1.2` 会说明这不是"配置膨胀"。

### 1.2 全仓库最大的目录不是代码，是合规扫描

`_lab/out/repo_stats.json` 的 `top_dirs`（前 7）：

| 目录 | 文件数 | 行数 | 是什么 |
|---|---:|---:|---|
| `container/compliance` | 61 | 652,930 | 见下文——license/SBOM 工具链，不是产品代码 |
| `lib/llm` | 497 | 359,818 | HTTP 服务、KV 路由客户端、发现机制、协议类型 |
| `components/src` | 861 | 234,088 | Python worker 胶水层（vllm/sglang/trtllm/planner/…） |
| `deploy/operator` | 429 | 212,101 | Go K8s operator |
| `docs/fern` | 341 | 85,800 | 文档站点源 |
| `lib/kv-router` | 146 | 84,117 | KV-aware 路由核心 |
| `lib/runtime` | 165 | 75,570 | 分布式运行时（发现、组件、endpoint） |

`container/compliance` 是全仓库单一最大目录，比排第二的 `lib/llm` 还多出近 30 万行，占全仓库总行数近三成。打开它会发现这不是任何一个子系统的实现，而是**容器镜像的许可证合规工具链**：`container/compliance/README.md:1`-`4` 自己写得很直接——"Inline pipeline that generates per-image license NOTICES at build time, gates the build on a license policy, and ships a base-image SBOM corpus"。真正占体积的是 `base_sboms/` 下的 CycloneDX JSON 快照，比如 `container/compliance/base_sboms/release@16a103b8-amd64.cdx.json` 单个文件就有 118,021 行——这是给每个发布镜像存的第三方软件物料清单（SBOM），用来做许可证漂移检测，和"Dynamo 怎么编排推理"这件事完全无关。**如果只用 `cloc` 之类工具粗看这个仓库的语言构成，JSON 行数第一会把人带偏——JSON 多不是因为配置复杂，是因为合规快照大。**

### 1.3 目录规模已经能看出三层分工的雏形

把 `top_dirs` 按语言对应回 `## 1.1`：`lib/llm`（359,818，Rust）+ `lib/kv-router`（84,117，Rust）+ `lib/runtime`（75,570，Rust）+ `lib/bindings`（51,942 行，PyO3 绑定，`_lab/out/repo_stats.json` `top_dirs` 第 8 位）合计超过 57 万行，构成"核心运行时"；`components/src`（234,088，Python）是"贴合各引擎的 worker 层"；`deploy/operator`（212,101，Go）独立成一个 K8s controller-runtime 项目。三层体量都是"能撑起一个独立项目"的规模，不是谁给谁打下手。

### 1.4 测试/产品比 1.114，12 个引擎里最高

`_lab/out/repo_stats.json` 的 `engines.dynamo.totals`：Python 产品码 193,827 行，测试码 215,985 行，`test_to_src_ratio` = **1.114**——测试代码比产品代码还多，是本库登记的 12 个引擎里最高的（vLLM 0.542、SGLang 未在本篇复核）。**本库推断**：这和 Dynamo 是一个分布式系统而非单进程库有关——正确性依赖的是"HTTP 前端、KV 路由、多个 worker 进程、K8s operator 之间的协议对不对得上"，这类集成/契约测试天然比"函数级单元测试"写得多；具体证据未逐项核对测试分类，标注为推断而非结论。

### 1.5 运行时底座：一套发现机制两种实现，一套事件总线三种传输

`lib/runtime`（Rust，75,570 行）是所有组件共享的地基：`DistributedRuntime`（`lib/runtime/src/distributed.rs:53`）是每个进程（不管是 HTTP 前端、KV 路由器、还是某个 worker）启动时持有的运行时句柄，`register_model`（`lib/bindings/python/rust/lib.rs:483`）这类 PyO3 导出函数最终都是在往这个句柄背后的服务发现系统里写一条记录。

服务发现被抽成了 `trait Discovery`（`lib/runtime/src/discovery/mod.rs:1441`），有两个具体实现：`KVStoreDiscovery`（`lib/runtime/src/discovery/kv_store.rs:66`，走 etcd，这是 CLI 旗标 `--discovery-backend` 默认值 `'etcd'` 对应的路径，见 `_lab/out/api_surface.json` 里 `components/src/dynamo/common/backend/sample_engine.py:141` 附近的 CLI flag 记录）和 `KubeDiscoveryClient`（`lib/runtime/src/discovery/kube.rs:72`，直接读 Kubernetes API）。**这条选择直接呼应 `## 4.3` 的部署形态**：裸机/单机部署没有 K8s API 可用，靠 etcd 做发现；跑在 Go operator 铺开的 K8s 集群里时，可以换成原生 K8s 发现，少运维一个 etcd 集群。事件总线（KV 事件、健康状态、配置变更）则可能走 NATS、ZMQ、TCP 三种传输（`lib/runtime/src/transports/{nats,zmq,tcp}.rs`），`## 3` 讲的 KV 事件线用的是 ZMQ（引擎侧）+ NATS（跨进程转发）的组合，不是单一协议贯穿始终。

## 2. 代码地图（文件 → 职责，带行号）

按"一次请求会经过的层 + 集群铺开的层"排列：

| 层 | 文件:行 | 干什么 |
|---|---|---|
| HTTP 路由装配 | `lib/llm/src/http/service/service_v2.rs:1111` | `HttpService::build`：把下面各条 `*_router()` 拼成一棵 axum 路由树 |
| HTTP 路径覆盖机制 | `lib/llm/src/http/service/service_v2.rs:1030`、`:1034`、`:1036` | `DYN_HTTP_SVC_HEALTH_PATH`/`DYN_HTTP_SVC_CHAT_PATH`/`DYN_HTTP_SVC_CMP_PATH` 环境变量声明——默认路径的真实覆盖入口 |
| HTTP 路由（OpenAI） | `lib/llm/src/http/service/openai.rs:3896` | `chat_completions_router`：`path.unwrap_or("/v1/chat/completions")` |
| HTTP 路由（Anthropic） | `lib/llm/src/http/service/anthropic.rs:72`、`:80` | `DEFAULT_MESSAGES_PATH` 常量 + `anthropic_messages_router` 的 `unwrap_or_else` |
| HTTP 路由（SGLang 原生协议） | `lib/llm/src/http/service/sglang_generate.rs:45` | `DEFAULT_PATH = "/generate"`——Dynamo 前端甚至直接兼容 SGLang 自己的 `/generate` 协议方言 |
| KV 索引核心 | `lib/kv-router/src/indexer/radix_tree.rs:18`、`:49` | `RadixBlock`（单个节点）、`RadixTree`（压缩基数树本体） |
| KV 索引服务 | `lib/kv-router/src/indexer/kv_indexer.rs:239`、`:581` | `KvIndexer` 结构体、`find_matches` 查询接口 |
| KV 事件线协议 | `lib/kv-router/src/protocols.rs:1123`、`:1353` | `KvCacheEventData` 枚举（Stored/Removed/Cleared）、`RouterEvent` 信封结构 |
| KV 事件发布（worker 侧） | `lib/llm/src/kv_router/publisher/mod.rs:190`、`:482` | `KvEventPublisher` 结构体、`publish()` 方法 |
| KV 事件来源配置 | `lib/llm/src/kv_router/publisher/mod.rs:66`、`:80` | `KvEventSourceConfig::Zmq`（"Currently, only ZMQ is supported"）、内部 `KvEventSource::Zmq` 枚举 |
| vLLM 原生 KV 事件线解码器 | `lib/kv-router/src/zmq_wire/mod.rs:4`、`:7` | 模块注释："mirror the Python `msgspec`-defined structures emitted by vLLM engines over ZMQ PUB sockets" |
| SGLang 侧 KV 事件发布 | `components/src/dynamo/sglang/publisher.py:15` | 直接 `from sglang.srt.disaggregation.kv_events import ZmqEventPublisher`——复用 SGLang 自己的原生发布器 |
| TRT-LLM 侧事件流拓扑说明 | `components/src/dynamo/trtllm/publisher.py:17`-`19` | 模块 docstring 原话画出两条路径（见 `## 3`） |
| 运行时底座 | `lib/runtime/src/distributed.rs:53` | `DistributedRuntime` 结构体：所有组件共享的运行时句柄 |
| 服务发现抽象 | `lib/runtime/src/discovery/mod.rs:1441` | `trait Discovery`——`KVStoreDiscovery`（etcd）与 `KubeDiscoveryClient`（原生 K8s）两个实现 |
| 选 worker 的代价函数 | `lib/kv-router/src/scheduling/selector/default.rs:93`、`:333` | `DefaultWorkerSelector` 结构体、`worker_cost`——重叠块数与负载合成一个 logit |
| PD 路由决策 | `lib/llm/src/kv_router/prefill_router/mod.rs:182`、`:278` | `PrefillRouter` 结构体、`Operator::generate`（拦截请求，决定是否走 prefill 跳） |
| PD 握手契约 | `lib/llm/src/protocols/common/preprocessor.rs:94`、`:195` | `BootstrapInfo`（KV 传输连接信息）、`PrefillResult`（engine-owned 的 `disaggregated_params`） |
| worker 统一契约（Python） | `components/src/dynamo/common/backend/engine.py:347`、`:357` | `LLMEngine(BaseEngine)` ABC、抽象方法 `generate` |
| worker 统一契约（Rust） | `lib/backend-common/src/engine.rs:174` | 镜像的 Rust `trait LLMEngine` |
| worker 生产集成（vLLM，老路径） | `components/src/dynamo/vllm/handlers.py:1095`、`:3125` | `BaseWorkerHandler` ABC、`DecodeWorkerHandler` 子类 |
| worker 生产集成（vLLM，新 Rust sidecar） | `lib/sidecar/vllm/src/main.rs:7` | `VllmSidecarEngine::from_args` + `dynamo_backend_common::run`，走 Rust 版 `LLMEngine` trait |
| 模型注册（Rust→Python 绑定） | `lib/bindings/python/rust/lib.rs:483` | `register_model`：PyO3 导出的注册函数，是 `dynamo.llm.register_model` 的真身 |
| K8s CRD 定义 | `deploy/operator/api/v1beta1/dynamographdeployment_types.go:28`、`:71` | `DynamoGraphDeploymentSpec`、`BackendFramework string`（枚举 `sglang;vllm;trtllm`） |
| K8s 主控制器 | `deploy/operator/internal/controller/dynamographdeployment_controller.go:115` | `DynamoGraphDeploymentReconciler.Reconcile` |
| K8s 弹性伸缩 CRD | `deploy/operator/api/v1beta1/dynamographdeploymentscalingadapter_types.go:34` | `Replicas int32`，注释写明由 "HPA/KEDA/Planner" 外部修改 |
| K8s 弹性伸缩控制器 | `deploy/operator/internal/controller/dynamographdeploymentscalingadapter_controller.go:60` | `DynamoGraphDeploymentScalingAdapterReconciler.Reconcile` |

以上 21 条引用只是骨架，`## 3` `## 4` 会摊开其中若干条的上下文。

## 3. 核心数据结构

**KV 事件线（worker → 路由器，单向广播）**

- **`KvCacheEventData`**（`lib/kv-router/src/protocols.rs:1123`）——三态枚举 `Stored`/`Removed`/`Cleared`：`Stored` 携带 `KvCacheStoreData`（`:1135`，含 `parent_hash`、`blocks: Vec<KvCacheStoredBlockData>`），`Removed` 携带 `KvCacheRemoveData`（`:1266`，一串要删除的块哈希）。
- **`KvCacheStoredBlockData`**（`:1252`）——单个 KV block 的最小描述：`block_hash`（`ExternalSequenceBlockHash`，`:1007`，全局唯一）、`tokens_hash`（`LocalBlockHash`，`:1000`，仅由这个块的 token 内容决定，不含祖先信息）。两个哈希分开存的原因：`tokens_hash` 用来算"这段 token 序列理论上应该匹配哪个块"，`block_hash` 用来算"这个块在这棵基数树里的唯一身份"——前缀匹配靠 `tokens_hash` 链，去重靠 `block_hash`。
- **`RouterEvent`**（`:1353`）——信封结构，把 `KvCacheEvent` 包上 `worker_id`、`storage_tier`（存储层级：显存/主机内存/磁盘，多层 KV cache 场景）、`residency_domain`。这是 `KvEventPublisher.publish()`（`lib/llm/src/kv_router/publisher/mod.rs:482`）真正发出去的东西。

**worker 怎么上报自己的 KV 状态：不是 Dynamo 发明的事件格式，是订阅引擎自己已有的事件流。** `KvEventSourceConfig`（`lib/llm/src/kv_router/publisher/mod.rs:66`-`67`）的注释写得很直接——"Currently, only ZMQ is supported"。往前追一层：`lib/kv-router/src/zmq_wire/mod.rs:4`-`8` 的模块注释说明这套解码器"mirror the Python `msgspec`-defined structures emitted by vLLM engines over ZMQ PUB sockets"——**vLLM 本来就有自己的原生 KV 事件 ZMQ PUB 机制**（这不是 Dynamo 独创的能力，vLLM 自己的 KV connector 生态里就有外部消费者会订阅它），Dynamo 只是写了一个专门的解码器去订阅、再转译成 `RouterEvent` 往下游转发。SGLang 这条路径甚至更直接：`components/src/dynamo/sglang/publisher.py:15` 直接 `from sglang.srt.disaggregation.kv_events import ZmqEventPublisher`——**复用的是 SGLang 自己代码库里的类**，不是 Dynamo 自己重新实现一遍。TRT-LLM 侧的模块 docstring 把两种拓扑画得很清楚（`components/src/dynamo/trtllm/publisher.py:17`-`19`，原话）：

  ```
  With Consolidator:    Engine → ZmqKvEventPublisher (ZMQ PUB) → Consolidator → KvEventPublisher (dynamo.llm, ZMQ SUB) → NATS → Router
  Without Consolidator: Engine → KvEventPublisher (NATS PUB) → Router
  ```

  也就是说事件从 worker 传到路由器，中间可能多一跳"Consolidator"（`components/src/dynamo/kv_dc_relay/`，未深挖具体聚合逻辑，标注未查证），用于把同一台机器上多个 data-parallel rank 各自的 ZMQ 流合并成一路，减少路由器要维护的连接数；但无论走哪条拓扑，终点都是同一份 `RouterEvent`/NATS 契约。

**KV 索引本体（路由器侧）**

- **`RadixBlock`**（`lib/kv-router/src/indexer/radix_tree.rs:18`）——树的节点，`children: FxHashMap<LocalBlockHash, SharedRadixBlock>` 按 token 哈希做子节点索引，`state: NodeState` 记录"哪些 worker 在这个前缀位置持有缓存"。
- **`RadixTree`**（`:49`）——单线程压缩基数树，`apply_event`（`:229`）消费 `RouterEvent` 更新树，`find_matches`（`:225`）给定一串 `LocalBlockHash` 序列，返回每个候选 worker 的重叠分数（`OverlapScores`）。"压缩"（compressed）意味着只有分叉点才是真实节点，一整段无分叉的公共前缀不会一格一格建节点——这是它能撑住高频 KV 事件流的关键设计。
- **`KvIndexer`**（`lib/kv-router/src/indexer/kv_indexer.rs:239`）——包在 `RadixTree` 外面的异步服务：单线程 Tokio runtime 处理事件流入和查询请求，避免 `RadixTree` 本身被多线程直接持有导致的锁竞争（源码模块级注释 `lib/kv-router/src/indexer/mod.rs:4`-`7` 明确了这是"KV RadixTree"模块）。

**选 worker 的代价函数**

- **`DefaultWorkerSelector`**（`lib/kv-router/src/scheduling/selector/default.rs:93`，文档注释写"matching the Python `_cost_function`"，说明这是从早期 Python 原型移植过来的算法）——`worker_cost`（`:333`）把两类信号合成一个 logit：`decode_cost_blocks`（当前解码负载）+ `prefill_cost_blocks`（`prefill_load_scale * adjusted_prefill_blocks`）+ `active_request_cost_blocks`，再减去 `overlap_credit_blocks`（由 `effective_overlap_blocks` 也就是 `RadixTree.find_matches` 算出的前缀命中块数换算而来）。**命中的前缀块越多，这个 worker 的"代价"就越低，越容易被选中**——这是"KV-aware"这四个字在代码里的落地位置。
- **`WorkerSelector` trait**（`lib/kv-router/src/scheduling/selector/mod.rs:34`）——把这套打分逻辑做成可插拔接口，`Cargo.toml` 里甚至专门给"自定义路由策略"开了独立的 crate（`examples/router/custom-policy-example/*`，见 `## 2` 之外的 workspace member 列表），说明这层被当作用户可扩展点而不是写死的黑盒。

**Python worker 统一契约**

- **`GenerateRequest`/`GenerateChunk`**（`components/src/dynamo/common/backend/engine.py:35`、`:73` 附近，TypedDict）——`LLMEngine.generate()` 的输入输出契约：`token_ids` 由 Rust 预处理器算好塞进来，`prefill_result`/`bootstrap_info` 两个键专门给 PD 分离场景用，工程注释明确写"set by the frontend's PrefillRouter on decode requests"。
- **`LLMEngine`（Python ABC）**（`:347`）与**`LLMEngine`（Rust trait）**（`lib/backend-common/src/engine.rs:174`）——两份签名几乎一一对应（`from_args`/`start`/`generate`/`abort`/`cleanup`），是同一份契约在两种语言里的镜像实现，`## 7` 会讲清楚它们目前各自的落地状态。

**K8s CRD**

- **`DynamoGraphDeploymentSpec`**（`deploy/operator/api/v1beta1/dynamographdeployment_types.go:28`）——一张图的顶层描述：`Components []DynamoComponentDeploymentSharedSpec`（图里的每个角色）、`BackendFramework string`（`:71`，`+kubebuilder:validation:Enum=sglang;vllm;trtllm`——**CRD 层面就把"可插拔的三个引擎"写进了枚举**）。
- **`DynamoGraphDeploymentScalingAdapterSpec`**（`deploy/operator/api/v1beta1/dynamographdeploymentscalingadapter_types.go:28`）——`Replicas int32`（`:34`）配 `DGDRef`（`:38`，指向某个具体 component），注释原话"modified by external autoscalers (HPA/KEDA/Planner)"——这是弹性伸缩的落地对象：外部信号只改这一个数字，不直接碰 Pod。

## 4. 主流程走读

### 4.1 一次普通请求（无 PD 分离）

1. **HTTP 入口。** 请求打到某个已注册路径（默认路径见 `## 2`，实际路径可能被 `DYN_HTTP_SVC_*_PATH` 覆盖），落到 `lib/llm/src/http/service/service_v2.rs:1111` `HttpService::build()` 拼好的路由树上，具体 handler 见对应 `*_router()` 文件。
2. **找候选 worker。** 请求经过预处理（tokenize，这一步在 Rust 侧完成，`GenerateRequest.token_ids` 因此总是"由 Rust 预处理器设置"）后，`ModelManager` 找出这个模型名对应的一组已注册 worker；如果开启了 KV-aware 路由，接下来不是轮询，而是把 token 序列切成块哈希，交给 `KvIndexer::find_matches`（`lib/kv-router/src/indexer/kv_indexer.rs:581`）查每个 worker 的前缀命中情况。
3. **打分选 worker。** `DefaultWorkerSelector::worker_cost`（`lib/kv-router/src/scheduling/selector/default.rs:333`）把命中块数（降低代价）和当前负载（提高代价）合成一个 logit，选代价最低的 `(worker_id, dp_rank)`。
4. **转发给 worker。** 请求被发给选中的 worker——如果这个 worker 是 vLLM 生产集成，落到 `BaseWorkerHandler` 子类（`components/src/dynamo/vllm/handlers.py:1095` 及其子类）的 `generate()`；如果是新的 Rust sidecar 路径，落到 `dynamo-vllm-sidecar` 里实现 Rust `LLMEngine` trait 的 `generate`。
5. **底层引擎真正生成。** worker 内部调用 vLLM/SGLang/TRT-LLM 各自的原生 Python/gRPC API 跑推理，产出的 token 流经统一的 `GenerateChunk` 或 Rust `LLMEngineOutput` 结构，流回前端。
6. **KV 状态回路。** worker 在生成过程中，通过 `KvEventPublisher.publish()`（`lib/llm/src/kv_router/publisher/mod.rs:482`）把新增/驱逐的 KV block 事件发出去，`KvIndexer` 消费这些事件更新 `RadixTree`——**这条回路和请求处理是完全解耦的异步流**，路由器看到的"缓存状态"永远是上一次事件到达时的快照，不是实时查询 worker 显存。

### 4.2 一次 PD 分离请求

1. `PrefillRouter::generate`（`lib/llm/src/kv_router/prefill_router/mod.rs:278`，实现 `Operator` trait，跑在前端进程内的请求管道里）先接住请求。
2. **条件性绕过判断先行。** `conditional_bypass.rs` 里的逻辑（`resolve_request_decode_pin` 等函数）先看这个请求是否已经有明确的 decode 亲和 worker、以及缓存命中是否已经足够高——如果命中率够高，直接绕过 prefill 跳，走 `## 4.1` 的单跳路径（这是一种"conditional disaggregation"：不是所有请求都无脑走两跳）。
3. **需要 prefill 跳时，选 prefill worker、发起请求。** `select_and_dispatch_prefill`（`lib/llm/src/kv_router/prefill_router/admission.rs:41`）复用同一套 `WorkerSelector` 打分逻辑选一个 prefill worker，把请求发过去。
4. **拿到握手信息，不拿到 KV 字节。** prefill worker 跑完后返回 `PrefillResult`（`lib/llm/src/protocols/common/preprocessor.rs:195`，核心字段 `disaggregated_params: serde_json::Value` 在 `:199`，源码注释"engine-owned; the framework reads this through … without interpretation"）和 `BootstrapInfo`（`:94`，`bootstrap_host`/`bootstrap_port`/`bootstrap_room`）。Dynamo 框架层**只转发这两个结构体，不解释里面的内容**。
5. **decode worker 拿着握手信息自己去搬 KV。** 这两个字段被塞进 `GenerateRequest` 的 `prefill_result`/`bootstrap_info` 键（`components/src/dynamo/common/backend/engine.py:35` 附近的字段注释已写明用途），decode worker 收到后，调用自己引擎原生的 KV connector——vLLM 侧按 `KvConnectorProtocol`（`components/src/dynamo/vllm/kv_connector_protocols.py:23`）分发到具体实现：NIXL 是"拉取式"（decode 读 prefill 给的 block 位置信息去拉），Mooncake 是"推送式"（prefill 主动把块推给一个预分配好的 `transfer_id`）。**真正的 KV 字节从来没有经过 Dynamo 的 Rust 路由层。**

### 4.3 集群怎么铺开：K8s operator 这一层

1. 用户提交一个 `DynamoGraphDeployment` CR，`spec.components` 描述图里每个角色（frontend/prefill/decode/router 等），`spec.backendFramework` 从 `sglang`/`vllm`/`trtllm` 三选一（`deploy/operator/api/v1beta1/dynamographdeployment_types.go:71`）。
2. `DynamoGraphDeploymentReconciler.Reconcile`（`deploy/operator/internal/controller/dynamographdeployment_controller.go:115`）把这份声明式规格渲染成实际 K8s 资源——目录里能看到它走的是 Grove（一种 Gang 调度友好的 PodCliqueSet 抽象，`dgd_grove_workloads_reconciler.go`、`dgd_grove_scaler.go` 等文件）而不是裸 `Deployment`。
3. **弹性伸缩闭环。** Python 侧的 `planner`（`components/src/dynamo/planner/`）观测集群负载后，不直接改 Pod 数量，而是修改 `DynamoGraphDeploymentScalingAdapter.spec.replicas`（`deploy/operator/api/v1beta1/dynamographdeploymentscalingadapter_types.go:34`）；`DynamoGraphDeploymentScalingAdapterReconciler.Reconcile`（`deploy/operator/internal/controller/dynamographdeploymentscalingadapter_controller.go:60`）监听到这个 CR 变化后再去调整真实副本数。**决策（Planner）和执行（operator）之间隔着一层 CRD，而不是 Planner 直接调用 K8s API 改 Deployment**——这条边界本身就是 Go/Python 分工的具体落点。

## 5. 设计决策与代价

### 决策一：编排层与推理引擎解耦，独立于任何单一引擎存在

- **为什么这么设计**：跨机 KV 亲和路由和弹性伸缩这类问题，单个推理引擎进程天然看不到集群全局状态（一个 vLLM 进程只知道自己的显存和自己的请求队列）。把 `RadixTree`（`lib/kv-router/src/indexer/radix_tree.rs:49`）、`DefaultWorkerSelector`（`lib/kv-router/src/scheduling/selector/default.rs:93`）这类跨 worker 的状态和决策做成一个独立于任何引擎的产品，可以让同一套索引/路由/弹性伸缩逻辑同时服务 vLLM/SGLang/TRT-LLM 三个后端（`deploy/operator/api/v1beta1/dynamographdeployment_types.go:71` 的三选一枚举是直接证据），而不是每家各写一遍。
- **不这样会怎样**：`## 6` 的对照证据已经说明后果——vLLM 自己给单进程加了 `/scale_elastic_ep` 端点（`vllm:vllm/entrypoints/serve/elastic_ep/api_router.py:29`），SGLang 自己孵化了一个几乎同构的 `sgl-router`（`sglang:experimental/sgl-router/src/policies/cache_aware_zmq.rs:1`-`9`）——同一个问题被至少三个团队各自独立解决了一遍，代码不共享，成熟度也不一致（SGLang 那份还挂在 `experimental/` 目录下）。
- **什么时候可以不这样**：单机单引擎部署、或者长期只用一种引擎、从不需要跨引擎路由能力时，直接用引擎自带的方案（vLLM 的 DP/`scale_elastic_ep`、SGLang 的 `sgl-router`）能省掉部署一整套 Rust 运行时 + Go operator 的运维负担——这也是"什么时候不该上 Dynamo"这个问题的直接答案。

### 决策二：KV 路由决策放在 Rust 里，不放在 Python

- **为什么这么设计**：查前缀树、算代价函数是每个请求都要走一次的热路径，对延迟敏感；`KvIndexer`（`lib/kv-router/src/indexer/kv_indexer.rs:239`）用单线程 Tokio runtime 处理事件流入和查询请求（模块注释见 `lib/kv-router/src/indexer/mod.rs:1`-`4`），避免多线程直接持有 `RadixTree` 带来的锁竞争，这类"共享可变状态 + 高并发查询"的服务化封装是 Rust 的强项而不是 Python 的强项。
- **不这样会怎样**：`DefaultWorkerSelector` 的文档注释自己写着"Default implementation matching the Python `_cost_function`"（`lib/kv-router/src/scheduling/selector/default.rs:92`）——这句话本身就是证据：这个打分算法最早是 Python 原型，后来被移植到 Rust。**本库推断**：移植的具体动机（性能瓶颈实测、还是纯粹架构决策）未见到变更记录，只看到移植后的结果，标注为推断而非结论。
- **什么时候可以不这样**：请求量很低、或集群只有个位数 worker 时，前缀匹配和打分的计算量本身很小，用 Python 实现路由决策也不会构成瓶颈；`WorkerSelector` trait（`lib/kv-router/src/scheduling/selector/mod.rs:34`）本身被设计成可插拔接口（配套的 `examples/router/custom-policy-example/*` workspace member），说明团队预期有人会换掉默认实现，不是把 Rust 焊死当唯一选项。

### 决策三：PD 路由决策与 KV 搬运解耦——决策在 Dynamo，字节在引擎自己的 connector

- **为什么这么设计**：KV cache 的物理排布（分页大小、显存布局、量化格式）是每个推理引擎的内部细节，vLLM 和 SGLang 的 KV block 在显存里长得不一样。`PrefillResult.disaggregated_params`（`lib/llm/src/protocols/common/preprocessor.rs:199`）的字段注释直接写"engine-owned; the framework reads this through … without interpretation"——把"谁跟谁配对"和"字节怎么搬"拆成两层，前者是通用逻辑（`PrefillRouter`，`lib/llm/src/kv_router/prefill_router/mod.rs:182`），后者留给熟悉自己内存布局的引擎 connector（vLLM 侧 NIXL/Mooncake 双实现，`components/src/dynamo/vllm/kv_connector_protocols.py:4`-`11`）。
- **不这样会怎样**：如果 Dynamo 的 Rust 核心要解析 `disaggregated_params` 里的具体内容，"without interpretation"这行注释就不成立了——意味着 vLLM 每升级一次 KV 格式，Dynamo 核心运行时都要跟着改，这正是三语言分工要避免的"通用核心和某个具体引擎版本强耦合"。
- **什么时候可以不这样**：如果一个部署只服务单一引擎、且愿意深度定制到能保证版本对齐，理论上可以把 KV 传输也收进核心运行时做更细粒度的零拷贝优化，但会牺牲"同一套 Rust 核心同时驱动三个引擎"这条当前架构最看重的通用性。

### 决策四：worker 贴合层保留胶水式 `BaseWorkerHandler`，没有强制迁到统一契约

- **为什么这么设计（本库推断，证据见 `## 2` `## 3` `## 7`）**：vLLM/SGLang/TRT-LLM 三个集成各自都要处理引擎特有能力（LoRA、多模态、投机解码、Structured output 等），`BaseWorkerHandler`（`components/src/dynamo/vllm/handlers.py:1095`）这类贴合类型允许在不动上层协议的前提下按引擎累积特例代码；而统一的 `LLMEngine` ABC（`components/src/dynamo/common/backend/engine.py:347`）是后来才出现的更干净设计，三个生产集成已经有数千行历史代码沉淀在各自 handler 里，重构成本远高于新写一个 demo。
- **不这样会怎样**：如果强制统一，短期内要把 `handlers.py`（4,220 行）和对应的 SGLang/TRT-LLM handler 全部按新契约重写；`common/backend/README.md` 列出的两个已知实现（`TokenspeedLLMEngine`、`SampleLLMEngine`）都是示例级引擎，没有证据表明新契约已经验证过 LoRA、多模态这类 vLLM handler 早就支持的复杂能力——强推可能引入功能回归。
- **什么时候可以不这样**：写一个新引擎集成（不是 vLLM/SGLang/TRT-LLM 这三个已经很重的老集成）时，应该直接实现新的 `LLMEngine` ABC/trait，不必重新发明一套 `BaseWorkerHandler`——`tokenspeed`（`components/src/dynamo/tokenspeed/llm_engine.py:33`）已经是"新集成该走哪条路"的活例子；而新出现的 Rust `dynamo-vllm-sidecar`（`lib/sidecar/vllm/src/main.rs:7`）体量已达 3,921 行，说明即使是"老引擎"，也有团队在用新契约重新实现一条路径。

### 决策五：K8s 编排单独用 Go 写 operator，不放进 Rust 运行时或 Python 组件里

- **为什么这么设计**：K8s 生态的 controller-runtime、client-go、CRD codegen 工具链是 Go 原生的，`deploy/operator/internal/controller/` 下大量 `*_envtest_test.go` 文件（如 `dynamographdeployment_controller_test.go`）证明它在用 K8s 官方推荐的 `envtest` 做集成测试，这条路径复用的是整个 Kubernetes 社区的成熟基础设施。
- **不这样会怎样**：如果用 Rust（`kube-rs`）或 Python（`kopf`）实现 controller，生态成熟度和 CRD codegen 自动化程度都明显弱于 Go 官方工具链；`deploy/operator` 已有 429 个文件、212,101 行规模，且同时维护 `v1alpha1`/`v1beta1` 两套 CRD 版本（`deploy/operator/api/v1alpha1/`、`v1beta1/` 均存在），版本迁移的工程量如果压在非官方语言栈上会更痛苦。
- **什么时候可以不这样**：不打算在 K8s 里跑、只做单机或裸机部署时完全用不到这层，Go operator 对这类用户是纯粹的死代码——这也是为什么 `deploy/operator` 是独立目录、独立 Go module（`go.mod` 与 workspace 根的 `Cargo.toml` 分开管理），不会被拉进 Rust/Python 的构建里。

### 决策六：请求准入用多优先级类的 DRR 排队，不是单一 FIFO 队列

- **为什么这么设计**：`lib/kv-router/src/scheduling/CLAUDE.md:7` 描述的三层结构——`SchedulerQueueActor` 拥有一个 `PolicyQueue`，`PolicyQueue` 下面按 `latency`/`agents`/`batch` 这类 profile 分出多个 `PolicyClassQueue`（`:42`）——用 deficit round robin（DRR，`:41`）让每个类按配置权重轮流拿到派发窗口。集群里的流量天然分好几种 SLA（交互式 chat 要低延迟，agent/batch 工作负载能容忍排队），单一 FIFO 队列没法表达"这一类应该更优先"这种业务语义。
- **不这样会怎样**：单队列 FIFO 下唯一的公平性保证是"先来后到"，想要不同 SLA 分别保证响应时间，只能让客户端自己限流，或者干脆拆成多个物理隔离的集群——后者的代价是本该能共享的 KV 前缀缓存收益（不同 SLA 的请求仍可能命中同一段前缀）也被隔离墙切断了。
- **什么时候可以不这样**：`lib/kv-router/src/scheduling/CLAUDE.md:44` 自己写得很直接——"A single-class profile still uses `PolicyQueue`, but DRR has no cross-class effect"。如果整个集群只服务单一 SLA 的同质流量，多 class 配置退化成单 class，机制还在但不体现收益，这时候直接用更简单的 FIFO 省掉分类/配权重的心智负担是合理的。

## 6. 同位对照：vLLM 与 SGLang 各自的多实例方案

**vLLM：扩缩容是"一个进程管理自己"，不是外挂路由。** vLLM 的 `/scale_elastic_ep` 端点（`vllm:vllm/entrypoints/serve/elastic_ep/api_router.py:29`-`30`，处理函数 `:39`）最终调用的是 `AsyncLLM.scale_elastic_ep`（`vllm:vllm/v1/engine/async_llm.py:1051`）——这条路径的语义是"让这一个 vLLM 引擎实例把自己的数据并行度从 N 改成 M"，操作对象是**单个引擎进程内部的并行拓扑**，不涉及"在多个独立的 vLLM 实例之间路由请求"这件事。换句话说，vLLM 自己解决的是"一个实例怎么弹性伸缩"，不是"多个实例之间怎么按 KV 亲和路由"——后者正是 Dynamo `kv-router` 要解决的问题，vLLM 单进程视角天然看不到。

**SGLang：自己孵化了一个几乎同构的外挂路由器，但还在 `experimental/`。** `_src/sglang` 下有一个独立的 `experimental/sgl-router/` 子项目，带自己的 `sgl-kv-indexer/`（独立的 KV 索引服务）和 `cache_aware_zmq.rs` 选择策略（`sglang:experimental/sgl-router/src/policies/cache_aware_zmq.rs:1`-`9`，模块文档原话"Combines the KV-event-fed `HashTree` with active-load scoring … to pick the worker most likely to already hold the request's prefix"）——这段描述和 Dynamo `DefaultWorkerSelector::worker_cost`（`lib/kv-router/src/scheduling/selector/default.rs:333`）的逻辑几乎是同一件事的两次独立实现：都是"前缀命中分数 + 实时负载"合成一个选择依据。差别在产品化程度：SGLang 的这套路由器挂在 `Policy` trait 之下（`sglang:experimental/sgl-router/src/policies/mod.rs:258`），是 SGLang 仓库里的一个 `experimental` 子目录，随 SGLang 一起发布、跟 SGLang 版本走；Dynamo 的 `kv-router` 是独立仓库、独立版本、同时服务三个引擎。

**核心问题：编排该由引擎自己做还是外挂一层？** 这份快照里三个答案同时活着——vLLM 选"引擎自己扩展一个管理端点"，SGLang 选"引擎自己孵化一个实验性路由 crate"，Dynamo 选"完全独立的产品，倒过来适配三个引擎"。三条路径没有哪个天然更优：引擎自带方案的好处是没有额外网络跳、没有额外要运维的组件、版本天然同步；外挂一层的好处是逻辑只写一次、能跨引擎复用、能做单个引擎看不到的全局决策（比如"这两个 prefill/decode worker 该配对到一起"）。SGLang 把自己的路由器留在 `experimental/` 而不是默认路径，某种程度上就是这个权衡的一个实际投票——值得在 [[04-工程规模与代码结构对比]] 一篇里用更硬的证据继续核实。

## 7. 踩坑与反直觉

- **全仓库最大的目录不是任何一个子系统，是license合规扫描。** `container/compliance`（652,930 行，`_lab/out/repo_stats.json` `top_dirs` 第一位）比排第二的 `lib/llm`（359,818 行）还多出近 30 万行，真身是 CycloneDX SBOM 快照（比如 `container/compliance/base_sboms/release@16a103b8-amd64.cdx.json` 单文件 118,021 行）和许可证扫描脚本，和推理编排毫无关系。如果只用 `cloc` 粗看这个仓库，会把 JSON 行数第一读成"配置很复杂"，实际是"合规快照很大"。
- **两个 CUDA 文件只搬内存，不算数学。** `copy_blocks_kernel`（`lib/llm/src/kernels/block_copy.cu:41`-`42`）和 `kvbm_kernels_block_to_universal_kernel`/`kvbm_kernels_universal_to_block_kernel`（`lib/kvbm-kernels/cuda/tensor_kernels.cu:151`-`152`、`:192`-`193`）从函数名就能确认：这是 KV Block Manager 用来在"分页块"和"连续张量"两种内存布局之间搬运数据的 kernel，不含 attention/GEMM 这类计算——本篇标题"它不是又一个推理引擎"在源码层面的最直接证据就是这两个文件。
- **"统一 worker 契约"目前是文档愿景，不是生产现实。** `common/backend/README.md` 里画的架构图只列出两个 `LLMEngine` 子类：`TokenspeedLLMEngine`（`components/src/dynamo/tokenspeed/llm_engine.py:33`）和 `SampleLLMEngine`，都是示例/压测用途；vLLM/SGLang/TRT-LLM 三个真正扛生产流量的集成，至今用的是各自的 `BaseWorkerHandler`/`HandlerBase` 体系（`components/src/dynamo/vllm/handlers.py:1095`，`components/src/dynamo/trtllm/request_handlers/handler_base.py:260`）。反倒是体量已经堪比老 handler（3,921 行 vs 4,220 行）的 Rust `dynamo-vllm-sidecar`（`lib/sidecar/vllm/src/main.rs:7`-`8`）在真正实现镜像的 Rust `LLMEngine` trait（`lib/backend-common/src/engine.rs:174`）——统一契约正在发生，但发生的位置和文档展示的示例不是同一个。
- **`_lab/out/api_surface.json` 对 Dynamo 的协议类统计是假的 0，不是真的 0。** `n_protocol_classes` 字段显示 dynamo 为 **0**（对照 vLLM/SGLang 都是两位数），但这不是 Dynamo 没有协议类型——`lib/llm/src/protocols/common/preprocessor.rs` 一个文件里就有 `BootstrapInfo`、`PrefillResult`、`MmRoutingInfo` 等多个 `#[derive(Serialize, Deserialize)]` 结构体。真相是 `_lab/api_surface.py` 的协议类提取器只认 Python 的 `pydantic.BaseModel`，对 Rust 的 `serde` 结构体完全没有对应的探测逻辑。同一根源还导致这份 JSON 里 dynamo 的全部 20 条路由 `method` 字段都是 `"?"`（正则抓到了 `.route(&path, ...)` 但没解析出里面的 `post`/`get`）——**读跨引擎协议对比数字时，这类字段为 0 或未知，先怀疑抽取工具看不懂对应语言，别直接当作"这个引擎没有"。**
- **默认路由路径不是硬编码，读代码时容易看漏。** `lib/llm/src/http/service/openai.rs:3896` 那行 `path.unwrap_or("/v1/chat/completions".to_string())` 表面上像是把路径写死了，实际上只是"没传参数时的默认值"；`lib/llm/src/http/service/service_v2.rs:1030`、`:1034`、`:1036` 里的 `DYN_HTTP_SVC_HEALTH_PATH`/`DYN_HTTP_SVC_CHAT_PATH`/`DYN_HTTP_SVC_CMP_PATH` 三个环境变量能在构建路由树时整体覆盖它们。生产环境如果设置了这些变量，`## 2` 的路径表就会和实际部署不符——这是本篇引用"默认路径"时反复强调"这是默认值不是硬编码值"的原因。

## 8. 可改进点

以下几条标注为**本库推断**，未提交 issue 或 PR 核实维护者是否已在计划中：

1. **两份 `LLMEngine` 契约缺一条"新集成该走哪条"的书面判据。** `common/backend/README.md` 的 Quick Start 只演示 `tokenspeed`/`sample` 这类玩具级实现，vLLM/SGLang/TRT-LLM 三个真正的生产集成仍各写各的 handler（`## 7` 已给证据）。一条类似"新引擎集成必须实现 `LLMEngine`，只有历史遗留可以留在 `BaseWorkerHandler`"的显式规则，会比现在"两条路径都在、看资历选"的隐性状态更容易维护，这和本库另一篇观察到的 vLLM 模型新旧布局缺迁移判据是同一类问题。
2. **`_lab/api_surface.py` 对 Rust `serde` 协议类型是盲区。** 给提取器加一条"在 `.rs` 文件里找 `#[derive(..., Serialize, Deserialize, ...)]` 附近的 `struct`/`enum` 定义"的规则，就能补上 `n_protocol_classes` 恒为 0 这个盲区，避免后续跨引擎对比篇目（比如 [[04-工程规模与代码结构对比]]）把 Dynamo 的协议表面错误地统计成"不存在"。
3. **20 条路由的 HTTP method 全部是 `?`。** 这个信息其实唾手可得——`.route(&path, post(handler))` 里的 `post`/`get` 就是函数名本身，正则只是没把它提出来。修起来成本不高，但目前任何依赖这份 JSON 做"这条路由是不是幂等"之类判断的下游分析都拿不到这个字段。
4. **弹性伸缩链路的决策延迟不透明。** Planner 改 `DynamoGraphDeploymentScalingAdapter.spec.replicas` 到 operator 真正调整副本数之间隔着一层 CR watch，本篇没有在 `components/src/dynamo/planner/` 下找到明确的轮询周期常量（未继续深挖，标注未查证）。如果这层延迟没有直接暴露成一个可观测指标，运维排查"扩容为什么慢了"时会缺一个直接入口。
5. **KV 事件的"ZMQ 还是 Push"两条路径缺一张集中的决策表。** `## 3` 已经证明 vLLM/SGLang 走的是订阅引擎原生 ZMQ 事件（`KvEventSourceConfig::Zmq`，`lib/llm/src/kv_router/publisher/mod.rs:66`），TRT-LLM 侧的 docstring（`components/src/dynamo/trtllm/publisher.py:17`-`19`）额外画出了"带 Consolidator"和"不带 Consolidator"两种拓扑——这些信息目前分散在三个引擎各自的 `publisher.py` 文件里，靠读代码才能拼出全貌。一张放在 `lib/kv-router/README.md` 或类似位置的"各引擎 KV 事件拓扑对照表"，会让新读者不用把三份 Python 文件都翻一遍才能确认某个引擎走的是哪条路径。

## 9. 自测题与延伸阅读

**闭卷自测**（不看正文，能答上来才算过）：

1. Dynamo 仓库里 CUDA 文件只有 2 个，它们分别叫什么名字、做什么事？这两个文件名支撑了本篇"Dynamo 不是推理引擎"这个论断的哪一部分证据？
2. 路由器怎么知道"哪个 worker 持有某段前缀的 KV cache"？worker 侧靠哪个结构把这个信息发布出去（提示：`publish()` 方法），路由侧靠哪个数据结构存、哪个方法查？
3. `DefaultWorkerSelector::worker_cost` 把哪几类信号合成一个 logit？"这个 worker 命中的前缀块数更多"会让它的代价变高还是变低？
4. PD 分离请求里，`BootstrapInfo` 和 `PrefillResult` 分别携带什么？Dynamo 的 Rust 核心会不会解析 `disaggregated_params` 字段里的具体内容？为什么源码注释要强调这一点？
5. vLLM/SGLang/TRT-LLM 三个正式引擎集成，目前有没有直接实现 `components/src/dynamo/common/backend/engine.py` 里定义的 `LLMEngine` ABC？现在真正在用镜像的 Rust `LLMEngine` trait 的是谁？
6. `deploy/operator/api/v1beta1/dynamographdeployment_types.go` 里 `BackendFramework` 字段的合法取值有哪三个？这个枚举本身证明了本篇哪个核心论点？
7. 全仓库里文件数/行数最大的单一目录是哪一个？它是 Dynamo 的产品代码吗？如果不是，它实际是什么、为什么会这么大？
8. worker 侧上报 KV 状态用的事件格式是 Dynamo 自己发明的，还是复用了某个推理引擎已有的机制？vLLM 和 SGLang 这条路径分别长什么样？
9. 服务发现有哪两种实现？分别适合什么部署形态，和 `## 4.3` 的 K8s operator 铺开方式是什么关系？

**延伸阅读**（双链只取自 `_PLAN.md` §6 名册）：

- [[08-Mooncake传输引擎]] —— 本篇 `## 4.2` 提到 decode worker 通过 KV connector 拉取 prefill 端数据，Mooncake 就是这类"推送式"KV 搬运的具体实现之一；那篇讲清楚了"KV 字节怎么搬"这件被 Dynamo 刻意甩给引擎自己的事，是本篇 PD 分离一节的下游延伸。
- [[12-vLLM-PD分离与KV-Connector]] —— vLLM 自己的 `KvConnectorProtocol`（`components/src/dynamo/vllm/kv_connector_protocols.py:23`）分发到的 NIXL/Mooncake 两种连接器，在 vLLM 原生（不经 Dynamo）的 PD 分离路径里长什么样，是本篇 `## 4.2` 的对照篇。
- [[11-开源推理引擎谱系图]] —— 把 Dynamo 放回全部 12 个引擎的谱系里看它的位置：本篇反复强调"它不是引擎"，这条结论在谱系图里应该体现为一个独立的分支而不是和 vLLM/SGLang 并列的节点。
- [[04-工程规模与代码结构对比]] —— 本篇 `## 1` `## 7` 用到的语言构成、目录规模、测试/产品比等数字，是这篇跨引擎对比的原始素材之一；`container/compliance` 这类"合规目录污染规模统计"的现象值得在那篇里系统化核实是否只有 Dynamo 一家。
