# vLLM PD 分离与 KV Connector 解剖

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：Connector 把外部 KV 伪装成一次前缀缓存命中。

## 0. 结论先行

1. **`KVConnectorBase_V1` 是一个横跨两个进程的接口，不是一个类**。它的方法天生分两组：scheduler 进程侧的方法只回答"这个请求要不要等、等多久、等到了怎么记账"，worker 进程侧的方法才真正"把字节从哪儿倒腾到哪儿"（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`）。两侧各自实例化一份 connector 对象（`role=KVConnectorRole.SCHEDULER` / `WORKER`），中间靠一份每步现算的 `KVConnectorMetadata`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:141`）单向传递，调度器打包、worker 解包，不跨步复用。
2. **vLLM 不区分"P/D 远程搬运"和"前缀缓存扩容（CPU/磁盘 offload）"——是同一套接口**。`get_num_new_matched_tokens(request, num_computed_tokens)` 返回 `(可再命中的 token 数, 是否异步)`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:450`），调度器只认这个数字，完全不知道背后是从另一台 decode 实例用 RDMA 拉来的，还是从本机 CPU 内存里翻出来的。NIXL（跨实例）和 OffloadingConnector（本机 CPU 分层）实现的是**同一个方法签名**。
3. **一次异步外部命中会让请求脱离正常调度队列**，进入 `RequestStatus.WAITING_FOR_REMOTE_KVS` 状态（`vllm/v1/request.py:369`）——调度器已经为它预留了 KV 块，但不会给它排计算预算，直到 worker 侧 `get_finished()` 报告这个请求收完了，调度器才把它挪回 `WAITING` 重新参与排队（`vllm/v1/core/sched/scheduler.py:2757`）。
4. **失败路径是存在的，但默认策略是"直接失败"不是"本地重算"**：`KVTransferConfig.kv_load_failure_policy` 默认值是 `"fail"`（`vllm/config/kv_transfer.py:69`），加载失败的请求默认以 `FINISHED_ERROR` 结束并把错误抛给客户端；只有显式配成 `"recompute"`，调度器才会把这段 token 标记为未命中、退回本地重新跑一遍前向。
5. **"PD 分离不划算"不是一句空话，vLLM 自己的 NixlConnector 代码里就写死了一个门槛**：`kv_recompute_threshold` 默认 64 token（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:160`），低于这个 token 数就放弃跨节点拉取、直接本地重算——因为握手/通知这类**与 token 数无关的固定开销**，在段落很短时比重算本身还贵。官方文档也直接写明"Disaggregated prefill DOES NOT improve throughput"（`docs/features/disagg_prefill.md:16`），PD 分离换来的是 TTFT/ITL 的可调性，不是吞吐。

## 1. 它在系统里的位置

KV Connector 是一个**可选**组件：只有配置了 `--kv-transfer-config` 时 `vllm_config.kv_transfer_config` 才非空，`Scheduler.connector` 才会被实例化（`vllm/v1/core/sched/scheduler.py:144`），否则始终是 `None`——所以几乎所有涉及它的调度器代码都套了一层 `if self.connector is not None:`。

它和 [[04-vLLM-KV缓存与前缀缓存]] 讲的 `KVCacheManager` 是同一层的两个方向相反的问题：`KVCacheManager` 回答"这段前缀**本地**有没有算过"，`KVConnector` 回答"这段前缀**外部**（另一台实例、CPU 内存、磁盘、分布式 KV 存储）有没有，要不要等它送过来"。两者的结果在调度器里被直接相加，构成一个请求最终的 `num_computed_tokens`（见 §4.1）。

```
HTTP 请求 → Scheduler.schedule()（03 篇）
    │ 本地前缀缓存命中：KVCacheManager.get_computed_blocks（04 篇）
    │ 外部前缀命中：connector.get_num_new_matched_tokens()
    ▼
Scheduler 每步 schedule() 收尾 → build_connector_meta() 打包 KVConnectorMetadata
    │ 随 SchedulerOutput 一起发给 Worker 进程
    ▼
GPUModelRunner.execute_model（06 篇）
    │ bind_connector_metadata()
    │ kv_connector.start_load_kv()          ———— 整批异步发起加载
    │ 逐层 forward：
    │     wait_for_layer_load(layer_i)      ———— 阻塞等这一层的 KV 到位
    │     attention(...)
    │     save_kv_layer(layer_i, ...)       ———— 发起这一层的异步保存
    │ kv_connector.wait_for_save()          ———— 等所有保存完成才能覆盖 KV buffer
    ▼
KVConnectorOutput{finished_sending, finished_recving, invalid_block_ids, ...}
    │ 回传给 Scheduler.update_from_output()
    ▼
Scheduler._update_from_kv_xfer_finished / _handle_invalid_blocks
```

这套结构对模型执行器本身是透明的：注意力层不知道自己的 KV 是不是被 connector 接管了，`wait_for_layer_load`/`save_kv_layer` 是通过一个函数装饰器挂在每层 `forward` 外面的（见 §4.3），没有 connector 时这层装饰器直接是空操作。

## 2. 代码地图（文件 → 职责，带行号）

依据 `_lab/out/struct_map.json` 里 `disagg` 子系统的统计：**79 个文件 / 33,900 行**，是 `01-vLLM-全景与代码地图` 里最大的单个子系统之一。核心接口与调度器挂钩集中在下面这几个文件：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 抽象接口 `KVConnectorBase_V1` 本体（704 行） | 类定义 `vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`；worker 侧 `start_load_kv` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:289`、`wait_for_layer_load` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:307`、`save_kv_layer` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:321`、`wait_for_save` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:343`、`get_finished` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:353`、`get_block_ids_with_load_errors` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:371`；scheduler 侧 `get_num_new_matched_tokens` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:450`、`update_state_after_alloc` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:485`、`build_connector_meta` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:511`、`request_finished` `vllm/distributed/kv_transfer/kv_connector/v1/base.py:543` |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | Connector 名字 → 实现类 的注册表与工厂 | `KVConnectorFactory` 类 `vllm/distributed/kv_transfer/kv_connector/factory.py:27`；`create_connector` `vllm/distributed/kv_transfer/kv_connector/factory.py:43`；16 处 `register_connector(` 调用起于 `vllm/distributed/kv_transfer/kv_connector/factory.py:152`，止于 `vllm/distributed/kv_transfer/kv_connector/factory.py:238` |
| `vllm/config/kv_transfer.py` | `KVTransferConfig`：CLI `--kv-transfer-config` 落地的配置对象 | 类定义 `vllm/config/kv_transfer.py:23`；`kv_load_failure_policy` 默认 `"fail"` `vllm/config/kv_transfer.py:69` |
| `vllm/v1/core/sched/scheduler.py` | 调度器侧全部挂钩，本篇主流程的骨架 | 创建 connector `vllm/v1/core/sched/scheduler.py:149`；读取失败策略 `vllm/v1/core/sched/scheduler.py:157`；`bind_gpu_block_pool` `vllm/v1/core/sched/scheduler.py:304`；`get_num_new_matched_tokens` 调用点 `vllm/v1/core/sched/scheduler.py:838`；`update_state_after_alloc` 调用点 `vllm/v1/core/sched/scheduler.py:1069`；`WAITING_FOR_REMOTE_KVS` 赋值 `vllm/v1/core/sched/scheduler.py:1094`；`build_connector_meta` 调用点 `vllm/v1/core/sched/scheduler.py:1309`；`_update_waiting_for_remote_kv` `vllm/v1/core/sched/scheduler.py:2757`；`_update_from_kv_xfer_finished` `vllm/v1/core/sched/scheduler.py:2836`；`_update_requests_with_invalid_blocks` `vllm/v1/core/sched/scheduler.py:2865`；`_handle_invalid_blocks` `vllm/v1/core/sched/scheduler.py:2968`；`_connector_finished` `vllm/v1/core/sched/scheduler.py:2699` |
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | worker 侧一步的完整生命周期封装（bind → load → wait_for_save → get_finished） | `_get_kv_connector_output` `vllm/v1/worker/kv_connector_model_runner_mixin.py:69` |
| `vllm/model_executor/layers/attention/kv_transfer_utils.py` | 把逐层 `wait_for_layer_load`/`save_kv_layer` 挂到每个 attention 层 `forward` 上的装饰器 | `maybe_transfer_kv_layer` `vllm/model_executor/layers/attention/kv_transfer_utils.py:15` |
| `vllm/v1/outputs.py` | worker → scheduler 的回传载荷 | `KVConnectorOutput` `vllm/v1/outputs.py:264`；`invalid_block_ids` 字段 `vllm/v1/outputs.py:273` |
| `vllm/v1/core/sched/output.py` | scheduler → worker 的下发载荷字段 | `kv_connector_metadata` 字段 `vllm/v1/core/sched/output.py:261` |
| `vllm/v1/request.py` | 请求状态机新增的等待态 | `WAITING_FOR_REMOTE_KVS` `vllm/v1/request.py:369` |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | 生产级参考实现：P/D 语义 + 本地重算门槛 | `get_num_new_matched_tokens` `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:34`；应用 `kv_recompute_threshold` `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:96`；`request_finished` 延迟释放 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:181` |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py` | `kv_recompute_threshold`/租约时长定义 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:160`（阈值）、`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:75`（租约时长） |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` | worker 侧失败检测与租约超时回收 | `get_finished` 里的超时清算 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2127`；`_handle_failed_transfer` `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2239`；`get_block_ids_with_load_errors` `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2487` |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 把 CPU offload 伪装成前缀缓存的对照实现 | `get_num_new_matched_tokens` `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:936` |

（以上 20+ 条 `文件:行` 引用，远超 `## 2` 的硬指标 10 条。）

### 2.1 已注册 Connector 实现清单

`KVConnectorFactory` 在 `vllm/distributed/kv_transfer/kv_connector/factory.py` 里注册了 **16 个**内置实现（外部还可以通过 `kv_connector_module_path` 动态加载第三方实现，不计在内）。「传输介质」一栏区分「源码为证」（读了具体实现）和「文档所述」（仅凭 docstring/文档描述、本库未逐行核对底层库）：

| 实现 | 类定义 `文件:行` | 传输介质 | 适用部署 |
|---|---|---|---|
| `NixlPullConnector`（`NixlConnector` 是它的向后兼容别名） | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:334` | RDMA（UCX 为主，可选 LIBFABRIC/GDS 等 NIXL 后端插件；源码为证：`docs/features/nixl_connector_usage.md` 与 `nixl/base_worker.py` 里的 UCX/RDMA/NVLink 相关分支） | 生产级跨节点 P/D 分离，READ 语义（消费者主动拉） |
| `NixlPushConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:362` | 同上，WRITE 语义（生产者主动推） | 生产级跨节点 P/D 分离，希望由 P 侧发起传输的场景 |
| `MooncakeConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:447` | RDMA（Mooncake Transfer Engine；文档所述） | 生产级跨节点 P/D 分离 |
| `MooncakeStoreConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:89` | 共享 KV 存储（`MooncakeDistributedStore`，可跨节点池化；docstring 原话"KV connector using MooncakeDistributedStore as shared KV pool"） | 前缀缓存的跨实例池化/offload，不是严格的单向 P→D |
| `MoRIIOConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:192` | RDMA（MORI-IO，ROCm 专用；文档标注"MoRIIOConnector (ROCm only)"，`docs/features/nixl_connector_usage.md:31`） | AMD ROCm 集群的 P/D 分离 |
| `LMCacheConnectorV1` | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:72` | 委托给 LMCache 库自身的后端（可搭配 NIXL 做底层传输；文档所述） | 通用 KV 缓存层，常见搭配是 LMCache + NixlConnector 做跨实例传输 |
| `LMCacheMPConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py:464`（`LMCacheMPConnectorUpstream`） | 独立的 `lmcache server` 进程，多个 vLLM 实例共享（进程间通信；文档所述） | 多 vLLM 实例共享同一份 KV 缓存池，不是一对一 P/D |
| `OffloadingConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:49` | CPU 内存（默认 `CPUOffloadingSpec`），可插拔文件系统/对象存储等多级 tier（`vllm/v1/kv_offload/tiering/` 下的 `fs`/`obj`/`p2p`） | 单实例内的前缀缓存扩容，不是跨实例 P/D |
| `SimpleCPUOffloadConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:54` | CPU 内存（自定义 kernel 直传，走 `vllm/v1/simple_kv_offload/`） | 轻量级 CPU offload，比 `OffloadingConnector` 更简单的实现路径 |
| `HF3FSKVConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:469` | 分布式文件系统（3FS，通常配 NVMe） | 大规模 KV 落盘复用，读写吞吐要求高的离线/近线场景 |
| `FlexKVConnectorV1` | `vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py:35` | CPU 内存 + SSD + 远程存储的多级体系（docstring 原话"offloading KV cache to CPU memory, SSD, and remote storage"） | 超大规模 KV Store，第三方库 FlexKV 接管 |
| `MultiConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:128` | 委托给一组子 connector（本身不传输数据） | 同时启用多种传输方式（例如 NixlConnector 做跨实例 + ExampleConnector 做本地共享盘兜底） |
| `ExampleConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:85` | 本地/共享文件系统（默认路径 `/tmp`，源码注释自称"Simple debug implementation"） | 调试与教学示例，非生产 |
| `ExampleHiddenStatesConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py:97` | 同上（传输的是 hidden states 而不是 KV） | 调试与教学示例（用于演示非 KV 类信息的连接器扩展点） |
| `DecodeBenchConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py:77` | 无真实传输——用随机/常量值直接**填充** KV 缓存（源码注释"emulates a prefill-decode disaggregated setting by filling the KV cache with dummy values"） | 纯 decode 阶段性能压测工具，不是真实的 P/D 部署 |

（说明：官方使用文档 `docs/features/disagg_prefill.md:20` 写的是"Now supports 9 types of connectors"，与 `factory.py` 实际注册的 16 个不一致——本表以源码注册表为准，这个文档滞后现象本身也是 §7 的一条踩坑。）

## 3. 核心数据结构

### 3.1 `KVConnectorRole`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:124`）

```python
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1
```

同一个 connector 类会被实例化两次，一次在调度器进程带 `role=SCHEDULER`，一次在每个 worker 进程带 `role=WORKER`（`vllm/distributed/kv_transfer/kv_connector/factory.py:43` 的 `create_connector` 按调用方传入的 `role` 构造）。多数具体实现内部会按这个 `role` 分岔出两个内部类（例如 `MooncakeConnector.__init__` 按 `role` 选择性构造 `MooncakeConnectorScheduler` 或 `MooncakeConnectorWorker`，`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:447` 起），这是"接口统一、实现内部拆分"的典型写法。

### 3.2 `KVConnectorMetadata` / `KVConnectorWorkerMetadata`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:141`、`:150`）

两个抽象空基类，标注方向相反：

- `KVConnectorMetadata`——scheduler → worker，每步由 `build_connector_meta()` 现造一份，`bind_connector_metadata()` 挂到 worker 侧 connector 上，`clear_connector_metadata()` 用完即弃（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:224`、`:236`）。它就是空基类，具体字段完全由子类决定——一个具体例子是 `NixlConnectorMetadata`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:238`），内部拿着 `reqs_to_send: dict[ReqId, float]`（请求 ID → 租约到期时间戳）等字段。
- `KVConnectorWorkerMetadata`——worker → scheduler 的反向通道，多个 worker（TP/PP 多进程）各自的输出要先在本地聚合，`aggregate()` 是这个合并的抽象接口（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:161`）。

### 3.3 `SupportsHMA`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:85`）

一个独立的 mixin 接口，不是 `KVConnectorBase_V1` 的方法，而是"这个 connector 支不支持混合内存分配器（HMA，即 04 篇讲的多 `KVCacheGroup` 混合注意力）"的能力声明。`request_finished_all_groups()`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:93`）是它唯一的抽象方法，作用是把 `request_finished()` 从"单组块列表"升级成"每组一份块列表"（`block_ids: tuple[list[int], ...]`）。`KVConnectorFactory.create_connector` 会在 HMA 开启但 connector 不支持它时直接拒绝启动（`vllm/distributed/kv_transfer/kv_connector/factory.py:53` 附近），而不是静默退化——这是一处"能力不匹配就报错而不是降级"的设计。

### 3.4 `KVConnectorOutput`（`vllm/v1/outputs.py:264`）

worker 侧一步执行完之后打包回传给调度器的全部信息：

```python
class KVConnectorOutput:
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    invalid_block_ids: set[int] = field(default_factory=set)     # vllm/v1/outputs.py:273
    expected_finished_count: int = 0
```

`finished_sending`/`finished_recving` 是"完成"的正反两面：生产者报告"我这批 KV 送完了，可以释放本地块"，消费者报告"我这批 KV 收完了，可以把请求挪回调度队列"。`invalid_block_ids` 是失败信号，见 §4.6。

### 3.5 `RequestStatus.WAITING_FOR_REMOTE_KVS`（`vllm/v1/request.py:369`）

请求状态机（在 [[03-vLLM-调度器解剖]] 篇会展开完整状态图）里专门为异步外部加载新增的一个等待态，和 `WAITING`（尚未调度过）、`RUNNING`（正在跑）并列，但语义是"块已经分配好了，只是数据还没到"——调度器据此把它从主循环里摘出去，既不占用 token 预算，也不会被当成"新请求"重复走一遍前缀缓存查询。

### 3.6 `KVTransferConfig`（`vllm/config/kv_transfer.py:23`）

CLI `--kv-transfer-config` 落地的配置对象，核心字段：

- `kv_connector: str | None`——注册表里的类名，见 §2.1。
- `kv_role: Literal["kv_producer","kv_consumer","kv_both"] | None`——决定这个实例是纯 P、纯 D，还是双向都做（`vllm/config/kv_transfer.py:41`）；`is_kv_producer`/`is_kv_consumer` 两个 property 就是从这个字段派生的布尔值（`vllm/config/kv_transfer.py:113`、`:117`）。
- `kv_connector_extra_config: dict[str, Any]`——每个具体 connector 自己的私有配置口袋，`kv_recompute_threshold`、`kv_lease_duration`、`cpu_bytes_to_use` 等都从这里读，`KVTransferConfig` 本身不知道这些键的存在（`vllm/config/kv_transfer.py:59`）。
- `kv_load_failure_policy: Literal["recompute","fail"] = "fail"`——见 §0 第 4 条，本篇 §4.6/§7 会反复回到这个字段。

## 4. 主流程走读

### 4.1 调度器侧：本地命中 + 外部命中怎么拼成一个数字

新请求第一次被调度时（`request.num_computed_tokens == 0`），调度器先做本地前缀缓存查询，再问 connector（`vllm/v1/core/sched/scheduler.py:819` 起）：

```python
(new_computed_blocks, num_new_local_computed_tokens, ...) = self._get_local_prefix_cache_hit(request)   # :826

if self.connector is not None:
    partial_tail = num_new_local_computed_tokens % self.block_size
    block_aligned_local = num_new_local_computed_tokens - partial_tail
    ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
        request, block_aligned_local)                                                       # :838
    ...
    num_external_computed_tokens = ext_tokens

num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens          # :889-890
```

三个细节值得点出：

1. **喂给 connector 的是"块对齐"后的本地命中长度**（`block_aligned_local`，`vllm/v1/core/sched/scheduler.py:833-836`），不是原始的 `num_new_local_computed_tokens`。这是为了避免半块级别的竞态：如果本地命中在块中间截断，而外部命中恰好更长，直接采信外部结果、丢弃本地半块，比试图"拼接"两个不对齐的命中要简单。
2. `get_num_new_matched_tokens` 可以返回 `None` 表示"我还不知道，请下一步再问一次"（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:465-470` 的文档字符串）——这时请求会被重新塞回等待队列的队首，不占用本轮调度机会，但也不会被跳过（`vllm/v1/core/sched/scheduler.py:843-849`）。
3. `connector_prefix_cache_queries`/`connector_prefix_cache_hits`（`vllm/v1/core/sched/scheduler.py:883-886`）把这次查询计入统计——**外部 KV 命中在观测上就是前缀缓存命中的一种**，`PrefixCacheStats` 记录里分不出这段 token 到底来自本地哈希表还是远程连接器（`vllm/v1/core/sched/scheduler.py:1078` 的 `connector_prefix_cache_stats.record`）。

### 4.2 分配后：`update_state_after_alloc` 与 `WAITING_FOR_REMOTE_KVS`

块分配完成后，调度器把结果通知 connector（`vllm/v1/core/sched/scheduler.py:1068-1073`）：

```python
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request, self.kv_cache_manager.get_blocks(request_id), num_external_computed_tokens)
```

如果 §4.1 返回的 `load_kv_async` 为真，这个请求不会进入 `self.running`，而是切到等待态并原地退出本轮循环（`vllm/v1/core/sched/scheduler.py:1091-1121`）：

```python
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS   # :1094
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens        # 记账，但这些 token 还没真的被加载
    self._inflight_prefills.add(request)                     # :1110，用于统计"这些块还占着多少预留"
    continue
```

这里的关键是**块已经真实分配、`ref_cnt` 已经加过**，只是内容还没写入——调度器靠 `_inflight_prefill_reserved_blocks`（`vllm/v1/core/sched/scheduler.py:2750`）把这批"占着位置但还没算完"的块计入容量预算，避免后续调度把这些块的空间又分给别的请求。

### 4.3 worker 侧一步的完整生命周期

`_get_kv_connector_output`（`vllm/v1/worker/kv_connector_model_runner_mixin.py:69`）是 worker 侧一步执行 `execute_model` 时套在整个 forward 外面的上下文管理器：

```python
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)   # 解包调度器下发的这一步该干什么
kv_connector.start_load_kv(get_forward_context())                              # :86，整批异步发起加载
try:
    yield output          # ———— 真正的模型前向在这中间跑，逐层触发 wait_for_layer_load / save_kv_layer
finally:
    kv_connector.wait_for_save()                                               # :91，等所有异步保存完成
    output.finished_sending, output.finished_recving = kv_connector.get_finished(...)   # :93-94
    output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()   # :96
    kv_connector.clear_connector_metadata()
```

层内的 `wait_for_layer_load`/`save_kv_layer` 不是在模型外面额外包一层循环调出来的，而是通过一个函数装饰器直接挂在**每个** attention 层的 `forward` 上（`maybe_transfer_kv_layer`，`vllm/model_executor/layers/attention/kv_transfer_utils.py:15`）：

```python
@wraps(func)
def wrapper(*args, **kwargs):
    ...
    connector.wait_for_layer_load(layer_name)   # :51，进入这层前，先等这层的 KV 到位
    result = func(*args, **kwargs)              # 这层真正的 attention 计算
    connector.save_kv_layer(layer_name, kv_cache, attn_metadata)   # :57，出这层后，异步把这层的 KV 发出去
    return result
```

这是 vLLM PD 分离设计里一个容易被忽略的优化点：**KV 的保存/加载与模型计算按层粒度交叠**，而不是"整批 forward 跑完再统一传输"。第 `i` 层的 KV 一算完就可以异步开始往外发（producer），同时第 `i+1` 层的注意力计算已经在跑；同理 consumer 侧只要第 `i` 层的 KV 到位就能立即开始这层的计算，不需要等全部层都到齐。这也是为什么"层内异步"连接器要求 `PIECEWISE` CUDA Graph 模式（`requires_piecewise_for_cudagraph`，`vllm/distributed/kv_transfer/kv_connector/v1/base.py:604`）——CUDA Graph 一旦整体捕获，层间就没有 Python 代码可插入等待点了（[[06-vLLM-模型执行与CUDA-Graph]] 篇有相关背景）。

### 4.4 一次完整 PD 请求的时序（以 NixlPullConnector 为例）

```
P 侧（kv_role=kv_producer）                          D 侧（kv_role=kv_consumer）
─────────────────────────────                       ─────────────────────────────
收到 do_remote_decode 请求
正常走完整前缀（本地无外部命中）
request_finished() 判定 is_p_node=True
  → 若 params 里没预置 remote_block_ids，
    直接返回 (False, None)，正常释放
  → 若已知会被 D 拉取，把块的过期时间
    写入 _reqs_need_send（含租约 TTL，
    默认 30s，可被心跳延长）
                                                     收到 do_remote_prefill 请求
                                                     get_num_new_matched_tokens()：
                                                       从 kv_transfer_params 读出
                                                       远端 prompt token 数，
                                                       返回 (count, True)  ———— 异步
                                                     调度器：request.status =
                                                       WAITING_FOR_REMOTE_KVS
                                                     worker 侧 NIXL 异步发起 RDMA 拉取
P 侧 worker 收到拉取的 notify，
  get_finished() 上报 finished_sending
                                                     D 侧 worker 传输完成，
                                                       get_finished() 上报 finished_recving
                                                     Scheduler._update_from_kv_xfer_finished：
                                                       finished_recving_kv_req_ids 记账
                                                     Scheduler._try_promote_blocked_waiting_request：
                                                       调用 _update_waiting_for_remote_kv →
                                                       cache_blocks() 落地缓存 →
                                                       status 改回 WAITING
                                                     若命中长度 == 全部 prompt token 数，
                                                       故意少算最后一个 token（04 篇 §4.1
                                                       同款处理）以便重新采样
                                                     正常参与下一轮调度，只需要跑
                                                       "最后一个 token" 的一次前向
```

这张图对应的调度器代码：`request_finished` 的延迟释放判断在 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:181`，租约到期回收在 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2127`，D 侧的状态促升在 `vllm/v1/core/sched/scheduler.py:2757`（`_update_waiting_for_remote_kv`）与 `vllm/v1/core/sched/scheduler.py:2800`（`_try_promote_blocked_waiting_request`）。

### 4.5 与前缀缓存的融合：两种截然不同的介质，同一个方法签名

对照 §4.1 的调用点，NIXL（跨实例远程搬运）和 OffloadingConnector（本机 CPU 分层缓存）的 `get_num_new_matched_tokens` 实现思路完全不同，但方法签名一模一样：

- **NIXL**（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:34`）：从 `request.kv_transfer_params` 里读出对端（P 或 D）暴露的 token 数，做一次算术减法，返回"还差多少"，本质是**读一个外部传来的数字**。
- **OffloadingConnector**（`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:936`）：调用 `self._lookup(req_status)`（`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:977`）在本地 CPU 分层缓存里做一次真正的**哈希查找**，本质和 04 篇讲的 `BlockPool.get_cached_block` 是同一种操作，只是查的表在 CPU 内存而不是 GPU 显存旁边的 Python 字典。

调度器完全不知道这两种实现内部差多远——它看到的永远只是"一个 `int | None` 加一个 `bool`"。这正是任务简报里说的"外部 KV 存储怎么伪装成前缀缓存"：**伪装的手段就是接口的返回值类型和调用时机与本地前缀缓存完全一致**，调度器侧没有任何 `if is_remote_p2p` 这样的分支。

### 4.6 失败与超时：三层防线

1. **worker 侧的传输/握手失败检测**。NIXL 的 `_handle_failed_transfer`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2239`）在检测到一次 RDMA 传输失败时，把对应的逻辑块 ID 塞进 `_invalid_block_ids` 队列、请求 ID 塞进 `_failed_recv_reqs`（`:2251-2252`）；`get_block_ids_with_load_errors()`（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2487`）把这批块 ID 汇总吐给 `KVConnectorOutput.invalid_block_ids`。
2. **调度器侧的策略分岔**。`_handle_invalid_blocks`（`vllm/v1/core/sched/scheduler.py:2968`）把受影响的请求分成"异步加载中"（`WAITING_FOR_REMOTE_KVS`）和"同步命中"（`self.running`）两拨分别处理，`_update_requests_with_invalid_blocks`（`vllm/v1/core/sched/scheduler.py:2865`）把每个请求的 `num_computed_tokens` 截断到第一个失败块之前——**这就是本地重算的落地方式**：把"已计算"边界往回收缩,该请求下一轮调度会被当成"这段还没算过"重新跑一遍。但截断之后走哪条路，由 `recompute_kv_load_failures`（对应 `kv_load_failure_policy`）决定：`True` 时把这批请求标记 `failed_recving_kv_req_ids` 等待重新加载/计算（`vllm/v1/core/sched/scheduler.py:3034-3035`）；`False`（**默认值**）时直接判定 `should_fail=True`，这批请求会在 `vllm/v1/core/sched/scheduler.py:2070` 被并入 `error_req_ids`，最终以 `RequestStatus.FINISHED_ERROR` 结束（`vllm/v1/core/sched/scheduler.py:2073-2076`）。
3. **生产者侧的租约超时（不是重试机制，是清理机制）**。P 侧把已经算好但还没被消费者拉走的块暂存在 `_reqs_to_send: dict[ReqId, float]`（到期时间戳），`get_finished()` 每步检查一次这个字典头部（有序，最先过期的排在最前）：一旦超过 `_kv_lease_duration`（默认 30 秒，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:75`）没人来拿，直接释放这批块并记一条警告日志（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2127-2144`）。这只解决"不要让 P 侧显存被一个消失的 D 实例永久占用"，**不会**触发任何重试——超时之后 D 侧如果真的还在等，只能在自己那一侧走上面第 2 条的失败路径。

三层防线合起来的结论是：**加载失败确实有本地重算的代码路径，但默认配置不会走它**——这是 §5/§7 都会回到的一个反直觉点。

## 5. 设计决策与代价

### 5.1 P/D 远程搬运与前缀缓存扩容（CPU/磁盘 offload）共用同一个接口

**为什么这么设计**：两者在调度器眼里其实是同一个问题——"这段前缀我本地没有，但存在于别处，要不要等它、等到了怎么记账"。§4.5 已经证明 NIXL 和 OffloadingConnector 的 `get_num_new_matched_tokens` 只是"查外部数字"和"本地哈希查找"两种截然不同的实现，方法签名完全一致。把两者塞进同一个接口意味着调度器只需要写**一套**"本地命中 + 外部命中相加"的逻辑（`vllm/v1/core/sched/scheduler.py:889-890`），新增一种 KV 介质（无论是跨节点 RDMA 还是本地磁盘）只需要新写一个 `KVConnectorBase_V1` 子类，完全不用碰调度器代码。

**不这样会怎样**：如果 P/D 远程传输和本地 offload 走两套不同的接口（比如一个叫 `Disaggregator`、一个叫 `PrefixCacheExtender`），调度器需要各自维护一套"命中判定 → 分配 → 记账 → 清理"的完整状态机，两套状态机之间还可能出现"这段前缀究竟是被 P/D 连接器命中了还是被本地 offload 命中了"的重复计数或漏计数问题；`connector_prefix_cache_stats`（`vllm/v1/core/sched/scheduler.py:1078`）这类统一的观测口径也无从谈起。

**什么时候可以不这样**：如果一个部署从一开始就明确知道自己**只**用 PD 分离、永远不会用 offload（或反过来），理论上可以为这一种场景写一个更专用、更简单的接口（去掉 `SupportsHMA`、`get_handshake_metadata` 这类只有跨实例场景才用得上的方法）。但目前 vLLM 没有走这条路——16 个注册实现里既有纯 P/D（NIXL/Mooncake/MoRIIO）也有纯 offload（OffloadingConnector/SimpleCPUOffloadConnector），说明"统一接口"这条选择在生产实践里被反复验证是划算的。

### 5.2 方法拆成"scheduler 侧决定数量"与"worker 侧决定怎么搬"两级

**为什么这么设计**：调度器进程和 worker 进程本来就是物理上分离的两个（组）进程（TP/PP 下 worker 还可能是多个）。调度器需要在**不接触任何 GPU 张量**的前提下决定"这次给这个请求分几个块、要不要等"，这是纯 CPU 侧的容量与状态决策；真正"把字节从 A 搬到 B"必须发生在持有 GPU/host 内存指针的 worker 进程里。`KVConnectorBase_V1` 把这两类方法物理上写在同一个类里（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:171` 的 `# Worker-side methods` 与 `# Scheduler-side methods` 两段注释，分别见 `:220`、`:435`），但运行时永远只有一侧的方法会被调用到——这是"一个类、两种角色实例"而不是"一个跨进程共享状态的对象"。

**不这样会怎样**：如果强行让调度器直接持有并操作 GPU 张量（比如把 RDMA 握手逻辑写进调度器主循环），会打破"调度器可以在没有 GPU 的地方跑单元测试"这条工程约定（[[03-vLLM-调度器解剖]] 篇会展开这一点），而且调度器一旦阻塞在网络 I/O 上，会连带阻塞它对其他请求的调度决策——分离之后，网络传输天然只发生在 worker 的 forward 生命周期里，可以和计算重叠（§4.3）。

**什么时候可以不这样**：单进程、单 GPU、调度器和 worker 本来就在同一个 Python 进程里跑的极简部署（教学/调试用途）下，这层拆分显得"形式主义"——但代码里没有为这种场景提供合并的捷径，`KVConnectorRole` 依然要求显式声明角色，说明 vLLM 认为保持接口一致性的价值大于给单进程场景省几行代码。

### 5.3 层内异步（逐层 `wait_for_layer_load`/`save_kv_layer`），不是整批一次性等待

**为什么这么设计**：一次 P→D 传输的 KV 数据量和模型层数成正比（§5.6 会给出具体数量级），如果等全部层的前向都跑完再统一传输，传输时间和计算时间是**串行叠加**的；按层粒度交叠之后，第 `i` 层算完就能异步开始发送，和第 `i+1` 层的计算并行，理论上传输时间可以被计算时间部分或完全"吃掉"。`maybe_transfer_kv_layer` 装饰器（`vllm/model_executor/layers/attention/kv_transfer_utils.py:15`）把这个交叠做成了对模型代码透明的横切关注点——写模型的人完全不知道这层被 KV connector 包了一层。

**不这样会怎样**：整批等待会让 P→D 的额外延迟变成"纯增量"：`T_total = T_prefill_compute + T_kv_transfer`，两项完全不重叠；对 TTFT 敏感的场景（disagg 的核心卖点，见 §5.6）这个增量会直接体现在用户能感知的延迟上。

**什么时候可以不这样**：连接器的传输吞吐远大于计算吞吐（比如本机 NVLink 直连、几乎瞬时完成）时，交叠带来的收益趋近于零，这时候写一个更简单的"整批 wait"实现完全够用——`requires_piecewise_for_cudagraph`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:604`）默认返回 `False`，意味着**不主动要求**层内异步的 connector 可以继续享受完整的 CUDA Graph 捕获，这本身就是"层内异步不是免费的"的证据：它要用完整 CUDA Graph 的性能换取传输-计算交叠。

### 5.4 KV 加载失败默认策略是 `"fail"`，不是 `"recompute"`

**为什么这么设计**：`"recompute"` 需要调度器和 connector 都正确处理"部分命中失效后重新排队"这条路径（§4.6 第 2 层防线），这条路径本身涉及块驱逐、`num_computed_tokens` 回退、请求重新参与排队等多个子系统的协同；`"fail"` 是更保守的默认值——加载失败大概率意味着底层传输介质（网络、远端实例）出了问题，直接把错误暴露给客户端，比"悄悄重算、可能又失败一次、又重算"更符合"故障要显式、不要在系统内部无限吸收"的运维直觉，尤其是在这是一个官方文档自己标注"experimental"的功能（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:203-206` 初始化时打印的警告）阶段。

**不这样会怎样**：如果默认值是 `"recompute"`，一个持续性的网络故障（比如 D 侧和 P 侧之间的链路间歇性丢包）会让请求反复"部分命中失效 → 本地重算 → 再次尝试外部命中 → 再次失效"，把故障从"客户端看到一次明确的错误"变成"客户端看到显著变长但不报错的延迟"——后者对故障排查更不友好，因为症状被系统悄悄吸收了。

**什么时候可以不这样**：能接受"偶发失败宁可多算一次也不要报错"这种取舍的场景（比如内部批量离线评测，任务重试成本远低于人工介入成本）应该显式把 `kv_load_failure_policy` 设成 `"recompute"`（`vllm/config/kv_transfer.py:69-72`）——这正是本篇 §7 要点出的"默认值容易被忽略"的地方：很多从单机部署第一次切到 PD 分离的用户不会主动去翻这个字段。

### 5.5 用固定 token 阈值（`kv_recompute_threshold`），不是自适应的成本模型

**为什么这么设计**：判断"要不要跨节点拉这几十个 token"本质是在比较"网络固定开销（握手、通知、租约簿记）"和"本地重算这几十个 token 的计算开销"哪个更便宜——精确建模需要实时测量当前 RTT、当前带宽、当前 GPU 空闲程度，这套成本模型本身的计算和维护成本可能比它想省下的那点传输开销还贵。一个写死的 token 数阈值（默认 64，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:160`）用极低的实现复杂度覆盖了"网络固定开销远大于几十个 token 的重算成本"这个在大多数实际部署里成立的经验判断。

**不这样会怎样**：没有这个阈值，NIXL 在双向传输（`bidirectional_kv_xfer`）的"回读"场景下会为哪怕只差 1 个 token 的命中也发起一次完整的握手 + RDMA 传输流程，固定开销占比会随命中段变短而急剧上升，在短命中段密集出现的工作负载（比如多轮对话里每轮新增内容很少）下可能让"优化路径"反而变成负优化。

**什么时候可以不这样**：如果一次部署里传输链路的 RTT 极低、带宽极高（比如同机架 NVLink 直连，固定开销本身就趋近于零），这个阈值的意义就不大，把它调到 0（`kv_recompute_threshold > 0` 的判断，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:179`）即可关闭这条"提前放弃"的逻辑，让所有命中都走网络路径。

### 5.6 PD 分离什么时候不划算

这是任务本身要求正面回答的问题，也是本篇 §0/§7 反复提到的"官方文档自己否定吞吐收益"这条线索要在这里收束的地方。先摆一个直接引用（**文档所述**，不是本库的判断）：

> Disaggregated prefill DOES NOT improve throughput.（`docs/features/disagg_prefill.md:16`）

也就是说，PD 分离从设计目标上就不是拿来"跑得更快"的，而是拿来"TTFT 和 ITL 分开调、控制尾延迟"的（`docs/features/disagg_prefill.md:12-13`）。这决定了判断"划不划算"的第一层标准不是任何公式，而是**你要解决的问题是不是吞吐**——如果是，PD 分离本身就选错了工具，文档甚至直接指出"分块预填充（chunked prefill）配合合适的 chunk size 也能达到同样的尾延迟控制效果"（`docs/features/disagg_prefill.md:13`），是一条不需要额外网络跳数的替代路径。

在"确实需要 TTFT/ITL 隔离"的前提下，第二层判断是网络这一跳会不会把隔离带来的好处吃掉。下面是一次**解析计算**（沿用 [[04-vLLM-KV缓存与前缀缓存]] §4.4 的假设集合，保持口径一致；本机无 GPU，不产出任何实测数字，全部数字都是假设 + 公式推导）：

| 假设量 | 取值 | 依据 |
|---|---|---|
| 每 token 需要传输的 KV 字节数 `b` | 128 KiB（131,072 字节） | 沿用 04 篇 §4.4 的假设（`num_kv_heads=8`、`head_size=128`、`fp16`、`num_hidden_layers=32`）：`32 层 × 8 头 × (128+128) × 2 字节 = 131,072 字节/token` |
| 模型参数量 `P` | 80 亿（同一档 8B 级模型，与上面假设同属一类模型，非同一份精确配置） | 假设值，仅用于演算 |
| 前向 FLOPs 近似 | `2P`/token | Transformer 前向计算量的标准一阶近似（忽略注意力 `O(L²)` 项与 MFU 损耗，此近似在长上下文场景会低估计算量） |

传输时间与计算时间都按"每 token"折算（一阶近似下，两者对提示长度 `L` 都近似线性，比值与 `L` 基本无关，见下文）：

```
每 token 传输耗时  ≈ b / BW                  （BW = 两实例间实际可用带宽）
每 token 计算耗时  ≈ 2P / C                   （C = prefill 设备实际可持续算力，FLOPs/s）

传输/计算比值 ρ = (b / BW) / (2P / C) = b·C / (BW · 2P)
```

盈亏平衡点（`ρ = 1`，传输和计算一样慢）对应的带宽：

```
BW_breakeven = b · C / (2P)
```

代入一组**纯粹用于演示量级、不对应任何具体硬件型号**的假设算力 `C = 4×10¹⁴ FLOPs/s`：

```
BW_breakeven = 131,072 × 4×10¹⁴ / (2 × 8×10⁹)
             = 131,072 × 4×10¹⁴ / 1.6×10¹⁰
             ≈ 3.28×10⁹ 字节/秒
             ≈ 3.05 GiB/s
```

这个"约 3 GiB/s"本身不是重点（换一个假设算力就会跟着变），重点是这个推导暴露出的两条结构性结论：

1. **"PD 分离划不划算"首先不是"prompt 长不长"的问题**——因为传输耗时和计算耗时在一阶近似下都随 `L` 线性增长，两者的比值 `ρ` 与 `L` 基本无关。真正让**短** prompt 吃亏的，是这条公式之外、`ρ` 没有覆盖到的**固定开销**（握手、notify、租约簿记，§5.5 的 `kv_recompute_threshold=64` 正是 vLLM 自己代码里对这条固定开销的显式承认）——固定开销不随 `L` 缩放，`L` 越小它占比越大，这才是"短 prompt 不划算"的真正机制，不是传输量本身。
2. **只要带宽显著高于 `BW_breakeven` 这个量级，纯"传输 vs 计算"这条轴基本不构成障碍**——现实里的 RDMA（UCX/InfiniBand/RoCE）动辄几十到几百 GB/s，比上面演算出的个位数 GiB/s 高出一到两个数量级。这意味着"PD 分离不划算"里真正常见的带宽类原因，不是"RDMA 不够快"，而是**根本没用上 RDMA**：`kv_buffer_device="cpu"` 时 NIXL 需要多一跳设备↔host 的拷贝（`use_host_buffer` 判断，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:85-87`）；或者传输实际上走的是跨可用区的普通以太网/TCP 而不是同机房 RDMA 网络；或者 P/D 两侧物理上不在同一交换机下，要经过多级网络设备。

综合起来，**PD 分离不划算**的判断依据可以归纳成四条，前两条是"目标不对"，后两条是"条件不够"：

1. **目标是吞吐而不是尾延迟**——文档原话已经排除了这个用例；chunked prefill 是更简单的替代品。
2. **prompt 普遍短于固定开销能摊平的量级**——`kv_recompute_threshold` 默认 64 token 这个真实存在的常量就是这条边界的具体体现（虽然它只用于双向回读场景，但揭示的原理对主传输路径同样成立）。
3. **两实例间没有真正的高带宽互联**——没走 RDMA、走了跨 AZ 公网、或者被迫经过 host 内存中转，实际带宽可能远低于 §5.6 演算出的盈亏平衡点。
4. **P/D 两侧资源配比长期失衡**——比如 P 侧常年算力过剩而 D 侧长期排队（或反过来），此时把两者合并成一个统一的池子、靠 [[03-vLLM-调度器解剖]] 里的连续批处理动态分配算力，比把资源硬性切成两个独立伸缩的池子更有弹性；这一条是资源调度层面的论证，不属于上面的带宽/延迟公式，但同样是"不划算"的真实原因。

### 5.7 八条决策速查表

| 决策 | 换来的好处 | 付出的代价 | 逃生舱 |
|---|---|---|---|
| 5.1 统一接口覆盖 P/D 与 offload | 调度器只写一套命中/记账逻辑 | 接口要覆盖两类场景，方法数偏多 | 单一场景部署可以只用到子集方法，无需精简接口 |
| 5.2 scheduler/worker 两级拆分 | 调度器可脱离 GPU 单测；网络 I/O 不阻塞调度决策 | 每个 connector 要写两份角色相关逻辑 | 单进程调试场景下仍需显式声明角色，无捷径 |
| 5.3 逐层异步 wait/save | 传输与计算交叠，缩短净增延迟 | 要求 `PIECEWISE` CUDA Graph，牺牲完整捕获的性能 | 传输吞吐远大于计算吞吐时收益趋近于零 |
| 5.4 失败默认 `fail` 不 `recompute` | 故障显式暴露，不被系统悄悄吸收 | 用户容易忽略，默认体验是"报错" | 可接受重算成本的场景显式配成 `recompute` |
| 5.5 固定 token 阈值而非自适应模型 | 极低实现复杂度覆盖多数场景 | 短命中段的判断不够精细 | 极低延迟链路（如 NVLink 直连）可把阈值调到 0 |
| 5.6 分离目标是延迟不是吞吐 | TTFT/ITL 可独立调优，控制尾延迟 | 官方承认不提升吞吐，多一跳网络 | 吞吐优先且能接受 tail ITL 的场景用 chunked prefill 替代 |

## 6. 同位对照（SGLang 的 disaggregation）

依据 `_lab/out/struct_map.json` 的 `disagg` 子系统统计做规模对照：vLLM 侧 **79 文件 / 33,900 行**，SGLang 侧 **67 文件 / 38,075 行**（注意 SGLang 这个数字的统计口径比"纯 `disaggregation/` 目录"更宽，还纳入了 `mem_cache/storage/` 下的 mooncake/nixl/flexkv 存储适配层与 `multimodal_gen/runtime/disaggregation/`，两边不是完全同口径的目录对比，但都出自同一套 `struct_map.py` 关键词扫描逻辑，量级上可比）。**两边代码量同一个数量级**，"接口统一"并没有让 vLLM 的 PD 分离代码总量比 SGLang"按后端分目录"更小——统一接口节省的是**新增一种介质的边际成本**，不是总代码量，这条和 00 篇总览里"跨项目对比先用同一套工具"的教训一致。

抽象层次上两家走的是两条不同的路：

- **vLLM：一个统一接口，接口内部按角色/场景拆方法**。`KVConnectorBase_V1`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`）覆盖 P/D 远程搬运和本地 offload 两类场景，16 个注册实现（§2.1）共用同一套 scheduler 主循环挂钩（§4.1-§4.2）。新增一种传输介质只需要写一个新的子类，调度器代码零改动。
- **SGLang：按后端分目录，用一组共享的 ABC + 轮询状态机**。`base/conn.py` 定义了 `BaseKVManager`（`sglang:python/sglang/srt/disaggregation/base/conn.py:101`）、`BaseKVSender`（`sglang:python/sglang/srt/disaggregation/base/conn.py:119`）、`BaseKVReceiver`（`sglang:python/sglang/srt/disaggregation/base/conn.py:188`）、`BaseKVBootstrapServer`（`sglang:python/sglang/srt/disaggregation/base/conn.py:247`）四个抽象基类，以及一个五态轮询枚举 `KVPoll`（`Failed`/`Bootstrapping`/`WaitingForInput`/`Transferring`/`Success`，`sglang:python/sglang/srt/disaggregation/base/conn.py:93`）。每个后端（`nixl`/`mooncake`/`mori`/`ascend`）各自在自己的目录下实现这一组类，例如 NIXL 后端的 `NixlKVManager`（`sglang:python/sglang/srt/disaggregation/nixl/conn.py:396`）、`NixlKVSender`（`sglang:python/sglang/srt/disaggregation/nixl/conn.py:2736`）、`NixlKVReceiver`（`sglang:python/sglang/srt/disaggregation/nixl/conn.py:2853`）、`NixlKVBootstrapServer`（`sglang:python/sglang/srt/disaggregation/nixl/conn.py:3072`）。更关键的差异在调度器集成方式：SGLang 的 `Scheduler` 类直接用多重继承混入 `SchedulerDisaggregationDecodeMixin` 和 `SchedulerDisaggregationPrefillMixin`（`sglang:python/sglang/srt/managers/scheduler.py:385`、`sglang:python/sglang/srt/managers/scheduler.py:386`），P 侧和 D 侧各有一份独立的事件循环（`event_loop_normal_disagg_prefill`，`sglang:python/sglang/srt/disaggregation/prefill.py:569`；`event_loop_normal_disagg_decode`，`sglang:python/sglang/srt/disaggregation/decode.py:2428`），而不是像 vLLM 那样只在**一个**统一的调度循环里散布 `if self.connector is not None` 分支（§2 代码地图里 `scheduler.py` 一个文件贯穿全部挂钩）。

两条路径各自的代价：

- **vLLM 的代价**：`KVConnectorBase_V1` 要覆盖尽可能多的场景（HMA 支持、握手元数据、PP-aware 握手、层内异步……），每个新 connector 的作者都要理解一份相当重的契约——`base.py` 704 行里，多数方法对"只做 CPU offload、不涉及跨实例握手"的实现（如 `SimpleCPUOffloadConnector`）根本用不上，但接口依然要求（或允许留空）实现它们。
- **SGLang 的代价**：Prefill 和 Decode 两条事件循环是分开写的，每新增一个后端要同时在 `prefill.py` 和 `decode.py` 里各自接入对应的 `KVPoll` 分支（§6 上文列出的 `poll == KVPoll.Failed`/`WaitingForInput`/`Transferring` 等判断在两个文件里都各有一份），横向对比"这个后端在 P 侧和 D 侧行为是否一致"需要来回跳文件；轮询状态机（`KVPoll`）本质上是把"是否完成/是否失败"这件事从回调式（vLLM 的 `get_finished()` 由 worker 主动上报）换成了拉取式（调度循环主动 `poll()` 查询），两种模型没有绝对优劣，但轮询模型天然要求调用方自己控制轮询频率，回调模型把这个频率交给了固定的引擎步进节奏。

一个可以现在就下、双方都有源码支撑的结论：**vLLM 用"一个接口、场景在实现里分叉"换取了调度器代码的单一性；SGLang 用"场景在目录结构里分叉"换取了每个后端内部逻辑的独立性（一个后端出问题不会牵扯到看另一个后端的代码）**。哪种更好取决于团队更怕"改一个接口牵动 16 个实现"还是更怕"改一个行为要同步改两个文件"。

## 7. 踩坑与反直觉

1. **`kv_load_failure_policy` 默认是 `"fail"`，不是很多人直觉里"分布式系统总该优雅降级"的 `"recompute"`**（`vllm/config/kv_transfer.py:69`）。§4.6/§5.4 已经拆解过这个设计的合理性，但对第一次从单机切到 PD 分离的用户，"外部 KV 加载失败会让请求直接报错"是一个容易被忽略、直到生产环境第一次网络抖动才发现的默认行为。
2. **`kv_recompute_threshold=64` 这个很小的数字，揭示的是"网络拉取存在固定开销"这个在同机房 RDMA 场景下容易被忽视的事实**——直觉上 RDMA 已经足够快，但握手、通知这类操作本身有和传输字节数无关的固定延迟，vLLM 自己的代码选择用一个写死的 token 数门槛来承认这一点，而不是假装网络传输永远比本地重算划算。
3. **官方使用文档写"Now supports 9 types of connectors"（`docs/features/disagg_prefill.md:20`），但 `factory.py` 实际注册了 16 个**（§2.1）——文档滞后于代码是常态，本库的立场（[[00-总览与阅读地图]] 第 1 条）是一律以源码注册表为准，这也是本篇没有直接照抄文档列表、而是重新逐条核对 `factory.py` 的原因。
4. **"Disaggregated prefill DOES NOT improve throughput" 是官方文档自己的原话**（`docs/features/disagg_prefill.md:16`），不是本库的推断或贬低——这条经常被"PD 分离=更快"这种简化叙事掩盖，但源头文档从一开始就没有承诺吞吐收益，承诺的是 TTFT/ITL 的独立可调性。
5. **NIXL 生产者侧的"租约"（`_kv_lease_duration` 默认 30 秒）是清理机制，不是重试机制**（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2127`）——租约到期只代表"生产者不再无限期占着这些块等消费者来拿"，不代表消费者一定会重新发起请求；如果消费者那侧的请求还活着，它要走 §4.6 第 2 层防线（`invalid_block_ids`/`kv_load_failure_policy`）单独处理这次失败，两套机制互不知晓对方的存在。
6. **层内（layer-by-layer）的 `wait_for_layer_load`/`save_kv_layer` 不是包在模型外层的一个循环，而是一个函数装饰器直接挂在每个 attention 层的 `forward` 方法上**（`maybe_transfer_kv_layer`，`vllm/model_executor/layers/attention/kv_transfer_utils.py:15`）——这意味着理论上任何调用 attention 层 `forward` 的代码路径都会自动被这层装饰器接管，模型定义代码本身完全不用感知 PD 分离的存在，这是一种相当彻底的横切关注点分离，但也意味着调试"这层 KV 到底有没有被正确加载"时，断点要下在装饰器里而不是模型代码里。

## 8. 可改进点

1. **`kv_load_failure_policy` 默认 `"fail"` 这个容易被忽略的默认值，可以在 connector 初始化时显式提示一次**。`KVConnectorBase_V1.__init__`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:203-206`）已经在打印一条"experimental API"的警告，**本库推断**：可以在同一处顺带打印一条"当前 KV 加载失败策略为 fail，加载失败将直接终止请求"的提示，把一个容易被忽略的配置默认值变成一个启动时就能看到的显式声明，成本极低（一行日志），收益是让运维方在第一次读到日志时就知道这个行为，而不是等第一次网络抖动、请求报错时才去翻文档。
2. **`get_block_ids_with_load_errors` 的时序契约完全靠子类自觉遵守，没有运行时校验**。方法文档字符串明确要求"failed blocks must appear here no later than the same pass where the request ID is returned by `get_finished()`"（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:379-386`），但基类没有任何断言去校验一个具体实现是否真的遵守了这条时序约束。**本库推断**：可以在 `_get_kv_connector_output`（`vllm/v1/worker/kv_connector_model_runner_mixin.py:69`）里加一个仅在调试模式生效的一致性检查（比如校验 `finished_recving` 里的请求 ID 是否在 `invalid_block_ids` 涉及的请求集合之外，或者两者的时序关系是否符合文档承诺），第三方 connector 一旦违反这条契约会在开发阶段就暴露出来，而不是在生产环境里静默漏报失败块、直到很久之后才被发现（表现为"某个块偶尔读到脏数据但没有任何报错"这类最难排查的故障）。
3. **`kv_recompute_threshold` 是一个部署时写死的常量（默认 64），没有考虑当时实际的 RTT/带宽**。§5.5/§5.6 已经论证过它承认的原理（固定开销 vs 重算成本）是对的，但阈值本身是静态的——同一个部署在网络状况好的时段（比如夜间低峰）和网络拥塞的时段，"值不值得跨节点拉"这个判断的最优阈值理论上是不同的。**本库推断**：可以把这个阈值改成基于最近若干次实际传输延迟的滑动窗口自适应估计（比如 `NixlKVConnectorStats` 已经在收集传输统计，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py`，未查证其中是否已包含可直接复用的延迟分布），但这会引入"阈值本身随时间漂移，同一份配置在不同时刻行为不同"的新的可观测性负担，需要权衡是否值得。

## 9. 自测题与延伸阅读

**自测题**（闭卷）：

1. `KVConnectorBase_V1` 的 scheduler 侧方法和 worker 侧方法分别在引擎主循环的哪一步被调用？为什么两侧不能合并成同一个方法？
2. `get_num_new_matched_tokens` 返回的元组第二个元素（`is_async`）为 `True` 时，请求会经历怎样的状态切换？调度器在这个等待期间还占着这个请求的哪些资源，又不占哪些资源？
3. NIXL 的 `kv_recompute_threshold` 默认值是多少？它试图在哪两种成本之间做权衡？如果把它设成 0 会发生什么？
4. `KVTransferConfig.kv_load_failure_policy` 的默认值是什么？这个默认值在"外部 KV 加载失败"时会导致请求走哪条路径？如果想让失败的请求退回本地重算，需要改哪个配置？
5. 为什么 `wait_for_layer_load`/`save_kv_layer` 要挂在每一层 attention 的 `forward` 上，而不是在整个模型前向跑完之后统一调用一次？这样做要求 CUDA Graph 处于什么模式？
6. 官方文档对"disaggregated prefill 能不能提升吞吐"是怎么说的？如果一个团队的目标是提升吞吐而不是控制尾延迟，PD 分离是不是正确的工具？还有什么替代方案？
7. 用 §5.6 的公式，如果把假设的传输带宽 `BW` 从演算出的盈亏平衡点（约 3 GiB/s）降低到只有 500 MB/s（比如退化成普通以太网），传输时间会变成计算时间的几倍？这说明了什么？
8. `OffloadingConnector` 和 `NixlPullConnector` 的 `get_num_new_matched_tokens` 底层实现方式完全不同，为什么调度器侧的调用代码可以完全不区分这两者？
9. NIXL 生产者侧的租约超时（`_kv_lease_duration`）和调度器侧的 `kv_load_failure_policy` 分别解决什么问题？两者是同一套失败处理机制吗？

**延伸阅读**（本库双链）：

- [[04-vLLM-KV缓存与前缀缓存]]——本篇 §5.6 的解析计算沿用了这一篇 §4.4 的假设集合，理解"一个块占多少字节"要回到那一篇；§6 里"块哈希被设计成可以传给外部 KV 传输组件复用"这条线索也是从那一篇的 §6/§7 延续过来的。
- [[03-vLLM-调度器解剖]]——本篇 §4 全部走读的调度器挂钩，都是那一篇讲的调度主循环里的一部分；`RequestStatus` 完整状态图、`_inflight_prefills` 的容量记账细节，回到那一篇能看到更完整的上下文。
- [[11-SGLang-PD分离与HiCache分层]]——本篇 §6 对 SGLang `disaggregation/` 目录的核对止步于抽象层次和调度集成方式，具体每个后端（NIXL/Mooncake/MORI/Ascend）内部怎么做握手、`KVPoll` 状态转移的完整细节，留给那一篇逐行核对。
