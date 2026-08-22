# SGLang PD 分离与 HiCache 分层缓存解剖

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：PD 分离把 KV 摆到别的机器上，HiCache 把 KV 摆到别的存储层上——两本账都靠同一套“轮询状态机 + 阈值旋钮”记。

## 0. 结论先行

1. **PD 分离的握手不是回调、不是消息队列，是一台五态轮询状态机**——`KVPoll`（`python/sglang/srt/disaggregation/base/conn.py:93`-`98`）只有 `Failed/Bootstrapping/WaitingForInput/Transferring/Success` 五个值，prefill 侧的 `PrefillBootstrapQueue.pop_bootstrapped`（`python/sglang/srt/disaggregation/prefill.py:383`）和 decode 侧的 `DecodeTransferQueue.pop_transferred`（`python/sglang/srt/disaggregation/decode.py:2223`）各自在自己的调度循环里主动去问“你好了没”，没有第三方仲裁者、没有事件总线唤醒。
2. **SGLang 不是“每个传输后端从零实现一遍”，而是三层继承栈**：`base/conn.py` 只定义抽象契约（`BaseKVManager`/`BaseKVSender`/`BaseKVReceiver`/`BaseKVBootstrapServer`），真正的 bootstrap 握手、房间号（`bootstrap_room`）管理、HTTP 注册这些**与传输介质无关**的逻辑全部收在 `common/conn.py`（1898 行）的 `CommonKVManager`（`python/sglang/srt/disaggregation/common/conn.py:142`）等类里，`nixl`/`mooncake`/`mori` 三个后端的 Manager 类都直接继承它，只重写“怎么把字节真正搬过去”这一段（见 §2、§6）。
3. **HiCache 的“值不值得往下写”是一个命中次数阈值，不是收益模型**——`write_through` 策略下命中 1 次就写 L2，`write_through_selective` 要求命中 ≥2 次（`python/sglang/srt/mem_cache/hiradix_cache.py:207`），阈值本身写死为 `1`/`2` 两个整数，源码自己留了一句 `# todo: dynamically adjust the threshold`（`python/sglang/srt/mem_cache/hiradix_cache.py:206`）——没有对比“写下去要花多少 PCIe 时间”和“将来省下多少次重算”。
4. **预取（prefetch）是异步的，但调度器这边是“轮询 + 跳过”而不是“状态机 + 唤醒”**——`Scheduler` 在组下一个 prefill batch 时对每个候选请求调 `check_prefetch_progress(req.rid)`（`python/sglang/srt/managers/scheduler.py:3386`），没完成就 `continue` 试队列里下一个，该请求本身停留在 `waiting_queue` 里等下一轮 tick 再问一次——没有专门的“等待预取”状态枚举。
5. **PD 分离与 HiCache 在 decode 侧有一条真实焊死的交叉线**：decode 收到请求后，`DecodePreallocQueue`（继承 `DecodeHiCachePreallocMixin`，`python/sglang/srt/disaggregation/decode.py:300`）会先查本地 L1/L2/L3 有没有现成前缀（`DecodePrefixMatch`，`python/sglang/srt/disaggregation/decode_hicache_mixin.py:24`），**只有本地都没命中的那一段，才需要真的从 prefill 侧网络传输过来**——这是本篇“同一个问题”的直接代码证据，不是比喻。
6. **一个容易踩的命名坑**：HiCache 的 L3 存储后端注册表里也有一个叫 `"mori"` 的选项（`python/sglang/srt/mem_cache/storage/backend_factory.py:241`-`245`，指向 `storage/umbp/umbp_store.py` 的 `UMBPStore`），和 PD 分离传输后端里的 `mori`（`python/sglang/srt/disaggregation/mori/conn.py`，AMD 的 `mori.io` RDMA 引擎）**是完全不同的两个库，只是同名**——正文 §7 第 1 条详细核对。

## 1. 它在系统里的位置

调度器（见 [[02-SGLang-Scheduler事件循环]]）对“这段 KV 现在在哪、该往哪搬”这件事，实际上要在两条正交的轴上做决定：

```
                     空间轴：这段 KV 该由谁算、传给谁
        prefill 实例 ──KV 传输(nixl/mooncake/ascend/mori)──▶ decode 实例
        （disaggregation_mode=prefill）                    （disaggregation_mode=decode）
                     │                                              │
                     ▼                                              ▼
        ┌─────────────────────── 时间轴：这段 KV 现在住在哪一层 ───────────────────────┐
        │  L1 GPU 显存 (RadixCache 本体，见 [[04-SGLang-内存池与KV布局]])              │
        │      ↕ L2TransferEngine (mem_cache/l2_transfer.py)                          │
        │  L2 CPU 内存池 (HostKVCache / memory_pool_host.py)                          │
        │      ↕ HiCacheStorage 后端 (nixl/mooncake/hf3fs/file/aibrix/…)               │
        │  L3 远端/磁盘存储 (mem_cache/storage/)                                       │
        └───────────────────────────────────────────────────────────────────────────┘
```

空间轴由 `disaggregation/` 整个目录负责，时间轴由 `HiRadixCache`（继承 [[03-SGLang-RadixAttention与前缀缓存]] 精读过的 `RadixCache`）加 `mem_cache/storage/` 负责。两者共享同一个下层事实——KV 张量本身的物理布局（[[04-SGLang-内存池与KV布局]] 讲过的 `KVCache`/`get_kv_buffer(layer_id)`）——但在这之上各自演化出一套独立的旋钮、独立的队列类、独立的命名空间（`server_args.py` 里 `disagg` 和 `memory` 两个不同的 `NS()` 分组）。

两条轴唯一的正式交叉点在 decode 侧：`decode_hicache_mixin.py` 文件顶部的 docstring 写得很直白——“HiCache integration mixins **for the decode side of PD disaggregation**”（`python/sglang/srt/disaggregation/decode_hicache_mixin.py:1`）。一个 decode 实例收到一个已经在 prefill 侧跑过的请求后，不会天真地假设“KV 一定要从网络传过来”——它先在本地的 L1/L2/L3 走一遍前缀匹配，只把本地确实没有的那一段标记为“需要从 prefill 网络传输”（详见 §4.5）。这意味着**一个开了 HiCache 的 decode 实例，本质上是把“该不该发起 PD 网络传输”这个决策，从“是否 PD 分离”单独一个开关，变成了“本地缓存命中率”和“网络传输代价”之间的一次即时比较**——本篇 §5 会把这次比较写成一条解析公式。

**角色划分里还缺一环：谁负责把外部请求分发到哪一对 prefill/decode 实例？** 这一环不在 `python/sglang/srt/disaggregation/` 目录里，也不在任何 Python 文件里——它是一个独立的 Rust crate `sgl-model-gateway`（2025 行的 `sgl-model-gateway/src/routers/http/pd_router.rs`，另有 gRPC 版 `sgl-model-gateway/src/routers/grpc/pd_router.rs`）。`PDRouter` 结构体（`sgl-model-gateway/src/routers/http/pd_router.rs:50`）持有 `worker_registry`（`WorkerRegistry`，`sgl-model-gateway/src/core/worker_registry.rs:180`）和 `policy_registry`（`PolicyRegistry`，`sgl-model-gateway/src/policies/registry.rs:19`）两个成员；每次请求进来，`select_pd_pair`（`sgl-model-gateway/src/routers/http/pd_router.rs:972`）分别从 prefill worker 池和 decode worker 池里各选一个（`prefill_policy`/`decode_policy` 是两条独立的负载均衡策略），再靠 `worker_registry.get_hash_ring(...)`（一致性哈希环，`HashRing` 定义在 `sgl-model-gateway/src/core/worker_registry.rs:43`）尽量把同一请求／同一前缀路由到之前处理过它的那个 prefill worker——这与 HiCache 的“同一前缀尽量别重算”是同一个目标，只是发生在请求分发这一层，而不是缓存这一层。选好一对之后 `execute_dual_dispatch`（`sgl-model-gateway/src/routers/http/pd_router.rs:365`）把请求同时转发给这一对 worker，双方再各自走本篇 §4.1 的 `KVPoll` 握手流程——**Rust 网关只负责“选谁”，不参与“怎么搬 KV”，两者是完全解耦的两层**。

**还有一条独立的轴，容易被误认成 PD 的“第三种角色”，但其实是另一回事**：`disaggregation/encoder/` 子目录服务的是多模态模型的 **Encode-Prefill-Decode（EPD）** 拆分，把图像/视频编码单独摘成第三种实例类型，与本篇主线的 P/D 两角色是不同维度的拆分（可以同时开，也可以只开 P/D 不开 E）。GPU 侧编码器封装在 `MMEncoder`（`python/sglang/srt/disaggregation/encoder/server.py:438`）；调度侧是 `EncoderScheduler`（`python/sglang/srt/disaggregation/encoder/runtime.py:101`）和 `EncoderRuntime`（`python/sglang/srt/disaggregation/encoder/runtime.py:353`）；语言模型侧（prefill/decode 所在的进程）用 `EncoderBootstrapServer`（`python/sglang/srt/disaggregation/encoder/receiver.py:65`）接收编码结果，走的是自己的一套 `MMReceiverHTTP`/`MMReceiverGrpc`（`python/sglang/srt/disaggregation/encoder/receiver.py:2393`/`2525`）接收管线，和 `PrefillBootstrapQueue`/`CommonKVBootstrapServer` 完全是两条不相干的代码路径（§7 第 4 条已经指出两者监听端口都不同）。本篇聚焦 P/D 这一条轴，E 这一条轴的传输细节（`encoder_transfer_backend`）留给专门讲多模态的篇目处理，这里只标出它的存在和入口，避免读者把 `disaggregation/` 目录下的三个角色（encoder/prefill/decode）误当成一套对称设计。

## 2. 代码地图（文件 → 职责，带行号）

### 2.1 PD 分离（`python/sglang/srt/disaggregation/`）

| 文件 | 职责 | 关键行 |
|---|---|---|
| `disaggregation/base/conn.py` | 抽象契约：任何后端都要实现的四个类 + `KVArgs`/`KVPoll` | `KVArgs` `python/sglang/srt/disaggregation/base/conn.py:43`；`KVPoll` `python/sglang/srt/disaggregation/base/conn.py:93`；`BaseKVManager` `python/sglang/srt/disaggregation/base/conn.py:101`；`BaseKVSender` `python/sglang/srt/disaggregation/base/conn.py:119`；`BaseKVReceiver` `python/sglang/srt/disaggregation/base/conn.py:188`；`BaseKVBootstrapServer` `python/sglang/srt/disaggregation/base/conn.py:247` |
| `disaggregation/common/conn.py` | 与传输介质无关的共享实现：bootstrap 握手、房间号、HTTP 注册（1898 行） | `CommonKVManager(BaseKVManager)` `python/sglang/srt/disaggregation/common/conn.py:142`；`register_to_bootstrap` `python/sglang/srt/disaggregation/common/conn.py:753`；`CommonKVSender(BaseKVSender)` `python/sglang/srt/disaggregation/common/conn.py:1141`；`CommonKVReceiver(BaseKVReceiver)` `python/sglang/srt/disaggregation/common/conn.py:1327`；`CommonKVBootstrapServer(BaseKVBootstrapServer)` `python/sglang/srt/disaggregation/common/conn.py:1612`；`/route` 路由注册 `python/sglang/srt/disaggregation/common/conn.py:1660` |
| `disaggregation/common/staging_handler.py` / `staging_buffer.py` | 传输前把分散的按头切片聚合成连续显存的“暂存缓冲区”，供 nixl/mooncake 复用 | `StagingManagerMixin` `python/sglang/srt/disaggregation/common/staging_handler.py:827` |
| `disaggregation/nixl/conn.py` | NIXL 后端：`StagingManagerMixin + CommonKVManager` 只重写真正搬数据那一段 | `NixlKVManager(StagingManagerMixin, CommonKVManager)` `python/sglang/srt/disaggregation/nixl/conn.py:396` |
| `disaggregation/mooncake/conn.py` | Mooncake TransferEngine 后端 | `MooncakeKVManager(StagingManagerMixin, CommonKVManager)` `python/sglang/srt/disaggregation/mooncake/conn.py:196` |
| `disaggregation/ascend/conn.py` | 华为昇腾后端，**直接继承 Mooncake 的 Manager，只换底层 TransferEngine** | `AscendKVManager(MooncakeKVManager)` `python/sglang/srt/disaggregation/ascend/conn.py:32` |
| `disaggregation/mori/conn.py` | AMD `mori.io` 后端 | `MoriKVManager(CommonKVManager)` `python/sglang/srt/disaggregation/mori/conn.py:302` |
| `disaggregation/fake/conn.py` | 不传输，纯状态机空转，给 warmup 请求用 | `FakeKVManager(BaseKVManager)` `python/sglang/srt/disaggregation/fake/conn.py:22` |
| `disaggregation/prefill.py` | prefill 侧队列与调度器 Mixin | `PrefillBootstrapQueue` `python/sglang/srt/disaggregation/prefill.py:119`；`pop_bootstrapped` `python/sglang/srt/disaggregation/prefill.py:383`；`finalize_bootstrap` `python/sglang/srt/disaggregation/prefill.py:336`；`SchedulerDisaggregationPrefillMixin` `python/sglang/srt/disaggregation/prefill.py:485`；`send_kv_chunk` `python/sglang/srt/disaggregation/prefill.py:1132`；`process_disagg_prefill_inflight_queue` `python/sglang/srt/disaggregation/prefill.py:841` |
| `disaggregation/decode.py` | decode 侧两级队列与调度器 Mixin | `DecodeReqToTokenPool` `python/sglang/srt/disaggregation/decode.py:119`；`DecodePreallocQueue(DecodeHiCachePreallocMixin)` `python/sglang/srt/disaggregation/decode.py:300`；`pop_preallocated` `python/sglang/srt/disaggregation/decode.py:1052`；`DecodeTransferQueue(DecodeHiCacheTransferMixin)` `python/sglang/srt/disaggregation/decode.py:1991`；`pop_transferred` `python/sglang/srt/disaggregation/decode.py:2223`；`SchedulerDisaggregationDecodeMixin` `python/sglang/srt/disaggregation/decode.py:2426`；`process_decode_queue` `python/sglang/srt/disaggregation/decode.py:2643` |
| `disaggregation/decode_hicache_mixin.py` | PD×HiCache 交叉点：decode 收请求前先查本地缓存 | 见 §2.2 |
| `disaggregation/decode_kvcache_offload_manager.py` | decode 侧异步把 KV 下沉进 HiCache（用于请求被抢占重试时不丢缓存） | `DecodeKVCacheOffloadManager` `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py:34` |
| `sgl-model-gateway/src/routers/http/pd_router.rs`（Rust，独立 crate，不在 `python/sglang/srt/` 下） | 角色划分里的“路由/mini-lb”：为每个请求各选一个 prefill worker 和一个 decode worker 并双发 | `PDRouter` `sgl-model-gateway/src/routers/http/pd_router.rs:50`；`select_pd_pair` `sgl-model-gateway/src/routers/http/pd_router.rs:972`；`execute_dual_dispatch` `sgl-model-gateway/src/routers/http/pd_router.rs:365`；`WorkerRegistry` `sgl-model-gateway/src/core/worker_registry.rs:180`；`HashRing` `sgl-model-gateway/src/core/worker_registry.rs:43`；`PolicyRegistry` `sgl-model-gateway/src/policies/registry.rs:19` |
| `disaggregation/utils.py` | `DisaggregationMode`/`TransferBackend` 枚举 + 按后端名取具体类的工厂函数 | `DisaggregationMode` `python/sglang/srt/disaggregation/utils.py:99`；`TransferBackend` `python/sglang/srt/disaggregation/utils.py:580`；`get_kv_class` `python/sglang/srt/disaggregation/utils.py:618` |
| `disaggregation/kv_events.py` | KV 块级事件总线（Store/Remove/Cleared），PD 与 HiCache 共用的旁路可观测性通道，本身不传输数据 | `StorageMedium` `python/sglang/srt/disaggregation/kv_events.py:80`；`BlockStored` `python/sglang/srt/disaggregation/kv_events.py:112`；`ZmqEventPublisher` `python/sglang/srt/disaggregation/kv_events.py:185` |
| `arg_groups/pd_disaggregation_hook.py` | 启动期对 `disaggregation_*` 参数做归一化（例如把 `mooncake_tcp` 改写成 `mooncake` + 环境变量） | `handle_pd_disaggregation` `python/sglang/srt/arg_groups/pd_disaggregation_hook.py:16`-`28` |
| `server_args.py` | `disaggregation_*` 全部旋钮 | 见 §3.2 表格，起始行 `python/sglang/srt/server_args.py:3159` |

### 2.2 HiCache（`python/sglang/srt/mem_cache/`）

| 文件 | 职责 | 关键行 |
|---|---|---|
| `mem_cache/hiradix_cache.py` | L1 的“分层版”：在 `RadixCache`（[[03-SGLang-RadixAttention与前缀缓存]] 精读对象）之上叠加 L2/L3 钩子 | `HiRadixCache(RadixCache)` `python/sglang/srt/mem_cache/hiradix_cache.py:77`；写入阈值 `python/sglang/srt/mem_cache/hiradix_cache.py:207`；`write_backup` `python/sglang/srt/mem_cache/hiradix_cache.py:841`；`evict` `python/sglang/srt/mem_cache/hiradix_cache.py:1189`；`evict_host` `python/sglang/srt/mem_cache/hiradix_cache.py:1339`；`can_terminate_prefetch` `python/sglang/srt/mem_cache/hiradix_cache.py:1584`；`check_prefetch_progress` `python/sglang/srt/mem_cache/hiradix_cache.py:1636` |
| `mem_cache/memory_pool_host.py` | L2 CPU 内存池的具体实现（按模型架构分派） | `LogicalHostPool` `python/sglang/srt/mem_cache/memory_pool_host.py:56`；`DeepSeekV4PagedHostPool(HiSparseHostPoolMixin, HostKVCache)` `python/sglang/srt/mem_cache/memory_pool_host.py:178`；`DeepSeekV4StateHostPool(HostKVCache)` `python/sglang/srt/mem_cache/memory_pool_host.py:576` |
| `mem_cache/pool_host/base.py` | `HostKVCache` 抽象基类，显式持有 `device_pool: KVCache` 引用 | `HostKVCache.__init__` `python/sglang/srt/mem_cache/pool_host/base.py:113`-`150` |
| `mem_cache/l2_transfer.py` | L1↔L2 搬运引擎，两条独立 CUDA stream | `L2TransferEngine` `python/sglang/srt/mem_cache/l2_transfer.py:49`；`submit_device_to_host` `python/sglang/srt/mem_cache/l2_transfer.py:57`；`submit_host_to_device`（逐层搬运+回调） `python/sglang/srt/mem_cache/l2_transfer.py:74`-`80` |
| `mem_cache/hicache_storage.py` | L3 抽象接口 + 内置文件后端 + 预取超时公式 | `HiCacheStorage(ABC)` `python/sglang/srt/mem_cache/hicache_storage.py:150`；`PrefetchTimeoutConfig` `python/sglang/srt/mem_cache/hicache_storage.py:50`-`55`；`HiCacheFile(HiCacheStorage)` `python/sglang/srt/mem_cache/hicache_storage.py:361` |
| `mem_cache/storage/backend_factory.py` | L3 后端注册表：名字 → 模块路径 → 类 | `StorageBackendFactory` `python/sglang/srt/mem_cache/storage/backend_factory.py:16`；`register_backend` `python/sglang/srt/mem_cache/storage/backend_factory.py:44`；11 个内置后端注册块 `python/sglang/srt/mem_cache/storage/backend_factory.py:197`-`251` |
| `mem_cache/evict_policy.py` | 7 种可插拔驱逐排序策略 | `EvictionStrategy(ABC)` `python/sglang/srt/mem_cache/evict_policy.py:10`；`LRUStrategy`/`LFUStrategy`/`FIFOStrategy`/`MRUStrategy`/`FILOStrategy`/`PriorityStrategy`/`SLRUStrategy` 分别在 `:16`/`:21`/`:26`/`:31`/`:36`/`:41`/`:49` |
| `managers/cache_controller.py` | 后台预取线程 | `prefetch_thread_func` `python/sglang/srt/managers/cache_controller.py:1128`；`prefetch()` `python/sglang/srt/managers/cache_controller.py:970`；`PrefetchOperation(StorageOperation)` `python/sglang/srt/managers/cache_controller.py:230` |
| `managers/scheduler.py` | 调度循环里轮询预取进度的那一行 | `python/sglang/srt/managers/scheduler.py:3386` |
| `entrypoints/http_server.py` | 运行期热挂载/卸载/查询/清空 L3 后端的管理面 API | `PUT /hicache/storage-backend` `python/sglang/srt/entrypoints/http_server.py:1078`-`1107`；`DELETE /hicache/storage-backend` `python/sglang/srt/entrypoints/http_server.py:1112`-`1134`；`GET /hicache/storage-backend` `python/sglang/srt/entrypoints/http_server.py:1139`-`1154`；`POST /hicache/storage-backend/clear` `python/sglang/srt/entrypoints/http_server.py:1058`-`1066` |
| `server_args.py` | `hicache_*` 全部旋钮 | 见 §3.4 表格，起始行 `python/sglang/srt/server_args.py:2717` |

（此二表合计 40 余条 `文件:行` 引用，已远超 `## 2` 硬指标 10 条；后续小节还会补充更细的行号。）

## 3. 核心数据结构

### 3.1 `KVPoll` 五态机与 `KVArgs`（`base/conn.py`）

```python
class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4
```

（`python/sglang/srt/disaggregation/base/conn.py:93`-`98`）这五个整数常量是 PD 分离全部状态转换的唯一词汇表——prefill 侧的 sender 和 decode 侧的 receiver 各自维护自己的一份，谁都不知道对方在这五个状态里具体停在哪一步的内部细节，只通过各自的 `poll()` 返回值同步。`KVArgs`（`python/sglang/srt/disaggregation/base/conn.py:43`）是握手时打包的“这个进程有哪些可传输的内存”的清单——不仅有 `kv_data_ptrs`/`kv_item_lens` 这类标准 KV 指针，还有 `state_types: List[StateType]`（`python/sglang/srt/disaggregation/base/conn.py:53`）和一整组 `state_data_ptrs`/`state_slice_outer_counts`/`state_conv_shard_groups` 字段，这是为了让同一套传输协议也能搬运 Mamba 状态、SWA 环形缓冲、DSA 索引器 K 缓存这些“不是标准 KV 但也要跟着请求走”的东西（对照 [[04-SGLang-内存池与KV布局]] §3.6 讲过的 `IndexKeyCache`）——PD 传输协议和 KV 池物理布局在“有哪些异构组件”这件事上是同步演进的。

### 3.2 `disaggregation_*` 旋钮（`python/sglang/srt/server_args.py:3159` 起）

| 参数 | 默认值 | 行号 | 作用 |
|---|---|---|---|
| `disaggregation_mode` | `"null"` | `python/sglang/srt/server_args.py:3159` | `null`/`prefill`/`decode` 三选一 |
| `disaggregation_transfer_backend` | `"mooncake"` | `python/sglang/srt/server_args.py:3164` | 见 §4.1 传输后端表 |
| `disaggregation_bootstrap_port` | `8998` | `python/sglang/srt/server_args.py:3172` | prefill 侧 bootstrap HTTP server 端口 |
| `disaggregation_ib_device` | `None` | `python/sglang/srt/server_args.py:3177` | IB 设备映射；`None` 时 mooncake 后端自动探测 |
| `disaggregation_decode_enable_radix_cache` | `False` | `python/sglang/srt/server_args.py:3182` | decode 侧开 radix cache 省重复传输 |
| `disaggregation_decode_enable_offload_kvcache` | `False` | `python/sglang/srt/server_args.py:3187` | decode 侧异步 KV offload（对应 `DecodeKVCacheOffloadManager`） |
| `disaggregation_decode_retraction_backup` | `None` | `python/sglang/srt/server_args.py:3192` | decode 请求被抢占重试时 KV 暂存位置：`cpu_tensor` 或 `host_pool`（复用 HiCache 保留池） |
| `num_reserved_decode_tokens` | `512` | `python/sglang/srt/server_args.py:3205` | 请求加入 running batch 时预留的 decode token 数 |
| `disaggregation_decode_extra_slots` | `None` | `python/sglang/srt/server_args.py:3210` | 给“在途传输”请求预留的额外 `req_to_token` 槽位 |
| `disaggregation_decode_polling_interval` | `1` | `python/sglang/srt/server_args.py:3215` | decode 侧轮询间隔，`>1` 降低轮询开销 |
| `optimistic_prefill_attempts` | `0` | `python/sglang/srt/server_args.py:3220` | 跳过 bootstrap 等待、乐观执行 prefill forward 的次数上限 |

`DISAGG_TRANSFER_BACKEND_CHOICES = ["mooncake", "nixl", "ascend", "fake", "mori", "mooncake_tcp"]`（`python/sglang/srt/server_args.py:247`-`254`）是 CLI 层允许的六个字符串，但 `TransferBackend` 枚举（`python/sglang/srt/disaggregation/utils.py:580`-`585`）只有五个成员——`mooncake_tcp` 不是独立后端，细节见 §7 第 3 条。

### 3.3 PD 请求生命周期的三个队列

- **`PrefillBootstrapQueue`**（`python/sglang/srt/disaggregation/prefill.py:119`）：等待 bootstrap 握手完成的请求，`pop_bootstrapped`（`python/sglang/srt/disaggregation/prefill.py:383`）按 `KVPoll` 状态分流——`Failed` 触发 `handle_bootstrap_failure`，`Bootstrapping` 在 `optimistic_prefill_attempts` 配额内提前放行（`python/sglang/srt/disaggregation/prefill.py:439`-`450`），`WaitingForInput` 调 `finalize_bootstrap`（`python/sglang/srt/disaggregation/prefill.py:336`）正式转入可调度队列。
- **`DecodePreallocQueue`**（`python/sglang/srt/disaggregation/decode.py:300`，混入 `DecodeHiCachePreallocMixin`）：decode 侧先按显存预算预分配 KV 槽位，再发起 receiver 握手；`pop_preallocated`（`python/sglang/srt/disaggregation/decode.py:1052`）把预分配成功的请求转给下一级队列。
- **`DecodeTransferQueue`**（`python/sglang/srt/disaggregation/decode.py:1991`，混入 `DecodeHiCacheTransferMixin`）：等 KV 真正传输完成；`pop_transferred`（`python/sglang/srt/disaggregation/decode.py:2223`）轮询完成后，`process_decode_queue`（`python/sglang/srt/disaggregation/decode.py:2643`）把结果 `extend` 进普通的 `self.waiting_queue`（`python/sglang/srt/disaggregation/decode.py:2677`）——从这一刻起，这个请求和一个从未经历过 PD 分离的普通请求在调度器眼里没有区别。

### 3.4 `hicache_*` 旋钮（`python/sglang/srt/server_args.py:2717` 起）

| 参数 | 默认值 | 行号 | 作用 |
|---|---|---|---|
| `hicache_host_memory_mode` | `"cache"` | `python/sglang/srt/server_args.py:2720` | `cache`：host 内存是持久 L2；`buffer_only`：只是 GPU↔L3 之间的临时中转，必须配 `hicache_storage_backend` |
| `hicache_ratio` | `None` | `python/sglang/srt/server_args.py:2728` | host 池 / device 池的容量比，缺省按模式取 2.0/1.2/0.2 |
| `hicache_size` | `0` | `python/sglang/srt/server_args.py:2733` | host 池绝对大小（GB），设置后覆盖 `hicache_ratio` |
| `hicache_write_policy` | `"write_through"` | `python/sglang/srt/server_args.py:2738` | `write_back`/`write_through`/`write_through_selective` 三选一 |
| `hicache_io_backend` | `"kernel"` | `python/sglang/srt/server_args.py:2746` | GPU↔CPU 传输走的 IO 路径：`direct`/`kernel`/`kernel_ascend` |
| `hicache_mem_layout` | `"page_first"` | `python/sglang/srt/server_args.py:2754` | host 内存池的物理布局，5 种可选 |
| `hicache_storage_backend` | `None` | `python/sglang/srt/server_args.py:2768` | `None` 即不启用 L3；否则见 §2.2 后端注册表 |
| `hicache_storage_prefetch_policy` | `"timeout"` | `python/sglang/srt/server_args.py:2788` | `best_effort`/`wait_complete`/`timeout` 三选一，见 §4.4 |
| `hicache_storage_backend_extra_config` | `None` | `python/sglang/srt/server_args.py:2796` | 给具体后端的 JSON 配置 |

`hicache_mem_layout` 的 5 个取值不是随意排列组合，`_resolve_layout_io_compatibility`（`python/sglang/srt/server_args.py:7569`）会在启动期做一次自动改写：`page_first` 配 `direct` IO 后端时会被静默改成 `page_first_direct` 并打 warning（`python/sglang/srt/server_args.py:7579`-`7586`），`page_first_direct` 配 `kernel` IO 后端时反过来把 IO 后端改成 `direct`（`python/sglang/srt/server_args.py:7571`-`7577`）——这与 [[04-SGLang-内存池与KV布局]] §5.4 讲过的“`page_size` 由 attention 后端反过来钉死”是**同一种工程模式**：布局与传输路径的合法组合不是靠文档约束用户，而是在参数解析阶段就自动纠正 + 打日志，把校验成本从“用户读文档”转移到“代码在启动时兜底”。`layer_first` 对应 §3.4（GPU 侧）已经讲过的“每层一份独立张量”的直觉搬到 host 侧；`page_first`/`page_first_direct` 则是把同一页里所有层的 KV 摆到一起（`server_args.py:5676`-`5680` 的注释提到"page_first"和"page_first_direct"都有专门的"split K/V transfer path"），对应 GPU 侧 `PageMajorMHATokenToKVPool` 那种“连续大 buffer”思路——**这组选择在 L2 层面重演了 04 篇在 L1 层面讲过的同一个权衡**：按层存取更简单、按页存取对批量 L1↔L2 搬运更友好。

### 3.5 `TreeNode` 上为 HiCache 准备的字段（继承自 [[03-SGLang-RadixAttention与前缀缓存]] 讲过的 `radix_cache.py`）

`host_value: Optional[torch.Tensor]`（`python/sglang/srt/mem_cache/radix_cache.py:256`）和 `host_ref_counter`（`python/sglang/srt/mem_cache/radix_cache.py:254`）是与 GPU 侧 `value`/`lock_ref` 完全独立的第二本账：`evicted`（`python/sglang/srt/mem_cache/radix_cache.py:269`）判定 GPU 上的 `value` 还在不在，`backuped`（`python/sglang/srt/mem_cache/radix_cache.py:273`-`274`，`return self.host_value is not None`）判定 CPU 上的备份还在不在，一个节点可以同时是“GPU 已驱逐、CPU 有备份”。`protect_host`/`release_host`（`python/sglang/srt/mem_cache/radix_cache.py:276`-`283`）是这份 CPU 备份专属的引用计数，独立于保护 GPU 值的 `lock_ref`。

### 3.6 `HiCacheStorage` 接口的 v1→v2 演进：从单池到多池

`HiCacheStorage(ABC)`（`python/sglang/srt/mem_cache/hicache_storage.py:150`）目前 v1、v2 两套读写接口并存：v1 的 `batch_get_v1`/`batch_set_v1`（`python/sglang/srt/mem_cache/hicache_storage.py:220`/`232`）签名很朴素——`keys: List[str]` 配一段 `host_indices: torch.Tensor`，一个 key 对应一段连续的 host 内存，这是“L3 只存标准 KV”这个假设下最简单的形状。v2 的 `batch_exists_v2`/`batch_get_v2`/`batch_set_v2`（`hicache_storage.py:165`/`198`/`209`）把参数换成了 `List[PoolTransfer]`，并且 `batch_exists_v2` 的 docstring（`hicache_storage.py:171`-`195`）明确写了它要处理“多个池子共同存在性”的问题：主 KV 池默认要求 `"all_pages"` 命中策略（前缀里每一页都必须存在，DSA 索引池就是这么严格），而 Mamba/SWA 这类"只覆盖前缀尾部"的辅助状态池可以用 `"trailing_pages"` 策略（只要最后几页存在就够，不要求从头连续）——**最终可用前缀长度取所有池子结果的最小值**，一个辅助池缺页会反过来缩短整个前缀的可用长度。

这条演进线和 §3.1 讲的 `KVArgs.state_types`（PD 传输协议里为 Mamba/SWA/DSA 各开一个 `StateType` 分支）是**同一个问题在两个子系统里各自的解法**：PD 传输协议用一个枚举字段区分"这块内存是 KV 还是某种额外状态"；HiCache L3 存储接口用"每个池子一条 `PoolTransfer`、外加命中策略"来表达同样的异构性——两边都是"标准 KV 场景先设计出来，混合模型的额外状态后补上去"的演进痕迹，只是补的方式不同（一个加枚举分支，一个加接口版本）。

### 3.7 PD×HiCache 交叉点：`DecodePrefixMatch`

```python
@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def restore_token_count(self) -> int:
        return self.decode_prefix_len - self.l1_prefix_len
```

（`python/sglang/srt/disaggregation/decode_hicache_mixin.py:24`-`47`）这个数据类把“decode 实例本地到底已经有多少这次请求需要的 KV”拆成三段：GPU 上现成的（`l1_prefix_len`）、CPU 上现成的（`l2_host_hit_length`）、远端存储上现成的（`l3_storage_hit_length`）——只有 `decode_prefix_len` 之后剩下的部分，才是真正需要通过 PD 网络从 prefill 侧拉过来的。`HiCacheRestoreResult`（`python/sglang/srt/disaggregation/decode_hicache_mixin.py:50`，`PENDING`/`READY`/`FAILED` 三态）是“本地 L2/L3 恢复”这个子状态机的状态，和 `KVPoll` 是两套独立但通过 `HiCacheRestoreGatedKVReceiver`（`python/sglang/srt/disaggregation/decode_hicache_mixin.py:154`）耦合在一起的状态——它包了一层 `poll()`，只有本地 HiCache 恢复也 `READY` 了，才会把底层 `KVPoll.Success` 真正放行给上层（`python/sglang/srt/disaggregation/decode_hicache_mixin.py:160`-`167`），否则伪装成 `KVPoll.Transferring` 继续等。

## 4. 主流程走读

### 4.1 一次 PD 请求的完整时序

```
[启动期]
CommonKVManager.register_to_bootstrap()                      common/conn.py:753
  → prefill 进程 HTTP PUT 到自己起的 bootstrap server（仅 prefill 侧起）
CommonKVBootstrapServer.__init__ 起 aiohttp app，注册 "/route" 路由   common/conn.py:1612, :1660
  监听端口 = disaggregation_bootstrap_port（默认 8998）

[请求到达 prefill]
1. 入队 PrefillBootstrapQueue.queue                           prefill.py:119
2. 每轮调度循环 pop_bootstrapped() 轮询 req.disagg_kv_sender.poll()  prefill.py:383
     KVPoll.Bootstrapping → 若在 optimistic_prefill_attempts 配额内，提前放行进 batch   prefill.py:439-450
     KVPoll.WaitingForInput → finalize_bootstrap() 正式转入可调度队列             prefill.py:336, 451-460
3. 正常 prefill forward 在 SchedulerDisaggregationPrefillMixin 的事件循环里跑完   prefill.py:485起
4. forward 结果处理后触发 send_kv_chunk(req, ...)（chunked/非 chunked 5 处调用点）  prefill.py:762/823/885/1077/1130 → 定义 :1132
     按 page_size 切片、组装 state_indices，转交底层 KVSender 发送
5. 已发送请求进 disagg_prefill_inflight_queue，process_disagg_prefill_inflight_queue 持续轮询  prefill.py:841
     WaitingForInput/Transferring → 继续等；Success → 解锁 radix 树节点、回收 sender

[decode 侧]
6. 请求先入 DecodePreallocQueue.queue，按显存预算预分配 KV 槽位后发起 CommonKVReceiver 握手  decode.py:300, common/conn.py:1327
7. pop_preallocated() 把预分配成功的请求转给下一级队列                          decode.py:1052
8. 转入 DecodeTransferQueue.queue，pop_transferred() 轮询 receiver.poll()        decode.py:1991, :2223
     KVPoll.Failed → 释放/中止；其余 → 继续等 Success
9. process_decode_queue() 把 transferred_reqs 塞进 self.waiting_queue           decode.py:2643, :2677
     —— 从这里开始就是普通调度器逻辑，请求进 running_batch 正常 decode forward
```

握手信号全靠 `KVPoll` 五态机，prefill/decode 两侧各自轮询自己的 sender/receiver，**没有第三方仲裁**——这与 §6 要对照的 vLLM 调度器直接调用 connector 方法的模式是两种不同的协作范式。

### 4.2 传输后端清单

| 后端 | 核心类（文件:行） | 传输介质 | 依赖的外部组件 |
|---|---|---|---|
| `mooncake` | `MooncakeKVManager` `python/sglang/srt/disaggregation/mooncake/conn.py:196` | RDMA，经 Mooncake TransferEngine | Mooncake 传输引擎（`get_mooncake_transfer_engine()`） |
| `nixl` | `NixlKVManager` `python/sglang/srt/disaggregation/nixl/conn.py:396`（`StagingManagerMixin+CommonKVManager`） | RDMA，经 UCX（NIXL 抽象） | `nixl` Python 包 + UCX，懒加载导入 |
| `ascend` | `AscendKVManager` `python/sglang/srt/disaggregation/ascend/conn.py:32`，**直接继承 `MooncakeKVManager`只换 TransferEngine** | 华为 NPU 侧 RDMA/HCCS | `memfabric_hybrid.TransferEngine`（`python/sglang/srt/disaggregation/ascend/transfer_engine.py:14`，未装则延迟到实例化时报错） |
| `mori` | `MoriKVManager(CommonKVManager)` `python/sglang/srt/disaggregation/mori/conn.py:302` | RDMA，AMD IO 引擎 | `mori.cpp.TransferStatus` + `mori.io.IOEngine` 系列 |
| `fake` | `FakeKVManager(BaseKVManager)` `python/sglang/srt/disaggregation/fake/conn.py:22` | 不传输，纯状态机空转 | 无——专供 warmup 请求 |
| `mooncake_tcp` | 复用 `MooncakeKVManager`，非独立类 | TCP（强制关闭 RDMA） | 见 §7 第 3 条 |

`common/conn.py`（1898 行）本身不出现在上表里，是因为它不传输任何字节——它是前四个后端（除 `fake`）共同复用的“非传输部分”：bootstrap 握手、房间号分配、注册表维护，`StagingManagerMixin`（`python/sglang/srt/disaggregation/common/staging_handler.py:827`）额外给 `nixl`/`mooncake` 提供了把分散头切片聚合成连续显存的暂存缓冲区（减少小颗粒 RDMA 请求数量）。

### 4.3 HiCache 准入与写入

`write_through_threshold` 在构造 `HiRadixCache` 时确定：

```python
self.write_through_threshold = (
    1 if server_args.hicache_write_policy == "write_through" else 2
)
```

（`python/sglang/srt/mem_cache/hiradix_cache.py:207`）三种策略的行为差异：

- **`write_through`**（默认）：命中 1 次即写——约等于“prefill 一算完，只要被复用过一次就落 L2”。
- **`write_through_selective`**：要求命中 ≥2 次才写——同一段 KV 被真正复用过一次才“够格”下沉，这是 admission 的核心判据是**复用次数**，不是 KV 大小或访问热度分数。
- **`write_back`**：从不主动写（`_inc_hit_count` 命中计数时对 `write_back` 策略直接跳过），只有在 GPU 侧驱逐、腾地方时才**顺带**把值得保留的节点搬到 host（§4.4）——这是“惰性准入”，靠驱逐压力反向触发写入。

实际写入函数 `write_backup`（`python/sglang/srt/mem_cache/hiradix_cache.py:841`）有一条不变式：write-through 模式下已备份节点必须从 root 开始连续，父节点还没备份，子节点直接跳过——保证 L2 里的前缀总是连续可复原的，不会出现“中间空洞”。写入时如果 host 侧分配失败，会现场触发 `evict_host` 腾地方再重试（`python/sglang/srt/mem_cache/hiradix_cache.py:855`-`858`）——这是 L2 驱逐的第一条触发路径，见下节。

### 4.4 HiCache 驱逐：GPU 侧与 Host 侧两条独立触发路径

`evict(params)`（`python/sglang/srt/mem_cache/hiradix_cache.py:1189`）按 `write_policy` 分叉：

```python
if self.cache_controller.write_policy == "write_back":
    num_evicted = self._evict_write_back(num_tokens)
else:
    num_evicted = self._evict_write_through(num_tokens)
```

- **`_evict_write_through`**（`python/sglang/srt/mem_cache/hiradix_cache.py:1213`）：只丢未备份的叶子、降级已备份的叶子（`_evict_backuped`/`_evict_regular`，`python/sglang/srt/mem_cache/hiradix_cache.py:1278`/`1284`），**驱逐过程本身不触发任何新的 L2 写入**（因为该备份的早在写入阶段就备份过了）。
- **`_evict_write_back`**（`python/sglang/srt/mem_cache/hiradix_cache.py:1231`）：遇到未备份的叶子会**现场**调 `write_backup(x, write_back=True)`（`python/sglang/srt/mem_cache/hiradix_cache.py:1254`）写一次 L2，写成功才算驱逐掉 GPU 空间；写失败（host 也满）就整棵子树硬丢（`_drop_subtree_no_host`，`python/sglang/srt/mem_cache/hiradix_cache.py:1294`，打 warning）。源码注释自己写了这条路径“将来会被弃用”（`python/sglang/srt/mem_cache/hiradix_cache.py:1233`）。
- **`evict_host(num_tokens)`**（`python/sglang/srt/mem_cache/hiradix_cache.py:1339`）是独立的 L2 驱逐，**不是从 `evict()` 内部主动调用的**，而是被动地在 `write_backup` 分配 host 空间失败时触发（§4.3 已提到的 `python/sglang/srt/mem_cache/hiradix_cache.py:855`-`858`）。

结论：**GPU 驱逐由 GPU 内存压力触发，Host 驱逐由 Host 内存压力（写入时分配失败）触发，二者各自独立、按各自资源池的水位反应，不存在全局统一的“先淘汰哪层”顺序**——这与 §5 要讨论的“分层设计是否需要一个全局协调者”直接相关。

### 4.5 预取：异步发起，轮询式协作调度

命中 L2/L3 时，SGLang **不做同步阻塞拉回**。后台线程 `HiCacheController.prefetch_thread_func`（`python/sglang/srt/managers/cache_controller.py:1128`）独立运行，`prefetch()`（`python/sglang/srt/managers/cache_controller.py:970`）发起一个 `PrefetchOperation`（`python/sglang/srt/managers/cache_controller.py:230`，继承 `StorageOperation`，`python/sglang/srt/managers/cache_controller.py:189`），完成后被塞进结果队列。

调度器侧是**非阻塞轮询**，不是阻塞等待：在 `waiting_queue` 遍历构建下一个 prefill batch 时（`python/sglang/srt/managers/scheduler.py:3386` 附近）：

```python
if self.enable_hicache_storage:
    prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
    if not prefetch_done:
        continue  # skip staging requests that are ongoing prefetch
```

预取没完成，这个请求就被跳过，调度器尝试队列里下一个请求；该请求本身留在 `waiting_queue` 里不变，等下一次调度循环 tick 再被问一遍——**没有专门的“等待预取”状态枚举**，请求在调度器眼里的可见状态是“还在等待队列里”。

“什么时候算完成”由 `hicache_storage_prefetch_policy` 决定，判据在 `can_terminate_prefetch`（`python/sglang/srt/mem_cache/hiradix_cache.py:1584`）：

- `best_effort`：直接放行（永远不等）——用当前已经拉到多少算多少，可能实际上等于白预取了一部分。
- `wait_complete`：必须等全部 token 到齐才放行——阻塞的是“这个请求何时能进入下一批”，不阻塞调度器主循环本身（其它请求仍在正常跑）。
- `timeout`（默认）：到齐或超时任一条件满足即放行，超时公式是纯线性插值：

```python
class PrefetchTimeoutConfig:
    base: float = 2.0          # 秒，固定开销
    per_ki_token: float = 0.1  # 秒/1024 token
    max: float = 30.0          # 秒，上限
```

（`python/sglang/srt/mem_cache/hicache_storage.py:50`-`55`），即 `timeout ≈ min(30.0, 2.0 + 0.1 × tokens/1024)`。

### 4.6 decode 侧主动下沉：`DecodeKVCacheOffloadManager`

§3.7 讲的是“decode 收请求前先查缓存”这个方向；反方向的交叉点是 `disaggregation_decode_enable_offload_kvcache=True` 时启用的 `DecodeKVCacheOffloadManager`（`python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py:34`）——它不是把 KV **拉进** decode 实例，而是把 decode 实例算出来的 KV **主动下沉**进 HiCache，用于请求被抢占重试（retraction）时不必整段丢弃。这个 manager 自己 `build_kv_host_pool`（`decode_kvcache_offload_manager.py:60`）建一份独立的 host 内存池，复用的是同一个 `HiCacheController` 机制，但物理上和 §2.2 表里 `HiRadixCache` 挂的那份 L2 池是分开的两块内存——**decode 侧的“备份用途”host 池，和 prefill/decode 共用的“常规 L2 缓存”host 池，是两个不同的分配单元**，只是复用同一套搬运代码。

下沉的粒度由 `offload_stride` 控制（`decode_kvcache_offload_manager.py:50`-`56`）：

```python
env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
if env_stride is None or env_stride <= 0:
    self.offload_stride = self.page_size
else:
    self.offload_stride = max(
        self.page_size, (env_stride // self.page_size) * self.page_size
    )
```

默认按 `page_size` 逐页下沉；环境变量 `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE` 可以把这个粒度调粗（但会被向下取整到 `page_size` 的整数倍）。这是一个典型的“批量摊薄固定开销 vs 丢失窗口变大”的旋钮：步幅越大，每次下沉操作的固定开销（发起一次 DMA/RPC）摊得越薄，但如果请求在还没走到下一个下沉边界时就被销毁（比如彻底失败而不是被抢占重试），这段还没来得及下沉的 KV 就白算了——步幅越大，这个窗口内的潜在浪费也越大。

## 5. 设计决策与代价

### 5.1 用轮询状态机而不是回调/事件驱动来做 PD 握手

**为什么这么设计**：`KVPoll` 的五个状态和 `poll()` 方法契约（`python/sglang/srt/disaggregation/base/conn.py:161`-`166`/`220`-`225`）让 prefill 和 decode 两个进程之间**除了状态机本身，不共享任何运行时对象**——两边可以是完全独立的进程、机器、甚至跑着不同的调度器 tick 频率，只要各自诚实地实现 `poll()`，调用方就永远知道该干什么。这对“两个独立部署的服务互相不知道对方内部实现”的场景是必要的解耦——不能假设对方能反向调用你的回调（网络分区、进程重启都可能让回调失效）。

**不这样会怎样**：如果换成回调/事件驱动（比如对方完成时主动 RPC 通知），调用方需要维护一个长期监听的 server 端点、处理乱序到达的通知、处理通知丢失的重试逻辑——这些复杂度全部会从“状态机内部”转移到“状态同步协议”上，且更难调试（轮询失败很容易看出“卡在哪个状态”，回调失败往往表现为“永远没收到通知”，更难定位）。

**什么时候可以不这样**：单机部署、prefill 和 decode 共享同一进程地址空间时——这正是 `disaggregation_mode="null"`（不开 PD 分离）时的默认路径，KV 直接留在本地 `TokenToKVPool` 里，连 `KVPoll` 都不会被实例化。

### 5.2 `common/conn.py` 吸收非传输逻辑，后端目录只放传输实现

**为什么这么设计**：bootstrap 握手、房间号分配、HTTP 注册这套协议本质上和“底层用 RDMA 还是用别的介质搬字节”无关——`CommonKVManager`（`python/sglang/srt/disaggregation/common/conn.py:142`）把这部分写一遍，`NixlKVManager`/`MooncakeKVManager`/`MoriKVManager` 继承它，只重写真正调用 NIXL/Mooncake/mori.io API 的那几个方法。`AscendKVManager` 甚至直接继承 `MooncakeKVManager`（`python/sglang/srt/disaggregation/ascend/conn.py:32`）只换掉底层 `TransferEngine`——说明“Mooncake 的握手协议”和“昇腾的传输原语”是可以完全解耦复用的两件事。

**不这样会怎样**：如果每个后端目录都从 `BaseKVManager` 直接实现一遍握手协议，1898 行的 `common/conn.py` 里的逻辑就要在 nixl/mooncake/ascend/mori 四个目录里各写一份——新增一个后端的门槛会从“实现传输原语”变成“实现传输原语 + 重新实现一遍握手协议”，且四份握手协议的行为很容易在演进中悄悄分叉（一个后端修了一个 bug，另外三个不知道）。

**什么时候可以不这样**：某个后端的传输语义和握手语义天然耦合到无法拆开时（比如某些硬件的传输原语本身就内置了地址协商），继承 `CommonKVManager` 反而是负担——但目前五个内置后端里没有一个走这条路，`fake` 后端甚至连 `BaseKVBootstrapServer` 都不需要（`python/sglang/srt/disaggregation/utils.py:690`-`695` 的类映射里没有它）。

### 5.3 HiCache 准入用命中次数阈值，不用带宽/收益模型

**为什么这么设计**：命中次数是一个**几乎零成本**的信号——`hit_count` 字段本来就要为 LFU/SLRU 驱逐策略维护（[[03-SGLang-RadixAttention与前缀缓存]] 已经讲过），复用它做准入判据不需要额外测量任何东西。相比之下，一个真正的“值不值得写”收益模型需要知道 PCIe 带宽、这段 KV 未来被复用的概率、复用时能省下多少次重算——这些量在写入那一刻大多是未知或高方差的，源码选择了“先用一个粗糙但便宜的代理指标，把精细化留给未来”（`python/sglang/srt/mem_cache/hiradix_cache.py:206` 那句 `# todo: dynamically adjust the threshold` 就是这个立场的直接证据）。

**不这样会怎样**：如果对每一次 prefill 结果都无条件写 L2（`write_through_threshold=0` 等价物），host 内存会被大量“只会被访问一次”的 KV 占满，挤出真正有复用价值的前缀，PCIe 带宽也会被这些一次性写入占用，反而拖慢正常请求的 L1↔L2 搬运。

**什么时候可以不这样**：离线批处理场景，如果能提前知道请求之间的前缀共享结构（比如一批请求本来就共享同一个 system prompt），完全可以绕过运行时启发式，在启动时就把这段共享前缀显式预热进 L2/L3——`hicache_storage_backend` 支持热挂载正是为这类场景留的接口（`PUT /hicache/storage-backend`，`python/sglang/srt/entrypoints/http_server.py:1078`）。

### 5.4 GPU 驱逐与 Host 驱逐各自独立触发，不设全局协调者

**为什么这么设计**：GPU 显存和 Host 内存是两个容量、带宽、失败模式都完全不同的资源池，各自的驱逐只需要对**自己的**水位负责——`_evict_write_through`/`_evict_write_back` 只关心 GPU 侧还能不能腾出请求需要的空间，`evict_host` 只关心 host 侧还能不能腾出 `write_backup` 需要的空间。把两者的触发条件强行统一（比如“host 快满了就先减慢 GPU 驱逐速度”）需要一个持续监控两边水位、做联合决策的组件，这个组件本身会成为新的单点和新的延迟来源。

**不这样会怎样**：当前设计下确实可能出现“GPU 驱逐正常运转，但 host 侧写入频繁因为空间不足而失败，只能反复触发 `evict_host`”这种局部抖动（§4.3 提到的按需驱逐路径）——`_evict_write_back` 遇到 host 也满的情况会直接整棵子树硬丢并打 warning（`python/sglang/srt/mem_cache/hiradix_cache.py:1294`-`1306`），这是当前架构下 host 内存压力的最终出口，不优雅但简单。

**什么时候可以不这样**：如果 host 内存长期处于紧张状态（`hicache_ratio` 设得太小），值得考虑在 `write_backup` 之前就用一个更悲观的准入门槛（比如提高 `write_through_selective` 的命中阈值）主动减少写入压力，而不是等分配失败了再被动 `evict_host`——这是 §8 的可改进点之一。

### 5.5 PD 分离本身值不值得开：解析计算

这是本篇任务要求的核心判据。把“该不该为这个请求走 PD 网络传输”写成一个显式的时间账本——**假设与口径先声明**：以下全部是**解析计算**（基于代码里已确认的公式结构做符号推导），不代入任何具体测得的带宽/算力数字，也不产出任何实测吞吐/延迟结论。

记：
- `P` = 需要跨网络传输的 token 数（对应 §3.7 的 `restore_token_count`，PD 分离场景下等于 prompt 长度减去 decode 侧本地已经命中的部分）；
- `cell_size` = 每 token 每层的 KV 字节数（[[04-SGLang-内存池与KV布局]] §4.1 已经给出精确公式：MHA 为 `n_kv_heads*(head_dim+v_head_dim)*num_layers*dtype_size`，MLA 为 `(kv_lora_rank+qk_rope_head_dim)*num_layers*dtype_size`）；
- `BW_net` = prefill/decode 两实例之间的有效传输带宽（RDMA 或 TCP，取决于 §4.2 选的后端）；
- `T_fixed` = 每个请求固定要付的握手/轮询开销（bootstrap round trip + `disaggregation_decode_polling_interval` 带来的取整延迟 + 元数据 RPC）；
- `t_prefill` = 该模型在该硬件上 prefill 阶段每 token 的计算耗时（近似线性于 token 数，因为 prefill 阶段矩阵乘法主导、是计算密集型的）。

则一次 PD 传输付出的额外开销：

```
T_transfer(P) ≈ P × cell_size / BW_net + T_fixed
```

这笔开销是**纯粹加在“monolithic（不分离）部署本来不需要付”的成本之上**——不分离时 KV 算完就在本地显存里，没有这一步。是否值得开 PD 分离，要看 `T_transfer(P)` 相对于这个请求总延迟（`P × t_prefill` 加上 decode 阶段耗时）的占比：

```
overhead_ratio(P) = T_transfer(P) / (P × t_prefill + T_decode)
```

**两个会让这个比值失控的边界情况**，直接对应任务要求的“什么时候不该开”：

1. **短 prompt**：`T_fixed` 与 `P` 无关，是常数；但分子里的 `P × t_prefill` 随 `P` 线性缩小。当 `P` 很小时（短对话、单轮问答、检索片段拼接类请求），`T_fixed` 可以轻易超过 `P × t_prefill` 本身——这时候 PD 分离带来的握手固定开销，比它想要节省下来的那部分 prefill 计算时间还要大。这正是源码里 `optimistic_prefill_attempts`（提前放行，跳过等待）和 `disaggregation_decode_polling_interval`（批量摊薄轮询次数）这两个旋钮存在的理由——它们是对“`T_fixed` 在短请求场景下占比过高”这个已知代价的显式补救，补救的存在本身就是代价存在的证据。
2. **`BW_net` 低（退化到 `mooncake_tcp` 或跨机房长距离网络）**：`P × cell_size / BW_net` 这一项随 `cell_size`（模型越大、KV 头越多，这个数越大）和 `P`（长 prompt）同时放大，当它的量级接近甚至超过 `P × t_prefill` 时，PD 分离把“prefill 计算换成网络传输”这笔交换整体上不再划算——尤其是当输出 token 数 `T_decode` 本身很短（比如摘要/分类类任务，答案很短）时，分母里能摊薄这笔固定+线性开销的“decode 阶段收益”也很有限。

**同一套 `cell_size` 公式也决定了 HiCache restore 是否划算**：把“从 L2/L3 拉回”类比成上面的 `T_transfer`，只是把 `BW_net` 换成 `BW_L2`（PCIe，量级上通常远高于跨机网络）或 `BW_L3`（远端存储/磁盘带宽，量级上通常低于 PCIe），把 `T_fixed` 换成一次 DMA/RPC 的固定开销。**结构完全一样，结论也一样**：极短的命中前缀（`P` 很小）时，固定开销可能超过重算这段前缀本身的计算时间——直接重算比“去 L2/L3 找一遍再搬回来”更快。当前源码里没有一处显式计算这个比值再决定要不要 restore（§4.3 的准入判据只看命中次数，§4.5 的预取终止策略只看时间/超时，都不是这个比值本身），这是 §8 的可改进点之一。

### 5.6 路由/负载均衡不放进 Python 调度进程，独立成一个 Rust 网关

**为什么这么设计**：`PDRouter`（`sgl-model-gateway/src/routers/http/pd_router.rs:50`）要做的事——维护全部 worker 的健康状态与一致性哈希环（`WorkerRegistry`/`HashRing`，`sgl-model-gateway/src/core/worker_registry.rs:43`/`180`）、对每个进来的 HTTP 请求做 `select_pd_pair`（`sgl-model-gateway/src/routers/http/pd_router.rs:972`）再 `execute_dual_dispatch`（`sgl-model-gateway/src/routers/http/pd_router.rs:365`）——是纯粹的**请求级、无状态、高频**的路径：它不需要碰 KV 张量、不需要读写 GPU 显存，只需要在极短时间内选出一对 worker 并转发字节流。这类工作负载正是 Rust 相对 Python 的强项（没有 GIL、原生异步 I/O、内存开销小），而 `Scheduler`（Python，见 [[02-SGLang-Scheduler事件循环]]）要处理的是需要直接操作 CUDA 张量、与模型前向紧密耦合的调度逻辑——把两者分进两个进程/两种语言，路由层的横向扩容（多开几个网关副本）和调度层的纵向优化（每个 GPU worker 专注自己的 batch）可以完全独立进行，互不掣肘。

**不这样会怎样**：如果把路由逻辑塞进某个 Python `Scheduler` 进程内部，这个进程就同时承担“面向外部世界的高频请求分发”和“面向 GPU 的低频重计算调度”两种截然不同的职责——前者的延迟敏感度是毫秒级，后者的一次 batch 组装/前向调用是几十到几百毫秒级，两者共享一个事件循环容易互相拖慢；而且这个 Python 进程会变成单点：它既要懂"HTTP 层怎么转发"又要懂"KV 传输状态机怎么轮询"，職责耦合导致独立扩容路由能力变得困难（想多开几个路由副本，会连带复制一份不必要的调度器状态）。

**什么时候可以不这样**：只有一对 prefill/decode 实例、不需要跨多组实例做负载均衡时（比如本地开发调试、单元测试），`Scheduler` 完全不需要经过任何路由层——直接用 `--disaggregation-mode` 分别起两个进程，客户端各自直连即可，`sgl-model-gateway` 这层在这种拓扑下是可选的，不是强制的。

## 6. 同位对照（vLLM 的 KVConnector 体系）

vLLM 把“KV 从哪来、传到哪去”这整件事收进**一个**抽象基类：`KVConnectorBase_V1`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:171` ``）。它把接口拆成调度器侧和 worker 侧两组方法，且深度嵌入到调度器的分配循环里——调度器在**分配 block 之前**就要调 `get_num_new_matched_tokens`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:450` ``）问 connector“这个请求外部缓存里有多少能用”，分配完之后调 `update_state_after_alloc`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:485` ``）告诉 connector“这些 block 归你了，可以开始异步加载”，每一步再调 `build_connector_meta`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:511` ``）把这一步要做的事打包传给 worker；worker 侧则有 `start_load_kv`/`save_kv_layer`/`get_finished`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:289`/`321`/`353` ``）这类按**层**粒度挂进模型前向的钩子。请求结束时 `request_finished`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:543` ``）决定 block 是立刻释放还是等一次异步保存完成再释放。

后端接入走一个纯字符串注册表：`KVConnectorFactory`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:27` ``）的 `register_connector`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:31` ``）把名字映射到“模块路径+类名”，`create_connector`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:43` ``）按需懒加载导入。截至本篇取证时，这个表里同时注册了 `NixlConnector`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:177` ``）、`MooncakeConnector`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:219` ``）、`HF3FSKVConnector`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:239` ``），**以及**一个 `OffloadingConnector`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:207` ``，实现类在 `` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:49` ``，`class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)`）。

**这是两边架构哲学的关键分歧点**：vLLM 把“PD 跨机传输”和“GPU→CPU 层级卸载”看成**同一类问题的两个实例**，都通过同一个 `KVConnectorBase_V1` 接口接入调度器——`OffloadingConnector` 和 `NixlConnector` 对调度器暴露的是完全相同的方法集合，调度器代码不需要知道自己在跟哪一种“外部 KV 来源”打交道。SGLang 则是**两套独立演化的子系统**：`disaggregation/` 用 `KVPoll` 轮询状态机对接调度器，`mem_cache/hiradix_cache.py` + `mem_cache/storage/` 用命中次数阈值+预取超时对接调度器，两者的调度器接入点（`prefill.py`/`decode.py` 的 Mixin vs `python/sglang/srt/managers/scheduler.py:3386` 的 `check_prefetch_progress`）在代码里是分开的调用路径，**只在 decode 侧靠 `decode_hicache_mixin.py` 这层专门的胶水代码手工打通**（§3.7）。

**代价对比**：vLLM 统一接口的代价是接口本身很厚——`KVConnectorBase_V1` 有十几个抽象/可覆盖方法，且深度耦合进调度器的 block 分配时序（`get_num_new_matched_tokens` 必须在分配前调用，返回值直接影响这一步分配多少 block），新写一个 connector 意味着要理解并正确实现这整套时序契约。SGLang 分目录的代价是**样板代码更少见**（`common/conn.py` 吸收了传输后端间的重复），但 PD 和 HiCache 是两条独立生长的分支，交叉处需要专门的 mixin 补丁（`DecodeHiCachePreallocMixin`/`DecodeHiCacheTransferMixin`）手工缝合，而不是像 vLLM 那样天然共享同一套调度器钩子——**统一抽象换来的是“新增后端的学习曲线陡”，分目录换来的是“系统级交叉点需要额外的胶水层”**，两者都不是免费的。

**一个更细的分歧点：谁负责处理“prefill 和 decode 的并行度不一样”**。真实部署里 prefill 实例和 decode 实例经常配不同的 TP size（prefill 算力密集，decode 显存密集，两边按各自资源特点独立配置并行度）。vLLM 把这件事收进一个**单独的、所有 offloading 后端共享**的模块：`canonical_mapping.py`（`` `vllm:vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py:1`-`9` ``）的模块 docstring 直接写明它是“整个 offload 栈里唯一处理并行度（TP/DCP/PCP）的地方，下游只消费字节映射”——把一个规范化的“canonical page”算好，后面所有存储后端都不用再关心 rank 怎么切分。SGLang 则是**每个传输后端各自处理一遍**：`NixlKVManager` 自己实现了 `_init_equal_tp_prep_handle`（`python/sglang/srt/disaggregation/nixl/conn.py:725`）、`_init_hetero_tp_prep_handle`（`python/sglang/srt/disaggregation/nixl/conn.py:760`）、`_init_mixed_equal_tp_prep_handles`（`python/sglang/srt/disaggregation/nixl/conn.py:917`）三套方法专门处理“prefill TP size 与 decode TP size 相等/不等/部分相等”这三种情况——Mooncake、ascend、mori 各自的 conn.py 里也有各自处理这个问题的代码，不共享 NIXL 那一份。**这是“统一抽象”和“分目录实现”这条分歧线在一个具体工程问题上的直接投影**：vLLM 选择在共享层一次性解决异构并行度，SGLang 选择让每个后端自己面对它——后者的好处是每个后端可以针对自己的传输原语做最贴合的优化（比如 NIXL 的 hetero-TP 路径可以利用它自己的内存注册机制），代价是同一个逻辑问题在多个文件里被解决了多次，修一个 bug 不保证另一个后端也修了。

## 7. 踩坑与反直觉

1. **两个不同的 `mori`**：PD 分离传输后端里的 `mori`（`python/sglang/srt/disaggregation/mori/conn.py:302` 的 `MoriKVManager`）是 AMD 的 `mori.io` RDMA 引擎；HiCache L3 存储后端注册表里也有一个叫 `"mori"` 的选项，但它实际指向 `storage/umbp/umbp_store.py` 的 `UMBPStore`（`python/sglang/srt/mem_cache/storage/backend_factory.py:241`-`245`）——**两个完全不同的库，只是同名**，读代码或读日志时如果不看具体导入路径，很容易把二者当成同一个东西。
2. **`get_transfer_engine_info`/`register_transfer_engine_info`/`init_weights_send_group_for_remote_instance` 这三个端点不是 PD-KV 传输的控制面**——它们属于“远程实例权重加载”功能（弹性扩缩容/热更权重场景），定义在 `python/sglang/srt/entrypoints/engine_info_bootstrap_server.py:51`-`52`/`:74`-`75` 和 `python/sglang/srt/entrypoints/http_server.py:1264`，传的是**模型权重**不是 KV cache（虽然它们也能选 `transfer_engine`/`nixl` 做搬运介质，`remote_instance_weight_loader_backend`，`python/sglang/srt/server_args.py:3338`）。真正 PD 握手用的端点是 bootstrap server 的 `/route`（`python/sglang/srt/disaggregation/common/conn.py:1660`），不经过 `entrypoints/http_server.py`。
3. **`mooncake_tcp` 不是一个独立的后端类**——`python/sglang/srt/arg_groups/pd_disaggregation_hook.py:21`-`24` 在参数归一化阶段把它改写：设置环境变量 `MC_FORCE_TCP=1`，把 `disaggregation_transfer_backend` 直接改成 `"mooncake"`，并清空 `disaggregation_ib_device`。`TransferBackend` 枚举（`python/sglang/srt/disaggregation/utils.py:580`-`585`）里从始至终只有 `MOONCAKE` 一个成员对应这两个 CLI 选项。
4. **`encoder/` 子目录不是 PD 的“第三种角色”，是另一条独立的轴**——`python/sglang/srt/server_args.py:3227`-`3268` 明确把它归在“Encode prefill disaggregation”（EPD，多模态编码器拆分）命名空间下，自成一套 bootstrap（`encoder_bootstrap_port` 默认 `8997`，`python/sglang/srt/server_args.py:3254`，和 PD 的 `disaggregation_bootstrap_port` 默认 `8998` 是两个不同端口），不复用 `PrefillBootstrapQueue`/`DecodePreallocQueue` 这套机制。
5. **`hisparse` 与 `hicache` 是同名不同物**——`enable_hisparse` 控制的是“hierarchical **sparse attention**”（DeepSeek DSA/MiniMax 一类稀疏索引结构的层级），走独立的 `mem_cache/allocator/hisparse.py`/`hisparse_memory_pool.py` 和专用 kernel，与本篇讲的 `HiCacheStorage`/`HostKVCache` 完全不共用代码路径——两个“hierarchical”指的是不同维度的“层级”。
6. **写入阈值目前是写死的常数**——`write_through_threshold` 只有 `1` 和 `2`两个可能值（`python/sglang/srt/mem_cache/hiradix_cache.py:207`），源码自己留了 `# todo: dynamically adjust the threshold`（`python/sglang/srt/mem_cache/hiradix_cache.py:206`）的注释，说明这不是一个经过充分调优的自适应机制，是一个先跑起来的粗糙代理。
7. **`_evict_write_back` 这条驱逐路径源码自己标注了将被弃用**（`python/sglang/srt/mem_cache/hiradix_cache.py:1233`）——读代码时如果看到这条路径的实现细节和 `_evict_write_through` 明显不对称（前者会现场触发写入，后者不会），不要当成两条对称设计的分支来理解，它们目前处于一个过渡态。
8. **不要只在 `python/sglang/srt/disaggregation/` 里找“路由/负载均衡”，那里没有**——Python 侧确实没有 `mini_lb`/`router` 之类的文件，真正做“给这个请求选哪一对 prefill/decode worker”的是独立的 Rust crate `sgl-model-gateway`（§1、§5.6）。如果只读 Python 代码库就下结论“SGLang 的 PD 分离没有中心路由，靠客户端自己选”，是不准确的——路由确实存在，只是不在你以为的那个目录、那个语言里。

## 8. 可改进点

1. **`write_through_threshold` 可以按 §5.5 的公式结构做成自适应**——源码自己承认目前是写死的 `1`/`2`（`python/sglang/srt/mem_cache/hiradix_cache.py:206`）。**本库推断**：一个成本可控的中间方案是把命中次数阈值和“这段前缀的 token 数”联合起来判断（短前缀即使命中一次也不值得写，因为重算比 PCIe 往返还快），而不是对所有长度的前缀用同一个命中次数门槛——这正是 §5.5 解析计算想说明的：admission 判据目前完全没有用到 `cell_size`/带宽这类已经在代码别处存在的量。
2. **`hicache_storage_prefetch_policy=best_effort` 缺一个反馈机制**——当前 `can_terminate_prefetch` 在 `best_effort` 下永远立即放行（`python/sglang/srt/mem_cache/hiradix_cache.py:1587`-`1588`），如果某次预取因为放行太早而“白预取”（数据还没到就已经决定不等了），这次浪费不会反馈回策略选择，下一次同样场景还是立即放行。**本库推断**：可以在 `PrefetchOperation` 里记一个“放行时实际完成比例”的滑动统计，供后续动态在 `best_effort`/`timeout` 之间切换，但这需要改动 `cache_controller.py` 的核心统计结构，具体实现成本本篇未评估。
3. **GPU 驱逐与 Host 驱逐的独立触发（§5.4）在 host 长期紧张时会退化成“反复分配失败→反复 `evict_host`”的抖动**——`write_backup` 分配失败时现场 `evict_host` 再重试（`python/sglang/srt/mem_cache/hiradix_cache.py:855`-`858`）这条路径没有对“最近是否刚驱逐过”做任何节流。**本库推断**：加一个简单的冷却窗口（比如“刚触发过 `evict_host` 的若干毫秒内，同一层级的新写入请求直接走 `write_through` 的降级路径而不是立刻再触发一次驱逐”）可以减少这种抖动，但会引入新的旋钮和新的边界情况，属于长期方向而非可以立刻提 PR 的改动。
4. **PD 传输后端处理“prefill/decode TP size 不一致”这件事目前每个后端各写一遍**（§6 与 vLLM `canonical_mapping.py` 的对照）——`NixlKVManager` 的 `_init_equal_tp_prep_handle`/`_init_hetero_tp_prep_handle`/`_init_mixed_equal_tp_prep_handles`（`python/sglang/srt/disaggregation/nixl/conn.py:725`/`760`/`917`）这套逻辑，Mooncake/ascend/mori 各自的 conn.py 里也要重新面对一次。**本库推断**：把“给定 prefill TP size 和 decode TP size，一段 KV 的物理槽位该怎么切分给对应的 decode rank”这个纯几何计算抽成一个后端无关的公共函数（放进 `common/conn.py` 或 `common/utils.py`），能减少这类 TP 映射 bug 在多个后端里各自出现一次的风险，但需要先确认几个后端目前的实现细节是否真的等价（本篇未逐行核对这一点，具体是否能直接抽取需要专门核对）。

## 9. 自测题与延伸阅读

**自测题（闭卷）**：

1. `KVPoll` 有哪五个状态？prefill 侧和 decode 侧分别在轮询谁的 `poll()`？谁负责最终仲裁两边状态是否一致？
2. `common/conn.py` 里的 `CommonKVManager` 承担了什么职责？`AscendKVManager` 为什么可以直接继承 `MooncakeKVManager` 而不是从 `BaseKVManager` 重新实现？
3. `hicache_write_policy` 的三个取值分别对应什么样的准入行为？`write_back` 策略下 KV 是在什么时机被写进 L2 的？
4. HiCache 的 GPU 侧驱逐和 Host 侧驱逐分别由什么触发？两者之间有没有全局协调？
5. `hicache_storage_prefetch_policy` 的三个取值（`best_effort`/`wait_complete`/`timeout`）分别在什么条件下认为预取“可以结束等待”？
6. `DecodePrefixMatch` 里的 `restore_token_count` 是怎么算出来的？它和这次请求真正需要走 PD 网络传输的 token 数是什么关系？
7. 为什么短 prompt 场景下开 PD 分离可能得不偿失？这个论证依赖哪两个量的相对大小？
8. 给一个请求选“哪一对 prefill/decode worker”这件事，在 SGLang 里发生在哪个进程/哪种语言写的代码里？它和 `KVPoll` 握手是不是同一层？
9. `HiCacheStorage` 的 v1 接口和 v2 接口在参数形状上最大的区别是什么？v2 引入的 `"trailing_pages"` 命中策略是为哪类辅助状态池准备的？

**延伸阅读（本库双链）**：

- [[03-SGLang-RadixAttention与前缀缓存]]——本篇 §3.5 用到的 `host_value`/`host_ref_counter`/`evicted`/`backuped` 字段和 `evict_policy.py` 的 7 种策略，完整机制在那一篇讲。
- [[04-SGLang-内存池与KV布局]]——本篇 §5.5 解析计算直接复用了那一篇给出的 `cell_size` 公式，L1 的物理张量布局也是那一篇的精读对象。
- [[09-SGLang-ServerArgs旋钮全景]]——`disagg`/`memory` 两个命名空间在全局 476 个旋钮里的位置，以及反射式 CLI 生成机制如何让这类专属旋钮低成本增长。
- [[12-vLLM-PD分离与KV-Connector]]——§6 同位对照只覆盖了 `KVConnectorBase_V1` 的接口形状，vLLM 侧调度器如何逐步驱动 `get_num_new_matched_tokens`→`update_state_after_alloc`→`build_connector_meta` 这条时序的完整逐行核对，留给那一篇（本篇写作时尚未成文，属于本库名册内的前向引用）。
