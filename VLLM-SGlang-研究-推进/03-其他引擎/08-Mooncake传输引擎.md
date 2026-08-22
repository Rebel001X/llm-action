# 08 · Mooncake 传输引擎

> **本篇取证基准**：`mooncake` @ `3b5a5941`（2026-08-22）
> **一句话**：它不是推理引擎，是被 vLLM/SGLang 当库直接链接的 KV 搬运工。

## 0. 结论先行

本库另外 11 篇讲的都是"一个请求怎么被调度、算完、吐 token"的推理引擎。这一篇不是。

先把本篇几条最不直观的发现摆在最前面，省得翻到后面才发现值得记的东西：

- Mooncake 的 Python 只有 63,986 行、只是绑定与测试，产品代码的主体是 548,895 行里的 C++/头文件（§0）。
- `_lab/api_surface.py` 抽出 0 条 HTTP 路由是脚本作者**故意**为它留的空配置，不是扫描失败（§0、§7）。
- 全仓库单目录代码量最大的 `mooncake-transfer-engine/tent`（85,136 行）不是核心业务代码，是一套默认关闭、需要环境变量才启用的备用实现，里面甚至藏着一条实验性的 Google TPU 支持路径（§7）。
- `_lab/struct_map.py` 报出的"disagg 子系统 268,180 行、命中 100% 文件"是一个工具假阳性，根源是关键词表里的 `"mooncake"` 撞上了这个仓库自己的目录命名（§7）。
- vLLM 和 SGLang 两个独立团队写出的 Mooncake 封装代码，连 `"P2PHANDSHAKE"` 这个字面量参数都一模一样（§4、§6）。

Mooncake（`github.com/kvcache-ai/Mooncake`，Moonshot AI / Kimi 的底座）解决的是一个更窄、更底层的问题：**一段显存里的字节，怎么又快又对地搬到另一台机器的显存或磁盘里**。它把这件事拆成两层，做成了两个可以分开使用的 C++ 库：

- **`mooncake-transfer-engine`**：点对点批量搬运库。不知道"KV cache"是什么概念，只认"从这段地址搬 N 字节到那个 segment 的某个 offset"，通过 RDMA/TCP/NVLink/NVMe-oF 等近 20 种传输介质之一去做。
- **`mooncake-store`**：在 Transfer Engine 之上加了一层"分布式 KV 对象存储"的语义——key、多副本、租约、驱逐——但它自己不做字节搬运，真正搬数据时还是调用内部持有的 Transfer Engine 实例。

`_lab/out/repo_stats.json` 里 `engines.mooncake.top_dirs` 按行数排出的前五个目录，摆在这里先给一个整体的体量感（数字直接取自该 JSON，不是本篇心算）：

| 目录 | 文件数 | 行数 |
|---|---:|---:|
| `mooncake-transfer-engine/tent` | 219 | 85,136 |
| `mooncake-store/tests` | 140 | 82,187 |
| `mooncake-store/src` | 115 | 78,393 |
| `mooncake-transfer-engine/src` | 75 | 47,138 |
| `mooncake-store/include` | 164 | 45,527 |

第一名 `tent` 是什么、为什么它比 `mooncake-store/src`（真正的 Store 业务代码）还大，留到 §7 详细说——先剧透一句：它不是"传输引擎的核心业务代码"，而是一套默认不启用的备用实现。

对外接口上有一件事值得先说清楚：用 `_lab/api_surface.py`（本库统一的 HTTP 路由抽取脚本）扫 Mooncake，抽到的路由数是 **0**。这不是漏报——脚本作者在 `_lab/api_surface.py` 的 `ENTRY_HINTS["mooncake"]` 一项就为 Mooncake 单独留了一条空的配置，注释写明"Mooncake 是传输引擎/KV 存储，不是 HTTP 推理服务；这里保留空 pattern，抽到 0 条是事实而不是 bug"。Mooncake 真正的对外接口是两个东西：**C++ 头文件里的类**（`mooncake-transfer-engine/include/transfer_engine.h`、`mooncake-store/include/master_service.h`）和**给它们包了一层 pybind11 的 Python 绑定**（`mooncake-integration/`）。vLLM 和 SGLang 都是直接 `import` 这层绑定、把它的类当本地对象用，而不是发 HTTP 请求。

**为什么这件事需要一个专门的传输引擎，而不是随手用 `socket.send()`——用一个解析计算（非实测，只是数量级估计）说明白量级**：一个 70B 量级、分组查询注意力（GQA，8 个 KV head）、隐藏维度 8192、80 层、FP16 的模型，每 token 每层的 KV cache 是 `2（K+V）× 8（kv_head）× 128（head_dim，按 hidden/head_num=128 估）× 2 字节 ＝ 4KB`；80 层合计每 token 320KB；一个 4096 token 的 prefill 请求，KV cache 总量约 `4096 × 320KB ≈ 1.3GB`。PD 分离场景下，这 1.3GB 要在 prefill 完成的那一刻整体搬到 decode 实例——这是**每个请求**要过一次网线的数据量，不是累计值。以上数字全部基于**假设的模型规格**做算术推导，不是任何具体模型或硬件的实测结果，只用来说明"为什么这不是一个简单的 socket 调用能扛住的搬运量"。

规模上一个容易读反的地方：Mooncake 的 Python 只有 63,986 行（`_lab/out/repo_stats.json` `engines.mooncake.languages`），而且这些 Python 绝大部分是绑定胶水代码和测试，不是产品逻辑；产品逻辑的主体是 C++（301,147 行）加 C/C++ 头文件（119,480 行）。这篇要回答的核心问题就是标题那句话：**当"把 KV 搬来搬去"这件事被单独抽成一个项目时，它长什么样、被集成的两家怎么用它、这样拆值不值。**

**术语速查**（后面几节会反复用到，先集中放在这里，不打断阅读）：

| 术语 | 在本篇里指什么 |
|---|---|
| RDMA | Remote Direct Memory Access，网卡直接读写远端内存，不经过对端 CPU 拷贝 |
| HCA | Host Channel Adapter，即支持 RDMA 的网卡（IB/RoCE） |
| `lkey`/`rkey` | RDMA 内存注册后拿到的本地/远端访问钥匙，§3 `BufferDesc` 里的字段 |
| NUMA | Non-Uniform Memory Access，CPU/内存/网卡分组后组间访问延迟不均，§2 `Topology` 依据它选网卡 |
| pybind11 | 把 C++ 类/函数暴露成 Python 模块的绑定库，§2"Python 绑定入口"一节的技术基础 |
| `coro_rpc` | yalantinglibs（`ylt`）提供的协程 RPC 框架，master 的控制面用它，不是 HTTP（§2/§7） |
| PD 分离 | Prefill-Decode disaggregation，prefill 和 decode 跑在不同实例上，中间靠搬 KV cache 衔接 |
| HiCache | SGLang 的分层 KV 缓存机制，Mooncake Store 是它的 L3（远端）存储后端之一（§6） |

## 1. 它在系统里的位置

先摆一个必须先破的误解：Mooncake **不是**一个 vLLM/SGLang 之外独立运行、被网络请求调用的服务。它是被 `pip install mooncake-transfer-engine` 之后**编译进/链接进**推理进程的一个库，它的对象（`TransferEngine`、`MooncakeDistributedStore`）活在调用方自己的 Python 进程里。README.md 开头写明这是 "Moonshot AI" 旗下 Kimi 的服务底座，并引用了 FAST'25 论文《Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving》（文档所述，未做二次核验，仅作背景引用）；这篇论文原本讲的是 PD 分离，但代码演进到今天已经不止服务 PD 分离——README 的更新日志里还提到 SGLang 用同一个 `TransferEngine` 做**大规模分布式 RL 的权重传输**（"7x faster weight updates for the 1T-parameter Kimi-K2 model"，文档所述），这一点在下面 §5 会用来论证"独立出来"这件事的价值。

两大组件在系统里的分层关系：

```
                     调用方进程（vLLM / SGLang worker 进程内）
┌──────────────────────────────────────────────────────────────┐
│  mooncake-store（可选）：分布式 KV 对象存储语义                 │
│    MasterService（控制面，coro_rpc，管 key→副本位置/租约/驱逐）  │
│    RealClient（持有一个 TransferEngine 实例做实际拉取/写入）     │
├──────────────────────────────────────────────────────────────┤
│  mooncake-transfer-engine（必选的底座）：点对点批量搬运          │
│    TransferEngine → Transport（RDMA/TCP/NVLink/NVMe-oF/...）    │
└──────────────────────────────────────────────────────────────┘
```

`mooncake-store` 的客户端代码在构造时直接把一个 `std::shared_ptr<TransferEngine>` 作为依赖注入进来（`mooncake-store/src/real_client.cpp:759`），说明 Store 没有重新发明一套数据搬运逻辑，它只是给 Transfer Engine 包了一层"这段数据放哪个副本、谁能读、什么时候过期"的元数据服务。这决定了本篇的读法：先读 Transfer Engine（真正干活的那层），再读 Store（管账的那层），最后看两家推理引擎分别接的是哪一层、甚至两层都接。

在本库里，这一篇和 [[12-vLLM-PD分离与KV-Connector]]、[[11-SGLang-PD分离与HiCache分层]] 是同一个话题的两个视角：那两篇讲"vLLM/SGLang 自己怎么定义 PD 分离与分层缓存的抽象"，这一篇讲"当它们需要真的把字节送过网线时，接的是哪个第三方库、这个库自己长什么样"。

`mooncake-store` 的存储语义也不止"内存里的一份 KV"。`ReplicateConfig`（`mooncake-store/include/replica.h:104`）里同时有 `replica_num`（内存副本数）、`nof_replica_num`（NVMe-oF 挂载的 SSD 副本数）、`dfs_replica_num`（分布式文件系统副本数）三个独立计数，外加 `with_hard_pin`（硬钉住，禁止被驱逐，区别于 `soft_pin_action` 那种"软钉住+可配置 TTL"）。对应的落盘代码分别是 `mooncake-store/src/nvme_kv_backend.cpp`（SPDK 驱动 NVMe SSD）和 `mooncake-store/src/hf3fs/hf3fs_file.cpp`（对接一个外部分布式文件系统的 USRBIO 数据面适配器）。后者的 README 开头就自报家门是"experimental"、需要显式设置 `MOONCAKE_DFS_FS_ADAPTER=hf3fs` 才会启用，且**不**被 Store 通常的容错/高可用/持久性保证覆盖（`mooncake-store/src/hf3fs/README.md` 原话，文档所述）——这提醒读者：本篇聚焦的"内存态 KV 搬运"只是 Store 已落地能力的核心层，磁盘/DFS 分层是往外延伸的第二圈，成熟度不在同一水平线上，本篇不展开细节。

repo 里还有 `mooncake-ep`、`mooncake-p2p-store`、`mooncake-pg`、`mooncake-reshard`、`mooncake-rl` 几个兄弟子项目（各自目录下多数没有 README，`mooncake-reshard/README.md` 提到它做"framework-neutral 的模型权重清单与逻辑规划"，文档所述），说明 Mooncake 这个仓库本身也在往"传输引擎之上的更多分布式原语"扩张，但那些不在 vLLM/SGLang 当前的 PD 分离/HiCache 接入路径上，超出本篇范围，不做展开。

## 2. 代码地图（文件 → 职责，带行号）

**Transfer Engine（点对点搬运）**

`mooncake-transfer-engine/src/transport/` 下按目录数了一遍每个后端的 `.cpp`/`.cu` 行数（直接 `wc -l`，不经过 `_lab` 工具），目标硬件里"确认"是指协议名/头文件里的行业通用术语能直接对上，"推断"是本篇没有逐行读实现、只从文件名/头文件包含关系做的判断：

| 目录 | 文件/行数 | 目标硬件或协议 |
|---|---|---|
| `rdma_transport/` | 5 / 5,896 | 通用 RDMA（InfiniBand/RoCE），默认首选后端（确认） |
| `ascend_transport/` | 14 / 6,261 | 华为昇腾 NPU（确认，`installTransport("ascend", ...)`） |
| `device/` | 5 / 3,081 | CUDA/MUSA/MACA 的设备端 P2P + IBGDA 路径（确认，见 `mooncake-transfer-engine/include/transfer_engine.h:40-44` 的 `device::P2pTransport`/`device::RdmaTransport` 前置声明） |
| `efa_transport/` | 3 / 2,889 | AWS EFA（确认） |
| `kunpeng_transport/` | 5 / 2,694 | 头文件名为 `ub_context.h`/`ub_endpoint.h`，大概率对应"ub"协议（华为鲲鹏平台的 Unified Bus 机间互联，推断） |
| `cxi_transport/` | 3 / 2,182 | HPE Slingshot CXI（确认） |
| `barex_transport/` | 2 / 1,736 | 头文件直接 `#include <infiniband/verbs.h>`，是另一条基于 ibverbs 的路径，与 `rdma_transport/` 的具体差异本篇未查证 |
| `nccl_transport/` | 1 / 1,287 | NCCL 宿主传输（确认，`USE_NCCL_HOST` 宏门控，与其他后端互斥安装，见 `mooncake-transfer-engine/src/multi_transport.cpp:409-414`） |
| `nvlink_transport/` | 1 / 1,284 | NVIDIA NVLink（确认） |
| `hip_transport/` | 3 / 1,161 | AMD ROCm/HIP（确认） |
| `rpc_communicator/` | 2 / 1,022 | 同样基于 `ylt::coro_rpc`（与 mooncake-store 的 master RPC 是同一个框架），推断是一条纯 RPC、不依赖专用网卡的软件传输路径 |
| `maca_transport/` | 1 / 961 | 沐曦 MACA（确认协议名，硬件细节未查证） |
| `intranode_nvlink_transport/` | 1 / 804 | 单机内 NVLink（与 `nvlink_transport/` 是两个独立目录，推断是跨机 vs 同机两种拓扑的实现拆分） |
| `sunrise_link/` | 1 / 764 | 协议全称未查证 |
| `tcp_transport/` | 1 / 748 | TCP，无专用硬件时的通用兜底（确认，见 §5 决策五） |
| `nvmeof_transport/` | 3 / 767 | NVMe-oF（确认，`cufile_desc_pool.cpp` 提示走的是 NVIDIA GPUDirect Storage 的 cuFile 接口） |
| `musa_transport/` | 1 / 327 | 摩尔线程 MUSA（确认协议名，硬件细节未查证） |
| `cxl_transport/` | 1 / 403 | CXL 内存扩展（确认，需配合 `MC_CXL_DEV_PATH` 环境变量，见 §2 下方 Store 小节） |
| `rdma_twosided/` | 2 / 445 | 双端协作模式的 RDMA 变体，与默认单边 RDMA 路径的差异未查证 |

这张表本身也印证了 §5 决策四的说法：19 个目录里"确认"能对上号的有 13 个，覆盖 InfiniBand/RoCE、NVIDIA（NVLink 跨机与同机两条独立路径、NVMe-oF/GPUDirect Storage）、AMD、华为昇腾、AWS、HPE 超算互联；剩下几个（`kunpeng_transport`/`barex_transport`/`rpc_communicator`/`sunrise_link`/`rdma_twosided`）本篇标注了推断依据但未逐行核实，这正是"可插拔接口"带来的一个副作用——外部读者不用理解每一种硬件的细节，就能从目录结构直接读出"这个项目支持的硬件面有多宽"，代价是要把每个目录都点开才能确认它具体做什么。

- `mooncake-transfer-engine/include/transfer_engine.h:70` —— `class TransferEngine`，最外层门面类，用户/绑定层直接拿到的对象。
- `mooncake-transfer-engine/include/transfer_engine.h:102` —— `int init(...)`，指定 metadata 连接串、本机地址、RPC 端口，内部据此决定装哪些 Transport。
- `mooncake-transfer-engine/include/transfer_engine.h:109` —— `Transport* installTransport(const std::string& proto, void** args)`，按协议字符串装载一个具体传输后端（"rdma"/"tcp"/"nvlink"/...）。
- `mooncake-transfer-engine/include/transfer_engine.h:127` —— `int registerLocalMemory(...)`，把一段本地内存注册成可被远端访问的 buffer（RDMA 场景下就是 `ibv_reg_mr`）。
- `mooncake-transfer-engine/include/transfer_engine.h:134` —— `Status submitTransfer(BatchID, const std::vector<TransferRequest>&)`，提交一批传输任务。
- `mooncake-transfer-engine/include/transfer_engine.h:211` —— `BatchID allocateBatchID(size_t batch_size)`，先分配一个批次句柄再往里提交请求。
- `mooncake-transfer-engine/include/transfer_engine.h:225` —— `Status getTransferStatus(BatchID, size_t task_id, TransferStatus&)`，轮询单个任务的完成状态。
- `mooncake-transfer-engine/include/transfer_engine.h:189`、`:197` —— `mp_registerLocalMemory`/`mp_submitTransfer`（`ENABLE_MULTI_PROTOCOL` 编译宏门控），允许同一段内存同时按 CXL/TCP/RDMA 等多种协议注册和传输，是比 §5 决策四"每次传输选一种协议"更进一步的"多协议并存"模式，本篇未深入这条路径的调度细节。
- `mooncake-transfer-engine/include/transport/transport.h:44` —— `class Transport`，所有传输后端必须实现的抽象基类。
- `mooncake-transfer-engine/include/transport/transport.h:477` —— `virtual const char* getName() const = 0`，纯虚函数之一，逼每个后端自报协议名。
- `mooncake-transfer-engine/src/multi_transport.cpp:406` —— `MultiTransport::installTransport(proto, topo)` 的实现：一串按编译宏门控的 `if (proto == "rdma") transport = new RdmaTransport(); else if (proto == "tcp") ... else if (proto == "nvmeof") ...`（`:417-437`），是"协议字符串 → 具体 Transport 子类"这张映射表的真实落点。
- `mooncake-transfer-engine/include/transfer_metadata.h:56` —— `struct BufferDesc`，一段已注册内存的元数据：地址、长度、RDMA `lkey`/`rkey`。
- `mooncake-transfer-engine/include/transfer_metadata.h:107` —— `struct SegmentDesc`，一整个远端 segment 的描述：设备列表、拓扑、`buffers`（`BufferDesc` 数组）。
- `mooncake-transfer-engine/include/transport/rdma_transport/rdma_transport.h:42` —— `class RdmaTransport : public Transport`，RDMA 后端对 `Transport` 接口的具体实现。
- `mooncake-transfer-engine/include/transport/rdma_transport/worker_pool.h:26` —— `class WorkerPool`，从队列里取 `TransferTask` 并真正发起 RDMA verbs 调用的工作线程池，实现在 `mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp`（1,469 行）。
- `mooncake-transfer-engine/include/transport/rdma_transport/rdma_context.h:106` —— `class RdmaContext`，每张网卡一个的上下文（PD/CQ 管理），实现在 `mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp`（1,487 行）。
- `mooncake-transfer-engine/include/transport/rdma_transport/rdma_endpoint.h:44` —— `class RdmaEndPoint`，每条 QP 连接对应一个实例，实现在 `mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp`（1,300 行）。
- `mooncake-transfer-engine/include/topology.h:64` —— `class Topology`，NUMA/PCIe 拓扑发现与 NIC 优先级矩阵。
- `mooncake-transfer-engine/src/topology.cpp:761` —— `Topology::selectDevice(...)` 的实现，按拓扑优先级+随机/轮询选一张网卡。
- `mooncake-transfer-engine/src/transfer_engine_impl.cpp:209` —— `init()` 内的协议选择逻辑起点：`MC_FORCE_TCP` 环境变量可以跳过所有硬件探测强制走 TCP。
- `mooncake-transfer-engine/src/transfer_engine_impl.cpp:227`、`:249`、`:298`、`:390`、`:447` —— 同一个 `init()` 函数体内按编译宏/环境变量依次尝试 `installTransport("ascend", ...)`、`("cxl", ...)`、`("rdma", ...)`（给 MACA 平台的 host 传输）、`("tcp", ...)`（无网卡时的兜底）、`("hip", ...)`（AMD），是"运行时按硬件自动选后端"的具体落点。
- `mooncake-transfer-engine/src/topology.cpp:524` —— `static bool isSameNumaNode(...)`，读 `/sys/bus/pci/devices/<bus>/numa_node`（`:528`）比较两个 PCI 设备是否同 NUMA 节点，是"拓扑感知"选网卡的判据来源。
- `mooncake-store/include/replica.h:104` —— `struct ReplicateConfig`，`replica_num`/`nof_replica_num`/`dfs_replica_num` 三级副本计数 + `with_hard_pin` 硬钉住标志，定义了一个 key 要在内存/SSD/DFS 各留几份、能不能被驱逐。

**Mooncake Store（分布式 KV 存储）**

- `mooncake-store/include/master_service.h:126` —— `class MasterService`，控制面的核心类：管段（segment）挂载、key 元数据、租约、驱逐。
- `mooncake-store/include/master_service.h:200` —— `MountSegment(const Segment&, const UUID&)`，客户端把一段本地显存/内存注册为可被写入的存储段。
- `mooncake-store/include/master_service.h:413` —— `GetReplicaList(const std::string& key, const TenantId&)`，读路径入口：查 key 对应的副本位置并授予租约。
- `mooncake-store/include/master_service.h:449` —— `PutStart(...)`，写路径的第一阶段：按大小分配占位副本槽位，返回给客户端去写。
- `mooncake-store/include/master_service.h:460` —— `PutEnd(...)`，写路径的第二阶段：客户端确认数据已写完，元数据才真正生效。
- `mooncake-store/include/master_service.h:647` —— `Remove(const std::string& key, const TenantId&, bool force)`。
- `mooncake-store/src/rpc_service.cpp:8` —— `#include <ylt/coro_rpc/coro_rpc_server.hpp>`，master 的 RPC 框架是协程 RPC（`ylt`/yalantinglibs），不是 HTTP，这是"0 条 HTTP 路由"的直接源头证据。
- `mooncake-store/src/master.cpp:118`、`:124` —— `DEFINE_int32(port, 50051, ...)` / `DEFINE_int32(metrics_port, 9003, ...)`：master 是一个独立跑的 gflags 命令行程序，RPC 走 50051，另开一个小 HTTP 端口 9003 只给 Prometheus 指标用（这个 HTTP 端口和"推理服务式 API"不是一回事，见 §7）。
- `mooncake-store/src/real_client.cpp:759` —— `setup_real(...)` 的形参里有 `const std::shared_ptr<TransferEngine>& transfer_engine`，证实 Store 客户端把 Transfer Engine 作为依赖注入，而不是自己再写一套。

值得单独指出的一点：`master_service.cpp` 一个文件就有 13,199 行（`_lab/out/repo_stats.json` 与本篇 `wc -l` 复核一致），是整个仓库最大的单个 `.cpp` 文件——本篇 §2 列出的 `MountSegment`/`GetReplicaList`/`PutStart`/`PutEnd`/`Remove` 全部实现在这一个文件里，而不是像很多项目那样按操作类型拆成多个小文件。这是不是好的工程实践本篇不下判断，只指出这个事实：读 master 的业务逻辑，绕不开打开这一个文件。

- `mooncake-store/src/master.cpp:144`、`:146` —— `DEFINE_double(eviction_ratio, ...)` / `DEFINE_double(eviction_high_watermark_ratio, ...)`，master 独立进程启动时可调的驱逐水位线命令行参数。

**顺带交代一下 Go**：`_lab/out/repo_stats.json` 里 Mooncake 有 5,277 行 Go，本篇正文没有专门讲，这里集中说明来源，省得读者以为是统计噪音——`mooncake-common/etcd/etcd_wrapper.go`（1,338 行）是把 etcd 官方 Go 客户端包一层给 C++ 侧调用（etcd 官方客户端库本身是 Go 写的，C++ 项目要用只能这样包一层）；`mooncake-common/k8s-lease/k8s_lease_wrapper.go`（549 行）是另一条协调后端，用 Kubernetes 原生的 Lease 对象做 leader 选举，作为不想额外运维 etcd 集群时的替代方案；`mooncake-store/go/mooncakestore/store.go`（540 行）则是一个独立的 **Go 客户端 SDK**，与 Python 绑定平级，说明 Mooncake Store 设计上并不假设调用方一定是 Python 进程——本篇没有找到 vLLM/SGLang 使用这个 Go SDK 的证据，它更像是面向 Python 生态之外的潜在接入方准备的。

`mooncake-p2p-store/` 下还有一整套独立的 Go 实现（`core.go`、`transfer_engine.go`、`metadata.go` 等），和本篇聚焦的两大组件是并列的兄弟子项目，不在本篇范围内。

**Python 绑定入口**

- `mooncake-integration/transfer_engine/transfer_engine_py.cpp:1163` —— `PYBIND11_MODULE(engine, m)`，对应 Python 里的 `mooncake.engine`。
- `mooncake-integration/transfer_engine/transfer_engine_py.cpp:1214-1320` —— `py::class_<TransferEnginePy>(m, "TransferEngine")` 及其 `.def(...)` 列表，把 `initialize`/`register_memory`/`batch_transfer_sync_write`/`get_rpc_port` 等 C++ 方法逐个暴露给 Python。
- `mooncake-integration/store/store_py.cpp:1891` —— `PYBIND11_MODULE(store, m)`，对应 `mooncake.store`。
- `mooncake-integration/store/store_py.cpp:2192` —— `py::class_<MooncakeStorePyWrapper>(m, "MooncakeDistributedStore")`，Store 的 Python 门面类。
- `mooncake-integration/store/store_py.cpp:2316`、`:2341` —— `.def("get", ...)` / `.def("remove", ...)`，Store 门面类上最基础的读写方法名，与 §6 里 SGLang `MooncakeStore.store.get/put` 调用的方法名一一对应。

以上共 30 条，`## 2` 要求的 10 条只是下限。

## 3. 核心数据结构

**`TransferRequest`**（`mooncake-transfer-engine/include/transport/transport.h:60`）是整个传输链路里最小的意图单元：

```cpp
struct TransferRequest {
    enum OpCode { READ, WRITE };
    OpCode opcode;
    void *source;
    SegmentID target_id;
    uint64_t target_offset;
    size_t length;
    ...
};
```

它只表达"从本地这块地址、往某个已注册 segment 的某个偏移、搬多少字节、读还是写"，不携带任何"这是第几层的 KV cache"之类的语义——那是调用方（vLLM/SGLang 的 connector）自己要维护的映射。`TransferRequest` 里没在上面代码块列出的 `advise_retry_cnt` 字段（`mooncake-transfer-engine/include/transport/transport.h:70`）值得单独一提：它是"建议重试次数"，命名里的"建议"二字暗示这更像是给底层 worker 的一个提示而非强约束，具体的重试策略（退避时间、换不换一条 `Slice`/换不换网卡）本篇没有去 `worker_pool.cpp` 里逐行核实，只确认了这个字段的存在——这是"批量传输 API 长什么样"里一个容易被忽略但生产环境会关心的细节：调用方能不能精确控制重试行为，还是只能给个建议由库自己决定。

`TransferStatusEnum`（`mooncake-transfer-engine/include/transport/transport.h:77`，`WAITING/PENDING/INVALID/CANCELED/COMPLETED/TIMEOUT/FAILED`）和 `TransferStatus{ s, transferred_bytes }`（`mooncake-transfer-engine/include/transport/transport.h:87`）是轮询用的状态机；`NicLoadStats{ device_name, inflight_bytes, ewma_bandwidth_bps }`（`mooncake-transfer-engine/include/transport/transport.h:92`）则是运行时的网卡负载统计，供更上层做负载感知的路径选择（与 §2 的拓扑静态优先级是两套互补机制：拓扑决定"哪些网卡够格"，负载统计决定"这一刻该挑哪个"）。

**`TopologyEntry`**（`mooncake-transfer-engine/include/topology.h:38`）是拓扑发现的产物：每个存储类型（比如某张 GPU）对应一组 `preferred_hca`（同 NUMA/最近 PCIe 距离的网卡）和 `avail_hca`（其余可用网卡），`selectDevice` 从 `preferred_hca` 里随机或轮询选一个。`TransferRequest` 描述的是"用户意图"，真正被 worker 线程消费的是更细粒度的 `Transport::Slice`（`mooncake-transfer-engine/include/transport/transport.h:118`）：一个 `TransferRequest` 在提交时会被拆成若干个 `Slice`，每个 `Slice` 带自己的 `peer_nic_path`、`dest_rkeys`（RDMA 远端内存钥匙）和独立的 `SliceStatus` 状态机（`PENDING/POSTED/SUCCESS/TIMEOUT/FAILED`），失败或超时可以只重试单个 slice 而不必推倒重来整个批次。

**`BufferDesc`/`SegmentDesc`**（`mooncake-transfer-engine/include/transfer_metadata.h:56`、`:107`）是让两端"知道对方内存长什么样"的握手信息：`BufferDesc` 记一段已注册内存的地址、长度、RDMA 场景下的 `lkey`（本端读写用）/`rkey`（远端读写用）；`SegmentDesc` 把一台机器上的所有 `BufferDesc`、可用设备列表、拓扑信息打包成一个可被远端查询到的"这台机器能被怎么访问"的描述。`registerLocalMemory`（`mooncake-transfer-engine/include/transfer_engine.h:127`）本质上就是本地生成一条 `BufferDesc` 并把它塞进本机的 `SegmentDesc`、通过 `TransferEngine::init` 里指定的 metadata 连接串同步给对端——对端后续发起 `submitTransfer` 时才知道该用哪个 `rkey` 直接 RDMA 写过来，不需要每次传输都重新握手。

**`ReplicateConfig`**（`mooncake-store/include/replica.h:104`，见 §1）是 Store 侧描述"这个 key 要怎么冗余"的配置对象，`replica_num`/`nof_replica_num`/`dfs_replica_num` 分别对应内存/SSD/DFS 三层的副本数，`preferred_segments` 与 `prefer_alloc_in_same_node` 让调用方可以提示 master"优先分配在哪些机器/同一节点"。

**Store 侧的两阶段写状态**：`PutStart` 返回 `std::vector<Replica::Descriptor>`（`mooncake-store/include/master_service.h:449`），`Replica::Descriptor` 这个类型本身也被暴露给了 Python（`mooncake-integration/store/store_py.cpp:1960` `py::class_<Replica::Descriptor>(m, "ReplicaDescriptor")`），描述的是"这个副本实际落在哪个 segment 的哪段地址"。这个 descriptor 就是客户端接下来要拿着去调用 Transfer Engine 真正写数据的地址凭证——master 只发凭证，不碰字节。

**`PyClient`/`RealClient`/`DummyClient`**：`mooncake-integration/store/store_py.cpp:2192` 的 `MooncakeStorePyWrapper` 内部持有的不是直接的 `RealClient`，而是一个抽象基类 `PyClient`（`mooncake-store/include/pyclient.h:212`），下面挂两个实现——`RealClient`（`mooncake-store/include/real_client.h:77`，本篇 §4 走的就是这条路径，真正建立 `TransferEngine` 连接、发 RPC）和 `DummyClient`（`mooncake-store/include/dummy_client.h:19`）。`DummyClient` 的存在解释了 §7 要讨论的一个疑问：Python 测试套件（`mooncake-wheel/tests/` 下能看到 `test_dummy_client.py`、`test_multi_dummy_clients.py`）能在不接真实 RDMA 硬件的 CI 环境里跑起来，很大程度上是因为有这样一个不需要真实网卡的桩实现可用——这是"控制面/数据面分离"（§5 决策二）在可测试性上的一个附带收益：只要 `PyClient` 这层接口稳定，需要碰真实硬件的部分就可以整体换成桩实现，不用为了测试元数据逻辑（租约、驱逐、多副本）而搭一套真实 RDMA 环境。

## 4. 主流程走读

**流程 A：点对点批量传输（vLLM/SGLang 的 PD 连接器走这条路）**

以 vLLM `MooncakeConnectorWorker` 为例（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`）：

1. `self.engine = TransferEngine()` 构造出绑定层对象（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:893`）。
2. `self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, device_name)`（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:915`），对应 C++ 侧 `TransferEnginePy::initialize(local_hostname, metadata_server, protocol, device_name)`（`mooncake-integration/transfer_engine/transfer_engine_py.h:59`）。

   这里有个接口细节值得展开：`metadata_server` 这个形参名并不是纯粹取错了名字。

   - `mooncake-transfer-engine/src/transfer_metadata_plugin.cpp:548`、`:585` 显示这个连接串真的支持 `etcd://...`、`http://...` 这类指向外部协调服务的地址。
   - `mooncake-transfer-engine/src/transfer_metadata.cpp:174` 里还专门有一条 `if (conn_string == P2PHANDSHAKE)` 分支——`P2PHANDSHAKE` 是这个字符串参数里一个特殊的哨兵值（sentinel），代表"跳过外部协调服务，两端直接握手"这种连接模式，和 `etcd://`/`http://` 是并列的几种取值之一，而不是一个被误用的参数。
   - vLLM 与 SGLang 两家不约而同选的都是这个哨兵值而不是真去接一个 etcd 集群——这是一个合理的部署选择（少一个要运维的外部依赖），但确实让"传个字符串 `P2PHANDSHAKE` 进一个叫 `metadata_server` 的形参"这行代码单看起来有点反直觉，读代码时容易误以为是接口设计错误（本篇一开始也这么以为，去翻了 `transfer_metadata_plugin.cpp` 才发现这个参数其实是"多态连接串"，见 §8 第 1 条更细的讨论）。
3. 把 GPU 上的 KV cache 显存整体批量注册：`self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)`（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1697`）。
4. 真正搬数据时调用 `self.engine.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)`（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1606`），阻塞直到这批 block 都写完对端。这一步的调用方是这段代码所在的方法 `_send_blocks`（方法名已经说明了方向）——用的是 `WRITE` 语义，也就是 prefill 侧主动把算好的 KV block **推**给 decode 侧已经注册好的显存地址，而不是 decode 侧发起一次 `READ` 去**拉**。谁 push、谁被动接收，在读代码之前光看"传输"这个词是看不出来的，得具体看调用方法名和 `opcode`。

把两家的构造+初始化代码并排摆出来，能直接看到这种"独立开发却写出近似代码"的程度（均为真实源码摘录，非本篇改写）：

```python
# vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:893,915
self.engine = TransferEngine()
...
ret_value = self.engine.initialize(
    self.hostname, "P2PHANDSHAKE", protocol, device_name
)

# sglang:python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:122,205
self.engine = TransferEngine()
...
ret_value = self.engine.initialize(
    hostname, "P2PHANDSHAKE", protocol,
    device_name if device_name is not None else "",
)
```

两段代码分别由两个团队独立维护，连 `"P2PHANDSHAKE"` 这个字面量都一字不差——这不是巧合，而是因为可选值就那么几个、抄官方示例代码几乎是唯一的落笔方式，间接说明这份 pybind API 的文档/示例代码本身就是事实标准的一部分。

SGLang 的 `MooncakeTransferEngine`（`sglang:python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`）走的是几乎一模一样的顺序：`initialize(hostname, "P2PHANDSHAKE", protocol, device_name)`（`sglang:python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:205`）→ `register_memory`/`batch_register_memory`（`sglang:python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:142`、`:163`）→ `batch_transfer_sync_write`（`sglang:python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:245`）。两个完全独立开发的推理引擎，各自写了一个几乎同构的薄封装类去调用同一套 pybind API——这一点在 §5 会用来支撑"独立成库有价值"的论证。

**流程 B：Store 的 put/get（分层缓存/共享 KV 池走这条路）**

```
写路径：
  客户端 --PutStart(key, size)--> master   （只分配占位副本槽位，不搬字节）
  master --Replica::Descriptor[]--> 客户端
  客户端 --用自己的 TransferEngine 直接写目标 segment 地址-->  （数据面，不经过 master）
  客户端 --PutEnd(key)--> master           （确认写完，key 变为可读）

读路径：
  客户端 --GetReplicaList(key)--> master   （查副本位置 + 授予租约）
  master --Replica::Descriptor[]--> 客户端
  客户端 --用自己的 TransferEngine 直接从目标 segment 拉数据-->  （数据面，不经过 master）
```

1. 写：客户端调 `PutStart(client_id, key, tenant_id, slice_length, config)`（`mooncake-store/include/master_service.h:449`），master 只在元数据表里挑好目标 segment、分配占位副本槽位，返回 `Replica::Descriptor` 列表——**这一步不搬一个字节**。
2. 客户端拿着这些 descriptor，直接用自己持有的 `TransferEngine` 把数据写进对应 segment 的地址（`mooncake-store/src/real_client.cpp:759` 注入的那个实例），走的还是流程 A 的 `TransferRequest`/`submitTransfer` 机制。
3. 写完后客户端调 `PutEnd(client_id, key, tenant_id, replica_type)`（`mooncake-store/include/master_service.h:460`）确认，master 才把这个 key 标记为可读。
4. 读：`GetReplicaList(key, tenant_id)`（`mooncake-store/include/master_service.h:413`）查到副本位置并授予一个租约（doc 注明"grants leases, trigger promotion"），客户端同样拿着 descriptor 直接用 Transfer Engine 去目标 segment 拉数据，master 全程不经手 payload。

SGLang 的 HiCache L3 后端把这四步包在更高一层的批量接口里：`MooncakeStore.store.setup(...)`（`sglang:python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:510`）先建连，随后热身阶段会先做一次 `self.store.put(warmup_key, warmup_value)` 后立刻 `self.store.get(warmup_key)`（`sglang:python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:663`、`:678`）自检链路，正式服务时走批量接口 `batch_put_from`/`batch_get_into`（`sglang:python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:1317`、`:1330`）一次性搬一组 KV block——这些批量方法底层仍然是同一个 `PutStart`/`PutEnd`/`GetReplicaList` 三段式，只是把多个 key 打包进一次 RPC 减少往返。

`mooncake-store/src/real_client.cpp:6291` 附近的注释直接点出了这个设计对上游的意义：*"vLLM pre-registers all GPU KV-cache memory with TransferEngine via register_buffer(), so the offload RDMA can scatter the on-disk blob into non-contiguous per-layer GPU destinations natively — no temp CPU buffer or H2D copy needed."* 也就是说，因为 vLLM 已经把整块 KV cache 显存注册进了 Transfer Engine，Store 从磁盘/远端读回来的数据可以直接 RDMA scatter 进 GPU 上按层切开的非连续目的地址，中间不用过一道 CPU 暂存或 H2D 拷贝。控制面（master）与数据面（Transfer Engine）分离，换来的是数据面可以做这种"零拷贝直接命中调用方已注册好的目标地址"的优化，而 master 完全不需要知道"KV cache 长什么样"。

## 5. 设计决策与代价

**决策一：把"传输"抽成独立项目，而不是每个引擎各自实现。**
- 为什么这么设计：RDMA/NVMe-oF/NVLink 这类硬件传输是一门专精活——每种硬件的 verbs API、错误恢复、多网卡拓扑感知都要专门维护，和"怎么调度请求""怎么做 attention"是完全不同的技能树；同一套传输代码还能同时服务 KV cache 搬运（推理）和权重传输（RL 训练，README 提到的 SGLang RL 场景），复用面比单个推理引擎宽。
- 不这样会怎样：每个引擎各写一遍 RDMA 代码，各自踩连接管理、内存注册、多网卡负载均衡的坑，bug 库互不共享——本库其他篇里能看到 vLLM 和 SGLang 在各自仓库里还留有 `nixl`、`mori`、`ascend` 等*另外几套*独立传输后端（见 §6），说明"每个引擎自己接一套传输方案"在生态里是常态而非例外，Mooncake 只是让"接一套现成的"变得可行。
- 什么时候这是负担：如果你的部署形态就是单机单进程、根本不做 PD 分离/分层缓存，那这整层抽象是纯负担——多一个要装的库、多一份要跟版本的依赖、出问题时要先判断"是 Mooncake 的锅还是调用方拼装参数的锅"。

**决策二：控制面（元数据 RPC）与数据面（字节搬运）彻底分离。**
- 为什么这么设计：见 §4 流程 B——master 只管"谁在哪、能不能读"，真正的大块数据搬运交给已经做好硬件适配的 Transfer Engine，两者可以独立扩缩容（master 是纯 CPU/内存的元数据服务，Transfer Engine 的性能瓶颈在网卡）。
- 不这样会怎样：如果 master 自己去转发数据（类似很多简单 KV 存储的做法），它就会变成所有读写的单点带宽瓶颈——`GetReplicaList` 这种元数据查询的 QPS 需求和"搬一整块 KV cache"的带宽需求完全不是一个数量级，绑在一起会互相拖累。
- 什么时候可以不这样：数据量足够小、副本数足够少（比如只做单机进程内的对象缓存）时，元数据和数据本来就该在一起管理，分离反而多一次 RPC 往返。

**决策三：把两阶段写（`PutStart`/`PutEnd`）而不是原子的单次 `put()` 作为核心协议。**
- 为什么这么设计：数据量可能是几百 MB 到几 GB 的 KV cache 块，不可能在一次 RPC 里把 payload 塞进去；两阶段协议让 master 先"许诺"一块地方，客户端用 Transfer Engine 异步/并行写完再回来确认，master 全程不持有大块内存缓冲区。
- 不这样会怎样：要么退化成"RPC 里带 payload"（master 变成带宽瓶颈，回到决策二想避免的问题），要么客户端自己维护一套"写了但还没告诉 master"的隐式约定（容易在客户端崩溃时产生孤儿数据/脏读）。
- 什么时候可以不这样：写入体积很小（比如只是一个标量元数据）时，两阶段协议纯粹是多一次往返的开销，直接原子写更简单。
- 一个自然要问的后续问题：如果客户端在 `PutStart` 之后、`PutEnd` 之前崩溃了，那些已分配但永远不会被确认的占位副本槽位怎么办？`mooncake-store/include/master_service.h:481` 有一个专门的 `PutRevoke(...)` 方法用来主动撤销未完成的写，配合客户端的租约/心跳机制做超时回收——这部分具体的超时策略与孤儿槽位清理逻辑本篇未逐行核对，只确认了接口存在——在具体部署里依赖这条清理路径之前，应当自行核实它的行为。

**决策四：Transport 做成可插拔接口（近 20 种后端共享同一套抽象），而不是硬编码 RDMA。**
真实的分派表长这样（`mooncake-transfer-engine/src/multi_transport.cpp:417-452` 摘录，宏门控省略）：

```cpp
if (proto == "rdma")       transport = new RdmaTransport();
else if (proto == "ub")     transport = new UbTransport();        // USE_UB
else if (proto == "barex")  transport = new BarexTransport();     // USE_BAREX
else if (proto == "tcp")    transport = new TcpTransport();       // USE_TCP
else if (proto == "nvmeof") transport = new NVMeoFTransport();    // USE_NVMEOF
else if (proto == "nccl")   transport = new NcclHostTransport();  // USE_NCCL_HOST
else if (proto == "ascend") transport = new AscendDirectTransport(); // USE_ASCEND_DIRECT
else if (proto == "ascend") transport = new HcclTransport();      // USE_ASCEND
```

（后两条 `"ascend"` 分支互斥地由不同编译宏门控，同一次编译只会激活其中一条——这不是笔误，是同一个协议名在不同编译配置下指向不同实现的一处细节，读代码时容易看漏。）

- 为什么这么设计：`Transport`（`mooncake-transfer-engine/include/transport/transport.h:44`）是一个带纯虚函数 `getName()`（`mooncake-transfer-engine/include/transport/transport.h:477`）的抽象基类，`TransferEngine::installTransport(proto, args)`（`mooncake-transfer-engine/include/transfer_engine.h:109`）按运行时字符串装配，`mooncake-transfer-engine/src/transport/` 下目前有 `rdma_transport`、`tcp_transport`、`nvmeof_transport`、`nvlink_transport`、`intranode_nvlink_transport`、`cxl_transport`、`hip_transport`（AMD）、`ascend_transport`/`kunpeng_transport`（华为昇腾）、`maca_transport`/`musa_transport`（沐曦/摩尔线程）、`efa_transport`（AWS EFA）、`cxi_transport`（HPE Slingshot）、`nccl_transport`、`rdma_twosided`、`sunrise_link`、`rpc_communicator` 等接近 20 个子目录——覆盖了从 InfiniBand 到国产 GPU 互联到超算专用网络的几乎全部主流选项。这样一套抽象让新硬件厂商只需要实现 `Transport` 接口就能接进来，不用碰调用方代码。
- 不这样会怎样：调用方（vLLM/SGLang）要么被迫为每种硬件写一份专门的传输代码，要么只能支持最主流的一两种硬件，长尾硬件厂商没有统一的接入点。
- 什么时候可以不这样：如果你的目标硬件从一开始就锁死（比如内部集群只有一种网卡型号，永远不会换），可插拔带来的间接层（虚函数调用、协议字符串解析、`installTransport` 的运行时分派）就是纯开销，直接写死更简单也更快。

**决策五：对外接口选择"当库直接编译/链接进调用进程"，而不是起一个独立的网络服务。**
- 为什么这么设计：呼应 §0——这解释了"0 条 HTTP 路由"的根本原因。KV cache 搬运是**微秒到毫秒级**的高频操作，且要直接操作调用方已经 `cudaMalloc` 好的显存地址（`registerLocalMemory` 要的是一个真实指针），如果隔一层 HTTP/gRPC 服务边界，光是序列化和跨进程内存拷贝的开销就足以抵消 RDMA 零拷贝带来的收益。
- 不这样会怎样：如果做成独立进程 + 网络协议，调用方每次搬 KV block 都要先把指针"翻译"成某种跨进程可传递的句柄（共享内存 fd、DMA-BUF 之类），协议设计的复杂度会远超现在这种"直接把指针传给同进程里的 C++ 对象"。
- 什么时候可以不这样：`mooncake-store` 的 master 恰恰是这样做的反例——它是独立跑的服务进程，走 RPC（`ylt::coro_rpc`，非 HTTP）。因为 master 只处理元数据（小、低频），拆成独立服务换来的是可以被多个客户端共享、可以独立做高可用：
  - `mooncake-store/src/ha/` 下按子目录能看出一套典型的主备架构雏形：`leadership/` 管选主，`oplog/` 是操作日志，`snapshot/` 做周期性状态快照，`standby_controller.cpp`（367 行，`class StandbyController` 定义在 `mooncake-store/include/ha/standby_controller.h:28`）与 `standby_metadata_store.cpp`（130 行）合起来处理"备节点怎么追状态、主节点挂了怎么接管"。本篇未逐行核实这套主备切换的正确性（比如脑裂场景怎么处理），只确认了目录结构对应的职责划分。
  - 这套主备协调依赖外部服务发现——`mooncake-store/include/etcd_helper.h:16` 的 `class EtcdHelper`（头文件注释原话是"a helper class for etcd operations... used to handle the requests to the etcd cluster"，源码为证）说明 master 的高可用 leader 选举要接一个外部 etcd 集群。
  - 这和 §4 提到的 Transfer Engine 自己那个可选的 `etcd://` 元数据连接模式是两回事：一个是 Store 控制面自己的主备协调依赖，一个是 Transfer Engine 点对点握手时可选的元数据交换后端，本篇没有去确认部署时这两者是否共用同一个 etcd 集群。
  - 也就是说：**数据面拒绝网络边界，控制面拥抱网络边界**——这不是矛盾，是同一套判断标准（"谁需要被多方共享/独立扩缩容"）在两个组件上给出的不同答案。

**决策六：把同一套源码按硬件预编译成多个独立的 PyPI 包，而不是发一个"通用轮子"让用户自己编译。** README.md 顶部挂着 `mooncake-transfer-engine`（CUDA ≤12.9）、`mooncake-transfer-engine-cuda13`、`mooncake-transfer-engine-non-cuda`、`mooncake-transfer-engine-npu`、`mooncake-transfer-engine-musa`、`mooncake-transfer-engine-efa` 六个并列的 PyPI 徽章（文档所述），对应 §5 决策四列出的近 20 种 Transport 里编译期就要二选一/多选的那部分（CUDA 版本、昇腾 NPU、摩尔线程 MUSA、AWS EFA）。
- 为什么这么设计：C++ 扩展模块（pybind11 生成的 `.so`）在编译期就要链接特定厂商的驱动/SDK头文件（`cuda_alike.h`、`hip_device_guard.h` 这类文件名本身就暗示了条件编译的复杂度），没有一种编译产物能同时兼容 CUDA 和昇腾 NPU 的运行时；拆成多个包让用户 `pip install` 时只拉自己硬件需要的那一份，不用装一堆用不上的依赖。
- 不这样会怎样：要么发一个体积臃肿、把所有硬件的依赖全打包进去的"全家桶"轮子（下载慢、还可能因为某个厂商 SDK 版本冲突装不上），要么强制用户自己从源码编译（丧失 `pip install` 的开箱即用）。
- 什么时候这是负担：对使用方而言，"我该装哪个包"本身变成了一个需要先弄清楚自己硬件型号的决策——vLLM/SGLang 的文档需要专门说明"NVLink 场景装哪个、EFA 场景装哪个"，一旦装错包（比如在 CUDA 13 机器上装了 CUDA ≤12.9 的轮子），报错通常是运行时的符号加载失败，而不是一个友好的"版本不匹配"提示，对新接入方不友好。

**决策七：对外接口只认裸指针/整数地址（`void*`/`uintptr_t`），不认识 PyTorch Tensor、不认识"这是第几层第几个 head 的 KV cache"。** `registerLocalMemory(void* addr, size_t length, ...)`（`mooncake-transfer-engine/include/transfer_engine.h:127`）要的就是一个地址和长度；调用方要先在自己代码里调 `tensor.data_ptr()` 把张量拆成裸指针（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1668`、`:1685`、`:2070` 均有这个调用）才能喂给 Mooncake。
- 为什么这么设计：Mooncake 要同时服务 vLLM、SGLang 甚至非 LLM 场景（README 提到的 RL 权重传输），这些调用方各自的显存布局、张量抽象完全不同（PyTorch Tensor、Ascend 的 NPU 内存对象等等），一个真正通用的传输库没法也不该去理解"KV cache 第几层"这种语义，理解语义是调用方的活；库只保证"给我地址和长度，我负责搬对"。
- 不这样会怎样：如果 Mooncake 试图理解"这是 vLLM 的 KV cache 布局"，它就得跟着 vLLM 的内部数据结构演进而演进，版本耦合会比现在紧得多，而且 SGLang 的 KV 布局和 vLLM 不一样（参见 [[04-SGLang-内存池与KV布局]]），一套"懂语义"的接口没法同时伺候两家。
- 什么时候这是负担：调用方要自己承担"地址算对了没有"的全部责任——`data_ptr()` 拿到的地址如果算错了 offset、或者张量的底层存储被 PyTorch 意外重新分配（比如某些 in-place 操作），Mooncake 不会也不可能帮忙检查，错误会以"数据传对了但语义错位"这种最难排查的方式出现，而不是一个干净的越界报错。

## 6. 同位对照

**vLLM 的 `KVConnectorBase_V1`**（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`）是 vLLM 自己定义的抽象，回答的是"调度器和 worker 之间怎么协作完成一次 KV 搬运"这个**引擎内部编排**问题：`register_kv_caches`（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:264`）、`start_load_kv`（`:289`）、`wait_for_layer_load`（`:307`）、`save_kv_layer`（`:321`）、`get_finished`（`:353`）——这些方法关心的是"在 forward 的哪个时机该发起/等待传输"。Mooncake 的 `TransferEngine` 回答的是完全不同层次的问题："字节怎么真正从 A 机器到 B 机器"。两者不是竞争关系而是叠加关系。

vLLM 实现了两份 `KVConnectorBase_V1` 子类去接 Mooncake——`MooncakeConnector`（走 §4 流程 A，直连 Transfer Engine）和 `MooncakeStoreConnector`（走 §4 流程 B，用 `MooncakeDistributedStore` 做共享 KV 池），二者在 `vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:219` 与 `:224` 分别注册。

更有意思的是，vLLM 的连接器加载支持按 `kv_connector_module_path` 动态 `importlib.import_module`（`vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:107`），这意味着 Mooncake 项目自己也在 `mooncake-wheel/mooncake/mooncake_connector_v1.py` 里维护了**第二份**独立于 vLLM 主仓库的 `KVConnectorBase_V1` 实现——同一个抽象接口，vendored 进 vLLM 树内一份、Mooncake 自己 wheel 里一份，靠 `kv_connector_module_path` 这个开关切换。这是"抽象设计得足够开放"换来的一个直接后果：第三方不需要等主仓库合并 PR 就能自带兼容实现。

**SGLang 的 disaggregation 抽象**分层更细：`BaseKVManager`/`BaseKVSender`/`BaseKVReceiver`（`sglang:python/sglang/srt/disaggregation/base/conn.py:101`）定义最上层接口，`CommonKVManager`/`CommonKVSender`/`CommonKVReceiver`/`CommonKVBootstrapServer`（`sglang:python/sglang/srt/disaggregation/common/conn.py:142`、`:1141`、`:1327`、`:1612`）实现了各后端共享的通用逻辑，`MooncakeKVManager`（`sglang:python/sglang/srt/disaggregation/mooncake/conn.py:196`）只需要继承 `CommonKVManager` 补上 Mooncake 特有的部分。这套结构清楚地暴露了 Mooncake 在 SGLang 里的真实地位：它只是 `disaggregation/` 目录下与 `nixl`（`NixlKVManager` 同样继承 `CommonKVManager`，`sglang:python/sglang/srt/disaggregation/nixl/conn.py:396`）、`mori`、`ascend` 并列的**其中一个可插拔后端**，不是唯一选项。此外 SGLang 的 HiCache 分层缓存还单独接了一次 Mooncake——`MooncakeStore(HiCacheStorage, MooncakeBaseStore)`（`sglang:python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:332`）直接包了 `mooncake.store.MooncakeDistributedStore`（`sglang:python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:270`、`:389`）当作 [[04-SGLang-内存池与KV布局]] 里 L3 层的存储后端，这条路径和 disaggregation 里那条走 Transfer Engine 的路径是分开的两根线，分别对应 Mooncake 的两大组件。

`MooncakeConnector` 对 `KVConnectorBase_V1` 各抽象方法的具体覆写位置（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`）：

| `KVConnectorBase_V1` 抽象方法 | 覆写行号 | 语义 |
|---|---|---|
| `register_kv_caches` | `:538`、`:1629`（调度器/worker 两侧各一份） | 把 KV cache 显存地址交给连接器，内部触发 §4 的 `batch_register_memory` |
| `start_load_kv` | `:549`、`:2000` | 在 forward 开始前发起一次跨机拉取 |
| `wait_for_layer_load` | `:554` | 等某一层的 KV 传输完成再继续算 |
| `save_kv_layer` | `:558` | decode 侧把算完的层标记为可发送 |
| `get_finished` | `:542`、`:1752` | 轮询哪些传输任务已完成，对应 §4 的 `getTransferStatus`/`getBatchTransferStatus` |

把上面这些名字放进一张表，方便对照着查：

| 层次 | vLLM | SGLang | Mooncake |
|---|---|---|---|
| 引擎内部编排抽象（ABC） | `KVConnectorBase_V1`（`vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`） | `BaseKVManager`/`BaseKVSender`/`BaseKVReceiver`（`sglang:python/sglang/srt/disaggregation/base/conn.py:101`） | 不提供，这层由调用方各自定义 |
| 多后端共享的通用实现 | 无独立一层（各 connector 直接实现 ABC） | `CommonKVManager` 等（`sglang:python/sglang/srt/disaggregation/common/conn.py:142`） | 不适用 |
| 接 Mooncake 的具体类 | `MooncakeConnector` + `MooncakeStoreConnector`（`vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:219`、`:224`） | `MooncakeKVManager`（`sglang:python/sglang/srt/disaggregation/mooncake/conn.py:196`）+ `MooncakeStore`（`sglang:python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py:332`） | 被调用的两个门面类：`TransferEngine`、`MooncakeDistributedStore` |
| 并列的其他后端 | `NixlConnector`（同一 factory 表内） | `nixl`/`mori`/`ascend`（`disaggregation/` 下同级目录） | 不适用 |

一句话总结这一节：**vLLM/SGLang 各自都已经把"KV 怎么传"设计成了可插拔点（`KVConnectorBase_V1` / `BaseKVManager` / `HiCacheStorage`），Mooncake 只是插进这些孔位里的众多实现之一——但因为它是两家都在用的唯一交集，从"被集成方"反看，它的 pybind API（`initialize`/`register_memory`/`batch_transfer_sync_write`/`get_rpc_port` 这几个方法名）已经事实上成了两个独立项目都要遵循的最小公约数接口。**

## 7. 踩坑与反直觉

**"disagg 子系统 268,180 行、压倒性最大"这个数字是个假信号，本篇特意验证过。** `_lab/struct_map.py` 用关键词子串匹配给文件打子系统标签，`disagg` 类别的关键词表里包含 `"mooncake"` 这个词（`_lab/struct_map.py` 里 `SUBSYSTEMS["disagg"]` 一项，本是为了在*其他*引擎的代码里识别"这段代码在接 Mooncake"）。但对 Mooncake 自己这个仓库做分析时，它的两个包根目录名字就是 `mooncake-transfer-engine` 和 `mooncake-store`（`_lab/struct_map.py` 的 `PKG_ROOTS` 表），扫描出的每一个相对路径天然都以 `mooncake-` 开头，天然包含子串 `"mooncake"`。核对 `_lab/out/struct_map.json` 里 `engines.mooncake.subsystems`：`disagg` 的 `n_files` 是 650，`files_scanned` 也是 650——**100% 命中**，而 `attention`/`quantization`/`structured_output` 全是 0，`kv_cache` 只有 1 个文件。这不是"Mooncake 的代码几乎全在做 PD 分离"，而是分类关键词和目录命名撞了车，一个纯粹的工具假阳性。这条坑和 `00-总览与阅读地图.md` §4 里记录的 `compare.py` 41→4 那次自我证伪是同一类教训：**跨项目对比的自动化脚本，最容易骗到的是写脚本的人自己。** 这份 268,180 的数字不建议在其他地方引用。

**"测试比产品代码还多"这句话要分层说，不能笼统套用 1.093 这个数。** 拆成几层证据看：

- **口径**：`_lab/repo_stats.py` 里 `test_to_src_ratio` 字段的定义是**纯 Python 口径**——`python_test_lines`（33,413 行，路径含 `test`）除以 `python_src_lines`（30,573 行）＝ 1.093。
- **这个口径覆盖不到项目主体**：Mooncake 的 Python 总共只有 63,986 行，是绑定与测试脚本的规模，不是这个项目的主体；主体是 548,895 行里的 C++（301,147）与头文件（119,480），而这部分**没有**一个对应的"测试/产品比"数字进到 `_lab` 的 JSON 里——`_lab/repo_stats.py` 根本没统计 C++ 的测试占比。
- **能直接观察到、但不是工具算出来的比例**：`_lab/out/repo_stats.json` 的 `top_dirs` 里 `mooncake-store/tests` 有 82,187 行，`mooncake-store/src` 是 78,393 行，两者体量相当；`mooncake-transfer-engine/tests` 另有 20,792 行。这只是"目录行数放在一起看差不多大"的观察，本篇不把它包装成一个数字。
- **测试文件的命名本身透出信号**：`mooncake-store/tests/` 下能看到 `master_service_evict_scenario_test.cpp`、`master_service_tenant_quota_test.cpp`、`master_service_processing_key_double_erase_test.cpp`、`dynamic_replication_test.cpp`、`health_check_test.cpp` 这类以"场景/故障注入"命名的测试。
- **`mooncake-transfer-engine/tests/`（57 个文件）能看出两条线**：一条是几乎每个 Transport 后端各有一个 `*_transport_test.cpp`（`rdma_transport_test.cpp`、`cxl_transport_test.cpp`、`nvlink_transport_test.cpp`、`hip_transport_test.cpp`、`nvmeof_transport_test.cpp`、`efa_transport_test.cpp`、`cxi_transport_test.cpp`……），另一条是专门针对"连接不稳定"这类边界条件的测试：`rdma_endpoint_reestablish_test.cpp`（连接断了重建）、`rdma_context_reprobe_test.cpp`（网卡状态重探测）、`rdma_async_event_drain_test.cpp`（异步事件队列排空）、`connect_pause_tracker_test.cpp`（连接暂停追踪）、`graceful_shutdown_test.cpp`（优雅关闭）。

这些名字本身就在暗示"RDMA 连接会断、会抖、需要专门代码应对"，与 §5 决策一里"RDMA 是一门需要专门团队维护的活"这个论点是同一件事的两个证据来源（一个是产品代码的可插拔设计，一个是测试代码要覆盖的故障面）。

`mooncake-store/tests/` 与 `mooncake-transfer-engine/tests/` 加起来 133 个文件，共同点是分布式系统的正确性很难靠人读代码判断，这种"每种故障场景/每种硬件后端各配一个专门测试文件"的组织方式是可以直接在目录列表里看到的事实，但测试内部具体验证了什么逻辑、覆盖是否到位，本篇未逐个打开核对，不做进一步断言。

把 `mooncake-store/tests/`（76 个文件）按文件名粗分几类（分类是本篇按名字归的，不是仓库自带的标签）：

| 关注点 | 代表性测试文件 |
|---|---|
| 驱逐/容量水位 | `eviction_strategy_test.cpp`、`offload_on_evict_test.cpp`、`promotion_on_hit_test.cpp`、`master_service_evict_scenario_test.cpp` |
| 多租户配额 | `tenant_id_test.cpp`、`tenant_quota_test.cpp`、`tenant_quota_ledger_test.cpp`、`master_service_tenant_quota_test.cpp` |
| 网络/协议边界 | `ipv6_client_test.cpp`、`host_port_fix_test.cpp`、`rpc_timeout_test.cpp`、`uds_transport_test.cpp`、`nof_heartbeat_test.cpp` |
| 存储后端（多层） | `nvme_kv_storage_backend_test.cpp`、`storage_backend_test.cpp`、`dfs_hf3fs_test.cpp`、`dfs_posix_test.cpp`、`ssd_metrics_test.cpp` |
| 并发原语 | `mmap_arena_test.cpp`、`mmap_arena_fallback_test.cpp`、`mutex_test.cpp`、`offset_allocator_test.cpp`、`buffer_allocator_test.cpp` |
| 端到端场景/压力 | `master_scenario_test.cpp`、`stress_workload_test.cpp`、`non_ha_reconnect_test.cpp`、`master_service_processing_key_double_erase_test.cpp` |

这张表本身也是"别过度解读"的一个练习：文件名多不代表覆盖率高，`master_service_processing_key_double_erase_test.cpp` 这类命名精确到"双重擦除"这种具体 bug 场景的文件，读者能推断出它对应过一次真实踩过的坑（否则不会专门为它开一个文件），但具体是哪次、修复了什么，本篇没有去翻 Git 历史核实（`_src/` 是 `--depth 1` 的浅克隆，见 `_PLAN.md` §2，本来也翻不到）。

**Transfer Engine 内部其实并存两套实现，一套还是隐藏功能。**

- `mooncake-transfer-engine/tent/` 是全仓库单个目录里代码量最大的一个（85,136 行，超过 `mooncake-store/src` 的 78,393）。
- `mooncake-transfer-engine/include/transfer_engine.h:35-37` 能看到 `namespace tent { class TransferEngine; }` 与外层 `mooncake::TransferEngine` 并存，外层类里有一个 `use_tent_` 布尔开关（`mooncake-transfer-engine/include/transfer_engine.h:117` `isUsingTent()`）。
- `mooncake-transfer-engine/src/transfer_engine.cpp:394` 显示这个开关由环境变量 `MC_USE_TENT` 或 `MC_USE_TEV1` 控制——全仓库最大的一个子目录，装的是一套默认不启用、需要显式设环境变量才会走的"备用/下一代"实现。`tent` 具体全称与它和经典实现的功能差异，本篇未查证，只确认了"两套并存、由环境变量切换"这个可验证的事实。
- 这套备用实现里还藏着一个和"Mooncake＝RDMA/NVIDIA 生态"这个刻板印象不符的角落：`mooncake-transfer-engine/tent/src/platform/tpu/README.md` 说明 `tent` 有一条编译期开关 `-DUSE_TPU=ON`（默认关闭）打开的 Google TPU（PJRT）支持路径，原理是"TPU 的 HBM 显存不能被网卡直接寻址，所以经 TPU 的传输要先在宿主机 DRAM 中转一跳，再用现有的 RDMA/TCP 传输走第二跳"（文档所述，原文用 `ProxyManager` 描述这条两跳流水线）。这条路径本篇未做代码级核实（只读了它的设计说明文档），但它至少说明"这套传输引擎的野心边界在往 NVIDIA/AMD/昇腾之外的加速器扩"，和多数人对 Mooncake"服务 NVIDIA RDMA 场景"的第一印象不完全一致。

**"0 条 HTTP 路由"不等于"完全没有 HTTP 端口"，两者容易被混为一谈。** `mooncake-store/src/master.cpp:124` 确实有一个 `metrics_port`（默认 9003），master 进程会在这个端口上起一个小 HTTP server 供 Prometheus 抓取指标；此外 `mooncake-store/include/http_metadata_server.h` 也存在一个基于 HTTP 的简易元数据交换服务（用于不方便跑 etcd 之类外部依赖的部署场景）。这些和 `_lab/api_surface.py` 想找的"OpenAI 兼容式推理服务路由"（`/v1/chat/completions` 那一类）完全不是一回事——本篇 §0/§1 说的"0 条 HTTP 路由"特指后者，指标端口和元数据交换端口这两个小型内部 HTTP 服务本篇未逐行核对其路由细节，只确认它们存在、且不在 `_lab/api_surface.py` 的抽取目标范围内。

## 8. 可改进点

1. **`initialize()` 第二个形参 `metadata_server` 是一个"多态连接串"，但类型签名完全看不出来。** §4 已经查证过它真正支持 `etcd://`/`http://` 这类地址（`mooncake-transfer-engine/src/transfer_metadata_plugin.cpp:548`、`:585`）以及 `P2PHANDSHAKE` 这个特殊哨兵值（`mooncake-transfer-engine/src/transfer_metadata.cpp:174`），但 C++/Python 两侧的类型签名都只写了 `const char*`/`str`，没有任何地方（本篇查证范围内）把这几种合法取值集中列出来。vLLM 和 SGLang 两个独立团队都选了 `P2PHANDSHAKE` 这一种，某种程度上说明"这里到底能传什么"这件事目前主要靠读示例代码，而不是靠读类型或文档搞懂——如果这个参数改成一个枚举/`Union` 类型（哨兵模式 vs. 显式地址两个变体），或者哪怕只是在头文件注释里把 `etcd://`/`http://`/`P2PHANDSHAKE` 三种取值并排列出来，能省掉后来者去翻 `transfer_metadata_plugin.cpp` 源码才搞懂这件事的过程。
2. **`## 7` 提到的 `disagg` 假阳性说明 `_lab/struct_map.py` 里 `SUBSYSTEMS` 的关键词表需要更小心处理"自指"情况。** 一个可落地的小改进：`_classify` 在处理某个引擎自身仓库时，应该把该引擎自己的包名从其他类别的关键词命中里排除（或者至少在 `struct_map.json` 里对"某类别命中了 100% 文件"这种情况打一个警示标记），这样下次写别的引擎的文档时不会重复踩同一个坑。
3. **`MC_FORCE_TCP`、`MC_USE_TENT`、`MC_CXL_DEV_PATH`、`MC_PATH_ROUNDROBIN` 这类环境变量散落在 `transfer_engine_impl.cpp`/`transfer_engine.cpp`/`topology.cpp` 各处（本篇 §2、§5、§7 引用到的就有 4 个），未见一个集中的清单文件。** 对比 vLLM `vllm/envs.py` 把所有 `VLLM_*` 环境变量集中声明成一张表（本库其他篇已经引用过这个模式），Mooncake 目前要靠 `grep getenv` 才能拼出完整的开关列表，对新接入方不友好。
4. **`mooncake-store/tools/` 下 `oplog_batch_auditor.cpp`/`oplog_batch_inspector.cpp` 这类运维审计工具没有在 README 里被提及（本篇查证范围内）。** 它们能审计/巡检 HA 的操作日志（`ha/oplog/`），对排查主备切换问题应该很有用，但发现它们完全靠自己翻 `mooncake-store/tools/` 目录——一个"运维工具一览"式的文档索引（哪怕只是列一下每个工具是干什么的、什么时候该用）会比现在"靠 `ls` 发现"友好很多。
5. **"这个包该装哪个"（§5 决策六）目前完全靠用户自己读 README 顶部的徽章分辨。** 如果 `pip install mooncake-transfer-engine` 之后第一次 `import` 时能检测当前机器的硬件（有没有 CUDA、CUDA 版本、有没有 NPU）并在明显不匹配时打印一条提示（而不是让符号加载失败之类的底层错误直接冒出来），能省掉不少新用户在这一步卡住去翻 issue 的时间。

## 9. 自测题与延伸阅读

1. `mooncake-transfer-engine` 和 `mooncake-store` 各自的职责边界是什么？为什么后者的客户端要持有一个前者的实例（提示：`mooncake-store/src/real_client.cpp:759`）？
2. `_lab/api_surface.py` 抽出 Mooncake 的 HTTP 路由数为 0，这是全量 AST 扫描后的结论，还是配置层面就没有去扫？两者的区别为什么重要？
3. `PutStart`/`PutEnd` 两阶段写协议解决的是什么问题？如果把它压缩成一次原子的 `put(key, bytes)`，会在什么场景下出问题？
4. vLLM 对 Mooncake 有两个连接器（`MooncakeConnector` 与 `MooncakeStoreConnector`），它们分别对应 Mooncake 的哪个组件、走的是 §4 的哪条流程？
5. `_lab/struct_map.json` 里 Mooncake 的 `disagg` 子系统占了 268,180 行、命中了 100% 的扫描文件，这个数字为什么不能直接引用为"Mooncake 主要在做 PD 分离"？
6. `Transport` 抽象基类（`mooncake-transfer-engine/include/transport/transport.h:44`）如果不存在，各硬件厂商要接入 Mooncake 需要多付出什么代价？
7. `ReplicateConfig`（`mooncake-store/include/replica.h:104`）里 `replica_num`/`nof_replica_num`/`dfs_replica_num` 分别对应哪一层存储？为什么 Store 要把冗余度拆成三个独立计数，而不是一个笼统的"副本数"？
8. Mooncake 把同一套源码按硬件拆成好几个独立的 PyPI 包（§5 决策六）而不是发一个通用轮子，这个选择的代价具体落在谁头上？

延伸阅读：

- [[12-vLLM-PD分离与KV-Connector]] —— vLLM 侧怎么定义 KV 连接器抽象，本篇 §6 的另一半。
- [[11-SGLang-PD分离与HiCache分层]] —— SGLang 的 disaggregation 后端矩阵与 HiCache 分层缓存，本篇 §6 引用的 `CommonKVManager`/`HiCacheStorage` 均出自那条主线。
- [[09-NVIDIA-Dynamo]] —— 另一个专注 KV-aware 路由与分离部署的项目，可与 Mooncake 做"传输/存储 vs 路由编排"的定位对照。
