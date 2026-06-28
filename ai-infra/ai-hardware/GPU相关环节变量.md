# GPU 相关环境变量

> 用一组「不写进代码、却悄悄改变 GPU 行为」的开关，把设备可见性、通信路径、显存分配、调试同步等运行期决策外置出来。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/网络/NCCL]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点：环境变量是"运行期开关" | 进程级 / 启动前生效 |
| 1 | 地基：环境变量是什么、谁来读、何时读 | env / 进程继承 / fork |
| 2 | 设备可见性：`CUDA_VISIBLE_DEVICES` 全家桶 | 隔离 / 重映射 / MIG |
| 3 | 调试类：`CUDA_LAUNCH_BLOCKING` 等 | 同步 / 报错定位 |
| 4 | 性能/连接类：`CUDA_DEVICE_MAX_CONNECTIONS` | 流并发 / 通信计算重叠 |
| 5 | 显存分配器：`PYTORCH_CUDA_ALLOC_CONF` | 碎片 / 缓存池 / OOM |
| 6 | NCCL 通信：`NCCL_*` 含义与选路 | IB / 网卡 / P2P / 调试 |
| 7 | 其他常见：`OMP_*` / `TORCH_*` / `HF_*` | 线程 / 日志 / 缓存 |
| 8 | 数值例子 / 对照 / 实践 | 手算 / 排错套路 |
| 9 | 常见问题 + 跳转 | FAQ |

## 0. 一句话锚点

> **环境变量 = 在程序"启动之前"就放进进程环境里的键值对；CUDA Runtime / NCCL / PyTorch 在初始化时读它，从而在"不改一行代码"的前提下改变 GPU 行为。**

核心心智模型：它们大多是 **进程级、一次性读取、启动前生效** 的。绝大多数 GPU 相关变量在库 **初始化的那一刻** 被读取并缓存——所以"在 `import torch` 之后再 `os.environ[...]=...`"往往 **不生效**。这是 90% "我设了为什么没用"问题的根因。

## 1. 地基：环境变量到底是什么

**环境变量（environment variable）** 是操作系统给每个进程附带的一张键值表。它不属于某个文件，而是随 **进程** 存在。

三条必须先内化的规则：

1. **继承性**：子进程 **复制** 父进程的环境。你在 shell 里 `export X=1` 后启动的 python，能读到 `X`；但 python 启动 **之后** 在另一个 shell 改 `X`，已运行的 python 看不到。
2. **读取时机**：库代码里通常是 `getenv("X")`。CUDA/NCCL 在 **第一次用到 GPU/通信时** 初始化并 `getenv`，之后缓存。所以"设置必须早于初始化"。
3. **作用域**：`export X=1`（或 PowerShell `$env:X=1`）影响 **当前 shell 及其子进程**；`X=1 python a.py` 只影响 **这一条命令**。

```
        [ shell 进程 ]  env = {CUDA_VISIBLE_DEVICES=0,1 , NCCL_DEBUG=INFO}
              │  fork + exec（复制环境表）
              ▼
        [ python 进程 ]  继承同一份 env
              │  import torch → CUDA Runtime 初始化
              ▼
        getenv("CUDA_VISIBLE_DEVICES") ──► 决定本进程"看见"哪些 GPU
              │
              ▼  之后再改 os.environ 多半已晚（库已缓存）
```

**为什么用环境变量而不是命令行参数？** 因为这些开关常常要 **穿透多层调用**：你的训练脚本 → 框架 → CUDA Runtime → 驱动 → NCCL，中间没人替你传参。环境变量是这条链上 **所有人都能读到** 的"公共广播"。

## 2. 设备可见性：`CUDA_VISIBLE_DEVICES`

这是最重要、最常用、也最容易踩坑的一个。

**是什么**：它告诉 CUDA Runtime"本进程只允许看见这些物理 GPU"。被排除的卡，对这个进程 **彻底不存在**。

**解决什么**：
- **资源隔离**：一台 8 卡机，多人/多任务共用，各自占不同卡互不干扰。
- **重映射**：把物理卡号映射成程序内的逻辑卡号（`cuda:0` 等）。

**核心机制——物理号 vs 逻辑号**：设了 `CUDA_VISIBLE_DEVICES=2,3` 后，程序内 `cuda:0` 指向 **物理 2 号卡**，`cuda:1` 指向 **物理 3 号卡**。程序内编号永远从 0 开始连续重排。

```
物理 GPU:   [0] [1] [2] [3] [4] [5] [6] [7]
                        │   │
CUDA_VISIBLE_DEVICES=2,3 │   │
                        ▼   ▼
程序看见:              cuda:0  cuda:1     ← 只有两张，且重新编号
其余 0,1,4,5,6,7 对本进程 = 不存在
```

**关键用法对照**：

| 设置 | 效果 |
|------|------|
| `CUDA_VISIBLE_DEVICES=0` | 只用物理 0 卡 |
| `CUDA_VISIBLE_DEVICES=2,3` | 用物理 2、3，程序内为 cuda:0/1 |
| `CUDA_VISIBLE_DEVICES=3,2` | 顺序影响映射：cuda:0=物理3，cuda:1=物理2 |
| `CUDA_VISIBLE_DEVICES=""`（空） | **一张都看不见**，强制走 CPU（调试常用） |
| `CUDA_VISIBLE_DEVICES=-1` | 等价于"无可用 GPU" |

**陷阱与权衡**：
- **顺序问题**：物理卡的编号默认按 **PCI 总线序**，可能和 `nvidia-smi` 显示的不一致。若依赖确定性映射，可配合 `CUDA_DEVICE_ORDER=PCI_BUS_ID`（让编号严格按 PCI 总线，而非驱动默认的"性能序"）。
- **必须早设**：在 `import torch` / 任何 CUDA 调用之前设置。Python 内用 `os.environ` 设置时，务必放在文件最顶端、`import torch` 之前。
- **多进程分布式**：`torchrun` / DDP 下，常由启动器为 **每个 rank** 设不同的 `CUDA_VISIBLE_DEVICES`，或让每个进程 `LOCAL_RANK` 对应一张卡——理解两种范式别混用。
- **MIG**：A100/H100 切分实例时，设备标识形如 `MIG-<UUID>`，可用 UUID 字符串作为 `CUDA_VISIBLE_DEVICES` 的值精确指定实例。

## 3. 调试类：让错误"在案发现场"暴露

### `CUDA_LAUNCH_BLOCKING=1`

**问题背景**：CUDA kernel 是 **异步** 提交的。CPU 把 kernel 丢进队列就返回，GPU 稍后才执行。于是当某个 kernel 出错（如越界访问），报错往往 **在后面几行" 不相干"的代码处** 才冒出来——堆栈指向的是"案发后路过的人"，不是真凶。

**机制**：设为 1 后，每次 kernel 启动都 **同步等待执行完成** 再返回 CPU。错误立刻在 **真正出错的那一行** 抛出。

```
异步(默认):  CPU: launch A → launch B → launch C → ...（A 其实崩了）
                                          ▲
                            报错在这里冒出，堆栈指向 C，误导

阻塞(=1):    CPU: launch A → [等 A 跑完] ← A 崩 → 立即报错在 A 这行
```

**权衡**：它会 **严重拖慢** 运行（消除了 CPU/GPU 重叠），**仅用于调试**，定位完务必关掉。它是定位 `CUDA error: device-side assert` / `illegal memory access` 的第一工具。

### 配套调试变量（讲含义）

- `TORCH_USE_CUDA_DSA`：开启 device-side assertion，让设备端断言带上更可读的信息（需配合相应构建/版本，**以官方文档为准**）。
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`：见上节，让卡号稳定可复现，调试多卡映射必备。

## 4. 性能/连接类：`CUDA_DEVICE_MAX_CONNECTIONS`

**是什么**：限制 host 到单个 device 的 **硬件连接（hardware work queue / connection）** 数量。一个 GPU 上多个 CUDA stream 要靠这些连接把任务投递到硬件队列。

**为什么在大模型训练里常被设为 1**：在 **张量并行（TP）+ 序列并行 / 通信计算重叠** 场景，框架（如 Megatron-LM）依赖 **通信 kernel 和计算 kernel 的提交顺序严格可控**，以保证"先发起的通信能优先占用 SM、与后续计算真正重叠"。把连接数设为 1，相当于把所有流的任务 **串到同一个硬件队列**，强制按提交顺序排队，从而让重叠调度 **确定且符合预期**。

```
多连接(默认):  stream0 ─► 队列A ┐
               stream1 ─► 队列B ┼─ 硬件可乱序/并发调度 → 重叠不确定
               commm   ─► 队列C ┘

=1:            stream0 ─┐
               stream1 ─┼─► 单队列(严格 FIFO) → 提交序=执行序 → 重叠可控
               commm   ─┘
```

**权衡**：它是一把"为了确定性牺牲一点并发"的刀。普通推理/单卡训练不需要；只有在 **TP 通信重叠** 这类对调度顺序敏感的场景，按框架文档建议设置。**具体取值与适用范围以框架/官方文档为准。**

## 5. 显存分配器：`PYTORCH_CUDA_ALLOC_CONF`

**先理解问题**：`cudaMalloc` 很慢且会同步。PyTorch 因此实现了 **缓存分配器（caching allocator）**：向驱动一次要一大块，自己切小块给张量，张量释放后 **不还给驱动，而是放回自己的池子** 复用。好处是快；副作用是 **显存碎片化**——空闲总量够，但凑不出一块连续的大块，于是抛 OOM。

`PYTORCH_CUDA_ALLOC_CONF` 就是 **调这个分配器行为** 的总开关（多个子项用逗号分隔）。

```
驱动持有的大块显存:
[■■■■□□□■■□□□□■■■■■]   ■=占用 □=空闲
                          空闲总量 6 块，但最大连续空隙只有 3 块
请求 4 连续块 → OOM（尽管"还有 6 块空闲"）→ 碎片化的典型症状
```

**关键子项的含义（讲含义，不背默认值）**：

| 子项 | 含义 / 解决什么 | 权衡 |
|------|----------------|------|
| `expandable_segments:True` | 让分配段可 **扩展/收缩**，显著缓解碎片化 OOM | 较新机制，行为以官方版本说明为准 |
| `max_split_size_mb:<N>` | 大于该阈值的块 **不再被切分** 复用，减少大块被切碎 | 设太小会降低复用率 |
| `garbage_collection_threshold:<f>` | 缓存占用超过该比例时触发回收 | 回收有开销 |
| `roundup_power2_divisions:<N>` | 分配尺寸向 2 的幂对齐的粒度，减少尺寸碎片 | 可能略增占用 |

**实践要点**：
- 报 `CUDA out of memory` 但 `reserved` 远大于 `allocated` → 典型 **碎片** 问题，优先试 `expandable_segments:True` 或调 `max_split_size_mb`。
- 它只改 **分配策略**，不会变出更多物理显存；真不够还得靠 batch/序列长度/激活重计算/并行策略解决。
- 诊断用 `torch.cuda.memory_summary()` 看 allocated / reserved 差距。

## 6. NCCL 通信：`NCCL_*`

NCCL（NVIDIA Collective Communications Library）负责多卡/多机的集合通信（all-reduce 等）。它的环境变量主要做两件事：**选对传输路径** 和 **打开调试可观测性**。详见 [[ai-infra/网络/NCCL]]。

### 选路与传输

| 变量 | 含义 | 何时用 |
|------|------|--------|
| `NCCL_IB_DISABLE=1` | 关闭 InfiniBand，强制走 TCP/Socket | 机器无 IB、或 IB 有问题想排除时 |
| `NCCL_SOCKET_IFNAME=bond0` | 指定走哪块 **网卡**（按名字，可前缀匹配） | 多网卡时避免 NCCL 选错网卡 |
| `NCCL_IB_HCA=mlx5` | 指定用哪些 IB HCA（网卡设备） | 多 IB 卡时绑定 |
| `NCCL_P2P_DISABLE=1` | 关闭 GPU 间 P2P（NVLink/PCIe 直连） | 排查 P2P 故障，**会变慢** |
| `NCCL_NET_GDR_LEVEL` | 控制 GPUDirect RDMA 启用层级 | 调优 GPU↔网卡直通 |
| `NCCL_ALGO` / `NCCL_PROTO` | 强制集合算法 / 协议 | 高级调优，慎用 |

```
NCCL 选路决策（简化）:
 集合通信(all-reduce)
        │
        ├─ 同机内多卡 ─► 优先 NVLink/P2P（最快）
        │               NCCL_P2P_DISABLE=1 会退化到 PCIe/共享内存(慢)
        │
        └─ 跨机 ─► 有 IB 且未禁 ─► InfiniBand/RDMA（快）
                  │
                  └─ NCCL_IB_DISABLE=1 ─► TCP Socket（按 NCCL_SOCKET_IFNAME 选网卡，慢）
```

### 调试与可观测

| 变量 | 含义 |
|------|------|
| `NCCL_DEBUG=INFO` | 打印 NCCL 选路/拓扑/版本日志，**排错第一步** |
| `NCCL_DEBUG=WARN` | 只打警告 |
| `NCCL_DEBUG_SUBSYS=ALL` | 细分子系统日志（INIT/NET/...） |

**为什么要手动指定网卡/IB**：自动探测在 **多网卡、bond、容器网络** 下经常选错——选到一块没连通或慢速的网卡，导致 **通信卡死或极慢**。`NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` 就是把这个决策从"自动猜"改成"明确指定"。**禁用类变量（IB_DISABLE/P2P_DISABLE）多是排错手段，问题定位后应去掉以恢复性能。**

## 7. 其他常见环境变量（顺带认识）

| 变量 | 归属 | 含义 |
|------|------|------|
| `OMP_NUM_THREADS` | OpenMP | 每进程 CPU 线程数；分布式下常设较小值避免超订 |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | PyTorch | DDP 详细调试（如参数未用警告） |
| `TORCH_NCCL_BLOCKING_WAIT` / `TORCH_NCCL_ASYNC_ERROR_HANDLING` | PyTorch | 通信超时/错误处理行为（名称随版本演进，**以官方为准**） |
| `MASTER_ADDR` / `MASTER_PORT` | PyTorch 分布式 | rendezvous 主节点地址/端口 |
| `RANK` / `WORLD_SIZE` / `LOCAL_RANK` | 分布式 | 全局/总数/本机内编号（通常由 `torchrun` 注入） |
| `HF_HOME` / `TRANSFORMERS_CACHE` | HuggingFace | 模型/数据缓存目录 |
| `TOKENIZERS_PARALLELISM` | HF tokenizers | 关闭并行避免 fork 警告 |

> 注意分类：第 2~5 节是 **CUDA/PyTorch 本地** 变量；第 6 节是 **通信** 变量；本节 `RANK/MASTER_*` 是 **分布式编排** 变量。理解它们处在链路的不同层，排错时才知道该看哪一层。

## 8. 数值例子 / 对照 / 实践

### 例 1：可见性映射手算

机器 8 卡，命令 `CUDA_VISIBLE_DEVICES=5,2 python train.py`：

- 程序里 `torch.cuda.device_count()` → **2**
- `torch.device('cuda:0')` → **物理 5 号卡**
- `torch.device('cuda:1')` → **物理 2 号卡**
- 想把张量放物理 2 号 → 代码写 `.to('cuda:1')`（不是 `cuda:2`！）

这就是"程序内逻辑号 ≠ 物理号"的实战意义。

### 例 2：碎片化 OOM 的量级直觉

设一张卡 80 GB，分配器已 `reserved`（向驱动要了）78 GB，其中 `allocated`（真用着）只有 60 GB。理论空闲 80−60=20 GB，但这 20 GB 散落在许多小空隙里。现在要分配一个 4 GB 连续激活张量：

$$\text{空闲总量}=20\text{ GB} \;>\; 4\text{ GB},\quad \text{但 } \max(\text{连续空隙}) = 3\text{ GB} < 4\text{ GB} \Rightarrow \text{OOM}$$

报错信息里 `reserved` 与 `allocated` 的 **18 GB 缺口** 就是碎片的体量——这正是该调 `PYTORCH_CUDA_ALLOC_CONF` 的信号。

### 例 3：通信路径对带宽的影响（数量级，约值，以实测为准）

| 路径 | 典型单向带宽（约） | 备注 |
|------|------------------|------|
| NVLink（同机 GPU 直连） | 数百 GB/s 量级 | 误把它禁了会断崖式变慢 |
| PCIe 4.0 ×16 | 约 32 GB/s 量级 | P2P 被禁后的退路之一 |
| InfiniBand HDR | 约 25 GB/s（200 Gb/s）量级 | 跨机首选 |
| 普通以太网 TCP | 约 1–12 GB/s（10–100 Gb/s） | `NCCL_IB_DISABLE=1` 后的退路 |

直觉：一次 all-reduce 通信量 $\approx 2\times$ 模型/梯度大小（ring all-reduce 约传 $2(N{-}1)/N$ 倍数据）。若你不小心让 NCCL 走了 TCP 而非 IB，**同样的通信量耗时可能差一个数量级**——这就是为什么 `NCCL_DEBUG=INFO` 确认实际选路如此重要。

### 排错套路（按顺序）

```
GPU 程序出问题
   │
   ├─ "看不到卡/卡号错" ──► 查 CUDA_VISIBLE_DEVICES + CUDA_DEVICE_ORDER（且设在 import 前）
   │
   ├─ "CUDA error 报在奇怪的行" ──► CUDA_LAUNCH_BLOCKING=1 重跑，定位真凶
   │
   ├─ "OOM 但好像还有显存" ──► 看 reserved vs allocated；试 PYTORCH_CUDA_ALLOC_CONF
   │
   └─ "多卡卡死/极慢" ──► NCCL_DEBUG=INFO 看选路；
                         必要时 NCCL_SOCKET_IFNAME 指定网卡 / NCCL_IB_DISABLE 排除 IB
```

### 实践要点

- **设在最前面**：可见性、ORDER、ALLOC_CONF 这类，必须在任何 CUDA 调用/`import torch` 之前。脚本内设置就放文件第一行区域。
- **别把排错开关带进生产**：`CUDA_LAUNCH_BLOCKING`、`NCCL_*_DISABLE`、`NCCL_DEBUG=INFO` 都会拖慢或刷屏，定位后删掉。
- **讲含义 > 背默认值**：默认值随驱动/库/版本变化，不要硬记数字；记住"这个变量调的是哪类行为、调它的代价是什么"。
- **可复现**：固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 让多机多卡映射一致，避免"换台机器卡号就乱"。

## 常见问题

| 问题 | 答案 |
|------|------|
| 我在代码里 `os.environ['CUDA_VISIBLE_DEVICES']='0'` 没生效？ | 多半设在 `import torch` 之后，库已初始化。挪到文件最顶端。 |
| 设了 `=2,3` 为什么代码里要写 `cuda:0/1`？ | 可见设备被重新编号为 0 开始；逻辑号 ≠ 物理号。 |
| `nvidia-smi` 卡号和程序里对不上？ | 默认编号可能非 PCI 序。设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。 |
| `CUDA_LAUNCH_BLOCKING=1` 能常开吗？ | 不能，仅调试。它消除 CPU/GPU 重叠，性能大降。 |
| OOM 报错说 reserved 很大、allocated 不大？ | 碎片化。试 `expandable_segments:True` / 调 `max_split_size_mb`。 |
| 多机训练通信巨慢怎么办？ | `NCCL_DEBUG=INFO` 确认选路；指定 `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`，确认走了 IB/NVLink。 |
| `NCCL_P2P_DISABLE=1` 该长期开吗？ | 不该，会退化到 PCIe/SHM，变慢。仅排错用。 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` 普通训练要设吗？ | 不需要。主要服务于 TP 通信计算重叠等对提交顺序敏感的场景，按框架文档。 |
| 这些默认值是多少？ | 随版本/驱动变化，**以官方文档为准**；记含义而非数字。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[ai-infra/ai-hardware/CUDA]] — CUDA 编程模型 / Runtime / stream，理解 LAUNCH_BLOCKING 与 MAX_CONNECTIONS 的底层
- [[ai-infra/网络/NCCL]] — 集合通信库的算法、拓扑与 NCCL_* 调优细节
