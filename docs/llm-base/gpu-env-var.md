# GPU 环境变量全解（CUDA / NCCL / PyTorch 分布式）

> 一句话定位：GPU 环境变量是「不改代码、用 `KEY=VALUE` 在进程启动前注入」来控制设备可见性、显存分配、集合通信路径、并行启动拓扑的旋钮——读懂它们等于读懂多卡训练/推理的「隐形配置层」。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]] · [[ai-infra/网络/集合通信原语]] · [[llm-inference/大模型推理张量并行]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]

## 阅读地图

| 节 | 你将学到 | 关键产物 |
|----|----------|----------|
| 0 | 一句话锚点：环境变量是什么、为什么存在 | 心智模型 |
| 1 | 地基：进程、环境、CUDA Runtime/Driver 分层 | 注入时机 |
| 2 | 设备可见性 `CUDA_VISIBLE_DEVICES` | 物理↔逻辑映射 |
| 3 | 设备枚举顺序 `CUDA_DEVICE_ORDER` | PCI vs 算力排序 |
| 4 | 显存分配器 `PYTORCH_CUDA_ALLOC_CONF` | 碎片/OOM 治理 |
| 5 | 同步调试 `CUDA_LAUNCH_BLOCKING` | 错误定位 |
| 6 | NCCL 集合通信变量族 | 通信路径调优 |
| 7 | PyTorch 分布式启动变量 | rank/world 拓扑 |
| 8 | 数值/精度/性能开关 | TF32、缓存、JIT |
| 9 | 数值手算：通信量与变量的关系 | AllReduce 字节数 |
| 10 | 排错速查 + 常见问题 | 故障矩阵 |

## 0. 一句话锚点

- **环境变量 = 进程启动前写入「环境块」的键值对**，被 CUDA Runtime、NCCL、PyTorch 在初始化时**读取一次**，从而改变行为而**无需重新编译或改代码**。
- 三大家族：
  - `CUDA_*` —— NVIDIA CUDA Runtime/Driver 读取，管「看得到哪些卡、卡怎么排序、同步还是异步」。
  - `NCCL_*` —— NVIDIA 集合通信库读取，管「AllReduce/AllGather 走哪条物理链路、用什么算法」。
  - `PYTORCH_* / 分布式 (`MASTER_ADDR` 等)` —— 框架层读取，管「显存分配策略、谁是 rank 0、共多少进程」。
- 黄金法则：**几乎所有变量只在「相关库第一次初始化前」生效**。在 `import torch` 之后再 `os.environ[...]=` 多半无效。

```
   你启动进程时                进程内
 KEY=VALUE python train.py
        │
        ▼  写入环境块(env block)
   ┌──────────────┐  import torch / cuda 初始化时读取一次
   │  环境变量字典 │ ──────────────┐
   └──────────────┘               ▼
        CUDA_VISIBLE_DEVICES → CUDA Runtime: 决定 device 0..N
        CUDA_DEVICE_ORDER    → CUDA Runtime: 决定谁是 device 0
        NCCL_*               → NCCL 初始化:   决定通信路径/算法
        MASTER_ADDR/PORT     → c10d:          决定 rendezvous 地址
        PYTORCH_CUDA_ALLOC_CONF→ 缓存分配器:  决定显存切分策略
```

## 1. 地基：进程 / 环境块 / CUDA 分层

要理解「为什么是环境变量而不是命令行参数」，先看软件栈分层：

```
┌─────────────────────────────────────────────┐
│  你的脚本 train.py (Python / PyTorch)          │  ← 读 PYTORCH_*, MASTER_*
├─────────────────────────────────────────────┤
│  PyTorch / framework (libtorch, c10d, NCCL封装)│  ← 读 NCCL_*, TORCH_*
├─────────────────────────────────────────────┤
│  CUDA Runtime  (libcudart, 高级API)            │  ← 读 CUDA_VISIBLE_DEVICES 等
├─────────────────────────────────────────────┤
│  CUDA Driver   (libcuda, 内核态接口)            │  ← 读 CUDA_DEVICE_ORDER 等
├─────────────────────────────────────────────┤
│  GPU 硬件 (SM / HBM / NVLink / PCIe)           │
└─────────────────────────────────────────────┘
```

- **环境变量是跨层、跨语言的统一配置通道**：底层 C 库无法接收 Python 的命令行参数，但任何语言都能读 `getenv()`。
- **「初始化即冻结」**：CUDA Runtime 在首个 CUDA 调用（如 `torch.cuda.init()`、第一个 `.cuda()`）时读取设备相关变量并建立映射，之后再改环境变量不会重映射。NCCL 在创建 communicator 时读取 `NCCL_*`。
- 注入方式（POSIX）：`KEY=VALUE python a.py`（仅本进程）、`export KEY=VALUE`（当前 shell 及其子进程）、`os.environ["KEY"]="V"`（必须在相关库初始化前）。Windows 用 `set KEY=VALUE` / `$env:KEY="V"`。具体生效细节以官方文档为准。

## 2. 设备可见性：`CUDA_VISIBLE_DEVICES`

最常用、最易踩坑的一个。它做**物理 GPU → 进程内逻辑 GPU 的过滤与重排**。

- 机器有 8 张物理卡（物理 id 0..7）。设置后，进程**只能看到列出的卡**，且**重新编号为 0,1,2...**。

```
物理卡:   GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
                       │         │
CUDA_VISIBLE_DEVICES=2,5
                       ▼         ▼
进程内可见:          cuda:0    cuda:1     ← 只剩两张，重新编号！
                  (=物理2)  (=物理5)

代码里 torch.device("cuda:0")  → 实际落到物理 GPU2
代码里 torch.device("cuda:1")  → 实际落到物理 GPU5
代码里 torch.device("cuda:2")  → 报错：进程看不到第3张
```

要点与陷阱：
- 顺序即映射：`CUDA_VISIBLE_DEVICES=5,2` 则 `cuda:0`=物理5、`cuda:1`=物理2。可用于**手动控制 NCCL 拓扑顺序**。
- 空字符串 `CUDA_VISIBLE_DEVICES=""` → 进程看不到任何 GPU（常用于强制 CPU 调试）。
- 无效 id（如越界）通常会让该位置及之后的卡不可见，行为以官方为准——务必核对实际可见卡数。
- 多任务隔离：两份训练任务分别 `CUDA_VISIBLE_DEVICES=0,1,2,3` 和 `4,5,6,7`，互不抢卡。
- 与 `nvidia-smi` 的物理 id 一致吗？**默认不一定**——见第 3 节 `CUDA_DEVICE_ORDER`。`nvidia-smi` 默认按 PCI 总线排，CUDA 默认按「算力快慢」排，二者编号可能不同，这是经典坑。

```
nvidia-smi 看到:        PCI顺序  GPU0 GPU1 GPU2 ...
CUDA默认(FASTEST_FIRST): 算力排序  可能 GPU0 ≠ smi 的 GPU0  ← 编号对不上！
解决: export CUDA_DEVICE_ORDER=PCI_BUS_ID  让两者统一
```

## 3. 设备枚举顺序：`CUDA_DEVICE_ORDER`

- 取值（以官方为准）：
  - `FASTEST_FIRST`（默认）：按 CUDA 估计的「算力从强到弱」排序，最强卡成为 device 0。
  - `PCI_BUS_ID`：按 PCI 总线物理地址排序，与 `nvidia-smi -L` 的编号一致。
- 为什么重要：脚本里写死 `cuda:0`，换台机器/换驱动后，`FASTEST_FIRST` 可能让 `cuda:0` 指向不同的物理卡，导致**亲和性、NVLink 拓扑、绑核**全错位。
- 强烈建议在多卡集群统一 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`，让 `CUDA_VISIBLE_DEVICES` 的数字与 `nvidia-smi` 完全对应，便于排查。

```
默认 FASTEST_FIRST:   物理拓扑被「算力」打乱
  CUDA device0 = 最快卡 (可能是 PCI 上的第3张)
PCI_BUS_ID:           顺序 == nvidia-smi == NVLink 文档拓扑图
  CUDA device0 = PCI bus 最小那张  ← 可预测、可绑核
```

## 4. 显存分配器：`PYTORCH_CUDA_ALLOC_CONF`

PyTorch 用「缓存分配器（caching allocator）」管理显存：向 CUDA 申请大块、内部切小块复用，避免频繁 `cudaMalloc`。该变量调它的策略，是治理「碎片化 OOM」的主力旋钮。

常见子项（语义稳定，具体默认值以官方为准）：

| 子项 | 作用 | 典型用途 |
|------|------|----------|
| `max_split_size_mb:N` | 大于 N MB 的块不再被切分复用 | 缓解大张量碎片 |
| `expandable_segments:True` | 用可扩展段，减少碎片 | 变长序列/动态 shape |
| `garbage_collection_threshold:f` | 占用超阈值触发回收 | 显存吃紧时 |
| `roundup_power2_divisions:N` | 块大小向 2 的幂对齐分档 | 规整分配 |

碎片化为什么会「明明有空闲却 OOM」：

```
显存物理空闲 = 2GB，但被切成不连续的碎片：
[已用512MB][空256MB][已用512MB][空256MB][已用512MB][空256MB]...
            └──┬──┘           └──┬──┘           └──┬──┘
要分配一个 700MB 连续块 → 找不到 → OOM！(虽然总空闲>700MB)

expandable_segments / 合理 max_split_size_mb 让分配器
把空洞合并/避免过度切分 → 同样显存能放下更大张量。
```

注意：分配器调参是**对症缓解**，根因往往是 batch/序列过大或激活未重计算——优先看 [[llm-algo/FLOPs]] 与显存估算，再调这里。

## 5. 同步调试：`CUDA_LAUNCH_BLOCKING`

- CUDA kernel 默认**异步**下发：CPU 把 kernel 丢进 stream 就继续往下跑，错误可能在**很多行之后**才报出来，堆栈指向无辜代码。
- `CUDA_LAUNCH_BLOCKING=1` 让每个 kernel **同步执行**（下发后立即等结果），错误**就地报出**，堆栈精确指向出错 kernel。

```
异步(默认):
  行10 launch kernelA ─┐(立刻返回)
  行11 launch kernelB  │  GPU 还在算 A...
  行12 ... 行50        │  A 在某刻挂了 → 错误在行50附近才冒出来!
                        ▼
CUDA_LAUNCH_BLOCKING=1:
  行10 launch kernelA → 等A算完 → A挂了立刻在行10报错 ✓
```

- 代价：关闭 CPU/GPU 重叠，**严重变慢**，仅用于定位 `illegal memory access`、`device-side assert` 等。**生产环境务必关掉**。

## 6. NCCL 集合通信变量族

NCCL 负责 AllReduce / AllGather / ReduceScatter 等（原理见 [[ai-infra/网络/集合通信原语]]）。这些变量决定**走哪条物理链路、用什么算法、要不要打日志**。语义稳定，取值/默认以官方为准。

| 变量 | 作用 | 典型场景 |
|------|------|----------|
| `NCCL_DEBUG` | 日志级别（如 INFO/WARN） | 排查通信、确认走没走 NVLink |
| `NCCL_DEBUG=INFO` | 打印拓扑、所选算法、通道数 | 第一手诊断 |
| `NCCL_IB_DISABLE` | 关/开 InfiniBand 走 IB 还是 socket | 无 IB 环境强制 TCP |
| `NCCL_SOCKET_IFNAME` | 指定走哪张网卡（如 `eth0`、`^docker`） | 多网卡机器选对网口 |
| `NCCL_P2P_DISABLE` | 关闭 GPU 间 P2P(NVLink/PCIe直连) | 排查 P2P 故障 |
| `NCCL_ALGO` / `NCCL_PROTO` | 强制 Ring/Tree 算法、传输协议 | 性能调优 |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 启用层级 | 跨节点 RDMA |

通信路径分层（NCCL 自动选最快可用，可被变量覆盖）：

```
同一节点 GPU↔GPU:   NVLink (最快)  >  PCIe P2P  >  经主机内存中转
跨节点 GPU↔GPU:     IB/RoCE + GDR (绕过CPU) >  TCP socket (慢)

NCCL_DEBUG=INFO 会打印类似:
  "via NVLink"  / "via P2P/IPC"  / "via SHM"  / "via NET/Socket"
据此判断有没有掉到慢路径——掉到 socket 通常意味着配置错了网卡或被禁了 P2P。
```

排错铁律：**多机训练卡在初始化/hang 住**，第一步就是 `NCCL_DEBUG=INFO` 看它停在哪个阶段（bootstrap / 拓扑探测 / all-reduce）。

## 7. PyTorch 分布式启动变量

`torch.distributed` 用 `env://` rendezvous 时，靠下列变量让各进程「找到彼此、知道自己是谁」：

| 变量 | 含义 | 谁设置 |
|------|------|--------|
| `MASTER_ADDR` | rank 0 所在主机地址 | 启动器/用户 |
| `MASTER_PORT` | 协调端口 | 启动器/用户 |
| `WORLD_SIZE` | 全局进程总数（= 总 GPU 数） | 启动器 |
| `RANK` | 本进程全局编号 0..WORLD_SIZE-1 | 启动器 |
| `LOCAL_RANK` | 本进程在**本节点内**的编号 | 启动器 |

```
2 节点 × 每节点 4 卡 = WORLD_SIZE 8
节点A (MASTER_ADDR=A的IP):                节点B:
 RANK0 LOCAL_RANK0 → cuda:0               RANK4 LOCAL_RANK0 → cuda:0
 RANK1 LOCAL_RANK1 → cuda:1               RANK5 LOCAL_RANK1 → cuda:1
 RANK2 LOCAL_RANK2 → cuda:2               RANK6 LOCAL_RANK2 → cuda:2
 RANK3 LOCAL_RANK3 → cuda:3               RANK7 LOCAL_RANK3 → cuda:3
        │                                        │
        └────── 都连到 MASTER_ADDR:MASTER_PORT ──┘  完成 rendezvous
```

- 关键区分：`RANK` 是**全局**唯一身份（决定数据分片、是否打印日志），`LOCAL_RANK` 决定**本机用哪张卡**（`torch.cuda.set_device(LOCAL_RANK)`）。
- `torchrun`（推荐）会**自动注入** `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*`，你只需 `--nproc_per_node` 等；手写 `os.environ` 易错。
- 与第 2 节联动：常见做法是**不**用 `CUDA_VISIBLE_DEVICES` 限卡，让每进程用 `LOCAL_RANK` 选卡；二者混用要小心映射叠加。

## 8. 数值 / 精度 / 性能开关

| 变量 | 作用 | 备注 |
|------|------|------|
| `NVIDIA_TF32_OVERRIDE` | 全局开/关 TF32 matmul | 影响精度/速度（细节以官方为准） |
| `CUDA_CACHE_PATH` / `CUDA_CACHE_DISABLE` | JIT 编译缓存位置/开关 | 首启编译慢可缓存 |
| `OMP_NUM_THREADS` | 每进程 OpenMP 线程数 | 多进程时防 CPU 超订 |
| `TOKENIZERS_PARALLELISM` | HF 分词器并行 | fork 警告抑制 |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | 关缓存分配器（调试） | 极慢，仅排查显存 |

- TF32：Ampere+ 上 FP32 matmul 默认可走 TF32（尾数精度降低、速度大涨）。精度敏感任务可关，详见 [[llm-base/FP16-BF16]] 与 [[llm-compression/quantization/量化基础]]。
- `OMP_NUM_THREADS` 在 `torchrun` 多进程下若不设，可能每进程都抢满全部 CPU 核，导致 CPU 抖动拖慢数据加载——经验上设为「物理核数 / 每节点进程数」附近。

## 9. 数值手算：变量到底省了多少通信 / 显存

**手算 A：数据并行 AllReduce 的通信量**（解释为何 `NCCL_*` 选对链路如此关键）

设模型参数量 $P = 7\times10^9$（7B），梯度用 FP16（每参数 2 字节），数据并行度（GPU 数）$N=8$。Ring-AllReduce 每个 rank 收发的总字节数近似：

$$
\text{Bytes per rank} \approx 2 \times \frac{N-1}{N} \times (P \times 2)
$$

代入：
- 梯度总字节 $P\times2 = 7\times10^9 \times 2 = 1.4\times10^{10}\ \text{B} = 14\ \text{GB}$。
- 系数 $2\times\frac{8-1}{8} = 2\times0.875 = 1.75$。
- 每 rank 通信量 $\approx 1.75 \times 14\ \text{GB} = 24.5\ \text{GB}$，**每步每卡**。

链路带宽对比（量级，具体以硬件为准）：
- 走 NVLink（设 ~300 GB/s）：$24.5 / 300 \approx 0.082\ \text{s} = 82\ \text{ms}$。
- 掉到 PCIe（设 ~16 GB/s）：$24.5 / 16 \approx 1.53\ \text{s}$。
- 掉到 TCP socket（设 ~1.25 GB/s ≈ 10 Gbps）：$24.5 / 1.25 \approx 19.6\ \text{s}$！

结论：一旦 `NCCL_DEBUG=INFO` 显示通信「via Socket」而非「via NVLink」，单步通信可能从 **82ms 暴涨到 20s**，慢两个数量级——这正是 `NCCL_SOCKET_IFNAME` / `NCCL_P2P_DISABLE` 配错的典型后果。计算/通信重叠原理见 [[llm-optimizer/计算通信重叠]]。

**手算 B：`CUDA_VISIBLE_DEVICES` 限卡对单卡显存的影响**

8 卡训练 70B 模型，参数 FP16 = $70\times10^9\times2 = 140\ \text{GB}$。
- 纯数据并行：每卡都要放整份 140GB → 单卡（80GB）**放不下**。
- 用张量并行把模型切到 8 卡（见 [[llm-inference/大模型推理张量并行]]）：每卡参数 $140/8 = 17.5\ \text{GB}$，可行。

这说明：`CUDA_VISIBLE_DEVICES` 决定了「这个进程有几张卡参与切分」，直接改变每卡显存账本——不是单纯「跑得快慢」，而是「能不能跑起来」。

**手算 C：显存碎片为何触发假性 OOM**

单卡 80GB，已用 60GB，剩 20GB 但被切成 40 个不连续 ~0.5GB 空洞。要分配 3GB 连续激活张量：
- 最大连续空闲块 ≈ 0.5GB < 3GB → 分配失败 → 报 OOM。
- 设 `expandable_segments:True` 后，分配器可把段扩展为更大连续区，3GB 得以放下。
- 量级直觉：碎片可能「浪费」掉 10%~20% 的显存（80GB 卡白白损失 8~16GB），这就是调 `PYTORCH_CUDA_ALLOC_CONF` 的收益来源。

## 10. 常见问题

| 问题 | 原因 | 处理 |
|------|------|------|
| `cuda:0` 跑到了意料外的物理卡 | 默认 `FASTEST_FIRST` 排序 | `export CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| 设了 `CUDA_VISIBLE_DEVICES` 没生效 | 在 `import torch`/首个 CUDA 调用**之后**才设 | 必须在最前面设，或用 `KEY=VALUE python ...` |
| 单卡只想用部分卡却 OOM | 没限卡，进程仍按 world_size 调度 | 正确设 `CUDA_VISIBLE_DEVICES` + `WORLD_SIZE` |
| 多机训练 hang 在初始化 | 网卡/IB 选错或被禁 | `NCCL_DEBUG=INFO` 看停在哪；查 `NCCL_SOCKET_IFNAME` |
| 通信奇慢 | 掉到 socket/PCIe 慢路径 | 日志确认链路；检查 `NCCL_P2P_DISABLE`/`IB_DISABLE` |
| 报错堆栈指向无关代码 | kernel 异步，错误延迟 | `CUDA_LAUNCH_BLOCKING=1` 复现定位（仅调试） |
| 有空闲显存却 OOM | 碎片化 | 调 `PYTORCH_CUDA_ALLOC_CONF`（`expandable_segments` 等） |
| CPU 占满、数据加载抖 | 多进程 OpenMP 超订 | 设 `OMP_NUM_THREADS` 合理值 |
| 强制 CPU 调试 | — | `CUDA_VISIBLE_DEVICES=""` |
| 改了环境变量行为没变 | 库已初始化，变量被冻结 | 重启进程；确认设置时机在初始化前 |

护栏提醒：本文讲**稳定原理与机制**；各变量的**精确取值、默认值、平台差异以 NVIDIA / PyTorch 官方文档为准**，不同版本可能调整。

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 硬件与计算：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 通信与并行：[[ai-infra/网络/集合通信原语]] · [[llm-inference/大模型推理张量并行]] · [[llm-optimizer/计算通信重叠]]
- 框架配置：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[llm-train/README]]
- 精度与显存：[[llm-base/FP16-BF16]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-algo/FLOPs]]
- 监控排错：[[llm-base/nvidia-smi]] · [[llm-base/monitor]]
