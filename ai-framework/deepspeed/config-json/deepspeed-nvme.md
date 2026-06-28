# DeepSpeed NVMe Offload（ZeRO-Infinity NVMe 卸载）

> 把"放不进 GPU 显存、也放不进 CPU 内存"的模型状态，进一步外溢到本地 NVMe SSD，用磁盘换显存，在少量 GPU 上训练超大模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] [[ai-framework/deepspeed/config-json/README]] [[llm-optimizer/kv-cache]] [[ai-infra/ai-hardware/README]]

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点 | 磁盘当显存 |
| 1 | 它解决什么问题（GPU 显存墙） | 模型状态、显存预算 |
| 2 | 内存层级与卸载路径 | GPU↔CPU↔NVMe |
| 3 | ZeRO-Infinity 的核心机制 | 切分 + 异步搬运 |
| 4 | 数据搬运引擎（AIO / libaio） | 直接 I/O、流水线 |
| 5 | 配置项逐个讲（offload_param / offload_optimizer） | nvme_path、buffer |
| 6 | aio 块配置（深层参数） | block_size、queue_depth |
| 7 | 调参 / 选盘 / 容量估算 | 带宽、寿命 |
| — | 完整配置示例 | JSON 模板 |
| — | 常见坑 | 表格 |

## 0. 一句话锚点

**NVMe Offload = ZeRO Stage 3 的"第三级存储"。** ZeRO 把模型状态切碎；当 CPU 内存也装不下时，DeepSpeed 把这些碎片继续写到本地 NVMe SSD，需要计算时再异步读回 GPU。它属于 **ZeRO-Infinity** 论文提出的能力，仅在 **ZeRO Stage 3** 下可用。

一句口诀：**显存不够 → 溢到内存 → 内存不够 → 溢到硬盘。** 代价是吞吐下降，收益是"单机也能跑万亿参数级别的模型"。

## 1. 地基：它解决什么问题——GPU 显存墙

训练时，每个参数对应一组"模型状态（model states）"。以混合精度 + Adam 为例，单个参数的常见显存账（量级，具体以实现为准）：

| 组成 | 精度 | 字节/参数（量级） |
|------|------|------------------|
| FP16 参数 | 2B | 2 |
| FP16 梯度 | 2B | 2 |
| FP32 参数副本（优化器主权重） | 4B | 4 |
| Adam 动量 momentum | 4B | 4 |
| Adam 方差 variance | 4B | 4 |
| 合计 | — | **约 16 B/参数** |

所以一个 **10B 参数**的模型，仅"模型状态"就要约 $10\times10^9 \times 16\text{B} \approx 160\,\text{GB}$，已远超单卡 80GB。再加上激活值（activations），单卡根本放不下。

**ZeRO（Zero Redundancy Optimizer）** 的思路：这些状态原本在每张卡上各存一份（冗余），改成"切分到 N 张卡，每卡只存 1/N"。

- **Stage 1**：切分优化器状态（动量/方差/FP32 副本）
- **Stage 2**：再切分梯度
- **Stage 3**：再切分参数本身

但即便切到 N 卡，N 不够大时总量仍超显存。**ZeRO-Infinity** 再加一招：把切片不仅放 GPU，还能放 **CPU 内存**，乃至 **NVMe SSD**。这就是 NVMe Offload。

```
              模型规模 →
              ┌──────────────────────────────────────────┐
单卡 GPU      │ 装不下                                     │
ZeRO-3 多卡   │ ████ 装得下（受卡数 N 限制）               │
+ CPU offload │ ████████ 更大（受机器内存限制）            │
+ NVMe offload│ ████████████████ 最大（受 SSD 容量限制）  │
              └──────────────────────────────────────────┘
                                       用吞吐换容量 →
```

## 2. 内存层级：卸载到底卸到哪

把存储想象成一座"金字塔"，越往下越大越慢越便宜：

```
   容量小 / 极快        ┌──────────────┐
                       │  GPU HBM      │  ~数十 GB,  ~1–3 TB/s
                       │  (显存)        │
                       └──────┬───────┘
                              │ PCIe / NVLink
   容量中 / 快         ┌──────┴───────┐
                       │  CPU DRAM     │  ~数百 GB,  ~数十–100 GB/s
                       │  (主机内存)    │
                       └──────┬───────┘
                              │ PCIe (NVMe)
   容量大 / 较慢       ┌──────┴───────┐
                       │  NVMe SSD     │  ~TB 级,    ~数 GB/s
                       │  (本地磁盘)    │
                       └──────────────┘
```

NVMe Offload 做的事：让**最慢最大的那一层**也参与"存模型状态"。
- `offload_param` → 把**参数**切片放到 CPU 或 NVMe
- `offload_optimizer` → 把**优化器状态**放到 CPU 或 NVMe

> ⚠️ 关键约束：**NVMe 卸载只对 ZeRO Stage 3 有效**；CPU 卸载优化器可用于 Stage 1/2/3，CPU 卸载参数只用于 Stage 3。

## 3. ZeRO-Infinity 核心机制：切分 + 按需异步搬运

为什么"把参数放硬盘"还能算得动？核心是**计算时只需要当前那一层的参数**，而不是整张网络。流程是一个不断"取出—算—放回"的流水线。

以 ZeRO-3 + NVMe 的一次前向为例（按层 / 参数组遍历）：

```
   for 每一层 layer_i:
   ┌────────────────────────────────────────────────────────┐
   │ 1) prefetch:  从 NVMe / CPU 异步读 layer_i 的参数分片     │
   │ 2) all-gather: 各卡把分片拼成完整 layer_i 权重(临时)       │
   │ 3) compute:   GPU 做 layer_i 的前向                       │
   │ 4) release:   算完立刻丢弃完整权重，只留自己的分片          │
   └────────────────────────────────────────────────────────┘
        ▲          ▲                                  ▲
     I/O 线程   通信(NCCL)                          计算(GPU)
     与计算重叠，让磁盘读取藏在上一层的计算时间里
```

关键设计点：

1. **流水线重叠（overlap）**：在算第 $i$ 层时，后台 I/O 已经在预取（prefetch）第 $i{+}1$ 层。理想情况下磁盘延迟被计算"盖住"，吞吐不至于崩。
2. **分块搬运（buffer）**：不是一次搬完整模型，而是用固定数量的"搬运缓冲区"轮转复用，控制峰值内存。
3. **优化器在 CPU 上算**：NVMe 卸载优化器时，常配合 DeepSpeed 的 **CPU Adam（融合实现）**，优化器更新在 CPU 完成，避免把巨大的优化器状态搬回 GPU。

> 本质：用 $\text{磁盘带宽} \times \text{重叠效率}$ 去逼近"假装这些数据一直在 GPU 上"。重叠做得好，吞吐损失小；磁盘太慢或重叠失败，训练就会被 I/O 拖死。

## 4. 数据搬运引擎：异步 I/O（aio）

GPU 不能直接读硬盘文件；DeepSpeed 内置一个 **异步 I/O 子系统**（在 Linux 上通常基于 `libaio`），负责高效地在 NVMe ↔ 固定内存（pinned memory）之间搬数据。要点：

- **Direct I/O（直接 I/O）**：绕过操作系统的页缓存（page cache），避免"数据被缓存进 DRAM 又占内存"，让带宽更可预测。
- **pinned memory（锁页内存）**：搬运缓冲区用不可换出的固定内存，GPU↔CPU 的 DMA 拷贝才能高速进行。`pin_memory: true` 即开启。
- **深的队列深度 + 多线程**：NVMe 是高并发设备，必须同时投递很多 I/O 请求（queue depth）才能跑满带宽。

```
   NVMe SSD ──libaio(direct, queue_depth)──> pinned DRAM ──DMA──> GPU HBM
       磁盘                  ↑                    ↑              显存
                       block_size            单线程/多线程
                       (一次读写多大)         (overlap_events)
```

> DeepSpeed 提供过一个"自动找最优 aio 参数"的基准工具（在源码 `csrc`/`op_builder` 相关目录下，名称以官方仓库为准），用于在你的盘上扫出最佳 `block_size`/`queue_depth`，避免手工瞎调。

## 5. 配置逐项讲（重含义，不背默认值）

NVMe 卸载在 `zero_optimization` 里通过两个子块开启。**下面强调每个键"做什么、怎么权衡"，具体默认值以官方文档/源码为准。**

### 5.1 `offload_param`（卸载参数，仅 Stage 3）

```json
"offload_param": {
  "device": "nvme",
  "nvme_path": "/local_nvme",
  "pin_memory": true,
  "buffer_count": 5,
  "buffer_size": 1e8,
  "max_in_cpu": 1e9
}
```

| 键 | 作用 | 权衡 |
|----|------|------|
| `device` | 卸载目标：`cpu` / `nvme`（不写则不卸载） | `nvme` 才会写盘；填 `cpu` 只到内存 |
| `nvme_path` | NVMe 上的工作目录 | 必须指向**本地 NVMe 盘**，别指向网络盘/机械盘 |
| `pin_memory` | 中转缓冲是否用锁页内存 | `true` 搬运快，但占用不可换出的物理内存 |
| `buffer_count` | 搬运缓冲区个数 | 越多越能重叠、吃内存越多 |
| `buffer_size` | 单个缓冲区字节数 | 大缓冲提高单次 I/O 效率，但增大内存峰值 |
| `max_in_cpu` | 允许常驻 CPU 的参数量上限 | 给"热"参数留一层 DRAM 缓存，减少回盘 |

### 5.2 `offload_optimizer`（卸载优化器状态）

```json
"offload_optimizer": {
  "device": "nvme",
  "nvme_path": "/local_nvme",
  "pin_memory": true,
  "buffer_count": 4,
  "fast_init": false
}
```

| 键 | 作用 | 权衡 |
|----|------|------|
| `device` | `cpu`（Stage 1/2/3）或 `nvme`（仅 Stage 3） | NVMe 省内存最多，吞吐损失也最大 |
| `nvme_path` | 优化器状态写盘目录 | 可与参数用不同盘分散 I/O 压力 |
| `pin_memory` | 同上 | 同上 |
| `buffer_count` | 优化器状态搬运缓冲数 | 同上 |
| `fast_init` | 加速 NVMe 卸载下的初始化 | 加速建表/分配，行为以官方文档为准 |

> 经验顺序：**先只卸优化器 → 仍不够再卸参数 → 仍不够再都上 NVMe。** 优化器状态最大（约占模型状态的 3/4），优先卸它收益最高，对吞吐影响相对小。

### 5.3 `aio`（顶层的异步 I/O 引擎参数，见第 6 节）

## 6. `aio` 块：决定磁盘能不能跑满

`aio` 是与 `zero_optimization` 平级的顶层配置，控制底层 I/O 行为：

```json
"aio": {
  "block_size": 1048576,
  "queue_depth": 8,
  "thread_count": 1,
  "single_submit": false,
  "overlap_events": true
}
```

| 键 | 含义 | 调大/调小的影响 |
|----|------|----------------|
| `block_size` | 单次读写的块大小 | 大块吞吐高、延迟高；小块相反 |
| `queue_depth` | 同时在途的 I/O 请求数 | 越深越能压满 NVMe 并发；过深增加内存与抖动 |
| `thread_count` | I/O 工作线程数 | 多盘/多控制器时多线程更满带宽 |
| `single_submit` | 是否一次性提交 | 影响提交开销与延迟分布 |
| `overlap_events` | 是否让 I/O 与计算事件重叠 | `true` 才能把磁盘延迟藏进计算里 |

> 这些值**强依赖具体盘型**，建议用官方基准脚本实测，而不是抄网上的数字。

## 7. 实践：选盘、估容量、判断值不值

**1）只有本地 NVMe 才有意义。** 卸载是延迟敏感的随机/顺序混合 I/O，必须是**直连本机的 NVMe SSD**。网络存储（NFS）、机械盘、甚至共享存储都会让训练慢到不可用。

**2）容量估算（量级）。** 全部卸到 NVMe 时，磁盘要装下约 16B/参数 的模型状态：

$$
\text{NVMe 占用} \approx N_{\text{params}} \times 16\,\text{B}
$$

10B 模型 ≈ 160GB；70B 模型 ≈ 1.1TB。**给磁盘留足余量并预留临时文件空间。**

**3）寿命与发热。** 训练会对 SSD 产生持续大量写入，要关注 SSD 的写入耐久（TBW/DWPD）与散热，消费级盘长期满载可能掉速或折寿。

**4）吞吐预期。** NVMe 卸载几乎必然降低 step 吞吐——这是设计上的取舍：**它的价值是"让本来跑不起来的训练能跑起来"，而不是更快。** 如果显存其实够用，别开。

```
   决策树：
   显存够? ──是──> 不要卸载（最快）
      │否
   多加几张卡能装下? ──是──> 用 ZeRO-3 多卡，不卸 NVMe
      │否
   CPU 内存能装下? ──是──> 卸到 CPU（offload device=cpu）
      │否
   有本地 NVMe? ──是──> 卸到 NVMe（device=nvme，仅 Stage 3）
      │否
   降模型 / 加机器 / 减优化器状态（如 8-bit / paged 优化器）
```

## 完整配置示例（说明性，数值需按盘实测）

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme/optim",
      "pin_memory": true,
      "buffer_count": 4
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme/param",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },

  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  }
}
```

几个与 NVMe 体验强相关的 stage3 子项（含义层面）：

- `stage3_prefetch_bucket_size`：预取参数的桶大小，越大重叠越好、内存峰值越高。
- `stage3_max_live_parameters`：同时"在显存里完整存在"的参数上限，控制峰值显存。
- `stage3_param_persistence_threshold`：小于该阈值的参数常驻不卸载，减少小张量频繁回盘的开销。

> 这些键的精确默认值与边界以官方 `config-json` 文档/源码为准，这里只解释"调它影响什么"。

## 常见问题 / 坑

| 现象 | 根因 | 处理 |
|------|------|------|
| 设置了 `device:nvme` 却报错/不生效 | 不是 ZeRO-3 | NVMe 卸载只在 `stage:3` 下可用 |
| 训练极慢，GPU 利用率忽高忽低 | 磁盘被 I/O 拖死、重叠失败 | 用本地 NVMe；增大 `queue_depth`/`buffer`；开 `overlap_events` |
| `nvme_path` 指向 NFS / 机械盘 | 误用慢存储 | 必须本地 NVMe SSD，禁用网络盘 |
| 初始化阶段卡很久 | 创建/分配卸载文件慢 | 试 `fast_init`；确认盘有空间 |
| 物理内存被吃光 | `pin_memory` + 大 `buffer` 占锁页内存 | 调小 `buffer_count`/`buffer_size`，或关 `pin_memory` |
| 磁盘很快写满 | 模型状态 ≈16B/参数 全落盘 | 按 $N\times16\text{B}$ 预留容量并留临时空间 |
| 优化器更新慢 | 优化器在 CPU 上跑 | 用 DeepSpeed 的 CPU/融合 Adam，确认已启用 |
| 多进程争抢同一块盘 | 同机多 rank 写同一 NVMe | 用不同 `nvme_path` 或多盘分摊 I/O |
| 抄了别人的 `block_size` 反而慢 | aio 参数与盘不匹配 | 用官方 aio 基准脚本在本机实测 |

> 一句总结：NVMe Offload 是"显存墙"的最后一道兜底——**它让你能训，但不让你训得快**。在确认更便宜的方案（加卡、降 batch、CPU 卸载、更省的优化器）都不够时再用它，并务必用本地 NVMe + 实测 aio 参数。

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-framework/deepspeed/README]] — DeepSpeed 总览与 ZeRO 各阶段
- [[ai-framework/deepspeed/config-json/README]] — DeepSpeed 配置 JSON 全量参数
- [[llm-optimizer/kv-cache]] — 另一类"用别处的内存换显存"的思路
- [[ai-infra/ai-hardware/README]] — NVMe/PCIe/HBM 等硬件层级
