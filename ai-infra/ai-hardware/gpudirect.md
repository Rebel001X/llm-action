# GPUDirect Storage（GDS）：让存储数据绕过 CPU 直达 GPU 显存

> 一句话定位：GPUDirect Storage 在 NVMe/NVMe-oF 存储与 GPU 显存之间建立**直接 DMA 通路**，砍掉 CPU bounce buffer 这道中转，解决"GPU 算得飞快但 I/O 喂不上"的瓶颈。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]]

---

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | GDS 到底省了哪一跳 | bounce buffer / 直接 DMA |
| 1. 地基/前置 | DMA、PCIe P2P、GPUDirect 家族 | RDMA / P2P / Pinned Memory |
| 2. 传统 I/O 路径的痛点 | 为什么数据非要先过 CPU 内存 | bounce buffer / 双拷贝 |
| 3. GDS 的核心思想 | 让 DMA 引擎直接读写显存 | cuFileRead / cuFileWrite |
| 4. 软件栈拆解 | libcufile.so 与 nvidia-fs.ko 的分工 | 地址替换 / VFS |
| 5. 为什么不能直接用 POSIX | dma_buf 与内核现状 | pread / pwrite 的缺口 |
| 6. 性能优化点 | 动态路由 / NVLink / 异步流 | Magnum IO |
| 实操：API 与软件依赖 | cuFile API、MLNX_OFED、校验 | 真实命令 |
| 常见问题/坑 | 上手前必看 | 对齐 / 文件系统支持 |

---

## 0. 一句话锚点

把数据从硬盘搬到 GPU 算，传统路径是 **存储 → CPU 内存（中转副本）→ GPU 显存**，要拷两次、占 CPU、加延迟。
GDS 把它变成 **存储 →（DMA 直接）→ GPU 显存**，少一跳。

记住一个画面：

```
传统：[NVMe] --DMA--> [CPU RAM: bounce buffer] --cudaMemcpy--> [GPU 显存]
                          ↑ 多一份副本，占 CPU + 内存带宽
GDS ：[NVMe] ----------DMA 直达----------------> [GPU 显存]
                          ↑ NVMe 的 DMA 引擎把目标地址直接写成显存地址
```

---

## 1. 地基/前置

理解 GDS 之前，先把三块地基钉牢。

### 1.1 DMA（直接内存访问）

DMA 引擎是设备（NVMe 控制器、网卡）里的一个"搬运工"，它可以**不打扰 CPU**，自己按给定的内存地址搬数据。关键点：DMA 引擎搬数据时，需要一个**物理地址**作为目标。传统上这个地址指向 CPU 内存；GDS 的本质就是让这个目标地址**指向 GPU 显存**。

### 1.2 PCIe Peer-to-Peer（P2P）

GPU、NVMe、网卡都挂在 PCIe 总线上。P2P 指**两个 PCIe 设备之间直接传数据**，不经过 CPU 根复合体的内存中转。GDS 依赖的就是 NVMe ↔ GPU 的 P2P DMA。

```
      ┌──────────── PCIe Switch / Root Complex ────────────┐
      │                                                     │
  [ GPU ]            [ NVMe SSD ]            [ NIC ]
      └────── P2P DMA（GDS 走这条）──────┘
```

> 注意：P2P 是否高效取决于拓扑——同一个 PCIe Switch 下的 GPU 与 NVMe 走 P2P 最快；如果要跨 CPU socket（经过 QPI/UPI），延迟和带宽都会打折。

### 1.3 GPUDirect 家族对照

GDS 不是孤立技术，它是 GPUDirect 家族里"对接存储"的那一支：

| 成员 | 直连对象 | 解决的问题 |
| --- | --- | --- |
| GPUDirect P2P | GPU ↔ GPU（同机） | 显存间直接拷贝，不过 CPU |
| GPUDirect RDMA | GPU ↔ NIC | 网卡与显存直接收发，提升网络带宽/延迟 |
| **GPUDirect Storage（GDS）** | **GPU ↔ NVMe / NVMe-oF** | **存储与显存直接传，本笔记主角** |

原文一句话点透了它的来历：**正如 GPUDirect RDMA 在 NIC 与 GPU 内存之间直接移动数据改善了带宽和延迟，GDS 让本地或远程存储（NVMe / NVMe-oF）与 GPU 显存之间直接移动数据。** GDS 可以看作"把 RDMA 的思路套到存储 I/O 上"。

---

## 2. 传统 I/O 路径的痛点：bounce buffer 这道中转

在 GPU 加速系统里，**所有 I/O 历来都由 CPU 控制**。原文讲得很直白：所有 IO 操作都会先经过主机端，需要 CPU 指令把数据传到主机内存里，然后才到达 GPU。

CPU 通常通过 **bounce buffer**（弹跳缓冲区）来实现传输——它是**系统内存中的一块区域，数据在传到 GPU 之前会在这里存一个副本**。

```
应用 read() → 内核 VFS → 文件系统 → NVMe DMA
                                      │
                                      ▼
            ┌────────── CPU 系统内存 ──────────┐
            │     bounce buffer（中转副本）     │   ← 第 1 跳：磁盘 → 这里
            └──────────────┬───────────────────┘
                           │ cudaMemcpy(H2D)        ← 第 2 跳：这里 → 显存
                           ▼
                     ┌──── GPU 显存 ────┐
```

这种中转的代价（原文点名的三宗罪）：

1. **额外延迟**——多走一段拷贝，整条链路变长。
2. **额外内存消耗**——bounce buffer 要占系统内存。
3. **占用 CPU 资源**——CPU 要为这次拷贝忙活，本该干别的活。

**为什么这越来越致命？** 原文给了趋势判断：随着 AI/HPC 数据集越来越大，加载数据的时间开始压制整体性能；计算从较慢的 CPU 转移到更快的 GPU 后，**I/O 越来越成为整体性能瓶颈**。一句话——**快的 GPU 被慢的 I/O 拖累，GPU 利用率显著下降**。这正是 GDS 要解决的问题。

### 数值直觉（手算）

假设一个数据块 4 GiB，要从 NVMe 喂进显存。

- PCIe 4.0 x4 NVMe 实测顺序读 ≈ 7 GB/s；PCIe 4.0 x16 上 GPU H2D 拷贝 ≈ 25 GB/s。
- **传统两跳**：磁盘→内存 $\;4/7 \approx 0.571\,\text{s}$，内存→显存 $\;4/25 = 0.16\,\text{s}$，且第二跳吃 CPU/内存带宽，合计 $\approx 0.73\,\text{s}$（还没算 CPU 调度抖动）。
- **GDS 一跳**：磁盘→显存 $\;4/7 \approx 0.571\,\text{s}$，且**不占 CPU、不占系统内存带宽**。

省的不只是 $0.16\,\text{s}$ 这一段——更重要的是把 CPU 和系统内存带宽**完全解放出来**，在多 GPU、多流并发时收益被成倍放大。

---

## 3. GDS 的核心思想：把 DMA 目标地址改成显存地址

GDS 的根本动作只有一句话：**让 NVMe / 网络适配器的 DMA 引擎，直接把数据搬进 GPU 显存，而不是先搬进 CPU 内存。**

原文给出了 GDS 暴露给应用的新 API：

> GDS 解决方案涉及新的 API：**cuFileRead** 或 **cuFileWrite**，它们与 POSIX 的 **pread / pwrite** 类似。

也就是说，开发者把熟悉的 `pread(fd, buf, ...)` 换成 `cuFileRead(handle, devPtr, ...)`，其中 `devPtr` 直接是 **GPU 显存指针**。语义对齐了 POSIX，迁移成本低，但数据落点从 CPU 内存变成了显存。

```
传统：pread (fd, cpu_buf, size, offset)    →  数据落在 CPU 内存
GDS ：cuFileRead(handle, gpu_devPtr, size, offset)  →  数据直达 GPU 显存
```

---

## 4. 软件栈拆解：libcufile.so 与 nvidia-fs.ko 的"地址接力"

GDS 最精妙的地方在于：**Linux 内核的 VFS（虚拟文件系统）目前不接受把 GPU 显存地址作为 DMA 目标传下去**——原文说这会"导致错误情况"。NVIDIA 用一招"地址接力"绕过了这个限制。

原文描述的两层替换，是整个 GDS 的灵魂，逐字保留并拆解：

> 当使用 cuFileRead 或 cuFileWrite 等 cuFile API 时，**libcufile.so 用户级库捕获 GPU 缓冲区地址并替换传递给 VFS 的代理 CPU 缓冲区地址**。就在缓冲区地址用于 DMA 之前，启用 GDS 的驱动程序对 **nvidia-fs.ko** 的调用会识别 CPU 缓冲区地址并**再次提供替代 GPU 缓冲区地址**，以便 DMA 可以正确进行。

拆成两次"地址掉包"：

```
① 用户态：libcufile.so
   应用给的是 GPU 显存地址 ──掉包──> 一个"代理 CPU 缓冲区地址"
   （因为 VFS 只认 CPU 地址，先骗它过关）
        │
        ▼  代理 CPU 地址一路下穿 VFS → 文件系统 → 块层
        │
② 内核态：nvidia-fs.ko（GDS 内核驱动）
   就在 DMA 真正发起前 ──掉包回来──> 真正的 GPU 显存地址
   于是 NVMe 的 DMA 引擎拿到显存地址，直接写显存
```

为什么要这么"绕"？因为原文点明的根本矛盾：**当前 Linux 实现的根本问题，是无法把 GPU 缓冲区地址作为 DMA 目标穿过 VFS 传下去**。NVIDIA 的解法就是"对上层传 CPU 地址骗过 VFS，对下层 DMA 换回 GPU 地址做真活"——一进一出两次替换。

软件栈分层图（对应原文"图 2：GDS 软件堆栈"）：

```
┌──────────────────────────────────────────────┐
│ 应用程序：cuFileRead / cuFileWrite             │  ← Magnum IO 抽象的入口
├──────────────────────────────────────────────┤
│ libcufile.so（用户态库）                        │
│   · 捕获 GPU 地址 → 替换成代理 CPU 地址          │
│   · 执行优化：动态路由 / 预固定缓冲 / 对齐        │
├──────────────────────────────────────────────┤
│ VFS → 文件系统 → 块设备层（内核标准路径）         │  ← 仍走标准 Linux I/O 栈
├──────────────────────────────────────────────┤
│ nvidia-fs.ko（GDS 内核驱动）                    │
│   · DMA 前识别代理 CPU 地址 → 换回真 GPU 地址     │
├──────────────────────────────────────────────┤
│ 支持 GDS 的存储驱动 → NVMe / NVMe-oF DMA 引擎    │  ← DMA 直达显存
└──────────────────────────────────────────────┘
```

原文把这套设计归为 **Magnum IO 灵活抽象架构原则**的一个范例：cuFile API 作为统一入口，让平台层去做特定优化（选择性缓冲、NVLink 的使用），上层应用不必关心底层走哪条路。

---

## 5. 为什么不能直接用 POSIX？dma_buf 与内核现状

一个自然的疑问：既然有 `pread/pwrite`，为何还要造 cuFile？原文给了完整答案，逐条保留：

| 现状/方案 | 说明 |
| --- | --- |
| POSIX `pread`/`pwrite` | 提供**存储与 CPU 缓冲区之间**的复制，但**尚未启用到 GPU 缓冲区的复制**——这是缺口所在 |
| Linux 内核 | 目前**不支持把 GPU 缓冲区作为 DMA 目标**，缺陷会随时间逐步解决 |
| `dma_buf`（开发中） | 一种正在开发的上游方案，可在 PCIe 总线上的对等设备（NIC/NVMe ↔ GPU）之间直接复制 |
| 厂商替代方案 | 在上游方案普及前，许多供应商提供支持 GDS 的方案，如 **MLNX_OFED**（Mellanox OFED 驱动栈） |

原文的态度很务实：**GDS 的性能提升空间太大，无法等待上游方案传播给所有用户**，所以先用 cuFile + nvidia-fs.ko 这套厂商方案落地。

而且 NVIDIA 认为 cuFile 不会因为内核补齐而被淘汰——原文给了理由：**动态路由、NVLink 的使用，以及只能从 GDS 获得的用于 CUDA 流的异步 API** 等优化，使 cuFile API 成为 CUDA 编程模型的**持久特性**，即便 Linux 文件系统缺陷被修复后也是如此。

---

## 6. GDS 带来的关键优化（cuFile 独有价值）

libcufile.so 内部承担了一系列优化（原文逐项列出，这里补"为什么有用"）：

| 优化 | 是什么 | 为什么重要 |
| --- | --- | --- |
| 动态路由（dynamic routing） | 运行时按拓扑选最优数据通路 | 跨 socket/Switch 时自动避开慢路径，保证带宽 |
| 预固定缓冲区（pre-pinned buffer） | 提前把显存页"钉住"（pin），免去每次传输前的固定开销 | DMA 必须用 pinned 内存；预固定省掉重复 pin/unpin 延迟 |
| 对齐（alignment） | 让 I/O 偏移/长度满足设备/页对齐 | 不对齐会触发额外读改写或回退到 bounce buffer，性能骤降 |
| NVLink 的使用 | 多 GPU 间走 NVLink 而非 PCIe | NVLink 带宽远高于 PCIe，跨 GPU 分发数据更快 |
| 用于 CUDA 流的异步 API | I/O 可挂到 CUDA stream 上异步执行 | 计算与 I/O 重叠，把 I/O 时间"藏"进计算里，**这是 GDS 独有、POSIX 给不了的** |

> 直觉：前面数值示例省的是"一跳拷贝"；这里的异步流优化省的是"等待时间"——让加载下一批数据和当前批次计算同时进行，进一步把 I/O 从关键路径上挤掉。

---

## 实操：API、软件依赖与校验

> 以下整理原文出现的真实组件与官方文档，便于上手；未列出的具体命令请以下方官方链接为准，不臆造参数。

### 关键 API（cuFile）

| API | 对应 POSIX | 作用 |
| --- | --- | --- |
| `cuFileRead(handle, devPtr, size, file_offset, devPtr_offset)` | `pread` | 从文件读到 **GPU 显存** |
| `cuFileWrite(handle, devPtr, size, file_offset, devPtr_offset)` | `pwrite` | 从 **GPU 显存** 写到文件 |

典型调用骨架（伪代码，体现"显存指针直接当 I/O 缓冲"）：

```c
// 1) 注册文件句柄
cuFileHandleRegister(&cf_handle, &descr);
// 2) 注册 GPU 显存缓冲（让 GDS 预固定/记录）
cuFileBufRegister(devPtr, size, 0);
// 3) 直接读入显存——数据不经过 CPU bounce buffer
ssize_t n = cuFileRead(cf_handle, devPtr, size, file_offset, 0);
```

### 软件依赖

- **libcufile.so**：用户态 cuFile 库（地址捕获 + 优化逻辑）。
- **nvidia-fs.ko**：GDS 内核驱动（DMA 前换回真 GPU 地址）。需随 NVIDIA 驱动/CUDA Toolkit 一并安装并 `insmod` 加载。
- **MLNX_OFED**：在上游 `dma_buf` 普及前，厂商提供的支持 GDS 的栈之一（尤其用于 NVMe-oF / RDMA 场景）。

### 官方文档（原文给出，原样保留）

- 设计指南：https://docs.nvidia.com/gpudirect-storage/design-guide/index.html
- 概览指南：https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- 入门指南：https://docs.nvidia.com/gpudirect-storage/getting-started/index.html

---

## 常见问题/坑

| 现象 / 坑 | 原因 | 应对 |
| --- | --- | --- |
| 开了 GDS 但带宽没提升，甚至更慢 | I/O 偏移/长度未对齐，GDS **回退到 bounce buffer 兼容路径** | 保证按设备/页边界对齐（对齐是 libcufile 的核心优化项之一） |
| 文件系统不支持 GDS | 并非所有 FS 都接入了 GDS 路径 | 用官方支持的文件系统（参考 design-guide），否则自动走 POSIX 兼容路径 |
| 跨 socket 时带宽腰斩 | NVMe 与 GPU 分属不同 CPU socket，P2P 需跨 UPI | 让 GPU 与 NVMe 在同一 PCIe Switch/同一 NUMA 节点；依赖动态路由择优 |
| nvidia-fs.ko 未加载 | 内核驱动缺失，cuFile 调用静默回退 | 确认 `nvidia-fs` 模块已编译并加载，匹配当前 NVIDIA 驱动版本 |
| 把 GDS 当作"网络方案" | GDS 管**存储↔显存**，GPUDirect **RDMA** 才管**网卡↔显存** | 分清家族成员（见 §1.3），NVMe-oF 远程存储场景两者常配合 |
| 显存缓冲未注册导致每次都慢 | 未 `cuFileBufRegister`，每次传输临时 pin | 预先注册显存缓冲，复用预固定优势 |
| 误以为 cuFile 会被内核补丁淘汰 | 只看到地址替换，忽略了异步流/NVLink/动态路由 | cuFile 是 CUDA 编程模型的**持久特性**，价值不止于绕过内核缺陷 |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 网络方向兄弟技术：[[ai-infra/网络/InfiniBand]]（GPUDirect RDMA 的载体）· [[ai-infra/网络/集合通信原语]]（多 GPU 数据分发）
- 硬件地基：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]]
- 软件生态：[[ai-infra/ai-hardware/AI芯片软件生态]]
- 框架落地：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 推理与显存：[[llm-optimizer/kv-cache]] · [[llm-inference/vllm/README]] · [[docs/transformer内存估算]]
- 性能评估：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
