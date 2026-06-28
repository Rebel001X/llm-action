# Nsight Systems / Nsight Compute：GPU 性能分析与瓶颈定位

> 用 NVIDIA 官方 profiler 把"GPU 利用率低/训练慢/推理慢"从玄学变成可量化的时间线。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/vllm/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

参考视频：NVIDIA 性能分析工具 Nsight Systems/Compute 的使用介绍（简介及分析案例）<https://www.bilibili.com/video/BV15P4y1R7VG>

---

## 阅读地图

| 你想解决的问题 | 看哪一节 |
| --- | --- |
| Systems 和 Compute 到底分工有什么不同 | §1 两把刀 |
| 一次 profiling 在硬件层面发生了什么 | §2 采样原理 |
| 时间线上 CPU/GPU/拷贝/通信怎么读 | §3 时间线四条泳道 |
| 数据加载/预处理慢怎么排查 | §4 checklist（原文真料） |
| 怎么减少同步、用多流、优化拷贝 | §5 优化策略（原文真料） |
| `nsys profile` 每个参数啥意思 | §6 实操命令（原文真料） |
| 抓出来的报告怎么看、看什么指标 | §7 报告判读 + 数值手算 |
| 踩坑 | 常见问题表 |

---

## 0. 一句话锚点

**Nsight = 给 GPU 程序装的"行车记录仪 + 发动机示波器"。**
- **Nsight Systems（nsys）**：宏观时间线，告诉你"时间花在哪"——CPU 在等？拷贝在堵？核函数有空隙？（**找瓶颈**）
- **Nsight Compute（ncu）**：微观单核函数，告诉你"这个 kernel 为什么慢"——算力受限还是访存受限？（**调单核**）

先用 nsys 圈出"哪几个 kernel 占了 80% 时间"，再用 ncu 钻进去优化。**顺序不能反**——脱离时间线去抠单个 kernel 是过早优化。

---

## 1. 地基：两把刀的分工

```
                  你的训练/推理脚本
                         │
        ┌────────────────┴─────────────────┐
        ▼                                   ▼
  Nsight Systems (nsys)              Nsight Compute (ncu)
  ─ 全程时间线                        ─ 单个 kernel 显微镜
  ─ CPU 线程 + CUDA API + GPU kernel  ─ SM 占用率 / 访存带宽
    + 内存拷贝 + NCCL 通信             ─ Roofline / warp stall 原因
  ─ 找"谁占时间、谁有空隙"            ─ 算 vs 访存谁是瓶颈
  ─ 开销小(采样)，可跑真实负载        ─ 开销大(重放kernel)，只对热点跑
        │                                   │
        └──→ 圈出 Top-N 热点 kernel ────────┘
```

**为什么要分两层？** 性能问题有两种完全不同的根因：
1. **编排问题**（GPU 经常空闲、被 CPU/IO/同步卡住）→ nsys 能看到 GPU 泳道里大片空白。
2. **核函数问题**（GPU 一直在跑但效率低）→ GPU 泳道很满，但单 kernel 算力利用率低，得用 ncu 看。

只有 nsys 能区分这两类。如果 GPU 泳道全是空隙，再怎么优化 kernel 都没用——病根在 CPU/IO 侧。

---

## 2. 采样原理：profiler 怎么"看见"GPU

GPU 是异步的：CPU 发射（launch）一个 kernel 后立即返回，GPU 排队执行。profiler 要把这两条异步时间线对齐：

```
CPU 时间线:  ──launch_A──launch_B──[cudaMemcpy 阻塞等待]──────────►
                  │         │
              入队(排队)   入队
                  ▼         ▼
GPU 时间线:  ─────[ kernel_A ][ kernel_B ]──[ 拷贝 ]──────────────►
                  ↑                      ↑
              CUPTI 在每个 kernel 起止打硬件时间戳
```

- **CUPTI（CUDA Profiling Tools Interface）**：英伟达驱动里的回调机制，在每次 CUDA API 调用、每个 kernel 起止、每次拷贝/通信处插入硬件时间戳。这是 nsys 数据的来源。
- **采样（sampling）vs 插桩（instrumentation）**：nsys 主要用低频采样 + CUPTI 事件，**开销小（通常 <5%）**，所以能在接近真实的负载上跑。ncu 用的是 **kernel 重放（replay）**——把一个 kernel 反复执行多遍、每遍测不同硬件计数器，所以**开销大、会让程序慢几十倍**，只能对少数热点 kernel 跑。

> 这就是为什么命令里要用 `--capture-range=cudaProfilerApi`：只抓你关心的那段稳态迭代，而不是把模型加载、第一个 warmup step 的编译开销也录进去污染数据。

---

## 3. 时间线四条泳道怎么读

打开 nsys 报告（`.nsys-rep`），核心是这几条横向泳道，从上到下对齐同一时间轴：

```
时间 ───────────────────────────────────────────────►
OS/CPU线程  │ ▓dataloader▓ │   │ python │     │ python │      ← osrt: 线程在干嘛/在等
CUDA API    │   │ memcpyH2D │launchK1│launchK2│ sync  │       ← 谁发起的、有没有阻塞等待
GPU(SM)     │        │░░空隙░░│ K1 │ K2 │░░░空隙░░░│ K3 │       ← 真正算的时间 + 空隙!!!
Memory      │ ███H2D███ │              │ █D2H█ │               ← 拷贝是否和计算重叠
NVTX        │◄── forward ──►│◄─ backward ─►│◄ optim ►│        ← 你打的业务标记
```

**判读三原则：**
1. **GPU 泳道的空隙（gap）= 浪费**。空隙前如果是 CPU 泳道在忙 → CPU 喂不饱 GPU（dataloader/前后处理瓶颈）。
2. **拷贝（Memory 泳道）和计算（GPU 泳道）是否重叠**。串行 = 没用多流/异步拷贝（见 §5）。
3. **NVTX 标记**：在代码里用 `torch.cuda.nvtx.range_push("forward")` 给阶段打标签，时间线上就能直接读出 forward/backward/optimizer 各占多少 ms。`-t nvtx` 就是开这条泳道。

---

## 4. 排查 checklist（原文真料 + 原理）

> 下面三段 checklist 是原文清单，逐条配上"在 nsys 里怎么看 / 为什么"。

### 4.1 数据加载阶段

| 检查项 | 为什么 / 在 nsys 怎么印证 |
| --- | --- |
| 小文件是否太多，导致文件 io 耗时太长，读取浪费在寻道上 | 机械盘寻道 ~10ms/次，百万小文件能拖垮训练；CPU 泳道大量 IO wait，GPU 泳道空 |
| 存储介质是否已达瓶颈，监控繁忙度，达瓶颈则增加存储缓解读取 | 存储 util 打满 → 换 SSD/NVMe 或增加并行盘 |
| 是否启用多进程并行读取（注意线程争用，监控线程等待是否过长，可用私有线程池） | DataLoader `num_workers>0`；osrt 泳道看 worker 是否在 lock 上长时间等待 |
| 是否启用提前加载（prefetch）实现 CPU 和 GPU 并行 | 让第 N+1 个 batch 的读取与第 N 个 batch 的 GPU 计算重叠，消掉 GPU 空隙 |

### 4.2 数据预处理阶段

| 检查项 | 为什么 |
| --- | --- |
| 是否开启共享内存 `pin_memory`，直接把数据放 pin_memory | 锁页内存才能走 DMA 异步拷贝（见 §5.3），否则 H2D 拷贝会阻塞 |
| 优化 I/O 和网络操作，确保数据以与计算相匹配的速率馈送到 GPU | 喂数速率 ≥ GPU 消费速率，否则 GPU 饿肚子 |
| A100 或以上机器可考虑开启 numa 绑定，缓解争用提升性能 | 跨 NUMA 节点访存延迟翻倍；绑定后 CPU 和它直连的内存/GPU 在同一 socket |

### 4.3 模型训练阶段

| 检查项 | 为什么 |
| --- | --- |
| 是否存在大量 CPU 运算，可用 GPU 实现或去除指定 CPU 设备，尽量让模型跑在 GPU | CPU 算子会在 GPU 泳道里制造空隙（GPU 等 CPU 结果） |
| 是否存在 GPU 利用率不均，尽量不在代码里硬编码指定 GPU 卡 | 多卡训练负载不均 → 快卡等慢卡，整体被拖慢 |
| 复杂、效率低的模块可实现融合的大 GPU 算子提升训练速度 | kernel fusion 减少 launch 次数和中间结果读写（典型：FlashAttention 把多步融成一个 kernel） |
| 避免指标和日志打印太频繁，CPU/GPU 频繁切换导致 GPU 利用率低 | `.item()`/打印会触发隐式同步，强制 GPU 排空、CPU 干等 |
| 是否开启 AMP 提升训练性能 | 混合精度走 Tensor Core，FP16/BF16 吞吐是 FP32 的数倍 |
| 使用最新高性能库和 GPU 驱动，cuda 是否升级到最新版本 | 新版 cuDNN/cuBLAS 常含针对新算子的优化 kernel |

---

## 5. 优化策略（原文真料 + 原理图）

### 5.1 减少不必要的同步

- 尽量减少显式同步调用，如 `cudaDeviceSynchronize`。
- 使用 `cudaStreamWaitEvent` 等事件机制实现更细粒度的同步控制。

**为什么同步要命：** GPU 本是异步流水线，CPU 发完命令就该去准备下一批。一次 `cudaDeviceSynchronize` 会把 CPU 钉在原地、把整条流水线排空：

```
有同步:   CPU: launch ──[█████ 干等 GPU 全部跑完 █████]── 下一步
          GPU:        [ kernel ]                          [新kernel]  ← 中间GPU也空了
                                                          ↑ 流水线断了

无同步:   CPU: launch ─ launch ─ launch ─ (去准备下个batch) ...
          GPU:    [ k1 ][ k2 ][ k3 ] ...   ← 满负荷流水
```

PyTorch 里 `tensor.item()`、`tensor.cpu()`、`print(loss)` 都会触发隐式同步——nsys 时间线上表现为 CUDA API 泳道一根长长的 `cudaStreamSynchronize` 阻塞条。

### 5.2 使用多个流（Stream）

- 将独立的 CUDA 操作分配到不同流，实现并行执行。
- 确保内核启动和内存拷贝尽可能在不同流中并行。

```
单流(默认):  ─[拷贝H2D]─[计算]─[拷贝D2H]─[拷贝H2D]─[计算]─[拷贝D2H]─   串行
多流:        流0 ─[拷贝]─[计算]─[拷贝]─
             流1 ──────[拷贝]─[计算]─[拷贝]─    拷贝与计算重叠，墙钟时间↓
```

同一个流内的操作严格顺序执行；不同流之间可并发。把"拷贝 batch N+1"和"计算 batch N"放不同流，就能让传输隐藏在计算之下。

### 5.3 优化内存拷贝

- 使用异步拷贝 `cudaMemcpyAsync` 并分配到不同流。
- 尽量减少 Host↔Device 拷贝次数，使用统一内存（Unified Memory）或零拷贝（Zero Copy）。

**异步拷贝的前提是锁页内存（pinned memory）**——这就是 §4.2 里 `pin_memory=True` 的意义：

```
可分页内存 + cudaMemcpy   : CPU 先把数据搬到临时锁页区 → 再 DMA → 慢且阻塞
锁页内存   + cudaMemcpyAsync: 直接 DMA，CPU 不参与，可与计算重叠 → 快且非阻塞
```

数值直觉：PCIe 4.0 x16 实测带宽约 25 GB/s。搬一个 batch 的 512MB 激活 ≈ 512/25600 s ≈ **20ms**——如果这 20ms 不和计算重叠，每个 step 白白多花 20ms，1000 steps 就是 20s 纯浪费。

---

## 6. 实操：nsys profile 命令（原文真料）

```bash
export CUDA_VISIBLE_DEVICES=2

nsys profile -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cudabacktrace=true -x true --force-overwrite true \
  -o nsys-sglang-046-14 python3 qwen25_sys.py
```

逐参数解释：

| 参数 | 含义 | 为什么这么设 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES=2` | 只对 2 号卡可见 | 隔离单卡，避免多卡数据互相干扰，报告更干净 |
| `-w true` | `--show-output`，把目标程序的 stdout/stderr 透传到终端 | 一边 profiling 一边能看到脚本日志 |
| `-t cuda,nvtx,osrt,cudnn,cublas` | `--trace`，追踪这些子系统 | cuda=核函数/拷贝；nvtx=你打的业务标记；osrt=OS 运行时(线程/IO)；cudnn/cublas=这两个库的 API |
| `-s cpu` | `--sample=cpu`，对 CPU 调用栈采样 | 定位 CPU 侧热点函数（哪段 Python/算子卡住） |
| `--capture-range=cudaProfilerApi` | 只在程序调用 `cudaProfilerStart()`～`cudaProfilerStop()` 区间内采集 | **关键**：只抓稳态迭代，排除模型加载、首个 step 的编译/warmup 开销 |
| `--capture-range-end=stop` | 遇到 `cudaProfilerStop()` 就结束采集 | 配合上一条，精确圈定区间 |
| `--cudabacktrace=true` | 为 CUDA API 调用记录回溯栈 | 报告里能从一个 kernel 反查是哪行 Python 代码发起的 |
| `-x true` | `--stop-on-exit`，目标程序退出即停止采集并落盘 | 防止程序结束后 profiler 还挂着 |
| `--force-overwrite true` | 覆盖同名旧报告 | 反复迭代调参时免去手动删文件 |
| `-o nsys-sglang-046-14` | 输出报告文件名（前缀） | 生成 `nsys-sglang-046-14.nsys-rep` |
| `python3 qwen25_sys.py` | 被分析的目标命令 | 你的训练/推理脚本 |

> 代码侧配套：在脚本里 `torch.cuda.cudart().cudaProfilerStart()` 和 `...Stop()` 包住要分析的迭代，`--capture-range` 才会生效。用 NVTX 给 forward/backward 打标签则需配合 `torch.cuda.nvtx.range_push/pop`。

抓完后用 GUI（Nsight Systems UI）打开 `.nsys-rep`，或命令行导出统计：`nsys stats nsys-sglang-046-14.nsys-rep`。

---

## 7. 报告判读 + 数值手算

**第一眼看 GPU 利用率（GPU 泳道占满比例）：**

设一个 step 墙钟时间 100ms，时间线里 GPU kernel 实际执行累计 60ms，则
$$\text{GPU 利用率} = \frac{60}{100} = 60\%$$
剩下 40ms 是空隙——去 CPU/Memory 泳道找它们对应在干嘛（dataloader？同步？拷贝没重叠？）。目标是把这 40ms 压下去，而不是急着优化那 60ms 里的 kernel。

**第二眼看 Top kernel 累计耗时**：nsys stats 会给出每类 kernel 的总耗时排名。假设 attention 类 kernel 占 GPU 时间 60ms 中的 30ms（50%），那么换 FlashAttention 把它砍一半 → 省 15ms → step 从 100ms 到 85ms，端到端提速 ~15%。**这就是"先 nsys 圈热点，再 ncu/换实现钻进去"的闭环。**

**第三眼看拷贝与计算是否重叠**：Memory 泳道和 GPU 泳道若完全错开（不重叠），说明没用异步拷贝/多流——按 §5.3 改，可把拷贝时间整段藏进计算。

---

## 常见问题 / 坑

| 现象 | 根因 | 解法 |
| --- | --- | --- |
| 报告里全是模型加载、第一个 step 巨慢的数据 | 没限制采集区间，把 warmup/编译也录了 | 用 `--capture-range=cudaProfilerApi` + 代码里 `cudaProfilerStart/Stop` 包稳态迭代 |
| nsys 报告体积巨大、GUI 打不开 | 采集时间太长、采样太密 | 只采几个 step；缩短被包区间 |
| GPU 利用率看着 90%+ 但训练还是慢 | 利用率高 ≠ 高效，单 kernel 算力利用率可能很低 | 这是 nsys 的盲区，对热点 kernel 上 ncu 看 Roofline/访存瓶颈 |
| 时间线上 CUDA API 泳道一根长阻塞条 | 隐式同步（`.item()`/`print(loss)`/`.cpu()`） | 减少同步，日志降频，必要时异步取回（§5.1） |
| 拷贝怎么都不和计算重叠 | 用了可分页内存 + 同步拷贝 | `pin_memory=True` + `cudaMemcpyAsync` + 分到独立流（§5.3） |
| ncu 一跑程序慢几十倍像卡死 | ncu 用 kernel replay，开销本就极大 | 正常现象；只对少数热点 kernel 跑，用 `-k`/`-s/-c` 限定 |
| 多卡 profiling 数据混乱看不懂 | 多卡时间线交织 | 先 `CUDA_VISIBLE_DEVICES` 锁单卡定位，再看多卡 NCCL 通信 |
| NVTX 泳道空白 | 代码里没打 `nvtx.range_push/pop` 或没加 `-t nvtx` | 两边都要：trace 开 nvtx + 代码打标记 |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件根基：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 通信与多卡（profiling 里看 NCCL 泳道）：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 被 profiling 优化的目标：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-algo/transformer/模型架构]]
- 推理侧落地：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 训练框架（NVTX/同步问题高发处）：[[llm-train/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 指标口径对齐：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
