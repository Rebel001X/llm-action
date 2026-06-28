# NVIDIA Nsight Systems 性能分析

> 系统级时间线剖析器：把 CPU/GPU/通信/IO 画在同一根时间轴上，找出训练与推理的"气泡"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]] · [[llm-optimizer/计算通信重叠]] · [[ai-infra/网络/集合通信原语]] · [[llm-algo/FLOPs]]

## 阅读地图

| 节 | 你将搞清楚 | 关键产物 |
|---|---|---|
| 0 | 一句话锚点 | Nsys = 整机时间线 |
| 1 | 地基：GPU 异步执行模型 | Stream / Queue / 隐藏延迟 |
| 2 | Nsys 在工具栈里的位置 | Systems vs Compute |
| 3 | 采集什么数据 | trace 域：cuda/nvtx/osrt/nccl |
| 4 | 工作流：采集 → 看 → 定位 | `nsys profile` |
| 5 | 时间线怎么读 | 行/泳道含义 |
| 6 | NVTX 打点 | 把代码语义投到时间线 |
| 7 | 典型瓶颈与图谱 | 6 大 anti-pattern |
| 8 | LLM 训练/推理实战 | 重叠、bubble、launch bound |
| 9 | 数值手算 | 利用率 / 带宽 / bubble 占比 |
| 10 | 开销与坑 | 低开销但非零 |

## 0. 一句话锚点

**Nsight Systems（简称 nsys）是一个"低开销、全系统、时间线"剖析器**：它不告诉你某个 kernel 内部 warp 为什么慢（那是 Nsight **Compute** 的活），而是告诉你**整台机器在每一微秒在干什么**——CPU 线程、CUDA API 调用、GPU kernel、内存拷贝、NCCL 通信、OS 调度、IO，全部对齐到同一根时间轴。

> 核心价值：**发现"GPU 在空转"**。GPU 很贵，只要它有一段时间没在算，就是钱在烧。Nsys 让"空转"肉眼可见。

```
            ┌───────────────── 一根全局时间轴 ─────────────────┐
 CPU 主线程  │ py │ launch │ launch │           │ py  │ launch │
 CUDA API    │    │ Memcpy │ K1     │ K2        │     │ K3     │
 GPU (SM)    │        ░░░░ │███K1███│██████K2██████│  ░░░  │█K3█│
 NCCL        │                     │  AllReduce  │            │
             └──────────────────────────────────────────────┘
                ░ = GPU 空转(气泡)   █ = GPU 在算
```

## 1. 地基：先理解 GPU 的异步执行模型

不懂异步，就看不懂时间线。CPU 调用 CUDA API（如 `kernelLaunch`）是**异步**的：CPU 把任务"丢进"GPU 的 **Stream（流，本质是一个 FIFO 队列）** 后立刻返回，GPU 在后台按顺序执行。

```
 CPU 侧                          GPU 侧
 ┌──────────┐   enqueue   ┌─────────── Stream 0 (FIFO) ───────────┐
 │ launch K1│ ──────────▶ │ [K1][K2][Memcpy][K3] ...              │
 │ launch K2│ ──────────▶ │   ▲ GPU 从队头依次取出执行            │
 │ (CPU 继续)│             └───────────────────────────────────────┘
 └──────────┘
   CPU 不等 K1 算完就发 K2 —— 这叫"延迟隐藏"
```

由此引出 nsys 要回答的两个根本问题：

- **GPU 队列空了吗？** 空了 = GPU 等 CPU 喂数据 → **CPU/launch bound（受限于 CPU）**。
- **CPU 喂得过来但 GPU 还是慢？** → **GPU/compute bound（受限于算力或访存）**，下一步用 Nsight Compute 钻进 kernel。

| 概念 | 含义 | 在时间线上看什么 |
|---|---|---|
| Stream | GPU 的任务队列 | 同 stream 串行，不同 stream 可并行 |
| 异步 | CPU launch 后立即返回 | CPU 行的 API 块远短于 GPU kernel 块 |
| 隐藏延迟 | CPU 提前排队填满队列 | GPU 行连续无缝 = 健康 |
| 同步点 | `cudaDeviceSynchronize`/`.item()` | CPU 行出现一段长"等待"块 |

## 2. Nsys 在工具栈里的位置：Systems vs Compute

NVIDIA 把性能工具一分为二，**用错工具是新手最大的浪费**：

```
                 你的瓶颈在"系统协同"还是"单个kernel内部"?
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼ 系统级 / 时间线                       ▼ kernel 微观 / 硬件计数器
 ┌─────────────────────┐               ┌──────────────────────────┐
 │ Nsight Systems(nsys)│ 先用这个定位 │ Nsight Compute (ncu)     │
 │ - CPU/GPU/通信/IO   │ ──找到热点──▶ │ - warp 占用率 / 访存效率 │
 │ - 谁在等谁          │               │ - roofline / 指令分布    │
 │ - 低开销, 看全局    │               │ - 高开销, 看一个 kernel  │
 └─────────────────────┘               └──────────────────────────┘
```

经验法则：**永远先 nsys 后 ncu**。先在整机时间线上找出"哪个 kernel/哪段通信吃掉了时间"，再用 ncu 单独 deep-dive 那个 kernel。直接上 ncu 等于在黑暗中乱挖。

## 3. 采集什么：trace 域（domain）

nsys 通过挂接各类"探针"采集事件，常用 trace 域：

| trace 域 | 采的是什么 | 为什么重要 |
|---|---|---|
| `cuda` | CUDA Runtime/Driver API、kernel、memcpy | 主战场，看 GPU 干活 |
| `nvtx` | 你代码里的手动打点 | 把"前向/反向/optimizer"投到时间线 |
| `osrt` | OS runtime（线程、互斥锁、`pthread`、`poll`） | 看 CPU 阻塞在哪个系统调用 |
| `nccl` | 集合通信（AllReduce 等） | 多卡/分布式必看，见 [[ai-infra/网络/集合通信原语]] |
| `cublas`/`cudnn` | 库级调用名 | 知道某 kernel 来自哪个库 |
| `osrt`+采样 | 周期性 backtrace 采样 | 找 CPU 端热点函数 |

> 工具/命令的精确默认值随版本变化，**以官方文档为准**；下面给出稳定的概念性用法。

## 4. 工作流：采集 → 打开 → 定位

```
 ① 采集                        ② 打开               ③ 定位
 nsys profile <你的程序>  ──▶  .nsys-rep 文件  ──▶  GUI 时间线 / CLI stats
   生成报告文件               (在 GUI 中加载)        找气泡、找长 kernel
```

最小可用命令（概念示例，参数以官方为准）：

```bash
# 采集一个 PyTorch 训练脚本，开启 cuda 与 nvtx trace
nsys profile -t cuda,nvtx,osrt,nccl \
             -o report_train \
             python train.py

# 命令行直接出统计摘要(适合无 GUI 的服务器)
nsys stats report_train.nsys-rep
```

实战要点（避免"报告 50GB 打不开"）：

- **限定范围**：长训练别全程采，用 `--capture-range` 配合 `cudaProfilerStart/Stop`，或只采几十个 step。
- **跳过 warmup**：前几个 step 有 cuDNN autotune、JIT 编译、显存分配，时间线会很"脏"，采集稳定态。
- **关 GUI 远程**：服务器上 `nsys profile` 出 `.nsys-rep`，下载到本地 GUI 看。

```
   采集时间窗的选择
   |<-- warmup -->|<-------- 稳定态(只采这段) -------->|
   step0  1  2  3 | 4   5   6   7   8   9  10  11  12 |
   ↑autotune/JIT  | ↑↑↑ 这里的时间线才代表真实性能 ↑↑↑
```

## 5. 时间线怎么读：泳道（row）含义

打开 GUI，从上到下典型泳道：

```
 ┌─ CPU rows ─────────────────────────────────────────────┐
 │ Thread 0 (python)  │ ▓osrt▓ │ cudaLaunchKernel │ ...    │
 │   └ NVTX: [Forward][Backward][Optimizer]  ← 你的语义层  │
 ├─ CUDA API row ─────────────────────────────────────────┤
 │   │cudaMemcpyAsync│cudaLaunchKernel│cudaStreamSync│     │
 ├─ GPU rows ─────────────────────────────────────────────┤
 │ Stream 7 (compute) │██gemm██│██gemm██│  ░░gap░░ │██..██ │
 │ Stream 13 (memcpy) │        │H2D│           │D2H│       │
 │ NCCL               │            │ AllReduce │           │
 └────────────────────────────────────────────────────────┘
        ▲ 关键阅读法：把目光放在 "GPU rows" 上找 ░░gap░░
```

读图三步：

1. **盯 GPU compute 行**：连续无缝 = 好；密集 ░░gap░░ = GPU 在等。
2. **gap 对齐上方**：gap 正下方时刻，CPU 行在干嘛？若 CPU 行也在忙 launch/Python → **CPU 喂不动（launch bound）**；若 CPU 行在 `cudaStreamSynchronize` 等待 → **强制同步打断了流水**。
3. **量 kernel 宽度**：哪个 kernel 块最宽 = 单 kernel 热点 → 记下名字，交给 ncu。

## 6. NVTX：把代码语义投影到时间线

裸时间线只有 `kernel_47`、`gemm_128x256` 这种名字，看不出"这是前向还是反向"。**NVTX（NVIDIA Tools Extension）** 让你在代码里打"区间标签"，nsys 把它画成 GPU 行上方的彩色长条。

```python
import torch.cuda.nvtx as nvtx

for step, batch in enumerate(loader):
    nvtx.range_push("forward")      # ┌── 区间开始
    out = model(batch)
    nvtx.range_pop()                # └── 区间结束

    nvtx.range_push("backward")
    loss = criterion(out, y); loss.backward()
    nvtx.range_pop()

    nvtx.range_push("optimizer")
    optimizer.step(); optimizer.zero_grad()
    nvtx.range_pop()
```

时间线效果：

```
 NVTX:  [   forward   ][   backward    ][ optim ]
 GPU :  ██gemm██ ██..██ ██grad██ ██..██  ██adam██
        └─────────────┘ └──────────────┘ └──────┘
         一眼知道每个阶段花了多少 GPU 时间
```

PyTorch 提供 `torch.profiler` + `emit_nvtx()`，或 `with torch.autograd.profiler.emit_nvtx()` 自动把每个算子名打成 NVTX 区间，配合 `nsys profile` 一起用，时间线上每个 kernel 都带 Python 算子名。

## 7. 六大典型瓶颈图谱

```
① CPU/Launch bound（GPU 饿死）
   CPU : launch launch launch ...(密)
   GPU : █ ░░░ █ ░░░ █ ░░░       ← kernel 太小, CPU launch 跟不上
   药:  增大 batch / 算子融合 / CUDA Graph 减少 launch 次数

② 强制同步（流水被打断）
   CPU : ...... [cudaStreamSynchronize 长等待] ......
   GPU : ████████░░░░░░░░░░░░░░░░░░░░░░████████
   药:  去掉 loss.item()/print(tensor)/.cpu() 在热循环里的调用

③ H2D/D2H 拷贝串行(数据搬运挡路)
   GPU compute: ██░░░░░░██░░░░░░██
   GPU memcpy :   ██H2D██  ██H2D██     ← 拷贝没和计算重叠
   药:  pin_memory + non_blocking + 多 stream 重叠, 见 ↓ 计算通信重叠

④ 通信不重叠(多卡)
   GPU compute: ████░░░░░░░░░░░░████
   NCCL       :     ██AllReduce██        ← 反向算完才通信, 全停等
   药:  梯度分桶 + 反向中逐桶 AllReduce, 见 [[llm-optimizer/计算通信重叠]]

⑤ DataLoader 饿死(IO bound)
   CPU worker : ░░░░读盘解码░░░░ (慢)
   GPU        : ████ ░░░░░░░░░ ████   ← 等下一个 batch
   药:  加 num_workers / prefetch / 预处理缓存

⑥ 单 kernel 过慢(compute bound)
   GPU : ███████████超宽 kernel███████████
   药:  这才该交给 Nsight Compute 钻进去看 occupancy/访存
```

## 8. LLM 训练/推理实战视角

**训练**（数据/张量/流水并行）：nsys 是验证"重叠是否生效"的唯一可信手段。
- ZeRO/FSDP 的 **all-gather 权重** 是否和前向计算重叠？看 NCCL 行 vs GPU compute 行是否错位并行。见 [[ai-framework/deepspeed/README]]。
- 张量并行的 **AllReduce**（每层两次）是否成为瓶颈？见 [[llm-inference/大模型推理张量并行]] 与 [[ai-infra/网络/集合通信原语]]。
- 流水并行的 **bubble**：在 NVTX 上标出 micro-batch，气泡占比一目了然。

**推理**（自回归解码）：
- **Decode 阶段几乎必是 launch/memory bound**：每步只生成 1 token，kernel 极小，GPU 行布满 ░gap，CPU launch 成为瓶颈 → **CUDA Graph** 把整步固化为一次提交。
- **KV-Cache** 访存：看 memcpy/attention kernel 形态，配合 [[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]。
- **FlashAttention** 是否真的在跑融合 kernel（而非朴素 attention 的一串小 kernel）？时间线上一个宽 kernel vs 十几个碎 kernel 立判，见 [[llm-optimizer/FlashAttention]]。

```
 Decode 步的典型病态时间线(launch bound)
 CPU : L L L L L L L L L L L L   ← 每个 token 一堆 launch
 GPU : ▪░▪░▪░▪░▪░▪░▪░▪░▪░▪░▪░    ← 小kernel + 大量气泡
 用 CUDA Graph 后:
 CPU : [G]            [G]         ← 一次 graph replay
 GPU : ▪▪▪▪▪▪▪▪▪▪▪▪  ▪▪▪▪▪▪▪▪    ← 气泡被压扁
```

## 9. 数值手算

### 9.1 GPU 利用率（来自时间线）

利用率不是 `nvidia-smi` 那个粗略数字，而是：

$$\text{Util}_{GPU} = \frac{\sum \text{kernel 实际占用时间}}{\text{墙钟总时长}}$$

设一个 step 墙钟 = 100 ms，时间线上 GPU compute 行累计有活的时间 = 62 ms：

$$\text{Util} = \frac{62}{100} = 62\%\ \Rightarrow\ 38\%\ \text{在烧空转}$$

每张 H100 按 $4/小时 估，38% 空转 ≈ 每卡每小时白烧 $1.5；千卡集群一年浪费量级在七位数美元——**这就是为什么要剖析**。

### 9.2 launch bound 判定（手算）

设单个 kernel GPU 执行 = 8 µs，CPU 发起一次 launch 开销 ≈ 10 µs。

- 若每步发 1 个 kernel：GPU 要 8 µs，CPU 要 10 µs → **CPU 比 GPU 慢，队列必然抽空 → launch bound**。
- 让队列不饿的条件：$\;t_{kernel} \ge t_{launch}\;$，即单 kernel 至少 10 µs。
- 解法：把 N 个小 kernel 融合成 1 个（CPU launch 1 次摊销），或 CUDA Graph 把 N 次 launch 变成 1 次 replay：

$$\text{launch 总开销}: \;N\times 10\,\mu s \;\xrightarrow{\text{CUDA Graph}}\; 1\times 10\,\mu s$$

N=50 时，launch 开销从 500 µs 降到 10 µs，省 490 µs/步。

### 9.3 拷贝是否能被隐藏（带宽手算）

PCIe Gen4 x16 ≈ 单向 32 GB/s（理论），实测约 25 GB/s。搬一个 batch 输入 = 256 MB：

$$t_{copy} = \frac{256\ \text{MB}}{25\ \text{GB/s}} = \frac{0.25}{25}\,s = 10\ \text{ms}$$

若该 batch 的前向计算 = 30 ms，则只要 **拷贝与计算重叠**（不同 stream + `non_blocking`），10 ms 拷贝完全藏在 30 ms 计算之下，时间线上 memcpy 行与 compute 行平行 → 净开销 0。否则串行就是 30+10=40 ms，多花 33%。

### 9.4 通信占比与流水气泡

张量并行：每个 Transformer 层前向 2 次 AllReduce，反向 2 次。设单次 AllReduce（2 卡，hidden 数据）= 0.4 ms，单层前向计算 = 3 ms：

$$\text{通信占比} = \frac{2\times 0.4}{3 + 2\times 0.4} = \frac{0.8}{3.8} \approx 21\%$$

时间线上 NCCL 行若**不与** compute 重叠，这 21% 就是纯停等。流水并行 bubble 占比经典式：

$$\text{bubble fraction} = \frac{p-1}{m + p - 1}$$

$p$=流水级数，$m$=micro-batch 数。$p=8, m=8 \Rightarrow \frac{7}{15}\approx 47\%$ 空转；$m=32 \Rightarrow \frac{7}{39}\approx 18\%$。**nsys 时间线就是验证这个公式实际值的尺子**：数一数 NVTX 上空白格占比即可。

## 10. 开销与常见坑

**Nsys 是"低开销"但非零**：cuda+nvtx trace 通常加几个百分点墙钟；但若打开高频 CPU 采样、回溯（backtrace）、详细 osrt，开销会显著上升，且报告体积暴涨。原则：**先粗（cuda+nvtx）后细**。

| 坑 | 现象 | 对策 |
|---|---|---|
| 报告太大打不开 | 全程采几十 GB | 限范围/限 step，跳 warmup |
| 时间线全是碎 kernel | 没融合/没用 Graph | 算子融合、CUDA Graph |
| GPU 行看不到名字 | 没开 nvtx/没库符号 | 加 `-t nvtx,cublas,cudnn` |
| 远程无 GUI | 服务器跑不了界面 | CLI 出 `.nsys-rep`，本地 GUI 看；或 `nsys stats` |
| 多进程多卡只看到一个 | 默认采主进程 | 每 rank 独立 `-o report_rank{N}` |
| 误把 nsys 当 ncu 用 | 想看 occupancy | 那是 Nsight Compute 的活 |

## 常见问题

| 问题 | 答案 |
|---|---|
| nsys 和 nvidia-smi 的"利用率"区别？ | smi 是粗粒度瞬时采样（只要有 1 个 kernel 在跑就算"忙"，会虚高）；nsys 是精确到 µs 的时间线积分，能看出 kernel 之间的气泡 |
| nsys 和 Nsight Compute 怎么分工？ | nsys 看"系统/谁等谁"，ncu 看"单 kernel 内部为何慢"。先 nsys 定位，再 ncu 深挖 |
| 看到 GPU 大量气泡先怀疑什么？ | 按 §7 顺序：launch bound → 强制同步 → 拷贝串行 → 通信不重叠 → DataLoader → 单 kernel 慢 |
| PyTorch 怎么和 nsys 配合？ | `torch.cuda.nvtx` 手动打点，或 `emit_nvtx()` 自动给每个算子打标签 |
| 推理 decode 为什么总是 launch bound？ | 每步只生 1 token，kernel 极小，CPU launch 跟不上 → 用 CUDA Graph |
| 多卡训练重点看哪行？ | NCCL 行是否与 GPU compute 行并行（重叠）；不并行就是纯停等 |
| 采集会拖慢程序吗？ | 基础 trace 几个百分点；开高频采样/回溯会明显变慢，按需开 |
| 命令默认参数记不住？ | 概念稳定即可，精确参数随版本变，以官方文档为准 |

## 🔗 跳转链接

- 硬件地基：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 重叠与并行：[[llm-optimizer/计算通信重叠]] · [[ai-infra/网络/集合通信原语]]
- 张量/流水并行：[[llm-inference/大模型推理张量并行]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 推理优化（剖析对象）：[[llm-optimizer/FlashAttention]] · [[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]] · [[llm-inference/解码策略]]
- 算量基础（解读时间线的标尺）：[[llm-algo/FLOPs]] · [[llm-algo/transformer/模型架构]]
- 总目录：[[llm-inference/README]] · [[llm-train/README]] · [[00-知识地图]]
