# Nsight Systems —— 给 LLM 训练/推理做"全身 CT"的系统级时间线 Profiler

> 一句话定位：Nsight Systems（nsys）是 NVIDIA 的**系统级时间线采样剖析器**，把 CPU 线程、CUDA Kernel、内存拷贝、NCCL 通信、CUDA Stream 全部画到同一条时间轴上，专治 LLM 里"GPU 跑不满、被谁卡住了"这类**端到端瓶颈**问题。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-tools/nvtx]] · [[llm-tools/Pytorch-Profiler]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[ai-infra/网络/集合通信原语]]

---

## 阅读地图

| 你想知道 | 跳到 |
|---|---|
| nsys 到底测什么、和别的 profiler 啥区别 | §0、§1 |
| 时间线上的"泳道"怎么读 | §2 |
| 采样 vs 追踪（Sampling/Tracing）原理 | §3 |
| 怎么采（命令行/Python 包裹/最小流程） | §4 |
| 用 NVTX 给时间线打标记 | §5 |
| 怎么用它定位 LLM 训练/推理瓶颈 | §6 |
| 该盯哪些关键指标/读图套路 | §7 |
| 经典 LLM 性能病征对照表 | §8 |
| 数值手算：算 GPU 利用率/带宽/通信占比 | §9 |
| 和 Nsight Compute / Torch Profiler 怎么分工 | §10 |
| 局限与避坑 | §11 |

---

## 0. 一句话锚点

> **Nsight Systems 回答的是"时间都花在哪了、GPU 为什么空转"；Nsight Compute 回答的是"这一个 Kernel 内部为什么慢"。** 前者看**广度（整条时间线 / 跨 GPU / CPU+GPU 协同）**，后者看**深度（单 Kernel 的 SM / 内存 / warp 利用）**。LLM 调优 90% 的故事先从 nsys 时间线开始：先确认你是被 CPU 卡、被通信卡、被启动开销卡，还是真的被算力卡。

官方文档：`https://docs.nvidia.com/nsight-systems/UserGuide/index.html`

---

## 1. 地基：为什么需要一个"时间线 Profiler"

### 1.1 问题背景

你训练一个 LLM，发现一个 step 要 800ms，理论上算力只需要 ~500ms。多出来的 300ms 去哪了？可能的嫌疑犯：

- **CPU 准备数据太慢**（dataloader / tokenizer 没喂上）→ GPU 饿着。
- **Python 主线程在 launch kernel**，launch 开销 > kernel 执行时间（小算子太多）。
- **NCCL AllReduce 没和反向重叠**，通信串行在算力后面。
- **频繁的 H2D / D2H 拷贝**（`.cpu()` / `.item()` / 同步点）打断流水。
- **kernel 之间有气泡**（gap），stream 没排满。

光看一个总耗时数字（`time.time()`）根本分不清。你需要把"谁在什么时候、在哪个硬件资源上、干了多久"全部摊开看——这就是**时间线（timeline）**。

### 1.2 nsys 在工具谱系里的位置

```
   关注点：广 ←──────────────────────────────────→ 深
   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐
   │  Nsight Systems │   │  PyTorch Profiler│   │  Nsight Compute    │
   │  (nsys)         │   │  (torch.profiler)│   │  (ncu)             │
   ├─────────────────┤   ├──────────────────┤   ├────────────────────┤
   │ 整机/多卡时间线 │   │ 框架算子级聚合   │   │ 单 kernel 微架构   │
   │ CPU+GPU+NCCL    │   │ + 部分时间线     │   │ SM/内存/warp 占用  │
   │ 找"系统级"瓶颈  │   │ 找"哪个算子贵"   │   │ 找"为什么这个慢"   │
   └─────────────────┘   └──────────────────┘   └────────────────────┘
        先用它             顺手用它（贴近代码）       最后用它（攻坚单点）
```

直觉：nsys 像医院的**全身 CT**（一眼看出哪个器官出问题），ncu 像对那个器官做**穿刺活检**（这个细胞为什么病变）。

---

## 2. 时间线长什么样：读懂"泳道"

nsys 采完会产出一个 `.nsys-rep`（旧版 `.qdrep`）报告，用 **Nsight Systems GUI** 打开，时间线是一堆横向"泳道（lane/row）"，时间从左到右流动：

```
时间 ───────────────────────────────────────────────────────────▶
┌──────────────────────────────────────────────────────────────┐
│ CPU thread 0 (Python 主线程)                                   │
│   [forward launch][backward launch]   ....idle....   [optim]   │  ← CPU 在排队发命令
├──────────────────────────────────────────────────────────────┤
│ CUDA API (cudaLaunchKernel / cudaMemcpyAsync ...)              │
│   |||||||||| 一堆细条 = launch 调用                            │  ← launch 开销在这层
├──────────────────────────────────────────────────────────────┤
│ NVTX ranges (你打的标记)                                       │
│   [=== Attention ===][== MLP ==][===== AllReduce =====]        │  ← 语义标签，对齐源码
├──────────────────────────────────────────────────────────────┤
│ GPU Stream 7  (compute)                                        │
│   [gemm][softmax][gemm] <gap> [gemm][........]                 │  ← 真正的算力执行
├──────────────────────────────────────────────────────────────┤
│ GPU Stream 13 (NCCL comm)                                      │
│            [====== ncclAllReduce ======]                       │  ← 通信 kernel
├──────────────────────────────────────────────────────────────┤
│ Memory (H2D / D2H / D2D)                                       │
│   [H2D]                              [D2H]                      │  ← 拷贝，常是隐藏的同步点
└──────────────────────────────────────────────────────────────┘
```

读图三连问：

1. **GPU compute 泳道有没有空隙（gap）？** 有 gap = GPU 空转，去上面找谁没喂上。
2. **NCCL 泳道和 compute 泳道是"并排（重叠）"还是"前后串行"？** 串行 = 通信没藏住。
3. **CPU launch 泳道是不是密密麻麻、且 GPU 在等它？** 是 = launch-bound（CPU 发命令跟不上 GPU 吃命令）。

---

## 3. 原理：采样（Sampling）与追踪（Tracing）

nsys 同时用两种数据采集机制，理解它俩的区别才能正确解读时间线。

### 3.1 Tracing（追踪）—— 精确记录"事件"

对 **API 调用**（CUDA Runtime/Driver、NCCL、cuDNN、cuBLAS、NVTX、OS runtime）做**插桩/拦截**，记录每个事件的**开始/结束时间戳**。所以 GPU kernel 的执行区间、NCCL 通信区间、memcpy 区间都是**精确**的（不是估的）。

- 优点：时间区间准、能对应到具体调用。
- 代价：高频小事件会带来 overhead（一个 step 几十万次 launch 时尤其明显）。

### 3.2 Sampling（采样）—— 周期性"偷看" CPU 在干嘛

按固定频率（如每隔若干微秒）中断，抓 **CPU 各线程的调用栈（backtrace）**。用统计意义告诉你"CPU 时间大头花在哪个函数"。GPU 这边则用 **CUPTI / PM sampling** 周期性读硬件计数器（如 SM 活跃度）。

- 采样是**统计**的：采到多 = 该处耗时占比高，但不保证抓到每一次。

### 3.3 一句话对照

```
Tracing  →  "事件级"，有头有尾，区间精确   →  GPU kernel、NCCL、memcpy、CUDA API
Sampling →  "周期偷拍"，统计占比          →  CPU 调用栈、SM 占用率热度
```

> 时间线上 GPU kernel 的"块"是 trace 来的（准）；左侧 CPU 那条"哪个函数最热"是 sample 来的（统计）。别把采样的统计值当成精确计时。

---

## 4. 怎么采：命令行与最小流程

### 4.1 最常用命令骨架

> ⚠️ 具体 flag 的默认值/可选项以 `nsys profile --help` 和官方文档为准，这里给最常用形态。

```bash
# 训练脚本整段采集（推荐：包住你的真实启动命令）
nsys profile \
    -t cuda,nvtx,nccl,osrt \      # trace 哪些子系统：CUDA/NVTX/NCCL/OS runtime
    -o my_train_report \          # 输出报告名（生成 my_train_report.nsys-rep）
    --force-overwrite true \
    python train.py --steps 50
```

- `-t / --trace`：选 trace 的子系统。LLM 场景常带 `cuda,nvtx,nccl`；CPU 侧瓶颈再加 `osrt`（OS runtime，能看到 mutex/同步/IO）。
- `-o`：输出文件名。
- `--force-overwrite true`：覆盖已有报告。

### 4.2 只采"稳态的几个 step"（强烈推荐）

整段采会很大、warmup 还会污染数据。正确做法是**只圈住稳态的几个 step**，两种方式：

**方式 A：延迟启动 + 限时长（粗粒度）**

```bash
nsys profile -t cuda,nvtx,nccl \
    --delay=20 \         # 跳过前 20s warmup（单位以文档为准，通常秒）
    --duration=5 \       # 只采 5s
    -o steady_state \
    python train.py
```

**方式 B：代码里精确圈选（细粒度，最干净）—— `cudaProfilerStart/Stop`**

```python
import torch
# 启动 nsys 时加 --capture-range=cudaProfilerApi
for step, batch in enumerate(loader):
    if step == 10:
        torch.cuda.cudart().cudaProfilerStart()   # 从第 10 步开始采
    train_one_step(batch)
    if step == 13:
        torch.cuda.cudart().cudaProfilerStop()    # 采 3 步就停
        break
```

```bash
nsys profile -t cuda,nvtx,nccl \
    --capture-range=cudaProfilerApi \   # 只在 Start/Stop 之间记录
    -o three_steps python train.py
```

> 经验法则：**采 3~5 个稳态 step 足矣**。step 太多报告巨大、GUI 卡；warmup 那几步（cuDNN autotune、CUDA graph 捕获、内存池预热）一定要排除。

### 4.3 多卡/多进程

每个 rank 是独立进程，nsys 会**为每个进程生成各自的轨迹**，再在 GUI 里合并成多 GPU 时间线。常见做法是让启动器（torchrun/mpirun）把 `nsys profile ...` 包在每个 rank 外层，并用 `%p`（PID）或 rank 号区分输出文件名，避免互相覆盖。

---

## 5. NVTX：给时间线打"语义标签"（关键技巧）

光看 GPU 上一堆 `ampere_sgemm_...` kernel，你根本不知道哪个属于 Attention、哪个属于 MLP。**NVTX（NVIDIA Tools Extension）** 让你在源码里插标记，nsys 把它画成时间线上的彩色区间，瞬间把"硬件事件"翻译回"业务语义"。详见 [[llm-tools/nvtx]]。

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("attention")     # 开一个区间
out = attention(q, k, v)
nvtx.range_pop()                 # 关区间

# 或用上下文管理器风格（torch 提供 torch.cuda.nvtx.range）
with torch.cuda.nvtx.range("mlp"):
    out = mlp(x)
```

效果：时间线 NVTX 泳道出现 `[attention][mlp]` 区间，你能直接量"Attention 这块在 GPU 上占了多少 ms、里头有没有 gap"。**没有 NVTX，多卡大模型的时间线基本没法读。**

```
NVTX 泳道:  [== embed ==][===== attn =====][=== mlp ===][== norm ==]
GPU 泳道 :  [emb_k][      qkv_gemm | softmax | av_gemm     ][ffn_gemm][...]
                    ↑ 现在你知道这堆 gemm 属于 attention
```

---

## 6. 实战：用 nsys 定位 LLM 性能瓶颈

把 §2 的读图三连问落到具体 LLM 病征上。下面是一套**自顶向下的诊断流程**。

### 6.1 第一步：GPU 忙不忙？（算 GPU 占用率）

在一个 step 的 NVTX 区间内，量 **GPU compute 泳道的"忙时间总和" / "区间墙钟时间"**。

```
GPU 占用率 ≈ Σ(kernel 执行时长)  /  step 墙钟时长

占用率 > 90%  → GPU 是瓶颈，去 §6.5 看是不是 kernel 本身慢（转 ncu）
占用率 < 80%  → 有人在卡 GPU，继续往下查谁
```

### 6.2 病征一：CPU-bound / launch-bound（GPU 在等 CPU 发命令）

时间线特征：

```
CPU API 泳道: ||||||  ||||||  ||||||   ← launch 密集
GPU 泳道    : [k][k]<gap>[k][k]<gap>   ← 一堆小 gap，kernel 之间总在等
```

- 现象：GPU kernel 很小很碎，kernel 之间频繁出现 microsecond 级 gap，CPU launch 跟不上。
- 根因：算子太多太碎、Python overhead 大、没用 CUDA Graph、batch 太小。
- 解法：**算子融合**（fused kernel）、**CUDA Graph**（把一串 launch 录成一个 replay）、增大 batch、用编译器（torch.compile）。

### 6.3 病征二：被 Host 同步打断（隐藏的 D2H / `.item()`）

时间线特征：GPU 泳道里出现一段大 gap，正好对齐 CPU 上一次 `cudaStreamSynchronize` 或 D2H memcpy。

- 现象：训练循环里有 `loss.item()`、`.cpu()`、`tensor.numpy()`、`print(tensor)`、`torch.cuda.synchronize()`，每次都强制 GPU 排空、CPU 等结果，制造大气泡。
- 解法：把日志/指标**攒着批量同步**（比如每 N 步才 `.item()` 一次），或用异步拷贝。

### 6.4 病征三：通信没藏住（NCCL 串行在算力后）

时间线特征：

```
✘ 没重叠（坏）：
GPU compute: [==== backward ====]              [next fwd]
NCCL comm  :                     [== AllReduce ==]        ← 串行，纯等

✔ 已重叠（好）：
GPU compute: [==== backward ====][next fwd ...]
NCCL comm  :        [== AllReduce ==]                     ← 和算力并排
```

- 现象：数据并行里 AllReduce / ReduceScatter 区间和 compute 区间**首尾相接而非并排**，通信时间几乎 100% 暴露。
- 根因：梯度通信没和反向重叠（bucket 设置、`overlap_grad_reduce` 没开）、PCIe/NVLink 带宽不足、small message 太多。
- 解法：开启梯度通信重叠（DDP/ZeRO 的 overlap）、调 bucket size、检查拓扑（NVLink vs PCIe）。配合 [[ai-infra/网络/集合通信原语]] 理解 AllReduce 成本。

### 6.5 病征四：流水线并行的"气泡"

时间线特征（多 GPU 多 stage）：

```
GPU0(stage0): [F][F][F]            [B][B][B]
GPU1(stage1):    [F][F][F]      [B][B][B]
GPU2(stage2):       [F][F][F][B][B][B]
                            ↑↑↑ 各 stage 起步/收尾的空白 = pipeline bubble
```

- nsys 是看 PP bubble、判断 micro-batch 数够不够、各 stage 是否负载均衡的最佳工具。配合 [[B07:llm-inference/大模型推理张量并行]] 理解并行切分。

### 6.6 推理场景特别关注

- **Prefill vs Decode 两阶段**：用 NVTX 分别圈 prefill 和 decode，看 decode 是不是被 KV-Cache 读写 / 小 batch 拖成 memory-bound（见 [[llm-inference/KV-Cache优化]]）。
- **连续批处理（continuous batching）**：看每个 decode step 的 GPU 是否被打满，调度间隙有没有 gap（vLLM 等，见 [[llm-inference/vllm/README]]）。
- **PD 分离**：prefill 集群和 decode 集群的时间线特征差异很大（compute-bound vs memory-bound），见 [[llm-inference/PD分离]]。

---

## 7. 该盯哪些关键读数（"看什么指标"）

nsys GUI 有 **Stats / Summary** 视图（也可 `nsys stats report.nsys-rep` 出表格），重点看：

| 指标 / 视图 | 说明 | LLM 解读 |
|---|---|---|
| **GPU Kernel Summary**（Top kernels by time） | 各 kernel 总耗时占比 | gemm 占大头才正常；若 elementwise/cast/copy 占很多 → 算子没融合 |
| **CUDA API Summary** | `cudaLaunchKernel`/`cudaMemcpyAsync` 等调用次数与耗时 | launch 次数巨大 → launch-bound，考虑 CUDA Graph |
| **NCCL / 通信占比** | AllReduce/ReduceScatter 总时长 | 通信时长 / step 时长 = 通信占比，>30% 要警惕 |
| **memcpy（H2D/D2H）** | 主机↔设备拷贝量与次数 | 频繁 D2H 常是隐藏同步点 |
| **GPU 空闲/利用** | compute 泳道 gap 总和 | gap 大 = 喂不饱 |
| **NVTX range 统计** | 每个语义区间的耗时 | 直接量 attn/mlp/comm 各占多少 |

> 训练/推理性能名词（MFU、吞吐、TTFT、TPOT、显存带宽利用率等）的精确定义见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

---

## 8. 经典 LLM 性能病征 → 时间线特征 对照表

| 病征 | 时间线长相 | 根因 | 处方 |
|---|---|---|---|
| GPU 空转、gap 多 | compute 泳道一堆小缝 | 算子碎、launch 慢 | 融合 / CUDA Graph / 增大 batch |
| launch-bound | CPU API 泳道密集且 GPU 在等 | Python+launch 开销 | CUDA Graph / torch.compile |
| 隐藏同步 | 周期性大 gap 对齐 D2H/sync | `.item()`/`.cpu()`/同步打印 | 批量同步、异步化 |
| 通信暴露 | NCCL 与 compute 串行 | 没开重叠 / 拓扑差 | 重叠 + bucket 调优 + NVLink |
| PP 气泡 | 各 stage 头尾空白 | micro-batch 太少 / 不均衡 | 增 micro-batch、均衡切分 |
| dataloader 饿 | 每 step 开头大 gap、CPU 在 IO | 数据没预取 | 多 worker、预取、pin_memory |
| decode memory-bound | decode kernel 小、SM 没占满 | KV 读写、batch 小 | 量化 KV / 增大并发 / PagedAttention |

---

## 9. 数值手算：把时间线读数变成结论

### 9.1 GPU 利用率

设某个 step 墙钟 = $T_{step}$，该区间内所有 GPU kernel 执行时长之和 = $T_{busy}$（GUI 能直接给）。

$$\text{GPU Util} = \frac{T_{busy}}{T_{step}}$$

例：一个 step 墙钟 $T_{step}=800\text{ms}$，kernel 忙 $T_{busy}=560\text{ms}$，则利用率 $=560/800=70\%$。结论：**有 30% 时间 GPU 空转**，去时间线找那 240ms gap 对齐谁——大概率是通信或同步。

### 9.2 通信占比 & 是否藏住

设反向计算 $T_{bwd}=300\text{ms}$，AllReduce $T_{comm}=120\text{ms}$。

- 若**完全串行**：这部分耗时 $=300+120=420\text{ms}$，通信暴露 100%。
- 若**完全重叠**：耗时 $=\max(300,120)=300\text{ms}$，通信被反向算力完全藏住，省下 120ms。
- 时间线上量到实际是 360ms，则**藏住了一半**：暴露 $=420-360=60$ms，重叠率 $=(420-360)/120=50\%$ → 还有优化空间。

### 9.3 launch-bound 判据（粗算）

设一个 step launch 了 $N=200{,}000$ 个 kernel，每个 CPU launch 开销约 $\sim5\mu s$（典型量级，**以实测为准**），则纯 launch 时间 $\approx 200000 \times 5\mu s = 1.0\text{s}$。若 step 墙钟才 0.8s，说明**单线程根本来不及 launch**，GPU 必然被卡 → 强烈提示上 CUDA Graph（把这 20 万次 launch 折叠成一次 replay）。

> ⚠️ 上面 $5\mu s$ 是数量级示意，真实 launch 开销随 GPU/驱动/参数差异很大，请从你自己的 nsys CUDA API Summary 里读实测值。

### 9.4 把"算力上限"作对照（MFU 思路）

时间线告诉你"花了多久"，要判断"该不该这么久"，需要和理论算力比。比如一个 GEMM 实测 $t$ ，已知它的 FLOPs 和 GPU 峰值 $P_{peak}$（如 H100 BF16 的 TFLOPS，**具体看官方规格**），则该 kernel 的 MFU $\approx \dfrac{\text{FLOPs}/t}{P_{peak}}$。MFU 低 → 这个 kernel 值得拿去 ncu 深挖（§10）。

---

## 10. 和 Nsight Compute / PyTorch Profiler 的分工

| 工具 | 粒度 | 回答的问题 | 何时用 |
|---|---|---|---|
| **Nsight Systems (nsys)** | 系统/时间线 | 时间花哪了、GPU 为何空转、通信/同步/IO 卡不卡 | **先用**，定位是哪一类瓶颈、跨卡协同 |
| **PyTorch Profiler** | 框架算子级 | 哪个 `aten::` 算子/层最贵，贴近 Python 代码 | 想从代码视角看热点、出 trace 给 TensorBoard，见 [[llm-tools/Pytorch-Profiler]] |
| **Nsight Compute (ncu)** | 单 Kernel 微架构 | 这个 kernel 为何慢（SM 占用/内存带宽/warp stall） | **最后用**，攻坚 nsys 找出的那一个慢 kernel |

典型工作流：

```
nsys 看时间线  →  发现某个 GEMM/Attention kernel 又大又频繁、MFU 低
        │
        ▼
ncu 单独剖那个 kernel  →  发现是 memory-bound / 占用率低 / bank conflict
        │
        ▼
改 kernel（融合/调 tile/换 FlashAttention）→ 回 nsys 验证 gap 是否消失
```

> Attention 的 kernel 级优化思路见 [[llm-optimizer/FlashAttention]]；KV-Cache 显存与带宽账见 [[llm-optimizer/kv-cache]] / [[docs/transformer内存估算]]。

---

## 11. 局限与避坑

| 坑 | 说明 / 对策 |
|---|---|
| **采样会有 overhead** | 高频小 launch 场景下 trace 本身拖慢程序；只采稳态几步、别全程采 |
| **报告巨大** | 采太多 step → 文件几个 GB、GUI 卡死；用 `--capture-range` 或 `--duration` 圈小 |
| **没 NVTX 等于盲读** | 大模型多卡时间线没语义标记几乎读不懂；先把 attn/mlp/comm 用 NVTX 圈好 |
| **warmup 污染** | 前几步有 autotune / graph capture / 内存池预热，必须排除 |
| **采样 ≠ 精确计时** | 左侧 CPU 热点是统计的；精确区间看 trace 的 kernel 块 |
| **多卡输出互相覆盖** | 每个 rank 用 `%p`/rank 区分输出名 |
| **它不告诉你 kernel 内部** | 单 kernel 为何慢要转 ncu，nsys 只给"块多长" |
| **版本/默认值会变** | 命令默认值、子系统名以你本机 `nsys --help` 与官方文档为准 |

---

## 一页流程图（速记）

```
┌─ 1. nsys profile -t cuda,nvtx,nccl，用 --capture-range 只采稳态 3~5 步
│
├─ 2. GUI 打开 .nsys-rep，先看 GPU compute 泳道有没有 gap
│        ├─ 没 gap 且占用>90% → 真算力瓶颈 → 转 ncu 深挖慢 kernel
│        └─ 有 gap → 往下抓凶手
│
├─ 3. gap 对齐谁？
│        ├─ 对齐密集 CPU launch        → launch-bound  → CUDA Graph/融合
│        ├─ 对齐 D2H / synchronize      → 隐藏同步      → 批量/异步化
│        ├─ 对齐 NCCL 串行              → 通信暴露      → 开重叠/调 bucket
│        ├─ step 开头对齐 IO            → dataloader 饿 → 预取/多 worker
│        └─ stage 头尾空白              → PP 气泡       → 增 micro-batch
│
└─ 4. 改一处 → 重采 → 看 gap 是否变小 → 迭代
```

---

## 🔗 跳转链接

- 返回知识地图：[[00-知识地图]]
- 配套工具：[[llm-tools/nvtx]] · [[llm-tools/Pytorch-Profiler]]
- 性能名词：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 通信与并行：[[ai-infra/网络/集合通信原语]] · [[B07:llm-inference/大模型推理张量并行]] · [[llm-train/pytorch/distribution/README]]
- Kernel 级优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理瓶颈：[[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/vllm/README]]
- 显存账：[[docs/transformer内存估算]]
- GPU 原理：[[ai-infra/算力/GPU工作原理]]
