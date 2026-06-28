# NVTX：给 GPU 时间线"贴标签"的性能标注工具

> NVTX 是 NVIDIA 提供的"代码区间标注"接口，让你在 profiler 时间线上看到的不再是匿名 kernel，而是 `attention` / `ffn` / `optimizer.step` 这样的语义化标签，从而精准定位 LLM 训练/推理的性能瓶颈。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-tools/nsight]] · [[llm-tools/Pytorch-Profiler]] · [[ai-infra/算力/GPU工作原理]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 节 | 内容 | 你将学会 |
|----|------|----------|
| 0 | 一句话锚点 | NVTX 到底是什么、不是什么 |
| 1 | 问题背景：profiler 看到的是"乱码" | 为什么裸时间线难读 |
| 2 | NVTX 是什么：Marker / Range / Domain | 三种标注原语 |
| 3 | 工作原理：标注如何进入时间线 | CPU 打点 + 异步 GPU 投影 |
| 4 | Python `nvtx` 用法全解 | 装饰器/上下文/范围/嵌套 |
| 5 | C/C++ 与 CUDA 侧用法 | `nvtxRangePush` 等 |
| 6 | 与 Nsight / PyTorch Profiler 联动 | 谁来"显示"标注 |
| 7 | 用 NVTX 定位 LLM 瓶颈实战 | 看什么、怎么读 |
| 8 | 数值示例：算 GPU 利用率 / 气泡 | 从时间线手算指标 |
| 9 | 最佳实践 & 开销分析 | 标多了会不会拖慢？ |
| 评价 | 对照其他手段 / 局限 | 何时用、何时别用 |

## 0. 一句话锚点

> **NVTX（NVIDIA Tools Extension）= 一套"插桩 API"，你在源码里调用它把一段代码标记为一个"命名区间"，profiler（Nsight Systems / Nsight Compute / PyTorch Profiler）会把这些区间画到时间线上，与底层 CUDA kernel / NCCL 通信 / 内存拷贝对齐。它本身不测量、不画图，只负责"贴标签"。**

记住这个分工：
- **NVTX**：在代码里"打标签"（生产者）。
- **Nsight / Profiler**：抓取、对齐、可视化这些标签（消费者）。
- 没有 profiler 在采集时，NVTX 调用几乎是空操作（接近零开销）。

## 1. 地基：为什么需要 NVTX（问题背景）

你用 Nsight Systems 抓了一段 LLM 训练，打开时间线，看到的是这样：

```
GPU 时间线（没有 NVTX）：
┌──────────────────────────────────────────────────────────────┐
│ ████ ██ ████████ █ ██ ████ ██████ █ ███ ██ ████████ ██ ████ │
│ sgemm  elementwise  reduce  sgemm  layernorm_kernel  sgemm ... │
└──────────────────────────────────────────────────────────────┘
       ↑ 几百个 kernel，名字是 cutlass / ampere_sgemm_128x... 这种
```

问题在于：
1. **kernel 名字是底层算子名**，不是业务语义。`ampere_sgemm_*` 既可能来自 attention 的 QKᵀ，也可能来自 FFN 的第一层 GEMM——你分不清。
2. **一个前向有成百上千个 kernel**，靠肉眼对应到"哪一层、哪个模块"几乎不可能。
3. **CPU 与 GPU 是异步的**：Python 代码 `model(x)` 早就返回了，GPU 还在跑上一批 kernel，时间线上"代码逻辑"和"kernel 执行"对不上。

NVTX 解决的就是这个"**语义鸿沟**"：你在代码里写 `with nvtx.annotate("attention"):`，时间线上就会出现一条横跨该区间所有 kernel 的"attention"色块。

```
GPU 时间线（有 NVTX）：
┌──────────────────────────────────────────────────────────────┐
│ NVTX:  [   attention   ][      ffn       ][ layernorm ][ ... ] │
│ CUDA:  ████ ██ ████████  █ ██ ████ ██████  █ ███ ██     ████   │
└──────────────────────────────────────────────────────────────┘
        ↑ 现在每段 kernel 都"归属"到一个有意义的名字
```

## 2. NVTX 是什么：三种标注原语

NVTX 提供三类"标注对象"，理解它们是用好 NVTX 的关键：

### 2.1 Marker（标记 / 瞬时点）
一个**没有持续时间**的时间点。像在时间线上钉一根针，表示"此刻发生了某事件"（如"开始第 100 步""触发 OOM 重试"）。Python 侧用得少，C 侧 `nvtxMarkA("event")`。

### 2.2 Range（范围 / 区间）★最常用
一段**有起止**的时间区间，画成一个有宽度的色块。又分两种：
- **Push/Pop 范围（栈式，可嵌套）**：`push_range` / `pop_range`，先进后出，天然形成层级（外层 step，内层 layer，再内层 attention）。
- **Start/End 范围（句柄式，可跨线程/不嵌套）**：`start_range` 返回一个句柄，`end_range(句柄)` 结束。适合异步、不满足栈嵌套关系的场景。

### 2.3 Domain（域 / 命名空间）
把标注分组到不同"命名空间"，避免不同库（你的代码 vs 框架 vs NCCL）的标注混在一起。大型项目才需要；入门可忽略。

```
三原语关系图：

Domain "myapp"
  │
  ├── Range "step_100"              ← 外层 Push/Pop
  │     ├── Range "forward"
  │     │     ├── Range "attention"  ← 嵌套
  │     │     └── Range "ffn"
  │     ├── Range "backward"
  │     └── Range "optimizer.step"
  │
  └── Marker "checkpoint_saved"     ← 瞬时点（无宽度）
```

## 3. 工作原理：一个标注是如何出现在时间线上的

关键认知：**NVTX 标注本质是 CPU 侧的"打点"，但要投影到 GPU 时间线上。**

```
机制图（CPU 打点 → 异步投影到 GPU 轴）：

CPU 线程（Python/C++）
  t0: nvtx.push_range("attention")  ──┐ 记录 t0 时间戳 + 标签
      launch QKᵀ kernel (异步)        │
      launch softmax kernel (异步)    │  这些 kernel 排进 CUDA stream，
      launch AV kernel (异步)         │  稍后才在 GPU 上真正执行
  t1: nvtx.pop_range()              ──┘ 记录 t1 时间戳

GPU 时间线（profiler 把 CPU 区间投影下来）
  CPU 轴:    [t0 ─────── "attention" ─────── t1]
  GPU 轴:        [QKᵀ][softmax][  AV  ]
                  ↑ profiler 用 stream/时间对齐，把 GPU 上这几个
                    kernel 归入 "attention" 这条 NVTX range
```

要点：
1. **只有当 profiler（Nsight/PyTorch Profiler）正在采集时**，NVTX 调用才被记录；否则它走的是一个"空实现"，几乎不耗时。
2. **CPU range 和 GPU kernel 的对齐**由 profiler 完成。因为是异步，CPU 上 `pop_range` 可能在 GPU kernel 还没跑完时就执行了——这正是 profiler 同时画"CPU NVTX 行"和"GPU CUDA 行"的原因，你能从两行的错位看出**launch 开销/排队延迟**。
3. NVTX 不会引入额外的 `cudaDeviceSynchronize`（除非你自己加），所以默认**不改变程序的异步行为**。

## 4. Python `nvtx` 用法全解

安装：`pip install nvtx`（NVIDIA 官方 Python 绑定；RAPIDS/CuPy 生态也常自带）。
官方文档：
- https://nvtx.readthedocs.io/en/latest/annotate.html
- https://github.com/NVIDIA/NVTX

### 4.1 装饰器：标注整个函数

```python
import nvtx

@nvtx.annotate(message="my_func", color="blue")
def my_func():
    pass
```

函数每次被调用，时间线上就出现一段名为 `my_func` 的蓝色 range，自动随函数进入/退出而开始/结束。最省事，适合给关键函数（如 `forward` / `step`）整体打标。

> 小贴士：`message` 不传时默认用函数名；`color` 接受颜色名或整型 ARGB。

### 4.2 上下文管理器：标注一个 `with` 块

```python
import nvtx

with nvtx.annotate(message="data_loading", color="green"):
    batch = next(loader)      # 这段时间被标为 data_loading
    batch = batch.to("cuda")
```

最灵活，想标哪段就 `with` 包哪段。LLM 训练里典型用法：分别包住 `forward` / `loss` / `backward` / `optimizer.step`，一眼看出各阶段占比。

### 4.3 Start/End 范围（句柄式，可跨作用域）

```python
import nvtx

rng = nvtx.start_range(message="prefill", color="blue")
# ... 执行 prefill 阶段（可能跨多个函数/回调）...
nvtx.end_range(rng)
```

`start_range` 返回句柄，`end_range(句柄)` 显式结束。适合**起止不在同一作用域**、或不满足栈嵌套的异步场景（如推理里 prefill 与 decode 的边界由回调触发）。

### 4.4 Push/Pop 范围（栈式，可嵌套）

```python
import nvtx

for i in range(num_batches):
    nvtx.push_range("batch " + str(i), color="blue")   # 外层
    nvtx.push_range("forward", color="orange")          #   内层
    # ... forward ...
    nvtx.pop_range()                                    #   结束 forward
    nvtx.push_range("backward", color="red")
    # ... backward ...
    nvtx.pop_range()                                    #   结束 backward
    nvtx.pop_range()                                    # 结束 batch i
```

`push`/`pop` 严格**后进先出**，天然形成"step → 阶段 → 层"的多级层次，在 Nsight 时间线上表现为**多行嵌套的色块**，可以折叠/展开，最适合给整个训练循环建立结构化视图。

### 4.5 四种用法选型

| 用法 | 形态 | 何时用 | 嵌套 |
|------|------|--------|------|
| `@annotate` 装饰器 | 整函数 | 给关键函数整体打标，最省事 | 随调用栈自然嵌套 |
| `with annotate` | 代码块 | 灵活标任意片段（最常用） | 随 `with` 嵌套 |
| `start_range`/`end_range` | 句柄 | 起止跨作用域/异步 | ✗（句柄各自独立）|
| `push_range`/`pop_range` | 栈 | 建多级层次结构 | ✓（先进后出）|

## 5. C / C++ / CUDA 侧用法（了解即可）

底层 C API（Python 绑定就是封装它）：

```c
#include "nvtx3/nvToolsExt.h"

nvtxRangePushA("forward");      // 入栈一个范围
//   ... launch kernels ...
nvtxRangePop();                 // 出栈

nvtxRangeId_t id = nvtxRangeStartA("prefill");  // 句柄式开始
//   ...
nvtxRangeEnd(id);                                // 句柄式结束

nvtxMarkA("oom_retry");         // 瞬时标记
```

自定义颜色/负载用 `nvtxEventAttributes_t` 结构体填 `color` / `payload` / `message`。框架（PyTorch、Megatron、TensorRT-LLM）内部已大量插了 NVTX，所以你常常**什么都不用写**，开 profiler 就能看到框架自带的语义标签。

## 6. 与 Nsight / PyTorch Profiler 的联动（谁来显示）

NVTX 是"标签生产者"，下面这些是"消费者/显示器"：

```
        ┌──────────── 你的代码（生产 NVTX 标签）────────────┐
        │  with nvtx.annotate("attention"): ...            │
        └───────────────────────┬─────────────────────────┘
                                 │ 标签事件
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  Nsight Systems          PyTorch Profiler         Nsight Compute
  (全局时间线,           (torch.profiler,          (单 kernel 微观,
   看阶段占比/重叠)       导出 chrome trace)         用 NVTX 圈定范围)
```

- **Nsight Systems**：`nsys profile -t cuda,nvtx,osrt,cudnn,cublas python train.py`——`-t nvtx` 显式打开 NVTX 采集。时间线上 NVTX 行与 CUDA/NCCL 行对齐，是看"阶段级"瓶颈的主力。详见 [[llm-tools/nsight]]。
- **PyTorch Profiler**：`torch.profiler.profile(..., with_stack=True)` 会把 `torch.cuda.nvtx.range_push/pop` 以及 autograd 自动插的 NVTX 一并记录；也可用 `torch.autograd.profiler.emit_nvtx()` 让每个 op 自动发 NVTX 给 Nsight。详见 [[llm-tools/Pytorch-Profiler]]。
- **Nsight Compute**：做单 kernel 深挖（SM 占用率、访存）时，用 `--nvtx --nvtx-include "attention/"` 只剖析某个 NVTX range 内的 kernel，避免全量采集太慢。

> PyTorch 提供两套等价入口：`torch.cuda.nvtx.range_push("x") / range_pop()`（底层）与第三方 `nvtx` 包的 `annotate`（更 Pythonic）。两者都能被 Nsight 抓到，按习惯选其一即可。

## 7. 用 NVTX 定位 LLM 性能瓶颈（实战看什么）

给训练循环加一组标准 NVTX 标注后，在 Nsight 时间线上重点看四件事：

```
一个训练 step 的 NVTX 时间线（示意）：

CPU/NVTX:  [data][      forward      ][   backward   ][opt.step][allreduce]
GPU CUDA:       ████████████████████  ██████████████  ███████   
NCCL:                                                           ▒▒▒▒▒▒▒  ← 通信
            ↑①         ↑②                  ↑③            ↑④       ↑⑤
```

**① data 段 GPU 空白**：dataloader 没喂上数据 → GPU 在等 CPU。瓶颈在数据管线（增 `num_workers` / 预取 / pin_memory）。
**② forward 占比异常大**：看内层嵌套，哪层最宽。常见 attention 段过宽 → 该上 FlashAttention（见 [[llm-optimizer/FlashAttention]]）。
**③ backward ≈ 2×forward**：正常（反向算两份梯度）。若远超，可能重计算（gradient checkpointing）太激进。
**④ optimizer.step 偏长**：优化器状态多（Adam 三份）或主机-设备同步过多。
**⑤ allreduce 与计算不重叠**：NVTX 的 NCCL 行单独排在计算之后 → 没开通信-计算重叠（overlap），有大段"通信气泡"，应启用梯度桶重叠（见 [[ai-infra/网络/集合通信原语]]）。

推理侧（vLLM / TensorRT-LLM）：用 NVTX 区分 **prefill** 与 **decode** 两段，看 decode 是否被 KV-Cache 访存拖成 memory-bound（见 [[llm-inference/KV-Cache优化]]、[[llm-inference/PD分离]]）。

## 8. 关键数值示例：从 NVTX 时间线手算指标

NVTX 给你区间起止时间戳，可直接算工程指标。设某 step 时间线读数（数字为演示，非实测）：

```
step 总墙钟  T_step      = 100 ms
  data        段（GPU 空闲）= 12 ms
  forward     段          = 25 ms（GPU 忙）
  backward    段          = 50 ms（GPU 忙）
  optimizer   段          = 5  ms（GPU 忙）
  allreduce   段（纯通信，未与计算重叠）= 8 ms（GPU 空闲）
```

**GPU 计算占用率（compute utilization）：**

$$U = \frac{T_{\text{forward}}+T_{\text{backward}}+T_{\text{opt}}}{T_{\text{step}}} = \frac{25+50+5}{100} = 80\%$$

**气泡/空闲占比（bubble）：**

$$B = \frac{T_{\text{data}}+T_{\text{allreduce}}}{T_{\text{step}}} = \frac{12+8}{100} = 20\%$$

**消除气泡后的理论加速比**（假设 data 预取掩盖、allreduce 与 backward 完全重叠）：

$$\text{Speedup} = \frac{T_{\text{step}}}{T_{\text{step}} - T_{\text{data}} - T_{\text{allreduce}}} = \frac{100}{100-12-8} = \frac{100}{80} = 1.25\times$$

也就是说 NVTX 直接告诉你"**有 20% 的时间 GPU 在摸鱼，理想优化能提速约 1.25 倍**"——这正是 NVTX 标注的价值：把模糊的"慢"量化成"哪一段、占多少、值不值得优化"。

**launch 开销估算**：若 NVTX 的 CPU `forward` range 宽度只有 8 ms，而对齐的 GPU kernel 实际跨度 25 ms，说明 CPU 早早 launch 完就返回了，GPU 在排队消化——CPU 不是瓶颈；反之若 CPU range 比 GPU 还宽，多半 launch/Python 开销过大（kernel 太碎，考虑 CUDA Graph / 算子融合）。

## 9. 最佳实践 & 开销分析

- **粒度适中**：标到"阶段/层"级别（forward/attention/ffn）足矣；别给每个微小 op 都套 `with`，否则时间线密密麻麻反而难读，且 Python 调用本身有开销。
- **用颜色编码语义**：通信红、计算蓝、数据绿，扫一眼颜色就知结构。
- **结合循环编号**：`push_range(f"step_{i}")` 方便定位"第几步开始变慢"（如内存碎片化、缓存增长）。
- **开销**：未采集时 NVTX 调用接近 no-op（约几纳秒），可常驻代码；采集时每个 range 约几十到上百纳秒量级的记录开销，对 ms 级 LLM 阶段可忽略。**不要在最内层热循环里高频打点**。
- **不要靠 NVTX 做同步计时**：它不插 `cudaDeviceSynchronize`，CPU range 宽度 ≠ GPU 执行时间；要测 GPU 真实耗时请看 GPU 轴或用 CUDA event 计时。

## 评价 / 对照 / 局限

| 维度 | NVTX | `time.time()` 手测 | CUDA Event 计时 | 框架自带 hook |
|------|------|--------------------|-----------------|---------------|
| 可视化对齐 GPU | ✅ 强（profiler 投影）| ❌ 只有数字 | ⚠️ 需自己配对 | ⚠️ 取决于框架 |
| 反映异步真实执行 | ✅（看 GPU 轴）| ❌（CPU 早返回，测不准）| ✅ | 视实现 |
| 侵入性 | 低（几行标注）| 低 | 中 | 零（已内置）|
| 采集外开销 | 接近零 | 零 | 低 | 零 |
| 适合场景 | 看"哪段慢、占比、重叠" | 粗略 CPU 耗时 | 精确测某段 GPU 时间 | 快速概览 |

**局限**：
1. **只标不测**——NVTX 自己不产出耗时数字，必须配合 profiler；脱离 Nsight/PyTorch Profiler 它什么也看不到。
2. **依赖 NVIDIA 生态**——是 NVIDIA 专有标注，AMD/ROCm 有对应 `roctx`，不通用。
3. **CPU range ≠ GPU 时长**——因异步，把 CPU range 宽度当 GPU 耗时是常见误读（见第 9 节）。
4. **标得太细会噪**——粒度需自己把控。

> 一句话收尾：**NVTX 是性能分析的"语义图层"，让 Nsight 时间线从一堆匿名 kernel 变成可读的业务地图；它不替你测速，但它让你知道"该去测哪里"。**

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同目录工具：[[llm-tools/nsight]] · [[llm-tools/Pytorch-Profiler]] · [[llm-tools/可视化]]
- 硬件/通信基础：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]]
- 优化对象：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理瓶颈：[[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/vllm/README]]
- 训练：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- 指标语义：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
