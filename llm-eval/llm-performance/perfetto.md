# Perfetto 性能分析

> 用一条交互式时间线（timeline）把 GPU kernel、通信、CPU 调度、内存拷贝逐微秒摊开，肉眼定位"谁在等谁"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-eval/llm-performance/推理性能测试]] [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：Perfetto 到底是什么 | trace 查看器 |
| 1 | 地基：trace / span / track 三个原子概念 | 时间线、嵌套区间 |
| 2 | 整体架构：从程序到时间线的数据流 | 采集→trace 文件→UI |
| 3 | timeline 怎么读：kernel / 通信 / 气泡 | track、bubble、空隙 |
| 4 | PyTorch Profiler 导出 Perfetto trace | `export_chrome_trace` |
| 5 | 瓶颈定位方法论：GPU 空闲 vs 通信暴露 | overlap、exposed comm |
| 6 | Perfetto SQL：从"看图"到"查表" | trace processor、SQL |
| 7 | 数值例子 + 对照 + 实践清单 | 手算气泡占比 |
| 8 | 常见问题表 | FAQ |

## 0. 一句话锚点

**Perfetto 是一个"打开 trace 文件、把所有事件按时间画成横向条带"的可视化与查询工具。** 它本身不产生数据，它只负责把别人采集到的"在某段时间内发生了什么"渲染成你能缩放、点击、测量的时间线，并且还能用 SQL 去查询这些事件。

- 它**解决什么**：性能问题的本质是"时间花在哪了"。光看吞吐数字（如 1200 tokens/s）你不知道慢在哪一步；Perfetto 把每个 GPU kernel、每次 NCCL 通信、每次 CPU launch、每次 `cudaMemcpy` 都画成一根带时长的横条，于是"GPU 空闲了 30%""通信没被算子盖住"这种问题一眼可见。
- 它**不是什么**：不是采集器（采集靠 PyTorch Profiler / Nsight / perfetto traced 等），不是 benchmark 框架（那是 [[llm-eval/llm-performance/推理性能测试]]）。它是"显微镜的目镜"，前面还得有"载玻片"（trace 文件）。
- **入口**：浏览器打开 <https://ui.perfetto.dev/>，把 `.json` / `.perfetto-trace` 文件拖进去即可，无需安装。

## 1. 地基：三个原子概念

性能 trace 不管来自哪个工具，本质上都由三种东西组成。先把这三个词钉死，后面全靠它们。

**(1) 事件 / span（区间）**：一件"有开始时间和结束时间"的事。比如一个 GPU kernel 从 `t=10.000ms` 跑到 `t=10.042ms`，这就是一个 span，时长 $42\mu s$。span 是时间线上的一根横条，条越长 = 耗时越久。

**(2) track（轨道）**：一行横向的"泳道"，同一个执行单元的事件画在同一行。常见 track：
- 每个 CPU 线程一条（Python 主线程、dataloader 线程……）
- 每个 CUDA **stream** 一条（GPU 上的指令队列）
- NCCL 通信常单独成 track

**(3) 嵌套（slice 栈）**：span 可以套 span。例如 `nn.Linear.forward`（外层，2ms）里面套 `aten::matmul`（1.8ms），里面再套 GPU kernel。Perfetto 用上下堆叠表示嵌套——外层在上、内层在下，像火焰图。

```
时间 →  0ms        1ms        2ms        3ms        4ms
CPU线程 |==forward(Linear)===|             |==backward==|
        |  |=aten::matmul==| |
GPU流0   |        |==sgemm kernel==|       |==dgrad kernel==|
            ^launch延迟        ^kernel真正在跑
```

> 关键直觉：**CPU track 上"发指令"和 GPU track 上"真在算"是两件事，中间有延迟。** 性能问题常藏在这个错位里。

## 2. 整体架构：数据怎么变成时间线

```
  你的训练/推理程序
        │  ① 插桩采集（PyTorch Profiler / Nsight / CUPTI）
        ▼
  原始事件流（kernel名、起止时间戳、stream、通信size...）
        │  ② 序列化
        ▼
  trace 文件  ─┬─ Chrome Trace JSON（.json，最常用、易导出）
              └─ Perfetto protobuf（.perfetto-trace，更省空间）
        │  ③ 拖进浏览器
        ▼
  Perfetto UI（ui.perfetto.dev）
        ├── Timeline 视图：横条可缩放/点击/框选测量
        └── Query(SQL)：trace processor 把事件存成表，可 SQL 聚合
```

三步要点：
- **①采集**：必须有人在程序里埋点。GPU 侧靠 NVIDIA 的 **CUPTI**（CUDA Profiling Tools Interface）拿到每个 kernel 的精确起止；CPU 侧靠 Profiler 记录 Python/aten 调用栈。
- **②文件格式**：两类。`Chrome Trace JSON` 人类可读、PyTorch 一行就能导（见第 4 节）；`Perfetto protobuf` 二进制、大 trace 更快。**两者 Perfetto UI 都能直接打开**。
- **③渲染**：纯前端，trace 不上传服务器（数据留在本地浏览器），适合公司内网。

## 3. timeline 怎么读：找 kernel、通信、气泡

打开 trace 后，你主要在三件事上花时间。

### 3.1 找到 GPU kernel

GPU 的 stream track 上每根横条就是一个 kernel。点开一根条，下方面板显示：kernel 名（如 `ampere_sgemm_128x128`）、时长、所属 stream。**横条越宽越值得优化**——按时长排序，最宽的几类 kernel 通常吃掉大部分时间（典型是 matmul/attention/layernorm）。

### 3.2 看通信（NCCL）

多卡训练里 `all-reduce`/`all-gather`/`reduce-scatter` 是独立的横条（常带 `nccl` 前缀）。你要判断的是它和计算 kernel 的**位置关系**：

```
理想（通信被算子盖住，overlap）：
GPU计算  |==层N前向==|==层N+1前向==|==层N+2前向==|
通信     |  |===all-reduce===|                          ← 藏在计算下面，不额外占时间

糟糕（通信暴露，exposed）：
GPU计算  |==层N前向==|              |==层N+1前向==|
通信                 |==all-reduce==|                    ← GPU 干等，这段是纯浪费
```

> **暴露通信（exposed communication）= 通信横条下面没有计算横条覆盖的那一段时间。** 这是分布式训练第一杀手，也是 Perfetto 最擅长揪出来的东西。

### 3.3 找气泡（bubble / GPU 空闲）

**气泡 = GPU stream track 上两根 kernel 之间的空白缝隙**，意思是这段时间 GPU 没活干、纯空转。气泡来源：

```
GPU流  |==kernelA==|        空白(气泡)        |==kernelB==|
                   └── 为什么空？ ───┐
   ├ CPU launch 太慢：Python 还没把 kernelB 发下来（CPU-bound）
   ├ 等数据：dataloader 没喂上 batch
   ├ 等同步：cudaStreamSynchronize / .item() / .cpu() 强制等
   └ 流水线气泡：pipeline 并行首尾的 warmup/cooldown
```

把 GPU track 所有 kernel 时长加起来，除以总墙钟时间，就是 **GPU 利用率（按时间）**：

$$\text{GPU\ busy\ ratio} = \frac{\sum \text{kernel 时长}}{\text{总墙钟时间}}$$

利用率低（比如 < 70%）说明气泡多，先去缝隙处点一点、看 CPU track 那一刻在干嘛。

## 4. PyTorch Profiler 导出 Perfetto trace

PyTorch 自带 profiler，能同时记录 CPU(aten 调用) 和 CUDA(kernel)，并一行导出 Chrome Trace JSON——这是产 Perfetto 输入最常用的路径。

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 同时抓 CPU 与 GPU
    record_shapes=True,        # 记录张量形状，便于事后归因到具体算子
    with_stack=True,           # 记录 Python 调用栈（开销略大）
    profile_memory=True,       # 记录显存分配（可选）
) as prof:
    for step in range(10):
        model(inputs)          # 你的前向/训练步
        prof.step()            # 配合 schedule 时推进阶段

# 关键：导出成 Perfetto/Chrome 能直接打开的 trace
prof.export_chrome_trace("trace.json")
```

参数的**含义与权衡**（讲含义，不背默认值，以官方文档为准）：

| 参数 | 含义 | 权衡 |
|---|---|---|
| `activities` | 抓哪些层（CPU、CUDA） | 不抓 CUDA 就看不到 kernel；抓得多 trace 越大 |
| `record_shapes` | 记张量形状 | 能区分"同名 kernel 不同 shape"，但增体积 |
| `with_stack` | 记 Python 栈 | 能把 kernel 归因到哪行代码；采样开销上升 |
| `profile_memory` | 记显存事件 | 排查 OOM/碎片有用；否则可关 |
| `schedule(wait,warmup,active,repeat)` | 只在指定步采集 | 避免把整段训练都录下来导致 trace 巨大；典型"跳过前几步预热，只录中间几步" |

> **为什么用 `schedule`**：第一步常含 cuDNN 算法搜索、显存预分配等一次性开销，会污染观测。让 profiler `wait` 几步、`warmup` 几步、只 `active` 录 2~3 步稳态，trace 小且代表性强。

导出后，把 `trace.json` 拖进 <https://ui.perfetto.dev/> 即可。

> 注：PyTorch 还可用 `on_trace_ready=torch.profiler.tensorboard_trace_handler(dir)` 写出供 TensorBoard 插件读取的格式；而 `export_chrome_trace` 产物是 Perfetto/Chrome `tracing` 都能开的 JSON。具体函数名/参数以你的 torch 版本官方文档为准。

## 5. 瓶颈定位方法论

不要漫无目的地滚动 timeline。按下面这个**判定树**走，几分钟锁定类型。

```
                 ┌─ GPU busy ratio 高(>90%)? ──是──> 计算瓶颈(compute-bound)
                 │                                    去看最宽的 kernel(matmul/attn)，
                 │                                    考虑算子融合/更优 kernel/低精度
 看 GPU 利用率 ──┤
                 │                          ┌─ 缝隙处 CPU 在忙 launch ─> CPU-bound
                 └─ 利用率低(有气泡) ──> 点缝隙 ┼─ 缝隙处在等通信     ─> 通信暴露
                                            ├─ 缝隙处在等 dataloader ─> 数据瓶颈
                                            └─ 缝隙前有 sync/.item() ─> 同步点拖累
```

四类典型瓶颈与"在 Perfetto 里长什么样 + 怎么治"：

**A. 计算瓶颈**：GPU track 几乎贴满，气泡极少。Perfetto 帮你找出最贵的 kernel 类别。对策：算子融合（减 launch 与访存）、混合精度/低精度、换更优 kernel（如 FlashAttention）。

**B. CPU-bound（launch 跟不上）**：GPU 频繁出现小气泡，缝隙正下方的 CPU track 在密集执行 Python/aten。说明 GPU 算得比 CPU 发指令还快。对策：CUDA Graph（把一串 launch 录成一次重放）、减少 Python 开销、增大 batch 摊薄。

**C. 通信暴露**：NCCL 横条有大段没被计算覆盖（见 3.2）。对策：计算/通信 overlap（如反向边算边 all-reduce、ZeRO 的通信重叠）、调通信桶大小、换更快互联（看 [[ai-infra/算力/GPU工作原理]] 里的 NVLink/PCIe 带宽）。

**D. 同步点**：`.item()`、`.cpu()`、`torch.cuda.synchronize()`、打印 loss 都会强制 CPU 等 GPU 排空，制造大气泡。Perfetto 里表现为一根 `cudaStreamSynchronize` 长条 + 其后 GPU 突然空一段。对策：减少不必要的 host-device 同步，loss 累积异步取。

> **核心心法**：先看"GPU 忙不忙"，再看"不忙的时候在等谁"。Perfetto 的价值就是把"在等谁"这件事变得可见。

## 6. Perfetto SQL：从"看图"到"查表"

肉眼适合定性，要定量就用 Perfetto 的 **Query (SQL)** 页。Perfetto 内部用 `trace processor` 把所有 span 存成一张 `slice` 表（每行一个事件，含 `name`、`dur` 时长、`track_id` 等）。例如按 kernel 名聚合总耗时：

```sql
SELECT name, COUNT(*) AS n, SUM(dur)/1e6 AS total_ms
FROM slice
GROUP BY name
ORDER BY total_ms DESC
LIMIT 10;
```

这比手动量横条精确得多，能直接产出"哪 10 类 kernel 吃掉 80% 时间"的表。`dur` 单位是纳秒，故除以 `1e6` 换算成毫秒。（表名/列名以 Perfetto 官方 schema 为准。）

## 7. 数值例子 + 对照 + 实践

### 7.1 手算：气泡占比与"治好"的收益

假设一个训练 step 的墙钟时间 $T=100\text{ms}$，从 Perfetto 量得 GPU kernel 时长合计 $\sum=70\text{ms}$：

$$\text{GPU busy} = \frac{70}{100} = 70\%,\quad \text{气泡} = 30\text{ms}\ (30\%)$$

再细看 30ms 气泡构成：通信暴露 20ms + dataloader 等待 6ms + 同步点 4ms。
若把通信完全 overlap（理论消掉那 20ms），新墙钟约 $100-20=80\text{ms}$：

$$\text{加速比} = \frac{100}{80} = 1.25\times,\quad \text{吞吐} \uparrow 25\%$$

**结论**：仅靠"让通信藏到计算下面"，吞吐就涨 1/4——这就是为什么定位暴露通信优先级极高。

### 7.2 手算：一次 all-reduce 该花多久（判断是否异常）

Ring all-reduce 传输的数据量约为 $2(N-1)/N \times S$（$N$=卡数，$S$=参数字节数）。设 $N=8$、梯度 $S=1.4\text{GB}$（7 亿参数 × fp16，约 1.4GB），单卡有效互联带宽 $BW\approx 200\text{GB/s}$（NVLink 量级，约，以官方为准）：

$$\text{通信量} \approx 2\times\frac{7}{8}\times1.4 \approx 2.45\text{GB}$$
$$t_{\text{allreduce}} \approx \frac{2.45\text{GB}}{200\text{GB/s}} \approx 12.3\text{ms}$$

若 Perfetto 里量到 all-reduce 横条 ≈ 12ms，说明带宽打满、属正常；若量到 40ms，则八成是走了慢链路（PCIe 而非 NVLink）或小桶太多——这就是 Perfetto 数字与理论值对照的用法。

### 7.3 工具对照

| 工具 | 角色 | 与 Perfetto 关系 |
|---|---|---|
| **Perfetto** | trace 查看 + SQL 查询 | 本文主角，看图与查表 |
| **PyTorch Profiler** | 采集器 | 产出喂给 Perfetto 的 `trace.json` |
| **Nsight Systems** | NVIDIA 系统级采集+查看 | 更深的 GPU/驱动细节，自带查看器；定位更底层时用 |
| **TensorBoard Profiler** | 可视化+建议 | 上手简单、给优化建议；细粒度不如 Perfetto |
| **`chrome://tracing`** | 老牌 Chrome 查看器 | 同格式 JSON，Perfetto 是其现代替代品 |

一句话取舍：**日常 PyTorch 调优 → Profiler 导 JSON + Perfetto 看**；要钻到驱动/SM 占用级别 → Nsight Systems；只想要一键建议 → TensorBoard。

### 7.4 实践清单

- 用 `schedule` 只录 2~3 个稳态 step，别录整个 epoch（trace 会几百 MB）。
- 录之前 warmup，避开首步的 cuDNN 算法搜索/显存预分配假象。
- 先看 GPU busy ratio 定大方向，再点气泡查"在等谁"。
- 通信问题永远先确认是否 overlap，再谈换硬件。
- 用 SQL `SUM(dur) GROUP BY name` 量化，别只靠肉眼估横条。
- trace 不上传服务器，内网/敏感模型可放心用。
- 大 trace 加载慢就改用 `.perfetto-trace`(protobuf) 而非 JSON。

## 常见问题

| 问题 | 答 |
|---|---|
| Perfetto 自己采数据吗？ | 不。它只看/查 trace，采集靠 Profiler/Nsight/CUPTI |
| trace 文件怎么来？ | PyTorch `prof.export_chrome_trace("trace.json")` 最常用 |
| 怎么知道是计算还是通信瓶颈？ | 看 GPU busy ratio；高=计算瓶颈，低=点气泡看在等谁 |
| 什么是"气泡"？ | GPU stream 上两 kernel 间的空白，GPU 空转的时间 |
| 什么是"通信暴露"？ | 通信横条没被计算覆盖的那段，纯浪费 |
| trace 会泄密吗？ | 不上传，纯前端本地渲染，适合内网 |
| trace 太大打不开？ | 用 `schedule` 少录几步；或导 protobuf 格式 |
| 看不到 kernel 只有 CPU？ | profiler `activities` 没加 `ProfilerActivity.CUDA` |
| 为什么 GPU 比 CPU 还快却在等？ | CPU launch 跟不上(CPU-bound)，考虑 CUDA Graph |
| Perfetto vs Nsight？ | Perfetto 看 PyTorch 级 trace；Nsight 钻驱动/SM 级 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，回到总图
- [[llm-eval/llm-performance/推理性能测试]] — 先量吞吐/延迟数字，再用 Perfetto 看时间花哪了
- [[ai-infra/算力/GPU工作原理]] — stream / kernel / NVLink 带宽的底层机制，看懂 timeline 的前置
- 官方 UI：<https://ui.perfetto.dev/>
- 官方文档：<https://perfetto.dev/>
