# 大模型性能评测（LLM Performance Evaluation）

> 用一套可复现的方法，量化"模型在某硬件+某框架上跑得多快、多省、多稳"，把"感觉很快"变成"P99 首 token 280ms、吞吐 3200 tok/s"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-eval/README]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] [[llm-inference/vllm/README]] [[ai-infra/网络/NCCL]]

本目录是 llm-action 中"性能评测"的总入口。它不评模型"答得对不对"（那是 [[llm-eval/llm-precision/README]] 精度评测的事），而专门评"跑得快不快、扛不扛得住压力、硬件用得满不满"。

---

## 阅读地图

| 你想知道的 | 看哪一节 | 对应文件/工具 |
|---|---|---|
| 性能评测到底在测什么 | §0、§1 | 本文 |
| 训练 / 推理性能各看哪些指标 | §2、§3 | [[大模型场景下训练和推理性能指标名词解释]] |
| 推理评测的整体流程 | §4 | 本文 |
| 用什么工具压测 | §5 | llmperf / vllm-benchmark / tgi-benchmark / wrk / Locust |
| 怎么定位瓶颈（哪一段慢） | §6 | perfetto / nsys / pynvml |
| GPU 利用率怎么读 | §7 | pynvml / nvidia-smi / DCGM |
| 报告怎么写、怎么避坑 | §8、常见问题 | 本文 |

---

## 0. 一句话锚点

**性能评测 = 在"固定的负载（workload）"下，测量"延迟、吞吐、资源占用"这三类指标，并保证可复现。**

- **延迟（Latency）**：单个请求多久有反应、多久答完。用户体感。
- **吞吐（Throughput）**：单位时间系统能处理多少 token / 多少请求。成本与容量。
- **资源占用（Utilization）**：GPU 算力/显存/带宽用了多少。钱花得值不值。

这三者互相拉扯：你想吞吐高就得加大 batch，但 batch 一大单请求延迟就上去了。性能评测的本质，就是**画出"延迟 vs 吞吐"这条权衡曲线**，再根据业务 SLA 选工作点。

---

## 1. 地基：性能评测解决什么问题

没有评测时，团队常陷入这些困境：

1. **选型靠玄学**：vLLM 和 TGI 谁快？换 H100 值不值？没数据，只能拍脑袋。
2. **容量规划无依据**：一张卡能扛多少 QPS？上线要几张卡？算不出来就只能超配，烧钱。
3. **优化没有标尺**：开了量化/PagedAttention/投机解码，到底快了多少？没基线就不知道。
4. **线上事故复盘难**：P99 飙到 5 秒，是模型慢、排队慢、还是网络慢？没分段指标查不出。

性能评测就是为这些问题提供**客观、可对比、可复现**的数字。一句话：**没有测量就没有优化。**

```
        ┌─────────────────────────────────────────────┐
        │              为什么要做性能评测                │
        ├──────────────┬──────────────┬───────────────┤
        │   选型对比     │   容量规划     │   优化验证      │
        │ vLLM vs TGI  │ 1卡扛多少QPS │ 量化前后对比    │
        │ H100 vs A100 │ 上线要几卡   │ 投机解码收益    │
        └──────┬───────┴──────┬───────┴───────┬───────┘
               │              │               │
               └──────────────┼───────────────┘
                              ▼
                  统一的指标 + 可复现的方法
```

---

## 2. 训练性能：看哪些指标

训练评测关心"把一个 step 算完要多久、卡用得满不满、多卡扩展得好不好"。核心指标：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| **Throughput（吞吐）** | tokens/s 或 samples/s | 直接决定训练一个 epoch 要多久 |
| **GPU 利用率** | SM 占用率（不是 nvidia-smi 的"显存"那一栏） | 低说明 GPU 在等数据/等通信，有优化空间 |
| **MFU** | Model FLOPs Utilization，实测算力 / 理论峰值算力 | 衡量"硬件潜力发挥了几成"，通常 40%~60% 算不错 |
| **单 step 耗时** | 一次前向+反向+优化器更新的时间 | 拆出 compute / communication / data 三段 |
| **扩展效率** | N 卡吞吐 / (1 卡吞吐 × N) | 看 [[ai-framework/megatron-lm/README]] 的并行策略是否高效 |

> MFU 直觉公式（概念示意，具体以官方实现为准）：
> $$\text{MFU} = \frac{\text{实际每秒完成的模型 FLOPs}}{\text{GPU 理论峰值 FLOPs}}$$
> 例如某卡理论峰值 312 TFLOPS（FP16），实测有效算力 156 TFLOPS，则 MFU ≈ 50%。

训练性能的具体测法见 [[llm-eval/llm-performance/训练性能测试]]，指标定义见 [[大模型场景下训练和推理性能指标名词解释]]。

---

## 3. 推理性能：看哪些指标（最常考、最常用）

推理评测是本目录的重头戏。LLM 推理分两个阶段，指标也随之分两类：

```
  请求到达 ────► [排队] ────► Prefill（预填充） ────► Decode（解码，逐 token） ────► 完成
                          一次算完整个 prompt      自回归，一个一个吐 token
                                │                        │
                                ▼                        ▼
                        决定 TTFT（首 token）      决定 TPOT（每 token 间隔）
```

| 指标 | 全称 / 含义 | 受什么影响 |
|---|---|---|
| **TTFT** | Time To First Token，从发请求到收到第 1 个 token | prompt 长度、排队、prefill 算力 |
| **TPOT / ITL** | Time Per Output Token / Inter-Token Latency，相邻 token 间隔 | decode 速度、batch 大小、显存带宽 |
| **E2E Latency** | 端到端总时延 ≈ TTFT + TPOT × (输出 token 数 − 1) | 上面两项叠加 |
| **Throughput** | 系统每秒吐出的总 token 数（所有并发请求合计） | batch、显存、并发数 |
| **Goodput** | 满足 SLA 约束（如 TTFT<X）的有效吞吐 | 比裸吞吐更贴近业务 |
| **并发数 / QPS** | 同时在跑的请求数 / 每秒请求数 | 压测的输入变量 |

**关键直觉**：
- 用户体感的"反应快慢"≈ **TTFT**；"打字快慢"≈ **TPOT**。
- 老板关心的"成本"≈ **吞吐**（每 token 成本 = 卡成本 / 吞吐）。
- **延迟和吞吐是反向的**：调大 batch → 吞吐↑ 但单请求延迟↑。

> 端到端时延的拆解（设输出 $n$ 个 token）：
> $$T_{e2e} = T_{TTFT} + (n-1)\times T_{TPOT}$$
> 例：TTFT=300ms，TPOT=20ms，输出 200 token → $T_{e2e} = 300 + 199\times20 = 4280\text{ms}$。可见**输出越长，TPOT 的权重越大**，长文本生成场景优化重点在 decode。

---

## 4. 推理评测的标准流程

无论用哪个工具，一次靠谱的评测都遵循同一套流程。下面是"调用链"视角：

```
 ┌──────────────┐   ①定义负载        ┌──────────────────┐
 │  测试者/脚本   │ ─────────────────► │  Benchmark 客户端  │
 │ (你的目标)    │  输入/输出长度分布   │ llmperf / wrk /   │
 └──────────────┘  并发/QPS/请求数    │ Locust / vllm-bench│
                                      └─────────┬────────┘
                                   ②发请求(HTTP/gRPC)│  并发打流
                                                  ▼
                                      ┌──────────────────┐
                                      │   推理服务 (SUT)   │  System Under Test
                                      │ vLLM/TGI/TRT-LLM  │  被测对象
                                      └─────────┬────────┘
                                   ③逐 token 返回 │  记录每个 token 的时间戳
                                                  ▼
                                      ┌──────────────────┐
                                      │  指标聚合 & 报告   │  TTFT/TPOT/吞吐
                                      │  P50/P90/P99 分位  │  + GPU 利用率
                                      └──────────────────┘
```

**七步法**：

1. **明确目标与 SLA**：是选型？容量规划？还是验证优化？SLA 比如"TTFT P99 < 500ms"。
2. **固定环境**：模型、权重精度（FP16/INT8/FP8）、框架版本、GPU 型号、驱动、并行策略——**全部记下来**，否则结果不可复现。
3. **设计负载**：输入长度、输出长度（用固定值还是真实分布）、并发数扫描区间。用真实业务的长度分布最有说服力。
4. **预热（warmup）**：先打一批请求让显存分配、编译缓存（如 CUDA graph、TRT engine）就位，**丢弃预热阶段数据**，否则首批请求会拖慢统计。
5. **扫描并发**：从低并发逐步加压（1, 2, 4, 8, …），直到吞吐不再上涨或延迟超 SLA，画出延迟-吞吐曲线。
6. **采集分位数**：报 **P50/P90/P99**，不要只报平均值（均值会掩盖长尾）。
7. **同时记录资源**：GPU 利用率、显存、功耗（见 §7），判断瓶颈在算力还是带宽。

---

## 5. 常用压测/评测工具一览

| 工具 | 定位 | 适用 | 仓库内文档 |
|---|---|---|---|
| **llmperf** | LLM 专用，原生支持 TTFT/TPOT 流式指标 | 推理服务标准评测 | [[llm-eval/llm-performance/llmperf]] |
| **vLLM benchmark** | vLLM 自带 `benchmark_serving.py` 等脚本 | 测 vLLM、对比框架 | [[llm-eval/llm-performance/vllm-benchmark]] |
| **TGI benchmark** | HuggingFace TGI 自带压测 | 测 TGI 服务 | [[llm-eval/llm-performance/tgi-benchmark]] |
| **GenAI-Perf** | Triton 团队出品，统一测各类生成式服务 | 跨框架、OpenAI 兼容接口 | （perf_analyzer 套件） |
| **EvalScope** | ModelScope，精度+性能一体 | 既测质量又测性能 | [[llm-eval/EvalScope]] |
| **wrk** | 通用 HTTP 压测，轻量高并发 | 粗测 QPS/延迟，不懂流式 | [[llm-eval/llm-performance/wrk-性能测试工具]] |
| **Locust** | Python 写场景脚本，易扩展 | 复杂业务流、自定义指标 | mindie/locust-lantency-throughput |
| **perfetto / nsys** | 时间线追踪，分段定位瓶颈 | 找"哪一段慢" | [[llm-eval/llm-performance/perfetto]] |

**选型建议**：
- 想要**开箱即用的 LLM 指标**（TTFT/TPOT/分位数）→ 用 **llmperf** 或 **GenAI-Perf**，它们理解流式 token。
- 只测某个框架的极限吞吐 → 用该框架**自带的 benchmark**（口径最准）。
- 通用 HTTP 接口、不关心 token 级指标 → **wrk**（极轻量）或 **Locust**（要写复杂场景）。
- ⚠️ **wrk/Locust 默认按"整次响应"算延迟**，对流式 LLM 不能直接得到 TTFT，需要自己解析 SSE/chunk 时间戳。

---

## 6. 瓶颈定位：哪一段慢了

光有"总时延 4 秒"还不够，要拆开看钱花在哪。常见瓶颈与判断方法：

| 现象 | 可能瓶颈 | 怎么确认 |
|---|---|---|
| TTFT 高，TPOT 正常 | prefill 慢 / 排队久 | 看队列等待时间、prompt 长度 |
| TPOT 高 | decode 受限于**显存带宽** | GPU 算力没满但带宽打满 |
| 吞吐上不去，GPU 利用率低 | batch 太小 / 调度未连续批处理 | 检查是否开 continuous batching |
| 多卡扩展差 | 通信瓶颈（[[ai-infra/网络/NCCL]]） | 用 nsys 看 NCCL 占比 |
| 偶发长尾 P99 | 显存碎片 / GC / 抢占 | 看时间线毛刺 |

定位工具：**perfetto / Nsight Systems（nsys）** 抓时间线，能看到 kernel、内存拷贝、NCCL 通信各占多少。这部分见 [[llm-eval/llm-performance/perfetto]]。

> 经验法则：**LLM decode 阶段几乎总是 memory-bound（显存带宽受限）而非 compute-bound**。所以 [[llm-optimizer/kv-cache]] 优化、量化、[[llm-optimizer/FlashAttention]] 这类"减少访存"的手段对 TPOT 提升明显。

---

## 7. GPU 资源采集：pynvml / nvidia-smi / DCGM

评测时必须同步记录硬件占用，否则无法判断瓶颈，也无法算成本。

**三个层次的工具**：

```
 ┌────────────────────────────────────────────────────┐
 │  应用内嵌    pynvml (Python 调 NVML 库)              │  细粒度、可写进脚本
 │             nvidia-ml-py / nvidia-ml-py3            │
 ├────────────────────────────────────────────────────┤
 │  命令行      nvidia-smi (人眼看 / 简单脚本轮询)        │  最方便、最常用
 ├────────────────────────────────────────────────────┤
 │  集群级      DCGM / dcgm-exporter + Prometheus       │  生产监控、多机
 └────────────────────────────────────────────────────┘
```

**关键认知坑**：`nvidia-smi` 里的 **"GPU-Util" 不等于算力利用率**！它表示"过去一段时间内有 kernel 在跑的时间比例"，哪怕只用了 5% 的 SM，只要一直有 kernel 在跑就显示 100%。真正衡量算力发挥要看 **SM 占用率 / MFU**（需要 DCGM 或 nsys 才能拿到）。

**pynvml 最小示例**（取自本目录原始内容，仅演示读显存，函数名以官方库为准）：

```python
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")
```

> 安装：`pip install nvidia-ml-py psutil`。NVML 还能读功耗、温度、时钟频率、ECC 错误等。具体 API 以 NVIDIA NVML 官方文档为准，**不要凭记忆背函数名**。

**要采集的字段**：显存占用、GPU-Util、功耗（算每 token 能耗）、温度（看是否降频）、显存带宽利用率（判断 memory-bound）。

---

## 8. 报告该怎么写

一份合格的性能报告至少包含：

1. **环境清单**：模型+权重精度、框架及版本、GPU 型号×数量、驱动/CUDA 版本、并行策略。
2. **负载定义**：输入/输出长度（或分布）、并发数、总请求数、是否流式。
3. **核心指标表**：TTFT / TPOT / E2E 的 **P50/P90/P99**、总吞吐（tok/s）、QPS。
4. **延迟-吞吐曲线**：随并发变化的曲线，标出 SLA 工作点。
5. **资源占用**：GPU-Util、显存、功耗。
6. **结论**：每 token 成本、推荐工作点、瓶颈定位。

**口径示例**（数值仅为格式演示，非真实基准）：

```
模型: Qwen-7B (FP16)  |  框架: vLLM  |  硬件: 1×A100-80G
负载: 输入512 / 输出128 token, 流式, 64并发, 共2000请求
─────────────────────────────────────────────────────
指标         P50      P90      P99
TTFT(ms)     180      260      410
TPOT(ms)     18       22       35
E2E(ms)      2470     2900     3600
─────────────────────────────────────────────────────
总吞吐: 3200 tok/s   |   QPS: 25   |   GPU-Util: 92%
```

---

## 常见问题 / 坑

| 坑 | 后果 | 正确做法 |
|---|---|---|
| 只报平均值不报分位数 | 掩盖长尾，线上 P99 爆炸 | 必报 **P50/P90/P99** |
| 不做 warmup | 首批含编译/分配开销，数据偏慢 | 预热并**丢弃预热数据** |
| 输入输出长度都用固定值 | 与真实业务不符，结论失真 | 用真实**长度分布**复测 |
| 客户端成瓶颈 | 测的是客户端不是服务 | 客户端单独成机/多进程发压 |
| 用整次响应算 LLM 延迟 | 拿不到 TTFT，掩盖流式体验 | 解析每个 token 的时间戳 |
| 把 nvidia-smi GPU-Util 当算力利用率 | 误判"已经满了" | 看 **SM 占用 / MFU**（DCGM/nsys） |
| 环境没记全 | 结果无法复现/对比 | 锁定并记录所有版本与配置 |
| 平均吞吐忽略 SLA | 高吞吐但大量请求超时 | 用 **Goodput**（满足 SLA 的有效吞吐） |
| 网络/序列化开销算进模型 | 误判模型慢 | 分段计时（排队/网络/计算分开） |

---

## 🔗 跳转链接

- 返回知识地图：[[00-知识地图]]
- 指标定义详解：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 评测总览：[[llm-eval/README]]
- 工具：[[llm-eval/llm-performance/llmperf]] · [[llm-eval/llm-performance/vllm-benchmark]] · [[llm-eval/llm-performance/tgi-benchmark]] · [[llm-eval/llm-performance/wrk-性能测试工具]] · [[llm-eval/llm-performance/perfetto]] · [[llm-eval/EvalScope]]
- 被测框架：[[llm-inference/vllm/README]] · [[llm-inference/sglang/README]] · [[llm-inference/tensorrt/README]]
- 优化相关：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[llm-compression/quantization/量化基础]]
- 底层支撑：[[ai-infra/网络/NCCL]] · [[ai-infra/ai-cluster/README]] · [[ai-infra/ai-hardware/README]]
