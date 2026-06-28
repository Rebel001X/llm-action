# Triton Inference Server（NVIDIA 推理服务框架）

> Triton 是 NVIDIA 开源的「通用推理服务器」：把任意框架训练出的模型（PyTorch / TensorFlow / ONNX / TensorRT / Python …）统一包成一个**高吞吐、多模型、可并发**的 HTTP/gRPC 服务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] [[llm-inference/tensorrt/README]] [[llm-inference/tensorrt-llm/README]] [[llmops/kubernetes]]

## 阅读地图

| 你想知道 | 看哪节 | 一句话 |
|---|---|---|
| Triton 到底是什么、解决什么问题 | §0 §1 | 把"模型文件"变成"在线服务"的通用容器 |
| 整体长什么样 | §2 | 前端协议 → 调度器 → Backend → 后端框架 |
| 模型仓库怎么组织 | §3 | 约定式目录 + `config.pbtxt` |
| `config.pbtxt` 各字段干嘛的 | §4 | 平台 / 输入输出 / 批大小 / 实例 / 动态批 |
| 怎么把多请求拼成大 batch | §5 | Dynamic Batching 攒一小会儿再算 |
| 怎么多模型/多副本并行 | §6 | Instance Groups + 并发执行 |
| 怎么把"前处理→模型→后处理"串起来 | §7 | Ensemble / BLS 业务逻辑脚本 |
| LLM 场景怎么用 | §8 | TensorRT-LLM Backend / vLLM Backend |
| 怎么跑起来、怎么压测 | 部署节 | `docker run` + `perf_analyzer` |
| 容易踩的坑 | 常见问题 | 名字对不上、维度对不上、显存炸 |

## 0. 一句话锚点

> **Triton = 一个"推理框架无关"的 Web 服务器**。你只要把模型按约定放进「模型仓库」，写一个 `config.pbtxt` 描述它的输入输出，Triton 就帮你搞定：HTTP/gRPC 协议、并发调度、动态批处理、多模型管理、GPU 共享、监控指标。你专注模型，它专注"把模型变成服务"。

```
                        ┌──────────── Triton Inference Server (一个容器) ────────────┐
  客户端                │                                                            │
  (HTTP/gRPC) ─请求──► │  [前端协议层] ─► [按模型路由] ─► [Per-Model 调度器] ─► [Backend] │ ─► GPU/CPU
                        │      ▲                              │ 动态批处理            │
  ◄──── 响应 ───────────│      └──────────── 组装结果 ◄────────┘                       │
                        │  同时托管:  model_A / model_B / ensemble_C ...               │
                        └────────────────────────────────────────────────────────────┘
```

## 1. 地基：它到底解决什么问题

把一个训练好的模型推上线，看似简单，实则要处理一大堆"和模型无关、但每次都要重写"的工程问题：

1. **协议**：怎么收 HTTP/gRPC 请求、怎么序列化张量、怎么返回结果。
2. **并发**：100 个用户同时请求，怎么不互相阻塞、怎么共享一张 GPU。
3. **攒批（batching）**：GPU 喜欢大矩阵，单条请求利用率极低；要把零散请求拼成 batch 才划算。
4. **多模型共存**：一台机器上同时跑检测+识别+排序三个模型，显存/算力怎么分。
5. **框架异构**：A 模型是 PyTorch、B 是 ONNX、C 是 TensorRT，难道每个都写一套服务？
6. **可观测**：QPS、延迟、GPU 利用率、队列长度，运维要能看到。

**Triton 的价值**：把上面 1–6 全部做成"标准化基础设施"，对任意后端框架统一暴露。你写的代码量从"一整套服务"压缩到"一个配置文件（有时再加一段 Python）"。

> 类比：Triton 之于推理，约等于 **Nginx + 应用服务器** 之于网页——它不"产生内容"（不训练模型），但负责把内容**高效、并发、可观测**地送出去。

### 1.1 和同类的关系（重要，别搞混）

| 名字 | 是什么 | 和 Triton 的关系 |
|---|---|---|
| **Triton Inference Server** | 通用**推理服务器**（本文主角） | 顶层服务/调度框架 |
| **TensorRT** | NVIDIA 的**推理优化引擎**（编译模型→`.plan`） | Triton 的一种 backend，负责"算得快" |
| **TensorRT-LLM** | 专为 LLM 的 TensorRT 扩展 | 通过 TRT-LLM backend 接入 Triton |
| **OpenAI Triton** | 一种写 **GPU kernel** 的编程语言/编译器 | **完全不同的东西，只是重名！** |

> 强调：本文的 Triton 指 **Inference Server**，不是写算子的那个 OpenAI Triton 语言。两者毫无关系，只是撞名。

## 2. 整体架构（从请求到 GPU 的调用链）

Triton 内部是清晰的分层流水线，理解这条链路就理解了 Triton：

```
请求 ─► ① 前端协议层 (HTTP/REST | gRPC | C-API)
        │     解析协议、反序列化输入张量
        ▼
     ② 路由：根据请求里的 model_name / version 找到对应模型
        ▼
     ③ Per-Model 调度器 (Scheduler)
        │   - Default：直接排队
        │   - Dynamic Batching：攒一小会儿，把多条请求拼大 batch
        │   - Sequence Batching：把同一会话的多次请求路由到同一实例(带状态)
        ▼
     ④ Backend (执行引擎，一种"框架适配器")
        │   pytorch / onnxruntime / tensorrt / python / tensorrtllm / vllm ...
        │   每个模型可有多个 Instance(副本) 并发执行
        ▼
     ⑤ 后端框架真正算前向 (GPU/CPU)
        ▼
     ⑥ 组装输出张量 ─► 序列化 ─► 返回客户端
```

**关键概念三件套**：
- **Backend（后端）**：一个共享库（`.so`），告诉 Triton"这类模型怎么加载、怎么执行"。每种框架一个 backend。
- **Model（模型）**：模型仓库里的一个目录，含权重文件 + `config.pbtxt`。
- **Instance（实例）**：模型在某个 GPU/CPU 上的一份运行副本，是并发执行的最小单位。

## 3. 模型仓库（Model Repository）——约定式目录

Triton 启动时通过 `--model-repository=<路径>` 指向一个目录，里面**每个子目录就是一个模型**。结构是强约定的：

```
model_repository/                ← 用 --model-repository 指向这里
├── resnet50/                    ← 模型名(=请求里的 model_name)
│   ├── config.pbtxt             ← 模型配置(协议契约)
│   ├── labels.txt               ← (可选)标签文件
│   └── 1/                       ← 版本号目录(必须是数字)
│       └── model.pt             ← 权重(文件名由 backend 约定)
├── text_detect/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── my_pipeline/                 ← Ensemble(把多个模型串成流水线)
    ├── config.pbtxt
    └── 1/                       ← Ensemble 的版本目录可为空
```

要点：
- **版本目录是数字**（`1/`、`2/`…）。Triton 支持版本策略：最新 N 个 / 指定 / 全部。
- **权重文件名由 backend 决定**：PyTorch 用 `model.pt`、ONNX 用 `model.onnx`、TensorRT 用 `model.plan`、Python backend 用 `model.py`。具体命名以官方文档为准。
- 模型仓库可放本地盘，也可放 S3 / GCS / 远程，方便云上部署。

## 4. `config.pbtxt`——模型与服务之间的"契约"

这是 Triton 的灵魂文件，用 protobuf 文本格式写。它声明"这个模型对外长什么样"。下面用本仓库里 `resnet50/config.pbtxt` 真实例子逐字段讲（字段含义讲清楚，**默认值与精确语义以官方文档为准**）：

```protobuf
name: "resnet50"               # 模型名，须与目录名一致
platform: "pytorch_libtorch"   # 用哪个后端/平台来加载执行
max_batch_size: 0              # 最大批大小；0 = 不做自动批维度(输入已含batch)
input [
  {
    name: "input__0"           # 输入张量名，客户端发请求时必须用这个名字
    data_type: TYPE_FP32       # 数据类型
    dims: [ 3, 224, 224 ]      # 形状(不含batch维)；-1 表示该维可变
    reshape { shape: [ 1, 3, 224, 224 ] }   # 喂给后端前重塑形状
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1, 1000, 1, 1 ]
    reshape { shape: [ 1, 1000 ] }
    label_filename: "labels.txt"  # 输出可关联标签文件，直接返回类别名
  }
]
```

### 4.1 几类核心配置项（按"做什么用"分类记）

| 配置项 | 作用 | 怎么权衡 |
|---|---|---|
| `platform` / `backend` | 指定用哪个执行后端 | 跟着模型格式走：`.pt`→pytorch，`.onnx`→onnxruntime，`.plan`→tensorrt，自定义逻辑→python |
| `max_batch_size` | 是否启用"自动批维度"及上限 | 0=输入里自带 batch；>0=Triton 帮你把 batch 维加在最前并自动拼批 |
| `input` / `output` | 声明张量**名字/类型/形状** | 名字必须和客户端、和模型真实 IO 一一对应；可变维写 `-1` |
| `instance_group` | 每个模型起几份副本、放哪些 GPU | 见 §6，副本越多并发越高但越吃显存 |
| `dynamic_batching` | 启用动态批处理 | 见 §5，靠"攒批"换吞吐 |
| `sequence_batching` | 有状态模型(同一会话连续请求) | 用于 RNN/对话等需要保持上下文的场景 |
| `version_policy` | 加载哪些版本 | latest N / specific / all，灰度上线常用 |
| `optimization` | 后端级优化开关 | 如开启 TensorRT 加速、FP16/INT8 等，按 backend 支持情况 |

> 一个最常见的报错根源：`config.pbtxt` 里写的 `name`/`dims`/`data_type` 和模型真实的输入输出**对不上**。Triton 不会替你猜，对不上就直接拒绝加载或推理报错。

## 5. Dynamic Batching（动态批处理）——吞吐的关键

单条请求喂给 GPU，矩阵又小又零碎，算力利用率极低。Triton 的动态批处理会**短暂等待一小段时间**，把这段时间内到达的多条请求**拼成一个大 batch**一次算完。

```
时间轴 ─────────────────────────────────────────►
请求到达:  R1      R2  R3        R4
           │       │   │         │
           ▼       ▼   ▼         ▼
        ┌────────── 队列(攒批窗口 max_queue_delay) ──────────┐
        │  R1 R2 R3  ── 到时间/到批量上限 ──► 拼成 batch[3] ─► GPU │
        └──────────────────────────────────────────────────┘
                                R4 进入下一个批 ...
```

核心权衡——**延迟 vs 吞吐的跷跷板**：
- 攒批窗口（如 `max_queue_delay_microseconds`）设大 → batch 更大、吞吐更高，但每条请求多等一会儿，**延迟升高**。
- 设小 → 延迟低，但 batch 小，**吞吐下降**。
- 实践：在线低延迟场景设小窗口；离线批量/可容忍延迟场景设大窗口。具体字段名与单位以官方文档为准。

> 数值直觉：假设单请求和 8 请求一个 batch 的 GPU 算时都约 20ms（GPU 没吃满时近似不变），那么"攒到 8 条再算"的吞吐约是逐条算的 **8 倍**，而每条只多等了攒批的几毫秒——这就是动态批处理几乎"白赚吞吐"的原因。

## 6. 并发执行：Instance Groups

`instance_group` 决定一个模型起**几份副本**、放在**哪些设备**上。多个实例可在同一/不同 GPU 上**并发执行**，互不阻塞。

```
instance_group [
  { count: 2  kind: KIND_GPU  gpus: [0] }   # GPU0 上放 2 个副本
  { count: 1  kind: KIND_GPU  gpus: [1] }   # GPU1 上放 1 个副本
]
```

权衡：
- 副本多 → 并发高、GPU 利用率高；但每副本各占一份显存，**显存是上限**。
- 还可配合 CUDA Stream，让多个实例在 GPU 上重叠执行。
- 多个不同模型也能共享同一张 GPU（Triton 的多模型并发是核心卖点）。

## 7. 把多步串成流水线：Ensemble 与 BLS

真实业务很少只调一个模型，常是"前处理 → 模型A → 模型B → 后处理"。Triton 提供两种方式编排：

**(1) Ensemble（声明式流水线）**：在 `config.pbtxt` 里用 DAG 把多个模型的输入输出连起来，Triton 自动按图调度。优点是数据在服务端内部流转，**少一次网络往返**。

```
请求 ─► [preprocess(python)] ─► [detect(onnx)] ─► [recognize(tensorrt)] ─► [postprocess(python)] ─► 响应
        └──────────────── 一个 Ensemble 模型，对客户端是单次调用 ───────────────┘
```

**(2) BLS（Business Logic Scripting，命令式）**：在 Python backend 里写代码，运行时按 if/else、循环动态决定调哪些模型。比 Ensemble 灵活，适合带条件分支/循环的复杂逻辑（LLM 的多步推理常用）。

| 维度 | Ensemble | BLS |
|---|---|---|
| 形式 | 声明式 DAG（静态图） | 命令式 Python 代码 |
| 灵活性 | 固定流程 | 可分支/循环/动态 |
| 适用 | 标准前后处理流水线 | 复杂控制流、LLM 编排 |

## 8. LLM 场景下的 Triton

普通 LLM 推理（自回归、KV Cache、连续批处理）不是裸 backend 能高效搞定的，需要专门后端：

- **TensorRT-LLM Backend**：把 TensorRT-LLM 编译出的引擎挂到 Triton，由 Triton 负责服务层（协议/调度/in-flight batching 编排）。是 NVIDIA 卡上高性能 LLM 服务的主路线。详见 [[llm-inference/tensorrt-llm/README]]。
- **vLLM Backend**：把 vLLM 作为 Triton 的一个 Python-based backend，借 Triton 的服务能力 + vLLM 的 PagedAttention/连续批处理。
- **In-flight / Continuous Batching**：LLM 每条请求生成长度不同，传统"等齐再返回"浪费严重。LLM backend 配合**持续批处理**，完成的请求即时退出、新请求即时加入，显著提升吞吐。

```
传统静态批:  [r1 r2 r3] 一起算到最长那条结束 → 短请求干等(浪费)
持续批处理:  r1完成立刻出队/返回, 空位立刻塞入新请求 r4 → GPU 不空转
```

## 部署与压测（典型流程）

**(1) 拉起服务**（NGC 官方容器，标签按你的 CUDA/驱动选，**精确版本以 NGC 为准**）：

```bash
# 启动 Triton，挂载本地模型仓库
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<tag>-py3 \
  tritonserver --model-repository=/models
```

三个端口约定俗成：**8000=HTTP**、**8001=gRPC**、**8002=Metrics(Prometheus)**。

**(2) 客户端 SDK 容器**（本仓库原文件里就是它，用于 `perf_analyzer`/示例客户端）：

```bash
docker run --privileged --gpus all -it --net=host -v ${PWD}:/workspace/ \
  nvcr.io/nvidia/tritonserver:23.05-py3-sdk bash
```

**(3) 健康检查 / 元信息**（REST）：

```bash
curl -v localhost:8000/v2/health/ready          # 服务是否就绪
curl -s  localhost:8000/v2/models/resnet50      # 某模型元信息
```

**(4) 压测**：用 `perf_analyzer` 灌不同并发，观察吞吐与 P99 延迟，调 `dynamic_batching` 与 `instance_group`：

```bash
perf_analyzer -m resnet50 --concurrency-range 1:8
```

> 调优套路：先看 GPU 利用率没吃满 → 加 `dynamic_batching` 或多 `instance`；显存吃紧 → 减实例/降精度/上 TensorRT 优化。

## 常见问题 / 坑（表格）

| 现象 | 根因 | 处理 |
|---|---|---|
| 模型加载失败/找不到 | 目录无数字版本子目录（缺 `1/`）；权重文件名不对 | 补 `1/`，确认 backend 约定的文件名 |
| 推理报输入名/维度错误 | `config.pbtxt` 的 `name`/`dims`/`data_type` 和真实 IO 对不上 | 用 ONNX/torch 工具核对真实 IO，逐一对齐；可变维用 `-1` |
| 吞吐上不去、GPU 没吃满 | 没开动态批处理或攒批窗口太小、实例太少 | 开 `dynamic_batching`、加大窗口、增 `instance_group.count` |
| 延迟突然变高 | 攒批窗口设太大，请求白等 | 在线场景调小 `max_queue_delay` |
| 显存 OOM | 实例副本×大模型×大 batch 叠加 | 减实例、降精度(FP16/INT8)、限制 `max_batch_size` |
| 把它当成写 kernel 的 Triton | 名字撞车（OpenAI Triton 语言） | 认清这是**Inference Server**，两者无关 |
| LLM 服务吞吐差 | 用了普通 backend，没用持续批处理 | 换 TensorRT-LLM / vLLM backend，启用 in-flight batching |
| 多框架想统一上线 | 误以为每框架要单独写服务 | 一个 Triton 同时托管多 backend，统一协议 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-inference/README]]
- [[llm-inference/tensorrt/README]]
- [[llm-inference/tensorrt-llm/README]]
- [[llm-inference/vllm/README]]
- [[llmops/kubernetes]]
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

> 参考（官方权威，精确参数/版本以此为准）：
> - 模型仓库与 backend 源码：https://github.com/triton-inference-server/server ，https://github.com/triton-inference-server/backend
> - 官方容器（含 SDK）：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
