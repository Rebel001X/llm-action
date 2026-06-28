# Triton 部署 ONNX 模型

> 用 NVIDIA Triton Inference Server 把一个 ONNX 模型变成可在线访问的高性能推理服务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] [[llm-inference/tensorrt/README]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] [[llmops/kubernetes]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | 这件事的本质 | Triton / ONNX / 模型仓库 |
| 1. 地基 | ONNX 和 Triton 各解决什么问题 | IR / 服务化 |
| 2. 整体架构 | 请求从网线到 GPU 的完整路径 | HTTP/gRPC / backend |
| 3. 模型仓库结构 | 目录为什么必须长这样 | model_repository / 版本号 |
| 4. config.pbtxt | 每个配置项做什么用、怎么权衡 | input/output / instance_group |
| 5. ONNX Runtime backend | Triton 怎么真正跑 ONNX | EP / 加速器 |
| 6. 动态批处理 | 吞吐怎么白白翻几倍 | dynamic_batching |
| 7. 实操流程 | 拉模型→起服务→发请求 | docker / curl / perf |
| 常见坑 | 部署时最容易翻车的点 | 维度 / 数据类型 |

## 0. 一句话锚点

**Triton 是一个"模型托管框架"，ONNX 是一种"跨框架模型文件格式"。** 你把 `.onnx` 文件按规定目录摆好，再写一个 `config.pbtxt` 告诉 Triton 这个模型的输入输出长什么样，Triton 就会自动起一个支持 HTTP/gRPC 的高性能推理服务，并帮你做批处理、多实例、多模型并发、版本管理。

一句话：**ONNX 负责"模型能被通用引擎读懂"，Triton 负责"把它高吞吐、低延迟地服务出去"。**

## 1. 地基：两个东西分别解决什么问题

### 1.1 ONNX 解决"模型格式割裂"

你在 PyTorch 训练得到 `.pt`，同事在 TensorFlow 得到 `SavedModel`，部署端却只想要一个**统一、与训练框架解耦**的文件。ONNX（Open Neural Network Exchange）就是这个统一中间表示（IR）：

- 它用一张**计算图**描述模型：节点是算子（Conv、MatMul、Softmax…），边是张量。
- 它把权重一起打包进 `.onnx` 文件（Protobuf 序列化）。
- 任何支持 ONNX 的运行时（ONNX Runtime、TensorRT、Triton）都能加载它，无需原始训练框架。

> 类比：ONNX 像 PDF。无论你用 Word 还是 LaTeX 写的，导出成 PDF 后，任何阅读器都能打开。

### 1.2 Triton 解决"把模型变成服务"

光有 `.onnx` 文件还不能上线。生产环境还需要：网络协议、并发、批处理、多模型共存、GPU 调度、健康检查、指标监控……自己手写一套 Flask + 推理代码很难做到高性能。**Triton Inference Server** 就是 NVIDIA 开源的、把这些"服务化"脏活一次性解决的框架。它的卖点：

- 多后端（backend）：同一个 Triton 能同时跑 ONNX、TensorRT、PyTorch、Python、vLLM 等。
- 多模型 / 多版本并发，共享一块或多块 GPU。
- **动态批处理**、并发实例、流水线编排（ensemble / BLS）。
- 内置 HTTP、gRPC 接口和 Prometheus 指标。

## 2. 整体架构：一个请求的旅程

```
  客户端 (curl / Python SDK / Java ...)
        │  HTTP:8000  或  gRPC:8001
        ▼
 ┌─────────────────────────────────────────────┐
 │              Triton Inference Server         │
 │                                              │
 │  ┌────────────┐   ┌────────────────────────┐ │
 │  │  接入层     │   │      调度器 Scheduler    │ │
 │  │ HTTP/gRPC  │──▶│  - 动态批处理(组batch)   │ │
 │  │  指标:8002 │   │  - 队列/优先级           │ │
 │  └────────────┘   └───────────┬────────────┘ │
 │                               ▼               │
 │            ┌──────────────────────────────┐  │
 │            │  Backend (这里是 onnxruntime) │  │
 │            │  加载 model.onnx，绑定 EP      │  │
 │            └───────────────┬──────────────┘  │
 │                            ▼                   │
 │                  CUDA EP / TensorRT EP / CPU  │
 └────────────────────────────┼──────────────────┘
                              ▼
                      GPU (执行 ONNX 计算图)
```

要点：
- `8000` HTTP、`8001` gRPC、`8002` Prometheus 指标，这三个端口是 Triton 的约定（容器 `-p` 映射时常见，**确切端口以官方文档为准**）。
- **Scheduler（调度器）** 是吞吐的关键：它把多个独立请求拼成一个大 batch 再丢给 GPU。
- **Backend** 是真正干活的插件。ONNX 模型用的是 `onnxruntime` backend，它内部再选 **Execution Provider（EP）** 决定算子跑在哪（CUDA / TensorRT / CPU）。

## 3. 模型仓库（Model Repository）结构

Triton 启动时只认一个参数：`--model-repository=<目录>`。这个目录里的**子目录结构是强约定**，Triton 靠目录名和层级来发现模型、识别版本。

```
model_repository/                  ← 启动时指向这里
│
└── densenet_onnx/                 ← 1层=模型名(请求里用它寻址)
    │
    ├── config.pbtxt              ← 模型配置(输入/输出/后端/批处理)
    │
    └── 1/                        ← 2层=版本号(必须是纯数字)
        │
        └── model.onnx           ← 实际权重文件，ONNX backend 默认叫这个名
```

为什么必须长这样（每一层的"为什么"）：

| 层级 | 含义 | 为什么不能乱起名 |
| --- | --- | --- |
| `densenet_onnx/` | **模型名**，客户端请求 `/v2/models/densenet_onnx/infer` 时用它寻址 | 它就是服务的对外标识 |
| `1/` `2/` … | **版本号**，必须是正整数 | Triton 靠数字大小判断"最新版"，支持灰度/回滚 |
| `model.onnx` | ONNX backend 约定的默认权重文件名 | 不叫这个名就要在 config 里显式指定 |
| `config.pbtxt` | 模型元数据 | 没有它，部分情况 Triton 无法推断输入输出 |

> 多版本共存示例：同一个 `densenet_onnx/` 下放 `1/`、`2/`、`3/`，Triton 会按版本策略（latest / all / specific）决定加载哪些版本，便于 A/B 与回滚。

## 4. config.pbtxt：每个配置项做什么用

`config.pbtxt` 是一个 **Protobuf 文本格式**文件，描述这个模型"对外长什么样"。下面讲**含义和权衡**，确切字段拼写以官方文档为准。

```protobuf
name: "densenet_onnx"          # 必须与目录名一致
platform: "onnxruntime_onnx"   # 用哪个 backend 跑该模型
max_batch_size: 8              # 0=不批处理；>0=允许动态批

input [
  {
    name: "data_0"             # 必须与 ONNX 图里输入张量名一致
    data_type: TYPE_FP32       # 数据类型，须与模型一致
    dims: [ 3, 224, 224 ]      # 单样本维度，不含batch维
  }
]
output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

instance_group [
  { count: 2, kind: KIND_GPU } # 并发实例数 / 跑在GPU还是CPU
]

dynamic_batching { }           # 开启动态批处理
```

各项的**作用与权衡**：

| 配置项 | 做什么用 | 怎么权衡 |
| --- | --- | --- |
| `platform` / `backend` | 指定用哪个后端跑该模型 | ONNX 选 `onnxruntime`；若追极致延迟可转 TensorRT |
| `max_batch_size` | batch 维上限，>0 才允许批处理 | 越大吞吐越高但单请求延迟和显存占用上升 |
| `input/output.name` | 张量名，**必须**与 ONNX 图内名字一致 | 名字错→加载失败，是头号坑 |
| `dims` | 每个张量除 batch 外的形状；`-1` 表示动态维 | 定长更快；变长（如序列）用 `-1` |
| `data_type` | 张量数据类型 | 须与模型导出时一致，否则结果错乱 |
| `instance_group.count` | 同一 GPU 上并发模型实例数 | 多实例提升并发利用率，但抢显存 |
| `kind` | 实例跑 GPU 还是 CPU | KIND_GPU 性能高；无卡或轻模型可 KIND_CPU |
| `dynamic_batching` | 把零散请求自动攒成 batch | 见第 6 节，吞吐利器 |

> 小提示：对于很多 ONNX 模型，Triton 能从 `.onnx` 里**自动推断** input/output（autocomplete），此时 config 可以很精简甚至省略。但**显式写出更可控、更不容易踩维度坑**，生产建议显式写。

## 5. ONNX Runtime backend 与 Execution Provider

Triton 跑 ONNX 模型是通过 **onnxruntime backend** 实现的。ONNX Runtime 内部有一个"Execution Provider（EP）"分发机制：同一张计算图，可以让不同算子跑在不同硬件后端上。

```
        model.onnx 计算图
              │
       ┌──────┴───────┐  ONNX Runtime 按 EP 优先级分发算子
       ▼              ▼
 ┌──────────┐   ┌─────────────┐   ┌──────────┐
 │ CUDA EP  │   │ TensorRT EP │   │  CPU EP  │
 │ (cuDNN)  │   │ (子图编译)   │   │ (兜底)    │
 └──────────┘   └─────────────┘   └──────────┘
```

- **CUDA EP**：直接用 cuDNN/cuBLAS 在 GPU 上跑算子，通用、稳定。
- **TensorRT EP**：把能优化的子图交给 TensorRT 做算子融合、精度降级（FP16/INT8），通常更快，但首次会有编译/构建开销。
- **CPU EP**：不支持的算子的兜底，保证模型一定能跑完。

权衡：默认 CUDA EP 足够好用；想榨干性能可在 config 的后端参数里启用 TensorRT EP（具体参数名以官方文档为准），代价是启动慢、构建缓存管理更复杂。

## 6. 动态批处理（Dynamic Batching）

这是 Triton 提升吞吐**最重要**的机制，单独拎出来讲。

问题：客户端请求是零散到达的，一次只来 1 个样本。GPU 擅长大矩阵并行，单样本推理会**严重浪费算力**。

解决：Triton 在调度器里设一个短暂的"攒单窗口"，把同一模型在窗口内到达的多个请求**拼成一个大 batch**，一次喂给 GPU。

```
时间轴 →
req1 ─┐
req2 ─┤  攒单窗口(如最多等 100 微秒)
req3 ─┤        │
req4 ─┘        ▼
          组成 batch=4 ──▶ GPU 一次算完 ──▶ 拆开各自返回
```

权衡点（这些是要在 config 里调的旋钮，含义为主，默认值以文档为准）：

| 旋钮 | 作用 | 调大的后果 |
| --- | --- | --- |
| 最大队列等待时延 | 攒单窗口允许等多久 | 等越久 batch 越大→吞吐↑ 但延迟↑ |
| `preferred_batch_size` | 优先凑到的 batch 尺寸 | 对齐 GPU 高效尺寸，命中则最划算 |
| `max_batch_size` | batch 上限 | 太大→显存爆 / 尾延迟拉长 |

经验数值：在线服务常把等待窗口设到几十~几百微秒，吞吐能比不批处理提升数倍；离线批量推理可把窗口和 batch 设得很大。**具体数值需按你的模型和 SLA 压测得出。**

## 7. 实操流程（拉模型 → 起服务 → 发请求）

下面是与本目录示例一致的端到端流程。**镜像 tag 形如 `<YY.MM>-py3`，请用 NGC 上的实际可用版本，不要照抄具体数字。**

### 7.1 摆好模型仓库并下载权重

```bash
mkdir -p model_repository/densenet_onnx/1
wget -O model_repository/densenet_onnx/1/model.onnx \
     <densenet 的 onnx 下载地址>
# 然后在 densenet_onnx/ 下放好 config.pbtxt
```

### 7.2 启动 Triton 服务端

```bash
docker pull nvcr.io/nvidia/tritonserver:<版本>-py3

docker run --gpus all --rm \
  -p 8000:8000 \   # HTTP
  -p 8001:8001 \   # gRPC
  -p 8002:8002 \   # 指标
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<版本>-py3 \
  tritonserver --model-repository=/models
```

启动日志里每个模型会显示 `READY / UNAVAILABLE`。**先看这张表确认模型 READY**，再发请求。

### 7.3 用 SDK 容器发请求 / 压测

```bash
docker pull nvcr.io/nvidia/tritonserver:<版本>-py3-sdk

docker run -it --rm --net=host \
  -v ${PWD}:/workspace/ \
  nvcr.io/nvidia/tritonserver:<版本>-py3-sdk \
  bash

# 容器内：
pip install torchvision
wget -O img1.jpg "<一张测试图片地址>"
# 用客户端示例脚本对 densenet_onnx 发推理请求
```

### 7.4 健康检查与压测

- **就绪/存活探针**（适合 K8s）：`GET /v2/health/ready`、`GET /v2/health/live`。
- **模型元数据**：`GET /v2/models/densenet_onnx`，回显 input/output，验证 config 是否被正确解析。
- **性能压测**：SDK 容器自带 `perf_analyzer`（精确命令以官方文档为准），用它扫不同并发/batch，找到吞吐-延迟拐点。

## 8. 与同类方案的对比

| 方案 | 定位 | 何时选它 |
| --- | --- | --- |
| **Triton + ONNX** | 通用模型托管，多后端多模型 | 模型杂、要统一平台、CV/小模型/集成流水线 |
| **Triton + TensorRT** | 极致低延迟推理 | 固定模型、追求最低延迟、愿意接受构建成本 |
| **vLLM / SGLang** | 大语言模型专用 | LLM 文本生成，需 PagedAttention/连续批处理 → 见 [[llm-inference/vllm/README]] [[llm-inference/sglang/README]] |
| 自写 Flask 服务 | 极简原型 | 仅 demo，无吞吐/并发要求 |

> 重要边界：Triton + ONNX 对 **CV / 推荐 / 小型模型**非常合适。对 **LLM 自回归生成**，更主流的是 vLLM/TGI/TensorRT-LLM 或 Triton 的 vLLM/TensorRT-LLM backend，因为它们专门优化了 KV Cache 与连续批处理。

## 常见问题 / 坑

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 模型一直 UNAVAILABLE | 目录结构错（缺版本号子目录 / 文件名不对） | 严格按 `模型名/版本号/model.onnx` 摆放 |
| 加载报输入张量找不到 | config 里 `input.name` ≠ ONNX 图里的真实张量名 | 用 Netron 打开 onnx 看真实名字 |
| 推理结果全错 | `data_type` 或 `dims` 与模型不符 | 核对维度顺序(是否含 batch 维)与精度 |
| 变长输入报维度不匹配 | dims 写成定长 | 动态维用 `-1`，并设好 `max_batch_size` |
| 起服务报 no GPU / CUDA | 没加 `--gpus all` 或驱动不匹配 | 加 `--gpus all`，对齐驱动与镜像版本 |
| 吞吐上不去 | 没开动态批处理 / 实例数=1 | 开 `dynamic_batching`，调 `instance_group` |
| 客户端连不上 | 端口没映射 / 用错协议端口 | HTTP 走 8000、gRPC 走 8001（以文档为准） |
| 镜像 tag 拉不到 | 写了不存在的版本号 | 去 NGC 查实际可用 tag，别硬编版本 |

> 终极原则：**目录结构、张量名、数据类型、维度**这四样必须和 ONNX 文件严格对齐，绝大多数部署失败都出在这里。

## 🔗 跳转链接

- 📍 [[00-知识地图]]
- 上层：[[llm-inference/README]]
- 同族：[[llm-inference/tensorrt/README]]（Triton 常配 TensorRT 后端） · [[llm-inference/vllm/README]] · [[llm-inference/sglang/README]]
- 部署运维：[[llmops/kubernetes]]（Triton on K8s 探针/扩缩容）
- 指标解读：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
