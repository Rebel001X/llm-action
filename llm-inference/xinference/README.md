# Xinference

> Xinference（Xorbits Inference）是一个**面向生产的分布式推理框架**：用一条命令把任意开源大模型（LLM / Embedding / Rerank / 图像 / 语音 / 多模态）拉起成一个**带 OpenAI 兼容 API 的集群服务**，底层可自由切换 vLLM / SGLang / llama.cpp / Transformers 等推理引擎。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/README]] [[llm-maas/README]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | 模型即服务 / 多后端 / 集群 |
| 1 | 它解决什么问题（地基） | 部署碎片化、引擎割裂、多模型编排 |
| 2 | 整体架构（Supervisor / Worker） | 控制面 / 数据面 / Actor |
| 3 | 模型注册与分发机制 | builtin / custom / registry |
| 4 | 多后端引擎抽象 | vLLM / SGLang / llama.cpp / Transformers |
| 5 | 一次请求的生命周期 | 调用链 / Replica / 路由 |
| 6 | 集群与分布式部署 | 多机多卡 / GPU 调度 |
| 7 | OpenAI 兼容层 | `/v1/chat/completions` / SDK 直连 |
| 8 | 关键参数与权衡 | replica / n_gpu / quantization |
| 9 | 与同类对比 + 实践要点 | vLLM/Ollama/Ray Serve/TGI |
| — | 配置示例 / 常见问题 / 跳转 | CLI / Python SDK |

---

## 0. 一句话锚点

> **Xinference = 「模型即服务」的编排层。** 它本身**不发明新的推理 kernel**，而是把"下载权重 → 选引擎 → 占显存 → 起 worker → 暴露 OpenAI API → 负载均衡"这一整条链路标准化，让你 `xinference launch --model-name qwen2.5-instruct` 一句话就得到一个可横向扩展的推理服务。

记住三个词：**多模型（model zoo）、多后端（engine abstraction）、多机器（cluster）**。这正是它区别于"单模型单引擎"工具（如裸 vLLM）的核心。

---

## 1. 地基：它到底解决什么问题

要理解 Xinference，先看不用它时，把一个开源模型上线到底有多碎：

```
痛点全景（裸手工部署）
┌─────────────────────────────────────────────────────────┐
│ 1. 权重从哪来？  HuggingFace? ModelScope? 本地路径? 量化版? │
│ 2. 用哪个引擎？  vLLM 吞吐高但吃显存; llama.cpp 能跑 CPU;   │
│                  Transformers 兼容性好但慢; 各自 API 不同  │
│ 3. 怎么对外？    自己写 FastAPI 包一层 OpenAI 协议?         │
│ 4. 多模型？      一台机器同时跑 chat+embedding+rerank?      │
│ 5. 多卡多机？    手动分配 GPU、起多进程、做反向代理负载均衡? │
│ 6. 多模态？      图像/语音/视频模型又是另一套接口?          │
└─────────────────────────────────────────────────────────┘
```

每一行都是一个"自己造轮子"的坑。**Xinference 把这 6 件事统一成一个控制面 + 一套 API。** 它的设计目标可以浓缩为：

- **统一抽象**：无论后端是 vLLM 还是 llama.cpp，对外都是同一个 OpenAI 风格接口。
- **一键拉起**：模型注册表里有的，给个名字就能 launch；没有的，注册一下也能 launch。
- **分布式原生**：从单机单卡到多机多卡，部署模型同一套命令，调度交给框架。
- **全模态**：LLM、Embedding、Rerank、Image、Audio（语音转写/合成）、Video 统一管理。

> 一句话：**vLLM 解决"一个模型怎么跑得快"，Xinference 解决"一堆模型怎么跑得稳、管得动、调得开"。** 二者是互补而非替代——Xinference 经常把 vLLM 当作自己的一个后端。

---

## 2. 整体架构：Supervisor / Worker 双层

Xinference 是典型的**控制面 + 数据面**分离架构，底层基于 **Xoscar Actor 模型**（一个轻量分布式 actor 框架，源自 Xorbits 项目，类似 Ray 的 actor 思想）做进程/节点间通信。

```
                        Xinference 集群拓扑
   ┌──────────────────────────────────────────────────────────┐
   │  Client (CLI / Python SDK / OpenAI SDK / WebUI / REST)     │
   └───────────────────────────┬──────────────────────────────┘
                               │ HTTP (RESTful API, 默认端口 9997)
                       ┌───────▼────────┐
                       │   Supervisor   │  ← 控制面（大脑）
                       │  (调度/路由/    │
                       │   元数据/状态)  │
                       └───┬────────┬───┘
              Actor RPC    │        │   Actor RPC
                ┌──────────▼──┐  ┌──▼───────────┐
                │  Worker A   │  │  Worker B    │  ← 数据面（干活）
                │  (GPU 节点) │  │  (GPU 节点)  │
                │ ┌─────────┐ │  │ ┌──────────┐ │
                │ │ModelActor│ │  │ │ModelActor│ │  ← 每个模型副本=一个Actor
                │ │ vLLM    │ │  │ │ llama.cpp│ │
                │ └─────────┘ │  │ └──────────┘ │
                └─────────────┘  └──────────────┘
```

各角色职责：

| 组件 | 角色 | 职责 |
|------|------|------|
| **Supervisor** | 控制面 / 协调者 | 维护集群全局状态：有哪些 worker、各 worker 的 GPU/内存余量、当前已加载哪些模型及其副本（replica）分布；接收 launch/terminate 指令并**调度**到合适的 worker；做请求**路由**（把一次推理转发到某个模型副本）。一个集群**只有一个** Supervisor。 |
| **Worker** | 数据面 / 执行者 | 注册到 Supervisor，上报本节点资源；在 Supervisor 指令下**真正加载模型权重、占用 GPU**、起推理引擎；为每个模型实例托管一个 **ModelActor**。 |
| **ModelActor** | 模型副本载体 | 一个被加载的模型实例（含其后端引擎），对外暴露 `generate / chat / create_embedding` 等方法；多个副本可分散在不同 worker 上做横向扩展。 |
| **API 层** | 接入面 | RESTful Server，把 HTTP 请求翻译成对 Supervisor / ModelActor 的 actor 调用；并提供 **OpenAI 兼容**端点。 |

> **为什么要 Actor 模型？** 因为推理服务天然是"有状态、长生命周期、跨进程跨机器"的：一个模型副本占着几十 GB 显存常驻，请求要精确路由到它。Actor（一个对象=一个独立邮箱+一个执行单元）正好把"一个模型副本"封装成一个可寻址、可远程调用、可独立崩溃重启的实体，比裸 socket 编排清晰得多。

**本地模式（local）**：单机起一个进程，Supervisor 和 Worker 合并在一起，`xinference-local` 即可——开发调试最常用。
**分布式模式（distributed）**：`xinference-supervisor` 起在一台机器，`xinference-worker` 起在 N 台机器并指向 supervisor 地址。

---

## 3. 模型注册与分发机制（Model Registry）

这是 Xinference 的灵魂之一。它要回答："你说 `qwen2.5-instruct`，框架怎么知道去哪下、有哪些尺寸、用什么 prompt 模板？"

### 3.1 三类模型来源

```
模型从哪来
┌───────────────────────────────────────────────────────────┐
│ ① Builtin（内置注册表）                                     │
│    框架自带的模型规格清单(JSON)，描述每个模型的:            │
│    name / 支持的 size(7B/14B...) / format(pytorch/gguf/awq) │
│    / quantization(int4/int8...) / prompt 模板 / 引擎能力    │
│    launch 时按 (name,size,format,quant) 定位具体权重去下载   │
│                                                            │
│ ② Custom（自定义注册）                                      │
│    register_model: 你把自己微调/私有的模型按同样的 schema   │
│    描述一份，告诉框架权重路径 + prompt 模板，即可像内置一样  │
│    launch。注册信息持久化在本地。                            │
│                                                            │
│ ③ 本地路径直挂                                              │
│    launch 时指定 model_path，跳过下载，直接加载本地权重      │
└───────────────────────────────────────────────────────────┘
```

### 3.2 模型规格（model spec）的本质

每个模型在注册表里是一条**结构化元数据**，核心字段（以官方文档/源码为准，此处讲含义）：

| 字段 | 含义 | 为什么重要 |
|------|------|-----------|
| `model_name` | 模型族名，如 `qwen2.5-instruct` | 用户面唯一标识 |
| `model_size_in_billions` | 参数量档位，如 7 / 14 / 72 | 同名模型多尺寸 |
| `model_format` | 权重格式：`pytorch` / `gguf` / `awq` / `gptq` / `mlx` | 决定能用哪个引擎 |
| `quantization` | 量化：`none` / `int4` / `int8` / `Q4_K_M` ... | 决定显存占用与精度 |
| `model_engine` | 后端引擎选择 | 见第 4 节 |
| `prompt_style` / chat template | 对话拼接模板 | 决定 chat 输出是否正确 |

> **关键洞察**：`(model_format, quantization)` 共同决定了**可用引擎集合**。比如 `gguf` 格式天然走 **llama.cpp**；`awq/gptq` 量化 + `pytorch` 通常走 **vLLM**；什么特殊格式都没有的 fp16 pytorch 可走 vLLM 或 Transformers。这就是"格式→引擎"的隐式约束。

### 3.3 下载源切换

国内最实用的一点：通过环境变量 `XINFERENCE_MODEL_SRC=modelscope` 把默认下载源从 HuggingFace 切到 **ModelScope（魔搭）**，避免境外网络问题。这是 Xinference 在中文社区广受欢迎的原因之一。

---

## 4. 多后端引擎抽象（Engine Abstraction）

Xinference **不自己写推理 kernel**，而是定义一层统一的 `Model` 接口，把各家引擎适配进来。

```
                  统一接口 Model
   generate() / chat() / create_embedding() / ...
                       │
        ┌──────────┬───┴────┬───────────┬──────────┐
        ▼          ▼        ▼           ▼          ▼
   ┌────────┐ ┌────────┐┌──────────┐┌──────────┐┌──────┐
   │  vLLM  │ │ SGLang ││llama.cpp ││Transform.││ MLX  │
   │PagedAttn│ │RadixAttn││ GGUF/CPU ││ baseline ││Apple │
   │ 高吞吐  │ │ 强缓存  ││ 可CPU/量化││ 全兼容   ││ M芯片│
   └────────┘ └────────┘└──────────┘└──────────┘└──────┘
```

| 后端 | 适用场景 | 优点 | 代价 |
|------|---------|------|------|
| **vLLM** | 生产高并发 GPU 服务 | PagedAttention + continuous batching，吞吐极高 | 吃显存、对模型/格式有支持范围 |
| **SGLang** | 高并发 + 复杂结构化输出 | RadixAttention 前缀缓存，多轮/共享前缀快 | 生态较新 |
| **llama.cpp** | 边缘 / CPU / 低显存 / GGUF | 可纯 CPU 或少量显存跑量化模型，门槛低 | 吞吐不及 vLLM |
| **Transformers** | 兼容性兜底 / 新模型首发 | HuggingFace 原生，几乎什么模型都能跑 | 慢、无高级 batching |
| **MLX** | Apple Silicon（Mac） | 针对 M 系芯片优化 | 仅 Mac |

> **抽象的价值**：你写的客户端代码（OpenAI SDK 调用）**完全不变**，只是 launch 时把 `--model-engine` 从 `vllm` 改成 `llama.cpp`，就能从"A100 高吞吐"无缝切到"笔记本 CPU 跑量化版"。**引擎是部署细节，对调用方透明**——这正是"框架"相对"裸引擎"的核心增益。

引擎选择决策树：

```
要部署一个 LLM，选哪个引擎？
        │
  有 GPU 且显存够? ──否──▶ 模型有 GGUF 量化版? ──是──▶ llama.cpp(CPU/小显存)
        │是                        │否
        ▼                          ▼
  追求最高吞吐? ──是──▶ vLLM      Transformers(兜底,慢)
        │否
        ▼
  模型太新 vLLM 还不支持? ──是──▶ Transformers
        │否
        ▼
       vLLM / SGLang
```

---

## 5. 一次请求的生命周期（调用链）

以一次 `POST /v1/chat/completions` 为例，把"用户敲下回车"到"流式吐字"全链路拆开：

```
请求生命周期（Chat 流式为例）
 Client                API Server         Supervisor          Worker/ModelActor
   │                       │                   │                     │
   │ 1.POST /v1/chat/...   │                   │                     │
   │──────────────────────▶│                   │                     │
   │                       │ 2.校验+解析为内部  │                     │
   │                       │   推理请求         │                     │
   │                       │ 3.查"哪个模型副本   │                     │
   │                       │   能服务它?"────────▶│                     │
   │                       │                   │ 4.路由:在已加载的    │
   │                       │                   │   replica 中选一个    │
   │                       │                   │  (负载/轮询)          │
   │                       │◀─返回目标 Actor 句柄─│                     │
   │                       │ 5.actor 调用 chat() ──────────────────────▶│
   │                       │                   │                     │ 6.后端引擎推理
   │                       │                   │                     │  (vLLM 批处理/
   │                       │                   │                     │   KV cache/采样)
   │                       │◀──────流式 token ──────────────────────────│
   │◀═══SSE data: chunks═══│  (逐块转 OpenAI 格式)                      │
   │                       │                   │                     │
   │◀── data:[DONE] ───────│                   │                     │
```

关键点：

1. **路由发生在 Supervisor**：它知道全局有哪些模型、各有几个副本、负载如何，选出一个 ModelActor。
2. **副本（replica）是横向扩展单位**：launch 时 `--replica 3` 会在集群里起 3 份同模型，请求被分摊，吞吐近似线性提升（受限于 GPU 总数）。
3. **批处理在引擎内部**：多个并发请求到了同一个 vLLM 副本，由 vLLM 的 continuous batching 合并，Xinference 不重复造这层。
4. **流式靠 SSE**：API 层把引擎吐出的 token 实时包装成 OpenAI 的 `data: {...}` chunk 回传。

---

## 6. 集群与分布式部署

### 6.1 起集群的三步

```
分布式部署拓扑
┌─────────────────────────────────────────────────────────┐
│ 机器0(控制+可选GPU):                                       │
│   $ xinference-supervisor -H 0.0.0.0 -p 9997 -P 9999      │
│     (-p REST端口, -P supervisor 内部端口)                  │
│                                                          │
│ 机器1(GPU):                                               │
│   $ xinference-worker -e http://机器0:9997 -H 机器1IP     │
│                                                          │
│ 机器2(GPU):                                               │
│   $ xinference-worker -e http://机器0:9997 -H 机器2IP     │
│                                                          │
│ 之后所有 launch 都打到机器0:9997, 框架自动选 worker 放模型 │
└─────────────────────────────────────────────────────────┘
```

> 上面的命令选项以官方文档/`xinference --help` 为准，此处展示**语义结构**：worker 通过 `-e/--endpoint` 指向 supervisor 完成注册。

### 6.2 GPU 调度与放置

- launch 时可指定 `--n-gpu`（这个模型副本要几张卡，用于**张量并行**大模型）或交给框架自动选。
- Supervisor 维护每个 worker 的 GPU 余量，按"哪台塞得下"放置副本；放不下则报资源不足。
- 一台机器可**同时跑多个不同模型**（chat + embedding + rerank），只要显存够——这是 Xinference 做"多模型编排"的直接体现。

```
单机多模型共存（一台 8×GPU 机器示例）
 GPU0-1: qwen2.5-72B (张量并行, n_gpu=2, vLLM)
 GPU2  : bge-large embedding
 GPU3  : bge-reranker
 GPU4  : qwen2.5-7B (replica#1)
 GPU5  : qwen2.5-7B (replica#2)   ← 同模型2副本做负载均衡
 GPU6-7: 空闲, 可继续 launch
```

---

## 7. OpenAI 兼容层

这是 Xinference 落地的"最后一公里"。它把内部接口映射成业界事实标准的 OpenAI API：

| Xinference 模型类型 | 暴露的 OpenAI 兼容端点 |
|--------------------|----------------------|
| LLM (chat) | `POST /v1/chat/completions` |
| LLM (base) | `POST /v1/completions` |
| Embedding | `POST /v1/embeddings` |
| 列表 | `GET /v1/models` |
| 图像 | `POST /v1/images/generations` |
| 语音 | `POST /v1/audio/transcriptions` / `/speech` |

> Rerank、注册管理等 Xinference 特有能力则走它自己的 RESTful 端点（OpenAI 协议里没有 rerank）。

**直接用 OpenAI 官方 SDK 即可**（把 base_url 指过来）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9997/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="qwen2.5-instruct",          # 即 launch 时给的 model_uid
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

> **价值**：任何已经写好的、对接 OpenAI 的应用（LangChain、Dify、各类 Agent 框架）几乎**零改动**就能切到本地 Xinference——只改 `base_url`。这也是它常被当作**私有化 MaaS 网关**底座的原因（见 [[llm-maas/README]]）。

---

## 8. 关键参数与权衡

launch 一个模型时最常调的旋钮（名称以官方为准，此处讲权衡）：

| 参数 | 含义 | 调大/调小的权衡 |
|------|------|----------------|
| `model_engine` | 选后端引擎 | vLLM 吞吐高吃显存；llama.cpp 省资源吞吐低（见第4节） |
| `model_format` | 权重格式 | gguf→llama.cpp；awq/gptq→vLLM 省显存但有精度损失 |
| `quantization` | 量化等级 | int4 显存 ≈ fp16 的 1/4，但可能掉点；精度敏感任务慎用 |
| `replica` | 副本数 | 增大提升并发吞吐，但占用成倍 GPU |
| `n_gpu` | 单副本占卡数 | 大模型需张量并行（>1），小模型设 1 即可，设多了浪费 |
| `gpu_idx` | 指定显卡 | 手动放置，避开已占用卡 |
| `max_model_len` | 上下文长度上限 | 越长 KV cache 越吃显存，需与显存平衡 |
| `download_hub` | 下载源 | `modelscope` 国内更快，`huggingface` 更全 |

**一个数值直觉（量化省显存）**：一个 7B 模型，权重约 $7\text{B} \times 2\text{ bytes} = 14\text{ GB}$（fp16）；换 int4 后约 $7\text{B} \times 0.5\text{ bytes} \approx 3.5\text{ GB}$，再加上 KV cache 和激活开销，一张 8GB 消费级显卡就能跑——这就是为什么边缘部署偏爱 gguf+int4。

KV cache 显存（粗略估算）：

$$ \text{KV}_{\text{bytes}} \approx 2 \times n_{\text{layer}} \times n_{\text{ctx}} \times d_{\text{model}} \times \text{bytes}_{\text{dtype}} \times \text{batch} $$

> 因子 2 来自 Key 和 Value 各一份。这解释了为何 `max_model_len`（即 $n_{\text{ctx}}$）和并发 batch 一起决定了显存压力——长上下文 + 高并发会让 KV cache 迅速膨胀。

---

## 9. 与同类对比 + 实践要点

### 9.1 横向对比

| 维度 | **Xinference** | 裸 **vLLM** | **Ollama** | **Ray Serve** | **TGI** |
|------|---------------|------------|-----------|--------------|--------|
| 定位 | 多模型分布式编排框架 | 单一高性能引擎 | 本地易用模型运行器 | 通用模型服务编排 | HF 官方推理服务 |
| 多后端 | ✅ vLLM/sglang/llama.cpp/transformers | 仅自身 | 仅 llama.cpp 系 | 需自接 | 仅自身 |
| 多模态 | ✅ LLM/Embed/Rerank/图/音/视频 | 偏 LLM | 偏 LLM | 自定义 | 偏 LLM |
| 集群分布式 | ✅ 原生 Supervisor/Worker | 单实例（需外部编排） | 单机为主 | ✅ 通用但要写代码 | 需外部编排 |
| OpenAI 兼容 | ✅ | ✅ | 部分 | 需自实现 | ✅(messages) |
| 上手成本 | 低（一条命令） | 中 | 极低 | 高（写部署代码） | 中 |
| 国内友好(ModelScope) | ✅ | ⚠️ | ⚠️ | — | ⚠️ |

> **一句话区分**：
> - **Ollama**：个人/本地玩家，简单但不为集群/多模态/高并发设计。
> - **裸 vLLM**：要极致单模型吞吐、自己有编排能力时直接用。
> - **Ray Serve**：通用、强大但偏底层，要写不少胶水代码。
> - **Xinference**：想要"开箱即用的多模型私有化推理平台"，且要 vLLM 的性能又不想自己拼装——它是**编排层 + 引擎复用 + OpenAI 网关**的合集。

### 9.2 实践要点（踩坑清单）

- **显存预估先行**：launch 前用 `参数量×字节数 + KV cache` 粗估，避免 OOM；不够就降 quantization 或换 llama.cpp。
- **引擎要对上格式**：gguf 配 llama.cpp、awq/gptq 配 vLLM，错配会直接 launch 失败。
- **国内务必设 `XINFERENCE_MODEL_SRC=modelscope`**，否则卡在 HuggingFace 下载。
- **多副本做 HA + 吞吐**：生产环境同一模型起 ≥2 replica，单副本崩了还有其他顶上。
- **embedding/rerank 与 chat 分开 launch**：它们是独立模型实例，RAG 场景三件套（chat+embedding+rerank）通常都要起。
- **model_uid 是调用主键**：launch 返回的 uid（或你指定的）就是 OpenAI 调用里的 `model` 字段。
- **WebUI 排障**：Xinference 自带 Web 界面（默认 9997），可视化看每个模型副本在哪台 worker、占多少显存，定位问题快。
- **生产建议固定引擎版本**：vLLM/llama.cpp 升级可能改变行为，锁版本保证可复现。

---

## 配置示例（讲含义）

CLI 一键拉起一个 vLLM 后端的 chat 模型：

```bash
# 1) 启动本地服务（开发）
xinference-local -H 0.0.0.0 --port 9997

# 2) launch 一个模型（语义：名字+尺寸+格式+引擎+副本+占卡）
xinference launch \
  --model-name qwen2.5-instruct \
  --model-engine vllm \
  --size-in-billions 7 \
  --model-format pytorch \
  --replica 2 \
  --n-gpu 1
# 返回一个 model_uid, 比如 "qwen2.5-instruct"
```

每个选项的含义：

- `--model-name`：去注册表查规格、定位权重。
- `--model-engine vllm`：用 vLLM 跑，要 GPU、吞吐高。
- `--size-in-billions 7` / `--model-format pytorch`：选 7B 的 fp16 pytorch 档。
- `--replica 2`：起 2 个副本横向扩展（占 2×显存）。
- `--n-gpu 1`：每副本占 1 张卡（小模型无需张量并行）。

Python SDK 等价操作：

```python
from xinference.client import Client
client = Client("http://127.0.0.1:9997")
uid = client.launch_model(
    model_name="qwen2.5-instruct",
    model_engine="vllm",
    model_size_in_billions=7,
    model_format="pytorch",
    replica=2,
)
model = client.get_model(uid)
print(model.chat(messages=[{"role": "user", "content": "解释一下注意力机制"}]))
```

> 以上参数名以官方文档/源码为准；核心是理解"**名字定位模型 + 引擎决定怎么跑 + replica/n_gpu 决定占多少资源**"这套心智模型。

---

## 常见问题

| 问题 | 解答 |
|------|------|
| Xinference 和 vLLM 是竞争关系吗？ | 不是。Xinference 是**编排框架**，vLLM 是它的一个**后端引擎**，二者常配合使用。 |
| 它自己实现了推理优化吗？ | 没有重造 kernel；性能来自底层引擎（vLLM 的 PagedAttention 等）。它的价值在编排、抽象、集群、API 统一。 |
| 能跑非 LLM 模型吗？ | 能。Embedding、Rerank、图像生成、语音转写/合成、多模态都支持，统一管理。 |
| 国内下载慢怎么办？ | 设 `XINFERENCE_MODEL_SRC=modelscope`，从魔搭下载。 |
| 单机能用吗？ | 能。`xinference-local` 单进程模式，Supervisor 和 Worker 合一，开发首选。 |
| 怎么横向扩展吞吐？ | 同模型多 `replica`，分布到多 worker/多 GPU，Supervisor 自动路由分摊。 |
| 大模型一张卡放不下？ | 设 `--n-gpu N` 做张量并行，跨多卡加载一个副本。 |
| 现有 OpenAI 应用怎么接？ | 只改 `base_url` 指向 `http://host:9997/v1`，SDK 不动。 |
| 怎么用自己的微调模型？ | `register_model` 注册自定义 spec（权重路径 + prompt 模板），之后像内置模型一样 launch。 |
| 怎么排查模型在哪台机器？ | 用自带 WebUI 或 `xinference list`，可视化查看副本分布与资源占用。 |

---

## 🔗 跳转链接

- 上层全景：[[00-知识地图]]
- 推理总览（本框架在推理体系中的位置）：[[llm-inference/README]]
- 模型即服务 / 私有化网关场景：[[llm-maas/README]]
- 官方文档（最权威，CLI 选项/参数/支持模型以此为准）：https://inference.readthedocs.io/zh-cn/latest/index.html

> 学习建议：先在单机 `xinference-local` 上 launch 一个 7B 模型跑通 OpenAI 调用，再尝试换引擎（vllm↔llama.cpp）感受抽象层，最后起 supervisor+worker 体会分布式调度。理解了"控制面/数据面 + 模型注册 + 引擎抽象"这三块，Xinference 的全部行为就都能推导出来了。
