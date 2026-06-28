# TensorRT-LLM Triton 服务启动参数

> 用 Triton Inference Server + TensorRT-LLM backend 把已构建好的引擎拉起成在线推理服务，本文讲清楚“启动一个 trtllm 服务到底要配哪些参数、每个参数解决什么问题、怎么权衡”。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/tensorrt-llm/README]] [[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]] [[llm-optimizer/kv-cache]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 小节 | 讲什么 | 你会带走什么 |
| :-- | :-- | :-- |
| 0. 一句话锚点 | 服务启动是引擎之后的最后一公里 | 知道本文在整条链路里的位置 |
| 1. 地基 | Triton + trtllm backend 解决什么问题 | 为什么不是直接 `python run.py` |
| 2. 整体架构 | Server / model repository / 5 个子模型 | 看懂 BLS ensemble 调用链 |
| 3. 服务进程级参数 | `tritonserver`/`launch` 启动开关 | 端口、并行、日志、metrics |
| 4. 模型仓库配置 | `config.pbtxt` 关键项 | batching、调度、KV、并发的旋钮 |
| 5. In-flight Batching | 连续批处理调度器参数 | 吞吐/显存/延迟三角的权衡 |
| 6. KV Cache 参数 | 显存占用的最大变量 | free_fraction / reuse / host offload |
| 7. 多 GPU 拉起 | TP/PP × world_size × MPI | 多卡多机的 launcher 心智模型 |
| 典型流程 | 从引擎到 curl 的完整步骤 | 可照着做的部署清单 |
| 常见坑 | 表格速查 | 少踩 80% 的雷 |

## 0. 一句话锚点

TensorRT-LLM 的工作分两步：**离线编译**（`trtllm-build` 把模型变成 `.engine`，见 [[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]）→ **在线服务**（本文：用 Triton 把引擎托管成 HTTP/gRPC 服务）。本文几乎不涉及精度/算子，只涉及“**怎么把引擎稳定、高吞吐地对外提供**”。

> ⚠️ 护栏声明：下文出现的参数名、默认值会随 TensorRT-LLM / Triton 版本变化。**精确的字段名与默认值以你所用版本的 `tensorrtllm_backend` 仓库 `docs/model_config.md` 与官方文档为准**，这里讲的是“每个旋钮是干什么的、往哪拧”。

## 1. 地基：Triton + trtllm backend 解决什么问题

光有一个 `.engine` 文件，你只能在自己进程里 `executor.enqueue()` 跑请求。生产环境还缺一大堆东西：

- **协议层**：对外要 HTTP/REST 和 gRPC，要能并发接很多客户端。
- **批处理**：多个用户的请求要攒成 batch 喂 GPU 才划算，但 LLM 的请求长度不一、生成步数不一，普通的“静态批”会被最长的那条拖死。
- **预处理/后处理**：文本 → token id（tokenize）和 token id → 文本（detokenize）要有人做。
- **多模型编排**：tokenize → 推理 → detokenize 要串成一条流水线。
- **可观测性**：QPS、延迟、GPU 利用率、KV 占用等 metrics。

Triton Inference Server 提供前四项的通用框架，**TensorRT-LLM backend** 是其中专门驱动 trtllm 引擎、并实现 **In-flight Batching（连续批处理）** 的插件。两者合起来就是 trtllm 的官方在线服务方案。

```
没有 Triton：               有 Triton + trtllm backend：

 client                      client ──HTTP/gRPC──┐
   │ 自己写 socket                                │
   ▼                          ┌──────────────────▼─────────────────┐
 你的 python                   │  Triton Server (C++ 进程)          │
   │ 自己 tokenize             │  ├ 协议/调度/metrics/健康检查      │
   ▼                          │  └ tensorrtllm backend             │
 executor.enqueue             │      └ In-flight Batching 调度器     │
   │ 自己 detokenize           └──────────────────┬─────────────────┘
   ▼                                              ▼
 你自己拼 batch（很难做好）              .engine（GPU 上常驻）
```

## 2. 整体架构：一个 trtllm 服务由 5 个“子模型”组成

Triton 的部署单位是 **model repository（模型仓库）**——一个目录，每个子目录是一个“模型”，含 `config.pbtxt` + 权重/脚本。trtllm backend 的官方模板把一次推理拆成 5 个模型，用 **ensemble** 或 **BLS（Business Logic Scripting）** 串起来：

```
                ┌────────────────── ensemble / BLS ─────────────────┐
 请求(prompt) ─▶│ preprocessing ─▶ tensorrt_llm ─▶ postprocessing   │─▶ 文本
                └───────────────────────────────────────────────────┘
                       │tokenize        │真正的引擎      │detokenize
                       │(python)        │(trtllm)        │(python)

 对外入口二选一：
   ensemble        —— 静态编排，config 里把三段连起来（无控制流）
   tensorrt_llm_bls—— python 写控制流（如 n-best、early-stop、guided decode）
```

5 个目录通常是：
- `preprocessing`：python backend，HF tokenizer 做 tokenize。
- `tensorrt_llm`：**核心**，加载 `.engine`，做 In-flight Batching，本文 4~6 节的参数都在它的 `config.pbtxt`。
- `postprocessing`：python backend，detokenize。
- `ensemble`：把上面三个静态串联（DAG）。
- `tensorrt_llm_bls`：用 python 脚本编排，灵活但多一层开销。

> 关键认知：**“启动参数”有两个层面**——(A) 启动 *进程* 的命令行开关（第 3 节）；(B) 写在 `config.pbtxt` 里、加载时读取的 *模型级* 参数（第 4~6 节）。新手常把两者混为一谈。

## 3. 服务进程级参数（命令行）

trtllm backend 官方提供一个 `launch_triton_server.py`（或直接 `tritonserver`）来拉起进程。**进程级**常见开关（按作用归类，名字以你的版本为准）：

| 类别 | 旋钮（语义） | 作用与权衡 |
| :-- | :-- | :-- |
| 模型仓库 | model repository 路径 | 指向上面那个含 5 个子目录的目录 |
| 协议端口 | http / grpc / metrics 端口 | 默认 8000/8001/8002；与已占端口冲突要改 |
| 多 GPU | world_size / 进程数 | 等于引擎构建时的 `TP×PP`，决定起几个 MPI rank（见第 7 节） |
| 启动器 | 用 `mpirun` 还是单进程 | 多卡时通过 MPI 拉起多个 worker |
| 日志 | log verbose 等级 | 调试时打开，看调度/KV 决策 |
| metrics | 是否暴露 Prometheus | 生产强烈建议开，接监控 |
| 健康/就绪 | readiness / liveness | K8s 探针靠它判断 Pod 是否可接流量（见 [[llmops/kubernetes]]） |
| 仓库模式 | 显式加载指定模型 | 只 load `ensemble`/`bls` 而非全部，避免加载无用模型 |

进程级参数解决的是“**进程怎么起、对谁开放、起几份**”，不决定推理行为本身。

## 4. 模型仓库配置：`tensorrt_llm/config.pbtxt` 关键项

这是真正影响推理行为的地方。`config.pbtxt` 用 protobuf 文本语法，trtllm backend 通过 `parameters { key: "X" value: {...} }` 读取一批自定义参数。按职责分组：

### 4.1 引擎与后端类型
- **引擎目录**：指向 `trtllm-build` 产出的 `.engine` + `config.json` 所在路径。**必填**，错了直接起不来。
- **后端/执行模式**：选 trtllm 的 executor / decoupled 模式。**decoupled（解耦）= 流式**：一个请求可分多次返回 token（SSE/gRPC stream），做 chat 必开；非 decoupled 则一次性返回。

### 4.2 batching 与调度（重头戏，见第 5 节）
- **batching 策略**：选 `inflight_fused_batching`（连续批处理）还是静态。生产几乎都用前者。
- **max batch size**：调度器一个 step 内最多并行多少个请求。越大吞吐越高，但显存（KV）和单步延迟也涨。
- **调度策略**：`max_utilization`（尽量塞满，吞吐优先，可能因显存不足而抢占/换出请求）vs `guaranteed_no_evict`（保证不驱逐，延迟稳定但吞吐略低）。这是**吞吐 vs 稳定性**的总开关。

### 4.3 KV cache（见第 6 节）
- `kv_cache_free_gpu_mem_fraction`、`enable_kv_cache_reuse`、host KV offload 等。

### 4.4 并发实例
- **instance group / count**：同一个引擎要不要起多份实例。注意：trtllm 引擎本身已通过 In-flight Batching 内部并发，**通常 count=1 即可**；盲目加实例会成倍吃显存。

### 4.5 输入约束
- 最大输入长度、最大生成 token 数等。必须 **≤ 引擎构建时的 `max_input_len` / `max_seq_len`**（见 [[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]）。运行期不能超过编译期上限——这是最常见的“为什么我的长 prompt 报错”。

## 5. In-flight Batching：trtllm 服务的灵魂

普通“静态批”：N 条请求一起进、一起出，最长那条没生成完，其它已生成完的也得**空等**，GPU 大量空转。

**In-flight Batching（连续批处理 / continuous batching）**：以 **token step** 为粒度调度——某条请求生成完就立刻退出、立刻让新请求**插队**进同一个 batch。GPU 几乎不空转。

```
静态批（差）：                    In-flight Batching（好）：
step→  t0 t1 t2 t3 t4            step→  t0 t1 t2 t3 t4
 R1    ■  ■  ✓  .  .              R1    ■  ■  ✓  ─  ─
 R2    ■  ■  ■  ■  ✓              R2    ■  ■  ■  ■  ✓
 R3    ■  ✓  .  .  .              R3    ■  ✓  R4 R4 R4   ← R3完→R4立刻插进来
        └ R3/R1 早完却空等 ┘             └ 槽位被新请求复用，无空转 ┘
 ✓=完成  ■=在算  .=空转(浪费)        ─=该请求已离场，槽位释放
```

调度相关的核心权衡参数（语义）：

| 参数（语义） | 拧大 | 拧小 |
| :-- | :-- | :-- |
| max batch size | 吞吐↑、显存↑、单步延迟↑ | 省显存、低延迟、吞吐↓ |
| max num tokens（一个 step 内 token 总预算） | 容纳更多/更长请求，吞吐↑ | 显存可控但易排队 |
| 调度策略 max_utilization | 吞吐最大化，但显存吃紧时会抢占（重算/换出） | guaranteed_no_evict 更稳 |
| chunked context（长 prompt 分块预填充） | 长输入不再一次性吃爆显存，TTFT 更平滑 | —— |

**直觉**：`max_utilization` 像把座位塞满的拼车，偶尔有人被请下车（抢占重算）；`guaranteed_no_evict` 像对号入座，上车就保证到站。在线 chat 怕长尾延迟，常选后者；离线批量打分要榨吞吐，选前者。

## 6. KV Cache 参数：显存占用的最大变量

LLM 推理显存 ≈ 引擎权重（固定）+ **KV cache（随并发与序列长度线性增长）** + 激活。KV 几乎是唯一能在服务侧调的大头。

```
GPU 显存 80GB 示意：
┌──────────────┬───────────────────────────────┬──────┐
│ 权重 ~固定    │  KV cache（free_fraction 控制）  │ 激活 │
│  例 ~40GB    │  剩余的 X% 全划给 KV，越多并发越高 │ ~少  │
└──────────────┴───────────────────────────────┴──────┘
            free_fraction 越大→可服务的并发/上下文越长，但留给临时分配的余量越小→易 OOM
```

核心参数（语义，名字以版本为准）：

- **`kv_cache_free_gpu_mem_fraction`**：把“扣掉权重后的空闲显存”按这个比例（如 0.9）划给 KV 池。数值例子：80GB 卡、权重 40GB、fraction=0.9 → KV 池 ≈ (80−40)×0.9 = 36GB。调大→并发/长上下文能力↑，但留给 cudaMalloc 临时分配的余量↓，**OOM 风险↑**。生产常从 0.85~0.9 起步压测。
- **`enable_kv_cache_reuse`**：开启 **prefix 复用**——相同前缀（如同一段 system prompt）的 KV 不重算，命中即复用。多轮对话/同模板批量场景收益巨大。对应引擎侧需 `use_paged_context_fmha`（见 [[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]）。
- **host KV cache（CPU offload）**：把不活跃请求的 KV 换到主机内存，扩大“可缓存”容量，代价是换入换出的 PCIe 带宽开销。
- **`tokens_per_block`（paged KV 块粒度）**：分页 KV 每块多少 token。块小→碎片少但管理开销大；块大→反之。
- **KV 量化（INT8/FP8）**：把 KV 用低精度存，单位 token 显存减半甚至更多，可服务更长上下文，代价是少量精度损失（属引擎/校准侧，服务侧呼应开启）。

> 一句话：**先用 `free_fraction` 把池子开到压测不 OOM 的最大值，再用 `reuse` + KV 量化进一步抠出并发。**

## 7. 多 GPU / 多机拉起

引擎在 `trtllm-build` 阶段就按 **张量并行 TP × 流水并行 PP** 切好分片（见 [[ai-infra/网络/集合通信原语]] 的 AllReduce）。服务阶段必须用**完全一致**的并行度拉起，否则分片对不上。

```
TP=2, PP=2  →  world_size = 2×2 = 4  → 起 4 个 rank（4 张 GPU）

           ┌── PP stage 0 ──┐   ┌── PP stage 1 ──┐
 输入 ─▶    │ rank0   rank1  │ ─▶│ rank2   rank3  │ ─▶ 输出
           │  GPU0    GPU1  │   │  GPU2    GPU3  │
           └── TP=2 一层内切 ┘   └── TP=2 一层内切 ┘
 同一层算子在 TP 组内用 NCCL AllReduce 合并；跨 stage 走 PP 流水点对点传递
```

关键点：
- **`world_size` 必须 = 引擎的 `TP×PP`**，由 launcher（通常 `mpirun -n <world_size>`）拉起对应数量的 worker 进程。
- 多机时还要配 MPI 的 host 列表与节点间网络（IB/RoCE，见 [[ai-infra/网络/InfiniBand]]、[[ai-infra/网络/NCCL]]）。
- 一个 Triton 服务进程组对应**一个**引擎的全部分片；要服务多个模型就起多组或用多模型仓库。

## 典型流程：从引擎到 curl

```
① 构建引擎      trtllm-build ... → engines/  （TP/PP、max_input_len、max_seq_len 在此定死）
        │
        ▼
② 准备仓库      cp 官方模板 all_models/inflight_batcher_llm → my_repo/
        │       并按需把 5 个子目录建好
        ▼
③ 填 config     用脚本(fill_template) 把占位符替换为实际值：
        │       - tensorrt_llm: 引擎路径、batching策略、free_fraction、decoupled
        │       - pre/postprocessing: tokenizer 路径
        │       - ensemble/bls: 选一个作对外入口
        ▼
④ 起服务        mpirun -n <world_size> tritonserver --model-repository=my_repo ...
        │       （单卡 world_size=1，可不用 mpirun）
        ▼
⑤ 探活+压测     curl :8000/v2/health/ready ；用流式/非流式接口发请求；看 :8002/metrics
```

发请求时，对外入口是 `ensemble` 或 `tensorrt_llm_bls` 模型，输入字段一般含 `text_input`、`max_tokens`、采样参数（temperature/top_p/top_k）、`stream` 等——**这些是请求级参数，不在启动参数范围，但会被服务端按 config 约束裁剪**。

## 常见问题 / 坑

| 现象 | 根因 | 处理 |
| :-- | :-- | :-- |
| 起服务直接退出/段错误 | `world_size` 与引擎 TP×PP 不一致 | 用构建时同样的并行度，`mpirun -n` 对齐 |
| 长 prompt 报超长 | 运行期长度 > 引擎 `max_input_len/max_seq_len` | 这是编译期上限，需**重建引擎**，无法在服务侧放宽 |
| 并发一上来就 OOM | `free_fraction` 过大 / 实例数>1 / KV 没量化 | 降 fraction、count=1、开 KV 量化与 reuse |
| 吞吐上不去 | 用了静态批 / max batch size 太小 | 改 `inflight_fused_batching`、调大 batch/num_tokens |
| chat 不是流式 | 没开 decoupled 模式 | `tensorrt_llm` 设 decoupled，用 stream 接口 |
| 长尾延迟抖动大 | 调度用 `max_utilization` 触发抢占重算 | 换 `guaranteed_no_evict`，或留更多 KV 余量 |
| tokenizer 报错/乱码 | pre/postprocessing 的 tokenizer 路径或版本不对 | 用与引擎同源的 tokenizer |
| 改了 config 不生效 | 模型未重新加载 | 重启服务或用模型管理 API reload |
| metrics 拿不到 | metrics 端口/开关未开 | 开 Prometheus 端点，接 [[llmops/kubernetes]] 监控 |
| 多机起不来 | MPI host/网络/NCCL 配置缺失 | 配 host 列表与 IB/RoCE，验证 NCCL 通信 |

> 收尾护栏：**参数的确切名字与默认值请以你所用 `tensorrtllm_backend` 版本的 `docs/model_config.md`、`tensorrt_llm` executor 文档以及 Triton 官方文档为准**；本文聚焦“每个旋钮的含义与权衡方向”，便于你在任意版本下快速定位该拧哪个。

## 🔗 跳转链接

- 📍 [[00-知识地图]]
- 同目录：[[llm-inference/tensorrt-llm/README]]、[[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]
- 概念底座：[[llm-optimizer/kv-cache]]、[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 多卡网络：[[ai-infra/网络/集合通信原语]]、[[ai-infra/网络/NCCL]]、[[ai-infra/网络/InfiniBand]]
- 上线运维：[[llmops/kubernetes]]
- 同类对比：[[llm-inference/vllm/README]]、[[llm-inference/sglang/README]]、[[llm-inference/tensorrt/README]]
