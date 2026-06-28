# vLLM 上手与全景导览

> 一句话定位：vLLM 是一个开源的高吞吐 LLM 推理/服务引擎——核心靠 **PagedAttention（把 KV-Cache 当虚拟内存分页管）** + **连续批处理（Continuous Batching）**，把 GPU 显存利用率和并发吞吐推到极限。本文是「从安装到上线」的全景上手页，更深的机制/参数/源码见同目录兄弟文档。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] [[llm-inference/README]] [[llm-inference/KV-Cache优化]] [[llm-inference/sglang/README]]

## 阅读地图

| 节 | 你会搞懂 | 一句话 |
|----|---------|--------|
| 0 | 一句话锚点 | vLLM 干嘛的、靠什么吃饭 |
| 1 | 它解决什么问题 | 朴素 HF 推理为什么慢、为什么浪费显存 |
| 2 | 两大支柱 | PagedAttention + Continuous Batching 直觉 |
| 3 | 整体架构 | Engine / Scheduler / BlockManager / Worker 怎么串 |
| 4 | 三种用法 | 离线批量 / OpenAI 兼容服务 / 原生 API |
| 5 | 关键启动参数 | 每个参数做什么用、怎么权衡（不背默认值） |
| 6 | 分布式与并行 | TP / PP / 单机多卡 / 多机怎么开 |
| 7 | 量化与显存账 | 怎么把大模型塞进有限显存 |
| 8 | 调用链时序 | 一条请求从进到出走了哪些组件 |
| 9 | 数值例子 | KV-Cache 显存手算一遍 |
| - | 对照表 / 常见坑 | vs HF/TGI/SGLang，部署易踩点 |

---

## 0. 一句话锚点

> **vLLM 把操作系统"虚拟内存 + 分页"的思想搬到了 GPU 上的 KV-Cache 管理，并配上"逐 token 进出"的连续批处理，于是同样的显卡能同时服务数倍的并发请求。**

记住这张图就抓住了灵魂：

```
                ┌──────────────────────────────────────┐
   vLLM 引擎 =  │  PagedAttention      → 省显存（消碎片）│
                │        +                              │
                │  Continuous Batching → 提吞吐（不空转）│
                └──────────────────────────────────────┘
```

剩下的一切（OpenAI 兼容 server、张量并行、量化、前缀缓存、chunked prefill）都是围绕这两根支柱搭起来的工程外壳。

---

## 1. 地基：它到底解决什么问题

要理解 vLLM 的价值，先看朴素推理（比如直接用 HuggingFace `model.generate`）做服务时的两大痛点。

### 1.1 自回归推理为什么必须有 KV-Cache

LLM 推理是**逐 token** 的自回归过程：

```
prompt:  "中国 的 首都 是"
          step1 → 北      (基于前4个token算注意力)
          step2 → 京      (基于前5个token算注意力)
          step3 → <eos>
```

每生成一个新 token，注意力都要让"当前 token"去看"前面所有 token"的 Key/Value。如果每步都把前面所有 token 重新算一遍，复杂度是 $O(n^2)$ 的重复劳动。于是把每个 token 的 K、V 算一次后**缓存**起来，下一步直接复用——这就是 **KV-Cache**。它把每步的注意力计算从"重算全序列"降到"只算新 token"。

代价是：KV-Cache 要占显存，且**随并发数 × 序列长度线性增长**。一个长对话、高并发的服务，KV-Cache 往往比模型权重还吃显存。

### 1.2 朴素方案的两个浪费

| 浪费 | 朴素做法 | 后果 |
|------|---------|------|
| **显存碎片** | 给每条请求按"最大可能长度"预留一整块连续显存 | 实际只用了一小半，剩下的被占着不能给别人用，内/外部碎片可吃掉 60%+ |
| **GPU 空转** | 静态批处理：凑齐一批一起跑，等批里最慢那条结束才放下一批 | 短请求早早算完却得干等，长尾拖累整批，GPU 利用率低 |

vLLM 用 PagedAttention 解决第一个，用 Continuous Batching 解决第二个。

---

## 2. 两大支柱的直觉

### 2.1 PagedAttention：KV-Cache 分页

类比操作系统：进程看到的是"连续的虚拟地址"，但物理内存其实是被切成**页（page）**离散分配的，靠页表做映射。vLLM 把这套搬到 KV-Cache：

- KV-Cache 被切成固定大小的 **block / page**（每块存若干个 token 的 K、V）。
- 每条请求维护一张**块表（block table）**，把"逻辑上连续的 token 序列"映射到"物理上离散的块"。
- 按需分配：要多少给多少，写满一块再要下一块 → 几乎没有内部碎片。
- 可共享：多条请求若有相同前缀（同一个 system prompt），可让块表指向**同一物理块** → 前缀缓存。

```
逻辑视角(请求看到的连续序列)        物理视角(显存里离散的块)
┌──┬──┬──┬──┬──┐                  ┌───┐ ┌───┐ ┌───┐
│t0│t1│t2│t3│t4│  ──块表映射──▶    │blk7│ │blk2│ │blk9│  (顺序无所谓)
└──┴──┴──┴──┴──┘                  └───┘ └───┘ └───┘
```

### 2.2 Continuous Batching：连续批处理

不再"凑整批一起进、一起出"，而是把调度粒度降到**每个解码步（iteration）**：

```
静态批处理：  [====A====]
              [==B==]      ← B早完，但要等A
              [======C======]   GPU此时部分空转

连续批处理：  每一步都重组batch，谁完成就让位，新请求随时插进来
              step_k: {A,B,C}  → B完成 → step_{k+1}: {A,C,D(新来的)} ...
```

效果：GPU 几乎一直满载，吞吐显著提升，新请求等待时间也更短。

> 这两根支柱的**深入推导、显存碎片量化、调度抢占/换出**细节，见 [[llm-inference/vllm/README]] 与 [[llm-inference/KV-Cache优化]]，本页不重复。

---

## 3. 整体架构：组件怎么串起来

vLLM 的代码骨架可以概括为"一个大脑 + 一组手脚 + 一本账"：

```
                 用户请求 (prompt + SamplingParams)
                          │
            ┌─────────────▼──────────────┐
            │         LLMEngine          │  ← 大脑：编排整个生命周期
            │  ┌──────────────────────┐  │
            │  │     Scheduler        │  │  ← 每一步决定"跑谁/抢占谁/换出谁"
            │  │   + BlockManager     │  │  ← 账本：管 KV block 的分配/释放/映射
            │  └──────────────────────┘  │
            └─────────────┬──────────────┘
                          │ 下发本步要算的 batch
            ┌─────────────▼──────────────┐
            │   Worker × N (每GPU一个)    │  ← 手脚：真正在 GPU 上算
            │   └ ModelRunner             │     前向 + 采样
            │       └ AttentionBackend    │     PagedAttention/FlashAttn 算子
            └─────────────┬──────────────┘
                          │ 新 token / logits
                          ▼
            Tokenizer/Detokenizer → 流式返回给用户
```

- **LLMEngine**：核心编排器，驱动"调度 → 执行 → 采样 → 出 token"的主循环。
- **Scheduler**：维护 `waiting / running / swapped` 三个队列，每步决定哪些请求进入这一轮计算，显存不够时按策略抢占/换出。
- **BlockManager**：KV-Cache 分页的"账本"，负责物理块的分配、释放、引用计数（前缀共享靠它）。
- **Worker / ModelRunner / AttentionBackend**：在 GPU 上做实际前向与注意力计算；多卡时每张卡一个 Worker。
- **入口层**：`LLM`（离线）、`AsyncLLMEngine` + `api_server`（在线服务）。

> 各组件的源码文件位置、类职责拆解见 [[llm-inference/vllm/源码]]；一条请求的逐步生命周期见 [[llm-inference/vllm/请求处理流程]]。

---

## 4. 三种用法：从离线到上线

### 4.1 安装（环境是第一道坑）

```bash
# 推荐独立环境，避免污染系统 torch
conda create -n vllm python=3.10 -y && conda activate vllm

# 直接装预编译轮子（自带匹配的 CUDA 算子）
pip install vllm
```

要点（**具体版本以官方文档为准**）：
- vLLM 的 wheel 与 **CUDA / PyTorch 版本强绑定**，装错版本最常见的报错就是 CUDA mismatch；不确定时用官方 Docker 镜像最省心。
- 国产卡（昇腾等）走单独的适配分支/镜像，见 [[ai-infra/算力/昇腾NPU]]，不要直接 `pip install vllm`。
- 共享内存：跑服务时常见的卡死与 `--shm-size` 太小有关，容器里要调大。

### 4.2 离线批量推理（`LLM` 类）

适合：一次性把一批 prompt 跑完（评测、数据合成、离线打分）。

```python
from vllm import LLM, SamplingParams

prompts = [
    "中国的首都是",
    "用一句话解释什么是注意力机制：",
]
# 采样参数：控制随机性与停止条件
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")   # 模型名/本地路径

outputs = llm.generate(prompts, sampling_params)
for o in outputs:
    print(o.prompt, "→", o.outputs[0].text)
```

`SamplingParams` 是"怎么采样"的旋钮：`temperature`（温度，越高越随机）、`top_p`/`top_k`（核采样/截断）、`max_tokens`（最多生成多少）、`stop`（停止词）、`n`（每条 prompt 生成几个候选）。

### 4.3 在线服务（OpenAI 兼容，最常用）

适合：对外提供 HTTP 接口，且想直接复用 OpenAI SDK 的客户端生态。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

起来后就是一个 **OpenAI 兼容**的服务，可用任何 OpenAI 客户端打：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role":"user","content":"你好"}]
      }'
```

它暴露 `/v1/completions`、`/v1/chat/completions`、`/v1/embeddings`（视模型而定）等端点，鉴权、流式（SSE）、多并发都内置。这是生产部署的首选形态。

### 4.4 原生 API Server

`vllm.entrypoints.api_server` 提供一个更轻量的原生 `/generate` 端点（早期演示用途，生产建议用上面的 OpenAI 兼容版）。三种用法的取舍：

```
离线批量(LLM)         → 跑批/评测/数据生产，无需常驻服务
OpenAI 兼容 server    → 线上服务首选，生态最全
原生 api_server       → 简单演示/自定义协议
```

> 完整的端点、请求字段、流式细节以官方文档与 [[llm-inference/vllm/服务启动参数]] 为准。

---

## 5. 关键启动参数：做什么用、怎么权衡

下面只讲**参数类别的作用与权衡**，不背具体默认值（默认值随版本变，**以 `--help` 与官方文档为准**）。

| 参数（类别） | 作用 | 怎么权衡 |
|---|---|---|
| `--model` | 指定模型名或本地路径 | HF repo 名会自动下载；离线环境用本地路径 |
| `--tensor-parallel-size` (TP) | 把一个模型切到多张卡上 | 单卡放不下时调大；TP 越大单步通信开销越大，需高速互联（NVLink/IB） |
| `--pipeline-parallel-size` (PP) | 按层切分，跨节点流水 | 多机扩展时用；引入流水气泡，延迟敏感场景慎用 |
| `--gpu-memory-utilization` | 允许 vLLM 占用的显存比例 | 调高→更多 KV block→更高并发；太高易 OOM，给别的进程留余量 |
| `--max-model-len` | 最大上下文长度 | 决定单请求最长 prompt+生成；越大单请求占的 KV 越多，并发越少 |
| `--max-num-seqs` | 同时在跑的最大序列数 | 吞吐 vs 单请求延迟的折中；调大吞吐高但可能加剧排队/抢占 |
| `--max-num-batched-tokens` | 一步最多处理多少 token | 与 chunked prefill 配合，平衡长 prompt 的 prefill 与 decode |
| `--dtype` | 计算精度（fp16/bf16 等） | bf16 数值更稳；影响显存与速度 |
| `--quantization` | 启用量化（如 AWQ/GPTQ/FP8 等） | 大幅省显存、可能略损精度，需匹配的量化权重 |
| `--enable-prefix-caching` | 开启前缀缓存 | 大量请求共享同一 system prompt 时显著省算力 |
| `--enable-chunked-prefill` | 长 prompt 分块预填充 | 避免一个超长 prefill 卡住所有 decode，改善长尾延迟 |
| `--swap-space` | CPU 换出空间 | 显存吃紧时把暂不活跃的 KV 换到内存，避免直接拒绝请求 |
| `--served-model-name` | 对外暴露的模型名 | 让客户端 `model` 字段与内部路径解耦 |

> 完整参数与默认值见 [[llm-inference/vllm/服务启动参数]]，源头是引擎的 `arg_utils`（**以源码/官方文档为准**）。

---

## 6. 分布式与并行：一张卡放不下怎么办

```
          单卡放得下           →  普通启动，最省事
                │
          单机多卡放得下       →  --tensor-parallel-size = 卡数 (走 NVLink)
                │
          单机放不下/要扩节点  →  TP(机内) × PP(跨机)  组合
```

- **张量并行（TP）**：把每层的权重矩阵横/纵切到多卡，每步都要 all-reduce 同步，**通信频繁**，因此最好在**机内 NVLink** 上做。原理见 [[llm-inference/大模型推理张量并行]] 与 [[ai-infra/网络/集合通信原语]]。
- **流水线并行（PP）**：按层把模型切成几段，分到不同节点，**跨机扩展**时用，但有流水气泡。
- **多机部署**：需要节点间高速网络（[[ai-infra/网络/InfiniBand]]）和一致的环境/权重，通信底座是 [[ai-infra/网络/NCCL]]。

经验法则：**先尽量用 TP 把机内卡吃满，再用 PP 跨机**；通信跟不上时并行收益会被通信吃掉。

---

## 7. 量化与显存账

把大模型塞进有限显存的两条路：

1. **降权重精度**：fp16/bf16 → int8/int4（AWQ、GPTQ）或 FP8，用 `--quantization` 指定，需配套量化权重。FP8 细节见 [[llm-inference/vllm/FP8]]。
2. **降 KV-Cache 占用**：分页本身已极大减少碎片；还可对 KV 做量化、用前缀缓存共享、用 `--swap-space` 换出。

显存粗账（运行时大致分三块）：

```
总显存 ≈ 模型权重  +  KV-Cache(随并发×长度涨)  +  激活/临时缓冲
```

`--gpu-memory-utilization` 决定 vLLM 在扣掉权重后，拿多大比例的剩余显存去开 KV block 池——这直接决定能撑多少并发。量化基础见 [[llm-compression/quantization/量化基础]]。

---

## 8. 调用链时序：一条请求走了哪些组件

```
客户端
  │ POST /v1/chat/completions
  ▼
api_server (OpenAI 兼容层)
  │ 解析请求 + 构造 SamplingParams
  ▼
AsyncLLMEngine → LLMEngine.add_request()
  │ tokenize，放进 Scheduler 的 waiting 队列
  ▼
Scheduler.schedule()  ←──┐ 每一步循环
  │ 选本步 batch          │
  │ BlockManager 分配KV块 │
  ▼                      │
Worker/ModelRunner       │
  │ 前向 → logits → 采样  │
  ▼                      │
得到新 token ────────────┘ 未结束则下一步继续 decode
  │ 结束(<eos>/达 max)
  ▼
Detokenize → 流式(SSE)回吐 → 释放该请求的 KV 块
```

> 这条链路每一步的队列状态、抢占/换出触发条件，详见 [[llm-inference/vllm/请求处理流程]]。

---

## 9. 数值例子：KV-Cache 显存手算

设一个 7B 模型（仅作演示，具体以模型 config 为准）：层数 $L=32$，KV 注意力头维度合计每 token 每层需存 K、V 两份。每 token 的 KV 字节数近似：

$$\text{bytes/token} = 2 \times L \times d_{kv} \times \text{bytes\_per\_elem}$$

假设这一项算下来约 **0.5 MB/token**（量级演示值）。那么：

- 一条 2048 token 的请求：$2048 \times 0.5\text{MB} \approx 1\text{GB}$ KV-Cache。
- 并发 40 条这样的请求：$40 \times 1\text{GB} = 40\text{GB}$ —— 已经超过模型权重（约 14GB fp16）本身。

结论很直观：**KV-Cache 才是高并发服务的显存大头**，所以"分页省碎片 + 量化 + 换出"才如此关键。真实数字请按你模型的 `hidden_size / num_kv_heads / num_layers` 套公式计算，本例只为建立量级直觉。

---

## 对照表：vLLM vs 同类引擎

| 维度 | vLLM | HF Transformers | TGI | SGLang | TensorRT-LLM |
|---|---|---|---|---|---|
| 定位 | 高吞吐通用服务 | 通用/研究基线 | 生产服务 | 高吞吐+结构化/复杂调度 | 极致延迟(NV专用) |
| 杀手锏 | PagedAttention+连续批 | 生态最全 | 工程化完善 | RadixAttention 前缀树 | 编译期深度优化 |
| 上手难度 | 低 | 低 | 中 | 中 | 高(需编译引擎) |
| OpenAI 兼容 | 是 | 否(需自封) | 是 | 是 | 需配套 |
| 适合 | 在线/离线通吃 | 快速验证 | 稳定上线 | Agent/多轮/约束生成 | 单一模型极致优化 |

详见 [[llm-inference/sglang/README]] 与 [[llm-inference/tensorrt/README]]。

---

## 常见问题 / 坑（表格）

| 现象 | 根因 | 解法 |
|---|---|---|
| `pip install vllm` 后 import 报 CUDA mismatch | wheel 与本机 CUDA/torch 版本不匹配 | 用官方 Docker 镜像，或装对应 CUDA 版本的 wheel（**版本以官方为准**） |
| 服务启动即 OOM | `--gpu-memory-utilization` 太高或 `--max-model-len`×并发超显存 | 调低利用率、降 max-len、开量化、减 `--max-num-seqs` |
| 容器里偶发卡死/通信失败 | `--shm-size` 太小 | docker run 加大 `--shm-size`（如 8g+） |
| 多卡未提速反而变慢 | TP 跨机走了慢网络，被通信吃掉收益 | TP 限机内 NVLink，跨机用 PP；检查 NCCL/IB |
| 长 prompt 一来其它请求全卡 | 超长 prefill 独占一步 | 开 `--enable-chunked-prefill`，调 `--max-num-batched-tokens` |
| 大量相同 system prompt 仍重复计算 | 没开前缀缓存 | 加 `--enable-prefix-caching` |
| 量化模型加载失败 | 权重格式与 `--quantization` 不匹配 | 用对应方法（AWQ/GPTQ/FP8）导出的权重，方法名要对上 |
| 并发一高延迟暴涨 | `--max-num-seqs` 过大导致排队/抢占频繁 | 降并发上限，按 SLA 在吞吐与延迟间取舍 |

---

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 同目录深入：[[llm-inference/vllm/README]]（PagedAttention 原理） · [[llm-inference/vllm/服务启动参数]] · [[llm-inference/vllm/请求处理流程]] · [[llm-inference/vllm/源码]] · [[llm-inference/vllm/长文本推理]] · [[llm-inference/vllm/FP8]] · [[llm-inference/vllm/FAQ]]
- 推理总览：[[llm-inference/README]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/大模型推理张量并行]]
- 同类引擎：[[llm-inference/sglang/README]] · [[llm-inference/tensorrt/README]]
- 底层与硬件：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/算力/昇腾NPU]]
- 压缩：[[llm-compression/quantization/量化基础]]
