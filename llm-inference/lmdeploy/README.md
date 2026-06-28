# LMDeploy 推理引擎

> LMDeploy 是上海人工智能实验室（InternLM 团队）开源的 LLM 部署工具箱，核心是用 C++/CUDA 手写的 **TurboMind** 推理引擎，主打"持久化批处理 + KV 量化 + 高效 kernel"，在国产大模型（InternLM/Qwen 等）上吞吐与延迟表现突出。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/vllm/README]] [[llm-inference/README]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | TurboMind / Persistent Batch |
| 1 | 它解决什么问题（地基） | 显存、批处理、吞吐 |
| 2 | 整体架构与两套后端 | TurboMind vs PyTorch engine |
| 3 | TurboMind 引擎内部 | C++/CUDA、kernel fusion |
| 4 | 持久化批处理（连续批处理） | Persistent Batch、槽位 |
| 5 | KV Cache 与 KV 量化 | PagedAttention、INT8/INT4 |
| 6 | 权重量化（W4A16/AWQ） | 量化推理 |
| 7 | 并行与服务化 | TP、api_server |
| 数值例子 | 显存/吞吐手算 | 7B 模型实算 |
| 对照表 | 与 vLLM / TGI 对比 | 选型 |
| 常见问题 | 易错点 | FAQ |

---

## 0. 一句话锚点

**LMDeploy = 一个"工具箱"（量化 + 部署 + 服务），它的灵魂是 TurboMind 这台用 C++/CUDA 写的"推理发动机"。**

它要做的事和 vLLM 一样：把一堆并发的生成请求，尽可能快、尽可能省显存地喂给 GPU。区别在于实现路线——vLLM 以 Python 为主、PagedAttention 为旗帜；TurboMind 干脆把整条前向路径用 C++ 重写，并把"持久化批处理"和"KV 量化"做成默认能力。

---

## 1. 地基：它到底解决什么问题

要理解任何推理引擎，先把 LLM 推理的三个"先天痛点"说清楚。

### 1.1 自回归生成是"一个一个字往外蹦"

LLM 生成文本是 **自回归（autoregressive）** 的：给定已有 token，预测下一个 token，再把它接回去预测再下一个。

```
prompt: "中国的首都是"
 step1 → "北"      (用 prompt 这 6 个 token 算)
 step2 → "京"      (用 7 个 token 算)
 step3 → "。"      (用 8 个 token 算)
```

这意味着一次请求要跑 N 次前向（N = 输出长度），每次只产出 1 个 token。这是 **decode 阶段**，特点是"计算少、访存多"，GPU 算力跑不满，瓶颈在 **显存带宽**。

而处理 prompt 的那一次叫 **prefill 阶段**，N 个输入 token 一次并行算完，特点是"计算密集"。

> 核心矛盾：prefill 算力密集，decode 带宽密集。引擎设计的一切技巧，都是为了让这两个阶段都不浪费 GPU。

### 1.2 KV Cache：用显存换计算

每生成一个新 token，注意力都要看"前面所有 token"。如果每步都重算前面所有 token 的 Key/Value，代价是 $O(N^2)$。

解决办法：把每个 token 算过的 Key、Value 缓存起来，这就是 **KV Cache**。于是 decode 每步只算"当前这一个新 token 的 Q，与缓存里所有 K、V 做注意力"，复杂度降到每步 $O(N)$。

代价：KV Cache 极其吃显存。单个 token 的 KV 大小为：

$$\text{bytes}_{\text{per token}} = 2 \times L \times H \times d_{head} \times \text{dtype\_bytes}$$

其中 $L$=层数，$H$=KV 头数，$d_{head}$=每头维度，$2$ 是 K 和 V 各一份。

### 1.3 静态批处理浪费严重

朴素做法（静态批处理）：凑齐一批请求一起跑，等**整批**都生成完才返回、才接新请求。

```
静态批处理（Static Batching）：木桶效应
req A: ████████████████████  (输出 200 token)
req B: ████░░░░░░░░░░░░░░░░░░  (输出 40 token，但要陪 A 等到底)
req C: ██░░░░░░░░░░░░░░░░░░░░  (输出 20 token，干等)
           ↑ 这些 ░ 是空转的 GPU
```

短请求被长请求拖住，GPU 大量空转。**持久化/连续批处理**就是来解决这个的（见第 4 节）。

---

## 2. 整体架构：一个工具箱，两套后端

LMDeploy 不是单一引擎，而是一套体系，包含两条推理后端：

```
                    ┌─────────────────────────────────────┐
                    │            LMDeploy 工具箱            │
                    │                                       │
   用户/请求 ──────▶│  ┌────────────┐   ┌────────────────┐ │
   (HTTP/Python)    │  │ api_server │   │  pipeline (py) │ │
                    │  │ (服务化层)  │   │   (离线批量)   │ │
                    │  └─────┬──────┘   └───────┬────────┘ │
                    │        │                  │          │
                    │        ▼                  ▼          │
                    │  ┌──────────────────────────────┐    │
                    │  │       调度 / 请求管理          │    │
                    │  └──────┬───────────────┬───────┘    │
                    │         │               │            │
                    │  ┌──────▼──────┐  ┌─────▼─────────┐  │
                    │  │ TurboMind   │  │ PyTorch engine│  │
                    │  │ (C++/CUDA)  │  │  (纯 Python)  │  │
                    │  │ 高性能默认  │  │  易扩展/调试  │  │
                    │  └──────┬──────┘  └───────────────┘  │
                    │         ▼                            │
                    │   量化工具(W4A16/AWQ, KV INT8/INT4)  │
                    └─────────────────────────────────────┘
                                  │
                                  ▼
                              GPU (CUDA)
```

两套后端的分工：

| 后端 | 语言 | 定位 | 何时用 |
|------|------|------|--------|
| **TurboMind** | C++/CUDA | 极致性能，生产首选 | 追求吞吐/延迟，模型在支持列表内 |
| **PyTorch engine** | 纯 Python（PyTorch） | 易扩展、好调试、模型支持广 | 新模型快速适配、需要改代码 |

> 设计哲学：TurboMind 用底层语言换性能，但适配新模型成本高；PyTorch engine 用 Python 换灵活性，新模型来了改起来快。两者共享上层的服务化与量化能力。具体支持哪些模型走哪条后端，以官方文档为准。

---

## 3. TurboMind 引擎内部

TurboMind 脱胎于 NVIDIA 的 FasterTransformer，但做了大量重写与扩展。它的"快"来自三个层面。

### 3.1 整条前向用 C++/CUDA 重写

普通 PyTorch 推理，每个算子都在 Python 层调度，有解释器开销、有 kernel 之间的启动间隙。TurboMind 把 attention、FFN、norm、采样等整条链路放进 C++ 运行时，直接调 CUDA kernel，Python 只负责"递请求、收结果"。

```
PyTorch 路线（每步都过 Python）：
  py → linear → py → attn → py → norm → py → ...   (虚线=Python 调度开销)
TurboMind 路线（一杆子到底）：
  py(收请求) ─▶ [ C++ runtime: linear·attn·norm·sample 全在 GPU 串起来 ] ─▶ py(返回)
```

### 3.2 Kernel 融合（fusion）

把多个小算子合并成一个 CUDA kernel，减少"读显存→算→写显存→再读"的来回。例如把 RMSNorm + 量化、QKV 投影 + RoPE 等融合。融合后中间结果留在寄存器/共享内存，不落显存，省带宽。

```
融合前：  [GEMM]→显存→[bias]→显存→[激活]→显存   (3 次写+读)
融合后：  [GEMM+bias+激活 一个 kernel]→显存       (1 次写)
```

decode 阶段是带宽瓶颈，少一次访存就直接变快，融合收益巨大。

### 3.3 自实现高效 attention kernel

TurboMind 有自己的 attention 实现，配合分页式 KV 管理，支持 GQA（分组查询注意力）、RoPE、ALiBi 等。对 decode 这种"1 个 query 对全部 KV"的场景做了专门优化（类似 FlashAttention 的 decode 变体思路）。

---

## 4. 持久化批处理（Persistent Batch / 连续批处理）

这是 TurboMind 吞吐高的关键，也是原始笔记里强调的部分。它就是业界常说的 **continuous batching / in-flight batching**，TurboMind 把它叫 **Persistent Batch**。

### 4.1 核心思想：批处理"常驻不解散"，按 token 步动态进出

把 batch 想象成一辆**始终在跑的班车**，车上有固定数量的"座位（槽位 slot）"。不像静态批处理"坐满才发车、到站全下车"，持久化批处理是**每到一站（每个 forward step）就让满员的乘客下车、让排队的乘客补上空座**。

### 4.2 调度循环（对应原始笔记的步骤）

```
                ┌──────────────┐
   新请求 ────▶ │  请求队列     │
                └──────┬───────┘
                       │ 拉取
                       ▼
        ┌─────────────────────────────────────┐
        │      Persistent 线程（常驻循环）      │
        │                                       │
        │  ① 有空闲槽位? ──是──▶ 从队列拉请求    │
        │                       尽量填满空槽    │
        │        │ 否                           │
        │        ▼                              │
        │  ② 对当前 batch 做一次 Forward        │
        │     (prefill 新进的 + decode 在跑的)  │
        │        │                              │
        │        ▼                              │
        │  ③ 检查每个请求是否生成完毕            │
        │     (遇到 EOS / 达到 max_len)         │
        │        │                              │
        │   完成的: 发回结果 + 释放槽位 ──┐     │
        │        │                        │     │
        │        └────────── 回到 ① ◀─────┘     │
        └─────────────────────────────────────┘
```

逐句对应原始笔记：

1. **请求队列**：所有推理请求先入队。
2. **Persistent 线程**：若 batch 有空闲槽位，就从队列拉请求尽量填满；若没有空槽，就继续对当前 batch 做 Forward。
3. **每 Forward 一次**：判断有无请求结束。结束的发结果、释放槽位，然后回到第 1 步。

### 4.3 为什么这样就快

```
持久化批处理：座位随时复用，几乎不空转
step:  1    2    3    4    5    6    7
slot0: A    A    A    A    A    A    A
slot1: B    B    B   [完]  E    E    E   ← B 走了 E 立刻补位
slot2: C   [完]  D    D    D    D   [完]  ← C 走了 D 立刻补位
                 ↑ 没有人陪着空等，GPU 一直满载
```

- **短请求不被长请求拖住**：B 生成完立刻释放，新请求 E 当 step 就能上车。
- **GPU 利用率高**：每一步 batch 都尽量是满的，算力不浪费。
- **吞吐显著提升**：相比静态批处理，连续批处理通常能把吞吐拉高数倍（具体倍数取决于请求长度分布）。

> 补充：prefill 和 decode 混在同一个 batch 里跑（新请求要先 prefill，老请求在 decode）。如何平衡这两者的调度，是各引擎实现细节的差异点，以官方文档为准。

---

## 5. KV Cache 管理与 KV 量化

### 5.1 分页式 KV Cache（PagedAttention 思路）

朴素 KV Cache 给每个请求预留"最大长度"的连续显存，绝大多数请求用不满，浪费巨大（内部碎片）。

分页方案：把 KV Cache 切成固定大小的 **block（页）**，请求用多少分多少，像操作系统的虚拟内存分页。TurboMind 同样采用分页式的 KV 管理。

```
请求 A 实际用了 3 页：[blk7][blk2][blk9]   ← 物理上不连续
请求 B 实际用了 2 页：[blk1][blk5]
        ↑ 一个全局 block 池，按需分配/回收，碎片极小
```

### 5.2 KV 量化：把 KV Cache 从 FP16 压成 INT8/INT4

这是 LMDeploy 的招牌能力之一。前面算过，KV Cache 极吃显存，是长上下文/高并发的主要瓶颈。**KV 量化** = 把缓存里的 K、V 从 16 bit 压到 8 bit（INT8）甚至 4 bit（INT4）存储。

```
FP16 KV:  每个数 16 bit  ████████████████
INT8 KV:  每个数  8 bit  ████████           (省一半)
INT4 KV:  每个数  4 bit  ████               (省四分之三)
```

带来的好处是双重的：

1. **显存省下来** → 同样显存能放下更多请求（更大 batch）或更长上下文。
2. **decode 是带宽瓶颈** → KV 变小，每步读 KV 的字节数变少，**速度也变快**。

代价：量化有精度损失。LMDeploy 在量化时会按通道/分组做缩放（scale）来尽量保精度，实测对生成质量影响通常很小，但 INT4 比 INT8 损失更明显。是否开、开到几 bit，要结合你的精度要求权衡，具体配置项以官方文档为准。

> 注意区分两种量化：**KV 量化**（压的是运行时的 KV Cache）和**权重量化**（压的是模型参数，见第 6 节），两者正交，可叠加。

---

## 6. 权重量化（W4A16 / AWQ）

KV 量化压"运行时缓存"，权重量化压"模型本身"。

- **W4A16**：Weight 用 4 bit 存，Activation 用 16 bit 算。模型权重显存直接降到约 1/4，访存大幅减少；激活仍用 FP16 保证精度。
- **AWQ（Activation-aware Weight Quantization）**：一种 4 bit 权重量化算法，核心思想是"重要权重通道少量保护、其余激进量化"，在 4 bit 下尽量保精度。LMDeploy 集成了 AWQ 量化与对应的高效 kernel。

```
FP16 7B 模型权重 ≈ 14 GB
  └─ W4A16 后 ≈ 3.5 GB   ← 消费级显卡也跑得动
```

为什么权重量化也能提速：decode 阶段要把全部权重从显存读一遍来算这一个 token，权重小了，读得快了。

---

## 7. 并行与服务化

### 7.1 张量并行（TP）

单卡装不下时，把每层的权重矩阵**按列/行切**到多张卡上，各卡算一部分，再用 all-reduce 合并。TurboMind 支持 TP，启动时指定 GPU 数即可把一个大模型摊到多卡。

```
TP=2:   每层权重切两半
  GPU0: [W 左半] ┐
                  ├─ 算完 all-reduce 拼起来
  GPU1: [W 右半] ┘
```

### 7.2 服务化：api_server

LMDeploy 提供 `api_server`，把引擎包成一个 HTTP 服务，并对外暴露 **OpenAI 兼容**接口。于是原本调用 OpenAI 的客户端代码，改个 base_url 就能指向你自己的部署。

```
客户端(OpenAI SDK) ──HTTP──▶ api_server ──▶ 调度 ──▶ TurboMind ──▶ GPU
                              (OpenAI 兼容 /v1/chat/completions)
```

> 具体启动命令、参数名、端口默认值等，请以官方文档与 `lmdeploy serve api_server --help` 为准，不同版本可能变化。

---

## 数值例子：7B 模型显存与吞吐手算

设定一个具体场景，把前面公式落到数字上（取整、便于心算）。

**模型**：类 LLaMA 7B，$L=32$ 层，hidden=4096，注意力头 32 个、每头 $d_{head}=128$，KV 头也是 32（标准 MHA）。

### 例 1：单 token KV Cache 大小（FP16）

$$\text{bytes} = 2 \times L \times H \times d_{head} \times 2 = 2 \times 32 \times 32 \times 128 \times 2$$

$$= 2 \times 32 \times 4096 \times 2 = 524288 \text{ bytes} \approx 0.5 \text{ MB/token}$$

（$H \times d_{head}=32\times128=4096$，即每层 K 是 4096 维、V 是 4096 维。）

### 例 2：一条 2048 token 的请求要多少 KV 显存

$$0.5 \text{ MB} \times 2048 \approx 1024 \text{ MB} = 1 \text{ GB}$$

一条满上下文的请求就要 1 GB KV！

### 例 3：KV 量化的收益

| 精度 | 单 token | 2048 token 一条 | 24GB 卡（扣权重 14GB，余 10GB）能并发几条 |
|------|----------|------------------|-----------|
| FP16 | 0.5 MB | 1.0 GB | ~10 条 |
| INT8 | 0.25 MB | 0.5 GB | ~20 条 |
| INT4 | 0.125 MB | 0.25 GB | ~40 条 |

**结论**：KV 量化到 INT4，同一张卡的并发能力翻 4 倍。若再叠加 W4A16 权重量化（权重 14GB→3.5GB），省出的 10.5GB 又能多放约 20~40 条请求。这就是 LMDeploy 在高并发下吞吐领先的物理来源。

### 例 4：为什么 KV 量化也提速（带宽视角）

decode 每步要读"全部 KV"。设 batch 内总缓存 8 GB，GPU 显存带宽假设 2 TB/s：

- FP16：每步读 8 GB，仅读 KV 就需 $8/2000 = 4\text{ ms}$。
- INT4：KV 变 2 GB，$2/2000 = 1\text{ ms}$。

光读 KV 这一项就省 3 ms/步，对动辄上百步的 decode 是实打实的提速。

---

## 对照表：LMDeploy(TurboMind) vs vLLM vs TGI

| 维度 | LMDeploy / TurboMind | vLLM | TGI (HuggingFace) |
|------|----------------------|------|--------------------|
| 核心引擎语言 | C++/CUDA 重写 | Python + CUDA kernel | Rust(服务) + Python/CUDA |
| 连续批处理 | ✅ Persistent Batch | ✅ Continuous Batching | ✅ |
| 分页 KV | ✅ | ✅ PagedAttention（首创） | ✅ |
| KV 量化 | ✅ INT8/INT4，招牌能力 | ✅（逐步完善） | 部分支持 |
| 权重量化 | ✅ AWQ / W4A16 等 | ✅ AWQ/GPTQ/FP8 等 | ✅ |
| 国产模型支持 | ✅ InternLM/Qwen 等很顺 | ✅ 广 | ✅ 广 |
| 易扩展新模型 | TurboMind 难，PyTorch 后端易 | 较易（Python） | 较易 |
| OpenAI 兼容 API | ✅ | ✅ | ✅ |
| 生态/社区规模 | 中（国内活跃） | 大（事实标准之一） | 大（HF 官方） |
| 一句话印象 | 底层重写，KV量化强，国产模型快 | 通用、生态最广 | HF 全家桶，集成方便 |

> 说明：三者在"连续批处理 + 分页 KV"上已趋同，差异主要在工程实现深度、量化成熟度、模型覆盖与生态。横向跑分高度依赖模型/硬件/负载，没有绝对赢家，以你的实际场景实测为准。

### 何时选 LMDeploy

- 主力模型是 **InternLM、Qwen** 等国产/受支持模型。
- 显存紧张，需要 **KV INT4 + W4A16** 把并发/上下文榨到极限。
- 看重 **延迟与吞吐**，愿意接受"新模型适配较慢"的取舍。
- 想要 OpenAI 兼容服务，快速对接现有客户端。

### 何时别选（或选 vLLM/TGI）

- 要跑的是冷门/最新模型，TurboMind 还没支持，且不想用 PyTorch 后端。
- 团队更熟悉 vLLM 生态、已有大量基于 vLLM 的工具链。

---

## 常见问题

| 问题 | 解答 |
|------|------|
| Persistent Batch 和 vLLM 的 continuous batching 是一回事吗？ | 思想一致（批常驻、按 step 动态进出），叫法不同，实现细节有差异。 |
| KV 量化和权重量化要二选一吗？ | 不用，二者正交，可同时开（如 W4A16 权重 + INT4 KV），收益叠加。 |
| 开了 KV INT4 会不会明显掉点？ | INT8 一般影响很小；INT4 损失更大些，对精度敏感任务建议先评测再上。 |
| 为什么 KV 量化既省显存又提速？ | decode 是显存带宽瓶颈，KV 变小→每步读的字节变少→更快；同时省下的显存能放更大 batch。 |
| TurboMind 和 PyTorch engine 怎么选？ | 追性能且模型受支持选 TurboMind；要适配新模型或改代码选 PyTorch engine。 |
| LMDeploy 一定比 vLLM 快吗？ | 不一定。依赖模型、硬件、负载与版本，务必在自己场景实测。 |
| 多卡怎么用？ | 用张量并行（TP），启动时指定 GPU 数，把大模型摊到多卡。 |
| 具体 CLI 参数 / API 字段去哪查？ | 以官方文档与 `--help` 为准，版本间会变，本文只讲稳定机制。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，从这里找其他主题
- [[llm-inference/vllm/README]] — vLLM 引擎，对照理解连续批处理与 PagedAttention
- [[llm-inference/README]] — LLM 推理总览，引擎选型与全局视角

参考（以官方为准）：

- LMDeploy 仓库：https://github.com/InternLM/lmdeploy
- api_server 文档：https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/serving/api_server.md
- CLI 工具：https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli/utils.py
- 镜像：https://hub.docker.com/r/openmmlab/lmdeploy/tags
