# Text Generation Inference (TGI)

> Hugging Face 官方出品的生产级 LLM 推理服务：Rust 写「外壳」(路由/批调度/Web 服务)，Python 写「内核」(模型前向)，开箱即用地把一个 Transformer 模型变成高吞吐、低延迟的 HTTP/gRPC 推理服务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/vllm/README]] [[llm-inference/README]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0. 一句话锚点 | 30 秒知道 TGI 是什么、为谁服务 | HF 官方 / 推理服务器 |
| 1. 地基/前置 | 没有 TGI 之前，自己起服务有多痛 | 静态批 / 队头阻塞 |
| 2. 整体架构 | Rust router + Python shard 两层是怎么分工的 | webserver / launcher / shard |
| 3. 连续批 (Continuous Batching) | 为什么吞吐能翻几倍 | iteration-level / token 级调度 |
| 4. 张量并行 (Tensor Parallel) | 一张卡放不下时怎么切模型 | TP / NCCL / all-reduce |
| 5. 量化集成 | 怎么把显存砍一半还能跑 | GPTQ / AWQ / bitsandbytes / FP8 |
| 6. Rust+Python 协作 | 两种语言各干什么、为什么这么选 | 安全/并发 vs 生态 |
| 7. 解码与采样 | 流式输出、约束生成、推测解码 | SSE / guidance / speculation |
| 数值例子 | 显存、批大小、吞吐手算一遍 | 7B / 70B / KV cache |
| 对照表 | TGI vs vLLM vs 原生 transformers | 选型 |
| 常见问题 | 踩坑速查 | OOM / 并行 / 量化 |

---

## 0. 一句话锚点

**TGI = 一个把「模型权重 + 一次 forward」包装成「能扛高并发的在线服务」的引擎。**

你给它一个模型(本地路径或 HF Hub 名字)，它给你一个 HTTP 端点。你发请求 `{"inputs": "你好", "parameters": {"max_new_tokens": 100}}`，它流式吐 token 回来。

它要操心的，不是「怎么算一次矩阵乘法」(那是 PyTorch/CUDA 的事)，而是**怎么把成百上千个用户的请求高效地塞进 GPU，让 GPU 一刻也别闲着**。这就是「推理服务器 (inference server)」和「推理引擎 (inference engine)」要解决的核心矛盾。

---

## 1. 地基/前置：没有推理服务器，自己起服务有多痛

先把「为什么需要 TGI」讲透。假设你有个 7B 模型，想做成聊天 API。最朴素的写法：

```python
# 朴素版：一个请求进来，跑一次 generate
@app.post("/generate")
def generate(req):
    return model.generate(req.inputs, max_new_tokens=req.max_tokens)
```

这套东西在生产里会死得很惨，原因有三个层层递进的问题：

### 问题 ①：GPU 利用率极低 (一次只服务一个人)

LLM 解码是**自回归**的：生成第 $t$ 个 token 要等第 $t-1$ 个出来。每生成 1 个 token，就要把整个模型权重从显存搬一遍到计算单元。对 7B 模型 (FP16)，权重 14 GB，而生成 1 个 token 的实际计算量极小 —— 这是典型的 **memory-bound (访存受限)**。

```
单请求解码：GPU 算力利用率示意
算力 ████░░░░░░░░░░░░░░░░░░  ← 只用了 ~5%
        ↑ 大量算力闲置，因为一次只算 batch=1
```

解决办法直觉上很简单：**攒一批请求一起算 (batching)**。权重只搬一次，却同时服务 32 个用户，吞吐近乎线性放大。

### 问题 ②：静态批 (static batching) 的队头阻塞

但「攒批」的朴素实现——**静态批**——有致命缺陷：

```
静态批：一批 4 个请求，必须等最慢的那个结束
请求A: ████ (4 token 就停了)        [停] 空转........... 
请求B: ████████████████ (16 token)  ←━━━ 整批被它拖住
请求C: ████ (4 token)               [停] 空转...........
请求D: ██████ (6 token)             [停] 空转...........
        └──── A/C/D 早就该走了，却被迫陪 B 跑完 ────┘
```

不同请求生成长度差异巨大(有人要 4 个 token，有人要 2000 个)。静态批里，**短请求被长请求绑架**，已经生成完的 slot 在空转，GPU 又浪费了。同时新来的请求只能干等当前批结束，**延迟**也爆炸。

### 问题 ③：KV cache 的显存管理

每个请求在解码时要缓存历史 token 的 Key/Value 向量(KV cache)，否则每步都要重算全部历史，复杂度从 $O(n)$ 退化成 $O(n^2)$。但 KV cache 会随序列变长而增长，长度又不可预测。朴素实现要么预留最大长度(浪费)，要么动态分配(碎片化)。

**TGI 就是把这三个问题在一个工程里系统性解决掉的产物：**

| 痛点 | TGI 的武器 |
|------|-----------|
| ① 单请求利用率低 | 批处理 |
| ② 静态批队头阻塞 | **连续批 (Continuous Batching)** — 第 3 节 |
| ③ KV cache 浪费 | PagedAttention / FlashAttention 类内核 + 显存预估 |
| 模型放不下单卡 | **张量并行 (Tensor Parallelism)** — 第 4 节 |
| 显存仍然不够 | **量化 (Quantization)** — 第 5 节 |

---

## 2. 整体架构：Rust 外壳 + Python 内核

TGI 是一个**两层 + 一个启动器**的结构。理解这张图，就理解了 TGI 的 80%。

```
                            HTTP / gRPC 请求 (成千上万并发)
                                      │
        ┌─────────────────────────────▼──────────────────────────────┐
        │                  Router / Webserver  (Rust)                 │
        │  • Axum 异步 HTTP 服务，OpenAI 兼容 + 原生接口               │
        │  • 请求队列 (queue) + Token 校验 + 参数验证                  │
        │  • Continuous Batching 调度器：决定每一步把谁放进 batch     │
        │  • 流式 SSE 返回、指标暴露 (Prometheus)                     │
        └─────────────────────────────┬──────────────────────────────┘
                                      │ gRPC (Protobuf)
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
        ┌─────────▼────────┐ ┌────────▼─────────┐ ┌───────▼──────────┐
        │  Shard 0 (Python) │ │ Shard 1 (Python) │ │ Shard 2 (Python) │   ← 张量并行
        │  模型的 1/N 切片   │ │  模型的 1/N 切片  │ │  模型的 1/N 切片  │
        │  PyTorch forward  │ │  PyTorch forward │ │  PyTorch forward │
        │  Flash/Paged 内核 │ │  ...             │ │  ...             │
        └─────────┬─────────┘ └────────┬─────────┘ └───────┬──────────┘
                  └─────── NCCL all-reduce 同步 ───────────┘
                              GPU0      GPU1      GPU2

        ───────────────────────────────────────────────────────────
        launcher (Rust)：把上面这一切拉起来、管进程、做健康检查
```

三个角色：

- **launcher**：命令行入口 (`text-generation-launcher`)。它读环境变量/参数，下载/分片模型权重，按张量并行度 fork 出 N 个 Python shard 子进程，再起 Rust router，把它们用 gRPC 连起来，并监控进程存活。
- **router (webserver)**：**Rust 写的服务大脑**。所有网络 I/O、并发、排队、批调度都在这里。它**不碰模型计算**，只决定「这一步，把哪些请求、哪些 token 打包成一个 batch，发给 shard」。
- **shard (model server)**：**Python 写的计算肌肉**。每个 shard 持有模型的一部分(张量并行切片)，收到 router 的 batch 后调用 PyTorch + 优化内核(FlashAttention、PagedAttention、量化 kernel)做真正的 forward，返回每个序列的 next-token logits。

> 为什么这么分？Rust 擅长无 GC、内存安全的高并发网络服务(成千上万连接不崩、延迟稳定)；Python 擅长 PyTorch/CUDA 生态和模型实现的快速迭代。**用对的语言干对的活**——第 6 节细讲。

---

## 3. 连续批 (Continuous Batching)：吞吐翻倍的核心

这是 TGI(以及 vLLM)相对朴素服务最关键的一招，也叫 **iteration-level scheduling / in-flight batching / token 级批处理**。

核心思想一句话：**不在「请求」粒度上批，而在「每生成一个 token 的迭代步」粒度上批。每跑完一步，立刻把已完成的请求踢出去、把排队的新请求塞进来。**

对比静态批，看时间轴：

```
静态批 (前面那张图)：B 没跑完，A/C/D 的 slot 全空转，新请求干等。

连续批：每一步动态重组 batch
时间步 →  t1   t2   t3   t4   t5   t6   t7   t8
A       ●    ●    ●    ✓                          A 4步完成→立刻退出
B       ●    ●    ●    ●    ●    ●    ●    ●       B 继续跑
C       ●    ●    ●    ✓                          C 完成→退出
D       ●    ●    ●    ●    ●    ✓                
E            (排队)        ●    ●    ●    ●    ●   ← A退出腾出slot，E立刻补进来!
F                          ●    ●    ●    ●    ●   ← C退出，F补进来!
        └ batch 始终被填满，GPU 不空转，新请求秒级进场 ┘
```

调度的两类操作 (在 router 里)：

- **Prefill (预填充)**：一个新请求进来，先把它的整个 prompt 一次性并行编码，建立初始 KV cache。这步**计算密集 (compute-bound)**，是「大块矩阵乘」。
- **Decode (解码)**：所有在跑的请求，每步各生成 1 个 token。这步**访存密集 (memory-bound)**，靠大 batch 摊薄权重搬运成本。

TGI 的调度器要不断权衡：什么时候插入新请求的 prefill(会短暂打断正在进行的 decode)，什么时候继续 decode。它用 `max_batch_total_tokens`、`max_waiting_tokens` 等预算约束，确保不会因为塞太多请求把 KV cache 撑爆 OOM。

> 直觉收益：实测连续批相对静态批，吞吐常有 **2~20 倍**提升(取决于请求长度分布越参差不齐、收益越大)。具体倍数以你的负载实测为准。

---

## 4. 张量并行 (Tensor Parallelism)：单卡放不下怎么办

7B 模型 FP16 约 14 GB，单张 24GB 卡能放。但 70B 约 140 GB，**任何单卡都放不下**。张量并行 (TP) 把**每一层的权重矩阵横/竖切开**，分到多张卡上，让它们**协同算一层**。

以 Transformer 里的 MLP 为例(两个线性层 $Y = \text{GeLU}(XA)B$)：

```
张量并行：把权重矩阵 A 按列切，B 按行切，分到 2 张卡
                 X (输入，每卡都有完整副本)
                 │
        ┌────────┴────────┐
        ▼                 ▼
  GPU0: X·A₁         GPU1: X·A₂        ← A 按列切成 [A₁ | A₂]
        │                 │
   GeLU(X·A₁)        GeLU(X·A₂)         ← 逐元素，无需通信
        │                 │
   GeLU(X·A₁)·B₁     GeLU(X·A₂)·B₂     ← B 按行切成 [B₁; B₂]
        └────────┬────────┘
                 ▼
            all-reduce 求和  ← NCCL 通信！把两卡部分和加起来 = 完整 Y
                 │
                 ▼  Y (完整输出)
```

关键点：

- **切权重 ≠ 切数据**。每张卡都看到完整的输入 token，但只持有权重的一部分，算出**部分结果**，再用 **NCCL all-reduce** 把部分结果合并。注意力头(attention heads)同理，按 head 维度切分。
- **每层要通信一次** (all-reduce)。所以 TP 对**卡间带宽极其敏感**——理想是同机 NVLink。跨机器 TP(走 PCIe/网络)通常会成为瓶颈。
- TGI 通过启动参数(常见为 `--num-shard` 或环境变量 `NUM_SHARD`，**以官方文档为准**)指定切几片，launcher 就 fork 几个 shard，每个 shard 绑一张 GPU。

数据流总览：

```
请求 → Router(Rust) ──gRPC──> [Shard0│Shard1│Shard2│Shard3]
                                  └─ 每层 forward 后 NCCL all-reduce 同步 ─┘
                              任一 shard 算完返回 logits 给 Router 采样
```

> TP 解决的是「**模型装不下单卡**」(显存墙)。它不是为了线性加速吞吐——通信开销让加速比 < 卡数。如果模型本来就装得下，多卡更应优先考虑**数据并行**(起多个独立副本)而非 TP。

---

## 5. 量化集成：把显存再砍一半

即使有了 TP，显存仍是硬约束。**量化 (quantization)** 用更少的比特表示权重(和有时的激活/KV cache)，直接减小显存占用、加快访存(memory-bound 场景同时提速)。

```
权重精度 vs 显存 (以 7B 模型 70 亿参数为例)
FP32  ████████████████  32-bit  ≈ 28 GB
FP16  ████████          16-bit  ≈ 14 GB   ← 常见基线
INT8  ████              8-bit   ≈  7 GB
INT4  ██                4-bit   ≈ 3.5 GB  ← 量化主战场
```

显存(权重部分)的估算公式：

$$\text{显存}_{\text{权重}} \approx N_{\text{params}} \times \frac{\text{bits}}{8} \text{ 字节}$$

TGI 集成了主流量化方案，按「训练后离线量化 (PTQ)」和「加载时量化」两类理解：

| 方案 | 比特 | 思路一句话 | 特点 |
|------|------|-----------|------|
| **GPTQ** | 4-bit | 逐层用 Hessian 信息做最优权重取整，需校准数据 | 精度损失小、需预量化产物 |
| **AWQ** | 4-bit | 保护「重要通道」不被量化，激活感知 | 速度快、精度好 |
| **bitsandbytes** | 8/4-bit | 加载时即时量化，无需预处理 | 开箱即用、相对慢 |
| **EETQ** | 8-bit | 轻量 INT8，加载快 | 简单场景 |
| **FP8** | 8-bit 浮点 | 新硬件(Hopper+)原生支持，含 KV cache | 精度/速度俱佳，需新卡 |

要点：

- **GPTQ/AWQ 是离线量化**——你加载的是已经量化好的权重文件(常带 `-GPTQ`/`-AWQ` 后缀的 HF 模型)，TGI 用对应 kernel 解包计算。
- **bitsandbytes 是在线量化**——加载原始 FP16 权重时即时压缩，方便但启动慢、推理也通常慢于 GPTQ/AWQ 的专用 kernel。
- 量化是**有损**的：4-bit 在多数任务上精度损失可接受，但对数学/代码等敏感任务要实测。**省显存 vs 保精度**是核心权衡。
- 量化还能用于 **KV cache**(FP8 KV cache)，进一步省显存、让批更大。

> 具体支持的量化类型/参数随版本演进，**以官方文档为准**；此处讲的是「为什么/怎么权衡」这层稳定知识。

---

## 6. Rust + Python 协作：为什么这么选

TGI 最有辨识度的设计，就是**双语言**。把这层讲透：

```
┌──────────────────── Rust 域 ────────────────────┐   ┌──── Python 域 ────┐
│  • 网络/HTTP/gRPC 服务 (Axum, Tonic)            │   │  • 模型结构定义     │
│  • 高并发连接管理、零成本异步 (Tokio)            │   │  • PyTorch forward  │
│  • 请求队列 + 连续批调度器(核心调度逻辑)         │◄─►│  • Flash/Paged 内核 │
│  • 内存安全、无 GC 抖动、延迟稳定                │gRPC│  • 量化 kernel      │
│  • 参数校验、限流、指标                          │   │  • 权重加载/分片     │
└─────────────────────────────────────────────────┘   └────────────────────┘
        擅长：高并发、低延迟、稳定                       擅长：ML 生态、快速迭代
```

为什么不全用 Python？

- Python 有 **GIL**，多线程并发网络服务受限；高 QPS 下 GC 暂停会造成**尾延迟抖动**。在线服务最怕「偶尔卡一下」。
- Rust 编译期保证内存安全、无 GC，配合 Tokio 异步运行时能用极少线程扛海量连接，**P99 延迟可控**。

为什么不全用 Rust？

- 模型实现、PyTorch、CUDA 自定义内核、量化算法的整个生态都在 Python/C++。用 Rust 重写模型层 = 跟整个 ML 社区为敌，迭代速度归零。

所以 TGI 的取舍是：**用 Rust 守住「服务质量(并发/延迟/稳定)」，用 Python 接住「模型生态(快速支持新模型/新内核)」，中间用 gRPC 解耦。** 这正是「在正确的地方用正确的工具」的工程典范。

---

## 7. 解码、采样与高级特性

router 拿到 shard 返回的 logits 后，负责**采样**和**输出控制**：

- **采样参数**：`temperature`(温度，越高越随机)、`top_k`(只在概率最高的 k 个里采)、`top_p`(核采样，累积概率到 p)、`repetition_penalty`(抑制重复)。
- **流式输出 (SSE)**：每生成一个 token 就通过 Server-Sent Events 推给客户端，用户「打字机」式看到结果，**首 token 延迟 (TTFT)** 是关键体验指标。
- **结构化/约束生成 (Guidance)**：可强制输出符合 JSON Schema/正则的文本(对 logits 做掩码)，让 LLM 输出可被程序解析。
- **推测解码 (Speculative Decoding)**：用小模型/草稿(如 Medusa 头、n-gram)先猜几个 token，大模型一次性并行验证，验证通过就「白赚」加速。本质是把多步 memory-bound decode 折叠成更少的 compute 步。

```
两个核心延迟指标：
请求到达 ──[排队+Prefill]──► 第1个token ──[每token decode]──► 完成
          └──── TTFT ────┘            └─ TPOT(每token时间) ─┘
TTFT 受 prefill 和排队影响；TPOT 受 batch 大小和模型规模影响。
```

---

## 数值例子：显存与批大小手算一遍

把抽象概念落到具体数字。**场景：单张 A100-80GB 跑 Llama-2-7B (FP16)，估算能放多大批。**

**第 1 步：权重显存**
$$7\text{B} \times 2\,\text{字节/参数(FP16)} = 14\,\text{GB}$$

**第 2 步：剩余给 KV cache 的显存**
$$80 - 14 - 4_{\text{(框架/激活余量)}} \approx 62\,\text{GB}$$

**第 3 步：单个 token 的 KV cache 大小**
Llama-2-7B：层数 $L=32$，注意力维度 $d=4096$，K 和 V 各一份，FP16：
$$\text{每 token KV} = 2_{(K,V)} \times L \times d \times 2_{\text{字节}} = 2 \times 32 \times 4096 \times 2 = 524288\,\text{字节} \approx 0.5\,\text{MB}$$

**第 4 步：能缓存多少 token**
$$\frac{62\,\text{GB}}{0.5\,\text{MB/token}} = \frac{62 \times 1024\,\text{MB}}{0.5\,\text{MB}} \approx 127000\,\text{token}$$

**第 5 步：折算成并发**
若平均每个请求上下文 2048 token：
$$\frac{127000}{2048} \approx 62 \text{ 个并发请求}$$

> 结论：一张 80GB 卡，7B/FP16，理论上可同时维持约 **60 路** 2K 上下文的并发(实际因碎片、prefill 峰值会打折)。

**对比量化的威力**：若改用 **4-bit GPTQ**，权重从 14GB 降到约 **3.5GB**，KV cache 可用显存升到约 72.5GB，并发能力进一步提升——这就是量化在服务场景的直接价值。

**张量并行的场景**：换成 70B/FP16，权重 140GB，单卡放不下。用 **TP=4** 切到 4 张 A100-80GB，每卡承担 35GB 权重 + 自己那份 KV cache 切片，于是「装得下」了；代价是每层一次 NCCL all-reduce 通信。

---

## 对照表：TGI vs vLLM vs 原生 transformers

| 维度 | 原生 `transformers.generate` | **TGI** | vLLM |
|------|------------------------------|---------|------|
| 定位 | 库 (写脚本用) | **HF 官方生产推理服务** | 高性能推理引擎/服务 |
| 连续批 | ❌(默认静态/单条) | ✅ | ✅ |
| KV cache 管理 | 朴素 | FlashAttention/Paged 类 | **PagedAttention(原创)** |
| 张量并行 | 需手动/accelerate | ✅ 内置 | ✅ 内置 |
| 量化 | bitsandbytes 等 | GPTQ/AWQ/bnb/FP8 等 | GPTQ/AWQ/FP8 等 |
| 架构 | 纯 Python | **Rust router + Python shard** | 主要 Python(+CUDA) |
| 接口 | 无(自己包) | HTTP/gRPC, OpenAI 兼容 | HTTP, OpenAI 兼容 |
| HF 生态贴合 | ★★★ 原生 | ★★★ 官方一家 | ★★ 良好 |
| 何时选 | 离线批/研究/小规模 | 要稳、要 HF 生态、官方支持 | 极致吞吐、PagedAttention 首发 |

**何时选 TGI**(本质 takeaway)：

- 你深度在 **Hugging Face 生态**里(模型都在 Hub、用 HF 工具链)，想要官方维护、与新模型同步快的生产服务。
- 你看重**服务稳定性与延迟可控**(Rust 服务层的价值)、要 OpenAI 兼容接口、要 Prometheus 指标、要部署在 K8s。
- 你需要开箱即用的**连续批 + TP + 量化**组合，不想自己拼轮子。

**何时考虑别的**：追求某些负载下极致吞吐、或要用 PagedAttention 的特定特性，可对比 vLLM([[llm-inference/vllm/README]]);纯离线批量打分、研究实验，原生 transformers 更轻。

> TGI 与 vLLM 在能力上高度趋同(都做连续批/TP/量化)，差异更多在**生态贴合度、架构哲学、特定优化**上。生产选型务必用**自己的真实负载**做基准测试，别只看 benchmark 数字。

---

## 常见问题

| 问题 | 原因 / 解法 |
|------|------------|
| 启动就 OOM | 权重 + KV cache 预算超显存。降批预算(`max_batch_total_tokens`)、上量化、或加 TP 分片。**以官方参数名为准**。 |
| 多卡没提速反而更慢 | TP 跨机/走 PCIe，all-reduce 通信成瓶颈。优先同机 NVLink；模型装得下时用多副本数据并行而非 TP。 |
| 量化后精度掉得厉害 | 4-bit 对数学/代码敏感。换 AWQ、升 8-bit、或对关键任务保 FP16。务必实测。 |
| 首 token 很慢 (TTFT 高) | 长 prompt 的 prefill 重 + 排队。缩短上下文、调度优先级、或加副本分流。 |
| 吞吐上不去 | batch 没填满。检查请求并发是否够、KV cache 预算是否过保守、是否被某些超长请求占满。 |
| 长序列被截断 | 受 `max_input_length`/`max_total_tokens` 限制。按需调高(代价是显存)。 |
| 想要 JSON 输出 | 用约束生成 (Guidance/Grammar)，对 logits 掩码强制结构。 |
| 换新模型不支持 | 模型结构需在 Python shard 层适配。看 TGI 版本是否已支持该架构。 |

---

## 🔗 跳转链接

- 📍 知识地图：[[00-知识地图]]
- 同类引擎对比：[[llm-inference/vllm/README]]
- 推理总览：[[llm-inference/README]]
- 官方仓库：https://github.com/huggingface/text-generation-inference (具体版本号、CLI 参数、API 签名以官方文档为准)
