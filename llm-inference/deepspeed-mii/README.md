# DeepSpeed-MII

> DeepSpeed-MII（Model Implementations for Inference）是微软 DeepSpeed 团队推出的**低延迟、高吞吐 LLM 推理服务化框架**：把底层的 DeepSpeed-Inference 内核 + 连续批处理 + Dynamic SplitFuse 打包成"几行代码即起服务"的部署层。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/DeepSpeed-Inference]] [[llm-optimizer/SplitFuse]] [[ai-framework/deepspeed/README]]

> 关键定位：**DeepSpeed-FastGen = DeepSpeed-MII（服务/调度层） + DeepSpeed-Inference（高性能内核层）的协同组合**。MII 负责"怎么把模型变成一个能扛并发的服务"，DeepSpeed-Inference 负责"单次前向算得多快"。

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | MII / FastGen / 服务化 |
| 1 | 地基：朴素推理服务为什么慢 | 静态 batch / prefill 卡顿 |
| 2 | MII 在技术栈中的位置 | 三层架构 |
| 3 | 两种部署模式 | 非持久化 / 持久化 |
| 4 | 持久化部署的架构 | gRPC / 负载均衡 / 多副本 |
| 5 | 连续批处理（Continuous Batching） | iteration-level 调度 |
| 6 | **Dynamic SplitFuse**（核心创新） | 拆长 prompt、拼短 token |
| 7 | 与 DeepSpeed-Inference 的关系 | 内核 vs 服务 |
| 8 | KV Cache 与显存管理 | Blocked KV / token 预算 |
| 9 | 请求生命周期（端到端） | 调用链 ASCII |
| 10 | 关键配置项与权衡 | tensor_parallel / replica |
| 11 | 与 vLLM / TGI / TRT-LLM 对比 | 选型 |
| 12 | 数值示例：SplitFuse 收益 | 算一笔账 |
| - | 配置示例 / 常见问题 / 跳转 | 速查 |

---

## 0. 一句话锚点

> **MII = 把 DeepSpeed-Inference 的"快内核"封装成"可持续在线服务"，并用 Continuous Batching + Dynamic SplitFuse 解决"长 prompt 阻塞短请求"和"GPU 利用率忽高忽低"两大痛点。**

记住三个名字的分工：**DeepSpeed-Inference** = 单次前向的高性能内核（融合算子/TP/kernel injection）；**DeepSpeed-MII** = 服务化封装+部署+调度（pipeline/持久化/负载均衡）；**DeepSpeed-FastGen** = 上面两者协同后对外宣传的"系统名"，主打高吞吐文本生成。

---

## 1. 地基：朴素推理服务为什么慢

要理解 MII，先看一个"手写 FastAPI + HuggingFace `model.generate()`"的朴素服务有哪些病。

### 1.1 自回归生成的两个阶段

LLM 生成分两个性质完全不同的阶段：

```
Prompt:  "请介绍一下杭州"   (长度 P 个 token，比如 P=512)

┌──────────────── Prefill 阶段 ────────────────┐ ┌─── Decode 阶段 ───┐
│ 一次性把 512 个 token 全部喂进去               │ │ 每步只算 1 个新 token │
│ 计算量 ∝ P（大、算力密集 / compute-bound）     │ │ 但要读全部 KV Cache  │
│ 产出：首个 token + 全部 KV Cache               │ │ 访存密集 / memory-bound│
└──────────────────────────────────────────────┘ └───────────────────┘
        "重而短"（一次）                              "轻而多"（生成 N 次）
```

**Prefill** 算力密集（矩阵乘大）但只发生一次；**Decode** 访存密集（每步搬运整个 KV Cache）但要重复 N 次（N=输出长度）。两个阶段的"理想 batch 形状"截然不同，这就是后面 SplitFuse 要解决的根源。

### 1.2 静态批处理（Static Batching）的浪费

```
请求 A: |■■■■■■■■■■| 10 token 才结束     时间轴 →
请求 B: |■■|         2 个就结束 → 等 A！   step1: A B C ✅
请求 C: |■■■■■|      5 个                step3: A _ C ❌ B 走了 slot 空转
                                        step10:A _ _ ❌ 浪费一大片
```

静态批处理"凑齐一批一起进、一起出"，**短请求被长请求拖死**，GPU 大量时间在算"padding/空 slot"。这是 vLLM/TGI/MII 共同要解决的第一痛点 → **连续批处理**（见第 5 节）。

### 1.3 长 prompt 阻塞（Prefill 卡顿）

```
一个 4096-token 的长 prompt 进来做 prefill，
这一个 forward 就占满 GPU 数十毫秒，
期间其它请求的 decode 全部"卡住"，
→ 在线服务的 P99 延迟剧烈抖动（长尾）。
```

这是 MII **Dynamic SplitFuse** 专门解决的第二痛点（见第 6 节）。

> 一句话：MII 的全部设计动机，都是为了消灭"空转 slot"和"prefill 卡顿"。

---

## 2. MII 在技术栈中的位置

```
┌──────────────────── 用户 / 应用 (HTTP / Python) ────────────────────┐
└──────────────────────────────────┬─────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────── 本文主角 ┐
│ DeepSpeed-MII（服务/调度层）                                            │
│  pipeline/serve API · 持久化(gRPC+负载均衡) · Continuous Batching       │
│  · Dynamic SplitFuse token 预算分配                                    │
└──────────────────────────────────┬─────────────────────────────────┘ 调用
┌──────────────────────────────────▼─────────────────────────────────┐
│ DeepSpeed-Inference（高性能内核层）                                     │
│  kernel injection/融合算子 · 张量并行(TP) · Blocked KV Cache 内核        │
└──────────────────────────────────┬─────────────────────────────────┘
┌──────────────────────────────────▼─────────────────────────────────┐
│ PyTorch / CUDA / GPU 硬件                                            │
└─────────────────────────────────────────────────────────────────────┘
```

**记忆点**：MII 是"上层"，DeepSpeed-Inference 是"下层内核"。你写 `mii.serve(...)`，它内部去拉起 DeepSpeed-Inference 的优化模型。

---

## 3. 两种部署模式

MII 提供两种粒度的使用方式，对应"离线跑一把"和"在线常驻服务"。

```
┌───────────────────────────┐      ┌───────────────────────────┐
│  非持久化 (Non-persistent)  │      │   持久化 (Persistent)        │
│  pipeline                  │      │   serve / client            │
├───────────────────────────┤      ├───────────────────────────┤
│ 进程内创建模型 → 直接生成     │      │ 后台常驻 gRPC server        │
│ 脚本结束模型即销毁           │      │ 多客户端可反复连接           │
│ 适合：批量离线推理、调试      │      │ 适合：线上 API、长期服务      │
│ 形态：一个 Python 函数       │      │ 形态：server + 多个 client   │
└───────────────────────────┘      └───────────────────────────┘
```

伪代码（API 名称以官方文档/源码为准，思路如下）：

```python
import mii

# ① 非持久化：用完即走
pipe = mii.pipeline("meta-llama/Llama-2-7b-hf")
out = pipe(["请介绍一下杭州", "写一首五言绝句"], max_new_tokens=128)

# ② 持久化：起一个常驻服务（一次）
client = mii.serve("meta-llama/Llama-2-7b-hf")
# 之后任意进程都可连接同一个服务
client = mii.client("meta-llama/Llama-2-7b-hf")
resp = client.generate(["请介绍一下杭州"], max_new_tokens=128)
# 不用了再关掉
client.terminate_server()
```

> 选择原则：**离线评测/数据生成 → `pipeline`；线上要扛并发、长期在线 → `serve`（持久化）**。

---

## 4. 持久化部署的架构

持久化部署是 MII"服务化"的核心。它把模型推理变成一个**前后端解耦的 gRPC 服务**，并支持"多副本横向扩展"。

```
  clients(多进程/机器)  c   c   c
                        └───┼───┘  gRPC 请求
                            ▼
        ┌──────────── Load Balancer (负载均衡) ───────────┐  ← 分发到空闲副本
        └───┬───────────────┬───────────────┬───────────┘
            ▼               ▼               ▼
        ┌───────┐       ┌───────┐       ┌───────┐   ← 每个副本=独立模型实例
        │Replica│#0     │Replica│#1     │Replica│#2    （deployment 的多副本）
        └───┬───┘       └───────┘       └───────┘
      ┌─────┴─────┐                                  每副本内部可再做张量并行(TP)
      │ GPU0  GPU1│  ← 一个副本横跨多张卡              ...           ...
      └───────────┘
```

四个层次概念务必分清：

| 层级 | 含义 | 由谁决定 |
|------|------|---------|
| Deployment | 一个被服务的模型部署（带名字） | 你 `serve` 时命名 |
| Replica（副本） | 同一模型的多个独立实例，用于扩吞吐 | `replica_num` |
| Tensor Parallel（张量并行） | 单个副本内把权重切到多卡 | `tensor_parallel` |
| gRPC server | 进程间通信载体，client 与之对话 | MII 自动管理 |

> 横向扩展（加副本）= 提吞吐 / 抗并发；纵向扩展（加 TP 度）= 单副本能放下更大模型 / 降单请求延迟。两者正交。

---

## 5. 连续批处理（Continuous Batching）

MII 用 **iteration-level（迭代级）调度**取代静态批处理：每生成一步就重新组 batch，谁结束谁立刻离场、新请求随时插入。

```
静态批处理（旧）              连续批处理（MII / FastGen）
step1: [A B C]              step1: [A B C]
step2: [A B C]              step2: [A B C]
step3: [A _ C]  B 空转       step3: [A   C D]   B 走、D 立刻补位 ✅
step4: [A _ _]  浪费         step4: [A   C D]
                            step5: [  E C D]   A 走、E 补位 ✅
GPU 利用率：忽高忽低           GPU 利用率：持续饱和
```

**核心思想**：调度单位从"整个请求"细化到"一次迭代"，让 GPU 永远在算有效 token、没有空 slot。这一步解决了"短请求被长请求拖死"，但没解决"长 prompt 的 prefill 一来就卡住所有人"——这正是 SplitFuse 登场的地方。

---

## 6. Dynamic SplitFuse（核心创新）⭐

这是 MII / FastGen 相对于普通连续批处理的**关键差异化技术**。一句话：

> **把每一次 forward 都凑成"大小恒定、形状均匀"的一批 token，做法是：把超长 prompt 的 prefill 切成几片（Split），并和很多请求的 decode token 拼在一起（Fuse）一起算。**

### 6.1 问题回顾

- Prefill：token 多、算力密集 → 一来就独占 GPU；Decode：每请求只 1 token、访存密集算力闲置。
- 普通连续批处理：prefill 批与 decode 批"形状忽大忽小"，GPU 时而 compute-bound 时而 memory-bound，跑不到峰值吞吐。

### 6.2 SplitFuse 怎么做

设定一个**每步固定的 token 预算** $B$（如 $B=2048$）。每个 forward step：**Split（拆）** 把长 prompt 的 prefill 切成若干 chunk、每步只处理一部分，**不让一个长 prompt 独占整步**；**Fuse（融）** 用其余预算装填正在 decode 的请求 token，**把"算力密集的 prefill 碎片"和"访存密集的 decode token"混在一起**互相填补 GPU 空闲。

```
固定 token 预算 B = 2048 / step

step k:   ┌──────── prefill chunk (长prompt的一片, 1536) ────────┬─ decode 512 个请求各1token ─┐
          └──────────────────────────────────────────────────────┴────────────────────────────┘
                       compute-bound 的部分          +         memory-bound 的部分
                                = 让 GPU 算力和带宽都不闲

step k+1: ┌──── 同一长prompt的下一片 (1536) ────┬──── 512 个 decode token ────┐
          └─────────────────────────────────────┴──────────────────────────────┘
```

### 6.3 为什么有效（两个收益）

```
收益①  消灭 prefill 长尾：
   长 prompt 不再"一口吞"，而是切片摊到多步，
   每步耗时被钳制在 ~恒定值 → 在线请求的 P99 延迟稳定。

收益②  逼近 GPU 峰值吞吐：
   每步 token 数恒定、prefill(算力)与decode(访存)混合，
   GPU 始终在"算力与带宽都满载"的甜点区运行。
```

直觉对比：无 SplitFuse 时，一块 4096 的 prefill 独占一整步、时长爆炸、别人全卡，且 decode 步又太轻让算力空闲；有 SplitFuse 后每步形状恒定、算力/带宽持续饱和。

> 与 vLLM 的对照：vLLM 经典做法是 prefill 与 decode 倾向分离调度（后来也引入了 chunked-prefill）；MII 从一开始就主张"split 长 prefill + fuse 进 decode 流"，把形状抹平。设计哲学略有差异，结论都指向"让 GPU 别空转"。

详细推导与变体见 [[llm-optimizer/SplitFuse]]。

---

## 7. 与 DeepSpeed-Inference 的关系

这是最容易混淆的点，单独讲清。

```
mii.serve(...) → [DeepSpeed-MII] 解析配置(副本/TP/dtype)、起 gRPC+负载均衡、初始化
              → (init_inference 风格封装) → [DeepSpeed-Inference] kernel injection、TP 切分、Blocked KV Cache
```

| 维度 | DeepSpeed-Inference | DeepSpeed-MII |
|------|--------------------|----------------|
| 角色 | 高性能**推理内核** | **服务化 / 部署 / 调度**层 |
| 关注 | 单次 forward 多快 | 多请求并发下吞吐与延迟 |
| 提供 | 融合算子、TP、KV 内核 | pipeline/serve、持久化、负载均衡、SplitFuse |
| 类比 | 发动机 | 整车 + 调度系统 |
| 单独可用 | 可（手动 `init_inference`） | 依赖前者作为底座 |

> **FastGen** 这个名字 = MII（调度/服务）+ DeepSpeed-Inference（内核）协同后的"高吞吐文本生成系统"对外称呼。本目录开头那句"DeepSpeed-FastGen 是 DeepSpeed-MII 和 DeepSpeed-Inference 的协同组合"就是这个意思。

内核细节见 [[llm-inference/DeepSpeed-Inference]]。

---

## 8. KV Cache 与显存管理

连续批处理 + SplitFuse 要成立，底层必须有**高效、可动态增删的 KV Cache**。

```
朴素 KV：每请求预留 max_len 连续显存 → 大量预留未用 → 浪费、装不下几个请求
分块 KV (Blocked)：把 KV 切成固定大小的 block，按需分配
                  ↓
请求 A: [blk3][blk7][blk1]   ← 逻辑连续、物理分散
请求 B: [blk2][blk9]
空闲池: [blk0][blk4][blk5][blk6][blk8]...
```

KV Cache 占用 $\approx 2 \times d_{head} \times H \times n_{layer} \times \text{dtype}$（per token；2 表示 K、V）。分块管理让"一台卡能同时容纳的并发请求数"大幅提升，是 Continuous Batching 能"随时插入新请求"的物理前提。当 token 预算或显存吃紧时，调度器按策略安排请求进入与（必要时）排队/抢占（具体淘汰/抢占策略以源码为准）。

---

## 9. 请求生命周期（端到端调用链）

把前面所有概念串成一条"一个请求从发出到拿到结果"的链路：

```
[client] generate(...) ─gRPC→ [Load Balancer] 选最空闲 Replica
   → [Replica 调度器] tokenize、纳入"在飞请求"集合
   → 进入迭代级循环，每个 forward step：
       · 用固定 token 预算 B 组本步 batch
       · Dynamic SplitFuse：长 prompt 切片 + decode 融合
       · 调 DeepSpeed-Inference 内核前向 + 读写 Blocked KV Cache
       · 每个在飞请求产出 1 个新 token
       · 命中 EOS / 达 max_new_tokens → 离场
   → 循环至该请求结束 → detokenize（流式则逐 token 回传）─gRPC→ [client]
```

---

## 10. 关键配置项与权衡

> 以下为概念性参数（确切字段名/默认值以官方文档/源码为准），重点理解**含义与权衡**。

| 配置（语义） | 含义 | 调大 ⬆ | 调小 ⬇ |
|------------|------|--------|--------|
| `tensor_parallel` | 单副本切到几张卡 | 能放更大模型 / 单请求更快；但通信开销增、卡利用率边际下降 | 省卡、通信少；大模型放不下 |
| `replica_num` | 同模型副本数 | 吞吐/并发线性提升 | 省显存；并发能力弱 |
| `max_length` / 上下文上限 | 单请求最大序列长度 | 支持长文本 | 每请求 KV 占用大、并发数下降 |
| token 预算（SplitFuse 每步） | 每个 forward step 处理的 token 总数 | 单步吞吐高、延迟略增 | 延迟低、单步吞吐降 |
| `dtype`（fp16/bf16/量化） | 计算/存储精度 | 量化更省显存、更快、精度略损 | 精度高、显存吞吐成本高 |
| `max_new_tokens` | 单请求最多生成多少 | 输出更长、占用 slot 更久 | 释放快、整体吞吐高 |

**权衡心法**：抗并发提 QPS → 先加 `replica_num`；模型太大放不下单卡 → 加 `tensor_parallel`；P99 抖 → 关注 token 预算（SplitFuse 让其更稳）；显存吃紧 → 降 `max_length` / 上量化 / 减并发。

---

## 11. 与 vLLM / TGI / TensorRT-LLM 对比

| 框架 | 核心卖点 | 批处理 | 长 prefill 处理 | 内核来源 | 适用 |
|------|---------|--------|----------------|---------|------|
| **DeepSpeed-MII / FastGen** | SplitFuse + DeepSpeed 生态、易部署 | 连续批处理 | **Dynamic SplitFuse**（切片+融合） | DeepSpeed-Inference | 已用 DeepSpeed 训练、要稳定低 P99 |
| vLLM | PagedAttention、社区生态最广 | 连续批处理 | chunked-prefill（后期引入） | 自研 + 集成 | 通用首选、模型覆盖广 |
| HF TGI | HuggingFace 生态、生产工具链全 | 连续批处理 | 支持 | 多后端 | 重 HF 生态、要开箱即用服务 |
| TensorRT-LLM | NVIDIA 极致内核优化 | in-flight batching | 支持 | TRT 编译内核 | 纯 NVIDIA、要榨干硬件 |

> 选 MII 的典型理由：①你的训练栈本来就是 DeepSpeed，推理想复用同一生态；②看重 SplitFuse 带来的稳定 P99；③想要"几行代码起持久化服务"的简洁部署体验。

横向更多见 [[llm-inference/vllm]]、[[llm-inference/tensorrt-llm]]、[[llm-inference/huggingface-tgi]]。

---

## 12. 数值示例：SplitFuse 收益直觉

> 纯示意，用于建立量级感，非实测基准。

设 token 预算 $B = 2048$/step，一个长 prompt $P = 6144$ token。

**无 SplitFuse**：长 prompt 一次性 prefill。

$$\text{prefill 单步 token} = 6144 \Rightarrow \text{该步耗时} \approx 3\times \text{常规步}$$

这一步期间，其它正在 decode 的请求全部卡住，P99 飙升。

**有 SplitFuse**：每步留 1536 给该 prefill、512 给 decode，则切片数 $=\lceil 6144/1536\rceil = 4$ 步，每步 token 恒为 $1536+512=2048=B$，耗时近似恒定。于是：长 prompt 摊到 4 步、**没有任何一步爆炸**；每步顺带推进 512 个 decode 请求 → 几乎**零额外等待**；GPU 每步满载（算力 prefill 片 + 访存 decode 片混合）。

结论：**同样的硬件，P99 延迟更平、整体吞吐更高**。这就是 FastGen 宣称高吞吐的机制来源。

---

## 配置示例（讲含义，API 以官方为准）

```python
import mii

# 持久化部署：起一个 2 副本、单副本 2 卡张量并行的服务
client = mii.serve(
    "meta-llama/Llama-2-13b-hf",
    tensor_parallel=2,   # 单副本横跨 2 张卡 → 放下 13B 并降单请求延迟
    replica_num=2,       # 2 个副本 → 抗并发、提 QPS
    # deployment_name / dtype 等以官方文档为准
)

# 客户端发请求（可流式）
resp = client.generate(
    ["请用三句话介绍杭州"],
    max_new_tokens=128,  # 控制单请求占用 slot 的时长
)
print(resp)

client.terminate_server()  # 下线服务，释放显存
```

要点解读：`tensor_parallel × replica_num` = 总卡数（此例 4 卡）；调大 `replica_num` 提吞吐、调大 `tensor_parallel` 放更大模型 —— 见第 10 节权衡表。

---

## 常见问题

| 问题 | 解答 |
|------|------|
| MII 和 DeepSpeed-Inference 啥关系？ | MII 是上层服务/调度，DeepSpeed-Inference 是下层内核；MII 内部调用它。见第 7 节。 |
| FastGen 是另一个框架吗？ | 不是。FastGen 是 MII + DeepSpeed-Inference 协同后的"高吞吐生成系统"对外名称。 |
| Dynamic SplitFuse 一句话？ | 把长 prompt 的 prefill 切片、和很多请求的 decode token 拼成"每步恒定大小"的批，抹平 GPU 负载、稳住 P99。 |
| 持久化 vs 非持久化怎么选？ | 离线批量/调试用 `pipeline`；线上长期服务用 `serve`（gRPC 常驻 + 负载均衡）。 |
| 怎么扩吞吐？ | 加 `replica_num`（横向）；模型太大放不下再加 `tensor_parallel`（纵向）。 |
| 和 vLLM 比优势？ | 深度绑定 DeepSpeed 生态、SplitFuse 带来稳定 P99、部署极简；vLLM 胜在模型覆盖与社区。 |
| 支持流式输出吗？ | 支持 token 级流式回传，长生成体验更好（具体 API 以官方为准）。 |
| 量化支持？ | 支持低精度/量化以省显存提速，精度权衡见第 10 节（具体方案以官方文档为准）。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-inference/DeepSpeed-Inference]] — MII 的底层高性能推理内核
- [[llm-optimizer/SplitFuse]] — Dynamic SplitFuse 的深入推导与变体
- [[ai-framework/deepspeed/README]] — DeepSpeed 训练/推理总框架
- 同类对比：[[llm-inference/vllm]] · [[llm-inference/tensorrt-llm]] · [[llm-inference/huggingface-tgi]] · [[llm-inference/lmdeploy]]
