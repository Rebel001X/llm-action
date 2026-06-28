# LLM 推理总览

> LLM 推理 = 把"训练好的权重"高效地变成"用户看到的逐字输出"，核心是在**显存有限**与**延迟约束**下榨干 GPU 算力。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/解码策略]] [[llm-inference/KV-Cache优化]] [[llm-inference/vllm/README]] [[llm-optimizer/kv-cache]]

## 阅读地图

| 你想知道 | 看哪节 | 一句话 |
|---|---|---|
| 推理到底分几步 | §1 §2 | Prefill（读题）+ Decode（逐字写答案）两阶段 |
| 用什么指标衡量快慢 | §3 | TTFT / TPOT / 吞吐 / 显存，延迟与吞吐是跷跷板 |
| 显存到底花在哪 | §4 | 权重 + KV Cache + 激活，KV Cache 随并发/长度暴涨 |
| 有哪些优化手段 | §5 | KV 管理 / 批处理 / 并行 / 量化 / 解码 / 算子 六大类 |
| 市面上有哪些引擎 | §6 | vLLM / TGI / TensorRT-LLM / LMDeploy / llama.cpp … |
| 我该选哪个 | §7 + 对照表 | 按"在线服务 / 离线批量 / 端侧 / 国产卡"分流 |
| 手算一遍显存和吞吐 | 数值例子节 | 7B 模型一条请求/一批请求的账本 |

## 0. 一句话锚点

> **推理就是反复做"下一个 token 预测"**：给定已有 token 序列，前向算一次得到下一个 token 的概率分布，采样出一个 token，拼回序列，再算下一次——直到遇到结束符或达到长度上限。
> 工程上所有花活，都是为了让"算一次前向"这件事在 GPU 上**更省显存、更高并发、更低延迟**。

```
用户问题 ──► [Tokenizer] ──► token ids ──► [Transformer 前向×N次] ──► token ids ──► [Detokenizer] ──► 回答文本
                                              ▲                    │
                                              └──── 把上一步采样出的 token 拼回去 ────┘  (自回归循环)
```

## 1. 地基：它到底解决什么问题

### 1.1 自回归（Autoregressive）是一切的根源

语言模型建模的是联合概率，按链式法则拆成逐字条件概率：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

含义：**第 $t$ 个 token 的概率，依赖它前面所有 token**。所以生成必须**串行**：算出第 $t$ 个才能算第 $t+1$ 个。这条"串行链"是 LLM 推理慢、难并行的根本原因，也是后面所有优化的出发点。

### 1.2 一次前向里发生了什么（最原子拆解）

输入 token 序列长度 $L$，模型隐藏维度 $d$，层数 $N$。每一层 Transformer 做两件事：

```
        ┌──────────────── 一层 Transformer ────────────────┐
token ─►│  RMSNorm ─► Attention(Q,K,V) ─► +残差            │─► 下一层
        │                  │                                │
        │  RMSNorm ─► FFN(两个大矩阵乘) ─► +残差            │
        └───────────────────────────────────────────────────┘
```

- **Attention**：把当前 token 的 Query 和**所有历史 token** 的 Key/Value 做加权求和——这就是"看上下文"。代价随序列长度增长。
- **FFN（前馈）**：两个大矩阵乘法，参数量占大头（约模型总参数的 2/3）。

关键洞察：Attention 需要"所有历史 token 的 K 和 V"。如果每生成一个新字都重算全部历史的 K/V，复杂度是 $O(L^2)$，极其浪费——**这就引出了 KV Cache**（§4、[[llm-inference/KV-Cache优化]]）。

### 1.3 为什么推理和训练不是一回事

| 维度 | 训练 | 推理 |
|---|---|---|
| 方向 | 前向+反向+更新 | 只有前向 |
| 批 | 大 batch、整齐对齐 | 请求随机到达、长度参差 |
| 瓶颈 | 算力 + 梯度通信 | **显存带宽** + 显存容量 |
| 目标 | 收敛 | 在 SLA 内最大化吞吐 |

> 记住一句话：**推理大多数时候不是"算不过来"，而是"内存搬不过来"**——Decode 阶段是典型的 memory-bound（见 §3.3）。

## 2. 两阶段：Prefill 与 Decode（最核心的概念）

LLM 推理把一次请求切成两个性质完全不同的阶段。

```
请求："请用一句话介绍长城"            ┌─ 输出阶段（一个一个字蹦） ─┐
       │                              │                            │
   ┌───┴────── Prefill ──────┐   ┌──┴── Decode ──┐  ┌── Decode ──┐ ...
   │ 一次性并行处理整个 prompt │   │ 生成第1个token │  │生成第2个token│
   │ 算出所有 prompt token 的  │   │ 只前向1个token │  │只前向1个token│
   │ K/V，写入 KV Cache        │   │ 复用全部历史KV │  │复用全部历史KV│
   └──────────┬───────────────┘   └──────┬────────┘  └────┬───────┘
              │ TTFT 计时到这里             │ 每步=TPOT       │
              ▼                            ▼                ▼
          首 token                      第2个token        第3个token
```

### 2.1 Prefill（预填充 / Prompt 阶段）

- 输入：整段 prompt（比如 500 个 token）。
- 动作：**一次前向就并行处理全部 prompt token**（因为它们已知，不必串行），算出每个 token 的 K/V 存进 KV Cache。
- 性质：**compute-bound（算力受限）**——大矩阵乘，GPU 算得满。
- 产物：第一个输出 token + 填好的 KV Cache。
- 决定指标：**TTFT（首 token 延迟）**。prompt 越长，Prefill 越久。

### 2.2 Decode（解码 / 生成阶段）

- 输入：上一步刚生成的**那一个** token。
- 动作：只前向 1 个 token，但要读取**全部历史 KV Cache** 做 Attention，采样出下一个 token。
- 性质：**memory-bound（带宽受限）**——计算量极小（就 1 个 token），但要把全部权重和 KV 从显存搬进计算单元，时间花在"搬数据"上。
- 重复：每生成一个 token 就是一次 Decode，直到结束。
- 决定指标：**TPOT（每 token 延迟）**、总吞吐。

### 2.3 为什么必须分开理解

两阶段的瓶颈相反，所以优化手段也相反：

```
        算力利用率高          算力利用率低（GPU 在等内存）
Prefill ████████████   Decode ██░░░░░░░░░░
        compute-bound          memory-bound
        → 优化算力/并行         → 优化内存带宽/批处理(把多个请求的Decode合并)
```

很多引擎的关键设计（如 vLLM 的 PagedAttention、chunked-prefill、prefill/decode 分离部署）本质都是在**调和这两个阶段**。

## 3. 衡量指标：吞吐 vs 延迟（这是一对跷跷板）

### 3.1 四个核心指标

| 指标 | 全称 | 含义 | 谁在乎 |
|---|---|---|---|
| **TTFT** | Time To First Token | 从请求到第一个字的时间 | 聊天体验（光标多久开始动）|
| **TPOT** | Time Per Output Token | 相邻两个字之间的时间 | 打字速度（流畅度）|
| **Latency** | 端到端延迟 | $\text{TTFT} + (\text{生成长度}-1)\times\text{TPOT}$ | 单个用户 |
| **Throughput** | 吞吐 | 整个系统每秒产出多少 token（所有请求合计）| 成本 / 服务方 |

用户**感知速度** ≈ $1/\text{TPOT}$（比如 TPOT=20ms 即 50 token/s，约等于人快速阅读速度）。

### 3.2 延迟与吞吐为什么是跷跷板

```
单条请求独占GPU                  把很多请求攒成一批一起算
  延迟最低 ✔                       GPU吃饱、总吞吐高 ✔
  GPU空转、吞吐极低 ✘             单条请求要排队/等凑批，延迟变高 ✘

batch size ──小──────────────大──►
延迟        低 ───────────────► 高
吞吐        低 ◄─────────────── 高
```

工程目标：在**满足延迟 SLA（如 TTFT<500ms、TPOT<50ms）的前提下，把 batch 尽量做大**以提升吞吐、摊薄成本。这正是连续批处理（§5.2）要解决的事。

### 3.3 为什么 Decode 是 memory-bound（关键直觉）

衡量一个算子是 compute-bound 还是 memory-bound，看**算术强度** = 计算量 / 访存量。

- Decode 时只算 1 个 token：计算量 $\approx 2 \times P$ FLOPs（$P$=参数量，每个参数一次乘一次加）。
- 但要把**全部权重**从显存读进来：访存量 $\approx P \times \text{字节/参数}$。
- 算术强度极低 → GPU 算力闲置，时间全花在等显存带宽。

**推论**：Decode 阶段如果只跑 1 条请求，GPU 利用率可能不到 5%。把 $B$ 条请求一起 Decode，权重只读一次却服务了 $B$ 条——**这就是批处理能大幅提升吞吐却几乎不增加单步耗时的根本原因**（在带宽打满之前）。

## 4. 显存账本：钱花在哪（理解优化的前提）

推理时显存三大块：

```
┌───────────────── GPU 显存 ─────────────────┐
│  ① 模型权重    （固定，和并发无关）          │
│  ② KV Cache    （随 并发数 × 序列长度 暴涨）│ ◄── 最容易爆、最值得优化
│  ③ 激活/临时   （随 batch、临时缓冲）        │
└─────────────────────────────────────────────┘
```

### 4.1 权重显存

$$\text{权重显存} = P \times \text{字节/参数}$$

- FP16/BF16：2 字节 → 7B 模型 ≈ $7\text{e}9 \times 2 = 14$ GB。
- INT8：1 字节 → ≈ 7 GB；INT4：0.5 字节 → ≈ 3.5 GB（量化的直接收益，§5.4）。

### 4.2 KV Cache 显存（推理的"隐形吞金兽"）

每个 token、每一层，都要存一份 K 和一份 V：

$$\text{KV} = 2 \times N_{\text{layer}} \times L \times d_{kv} \times \text{字节} \times B$$

其中 2 是 K 和 V，$L$ 是序列长度，$d_{kv}$ 是 KV 的总维度（注意 GQA/MQA 会让 $d_{kv}$ 远小于 $d$），$B$ 是并发请求数。

**关键认知**：KV Cache **随并发数和上下文长度线性增长**。长上下文 + 高并发时，KV Cache 会**比权重还大**，成为真正的瓶颈。这就是为什么有了 PagedAttention、KV 量化、GQA/MQA、KV 卸载等一整套技术（[[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]）。

## 5. 优化全景：六大类武器

```
                    ┌─────────────────────────────┐
                    │      LLM 推理优化六大类       │
                    └─────────────────────────────┘
   ① KV 管理        ② 批处理         ③ 并行
   PagedAttention   连续批处理        TP/PP/EP/SP
   GQA/MQA          chunked prefill   多卡多机
   KV量化/卸载      P/D 分离

   ④ 量化           ⑤ 解码加速        ⑥ 算子/Kernel
   W8A8/W4A16       投机解码          FlashAttention
   GPTQ/AWQ/FP8     Medusa/EAGLE      算子融合/CUDA Graph
   KV-Cache量化     并行/前缀缓存     PagedAttention kernel
```

### 5.1 KV 管理：PagedAttention（vLLM 的招牌）

**问题**：传统实现给每条请求预留"最大长度"的连续显存，但实际长度参差 → 大量碎片浪费（内部碎片可达 60%+）。

**解法**：借鉴操作系统**虚拟内存分页**思想——把 KV Cache 切成固定大小的"块（block）"，按需分配、非连续存储，用一张"块表"映射逻辑位置到物理块。

```
逻辑视图(一条请求的KV)：[blk0][blk1][blk2]...
                          │     │     │
块表映射                  ▼     ▼     ▼
物理显存(共享池)：  ...[B7]...[B2]...[B9]...  ← 不连续，按需取，无浪费
```

收益：显存利用率接近 100%，相同显存能容纳更多并发请求 → 吞吐大涨；前缀相同的请求还能**共享物理块**（prefix caching）。详见 [[llm-inference/vllm/README]]、[[llm-inference/KV-Cache优化]]。

### 5.2 批处理：连续批处理（Continuous Batching）

**静态批处理**的痛点：一批里有的请求生成 10 个 token 就结束，有的要 500 个，整批必须等最慢的，GPU 大量空转。

**连续批处理（又称 in-flight batching）**：以 **iteration（每个 Decode 步）为粒度**调度——某请求结束就立刻把空位让给排队的新请求，无需等整批。

```
静态批：   req完成的早 ░░░░░[done]......空转等待......
                       ░░░░░░░░░░░░░░░░░░░░[done]

连续批：   req完成的早 ░░░░░[done]►换上新req 继续填满
                       ░░░░░░░░░░░░░░░░░░░░[done]
                       GPU 始终满载
```

这是现代推理引擎（vLLM、TGI、TensorRT-LLM）吞吐高的核心机制之一。

### 5.3 并行：把大模型/大负载摊到多卡多机

| 并行方式 | 切什么 | 适用 | 通信代价 |
|---|---|---|---|
| **TP** 张量并行 | 把每层的大矩阵按维度切到多卡 | 单层放不下 / 降延迟 | 每层 AllReduce，高，需 NVLink |
| **PP** 流水线并行 | 把不同层分到不同卡 | 模型太大、跨机 | 层间传激活，低，但有气泡 |
| **EP** 专家并行 | MoE 的专家分到不同卡 | MoE 模型 | All-to-All |
| **SP/CP** 序列/上下文并行 | 把长序列切到多卡 | 超长上下文 | 中 |

直觉：**TP 降延迟但费带宽（适合机内 NVLink）；PP 省带宽但有流水线气泡（适合跨机）**。实际大模型常 TP×PP 组合。

### 5.4 量化：用更少比特存权重/KV

把 FP16 权重压成 INT8/INT4/FP8，省显存 + 省带宽（Decode 是带宽受限，所以量化常**直接提速**）。

```
FP16  ████████████████  16 bit  基线
FP8   ████████          8 bit   省一半，精度损失小（需硬件支持）
INT8  ████████          8 bit   W8A8，成熟
INT4  ████              4 bit   W4A16(权重4bit/激活16bit)，省最多，需校准
```

主流方案：**GPTQ**（逐层校准）、**AWQ**（保护重要权重通道，见仓库 AWQ/AutoAWQ）、**SmoothQuant**（平滑激活离群值）、**FP8**（新卡原生）。还有 **KV-Cache 量化**专门压 KV。权衡：比特越低越省，但精度损失越大，需要校准数据和评测把关。

### 5.5 解码加速：投机解码（Speculative Decoding）

**痛点**：Decode 串行、一次只出 1 个 token，慢。
**思路**：用一个**小而快的草稿模型**一口气猜出未来 $k$ 个 token，再用**大模型一次前向并行验证**这 $k$ 个——猜对的就白赚，猜错的丢弃回退。因为大模型"验证 k 个"和"生成 1 个"耗时接近（都是一次前向），整体提速。

```
草稿模型(快)： 猜 → t1 t2 t3 t4 t5
大模型(慢)：   一次前向并行验证 t1..t5
              接受 t1 t2 t3 ✔  拒绝 t4 ✘ → 用大模型自己的 t4 替换
              本步净产出 4 个 token（而非 1 个）
```

变体：Medusa（多头自己猜）、EAGLE、Lookahead、n-gram。**无损**（输出分布和原模型一致）。更多采样/解码细节见 [[llm-inference/解码策略]]。

### 5.6 算子 / Kernel：让每次前向更快

- **FlashAttention**：把 Attention 的中间结果留在片上 SRAM、分块计算，避免反复读写 HBM，省显存+提速。
- **算子融合 + CUDA Graph**：把零碎小算子合并、固化 kernel 启动序列，减少 launch 开销（Decode 小算子多，收益明显）。
- **量化 kernel**：如仓库提到的 **marlin**（FP16×INT4 高效推理 kernel），让低比特真正跑得快。

## 6. 主流引擎地图

> 选项之多容易迷路。先按"它最擅长的场景"分类，再看下面对照表。**具体版本/CLI/API 以各官方文档为准**，这里讲稳定的定位与机制。

```
                        LLM 推理引擎地图
   在线高并发服务            离线/通用              端侧 & CPU
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │ vLLM        │        │ HF TGI      │        │ llama.cpp   │
   │ TensorRT-LLM│        │ OpenLLM     │        │ (GGUF量化)  │
   │ LMDeploy    │        │ Triton(后端)│        │ MLC / ggml  │
   │ SGLang      │        └─────────────┘        └─────────────┘
   └─────────────┘
   国产卡/本土生态                          研究 / 教学
   ┌─────────────┐                         ┌─────────────┐
   │ ZhiLight    │                         │ lite_llama  │
   │ 赤兔 chitu  │                         │ (mini框架)  │
   │ BigDL/IPEX  │                         └─────────────┘
   └─────────────┘
```

仓库收录的引擎与项目：

- **[vLLM](https://github.com/vllm-project/vllm)**：PagedAttention + 连续批处理的代表，吞吐标杆，生态最广，OpenAI 兼容接口（详见 [[llm-inference/vllm/README]]）。
- **[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)**：HuggingFace 出品，开箱即用、与 HF 生态无缝。
- **TensorRT-LLM / FasterTransformer**：NVIDIA 系。[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 是其前身高性能 kernel 库；TensorRT-LLM 编译优化极致，常配 **Triton** 推理服务器对外服务。
- **LMDeploy / TurboMind**：上海 AI Lab 出品，TurboMind 是其高性能 C++ 后端，量化与吞吐表现好。
- **[OpenLLM](https://github.com/bentoml/OpenLLM)**：BentoML 出品，偏"生产部署平台/统一接口"。
- **[ZhiLight](https://github.com/zhihu/ZhiLight)**（知乎 & 面壁智能）、**[赤兔 chitu](https://github.com/thu-pacman/chitu)**（清华）：国产推理引擎，关注国产硬件/成本。
- **[BigDL/IPEX](https://github.com/intel-analytics/BigDL)、[OpenVINO](https://github.com/openvinotoolkit/openvino)**：Intel 系，CPU/集成显卡推理。
- **llama.cpp / fastLLM / lightLLM**：轻量、端侧、CPU、GGUF 量化友好，个人本地跑模型首选。
- **[LLM Accelerator (LMOps)](https://github.com/microsoft/LMOps)**：微软，解码/系统层加速研究。
- **[lite_llama](https://github.com/harleyszhang/lite_llama/tree/main)**：mini 推理框架，适合读源码学原理。
- **量化相关**：AWQ / AutoAWQ（量化算法）、[marlin](https://github.com/IST-DASLab/marlin)（FP16×INT4 kernel）。
- **延伸阅读**：综述论文《A Survey on Inference Engines for Large Language Models》（[arXiv:2505.01658](https://arxiv.org/pdf/2505.01658)）。

## 数值例子 / 典型场景：手算一遍 7B 模型的账本

设：模型 7B、FP16、层数 $N=32$、隐藏维 $d=4096$、采用 **GQA**（8 个 KV 头、每头 128 维 → $d_{kv}=8\times128=1024$）、单卡 A100-80GB。

**① 权重显存**
$$7\text{e}9 \times 2\text{B} = 14\ \text{GB}$$

**② 每个 token 的 KV Cache（单条请求、每 token）**
$$2 \times N \times d_{kv} \times 2\text{B} = 2 \times 32 \times 1024 \times 2 = 131072\ \text{B} \approx 0.125\ \text{MB/token}$$

**③ 一条 2K 上下文请求的 KV**
$$0.125\ \text{MB} \times 2048 \approx 256\ \text{MB}$$

**④ 这张卡能放多少并发？**（留 14GB 权重 + ~6GB 激活/开销，剩 ~60GB 给 KV）
$$60\,000\ \text{MB} \div 256\ \text{MB} \approx 234\ \text{条 2K 请求}$$

> 对比：若不用 GQA 而用 MHA（32 个 KV 头），KV 大 4 倍，每请求 ~1GB，只能放 ~60 条——**GQA 直接把并发翻了约 4 倍**，这就是它被广泛采用的原因。

**⑤ Decode 吞吐的带宽上界（直觉估算）**
A100 显存带宽 ≈ 2 TB/s。单条 Decode 每步至少要读一遍 14GB 权重：
$$\text{每步耗时} \gtrsim 14\ \text{GB} \div 2000\ \text{GB/s} = 7\ \text{ms} \Rightarrow \text{单条} \lesssim 140\ \text{token/s}$$
但批处理时权重只读一次服务整批：放 200 条并发，**总吞吐可达数千 token/s**（直到带宽/算力打满）。这就是 §3.3 "批处理摊薄带宽"的量化体现。

## 对照表：引擎选型对比

| 引擎 | 强项 | 典型场景 | 关键机制 | 注意 |
|---|---|---|---|---|
| **vLLM** | 吞吐、生态、易用 | 在线高并发服务 | PagedAttention + 连续批处理 | 通用首选 |
| **TensorRT-LLM** | 极致单卡/低延迟 | NVIDIA 卡、追求极限 | 编译优化、定制 kernel | 上手成本高，绑 N 卡 |
| **TGI** | 开箱即用 | 快速上线、HF 生态 | 连续批处理 | — |
| **LMDeploy/TurboMind** | 量化+吞吐均衡 | 国内常用、量化部署 | C++ 后端 | — |
| **SGLang** | 复杂提示/Agent | 多轮、结构化、前缀复用 | RadixAttention 前缀缓存 | — |
| **llama.cpp** | 轻量、端侧、CPU | 本地/个人/边缘 | GGUF 量化 | 并发吞吐非强项 |
| **OpenLLM/Triton** | 部署平台/服务化 | 生产编排、多模型 | 服务框架（可挂上面引擎）| 本身非 kernel 引擎 |
| **ZhiLight/赤兔** | 国产硬件/成本 | 国产卡、降本 | 本土优化 | — |

## 常见问题

| 问题 | 答 |
|---|---|
| Prefill 和 Decode 哪个慢？ | 单步 Prefill 重（算整段 prompt），但 Decode 步数多、串行，长回答里 Decode 是总耗时大头 |
| 为什么加大 batch 几乎不增加单步耗时却暴涨吞吐？ | Decode 是 memory-bound，权重读一次服务整批，带宽打满前几乎"白送"（§3.3）|
| KV Cache 为什么这么重要？ | 它随并发×长度线性涨，长上下文/高并发时比权重还大，是显存瓶颈（§4.2）|
| 量化一定提速吗？ | 省显存是必然；提速主要来自带宽下降（Decode 受益），但需硬件支持低比特 kernel 且控好精度 |
| 投机解码是有损的吗？ | 标准投机解码**无损**，输出分布与原模型一致，只改速度 |
| TP 和 PP 怎么选？ | 机内 NVLink 用 TP（降延迟）；跨机省带宽用 PP（有流水线气泡）；大模型常组合 |
| 在线服务该选哪个？ | 通常 vLLM 起步；追求极限延迟上 TensorRT-LLM；国产卡看 LMDeploy/ZhiLight/赤兔 |
| 个人本地跑模型？ | llama.cpp + GGUF 量化，CPU/小显存也能跑 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，从这里出发
- [[llm-inference/解码策略]] — 贪心/采样/Top-k/Top-p/温度/投机解码细节
- [[llm-inference/KV-Cache优化]] — PagedAttention、GQA/MQA、KV 量化、前缀缓存
- [[llm-inference/vllm/README]] — vLLM 架构与实战
- [[llm-optimizer/kv-cache]] — KV Cache 在优化体系中的位置
