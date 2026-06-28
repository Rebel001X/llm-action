# 高性能 LLM 推理框架的设计与实现

> 把一个"会算 Transformer 的模型"变成一个"能扛住高并发、低延迟、低成本"的在线服务，靠的不是模型本身，而是它外面那层推理框架。本文从零拆解一个高性能 LLM 推理框架要解决什么、靠什么机制解决、各模块如何拼在一起。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/连续批处理]] · [[llm-inference/PD分离]]

> 参考原文：高性能 LLM 推理框架的设计与实现 https://zhuanlan.zhihu.com/p/682872971 （文中具体数字以原文/官方为准）

## 阅读地图

| 节 | 你将学到 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：推理框架到底在优化什么 | 吞吐 vs 延迟 |
| 1 | 地基：自回归推理的"两阶段"本质与三大痛点 | Prefill / Decode / 访存墙 |
| 2 | KV-Cache：用空间换时间，以及它带来的新问题 | KV 显存账 |
| 3 | PagedAttention：把显存当虚拟内存管 | 分页 / 碎片 |
| 4 | 连续批处理 Continuous Batching：调度的灵魂 | iteration-level 调度 |
| 5 | 算子层优化：FlashAttention / 融合 / 量化 | 访存 / Kernel |
| 6 | 并行与分布式：TP / PP / EP 与多卡多机 | 张量并行 / 通信账 |
| 7 | PD 分离：让 Prefill 和 Decode 各自最优 | 角色分工 |
| 8 | 一个完整框架的分层架构与请求生命周期 | Engine / Scheduler / Worker |
| 9 | 关键公式 / 显存账 / 数值手算 | 吞吐估算 |
| 10 | 主流框架对照与局限 | vLLM / TGI / TRT-LLM |

---

## 0. 一句话锚点

**LLM 推理框架 = 一个专门为"自回归 Transformer + 高并发请求"设计的调度与执行系统**。它的全部努力，可以浓缩成一句话：

> 在有限的 GPU 显存和算力下，尽可能把 GPU 算力"喂饱"（提高吞吐 throughput），同时让每个用户感觉"够快"（控制延迟 latency）。

这两个目标天然冲突。框架设计的本质，就是在 **吞吐 ↔ 延迟** 这条跷跷板上，用各种机制把帕累托前沿往外推。

```
        延迟低 (好)
            ^
            |   . 朴素逐请求推理 (低吞吐高延迟, 最差)
            |
   理想区 ->| *  现代框架想去的地方
            |        .  纯吞吐优化(大batch, 延迟差)
            +-----------------------> 吞吐高 (好)
```

---

## 1. 地基：自回归推理的"两阶段"与三大痛点

### 1.1 LLM 推理为什么特殊？

普通深度学习推理（图像分类）：**输入一次 → 前向一次 → 输出**。计算量固定、batch 容易凑。

LLM 推理是**自回归（autoregressive）**的：一次只生成一个 token，把它拼回输入，再算下一个，循环到结束。生成 200 个 token 就要前向 200 次。

```
输入: "今天天气"
  step1: [今天天气] -> 模型 -> "真"
  step2: [今天天气真] -> 模型 -> "好"
  step3: [今天天气真好] -> 模型 -> <eos>  停止
```

### 1.2 Prefill 与 Decode：两个性格完全不同的阶段

| 阶段 | 做什么 | 序列长度 | 计算特征 | 瓶颈 |
|---|---|---|---|---|
| **Prefill（预填充）** | 把整个 prompt 一次性前向，建立 KV-Cache | 一次处理 N 个 token | 大矩阵乘，**算力受限 compute-bound** | GPU FLOPs |
| **Decode（解码）** | 每步只算 1 个新 token | 每步 1 个 token | 矩阵×向量，**访存受限 memory-bound** | 显存带宽 |

这是理解一切优化的钥匙：

```
Prefill:  [t1 t2 t3 ... t_N]  一次算完, GPU 利用率高 (大GEMM)
                |
                v  生成 KV-Cache
Decode:   [.................. t_{N+1}]  算 1 个
          [.................. t_{N+1} t_{N+2}]  再算 1 个
          每步都要读全部权重 + 全部历史 KV, 但只算 1 个 token
          => 算力闲置, 卡在"搬数据"
```

**直觉**：Decode 阶段，GPU 要把几十 GB 的模型权重从显存搬到计算单元，却只为算 1 个 token 的输出。算得快没用，搬得慢才是命门——这就是 **memory-bound（访存墙）**。

### 1.3 三大痛点

1. **访存墙（Decode 慢）**：每步重复读全部权重，算力浪费。→ 靠 **批处理** 把多个请求的 token 凑成一批，分摊权重读取成本。
2. **显存爆炸（KV-Cache）**：每个请求要存历史 KV，长序列 + 高并发 = 显存瞬间打满。→ 靠 **PagedAttention** 精细管理。
3. **请求长短不一 + 动态到达**：有的生成 10 token，有的生成 1000 token；请求随时来随时走。静态 batch 会被最长的拖死。→ 靠 **连续批处理** 在 token 粒度上动态调度。

---

## 2. KV-Cache：用空间换时间

### 2.1 为什么需要 KV-Cache？

注意力机制里，第 $t$ 个 token 要和前面所有 token 算注意力：

$$\text{Attn}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\!\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d}}\right) V_{1:t}$$

注意：每生成一个新 token，前面 token 的 $K, V$ **完全不变**。如果每步都重算所有历史的 $K,V$，复杂度是 $O(N^2)$ 的重复劳动。

**KV-Cache**：把每个 token 算出的 $K_i, V_i$ 缓存起来，下一步只算新 token 的 $Q,K,V$，历史直接读缓存。把 Decode 每步从"重算全部"降为"算 1 个 + 读缓存"。

```
无缓存 step t: 重算 K_1..K_t, V_1..V_t   (浪费)
有缓存 step t: 只算 K_t, V_t, 追加到 cache; 读 cache 中 K_1..K_{t-1}
```

### 2.2 KV-Cache 显存账（务必会手算）

单个 token、单层的 KV 大小：

$$\text{bytes/token/layer} = 2 \times n_{kv} \times d_{head} \times \text{dtype}$$

其中 2 是 K 和 V，$n_{kv}$ 是 KV 头数（GQA 下远小于注意力头数）。整模型整序列：

$$\text{KV} = 2 \times L \times n_{kv} \times d_{head} \times S \times B \times \text{bytes}$$

（$L$ 层数，$S$ 序列长，$B$ 并发数）

**数值手算（以 Llama-2-13B 量级、FP16 为例，数字仅作量级示意，精确值见模型配置）**：
设 $L=40$、隐藏维 $d=5120$（MHA 下 $n_{kv}\times d_{head}=5120$）、FP16（2 字节）。

- 每 token 每层 KV：$2 \times 5120 \times 2 = 20480$ 字节 ≈ 20 KB
- 每 token 全模型：$20\text{KB} \times 40 = 800$ KB ≈ 0.78 MB
- 一条 2048-token 的请求：$0.78\text{MB} \times 2048 ≈ 1.6$ GB

**结论**：一个并发就吃 1.6 GB KV！并发 30 条就 ~48 GB，单卡 80GB 装完权重（~26GB FP16）后所剩无几。**KV-Cache 是显存的头号消耗者，也是并发上限的硬约束。** GQA/MQA（减少 $n_{kv}$）和量化 KV-Cache 都是为了砍它。

---

## 3. PagedAttention：把显存当虚拟内存管

### 3.1 朴素 KV-Cache 的浪费

传统做法：为每个请求**预留一段连续显存**，按"可能的最大长度"分配。问题：

- **内部碎片**：请求实际只用 100 token，却按 2048 预留，浪费 95%。
- **外部碎片**：请求长度各异，连续大块难凑，显存看似够却分不出来。
- 实测朴素方案有效利用率可能只有 **20%~40%**（具体见 vLLM 论文，约数）。

### 3.2 PagedAttention 的核心思想：借鉴操作系统分页

把 KV-Cache 切成固定大小的 **block（页）**（例如每块 16 个 token 的 KV），**物理上不要求连续**，用一张 **block table（页表）** 把逻辑序列映射到物理块。

```
逻辑视角(请求A的KV序列):  [tok0..15][tok16..31][tok32..47]
                            block0     block1     block2
                              |          |          |
   page table(A): [phys#7, phys#2, phys#9]   <- 不连续!
                              |          |          |
物理显存池: [#0][#1][#2(A)][#3][#4][#5][#6][#7(A)][#8][#9(A)]...
```

**收益**：
1. **几乎零碎片**：按需一块块分配，用多少给多少，利用率可达 90%+。
2. **共享（Copy-on-Write）**：多个请求共享同一 prompt（如 few-shot 模板、并行采样 beam/n）时，可共享物理块，写时才复制，省显存。
3. **动态增长**：序列变长就追加一块，不用预留最大长度。

```
并行采样(同一prompt生成3个候选):
  seqA ─┐
  seqB ─┼─> 共享 prompt 的物理块(只读) ──> 各自分支后才分配新块
  seqC ─┘
```

**代价**：注意力 kernel 要支持"按页表 gather KV"，比连续访问略复杂——这正是 PagedAttention 这个**定制算子**的工作。

---

## 4. 连续批处理（Continuous Batching）：调度的灵魂

### 4.1 静态批处理的问题

朴素批处理（static / request-level batching）：凑齐 B 个请求一起跑，**全部生成完才返回，才能接新请求**。

```
静态batch (B=4), 各请求生成长度不同:
  req1: ████ (4)         done, 但要等...
  req2: ██████████ (10)  
  req3: ██ (2)           done, 但要等...
  req4: ███████ (7)      
时间 -->  [          全部等到 req2 的第10步才一起结束          ]
          ^^^ req1/req3 早就算完, GPU 却在为它们空转 (气泡)
```

短请求被长请求"绑架"，GPU 利用率低，新请求要排队。

### 4.2 连续批处理：迭代级（iteration-level）调度

核心洞察：**调度粒度从"一整个请求"降到"一次迭代（一个 token step）"**。每生成完一步：

- 已结束（出 `<eos>` 或达 max_len）的请求**立即移出** batch、返回结果；
- 队列里**等待的新请求立即填补**空位（先做它的 prefill，再加入 decode batch）。

```
连续批处理:
  step:   1  2  3  4  5  6  7  8  9 10
  req1:   ■  ■  ■  ■ done
  req2:   ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  req3:   ■  ■ done
  req5:         ■  ■  ■  ■  ■ <- req3走后立即补进来
  req6:               ■  ■  ■  ■  ■  ■
  GPU 始终满载, 没有"等齐"气泡
```

**效果**：相比静态批处理，吞吐可提升数倍（vLLM 论文称达数十倍的特定场景，约数/见原文），延迟也更稳。这是现代框架（vLLM 的 ORCA 思想、TGI、TRT-LLM）的标配。

### 4.3 调度还要管什么？

- **Prefill / Decode 混批**：新请求的 prefill（compute-bound）和老请求的 decode（memory-bound）混在一起跑，互补利用算力与带宽。常用 **chunked prefill**（把长 prompt 的 prefill 切块，避免长 prefill 阻塞 decode，平滑延迟）。
- **抢占与换出（preemption / swapping）**：显存不够时，把某些请求的 KV 换出到 CPU 内存（swap）或直接重算（recompute），腾地方给高优先级请求，之后再换回。
- **优先级 / 公平性**：避免长请求饿死短请求。

---

## 5. 算子层优化：把每个 Kernel 榨干

调度解决"怎么排"，算子解决"每步算得多快"。

### 5.1 FlashAttention：不把注意力矩阵落显存

朴素注意力要显式生成 $N\times N$ 的注意力分数矩阵，读写 HBM（显存）次数 $O(N^2)$，是访存瓶颈。

**FlashAttention** 用 **tiling（分块）+ online softmax**，把 Q/K/V 分块加载到 SRAM（片上高速缓存），在片上完成 softmax 与加权求和，**永不把完整 $N\times N$ 矩阵写回 HBM**。

```
朴素:  Q,K -> S(N×N 写HBM) -> softmax(读写HBM) -> ×V  (访存爆炸)
Flash: 分块加载到SRAM, online softmax 增量更新, 只写最终 O
       HBM 访存从 O(N^2) 降到 ~O(N)  => 大幅加速且省显存
```

详见 [[llm-optimizer/FlashAttention]]。Decode 阶段则用 **FlashDecoding / Paged 变体**，按页表读 KV。

### 5.2 算子融合（Kernel Fusion）

把多个小算子（如 LayerNorm + 残差、Q/K/V 投影、SwiGLU 的几步）融成一个 kernel，减少 kernel 启动开销和中间结果的 HBM 往返。

```
未融合: x -> norm(读写HBM) -> linear(读写HBM) -> act(读写HBM)
融合:   x -> [norm+linear+act 一个kernel, 中间值留寄存器/SRAM] -> y
```

### 5.3 量化（Quantization）

把权重/激活/KV 从 FP16 降到 INT8、FP8、INT4，**减少访存量 = 直接给 Decode 提速**，同时省显存（可上更大 batch）。

- 权重量化（W8A16 / W4A16，如 GPTQ/AWQ）：主要省显存 + 加速 Decode（访存）。
- 权重+激活（W8A8、FP8）：连 Prefill 的算力也能加速（用 INT8/FP8 Tensor Core）。
- KV-Cache 量化：直接砍第 2 节算的那笔 KV 显存账，换更高并发。

详见 [[llm-compression/quantization/量化基础]] 与 [[llm-compression/quantization/fp8]]。

---

## 6. 并行与分布式：单卡装不下就拆

当模型大到单卡放不下，或要追求更低延迟，就要多卡。

### 6.1 张量并行 TP（Tensor Parallelism）

把单层的权重矩阵**横/纵切开**分到多卡，每卡算一部分，再用 **All-Reduce / All-Gather** 把结果拼回。延迟敏感、卡间带宽高（NVLink）时用，通常在单机内做。

```
权重 W 切成 W1|W2 分到 GPU0/GPU1
  GPU0: x·W1 ─┐
  GPU1: x·W2 ─┴─ All-Reduce ─> 完整结果
每层都要一次集合通信 => 吃 NVLink 带宽
```

**通信账（粗略）**：每个 Transformer 层 TP 大约需 2 次 All-Reduce（注意力后、MLP 后），通信量 ∝ $B \times S \times d$。所以 TP 度数越高，通信占比越大，跨机做 TP（走 PCIe/网络）会很亏。详见 [[B07:llm-inference/大模型推理张量并行]] 与 [[ai-infra/网络/集合通信原语]]。

### 6.2 流水线并行 PP（Pipeline Parallelism）

把**不同层**分到不同卡（卡0 跑 1-10 层，卡1 跑 11-20 层…），像流水线一样传递激活。通信少（只传层间激活），适合跨机扩展，但有 **流水线气泡**。

### 6.3 专家并行 EP（Expert Parallelism）

MoE 模型把不同 **专家（expert）** 分到不同卡，token 经 router 用 **All-to-All** 发到对应专家卡。详见 [[llm-algo/moe/README]]。

| 并行方式 | 切什么 | 通信 | 适用 |
|---|---|---|---|
| TP | 单层权重 | All-Reduce（重） | 单机多卡、降延迟 |
| PP | 层 | 点对点（轻） | 跨机扩展 |
| EP | 专家 | All-to-All | MoE 模型 |
| DP/副本 | 整模型多副本 | 无（请求级） | 提吞吐、横向扩 |

实战常用 **TP×PP×DP** 的混合并行。

---

## 7. PD 分离（Prefill-Decode Disaggregation）

### 7.1 动机

回顾第 1 节：Prefill 是 **compute-bound**，Decode 是 **memory-bound**。把两者塞在同一组 GPU 上混跑，会互相干扰：

- 长 prompt 的 prefill 占满算力，让正在 decode 的请求**卡顿（TBT 抖动）**；
- 为 decode 优化的配置（大 batch、显存全给 KV）不利于 prefill。

### 7.2 思路：分两组机器，各自最优

**PD 分离**：用一组 GPU 专做 Prefill（算力型配置），另一组专做 Decode（带宽/显存型配置）。Prefill 算完把 **KV-Cache 通过高速网络（如 NVLink/RDMA）传给 Decode 节点**，由后者继续生成。

```
请求 ──> [Prefill 集群]  算 prompt, 产出首token + KV-Cache
                |  KV-Cache 传输 (RDMA/NVLink)
                v
         [Decode 集群]  逐 token 生成, 大batch 高吞吐
                |
                v  流式返回
```

**收益**：两阶段各自配比、各自 batch 策略，互不打架；首 token 延迟（TTFT）和 token 间延迟（TBT）都更可控。**代价**：要传 KV-Cache（带宽/延迟成本）、调度更复杂。详见 [[llm-inference/PD分离]]。

---

## 8. 一个完整框架的分层架构与请求生命周期

把上面所有机制拼起来，一个高性能推理框架长这样：

```
┌──────────────────────────────────────────────────────────┐
│  API 层 (OpenAI 兼容 HTTP/gRPC, 流式 SSE)                   │
├──────────────────────────────────────────────────────────┤
│  Engine / Scheduler 调度层                                  │
│   - 请求队列 (waiting / running / swapped)                  │
│   - 连续批处理调度 (iteration-level)                        │
│   - KV Block 管理器 (PagedAttention 页表 + 分配/回收)       │
│   - 抢占/换出策略, chunked prefill                          │
├──────────────────────────────────────────────────────────┤
│  Model Executor / Worker 执行层 (每卡一个 worker)           │
│   - 并行通信 (TP/PP/EP 的 All-Reduce/All-to-All)            │
│   - 前向计算: 融合算子 / FlashAttention / 量化 GEMM         │
│   - 采样 (top-k/top-p/temperature, 投机解码可选)            │
├──────────────────────────────────────────────────────────┤
│  显存 / 资源层: KV 物理块池, 权重, CUDA Graph, 通信缓冲     │
└──────────────────────────────────────────────────────────┘
```

### 8.1 一个请求的完整生命周期

```
1. 到达: HTTP 请求 -> API 层解析(prompt, max_tokens, 采样参数) -> 入 waiting 队列
2. 调度: Scheduler 选中 -> 分配 KV blocks -> 做 Prefill (一次大前向)
         -> 产出首 token (TTFT 在此刻确定)
3. 进入 running batch, 每个 iteration:
     Scheduler 把所有 running 请求的 1 个 step 凑成一批 ->
     Worker 前向(读权重+按页表读KV) -> 采样出各请求的下一 token ->
     追加到各自 KV(可能新分配一块) -> 流式吐回该 token
4. 某请求出 <eos> 或达 max_len: 移出 batch, 释放其 KV blocks 回池,
   waiting 队列的新请求立即补位
5. 显存紧张: 低优先级请求被抢占, KV 换出 CPU 或丢弃重算
6. 结束: 流式响应收尾, 资源全部回收
```

### 8.2 性能加速利器：CUDA Graph

Decode 阶段每步的 kernel 序列固定且很多很小，CPU 启动 kernel 的开销（launch overhead）反而成瓶颈。**CUDA Graph** 把整串 kernel 录制成一张图，一次性提交，省掉逐个启动的 CPU 开销，对小 batch 的 decode 提速明显。

### 8.3 投机解码（Speculative Decoding，可选）

用一个**小草稿模型**一次猜多个 token，再用大模型**一次性并行验证**，接受对的、拒绝错的。把"逐 token"变成"一批一验证"，在 memory-bound 的 decode 上偷算力，降低延迟（不改变输出分布）。

---

## 9. 关键公式 / 显存账 / 数值手算

### 9.1 显存总账

$$\text{总显存} = \underbrace{\text{权重}}_{P \times b_w} + \underbrace{\text{KV-Cache}}_{2 L\, n_{kv} d_h S B\, b_{kv}} + \underbrace{\text{激活 + 通信缓冲 + 其它}}_{\text{额外}}$$

**手算（13B、FP16、单卡 80GB，量级示意）**：
- 权重：$13\text{B} \times 2 \approx 26$ GB
- 系统/激活预留：约 10 GB
- 留给 KV：$80 - 26 - 10 = 44$ GB
- 每请求 2048-token KV ≈ 1.6 GB（见 §2.2）
- **最大并发 ≈ $44 / 1.6 \approx 27$ 条**

要提高并发 → 量化权重（腾出空间）+ 量化 KV（减小每条占用）+ GQA（减小 $n_{kv}$）。

### 9.2 Decode 的访存上界（Roofline 直觉）

Decode 每步主要在"读权重 + 读 KV"。单步时间下界：

$$t_{step} \gtrsim \frac{P \times b_w + \text{KV读取量}}{\text{显存带宽}}$$

**手算**：权重 26 GB，带宽设 2 TB/s（A100/H100 量级），则纯读权重下界 $\approx 26/2000 \approx 13$ ms/step。
- 单请求吞吐上界 ≈ $1000/13 \approx 77$ token/s。
- **但批处理时，B 个请求共享同一次权重读取！** 26GB 只读一次就服务整批 → 总吞吐 ≈ $77 \times B$（直到被算力或 KV 带宽重新卡住）。
- 这就是为什么**批处理对吞吐是数量级提升**，也解释了 §1 那张访存墙图的工程价值。

### 9.3 三个核心延迟指标

| 指标 | 含义 | 受谁影响 |
|---|---|---|
| **TTFT**（Time To First Token） | 首 token 延迟 | Prefill 速度、排队、prompt 长度 |
| **TBT / ITL**（Time Between Tokens） | token 间延迟 | Decode 单步、batch 大小、混批干扰 |
| **吞吐**（tokens/s, 全系统） | 总产出 | batch、并行、显存利用率 |

详见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

---

## 10. 主流框架对照与局限

| 框架 | 代表机制 | 定位 | 备注 |
|---|---|---|---|
| **vLLM** | PagedAttention + 连续批处理 | 开源高吞吐通用引擎 | 生态广，OpenAI 兼容 |
| **TGI**（HF） | 连续批处理 + 张量并行 | HF 生态服务化 | 与 transformers 紧耦合 |
| **TensorRT-LLM** | 编译期算子优化 + In-flight batching | NVIDIA 极致性能 | 需编译，绑定 N 卡 |
| **SGLang** | RadixAttention（前缀树共享 KV） | 复杂提示/多轮共享 | 前缀复用强 |
| **DeepSpeed-Inference / LMDeploy 等** | 各家融合算子 + 并行 | 各有侧重 | — |

（以上为定性对照，具体性能随版本/硬件/负载变化很大，**以官方基准为准**。）

### 局限与权衡

| 维度 | 取舍 |
|---|---|
| 吞吐 vs 延迟 | 大 batch 提吞吐但可能拉高 TBT；要按 SLO 调度 |
| 显存 vs 精度 | 量化省显存提速，但可能损精度（需校准/评测） |
| PD 分离 | 减少阶段干扰，但增加 KV 传输与运维复杂度 |
| 投机解码 | 降延迟，但草稿模型占显存、接受率低时白算 |
| 并行度 | TP 降延迟但通信重；PP 省通信但有气泡 |
| 长上下文 | KV 随 $S$ 线性涨，需 KV 量化/驱逐/分级缓存 |

**一句话总收**：高性能推理框架 = **PagedAttention（管好显存）+ 连续批处理（调好顺序）+ 算子优化（算得快）+ 并行/PD 分离（拆得开）**，四者围绕"喂饱 GPU 又不拖慢用户"这一根主线协同。

---

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 模型基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
- 算子优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理核心：[[llm-inference/vllm/README]] · [[llm-inference/连续批处理]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-compression/sparsity/README]]
- 并行分布式：[[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]] · [[llm-train/pytorch/distribution/README]]
- 指标与估算：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
