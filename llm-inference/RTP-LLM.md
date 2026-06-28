# RTP-LLM

> 阿里巴巴开源的高性能 LLM 推理 / serving 引擎：以 FasterTransformer 为底座，融合 TensorRT-LLM 的 kernel 与 vLLM 的连续批处理思想，主打"生产级在线服务"的低延迟与高吞吐。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/vllm/README]] [[llm-inference/README]]

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | RTP = Real-Time Prediction |
| 1 | 它到底解决什么问题 | 在线 serving 的延迟/吞吐/显存三角 |
| 2 | 整体架构与分层 | Python 前端 + C++ 内核 + Op 库 |
| 3 | 连续批处理(Continuous Batching) | iteration-level 调度 |
| 4 | PagedAttention 式 KV 显存管理 | 分页、碎片、复用 |
| 5 | 核心 kernel 与算子来源 | FT / TRT-LLM / FlashAttention2 / cutlass |
| 6 | 高级特性 | 投机采样 / 多模态 / 量化 / P-D 分离 |
| 7 | 并行与分布式 | TP / PP / EP |
| 数值例子 | 显存与吞吐手算 | 7B/13B 场景 |
| 对照表 | 与 vLLM / TRT-LLM 对比 | 选型 |
| 常见问题 | 易错点澄清 | FAQ |

> 说明：RTP-LLM 仍在快速演进，具体版本号、CLI 参数、API 签名请以 [官方仓库](https://github.com/alibaba/rtp-llm) 为准；本文聚焦"稳定的机制与权衡"。

---

## 0. 一句话锚点

**RTP-LLM = 一个把"离线 kernel 性能"和"在线服务调度"缝合在一起的 LLM 推理引擎。**

- **RTP** 取自阿里内部 **R**eal-**T**ime **P**rediction（实时预测）平台，天生为"在线请求服务"而生，而非离线批量跑分。
- 底层"算得快"靠 **FasterTransformer（FT）+ TensorRT-LLM（TRT-LLM）的 kernel**；
- 上层"调度省"靠 **Continuous Batching + 分页 KV Cache**（思想源自 vLLM）。

记住这张"夹心饼"：

```
  ┌─ 连续批处理 + 分页KV ─┐  ← 上层(服务/调度), 借鉴 vLLM 思想
  └─ 高性能算子内核 ──────┘  ← 下层(kernel), 借鉴 FT / TRT-LLM
```

---

## 1. 地基：它解决什么问题

### 1.1 在线 serving 的"不可能三角"

LLM 推理在生产环境要同时满足三件互相打架的事：

```
        低延迟(TTFT/TPOT)
            ▲
           ╱ ╲
          ╱   ╲
         ╱     ╲
高吞吐 ◀─────────▶ 省显存
(QPS)              (KV Cache)
```

- **TTFT**（Time To First Token，首 token 延迟）：用户点回车到看到第一个字。
- **TPOT**（Time Per Output Token，每输出 token 延迟）：后续字一个个蹦出来的速度。
- **吞吐（Throughput）**：单卡每秒服务多少 token / 多少并发请求。
- **显存**：KV Cache 会随上下文长度线性膨胀，是并发数的硬天花板。

朴素实现（一个请求一个请求地跑、KV Cache 一次性按 max_len 预留）会让三角全崩：延迟高、吞吐低、显存浪费。

### 1.2 为什么"自回归生成"天生难优化

LLM 解码是**自回归**的：第 $t$ 个 token 依赖前 $t-1$ 个。

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{1:t-1})
$$

这导致两个结构性难题：

1. **Decode 阶段是 memory-bound**。每生成 1 个 token，只算 1 行，却要把整个模型权重 + 全部历史 KV 从显存搬一遍。算力闲置、带宽打满。
2. **不同请求长度天差地别**。有人输入 10 token、有人 2000；有人输出 5 个、有人 500 个。静态 batch 会被最长的那个拖死（短的算完了还得等长的）。

RTP-LLM 的全部优化，本质都在对付这两点：**把 memory-bound 的 decode 喂饱（kernel + batching），把显存碎片榨干（分页 KV）**。

### 1.3 Prefill vs Decode：两种性质完全不同的阶段

```
请求 "中国的首都是" → 模型 → "北京" "，" "是" ...

Prefill(预填充): 一次性并行处理所有 prompt token, 计算并缓存 KV
   → 计算密集(compute-bound), 算力打满, 类似一次大矩阵乘
Decode(解码):   每次只处理 1 个新 token, 复用历史 KV
   → 访存密集(memory-bound), 带宽打满, batch 越大越划算
```

理解这个二分法是理解后面所有调度策略的钥匙。

---

## 2. 整体架构

RTP-LLM 是典型的 **"Python 易用前端 + C++ 高性能内核"** 双层结构。

```
┌─ Python 前端(易用层) ──────────────────────────────────┐
│  HTTP/OpenAI 兼容 API Server · 模型加载/权重转换/Tokenizer │
│  请求接入、参数校验、流式返回(SSE)                         │
└────────────────────┬─ (pybind / C++ 绑定) ─────────────┘
┌────────────────────▼─ C++ 调度内核(Engine) ────────────┐
│  Scheduler: iteration-level 连续批处理                   │
│  CacheManager: 分页 KV Cache 分配/回收/复用              │
│  Sampler: greedy/top-k/top-p/温度 · SpeculativeEngine    │
└────────────────────┬───────────────────────────────────┘
┌────────────────────▼─ Op/Kernel 层(算得快) ────────────┐
│  FasterTransformer 算子 + TensorRT-LLM kernel           │
│  + FlashAttention2 + cutlass GEMM   (GPU: CUDA)         │
└────────────────────────────────────────────────────────┘
```

**为什么这样分层？** Python 改起来快、生态全（HuggingFace 权重、tokenizer），适合"接客"；C++/CUDA 跑得快、可控内存，适合"干活"。绑定层（pybind）让两者各司其职——这是几乎所有现代推理引擎（vLLM、TGI、TRT-LLM）的共同范式。

### 2.1 一个请求的生命周期（数据流）

```
client ─HTTP→ API Server(tokenize) → Request Queue(等待池)
   │ 2. Scheduler 每 iteration 挑选: 显存够? 优先级? 拼进当前 batch?
   ▼
CacheManager 分配 KV 页 → 跑一个 step(Prefill 或 Decode) → Sampler 采样 next token
   │ 4. 流式吐 token 回 client(SSE)
   ▼
遇到 EOS / 达到 max_len? → 释放 KV 页, 请求结束
```

第 2~4 步在一个**循环**里反复发生——这就是"continuous"的含义：调度发生在**每一步迭代**，而非"凑齐一个 batch 跑到底"。

---

## 3. 连续批处理（Continuous Batching）

这是相对静态批处理的核心升级，思想直接参考 vLLM。

### 3.1 静态批处理的浪费

```
静态批处理 (Static Batching) —— ✗ 浪费:
时间 →
R1 ████████░░░░░░░░   (输出 4 个就完了, 后面干等)
R2 ████████████████   (输出 16 个)
R3 ██████░░░░░░░░░░░░ (输出 3 个就完了, 干等)
   ↑ 整个 batch 必须等最长的 R2 跑完才能换下一批
   ░ = GPU 算力被浪费(在算 padding / 空转)
```

### 3.2 连续批处理：迭代级调度

```
连续批处理 (Continuous Batching) —— ✓ 高效:
时间 →
R1 ████        ← R1 完成后, 槽位立刻被 R4 顶上
R2 ████████████████
R3 ██████
R4     ██████████    ← R1 一空出来就插进来
R5         ████████  ← R3 一空出来就插进来
   ↑ 每个 iteration 都重新组 batch, 槽位永不空转
```

机制要点：

1. **iteration-level scheduling**：每生成一个 token（一次迭代）后，调度器重新审视——谁完成了就踢出、释放显存；等待队列里谁能进就拉进来。
2. **Prefill 与 Decode 混合**：新请求的 prefill 可以和老请求的 decode 拼在同一个 batch（或交替调度），让 GPU 始终满载。
3. **结果**：吞吐相对静态批处理可提升数倍，尾延迟也更稳。

> 直觉：把 GPU 想象成餐厅厨房，静态批处理是"这桌全上齐才接下一桌"，连续批处理是"哪道菜出锅就立刻上、空灶马上炒新菜"。

---

## 4. PagedAttention 式 KV Cache 显存管理

### 4.1 为什么 KV Cache 是头号显存杀手

每个 token 在每一层都要缓存 Key 和 Value 向量，供后续 token 做注意力。单请求 KV Cache 显存：

$$
\text{KV bytes} = 2 \times L \times H \times d_{head} \times n_{layer} \times \text{seq\_len} \times \text{dtype\_bytes}
$$

其中 $2$ 是 K 和 V，$L$ 这里指批内序列条数。注意它**随上下文长度线性增长**——上下文越长、并发越多，吃显存越凶。

### 4.2 朴素方案的两宗罪

```
朴素: 每个请求按 max_len 预留一整块连续显存

请求A (实际用 50 token, max_len=2048):
[████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 用了  ←──── 内部碎片(浪费) ────→

请求B 想进来, 但剩余显存是零碎的:
[██A██][空][██C██][空][██E██]  ← 外部碎片
           想塞 B 进去? 没有足够大的连续块! ✗
```

- **内部碎片**：按最大长度预留，实际只用一小段。
- **外部碎片**：分配/释放后显存被切得七零八落，明明总量够却塞不下新请求。

### 4.3 分页：操作系统虚拟内存的思想

把 KV Cache 切成固定大小的**页（block）**（如每页存 16 个 token 的 KV），用**页表（block table）**把"逻辑序列位置"映射到"物理显存页"——逻辑连续、物理可离散。

```
逻辑序列 (请求看到的):   物理显存页 (实际存储):
token 0-15  ─┐          ┌─▶ [Page 7 ]
token 16-31 ─┼─页表映射─┼─▶ [Page 2 ]
token 32-47 ─┘          └─▶ [Page 9 ]
                           (物理上不连续, 逻辑上连续)
```

收益：

1. **消除外部碎片**：任何空闲页都能用，不需要大块连续显存。
2. **内部碎片≤1页**：只有最后一页可能没填满，浪费极小。
3. **显存利用率大幅提升** → 同样显存能容纳更多并发请求 → 吞吐上去了。
4. **天然支持前缀复用（Prefix Caching）**：多个请求共享相同 prompt 前缀（如同一 system prompt）时，可共享同一批物理页，靠引用计数管理，省显存又省重复 prefill。

```
Prefix Caching:  请求A: [系统提示页P1][P2]+[A内容页P5]
                 请求B: [系统提示页P1][P2]+[B内容页P8]
                          ↑ 共享同一批物理页, 不重复算 prefill
```

---

## 5. 核心 kernel 与算子来源

RTP-LLM 不重复造轮子，而是站在巨人肩上，把各家最快的 kernel 集成进来。**这正是它原始 README 强调的：**

| 来源 | 贡献的能力 | 为什么用它 |
|------|-----------|-----------|
| **FasterTransformer** | 项目主底座，融合 LayerNorm/GEMM/激活等高度优化算子 | 久经考验的 Transformer kernel，性能可靠 |
| **TensorRT-LLM** | 集成其部分 kernel 实现 | NVIDIA 官方深度调优，attention/量化 kernel 强 |
| **FlashAttention2** | IO 感知的精确注意力 | 不实体化 $N{\times}N$ 注意力矩阵，省显存、提速 |
| **cutlass** | 高性能 GEMM 模板库 | 灵活定制融合 GEMM，喂饱 Tensor Core |
| **vLLM** | continuous batching + increment decoding 参考 | 业界连续批处理标杆 |
| **transformers** | 采样逻辑参考 | 行为与 HF 对齐，结果可复现 |
| **Medusa** | 投机采样实现 | 多头并行预测加速 decode |
| **llava / qwen-vl** | 多模态能力 | 直接支持图文输入 |

### 5.1 FlashAttention2 为什么快（最底层直觉）

标准注意力要先算出完整的 $S = QK^\top$（大小 $N\times N$），存到显存再 softmax，访存量是 $O(N^2)$。FlashAttention 用**分块（tiling）+ online softmax**，边算边累加，永不实体化整张 $N\times N$ 矩阵：

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

把 $Q,K,V$ 切块装进 SRAM（片上高速缓存）逐块计算，HBM（显存）访问从 $O(N^2)$ 降到 $O(N)$ 级别。注意力是 memory-bound，省访存=直接提速，长序列收益尤其大。

---

## 6. 高级特性

### 6.1 投机采样（Speculative Decoding，集成 Medusa）

Decode 是 memory-bound，"一次只产 1 个 token"太亏。投机采样思路：**先用便宜方式猜一串 token，再用大模型一次性并行验证。**

```
传统 decode:  大模型 → 1 token → 大模型 → 1 token → ...  (慢, 串行)

投机采样:
  草稿(Medusa 多头/小模型) → 猜 4 个: [t1 t2 t3 t4]
  大模型一次 forward 并行验证这 4 个
    ├ 全对   → 一步前进 4 个 token (4x 加速)
    └ 第3个错 → 接受 t1 t2, 丢弃 t3 t4, 从 t3 重来
  结果分布严格等价于大模型直接采样(无损加速)
```

Medusa 不用独立小模型，而是在主模型上加几个"预测头"并行猜多个后续 token，省去额外模型的部署成本。

### 6.2 多模态（llava / qwen-vl）

图像先过视觉编码器变成 embedding，作为"软 token"拼到文本 token 序列前，后续走和纯文本一样的 LLM 流程。RTP-LLM 集成了 llava、qwen-vl 的处理逻辑，开箱支持图文输入。

### 6.3 量化（Weight-Only / KV Cache 量化）

把权重从 FP16 压成 INT8/INT4，显存减半甚至减到 1/4，访存量同步下降 → decode 提速。常见 weight-only（仅权重量化、计算时反量化），以及 KV Cache 量化（直接压缩 KV 占用）。代价是精度略降，需校准。

### 6.4 Prefill-Decode 分离（P-D 分离）

Prefill 是 compute-bound、Decode 是 memory-bound，混在同一组卡上会互相干扰（长 prompt 的 prefill 会卡住其他请求的 decode，拉高 TPOT 抖动）。P-D 分离把两阶段拆到不同 GPU 池，靠 KV Cache 传输衔接：

```
请求 →[ Prefill 集群(吃算力) ]──KV传输──▶[ Decode 集群(吃带宽/显存) ]→ 流式输出
```

各自独立扩缩容、互不干扰，首 token 与每 token 延迟都更稳，属当前生产级 serving 主流方向。

---

## 7. 并行与分布式

单卡放不下大模型时，按维度切分：

```
张量并行 TP: 把每层权重矩阵切到多卡  [W]=[W1|W2|W3|W4], AllReduce 合并
  → 降单卡显存&延迟, 但需高速互联(NVLink), 通信频繁
流水线并行 PP: 不同层放不同卡  卡0:层0-7 → 卡1:层8-15 → 卡2:层16-23
  → 跨机扩展友好, 但有流水线气泡(bubble)
专家并行 EP: MoE 把不同专家放不同卡 → 适配 MoE 大模型(Mixtral/Qwen-MoE)
```

实践中常 **TP（机内）+ PP（跨机）** 组合：机内用 NVLink 跑 TP（通信快），跨机用 PP（通信少）。

---

## 数值例子：显存与吞吐手算

### 例 1：单请求 KV Cache 占用（Llama-2 7B）

参数（约值）：层数 $n_{layer}=32$，注意力头 $n_{head}=32$，$d_{head}=128$，隐藏维 $H = 32\times128 = 4096$，FP16（2 字节）。

单 token 的 KV Cache（K 和 V，每层）：

$$
2 \times n_{layer} \times H \times 2\text{B} = 2 \times 32 \times 4096 \times 2 = 524{,}288 \text{ B} \approx 0.5 \text{ MB/token}
$$

一个 2048 token 的请求：

$$
0.5 \text{ MB} \times 2048 \approx 1024 \text{ MB} = 1 \text{ GB}
$$

**结论**：一条 2048 上下文请求就要 1GB KV！一张 80GB A100 跑 7B（权重约 14GB FP16），剩约 60GB，朴素按 max_len 预留只能放 ~60 条并发；用 INT8 KV 量化（0.25MB/token）可翻倍到 ~120 条——这就是**分页 + 量化**省显存的直接价值。

### 例 2：Decode 为何 memory-bound

7B 模型 decode 一个 token，要把约 14GB（FP16 权重）从显存读一遍。A100 显存带宽约 2 TB/s：

$$
t_{mem} = \frac{14\text{ GB}}{2\text{ TB/s}} = 7 \text{ ms}
$$

而其算力（FLOPs）远远算不满这点时间——说明瓶颈在**访存而非算力**。

**推论**：batch 越大越划算！batch=1 和 batch=32 都只读一遍权重（14GB），但 batch=32 一次产 32 个 token，单 token 摊薄成本骤降——这正是**连续批处理把 GPU 喂饱**的吞吐来源。

### 例 3：连续批处理吞吐增益（场景化）

设静态批处理因长度不齐、槽位空转，平均利用率约 30%；连续批处理把利用率拉到约 80%：

$$
\text{吞吐增益} \approx \frac{80\%}{30\%} \approx 2.7\times
$$

叠加分页 KV 提升的并发数，端到端吞吐相对朴素实现常见 **数倍** 提升（实际数字依模型/负载而变，以实测为准）。

---

## 对照表：RTP-LLM vs vLLM vs TensorRT-LLM

| 维度 | **RTP-LLM** | **vLLM** | **TensorRT-LLM** |
|------|-------------|----------|------------------|
| 出品方 | 阿里巴巴 | UC Berkeley → 社区 | NVIDIA |
| 底层 kernel | FT + TRT-LLM kernel + FA2 + cutlass | 自研 + FA + 自定义 CUDA | TensorRT 编译 + NVIDIA 深度优化 |
| 连续批处理 | ✅（参考 vLLM） | ✅（首创/标杆） | ✅（in-flight batching） |
| 分页 KV | ✅ | ✅（PagedAttention 发源地） | ✅ |
| 易用性 | 较好，HF 权重直接加载 | 很好，社区生态最大 | 需先编译 engine，较重 |
| 定位 | 阿里内部在线 serving 场景打磨 | 通用、社区活跃、迭代最快 | 极致单卡性能（绑定 NVIDIA） |
| 硬件绑定 | 相对灵活 | 灵活，多后端 | 强绑 NVIDIA GPU |
| 投机采样 | ✅（集成 Medusa） | ✅ | ✅ |
| 多模态 | ✅（llava/qwen-vl） | ✅ | 部分支持 |
| 上手成本 | 中 | 低 | 高（编译期长） |

**一句话选型**：
- 要**社区最活跃、上手最快、模型支持最全** → **vLLM**。
- 要**纯 NVIDIA 卡上榨干极致性能**、能接受编译流程 → **TensorRT-LLM**。
- 在**阿里生态 / 中文大模型（Qwen 系）/ 内部 serving 平台**落地，想要兼顾性能与工程集成 → **RTP-LLM**。

---

## 何时选 RTP-LLM（适用场景）

```
✅ 适合:
  • 阿里云 / 阿里内部生态, 对 Qwen 系模型适配好
  • 在线低延迟 serving (实时预测平台, RTP 本名所指)
  • 想要 FT/TRT-LLM kernel 的可靠性能, 又要 vLLM 式调度
  • 需要多模态(图文)、投机采样、量化等生产特性齐全

⚠️ 谨慎/可选其他:
  • 只想快速 demo / 试新模型 → vLLM 社区支持更即时
  • 极致追求单卡基准跑分且全 NVIDIA → TRT-LLM
  • 非主流硬件 / 非主流模型 → 先确认其支持矩阵
```

---

## 关键权衡（trade-off 速记）

| 想要 | 用什么机制 | 代价 |
|------|-----------|------|
| 高吞吐 | 连续批处理 + 大 batch | TPOT 可能略升、调度复杂 |
| 省显存/高并发 | 分页 KV + 量化 | 量化掉精度、需校准 |
| 低首 token 延迟 | P-D 分离 + prefix caching | 多一次 KV 传输、架构变重 |
| Decode 加速 | 投机采样 | 草稿头/小模型额外开销 |
| 放下大模型 | TP/PP/EP | 通信开销、需高速互联 |

---

## 常见问题（FAQ）

| 问题 | 澄清 |
|------|------|
| RTP-LLM 是从零写的吗？ | 不是。以 FasterTransformer 为主底座，集成 TRT-LLM 部分 kernel，借鉴 vLLM 的连续批处理/增量解码、transformers 的采样。 |
| 它和 vLLM 是竞争还是借鉴？ | 既借鉴（continuous batching、increment decoding 参考 vLLM）也定位不同（RTP-LLM 更偏阿里在线 serving 工程落地）。 |
| "RTP" 是什么缩写？ | Real-Time Prediction，源自阿里实时预测平台，强调在线低延迟服务。 |
| 连续批 vs 静态批？ | 静态批必须等整批最长请求跑完；连续批每次迭代重组 batch，完成的踢出、等待的插入，槽位不空转。 |
| 分页 KV 解决什么？ | 消除 KV Cache 的内部+外部碎片，提升显存利用率与并发，并支持前缀共享。 |
| 投机采样会改变输出分布吗？ | 不会。验证步保证分布与大模型直接采样等价，是无损加速。 |
| 具体 CLI / API 怎么写？ | 演进快，以官方仓库 README 为准，本文只讲稳定机制。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，从这里找到所有推理引擎主题
- [[llm-inference/vllm/README]] — vLLM 详解（连续批处理 / PagedAttention 的发源地，RTP-LLM 调度思想之源）
- [[llm-inference/README]] — LLM 推理总览（Prefill/Decode、KV Cache、量化、并行的系统介绍）
- 官方仓库：https://github.com/alibaba/rtp-llm

> 学习路径建议：先读 [[llm-inference/README]] 建立"推理优化全景图" → 精读 [[llm-inference/vllm/README]] 吃透连续批处理与分页 KV → 回到本文理解 RTP-LLM 如何"缝合 kernel + 调度"并对比选型。
