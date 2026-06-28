# ORCA：面向 Transformer 生成式模型的分布式推理服务系统（论文精读）

> 一句话定位：ORCA（OSDI'22）用「迭代级调度（continuous/in-flight batching）」+「选择性批处理（selective batching）」把"按请求批"改成"按 token 步批"，是现代 LLM 推理服务（vLLM、TGI、TensorRT-LLM 的连续批处理）的思想源头。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/连续批处理]] · [[llm-inference/vllm/README]] · [[llm-optimizer/kv-cache]] · [[llm-inference/PD分离]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|--------------|--------|
| 0 一句话锚点 | ORCA 到底改了什么 | iteration-level scheduling |
| 1 地基 | 自回归生成 + 为什么传统 batch 服务效率低 | prefill / decode / KV cache |
| 2 问题诊断 | request-level batching 的两大病灶 | early-finish 等待、late-arrival 排队 |
| 3 核心一 | 迭代级调度（continuous batching） | 每步重排 batch |
| 4 核心二 | 选择性批处理（selective batching） | Attention 不能 batch、Linear 能 batch |
| 5 系统结构 | Request Pool / Scheduler / Engine | 控制面 vs 数据面 |
| 6 调度算法 | FCFS + max-batch + 内存预留 | reserve KV slots |
| 7 并行 | 层内（张量并行）+ 层间（流水并行） | NCCL / 分布式 |
| 公式/算法/数值 | 吞吐手算 + 算法伪码 | 36.9× 的来源 |
| 评价/局限 | 与 vLLM / FT 对照 | 内存碎片是 ORCA 的遗留问题 |

## 0. 一句话锚点

> **ORCA 把推理服务的调度粒度从「一个请求（整段生成）」降到「一次迭代（生成 1 个 token）」**，于是先完成的请求能立刻离场、刚到的请求能立刻入场，GPU 几乎不再空转。这就是今天人人都说的 **continuous batching / in-flight batching**。其代价是 Attention 算子无法跨请求批处理，ORCA 用 **selective batching** 解决：只把能批的算子（QKV/FFN 等 Linear）批起来，Attention 拆开单独算。

---

## 1. 地基：自回归生成为什么难"批"

### 1.1 生成是「多迭代」的，分类/embedding 是「单迭代」的

Transformer 生成式模型（GPT 类）产出一段文本不是一次算完，而是**逐 token 迭代**：

```
输入 prompt:  "AI infra is"
迭代1 (prefill): 一次吃完 prompt 全部 token, 算出第1个新 token -> " the"
迭代2 (decode):  喂入 " the",             算出第2个新 token -> " future"
迭代3 (decode):  喂入 " future",          算出第3个新 token -> " of"
...
迭代k:          直到生成 <EOS> 或达到 max_len 才停
```

关键事实：**一个请求要跑很多次模型前向（迭代），而每次迭代生成的 token 数 = 1（decode 阶段）**。这与图像分类/句子分类（一次前向出结果）截然不同 —— 这正是传统服务系统"水土不服"的根因。

- **prefill（预填充）阶段**：第 1 次迭代，并行处理 prompt 的全部 $L$ 个 token，是 **compute-bound**（算力受限）。
- **decode（解码）阶段**：之后每次迭代只处理 1 个 token，是 **memory-bound**（访存受限，要反复读权重+KV）。

### 1.2 KV Cache：让 decode 从 $O(L^2)$ 降到 $O(L)$

Attention 中每个新 token 都要和**前面所有 token** 的 Key/Value 做注意力。若每步都重算所有历史 token 的 K/V，复杂度爆炸。**KV Cache** 把每个 token 算过的 K、V 存下来，decode 时只算当前这 1 个新 token 的 Q/K/V，再与缓存拼接：

```
            ┌───────────── KV Cache (随生成增长) ─────────────┐
迭代t:  Q_t ·[ K_1 K_2 ... K_{t-1} | K_t ]   V_1...V_{t-1}|V_t  -> out_t
              └── 从缓存读 ──┘  └新算┘
```

**代价**：KV Cache 占显存，且大小 = `2 · 层数 · 序列长 · hidden · dtype字节`，**每个请求各占一份、长度还在变** —— 这使"把不同请求塞进一个 batch"变得棘手（见第 4 节）。详见 [[llm-optimizer/kv-cache]]。

---

## 2. 问题诊断：request-level（请求级）批处理的两大病灶

传统系统（如 NVIDIA FasterTransformer、Triton）以**整个请求**为调度单位：选一批请求 → 跑完整段生成 → 整批一起返回 → 再取下一批。把"一次迭代"画成一格：

```
请求级批处理 (request-level batching):  ['#'=在算  '.'=空转/等待]

时间 →  it1  it2  it3  it4  it5  it6
ReqA   [#]  [#]  [#]  [#]  [#]  [#]   <- A 要 6 步
ReqB   [#]  [#]  [.]  [.]  [.]  [.]   <- B 第2步就出 EOS, 但被锁在 batch 里空转 4 步!
ReqC                              ↑ C 在 it1 就到了, 却要等整批(到it6)结束才能进
```

- **病灶① early-finish 等待（短请求被长请求拖累）**：同一 batch 内各请求生成长度不同。B 第 2 步就完成了，但整批必须等最慢的 A 跑到第 6 步才返回，B 白白占着 GPU/显存空转，**且结果迟迟不返回给客户端**（延迟变大）。
- **病灶② late-arrival 排队（新请求进不来）**：C 在 it1 就到达，但调度以请求为单位，只能等当前整批全部结束才放新请求进来 —— **排队延迟**。

> 直觉：request-level batching = "公交车必须等全车人都到站才开门，中途上不了人、提前到站的人也下不了车"。ORCA 要改成"每一站都能上下客"。

---

## 3. 核心创新一：迭代级调度（Iteration-level Scheduling）

**把调度粒度从"一个请求"降到"一次迭代"**：执行引擎每跑完**一次迭代**（即给当前 batch 里每个请求各生成 1 个 token）就**交回控制权给调度器**，调度器立刻：
1. 把这一步**已生成 EOS / 达上限**的请求**移出** batch、返回客户端；
2. 把**新到达**的请求**加入**下一步的 batch；
3. 组成新 batch，再让引擎跑下一次迭代。

```
迭代级调度 (iteration-level / continuous batching):

时间 →     it1   it2   it3   it4   it5
slot0     [A]   [A]   [A]   [A]   [A]
slot1     [B]   [B]→done!→[C]  [C]   [C]   <- B 第2步完成立即离场, C 立刻补位
slot2     [D]   [D]   [D]→done→[E]  [E]   <- 槽位"流水线"式滚动, GPU 不空转
                  ↑ 每跑1步就重排, 不必等整批结束
```

效果：
- **短请求即时离场** → 不再空转、结果立即返回（尾延迟↓）。
- **新请求即时补位** → 排队延迟≈一步迭代时间（毫秒级），而非"等整批"。
- **GPU 利用率拉满** → batch 槽位持续被填满，吞吐↑。

这就是后来 vLLM/TGI/TensorRT-LLM 所称的 **continuous batching / in-flight batching**，是 ORCA 最被广泛继承的思想。

> 为什么以前没人这么做？因为"每步重排 batch"会暴露一个硬骨头：**同一个 batch 里不同请求的序列长度/阶段不同，Attention 算子没法直接批**（见下节）。ORCA 的第二个创新就是为此而生。

---

## 4. 核心创新二：选择性批处理（Selective Batching）

### 4.1 矛盾：迭代级调度后，batch 里的请求"形状不齐"

每步重排后，一个 batch 可能同时含：
- 处于 **prefill** 的请求（要处理 prompt 的 $L$ 个 token，张量是 `[L, hidden]`）；
- 处于 **decode** 的请求（只处理 1 个 token，张量是 `[1, hidden]`）；
- 各请求**历史长度不同**（KV Cache 长度 $t$ 各异）。

对**逐 token 独立**的算子（Linear/LayerNorm/FFN/逐元素），只要把所有 token 在 batch 维拼起来照算即可，**与"谁属于哪个请求、历史多长"无关**。但 **Attention 不行**：它要让"本请求的 Q"只去注意"本请求自己的 K/V 历史"，绝不能跨请求注意。若强行 batch，要么形状对不齐（不同 $t$），要么注意力串味。

### 4.2 解法：能批的批，不能批的拆

ORCA 对一次迭代的算子做**分类处理**：

```
            ┌──────────────── 一个迭代内的算子流 ────────────────┐
 多请求 token 全部拉平拼成一个大张量 [Σtokens, hidden]
        │
   ┌────┴─── 能 batch 的算子 (与请求无关, 逐token独立) ───┐
   │  QKV 投影(Linear) / LayerNorm / FFN(两层Linear) /    │  <- 整体一把算, GEMM 大而高效
   │  残差 / 激活 ... 全部按 [Σtokens, hidden] 批处理        │
   └──────────────────────────────────────────────────────┘
        │
   ┌────┴─── 不能 batch 的算子 (Attention) ───────────────┐
   │  Split: 把大张量按"每个请求"切回各自的 token         │
   │  for 每个请求 i:  Attn(Q_i, [KVcache_i; K_i], ...)     │  <- 逐请求单独算注意力
   │  Merge: 算完再拼回 [Σtokens, hidden]                  │
   └──────────────────────────────────────────────────────┘
        │
   继续走能 batch 的输出投影 / FFN ...
```

- **Linear/FFN/LayerNorm 等**：把全 batch 所有 token 拍平成一个大矩阵做 GEMM，**矩阵越大越能喂饱 GPU**，效率高（这是 batch 的全部收益所在）。
- **Attention**：**Split → 逐请求各算各的 → Merge**。注意力本身计算量占比不大，逐请求循环的开销可接受，却换来了"任意形状请求可同 batch"的自由。

> 一句话：**selective batching = 把"必须批才划算"的 GEMM 类算子批起来吃满 GPU，把"批不了"的 Attention 按请求拆开算**。它是迭代级调度能落地的"配套地基"。
>
> （后续 vLLM 用 **PagedAttention** 进一步把 Attention 的 KV 显存分页管理，并用专门 kernel 批量处理变长 Attention，相当于把 ORCA 这里的"逐请求循环"也做了高效批化 —— 见 [[llm-inference/vllm/README]]。）

---

## 5. 系统结构：控制面与数据面分离

```
                    ┌──────────────────────────────────────────┐
  客户端请求 ──────▶ │  Request Pool (请求池)                     │
                    │   保存所有在途请求的状态:                  │
                    │   token 序列 / 已生成长度 / KV slot 句柄    │
                    └───────────────┬──────────────────────────┘
                                    │ 选请求(每次迭代)
                          ┌─────────▼──────────┐
                          │  Scheduler 调度器   │  控制面 (CPU)
                          │  - FCFS 选 batch    │  每"迭代"运行一次:
                          │  - 预留 KV 内存      │   1. 挑请求组 batch
                          │  - 回收完成请求      │   2. 下发给 Engine
                          └─────────┬──────────┘   3. 收回结果, 更新池
                                    │ 下发一个 batch
              ┌─────────────────────▼─────────────────────────┐
              │  Execution Engine 执行引擎 (数据面, GPU 集群)    │
              │  跨多 GPU/多机: 层内并行(张量) + 层间并行(流水)   │
              │  跑"一次迭代": selective batching 前向, 出 next  │
              │  token logits -> 采样 -> 返回每个请求的新 token  │
              └────────────────────────────────────────────────┘
```

- **Request Pool**：所有在途请求的"档案"，含已生成 token、当前长度、KV Cache 占用。
- **Scheduler（调度器）**：每次迭代被调用一次，决定"这一步谁进 batch"，并负责**为新请求预留 KV 显存**、回收完成请求的显存。它是 ORCA 把粒度做细的关键控制点。
- **Execution Engine（执行引擎）**：真正在 GPU 上跑前向的"数据面"，内部按下一节做并行。控制面（轻量、CPU）与数据面（重、GPU）解耦，让"每步重排"的开销可控。

---

## 6. 调度算法：FCFS + 内存预留

调度器每一步要回答两件事：**选谁** + **能不能塞得下**。论文给出的策略要点：

1. **FCFS（先到先服务）选请求**：按到达顺序从请求池取请求，尽量填满到 `max_batch_size`。
2. **KV 内存预留（reservation）**：一个请求被选入运行集后，要为它**未来可能生成的最大长度**预留 KV Cache 显存槽位，避免跑到一半显存爆掉（OOM）导致请求被迫中断。**只有当剩余 KV 显存够预留时，新请求才被允许加入**，否则继续排队。
3. **完成即回收**：请求出 EOS / 到上限后，立即释放其 KV 槽位，供新请求复用。

> 这里埋下了 ORCA 的**局限**：按"最大长度"**预留**会**高估**实际占用（多数请求不会生成到最大长度），造成显存**预留性浪费 + 碎片**。vLLM 的 **PagedAttention** 正是针对这一点 —— 按需分页、几乎零浪费（见局限表）。

伪代码（迭代级调度主循环，简化）：

```
loop forever:                              # 服务进程常驻
    batch = []
    for req in request_pool by FCFS:       # 选请求
        if len(batch) == max_batch: break
        if not enough_KV_to_reserve(req): continue   # 显存不够则跳过, 留在池里
        reserve_KV(req); batch.append(req)
    outs = engine.run_one_iteration(batch) # 只跑"一步": selective batching 前向
    for req, tok in zip(batch, outs):
        req.append(tok)                    # 记录新 token
        if tok == EOS or req.full():
            emit_result(req); free_KV(req) # 完成: 返回客户端 + 释放显存
        # 未完成的请求自动留在池里, 下一轮继续被选(滚动批处理)
```

---

## 7. 分布式并行：层内 + 层间

为支撑 GPT-3 量级（百亿~千亿参数，单卡装不下）的模型，ORCA 在执行引擎里用两类模型并行：

```
        层间并行 (inter-layer / 流水线并行 Pipeline)
        把不同 Transformer 层切到不同 GPU 组
   GPU组0 [L0~L7] ──激活──▶ GPU组1 [L8~L15] ──▶ GPU组2 [L16~L23] ─▶...
        │
        └─ 每个 GPU 组内部再做:
           层内并行 (intra-layer / 张量并行 Tensor Parallel)
           把一层(QKV/FFN 的大矩阵)按列/行切到组内多卡, NCCL all-reduce 合并
   GPU0  GPU1  GPU2  GPU3   <- 一个张量并行组, 协同算"一层"
```

- **层内并行（张量并行）**：单层的大 GEMM（注意力的 QKV、FFN）按维度切分到多卡，靠集合通信（all-reduce）合并结果。延迟敏感、要求高带宽（NVLink/同机）。见 [[B07:llm-inference/大模型推理张量并行]]。
- **层间并行（流水线并行）**：不同层放不同卡，激活在卡间流动。ORCA 的迭代级调度天然能让流水线**保持忙碌**（连续有迭代喂入），缓解流水线"气泡"。
- 通信基于 NCCL 等集合通信原语（all-reduce/all-gather），见 [[ai-infra/网络/集合通信原语]]。

> 要点：**迭代级调度 + 模型并行是正交且互补的** —— 前者解决"批怎么组、何时进出"，后者解决"单步前向怎么在多卡上跑"。

---

## 关键公式 / 算法 / 数值示例

### (1) KV Cache 显存（理解"为什么要预留、为什么会浪费"）

单个请求的 KV Cache 字节数：

$$
\text{KV bytes} = 2 \times n_{\text{layer}} \times s \times d_{\text{model}} \times b_{\text{dtype}}
$$

其中 2 = K 和 V 两份，$s$ = 当前序列长度，$d_{\text{model}}$ = 隐藏维，$b_{\text{dtype}}$ = 每元素字节（fp16=2）。

**手算（以 GPT-3 13B 量级近似：$n_{\text{layer}}=40,\ d_{\text{model}}=5120$，fp16）**，单 token 的 KV：

$$
2 \times 40 \times 1 \times 5120 \times 2\,\text{B} = 819{,}200\,\text{B} \approx 0.78\ \text{MB / token}
$$

一个长度 2048 的请求约占 $0.78 \times 2048 \approx 1.6\ \text{GB}$。**调度器若按 max_len=2048 预留**，而该请求实际只生成 100 token（≈0.08 GB），则**预留了约 1.6 GB 却只用 0.08 GB** —— 这就是 ORCA"预留式"管理的浪费来源（数字为量级演示，精确值以模型配置为准）。

### (2) 吞吐近似：为什么迭代级调度能涨这么多

把有效吞吐写成：

$$
\text{Throughput} \approx \frac{\text{batch 中真正在算的有效 token 数}}{\text{每次迭代耗时}} \times \text{GPU 利用率}
$$

request-level 批处理里，因 early-finish 的请求**仍占着 batch 槽位空转**，"有效 token 数 / 槽位数"被严重稀释；迭代级调度让每个槽位**几乎始终在算有效 token**，分子被拉满 → 吞吐数量级提升。

### (3) 论文实测（定性，数字标"约/见原文"）

| 对照 | 结果（约/见原文） |
|------|------|
| ORCA vs NVIDIA FasterTransformer，GPT-3 量级 | **相同延迟水平下吞吐约 36.9×**（论文 Abstract，见原文 Fig.） |
| 增益来源 | 迭代级调度消除空转/排队 + 选择性批处理保持 GEMM 高效 |
| 适用 | 生成长度分布差异大、请求异步到达的在线服务，增益最大 |

> 注：36.9× 是论文报告的标志性数字，具体测试配置（GPU 型号、batch、序列分布）请以原文实验章节为准。

---

## 评价 / 对照 / 局限

| 维度 | ORCA（2022, OSDI） | 后续/对照 |
|------|------|-----------|
| 调度粒度 | **迭代级**（首创 continuous batching） | vLLM/TGI/TRT-LLM 全部继承此思想 |
| 变长请求批处理 | **selective batching**（Attention 拆、其余批） | vLLM 用 PagedAttention + 变长 kernel 做得更彻底 |
| KV 显存管理 | **按 max_len 预留**，有浪费/碎片 | **vLLM PagedAttention**：分页按需分配，碎片≈0 → 更大 batch、更高吞吐 |
| prefill/decode | 同 batch 混跑（chunk 思想未显式提出） | 后续有 **PD 分离**（[[llm-inference/PD分离]]）、chunked prefill 优化 |
| 工程影响 | 奠定现代 LLM 服务范式 | 是 vLLM/TGI/TensorRT-LLM 连续批处理的鼻祖 |

**局限小结**：
1. **KV 内存预留浪费**：按最大长度预留显存，多数请求用不满 → 显存利用率低、限制了最大并发 batch（vLLM 的 PagedAttention 主攻此点）。
2. **Attention 逐请求循环**：selective batching 把 Attention 拆开循环，未做高效变长批 kernel（后续 FlashAttention 变长版 + PagedAttention kernel 解决）。
3. **prefill 与 decode 同批**：长 prompt 的 prefill（compute-bound）混入 decode（memory-bound）batch，会拉高 decode 请求的尾延迟（chunked prefill / PD 分离后续优化）。
4. **公平性/抢占**：FCFS 简单但对长请求"霸占"槽位的公平性、抢占机制讨论有限。

**对工程的启示**：
- 做 LLM 推理服务，**continuous batching 是第一性收益**，先把它做对，再谈算子优化。
- **显存（KV Cache）是吞吐的真实瓶颈**：能装下多少并发 = 能跑多大 batch = 吞吐上限。ORCA 告诉你"预留会浪费"，vLLM 告诉你"分页能救"。
- 调度的**控制面要轻**（CPU、毫秒级），否则"每步重排"的开销会吃掉收益。

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 直接相关：[[llm-inference/连续批处理]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]]
- 机制基础：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[llm-algo/transformer/模型架构]]
- 并行/通信：[[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[llm-train/pytorch/distribution/README]]
- 指标语境：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
