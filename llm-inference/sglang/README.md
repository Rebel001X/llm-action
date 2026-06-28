# SGLang 推理引擎

> 一句话定位：SGLang 是一个把「**前端结构化生成语言**」和「**后端高吞吐运行时**」合二为一的 LLM 推理引擎，核心杀手锏是 **RadixAttention**——用一棵基数树（Radix Tree）自动复用 KV Cache 前缀，让多轮对话 / few-shot / 共享系统提示词的场景免重算。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/vllm/README]] [[llm-inference/GuidedGeneration]] [[llm-inference/README]]

---

## 阅读地图

| 你想知道 | 看哪一节 |
| --- | --- |
| SGLang 到底解决什么痛点 | §0 锚点、§1 地基 |
| KV Cache 为什么能复用、PagedAttention 漏了什么 | §2 |
| RadixTree 前缀缓存怎么自动命中（核心） | §3 + §3.1 手算 |
| 连续批 / 调度器怎么和缓存配合 | §4 |
| 结构化生成（JSON/正则/语法）怎么压缩状态机 | §5 |
| 整体架构、组件怎么连起来 | §6 |
| 一个端到端的显存 / 命中率数值例子 | §7 数值例子 |
| 和 vLLM / TensorRT-LLM 怎么选 | §8 对照表 + §9 何时选 |
| 踩坑 / 常见疑问 | §10 常见问题 |

---

## 0. 一句话锚点

把一次 LLM 服务请求拆成两层来看：

- **前端（Frontend）**：你怎么"描述"一个生成程序——多轮对话、并行分支、强制输出 JSON。SGLang 提供一套 DSL（`gen`、`select`、`fork` 等原语）。
- **后端（Backend / Runtime）**：怎么把成千上万条这样的请求**塞进一张 GPU** 跑得又快又省显存。这一层的灵魂是 **RadixAttention**。

> **一句话**：vLLM 解决的是"**单条请求内部**的 KV 显存碎片"（PagedAttention 分页）；SGLang 在此之上又解决"**跨请求 / 跨轮次之间**的 KV 重复计算"（RadixAttention 前缀树复用）。两者正交、可叠加。

---

## 1. 地基：它到底解决什么问题

### 1.1 先回忆 KV Cache 是什么（拆到最原子）

自回归生成时，Transformer 每生成一个 token，都要让它去"看"前面所有 token。注意力的核心是：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

其中第 $t$ 个新 token 的 query $q_t$ 要和**前面所有** token 的 key $k_1..k_t$、value $v_1..v_t$ 做运算。如果每生成一个 token 都把前面所有 token 的 $K,V$ 重新算一遍，复杂度是 $O(n^2)$ 的重复劳动。

**KV Cache**：把已经算过的 $k_i,v_i$ 存下来（它们不随后续 token 改变），下一步只算新 token 的 $q,k,v$，再去查缓存。于是单步从"重算整段"降为"算 1 个 + 查表"。

一个 token 的 KV 显存（FP16）：

$$\text{bytes/token}=2\,(K,V)\times L_{\text{layers}}\times H_{\text{kv}}\times d_{\text{head}}\times 2\,(\text{FP16})$$

### 1.2 痛点：缓存只活在"一条请求"里

朴素实现 / 早期框架里，KV Cache 是**请求私有**的：请求结束就丢。可现实里大量请求**共享前缀**：

```
请求A: <长系统提示词 800 token> + "今天天气?"
请求B: <同一长系统提示词 800 token> + "讲个笑话"
多轮对话: 第2轮 = 第1轮全部历史 + 新问题   <- 前缀几乎全重叠
Few-shot: 10 个示例 5000 token 是公共前缀，只有最后一句不同
```

这 800 / 5000 个公共 token 的 $K,V$ 每条请求都**从头算一遍**——纯浪费。SGLang 的洞察：**把这些前缀的 KV 当成可共享的不可变数据，用一棵树管起来，能命中就直接复用**。这就是 RadixAttention。

---

## 2. KV 缓存复用的前提：PagedAttention 给了什么、漏了什么

vLLM 的 **PagedAttention** 把每条请求的 KV Cache 切成固定大小的 **block（页）**，用一张 block table 把逻辑序列映射到物理页——解决了显存碎片，让多请求能挤进同一张卡。

```
逻辑序列(请求A的token) ──block table──> 物理KV池(分页)
  [t0 t1 ... t15][t16 ...]              [page#7][page#3] ...
```

但 PagedAttention 默认按"请求"分配页，**两条请求即使前 800 token 完全相同，也各占各的页**（除非显式做 prefix caching）。SGLang 的 RadixAttention 把"页"组织成**带前缀语义的树**，于是"相同前缀 → 共享同一批物理 KV 页"变成**自动、默认、且能动态淘汰**的行为。

> 关系：RadixAttention ≈ PagedAttention 的分页内存 + **一棵索引前缀的基数树** + **LRU 淘汰**。延伸阅读 [[llm-inference/KV-Cache优化]]。

---

## 3. RadixTree 前缀缓存原理（核心机制）

### 3.1 基数树是什么

**Radix Tree（基数树 / 压缩前缀树 Trie）**：一种"边上带一段 token 序列"的前缀树。普通 Trie 每条边一个 token；基数树把"只有一个孩子的链"压缩成一条边存一段 token，省内存、查得快。

在 SGLang 里：

- **键（key）**：token 序列（不是字符，是 token id）。
- **值（value）**：这段 token 对应的 **KV Cache 张量指针**（指向分页 KV 池里的物理页）。
- **一条从根到某节点的路径** = 一个被缓存的前缀，其 KV 已经算好。

```
                       root
                        │  "You are a helpful assistant. "   <- 公共系统提示词(已算KV)
                        ▼
                     ┌──●──┐
       "今天天气?"   │     │  "讲个笑话"
                     ▼     ▼
                   ●(A)   ●(B)        A、B 共享上面那段前缀的KV，
                                       各自只新算分叉后的部分
```

### 3.2 一条请求来了，发生什么（match → 复用 → 插入）

```
新请求 prompt = [t0 t1 t2 ... tN]
        │
        ▼
① 从 root 沿树做最长前缀匹配(prefix match)
   命中到节点P：t0..tk 这段KV已经在缓存里 ──> 直接复用，跳过prefill计算
        │
        ▼
② 未命中的后缀 t(k+1)..tN：正常做 prefill，算出KV
        │
        ▼
③ 把新算出的后缀 token 作为新边/新节点 insert 回树
   该路径的引用计数+1(说明有请求正在用它，不能淘汰)
        │
        ▼
④ 进入 decode：每生成一个token，沿当前路径继续延长，
   生成的KV也挂到树上(后续相同对话历史可复用)
        │
        ▼
⑤ 请求结束：引用计数-1。计数归0的节点变为"可淘汰"候选
```

**为什么这样设计**：

- **自动**：用户不用声明"这段是公共前缀"，匹配是引擎按 token 算的，天然适配系统提示词、多轮历史、few-shot。
- **不可变性保证正确**：被匹配复用的前缀，其 $K,V$ 只依赖该前缀本身，不依赖后面会接什么——所以共享绝对安全，不会污染。
- **LRU 淘汰**：显存满了，按"最近最少使用 + 引用计数为 0"淘汰叶子节点（释放物理 KV 页）。常被复用的系统提示词会一直"热"在树里。

### 3.3 与连续批 / 调度的耦合点

调度器在挑下一批请求时，会**优先成批那些命中同一前缀的请求**（cache-aware scheduling），命中率越高 → prefill 省得越多 → 吞吐越高。详见 §4。

### 3.4 命中率手算例子

设系统提示词 $P=800$ token，用户问句平均 $U=40$ token。100 个并发请求共享同一个 $P$：

- 朴素（无前缀复用）：prefill 总 token $=100\times(800+40)=84000$。
- RadixAttention：$P$ 只算一次 $=800$，其余 99 条命中前缀，只算各自 $U$。总 $=800+100\times40=4800$。

prefill 计算量降为原来的 $\dfrac{4800}{84000}\approx \mathbf{5.7\%}$，即 prefill 阶段约 **17.5×** 的节省（理论上限，实际受批大小、调度命中顺序影响）。

---

## 4. 连续批（Continuous Batching）与零开销调度

### 4.1 为什么需要连续批

静态批（static batching）：把 N 条请求凑一批一起跑，**必须等最长的那条生成完**，短请求早早算完却干等——GPU 大量空转。

**连续批 / in-flight batching**：以"**迭代（一次生成 1 个 token 的前向）**"为粒度调度。某条请求生成到 EOS 就立刻退出腾出槽位，新请求随时插入填补，GPU 一刻不闲。

```
时间步 →
请求    s0   s1   s2   s3   s4   s5
A      [██] [██] [EOS]                 A第2步就结束，槽位让出
B      [██] [██] [██] [██] [EOS]
C      [██] [██] [██] [██] [██] ...
D(新)            [██] [██] [██] ...    D在s2插进A空出的槽
       └──── 每一列=一次前向，批内成员动态变化 ────┘
```

### 4.2 SGLang 的"零开销批调度器"

朴素连续批每一步都有 CPU 侧开销：构造下一批的元数据、采样参数、attention 的索引（哪些 token 属于哪条请求）。这些 CPU 工作若**串行**夹在 GPU 前向之间，GPU 会被 CPU 拖住出现"气泡"。

SGLang v0.4 的做法（稳定思想，细节以官方文档为准）：把**下一步的批准备工作和当前步的 GPU 前向重叠（overlap）**——CPU 在 GPU 算第 $i$ 步时就把第 $i{+}1$ 步的调度元数据备好。于是 CPU 开销被"藏"进 GPU 计算的影子里，趋近零开销。

```
传统:  GPU[fwd s0]  CPU[prep s1]  GPU[fwd s1]  CPU[prep s2] ...   <- 串行，有气泡
重叠:  GPU[fwd s0][fwd s1][fwd s2] ...  (连续不断)
       CPU      [prep s1][prep s2] ...  (与GPU并行，藏在影子里)
```

### 4.3 Prefill / Decode 的取舍

一次迭代里既可能有"新请求做 prefill（一次吃整段 prompt）"，也有"老请求做 decode（一次 1 token）"。两者算力特征不同：prefill 是计算密集，decode 是访存密集。SGLang 用 **chunked prefill**（把长 prompt 切块）等手段避免长 prompt 的 prefill 长时间霸占批、饿死 decode。相关分离思路见 [[llm-inference/PD分离]]。

---

## 5. 结构化生成（约束解码）

### 5.1 问题：让模型"只能"吐合法 JSON

很多场景要模型输出**严格合法**的 JSON / 满足正则 / 符合某个语法。靠 prompt 求着模型"请输出 JSON"不可靠。**约束解码**的本质：在每一步采样时，把"会让输出变非法"的 token 的概率**直接屏蔽（mask 成 $-\infty$）**，只在合法 token 集合里采样。

```
词表 logits:  [ ... 5万个token的分数 ... ]
                        │
                  应用一个mask(0/1向量): 1=此刻语法允许, 0=禁止
                        ▼
合法集合:     [ '"'  '{'  数字 ... ]   <- 只在这些里softmax+采样
非法token直接 -inf，采样概率=0 → 输出100%合法
```

这个 mask 从哪来？把约束（JSON Schema / 正则 / EBNF 语法）**编译成一个有限状态机（FSM）/ 下推自动机**。当前处于状态 $s$，FSM 告诉你"哪些 token 能让状态合法地往前走"，这就是当前步的合法集合。每采样一个 token，状态机转移到下一状态。详细原理见 [[llm-inference/GuidedGeneration]]。

### 5.2 SGLang 的加速：压缩 + 跳跃

朴素约束解码每步都要查 FSM、构造 mask，开销不小。SGLang（结合 xgrammar / compressed FSM 等）做了两件事：

- **状态机压缩**：把语法编译成更紧凑的自动机，转移查询更快。
- **Jump-Forward（跳跃前进）**：当 FSM 在某个状态下"只有唯一一条合法路径"（比如 JSON 里 `"name":` 后面必然紧跟若干固定字符），就**不必逐 token 让模型采样**，直接把这串确定的 token 一次性填进去，跳过若干次前向。

```
生成 {"name": "...   一旦决定了key是"name"，
冒号、引号这些"语法上唯一确定"的字符直接jump填入，
不浪费GPU去逐个采样 → 结构化输出更快
```

> 这与 RadixAttention 正交：约束解码管"采样选哪个 token"，RadixAttention 管"已算的 KV 复不复用"。

---

## 6. 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                      SGLang Frontend (DSL)                     │
│  gen / select / fork / 多轮 / 并行分支；可接 OpenAI 兼容 API   │
└───────────────────────────────┬──────────────────────────────┘
                                 │  请求 (prompt + 采样/约束参数)
                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                    Runtime / Scheduler                         │
│  ① Tokenizer  ② Cache-aware 调度(优先成批命中同前缀的请求)     │
│  ③ 连续批 + 零开销重叠调度  ④ chunked prefill                  │
└───────┬───────────────────────────────────┬──────────────────┘
        │ 查/插前缀                          │ 约束步进
        ▼                                    ▼
┌─────────────────────┐            ┌──────────────────────────┐
│   RadixAttention    │            │  Constrained Decoding    │
│  Radix Tree(前缀)   │            │  FSM/语法 + jump-forward  │
│      │ 指向         │            └──────────────────────────┘
│      ▼                                       │
│  Paged KV Cache 池(物理显存, LRU淘汰)        │
└───────────────────────┬──────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│   Attention Kernel 层 (FlashInfer / Triton kernels)，          │
│   TP/PP 并行，量化(FP8/AWQ/GPTQ)，CUDA Graph                   │
└──────────────────────────────────────────────────────────────┘
```

各层一句话：

| 层 | 职责 | 关联笔记 |
| --- | --- | --- |
| Frontend DSL | 描述"生成程序"：多轮、并行、约束 | — |
| Scheduler | 连续批 + cache-aware 成批 + 零开销重叠 | [[llm-inference/PD分离]] |
| RadixAttention | 前缀树索引 + 分页 KV + LRU | [[llm-inference/KV-Cache优化]] |
| 约束解码 | FSM mask + jump-forward | [[llm-inference/GuidedGeneration]] |
| Kernel/并行 | FlashInfer/Triton、TP、量化、CUDA Graph | [[llm-inference/FlashInfer]]、[[llm-inference/大模型推理张量并行]] |

---

## 7. 数值例子 / 典型场景

**场景**：在 1 张 80GB GPU（如 A100/H100）上服务 Llama-3-8B（FP16），做"统一系统提示词 + 多轮客服对话"。

**Step 1：模型权重占用**
8B 参数 × 2 字节（FP16）≈ $16$ GB。

**Step 2：单 token KV 显存**（Llama-3-8B：$L=32$ 层，GQA 把 KV 头压到 $H_{kv}=8$，$d_{head}=128$）：

$$2\times 32\times 8\times 128\times 2\ \text{B}=131072\ \text{B}\approx 0.125\ \text{MB/token}$$

**Step 3：KV 可用预算**
80 − 16（权重）− 约 4（激活/CUDA graph 等开销）≈ $60$ GB 给 KV。

$$\text{可缓存 token 数}=\frac{60\times 1024\ \text{MB}}{0.125\ \text{MB}}\approx 4.9\times 10^5\ \text{token}$$

**Step 4：前缀复用带来的"等效扩容"**
设系统提示词 $P=1000$ token，1000 个活跃对话各自历史平均 $H=1500$ token、共享同一 $P$。

- 无前缀复用：需缓存 $1000\times(1000+1500)=2.5\times10^6$ token → **超出预算 5 倍，必须频繁换出/重算**。
- RadixAttention：$P$ 只存一份 $=1000$ token，历史部分若多轮间高度重叠，实际驻留远低于理论值；公共前缀这部分直接省掉 $1000\times1000=10^6$ token 的存储与重算。

**结论**：共享前缀越长、并发越高，RadixAttention 的"等效显存放大"和吞吐收益越夸张——这正是 SGLang 在 agent/few-shot/多轮 benchmark 上相对优势最大的原因。（以上为机制级估算，具体数字随版本、量化、kernel 实现变化，以官方 benchmark 为准。）

---

## 8. 对照表（与同类对比）

| 维度 | **SGLang** | **vLLM** | **TensorRT-LLM** |
| --- | --- | --- | --- |
| KV 显存管理 | 分页 + **RadixTree 前缀树自动复用** | PagedAttention 分页（prefix caching 可选/后加入） | 自有 KV 管理 + paged |
| 跨请求前缀复用 | **默认、自动、带 LRU** 的核心特性 | 需开启 prefix caching | 支持，配置较重 |
| 连续批 | 是 + **零开销重叠调度器** | 是（连续批的代表实现） | 是（in-flight batching） |
| 结构化生成 | **一等公民**（FSM + jump-forward，很快） | 支持（outlines/xgrammar 等） | 支持 |
| 前端编程模型 | **DSL**（fork/select/多轮），表达力强 | 偏 API/server | 偏 API/编译产物 |
| 易用性 | Python，装即用 | Python，装即用、生态最大 | 需**编译 engine**，门槛高 |
| 极致单卡延迟 | 强 | 强 | **通常最快**（深度 kernel 融合） |
| 量化 | FP8/AWQ/GPTQ 等 | 全面 | 全面，编译期固化 |
| 适配硬件 | NVIDIA 为主，亦扩展（见 ascend 等） | 多后端 | **仅 NVIDIA** |

参考对比博客见底部链接（SGLang vs TensorRT-LLM/vLLM 的 Llama3 serving）。延伸 [[llm-inference/vllm/README]]、[[llm-inference/tensorrt-llm]]。

---

## 9. 何时选 SGLang（决策清单）

**强烈倾向 SGLang：**
- 工作负载有**大量共享前缀**：长系统提示词、RAG 固定模板、few-shot、**多轮对话**、**Agent 树状/并行调用**——RadixAttention 收益最大。
- 需要**高质量结构化输出**（严格 JSON / 工具调用参数 / 受限语法），且要快。
- 想用**前端 DSL** 优雅地写并行分支 / 多次采样 / self-consistency。

**可能更适合别的：**
- 追求**单条请求极致最低延迟**、且愿意付出编译成本 → 看 **TensorRT-LLM**。
- 想要**最大社区生态 / 最广模型与后端覆盖**、对前缀复用不敏感 → **vLLM** 同样优秀（且 vLLM 也在补 prefix caching）。
- 非 NVIDIA 异构平台为主 → 评估各引擎的硬件后端支持。

> 经验法则：**前缀重叠度高 + 要结构化输出 + 多轮/Agent → SGLang**；**单请求极致延迟 + 可接受编译 → TensorRT-LLM**；**通用、生态、快速上手 → vLLM**。三者机制正交，很多团队按业务分场景混用。

---

## 10. 常见问题

| 问题 | 解答 |
| --- | --- |
| RadixAttention 和 PagedAttention 冲突吗？ | 不冲突。RadixAttention = 分页 KV 之上加一棵前缀树做复用与淘汰，是叠加关系。 |
| 前缀复用会不会串扰、输出被别人污染？ | 不会。前缀的 KV 只由该前缀本身决定，与后续 token 无关，复用绝对安全。 |
| 缓存满了怎么办？ | 按 LRU + 引用计数淘汰：正在被请求使用（引用计数>0）的节点不淘汰，热门系统提示词长期驻留。 |
| 命中率取决于什么？ | 取决于请求间**前缀重叠程度**和**调度是否把同前缀请求成批**（cache-aware scheduling）。无重叠时退化为普通连续批。 |
| jump-forward 会影响生成质量吗？ | 不会。它只对"语法上唯一确定"的 token 跳过采样，本就没有选择空间，纯省算力。 |
| 和 vLLM 比一定更快？ | 看负载。前缀重叠高 / 结构化输出多时 SGLang 优势明显；无共享前缀的纯随机 prompt 上差距收窄。 |
| 具体 CLI 参数 / API 怎么写？ | 各版本有差异，**以官方文档为准**（见底部链接），本文只讲稳定机制。 |
| 量化怎么配？ | 支持 FP8/AWQ/GPTQ 等，见官方 quantization 文档（底部链接）。 |

---

## 🔗 跳转链接

**本仓库内（双链）**
- [[00-知识地图]]
- [[llm-inference/README]]
- [[llm-inference/vllm/README]] — 同类引擎对比基准（PagedAttention）
- [[llm-inference/GuidedGeneration]] — 约束/引导解码原理详解
- [[llm-inference/KV-Cache优化]] — KV Cache 复用与压缩
- [[llm-inference/PD分离]] — Prefill/Decode 分离调度
- [[llm-inference/FlashInfer]] — SGLang 常用的 attention kernel 后端
- [[llm-inference/大模型推理张量并行]] — TP 并行
- [[llm-inference/tensorrt-llm]] — 对比对象之一

**官方 / 外部资料**
- SGLang 后端代码解析（中文）：https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md
- 学习材料合集：https://github.com/sgl-project/sgl-learning-materials
- RadixAttention 原理博客：https://lmsys.org/blog/2024-01-17-sglang/
- vs TensorRT-LLM / vLLM（Llama3 serving）：https://lmsys.org/blog/2024-07-25-sglang-llama3/
- v0.3（7x DeepSeek MLA）：https://lmsys.org/blog/2024-09-04-sglang-v0-3/
- v0.4（零开销批调度 / cache-aware 负载均衡 / 更快结构化输出）：https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- 量化文档：https://docs.sglang.ai/backend/quantization.html
- 镜像：https://hub.docker.com/r/lmsysorg/sglang/tags ；`docker pull lmsysorg/sglang:v0.4.5-cu125`
- Qwen3 关闭 think：https://qwen.readthedocs.io/zh-cn/latest/deployment/sglang.html
