# vLLM 压测

> 用 vLLM 自带的 benchmark 脚本，把"延迟、吞吐、并发"三件事量化成可对比的数字，并画出吞吐-延迟曲线来选工作点。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-eval/llm-performance/推理性能测试]] [[llm-inference/vllm/README]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：压测到底在测什么 | 延迟 vs 吞吐的根本矛盾 |
| 1 | 地基：三个脚本、三类指标、为什么要 warmup | latency/throughput/serving |
| 2 | benchmark_serving 全流程拆解 | 客户端、泊松到达、在线 |
| 3 | 请求率 vs 并发：两种"压力旋钮"的本质区别 | request-rate / max-concurrency |
| 4 | TTFT / TPOT / ITL / E2E：每个指标从哪来 | prefill / decode |
| 5 | 吞吐：requests/s 与 tokens/s 怎么算 | goodput |
| 6 | ShareGPT 数据：为什么用它、长度分布的坑 | 真实负载 |
| 7 | 如何画吞吐-延迟曲线（含 ASCII 曲线） | 扫 QPS、找拐点 |
| 8 | 调参方法论（讲思路不背 CLI 默认值） | batch / KV / 并行 |
| 9 | 数值例子 + 手算 + 对照表 | TPOT→tokens/s |
| 10 | 常见问题 | FAQ |

---

## 0. 一句话锚点

压测 LLM 服务，本质是测一个**排队系统**：请求像水流一样进来，GPU 像一个有限带宽的水管。你想知道两件事——**单个请求多快**（延迟）和**单位时间能处理多少**（吞吐）。这两者天然冲突：要吞吐就得攒大 batch，攒 batch 就要等，延迟变大。压测就是把这条权衡曲线画出来，找到"延迟还能接受、吞吐又最高"的那个工作点。

vLLM 用三个脚本覆盖三个场景：

```
benchmark_latency.py      → 固定 batch，单批跑完，测"纯计算延迟"（离线、无排队）
benchmark_throughput.py   → 一次性灌入 N 条请求，测"极限吞吐"（离线、batch 拉满）
benchmark_serving.py      → 模拟客户端按某个速率发请求，测"在线服务"真实体验 ★最重要
```

> ★ 真正贴近生产的是 `benchmark_serving.py`，因为它有"请求按时间陆续到达"这一关键特征，能同时给出延迟分布和吞吐。下文重点讲它。

---

## 1. 地基：三类指标 + 为什么要 warmup

### 1.1 三个原子概念（不假设你记得）

- **prefill（预填充）**：把整段输入 prompt 一次性喂进模型，算出第一个输出 token。这一步是**计算密集**（compute-bound），因为要对全部 prompt token 做一次大矩阵乘。
- **decode（解码）**：之后逐个生成 token，每生成一个就要读一遍全部模型权重 + KV 缓存。这一步是**访存密集**（memory-bound），算得快但被显存带宽卡住。
- **KV Cache**：把每个 token 在每层算出的 Key/Value 存下来，下一步生成时直接复用，避免重算历史。它占的显存随"序列长度 × 并发数"线性增长，是吞吐的真正瓶颈。

```
   一次请求的时间轴
   ┌─────────────┬───────────────────────────────────┐
   │   prefill   │   decode  decode  decode  ...  EOS  │
   └─────────────┴───────────────────────────────────┘
        │              │      │
        │              │      └─ ITL (inter-token latency，token 间隔)
        │              └──────── 每个 decode 步
        └─────────────────────── TTFT 的主要构成（首 token 时间）
```

### 1.2 为什么第一次要 warmup（预热）

第一次 `generate` 会触发：CUDA kernel 的 JIT 编译 / 加载、cuDNN 算法搜索、显存分配器初始化、（开了 CUDA Graph 时）图捕获。这些是一次性开销，混进统计里会让首批数字虚高。所以原文件里那段——

```python
print("Warming up...")
run_to_completion(profile_dir=None)   # 预热，结果丢弃
latencies = []
for _ in tqdm(range(args.num_iters)):
    latencies.append(run_to_completion(profile_dir=None))  # 才开始记
```

——先空跑一次再正式计时，是所有性能测试的通用纪律。

---

## 2. benchmark_serving 全流程拆解

`benchmark_serving.py` 是一个**异步压测客户端**：它不直接调用模型，而是像真实用户一样，按设定的速率向已经启动的 vLLM OpenAI 兼容服务（`vllm serve ...`）发 HTTP 请求，记录每个 token 的到达时刻。

```
 ┌──────────────────────────┐         HTTP /v1/completions          ┌─────────────────┐
 │  benchmark_serving.py     │  ── req1 ──────────────────────────▶ │                 │
 │  (异步客户端)             │  ── req2 ───────────────────────▶    │   vLLM Server   │
 │                          │  ── req3 ──────────────▶              │  (连续批处理)    │
 │  - 读 ShareGPT 取 prompt  │                                       │  ┌───────────┐  │
 │  - 按 request-rate 发请求 │  ◀── stream: tok,tok,tok,...EOS ──    │  │ scheduler │  │
 │  - 用 stream 记每 tok 时间│                                       │  │  + KV mgr │  │
 │  - 汇总 TTFT/TPOT/吞吐    │  ◀── stream: tok,tok,...              │  └───────────┘  │
 └──────────────────────────┘                                       └─────────────────┘
        客户端侧测量                                                     服务端侧调度
```

关键点：**客户端和服务端分离**。客户端负责"按节奏发请求 + 计时"，服务端负责"把同时在跑的请求拼成 batch 连续处理"（continuous batching）。压测测的是这套组合的端到端表现。

为什么必须用 streaming（流式返回）？因为要测 **TTFT**（首 token 时间）就必须知道第一个 token 什么时候到，而不是等整段回完。只有流式才能逐 token 打时间戳。

---

## 3. 请求率 vs 并发：两种压力旋钮

这是最容易混的地方。vLLM serving 压测有两个截然不同的"加压"方式：

### 3.1 request-rate（请求到达率，开环 / open-loop）

设 `--request-rate = R`（单位 req/s），客户端**按时间表发请求，不管前面的有没有回**。请求到达间隔通常服从**泊松过程**（指数分布的间隔），模拟真实世界"用户随机来"的情形。`--request-rate inf` 表示瞬间把全部请求一次性发出（最极端压力）。

```
 开环（按到达率）：发请求的节奏由 R 决定，与服务快慢无关
   t →  req  req    req  req  req      req   ...   （间隔随机，均值 1/R）
        ↓    ↓      ↓    ↓    ↓        ↓
   如果服务跟不上，请求在服务端排队 → 队列变长 → 延迟暴涨
```

**本质**：开环施加的是"外部需求"。当 R 超过系统极限吞吐，队列无限堆积，延迟发散——这正是我们想找的"崩溃点"。

### 3.2 max-concurrency（最大并发，闭环 / closed-loop）

设 `--max-concurrency = C`，客户端**始终维持最多 C 个在途请求**：一个回完了，立刻补发下一个。

```
 闭环（固定并发）：永远有 C 个请求在飞，回一个补一个
   [req][req][req] ← C=3，始终满
     回 ↓  补 ↑
   服务多快，发多快 → 不会无限排队，测的是"稳态吞吐 @ 给定并发"
```

**本质**：闭环施加的是"固定在途量"，系统不会被压垮，适合扫"并发=1,2,4,8,…"画稳定的吞吐-延迟曲线。

### 3.3 何时用哪个

| 旋钮 | 模型 | 回答的问题 | 风险 |
|---|---|---|---|
| `--request-rate R` | 开环 | "QPS 达到多少时延迟超标 / 系统崩？" | 超极限时队列爆，延迟数据失真 |
| `--max-concurrency C` | 闭环 | "并发 C 时，稳定的吞吐和延迟各是多少？" | 测不到"过载"行为 |

实践上常**两者结合**：用 `--request-rate` 找极限 QPS，用 `--max-concurrency` 扫出干净的曲线。

---

## 4. TTFT / TPOT / ITL / E2E：每个指标从哪来

这些是延迟侧的四个核心量。先用一张时间轴把它们钉死：

```
 客户端视角的一次请求：
   发出           首 tok 到        第2 tok      ...      最后 tok
   t0 ───────────▶ t1 ──────────▶ t2 ─────── ... ──────▶ tN
   │◀──  TTFT  ──▶│◀─ ITL ─▶│                          │
   │                                                     │
   │◀──────────────  E2E (端到端总时延)  ───────────────▶│

   TTFT  = t1 - t0          首 token 时间 = 排队 + prefill
   ITL   = t(i) - t(i-1)    相邻 token 间隔（decode 速度）
   TPOT  = (tN - t1) / (N-1) 平均每个输出 token 时间 = ITL 的均值
   E2E   = tN - t0          整条请求的总时间
```

逐个讲"为什么"：

- **TTFT（Time To First Token）**：用户按下回车到看见第一个字的时间。它 = **排队等待时间 + prefill 计算时间**。prompt 越长、并发越高、调度越拥挤，TTFT 越大。对话/搜索类业务最敏感的就是它。
- **TPOT（Time Per Output Token）**：吐字的快慢，决定"打字机效果"流不流畅。它由 decode 步速决定，受显存带宽和当前 batch 大小影响。$\text{TPOT} \approx 1/\text{每秒每请求生成 token 数}$。
- **ITL（Inter-Token Latency）**：单个 token 间隔，TPOT 是它的统计均值。看 ITL 的尾部（P99）能发现"卡顿"。
- **E2E**：总时延，$\text{E2E} \approx \text{TTFT} + (N_{out}-1)\times \text{TPOT}$。

> 永远看**分布而非均值**：报告 P50 / P90 / P99。LLM 服务的延迟长尾很重，均值会骗人——一个被插队的请求可能 P99 TTFT 是 P50 的 5 倍。

---

## 5. 吞吐：requests/s 与 tokens/s

吞吐有两种口径，对应原文件里 `benchmark_throughput.py` 的输出：

```python
total_num_tokens = sum(prompt_len + output_len for _, prompt_len, output_len in requests)
print(f"{len(requests)/elapsed_time:.2f} requests/s, "
      f"{total_num_tokens/elapsed_time:.2f} tokens/s")
```

- **requests/s（QPS）**：每秒完成多少条请求。受请求长度影响大（长请求拉低 QPS）。
- **tokens/s**：每秒处理多少 token。又分两种，务必说清是哪种：
  - **total tokens/s** =（输入+输出）token / 时间 → 衡量系统总处理量。
  - **output tokens/s** = 仅输出 token / 时间 → 衡量"生成"能力，更贴近用户感知的产出。

```
 吞吐口径区分：
   ┌─ requests/s ──── 条数视角（受长度干扰）
   tokens/s ─┬─ total ──── 输入+输出，看系统总带宽利用
             └─ output ─── 只数生成，看真正的"产能"
```

### goodput（达标吞吐）

新版 vLLM 引入 **goodput**：只统计"同时满足 TTFT/TPOT/E2E SLO 阈值"的请求所贡献的吞吐。普通吞吐可能很高，但若一半请求延迟超标，对业务无意义。goodput 把"快且达标"的那部分单独拎出来，是更诚实的指标。

---

## 6. ShareGPT 数据：为什么用它

压测最怕"假数据"——所有请求都是定长 128 输入 / 128 输出，那 batch 永远对齐，结果偏乐观。真实流量里 prompt 和回答的长度**千差万别**，长短请求混在一个 batch 里会产生**填充浪费**和**调度复杂度**。

**ShareGPT** 是从真实 ChatGPT 对话导出的数据集，长度分布贴近真实使用：

```
 prompt 长度分布（示意，真实多为长尾）
   条数
    │ ██
    │ ████
    │ ██████
    │ ████████▁▁
    │ ██████████▁▁▁▁▁▁__________  长尾（少量超长）
    └────────────────────────────▶ token 长度
      短            中           长
```

要点：
- benchmark_serving 用 `--dataset-name sharegpt --dataset-path <ShareGPT json>` 加载，从中采样真实 prompt 和"参考回答长度"。
- 也可用 `random` 合成数据集精确控制 in/out 长度，做**受控实验**（比如固定 1024 in / 256 out，专门压 prefill）。
- **同一组对比实验必须用同一份数据 + 同一随机种子**，否则长度分布漂移会让结论不可比。

| 数据集 | 用途 | 特点 |
|---|---|---|
| ShareGPT | 模拟真实在线流量 | 长度长尾，结论贴近生产 |
| random（合成） | 受控压测某一环节 | 可固定 in/out 长度，可复现 |
| sonnet / 自定义 | 特定业务回放 | 用自己的线上日志最准 |

---

## 7. 如何画吞吐-延迟曲线

这是压测的**最终交付物**。方法：**固定其它变量，扫某一个压力旋钮，每个点跑一次，把（吞吐, 延迟）描出来**。

### 7.1 步骤

```
for QPS in 1 2 4 8 16 32 ...:          # 或 for concurrency in 1 2 4 8 ...
    跑 benchmark_serving --request-rate QPS --num-prompts N
    记录：实际吞吐 (output tokens/s)  +  P99 TTFT / P99 TPOT
画图：x = 吞吐，y = 延迟
```

### 7.2 曲线长什么样（ASCII）

```
 P99 延迟
   │                                   ╱  ← "膝盖/拐点"后延迟陡升
   │                                 ╱      （系统接近极限，开始排队）
   │                              ╱
   │                          ╱
   │                    ╱╱╱        ← 这一段平缓：还有余量
   │        ╱╱╱╱╱╱╱
   │ ─────────                      ← 低负载：延迟几乎不变
   └──────────────────────────────────▶ 吞吐 (output tokens/s)
   低负载        甜点区(选这里)     过载区
```

- **平缓段**：负载低，加请求几乎不增延迟——这里浪费了 GPU。
- **拐点（膝盖 knee）**：再加负载延迟开始陡升。**最佳工作点就在拐点稍偏左**：既榨干了吞吐，延迟又没失控。
- **垂直段**：吞吐已到天花板，再压只让延迟爆炸、队列堆积，毫无收益。

### 7.3 怎么读这张图做决策

给定业务 SLO（如"P99 TTFT < 500ms"），在 y 轴画一条横线，它与曲线的交点对应的 x 值，就是**在满足 SLO 前提下能跑到的最大吞吐**。两套配置（如不同 GPU、不同并行策略）画在同一张图上，谁的曲线"更靠右下"（同延迟下吞吐更高）谁更优。

---

## 8. 调参方法论（讲思路，不背 CLI 默认值）

调参不是背参数表，而是**理解每个旋钮在权衡什么**。下面按"它解决什么、往哪拧、代价是什么"来讲（具体默认值以官方文档为准）。

### 8.1 批处理相关

```
 调大 batch ──▶ GPU 利用率↑、吞吐↑ ──▶ 但单请求要等同伴 ──▶ 延迟↑
 调小 batch ──▶ 延迟↓ ──▶ 但 GPU 喂不饱 ──▶ 吞吐↓
```

- **max-num-seqs（最大并发序列数）**：一个 batch 里最多塞多少条请求。拧大吞吐升、延迟升；受 KV 显存上限约束。**这是吞吐-延迟权衡的主旋钮**。
- **max-num-batched-tokens（每批最大 token 数）**：限制一个 step 处理的 token 总量。配合 **chunked prefill**（把长 prompt 的 prefill 切块，与 decode 交错）使用，能避免一个超长 prompt 的 prefill 把所有 decode 请求"卡住"，从而压住 TTFT 长尾。

### 8.2 KV Cache / 显存

- **gpu-memory-utilization**：允许 vLLM 占用多少比例显存给 KV Cache。拧大 → KV 空间大 → 能并发更多/更长序列 → 吞吐升；但留给激活/碎片的余量变小，**太满会 OOM**。
- **PagedAttention**：vLLM 的核心机制——把 KV Cache 像操作系统分页一样切成固定大小的 block 管理，几乎消除显存碎片，让有效并发数显著提升。压测时它的收益体现为"同显存下能跑更高并发"。

```
 显存预算示意（gpu-mem-util 决定 KV 这块多大）
 ┌───────────┬──────────────────────────────┬──────┐
 │  模型权重  │        KV Cache（可并发数）     │ 余量 │
 └───────────┴──────────────────────────────┴──────┘
   固定        ←─ 拧 util 把这块撑大 ─→          别撑没了否则OOM
```

### 8.3 并行策略（多卡）

- **tensor-parallel（张量并行 TP）**：把单层权重切到多卡，降低单请求延迟、突破单卡显存，但卡间通信（all-reduce）有开销，跨节点时通信会拖后腿。
- **pipeline-parallel（流水线并行 PP）**：按层切分到多卡，通信量小，适合跨节点扩展，但有流水线气泡。
- 经验：**单机内优先 TP**（NVLink 带宽高），**跨机才上 PP**。压测时分别画曲线对比。

### 8.4 量化与精度

FP16 → INT8/FP8/AWQ 等量化能减小权重和 KV 占用，腾出空间提吞吐、降访存延迟，但可能损精度。压测要**同时报吞吐和质量指标**，不能只看快。

### 8.5 调参的科学方法

```
 1. 定基线：固定数据集+种子，跑一次，记全部指标
 2. 单变量：每次只改一个旋钮（控制变量法）
 3. 重画曲线：看拐点是右移（变好）还是延迟上抬（变差）
 4. 对照 SLO：达标前提下吞吐最大者胜出
 5. 复现：换台机/重跑确认不是噪声
```

> 反模式：一次改三个参数，结果好了不知道是谁的功劳，坏了不知道怪谁。**永远控制变量。**

---

## 9. 数值例子 / 对照 / 实践

### 9.1 从 TPOT 反推单请求生成速度（手算）

设某配置下测得平均 **TPOT = 25 ms/token**。那么单条请求的生成速度：

$$\text{单请求 tokens/s} = \frac{1}{\text{TPOT}} = \frac{1}{0.025\,\text{s}} = 40\ \text{tokens/s}$$

人眼舒适阅读约 10–15 tokens/s，所以 40 tokens/s 体感"刷刷地出"。

### 9.2 系统总吞吐估算（手算）

设服务端稳定维持 **并发 = 64** 条请求同时 decode，每条 TPOT = 25 ms：

$$\text{系统 output tokens/s} \approx \frac{\text{并发数}}{\text{TPOT}} = \frac{64}{0.025} = 2560\ \text{tokens/s}$$

注意：这是**理想上界**。真实值更低，因为有 prefill 抢占、调度开销、长短请求不齐导致的气泡。压测脚本测出的"实际 2100 tokens/s"与这个上界的差距，就是优化空间。

### 9.3 一次完整请求的 E2E 估算（手算）

prompt 长 512 token，TTFT 实测 180 ms（含排队），输出 200 token，TPOT 25 ms：

$$\text{E2E} \approx \text{TTFT} + (N_{out}-1)\times\text{TPOT} = 0.18 + 199\times0.025 \approx 0.18 + 4.98 = 5.16\ \text{s}$$

可见对长输出请求，**E2E 几乎完全由 TPOT × 输出长度主导**，TTFT 只占零头——所以长生成场景优化 decode（TPOT）比优化 prefill 收益大。

### 9.4 三个脚本对照

| 脚本 | 模式 | 加压方式 | 主要产出 | 像什么场景 |
|---|---|---|---|---|
| benchmark_latency | 离线、单批 | 固定 batch | 纯计算延迟均值 | 摸单批理论延迟 |
| benchmark_throughput | 离线、灌满 | 一次性全发 | 极限 requests/s、tokens/s | 批量离线推理上限 |
| benchmark_serving | 在线、流式 | request-rate / concurrency | TTFT/TPOT/E2E 分布 + 吞吐 + goodput | 真实在线服务 ★ |

### 9.5 实践清单

1. **先 warmup，再正式跑**；丢弃首批数据。
2. **报分布（P50/P90/P99），别只报均值**；延迟长尾才是用户痛点。
3. **固定数据集 + 随机种子**，否则结论不可比。
4. **服务端与客户端放不同进程/机器**，避免客户端 CPU 成瓶颈反过来污染数据。
5. **扫一组 QPS/并发画曲线**，单点数字没意义。
6. **对照业务 SLO 选工作点**，关注拐点和 goodput。
7. 改配置时**控制变量**，一次一个旋钮。

---

## 常见问题

| 问题 | 解答 |
|---|---|
| TTFT 突然飙高是什么原因？ | 多半是请求在服务端排队（负载越过拐点），或被超长 prompt 的 prefill 卡住；试 chunked prefill、调小并发或加卡。 |
| 吞吐上不去，GPU 利用率却不满？ | batch 没喂饱：调大 `max-num-seqs`/`max-num-batched-tokens`，或检查 KV 显存是否成为并发上限（拧 `gpu-memory-utilization`）。 |
| request-rate 和 max-concurrency 该用哪个？ | 找"系统崩溃点"用 request-rate（开环）；画干净稳定的吞吐-延迟曲线用 max-concurrency（闭环）。 |
| tokens/s 数字两次跑差很多？ | 大概率没固定数据/种子，长度分布变了；或没 warmup；或 total 与 output 口径混了。 |
| 为什么一定要用 ShareGPT 而不是定长数据？ | 定长会让 batch 完美对齐、结果偏乐观；真实流量长度长尾，ShareGPT 才能暴露调度和填充浪费。 |
| 延迟达标但 goodput 很低？ | 普通吞吐里混了大量超 SLO 的请求；goodput 只数达标的，更诚实，应以它为准。 |
| 单卡放不下模型怎么压？ | 上 tensor-parallel（单机优先）或 pipeline-parallel（跨机），分别压测画曲线对比通信开销。 |
| 均值很好但用户喊卡？ | 看 P99 ITL/TTFT——长尾被均值掩盖了，单个被插队的请求就能毁掉体验。 |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-eval/llm-performance/推理性能测试]]
- [[llm-inference/vllm/README]]
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

> 参考脚本（版本以官方仓库为准）：
> - benchmark_latency.py：https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_latency.py
> - benchmark_throughput.py：https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py
> - benchmark_serving.py：https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
